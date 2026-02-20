"""PyAV-based video decoder with TorchCodec-compatible playback semantics."""

import gc
import warnings
from fractions import Fraction
from typing import List, Optional

import av
import cv2
import numpy as np
import numpy.typing as npt

from .. import cached_av
from .._typing import PathLike
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .types import VideoStreamMetadata

# Garbage collection interval for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

# Threshold for sparse query detection (seconds between consecutive timestamps)
_SPARSE_QUERY_GAP_THRESHOLD = 1.0


def _frame_to_rgba(frame: av.VideoFrame) -> npt.NDArray[np.uint8]:
    """Convert PyAV frame to RGBA numpy array.

    NOTE: Convert ARGB to RGBA manually instead of using `to_ndarray(format="rgba")`.
    Direct RGBA conversion causes memory corruption on certain videos:
      Error: "malloc_consolidate(): invalid chunk size; Fatal Python error: Aborted"
      Example: https://huggingface.co/datasets/open-world-agents/example_dataset/resolve/main/example.mkv (pts_ns=1_000_000_000)
    Possibly related to:
      - PyAV issue: https://github.com/PyAV-Org/PyAV/issues/1269
      - FFmpeg ticket: https://trac.ffmpeg.org/ticket/9254

    Args:
        frame: PyAV VideoFrame to convert

    Returns:
        RGBA numpy array (H, W, 4) with uint8 dtype
    """
    argb_array = frame.to_ndarray(format="argb")
    # ARGB format stores channels as [A, R, G, B], so we reorder to [R, G, B, A]
    rgba_array: npt.NDArray[np.uint8] = argb_array[:, :, [1, 2, 3, 0]]
    return rgba_array


def _convert_av_frames_to_nchw(av_frames: List[av.VideoFrame]) -> List[npt.NDArray[np.uint8]]:
    """Convert a list of PyAV frames to NCHW numpy arrays (RGB)."""
    frames = []
    for frame in av_frames:
        rgba_array = _frame_to_rgba(frame)
        rgb_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2RGB)
        frame_nchw = np.transpose(rgb_array, (2, 0, 1)).astype(np.uint8)
        frames.append(frame_nchw)
    return frames


class PyAVVideoDecoder(BaseVideoDecoder):
    """Video decoder using PyAV with TorchCodec-compatible playback semantics.

    Implements the playback model where a frame is displayed from its pts until
    the next frame's pts. For timestamp t, returns frame[i] where:
        frame[i].pts <= t < frame[i+1].pts

    Args:
        source: Path to video file or URL

    Examples:
        >>> with PyAVVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
        ...     print(batch.data.shape)  # (3, 3, H, W)
    """

    def __init__(self, source: PathLike, **kwargs):
        """Initialize PyAV video decoder."""
        super().__init__(source, **kwargs)
        self._container = cached_av.open(source, "r", keep_av_open=True)
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container.

        Decodes only the first frame to get accurate begin_stream_seconds.
        Uses header metadata for duration/end_stream_seconds.
        """
        container = self._container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self.source}")
        stream = container.streams.video[0]

        # Determine frame rate
        if stream.average_rate:
            average_rate = Fraction(stream.average_rate)
        else:
            raise ValueError("Failed to determine average rate")

        # Determine video duration from header metadata
        if stream.duration and stream.time_base:
            duration_seconds = Fraction(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds = Fraction(container.duration, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # Decode first frame to get accurate begin_stream_seconds
        # This is fast - just one seek + one decode
        container.seek(0)
        first_pts = Fraction(0)
        for frame in container.decode(video=0):
            if frame.time is not None:
                first_pts = Fraction(frame.time).limit_denominator(1000000)
            break

        # Reset container position
        container.seek(0)

        # begin_stream_seconds is the first frame's PTS
        begin_stream_seconds = first_pts

        # end_stream_seconds = begin + duration (from header)
        # Note: This assumes header duration is relative to stream start
        end_stream_seconds = begin_stream_seconds + duration_seconds

        # Determine frame count
        num_frames = stream.frames if stream.frames else int(duration_seconds * average_rate)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_rate=average_rate,
            width=stream.width,
            height=stream.height,
            begin_stream_seconds=begin_stream_seconds,
            end_stream_seconds=end_stream_seconds,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._metadata

    def _create_empty_batch(self) -> FrameBatch:
        """Create an empty FrameBatch with correct spatial dimensions."""
        return FrameBatch(
            data=np.empty((0, 3, self._metadata.height, self._metadata.width), dtype=np.uint8),
            pts_seconds=np.array([], dtype=np.float64),
            duration_seconds=np.array([], dtype=np.float64),
        )

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames that would be displayed at specific timestamps.

        Follows TorchCodec's playback semantics: returns the frame where
        frame[i].pts <= timestamp < frame[i+1].pts.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch with frame data in NCHW format

        Raises:
            ValueError: If any timestamp is outside [begin_stream_seconds, end_stream_seconds)
        """
        if not seconds:
            return self._create_empty_batch()

        # Validate timestamps per playback_semantics.md boundary conditions
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)  # type: ignore[arg-type]
        for t in seconds:
            if t < begin_stream:
                raise ValueError(f"Timestamp {t}s < begin_stream_seconds ({begin_stream}s)")
            if t >= end_stream:
                raise ValueError(f"Timestamp {t}s >= end_stream_seconds ({end_stream}s)")

        # Get frames using playback semantics
        av_frames = self._get_frames_played_at(seconds)

        # Convert to RGB numpy arrays in NCHW format
        frames = _convert_av_frames_to_nchw(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
        )

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        """Return multiple frames in the given range [start_seconds, stop_seconds).

        Args:
            start_seconds: Time, in seconds, of the start of the range.
            stop_seconds: Time, in seconds, of the end of the range (excluded).
            fps: If specified, resample output to this frame rate by
                duplicating or dropping frames as necessary. If None,
                returns frames at the source video's frame rate.

        Returns:
            FrameBatch with frame data in NCHW format.

        Raises:
            ValueError: If the range parameters are invalid.
        """
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)

        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not begin_stream <= start_seconds < end_stream:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {begin_stream} "
                f"and less than {end_stream}."
            )
        if not stop_seconds <= end_stream:
            raise ValueError(f"Invalid stop seconds: {stop_seconds}. It must be less than or equal to {end_stream}.")

        if fps is not None:
            # Resample: generate timestamps at the given fps and get frames
            timestamps = np.arange(start_seconds, stop_seconds, 1.0 / fps).tolist()
            if not timestamps:
                return self._create_empty_batch()
            return self.get_frames_played_at(timestamps)

        # Native frame rate: decode all frames with pts in [start_seconds, stop_seconds)
        self._seek_to_or_before(start_seconds)

        av_frames: List[av.VideoFrame] = []
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            frame_pts = float(frame.time)
            if frame_pts >= stop_seconds:
                break
            if frame_pts >= start_seconds:
                av_frames.append(frame)

        if not av_frames:
            return self._create_empty_batch()

        frames = _convert_av_frames_to_nchw(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(av_frames), duration, dtype=np.float64),
        )

    def _get_frames_played_at(self, seconds: List[float]) -> List[av.VideoFrame]:
        """Get frames using TorchCodec playback semantics.

        For each timestamp, returns the frame where:
            frame[i].pts <= timestamp < frame[i+1].pts

        This is the "frame being played at" semantic - the frame that would
        be displayed on screen at the given timestamp.
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Sort queries for efficient sequential access while preserving output order
        indexed_queries = sorted(enumerate(seconds), key=lambda x: x[1])
        results: List[av.VideoFrame] = [None] * len(seconds)  # type: ignore

        # Warn if consecutive sorted timestamps are far apart (single-pass is wasteful)
        if len(indexed_queries) >= 2:
            sorted_times = [t for _, t in indexed_queries]
            max_gap = max(b - a for a, b in zip(sorted_times, sorted_times[1:]))
            if max_gap > _SPARSE_QUERY_GAP_THRESHOLD:
                warnings.warn(
                    f"Sparse timestamp query detected: max gap between consecutive "
                    f"timestamps is {max_gap:.1f}s (threshold: {_SPARSE_QUERY_GAP_THRESHOLD}s). "
                    f"Single-pass decoding will decode all intermediate frames. "
                    f"For widely spaced timestamps, consider per-timestamp seeking instead.",
                    UserWarning,
                    stacklevel=3,
                )

        query_idx = 0
        prev_frame: av.VideoFrame | None = None
        prev_frame_pts: float = float("-inf")

        # Smart seek: try to seek to first query, fall back to start if we overshoot
        first_query_time = indexed_queries[0][1]
        self._seek_to_or_before(first_query_time)

        # Decode frames and match to queries using nextPts logic
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")

            frame_pts = float(frame.time)

            # Process all queries where: prev_frame.pts <= query < frame.pts
            while query_idx < len(indexed_queries):
                orig_idx, query_time = indexed_queries[query_idx]
                if query_time < frame_pts:
                    # This query's timestamp is before current frame's pts
                    # Verify prev_frame.pts <= query_time (should always be true if validated)
                    if prev_frame is not None and prev_frame_pts <= query_time:
                        results[orig_idx] = prev_frame
                        query_idx += 1
                    elif prev_frame is None:
                        # Query is before first frame - this should have been caught by validation
                        raise ValueError(f"Timestamp {query_time}s is before first frame (pts={frame_pts}s)")
                    else:
                        # prev_frame_pts > query_time - should not happen with correct validation
                        raise ValueError(
                            f"Internal error: query {query_time}s not in range [{prev_frame_pts}, {frame_pts})"
                        )
                else:
                    break

            prev_frame = frame
            prev_frame_pts = frame_pts

            # Check if we've processed all queries
            if query_idx >= len(indexed_queries):
                break

        # Handle remaining queries (timestamps at or after last decoded frame)
        # These should satisfy: prev_frame.pts <= query < end_stream_seconds
        while query_idx < len(indexed_queries):
            orig_idx, query_time = indexed_queries[query_idx]
            if prev_frame is not None and prev_frame_pts <= query_time:
                results[orig_idx] = prev_frame
                query_idx += 1
            else:
                raise ValueError(f"Could not find frame for timestamp {query_time}s")

        # Verify all queries were satisfied
        for i, result in enumerate(results):
            if result is None:
                raise ValueError(f"Could not find frame for timestamp {seconds[i]}s")

        return results

    def _seek_to_or_before(self, target_seconds: float) -> None:
        """Seek to target time or before it, using exponential backoff if needed.

        PyAV seeks to keyframes, which may land past the target if keyframes are sparse.
        This method detects overshooting and backs off exponentially until we find a
        valid seek point before the target.
        """
        stream = self._container.streams.video[0]
        time_base = float(stream.time_base)
        begin_stream = float(self._metadata.begin_stream_seconds)

        # Start with seeking to the target
        seek_target = target_seconds
        buffer = 1.0  # Initial backoff buffer in seconds

        while True:
            seek_pts = int(seek_target / time_base)
            self._container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

            # Check if we overshot by peeking at the first frame
            try:
                frame = next(self._container.decode(video=0))
            except StopIteration:
                # No frames at all - nothing we can do
                return

            if frame.time is not None and frame.time <= target_seconds:
                # Good! We landed at or before the target
                # Re-seek to restore position (we consumed one frame)
                self._container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
                return

            # We overshot! Back off and try again
            seek_target = target_seconds - buffer
            buffer *= 2  # Exponential backoff

            # If we've backed off past the stream start, seek to start
            if seek_target <= begin_stream:
                self._container.seek(0)
                return

    def close(self):
        """Release video decoder resources."""
        if hasattr(self, "_container"):
            self._container.close()
