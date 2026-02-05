"""PyAV-based video decoder with TorchCodec-compatible playback semantics."""

import gc
from fractions import Fraction
from typing import List

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
        """Extract video stream metadata from container."""
        container = self._container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self.source}")
        stream = container.streams.video[0]

        # Determine video duration
        if stream.duration and stream.time_base:
            duration_seconds = Fraction(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds = Fraction(container.duration, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # Determine frame rate
        if stream.average_rate:
            average_rate = Fraction(stream.average_rate)
        else:
            raise ValueError("Failed to determine average rate")

        # Determine frame count
        num_frames = stream.frames if stream.frames else int(duration_seconds * average_rate)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_rate=average_rate,
            width=stream.width,
            height=stream.height,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._metadata

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames that would be displayed at specific timestamps.

        Follows TorchCodec's playback semantics: returns the frame where
        frame[i].pts <= timestamp < frame[i+1].pts.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch with frame data in NCHW format

        Raises:
            ValueError: If any timestamp is negative or >= video duration
        """
        if not seconds:
            return FrameBatch(
                data=np.empty((0, 3, self._metadata.height, self._metadata.width), dtype=np.uint8),
                pts_seconds=np.array([], dtype=np.float64),
                duration_seconds=np.array([], dtype=np.float64),
            )

        # Validate timestamps per torchcodec_design.md boundary conditions
        end_stream = float(self._metadata.duration_seconds)
        for t in seconds:
            if t < 0:
                raise ValueError(f"Timestamp {t}s is negative")
            if t >= end_stream:
                raise ValueError(f"Timestamp {t}s >= end_stream_seconds ({end_stream}s)")

        # Get frames using playback semantics
        av_frames = self._get_frames_played_at(seconds)

        # Convert to RGB numpy arrays in NCHW format
        frames = []
        for frame in av_frames:
            rgba_array = _frame_to_rgba(frame)
            rgb_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2RGB)
            frame_nchw = np.transpose(rgb_array, (2, 0, 1)).astype(np.uint8)
            frames.append(frame_nchw)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
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

        query_idx = 0
        prev_frame: av.VideoFrame | None = None

        # Seek to before the first query
        first_query_time = indexed_queries[0][1]
        seek_ts = int(av.time_base * first_query_time)
        self._container.seek(seek_ts, any_frame=False)

        # Decode frames and match to queries using nextPts logic
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")

            frame_pts = float(frame.time)

            # Process all queries that fall before this frame's pts
            # These queries are "played" by the previous frame
            while query_idx < len(indexed_queries):
                orig_idx, query_time = indexed_queries[query_idx]
                if query_time < frame_pts:
                    # This query's timestamp is before current frame's pts
                    # So the previous frame is what would be displayed
                    if prev_frame is not None:
                        results[orig_idx] = prev_frame
                        query_idx += 1
                    else:
                        # No previous frame - query is before first frame
                        # Assign current frame as it's the first available
                        results[orig_idx] = frame
                        query_idx += 1
                else:
                    break

            prev_frame = frame

            # Check if we've processed all queries
            if query_idx >= len(indexed_queries):
                break

        # Handle remaining queries (timestamps at or after last decoded frame)
        while query_idx < len(indexed_queries):
            orig_idx, _ = indexed_queries[query_idx]
            if prev_frame is not None:
                results[orig_idx] = prev_frame
            query_idx += 1

        # Verify all queries were satisfied
        for i, result in enumerate(results):
            if result is None:
                raise ValueError(f"Could not find frame for timestamp {seconds[i]}s")

        return results

    def close(self):
        """Release video decoder resources."""
        if hasattr(self, "_container"):
            self._container.close()
