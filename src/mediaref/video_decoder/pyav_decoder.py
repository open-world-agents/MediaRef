"""PyAV-based video decoder with TorchCodec-compatible interface."""

import gc
from fractions import Fraction
from typing import Generator, List, Optional, Union

import av
import cv2
import numpy as np
import numpy.typing as npt

from .. import cached_av
from .._internal import _frame_to_rgba
from .._typing import PathLike
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .types import SECOND_TYPE, BatchDecodingStrategy, VideoStreamMetadata

# Garbage collection counters for PyAV reference cycles
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


class PyAVVideoDecoder(BaseVideoDecoder):
    """TorchCodec-compatible video decoder built on PyAV.

    This decoder uses PyAV (Python bindings for FFmpeg) to decode video frames
    with support for batch decoding strategies and efficient keyframe-aware reading.

    Features:
        - Efficient batch decoding with multiple strategies
        - Keyframe-aware reading for optimal performance
        - Support for both frame indices and timestamps
        - Automatic container caching for repeated access
        - Context manager support for resource cleanup

    Args:
        source: Path to video file or URL (HTTP/HTTPS supported)
        **kwargs: Additional arguments (reserved for future use)

    Examples:
        >>> # Basic usage
        >>> decoder = PyAVVideoDecoder("video.mp4")
        >>> frame = decoder[0]  # Get first frame
        >>> decoder.close()
        >>>
        >>> # Batch decoding
        >>> with PyAVVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_at([0, 10, 20, 30])
        ...     print(batch.data.shape)  # (4, 3, H, W)
        >>>
        >>> # Timestamp-based access
        >>> decoder = PyAVVideoDecoder("video.mp4")
        >>> batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
        >>> decoder.close()
    """

    def __init__(self, source: PathLike, **kwargs):
        """Initialize PyAV video decoder.

        Args:
            source: Path to video file or URL
            **kwargs: Additional arguments (reserved for future use)
        """
        super().__init__(source, **kwargs)
        self._container = cached_av.open(source, "r", keep_av_open=True)
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container.

        Returns:
            VideoStreamMetadata with frame count, duration, fps, and dimensions

        Raises:
            ValueError: If no video stream found or metadata cannot be determined
        """
        container = self._container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self.source}")
        stream = container.streams.video[0]

        # Determine video duration
        if stream.duration and stream.time_base:
            duration_seconds = stream.duration * stream.time_base
        elif container.duration:
            duration_seconds = container.duration * Fraction(1, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # Determine frame rate
        if stream.average_rate:
            average_rate = stream.average_rate
        else:
            raise ValueError("Failed to determine average rate")

        # Determine frame count
        if stream.frames:
            num_frames = stream.frames
        else:
            num_frames = int(duration_seconds * average_rate)

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

    def __getitem__(self, key: Union[int, slice]) -> npt.NDArray[np.uint8]:
        """Enable array-like indexing for frame access."""
        if isinstance(key, int):
            return self.get_frames_at([key]).data[0]
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.metadata.num_frames)
            return self.get_frames_at(list(range(start, stop, step))).data
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def get_frames_at(
        self,
        indices: List[int],
        *,
        strategy: Optional[BatchDecodingStrategy] = None,
    ) -> FrameBatch:
        """Retrieve frames at specific frame indices."""
        # Default to SEQUENTIAL_PER_KEYFRAME_BLOCK if not specified
        if strategy is None:
            strategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK

        indices = [index % self.metadata.num_frames for index in indices]
        pts = [float(idx / self.metadata.average_rate) for idx in indices]
        return self.get_frames_played_at(seconds=pts, strategy=strategy)

    def get_frames_played_at(
        self,
        seconds: List[float],
        *,
        strategy: Optional[BatchDecodingStrategy] = None,
    ) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Args:
            seconds: List of timestamps in seconds to retrieve frames at
            strategy: Decoding strategy (SEPARATE, SEQUENTIAL_PER_KEYFRAME_BLOCK, or SEQUENTIAL).
                Defaults to SEQUENTIAL_PER_KEYFRAME_BLOCK if not specified.

        Returns:
            FrameBatch containing frame data and timing information

        Raises:
            ValueError: If any timestamp exceeds video duration or frames cannot be found
        """
        # Default to SEQUENTIAL_PER_KEYFRAME_BLOCK if not specified
        if strategy is None:
            strategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK

        if not seconds:
            return FrameBatch(
                data=np.empty((0, 3, self.metadata.height, self.metadata.width), dtype=np.uint8),
                pts_seconds=np.array([], dtype=np.float64),
                duration_seconds=np.array([], dtype=np.float64),
            )

        if max(seconds) > self.metadata.duration_seconds:
            raise ValueError(
                f"Requested time {max(seconds)}s exceeds video duration {self.metadata.duration_seconds}s"
            )

        # Get AV frames using internal method
        av_frames = self._get_frames_at_timestamps(seconds, strategy)

        # Convert to RGB numpy arrays in NCHW format.
        frames = []
        for frame in av_frames:
            # Use shared helper function to convert frame to RGBA (with FFmpeg SSSE3 bug workaround)
            rgba_array = _frame_to_rgba(frame)
            # Convert RGBA to RGB using cv2
            rgb_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2RGB)
            # Transpose to NCHW format
            frame_nchw = np.transpose(rgb_array, (2, 0, 1)).astype(np.uint8)
            frames.append(frame_nchw)

        pts_list = [frame.time for frame in av_frames]

        duration = 1.0 / self.metadata.average_rate

        return FrameBatch(
            data=np.stack(frames, axis=0),  # [N, C, H, W]
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
        )

    def _get_frames_at_timestamps(
        self,
        seconds: List[float],
        strategy: BatchDecodingStrategy,
    ) -> list[av.VideoFrame]:
        """Internal method to get AV frames at specific timestamps.

        Args:
            seconds: List of timestamps in seconds
            strategy: Decoding strategy

        Returns:
            List of AV video frames in the same order as input timestamps
        """
        # Decode each frame separately (preserves input order, no sorting needed)
        if strategy == BatchDecodingStrategy.SEPARATE:
            return [self._read_frame_at(pts=s) for s in seconds]

        # For batch strategies, sort queries for efficient sequential decoding
        queries = sorted([(s, i) for i, s in enumerate(seconds)])
        frames: list[av.VideoFrame] = [None] * len(queries)  # type: ignore

        # Epsilon for floating-point comparison (handles pts_ns conversion precision loss)
        EPSILON = 1e-6

        # Read all frames in one go (floor semantics: for query Q, find last frame F where F.time <= Q)
        if strategy == BatchDecodingStrategy.SEQUENTIAL:
            start_pts = queries[0][0]
            found = 0
            prev_frame: Optional[av.VideoFrame] = None

            # Use include_preceding=True to get frames before start_pts for proper floor semantics
            for frame in self._read_frames(start_pts, include_preceding=True):
                # Playback semantics: assign prev_frame to queries where
                # prev_frame.time <= Q < frame.time (query falls within prev_frame's playback range)
                # Use EPSILON on lower bound to handle floating point, but strict < on upper bound
                if prev_frame is not None:
                    while found < len(queries) and prev_frame.time - EPSILON <= queries[found][0] < frame.time:
                        frames[queries[found][1]] = prev_frame
                        found += 1

                prev_frame = frame
                if found >= len(queries):
                    break

            # Assign remaining queries (timestamps > last frame seen) to the last frame
            if prev_frame is not None:
                while found < len(queries):
                    frames[queries[found][1]] = prev_frame
                    found += 1

        # Restart-on-keyframe logic (floor semantics)
        elif strategy == BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK:
            query_idx = 0

            # Outer loop: restart/resume for each segment
            while query_idx < len(queries):
                target_time = queries[query_idx][0]
                first_keyframe_seen = False
                query_idx_before_segment = query_idx
                prev_frame_in_segment: Optional[av.VideoFrame] = None
                next_keyframe_time: Optional[float] = None

                # Inner loop: read frames until keyframe detected or all targets found
                # Use include_preceding=True to get frames before target_time for floor semantics
                for frame in self._read_frames(start_pts=target_time, include_preceding=True):
                    frame_time = frame.time
                    if frame_time is None:
                        raise ValueError("Frame time is None")

                    # Check for second keyframe BEFORE processing
                    if frame.key_frame and first_keyframe_seen:
                        # Found second keyframe - this marks the END of current block
                        # All remaining queries with time < next_keyframe should use prev_frame
                        next_keyframe_time = frame_time
                        break

                    if frame.key_frame:
                        first_keyframe_seen = True

                    # Playback semantics: assign prev_frame to queries where
                    # prev_frame.time <= Q < frame.time (query falls within prev_frame's playback range)
                    # Use EPSILON on lower bound to handle floating point, but strict < on upper bound
                    if prev_frame_in_segment is not None:
                        while query_idx < len(queries) and prev_frame_in_segment.time - EPSILON <= queries[query_idx][0] < frame_time:
                            if frames[queries[query_idx][1]] is None:
                                frames[queries[query_idx][1]] = prev_frame_in_segment
                            query_idx += 1

                    prev_frame_in_segment = frame

                    # Stop condition for inner loop
                    if query_idx >= len(queries):
                        break

                # After inner loop: assign remaining queries up to next_keyframe_time
                if prev_frame_in_segment is not None:
                    if next_keyframe_time is not None:
                        # We broke on second keyframe - assign remaining queries < next_keyframe_time
                        while query_idx < len(queries) and queries[query_idx][0] < next_keyframe_time - EPSILON:
                            if frames[queries[query_idx][1]] is None:
                                frames[queries[query_idx][1]] = prev_frame_in_segment
                            query_idx += 1
                    else:
                        # Reached end of video - assign all remaining queries to last frame
                        while query_idx < len(queries) and frames[queries[query_idx][1]] is None:
                            frames[queries[query_idx][1]] = prev_frame_in_segment
                            query_idx += 1

                # If no progress made in inner loop, raise error
                if query_idx_before_segment == query_idx:
                    raise ValueError(
                        f"No matching frames found for query starting at {target_time:.3f}s. "
                        f"This may indicate a corrupted video file or a decoding issue."
                    )

        if any(f is None for f in frames):
            missing_seconds = [s for i, s in enumerate(seconds) if frames[i] is None]
            raise ValueError(f"Could not find frames for the following timestamps: {missing_seconds}")

        return frames

    def _read_frames(
        self,
        start_pts: SECOND_TYPE = 0.0,
        end_pts: Optional[SECOND_TYPE] = None,
        include_preceding: bool = False,
    ) -> Generator[av.VideoFrame, None, None]:
        """Yield frames between start_pts and end_pts in seconds.

        Args:
            start_pts: Start time in seconds
            end_pts: End time in seconds (None = read until end)
            include_preceding: If True, include frames before start_pts that are
                decoded after seeking (useful for floor semantics)

        Yields:
            Video frames in the specified time range
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Handle negative end_pts (Python-style indexing)
        if end_pts is not None and float(end_pts) < 0:
            if self._container.duration is None:
                raise ValueError("Video duration unavailable for negative end_pts")
            duration = self._container.duration / av.time_base
            end_pts = duration + float(end_pts)

        end_pts_float = float(end_pts) if end_pts is not None else float("inf")

        # Seek to start position
        timestamp_ts = int(av.time_base * float(start_pts))
        # NOTE: seek with anyframe=False must present before decoding to ensure flawless decoding
        self._container.seek(timestamp_ts, any_frame=False)

        # Yield frames in interval
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            if not include_preceding and frame.time < float(start_pts):
                continue
            if frame.time > end_pts_float:
                break
            yield frame

    def _read_frame_at(self, pts: SECOND_TYPE = 0.0) -> av.VideoFrame:
        """Read the frame visible at the given timestamp.

        Returns the frame that would be displayed at the given timestamp during
        video playback. Uses frame.duration to determine if the query falls within
        [frame.time, frame.time + duration) - the valid playback range.

        Args:
            pts: Timestamp in seconds

        Returns:
            Video frame visible at the specified timestamp

        Raises:
            ValueError: If frame not found
        """
        pts_float = float(pts)

        # Seek to a position before the target timestamp
        timestamp_ts = int(av.time_base * pts_float)
        self._container.seek(timestamp_ts, any_frame=False)

        stream = self._container.streams.video[0]

        # Track last valid frame for playback semantics
        last_valid_frame: Optional[av.VideoFrame] = None

        # Find the frame where pts falls within [frame.time, frame.time + duration)
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")

            # If frame has duration, check for exact playback interval match
            if frame.duration is not None:
                duration_s = float(frame.duration * stream.time_base)
                if frame.time <= pts_float < frame.time + duration_s:
                    return frame

            # If we've decoded past the query time, the previous frame is the correct one
            if frame.time > pts_float:
                if last_valid_frame:
                    return last_valid_frame
                break

            last_valid_frame = frame

        # If query is after the last frame, check if it's within the last frame's duration
        if last_valid_frame:
            if last_valid_frame.duration is not None:
                duration_s = float(last_valid_frame.duration * stream.time_base)
                if last_valid_frame.time <= pts_float < last_valid_frame.time + duration_s:
                    return last_valid_frame

        raise ValueError(f"Frame not found at {pts_float:.2f}s in {self.source}")

    def close(self):
        """Release video decoder resources.

        Safe to call multiple times. Closes the underlying PyAV container
        and releases any cached resources.
        """
        if hasattr(self, "_container"):
            self._container.close()
