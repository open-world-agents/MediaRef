"""PyAV-based video decoder with TorchCodec-compatible interface."""

import gc
from fractions import Fraction
from typing import Generator, List, Optional, Union

import av
import cv2
import numpy as np
import numpy.typing as npt

from .. import cached_av
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
            # Convert to RGBA first. BUG: need rgba since rgb24 yields incorrect value ONLY on OSX.
            rgba_array = frame.to_ndarray(format="rgba")
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

        # Read all frames in one go
        if strategy == BatchDecodingStrategy.SEQUENTIAL:
            start_pts = queries[0][0]
            found = 0

            for frame in self._read_frames(start_pts):  # do not specify end_pts to avoid early termination
                while found < len(queries) and frame.time >= queries[found][0]:
                    frames[queries[found][1]] = frame
                    found += 1
                if found >= len(queries):
                    break

        # Restart-on-keyframe logic
        elif strategy == BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK:
            query_idx = 0

            # Outer loop: restart/resume for each segment
            while query_idx < len(queries):
                target_time = queries[query_idx][0]
                first_keyframe_seen = False
                query_idx_before_segment = query_idx

                # Inner loop: read frames until keyframe detected or all targets found
                for frame in self._read_frames(start_pts=target_time):
                    # Track keyframes
                    if frame.key_frame:
                        if first_keyframe_seen:
                            # Hit second keyframe - stop segment
                            break
                        first_keyframe_seen = True

                    # Match frames to queries in this segment
                    while query_idx < len(queries) and frame.time >= queries[query_idx][0]:
                        frames[queries[query_idx][1]] = frame
                        query_idx += 1

                    # Stop condition for inner loop
                    if query_idx >= len(queries):
                        # Found all remaining frames
                        break

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
        self, start_pts: SECOND_TYPE = 0.0, end_pts: Optional[SECOND_TYPE] = None
    ) -> Generator[av.VideoFrame, None, None]:
        """Yield frames between start_pts and end_pts in seconds.

        Args:
            start_pts: Start time in seconds
            end_pts: End time in seconds (None = read until end)

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
            if frame.time < float(start_pts):
                continue
            if frame.time > end_pts_float:
                break
            yield frame

    def _read_frame_at(self, pts: SECOND_TYPE = 0.0) -> av.VideoFrame:
        """Read single frame at or after given timestamp.

        Args:
            pts: Timestamp in seconds

        Returns:
            Video frame at or after the specified timestamp

        Raises:
            ValueError: If frame not found
        """
        for frame in self._read_frames(start_pts=pts, end_pts=None):
            return frame
        raise ValueError(f"Frame not found at {float(pts):.2f}s in {self.source}")

    def close(self):
        """Release video decoder resources.

        Safe to call multiple times. Closes the underlying PyAV container
        and releases any cached resources.
        """
        if hasattr(self, "_container"):
            self._container.close()
