"""Base interface for video decoders."""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy.typing as npt

from .._typing import PathLike
from .frame_batch import FrameBatch
from .types import BatchDecodingStrategy, VideoStreamMetadata


class BaseVideoDecoder(ABC):
    """Abstract base class defining the unified interface for all video decoders.

    This interface is designed to be compatible with TorchCodec's VideoDecoder API
    while providing flexibility for different decoder implementations (PyAV, TorchCodec, etc.).

    All decoders must implement:
    - Frame access by index or timestamp
    - Batch decoding with configurable strategies
    - Metadata access
    - Resource management (context manager protocol)

    Examples:
        >>> # Using PyAV decoder
        >>> decoder = PyAVVideoDecoder("video.mp4")
        >>> frames = decoder.get_frames_at([0, 10, 20])
        >>> print(frames.data.shape)  # (3, C, H, W)
        >>>
        >>> # Using TorchCodec decoder
        >>> decoder = TorchCodecVideoDecoder("video.mp4")
        >>> frames = decoder.get_frames_played_at([0.0, 1.0, 2.0])
        >>> decoder.close()
        >>>
        >>> # Context manager usage
        >>> with PyAVVideoDecoder("video.mp4") as decoder:
        ...     frame = decoder[0]  # Array-like indexing
    """

    def __init__(self, source: PathLike, **kwargs):
        """Initialize video decoder.

        Args:
            source: Path to video file or URL
            **kwargs: Decoder-specific options
        """
        self.source = source

    @property
    @abstractmethod
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata.

        Returns:
            VideoStreamMetadata containing:
                - num_frames: Total number of frames
                - duration_seconds: Video duration
                - average_rate: Average frame rate
                - width: Frame width in pixels
                - height: Frame height in pixels

        Examples:
            >>> decoder = PyAVVideoDecoder("video.mp4")
            >>> print(decoder.metadata.num_frames)
            >>> print(decoder.metadata.duration_seconds)
        """
        pass

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]) -> npt.NDArray:
        """Enable array-like indexing for frame access.

        Args:
            key: Frame index (int) or slice

        Returns:
            Frame data as numpy array in NCHW format
            - Single frame: (C, H, W)
            - Multiple frames: (N, C, H, W)

        Raises:
            TypeError: If key is not int or slice
            IndexError: If index is out of bounds

        Examples:
            >>> decoder = PyAVVideoDecoder("video.mp4")
            >>> frame = decoder[0]  # First frame
            >>> frames = decoder[10:20]  # Frames 10-19
            >>> frames = decoder[::10]  # Every 10th frame
        """
        pass

    @abstractmethod
    def get_frames_at(
        self,
        indices: List[int],
        *,
        strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    ) -> FrameBatch:
        """Retrieve frames at specific frame indices.

        Args:
            indices: List of frame indices to retrieve
            strategy: Batch decoding strategy:
                - SEPARATE: Decode each frame separately (best for sparse queries)
                - SEQUENTIAL_PER_KEYFRAME_BLOCK: Decode in batches per keyframe block (balanced)
                - SEQUENTIAL: Decode all frames in one pass (best for dense queries)

        Returns:
            FrameBatch containing:
                - data: Frame data in NCHW format (N, C, H, W)
                - pts_seconds: Presentation timestamps in seconds
                - duration_seconds: Frame durations in seconds

        Examples:
            >>> decoder = PyAVVideoDecoder("video.mp4")
            >>> batch = decoder.get_frames_at([0, 10, 20, 30])
            >>> print(batch.data.shape)  # (4, 3, H, W)
            >>> print(batch.pts_seconds)  # [0.0, 0.333..., 0.666..., 1.0]
        """
        pass

    @abstractmethod
    def get_frames_played_at(
        self,
        seconds: List[float],
        *,
        strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    ) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Args:
            seconds: List of timestamps in seconds to retrieve frames at
            strategy: Batch decoding strategy (see get_frames_at for details)

        Returns:
            FrameBatch containing frame data and timing information

        Raises:
            ValueError: If any timestamp exceeds video duration

        Examples:
            >>> decoder = PyAVVideoDecoder("video.mp4")
            >>> batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
            >>> print(batch.data.shape)  # (3, 3, H, W)
        """
        pass

    @abstractmethod
    def close(self):
        """Release video decoder resources.

        Should be called when done with the decoder to free up resources.
        Safe to call multiple times.

        Examples:
            >>> decoder = PyAVVideoDecoder("video.mp4")
            >>> frames = decoder.get_frames_at([0, 1, 2])
            >>> decoder.close()
        """
        pass

    def __enter__(self):
        """Enter context manager.

        Returns:
            self

        Examples:
            >>> with PyAVVideoDecoder("video.mp4") as decoder:
            ...     frames = decoder.get_frames_at([0, 1, 2])
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        """Exit context manager and release resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.close()


__all__ = ["BaseVideoDecoder"]
