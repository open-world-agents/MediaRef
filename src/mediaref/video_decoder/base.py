"""Base interface for video decoders."""

from abc import ABC, abstractmethod
from typing import List

from .._typing import PathLike
from .frame_batch import FrameBatch


class BaseVideoDecoder(ABC):
    """Abstract base class defining the minimal interface for video decoders.

    This interface provides the essential functionality required by batch_decode:
    - get_frames_played_at: Retrieve frames at specific timestamps
    - Context manager protocol for resource management

    The design follows TorchCodec's playback semantics where frames are displayed
    from their pts until the next frame's pts.

    Examples:
        >>> with PyAVVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
        ...     print(batch.data.shape)  # (3, 3, H, W)
    """

    def __init__(self, source: PathLike, **kwargs):
        """Initialize video decoder.

        Args:
            source: Path to video file or URL
            **kwargs: Decoder-specific options
        """
        self.source = source

    @abstractmethod
    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames that would be displayed at specific timestamps.

        Follows TorchCodec's playback semantics: a frame is displayed from its pts
        until the next frame's pts. For a given timestamp t, returns the frame where
        frame[i].pts <= t < frame[i+1].pts.

        Args:
            seconds: List of timestamps in seconds to retrieve frames at

        Returns:
            FrameBatch containing:
                - data: Frame data in NCHW format (N, C, H, W)
                - pts_seconds: Presentation timestamps in seconds
                - duration_seconds: Frame durations in seconds

        Raises:
            ValueError: If timestamp < 0 or timestamp >= end_stream_seconds

        Examples:
            >>> with PyAVVideoDecoder("video.mp4") as decoder:
            ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
            ...     print(batch.data.shape)  # (3, 3, H, W)
        """
        pass

    @abstractmethod
    def close(self):
        """Release video decoder resources.

        Should be called when done with the decoder to free up resources.
        Safe to call multiple times.
        """
        pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        """Exit context manager and release resources."""
        self.close()


__all__ = ["BaseVideoDecoder"]
