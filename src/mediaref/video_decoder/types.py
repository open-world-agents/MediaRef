"""Type definitions for video decoding."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

# Type aliases
SECOND_TYPE = Union[float, Fraction]


@dataclass
class VideoStreamMetadata:
    """Video stream metadata container.

    Attributes:
        num_frames: Total number of frames in the video
        duration_seconds: Video duration in seconds (as Fraction for precision)
        average_rate: Average frame rate (as Fraction for precision)
        width: Frame width in pixels
        height: Frame height in pixels
        begin_stream_seconds: First frame's PTS in seconds (default 0)
        end_stream_seconds: End of stream in seconds (last_frame.pts + last_frame.duration)

    Examples:
        >>> metadata = VideoStreamMetadata(
        ...     num_frames=300,
        ...     duration_seconds=Fraction(10, 1),
        ...     average_rate=Fraction(30, 1),
        ...     width=1920,
        ...     height=1080
        ... )
        >>> print(f"FPS: {float(metadata.average_rate)}")
        FPS: 30.0
    """

    num_frames: int
    duration_seconds: Fraction
    average_rate: Fraction
    width: int
    height: int
    begin_stream_seconds: Fraction = Fraction(0)
    end_stream_seconds: Optional[Fraction] = None

    def __post_init__(self):
        """Set end_stream_seconds to begin + duration if not provided."""
        if self.end_stream_seconds is None:
            self.end_stream_seconds = self.begin_stream_seconds + self.duration_seconds


__all__ = [
    "SECOND_TYPE",
    "VideoStreamMetadata",
]
