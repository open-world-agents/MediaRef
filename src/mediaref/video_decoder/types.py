"""Type definitions for video decoding."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

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


__all__ = [
    "SECOND_TYPE",
    "VideoStreamMetadata",
]
