"""Type definitions for video decoding."""

import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Union

# Type aliases
SECOND_TYPE = Union[float, Fraction]
PTSUnit = Literal["pts", "sec"]


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


class BatchDecodingStrategy(str, enum.Enum):
    """Batch decoding strategy for video frames.
    
    Different strategies optimize for different access patterns:
    
    - SEPARATE: Decode each frame independently by seeking to each timestamp.
      Best for sparse queries (e.g., frames [0, 100, 200, 300]).
      
    - SEQUENTIAL_PER_KEYFRAME_BLOCK: Decode frames in batches, one batch per
      keyframe interval. Balanced approach that works well for both sparse
      and dense queries.
      
    - SEQUENTIAL: Decode all frames in one sequential pass from the first
      requested frame to the last. Best for dense queries (e.g., frames [0-100]).
    
    Examples:
        >>> # Sparse query - use SEPARATE
        >>> decoder.get_frames_at([0, 100, 200], strategy=BatchDecodingStrategy.SEPARATE)
        >>>
        >>> # Dense query - use SEQUENTIAL
        >>> decoder.get_frames_at(list(range(100)), strategy=BatchDecodingStrategy.SEQUENTIAL)
        >>>
        >>> # Mixed query - use SEQUENTIAL_PER_KEYFRAME_BLOCK (default)
        >>> decoder.get_frames_at([0, 10, 20, 100, 110], strategy=BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK)
    """

    SEPARATE = "separate"
    SEQUENTIAL_PER_KEYFRAME_BLOCK = "sequential_per_keyframe_block"
    SEQUENTIAL = "sequential"


__all__ = [
    "SECOND_TYPE",
    "PTSUnit",
    "VideoStreamMetadata",
    "BatchDecodingStrategy",
]

