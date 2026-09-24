"""Header-only video inspection; absent values are not estimated."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .. import cached_av


@dataclass(frozen=True)
class VideoHeader:
    num_frames: Optional[int]
    average_rate: Optional[Fraction]
    duration_seconds: Optional[Fraction]
    width: int
    height: int
    pixel_format: Optional[str]


def probe_video(source, *, storage_options=None) -> VideoHeader:
    """Inspect the first video stream without decoding or estimating frame count."""
    with cached_av.open(source, "r", storage_options=storage_options) as container:
        if not container.streams.video:
            raise ValueError("No video stream found")
        stream = container.streams.video[0]
        return VideoHeader(
            num_frames=stream.frames or None,
            average_rate=Fraction(stream.average_rate) if stream.average_rate else None,
            duration_seconds=(
                Fraction(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base
                else None
            ),
            width=stream.width,
            height=stream.height,
            pixel_format=stream.format.name if stream.format else None,
        )
