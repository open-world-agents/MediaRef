"""Video decoder module providing unified interface for video decoding.

This module provides a minimal interface for video decoding through the
BaseVideoDecoder abstract class, with implementations for PyAV and TorchCodec.

Classes:
    BaseVideoDecoder: Abstract base class defining the decoder interface
    FrameBatch: Data structure for batch frame data
    PyAVVideoDecoder: PyAV-based decoder implementation
    TorchCodecVideoDecoder: TorchCodec-based decoder implementation (lazy; optional)

Examples:
    >>> with PyAVVideoDecoder("video.mp4") as decoder:
    ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
"""

from typing import TYPE_CHECKING

from .._features import require_video
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .pyav_decoder import PyAVVideoDecoder
from .types import VideoStreamMetadata

require_video()

if TYPE_CHECKING:
    from .torchcodec_decoder import TorchCodecVideoDecoder  # noqa: F401

__all__ = [
    "BaseVideoDecoder",
    "FrameBatch",
    "PyAVVideoDecoder",
    "TorchCodecVideoDecoder",
    "VideoStreamMetadata",
]


def __getattr__(name: str):
    """Lazy resolver (PEP 562) for optional decoders."""
    if name == "TorchCodecVideoDecoder":
        try:
            from .torchcodec_decoder import TorchCodecVideoDecoder
        except ImportError as e:
            raise ImportError(
                "TorchCodecVideoDecoder requires the optional `torchcodec` package: pip install torchcodec"
            ) from e
        # OSError / RuntimeError from torchcodec's .so load propagate
        # unchanged — users see the real FFmpeg-ABI cause and can run
        # `patch-torchcodec` (see scripts/patch_torchcodec/).
        globals()[name] = TorchCodecVideoDecoder
        return TorchCodecVideoDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
