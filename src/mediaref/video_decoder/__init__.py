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

# Ensure video dependencies are available
require_video()

if TYPE_CHECKING:
    # IDE / mypy / static analysis can see the symbol; runtime resolution is lazy.
    from .torchcodec_decoder import TorchCodecVideoDecoder  # noqa: F401

__all__ = [
    "BaseVideoDecoder",
    "FrameBatch",
    "PyAVVideoDecoder",
    "TorchCodecVideoDecoder",  # surfaced lazily via __getattr__ below
    "VideoStreamMetadata",
]


def __getattr__(name: str):
    """Lazy resolver for optional decoders (PEP 562).

    Defers loading torchcodec until first access. PyAV-only callers never
    trigger torchcodec's shared-library load, so a broken torchcodec install
    (e.g. FFmpeg ABI mismatch) cannot break ``import mediaref``.
    """
    if name == "TorchCodecVideoDecoder":
        try:
            from .torchcodec_decoder import TorchCodecVideoDecoder
        except ImportError as e:
            raise ImportError(
                "TorchCodecVideoDecoder requires the optional `torchcodec` "
                "package: pip install torchcodec"
            ) from e
        # OSError / RuntimeError from torchcodec's .so load (e.g. FFmpeg ABI
        # mismatch — `libavcodec.so.NN: cannot open shared object file`)
        # propagate unchanged so users see the real cause and can run
        # `patch-torchcodec` to repair (see scripts/patch_torchcodec/).
        globals()[name] = TorchCodecVideoDecoder
        return TorchCodecVideoDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
