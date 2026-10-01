"""Video decoder module providing unified interface for video decoding.

This module provides a minimal interface for video decoding through the
BaseVideoDecoder abstract class, with TensorCodec as the default and optional PyAV/TorchCodec backends.

Classes:
    BaseVideoDecoder: Abstract base class defining the decoder interface
    FrameBatch: Data structure for batch frame data
    TensorCodecVideoDecoder: Default CPU decoder with NumPy output
    PyAVVideoDecoder: PyAV-based decoder implementation
    TorchCodecVideoDecoder: TorchCodec-based decoder implementation (lazy; optional)

Examples:
    >>> with TensorCodecVideoDecoder("video.mp4") as decoder:
    ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
"""

from typing import TYPE_CHECKING

from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .types import VideoStreamMetadata

if TYPE_CHECKING:
    from .tensorcodec_decoder import TensorCodecVideoDecoder
    from .pyav_decoder import PyAVVideoDecoder
    from .torchcodec_decoder import TorchCodecVideoDecoder

__all__ = [
    "BaseVideoDecoder",
    "FrameBatch",
    "TensorCodecVideoDecoder",
    "PyAVVideoDecoder",
    "TorchCodecVideoDecoder",
    "VideoStreamMetadata",
]


def __getattr__(name: str):
    """Lazy resolver (PEP 562) for optional decoders."""
    if name == "TensorCodecVideoDecoder":
        try:
            from .tensorcodec_decoder import TensorCodecVideoDecoder
        except ImportError as error:
            raise ImportError(
                "The default TensorCodec decoder requires Python>=3.10 and the video extra. "
                "Install with: pip install 'mediaref[video]'. "
                "Linux x86_64 wheels bundle FFmpeg; other platforms need a source build. "
                "For the legacy backend install 'mediaref[pyav]' and select decoder='pyav'."
            ) from error
        globals()[name] = TensorCodecVideoDecoder
        return TensorCodecVideoDecoder
    if name == "PyAVVideoDecoder":
        from .._features import require_video

        require_video()
        from .pyav_decoder import PyAVVideoDecoder

        globals()[name] = PyAVVideoDecoder
        return PyAVVideoDecoder
    if name == "TorchCodecVideoDecoder":
        try:
            from .torchcodec_decoder import TorchCodecVideoDecoder
        except ImportError as e:
            raise ImportError(
                "TorchCodecVideoDecoder requires the TorchCodec extra. "
                "Install with: pip install 'mediaref[torchcodec]'"
            ) from e
        # TorchCodec 0.16 defers FFmpeg load failures until a video operation;
        # construction errors propagate unchanged so callers see the root cause.
        globals()[name] = TorchCodecVideoDecoder
        return TorchCodecVideoDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
