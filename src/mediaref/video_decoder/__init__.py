"""Video decoder module providing unified interface for video decoding.

This module provides a unified interface for video decoding through the
BaseVideoDecoder abstract class, with implementations for PyAV and TorchCodec.

Classes:
    BaseVideoDecoder: Abstract base class defining the decoder interface
    FrameBatch: Data structure for batch frame data
    PyAVVideoDecoder: PyAV-based decoder implementation
    TorchCodecVideoDecoder: TorchCodec-based decoder implementation (optional)

Examples:
    >>> # Using PyAV decoder
    >>> from mediaref.video_decoder import PyAVVideoDecoder
    >>> decoder = PyAVVideoDecoder("video.mp4")
    >>> batch = decoder.get_frames_at([0, 10, 20])
    >>> decoder.close()
    >>>
    >>> # Using TorchCodec decoder (if available)
    >>> from mediaref.video_decoder import TorchCodecVideoDecoder
    >>> with TorchCodecVideoDecoder("video.mp4") as decoder:
    ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
"""

from .._typing import PathLike
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .pyav_decoder import PyAVVideoDecoder
from .types import BatchDecodingStrategy, VideoStreamMetadata

# Conditional import with graceful fallback for optional dependency
try:
    from .torchcodec_decoder import TorchCodecVideoDecoder

    __all__ = [
        "BaseVideoDecoder",
        "FrameBatch",
        "PyAVVideoDecoder",
        "TorchCodecVideoDecoder",
        "BatchDecodingStrategy",
        "VideoStreamMetadata",
    ]
except ImportError:
    # Provide informative error when TorchCodec is unavailable
    class TorchCodecVideoDecoder:
        """Placeholder for TorchCodec decoder when dependency is not installed."""

        def __init__(self, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

        def __new__(cls, source: PathLike, **kwargs):
            raise ImportError("TorchCodec is not available. Please install it with: pip install torchcodec>=0.4.0")

    __all__ = [
        "BaseVideoDecoder",
        "FrameBatch",
        "PyAVVideoDecoder",
        "TorchCodecVideoDecoder",
        "BatchDecodingStrategy",
        "VideoStreamMetadata",
    ]
