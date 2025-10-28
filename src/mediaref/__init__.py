"""MediaRef - Lightweight media reference management for images and videos."""

from .core import MediaRef

__version__ = "0.1.0"
__all__ = ["MediaRef"]

# Optional loader module (requires extra dependencies)
try:
    from .loader import cleanup_cache, load_batch
    from .video_decoder import BaseVideoDecoder, FrameBatch, PyAVVideoDecoder, TorchCodecVideoDecoder
    from .video_decoder.types import BatchDecodingStrategy, VideoStreamMetadata

    __all__.extend(
        [
            "load_batch",
            "cleanup_cache",
            "BatchDecodingStrategy",
            "VideoStreamMetadata",
            "BaseVideoDecoder",
            "FrameBatch",
            "PyAVVideoDecoder",
            "TorchCodecVideoDecoder",
        ]
    )
except ImportError:
    pass
