"""MediaRef - Lightweight media reference management for images and videos."""

from .core import MediaRef

__version__ = "0.1.0"
__all__ = ["MediaRef"]

# Optional loader module (requires extra dependencies)
try:
    from .loader import cleanup_cache, load_batch
    from .video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

    __all__.extend(["load_batch", "cleanup_cache", "PyAVVideoDecoder", "TorchCodecVideoDecoder"])
except ImportError:
    pass
