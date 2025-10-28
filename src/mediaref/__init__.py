"""MediaRef - Lightweight media reference management for images and videos."""

from .core import MediaRef

# Version is managed by hatch-vcs from Git tags
try:
    from ._version import __version__
except ImportError:
    # Fallback for editable installs without build
    try:
        from importlib.metadata import version

        __version__ = version("mediaref")
    except Exception:
        __version__ = "0.0.0.dev0"

__all__ = ["MediaRef"]

# Optional loader module (requires extra dependencies)
try:
    from .loader import cleanup_cache, load_batch  # noqa: F401
    from .video_decoder import BaseVideoDecoder, FrameBatch, PyAVVideoDecoder, TorchCodecVideoDecoder  # noqa: F401
    from .video_decoder.types import BatchDecodingStrategy, VideoStreamMetadata  # noqa: F401

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
