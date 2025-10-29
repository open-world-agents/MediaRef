"""MediaRef - Lightweight media reference management for images and videos.

Public API:
    - MediaRef: Core class for media references
    - DataURI: Data URI encoding and decoding
    - batch_decode: Efficient batch decoding of multiple media references
    - cleanup_cache: Clear video container cache
"""

from loguru import logger

from .core import MediaRef
from .data_uri import DataURI

# Disable logging by default, which is best practice for library code
logger.disable("mediaref")

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


# Lazy import for batch_decode to avoid importing video dependencies at module level
def __getattr__(name: str):
    """Lazy import for batch_decode and related functions."""
    if name == "batch_decode":
        from .batch import batch_decode

        return batch_decode
    elif name == "cleanup_cache":
        from .batch import cleanup_cache

        return cleanup_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MediaRef", "DataURI", "batch_decode", "cleanup_cache"]
