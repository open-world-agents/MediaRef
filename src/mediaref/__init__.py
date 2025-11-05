"""MediaRef - Lightweight media reference management for images and videos.

Public API:
    - MediaRef: Core class for media references
    - DataURI: Data URI encoding and decoding
    - batch_decode: Efficient batch decoding of multiple media references
    - cleanup_cache: Clear video container cache
"""

from loguru import logger

from .batch import batch_decode, cleanup_cache
from .core import MediaRef
from .data_uri import DataURI

# Disable logging by default, which is best practice for library code
logger.disable("mediaref")

# Version is managed by hatch-vcs from Git tags
try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    # Fallback for editable installs without build
    try:
        from importlib.metadata import version

        __version__ = version("mediaref")
    except Exception:
        __version__ = "0.0.0.dev0"


__all__ = ["MediaRef", "DataURI", "batch_decode", "cleanup_cache"]
