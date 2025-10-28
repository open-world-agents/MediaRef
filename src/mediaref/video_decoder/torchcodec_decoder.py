from typing import ClassVar

from torchcodec.decoders import VideoDecoder

from .._typing import PathLike
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder


class TorchCodecVideoDecoder(VideoDecoder, BaseVideoDecoder):
    """Cached TorchCodec video decoder for efficient resource management.

    This decoder wraps TorchCodec's VideoDecoder with caching support for
    efficient resource management. It automatically caches decoder instances
    and reuses them when the same video is accessed multiple times.

    The decoder inherits from both TorchCodec's VideoDecoder (for functionality)
    and BaseVideoDecoder (for interface compatibility), ensuring it works
    seamlessly with the MediaRef ecosystem.

    Args:
        source: Path to video file or URL
        **kwargs: Additional arguments passed to TorchCodec's VideoDecoder

    Examples:
        >>> # Basic usage with caching
        >>> decoder1 = TorchCodecVideoDecoder("video.mp4")
        >>> decoder2 = TorchCodecVideoDecoder("video.mp4")  # Reuses cached instance
        >>> assert decoder1 is decoder2
        >>>
        >>> # Context manager usage
        >>> with TorchCodecVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_at([0, 10, 20])
        ...     # Cache reference released on exit
    """

    cache: ClassVar[ResourceCache[VideoDecoder]] = ResourceCache(max_size=10)
    _skip_init = False

    def __new__(cls, source: PathLike, **kwargs):
        """Create or retrieve cached decoder instance."""
        cache_key = str(source)
        if cache_key in cls.cache:
            instance = cls.cache[cache_key].obj
            instance._skip_init = True
        else:
            instance = super().__new__(cls)
            instance._skip_init = False
        return instance

    def __init__(self, source: PathLike, **kwargs):
        """Initialize decoder if not retrieved from cache."""
        if getattr(self, "_skip_init", False):
            return
        super().__init__(str(source), **kwargs)
        self._cache_key = str(source)
        # Register with cache using no-op cleanup (TorchCodec handles cleanup internally)
        self.cache.add_entry(self._cache_key, self, lambda: None)

    def close(self):
        """Release cache reference and decoder resources.

        This method releases the cache reference, allowing the decoder to be
        evicted from the cache when no longer in use. TorchCodec handles
        internal resource cleanup automatically.
        """
        if hasattr(self, "_cache_key"):
            self.cache.release_entry(self._cache_key)
