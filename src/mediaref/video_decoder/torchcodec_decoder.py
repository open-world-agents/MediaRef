import warnings
from typing import ClassVar, List, Optional

from torchcodec.decoders import VideoDecoder

from .._typing import PathLike
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch
from .types import BatchDecodingStrategy


class TorchCodecVideoDecoder(VideoDecoder, BaseVideoDecoder):
    """Cached TorchCodec video decoder for efficient resource management.

    This decoder wraps TorchCodec's VideoDecoder with caching support for
    efficient resource management. It automatically caches decoder instances
    and reuses them when the same video is accessed multiple times.

    The decoder inherits from both TorchCodec's VideoDecoder (for functionality)
    and BaseVideoDecoder (for interface compatibility), ensuring it works
    seamlessly with the MediaRef ecosystem.

    Note:
        TorchCodec does not implement batch decoding strategies (SEPARATE,
        SEQUENTIAL, SEQUENTIAL_PER_KEYFRAME_BLOCK). The strategy parameter
        is accepted for API compatibility but ignored. For batch decoding
        strategy support, use PyAVVideoDecoder instead.

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

    @property
    def metadata(self):
        """Access video stream metadata set by TorchCodec's VideoDecoder."""
        return self.__dict__["metadata"]

    @metadata.setter
    def metadata(self, value):
        self.__dict__["metadata"] = value

    def close(self):
        """Release cache reference and decoder resources.

        This method releases the cache reference, allowing the decoder to be
        evicted from the cache when no longer in use. TorchCodec handles
        internal resource cleanup automatically.
        """
        if hasattr(self, "_cache_key"):
            self.cache.release_entry(self._cache_key)

    def get_frames_at(
        self,
        indices: List[int],
        *,
        strategy: Optional[BatchDecodingStrategy] = None,
    ) -> FrameBatch:
        """Retrieve frames at specific frame indices.

        Note:
            TorchCodec does not implement batch decoding strategies. The strategy
            parameter is accepted for API compatibility but is ignored. If a
            strategy is explicitly specified, a warning will be issued.

        Args:
            indices: List of frame indices to retrieve
            strategy: Batch decoding strategy (ignored by TorchCodec, only supported by PyAV)

        Returns:
            FrameBatch containing frame data and timing information
        """
        if strategy is not None:
            warnings.warn(
                f"TorchCodec decoder does not support batch decoding strategies. "
                f"The '{strategy}' strategy is ignored. "
                f"Use PyAVVideoDecoder for batch decoding strategy support.",
                UserWarning,
                stacklevel=2,
            )
        # Call parent's get_frames_at without strategy parameter
        return super().get_frames_at(indices)

    def get_frames_played_at(
        self,
        seconds: List[float],
        *,
        strategy: Optional[BatchDecodingStrategy] = None,
    ) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Note:
            TorchCodec does not implement batch decoding strategies. The strategy
            parameter is accepted for API compatibility but is ignored. If a
            strategy is explicitly specified, a warning will be issued.

        Args:
            seconds: List of timestamps in seconds to retrieve frames at
            strategy: Batch decoding strategy (ignored by TorchCodec, only supported by PyAV)

        Returns:
            FrameBatch containing frame data and timing information
        """
        if strategy is not None:
            warnings.warn(
                f"TorchCodec decoder does not support batch decoding strategies. "
                f"The '{strategy}' strategy is ignored. "
                f"Use PyAVVideoDecoder for batch decoding strategy support.",
                UserWarning,
                stacklevel=2,
            )
        # Call parent's get_frames_played_at without strategy parameter
        return super().get_frames_played_at(seconds)
