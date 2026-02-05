"""TorchCodec-based video decoder."""

from typing import ClassVar, List

from torchcodec.decoders import VideoDecoder

from .._typing import PathLike
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch


class TorchCodecVideoDecoder(VideoDecoder, BaseVideoDecoder):
    """Cached TorchCodec video decoder.

    Wraps TorchCodec's VideoDecoder with caching for efficient resource management.

    Args:
        source: Path to video file or URL
        **kwargs: Additional arguments passed to TorchCodec's VideoDecoder

    Examples:
        >>> with TorchCodecVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
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
        self.cache.add_entry(self._cache_key, self, lambda: None)

    @property
    def metadata(self):
        """Access video stream metadata."""
        return self.__dict__["metadata"]

    @metadata.setter
    def metadata(self, value):
        self.__dict__["metadata"] = value

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Delegates to TorchCodec's native get_frames_played_at implementation
        which already implements the correct playback semantics.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch containing frame data and timing information
        """
        return super().get_frames_played_at(seconds)

    def close(self):
        """Release cache reference."""
        if hasattr(self, "_cache_key"):
            self.cache.release_entry(self._cache_key)
