"""TorchCodec-based video decoder."""

from typing import ClassVar, List, Optional

import numpy as np
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

    Thread-safety: cache lookup uses atomic ``try_acquire`` / ``try_add``
    primitives. The heavy ``super().__init__`` runs outside the cache
    lock, so concurrent constructors for *distinct* sources do not
    serialize. For the *same* source, concurrent constructors may both
    build a fresh decoder; the loser's instance becomes uncached and
    its underlying decoder is closed by its own :meth:`close`. Each
    cached caller is responsible for exactly one matching ``close``.
    """

    cache: ClassVar[ResourceCache[VideoDecoder]] = ResourceCache(max_size=10)
    _skip_init = False

    def __new__(cls, source: PathLike, **kwargs):
        """Create or retrieve cached decoder instance.

        Uses atomic ``try_acquire`` so that concurrent constructors do not
        race the cache check. If the entry is absent, the fresh instance is
        built outside the cache lock and only committed in :meth:`__init__`.
        """
        cache_key = str(source)
        cached = cls.cache.try_acquire(cache_key)
        if cached is not None:
            cached._skip_init = True
            return cached
        instance = super().__new__(cls)
        instance._skip_init = False
        return instance

    def __init__(self, source: PathLike, **kwargs):
        """Initialize decoder if not retrieved from cache."""
        if getattr(self, "_skip_init", False):
            return
        super().__init__(str(source), **kwargs)
        # Atomic try_add: returns False if a concurrent thread cached an
        # instance for this key first. In that case the fresh instance is
        # still functional for this caller (the construction above
        # already opened the file), but it's not the cached one, so
        # close() must close it directly rather than touch the cache.
        cache_key = str(source)
        if self.cache.try_add(cache_key, self, lambda: VideoDecoder.close(self)):
            self._cache_key = cache_key
        else:
            self._cache_key = None

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Delegates to TorchCodec's native get_frames_played_at
        which implements the correct playback semantics.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch containing frame data and timing information
        """
        # TorchCodec returns its own FrameBatch (dataclass), convert to ours
        torchcodec_batch = VideoDecoder.get_frames_played_at(self, seconds)
        return FrameBatch(
            data=torchcodec_batch.data.numpy(),
            # Use .numpy() first to avoid DeprecationWarning from numpy 2.0
            # about __array__ not accepting the 'copy' keyword
            pts_seconds=torchcodec_batch.pts_seconds.numpy().astype(np.float64),
            duration_seconds=torchcodec_batch.duration_seconds.numpy().astype(np.float64),
        )

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        """Return multiple frames in the given range.

        Delegates to TorchCodec's native get_frames_played_in_range.

        Args:
            start_seconds: Time, in seconds, of the start of the range.
            stop_seconds: Time, in seconds, of the end of the range (excluded).
            fps: If specified, resample output to this frame rate. If None,
                returns frames at the source video's frame rate.

        Returns:
            FrameBatch containing frame data and timing information.

        Raises:
            NotImplementedError: If ``fps`` is specified but the installed
                TorchCodec version (<=0.10.0) does not support it.
        """
        if fps is not None:
            try:
                torchcodec_batch = VideoDecoder.get_frames_played_in_range(
                    self, start_seconds=start_seconds, stop_seconds=stop_seconds, fps=fps
                )
            except TypeError:
                raise NotImplementedError(
                    "The installed version of TorchCodec (<=0.10.0) does not support "
                    "the 'fps' parameter in get_frames_played_in_range. "
                    "Upgrade TorchCodec or use fps=None."
                )
        else:
            torchcodec_batch = VideoDecoder.get_frames_played_in_range(
                self, start_seconds=start_seconds, stop_seconds=stop_seconds
            )
        return FrameBatch(
            data=torchcodec_batch.data.numpy(),
            pts_seconds=torchcodec_batch.pts_seconds.numpy().astype(np.float64),
            duration_seconds=torchcodec_batch.duration_seconds.numpy().astype(np.float64),
        )

    def close(self):
        """Release cache reference, or close the underlying decoder if uncached.

        For cached instances: each caller decrements the refs by one;
        eviction at refs=0 closes the underlying decoder via the cleanup
        callback registered in :meth:`__init__`. Calling ``close`` after
        the cache has evicted the entry (e.g. via ``cleanup_cache()``)
        is a silent no-op.

        For uncached instances (race-lost during ``__init__``): the
        underlying decoder is closed directly.
        """
        if self._cache_key is not None:
            try:
                self.cache.release_entry(self._cache_key)
            except KeyError:
                # Already evicted by LRU or cleanup_cache().
                pass
        else:
            VideoDecoder.close(self)
