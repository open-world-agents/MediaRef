import os
from typing import Literal, overload

import av
import av.container

from .._features import require_video
from .._typing import PathLike
from ..resource_cache import ResourceCache
from .input_container_mixin import InputContainerMixin

# Ensure video dependencies are available
require_video()

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))

# Global container cache for efficient video file access
_container_cache: ResourceCache["MockedInputContainer"] = ResourceCache(max_size=DEFAULT_CACHE_SIZE)


@overload
def open(
    file: PathLike, mode: Literal["r"], *, keep_av_open: bool = False, **kwargs
) -> av.container.InputContainer: ...


@overload
def open(file: PathLike, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(file: PathLike, mode: Literal["r", "w"], *, keep_av_open: bool = False, **kwargs):
    """
    Open video container with optional caching for read operations.

    Args:
        file: Video file path or URL
        mode: Access mode ('r' for read, 'w' for write)
        keep_av_open: Enable caching for read containers
        **kwargs: Additional arguments passed to av.open

    Thread-safety: when ``keep_av_open=True`` the cache lookup and the
    insertion of a freshly-opened container are atomic via
    :meth:`ResourceCache.try_acquire` / :meth:`ResourceCache.try_add`.
    The actual ``av.open`` call (which may be slow for remote files)
    runs *outside* the cache lock, so concurrent opens for *different*
    files do not serialize. Concurrent opens for the *same* file produce
    a single cached container with refs incremented per caller; if two
    threads race and both build a fresh container, the loser closes its
    throwaway and acquires the winner's cached entry instead.
    """
    if mode == "r":
        if not keep_av_open:
            # Direct access without caching
            return av.open(file, "r", **kwargs)

        cache_key = str(file)
        return _open_cached(cache_key, file, **kwargs)
    else:
        return av.open(file, mode, **kwargs)


def _open_cached(cache_key: str, file: PathLike, **kwargs) -> "MockedInputContainer":
    """Atomic open-and-cache without serializing ``av.open`` across cache keys.

    Heavy construction happens outside :class:`ResourceCache`'s lock; the
    cache is only locked for the constant-time ``try_acquire`` /
    ``try_add`` checkpoints.
    """
    # Bound the retry loop defensively: with concurrent ``cleanup_cache()``
    # calls a thread could in principle observe an empty cache, build
    # again, and lose another race. In practice this converges in 1–2
    # iterations.
    for _ in range(8):
        cached = _container_cache.try_acquire(cache_key)
        if cached is not None:
            return cached

        # Heavy work outside the cache lock.
        container = MockedInputContainer(file, **kwargs)
        # Cleanup callback: when the cache pops or clears this entry, close
        # the underlying av container directly. Going through
        # MockedInputContainer.close() would route back into release_entry
        # and is wrong at eviction time (the entry is already being
        # destroyed).
        if _container_cache.try_add(cache_key, container, container._underlying_close):
            return container

        # Lost the race: another thread inserted first. Discard our
        # throwaway container and loop to acquire the winner's entry.
        container._underlying_close()

    raise RuntimeError(f"cached_av.open: persistent cache contention prevented stable insertion for {cache_key!r}")


def cleanup_cache():
    """Clear all cached video containers from memory."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached wrapper for PyAV InputContainer with reference counting.

    A single instance is shared across concurrent callers of
    :func:`open` for the same file; each caller is responsible for
    exactly one matching :meth:`close` call. ``close`` decrements the
    cache reference exactly once per call and silently no-ops if the
    entry has already been evicted.
    """

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)

    def __enter__(self) -> "MockedInputContainer":
        return self

    def _underlying_close(self) -> None:
        """Close the underlying PyAV container without touching the cache.

        Used by the cache as the eviction cleanup callback and by
        :func:`_open_cached` to discard a throwaway container after a
        lost race.
        """
        self._container.close()

    def close(self):
        """Release the cache reference held by this caller.

        Each call to :func:`cached_av.open` increments the cache refs by
        one; one matching :meth:`close` decrements by one. Calling
        ``close`` more times than ``open`` is a contract violation.
        Calling ``close`` after the cache has evicted the entry (e.g.
        via ``cleanup_cache()``) is a silent no-op.
        """
        try:
            _container_cache.release_entry(self._cache_key)
        except KeyError:
            pass


__all__ = ["open", "cleanup_cache"]
