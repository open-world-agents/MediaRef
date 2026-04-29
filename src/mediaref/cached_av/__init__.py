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

    When ``keep_av_open=True`` the cache lookup and the insertion of a freshly-opened
    container are atomic via :meth:`ResourceCache.try_acquire` / :meth:`ResourceCache.try_add`.
    The actual ``av.open`` runs outside the cache lock, so concurrent opens for different
    files do not serialize.
    """
    if mode == "r":
        if not keep_av_open:
            return av.open(file, "r", **kwargs)
        return _open_cached(str(file), file, **kwargs)
    return av.open(file, mode, **kwargs)


def _open_cached(cache_key: str, file: PathLike, **kwargs) -> "MockedInputContainer":
    """Atomic open-and-cache without serializing ``av.open`` across cache keys."""
    cached = _container_cache.try_acquire(cache_key)
    if cached is not None:
        return cached

    container = MockedInputContainer(file, **kwargs)
    # The eviction callback closes the underlying av container directly; routing through
    # ``MockedInputContainer.close`` would re-enter ``release`` on an entry that is
    # already being evicted.
    if _container_cache.try_add(cache_key, container, lambda c=container: InputContainerMixin.close(c)):
        return container

    # Lost the race: another thread inserted first. Discard our throwaway and acquire theirs.
    InputContainerMixin.close(container)
    cached = _container_cache.try_acquire(cache_key)
    if cached is not None:
        return cached

    # Pathological — the entry was inserted and then evicted (e.g. by a concurrent
    # ``cleanup_cache()``) between our ``try_add`` failure and the second ``try_acquire``.
    raise RuntimeError(f"cached_av.open: cache entry vanished mid-insert for {cache_key!r}")


def cleanup_cache():
    """Clear all cached video containers from memory."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached wrapper for PyAV InputContainer with reference counting.

    A single instance is shared across concurrent callers of :func:`open` for the same
    file; each caller is responsible for exactly one matching :meth:`close` call. ``close``
    decrements the cache reference once per call and silently no-ops if the entry has
    already been evicted.
    """

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Release the cache reference held by this caller. No-op if already evicted."""
        try:
            _container_cache.release(self._cache_key)
        except KeyError:
            pass


__all__ = ["open", "cleanup_cache"]
