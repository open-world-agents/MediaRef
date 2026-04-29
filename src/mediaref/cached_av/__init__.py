import os
import threading
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

    Thread-safety: when ``keep_av_open=True``, the cache lookup-and-insert
    is atomic, so concurrent calls for the same ``file`` yield a single
    cached container with refs incremented per caller.
    """
    if mode == "r":
        if not keep_av_open:
            # Direct access without caching
            return av.open(file, "r", **kwargs)

        # Atomic get-or-create: factory builds a fresh MockedInputContainer
        # only on the cache-miss path; concurrent callers receive the same
        # instance with refs incremented.
        cache_key = str(file)

        def _factory():
            container = MockedInputContainer(file, **kwargs)
            return container, lambda: container.__exit__(None, None, None)

        return _container_cache.get_or_add(cache_key, _factory)
    else:
        return av.open(file, mode, **kwargs)


def cleanup_cache():
    """Clear all cached video containers from memory."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached wrapper for PyAV InputContainer with reference counting.

    Cache registration is performed by :func:`open` via
    ``ResourceCache.get_or_add``; this class no longer self-registers in
    its ``__init__``. ``close`` is idempotent — repeated calls on the
    same instance release the cache reference at most once.
    """

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)
        self._close_lock = threading.Lock()
        self._released = False

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Release container reference and cleanup when no longer needed.

        Idempotent and thread-safe: repeated calls after the first are
        no-ops, even from different threads.
        """
        with self._close_lock:
            if self._released:
                return
            self._released = True
        if self._cache_key in _container_cache and _container_cache[self._cache_key].refs > 0:
            _container_cache.release_entry(self._cache_key)


__all__ = ["open", "cleanup_cache"]
