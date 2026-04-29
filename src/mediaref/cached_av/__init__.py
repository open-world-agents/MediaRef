import os
from typing import Literal, overload

import av
import av.container

from .._features import require_video
from .._typing import PathLike
from ..resource_cache import ResourceCache
from .input_container_mixin import InputContainerMixin

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
    """Open a video container, optionally caching read containers across calls."""
    if mode == "r":
        if not keep_av_open:
            return av.open(file, "r", **kwargs)
        return _open_cached(str(file), file, **kwargs)
    return av.open(file, mode, **kwargs)


def _open_cached(cache_key: str, file: PathLike, **kwargs) -> "MockedInputContainer":
    cached = _container_cache.try_acquire(cache_key)
    if cached is not None:
        return cached

    container = MockedInputContainer(file, **kwargs)
    # Eviction callback closes the underlying av container; routing through
    # MockedInputContainer.close would re-enter release on the entry being evicted.
    canonical, was_added = _container_cache.try_insert_or_acquire(
        cache_key, container, lambda c=container: InputContainerMixin.close(c)
    )
    if was_added:
        return container
    InputContainerMixin.close(container)
    return canonical


def cleanup_cache():
    """Clear all cached video containers from memory."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached PyAV InputContainer wrapper. One instance is shared across concurrent
    callers of :func:`open` for the same file; each caller pairs its acquire with a
    single :meth:`close`."""

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._container: av.container.InputContainer = av.open(file, "r", **kwargs)

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        try:
            _container_cache.release(self._cache_key)
        except KeyError:
            pass


__all__ = ["open", "cleanup_cache"]
