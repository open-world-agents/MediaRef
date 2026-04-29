import os
from typing import Literal, Optional, overload

import av
import av.container
import fsspec

from .._features import require_video
from .._internal import is_cloud_uri
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
    """Open a video container, optionally caching read containers across calls.

    For cloud URIs (``hf://``, ``s3://``, …) the cached path opens the
    underlying fsspec file-like internally and ties its lifetime to the
    cache entry, so cross-call HITs get a container whose I/O backend is
    still alive. Raw file-likes from external callers cannot be cached
    safely (their owner / lifetime is unknown to us) and bypass the cache.
    """
    if mode == "r":
        if not isinstance(file, (str, os.PathLike)):
            # Externally-owned file-like: not cacheable.
            return av.open(file, "r", **kwargs)
        if not keep_av_open:
            return av.open(file, "r", **kwargs)
        return _open_cached(str(file), file, **kwargs)
    return av.open(file, mode, **kwargs)


def _open_cached(cache_key: str, file: PathLike, **kwargs) -> "MockedInputContainer":
    cached = _container_cache.try_acquire(cache_key)
    if cached is not None:
        return cached

    container = MockedInputContainer(file, **kwargs)
    # Eviction callback disposes both the av container and any fsspec
    # file-like the cache entry owns.
    canonical, was_added = _container_cache.try_insert_or_acquire(
        cache_key, container, lambda c=container: c._dispose()
    )
    if was_added:
        return container
    container._dispose()
    return canonical


def cleanup_cache():
    """Clear all cached video containers and their owned I/O resources."""
    _container_cache.clear()


class MockedInputContainer(InputContainerMixin):
    """Cached PyAV InputContainer wrapper. One instance is shared across
    concurrent callers of :func:`open` for the same source; each caller
    pairs its acquire with a single :meth:`close`.

    For cloud URIs the entry also owns the underlying fsspec file-like:
    a single ``MockedInputContainer`` ⊇ one ``av.container`` ⊇ one fsspec
    file-like, all disposed together by :meth:`_dispose` at eviction.
    """

    _owned_filelike: Optional[object]

    def __init__(self, file: PathLike, **kwargs):
        self._cache_key = str(file)
        self._owned_filelike = None
        try:
            if isinstance(file, str) and is_cloud_uri(file):
                # Take ownership of the fsspec file-like for the lifetime
                # of this cache entry. ``fsspec.open(...).__enter__()``
                # returns the underlying file (e.g. ``HfFileSystemFile``)
                # without binding it to an outer ``with`` block; we close
                # it ourselves in :meth:`_dispose`.
                self._owned_filelike = fsspec.open(file, "rb").__enter__()
                self._container = av.open(self._owned_filelike, "r", **kwargs)
            else:
                self._container = av.open(file, "r", **kwargs)
        except Exception:
            if self._owned_filelike is not None:
                self._owned_filelike.close()
                self._owned_filelike = None
            raise

    def __enter__(self) -> "MockedInputContainer":
        return self

    def close(self):
        """Public close = release one cache reference. The container and
        its owned I/O are not actually torn down until the cache decides
        to evict (capacity LRU or :func:`cleanup_cache`)."""
        try:
            _container_cache.release(self._cache_key)
        except KeyError:
            pass

    def _dispose(self):
        """Eviction-time real cleanup. Closes the av container and any
        fsspec file-like this entry owns. Idempotent."""
        try:
            self._container.close()
        finally:
            if self._owned_filelike is not None:
                self._owned_filelike.close()
                self._owned_filelike = None


__all__ = ["open", "cleanup_cache"]
