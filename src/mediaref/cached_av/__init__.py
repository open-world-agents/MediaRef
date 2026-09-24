import os
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Literal, Mapping, Optional, Type, overload

import av
import av.container

from .._features import require_video
from .._internal import is_cloud_uri, make_cache_key, open_cloud
from .._typing import PathLike
from ..resource_cache import ResourceCache
from .input_container_mixin import InputContainerMixin

require_video()

DEFAULT_CACHE_SIZE = int(os.environ.get("AV_CACHE_SIZE", 10))

_container_cache: ResourceCache["MockedInputContainer"] = ResourceCache(max_size=DEFAULT_CACHE_SIZE)


@overload
def open(
    file: PathLike,
    mode: Literal["r"],
    *,
    keep_av_open: bool = False,
    storage_options: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> av.container.InputContainer: ...


@overload
def open(file: PathLike, mode: Literal["w"], **kwargs) -> av.container.OutputContainer: ...


def open(
    file: PathLike,
    mode: Literal["r", "w"],
    *,
    keep_av_open: bool = False,
    storage_options: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """Open a video container, optionally caching read containers across calls.

    For cloud URIs (``hf://``, ``s3://``, …) the cached path opens the
    underlying fsspec file-like internally and ties its lifetime to the
    cache entry, so cross-call HITs get a container whose I/O backend is
    still alive. Raw file-likes from external callers cannot be cached
    safely (their owner / lifetime is unknown to us) and bypass the cache.

    Cached containers are per thread, so the returned container must not be
    used from another thread.
    """
    if mode == "r":
        if not isinstance(file, (str, os.PathLike)):
            # Externally-owned file-like: not cacheable.
            return av.open(file, "r", **kwargs)
        if isinstance(file, str) and is_cloud_uri(file):
            if not keep_av_open:
                return MockedInputContainer(
                    file,
                    cache_key=None,
                    storage_options=storage_options,
                    **kwargs,
                )
            cache_key = _thread_cache_key(make_cache_key(file, kwargs, storage_options or {}))
            return _open_cached(cache_key, file, storage_options=storage_options, **kwargs)
        if not keep_av_open:
            return av.open(file, "r", **kwargs)
        cache_key = _thread_cache_key(make_cache_key(str(file), kwargs))
        return _open_cached(cache_key, file, storage_options=storage_options, **kwargs)
    if storage_options:
        raise ValueError("storage_options are only supported for reading")
    return av.open(file, mode, **kwargs)


def _thread_cache_key(key: str) -> str:
    # A PyAV container is not thread-safe: concurrent seek/decode on one
    # AVFormatContext segfaults. Scope cached containers to the calling thread.
    return f"{key}#thread={threading.get_ident()}"


def _open_cached(
    cache_key: str,
    file: PathLike,
    *,
    storage_options: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> "MockedInputContainer":
    cached = _container_cache.try_acquire(cache_key)
    if cached is not None:
        return cached

    container = MockedInputContainer(
        file,
        cache_key=cache_key,
        storage_options=storage_options,
        **kwargs,
    )
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
    """Refcounted PyAV InputContainer wrapper. One instance is shared by
    concurrent acquirers of the same source; each pairs its acquire with
    one :meth:`close`. For cloud URIs the entry also owns the fsspec
    ``OpenFile`` so its lifetime cannot outlast the av container's I/O.
    """

    _owned_open_ctx: Optional[AbstractContextManager]

    def __init__(
        self,
        file: PathLike,
        *,
        cache_key: Optional[str],
        storage_options: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        self._cache_key = cache_key
        self._owned_open_ctx = None
        self._disposed = False
        try:
            if isinstance(file, str) and is_cloud_uri(file):
                # Own the OpenFile (not just its yielded file): some fsspec
                # backends do connection-pool / tempfile cleanup in __exit__
                # beyond closing the inner file.
                self._owned_open_ctx = open_cloud(file, storage_options=storage_options)
                fileobj = self._owned_open_ctx.__enter__()
                self._container = av.open(fileobj, "r", **kwargs)
            else:
                self._container = av.open(file, "r", **kwargs)
        except Exception:
            if self._owned_open_ctx is not None:
                self._owned_open_ctx.__exit__(None, None, None)
                self._owned_open_ctx = None
            raise

    def __enter__(self) -> "MockedInputContainer":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False

    def close(self):
        """Release one cache reference. Real teardown happens at eviction.
        Idempotent: extra calls past the acquire count are no-ops."""
        if self._cache_key is None:
            self._dispose()
            return
        if _container_cache.refs(self._cache_key) <= 0:
            return
        _container_cache.release_if(self._cache_key, self)

    def _dispose(self):
        """Eviction-time teardown: close av container and owned OpenFile."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self._container.close()
        finally:
            if self._owned_open_ctx is not None:
                self._owned_open_ctx.__exit__(None, None, None)
                self._owned_open_ctx = None


__all__ = ["open", "cleanup_cache"]
