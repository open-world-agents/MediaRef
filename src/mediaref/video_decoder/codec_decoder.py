"""Shared leases for decoders implementing the TorchCodec playback interface."""

from __future__ import annotations

import os
import threading
from abc import abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from .._internal import is_cloud_uri, make_cache_key, open_cloud
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch


@dataclass
class _DecoderState:
    decoder: Any
    lock: threading.RLock
    owned_open_context: Optional[AbstractContextManager] = None
    disposed: bool = False


# A decoder inherited across fork must never be finalized in the child while
# FFmpeg worker threads/locks belong to the parent process.
_inherited_states: list[_DecoderState] = []


class CodecVideoDecoder(BaseVideoDecoder):
    """Common cache, ownership, locking and NumPy batch interface for codec adapters.

    Concrete backends provide a native decoder factory and array conversion.
    Each backend owns a separate cache; active leases reopen after cache clear.
    """

    cache: ClassVar[ResourceCache[_DecoderState]]

    @staticmethod
    @abstractmethod
    def _open_decoder(source: Any, **options: Any) -> Any:
        """Construct a backend decoder without taking ownership of caller files."""

    @staticmethod
    @abstractmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Expose a backend array as a host NumPy array."""

    def __init__(
        self,
        source: Any,
        *,
        storage_options: Optional[Mapping[str, Any]] = None,
        output_format: str = "rgb",
        **kwargs: Any,
    ):
        super().__init__(source, **kwargs)
        if output_format != "rgb":
            raise ValueError("Native video output requires the PyAV backend")
        self._options = dict(kwargs)
        self._storage_options = dict(storage_options or {})
        self._dimension_order = kwargs.get("dimension_order", "NCHW")
        self._closed = False
        self._pid = os.getpid()
        self._lease_lock = threading.RLock()
        cacheable = isinstance(source, (str, os.PathLike))
        self._cache_key = make_cache_key(str(source), kwargs, self._storage_options) if cacheable else None
        self._state: Optional[_DecoderState] = self._acquire_state()

    def _acquire_state(self) -> _DecoderState:
        state = self.cache.try_acquire(self._cache_key) if self._cache_key is not None else None
        if state is not None:
            if state.disposed:
                self.cache.release_if(self._cache_key, state)
                raise RuntimeError("Cached decoder cleanup failed; retry cleanup_cache() before decoding")
            return state
        context = None
        source = self.source
        try:
            if isinstance(source, str) and is_cloud_uri(source):
                context = open_cloud(source, storage_options=self._storage_options)
                source = context.__enter__()
            elif isinstance(source, os.PathLike):
                source = str(source)
            elif hasattr(source, "read") and hasattr(source, "seek"):
                # A reopened file-like decoder must start at the container header.
                source.seek(0)
            candidate = _DecoderState(self._open_decoder(source, **self._options), threading.RLock(), context)
        except BaseException:
            if context is not None:
                context.__exit__(None, None, None)
            raise
        if self._cache_key is None:
            return candidate
        state, _ = self.cache.try_insert_or_acquire(
            self._cache_key,
            candidate,
            cleanup_callback=lambda state=candidate: CodecVideoDecoder._dispose_state(state),
        )
        if state is not candidate:
            self._dispose_state(candidate)
        return state

    @staticmethod
    def _dispose_state(state: _DecoderState) -> None:
        with state.lock:
            if state.disposed:
                return
            state.disposed = True
            try:
                decoder, state.decoder = state.decoder, None
                close = getattr(decoder, "close", None)
                try:
                    if close is not None:
                        close()
                finally:
                    # TorchCodec releases native resources when its last reference drops.
                    del close, decoder
            finally:
                context, state.owned_open_context = state.owned_open_context, None
                if context is not None:
                    context.__exit__(None, None, None)

    def _after_fork(self) -> None:
        if self._pid != os.getpid():
            if self._state is not None and not self._state.disposed:
                _inherited_states.append(self._state)
            self._state = None
            self._lease_lock = threading.RLock()
            self._pid = os.getpid()

    @contextmanager
    def _guard(self):
        self._after_fork()  # Replace inherited locks before trying to acquire one.
        with self._lease_lock:
            if self._closed:
                raise RuntimeError(f"{type(self).__name__} is closed")
            while True:
                if self._state is None:
                    self._state = self._acquire_state()
                state = self._state
                with state.lock:
                    if not state.disposed:
                        yield state.decoder
                        return
                # Cache clear (including the before-fork handler) invalidates the
                # old state. Active leases reopen without releasing a successor.
                self._state = None

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._guard() as decoder:
            return getattr(decoder, name)(*args, **kwargs)

    @property
    def metadata(self) -> Any:
        with self._guard() as decoder:
            return decoder.metadata

    def __len__(self) -> int:
        with self._guard() as decoder:
            return len(decoder)

    def __getitem__(self, key: Any) -> Any:
        with self._guard() as decoder:
            return decoder[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        with self._guard() as decoder:
            attribute = getattr(decoder, name)
        if callable(attribute):
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        return attribute

    def _to_frame_batch(self, batch: Any) -> FrameBatch:
        data = self._to_numpy(batch.data)
        if self._dimension_order == "NHWC":
            data = np.moveaxis(data, -1, 1)
        return FrameBatch(
            data,
            self._to_numpy(batch.pts_seconds).astype(np.float64, copy=False),
            self._to_numpy(batch.duration_seconds).astype(np.float64, copy=False),
        )

    def get_frames_played_at(self, seconds: list[float]) -> FrameBatch:
        return self._to_frame_batch(self._call("get_frames_played_at", seconds))

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        options = {"start_seconds": start_seconds, "stop_seconds": stop_seconds}
        if fps is not None:
            options["fps"] = fps
        return self._to_frame_batch(self._call("get_frames_played_in_range", **options))

    def close(self) -> None:
        self._after_fork()
        with self._lease_lock:
            if self._closed:
                return
            self._closed = True
            state, self._state = self._state, None
            if state is not None:
                if self._cache_key is None:
                    self._dispose_state(state)
                else:
                    self.cache.release_if(self._cache_key, state)

    @classmethod
    def clear_cache(cls) -> None:
        cls.cache.clear()
