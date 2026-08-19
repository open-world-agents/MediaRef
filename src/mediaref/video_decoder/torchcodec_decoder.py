"""TorchCodec-based video decoder."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import numpy as np
from torchcodec.decoders import VideoDecoder

from .._typing import PathLike
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch


@dataclass
class _DecoderState:
    decoder: VideoDecoder
    lock: threading.RLock


class TorchCodecVideoDecoder(BaseVideoDecoder):
    """A lease on a cached TorchCodec video decoder.

    Leases for the same source and decoder options share the underlying decoder,
    but are independent objects that can each be closed exactly once. Calls on a
    shared decoder are serialized because TorchCodec decoders maintain seek state.

    Args:
        source: Path to video file or URL.
        **kwargs: Additional arguments passed to TorchCodec's ``VideoDecoder``.

    Examples:
        >>> with TorchCodecVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
    """

    cache: ClassVar[ResourceCache[_DecoderState]] = ResourceCache(max_size=10)

    def __init__(self, source: PathLike, **kwargs: Any):
        super().__init__(source, **kwargs)
        self._cache_key = self._make_cache_key(source, kwargs)
        self._dimension_order = kwargs.get("dimension_order", "NCHW")
        self._closed = False

        state = self.cache.try_acquire(self._cache_key)
        if state is None:
            candidate = _DecoderState(
                decoder=VideoDecoder(str(source), **kwargs),
                lock=threading.RLock(),
            )
            state, _ = self.cache.try_insert_or_acquire(
                self._cache_key,
                candidate,
                # TorchCodec's VideoDecoder has no public close() method. Dropping
                # the final state reference lets its native resources be released.
                cleanup_callback=lambda: None,
            )
        self._state: Optional[_DecoderState] = state

    @staticmethod
    def _make_cache_key(source: PathLike, kwargs: dict[str, Any]) -> str:
        source_key = str(source)
        if not kwargs:
            return source_key
        options = tuple(sorted(kwargs.items()))
        digest = hashlib.sha256(repr(options).encode()).hexdigest()
        return f"{source_key}#{digest}"

    def _require_state(self) -> _DecoderState:
        if self._closed or self._state is None:
            raise RuntimeError("TorchCodecVideoDecoder is closed")
        return self._state

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        state = self._require_state()
        with state.lock:
            return getattr(state.decoder, name)(*args, **kwargs)

    @property
    def metadata(self) -> Any:
        state = self._require_state()
        with state.lock:
            return state.decoder.metadata

    def __len__(self) -> int:
        state = self._require_state()
        with state.lock:
            return len(state.decoder)

    def __getitem__(self, key: Any) -> Any:
        state = self._require_state()
        with state.lock:
            return state.decoder[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        state = self._require_state()
        with state.lock:
            attribute = getattr(state.decoder, name)
        if not callable(attribute):
            return attribute

        def locked_call(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, *args, **kwargs)

        return locked_call

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Move a TorchCodec tensor to host memory and expose it as NumPy."""
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        return tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)

    def _to_frame_batch(self, torchcodec_batch: Any) -> FrameBatch:
        data = self._to_numpy(torchcodec_batch.data)
        if self._dimension_order == "NHWC":
            data = np.moveaxis(data, -1, 1)
        return FrameBatch(
            data=data,
            pts_seconds=self._to_numpy(torchcodec_batch.pts_seconds).astype(np.float64),
            duration_seconds=self._to_numpy(torchcodec_batch.duration_seconds).astype(np.float64),
        )

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames displayed at the requested timestamps."""
        return self._to_frame_batch(self._call("get_frames_played_at", seconds))

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        """Return frames in ``[start_seconds, stop_seconds)``.

        Raises:
            NotImplementedError: If ``fps`` is specified but TorchCodec <=0.10
                is installed.
        """
        kwargs = {"start_seconds": start_seconds, "stop_seconds": stop_seconds}
        if fps is not None:
            kwargs["fps"] = fps
        try:
            batch = self._call("get_frames_played_in_range", **kwargs)
        except TypeError:
            if fps is None:
                raise
            raise NotImplementedError(
                "The installed version of TorchCodec (<=0.10.0) does not support "
                "the 'fps' parameter in get_frames_played_in_range. "
                "Upgrade TorchCodec or use fps=None."
            ) from None
        return self._to_frame_batch(batch)

    def close(self) -> None:
        """Release this lease. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        state = self._state
        self._state = None
        if state is not None:
            self.cache.release_if(self._cache_key, state)

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all cached decoder states."""
        cls.cache.clear()
