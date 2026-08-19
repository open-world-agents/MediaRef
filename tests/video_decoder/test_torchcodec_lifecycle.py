"""Unit tests for TorchCodec leases that do not require TorchCodec or FFmpeg."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


class _FakeTensor:
    def __init__(self, value: Any, device: str = "cpu"):
        self._value = np.asarray(value)
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return _FakeTensor(self._value)

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise TypeError("cannot convert a CUDA tensor directly to NumPy")
        return self._value


class _FakeVideoDecoder:
    created = 0
    active_calls = 0
    max_active_calls = 0
    activity_lock = threading.Lock()

    def __init__(self, source: str, **kwargs: Any):
        type(self).created += 1
        self.source = source
        self.options = kwargs
        self.metadata = SimpleNamespace(width=4, height=3)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: Any) -> Any:
        return key

    def get_frames_played_at(self, seconds: list[float]) -> SimpleNamespace:
        with type(self).activity_lock:
            type(self).active_calls += 1
            type(self).max_active_calls = max(type(self).max_active_calls, type(self).active_calls)
        time.sleep(0.02)
        with type(self).activity_lock:
            type(self).active_calls -= 1
        count = len(seconds)
        dimension_order = self.options.get("dimension_order", "NCHW")
        shape = (count, 3, 4, 3) if dimension_order == "NHWC" else (count, 3, 3, 4)
        device = self.options.get("device", "cpu")
        return SimpleNamespace(
            data=_FakeTensor(np.zeros(shape, dtype=np.uint8), device),
            pts_seconds=_FakeTensor(seconds, device),
            duration_seconds=_FakeTensor(np.full(count, 0.1), device),
        )

    def get_frames_played_in_range(self, **kwargs: Any) -> SimpleNamespace:
        return self.get_frames_played_at([kwargs["start_seconds"]])


@pytest.fixture
def torchcodec_decoder_module(monkeypatch: pytest.MonkeyPatch):
    """Load the wrapper against a small in-memory TorchCodec substitute."""
    torchcodec = ModuleType("torchcodec")
    decoders = ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = _FakeVideoDecoder
    torchcodec.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)

    module_name = "mediaref.video_decoder.torchcodec_decoder"
    previous = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module.TorchCodecVideoDecoder.clear_cache()
    _FakeVideoDecoder.created = 0
    _FakeVideoDecoder.active_calls = 0
    _FakeVideoDecoder.max_active_calls = 0
    yield module
    module.TorchCodecVideoDecoder.clear_cache()
    sys.modules.pop(module_name, None)
    if previous is not None:
        sys.modules[module_name] = previous


def test_cache_returns_independent_leases(torchcodec_decoder_module):
    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder

    first = decoder_class("video.mp4")
    second = decoder_class("video.mp4")

    assert first is not second
    assert first._state is second._state
    assert _FakeVideoDecoder.created == 1
    assert decoder_class.cache.refs("video.mp4") == 2

    first.close()
    first.close()
    assert decoder_class.cache.refs("video.mp4") == 1
    assert second.get_frames_played_at([0.0]).data.shape == (1, 3, 3, 4)

    second.close()
    assert decoder_class.cache.refs("video.mp4") == 0
    with pytest.raises(RuntimeError, match="closed"):
        second.get_frames_played_at([0.0])


def test_decoder_options_are_part_of_cache_identity(torchcodec_decoder_module):
    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder

    exact = decoder_class("video.mp4", seek_mode="exact")
    approximate = decoder_class("video.mp4", seek_mode="approximate")
    exact_again = decoder_class("video.mp4", seek_mode="exact")

    assert exact._state is exact_again._state
    assert exact._state is not approximate._state
    assert _FakeVideoDecoder.created == 2

    exact.close()
    approximate.close()
    exact_again.close()


def test_cuda_and_nhwc_outputs_are_normalized(torchcodec_decoder_module):
    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder
    decoder = decoder_class("video.mp4", device="cuda", dimension_order="NHWC")

    batch = decoder.get_frames_played_at([0.0, 0.1])

    assert batch.data.shape == (2, 3, 3, 4)
    assert batch.pts_seconds.dtype == np.float64
    assert batch.duration_seconds.dtype == np.float64
    decoder.close()


def test_shared_decoder_calls_are_serialized(torchcodec_decoder_module):
    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder
    first = decoder_class("video.mp4")
    second = decoder_class("video.mp4")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first.get_frames_played_at, [0.0]),
            pool.submit(second.get_frames_played_at, [0.1]),
        ]
        for future in futures:
            future.result()

    assert _FakeVideoDecoder.max_active_calls == 1
    first.close()
    second.close()


def test_cleanup_cache_drops_loaded_torchcodec_state(torchcodec_decoder_module):
    from mediaref import cleanup_cache

    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder
    decoder = decoder_class("video.mp4")
    assert len(decoder_class.cache) == 1

    cleanup_cache()

    assert len(decoder_class.cache) == 0
    decoder.close()


def test_old_lease_does_not_release_replacement_state(torchcodec_decoder_module):
    decoder_class = torchcodec_decoder_module.TorchCodecVideoDecoder
    old = decoder_class("video.mp4")

    decoder_class.clear_cache()
    replacement = decoder_class("video.mp4")
    assert old._state is not replacement._state
    assert decoder_class.cache.refs("video.mp4") == 1

    old.close()

    assert decoder_class.cache.refs("video.mp4") == 1
    assert replacement.get_frames_played_at([0.0]).data.shape == (1, 3, 3, 4)
    replacement.close()
