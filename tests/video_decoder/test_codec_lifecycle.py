"""Shared codec lease contract tested without optional codecs or FFmpeg."""

from __future__ import annotations

import importlib
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from typing import Any

import fsspec
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
    sources = []
    active_calls = 0
    max_active_calls = 0
    closed = 0
    activity_lock = threading.Lock()

    def __init__(self, source: Any, **kwargs: Any):
        type(self).created += 1
        type(self).sources.append(source)
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


class _FakeNumpyDecoder(_FakeVideoDecoder):
    def close(self):
        type(self).closed += 1

    def get_frames_played_at(self, seconds):
        batch = super().get_frames_played_at(seconds)
        return SimpleNamespace(**{name: value.cpu().numpy() for name, value in vars(batch).items()})


@pytest.fixture(params=["torchcodec", "tensorcodec"])
def codec_adapter(request, monkeypatch: pytest.MonkeyPatch):
    backend = request.param
    fake = _FakeVideoDecoder if backend == "torchcodec" else _FakeNumpyDecoder
    package = ModuleType(backend)
    decoders = ModuleType(f"{backend}.decoders")
    decoders.VideoDecoder = fake
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, backend, package)
    monkeypatch.setitem(sys.modules, f"{backend}.decoders", decoders)
    module_name = f"mediaref.video_decoder.{backend}_decoder"
    previous = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    name = "TorchCodecVideoDecoder" if backend == "torchcodec" else "TensorCodecVideoDecoder"
    adapter = getattr(module, name)
    decoder_package = importlib.import_module("mediaref.video_decoder")
    previous_export = decoder_package.__dict__.get(name)
    setattr(decoder_package, name, adapter)
    adapter.clear_cache()
    fake.created = fake.active_calls = fake.max_active_calls = fake.closed = 0
    fake.sources = []
    yield SimpleNamespace(adapter=adapter, fake=fake, backend=backend)
    adapter.clear_cache()
    sys.modules.pop(module_name, None)
    if previous_export is None:
        decoder_package.__dict__.pop(name, None)
    else:
        setattr(decoder_package, name, previous_export)
    if previous is not None:
        sys.modules[module_name] = previous


def test_cache_returns_independent_leases(codec_adapter):
    decoder_class = codec_adapter.adapter

    first = decoder_class("video.mp4")
    second = decoder_class("video.mp4")

    assert first is not second
    assert first._state is second._state
    assert codec_adapter.fake.created == 1
    assert decoder_class.cache.refs("video.mp4") == 2

    first.close()
    first.close()
    assert decoder_class.cache.refs("video.mp4") == 1
    assert second.get_frames_played_at([0.0]).data.shape == (1, 3, 3, 4)

    second.close()
    assert decoder_class.cache.refs("video.mp4") == 0
    with pytest.raises(RuntimeError, match="closed"):
        second.get_frames_played_at([0.0])


def test_decoder_options_are_part_of_cache_identity(codec_adapter):
    decoder_class = codec_adapter.adapter

    exact = decoder_class("video.mp4", seek_mode="exact")
    approximate = decoder_class("video.mp4", seek_mode="approximate")
    exact_again = decoder_class("video.mp4", seek_mode="exact")

    assert exact._state is exact_again._state
    assert exact._state is not approximate._state
    assert codec_adapter.fake.created == 2

    exact.close()
    approximate.close()
    exact_again.close()


def test_cuda_and_nhwc_outputs_are_normalized(codec_adapter):
    decoder_class = codec_adapter.adapter
    decoder = decoder_class(
        "video.mp4", device="cuda" if codec_adapter.backend == "torchcodec" else "cpu", dimension_order="NHWC"
    )

    batch = decoder.get_frames_played_at([0.0, 0.1])

    assert batch.data.shape == (2, 3, 3, 4)
    assert batch.pts_seconds.dtype == np.float64
    assert batch.duration_seconds.dtype == np.float64
    decoder.close()


def test_shared_decoder_calls_are_serialized(codec_adapter):
    decoder_class = codec_adapter.adapter
    first = decoder_class("video.mp4")
    second = decoder_class("video.mp4")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first.get_frames_played_at, [0.0]),
            pool.submit(second.get_frames_played_at, [0.1]),
        ]
        for future in futures:
            future.result()

    assert codec_adapter.fake.max_active_calls == 1
    first.close()
    second.close()


def test_cleanup_cache_drops_loaded_torchcodec_state(codec_adapter):
    from mediaref import cleanup_cache

    decoder_class = codec_adapter.adapter
    decoder = decoder_class("video.mp4")
    assert len(decoder_class.cache) == 1

    cleanup_cache()

    assert len(decoder_class.cache) == 0
    decoder.close()


def test_old_lease_does_not_release_replacement_state(codec_adapter):
    decoder_class = codec_adapter.adapter
    old = decoder_class("video.mp4")

    decoder_class.clear_cache()
    replacement = decoder_class("video.mp4")
    assert old._state is not replacement._state
    assert decoder_class.cache.refs("video.mp4") == 1

    old.close()

    assert decoder_class.cache.refs("video.mp4") == 1
    assert replacement.get_frames_played_at([0.0]).data.shape == (1, 3, 3, 4)
    replacement.close()


def test_fsspec_source_lifetime_is_owned_by_cache(codec_adapter):
    decoder_class = codec_adapter.adapter
    uri = "memory://torchcodec/clip.mp4"
    with fsspec.open(uri, "wb") as file:
        file.write(b"video bytes")

    decoder = decoder_class(uri, storage_options={"client_kwargs": {"region": "test"}})
    opened_file = codec_adapter.fake.sources[-1]
    state = decoder._state

    assert not isinstance(opened_file, str)
    assert not opened_file.closed
    assert state.owned_open_context is not None
    decoder.close()
    assert state.owned_open_context is not None

    decoder_class.clear_cache()
    assert state.owned_open_context is None


def test_storage_options_isolate_cache_without_exposing_values(codec_adapter):
    decoder_class = codec_adapter.adapter
    uri = "memory://torchcodec/options.mp4"
    with fsspec.open(uri, "wb") as file:
        file.write(b"video bytes")

    first = decoder_class(uri, storage_options={"token": "first-secret"})
    second = decoder_class(uri, storage_options={"token": "second-secret"})

    assert first._state is not second._state
    assert len(decoder_class.cache) == 2
    assert all("secret" not in key for key in decoder_class.cache._entries)
    first.close()
    second.close()


def test_external_file_like_is_not_cached_or_closed(codec_adapter):
    decoder_class = codec_adapter.adapter
    source = io.BytesIO(b"video bytes")

    first = decoder_class(source)
    second = decoder_class(source)

    assert first._state is not second._state
    assert len(decoder_class.cache) == 0
    first.close()
    second.close()
    assert not source.closed


def test_mediaref_single_frame_uses_backend_with_fsspec(codec_adapter):
    from mediaref import MediaRef

    uri = "memory://torchcodec/mediaref.mp4"
    with fsspec.open(uri, "wb") as file:
        file.write(b"video bytes")

    frame = MediaRef(uri=uri, pts_ns=0).to_ndarray(
        decoder=codec_adapter.backend,
        storage_options={"token": "secret"},
    )

    assert frame.shape == (3, 4, 3)


def test_active_cloud_lease_reopens_after_cache_clear(codec_adapter):
    uri = f"memory://{codec_adapter.backend}/reopen.mp4"
    with fsspec.open(uri, "wb") as file:
        file.write(b"video")
    with codec_adapter.adapter(uri) as lease:
        old = lease._state
        codec_adapter.adapter.clear_cache()
        assert old.disposed and old.decoder is None
        assert old.owned_open_context is None
        assert lease.get_frames_played_at([0]).data.shape == (1, 3, 3, 4)
        assert lease._state is not old
        assert not codec_adapter.fake.sources[-1].closed


def test_uncached_state_is_disposed_on_close(codec_adapter):
    source = io.BytesIO(b"video")
    lease = codec_adapter.adapter(source)
    state = lease._state
    lease.close()
    lease.close()
    assert state.disposed and state.decoder is None
    assert not source.closed
    if codec_adapter.backend == "tensorcodec":
        assert codec_adapter.fake.closed == 1


def test_internal_typeerror_is_not_reported_as_missing_fps(codec_adapter, monkeypatch):
    with codec_adapter.adapter("video.mp4") as lease:

        def fail(**kwargs):
            raise TypeError("invalid internal conversion")

        monkeypatch.setattr(lease._state.decoder, "get_frames_played_in_range", fail)
        with pytest.raises(TypeError, match="internal conversion"):
            lease.get_frames_played_in_range(0, 1, fps=2)
