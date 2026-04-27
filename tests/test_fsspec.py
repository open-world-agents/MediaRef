"""Tests for fsspec-backed cloud-storage URIs.

Uses fsspec's built-in ``memory://`` filesystem so the suite runs fully
offline — no S3, no GCS, no network.
"""

from __future__ import annotations

import io
from fractions import Fraction

import cv2
import fsspec
import numpy as np
import numpy.typing as npt
import pytest

from mediaref import MediaRef, batch_decode
from mediaref._internal import _DIRECT_URI_SCHEMES, is_cloud_uri


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_memory_fs():
    """Wipe the in-process MemoryFileSystem between tests."""
    fs = fsspec.filesystem("memory")
    # Reset the underlying store directly (no public clear()).
    if hasattr(fs, "store"):
        fs.store.clear()
    if hasattr(fs, "pseudo_dirs"):
        fs.pseudo_dirs.clear()
    yield
    if hasattr(fs, "store"):
        fs.store.clear()
    if hasattr(fs, "pseudo_dirs"):
        fs.pseudo_dirs.clear()


def _put_bytes(uri: str, data: bytes) -> None:
    """Write raw bytes to an fsspec URI."""
    with fsspec.open(uri, "wb") as f:
        f.write(data)


def _png_bytes(rgb_image: npt.NDArray[np.uint8]) -> bytes:
    """Encode an RGB array to PNG bytes via cv2."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    assert success
    return buf.tobytes()


# ---------------------------------------------------------------------------
# is_cloud_uri / properties
# ---------------------------------------------------------------------------


class TestIsCloudUri:
    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/key.mp4",
            "gs://bucket/key.png",
            "hf://datasets/foo/bar.mp4",
            "az://container/blob.png",
            "abfs://container/blob.png",
            "ftp://host/path.png",
            "memory://test/file.png",
        ],
    )
    def test_recognized_cloud_schemes(self, uri):
        assert is_cloud_uri(uri) is True

    @pytest.mark.parametrize(
        "uri",
        [
            "image.png",  # bare relative path
            "/absolute/path.mp4",  # absolute POSIX path
            "file:///etc/passwd",  # file:// is NOT a cloud URI
            "http://example.com/x.png",  # http(s) is NOT a cloud URI
            "https://example.com/x.png",
            "data:image/png;base64,iVBOR...",  # data URI
            "C:/Windows/path.mp4",  # Windows-style path (no scheme)
        ],
    )
    def test_non_cloud_schemes(self, uri):
        assert is_cloud_uri(uri) is False

    def test_unknown_scheme_is_cloud_open_set(self):
        # Open-set: any scheme not in _DIRECT_URI_SCHEMES (and not a bare
        # path) routes to fsspec — even unfamiliar / future ones. Whether
        # the call succeeds at runtime is fsspec's responsibility.
        assert is_cloud_uri("custom-scheme://thing") is True
        assert is_cloud_uri("webdav://host/path") is True
        assert is_cloud_uri("gdrive://folder/file.png") is True

    def test_direct_schemes_excluded_from_cloud(self):
        # _DIRECT_URI_SCHEMES is the closed set MediaRef handles itself.
        assert _DIRECT_URI_SCHEMES == frozenset({"http", "https", "file", "data"})


class TestMediaRefCloudUri:
    def test_property_true_for_s3(self):
        assert MediaRef(uri="s3://bucket/clip.mp4", pts_ns=0).is_cloud_uri is True

    def test_property_false_for_local(self):
        assert MediaRef(uri="image.png").is_cloud_uri is False

    def test_property_false_for_http(self):
        assert MediaRef(uri="https://example.com/x.png").is_cloud_uri is False

    def test_is_relative_path_false_for_cloud(self):
        # Cloud URIs are absolute by construction.
        assert MediaRef(uri="s3://bucket/k.png").is_relative_path is False

    def test_validate_uri_raises_for_cloud(self):
        with pytest.raises(NotImplementedError, match="[Cc]loud"):
            MediaRef(uri="s3://bucket/k.png").validate_uri()

    def test_resolve_relative_path_warns_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        with pytest.warns(UserWarning, match="cloud"):
            out = ref.resolve_relative_path("/data")
        assert out is ref

    def test_resolve_relative_path_error_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        with pytest.raises(ValueError, match="cloud"):
            ref.resolve_relative_path("/data", on_unresolvable="error")

    def test_resolve_relative_path_ignore_for_cloud(self):
        ref = MediaRef(uri="s3://bucket/k.png")
        # "ignore" must not warn or raise.
        out = ref.resolve_relative_path("/data", on_unresolvable="ignore")
        assert out is ref


# ---------------------------------------------------------------------------
# Image loading via memory://
# ---------------------------------------------------------------------------


class TestImageLoadingFromMemoryFS:
    def test_load_png_from_memory_uri(self, sample_rgb_array):
        _put_bytes("memory://imgs/red.png", _png_bytes(sample_rgb_array))
        ref = MediaRef(uri="memory://imgs/red.png")
        loaded = ref.to_ndarray(format="rgb")
        assert loaded.shape == sample_rgb_array.shape
        # PNG is lossless: compare exactly.
        np.testing.assert_array_equal(loaded, sample_rgb_array)

    def test_to_pil_image_from_memory_uri(self, sample_rgb_array):
        _put_bytes("memory://imgs/red.png", _png_bytes(sample_rgb_array))
        ref = MediaRef(uri="memory://imgs/red.png")
        pil = ref.to_pil_image()
        assert pil.size == (sample_rgb_array.shape[1], sample_rgb_array.shape[0])

    def test_loading_from_missing_memory_uri_errors(self):
        ref = MediaRef(uri="memory://does/not/exist.png")
        with pytest.raises((ValueError, FileNotFoundError)):
            ref.to_ndarray()


# ---------------------------------------------------------------------------
# Video loading via memory:// (requires PyAV)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_video_bytes() -> tuple[bytes, list[int]]:
    """In-memory 5-frame H.264 MP4. Returns (bytes, list of pts_ns)."""
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=10)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    pts_ns = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000]
    for i in range(5):
        arr = np.full((48, 64, 3), i * 50, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, 10)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue(), pts_ns


@pytest.mark.video
class TestVideoLoadingFromMemoryFS:
    def test_to_ndarray_single_frame_from_memory_uri(self, sample_video_bytes):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)

        ref = MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[2])
        frame = ref.to_ndarray(format="rgb")
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_batch_decode_from_memory_uri(self, sample_video_bytes):
        data, pts_ns_list = sample_video_bytes
        _put_bytes("memory://videos/clip.mp4", data)

        refs = [MediaRef(uri="memory://videos/clip.mp4", pts_ns=t) for t in pts_ns_list]
        frames = batch_decode(refs)
        assert len(frames) == len(refs)
        for f in frames:
            assert f.shape == (48, 64, 3)
            assert f.dtype == np.uint8

    def test_batch_decode_mixed_local_and_cloud(self, sample_video_bytes, tmp_path):
        # Same video served via both memory:// and a local path; batch_decode
        # should handle both groups in one call.
        data, pts_ns_list = sample_video_bytes
        local_path = tmp_path / "clip.mp4"
        local_path.write_bytes(data)
        _put_bytes("memory://videos/clip.mp4", data)

        refs = [
            MediaRef(uri=str(local_path), pts_ns=pts_ns_list[1]),
            MediaRef(uri="memory://videos/clip.mp4", pts_ns=pts_ns_list[3]),
        ]
        frames = batch_decode(refs)
        assert len(frames) == 2
        assert all(f.shape == (48, 64, 3) for f in frames)
