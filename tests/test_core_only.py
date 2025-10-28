"""Test that core MediaRef works without video dependencies."""

import pytest

# Check if video dependencies are installed
try:
    import av  # noqa: F401

    VIDEO_INSTALLED = True
except ImportError:
    VIDEO_INSTALLED = False


def test_mediaref_import_without_video():
    """MediaRef should be importable without video dependencies."""
    from mediaref import MediaRef

    # Should be able to create MediaRef instances
    ref = MediaRef(uri="image.png")
    assert ref.uri == "image.png"
    assert ref.pts_ns is None

    # Video reference (creation should work, loading will fail without video extra)
    video_ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
    assert video_ref.uri == "video.mp4"
    assert video_ref.pts_ns == 1_000_000_000


def test_mediaref_properties_without_video():
    """MediaRef properties should work without video dependencies."""
    from mediaref import MediaRef

    # Test various URI types
    local_ref = MediaRef(uri="image.png")
    assert local_ref.is_local
    assert not local_ref.is_remote
    assert not local_ref.is_embedded
    assert not local_ref.is_video

    remote_ref = MediaRef(uri="https://example.com/image.jpg")
    assert remote_ref.is_remote
    assert not remote_ref.is_local

    data_ref = MediaRef(uri="data:image/png;base64,iVBORw0KGgo=")
    assert data_ref.is_embedded
    assert not data_ref.is_local

    video_ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
    assert video_ref.is_video


def test_mediaref_path_utilities_without_video():
    """MediaRef path utilities should work without video dependencies."""
    from mediaref import MediaRef

    # Relative path detection
    rel_ref = MediaRef(uri="relative/path.png")
    assert rel_ref.is_relative_path

    abs_ref = MediaRef(uri="/absolute/path.png")
    assert not abs_ref.is_relative_path

    # Path resolution
    rel_ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
    resolved = rel_ref.resolve_relative_path("/data/recording.mcap")
    assert resolved.uri == "/data/relative/video.mkv"
    assert resolved.pts_ns == 123456


@pytest.mark.skipif(
    VIDEO_INSTALLED,
    reason="This test requires core-only installation (no video extra). Run with: uv sync --extra dev && uv run pytest tests/test_core_only.py::test_video_methods_raise_import_error -v",
)
def test_video_methods_raise_import_error():
    """Video loading methods should raise ImportError without video dependencies.

    Note: Run this test with core-only installation:
        uv sync --extra dev  # Install only core + dev dependencies (no video)
        uv run pytest tests/test_core_only.py::test_video_methods_raise_import_error -v
    """
    from mediaref import MediaRef

    # Video frame reference
    ref = MediaRef(uri="tests/data/test_video.mp4", pts_ns=1_000_000_000)

    # These should raise ImportError when video extra is not installed
    with pytest.raises(ImportError, match="video.*extra"):
        ref.to_rgb_array()

    with pytest.raises(ImportError, match="video.*extra"):
        ref.to_pil_image()

    with pytest.raises(ImportError, match="video.*extra"):
        ref.embed_as_data_uri()


def test_pydantic_serialization_without_video():
    """Pydantic serialization should work without video dependencies."""
    from mediaref import MediaRef

    ref = MediaRef(uri="image.png", pts_ns=1_000_000_000)

    # Test model_dump
    data = ref.model_dump()
    assert data == {"uri": "image.png", "pts_ns": 1_000_000_000}

    # Test model_dump_json
    json_str = ref.model_dump_json()
    assert "image.png" in json_str
    assert "1000000000" in json_str

    # Test model_validate
    restored = MediaRef.model_validate(data)
    assert restored.uri == ref.uri
    assert restored.pts_ns == ref.pts_ns
