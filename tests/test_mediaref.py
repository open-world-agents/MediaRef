"""Tests for MediaRef class."""

from pathlib import Path

import numpy as np
import pytest
from mediaref import MediaRef, cleanup_cache, load_batch


class TestMediaRefProperties:
    """Test MediaRef properties."""

    def test_is_embedded(self):
        """Test is_embedded property."""
        ref = MediaRef(uri="data:image/png;base64,iVBORw0KG...")
        assert ref.is_embedded

        ref = MediaRef(uri="image.png")
        assert not ref.is_embedded

    def test_is_video(self):
        """Test is_video property."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        assert ref.is_video

        ref = MediaRef(uri="image.png")
        assert not ref.is_video

    def test_is_remote(self):
        """Test is_remote property."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        assert ref.is_remote

        ref = MediaRef(uri="http://example.com/image.jpg")
        assert ref.is_remote

        ref = MediaRef(uri="image.png")
        assert not ref.is_remote

    def test_is_local(self):
        """Test is_local property."""
        ref = MediaRef(uri="/path/to/image.png")
        assert ref.is_local

        ref = MediaRef(uri="https://example.com/image.jpg")
        assert not ref.is_local

        ref = MediaRef(uri="data:image/png;base64,...")
        assert not ref.is_local

    def test_is_relative_path(self):
        """Test is_relative_path property."""
        ref = MediaRef(uri="relative/path/image.png")
        assert ref.is_relative_path

        ref = MediaRef(uri="/absolute/path/image.png")
        assert not ref.is_relative_path

        ref = MediaRef(uri="file:///path/image.png")
        assert not ref.is_relative_path

        ref = MediaRef(uri="https://example.com/image.jpg")
        assert not ref.is_relative_path


class TestMediaRefPathResolution:
    """Test path resolution."""

    def test_resolve_relative_path_with_mcap(self):
        """Test resolving relative path against MCAP file."""
        ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
        resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "/data/relative/video.mkv"
        assert resolved.pts_ns == 123456

    def test_resolve_relative_path_with_directory(self):
        """Test resolving relative path against directory."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path("/data/dataset")

        assert resolved.uri == "/data/dataset/images/test.jpg"

    def test_resolve_absolute_path_unchanged(self):
        """Test that absolute paths remain unchanged."""
        ref = MediaRef(uri="/absolute/path/image.png")
        resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "/absolute/path/image.png"

    def test_resolve_remote_path_with_warning(self):
        """Test that remote paths generate warning."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        with pytest.warns(UserWarning, match="Cannot resolve non-local path"):
            resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "https://example.com/image.jpg"

    def test_resolve_embedded_path_with_warning(self):
        """Test that embedded paths generate warning."""
        ref = MediaRef(uri="data:image/png;base64,...")

        with pytest.warns(UserWarning, match="Cannot resolve non-local path"):
            resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "data:image/png;base64,..."


class TestMediaRefValidation:
    """Test URI validation."""

    def test_validate_embedded_uri(self):
        """Test validating embedded URI."""
        ref = MediaRef(uri="data:image/png;base64,...")
        assert ref.validate_uri()

    def test_validate_local_file_exists(self, tmp_path):
        """Test validating existing local file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        ref = MediaRef(uri=str(test_file))
        assert ref.validate_uri()

    def test_validate_local_file_not_exists(self, tmp_path):
        """Test validating non-existent local file."""
        ref = MediaRef(uri=str(tmp_path / "nonexistent.txt"))
        assert not ref.validate_uri()

    def test_validate_remote_uri_not_implemented(self):
        """Test that remote URI validation raises NotImplementedError."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        with pytest.raises(NotImplementedError):
            ref.validate_uri()


class TestMediaRefSerialization:
    """Test serialization."""

    def test_model_dump(self):
        """Test model_dump."""
        ref = MediaRef(uri="image.png", pts_ns=123456)
        data = ref.model_dump()

        assert data == {"uri": "image.png", "pts_ns": 123456}

    def test_model_dump_json(self):
        """Test model_dump_json."""
        ref = MediaRef(uri="image.png")
        json_str = ref.model_dump_json()

        assert "image.png" in json_str

    def test_model_validate(self):
        """Test model_validate."""
        data = {"uri": "image.png", "pts_ns": 123456}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "image.png"
        assert ref.pts_ns == 123456

    def test_model_validate_json(self):
        """Test model_validate_json."""
        json_str = '{"uri": "image.png", "pts_ns": 123456}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "image.png"
        assert ref.pts_ns == 123456


class TestMediaRefLoading:
    """Test loading methods (requires test files)."""

    @pytest.fixture
    def sample_image_file(self, tmp_path):
        """Create a sample image file."""
        import cv2

        image_path = tmp_path / "test_image.png"
        test_image = np.zeros((48, 64, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel
        cv2.imwrite(str(image_path), test_image)

        return image_path

    def test_to_rgb_array(self, sample_image_file):
        """Test to_rgb_array method."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_pil_image(self, sample_image_file):
        """Test to_pil_image method."""
        from PIL import Image

        ref = MediaRef(uri=str(sample_image_file))
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)  # PIL uses (width, height)

    def test_embed_as_data_uri(self, sample_image_file):
        """Test embed_as_data_uri method."""
        ref = MediaRef(uri=str(sample_image_file))
        data_uri = ref.embed_as_data_uri(format="png")

        assert data_uri.startswith("data:image/png;base64,")

        # Create new ref from data URI
        embedded_ref = MediaRef(uri=data_uri)
        assert embedded_ref.is_embedded

        # Should be able to load from embedded ref
        rgb = embedded_ref.to_rgb_array()
        assert isinstance(rgb, np.ndarray)


class TestBatchLoading:
    """Test batch loading functionality."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create multiple sample image files."""
        import cv2

        images = []
        for i in range(3):
            image_path = tmp_path / f"test_image_{i}.png"
            test_image = np.full((48, 64, 3), i * 50, dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            images.append(image_path)

        return images

    @pytest.fixture
    def sample_video(self, tmp_path):
        """Create a sample video file."""
        import av

        video_path = tmp_path / "test_video.mp4"

        # Create video with av
        container = av.open(str(video_path), "w")
        stream = container.add_stream("h264", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        # Write frames
        for i in range(10):
            frame = av.VideoFrame(64, 48, "rgb24")
            arr = np.full((48, 64, 3), i * 25, dtype=np.uint8)
            frame.planes[0].update(arr)
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        return video_path

    def test_load_batch_images(self, sample_images):
        """Test batch loading of images."""
        refs = [MediaRef(uri=str(img)) for img in sample_images]
        results = load_batch(refs)

        assert len(results) == 3
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)

    def test_load_batch_video_frames(self, sample_video):
        """Test batch loading of video frames."""
        refs = [
            MediaRef(uri=str(sample_video), pts_ns=0),
            MediaRef(uri=str(sample_video), pts_ns=100_000_000),
            MediaRef(uri=str(sample_video), pts_ns=200_000_000),
        ]
        results = load_batch(refs)

        assert len(results) == 3
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)

    def test_load_batch_mixed(self, sample_images, sample_video):
        """Test batch loading of mixed images and videos."""
        refs = [
            MediaRef(uri=str(sample_images[0])),
            MediaRef(uri=str(sample_video), pts_ns=0),
            MediaRef(uri=str(sample_images[1])),
            MediaRef(uri=str(sample_video), pts_ns=100_000_000),
        ]
        results = load_batch(refs)

        assert len(results) == 4
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)

    def test_load_batch_empty(self):
        """Test batch loading with empty list."""
        results = load_batch([])
        assert results == []

    def test_cleanup_cache(self, sample_video):
        """Test cache cleanup."""
        # Load some frames with caching
        refs = [MediaRef(uri=str(sample_video), pts_ns=i * 100_000_000) for i in range(3)]
        load_batch(refs)

        # Cleanup should not raise
        cleanup_cache()
