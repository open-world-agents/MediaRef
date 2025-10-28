"""Tests for MediaRef loading methods (to_rgb_array, to_pil_image, embed_as_data_uri).

These tests require the [loader] extra to be installed.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mediaref import MediaRef


class TestToRgbArrayImage:
    """Test to_rgb_array method for images."""

    def test_to_rgb_array_from_file(self, sample_image_file: Path):
        """Test loading RGB array from image file."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_rgb_array_color_correctness(self, sample_image_file: Path):
        """Test that RGB array has correct color values."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_rgb_array()

        # sample_image_file has BGR = (255, 0, 0) which is blue in BGR format
        # When converted to RGB, it should be (0, 0, 255) - blue in RGB format
        assert rgb[0, 0, 0] == 0  # Red channel
        assert rgb[0, 0, 1] == 0  # Green channel
        assert rgb[0, 0, 2] == 255  # Blue channel

    def test_to_rgb_array_from_data_uri(self, sample_data_uri: str):
        """Test loading RGB array from data URI."""
        ref = MediaRef(uri=sample_data_uri)
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    @pytest.mark.network
    def test_to_rgb_array_from_remote_url(self, remote_test_image_url: str):
        """Test loading RGB array from remote URL."""
        ref = MediaRef(uri=remote_test_image_url)
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert len(rgb.shape) == 3
        assert rgb.shape[2] == 3
        assert rgb.dtype == np.uint8

    def test_to_rgb_array_nonexistent_file(self):
        """Test that loading from nonexistent file raises error."""
        ref = MediaRef(uri="/nonexistent/file.png")

        with pytest.raises(Exception):  # Should raise ValueError or FileNotFoundError
            ref.to_rgb_array()


@pytest.mark.video
class TestToRgbArrayVideo:
    """Test to_rgb_array method for video frames."""

    def test_to_rgb_array_from_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading RGB array from video frame."""
        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Second frame (already in nanoseconds)

        ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_rgb_array_video_first_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading first frame from video."""
        video_path, timestamps = sample_video_file

        ref = MediaRef(uri=str(video_path), pts_ns=0)
        rgb = ref.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)

    def test_to_rgb_array_video_different_frames(self, sample_video_file: tuple[Path, list[int]]):
        """Test that different frames have different content."""
        video_path, timestamps = sample_video_file

        ref1 = MediaRef(uri=str(video_path), pts_ns=timestamps[0])
        ref2 = MediaRef(uri=str(video_path), pts_ns=timestamps[2])

        rgb1 = ref1.to_rgb_array()
        rgb2 = ref2.to_rgb_array()

        # Frames should be different (different intensities)
        assert not np.array_equal(rgb1, rgb2)

    def test_to_rgb_array_video_nonexistent_file(self):
        """Test that loading from nonexistent video raises error."""
        ref = MediaRef(uri="/nonexistent/video.mp4", pts_ns=0)

        with pytest.raises(Exception):  # Should raise FileNotFoundError
            ref.to_rgb_array()

    def test_to_rgb_array_video_without_pts_ns_raises_error(self):
        """Test that loading video without pts_ns raises ImportError."""
        ref = MediaRef(uri="video.mp4")  # No pts_ns

        # Should raise ImportError because it tries to load as image
        # but video files need pts_ns
        with pytest.raises(Exception):
            ref.to_rgb_array()


class TestToPilImage:
    """Test to_pil_image method."""

    def test_to_pil_image_from_file(self, sample_image_file: Path):
        """Test loading PIL Image from file."""
        ref = MediaRef(uri=str(sample_image_file))
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)  # PIL uses (width, height)
        assert pil_img.mode == "RGB"

    def test_to_pil_image_from_data_uri(self, sample_data_uri: str):
        """Test loading PIL Image from data URI."""
        ref = MediaRef(uri=sample_data_uri)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)
        assert pil_img.mode == "RGB"

    @pytest.mark.video
    def test_to_pil_image_from_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test loading PIL Image from video frame."""
        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Already in nanoseconds

        ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)
        assert pil_img.mode == "RGB"

    def test_to_pil_image_matches_to_rgb_array(self, sample_image_file: Path):
        """Test that to_pil_image matches to_rgb_array."""
        ref = MediaRef(uri=str(sample_image_file))

        rgb_array = ref.to_rgb_array()
        pil_img = ref.to_pil_image()
        pil_array = np.array(pil_img)

        assert np.array_equal(rgb_array, pil_array)

    @pytest.mark.network
    def test_to_pil_image_from_remote_url(self, remote_test_image_url: str):
        """Test loading PIL Image from remote URL."""
        ref = MediaRef(uri=remote_test_image_url)
        pil_img = ref.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"


class TestEmbedAsDataUri:
    """Test embed_as_data_uri method."""

    def test_embed_as_data_uri_png(self, sample_image_file: Path):
        """Test embedding as PNG data URI."""
        ref = MediaRef(uri=str(sample_image_file))
        data_uri = ref.embed_as_data_uri(format="png")

        assert data_uri.startswith("data:image/png;base64,")
        assert len(data_uri) > 100  # Should have substantial base64 data

    def test_embed_as_data_uri_jpeg(self, sample_image_file: Path):
        """Test embedding as JPEG data URI."""
        ref = MediaRef(uri=str(sample_image_file))
        data_uri = ref.embed_as_data_uri(format="jpeg")

        assert data_uri.startswith("data:image/jpeg;base64,")
        assert len(data_uri) > 100

    def test_embed_as_data_uri_bmp(self, sample_image_file: Path):
        """Test embedding as BMP data URI."""
        ref = MediaRef(uri=str(sample_image_file))
        data_uri = ref.embed_as_data_uri(format="bmp")

        assert data_uri.startswith("data:image/bmp;base64,")
        assert len(data_uri) > 100

    def test_embed_as_data_uri_jpeg_quality(self, sample_image_file: Path):
        """Test embedding as JPEG with quality parameter."""
        ref = MediaRef(uri=str(sample_image_file))

        high_quality = ref.embed_as_data_uri(format="jpeg", quality=95)
        low_quality = ref.embed_as_data_uri(format="jpeg", quality=10)

        # High quality should generally produce larger files
        assert len(high_quality) >= len(low_quality)

    def test_embed_as_data_uri_can_be_loaded(self, sample_image_file: Path):
        """Test that embedded data URI can be loaded back."""
        ref = MediaRef(uri=str(sample_image_file))
        data_uri = ref.embed_as_data_uri(format="png")

        # Create new ref from data URI
        embedded_ref = MediaRef(uri=data_uri)
        assert embedded_ref.is_embedded

        # Should be able to load from embedded ref
        rgb = embedded_ref.to_rgb_array()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)

    @pytest.mark.video
    def test_embed_as_data_uri_from_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test embedding video frame as data URI."""
        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Already in nanoseconds

        ref = MediaRef(uri=str(video_path), pts_ns=pts_ns)
        data_uri = ref.embed_as_data_uri(format="png")

        assert data_uri.startswith("data:image/png;base64,")

        # Should be loadable
        embedded_ref = MediaRef(uri=data_uri)
        rgb = embedded_ref.to_rgb_array()
        assert rgb.shape == (48, 64, 3)

    def test_embed_as_data_uri_roundtrip_lossless(self, sample_image_file: Path):
        """Test that PNG embedding is lossless."""
        ref = MediaRef(uri=str(sample_image_file))
        original_rgb = ref.to_rgb_array()

        # Embed and load back
        data_uri = ref.embed_as_data_uri(format="png")
        embedded_ref = MediaRef(uri=data_uri)
        restored_rgb = embedded_ref.to_rgb_array()

        # PNG is lossless, so should be identical
        assert np.array_equal(original_rgb, restored_rgb)

    def test_embed_as_data_uri_roundtrip_lossy(self, sample_image_file: Path):
        """Test that JPEG embedding is lossy but close."""
        ref = MediaRef(uri=str(sample_image_file))
        original_rgb = ref.to_rgb_array()

        # Embed and load back
        data_uri = ref.embed_as_data_uri(format="jpeg", quality=85)
        embedded_ref = MediaRef(uri=data_uri)
        restored_rgb = embedded_ref.to_rgb_array()

        # JPEG is lossy, but should be close
        assert original_rgb.shape == restored_rgb.shape
        # Allow some difference due to JPEG compression
        assert np.allclose(original_rgb, restored_rgb, atol=30)


class TestLoadingErrorHandling:
    """Test error handling in loading methods."""

    def test_load_corrupted_image_file(self, tmp_path: Path):
        """Test loading corrupted image file raises error."""
        corrupted_file = tmp_path / "corrupted.png"
        corrupted_file.write_bytes(b"not an image")

        ref = MediaRef(uri=str(corrupted_file))

        with pytest.raises(Exception):  # Should raise ValueError
            ref.to_rgb_array()

    def test_load_invalid_data_uri(self):
        """Test loading invalid data URI raises error."""
        ref = MediaRef(uri="data:image/png;base64,invalid_base64!")

        with pytest.raises(Exception):  # Should raise ValueError
            ref.to_rgb_array()

    def test_load_unsupported_format(self):
        """Test that unsupported format in embed raises error."""
        ref = MediaRef(uri="image.png")

        with pytest.raises(Exception):  # Should raise ValueError
            ref.embed_as_data_uri(format="invalid")  # type: ignore
