"""Tests for MediaRef loading methods (to_rgb_array, to_pil_image) and DataURI.

These tests require the [loader] extra to be installed.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mediaref import DataURI, MediaRef


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


class TestDataURICreation:
    """Test DataURI creation from various sources."""

    def test_from_file(self, sample_image_file: Path):
        """Test creating DataURI from file."""
        data_uri = DataURI.from_file(sample_image_file)

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.is_image
        assert data_uri.uri.startswith("data:image/png;base64,")
        assert len(data_uri) > 0

    def test_from_image_numpy_array(self, sample_image_file: Path):
        """Test creating DataURI from numpy array."""
        rgb = MediaRef(uri=str(sample_image_file)).to_rgb_array()
        data_uri = DataURI.from_image(rgb, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_from_image_pil_image(self, sample_image_file: Path):
        """Test creating DataURI from PIL Image."""
        pil_img = MediaRef(uri=str(sample_image_file)).to_pil_image()
        data_uri = DataURI.from_image(pil_img, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.is_base64
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_from_uri_string(self, sample_image_file: Path):
        """Test parsing DataURI from string."""
        original = DataURI.from_file(sample_image_file)
        parsed = DataURI.from_uri(original.uri)

        assert parsed.mimetype == original.mimetype
        assert parsed.is_base64 == original.is_base64
        assert parsed.data == original.data

    @pytest.mark.video
    def test_from_video_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test creating DataURI from video frame."""
        video_path, timestamps = sample_video_file
        rgb = MediaRef(uri=str(video_path), pts_ns=timestamps[1]).to_rgb_array()
        data_uri = DataURI.from_image(rgb, format="png")

        assert data_uri.mimetype == "image/png"
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_data_stored_as_is_base64(self, sample_image_file: Path):
        """Test that base64 data is stored as-is (base64 string as bytes)."""
        import base64

        data_uri = DataURI.from_file(sample_image_file)

        # data field should store base64 string as bytes
        assert isinstance(data_uri.data, bytes)
        assert data_uri.is_base64

        # data should be the base64 string (as bytes), not decoded
        base64_str = data_uri.data.decode("utf-8")
        # Should be valid base64
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

        # uri should contain the same base64 string
        assert base64_str in data_uri.uri

    def test_data_stored_as_is_non_base64(self):
        """Test that non-base64 data is stored as-is (URL-encoded text as bytes)."""
        # Create a non-base64 data URI with URL-encoded text
        uri_str = "data:text/plain,Hello%20World"
        data_uri = DataURI.from_uri(uri_str)

        # data field should store URL-encoded text as bytes
        assert isinstance(data_uri.data, bytes)
        assert not data_uri.is_base64
        assert data_uri.data == b"Hello%20World"

        # uri should reconstruct correctly
        assert data_uri.uri == uri_str

    def test_quote_validation_accepts_quoted_data(self):
        """Test that properly URL-encoded data is accepted."""
        # Properly quoted data should be accepted
        uri_str = "data:text/plain,Hello%20World%21"
        data_uri = DataURI.from_uri(uri_str)

        assert data_uri.mimetype == "text/plain"
        assert not data_uri.is_base64
        assert data_uri.uri == uri_str

    def test_quote_validation_rejects_unquoted_data(self):
        """Test that unquoted data is rejected."""
        from pydantic import ValidationError

        # Unquoted space should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello World")

        # Unquoted newline should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello\nWorld")

        # Unquoted special characters should be rejected
        with pytest.raises(ValidationError, match="unquoted characters"):
            DataURI(mimetype="text/plain", is_base64=False, data=b"Hello&World")

    def test_base64_data_does_not_need_url_encoding(self):
        """Test that base64 data is accepted without URL encoding validation."""
        import base64

        # Base64 data can contain characters that would need quoting in non-base64
        # This should be accepted because is_base64=True
        base64_data = base64.b64encode(b"Hello World!").decode("utf-8")
        data_uri = DataURI(mimetype="text/plain", is_base64=True, data=base64_data.encode("utf-8"))

        assert data_uri.is_base64
        assert data_uri.decoded_data == b"Hello World!"


class TestDataURIFormats:
    """Test different image formats for DataURI."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> np.ndarray:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_rgb_array()

    def test_format_png(self, sample_rgb: np.ndarray):
        """Test PNG format."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        assert data_uri.mimetype == "image/png"
        assert data_uri.uri.startswith("data:image/png;base64,")

    def test_format_jpeg(self, sample_rgb: np.ndarray):
        """Test JPEG format."""
        data_uri = DataURI.from_image(sample_rgb, format="jpeg")
        assert data_uri.mimetype == "image/jpeg"
        assert data_uri.uri.startswith("data:image/jpeg;base64,")

    def test_format_bmp(self, sample_rgb: np.ndarray):
        """Test BMP format."""
        data_uri = DataURI.from_image(sample_rgb, format="bmp")
        assert data_uri.mimetype == "image/bmp"
        assert data_uri.uri.startswith("data:image/bmp;base64,")

    def test_jpeg_quality_parameter(self, sample_rgb: np.ndarray):
        """Test JPEG quality affects output size."""
        high_quality = DataURI.from_image(sample_rgb, format="jpeg", quality=95)
        low_quality = DataURI.from_image(sample_rgb, format="jpeg", quality=10)

        assert len(high_quality) > len(low_quality)


class TestDataURIConversion:
    """Test DataURI conversion methods."""

    @pytest.fixture
    def sample_data_uri(self, sample_image_file: Path) -> DataURI:
        """Provide sample DataURI."""
        return DataURI.from_file(sample_image_file)

    def test_to_rgb_array(self, sample_data_uri: DataURI):
        """Test converting DataURI to RGB array."""
        rgb = sample_data_uri.to_rgb_array()

        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_pil_image(self, sample_data_uri: DataURI):
        """Test converting DataURI to PIL Image."""
        pil_img = sample_data_uri.to_pil_image()

        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 48)  # PIL uses (width, height)
        assert pil_img.mode == "RGB"

    def test_str_conversion(self, sample_data_uri: DataURI):
        """Test __str__ returns URI string."""
        uri_str = str(sample_data_uri)

        assert uri_str == sample_data_uri.uri
        assert uri_str.startswith("data:image/png;base64,")


class TestDataURIRoundtrip:
    """Test DataURI encoding/decoding roundtrips."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> np.ndarray:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_rgb_array()

    def test_png_lossless_roundtrip(self, sample_rgb: np.ndarray):
        """Test PNG encoding is lossless."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        restored_rgb = data_uri.to_rgb_array()

        assert np.array_equal(sample_rgb, restored_rgb)

    def test_bmp_lossless_roundtrip(self, sample_rgb: np.ndarray):
        """Test BMP encoding is lossless."""
        data_uri = DataURI.from_image(sample_rgb, format="bmp")
        restored_rgb = data_uri.to_rgb_array()

        assert np.array_equal(sample_rgb, restored_rgb)

    def test_jpeg_lossy_roundtrip(self, sample_rgb: np.ndarray):
        """Test JPEG encoding is lossy but close."""
        data_uri = DataURI.from_image(sample_rgb, format="jpeg", quality=85)
        restored_rgb = data_uri.to_rgb_array()

        assert sample_rgb.shape == restored_rgb.shape
        assert np.allclose(sample_rgb, restored_rgb, atol=30)

    def test_uri_string_roundtrip(self, sample_rgb: np.ndarray):
        """Test parsing DataURI from string preserves data."""
        original = DataURI.from_image(sample_rgb, format="png")
        parsed = DataURI.from_uri(original.uri)

        assert parsed.mimetype == original.mimetype
        assert parsed.is_base64 == original.is_base64
        assert np.array_equal(parsed.to_rgb_array(), original.to_rgb_array())


class TestDataURIWithMediaRef:
    """Test DataURI integration with MediaRef."""

    @pytest.fixture
    def sample_rgb(self, sample_image_file: Path) -> np.ndarray:
        """Provide sample RGB array."""
        return MediaRef(uri=str(sample_image_file)).to_rgb_array()

    def test_mediaref_accepts_datauri_object(self, sample_rgb: np.ndarray):
        """Test MediaRef accepts DataURI object directly."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        ref = MediaRef(uri=data_uri)  # type: ignore[arg-type]

        assert ref.is_embedded
        assert np.array_equal(ref.to_rgb_array(), sample_rgb)

    def test_mediaref_accepts_datauri_string(self, sample_rgb: np.ndarray):
        """Test MediaRef accepts DataURI string."""
        data_uri = DataURI.from_image(sample_rgb, format="png")
        ref = MediaRef(uri=str(data_uri))

        assert ref.is_embedded
        assert np.array_equal(ref.to_rgb_array(), sample_rgb)

    @pytest.mark.video
    def test_video_frame_to_datauri(self, sample_video_file: tuple[Path, list[int]]):
        """Test creating DataURI from video frame."""
        video_path, timestamps = sample_video_file
        original_rgb = MediaRef(uri=str(video_path), pts_ns=timestamps[1]).to_rgb_array()

        data_uri = DataURI.from_image(original_rgb, format="png")
        ref = MediaRef(uri=data_uri)  # type: ignore[arg-type]

        assert ref.is_embedded
        assert np.array_equal(ref.to_rgb_array(), original_rgb)


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

    def test_load_unsupported_format(self, sample_image_file: Path):
        """Test that unsupported format in DataURI raises error."""
        ref = MediaRef(uri=str(sample_image_file))
        rgb = ref.to_rgb_array()

        with pytest.raises(Exception):  # Should raise ValueError
            DataURI.from_image(rgb, format="invalid")  # type: ignore
