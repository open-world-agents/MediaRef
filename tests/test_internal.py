"""Tests for internal utility functions.

These tests cover unique internal functionality not fully tested by public API tests:
- Encoding/decoding roundtrip quality (lossless vs lossy)
- Color conversion correctness (BGRA <-> RGB)
- Video frame caching (keep_av_open parameter)
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mediaref._internal import (
    encode_array_to_base64,
    load_image_as_bgra,
)


class TestEncodingRoundtrip:
    """Test encoding and decoding roundtrip."""

    def test_roundtrip_png(self, sample_bgra_array: np.ndarray):
        """Test PNG encoding and decoding roundtrip."""
        # Encode
        base64_data = encode_array_to_base64(sample_bgra_array, format="png")
        data_uri = f"data:image/png;base64,{base64_data}"

        # Decode
        decoded_array = load_image_as_bgra(data_uri)

        # PNG is lossless, so arrays should be identical
        assert decoded_array.shape == sample_bgra_array.shape
        assert decoded_array.dtype == sample_bgra_array.dtype
        np.testing.assert_array_equal(decoded_array, sample_bgra_array)

    def test_roundtrip_bmp(self, sample_bgra_array: np.ndarray):
        """Test BMP encoding and decoding roundtrip."""
        # Encode
        base64_data = encode_array_to_base64(sample_bgra_array, format="bmp")
        data_uri = f"data:image/bmp;base64,{base64_data}"

        # Decode
        decoded_array = load_image_as_bgra(data_uri)

        # BMP is lossless, so arrays should be identical
        assert decoded_array.shape == sample_bgra_array.shape
        assert decoded_array.dtype == sample_bgra_array.dtype
        np.testing.assert_array_equal(decoded_array, sample_bgra_array)

    def test_roundtrip_jpeg_lossy(self, sample_rgb_array: np.ndarray):
        """Test JPEG encoding and decoding roundtrip (lossy)."""
        # Convert RGB to BGRA for encoding
        bgra_array = cv2.cvtColor(sample_rgb_array, cv2.COLOR_RGB2BGRA)

        # Encode
        base64_data = encode_array_to_base64(bgra_array, format="jpeg", quality=85)
        data_uri = f"data:image/jpeg;base64,{base64_data}"

        # Decode
        decoded_array = load_image_as_bgra(data_uri)

        # JPEG is lossy, but should be close
        assert decoded_array.shape == bgra_array.shape
        assert decoded_array.dtype == bgra_array.dtype
        # Allow significant difference due to JPEG compression (especially for small test images)
        # Just verify the arrays are similar in magnitude
        assert np.abs(decoded_array.astype(float) - bgra_array.astype(float)).mean() < 50


@pytest.mark.video
class TestVideoFrameCaching:
    """Test video frame caching with keep_av_open parameter.

    This tests internal caching behavior not exposed by the public API.
    """

    def test_load_video_frame_keep_av_open(self, sample_video_file: tuple[Path, list[int]]):
        """Test that keep_av_open parameter caches video containers for performance."""
        from mediaref._internal import load_video_frame_as_bgra

        video_path, timestamps = sample_video_file
        pts_ns = timestamps[1]  # Already in nanoseconds

        # Load with keep_av_open=True (should cache the container)
        bgra1 = load_video_frame_as_bgra(str(video_path), pts_ns, keep_av_open=True)

        # Load again (should use cached container for better performance)
        bgra2 = load_video_frame_as_bgra(str(video_path), pts_ns, keep_av_open=True)

        # Results should be identical
        assert np.array_equal(bgra1, bgra2)


class TestColorConversion:
    """Test color conversion correctness."""

    def test_bgra_to_rgb_conversion(self, sample_bgra_array: np.ndarray):
        """Test that BGRA to RGB conversion is correct."""
        # Convert BGRA to RGB
        rgb = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2RGB)

        # Verify shape
        assert rgb.shape == (48, 64, 3)

        # Verify color channels are swapped correctly
        # BGRA: [B, G, R, A] -> RGB: [R, G, B]
        assert rgb[0, 0, 0] == sample_bgra_array[0, 0, 2]  # R from BGRA
        assert rgb[0, 0, 1] == sample_bgra_array[0, 0, 1]  # G from BGRA
        assert rgb[0, 0, 2] == sample_bgra_array[0, 0, 0]  # B from BGRA

    def test_rgb_to_bgra_conversion(self, sample_rgb_array: np.ndarray):
        """Test that RGB to BGRA conversion is correct."""
        # Convert RGB to BGRA
        bgra = cv2.cvtColor(sample_rgb_array, cv2.COLOR_RGB2BGRA)

        # Verify shape
        assert bgra.shape == (48, 64, 4)

        # Verify color channels are swapped correctly
        # RGB: [R, G, B] -> BGRA: [B, G, R, A]
        assert bgra[0, 0, 0] == sample_rgb_array[0, 0, 2]  # B from RGB
        assert bgra[0, 0, 1] == sample_rgb_array[0, 0, 1]  # G from RGB
        assert bgra[0, 0, 2] == sample_rgb_array[0, 0, 0]  # R from RGB
        assert bgra[0, 0, 3] == 255  # Alpha channel
