"""Tests for internal utility functions.

These tests verify internal implementation details not exposed by the public API:
- Internal BGRA format handling (load_image_as_bgra)
- Video frame caching behavior (keep_av_open parameter)
- Color conversion correctness (BGRA <-> RGB)
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mediaref._internal import load_image_as_bgra
from mediaref.data_uri import DataURI


class TestInternalBGRAHandling:
    """Test internal BGRA format handling.

    These tests verify that load_image_as_bgra() correctly handles different
    image formats and maintains data integrity through encode/decode cycles.
    """

    def test_load_png_as_bgra_lossless(self, sample_bgra_array: np.ndarray):
        """Test that PNG encoding/decoding via load_image_as_bgra is lossless."""
        # Create data URI from BGRA array
        rgb_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="png").uri

        # Decode using internal function
        decoded_bgra = load_image_as_bgra(data_uri)

        # Verify lossless roundtrip
        assert decoded_bgra.shape == sample_bgra_array.shape
        assert decoded_bgra.dtype == sample_bgra_array.dtype
        np.testing.assert_array_equal(decoded_bgra, sample_bgra_array)

    def test_load_bmp_as_bgra_lossless(self, sample_bgra_array: np.ndarray):
        """Test that BMP encoding/decoding via load_image_as_bgra is lossless."""
        # Create data URI from BGRA array
        rgb_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="bmp").uri

        # Decode using internal function
        decoded_bgra = load_image_as_bgra(data_uri)

        # Verify lossless roundtrip
        assert decoded_bgra.shape == sample_bgra_array.shape
        assert decoded_bgra.dtype == sample_bgra_array.dtype
        np.testing.assert_array_equal(decoded_bgra, sample_bgra_array)

    def test_load_jpeg_as_bgra_lossy(self, sample_bgra_array: np.ndarray):
        """Test that JPEG encoding/decoding via load_image_as_bgra handles lossy compression."""
        # Create data URI from BGRA array
        rgb_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="jpeg", quality=85).uri

        # Decode using internal function
        decoded_bgra = load_image_as_bgra(data_uri)

        # Verify shape and dtype
        assert decoded_bgra.shape == sample_bgra_array.shape
        assert decoded_bgra.dtype == sample_bgra_array.dtype

        # JPEG is lossy - verify arrays are similar but not identical
        assert not np.array_equal(decoded_bgra, sample_bgra_array)
        assert np.abs(decoded_bgra.astype(float) - sample_bgra_array.astype(float)).mean() < 50


@pytest.mark.video
class TestVideoFrameCaching:
    """Test video frame caching with keep_av_open parameter.

    This tests internal caching behavior not exposed by the public API.
    """

    def test_keep_av_open_caches_container(self, sample_video_file: tuple[Path, list[int]]):
        """Test that keep_av_open=True caches video containers and increments refs."""
        from mediaref import cached_av
        from mediaref._internal import load_video_frame_as_bgra

        video_path, timestamps = sample_video_file
        cache_key = str(video_path)

        # Clear cache first
        cached_av.cleanup_cache()
        assert cache_key not in cached_av._container_cache

        # First load with keep_av_open=True (should add to cache with refs=1)
        bgra1 = load_video_frame_as_bgra(str(video_path), timestamps[1], keep_av_open=True)
        assert cache_key in cached_av._container_cache
        assert cached_av._container_cache[cache_key].refs == 1

        # Second load (should reuse cached container and increment refs to 2)
        bgra2 = load_video_frame_as_bgra(str(video_path), timestamps[2], keep_av_open=True)
        assert cache_key in cached_av._container_cache
        assert cached_av._container_cache[cache_key].refs == 2

        # Results should be valid BGRA arrays
        assert bgra1.shape[2] == 4  # BGRA
        assert bgra2.shape[2] == 4  # BGRA
        assert bgra1.dtype == np.uint8
        assert bgra2.dtype == np.uint8

    def test_keep_av_open_false_does_not_cache(self, sample_video_file: tuple[Path, list[int]]):
        """Test that keep_av_open=False does not cache containers."""
        from mediaref import cached_av
        from mediaref._internal import load_video_frame_as_bgra

        video_path, timestamps = sample_video_file
        cache_key = str(video_path)

        # Clear cache first
        cached_av.cleanup_cache()
        assert cache_key not in cached_av._container_cache

        # Load with keep_av_open=False (should NOT cache)
        bgra = load_video_frame_as_bgra(str(video_path), timestamps[1], keep_av_open=False)

        # Verify container is NOT in cache
        assert cache_key not in cached_av._container_cache
        assert bgra.shape[2] == 4  # BGRA


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
