"""Tests for internal utility functions.

These tests verify internal implementation details not exposed by the public API:
- Internal RGBA format handling (load_image_as_rgba)
- Video frame caching behavior (keep_av_open parameter)
- Color conversion correctness (RGBA <-> RGB)
"""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mediaref._internal import load_image_as_rgba
from mediaref.data_uri import DataURI


class TestInternalRGBAHandling:
    """Test internal RGBA format handling.

    These tests verify that load_image_as_rgba() correctly handles different
    image formats and maintains data integrity through encode/decode cycles.
    """

    def test_load_png_as_rgba_lossless(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that PNG encoding/decoding via load_image_as_rgba is lossless."""
        # Create data URI from RGBA array
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="png").uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify lossless roundtrip
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype
        np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)

    def test_load_bmp_as_rgba_lossless(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that BMP encoding/decoding via load_image_as_rgba is lossless."""
        # Create data URI from RGBA array
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="bmp").uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify lossless roundtrip
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype
        np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)

    def test_load_jpeg_as_rgba_lossy(self, sample_rgba_array: npt.NDArray[np.uint8]):
        """Test that JPEG encoding/decoding via load_image_as_rgba handles lossy compression."""
        # Create data URI from RGBA array
        rgb_array = cv2.cvtColor(sample_rgba_array, cv2.COLOR_RGBA2RGB)
        data_uri = DataURI.from_image(rgb_array, format="jpeg", quality=85).uri

        # Decode using internal function
        decoded_rgba = load_image_as_rgba(data_uri)

        # Verify shape and dtype
        assert decoded_rgba.shape == sample_rgba_array.shape
        assert decoded_rgba.dtype == sample_rgba_array.dtype

        # JPEG is lossy - verify arrays are similar but not identical
        assert not np.array_equal(decoded_rgba, sample_rgba_array)
        assert np.abs(decoded_rgba.astype(float) - sample_rgba_array.astype(float)).mean() < 50


@pytest.mark.video
class TestVideoFrameCaching:
    """Test video frame caching with keep_av_open parameter.

    This tests internal caching behavior not exposed by the public API.
    """

    def test_keep_av_open_caches_container(self, sample_video_file: tuple[Path, list[int]]):
        """Test that keep_av_open=True caches video containers and increments refs."""
        from mediaref import cached_av
        from mediaref._internal import load_video_frame_as_rgba

        video_path, timestamps = sample_video_file
        cache_key = str(video_path)

        # Clear cache first
        cached_av.cleanup_cache()
        assert cache_key not in cached_av._container_cache

        # First load with keep_av_open=True (should add to cache with refs=1)
        rgba1 = load_video_frame_as_rgba(str(video_path), timestamps[1], keep_av_open=True)
        assert cache_key in cached_av._container_cache
        assert cached_av._container_cache[cache_key].refs == 1

        # Second load (should reuse cached container and increment refs to 2)
        rgba2 = load_video_frame_as_rgba(str(video_path), timestamps[2], keep_av_open=True)
        assert cache_key in cached_av._container_cache
        assert cached_av._container_cache[cache_key].refs == 2

        # Results should be valid RGBA arrays
        assert rgba1.shape[2] == 4  # RGBA
        assert rgba2.shape[2] == 4  # RGBA
        assert rgba1.dtype == np.uint8
        assert rgba2.dtype == np.uint8

    def test_keep_av_open_false_does_not_cache(self, sample_video_file: tuple[Path, list[int]]):
        """Test that keep_av_open=False does not cache containers."""
        from mediaref import cached_av
        from mediaref._internal import load_video_frame_as_rgba

        video_path, timestamps = sample_video_file
        cache_key = str(video_path)

        # Clear cache first
        cached_av.cleanup_cache()
        assert cache_key not in cached_av._container_cache

        # Load with keep_av_open=False (should NOT cache)
        rgba = load_video_frame_as_rgba(str(video_path), timestamps[1], keep_av_open=False)

        # Verify container is NOT in cache
        assert cache_key not in cached_av._container_cache
        assert rgba.shape[2] == 4  # RGBA
