"""Tests for batch_decode functionality with performance benchmarks.

These tests require the [loader] extra to be installed.
"""

import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mediaref import MediaRef, batch_decode, cleanup_cache


@pytest.mark.video
class TestBatchDecodeImages:
    """Test batch decoding of images (requires video extra for batch_decode)."""

    def test_batch_decode_single_image(self, sample_image_files: list[Path]):
        """Test batch decoding with single image."""
        refs = [MediaRef(uri=str(sample_image_files[0]))]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)
        assert results[0].shape == (48, 64, 3)

    def test_batch_decode_multiple_images(self, sample_image_files: list[Path]):
        """Test batch decoding with multiple images."""
        refs = [MediaRef(uri=str(img)) for img in sample_image_files]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        assert len(results) == 3
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)
            assert rgb.dtype == np.uint8

    def test_batch_decode_images_different_content(self, sample_image_files: list[Path]):
        """Test that different images have different content."""
        refs = [MediaRef(uri=str(img)) for img in sample_image_files]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        # Images should be different (different intensities)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(results[0], results[1])
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(results[1], results[2])

    def test_batch_decode_empty_list(self):
        """Test batch decoding with empty list."""
        results = batch_decode([])
        assert results == []

    def test_batch_decode_preserves_order(self, sample_image_files: list[Path]):
        """Test that batch_decode preserves input order."""
        refs = [MediaRef(uri=str(img)) for img in sample_image_files]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        # Verify order by checking individual loads
        for ref, result in zip(refs, results):
            individual_result = ref.to_ndarray()
            np.testing.assert_array_equal(result, individual_result)


@pytest.mark.video
class TestBatchDecodeVideo:
    """Test batch decoding of video frames."""

    def test_batch_decode_single_video_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test batch decoding with single video frame."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=timestamps[0])]  # Already in nanoseconds
        results = batch_decode(refs)

        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)
        assert results[0].shape == (48, 64, 3)

    def test_batch_decode_multiple_frames_same_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test batch decoding multiple frames from same video."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds
        results = batch_decode(refs)

        assert len(results) == 3
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)
            assert rgb.dtype == np.uint8

    def test_batch_decode_video_frames_different_content(self, sample_video_file: tuple[Path, list[int]]):
        """Test that different video frames have different content."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds
        results = batch_decode(refs)

        # Frames should be different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(results[0], results[1])
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(results[1], results[2])

    def test_batch_decode_preserves_order_video(self, sample_video_file: tuple[Path, list[int]]):
        """Test that batch_decode preserves order for video frames."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds
        results = batch_decode(refs)

        # Verify order by checking individual loads
        for ref, result in zip(refs, results):
            individual_result = ref.to_ndarray()
            np.testing.assert_array_equal(result, individual_result)


@pytest.mark.video
class TestBatchDecodeMixed:
    """Test batch decoding with mixed images and videos."""

    def test_batch_decode_mixed_images_and_videos(
        self, sample_image_files: list[Path], sample_video_file: tuple[Path, list[int]]
    ):
        """Test batch decoding with mixed images and videos."""
        video_path, timestamps = sample_video_file

        refs = [
            MediaRef(uri=str(sample_image_files[0])),
            MediaRef(uri=str(video_path), pts_ns=timestamps[0]),  # Already in nanoseconds
            MediaRef(uri=str(sample_image_files[1])),
            MediaRef(uri=str(video_path), pts_ns=timestamps[1]),  # Already in nanoseconds
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        assert len(results) == 4
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)

    def test_batch_decode_mixed_preserves_order(
        self, sample_image_files: list[Path], sample_video_file: tuple[Path, list[int]]
    ):
        """Test that mixed batch decoding preserves order."""
        video_path, timestamps = sample_video_file

        refs = [
            MediaRef(uri=str(sample_image_files[0])),
            MediaRef(uri=str(video_path), pts_ns=timestamps[0]),  # Already in nanoseconds
            MediaRef(uri=str(sample_image_files[1])),
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
            results = batch_decode(refs)

        # Verify order
        for ref, result in zip(refs, results):
            individual_result = ref.to_ndarray()
            np.testing.assert_array_equal(result, individual_result)


@pytest.mark.video
class TestBatchDecodeDecoders:
    """Test different decoder backends."""

    def test_batch_decode_pyav_decoder(self, sample_video_file: tuple[Path, list[int]]):
        """Test batch decoding with PyAV decoder."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds
        results = batch_decode(refs, decoder="pyav")

        assert len(results) == 3
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.shape == (48, 64, 3)

    def test_batch_decode_invalid_decoder(self, sample_video_file: tuple[Path, list[int]]):
        """Test that invalid decoder raises ValueError."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=timestamps[0])]  # Already in nanoseconds

        with pytest.raises(ValueError, match="Unknown decoder backend"):
            batch_decode(refs, decoder="invalid")  # type: ignore

    def test_batch_decode_torchcodec_not_installed(self, sample_video_file: tuple[Path, list[int]]):
        """Test that TorchCodec decoder raises ImportError when not installed."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=timestamps[0])]  # Already in nanoseconds

        # TorchCodec is not installed in test environment
        with pytest.raises(ImportError, match="TorchCodec.*not.*install"):
            batch_decode(refs, decoder="torchcodec")

    def test_batch_decode_without_video_extra_shows_helpful_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that batch_decode shows helpful error when [video] extra is not installed.

        This test simulates the scenario where someone tries to use batch_decode
        without installing the [video] extra.
        """
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=timestamps[0])]

        # Mock HAS_VIDEO to simulate [video] extra not being installed
        with patch("mediaref._features.HAS_VIDEO", False):
            with patch("mediaref._features.VIDEO_ERROR", "No module named 'av'"):
                # Clear the module cache to force re-import with mocked values
                if "mediaref.video_decoder" in sys.modules:
                    del sys.modules["mediaref.video_decoder"]
                if "mediaref.video_decoder.pyav_decoder" in sys.modules:
                    del sys.modules["mediaref.video_decoder.pyav_decoder"]

                # Now trying to use batch_decode should raise ImportError with helpful message
                with pytest.raises(ImportError, match="Video frame extraction requires.*video.*extra"):
                    batch_decode(refs, decoder="pyav")


class TestBatchDecodeCache:
    """Test cache cleanup functionality."""

    @pytest.mark.video
    def test_cleanup_cache(self, sample_video_file: tuple[Path, list[int]]):
        """Test that cleanup_cache doesn't raise errors."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds

        # Load some frames
        batch_decode(refs)

        # Cleanup should not raise
        cleanup_cache()

    @pytest.mark.video
    def test_cleanup_cache_multiple_times(self, sample_video_file: tuple[Path, list[int]]):
        """Test that cleanup_cache can be called multiple times."""
        cleanup_cache()
        cleanup_cache()
        cleanup_cache()

    @pytest.mark.video
    def test_batch_decode_after_cleanup(self, sample_video_file: tuple[Path, list[int]]):
        """Test that batch_decode works after cache cleanup."""
        video_path, timestamps = sample_video_file
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:3]]  # Already in nanoseconds

        # Load, cleanup, load again
        results1 = batch_decode(refs)
        cleanup_cache()
        results2 = batch_decode(refs)

        # Results should be the same
        for r1, r2 in zip(results1, results2):
            np.testing.assert_array_equal(r1, r2)


@pytest.mark.performance
@pytest.mark.video
class TestBatchDecodePerformance:
    """Performance benchmarks for batch decoding."""

    def test_batch_decode_performance_vs_individual(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test that batch decoding is faster than individual loading."""
        video_path, timestamps = sample_video_file_large
        # Use 10 frames for performance test (timestamps already in nanoseconds)
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:10]]

        # Measure batch decoding time
        start_batch = time.perf_counter()
        batch_results = batch_decode(refs)
        batch_time = time.perf_counter() - start_batch

        # Measure individual loading time
        cleanup_cache()  # Clear cache to ensure fair comparison
        start_individual = time.perf_counter()
        individual_results = [ref.to_ndarray() for ref in refs]
        individual_time = time.perf_counter() - start_individual

        # Verify results are the same
        for batch_result, individual_result in zip(batch_results, individual_results):
            np.testing.assert_array_equal(batch_result, individual_result)

        # Batch should be faster (at least 20% faster)
        print(f"\nBatch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
        print(f"Speedup: {individual_time / batch_time:.2f}x")
        assert batch_time < individual_time * 0.8, "Batch decoding should be at least 20% faster"

    def test_batch_decode_throughput(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test batch decoding throughput."""
        video_path, timestamps = sample_video_file_large
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps]  # Already in nanoseconds

        start = time.perf_counter()
        results = batch_decode(refs)
        elapsed = time.perf_counter() - start

        fps = len(results) / elapsed
        print(f"\nBatch decode throughput: {fps:.2f} frames/second")
        print(f"Total frames: {len(results)}, Time: {elapsed:.4f}s")

        # Should be able to decode at least 10 fps
        assert fps > 10, f"Throughput too low: {fps:.2f} fps"

    def test_batch_decode_memory_efficiency(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test that batch decoding doesn't use excessive memory."""
        video_path, timestamps = sample_video_file_large
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps]  # Already in nanoseconds

        # This test just ensures batch_decode completes without memory errors
        results = batch_decode(refs)

        assert len(results) == len(refs)
        # Verify all results are valid
        for rgb in results:
            assert isinstance(rgb, np.ndarray)
            assert rgb.dtype == np.uint8


@pytest.mark.video
class TestBatchDecodeErrorHandling:
    """Test error handling in batch decoding."""

    def test_batch_decode_with_empty_list(self):
        """Test that batch_decode with empty list returns empty list."""
        result = batch_decode([])
        assert result == []

    def test_batch_decode_with_nonexistent_video(self):
        """Test that batch_decode with nonexistent video raises error."""
        refs = [MediaRef(uri="/nonexistent/video.mp4", pts_ns=0)]

        with pytest.raises(Exception):  # Should raise ValueError or FileNotFoundError
            batch_decode(refs)

    def test_batch_decode_mixed_with_error(self, sample_image_files: list[Path]):
        """Test batch_decode behavior when one item fails."""
        refs = [
            MediaRef(uri=str(sample_image_files[0])),
            MediaRef(uri="/nonexistent/image.png"),  # This will fail
        ]

        with pytest.raises(Exception):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="batch_decode.*received.*image")
                batch_decode(refs)
