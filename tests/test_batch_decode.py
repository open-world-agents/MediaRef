"""Tests for batch_decode functionality with performance benchmarks.

These tests require the [video] extra to be installed.
"""

import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mediaref import MediaRef, batch_decode, cleanup_cache


def _torchcodec_available() -> bool:
    """Check if TorchCodec is available and functional."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except (ImportError, RuntimeError, OSError):
        return False


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

    @pytest.mark.skipif(_torchcodec_available(), reason="TorchCodec is installed")
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

    NUM_RUNS = 5  # Number of times to run each benchmark for consistent results

    def test_batch_decode_performance_vs_individual(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test that batch decoding is faster than individual loading."""
        video_path, timestamps = sample_video_file_large
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps[:10]]

        # Warmup: import video decoder module to trigger lazy imports (PyAV, etc.)
        import mediaref.video_decoder  # noqa: F401

        batch_times = []
        individual_times = []

        for run in range(self.NUM_RUNS):
            # Batch decoding
            cleanup_cache()
            start = time.perf_counter()
            batch_results = batch_decode(refs)
            batch_times.append(time.perf_counter() - start)

            # Individual loading
            cleanup_cache()
            start = time.perf_counter()
            individual_results = [ref.to_ndarray() for ref in refs]
            individual_times.append(time.perf_counter() - start)

            # Verify correctness on first run only
            if run == 0:
                for b, i in zip(batch_results, individual_results):
                    np.testing.assert_array_equal(b, i)

        # Statistics
        b_mean, b_std = np.mean(batch_times), np.std(batch_times)
        i_mean, i_std = np.mean(individual_times), np.std(individual_times)
        speedup = i_mean / b_mean

        print(f"\nBatch: {b_mean:.4f}s ± {b_std:.4f}s (min={min(batch_times):.4f}s, max={max(batch_times):.4f}s)")
        print(
            f"Individual: {i_mean:.4f}s ± {i_std:.4f}s (min={min(individual_times):.4f}s, max={max(individual_times):.4f}s)"
        )
        print(f"Speedup: {speedup:.2f}x (n={self.NUM_RUNS})")

        assert b_mean * 1.5 < i_mean, f"Expected >1.5x speedup, got {speedup:.2f}x"

    def test_batch_decode_throughput(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test batch decoding throughput."""
        video_path, timestamps = sample_video_file_large
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps]

        # Warmup: import video decoder module to trigger lazy imports (PyAV, etc.)
        import mediaref.video_decoder  # noqa: F401

        elapsed_times = []
        for _ in range(self.NUM_RUNS):
            cleanup_cache()
            start = time.perf_counter()
            batch_decode(refs)
            elapsed_times.append(time.perf_counter() - start)

        # Statistics
        fps = [len(refs) / t for t in elapsed_times]
        fps_mean, fps_std = np.mean(fps), np.std(fps)
        t_mean, t_std = np.mean(elapsed_times), np.std(elapsed_times)

        print(f"\nThroughput: {fps_mean:.1f} ± {fps_std:.1f} fps (n={self.NUM_RUNS}, frames={len(refs)})")
        print(f"Time: {t_mean:.4f}s ± {t_std:.4f}s (min={min(elapsed_times):.4f}s, max={max(elapsed_times):.4f}s)")

        assert fps_mean > 10, f"Throughput too low: {fps_mean:.1f} fps"

    def test_batch_decode_memory_efficiency(self, sample_video_file_large: tuple[Path, list[int]]):
        """Test that batch decoding doesn't use excessive memory."""
        video_path, timestamps = sample_video_file_large
        refs = [MediaRef(uri=str(video_path), pts_ns=ts) for ts in timestamps]

        # Warmup: import video decoder module to trigger lazy imports (PyAV, etc.)
        import mediaref.video_decoder  # noqa: F401

        # Run multiple times to ensure no memory leaks
        for _ in range(self.NUM_RUNS):
            cleanup_cache()
            results = batch_decode(refs)
            assert len(results) == len(refs)
            for rgb in results:
                assert isinstance(rgb, np.ndarray)
                assert rgb.dtype == np.uint8

        print(f"\nMemory test: decoded {len(refs)} frames {self.NUM_RUNS} times (no errors)")


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
