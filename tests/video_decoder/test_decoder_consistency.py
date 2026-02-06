"""Tests for decoder consistency between PyAVVideoDecoder and TorchCodecVideoDecoder.

These tests verify that both decoders produce consistent outputs for the same video.
Due to differences in codec implementations, pixel values may differ slightly
(typically by 1-2 values). Tests allow for small tolerances where appropriate.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pytest_subtests import SubTests


def _torchcodec_available() -> bool:
    """Check if TorchCodec is available and functional."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except (ImportError, RuntimeError, OSError):
        # ImportError: torchcodec not installed
        # RuntimeError: FFmpeg libraries not found
        # OSError: shared library loading issues
        return False


def _compare_decoder_outputs(
    video_path: str,
    timestamps: list[float],
    max_pixel_diff: int = 3,
) -> tuple[bool, str, dict]:
    """Compare outputs from PyAV and TorchCodec decoders.

    Returns:
        Tuple of (success, error_message, stats_dict)
    """
    from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

    with PyAVVideoDecoder(video_path) as pyav_decoder:
        pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

    with TorchCodecVideoDecoder(video_path) as torchcodec_decoder:
        torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

    stats = {
        "num_frames": len(timestamps),
        "pyav_shape": pyav_batch.data.shape,
        "torchcodec_shape": torchcodec_batch.data.shape,
    }

    # Check shapes match
    if pyav_batch.data.shape != torchcodec_batch.data.shape:
        return False, f"Shape mismatch: {pyav_batch.data.shape} vs {torchcodec_batch.data.shape}", stats

    # Check PTS values match
    pts_diff = np.max(np.abs(pyav_batch.pts_seconds - torchcodec_batch.pts_seconds))
    stats["max_pts_diff"] = pts_diff
    if pts_diff > 0.01:  # 10ms tolerance
        return False, f"PTS mismatch: max diff = {pts_diff:.4f}s", stats

    # Check pixel values
    max_diffs = []
    for i in range(len(timestamps)):
        diff = np.max(np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int)))
        max_diffs.append(diff)

    stats["max_pixel_diffs"] = max_diffs
    stats["overall_max_pixel_diff"] = max(max_diffs) if max_diffs else 0

    if stats["overall_max_pixel_diff"] > max_pixel_diff:
        return False, f"Pixel diff too large: {stats['overall_max_pixel_diff']} > {max_pixel_diff}", stats

    return True, "", stats


@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistency:
    """Test that PyAVVideoDecoder and TorchCodecVideoDecoder produce consistent outputs."""

    # Maximum allowed pixel difference between decoders (due to codec implementation differences)
    MAX_PIXEL_DIFF = 3

    def test_same_frames_at_exact_pts(self, sample_video_file: tuple[Path, list[int]]):
        """Test that both decoders return consistent frames at exact PTS values."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        # Verify same number of frames
        assert pyav_batch.data.shape == torchcodec_batch.data.shape

        # Verify PTS values are the same
        np.testing.assert_array_almost_equal(
            pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2
        )

        # Verify frame data is consistent (allowing small codec differences)
        for i in range(len(timestamps)):
            max_diff = np.max(
                np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int))
            )
            assert max_diff <= self.MAX_PIXEL_DIFF, (
                f"Frame {i} differs too much between PyAV and TorchCodec: max_diff={max_diff}"
            )

    def test_same_frames_between_pts(self, sample_video_file: tuple[Path, list[int]]):
        """Test that both decoders return same frames for timestamps between frames."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        # Timestamps between frame boundaries (10fps video)
        timestamps = [0.05, 0.15, 0.25, 0.35]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        # Both should return frames at: 0.0, 0.1, 0.2, 0.3
        assert pyav_batch.data.shape == torchcodec_batch.data.shape

        np.testing.assert_array_almost_equal(
            pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2
        )

        for i in range(len(timestamps)):
            max_diff = np.max(
                np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int))
            )
            assert max_diff <= self.MAX_PIXEL_DIFF, (
                f"Frame {i} differs too much between PyAV and TorchCodec: max_diff={max_diff}"
            )

    def test_same_frames_unsorted_timestamps(self, sample_video_file: tuple[Path, list[int]]):
        """Test both decoders handle unsorted timestamps consistently."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        timestamps = [0.3, 0.1, 0.2, 0.0]  # Unsorted

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        assert pyav_batch.data.shape == torchcodec_batch.data.shape

        np.testing.assert_array_almost_equal(
            pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2
        )

        for i in range(len(timestamps)):
            max_diff = np.max(
                np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int))
            )
            assert max_diff <= self.MAX_PIXEL_DIFF, (
                f"Frame {i} differs too much between PyAV and TorchCodec: max_diff={max_diff}"
            )

    def test_same_frames_duplicate_timestamps(self, sample_video_file: tuple[Path, list[int]]):
        """Test both decoders handle duplicate timestamps consistently."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        timestamps = [0.0, 0.0, 0.1, 0.1, 0.2]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        assert pyav_batch.data.shape == torchcodec_batch.data.shape

        np.testing.assert_array_almost_equal(
            pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2
        )

        for i in range(len(timestamps)):
            max_diff = np.max(
                np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int))
            )

    def test_same_single_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test both decoders return consistent single frame."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at([0.0])

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at([0.0])

        assert pyav_batch.data.shape == torchcodec_batch.data.shape
        max_diff = np.max(
            np.abs(pyav_batch.data[0].astype(int) - torchcodec_batch.data[0].astype(int))
        )
        assert max_diff <= self.MAX_PIXEL_DIFF, f"Single frame differs too much: max_diff={max_diff}"

    def test_same_last_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test both decoders return consistent last frame."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        # Last frame at pts=0.4, request just before duration end
        timestamp = [0.49]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamp)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamp)

        assert pyav_batch.data.shape == torchcodec_batch.data.shape

        np.testing.assert_array_almost_equal(
            pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2
        )

        max_diff = np.max(
            np.abs(pyav_batch.data[0].astype(int) - torchcodec_batch.data[0].astype(int))
        )
        assert max_diff <= self.MAX_PIXEL_DIFF, f"Last frame differs too much: max_diff={max_diff}"

    def test_timestamp_patterns(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test various timestamp patterns using subtests."""
        video_path, _ = sample_video_file

        # Various timestamp patterns to test
        patterns = [
            ("sequential_start", [0.0, 0.01, 0.02, 0.03, 0.04]),
            ("sequential_mid", [0.1, 0.15, 0.2, 0.25, 0.3]),
            ("sparse", [0.0, 0.1, 0.2, 0.3, 0.4]),
            ("reverse_order", [0.4, 0.3, 0.2, 0.1, 0.0]),
            ("random_order", [0.2, 0.0, 0.4, 0.1, 0.3]),
            ("near_boundaries", [0.099, 0.199, 0.299, 0.399]),
            ("just_after_pts", [0.001, 0.101, 0.201, 0.301]),
            ("many_same_frame", [0.0, 0.001, 0.002, 0.003, 0.004]),
        ]

        for name, timestamps in patterns:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_single_frame_positions(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test single frame extraction at various positions using subtests."""
        video_path, _ = sample_video_file

        # Single timestamp positions to test
        positions = [
            ("first_frame", [0.0]),
            ("very_early", [0.005]),
            ("quarter", [0.125]),
            ("half", [0.25]),
            ("three_quarter", [0.375]),
            ("near_end", [0.48]),
        ]

        for name, timestamps in positions:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_batch_sizes(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test different batch sizes using subtests."""
        video_path, _ = sample_video_file

        # Different batch sizes with evenly distributed timestamps
        batch_configs = [
            ("batch_1", 1),
            ("batch_2", 2),
            ("batch_5", 5),
            ("batch_10", 10),
            ("batch_15", 15),
        ]

        for name, batch_size in batch_configs:
            # Generate evenly distributed timestamps
            timestamps = [i * 0.4 / batch_size for i in range(batch_size)]
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_edge_case_timestamps(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test edge case timestamp values using subtests."""
        video_path, _ = sample_video_file

        edge_cases = [
            ("zero", [0.0]),
            ("tiny_epsilon", [1e-9]),
            ("small_epsilon", [1e-6]),
            ("near_frame_boundary_low", [0.0999]),
            ("near_frame_boundary_high", [0.1001]),
            ("mixed_precision", [0.0, 0.100000001, 0.2, 0.299999999]),
        ]

        for name, timestamps in edge_cases:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"



@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistencyAdvanced:
    """Advanced decoder consistency tests with more comprehensive coverage."""

    MAX_PIXEL_DIFF = 3

    def test_random_timestamp_sampling(
        self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"
    ):
        """Test random timestamp sampling with fixed seeds for reproducibility."""
        import random

        video_path, _ = sample_video_file

        # Use fixed seeds for reproducibility
        sample_configs = [
            ("seed_42_5samples", 42, 5),
            ("seed_123_5samples", 123, 5),
            ("seed_456_8samples", 456, 8),
            ("seed_789_3samples", 789, 3),
        ]

        for name, seed, num_samples in sample_configs:
            rng = random.Random(seed)
            # Video is 0.5s, use 0.0-0.45 range to stay within bounds
            timestamps = sorted([rng.uniform(0, 0.45) for _ in range(num_samples)])
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_interleaved_timestamps(
        self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"
    ):
        """Test interleaved timestamp patterns using subtests."""
        video_path, _ = sample_video_file

        patterns = [
            ("interleave_2", [0.0, 0.2, 0.1, 0.3, 0.05, 0.25]),
            ("zigzag", [0.0, 0.4, 0.1, 0.3, 0.2]),
            ("clustered_start", [0.0, 0.01, 0.02, 0.2, 0.21, 0.22]),
            ("clustered_end", [0.0, 0.1, 0.35, 0.36, 0.37, 0.38]),
        ]

        for name, timestamps in patterns:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_many_duplicates(
        self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"
    ):
        """Test many duplicate timestamps using subtests."""
        video_path, _ = sample_video_file

        patterns = [
            ("all_same", [0.1] * 5),
            ("pairs", [0.0, 0.0, 0.2, 0.2, 0.4, 0.4]),
            ("triples", [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]),
            ("mixed_dups", [0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3]),
        ]

        for name, timestamps in patterns:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"

    def test_dense_sampling(
        self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"
    ):
        """Test dense timestamp sampling at different video regions."""
        video_path, _ = sample_video_file

        regions = [
            ("dense_start", [i * 0.01 for i in range(10)]),  # 0.0-0.09
            ("dense_mid", [0.2 + i * 0.01 for i in range(10)]),  # 0.2-0.29
            ("dense_end", [0.35 + i * 0.01 for i in range(10)]),  # 0.35-0.44
        ]

        for name, timestamps in regions:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(
                    str(video_path), timestamps, self.MAX_PIXEL_DIFF
                )
                assert success, f"{name}: {error} (stats: {stats})"
