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

    # Check duration values match
    duration_diff = np.max(np.abs(pyav_batch.duration_seconds - torchcodec_batch.duration_seconds))
    stats["max_duration_diff"] = duration_diff
    if duration_diff > 0.01:  # 10ms tolerance
        return False, f"Duration mismatch: max diff = {duration_diff:.4f}s", stats

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
        np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2)

        # Verify frame data is consistent (allowing small codec differences)
        for i in range(len(timestamps)):
            max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int)))
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

        np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2)

        for i in range(len(timestamps)):
            max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int)))
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

        np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2)

        for i in range(len(timestamps)):
            max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int)))
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

        np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2)

        for i in range(len(timestamps)):
            max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - torchcodec_batch.data[i].astype(int)))
            assert max_diff <= self.MAX_PIXEL_DIFF, (
                f"Frame {i} differs too much between PyAV and TorchCodec: max_diff={max_diff}"
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
        max_diff = np.max(np.abs(pyav_batch.data[0].astype(int) - torchcodec_batch.data[0].astype(int)))
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

        np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, torchcodec_batch.pts_seconds, decimal=2)

        max_diff = np.max(np.abs(pyav_batch.data[0].astype(int) - torchcodec_batch.data[0].astype(int)))
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"


@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistencyAdvanced:
    """Advanced decoder consistency tests with more comprehensive coverage."""

    MAX_PIXEL_DIFF = 3

    def test_random_timestamp_sampling(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_interleaved_timestamps(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_many_duplicates(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
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
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_dense_sampling(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test dense timestamp sampling at different video regions."""
        video_path, _ = sample_video_file

        regions = [
            ("dense_start", [i * 0.01 for i in range(10)]),  # 0.0-0.09
            ("dense_mid", [0.2 + i * 0.01 for i in range(10)]),  # 0.2-0.29
            ("dense_end", [0.35 + i * 0.01 for i in range(10)]),  # 0.35-0.44
        ]

        for name, timestamps in regions:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_large_jumps(self, sample_video_file: tuple[Path, list[int]], subtests: "SubTests"):
        """Test large timestamp jumps (start to end, end to start)."""
        video_path, _ = sample_video_file

        patterns = [
            ("start_to_end", [0.0, 0.4]),
            ("end_to_start", [0.4, 0.0]),
            ("jump_forward_backward", [0.0, 0.4, 0.1]),
            ("zigzag_extreme", [0.0, 0.4, 0.1, 0.3, 0.2]),
        ]

        for name, timestamps in patterns:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_sequential_calls_same_decoder(self, sample_video_file: tuple[Path, list[int]]):
        """Test multiple sequential calls on the same decoder instance."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        # Multiple calls with different starting points
        call_sequences = [
            [0.1],
            [0.3],
            [0.0],
            [0.4],
            [0.2],
        ]

        with PyAVVideoDecoder(str(video_path)) as pyav:
            with TorchCodecVideoDecoder(str(video_path)) as tc:
                for timestamps in call_sequences:
                    pyav_batch = pyav.get_frames_played_at(timestamps)
                    tc_batch = tc.get_frames_played_at(timestamps)

                    # Verify PTS match
                    np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, tc_batch.pts_seconds, decimal=2)

                    # Verify pixel values
                    for i in range(len(timestamps)):
                        max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - tc_batch.data[i].astype(int)))
                        assert max_diff <= self.MAX_PIXEL_DIFF, f"Sequential call {timestamps}: max_diff={max_diff}"


@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistencyLongVideo:
    """Decoder consistency tests using longer video (10 seconds)."""

    MAX_PIXEL_DIFF = 3

    def test_various_start_positions(self, sample_video_file_long: tuple[Path, float], subtests: "SubTests"):
        """Test consistency when starting from various positions in a longer video."""
        video_path, duration = sample_video_file_long

        # Test starting from different positions
        start_positions = [
            ("from_start", [0.0, 0.1, 0.2]),
            ("from_1s", [1.0, 1.1, 1.2]),
            ("from_3s", [3.0, 3.1, 3.2]),
            ("from_5s", [5.0, 5.1, 5.2]),
            ("from_8s", [8.0, 8.1, 8.2]),
            ("near_end", [9.5, 9.6, 9.7]),
        ]

        for name, timestamps in start_positions:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_large_jumps_long_video(self, sample_video_file_long: tuple[Path, float], subtests: "SubTests"):
        """Test large timestamp jumps in longer video."""
        video_path, duration = sample_video_file_long

        patterns = [
            ("start_to_end", [0.0, 9.5]),
            ("end_to_start", [9.5, 0.0]),
            ("jump_5s_forward", [0.0, 5.0]),
            ("jump_5s_backward", [5.0, 0.0]),
            ("multiple_large_jumps", [0.0, 9.0, 2.0, 7.0, 4.0]),
        ]

        for name, timestamps in patterns:
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_random_sampling_long_video(self, sample_video_file_long: tuple[Path, float], subtests: "SubTests"):
        """Test random timestamp sampling across longer video."""
        import random

        video_path, duration = sample_video_file_long

        configs = [
            ("random_20_seed42", 42, 20),
            ("random_50_seed123", 123, 50),
            ("random_100_seed456", 456, 100),
        ]

        for name, seed, num_samples in configs:
            rng = random.Random(seed)
            timestamps = sorted([rng.uniform(0, duration - 0.5) for _ in range(num_samples)])
            with subtests.test(msg=name):
                success, error, stats = _compare_decoder_outputs(str(video_path), timestamps, self.MAX_PIXEL_DIFF)
                assert success, f"{name}: {error} (stats: {stats})"

    def test_sequential_calls_long_video(self, sample_video_file_long: tuple[Path, float]):
        """Test multiple sequential calls with different start positions on same decoder."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, duration = sample_video_file_long

        # Simulate real usage: jumping around the video
        call_sequences = [
            [0.5],  # Start near beginning
            [5.0],  # Jump to middle
            [2.0],  # Jump backward
            [8.0],  # Jump to near end
            [1.0],  # Jump backward again
            [9.0],  # Jump to end
            [0.0],  # Jump to start
        ]

        with PyAVVideoDecoder(str(video_path)) as pyav:
            with TorchCodecVideoDecoder(str(video_path)) as tc:
                for timestamps in call_sequences:
                    pyav_batch = pyav.get_frames_played_at(timestamps)
                    tc_batch = tc.get_frames_played_at(timestamps)

                    # Verify PTS match
                    np.testing.assert_array_almost_equal(pyav_batch.pts_seconds, tc_batch.pts_seconds, decimal=2)

                    # Verify pixel values
                    for i in range(len(timestamps)):
                        max_diff = np.max(np.abs(pyav_batch.data[i].astype(int) - tc_batch.data[i].astype(int)))
                        assert max_diff <= self.MAX_PIXEL_DIFF, f"Sequential call {timestamps}: max_diff={max_diff}"


@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistencyRealVideos:
    """Decoder consistency tests using real video files.

    NOTE: Some tests use @pytest.mark.parametrize instead of subtests because
    pytest-subtests doesn't work correctly with @pytest.mark.xfail - the test
    shows XPASS even when all subtests fail.
    """

    MAX_PIXEL_DIFF = 3

    # NOTE: Using parametrize instead of subtests for proper xfail behavior
    @pytest.mark.xfail(reason="TorchCodec seek behavior differs from PyAV on HEVC videos")
    @pytest.mark.parametrize(
        "name,timestamps",
        [
            ("start", [0.0, 0.1, 0.2]),
            ("middle", [3.0, 3.5, 4.0]),
            ("near_end", [6.5, 6.8, 6.9]),
            ("sparse", [0.0, 2.0, 4.0, 6.0]),
        ],
    )
    def test_real_video_hevc(self, example_mkv_path: Path, name: str, timestamps: list[float]):
        """Test consistency with real HEVC video (example.mkv)."""
        success, error, stats = _compare_decoder_outputs(str(example_mkv_path), timestamps, self.MAX_PIXEL_DIFF)
        assert success, f"{name}: {error} (stats: {stats})"

    def test_sparse_keyframe_video_early(self, example_video_path: Path):
        """Test consistency at video start (before keyframe issues)."""
        # Early frames work correctly on both decoders
        timestamps = [0.01, 0.02, 0.03]
        success, error, stats = _compare_decoder_outputs(str(example_video_path), timestamps, self.MAX_PIXEL_DIFF)
        assert success, f"early_frames: {error} (stats: {stats})"

    # NOTE: Using parametrize instead of subtests for proper xfail behavior
    @pytest.mark.xfail(reason="TorchCodec seeks to keyframe instead of decoding from start")
    @pytest.mark.parametrize(
        "name,timestamps",
        [
            ("across_keyframe_gap", [0.5, 1.0, 2.0, 3.0]),
            ("dense_middle", [2.0, 2.1, 2.2, 2.3, 2.4]),
        ],
    )
    def test_sparse_keyframe_video_across_gap(self, example_video_path: Path, name: str, timestamps: list[float]):
        """Test consistency across sparse keyframe gaps."""
        success, error, stats = _compare_decoder_outputs(str(example_video_path), timestamps, self.MAX_PIXEL_DIFF)
        assert success, f"{name}: {error} (stats: {stats})"


@pytest.mark.video
@pytest.mark.skipif(not _torchcodec_available(), reason="TorchCodec not installed")
class TestDecoderConsistencyEdgeCases:
    """Edge case tests for decoder consistency."""

    MAX_PIXEL_DIFF = 3

    def test_empty_timestamp_list(self, sample_video_file: tuple[Path, list[int]]):
        """Test both decoders handle empty timestamp list consistently."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at([])

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at([])

        # Both should return empty batches
        assert pyav_batch.data.shape[0] == 0
        assert torchcodec_batch.data.shape[0] == 0
        assert len(pyav_batch.pts_seconds) == 0
        assert len(torchcodec_batch.pts_seconds) == 0

    def test_output_format_consistency(self, sample_video_file: tuple[Path, list[int]]):
        """Test that both decoders return data in same format."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        timestamps = [0.0, 0.1, 0.2]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        # Check data format: NCHW (N, 3, H, W)
        assert pyav_batch.data.ndim == 4, "PyAV data should be 4D (NCHW)"
        assert torchcodec_batch.data.ndim == 4, "TorchCodec data should be 4D (NCHW)"
        assert pyav_batch.data.shape[1] == 3, "PyAV should have 3 channels (RGB)"
        assert torchcodec_batch.data.shape[1] == 3, "TorchCodec should have 3 channels (RGB)"

        # Check dtype
        assert pyav_batch.data.dtype == np.uint8, "PyAV data should be uint8"
        assert torchcodec_batch.data.dtype == np.uint8, "TorchCodec data should be uint8"

        # Check pts_seconds and duration_seconds dtype
        assert pyav_batch.pts_seconds.dtype == np.float64
        assert torchcodec_batch.pts_seconds.dtype == np.float64
        assert pyav_batch.duration_seconds.dtype == np.float64
        assert torchcodec_batch.duration_seconds.dtype == np.float64

    def test_duration_seconds_consistency(self, sample_video_file: tuple[Path, list[int]]):
        """Test that duration_seconds values are consistent between decoders."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_batch = pyav_decoder.get_frames_played_at(timestamps)

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_batch = torchcodec_decoder.get_frames_played_at(timestamps)

        # Duration should be consistent (within tolerance)
        np.testing.assert_array_almost_equal(
            pyav_batch.duration_seconds,
            torchcodec_batch.duration_seconds,
            decimal=2,
            err_msg="duration_seconds should match between decoders",
        )

    def test_metadata_consistency(self, sample_video_file: tuple[Path, list[int]]):
        """Test that metadata is consistent between decoders."""
        from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as pyav_decoder:
            pyav_meta = pyav_decoder.metadata

        with TorchCodecVideoDecoder(str(video_path)) as torchcodec_decoder:
            torchcodec_meta = torchcodec_decoder.metadata

        # Compare key metadata fields
        assert pyav_meta.width == torchcodec_meta.width, "Width should match"
        assert pyav_meta.height == torchcodec_meta.height, "Height should match"

        # Frame rate comparison (allow small tolerance due to Fraction vs float)
        pyav_fps = float(pyav_meta.average_rate)
        torchcodec_fps = float(torchcodec_meta.average_fps)
        assert abs(pyav_fps - torchcodec_fps) < 0.1, f"FPS mismatch: {pyav_fps} vs {torchcodec_fps}"

        # Duration comparison (allow 100ms tolerance)
        pyav_duration = float(pyav_meta.duration_seconds)
        torchcodec_duration = float(torchcodec_meta.duration_seconds)
        assert abs(pyav_duration - torchcodec_duration) < 0.1, (
            f"Duration mismatch: {pyav_duration} vs {torchcodec_duration}"
        )
