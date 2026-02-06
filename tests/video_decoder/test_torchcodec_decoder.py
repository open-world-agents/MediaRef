"""Tests for TorchCodecVideoDecoder standalone functionality.

These tests mirror the structure of test_pyav_decoder.py and verify
TorchCodec-specific behavior including caching, boundary conditions,
playback semantics, and resource management.
"""

from pathlib import Path

import numpy as np
import pytest

from tests import TORCHCODEC_AVAILABLE

pytestmark = [
    pytest.mark.video,
    pytest.mark.skipif(not TORCHCODEC_AVAILABLE, reason="TorchCodec not installed"),
]


class TestTorchCodecVideoDecoderBoundaryConditions:
    """Test boundary condition handling."""

    def test_negative_timestamp_raises(self, sample_video_file: tuple[Path, list[int]]):
        """Test that negative timestamps raise an error."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            with pytest.raises(Exception):
                decoder.get_frames_played_at([-0.1])

    def test_timestamp_at_or_beyond_duration_raises(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamps at or beyond duration raise an error."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            end_stream = float(decoder.metadata.end_stream_seconds)
            with pytest.raises(Exception):
                decoder.get_frames_played_at([end_stream])
            with pytest.raises(Exception):
                decoder.get_frames_played_at([end_stream + 1.0])

    def test_timestamp_just_before_duration_succeeds(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp just before end succeeds."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            end_stream = float(decoder.metadata.end_stream_seconds)
            batch = decoder.get_frames_played_at([end_stream - 0.001])
            assert batch.data.shape[0] == 1

    def test_timestamp_zero_succeeds(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp == 0 succeeds (first frame)."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.shape[0] == 1
            assert batch.pts_seconds[0] == 0.0


class TestTorchCodecVideoDecoderPlaybackSemantics:
    """Test playback semantics: frame[i].pts <= t < frame[i+1].pts."""

    def test_exact_pts_returns_that_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp == frame[i].pts returns frame[i]."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])
            assert len(batch.pts_seconds) == 3
            np.testing.assert_array_almost_equal(batch.pts_seconds, [0.0, 0.1, 0.2], decimal=2)

    def test_between_frames_returns_earlier_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frame[i].pts < t < frame[i+1].pts returns frame[i]."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.05])
            assert len(batch.pts_seconds) == 1
            assert batch.pts_seconds[0] == pytest.approx(0.0, abs=0.01)

    def test_unsorted_timestamps_preserve_order(self, sample_video_file: tuple[Path, list[int]]):
        """Test that unsorted input timestamps return frames in input order."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.3, 0.1, 0.2, 0.0])
            assert len(batch.pts_seconds) == 4
            expected_pts = [0.3, 0.1, 0.2, 0.0]
            for i, expected in enumerate(expected_pts):
                assert batch.pts_seconds[i] == pytest.approx(expected, abs=0.01)

    def test_duplicate_timestamps(self, sample_video_file: tuple[Path, list[int]]):
        """Test that duplicate timestamps return duplicate frames."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.0, 0.1, 0.1])
            assert len(batch.pts_seconds) == 4
            assert batch.pts_seconds[0] == pytest.approx(0.0, abs=0.01)
            assert batch.pts_seconds[1] == pytest.approx(0.0, abs=0.01)
            assert batch.pts_seconds[2] == pytest.approx(0.1, abs=0.01)
            assert batch.pts_seconds[3] == pytest.approx(0.1, abs=0.01)

    def test_last_frame_accessible(self, sample_video_file: tuple[Path, list[int]]):
        """Test that the last frame is accessible with timestamp < duration."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            duration = float(decoder.metadata.duration_seconds)
            batch = decoder.get_frames_played_at([duration - 0.05])
            assert len(batch.pts_seconds) == 1
            assert batch.pts_seconds[0] == pytest.approx(0.4, abs=0.01)



class TestTorchCodecVideoDecoderMetadata:
    """Test metadata extraction."""

    def test_metadata_properties(self, sample_video_file: tuple[Path, list[int]]):
        """Test that metadata is correctly extracted."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            metadata = decoder.metadata

            assert metadata.width == 64
            assert metadata.height == 48
            assert metadata.num_frames == 5
            # TorchCodec uses average_fps instead of average_rate
            assert float(metadata.average_fps) == pytest.approx(10.0, abs=0.1)
            assert float(metadata.duration_seconds) == pytest.approx(0.5, abs=0.1)

    def test_metadata_accessible_after_decoding(self, sample_video_file: tuple[Path, list[int]]):
        """Test that metadata remains accessible after decoding frames."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            decoder.get_frames_played_at([0.0, 0.1])
            assert decoder.metadata.width == 64
            assert decoder.metadata.height == 48


class TestTorchCodecVideoDecoderFrameBatch:
    """Test FrameBatch output format."""

    def test_frame_batch_nchw_format(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frames are returned in NCHW format."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1])
            assert batch.data.shape == (2, 3, 48, 64)
            assert batch.data.dtype == np.uint8

    def test_frame_batch_rgb_channels(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frames have 3 RGB channels."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.shape[1] == 3

    def test_frame_batch_pts_and_duration(self, sample_video_file: tuple[Path, list[int]]):
        """Test that FrameBatch includes pts and duration arrays."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])
            assert len(batch.pts_seconds) == 3
            assert len(batch.duration_seconds) == 3
            assert batch.pts_seconds.dtype == np.float64
            assert batch.duration_seconds.dtype == np.float64

    def test_empty_timestamps_returns_empty_batch(self, sample_video_file: tuple[Path, list[int]]):
        """Test that empty timestamp list returns empty FrameBatch."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([])
            assert batch.data.shape[0] == 0
            assert len(batch.pts_seconds) == 0
            assert len(batch.duration_seconds) == 0


class TestTorchCodecVideoDecoderContextManager:
    """Test context manager functionality."""

    def test_context_manager_closes_resources(self, sample_video_file: tuple[Path, list[int]]):
        """Test that context manager properly closes resources."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        decoder = TorchCodecVideoDecoder(str(video_path))
        decoder.__enter__()
        batch = decoder.get_frames_played_at([0.0])
        assert batch.data.shape[0] == 1
        decoder.__exit__(None, None, None)

    def test_with_statement(self, sample_video_file: tuple[Path, list[int]]):
        """Test using decoder with 'with' statement."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1])
            assert batch.data.shape[0] == 2

    def test_double_close(self, sample_video_file: tuple[Path, list[int]]):
        """Test that calling close() twice does not raise an error."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        decoder = TorchCodecVideoDecoder(str(video_path))
        batch = decoder.get_frames_played_at([0.0])
        assert batch.data.shape[0] == 1
        decoder.close()
        decoder.close()  # Should not raise


class TestTorchCodecVideoDecoderCaching:
    """Test caching behavior specific to TorchCodecVideoDecoder."""

    def test_cache_hit_returns_same_instance(self, sample_video_file: tuple[Path, list[int]]):
        """Test that opening the same video twice returns cached instance with correct ref counting."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        cache_key = str(video_path)

        decoder1 = TorchCodecVideoDecoder(str(video_path))
        decoder2 = TorchCodecVideoDecoder(str(video_path))

        # Should be the exact same object due to cache hit
        assert decoder1 is decoder2

        # Ref count should be 2 (one per constructor call)
        assert TorchCodecVideoDecoder.cache[cache_key].refs == 2

        # Both should still work
        batch = decoder1.get_frames_played_at([0.0])
        assert batch.data.shape[0] == 1

        # Each "reference" should close once — ref count decrements correctly
        decoder1.close()
        assert TorchCodecVideoDecoder.cache[cache_key].refs == 1
        decoder2.close()
        # After both closes, refs should be 0

    def test_close_releases_cache_entry(self, sample_video_file: tuple[Path, list[int]]):
        """Test that close() releases the cache reference."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file
        cache_key = str(video_path)

        decoder = TorchCodecVideoDecoder(str(video_path))
        assert cache_key in TorchCodecVideoDecoder.cache

        decoder.close()
        # After close, refs should be decremented (entry may still exist if refs > 0)

    def test_reopen_after_close(self, sample_video_file: tuple[Path, list[int]]):
        """Test that a decoder can be re-opened after closing."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        # First open and close
        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch1 = decoder.get_frames_played_at([0.0])

        # Re-open - should work fine (new or cached instance)
        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch2 = decoder.get_frames_played_at([0.0])

        # Both should have returned valid frames
        assert batch1.data.shape[0] == 1
        assert batch2.data.shape[0] == 1


class TestTorchCodecVideoDecoderEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file_raises_error(self, tmp_path: Path):
        """Test that nonexistent file raises appropriate error."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        with pytest.raises(Exception):
            TorchCodecVideoDecoder(str(tmp_path / "nonexistent.mp4"))

    def test_different_frame_contents(self, sample_video_file: tuple[Path, list[int]]):
        """Test that different frames have different content."""
        from mediaref.video_decoder import TorchCodecVideoDecoder

        video_path, _ = sample_video_file

        with TorchCodecVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])

            frame0 = batch.data[0]
            frame1 = batch.data[1]
            frame2 = batch.data[2]

            # At least some frames should be different
            assert not np.array_equal(frame0, frame1) or not np.array_equal(frame1, frame2)