"""Tests for PyAVVideoDecoder with TorchCodec playback semantics.

These tests verify the implementation follows playback_semantics.md:
- Playback semantics: frame[i].pts <= timestamp < frame[i+1].pts
- Boundary conditions: negative timestamps, timestamps >= duration
- Frame selection logic using nextPts
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.video
class TestPyAVVideoDecoderBoundaryConditions:
    """Test boundary condition handling per playback_semantics.md."""

    def test_negative_timestamp_raises_value_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp < 0 raises ValueError."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            with pytest.raises(ValueError, match="< begin_stream_seconds"):
                decoder.get_frames_played_at([-0.1])

    def test_negative_timestamp_various_values(self, sample_video_file: tuple[Path, list[int]]):
        """Test various negative timestamps all raise ValueError."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            for neg_ts in [-0.001, -1.0, -100.0, -1e-9]:
                with pytest.raises(ValueError, match="< begin_stream_seconds"):
                    decoder.get_frames_played_at([neg_ts])

    def test_timestamp_at_duration_raises_value_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp == end_stream_seconds raises ValueError."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            end_stream = float(decoder.metadata.end_stream_seconds)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match=">=.*end_stream"):
                decoder.get_frames_played_at([end_stream])

    def test_timestamp_beyond_duration_raises_value_error(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp > end_stream_seconds raises ValueError."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            end_stream = float(decoder.metadata.end_stream_seconds)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match=">=.*end_stream"):
                decoder.get_frames_played_at([end_stream + 0.1])
            with pytest.raises(ValueError, match=">=.*end_stream"):
                decoder.get_frames_played_at([end_stream + 100.0])

    def test_timestamp_just_before_duration_succeeds(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp just before end_stream_seconds succeeds."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            end_stream = float(decoder.metadata.end_stream_seconds)  # type: ignore[arg-type]
            # Just before the end should work
            batch = decoder.get_frames_played_at([end_stream - 0.001])
            assert batch.data.shape[0] == 1

    def test_timestamp_zero_succeeds(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp == 0 succeeds (first frame)."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.shape[0] == 1
            assert batch.pts_seconds[0] == 0.0


@pytest.mark.video
class TestPyAVVideoDecoderPlaybackSemantics:
    """Test TorchCodec playback semantics: frame[i].pts <= t < frame[i+1].pts."""

    def test_exact_pts_returns_that_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test that timestamp == frame[i].pts returns frame[i]."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, timestamps_ns = sample_video_file
        # Video has 5 frames at 10fps: pts = 0.0, 0.1, 0.2, 0.3, 0.4

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Request exactly at frame pts
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])

            # Should return frames with those exact pts
            assert len(batch.pts_seconds) == 3
            np.testing.assert_array_almost_equal(batch.pts_seconds, [0.0, 0.1, 0.2], decimal=2)

    def test_between_frames_returns_earlier_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frame[i].pts < t < frame[i+1].pts returns frame[i]."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file
        # Video has 5 frames at 10fps: pts = 0.0, 0.1, 0.2, 0.3, 0.4

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Request between frame 0 (pts=0.0) and frame 1 (pts=0.1)
            batch = decoder.get_frames_played_at([0.05])

            # Should return frame 0 (pts=0.0)
            assert len(batch.pts_seconds) == 1
            assert batch.pts_seconds[0] == pytest.approx(0.0, abs=0.01)

    def test_just_before_next_frame_returns_current_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test timestamp just before next frame's pts returns current frame."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Request at 0.099s (just before frame 1 at 0.1s)
            batch = decoder.get_frames_played_at([0.099])

            # Should return frame 0 (pts=0.0)
            assert batch.pts_seconds[0] == pytest.approx(0.0, abs=0.01)

    def test_multiple_timestamps_same_frame(self, sample_video_file: tuple[Path, list[int]]):
        """Test multiple timestamps mapping to the same frame."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # All these should return frame 0 (pts=0.0)
            batch = decoder.get_frames_played_at([0.0, 0.01, 0.05, 0.09])

            assert len(batch.pts_seconds) == 4
            # All should have pts=0.0
            for pts in batch.pts_seconds:
                assert pts == pytest.approx(0.0, abs=0.01)

    def test_unsorted_timestamps_preserve_order(self, sample_video_file: tuple[Path, list[int]]):
        """Test that unsorted input timestamps return frames in input order."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Request in reverse order
            batch = decoder.get_frames_played_at([0.3, 0.1, 0.2, 0.0])

            assert len(batch.pts_seconds) == 4
            # Results should be in input order
            expected_pts = [0.3, 0.1, 0.2, 0.0]
            for i, expected in enumerate(expected_pts):
                assert batch.pts_seconds[i] == pytest.approx(expected, abs=0.01)

    def test_last_frame_accessible(self, sample_video_file: tuple[Path, list[int]]):
        """Test that the last frame is accessible with timestamp < duration."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file
        # Video has 5 frames at 10fps: pts = 0.0, 0.1, 0.2, 0.3, 0.4
        # Duration should be 0.5s

        with PyAVVideoDecoder(str(video_path)) as decoder:
            duration = float(decoder.metadata.duration_seconds)
            # Request just before end - should get last frame
            batch = decoder.get_frames_played_at([duration - 0.05])

            assert len(batch.pts_seconds) == 1
            # Should be the last frame (pts=0.4)
            assert batch.pts_seconds[0] == pytest.approx(0.4, abs=0.01)


@pytest.mark.video
class TestPyAVVideoDecoderMetadata:
    """Test metadata extraction."""

    def test_metadata_properties(self, sample_video_file: tuple[Path, list[int]]):
        """Test that metadata is correctly extracted."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            metadata = decoder.metadata

            assert metadata.width == 64
            assert metadata.height == 48
            assert metadata.num_frames == 5
            assert float(metadata.average_rate) == pytest.approx(10.0, abs=0.1)
            assert float(metadata.duration_seconds) == pytest.approx(0.5, abs=0.1)

    def test_metadata_accessible_after_decoding(self, sample_video_file: tuple[Path, list[int]]):
        """Test that metadata remains accessible after decoding frames."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Decode some frames
            decoder.get_frames_played_at([0.0, 0.1])

            # Metadata should still be accessible
            assert decoder.metadata.width == 64
            assert decoder.metadata.height == 48


@pytest.mark.video
class TestPyAVVideoDecoderFrameBatch:
    """Test FrameBatch output format."""

    def test_frame_batch_nchw_format(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frames are returned in NCHW format."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1])

            # Shape should be (N, C, H, W)
            assert batch.data.shape == (2, 3, 48, 64)
            assert batch.data.dtype == np.uint8

    def test_frame_batch_rgb_channels(self, sample_video_file: tuple[Path, list[int]]):
        """Test that frames have 3 RGB channels."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0])

            # Should have 3 channels (RGB)
            assert batch.data.shape[1] == 3

    def test_frame_batch_pts_and_duration(self, sample_video_file: tuple[Path, list[int]]):
        """Test that FrameBatch includes pts and duration arrays."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])

            assert len(batch.pts_seconds) == 3
            assert len(batch.duration_seconds) == 3
            assert batch.pts_seconds.dtype == np.float64
            assert batch.duration_seconds.dtype == np.float64

    def test_empty_timestamps_returns_empty_batch(self, sample_video_file: tuple[Path, list[int]]):
        """Test that empty timestamp list returns empty FrameBatch."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([])

            assert batch.data.shape[0] == 0
            assert len(batch.pts_seconds) == 0
            assert len(batch.duration_seconds) == 0


@pytest.mark.video
class TestPyAVVideoDecoderContextManager:
    """Test context manager functionality."""

    def test_context_manager_closes_resources(self, sample_video_file: tuple[Path, list[int]]):
        """Test that context manager properly closes resources."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        decoder = PyAVVideoDecoder(str(video_path))
        decoder.__enter__()
        batch = decoder.get_frames_played_at([0.0])
        assert batch.data.shape[0] == 1
        decoder.__exit__(None, None, None)

        # After exit, the container should be closed
        # (we can't easily verify this without accessing internals)

    def test_with_statement(self, sample_video_file: tuple[Path, list[int]]):
        """Test using decoder with 'with' statement."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1])
            assert batch.data.shape[0] == 2

    def test_double_close(self, sample_video_file: tuple[Path, list[int]]):
        """Test that calling close() twice does not raise an error."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        decoder = PyAVVideoDecoder(str(video_path))
        batch = decoder.get_frames_played_at([0.0])
        assert batch.data.shape[0] == 1
        decoder.close()
        decoder.close()  # Should not raise


@pytest.mark.video
class TestPyAVVideoDecoderEdgeCases:
    """Test edge cases and error handling."""

    def test_single_frame_video(self, tmp_path: Path):
        """Test decoding a video with only one frame."""
        import av
        from fractions import Fraction

        from mediaref.video_decoder import PyAVVideoDecoder

        # Create a single-frame video
        video_path = tmp_path / "single_frame.mp4"
        container = av.open(str(video_path), "w")
        stream = container.add_stream("h264", rate=1)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame(64, 48, "rgb24")
        arr = np.full((48, 64, 3), 128, dtype=np.uint8)
        frame.planes[0].update(arr)
        frame.pts = 0
        frame.time_base = Fraction(1, 1)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.shape[0] == 1

    def test_nonexistent_file_raises_error(self, tmp_path: Path):
        """Test that nonexistent file raises appropriate error."""
        from mediaref.video_decoder import PyAVVideoDecoder

        with pytest.raises(Exception):  # Could be FileNotFoundError or av.error
            PyAVVideoDecoder(str(tmp_path / "nonexistent.mp4"))

    def test_different_frame_contents(self, sample_video_file: tuple[Path, list[int]]):
        """Test that different frames have different content."""
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path, _ = sample_video_file

        with PyAVVideoDecoder(str(video_path)) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.2])

            # Frames should have different content
            frame0 = batch.data[0]
            frame1 = batch.data[1]
            frame2 = batch.data[2]

            # At least some frames should be different
            assert not np.array_equal(frame0, frame1) or not np.array_equal(frame1, frame2)

    def test_nonzero_begin_stream_rejects_earlier_timestamps(self, tmp_path: Path):
        """Test that videos with non-zero begin_stream_seconds reject earlier timestamps."""
        import av
        from fractions import Fraction

        from mediaref.video_decoder import PyAVVideoDecoder

        # Create a video where first frame PTS is 0.5 (not 0)
        video_path = tmp_path / "nonzero_start.mp4"
        container = av.open(str(video_path), "w")
        stream = container.add_stream("h264", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        # Create 5 frames with pts starting at 0.5
        for i in range(5):
            frame = av.VideoFrame(64, 48, "rgb24")
            arr = np.full((48, 64, 3), i * 50, dtype=np.uint8)
            frame.planes[0].update(arr)
            frame.pts = i + 5  # PTS = 5, 6, 7, 8, 9 in time_base 1/10 → 0.5, 0.6, ...
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Verify begin_stream_seconds is 0.5
            assert float(decoder.metadata.begin_stream_seconds) == pytest.approx(0.5, abs=0.01)

            # Timestamp before begin_stream_seconds should raise ValueError
            with pytest.raises(ValueError, match="< begin_stream_seconds"):
                decoder.get_frames_played_at([0.2])

            # Timestamp at begin_stream_seconds should succeed
            batch = decoder.get_frames_played_at([0.5])
            assert batch.data.shape[0] == 1
            assert batch.pts_seconds[0] == pytest.approx(0.5, abs=0.01)

    def test_sparse_keyframes_seek_fallback(self, example_video_path: Path):
        """Test that videos with sparse keyframes are handled correctly.

        The example_video.mkv has only 2 keyframes (at 0.010s and 3.690s).
        Seeking to 0.018s would normally land at 3.690s, but our implementation
        should detect this and fall back to decoding from start.
        """
        from mediaref.video_decoder import PyAVVideoDecoder

        video_path = example_video_path

        with PyAVVideoDecoder(str(video_path)) as decoder:
            # Query for timestamp 0.018s - should return frame at 0.017s
            # (the frame that was being displayed at 0.018s)
            batch = decoder.get_frames_played_at([0.018])
            assert batch.data.shape[0] == 1
            # Should get the frame at 0.017s (frame 1), not 3.690s
            assert batch.pts_seconds[0] == pytest.approx(0.017, abs=0.001)

            # Also test multiple queries across the sparse keyframe gap
            batch2 = decoder.get_frames_played_at([0.010, 0.018, 0.5, 1.0])
            assert batch2.data.shape[0] == 4
            assert batch2.pts_seconds[0] == pytest.approx(0.010, abs=0.001)  # First frame
            assert batch2.pts_seconds[1] == pytest.approx(0.017, abs=0.001)  # Second frame
            # 0.5s and 1.0s should return frame at 0.480s (frame 11)
            assert batch2.pts_seconds[2] == pytest.approx(0.480, abs=0.001)
            assert batch2.pts_seconds[3] == pytest.approx(0.480, abs=0.001)


# Note: Decoder consistency tests (PyAV vs TorchCodec) have been moved to
# tests/video_decoder/test_decoder_consistency.py
