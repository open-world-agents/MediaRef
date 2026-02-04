"""Tests for internal utility functions.

These tests verify internal implementation details not exposed by the public API:
- Internal RGBA format handling (load_image_as_rgba)
- Playback semantics for video frame seeking
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
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(decoded_rgba, sample_rgba_array)
        assert np.abs(decoded_rgba.astype(float) - sample_rgba_array.astype(float)).mean() < 50


@pytest.mark.video
class TestPlaybackSemantics:
    """Test playback-accurate frame seeking.

    For video playback, the frame visible at time T should be the frame where:
        frame.time <= T < frame.time + frame.duration

    This means querying at any time within a frame's duration should return that frame.
    """

    @pytest.fixture
    def known_framerate_video(self, tmp_path: Path) -> tuple[Path, float, int]:
        """Create a video with a well-defined framerate for precise timing tests.

        Creates a 10fps video with 10 frames (1 second duration).
        Each frame has a distinct color intensity to enable identification.

        Returns:
            Tuple of (video_path, frame_duration_seconds, num_frames)
        """
        import av
        from fractions import Fraction

        video_path = tmp_path / "test_10fps.mp4"
        fps = 10
        num_frames = 10
        frame_duration = 1.0 / fps

        container = av.open(str(video_path), "w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        for i in range(num_frames):
            frame = av.VideoFrame(64, 48, "rgb24")
            # Create distinct intensity for each frame (0, 25, 50, 75, ...)
            intensity = i * 25
            arr = np.full((48, 64, 3), intensity, dtype=np.uint8)
            frame.planes[0].update(arr)
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        return video_path, frame_duration, num_frames

    def test_query_at_exact_frame_start_returns_that_frame(self, known_framerate_video: tuple[Path, float, int]):
        """Test that querying at exact frame start time returns that frame."""
        from mediaref.video_decoder import PyAVVideoDecoder, BatchDecodingStrategy

        video_path, frame_duration, num_frames = known_framerate_video

        with PyAVVideoDecoder(video_path) as decoder:
            for frame_idx in range(num_frames):
                expected_intensity = frame_idx * 25
                query_time = frame_idx * frame_duration

                batch = decoder.get_frames_played_at([query_time], strategy=BatchDecodingStrategy.SEPARATE)

                actual_intensity = batch.data[0].mean()
                assert abs(actual_intensity - expected_intensity) < 5, (
                    f"Query at {query_time:.3f}s (frame {frame_idx}) expected intensity "
                    f"~{expected_intensity}, got {actual_intensity:.1f}"
                )

    def test_query_at_mid_frame_returns_that_frame(self, known_framerate_video: tuple[Path, float, int]):
        """Test that querying at middle of frame duration returns that frame."""
        from mediaref.video_decoder import PyAVVideoDecoder, BatchDecodingStrategy

        video_path, frame_duration, num_frames = known_framerate_video

        with PyAVVideoDecoder(video_path) as decoder:
            for frame_idx in range(num_frames):
                expected_intensity = frame_idx * 25
                query_time = frame_idx * frame_duration + frame_duration / 2

                batch = decoder.get_frames_played_at([query_time], strategy=BatchDecodingStrategy.SEPARATE)

                actual_intensity = batch.data[0].mean()
                assert abs(actual_intensity - expected_intensity) < 5, (
                    f"Query at {query_time:.3f}s (mid-frame {frame_idx}) expected intensity "
                    f"~{expected_intensity}, got {actual_intensity:.1f}"
                )

    def test_query_just_before_next_frame_returns_current_frame(self, known_framerate_video: tuple[Path, float, int]):
        """Test that querying just before next frame starts returns current frame."""
        from mediaref.video_decoder import PyAVVideoDecoder, BatchDecodingStrategy

        video_path, frame_duration, num_frames = known_framerate_video
        epsilon = 0.0001

        with PyAVVideoDecoder(video_path) as decoder:
            for frame_idx in range(num_frames - 1):
                expected_intensity = frame_idx * 25
                query_time = (frame_idx + 1) * frame_duration - epsilon

                batch = decoder.get_frames_played_at([query_time], strategy=BatchDecodingStrategy.SEPARATE)

                actual_intensity = batch.data[0].mean()
                assert abs(actual_intensity - expected_intensity) < 5, (
                    f"Query at {query_time:.3f}s (just before frame {frame_idx + 1}) "
                    f"expected intensity ~{expected_intensity}, got {actual_intensity:.1f}"
                )

    def test_all_strategies_return_same_frame(self, known_framerate_video: tuple[Path, float, int]):
        """Test that all decoding strategies return the same frame for a given query."""
        from mediaref.video_decoder import PyAVVideoDecoder, BatchDecodingStrategy

        video_path, frame_duration, num_frames = known_framerate_video
        strategies = [
            BatchDecodingStrategy.SEPARATE,
            BatchDecodingStrategy.SEQUENTIAL,
            BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
        ]

        with PyAVVideoDecoder(video_path) as decoder:
            for frame_idx in range(num_frames):
                query_time = frame_idx * frame_duration + frame_duration / 2

                results = {}
                for strategy in strategies:
                    batch = decoder.get_frames_played_at([query_time], strategy=strategy)
                    results[strategy.name] = batch.data[0].mean()

                intensities = list(results.values())
                intensity_range = max(intensities) - min(intensities)
                assert intensity_range < 5, f"Strategies disagree at {query_time:.3f}s: {results}"

    def test_batch_query_returns_correct_frames(self, known_framerate_video: tuple[Path, float, int]):
        """Test that batch queries return correct frames in correct order."""
        from mediaref.video_decoder import PyAVVideoDecoder, BatchDecodingStrategy

        video_path, frame_duration, num_frames = known_framerate_video

        query_times = [i * frame_duration + frame_duration / 2 for i in range(num_frames)]
        expected_intensities = [i * 25 for i in range(num_frames)]

        for strategy in [
            BatchDecodingStrategy.SEPARATE,
            BatchDecodingStrategy.SEQUENTIAL,
            BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
        ]:
            with PyAVVideoDecoder(video_path) as decoder:
                batch = decoder.get_frames_played_at(query_times, strategy=strategy)

                for i, (expected, frame) in enumerate(zip(expected_intensities, batch.data)):
                    actual = frame.mean()
                    assert abs(actual - expected) < 5, (
                        f"Strategy {strategy.name}: Frame {i} at {query_times[i]:.3f}s "
                        f"expected intensity ~{expected}, got {actual:.1f}"
                    )

    def test_individual_and_batch_return_same_frames(self, known_framerate_video: tuple[Path, float, int]):
        """Test that individual to_ndarray() and batch_decode() return same frames."""
        from mediaref import MediaRef, batch_decode, cleanup_cache

        video_path, frame_duration, num_frames = known_framerate_video

        pts_ns_list = [int((i * frame_duration + frame_duration / 2) * 1_000_000_000) for i in range(num_frames)]
        refs = [MediaRef(uri=str(video_path), pts_ns=pts_ns) for pts_ns in pts_ns_list]

        cleanup_cache()
        batch_results = batch_decode(refs)

        cleanup_cache()
        individual_results = [ref.to_ndarray() for ref in refs]

        for i, (batch_frame, individual_frame) in enumerate(zip(batch_results, individual_results)):
            np.testing.assert_array_equal(
                batch_frame,
                individual_frame,
                err_msg=f"Frame {i}: batch and individual decode returned different frames",
            )
