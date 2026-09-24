from fractions import Fraction

import av
import numpy as np
import pytest

from mediaref.video_decoder import PyAVVideoDecoder, probe_video


@pytest.fixture(params=["gray12le", "gray16le", "gray16be"])
def native_video(tmp_path, request):
    pixel_format = request.param
    path = tmp_path / "native.nut"
    expected = np.arange(256, dtype=np.uint16).reshape(16, 16) * (7 if pixel_format == "gray12le" else 127)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("rawvideo", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = pixel_format
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(expected + i, format=pixel_format)
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path, pixel_format, expected


def test_native_values_playback_and_nearest(native_video):
    path, fmt, expected = native_video
    with PyAVVideoDecoder(path, output_format="native", expected_pixel_format=fmt) as decoder:
        batch = decoder.get_frames_played_at([0.19, 0.0, 0.19])
        assert batch.pixel_format == fmt
        assert batch.data.dtype == np.uint16
        assert batch.data.shape == (3, 1, 16, 16)
        np.testing.assert_array_equal(batch.data[:, 0], [expected + 1, expected, expected + 1])
        nearest = decoder.get_frames_nearest_at([0.19, 0.05, 0.19], tolerance=0.051)
        np.testing.assert_array_equal(nearest.data[:, 0], [expected + 2, expected, expected + 2])
        with pytest.raises(ValueError, match="No frame"):
            decoder.get_frames_nearest_at([0.19], tolerance=0.001)
        empty = decoder.get_frames_played_at([])
        assert empty.data.shape == (0, 1, 16, 16) and empty.data.dtype == np.uint16
        selected = decoder.get_frames_played_in_range(0, 0.15)
        np.testing.assert_array_equal(selected.data[:, 0], [expected, expected + 1])
    with PyAVVideoDecoder(path, output_format="native", expected_pixel_format="gray") as decoder:
        with pytest.raises(ValueError, match="Expected source"):
            decoder.get_frames_played_at([0])
    header = probe_video(path)
    assert header.pixel_format == fmt
    assert (header.width, header.height) == (16, 16)
    assert header.num_frames is None  # NUT has no declared count; do not estimate it.


def test_native_unsupported_format(tmp_path):
    path = tmp_path / "rgb.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in [*stream.encode(frame), *stream.encode()]:
            container.mux(packet)
    with PyAVVideoDecoder(path, output_format="native") as decoder:
        with pytest.raises(ValueError, match="does not support"):
            decoder.get_frames_played_at([0])
    with PyAVVideoDecoder(path) as decoder:
        assert decoder.get_frames_played_at([0]).data.shape == (1, 3, 16, 16)


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -1])
def test_native_invalid_time(native_video, time):
    with PyAVVideoDecoder(native_video[0], output_format="native") as decoder:
        with pytest.raises(ValueError):
            decoder.get_frames_nearest_at([time], tolerance=0.1)
