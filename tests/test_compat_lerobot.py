"""Tests for ``mediaref.compat.lerobot`` interop helpers.

These tests do not import lerobot — the helpers operate on plain dicts.
"""

from __future__ import annotations

import pytest

from mediaref import MediaRef
from mediaref.compat.lerobot import (
    from_videoframe,
    lerobot_episode_to_refs,
    to_videoframe,
)


class TestFromVideoFrame:
    def test_basic_conversion(self):
        ref = from_videoframe({"path": "videos/clip.mp4", "timestamp": 0.5})
        assert ref == MediaRef(uri="videos/clip.mp4", pts_ns=500_000_000)

    def test_zero_timestamp(self):
        ref = from_videoframe({"path": "v.mp4", "timestamp": 0.0})
        assert ref.pts_ns == 0

    def test_int_timestamp_accepted(self):
        # lerobot uses float32 but ints arithmetically equal floats are fine.
        ref = from_videoframe({"path": "v.mp4", "timestamp": 2})
        assert ref.pts_ns == 2_000_000_000

    def test_extra_keys_ignored(self):
        ref = from_videoframe({"path": "v.mp4", "timestamp": 0.1, "extra": "ignored"})
        assert ref.pts_ns == 100_000_000

    def test_rounds_to_nearest_ns(self):
        # Float arithmetic: 0.1 sec ≈ 0.1 → 100_000_000 ns (rounding away IEEE noise).
        ref = from_videoframe({"path": "v.mp4", "timestamp": 0.1})
        assert ref.pts_ns == 100_000_000

    def test_missing_path_raises(self):
        with pytest.raises(KeyError):
            from_videoframe({"timestamp": 0.0})

    def test_missing_timestamp_raises(self):
        with pytest.raises(KeyError):
            from_videoframe({"path": "v.mp4"})

    def test_non_string_path_raises(self):
        with pytest.raises(TypeError):
            from_videoframe({"path": 42, "timestamp": 0.0})

    def test_non_numeric_timestamp_raises(self):
        with pytest.raises(TypeError):
            from_videoframe({"path": "v.mp4", "timestamp": "0.5"})


class TestToVideoFrame:
    def test_basic_conversion(self):
        ref = MediaRef(uri="v.mp4", pts_ns=500_000_000)
        assert to_videoframe(ref) == {"path": "v.mp4", "timestamp": 0.5}

    def test_zero_pts(self):
        ref = MediaRef(uri="v.mp4", pts_ns=0)
        assert to_videoframe(ref) == {"path": "v.mp4", "timestamp": 0.0}

    def test_image_pts_none_raises(self):
        ref = MediaRef(uri="img.png")  # pts_ns=None ⇒ still image
        with pytest.raises(ValueError, match="pts_ns is None"):
            to_videoframe(ref)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "vf",
        [
            {"path": "v.mp4", "timestamp": 0.0},
            {"path": "v.mp4", "timestamp": 0.5},
            {"path": "videos/episode_42/cam.mp4", "timestamp": 9.96},
        ],
    )
    def test_videoframe_roundtrip(self, vf):
        # VideoFrame uses float32 sec; round-trip through int64 ns is exact for
        # multiples of 1ns. Cap precision at 1 microsecond to avoid float noise.
        ref = from_videoframe(vf)
        back = to_videoframe(ref)
        assert back["path"] == vf["path"]
        assert abs(back["timestamp"] - vf["timestamp"]) < 1e-6


class TestLerobotEpisodeToRefs:
    def test_basic_episode(self):
        # A 3-frame episode starting at offset 1.0s in a shared mp4, recorded
        # at 30fps. Frame timestamps are episode-local (0.0, 1/30, 2/30).
        refs = lerobot_episode_to_refs(
            video_path="videos/file_000.mp4",
            from_timestamp=1.0,
            frame_timestamps=[0.0, 1 / 30, 2 / 30],
        )
        assert len(refs) == 3
        assert all(r.uri == "videos/file_000.mp4" for r in refs)
        # Within-shard absolute time = 1.0 + episode_local_t
        assert refs[0].pts_ns == 1_000_000_000
        assert abs(refs[1].pts_ns - 1_033_333_333) <= 1
        assert abs(refs[2].pts_ns - 1_066_666_667) <= 1

    def test_empty_frame_list(self):
        refs = lerobot_episode_to_refs(video_path="v.mp4", from_timestamp=0.0, frame_timestamps=[])
        assert refs == []

    def test_iterable_input(self):
        # Accepts any iterable, not just list.
        gen = (i * 0.1 for i in range(5))
        refs = lerobot_episode_to_refs(video_path="v.mp4", from_timestamp=0.0, frame_timestamps=gen)
        assert len(refs) == 5
        assert refs[0].pts_ns == 0
        assert refs[4].pts_ns == 400_000_000
