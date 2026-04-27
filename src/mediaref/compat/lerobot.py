"""Interop helpers for `lerobot <https://github.com/huggingface/lerobot>`_.

lerobot's ``VideoFrame`` (defined in
``lerobot/datasets/video_utils.py``) represents a frame as
``{"path": str, "timestamp": float32 seconds}``. MediaRef represents the
same concept as ``(uri: str, pts_ns: int64 nanoseconds)``.

These helpers convert between the two without requiring lerobot to be
installed — they only manipulate plain Python values.

For the v3.0 LeRobotDataset directory layout (multi-episode shared mp4
shards) use :func:`lerobot_episode_to_refs`, which combines an episode's
``from_timestamp`` offset with a list of in-episode frame timestamps.

Example:
    >>> from mediaref.compat.lerobot import from_videoframe
    >>> from_videoframe({"path": "videos/clip.mp4", "timestamp": 0.5})
    MediaRef(uri='videos/clip.mp4', pts_ns=500000000)
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .._internal import NANOSECOND
from ..core import MediaRef


def _sec_to_ns(seconds: float) -> int:
    """Round seconds (float) to int64 nanoseconds. Used both for whole
    timestamps and for episode-local offsets, so it's worth one place."""
    return int(round(float(seconds) * NANOSECOND))


def from_videoframe(vf: Mapping[str, object]) -> MediaRef:
    """Convert a lerobot ``VideoFrame`` dict to a :class:`MediaRef`.

    Args:
        vf: A dict with at least ``path`` (str) and ``timestamp`` (float
            seconds). Extra keys are ignored.

    Returns:
        ``MediaRef(uri=vf["path"], pts_ns=int(round(vf["timestamp"] * 1e9)))``.

    Raises:
        KeyError: If ``path`` or ``timestamp`` is missing.
        TypeError: If ``path`` is not a str or ``timestamp`` is not numeric.
    """
    path = vf["path"]
    ts = vf["timestamp"]
    if not isinstance(path, str):
        raise TypeError(f"VideoFrame.path must be str; got {type(path).__name__}")
    if not isinstance(ts, (int, float)):
        raise TypeError(f"VideoFrame.timestamp must be numeric; got {type(ts).__name__}")
    return MediaRef(uri=path, pts_ns=_sec_to_ns(ts))


def to_videoframe(ref: MediaRef) -> dict:
    """Convert a :class:`MediaRef` to a lerobot ``VideoFrame`` dict.

    Args:
        ref: A MediaRef. ``ref.pts_ns`` must not be ``None`` (lerobot's
            ``VideoFrame`` requires a timestamp).

    Returns:
        ``{"path": ref.uri, "timestamp": ref.pts_ns / 1e9}``.

    Raises:
        ValueError: If ``ref.pts_ns`` is ``None`` (still images cannot be
            represented as VideoFrame).
    """
    if ref.pts_ns is None:
        raise ValueError(
            "Cannot convert MediaRef to VideoFrame: pts_ns is None (lerobot's VideoFrame requires a timestamp)."
        )
    return {"path": ref.uri, "timestamp": ref.pts_ns / NANOSECOND}


def lerobot_episode_to_refs(
    *,
    video_path: str,
    from_timestamp: float,
    frame_timestamps: Iterable[float],
) -> list[MediaRef]:
    """Build MediaRefs for one episode's frames in a v3.0 LeRobotDataset.

    LeRobotDataset v3.0 shards multiple episodes into a single ``.mp4``;
    each frame's true presentation time inside that mp4 is
    ``from_timestamp + episode_local_timestamp``.

    Args:
        video_path: Path or URI to the shared ``.mp4`` shard. Resolve via
            ``LeRobotDatasetMetadata.get_video_file_path(ep_idx, vid_key)``.
        from_timestamp: Episode start offset in seconds within the shared
            mp4. Read from
            ``meta.episodes[ep_idx][f"videos/{vid_key}/from_timestamp"]``.
        frame_timestamps: Per-frame timestamps in seconds *relative to the
            episode start* (i.e. the values in the parquet ``timestamp``
            column for that episode's rows).

    Returns:
        A list of MediaRefs, one per frame, each pointing to the same
        ``video_path`` with absolute ``pts_ns`` inside that mp4.
    """
    base_ns = _sec_to_ns(from_timestamp)
    return [MediaRef(uri=video_path, pts_ns=base_ns + _sec_to_ns(t)) for t in frame_timestamps]


__all__ = ["from_videoframe", "to_videoframe", "lerobot_episode_to_refs"]
