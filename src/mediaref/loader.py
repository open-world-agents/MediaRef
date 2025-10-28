"""Batch loading utilities for MediaRef."""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from . import cached_av
from .video.reader import BatchDecodingStrategy, VideoReader

if TYPE_CHECKING:
    from .core import MediaRef

NANOSECOND = 1_000_000_000  # 1 second in nanoseconds


def load_batch(
    refs: list["MediaRef"],
    strategy: BatchDecodingStrategy = BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK,
    **kwargs,
) -> list[np.ndarray]:
    """Load multiple MediaRef objects efficiently using optimized batch decoding.

    This function groups MediaRefs by video file and uses PyAV's batch decoding API
    to decode multiple frames in one pass, which is much more efficient than loading
    each frame separately.

    Args:
        refs: List of MediaRef objects to load
        strategy: Batch decoding strategy (SEPARATE, SEQUENTIAL_PER_KEYFRAME_BLOCK, or SEQUENTIAL)
        **kwargs: Additional options passed to to_rgb_array() for image loading

    Returns:
        List of RGB numpy arrays in the same order as input refs

    Raises:
        TypeError: If refs is not a list or contains invalid types
        ValueError: If refs contains None values or video refs missing pts_ns
        ValueError: If batch loading fails for any video

    Examples:
        >>> from mediaref import MediaRef, load_batch
        >>>
        >>> # Load multiple frames from same video efficiently
        >>> refs = [
        ...     MediaRef(uri="video.mp4", pts_ns=0),
        ...     MediaRef(uri="video.mp4", pts_ns=1_000_000_000),
        ...     MediaRef(uri="video.mp4", pts_ns=2_000_000_000),
        ... ]
        >>> frames = load_batch(refs)
        >>>
        >>> # Also works with mixed images and videos
        >>> refs = [
        ...     MediaRef(uri="image1.png"),
        ...     MediaRef(uri="video.mp4", pts_ns=0),
        ...     MediaRef(uri="image2.png"),
        ... ]
        >>> media = load_batch(refs)
    """
    # Input validation
    if not refs:
        return []

    if not isinstance(refs, list):
        raise TypeError(f"refs must be a list, got {type(refs).__name__}")

    if any(ref is None for ref in refs):
        raise ValueError("refs list contains None values")

    if not isinstance(strategy, BatchDecodingStrategy):
        raise TypeError(f"strategy must be BatchDecodingStrategy, got {type(strategy).__name__}")

    # Group refs by video file for efficient batch loading
    video_groups = defaultdict(list)
    image_refs = []

    for i, ref in enumerate(refs):
        if ref.is_video:
            video_groups[ref.uri].append((i, ref))
        else:
            image_refs.append((i, ref))

    # Prepare results array
    results: list[np.ndarray | None] = [None] * len(refs)

    # Load images (no batching needed)
    for i, ref in image_refs:
        results[i] = ref.to_rgb_array(**kwargs)

    # Load video frames using optimized batch decoding
    for uri, group in video_groups.items():
        # Extract timestamps and original indices
        indices = [i for i, _ in group]

        # Validate pts_ns and convert to seconds
        pts_seconds = []
        for _, ref in group:
            if ref.pts_ns is None:
                raise ValueError(f"Video reference missing pts_ns: {ref.uri}")
            pts_seconds.append(ref.pts_ns / NANOSECOND)

        # Use VideoReader for batch decoding
        try:
            with VideoReader(uri, keep_av_open=True) as reader:
                av_frames = reader.get_frames_played_at(pts_seconds, strategy=strategy)

                # Convert AV frames to RGB arrays
                for idx, av_frame in zip(indices, av_frames):
                    rgb_array = av_frame.to_ndarray(format="rgb24")
                    results[idx] = rgb_array
        except Exception as e:
            raise ValueError(f"Failed to load batch from '{uri}': {e}") from e

    return results  # type: ignore[return-value]


def cleanup_cache():
    """Clear all cached video containers from memory.

    This function should be called when you're done with batch loading
    to free up resources. It's automatically called on process exit.

    Examples:
        >>> from mediaref import MediaRef, load_batch, cleanup_cache
        >>>
        >>> # Load many frames
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i * 1_000_000_000) for i in range(100)]
        >>> frames = load_batch(refs)
        >>>
        >>> # Clean up when done
        >>> cleanup_cache()
    """
    cached_av.cleanup_cache()
