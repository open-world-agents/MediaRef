"""Batch loading utilities for MediaRef."""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from . import cached_av

if TYPE_CHECKING:
    from .core import MediaRef


def load_batch(refs: list["MediaRef"], **kwargs) -> list[np.ndarray]:
    """Load multiple MediaRef objects efficiently with automatic caching.

    This function groups MediaRefs by video file and loads frames in batches,
    which is much more efficient than loading each frame separately.

    Args:
        refs: List of MediaRef objects to load
        **kwargs: Additional options passed to to_rgb_array()

    Returns:
        List of RGB numpy arrays in the same order as input refs

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
    if not refs:
        return []

    # Group refs by video file for efficient batch loading
    video_groups = defaultdict(list)
    image_refs = []

    for i, ref in enumerate(refs):
        if ref.is_video:
            video_groups[ref.uri].append((i, ref))
        else:
            image_refs.append((i, ref))

    # Prepare results array
    results = [None] * len(refs)

    # Load images (no batching needed)
    for i, ref in image_refs:
        results[i] = ref.to_rgb_array(**kwargs)

    # Load video frames with caching enabled
    for uri, group in video_groups.items():
        # Enable caching for batch loading
        batch_kwargs = {**kwargs, "keep_av_open": True}

        for i, ref in group:
            results[i] = ref.to_rgb_array(**batch_kwargs)

    return results


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
