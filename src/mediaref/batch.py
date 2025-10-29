"""Batch loading utilities for MediaRef."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, List, Literal, Optional, Type

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .core import MediaRef
    from .video_decoder import BaseVideoDecoder
    from .video_decoder.types import BatchDecodingStrategy

NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Type alias for decoder backend selection
DecoderBackend = Literal["pyav", "torchcodec"]


def _get_decoder_class(backend: DecoderBackend) -> Type["BaseVideoDecoder"]:
    """Get decoder class for the specified backend."""
    if backend == "pyav":
        from .video_decoder import PyAVVideoDecoder

        return PyAVVideoDecoder
    elif backend == "torchcodec":
        try:
            from .video_decoder import TorchCodecVideoDecoder

            return TorchCodecVideoDecoder
        except ImportError as e:
            raise ImportError(
                "TorchCodec decoder requested but torchcodec is not installed. "
                "Install it with: pip install torchcodec>=0.4.0"
            ) from e
    else:
        raise ValueError(f"Unknown decoder backend: {backend}. Must be 'pyav' or 'torchcodec'")


def batch_decode(
    refs: list["MediaRef"],
    strategy: "BatchDecodingStrategy | None" = None,
    decoder: DecoderBackend = "pyav",
    **kwargs,
) -> list[npt.NDArray[np.uint8]]:
    """Decode multiple media references efficiently using batch decoding.

    For images, decodes them individually. For video frames from the same video,
    uses optimized batch decoding API to decode multiple frames in one pass.

    This function groups MediaRefs by video file and uses the specified decoder's
    batch decoding API to decode multiple frames in one pass, which is much more
    efficient than loading each frame separately.

    Args:
        refs: List of MediaRef objects to load
        strategy: Batch decoding strategy (SEPARATE, SEQUENTIAL_PER_KEYFRAME_BLOCK, or SEQUENTIAL).
            Only supported by PyAV decoder. TorchCodec decoder ignores this parameter.
        decoder: Video decoder backend to use ('pyav' or 'torchcodec'). Default: 'pyav'
        **kwargs: Additional options passed to to_rgb_array() for image loading

    Returns:
        List of RGB numpy arrays in the same order as input refs

    Raises:
        TypeError: If refs is not a list or contains invalid types
        ValueError: If refs contains None values or video refs missing pts_ns
        ValueError: If batch loading fails for any video
        ImportError: If torchcodec decoder is requested but not installed

    Examples:
        >>> from mediaref import MediaRef, batch_decode
        >>> from mediaref.video_decoder import BatchDecodingStrategy
        >>>
        >>> # Decode multiple frames from same video efficiently (default PyAV decoder)
        >>> refs = [
        ...     MediaRef(uri="video.mp4", pts_ns=0),
        ...     MediaRef(uri="video.mp4", pts_ns=1_000_000_000),
        ...     MediaRef(uri="video.mp4", pts_ns=2_000_000_000),
        ... ]
        >>> frames = batch_decode(refs)
        >>>
        >>> # Use batch decoding strategy (PyAV only)
        >>> frames = batch_decode(refs, strategy=BatchDecodingStrategy.SEQUENTIAL)
        >>>
        >>> # Use TorchCodec decoder for GPU acceleration (strategy ignored)
        >>> frames = batch_decode(refs, decoder="torchcodec")
        >>>
        >>> # Also works with mixed images and videos
        >>> refs = [
        ...     MediaRef(uri="image1.png"),
        ...     MediaRef(uri="video.mp4", pts_ns=0),
        ...     MediaRef(uri="image2.png"),
        ... ]
        >>> media = batch_decode(refs)
    """
    # Input validation
    if not refs:
        return []

    if not isinstance(refs, list):
        raise TypeError(f"refs must be a list, got {type(refs).__name__}")

    if any(ref is None for ref in refs):
        raise ValueError("refs list contains None values")

    # Get the decoder class for the specified backend
    decoder_class = _get_decoder_class(decoder)

    # Group refs by video file for efficient batch loading
    video_groups = defaultdict(list)
    image_refs: list[tuple[int, "MediaRef"]] = []

    for i, ref in enumerate(refs):
        if ref.is_video:
            video_groups[ref.uri].append((i, ref))
        else:
            image_refs.append((i, ref))

    # Prepare results array
    results: List[Optional[npt.NDArray[np.uint8]]] = [None] * len(refs)

    # Load images (no batching needed)
    if image_refs:
        warnings.warn(
            f"batch_decode() received {len(image_refs)} image reference(s). "
            f"Batch decoding is only optimized for video frames. "
            f"Images will be decoded individually. "
            f"Consider using ref.to_rgb_array() directly for images.",
            UserWarning,
            stacklevel=2,
        )
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

        # Use selected decoder for batch decoding
        try:
            with decoder_class(uri) as video_decoder:
                # Get frames as FrameBatch
                # Pass strategy only if explicitly provided
                if strategy is not None:
                    batch = video_decoder.get_frames_played_at(pts_seconds, strategy=strategy)
                else:
                    batch = video_decoder.get_frames_played_at(pts_seconds)

                # Convert from NCHW to HWC format
                for idx, frame_nchw in zip(indices, batch.data):
                    # Transpose from (C, H, W) to (H, W, C)
                    rgb_array = np.transpose(frame_nchw, (1, 2, 0))
                    results[idx] = rgb_array
        except ImportError:
            # Re-raise ImportError for missing decoder dependencies
            raise
        except Exception as e:
            raise ValueError(f"Failed to load batch from '{uri}': {e}") from e

    return results  # type: ignore[return-value]


def cleanup_cache():
    """Clear all cached video containers from memory.

    This function should be called when you're done with batch decoding
    to free up resources. It's automatically called on process exit.

    Examples:
        >>> from mediaref import MediaRef, batch_decode, cleanup_cache
        >>>
        >>> # Decode many frames
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i * 1_000_000_000) for i in range(100)]
        >>> frames = batch_decode(refs)
        >>>
        >>> # Clean up when done
        >>> cleanup_cache()
    """
    from . import cached_av

    cached_av.cleanup_cache()
