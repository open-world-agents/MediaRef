"""Batch loading utilities for MediaRef."""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, List, Literal, Optional, Type

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .core import MediaRef
    from .video_decoder import BaseVideoDecoder

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
                "Install it separately: pip install torchcodec"
            ) from e
    else:
        raise ValueError(f"Unknown decoder backend: {backend}. Must be 'pyav' or 'torchcodec'")


def batch_decode(
    refs: list["MediaRef"],
    decoder: DecoderBackend = "pyav",
    **kwargs,
) -> list[npt.NDArray[np.uint8]]:
    """Decode multiple media references efficiently using batch decoding.

    Groups video frames by file and decodes them in one pass for efficiency.
    Images are decoded individually.

    Args:
        refs: List of MediaRef objects to decode
        decoder: Decoder backend ('pyav' or 'torchcodec'). Default: 'pyav'
        **kwargs: Additional options passed to to_ndarray() for image loading

    Returns:
        List of RGB numpy arrays in the same order as input refs

    Examples:
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i*1_000_000_000) for i in range(3)]
        >>> frames = batch_decode(refs)
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
            f"Consider using ref.to_ndarray() directly for images.",
            UserWarning,
            stacklevel=2,
        )
    for i, ref in image_refs:
        results[i] = ref.to_ndarray(**kwargs)

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
                batch = video_decoder.get_frames_played_at(pts_seconds)

                # Convert from NCHW to HWC format
                for idx, frame_nchw in zip(indices, batch.data):
                    # Transpose from (C, H, W) to (H, W, C) - decoder outputs RGB
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
