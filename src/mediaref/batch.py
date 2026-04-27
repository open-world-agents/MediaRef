"""Batch loading utilities for MediaRef."""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, List, Literal, Optional, Type

import numpy as np
import numpy.typing as npt

from ._internal import open_media_source

if TYPE_CHECKING:
    from .core import MediaRef
    from .video_decoder import BaseVideoDecoder

NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Type alias for decoder backend selection
DecoderBackend = Literal["pyav", "torchcodec"]


def _decode_video_group(
    decoder_class: Type["BaseVideoDecoder"],
    source,
    pts_seconds: List[float],
) -> List[npt.NDArray[np.uint8]]:
    """Decode all timestamps from one source. Returns RGB HWC frames in input order."""
    with decoder_class(source) as video_decoder:
        batch = video_decoder.get_frames_played_at(pts_seconds)
        return [np.transpose(f, (1, 2, 0)) for f in batch.data]


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
        indices = [i for i, _ in group]

        # Validate pts_ns and convert to seconds
        pts_seconds = []
        for _, ref in group:
            if ref.pts_ns is None:
                raise ValueError(f"Video reference missing pts_ns: {ref.uri}")
            pts_seconds.append(ref.pts_ns / NANOSECOND)

        try:
            # open_media_source yields a file-like for fsspec URIs (one
            # range-served handle reused across the group's timestamps; cached_av
            # cannot retain file-likes so cross-call caching is forfeited) or a
            # verified local path string for file://-and-bare-paths.
            with open_media_source(uri) as source:
                frames = _decode_video_group(decoder_class, source, pts_seconds)
            for idx, frame in zip(indices, frames):
                results[idx] = frame
        except ImportError:
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
