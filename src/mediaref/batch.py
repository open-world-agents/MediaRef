"""Batch loading utilities for MediaRef."""

from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Type

import numpy as np
import numpy.typing as npt

from ._internal import NANOSECOND, resolve_video_source

if TYPE_CHECKING:
    from .core import MediaRef
    from .video_decoder import BaseVideoDecoder

# Type alias for decoder backend selection
DecoderBackend = Literal["pyav", "torchcodec"]


def _split_by_gap(
    indices: list[int],
    pts_seconds: list[float],
    gap_threshold: float,
) -> list[tuple[list[int], list[float]]]:
    """Split timestamps into contiguous chunks at gaps exceeding *gap_threshold*.

    Returns a list of ``(indices, pts_seconds)`` tuples — one per chunk — sorted by time within each chunk.
    """
    if len(pts_seconds) <= 1:
        return [(indices, pts_seconds)]

    paired = sorted(zip(indices, pts_seconds), key=lambda x: x[1])

    chunks: list[tuple[list[int], list[float]]] = [([paired[0][0]], [paired[0][1]])]
    for i in range(1, len(paired)):
        if paired[i][1] - paired[i - 1][1] > gap_threshold:
            chunks.append(([], []))
        chunks[-1][0].append(paired[i][0])
        chunks[-1][1].append(paired[i][1])
    return chunks


def _decode_video_chunks(
    decoder_class: Type["BaseVideoDecoder"],
    uri: str,
    chunks: list[list[float]],
) -> list[list[npt.NDArray[np.uint8]]]:
    """Decode multiple timestamp chunks from one source with a single decoder open.

    Each chunk is decoded via a separate ``get_frames_played_at`` call so the decoder can seek past
    large gaps instead of decoding every intermediate frame.  Returns one list of RGB HWC frames per
    chunk, in input order.
    """
    source = resolve_video_source(uri)
    with decoder_class(source) as video_decoder:
        return [
            [np.transpose(f, (1, 2, 0)) for f in video_decoder.get_frames_played_at(chunk).data] for chunk in chunks
        ]


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
    *,
    allow_images: bool = False,
    allow_multiple_videos: bool = False,
    gap_threshold: float = 2.0,
    allow_gap: bool = True,
    **kwargs,
) -> list[npt.NDArray[np.uint8]]:
    """Decode multiple media references efficiently using batch decoding.

    Groups video frames by file and decodes them in one pass for efficiency.  When timestamps within
    a single video have large gaps, the decoder automatically splits them into contiguous chunks and
    seeks between them instead of decoding all intermediate frames.

    Args:
        refs: List of MediaRef objects to decode.
        decoder: Decoder backend (``'pyav'`` or ``'torchcodec'``).
        allow_images: If ``True``, image refs are accepted and decoded individually.
            If ``False`` (default), image refs raise ``ValueError``.
        allow_multiple_videos: If ``True``, refs may span multiple video files.
            If ``False`` (default), more than one video URI raises ``ValueError``.
        gap_threshold: Minimum gap in seconds between consecutive sorted timestamps that triggers
            chunking (or an error when *allow_gap* is ``False``).
        allow_gap: If ``True`` (default), gaps exceeding *gap_threshold* cause automatic chunk
            splitting for efficient decoding.  If ``False``, such gaps raise ``ValueError``.
        **kwargs: Additional options forwarded to ``to_ndarray()`` for image loading
            (only used when *allow_images* is ``True``).

    Returns:
        List of RGB numpy arrays in the same order as *refs*.

    Raises:
        ValueError: When a constraint (*allow_images*, *allow_multiple_videos*, or *allow_gap*)
            is violated.

    Examples:
        >>> refs = [MediaRef(uri="video.mp4", pts_ns=i * 1_000_000_000) for i in range(3)]
        >>> frames = batch_decode(refs)

        >>> # Strict mode: single video, no large gaps
        >>> frames = batch_decode(refs, allow_gap=False)
    """
    if not refs:
        return []

    if not isinstance(refs, list):
        raise TypeError(f"refs must be a list, got {type(refs).__name__}")

    if any(ref is None for ref in refs):
        raise ValueError("refs list contains None values")

    decoder_class = _get_decoder_class(decoder)

    video_groups: dict[str, list[tuple[int, "MediaRef"]]] = defaultdict(list)
    image_refs: list[tuple[int, "MediaRef"]] = []

    for i, ref in enumerate(refs):
        if ref.is_video:
            video_groups[ref.uri].append((i, ref))
        else:
            image_refs.append((i, ref))

    if image_refs and not allow_images:
        raise ValueError(
            f"batch_decode() received {len(image_refs)} image reference(s) "
            f"but allow_images=False. Pass allow_images=True to decode images, "
            f"or use ref.to_ndarray() directly."
        )

    if len(video_groups) > 1 and not allow_multiple_videos:
        uris = list(video_groups.keys())
        raise ValueError(
            f"batch_decode() received refs from {len(uris)} different video files "
            f"but allow_multiple_videos=False. URIs: {uris}"
        )

    results: list[npt.NDArray[np.uint8] | None] = [None] * len(refs)

    # Decode images individually
    for i, ref in image_refs:
        results[i] = ref.to_ndarray(**kwargs)

    # Decode video frames with gap-aware chunking
    for uri, group in video_groups.items():
        indices = [i for i, _ in group]

        pts_seconds: list[float] = []
        for _, ref in group:
            if ref.pts_ns is None:
                raise ValueError(f"Video reference missing pts_ns: {ref.uri}")
            pts_seconds.append(ref.pts_ns / NANOSECOND)

        chunks = _split_by_gap(indices, pts_seconds, gap_threshold)

        if len(chunks) > 1 and not allow_gap:
            max_gap = max(chunks[j + 1][1][0] - chunks[j][1][-1] for j in range(len(chunks) - 1))
            raise ValueError(
                f"Timestamps for '{uri}' have a gap of {max_gap:.1f}s "
                f"(threshold: {gap_threshold}s) but allow_gap=False."
            )

        try:
            chunk_pts = [pts for _, pts in chunks]
            decoded = _decode_video_chunks(decoder_class, uri, chunk_pts)
            for (chunk_indices, _), frames in zip(chunks, decoded):
                for idx, frame in zip(chunk_indices, frames):
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

    No-op when the ``[video]`` extra isn't installed (nothing was ever
    cached). Calling this without ``av`` MUST NOT raise — callers may
    invoke it defensively without knowing which extras are present.

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
    try:
        from . import cached_av
    except ImportError:
        return
    cached_av.cleanup_cache()
