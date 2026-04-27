"""Internal loading and encoding utilities."""

import io
import os
from pathlib import Path
from typing import Union

import fsspec
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps

# Constants
NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# URI schemes MediaRef handles directly (without fsspec). Anything else with a
# scheme — including http(s):// — is delegated to fsspec.open(), which dispatches
# to the appropriate backend (HTTPFileSystem, S3FileSystem, GCSFileSystem,
# HfFileSystem, …). All backends return seekable file-like objects that map
# seek/read to HTTP Range requests, enabling sparse video frame access without
# downloading the full file.
_DIRECT_URI_SCHEMES = frozenset({"file", "data"})


def _scheme_of(uri: str) -> str:
    """Lowercased URI scheme (the part before ``://``), or ``""`` for paths."""
    scheme, sep, _ = uri.partition("://")
    return scheme.lower() if sep else ""


def is_cloud_uri(uri: str) -> bool:
    """True if ``uri`` is delegated to fsspec.

    Open-set: any URI whose scheme is not in :data:`_DIRECT_URI_SCHEMES`
    (``file``, ``data``) and is not a bare path is treated as fsspec-routable.
    This includes ``http(s)://`` (handled by fsspec's HTTPFileSystem +
    aiohttp backend) and any cloud scheme whose backend is installed
    (``s3fs`` for ``s3://``, ``gcsfs`` for ``gs://``, ``huggingface_hub``
    for ``hf://``, …). Backend missing → fsspec raises a clear error.
    """
    scheme = _scheme_of(uri)
    return bool(scheme) and scheme not in _DIRECT_URI_SCHEMES


def open_cloud(uri: str):
    """Open a fsspec-routed URI as a binary file-like (context manager)."""
    return fsspec.open(uri, "rb")


def _file_uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a local filesystem path."""
    from urllib.request import url2pathname

    # url2pathname handles URL decoding (unquote) internally
    return url2pathname(uri[7:])


# ============================================================================
# Image Loading
# ============================================================================


def load_image_as_rgba(path_or_uri: str) -> npt.NDArray[np.uint8]:
    """Load image from any source and return as RGBA numpy array.

    Args:
        path_or_uri: File path, URL, or data URI

    Returns:
        RGBA numpy array

    Raises:
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        if path_or_uri.startswith("data:"):
            from .data_uri import DataURI

            return DataURI.from_uri(path_or_uri).to_ndarray(format="rgba")
        else:
            # Load as PIL image and convert to RGBA
            pil_image = _load_pil_image(path_or_uri)
            return np.array(pil_image.convert("RGBA"))
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {path_or_uri}: {e}") from e


def _load_pil_image(
    image: Union[str, PIL.Image.Image],
) -> PIL.Image.Image:
    """Load image to PIL Image."""
    if isinstance(image, str):
        if is_cloud_uri(image):
            # PIL needs to seek (magic-byte detection); fsspec's HTTPFile is
            # streaming-only when the server doesn't advertise byte-range
            # support. Materializing bytes once keeps image loading robust
            # across all backends (s3, gs, hf, http(s) chunked, …).
            with open_cloud(image) as f:
                data = f.read()
            image = PIL.Image.open(io.BytesIO(data))
            image.load()
        elif image.startswith("file://"):
            file_path = _file_uri_to_path(image)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path} (from URI: {image})")
            image = PIL.Image.open(file_path)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(f"Not a valid local path or supported URI: {image}")
    elif isinstance(image, PIL.Image.Image):
        pass
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    # Handle EXIF orientation
    image = PIL.ImageOps.exif_transpose(image)

    # Convert to RGBA
    image = image.convert("RGBA")

    return image


# ============================================================================
# Video Loading
# ============================================================================


def load_video_frame_as_rgba(
    path_or_url: str,
    pts_ns: int,
) -> npt.NDArray[np.uint8]:
    """Load video frame and return as RGBA numpy array.

    Args:
        path_or_url: File path or URL to video
        pts_ns: Presentation timestamp in nanoseconds

    Returns:
        RGBA numpy array

    Raises:
        ImportError: If video dependencies are not installed
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    from .video_decoder import PyAVVideoDecoder

    pts_seconds = pts_ns / NANOSECOND

    def _decode(source) -> npt.NDArray[np.uint8]:
        with PyAVVideoDecoder(source) as decoder:
            batch = decoder.get_frames_played_at([pts_seconds])
            # batch.data is NCHW format (1, 3, H, W) with RGB channels
            rgb_nchw = batch.data[0]  # (3, H, W)
            rgb_hwc = np.transpose(rgb_nchw, (1, 2, 0))  # (H, W, 3)
            alpha = np.full((*rgb_hwc.shape[:2], 1), 255, dtype=np.uint8)
            return np.concatenate([rgb_hwc, alpha], axis=2)

    try:
        # fsspec-routed URIs (http(s)://, s3://, gs://, hf://, …): open as a
        # file-like, hand to PyAV. fsspec backends provide seekable file-likes
        # that map seek to HTTP Range requests, so sparse frame access reads
        # only the byte ranges PyAV asks for. The cached_av cache cannot retain
        # file-like objects across calls, so cross-call container caching is
        # forfeited for fsspec URIs; within a single batch_decode call the same
        # handle is reused as expected.
        if is_cloud_uri(path_or_url):
            with open_cloud(path_or_url) as f:
                return _decode(f)

        # Local file: file:// URI or bare path.
        local_path = _file_uri_to_path(path_or_url) if path_or_url.startswith("file://") else path_or_url
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")
        return _decode(local_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e
