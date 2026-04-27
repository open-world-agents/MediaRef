"""Internal loading and encoding utilities."""

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union
from urllib.request import url2pathname

import fsspec
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps

NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# URI schemes MediaRef handles directly. Anything else with a scheme is
# delegated to fsspec.open(), which dispatches to the appropriate backend
# (HTTPFileSystem, S3FileSystem, GCSFileSystem, HfFileSystem, …). All
# backends return seekable file-likes that map seek to HTTP Range requests,
# enabling sparse video frame access without downloading the full file.
_DIRECT_URI_SCHEMES = frozenset({"file", "data"})
_FILE_URI_PREFIX = "file://"


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
    """Convert a ``file://`` URI to a local filesystem path (handles URL decoding)."""
    return url2pathname(uri[len(_FILE_URI_PREFIX) :])


def _resolve_to_local_path(uri: str) -> str:
    """Resolve a ``file://`` URI or bare path to an existing local path.

    Raises:
        FileNotFoundError: If the resolved path doesn't exist.
    """
    path = _file_uri_to_path(uri) if uri.startswith(_FILE_URI_PREFIX) else uri
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


@contextmanager
def open_media_source(uri: str) -> Iterator:
    """Yield whatever the decoder layer can consume directly.

    For fsspec-routed URIs (cloud, http(s), …): an open file-like.
    For ``file://`` URIs and bare paths: a verified local path string.

    Raises:
        FileNotFoundError: For local paths that don't exist.
    """
    if is_cloud_uri(uri):
        with open_cloud(uri) as f:
            yield f
    else:
        yield _resolve_to_local_path(uri)


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
        pil_image = _load_pil_image(path_or_uri)
        return np.array(pil_image.convert("RGBA"))
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {path_or_uri}: {e}") from e


def _load_pil_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """Load image to PIL Image."""
    if isinstance(image, str):
        if is_cloud_uri(image):
            # PIL needs seek (magic-byte detection); fsspec's HTTPFile is
            # streaming-only when the server doesn't advertise byte-range
            # support. Materializing bytes once keeps loading robust across
            # all backends (s3, gs, hf, http(s) chunked, …).
            with open_cloud(image) as f:
                data = f.read()
            image = PIL.Image.open(io.BytesIO(data))
            image.load()
        else:
            image = PIL.Image.open(_resolve_to_local_path(image))
    elif not isinstance(image, PIL.Image.Image):
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)
    return image.convert("RGBA")


# ============================================================================
# Video Loading
# ============================================================================


def load_video_frame_as_rgba(path_or_url: str, pts_ns: int) -> npt.NDArray[np.uint8]:
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

    try:
        with open_media_source(path_or_url) as source, PyAVVideoDecoder(source) as decoder:
            batch = decoder.get_frames_played_at([pts_seconds])
            rgb_nchw = batch.data[0]
            rgb_hwc = np.transpose(rgb_nchw, (1, 2, 0))
            alpha = np.full((*rgb_hwc.shape[:2], 1), 255, dtype=np.uint8)
            return np.concatenate([rgb_hwc, alpha], axis=2)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e
