"""Internal loading and encoding utilities."""

import os
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps
import requests

# Constants
REQUEST_TIMEOUT = 60  # HTTP request timeout in seconds
NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Schemes routed through fsspec when available. http(s) and file:// are handled
# directly by requests / pathlib so they're intentionally excluded; data: is its
# own thing. New schemes can be added here as needed (cf. SPEC §2.1).
_CLOUD_URI_SCHEMES = frozenset(
    {
        "s3",
        "gs",
        "gcs",
        "hf",
        "az",
        "azure",
        "abfs",
        "abfss",
        "adl",
        "r2",
        "ftp",
        "sftp",
        "ssh",
        "memory",  # fsspec's in-memory filesystem (testing & ephemeral pipelines)
    }
)


def _scheme_of(uri: str) -> str:
    """Lowercased URI scheme (the part before ``://``), or ``""`` for paths."""
    scheme, sep, _ = uri.partition("://")
    return scheme.lower() if sep else ""


def is_cloud_uri(uri: str) -> bool:
    """True if ``uri`` uses a scheme delegated to fsspec.

    Schemes covered: ``s3://``, ``gs://``, ``gcs://``, ``hf://``, ``az://``,
    ``azure://``, ``abfs(s)://``, ``adl://``, ``r2://``, ``ftp://``, ``sftp://``,
    ``ssh://``, ``memory://``.

    ``http(s)://``, ``file://``, ``data:`` and POSIX paths are NOT cloud URIs —
    they are handled directly without fsspec.
    """
    return _scheme_of(uri) in _CLOUD_URI_SCHEMES


def _require_fsspec(uri: str):
    """Import fsspec or raise a focused ImportError naming the offending scheme."""
    try:
        import fsspec  # noqa: PLC0415 — lazy import; optional dependency
    except ImportError as e:  # pragma: no cover — exercised in test_fsspec
        raise ImportError(
            f"Loading from {_scheme_of(uri) or uri}:// requires the fsspec optional dependency. "
            "Install with: pip install 'mediaref[fsspec]'\n"
            "For specific cloud backends you may also need: "
            "s3fs (s3://), gcsfs (gs://), huggingface_hub[hf_transfer] (hf://), "
            "adlfs (az://, abfs://). See https://filesystem-spec.readthedocs.io."
        ) from e
    return fsspec


def open_cloud(uri: str):
    """Open a cloud URI as a binary file-like via fsspec (use as context manager).

    Wraps the fsspec import + ``fsspec.open(uri, "rb")`` boilerplate that
    image loading, video loading, and batch decoding all share. Raises a
    focused :class:`ImportError` when fsspec or the relevant backend isn't
    installed.
    """
    return _require_fsspec(uri).open(uri, "rb")


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
    """Load image to PIL Image.

    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/loading_utils.py
    """
    if isinstance(image, str):
        if is_cloud_uri(image):
            with open_cloud(image) as f:
                image = PIL.Image.open(f)
                image.load()  # force the bytes through PIL before f closes
        elif image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True, timeout=REQUEST_TIMEOUT).raw)
        elif image.startswith("file://"):
            # Convert file:// URI to local path
            from urllib.request import url2pathname

            # Remove 'file://' prefix and convert to local path
            # url2pathname handles URL decoding (unquote) internally
            file_path = url2pathname(image[7:])
            if os.path.isfile(file_path):
                image = PIL.Image.open(file_path)
            else:
                raise FileNotFoundError(f"File not found: {file_path} (from URI: {image})")
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://`, `https://`, or `file://`, "
                f"and {image} is not a valid path."
            )
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
        # Cloud URIs (s3://, gs://, hf://, …) — open via fsspec as a file-like
        # and hand that to PyAV. The cached_av cache cannot retain file-like
        # objects across calls, so cloud-backed videos do not benefit from
        # cross-call caching; within a single batch_decode the same handle is
        # reused as expected.
        if is_cloud_uri(path_or_url):
            with open_cloud(path_or_url) as f:
                return _decode(f)

        # file:// URI → local path
        actual_path = path_or_url
        if path_or_url.startswith("file://"):
            from urllib.request import url2pathname

            actual_path = url2pathname(path_or_url[7:])

        # Validate local file exists (skip for http(s)://, which PyAV opens directly)
        if not path_or_url.startswith(("http://", "https://")):
            if not Path(actual_path).exists():
                raise FileNotFoundError(f"Video file not found: {actual_path}")

        return _decode(actual_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e
