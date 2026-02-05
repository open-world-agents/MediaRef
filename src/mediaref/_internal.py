"""Internal loading and encoding utilities."""

import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps
import requests

if TYPE_CHECKING:
    import av

# Constants
REQUEST_TIMEOUT = 60  # HTTP request timeout in seconds
NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Garbage collection for PyAV reference cycles
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


# ============================================================================
# PyAV Frame Conversion
# ============================================================================


def _frame_to_rgba(frame: "av.VideoFrame") -> npt.NDArray[np.uint8]:
    """Convert PyAV frame to RGBA numpy array.

    NOTE: Convert ARGB to RGBA manually instead of using `to_ndarray(format="rgba")`.
    Direct RGBA conversion causes memory corruption on certain videos:
      Error: "malloc_consolidate(): invalid chunk size; Fatal Python error: Aborted"
      Example: https://huggingface.co/datasets/open-world-agents/example_dataset/resolve/main/example.mkv (pts_ns=1_000_000_000)
    Possibly related to:
      - PyAV issue: https://github.com/PyAV-Org/PyAV/issues/1269
      - FFmpeg ticket: https://trac.ffmpeg.org/ticket/9254

    Args:
        frame: PyAV VideoFrame to convert

    Returns:
        RGBA numpy array (H, W, 4) with uint8 dtype
    """
    argb_array = frame.to_ndarray(format="argb")
    # ARGB format stores channels as [A, R, G, B], so we reorder to [R, G, B, A]
    rgba_array: npt.NDArray[np.uint8] = argb_array[:, :, [1, 2, 3, 0]]
    return rgba_array


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
        if image.startswith("http://") or image.startswith("https://"):
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

    global _CALLED_TIMES
    _CALLED_TIMES += 1
    if _CALLED_TIMES % GC_COLLECTION_INTERVAL == 0:
        gc.collect()

    try:
        # Convert file:// URI to local path if needed
        actual_path = path_or_url
        if path_or_url.startswith("file://"):
            from urllib.request import url2pathname

            # url2pathname handles URL decoding (unquote) internally
            actual_path = url2pathname(path_or_url[7:])

        # Validate local file exists
        if not path_or_url.startswith(("http://", "https://")):
            if not Path(actual_path).exists():
                raise FileNotFoundError(f"Video file not found: {actual_path}")

        # Convert nanoseconds to seconds
        pts_seconds = pts_ns / NANOSECOND

        # Use PyAVVideoDecoder for consistent playback semantics with batch_decode
        with PyAVVideoDecoder(actual_path) as decoder:
            batch = decoder.get_frames_played_at([pts_seconds])
            # batch.data is NCHW format (1, 3, H, W) with RGB channels
            # Convert to RGBA HWC format (H, W, 4)
            rgb_nchw = batch.data[0]  # (3, H, W)
            rgb_hwc = np.transpose(rgb_nchw, (1, 2, 0))  # (H, W, 3)
            # Add alpha channel (fully opaque)
            alpha = np.full((*rgb_hwc.shape[:2], 1), 255, dtype=np.uint8)
            rgba_hwc: npt.NDArray[np.uint8] = np.concatenate([rgb_hwc, alpha], axis=2)
            return rgba_hwc
    except FileNotFoundError:
        raise
    except Exception as e:
        pts_seconds = pts_ns / NANOSECOND
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e
