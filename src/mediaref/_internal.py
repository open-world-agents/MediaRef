"""Internal loading and encoding utilities."""

import gc
import os
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps
import requests

if TYPE_CHECKING:
    import av
    import av.container

# Constants
REQUEST_TIMEOUT = 60  # HTTP request timeout in seconds
NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Garbage collection for PyAV reference cycles
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


# ============================================================================
# Image Loading
# ============================================================================


def load_image_as_bgra(path_or_uri: str) -> npt.NDArray[np.uint8]:
    """Load image from any source and return as BGRA numpy array.

    Args:
        path_or_uri: File path, URL, or data URI

    Returns:
        BGRA numpy array

    Raises:
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    try:
        if path_or_uri.startswith("data:"):
            return _load_from_data_uri(path_or_uri)
        else:
            # Load as PIL image and convert to BGRA
            pil_image = _load_pil_image(path_or_uri)
            return _pil_to_bgra_array(pil_image)
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

    # Convert to RGB
    image = image.convert("RGB")

    return image


def _pil_to_bgra_array(pil_image: PIL.Image.Image) -> npt.NDArray[np.uint8]:
    """Convert PIL image to BGRA numpy array."""
    # Ensure image is in RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array and then to BGRA
    rgb_array = np.array(pil_image, dtype=np.uint8)
    bgra_array: npt.NDArray[np.uint8] = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)  # type: ignore[assignment]
    return bgra_array


# ============================================================================
# Video Loading
# ============================================================================


def load_video_frame_as_bgra(
    path_or_url: str,
    pts_ns: int,
    *,
    keep_av_open: bool = False,
) -> npt.NDArray[np.uint8]:
    """Load video frame and return as BGRA numpy array.

    Args:
        path_or_url: File path or URL to video
        pts_ns: Presentation timestamp in nanoseconds
        keep_av_open: Keep AV container open in cache

    Returns:
        BGRA numpy array

    Raises:
        ImportError: If video dependencies are not installed
        ValueError: If loading fails
        FileNotFoundError: If local file doesn't exist
    """
    from . import cached_av

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

        # Convert nanoseconds to fraction
        pts_fraction = Fraction(pts_ns, NANOSECOND)

        # Open video and read frame
        container = cached_av.open(actual_path, "r", keep_av_open=keep_av_open)
        try:
            frame = _read_frame_at_pts(container, pts_fraction)
            rgb_array = frame.to_ndarray(format="rgb24")
            bgra_array: npt.NDArray[np.uint8] = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)  # type: ignore[assignment]
            return bgra_array
        finally:
            if not keep_av_open:
                container.close()
    except FileNotFoundError:
        raise
    except Exception as e:
        pts_seconds = pts_ns / NANOSECOND
        raise ValueError(f"Failed to load frame at {pts_seconds:.3f}s from {path_or_url}: {e}") from e


def _read_frame_at_pts(
    container: "av.container.InputContainer",
    pts: Fraction,
) -> "av.VideoFrame":
    """Read single frame at or after given timestamp."""
    if not container.streams.video:
        raise ValueError("No video streams found")

    stream = container.streams.video[0]

    # Seek to the timestamp
    container.seek(int(pts / stream.time_base), stream=stream)

    # Decode frames until we find the right one
    for frame in container.decode(stream):
        if frame.time >= float(pts):
            return frame

    raise ValueError(f"Frame not found at {float(pts):.2f}s")


# ============================================================================
# Data URI Handling
# ============================================================================


def _load_from_data_uri(data_uri: str) -> npt.NDArray[np.uint8]:
    """Load image from data URI and return as BGRA array."""
    from .data_uri import DataURI

    data_uri_obj = DataURI.from_uri(data_uri)
    # DataURI.to_rgb_array() returns RGB, convert to BGRA
    rgb_array = data_uri_obj.to_rgb_array()
    bgra_array: npt.NDArray[np.uint8] = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)  # type: ignore[assignment]
    return bgra_array
