"""Internal loading and encoding utilities."""

import base64
import gc
import os
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import requests

from ._features import require_video

if TYPE_CHECKING:
    import av

# Constants
REQUEST_TIMEOUT = 60  # HTTP request timeout in seconds
NANOSECOND = 1_000_000_000  # 1 second in nanoseconds

# Garbage collection for PyAV reference cycles
_CALLED_TIMES = 0
GC_COLLECTION_INTERVAL = 10


# ============================================================================
# Image Loading
# ============================================================================


def load_image_as_bgra(path_or_uri: str) -> np.ndarray:
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
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, "
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


def _pil_to_bgra_array(pil_image: PIL.Image.Image) -> np.ndarray:
    """Convert PIL image to BGRA numpy array."""
    # Ensure image is in RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array and then to BGRA
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)


# ============================================================================
# Video Loading
# ============================================================================


def load_video_frame_as_bgra(
    path_or_url: str,
    pts_ns: int,
    *,
    keep_av_open: bool = False,
) -> np.ndarray:
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
        # Validate local file exists
        if not path_or_url.startswith(("http://", "https://")):
            if not Path(path_or_url).exists():
                raise FileNotFoundError(f"Video file not found: {path_or_url}")

        # Convert nanoseconds to fraction
        pts_fraction = Fraction(pts_ns, NANOSECOND)

        # Open video and read frame
        container = cached_av.open(path_or_url, "r", keep_av_open=keep_av_open)
        try:
            frame = _read_frame_at_pts(container, pts_fraction)
            rgb_array = frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)
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


def _load_from_data_uri(data_uri: str) -> np.ndarray:
    """Load image from data URI."""
    parsed = urlparse(data_uri)
    if parsed.scheme != "data":
        raise ValueError(f"Invalid data URI scheme: {parsed.scheme}")

    try:
        # Extract base64 data from data URI
        data_part = parsed.path.split(",", 1)[1]
        return _decode_from_base64(data_part)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid data URI format: {e}") from e


def _decode_from_base64(data: str) -> np.ndarray:
    """Decode base64 string to BGRA numpy array."""
    try:
        image_bytes = base64.b64decode(data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_array is None:
            raise ValueError("Failed to decode base64 image data")

        return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}") from e


# ============================================================================
# Encoding
# ============================================================================


def encode_array_to_base64(
    array: np.ndarray,
    format: Literal["png", "jpeg", "bmp"],
    quality: Optional[int] = None,
) -> str:
    """Encode BGRA numpy array to base64 string.

    Args:
        array: BGRA numpy array
        format: Output format ('png', 'jpeg', or 'bmp')
        quality: JPEG quality (1-100), ignored for PNG and BMP

    Returns:
        Base64 encoded string
    """
    # Convert BGRA to BGR for cv2 encoding
    bgr_array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

    # Encode based on format
    if format == "png":
        success, encoded = cv2.imencode(".png", bgr_array)
    elif format == "jpeg":
        if quality is None:
            quality = 85
        if not (1 <= quality <= 100):
            raise ValueError("JPEG quality must be between 1 and 100")
        success, encoded = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif format == "bmp":
        success, encoded = cv2.imencode(".bmp", bgr_array)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise ValueError(f"Failed to encode image as {format}")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")
