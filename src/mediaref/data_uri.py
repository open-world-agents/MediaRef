"""DataURI class for handling data URI encoding and decoding."""

import base64
import mimetypes
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import quote, urlparse

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, Field, model_validator

# ============================================================================
# Internal image encoding/decoding functions (cv2-based for performance)
# ============================================================================


def _encode_image_to_bytes(
    array: npt.NDArray[np.uint8],
    format: Literal["png", "jpeg", "bmp"],
    quality: Optional[int] = None,
) -> bytes:
    """Encode BGRA numpy array to image bytes using cv2.

    Args:
        array: BGRA numpy array
        format: Output format ('png', 'jpeg', or 'bmp')
        quality: JPEG quality (1-100), ignored for PNG and BMP

    Returns:
        Encoded image bytes
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

    return encoded.tobytes()


def _decode_image_to_bgra(image_bytes: bytes) -> npt.NDArray[np.uint8]:
    """Decode image bytes to BGRA numpy array using cv2.

    Args:
        image_bytes: Encoded image data

    Returns:
        BGRA numpy array
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_array is None:
            raise ValueError("Failed to decode image data")

        bgra_array: npt.NDArray[np.uint8] = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2BGRA)  # type: ignore[assignment]
        return bgra_array
    except Exception as e:
        raise ValueError(f"Failed to decode image data: {e}") from e


class DataURI(BaseModel):
    """Data URI handler for encoding and decoding media.

    Supports RFC 2397 data URI scheme for embedding media data directly in URIs.

    Encoding Requirements:
        - Binary data (images, etc.): Use base64 encoding (automatic in from_image/from_file)
        - Text data: Must be percent-encoded if it contains reserved characters, spaces,
          newlines, or other non-printing characters (RFC 3986)

    Examples:
        >>> # From numpy array
        >>> import numpy as np
        >>> array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> data_uri = DataURI.from_image(array, format="png")
        >>> print(data_uri.uri)
        >>>
        >>> # From file
        >>> data_uri = DataURI.from_file("image.png")
        >>> print(data_uri.mimetype)  # "image/png"
        >>>
        >>> # Parse existing data URI
        >>> uri_str = "data:image/png;base64,iVBORw0KG..."
        >>> data_uri = DataURI.from_uri(uri_str)
        >>> array = data_uri.to_rgb_array()
    """

    mimetype: str = Field(description="MIME type (e.g., 'image/png')")
    is_base64: bool = Field(default=True, description="Whether data is base64 encoded")
    data: bytes = Field(description="Data payload (base64 string as bytes if is_base64=True, raw bytes otherwise)")

    @model_validator(mode="after")
    def _validate_data_encoding(self) -> "DataURI":
        """Validate that non-base64 data is properly URL-encoded."""
        if self.is_base64:
            return self

        try:
            text_data = self.data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Non-base64 data must be valid UTF-8 text: {e}") from e

        # Check if data is already quoted by comparing with quoted version
        if text_data != quote(text_data, safe=""):
            raise ValueError(
                "Non-base64 data URI contains unquoted characters. "
                "Data should be URL-encoded before creating DataURI, or use base64 encoding."
            )
        return self

    # ========== Properties ==========

    @property
    def decoded_data(self) -> bytes:
        """Get the decoded data payload.

        If data is base64 encoded, this decodes it. Otherwise returns raw data.

        Returns:
            Decoded bytes
        """
        if self.is_base64:
            return base64.b64decode(self.data)
        else:
            return self.data

    @property
    def uri(self) -> str:
        """Construct and return the full data URI string.

        Returns:
            Data URI string in format: data:[mimetype];base64,[data]
        """
        data_str = self.data.decode("utf-8")
        return f"data:{self.mimetype};base64,{data_str}" if self.is_base64 else f"data:{self.mimetype},{data_str}"

    @property
    def is_image(self) -> bool:
        """True if MIME type is image/*.

        Returns:
            True if mimetype starts with 'image/'
        """
        return self.mimetype.startswith("image/")

    # ========== Class Methods for Construction ==========

    @classmethod
    def from_uri(cls, uri: str) -> "DataURI":
        """Create DataURI from a data URI string.

        Args:
            uri: Data URI string (e.g., "data:image/png;base64,...")

        Returns:
            DataURI instance

        Raises:
            ValueError: If URI is invalid or not a data URI
        """
        parsed = urlparse(uri)
        if parsed.scheme != "data":
            raise ValueError(f"Invalid data URI scheme: {parsed.scheme}")

        try:
            # Split on first comma to separate metadata from data
            metadata, data_part = parsed.path.split(",", 1)

            # Parse metadata
            parts = metadata.split(";")
            mimetype = parts[0] if parts[0] else "text/plain"
            is_base64 = "base64" in parts

            # Store data as-is, expecting input to be correct based on spec
            data_bytes = data_part.encode("utf-8")

            return cls(mimetype=mimetype, is_base64=is_base64, data=data_bytes)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid data URI format: {e}") from e

    @classmethod
    def from_image(
        cls,
        image: Union[npt.NDArray[np.uint8], PIL.Image.Image],
        format: Literal["png", "jpeg", "bmp"] = "png",
        quality: Optional[int] = None,
    ) -> "DataURI":
        """Create from numpy array or PIL Image.

        Args:
            image: Numpy array (H, W, 3) RGB or PIL Image
            format: Output format ('png', 'jpeg', 'bmp')
            quality: JPEG quality (1-100), ignored for PNG/BMP

        Returns:
            DataURI instance

        Raises:
            ValueError: If format is invalid or encoding fails
        """
        # Convert to RGB numpy array if PIL Image
        if isinstance(image, PIL.Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            rgb_array = np.array(image, dtype=np.uint8)
        else:
            # Assume RGB format
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB array with shape (H, W, 3), got {image.shape}")
            rgb_array = image

        # Convert RGB to BGRA for cv2 encoding
        bgra_array: npt.NDArray[np.uint8] = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGRA)  # type: ignore[assignment]

        # Encode to image bytes
        image_bytes = _encode_image_to_bytes(bgra_array, format=format, quality=quality)

        # Store as base64 encoded string (as bytes)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        mimetype = f"image/{format}"
        return cls(mimetype=mimetype, is_base64=True, data=base64_str.encode("utf-8"))

    @classmethod
    def from_file(cls, path: Union[str, Path], format: Optional[str] = None) -> "DataURI":
        """Create from file path (auto-detect format if not specified).

        Args:
            path: File path
            format: Optional format override (e.g., "png", "jpeg")

        Returns:
            DataURI instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format cannot be determined
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Determine MIME type
        if format:
            mimetype = f"image/{format}"
        else:
            # Auto-detect from file extension
            guessed_type = mimetypes.guess_type(str(path_obj))[0]
            if guessed_type:
                mimetype = guessed_type
            else:
                # Default to application/octet-stream if unknown
                mimetype = "application/octet-stream"

        # Read file data
        with open(path_obj, "rb") as f:
            raw_data = f.read()

        # Store as base64 encoded string (as bytes)
        base64_str = base64.b64encode(raw_data).decode("utf-8")
        return cls(mimetype=mimetype, is_base64=True, data=base64_str.encode("utf-8"))

    # ========== Conversion Methods ==========

    def to_pil_image(self) -> PIL.Image.Image:
        """Convert to PIL Image.

        Returns:
            PIL Image object

        Raises:
            ValueError: If data is not a valid image
        """
        # Convert to RGB array first, then to PIL
        rgb_array = self.to_rgb_array()
        return PIL.Image.fromarray(rgb_array, mode="RGB")

    def to_rgb_array(self) -> npt.NDArray[np.uint8]:
        """Convert to RGB numpy array (H, W, 3).

        Returns:
            RGB numpy array with shape (H, W, 3) and dtype uint8

        Raises:
            ValueError: If data is not a valid image
        """
        if not self.is_image:
            raise ValueError(f"Cannot convert non-image MIME type '{self.mimetype}' to PIL Image")

        try:
            # Get decoded data (handles base64 decoding if needed)
            image_bytes = self.decoded_data

            # Decode image bytes to BGRA using cv2
            bgra_array = _decode_image_to_bgra(image_bytes)

            # Convert BGRA to RGB
            rgb_array: npt.NDArray[np.uint8] = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGB)  # type: ignore[assignment]
            return rgb_array
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}") from e

    # ========== String Representation ==========

    def __str__(self) -> str:
        """Return the full data URI string (same as .uri property).

        Returns:
            Data URI string
        """
        return self.uri

    def __len__(self) -> int:
        """Return the size of the decoded data in bytes (same as .size property).

        Returns:
            Size of the decoded data in bytes
        """
        return len(self.uri)
