"""DataURI class for handling data URI encoding and decoding."""

import base64
import io
import mimetypes
from pathlib import Path
from typing import IO, Literal, Optional, Union
from urllib.parse import quote, unquote, urlparse

import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, Field


class DataURI(BaseModel):
    """Data URI handler for encoding and decoding media.

    Supports RFC 2397 data URI scheme for embedding media data directly in URIs.

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
    data: bytes = Field(description="Decoded data payload")

    # ========== Properties ==========

    @property
    def uri(self) -> str:
        """Construct and return the full data URI string.

        Returns:
            Data URI string in format: data:[mimetype];base64,[data]
        """
        if self.is_base64:
            encoded_data = base64.b64encode(self.data).decode("utf-8")
            return f"data:{self.mimetype};base64,{encoded_data}"
        else:
            # URL-encode the data for non-base64 URIs
            encoded_data = quote(self.data.decode("utf-8"))
            return f"data:{self.mimetype},{encoded_data}"

    @property
    def size(self) -> int:
        """Size of data in bytes.

        Returns:
            Size of the decoded data payload in bytes
        """
        return len(self.data)

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

            # Decode data
            if is_base64:
                data_bytes = base64.b64decode(data_part)
            else:
                data_bytes = unquote(data_part).encode("utf-8")

            return cls(mimetype=mimetype, is_base64=is_base64, data=data_bytes)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid data URI format: {e}") from e

    @classmethod
    def from_stream(cls, stream: IO[bytes], mimetype: str) -> "DataURI":
        """Create from byte stream with explicit MIME type.

        Args:
            stream: Byte stream to read from
            mimetype: MIME type (e.g., "image/png")

        Returns:
            DataURI instance
        """
        data = stream.read()
        return cls(mimetype=mimetype, is_base64=True, data=data)

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
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            # Assume RGB format
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB array with shape (H, W, 3), got {image.shape}")
            pil_image = PIL.Image.fromarray(image, mode="RGB")
        else:
            pil_image = image

        # Encode to bytes
        buffer = io.BytesIO()
        if format == "png":
            pil_image.save(buffer, format="PNG")
        elif format == "jpeg":
            if quality is None:
                quality = 85
            if not (1 <= quality <= 100):
                raise ValueError("JPEG quality must be between 1 and 100")
            pil_image.save(buffer, format="JPEG", quality=quality)
        elif format == "bmp":
            pil_image.save(buffer, format="BMP")
        else:
            raise ValueError(f"Unsupported format: {format}")

        data = buffer.getvalue()
        mimetype = f"image/{format}"

        return cls(mimetype=mimetype, is_base64=True, data=data)

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
            data = f.read()

        return cls(mimetype=mimetype, is_base64=True, data=data)

    # ========== Conversion Methods ==========

    def to_pil_image(self) -> PIL.Image.Image:
        """Convert to PIL Image.

        Returns:
            PIL Image object

        Raises:
            ValueError: If data is not a valid image
        """
        if not self.is_image:
            raise ValueError(f"Cannot convert non-image MIME type '{self.mimetype}' to PIL Image")

        try:
            buffer = io.BytesIO(self.data)
            image = PIL.Image.open(buffer)
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}") from e

    def to_rgb_array(self) -> npt.NDArray[np.uint8]:
        """Convert to RGB numpy array (H, W, 3).

        Returns:
            RGB numpy array with shape (H, W, 3) and dtype uint8

        Raises:
            ValueError: If data is not a valid image
        """
        pil_image = self.to_pil_image()
        return np.array(pil_image, dtype=np.uint8)

    # ========== String Representation ==========

    def __str__(self) -> str:
        """Return the full data URI string (same as .uri property).

        Returns:
            Data URI string
        """
        return self.uri
