"""Core MediaRef class."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, BeforeValidator, Field

if TYPE_CHECKING:
    from .data_uri import DataURI


def _convert_datauri_to_str(v: Union[str, "DataURI"]) -> str:
    """Convert DataURI object to string if provided."""
    from .data_uri import DataURI  # noqa: E402

    if isinstance(v, DataURI):
        return v.uri
    return v


class MediaRef(BaseModel):
    """Media reference for images and video frames.

    Supports multiple URI schemes:
    - File paths: "/absolute/path" or "relative/path"
    - File URIs: "file:///path/to/media"
    - HTTP/HTTPS URLs: "https://example.com/image.jpg"
    - Data URIs: "data:image/png;base64,..."
    - Video frames: Any of the above with pts_ns set

    Examples:
        >>> # Image reference
        >>> ref = MediaRef(uri="image.png")
        >>> rgb = ref.to_ndarray(format="rgb")  # Default RGB format
        >>>
        >>> # Video frame reference
        >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        >>> frame = ref.to_ndarray()
        >>>
        >>> # Remote URL
        >>> ref = MediaRef(uri="https://example.com/image.jpg")
        >>> pil_img = ref.to_pil_image()
        >>>
        >>> # Embedded data URI (from file or array)
        >>> from mediaref import DataURI
        >>> data_uri = DataURI.from_file("image.png")  # or DataURI.from_image(array)
        >>> ref = MediaRef(uri=data_uri)  # Can pass DataURI directly
    """

    uri: Annotated[str, BeforeValidator(_convert_datauri_to_str)] = Field(
        ...,
        description="URI (data:image/png;base64,... | file:///path | http[s]://...) or posix file path (/absolute/path | relative/path)",
    )
    pts_ns: Optional[int] = Field(
        default=None,
        description="Video frame timestamp in nanoseconds",
    )

    def __init__(self, *, uri: Union[str, "DataURI"], pts_ns: Optional[int] = None, **kwargs) -> None:
        """Initialize MediaRef with URI and optional timestamp.

        Args:
            uri: URI string or DataURI object
            pts_ns: Optional video frame timestamp in nanoseconds
            **kwargs: Additional fields (for internal use)
        """
        super().__init__(uri=uri, pts_ns=pts_ns, **kwargs)

    # ========== Properties ==========

    @property
    def is_embedded(self) -> bool:
        """True if this is embedded data (data URI)."""
        return self.uri.startswith("data:")

    @property
    def is_video(self) -> bool:
        """True if this references video media."""
        return self.pts_ns is not None

    @property
    def is_remote(self) -> bool:
        """True if this references a remote URL (http/https)."""
        return self.uri.startswith(("http://", "https://"))

    @property
    def is_relative_path(self) -> bool:
        """True if this is a relative path (not absolute, not URI).

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).
        """
        if self.is_embedded or self.is_remote or self.uri.startswith("file://"):
            return False
        return not Path(self.uri).is_absolute()

    # ========== Path Utilities ==========

    def validate_uri(self) -> bool:
        """Validate that the URI exists (local files only).

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).

        Returns:
            True if URI is valid/accessible

        Raises:
            NotImplementedError: For remote URI validation
        """
        if self.is_remote:
            raise NotImplementedError("Remote URI validation not implemented")
        if self.is_embedded:
            return True  # Embedded data is always "valid"
        return Path(self.uri).exists()

    def resolve_relative_path(
        self,
        base_path: str,
        on_unresolvable: Literal["error", "warn", "ignore"] = "warn",
    ) -> "MediaRef":
        """Resolve relative path against a base path.

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).

        Args:
            base_path: Base path to resolve against
            on_unresolvable: How to handle unresolvable URIs (embedded/remote):
                - "error": Raise ValueError
                - "warn": Issue warning and return unchanged (default)
                - "ignore": Silently return unchanged

        Returns:
            New MediaRef with resolved absolute path

        Raises:
            ValueError: If URI is unresolvable and on_unresolvable="error"

        Examples:
            >>> ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
            >>> ref_resolved = ref.resolve_relative_path("/data/recordings")
            >>> # ref_resolved.uri == "/data/recordings/relative/video.mkv"
            >>>
            >>> # Handle unresolvable URIs
            >>> remote = MediaRef(uri="https://example.com/image.jpg")
            >>> remote.resolve_relative_path("/data/recordings", on_unresolvable="ignore")
        """
        if self.is_embedded or self.is_remote:
            if on_unresolvable == "error":
                raise ValueError(f"Cannot resolve unresolvable URI (embedded or remote): {self.uri}")
            elif on_unresolvable == "warn":
                warnings.warn(f"Cannot resolve unresolvable URI (embedded or remote): {self.uri}")
            return self  # Nothing to resolve for embedded/remote URIs

        if not self.is_relative_path:
            return self  # Already absolute or not a local path

        base_path_obj = Path(base_path)
        resolved_path = (base_path_obj / self.uri).as_posix()
        return MediaRef(uri=resolved_path, pts_ns=self.pts_ns)

    # ========== Loading Methods ==========

    def to_ndarray(self, format: Literal["rgb", "bgr", "rgba", "bgra", "gray"] = "rgb") -> npt.NDArray[np.uint8]:
        """Load and return media as numpy ndarray in specified format.

        Args:
            format: Output format (default: "rgb")
                - "rgb": RGB color (H, W, 3)
                - "bgr": BGR color (H, W, 3)
                - "rgba": RGB with alpha (H, W, 4)
                - "bgra": BGR with alpha (H, W, 4)
                - "gray": Grayscale (H, W)
        Returns:
            Numpy ndarray in requested format

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
            ValueError: If format is invalid

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> rgb = ref.to_ndarray(format="rgb")  # Default RGB format
            >>>
            >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
            >>> frame = ref.to_ndarray()  # Requires: pip install mediaref[video]
        """
        rgba = self._load_as_rgba()

        CONVERSION_MAP = {
            "rgb": cv2.COLOR_RGBA2RGB,
            "bgr": cv2.COLOR_RGBA2BGR,
            "bgra": cv2.COLOR_RGBA2BGRA,
            "gray": cv2.COLOR_RGBA2GRAY,
        }
        if format == "rgba":
            return rgba
        if format in CONVERSION_MAP:
            return cv2.cvtColor(rgba, CONVERSION_MAP[format])  # type: ignore[return-value]

        raise ValueError(f"Unsupported format: {format}. Must be one of: rgb, bgr, rgba, bgra, gray")

    def to_pil_image(self, format: Literal["rgb", "rgba", "gray"] = "rgb") -> PIL.Image.Image:
        """Load and return media as PIL Image.

        Args:
            format: Output format (default: "rgb")
                - "rgb": RGB color
                - "rgba": RGB with alpha
                - "gray": Grayscale

        Returns:
            PIL Image object

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
            ValueError: If format is invalid

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> img = ref.to_pil_image()
        """
        if format in ("bgr", "bgra"):
            raise ValueError(f"Format '{format}' is not compatible with to_pil_image. Use 'rgb', 'rgba', or 'gray'.")

        return PIL.Image.fromarray(self.to_ndarray(format=format))

    # ========== Internal ==========

    def _load_as_rgba(self) -> npt.NDArray[np.uint8]:
        """Internal: Load media as RGBA array.

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
        """
        from ._internal import load_image_as_rgba, load_video_frame_as_rgba

        if self.is_video:
            assert self.pts_ns is not None  # Type guard: is_video ensures pts_ns is not None
            return load_video_frame_as_rgba(self.uri, self.pts_ns)
        else:
            return load_image_as_rgba(self.uri)
