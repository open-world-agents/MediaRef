"""Core MediaRef class."""

import warnings
from pathlib import Path
from typing import Annotated, Literal, Optional

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, BeforeValidator, Field


def _convert_datauri_to_str(v) -> str:
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
        >>> rgb = ref.to_rgb_array()
        >>>
        >>> # Video frame reference
        >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        >>> frame = ref.to_rgb_array()
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
            base_path: Base path (typically MCAP file path) to resolve against
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
            >>> ref_resolved = ref.resolve_relative_path("/data/recording.mcap")
            >>> # ref_resolved.uri == "/data/relative/video.mkv"
            >>>
            >>> # Handle unresolvable URIs
            >>> remote = MediaRef(uri="https://example.com/image.jpg")
            >>> remote.resolve_relative_path("/data/base.mcap", on_unresolvable="ignore")
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
        # If base path is an MCAP file, use its parent directory. TODO: remove this
        if base_path_obj.suffix == ".mcap":
            base_path_obj = base_path_obj.parent

        resolved_path = (base_path_obj / self.uri).as_posix()
        return MediaRef(uri=resolved_path, pts_ns=self.pts_ns)

    # ========== Loading Methods ==========
    # TODO: non-rgb support
    def to_rgb_array(self, **kwargs) -> npt.NDArray[np.uint8]:
        """Load and return media as RGB numpy array.

        Args:
            **kwargs: Additional options (e.g., keep_av_open for videos)

        Returns:
            RGB numpy array (H, W, 3)

        Raises:
            ImportError: If video dependencies are not installed (for video frames)

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> rgb = ref.to_rgb_array()
            >>>
            >>> ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
            >>> frame = ref.to_rgb_array()  # Requires: pip install mediaref[video]
        """
        bgra = self._load_as_bgra(**kwargs)
        rgb_array: npt.NDArray[np.uint8] = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)  # type: ignore[assignment]
        return rgb_array

    def to_pil_image(self, **kwargs) -> PIL.Image.Image:
        """Load and return media as PIL Image.

        Args:
            **kwargs: Additional options (e.g., keep_av_open for videos)

        Returns:
            PIL Image object

        Raises:
            ImportError: If video dependencies are not installed (for video frames)

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> img = ref.to_pil_image()
        """
        rgb_array = self.to_rgb_array(**kwargs)
        return PIL.Image.fromarray(rgb_array)

    # ========== Internal ==========

    def _load_as_bgra(self, **kwargs) -> npt.NDArray[np.uint8]:
        """Internal: Load media as BGRA array.

        Raises:
            ImportError: If video dependencies are not installed (for video frames)
        """
        from ._internal import load_image_as_bgra, load_video_frame_as_bgra

        if self.is_video:
            assert self.pts_ns is not None  # Type guard: is_video ensures pts_ns is not None
            return load_video_frame_as_bgra(self.uri, self.pts_ns, **kwargs)
        else:
            return load_image_as_bgra(self.uri)
