"""Core MediaRef class."""

import warnings
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from pydantic import BaseModel, Field


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
        >>> # Embedded data URI
        >>> data_uri = ref.embed_as_data_uri(format="png")
        >>> embedded_ref = MediaRef(uri=data_uri)
    """

    uri: str = Field(
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
    def is_local(self) -> bool:
        """True if this references a local file path (not embedded or remote)."""
        return not self.is_embedded and not self.is_remote

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
        allow_nonlocal: bool = False,
    ) -> "MediaRef":
        """Resolve relative path against a base path.

        Uses platform-specific path semantics (behavior differs on Windows vs POSIX).

        Args:
            base_path: Base path (typically MCAP file path) to resolve against
            allow_nonlocal: Allow non-local paths (embedded, remote) to be resolved

        Returns:
            New MediaRef with resolved absolute path

        Examples:
            >>> ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
            >>> ref_resolved = ref.resolve_relative_path("/data/recording.mcap")
            >>> # ref_resolved.uri == "/data/relative/video.mkv"
        """
        if not self.is_local:
            if allow_nonlocal:
                return self
            warnings.warn(f"Cannot resolve non-local path: {self.uri}")
            return self  # Nothing to resolve for non-local paths

        if not self.is_relative_path:
            return self  # Already absolute or not a local path

        base_path_obj = Path(base_path)
        # If base path is an MCAP file, use its parent directory
        if base_path_obj.suffix == ".mcap":
            base_path_obj = base_path_obj.parent

        resolved_path = (base_path_obj / self.uri).as_posix()
        return MediaRef(uri=resolved_path, pts_ns=self.pts_ns)

    # ========== Loading Methods ==========

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

    def embed_as_data_uri(
        self,
        format: Literal["png", "jpeg", "bmp"] = "png",
        quality: Optional[int] = None,
    ) -> str:
        """Load media and encode as data URI.

        Args:
            format: Image format ('png', 'jpeg', 'bmp')
            quality: JPEG quality (1-100), ignored for PNG/BMP

        Returns:
            Data URI string

        Raises:
            ImportError: If video dependencies are not installed (for video frames)

        Examples:
            >>> ref = MediaRef(uri="image.png")
            >>> data_uri = ref.embed_as_data_uri(format="png")
            >>> embedded_ref = MediaRef(uri=data_uri)
            >>> embedded_ref.is_embedded
            True
        """
        from ._internal import encode_array_to_base64

        bgra = self._load_as_bgra()
        base64_data = encode_array_to_base64(bgra, format, quality)
        return f"data:image/{format};base64,{base64_data}"

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
