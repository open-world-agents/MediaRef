"""Feature detection for optional dependencies.

This module detects available features once at import time and provides
a single source of truth for feature availability throughout the package.
"""

# Video support (PyAV)
try:
    import av  # noqa: F401

    HAS_VIDEO = True
    VIDEO_ERROR = None
except ImportError as e:
    HAS_VIDEO = False
    VIDEO_ERROR = str(e)


def require_video() -> None:
    """Raise ImportError if video support is not available.

    Raises:
        ImportError: If PyAV is not installed
    """
    if not HAS_VIDEO:
        raise ImportError(
            "Video frame extraction requires the 'video' extra. "
            "Install with: pip install mediaref[video]\n"
            f"Original error: {VIDEO_ERROR}"
        )


__all__ = ["HAS_VIDEO", "require_video"]

