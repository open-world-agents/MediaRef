"""Feature detection for optional dependencies.

This module detects available features once at import time and provides
a single source of truth for feature availability throughout the package.
"""

# Optional PyAV backend
try:
    import av  # noqa: F401

    HAS_PYAV = True
    PYAV_ERROR = None
except ImportError as e:
    HAS_PYAV = False
    PYAV_ERROR = str(e)


def require_pyav() -> None:
    """Require the optional PyAV backend."""
    if not HAS_PYAV:
        raise ImportError(
            "PyAV video frame extraction requires the 'pyav' extra. "
            "Install with: pip install mediaref[pyav]\n"
            f"Original error: {PYAV_ERROR}"
        )


__all__ = ["HAS_PYAV", "require_pyav"]
