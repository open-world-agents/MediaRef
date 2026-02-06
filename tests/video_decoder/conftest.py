"""Video decoder test fixtures.

This module contains fixtures specific to video decoder testing.
Common fixtures (sample_video_file, etc.) are inherited from the parent conftest.py.
"""

from pathlib import Path

import pytest


@pytest.fixture
def example_video_path() -> Path:
    """Return path to the example_video.mkv test asset.

    This video has sparse keyframes (only 2 keyframes at 0.010s and 3.690s),
    making it useful for testing seek fallback behavior.

    Returns:
        Path to example_video.mkv, or pytest.skip if not found.
    """
    video_path = Path(__file__).parent.parent / "assets" / "example_video.mkv"
    if not video_path.exists():
        pytest.skip("example_video.mkv not found")
    return video_path


@pytest.fixture
def example_mkv_path() -> Path:
    """Return path to the example.mkv test asset.

    Returns:
        Path to example.mkv, or pytest.skip if not found.
    """
    video_path = Path(__file__).parent.parent / "assets" / "example.mkv"
    if not video_path.exists():
        pytest.skip("example.mkv not found")
    return video_path

