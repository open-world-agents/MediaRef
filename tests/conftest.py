"""Shared test fixtures for MediaRef test suite."""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

# ============================================================================
# Image Fixtures
# ============================================================================


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a sample image file (48x64 BGR).

    Returns:
        Path to the created PNG image file.
    """
    image_path = tmp_path / "test_image.png"
    test_image = np.zeros((48, 64, 3), dtype=np.uint8)
    test_image[:, :, 0] = 255  # Blue channel (BGR format)
    cv2.imwrite(str(image_path), test_image)
    return image_path


@pytest.fixture
def sample_image_files(tmp_path: Path) -> list[Path]:
    """Create multiple sample image files with different colors.

    Returns:
        List of paths to created image files.
    """
    images = []
    for i in range(3):
        image_path = tmp_path / f"test_image_{i}.png"
        test_image = np.full((48, 64, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        images.append(image_path)
    return images


@pytest.fixture
def sample_bgra_array() -> np.ndarray:
    """Create a sample BGRA numpy array with gradient pattern.

    Returns:
        BGRA numpy array (48, 64, 4).
    """
    height, width = 48, 64
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Create gradient pattern for easy identification
    for y in range(height):
        for x in range(width):
            frame[y, x] = [x * 4, y * 5, (x + y) * 2, 255]  # BGRA

    return frame


@pytest.fixture
def sample_rgb_array() -> np.ndarray:
    """Create a sample RGB numpy array.

    Returns:
        RGB numpy array (48, 64, 3).
    """
    height, width = 48, 64
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Create simple color blocks
    frame[:24, :32] = [255, 0, 0]  # Red
    frame[:24, 32:] = [0, 255, 0]  # Green
    frame[24:, :32] = [0, 0, 255]  # Blue
    frame[24:, 32:] = [255, 255, 0]  # Yellow

    return frame


# ============================================================================
# Video Fixtures
# ============================================================================


@pytest.fixture
def sample_video_file(tmp_path: Path) -> tuple[Path, list[float]]:
    """Create a sample video file with known frames at specific timestamps.

    Returns:
        Tuple of (video_path, list of timestamps in nanoseconds).
    """
    try:
        import av
    except ImportError:
        pytest.skip("Video dependencies not installed (av)")

    from fractions import Fraction

    video_path = tmp_path / "test_video.mp4"
    # 5 frames at 10fps = 0.1 second intervals
    # pts values: 0, 1, 2, 3, 4 (frame numbers)
    # timestamps in nanoseconds
    timestamps_ns = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000]

    # Create video with av
    container = av.open(str(video_path), "w")
    stream = container.add_stream("h264", rate=10)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"

    # Write frames with distinct colors
    for i in range(5):
        frame = av.VideoFrame(64, 48, "rgb24")
        # Create distinct color for each frame
        arr = np.full((48, 64, 3), i * 50, dtype=np.uint8)
        frame.planes[0].update(arr)
        frame.pts = i  # Frame number
        frame.time_base = Fraction(1, 10)  # 10 fps
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return video_path, timestamps_ns


@pytest.fixture
def sample_video_file_large(tmp_path: Path) -> tuple[Path, list[int]]:
    """Create a larger video file for performance testing.

    Returns:
        Tuple of (video_path, list of timestamps in nanoseconds).
    """
    try:
        import av
    except ImportError:
        pytest.skip("Video dependencies not installed (av)")

    from fractions import Fraction

    video_path = tmp_path / "test_video_large.mp4"
    # 30 frames at 30fps = 1 second of video
    # timestamps in nanoseconds
    timestamps_ns = [int(i * 1_000_000_000 / 30) for i in range(30)]

    container = av.open(str(video_path), "w")
    stream = container.add_stream("h264", rate=30)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = "yuv420p"

    for i in range(30):
        frame = av.VideoFrame(640, 480, "rgb24")
        # Create frame with varying intensity
        arr = np.full((480, 640, 3), (i * 8) % 256, dtype=np.uint8)
        frame.planes[0].update(arr)
        frame.pts = i
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return video_path, timestamps_ns


# ============================================================================
# URI Fixtures
# ============================================================================


@pytest.fixture
def sample_data_uri(sample_bgra_array: np.ndarray) -> str:
    """Create a valid data URI from BGRA array.

    Returns:
        Data URI string (PNG format).
    """
    # Convert BGRA to BGR for encoding
    bgr_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2BGR)
    success, encoded = cv2.imencode(".png", bgr_array)
    if not success:
        raise ValueError("Failed to encode image")

    base64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


@pytest.fixture
def sample_data_uri_jpeg(sample_bgra_array: np.ndarray) -> str:
    """Create a valid JPEG data URI from BGRA array.

    Returns:
        Data URI string (JPEG format).
    """
    bgr_array = cv2.cvtColor(sample_bgra_array, cv2.COLOR_BGRA2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image")

    base64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


@pytest.fixture
def remote_test_image_url() -> str:
    """Return a reliable remote test image URL.

    Returns:
        URL to a test image (httpbingo.org).
    """
    return "https://httpbingo.org/image/png"


@pytest.fixture
def remote_test_video_url() -> str:
    """Return a reliable remote test video URL.

    Returns:
        URL to a test video (Hugging Face dataset).
    """
    return "https://huggingface.co/datasets/open-world-agents/example_dataset/resolve/main/example.mkv"


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def temp_directory(tmp_path: Path) -> Path:
    """Create a temporary directory structure for path testing.

    Returns:
        Path to temporary directory.
    """
    # Create nested directory structure
    (tmp_path / "data" / "recordings").mkdir(parents=True)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "videos").mkdir(parents=True)

    # Create some dummy files
    (tmp_path / "data" / "recording.mcap").touch()
    (tmp_path / "data" / "images" / "frame.png").touch()
    (tmp_path / "data" / "videos" / "clip.mp4").touch()

    return tmp_path


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "network: tests requiring network access")
    config.addinivalue_line("markers", "video: tests requiring video dependencies")
    config.addinivalue_line("markers", "slow: slow tests (batch processing, large files)")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "performance: performance benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers and available dependencies."""
    # Check if video dependencies are available
    try:
        import av  # noqa: F401

        video_available = True
    except ImportError:
        video_available = False

    skip_video = pytest.mark.skip(reason="Video dependencies not installed (av)")

    for item in items:
        # Skip video tests if video dependencies not available
        if "video" in item.keywords and not video_available:
            item.add_marker(skip_video)
