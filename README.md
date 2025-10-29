# MediaRef

[![CI](https://img.shields.io/github/actions/workflow/status/open-world-agents/MediaRef/ci.yml?branch=main&logo=github&label=CI)](https://github.com/open-world-agents/MediaRef/actions?query=event%3Apush+branch%3Amain+workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/mediaref.svg)](https://pypi.python.org/pypi/mediaref)
[![versions](https://img.shields.io/pypi/pyversions/mediaref.svg)](https://github.com/open-world-agents/MediaRef)
[![license](https://img.shields.io/github/license/open-world-agents/MediaRef.svg)](https://github.com/open-world-agents/MediaRef/blob/main/LICENSE)

<!-- [![downloads](https://static.pepy.tech/badge/mediaref/month)](https://pepy.tech/project/mediaref) -->

Pydantic-based media reference for images and video frames. Supports file paths, URLs, data URIs, and video timestamps. Designed for dataset metadata and lazy loading.

## Installation

```bash
# Core package with image loading support
pip install mediaref

# With video support (adds PyAV for video frame extraction)
pip install mediaref[video]
```

## Quick Start

### Basic Usage

```python
from mediaref import MediaRef, DataURI, batch_decode
import numpy as np

# 1. Create references (lightweight, no loading yet)
ref = MediaRef(uri="image.png")                        # Local file
ref = MediaRef(uri="https://example.com/image.jpg")    # Remote URL
ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)  # Video frame at 1.0s

# 2. Load media
rgb = ref.to_rgb_array()                               # Returns (H, W, 3) numpy array
pil = ref.to_pil_image()                               # Returns PIL.Image

# 3. Embed as data URI
data_uri = DataURI.from_image(rgb, format="png")
ref = MediaRef(uri=data_uri)                           # Self-contained reference

# 4. Batch decode video frames (opens video once, reuses handle)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)                            # Much faster than loading individually
```

### DataURI - Embed Media as Base64

DataURI allows you to encode images as self-contained data URIs (base64-encoded).

```python
from mediaref import DataURI
import numpy as np

# Create from numpy array
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
data_uri = DataURI.from_image(rgb, format="png")      # Supports: png, jpeg, bmp

# Create from file
data_uri = DataURI.from_file("image.png")

# Create from PIL Image
from PIL import Image
pil_img = Image.open("image.png")
data_uri = DataURI.from_image(pil_img, format="jpeg", quality=90)

# Use with MediaRef
ref = MediaRef(uri=data_uri)                           # Accepts DataURI object
ref = MediaRef(uri=str(data_uri))                      # Or string

# Convert back to image
rgb = data_uri.to_rgb_array()                          # (H, W, 3) numpy array
pil = data_uri.to_pil_image()                          # PIL Image

# Properties
print(data_uri.mimetype)                               # "image/png"
print(len(data_uri))                                   # URI length in bytes
print(data_uri.is_image)                               # True for image/* types
```

### Batch Decoding - Optimized Video Frame Loading

When loading multiple frames from the same video, `batch_decode()` opens the video file once and reuses the handle, avoiding repeated file I/O overhead.

**Decoding Strategies (PyAV only):**

MediaRef provides three strategies optimized for different access patterns:

- **SEQUENTIAL_PER_KEYFRAME_BLOCK** (default)
  - Decodes frames in batches, restarting at each keyframe interval
  - **Best for:** Mixed queries (both sparse and dense)
  - **Performance:** Balanced approach, ~X times faster than individual loading (TODO: benchmark)

- **SEQUENTIAL**
  - Decodes all frames in one sequential pass from first to last requested frame
  - **Best for:** Dense queries where frames are close together (e.g., every 1 second for 10 seconds)
  - **Performance:** Fastest for dense queries, ~X times faster (TODO: benchmark)
  - **Trade-off:** May decode unnecessary frames between sparse timestamps

- **SEPARATE**
  - Seeks and decodes each frame independently
  - **Best for:** Very sparse queries (e.g., frames at 0s, 100s, 500s, 1000s)
  - **Performance:** ~X times faster than individual loading (TODO: benchmark)
  - **Trade-off:** More seeking overhead than sequential strategies

```python
from mediaref import MediaRef, batch_decode
from mediaref.video_decoder import BatchDecodingStrategy

# Default: SEQUENTIAL_PER_KEYFRAME_BLOCK (works well for most cases)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)                            # Frames at 0s, 1s, 2s, ..., 9s

# Dense queries: Use SEQUENTIAL for maximum speed
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(100)]
frames = batch_decode(refs, strategy=BatchDecodingStrategy.SEQUENTIAL)

# Sparse queries: Use SEPARATE to avoid decoding unnecessary frames
timestamps = [0, 100_000_000_000, 500_000_000_000, 1000_000_000_000]  # 0s, 100s, 500s, 1000s
refs = [MediaRef(uri="video.mp4", pts_ns=t) for t in timestamps]
frames = batch_decode(refs, strategy=BatchDecodingStrategy.SEPARATE)

# Use TorchCodec for GPU acceleration (strategy parameter ignored)
frames = batch_decode(refs, decoder="torchcodec")
```

### Path Resolution & Serialization

```python
# Resolve relative paths (useful for MCAP/rosbag datasets)
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
resolved = ref.resolve_relative_path("/data/recording.mcap")

# Serialization (Pydantic-based)
data = ref.model_dump()                                # {'uri': '...', 'pts_ns': ...}
json_str = ref.model_dump_json()                       # JSON string
ref = MediaRef.model_validate(data)                    # From dict
ref = MediaRef.model_validate_json(json_str)           # From JSON
```

## API Reference

### MediaRef(uri: str | DataURI, pts_ns: int | None = None)

**Properties:** `is_embedded`, `is_video`, `is_remote`, `is_local`, `is_relative_path`

**Methods:**
- `to_rgb_array(**kwargs) -> np.ndarray` - Load as RGB array (H, W, 3)
- `to_pil_image(**kwargs) -> PIL.Image` - Load as PIL Image
- `resolve_relative_path(base_path, allow_nonlocal=False) -> MediaRef` - Resolve relative paths
- `validate_uri() -> bool` - Check if URI exists (local files only)
- `model_dump() -> dict` - Serialize to dict
- `model_dump_json() -> str` - Serialize to JSON
- `model_validate(data) -> MediaRef` - Deserialize from dict
- `model_validate_json(json_str) -> MediaRef` - Deserialize from JSON

### DataURI

**Fields:**
- `mimetype: str` - MIME type (e.g., "image/png")
- `is_base64: bool` - Whether data is base64 encoded
- `data: bytes` - Data payload (base64 string as bytes if is_base64=True)

**Properties:**
- `uri: str` - Full data URI string
- `decoded_data: bytes` - Decoded data payload (handles base64 decoding)
- `is_image: bool` - True if MIME type is image/*

**Class Methods:**
- `from_uri(uri: str) -> DataURI` - Parse data URI string
- `from_image(image: np.ndarray | PIL.Image, format="png", quality=None) -> DataURI` - Create from image
- `from_file(path: str | Path, format=None) -> DataURI` - Create from file

**Methods:**
- `to_pil_image() -> PIL.Image` - Convert to PIL Image
- `to_rgb_array() -> np.ndarray` - Convert to RGB array (H, W, 3)
- `__str__() -> str` - Return data URI string
- `__len__() -> int` - Return URI length in bytes

### Functions

- `batch_decode(refs, strategy=None, decoder="pyav", **kwargs) -> list[np.ndarray]` - Batch decode using optimized batch decoding API
  - `refs`: List of MediaRef objects to decode
  - `strategy`: Batch decoding strategy (PyAV only): `SEPARATE`, `SEQUENTIAL`, or `SEQUENTIAL_PER_KEYFRAME_BLOCK` (default)
  - `decoder`: Decoder backend (`"pyav"` or `"torchcodec"`)
- `cleanup_cache()` - Clear video container cache (PyAV only)

### Video Decoders (requires `[video]` extra)

- `PyAVVideoDecoder(source)` - PyAV-based decoder with batch decoding strategies
  - Supports batch decoding strategies: `SEPARATE`, `SEQUENTIAL`, `SEQUENTIAL_PER_KEYFRAME_BLOCK`
  - CPU-based decoding using FFmpeg
  - Automatic container caching with reference counting
- `TorchCodecVideoDecoder(source)` - TorchCodec-based decoder for GPU acceleration
  - Requires `torchcodec>=0.4.0` (install separately)
  - GPU-accelerated decoding with CUDA support
  - Does not support batch decoding strategies (parameter ignored)

**Decoder Comparison:**

| Feature | PyAVVideoDecoder | TorchCodecVideoDecoder |
|---------|------------------|------------------------|
| Batch decoding strategies | ✅ Full support | ❌ Not supported (ignored) |
| GPU acceleration | ❌ CPU only | ✅ CUDA support |
| Backend | PyAV (FFmpeg) | TorchCodec (FFmpeg) |
| Installation | `pip install mediaref[video]` | `pip install torchcodec>=0.4.0` |

**When to use:**
- Use `PyAVVideoDecoder` (default) for fine-grained control over batch decoding strategies
- Use `TorchCodecVideoDecoder` for GPU-accelerated decoding when processing large batches

## Design Notes

- **Video container caching**: Uses reference counting with LRU eviction (default: 10 containers)
- **MCAP file path resolution**: Detects `.mcap` suffix and uses parent directory as base path
- **Garbage collection**: Triggered every 10 PyAV operations to handle FFmpeg reference cycles
- **Cache size**: Configurable via `AV_CACHE_SIZE` environment variable
- **Lazy loading**: Video dependencies only imported when needed (not at module import time)

## Acknowledgments

The video decoder interface design references [TorchCodec](https://github.com/pytorch/torchcodec)'s API design.

## Dependencies

**Core dependencies** (automatically installed):
- `pydantic>=2.0` - Data validation and serialization (requires Pydantic v2 API)
- `numpy` - Array operations
- `opencv-python` - Image loading and color conversion
- `pillow>=9.4.0` - Image loading from various sources
- `requests>=2.32.2` - HTTP/HTTPS URL loading
- `loguru` - Logging (disabled by default for library code)

**Optional dependencies**:
- `[video]` extra: `av>=15.0` (PyAV for video frame extraction)
- TorchCodec: `torchcodec>=0.4.0` (install separately for GPU-accelerated decoding)

