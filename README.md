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

## Usage

```python
from mediaref import MediaRef, batch_decode

# Reference creation - supports multiple URI schemes
MediaRef(uri="image.png")                              # Local file
MediaRef(uri="https://example.com/image.jpg")          # Remote URL
MediaRef(uri="video.mp4", pts_ns=1_000_000_000)        # Video frame at 1.0s
MediaRef(uri="data:image/png;base64,...")              # Embedded data URI

# Loading
ref.to_rgb_array()                                     # Returns (H, W, 3) numpy array
ref.to_pil_image()                                     # Returns PIL.Image

# Batch decoding with automatic caching (requires [video] extra)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)                              # Default: PyAV decoder

# Use TorchCodec decoder for GPU acceleration (requires torchcodec>=0.4.0)
frames = batch_decode(refs, decoder="torchcodec")

# Use batch decoding strategy (PyAV only)
from mediaref.video_decoder import BatchDecodingStrategy
frames = batch_decode(refs, strategy=BatchDecodingStrategy.SEQUENTIAL)

# Embedding
data_uri = ref.embed_as_data_uri(format="png")         # Encode to data URI
MediaRef(uri=data_uri)                                 # Create from data URI

# Path resolution for MCAP/rosbag datasets
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
ref.resolve_relative_path("/data/recording.mcap")      # Returns absolute path

# Serialization (Pydantic-based)
ref.model_dump()                                       # {'uri': '...', 'pts_ns': ...}
ref.model_dump_json()                                  # '{"uri":"...","pts_ns":...}'
MediaRef.model_validate(data)                          # From dict
MediaRef.model_validate_json(json_str)                 # From JSON string
```

## API Reference

### MediaRef(uri: str, pts_ns: int | None = None)

**Properties:** `is_embedded`, `is_video`, `is_remote`, `is_local`, `is_relative_path`

**Methods:**
- `to_rgb_array(**kwargs) -> np.ndarray` - Load as RGB array (H, W, 3)
- `to_pil_image(**kwargs) -> PIL.Image` - Load as PIL Image
- `embed_as_data_uri(format="png", quality=None) -> str` - Encode to data URI
- `resolve_relative_path(base_path, allow_nonlocal=False) -> MediaRef` - Resolve relative paths
- `validate_uri() -> bool` - Check if URI exists (local files only)
- `model_dump() -> dict` - Serialize to dict
- `model_dump_json() -> str` - Serialize to JSON
- `model_validate(data) -> MediaRef` - Deserialize from dict
- `model_validate_json(json_str) -> MediaRef` - Deserialize from JSON

### Functions

- `batch_decode(refs, strategy=None, decoder="pyav", **kwargs) -> list[np.ndarray]` - Batch decode using optimized batch decoding API
  - `refs`: List of MediaRef objects to decode
  - `strategy`: Batch decoding strategy (PyAV only): `SEPARATE`, `SEQUENTIAL`, or `SEQUENTIAL_PER_KEYFRAME_BLOCK`
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

