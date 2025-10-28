# MediaRef

Pydantic-based media reference for images and video frames. Supports file paths, URLs, data URIs, and video timestamps. Designed for dataset metadata and lazy loading.

## Installation

```bash
# Core package (image loading, batch loading)
pip install mediaref

# With video support (video frame extraction)
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

# Batch decoding with automatic caching (default: PyAV decoder)
refs = [MediaRef(uri="video.mp4", pts_ns=i*1e9) for i in range(10)]
frames = batch_decode(refs)                              # Uses batch decoding API

# Use TorchCodec decoder for GPU acceleration (requires torchcodec>=0.4.0)
frames = batch_decode(refs, decoder="torchcodec")

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

- `batch_decode(refs: list[MediaRef], **kwargs) -> list[np.ndarray]` - Batch decode using optimized batch decoding API
- `cleanup_cache()` - Clear video container cache

### Video Decoders (requires `[video]` extra)

- `PyAVVideoDecoder(video_path)` - PyAV-based decoder with TorchCodec-compatible interface
- `TorchCodecVideoDecoder(video_path)` - TorchCodec-based decoder (requires `torchcodec>=0.4.0`)

## Design Notes

- Video container caching uses reference counting with LRU eviction (default: 10 containers)
- MCAP file path resolution: detects `.mcap` suffix and uses parent directory as base
- Garbage collection triggered every 10 PyAV operations to handle reference cycles
- Cache size configurable via `AV_CACHE_SIZE` environment variable

## Acknowledgments

The video decoder interface design references [TorchCodec](https://github.com/pytorch/torchcodec)'s API design.

## Dependencies

**Core:** `pydantic>=2.0` (requires Pydantic v2 API)

**Loader (optional):** `numpy`, `opencv-python`, `pillow`, `av`, `requests`

The loader dependencies use stable APIs with no version constraints. Install with `pip install mediaref[loader]` to enable batch loading and video decoding.

