# MediaRef

[![CI](https://img.shields.io/github/actions/workflow/status/open-world-agents/MediaRef/ci.yml?branch=main&logo=github&label=CI)](https://github.com/open-world-agents/MediaRef/actions?query=event%3Apush+branch%3Amain+workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/mediaref.svg)](https://pypi.python.org/pypi/mediaref)
[![versions](https://img.shields.io/pypi/pyversions/mediaref.svg)](https://github.com/open-world-agents/MediaRef)
[![license](https://img.shields.io/github/license/open-world-agents/MediaRef.svg)](https://github.com/open-world-agents/MediaRef/blob/main/LICENSE)

<!-- [![downloads](https://static.pepy.tech/badge/mediaref/month)](https://pepy.tech/project/mediaref) -->

Pydantic media reference for images and video frames (with timestamp support) from data URIs, HTTP URLs, file URIs, and local paths. Features lazy loading and optimized batch video decoding.

Works with any container format (Parquet, HDF5, mcap, rosbag, etc.) and any media format (JPEG, PNG, H.264, H.265, AV1, etc.).

## Why MediaRef?

**1. Separate heavy media from lightweight metadata**

Store 1TB of videos separately while keeping only 1MB of references in your dataset tables. Break free from rigid structures where media must be embedded inside tables—MediaRef enables flexible, decoupled storage architectures for any format that stores strings.

```python
# Store lightweight references in your dataset, not heavy media
import pandas as pd

# Image references: 37 bytes vs entire embedded image(>100KB)
df_images = pd.DataFrame([
    {"action": [0.1, 0.2], "observation": MediaRef(uri="frame_001.png").model_dump()},
    {"action": [0.3, 0.4], "observation": MediaRef(uri="frame_002.png").model_dump()},
])

# Video frame references: 35-42 bytes vs entire video file embedded(several GBs)
df_video = pd.DataFrame([
    {"action": [0.1, 0.2], "observation": MediaRef(uri="episode_01.mp4", pts_ns=0).model_dump()},
    {"action": [0.3, 0.4], "observation": MediaRef(uri="episode_01.mp4", pts_ns=50_000_000).model_dump()},
])

# Works with any container format (Parquet, HDF5, mcap, rosbag, etc.)
# and any media format (JPEG, PNG, H.264, H.265, AV1, etc.)
```

MediaRef is already used in production ML data formats at scale. For example, the [D2E research project](https://worv-ai.github.io/d2e/) uses MediaRef via [OWAMcap](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/) to store **10TB+** of gameplay data with [screen observations](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-msgs/owa/msgs/desktop/screen.py#L49).

**2. Future-proof specification built on standards**

The MediaRef schema(`uri`, `pts_ns`) is designed to be **permanent**, built entirely on established standards ([RFC 2397](https://datatracker.ietf.org/doc/html/rfc2397) for data URIs, [RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986) for URI syntax). Use it anywhere with confidence—no proprietary formats, no breaking changes.

**3. Optimized performance where it matters**

Due to lazy lazy loading, MediaRef has **zero CPU and I/O overhead** when the media is not accessed. When you do need to load the media, convenient APIs handle the complexity of multi-source media (local files, URLs, embedded data) with a single unified interface.

When loading multiple frames from the same video, `batch_decode()` opens the video file once and reuses the handle with adaptive batching strategies that automatically optimize decoding based on frame access patterns, achieving **4.9× faster throughput** and **41× better I/O efficiency** compared to existing methods.

<p align="center">
  <img src=".github/assets/decoding_benchmark.png" alt="Decoding Benchmark" width="800">
</p>

> **Benchmark details**: Decoding throughput = decoded frames per second during dataloading; I/O efficiency = inverse of disk I/O operations per frame loaded. Measured on real ML dataloader workloads (Minecraft dataset: 64×5 min episodes, 640×360 @ 20Hz, FSLDataset with 4096 token sequences) vs baseline and TorchCodec v0.6.0. See [D2E paper](https://worv-ai.github.io/d2e/) Section 3 and Appendix A for full methodology.

## Installation

**Quick install:**
```bash
# Core package with image loading support
pip install mediaref

# With video decoding support (adds PyAV for video frame extraction)
pip install mediaref[video]
```

**Add to your project:**
```bash
# Core package
uv add mediaref~=0.4.1

# With video decoding support
uv add 'mediaref[video]~=0.4.1'
```

**Versioning Policy**: MediaRef follows [semantic versioning](https://semver.org/). Patch releases (e.g., 0.4.1 → 0.4.2) contain only bug fixes and performance improvements with **no API changes**. Minor releases (e.g., 0.4.x → 0.5.0) may introduce new features while maintaining backward compatibility. Use `~=0.4.1` to automatically receive patch updates.

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
rgb = ref.to_ndarray()                                 # Returns (H, W, 3) RGB array
pil = ref.to_pil_image()                               # Returns PIL.Image

# 3. Embed as data URI
data_uri = DataURI.from_image(rgb, format="png")       # e.g., "data:image/png;base64,iVBORw0KG..."
ref = MediaRef(uri=data_uri)                           # Self-contained reference

# 4. Batch decode video frames (opens video once, reuses handle)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)                            # Much faster than loading individually

# 5. Serialize for storage in any container format (Parquet, HDF5, mcap, rosbag, etc.)
json_str = ref.model_dump_json()                       # Lightweight JSON string
# Store in your dataset format of choice - works with any format that stores strings
```

### Batch Decoding - Optimized Video Frame Loading

When loading multiple frames from the same video, use `batch_decode()` to open the video file once and reuse the handle with adaptive batching strategies that automatically optimize decoding based on frame access patterns—achieving significantly better performance than loading frames individually.

```python
from mediaref import MediaRef, batch_decode
from mediaref.video_decoder import BatchDecodingStrategy

# Use optimized batch decoding with adaptive strategy (default, recommended)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(
    refs,
    # Our optimized implementation based on PyAV
    decoder="pyav",
    # Our adaptive strategy for optimal performance
    strategy=BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK
)

# Or use TorchCodec for GPU-accelerated decoding
frames = batch_decode(refs, decoder="torchcodec")  # Requires: pip install torchcodec>=0.4.0
```

### Embedding Media Directly in MediaRef

You can embed image data directly into `MediaRef` objects, making them self-contained and portable (useful for serialization, caching, or sharing).

```python
from mediaref import MediaRef, DataURI
import numpy as np

# Create embedded MediaRef from numpy array
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
embedded_ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))

# Or from file
embedded_ref = MediaRef(uri=DataURI.from_file("image.png"))

# Or from PIL Image
from PIL import Image
pil_img = Image.open("image.png")
embedded_ref = MediaRef(uri=DataURI.from_image(pil_img, format="jpeg", quality=90))

# Or from BGR array (OpenCV uses BGR by default - input_format="bgr" is REQUIRED)
import cv2
bgr_array = cv2.imread("image.jpg")  # OpenCV loads as BGR, not RGB!
embedded_ref = MediaRef(uri=DataURI.from_image(bgr_array, format="png", input_format="bgr"))

# Use just like any other MediaRef
rgb = embedded_ref.to_ndarray()                        # (H, W, 3) RGB array
pil = embedded_ref.to_pil_image()                      # PIL Image

# Serialize with embedded data
serialized = embedded_ref.model_dump_json()            # Contains image data
restored = MediaRef.model_validate_json(serialized)    # No external file needed!

# Properties
print(data_uri.mimetype)                               # "image/png"
print(len(data_uri))                                   # URI length in bytes
print(data_uri.is_image)                               # True for image/* types
```

### Path Resolution & Serialization

Resolve relative paths and serialize MediaRef objects for storage in any container format (Parquet, HDF5, mcap, rosbag, etc.).

```python
# Resolve relative paths
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
resolved = ref.resolve_relative_path("/data/recordings")

# Handle unresolvable URIs (embedded/remote)
remote = MediaRef(uri="https://example.com/image.jpg")
resolved = remote.resolve_relative_path("/data", on_unresolvable="ignore")  # No warning

# Serialization (Pydantic-based) - works with any container format
ref = MediaRef(uri="video.mp4", pts_ns=1_500_000_000)

# As dict (for Python-based formats)
data = ref.model_dump()
# Output: {'uri': 'video.mp4', 'pts_ns': 1500000000}

# As JSON string (for Parquet, HDF5, mcap, rosbag, etc.)
json_str = ref.model_dump_json()
# Output: '{"uri":"video.mp4","pts_ns":1500000000}'

# Deserialization
ref = MediaRef.model_validate(data)                    # From dict
ref = MediaRef.model_validate_json(json_str)           # From JSON
```

## API Reference

See [API Documentation](docs/API.md) for detailed API reference.

## Potential Future Enhancements

- [ ] **HuggingFace datasets integration**: Add native `MediaRef` feature type to [HuggingFace datasets](https://github.com/huggingface/datasets) for seamless integration with the ML ecosystem
- [ ] **msgspec support**: Replace pydantic BaseModel into [msgspec](https://jcristharif.com/msgspec/)
- [ ] **Thread-safe resource caching**: Implement thread-safe `ResourceCache` for concurrent video decoding workloads
- [ ] **Audio support**: Extend MediaRef to support audio references with timestamp-based extraction
- [ ] **Cloud storage support**: Integrate `fsspec` for cloud URIs (e.g., `s3://`, `gs://`, `az://`)
- [ ] **Additional video decoders**: Support for more decoder backends (e.g., OpenCV, decord)

## Dependencies

**Core dependencies** (automatically installed):
- `pydantic>=2.0` - Data validation and serialization (requires Pydantic v2 API)
- `numpy` - Array operations
- `opencv-python-headless` - Image loading and color conversion
- `pillow>=9.4.0` - Image loading from various sources
- `requests>=2.32.2` - HTTP/HTTPS URL loading
- `loguru` - Logging (disabled by default for library code)

**Optional dependencies**:
- `[video]` extra: `av>=15.0` (PyAV for video frame extraction)
- TorchCodec: `torchcodec>=0.4.0` (install separately for GPU-accelerated decoding)

## Acknowledgments

The video decoder interface design references [TorchCodec](https://github.com/pytorch/torchcodec)'s API design.

## License

MediaRef is released under the [MIT License](LICENSE).

