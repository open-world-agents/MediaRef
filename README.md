# MediaRef

[![CI](https://img.shields.io/github/actions/workflow/status/open-world-agents/MediaRef/ci.yml?branch=main&logo=github&label=CI)](https://github.com/open-world-agents/MediaRef/actions?query=event%3Apush+branch%3Amain+workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/mediaref.svg)](https://pypi.python.org/pypi/mediaref)
[![versions](https://img.shields.io/pypi/pyversions/mediaref.svg)](https://github.com/open-world-agents/MediaRef)
[![license](https://img.shields.io/github/license/open-world-agents/MediaRef.svg)](https://github.com/open-world-agents/MediaRef/blob/main/LICENSE)

<!-- [![downloads](https://static.pepy.tech/badge/mediaref/month)](https://pepy.tech/project/mediaref) -->

Pydantic media reference for images and video frames (with timestamp support) from data URIs, HTTP URLs, file URIs, cloud storage, and local paths. Features lazy loading and optimized batch video decoding.

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

Due to lazy loading, MediaRef has **zero CPU and I/O overhead** when the media is not accessed. When you do need to load the media, convenient APIs handle the complexity of multi-source media (local files, URLs, embedded data, cloud storage) with a single unified interface.

When loading multiple frames from the same video, `batch_decode()` opens the video file once and reuses the handle, achieving **4.9× faster throughput** and **2.2× better I/O efficiency** compared to sequential decoding.

<p align="center">
  <img src=".github/assets/decoding_benchmark.png" alt="Decoding Benchmark" width="800">
</p>

> **Benchmark details**: Decoding throughput = decoded frames per second during dataloading; I/O efficiency = inverse of disk I/O operations per frame loaded. Measured on real ML dataloader workloads (Minecraft dataset: 64×5 min episodes, 640×360 @ 20Hz, FSLDataset with 4096 token sequences). See [D2E paper](https://worv-ai.github.io/d2e/) Section 3 and Appendix A for full methodology.

## Installation

**Quick install:**
```bash
# Core package — image loading, cloud-storage URIs (s3://, gs://, hf://, …) via fsspec
pip install mediaref

# With video decoding (adds PyAV)
pip install 'mediaref[video]'

# With HuggingFace datasets integration (registers the MediaRef feature type)
pip install 'mediaref[hf]'

# All extras
pip install 'mediaref[video,hf]'
```

**Add to your project:**
```bash
# Core package
uv add mediaref~=0.5.0

# With video decoding support
uv add 'mediaref[video]~=0.5.0'
```

**Versioning Policy**: MediaRef follows [semantic versioning](https://semver.org/). Patch releases (e.g., 0.5.0 → 0.5.1) contain only bug fixes and performance improvements with **no API changes**. Minor releases (e.g., 0.5.x → 0.6.0) may introduce new features while maintaining backward compatibility. Use `~=0.5.0` to automatically receive patch updates.

## Quick Start

### Basic Usage

```python
from mediaref import MediaRef, DataURI, batch_decode
import numpy as np

# 1. Create references (lightweight, no loading yet)
ref = MediaRef(uri="image.png")                            # Local file
ref = MediaRef(uri="https://example.com/image.jpg")        # HTTP(S) URL
ref = MediaRef(uri="s3://bucket/image.jpg")                # Cloud storage (fsspec)
ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)      # Video frame at 1.0s

# 2. Load media
rgb = ref.to_ndarray()                                     # Returns (H, W, 3) RGB array
pil = ref.to_pil_image()                                   # Returns PIL.Image

# 3. Embed as data URI
data_uri = DataURI.from_image(rgb, format="png")           # e.g., "data:image/png;base64,iVBORw0KG..."
ref = MediaRef(uri=data_uri)                               # Self-contained reference

# 4. Batch decode video frames (opens video once, reuses handle)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)                                # Much faster than loading individually

# 5. Serialize for storage in any container format (Parquet, HDF5, mcap, rosbag, etc.)
json_str = ref.model_dump_json()                           # Lightweight JSON string
# Store in your dataset format of choice - works with any format that stores strings
```

### Batch Decoding - Optimized Video Frame Loading

When loading multiple frames from the same video, use `batch_decode()` to open the video file once and reuse the handle—achieving significantly better performance than loading frames individually.

```python
from mediaref import MediaRef, batch_decode

# Use optimized batch decoding (default: PyAV backend)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)

# Or use TorchCodec for GPU-accelerated decoding
frames = batch_decode(refs, decoder="torchcodec")  # Requires: pip install torchcodec
```

Both decoders follow unified [playback semantics](docs/playback_semantics.md)—querying a timestamp returns the frame being displayed at that moment, ensuring consistent behavior across backends.

### Embedding Media Directly in MediaRef

Embed image data into a `MediaRef` to make it self-contained — useful for serialization, caching, or sharing without external files.

| Input | Constructor |
| --- | --- |
| `numpy` ndarray (RGB) | `DataURI.from_image(rgb, format="png")` |
| File on disk | `DataURI.from_file("image.png")` |
| `PIL.Image` | `DataURI.from_image(pil_img, format="jpeg", quality=90)` |
| `numpy` ndarray (BGR, e.g. `cv2.imread`) | `DataURI.from_image(bgr, format="png", input_format="bgr")` |

> `input_format="bgr"` is required for arrays from OpenCV — `cv2.imread` returns BGR, not RGB. Pass an RGB array without `input_format` and it Just Works.

```python
from mediaref import MediaRef, DataURI
import numpy as np

# Build from any input above (here: numpy RGB)
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))

# Use exactly like any other MediaRef
ref.to_ndarray()                                           # (H, W, 3) RGB array
ref.to_pil_image()                                         # PIL Image

# Serialize — image data lives inside the MediaRef
serialized = ref.model_dump_json()
restored = MediaRef.model_validate_json(serialized)        # no external file needed

# DataURI properties
data_uri = DataURI.from_image(rgb, format="png")
data_uri.mimetype, len(data_uri), data_uri.is_image        # ('image/png', N, True)
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
ref = MediaRef.model_validate(data)                        # From dict
ref = MediaRef.model_validate_json(json_str)               # From JSON
```

## Cloud storage URIs

Any URI whose scheme is not `file://` or `data:` is delegated to [fsspec](https://filesystem-spec.readthedocs.io). `s3://`, `gs://`, `hf://`, `az://`, `webdav://`, `gdrive://`, `ipfs://`, `http(s)://`, and any future fsspec backend all work without scheme-specific code:

```python
from mediaref import MediaRef, batch_decode

ref = MediaRef(uri="s3://my-bucket/episode_42.mp4", pts_ns=1_500_000_000)
frame = ref.to_ndarray()                                    # range read via fsspec — no full download

# batch_decode opens the cloud-backed video once and decodes many frames
refs = [MediaRef(uri="hf://datasets/me/clips/cam.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)
```

`fsspec` is a core dependency. Each cloud backend (`s3fs` for `s3://`, `gcsfs` for `gs://`, `huggingface_hub` for `hf://`, `adlfs` for `az://`/`abfs://`, …) must be installed separately for the schemes it serves; fsspec raises a clear error otherwise.

## HuggingFace `datasets` integration

`mediaref.hf` registers `MediaRef` as a first-class `datasets` feature (Arrow storage `struct<uri: string, pts_ns: int64>`). The feature type survives every official `datasets` persistence path — `save_to_disk`, `push_to_hub`, parquet export — so the column you write as `MediaRef` reads back as `MediaRef` anywhere consumers have `mediaref[hf]` installed.

```python
# pip install 'mediaref[hf]'
from datasets import Dataset, Features, load_from_disk, load_dataset
from mediaref import MediaRef
from mediaref.hf import MediaRefFeature

# 1. Create — pass MediaRef objects (or dicts) and declare the feature type.
ds = Dataset.from_dict(
    {"frame": [MediaRef(uri="video.mp4", pts_ns=0),
               MediaRef(uri="video.mp4", pts_ns=33_333_333)]},
    features=Features({"frame": MediaRefFeature()}),
)
ds[0]["frame"]                       # MediaRef(uri='video.mp4', pts_ns=0)  ← lazy

# 2. Local persistence (canonical save/load, uses Arrow IPC + features JSON).
ds.save_to_disk("path/to/ds")
ds2 = load_from_disk("path/to/ds")
assert isinstance(ds2.features["frame"], MediaRefFeature)
ds2[0]["frame"].to_ndarray()         # decode lazily on access

# 3. Sharing via the Hub (push_to_hub / load_dataset). Same guarantee —
#    the registered feature name is "MediaRef", so consumers get MediaRef
#    objects back automatically.
# ds.push_to_hub("my-org/my-dataset")
# load_dataset("my-org/my-dataset")["train"][0]["frame"]
#   → MediaRef(uri='video.mp4', pts_ns=0)

# (parquet export also preserves the feature via Arrow schema metadata —
#  use `ds.to_parquet(...)` / `Dataset.from_parquet(...)` if you need that
#  specific format.)
```

`MediaRefFeature(decode=False)` returns the raw `{"uri": ..., "pts_ns": ...}` dict instead of a `MediaRef` instance — useful when you want to defer object construction or pass values straight into PyArrow compute.

## lerobot interop

`mediaref.compat.lerobot` converts to and from lerobot's `VideoFrame` representation (`{path, timestamp seconds}`) and reconstructs MediaRefs from a v3.0 LeRobotDataset episode without needing lerobot installed:

```python
from mediaref.compat.lerobot import (
    from_videoframe, to_videoframe, lerobot_episode_to_refs,
)

# 1. Convert a single VideoFrame dict
ref = from_videoframe({"path": "videos/clip.mp4", "timestamp": 0.5})
# MediaRef(uri='videos/clip.mp4', pts_ns=500000000)

# 2. Build refs for an entire episode in a v3.0 LeRobotDataset shared mp4 shard
refs = lerobot_episode_to_refs(
    video_path="videos/observation.images.front_left/chunk-000/file-000.mp4",
    from_timestamp=12.34,           # meta.episodes[ep_idx][f"videos/{vid_key}/from_timestamp"]
    frame_timestamps=[0.0, 1/30, 2/30],  # episode-local timestamps
)
```

## Documentation

- **[API Reference](docs/API.md)** - Detailed API documentation
- **[Playback Semantics](docs/playback_semantics.md)** - How frame selection works at specific timestamps

## Potential Future Enhancements

- [ ] **msgspec support**: Replace pydantic BaseModel into [msgspec](https://jcristharif.com/msgspec/)
- [ ] **Thread-safe resource caching**: Implement thread-safe `ResourceCache` for concurrent video decoding workloads
- [ ] **Audio support**: Extend MediaRef to support audio references with timestamp-based extraction
- [ ] **Additional video decoders**: Support for more decoder backends (e.g., OpenCV, decord)

## Dependencies

**Core dependencies** (automatically installed):
- `pydantic>=2.0` - Data validation and serialization (requires Pydantic v2 API)
- `numpy` - Array operations
- `opencv-python` - Image loading and color conversion
- `pillow>=9.4.0` - Image loading from various sources
- `fsspec[http]>=2024.2.0` - `http(s)://`, `s3://`, `gs://`, `hf://`, … URI dispatch
- `loguru` - Logging (disabled by default for library code)

**Optional dependencies**:
- `[video]` extra: `av>=15.0` (PyAV for video frame extraction, 15.0+ for FFmpeg 7.0 support)
- `[hf]` extra: `datasets>=2.14.0` + `pyarrow` (HuggingFace datasets feature registration)
- TorchCodec: `torchcodec` (install separately for GPU-accelerated decoding)
- Per-backend cloud storage extras: `s3fs` (s3://), `gcsfs` (gs://), `huggingface_hub` (hf://), `adlfs` (az://, abfs://). See [filesystem-spec docs](https://filesystem-spec.readthedocs.io).

## Acknowledgments

The video decoder interface design references [TorchCodec](https://github.com/pytorch/torchcodec)'s API design.

## License

MediaRef is released under the [MIT License](LICENSE).
