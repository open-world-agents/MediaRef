# MediaRef

[![CI](https://img.shields.io/github/actions/workflow/status/open-world-agents/MediaRef/ci.yml?branch=main&logo=github&label=CI)](https://github.com/open-world-agents/MediaRef/actions?query=event%3Apush+branch%3Amain+workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/mediaref.svg)](https://pypi.python.org/pypi/mediaref)
[![versions](https://img.shields.io/pypi/pyversions/mediaref.svg)](https://github.com/open-world-agents/MediaRef)
[![license](https://img.shields.io/github/license/open-world-agents/MediaRef.svg)](https://github.com/open-world-agents/MediaRef/blob/main/LICENSE)

<!-- [![downloads](https://static.pepy.tech/badge/mediaref/month)](https://pepy.tech/project/mediaref) -->

**The portable frame-level media reference primitive — container-agnostic, fps-free, RFC-based.**

`(uri, pts_ns)` is the entire schema. URIs follow [RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986) (with [RFC 2397](https://datatracker.ietf.org/doc/html/rfc2397) for embedded data); `pts_ns` is an int64 nanosecond presentation timestamp. The schema is frozen for the life of [MediaRef Spec 1.x](docs/SPEC.md). Works in any container (Parquet, mcap, rosbag, HDF5) and any standard media format (JPEG, PNG, H.264, H.265, AV1).

## How MediaRef relates to other media features

|                       | `datasets.Video`                                   | lerobot `VideoFrame`                                                | **MediaRef**                                                              |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Granularity           | one row = one **video file**                       | one row = one frame, **inside LeRobotDataset's layout**             | one row = **one frame, anywhere**                                         |
| Storage assumption    | Embedded `bytes` + `path`                          | Path templates + per-episode metadata; fixed global fps             | None — just `(uri, pts_ns)`                                               |
| Container             | Parquet                                            | LeRobotDataset directory layout                                     | Any (Parquet, mcap, rosbag, HDF5, JSON, …)                                |
| Standards basis       | Arrow / Parquet schema                             | Arrow / Parquet schema                                              | RFC 3986 / RFC 2397                                                       |
| Cross-call batching for many frames per video | One `VideoDecoder` per row; multi-frame access from that decoder | LRU-cached decoders + episode-shard prefetch inside lerobot's loaders | `batch_decode()` groups refs by URI and opens each container once: [4.9× throughput, 2.2× I/O on a sparse-frame ML dataloader workload](#batch-decoding---optimized-video-frame-loading) |
| Hub-native            | yes                                                | yes (lerobot-style datasets)                                        | yes via `mediaref.hf` (`register_feature("MediaRef", …)`)                 |

**When to use which.** If your row *is* a whole clip you'll decode end-to-end, `datasets.Video` is the simplest answer. If your data fits LeRobotDataset's layout (single global fps, episode-based directory structure), `VideoFrame` is the right primitive. MediaRef is the in-between case: per-frame references with no assumptions about container, fps, or directory layout. See [`docs/SPEC.md`](docs/SPEC.md) for the wire format.

## Why MediaRef?

**1. Separate heavy media from lightweight metadata**

Store 1TB of videos separately while keeping only 1MB of references in your dataset tables. Break free from rigid structures where media must be embedded inside tables — MediaRef enables flexible, decoupled storage architectures for any format that stores strings.

```python
# Store lightweight references in your dataset, not heavy media
import pandas as pd
from mediaref import MediaRef

# Image references: 37 bytes vs entire embedded image (>100KB)
df_images = pd.DataFrame([
    {"action": [0.1, 0.2], "observation": MediaRef(uri="frame_001.png").model_dump()},
    {"action": [0.3, 0.4], "observation": MediaRef(uri="frame_002.png").model_dump()},
])

# Video frame references: 35-42 bytes vs entire video file embedded (several GBs)
df_video = pd.DataFrame([
    {"action": [0.1, 0.2], "observation": MediaRef(uri="episode_01.mp4", pts_ns=0).model_dump()},
    {"action": [0.3, 0.4], "observation": MediaRef(uri="episode_01.mp4", pts_ns=50_000_000).model_dump()},
])

# Works with any container format (Parquet, HDF5, mcap, rosbag, etc.)
# and any media format (JPEG, PNG, H.264, H.265, AV1, etc.)
```

MediaRef is already used in production ML data formats at scale. For example, the [D2E research project](https://worv-ai.github.io/d2e/) uses MediaRef via [OWAMcap](https://open-world-agents.github.io/open-world-agents/data/technical-reference/format-guide/) to store **10TB+** of gameplay data with [screen observations](https://github.com/open-world-agents/open-world-agents/blob/main/projects/owa-msgs/owa/msgs/desktop/screen.py#L49). See [Datasets shipped with MediaRef](#datasets-shipped-with-mediaref) for the full list.

**2. Future-proof specification built on standards**

The MediaRef schema (`uri`, `pts_ns`) is **permanent** under semantic versioning, built entirely on RFCs. Use it anywhere with confidence — no proprietary formats, no breaking changes. The full grammar, semantics, and conformance criteria are in [`docs/SPEC.md`](docs/SPEC.md).

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

# With video decoding
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
# Store in your dataset format of choice — works with any format that stores strings
```

### Batch Decoding - Optimized Video Frame Loading

When loading multiple frames from the same video, use `batch_decode()` to open the video file once and reuse the handle — achieving significantly better performance than loading frames individually.

```python
from mediaref import MediaRef, batch_decode

# Use optimized batch decoding (default: PyAV backend)
refs = [MediaRef(uri="video.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)

# Or use TorchCodec for GPU-accelerated decoding
frames = batch_decode(refs, decoder="torchcodec")  # Requires: pip install torchcodec
```

Both decoders follow unified [playback semantics](docs/playback_semantics.md) — querying a timestamp returns the frame being displayed at that moment, ensuring consistent behavior across backends.

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
from PIL import Image
import numpy as np

# Build from any input above (here: numpy RGB)
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))

# PIL.Image input requires you to obtain a PIL.Image first:
pil_img = Image.open("image.png")
ref_pil = MediaRef(uri=DataURI.from_image(pil_img, format="jpeg", quality=90))

# Use exactly like any other MediaRef
ref.to_ndarray()                                           # (H, W, 3) RGB array
ref.to_pil_image()                                         # PIL Image

# Serialize — image data lives inside the MediaRef
serialized = ref.model_dump_json()
restored = MediaRef.model_validate_json(serialized)        # no external file needed

# DataURI properties
data_uri = DataURI.from_image(rgb, format="png")
print(data_uri.mimetype)                                   # "image/png"
print(len(data_uri))                                       # length of URI in bytes
print(data_uri.is_image)                                   # True for image/* mimetypes
```

### Path Resolution & Serialization

Resolve relative paths and serialize MediaRef objects for storage in any container format (Parquet, HDF5, mcap, rosbag, etc.).

```python
# Resolve relative paths
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
resolved = ref.resolve_relative_path("/data/recordings")

# Handle unresolvable URIs (embedded/cloud)
remote = MediaRef(uri="https://example.com/image.jpg")
resolved = remote.resolve_relative_path("/data", on_unresolvable="ignore")  # No warning

# Serialization (Pydantic-based) — works with any container format
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

`mediaref.hf` registers `MediaRef` as a first-class `datasets` feature (Arrow `struct<uri: string, pts_ns: int64>`), preserved across `save_to_disk`, `push_to_hub`, and parquet export.

```python
# pip install 'mediaref[hf]'
from datasets import Dataset, Features, load_from_disk
from mediaref import MediaRef
from mediaref.hf import MediaRefFeature

ds = Dataset.from_dict(
    {"frame": [MediaRef(uri="video.mp4", pts_ns=0),
               MediaRef(uri="video.mp4", pts_ns=33_333_333)]},
    features=Features({"frame": MediaRefFeature()}),
)
ds.save_to_disk("path/to/ds")
load_from_disk("path/to/ds")[0]["frame"].to_ndarray()  # round-trips as MediaRef
```

`push_to_hub` / `load_dataset` round-trip identically. Pass `MediaRefFeature(decode=False)` to receive the raw `{"uri", "pts_ns"}` dict instead of a `MediaRef` instance.

### Required: register the feature in the consumer process

`datasets` has no feature autodiscovery — `load_from_disk` / `load_dataset` raises `ValueError: Feature type 'MediaRef' not found` unless the consumer process imports `mediaref.hf` first. Two options:

- **Explicit import.** Add `from mediaref.hf import MediaRefFeature` before any load call.
- **Permanent CLI patch.** `mediaref enable-hf-feature` (idempotent; re-run after `pip upgrade datasets`) source-patches `datasets/features/features.py` so MediaRef auto-registers on every `import datasets`. Reverse with `mediaref disable-hf-feature`; check with `mediaref status`. Same pattern as this repo's `patch_torchcodec`.

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

## Datasets shipped with MediaRef

These are projects from the author's own work that use MediaRef on the storage path. External adopters welcome — open a PR to add yours.

| Dataset | Domain | Scale |
| --- | --- | --- |
| [open-world-agents/D2E-Original](https://huggingface.co/datasets/open-world-agents/D2E-Original) | Game agents (29 PC games) | 273.4 hours, 1.83 TB |
| [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p) | Game agents (downsampled) | — |
| [maum-ai/CostNav-Teleop-Dataset](https://huggingface.co/datasets/maum-ai/CostNav-Teleop-Dataset) | Delivery-robot navigation / teleop | — |

Tagging a HuggingFace dataset with `mediaref` makes it discoverable at [huggingface.co/datasets?other=mediaref](https://huggingface.co/datasets?other=mediaref).

## Documentation

- **[MediaRef Specification 1.0](docs/SPEC.md)** — wire format, URI grammar, `pts_ns` semantics, conformance criteria
- **[API Reference](docs/API.md)** — detailed API documentation
- **[Playback Semantics](docs/playback_semantics.md)** — how frame selection works at specific timestamps

## Citation

If you reference MediaRef in writing, the [`CITATION.cff`](CITATION.cff) file at repo root has the canonical metadata. BibTeX:

```bibtex
@software{mediaref,
  author = {Choi, Suhwan},
  title  = {MediaRef: a portable frame-level media reference primitive},
  url    = {https://github.com/open-world-agents/MediaRef},
  year   = {2026}
}
```

## Potential Future Enhancements

- [ ] **msgspec support**: Replace pydantic BaseModel into [msgspec](https://jcristharif.com/msgspec/)
- [ ] **Thread-safe resource caching**: Implement thread-safe `ResourceCache` for concurrent video decoding workloads
- [ ] **Audio support**: Extend MediaRef to support audio references with timestamp-based extraction
- [ ] **Additional video decoders**: Support for more decoder backends (e.g., OpenCV, decord)

## Dependencies

**Core dependencies** (automatically installed):
- `pydantic>=2.0` — data validation and serialization (requires Pydantic v2 API)
- `numpy` — array operations
- `opencv-python` — image loading and color conversion
- `pillow>=9.4.0` — image loading from various sources
- `fsspec[http]>=2024.2.0` — `http(s)://`, `s3://`, `gs://`, `hf://`, …  URI dispatch
- `loguru` — logging (disabled by default for library code)

**Optional dependencies**:
- `[video]` extra: `av>=15.0` (PyAV for video frame extraction; 15.0+ for FFmpeg 7.0 support)
- `[hf]` extra: `datasets>=2.14.0` + `pyarrow` (HuggingFace datasets feature registration)
- TorchCodec: `torchcodec` (install separately for GPU-accelerated decoding)
- Per-backend cloud storage extras: `s3fs` (s3://), `gcsfs` (gs://), `huggingface_hub` (hf://), `adlfs` (az://, abfs://). See [filesystem-spec docs](https://filesystem-spec.readthedocs.io).

## Acknowledgments

The video decoder interface design references [TorchCodec](https://github.com/pytorch/torchcodec)'s API design.

## License

MediaRef is released under the [MIT License](LICENSE).
