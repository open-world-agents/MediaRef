# API Reference

- [`MediaRef`](#mediaref)
- [`DataURI`](#datauri)
- [`batch_decode`](#batch_decode)
- [Cloud storage URIs](#cloud-storage-uris)
- [HuggingFace `datasets` integration](#huggingface-datasets-integration)
- [lerobot interop](#lerobot-interop)
- [CLI](#cli)

---

## `MediaRef`

```python
class MediaRef(BaseModel):
    uri: str
    pts_ns: int | None = None
```

A two-field Pydantic v2 model. URIs follow [RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986); `pts_ns` is an int64 nanosecond presentation timestamp. See [`SPEC.md`](SPEC.md) for the wire-format specification.

```python
from mediaref import MediaRef

ref = MediaRef(uri="image.png")
ref = MediaRef(uri="https://example.com/img.jpg")
ref = MediaRef(uri="s3://bucket/clip.mp4", pts_ns=1_500_000_000)
ref = MediaRef(uri="data:image/png;base64,iVBORw0…")
```

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `is_embedded` | `bool` | True if the URI is a `data:` URI carrying embedded bytes. |
| `is_video` | `bool` | True if `pts_ns is not None`. |
| `is_remote` | `bool` | True if the URI is `http://` or `https://`. |
| `is_relative_path` | `bool` | True if the URI is a relative POSIX path. |

### Methods

`to_ndarray(format="rgb") -> np.ndarray`
- Loads the media as a numpy array in the requested format.
- Formats: `"rgb"` (default), `"bgr"`, `"rgba"`, `"bgra"`, `"gray"`.
- Returns shape: `(H, W, 3)` for RGB/BGR, `(H, W, 4)` for RGBA/BGRA, `(H, W)` for grayscale.
- For video URIs (`pts_ns is not None`), decodes the single frame at that timestamp.

`to_pil_image(format="rgb") -> PIL.Image`
- Same as `to_ndarray` but returns a PIL Image. Formats: `"rgb"`, `"rgba"`, `"gray"`.

`resolve_relative_path(base_path, on_unresolvable="warn") -> MediaRef`
- Returns a new `MediaRef` with the relative path resolved against `base_path`.
- `on_unresolvable`: `"error"`, `"warn"` (default), or `"ignore"` — controls behavior for embedded/cloud URIs which can't be resolved against a base path.

```python
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
ref.resolve_relative_path("/data/recordings")
# MediaRef(uri='/data/recordings/relative/video.mkv', pts_ns=123456)

remote = MediaRef(uri="https://example.com/image.jpg")
remote.resolve_relative_path("/data", on_unresolvable="ignore")  # returned unchanged
```

`validate_uri() -> bool` — checks if the URI exists (local files only).

### Serialization

Standard Pydantic v2 model methods. The dict / JSON form is the canonical wire representation — store it in any string-holding format (Parquet, HDF5, mcap, rosbag, JSON, Postgres `jsonb`, …).

```python
ref = MediaRef(uri="video.mp4", pts_ns=1_500_000_000)

ref.model_dump()        # {'uri': 'video.mp4', 'pts_ns': 1500000000}
ref.model_dump_json()   # '{"uri":"video.mp4","pts_ns":1500000000}'

MediaRef.model_validate({"uri": "video.mp4", "pts_ns": 0})
MediaRef.model_validate_json('{"uri":"video.mp4","pts_ns":0}')
```

---

## `DataURI`

For embedding media bytes directly inside a `MediaRef`. Useful for self-contained, serializable references that don't depend on external files.

### Construction

`DataURI.from_image(image, format="png", quality=None, input_format="rgb") -> DataURI`
- `image`: a `numpy.ndarray` or `PIL.Image`.
- `format`: output media format — `"png"`, `"jpeg"`, or `"bmp"`.
- `quality`: JPEG quality 1–100 (ignored for PNG/BMP).
- `input_format`: input channel order for numpy arrays. `"rgb"` (default), `"bgr"`, `"rgba"`, `"bgra"`. **Required as `"bgr"`** when passing the result of `cv2.imread`, which returns BGR. Ignored for `PIL.Image`.

PNG preserves alpha; JPEG and BMP drop it.

`DataURI.from_file(path, format=None) -> DataURI` — read raw bytes from disk and wrap.

`DataURI.from_uri(uri) -> DataURI` — parse an existing `data:…` URI string.

### Examples

```python
from mediaref import MediaRef, DataURI
from PIL import Image
import cv2
import numpy as np

# numpy RGB
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))

# OpenCV BGR — input_format is REQUIRED
bgr = cv2.imread("photo.jpg")
ref = MediaRef(uri=DataURI.from_image(bgr, format="png", input_format="bgr"))

# PIL.Image
pil_img = Image.open("photo.png")
ref = MediaRef(uri=DataURI.from_image(pil_img, format="jpeg", quality=90))

# File on disk
ref = MediaRef(uri=DataURI.from_file("photo.png"))
```

### Methods and properties

`to_ndarray(format="rgb") -> np.ndarray` — decode to numpy. Same formats as `MediaRef.to_ndarray`.

`to_pil_image() -> PIL.Image` — decode to PIL Image.

| Property | Description |
| --- | --- |
| `uri` | Full data URI string. |
| `mimetype` | e.g. `"image/png"`. |
| `is_image` | True for `image/*` MIME types. |
| `len(data_uri)` | URI length in bytes. |

---

## `batch_decode`

```python
batch_decode(refs, decoder="pyav") -> list[np.ndarray]
```

Decode many `MediaRef` video frames efficiently by grouping refs that share a URI, opening each container once, and seeking through the requested timestamps in order. Significantly faster than per-ref decoding when refs cluster on the same video file.

```python
from mediaref import MediaRef, batch_decode

refs = [MediaRef(uri="episode.mp4", pts_ns=int(i * 1e9)) for i in range(10)]
frames = batch_decode(refs)                          # default: PyAV (CPU)
frames = batch_decode(refs, decoder="torchcodec")    # GPU-accelerated
```

### Decoder backends

| | `"pyav"` (default) | `"torchcodec"` |
| --- | --- | --- |
| Backend | PyAV (FFmpeg) | TorchCodec (FFmpeg) |
| Acceleration | CPU only | CUDA |
| Install | `pip install 'mediaref[video]'` | `pip install torchcodec` separately |

Both backends share unified [playback semantics](playback_semantics.md), so a given `pts_ns` returns the same frame regardless of decoder.

`cleanup_cache()` — clears the PyAV container cache. Call between long-running decode sessions if you want to release decoder memory before automatic eviction.

### Direct decoder use

For finer control, use the decoder classes directly:

```python
from mediaref.decoders import PyAVVideoDecoder, TorchCodecVideoDecoder

with PyAVVideoDecoder("episode.mp4") as dec:
    frame = dec.get_frame_at_pts_ns(1_500_000_000)
```

---

## Cloud storage URIs

Any URI whose scheme is not `file://` or `data:` is delegated to [fsspec](https://filesystem-spec.readthedocs.io). `s3://`, `gs://`, `hf://`, `az://`, `webdav://`, `gdrive://`, `ipfs://`, `http(s)://`, and any future fsspec backend all work without scheme-specific code:

```python
from mediaref import MediaRef, batch_decode

ref = MediaRef(uri="s3://my-bucket/episode.mp4", pts_ns=1_500_000_000)
frame = ref.to_ndarray()    # range read via fsspec — no full download

refs = [MediaRef(uri="hf://datasets/me/clips/cam.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs)
```

`fsspec` is a core dependency. Each cloud backend (`s3fs` for `s3://`, `gcsfs` for `gs://`, `huggingface_hub` for `hf://`, `adlfs` for `az://`/`abfs://`, …) must be installed separately for the schemes it serves; fsspec raises a clear error otherwise.

---

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

---

## lerobot interop

`mediaref.compat.lerobot` converts to and from lerobot's `VideoFrame` representation (`{path, timestamp seconds}`) and reconstructs MediaRefs from a v3.0 LeRobotDataset episode without needing lerobot installed:

```python
from mediaref.compat.lerobot import (
    from_videoframe, to_videoframe, lerobot_episode_to_refs,
)

# Convert a single VideoFrame dict
ref = from_videoframe({"path": "videos/clip.mp4", "timestamp": 0.5})
# MediaRef(uri='videos/clip.mp4', pts_ns=500000000)

# Build refs for an entire episode in a v3.0 LeRobotDataset shared mp4 shard
refs = lerobot_episode_to_refs(
    video_path="videos/observation.images.front_left/chunk-000/file-000.mp4",
    from_timestamp=12.34,                # meta.episodes[ep_idx][f"videos/{vid_key}/from_timestamp"]
    frame_timestamps=[0.0, 1/30, 2/30],  # episode-local timestamps
)
```

---

## CLI

The `mediaref` CLI handles environment-level concerns:

| Command | Effect |
| --- | --- |
| `mediaref enable-hf-feature` | Source-patch the installed `datasets` package to auto-register `MediaRefFeature` on every `import datasets`. Idempotent. |
| `mediaref disable-hf-feature` | Reverse the patch. |
| `mediaref status` | Show whether the patch is currently applied. |

Re-run `enable-hf-feature` after `pip upgrade datasets` (the upgrade overwrites the patched file).
