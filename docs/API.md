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
from mediaref import MediaRef, DataURI

ref = MediaRef(uri="image.png")
ref = MediaRef(uri="https://example.com/img.jpg")
ref = MediaRef(uri="s3://bucket/clip.mp4", pts_ns=1_500_000_000)
ref = MediaRef(uri="data:image/png;base64,iVBORw0…")

# A DataURI instance is accepted directly — its `uri` string is extracted.
ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))
```

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `is_embedded` | `bool` | True if the URI is a `data:` URI carrying embedded bytes. |
| `is_video` | `bool` | True if `pts_ns is not None`. |
| `is_remote` | `bool` | True if the URI is `http://` or `https://`. |
| `is_cloud_uri` | `bool` | True if the URI is delegated to fsspec (any scheme other than `file:` / `data:` / bare path — includes `http(s)://`, `s3://`, `gs://`, `hf://`, …). Overlaps with `is_remote` on http(s). |
| `is_relative_path` | `bool` | True if the URI is a relative path (not absolute, not a URI scheme). Uses platform-specific path semantics — behavior differs on Windows vs POSIX. |

### Methods

`to_ndarray(format="rgb", *, decoder="pyav", decoder_options=None, image_decoder="pillow", image_decoder_options=None, storage_options=None) -> np.ndarray`
- Loads the media as a numpy array in the requested format.
- Formats: `"rgb"` (default), `"bgr"`, `"rgba"`, `"bgra"`, `"gray"`.
- Returns shape: `(H, W, 3)` for RGB/BGR, `(H, W, 4)` for RGBA/BGRA, `(H, W)` for grayscale.
- For video URIs (`pts_ns is not None`), decodes the single frame at that timestamp.
- `decoder` and `decoder_options` select and configure the video backend.
- `image_decoder` selects Pillow (default) or TorchCodec 0.16+; `image_decoder_options` configures TorchCodec's `decode_image` call.
- `storage_options` is passed to fsspec for either media type.

`to_pil_image(format="rgb", *, decoder="pyav", decoder_options=None, image_decoder="pillow", image_decoder_options=None, storage_options=None) -> PIL.Image`
- Same as `to_ndarray` but returns a PIL Image. Formats: `"rgb"`, `"rgba"`, `"gray"`.
- PIL output requires `uint8`; use `to_ndarray` when preserving TorchCodec `uint16` image or `float32` HDR video output.

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

`validate_uri(*, storage_options=None) -> bool` — checks whether local, embedded, or fsspec-routed media exists.

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
batch_decode(refs, decoder="pyav", *, decoder_options=None, image_decoder="pillow", image_decoder_options=None, storage_options=None, ...) -> list[np.ndarray]
```

Decode many `MediaRef` video frames efficiently by grouping refs that share a URI, opening each container once, and seeking through the requested timestamps in order. Significantly faster than per-ref decoding when refs cluster on the same video file.

PyAV requests are split at `gap_threshold` so it can seek over large gaps. TorchCodec receives one sorted sparse request per URI, allowing TorchCodec 0.15+ to apply its native skip/seek optimization. `allow_gap=False` validates the same threshold for both backends.

To compare native and manually chunked requests on your media and hardware, run `python benchmarks/benchmark_sparse_batching.py VIDEO --backend torchcodec --timestamps 0 10 30 60`.

```python
from mediaref import MediaRef, batch_decode

refs = [MediaRef(uri="episode.mp4", pts_ns=int(i * 1e9)) for i in range(10)]
frames = batch_decode(refs)  # default: PyAV (CPU)
frames = batch_decode(refs, decoder="torchcodec")  # TorchCodec on CPU by default
frames = batch_decode(
    refs,
    decoder="torchcodec",
    decoder_options={"device": "cuda", "seek_mode": "approximate"},
)

images = batch_decode(
    [MediaRef(uri="s3://bucket/frame.png")],
    allow_images=True,
    image_decoder="torchcodec",
    image_decoder_options={"output_dtype": "auto"},
    storage_options={"anon": False},
)
```

### Decoder backends

| | `"pyav"` (default) | `"torchcodec"` |
| --- | --- | --- |
| Backend | PyAV (FFmpeg) | TorchCodec (FFmpeg) |
| Acceleration | CPU only | CPU by default; CUDA with `decoder_options={"device": "cuda"}` |
| Install | `pip install 'mediaref[video]'` | `pip install 'mediaref[torchcodec]'` (PyAV not required) |
| Sources | local paths, external file-likes, and any fsspec URI | local paths, bytes, external file-likes, and any fsspec URI |

Both backends share unified [playback semantics](playback_semantics.md), so a given `pts_ns` (when supported by both) returns the same frame regardless of decoder.

`decoder_options` is passed to the selected decoder constructor. TorchCodec options include `device`, `seek_mode`, `num_ffmpeg_threads`, `dimension_order`, `stream_index`, `transforms`, and `output_dtype`. MediaRef always normalizes the returned frame batch to NCHW internally and the final `batch_decode` result to RGB HWC NumPy arrays, including when TorchCodec decodes on CUDA or uses `dimension_order="NHWC"`. TorchCodec 0.14+ accepts `output_dtype="auto"`, returning `uint8` for SDR and `float32` in `[0, 1]` for detected HDR content; MediaRef preserves that dtype and uses `1.0` for an added opaque alpha channel.

```python
frames = batch_decode(
    refs,
    decoder="torchcodec",
    decoder_options={
        "output_dtype": "auto",
        "transforms": [resize_transform],
    },
)
```

### Image decoders

Pillow remains the default for compatibility. On Python 3.10+, install `mediaref[torchcodec-image]` to guarantee TorchCodec 0.16+, which can decode JPEG, PNG, WebP, GIF, AVIF, and HEIC without loading FFmpeg:

```python
image = MediaRef(uri="s3://bucket/high-bit-depth.png").to_ndarray(
    image_decoder="torchcodec",
    image_decoder_options={"output_dtype": "auto"},
    storage_options={"anon": False},
)
```

MediaRef requests `RGB_ALPHA` internally so its existing `format=` conversion remains authoritative. Pass `output_dtype="auto"` to preserve 16-bit PNG/AVIF/HEIC data as `uint16`; the default is `uint8`. Animated or multi-image inputs use their first frame, matching MediaRef's single-image reference model. HEIC additionally requires `libheif`, as documented by TorchCodec. Local paths are passed directly; data URIs and fsspec sources are materialized as encoded bytes because TorchCodec's image entry point accepts paths, bytes, or tensors rather than file-like objects.

`storage_options` is passed unchanged to fsspec and applies to every image or video URI in the call. Both decoder caches include these options in an opaque hash, so calls using different credentials or backend settings never share an open resource and secrets are not embedded in cache keys.

**TorchCodec install note.** TorchCodec 0.16 can import and decode images without FFmpeg, but `VideoDecoder` still requires compatible FFmpeg shared libraries. Test the video runtime explicitly:

```bash
python -c 'from torchcodec._core import get_ffmpeg_library_versions; print(get_ffmpeg_library_versions())'
```

Install shared FFmpeg using TorchCodec's official instructions when that probe fails. [`patch-torchcodec`](../scripts/patch_torchcodec/) is an optional Linux fallback for environments that already have PyAV and want TorchCodec to reuse PyAV's bundled libraries. Its `--verify` command performs the same FFmpeg-backed probe; a plain `import torchcodec` is intentionally not considered sufficient. PyAV-only callers remain unaffected because decoder backends are loaded independently.

`cleanup_cache()` — clears loaded PyAV and TorchCodec caches. Call between long-running decode sessions if you want to release decoder memory before automatic eviction.

### Direct decoder use

For finer control, use the decoder classes directly:

```python
import numpy as np
from mediaref.video_decoder import PyAVVideoDecoder, TorchCodecVideoDecoder

with PyAVVideoDecoder("episode.mp4") as dec:
    batch = dec.get_frames_played_at([1.5])
    frame = np.transpose(batch.data[0], (1, 2, 0))

with TorchCodecVideoDecoder("s3://bucket/episode.mp4", device="cuda", storage_options={"anon": True}) as dec:
    batch = dec.get_frames_played_at([1.5])  # returned as host NumPy arrays
```

---

## Cloud storage URIs

Any URI whose scheme is not `file://` or `data:` is delegated to [fsspec](https://filesystem-spec.readthedocs.io). `s3://`, `gs://`, `hf://`, `az://`, `webdav://`, `gdrive://`, `ipfs://`, `http(s)://`, and any future fsspec backend all work without scheme-specific code:

```python
from mediaref import MediaRef, batch_decode

ref = MediaRef(uri="s3://my-bucket/episode.mp4", pts_ns=1_500_000_000)
frame = ref.to_ndarray(
    decoder="torchcodec",
    decoder_options={"seek_mode": "approximate"},
    storage_options={"anon": True},
)

refs = [MediaRef(uri="hf://datasets/me/clips/cam.mp4", pts_ns=int(i*1e9)) for i in range(10)]
frames = batch_decode(refs, storage_options={"token": "hf_..."})

exists = ref.validate_uri(storage_options={"anon": True})
```

`fsspec` is a core dependency. Each cloud backend (`s3fs` for `s3://`, `gcsfs` for `gs://`, `huggingface_hub` for `hf://`, `adlfs` for `az://`/`abfs://`, …) must be installed separately for the schemes it serves; fsspec raises a clear error otherwise.

**Credentials and per-backend configuration.** Pass the same keyword arguments accepted by the relevant fsspec backend through `storage_options` on `validate_uri`, `to_ndarray`, `to_pil_image`, or `batch_decode`. Environment variables, per-user config files, and `fsspec.config` remain valid alternatives.

- environment variables — e.g. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for `s3fs`, `HF_TOKEN` for `huggingface_hub`, `GOOGLE_APPLICATION_CREDENTIALS` for `gcsfs`;
- per-user config files — `~/.aws/credentials`, `~/.config/gcloud/...`, `~/.cache/huggingface/token`;
- programmatic registration — `fsspec.config.set(...)` or backend-specific kwargs registered globally before calling `to_ndarray` / `batch_decode`.

For the common public-bucket case (e.g. the D2E HuggingFace datasets) no configuration is required.

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

`push_to_hub` / `load_dataset` round-trip identically.

### `MediaRefFeature`

```python
@dataclass
class MediaRefFeature:
    decode: bool = True
```

| Field / method | Description |
| --- | --- |
| `decode` (default `True`) | When `True`, accessing a column returns a `MediaRef` instance. When `False`, returns the raw `{"uri": ..., "pts_ns": ...}` dict — useful for deferring object construction or feeding values straight into PyArrow compute. |
| `pa_type` | Arrow storage type: `struct<uri: string, pts_ns: int64>`. |
| `encode_example(value)` | Accepts a `MediaRef` or a dict; emits the canonical struct dict for storage. |
| `decode_example(value)` | The inverse — used by `datasets` on row access. Honors `self.decode`. |

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

`to_videoframe` raises `ValueError` if `ref.pts_ns is None` (still-image MediaRefs can't be expressed as a `VideoFrame`, which always carries a timestamp).

---

## CLI

The `mediaref` CLI handles environment-level concerns:

| Command | Effect |
| --- | --- |
| `mediaref enable-hf-feature` | Source-patch the installed `datasets` package to auto-register `MediaRefFeature` on every `import datasets`. Idempotent. |
| `mediaref disable-hf-feature` | Reverse the patch. |
| `mediaref status` | Show whether the patch is currently applied. |

Caveats:

- Re-run `enable-hf-feature` after `pip upgrade datasets` — the upgrade overwrites the patched file.
- The patch writes to the `datasets` package inside site-packages. In a system Python or [PEP 668](https://peps.python.org/pep-0668/) environment, the write will fail with `PermissionError`; install MediaRef in a virtualenv or user-writable environment first.
