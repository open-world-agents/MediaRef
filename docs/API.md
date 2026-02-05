# API Reference

## MediaRef(uri: str | DataURI, pts_ns: int | None = None)

**Properties:** `is_embedded`, `is_video`, `is_remote`, `is_relative_path`

**Methods:**
- `to_ndarray(format="rgb") -> np.ndarray` - Load as numpy array
  - Formats: `"rgb"` (default), `"bgr"`, `"rgba"`, `"bgra"`, `"gray"`
  - Returns: (H, W, 3) for RGB/BGR, (H, W, 4) for RGBA/BGRA, (H, W) for grayscale
- `to_pil_image(format="rgb") -> PIL.Image` - Load as PIL Image
  - Formats: `"rgb"` (default), `"rgba"`, `"gray"`
- `resolve_relative_path(base_path, on_unresolvable="warn") -> MediaRef` - Resolve relative paths
  - `on_unresolvable`: How to handle embedded/remote URIs: `"error"`, `"warn"` (default), or `"ignore"`
- `validate_uri() -> bool` - Check if URI exists (local files only)
- `model_dump() -> dict` - Serialize to dict
- `model_dump_json() -> str` - Serialize to JSON
- `model_validate(data) -> MediaRef` - Deserialize from dict
- `model_validate_json(json_str) -> MediaRef` - Deserialize from JSON

## DataURI (for embedding media)

**Class Methods:**
- `from_image(image: np.ndarray | PIL.Image, format="png", quality=None, input_format="rgb") -> DataURI` - Create from image
  - `format`: Output format (`"png"`, `"jpeg"`, `"bmp"`)
  - `quality`: JPEG quality (1-100), ignored for PNG/BMP
  - `input_format`: Input channel order for numpy arrays. Default: `"rgb"`. Ignored for PIL Images.
    - `"rgb"`: RGB format (3 channels)
    - `"bgr"`: BGR format (3 channels) - **REQUIRED for OpenCV arrays** (e.g., `cv2.imread()`)
    - `"rgba"`: RGBA format (4 channels)
    - `"bgra"`: BGRA format (4 channels)
  - PNG format preserves alpha channel; JPEG/BMP drop alpha
- `from_file(path: str | Path, format=None) -> DataURI` - Create from file
- `from_uri(uri: str) -> DataURI` - Parse data URI string

**Methods:**
- `to_ndarray(format="rgb") -> np.ndarray` - Convert to numpy array
  - Formats: `"rgb"` (default), `"bgr"`, `"rgba"`, `"bgra"`, `"gray"`
- `to_pil_image() -> PIL.Image` - Convert to PIL Image

**Properties:**
- `uri: str` - Full data URI string
- `is_image: bool` - True if MIME type is image/*

## Functions

- `batch_decode(refs, strategy=None, decoder="pyav") -> list[np.ndarray]` - Batch decode using optimized batch decoding API
  - `refs`: List of MediaRef objects to decode
  - `strategy`: Batch decoding strategy (PyAV only): `SEPARATE`, `SEQUENTIAL`, or `SEQUENTIAL_PER_KEYFRAME_BLOCK`. Default: `None` (auto-selects `SEQUENTIAL_PER_KEYFRAME_BLOCK`)
  - `decoder`: Decoder backend (`"pyav"` or `"torchcodec"`)
- `cleanup_cache()` - Clear video container cache (PyAV only)

## Video Decoders (requires `[video]` extra)

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

