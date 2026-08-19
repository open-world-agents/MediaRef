# patch-torchcodec

Diagnose TorchCodec's FFmpeg video runtime and, when needed on Linux, patch TorchCodec to use PyAV's bundled libraries.

## Problem

TorchCodec video decoding requires FFmpeg shared libraries. Since TorchCodec 0.16, `import torchcodec` can succeed without FFmpeg, so import-only checks can report a false success.

## Solution

Use TorchCodec's official shared-FFmpeg installation instructions first. If a Linux environment already has PyAV, `patch-torchcodec` can instead patch TorchCodec's RPATH to find PyAV's bundle.

## Quick Start

```bash
pip install torchcodec
pip install patch-torchcodec    # installs the optional PyAV/patchelf fallback
patch-torchcodec --verify       # actual FFmpeg runtime probe
patch-torchcodec                # run only if the probe fails and PyAV reuse is desired
```

Verification calls `get_ffmpeg_library_versions()` in a fresh process. This forces the FFmpeg-backed runtime to load and prints its original diagnostic on failure.

```python
from torchcodec._core import get_ffmpeg_library_versions
print(get_ffmpeg_library_versions())
```

## Command Line Options

```bash
patch-torchcodec               # Patch RPATH (default, recommended)
patch-torchcodec --env-only    # Symlinks only (requires LD_LIBRARY_PATH)
patch-torchcodec --status      # Check current setup status
patch-torchcodec --verify      # Verify an FFmpeg-backed operation works
patch-torchcodec --quiet       # Silent mode
```

## Python API

```python
from patch_torchcodec import setup_with_patchelf, verify_torchcodec, is_rpath_patched

# Patch RPATH (recommended) — persists across sessions
setup_with_patchelf(verbose=True)

# Verify
assert verify_torchcodec(require_env=False)
```

## How It Works

1. **Finds PyAV's bundled FFmpeg** in `site-packages/av.libs/`
2. **Creates symbolic links** with standard names (e.g., `libavcodec.so.62`)
3. **Patches TorchCodec's `.so` files** with `patchelf` to include PyAV's library path in RPATH

## Limitations

- **Linux only** for patching (verification itself is portable)
- **Same virtualenv**: PyAV and TorchCodec must be in the same environment
- **Re-run after reinstall**: If you reinstall TorchCodec, run `patch-torchcodec` again
- **Use only when needed**: a working system/conda FFmpeg installation should not be patched

## License

MIT
