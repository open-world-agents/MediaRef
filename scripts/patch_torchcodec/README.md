# patch-torchcodec

Patch TorchCodec to use PyAV's bundled FFmpeg — **one command, no system FFmpeg needed**.

## Problem

TorchCodec requires FFmpeg shared libraries (`libavcodec.so.62`, etc.), but installing FFmpeg system-wide can be complex and may cause version conflicts.

## Solution

PyAV already bundles FFmpeg. `patch-torchcodec` patches TorchCodec's RPATH to find them — no environment variables needed!

## Quick Start

```bash
pip install torchcodec          # install torchcodec (with PyTorch)
pip install patch-torchcodec    # install patcher (av & patchelf included)
patch-torchcodec                # patches RPATH — done!
```

That's it. TorchCodec now works:

```python
from torchcodec.decoders import VideoDecoder  # ✓ just works
```

## Command Line Options

```bash
patch-torchcodec               # Patch RPATH (default, recommended)
patch-torchcodec --env-only    # Symlinks only (requires LD_LIBRARY_PATH)
patch-torchcodec --status      # Check current setup status
patch-torchcodec --verify      # Verify TorchCodec works
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

- **Linux only** (RPATH is Linux-specific)
- **Same virtualenv**: PyAV and TorchCodec must be in the same environment
- **Re-run after reinstall**: If you reinstall TorchCodec, run `patch-torchcodec` again

## License

MIT

