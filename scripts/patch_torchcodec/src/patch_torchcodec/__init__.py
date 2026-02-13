"""Patch TorchCodec to use PyAV's bundled FFmpeg — no system FFmpeg needed.

Usage:
    pip install patch-torchcodec   # installs av + patchelf
    patch-torchcodec               # patches RPATH — done!

Python API:
    from patch_torchcodec import setup_with_patchelf
    setup_with_patchelf()  # One-time setup, persists across sessions

Alternative (no binary patching):
    patch-torchcodec --env-only
    export LD_LIBRARY_PATH="<shown path>:$LD_LIBRARY_PATH"
"""

from __future__ import annotations

__version__ = "0.1.4"

from .core import (
    create_all_symlinks,
    create_symlinks,
    find_av_libs_dir,
    find_patchelf,
    find_torchcodec_libs,
    get_ld_library_path,
    get_library_mappings,
    is_rpath_patched,
    patch_rpath,
    setup,
    setup_with_patchelf,
    verify_torchcodec,
)

__all__ = [
    "create_all_symlinks",
    "create_symlinks",
    "find_av_libs_dir",
    "find_patchelf",
    "find_torchcodec_libs",
    "get_ld_library_path",
    "get_library_mappings",
    "is_rpath_patched",
    "patch_rpath",
    "setup",
    "setup_with_patchelf",
    "verify_torchcodec",
]

