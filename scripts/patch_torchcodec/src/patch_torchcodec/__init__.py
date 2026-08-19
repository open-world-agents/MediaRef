"""Diagnose TorchCodec video support and optionally reuse PyAV's FFmpeg.

Usage:
    pip install patch-torchcodec   # portable runtime verifier
    patch-torchcodec --verify      # checks an FFmpeg-backed operation
    patch-torchcodec               # patches RPATH when recovery is needed

Python API:
    from patch_torchcodec import setup_with_patchelf
    setup_with_patchelf()  # One-time setup, persists across sessions

Alternative (no binary patching):
    patch-torchcodec --env-only
    export LD_LIBRARY_PATH="<shown path>:$LD_LIBRARY_PATH"

Install ``patch-torchcodec[patch]`` before using patch operations.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .core import (
    VerificationResult,
    create_all_symlinks,
    create_symlinks,
    diagnose_torchcodec,
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
    "diagnose_torchcodec",
    "find_av_libs_dir",
    "find_patchelf",
    "find_torchcodec_libs",
    "get_ld_library_path",
    "get_library_mappings",
    "is_rpath_patched",
    "patch_rpath",
    "setup",
    "setup_with_patchelf",
    "VerificationResult",
    "verify_torchcodec",
]
