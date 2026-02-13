"""Core functionality for setting up FFmpeg libraries for TorchCodec."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_av_libs_dir() -> Path | None:
    """Find PyAV's bundled FFmpeg libraries directory.

    Returns:
        Path to av.libs directory, or None if not found.
    """
    try:
        import av

        av_path = Path(av.__file__).parent
        libs_dir = av_path.parent / "av.libs"
        if libs_dir.exists():
            return libs_dir
    except ImportError:
        pass
    return None


def get_library_mappings(libs_dir: Path) -> dict[str, Path]:
    """Get mapping from standard library names to actual PyAV library files.

    Args:
        libs_dir: Path to PyAV's av.libs directory.

    Returns:
        Dict mapping standard names (e.g., 'libavcodec.so.62') to actual file paths.
    """
    required_libs = [
        "libavcodec",
        "libavformat",
        "libavutil",
        "libswscale",
        "libswresample",
        "libavdevice",
        "libavfilter",
    ]

    mappings = {}

    for lib_name in required_libs:
        # Find the actual library file (e.g., libavcodec-e57b519c.so.62.11.100)
        pattern = re.compile(rf"^{lib_name}-[a-f0-9]+\.so\.(\d+)\.\d+\.\d+$")

        for file in libs_dir.iterdir():
            match = pattern.match(file.name)
            if match:
                major_version = match.group(1)
                standard_name = f"{lib_name}.so.{major_version}"
                mappings[standard_name] = file
                break

    return mappings


def create_symlinks(libs_dir: Path, mappings: dict[str, Path] | None = None) -> list[Path]:
    """Create symbolic links for FFmpeg libraries.

    Args:
        libs_dir: Path to PyAV's av.libs directory.
        mappings: Optional pre-computed library mappings.

    Returns:
        List of created symlink paths.
    """
    if mappings is None:
        mappings = get_library_mappings(libs_dir)

    created = []

    for standard_name, actual_file in mappings.items():
        symlink_path = libs_dir / standard_name

        # Remove existing symlink if present
        if symlink_path.is_symlink():
            symlink_path.unlink()
        elif symlink_path.exists():
            continue

        # Create relative symlink
        symlink_path.symlink_to(actual_file.name)
        created.append(symlink_path)

    return created


def get_ld_library_path() -> str | None:
    """Get the LD_LIBRARY_PATH value needed for TorchCodec.

    Returns:
        Path string to add to LD_LIBRARY_PATH, or None if PyAV not found.
    """
    libs_dir = find_av_libs_dir()
    if libs_dir is None:
        return None
    return str(libs_dir)


def setup(verbose: bool = False) -> bool:
    """Setup FFmpeg libraries for TorchCodec.

    Creates symbolic links from PyAV's bundled FFmpeg libraries to standard
    library names that TorchCodec expects.

    Args:
        verbose: If True, print progress messages.

    Returns:
        True if setup was successful, False otherwise.
    """
    libs_dir = find_av_libs_dir()
    if libs_dir is None:
        if verbose:
            print("Error: PyAV not found. Install with: pip install av", file=sys.stderr)
        return False

    mappings = get_library_mappings(libs_dir)
    if not mappings:
        if verbose:
            print("Error: No FFmpeg libraries found in PyAV bundle.", file=sys.stderr)
        return False

    created = create_symlinks(libs_dir, mappings)

    if verbose:
        print(f"Created {len(created)} symbolic links in {libs_dir}")
        print("\nTo use TorchCodec, set LD_LIBRARY_PATH:")
        print(f'  export LD_LIBRARY_PATH="{libs_dir}:$LD_LIBRARY_PATH"')

    return True


def find_torchcodec_libs() -> list[Path]:
    """Find TorchCodec shared libraries.

    Uses importlib.util.find_spec to locate the package without importing it,
    avoiding the chicken-and-egg problem where import torchcodec fails because
    FFmpeg libraries aren't available yet (which is the whole reason for patching).

    Returns:
        List of paths to TorchCodec .so files.
    """
    import importlib.util

    spec = importlib.util.find_spec("torchcodec")
    if spec is None or spec.origin is None:
        return []
    torchcodec_dir = Path(spec.origin).parent
    return list(torchcodec_dir.glob("libtorchcodec_*.so"))


def find_patchelf() -> Path | None:
    """Find patchelf executable.

    Returns:
        Path to patchelf executable, or None if not found.
    """
    # Check in current Python environment first
    try:
        import site

        for sp in site.getsitepackages():
            patchelf = Path(sp).parent.parent / "bin" / "patchelf"
            if patchelf.exists():
                return patchelf
    except Exception:
        pass

    # Check system PATH
    patchelf = shutil.which("patchelf")
    if patchelf:
        return Path(patchelf)

    return None


def patch_rpath(library_path: Path, rpath: str, patchelf: Path | None = None) -> bool:
    """Set RPATH on a shared library using patchelf.

    Args:
        library_path: Path to the .so file to patch.
        rpath: RPATH value to set.
        patchelf: Path to patchelf executable. Auto-detected if None.

    Returns:
        True if successful, False otherwise.
    """
    if patchelf is None:
        patchelf = find_patchelf()
    if patchelf is None:
        return False

    try:
        subprocess.run(
            [str(patchelf), "--force-rpath", "--set-rpath", rpath, str(library_path)],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, Exception):
        return False


def _find_torch_lib_dir() -> Path | None:
    """Find PyTorch's lib directory containing libtorch.so, libc10.so, etc."""
    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    torch_lib = Path(spec.origin).parent / "lib"
    if torch_lib.is_dir():
        return torch_lib.resolve()
    return None


def _find_nvidia_lib_dirs() -> list[Path]:
    """Find NVIDIA library directories (nvidia/*/lib) in site-packages.

    These contain libnppicc.so, libnvrtc.so, libcudart.so, etc.
    """
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return []
    # nvidia is a namespace package; use submodule_search_locations
    search_paths = spec.submodule_search_locations
    if not search_paths:
        return []
    nvidia_dir = Path(list(search_paths)[0])
    lib_dirs = []
    for sub in sorted(nvidia_dir.iterdir()):
        lib_path = sub / "lib"
        if lib_path.is_dir():
            lib_dirs.append(lib_path.resolve())
    return lib_dirs


def setup_with_patchelf(verbose: bool = False) -> bool:
    """Setup TorchCodec to use PyAV's FFmpeg libraries via RPATH patching.

    This method patches TorchCodec's shared libraries to include PyAV's av.libs
    directory in their RPATH, eliminating the need for LD_LIBRARY_PATH.

    Requires: patchelf (install with: pip install patchelf)

    Args:
        verbose: If True, print progress messages.

    Returns:
        True if setup was successful, False otherwise.
    """
    # Find dependencies
    libs_dir = find_av_libs_dir()
    if libs_dir is None:
        if verbose:
            print("Error: PyAV not found. Install with: pip install av", file=sys.stderr)
        return False

    torchcodec_libs = find_torchcodec_libs()
    if not torchcodec_libs:
        if verbose:
            print("Error: TorchCodec not found. Install with: pip install torchcodec", file=sys.stderr)
        return False

    patchelf = find_patchelf()
    if patchelf is None:
        if verbose:
            print("Error: patchelf not found. Install with: pip install patchelf", file=sys.stderr)
        return False

    # Create symlinks for all libraries in av.libs
    if verbose:
        print(f"Creating symbolic links in {libs_dir}...")

    created_symlinks = create_all_symlinks(libs_dir)
    if verbose:
        print(f"  Created {len(created_symlinks)} symbolic links")

    # Build RPATH with all required library directories
    rpath_dirs = [str(libs_dir.resolve())]

    # Add torch/lib for libtorch.so, libc10.so, etc.
    torch_lib_dir = _find_torch_lib_dir()
    if torch_lib_dir is not None:
        rpath_dirs.append(str(torch_lib_dir))
        if verbose:
            print(f"  torch lib: {torch_lib_dir}")

    # Add nvidia/*/lib for libnppicc.so, libnvrtc.so, etc.
    nvidia_lib_dirs = _find_nvidia_lib_dirs()
    for d in nvidia_lib_dirs:
        rpath_dirs.append(str(d))
    if verbose and nvidia_lib_dirs:
        print(f"  nvidia lib dirs: {len(nvidia_lib_dirs)}")

    rpath = ":".join(rpath_dirs)

    # Patch TorchCodec libraries
    if verbose:
        print(f"\nPatching TorchCodec libraries with RPATH ({len(rpath_dirs)} dirs)")
    patched = 0
    for lib in torchcodec_libs:
        if patch_rpath(lib, rpath, patchelf):
            patched += 1
            if verbose:
                print(f"  Patched: {lib.name}")
        elif verbose:
            print(f"  Failed: {lib.name}", file=sys.stderr)

    if verbose:
        print(f"\nPatched {patched}/{len(torchcodec_libs)} libraries")

    return patched == len(torchcodec_libs)


def create_all_symlinks(libs_dir: Path) -> list[Path]:
    """Create symbolic links for ALL libraries in av.libs.

    This creates symlinks from hash-included names to standard names,
    e.g., libdrm-b0291a67.so.2.4.0 -> libdrm.so.2

    Args:
        libs_dir: Path to PyAV's av.libs directory.

    Returns:
        List of created symlink paths.
    """
    created = []

    # Pattern: libname-hash.so.major.minor.patch
    pattern = re.compile(r"^(.+)-[a-f0-9]+\.so\.(\d+)(?:\.\d+)*$")

    for file in libs_dir.iterdir():
        if file.is_symlink():
            continue

        match = pattern.match(file.name)
        if match:
            lib_name = match.group(1)
            major_version = match.group(2)
            standard_name = f"{lib_name}.so.{major_version}"
            symlink_path = libs_dir / standard_name

            if not symlink_path.exists():
                symlink_path.symlink_to(file.name)
                created.append(symlink_path)

    return created


def verify_torchcodec(libs_dir: Path | None = None, require_env: bool = True) -> bool:
    """Verify TorchCodec can load with the configured libraries.

    Args:
        libs_dir: Path to av.libs directory. If None, will be auto-detected.
        require_env: If True, set LD_LIBRARY_PATH for verification.
                     If False, test without setting LD_LIBRARY_PATH (for RPATH-patched installs).

    Returns:
        True if TorchCodec loads successfully, False otherwise.
    """
    if libs_dir is None:
        libs_dir = find_av_libs_dir()

    env = os.environ.copy()
    if require_env and libs_dir is not None:
        env["LD_LIBRARY_PATH"] = f"{libs_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", "from torchcodec.decoders import VideoDecoder; print('OK')"],
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


def is_rpath_patched() -> bool:
    """Check if TorchCodec libraries have been RPATH-patched.

    Returns:
        True if RPATH is set to av.libs directory.
    """
    libs_dir = find_av_libs_dir()
    if libs_dir is None:
        return False

    torchcodec_libs = find_torchcodec_libs()
    if not torchcodec_libs:
        return False

    # Check RPATH of one library
    try:
        result = subprocess.run(
            ["readelf", "-d", str(torchcodec_libs[0])],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return str(libs_dir) in result.stdout
    except Exception:
        pass

    return False
