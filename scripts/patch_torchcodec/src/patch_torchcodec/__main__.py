#!/usr/bin/env python3
"""Command-line interface for patch-torchcodec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    create_all_symlinks,
    create_symlinks,
    find_av_libs_dir,
    find_patchelf,
    find_torchcodec_libs,
    get_library_mappings,
    is_rpath_patched,
    setup_with_patchelf,
    verify_torchcodec,
)


def main():
    parser = argparse.ArgumentParser(
        prog="patch-torchcodec",
        description="Patch TorchCodec to use PyAV's bundled FFmpeg libraries via RPATH.",
        epilog="""
Examples:
  patch-torchcodec               # Patch RPATH (recommended, one-time)
  patch-torchcodec --env-only    # Symlinks only (requires LD_LIBRARY_PATH)
  patch-torchcodec --verify      # Verify TorchCodec works
  patch-torchcodec --status      # Check current setup status
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Only create symlinks, don't patch RPATH (requires LD_LIBRARY_PATH)",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print shell command to set LD_LIBRARY_PATH",
    )
    parser.add_argument(
        "--create-activate",
        action="store_true",
        help="Create an activation script (activate_torchcodec.sh)",
    )
    parser.add_argument(
        "--verify", "--verify-only",
        action="store_true",
        dest="verify_only",
        help="Only verify TorchCodec works, don't make changes",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current setup status",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    # Find PyAV libraries
    libs_dir = find_av_libs_dir()
    if libs_dir is None:
        print("Error: Could not find PyAV's av.libs directory.", file=sys.stderr)
        print("Make sure PyAV is installed: pip install av", file=sys.stderr)
        sys.exit(1)

    # Status mode
    if args.status:
        print("patch-torchcodec: Status")
        print("=" * 40)
        print(f"PyAV av.libs: {libs_dir}")

        torchcodec_libs = find_torchcodec_libs()
        if torchcodec_libs:
            print(f"TorchCodec libraries: {len(torchcodec_libs)} found")
        else:
            print("TorchCodec: NOT INSTALLED")

        patchelf = find_patchelf()
        print(f"patchelf: {'found at ' + str(patchelf) if patchelf else 'NOT FOUND'}")

        rpath_patched = is_rpath_patched()
        print(f"RPATH patched: {'YES' if rpath_patched else 'NO'}")

        print()
        if verify_torchcodec(libs_dir, require_env=False):
            print("✓ TorchCodec works WITHOUT LD_LIBRARY_PATH")
        elif verify_torchcodec(libs_dir, require_env=True):
            print("⚠ TorchCodec works only WITH LD_LIBRARY_PATH")
        else:
            print("✗ TorchCodec is NOT working")
        sys.exit(0)

    # Verify only mode
    if args.verify_only:
        if verbose:
            print("Verifying TorchCodec...")

        # Try without LD_LIBRARY_PATH first
        if verify_torchcodec(libs_dir, require_env=False):
            if verbose:
                print("✓ TorchCodec works without LD_LIBRARY_PATH (RPATH patched)")
            sys.exit(0)
        elif verify_torchcodec(libs_dir, require_env=True):
            if verbose:
                print("✓ TorchCodec works with LD_LIBRARY_PATH")
                print(f"  Run: export LD_LIBRARY_PATH=\"{libs_dir}:$LD_LIBRARY_PATH\"")
            sys.exit(0)
        else:
            if verbose:
                print("✗ TorchCodec verification failed.")
            sys.exit(1)

    if verbose:
        print("patch-torchcodec: Patching")
        print("=" * 40)
        print(f"PyAV av.libs: {libs_dir}")

    # Decide setup method
    if args.env_only:
        # LD_LIBRARY_PATH only mode
        if verbose:
            print("\nSetup mode: LD_LIBRARY_PATH (symlinks only)")
            print("\nCreating symbolic links...")

        mappings = get_library_mappings(libs_dir)
        if not mappings:
            print("Error: No FFmpeg libraries found in PyAV bundle.", file=sys.stderr)
            sys.exit(1)

        created = create_symlinks(libs_dir, mappings)
        # Also create symlinks for all other libraries (libdrm, etc.)
        all_created = create_all_symlinks(libs_dir)

        if verbose:
            print(f"  Created {len(created) + len(all_created)} symbolic links")
            print(f"\n✓ Symlinks created")
            print(f"\nTo use TorchCodec, set LD_LIBRARY_PATH:")
            print(f'  export LD_LIBRARY_PATH="{libs_dir}:$LD_LIBRARY_PATH"')

        if args.print_env:
            print(f'\nexport LD_LIBRARY_PATH="{libs_dir}:$LD_LIBRARY_PATH"')
    else:
        # RPATH patching mode (default)
        if verbose:
            print("\nSetup mode: RPATH patching (recommended)")

        patchelf = find_patchelf()
        if patchelf is None:
            print("\nError: patchelf not found.", file=sys.stderr)
            print("Install with: pip install patchelf", file=sys.stderr)
            print("\nAlternatively, use --env-only to setup with LD_LIBRARY_PATH.", file=sys.stderr)
            sys.exit(1)

        torchcodec_libs = find_torchcodec_libs()
        if not torchcodec_libs:
            print("\nError: TorchCodec not found.", file=sys.stderr)
            print("Install with: pip install torchcodec", file=sys.stderr)
            sys.exit(1)

        if verbose:
            print(f"  patchelf: {patchelf}")
            print(f"  TorchCodec libraries: {len(torchcodec_libs)}")

        success = setup_with_patchelf(verbose=verbose)

        if not success:
            print("\nError: RPATH patching failed.", file=sys.stderr)
            sys.exit(1)

        if verbose:
            print("\n✓ RPATH patching complete!")
            print("  TorchCodec will now automatically find FFmpeg libraries.")
            print("  No LD_LIBRARY_PATH needed!")

    # Create activation script if requested
    if args.create_activate:
        script_path = Path("activate_torchcodec.sh")
        script_content = f"""#!/bin/bash
# Activation script for TorchCodec FFmpeg libraries
# Source this file: source {script_path}

export LD_LIBRARY_PATH="{libs_dir}:$LD_LIBRARY_PATH"
echo "TorchCodec FFmpeg libraries activated"
"""
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        if verbose:
            print(f"\n✓ Created activation script: {script_path}")
            print(f"  Usage: source {script_path}")

    # Verify
    if verbose:
        print("\nVerifying TorchCodec...")

    require_env = args.env_only
    if verify_torchcodec(libs_dir, require_env=require_env):
        if verbose:
            if require_env:
                print("✓ TorchCodec works (with LD_LIBRARY_PATH)")
            else:
                print("✓ TorchCodec works (without LD_LIBRARY_PATH)")
    else:
        if verbose:
            print("⚠ TorchCodec verification failed.")
            if require_env:
                print("  Make sure to set LD_LIBRARY_PATH before running your code.")
        sys.exit(1)


if __name__ == "__main__":
    main()

