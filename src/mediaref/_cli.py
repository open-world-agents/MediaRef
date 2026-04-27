"""CLI for permanently registering MediaRef as a HuggingFace ``datasets`` feature.

Without this CLI, every consumer process must explicitly
``from mediaref.hf import MediaRefFeature`` before calling
``load_from_disk`` / ``load_dataset`` — otherwise datasets' global feature
registry doesn't have ``"MediaRef"`` and the load raises
``ValueError: Feature type 'MediaRef' not found``.

`mediaref enable-hf-feature` patches ``datasets/features/features.py`` on
disk to import ``mediaref.hf`` during datasets' own initialization. The
feature is then registered in every Python process in this environment
without an explicit import. Same approach as ``patch_torchcodec`` in this
repo (which patches torchcodec's RPATH on disk).

Caveats:
    - The patch is in the installed datasets package. ``pip upgrade
      datasets`` overwrites the file, removing the patch. Re-run
      ``mediaref enable-hf-feature`` after upgrades.
    - Requires write permission to the env's site-packages. In system
      Python or PEP 668 environments, use a virtualenv.
    - Reverse with ``mediaref disable-hf-feature``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional

_BEGIN = "# === BEGIN: mediaref auto-registration ==="
_END = "# === END: mediaref auto-registration ==="

_PATCH_BLOCK = f"""
{_BEGIN}
# Added by `mediaref enable-hf-feature`. Idempotent. Re-apply after
# `pip upgrade datasets`. Reverse with `mediaref disable-hf-feature`.
try:
    from mediaref.hf import MediaRefFeature as _mediaref_MediaRefFeature

    _FEATURE_TYPES["MediaRef"] = _mediaref_MediaRefFeature
except ImportError:
    pass
{_END}
"""


# ---------------------------------------------------------------------------
# Pure-text patch operations (testable without mutating any real files)
# ---------------------------------------------------------------------------


def is_enabled(text: str) -> bool:
    """True if ``text`` already contains the mediaref patch sentinel."""
    return _BEGIN in text


def apply_patch(text: str) -> str:
    """Append the mediaref patch block to ``text`` if not already present.

    Idempotent: applying twice returns the same text as applying once.
    """
    if is_enabled(text):
        return text
    if text and not text.endswith("\n"):
        text += "\n"
    return text + _PATCH_BLOCK


def remove_patch(text: str) -> str:
    """Remove the mediaref patch block from ``text``. No-op if absent."""
    if not is_enabled(text):
        return text
    start = text.index(_BEGIN)
    end = text.index(_END, start) + len(_END)
    before = text[:start].rstrip("\n")
    after = text[end:].lstrip("\n")
    if before and after:
        return before + "\n" + after
    return (before or after).rstrip() + "\n" if (before or after) else ""


# ---------------------------------------------------------------------------
# File-system entry points (used by the CLI subcommands)
# ---------------------------------------------------------------------------


def _datasets_features_path() -> Path:
    """Locate ``datasets/features/features.py`` on disk."""
    try:
        feat_mod = importlib.import_module("datasets.features.features")
    except ImportError as e:
        raise SystemExit("`datasets` is not installed. Run `pip install 'mediaref[hf]'` first.") from e
    src = getattr(feat_mod, "__file__", None)
    if not src:
        raise SystemExit("Could not determine the path of datasets.features.features.")
    path = Path(src)
    if not path.is_file():
        raise SystemExit(f"datasets.features.features not found at: {path}")
    return path


def cmd_enable() -> int:
    path = _datasets_features_path()
    text = path.read_text(encoding="utf-8")
    if is_enabled(text):
        print(f"Already enabled: {path}")
        return 0
    new_text = apply_patch(text)
    try:
        path.write_text(new_text, encoding="utf-8")
    except PermissionError as e:
        print(
            f"error: cannot write to {path}: {e}\n"
            "Permission denied. If this is a system Python environment, install in a virtualenv first.",
            file=sys.stderr,
        )
        return 1
    print(f"Enabled: {path}")
    print("MediaRef is now registered with HuggingFace datasets in this environment.")
    print("Re-run after `pip upgrade datasets`.")
    return 0


def cmd_disable() -> int:
    path = _datasets_features_path()
    text = path.read_text(encoding="utf-8")
    if not is_enabled(text):
        print(f"Not enabled: {path}")
        return 0
    new_text = remove_patch(text)
    try:
        path.write_text(new_text, encoding="utf-8")
    except PermissionError as e:
        print(f"error: cannot write to {path}: {e}", file=sys.stderr)
        return 1
    print(f"Disabled: {path}")
    return 0


def cmd_status() -> int:
    path = _datasets_features_path()
    text = path.read_text(encoding="utf-8")
    if is_enabled(text):
        print(f"enabled\n  features.py: {path}")
    else:
        print(f"disabled\n  features.py: {path}\n  Run `mediaref enable-hf-feature` to register MediaRef permanently.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mediaref",
        description="MediaRef command-line utilities.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "enable-hf-feature",
        help="Patch datasets/features/features.py to auto-register MediaRef on every datasets import.",
    )
    sub.add_parser(
        "disable-hf-feature",
        help="Remove the mediaref patch from datasets/features/features.py.",
    )
    sub.add_parser(
        "status",
        help="Show whether MediaRef is permanently registered with datasets in this environment.",
    )
    args = parser.parse_args(argv)
    dispatch = {
        "enable-hf-feature": cmd_enable,
        "disable-hf-feature": cmd_disable,
        "status": cmd_status,
    }
    return dispatch[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
