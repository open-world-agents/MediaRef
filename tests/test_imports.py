"""Tests for mediaref's import-time behavior.

These run in subprocesses because they assert on a fresh interpreter's
``sys.modules`` state — mutating the caller's module table would break any
other test that already holds bound references to ``mediaref`` symbols.
"""

import subprocess
import sys
import textwrap


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_pyav_only_import_does_not_load_torchcodec():
    """``import mediaref.video_decoder`` must not transitively import
    ``torchcodec``. PyAV-only users should never pay torchcodec's import
    cost, and a broken torchcodec install (e.g. FFmpeg ABI mismatch) must
    not break ``import mediaref``.

    See ``video_decoder/__init__.py``: ``TorchCodecVideoDecoder`` is
    exposed lazily via PEP 562 ``__getattr__``.
    """
    result = _run("""
        import sys
        import mediaref.video_decoder  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("torchcodec")]
        assert not leaked, f"torchcodec leaked: {leaked}"
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_top_level_mediaref_import_does_not_load_torchcodec():
    """``import mediaref`` must not load torchcodec either."""
    result = _run("""
        import sys
        import mediaref  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("torchcodec")]
        assert not leaked, f"torchcodec leaked: {leaked}"
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"
