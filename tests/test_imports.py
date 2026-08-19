"""Tests for mediaref's import-time behavior.

These run in subprocesses because they assert on a fresh interpreter's
``sys.modules`` state — mutating the caller's module table would break any
other test that already holds bound references to ``mediaref`` symbols.
"""

import subprocess
import sys
import textwrap

import pytest

from tests import _torchcodec_video_available


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


@pytest.mark.video
def test_pyav_only_import_does_not_load_torchcodec():
    """`import mediaref.video_decoder` must not transitively load torchcodec.

    Marked ``video`` because the import requires PyAV (`require_video()`).
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
    """`import mediaref` must not load torchcodec either. Runs unconditionally
    — the top-level package has no PyAV dependency."""
    result = _run("""
        import sys
        import mediaref  # noqa: F401
        leaked = [m for m in sys.modules if m.startswith("torchcodec")]
        assert not leaked, f"torchcodec leaked: {leaked}"
        print("OK")
    """)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


def test_torchcodec_video_availability_calls_ffmpeg_operation():
    calls = 0

    def get_ffmpeg_library_versions():
        nonlocal calls
        calls += 1

    assert _torchcodec_video_available(get_ffmpeg_library_versions)
    assert calls == 1


def test_torchcodec_video_availability_rejects_missing_ffmpeg():
    def get_ffmpeg_library_versions():
        raise RuntimeError("FFmpeg is unavailable")

    assert not _torchcodec_video_available(get_ffmpeg_library_versions)
