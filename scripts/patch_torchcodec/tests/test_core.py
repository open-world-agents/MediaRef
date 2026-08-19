"""Tests for TorchCodec runtime verification."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))
core = importlib.import_module("patch_torchcodec.core")


def test_diagnose_exercises_ffmpeg_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="{'libavcodec': (62, 0, 0)}\n", stderr="")

    monkeypatch.setattr(core.subprocess, "run", run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient")

    result = core.diagnose_torchcodec(tmp_path, require_env=False)

    assert result.ok
    assert "get_ffmpeg_library_versions" in captured["command"][2]
    assert "LD_LIBRARY_PATH" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["timeout"] == 30


def test_diagnose_can_prepend_pyav_libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    def run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="versions\n", stderr="")

    monkeypatch.setattr(core.subprocess, "run", run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient")

    result = core.diagnose_torchcodec(tmp_path, require_env=True)

    assert result.ok
    assert captured["env"]["LD_LIBRARY_PATH"] == f"{tmp_path}:/ambient"


def test_diagnose_preserves_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="FFmpeg shared libraries unavailable")

    monkeypatch.setattr(core.subprocess, "run", run)

    result = core.diagnose_torchcodec(tmp_path, require_env=False)

    assert not result.ok
    assert result.returncode == 1
    assert result.details == "FFmpeg shared libraries unavailable"


def test_diagnose_reports_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def run(command, **kwargs):
        raise core.subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(core.subprocess, "run", run)

    result = core.diagnose_torchcodec(tmp_path, require_env=False)

    assert not result.ok
    assert result.returncode == 124
    assert "timed out" in result.details


def test_verify_torchcodec_keeps_boolean_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        core,
        "diagnose_torchcodec",
        lambda *args, **kwargs: core.VerificationResult(ok=True),
    )

    assert core.verify_torchcodec()


def test_cli_verify_does_not_require_pyav(monkeypatch: pytest.MonkeyPatch):
    cli = importlib.import_module("patch_torchcodec.__main__")
    monkeypatch.setattr(cli, "find_av_libs_dir", lambda: None)
    monkeypatch.setattr(
        cli,
        "diagnose_torchcodec",
        lambda *args, **kwargs: core.VerificationResult(ok=True, stdout="versions"),
    )
    monkeypatch.setattr(sys, "argv", ["patch-torchcodec", "--verify"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
