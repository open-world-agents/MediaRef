# Tests for mediaref package

from __future__ import annotations

import importlib.util
from typing import Callable

# Shared constants for TorchCodec availability — used by skipif markers across test files.
#
# TORCHCODEC_AVAILABLE: torchcodec's FFmpeg-backed video operations load
# successfully. Since TorchCodec 0.16, importing VideoDecoder alone is not a
# sufficient smoke test: image-only installs intentionally import without
# FFmpeg.
# TORCHCODEC_INSTALLED: torchcodec is present in the environment, regardless of
#   whether its native libraries load. Used to gate "not installed" tests so they
#   don't run on machines where torchcodec is installed but its FFmpeg ABI is
#   incompatible (in which case the real failure mode is a RuntimeError, not the
#   ImportError those tests assume).
TORCHCODEC_INSTALLED = importlib.util.find_spec("torchcodec") is not None


def _torchcodec_video_available(get_versions: Callable[[], object] | None = None) -> bool:
    try:
        if get_versions is None:
            from torchcodec._core import get_ffmpeg_library_versions

            get_versions = get_ffmpeg_library_versions
        get_versions()
    except (ImportError, RuntimeError, OSError):
        return False
    return True


TORCHCODEC_AVAILABLE = _torchcodec_video_available()
