# Tests for mediaref package

import importlib.util

# Shared constants for TorchCodec availability — used by skipif markers across test files.
#
# TORCHCODEC_AVAILABLE: torchcodec imports AND loads successfully (full smoke).
# TORCHCODEC_INSTALLED: torchcodec is present in the environment, regardless of
#   whether its native libraries load. Used to gate "not installed" tests so they
#   don't run on machines where torchcodec is installed but its FFmpeg ABI is
#   incompatible (in which case the real failure mode is a RuntimeError, not the
#   ImportError those tests assume).
TORCHCODEC_INSTALLED = importlib.util.find_spec("torchcodec") is not None

try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    TORCHCODEC_AVAILABLE = False
