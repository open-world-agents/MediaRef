# Tests for mediaref package

# Shared constant for TorchCodec availability — used by skipif markers across test files.
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    TORCHCODEC_AVAILABLE = False
