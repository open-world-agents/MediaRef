"""TorchCodec adapter with host NumPy output."""

import inspect
from typing import Any, ClassVar, Optional

import numpy as np
from torchcodec.decoders import VideoDecoder

from ..resource_cache import ResourceCache
from .codec_decoder import CodecVideoDecoder, _DecoderState
from .frame_batch import FrameBatch


class TorchCodecVideoDecoder(CodecVideoDecoder):
    """TorchCodec implementation of the cached playback interface."""

    cache: ClassVar[ResourceCache[_DecoderState]] = ResourceCache(max_size=10)
    _open_decoder = staticmethod(VideoDecoder)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: Optional[float] = None
    ) -> FrameBatch:
        parameters = inspect.signature(VideoDecoder.get_frames_played_in_range).parameters
        supports_fps = "fps" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
        if fps is not None and not supports_fps:
            raise NotImplementedError(
                "The installed version of TorchCodec (<=0.10.0) does not support "
                "the 'fps' parameter in get_frames_played_in_range. "
                "Upgrade TorchCodec or use fps=None."
            )
        return super().get_frames_played_in_range(start_seconds, stop_seconds, fps)
