"""CPU codec adapter with NumPy output and no Torch dependency."""

from typing import Any, ClassVar

import numpy as np
from tensorcodec.decoders import VideoDecoder

from ..resource_cache import ResourceCache
from .codec_decoder import CodecVideoDecoder, _DecoderState


class TensorCodecVideoDecoder(CodecVideoDecoder):
    """TensorCodec implementation of the cached playback interface."""

    cache: ClassVar[ResourceCache[_DecoderState]] = ResourceCache(max_size=10)
    _open_decoder = staticmethod(VideoDecoder)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        return np.asarray(value)
