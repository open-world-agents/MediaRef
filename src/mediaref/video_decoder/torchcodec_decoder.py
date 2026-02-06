"""TorchCodec-based video decoder."""

from typing import ClassVar, List

import numpy as np
from torchcodec.decoders import VideoDecoder

from .._typing import PathLike
from ..resource_cache import ResourceCache
from .base import BaseVideoDecoder
from .frame_batch import FrameBatch


class TorchCodecVideoDecoder(VideoDecoder, BaseVideoDecoder):
    """Cached TorchCodec video decoder.

    Wraps TorchCodec's VideoDecoder with caching for efficient resource management.

    Args:
        source: Path to video file or URL
        **kwargs: Additional arguments passed to TorchCodec's VideoDecoder

    Examples:
        >>> with TorchCodecVideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 1.0, 2.0])
    """

    cache: ClassVar[ResourceCache[VideoDecoder]] = ResourceCache(max_size=10)
    _skip_init = False

    def __new__(cls, source: PathLike, **kwargs):
        """Create or retrieve cached decoder instance."""
        cache_key = str(source)
        if cache_key in cls.cache:
            instance = cls.cache.acquire_entry(cache_key)
            instance._skip_init = True
        else:
            instance = super().__new__(cls)
            instance._skip_init = False
        return instance

    def __init__(self, source: PathLike, **kwargs):
        """Initialize decoder if not retrieved from cache."""
        if getattr(self, "_skip_init", False):
            return
        super().__init__(str(source), **kwargs)
        self._cache_key = str(source)
        self.cache.add_entry(self._cache_key, self, lambda: None)

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        Delegates to TorchCodec's native get_frames_played_at
        which implements the correct playback semantics.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch containing frame data and timing information
        """
        # TorchCodec returns its own FrameBatch (dataclass), convert to ours
        torchcodec_batch = VideoDecoder.get_frames_played_at(self, seconds)
        return FrameBatch(
            data=torchcodec_batch.data.numpy(),
            # Use .numpy() first to avoid DeprecationWarning from numpy 2.0
            # about __array__ not accepting the 'copy' keyword
            pts_seconds=torchcodec_batch.pts_seconds.numpy().astype(np.float64),
            duration_seconds=torchcodec_batch.duration_seconds.numpy().astype(np.float64),
        )

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: float | None = None
    ) -> FrameBatch:
        """Return multiple frames in the given range.

        Delegates to TorchCodec's native get_frames_played_in_range.

        Args:
            start_seconds: Time, in seconds, of the start of the range.
            stop_seconds: Time, in seconds, of the end of the range (excluded).
            fps: If specified, resample output to this frame rate. If None,
                returns frames at the source video's frame rate.

        Returns:
            FrameBatch containing frame data and timing information.

        Raises:
            NotImplementedError: If ``fps`` is specified but the installed
                TorchCodec version (<=0.10.0) does not support it.
        """
        if fps is not None:
            try:
                torchcodec_batch = VideoDecoder.get_frames_played_in_range(
                    self, start_seconds=start_seconds, stop_seconds=stop_seconds, fps=fps
                )
            except TypeError:
                raise NotImplementedError(
                    "The installed version of TorchCodec (<=0.10.0) does not support "
                    "the 'fps' parameter in get_frames_played_in_range. "
                    "Upgrade TorchCodec or use fps=None."
                )
        else:
            torchcodec_batch = VideoDecoder.get_frames_played_in_range(
                self, start_seconds=start_seconds, stop_seconds=stop_seconds
            )
        return FrameBatch(
            data=torchcodec_batch.data.numpy(),
            pts_seconds=torchcodec_batch.pts_seconds.numpy().astype(np.float64),
            duration_seconds=torchcodec_batch.duration_seconds.numpy().astype(np.float64),
        )

    def close(self):
        """Release cache reference. Safe to call multiple times."""
        if hasattr(self, "_cache_key") and self._cache_key in self.cache and self.cache[self._cache_key].refs > 0:
            self.cache.release_entry(self._cache_key)
