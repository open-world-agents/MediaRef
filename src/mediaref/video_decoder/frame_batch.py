"""Frame batch data structure for video decoders."""

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


# Copied from https://github.com/pytorch/torchcodec/blob/main/src/torchcodec/_frame.py#L14-L27
def _frame_repr(self):
    """Utility to replace __repr__ method of dataclasses.

    This prints the shape of the .data tensor rather than printing the
    (potentially very long) data tensor itself.
    """
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class FrameBatch:
    """Batch of video frames with timing information in NCHW format.

    This data structure is compatible with TorchCodec's frame batch format
    and provides a unified interface for batch frame data across different
    decoder implementations.

    Attributes:
        data: Frame data in NCHW format (N, C, H, W) where:
            - N: Number of frames
            - C: Number of channels (typically 3 for RGB)
            - H: Frame height in pixels
            - W: Frame width in pixels
        pts_seconds: Presentation timestamps in seconds for each frame (N,)
        duration_seconds: Duration of each frame in seconds (N,)

    Examples:
        >>> import numpy as np
        >>> data = np.random.randint(0, 255, (10, 3, 720, 1280), dtype=np.uint8)
        >>> pts = np.linspace(0, 1, 10, dtype=np.float64)
        >>> duration = np.full(10, 0.1, dtype=np.float64)
        >>> batch = FrameBatch(data=data, pts_seconds=pts, duration_seconds=duration)
        >>> print(batch)
        FrameBatch:
          data (shape): (10, 3, 720, 1280)
          pts_seconds: [0.  0.11111111 0.22222222 ...]
          duration_seconds: [0.1 0.1 0.1 ...]
    """

    data: npt.NDArray[np.uint8]  # [N, C, H, W]
    pts_seconds: npt.NDArray[np.float64]  # [N]
    duration_seconds: npt.NDArray[np.float64]  # [N]

    __repr__ = _frame_repr


__all__ = ["FrameBatch"]
