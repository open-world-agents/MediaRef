"""HuggingFace ``datasets`` integration for MediaRef.

This module provides ``MediaRefFeature``, a custom ``datasets`` feature type
that serializes :class:`MediaRef` objects into Arrow's
``struct<uri: string, pts_ns: int64>`` storage format and registers itself
with the global feature registry on import.

Example:
    >>> from datasets import Dataset, Features
    >>> from mediaref import MediaRef
    >>> from mediaref.hf import MediaRefFeature
    >>> ds = Dataset.from_dict(
    ...     {"frame": [MediaRef(uri="v.mp4", pts_ns=0),
    ...                MediaRef(uri="v.mp4", pts_ns=33_333_333)]},
    ...     features=Features({"frame": MediaRefFeature()}),
    ... )
    >>> ds[0]["frame"]
    MediaRef(uri='v.mp4', pts_ns=0)

Requires ``datasets``. Install via ``pip install mediaref[hf]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union

import pyarrow as pa

try:
    from datasets.features.features import register_feature
except ImportError as e:  # pragma: no cover
    raise ImportError("mediaref.hf requires the `datasets` package. Install with: pip install 'mediaref[hf]'") from e

from .core import MediaRef


@dataclass
class MediaRefFeature:
    """``datasets`` feature type for :class:`MediaRef`.

    Stores values in Arrow as ``struct<uri: string, pts_ns: int64>`` per the
    MediaRef Specification 1.0.

    Args:
        decode: If ``True`` (default), :meth:`decode_example` returns a
            :class:`MediaRef` object that can be lazily loaded via
            ``.to_ndarray()``. If ``False``, returns the raw
            ``{"uri": ..., "pts_ns": ...}`` dict.
        id: Optional identifier passed through by ``datasets``.
    """

    decode: bool = True
    id: Optional[str] = field(default=None, repr=False)

    # Automatically constructed (ClassVar — not part of the dataclass init)
    dtype: ClassVar[str] = "mediaref.MediaRef"
    pa_type: ClassVar[Any] = pa.struct({"uri": pa.string(), "pts_ns": pa.int64()})
    _type: str = field(default="MediaRef", init=False, repr=False)

    def __call__(self) -> pa.StructType:
        return self.pa_type

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode_example(self, value: Union[MediaRef, dict, str]) -> dict:
        """Encode a MediaRef-like input into the Arrow storage shape.

        Accepts:
            - :class:`MediaRef` instance
            - ``dict`` with ``uri`` and optional ``pts_ns``
            - ``str``: interpreted as a URI with ``pts_ns = None`` (still image)
        """
        if isinstance(value, MediaRef):
            return {"uri": value.uri, "pts_ns": value.pts_ns}
        if isinstance(value, dict):
            uri = value.get("uri")
            if uri is None:
                raise ValueError(f"MediaRefFeature: dict must include 'uri'; got {value!r}")
            pts = value.get("pts_ns")
            if pts is not None and not isinstance(pts, int):
                raise TypeError(f"MediaRefFeature: 'pts_ns' must be int or None; got {type(pts).__name__}")
            return {"uri": str(uri), "pts_ns": pts}
        if isinstance(value, str):
            return {"uri": value, "pts_ns": None}
        raise TypeError(f"MediaRefFeature.encode_example: unsupported type {type(value).__name__}")

    def decode_example(
        self,
        value: dict,
        token_per_repo_id: Optional[dict] = None,  # noqa: ARG002 — datasets API contract
    ) -> Union[MediaRef, dict]:
        """Decode the stored struct value back into a :class:`MediaRef`.

        When ``self.decode`` is False, returns the raw dict unchanged.
        """
        if not self.decode:
            return value
        return MediaRef(uri=value["uri"], pts_ns=value.get("pts_ns"))

    # ------------------------------------------------------------------
    # storage casting
    # ------------------------------------------------------------------

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the MediaRef struct storage type.

        Accepted input:
            - ``pa.string()``: treated as a URI with ``pts_ns = null``.
            - ``pa.struct`` containing at least a ``uri`` field; ``pts_ns`` is
              added as null when missing.

        Returns a ``StructArray`` of type ``self.pa_type``.
        """
        if pa.types.is_string(storage.type):
            uri_array = storage
            pts_array = pa.array([None] * len(storage), type=pa.int64())
            return pa.StructArray.from_arrays([uri_array, pts_array], ["uri", "pts_ns"], mask=storage.is_null())

        if pa.types.is_struct(storage.type):
            if storage.type.get_field_index("uri") < 0:
                raise ValueError(
                    "MediaRefFeature.cast_storage: struct must contain a "
                    f"'uri' field; got {[f.name for f in storage.type]}"
                )
            uri_array = storage.field("uri").cast(pa.string())
            if storage.type.get_field_index("pts_ns") >= 0:
                pts_array = storage.field("pts_ns").cast(pa.int64())
            else:
                pts_array = pa.array([None] * len(storage), type=pa.int64())
            return pa.StructArray.from_arrays([uri_array, pts_array], ["uri", "pts_ns"], mask=storage.is_null())

        raise TypeError(
            f"MediaRefFeature.cast_storage: cannot cast Arrow type {storage.type}; "
            "expected pa.string() or pa.struct(uri, pts_ns)."
        )


# Register with the HuggingFace datasets feature registry on import.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental.*",
        category=UserWarning,
    )
    register_feature(MediaRefFeature, "MediaRef")


__all__ = ["MediaRefFeature"]
