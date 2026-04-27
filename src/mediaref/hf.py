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

# Single source of truth for the feature type identifier (registry key,
# `_type` discriminator, and dtype suffix all reuse this).
_FEATURE_NAME = "MediaRef"


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

    dtype: ClassVar[str] = f"mediaref.{_FEATURE_NAME}"
    pa_type: ClassVar[Any] = pa.struct({"uri": pa.string(), "pts_ns": pa.int64()})
    _type: str = field(default=_FEATURE_NAME, init=False, repr=False)

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

        Dicts are validated through :class:`MediaRef` itself so the same
        coercions that work in the constructor (``DataURI`` → str, numpy
        integers, ``Optional[int]`` semantics for ``pts_ns``) work here.
        """
        if isinstance(value, MediaRef):
            return {"uri": value.uri, "pts_ns": value.pts_ns}
        if isinstance(value, dict):
            if "uri" not in value or value["uri"] is None:
                raise ValueError(f"MediaRefFeature: dict must include 'uri'; got {value!r}")
            ref = MediaRef.model_validate(value)
            return {"uri": ref.uri, "pts_ns": ref.pts_ns}
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
            pts_array = pa.nulls(len(storage), type=pa.int64())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("uri") < 0:
                raise ValueError(
                    "MediaRefFeature.cast_storage: struct must contain a "
                    f"'uri' field; got {[f.name for f in storage.type]}"
                )
            uri_array = storage.field("uri").cast(pa.string())
            if storage.type.get_field_index("pts_ns") >= 0:
                pts_array = storage.field("pts_ns").cast(pa.int64())
            else:
                pts_array = pa.nulls(len(storage), type=pa.int64())
        else:
            raise TypeError(
                f"MediaRefFeature.cast_storage: cannot cast Arrow type {storage.type}; "
                "expected pa.string() or pa.struct(uri, pts_ns)."
            )
        return pa.StructArray.from_arrays([uri_array, pts_array], ["uri", "pts_ns"], mask=storage.is_null())


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental.*",
        category=UserWarning,
    )
    register_feature(MediaRefFeature, _FEATURE_NAME)


__all__ = ["MediaRefFeature"]
