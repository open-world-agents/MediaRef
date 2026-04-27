"""Interop helpers for `mcap <https://mcap.dev>`_.

mcap is a containerized log format used in robotics, autonomous driving,
and OWA-style desktop agent recordings. It stores timestamped messages
on named channels with a registered schema. This module provides the
canonical way to put :class:`MediaRef` values into mcap streams and read
them back out.

Wire format:
    Messages are encoded as JSON per :doc:`MediaRef Specification 1.0 §1.2
    <../docs/SPEC.md>`. Schemas use the ``jsonschema`` encoding so that
    mcap-aware tools (Foxglove, ``mcap doctor``, etc.) can validate them.

Example:
    >>> import io
    >>> from mcap.writer import Writer
    >>> from mcap.reader import make_reader
    >>> from mediaref import MediaRef
    >>> from mediaref.compat.mcap import (
    ...     register_mediaref_schema, register_mediaref_channel,
    ...     write_mediaref, iter_mediarefs,
    ... )
    >>>
    >>> buf = io.BytesIO()
    >>> w = Writer(buf)
    >>> w.start()
    >>> sid = register_mediaref_schema(w)
    >>> cid = register_mediaref_channel(w, "/screen", sid)
    >>> write_mediaref(w, cid, 1_000_000_000,
    ...                MediaRef(uri="clip.mp4", pts_ns=0))
    >>> w.finish()
    >>> _ = buf.seek(0)
    >>> for ts, ref in iter_mediarefs(make_reader(buf), topics=["/screen"]):
    ...     print(ts, ref.uri, ref.pts_ns)

Requires the ``mcap`` package (``pip install 'mediaref[mcap]'``). The
``mcap`` library has no MediaRef knowledge; this module is the bridge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Union

from ..core import MediaRef

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from mcap.reader import McapReader
    from mcap.writer import Writer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical mcap schema name for a MediaRef message.
MEDIAREF_SCHEMA_NAME: str = "MediaRef"

#: Schema encoding registered with mcap. Foxglove and ``mcap doctor`` accept
#: ``jsonschema`` natively.
MEDIAREF_SCHEMA_ENCODING: str = "jsonschema"

#: Per-message encoding registered with mcap. Bytes on the wire are UTF-8
#: JSON matching :data:`MEDIAREF_SCHEMA`.
MEDIAREF_MESSAGE_ENCODING: str = "json"

#: JSON Schema (Draft 2020-12) for a single MediaRef message. Mirrors
#: MediaRef Specification 1.0 §1.2.
MEDIAREF_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/open-world-agents/MediaRef/spec/v1/mcap-message.schema.json",
    "title": "MediaRef",
    "description": "A frame-level media reference per MediaRef Specification 1.0.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "uri": {
            "type": "string",
            "description": "RFC 3986 URI, RFC 2397 data URI, or POSIX path.",
        },
        "pts_ns": {
            "type": ["integer", "null"],
            "description": "Frame presentation timestamp in nanoseconds; null/absent for still images.",
            "minimum": 0,
        },
    },
    "required": ["uri"],
}


# ---------------------------------------------------------------------------
# Writer-side helpers
# ---------------------------------------------------------------------------


def register_mediaref_schema(writer: "Writer") -> int:
    """Register the canonical MediaRef JSON schema in an open mcap writer.

    Idempotent at the API level: each call adds a new mcap schema record,
    so prefer to call once per writer and reuse the returned id.

    Args:
        writer: An :class:`mcap.writer.Writer` after ``writer.start()``.

    Returns:
        The integer ``schema_id`` to pass to
        :func:`register_mediaref_channel`.
    """
    return writer.register_schema(
        name=MEDIAREF_SCHEMA_NAME,
        encoding=MEDIAREF_SCHEMA_ENCODING,
        data=json.dumps(MEDIAREF_SCHEMA, separators=(",", ":")).encode("utf-8"),
    )


def register_mediaref_channel(
    writer: "Writer",
    topic: str,
    schema_id: int,
    *,
    metadata: Optional[dict] = None,
) -> int:
    """Register a MediaRef channel on ``topic``.

    Args:
        writer: An open mcap writer.
        topic: mcap topic name (typical convention: leading slash, e.g.
            ``/observation/image_front``).
        schema_id: Schema id from :func:`register_mediaref_schema`.
        metadata: Optional channel metadata (added to mcap channel
            record).

    Returns:
        The integer ``channel_id`` to pass to :func:`write_mediaref`.
    """
    return writer.register_channel(
        topic=topic,
        message_encoding=MEDIAREF_MESSAGE_ENCODING,
        schema_id=schema_id,
        metadata=metadata or {},
    )


def write_mediaref(
    writer: "Writer",
    channel_id: int,
    log_time_ns: int,
    ref: MediaRef,
    *,
    publish_time_ns: Optional[int] = None,
    sequence: int = 0,
) -> None:
    """Write a :class:`MediaRef` to an mcap stream as one JSON message.

    Args:
        writer: An open mcap writer.
        channel_id: Channel id from :func:`register_mediaref_channel`.
        log_time_ns: mcap log time (nanoseconds since Unix epoch is the
            mcap convention, but any monotonic timeline works as long as
            it's consistent across the file).
        ref: The MediaRef to encode.
        publish_time_ns: mcap publish time. Defaults to ``log_time_ns``
            when omitted.
        sequence: Optional channel sequence number.
    """
    payload = ref.model_dump_json(exclude_none=True).encode("utf-8")
    writer.add_message(
        channel_id=channel_id,
        log_time=int(log_time_ns),
        data=payload,
        publish_time=int(publish_time_ns if publish_time_ns is not None else log_time_ns),
        sequence=int(sequence),
    )


# ---------------------------------------------------------------------------
# Reader-side helpers
# ---------------------------------------------------------------------------


def iter_mediarefs(
    reader: "McapReader",
    *,
    topics: Optional[Iterable[str]] = None,
    start_time_ns: Optional[int] = None,
    end_time_ns: Optional[int] = None,
) -> Iterator[tuple[int, MediaRef]]:
    """Iterate ``(log_time_ns, MediaRef)`` pairs from an mcap reader.

    Args:
        reader: A reader from :func:`mcap.reader.make_reader`.
        topics: If given, only yield messages on these topics.
        start_time_ns: Inclusive lower bound on ``log_time``.
        end_time_ns: Exclusive upper bound on ``log_time``.

    Yields:
        ``(log_time_ns, MediaRef)`` pairs in mcap log-time order.

    Raises:
        ValueError: If a non-MediaRef message is encountered on a
            channel claimed to use the MediaRef schema.
    """
    iter_kwargs: dict = {}
    if topics is not None:
        iter_kwargs["topics"] = list(topics)
    if start_time_ns is not None:
        iter_kwargs["start_time"] = int(start_time_ns)
    if end_time_ns is not None:
        iter_kwargs["end_time"] = int(end_time_ns)

    for _schema, _channel, message in reader.iter_messages(**iter_kwargs):
        try:
            payload = json.loads(message.data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Channel '{_channel.topic}' message at log_time={message.log_time} is not valid JSON: {e}"
            ) from e
        try:
            ref = MediaRef.model_validate(payload)
        except Exception as e:
            raise ValueError(
                f"Channel '{_channel.topic}' message at log_time={message.log_time} is not a MediaRef: {e}"
            ) from e
        yield message.log_time, ref


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_against_mcap(
    refs: Iterable[MediaRef],
    mcap_path: Union[str, Path],
    on_unresolvable: str = "ignore",
) -> list[MediaRef]:
    """Resolve relative URIs against the directory containing ``mcap_path``.

    mcap files commonly sit next to the ``.mp4``/``.mkv`` they reference,
    and the parquet/JSON ``uri`` field is recorded as a relative path.
    Use this helper after :func:`iter_mediarefs` to obtain absolute refs.

    Args:
        refs: Iterable of :class:`MediaRef` (e.g. the output of
            :func:`iter_mediarefs`, after extracting the second tuple
            element).
        mcap_path: Path to the ``.mcap`` file. Its parent directory is
            used as the resolution base.
        on_unresolvable: Forwarded to
            :meth:`MediaRef.resolve_relative_path` for embedded / remote
            / cloud URIs (default ``"ignore"`` — they pass through
            unchanged).

    Returns:
        A list of resolved MediaRefs. Order matches input order.
    """
    base = str(Path(mcap_path).resolve().parent)
    return [r.resolve_relative_path(base, on_unresolvable=on_unresolvable) for r in refs]


__all__ = [
    "MEDIAREF_SCHEMA",
    "MEDIAREF_SCHEMA_NAME",
    "MEDIAREF_SCHEMA_ENCODING",
    "MEDIAREF_MESSAGE_ENCODING",
    "register_mediaref_schema",
    "register_mediaref_channel",
    "write_mediaref",
    "iter_mediarefs",
    "resolve_against_mcap",
]
