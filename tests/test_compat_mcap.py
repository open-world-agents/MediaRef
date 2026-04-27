"""Tests for ``mediaref.compat.mcap`` interop helpers.

The full round-trip (write → read) runs in-memory via ``io.BytesIO`` —
no temp files, no filesystem dependency.
"""

from __future__ import annotations

import io
import json

import pytest

mcap_lib = pytest.importorskip("mcap")
from mcap.reader import make_reader  # noqa: E402
from mcap.writer import Writer  # noqa: E402

from mediaref import MediaRef  # noqa: E402
from mediaref.compat.mcap import (  # noqa: E402
    MEDIAREF_MESSAGE_ENCODING,
    MEDIAREF_SCHEMA,
    MEDIAREF_SCHEMA_ENCODING,
    MEDIAREF_SCHEMA_NAME,
    iter_mediarefs,
    register_mediaref_channel,
    register_mediaref_schema,
    resolve_against_mcap,
    write_mediaref,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def open_writer():
    """Yield a started mcap writer over BytesIO; finalizes on teardown."""
    buf = io.BytesIO()
    w = Writer(buf)
    w.start()
    try:
        yield w, buf
    finally:
        try:
            w.finish()
        except Exception:  # pragma: no cover — already finalized in test
            pass


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_schema_name(self):
        assert MEDIAREF_SCHEMA_NAME == "MediaRef"

    def test_schema_encoding(self):
        assert MEDIAREF_SCHEMA_ENCODING == "jsonschema"

    def test_message_encoding(self):
        assert MEDIAREF_MESSAGE_ENCODING == "json"

    def test_schema_required_uri_field(self):
        assert "uri" in MEDIAREF_SCHEMA["required"]
        assert MEDIAREF_SCHEMA["properties"]["uri"]["type"] == "string"

    def test_schema_pts_ns_nullable_int(self):
        prop = MEDIAREF_SCHEMA["properties"]["pts_ns"]
        # Allow null for still images per Spec §3.3.
        assert "integer" in prop["type"]
        assert "null" in prop["type"]

    def test_schema_disallows_extra_fields(self):
        # Spec §1: producers MUST NOT emit additional fields.
        assert MEDIAREF_SCHEMA["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Writer-side
# ---------------------------------------------------------------------------


class TestRegisterAndWrite:
    def test_register_schema_returns_int_id(self, open_writer):
        w, _ = open_writer
        sid = register_mediaref_schema(w)
        assert isinstance(sid, int)

    def test_register_channel_returns_int_id(self, open_writer):
        w, _ = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/screen", sid)
        assert isinstance(cid, int)

    def test_write_does_not_raise_for_video_frame(self, open_writer):
        w, _ = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/screen", sid)
        write_mediaref(w, cid, 1_000_000_000, MediaRef(uri="clip.mp4", pts_ns=0))

    def test_write_does_not_raise_for_still_image(self, open_writer):
        w, _ = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/photo", sid)
        write_mediaref(w, cid, 1_000_000_000, MediaRef(uri="photo.png"))

    def test_publish_time_defaults_to_log_time(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/t", sid)
        write_mediaref(w, cid, 42, MediaRef(uri="x.png"))
        w.finish()
        buf.seek(0)
        msgs = list(make_reader(buf).iter_messages())
        assert len(msgs) == 1
        _, _, m = msgs[0]
        assert m.log_time == 42
        assert m.publish_time == 42


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_single_video_frame(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/screen", sid)
        write_mediaref(w, cid, 1_000_000_000, MediaRef(uri="episode.mp4", pts_ns=33_333_333))
        w.finish()

        buf.seek(0)
        out = list(iter_mediarefs(make_reader(buf)))
        assert len(out) == 1
        ts, ref = out[0]
        assert ts == 1_000_000_000
        assert ref == MediaRef(uri="episode.mp4", pts_ns=33_333_333)

    def test_still_image_pts_omitted(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/photo", sid)
        write_mediaref(w, cid, 0, MediaRef(uri="photo.png"))
        w.finish()

        buf.seek(0)
        # On the wire, pts_ns must be omitted (size optimization, Spec §1.2).
        msgs = list(make_reader(buf).iter_messages())
        _, _, m = msgs[0]
        payload = json.loads(m.data.decode("utf-8"))
        assert payload == {"uri": "photo.png"}
        assert "pts_ns" not in payload

        buf.seek(0)
        ts, ref = next(iter_mediarefs(make_reader(buf)))
        assert ts == 0
        assert ref == MediaRef(uri="photo.png")

    def test_many_messages_one_topic(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/screen", sid)
        N = 25
        for i in range(N):
            write_mediaref(w, cid, i * 1_000_000, MediaRef(uri="v.mp4", pts_ns=i * 33_333_333))
        w.finish()

        buf.seek(0)
        out = list(iter_mediarefs(make_reader(buf)))
        assert len(out) == N
        for i, (ts, ref) in enumerate(out):
            assert ts == i * 1_000_000
            assert ref.uri == "v.mp4"
            assert ref.pts_ns == i * 33_333_333

    def test_multiple_topics_filtering(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        ca = register_mediaref_channel(w, "/cam_a", sid)
        cb = register_mediaref_channel(w, "/cam_b", sid)
        write_mediaref(w, ca, 1, MediaRef(uri="a.mp4", pts_ns=0))
        write_mediaref(w, cb, 2, MediaRef(uri="b.mp4", pts_ns=0))
        write_mediaref(w, ca, 3, MediaRef(uri="a.mp4", pts_ns=10))
        w.finish()

        buf.seek(0)
        only_a = list(iter_mediarefs(make_reader(buf), topics=["/cam_a"]))
        assert len(only_a) == 2
        assert all(r.uri == "a.mp4" for _, r in only_a)

        buf.seek(0)
        all_msgs = list(iter_mediarefs(make_reader(buf)))
        assert len(all_msgs) == 3

    def test_time_range_filtering(self, open_writer):
        w, buf = open_writer
        sid = register_mediaref_schema(w)
        cid = register_mediaref_channel(w, "/screen", sid)
        for i in range(10):
            write_mediaref(w, cid, i, MediaRef(uri="v.mp4", pts_ns=i))
        w.finish()

        buf.seek(0)
        # log_time ∈ [3, 7) — mcap end_time is exclusive
        out = list(iter_mediarefs(make_reader(buf), start_time_ns=3, end_time_ns=7))
        assert [ts for ts, _ in out] == [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestReaderRejectsBadPayloads:
    def _writer_with_one_raw_message(self, schema_data: bytes, payload: bytes):
        buf = io.BytesIO()
        w = Writer(buf)
        w.start()
        sid = w.register_schema(
            name=MEDIAREF_SCHEMA_NAME,
            encoding=MEDIAREF_SCHEMA_ENCODING,
            data=schema_data,
        )
        cid = w.register_channel(topic="/x", message_encoding=MEDIAREF_MESSAGE_ENCODING, schema_id=sid)
        w.add_message(channel_id=cid, log_time=0, data=payload, publish_time=0)
        w.finish()
        buf.seek(0)
        return buf

    def test_invalid_json_raises(self):
        buf = self._writer_with_one_raw_message(
            schema_data=json.dumps(MEDIAREF_SCHEMA).encode("utf-8"),
            payload=b"{not json",
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            list(iter_mediarefs(make_reader(buf)))

    def test_missing_uri_raises(self):
        buf = self._writer_with_one_raw_message(
            schema_data=json.dumps(MEDIAREF_SCHEMA).encode("utf-8"),
            payload=json.dumps({"pts_ns": 0}).encode("utf-8"),
        )
        with pytest.raises(ValueError, match="MediaRef"):
            list(iter_mediarefs(make_reader(buf)))


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolveAgainstMcap:
    def test_resolves_relative_uri(self, tmp_path):
        mcap_file = tmp_path / "session.mcap"
        mcap_file.touch()
        refs = [
            MediaRef(uri="videos/clip.mp4", pts_ns=0),
            MediaRef(uri="videos/clip.mp4", pts_ns=1_000_000_000),
        ]
        out = resolve_against_mcap(refs, mcap_file)
        assert all(r.uri.endswith("videos/clip.mp4") for r in out)
        assert all(r.uri.startswith(str(tmp_path)) for r in out)
        # pts_ns preserved
        assert [r.pts_ns for r in out] == [0, 1_000_000_000]

    def test_passes_through_absolute_uri(self, tmp_path):
        mcap_file = tmp_path / "session.mcap"
        mcap_file.touch()
        absolute = MediaRef(uri="/data/clip.mp4", pts_ns=0)
        out = resolve_against_mcap([absolute], mcap_file)
        assert out[0].uri == "/data/clip.mp4"

    def test_passes_through_remote_uri(self, tmp_path):
        mcap_file = tmp_path / "session.mcap"
        mcap_file.touch()
        remote = MediaRef(uri="https://example.com/v.mp4", pts_ns=0)
        out = resolve_against_mcap([remote], mcap_file, on_unresolvable="ignore")
        assert out[0].uri == "https://example.com/v.mp4"
