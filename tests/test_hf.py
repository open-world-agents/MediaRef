"""Tests for the ``mediaref.hf`` HuggingFace ``datasets`` adapter."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")

from mediaref import MediaRef  # noqa: E402
from mediaref.hf import MediaRefFeature  # noqa: E402


class TestPaType:
    """Wire-format guarantees per MediaRef Specification 1.0."""

    def test_pa_type_struct_uri_pts_ns(self):
        feat = MediaRefFeature()
        t = feat()
        assert pa.types.is_struct(t)
        names = {f.name for f in t}
        assert names == {"uri", "pts_ns"}
        assert t.field("uri").type == pa.string()
        assert t.field("pts_ns").type == pa.int64()

    def test_class_var_pa_type_matches_call(self):
        # Both the ClassVar and __call__ must agree.
        assert MediaRefFeature.pa_type == MediaRefFeature()()

    def test_type_string_is_MediaRef(self):
        # `_type` is what `register_feature` keys on for deserialization.
        assert MediaRefFeature()._type == "MediaRef"


class TestEncodeExample:
    """encode_example accepts MediaRef, dict, or str."""

    def test_encode_mediaref_with_pts(self):
        ref = MediaRef(uri="v.mp4", pts_ns=1_500_000_000)
        assert MediaRefFeature().encode_example(ref) == {
            "uri": "v.mp4",
            "pts_ns": 1_500_000_000,
        }

    def test_encode_mediaref_image(self):
        ref = MediaRef(uri="img.png")
        assert MediaRefFeature().encode_example(ref) == {
            "uri": "img.png",
            "pts_ns": None,
        }

    def test_encode_dict_with_pts(self):
        out = MediaRefFeature().encode_example({"uri": "v.mp4", "pts_ns": 0})
        assert out == {"uri": "v.mp4", "pts_ns": 0}

    def test_encode_dict_without_pts(self):
        # Absent pts_ns means still image (None).
        out = MediaRefFeature().encode_example({"uri": "img.png"})
        assert out == {"uri": "img.png", "pts_ns": None}

    def test_encode_string_is_uri(self):
        out = MediaRefFeature().encode_example("img.png")
        assert out == {"uri": "img.png", "pts_ns": None}

    def test_encode_dict_missing_uri_raises(self):
        with pytest.raises(ValueError, match="uri"):
            MediaRefFeature().encode_example({"pts_ns": 0})

    def test_encode_dict_with_non_int_pts_raises(self):
        with pytest.raises(TypeError, match="pts_ns"):
            MediaRefFeature().encode_example({"uri": "x", "pts_ns": 1.5})

    def test_encode_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            MediaRefFeature().encode_example(42)


class TestDecodeExample:
    """decode_example yields MediaRef (or raw dict when decode=False)."""

    def test_decode_returns_mediaref(self):
        out = MediaRefFeature().decode_example({"uri": "v.mp4", "pts_ns": 0})
        assert isinstance(out, MediaRef)
        assert out.uri == "v.mp4"
        assert out.pts_ns == 0

    def test_decode_image_with_null_pts(self):
        out = MediaRefFeature().decode_example({"uri": "img.png", "pts_ns": None})
        assert isinstance(out, MediaRef)
        assert out.uri == "img.png"
        assert out.pts_ns is None

    def test_decode_disabled_returns_raw_dict(self):
        raw = {"uri": "v.mp4", "pts_ns": 0}
        out = MediaRefFeature(decode=False).decode_example(raw)
        assert out is raw  # identity — no copy when decode disabled


class TestCastStorage:
    """Arrow casting handles common input shapes per Spec §1.1."""

    def test_cast_from_string_array(self):
        arr = pa.array(["a.png", "b.png"], type=pa.string())
        out = MediaRefFeature().cast_storage(arr)
        assert pa.types.is_struct(out.type)
        assert out.field("uri").to_pylist() == ["a.png", "b.png"]
        assert out.field("pts_ns").to_pylist() == [None, None]

    def test_cast_from_full_struct(self):
        arr = pa.array(
            [{"uri": "v.mp4", "pts_ns": 0}, {"uri": "v.mp4", "pts_ns": 33_333_333}],
            type=pa.struct({"uri": pa.string(), "pts_ns": pa.int64()}),
        )
        out = MediaRefFeature().cast_storage(arr)
        assert out.field("uri").to_pylist() == ["v.mp4", "v.mp4"]
        assert out.field("pts_ns").to_pylist() == [0, 33_333_333]

    def test_cast_from_struct_missing_pts_field(self):
        arr = pa.array(
            [{"uri": "img.png"}, {"uri": "img2.png"}],
            type=pa.struct({"uri": pa.string()}),
        )
        out = MediaRefFeature().cast_storage(arr)
        assert out.field("uri").to_pylist() == ["img.png", "img2.png"]
        assert out.field("pts_ns").to_pylist() == [None, None]

    def test_cast_struct_without_uri_raises(self):
        arr = pa.array([{"pts_ns": 0}], type=pa.struct({"pts_ns": pa.int64()}))
        with pytest.raises(ValueError, match="uri"):
            MediaRefFeature().cast_storage(arr)

    def test_cast_unsupported_type_raises(self):
        arr = pa.array([1, 2, 3], type=pa.int32())
        with pytest.raises(TypeError):
            MediaRefFeature().cast_storage(arr)


class TestRegistration:
    """Importing the module registers ``MediaRef`` in the global feature registry."""

    def test_registered_in_global_feature_types(self):
        from datasets.features.features import _FEATURE_TYPES

        assert "MediaRef" in _FEATURE_TYPES
        assert _FEATURE_TYPES["MediaRef"] is MediaRefFeature

    def test_features_from_dict_round_trip(self):
        # The registry powers Features deserialization via the `_type` field.
        from datasets import Features

        feats = Features({"frame": MediaRefFeature()})
        as_dict = feats.to_dict()
        restored = Features.from_dict(as_dict)
        assert isinstance(restored["frame"], MediaRefFeature)


class TestDatasetIntegration:
    """End-to-end smoke test against ``datasets.Dataset``."""

    def test_from_dict_with_mediarefs(self):
        from datasets import Dataset, Features

        refs = [
            MediaRef(uri="v.mp4", pts_ns=0),
            MediaRef(uri="v.mp4", pts_ns=33_333_333),
            MediaRef(uri="img.png"),  # still image
        ]
        ds = Dataset.from_dict(
            {"frame": refs},
            features=Features({"frame": MediaRefFeature()}),
        )
        assert len(ds) == 3
        out0 = ds[0]["frame"]
        out1 = ds[1]["frame"]
        out2 = ds[2]["frame"]
        assert isinstance(out0, MediaRef)
        assert (out0.uri, out0.pts_ns) == ("v.mp4", 0)
        assert (out1.uri, out1.pts_ns) == ("v.mp4", 33_333_333)
        assert (out2.uri, out2.pts_ns) == ("img.png", None)

    def test_from_dict_with_plain_dicts(self):
        from datasets import Dataset, Features

        rows = [
            {"uri": "v.mp4", "pts_ns": 0},
            {"uri": "img.png", "pts_ns": None},
        ]
        ds = Dataset.from_dict(
            {"frame": rows},
            features=Features({"frame": MediaRefFeature()}),
        )
        out = ds[1]["frame"]
        assert isinstance(out, MediaRef)
        assert out.uri == "img.png"
        assert out.pts_ns is None

    def test_decode_disabled_returns_dicts(self):
        from datasets import Dataset, Features

        refs = [MediaRef(uri="v.mp4", pts_ns=0)]
        ds = Dataset.from_dict(
            {"frame": refs},
            features=Features({"frame": MediaRefFeature(decode=False)}),
        )
        out = ds[0]["frame"]
        assert isinstance(out, dict)
        assert out["uri"] == "v.mp4"
        assert out["pts_ns"] == 0
