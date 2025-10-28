"""Tests for MediaRef Pydantic serialization and deserialization."""

import json

import pytest

from mediaref import MediaRef


class TestModelDump:
    """Test model_dump method."""

    def test_model_dump_image(self):
        """Test model_dump for image reference."""
        ref = MediaRef(uri="image.png")
        data = ref.model_dump()

        assert data == {"uri": "image.png", "pts_ns": None}

    def test_model_dump_video(self):
        """Test model_dump for video reference."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        data = ref.model_dump()

        assert data == {"uri": "video.mp4", "pts_ns": 1_000_000_000}

    def test_model_dump_data_uri(self):
        """Test model_dump for data URI."""
        data_uri = "data:image/png;base64,iVBORw0KG..."
        ref = MediaRef(uri=data_uri)
        data = ref.model_dump()

        assert data == {"uri": data_uri, "pts_ns": None}

    def test_model_dump_remote_url(self):
        """Test model_dump for remote URL."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        data = ref.model_dump()

        assert data == {"uri": "https://example.com/image.jpg", "pts_ns": None}

    def test_model_dump_exclude_none(self):
        """Test model_dump with exclude_none=True."""
        ref = MediaRef(uri="image.png")
        data = ref.model_dump(exclude_none=True)

        assert data == {"uri": "image.png"}
        assert "pts_ns" not in data

    def test_model_dump_exclude_none_with_pts_ns(self):
        """Test model_dump with exclude_none=True when pts_ns is set."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        data = ref.model_dump(exclude_none=True)

        assert data == {"uri": "video.mp4", "pts_ns": 1_000_000_000}


class TestModelDumpJson:
    """Test model_dump_json method."""

    def test_model_dump_json_image(self):
        """Test model_dump_json for image reference."""
        ref = MediaRef(uri="image.png")
        json_str = ref.model_dump_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data == {"uri": "image.png", "pts_ns": None}

    def test_model_dump_json_video(self):
        """Test model_dump_json for video reference."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        json_str = ref.model_dump_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data == {"uri": "video.mp4", "pts_ns": 1_000_000_000}

    def test_model_dump_json_special_characters(self):
        """Test model_dump_json with special characters in URI."""
        ref = MediaRef(uri="images/test image (1).png")
        json_str = ref.model_dump_json()

        data = json.loads(json_str)
        assert data["uri"] == "images/test image (1).png"

    def test_model_dump_json_unicode(self):
        """Test model_dump_json with Unicode characters."""
        ref = MediaRef(uri="images/测试图片.jpg")
        json_str = ref.model_dump_json()

        data = json.loads(json_str)
        assert data["uri"] == "images/测试图片.jpg"

    def test_model_dump_json_exclude_none(self):
        """Test model_dump_json with exclude_none=True."""
        ref = MediaRef(uri="image.png")
        json_str = ref.model_dump_json(exclude_none=True)

        data = json.loads(json_str)
        assert data == {"uri": "image.png"}
        assert "pts_ns" not in data


class TestModelValidate:
    """Test model_validate method."""

    def test_model_validate_image(self):
        """Test model_validate for image reference."""
        data = {"uri": "image.png", "pts_ns": None}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "image.png"
        assert ref.pts_ns is None

    def test_model_validate_video(self):
        """Test model_validate for video reference."""
        data = {"uri": "video.mp4", "pts_ns": 1_000_000_000}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "video.mp4"
        assert ref.pts_ns == 1_000_000_000

    def test_model_validate_without_pts_ns(self):
        """Test model_validate without pts_ns field."""
        data = {"uri": "image.png"}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "image.png"
        assert ref.pts_ns is None

    def test_model_validate_with_zero_pts_ns(self):
        """Test model_validate with pts_ns=0."""
        data = {"uri": "video.mp4", "pts_ns": 0}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "video.mp4"
        assert ref.pts_ns == 0

    def test_model_validate_missing_uri(self):
        """Test model_validate with missing URI raises error."""
        data = {"pts_ns": 1_000_000_000}

        with pytest.raises(Exception):  # Pydantic ValidationError
            MediaRef.model_validate(data)

    def test_model_validate_invalid_pts_ns_type(self):
        """Test model_validate with invalid pts_ns type raises error."""
        data = {"uri": "video.mp4", "pts_ns": "invalid"}

        with pytest.raises(Exception):  # Pydantic ValidationError
            MediaRef.model_validate(data)


class TestModelValidateJson:
    """Test model_validate_json method."""

    def test_model_validate_json_image(self):
        """Test model_validate_json for image reference."""
        json_str = '{"uri": "image.png", "pts_ns": null}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "image.png"
        assert ref.pts_ns is None

    def test_model_validate_json_video(self):
        """Test model_validate_json for video reference."""
        json_str = '{"uri": "video.mp4", "pts_ns": 1000000000}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "video.mp4"
        assert ref.pts_ns == 1_000_000_000

    def test_model_validate_json_without_pts_ns(self):
        """Test model_validate_json without pts_ns field."""
        json_str = '{"uri": "image.png"}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "image.png"
        assert ref.pts_ns is None

    def test_model_validate_json_special_characters(self):
        """Test model_validate_json with special characters."""
        json_str = '{"uri": "images/test image (1).png"}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "images/test image (1).png"

    def test_model_validate_json_unicode(self):
        """Test model_validate_json with Unicode characters."""
        json_str = '{"uri": "images/测试图片.jpg"}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "images/测试图片.jpg"

    def test_model_validate_json_invalid_json(self):
        """Test model_validate_json with invalid JSON raises error."""
        json_str = '{"uri": "image.png", invalid}'

        with pytest.raises(Exception):  # JSON decode error
            MediaRef.model_validate_json(json_str)


class TestSerializationRoundtrip:
    """Test serialization round-trip (serialize → deserialize → verify)."""

    def test_roundtrip_model_dump_validate(self):
        """Test round-trip with model_dump and model_validate."""
        original = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        data = original.model_dump()
        restored = MediaRef.model_validate(data)

        assert restored == original
        assert restored.uri == original.uri
        assert restored.pts_ns == original.pts_ns

    def test_roundtrip_model_dump_json_validate_json(self):
        """Test round-trip with model_dump_json and model_validate_json."""
        original = MediaRef(uri="image.png")
        json_str = original.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored == original
        assert restored.uri == original.uri
        assert restored.pts_ns == original.pts_ns

    def test_roundtrip_data_uri(self):
        """Test round-trip with data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        original = MediaRef(uri=data_uri)
        json_str = original.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored == original
        assert restored.uri == original.uri

    def test_roundtrip_remote_url(self):
        """Test round-trip with remote URL."""
        original = MediaRef(uri="https://example.com/video.mp4", pts_ns=2_000_000_000)
        data = original.model_dump()
        restored = MediaRef.model_validate(data)

        assert restored == original
        assert restored.uri == original.uri
        assert restored.pts_ns == original.pts_ns

    def test_roundtrip_special_characters(self):
        """Test round-trip with special characters."""
        original = MediaRef(uri="images/test image (1).png")
        json_str = original.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored == original
        assert restored.uri == original.uri

    def test_roundtrip_unicode(self):
        """Test round-trip with Unicode characters."""
        original = MediaRef(uri="images/测试图片.jpg")
        json_str = original.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored == original
        assert restored.uri == original.uri


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_serialize_very_long_uri(self):
        """Test serialization with very long URI."""
        long_uri = "https://example.com/" + "a" * 10000 + ".jpg"
        ref = MediaRef(uri=long_uri)
        json_str = ref.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored.uri == long_uri

    def test_serialize_negative_pts_ns(self):
        """Test serialization with negative pts_ns."""
        ref = MediaRef(uri="video.mp4", pts_ns=-1_000_000_000)
        data = ref.model_dump()
        restored = MediaRef.model_validate(data)

        assert restored.pts_ns == -1_000_000_000

    def test_serialize_large_pts_ns(self):
        """Test serialization with very large pts_ns."""
        large_pts = 9_999_999_999_999_999_999
        ref = MediaRef(uri="video.mp4", pts_ns=large_pts)
        json_str = ref.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored.pts_ns == large_pts

    def test_serialize_empty_uri(self):
        """Test serialization with empty URI."""
        ref = MediaRef(uri="")
        data = ref.model_dump()
        restored = MediaRef.model_validate(data)

        assert restored.uri == ""

    def test_serialize_uri_with_newlines(self):
        """Test serialization with URI containing newlines (should be escaped)."""
        # Note: This is an edge case - URIs shouldn't contain newlines
        uri_with_newline = "data:text/plain;base64,SGVsbG8KV29ybGQ="
        ref = MediaRef(uri=uri_with_newline)
        json_str = ref.model_dump_json()
        restored = MediaRef.model_validate_json(json_str)

        assert restored.uri == uri_with_newline
