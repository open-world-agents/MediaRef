"""Tests for core MediaRef functionality without video dependencies.

These tests ensure MediaRef works correctly even when video dependencies
are not installed (core-only installation).
"""

import pytest

from mediaref import MediaRef


class TestMediaRefCreation:
    """Test MediaRef instance creation."""

    def test_create_with_uri_only(self):
        """Test creating MediaRef with URI only."""
        ref = MediaRef(uri="image.png")

        assert ref.uri == "image.png"
        assert ref.pts_ns is None

    def test_create_with_uri_and_pts_ns(self):
        """Test creating MediaRef with URI and pts_ns."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)

        assert ref.uri == "video.mp4"
        assert ref.pts_ns == 1_000_000_000

    def test_create_with_dict(self):
        """Test creating MediaRef from dict using model_validate."""
        data = {"uri": "image.png", "pts_ns": 123456}
        ref = MediaRef.model_validate(data)

        assert ref.uri == "image.png"
        assert ref.pts_ns == 123456

    def test_create_with_json(self):
        """Test creating MediaRef from JSON string."""
        json_str = '{"uri": "video.mp4", "pts_ns": 1000000000}'
        ref = MediaRef.model_validate_json(json_str)

        assert ref.uri == "video.mp4"
        assert ref.pts_ns == 1_000_000_000

    def test_create_with_data_uri(self):
        """Test creating MediaRef with data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ref = MediaRef(uri=data_uri)

        assert ref.uri == data_uri
        assert ref.pts_ns is None

    def test_create_with_http_url(self):
        """Test creating MediaRef with HTTP URL."""
        ref = MediaRef(uri="http://example.com/image.jpg")

        assert ref.uri == "http://example.com/image.jpg"
        assert ref.pts_ns is None

    def test_create_with_https_url(self):
        """Test creating MediaRef with HTTPS URL."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        assert ref.uri == "https://example.com/image.jpg"
        assert ref.pts_ns is None

    def test_create_with_file_uri(self):
        """Test creating MediaRef with file:// URI."""
        ref = MediaRef(uri="file:///path/to/image.png")

        assert ref.uri == "file:///path/to/image.png"
        assert ref.pts_ns is None

    def test_create_with_relative_path(self):
        """Test creating MediaRef with relative path."""
        ref = MediaRef(uri="relative/path/image.png")

        assert ref.uri == "relative/path/image.png"
        assert ref.pts_ns is None

    def test_create_with_absolute_path(self):
        """Test creating MediaRef with absolute path."""
        ref = MediaRef(uri="/absolute/path/image.png")

        assert ref.uri == "/absolute/path/image.png"
        assert ref.pts_ns is None


class TestMediaRefEquality:
    """Test MediaRef equality and hashing."""

    def test_equality_same_uri(self):
        """Test that MediaRefs with same URI are equal."""
        ref1 = MediaRef(uri="image.png")
        ref2 = MediaRef(uri="image.png")

        assert ref1 == ref2

    def test_equality_same_uri_and_pts_ns(self):
        """Test that MediaRefs with same URI and pts_ns are equal."""
        ref1 = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        ref2 = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)

        assert ref1 == ref2

    def test_inequality_different_uri(self):
        """Test that MediaRefs with different URIs are not equal."""
        ref1 = MediaRef(uri="image1.png")
        ref2 = MediaRef(uri="image2.png")

        assert ref1 != ref2

    def test_inequality_different_pts_ns(self):
        """Test that MediaRefs with different pts_ns are not equal."""
        ref1 = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        ref2 = MediaRef(uri="video.mp4", pts_ns=2_000_000_000)

        assert ref1 != ref2

    def test_inequality_one_with_pts_ns(self):
        """Test that MediaRefs with/without pts_ns are not equal."""
        ref1 = MediaRef(uri="video.mp4")
        ref2 = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)

        assert ref1 != ref2


class TestMediaRefStringRepresentation:
    """Test MediaRef string representation."""

    def test_repr_image(self):
        """Test __repr__ for image reference."""
        ref = MediaRef(uri="image.png")
        repr_str = repr(ref)

        assert "MediaRef" in repr_str
        assert "image.png" in repr_str

    def test_repr_video(self):
        """Test __repr__ for video reference."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        repr_str = repr(ref)

        assert "MediaRef" in repr_str
        assert "video.mp4" in repr_str
        assert "1000000000" in repr_str

    def test_str_image(self):
        """Test __str__ for image reference."""
        ref = MediaRef(uri="image.png")
        str_repr = str(ref)

        assert "image.png" in str_repr


class TestMediaRefCopy:
    """Test MediaRef copying behavior."""

    def test_model_copy(self):
        """Test that model_copy creates a new instance."""
        ref1 = MediaRef(uri="image.png", pts_ns=123456)
        ref2 = ref1.model_copy()

        assert ref1 == ref2
        assert ref1 is not ref2

    def test_model_copy_deep(self):
        """Test that model_copy(deep=True) creates a deep copy."""
        ref1 = MediaRef(uri="image.png", pts_ns=123456)
        ref2 = ref1.model_copy(deep=True)

        assert ref1 == ref2
        assert ref1 is not ref2

    def test_model_copy_with_update(self):
        """Test model_copy with update parameter."""
        ref1 = MediaRef(uri="image.png", pts_ns=123456)
        ref2 = ref1.model_copy(update={"pts_ns": 789012})

        assert ref2.uri == "image.png"
        assert ref2.pts_ns == 789012
        assert ref1.pts_ns == 123456  # Original unchanged


class TestMediaRefValidation:
    """Test Pydantic validation."""

    def test_validation_missing_uri(self):
        """Test that missing URI raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            MediaRef()  # type: ignore

    def test_validation_invalid_pts_ns_type(self):
        """Test that invalid pts_ns type raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            MediaRef(uri="video.mp4", pts_ns="invalid")  # type: ignore

    def test_validation_negative_pts_ns(self):
        """Test that negative pts_ns is allowed (seeking backwards)."""
        ref = MediaRef(uri="video.mp4", pts_ns=-1_000_000_000)
        assert ref.pts_ns == -1_000_000_000

    def test_validation_zero_pts_ns(self):
        """Test that zero pts_ns is allowed."""
        ref = MediaRef(uri="video.mp4", pts_ns=0)
        assert ref.pts_ns == 0


class TestMediaRefImmutability:
    """Test that MediaRef fields can be accessed but model_copy should be used for changes."""

    def test_uri_is_accessible(self):
        """Test that URI can be accessed."""
        ref = MediaRef(uri="image.png")
        assert ref.uri == "image.png"

    def test_pts_ns_is_accessible(self):
        """Test that pts_ns can be accessed."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        assert ref.pts_ns == 1_000_000_000

    def test_use_model_copy_for_changes(self):
        """Test that model_copy should be used to create modified versions."""
        ref = MediaRef(uri="image.png")
        modified = ref.model_copy(update={"uri": "new_image.png"})

        assert ref.uri == "image.png"  # Original unchanged
        assert modified.uri == "new_image.png"  # New instance has new value
