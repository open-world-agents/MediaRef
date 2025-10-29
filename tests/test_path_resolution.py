"""Tests for MediaRef path resolution and validation."""

import os
from pathlib import Path

import pytest

from mediaref import MediaRef


class TestResolveRelativePath:
    """Test resolve_relative_path method."""

    def test_resolve_relative_path_with_mcap_file(self):
        """Test resolving relative path against MCAP file."""
        ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
        resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "/data/relative/video.mkv"
        assert resolved.pts_ns == 123456

    def test_resolve_relative_path_with_directory(self):
        """Test resolving relative path against directory."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path("/data/dataset")

        assert resolved.uri == "/data/dataset/images/test.jpg"
        assert resolved.pts_ns is None

    def test_resolve_relative_path_with_trailing_slash(self):
        """Test resolving relative path with trailing slash in base."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path("/data/dataset/")

        assert resolved.uri == "/data/dataset/images/test.jpg"

    def test_resolve_nested_relative_path(self):
        """Test resolving deeply nested relative path."""
        ref = MediaRef(uri="data/recordings/videos/clip.mp4", pts_ns=1_000_000_000)
        resolved = ref.resolve_relative_path("/base/path.mcap")

        assert resolved.uri == "/base/data/recordings/videos/clip.mp4"
        assert resolved.pts_ns == 1_000_000_000

    @pytest.mark.skipif(os.name == "nt", reason="POSIX absolute paths are relative on Windows")
    def test_resolve_absolute_path_unchanged(self):
        """Test that absolute paths remain unchanged (POSIX only)."""
        ref = MediaRef(uri="/absolute/path/image.png")
        resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "/absolute/path/image.png"
        assert resolved is ref  # Should return same instance

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_resolve_windows_absolute_path_unchanged(self):
        """Test that Windows absolute paths remain unchanged (Windows only)."""
        ref = MediaRef(uri="C:/absolute/path/image.png")
        resolved = ref.resolve_relative_path("D:/data/recording.mcap")

        assert resolved.uri == "C:/absolute/path/image.png"
        assert resolved is ref  # Should return same instance

    def test_resolve_returns_new_instance(self):
        """Test that resolve_relative_path returns new instance for relative paths."""
        ref = MediaRef(uri="relative/path.jpg")
        resolved = ref.resolve_relative_path("/base/path.mcap")

        assert resolved is not ref
        assert ref.uri == "relative/path.jpg"  # Original unchanged
        assert resolved.uri == "/base/relative/path.jpg"

    def test_resolve_with_mcap_uses_parent_directory(self):
        """Test that .mcap files use parent directory as base."""
        ref = MediaRef(uri="videos/clip.mp4", pts_ns=0)
        resolved = ref.resolve_relative_path("/data/recordings/session.mcap")

        # Should use /data/recordings/ as base, not /data/recordings/session.mcap
        assert resolved.uri == "/data/recordings/videos/clip.mp4"

    def test_resolve_with_non_mcap_file_uses_path_directly(self):
        """Test that non-MCAP files use the path directly as base."""
        ref = MediaRef(uri="images/frame.png")
        resolved = ref.resolve_relative_path("/data/dataset")

        # Should use /data/dataset directly as base (not parent)
        assert resolved.uri == "/data/dataset/images/frame.png"


class TestResolveRelativePathWarnings:
    """Test handling of unresolvable paths."""

    def test_resolve_remote_path_warns(self):
        """Test that resolving remote path generates warning by default."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        with pytest.warns(UserWarning, match="Cannot resolve unresolvable URI"):
            resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "https://example.com/image.jpg"
        assert resolved is ref

    def test_resolve_embedded_path_warns(self):
        """Test that resolving embedded path generates warning by default."""
        ref = MediaRef(uri="data:image/png;base64,...")

        with pytest.warns(UserWarning, match="Cannot resolve unresolvable URI"):
            resolved = ref.resolve_relative_path("/data/recording.mcap")

        assert resolved.uri == "data:image/png;base64,..."
        assert resolved is ref

    def test_resolve_remote_with_ignore(self):
        """Test that on_unresolvable='ignore' suppresses warning for remote paths."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        # Should not warn when on_unresolvable="ignore"
        resolved = ref.resolve_relative_path("/data/recording.mcap", on_unresolvable="ignore")

        assert resolved.uri == "https://example.com/image.jpg"
        assert resolved is ref

    def test_resolve_embedded_with_ignore(self):
        """Test that on_unresolvable='ignore' suppresses warning for embedded paths."""
        ref = MediaRef(uri="data:image/png;base64,...")

        # Should not warn when on_unresolvable="ignore"
        resolved = ref.resolve_relative_path("/data/recording.mcap", on_unresolvable="ignore")

        assert resolved.uri == "data:image/png;base64,..."
        assert resolved is ref

    def test_resolve_remote_with_error(self):
        """Test that on_unresolvable='error' raises ValueError for remote paths."""
        ref = MediaRef(uri="https://example.com/image.jpg")

        with pytest.raises(ValueError, match="Cannot resolve unresolvable URI"):
            ref.resolve_relative_path("/data/recording.mcap", on_unresolvable="error")

    def test_resolve_embedded_with_error(self):
        """Test that on_unresolvable='error' raises ValueError for embedded paths."""
        ref = MediaRef(uri="data:image/png;base64,...")

        with pytest.raises(ValueError, match="Cannot resolve unresolvable URI"):
            ref.resolve_relative_path("/data/recording.mcap", on_unresolvable="error")


class TestResolveRelativePathCrossPlatform:
    """Test cross-platform path resolution."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-specific test")
    def test_resolve_posix_path(self):
        """Test resolving POSIX paths."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path("/data/recordings/session.mcap")

        assert resolved.uri == "/data/recordings/images/test.jpg"

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_resolve_windows_path(self):
        """Test resolving Windows paths."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path("C:/data/recordings/session.mcap")

        # Should use forward slashes (as_posix())
        assert resolved.uri == "C:/data/recordings/images/test.jpg"

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_resolve_windows_backslash_path(self):
        """Test resolving Windows paths with backslashes."""
        ref = MediaRef(uri="images/test.jpg")
        resolved = ref.resolve_relative_path(r"C:\data\recordings\session.mcap")

        # Should convert to forward slashes
        assert resolved.uri == "C:/data/recordings/images/test.jpg"

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_absolute_path_unchanged(self):
        """Test that Windows absolute paths remain unchanged."""
        ref = MediaRef(uri="C:/absolute/path/image.png")
        resolved = ref.resolve_relative_path("D:/data/recording.mcap")

        assert resolved.uri == "C:/absolute/path/image.png"
        assert resolved is ref


class TestValidateUri:
    """Test validate_uri method."""

    def test_validate_embedded_uri(self):
        """Test that embedded URI is always valid."""
        ref = MediaRef(uri="data:image/png;base64,iVBORw0KG...")
        assert ref.validate_uri()

    def test_validate_local_file_existence(self, tmp_path: Path):
        """Test validating local file and directory existence."""
        # Existing file should be valid
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        ref_file = MediaRef(uri=str(test_file))
        assert ref_file.validate_uri()

        # Non-existent file should be invalid
        ref_nonexistent = MediaRef(uri=str(tmp_path / "nonexistent.txt"))
        assert not ref_nonexistent.validate_uri()

        # Existing directory should be valid
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        ref_dir = MediaRef(uri=str(test_dir))
        assert ref_dir.validate_uri()

    def test_validate_remote_uri_not_implemented(self):
        """Test that remote URI validation raises NotImplementedError."""
        ref_https = MediaRef(uri="https://example.com/image.jpg")
        with pytest.raises(NotImplementedError):
            ref_https.validate_uri()

        ref_http = MediaRef(uri="http://example.com/image.jpg")
        with pytest.raises(NotImplementedError):
            ref_http.validate_uri()


class TestPathEdgeCases:
    """Test edge cases in path handling."""

    def test_resolve_empty_relative_path(self):
        """Test resolving empty relative path."""
        ref = MediaRef(uri="")
        resolved = ref.resolve_relative_path("/base/path.mcap")

        # Empty string is relative, so should be resolved
        assert resolved.uri == "/base"

    def test_resolve_dot_relative_path(self):
        """Test resolving . (current directory) relative path."""
        ref = MediaRef(uri=".")
        resolved = ref.resolve_relative_path("/base/path.mcap")

        assert "/base" in resolved.uri

    def test_resolve_dotdot_relative_path(self):
        """Test resolving .. (parent directory) relative path."""
        ref = MediaRef(uri="../images/test.jpg")
        resolved = ref.resolve_relative_path("/data/recordings/session.mcap")

        # Should resolve .. correctly
        assert "images/test.jpg" in resolved.uri

    def test_resolve_with_special_characters(self):
        """Test resolving paths with special characters."""
        ref = MediaRef(uri="images/test image (1).jpg")
        resolved = ref.resolve_relative_path("/data/recordings/session.mcap")

        assert resolved.uri == "/data/recordings/images/test image (1).jpg"

    def test_resolve_with_unicode_characters(self):
        """Test resolving paths with Unicode characters."""
        ref = MediaRef(uri="images/测试图片.jpg")
        resolved = ref.resolve_relative_path("/data/recordings/session.mcap")

        assert "测试图片.jpg" in resolved.uri

    def test_validate_file_with_special_characters(self, tmp_path: Path):
        """Test validating file with special characters."""
        test_file = tmp_path / "test image (1).png"
        test_file.write_text("test")

        ref = MediaRef(uri=str(test_file))
        assert ref.validate_uri()

    def test_resolve_preserves_pts_ns(self):
        """Test that path resolution preserves pts_ns."""
        ref = MediaRef(uri="videos/clip.mp4", pts_ns=123456789)
        resolved = ref.resolve_relative_path("/base/path.mcap")

        assert resolved.pts_ns == 123456789

    def test_resolve_with_file_uri_scheme(self):
        """Test that file:// URIs are not resolved (already absolute)."""
        ref = MediaRef(uri="file:///path/to/image.png")
        resolved = ref.resolve_relative_path("/base/path.mcap")

        assert resolved.uri == "file:///path/to/image.png"
        assert resolved is ref
