"""Tests for MediaRef property methods.

Tests all boolean properties: is_embedded, is_video, is_remote, is_local, is_relative_path.
"""

import os

import pytest

from mediaref import MediaRef


class TestIsEmbedded:
    """Test is_embedded property."""

    def test_is_embedded_data_uri_png(self):
        """Test that data URI with PNG is detected as embedded."""
        ref = MediaRef(uri="data:image/png;base64,iVBORw0KG...")
        assert ref.is_embedded

    def test_is_embedded_data_uri_jpeg(self):
        """Test that data URI with JPEG is detected as embedded."""
        ref = MediaRef(uri="data:image/jpeg;base64,/9j/4AAQ...")
        assert ref.is_embedded

    def test_is_embedded_data_uri_bmp(self):
        """Test that data URI with BMP is detected as embedded."""
        ref = MediaRef(uri="data:image/bmp;base64,Qk0...")
        assert ref.is_embedded

    def test_not_embedded_local_file(self):
        """Test that local file path is not embedded."""
        ref = MediaRef(uri="image.png")
        assert not ref.is_embedded

    def test_not_embedded_absolute_path(self):
        """Test that absolute path is not embedded."""
        ref = MediaRef(uri="/path/to/image.png")
        assert not ref.is_embedded

    def test_not_embedded_http_url(self):
        """Test that HTTP URL is not embedded."""
        ref = MediaRef(uri="http://example.com/image.jpg")
        assert not ref.is_embedded

    def test_not_embedded_https_url(self):
        """Test that HTTPS URL is not embedded."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        assert not ref.is_embedded

    def test_not_embedded_file_uri(self):
        """Test that file:// URI is not embedded."""
        ref = MediaRef(uri="file:///path/to/image.png")
        assert not ref.is_embedded


class TestIsVideo:
    """Test is_video property."""

    def test_is_video_with_pts_ns(self):
        """Test that MediaRef with pts_ns is detected as video."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        assert ref.is_video

    def test_is_video_with_zero_pts_ns(self):
        """Test that MediaRef with pts_ns=0 is detected as video."""
        ref = MediaRef(uri="video.mp4", pts_ns=0)
        assert ref.is_video

    def test_is_video_with_negative_pts_ns(self):
        """Test that MediaRef with negative pts_ns is detected as video."""
        ref = MediaRef(uri="video.mp4", pts_ns=-1_000_000_000)
        assert ref.is_video

    def test_not_video_without_pts_ns(self):
        """Test that MediaRef without pts_ns is not video."""
        ref = MediaRef(uri="video.mp4")
        assert not ref.is_video

    def test_not_video_image_file(self):
        """Test that image file without pts_ns is not video."""
        ref = MediaRef(uri="image.png")
        assert not ref.is_video

    def test_not_video_data_uri(self):
        """Test that data URI without pts_ns is not video."""
        ref = MediaRef(uri="data:image/png;base64,...")
        assert not ref.is_video

    def test_video_with_remote_url(self):
        """Test that remote video URL with pts_ns is detected as video."""
        ref = MediaRef(uri="https://example.com/video.mp4", pts_ns=1_000_000_000)
        assert ref.is_video


class TestIsRemote:
    """Test is_remote property."""

    def test_is_remote_http(self):
        """Test that HTTP URL is detected as remote."""
        ref = MediaRef(uri="http://example.com/image.jpg")
        assert ref.is_remote

    def test_is_remote_https(self):
        """Test that HTTPS URL is detected as remote."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        assert ref.is_remote

    def test_is_remote_http_video(self):
        """Test that HTTP video URL is detected as remote."""
        ref = MediaRef(uri="http://example.com/video.mp4", pts_ns=1_000_000_000)
        assert ref.is_remote

    def test_is_remote_https_video(self):
        """Test that HTTPS video URL is detected as remote."""
        ref = MediaRef(uri="https://example.com/video.mp4", pts_ns=1_000_000_000)
        assert ref.is_remote

    def test_not_remote_local_file(self):
        """Test that local file is not remote."""
        ref = MediaRef(uri="image.png")
        assert not ref.is_remote

    def test_not_remote_absolute_path(self):
        """Test that absolute path is not remote."""
        ref = MediaRef(uri="/path/to/image.png")
        assert not ref.is_remote

    def test_not_remote_data_uri(self):
        """Test that data URI is not remote."""
        ref = MediaRef(uri="data:image/png;base64,...")
        assert not ref.is_remote

    def test_not_remote_file_uri(self):
        """Test that file:// URI is not remote."""
        ref = MediaRef(uri="file:///path/to/image.png")
        assert not ref.is_remote


class TestIsRelativePath:
    """Test is_relative_path property."""

    def test_is_relative_path_simple(self):
        """Test that simple relative path is detected."""
        ref = MediaRef(uri="image.png")
        assert ref.is_relative_path

    def test_is_relative_path_nested(self):
        """Test that nested relative path is detected."""
        ref = MediaRef(uri="images/test.jpg")
        assert ref.is_relative_path

    def test_is_relative_path_deep_nested(self):
        """Test that deeply nested relative path is detected."""
        ref = MediaRef(uri="data/recordings/images/frame.png")
        assert ref.is_relative_path

    def test_is_relative_path_with_dots(self):
        """Test that relative path with .. is detected."""
        ref = MediaRef(uri="../images/test.jpg")
        assert ref.is_relative_path

    def test_is_relative_path_current_dir(self):
        """Test that ./ relative path is detected."""
        ref = MediaRef(uri="./images/test.jpg")
        assert ref.is_relative_path

    @pytest.mark.skipif(os.name == "nt", reason="POSIX absolute paths are relative on Windows")
    def test_not_relative_path_absolute_posix(self):
        """Test that absolute POSIX path is not relative (POSIX only)."""
        ref = MediaRef(uri="/absolute/path/image.png")
        assert not ref.is_relative_path

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_not_relative_path_absolute_windows(self):
        """Test that absolute Windows path is not relative (Windows only)."""
        ref = MediaRef(uri="C:/absolute/path/image.png")
        assert not ref.is_relative_path

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_posix_path_is_relative_on_windows(self):
        """Test that POSIX absolute path is relative on Windows (no drive letter)."""
        ref = MediaRef(uri="/absolute/path/image.png")
        assert ref.is_relative_path

    def test_not_relative_path_http(self):
        """Test that HTTP URL is not relative path."""
        ref = MediaRef(uri="http://example.com/image.jpg")
        assert not ref.is_relative_path

    def test_not_relative_path_https(self):
        """Test that HTTPS URL is not relative path."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        assert not ref.is_relative_path

    def test_not_relative_path_data_uri(self):
        """Test that data URI is not relative path."""
        ref = MediaRef(uri="data:image/png;base64,...")
        assert not ref.is_relative_path

    def test_not_relative_path_file_uri(self):
        """Test that file:// URI is not relative path."""
        ref = MediaRef(uri="file:///path/to/image.png")
        assert not ref.is_relative_path


class TestPropertyCombinations:
    """Test combinations of properties."""

    def test_embedded_not_remote(self):
        """Test that embedded URI is not remote."""
        ref = MediaRef(uri="data:image/png;base64,...")
        assert ref.is_embedded
        assert not ref.is_remote
        assert not ref.is_relative_path

    def test_local_file_not_embedded_not_remote(self):
        """Test that local file is not embedded or remote."""
        ref = MediaRef(uri="image.png")
        assert not ref.is_embedded
        assert not ref.is_remote
        assert ref.is_relative_path

    def test_remote_not_embedded(self):
        """Test that remote URL is not embedded."""
        ref = MediaRef(uri="https://example.com/image.jpg")
        assert ref.is_remote
        assert not ref.is_embedded
        assert not ref.is_relative_path

    def test_video_can_be_local_file(self):
        """Test that video can be a local file."""
        ref = MediaRef(uri="video.mp4", pts_ns=1_000_000_000)
        assert ref.is_video
        assert not ref.is_remote
        assert not ref.is_embedded

    def test_video_can_be_remote(self):
        """Test that video can be remote."""
        ref = MediaRef(uri="https://example.com/video.mp4", pts_ns=1_000_000_000)
        assert ref.is_video
        assert ref.is_remote
        assert not ref.is_embedded

    @pytest.mark.skipif(os.name == "nt", reason="POSIX absolute paths are relative on Windows")
    def test_absolute_path_not_relative(self):
        """Test that absolute path is not relative (POSIX only)."""
        ref = MediaRef(uri="/absolute/path/image.png")
        assert not ref.is_relative_path
        assert not ref.is_remote
        assert not ref.is_embedded

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_absolute_path_not_relative(self):
        """Test that Windows absolute path is not relative (Windows only)."""
        ref = MediaRef(uri="C:/absolute/path/image.png")
        assert not ref.is_relative_path
        assert not ref.is_remote
        assert not ref.is_embedded
