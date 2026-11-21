"""
Tests for video downloader and local file ingestion.
"""

from pathlib import Path
import pytest

from src.downloader import VideoDownloader, VideoInfo
from tests.test_audio import sample_video  # re-use sample_video fixture


def test_ingest_local_file(sample_video: Path, tmp_path: Path):
    """Verify local video file ingestion and metadata extraction."""
    downloader = VideoDownloader(output_dir=tmp_path)
    info = downloader.ingest_local_file(sample_video, copy_to_downloads=True)

    assert isinstance(info, VideoInfo)
    assert info.title == "sample_test_video"
    assert info.is_local is True
    assert 2.8 <= info.duration <= 3.2
    assert info.width == 1280
    assert info.height == 720
    assert info.fps == 30.0
    assert info.file_path.exists()


def test_invalid_local_file(tmp_path: Path):
    """Verify non-existent local file raises FileNotFoundError."""
    downloader = VideoDownloader(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        downloader.ingest_local_file(tmp_path / "non_existent.mp4")


def test_invalid_youtube_url(tmp_path: Path):
    """Verify bad YouTube URL raises descriptive error."""
    downloader = VideoDownloader(output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        downloader.extract_info("https://www.youtube.com/watch?v=invalid_id_9999999999")
