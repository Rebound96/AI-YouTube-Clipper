"""
Unit tests for configuration and utility functions.
"""

from pathlib import Path
import pytest

from src.config import FFMPEG_PATH, FFPROBE_PATH, PROJECT_ROOT, ensure_directories
from src.utils import (
    format_ass_timestamp,
    format_srt_timestamp,
    load_json,
    sanitize_filename,
    save_json,
    seconds_to_display_time,
)


def test_ensure_directories():
    """Verify directories are created."""
    ensure_directories()
    assert (PROJECT_ROOT / "data").exists()
    assert (PROJECT_ROOT / "data" / "downloads").exists()
    assert (PROJECT_ROOT / "data" / "audio").exists()
    assert (PROJECT_ROOT / "data" / "transcripts").exists()
    assert (PROJECT_ROOT / "data" / "clips").exists()


def test_ffmpeg_detected():
    """Verify ffmpeg and ffprobe paths are resolved."""
    assert FFMPEG_PATH is not None
    assert FFPROBE_PATH is not None
    assert Path(FFMPEG_PATH).exists()
    assert Path(FFPROBE_PATH).exists()


def test_sanitize_filename():
    """Verify filename sanitization works properly on Windows."""
    assert sanitize_filename("My: Cool / Video * Title? <3>") == "My_Cool_Video_Title_3"
    assert sanitize_filename("   Spaces   Everywhere   ") == "Spaces_Everywhere"
    assert sanitize_filename("") == "untitled"


def test_format_ass_timestamp():
    """Verify ASS timestamp formatting."""
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(65.45) == "0:01:05.45"
    assert format_ass_timestamp(3661.05) == "1:01:01.05"


def test_format_srt_timestamp():
    """Verify SRT timestamp formatting."""
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(65.456) == "00:01:05,456"
    assert format_srt_timestamp(3661.123) == "01:01:01,123"


def test_seconds_to_display_time():
    """Verify display time formatting."""
    assert seconds_to_display_time(45.0) == "00:45"
    assert seconds_to_display_time(125.0) == "02:05"
    assert seconds_to_display_time(3665.0) == "1:01:05"


def test_json_io(tmp_path: Path):
    """Verify JSON save and load."""
    sample_file = tmp_path / "sample.json"
    data = {"name": "test_video", "score": 9.5, "tags": ["viral", "hook"]}
    save_json(data, sample_file)

    loaded = load_json(sample_file)
    assert loaded == data

    # Non-existent file returns None
    assert load_json(tmp_path / "does_not_exist.json") is None
