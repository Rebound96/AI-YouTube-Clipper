"""
Unit tests for Hormozi-style animated ASS subtitles.
"""

from pathlib import Path
import pytest

from src.config import SubtitleConfig
from src.subtitle_generator import SubtitleGenerator
from src.transcriber import WordTiming


def test_generate_ass_subtitles(tmp_path: Path):
    """Verify ASS file generation, Hormozi styling, and active word highlighting tags."""
    config = SubtitleConfig(
        font_name="Impact",
        font_size=50,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000FFFF",
        max_words_per_line=3,
        uppercase=True
    )
    generator = SubtitleGenerator(config)

    words = [
        WordTiming(word="This", start=10.0, end=10.3),
        WordTiming(word="is", start=10.3, end=10.5),
        WordTiming(word="actually", start=10.5, end=11.0),
        WordTiming(word="insane", start=11.1, end=11.6),
    ]

    output_ass = tmp_path / "test_sub.ass"
    # Clip starts at 10.0s, so word timestamps should be offset by 10.0s
    generator.generate_ass(
        words=words,
        output_ass_path=output_ass,
        clip_start_time=10.0,
        video_width=1080,
        video_height=1920
    )

    assert output_ass.exists()
    content = output_ass.read_text(encoding="utf-8")

    # Check ASS Headers
    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Style: Hormozi,Impact,50,&H00FFFFFF" in content

    # Check Dialogue Events
    assert "Dialogue: 0," in content
    # Words should be uppercased
    assert "THIS" in content
    assert "ACTUALLY" in content
    assert "INSANE" in content

    # Check active word color highlighting tag
    assert r"\c&H0000FFFF&" in content  # Yellow highlight
    assert r"\c&H00FFFFFF&" in content  # White text


def test_generate_ass_empty_words(tmp_path: Path):
    """Verify generator handles empty word list without crashing."""
    generator = SubtitleGenerator()
    output_ass = tmp_path / "empty.ass"
    generator.generate_ass([], output_ass)

    assert output_ass.exists()
    content = output_ass.read_text(encoding="utf-8")
    assert "[Script Info]" in content
