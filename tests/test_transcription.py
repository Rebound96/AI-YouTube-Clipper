"""
Tests for transcription data models and faster-whisper transcription.
"""

from pathlib import Path
import pytest

from src.config import WhisperConfig
from src.transcriber import (
    Transcriber,
    TranscriptionResult,
    TranscriptSegment,
    WordTiming,
)


def test_transcription_models_roundtrip():
    """Verify serialization, deserialization and word querying."""
    words = [
        WordTiming(word="Hello", start=0.0, end=0.4, probability=0.98),
        WordTiming(word="world", start=0.5, end=0.9, probability=0.95),
    ]
    segment = TranscriptSegment(
        id=1,
        start=0.0,
        end=1.0,
        text="Hello world",
        words=words,
        avg_logprob=-0.2,
        no_speech_prob=0.01,
    )
    result = TranscriptionResult(
        segments=[segment],
        language="en",
        language_probability=0.99,
        duration=1.0,
        full_text="Hello world"
    )

    data = result.to_dict()
    restored = TranscriptionResult.from_dict(data)

    assert restored.language == "en"
    assert restored.duration == 1.0
    assert len(restored.segments) == 1
    assert restored.segments[0].text == "Hello world"
    assert len(restored.segments[0].words) == 2
    assert restored.segments[0].words[1].word == "world"

    # Test SRT generation
    srt = result.to_srt()
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "Hello world" in srt

    # Test get_words_in_range
    matched = result.get_words_in_range(0.0, 0.5)
    assert len(matched) >= 1
    assert matched[0].word == "Hello"


def test_transcription_cache(tmp_path: Path):
    """Verify transcription caching mechanism."""
    cache_file = tmp_path / "cached_transcript.json"
    result = TranscriptionResult(
        segments=[TranscriptSegment(id=1, start=0.0, end=2.0, text="Cached test", words=[])],
        language="en",
        language_probability=1.0,
        duration=2.0,
        full_text="Cached test"
    )
    from src.utils import save_json
    save_json(result.to_dict(), cache_file)

    config = WhisperConfig(model_size="tiny", device="cpu", compute_type="int8")
    transcriber = Transcriber(config=config)

    # Calling transcribe with existing cache should return cached result without loading model
    loaded = transcriber.transcribe(audio_path=tmp_path / "dummy.wav", output_json_path=cache_file, cache=True)
    assert loaded.full_text == "Cached test"
    assert len(loaded.segments) == 1
