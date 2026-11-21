"""
Live end-to-end transcription test using faster-whisper tiny model.
"""

from pathlib import Path
import pytest

from src.audio import extract_audio
from src.config import WhisperConfig
from src.transcriber import Transcriber, TranscriptionResult
from tests.test_audio import sample_video


def test_live_whisper_transcription(sample_video: Path, tmp_path: Path):
    """
    Test that faster-whisper model loads and transcribes audio into structured segments.
    """
    audio_path = tmp_path / "test_speech.wav"
    extract_audio(sample_video, output_audio_path=audio_path)
    assert audio_path.exists()

    config = WhisperConfig(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        word_timestamps=True,
    )
    transcriber = Transcriber(config=config)
    output_json = tmp_path / "transcription.json"

    result = transcriber.transcribe(audio_path, output_json_path=output_json, cache=False)

    assert isinstance(result, TranscriptionResult)
    assert output_json.exists()
    assert result.duration >= 2.0
