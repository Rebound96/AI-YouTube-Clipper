"""
Tests for audio extraction and media probing.
"""

from pathlib import Path
import pytest

from src.audio import extract_audio, get_media_duration, probe_media
from src.config import FFMPEG_PATH
from src.utils import run_ffmpeg


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """Generate a clean 3-second 720p MP4 test video with audio using FFmpeg."""
    temp_dir = tmp_path_factory.mktemp("media")
    video_path = temp_dir / "sample_test_video.mp4"

    cmd = [
        "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(video_path)
    ]
    run_ffmpeg(cmd, executable=FFMPEG_PATH, description="Create test video")
    assert video_path.exists()
    return video_path


def test_probe_media(sample_video: Path):
    """Verify probe_media returns audio/video stream data and duration."""
    info = probe_media(sample_video)
    assert "streams" in info
    assert "format" in info

    streams = info["streams"]
    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]

    assert len(video_streams) >= 1
    assert video_streams[0]["width"] == 1280
    assert video_streams[0]["height"] == 720

    assert len(audio_streams) >= 1

    duration = get_media_duration(sample_video)
    assert 2.8 <= duration <= 3.2


def test_extract_audio(sample_video: Path, tmp_path: Path):
    """Verify audio extraction generates a 16kHz mono WAV file."""
    output_wav = tmp_path / "extracted.wav"
    result_path = extract_audio(sample_video, output_audio_path=output_wav)

    assert result_path == output_wav
    assert output_wav.exists()
    assert output_wav.stat().st_size > 0

    # Probe extracted audio
    audio_info = probe_media(output_wav)
    audio_stream = audio_info["streams"][0]
    assert audio_stream["codec_name"] == "pcm_s16le"
    assert audio_stream["sample_rate"] == "16000"
    assert audio_stream["channels"] == 1
