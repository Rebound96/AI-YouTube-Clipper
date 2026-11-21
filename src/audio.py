"""
Audio extraction and media probing module using FFmpeg and FFprobe.
Optimized for feeding speech data into Whisper / faster-whisper.
"""

import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional

from src.config import AUDIO_DIR, FFMPEG_PATH, FFPROBE_PATH
from src.utils import run_ffmpeg, sanitize_filename

logger = logging.getLogger("AI-YouTube-Clipper.audio")


def probe_media(file_path: Path) -> Dict[str, Any]:
    """
    Run ffprobe on a media file to retrieve streams and format information as JSON.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {path}")

    args = [
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]

    proc = run_ffmpeg(args, executable=FFPROBE_PATH, description=f"ffprobe {path.name}")
    try:
        data = json.loads(proc.stdout)
        return data
    except json.JSONDecodeError as err:
        logger.error("Failed to parse ffprobe JSON output for %s: %s", path, err)
        raise RuntimeError(f"Failed to parse ffprobe metadata: {err}") from err


def get_media_duration(file_path: Path) -> float:
    """Extract media duration in seconds."""
    data = probe_media(file_path)
    try:
        return float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


def extract_audio(
    video_path: Path,
    output_audio_path: Optional[Path] = None,
    sample_rate: int = 16000,
    overwrite: bool = False
) -> Path:
    """
    Extract audio from a video file and save as 16kHz mono WAV for optimal speech recognition.

    :param video_path: Source video file path.
    :param output_audio_path: Destination audio file path (defaults to data/audio/<stem>.wav).
    :param sample_rate: Audio sample rate in Hz (default 16000).
    :param overwrite: Whether to overwrite existing extracted audio file.
    :return: Path to the generated WAV audio file.
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_audio_path is None:
        safe_stem = sanitize_filename(video_path.stem)
        output_audio_path = AUDIO_DIR / f"{safe_stem}_16k.wav"

    output_audio_path = Path(output_audio_path).resolve()
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    if output_audio_path.exists() and not overwrite:
        logger.info("Using cached audio file: %s", output_audio_path)
        return output_audio_path

    logger.info("Extracting 16kHz mono audio from %s to %s", video_path.name, output_audio_path.name)

    # -y: overwrite
    # -vn: disable video recording
    # -acodec pcm_s16le: standard uncompressed 16-bit PCM
    # -ar: sample rate
    # -ac: number of audio channels (1 = mono)
    args = [
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_audio_path)
    ]

    run_ffmpeg(args, executable=FFMPEG_PATH, description=f"Extract audio ({video_path.name})")

    if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction failed; output file is missing or empty: {output_audio_path}")

    return output_audio_path
