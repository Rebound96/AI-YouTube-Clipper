"""
Utility functions for AI-YouTube-Clipper.
Includes time formatting, filename sanitization, JSON I/O, and subprocess wrappers.
"""

import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AI-YouTube-Clipper")


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """Sanitize a string to be safely used as a filename across operating systems and CLI tools."""
    # Remove quotes, apostrophes, exclamation marks, and all non-alphanumeric punctuation
    cleaned = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    # Replace whitespace sequences with single underscores
    cleaned = re.sub(r"[\s-]+", "_", cleaned).strip(" ._-")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_length]


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS subtitle timestamp format: H:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:
        secs += 1
        centis = 0
        if secs == 60:
            minutes += 1
            secs = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
        if secs == 60:
            minutes += 1
            secs = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_display_time(seconds: float) -> str:
    """Format seconds into a human-readable string: MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save serializable data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load JSON data from a file, returning None if the file doesn't exist."""
    path = Path(file_path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ffmpeg(
    cmd_args: List[str],
    executable: str = "ffmpeg",
    description: str = "FFmpeg command"
) -> subprocess.CompletedProcess:
    """
    Execute an FFmpeg or FFprobe command safely and capture output.
    Raises RuntimeError if execution fails.
    """
    full_cmd = [executable] + cmd_args
    logger.info("Executing %s: %s", description, " ".join(full_cmd[:8]) + "...")

    try:
        process = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False
        )
    except FileNotFoundError as err:
        logger.error("Executable not found: %s", executable)
        raise RuntimeError(f"Executable not found: {executable}. Ensure FFmpeg is installed.") from err

    if process.returncode != 0:
        error_msg = f"{description} failed with return code {process.returncode}.\nSTDERR:\n{process.stderr[-1000:]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return process
