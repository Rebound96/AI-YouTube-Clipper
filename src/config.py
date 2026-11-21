"""
Configuration module for AI-YouTube-Clipper.
Centralizes paths, models, processing constants, and system environment checks.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import Optional


# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage Directories
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CLIPS_DIR = DATA_DIR / "clips"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_CLIPS_DIR = OUTPUT_DIR / "clips"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_directories():
    """Ensure all required project data and output directories exist."""
    for directory in [
        DATA_DIR, DOWNLOADS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, CLIPS_DIR, TEMP_DIR,
        OUTPUT_DIR, OUTPUT_CLIPS_DIR, LOGS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def find_binary(binary_name: str) -> Optional[str]:
    """Find the path of a binary executable in PATH or common Windows locations."""
    found = shutil.which(binary_name)
    if found:
        return found

    # Check common Windows winget / local paths
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_pattern = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pattern.exists():
            matches = list(winget_pattern.glob(f"**/{binary_name}.exe"))
            if matches:
                return str(matches[0])

    return None


FFMPEG_PATH = find_binary("ffmpeg") or "ffmpeg"
FFPROBE_PATH = find_binary("ffprobe") or "ffprobe"


@dataclass
class WhisperConfig:
    """Configuration for faster-whisper transcription."""
    model_size: str = "small"  # "tiny", "base", "small", "medium", "large-v3"
    device: str = "auto"       # "auto", "cpu", "cuda"
    compute_type: str = "auto" # "auto", "int8", "float16", "float32"
    beam_size: int = 5
    language: Optional[str] = None  # None for auto-detect
    word_timestamps: bool = True
    vad_filter: bool = True


@dataclass
class GamingLayoutConfig:
    """Configuration for Gaming Split-Screen layouts."""
    layout_type: str = "streamer_top_gameplay_bottom"  # "streamer_top_gameplay_bottom", "gameplay_top_streamer_bottom", "speaker_focus", "original", "auto"
    top_ratio: float = 0.45    # 45% = 864px of 1920
    bottom_ratio: float = 0.55 # 55% = 1056px of 1920
    facecam_position: str = "auto"  # "auto", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"
    enable_face_tracking: bool = True
    manual_facecam_box: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)
    manual_gameplay_box: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclass
class SubtitleConfig:
    """Configuration for Hormozi-style animated subtitles."""
    font_name: str = "Impact"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"   # White (AABBGGRR in ASS)
    highlight_color: str = "&H0000FFFF" # Vivid Yellow
    outline_color: str = "&H00000000"   # Black border
    outline_width: int = 4
    shadow_width: int = 2
    margin_v: int = 280                 # Vertical margin from bottom (9:16 safe zone)
    max_words_per_line: int = 4
    uppercase: bool = True


@dataclass
class VideoConfig:
    """Configuration for 9:16 vertical video clipping."""
    target_width: int = 1080
    target_height: int = 1920
    fps: int = 30
    min_clip_duration: float = 20.0  # seconds (default min 20s)
    max_clip_duration: float = 60.0  # seconds (default max 60s)
    default_crop_mode: str = "blur_background"  # "blur_background", "center_crop", "smart_crop"


@dataclass
class OllamaConfig:
    """Optional configuration for Ollama local LLM integration."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: int = 30
    enabled: bool = False


@dataclass
class AppConfig:
    """Global Application Configuration."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    gaming_layout: GamingLayoutConfig = field(default_factory=GamingLayoutConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    ffmpeg_path: str = FFMPEG_PATH
    ffprobe_path: str = FFPROBE_PATH


# Initialize directories on import
ensure_directories()
