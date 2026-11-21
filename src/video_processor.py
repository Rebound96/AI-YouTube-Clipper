"""
Video processing and rendering engine for AI-YouTube-Clipper.
Handles cutting clips, applying gaming split-screen layouts, vertical 9:16 reframing,
and burning synchronized Hormozi-style ASS subtitles via FFmpeg.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Dict, Optional, Tuple, Union

from src.audio import probe_media
from src.config import FFMPEG_PATH, GamingLayoutConfig, OUTPUT_CLIPS_DIR, SubtitleConfig, TEMP_DIR, VideoConfig
from src.gaming_layout import GamingLayoutManager, GamingLayoutPlan
from src.moment_detector import CandidateClip
from src.subtitle_generator import SubtitleGenerator
from src.transcriber import TranscriptionResult
from src.utils import run_ffmpeg, sanitize_filename

logger = logging.getLogger("AI-YouTube-Clipper.video_processor")


def escape_ffmpeg_filter_path(path: Path) -> str:
    """
    Format a file path for FFmpeg filter arguments (handles Windows drive colons and backslashes).
    Example: C:\\Users\\Admin\\clip.ass -> C\\:/Users/Admin/clip.ass
    """
    posix_str = path.resolve().as_posix()
    # Escape colon after drive letter: C:/ -> C\:/
    if len(posix_str) > 1 and posix_str[1] == ":":
        posix_str = posix_str[0] + "\\:" + posix_str[2:]
    # Escape any single quotes for FFmpeg filter parameter
    posix_str = posix_str.replace("'", r"\'")
    return posix_str


@dataclass
class ProcessedClipResult:
    """Descriptor for a rendered final video clip."""
    clip_id: str
    output_path: Path
    duration: float
    width: int
    height: int
    file_size_bytes: int
    title: str
    hook: str
    engagement_score: float
    category: str
    transcript_text: str
    subtitles_burned: bool
    layout_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "output_path": str(self.output_path),
            "filename": self.output_path.name,
            "duration": round(self.duration, 2),
            "width": self.width,
            "height": self.height,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "title": self.title,
            "hook": self.hook,
            "engagement_score": round(self.engagement_score, 1),
            "category": self.category,
            "transcript_text": self.transcript_text,
            "subtitles_burned": self.subtitles_burned,
            "layout_used": self.layout_used,
        }


class VideoProcessor:
    """Processes, edits, and exports vertical short-form video clips."""

    def __init__(
        self,
        video_config: Optional[VideoConfig] = None,
        gaming_config: Optional[GamingLayoutConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None
    ):
        self.video_config = video_config or VideoConfig()
        self.gaming_config = gaming_config or GamingLayoutConfig()
        self.subtitle_config = subtitle_config or SubtitleConfig()
        self.layout_manager = GamingLayoutManager(self.gaming_config)
        self.sub_generator = SubtitleGenerator(self.subtitle_config)

    def process_clip(
        self,
        source_video: Path,
        candidate: CandidateClip,
        transcription: TranscriptionResult,
        output_dir: Optional[Path] = None,
        layout_override: Optional[str] = None,
        burn_subtitles: bool = True,
        manual_facecam: Optional[Tuple[int, int, int, int]] = None,
        manual_gameplay: Optional[Tuple[int, int, int, int]] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ProcessedClipResult:
        """
        Produce a finished vertical 9:16 clip from candidate specifications.
        """
        source_video = Path(source_video).resolve()
        if not source_video.is_file():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        out_dir = Path(output_dir or OUTPUT_CLIPS_DIR).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = sanitize_filename(candidate.title, max_length=50)
        output_filename = f"{candidate.clip_id}_{safe_title}.mp4"
        output_path = out_dir / output_filename

        # Always store temporary ASS subtitles in dedicated TEMP_DIR with a clean path
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        safe_clip_id = sanitize_filename(candidate.clip_id)
        temp_ass_path = TEMP_DIR / f"{safe_clip_id}_{os.getpid()}_temp.ass"

        if progress_callback:
            progress_callback(f"Preparing layout and subtitles for clip: {candidate.title}")

        # Step 1: Probe video dimensions
        media_info = probe_media(source_video)
        video_stream = next((s for s in media_info.get("streams", []) if s["codec_type"] == "video"), None)
        orig_w = int(video_stream["width"]) if video_stream and "width" in video_stream else 1920
        orig_h = int(video_stream["height"]) if video_stream and "height" in video_stream else 1080

        # Step 2: Plan Layout
        layout_plan = self.layout_manager.plan_layout(
            video_path=source_video,
            video_width=orig_w,
            video_height=orig_h,
            layout_type_override=layout_override,
            manual_facecam=manual_facecam,
            manual_gameplay=manual_gameplay
        )

        # Step 3: Generate Hormozi-style ASS Subtitles for this clip's time range
        words_in_clip = transcription.get_words_in_range(candidate.start_time, candidate.end_time)
        has_subtitles = burn_subtitles and len(words_in_clip) > 0

        if has_subtitles:
            self.sub_generator.generate_ass(
                words=words_in_clip,
                output_ass_path=temp_ass_path,
                clip_start_time=candidate.start_time,
                video_width=self.video_config.target_width,
                video_height=self.video_config.target_height
            )

        # Step 4: Build FFmpeg filter_complex
        # Chain layout filter to subtitles filter
        escaped_ass = escape_ffmpeg_filter_path(temp_ass_path)
        if has_subtitles and temp_ass_path.exists():
            full_filter_complex = f"{layout_plan.filter_complex};[v_base]ass='{escaped_ass}'[v_out]"
            output_v_map = "[v_out]"
        else:
            full_filter_complex = f"{layout_plan.filter_complex}"
            output_v_map = "[v_base]"

        if progress_callback:
            progress_callback(f"Rendering vertical video for: {candidate.title}")

        # Step 5: Execute FFmpeg render
        # -ss before -i for fast accurate seek
        cmd_args = [
            "-y",
            "-ss", str(candidate.start_time),
            "-i", str(source_video),
            "-t", str(candidate.duration),
            "-filter_complex", full_filter_complex,
            "-map", output_v_map,
            "-map", "0:a?",  # Map audio if present
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            run_ffmpeg(cmd_args, executable=FFMPEG_PATH, description=f"Render Clip {candidate.clip_id}")
        finally:
            # Clean up temporary ASS file
            if temp_ass_path.exists():
                try:
                    temp_ass_path.unlink()
                except Exception:
                    pass

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Clip rendering failed; output file is missing or empty: {output_path}")

        file_size = output_path.stat().st_size
        logger.info(
            "Successfully rendered clip %s (%.1f sec, %d MB) at %s",
            candidate.clip_id, candidate.duration, file_size // (1024 * 1024), output_path.name
        )

        return ProcessedClipResult(
            clip_id=candidate.clip_id,
            output_path=output_path,
            duration=candidate.duration,
            width=self.video_config.target_width,
            height=self.video_config.target_height,
            file_size_bytes=file_size,
            title=candidate.title,
            hook=candidate.hook,
            engagement_score=candidate.engagement_score,
            category=candidate.category,
            transcript_text=candidate.transcript_text,
            subtitles_burned=has_subtitles,
            layout_used=layout_plan.layout_type
        )
