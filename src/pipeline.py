"""
Complete End-to-End Clipping Pipeline for AI-YouTube-Clipper.
Orchestrates downloading, audio extraction, faster-whisper transcription,
moment detection, gaming split-screen reframing, Hormozi subtitles, and MP4 packaging.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.audio import extract_audio
from src.clip_selector import ClipSelector
from src.config import AppConfig, OUTPUT_CLIPS_DIR
from src.downloader import VideoDownloader, VideoInfo
from src.exporter import ClipExporter
from src.hook_generator import HookGenerator
from src.moment_detector import CandidateClip
from src.transcriber import Transcriber, TranscriptionResult
from src.utils import sanitize_filename
from src.video_processor import ProcessedClipResult, VideoProcessor

logger = logging.getLogger("AI-YouTube-Clipper.pipeline")


@dataclass
class PipelineProgress:
    """Live progress updates during pipeline execution."""
    stage: str
    percent: float  # 0.0 to 100.0
    message: str
    current_clip: Optional[int] = None
    total_clips: Optional[int] = None


class ClipperPipeline:
    """High-level orchestrator for automated YouTube clipping and short-form creation."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.downloader = VideoDownloader()
        self.transcriber = Transcriber(self.config.whisper)
        self.hook_generator = HookGenerator()
        self.clip_selector = ClipSelector(
            min_duration=self.config.video.min_clip_duration,
            max_duration=self.config.video.max_clip_duration,
            hook_generator=self.hook_generator
        )
        self.video_processor = VideoProcessor(
            video_config=self.config.video,
            gaming_config=self.config.gaming_layout,
            subtitle_config=self.config.subtitles
        )
        self.exporter = ClipExporter(OUTPUT_CLIPS_DIR)

    def process_video(
        self,
        input_source: Union[str, Path],
        target_clip_count: int = 5,
        min_clip_duration: Optional[float] = None,
        max_clip_duration: Optional[float] = None,
        layout_type: Optional[str] = None,
        burn_subtitles: bool = True,
        manual_facecam: Optional[Tuple[int, int, int, int]] = None,
        manual_gameplay: Optional[Tuple[int, int, int, int]] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute full pipeline from input video to finished 9:16 vertical clips.

        :param input_source: YouTube URL or path to local video file.
        :param target_clip_count: Number of candidate clips to generate.
        :param min_clip_duration: Minimum duration per clip in seconds (default 20s).
        :param max_clip_duration: Maximum duration per clip in seconds (default 60s).
        :param layout_type: Gaming layout style or None for config default.
        :param burn_subtitles: Whether to burn Hormozi ASS subtitles into the MP4.
        :param manual_facecam: Optional explicit facecam crop coordinates (x, y, w, h).
        :param manual_gameplay: Optional explicit gameplay crop coordinates (x, y, w, h).
        :param progress_callback: Optional status listener for UI updates.
        :return: Project summary dictionary containing clips, metadata, and ZIP path.
        """
        start_time = time.time()

        def _report(stage: str, percent: float, msg: str, curr: Optional[int] = None, total: Optional[int] = None):
            logger.info("[%s %.0f%%] %s", stage, percent, msg)
            if progress_callback:
                progress_callback(PipelineProgress(stage=stage, percent=percent, message=msg, current_clip=curr, total_clips=total))

        # ---------------------------------------------------------
        # STAGE 1: Ingestion (Download or Local File)
        # ---------------------------------------------------------
        _report("Ingestion", 5.0, "Ingesting video input...")
        input_str = str(input_source).strip()

        if input_str.startswith(("http://", "https://", "www.")):
            _report("Download", 10.0, f"Downloading YouTube video: {input_str}")
            video_info = self.downloader.download_youtube_video(input_str)
        else:
            local_path = Path(input_str)
            _report("Ingestion", 10.0, f"Loading local file: {local_path.name}")
            video_info = self.downloader.ingest_local_file(local_path)

        project_name = sanitize_filename(video_info.title, max_length=60)
        project_dir = OUTPUT_CLIPS_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # STAGE 2: Audio Extraction
        # ---------------------------------------------------------
        _report("Audio", 20.0, "Extracting speech audio for transcription...")
        audio_path = extract_audio(video_info.file_path)

        # ---------------------------------------------------------
        # STAGE 3: Faster-Whisper Transcription
        # ---------------------------------------------------------
        _report("Transcription", 35.0, f"Running faster-whisper ({self.config.whisper.model_size})...")
        transcription = self.transcriber.transcribe(audio_path, cache=True)

        # ---------------------------------------------------------
        # STAGE 4: AI Viral Moment Detection & Hook Generation
        # ---------------------------------------------------------
        _report("Moment Detection", 55.0, "Analyzing complete transcript and selecting viral moments...")
        candidates = self.clip_selector.select_clips(
            transcription=transcription,
            target_clip_count=target_clip_count,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration
        )

        if not candidates:
            raise RuntimeError("No suitable moments detected within the specified duration constraints.")

        # ---------------------------------------------------------
        # STAGE 5: Gaming Layout & Video Rendering
        # ---------------------------------------------------------
        total_clips = len(candidates)
        processed_clips: List[ProcessedClipResult] = []

        for idx, candidate in enumerate(candidates, 1):
            clip_percent = 60.0 + (30.0 * (idx - 1) / total_clips)
            _report(
                "Rendering",
                clip_percent,
                f"Rendering clip {idx}/{total_clips}: '{candidate.title}'",
                curr=idx,
                total=total_clips
            )

            result = self.video_processor.process_clip(
                source_video=video_info.file_path,
                candidate=candidate,
                transcription=transcription,
                output_dir=project_dir,
                layout_override=layout_type,
                burn_subtitles=burn_subtitles,
                manual_facecam=manual_facecam,
                manual_gameplay=manual_gameplay,
                progress_callback=lambda m: _report("Rendering", clip_percent, m, curr=idx, total=total_clips)
            )
            processed_clips.append(result)

        # ---------------------------------------------------------
        # STAGE 6: Packaging, Thumbnails & Export
        # ---------------------------------------------------------
        _report("Packaging", 95.0, "Generating thumbnails and packaging ZIP...")
        package_info = self.exporter.package_project(
            project_name=project_name,
            clips=processed_clips,
            project_metadata={
                "video_id": video_info.video_id,
                "video_title": video_info.title,
                "duration": video_info.duration,
                "whisper_model": self.config.whisper.model_size,
                "language": transcription.language,
                "pipeline_time_seconds": round(time.time() - start_time, 2),
            }
        )

        _report("Complete", 100.0, f"Successfully processed {len(processed_clips)} clips in {time.time() - start_time:.1f}s!")
        return package_info
