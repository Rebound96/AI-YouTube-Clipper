"""
Integration tests for video processing, gaming split-screen rendering,
subtitle burn-in, and MP4 packaging.
"""

from pathlib import Path
import pytest

from src.audio import probe_media
from src.config import FFMPEG_PATH, VideoConfig
from src.exporter import ClipExporter
from src.moment_detector import CandidateClip
from src.transcriber import TranscriptSegment, TranscriptionResult, WordTiming
from src.utils import run_ffmpeg
from src.video_processor import VideoProcessor


@pytest.fixture(scope="session")
def full_hd_sample_video(tmp_path_factory) -> Path:
    """Generate a 1920x1080 3-second test video with sine audio."""
    temp_dir = tmp_path_factory.mktemp("video_proc")
    video_path = temp_dir / "sample_1080p.mp4"

    cmd = [
        "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(video_path)
    ]
    run_ffmpeg(cmd, executable=FFMPEG_PATH, description="Create 1080p test video")
    assert video_path.exists()
    return video_path


def test_video_processor_render_split_screen(full_hd_sample_video: Path, tmp_path: Path):
    """
    Verify complete rendering of vertical 9:16 (1080x1920) gaming split-screen video
    with burned-in animated subtitles.
    """
    processor = VideoProcessor(video_config=VideoConfig(target_width=1080, target_height=1920))

    candidate = CandidateClip(
        clip_id="clip_0000_0002",
        start_time=0.0,
        end_time=2.0,
        duration=2.0,
        title="Epic Clutch Play",
        hook="WATCH WHAT HAPPENS NOW 🔥",
        engagement_score=85.0,
        category="gaming_reaction",
        reason="Test high engagement render",
        transcript_text="WATCH WHAT HAPPENS NOW",
        segment_ids=[1]
    )

    words = [
        WordTiming(word="WATCH", start=0.0, end=0.4),
        WordTiming(word="WHAT", start=0.4, end=0.7),
        WordTiming(word="HAPPENS", start=0.7, end=1.2),
        WordTiming(word="NOW", start=1.2, end=1.8),
    ]
    segment = TranscriptSegment(id=1, start=0.0, end=2.0, text="WATCH WHAT HAPPENS NOW", words=words)
    transcription = TranscriptionResult(segments=[segment], language="en", language_probability=1.0, duration=2.0, full_text="WATCH WHAT HAPPENS NOW")

    # Render clip using streamer_top_gameplay_bottom split-screen
    result = processor.process_clip(
        source_video=full_hd_sample_video,
        candidate=candidate,
        transcription=transcription,
        output_dir=tmp_path,
        layout_override="streamer_top_gameplay_bottom",
        burn_subtitles=True,
        manual_facecam=(0, 0, 600, 700),
        manual_gameplay=(600, 0, 1320, 1080)
    )

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0

    # Probe rendered video to verify exact dimensions and audio sync
    info = probe_media(result.output_path)
    streams = info["streams"]
    video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

    assert video_stream is not None
    assert video_stream["width"] == 1080
    assert video_stream["height"] == 1920
    assert audio_stream is not None

    duration = float(info["format"]["duration"])
    assert 1.8 <= duration <= 2.2
    assert result.subtitles_burned is True


def test_video_processor_render_without_subtitles(full_hd_sample_video: Path, tmp_path: Path):
    """
    Verify complete rendering of vertical 9:16 split-screen video with subtitles DISABLED (clean video).
    """
    processor = VideoProcessor(video_config=VideoConfig(target_width=1080, target_height=1920))

    candidate = CandidateClip(
        clip_id="clip_nosub_001",
        start_time=0.0,
        end_time=2.0,
        duration=2.0,
        title="Clean Clip Without Subs",
        hook="CLEAN HOOK 🔥",
        engagement_score=80.0,
        category="gaming_reaction",
        reason="Test clean render without subtitles",
        transcript_text="NO SUBTITLES HERE",
        segment_ids=[1]
    )

    segment = TranscriptSegment(id=1, start=0.0, end=2.0, text="NO SUBTITLES HERE", words=[])
    transcription = TranscriptionResult(segments=[segment], language="en", language_probability=1.0, duration=2.0, full_text="NO SUBTITLES HERE")

    # Render clip with burn_subtitles=False
    result = processor.process_clip(
        source_video=full_hd_sample_video,
        candidate=candidate,
        transcription=transcription,
        output_dir=tmp_path,
        layout_override="streamer_top_gameplay_bottom",
        burn_subtitles=False,
        manual_facecam=(0, 0, 500, 500),
        manual_gameplay=(500, 0, 1420, 1080)
    )

    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
    assert result.subtitles_burned is False

    info = probe_media(result.output_path)
    video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    assert video_stream["width"] == 1080
    assert video_stream["height"] == 1920


def test_exporter_packaging(full_hd_sample_video: Path, tmp_path: Path):
    """Verify thumbnail extraction, project metadata, and ZIP bundling."""
    exporter = ClipExporter(base_output_dir=tmp_path)

    # Test thumbnail generation
    thumb = exporter.generate_thumbnail(full_hd_sample_video)
    assert thumb.exists()
    assert thumb.stat().st_size > 0

    # Test packaging
    from src.video_processor import ProcessedClipResult
    mock_clip = ProcessedClipResult(
        clip_id="test_01",
        output_path=full_hd_sample_video,
        duration=3.0,
        width=1920,
        height=1080,
        file_size_bytes=full_hd_sample_video.stat().st_size,
        title="Test Clip",
        hook="TEST HOOK",
        engagement_score=90.0,
        category="gaming_reaction",
        transcript_text="Test speech",
        subtitles_burned=True,
        layout_used="streamer_top_gameplay_bottom"
    )

    pkg = exporter.package_project("Test_Project", [mock_clip])
    assert Path(pkg["zip_path"]).exists()
    assert Path(pkg["metadata_path"]).exists()
    assert len(pkg["clips"]) == 1
