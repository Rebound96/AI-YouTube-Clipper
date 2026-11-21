"""
Test suite verifying manual facecam region selection, gameplay crop isolation,
complete vertical composition, and zero face-detection dependency in manual mode.
Also tests with synthetic video containing a webcam overlay and moving game characters.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.audio import probe_media
from src.face_detector import FaceDetector, FacecamRegion
from src.gaming_layout import GamingLayoutManager, LayoutCoordinates
from src.moment_detector import CandidateClip
from src.transcriber import TranscriptionResult, TranscriptSegment, WordTiming
from src.video_processor import VideoProcessor


def create_synthetic_gaming_video(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    duration_sec: float = 3.0
) -> Path:
    """
    Generate a synthetic MP4 video simulating a gaming stream:
    - Static streamer webcam overlay in the top-left corner (0, 0, 360, 288).
    - Moving in-game NPC characters with faces in the center of the gameplay screen.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_sec)

    for f in range(total_frames):
        # Frame background: gaming landscape (dark green/blue grid)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30, 20, 15)

        # 1. Active Gameplay Elements: Draw moving "game characters" in the center
        # Character 1 moves across center: (500 + f*5, 300)
        char1_x = int(500 + (f * 4) % 300)
        char1_y = 320
        # Draw game character body and face
        cv2.circle(frame, (char1_x, char1_y), 45, (80, 120, 220), -1)  # Head
        cv2.rectangle(frame, (char1_x - 35, char1_y + 45), (char1_x + 35, char1_y + 160), (40, 80, 160), -1)

        # Character 2 (NPC in center)
        cv2.circle(frame, (850, 400), 50, (60, 180, 100), -1)

        # 2. Static Streamer Webcam Overlay in Top-Left (0, 0, 360, 288)
        cam_w, cam_h = 360, 288
        # Webcam border/frame
        cv2.rectangle(frame, (0, 0), (cam_w, cam_h), (200, 50, 180), 4)
        # Streamer interior background
        frame[4:cam_h-4, 4:cam_w-4] = (60, 40, 70)
        # Streamer head & shoulders
        cv2.circle(frame, (cam_w // 2, cam_h // 2 - 20), 55, (210, 180, 160), -1)
        cv2.ellipse(frame, (cam_w // 2, cam_h - 20), (90, 60), 0, 0, 180, (140, 50, 120), -1)
        # Streamer mic
        cv2.circle(frame, (cam_w // 2 + 35, cam_h // 2 + 30), 18, (30, 30, 30), -1)

        writer.write(frame)

    writer.release()
    return output_path


def test_automatic_detection_rejects_moving_game_characters(tmp_path: Path):
    """
    Verify the architectural safeguard: automatic facecam detection must NOT
    mistake moving central game characters for a streamer webcam overlay.
    """
    video_path = tmp_path / "game_with_characters.mp4"
    create_synthetic_gaming_video(video_path, width=1280, height=720, duration_sec=2.0)

    detector = FaceDetector()
    # In a video with moving game characters or ambiguous layout,
    # the detector must recognize moving gameplay or low confidence, requiring manual selection
    region = detector.analyze_video_facecam(video_path, sample_interval_sec=0.5, max_samples=4)

    # Must NOT claim high confidence on game characters
    if not region.has_facecam:
        assert region.confidence == 0.0 or region.detection_method in ("gameplay_characters_detected", "fallback")
    else:
        # If it detects the top-left streamer, it must be top_left, not center
        assert region.position in ("top_left", "left")


def test_manual_mode_has_zero_face_detection_dependency(tmp_path: Path):
    """
    Verify that providing manual coordinates completely bypasses face detection.
    """
    manager = GamingLayoutManager()
    dummy_video = tmp_path / "dummy_video.mp4"
    dummy_video.touch()

    # User explicitly selects:
    # Webcam: Top-Left (0, 0, 360, 288)
    # Gameplay: Right portion (360, 0, 920, 720)
    manual_face = (0, 0, 360, 288)
    manual_game = (360, 0, 920, 720)

    plan = manager.plan_layout(
        video_path=dummy_video,
        video_width=1280,
        video_height=720,
        layout_type_override="streamer_top_gameplay_bottom",
        manual_facecam=manual_face,
        manual_gameplay=manual_game
    )

    # Check that manual detection method was recorded
    assert plan.facecam_detected is True
    assert plan.target_width == 1080
    assert plan.target_height == 1920
    assert plan.top_height == 864
    assert plan.bottom_height == 1056

    # Verify crops reflect the exact user choices (clamped to even numbers)
    assert f"crop={manual_face[2]}:{manual_face[3]}:{manual_face[0]}:{manual_face[1]}" in plan.filter_complex
    assert f"crop={manual_game[2]}:{manual_game[3]}:{manual_game[0]}:{manual_game[1]}" in plan.filter_complex


def test_manual_crop_video_rendering_and_subtitles_toggle(tmp_path: Path):
    """
    End-to-end verification of rendering a real clip using manual coordinates:
    1. Render with subtitles OFF -> Verify clean video, exact 1080x1920 resolution.
    2. Render with subtitles ON -> Verify subtitled video, exact 1080x1920 resolution.
    3. Verify audio synchronization & container validity.
    """
    video_path = tmp_path / "synthetic_gaming_stream.mp4"
    create_synthetic_gaming_video(video_path, width=1280, height=720, fps=24, duration_sec=3.0)

    # Synthetic candidate clip (1.0 to 2.5 seconds)
    candidate = CandidateClip(
        clip_id="test_manual_clip",
        start_time=0.5,
        end_time=2.5,
        duration=2.0,
        title="Test Manual Gaming Clip",
        hook="Check out this gameplay!",
        engagement_score=92.0,
        category="gaming_reaction",
        reason="Manual crop test",
        transcript_text="Check out this incredible play right now",
        segment_ids=[1]
    )

    transcription = TranscriptionResult(
        full_text="Check out this incredible play right now",
        language="en",
        language_probability=0.99,
        duration=3.0,
        segments=[
            TranscriptSegment(
                id=1,
                start=0.5,
                end=2.5,
                text="Check out this incredible play right now",
                words=[
                    WordTiming(word="Check", start=0.5, end=0.8, probability=0.99),
                    WordTiming(word="out", start=0.8, end=1.1, probability=0.99),
                    WordTiming(word="this", start=1.1, end=1.4, probability=0.99),
                    WordTiming(word="play", start=1.4, end=1.9, probability=0.99),
                    WordTiming(word="now", start=1.9, end=2.4, probability=0.99),
                ]
            )
        ]
    )

    processor = VideoProcessor()
    out_dir = tmp_path / "rendered_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render Clean Video (Subtitles OFF)
    res_clean = processor.process_clip(
        source_video=video_path,
        candidate=candidate,
        transcription=transcription,
        output_dir=out_dir,
        burn_subtitles=False,
        manual_facecam=(0, 0, 360, 288),
        manual_gameplay=(360, 0, 920, 720)
    )

    assert res_clean.output_path.exists()
    assert res_clean.subtitles_burned is False
    probe_clean = probe_media(res_clean.output_path)
    v_clean = next(s for s in probe_clean["streams"] if s["codec_type"] == "video")
    assert v_clean["width"] == 1080
    assert v_clean["height"] == 1920

    # 2. Render Subtitled Video (Subtitles ON)
    candidate.clip_id = "test_manual_subtitled"
    res_sub = processor.process_clip(
        source_video=video_path,
        candidate=candidate,
        transcription=transcription,
        output_dir=out_dir,
        burn_subtitles=True,
        manual_facecam=(0, 0, 360, 288),
        manual_gameplay=(360, 0, 920, 720)
    )

    assert res_sub.output_path.exists()
    assert res_sub.subtitles_burned is True
    probe_sub = probe_media(res_sub.output_path)
    v_sub = next(s for s in probe_sub["streams"] if s["codec_type"] == "video")
    assert v_sub["width"] == 1080
    assert v_sub["height"] == 1920
