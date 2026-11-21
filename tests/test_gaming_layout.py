"""
Unit tests for facecam detection and gaming split-screen layout planning.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.face_detector import FaceDetector, FaceTracker, FacecamRegion
from src.gaming_layout import (
    GamingLayoutManager,
    LayoutCoordinates,
    crop_and_scale_fit,
    draw_detection_overlay,
    generate_split_preview_image,
)


def test_face_tracker_smoothing():
    """Verify face tracker applies exponential smoothing."""
    tracker = FaceTracker(smoothing_factor=0.5)

    # Initial update
    box1 = (100, 100, 200, 200)
    res1 = tracker.update(box1)
    assert res1 == box1

    # Second update
    box2 = (120, 120, 200, 200)
    res2 = tracker.update(box2)
    # Expected: 0.5 * 120 + 0.5 * 100 = 110
    assert res2[0] == 110
    assert res2[1] == 110


def test_gaming_layout_manager_split_screen(tmp_path: Path):
    """Verify default Streamer Top 45% / Gameplay Bottom 55% layout calculation."""
    manager = GamingLayoutManager()

    # Create dummy video path
    dummy_video = tmp_path / "dummy_stream.mp4"
    dummy_video.touch()

    # Manual facecam on the left (Kai Cenat stream style: 0, 0, 500, 600 in 1920x1080)
    manual_face = (0, 0, 500, 600)
    manual_game = (500, 0, 1420, 1080)

    plan = manager.plan_layout(
        video_path=dummy_video,
        video_width=1920,
        video_height=1080,
        layout_type_override="streamer_top_gameplay_bottom",
        manual_facecam=manual_face,
        manual_gameplay=manual_game
    )

    assert plan.target_width == 1080
    assert plan.target_height == 1920
    assert plan.top_height == 864   # 45% of 1920
    assert plan.bottom_height == 1056 # 55% of 1920
    assert plan.top_height + plan.bottom_height == 1920

    # Verify filter complex
    assert "crop=500:600:0:0" in plan.filter_complex
    assert "crop=1420:1080:500:0" in plan.filter_complex
    assert "scale=1080:864" in plan.filter_complex
    assert "scale=1080:1056" in plan.filter_complex
    assert "vstack=inputs=2[v_base]" in plan.filter_complex


def test_gaming_layout_manager_other_layouts(tmp_path: Path):
    """Verify gameplay top, speaker focus, and original blur background layouts."""
    manager = GamingLayoutManager()
    dummy_video = tmp_path / "dummy_stream.mp4"
    dummy_video.touch()

    # 1. Gameplay Top / Streamer Bottom
    plan_reverse = manager.plan_layout(
        video_path=dummy_video,
        video_width=1920,
        video_height=1080,
        layout_type_override="gameplay_top_streamer_bottom"
    )
    assert plan_reverse.top_height == 1056
    assert plan_reverse.bottom_height == 864
    assert "vstack=inputs=2[v_base]" in plan_reverse.filter_complex

    # 2. Speaker Focus
    plan_speaker = manager.plan_layout(
        video_path=dummy_video,
        video_width=1920,
        video_height=1080,
        layout_type_override="speaker_focus"
    )
    assert plan_speaker.target_width == 1080
    assert plan_speaker.target_height == 1920
    assert "scale=1080:1920[v_base]" in plan_speaker.filter_complex

    # 3. Original Blur Background
    plan_blur = manager.plan_layout(
        video_path=dummy_video,
        video_width=1920,
        video_height=1080,
        layout_type_override="original"
    )
    assert "boxblur=" in plan_blur.filter_complex


def test_facecam_isolation_left():
    """Verify facecam on left is isolated without capturing excessive gameplay."""
    detector = FaceDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Simulated face on left at (150, 200, 120, 140)
    region = detector.find_webcam_box_in_frame(frame, face_box=(150, 200, 120, 140))
    assert region is not None
    assert "left" in region.position
    assert region.has_facecam is True
    # Verify width is reasonably tight around the streamer (not spanning across the whole game)
    assert 200 <= region.w <= 650
    assert region.x <= 150

    # Verify gameplay region calculation excludes facecam
    manager = GamingLayoutManager()
    game_coords = manager._compute_gameplay_region(region, 1920, 1080)
    assert game_coords.x >= region.x + region.w or game_coords.x >= region.w
    assert game_coords.w >= 800


def test_facecam_isolation_right():
    """Verify facecam on right is detected and gameplay is extracted to its left."""
    detector = FaceDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Simulated face on right at (1600, 200, 120, 140)
    region = detector.find_webcam_box_in_frame(frame, face_box=(1600, 200, 120, 140))
    assert region is not None
    assert "right" in region.position
    assert region.x >= 1400

    manager = GamingLayoutManager()
    game_coords = manager._compute_gameplay_region(region, 1920, 1080)
    # Gameplay starts at 0 and ends before/at the facecam box
    assert game_coords.x == 0
    assert game_coords.w <= region.x + 50


def test_facecam_corner_detection():
    """Verify quadrant position detection for corner facecams."""
    detector = FaceDetector()
    w, h = 1920, 1080

    # Top-Left
    assert detector._classify_position(200, 150, w, h) == "top_left"
    # Top-Right
    assert detector._classify_position(1750, 150, w, h) == "top_right"
    # Bottom-Left
    assert detector._classify_position(200, 950, w, h) == "bottom_left"
    # Bottom-Right
    assert detector._classify_position(1750, 950, w, h) == "bottom_right"


def test_split_screen_preview_generation():
    """Verify interactive split-screen preview generates exact 1080x1920 stacked image."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    face_box = (100, 100, 350, 280)
    game_box = (450, 0, 1470, 1080)

    preview = generate_split_preview_image(frame, face_box, game_box, target_w=1080, target_h=1920)
    assert preview.shape == (1920, 1080, 3)

    annotated = draw_detection_overlay(frame, face_box, game_box)
    assert annotated.shape == (1080, 1920, 3)
