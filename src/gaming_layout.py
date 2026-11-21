"""
Gaming Split-Screen Layout Editor for AI-YouTube-Clipper.
Manages region detection, aspect-ratio preserving crops, visual preview generation,
and FFmpeg filtergraph construction for vertical shorts (Facecam Top 45%, Gameplay Bottom 55%).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import GamingLayoutConfig
from src.face_detector import FaceDetector, FacecamRegion

logger = logging.getLogger("AI-YouTube-Clipper.gaming_layout")


@dataclass
class LayoutCoordinates:
    """Bounding box coordinates (x, y, width, height) in source video pixels."""
    x: int
    y: int
    w: int
    h: int

    def clamp(self, max_w: int, max_h: int) -> "LayoutCoordinates":
        """Clamp coordinates so they never exceed video boundaries and have even dimensions."""
        cx = max(0, min(self.x, max_w - 2))
        cy = max(0, min(self.y, max_h - 2))
        cw = max(2, min(self.w, max_w - cx))
        ch = max(2, min(self.h, max_h - cy))
        if cw % 2 != 0:
            cw -= 1
        if ch % 2 != 0:
            ch -= 1
        return LayoutCoordinates(cx, cy, max(2, cw), max(2, ch))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_crop_filter(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"


@dataclass
class GamingLayoutPlan:
    """Full execution plan for rendering the vertical video."""
    layout_type: str
    target_width: int
    target_height: int
    top_region: LayoutCoordinates
    bottom_region: Optional[LayoutCoordinates]
    top_height: int
    bottom_height: int
    filter_complex: str
    facecam_detected: bool
    facecam_position: str


def crop_and_scale_fit(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Scale and center-crop an image region to fill target_w x target_h without distortion.
    """
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = max(target_w / float(iw), target_h / float(ih))
    new_w = max(int(round(iw * scale)), target_w)
    new_h = max(int(round(ih * scale)), target_h)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    return resized[y1:y1 + target_h, x1:x1 + target_w]


def generate_split_preview_image(
    frame: np.ndarray,
    facecam_box: Tuple[int, int, int, int],
    gameplay_box: Tuple[int, int, int, int],
    target_w: int = 1080,
    target_h: int = 1920,
    top_ratio: float = 0.45
) -> np.ndarray:
    """
    Generate an in-memory 1080x1920 vertical preview of the split-screen layout.
    """
    top_h = int(target_h * top_ratio)
    bottom_h = target_h - top_h

    fh, fw = frame.shape[:2]

    # Crop Facecam Box
    fx, fy, f_w, f_h = facecam_box
    fx = max(0, min(fx, fw - 10))
    fy = max(0, min(fy, fh - 10))
    f_w = max(10, min(f_w, fw - fx))
    f_h = max(10, min(f_h, fh - fy))
    face_crop = frame[fy:fy + f_h, fx:fx + f_w]

    # Crop Gameplay Box
    gx, gy, g_w, g_h = gameplay_box
    gx = max(0, min(gx, fw - 10))
    gy = max(0, min(gy, fh - 10))
    g_w = max(10, min(g_w, fw - gx))
    g_h = max(10, min(g_h, fh - gy))
    game_crop = frame[gy:gy + g_h, gx:gx + g_w]

    # Resize and fit without distortion
    top_scaled = crop_and_scale_fit(face_crop, target_w, top_h)
    bottom_scaled = crop_and_scale_fit(game_crop, target_w, bottom_h)

    # Stack vertically into 1080x1920
    stacked = np.vstack([top_scaled, bottom_scaled])
    return stacked


def draw_detection_overlay(
    frame: np.ndarray,
    facecam_box: Tuple[int, int, int, int],
    gameplay_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Draw annotated bounding boxes on the original frame for interactive preview in the UI.
    """
    annotated = frame.copy()
    fh, fw = frame.shape[:2]

    # 1. Draw Gameplay Region (Cyan)
    gx, gy, gw, gh = gameplay_box
    cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (255, 230, 0), 3)
    cv2.putText(
        annotated,
        "GAMEPLAY REGION",
        (gx + 15, gy + 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 230, 0),
        2,
        cv2.LINE_AA
    )

    # 2. Draw Facecam Overlay (Vibrant Magenta/Purple)
    fx, fy, f_w, f_h = facecam_box
    cv2.rectangle(annotated, (fx, fy), (fx + f_w, fy + f_h), (255, 0, 220), 4)
    cv2.putText(
        annotated,
        "STREAMER FACECAM",
        (fx + 10, max(fy - 12, 30)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 0, 220),
        2,
        cv2.LINE_AA
    )

    return annotated


class GamingLayoutManager:
    """Computes facecam and gameplay crops and produces FFmpeg filtergraph strings."""

    def __init__(self, config: Optional[GamingLayoutConfig] = None):
        self.config = config or GamingLayoutConfig()
        self.face_detector = FaceDetector()

    def plan_layout(
        self,
        video_path: Path,
        video_width: int = 1920,
        video_height: int = 1080,
        layout_type_override: Optional[str] = None,
        manual_facecam: Optional[Tuple[int, int, int, int]] = None,
        manual_gameplay: Optional[Tuple[int, int, int, int]] = None
    ) -> GamingLayoutPlan:
        """
        Analyze video and determine crop regions and FFmpeg filtergraph for vertical 9:16 layout.
        """
        layout_type = layout_type_override or self.config.layout_type
        target_w = 1080
        target_h = 1920

        # Heights according to ratios (Default: Top 45% = 864, Bottom 55% = 1056)
        top_h = int(target_h * self.config.top_ratio)
        bottom_h = target_h - top_h  # guarantees exact sum = 1920

        # Step 1: Detect or apply facecam region
        if manual_facecam:
            face_box = FacecamRegion(
                x=manual_facecam[0],
                y=manual_facecam[1],
                w=manual_facecam[2],
                h=manual_facecam[3],
                position="manual",
                confidence=1.0,
                has_facecam=True,
                detection_method="manual"
            )
        elif self.config.manual_facecam_box:
            mf = self.config.manual_facecam_box
            face_box = FacecamRegion(
                x=mf[0], y=mf[1], w=mf[2], h=mf[3],
                position="manual", confidence=1.0, has_facecam=True, detection_method="manual"
            )
        else:
            face_box = self.face_detector.analyze_video_facecam(video_path)

        # Handle 'auto' layout type
        if layout_type == "auto":
            if face_box.has_facecam and face_box.confidence >= 0.35:
                layout_type = "streamer_top_gameplay_bottom"
            else:
                layout_type = "original"

        # Step 2: Compute Gameplay Region
        if manual_gameplay:
            game_box = LayoutCoordinates(*manual_gameplay)
        elif self.config.manual_gameplay_box:
            game_box = LayoutCoordinates(*self.config.manual_gameplay_box)
        else:
            game_box = self._compute_gameplay_region(face_box, video_width, video_height)

        face_coords = LayoutCoordinates(face_box.x, face_box.y, face_box.w, face_box.h).clamp(video_width, video_height)
        game_box = game_box.clamp(video_width, video_height)

        # Step 3: Construct FFmpeg Filter Complex based on selected layout
        if layout_type == "streamer_top_gameplay_bottom":
            # Streamer Top (864px), Gameplay Bottom (1056px)
            filter_complex = (
                f"[0:v]{face_coords.to_crop_filter()},"
                f"scale={target_w}:{top_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{top_h}[top];"
                f"[0:v]{game_box.to_crop_filter()},"
                f"scale={target_w}:{bottom_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{bottom_h}[bottom];"
                f"[top][bottom]vstack=inputs=2[v_base]"
            )
            return GamingLayoutPlan(
                layout_type=layout_type,
                target_width=target_w,
                target_height=target_h,
                top_region=face_coords,
                bottom_region=game_box,
                top_height=top_h,
                bottom_height=bottom_h,
                filter_complex=filter_complex,
                facecam_detected=face_box.has_facecam,
                facecam_position=face_box.position
            )

        elif layout_type == "gameplay_top_streamer_bottom":
            # Gameplay Top (1056px), Streamer Bottom (864px)
            filter_complex = (
                f"[0:v]{game_box.to_crop_filter()},"
                f"scale={target_w}:{bottom_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{bottom_h}[top];"
                f"[0:v]{face_coords.to_crop_filter()},"
                f"scale={target_w}:{top_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{top_h}[bottom];"
                f"[top][bottom]vstack=inputs=2[v_base]"
            )
            return GamingLayoutPlan(
                layout_type=layout_type,
                target_width=target_w,
                target_height=target_h,
                top_region=game_box,
                bottom_region=face_coords,
                top_height=bottom_h,
                bottom_height=top_h,
                filter_complex=filter_complex,
                facecam_detected=face_box.has_facecam,
                facecam_position=face_box.position
            )

        elif layout_type == "speaker_focus":
            # Full screen 9:16 crop centered on facecam / speaker
            crop_w = int(video_height * (9 / 16))
            crop_h = video_height
            cx = face_coords.x + face_coords.w // 2
            crop_x = max(0, min(cx - crop_w // 2, video_width - crop_w))
            crop_y = 0
            speaker_coords = LayoutCoordinates(crop_x, crop_y, crop_w, crop_h)

            filter_complex = (
                f"[0:v]{speaker_coords.to_crop_filter()},"
                f"scale={target_w}:{target_h}[v_base]"
            )
            return GamingLayoutPlan(
                layout_type=layout_type,
                target_width=target_w,
                target_height=target_h,
                top_region=speaker_coords,
                bottom_region=None,
                top_height=target_h,
                bottom_height=0,
                filter_complex=filter_complex,
                facecam_detected=face_box.has_facecam,
                facecam_position=face_box.position
            )

        else:
            # "original" / Blur background landscape into 9:16
            filter_complex = (
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},boxblur=25:5[bg];"
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_base]"
            )
            full_frame = LayoutCoordinates(0, 0, video_width, video_height)
            return GamingLayoutPlan(
                layout_type="original",
                target_width=target_w,
                target_height=target_h,
                top_region=full_frame,
                bottom_region=None,
                top_height=target_h,
                bottom_height=0,
                filter_complex=filter_complex,
                facecam_detected=face_box.has_facecam,
                facecam_position=face_box.position
            )

    def _compute_gameplay_region(
        self,
        facecam: FacecamRegion,
        video_w: int,
        video_h: int
    ) -> LayoutCoordinates:
        """
        Determine the gameplay bounding box by excluding the streamer facecam overlay
        to prevent duplicate webcam artifacts in the gameplay portion.
        """
        pos = facecam.position
        min_gw = max(100, int(video_w * 0.35))

        if pos in ("left", "top_left", "bottom_left"):
            # Facecam is on left; gameplay takes the clean right portion
            gx = min(video_w - min_gw, facecam.x + facecam.w)
            gw = video_w - gx
            return LayoutCoordinates(x=gx, y=0, w=gw, h=video_h).clamp(video_w, video_h)

        elif pos in ("right", "top_right", "bottom_right"):
            # Facecam is on right; gameplay takes the clean left portion
            gw = max(min_gw, facecam.x)
            return LayoutCoordinates(x=0, y=0, w=gw, h=video_h).clamp(video_w, video_h)

        else:
            # Center overlay or unknown: take the main 16:9 gameplay canvas
            return LayoutCoordinates(x=0, y=0, w=video_w, h=video_h).clamp(video_w, video_h)
