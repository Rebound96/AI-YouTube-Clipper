"""
Face detection and streamer facecam overlay tracking module using OpenCV.
Detects the complete rectangular webcam/facecam overlay (including head, shoulders,
and microphone) while isolating it from surrounding gameplay.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("AI-YouTube-Clipper.face_detector")


@dataclass
class FacecamRegion:
    """Represents an identified streamer facecam region in source video pixels."""
    x: int
    y: int
    w: int
    h: int
    position: str  # "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "center"
    confidence: float
    has_facecam: bool
    detection_method: str = "auto"  # "contour", "upper_body", "manual", "fallback"

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class FaceDetector:
    """Detects streamer faces and isolates the complete rectangular webcam overlay."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning("Failed to load OpenCV frontal face Haar cascade from %s", cascade_path)

    def detect_faces(
        self,
        frame: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        scale_factor: float = 1.15,
        min_neighbors: int = 5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single BGR video frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if self.face_cascade.empty() or frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def find_webcam_box_in_frame(
        self,
        frame: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[FacecamRegion]:
        """
        Identify the complete rectangular webcam overlay enclosing the streamer.
        Avoids capturing gameplay alongside the streamer.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        if face_box is None:
            faces = self.detect_faces(frame)
            if not faces:
                return None
            # Take largest face (primary streamer)
            face_box = max(faces, key=lambda b: b[2] * b[3])

        fx, fy, fw, fh = face_box
        cx = fx + fw // 2
        cy = fy + fh // 2
        pos = self._classify_position(cx, cy, w, h)

        # If face is in the central gameplay canvas and has no distinct contour border,
        # it is an in-game character, not a streamer overlay. Never assume central faces are streamers.
        if pos == "center":
            contour_box = self._find_contour_border(frame, fx, fy, fw, fh)
            if not contour_box:
                logger.info("Ignoring central face at (%d, %d); classified as in-game gameplay character.", cx, cy)
                return None

        # Strategy 1: Search for an OBS webcam border / window rectangle enclosing the face
        contour_box = self._find_contour_border(frame, fx, fy, fw, fh)
        if contour_box:
            bx, by, bw, bh = contour_box
            return FacecamRegion(
                x=bx,
                y=by,
                w=bw,
                h=bh,
                position=pos,
                confidence=0.90,
                has_facecam=True,
                detection_method="contour"
            )

        # Strategy 2: Tightly bounded upper-body streamer crop (head, hair, shoulders, mic)
        # Avoids grabbing arbitrary 30-50% width of the frame which includes gameplay!
        body_box = self._compute_upper_body_crop(cx, cy, fw, fh, w, h, pos)
        return FacecamRegion(
            x=body_box[0],
            y=body_box[1],
            w=body_box[2],
            h=body_box[3],
            position=pos,
            confidence=0.80,
            has_facecam=True,
            detection_method="upper_body"
        )

    def _find_contour_border(
        self,
        frame: np.ndarray,
        fx: int,
        fy: int,
        fw: int,
        fh: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Search for a rectangular overlay border around the detected face.
        """
        h, w = frame.shape[:2]

        # Search area: 2.5x face width, 3.5x face height around face
        roi_x1 = max(0, fx - int(fw * 0.8))
        roi_y1 = max(0, fy - int(fh * 0.6))
        roi_x2 = min(w, fx + int(fw * 1.8))
        roi_y2 = min(h, fy + int(fh * 2.8))

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)

        # Find external contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        face_area = fw * fh

        best_rect: Optional[Tuple[int, int, int, int]] = None
        best_area = 0

        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            rect_area = rw * rh

            # Must be larger than face, but reasonable webcam size
            if 1.8 * face_area <= rect_area <= 12.0 * face_area:
                # Convert back to full frame coordinates
                abs_x = roi_x1 + rx
                abs_y = roi_y1 + ry
                # Check that this rectangle actually encloses the face
                if abs_x <= fx and (abs_x + rw) >= (fx + fw) and abs_y <= fy:
                    aspect_ratio = rw / float(rh) if rh > 0 else 0
                    if 0.8 <= aspect_ratio <= 2.2:
                        if rect_area > best_area:
                            best_area = rect_area
                            best_rect = (abs_x, abs_y, rw, rh)

        return best_rect

    def _compute_upper_body_crop(
        self,
        cx: int,
        cy: int,
        fw: int,
        fh: int,
        video_w: int,
        video_h: int,
        position: str
    ) -> Tuple[int, int, int, int]:
        """
        Construct a clean upper-body webcam crop (head + torso + microphone)
        with an optimal aspect ratio (1.25 : 1, matching 1080:864 top container).
        Ensures NO adjacent gameplay is included.
        """
        # Shoulder width is typically 2.2x to 2.5x head width
        target_w = int(fw * 2.4)
        # Target height matching 1.25:1 aspect ratio (w/h = 1080/864 = 1.25)
        target_h = int(target_w / 1.25)

        # Make sure webcam isn't unrealistically small or huge
        target_w = max(target_w, int(video_w * 0.18))
        target_w = min(target_w, int(video_w * 0.45))
        target_h = int(target_w / 1.25)
        target_h = min(target_h, video_h)

        # Position box: give 25% headroom above face center, rest for shoulders/mic
        box_y = int(cy - target_h * 0.35)
        box_x = int(cx - target_w // 2)

        # Adjust for edge corners
        if "left" in position and box_x < 0:
            box_x = 0
        elif "right" in position and (box_x + target_w) > video_w:
            box_x = video_w - target_w

        if "top" in position and box_y < 0:
            box_y = 0
        elif "bottom" in position and (box_y + target_h) > video_h:
            box_y = video_h - target_h

        # Final bounds clamp
        final_x = max(0, min(box_x, video_w - target_w))
        final_y = max(0, min(box_y, video_h - target_h))

        return (final_x, final_y, target_w, target_h)

    def _classify_position(self, cx: int, cy: int, w: int, h: int) -> str:
        """Determine stream quadrant location."""
        rel_x = cx / float(w)
        rel_y = cy / float(h)

        if rel_x < 0.35:
            if rel_y < 0.35:
                return "top_left"
            elif rel_y > 0.65:
                return "bottom_left"
            return "left"
        elif rel_x > 0.65:
            if rel_y < 0.35:
                return "top_right"
            elif rel_y > 0.65:
                return "bottom_right"
            return "right"
        else:
            return "center"

    def analyze_video_facecam(
        self,
        video_path: Path,
        sample_interval_sec: float = 1.0,
        max_samples: int = 25
    ) -> FacecamRegion:
        """
        Sample video frames to aggregate face positions and return stable webcam bounds.
        """
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Could not open video file %s for facecam analysis.", video_path)
            return self._default_fallback_region(1920, 1080)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        step_frames = max(int(fps * sample_interval_sec), 1)
        total_samples = min(frame_count // step_frames if step_frames > 0 else 0, max_samples)

        detected_faces: List[Tuple[int, int, int, int]] = []
        samples_checked = 0

        for s in range(max(total_samples, 1)):
            frame_idx = s * step_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            samples_checked += 1
            faces = self.detect_faces(frame)
            if faces:
                largest = max(faces, key=lambda b: b[2] * b[3])
                detected_faces.append(largest)

        cap.release()

        if not detected_faces:
            logger.info("No face detected across %d samples. Using default left facecam layout.", samples_checked)
            return self._default_fallback_region(width, height)

        # Compute median face location
        xs = [f[0] for f in detected_faces]
        ys = [f[1] for f in detected_faces]
        std_x = float(np.std(xs)) if len(xs) > 1 else 0.0
        std_y = float(np.std(ys)) if len(ys) > 1 else 0.0

        med_x = int(np.median(xs))
        med_y = int(np.median(ys))
        med_w = int(np.median([f[2] for f in detected_faces]))
        med_h = int(np.median([f[3] for f in detected_faces]))

        cx = med_x + med_w // 2
        cy = med_y + med_h // 2
        pos = self._classify_position(cx, cy, width, height)
        confidence = len(detected_faces) / max(samples_checked, 1)

        # Architectural Guard: In gaming videos, characters move throughout the central world.
        # High positional variance (>8% of frame) or central position means moving gameplay characters.
        # We NEVER classify in-game characters as the streamer webcam.
        is_moving_characters = (std_x > width * 0.08) or (std_y > height * 0.08)
        is_center_gameplay = (pos == "center")

        if is_moving_characters or is_center_gameplay:
            logger.warning(
                "Face detections appear to be in-game characters (std_x=%.1f, std_y=%.1f, pos=%s) rather than a stationary streamer overlay. "
                "Disabling automatic facecam assumption; requiring manual selection.",
                std_x, std_y, pos
            )
            return FacecamRegion(
                x=0,
                y=int(height * 0.10),
                w=int(width * 0.28),
                h=int(width * 0.28 / 1.25),
                position="left",
                confidence=0.0,
                has_facecam=False,
                detection_method="gameplay_characters_detected"
            )

        body_box = self._compute_upper_body_crop(cx, cy, med_w, med_h, width, height, pos)

        logger.info(
            "Identified facecam position '%s' at %s (confidence: %.2f)",
            pos, body_box, confidence
        )

        return FacecamRegion(
            x=body_box[0],
            y=body_box[1],
            w=body_box[2],
            h=body_box[3],
            position=pos,
            confidence=round(confidence, 2),
            has_facecam=True,
            detection_method="upper_body"
        )

    def extract_preview_frame(
        self,
        video_path: Path,
        timestamp_sec: float = 3.0
    ) -> Optional[np.ndarray]:
        """Extract a single BGR frame from video at specified timestamp."""
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback to frame 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        cap.release()
        return frame if ret else None

    def _default_fallback_region(self, width: int, height: int) -> FacecamRegion:
        """Default fallback assuming left-side facecam overlay."""
        box_w = int(width * 0.28)
        box_h = int(box_w / 1.25)
        return FacecamRegion(
            x=0,
            y=int(height * 0.15),
            w=box_w,
            h=box_h,
            position="left",
            confidence=0.5,
            has_facecam=False,
            detection_method="fallback"
        )


class FaceTracker:
    """Smoothly tracks face positions over time to prevent jittery camera crops."""

    def __init__(self, smoothing_factor: float = 0.25):
        self.smoothing = smoothing_factor
        self.current_box: Optional[Tuple[float, float, float, float]] = None

    def update(self, detected_box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Update tracker with new detection and return smoothed box."""
        if detected_box is None:
            if self.current_box is None:
                return None
            return (
                int(round(self.current_box[0])),
                int(round(self.current_box[1])),
                int(round(self.current_box[2])),
                int(round(self.current_box[3]))
            )

        dx, dy, dw, dh = detected_box
        if self.current_box is None:
            self.current_box = (float(dx), float(dy), float(dw), float(dh))
        else:
            sx = self.smoothing * dx + (1.0 - self.smoothing) * self.current_box[0]
            sy = self.smoothing * dy + (1.0 - self.smoothing) * self.current_box[1]
            sw = self.smoothing * dw + (1.0 - self.smoothing) * self.current_box[2]
            sh = self.smoothing * dh + (1.0 - self.smoothing) * self.current_box[3]
            self.current_box = (sx, sy, sw, sh)

        return (
            int(round(self.current_box[0])),
            int(round(self.current_box[1])),
            int(round(self.current_box[2])),
            int(round(self.current_box[3]))
        )
