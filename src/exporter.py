"""
Export and project management module for AI-YouTube-Clipper.
Handles metadata packaging, thumbnail extraction, ZIP archiving, and gallery indexing.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union
import zipfile

from src.config import FFMPEG_PATH, OUTPUT_CLIPS_DIR, OUTPUT_DIR
from src.video_processor import ProcessedClipResult
from src.utils import run_ffmpeg, sanitize_filename, save_json

logger = logging.getLogger("AI-YouTube-Clipper.exporter")


class ClipExporter:
    """Manages exporting clips, metadata, thumbnails, and ZIP archives."""

    def __init__(self, base_output_dir: Optional[Path] = None):
        self.output_dir = Path(base_output_dir or OUTPUT_CLIPS_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_thumbnail(self, video_path: Path, output_thumb_path: Optional[Path] = None) -> Path:
        """Extract a high quality JPEG frame from the clip as a thumbnail."""
        video_path = Path(video_path).resolve()
        if output_thumb_path is None:
            output_thumb_path = video_path.with_suffix(".jpg")

        output_thumb_path = Path(output_thumb_path).resolve()
        output_thumb_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "-y",
            "-ss", "1.0",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_thumb_path)
        ]
        try:
            run_ffmpeg(cmd, executable=FFMPEG_PATH, description=f"Extract thumbnail for {video_path.name}")
        except Exception as e:
            logger.warning("Thumbnail extraction failed: %s. Using fallback frame.", e)
            # Try at 0.0 seconds
            cmd[2] = "0.0"
            try:
                run_ffmpeg(cmd, executable=FFMPEG_PATH, description=f"Extract thumbnail fallback {video_path.name}")
            except Exception:
                pass

        return output_thumb_path

    def package_project(
        self,
        project_name: str,
        clips: List[ProcessedClipResult],
        project_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Package a project's clips, generate thumbnails, create a ZIP file, and save metadata JSON.
        """
        project_dir = self.output_dir / sanitize_filename(project_name)
        project_dir.mkdir(parents=True, exist_ok=True)

        packaged_clips: List[Dict[str, Any]] = []
        clip_paths: List[Path] = []

        for clip in clips:
            thumb_path = self.generate_thumbnail(clip.output_path)
            clip_dict = clip.to_dict()
            clip_dict["thumbnail_path"] = str(thumb_path) if thumb_path.exists() else None
            packaged_clips.append(clip_dict)
            clip_paths.append(clip.output_path)

        # Save metadata JSON
        meta_file = project_dir / "project_summary.json"
        payload = {
            "project_name": project_name,
            "clip_count": len(packaged_clips),
            "clips": packaged_clips,
            "extra": project_metadata or {}
        }
        save_json(payload, meta_file)

        # Create ZIP bundle
        zip_file = project_dir / f"{sanitize_filename(project_name)}_all_clips.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for c_path in clip_paths:
                if c_path.exists():
                    zf.write(c_path, arcname=c_path.name)
            if meta_file.exists():
                zf.write(meta_file, arcname=meta_file.name)

        logger.info("Packaged project '%s' (%d clips, ZIP: %s)", project_name, len(clips), zip_file.name)
        return {
            "project_dir": str(project_dir),
            "metadata_path": str(meta_file),
            "zip_path": str(zip_file),
            "clips": packaged_clips
        }

    def list_exported_clips(self) -> List[Dict[str, Any]]:
        """List all MP4 clips present in the export directory."""
        results: List[Dict[str, Any]] = []
        if not self.output_dir.exists():
            return results

        mp4_files = sorted(self.output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        for mp4 in mp4_files:
            thumb = mp4.with_suffix(".jpg")
            results.append({
                "path": str(mp4),
                "name": mp4.name,
                "size_mb": round(mp4.stat().st_size / (1024 * 1024), 2),
                "modified": mp4.stat().st_mtime,
                "thumbnail": str(thumb) if thumb.exists() else None
            })
        return results

    def delete_clip(self, file_path: Union[str, Path]) -> bool:
        """Delete an exported clip and its thumbnail."""
        p = Path(file_path).resolve()
        if p.exists() and p.is_file():
            try:
                p.unlink()
                thumb = p.with_suffix(".jpg")
                if thumb.exists():
                    thumb.unlink()
                return True
            except Exception as e:
                logger.error("Failed to delete clip %s: %s", p, e)
                return False
        return False

    def open_output_folder(self) -> bool:
        """Open the output directory in Windows File Explorer."""
        try:
            os.startfile(str(self.output_dir))
            return True
        except Exception as e:
            logger.warning("Could not open output folder: %s", e)
            return False
