"""
Video downloader and local file ingestion module.
Uses yt-dlp to download videos and extract rich metadata.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Optional

import yt_dlp

from src.config import DOWNLOADS_DIR, FFMPEG_PATH
from src.utils import sanitize_filename

logger = logging.getLogger("AI-YouTube-Clipper.downloader")


@dataclass
class VideoInfo:
    """Metadata describing an ingested or downloaded video."""
    video_id: str
    title: str
    duration: float
    file_path: Path
    uploader: str = "Unknown"
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    is_local: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert VideoInfo to dictionary representation."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "duration": self.duration,
            "file_path": str(self.file_path),
            "uploader": self.uploader,
            "thumbnail_url": self.thumbnail_url,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_local": self.is_local,
        }


class VideoDownloader:
    """Handles downloading YouTube videos via yt-dlp and ingesting local video files."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or DOWNLOADS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_info(self, url: str) -> Dict[str, Any]:
        """Extract metadata for a YouTube URL without downloading the media."""
        ydl_opts = {
            "extract_flat": False,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return info
            except Exception as e:
                logger.error("Failed to extract info from URL %s: %s", url, e)
                raise RuntimeError(f"Could not extract info from URL: {e}") from e

    def download_youtube_video(
        self,
        url: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_height: int = 1080
    ) -> VideoInfo:
        """
        Download a YouTube video at up to `max_height` resolution (MP4 container with AAC audio).
        """
        logger.info("Starting download for YouTube URL: %s", url)

        # Hook for yt-dlp progress
        def _hook(d):
            if progress_callback:
                progress_callback(d)

        # Output template with sanitized id/title
        outtmpl = str(self.output_dir / "%(id)s_%(title).60s.%(ext)s")

        ydl_opts = {
            # Prefer MP4 container with H.264/AAC for best FFmpeg compatibility
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "progress_hooks": [_hook],
            "quiet": False,
            "no_warnings": True,
            "ffmpeg_location": FFMPEG_PATH,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as e:
                logger.error("Download failed for %s: %s", url, e)
                raise RuntimeError(f"Failed to download video: {e}") from e

            # Get final downloaded filepath
            video_id = info.get("id", "unknown")
            downloaded_path = None
            if "requested_downloads" in info and info["requested_downloads"]:
                downloaded_path = Path(info["requested_downloads"][0]["filepath"])
            else:
                downloaded_path = Path(ydl.prepare_filename(info))
                # When merged into mp4, extension might change to .mp4
                if not downloaded_path.exists() and downloaded_path.with_suffix(".mp4").exists():
                    downloaded_path = downloaded_path.with_suffix(".mp4")

            if not downloaded_path or not downloaded_path.exists():
                # Search directory for file matching video_id
                matches = list(self.output_dir.glob(f"{video_id}*.*"))
                if matches:
                    downloaded_path = matches[0]
                else:
                    raise FileNotFoundError(f"Downloaded file not found for video id {video_id}")

            return VideoInfo(
                video_id=video_id,
                title=info.get("title", "Untitled Video"),
                duration=float(info.get("duration", 0)),
                file_path=downloaded_path,
                uploader=info.get("uploader", "Unknown"),
                thumbnail_url=info.get("thumbnail"),
                description=info.get("description"),
                width=info.get("width"),
                height=info.get("height"),
                fps=info.get("fps"),
                is_local=False,
            )

    def ingest_local_file(self, file_path: Path, copy_to_downloads: bool = False) -> VideoInfo:
        """
        Ingest a local video file, verifying its existence and probing dimensions and duration.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local video file not found: {path}")

        target_path = path
        if copy_to_downloads and path.parent != self.output_dir:
            dest = self.output_dir / path.name
            shutil.copy2(path, dest)
            target_path = dest

        # Use ffprobe to get details
        from src.audio import probe_media
        media_info = probe_media(target_path)

        duration = float(media_info.get("format", {}).get("duration", 0.0))
        video_stream = None
        for stream in media_info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        width = int(video_stream.get("width")) if video_stream and "width" in video_stream else None
        height = int(video_stream.get("height")) if video_stream and "height" in video_stream else None
        fps = None
        if video_stream and "r_frame_rate" in video_stream:
            try:
                num, den = map(int, video_stream["r_frame_rate"].split("/"))
                fps = round(num / den, 2) if den != 0 else None
            except Exception:
                fps = None

        video_id = sanitize_filename(path.stem)
        return VideoInfo(
            video_id=video_id,
            title=path.stem,
            duration=duration,
            file_path=target_path,
            uploader="Local File",
            thumbnail_url=None,
            description="Uploaded local video file",
            width=width,
            height=height,
            fps=fps,
            is_local=True,
        )
