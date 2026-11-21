"""
AI-YouTube-Clipper - Professional Dark Streamlit Dashboard
Automated AI short-form video editor with manual gaming split-screen reframing and animated subtitles.
"""

from pathlib import Path
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Set up app logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AI-YouTube-Clipper.app")

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    AppConfig,
    FFMPEG_PATH,
    FFPROBE_PATH,
    OUTPUT_CLIPS_DIR,
    PROJECT_ROOT,
    WhisperConfig,
)
from src.downloader import VideoDownloader
from src.exporter import ClipExporter
from src.face_detector import FaceDetector, FacecamRegion
from src.gaming_layout import draw_detection_overlay, generate_split_preview_image
from src.moment_detector import OllamaClient
from src.pipeline import ClipperPipeline, PipelineProgress
from src.utils import load_json, seconds_to_display_time

# Page Configuration
st.set_page_config(
    page_title="AI-YouTube-Clipper",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PREMIUM HIGH-CONTRAST DARK THEME STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Global Typography & Palette */
    :root {
        --bg-main: #0B0F19;
        --bg-sidebar: #111827;
        --bg-card: #1F2937;
        --border-card: #374151;
        --text-primary: #F9FAFB;
        --text-secondary: #E5E7EB;
        --text-muted: #D1D5DB;
        --accent-purple: #8B5CF6;
        --accent-blue: #3B82F6;
        --success: #22C55E;
        --warning: #F59E0B;
        --error: #EF4444;
    }

    /* Main Container */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F9FAFB !important;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
        font-size: 15px !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h1 { font-size: 2.2rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.6rem !important; margin-top: 1.2rem !important; }
    h3 { font-size: 1.25rem !important; margin-top: 1rem !important; }

    /* Paragraphs and normal text */
    p, span, label, li {
        color: #E5E7EB !important;
        font-size: 14.5px !important;
        line-height: 1.55 !important;
    }

    /* Secondary Descriptions and Captions */
    small, .caption, div[data-testid="stCaptionContainer"] p {
        color: #D1D5DB !important;
        font-size: 13.5px !important;
        font-weight: 400 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #374151 !important;
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F9FAFB !important;
    }

    /* Navigation Radio Items in Sidebar */
    div[data-testid="stRadio"] > label {
        color: #F9FAFB !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        background-color: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        background-color: #2D3748 !important;
        border-color: #8B5CF6 !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.35) 0%, rgba(59, 130, 246, 0.2) 100%) !important;
        border-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Cards */
    .dashboard-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        transition: border-color 0.2s ease;
    }
    .dashboard-card:hover {
        border-color: #4B5563;
    }

    /* Metric Badges in Dashboard */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #A78BFA;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #D1D5DB;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    /* Score Badges */
    .score-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-high {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
    }
    .score-med {
        background-color: #78350F;
        color: #FBBF24;
        border: 1px solid #D97706;
    }
    .score-low {
        background-color: #374151;
        color: #E5E7EB;
        border: 1px solid #4B5563;
    }

    /* Form Inputs & Dropdowns */
    input, textarea, select {
        background-color: #1F2937 !important;
        color: #F9FAFB !important;
        border: 1px solid #4B5563 !important;
        border-radius: 8px !important;
        font-size: 14.5px !important;
        padding: 10px 12px !important;
    }
    div[data-baseweb="input"] {
        background-color: #1F2937 !important;
        border: 1px solid #4B5563 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.4) !important;
    }
    div[data-baseweb="select"] {
        background-color: #1F2937 !important;
        border: 1px solid #4B5563 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        background-color: #1F2937 !important;
        color: #F9FAFB !important;
    }

    /* Labels */
    label[data-testid="stWidgetLabel"] p {
        color: #F9FAFB !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }

    /* Primary Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 0.65rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        opacity: 0.95 !important;
    }

    /* Secondary / Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background-color: #1F2937 !important;
        border: 1px solid #3B82F6 !important;
        color: #F9FAFB !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }

    /* Checkboxes */
    div[data-testid="stCheckbox"] label span {
        color: #F9FAFB !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    /* Instruction Alert Box */
    .instruction-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid #374151;
        border-left: 5px solid #8B5CF6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #D1D5DB !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #F9FAFB !important;
        border-bottom-color: #8B5CF6 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Exporter
exporter = ClipExporter()


def get_all_projects_summary() -> List[Dict[str, Any]]:
    """Scan output folder for existing project metadata."""
    projects = []
    if OUTPUT_CLIPS_DIR.exists():
        for proj_folder in sorted(OUTPUT_CLIPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if proj_folder.is_dir():
                meta_file = proj_folder / "project_summary.json"
                if meta_file.exists():
                    data = load_json(meta_file)
                    if data:
                        projects.append(data)
                else:
                    mp4s = list(proj_folder.glob("*.mp4"))
                    if mp4s:
                        projects.append({
                            "project_name": proj_folder.name,
                            "clip_count": len(mp4s),
                            "clips": [{"output_path": str(p), "title": p.stem} for p in mp4s]
                        })
    return projects


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluent/96/video-editing.png", width=64)
st.sidebar.title("AI-YouTube-Clipper")
st.sidebar.caption("High-Performance Local AI Video Editor")

menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "⚡ Create Clips", "⏳ Processing Queue", "🎞️ My Clips", "⚙️ Settings"],
    index=1
)

st.sidebar.divider()
st.sidebar.markdown(f"**FFmpeg:** `{Path(FFMPEG_PATH).stem}` (Ready)")
st.sidebar.markdown(f"**Storage Directory:** `{OUTPUT_CLIPS_DIR.name}/`")


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if menu == "📊 Dashboard":
    st.title("Project Dashboard")
    st.write("Overview of ingested gaming streams, generated clips, and local processing engines.")

    projects = get_all_projects_summary()
    total_clips = sum(p.get("clip_count", 0) for p in projects)
    total_videos = len(projects)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-label">Videos Ingested</div>
            <div class="metric-value">{total_videos}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-label">Clips Generated</div>
            <div class="metric-value">{total_clips}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="metric-label">Transcription Engine</div>
            <div class="metric-value" style="color: #34D399; font-size: 1.5rem; padding-top: 6px;">faster-whisper</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ollama_active = OllamaClient().is_available()
        status_color = "#34D399" if ollama_active else "#9CA3AF"
        status_text = "Online" if ollama_active else "Offline (Rule Engine Active)"
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="metric-label">Ollama LLM</div>
            <div class="metric-value" style="color: {status_color}; font-size: 1.25rem; padding-top: 8px;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Recent Projects")
    if projects:
        for p in projects[:6]:
            p_name = p.get("project_name", "Untitled")
            p_clips = p.get("clip_count", 0)
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                st.markdown(f"📁 **{p_name}**")
            with col_b:
                st.markdown(f"**{p_clips} clips**")
            with col_c:
                st.markdown("<span class='score-badge score-high'>Completed</span>", unsafe_allow_html=True)
            st.divider()
    else:
        st.info("No projects created yet. Go to **Create Clips** to process your first video!")


# ==============================================================================
# PAGE 2: CREATE CLIPS
# ==============================================================================
elif menu == "⚡ Create Clips":
    st.title("Create AI Shorts & Gaming Clips")
    st.write("Extract viral moments with manual gaming split-screen reframing and optional animated subtitles.")

    tab_url, tab_upload = st.tabs(["🔗 YouTube URL", "📁 Upload Local Video"])
    video_source: Optional[str] = None

    with tab_url:
        yt_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any authorized YouTube URL."
        )
        if yt_url:
            video_source = yt_url

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=["mp4", "mov", "mkv", "webm"],
            help="Upload your recorded gameplay, stream, or podcast video."
        )
        if uploaded_file is not None:
            save_upload_path = PROJECT_ROOT / "data" / "downloads" / uploaded_file.name
            save_upload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_source = str(save_upload_path)
            st.success(f"Uploaded: {uploaded_file.name}")

    st.divider()
    st.subheader("⚙️ Processing Configuration")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        num_clips = st.slider("Target Number of Clips", min_value=1, max_value=10, value=5)
        duration_range = st.slider(
            "Clip Duration Range (Seconds)",
            min_value=15,
            max_value=90,
            value=(20, 60),
            help="Standard 20s to 60s length for YouTube Shorts & TikTok."
        )
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium"],
            index=2,
            help="'small' provides optimal speed and high accuracy on local hardware."
        )

    with col_cfg2:
        layout_option = st.selectbox(
            "Video Layout & Reframing Mode",
            [
                "Streamer Top (45%) / Gameplay Bottom (55%) [MANUAL REGIONS - RECOMMENDED]",
                "Gameplay Top (55%) / Streamer Bottom (45%)",
                "Speaker Focus Full Screen (9:16 Crop)",
                "Original Layout (Blur Background 9:16)"
            ],
            index=0
        )

        enable_subtitles = st.checkbox(
            "Enable Hormozi Animated Subtitles",
            value=True,
            help="When checked, generates bold animated word-by-word subtitles. When unchecked, exports clean video without captions."
        )

        if enable_subtitles:
            sub_font_size = st.slider("Subtitle Font Size", min_value=32, max_value=72, value=48)
            sub_margin = st.slider("Subtitle Bottom Margin", min_value=180, max_value=450, value=280)
        else:
            sub_font_size = 48
            sub_margin = 280
            st.info("ℹ️ **Export Mode:** Clean Video (Subtitles Disabled).")

    # ==========================================================================
    # MANUAL STREAMER FACECAM & GAMEPLAY REGION SELECTOR
    # ==========================================================================
    st.divider()
    st.subheader("🎮 Streamer Facecam & Gameplay Region Inspector")

    st.markdown("""
    <div class="instruction-card">
        <h4 style="margin-top: 0; color: #F9FAFB;">📋 Streamer & Gameplay Crop Selection Instructions</h4>
        <ol style="margin-bottom: 0; padding-left: 20px;">
            <li><b>Select a Layout Preset</b> below or manually enter crop coordinates.</li>
            <li>Verify that the <span style="color: #FF00DC; font-weight: bold;">🟪 STREAMER FACECAM</span> box isolates the complete webcam (head, upper body, shoulders, mic).</li>
            <li>Verify that the <span style="color: #00E6FF; font-weight: bold;">🟦 GAMEPLAY REGION</span> box frames the main game action cleanly.</li>
            <li>Inspect the live <b>9:16 Vertical Preview</b> on the right to see the exact composition before exporting.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Determine reference video for preview
    local_preview_path: Optional[Path] = None
    if video_source and Path(video_source).is_file():
        local_preview_path = Path(video_source)
    else:
        dl_files = list((PROJECT_ROOT / "data" / "downloads").glob("*.mp4"))
        if dl_files:
            local_preview_path = dl_files[0]

    # Timestamp scrubber to extract frame where streamer is clearly visible
    col_scrub1, col_scrub2 = st.columns([3, 1])
    with col_scrub1:
        scrub_timestamp = st.slider(
            "Preview Frame Timestamp (Seconds)",
            min_value=0.0,
            max_value=120.0,
            value=3.0,
            step=1.0,
            help="Scrub to a video timestamp where the streamer's facecam is clearly visible."
        )
    with col_scrub2:
        st.write("")
        st.caption(f"Active Preview: `{seconds_to_display_time(scrub_timestamp)}`")

    preview_frame = None
    if local_preview_path and local_preview_path.exists():
        preview_frame = FaceDetector().extract_preview_frame(local_preview_path, timestamp_sec=scrub_timestamp)

    face_box_tuple: Optional[Tuple[int, int, int, int]] = None
    game_box_tuple: Optional[Tuple[int, int, int, int]] = None

    if preview_frame is not None:
        p_h, p_w = preview_frame.shape[:2]

        # Preset selection
        preset_choice = st.selectbox(
            "📐 Stream Layout Presets",
            [
                "Left Sidebar Facecam (Kai Cenat / Twitch Standard) [RECOMMENDED DEFAULT]",
                "Top-Left Corner Overlay",
                "Top-Right Corner Overlay",
                "Bottom-Left Corner Overlay",
                "Bottom-Right Corner Overlay",
                "Full 16:9 Screen Gameplay (No Webcam)",
                "Custom Freeform Coordinates",
                "Suggest from Video (Optional AI Hint)"
            ],
            index=0
        )

        # Calculate initial coordinates based on video dimensions and preset
        if preset_choice == "Left Sidebar Facecam (Kai Cenat / Twitch Standard) [RECOMMENDED DEFAULT]":
            def_fw = int(p_w * 0.28)
            def_fh = int(def_fw / 1.25)
            def_fx = 0
            def_fy = int(p_h * 0.10)
            def_gx = def_fw
            def_gy = 0
            def_gw = p_w - def_fw
            def_gh = p_h
        elif preset_choice == "Top-Left Corner Overlay":
            def_fw = int(p_w * 0.32)
            def_fh = int(p_h * 0.38)
            def_fx = 0
            def_fy = 0
            def_gx = int(p_w * 0.25)
            def_gy = 0
            def_gw = int(p_w * 0.75)
            def_gh = p_h
        elif preset_choice == "Top-Right Corner Overlay":
            def_fw = int(p_w * 0.32)
            def_fh = int(p_h * 0.38)
            def_fx = int(p_w * 0.68)
            def_fy = 0
            def_gx = 0
            def_gy = 0
            def_gw = int(p_w * 0.75)
            def_gh = p_h
        elif preset_choice == "Bottom-Left Corner Overlay":
            def_fw = int(p_w * 0.32)
            def_fh = int(p_h * 0.38)
            def_fx = 0
            def_fy = int(p_h * 0.62)
            def_gx = int(p_w * 0.25)
            def_gy = 0
            def_gw = int(p_w * 0.75)
            def_gh = p_h
        elif preset_choice == "Bottom-Right Corner Overlay":
            def_fw = int(p_w * 0.32)
            def_fh = int(p_h * 0.38)
            def_fx = int(p_w * 0.68)
            def_fy = int(p_h * 0.62)
            def_gx = 0
            def_gy = 0
            def_gw = int(p_w * 0.75)
            def_gh = p_h
        elif preset_choice == "Full 16:9 Screen Gameplay (No Webcam)":
            def_fx = 0
            def_fy = 0
            def_fw = p_w
            def_fh = int(p_w / 1.25)
            def_gx = 0
            def_gy = 0
            def_gw = p_w
            def_gh = p_h
        elif preset_choice == "Suggest from Video (Optional AI Hint)":
            detected_cam = FaceDetector().find_webcam_box_in_frame(preview_frame)
            if detected_cam and detected_cam.has_facecam and detected_cam.confidence >= 0.70:
                def_fx, def_fy, def_fw, def_fh = detected_cam.to_tuple()
                def_gx = min(p_w - 600, def_fx + def_fw) if def_fx < p_w // 2 else 0
                def_gw = p_w - def_gx if def_fx < p_w // 2 else max(600, def_fx)
                def_gy, def_gh = 0, p_h
                st.info(f"💡 AI Suggestion found stationary webcam at ({def_fx}, {def_fy}, {def_fw}, {def_fh}).")
            else:
                st.warning("⚠️ No stationary corner webcam identified (or in-game characters detected). Defaulting to standard left-sidebar preset.")
                def_fw = int(p_w * 0.28)
                def_fh = int(def_fw / 1.25)
                def_fx = 0
                def_fy = int(p_h * 0.10)
                def_gx = def_fw
                def_gy = 0
                def_gw = p_w - def_fw
                def_gh = p_h
        else:
            # Custom default
            def_fw = int(p_w * 0.28)
            def_fh = int(def_fw / 1.25)
            def_fx = 0
            def_fy = int(p_h * 0.10)
            def_gx = def_fw
            def_gy = 0
            def_gw = p_w - def_fw
            def_gh = p_h

        # Coordinate Controls
        with st.expander("🛠️ Fine-Tune Crop Coordinates (X, Y, Width, Height)", expanded=True):
            col_crop_f, col_crop_g = st.columns(2)
            with col_crop_f:
                st.markdown("##### 🟪 Streamer Facecam Region (Top 45%)")
                c_fx = st.number_input("Facecam X", 0, max(p_w - 20, 0), value=min(def_fx, p_w - 20), step=10, key="inp_fx")
                c_fy = st.number_input("Facecam Y", 0, max(p_h - 20, 0), value=min(def_fy, p_h - 20), step=10, key="inp_fy")
                c_fw = st.number_input("Facecam Width", 40, p_w, value=min(def_fw, p_w), step=10, key="inp_fw")
                c_fh = st.number_input("Facecam Height", 40, p_h, value=min(def_fh, p_h), step=10, key="inp_fh")
            with col_crop_g:
                st.markdown("##### 🟦 Gameplay Region (Bottom 55%)")
                c_gx = st.number_input("Gameplay X", 0, max(p_w - 20, 0), value=min(def_gx, p_w - 20), step=10, key="inp_gx")
                c_gy = st.number_input("Gameplay Y", 0, max(p_h - 20, 0), value=min(def_gy, p_h - 20), step=10, key="inp_gy")
                c_gw = st.number_input("Gameplay Width", 40, p_w, value=min(def_gw, p_w), step=10, key="inp_gw")
                c_gh = st.number_input("Gameplay Height", 40, p_h, value=min(def_gh, p_h), step=10, key="inp_gh")

        face_box_tuple = (int(c_fx), int(c_fy), int(c_fw), int(c_fh))
        game_box_tuple = (int(c_gx), int(c_gy), int(c_gw), int(c_gh))

        # Side-by-Side Visual Inspection & 9:16 Vertical Live Preview
        col_view1, col_view2 = st.columns([3, 2])
        with col_view1:
            st.markdown("##### 1. Source Video Frame with Selection Boxes")
            annotated_frame = draw_detection_overlay(preview_frame, face_box_tuple, game_box_tuple)
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Source Resolution: {p_w}×{p_h} | Facecam: {face_box_tuple} | Gameplay: {game_box_tuple}")

        with col_view2:
            st.markdown("##### 2. Live 9:16 Vertical Preview (1080×1920)")
            split_preview = generate_split_preview_image(preview_frame, face_box_tuple, game_box_tuple)
            st.image(cv2.cvtColor(split_preview, cv2.COLOR_BGR2RGB), width=300)
            st.caption("Top 45% Facecam (864px) / Bottom 55% Gameplay (1056px)")
    else:
        st.info("💡 Enter a YouTube URL or upload a local video above to preview and adjust streamer facecam / gameplay crops.")

    # Layout type mapping
    layout_map = {
        "Streamer Top (45%) / Gameplay Bottom (55%) [MANUAL REGIONS - RECOMMENDED]": "streamer_top_gameplay_bottom",
        "Gameplay Top (55%) / Streamer Bottom (45%)": "gameplay_top_streamer_bottom",
        "Speaker Focus Full Screen (9:16 Crop)": "speaker_focus",
        "Original Layout (Blur Background 9:16)": "original"
    }

    st.write("")
    start_btn = st.button("🚀 Start AI Video Clipping", use_container_width=True)

    if start_btn:
        if not video_source:
            st.error("Please enter a valid YouTube URL or upload a local video file.")
        else:
            progress_bar = st.progress(0, text="Initializing processing pipeline...")
            status_placeholder = st.empty()

            def _on_progress(p: PipelineProgress):
                progress_bar.progress(int(p.percent), text=f"[{p.stage}] {p.message}")
                status_placeholder.info(f"**Stage:** {p.stage} ({p.percent:.0f}%) — {p.message}")

            try:
                # Custom configuration
                cfg = AppConfig()
                cfg.whisper.model_size = whisper_model
                cfg.subtitles.font_size = sub_font_size
                cfg.subtitles.margin_v = sub_margin
                cfg.video.min_clip_duration = float(duration_range[0])
                cfg.video.max_clip_duration = float(duration_range[1])

                pipeline = ClipperPipeline(cfg)
                result = pipeline.process_video(
                    input_source=video_source,
                    target_clip_count=num_clips,
                    min_clip_duration=float(duration_range[0]),
                    max_clip_duration=float(duration_range[1]),
                    layout_type=layout_map[layout_option],
                    burn_subtitles=enable_subtitles,
                    manual_facecam=face_box_tuple,
                    manual_gameplay=game_box_tuple,
                    progress_callback=_on_progress
                )

                progress_bar.progress(100, text="Completed!")
                st.success("🎉 Video processing and vertical clipping completed successfully!")
                st.balloons()

                # Display Results
                st.subheader(f"Generated Clips ({len(result.get('clips', []))})")
                for clip in result.get("clips", []):
                    with st.expander(f"🎬 {clip.get('title')} ({clip.get('duration')}s) — Score: {clip.get('engagement_score')}/100", expanded=True):
                        col_v, col_m = st.columns([1, 1])
                        with col_v:
                            if Path(clip.get("output_path", "")).exists():
                                st.video(clip["output_path"])
                        with col_m:
                            st.write(f"**Viral Hook:** `{clip.get('hook')}`")
                            st.write(f"**Category:** `{clip.get('category')}`")
                            st.write(f"**File Size:** {clip.get('file_size_mb')} MB")
                            st.caption(f"**Transcript:** {clip.get('transcript_text')}")
                            with open(clip["output_path"], "rb") as f_clip:
                                st.download_button(
                                    label="⬇️ Download MP4",
                                    data=f_clip,
                                    file_name=clip.get("filename"),
                                    mime="video/mp4",
                                    key=f"dl_{clip.get('clip_id')}"
                                )

                # Full ZIP Archive Download
                zip_path = result.get("zip_path")
                if zip_path and Path(zip_path).exists():
                    with open(zip_path, "rb") as f_zip:
                        st.download_button(
                            label="📦 Download All Clips (ZIP)",
                            data=f_zip,
                            file_name=Path(zip_path).name,
                            mime="application/zip",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Processing error: {e}")
                logger.exception("Pipeline execution failed:")


# ==============================================================================
# PAGE 3: PROCESSING QUEUE
# ==============================================================================
elif menu == "⏳ Processing Queue":
    st.title("Processing Queue")
    st.write("Live status of running AI inference, audio extraction, and video render tasks.")

    st.markdown("""
    <div class="dashboard-card">
        <h4 style="margin-top: 0; color: #F9FAFB;">System Resource Allocation</h4>
        <p>• <b>AI Inference:</b> Sequential execution to prevent GPU VRAM exhaustion.</p>
        <p>• <b>Video Encoding:</b> Multi-threaded FFmpeg libx264 with optimized fast presets.</p>
        <p>• <b>Status:</b> <span class="score-badge score-high">System Ready</span></p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: MY CLIPS / GALLERY
# ==============================================================================
elif menu == "🎞️ My Clips":
    st.title("Clip Gallery & Exports")
    st.write("Preview, download, and manage your generated vertical 9:16 video clips.")

    col_act1, col_act2 = st.columns([4, 1])
    with col_act2:
        if st.button("📂 Open Output Folder"):
            exporter.open_output_folder()

    projects = get_all_projects_summary()

    if not projects:
        st.info("No clips exported yet. Use the **Create Clips** tab to generate your first clip package.")
    else:
        project_names = [p.get("project_name", "Untitled") for p in projects]
        selected_proj_name = st.selectbox("Select Project", project_names)

        selected_proj = next((p for p in projects if p.get("project_name") == selected_proj_name), None)

        if selected_proj:
            clips = selected_proj.get("clips", [])
            st.markdown(f"Showing **{len(clips)}** clips from project: `{selected_proj_name}`")

            # Full ZIP download button
            proj_dir = OUTPUT_CLIPS_DIR / selected_proj_name
            zip_files = list(proj_dir.glob("*.zip"))
            if zip_files and zip_files[0].exists():
                with open(zip_files[0], "rb") as fz:
                    st.download_button(
                        label="📦 Download Full Project ZIP",
                        data=fz,
                        file_name=zip_files[0].name,
                        mime="application/zip"
                    )

            st.write("")

            for clip in clips:
                c_path = Path(clip.get("output_path", ""))
                if c_path.exists():
                    st.markdown("""<div class="dashboard-card">""", unsafe_allow_html=True)
                    col_vid, col_info = st.columns([1, 1])
                    with col_vid:
                        st.video(str(c_path))
                    with col_info:
                        st.subheader(clip.get("title", c_path.stem))
                        st.markdown(f"**Viral Hook:** `{clip.get('hook', 'N/A')}`")
                        score = clip.get("engagement_score", 50.0)
                        score_class = "score-high" if score >= 70 else ("score-med" if score >= 45 else "score-low")
                        st.markdown(f"""Score: <span class="score-badge {score_class}">{score:.1f}/100</span>""", unsafe_allow_html=True)
                        st.write(f"**Duration:** {clip.get('duration', 'N/A')}s | **Size:** {clip.get('file_size_mb', 'N/A')} MB")
                        st.caption(f"**Transcript:** {clip.get('transcript_text', 'N/A')}")

                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            with open(c_path, "rb") as fp:
                                st.download_button(
                                    "⬇️ Download MP4",
                                    data=fp,
                                    file_name=c_path.name,
                                    mime="video/mp4",
                                    key=f"gal_dl_{c_path.stem}"
                                )
                        with col_b2:
                            if st.button("🗑️ Delete Clip", key=f"del_{c_path.stem}"):
                                exporter.delete_clip(c_path)
                                st.rerun()

                    st.markdown("""</div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: SETTINGS
# ==============================================================================
elif menu == "⚙️ Settings":
    st.title("Application Settings")

    st.subheader("Transcription Settings")
    st.selectbox("Default faster-whisper Model", ["tiny", "base", "small", "medium"], index=2)
    st.selectbox("Hardware Acceleration", ["Auto (CUDA GPU if available, else CPU)", "Force CPU int8", "Force CUDA float16"])

    st.subheader("Optional Ollama Integration")
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama Model Name", value="llama3.2")
    if st.button("Test Ollama Connection"):
        client = OllamaClient()
        if client.is_available():
            st.success("✅ Ollama is online and reachable!")
        else:
            st.warning("⚠️ Ollama is not running. Local rule-based engine will be used automatically (no error).")

    st.subheader("Directories & Storage")
    st.text_input("Output Directory", value=str(OUTPUT_CLIPS_DIR), disabled=True)
    st.caption("All processed clips, transcripts, and thumbnails are stored in this directory.")
