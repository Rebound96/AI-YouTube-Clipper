# AI-YouTube-Clipper 🎬⚡

A professional, local AI-powered video clipping and short-form video generator designed for YouTube Shorts, TikTok, and Instagram Reels. Includes specialized **Gaming Split-Screen Reframing** (Streamer Facecam Top 45% / Gameplay Bottom 55%), **Hormozi-style animated word-by-word subtitles**, **OpenCV face tracking**, and **automatic viral moment detection**.

---

## Key Capabilities

1. **YouTube & Local Video Ingestion:**
   * Download authorized YouTube videos in up to 1080p MP4 via `yt-dlp`.
   * Upload or ingest local `.mp4`, `.mov`, `.mkv`, and `.webm` files.
2. **Local Faster-Whisper Speech Transcription:**
   * Accelerated word-level timestamps with automatic language detection.
   * Auto GPU (CUDA) acceleration with smooth fallback to CPU `int8`.
   * Automatic transcript caching in JSON and SRT formats.
3. **AI Viral Moment Detection:**
   * Full transcript analysis (scans the complete timeline, not just the intro).
   * Identifies funny moments, unexpected reactions, screaming, clutch gaming plays, debates, and hot takes.
   * Context preservation: avoids cutting sentences, ensures setup precedes punchlines, and includes reactions.
   * Non-Maximum Suppression (NMS) prevents duplicate or overlapping clips.
   * Transparent 1–100 scoring system based on hook quality, content density, climax structure, and pacing.
4. **Advanced Gaming Split-Screen Editor (Kai Cenat / Twitch Style):**
   * **Top 45% (864px):** Streamer facecam.
   * **Bottom 55% (1056px):** Full-screen gameplay action.
   * **Output Canvas:** 1080x1920 vertical format.
   * Layouts:
     * Streamer Top / Gameplay Bottom (Default)
     * Gameplay Top / Streamer Bottom
     * Speaker Focus Full Screen (9:16 crop)
     * Original Video Layout (16:9 on blurred 9:16 background)
     * Automatic Layout
   * OpenCV Facecam Detection & Tracking across video frames.
   * Optional manual coordinates override `(x, y, w, h)`.
5. **Hormozi-Style Animated Subtitles:**
   * Dynamic word-by-word active highlighting (vivid yellow active word, white body text).
   * Thick black outline and drop shadow for maximum legibility.
   * Standard ASS subtitle burn-in via FFmpeg.
6. **Automatic Hook Generation:**
   * Generates catchy, context-accurate hooks with emojis (e.g. `HE REALLY DID THAT 💀`, `WAIT FOR HIS REACTION 😱`).
7. **Clip Gallery & One-Click ZIP Export:**
   * In-browser video preview and playback.
   * Individual MP4 downloads and complete project ZIP bundling.
   * Direct "Open Output Folder" button for Windows Explorer.

---

## Project Structure

```
AI-YouTube-Clipper/
├── .venv/                         # Python 3.11.9 virtual environment
├── app.py                         # Streamlit dark-theme dashboard application
├── requirements.txt               # Locked project dependencies
├── README.md                      # Documentation & user guide
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, models & directory setup
│   ├── utils.py                   # Subprocess FFmpeg wrapper, timestamps, JSON I/O
│   ├── downloader.py              # yt-dlp YouTube downloader & local ingestion
│   ├── audio.py                   # 16kHz mono audio extraction & ffprobe probe
│   ├── transcriber.py             # faster-whisper with word timestamps & cache
│   ├── moment_detector.py         # Viral moment detection & transparent scoring
│   ├── hook_generator.py          # Category-aligned short-form viral hooks
│   ├── face_detector.py           # OpenCV facecam detection & tracking
│   ├── gaming_layout.py           # Gaming split-screen crop & filter complex planner
│   ├── subtitle_generator.py      # Hormozi animated ASS subtitle generator
│   ├── video_processor.py         # FFmpeg video clipping & vertical rendering engine
│   ├── clip_selector.py           # Top moment filtering and job packaging
│   ├── exporter.py                # Thumbnail extraction, metadata, and ZIP exporter
│   └── pipeline.py                # End-to-end ClipperPipeline orchestrator
├── tests/
│   ├── test_config_utils.py       # Configuration and utility tests
│   ├── test_audio.py              # Audio extraction & ffprobe tests
│   ├── test_downloader.py         # Ingestion and metadata tests
│   ├── test_transcription.py      # Whisper data model & caching tests
│   ├── test_live_transcription.py # Live faster-whisper inference test
│   ├── test_moment_detector.py    # Moment detection, NMS & scoring tests
│   ├── test_gaming_layout.py      # Facecam layout & crop filter tests
│   ├── test_subtitles.py          # Hormozi ASS subtitle generation tests
│   └── test_video_processor.py    # Split-screen FFmpeg rendering integration tests
├── data/
│   ├── downloads/                 # Downloaded source videos
│   ├── audio/                     # Extracted 16kHz speech WAVs
│   ├── transcripts/               # Cached transcription JSONs
│   └── clips/                     # Temporary working files
└── output/
    └── clips/                     # Final exported 1080x1920 MP4 clips & ZIPs
```

---

## Quickstart & Launch Instructions

### 1. Launch the Streamlit Dashboard (Windows PowerShell)

```powershell
cd C:\Users\Admin\Documents\AI-YouTube-Clipper
.\.venv\Scripts\streamlit.exe run app.py
```

The application opens at: `http://localhost:8501`.

---

### 2. Running Automated Tests

Run the full pytest suite (28 comprehensive unit & integration tests):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Hardware & Acceleration

* **Optimized for:** NVIDIA RTX GPUs (CUDA) with 12GB+ VRAM & 16GB RAM.
* **CPU Fallback:** Fully supported on standard multi-core CPUs via `int8` quantization.
* **Optional LLM:** Compatible with local Ollama (`http://localhost:11434`, e.g. `llama3.2`). Completely optional; the built-in rule engine operates 100% offline without dependencies.
