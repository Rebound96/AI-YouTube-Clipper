"""
Transcription module powered by faster-whisper.
Produces word-level timestamps, segment summaries, and caching.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

from faster_whisper import WhisperModel

from src.config import TRANSCRIPTS_DIR, WhisperConfig
from src.utils import format_srt_timestamp, load_json, sanitize_filename, save_json

logger = logging.getLogger("AI-YouTube-Clipper.transcriber")


def setup_cuda_dll_paths():
    """Register nvidia CUDA and cuBLAS DLL directories with the Windows loader."""
    if os.name == "nt":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
        if site_packages.exists():
            for p in site_packages.glob("nvidia/*/bin"):
                if p.is_dir():
                    try:
                        os.add_dll_directory(str(p.resolve()))
                        os.environ["PATH"] = str(p.resolve()) + ";" + os.environ.get("PATH", "")
                    except Exception:
                        pass


# Register CUDA DLLs on import
setup_cuda_dll_paths()


def is_cuda_usable() -> bool:
    """Check if CUDA device is present and runtime cuBLAS libraries can be loaded."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if os.name == "nt":
            import ctypes
            setup_cuda_dll_paths()
            try:
                ctypes.CDLL("cublas64_12.dll")
                return True
            except Exception:
                try:
                    ctypes.CDLL("cublas64_11.dll")
                    return True
                except Exception:
                    return False
        return True
    except Exception:
        return False


@dataclass
class WordTiming:
    """Represents a single word with precise start and end timestamps."""
    word: str
    start: float
    end: float
    probability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTiming":
        return cls(
            word=data["word"],
            start=float(data["start"]),
            end=float(data["end"]),
            probability=float(data.get("probability", 1.0))
        )


@dataclass
class TranscriptSegment:
    """Represents a speech segment with sentence-level text and word timings."""
    id: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            words=[WordTiming.from_dict(w) for w in data.get("words", [])],
            avg_logprob=float(data.get("avg_logprob", 0.0)),
            no_speech_prob=float(data.get("no_speech_prob", 0.0))
        )


@dataclass
class TranscriptionResult:
    """Complete result of a transcription run."""
    segments: List[TranscriptSegment]
    language: str
    language_probability: float
    duration: float
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        return cls(
            language=data.get("language", "en"),
            language_probability=float(data.get("language_probability", 1.0)),
            duration=float(data.get("duration", 0.0)),
            full_text=data.get("full_text", ""),
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])]
        )

    def to_srt(self) -> str:
        """Export transcription result to standard SRT format."""
        srt_lines = []
        for i, seg in enumerate(self.segments, 1):
            start = format_srt_timestamp(seg.start)
            end = format_srt_timestamp(seg.end)
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
        return "\n".join(srt_lines)

    def get_words_in_range(self, start_sec: float, end_sec: float) -> List[WordTiming]:
        """Retrieve all words that fall within a specified time range."""
        matching_words: List[WordTiming] = []
        for seg in self.segments:
            if seg.end < start_sec or seg.start > end_sec:
                continue
            for w in seg.words:
                if w.start >= start_sec - 0.2 and w.end <= end_sec + 0.2:
                    matching_words.append(w)
        return matching_words


class Transcriber:
    """Wrapper around faster-whisper model with intelligent fallback and caching."""

    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._model: Optional[WhisperModel] = None

    def _determine_device_and_compute(self) -> tuple[str, str]:
        """Determine appropriate device and compute type with fallback logic."""
        device = self.config.device.lower()
        compute = self.config.compute_type.lower()

        if device == "auto":
            if is_cuda_usable():
                device = "cuda"
                if compute == "auto":
                    compute = "float16"
            else:
                device = "cpu"
                if compute == "auto":
                    compute = "int8"
        elif device == "cuda":
            if not is_cuda_usable():
                logger.warning("CUDA requested but cuBLAS DLL is unavailable. Falling back to CPU int8.")
                device = "cpu"
                compute = "int8"
            elif compute == "auto":
                compute = "float16"
        elif device == "cpu" and compute == "auto":
            compute = "int8"

        return device, compute

    def load_model(self) -> WhisperModel:
        """Load and cache the WhisperModel instance."""
        if self._model is not None:
            return self._model

        device, compute_type = self._determine_device_and_compute()
        logger.info(
            "Loading faster-whisper model '%s' on %s (compute_type=%s)...",
            self.config.model_size, device, compute_type
        )

        try:
            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as e:
            logger.warning("Failed to load model on %s: %s. Falling back to CPU int8.", device, e)
            self._model = WhisperModel(
                self.config.model_size,
                device="cpu",
                compute_type="int8",
            )

        return self._model

    def transcribe(
        self,
        audio_path: Union[str, Path],
        output_json_path: Optional[Union[str, Path]] = None,
        cache: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe an audio file and extract sentence segments and word-level timestamps.
        """
        if output_json_path is None:
            audio_path_obj = Path(audio_path).resolve()
            stem = sanitize_filename(audio_path_obj.stem.replace("_16k", ""))
            output_json_path = TRANSCRIPTS_DIR / f"{stem}_transcript.json"

        output_json_path = Path(output_json_path).resolve()

        # Check cache first
        if cache and output_json_path.is_file():
            cached_data = load_json(output_json_path)
            if cached_data and "segments" in cached_data:
                logger.info("Loaded cached transcript from: %s", output_json_path)
                return TranscriptionResult.from_dict(cached_data)

        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.load_model()
        logger.info("Transcribing audio file: %s", audio_path.name)

        segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                beam_size=self.config.beam_size,
                language=self.config.language,
                word_timestamps=self.config.word_timestamps,
                vad_filter=self.config.vad_filter,
            )

            for seg in segments_iter:
                words: List[WordTiming] = []
                if seg.words:
                    for w in seg.words:
                        words.append(
                            WordTiming(
                                word=w.word,
                                start=round(w.start, 3),
                                end=round(w.end, 3),
                                probability=round(w.probability, 3)
                            )
                        )

                transcript_seg = TranscriptSegment(
                    id=seg.id,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    words=words,
                    avg_logprob=round(seg.avg_logprob, 3),
                    no_speech_prob=round(seg.no_speech_prob, 3)
                )
                segments.append(transcript_seg)
                full_text_parts.append(transcript_seg.text)

        except Exception as err:
            err_str = str(err).lower()
            if ("cublas" in err_str or "cudnn" in err_str or "cuda" in err_str) and (self.config.device != "cpu"):
                logger.warning(
                    "CUDA inference failed (%s). Automatically falling back to CPU int8...",
                    err
                )
                self._model = WhisperModel(
                    self.config.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                segments_iter, info = self._model.transcribe(
                    str(audio_path),
                    beam_size=self.config.beam_size,
                    language=self.config.language,
                    word_timestamps=self.config.word_timestamps,
                    vad_filter=self.config.vad_filter,
                )
                segments = []
                full_text_parts = []
                for seg in segments_iter:
                    words = []
                    if seg.words:
                        for w in seg.words:
                            words.append(
                                WordTiming(
                                    word=w.word,
                                    start=round(w.start, 3),
                                    end=round(w.end, 3),
                                    probability=round(w.probability, 3)
                                )
                            )
                    transcript_seg = TranscriptSegment(
                        id=seg.id,
                        start=round(seg.start, 3),
                        end=round(seg.end, 3),
                        text=seg.text.strip(),
                        words=words,
                        avg_logprob=round(seg.avg_logprob, 3),
                        no_speech_prob=round(seg.no_speech_prob, 3)
                    )
                    segments.append(transcript_seg)
                    full_text_parts.append(transcript_seg.text)
            else:
                raise

        result = TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 3),
            full_text=" ".join(full_text_parts)
        )

        # Save to cache
        save_json(result.to_dict(), output_json_path)
        logger.info("Transcription completed (%d segments). Cached to: %s", len(segments), output_json_path)

        return result
