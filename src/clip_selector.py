"""
Clip selection and candidate curation module for AI-YouTube-Clipper.
Selects top candidate clips, ensures duration constraints, attaches viral hooks,
and formats jobs for video rendering.
"""

import logging
from typing import List, Optional

from src.hook_generator import HookGenerator
from src.moment_detector import CandidateClip, MomentDetector
from src.transcriber import TranscriptionResult

logger = logging.getLogger("AI-YouTube-Clipper.clip_selector")


class ClipSelector:
    """Orchestrates candidate selection and viral hook enrichment."""

    def __init__(
        self,
        min_duration: float = 20.0,
        max_duration: float = 60.0,
        hook_generator: Optional[HookGenerator] = None
    ):
        self.detector = MomentDetector(min_duration=min_duration, max_duration=max_duration)
        self.hook_generator = hook_generator or HookGenerator()

    def select_clips(
        self,
        transcription: TranscriptionResult,
        target_clip_count: int = 5,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> List[CandidateClip]:
        """
        Detect, rank, and enrich top candidate clips for production.
        """
        logger.info(
            "Selecting up to %d engaging clips (duration: %.0f-%.0fs)...",
            target_clip_count, min_duration or self.detector.min_duration, max_duration or self.detector.max_duration
        )

        candidates = self.detector.detect_moments(
            transcription=transcription,
            max_clips=target_clip_count,
            min_duration=min_duration,
            max_duration=max_duration
        )

        # Enrich each candidate with an attention-grabbing hook
        for clip in candidates:
            clip.hook = self.hook_generator.generate_hook(clip)

        logger.info("Selected %d candidate clips with viral hooks.", len(candidates))
        return candidates
