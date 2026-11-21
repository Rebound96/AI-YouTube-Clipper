"""
Automatic interesting and viral moment detection module for AI-YouTube-Clipper.
Analyzes full transcription results to detect high-engagement segments using
transparent heuristic scoring and optional Ollama LLM enhancement.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from src.config import CLIPS_DIR, OllamaConfig
from src.transcriber import TranscriptSegment, TranscriptionResult
from src.utils import load_json, sanitize_filename, save_json, seconds_to_display_time

logger = logging.getLogger("AI-YouTube-Clipper.moment_detector")


# Pattern dictionaries mapping categories to high-engagement keyword triggers
CATEGORY_PATTERNS = {
    "funny": [
        r"\b(?:haha|hahaha|lmao|lmfao|rofl|hilarious|laughing|cracking up|joke|ridiculous|clown)\b",
        r"\[laughter\]|\[laughs\]",
    ],
    "surprise": [
        r"\b(?:no way|what the hell|what the heck|are you serious|are you kidding|i can't believe|i cannot believe|holy crap|holy cow|holy shit|mind blown|mind-blowing|wait what|insane|unbelievable)\b",
    ],
    "argument": [
        r"\b(?:you're wrong|you are wrong|i disagree|that makes no sense|bullshit|stop lying|don't lie|shut up|shut your|liar|nonsense|prove it|fight me|cut it out)\b",
    ],
    "strong_opinion": [
        r"\b(?:the truth is|hot take|nobody talks about|biggest mistake|worst mistake|the secret to|never do this|always remember|the problem with|listen to me|the reality is|unpopular opinion|mark my words)\b",
    ],
    "gaming_reaction": [
        r"\b(?:let's go|lets go|clutch|headshot|no scope|we win|pog|poggers|get wrecked|get good|watch this|did you see that|epic play|game over|run run|behind you)\b",
    ],
    "emotional": [
        r"\b(?:i cried|brought tears|broke my heart|so grateful|never felt so|most touching|changed my life|heartbreaking|devastating|deepest regret|forever changed)\b",
    ],
    "question_curiosity": [
        r"\b(?:did you know|have you ever wondered|what if|how is it possible|why does everyone|can someone explain|guess what|the real reason why)\b",
    ],
}

# Hook starters that indicate strong opening sentences
HOOK_STARTERS = [
    r"^(?:did you know|what if|have you ever|the biggest mistake|here is why|stop doing|this is how|nobody knows|i bet you|listen up|watch what happens)",
    r"^(?:you won't believe|never in my life|i was completely wrong|if you're doing this|the craziest thing)",
]


@dataclass
class CandidateClip:
    """A detected candidate clip representing an engaging moment."""
    clip_id: str
    start_time: float
    end_time: float
    duration: float
    title: str
    hook: str
    engagement_score: float  # Scale 1 - 100
    category: str
    reason: str
    transcript_text: str
    segment_ids: List[int]
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "start_display": seconds_to_display_time(self.start_time),
            "end_display": seconds_to_display_time(self.end_time),
            "title": self.title,
            "hook": self.hook,
            "engagement_score": round(self.engagement_score, 1),
            "category": self.category,
            "reason": self.reason,
            "transcript_text": self.transcript_text,
            "segment_ids": self.segment_ids,
            "score_breakdown": {k: round(v, 2) for k, v in self.score_breakdown.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateClip":
        return cls(
            clip_id=data["clip_id"],
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            duration=float(data.get("duration", data["end_time"] - data["start_time"])),
            title=data.get("title", "Untitled Moment"),
            hook=data.get("hook", ""),
            engagement_score=float(data.get("engagement_score", 50.0)),
            category=data.get("category", "highlight"),
            reason=data.get("reason", "Detected as high engagement candidate"),
            transcript_text=data.get("transcript_text", ""),
            segment_ids=data.get("segment_ids", []),
            score_breakdown=data.get("score_breakdown", {}),
        )


class OllamaClient:
    """Optional client for interacting with a local Ollama instance."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def enhance_moment(self, clip: CandidateClip) -> CandidateClip:
        """
        Optionally refine title and hook with Ollama if running.
        Falls back seamlessly if Ollama fails or times out.
        """
        if not self.config.enabled or not self.is_available():
            return clip

        prompt = (
            f"You are a short-form video editor for YouTube Shorts and TikTok.\n"
            f"Given this transcript clip ({clip.duration:.0f} seconds):\n"
            f"\"{clip.transcript_text}\"\n\n"
            f"Return JSON strictly with format:\n"
            f'{{"title": "Punchy Title Under 6 Words", "hook": "First 1-2 sentence hook", "reason": "Why viewers will watch"}}\n'
            f"Do not include any other markdown or text."
        )

        try:
            resp = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=self.config.timeout
            )
            if resp.status_code == 200:
                result_json = resp.json().get("response", "{}")
                parsed = json.loads(result_json)
                if "title" in parsed and parsed["title"]:
                    clip.title = parsed["title"].strip('" ')
                if "hook" in parsed and parsed["hook"]:
                    clip.hook = parsed["hook"].strip('" ')
                if "reason" in parsed and parsed["reason"]:
                    clip.reason = f"[AI-Ollama] {parsed['reason']}"
        except Exception as e:
            logger.warning("Ollama enhancement failed, preserving heuristic metadata: %s", e)

        return clip


class MomentDetector:
    """
    Intelligent moment detector for identifying engaging short-form candidates
    across full transcriptions with context preservation and duplicate suppression.
    """

    def __init__(
        self,
        min_duration: float = 20.0,
        max_duration: float = 60.0,
        ollama_config: Optional[OllamaConfig] = None
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.ollama = OllamaClient(ollama_config)

    def _score_segment_text(self, text: str) -> Tuple[float, str, List[str]]:
        """
        Score a text snippet based on keyword categories, intensity, and exclamation marks.
        Returns: (keyword_score, primary_category, matched_reasons)
        """
        lower = text.lower()
        score = 0.0
        matched_categories: Dict[str, int] = {}
        matched_reasons: List[str] = []

        for category, patterns in CATEGORY_PATTERNS.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, lower, re.IGNORECASE)
                if matches:
                    count += len(matches)
            if count > 0:
                matched_categories[category] = count
                score += count * 15.0
                matched_reasons.append(f"{category} ({count} match{'es' if count > 1 else ''})")

        # Punctuation & screaming emphasis
        exclamations = text.count("!")
        if exclamations > 0:
            score += min(exclamations * 5.0, 20.0)
            matched_reasons.append(f"Exclamation intensity ({exclamations})")

        question_marks = text.count("?")
        if question_marks > 0:
            score += min(question_marks * 4.0, 15.0)
            matched_reasons.append(f"Question hook ({question_marks})")

        # Check for ALL-CAPS words (excluding single letters like I or A)
        words = text.split()
        shouting_words = [w for w in words if len(w) > 2 and w.isupper() and w.isalpha()]
        if shouting_words:
            score += min(len(shouting_words) * 6.0, 25.0)
            matched_reasons.append(f"High vocal energy ({', '.join(shouting_words[:3])})")

        primary_category = "highlight"
        if matched_categories:
            primary_category = max(matched_categories, key=matched_categories.get)

        return min(score, 100.0), primary_category, matched_reasons

    def _score_hook(self, first_segment_text: str) -> Tuple[float, str]:
        """
        Evaluate how captivating the opening line is for short-form retention.
        """
        text = first_segment_text.strip()
        lower = text.lower()
        score = 40.0  # Base line hook score

        for starter_pattern in HOOK_STARTERS:
            if re.search(starter_pattern, lower):
                score += 35.0
                break

        if text.endswith("?"):
            score += 15.0
        elif text.endswith("!"):
            score += 10.0

        # Penalize weak/dangling openings like starting with "And", "Because", "Or", "So"
        if re.match(r"^(?:and|because|or|so|but)\b", lower):
            score -= 20.0

        # Extract opening sentence as the hook
        sentences = re.split(r"(?<=[.!?])\s+", text)
        hook = sentences[0] if sentences else text[:80]

        return max(5.0, min(score, 100.0)), hook

    def _generate_title(self, hook: str, category: str, text: str) -> str:
        """Create a clean, engaging title for the candidate moment."""
        # Clean hook punctuation
        cleaned = re.sub(r'[^\w\s-]', '', hook).strip()
        words = cleaned.split()

        if 3 <= len(words) <= 7:
            return " ".join(words).title()

        if len(words) > 7:
            # Shorten to first 6-7 words
            return " ".join(words[:6]).title() + "..."

        # Category based templates
        templates = {
            "funny": "Hilarious Reaction",
            "surprise": "Wait Till The End",
            "argument": "Things Got Intense",
            "strong_opinion": "The Brutal Truth",
            "gaming_reaction": "Unbelievable Play",
            "emotional": "Truly Touching Moment",
            "question_curiosity": "Did You Know This?",
            "highlight": "Viral Highlight Moment"
        }
        return templates.get(category, "Viral Highlight Moment")

    def _is_clean_start(self, segment: TranscriptSegment) -> bool:
        """
        Determine if a segment is a natural place to begin a clip without
        cutting in the middle of a thought.
        """
        text = segment.text.strip()
        if not text:
            return False
        # Avoid lower-case continuations
        if text[0].islower():
            return False
        # Avoid conjunction openings
        lower = text.lower()
        if re.match(r"^(?:and|so|because|or|but|then)\b", lower):
            return False
        return True

    def _is_clean_end(self, segment: TranscriptSegment) -> bool:
        """
        Determine if a segment is a natural place to end a clip (sentence terminator).
        """
        text = segment.text.strip()
        if not text:
            return False
        return text[-1] in {".", "!", "?", "\"", "'"}

    def detect_moments(
        self,
        transcription: TranscriptionResult,
        max_clips: int = 5,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_overlap_ratio: float = 0.15
    ) -> List[CandidateClip]:
        """
        Analyze the full transcript and generate candidate clips.

        :param transcription: The TranscriptionResult containing segments.
        :param max_clips: Maximum number of non-overlapping clips to return.
        :param min_duration: Override minimum duration (seconds).
        :param max_duration: Override maximum duration (seconds).
        :param max_overlap_ratio: Max allowed overlap between selected clips (0.0 to 1.0).
        :return: List of CandidateClip instances ranked by engagement_score.
        """
        min_dur = min_duration or self.min_duration
        max_dur = max_duration or self.max_duration
        segments = transcription.segments

        if not segments:
            logger.warning("Empty transcription result provided to detect_moments.")
            return []

        # Step 1: Pre-score each individual segment
        segment_scores = []
        for seg in segments:
            k_score, cat, reasons = self._score_segment_text(seg.text)
            segment_scores.append({
                "segment": seg,
                "score": k_score,
                "category": cat,
                "reasons": reasons
            })

        # Step 2: Form candidate multi-segment windows respecting duration and context
        candidates: List[CandidateClip] = []
        n = len(segments)

        for i in range(n):
            # Context preservation: prefer starting at natural sentence boundaries
            start_seg = segments[i]
            is_start_clean = self._is_clean_start(start_seg)

            window_text_parts = []
            window_seg_ids = []

            for j in range(i, n):
                curr_seg = segments[j]
                window_text_parts.append(curr_seg.text)
                window_seg_ids.append(curr_seg.id)

                start_time = start_seg.start
                end_time = curr_seg.end
                duration = end_time - start_time

                if duration < min_dur:
                    continue
                if duration > max_dur:
                    break

                # Context preservation: prefer ending on a complete sentence
                is_end_clean = self._is_clean_end(curr_seg)

                full_clip_text = " ".join(window_text_parts).strip()
                word_count = len(full_clip_text.split())
                wps = word_count / max(duration, 1.0)  # words per second

                # Compute window subscores
                # 1. Hook score
                hook_score, hook_str = self._score_hook(start_seg.text)

                # 2. Window Keyword / Emotional density score
                window_kw_score, primary_category, reasons = self._score_segment_text(full_clip_text)

                # 3. Pacing score (optimal speaking rate: 2.2 to 3.8 words/sec)
                if 2.2 <= wps <= 3.8:
                    pacing_score = 90.0
                elif 1.5 <= wps < 2.2 or 3.8 < wps <= 4.5:
                    pacing_score = 70.0
                else:
                    pacing_score = 45.0

                # 4. Climax and setup structure
                # Check if there is a peak emotional/punchline moment inside the window
                internal_scores = [segment_scores[k]["score"] for k in range(i, j + 1)]
                peak_score = max(internal_scores) if internal_scores else 0.0
                peak_idx = internal_scores.index(peak_score) if internal_scores else 0
                has_setup = peak_idx > 0
                has_reaction = peak_idx < len(internal_scores) - 1
                climax_score = peak_score * 0.7 + (15.0 if has_setup else 0.0) + (15.0 if has_reaction else 0.0)
                climax_score = min(climax_score, 100.0)

                # 5. Completeness score
                completeness_score = (25.0 if is_start_clean else 10.0) + (25.0 if is_end_clean else 10.0)
                completeness_score = min(completeness_score * 2.0, 100.0)

                # Weighted aggregate engagement score (1 - 100 scale)
                total_engagement = (
                    (hook_score * 0.25) +
                    (window_kw_score * 0.30) +
                    (climax_score * 0.20) +
                    (pacing_score * 0.15) +
                    (completeness_score * 0.10)
                )

                # Cap between 1 and 99.5
                total_engagement = max(10.0, min(total_engagement, 99.5))

                # Build reasons string
                reason_summary = (
                    f"Category: {primary_category.replace('_', ' ').title()}; "
                    f"Pacing: {wps:.1f} wps; "
                    f"Setup & reaction: {'Yes' if (has_setup and has_reaction) else 'Partial'}; "
                    f"Triggers: {', '.join(reasons[:2]) if reasons else 'Natural conversational flow'}"
                )

                title = self._generate_title(hook_str, primary_category, full_clip_text)
                clip_id = f"clip_{int(start_time):04d}_{int(end_time):04d}"

                candidate = CandidateClip(
                    clip_id=clip_id,
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    duration=round(duration, 2),
                    title=title,
                    hook=hook_str,
                    engagement_score=round(total_engagement, 1),
                    category=primary_category,
                    reason=reason_summary,
                    transcript_text=full_clip_text,
                    segment_ids=list(window_seg_ids),
                    score_breakdown={
                        "hook": round(hook_score, 1),
                        "content_density": round(window_kw_score, 1),
                        "climax_structure": round(climax_score, 1),
                        "pacing": round(pacing_score, 1),
                        "completeness": round(completeness_score, 1),
                    }
                )
                candidates.append(candidate)

        # Step 3: Duplicate Prevention / Non-Maximum Suppression (NMS)
        # Sort candidates descending by engagement score
        candidates.sort(key=lambda c: c.engagement_score, reverse=True)

        selected_clips: List[CandidateClip] = []
        for candidate in candidates:
            if len(selected_clips) >= max_clips:
                break

            overlap = False
            for selected in selected_clips:
                # Calculate intersection over minimum duration
                inter_start = max(candidate.start_time, selected.start_time)
                inter_end = min(candidate.end_time, selected.end_time)
                intersection = max(0.0, inter_end - inter_start)
                min_len = min(candidate.duration, selected.duration)

                if min_len > 0 and (intersection / min_len) > max_overlap_ratio:
                    overlap = True
                    break

            if not overlap:
                # Optional enhancement with Ollama if enabled & reachable
                enhanced = self.ollama.enhance_moment(candidate)
                selected_clips.append(enhanced)

        # Sort selected clips chronologically for clean playback / export
        selected_clips.sort(key=lambda c: c.start_time)
        logger.info(
            "Detected %d distinct high-engagement candidate clips across transcript.",
            len(selected_clips)
        )
        return selected_clips


def save_moments_json(
    moments: List[CandidateClip],
    output_path: Union[str, Path]
) -> Path:
    """Save detected candidate moments to a structured JSON file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(moments),
        "disclaimer": "Engagement scores are heuristic approximations of hook and emotional cues; viral performance is not guaranteed.",
        "moments": [m.to_dict() for m in moments]
    }
    save_json(payload, path)
    logger.info("Saved %d detected moments to: %s", len(moments), path)
    return path


def load_moments_json(input_path: Union[str, Path]) -> List[CandidateClip]:
    """Load candidate moments from a JSON file."""
    data = load_json(input_path)
    if not data or "moments" not in data:
        return []
    return [CandidateClip.from_dict(m) for m in data["moments"]]
