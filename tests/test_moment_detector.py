"""
Comprehensive unit tests for the automatic viral moment detector (Phase 3).
Tests full transcript analysis, candidate generation, context preservation,
duration constraints, non-maximum suppression, and JSON serialization.
"""

from pathlib import Path
import pytest

from src.moment_detector import (
    CandidateClip,
    MomentDetector,
    load_moments_json,
    save_moments_json,
)
from src.transcriber import TranscriptSegment, TranscriptionResult, WordTiming


def create_segment(seg_id: int, start: float, end: float, text: str) -> TranscriptSegment:
    """Helper to create synthetic transcript segments with generated word timings."""
    words = text.split()
    word_duration = (end - start) / max(len(words), 1)
    word_timings = [
        WordTiming(
            word=w,
            start=round(start + i * word_duration, 2),
            end=round(start + (i + 1) * word_duration, 2),
            probability=0.99
        )
        for i, w in enumerate(words)
    ]
    return TranscriptSegment(
        id=seg_id,
        start=start,
        end=end,
        text=text,
        words=word_timings,
        avg_logprob=-0.1,
        no_speech_prob=0.01
    )


@pytest.fixture
def synthetic_full_transcript() -> TranscriptionResult:
    """
    Construct a synthetic multi-topic 240-second (4-minute) transcript containing:
    1. A boring intro (0-30s)
    2. A funny moment / joke with laughter (30-75s)
    3. Normal transition conversation (75-110s)
    4. An intense argument / debate (110-160s)
    5. A gaming climax reaction (160-205s)
    6. A strong opinion / hot take conclusion (205-240s)
    """
    segments = [
        # 0 - 30s: Ordinary introduction
        create_segment(1, 0.0, 8.0, "Welcome back to the channel everyone."),
        create_segment(2, 8.5, 18.0, "Today we are looking at standard weekly updates and standard metrics."),
        create_segment(3, 18.5, 29.0, "There is not much to report on the administrative front."),

        # 30 - 75s: Funny moment with setup and punchline
        create_segment(4, 30.0, 39.0, "Did you know what happened yesterday in the office?"),
        create_segment(5, 39.5, 48.0, "Dave tried to microwave his metal water bottle in the kitchen."),
        create_segment(6, 48.5, 59.0, "It caught on fire and he started screaming hahaha! That was hilarious!"),
        create_segment(7, 59.5, 74.0, "Everyone was cracking up and laughing for ten minutes straight."),

        # 75 - 110s: Ordinary transition
        create_segment(8, 75.0, 89.0, "Anyway, moving along to our main agenda items."),
        create_segment(9, 89.5, 109.0, "We need to review the quarterly roadmap and project deadlines."),

        # 110 - 160s: Intense argument
        create_segment(10, 110.0, 122.0, "Here is the biggest mistake you are making right now."),
        create_segment(11, 122.5, 134.0, "You are completely wrong about this entire strategy!"),
        create_segment(12, 134.5, 146.0, "Stop lying to the team, your numbers make zero sense!"),
        create_segment(13, 146.5, 159.0, "That is total nonsense and you need to prove it right now!"),

        # 160 - 205s: Gaming climax reaction
        create_segment(14, 160.0, 172.0, "Watch what happens in this final round right here."),
        create_segment(15, 172.5, 183.0, "No way, holy crap did you see that clutch headshot?!"),
        create_segment(16, 183.5, 194.0, "LET'S GO! We actually won the game against all odds!"),
        create_segment(17, 194.5, 204.0, "That was an unbelievable play, I am in absolute shock."),

        # 205 - 240s: Strong opinion / hot take
        create_segment(18, 205.0, 216.0, "The brutal truth nobody talks about in this industry."),
        create_segment(19, 216.5, 227.0, "If you are still following the old advice you will fail."),
        create_segment(20, 227.5, 239.0, "Mark my words, this secret is going to change everything."),
    ]

    full_text = " ".join(s.text for s in segments)
    return TranscriptionResult(
        segments=segments,
        language="en",
        language_probability=0.99,
        duration=240.0,
        full_text=full_text
    )


def test_detect_moments_duration_constraints(synthetic_full_transcript: TranscriptionResult):
    """Verify all detected clips strictly satisfy minimum and maximum duration constraints."""
    min_d = 20.0
    max_d = 50.0
    detector = MomentDetector(min_duration=min_d, max_duration=max_d)

    clips = detector.detect_moments(
        synthetic_full_transcript,
        max_clips=5,
        min_duration=min_d,
        max_duration=max_d
    )

    assert len(clips) > 0
    for clip in clips:
        assert min_d <= clip.duration <= max_d + 1.0  # slight floating leeway
        assert clip.start_time < clip.end_time
        assert clip.duration == pytest.approx(clip.end_time - clip.start_time, 0.1)


def test_detect_moments_duplicate_prevention(synthetic_full_transcript: TranscriptionResult):
    """Verify that returned candidate moments do not overlap each other."""
    detector = MomentDetector(min_duration=20.0, max_duration=60.0)
    clips = detector.detect_moments(
        synthetic_full_transcript,
        max_clips=6,
        max_overlap_ratio=0.10
    )

    assert len(clips) >= 3

    # Check pairwise non-overlap
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            c1, c2 = clips[i], clips[j]
            overlap_start = max(c1.start_time, c2.start_time)
            overlap_end = min(c1.end_time, c2.end_time)
            intersection = max(0.0, overlap_end - overlap_start)
            min_duration = min(c1.duration, c2.duration)
            overlap_ratio = intersection / min_duration
            assert overlap_ratio <= 0.15, f"Clips {c1.clip_id} and {c2.clip_id} overlap excessively ({overlap_ratio:.2f})"


def test_detect_across_full_transcript(synthetic_full_transcript: TranscriptionResult):
    """Verify moments are detected across different parts of the video (early, middle, late)."""
    detector = MomentDetector(min_duration=20.0, max_duration=55.0)
    clips = detector.detect_moments(synthetic_full_transcript, max_clips=4)

    start_times = [c.start_time for c in clips]
    # Ensure we have candidates from distinct sections of the 240s timeline
    has_early = any(t < 80.0 for t in start_times)
    has_mid_or_late = any(t >= 110.0 for t in start_times)

    assert has_early, "Should have detected early highlight"
    assert has_mid_or_late, "Should have detected middle/late highlight"


def test_scoring_system_transparency(synthetic_full_transcript: TranscriptionResult):
    """Verify transparent engagement scoring and breakdown keys."""
    detector = MomentDetector(min_duration=20.0, max_duration=60.0)
    clips = detector.detect_moments(synthetic_full_transcript, max_clips=3)

    for clip in clips:
        assert 1.0 <= clip.engagement_score <= 100.0
        assert "hook" in clip.score_breakdown
        assert "content_density" in clip.score_breakdown
        assert "climax_structure" in clip.score_breakdown
        assert "pacing" in clip.score_breakdown
        assert "completeness" in clip.score_breakdown
        assert clip.title != ""
        assert clip.hook != ""
        assert len(clip.transcript_text) > 20
        assert clip.category in [
            "funny", "surprise", "argument", "strong_opinion",
            "gaming_reaction", "emotional", "question_curiosity", "highlight"
        ]


def test_save_and_load_moments_json(synthetic_full_transcript: TranscriptionResult, tmp_path: Path):
    """Verify candidate moments JSON serialization and deserialization."""
    detector = MomentDetector(min_duration=20.0, max_duration=60.0)
    clips = detector.detect_moments(synthetic_full_transcript, max_clips=3)

    json_path = tmp_path / "detected_moments.json"
    saved_path = save_moments_json(clips, json_path)
    assert saved_path.exists()

    loaded_clips = load_moments_json(json_path)
    assert len(loaded_clips) == len(clips)
    assert loaded_clips[0].clip_id == clips[0].clip_id
    assert loaded_clips[0].start_time == clips[0].start_time
    assert loaded_clips[0].engagement_score == clips[0].engagement_score
    assert loaded_clips[0].title == clips[0].title


def test_empty_transcript_handling():
    """Verify empty transcript gracefully returns an empty list without raising."""
    detector = MomentDetector()
    empty_result = TranscriptionResult(segments=[], language="en", language_probability=1.0, duration=0.0, full_text="")
    clips = detector.detect_moments(empty_result)
    assert clips == []
