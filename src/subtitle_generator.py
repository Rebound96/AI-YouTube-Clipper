"""
Hormozi-Style Animated Subtitle Generator for AI-YouTube-Clipper.
Generates Advanced SubStation Alpha (.ass) subtitles with bold typography,
thick black outlines, and synchronized word-by-word active highlighting.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import List, Optional, Union

from src.config import SubtitleConfig
from src.transcriber import WordTiming
from src.utils import format_ass_timestamp

logger = logging.getLogger("AI-YouTube-Clipper.subtitles")


@dataclass
class SubtitleLineGroup:
    """Group of words displayed together on 1-2 lines on screen."""
    words: List[WordTiming]
    start: float
    end: float


class SubtitleGenerator:
    """Generates Hormozi-style ASS subtitles from word-level timings."""

    def __init__(self, config: Optional[SubtitleConfig] = None):
        self.config = config or SubtitleConfig()

    def generate_ass(
        self,
        words: List[WordTiming],
        output_ass_path: Union[str, Path],
        clip_start_time: float = 0.0,
        video_width: int = 1080,
        video_height: int = 1920
    ) -> Path:
        """
        Create a styled ASS subtitle file with word-by-word highlighting.

        :param words: List of WordTiming objects from faster-whisper.
        :param output_ass_path: File destination for the .ass subtitle.
        :param clip_start_time: Offset to adjust word timestamps relative to clip start.
        :param video_width: Target canvas width (default 1080).
        :param video_height: Target canvas height (default 1920).
        :return: Path to generated .ass file.
        """
        output_path = Path(output_ass_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Adjust word timestamps relative to clip start
        adjusted_words: List[WordTiming] = []
        for w in words:
            rel_start = max(0.0, w.start - clip_start_time)
            rel_end = max(rel_start + 0.05, w.end - clip_start_time)
            word_text = w.word.strip()
            if self.config.uppercase:
                word_text = word_text.upper()

            if word_text:
                adjusted_words.append(
                    WordTiming(
                        word=word_text,
                        start=rel_start,
                        end=rel_end,
                        probability=w.probability
                    )
                )

        if not adjusted_words:
            logger.warning("No words provided for subtitle generation. Creating empty ASS file.")
            self._write_empty_ass(output_path, video_width, video_height)
            return output_path

        # Step 2: Group words into short 1-2 line display chunks (max words per chunk)
        groups = self._group_words_into_lines(adjusted_words, max_words=self.config.max_words_per_line)

        # Step 3: Build ASS file contents
        ass_lines = self._build_ass_header(video_width, video_height)

        # Step 4: Generate dialogue events with word highlighting
        for group in groups:
            group_words = group.words
            for i, active_word in enumerate(group_words):
                # Timing for this word's highlight
                evt_start = format_ass_timestamp(active_word.start)
                # Word highlight extends until next word starts, or group ends
                if i + 1 < len(group_words):
                    evt_end_sec = group_words[i + 1].start
                else:
                    evt_end_sec = active_word.end

                if evt_end_sec <= active_word.start:
                    evt_end_sec = active_word.start + 0.1

                evt_end = format_ass_timestamp(evt_end_sec)

                # Format text with the active word in vivid yellow highlight
                formatted_text = self._format_highlighted_group(group_words, active_index=i)

                dialogue_line = (
                    f"Dialogue: 0,{evt_start},{evt_end},Hormozi,,0,0,0,,{formatted_text}"
                )
                ass_lines.append(dialogue_line)

        # Write to file (UTF-8)
        content = "\n".join(ass_lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Generated animated ASS subtitles with %d events at: %s", len(groups), output_path)
        return output_path

    def _group_words_into_lines(
        self,
        words: List[WordTiming],
        max_words: int = 4,
        max_pause_sec: float = 0.6
    ) -> List[SubtitleLineGroup]:
        """
        Group words into visually clean subtitle lines based on word count and pauses.
        """
        groups: List[SubtitleLineGroup] = []
        current: List[WordTiming] = []

        for w in words:
            # Check for natural pauses or sentence ending punctuation
            prev_end = current[-1].end if current else None
            pause_detected = prev_end is not None and (w.start - prev_end > max_pause_sec)
            sentence_ended = current and current[-1].word.endswith((".", "!", "?"))

            if (len(current) >= max_words) or pause_detected or sentence_ended:
                if current:
                    groups.append(
                        SubtitleLineGroup(
                            words=current,
                            start=current[0].start,
                            end=current[-1].end
                        )
                    )
                    current = []

            current.append(w)

        if current:
            groups.append(
                SubtitleLineGroup(
                    words=current,
                    start=current[0].start,
                    end=current[-1].end
                )
            )

        return groups

    def _format_highlighted_group(
        self,
        words: List[WordTiming],
        active_index: int
    ) -> str:
        """
        Construct ASS inline styling tags to color the active word in yellow
        and inactive words in white.
        """
        highlight_color = self.config.highlight_color  # Yellow
        primary_color = self.config.primary_color      # White

        tokens = []
        for idx, w in enumerate(words):
            word_str = w.word
            if idx == active_index:
                # Active highlighted word with subtle scale pop {\fscx108\fscy108}
                tokens.append(f"{{\\c{highlight_color}&\\fscx108\\fscy108}}{word_str}{{\\c{primary_color}&\\fscx100\\fscy100}}")
            else:
                tokens.append(word_str)

        return " ".join(tokens)

    def _build_ass_header(self, width: int, height: int) -> List[str]:
        """Build ASS file header with Hormozi typography style."""
        cfg = self.config
        header = [
            "[Script Info]",
            "Title: Hormozi Animated Subtitles",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            (
                f"Style: Hormozi,{cfg.font_name},{cfg.font_size},"
                f"{cfg.primary_color},&H000000FF,{cfg.outline_color},&H80000000,"
                f"-1,0,0,0,100,100,1,0,1,{cfg.outline_width},{cfg.shadow_width},2,40,40,{cfg.margin_v},1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        return header

    def _write_empty_ass(self, path: Path, width: int, height: int):
        """Write an empty ASS file if no speech is detected."""
        header = self._build_ass_header(width, height)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
