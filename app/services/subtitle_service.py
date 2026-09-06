from __future__ import annotations

import textwrap
from pathlib import Path

from app.config import EncodingSection, SubtitleSection
from app.models import (
    ClipRange,
    TranscriptSegment,
    TranscriptionResult,
    WordTiming,
)
from app.utils.timecodes import format_srt_timestamp


class SubtitleService:
    @staticmethod
    def _wrapped_text(text: str, max_chars: int) -> str:
        return "\n".join(
            textwrap.wrap(
                " ".join(text.split()),
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    @staticmethod
    def segments_for_clip(
        transcription: TranscriptionResult,
        clip_range: ClipRange,
    ) -> list[TranscriptSegment]:
        clipped: list[TranscriptSegment] = []
        for segment in transcription.segments:
            start = max(segment.start, clip_range.start)
            end = min(segment.end, clip_range.end)
            if end <= start or not segment.text.strip():
                continue
            clipped.append(
                segment.model_copy(
                    update={
                        "start": start - clip_range.start,
                        "end": end - clip_range.start,
                    }
                )
            )
        return clipped

    @staticmethod
    def _partition_words(
        words: list[WordTiming],
        *,
        words_per_caption: int,
        max_word_gap: float,
    ) -> list[list[WordTiming]]:
        runs: list[list[WordTiming]] = []
        current: list[WordTiming] = []
        for word in words:
            if (
                current
                and word.start - current[-1].end > max_word_gap
            ):
                runs.append(current)
                current = []
            current.append(word)
            if word.text.rstrip().endswith((".", "!", "?", "…")):
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        groups: list[list[WordTiming]] = []
        for run in runs:
            position = 0
            while position < len(run):
                remaining = len(run) - position
                if remaining <= words_per_caption:
                    size = remaining
                elif remaining == words_per_caption + 1:
                    size = 2
                else:
                    size = words_per_caption
                groups.append(run[position : position + size])
                position += size
        return groups

    @staticmethod
    def _estimated_words(segment: TranscriptSegment) -> list[WordTiming]:
        texts = segment.text.split()
        if not texts:
            return []
        duration = max(0.01, segment.end - segment.start)
        step = duration / len(texts)
        return [
            WordTiming(
                start=segment.start + index * step,
                end=segment.start + (index + 1) * step,
                text=text,
            )
            for index, text in enumerate(texts)
        ]

    @classmethod
    def caption_segments_for_clip(
        cls,
        transcription: TranscriptionResult,
        clip_range: ClipRange,
        *,
        words_per_caption: int,
        max_word_gap: float,
    ) -> list[TranscriptSegment]:
        captions: list[TranscriptSegment] = []
        for source_segment in transcription.segments:
            if (
                source_segment.end <= clip_range.start
                or source_segment.start >= clip_range.end
            ):
                continue
            source_words = (
                source_segment.words
                if source_segment.words
                else cls._estimated_words(source_segment)
            )
            clipped_words = [
                word.model_copy(
                    update={
                        "start": max(
                            word.start,
                            clip_range.start,
                        )
                        - clip_range.start,
                        "end": min(
                            word.end,
                            clip_range.end,
                        )
                        - clip_range.start,
                        "text": word.text.strip(),
                    }
                )
                for word in source_words
                if word.end > clip_range.start
                and word.start < clip_range.end
                and word.text.strip()
            ]
            for group in cls._partition_words(
                clipped_words,
                words_per_caption=words_per_caption,
                max_word_gap=max_word_gap,
            ):
                if not group:
                    continue
                captions.append(
                    TranscriptSegment(
                        id=len(captions),
                        start=group[0].start,
                        end=max(group[-1].end, group[0].start + 0.05),
                        text=" ".join(
                            word.text.strip() for word in group
                        ),
                        words=group,
                        confidence=source_segment.confidence,
                        no_speech_probability=(
                            source_segment.no_speech_probability
                        ),
                    )
                )
        return captions

    @staticmethod
    def generate_srt(
        segments: list[TranscriptSegment],
        *,
        max_chars: int,
    ) -> str:
        blocks: list[str] = []
        for index, segment in enumerate(segments, start=1):
            text = SubtitleService._wrapped_text(segment.text, max_chars)
            blocks.append(
                f"{index}\n"
                f"{format_srt_timestamp(segment.start)} --> "
                f"{format_srt_timestamp(segment.end)}\n"
                f"{text}"
            )
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def write_srt(
        self,
        transcription: TranscriptionResult,
        clip_range: ClipRange,
        destination: Path,
        *,
        max_chars: int,
        words_per_caption: int = 3,
        max_word_gap: float = 0.55,
    ) -> tuple[Path, list[TranscriptSegment]]:
        segments = self.caption_segments_for_clip(
            transcription,
            clip_range,
            words_per_caption=words_per_caption,
            max_word_gap=max_word_gap,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.generate_srt(segments, max_chars=max_chars),
            encoding="utf-8",
        )
        return destination, segments

    @staticmethod
    def _ass_timestamp(seconds: float) -> str:
        centiseconds = max(0, round(seconds * 100))
        hours, remainder = divmod(centiseconds, 360_000)
        minutes, remainder = divmod(remainder, 6_000)
        whole_seconds, centis = divmod(remainder, 100)
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centis:02d}"

    @staticmethod
    def escape_ass_text(text: str) -> str:
        return (
            text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
        )

    def write_ass(
        self,
        segments: list[TranscriptSegment],
        destination: Path,
        *,
        subtitle_settings: SubtitleSection,
        encoding_settings: EncodingSection,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        font_name = subtitle_settings.font.replace(",", " ")
        bold = -1 if subtitle_settings.bold else 0
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {encoding_settings.width}\n"
            f"PlayResY: {encoding_settings.height}\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{subtitle_settings.font_size},"
            "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
            f"{bold},0,0,0,100,100,0,0,1,{subtitle_settings.outline},"
            f"{subtitle_settings.shadow},2,60,60,"
            f"{subtitle_settings.margin_bottom},1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
        )
        dialogue = []
        for segment in segments:
            wrapped = self._wrapped_text(
                segment.text,
                subtitle_settings.max_chars,
            )
            escaped = self.escape_ass_text(wrapped)
            dialogue.append(
                "Dialogue: 0,"
                f"{self._ass_timestamp(segment.start)},"
                f"{self._ass_timestamp(segment.end)},"
                f"Default,,0,0,0,,{escaped}"
            )
        destination.write_text(
            header + "\n".join(dialogue) + ("\n" if dialogue else ""),
            encoding="utf-8-sig",
        )
        return destination
