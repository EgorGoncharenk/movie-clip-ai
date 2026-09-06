from __future__ import annotations

import math
import re

from app.config import CandidateGenerationSection
from app.exceptions import ClipAppError
from app.models import (
    ClipCandidate,
    SceneSegment,
    TranscriptSegment,
    TranscriptionResult,
)

_WORD_PATTERN = re.compile(r"[\wёЁ]+", re.UNICODE)
_HOOK_WORDS = {
    "как",
    "почему",
    "зачем",
    "представь",
    "никогда",
    "самый",
    "главный",
    "секрет",
    "правда",
    "ошибка",
    "смотри",
    "знаешь",
    "but",
    "how",
    "why",
    "never",
    "secret",
    "truth",
    "mistake",
}
_EMOTION_STEMS = {
    "люб",
    "ненав",
    "страх",
    "смерт",
    "уби",
    "шок",
    "ужас",
    "счаст",
    "боль",
    "невозмож",
    "опас",
    "предал",
    "love",
    "hate",
    "fear",
    "death",
    "kill",
    "shock",
    "pain",
    "danger",
}
_CONFLICT_STEMS = {
    "но",
    "против",
    "враг",
    "спор",
    "лж",
    "обман",
    "должен",
    "нельзя",
    "but",
    "against",
    "enemy",
    "lie",
    "fight",
    "must",
    "can't",
}
_WEAK_OPENINGS = {
    "он",
    "она",
    "они",
    "это",
    "этот",
    "эта",
    "тогда",
    "там",
    "такой",
    "такая",
    "he",
    "she",
    "they",
    "it",
    "that",
    "then",
}


def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_PATTERN.findall(text)]


def _contains_stem(words: list[str], stems: set[str]) -> bool:
    return any(
        word == stem or word.startswith(stem)
        for word in words
        for stem in stems
    )


class CandidateGenerationService:
    def __init__(self, settings: CandidateGenerationSection) -> None:
        self.settings = settings

    def _is_natural_end(
        self,
        segments: list[TranscriptSegment],
        end_index: int,
    ) -> bool:
        text = segments[end_index].text.rstrip()
        if text.endswith((".", "!", "?", "…")):
            return True
        if end_index + 1 >= len(segments):
            return True
        pause = segments[end_index + 1].start - segments[end_index].end
        return pause >= self.settings.minimum_pause

    def _score(
        self,
        text: str,
        duration: float,
        natural_end: bool,
        scene_aligned: bool = False,
    ) -> tuple[float, dict[str, float], list[str]]:
        words = _words(text)
        if not words:
            return 0.0, {}, ["Нет распознанной речи"]

        target = min(
            self.settings.max_duration,
            max(self.settings.min_duration, self.settings.target_duration),
        )
        duration_distance = abs(duration - target) / max(target, 1)
        duration_score = max(0.0, 15.0 * (1.0 - duration_distance))

        opening = words[:18]
        hook_score = 3.0
        reasons: list[str] = []
        if "?" in text[:180]:
            hook_score += 8.0
            reasons.append("вопрос в начале")
        if any(word in _HOOK_WORDS for word in opening):
            hook_score += 7.0
            reasons.append("сильная вводная фраза")
        hook_score = min(hook_score, 20.0)

        emotion_score = 2.0
        if _contains_stem(words, _EMOTION_STEMS):
            emotion_score += 10.0
            reasons.append("эмоциональная лексика")
        if "!" in text:
            emotion_score += 5.0
        emotion_score = min(emotion_score, 17.0)

        conflict_score = 0.0
        if _contains_stem(words, _CONFLICT_STEMS):
            conflict_score = 13.0
            reasons.append("конфликт или противопоставление")

        completion_score = 17.0 if natural_end else 7.0
        if natural_end:
            reasons.append("законченная реплика")

        density = len(words) / max(duration, 1)
        density_score = max(0.0, 15.0 - abs(density - 2.4) * 5.0)
        if 1.4 <= density <= 3.5:
            reasons.append("хороший темп речи")

        independence_score = 13.0
        if opening and opening[0] in _WEAK_OPENINGS:
            independence_score = 5.0
        elif len(words) >= 35:
            reasons.append("достаточно контекста")

        punctuation_bonus = min(
            3.0,
            math.log2(1 + text.count("?") + text.count("!")),
        )
        scene_score = 6.0 if scene_aligned else 0.0
        if scene_aligned:
            reasons.append("границы совпадают со сменой сцены")
        breakdown = {
            "hook": round(hook_score, 2),
            "emotion": round(emotion_score, 2),
            "conflict": round(conflict_score, 2),
            "completion": round(completion_score, 2),
            "speech_density": round(density_score, 2),
            "context": round(independence_score, 2),
            "duration": round(duration_score, 2),
            "scene_alignment": scene_score,
        }
        total = min(
            100.0,
            sum(breakdown.values()) + punctuation_bonus,
        )
        if not reasons:
            reasons.append("связный речевой фрагмент")
        return round(total, 2), breakdown, reasons[:4]

    def _snap_to_scene_boundaries(
        self,
        start: float,
        end: float,
        *,
        speech_start: float,
        speech_end: float,
        scenes: list[SceneSegment],
    ) -> tuple[float, float, bool]:
        window = self.settings.scene_snap_window
        if not scenes or window <= 0:
            return start, end, False
        boundaries = sorted(
            {
                round(value, 3)
                for scene in scenes
                for value in (scene.start, scene.end)
            }
        )
        start_options = [
            boundary
            for boundary in boundaries
            if boundary <= speech_start and abs(boundary - start) <= window
        ]
        end_options = [
            boundary
            for boundary in boundaries
            if boundary >= speech_end and abs(boundary - end) <= window
        ]
        snapped_start = (
            min(start_options, key=lambda value: abs(value - start))
            if start_options
            else start
        )
        snapped_end = (
            min(end_options, key=lambda value: abs(value - end))
            if end_options
            else end
        )
        duration = snapped_end - snapped_start
        if not self.settings.min_duration <= duration <= self.settings.max_duration:
            return start, end, False
        aligned = (
            abs(snapped_start - start) > 0.01
            or abs(snapped_end - end) > 0.01
        )
        return snapped_start, snapped_end, aligned

    def generate(
        self,
        transcription: TranscriptionResult,
        *,
        video_duration: float,
        scenes: list[SceneSegment] | None = None,
    ) -> list[ClipCandidate]:
        segments = [
            segment
            for segment in transcription.segments
            if segment.text.strip()
            and (segment.no_speech_probability or 0.0) < 0.85
        ]
        if not segments:
            raise ClipAppError(
                "В видео не найдено достаточно распознанной речи "
                "для автоматического выбора моментов."
            )

        candidates: list[ClipCandidate] = []
        seen_ranges: set[tuple[int, int]] = set()
        for start_index, first_segment in enumerate(segments):
            start = max(
                0.0,
                first_segment.start - self.settings.padding_before,
            )
            best_endpoint: tuple[float, int, bool] | None = None
            for end_index in range(start_index, len(segments)):
                end = min(
                    video_duration,
                    segments[end_index].end + self.settings.padding_after,
                )
                duration = end - start
                if duration > self.settings.max_duration:
                    break
                if duration < self.settings.min_duration:
                    continue

                natural_end = self._is_natural_end(segments, end_index)
                target_distance = abs(
                    duration - self.settings.target_duration
                )
                endpoint_rank = target_distance - (4.0 if natural_end else 0.0)
                if best_endpoint is None or endpoint_rank < best_endpoint[0]:
                    best_endpoint = (endpoint_rank, end_index, natural_end)

            if best_endpoint is None:
                continue

            _, end_index, natural_end = best_endpoint
            end = min(
                video_duration,
                segments[end_index].end + self.settings.padding_after,
            )
            start, end, scene_aligned = self._snap_to_scene_boundaries(
                start,
                end,
                speech_start=first_segment.start,
                speech_end=segments[end_index].end,
                scenes=scenes or [],
            )
            range_key = (round(start * 2), round(end * 2))
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)
            text = " ".join(
                segment.text.strip()
                for segment in segments[start_index : end_index + 1]
            )
            score, breakdown, reasons = self._score(
                text,
                end - start,
                natural_end,
                scene_aligned,
            )
            candidates.append(
                ClipCandidate(
                    index=len(candidates) + 1,
                    start=start,
                    end=end,
                    score=score,
                    score_breakdown=breakdown,
                    reasons=reasons,
                    transcript=text,
                )
            )

        if not candidates and video_duration >= 3:
            start = max(0.0, segments[0].start - self.settings.padding_before)
            end = min(
                video_duration,
                segments[-1].end + self.settings.padding_after,
            )
            if end > start:
                text = " ".join(segment.text.strip() for segment in segments)
                score, breakdown, reasons = self._score(
                    text,
                    end - start,
                    True,
                )
                candidates.append(
                    ClipCandidate(
                        index=1,
                        start=start,
                        end=end,
                        score=score,
                        score_breakdown=breakdown,
                        reasons=reasons,
                        transcript=text,
                    )
                )
        return candidates
