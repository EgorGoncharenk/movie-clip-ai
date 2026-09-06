from __future__ import annotations

import re

from app.config import SelectionSection
from app.models import ClipCandidate

_WORD_PATTERN = re.compile(r"[\wёЁ]+", re.UNICODE)


def overlap_ratio(first: ClipCandidate, second: ClipCandidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    if overlap <= 0:
        return 0.0
    return overlap / min(first.duration, second.duration)


def text_similarity(first: str, second: str) -> float:
    first_words = {
        word.casefold() for word in _WORD_PATTERN.findall(first) if len(word) > 2
    }
    second_words = {
        word.casefold() for word in _WORD_PATTERN.findall(second) if len(word) > 2
    }
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


class ClipSelectionService:
    def __init__(self, settings: SelectionSection) -> None:
        self.settings = settings

    def select(
        self,
        candidates: list[ClipCandidate],
        *,
        video_duration: float,
    ) -> list[ClipCandidate]:
        remaining = sorted(candidates, key=lambda item: item.score, reverse=True)
        selected: list[ClipCandidate] = []

        while remaining and len(selected) < self.settings.clips:
            if not selected:
                best = remaining.pop(0)
            else:
                def adjusted_score(candidate: ClipCandidate) -> float:
                    distance = min(
                        abs((candidate.start + candidate.end) / 2 - (
                            chosen.start + chosen.end
                        ) / 2)
                        for chosen in selected
                    )
                    spread_bonus = 10.0 * distance / max(video_duration, 1)
                    return candidate.score + spread_bonus

                best = max(remaining, key=adjusted_score)
                remaining.remove(best)

            if any(
                overlap_ratio(best, chosen) > self.settings.max_overlap_ratio
                or text_similarity(best.transcript, chosen.transcript)
                > self.settings.max_text_similarity
                for chosen in selected
            ):
                continue
            selected.append(best)

        return [
            candidate.model_copy(update={"index": index})
            for index, candidate in enumerate(selected, start=1)
        ]
