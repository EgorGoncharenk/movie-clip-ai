from app.config import SelectionSection
from app.models import ClipCandidate
from app.services.clip_selection_service import (
    ClipSelectionService,
    overlap_ratio,
    text_similarity,
)


def _candidate(index, start, end, score, text):
    return ClipCandidate(
        index=index,
        start=start,
        end=end,
        score=score,
        transcript=text,
    )


def test_overlap_ratio_uses_shorter_clip():
    first = _candidate(1, 0, 30, 90, "первый текст")
    second = _candidate(2, 20, 40, 80, "второй текст")
    assert overlap_ratio(first, second) == 0.5


def test_text_similarity_detects_near_duplicates():
    assert text_similarity(
        "Это очень важная правда о нашей истории",
        "Важная правда о нашей общей истории",
    ) > 0.5


def test_selection_removes_overlaps_and_spreads_clips():
    candidates = [
        _candidate(1, 0, 30, 95, "сильный первый уникальный момент"),
        _candidate(2, 5, 35, 94, "почти тот же первый момент"),
        _candidate(3, 80, 110, 88, "другой момент в середине фильма"),
        _candidate(4, 170, 200, 85, "финальный отдельный момент истории"),
    ]
    service = ClipSelectionService(
        SelectionSection(
            clips=3,
            max_overlap_ratio=0.2,
            max_text_similarity=0.75,
        )
    )

    selected = service.select(candidates, video_duration=200)

    assert len(selected) == 3
    assert selected[0].start == 0
    assert all(
        overlap_ratio(first, second) <= 0.2
        for index, first in enumerate(selected)
        for second in selected[index + 1 :]
    )
