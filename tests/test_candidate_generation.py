from pathlib import Path

from app.config import CandidateGenerationSection
from app.models import SceneSegment, TranscriptSegment, TranscriptionResult
from app.services.candidate_generation_service import CandidateGenerationService


def _transcription() -> TranscriptionResult:
    texts = [
        "Почему ты никогда не говорил мне правду?",
        "Потому что я боялся потерять всё.",
        "Но теперь выбора больше нет.",
        "Это самая важная ошибка в моей жизни.",
        "Ты должен услышать меня до конца.",
        "Я не был твоим врагом.",
        "Настоящий враг всё это время был рядом.",
        "И сейчас я могу это доказать.",
        "Посмотри на письмо в моих руках.",
        "В нём есть имя настоящего предателя.",
        "Теперь ты знаешь всю правду.",
        "И назад дороги уже нет.",
    ]
    return TranscriptionResult(
        source_file=Path("movie.mp4"),
        language="ru",
        duration=60,
        model="small",
        device="cpu",
        segments=[
            TranscriptSegment(
                id=index,
                start=index * 5.0,
                end=index * 5.0 + 4.3,
                text=text,
            )
            for index, text in enumerate(texts)
        ],
    )


def test_generates_sentence_aligned_candidates():
    service = CandidateGenerationService(
        CandidateGenerationSection(
            min_duration=20,
            max_duration=30,
            target_duration=25,
        )
    )
    candidates = service.generate(_transcription(), video_duration=60)

    assert candidates
    assert all(20 <= candidate.duration <= 30 for candidate in candidates)
    assert all(0 <= candidate.score <= 100 for candidate in candidates)
    assert any("вопрос в начале" in candidate.reasons for candidate in candidates)


def test_candidates_snap_to_scene_boundaries_without_cutting_speech():
    service = CandidateGenerationService(
        CandidateGenerationSection(
            min_duration=20,
            max_duration=30,
            target_duration=25,
            scene_snap_window=2,
        )
    )
    scenes = [
        SceneSegment(index=1, start=0, end=25),
        SceneSegment(index=2, start=25, end=60),
    ]

    candidates = service.generate(
        _transcription(),
        video_duration=60,
        scenes=scenes,
    )

    aligned = [
        candidate
        for candidate in candidates
        if candidate.score_breakdown["scene_alignment"] > 0
    ]
    assert aligned
    assert any(candidate.end == 25 for candidate in aligned)
    assert all(20 <= candidate.duration <= 30 for candidate in aligned)
