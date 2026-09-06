from pathlib import Path

from app.config import EncodingSection, SubtitleSection
from app.models import (
    ClipRange,
    TranscriptSegment,
    TranscriptionResult,
    WordTiming,
)
from app.services.subtitle_service import SubtitleService


def _transcription() -> TranscriptionResult:
    return TranscriptionResult(
        source_file=Path("source.mp4"),
        language="ru",
        duration=20,
        model="small",
        device="cpu",
        segments=[
            TranscriptSegment(
                id=0,
                start=4,
                end=7,
                text="Первая реплика",
            ),
            TranscriptSegment(
                id=1,
                start=9,
                end=13,
                text="Вторая достаточно длинная реплика",
            ),
        ],
    )


def test_segments_are_clipped_and_shifted_to_zero():
    result = SubtitleService.segments_for_clip(
        _transcription(),
        ClipRange(start=5, end=10),
    )
    assert [(item.start, item.end) for item in result] == [(0, 2), (4, 5)]


def test_generate_srt(tmp_path):
    service = SubtitleService()
    destination = tmp_path / "clip.srt"
    _, segments = service.write_srt(
        _transcription(),
        ClipRange(start=5, end=10),
        destination,
        max_chars=18,
    )
    content = destination.read_text(encoding="utf-8")
    assert len(segments) == 2
    assert "00:00:00,000 --> 00:00:02,000" in content
    assert "Первая реплика" in content
    assert "00:00:04,000 --> 00:00:05,000" in content


def test_generate_ass_with_explicit_canvas(tmp_path):
    service = SubtitleService()
    segments = service.segments_for_clip(
        _transcription(),
        ClipRange(start=5, end=10),
    )
    destination = service.write_ass(
        segments,
        tmp_path / "clip.ass",
        subtitle_settings=SubtitleSection(
            font_size=58,
            margin_bottom=220,
        ),
        encoding_settings=EncodingSection(),
    )
    content = destination.read_text(encoding="utf-8-sig")
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Style: Default,Arial,58" in content
    assert r"00.00,0:00:02.00" in content


def test_word_timestamps_are_grouped_into_two_or_three_words():
    transcription = TranscriptionResult(
        source_file=Path("source.mp4"),
        language="ru",
        duration=4,
        model="medium",
        device="cuda",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.2,
                end=2.7,
                text="Раз два три четыре пять",
                words=[
                    WordTiming(
                        start=0.2 + index * 0.5,
                        end=0.7 + index * 0.5,
                        text=text,
                    )
                    for index, text in enumerate(
                        ["Раз", "два", "три", "четыре", "пять"]
                    )
                ],
            )
        ],
    )

    captions = SubtitleService.caption_segments_for_clip(
        transcription,
        ClipRange(start=0, end=4),
        words_per_caption=3,
        max_word_gap=0.55,
    )

    assert [caption.text for caption in captions] == [
        "Раз два три",
        "четыре пять",
    ]
    assert [(caption.start, caption.end) for caption in captions] == [
        (0.2, 1.7),
        (1.7, 2.7),
    ]
    assert all(2 <= len(caption.words) <= 3 for caption in captions)
