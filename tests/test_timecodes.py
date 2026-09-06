import pytest
from pydantic import ValidationError

from app.exceptions import InvalidClipRangeError
from app.models import ClipRange
from app.utils.timecodes import (
    format_srt_timestamp,
    parse_clip_range,
    parse_timecode,
    validate_clip_range,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12.5", 12.5),
        ("01:02", 62.0),
        ("01:02.250", 62.25),
        ("01:02:03,500", 3723.5),
    ],
)
def test_parse_timecode(value, expected):
    assert parse_timecode(value) == pytest.approx(expected)


def test_parse_clip_range():
    result = parse_clip_range("00:10-00:25.5")
    assert result.start == 10
    assert result.end == 25.5
    assert result.duration == 15.5


def test_rejects_reversed_range():
    with pytest.raises(InvalidClipRangeError):
        parse_clip_range("10-5")


def test_model_rejects_reversed_range():
    with pytest.raises(ValidationError):
        ClipRange(start=10, end=5)


def test_rejects_range_past_video_end():
    with pytest.raises(InvalidClipRangeError):
        validate_clip_range(ClipRange(start=5, end=11), video_duration=10)


def test_format_srt_timestamp_rounds_milliseconds():
    assert format_srt_timestamp(3661.2346) == "01:01:01,235"
