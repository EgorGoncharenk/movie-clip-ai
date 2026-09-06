from __future__ import annotations

import re

from app.exceptions import InvalidClipRangeError
from app.models import ClipRange

_TIMECODE_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):"
    r"(?P<seconds>\d{1,2}(?:[.,]\d+)?)$"
)


def parse_timecode(value: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Timecode cannot be empty")

    try:
        seconds = float(cleaned.replace(",", "."))
    except ValueError:
        match = _TIMECODE_PATTERN.fullmatch(cleaned)
        if not match:
            raise ValueError(
                f"Invalid timecode '{value}'. Use seconds, MM:SS or HH:MM:SS."
            ) from None

        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds_part = float(match.group("seconds").replace(",", "."))
        if minutes >= 60 or seconds_part >= 60:
            raise ValueError(f"Invalid timecode '{value}'")
        seconds = hours * 3600 + minutes * 60 + seconds_part

    if seconds < 0:
        raise ValueError("Timecode cannot be negative")
    return seconds


def parse_clip_range(value: str) -> ClipRange:
    if ".." in value:
        start_raw, end_raw = value.split("..", maxsplit=1)
    elif "-" in value:
        start_raw, end_raw = value.split("-", maxsplit=1)
    else:
        raise InvalidClipRangeError(
            f"Invalid clip '{value}'. Use START-END, for example 00:01:10-00:01:40."
        )

    try:
        clip_range = ClipRange(
            start=parse_timecode(start_raw),
            end=parse_timecode(end_raw),
        )
    except (ValueError, TypeError) as exc:
        raise InvalidClipRangeError(str(exc)) from exc

    if clip_range.end <= clip_range.start:
        raise InvalidClipRangeError(
            f"Clip end must be later than its start: '{value}'."
        )
    return clip_range


def validate_clip_range(clip_range: ClipRange, video_duration: float) -> None:
    if clip_range.end > video_duration + 0.05:
        raise InvalidClipRangeError(
            f"Clip ends at {clip_range.end:.3f}s, but the video is only "
            f"{video_duration:.3f}s long."
        )


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"
