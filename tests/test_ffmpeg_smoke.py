from pathlib import Path

import pytest

from app.config import EncodingSection, Settings, SubtitleSection
from app.exceptions import ClipAppError
from app.models import ClipRange, TranscriptSegment, TranscriptionResult
from app.pipeline import ClipPipeline
from app.services.ffmpeg_service import FFmpegService


@pytest.mark.integration
def test_blurred_background_pipeline_smoke(tmp_path):
    ffmpeg = FFmpegService()
    if not ffmpeg.ffmpeg or not ffmpeg.ffprobe:
        pytest.skip("FFmpeg/FFprobe are not installed")
    if not ffmpeg.has_subtitles_filter():
        pytest.skip("FFmpeg subtitles filter is unavailable")

    source = tmp_path / "Видео с пробелом.mp4"
    ffmpeg.run_process(
        [
            ffmpeg.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            source,
        ],
        error_message="Could not create synthetic test video.",
    )
    settings = Settings(
        paths={"output": tmp_path / "output", "temp": tmp_path / "temp"},
        encoding=EncodingSection(
            width=360,
            height=640,
            encoder="cpu",
            quality_mode="custom",
            preset="ultrafast",
            crf=28,
        ),
        subtitles=SubtitleSection(
            font_size=26,
            margin_bottom=70,
        ),
    )
    transcription = TranscriptionResult(
        source_file=source,
        language="ru",
        duration=2,
        model="small",
        device="cpu",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.1,
                end=1.7,
                text="Проверка субтитров",
            )
        ],
    )

    report = ClipPipeline(settings).render(
        source,
        transcription,
        [ClipRange(start=0, end=1.8)],
    )

    clip = report.clips[0]
    assert clip.video_file.exists()
    assert clip.video_file.stat().st_size > 0
    assert clip.subtitle_file and clip.subtitle_file.exists()
    assert clip.ass_file and clip.ass_file.exists()
    assert clip.preview_file and clip.preview_file.exists()
    assert clip.reframe_plan_file and clip.reframe_plan_file.exists()
    assert (clip.video_file.parent / "reframe_commands.txt").exists()
    assert clip.reframe_mode in {"smart_face", "center_crop_no_faces"}
    output_metadata = ffmpeg.probe(clip.video_file)
    assert output_metadata.width == 360
    assert output_metadata.height == 640
    assert output_metadata.audio_tracks == 1
    assert (report.output_directory / "report.json").exists()
    assert (report.output_directory / "report.csv").exists()
    assert report.metrics is not None
    assert any(
        stage.name == "video_export"
        for stage in report.metrics.stages
    )


def test_pipeline_rejects_transcript_from_another_video(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    transcription = TranscriptionResult(
        source_file=tmp_path / "other.mp4",
        language="ru",
        duration=2,
        model="small",
        device="cpu",
        segments=[],
    )

    with pytest.raises(ClipAppError, match="другого"):
        ClipPipeline(Settings()).render(
            source,
            transcription,
            [ClipRange(start=0, end=1)],
        )
