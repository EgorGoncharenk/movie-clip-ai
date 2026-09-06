from pathlib import Path

from app.config import EncodingSection, SubtitleSection
from app.exceptions import MediaProbeError
from app.models import ClipRange
from app.services.export_service import ExportService


class FakeFFmpeg:
    ffmpeg = Path("ffmpeg")

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def require_tools(self) -> None:
        pass

    def has_subtitles_filter(self) -> bool:
        return True

    def has_encoder(self, encoder: str) -> bool:
        return encoder == "h264_nvenc"

    def detect_crop(self, *args, **kwargs) -> None:
        return None

    def run_process(self, arguments, *, error_message):
        command = [str(argument) for argument in arguments]
        self.calls.append(command)
        if "h264_nvenc" in command:
            raise MediaProbeError("NVENC unavailable")
        Path(command[-1]).write_bytes(b"video")


def test_auto_encoder_falls_back_to_cpu(tmp_path):
    ffmpeg = FakeFFmpeg()
    service = ExportService(
        ffmpeg,
        EncodingSection(
            encoder="auto",
            quality_mode="fast",
            remove_black_bars=False,
        ),
        SubtitleSection(),
    )
    destination = tmp_path / "clip.mp4"

    service.render_blurred_background(
        tmp_path / "source.mp4",
        ClipRange(start=0, end=10),
        destination,
        subtitle_file=None,
    )

    assert len(ffmpeg.calls) == 2
    assert "h264_nvenc" in ffmpeg.calls[0]
    assert "libx264" in ffmpeg.calls[1]
    assert service.last_video_encoder == "libx264"
    assert destination.exists()


def test_quality_profiles_build_expected_arguments():
    ffmpeg = FakeFFmpeg()
    service = ExportService(
        ffmpeg,
        EncodingSection(quality_mode="quality"),
        SubtitleSection(),
    )

    assert service._video_encoding_arguments("h264_nvenc") == [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "0",
    ]
    assert service._video_encoding_arguments("libx264") == [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
    ]
