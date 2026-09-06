import pytest

from app.config import SceneDetectionSection
from app.services.ffmpeg_service import FFmpegService
from app.services.scene_detection_service import SceneDetectionService


@pytest.mark.integration
def test_scene_detection_finds_hard_cut_and_uses_cache(tmp_path):
    ffmpeg = FFmpegService()
    if not ffmpeg.ffmpeg:
        pytest.skip("FFmpeg is not installed")
    source = tmp_path / "two-scenes.mp4"
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
            "color=c=red:size=320x180:rate=25:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=320x180:rate=25:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            source,
        ],
        error_message="Could not create scene test video.",
    )
    cache_file = tmp_path / "scenes.json"
    service = SceneDetectionService(
        SceneDetectionSection(threshold=10, min_scene_length=0.2)
    )

    first = service.detect(
        source,
        cache_file,
        video_duration=2,
    )
    second = service.detect(
        source,
        cache_file,
        video_duration=2,
    )

    assert len(first.scenes) >= 2
    assert first.scenes[0].end == pytest.approx(1, abs=0.16)
    assert second.created_at == first.created_at
    assert cache_file.exists()
