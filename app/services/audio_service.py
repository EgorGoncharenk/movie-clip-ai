from __future__ import annotations

from pathlib import Path

from app.exceptions import ClipAppError
from app.models import VideoMetadata
from app.services.ffmpeg_service import FFmpegService


class AudioService:
    def __init__(self, ffmpeg: FFmpegService) -> None:
        self.ffmpeg = ffmpeg

    def extract(
        self,
        source: Path,
        destination: Path,
        metadata: VideoMetadata,
        *,
        overwrite: bool = False,
    ) -> Path:
        if metadata.audio_tracks == 0:
            raise ClipAppError(
                "The input video has no audio track, so it cannot be transcribed."
            )
        if destination.exists() and not overwrite:
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.require_tools()
        self.ffmpeg.run_process(
            [
                self.ffmpeg.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source,
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                destination,
            ],
            error_message="FFmpeg could not extract the audio track.",
        )
        return destination
