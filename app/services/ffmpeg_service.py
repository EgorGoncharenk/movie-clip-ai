from __future__ import annotations

import json
import os
import shutil
import subprocess
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from app.exceptions import EnvironmentCheckError, MediaProbeError
from app.models import VideoMetadata


def _find_windows_program(name: str) -> Path | None:
    executable = f"{name}.exe"
    candidates: list[Path] = []

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        programs = Path(local_app_data) / "Programs"
        candidates.extend(programs.glob(f"ffmpeg/**/bin/{executable}"))
        candidates.append(programs / "ffmpeg" / "bin" / executable)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_executable(name: str) -> Path | None:
    discovered = shutil.which(name)
    if discovered:
        return Path(discovered)
    if os.name == "nt":
        return _find_windows_program(name)
    return None


class FFmpegService:
    def __init__(self) -> None:
        self.ffmpeg = find_executable("ffmpeg")
        self.ffprobe = find_executable("ffprobe")
        self._encoders: set[str] | None = None

    def require_tools(self) -> None:
        missing = [
            name
            for name, path in (
                ("FFmpeg", self.ffmpeg),
                ("FFprobe", self.ffprobe),
            )
            if path is None
        ]
        if missing:
            joined = " and ".join(missing)
            raise EnvironmentCheckError(
                f"{joined} not found. Install the Windows essentials build and "
                "add its bin directory to PATH."
            )

    @staticmethod
    def run_process(
        arguments: Sequence[str | Path],
        *,
        error_message: str,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(argument) for argument in arguments]
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except FileNotFoundError as exc:
            raise EnvironmentCheckError(
                f"Executable not found: {command[0]}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            if len(details) > 2000:
                details = details[-2000:]
            raise MediaProbeError(f"{error_message}\n{details}") from exc

    def version(self, tool: str) -> str:
        path = self.ffmpeg if tool == "ffmpeg" else self.ffprobe
        if path is None:
            return "not found"
        result = self.run_process(
            [path, "-version"],
            error_message=f"Could not read {tool} version.",
        )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"

    def has_subtitles_filter(self) -> bool:
        if self.ffmpeg is None:
            return False
        result = self.run_process(
            [self.ffmpeg, "-hide_banner", "-filters"],
            error_message="Could not inspect FFmpeg filters.",
        )
        return any(
            line.split()[1:2] == ["subtitles"]
            for line in result.stdout.splitlines()
            if line.strip()
        )

    def has_encoder(self, encoder: str) -> bool:
        if self.ffmpeg is None:
            return False
        if self._encoders is None:
            result = self.run_process(
                [self.ffmpeg, "-hide_banner", "-encoders"],
                error_message="Could not inspect FFmpeg encoders.",
            )
            self._encoders = {
                parts[1]
                for line in result.stdout.splitlines()
                if len(parts := line.split()) >= 2
                and not parts[0].startswith("-")
            }
        return encoder in self._encoders

    def probe(self, source: Path) -> VideoMetadata:
        self.require_tools()
        source = source.resolve()
        if not source.is_file():
            raise MediaProbeError(f"Input video does not exist: {source}")

        result = self.run_process(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                source,
            ],
            error_message=f"FFprobe could not read '{source}'.",
        )
        try:
            payload = json.loads(result.stdout)
            streams = payload.get("streams", [])
            video_stream = next(
                stream for stream in streams if stream.get("codec_type") == "video"
            )
            audio_streams = [
                stream for stream in streams if stream.get("codec_type") == "audio"
            ]
            duration = float(
                payload.get("format", {}).get("duration")
                or video_stream.get("duration")
            )
            fps_raw = (
                video_stream.get("avg_frame_rate")
                or video_stream.get("r_frame_rate")
                or "0/1"
            )
            fps = float(Fraction(fps_raw))
        except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
            raise MediaProbeError(
                "The file has no readable video stream or valid duration."
            ) from exc

        first_audio = audio_streams[0] if audio_streams else None
        sample_rate = None
        if first_audio and first_audio.get("sample_rate"):
            sample_rate = int(first_audio["sample_rate"])

        file_size = source.stat().st_size
        audio_cache_size = int(duration * 16_000 * 2)
        estimated_temp_size = audio_cache_size + int(file_size * 1.5)
        return VideoMetadata(
            source_file=source,
            duration=duration,
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            fps=fps,
            video_start_time=float(video_stream.get("start_time") or 0),
            video_codec=str(video_stream.get("codec_name", "unknown")),
            audio_codec=(
                str(first_audio.get("codec_name", "unknown"))
                if first_audio
                else None
            ),
            audio_tracks=len(audio_streams),
            sample_rate=sample_rate,
            file_size=file_size,
            estimated_temp_size=estimated_temp_size,
        )

    def detect_crop(
        self,
        source: Path,
        *,
        start: float,
        duration: float,
    ) -> str | None:
        """Return FFmpeg crop parameters that remove stable black borders."""
        self.require_tools()
        sample_duration = min(8.0, max(1.0, duration))
        result = self.run_process(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "info",
                "-ss",
                f"{start:.3f}",
                "-i",
                source,
                "-t",
                f"{sample_duration:.3f}",
                "-vf",
                "cropdetect=limit=24:round=2:reset=0",
                "-an",
                "-f",
                "null",
                "-",
            ],
            error_message="FFmpeg could not detect black borders.",
        )
        matches = re.findall(
            r"crop=(\d+:\d+:\d+:\d+)",
            result.stderr or "",
        )
        if not matches:
            return None
        crop, _ = Counter(matches).most_common(1)[0]
        return crop
