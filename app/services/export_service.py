from __future__ import annotations

import logging
from pathlib import Path

from app.config import EncodingSection, SubtitleSection
from app.exceptions import ExportError, MediaProbeError
from app.models import ClipRange, ReframePlan
from app.services.ffmpeg_service import FFmpegService

LOGGER = logging.getLogger(__name__)

QUALITY_PROFILES = {
    "fast": {
        "cpu_preset": "veryfast",
        "cpu_crf": "24",
        "nvenc_preset": "p3",
        "nvenc_cq": "25",
    },
    "balanced": {
        "cpu_preset": "medium",
        "cpu_crf": "20",
        "nvenc_preset": "p5",
        "nvenc_cq": "21",
    },
    "quality": {
        "cpu_preset": "slow",
        "cpu_crf": "18",
        "nvenc_preset": "p7",
        "nvenc_cq": "18",
    },
}


def _escape_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _escape_ass_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(",", "\\,")


class ExportService:
    def __init__(
        self,
        ffmpeg: FFmpegService,
        encoding: EncodingSection,
        subtitles: SubtitleSection,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.encoding = encoding
        self.subtitles = subtitles
        self.last_video_encoder = encoding.video_codec

    def _preferred_encoder(self) -> str:
        if self.encoding.encoder == "cpu":
            return "libx264"
        if self.encoding.encoder == "nvenc":
            return "h264_nvenc"
        if self.ffmpeg.has_encoder("h264_nvenc"):
            return "h264_nvenc"
        return "libx264"

    def _video_encoding_arguments(self, encoder: str) -> list[str]:
        profile = QUALITY_PROFILES.get(self.encoding.quality_mode)
        if encoder == "h264_nvenc":
            if profile:
                return [
                    "-c:v",
                    encoder,
                    "-preset",
                    profile["nvenc_preset"],
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    profile["nvenc_cq"],
                    "-b:v",
                    "0",
                ]
            return [
                "-c:v",
                encoder,
                "-preset",
                "p5",
                "-cq",
                str(self.encoding.crf),
                "-b:v",
                "0",
            ]
        if profile:
            return [
                "-c:v",
                "libx264",
                "-preset",
                profile["cpu_preset"],
                "-crf",
                profile["cpu_crf"],
            ]
        return [
            "-c:v",
            self.encoding.video_codec,
            "-preset",
            self.encoding.preset,
            "-crf",
            str(self.encoding.crf),
        ]

    def _video_filter(
        self,
        subtitle_file: Path | None,
        crop: str | None,
    ) -> str:
        width = self.encoding.width
        height = self.encoding.height
        blur = self.encoding.background_blur
        source_chain = (
            f"[0:v]crop={crop}[clean];"
            if crop
            else "[0:v]null[clean];"
        )
        chain = (
            f"{source_chain}[clean]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma={blur}[bgv];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgv];"
            f"[bgv][fgv]overlay=(W-w)/2:(H-h)/2[base]"
        )
        return self._append_subtitles(chain, "base", subtitle_file)

    def _append_subtitles(
        self,
        chain: str,
        input_label: str,
        subtitle_file: Path | None,
    ) -> str:
        if subtitle_file is None:
            return f"{chain};[{input_label}]null[v]"

        escaped_path = _escape_filter_path(subtitle_file)
        if subtitle_file.suffix.casefold() == ".ass":
            return (
                f"{chain};[{input_label}]"
                f"subtitles=filename='{escaped_path}'[v]"
            )

        style = ",".join(
            [
                f"FontName={_escape_ass_value(self.subtitles.font)}",
                f"FontSize={self.subtitles.font_size}",
                f"Bold={-1 if self.subtitles.bold else 0}",
                f"Outline={self.subtitles.outline}",
                f"Shadow={self.subtitles.shadow}",
                "Alignment=2",
                f"MarginV={self.subtitles.margin_bottom}",
                "PrimaryColour=&H00FFFFFF",
                "OutlineColour=&H00000000",
            ]
        )
        return (
            f"{chain};[{input_label}]subtitles=filename='{escaped_path}':"
            f"charenc=UTF-8:force_style='{style}'[v]"
        )

    @staticmethod
    def _positions(
        plan: ReframePlan,
        *,
        axis: str,
    ) -> list[float]:
        if axis == "x":
            crop_size = plan.crop_width
            content_size = plan.content_width
            centers = [keyframe.center_x for keyframe in plan.keyframes]
        else:
            crop_size = plan.crop_height
            content_size = plan.content_height
            centers = [keyframe.center_y for keyframe in plan.keyframes]
        return [
            min(
                max(center - crop_size / 2, 0),
                content_size - crop_size,
            )
            for center in centers
        ]

    @classmethod
    def _position_expression(
        cls,
        plan: ReframePlan,
        *,
        axis: str,
    ) -> str:
        positions = cls._positions(plan, axis=axis)
        if len(positions) <= 1 or max(positions) - min(positions) < 0.01:
            return f"{positions[0] if positions else 0:.2f}"

        expression = f"{positions[-1]:.2f}"
        for index in range(len(positions) - 2, -1, -1):
            start = plan.keyframes[index].time
            end = plan.keyframes[index + 1].time
            duration = max(0.001, end - start)
            position = positions[index]
            delta = positions[index + 1] - position
            segment = (
                f"{position:.2f}+({delta:.2f})"
                f"*(t-{start:.3f})/{duration:.3f}"
            )
            expression = (
                f"if(lt(t,{end:.3f}),{segment},{expression})"
            )
        return expression

    @classmethod
    def _write_reframe_commands(
        cls,
        plan: ReframePlan,
        destination: Path,
    ) -> Path:
        x_positions = cls._positions(plan, axis="x")
        y_positions = cls._positions(plan, axis="y")
        lines: list[str] = []
        for index, keyframe in enumerate(plan.keyframes):
            if index + 1 < len(plan.keyframes):
                following = plan.keyframes[index + 1]
                duration = max(0.001, following.time - keyframe.time)
                same_scene = (
                    keyframe.scene_index is None
                    or following.scene_index is None
                    or keyframe.scene_index == following.scene_index
                )
                if same_scene:
                    x_delta = x_positions[index + 1] - x_positions[index]
                    y_delta = y_positions[index + 1] - y_positions[index]
                    x_value = (
                        f"{x_positions[index]:.2f}+({x_delta:.2f})"
                        f"*(t-{keyframe.time:.3f})/{duration:.3f}"
                    )
                    y_value = (
                        f"{y_positions[index]:.2f}+({y_delta:.2f})"
                        f"*(t-{keyframe.time:.3f})/{duration:.3f}"
                    )
                else:
                    x_value = f"{x_positions[index]:.2f}"
                    y_value = f"{y_positions[index]:.2f}"
            else:
                x_value = f"{x_positions[index]:.2f}"
                y_value = f"{y_positions[index]:.2f}"
            lines.append(
                f"{keyframe.time:.6f} "
                f"[enter] crop@smart x {x_value},"
                f"[enter] crop@smart y {y_value};"
            )
        destination.write_text("\n".join(lines) + "\n", encoding="ascii")
        return destination

    def _smart_reframe_filter(
        self,
        plan: ReframePlan,
        subtitle_file: Path | None,
        command_file: Path,
    ) -> str:
        x_positions = self._positions(plan, axis="x")
        y_positions = self._positions(plan, axis="y")
        escaped_commands = _escape_filter_path(command_file)
        chains = [
            f"[0:v]crop={plan.content_width}:{plan.content_height}:"
            f"{plan.content_x}:{plan.content_y}[content]"
        ]
        guarded_input = "content"
        for index, guard in enumerate(plan.transition_guards):
            base = f"guard_base_{index}"
            replacement = f"guard_replacement_{index}"
            output = f"guarded_{index}"
            chains.append(
                f"[{guarded_input}]split=2[{base}][{replacement}]"
            )
            chains.append(
                f"[{base}][{replacement}]freezeframes="
                f"first={guard.first_frame}:"
                f"last={guard.last_frame}:"
                f"replace={guard.replace_frame}[{output}]"
            )
            guarded_input = output
        chains.append(
            f"[{guarded_input}]sendcmd=f='{escaped_commands}',"
            f"crop@smart={plan.crop_width}:{plan.crop_height}:"
            f"x={x_positions[0]:.2f}:y={y_positions[0]:.2f},"
            f"scale={self.encoding.width}:{self.encoding.height}[base]"
        )
        chain = ";".join(chains)
        return self._append_subtitles(chain, "base", subtitle_file)

    def _render_arguments(
        self,
        source: Path,
        clip_range: ClipRange,
        destination: Path,
        video_filter: str,
        *,
        overwrite: bool,
        encoder: str,
    ) -> list[str | Path]:
        arguments: list[str | Path] = [
            self.ffmpeg.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{clip_range.start:.3f}",
            "-t",
            f"{clip_range.duration:.3f}",
            "-i",
            source,
            "-filter_complex",
            video_filter,
            "-map",
            "[v]",
            "-map",
            "0:a:0?",
        ]
        arguments.extend(self._video_encoding_arguments(encoder))
        arguments.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                self.encoding.audio_codec,
                "-b:a",
                self.encoding.audio_bitrate,
                "-movflags",
                "+faststart",
                "-shortest",
                destination,
            ]
        )
        return arguments

    def _run_render(
        self,
        arguments: list[str | Path],
        destination: Path,
        encoder: str,
    ) -> None:
        try:
            self.ffmpeg.run_process(
                arguments,
                error_message=f"FFmpeg could not render '{destination.name}'.",
            )
        except MediaProbeError as exc:
            if encoder != "h264_nvenc":
                raise ExportError(str(exc)) from exc
            LOGGER.warning(
                "NVENC export failed; retrying with libx264: %s",
                exc,
            )
            destination.unlink(missing_ok=True)
            cpu_arguments = list(arguments)
            video_options_start = cpu_arguments.index("-c:v")
            pixel_format_index = cpu_arguments.index("-pix_fmt")
            cpu_arguments[video_options_start:pixel_format_index] = (
                self._video_encoding_arguments("libx264")
            )
            try:
                self.ffmpeg.run_process(
                    cpu_arguments,
                    error_message=(
                        f"FFmpeg could not render '{destination.name}' "
                        "after NVENC fallback."
                    ),
                )
            except MediaProbeError as fallback_exc:
                raise ExportError(str(fallback_exc)) from fallback_exc
            encoder = "libx264"
        self.last_video_encoder = encoder

    def render_blurred_background(
        self,
        source: Path,
        clip_range: ClipRange,
        destination: Path,
        *,
        subtitle_file: Path | None,
        overwrite: bool = False,
    ) -> Path:
        if destination.exists() and not overwrite:
            raise ExportError(
                f"Output already exists: {destination}. Use --overwrite to replace it."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.require_tools()
        if subtitle_file and not self.ffmpeg.has_subtitles_filter():
            raise ExportError(
                "This FFmpeg build has no subtitles filter. Install a build with libass."
            )

        crop = None
        if self.encoding.remove_black_bars:
            crop = self.ffmpeg.detect_crop(
                source,
                start=clip_range.start,
                duration=clip_range.duration,
            )

        encoder = self._preferred_encoder()
        arguments = self._render_arguments(
            source,
            clip_range,
            destination,
            self._video_filter(subtitle_file, crop),
            overwrite=overwrite,
            encoder=encoder,
        )
        self._run_render(arguments, destination, encoder)
        return destination

    def render_smart_reframe(
        self,
        source: Path,
        clip_range: ClipRange,
        destination: Path,
        plan: ReframePlan,
        *,
        subtitle_file: Path | None,
        overwrite: bool = False,
    ) -> Path:
        if destination.exists() and not overwrite:
            raise ExportError(
                f"Output already exists: {destination}. "
                "Use --overwrite to replace it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.require_tools()
        if subtitle_file and not self.ffmpeg.has_subtitles_filter():
            raise ExportError(
                "This FFmpeg build has no subtitles filter. "
                "Install a build with libass."
            )
        encoder = self._preferred_encoder()
        command_file = self._write_reframe_commands(
            plan,
            destination.parent / "reframe_commands.txt",
        )
        arguments = self._render_arguments(
            source,
            clip_range,
            destination,
            self._smart_reframe_filter(
                plan,
                subtitle_file,
                command_file,
            ),
            overwrite=overwrite,
            encoder=encoder,
        )
        self._run_render(arguments, destination, encoder)
        return destination

    def create_preview(self, video_file: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.ffmpeg.run_process(
                [
                    self.ffmpeg.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    "1",
                    "-i",
                    video_file,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    destination,
                ],
                error_message="FFmpeg could not create the preview image.",
            )
        except MediaProbeError as exc:
            raise ExportError(str(exc)) from exc
        return destination
