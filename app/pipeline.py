from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable

from pydantic import TypeAdapter

from app import __version__
from app.config import Settings
from app.exceptions import ClipAppError, OperationCancelled, TranscriptionError
from app.models import (
    ClipCandidate,
    ClipRange,
    RenderedClip,
    RunReport,
    SceneSegment,
    TranscriptionResult,
    VideoMetadata,
)
from app.services.audio_service import AudioService
from app.services.candidate_generation_service import CandidateGenerationService
from app.services.clip_selection_service import ClipSelectionService
from app.services.export_service import ExportService
from app.services.ffmpeg_service import FFmpegService
from app.services.face_tracking_service import FaceTrackingService
from app.services.scene_detection_service import SceneDetectionService
from app.services.subtitle_service import SubtitleService
from app.services.transcription_service import TranscriptionService
from app.utils.metrics import MetricsCollector
from app.utils.paths import cache_directory, create_run_directory
from app.utils.timecodes import validate_clip_range

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]


def _check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check and cancel_check():
        raise OperationCancelled("Обработка отменена пользователем.")


class ClipPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        sample_resources: bool = False,
    ) -> None:
        self.settings = settings
        self.metrics = MetricsCollector(sample_resources=sample_resources)
        self.ffmpeg = FFmpegService()
        self.audio = AudioService(self.ffmpeg)
        self.transcriber = TranscriptionService(settings.transcription)
        self.candidate_generator = CandidateGenerationService(
            settings.candidate_generation
        )
        self.clip_selector = ClipSelectionService(settings.selection)
        self.scene_detector = SceneDetectionService(settings.scene_detection)
        self.face_tracker = FaceTrackingService(
            settings.reframe,
            settings.encoding,
        )
        self._scenes: dict[Path, list[SceneSegment]] = {}
        self.subtitles = SubtitleService()
        self.exporter = ExportService(
            self.ffmpeg,
            settings.encoding,
            settings.subtitles,
        )

    def inspect(self, source: Path) -> VideoMetadata:
        LOGGER.info("Reading video metadata: %s", source)
        with self.metrics.stage("media_probe", source=source.name):
            return self.ffmpeg.probe(source)

    def transcribe(
        self,
        source: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> tuple[TranscriptionResult, Path, Path]:
        _check_cancelled(cancel_check)
        cache_dir = cache_directory(source, self.settings.paths.temp)
        audio_file = cache_dir / "audio.wav"
        transcript_file = cache_dir / "transcription.json"
        with self.metrics.stage(
            "transcription_cache",
            cache_file=str(transcript_file),
            overwrite=overwrite,
        ) as cache_details:
            cache_details["cache_hit"] = False
            if transcript_file.exists() and not overwrite:
                try:
                    cached = self.transcriber.load(transcript_file)
                    if (
                        cached.source_file.resolve() == source.resolve()
                        and cached.model == self.settings.transcription.model
                        and (
                            not self.settings.transcription.language
                            or cached.language == self.settings.transcription.language
                        )
                    ):
                        cache_details["cache_hit"] = True
                        LOGGER.info(
                            "Using cached transcription: %s",
                            transcript_file,
                        )
                        if progress_callback:
                            progress_callback(
                                "Используется готовая транскрипция",
                                3,
                                3,
                            )
                        return cached, transcript_file, audio_file
                except TranscriptionError:
                    LOGGER.warning(
                        "Ignoring unreadable transcription cache: %s",
                        transcript_file,
                    )

        self.transcriber.require_local_model(
            self.settings.paths.temp / "models"
        )
        if progress_callback:
            progress_callback("Чтение метаданных", 0, 3)
        metadata = self.inspect(source)
        _check_cancelled(cancel_check)
        if progress_callback:
            progress_callback("Извлечение аудио", 1, 3)
        LOGGER.info("Extracting audio to %s", audio_file)
        with self.metrics.stage(
            "audio_extract",
            duration_seconds=round(metadata.duration, 3),
        ):
            self.audio.extract(
                source,
                audio_file,
                metadata,
                overwrite=overwrite,
            )
        _check_cancelled(cancel_check)
        if progress_callback:
            progress_callback("Распознавание речи", 2, 3)
        LOGGER.info("Transcribing speech")
        def transcription_progress(
            processed_seconds: float,
            total_seconds: float,
        ) -> None:
            if not progress_callback or total_seconds <= 0:
                return
            percent = min(
                100,
                max(0, round(processed_seconds / total_seconds * 100)),
            )
            progress_callback(
                f"Распознавание речи: {percent}% "
                f"({processed_seconds / 60:.1f} из "
                f"{total_seconds / 60:.1f} мин.)",
                percent,
                100,
            )

        with self.metrics.stage(
            "speech_to_text",
            model=self.settings.transcription.model,
            requested_device=self.settings.transcription.device,
        ) as transcription_details:
            result = self.transcriber.transcribe(
                audio_file,
                source,
                transcript_file,
                overwrite=overwrite,
                model_cache=self.settings.paths.temp / "models",
                cancel_check=cancel_check,
                progress_callback=transcription_progress,
            )
            transcription_details["actual_device"] = result.device
            transcription_details["segments"] = len(result.segments)
        if progress_callback:
            progress_callback("Транскрипция готова", 3, 3)
        return result, transcript_file, audio_file

    def analyze(
        self,
        source: Path,
        transcription: TranscriptionResult,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[ClipCandidate]:
        _check_cancelled(cancel_check)
        if transcription.source_file.resolve() != source.resolve():
            raise ClipAppError(
                "Файл транскрипции создан для другого исходного видео."
            )
        if progress_callback:
            progress_callback("Анализ сцен", 0, 3)
        metadata = self.inspect(source)
        cache_dir = cache_directory(source, self.settings.paths.temp)
        def scene_progress(
            processed_seconds: float,
            total_seconds: float,
        ) -> None:
            if not progress_callback or total_seconds <= 0:
                return
            percent = min(
                100,
                max(0, round(processed_seconds / total_seconds * 100)),
            )
            progress_callback(
                f"Анализ сцен: {percent}% "
                f"({processed_seconds / 60:.1f} из "
                f"{total_seconds / 60:.1f} мин.)",
                percent,
                100,
            )

        with self.metrics.stage(
            "scene_detection",
            threshold=self.settings.scene_detection.threshold,
        ) as scene_details:
            scene_result = self.scene_detector.detect(
                source,
                cache_dir / "scenes.json",
                video_duration=metadata.duration,
                progress_callback=scene_progress,
                cancel_check=cancel_check,
            )
            scene_details["scenes"] = len(scene_result.scenes)
            self._scenes[source.resolve()] = scene_result.scenes
        if progress_callback:
            progress_callback("Формирование кандидатов", 2, 3)
        with self.metrics.stage(
            "candidate_generation",
            transcript_segments=len(transcription.segments),
        ) as generation_details:
            candidates = self.candidate_generator.generate(
                transcription,
                video_duration=metadata.duration,
                scenes=scene_result.scenes,
            )
            generation_details["candidates"] = len(candidates)
        _check_cancelled(cancel_check)
        if progress_callback:
            progress_callback("Выбор лучших моментов", 2, 3)
        with self.metrics.stage(
            "clip_selection",
            candidates=len(candidates),
        ) as selection_details:
            selected = self.clip_selector.select(
                candidates,
                video_duration=metadata.duration,
            )
            selection_details["selected"] = len(selected)
        (cache_dir / "candidates.json").write_text(
            TypeAdapter(list[ClipCandidate])
            .dump_json(selected, indent=2)
            .decode("utf-8"),
            encoding="utf-8",
        )
        if progress_callback:
            progress_callback("Моменты выбраны", 3, 3)
        return selected

    def render(
        self,
        source: Path,
        transcription: TranscriptionResult,
        clip_ranges: list[ClipRange],
        *,
        output_root: Path | None = None,
        overwrite: bool = False,
        candidates: list[ClipCandidate] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> RunReport:
        _check_cancelled(cancel_check)
        if transcription.source_file.resolve() != source.resolve():
            raise ClipAppError(
                "Файл транскрипции создан для другого исходного видео."
            )
        metadata = self.inspect(source)
        source_key = source.resolve()
        if source_key not in self._scenes:
            cached_scenes = self.scene_detector.load_cached(
                source,
                cache_directory(source, self.settings.paths.temp)
                / "scenes.json",
            )
            if cached_scenes:
                self._scenes[source_key] = cached_scenes.scenes
        for clip_range in clip_ranges:
            validate_clip_range(clip_range, metadata.duration)

        root = output_root or self.settings.paths.output
        run_directory = create_run_directory(source, root)
        rendered: list[RenderedClip] = []

        for index, clip_range in enumerate(clip_ranges, start=1):
            _check_cancelled(cancel_check)
            if progress_callback:
                progress_callback(
                    f"Рендер клипа {index} из {len(clip_ranges)}",
                    index - 1,
                    len(clip_ranges),
                )
            LOGGER.info(
                "Rendering clip %d/%d: %.3fs-%.3fs",
                index,
                len(clip_ranges),
                clip_range.start,
                clip_range.end,
            )
            clip_directory = run_directory / f"clip_{index:02d}"
            clip_directory.mkdir(parents=True)
            video_file = clip_directory / f"clip_{index:02d}.mp4"
            subtitle_file = clip_directory / f"clip_{index:02d}.srt"
            ass_file = clip_directory / f"clip_{index:02d}.ass"
            preview_file = clip_directory / "preview.jpg"
            reframe_plan_file = clip_directory / "reframe.json"
            warnings: list[str] = []
            candidate = (
                candidates[index - 1]
                if candidates and index <= len(candidates)
                else None
            )

            with self.metrics.stage(
                "subtitle_generation",
                clip=index,
            ) as subtitle_details:
                subtitle_path, subtitle_segments = self.subtitles.write_srt(
                    transcription,
                    clip_range,
                    subtitle_file,
                    max_chars=self.settings.subtitles.max_chars,
                    words_per_caption=(
                        self.settings.subtitles.words_per_caption
                    ),
                    max_word_gap=self.settings.subtitles.max_word_gap,
                )
                ass_path = self.subtitles.write_ass(
                    subtitle_segments,
                    ass_file,
                    subtitle_settings=self.settings.subtitles,
                    encoding_settings=self.settings.encoding,
                )
                subtitle_details["segments"] = len(subtitle_segments)
            burn_subtitles: Path | None = ass_path
            if not self.settings.subtitles.enabled:
                burn_subtitles = None
            elif not subtitle_segments:
                burn_subtitles = None
                warnings.append("В выбранном интервале нет распознанной речи.")

            reframe_plan = None
            if self.settings.reframe.mode in {"smart", "center"}:
                content_crop = None
                if self.settings.encoding.remove_black_bars:
                    content_crop = self.ffmpeg.detect_crop(
                        source,
                        start=clip_range.start,
                        duration=clip_range.duration,
                    )
                if progress_callback:
                    progress_callback(
                        f"Отслеживание лиц: клип {index} из "
                        f"{len(clip_ranges)}",
                        index - 1,
                        len(clip_ranges),
                    )
                with self.metrics.stage(
                    "face_tracking",
                    clip=index,
                    requested_mode=self.settings.reframe.mode,
                ) as tracking_details:
                    reframe_plan = self.face_tracker.plan(
                        source,
                        clip_range,
                        metadata,
                        content_crop=content_crop,
                        scenes=self._scenes.get(source.resolve(), []),
                        cancel_check=cancel_check,
                    )
                    tracking_details["sampled_frames"] = (
                        reframe_plan.sampled_frames
                    )
                    tracking_details["frames_with_faces"] = (
                        reframe_plan.frames_with_faces
                    )
                    tracking_details["actual_mode"] = reframe_plan.mode
                reframe_plan_file.write_text(
                    reframe_plan.model_dump_json(indent=2),
                    encoding="utf-8",
                )

            with self.metrics.stage(
                "video_export",
                clip=index,
                duration_seconds=round(clip_range.duration, 3),
                requested_encoder=self.settings.encoding.encoder,
                quality_mode=self.settings.encoding.quality_mode,
            ) as export_details:
                if reframe_plan:
                    self.exporter.render_smart_reframe(
                        source,
                        clip_range,
                        video_file,
                        reframe_plan,
                        subtitle_file=burn_subtitles,
                        overwrite=overwrite,
                    )
                else:
                    self.exporter.render_blurred_background(
                        source,
                        clip_range,
                        video_file,
                        subtitle_file=burn_subtitles,
                        overwrite=overwrite,
                    )
                export_details["actual_encoder"] = (
                    self.exporter.last_video_encoder
                )
            with self.metrics.stage("preview_export", clip=index):
                self.exporter.create_preview(video_file, preview_file)

            transcript_text = " ".join(
                segment.text.strip() for segment in subtitle_segments
            ).strip()
            clip = RenderedClip(
                index=index,
                source_file=source.resolve(),
                start=clip_range.start,
                end=clip_range.end,
                duration=clip_range.duration,
                video_file=video_file.resolve(),
                subtitle_file=subtitle_file.resolve(),
                ass_file=ass_file.resolve(),
                preview_file=preview_file.resolve(),
                reframe_plan_file=(
                    reframe_plan_file.resolve()
                    if reframe_plan
                    else None
                ),
                reframe_mode=(
                    reframe_plan.mode
                    if reframe_plan
                    else "blurred_background"
                ),
                resolution=(
                    f"{self.settings.encoding.width}x"
                    f"{self.settings.encoding.height}"
                ),
                video_encoder=self.exporter.last_video_encoder,
                transcript=transcript_text,
                heuristic_score=candidate.score if candidate else None,
                selection_reasons=candidate.reasons if candidate else [],
                warnings=warnings,
                app_version=__version__,
            )
            (clip_directory / f"clip_{index:02d}.json").write_text(
                clip.model_dump_json(indent=2),
                encoding="utf-8",
            )
            rendered.append(clip)

        if progress_callback:
            progress_callback(
                "Рендер завершён",
                len(clip_ranges),
                len(clip_ranges),
            )
        report = RunReport(
            source_file=source.resolve(),
            output_directory=run_directory.resolve(),
            metadata=metadata,
            selected_candidates=candidates or [],
            clips=rendered,
            metrics=self.metrics.snapshot(
                source_duration_seconds=metadata.duration,
            ),
        )
        (run_directory / "report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        with (run_directory / "report.csv").open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "clip",
                    "start",
                    "end",
                    "duration",
                    "heuristic_score",
                    "selection_reasons",
                    "reframe_mode",
                    "video_encoder",
                    "video_file",
                ],
            )
            writer.writeheader()
            for clip in rendered:
                writer.writerow(
                    {
                        "clip": clip.index,
                        "start": round(clip.start, 3),
                        "end": round(clip.end, 3),
                        "duration": round(clip.duration, 3),
                        "heuristic_score": clip.heuristic_score or "",
                        "selection_reasons": "; ".join(
                            clip.selection_reasons
                        ),
                        "reframe_mode": clip.reframe_mode,
                        "video_encoder": clip.video_encoder or "",
                        "video_file": str(clip.video_file),
                    }
                )
        LOGGER.info("Completed. Output: %s", run_directory)
        return report
