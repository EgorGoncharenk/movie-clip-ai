from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from pydantic import ValidationError
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from app.config import SceneDetectionSection
from app.exceptions import ClipAppError, OperationCancelled
from app.models import SceneDetectionResult, SceneSegment

LOGGER = logging.getLogger(__name__)


class SceneDetectionService:
    def __init__(self, settings: SceneDetectionSection) -> None:
        self.settings = settings

    def detect(
        self,
        source: Path,
        cache_file: Path,
        *,
        video_duration: float,
        overwrite: bool = False,
        progress_callback: Callable[[float, float], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> SceneDetectionResult:
        if cache_file.exists() and not overwrite:
            cached = self.load_cached(source, cache_file)
            if cached:
                LOGGER.info("Using cached scene analysis: %s", cache_file)
                if progress_callback:
                    progress_callback(video_duration, video_duration)
                return cached

        if not self.settings.enabled:
            result = SceneDetectionResult(
                source_file=source.resolve(),
                duration=video_duration,
                detector="disabled",
                threshold=self.settings.threshold,
                frame_skip=self.settings.frame_skip,
                scenes=[
                    SceneSegment(index=1, start=0, end=video_duration)
                ],
            )
            self._save(result, cache_file)
            return result

        try:
            video = open_video(str(source.resolve()))
            manager = SceneManager()
            manager.auto_downscale = True
            manager.add_detector(
                ContentDetector(
                    threshold=self.settings.threshold,
                    min_scene_len=self.settings.min_scene_length,
                )
            )
            monitor_stop = threading.Event()

            def monitor_progress() -> None:
                last_percent = -1
                while not monitor_stop.wait(0.5):
                    if cancel_check and cancel_check():
                        manager.stop()
                        return
                    if not progress_callback:
                        continue
                    try:
                        position = min(video_duration, video.position.seconds)
                    except Exception:
                        continue
                    percent = int(position / max(video_duration, 1) * 100)
                    if percent != last_percent:
                        last_percent = percent
                        progress_callback(position, video_duration)

            monitor = threading.Thread(
                target=monitor_progress,
                name="scene-progress",
                daemon=True,
            )
            monitor.start()
            try:
                manager.detect_scenes(
                    video,
                    frame_skip=self.settings.frame_skip,
                    show_progress=False,
                )
            finally:
                monitor_stop.set()
                monitor.join(timeout=1)
            if cancel_check and cancel_check():
                raise OperationCancelled("Обработка отменена пользователем.")
            detected = manager.get_scene_list(start_in_scene=True)
            if progress_callback:
                progress_callback(video_duration, video_duration)
        except OperationCancelled:
            raise
        except Exception as exc:
            raise ClipAppError(
                f"Не удалось выполнить анализ сцен: {exc}"
            ) from exc

        scenes = [
            SceneSegment(
                index=index,
                start=max(0.0, start.seconds),
                end=min(video_duration, end.seconds),
            )
            for index, (start, end) in enumerate(detected, start=1)
            if end.seconds > start.seconds
        ]
        if not scenes:
            scenes = [
                SceneSegment(index=1, start=0, end=video_duration)
            ]
        result = SceneDetectionResult(
            source_file=source.resolve(),
            duration=video_duration,
            detector="content",
            threshold=self.settings.threshold,
            frame_skip=self.settings.frame_skip,
            scenes=scenes,
        )
        self._save(result, cache_file)
        LOGGER.info("Detected %d scenes", len(scenes))
        return result

    def load_cached(
        self,
        source: Path,
        cache_file: Path,
    ) -> SceneDetectionResult | None:
        if not cache_file.exists():
            return None
        try:
            cached = SceneDetectionResult.model_validate_json(
                cache_file.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            LOGGER.warning("Ignoring unreadable scene cache: %s", cache_file)
            return None
        if (
            cached.source_file.resolve() != source.resolve()
            or cached.threshold != self.settings.threshold
            or cached.frame_skip not in {0, self.settings.frame_skip}
        ):
            return None
        return cached

    @staticmethod
    def _save(result: SceneDetectionResult, cache_file: Path) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
