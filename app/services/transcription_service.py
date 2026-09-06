from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable

from app.config import TranscriptionSection
from app.exceptions import OperationCancelled, TranscriptionError
from app.models import TranscriptSegment, TranscriptionResult, WordTiming
from app.utils.cuda_runtime import configure_cuda_dlls

LOGGER = logging.getLogger(__name__)
MINIMUM_MODEL_SIZE = 100 * 1024 * 1024


def find_cached_whisper_model(
    model: str,
    model_cache: Path,
) -> Path | None:
    repository = model_cache / f"models--Systran--faster-whisper-{model}"
    snapshots = repository / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in snapshots.iterdir():
        model_file = snapshot / "model.bin"
        if (
            snapshot.is_dir()
            and model_file.is_file()
            and model_file.stat().st_size >= MINIMUM_MODEL_SIZE
            and (snapshot / "config.json").is_file()
            and (snapshot / "tokenizer.json").is_file()
        ):
            return snapshot.resolve()
    return None


def cuda_device_count() -> int:
    try:
        configure_cuda_dlls()
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except (ImportError, RuntimeError, OSError):
        return 0


class TranscriptionService:
    def __init__(self, settings: TranscriptionSection) -> None:
        self.settings = settings

    def resolve_device(self) -> tuple[str, str]:
        requested = self.settings.device
        available = cuda_device_count() > 0
        if requested == "cuda" and not available:
            raise TranscriptionError(
                "CUDA was requested, but CTranslate2 cannot access a CUDA device. "
                "Use --device cpu or install compatible CUDA/cuDNN libraries."
            )
        if requested == "cuda" or (requested == "auto" and available):
            return "cuda", self.settings.cuda_compute_type
        return "cpu", self.settings.cpu_compute_type

    def require_local_model(self, model_cache: Path) -> Path:
        model_path = find_cached_whisper_model(
            self.settings.model,
            model_cache,
        )
        if model_path:
            return model_path
        raise TranscriptionError(
            f"Модель Whisper '{self.settings.model}' не установлена полностью. "
            "Выберите модель medium: она уже установлена на этом компьютере. "
            "Скачивание моделей во время обработки отключено."
        )

    @staticmethod
    def save(result: TranscriptionResult, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(source: Path) -> TranscriptionResult:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return TranscriptionResult.model_validate(payload)
        except (OSError, ValueError) as exc:
            raise TranscriptionError(
                f"Could not read transcription cache: {source}"
            ) from exc

    def transcribe(
        self,
        audio_file: Path,
        source_video: Path,
        destination: Path,
        *,
        overwrite: bool = False,
        model_cache: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> TranscriptionResult:
        if destination.exists() and not overwrite:
            cached = self.load(destination)
            if (
                cached.source_file.resolve() == source_video.resolve()
                and cached.model == self.settings.model
                and (
                    not self.settings.language
                    or cached.language == self.settings.language
                )
            ):
                LOGGER.info("Using cached transcription: %s", destination)
                return cached

        try:
            configure_cuda_dlls()
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed in the active environment."
            ) from exc

        device, compute_type = self.resolve_device()
        if model_cache is None:
            raise TranscriptionError(
                "Не задан локальный каталог моделей Whisper."
            )
        model_path = self.require_local_model(model_cache)
        LOGGER.info(
            "Loading Whisper model %s on %s (%s)",
            self.settings.model,
            device,
            compute_type,
        )

        def run_model(selected_device: str, selected_compute_type: str):
            model = WhisperModel(
                str(model_path),
                device=selected_device,
                compute_type=selected_compute_type,
                local_files_only=True,
            )
            generated, model_info = model.transcribe(
                str(audio_file),
                language=self.settings.language or None,
                beam_size=self.settings.beam_size,
                vad_filter=self.settings.vad_filter,
                word_timestamps=True,
            )
            collected = []
            for segment in generated:
                if cancel_check and cancel_check():
                    raise OperationCancelled("Обработка отменена пользователем.")
                collected.append(segment)
                if progress_callback:
                    progress_callback(
                        min(float(segment.end), float(model_info.duration)),
                        float(model_info.duration),
                    )
            return collected, model_info

        try:
            try:
                raw_segments, info = run_model(device, compute_type)
            except OperationCancelled:
                raise
            except Exception as cuda_exc:
                if device != "cuda" or self.settings.device == "cuda":
                    raise
                LOGGER.warning(
                    "CUDA transcription failed (%s). Retrying on CPU int8.",
                    cuda_exc,
                )
                device = "cpu"
                compute_type = self.settings.cpu_compute_type
                raw_segments, info = run_model(device, compute_type)

            segments: list[TranscriptSegment] = []
            for raw in raw_segments:
                confidence = None
                if raw.avg_logprob is not None:
                    confidence = min(1.0, max(0.0, math.exp(raw.avg_logprob)))
                words = [
                    WordTiming(
                        start=max(0.0, float(word.start)),
                        end=max(0.0, float(word.end)),
                        text=word.word,
                        probability=word.probability,
                    )
                    for word in (raw.words or [])
                    if word.start is not None and word.end is not None
                ]
                segments.append(
                    TranscriptSegment(
                        id=int(raw.id),
                        start=max(0.0, float(raw.start)),
                        end=max(0.0, float(raw.end)),
                        text=raw.text.strip(),
                        words=words,
                        no_speech_probability=raw.no_speech_prob,
                        confidence=confidence,
                    )
                )
        except OperationCancelled:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper transcription failed: {exc}"
            ) from exc

        result = TranscriptionResult(
            source_file=source_video.resolve(),
            language=info.language,
            language_probability=info.language_probability,
            duration=float(info.duration),
            model=self.settings.model,
            device=device,
            segments=segments,
        )
        self.save(result, destination)
        return result
