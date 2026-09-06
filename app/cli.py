from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from app import __version__
from app.config import (
    CandidateGenerationSection,
    EncodingSection,
    ReframeSection,
    SelectionSection,
    Settings,
    TranscriptionSection,
    load_settings,
)
from app.exceptions import ClipAppError
from app.pipeline import ClipPipeline
from app.services.ffmpeg_service import FFmpegService, find_executable
from app.services.transcription_service import (
    TranscriptionService,
    cuda_device_count,
    find_cached_whisper_model,
)
from app.utils.logging_setup import configure_logging
from app.utils.cuda_runtime import cuda_runtime_files
from app.utils.timecodes import parse_clip_range

app = typer.Typer(
    name="movie-clip-ai",
    help=(
        "Локальная подготовка вертикальных клипов из видео. "
        "Автоматический поиск моментов, Whisper, SRT и blurred background."
    ),
    no_args_is_help=True,
)
console = Console()

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        help="Путь к YAML-конфигурации.",
        dir_okay=False,
    ),
]
VideoArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Локальный видеофайл.",
    ),
]


def _settings(config: Path | None) -> Settings:
    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.file)
    return settings


def _fail(exc: Exception) -> None:
    console.print(f"[bold red]Ошибка:[/bold red] {exc}")
    raise typer.Exit(code=1)


def _apply_transcription_overrides(
    settings: Settings,
    *,
    language: str | None,
    model: str | None,
    device: str | None,
) -> Settings:
    changes = {}
    if language:
        changes["language"] = language
    if model:
        changes["model"] = model
    if device:
        changes["device"] = device
    if not changes:
        return settings
    transcription = TranscriptionSection.model_validate(
        settings.transcription.model_dump() | changes
    )
    return settings.model_copy(update={"transcription": transcription})


def _parse_ranges(values: list[str]):
    if not values:
        raise ClipAppError(
            "Укажите хотя бы один интервал через --clip "
            '"00:01:10-00:01:40".'
        )
    return [parse_clip_range(value) for value in values]


def _apply_analysis_overrides(
    settings: Settings,
    *,
    clips: int,
    min_duration: int,
    max_duration: int,
) -> Settings:
    candidate_generation = CandidateGenerationSection.model_validate(
        settings.candidate_generation.model_dump()
        | {
            "min_duration": min_duration,
            "max_duration": max_duration,
            "target_duration": min(
                max_duration,
                max(min_duration, (min_duration + max_duration) // 2),
            ),
        }
    )
    selection = SelectionSection.model_validate(
        settings.selection.model_dump() | {"clips": clips}
    )
    return settings.model_copy(
        update={
            "candidate_generation": candidate_generation,
            "selection": selection,
        }
    )


def _apply_encoding_overrides(
    settings: Settings,
    *,
    quality: str,
    encoder: str,
) -> Settings:
    encoding = EncodingSection.model_validate(
        settings.encoding.model_dump()
        | {
            "quality_mode": quality,
            "encoder": encoder,
        }
    )
    return settings.model_copy(update={"encoding": encoding})


def _apply_reframe_override(
    settings: Settings,
    *,
    reframe: str,
) -> Settings:
    section = ReframeSection.model_validate(
        settings.reframe.model_dump() | {"mode": reframe}
    )
    return settings.model_copy(update={"reframe": section})


def _print_metrics_table(metrics) -> None:
    table = Table(title="Benchmark Movie Clip AI")
    table.add_column("Этап")
    table.add_column("Время", justify="right")
    table.add_column("CPU", justify="right")
    table.add_column("Пик RAM", justify="right")
    table.add_column("Пик GPU", justify="right")
    for stage in metrics.stages:
        table.add_row(
            stage.name,
            f"{stage.elapsed_seconds:.2f} с",
            f"{stage.cpu_seconds:.2f} с",
            f"{stage.peak_memory_mb:.1f} МБ",
            (
                f"{stage.peak_gpu_memory_mb:.1f} МБ"
                if stage.peak_gpu_memory_mb is not None
                else "н/д"
            ),
        )
    console.print(table)
    speed = (
        f", скорость {metrics.speed_vs_realtime:.2f}x реального времени"
        if metrics.speed_vs_realtime is not None
        else ""
    )
    console.print(
        f"Общее время: {metrics.total_elapsed_seconds:.2f} с{speed}"
    )


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Проверить окружение и показать понятную диагностику."""
    try:
        settings = _settings(config)
        ffmpeg = FFmpegService()
        table = Table(title="Диагностика Movie Clip AI")
        table.add_column("Компонент")
        table.add_column("Статус")
        table.add_column("Подробности")

        python_ok = (3, 11) <= sys.version_info[:2] < (3, 13)
        table.add_row(
            "Python",
            "[green]OK[/green]" if python_ok else "[yellow]Внимание[/yellow]",
            sys.version.split()[0],
        )
        for tool in ("ffmpeg", "ffprobe"):
            path = getattr(ffmpeg, tool)
            status = "[green]OK[/green]" if path else "[red]Нет[/red]"
            details = ffmpeg.version(tool) if path else "Не найден в PATH"
            table.add_row(tool, status, details)

        subtitle_ok = False
        if ffmpeg.ffmpeg:
            subtitle_ok = ffmpeg.has_subtitles_filter()
        table.add_row(
            "FFmpeg subtitles",
            "[green]OK[/green]" if subtitle_ok else "[red]Нет[/red]",
            "libass доступен" if subtitle_ok else "Нужна сборка FFmpeg с libass",
        )

        nvidia_smi = find_executable("nvidia-smi")
        gpu_details = "NVIDIA GPU не обнаружена"
        if nvidia_smi:
            result = subprocess.run(
                [
                    str(nvidia_smi),
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_details = result.stdout.strip()
        table.add_row(
            "NVIDIA GPU",
            "[green]OK[/green]" if nvidia_smi else "[yellow]Не найдена[/yellow]",
            gpu_details,
        )

        cuda_count = cuda_device_count()
        runtime_files = cuda_runtime_files()
        runtime_ok = all(runtime_files.values())
        table.add_row(
            "CUDA для Whisper",
            (
                "[green]OK[/green]"
                if cuda_count and runtime_ok
                else "[yellow]CPU fallback[/yellow]"
            ),
            (
                f"Устройств: {cuda_count}; "
                + ", ".join(
                    f"{name}: {'OK' if present else 'нет'}"
                    for name, present in runtime_files.items()
                )
                if cuda_count
                else "Whisper будет использовать CPU int8"
            ),
        )

        output_parent = settings.paths.output.resolve().parent
        free = shutil.disk_usage(output_parent).free
        free_gb = free / (1024**3)
        table.add_row(
            "Свободное место",
            "[green]OK[/green]" if free_gb >= 10 else "[yellow]Мало[/yellow]",
            f"{free_gb:.1f} ГБ на {output_parent.drive or output_parent}",
        )

        model_cache = settings.paths.temp / "models"
        model_path = find_cached_whisper_model(
            settings.transcription.model,
            model_cache,
        )
        model_present = model_path is not None
        table.add_row(
            f"Whisper {settings.transcription.model}",
            "[green]Кеш найден[/green]" if model_present else "[yellow]Не загружена[/yellow]",
            (
                str(model_path)
                if model_present
                else "Будет загружена автоматически при первой транскрибации"
            ),
        )
        console.print(table)
    except (ClipAppError, ValidationError, OSError) as exc:
        _fail(exc)


@app.command("inspect")
def inspect_video(video: VideoArgument, config: ConfigOption = None) -> None:
    """Прочитать технические метаданные видео через FFprobe."""
    try:
        pipeline = ClipPipeline(_settings(config))
        metadata = pipeline.inspect(video)
        console.print_json(metadata.model_dump_json(indent=2))
    except (ClipAppError, ValidationError) as exc:
        _fail(exc)


@app.command()
def transcribe(
    video: VideoArgument,
    config: ConfigOption = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    whisper_model: Annotated[
        str | None,
        typer.Option(
            "--whisper-model",
            help="small, medium, large-v3 или turbo.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="auto, cpu или cuda."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Пересоздать кеш транскрипции."),
    ] = False,
) -> None:
    """Извлечь аудио и сохранить транскрипцию Whisper в JSON."""
    try:
        settings = _apply_transcription_overrides(
            _settings(config),
            language=language,
            model=whisper_model,
            device=device,
        )
        result, transcript_file, _ = ClipPipeline(settings).transcribe(
            video,
            overwrite=overwrite,
        )
        console.print(
            f"[green]Готово:[/green] {len(result.segments)} сегментов, "
            f"язык {result.language}, файл {transcript_file.resolve()}"
        )
    except (ClipAppError, ValidationError, ValueError) as exc:
        _fail(exc)


@app.command()
def render(
    video: VideoArgument,
    transcript: Annotated[
        Path,
        typer.Option(
            "--transcript",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Готовый JSON транскрипции.",
        ),
    ],
    clip: Annotated[
        list[str],
        typer.Option(
            "--clip",
            help='Интервал START-END. Параметр можно повторять.',
        ),
    ] = [],
    config: ConfigOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", file_okay=False),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Отрендерить ручные интервалы по готовой транскрипции."""
    try:
        settings = _settings(config)
        transcription = TranscriptionService.load(transcript)
        report = ClipPipeline(settings).render(
            video,
            transcription,
            _parse_ranges(clip),
            output_root=output,
            overwrite=overwrite,
        )
        console.print(
            f"[bold green]Готово:[/bold green] {len(report.clips)} клипов в "
            f"{report.output_directory}"
        )
    except (ClipAppError, ValidationError, ValueError) as exc:
        _fail(exc)


@app.command()
def run(
    video: VideoArgument,
    clip: Annotated[
        list[str],
        typer.Option(
            "--clip",
            help='Интервал START-END. Пример: --clip "00:01:10-00:01:40".',
        ),
    ] = [],
    config: ConfigOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", file_okay=False),
    ] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model"),
    ] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    clips: Annotated[
        int,
        typer.Option(
            "--clips",
            min=1,
            max=20,
            help="Количество клипов в автоматическом режиме.",
        ),
    ] = 3,
    min_duration: Annotated[
        int,
        typer.Option("--min-duration", min=10, max=180),
    ] = 20,
    max_duration: Annotated[
        int,
        typer.Option("--max-duration", min=10, max=180),
    ] = 60,
    quality: Annotated[
        str,
        typer.Option(
            "--quality",
            help="fast, balanced или quality.",
        ),
    ] = "balanced",
    encoder: Annotated[
        str,
        typer.Option(
            "--encoder",
            help="auto, nvenc или cpu.",
        ),
    ] = "auto",
    reframe: Annotated[
        str,
        typer.Option(
            "--reframe",
            help="smart, center или blurred.",
        ),
    ] = "smart",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    keep_temp: Annotated[bool, typer.Option("--keep-temp")] = False,
) -> None:
    """Автоматически найти моменты и создать вертикальные клипы."""
    audio_file: Path | None = None
    try:
        settings = _apply_reframe_override(
            _apply_encoding_overrides(
                _apply_analysis_overrides(
                    _apply_transcription_overrides(
                        _settings(config),
                        language=language,
                        model=whisper_model,
                        device=device,
                    ),
                    clips=clips,
                    min_duration=min_duration,
                    max_duration=max_duration,
                ),
                quality=quality,
                encoder=encoder,
            ),
            reframe=reframe,
        )
        ranges = [parse_clip_range(value) for value in clip]
        pipeline = ClipPipeline(settings)
        transcription, transcript_file, audio_file = pipeline.transcribe(
            video,
            overwrite=overwrite,
        )
        console.print(f"Транскрипция: {transcript_file.resolve()}")
        selected = []
        if not ranges:
            selected = pipeline.analyze(video, transcription)
            ranges = [candidate.as_range() for candidate in selected]
            if not ranges:
                raise ClipAppError(
                    "Не найдено подходящих фрагментов. "
                    "Попробуйте уменьшить --min-duration."
                )
            for candidate in selected:
                console.print(
                    f"#{candidate.index}: {candidate.start:.1f}-"
                    f"{candidate.end:.1f}с, оценка {candidate.score:.1f}"
                )
        report = pipeline.render(
            video,
            transcription,
            ranges,
            output_root=output,
            overwrite=overwrite,
            candidates=selected,
        )
        if audio_file.exists() and not keep_temp:
            audio_file.unlink()
        console.print(
            f"[bold green]Готово:[/bold green] {len(report.clips)} клипов в "
            f"{report.output_directory}"
        )
    except (ClipAppError, ValidationError, ValueError, OSError) as exc:
        _fail(exc)


@app.command()
def benchmark(
    video: VideoArgument,
    config: ConfigOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", file_okay=False),
    ] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    whisper_model: Annotated[
        str | None,
        typer.Option("--whisper-model"),
    ] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    clips: Annotated[
        int,
        typer.Option(
            "--clips",
            min=1,
            max=20,
            help="Количество клипов для контрольного экспорта.",
        ),
    ] = 1,
    min_duration: Annotated[
        int,
        typer.Option("--min-duration", min=10, max=180),
    ] = 20,
    max_duration: Annotated[
        int,
        typer.Option("--max-duration", min=10, max=180),
    ] = 60,
    quality: Annotated[
        str,
        typer.Option(
            "--quality",
            help="fast, balanced или quality.",
        ),
    ] = "balanced",
    encoder: Annotated[
        str,
        typer.Option(
            "--encoder",
            help="auto, nvenc или cpu.",
        ),
    ] = "auto",
    reframe: Annotated[
        str,
        typer.Option(
            "--reframe",
            help="smart, center или blurred.",
        ),
    ] = "smart",
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Не использовать кеш транскрипции.",
        ),
    ] = False,
    keep_temp: Annotated[
        bool,
        typer.Option("--keep-temp", help="Оставить извлечённый WAV."),
    ] = False,
) -> None:
    """Измерить время, RAM и GPU на полном конвейере."""
    audio_file: Path | None = None
    try:
        settings = _apply_reframe_override(
            _apply_encoding_overrides(
                _apply_analysis_overrides(
                    _apply_transcription_overrides(
                        _settings(config),
                        language=language,
                        model=whisper_model,
                        device=device,
                    ),
                    clips=clips,
                    min_duration=min_duration,
                    max_duration=max_duration,
                ),
                quality=quality,
                encoder=encoder,
            ),
            reframe=reframe,
        )
        pipeline = ClipPipeline(settings, sample_resources=True)
        transcription, _, audio_file = pipeline.transcribe(
            video,
            overwrite=overwrite,
        )
        selected = pipeline.analyze(video, transcription)
        if not selected:
            raise ClipAppError(
                "Не найдено подходящих фрагментов для benchmark."
            )
        report = pipeline.render(
            video,
            transcription,
            [candidate.as_range() for candidate in selected],
            output_root=output,
            overwrite=overwrite,
            candidates=selected,
        )
        if audio_file.exists() and not keep_temp:
            audio_file.unlink()
        if report.metrics is None:
            raise ClipAppError("Метрики benchmark не сформированы.")

        benchmark_file = report.output_directory / "benchmark.json"
        benchmark_file.write_text(
            report.metrics.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _print_metrics_table(report.metrics)
        console.print(
            f"[bold green]Benchmark готов:[/bold green] "
            f"{benchmark_file.resolve()} (Movie Clip AI {__version__})"
        )
    except (ClipAppError, ValidationError, ValueError, OSError) as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
