from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class AppSection(BaseModel):
    name: str = "Movie Clip AI"
    version: str = "0.8.0"


class PathsSection(BaseModel):
    input: Path = Path("input")
    output: Path = Path("output")
    temp: Path = Path("temp")


class TranscriptionSection(BaseModel):
    model: Literal["small", "medium", "large-v3", "turbo"] = "medium"
    language: str = "ru"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_compute_type: str = "int8"
    cuda_compute_type: str = "float16"
    beam_size: int = Field(default=5, ge=1, le=20)
    vad_filter: bool = True


class SubtitleSection(BaseModel):
    enabled: bool = True
    font: str = "Arial"
    font_size: int = Field(default=48, ge=12, le=160)
    bold: bool = True
    outline: int = Field(default=3, ge=0, le=10)
    shadow: int = Field(default=1, ge=0, le=10)
    margin_bottom: int = Field(default=220, ge=0, le=800)
    max_chars: int = Field(default=38, ge=10, le=100)
    words_per_caption: int = Field(default=3, ge=2, le=5)
    max_word_gap: float = Field(default=0.55, ge=0.1, le=2)


class CandidateGenerationSection(BaseModel):
    min_duration: int = Field(default=20, ge=10, le=180)
    max_duration: int = Field(default=60, ge=10, le=180)
    target_duration: int = Field(default=35, ge=10, le=180)
    padding_before: float = Field(default=0.8, ge=0, le=3)
    padding_after: float = Field(default=1.2, ge=0, le=3)
    minimum_pause: float = Field(default=0.45, ge=0, le=5)
    scene_snap_window: float = Field(default=2.0, ge=0, le=10)

    @field_validator("max_duration")
    @classmethod
    def require_valid_duration_range(cls, value: int, info):
        minimum = info.data.get("min_duration")
        if minimum and value < minimum:
            raise ValueError(
                "candidate_generation.max_duration must be >= min_duration"
            )
        return value


class SelectionSection(BaseModel):
    clips: int = Field(default=3, ge=1, le=20)
    max_overlap_ratio: float = Field(default=0.2, ge=0, le=1)
    max_text_similarity: float = Field(default=0.75, ge=0, le=1)


class SceneDetectionSection(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=27.0, ge=1, le=100)
    min_scene_length: float = Field(default=0.6, ge=0.1, le=30)
    frame_skip: int = Field(default=3, ge=0, le=20)


class ReframeSection(BaseModel):
    mode: Literal["smart", "center", "blurred"] = "smart"
    face_model: Path = Path(
        "assets/models/face_detection_yunet_2026may.onnx"
    )
    sample_interval: float = Field(default=0.5, ge=0.2, le=3)
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1)
    smoothing: float = Field(default=0.45, ge=0.05, le=1)
    deadzone_ratio: float = Field(default=0.04, ge=0, le=0.2)
    max_pan_speed_ratio: float = Field(default=0.18, ge=0.01, le=1)
    lock_to_scenes: bool = False
    refine_scene_cuts: bool = True
    cut_refine_window: float = Field(default=0.35, ge=0.1, le=1)
    transition_lead_frames: int = Field(default=2, ge=0, le=6)
    transition_hold_frames: int = Field(default=2, ge=0, le=6)


class EncodingSection(BaseModel):
    width: int = Field(default=1080, ge=320, le=4320)
    height: int = Field(default=1920, ge=320, le=7680)
    quality_mode: Literal["fast", "balanced", "quality", "custom"] = "balanced"
    encoder: Literal["auto", "cpu", "nvenc"] = "auto"
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = Field(default=20, ge=0, le=51)
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    background_blur: int = Field(default=30, ge=1, le=100)
    remove_black_bars: bool = True

    @field_validator("height")
    @classmethod
    def require_vertical_resolution(cls, value: int, info):
        width = info.data.get("width")
        if width and value <= width:
            raise ValueError("encoding.height must be greater than encoding.width")
        return value


class LoggingSection(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path = Path("logs/movie_clip_ai.log")


class Settings(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    transcription: TranscriptionSection = Field(
        default_factory=TranscriptionSection
    )
    subtitles: SubtitleSection = Field(default_factory=SubtitleSection)
    candidate_generation: CandidateGenerationSection = Field(
        default_factory=CandidateGenerationSection
    )
    selection: SelectionSection = Field(default_factory=SelectionSection)
    scene_detection: SceneDetectionSection = Field(
        default_factory=SceneDetectionSection
    )
    reframe: ReframeSection = Field(default_factory=ReframeSection)
    encoding: EncodingSection = Field(default_factory=EncodingSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or Path("config.yaml")
    if not config_path.exists():
        return Settings()

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return Settings.model_validate(raw)
