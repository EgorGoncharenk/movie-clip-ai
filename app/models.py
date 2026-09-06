from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, computed_field, model_validator


class VideoMetadata(BaseModel):
    source_file: Path
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    video_start_time: float = 0
    video_codec: str
    audio_codec: str | None = None
    audio_tracks: int = Field(ge=0)
    sample_rate: int | None = Field(default=None, gt=0)
    file_size: int = Field(ge=0)
    estimated_temp_size: int = Field(ge=0)


class WordTiming(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    probability: float | None = Field(default=None, ge=0, le=1)


class TranscriptSegment(BaseModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[WordTiming] = Field(default_factory=list)
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TranscriptionResult(BaseModel):
    source_file: Path
    language: str
    language_probability: float | None = Field(default=None, ge=0, le=1)
    duration: float = Field(ge=0)
    model: str
    device: str
    segments: list[TranscriptSegment]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SceneSegment(BaseModel):
    index: int = Field(gt=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_duration(self):
        if self.end <= self.start:
            raise ValueError("Scene end must be later than its start")
        return self

    @computed_field
    @property
    def duration(self) -> float:
        return self.end - self.start


class SceneDetectionResult(BaseModel):
    source_file: Path
    duration: float = Field(gt=0)
    detector: str
    threshold: float = Field(gt=0)
    frame_skip: int = Field(default=0, ge=0)
    scenes: list[SceneSegment] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReframeKeyframe(BaseModel):
    time: float = Field(ge=0)
    center_x: float = Field(ge=0)
    center_y: float = Field(ge=0)
    face_count: int = Field(ge=0)
    tracked_face_score: float | None = Field(default=None, ge=0, le=1)
    scene_index: int | None = Field(default=None, gt=0)


class TransitionGuard(BaseModel):
    first_frame: int = Field(ge=1)
    last_frame: int = Field(ge=1)
    replace_frame: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_range(self):
        if self.last_frame < self.first_frame:
            raise ValueError("Transition guard range is reversed")
        if self.replace_frame >= self.first_frame:
            raise ValueError("Replacement frame must precede the guard")
        return self


class ReframePlan(BaseModel):
    source_file: Path
    clip_start: float = Field(ge=0)
    clip_end: float = Field(gt=0)
    content_x: int = Field(ge=0)
    content_y: int = Field(ge=0)
    content_width: int = Field(gt=0)
    content_height: int = Field(gt=0)
    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)
    mode: str
    sampled_frames: int = Field(ge=0)
    frames_with_faces: int = Field(ge=0)
    keyframes: list[ReframeKeyframe] = Field(default_factory=list)
    transition_guards: list[TransitionGuard] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_valid_clip(self):
        if self.clip_end <= self.clip_start:
            raise ValueError("Reframe clip end must be later than start")
        return self


class ClipRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def require_positive_duration(self):
        if self.end <= self.start:
            raise ValueError("Clip end must be later than its start")
        return self

    @computed_field
    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipCandidate(BaseModel):
    index: int = Field(gt=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    transcript: str

    @model_validator(mode="after")
    def require_positive_duration(self):
        if self.end <= self.start:
            raise ValueError("Candidate end must be later than its start")
        return self

    @computed_field
    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_range(self) -> ClipRange:
        return ClipRange(start=self.start, end=self.end)


class RenderedClip(BaseModel):
    index: int = Field(gt=0)
    source_file: Path
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    duration: float = Field(gt=0)
    video_file: Path
    subtitle_file: Path | None = None
    ass_file: Path | None = None
    preview_file: Path | None = None
    reframe_plan_file: Path | None = None
    reframe_mode: str
    resolution: str
    video_encoder: str | None = None
    transcript: str
    heuristic_score: float | None = Field(default=None, ge=0, le=100)
    selection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    app_version: str


class StageMetric(BaseModel):
    name: str
    elapsed_seconds: float = Field(ge=0)
    cpu_seconds: float = Field(ge=0)
    peak_memory_mb: float = Field(ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class PipelineMetrics(BaseModel):
    started_at: datetime
    completed_at: datetime
    total_elapsed_seconds: float = Field(ge=0)
    source_duration_seconds: float | None = Field(default=None, ge=0)
    speed_vs_realtime: float | None = Field(default=None, ge=0)
    resource_sampling_enabled: bool = False
    stages: list[StageMetric] = Field(default_factory=list)


class RunReport(BaseModel):
    source_file: Path
    output_directory: Path
    metadata: VideoMetadata
    selected_candidates: list[ClipCandidate] = Field(default_factory=list)
    clips: list[RenderedClip]
    metrics: PipelineMetrics | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
