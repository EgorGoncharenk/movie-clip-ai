from __future__ import annotations

import logging
import math
import os
import shutil
import statistics
from pathlib import Path
from typing import Callable

import cv2

from app.config import EncodingSection, ReframeSection
from app.exceptions import ClipAppError, OperationCancelled
from app.models import (
    ClipRange,
    ReframeKeyframe,
    ReframePlan,
    SceneSegment,
    TransitionGuard,
    VideoMetadata,
)

LOGGER = logging.getLogger(__name__)


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _parse_content_crop(
    crop: str | None,
    metadata: VideoMetadata,
) -> tuple[int, int, int, int]:
    if not crop:
        return 0, 0, metadata.width, metadata.height
    try:
        width, height, x, y = (int(part) for part in crop.split(":"))
    except (TypeError, ValueError):
        return 0, 0, metadata.width, metadata.height
    return x, y, width, height


class FaceTrackingService:
    def __init__(
        self,
        settings: ReframeSection,
        encoding: EncodingSection,
    ) -> None:
        self.settings = settings
        self.encoding = encoding

    def plan(
        self,
        source: Path,
        clip_range: ClipRange,
        metadata: VideoMetadata,
        *,
        content_crop: str | None = None,
        scenes: list[SceneSegment] | None = None,
        progress_callback: Callable[[float, float], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ReframePlan:
        content_x, content_y, content_width, content_height = (
            _parse_content_crop(content_crop, metadata)
        )
        target_ratio = self.encoding.width / self.encoding.height
        content_ratio = content_width / content_height
        if content_ratio > target_ratio:
            crop_height = _even(content_height)
            crop_width = _even(crop_height * target_ratio)
        else:
            crop_width = _even(content_width)
            crop_height = _even(crop_width / target_ratio)

        if self.settings.mode != "smart":
            return self._center_plan(
                source,
                clip_range,
                content_x,
                content_y,
                content_width,
                content_height,
                crop_width,
                crop_height,
                mode=f"{self.settings.mode}_crop",
            )

        model_path = self.settings.face_model.resolve()
        if not model_path.is_file():
            raise ClipAppError(
                f"Локальная модель детекции лиц не найдена: {model_path}"
            )
        model_path = self._ascii_model_path(model_path)
        tracking_scenes = list(scenes or [])
        if self.settings.refine_scene_cuts and tracking_scenes:
            tracking_scenes = self._refine_scene_boundaries(
                source,
                tracking_scenes,
                clip_range,
                content_x,
                content_y,
                content_width,
                content_height,
                metadata.fps,
                metadata.video_start_time,
            )

        capture = cv2.VideoCapture(str(source.resolve()))
        if not capture.isOpened():
            raise ClipAppError("Не удалось открыть видео для отслеживания лиц.")
        fps = capture.get(cv2.CAP_PROP_FPS) or metadata.fps
        detector_width = min(640, content_width)
        detector_height = max(
            2,
            round(content_height * detector_width / content_width),
        )
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (detector_width, detector_height),
            self.settings.confidence_threshold,
            0.3,
            5000,
        )
        capture.set(cv2.CAP_PROP_POS_MSEC, clip_range.start * 1000)
        sample_interval = self.settings.sample_interval
        next_sample = 0.0
        sampled = 0
        frames_with_faces = 0
        previous_center: tuple[float, float] | None = None
        previous_sample_time: float | None = None
        previous_scene: int | None = None
        keyframes: list[ReframeKeyframe] = []
        try:
            while capture.isOpened():
                if cancel_check and cancel_check():
                    raise OperationCancelled("Обработка отменена пользователем.")
                success = capture.grab()
                if not success:
                    break
                absolute_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                relative_time = max(0.0, absolute_time - clip_range.start)
                if absolute_time > clip_range.end + 0.05:
                    break
                if relative_time + 0.001 < next_sample:
                    continue
                success, frame = capture.retrieve()
                if not success:
                    continue
                next_sample += sample_interval
                sampled += 1
                content = frame[
                    content_y : content_y + content_height,
                    content_x : content_x + content_width,
                ]
                resized = cv2.resize(
                    content,
                    (detector_width, detector_height),
                    interpolation=cv2.INTER_AREA,
                )
                detector.setInputSize((detector_width, detector_height))
                _, detected = detector.detect(resized)
                faces = self._scale_faces(
                    detected,
                    content_width / detector_width,
                    content_height / detector_height,
                )
                scene = self._scene_at(
                    absolute_time,
                    tracking_scenes,
                )
                scene_index = scene.index if scene else None
                scene_changed = (
                    previous_scene is not None
                    and scene_index is not None
                    and scene_index != previous_scene
                )
                if scene_changed:
                    previous_center = None
                selected = self._select_face(
                    faces,
                    previous_center,
                    content_width,
                    content_height,
                )
                if selected:
                    frames_with_faces += 1
                    target = (
                        selected[0] + selected[2] / 2,
                        selected[1] + selected[3] / 2,
                    )
                    if self.settings.lock_to_scenes and tracking_scenes:
                        center = target
                    else:
                        center = self._smooth_center(
                            previous_center,
                            target,
                            content_width,
                            content_height,
                            elapsed=(
                                sample_interval
                                if previous_sample_time is None
                                else max(
                                    0.001,
                                    relative_time - previous_sample_time,
                                )
                            ),
                        )
                    score = selected[4]
                else:
                    center = previous_center or (
                        content_width / 2,
                        content_height / 2,
                    )
                    score = None
                center = self._clamp_center(
                    center,
                    content_width,
                    content_height,
                    crop_width,
                    crop_height,
                )
                previous_center = center
                previous_sample_time = relative_time
                previous_scene = scene_index
                keyframe_time = (
                    0.0
                    if sampled == 1
                    else round(relative_time, 3)
                )
                if scene_changed and scene is not None:
                    keyframe_time = round(
                        max(0.0, scene.start - clip_range.start),
                        3,
                    )
                keyframes.append(
                    ReframeKeyframe(
                        time=keyframe_time,
                        center_x=round(center[0], 2),
                        center_y=round(center[1], 2),
                        face_count=len(faces),
                        tracked_face_score=score,
                        scene_index=scene_index,
                    )
                )
                if progress_callback:
                    progress_callback(relative_time, clip_range.duration)
        finally:
            capture.release()

        if not keyframes:
            return self._center_plan(
                source,
                clip_range,
                content_x,
                content_y,
                content_width,
                content_height,
                crop_width,
                crop_height,
                mode="center_crop_no_frames",
            )
        if progress_callback:
            progress_callback(clip_range.duration, clip_range.duration)
        if self.settings.lock_to_scenes:
            keyframes = self._lock_keyframes_to_scenes(
                keyframes,
                tracking_scenes,
                clip_range,
                content_width,
                content_height,
                crop_width,
                crop_height,
            )
        keyframes, transition_guards = self._apply_transition_guards(
            keyframes,
            fps,
            metadata.video_start_time,
            clip_range,
        )
        mode = "smart_face" if frames_with_faces else "center_crop_no_faces"
        LOGGER.info(
            "Reframe plan: %d/%d frames with faces",
            frames_with_faces,
            sampled,
        )
        return ReframePlan(
            source_file=source.resolve(),
            clip_start=clip_range.start,
            clip_end=clip_range.end,
            content_x=content_x,
            content_y=content_y,
            content_width=content_width,
            content_height=content_height,
            crop_width=crop_width,
            crop_height=crop_height,
            mode=mode,
            sampled_frames=sampled,
            frames_with_faces=frames_with_faces,
            keyframes=keyframes,
            transition_guards=transition_guards,
        )

    @staticmethod
    def _ascii_model_path(source: Path) -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            return source
        destination = (
            Path(local_app_data)
            / "MovieClipAI"
            / "models"
            / source.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            not destination.exists()
            or destination.stat().st_size != source.stat().st_size
        ):
            shutil.copyfile(source, destination)
        return destination

    @staticmethod
    def _scale_faces(
        detected,
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[float, float, float, float, float]]:
        if detected is None:
            return []
        return [
            (
                float(face[0]) * scale_x,
                float(face[1]) * scale_y,
                float(face[2]) * scale_x,
                float(face[3]) * scale_y,
                float(face[-1]),
            )
            for face in detected
        ]

    @staticmethod
    def _scene_at(
        timestamp: float,
        scenes: list[SceneSegment],
    ) -> SceneSegment | None:
        for scene in scenes:
            if scene.start <= timestamp < scene.end:
                return scene
        return None

    def _refine_scene_boundaries(
        self,
        source: Path,
        scenes: list[SceneSegment],
        clip_range: ClipRange,
        content_x: int,
        content_y: int,
        content_width: int,
        content_height: int,
        fps: float,
        video_start_time: float,
    ) -> list[SceneSegment]:
        active = [
            scene.model_copy()
            for scene in scenes
            if scene.end > clip_range.start
            and scene.start < clip_range.end
        ]
        if len(active) < 2 or fps <= 0:
            return active or scenes

        refined_count = 0
        capture = cv2.VideoCapture(str(source.resolve()))
        if not capture.isOpened():
            return active
        try:
            for index in range(1, len(active)):
                approximate = active[index].start
                refined = self._find_hard_cut(
                    capture,
                    approximate,
                    content_x,
                    content_y,
                    content_width,
                    content_height,
                    fps,
                    video_start_time,
                )
                lower = active[index - 1].start + 1 / fps
                upper = active[index].end - 1 / fps
                refined = min(max(refined, lower), upper)
                if abs(refined - approximate) >= 0.5 / fps:
                    refined_count += 1
                active[index - 1] = active[index - 1].model_copy(
                    update={"end": refined}
                )
                active[index] = active[index].model_copy(
                    update={"start": refined}
                )
        finally:
            capture.release()
        if refined_count:
            LOGGER.info(
                "Refined %d scene boundaries to exact video frames",
                refined_count,
            )
        return active

    def _find_hard_cut(
        self,
        capture: cv2.VideoCapture,
        approximate: float,
        content_x: int,
        content_y: int,
        content_width: int,
        content_height: int,
        fps: float,
        video_start_time: float = 0,
    ) -> float:
        radius = self.settings.cut_refine_window
        first_frame = max(
            0,
            math.floor(
                (approximate - radius - video_start_time) * fps
            ),
        )
        last_frame = max(
            first_frame + 1,
            math.ceil(
                (approximate + radius - video_start_time) * fps
            ),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        previous = None
        best_score = -1.0
        best_frame = round(approximate * fps)
        for frame_number in range(first_frame, last_frame + 1):
            success, frame = capture.read()
            if not success:
                break
            content = frame[
                content_y : content_y + content_height,
                content_x : content_x + content_width,
            ]
            detector_width = min(320, content_width)
            detector_height = max(
                2,
                round(
                    content_height * detector_width / content_width
                ),
            )
            gray = cv2.cvtColor(
                cv2.resize(
                    content,
                    (detector_width, detector_height),
                    interpolation=cv2.INTER_AREA,
                ),
                cv2.COLOR_BGR2GRAY,
            )
            if previous is not None:
                score = float(cv2.absdiff(previous, gray).mean())
                if score > best_score:
                    best_score = score
                    best_frame = frame_number
            previous = gray
        return video_start_time + best_frame / fps

    @staticmethod
    def _lock_keyframes_to_scenes(
        keyframes: list[ReframeKeyframe],
        scenes: list[SceneSegment],
        clip_range: ClipRange,
        content_width: int,
        content_height: int,
        crop_width: int,
        crop_height: int,
    ) -> list[ReframeKeyframe]:
        if not keyframes or not scenes:
            return keyframes
        scene_map = {scene.index: scene for scene in scenes}
        if not any(
            keyframe.scene_index in scene_map for keyframe in keyframes
        ):
            return keyframes

        locked: list[ReframeKeyframe] = []
        position = 0
        while position < len(keyframes):
            scene_index = keyframes[position].scene_index
            end = position + 1
            while (
                end < len(keyframes)
                and keyframes[end].scene_index == scene_index
            ):
                end += 1
            group = keyframes[position:end]
            scene = scene_map.get(scene_index)
            if scene is None:
                locked.extend(group)
                position = end
                continue

            detected = [
                keyframe
                for keyframe in group
                if keyframe.tracked_face_score is not None
            ]
            anchors = detected or group
            center = FaceTrackingService._clamp_center(
                (
                    statistics.median(
                        keyframe.center_x for keyframe in anchors
                    ),
                    statistics.median(
                        keyframe.center_y for keyframe in anchors
                    ),
                ),
                content_width,
                content_height,
                crop_width,
                crop_height,
            )
            representative = min(
                anchors,
                key=lambda keyframe: math.hypot(
                    keyframe.center_x - center[0],
                    keyframe.center_y - center[1],
                ),
            )
            locked.append(
                representative.model_copy(
                    update={
                        "time": round(
                            max(0.0, scene.start - clip_range.start),
                            6,
                        ),
                        "center_x": round(center[0], 2),
                        "center_y": round(center[1], 2),
                        "face_count": max(
                            keyframe.face_count for keyframe in group
                        ),
                    }
                )
            )
            if (
                len(locked) >= 2
                and abs(locked[-1].time - locked[-2].time) < 0.0005
            ):
                latest = locked.pop()
                locked[-1] = latest
            position = end
        return locked

    @staticmethod
    def _select_face(
        faces: list[tuple[float, float, float, float, float]],
        previous_center: tuple[float, float] | None,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float, float] | None:
        if not faces:
            return None

        def rank(face) -> float:
            x, y, face_width, face_height, confidence = face
            area = face_width * face_height / max(width * height, 1)
            value = math.sqrt(area) * 2.2 + confidence * 0.25
            if previous_center:
                center_x = x + face_width / 2
                center_y = y + face_height / 2
                distance = math.hypot(
                    (center_x - previous_center[0]) / width,
                    (center_y - previous_center[1]) / height,
                )
                value -= distance * 0.55
            return value

        return max(faces, key=rank)

    def _smooth_center(
        self,
        previous: tuple[float, float] | None,
        target: tuple[float, float],
        width: int,
        height: int,
        *,
        elapsed: float = 1.0,
    ) -> tuple[float, float]:
        if previous is None:
            return target
        delta_x = target[0] - previous[0]
        delta_y = target[1] - previous[1]
        if abs(delta_x) < width * self.settings.deadzone_ratio:
            delta_x = 0
        if abs(delta_y) < height * self.settings.deadzone_ratio:
            delta_y = 0
        alpha = self.settings.smoothing
        delta_x *= alpha
        delta_y *= alpha
        max_x = width * self.settings.max_pan_speed_ratio * elapsed
        max_y = height * self.settings.max_pan_speed_ratio * elapsed
        return (
            previous[0] + min(max(delta_x, -max_x), max_x),
            previous[1] + min(max(delta_y, -max_y), max_y),
        )

    def _apply_transition_guards(
        self,
        keyframes: list[ReframeKeyframe],
        fps: float,
        video_start_time: float,
        clip_range: ClipRange,
    ) -> tuple[list[ReframeKeyframe], list[TransitionGuard]]:
        hold_frames = self.settings.transition_hold_frames
        lead_frames = self.settings.transition_lead_frames
        if (
            len(keyframes) < 2
            or hold_frames + lead_frames <= 0
            or fps <= 0
        ):
            return keyframes, []

        first_absolute_frame = math.ceil(
            (clip_range.start - video_start_time) * fps - 1e-6
        )
        first_frame_time = (
            video_start_time
            + first_absolute_frame / fps
            - clip_range.start
        )
        guarded = [keyframes[0]]
        guards: list[TransitionGuard] = []
        total_frames = max(1, math.ceil(clip_range.duration * fps))
        for index, keyframe in enumerate(keyframes[1:], start=1):
            previous_keyframe = keyframes[index - 1]
            if (
                previous_keyframe.scene_index is None
                or keyframe.scene_index is None
                or previous_keyframe.scene_index
                == keyframe.scene_index
            ):
                guarded.append(keyframe)
                continue
            first_new_frame = round(
                (keyframe.time - first_frame_time) * fps
            )
            first_guard_frame = max(
                1,
                first_new_frame - lead_frames,
            )
            if first_new_frame < 1 or first_guard_frame < 1:
                guarded[-1] = keyframe.model_copy(update={"time": 0.0})
                continue
            previous_boundary = previous_keyframe.time
            stable_time = (
                previous_boundary + keyframe.time
            ) / 2
            replace_frame = min(
                first_guard_frame - 1,
                max(
                    0,
                    round(
                        (stable_time - first_frame_time) * fps
                    ),
                ),
            )
            last_frame = min(
                total_frames - 1,
                first_new_frame + hold_frames - 1,
            )
            if last_frame < first_guard_frame:
                guarded.append(keyframe)
                continue
            guards.append(
                TransitionGuard(
                    first_frame=first_guard_frame,
                    last_frame=last_frame,
                    replace_frame=replace_frame,
                )
            )
            switch_time = first_frame_time + (
                last_frame + 1
            ) / fps - 1e-6
            guarded.append(
                keyframe.model_copy(
                    update={"time": round(switch_time, 6)}
                )
            )
        return guarded, guards

    @staticmethod
    def _clamp_center(
        center: tuple[float, float],
        width: int,
        height: int,
        crop_width: int,
        crop_height: int,
    ) -> tuple[float, float]:
        half_width = crop_width / 2
        half_height = crop_height / 2
        return (
            min(max(center[0], half_width), width - half_width),
            min(max(center[1], half_height), height - half_height),
        )

    @staticmethod
    def _center_plan(
        source: Path,
        clip_range: ClipRange,
        content_x: int,
        content_y: int,
        content_width: int,
        content_height: int,
        crop_width: int,
        crop_height: int,
        *,
        mode: str,
    ) -> ReframePlan:
        return ReframePlan(
            source_file=source.resolve(),
            clip_start=clip_range.start,
            clip_end=clip_range.end,
            content_x=content_x,
            content_y=content_y,
            content_width=content_width,
            content_height=content_height,
            crop_width=crop_width,
            crop_height=crop_height,
            mode=mode,
            sampled_frames=0,
            frames_with_faces=0,
            keyframes=[
                ReframeKeyframe(
                    time=0,
                    center_x=content_width / 2,
                    center_y=content_height / 2,
                    face_count=0,
                )
            ],
        )
