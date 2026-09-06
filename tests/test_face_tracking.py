from pathlib import Path

import numpy as np

from app.config import EncodingSection, ReframeSection, SubtitleSection
from app.models import (
    ClipRange,
    ReframeKeyframe,
    ReframePlan,
    SceneSegment,
)
from app.services.export_service import ExportService
from app.services.face_tracking_service import FaceTrackingService


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0

    def set(self, _property: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame


def test_face_selector_prefers_largest_face_then_track_continuity():
    faces = [
        (50.0, 50.0, 100.0, 100.0, 0.95),
        (500.0, 40.0, 180.0, 180.0, 0.9),
    ]

    largest = FaceTrackingService._select_face(
        faces,
        None,
        800,
        450,
    )
    tracked = FaceTrackingService._select_face(
        faces,
        (100, 100),
        800,
        450,
    )

    assert largest == faces[1]
    assert tracked == faces[0]


def test_smoothing_ignores_small_jitter_and_limits_large_jump():
    service = FaceTrackingService(
        ReframeSection(
            smoothing=0.4,
            deadzone_ratio=0.025,
            max_pan_speed_ratio=0.1,
        ),
        EncodingSection(),
    )

    assert service._smooth_center(
        (500, 300),
        (510, 305),
        1000,
        600,
    ) == (500, 300)
    assert service._smooth_center(
        (100, 300),
        (800, 300),
        1000,
        600,
        elapsed=0.5,
    ) == (150, 300)


def test_shot_lock_uses_one_stable_crop_and_exact_scene_cut():
    keyframes = [
        ReframeKeyframe(
            time=0,
            center_x=200,
            center_y=250,
            face_count=1,
            tracked_face_score=0.9,
            scene_index=1,
        ),
        ReframeKeyframe(
            time=0.75,
            center_x=230,
            center_y=250,
            face_count=1,
            tracked_face_score=0.92,
            scene_index=1,
        ),
        ReframeKeyframe(
            time=1.5,
            center_x=800,
            center_y=250,
            face_count=1,
            tracked_face_score=0.75,
            scene_index=1,
        ),
        ReframeKeyframe(
            time=3,
            center_x=700,
            center_y=250,
            face_count=1,
            tracked_face_score=0.9,
            scene_index=2,
        ),
        ReframeKeyframe(
            time=3.75,
            center_x=720,
            center_y=250,
            face_count=1,
            tracked_face_score=0.91,
            scene_index=2,
        ),
    ]
    scenes = [
        SceneSegment(index=1, start=8, end=12.5),
        SceneSegment(index=2, start=12.5, end=20),
    ]

    locked = FaceTrackingService._lock_keyframes_to_scenes(
        keyframes,
        scenes,
        ClipRange(start=10, end=16),
        1000,
        500,
        280,
        500,
    )

    assert len(locked) == 2
    assert [keyframe.time for keyframe in locked] == [0, 2.5]
    assert [keyframe.center_x for keyframe in locked] == [230, 710]


def test_hard_cut_refinement_finds_first_frame_of_new_shot():
    frames = [
        np.full((100, 160, 3), 20, dtype=np.uint8)
        for _ in range(24)
    ] + [
        np.full((100, 160, 3), 230, dtype=np.uint8)
        for _ in range(24)
    ]
    service = FaceTrackingService(
        ReframeSection(cut_refine_window=0.35),
        EncodingSection(),
    )

    refined = service._find_hard_cut(
        _FakeCapture(frames),
        approximate=0.9,
        content_x=0,
        content_y=0,
        content_width=160,
        content_height=100,
        fps=24,
    )
    refined_with_stream_offset = service._find_hard_cut(
        _FakeCapture(frames),
        approximate=0.98,
        content_x=0,
        content_y=0,
        content_width=160,
        content_height=100,
        fps=24,
        video_start_time=0.083,
    )

    assert refined == 1.0
    assert refined_with_stream_offset == 1.083


def test_shot_lock_keeps_latest_scene_at_duplicate_start_time():
    keyframes = [
        ReframeKeyframe(
            time=0,
            center_x=200,
            center_y=250,
            face_count=1,
            tracked_face_score=0.9,
            scene_index=1,
        ),
        ReframeKeyframe(
            time=0.1,
            center_x=700,
            center_y=250,
            face_count=1,
            tracked_face_score=0.9,
            scene_index=2,
        ),
    ]
    scenes = [
        SceneSegment(index=1, start=9, end=10),
        SceneSegment(index=2, start=10, end=12),
    ]

    locked = FaceTrackingService._lock_keyframes_to_scenes(
        keyframes,
        scenes,
        ClipRange(start=10, end=12),
        1000,
        500,
        280,
        500,
    )

    assert len(locked) == 1
    assert locked[0].time == 0
    assert locked[0].scene_index == 2


def test_transition_guard_holds_last_clean_frame_without_time_shift():
    service = FaceTrackingService(
        ReframeSection(transition_hold_frames=2),
        EncodingSection(),
    )
    keyframes = [
        ReframeKeyframe(
            time=0,
            center_x=200,
            center_y=250,
            face_count=1,
            scene_index=1,
        ),
        ReframeKeyframe(
            time=3.249,
            center_x=700,
            center_y=250,
            face_count=1,
            scene_index=2,
        ),
    ]

    guarded, replacements = service._apply_transition_guards(
        keyframes,
        fps=24,
        video_start_time=0.083,
        clip_range=ClipRange(start=3898.167, end=3985.458),
    )

    assert len(replacements) == 1
    assert replacements[0].first_frame == 75
    assert replacements[0].last_frame == 78
    assert replacements[0].replace_frame == 38
    assert guarded[1].time == 3.332666


def test_reframe_expression_interpolates_keyframes():
    plan = ReframePlan(
        source_file=Path("movie.mp4"),
        clip_start=0,
        clip_end=2,
        content_x=0,
        content_y=0,
        content_width=1000,
        content_height=500,
        crop_width=280,
        crop_height=500,
        mode="smart_face",
        sampled_frames=2,
        frames_with_faces=2,
        keyframes=[
            ReframeKeyframe(
                time=0,
                center_x=200,
                center_y=250,
                face_count=1,
            ),
            ReframeKeyframe(
                time=1,
                center_x=800,
                center_y=250,
                face_count=1,
            ),
        ],
    )
    service = ExportService(
        object(),
        EncodingSection(),
        SubtitleSection(),
    )

    x_expression = service._position_expression(plan, axis="x")
    y_expression = service._position_expression(plan, axis="y")

    assert x_expression.startswith("if(lt(t,1.000)")
    assert "600.00" in x_expression
    assert y_expression == "0.00"
