from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import sim.mujoco.rgb_snapshot as rgb_snapshot_module
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.rgb_deployment import (
    MUJOCO_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
    MujocoRgbCaptureProfile,
    build_mujoco_rgb_capture_registration,
)
from agent_gateway.profiles.mujoco.visual_observation import (
    MujocoAgentRgbCaptureProvider,
)
from agent_gateway.skills.perception.definition import CAPTURE_RGB_SKILL_ID_V1
from sim.mujoco.rgb_snapshot import (
    EncodedMujocoRgbFrame,
    EncodedMujocoRgbSnapshot,
    MujocoRgbSnapshotSource,
)

pytestmark = [pytest.mark.unit]


class _FakeScene:
    camera_names = ("front", "right_wrist")

    def camera_metadata(self):
        return {
            "front": {"width": 640, "height": 480},
            "right_wrist": {"width": 320, "height": 240},
        }


class _FakeRenderer:
    def __init__(self, scene, *, width: int, height: int) -> None:
        self.scene = scene
        self.width = width
        self.height = height
        self.closed = False
        self.render_calls: list[tuple[str, ...]] = []

    def render_all(self, source_ids):
        selected = tuple(source_ids)
        self.render_calls.append(selected)
        return {
            source_id: np.full(
                (self.height, self.width, 3),
                fill_value=index * 40,
                dtype=np.uint8,
            )
            for index, source_id in enumerate(selected, start=1)
        }

    def close(self) -> None:
        self.closed = True


def _registration():
    return build_mujoco_rgb_capture_registration(
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 640,
                "native_height": 480,
            },
            {
                "source_id": "right_wrist",
                "role": "wrist_right",
                "native_width": 320,
                "native_height": 240,
            },
        ),
        runtime_session_id="session:mujoco-rgb",
        control_domain_id="domain:mujoco",
        target_binding_id="target:mujoco",
        station_manifest_hash="station:hash",
        execution_manifest_hash="execution:hash",
    )


def test_rgb_registration_is_strictly_read_only_and_artifact_bearing() -> None:
    bound = _registration()
    template = bound.registration
    route = template.routes[0]
    slot = template.io_contract.output_artifact_slots[0]

    assert template.definition.skill_id == CAPTURE_RGB_SKILL_ID_V1
    assert template.definition.skill_kind == "inspection"
    assert template.definition.controlled_component_ids == ()
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert not route.may_move_robot
    assert not route.may_open_publication
    assert route.cancellation_policy == "cancel_only"
    assert not route.requires_task_evaluation
    assert slot.slot_id == MUJOCO_CAPTURE_RGB_OUTPUT_SLOT_ID_V1
    assert (slot.min_count, slot.max_count) == (1, 2)


def test_rgb_registration_validates_roles_dimensions_hashes_and_bounds() -> None:
    bound = _registration()
    parameters = {
        "selected_roles": ["main", "wrist_right"],
        "width": 32,
        "height": 24,
        "max_total_bytes": 1024,
    }
    result = {
        "snapshot_id": "rgb-snapshot:test",
        "station_revision": 4,
        "captured_at": 1.25,
        "frames": [
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "role": role,
                "media_type": "image/jpeg",
                "width": 32,
                "height": 24,
                "media_content_hash": character * 64,
            }
            for ordinal, (source_id, role, character) in enumerate(
                (("front", "main", "a"), ("right_wrist", "wrist_right", "b"))
            )
        ],
        "total_bytes": 128,
    }
    validator = bound.payload_validator
    validator.validate_parameters(parameters)
    validator.validate_execution_result(result)
    validator.validate_execution_result_for_parameters(parameters, result)

    with pytest.raises(ValueError, match="registered roles"):
        validator.validate_parameters({**parameters, "selected_roles": ["missing"]})
    with pytest.raises(ValueError, match="registered roles"):
        validator.validate_parameters({**parameters, "selected_roles": [{}]})
    with pytest.raises(ValueError, match="ordinal"):
        validator.validate_execution_result(
            {
                **result,
                "frames": [{**result["frames"][0], "ordinal": 1}],
            }
        )
    with pytest.raises(ValueError, match="dimensions differ"):
        validator.validate_execution_result_for_parameters(
            parameters,
            {
                **result,
                "frames": [{**frame, "width": 31} for frame in result["frames"]],
            },
        )


def test_rgb_profile_rejects_duplicate_roles_and_unbounded_frame_count() -> None:
    with pytest.raises(ValueError, match="unique"):
        MujocoRgbCaptureProfile(
            sources=(("front", "main", 2, 2), ("other", "main", 2, 2)),
        )
    with pytest.raises(ValueError, match="eight"):
        MujocoRgbCaptureProfile(
            sources=(("front", "main", 2, 2),),
            max_frames=9,
        )


def test_lower_rgb_source_owns_catalog_renderer_cache_encoding_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _FakeScene()
    renderers: list[_FakeRenderer] = []
    capture_depth = 0

    def capture_scene(callback):
        nonlocal capture_depth
        capture_depth += 1
        try:
            return callback(scene)
        finally:
            capture_depth -= 1

    def renderer_factory(scene_arg, *, width: int, height: int):
        assert capture_depth == 1
        renderer = _FakeRenderer(scene_arg, width=width, height=height)
        renderers.append(renderer)
        return renderer

    original_encode = rgb_snapshot_module._encode_jpeg

    def encode_outside_capture(image: object) -> bytes:
        assert capture_depth == 0
        return original_encode(image)

    monkeypatch.setattr(rgb_snapshot_module, "_encode_jpeg", encode_outside_capture)

    source = MujocoRgbSnapshotSource(
        capture_scene=capture_scene,
        renderer_factory=renderer_factory,
    )

    assert source.source_catalog() == (
        {"source_id": "front", "native_width": 640, "native_height": 480},
        {
            "source_id": "right_wrist",
            "native_width": 320,
            "native_height": 240,
        },
    )
    frames = source.capture_jpeg_snapshot(
        source_ids=("front", "right_wrist"),
        width=16,
        height=12,
    ).frames
    repeated = source.capture_jpeg_snapshot(
        source_ids=("front",),
        width=16,
        height=12,
    ).frames

    assert len(renderers) == 1
    assert tuple(frames) == ("front", "right_wrist")
    assert frames["front"].media_type == "image/jpeg"
    assert frames["front"].data.startswith(b"\xff\xd8")
    assert frames["front"].data.endswith(b"\xff\xd9")
    assert repeated["front"].width == 16
    snapshot = source.capture_jpeg_snapshot(
        source_ids=("front",),
        width=8,
        height=8,
    )
    assert snapshot.captured_at > 0.0
    assert tuple(snapshot.frames) == ("front",)
    source.close()
    assert renderers[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        source.source_catalog()


def test_motion_application_marks_closed_before_closing_visual_source() -> None:
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application._lock = threading.RLock()
    application._started = False
    application._held = False
    application._closed = False
    application._active_operation_id = "operation:fixture"
    runtime_closed_states: list[bool] = []
    application.runtime = SimpleNamespace(
        close=lambda: runtime_closed_states.append(application._closed)
    )
    closed_states: list[bool] = []
    application._visual_snapshot_source = SimpleNamespace(
        close=lambda: closed_states.append(application._closed)
    )

    application.close_after_hold()

    assert runtime_closed_states == [False]
    assert closed_states == [True]
    assert application._active_operation_id is None


def test_lower_rgb_source_rejects_unknown_or_non_rgb_frames() -> None:
    scene = _FakeScene()

    class InvalidRenderer(_FakeRenderer):
        def render_all(self, source_ids):
            return {
                source_id: np.zeros((8, 8), dtype=np.uint8)
                for source_id in source_ids
            }

    source = MujocoRgbSnapshotSource(
        capture_scene=lambda callback: callback(scene),
        renderer_factory=InvalidRenderer,
    )
    with pytest.raises(ValueError, match="unknown visual sources"):
        source.capture_jpeg_snapshot(
            source_ids=("missing",),
            width=8,
            height=8,
        )
    with pytest.raises(RuntimeError, match="uint8 HWC RGB"):
        source.capture_jpeg_snapshot(
            source_ids=("front",),
            width=8,
            height=8,
        )

    nonfinite_clock = MujocoRgbSnapshotSource(
        capture_scene=lambda callback: callback(scene),
        clock=lambda: float("nan"),
        renderer_factory=_FakeRenderer,
    )
    with pytest.raises(RuntimeError, match="clock is non-finite"):
        nonfinite_clock.capture_jpeg_snapshot(
            source_ids=("front",),
            width=8,
            height=8,
        )


def test_rgb_provider_resolves_roles_and_preserves_raw_artifact_bytes() -> None:
    jpeg = b"fixture-jpeg"
    calls: list[dict[str, object]] = []
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )

    def capture_visual_jpeg_snapshot(**kwargs):
        calls.append(dict(kwargs))
        return EncodedMujocoRgbSnapshot(
            captured_at=1.25,
            frames={
                "front": EncodedMujocoRgbFrame(
                    source_id="front",
                    width=32,
                    height=24,
                    media_type="image/jpeg",
                    data=jpeg,
                )
            },
        )

    application.capture_visual_jpeg_snapshot = capture_visual_jpeg_snapshot
    provider = MujocoAgentRgbCaptureProvider(
        application=application,
        current_station_revision=lambda: 7,
    )

    result = provider.capture(
        {
            "selected_roles": ["main"],
            "width": 32,
            "height": 24,
            "max_total_bytes": 1024,
        },
        expected_station_revision=7,
    )

    assert calls == [{"source_ids": ("front",), "width": 32, "height": 24}]
    assert result.captured_at == 1.25
    assert result.station_revision == 7
    source_id, role, frame = result.frames[0]
    assert (source_id, role) == ("front", "main")
    assert frame.data == jpeg


def test_rgb_provider_rejects_mismatched_lower_frame_contract() -> None:
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )
    application.capture_visual_jpeg_snapshot = lambda **_kwargs: SimpleNamespace(
        captured_at=1.0,
        frames={
            "front": SimpleNamespace(
                source_id="front",
                width=31,
                height=24,
                media_type="image/jpeg",
                data=b"jpeg",
            )
        },
    )
    provider = MujocoAgentRgbCaptureProvider(
        application=application,
        current_station_revision=lambda: 4,
    )

    with pytest.raises(RuntimeError, match="frame_contract_mismatch"):
        provider.capture(
            {
                "selected_roles": ["main"],
                "width": 32,
                "height": 24,
                "max_total_bytes": 1024,
            },
            expected_station_revision=4,
        )

    with pytest.raises(ValueError, match="exact schema fields"):
        provider.capture(
            {
                "selected_roles": ["main"],
                "width": 32,
                "height": 24,
                "max_total_bytes": 1024,
                "unexpected": True,
            },
            expected_station_revision=4,
        )


def test_rgb_provider_rejects_changed_revision_and_total_byte_overflow() -> None:
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )
    application.capture_visual_jpeg_snapshot = lambda **_kwargs: (
        EncodedMujocoRgbSnapshot(
            captured_at=1.0,
            frames={
                "front": EncodedMujocoRgbFrame(
                    source_id="front",
                    width=32,
                    height=24,
                    media_type="image/jpeg",
                    data=b"jpeg",
                )
            },
        )
    )
    revisions = iter((4, 5))
    provider = MujocoAgentRgbCaptureProvider(
        application=application,
        current_station_revision=lambda: next(revisions),
    )
    parameters = {
        "selected_roles": ["main"],
        "width": 32,
        "height": 24,
        "max_total_bytes": 1024,
    }
    with pytest.raises(RuntimeError, match="revision_changed"):
        provider.capture(parameters, expected_station_revision=4)

    provider = MujocoAgentRgbCaptureProvider(
        application=application,
        current_station_revision=lambda: 4,
    )
    with pytest.raises(RuntimeError, match="artifact_limit"):
        provider.capture(
            {**parameters, "max_total_bytes": 3},
            expected_station_revision=4,
        )
