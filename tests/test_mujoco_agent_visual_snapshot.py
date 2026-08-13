from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import sim.mujoco.rgb_snapshot as rgb_snapshot_module
from control_station.sim_hil_profile.agent.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from control_station.sim_hil_profile.agent.visual_observation import (
    MUJOCO_AGENT_VISUAL_FRAME_SCHEMA_V1,
    MUJOCO_AGENT_VISUAL_SNAPSHOT_SCHEMA_V1,
    MujocoAgentVisualSnapshotProvider,
)
from sim.mujoco.rgb_snapshot import (
    EncodedMujocoRgbFrame,
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
    frames = source.capture_jpeg_frames(
        source_ids=("front", "right_wrist"),
        width=16,
        height=12,
    )
    repeated = source.capture_jpeg_frames(
        source_ids=("front",),
        width=16,
        height=12,
    )

    assert len(renderers) == 1
    assert tuple(frames) == ("front", "right_wrist")
    assert frames["front"].media_type == "image/jpeg"
    assert frames["front"].data.startswith(b"\xff\xd8")
    assert frames["front"].data.endswith(b"\xff\xd9")
    assert repeated["front"].width == 16
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
    closed_states: list[bool] = []
    application._visual_snapshot_source = SimpleNamespace(
        close=lambda: closed_states.append(application._closed)
    )

    application.close_after_hold()

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
        source.capture_jpeg_frames(
            source_ids=("missing",),
            width=8,
            height=8,
        )
    with pytest.raises(RuntimeError, match="uint8 HWC RGB"):
        source.capture_jpeg_frames(
            source_ids=("front",),
            width=8,
            height=8,
        )


def test_station_provider_projects_lower_encoded_frames_to_wire_contract() -> None:
    jpeg = b"fixture-jpeg"
    calls: list[dict[str, object]] = []
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )

    def capture_visual_jpeg_frames(**kwargs):
        calls.append(dict(kwargs))
        return {
            "front": EncodedMujocoRgbFrame(
                source_id="front",
                width=32,
                height=24,
                media_type="image/jpeg",
                data=jpeg,
            )
        }

    application.capture_visual_jpeg_frames = capture_visual_jpeg_frames
    provider = MujocoAgentVisualSnapshotProvider(
        application=application,
        current_station_revision=lambda: 7,
    )

    result = provider.capture(
        {
            "client_id": "client:fixture",
            "request_id": "request:fixture",
            "selected_roles": ["main"],
            "max_images": 1,
            "width": 32,
            "height": 24,
            "max_total_bytes": 1024,
            "expected_station_revision": 7,
        }
    )

    assert calls == [{"source_ids": ("front",), "width": 32, "height": 24}]
    assert result["schema"] == MUJOCO_AGENT_VISUAL_SNAPSHOT_SCHEMA_V1
    assert result["total_bytes"] == len(jpeg)
    frame = result["frames"][0]
    assert frame["schema"] == MUJOCO_AGENT_VISUAL_FRAME_SCHEMA_V1
    assert frame["source_id"] == "front"
    assert frame["role"] == "main"
    assert frame["media_content_hash"] == hashlib.sha256(jpeg).hexdigest()


def test_station_provider_rejects_mismatched_lower_frame_contract() -> None:
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )
    application.capture_visual_jpeg_frames = lambda **_kwargs: {
        "front": SimpleNamespace(
            source_id="front",
            width=31,
            height=24,
            media_type="image/jpeg",
            data=b"jpeg",
        )
    }
    provider = MujocoAgentVisualSnapshotProvider(
        application=application,
        current_station_revision=lambda: 4,
    )

    with pytest.raises(RuntimeError, match="frame_contract_mismatch"):
        provider.capture(
            {
                "client_id": "client:fixture",
                "request_id": "request:fixture",
                "selected_roles": ["main"],
                "max_images": 1,
                "width": 32,
                "height": 24,
                "max_total_bytes": 1024,
                "expected_station_revision": 4,
            }
        )
