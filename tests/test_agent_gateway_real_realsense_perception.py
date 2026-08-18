from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real import realsense_perception

from agent_gateway.profiles.real.perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
    RealPerceptionCaptureProfile,
    build_real_perception_capture_registration,
)
from agent_gateway.profiles.real.realsense_perception import (
    ExclusiveRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
    build_real_perception_source_binding,
    start_realsense_sources,
)
from teleop_runtime.collection.video_frame import VideoFrame as LegacyVideoFrame
from teleop_runtime.video_frame import VideoFrame


class _Source:
    def __init__(
        self,
        source_id: str,
        *,
        timestamp: float,
        enable_depth: bool,
        serial: str,
    ) -> None:
        self.config = SimpleNamespace(
            camera_key=source_id,
            width=4,
            height=3,
            depth_width=4,
            depth_height=3,
            align_depth_to_color=True,
            enable_depth=enable_depth,
            color_space="rgb",
            serial=serial,
        )
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[..., 0] = 255 if source_id == "front" else 0
        image[..., 1] = 255 if source_id != "front" else 0
        self.frame = VideoFrame(
            image=image,
            source_timestamp=timestamp,
            receive_timestamp=timestamp,
            available_timestamp=timestamp + 0.001,
            frame_index=3,
            metadata={
                "camera_key": source_id,
                "serial": serial,
                "source_generation": 1,
            },
            depth_image=(
                np.asarray([[0, 100, 200, 300]] * 3, dtype=np.uint16)
                if enable_depth
                else None
            ),
            depth_metadata=(
                {
                    "aligned_to": "color",
                    "depth_scale_m_per_unit": 0.001,
                }
                if enable_depth
                else {}
            ),
        )
        self.info = {
            "running": True,
            "lifecycle_state": "running",
            "serial_number": serial,
            "generation": 1,
            "worker_alive": True,
            "intrinsics": {
                "width": 4,
                "height": 3,
                "fx": 4.0,
                "fy": 3.0,
                "ppx": 2.0,
                "ppy": 1.5,
            },
        }
        self.starts = 0
        self.closes = 0

    def get_latest_frame(self):
        return self.frame

    def device_info(self):
        return dict(self.info)

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1


def _provider(*, front_at: float = 9.95, wrist_at: float = 9.97):
    sources = {
        "front": _Source(
            "front",
            timestamp=front_at,
            enable_depth=True,
            serial="front-serial",
        ),
        "left_wrist": _Source(
            "left_wrist",
            timestamp=wrist_at,
            enable_depth=False,
            serial="left-serial",
        ),
    }
    source_binding = RealPerceptionSourceBinding(
        camera_sources=sources,
        world_from_depth_camera_by_source={
            "front": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        },
        semantic_content_hash="source-binding:test",
    )
    revision = {"value": 7}
    provider = RealSenseAgentPerceptionProvider(
        source_binding=source_binding,
        current_station_revision=lambda: revision["value"],
        clock=lambda: 10.0,
    )
    profile = RealPerceptionCaptureProfile(
        rgb_sources=(
            ("front", "main", 4, 3),
            ("left_wrist", "wrist_left", 4, 3),
        ),
        depth_sources=(("front", "main", 4, 3),),
        source_binding_hash=source_binding.semantic_content_hash,
    )
    provider.bind_profile(profile)
    return provider, sources, revision, profile


def test_legacy_video_frame_export_is_object_identical() -> None:
    assert LegacyVideoFrame is VideoFrame


def test_real_perception_registration_is_t0_read_only_and_artifact_bearing() -> None:
    provider, _sources, _revision, _profile = _provider()
    registration = build_real_perception_capture_registration(
        rgb_source_catalog=provider.rgb_source_catalog(),
        depth_source_catalog=provider.depth_source_catalog(),
        runtime_session_id="real-session:test",
        control_domain_id="real-domain:test",
        target_binding_id="real-target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        source_binding_hash=provider.source_binding.semantic_content_hash,
    )

    for item in (registration.rgb, registration.depth):
        route = item.routes[0]
        assert route.target_kind == "real"
        assert route.supported_modes == ("real_active",)
        assert route.control_effect_class == "read_only"
        assert route.privilege_tier == "t0_read_only"
        assert route.may_move_robot is False
        assert route.requires_operator_confirmation is False
        assert route.skill_io_contract_hash == item.io_contract.semantic_content_hash
    assert registration.rgb.io_contract.output_artifact_slots[0].slot_id == (
        REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1
    )
    assert registration.depth.io_contract.output_artifact_slots[0].slot_id == (
        REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1
    )


def test_real_provider_encodes_rgb_and_metric_aligned_depth_without_control_io() -> None:
    provider, _sources, _revision, _profile = _provider()
    rgb = provider.capture_rgb(
        {
            "selected_roles": ["wrist_left", "main"],
            "width": 2,
            "height": 2,
            "max_total_bytes": 100_000,
        },
        expected_station_revision=7,
    )
    assert rgb.station_revision == 7
    assert [frame.role for frame in rgb.frames] == ["wrist_left", "main"]
    assert all(Image.open(io.BytesIO(frame.data)).size == (2, 2) for frame in rgb.frames)

    depth = provider.capture_depth(role="main", expected_station_revision=7)
    array = np.load(io.BytesIO(depth.data), allow_pickle=False)
    assert array.dtype == np.float32
    assert array.shape == (3, 4)
    np.testing.assert_allclose(array[0], (0.0, 0.1, 0.2, 0.3))
    assert depth.camera_pose_world_xyz_quat_xyzw == pytest.approx(
        (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
    )
    assert depth.intrinsics["width"] == 4
    assert depth.intrinsics["height"] == 3
    assert depth.intrinsics["cx"] == 2.0
    assert depth.intrinsics["fovy_deg"] > 0.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda provider, sources, revision: setattr(
                sources["front"], "frame", None
            ),
            "unavailable",
        ),
        (
            lambda provider, sources, revision: sources["front"].info.update(
                {"serial_number": "wrong"}
            ),
            "serial_mismatch",
        ),
        (
            lambda provider, sources, revision: sources["front"].info.update(
                {"running": False}
            ),
            "not_running",
        ),
        (
            lambda provider, sources, revision: sources["front"].frame.metadata.update(
                {"source_generation": 0}
            ),
            "generation_mismatch",
        ),
        (
            lambda provider, sources, revision: sources["front"].frame.metadata.update(
                {"camera_key": "left_wrist"}
            ),
            "identity_mismatch",
        ),
        (lambda provider, sources, revision: revision.update(value=8), "revision_mismatch"),
    ),
)
def test_real_provider_fails_closed_on_invalid_live_binding(mutation, match) -> None:
    provider, sources, revision, _profile = _provider()
    mutation(provider, sources, revision)
    with pytest.raises(RuntimeError, match=match):
        provider.capture_rgb(
            {
                "selected_roles": ["main"],
                "width": 2,
                "height": 2,
                "max_total_bytes": 100_000,
            },
            expected_station_revision=7,
        )


def test_real_provider_rejects_stale_skewed_and_schema_drifted_frames() -> None:
    stale, _sources, _revision, _profile = _provider(front_at=9.0)
    with pytest.raises(RuntimeError, match="stale"):
        stale.capture_depth(role="main", expected_station_revision=7)

    skewed, _sources, _revision, _profile = _provider(
        front_at=9.86,
        wrist_at=9.98,
    )
    with pytest.raises(RuntimeError, match="inter_camera_skew"):
        skewed.capture_rgb(
            {
                "selected_roles": ["main", "wrist_left"],
                "width": 2,
                "height": 2,
                "max_total_bytes": 100_000,
            },
            expected_station_revision=7,
        )

    provider, sources, _revision, _profile = _provider()
    sources["front"].frame.depth_metadata["aligned_to"] = "none"
    with pytest.raises(RuntimeError, match="alignment_changed"):
        provider.capture_depth(role="main", expected_station_revision=7)


def test_checked_in_binding_exposes_all_rgb_but_only_trusted_static_depth() -> None:
    binding = build_real_perception_source_binding(
        camera_config_path="teleop_runtime/configs/devices/agent_gateway_realsense.json",
        camera_registry_path="robot_motion/configs/rm65_camera_registry.json",
        scene_world_from_arm_base_by_side={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )
    provider = RealSenseAgentPerceptionProvider(
        source_binding=binding,
        current_station_revision=lambda: 0,
    )
    assert {item["role"] for item in provider.rgb_source_catalog()} == {
        "main",
        "wrist_left",
        "wrist_right",
    }
    assert tuple(item["role"] for item in provider.depth_source_catalog()) == (
        "main",
    )
    assert set(binding.world_from_depth_camera_by_source) == {"front"}
    assert all(
        source.device_info()["running"] is False
        for source in binding.camera_sources.values()
    )


def test_binding_rejects_untrusted_wrist_depth_before_hardware_start(tmp_path) -> None:
    payload = json.loads(
        Path("teleop_runtime/configs/devices/agent_gateway_realsense.json").read_text(
            encoding="utf-8"
        )
    )
    payload["left_wrist"]["enable_depth"] = True
    config_path = tmp_path / "cameras.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one trusted static"):
        build_real_perception_source_binding(
            camera_config_path=config_path,
            camera_registry_path="robot_motion/configs/rm65_camera_registry.json",
            scene_world_from_arm_base_by_side={
                "left": np.eye(4),
                "right": np.eye(4),
            },
        )


def test_resource_transaction_admits_cameras_before_lower_and_closes_reverse() -> None:
    events = []
    sources = {
        "front": _Source("front", timestamp=1.0, enable_depth=True, serial="a"),
        "wrist": _Source("left_wrist", timestamp=1.0, enable_depth=False, serial="b"),
    }

    class _Lower:
        def start(self):
            events.append("lower.start")

        def close(self):
            events.append("lower.close")

    def start_cameras(values):
        events.append("cameras.start")
        for source in values.values():
            source.start()

    resources = ExclusiveRealPerceptionResources(
        lower=_Lower(),
        camera_sources=sources,
        start_cameras=start_cameras,
        admit_cameras=lambda _values: events.append("cameras.admit"),
    )
    resources.start()
    resources.close()
    assert events == [
        "cameras.start",
        "cameras.admit",
        "lower.start",
        "lower.close",
    ]
    assert [source.closes for source in sources.values()] == [1, 1]


def test_resource_transaction_rolls_back_cameras_without_starting_lower() -> None:
    source = _Source("front", timestamp=1.0, enable_depth=True, serial="a")
    lower = SimpleNamespace(starts=0, closes=0)
    artifact_store = SimpleNamespace(closes=0)
    lower.start = lambda: setattr(lower, "starts", lower.starts + 1)
    lower.close = lambda: setattr(lower, "closes", lower.closes + 1)
    artifact_store.close = lambda: setattr(
        artifact_store, "closes", artifact_store.closes + 1
    )
    resources = ExclusiveRealPerceptionResources(
        lower=lower,
        camera_sources={"front": source},
        artifact_store=artifact_store,
        start_cameras=lambda values: next(iter(values.values())).start(),
        admit_cameras=lambda _values: (_ for _ in ()).throw(RuntimeError("bad frames")),
    )

    with pytest.raises(RuntimeError, match="bad frames"):
        resources.start()
    assert lower.starts == 0
    assert lower.closes == 0
    assert source.closes == 1
    assert artifact_store.closes == 1
    with pytest.raises(RuntimeError, match="cannot restart"):
        resources.start()


def test_resource_transaction_closes_partial_lower_start() -> None:
    source = _Source("front", timestamp=1.0, enable_depth=True, serial="a")
    lower = SimpleNamespace(closes=0)
    lower.start = lambda: (_ for _ in ()).throw(RuntimeError("partial lower"))
    lower.close = lambda: setattr(lower, "closes", lower.closes + 1)
    resources = ExclusiveRealPerceptionResources(
        lower=lower,
        camera_sources={"front": source},
        start_cameras=lambda values: next(iter(values.values())).start(),
        admit_cameras=lambda _values: None,
    )

    with pytest.raises(RuntimeError, match="partial lower"):
        resources.start()
    assert lower.closes == 1
    assert source.closes == 1


def test_gateway_camera_startup_delegates_to_formal_teleop_lifecycle(
    monkeypatch,
) -> None:
    sources = {"front": object()}
    calls = []
    monkeypatch.setattr(
        realsense_perception,
        "_formal_start_camera_sources",
        lambda values: calls.append(values),
    )

    start_realsense_sources(sources)

    assert calls == [sources]
