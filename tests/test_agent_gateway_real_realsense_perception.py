from __future__ import annotations

import io
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.back_projection import (
    LiveBackProjectionProvider,
    build_back_projection_registration,
)
from agent_gateway.profiles.real import realsense_perception
from agent_gateway.profiles.real.back_projection import (
    RealSenseBackProjectionCapturePort,
)
from agent_gateway.profiles.real.perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
    RealPerceptionCaptureProfile,
    build_real_perception_capture_registration,
    build_real_perception_capture_registration_v2,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.realsense_perception import (
    CameraOnlyRealPerceptionResources,
    ExclusiveRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealRgbCameraGeometryCut,
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
        self.latest_frame_reads = 0

    def get_latest_frame(self):
        self.latest_frame_reads += 1
        return self.frame

    def device_info(self):
        return dict(self.info)

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1


class _RecordingRgbCameraGeometry:
    semantic_content_hash = "geometry:test"

    def __init__(self, *, failure: str | None = None) -> None:
        self.pose_cut_calls: list[tuple[str, float, float]] = []
        self.stationary_cut_calls: list[tuple[str, float, float]] = []
        self.failure = failure

    @staticmethod
    def source_catalog():
        return (
            {
                "source_id": "front",
                "role": "main",
                "native_width": 4,
                "native_height": 3,
            },
            {
                "source_id": "left_wrist",
                "role": "wrist_left",
                "native_width": 4,
                "native_height": 3,
            },
        )

    def pose_cut(self, *, source_id: str, frame_acquired_at: float, now: float):
        self.pose_cut_calls.append((source_id, frame_acquired_at, now))
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if now < 10.2:
            raise RuntimeError(f"real_rgb_camera_pose_from_future:{source_id}")
        return RealRgbCameraGeometryCut(
            source_id=source_id,
            source_from_camera=tuple(float(value) for value in np.eye(4).reshape(-1)),
            measured_at=frame_acquired_at,
            available_at=10.2,
            geometry_binding_hash=self.semantic_content_hash,
            geometry_cut_hash="geometry-cut:test",
            evidence_refs=("geometry-cut:test",),
        )

    def stationary_geometry_cut(self, *, source_id, frame_acquired_at, now):
        self.stationary_cut_calls.append((source_id, frame_acquired_at, now))
        cut = self.pose_cut(source_id=source_id, frame_acquired_at=frame_acquired_at, now=now)
        return cut, {side: np.zeros(6) for side in ("left", "right")}, cut.measured_at, cut.available_at, cut.evidence_refs


def _provider(
    *,
    front_at: float = 9.95,
    wrist_at: float = 9.97,
    wrist_depth: bool = False,
    rgb_camera_geometry=None,
    clock=None,
):
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
            enable_depth=wrist_depth,
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
        rgb_camera_geometry=rgb_camera_geometry,
        clock=(lambda: 10.0) if clock is None else clock,
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


def test_rgb_geometry_observes_dynamic_pose_after_frame_processing() -> None:
    geometry = _RecordingRgbCameraGeometry()
    observed_times = iter((10.0, 10.2))
    provider, _sources, _revision, _profile = _provider(
        rgb_camera_geometry=geometry,
        clock=lambda: next(observed_times),
    )

    capture = provider.capture_rgb_geometry(
        role="wrist_left",
        expected_station_revision=7,
    )

    assert geometry.pose_cut_calls == [("left_wrist", 9.97, 10.2)]
    assert capture.pose_available_at == 10.2


def test_rgb_observes_frame_after_obtaining_concurrent_source_snapshot() -> None:
    observed_source: dict[str, _Source] = {}

    def clock() -> float:
        source = observed_source.get("source")
        return 10.2 if source is not None and source.latest_frame_reads else 10.0

    provider, sources, _revision, _profile = _provider(
        wrist_at=10.1,
        clock=clock,
    )
    observed_source["source"] = sources["left_wrist"]

    capture = provider.capture_rgb(
        {
            "selected_roles": ["wrist_left"],
            "width": 2,
            "height": 2,
            "max_total_bytes": 100_000,
        },
        expected_station_revision=7,
    )

    assert sources["left_wrist"].latest_frame_reads == 1
    assert capture.frames[0].acquired_at == 10.1


def test_rgb_rejects_frame_still_from_future_after_observation() -> None:
    provider, _sources, _revision, _profile = _provider(
        wrist_at=10.1,
        clock=lambda: 10.0,
    )

    with pytest.raises(RuntimeError, match="frame_from_future:left_wrist"):
        provider.capture_rgb(
            {
                "selected_roles": ["wrist_left"],
                "width": 2,
                "height": 2,
                "max_total_bytes": 100_000,
            },
            expected_station_revision=7,
        )


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
    assert registration.frame_binding is None
    assert registration.depth.routes[0].route_version == "1"


def test_real_depth_v2_binds_and_transforms_nontrivial_station_frame() -> None:
    provider, _sources, _revision, _profile = _provider()
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile="depth-v2-test",
        scene_mount_hash="scene-mounts:depth-v2-test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )
    registration = build_real_perception_capture_registration_v2(
        rgb_source_catalog=provider.rgb_source_catalog(),
        depth_source_catalog=provider.depth_source_catalog(),
        aligned_rgbd_source_catalog=provider.aligned_rgbd_source_catalog(),
        runtime_session_id="real-session:v2-test",
        control_domain_id="real-domain:v2-test",
        target_binding_id="real-target:v2-test",
        station_manifest_hash="station:v2-test",
        execution_manifest_hash="execution:v2-test",
        source_binding_hash=provider.source_binding.semantic_content_hash,
        frame_binding=frame_binding,
    )

    assert registration.frame_binding is frame_binding
    assert registration.rgb.definition.skill_version == "2"
    assert registration.rgb.routes[0].route_version == "2"
    assert registration.rgb_binding.deployment_binding_version == "2"
    assert registration.depth.routes[0].route_version == "2"
    assert registration.depth_binding.reference_frame_id == "canonical_station"
    assert registration.depth_binding.frame_graph_hash == (
        frame_binding.frame_graph.semantic_content_hash
    )
    camera_pose_reference = frame_binding.camera_pose_reference_xyz_quat_xyzw(
        (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
    )
    assert camera_pose_reference == pytest.approx(
        (0.3, -0.4, 0.4, 0.0, 0.0, 1.0, 0.0),
        abs=1e-6,
    )


def test_real_provider_encodes_rgb_and_metric_aligned_depth_without_control_io() -> (
    None
):
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
    assert [frame.acquired_at for frame in rgb.frames] == [9.97, 9.95]
    assert rgb.capture_window_started_at == 9.95
    assert rgb.capture_window_ended_at == 9.97
    assert all(
        Image.open(io.BytesIO(frame.data)).size == (2, 2) for frame in rgb.frames
    )

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


def test_real_provider_captures_one_correlated_immutable_rgbd_frame() -> None:
    provider, sources, _revision, _profile = _provider()

    capture = provider.capture_aligned_rgbd(
        role="main",
        expected_station_revision=7,
    )

    assert sources["front"].latest_frame_reads == 1
    assert capture.source_id == "front"
    assert capture.role == "main"
    assert capture.station_revision == 7
    assert capture.source_generation == 1
    assert capture.frame_index == 3
    assert capture.capture_group_id.startswith("real-aligned-rgbd:")
    assert capture.rgb.shape == (3, 4, 3)
    assert capture.rgb.dtype == np.uint8
    assert capture.rgb.flags.writeable is False
    assert capture.depth_m.shape == (3, 4)
    assert capture.depth_m.dtype == np.float32
    assert capture.depth_m.flags.writeable is False
    np.testing.assert_allclose(capture.depth_m[0], (0.0, 0.1, 0.2, 0.3))
    assert len(capture.rgb_content_hash) == 64
    assert len(capture.depth_content_hash) == 64


def test_real_back_projection_uses_the_bound_station_camera_pose() -> None:
    provider, _sources, _revision, _profile = _provider()
    frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile="back-projection-test",
        scene_mount_hash="scene-mounts:back-projection-test",
        scene_world_from_canonical_station=np.eye(4),
    )
    registration = build_back_projection_registration(
        target_kind="real",
        current_mode="real_active",
        source_catalog=provider.aligned_rgbd_source_catalog(),
        reference_frame_id=frame_binding.reference_frame_id,
        reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
        frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        runtime_session_id="real-session:back-projection",
        control_domain_id="real-domain:back-projection",
        target_binding_id="real-target:back-projection",
        station_manifest_hash="station:back-projection",
        execution_manifest_hash="execution:back-projection",
    )
    back_projection = LiveBackProjectionProvider(
        capture_port=RealSenseBackProjectionCapturePort(
            perception_provider=provider,
            frame_binding=frame_binding,
        ),
        registration=registration,
    )

    result = back_projection.project_request(
        SimpleNamespace(
            expected_station_revision=7,
            parameters={
                "role": "main",
                "target_pixel_uv": [2, 1],
                "pixel_region_xyxy": None,
                "world_z_min_m": None,
                "world_z_max_m": None,
            },
        )
    )

    assert result["world_xyz_m"] == pytest.approx(
        [0.1, 1.0 / 6.0, 0.5], abs=1e-6
    )
    assert result["capture_group_id"].startswith("real-aligned-rgbd:")


def test_aligned_wrist_rgbd_uses_dynamic_geometry_from_the_same_frame_time() -> None:
    geometry = _RecordingRgbCameraGeometry()
    observed_times = iter((10.0, 10.2))
    provider, sources, _revision, _profile = _provider(
        wrist_depth=True,
        rgb_camera_geometry=geometry,
        clock=lambda: next(observed_times),
    )

    assert tuple(item["role"] for item in provider.depth_source_catalog()) == ("main",)
    assert tuple(item["role"] for item in provider.aligned_rgbd_source_catalog()) == (
        "main",
        "wrist_left",
    )

    capture = provider.capture_aligned_rgbd(
        role="wrist_left",
        expected_station_revision=7,
    )

    assert sources["left_wrist"].latest_frame_reads == 1
    assert geometry.stationary_cut_calls == [("left_wrist", 9.97, 10.2)]
    assert geometry.pose_cut_calls == [("left_wrist", 9.97, 10.2)]
    assert capture.source_id == "left_wrist"
    assert capture.role == "wrist_left"
    assert capture.source_from_camera == pytest.approx(
        tuple(float(value) for value in np.eye(4).reshape(-1))
    )


def test_plain_depth_capture_does_not_expose_dynamic_wrist_geometry() -> None:
    provider, _sources, _revision, _profile = _provider(
        wrist_depth=True,
        rgb_camera_geometry=_RecordingRgbCameraGeometry(),
        clock=lambda: 10.2,
    )

    with pytest.raises(ValueError, match="Depth role is not registered"):
        provider.capture_depth(role="wrist_left", expected_station_revision=7)


@pytest.mark.parametrize(
    "failure",
    (
        "real_rgb_camera_pose_unavailable:left_wrist",
        "real_rgb_camera_pose_stale:left_wrist",
        "real_rgb_camera_frame_pose_skew_exceeded:left_wrist",
        "real_rgb_camera_pose_from_future:left_wrist",
    ),
)
def test_aligned_wrist_rgbd_preserves_dynamic_geometry_fail_closed_errors(
    failure,
) -> None:
    observed_times = iter((10.0, 10.2))
    provider, _sources, _revision, _profile = _provider(
        wrist_depth=True,
        rgb_camera_geometry=_RecordingRgbCameraGeometry(failure=failure),
        clock=lambda: next(observed_times),
    )

    with pytest.raises(RuntimeError, match=failure):
        provider.capture_aligned_rgbd(
            role="wrist_left",
            expected_station_revision=7,
        )


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
        (
            lambda provider, sources, revision: revision.update(value=8),
            "revision_mismatch",
        ),
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
    assert tuple(item["role"] for item in provider.depth_source_catalog()) == ("main",)
    assert tuple(item["role"] for item in provider.aligned_rgbd_source_catalog()) == (
        "main",
    )
    assert set(binding.world_from_depth_camera_by_source) == {"front"}
    assert all(source.config.enable_depth for source in binding.camera_sources.values())
    assert all(
        source.device_info()["running"] is False
        for source in binding.camera_sources.values()
    )


def test_source_binding_rejects_static_pose_for_depth_disabled_source() -> None:
    with pytest.raises(ValueError, match="belong to enabled depth sources"):
        RealPerceptionSourceBinding(
            camera_sources={
                "front": _Source(
                    "front",
                    timestamp=1.0,
                    enable_depth=False,
                    serial="front-serial",
                )
            },
            world_from_depth_camera_by_source={"front": np.eye(4)},
            semantic_content_hash="source-binding:test",
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


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_static_preparation_runs_once_before_any_device_start(monkeypatch, fail_preparation):
    events = []

    def prepare():
        events.append("prepare")
        if fail_preparation:
            raise ModuleNotFoundError("missing CAD runtime")

    monkeypatch.setattr(realsense_perception, "_formal_close_camera_sources",
        lambda _: events.append("camera.close"))
    resource = lambda name: SimpleNamespace(
        start=lambda: events.append(name + ".start"),
        close=lambda: events.append(name + ".close"))
    resources = ExclusiveRealPerceptionResources(
        lower=resource("arm"), camera_sources={"front": object()},
        artifact_store=resource("store"), additional_resources=(resource("model"),),
        prepare_static_geometry=prepare,
        start_cameras=lambda _: events.append("camera.start"),
        admit_cameras=lambda _: events.append("camera.admit"),
    )
    if fail_preparation:
        with pytest.raises(ModuleNotFoundError, match="missing CAD runtime"):
            resources.start()
        assert resources._closed
        with pytest.raises(RuntimeError, match="cannot restart"):
            resources.start()
        resources.close()
        resources.close()
        assert events == ["prepare", "store.close"]
    else:
        resources.start()
        resources.start()
        assert events == ["prepare", "camera.start", "camera.admit", "model.start", "arm.start"]
        resources.close()
        resources.close()
        assert events[5:] == ["arm.close", "model.close", "camera.close", "store.close"]


def test_static_preparation_failure_retries_only_unfinished_store_close(monkeypatch):
    events = []

    def prepare():
        raise ModuleNotFoundError("missing CAD runtime")

    def close_store():
        events.append("store.close")
        if len(events) == 1:
            raise RuntimeError("store is still busy")

    def unexpected():
        pytest.fail("No device or model lifecycle may run after static preparation failed")

    monkeypatch.setattr(realsense_perception, "_formal_close_camera_sources", lambda _: unexpected())
    resources = ExclusiveRealPerceptionResources(
        lower=SimpleNamespace(start=unexpected, close=unexpected),
        camera_sources={"front": object()},
        additional_resources=(SimpleNamespace(start=unexpected, close=unexpected),),
        artifact_store=SimpleNamespace(close=close_store), prepare_static_geometry=prepare,
        start_cameras=lambda _: unexpected(), admit_cameras=lambda _: unexpected(),
    )
    with pytest.raises(ModuleNotFoundError, match="missing CAD runtime"):
        resources.start()
    assert not resources._closed
    with pytest.raises(RuntimeError, match="cannot restart"):
        resources.start()
    resources.close()
    resources.close()
    assert resources._closed and events == ["store.close", "store.close"]


def test_resource_transaction_retries_only_incomplete_close_stages(monkeypatch) -> None:
    events: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name
            self.starts = 0
            self.closes = 0

        def start(self) -> None:
            self.starts += 1

        def close(self) -> None:
            self.closes += 1
            events.append(f"{self.name}.close:{self.closes}")

    lower = _Resource("lower")
    model = _Resource("model")
    store = _Resource("store")
    camera_close_calls = 0

    def close_cameras(_sources) -> None:
        nonlocal camera_close_calls
        camera_close_calls += 1
        events.append(f"cameras.close:{camera_close_calls}")
        if camera_close_calls == 1:
            raise RuntimeError("injected camera close failure")

    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        close_cameras,
    )
    resources = ExclusiveRealPerceptionResources(
        lower=lower,
        camera_sources={"front": object()},
        artifact_store=store,
        additional_resources=(model,),
        start_cameras=lambda _sources: None,
        admit_cameras=lambda _sources: None,
    )
    resources.start()

    with pytest.raises(RuntimeError, match="injected camera close failure"):
        resources.close()
    resources.close()
    resources.close()

    assert lower.closes == 1
    assert model.closes == 1
    assert store.closes == 1
    assert camera_close_calls == 2
    assert events == [
        "lower.close:1",
        "model.close:1",
        "cameras.close:1",
        "store.close:1",
        "cameras.close:2",
    ]


def test_camera_only_resources_retry_only_incomplete_close_stages(monkeypatch) -> None:
    model = SimpleNamespace(starts=0, closes=0)
    store = SimpleNamespace(closes=0)
    model.start = lambda: setattr(model, "starts", model.starts + 1)
    model.close = lambda: setattr(model, "closes", model.closes + 1)
    store.close = lambda: setattr(store, "closes", store.closes + 1)
    camera_close_calls = 0

    def close_cameras(_sources) -> None:
        nonlocal camera_close_calls
        camera_close_calls += 1
        if camera_close_calls == 1:
            raise RuntimeError("injected camera close failure")

    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        close_cameras,
    )
    resources = CameraOnlyRealPerceptionResources(
        camera_sources={"front": object()},
        artifact_store=store,
        additional_resources=(model,),
        start_cameras=lambda _sources: None,
        admit_cameras=lambda _sources: None,
    )
    resources.start()

    with pytest.raises(RuntimeError, match="injected camera close failure"):
        resources.close()
    resources.close()
    resources.close()

    assert model.closes == 1
    assert store.closes == 1
    assert camera_close_calls == 2


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


def test_camera_only_resources_do_not_claim_close_until_native_stop_finishes():
    from teleop_runtime.sources.camera import (
        RealSenseRGBCameraConfig,
        RealSenseRGBSource,
    )
    release = threading.Event()
    calls = []
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="unit", pipeline_stop_timeout_sec=.02))
    def stop():
        calls.append("stop")
        release.wait(2)
    source._pipeline = SimpleNamespace(stop=stop)
    resources = CameraOnlyRealPerceptionResources(
        camera_sources={"left_wrist": source},
        artifact_store=SimpleNamespace(close=lambda: calls.append("store.close")),
    )
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="pipeline.stop"):
                resources.close()
            assert not resources._closed
            assert not resources._cameras_closed
        assert calls == ["stop", "store.close"]
    finally:
        release.set()
        assert source.wait_for_pending_pipeline_stop(timeout_sec=1) is None
        resources.close()
    assert resources._closed and resources._cameras_closed
    assert calls == ["stop", "store.close"]


@pytest.mark.parametrize("color_space", ["rgb", "bgr"])
def test_rgb_cuts_are_frozen_before_encoding_and_preserve_requested_order(monkeypatch, color_space):
    provider, sources, _revision, _profile = _provider()
    events, images, image_refs = [], [], []
    expected = {}
    for key, source in sources.items():
        source.config.color_space = color_space
        expected[key] = source.frame.image.copy()
        if color_space == "bgr":
            expected[key] = expected[key][..., ::-1]
        latest = source.get_latest_frame
        def observed(*, key=key, latest=latest):
            events.append(("read", key))
            return latest()
        monkeypatch.setattr(source, "get_latest_frame", observed)
    encode = realsense_perception._encode_rgb_jpeg
    def encoding(image, **kwargs):
        assert [e for e in events if e[0] == "read"] == [("read", "left_wrist"), ("read", "front")]
        events.append(("encode", len(images)))
        # Emulate a camera worker reusing its input buffers during compression.
        for source in sources.values():
            source.frame.image[:] = 0
        image_refs.append(weakref.ref(image))
        images.append(image.copy())
        return encode(image, **kwargs)
    monkeypatch.setattr(realsense_perception, "_encode_rgb_jpeg", encoding)
    result = provider.capture_rgb({"selected_roles": ["wrist_left", "main"],
        "width": 2, "height": 2, "max_total_bytes": 100_000}, expected_station_revision=7)
    assert [f.role for f in result.frames] == ["wrist_left", "main"]
    for image, key in zip(images, ("left_wrist", "front"), strict=True):
        np.testing.assert_array_equal(image, expected[key])
        assert not np.shares_memory(image, sources[key].frame.image)
    assert all(reference() is None for reference in image_refs)
    assert [source.starts for source in sources.values()] == [0, 0]


def test_rgb_checks_skew_before_encoding_and_revision_after_encoding(monkeypatch):
    skewed, _sources, _revision, _profile = _provider(front_at=9.86, wrist_at=9.98)
    monkeypatch.setattr(realsense_perception, "_encode_rgb_jpeg", lambda *a, **kw: pytest.fail("encoded rejected cuts"))
    parameters = {"selected_roles": ["main", "wrist_left"], "width": 2, "height": 2, "max_total_bytes": 100_000}
    with pytest.raises(RuntimeError, match="inter_camera_skew"):
        skewed.capture_rgb(parameters, expected_station_revision=7)
    provider, sources, revision, _profile = _provider()
    def changed(image, **kwargs):
        revision["value"] = 8
        return b"jpeg"
    monkeypatch.setattr(realsense_perception, "_encode_rgb_jpeg", changed)
    with pytest.raises(RuntimeError, match="station_revision_changed"):
        provider.capture_rgb(parameters, expected_station_revision=7)
    for source in sources.values():
        assert source.starts == source.closes == 0


def test_single_start_policy_is_bound_before_sources_start() -> None:
    options = {
        "camera_config_path": "teleop_runtime/configs/devices/agent_gateway_realsense.json",
        "camera_registry_path": "robot_motion/configs/rm65_camera_registry.json",
        "scene_world_from_arm_base_by_side": {"left": np.eye(4), "right": np.eye(4)},
    }
    default = build_real_perception_source_binding(**options)
    binding = build_real_perception_source_binding(**options, single_start=True)
    assert binding.semantic_content_hash != default.semantic_content_hash
    for key, source in binding.camera_sources.items():
        config = source.config
        assert (config.width, config.height, config.fps) == (640, 480, 15)
        assert config.serial == default.camera_sources[key].config.serial
        assert config.enable_depth and config.align_depth_to_color
        assert config.require_complete_frameset
        assert config.startup_software_retry_attempts == 0
        assert config.auto_exposure_priority is None
        assert not any(getattr(config, field) for field in (
            "startup_reset", "startup_hardware_reset", "startup_hardware_reset_on_failure",
            "startup_hardware_reset_on_warmup_failure", "restart_on_stale_timeout",
            "restart_on_runtime_error", "restart_on_black_frames", "restart_on_incomplete_frameset",
        ))
        assert source.device_info()["running"] is False
    assert default.camera_sources["front"].config.restart_on_incomplete_frameset
