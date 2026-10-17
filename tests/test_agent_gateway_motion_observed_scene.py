import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.motion_observed_scene import (
    ObservedSafeMovePlanner,
    SafeMotionObservationConfig,
    obstacle_cloud_from_captures,
)
from agent_gateway.profiles.real.motion_observed_scene_debug import (
    SafeMotionDebugRecord,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealAlignedRgbdCapture,
    RealPerceptionSourceBinding,
)
from agent_gateway.profiles.real.rm65_camera_geometry import (
    RM65RgbCameraGeometry,
    RM65StationaryGeometryUnavailable,
)
from robot_motion.observed_obstacles import ObservedObstacleCloud
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilterResult,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def capture(role="main", **changes):
    base = RealAlignedRgbdCapture(source_id=role, role=role, acquired_at=9.95,
        station_revision=7, source_generation=2, frame_index=10, capture_group_id="capture:1",
        source_binding_hash="source:test", width=4, height=4,
        intrinsics={"fx": 100., "fy": 100., "cx": 2., "cy": 2.},
        source_from_camera=tuple(np.eye(4).reshape(-1)), rgb=np.zeros((4, 4, 3), np.uint8),
        depth_m=np.full((4, 4), .5), rgb_content_hash="rgb:test", depth_content_hash="depth:test",
        capture_qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        pose_measured_at=9.95, pose_available_at=9.951, pose_evidence_refs=("qpos:test",))
    return replace(base, **changes)


def self_filter(**kwargs):
    assert not kwargs["target_mask"].any()  # No implicit allowed contact with a target.
    depth = kwargs["depth_m"].copy()
    removed = np.zeros(depth.shape, bool)
    removed[0, :] = True
    depth[removed] = 0
    return RM65CapturedRobotObservationFilterResult(depth, removed, {"method": "test_self_filter"})


def test_model_independent_scene_keeps_metric_depth_and_freezes_all_views():
    captures = [capture(), capture("wrist_left"), capture("wrist_right")]
    cloud = obstacle_cloud_from_captures(captures=captures, config=SafeMotionObservationConfig(),
        reference_frame_id="world", self_filter=self_filter,
        planning_qpos_by_side=captures[0].capture_qpos_by_side, max_inter_camera_skew_sec=.1)
    assert cloud.points_m[:, 2].mean() == pytest.approx(.502)  # Meters, no second depth scale.
    assert len(cloud.evidence["views"]) == 3
    assert not cloud.points_m.flags.writeable
    with pytest.raises(TypeError):
        cloud.evidence["views"][0]["role"] = "changed"
    assert all(v["self_filter"]["method"] == "test_self_filter" for v in cloud.evidence["views"])


def test_rgbd_stationarity_is_resolved_before_pausing_lower_measurements():
    now, paused, events = [10.], [False], []
    binding = RealPerceptionSourceBinding(camera_sources={"front": SimpleNamespace(config=SimpleNamespace(enable_depth=True))},
        world_from_depth_camera_by_source={"front": np.eye(4)}, semantic_content_hash="source:test")
    geometry = RM65RgbCameraGeometry(source_binding=binding,
        target_builder=SimpleNamespace(solver_world_targets_for_qpos_pair=lambda **kw:
            {side: np.eye(4) for side in ("left", "right")}),
        static_source_from_camera={"front": np.eye(4)}, dynamic_mounts=(), urdf_sha256="cad:test")
    for measured in (9.80, 9.85, 9.90, 9.95, 9.98):
        geometry.publish_measured_qpos(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
            measured_at=measured, available_at=measured+.001, evidence_refs=("qpos:test",))

    @contextmanager
    def window():
        paused[0] = True
        now[0] += .10  # SDK/read-window or debug-write latency while sampling is paused.
        events.append("pause")
        try:
            yield
        finally:
            paused[0] = False

    def read(**kwargs):
        events.append("capture")
        _cut, pair, measured, available, refs = geometry.stationary_geometry_cut(
            source_id="front", frame_acquired_at=now[0]-.01, now=now[0])
        return capture(acquired_at=now[0]-.01, capture_qpos_by_side=pair,
            pose_measured_at=measured, pose_available_at=available, pose_evidence_refs=refs)

    def filt(**kwargs):
        assert paused[0]  # Expensive work still runs inside the protected read window.
        events.append("filter")
        return self_filter(**kwargs)

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=read,
        max_inter_camera_skew_ms=100), current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=filt, config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=lambda cloud: lambda **kw: "trajectory", clock=lambda: now[0],
        non_motion_read_window=window)
    result = adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, require_grasp_approach=False)
    assert result == "trajectory" and events == ["capture", "pause", "filter"]
    assert not paused[0]


def test_stationary_failure_details_survive_planning_and_debug_record(tmp_path):
    diagnostic = {"failed_conditions": ["last_sample_before_frame"], "sample_count": 4,
                  "frame_acquired_at": 10., "last_measured_at": 9.9}
    def read(**kwargs):
        raise RM65StationaryGeometryUnavailable("grasp_observation_stationary_history_unavailable", diagnostic)
    def forbidden(cloud):
        pytest.fail("Stationary admission failure must not construct a planner")
    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=read,
        max_inter_camera_skew_ms=100), current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=self_filter, config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=forbidden, clock=lambda: 10., debug_record_dir=tmp_path)
    with pytest.raises(RM65GraspTrajectoryPlanningError) as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, require_grasp_approach=False)
    assert caught.value.reason_code == "safe_motion_observation_unavailable"
    assert caught.value.details["observation_diagnostic"] == diagnostic
    directory, = tmp_path.glob("scene-*")
    record = json.loads((directory / "record.json").read_text())
    assert record["error"]["details"]["observation_diagnostic"] == diagnostic


@pytest.mark.parametrize("problem", ["missing_view", "skew", "missing_qpos", "drift", "empty_depth"])
def test_observed_scene_rejects_incomplete_or_incoherent_cuts(problem):
    captures = [capture(), capture("wrist_left"), capture("wrist_right")]
    if problem == "missing_view":
        captures.pop()
    elif problem == "skew":
        captures[1] = replace(captures[1], acquired_at=9.7)
    elif problem == "missing_qpos":
        captures[1] = replace(captures[1], capture_qpos_by_side=None)
    elif problem == "drift":
        captures[1] = replace(captures[1], capture_qpos_by_side={s: np.ones(6)*.02 for s in ("left", "right")})
    else:
        captures[1] = replace(captures[1], depth_m=np.full((4, 4), 6.553))
    with pytest.raises(ValueError):
        obstacle_cloud_from_captures(captures=captures, config=SafeMotionObservationConfig(),
            reference_frame_id="world", self_filter=self_filter,
            planning_qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, max_inter_camera_skew_sec=.1)


@pytest.mark.parametrize("problem", [None, "missing", "expired", "revision", "planner_failed"])
def test_live_adapter_reads_existing_provider_once_and_never_falls_back(problem):
    events, now, revision = [], [10.], [7]

    @contextmanager
    def window():
        events.append("window_enter")
        yield
        events.append("window_exit")

    def read(**kwargs):
        assert kwargs["require_stationary"] and kwargs["expected_station_revision"] == 7
        events.append("read:" + kwargs["role"])
        if problem == "missing":
            raise RuntimeError("real_perception_frame_unavailable")
        return capture(kwargs["role"])

    def factory(cloud):
        events.append("planner_factory")
        assert cloud.evidence["views"][0]["depth_content_hash"] == "depth:test"
        def plan(**kwargs):
            events.append("plan")
            if problem == "expired":
                now[0] = 41.
            elif problem == "revision":
                revision[0] = 8
            elif problem == "planner_failed":
                raise RM65GraspTrajectoryPlanningError("path_search_exhausted")
            return "trajectory"
        return plan

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=read,
        max_inter_camera_skew_ms=100), current_station_revision=lambda: revision[0],
        reference_frame_id="world", self_filter=self_filter, config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=factory, clock=lambda: now[0], non_motion_read_window=window)
    kwargs = {"qpos_by_side": {s: np.zeros(6) for s in ("left", "right")}, "require_grasp_approach": False}
    if problem is None:
        assert adapter(**kwargs) == "trajectory"
        assert events == ["read:main", "window_enter", "planner_factory", "plan", "window_exit"]
    else:
        with pytest.raises(RM65GraspTrajectoryPlanningError):
            adapter(**kwargs)
        assert events.count("plan") == (0 if problem == "missing" else 1)


def debug_adapter(tmp_path, *, problem="rejected", now=None, events=None):
    from robot_motion.path_planning import JointPathPlannerConfig
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspTrajectoryPlanningConfig,
    )
    now = [10.] if now is None else now
    events = [] if events is None else events
    context_file = tmp_path / "input-config.json"
    context_file.write_text('{"serial":"unit-test-camera"}')

    def read(**kwargs):
        events.append("read:" + kwargs["role"])
        if problem == "missing" and kwargs["role"] == "wrist_left":
            raise RuntimeError("missing view")
        return capture(kwargs["role"])

    def factory(cloud):
        events.append("factory")

        class Planner:
            config = RM65GraspTrajectoryPlanningConfig()
            planner_config = JointPathPlannerConfig()
            scene = SimpleNamespace(to_dict=lambda: {"minimum_clearance_m": .02})
            profile = SimpleNamespace(semantic_content_hash="profile:test",
                tool_tcp_pose_flange_m_rad_by_side={side: np.zeros(6) for side in ("left", "right")})
            path_collision_scene_profile = "test:scene"
            oracle = SimpleNamespace(world_obstacle_proxy_boxes=lambda pair: (
                ("arm_link3_proxy_0", np.eye(4), np.array([.05, .08, .06])),))

            def __call__(self, **kwargs):
                events.append("plan")
                # Inputs must already be inspectable if this process dies here.
                record = json.loads(next(tmp_path.glob("scene-*/record.json")).read_text())
                assert record["state"] == "recording" and record["stages"][-1]["phase"] == "planning"
                assert set(record["views"]) == {"main", "wrist_left", "wrist_right"}
                if problem == "rejected":
                    raise RM65GraspTrajectoryPlanningError("path_start_collision_blocked", details={
                        "minimum_distance_m": -.014,
                        "nearest_observed": MappingProxyType({"geometry_name": "arm_link3_proxy_0",
                            "occupied_center_world_m": (-.182, -.190, .462)})})
                if problem == "interrupt":
                    raise KeyboardInterrupt()
                if problem == "generic":
                    raise RuntimeError("native geometry failed")
                return SimpleNamespace(semantic_content_hash="trajectory:test")

        return Planner()

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=read,
        max_inter_camera_skew_ms=100), current_station_revision=lambda: 7,
        reference_frame_id="world", self_filter=self_filter, config=SafeMotionObservationConfig(),
        planner_factory=factory, clock=lambda: now[0], debug_record_dir=tmp_path,
        debug_context={"runtime_session_id": "session:test"},
        debug_context_files={"camera-config.json": context_file})
    kwargs = {"qpos_by_side": {s: np.zeros(6) for s in ("left", "right")},
        "side": "right", "target_tool_tcp_world": np.eye(4), "speed_scale": .2,
        "require_grasp_approach": False}
    return adapter, kwargs, events


@pytest.mark.parametrize("problem,state", [("rejected", "rejected"), ("success", "planned"),
    ("interrupt", "interrupted"), ("generic", "error")])
def test_debug_keeps_exact_inputs_masks_cloud_geometry_and_outcome(tmp_path, problem, state):
    adapter, kwargs, events = debug_adapter(tmp_path, problem=problem)
    if problem == "success":
        assert adapter(**kwargs).semantic_content_hash == "trajectory:test"
    else:
        expected = KeyboardInterrupt if problem == "interrupt" else RuntimeError
        with pytest.raises(expected) as caught:
            adapter(**kwargs)
    directory, = tmp_path.glob("scene-*")
    record = json.loads((directory / "record.json").read_text())
    assert record["state"] == state
    timings = record["timing"]
    assert [e["role"] for e in timings if e["phase"] == "capture_rgbd"] == list(adapter.config.roles)
    assert [e["role"] for e in timings if e["phase"] == "self_filter"] == list(adapter.config.roles)
    assert timings[-1]["phase"] == "path_planning"
    assert all(e["wall_sec"] >= 0 for e in timings)
    assert events[:3] == ["read:main", "read:wrist_left", "read:wrist_right"]
    assert events.count("plan") == 1
    assert record["depth_unit"] == "meters" and record["rgb_order"] == "RGB"
    assert record["context"]["runtime_session_id"] == "session:test"
    assert record["request"]["side"] == "right"
    assert record["planner"]["start_obstacle_proxy_boxes"][0]["name"] == "arm_link3_proxy_0"
    assert (directory / "camera-config.json").read_bytes() == (tmp_path / "input-config.json").read_bytes()
    for role, view in record["views"].items():
        original = capture(role)
        assert view["acquired_at"] == original.acquired_at
        assert view["depth_content_hash"] == original.depth_content_hash
        assert view["capture_group_id"] == original.capture_group_id
        assert view["pose_available_at"] == original.pose_available_at
        with np.load(directory / view["rgbd_file"], allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["rgb"], original.rgb)
            np.testing.assert_array_equal(raw["depth_m"], original.depth_m)
        with np.load(directory / view["filtered_file"], allow_pickle=False) as filtered:
            assert filtered["self_removed_mask"][0].all()
            assert not filtered["background_valid_mask"][0].any()
            assert np.all(filtered["self_filtered_depth_m"][0] == 0)
            points = filtered["points_world_m"]
            uv = filtered["pixel_uv"]
            expected = np.column_stack(((uv[:, 0] - 2) * .5 / 100, (uv[:, 1] - 2) * .5 / 100,
                                        np.full(len(uv), .5)))
            np.testing.assert_array_equal(points, expected)
        assert (directory / view["rgb_preview"]).is_file()
    with np.load(directory / record["cloud"]["file"], allow_pickle=False) as data:
        replay = ObservedObstacleCloud(reference_frame_id=record["reference_frame_id"],
            points_m=data["points_m"], evidence=record["cloud"]["evidence"],
            voxel_size_m=record["cloud"]["voxel_size_m"])
        assert replay.semantic_content_hash == record["cloud"]["scene_hash"]
    baseline = obstacle_cloud_from_captures(captures=[capture(role) for role in record["policy"]["roles"]],
        config=SafeMotionObservationConfig(), reference_frame_id="world", self_filter=self_filter,
        planning_qpos_by_side=kwargs["qpos_by_side"], max_inter_camera_skew_sec=.1)
    assert baseline.semantic_content_hash == record["cloud"]["scene_hash"]  # Recording changes no scene input.
    if problem == "rejected":
        assert record["error"]["reason_code"] == "path_start_collision_blocked"
        assert record["error"]["details"]["nearest_observed"]["geometry_name"] == "arm_link3_proxy_0"
        assert caught.value.details["debug_record_dir"] == str(directory)
    assert record["recording_io_sec"] >= 0
    assert directory.stat().st_mode & 0o077 == 0
    assert not list(directory.glob(".partial-*"))


def test_debug_retains_partial_capture_without_planning_or_recapture(tmp_path):
    adapter, kwargs, events = debug_adapter(tmp_path, problem="missing")
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_observation_unavailable"):
        adapter(**kwargs)
    directory, = tmp_path.glob("scene-*")
    record = json.loads((directory / "record.json").read_text())
    assert record["state"] == "rejected" and set(record["views"]) == {"main"}
    assert (directory / "main.npz").is_file()
    assert events == ["read:main", "read:wrist_left"]


def test_debug_write_failure_prevents_planning_and_retains_completed_files(tmp_path, monkeypatch):
    adapter, kwargs, events = debug_adapter(tmp_path)
    original = np.savez
    def fail_cloud(stream, **arrays):
        if set(arrays) == {"points_m"}:
            raise OSError("disk full")
        return original(stream, **arrays)
    monkeypatch.setattr(np, "savez", fail_cloud)
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_debug_record_failed"):
        adapter(**kwargs)
    directory, = tmp_path.glob("scene-*")
    assert (directory / "main.npz").is_file() and not (directory / "obstacles.npz").exists()
    assert "plan" not in events and "factory" not in events
    assert json.loads((directory / "record.json").read_text())["state"] == "recording"


def test_debug_cannot_extend_scene_freshness_after_successful_planning(tmp_path, monkeypatch):
    now = [10.]
    adapter, kwargs, _ = debug_adapter(tmp_path, problem="success", now=now)
    original = SafeMotionDebugRecord.finish
    def delayed_finish(self, **outcome):
        original(self, **outcome)
        if outcome.get("trajectory") is not None:
            now[0] = 41.
    monkeypatch.setattr(SafeMotionDebugRecord, "finish", delayed_finish)
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_observation_expired"):
        adapter(**kwargs)
    record = json.loads(next(tmp_path.glob("scene-*/record.json")).read_text())
    assert record["state"] == "rejected"


def test_debug_projects_nested_rejection_points_into_the_same_saved_frame(tmp_path):
    record = SafeMotionDebugRecord(tmp_path, reference_frame_id="world", config=SafeMotionObservationConfig(),
        request={}, context={"runtime_session_id": "test"})
    record.capture(capture())
    error = RM65GraspTrajectoryPlanningError("path_target_collision_blocked", details={"checks": [{
        "nearest_observed": {"geometry_name": "test_proxy", "occupied_center_world_m": [0, 0, .5]}}]})
    record.finish(error=error)
    data = json.loads((record.directory / "record.json").read_text())
    projection, = data["nearest_point_projections"]
    assert projection["sampled_pixel_uv"] == [2, 2] and projection["in_image"]
    assert projection["projected_depth_m"] == projection["observed_depth_m"] == .5
    assert projection["depth_residual_m"] == 0


def test_debug_directory_creation_failure_is_explicit_and_never_captures(tmp_path, monkeypatch):
    from agent_gateway.profiles.real import motion_observed_scene_debug
    adapter, kwargs, events = debug_adapter(tmp_path)
    def fail(**kwargs):
        raise PermissionError("unwritable directory")
    monkeypatch.setattr(motion_observed_scene_debug.tempfile, "mkdtemp", fail)
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_debug_record_failed"):
        adapter(**kwargs)
    assert events == []


def test_debug_finish_write_failure_keeps_the_original_planning_reason(tmp_path, monkeypatch):
    adapter, kwargs, events = debug_adapter(tmp_path)
    original = SafeMotionDebugRecord._checkpoint
    def fail(self):
        if self.payload["state"] == "rejected":
            raise RM65GraspTrajectoryPlanningError("safe_motion_debug_record_failed")
        original(self)
    monkeypatch.setattr(SafeMotionDebugRecord, "_checkpoint", fail)
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_debug_record_failed") as caught:
        adapter(**kwargs)
    assert caught.value.details["original_error"] == "path_start_collision_blocked"
    assert events.count("plan") == 1


@pytest.mark.parametrize("problem", ["symlink", "writable", "missing", "relative"])
def test_debug_root_is_validated_before_capture(tmp_path, problem):
    root = tmp_path / "debug"
    root.mkdir()
    if problem == "symlink":
        link = tmp_path / "link"
        link.symlink_to(root)
        root = link
    elif problem == "writable":
        root.chmod(0o777)
    elif problem == "missing":
        root = tmp_path / "absent"
    else:
        root = Path(".")
    with pytest.raises((ValueError, OSError)):
        ObservedSafeMovePlanner(provider=None, current_station_revision=None,
            reference_frame_id="world", self_filter=self_filter, config=SafeMotionObservationConfig(),
            planner_factory=None, clock=None, debug_record_dir=root)


@pytest.mark.parametrize("failure", [None, "collision", "capture", "interrupt", "disk"])
def test_normal_safe_skill_retains_its_exact_inputs_without_second_capture(tmp_path, monkeypatch, failure):
    reads = []
    planned = []

    def read(**kwargs):
        reads.append(kwargs["role"])
        if failure == "capture" and kwargs["role"] == "wrist_left":
            raise RuntimeError("camera missing")
        return capture(kwargs["role"])

    def factory(cloud):
        def plan(**kwargs):
            planned.append(True)
            record = next(tmp_path.iterdir())
            assert (record / "snapshot.json").is_file()  # Before planner can fail or be interrupted.
            assert (record / "obstacles.npz").is_file()
            if failure == "collision":
                raise RM65GraspTrajectoryPlanningError("path_start_collision_blocked",
                    details={"nearest_observed": {"geometry_name": "arm_link3_proxy_0"}})
            if failure == "interrupt":
                raise KeyboardInterrupt()
            return "trajectory"
        return plan

    if failure == "disk":
        monkeypatch.setattr(np, "savez", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))
    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=read,
        max_inter_camera_skew_ms=100), current_station_revision=lambda: 7,
        reference_frame_id="world", self_filter=self_filter, config=SafeMotionObservationConfig(),
        planner_factory=factory, clock=lambda: 10., observation_record_dir=tmp_path)
    kwargs = {"qpos_by_side": {s: np.zeros(6) for s in ("left", "right")},
              "side": "right", "target_tool_tcp_world": np.eye(4), "require_grasp_approach": False}
    if failure is None:
        assert adapter(**kwargs) == "trajectory"
    else:
        with pytest.raises((RM65GraspTrajectoryPlanningError, KeyboardInterrupt, OSError)):
            adapter(**kwargs)
    record = next(tmp_path.iterdir())
    result = json.loads((record / "result.json").read_text())
    assert result["status"] == ("path_found" if failure is None else "failed")
    assert reads == (["main", "wrist_left"] if failure == "capture" else list(adapter.config.roles))
    assert len(planned) == (0 if failure in {"capture", "disk"} else 1)
    assert not list(record.glob("*.html"))
    if failure != "disk":
        saved = json.loads((record / "snapshot.json").read_text())
        assert saved["request"]["side"] == "right"
        with np.load(record / "main.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["depth_m"], capture().depth_m)
            np.testing.assert_array_equal(arrays["rgb"], capture().rgb)
        assert (record / "main.jpg").is_file()
    if failure == "collision":
        assert result["details"]["nearest_observed"]["geometry_name"] == "arm_link3_proxy_0"
    if failure in {None, "collision", "interrupt"}:
        from scripts.probes.inspect_safe_move_snapshot import load_captures
        frozen = load_captures(record, None, None, None)
        assert [cut.role for cut in frozen] == list(adapter.config.roles)
        assert frozen[0].source_from_camera == capture().source_from_camera
        assert frozen[0].pose_measured_at == capture().pose_measured_at
        np.testing.assert_array_equal(frozen[0].depth_m, capture().depth_m)



def test_rgbd_stationary_capture_precedes_qpos_pause_with_native_geometry():
    from agent_gateway.profiles.real.rm65_camera_geometry import RM65RgbCameraGeometry
    from test_agent_gateway_real_realsense_perception import _provider

    now, paused = [10.], [False]
    provider, sources, _, _ = _provider(clock=lambda: now[0])
    geometry = RM65RgbCameraGeometry(source_binding=provider.source_binding,
        target_builder=SimpleNamespace(solver_world_targets_for_qpos_pair=lambda **_: {
            "left": np.eye(4), "right": np.eye(4)}),
        static_source_from_camera={source: np.eye(4) for source in sources},
        dynamic_mounts=(), urdf_sha256="test")
    provider.rgb_camera_geometry = geometry
    for timestamp in (9.8, 9.85, 9.9, 9.95, 9.99):
        geometry.publish_measured_qpos(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
            measured_at=timestamp, available_at=timestamp+.001, evidence_refs=("lower:qpos",))
    reads = []
    source = sources["front"]

    def current_frame():
        reads.append(paused[0])
        return replace(source.frame, source_timestamp=now[0]-.02,
                       receive_timestamp=now[0]-.02, available_timestamp=now[0]-.019,
                       depth_image=np.full((3, 4), 500, dtype=np.uint16))

    source.get_latest_frame = current_frame

    @contextmanager
    def window():
        paused[0] = True
        now[0] += .16  # Entering this window can wait for an in-flight SDK read.
        try:
            yield
        finally:
            paused[0] = False

    def factory(cloud):
        assert paused[0]  # Only frozen processing/planning pauses qpos refresh.
        assert cloud.evidence["views"][0]["acquired_at"] == pytest.approx(9.98)
        return lambda **_: "trajectory"

    adapter = ObservedSafeMovePlanner(provider=provider, current_station_revision=lambda: 7,
        reference_frame_id="world", self_filter=self_filter,
        config=SafeMotionObservationConfig(roles=("main",)), planner_factory=factory,
        clock=lambda: now[0], non_motion_read_window=window)
    assert adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                   require_grasp_approach=False) == "trajectory"
    assert reads == [False]



def test_capture_failure_reports_stage_role_and_inner_error():
    def capture_error(**kwargs):
        raise RuntimeError("grasp_observation_stationary_history_unavailable")

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(capture_aligned_rgbd=capture_error),
        current_station_revision=lambda: 7, reference_frame_id="world", self_filter=self_filter,
        config=SafeMotionObservationConfig(roles=("wrist_left",)),
        planner_factory=lambda _: pytest.fail("No planning after capture failure"), clock=lambda: 10.)
    with pytest.raises(RM65GraspTrajectoryPlanningError) as failure:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, require_grasp_approach=False)
    assert failure.value.reason_code == "safe_motion_observation_unavailable"
    assert failure.value.details["stage"] == "capture_rgbd"
    assert failure.value.details["role"] == "wrist_left"
    assert failure.value.details["message"] == "grasp_observation_stationary_history_unavailable"



def test_cloud_failure_keeps_capture_identity_and_depth_counts():
    def empty_filter(**kwargs):
        depth = np.zeros_like(kwargs["depth_m"])
        return RM65CapturedRobotObservationFilterResult(depth, np.ones(depth.shape, bool), {})

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(
        capture_aligned_rgbd=lambda **kwargs: capture(kwargs["role"]), max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world", self_filter=empty_filter,
        config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=lambda _: pytest.fail("No planning with empty obstacle depth"), clock=lambda: 10.)
    with pytest.raises(RM65GraspTrajectoryPlanningError) as failure:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, require_grasp_approach=False)
    assert failure.value.details["stage"] == "obstacle_cloud"
    frame = failure.value.details["capture_frames"][0]
    assert (frame["role"], frame["source_id"], frame["frame_index"]) == ("main", "main", 10)
    assert frame["valid_depth_pixels"] == frame["in_range_depth_pixels"] == 16



@pytest.mark.parametrize("change", ["revision", "age"])
def test_scene_changed_while_entering_window_is_rejected_before_processing(change):
    now, revision = [10.], [7]

    @contextmanager
    def window():
        if change == "revision":
            revision[0] += 1
        else:
            now[0] = 41.
        yield

    adapter = ObservedSafeMovePlanner(provider=SimpleNamespace(
        capture_aligned_rgbd=lambda **kwargs: capture(kwargs["role"]), max_inter_camera_skew_ms=100),
        current_station_revision=lambda: revision[0], reference_frame_id="world",
        self_filter=lambda **_: pytest.fail("Changed scene must not enter cloud processing"),
        config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=lambda _: pytest.fail("No planning after scene changes"),
        clock=lambda: now[0], non_motion_read_window=window)
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_observation_expired"):
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, require_grasp_approach=False)



@pytest.mark.parametrize("failure", [None, "preparation", "planning"])
def test_safe_motion_budget_includes_capture_and_scene_preparation(monkeypatch, failure):
    elapsed, budgets = [0.0], []
    monkeypatch.setattr(
        "agent_gateway.profiles.real.motion_observed_scene.time.monotonic",
        lambda: elapsed[0],
    )

    def read(**_kwargs):
        elapsed[0] += 11.0 if failure == "preparation" else 4.0
        return capture()

    def factory(_cloud):
        elapsed[0] += 1.0

        def plan(**kwargs):
            budgets.append(kwargs["time_budget_sec"])
            elapsed[0] += 6.0 if failure == "planning" else 2.0
            return "trajectory"

        return plan

    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=read, max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=self_filter, config=SafeMotionObservationConfig(roles=("main",)),
        planner_factory=factory, clock=lambda: 10.0 + elapsed[0],
    )
    kwargs = {"qpos_by_side": {s: np.zeros(6) for s in ("left", "right")},
        "require_grasp_approach": False, "time_budget_sec": 10.0}
    if failure is None:
        assert adapter(**kwargs) == "trajectory"
    else:
        with pytest.raises(RM65GraspTrajectoryPlanningError) as error:
            adapter(**kwargs)
        assert error.value.reason_code == "path_planning_time_budget_exhausted"
    assert budgets == ([] if failure == "preparation" else [5.0])



@pytest.mark.parametrize("limit", ["age", "budget"])
def test_normal_result_record_cannot_extend_scene_or_operation_admission(tmp_path, monkeypatch, limit):
    from agent_gateway.profiles.real import motion_observed_scene as module

    elapsed, planned = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed[0])
    original_write = module._save_json

    def write(path, payload):
        original_write(path, payload)
        if path.name == "result.json" and payload.get("status") == "path_found":
            elapsed[0] = .2

    monkeypatch.setattr(module, "_save_json", write)
    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=lambda **kw: capture(kw["role"]),
                                 max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=self_filter,
        config=SafeMotionObservationConfig(roles=("main",),
            maximum_scene_age_sec=.1 if limit == "age" else 30.),
        planner_factory=lambda cloud: lambda **kw: planned.append(True) or "trajectory",
        clock=lambda: 10. + elapsed[0], observation_record_dir=tmp_path)
    with pytest.raises(RM65GraspTrajectoryPlanningError) as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False, time_budget_sec=.1 if limit == "budget" else 1.)
    assert caught.value.reason_code == (
        "safe_motion_observation_expired" if limit == "age" else "path_planning_time_budget_exhausted")
    assert planned == [True]
    result = json.loads((next(tmp_path.iterdir()) / "result.json").read_text())
    assert result["status"] == "failed"


def test_window_entry_failure_retains_same_partial_evidence_in_both_formats(tmp_path):
    reads = []

    @contextmanager
    def window():
        raise RuntimeError("HOLD acknowledgement unavailable")
        yield

    def read(**kw):
        reads.append(kw["role"])
        return capture(kw["role"])

    (tmp_path / "debug").mkdir()
    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=read, max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=lambda **kw: pytest.fail("No cloud after failed HOLD window"),
        config=SafeMotionObservationConfig(),
        planner_factory=lambda cloud: pytest.fail("No planning after failed HOLD window"),
        clock=lambda: 10., non_motion_read_window=window,
        observation_record_dir=tmp_path / "normal", debug_record_dir=tmp_path / "debug")
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="safe_motion_observation_unavailable"):
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False)
    assert reads == list(adapter.config.roles)
    normal = next((tmp_path / "normal").iterdir())
    debug = next((tmp_path / "debug").iterdir())
    snapshot = json.loads((normal / "snapshot.json").read_text())
    assert [v["role"] for v in snapshot["views"]] == reads
    assert json.loads((normal / "result.json").read_text())["status"] == "failed"
    record = json.loads((debug / "record.json").read_text())
    assert list(record["views"]) == reads
    assert record["state"] == "rejected"


@pytest.mark.parametrize("exit_problem", ["age", "budget", "error"])
def test_non_motion_window_exit_is_included_in_final_admission_and_failure_record(
        tmp_path, monkeypatch, exit_problem):
    from agent_gateway.profiles.real import motion_observed_scene as module

    elapsed, planned = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed[0])

    @contextmanager
    def window():
        yield
        if exit_problem == "error":
            raise RuntimeError("measurement resume failed")
        elapsed[0] = .2

    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=lambda **kw: capture(kw["role"]),
                                 max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=self_filter,
        config=SafeMotionObservationConfig(roles=("main",),
            maximum_scene_age_sec=.1 if exit_problem == "age" else 30.),
        planner_factory=lambda cloud: lambda **kw: planned.append(True) or "trajectory",
        clock=lambda: 10. + elapsed[0], observation_record_dir=tmp_path,
        non_motion_read_window=window)
    with pytest.raises(RuntimeError) as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False,
                time_budget_sec=.1 if exit_problem == "budget" else 1.)
    if exit_problem != "error":
        assert caught.value.reason_code == (
            "safe_motion_observation_expired" if exit_problem == "age"
            else "path_planning_time_budget_exhausted")
    else:
        assert str(caught.value) == "measurement resume failed"
    assert planned == [True]
    result = json.loads((next(tmp_path.iterdir()) / "result.json").read_text())
    assert result["status"] == "failed"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("secondary", ["window", "record"])
def test_capture_interrupt_survives_window_entry_or_input_record_failure(
        tmp_path, monkeypatch, interrupt, secondary):
    reads = []

    @contextmanager
    def window():
        if secondary == "window":
            raise RuntimeError("HOLD window unavailable")
        yield

    def read(**kw):
        reads.append(kw["role"])
        if kw["role"] == "wrist_left":
            raise interrupt("capture cancelled")
        return capture(kw["role"])

    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=read, max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=lambda **kw: pytest.fail("No filtering after interrupted capture"),
        config=SafeMotionObservationConfig(),
        planner_factory=lambda cloud: pytest.fail("No planning after interrupted capture"),
        clock=lambda: 10., observation_record_dir=tmp_path, non_motion_read_window=window)
    if secondary == "record":
        def fail_record(*args):
            raise OSError("input record unavailable")
        monkeypatch.setattr(adapter, "_save_inputs", fail_record)
    with pytest.raises(interrupt, match="capture cancelled") as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False)
    assert caught.value.__cause__ is not None
    assert reads == ["main", "wrist_left"]
    result = json.loads((next(tmp_path.iterdir()) / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["exception_type"] == interrupt.__name__


def test_window_entry_interrupt_survives_failed_input_record(tmp_path, monkeypatch):
    @contextmanager
    def window():
        raise KeyboardInterrupt("interrupted while waiting for HOLD")
        yield

    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=lambda **kw: capture(kw["role"]),
                                 max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=lambda **kw: pytest.fail("No filtering after window interrupt"),
        config=SafeMotionObservationConfig(),
        planner_factory=lambda cloud: pytest.fail("No planning after window interrupt"),
        clock=lambda: 10., observation_record_dir=tmp_path, non_motion_read_window=window)

    def fail_inputs(*args):
        raise OSError("input record unavailable")

    monkeypatch.setattr(adapter, "_save_inputs", fail_inputs)
    with pytest.raises(KeyboardInterrupt, match="interrupted while waiting for HOLD") as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False)
    assert isinstance(caught.value.__cause__, OSError)
    result = json.loads((next(tmp_path.iterdir()) / "result.json").read_text())
    assert result["exception_type"] == "KeyboardInterrupt"


def test_planning_interrupt_survives_debug_timing_record_failure(tmp_path, monkeypatch):
    adapter, kwargs, _ = debug_adapter(tmp_path, problem="interrupt")
    original_timing = SafeMotionDebugRecord.timing

    def timing(record, entry):
        if entry["phase"] == "path_planning":
            raise RM65GraspTrajectoryPlanningError("safe_motion_debug_record_failed",
                                                   details={"stage": "timing"})
        original_timing(record, entry)

    monkeypatch.setattr(SafeMotionDebugRecord, "timing", timing)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapter(**kwargs)
    assert caught.value.__cause__.reason_code == "safe_motion_debug_record_failed"
    record = json.loads(next(tmp_path.glob("scene-*/record.json")).read_text())
    assert record["state"] == "interrupted"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_capture_error_then_window_interrupt_survives_input_record_failure(
        tmp_path, monkeypatch, interrupt):
    @contextmanager
    def window():
        raise interrupt("cancelled while waiting for HOLD")
        yield

    def read(**kw):
        raise RuntimeError("camera unavailable")

    adapter = ObservedSafeMovePlanner(
        provider=SimpleNamespace(capture_aligned_rgbd=read, max_inter_camera_skew_ms=100),
        current_station_revision=lambda: 7, reference_frame_id="world",
        self_filter=lambda **kw: pytest.fail("No filtering after capture or window failure"),
        config=SafeMotionObservationConfig(),
        planner_factory=lambda cloud: pytest.fail("No planning after capture or window failure"),
        clock=lambda: 10., observation_record_dir=tmp_path, non_motion_read_window=window)

    def fail_inputs(*args):
        raise OSError("input record unavailable")

    monkeypatch.setattr(adapter, "_save_inputs", fail_inputs)
    with pytest.raises(interrupt, match="cancelled while waiting for HOLD") as caught:
        adapter(qpos_by_side={s: np.zeros(6) for s in ("left", "right")},
                require_grasp_approach=False)
    assert isinstance(caught.value.__cause__, OSError)
    result = json.loads((next(tmp_path.iterdir()) / "result.json").read_text())
    assert result["exception_type"] == interrupt.__name__
