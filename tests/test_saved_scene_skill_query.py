"""Read-only query/reference contracts; native geometry is exercised separately."""

import json
import hashlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from scripts.probes.query_safe_move_snapshot import SavedSceneSkillQuery, file_sha256, verify_inputs, capture_inputs
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError

pytestmark = pytest.mark.unit


class Planner:
    def __init__(self):
        self.calls = []
        self.error = None

    def plan_with_observed_cloud(self, cloud, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        self.target = kwargs["target_tool_tcp_world"]
        # Try to mutate caller input. Subsequent queries must still use q0.
        kwargs["qpos_by_side"]["right"][0] = 2.
        return SimpleNamespace(time_from_start_sec=np.array([0., 1.]),
            qpos_by_side={s: np.zeros((2, 6)) for s in ("left", "right")},
            semantic_content_hash="trajectory-hash", planner_evidence_refs=("planner-ref",))

    def tool_tcp_poses_for_qpos_pair(self, **kwargs):
        return {s: self.target for s in ("left", "right")}


@pytest.fixture
def session():
    planner = Planner()
    pose = [0., 0., .3, 0., 0., np.sqrt(.5), np.sqrt(.5)]
    query = SavedSceneSkillQuery(scene_id="a" * 64, reference_frame_id="station",
        qpos_by_side={s: [0.] * 6 for s in ("left", "right")},
        tcp_pose_by_side={s: pose for s in ("left", "right")},
        anchors=[{"reference": "point-a", "label": "blue cube", "kind": "saved_visible_surface_point",
                  "world_xyz_m": [.1, .2, .3], "acquired_at": 17., "role": "main",
                  "mask_sha256": "private-audit"}],
        planner=planner, cloud=object(), source_evidence={"binding_hash": "private-audit"})
    yield query, planner
    query.close()


def test_relation_reuses_original_tool_axes_and_exact_measured_start(session):
    query, planner = session
    result = query.plan_relation(side="right", target_reference="point-a",
        offset_xyz_m=[.03, 0., 0.], offset_frame="tool")
    assert result["status"] == "path_found"
    assert "trajectory_hash" not in result
    assert query.audit()["queries"][0]["trajectory_hash"] == "trajectory-hash"
    np.testing.assert_allclose(planner.target[:3, 3], [.1, .23, .3])
    original = query.audit()["qpos_by_side"]
    query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .03])
    assert original == query.audit()["qpos_by_side"] == {s: [0.] * 6 for s in ("left", "right")}
    assert all(not call["require_grasp_approach"] for call in planner.calls)
    assert all(call["pose_goal_policy"] == "within_arrival_tolerance" for call in planner.calls)


@pytest.mark.parametrize("side,ref,offset", [
    ("both", "point-a", [0., 0., .03]), ("right", "other-scene", [0., 0., .03]),
    ("right", "point-a", [False, 0., .03]), ("right", "point-a", [float("nan"), 0., .03]),
])
def test_invalid_relation_does_not_call_planner(session, side, ref, offset):
    query, planner = session
    with pytest.raises(ValueError):
        query.plan_relation(side=side, target_reference=ref, offset_xyz_m=offset)
    assert not planner.calls


def test_model_view_keeps_references_time_and_frame_without_full_audit(session):
    query, _ = session
    visible = query.read_scene()
    assert visible["references"][0]["reference"] == "point-a"
    assert visible["references"][0]["acquired_at"] == 17.
    assert visible["reference_frame_id"] == "station"
    assert "private-audit" not in json.dumps(visible)
    assert "world_xyz_m" not in visible["references"][0]
    assert query.audit()["references"][0]["mask_sha256"] == "private-audit"
    visible["references"][0]["label"] = "changed"
    assert query.read_scene()["references"][0]["label"] == "blue cube"
    atomic = query.read_scene(include_geometry=True)
    assert atomic["references"][0]["world_xyz_m"] == [.1, .2, .3]


def test_native_rejection_is_not_infrastructure_failure_or_motion_success(session):
    query, planner = session
    planner.error = RM65GraspTrajectoryPlanningError("path_endpoint_collision_blocked", details={"pair": "table"})
    result = query.plan_relation(side="left", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    assert result["status"] == "path_rejected"
    assert result["reason"] == "path_endpoint_collision_blocked"
    assert result["physical_action_occurred"] is False
    assert "table" in query.audit()["queries"][0]["failure_details"]
    planner.error = OSError("native query unavailable")
    with pytest.raises(OSError, match="unavailable"):
        query.plan_relation(side="left", target_reference="point-a", offset_xyz_m=[0., 0., .08])


def test_atomic_and_reference_targets_use_identical_lower_port(session):
    query, planner = session
    query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    target = planner.target.copy()
    query.plan_pose(side="right", pose_xyz_quat_xyzw=[.1, .2, .38, 0., 0., np.sqrt(.5), np.sqrt(.5)])
    np.testing.assert_array_equal(target, planner.target)


def test_close_and_parallel_query_cannot_claim_completion_while_query_active(session):
    query, planner = session
    entered, release = threading.Event(), threading.Event()
    original = planner.plan_with_observed_cloud

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    planner.plan_with_observed_cloud = blocked
    errors = []

    def run():
        try:
            query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="still running"):
            query.close()
        with pytest.raises(RuntimeError, match="already running"):
            query.plan_pose(side="left", pose_xyz_quat_xyzw=[0., 0., .3, 0., 0., 0., 1.])
        assert query.audit()["closed"] is False
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    query.close()
    with pytest.raises(RuntimeError, match="closed"):
        query.read_scene()


def test_trajectory_export_is_exclusive_and_nonexecutable(session, tmp_path):
    query, _ = session
    result = query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    target = tmp_path / "trajectory.npz"
    query.save_trajectory(result["query_id"], target)
    with np.load(target, allow_pickle=False) as data:
        assert set(data.files) == {"time_from_start_sec", "left_qpos", "right_qpos"}
    with pytest.raises(FileExistsError):
        query.save_trajectory(result["query_id"], target)


@pytest.fixture
def input_manifest(tmp_path):
    names = ["camera_mount_urdf.urdf", "camera_config.json", "camera_registry.json",
             "left_tool_profile.json", "right_tool_profile.json", "envelope_profile.json", "arm_ip_assignment.json"]
    for name in names:
        (tmp_path / name).write_text("test-only")
    (tmp_path / "main.npz").write_bytes(b"frozen")
    (tmp_path / "mask.npz").write_bytes(b"mask")
    (tmp_path / "snapshot.json").write_text(json.dumps({"schema": "agenticrobot.safe_move_inputs.v1",
                                                       "views": [{"file": "main.npz"}]}))
    frozen = {str(p): file_sha256(p) for p in tmp_path.iterdir()}
    return {"schema": "agenticrobot.saved_scene_query.v1", "replay_dir": str(tmp_path),
            "source_single_start": True,
            "context_dir": str(tmp_path), "urdf": str(tmp_path / names[0]), "fixed_files": frozen,
            "anchors": [{"mask_path": str(tmp_path / "mask.npz"),
                         "mask_sha256": frozen[str(tmp_path / "mask.npz")]}]}


def test_manifest_requires_frozen_calibration_frames_and_masks(input_manifest):
    verify_inputs(input_manifest)
    for key in ("camera_registry.json", "main.npz", "mask.npz"):
        broken = json.loads(json.dumps(input_manifest))
        path = next(p for p in broken["fixed_files"] if p.endswith("/" + key))
        del broken["fixed_files"][path]
        with pytest.raises(ValueError):
            verify_inputs(broken)


def test_changed_input_rejected_before_native_owner_construction(input_manifest):
    from pathlib import Path
    Path(input_manifest["context_dir"], "camera_registry.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        SavedSceneSkillQuery.from_manifest(input_manifest)


@pytest.fixture
def capture_fixture():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.full((3, 4), 1.7, dtype=np.float32)
    def content(array, name, extra):
        return canonical_content_hash({"dtype": str(array.dtype), "shape": list(array.shape),
            "byte_sha256": hashlib.sha256(array.tobytes()).hexdigest(), **extra},
            domain="dexteleop.real_aligned_rgbd." + name + "_content.v1")
    capture = SimpleNamespace(rgb=rgb, depth_m=depth, source_binding_hash="source-a",
        rgb_content_hash=content(rgb, "rgb", {}),
        depth_content_hash=content(depth, "depth", {"unit": "meter", "invalid_value": 0.0}),
        capture_qpos_by_side={s: np.zeros(6) for s in ("left", "right")})
    snapshot = {"reference_frame_id": "station", "configuration": {"maximum_depth_m": 2.0},
                "request": {"qpos_by_side": {s: [.001] * 6 for s in ("left", "right")}}}
    return capture, snapshot


def test_original_planning_q0_and_depth_policy_preserved(capture_fixture):
    capture, snapshot = capture_fixture
    config, qpos, source = capture_inputs(snapshot, [capture], reference_frame_id="station", source_binding_hash="source-a")
    assert config.maximum_depth_m == 2.0
    assert source == "saved_planning_request"
    np.testing.assert_array_equal(qpos["right"], np.full(6, .001))


@pytest.mark.parametrize("corruption", ["frame", "binding", "rgb", "depth", "missing_q0"])
def test_cross_file_identity_and_missing_q0_fail_closed(capture_fixture, corruption):
    capture, snapshot = capture_fixture
    if corruption == "frame":snapshot["reference_frame_id"] = "another-frame"
    elif corruption == "binding":capture.source_binding_hash = "source-b"
    elif corruption == "rgb":capture.rgb[0, 0, 0] = 1
    elif corruption == "depth":capture.depth_m[0, 0] = 0.
    elif corruption == "missing_q0":snapshot["request"] = {}
    with pytest.raises(ValueError):
        capture_inputs(snapshot, [capture], reference_frame_id="station", source_binding_hash="source-a")


def test_accepted_operator_cut_uses_explicit_saved_stationary_association(capture_fixture):
    capture, snapshot = capture_fixture
    snapshot.update(request={}, operator_scene_sample_id="accepted", source_capture_status="complete")
    _, qpos, source = capture_inputs(snapshot, [capture], reference_frame_id="station", source_binding_hash="source-a")
    assert source == "operator_sample_stationary_association"
    np.testing.assert_array_equal(qpos["left"], np.zeros(6))


def test_export_participates_in_close_fence(session, tmp_path, monkeypatch):
    query, _ = session
    result = query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    entered, release = threading.Event(), threading.Event()
    original = np.savez_compressed
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "savez_compressed", blocked)
    errors = []
    def run():
        try:query.save_trajectory(result["query_id"], tmp_path / "path.npz")
        except BaseException as error:errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="still running"):query.close()
    finally:
        release.set(); worker.join(5)
    assert not errors and not worker.is_alive()


@pytest.mark.parametrize("corruption", [None, "qpos", "time", "camera", "context", "invented_request"])
def test_operator_export_roundtrip_uses_original_codec_and_measurements(input_manifest, corruption):
    from pathlib import Path
    from dataset_contracts.grasp_multiview_snapshot import encode_grasp_multiview_arrays, GRASP_MULTIVIEW_SCHEMA_ID
    from dataset_contracts.grasp_rgbd_snapshot import GRASP_RGBD_SNAPSHOT_SCHEMA_ID
    root = Path(input_manifest["replay_dir"])
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    measurement = {"measured_at": 1., "available_at": 1.01, "source_evidence_refs": ["qpos-cut"]}
    metadata = {"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": "main", "acquired_at": 1.,
        "rgb_content_hash": "rgb", "depth_content_hash": "depth", "source_binding_hash": "binding",
        "station_revision": 0, "source_generation": 1, "frame_index": 0,
        "camera_intrinsics": {"fx": 100., "fy": 100., "cx": 1., "cy": 1.},
        "T_source_from_camera": np.eye(4).tolist(), "measured_qpos": measurement}
    payload = encode_grasp_multiview_arrays(views=[dict(rgb=rgb, depth_m=depth,
        target_mask=np.ones((3, 4), dtype=bool), metadata=metadata,
        qpos_by_side={s: np.zeros(6) for s in ("left", "right")})],
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    archive = root / "artifact-0.zip"; archive.write_bytes(payload)
    np.savez_compressed(root / "main.npz", rgb=rgb, depth_m=depth)
    view = {k: metadata[k] for k in ("role", "acquired_at", "rgb_content_hash", "depth_content_hash",
                                   "source_binding_hash", "station_revision", "source_generation", "frame_index")}
    view.update(file="main.npz", intrinsics=metadata["camera_intrinsics"],
        T_world_from_camera=np.eye(4).tolist(), capture_qpos_by_side={s: [0.] * 6 for s in ("left", "right")},
        pose_measured_at=1., pose_available_at=1.01, pose_evidence_refs=["qpos-cut"])
    snapshot = {"schema": "agenticrobot.safe_move_inputs.v1", "operator_scene_sample_id": "record-a",
                "source_capture_status": "complete", "request": {}, "views": [view]}
    context = [{"file": "context/" + Path(name).name, "sha256": sha}
               for name, sha in input_manifest["fixed_files"].items()
               if name.endswith("camera_registry.json")]
    record = {"schema": "rpent.operator_scene_sample.v1", "sample_id": "record-a", "status": "complete", "context_complete": True,
        "context": context, "artifacts": [{"file": archive.name, "sha256": file_sha256(archive)}],
        "observation_artifact_index": 0}
    if corruption == "qpos":view["capture_qpos_by_side"]["right"][0] = .01
    elif corruption == "time":view["pose_measured_at"] = 2.
    elif corruption == "camera":view["T_world_from_camera"][0][3] = .1
    elif corruption == "context":context[0]["sha256"] = "wrong"
    elif corruption == "invented_request":snapshot["request"] = {"qpos_by_side": view["capture_qpos_by_side"]}
    (root / "snapshot.json").write_text(json.dumps(snapshot))
    (root / "record.json").write_text(json.dumps(record))
    input_manifest.update(source_record=str(root / "record.json"))
    # Even recomputing the experiment's file freeze cannot forge original q0.
    input_manifest["fixed_files"] = {str(p): file_sha256(p) for p in root.iterdir() if p.is_file()}
    if corruption is None:
        verify_inputs(input_manifest)
    else:
        with pytest.raises(ValueError):verify_inputs(input_manifest)


def region_parameters():
    return dict(side="right", target_reference="point-a",
                offset_bounds_xyz_m=[[-.02, .02], [-.03, .03], [.08, .12]])


def test_region_search_reuses_shared_samples_and_original_q0(session):
    from robot_motion.spatial_goal_regions import generate_relation_region_goals
    query, planner = session
    original = planner.plan_with_observed_cloud
    errors = iter([RM65GraspTrajectoryPlanningError("endpoint_collision_blocked"),
                   RM65GraspTrajectoryPlanningError("ik_search_exhausted"), None])
    def plan(*args, **kwargs):
        planner.error = next(errors)
        return original(*args, **kwargs)
    planner.plan_with_observed_cloud = plan
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_found"
    assert result["selected_candidate_index"] == 2
    assert result["attempted_candidate_count"] == 3
    assert len(planner.calls) == 3
    candidates = generate_relation_region_goals(
        current_tcp_pose_xyz_quat_xyzw=query.read_scene(include_geometry=True)["measured_tcp_pose_by_side"]["right"],
        anchor_xyz_m=[.1, .2, .3], offset_bounds_xyz_m=region_parameters()["offset_bounds_xyz_m"],
        offset_frame="reference", orientation_mode="preserve", tool_axis=None,
        axis_reference_xyz=None,
        tool_axis_rotation_bounds_rad=[0., 0.], max_candidates=8)
    np.testing.assert_allclose(planner.target[:3, 3], candidates[2].goal.pose_xyz_quat_xyzw[:3])
    assert query.audit()["qpos_by_side"] == {s: [0.] * 6 for s in ("left", "right")}
    assert all(c["time_budget_sec"] <= 20. for c in planner.calls)


@pytest.mark.parametrize("reason", ["path_start_collision_blocked", "ik_solver_error",
                                    "path_collision_check_unavailable", "stationary_arm_changed"])
def test_region_does_not_retry_start_or_native_faults(session, reason):
    query, planner = session
    planner.error = RM65GraspTrajectoryPlanningError(reason)
    result = query.plan_relation_region(**region_parameters())
    assert result["reason"] == reason and result["attempted_candidate_count"] == 1
    assert len(planner.calls) == 1


def test_region_native_error_remains_an_error_and_unlocks(session):
    query, planner = session
    planner.error = OSError("native oracle failed")
    with pytest.raises(OSError):query.plan_relation_region(**region_parameters())
    assert query.audit()["region_queries"][0]["status"] == "query_failed"
    query.close()


def test_atomic_and_region_share_application_call_budget(session):
    query, planner = session
    query._query_budget = {"maximum_calls": 2, "maximum_planning_sec": 20.}
    query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    planner.error = RM65GraspTrajectoryPlanningError("endpoint_collision_blocked")
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "query_incomplete" and result["reason"] == "query_budget_exhausted"
    assert result["attempted_candidate_count"] == 1 and len(planner.calls) == 2
    assert query.audit()["planning_calls"] == 2


def test_late_path_is_not_accepted_and_budget_is_charged(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2.}
    original = planner.plan_with_observed_cloud
    def slow(*args, **kwargs):
        now[0] += 3.
        return original(*args, **kwargs)
    planner.plan_with_observed_cloud = slow
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "query_incomplete"
    assert len(planner.calls) == 1
    assert query.audit()["planning_seconds"] == 3.
    assert query.audit()["queries"][0]["path_planning_wall_sec"] == 3.


def test_endpoint_fk_is_inside_the_same_query_budget(session, tmp_path):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2.}
    original = planner.tool_tcp_poses_for_qpos_pair
    def slow_fk(**kwargs):
        now[0] += 3.
        return original(**kwargs)
    planner.tool_tcp_poses_for_qpos_pair = slow_fk
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "query_incomplete" and result["reason"] == "query_budget_exhausted"
    audit = query.audit()
    assert audit["planning_calls"] == 1 and audit["planning_seconds"] == 3.
    assert audit["queries"][0]["outcome_before_budget_check"]["status"] == "path_found"
    with pytest.raises(KeyError):query.save_trajectory(audit["queries"][0]["query_id"], tmp_path / "late.npz")


def test_region_checks_achieved_endpoint_not_just_requested_pose(session):
    query, planner = session
    original = planner.tool_tcp_poses_for_qpos_pair
    def displaced(**kwargs):
        poses = original(**kwargs)
        moved = {s: p.copy() for s, p in poses.items()}
        moved["right"][2, 3] += .001
        return moved
    planner.tool_tcp_poses_for_qpos_pair = displaced
    result = query.plan_relation_region(side="right", target_reference="point-a",
        offset_bounds_xyz_m=[[0., 0.], [0., 0.], [.08, .08]])
    assert result["status"] == "path_rejected" and result["reason"] == "region_no_candidate_path"
    audit = query.audit()
    assert audit["queries"][0]["status"] == "path_found"
    assert audit["region_queries"][0]["candidate_validations"][0]["inside_requested_region"] is False


def test_region_invalid_bounds_are_rejected_before_planning(session):
    query, planner = session
    with pytest.raises(ValueError):
        query.plan_relation_region(**{**region_parameters(), "offset_bounds_xyz_m": [[1., -1.], [0., 0.], [.1, .2]]})
    assert not planner.calls
