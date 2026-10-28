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


@pytest.mark.parametrize("value", (None, 1, "true"))
def test_clearance_mode_rejects_nonboolean_before_loading_inputs(monkeypatch, value):
    from scripts.probes import query_safe_move_snapshot as module
    monkeypatch.setattr(module, "verify_inputs", lambda _: pytest.fail("Invalid mode read inputs"))
    with pytest.raises(ValueError, match="enable_inherited_start_clearance"):
        SavedSceneSkillQuery.from_manifest({}, enable_inherited_start_clearance=value)


@pytest.mark.parametrize("flags, expected", (
    ((), True), (("--enable-inherited-start-clearance",), True), (("--strict-start-clearance",), False)))
def test_cli_defaults_to_inherited_clearance_with_explicit_strict_opt_out(tmp_path, monkeypatch, flags, expected):
    import sys
    from scripts.probes import query_safe_move_snapshot as module
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    seen = []

    class Stop(Exception):
        pass

    def factory(cls, value, **kwargs):
        seen.append((value, kwargs))
        raise Stop
    monkeypatch.setattr(module.SavedSceneSkillQuery, "from_manifest", classmethod(factory))
    monkeypatch.setattr(sys, "argv", ["query", "--manifest", str(manifest), *flags])
    with pytest.raises(Stop):
        module.main()
    assert seen == [({}, {"enable_inherited_start_clearance": expected,
                          "endpoint_candidate_ordering": "joint_distance"})]


def test_cli_rejects_conflicting_start_clearance_flags(tmp_path, monkeypatch):
    import sys
    from scripts.probes import query_safe_move_snapshot as module
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(module.SavedSceneSkillQuery, "from_manifest",
                        classmethod(lambda *_a, **_k: pytest.fail("Conflicting flags reached the factory")))
    monkeypatch.setattr(sys, "argv", ["query", "--manifest", str(manifest),
                                      "--enable-inherited-start-clearance", "--strict-start-clearance"])
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2


def test_factory_defaults_to_inherited_clearance_and_binds_strict_opt_out(input_manifest, capture_fixture, monkeypatch):
    from pathlib import Path
    from scripts.probes import query_safe_move_snapshot as module
    capture, snapshot = capture_fixture
    capture.pose_measured_at = 1.
    snapshot.update(schema="agenticrobot.safe_move_inputs.v1", views=[{"file": "main.npz"}])
    snapshot_path = Path(input_manifest["replay_dir"]) / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    input_manifest["fixed_files"][str(snapshot_path)] = file_sha256(snapshot_path)
    input_manifest.update(anchors=[], source_single_start=False)
    profile = SimpleNamespace(station_world_frame_id="station",
        tool_tcp_pose_flange_m_rad_by_side={"left": np.array([0., 0., .14, 0., 0., 0.]),
                                            "right": np.array([0., 0., .15, 0., 0., 0.])})
    builder = SimpleNamespace(pre_ik_target_safety_filter=lambda value: value,
        tool_tcp_world_targets_for_qpos_pair=lambda **_: {s: np.eye(4) for s in ("left", "right")})
    monkeypatch.setattr(module, "resolve_inputs", lambda *_a, **_k:
        (profile, builder, SimpleNamespace(semantic_content_hash="source-a"), object(), object()))
    monkeypatch.setattr(module, "bind_exclusive_rm65_skill_target_safety", lambda **_: None)
    monkeypatch.setattr(module, "load_captures", lambda *_: [capture])
    monkeypatch.setattr(module, "obstacle_cloud_from_captures", lambda **_: SimpleNamespace(points_m=np.zeros((1, 3))))
    configs, identities = [], []
    def create_planner(**kwargs):
        configs.append(kwargs["config"])
        return Planner()
    monkeypatch.setattr(module, "RM65GraspJointTrajectoryPlanner", create_planner)
    for kwargs, enabled in (({}, True), ({"enable_inherited_start_clearance": True}, True),
                            ({"enable_inherited_start_clearance": False}, False)):
        query = SavedSceneSkillQuery.from_manifest(input_manifest, **kwargs)
        audit = query.audit()
        identities.append(audit["scene_id"])
        assert audit["source"]["planning_policy"] == {
            "inherited_start_clearance": enabled, "ordinary_clearance_m": .02,
            "endpoint_candidate_ordering": "joint_distance"}
        assert configs[-1].enable_inherited_start_clearance is enabled
        assert audit["dispatches"] == 0 and audit["queries"] == []
        region = query.plan_relation_region(
            side="right", target_reference=query.read_scene()["references"][0]["reference"],
            offset_bounds_xyz_m=[[-.02, .02], [-.03, .03], [.08, .12]])
        assert region["status"] == "path_found" and region["candidate_policy"] == "proximity"
        np.testing.assert_allclose(region["selected_offset_xyz_m"], [0., 0., .09])
        query.close()
    # The strict opt-out keeps the identity of earlier strict-default queries.
    assert identities[0] == identities[1] and identities[2] != identities[0]

    # Only an arm whose TCP matches the reviewed gripper alignment gets the entity.
    query = SavedSceneSkillQuery.from_manifest(input_manifest)
    assert set(query.audit()["source"]["tool_entities"]) == {"left"}
    with pytest.raises(ValueError, match="tool entity"):
        query.plan_entity_hover_region(side="right",
            target_reference=query.read_scene()["references"][0]["reference"],
            tip_offset_bounds_xy_m=[[-.01, .01], [-.01, .01]], clearance_bounds_m=[.03, .05])
    query.close()

    # An input manifest still cannot select the planning policy either way.
    input_manifest["enable_inherited_start_clearance"] = False
    query = SavedSceneSkillQuery.from_manifest(input_manifest)
    assert query.audit()["source"]["planning_policy"]["inherited_start_clearance"] is True
    assert query.audit()["scene_id"] == identities[0]
    assert configs[-1].enable_inherited_start_clearance is True
    query.close()
    del input_manifest["enable_inherited_start_clearance"]

    # Endpoint ordering is an explicit reviewed planner mode, bound into identity.
    query = SavedSceneSkillQuery.from_manifest(input_manifest,
                                               endpoint_candidate_ordering="estimated_execution_time")
    assert configs[-1].endpoint_candidate_ordering == "estimated_execution_time"
    assert query.audit()["source"]["planning_policy"]["endpoint_candidate_ordering"] == "estimated_execution_time"
    assert query.audit()["scene_id"] != identities[0]
    query.close()
    with pytest.raises(ValueError, match="endpoint_candidate_ordering"):
        SavedSceneSkillQuery.from_manifest(input_manifest, endpoint_candidate_ordering="fastest")

    input_manifest["region_candidate_policy"] = "halton"
    query = SavedSceneSkillQuery.from_manifest(input_manifest)
    region = query.plan_relation_region(
        side="right", target_reference=query.read_scene()["references"][0]["reference"],
        offset_bounds_xyz_m=[[-.02, .02], [-.03, .03], [.08, .12]])
    assert region["status"] == "path_found" and region["candidate_policy"] == "halton"
    np.testing.assert_allclose(region["selected_offset_xyz_m"], [0., 0., .1])
    query.close()


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
    # Keep the original Halton-order assertions when opting into that policy.
    query._region_candidate_policy = "halton"
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


def test_proximity_policy_keeps_q0_and_checks_achieved_region(session):
    query, planner = session
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_found"
    record = query.audit()["region_queries"][0]
    assert record["candidate_policy"] == "proximity"
    assert record["candidate_validations"][0]["inside_requested_region"] is True
    assert query.audit()["qpos_by_side"] == {s: [0.] * 6 for s in ("left", "right")}
    assert planner.calls[0]["time_budget_sec"] <= 3.


def test_one_candidate_timeout_does_not_starve_the_whole_region(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 20., "maximum_call_planning_sec": 3.}
    original = planner.plan_with_observed_cloud
    calls = []
    def plan(*args, **kwargs):
        calls.append(kwargs["time_budget_sec"])
        if len(calls) == 1:
            now[0] += 3.
            raise RM65GraspTrajectoryPlanningError("path_planning_time_budget_exhausted")
        now[0] += .1
        return original(*args, **kwargs)
    planner.plan_with_observed_cloud = plan
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_found" and result["selected_candidate_index"] == 1
    assert result["unresolved_candidate_count"] == 1 and result["attempted_candidate_count"] == 2
    assert calls == [3., 3.]


def test_region_deadline_also_covers_last_out_of_region_validation(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    original_plan = query._plan_pose_locked
    def delayed_validation(**kwargs):
        result = original_plan(**kwargs)
        now[0] = 21.
        return result
    query._plan_pose_locked = delayed_validation
    original = planner.tool_tcp_poses_for_qpos_pair
    def outside(**kwargs):
        poses = {s: p.copy() for s, p in original(**kwargs).items()}
        poses["right"][2, 3] += .001
        return poses
    planner.tool_tcp_poses_for_qpos_pair = outside
    result = query.plan_relation_region(side="right", target_reference="point-a",
        offset_bounds_xyz_m=[[0., 0.], [0., 0.], [.08, .08]])
    assert result["status"] == "query_incomplete"
    assert query.audit()["region_queries"][0]["selected_plan_query_id"] is None
    # The lower point query remains inspectable evidence, never a region grant.
    assert query.audit()["queries"][0]["status"] == "path_found"


def test_region_validation_charges_remaining_session_budget_and_revokes_export(session, tmp_path):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 20., "maximum_call_planning_sec": 3.}
    query._planning_seconds = 19.95
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .04
        return original_planner(*args, **kwargs)
    def delayed(**kwargs):
        result = original_query(**kwargs)
        now[0] += .10
        return result
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert result["status"] == "query_incomplete" and result["reason"] == "query_budget_exhausted"
    assert audit["planning_seconds"] == pytest.approx(20.09)
    assert audit["queries"][0]["path_planning_wall_sec"] == pytest.approx(.04)
    assert audit["queries"][0]["region_candidate_wall_sec"] == pytest.approx(.14)
    assert audit["queries"][0]["region_validation_budget_exceeded"] is True
    assert audit["queries"][0]["status"] == "path_found"
    assert audit["region_queries"][0]["selected_plan_query_id"] is None
    for query_id in (result["query_id"], audit["queries"][0]["query_id"]):
        with pytest.raises(KeyError):
            query.save_trajectory(query_id, tmp_path / "late.npz")


def test_region_call_budget_overrun_retries_without_double_charge(session, tmp_path):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2., "maximum_call_planning_sec": .1}
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .04
        return original_planner(*args, **kwargs)
    def delayed(**kwargs):
        result = original_query(**kwargs)
        if len(planner.calls) == 1:
            now[0] += .1
        return result
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert result["status"] == "path_found" and result["selected_candidate_index"] == 1
    assert result["attempted_candidate_count"] == 2 and result["unresolved_candidate_count"] == 1
    assert audit["planning_seconds"] == pytest.approx(.18)
    assert audit["queries"][0]["region_validation_budget_exceeded"] is True
    with pytest.raises(KeyError):
        query.save_trajectory(audit["queries"][0]["query_id"], tmp_path / "late.npz")
    query.save_trajectory(result["query_id"], tmp_path / "accepted.npz")


def test_region_generation_is_charged_once_before_native_planning(session, monkeypatch):
    from scripts.probes import query_safe_move_snapshot as module
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": .25}
    original = module.generate_relation_region_goals
    def slow_generation(**kwargs):
        now[0] += .3
        return original(**kwargs)
    monkeypatch.setattr(module, "generate_relation_region_goals", slow_generation)
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "query_incomplete" and not planner.calls
    assert query.audit()["planning_seconds"] == pytest.approx(.3)
    assert query.audit()["region_queries"][0]["candidate_generation_wall_sec"] == pytest.approx(.3)


def test_region_deadline_expiring_before_lower_entry_never_starts_planner(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 30., "maximum_call_planning_sec": 3.}
    original = query._plan_pose_locked
    def interrupted_before_entry(**kwargs):
        now[0] = 21.
        return original(**kwargs)
    query._plan_pose_locked = interrupted_before_entry
    result = query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert result["status"] == "query_incomplete" and not planner.calls
    assert audit["planning_calls"] == 0 and audit["planning_seconds"] == pytest.approx(21.)
    assert audit["queries"][0]["reason"] == "query_budget_exhausted"
    assert audit["region_queries"][0]["attempted_candidate_count"] == 0


def test_region_lower_entry_receives_only_remaining_absolute_time(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 30., "maximum_call_planning_sec": 3.}
    original = query._plan_pose_locked
    def delayed_before_entry(**kwargs):
        now[0] = 2.
        return original(**kwargs)
    query._plan_pose_locked = delayed_before_entry
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_found" and len(planner.calls) == 1
    assert planner.calls[0]["time_budget_sec"] == pytest.approx(1.)
    assert query.audit()["planning_seconds"] == pytest.approx(2.)


def test_atomic_region_atomic_budget_is_charged_once(session, monkeypatch):
    from scripts.probes import query_safe_move_snapshot as module
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2.}
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    original_generate = module.generate_relation_region_goals
    def plan(*args, **kwargs):
        now[0] += .1
        return original_planner(*args, **kwargs)
    def delayed(**kwargs):
        result = original_query(**kwargs)
        if kwargs.get("relation", {}).get("region_query_id"):
            now[0] += .03
        return result
    def generate(**kwargs):
        now[0] += .02
        return original_generate(**kwargs)
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    monkeypatch.setattr(module, "generate_relation_region_goals", generate)
    args = dict(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    assert query.plan_relation(**args)["status"] == "path_found"
    assert query.plan_relation_region(**region_parameters())["status"] == "path_found"
    assert query.plan_relation(**args)["status"] == "path_found"
    assert query.audit()["planning_seconds"] == pytest.approx(.35)
    assert query.audit()["planning_calls"] == 3


def test_region_validation_exception_charges_and_removes_candidate(session, tmp_path):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .04
        return original_planner(*args, **kwargs)
    def broken(**kwargs):
        result = original_query(**kwargs)
        now[0] += .06
        query._queries[-1]["planned_endpoint_xyz_m"] = "invalid endpoint"
        return result
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, broken
    with pytest.raises((TypeError, ValueError)):
        query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert audit["planning_seconds"] == pytest.approx(.1)
    assert audit["region_queries"][0]["status"] == "query_failed"
    assert audit["region_queries"][0]["selected_plan_query_id"] is None
    with pytest.raises(KeyError):
        query.save_trajectory(audit["queries"][0]["query_id"], tmp_path / "broken.npz")
    query.close()


def test_region_interrupted_before_result_assignment_drops_only_new_candidate(session, tmp_path):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    original_planner = planner.plan_with_observed_cloud
    def plan(*args, **kwargs):
        now[0] += .04
        return original_planner(*args, **kwargs)
    planner.plan_with_observed_cloud = plan
    previous = query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    original_query = query._plan_pose_locked
    def interrupted(**kwargs):
        original_query(**kwargs)
        now[0] += .06
        raise RuntimeError("interrupted after lower result")
    query._plan_pose_locked = interrupted
    with pytest.raises(RuntimeError, match="interrupted after lower result"):
        query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert audit["planning_seconds"] == pytest.approx(.14)
    assert audit["region_queries"][0]["status"] == "query_failed"
    assert audit["region_queries"][0]["attempted_candidate_count"] == 1
    assert audit["region_queries"][0]["attempt_query_ids"] == [audit["queries"][-1]["query_id"]]
    query.save_trajectory(previous["query_id"], tmp_path / "previous.npz")
    with pytest.raises(KeyError):
        query.save_trajectory(audit["queries"][-1]["query_id"], tmp_path / "interrupted.npz")


def test_region_lower_finally_failure_removes_unrecorded_trajectory(session, tmp_path):
    query, _ = session
    previous = query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    original_ids, original_rows = set(query._trajectories), len(query._queries)
    failed, orphan_id = [False], [None]
    def clock():
        newly_stored = set(query._trajectories) - original_ids
        if newly_stored and len(query._queries) == original_rows and not failed[0]:
            failed[0] = True
            orphan_id[0] = next(iter(newly_stored))
            raise RuntimeError("synthetic lower finally clock failure")
        return 0.
    query._clock = clock
    with pytest.raises(RuntimeError, match="lower finally clock failure"):
        query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert orphan_id[0] is not None and len(audit["queries"]) == original_rows
    assert audit["region_queries"][0]["status"] == "query_failed"
    assert audit["region_queries"][0]["attempted_candidate_count"] == 1
    assert audit["region_queries"][0]["attempt_query_ids"] == []
    with pytest.raises(KeyError):
        query.save_trajectory(orphan_id[0], tmp_path / "orphan.npz")
    query.save_trajectory(previous["query_id"], tmp_path / "previous.npz")
    query.close()


def test_region_trajectory_snapshot_failure_releases_lock(session):
    query, planner = session
    class SnapshotFailure(dict):
        def __iter__(self):
            raise MemoryError("synthetic trajectory snapshot failure")
    query._trajectories = SnapshotFailure(query._trajectories)
    with pytest.raises(MemoryError, match="snapshot failure"):
        query.plan_relation_region(**region_parameters())
    assert not planner.calls
    assert query._lock.acquire(blocking=False)
    query._lock.release()
    query.close()


@pytest.mark.parametrize("reason", ["path_start_collision_blocked", "path_collision_check_unavailable"])
def test_region_does_not_retry_fatal_fault_after_post_return_delay(session, reason):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2., "maximum_call_planning_sec": .1}
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    planner.error = RM65GraspTrajectoryPlanningError(reason)
    def plan(*args, **kwargs):
        now[0] += .04
        return original_planner(*args, **kwargs)
    def delayed(**kwargs):
        result = original_query(**kwargs)
        now[0] += .1
        return result
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_rejected" and result["reason"] == reason
    assert result["attempted_candidate_count"] == 1 and len(planner.calls) == 1
    assert query.audit()["planning_seconds"] == pytest.approx(.14)


@pytest.mark.parametrize("sequence", ["timeout_then_fatal", "fatal_then_region_deadline"])
def test_region_preserves_fatal_fault_after_unresolved_or_late_delivery(session, sequence):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 30., "maximum_call_planning_sec": .2}
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .1
        planner.error = RM65GraspTrajectoryPlanningError(
            "path_planning_time_budget_exhausted" if sequence == "timeout_then_fatal" and not planner.calls
            else "path_start_collision_blocked")
        return original_planner(*args, **kwargs)
    def delayed(**kwargs):
        result = original_query(**kwargs)
        if sequence == "fatal_then_region_deadline":
            now[0] = 21.
        return result
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "path_rejected" and result["reason"] == "path_start_collision_blocked"
    assert len(planner.calls) == (2 if sequence == "timeout_then_fatal" else 1)
    assert result["unresolved_candidate_count"] == (1 if sequence == "timeout_then_fatal" else 0)


@pytest.mark.parametrize("reason", ["path_start_collision_blocked", "path_collision_check_unavailable"])
def test_region_never_retries_fatal_fault_returned_after_lower_budget(session, reason):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2., "maximum_call_planning_sec": .1}
    original = planner.plan_with_observed_cloud
    planner.error = RM65GraspTrajectoryPlanningError(reason)
    def slow(*args, **kwargs):
        now[0] += .15
        return original(*args, **kwargs)
    planner.plan_with_observed_cloud = slow
    result = query.plan_relation_region(**region_parameters())
    assert result["status"] == "query_incomplete" and result["reason"] == "query_budget_exhausted"
    assert result["terminal_fault_reason"] == reason and len(planner.calls) == 1
    assert query.audit()["queries"][0]["outcome_before_budget_check"] == {
        "status": "path_rejected", "reason": reason}


def test_region_late_final_rejection_respects_session_budget(session):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": .5, "maximum_call_planning_sec": 1.}
    planner.error = RM65GraspTrajectoryPlanningError("endpoint_collision_blocked")
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .4
        return original_planner(*args, **kwargs)
    class DelayedResult(dict):
        reads = 0
        def __getitem__(self, key):
            if key == "status":
                self.reads += 1
                if self.reads == 2:
                    now[0] += .2
            return super().__getitem__(key)
    def delayed(**kwargs):
        return DelayedResult(original_query(**kwargs))
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**{**region_parameters(), "max_candidates": 1})
    assert result["status"] == "query_incomplete"
    assert result["reason"] == "region_candidate_search_incomplete"
    assert len(planner.calls) == 1 and query.audit()["planning_seconds"] == pytest.approx(.6)


@pytest.mark.parametrize("phase", ["after_selection", "final_delivery"])
def test_region_exception_after_selection_revokes_all_new_exports(session, tmp_path, phase):
    query, _ = session
    previous = query.plan_relation(side="right", target_reference="point-a", offset_xyz_m=[0., 0., .08])
    now, selected_reads, failed = [0.], [0], [False]
    def clock():
        if any(key.startswith("region:") for key in query._trajectories):
            selected_reads[0] += 1
            trigger = 1 if phase == "after_selection" else 2
            if selected_reads[0] == trigger and not failed[0]:
                failed[0] = True
                raise RuntimeError("synthetic delivery clock failure")
        return now[0]
    query._clock = clock
    with pytest.raises(RuntimeError, match="delivery clock failure"):
        query.plan_relation_region(**region_parameters())
    audit = query.audit()
    region = audit["region_queries"][0]
    assert region["status"] == "query_failed" and region["selected_plan_query_id"] is None
    assert region["attempted_candidate_count"] == 1
    assert region["attempt_query_ids"] == [audit["queries"][-1]["query_id"]]
    for query_id in (region["query_id"], audit["queries"][-1]["query_id"]):
        with pytest.raises(KeyError):
            query.save_trajectory(query_id, tmp_path / "revoked.npz")
    query.save_trajectory(previous["query_id"], tmp_path / "previous.npz")
    query.close()


@pytest.mark.parametrize("first_outcome", ["retryable", "success"])
def test_region_accounts_for_scheduling_between_candidate_and_decision(session, tmp_path, first_outcome):
    query, planner = session
    now = [0.]
    query._clock = lambda: now[0]
    query._query_budget = {"maximum_calls": 8, "maximum_planning_sec": 2., "maximum_call_planning_sec": 1.}
    if first_outcome == "retryable":
        planner.error = RM65GraspTrajectoryPlanningError("endpoint_collision_blocked")
    original_planner, original_query = planner.plan_with_observed_cloud, query._plan_pose_locked
    def plan(*args, **kwargs):
        now[0] += .1
        return original_planner(*args, **kwargs)
    class DelayedResult(dict):
        reads = 0
        def __getitem__(self, key):
            if key == "status":
                self.reads += 1
                if self.reads == 2:
                    now[0] += 3.
            return super().__getitem__(key)
    def delayed(**kwargs):
        return DelayedResult(original_query(**kwargs))
    planner.plan_with_observed_cloud, query._plan_pose_locked = plan, delayed
    result = query.plan_relation_region(**region_parameters())
    audit = query.audit()
    assert result["status"] == "query_incomplete"
    assert len(planner.calls) == 1 and audit["planning_seconds"] == pytest.approx(3.1)
    assert audit["region_queries"][0]["selected_plan_query_id"] is None
    if first_outcome == "success":
        with pytest.raises(KeyError):
            query.save_trajectory(audit["queries"][0]["query_id"], tmp_path / "late.npz")
        with pytest.raises(KeyError):
            query.save_trajectory(result["query_id"], tmp_path / "late-region.npz")


def _entity_query(planner, **kwargs):
    from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry
    from robot_motion.tool_entity_goals import ToolEntityBox
    entity = ToolEntityBox.from_parallel_jaw_geometry(ParallelJawSolidGeometry(palm_z_m=-.02,
        finger_length_m=.08, finger_thickness_m=.01, pad_height_m=.02, maximum_opening_m=.06),
        source={"test": "box"})
    grid = np.stack(np.meshgrid(np.arange(-.2, .4, .01), np.arange(-.1, .5, .01)), -1).reshape(-1, 2)
    top = np.stack(np.meshgrid(np.arange(.08, .121, .005), np.arange(.18, .221, .005)), -1).reshape(-1, 2)
    points = np.vstack([np.column_stack([grid, np.zeros(len(grid))]),
                        np.column_stack([top, np.full(len(top), .05)])])
    down = [.1, .2, .5, 1., 0., 0., 0.]  # TCP +Z points to reference -Z
    return SavedSceneSkillQuery(scene_id="b" * 64, reference_frame_id="station",
        qpos_by_side={s: [0.] * 6 for s in ("left", "right")},
        tcp_pose_by_side={s: down for s in ("left", "right")},
        anchors=[{"reference": "top-a", "label": "cube top", "kind": "saved_visible_surface_point",
                  "world_xyz_m": [.1, .2, .05], "acquired_at": 3., "role": "main"}],
        planner=planner, cloud=SimpleNamespace(points_m=points, quantization_margin_m=.002),
        source_evidence={}, tool_entity_by_side={"right": entity}, **kwargs)


@pytest.fixture
def entity_session():
    planner = Planner()
    query = _entity_query(planner)
    yield query, planner
    query.close()


HOVER = dict(side="right", target_reference="top-a",
             tip_offset_bounds_xy_m=[[-.02, .02], [-.02, .02]], clearance_bounds_m=[.03, .06])


def test_entity_hover_targets_the_gripper_box_and_checks_achieved_fk(entity_session):
    query, planner = entity_session
    result = query.plan_entity_hover_region(**HOVER)
    assert result["status"] == "path_found" and result["selected_candidate_index"] == 0
    assert result["semantics"] == "agenticrobot.relation.tool_entity_hover.v1"
    assert result["selected_clearance_m"] == pytest.approx(.05)
    np.testing.assert_allclose(result["selected_tip_offset_xy_m"], [0., 0.], atol=1e-12)
    assert not {"tool_entity", "clearance_bounds_m", "tip_offset_bounds_xy_m"} & set(result)
    # Support is the object top plus the voxel margin; the fingertip face hangs 6 cm below TCP.
    np.testing.assert_allclose(planner.target[:3, 3], [.1, .2, .052 + .05 + .06], atol=1e-9)
    np.testing.assert_allclose(planner.target[:3, 2], [0., 0., -1.], atol=1e-9)
    region = query.audit()["region_queries"][0]
    assert region["tool_entity"]["source"] == {"test": "box"}
    assert region["clearance_bounds_m"] == [.03, .06]
    validation = region["candidate_validations"][0]
    assert validation["inside_requested_region"] is True
    assert validation["achieved_clearance_m"] == pytest.approx(.05)
    row = query.audit()["queries"][0]
    assert row["relation"]["entity_hover"]["support_height_m"] == pytest.approx(.052)
    assert len(row["planned_endpoint_pose_xyz_quat_xyzw"]) == 7
    assert all(not call["require_grasp_approach"] for call in planner.calls)
    assert query.audit()["qpos_by_side"] == {s: [0.] * 6 for s in ("left", "right")}


class SinkingPlanner(Planner):
    """Arrives 3 cm lower than requested: inside TCP tolerance, outside the hover."""

    def tool_tcp_poses_for_qpos_pair(self, **kwargs):
        actual = self.target.copy()
        actual[2, 3] -= .03
        return {s: actual for s in ("left", "right")}


def test_entity_hover_rejects_achieved_pose_outside_requested_clearance():
    planner = SinkingPlanner()
    query = _entity_query(planner)
    result = query.plan_entity_hover_region(**HOVER)
    assert result["status"] == "path_rejected" and result["reason"] == "region_no_candidate_path"
    assert len(planner.calls) == result["attempted_candidate_count"] == 8
    validations = query.audit()["region_queries"][0]["candidate_validations"]
    assert len(validations) == 8 and not any(v["inside_requested_region"] for v in validations)
    assert all(v["achieved_clearance_m"] < .03 for v in validations)
    query.close()


def test_entity_hover_requires_entity_policy_reference_and_bounds(entity_session, session):
    query, planner = entity_session
    with pytest.raises(ValueError, match="tool entity"):
        query.plan_entity_hover_region(**{**HOVER, "side": "left"})
    with pytest.raises(ValueError, match="orientation"):
        query.plan_entity_hover_region(**HOVER, orientation_policy="free")
    with pytest.raises(ValueError):
        query.plan_entity_hover_region(**{**HOVER, "target_reference": "point-a"})
    with pytest.raises(ValueError):
        query.plan_entity_hover_region(**{**HOVER, "clearance_bounds_m": [.06, .03]})
    assert not planner.calls
    plain, plain_planner = session
    with pytest.raises(ValueError, match="tool entity"):
        plain.plan_entity_hover_region(**{**HOVER, "target_reference": "point-a"})
    assert not plain_planner.calls


def test_entity_hover_shares_the_session_query_budget():
    planner = Planner()
    query = _entity_query(planner, query_budget={"maximum_calls": 2, "maximum_planning_sec": 20.})
    planner.error = RM65GraspTrajectoryPlanningError("ik_search_exhausted", details={})
    query.plan_pose(side="right", pose_xyz_quat_xyzw=[.1, .2, .3, 1., 0., 0., 0.])
    result = query.plan_entity_hover_region(**HOVER)
    assert len(planner.calls) == 2
    assert result["status"] == "query_incomplete" and result["reason"] == "query_budget_exhausted"
    query.close()


def test_entity_hover_tool_axis_down_turns_a_horizontal_tool():
    from scipy.spatial.transform import Rotation
    planner = Planner()
    query = _entity_query(planner)
    horizontal = Rotation.from_euler("y", 90, degrees=True).as_quat().tolist()
    query._tcp["right"] = (.1, .2, .5, *horizontal)
    result = query.plan_entity_hover_region(**HOVER, orientation_policy="tool_axis_down")
    assert result["status"] == "path_found" and result["orientation_policy"] == "tool_axis_down"
    np.testing.assert_allclose(planner.target[:3, 2], [0., 0., -1.], atol=1e-9)
    query.close()
