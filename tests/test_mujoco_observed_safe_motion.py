"""Sensor-shaped assembly/validation; native acceptance is a separate gate."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco import motion_observed_scene as module
from agent_gateway.profiles.mujoco.motion_deployment import MujocoAgentTcpPoseMotionProfile
from robot_motion.observed_planning import ObservedJointPlanningInput
from robot_motion.observed_scene import SafeMotionObservationConfig
from robot_motion.path_planning import JointConfigurationCheck
from sim.mujoco.deploy.planned_tcp_motion import MujocoObservedJointPlanningInput, MujocoTcpPlanRejected

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _fixture(tmp_path, monkeypatch, *, start_clearance="strict"):
    path = tmp_path / "robot.urdf"
    path.write_text('<robot name="fixture"/>')
    monkeypatch.setattr(module, "build_dual_arm_scene_profile", lambda name: SimpleNamespace(
        urdf_path=path, collision_pairs=(), fixed_scene_box_overrides=()))
    config = module.MujocoObservedSafeMotionConfig(path, "fixture:known-robot",
        SafeMotionObservationConfig(roles=("main",)), start_clearance=start_clearance)
    profile = MujocoAgentTcpPoseMotionProfile(reference_frame_id="canonical_station",
        reference_frame_spec_hash="f"*64, frame_graph_hash="g"*64,
        tool_config_hash="t"*64, calibration_hash="c"*64,
        profile_id="mujoco.agent_safe_tcp_pose.rgbd." + config.bound_profile_hash(
            minimum_clearance_m=.002, planning_clearance_padding_m=.002))
    pair = {s: np.zeros(6) for s in ("left", "right")}
    calls = SimpleNamespace(captures=0, filters=[], clouds=[])
    depth = SimpleNamespace(role="main", source_id="camera:main", captured_at=10.,
        station_revision=3, depth_content_hash="d"*64, capture_group_id="cut:1",
        depth_m=np.full((20, 20), .5, dtype=np.float32),
        intrinsics={"width": 20, "height": 20, "fx": 100., "fy": 100., "cx": 10., "cy": 10.},
        reference_from_camera=np.eye(4))
    view = SimpleNamespace(view_id="camera:main", depth_m=depth.depth_m,
        intrinsics=depth.intrinsics,
        optical_camera_pose_world_xyz_quat_xyzw=(0., 0., 0., 0., 0., 0., 1.))
    cut = SimpleNamespace(captured_at=10., arm_qpos_by_side=pair, views=(view,))
    def capture(**kwargs):
        calls.captures += 1
        assert kwargs["operation_id"] == "op:observed"
        assert kwargs["source_ids"] == ("camera:main",) and callable(kwargs["cancel_requested"])
        return cut
    def self_filter(**kwargs):
        assert not kwargs["target_mask"].any()
        calls.filters.append(kwargs)
        return SimpleNamespace(depth_m=kwargs["depth_m"], evidence={"fixture": True})
    self_filter.prepare_geometry = lambda: None
    def with_cloud(cloud):
        calls.clouds.append(cloud)
        return SimpleNamespace(scene_hash=cloud.semantic_content_hash,
            check=lambda q: JointConfigurationCheck(safe=True))
    monkeypatch.setattr(module, "RM65CapturedRobotObservationFilter", lambda **kw: self_filter)
    monkeypatch.setattr(module, "build_rm65_profile_obstacle_collision_oracle",
        lambda **kwargs: SimpleNamespace(for_observed_cloud=with_cloud))
    rig = SimpleNamespace(capture_scene=lambda f: f(SimpleNamespace(arm_qpos_by_side=lambda: pair)),
        config=SimpleNamespace(realtime_worker=False),
        scene=SimpleNamespace(load_result=SimpleNamespace(conversion=SimpleNamespace(urdf_path=path)),
            scene_spec=SimpleNamespace(collision=SimpleNamespace(safety_minimum_clearance_m=.002))))
    provider = module.MujocoObservedSafeMotionProvider(config=config,
        capture_port=SimpleNamespace(
            source_id_by_role={"main": "camera:main"},
            source_frame_id="fixture:urdf-world", reference_frame_id="canonical_station",
            application=SimpleNamespace(target_frame_profile=SimpleNamespace(frame_graph=SimpleNamespace(
                frame_by_id={"fixture:urdf-world": SimpleNamespace(semantic_content_hash="w"*64)})),
                capture_planning_rgbd_cut=capture,
                transform_tcp_pose_xyzquat=lambda pose, **kw: (
                    (1., 2., 3., 0., 0., 0., 1.) if kw["source_frame_id"] == "canonical_station" else pose))),
        rig=rig, profile=profile,
        current_station_revision=lambda: 3, clock=lambda: 10., planning_clearance_padding_m=.002,
        source_binding_hash="b"*64)
    return SimpleNamespace(provider=provider, pair=pair, config=config, profile=profile,
                           cut=cut, calls=calls)


def test_provider_consumes_one_correlated_sensor_cut_and_keeps_obstacles(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    source = f.provider.capture(operation_id="op:observed", expected_station_revision=3, cancel_requested=lambda: False)
    assert f.calls.captures == len(f.calls.filters) == len(f.calls.clouds) == 1
    assert source.oracle.scene_hash == f.calls.clouds[0].semantic_content_hash
    assert len(f.calls.clouds[0].points_m) > 0
    np.testing.assert_allclose(f.calls.filters[0]["source_from_camera"][:3, 3], [1., 2., 3.])
    assert f.calls.clouds[0].reference_frame_id == "fixture:urdf-world"
    assert f.calls.clouds[0].points_m[:, 2].min() > 3.
    view = f.calls.clouds[0].evidence["views"][0]
    assert view["source_binding_hash"] == "b"*64
    assert view["frame_index"] is None and view["source_generation"] is None
    assert not view["pose_evidence_refs"]
    f.pair["right"][0] = .1
    np.testing.assert_array_equal(source.qpos_by_component["right"], np.zeros(6))
    with pytest.raises(ValueError, match="start_changed"):
        source.require_current(f.pair)
    with pytest.raises(ValueError):
        source.qpos_by_component["right"].setflags(write=True)


@pytest.mark.parametrize("failure", ("cancel", "revision", "cut_identity", "stale", "drift"))
def test_provider_rejects_without_native_or_scene_fallback(tmp_path, monkeypatch, failure):
    f = _fixture(tmp_path, monkeypatch)
    if failure == "revision":
        f.provider.current_station_revision = lambda: 4
    elif failure == "cut_identity":
        # One raw cut owns both image timestamps; reject a mismatched camera
        # instead of constructing two impossible clocks for the same view.
        f.cut.views[0].view_id = "other-camera"
    elif failure == "stale":
        f.provider.clock = lambda: 41.
    elif failure == "drift":
        original = f.provider.capture_port.application.capture_planning_rgbd_cut
        def drift(**kwargs):
            # Return a fresh measurement, without mutating the prior cut.
            result = original(**kwargs)
            f.provider.rig.capture_scene = lambda fun: fun(SimpleNamespace(
                arm_qpos_by_side=lambda: {s: q + .02 for s, q in f.pair.items()}))
            return result
        f.provider.capture_port.application.capture_planning_rgbd_cut = drift
    with pytest.raises(MujocoTcpPlanRejected):
        f.provider.capture(operation_id="op:observed", expected_station_revision=3,
            cancel_requested=lambda: failure == "cancel")


def test_observation_policy_changes_exact_safe_profile_hash_only_when_enabled(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    default = replace(f.profile, profile_id="mujoco.agent_safe_tcp_pose.v1")
    assert default.semantic_content_hash != f.profile.semantic_content_hash
    other = replace(f.config, observation=replace(f.config.observation, voxel_size_m=.005))
    assert other.semantic_content_hash != f.config.semantic_content_hash
    assert replace(f.profile, profile_id="mujoco.agent_safe_tcp_pose.rgbd." + other.bound_profile_hash(
        minimum_clearance_m=.002, planning_clearance_padding_m=.002)).semantic_content_hash != f.profile.semantic_content_hash
    assert other.bound_profile_hash(minimum_clearance_m=.002, planning_clearance_padding_m=0.) != other.bound_profile_hash(
        minimum_clearance_m=.002, planning_clearance_padding_m=.002)


def test_start_clearance_changes_the_safe_profile_identity_only_when_inherited(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    bound = {"minimum_clearance_m": .002, "planning_clearance_padding_m": .002}
    # Captured at ed18f3952 with this fixture, before start_clearance existed.
    assert f.config.start_clearance == "strict"
    assert f.config.semantic_content_hash == "b1de324ce6ec4408937740397a4429033d8c5910d4540a2e6cd027292493fd25"
    assert f.config.bound_profile_hash(**bound) == "58ced87bed5c820c3a2c1d6e44ef5dbee3d271078f53eaccf2190024b6813794"
    inherited = replace(f.config, start_clearance="inherited-positive-v1")
    assert inherited.semantic_content_hash != f.config.semantic_content_hash
    assert inherited.bound_profile_hash(**bound) != f.config.bound_profile_hash(**bound)
    inherited.require_resources_unchanged()
    assert replace(f.profile, profile_id="mujoco.agent_safe_tcp_pose.rgbd." + inherited.bound_profile_hash(
        **bound)).semantic_content_hash != f.profile.semantic_content_hash


@pytest.mark.parametrize("value", ("STRICT", "", "inherited-positive-v2", None, True))
def test_start_clearance_must_be_an_exact_reviewed_mode(tmp_path, monkeypatch, value):
    f = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="start_clearance"):
        replace(f.config, start_clearance=value)


@pytest.mark.parametrize("mode", ("strict", "inherited-positive-v1"))
def test_provider_carries_the_configured_start_clearance_to_the_planner(tmp_path, monkeypatch, mode):
    f = _fixture(tmp_path, monkeypatch, start_clearance=mode)
    source = f.provider.capture(operation_id="op:observed", expected_station_revision=3, cancel_requested=lambda: False)
    assert isinstance(source, MujocoObservedJointPlanningInput) and isinstance(source, ObservedJointPlanningInput)
    assert source.start_clearance == mode
    assert f.provider.semantic_content_hash == f.config.bound_profile_hash(
        minimum_clearance_m=.002, planning_clearance_padding_m=.002)


def test_identity_includes_resolved_urdf_mesh_content_and_collision_pairs(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("first-geometry")
    f.config.robot_urdf_path.write_text('<robot name="fixture"><link name="part"><visual><geometry>'
        '<mesh filename="mesh.obj"/></geometry></visual></link></robot>')
    first = replace(f.config)
    mesh.write_text("other-geometry")
    second = replace(f.config)
    assert first.semantic_content_hash != second.semantic_content_hash
    with pytest.raises(ValueError, match="resources changed"):
        first.require_resources_unchanged()
    second.require_resources_unchanged()
    monkeypatch.setattr(module, "build_dual_arm_scene_profile", lambda name: SimpleNamespace(
        urdf_path=f.config.robot_urdf_path, collision_pairs=(("first", "second"),), fixed_scene_box_overrides=()))
    assert replace(f.config).semantic_content_hash != second.semantic_content_hash


def test_direct_assembly_rejects_realtime_owner_before_any_resources(tmp_path, monkeypatch):
    from agent_gateway.profiles.mujoco.motion_application import MujocoAgentSkillMotionApplication
    from agent_gateway.profiles.mujoco.motion_deployment import MujocoAgentTcpPoseContractBundle
    from agent_gateway.profiles.mujoco.motion_gateway_session import assemble_mujoco_agent_tcp_pose_gateway_session
    f = _fixture(tmp_path, monkeypatch)
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.runtime = SimpleNamespace(rig=SimpleNamespace(config=SimpleNamespace(realtime_worker=True)))
    bundle = object.__new__(MujocoAgentTcpPoseContractBundle)
    with pytest.raises(ValueError, match="deterministic physics owner"):
        assemble_mujoco_agent_tcp_pose_gateway_session(application=application, contract_bundle=bundle,
            allowed_client_ids=("fixture",), safe_motion_observation=f.config)


@pytest.mark.parametrize("field,value", [("maximum_age_sec", True), ("maximum_age_sec", np.bool_(True)),
    ("maximum_age_sec", 0), ("maximum_start_drift_rad", -1), ("acquired_at", float("nan"))])
def test_observed_input_rejects_invalid_admission_bounds(field, value):
    values = dict(oracle=SimpleNamespace(check=lambda q: JointConfigurationCheck(safe=True), scene_hash="sensor"),
        qpos_by_component={s: np.zeros(6) for s in ("left", "right")},
        acquired_at=10., maximum_age_sec=30., maximum_start_drift_rad=.01, clock=lambda: 10.)
    values[field] = value
    with pytest.raises(ValueError):
        ObservedJointPlanningInput(**values)


@pytest.mark.integration
@pytest.mark.parametrize("blocked,version", [
    pytest.param(False, "1", id="clear"),
    pytest.param(True, "1", id="visible-obstacle"),
    pytest.param(False, "2", id="v2-clear"),
    pytest.param(True, "2", id="v2-visible-obstacle"),
])
def test_native_observed_safe_move_uses_original_gateway_and_normal_cleanup(tmp_path, record_property, monkeypatch, blocked, version):
    """Native sensor/planner/executor gate, not an Agent or grasp success grade."""
    import json
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig, build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from project_paths import resolve_path
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT, _request

    captured, rendered, filters, rejections = [], [], [], []
    original_cloud = module.obstacle_cloud_from_captures
    def record_cloud(**kwargs):
        captured.extend(kwargs["captures"])
        return original_cloud(**kwargs)
    monkeypatch.setattr(module, "obstacle_cloud_from_captures", record_cloud)
    original_render = module.RM65CapturedRobotObservationFilter._render_depth_maps
    def record_render(self, **kwargs):
        filters.append(self)
        result = original_render(self, **kwargs)
        rendered.extend(result[0])
        return result
    monkeypatch.setattr(module.RM65CapturedRobotObservationFilter, "_render_depth_maps", record_render)

    config = module.MujocoObservedSafeMotionConfig(
        robot_urdf_path=resolve_path("assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"),
        collision_scene_profile="dual_arm_hand_proxy_dmtac_grasp",
        observation=SafeMotionObservationConfig(roles=("main",)))
    (tmp_path / "artifacts").mkdir()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
            realtime_worker=False, runtime_session_id="test:observed-safe",
            control_domain_id="test:observed-safe",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_safe_tcp_pose=True,
        enable_safe_tcp_path_constraints=version == "2",
        artifact_storage=MujocoAgentArtifactStorageConfig(root=tmp_path / "artifacts"),
        safe_motion_observation=config)
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        start = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"]["right_arm"]
        target = list(start)
        target[2] += .09 if blocked else .03
        if blocked:
            # Scenario setup only. The policy still receives no body pose,
            # native label or ground-truth obstacle list.
            world_target = session.application.transform_tcp_pose_xyzquat(target,
                source_frame_id=session.contract_bundle.motion_profile.reference_frame_id,
                target_frame_id=session.back_projection_provider.capture_port.source_frame_id)
            def place_obstacle(scene):
                address = scene.joint_qpos_addr("debug_cube_freejoint")
                scene.data.qpos[address:address+3] = world_target[:3]
                scene.forward()
            session.application.runtime.rig.capture_scene(place_obstacle)
        before_time = float(session.application.scene.data.time)
        planner = session.application.target.planned_tcp_motion
        original_plan = planner.plan
        def record_plan(*args, **kwargs):
            try:
                return original_plan(*args, **kwargs)
            except MujocoTcpPlanRejected as error:
                rejections.append(error)
                nearest = error.details.get("metadata", {}).get("nearest_observed")
                if nearest and captured:
                    capture = captured[0]
                    point = np.asarray(nearest["occupied_center_world_m"])
                    pose = np.asarray(capture.source_from_camera)
                    camera = pose[:3, :3].T @ (point - pose[:3, 3])
                    k = capture.intrinsics
                    u, v = int(round(k["fx"]*camera[0]/camera[2]+k["cx"])), int(round(k["fy"]*camera[1]/camera[2]+k["cy"]))
                    rows, cols = np.indices(capture.depth_m.shape)
                    z = capture.depth_m
                    points = np.stack(((cols-k["cx"])*z/k["fx"], (rows-k["cy"])*z/k["fy"], z), axis=-1)
                    world = points @ pose[:3, :3].T + pose[:3, 3]
                    voxel = np.floor(point / config.observation.voxel_size_m)
                    matches = np.all(np.floor(world / config.observation.voxel_size_m) == voxel, axis=-1) & (z > 0)
                    source_pixels = []
                    # Diagnostic labels only: never supplied to the planning
                    # provider, filtering, target resolver or Agent.
                    def labels(scene):
                        def draw(renderer):
                            renderer.renderer.enable_segmentation_rendering()
                            try:
                                renderer.renderer.update_scene(scene.data, camera="front")
                                return renderer.renderer.render().copy()
                            finally:
                                renderer.renderer.disable_segmentation_rendering()
                        return session.application._depth_snapshot_source._renderer_cache.render(
                            scene, width=capture.width, height=capture.height, draw=draw)
                    labels_image = session.application._capture_visual_scene(labels)
                    for row, col in zip(*np.nonzero(matches)):
                        geom_id, kind = labels_image[row, col]
                        scene = session.application.scene
                        geom_name = (scene.mujoco.mj_id2name(scene.model, scene.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
                                     if int(kind) == int(scene.mujoco.mjtObj.mjOBJ_GEOM) else None)
                        source_pixels.append({"pixel": [int(col), int(row)], "depth": float(z[row, col]),
                            "self_depths": [float(image[row, col]) for image in rendered],
                            "native_diagnostic_geom_name": geom_name})
                    renderer = filters[0]
                    pin, _, _, model, visual, selected = renderer._runtime_components()
                    q = np.concatenate((capture.capture_qpos_by_side["left"], np.zeros(6),
                                        capture.capture_qpos_by_side["right"], np.zeros(6)))
                    data, visual_data = model.createData(), pin.GeometryData(visual)
                    pin.updateGeometryPlacements(model, data, visual, visual_data, q)
                    import hppfcl
                    distances = []
                    for index in selected:
                        item, placement = visual.geometryObjects[index], visual_data.oMg[index]
                        local = np.asarray(placement.rotation).T @ (point-np.asarray(placement.translation))
                        distance = hppfcl.distance(item.geometry, hppfcl.Transform3f(), hppfcl.Sphere(0),
                            hppfcl.Transform3f(np.eye(3), local), hppfcl.DistanceRequest(), hppfcl.DistanceResult())
                        distances.append((float(distance), item.name))
                    record_property("nearest_sensor_diagnostic_json", json.dumps({
                        "world_point": point.tolist(), "pixel": [u, v],
                        "camera_point": camera.tolist(),
                        "measured_depth_m": float(capture.depth_m[v, u]),
                        "self_rendered_depths_m": [float(image[v, u]) for image in rendered],
                        "exact_source_pixels": source_pixels, "CAD_distances_m": sorted(distances)[:5],
                        "rejection": error.details}, default=str))
                raise
        monkeypatch.setattr(planner, "plan", record_plan)
        admitted = adapter.submit_skill_request(_request(session, adapter, target, version=version).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        record_property("native_observed_safe_operation_json", json.dumps(operation.to_wire(), default=str))
        if blocked:
            assert operation.execution_outcome == "failed", operation
            checks = [error.details for error in rejections]
            checks.extend(attempt["collision_check"] for error in rejections
                for attempt in error.details.get("endpoint_attempts", ()) if "collision_check" in attempt)
            assert any(check.get("metadata", {}).get("observed_scene_hash")
                       and check.get("reason") == "observed_obstacle_clearance_blocked" for check in checks), rejections
            assert float(session.application.scene.data.time) == before_time
        else:
            assert operation.execution_outcome == "completed", operation
            assert operation.task_outcome == "succeeded", operation
            final = adapter.get_observation_snapshot()["measurement_summary"][
                "pose_xyz_quat_xyzw_by_component"]["right_arm"]
            error_m = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
            assert error_m <= session.contract_bundle.motion_profile.translation_tolerance_m
            record_property("measured_target_error_m", error_m)
    finally:
        closed = session.close()
        receipt = closed.port_receipt
        record_property("normal_cleanup_json", json.dumps({
            "accepted": closed.accepted, "application_closed": session.application._closed,
            "receipt_id": None if receipt is None else receipt.receipt_id,
            "succeeded": False if receipt is None else receipt.succeeded,
            "proves_motion_or_hold": False if receipt is None else receipt.proves_motion_or_hold,
            "evidence_refs": [] if receipt is None else list(receipt.evidence_refs)}))
        assert closed.accepted
        assert session.application._closed
        assert receipt is not None and receipt.succeeded and receipt.proves_motion_or_hold
        assert receipt.evidence_refs
