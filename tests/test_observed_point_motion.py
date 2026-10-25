"""Retained sensor reference isolation and original safe-motion integration."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.skills.motion.observed_point import SKILL_ID, validate_observed_point_parameters

pytestmark = pytest.mark.contract


def _parameters(source="observed:1"):
    return {"component_id": "right_arm", "observation_operation_id": source,
            "point_index": 0, "offset_xyz_m": [0., 0., .08], "offset_frame": "reference"}


def _retained_fixture(version="1"):
    from test_agent_gateway_service import _capability, _manifest, _request
    route = replace(_capability(), skill_id="perception.back_project", skill_version=version,
                    skill_io_contract_hash=None)
    manifest = _manifest(route)
    source = _request(manifest, route)
    result = {"station_revision": 4, "world_xyz_m": [.1, .2, .3],
              "reference_frame_id": "station", "reference_frame_spec_hash": "frame:1",
              "frame_graph_hash": "graph:1", "depth_content_hash": "depth:1", "capture_group_id": "cut:1"}
    if version == "2":
        result.update(points=[{"point_reference_m": [.1, .2, .3]}], acquired_at=10.)
        del result["world_xyz_m"]
    op = AgentSkillOperation(operation_id="observed:1", request_hash=source.semantic_content_hash,
        runtime_session_id=manifest.runtime_session_id, status="completed", reason_code="measured",
        station_revision_at_admission=4, current_station_revision=4,
        started_at=10., updated_at=10.1, deadline_at=11., execution_outcome="completed",
        execution_result=result)
    request = replace(source, parameters=_parameters(), expected_station_revision=7)
    calls = []
    validators = SimpleNamespace(validate_parameters=lambda **kw: calls.append(kw),
                                 validate_execution_result=lambda **kw: calls.append(kw))
    return SimpleNamespace(source=source, op=op, request=request, manifest=manifest,
        validators=validators, entry=manifest.skill_routes[0], calls=calls)


def _resolve(f, pair=True):
    return resolve_retained_observed_point(request=f.request, lookup=lambda _: (f.source, f.op) if pair else None,
        manifest=f.manifest, validators=f.validators, current_entry=f.entry)


@pytest.mark.parametrize("version", ("1", "2"))
def test_saved_point_reuse_across_motion_revision_is_not_object_tracking(version):
    f = _retained_fixture(version)
    if version == "2":
        f.op = replace(f.op, execution_result={**f.op.execution_result, "station_revision": 2})
    point = _resolve(f)
    assert point.point_reference_m == (.1, .2, .3)
    assert point.station_revision == (4 if version == "1" else 2)
    assert point.station_revision != f.request.expected_station_revision
    assert point.acquired_at == (None if version == "1" else 10.)
    assert f.calls[1]["request_parameters"] == f.source.parameters


@pytest.mark.parametrize("fault", ("evicted", "client", "session", "manifest", "hash", "route",
                                   "binding", "profile", "revision", "pending", "index", "inferred"))
def test_reference_authentication_rejects_cross_context_or_non_evidence(fault):
    f = _retained_fixture()
    if fault == "client": f.request = replace(f.request, client_id="other")
    elif fault == "session": f.request = replace(f.request, runtime_session_id="other")
    elif fault == "manifest": f.request = replace(f.request, skill_route_manifest_hash="other")
    elif fault == "hash": f.op = replace(f.op, request_hash="other")
    elif fault == "revision": f.op = replace(f.op, execution_result={**f.op.execution_result, "station_revision": 3})
    elif fault == "pending": f.op = replace(f.op, status="running", execution_outcome="pending", execution_result=None)
    elif fault == "index": f.request = replace(f.request, parameters={**f.request.parameters, "point_index": 1})
    elif fault in ("route", "binding", "profile", "inferred"):
        field = {"route": "route_hash", "binding": "deployment_binding_hash",
                 "profile": "parameter_profile_hash", "inferred": "skill_id"}[fault]
        f.source = replace(f.source, **{field: "other.skill" if field == "skill_id" else "other"})
        f.op = replace(f.op, request_hash=f.source.semantic_content_hash)
    with pytest.raises((ValueError, KeyError)):
        _resolve(f, pair=fault != "evicted")


@pytest.mark.parametrize("change", ({"point_index": True}, {"point_index": -1},
    {"offset_xyz_m": [0, 0, float("nan")]}, {"component_id": "left_arm"},
    {"offset_frame": "camera"}, {"reference_frame_id": "override"}))
def test_compact_parameters_fail_closed(change):
    with pytest.raises(ValueError):
        validate_observed_point_parameters({**_parameters(), **change}, component_id="right_arm")


def make_native_session(tmp_path):
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig, build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1)
    from agent_gateway.profiles.mujoco.motion_observed_scene import MujocoObservedSafeMotionConfig
    from robot_motion.observed_scene import SafeMotionObservationConfig
    from project_paths import resolve_path
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT
    (tmp_path / "artifacts").mkdir()
    return build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
            realtime_worker=False, runtime_session_id="test:observed-point",
            control_domain_id="test:observed-point",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_safe_tcp_pose=True, enable_observed_point_motion=True,
        enable_spatial_goals=True, artifact_storage=MujocoAgentArtifactStorageConfig(root=tmp_path / "artifacts"),
        safe_motion_observation=MujocoObservedSafeMotionConfig(
            robot_urdf_path=resolve_path("assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"),
            collision_scene_profile="dual_arm_hand_proxy_dmtac_grasp",
            observation=SafeMotionObservationConfig(roles=("main",))))


def call_native(session, adapter, skill, parameters, sequence):
    from test_agent_gateway_mujoco_safe_tcp_pose import _request
    request = replace(_request(session, adapter, [0., 0., 0., 0., 0., 0., 1.],
                               sequence=sequence, skill_id=skill), parameters=parameters)
    admitted = adapter.submit_skill_request(request.to_wire())
    assert admitted["accepted"], admitted
    operation_id = admitted["operation"]["operation_id"]
    if skill == "perception.back_project":
        assert session.back_projection_executor.wait(operation_id, timeout_sec=10.)
    else:
        assert session.wait(timeout_sec=35.)
    return session.service.get_operation(operation_id)


@pytest.mark.integration
@pytest.mark.parametrize("blocked", (False, True), ids=("reuse", "new-obstacle"))
def test_native_retained_point_relation_moves_then_reuses_the_same_sensor_evidence(tmp_path, record_property, blocked):
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT
    session = make_native_session(tmp_path)
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        start = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        source = call_native(session, adapter, "perception.back_project", {
            "role": "main", "target_pixel_uv": [320, 240], "pixel_region_xyxy": None,
            "world_z_min_m": None, "world_z_max_m": None}, 0)
        assert source.execution_outcome == "completed", source.to_wire()
        result = source.execution_result
        anchor = session.application.transform_tcp_pose_xyzquat((*result["world_xyz_m"], 0., 0., 0., 1.),
            source_frame_id=result["reference_frame_id"], target_frame_id=session.contract_bundle.motion_profile.reference_frame_id)
        if blocked:
            world_target = session.application.transform_tcp_pose_xyzquat(
                (*start[:2], start[2]+.09, *start[3:]),
                source_frame_id=session.contract_bundle.motion_profile.reference_frame_id,
                target_frame_id=session.back_projection_provider.capture_port.source_frame_id)
            def place_obstacle(scene):
                address = scene.joint_qpos_addr("debug_cube_freejoint")
                scene.data.qpos[address:address+3] = world_target[:3]
                scene.forward()
            session.application.runtime.rig.capture_scene(place_obstacle)
        for seq, dz in enumerate((.09,) if blocked else (.03, .01), 1):
            target = np.asarray(start[:3]) + [0., 0., dz]
            parameters = {**_parameters(source.operation_id), "offset_xyz_m": (target-np.asarray(anchor[:3])).tolist()}
            before = float(session.application.scene.data.time)
            operation = call_native(session, adapter, SKILL_ID, parameters, seq)
            record_property("motion_" + str(seq), json.dumps(operation.to_wire()))
            if blocked:
                assert operation.execution_outcome == "failed", operation.to_wire()
                assert operation.reason_code == "safe_tcp_target_blocked", operation.to_wire()
                assert float(session.application.scene.data.time) == before
                assert operation.progress_summary["lower_hold_confirmed"]
                continue
            assert operation.execution_outcome == "completed", operation.to_wire()
            report = operation.execution_result["observed_point_report"]
            assert report["translation_error_m"] < .003
            assert report["observation_operation_id"] == source.operation_id
            np.testing.assert_allclose(report["requested_pose_xyz_quat_xyzw"][:3], target, atol=1e-7)
        # Evicted/foreign references are refused without advancing physics.
        before = float(session.application.scene.data.time)
        failed = call_native(session, adapter, SKILL_ID, _parameters("missing:observation"), 3)
        assert failed.execution_outcome == "failed"
        assert float(session.application.scene.data.time) == before
        assert failed.progress_summary["lower_hold_confirmed"]
    finally:
        closed = session.close()
        receipt = closed.port_receipt
        record_property("cleanup", json.dumps({"accepted": closed.accepted,
            "succeeded": receipt is not None and receipt.succeeded,
            "hold": receipt is not None and receipt.proves_motion_or_hold,
            "closed": session.application._closed}))
        assert closed.accepted and session.application._closed
        assert receipt is not None and receipt.succeeded and receipt.proves_motion_or_hold and receipt.evidence_refs
