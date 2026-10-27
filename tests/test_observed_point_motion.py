"""Retained sensor reference isolation and original safe-motion integration."""

from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.skills.motion.observed_point import SKILL_ID, validate_observed_point_parameters
from agent_gateway.skills.motion.observed_point import (
    observed_anchor_definition_v2, observed_anchor_schema_v2,
    observed_anchor_region_definition_v3, observed_anchor_region_schema_v3,
    observed_anchor_region_offsets_v3, validate_observed_anchor_region_parameters_v3,
    validate_observed_anchor_parameters_v2,
)

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


def make_native_session(tmp_path, *, anchor_v2=False, region_v3=False):
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig, build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1)
    from agent_gateway.profiles.mujoco.motion_observed_scene import MujocoObservedSafeMotionConfig
    from robot_motion.observed_scene import SafeMotionObservationConfig
    from project_paths import resolve_path
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT
    from test_agent_gateway_surface_anchor import _AnchorSegmenter
    (tmp_path / "artifacts").mkdir()
    return build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
            realtime_worker=False, runtime_session_id="test:observed-point",
            control_domain_id="test:observed-point",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_safe_tcp_pose=True, enable_observed_point_motion=True,
        enable_anchor_motion_v2=anchor_v2,
        enable_anchor_region_motion_v3=region_v3,
        enable_object_goal_queries=anchor_v2, enable_surface_anchors=anchor_v2,
        enable_surface_anchor_v2=anchor_v2,
        segmentation_localizer=_AnchorSegmenter() if anchor_v2 else None,
        safe_tcp_motion_profile="object_relation" if anchor_v2 else "local_step",
        enable_spatial_goals=True, artifact_storage=MujocoAgentArtifactStorageConfig(root=tmp_path / "artifacts"),
        safe_motion_observation=MujocoObservedSafeMotionConfig(
            robot_urdf_path=resolve_path("assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"),
            collision_scene_profile="dual_arm_hand_proxy_dmtac_grasp",
            observation=SafeMotionObservationConfig(roles=("main",))))


def test_anchor_motion_v2_contract_has_no_caller_pose_or_source_xyz():
    profile = SimpleNamespace(component_id="right_arm", reference_frame_id="canonical_station",
        profile_id="profile:test", semantic_content_hash="a" * 64,
        workspace_bounds=(-1., 1., -1., 1., -1., 1.))
    definition = observed_anchor_definition_v2(controlled_component_ids=("right_arm",))
    schema = observed_anchor_schema_v2(definition=definition, profile=profile)
    assert (definition.skill_id, definition.skill_version) == (SKILL_ID, "2")
    assert tuple(schema.parameter_schema["required"]) == (
        "component_id", "offset_frame", "offset_xyz_m", "surface_anchor_operation_id")
    for extra in ({"anchor_xyz_m": [0., 0., 0.]},
                  {"pose_targets_by_component": {}}, {"observation_operation_id": "old"}):
        with pytest.raises(ValueError):
            validate_observed_anchor_parameters_v2({"component_id": "right_arm",
                "surface_anchor_operation_id": "source:anchor", "offset_xyz_m": [0., 0., .08],
                "offset_frame": "reference", **extra},
                component_id="right_arm")


def test_anchor_region_v3_generates_bounded_reference_offsets_without_pose_input():
    parameters = {"component_id": "right_arm", "surface_anchor_operation_id": "source:anchor",
                  "offset_xyz_m": [0., 0., .08], "lateral_radius_m": .05,
                  "height_extra_m": .10}
    offsets = observed_anchor_region_offsets_v3(parameters)
    assert offsets == ((0., 0., .08), (0., 0., .18), (-.05, 0., .18),
                       (.05, 0., .18), (0., -.05, .18), (0., .05, .18))
    assert observed_anchor_region_offsets_v3({**parameters, "lateral_radius_m": 0.,
                                              "height_extra_m": 0.}) == ((0., 0., .08),)
    profile = SimpleNamespace(component_id="right_arm", reference_frame_id="canonical_station",
        profile_id="profile:test", semantic_content_hash="a" * 64,
        workspace_bounds=(-1., 1., -1., 1., -1., 1.))
    definition = observed_anchor_region_definition_v3(controlled_component_ids=("right_arm",))
    schema = observed_anchor_region_schema_v3(definition=definition, profile=profile)
    assert definition.skill_version == schema.skill_version == "3"
    assert set(schema.parameter_schema["required"]) == set(parameters)
    assert not {"anchor_xyz_m", "pose_targets_by_component", "input_artifact_refs"} & set(
        schema.parameter_schema["properties"])
    with pytest.raises(ValueError):
        validate_observed_anchor_region_parameters_v3({**parameters, "height_extra_m": .31})


def test_anchor_region_v3_discards_overlong_pending_plan_then_checks_next_candidate():
    from agent_gateway.profiles.mujoco.observed_point_motion import MujocoAnchorMotionLifecycle
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

    calls = []

    def plan(_target, **_kwargs):
        calls.append("plan")
        if calls.count("plan") == 1:
            return SimpleNamespace(reference_poses=(None,) * 5)
        raise MujocoTcpPlanRejected("safe_tcp_target_blocked")

    lifecycle = object.__new__(MujocoAnchorMotionLifecycle)
    lifecycle.route = SimpleNamespace(skill_version="3")
    lifecycle.request = SimpleNamespace(parameters={
        "component_id": "right_arm", "surface_anchor_operation_id": "source:anchor",
        "offset_xyz_m": [0., 0., .08], "lateral_radius_m": 0., "height_extra_m": .10},
        expected_station_revision=0)
    lifecycle.operation = SimpleNamespace(operation_id="operation:region")
    lifecycle.motion_profile = SimpleNamespace(workspace_bounds=(-1., 1., -1., 1., -1., 1.),
        max_translation_delta_m=.5, max_rotation_delta_rad=1., normalize_ik_objective=False,
        stable_measurement_count=1, max_control_cycles=5,
        reference_frame_id="station", execution_reference_frame_id="station")
    lifecycle.application = SimpleNamespace(transform_tcp_pose_xyzquat=lambda pose, **_: pose)
    lifecycle._observed_planning_provider = SimpleNamespace(capture=lambda **_: object())
    lifecycle._activation_public_pose = (0., 0., .3, 0., 0., 0., 1.)
    lifecycle._anchor_public_xyz = (0., 0., .1)
    lifecycle._path_search_time_budget_sec = 3.
    lifecycle._planning_goal_tolerances = lambda: (.01, .1)
    lifecycle._planner = SimpleNamespace(plan=plan, clear=lambda: calls.append("clear"))
    with pytest.raises(MujocoTcpPlanRejected) as rejected:
        lifecycle._prepare_plan(constraint=None, constraint_identity="constraint:test",
                                cancel_requested=lambda: False)
    assert rejected.value.reason_code == "safe_tcp_no_candidate_path"
    assert calls == ["plan", "clear", "plan"]
    assert rejected.value.details["candidate_reasons"] == (
        (0, "safe_tcp_path_exceeds_control_budget"), (1, "safe_tcp_target_blocked"))


def test_anchor_region_v3_cancellation_stops_candidate_search():
    from agent_gateway.profiles.mujoco.observed_point_motion import MujocoAnchorMotionLifecycle
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

    cancelled = [False]
    plans = []

    def plan(*_args, **_kwargs):
        plans.append(1)
        cancelled[0] = True
        raise MujocoTcpPlanRejected("safe_tcp_target_blocked")

    lifecycle = object.__new__(MujocoAnchorMotionLifecycle)
    lifecycle.route = SimpleNamespace(skill_version="3")
    lifecycle.request = SimpleNamespace(parameters={
        "component_id": "right_arm", "surface_anchor_operation_id": "source:anchor",
        "offset_xyz_m": [0., 0., .08], "lateral_radius_m": 0., "height_extra_m": .10},
        expected_station_revision=0)
    lifecycle.operation = SimpleNamespace(operation_id="operation:region")
    lifecycle.motion_profile = SimpleNamespace(workspace_bounds=(-1., 1., -1., 1., -1., 1.),
        max_translation_delta_m=.5, max_rotation_delta_rad=1., normalize_ik_objective=False,
        stable_measurement_count=1, max_control_cycles=100,
        reference_frame_id="station", execution_reference_frame_id="station")
    lifecycle.application = SimpleNamespace(transform_tcp_pose_xyzquat=lambda pose, **_: pose)
    lifecycle._observed_planning_provider = SimpleNamespace(capture=lambda **_: object())
    lifecycle._activation_public_pose = (0., 0., .3, 0., 0., 0., 1.)
    lifecycle._anchor_public_xyz = (0., 0., .1)
    lifecycle._path_search_time_budget_sec = 3.
    lifecycle._planning_goal_tolerances = lambda: (.01, .1)
    lifecycle._planner = SimpleNamespace(plan=plan)
    with pytest.raises(MujocoTcpPlanRejected) as rejected:
        lifecycle._prepare_plan(constraint=None, constraint_identity="constraint:test",
                                cancel_requested=lambda: cancelled[0])
    assert rejected.value.reason_code == "safe_tcp_planning_cancelled"
    assert len(plans) == 1


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
        assert session.contract_bundle.observed_anchor_contract_v2 is None
        assert session.observed_anchor_executor_v2 is None
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


@pytest.mark.integration
def test_native_authenticated_anchor_v2_moves_via_original_safe_owner(tmp_path, record_property):
    from agent_gateway.protocol.contracts import AgentSkillRequest
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT, _request

    session = make_native_session(tmp_path, anchor_v2=True)
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        snapshot = adapter.get_observation_snapshot()
        start = snapshot["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        capture_registration = next(item for item in session.contract_bundle.surface_anchor_registrations
            if (item.skill.definition.skill_id, item.skill.definition.skill_version) ==
            ("perception.capture_surface_anchor", "2"))
        capture_route = capture_registration.entry.route
        capture_profile = capture_route.allowed_parameter_profiles[0]
        capture_request = AgentSkillRequest(
            client_id=CLIENT, request_id="request:anchor:v2", idempotency_key="anchor:v2",
            client_sequence=0, runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
            skill_route_manifest_hash=session.contract_bundle.route_manifest.semantic_content_hash,
            expected_station_revision=snapshot["station_revision"],
            observation_snapshot_id=snapshot["snapshot_id"],
            skill_id=capture_route.skill_id, skill_version="2",
            skill_definition_hash=capture_route.skill_definition_hash,
            route_id=capture_route.route_id, route_version=capture_route.route_version,
            route_hash=capture_route.semantic_content_hash,
            deployment_binding_hash=capture_registration.entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=capture_profile.parameter_profile_id,
            parameter_profile_hash=capture_profile.parameter_profile_hash,
            parameters={"role": "main", "selection_mode": "exact", "target_pixel_uv": [320, 240],
                        "language_prompt": None, "min_score": .25},
            requested_mode="sim_active", requested_timeout_ms=capture_route.max_operation_duration_ms,
        )
        admitted = session.service.submit_request(
            request=capture_request, current_station_revision=snapshot["station_revision"])
        assert admitted.accepted, admitted
        assert session.service.launch_operation(admitted.operation.operation_id).accepted
        capture_executor = next(executor for item, executor in zip(
            session.contract_bundle.surface_anchor_registrations, session.surface_anchor_executors, strict=True)
            if item is capture_registration)
        assert capture_executor.wait(admitted.operation.operation_id, timeout_sec=20.)
        captured = session.service.get_operation(admitted.operation.operation_id)
        assert captured.execution_outcome == "completed", captured.to_wire()
        assert session.segmentation_localizer.calls == 0
        outputs = tuple(binding
            for binding in captured.output_artifact_bindings if binding.slot_id == "surface_anchor")
        assert len(outputs) == 1
        measure = captured.execution_result["measurement"]
        public_anchor = session.application.transform_tcp_pose_xyzquat(
            (*measure["world_xyz_m"], 0., 0., 0., 1.),
            source_frame_id=measure["reference_frame_id"],
            target_frame_id=session.contract_bundle.motion_profile.reference_frame_id)
        target = np.asarray(start[:3]) + [0., 0., .03]
        offset = (target - np.asarray(public_anchor[:3])).tolist()

        route_entry = session.contract_bundle.route_manifest.route(SKILL_ID, "2")
        route = route_entry.route
        parameter_profile = route.allowed_parameter_profiles[0]
        motion = replace(_request(session, adapter, start, sequence=1, skill_id=SKILL_ID),
            skill_version="2", skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id, route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=route_entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter_profile.parameter_profile_id,
            parameter_profile_hash=parameter_profile.parameter_profile_hash,
            parameters={"component_id": "right_arm",
                        "surface_anchor_operation_id": captured.operation_id, "offset_xyz_m": offset,
                        "offset_frame": "reference"},
            requested_timeout_ms=route.max_operation_duration_ms)
        result = adapter.submit_skill_request(motion.to_wire())
        assert result["accepted"], result
        assert session.wait(timeout_sec=40.)
        operation = session.service.get_operation(result["operation"]["operation_id"])
        record_property("anchor_motion", json.dumps(operation.to_wire()))
        assert operation.execution_outcome == "completed", operation.to_wire()
        report = operation.execution_result["surface_anchor_report"]
        assert report["source_operation_id"] == captured.operation_id
        assert report["anchor_artifact_id"] == outputs[0].artifact.artifact_id
        assert report["translation_error_m"] < .003
        np.testing.assert_allclose(report["requested_pose_xyz_quat_xyzw"][:3], target, atol=1e-7)
        assert not session.application.target.planned_tcp_motion.active
        fresh = adapter.get_observation_snapshot()
        without_anchor = replace(motion, request_id="request:anchor:missing",
            idempotency_key="anchor:missing", client_sequence=2,
            expected_station_revision=fresh["station_revision"],
            observation_snapshot_id=fresh["snapshot_id"],
            parameters={**motion.parameters, "surface_anchor_operation_id": "missing:anchor"})
        before = float(session.application.scene.data.time)
        denied = adapter.submit_skill_request(without_anchor.to_wire())
        assert denied["accepted"], denied
        assert session.wait(timeout_sec=20.)
        failed = session.service.get_operation(denied["operation"]["operation_id"])
        assert failed.execution_outcome == "failed", failed.to_wire()
        assert failed.progress_summary["lower_hold_confirmed"]
        assert float(session.application.scene.data.time) == before
        assert not session.application.target.planned_tcp_motion.active
    finally:
        closed = session.close()
        record_property("anchor_motion_cleanup", json.dumps({
            "accepted": closed.accepted, "application_closed": session.application._closed,
            "hold": closed.port_receipt is not None and closed.port_receipt.proves_motion_or_hold}))
        assert closed.accepted and session.application._closed
        assert closed.port_receipt is not None and closed.port_receipt.proves_motion_or_hold


@pytest.mark.integration
@pytest.mark.parametrize("nominal,radius,extra,completed", (
    ([0., 0., .08], .05, .10, True),
    ([0., 0., -.10], .05, .28, True),
    ([0., 0., .08], 0., 0., False),
))
def test_native_anchor_region_v3_uses_one_observed_scene_and_original_motion_owner(
        tmp_path, record_property, nominal, radius, extra, completed):
    from agent_gateway.protocol.contracts import AgentSkillRequest
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT, _request

    session = make_native_session(tmp_path, anchor_v2=True, region_v3=True)
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        snapshot = adapter.get_observation_snapshot()
        start = snapshot["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        registration = next(item for item in session.contract_bundle.surface_anchor_registrations
            if (item.skill.definition.skill_id, item.skill.definition.skill_version)
            == ("perception.capture_surface_anchor", "2"))
        capture_route = registration.entry.route
        capture_profile = capture_route.allowed_parameter_profiles[0]
        capture = AgentSkillRequest(
            client_id=CLIENT, request_id="request:region:anchor", idempotency_key="region:anchor",
            client_sequence=0, runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
            skill_route_manifest_hash=session.contract_bundle.route_manifest.semantic_content_hash,
            expected_station_revision=snapshot["station_revision"],
            observation_snapshot_id=snapshot["snapshot_id"],
            skill_id=capture_route.skill_id, skill_version="2",
            skill_definition_hash=capture_route.skill_definition_hash,
            route_id=capture_route.route_id, route_version=capture_route.route_version,
            route_hash=capture_route.semantic_content_hash,
            deployment_binding_hash=registration.entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=capture_profile.parameter_profile_id,
            parameter_profile_hash=capture_profile.parameter_profile_hash,
            parameters={"role": "main", "selection_mode": "exact", "target_pixel_uv": [326, 280],
                        "language_prompt": None, "min_score": .25},
            requested_mode="sim_active", requested_timeout_ms=capture_route.max_operation_duration_ms)
        admitted = session.service.submit_request(
            request=capture, current_station_revision=snapshot["station_revision"])
        assert admitted.accepted
        assert session.service.launch_operation(admitted.operation.operation_id).accepted
        executor = next(executor for item, executor in zip(
            session.contract_bundle.surface_anchor_registrations,
            session.surface_anchor_executors, strict=True) if item is registration)
        assert executor.wait(admitted.operation.operation_id, timeout_sec=20.)
        source = session.service.get_operation(admitted.operation.operation_id)
        assert source.execution_outcome == "completed"
        assert session.segmentation_localizer.calls == 0

        original_capture = session.application.capture_planning_rgbd_cut
        observed_cuts = []

        def counted_capture(**kwargs):
            observed_cuts.append(kwargs["operation_id"])
            return original_capture(**kwargs)

        session.application.capture_planning_rgbd_cut = counted_capture

        entry = session.contract_bundle.route_manifest.route(SKILL_ID, "3")
        route = entry.route
        profile = route.allowed_parameter_profiles[0]
        motion = replace(_request(session, adapter, start, sequence=1, skill_id=SKILL_ID),
            skill_version="3", skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id, route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.parameter_profile_hash,
            parameters={"component_id": "right_arm",
                        "surface_anchor_operation_id": source.operation_id,
                        "offset_xyz_m": nominal,
                        "lateral_radius_m": radius, "height_extra_m": extra},
            requested_timeout_ms=route.max_operation_duration_ms)
        admitted = adapter.submit_skill_request(motion.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=60.)
        operation = session.service.get_operation(admitted["operation"]["operation_id"])
        assert observed_cuts == [operation.operation_id]
        if completed:
            assert operation.execution_outcome == "completed", operation.to_wire()
            report = operation.execution_result["surface_anchor_region_report"]
            assert report["source_operation_id"] == source.operation_id
            np.testing.assert_allclose(report["selected_offset_xyz_m"], [0., -.05, .18])
            assert report["selected_candidate_index"] == 4
            assert report["attempted_candidate_count"] == 5
            assert report["translation_error_m"] < .003
            assert float(session.application.scene.data.time) > 0.
            assert any(ref.startswith("mujoco-safe-joint-trajectory:")
                       for ref in operation.execution_receipt_refs)
            validator = session.contract_bundle.payload_validators.validator_for(route)
            altered = deepcopy(operation.to_wire()["execution_result"])
            altered["surface_anchor_region_report"]["selected_offset_xyz_m"] = [0., 0., .08]
            with pytest.raises(ValueError, match="differs from the requested"):
                validator.validate_execution_result_for_parameters(motion.parameters, altered)
        else:
            assert operation.execution_outcome == "failed", operation.to_wire()
            assert operation.reason_code == "safe_tcp_no_candidate_path"
            assert operation.progress_summary["lower_hold_confirmed"]
            assert float(session.application.scene.data.time) == 0.
        record_property("region_motion", json.dumps({
            "outcome": operation.execution_outcome,
            "reason": operation.reason_code,
            "sim_time": float(session.application.scene.data.time)}))
    finally:
        closed = session.close()
        assert closed.accepted and session.application._closed
        assert closed.port_receipt is not None and closed.port_receipt.proves_motion_or_hold
