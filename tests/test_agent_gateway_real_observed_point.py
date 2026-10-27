"""Real-profile semantic wiring with measured/sensor ports, never devices."""

from dataclasses import replace

import numpy as np
import pytest

from agent_gateway.profiles.back_projection import LiveAlignedDepthCapture
from agent_gateway.profiles.real.motion_observed_scene import SafeMotionObservationConfig
from agent_gateway.profiles.real.observed_point_motion import RealObservedPointMotionLifecycle
from agent_gateway.profiles.real.rm65_composition import assemble_exclusive_rm65_agent_primitive_composition
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN, _requested_action_and_component_ids,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.access_policy import assess_skill_request_access
from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.skills.motion.observed_point import SKILL_ID
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.runtime import TeleopObservation
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPhase
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError
from test_agent_gateway_real_motion_lifecycle import _OwnerSession, _publish_qpos
from test_agent_gateway_real_rm65_composition import (
    _lower, REPO_ROOT, DEFAULT_RM65_CAMERA_REGISTRY_V4, DEFAULT_RM65_CAMERA_MOUNT_URDF,
)

pytestmark = pytest.mark.contract


def _composition(monkeypatch, tmp_path, *, primary="right", dual=False, action="move_to_observed_point",
                 projection=True, observed=True, max_age=30.):
    lower, guard = _lower(monkeypatch, envelope_side="dual" if dual else primary, absolute=True)
    # Pure wiring only while the shared native-resource lock belongs to the operator.
    def forbidden_native(**_):
        raise AssertionError("This focused suite must not run native planning or prepare geometry")

    monkeypatch.setattr("agent_gateway.profiles.real.rm65_composition.RM65GraspJointTrajectoryPlanner",
                        lambda **_: forbidden_native)
    monkeypatch.setattr("agent_gateway.profiles.real.rm65_composition.build_exclusive_rm65_skill_target_builder",
                        lambda *_: lower.target_builder)
    monkeypatch.setattr("robot_motion.rm65.captured_robot_observations.RM65CapturedRobotObservationFilter.prepare",
                        forbidden_native)
    now = [10.]
    (tmp_path / "artifacts").mkdir()
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower, socket_path=tmp_path / "gateway.sock", authorized_side=primary,
        authorized_action=action, authorized_motion_sides=("left", "right") if dual else (primary,),
        confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id="rpent:test", client_uid=1000, runtime_session_id="observed:real",
        control_domain_id="observed:domain", embodiment_instance_id="observed:arms",
        camera_config_path=REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json",
        camera_registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        camera_mount_urdf_path=DEFAULT_RM65_CAMERA_MOUNT_URDF, artifact_root=tmp_path / "artifacts",
        enable_back_projection=projection,
        safe_motion_observation_config=SafeMotionObservationConfig(roles=("main",), maximum_scene_age_sec=max_age) if observed else None,
        clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
    assert guard.starts == 0 and all(backend.starts == 0 for backend in lower.arm_backends.values())
    assert composition.deployment.gateway_session.localization_provider is None
    return composition, lower, now


def _request(bundle, skill_id, parameters, *, sequence=1, version="1"):
    entry = bundle.route_manifest.route(skill_id, version)
    route, profile = entry.route, entry.route.allowed_parameter_profiles[0]
    return AgentSkillRequest(client_id="rpent:test", request_id="request:"+str(sequence),
        idempotency_key="idempotency:"+str(sequence), client_sequence=sequence,
        runtime_session_id=bundle.attachment.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=0, skill_id=skill_id, skill_version=version,
        skill_definition_hash=route.skill_definition_hash, route_id=route.route_id,
        route_version=route.route_version, route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters, observation_snapshot_id="observation:test", requested_mode="real_active",
        requested_timeout_ms=1000, skill_execution_grant_id=(
            "agent-skill-grant:"+"7"*64 if route.may_move_robot else None))


def _operation(request, *, completed=False, result=None):
    return AgentSkillOperation(operation_id="operation:"+str(request.client_sequence),
        request_hash=request.semantic_content_hash, runtime_session_id=request.runtime_session_id,
        status="completed" if completed else "admitted", reason_code="test",
        station_revision_at_admission=0, current_station_revision=0,
        started_at=10., updated_at=10., deadline_at=11., execution_result=result,
        execution_outcome="completed" if completed else "pending")


def _source(session, monkeypatch, anchor):
    registration = session.contract_bundle.perception_capture.back_projection
    profile = registration.profile
    width, height = profile.dimensions_by_role["main"]
    camera = np.eye(4)
    camera[:3, 3] = np.asarray(anchor) - [0., 0., .2]
    capture = LiveAlignedDepthCapture(source_id=profile.source_id_by_role["main"], role="main",
        captured_at=10., station_revision=0, capture_group_id="captured:metric-depth",
        depth_content_hash="a"*64, intrinsics={"width":width,"height":height,
            "fx":400.,"fy":400.,"cx":width/2,"cy":height/2},
        reference_from_camera=camera, depth_m=np.full((height,width),.2,dtype=np.float32))
    monkeypatch.setattr(session.back_projection_provider.capture_port, "capture_aligned_depth", lambda **_: capture)
    request = _request(session.contract_bundle, "perception.back_project", {
        "role":"main", "target_pixel_uv":[width//2,height//2], "pixel_region_xyxy":None,
        "world_z_min_m":None, "world_z_max_m":None}, sequence=0)
    result = session.back_projection_provider.project_request(request)
    operation = _operation(request, completed=True, result=result)
    registration.payload_validator.validate_parameters(request.parameters)
    registration.payload_validator.validate_execution_result(result)
    return request, operation


@pytest.mark.parametrize("primary,dual", [("right",False),("left",False),("right",True)])
def test_observed_point_composition_uses_original_safe_owner_and_neutral_contract(monkeypatch,tmp_path,primary,dual):
    composition, lower, now = _composition(monkeypatch,tmp_path,primary=primary,dual=dual)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        entry = bundle.route_manifest.route(SKILL_ID,"1")
        assert entry.route.target_kind == "real" and entry.route.requires_operator_confirmation
        assert bundle.observed_point_profile.component_ids == bundle.attachment.component_ids
        assert session.port is composition.deployment.control_runtime.motion_port
        profile = bundle.observed_point_profile
        measured_public, _ = bundle.frame_graph.transform_pose_matrix(np.eye(4),
            source_reference_frame_id=profile.execution_reference_frame_id,
            target_reference_frame_id=profile.reference_frame_id)
        source_request, source_op = _source(session,monkeypatch,measured_public[:3,3])
        for component in profile.component_ids:
            parameters={"component_id":component,"observation_operation_id":source_op.operation_id,
                        "point_index":0,"offset_xyz_m":[0.,0.,.03],"offset_frame":"reference"}
            request = _request(bundle,SKILL_ID,parameters)
            operation = _operation(request)
            validator = bundle.payload_validators.validator_for(entry.route)
            validator.validate_parameters(parameters)
            assert _requested_action_and_component_ids(request)==("move_to_observed_point",(component,))
            lifecycle = RealObservedPointMotionLifecycle(port=session.port,contract_bundle=bundle,
                route=entry.route,binding=bundle.binding_for_skill(SKILL_ID),profile=profile,
                request=request,operation=operation,grasp_trajectory_planner=lambda **_:None,
                observation_config=SafeMotionObservationConfig(), lookup_observation=lambda _: (source_request,source_op),clock=lambda:now[0])
            resolved, _, _ = lifecycle._resolve_target((0.,0.,0.,0.,0.,0.,1.))
            np.testing.assert_allclose(xyzquat_to_pose_matrix(resolved.target_pose),
                np.eye(4)+np.array([[0,0,0,0],[0,0,0,0],[0,0,0,.03],[0,0,0,0]]),atol=1e-7)
            assert lifecycle.side==component.removesuffix("_arm")
    finally:
        session.close_unused()


@pytest.mark.parametrize("projection,observed", [(False,True),(True,False),(False,False)])
def test_observed_point_rejects_missing_dependencies_before_devices(monkeypatch,tmp_path,projection,observed):
    with pytest.raises(ValueError,match="requires back projection and the RGB-D safe scene"):
        _composition(monkeypatch,tmp_path,projection=projection,observed=observed)


def test_default_safe_composition_does_not_publish_observed_point(monkeypatch,tmp_path):
    composition,_,_ = _composition(monkeypatch,tmp_path,action="move_tcp_to_pose_safe")
    session=composition.deployment.gateway_session
    try:
        assert session.contract_bundle.observed_point_profile is None
        assert SKILL_ID not in {e.route.skill_id for e in session.contract_bundle.route_manifest.skill_routes}
    finally:
        session.close_unused()


def test_observed_motion_requires_its_own_action_side_and_fresh_request_bound_grant(monkeypatch, tmp_path):
    from agent_gateway.profiles.real.session_authorization import RealAgentPrimitiveSessionAuthorization

    composition, _, now = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        source = _operation(_request(session.contract_bundle, SKILL_ID, {
            "component_id": "right_arm", "observation_operation_id": "source:1",
            "point_index": 0, "offset_xyz_m": [0., 0., .03], "offset_frame": "reference"}))
        request = replace(_request(session.contract_bundle, SKILL_ID, _parameters(source)),
                          skill_execution_grant_id=None)
        legacy = RealAgentPrimitiveSessionAuthorization(gateway_session=session, side="right",
            action="move_tcp_to_pose_safe", client_id=request.client_id,
            confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN, clock=lambda: now[0])
        assert legacy.grant_resolver(request) is None
        assert composition.authorization.grant_resolver(replace(request, parameters={
            **request.parameters, "component_id": "left_arm"})) is None
        assert composition.authorization.grant_resolver(replace(request, client_id="other-client")) is None
        grant = composition.authorization.grant_resolver(request)
        assert grant is not None
        request = replace(request, skill_execution_grant_id=grant.grant_id)

        def access(candidate=request, **changes):
            return assess_skill_request_access(policy=session.service.access_policy,
                manifest=session.contract_bundle.route_manifest, request=candidate,
                current_station_revision=0, now=changes.pop("now", now[0]), grant=grant, **changes)

        assert access().accepted
        for change in ({"observation_operation_id": "source:2"}, {"point_index": 1},
                       {"offset_xyz_m": [0., 0., .04]}, {"offset_frame": "tool"}):
            assert not access(replace(request, parameters={**request.parameters, **change})).accepted
        assert not access(grant_invocations_used=1).accepted
        assert not access(now=grant.expires_at + .001).accepted
        assert session.port.snapshot().dispatch_count == 0
    finally:
        session.close_unused()


def test_observed_motion_assembly_refuses_a_static_planner_or_missing_profile(monkeypatch, tmp_path):
    from agent_gateway.profiles.real.motion_gateway_session import assemble_real_agent_primitive_gateway_session

    composition, _, now = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        with pytest.raises(ValueError, match="profile and Route must coexist"):
            replace(session.contract_bundle, observed_point_profile=None)
        with pytest.raises(ValueError, match="existing RGB-D safe planner and policy"):
            assemble_real_agent_primitive_gateway_session(contract_bundle=session.contract_bundle,
                port=session.port, owner_state_provider=lambda: None, allowed_client_ids=("rpent:test",),
                perception_provider=session.perception_provider, artifact_store=session.artifact_store,
                grasp_trajectory_planner=lambda **_: None, safe_pose_trajectory_planner=lambda **_: None,
                emergency_stop=lambda _: None,
                clock=lambda: now[0])
        assert session.port.snapshot().dispatch_count == 0
    finally:
        session.close_unused()


def _parameters(source_operation, *, component="right_arm", offset=(0., 0., .03), frame="reference"):
    return {"component_id": component, "observation_operation_id": source_operation.operation_id,
            "point_index": 0, "offset_xyz_m": list(offset), "offset_frame": frame}


def _v2_source(session, monkeypatch, *, acquired_at=9.5):
    _, measured = _source(session, monkeypatch, [.1, .2, .3])
    source = _request(session.contract_bundle, "perception.back_project", {
        "observation_operation_id": "captured-rgbd:1", "role": "main",
        "pixels_uv": [[320, 240], [321, 240]]}, sequence=0, version="2")
    original = measured.execution_result
    result = {key: original[key] for key in (
        "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
        "depth_content_hash", "capture_group_id", "source_id", "station_revision")}
    result.update(observation_operation_id="captured-rgbd:1", role="main", acquired_at=acquired_at,
        rgb_content_hash="b" * 64, point_semantics="observed_surface_not_object_center_or_pose",
        points=[{"pixel_uv": pixel, "depth_m": .2,
            "point_camera_m": [.0005 * index, 0., .2],
            "point_reference_m": [.1 + .0005 * index, .2, .3]}
            for index, pixel in enumerate(source.parameters["pixels_uv"])])
    return source, _operation(source, completed=True, result=result)


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("age", [.5, 29.79, 30., 30.001, -.001])
def test_reference_age_uses_bound_scene_ttl_without_refreshing_saved_evidence(
        monkeypatch, tmp_path, version, age):
    composition, _, now = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        now[0] = 100.
        bundle = session.contract_bundle
        source, op = (_source(session, monkeypatch, [.1, .2, .3]) if version == "1"
                      else _v2_source(session, monkeypatch, acquired_at=now[0] - age))
        op = replace(op, started_at=now[0] - age if version == "1" else now[0],
                     updated_at=max(now[0], now[0] - age), deadline_at=now[0] + 1.)
        request = _request(bundle, SKILL_ID, _parameters(op))
        entry = bundle.route_manifest.route(SKILL_ID, "1")
        lifecycle = session.executors_by_skill_id[SKILL_ID]._lifecycle_factory(
            request=request, operation=_operation(request))
        lifecycle._lookup = lambda _: (source, op)
        maximum_age = 30. - bundle.perception_capture.profile.max_frame_age_ms / 1000. if version == "1" else 30.
        if 0 <= age <= maximum_age:
            lifecycle._resolve_target((0., 0., 0., 0., 0., 0., 1.))
            lifecycle._validate_target_before_activation()
            assert lifecycle._source.acquired_at == (None if version == "1" else now[0] - age)
        else:
            with pytest.raises(ValueError, match="expired or from the future"):
                lifecycle._resolve_target((0., 0., 0., 0., 0., 0., 1.))
        assert session.port.snapshot().dispatch_count == 0
        assert entry.route.control_effect_class == "motion"
    finally:
        session.close_unused()


@pytest.mark.parametrize("fault", ["missing", "client", "session", "manifest", "hash", "route",
                                    "binding", "profile", "revision", "pending", "index", "inferred",
                                    "frame_graph", "frame_spec", "malformed_point"])
def test_retained_source_authentication_uses_current_real_validators(monkeypatch, tmp_path, fault):
    composition, _, _ = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        source, op = _source(session, monkeypatch, [.1, .2, .3])
        request = _request(bundle, SKILL_ID, _parameters(op))
        if fault == "client":
            request = replace(request, client_id="other-client")
        elif fault == "session":
            request = replace(request, runtime_session_id="other-session")
        elif fault == "manifest":
            request = replace(request, skill_route_manifest_hash="other-manifest")
        elif fault == "hash":
            op = replace(op, request_hash="other-request")
        elif fault == "revision":
            op = replace(op, execution_result={**op.execution_result, "station_revision": 1})
        elif fault == "pending":
            op = replace(op, status="running", execution_outcome="pending", execution_result=None)
        elif fault == "index":
            request = replace(request, parameters={**request.parameters, "point_index": 1})
        elif fault in {"frame_graph", "frame_spec", "malformed_point"}:
            field = {"frame_graph": "frame_graph_hash", "frame_spec": "reference_frame_spec_hash",
                     "malformed_point": "world_xyz_m"}[fault]
            op = replace(op, execution_result={**op.execution_result,
                field: [.1, .2] if fault == "malformed_point" else "b" * 64})
        elif fault in {"route", "binding", "profile", "inferred"}:
            field = {"route": "route_hash", "binding": "deployment_binding_hash",
                     "profile": "parameter_profile_hash", "inferred": "skill_id"}[fault]
            source = replace(source, **{field: "other.skill" if field == "skill_id" else "other"})
            op = replace(op, request_hash=source.semantic_content_hash)
        with pytest.raises((ValueError, KeyError)):
            lifecycle = session.executors_by_skill_id[SKILL_ID]._lifecycle_factory(
                request=request, operation=_operation(request))
            lifecycle._lookup = lambda _: None if fault == "missing" else (source, op)
            lifecycle._resolve_target((0., 0., 0., 0., 0., 0., 1.))
        assert session.port.snapshot().dispatch_count == 0
    finally:
        session.close_unused()


@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_relation_transforms_once_and_preserves_activation_orientation(monkeypatch, tmp_path, frame):
    composition, _, now = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        profile = bundle.observed_point_profile
        anchor = [.13, -.08, .21]
        source, source_op = _source(session, monkeypatch, anchor)
        request = replace(_request(bundle, SKILL_ID, _parameters(source_op,
            offset=(.01, .02, .03), frame=frame)), expected_station_revision=7)
        operation = replace(_operation(request), station_revision_at_admission=7,
                            current_station_revision=7)
        entry = bundle.route_manifest.route(SKILL_ID, "1")
        lifecycle = RealObservedPointMotionLifecycle(port=session.port, contract_bundle=bundle,
            route=entry.route, binding=bundle.binding_for_skill(SKILL_ID), profile=profile,
            request=request, operation=operation, grasp_trajectory_planner=lambda **_: None,
            observation_config=SafeMotionObservationConfig(), lookup_observation=lambda _: (source, source_op), clock=lambda: now[0])
        measured = np.eye(4)
        measured[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
        measured[:3, 3] = [.1, -.2, .3]
        resolved, _, _ = lifecycle._resolve_target(tuple(pose_matrix_to_xyzquat(measured)))
        public, _ = bundle.frame_graph.transform_pose_matrix(measured,
            source_reference_frame_id=profile.execution_reference_frame_id,
            target_reference_frame_id=profile.reference_frame_id)
        expected = public.copy()
        offset = np.array([.01, .02, .03])
        expected[:3, 3] = anchor + (public[:3, :3] @ offset if frame == "tool" else offset)
        actual_public, _ = bundle.frame_graph.transform_pose_matrix(
            xyzquat_to_pose_matrix(resolved.target_pose),
            source_reference_frame_id=profile.execution_reference_frame_id,
            target_reference_frame_id=profile.reference_frame_id)
        np.testing.assert_allclose(actual_public, expected, atol=1e-7)
        assert lifecycle._source.station_revision == 0
        assert request.expected_station_revision == 7
        assert lifecycle._source.acquired_at is None
    finally:
        session.close_unused()


def test_v2_point_selection_uses_real_schema_and_retains_acquisition_identity(monkeypatch, tmp_path):
    composition, _, _ = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        source, operation = _v2_source(session, monkeypatch)
        result = operation.execution_result
        request = replace(_request(bundle, SKILL_ID, {
            **_parameters(operation), "point_index": 1}), expected_station_revision=7)
        point = resolve_retained_observed_point(request=request, lookup=lambda _: (source, operation),
            manifest=bundle.route_manifest, validators=bundle.payload_validators,
            current_entry=bundle.route_manifest.route(SKILL_ID, "1"))
        assert point.point_reference_m == (.1005, .2, .3)
        assert point.acquired_at == 9.5 and point.station_revision == 0
        bad = replace(operation, execution_result={**result, "points": list(reversed(result["points"]))})
        with pytest.raises(ValueError, match="requested observation, role or pixels"):
            resolve_retained_observed_point(request=request, lookup=lambda _: (source, bad),
                manifest=bundle.route_manifest, validators=bundle.payload_validators,
                current_entry=bundle.route_manifest.route(SKILL_ID, "1"))
    finally:
        session.close_unused()


def test_compact_request_rejects_unknown_fields_invalid_vectors_and_unbound_arms(monkeypatch, tmp_path):
    composition, _, _ = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        route = bundle.route_manifest.route(SKILL_ID, "1").route
        parameters = {"component_id": "right_arm", "observation_operation_id": "source:1",
                      "point_index": 0, "offset_xyz_m": [0., 0., .03], "offset_frame": "reference"}
        for change in ({"point_index": True}, {"point_index": -1}, {"point_index": 128},
                       {"offset_xyz_m": [0., 0., float("nan")]}, {"offset_xyz_m": [0., True, 0.]},
                       {"component_id": "left_arm"}, {"offset_frame": "camera"},
                       {"speed_scale": 1.}, {"reference_frame_id": "override"}):
            with pytest.raises(ValueError):
                bundle.payload_validators.validate_parameters(route=route, payload={**parameters, **change})
    finally:
        session.close_unused()


def test_composed_factory_resolves_from_the_service_operation_store(monkeypatch, tmp_path):
    from agent_gateway.profiles.real.motion_observed_scene import ObservedSafeMovePlanner

    composition, _, now = _composition(monkeypatch, tmp_path)
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        source, measured = _source(session, monkeypatch, [.1, .2, .3])
        decision = assess_skill_request_access(policy=session.service.access_policy,
            manifest=bundle.route_manifest, request=source, current_station_revision=0, now=now[0])
        assert decision.accepted, decision.reason_code
        store = session.service.operation_store
        outcome = store.admit(request=source, access_decision=decision,
                              current_station_revision=0, admitted_at=now[0])
        assert outcome.accepted
        for status in ("activated", "running", "completed"):
            op = outcome.operation
            outcome = store.transition_worker_projection(operation_id=op.operation_id,
                expected_operation_hash=op.semantic_content_hash, next_status=status,
                reason_code="test_boundary_projection", current_station_revision=0, updated_at=now[0],
                **({"execution_result": measured.execution_result} if status == "completed" else {}))
            assert outcome.accepted, outcome.reason_code
        request = replace(_request(bundle, SKILL_ID, _parameters(outcome.operation)),
                          skill_execution_grant_id=None)
        grant = composition.authorization.grant_resolver(request)
        assert grant is not None
        request = replace(request, skill_execution_grant_id=grant.grant_id)
        lifecycle = session.executors_by_skill_id[SKILL_ID]._lifecycle_factory(
            request=request, operation=_operation(request))
        assert isinstance(lifecycle, RealObservedPointMotionLifecycle)
        assert isinstance(lifecycle.grasp_trajectory_planner, ObservedSafeMovePlanner)
        assert lifecycle.port is composition.deployment.control_runtime.motion_port
        lifecycle._resolve_target((0., 0., 0., 0., 0., 0., 1.))
        assert lifecycle._source.operation_id == outcome.operation.operation_id
        assert lifecycle._source.request_hash == source.semantic_content_hash
        assert session.port.snapshot().dispatch_count == 0
    finally:
        session.port.request_domain_hold(reason="test_service_cleanup", requested_at=now[0])
        session.port.apply_owner_requests(session=_OwnerSession(), now=now[0])
        assert session.close().accepted


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("outcome", ["completed", "no_path", "cancel", "deadline", "stale", "missing_source",
                                     "expired_during_planning", "evicted_during_planning", "unrefreshed"])
def test_original_motion_owner_plans_executes_and_holds(monkeypatch, tmp_path, side, outcome):
    composition, lower, now = _composition(monkeypatch, tmp_path, primary=side,
        max_age=.22 if outcome == "expired_during_planning" else 30.)
    session = composition.deployment.gateway_session
    owner = _OwnerSession()
    qpos = {arm: np.zeros(6) for arm in ("left", "right")}
    calls, commands = [], []

    def fk(**kwargs):
        targets = {arm: np.eye(4) for arm in qpos}
        for arm in qpos:
            targets[arm][:3, 3] = kwargs[arm + "_qpos"][:3]
        return targets

    def publish():
        _publish_qpos(session.port, qpos_by_side=qpos, measured_at=now[0], available_at=now[0])

    def advance(dt):
        now[0] += dt
        session.port.apply_owner_requests(session=owner, now=now[0])
        if session.port.snapshot().phase == RealAgentMotionPhase.ACTIVE:
            command = session.port.command_for_observation(TeleopObservation(timestamp=now[0]), now[0])
            commands.append(command)
            for item in command.commands.values():
                qpos[item.side] = np.asarray(item.value)
            session.port.observe_session_step(command=command, command_dispatched=True)
        if outcome not in {"stale", "unrefreshed"}:
            publish()

    def planner(**kwargs):
        calls.append(kwargs)
        assert kwargs["side"] == side
        assert kwargs["require_grasp_approach"] is False
        assert kwargs["pose_goal_policy"] == "preserve_target"
        assert kwargs["speed_scale"] == .2
        assert 0 < kwargs["time_budget_sec"] <= 1.
        if outcome == "no_path":
            raise RM65GraspTrajectoryPlanningError("path_start_collision_blocked")
        if outcome in {"deadline", "expired_during_planning"}:
            now[0] += 2. if outcome == "deadline" else .03
            publish()
        target = {arm: qpos[arm].copy() for arm in qpos}
        target[side][:3] = kwargs["target_tool_tcp_world"][:3, 3]
        return RealAgentJointTrajectory(time_from_start_sec=np.array([0., .05]),
            qpos_by_side={arm: np.stack([qpos[arm], target[arm]]) for arm in qpos},
            planner_evidence_refs=("rm65-production-urdf-fcl-path:test-port-only",))

    try:
        lower.target_builder.tool_tcp_world_targets_for_qpos_pair = fk
        publish()
        bundle = session.contract_bundle
        profile = bundle.observed_point_profile
        current_public, _ = bundle.frame_graph.transform_pose_matrix(np.eye(4),
            source_reference_frame_id=profile.execution_reference_frame_id,
            target_reference_frame_id=profile.reference_frame_id)
        source, source_op = _source(session, monkeypatch, current_public[:3, 3])
        request = _request(bundle, SKILL_ID, _parameters(source_op, component=side + "_arm"))
        operation = _operation(request)
        entry = bundle.route_manifest.route(SKILL_ID, "1")
        lifecycle = RealObservedPointMotionLifecycle(port=session.port, contract_bundle=bundle,
            route=entry.route, binding=bundle.binding_for_skill(SKILL_ID), profile=profile,
            request=request, operation=operation, grasp_trajectory_planner=planner,
            observation_config=SafeMotionObservationConfig(maximum_scene_age_sec=.22
                if outcome == "expired_during_planning" else 30.),
            lookup_observation=lambda _: None if outcome == "missing_source" or (
                outcome == "evicted_during_planning" and calls) else (source, source_op),
            clock=lambda: now[0], sleep=advance)
        cancel = lambda: outcome == "cancel" and bool(calls)
        activation = lifecycle.activate(request=request, operation=operation, cancel_requested=cancel)
        if outcome == "completed":
            assert activation.succeeded
            completion = lifecycle.run(request=request, operation=operation, cancel_requested=cancel)
            assert completion.succeeded and completion.terminal_hold_confirmed
            result = completion.execution_result
            bundle.payload_validators.validate_execution_result(route=entry.route, payload=result,
                                                               request_parameters=request.parameters)
            report = result["observed_point_report"]
            assert report["translation_error_m"] < 1e-7
            assert report["rotation_error_rad"] < 1e-7
            assert report["source_request_hash"] == source.semantic_content_hash
            assert result["final_measured_state_hash"]
            assert commands
            for changes in ({"observation_operation_id": "different-source"}, {"point_index": 1},
                            {"translation_error_m": float("nan")}, {"reference_frame_id": "other-frame"}):
                with pytest.raises(ValueError):
                    bundle.payload_validators.validate_execution_result(route=entry.route,
                        payload={**result, "observed_point_report": {**report, **changes}},
                        request_parameters=request.parameters)
        else:
            assert not activation.succeeded and activation.lower_hold_confirmed
            assert not commands
            assert activation.status == {
                "no_path": "path_start_collision_blocked",
                "cancel": "real_primitive_cancelled_before_activation",
                "deadline": "grasp_path_exceeds_operation_deadline",
                "stale": "grasp_path_postplan_measured_qpos_stale",
                "missing_source": "observed_point_reference_unavailable",
                "expired_during_planning": "observed_point_reference_unavailable",
                "evicted_during_planning": "observed_point_reference_unavailable",
                "unrefreshed": "grasp_path_postplan_measured_qpos_stale",
            }[outcome]
        assert len(calls) == (0 if outcome == "missing_source" else 1)
        assert session.port.snapshot().phase == RealAgentMotionPhase.HELD
        assert owner.holds
        assert all(backend.starts == 0 for backend in lower.arm_backends.values())
    finally:
        session.close_unused()
