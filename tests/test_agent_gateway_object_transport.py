import io
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.object_transport import (
    build_object_transport_registration, build_object_transport_reader, build_object_transport_executor,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.skills.perception.object_transport import (
    validate_object_transport_parameters, validate_object_transport_result,
    validate_object_transport_result_for_parameters,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory
from teleop_runtime.skill_deploy.rm65_object_transport import ObjectTransportPlan
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError
from test_object_transport import observation, pose
from test_agent_gateway_grasp_observation import frozen_stack


def test_registered_skill_is_bound_in_gateway_and_missing_owner_is_rejected(tmp_path):
    from agent_gateway.profiles.real.motion_gateway_session import assemble_real_agent_primitive_gateway_session
    from agent_gateway.profiles.real.motion_primitive_profiles import build_real_agent_primitive_contract_bundle
    from agent_gateway.profiles.real.perception_deployment import build_real_perception_capture_registration
    from agent_gateway.protocol.contracts import AgentSkillRequest
    from test_agent_gateway_real_motion_gateway_session import _runtime_fixture, _owner_state, CLIENT_ID

    base, port, _ = _runtime_fixture()
    source_provider, _ = frozen_stack()
    provider = source_provider.perception_provider
    attachment = base.attachment
    perception = build_real_perception_capture_registration(
        rgb_source_catalog=provider.rgb_source_catalog(), depth_source_catalog=provider.depth_source_catalog(),
        source_binding_hash=provider.source_binding.semantic_content_hash,
        **{key: getattr(attachment, key) for key in ("runtime_session_id", "control_domain_id", "target_binding_id",
            "station_manifest_hash", "execution_manifest_hash")})
    registration = build_object_transport_registration(
        base_registration=SimpleNamespace(skill=perception.rgb, binding=perception.rgb_binding),
        observation_profile_hash="test:verified-pose-owner")
    bundle = build_real_agent_primitive_contract_bundle(attachment=attachment, port_config=port.config,
        frame_graph=base.frame_graph, primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=base.deployment_bindings[0].safety_envelope_refs,
        perception_capture=perception, object_transport_registration=registration)
    store = RecoverableAgentArtifactStore(root=tmp_path, quota_bytes=32*1024*1024, max_artifact_bytes=16*1024*1024)
    kwargs = dict(contract_bundle=bundle, port=port, owner_state_provider=lambda: _owner_state(bundle, captured_at=10.),
        allowed_client_ids=(CLIENT_ID,), perception_provider=provider, artifact_store=store, clock=lambda: 10.)
    with pytest.raises(ValueError, match="bound observation owner"):
        assemble_real_agent_primitive_gateway_session(**kwargs)
    calls = []
    def observe(**kw):
        calls.append(kw)
        return observation(scene_revision=kw["station_revision"]), ObstacleSceneSnapshot("world", "frame", kw["station_revision"])
    class Planner:
        def plan_object_move(self, **kw):
            return ObjectTransportPlan(RealAgentJointTrajectory(np.array([0., 1.]),
                {s: np.zeros((2, 6)) for s in ("left", "right")}, ("sim:planned",)), {})
    gateway = assemble_real_agent_primitive_gateway_session(**kwargs,
        held_object_observer=observe, object_transport_planner_factory=Planner)
    assert "perception.plan_object_move" in gateway.passive_read_executors_by_skill_id
    assert "perception.plan_object_move" not in gateway.executors_by_skill_id
    port.publish_measured_pose(pose_world_by_component={"right_arm": np.eye(4)}, measured_at=10., available_at=10.,
        source_evidence_refs=("sim:state",))
    snapshot = gateway.observation_producer.capture_and_publish(gateway.service).snapshot
    entry = bundle.route_manifest.route("perception.plan_object_move", "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(client_id=CLIENT_ID, request_id="transport:test", idempotency_key="transport:test",
        client_sequence=1, runtime_session_id=attachment.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision, skill_id=route.skill_id, skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash, route_id=route.route_id, route_version=route.route_version,
        route_hash=route.semantic_content_hash, deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=params(), observation_snapshot_id=snapshot.snapshot_id, requested_mode="real_active", requested_timeout_ms=30_000)
    submitted = gateway.service.submit_request(request=request, current_station_revision=snapshot.station_revision, now=10.)
    assert submitted.accepted, submitted.reason_code
    phase_before_launch = port.snapshot().phase
    launched = gateway.service.launch_operation(submitted.operation.operation_id)
    assert launched.accepted, launched.reason_code
    assert gateway.object_transport_executor.wait_all(timeout_sec=3.)
    outcome = gateway.object_transport_executor.completion_for(submitted.operation.operation_id)
    assert outcome.accepted
    assert outcome.operation.execution_result["status"] == "path_found"
    assert calls == [dict(object_id="cube", side="right", station_revision=snapshot.station_revision)]
    assert port.snapshot().phase == phase_before_launch
    store.close()


def test_request_and_result_rigid_frames_and_tuple_roundtrip():
    parameter = params()
    reflected = pose()
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        validate_object_transport_parameters({**parameter, "T_reference_from_object_goal": reflected.reshape(-1).tolist()})
    result = dict(parameter)
    result.pop("speed_scale")
    result.update(status="rejected", reason_code="collision", scene_revision=1,
        T_reference_from_tool_goal=parameter["T_reference_from_object_goal"], T_tool_from_object=pose().reshape(-1).tolist(),
        trajectory_hash=None, duration_sec=0., sample_count=0, motion_executed=False, execution_authorized=False)
    validate_object_transport_result(result)
    validate_object_transport_result_for_parameters(
        {**parameter, "T_reference_from_object_goal": tuple(parameter["T_reference_from_object_goal"])}, result)
    wrong = dict(result, T_tool_from_object=pose((.1, 0, 0)).reshape(-1).tolist())
    with pytest.raises(ValueError, match="relation disagree"):
        validate_object_transport_result(wrong)


def params():
    return dict(object_id="cube", side="right", reference_frame_id="world",
                T_reference_from_object_goal=pose((.4, -.2, .5)).reshape(-1).tolist(), speed_scale=.2)


def test_skill_registration_closes_contract_without_control_effects(tmp_path):
    _, base = frozen_stack()
    reg = build_object_transport_registration(base_registration=base, observation_profile_hash="verified-geometry-v1")
    assert reg.skill.definition.skill_id == "perception.plan_object_move"
    route = reg.skill.routes[0]
    assert route.control_effect_class == "read_only" and not route.may_move_robot
    assert route.skill_io_contract_hash == reg.skill.io_contract.semantic_content_hash
    reg.payload_validator.validate_parameters(params())
    executor = build_object_transport_executor(registration=reg, max_retained_operations=4, clock=lambda: 10.,
        runtime_session_id="session:test", current_station_revision=lambda: 1,
        artifact_store=RecoverableAgentArtifactStore(root=tmp_path, quota_bytes=32*1024*1024,
            max_artifact_bytes=16*1024*1024), observe_held_object=lambda **kw: None,
        planner_factory=lambda: None)
    assert executor.control_effect_class == "read_only"


def test_caller_cannot_forge_grasp_relation_or_geometry():
    for key in ("T_tool_from_object", "grasp_verified", "disable_collision", "box_size_m"):
        with pytest.raises(ValueError):
            validate_object_transport_parameters({**params(), key: True})


@pytest.mark.parametrize("failure", [None, "collision", "stale", "scene_changed"])
def test_readonly_chain_preserves_identity_artifact_and_failure(tmp_path, failure):
    held = observation()
    scene = ObstacleSceneSnapshot("world", "frame", 1)
    requested = []
    version = [1]
    class Planner:
        def plan_object_move(self, **kwargs):
            requested.append(kwargs)
            assert kwargs["observation"] is held
            if failure == "collision":
                raise RM65GraspTrajectoryPlanningError("path_collision_blocked")
            if failure == "scene_changed":
                version[0] = 2
            trajectory = RealAgentJointTrajectory(np.array([0., 1.]),
                {s: np.zeros((2, 6)) for s in ("left", "right")}, ("planned:test",))
            return ObjectTransportPlan(trajectory, {"planning_kind": "object_transport"})
    store = RecoverableAgentArtifactStore(root=tmp_path, quota_bytes=32*1024*1024, max_artifact_bytes=16*1024*1024)
    def observed(**kw):
        assert kw == dict(object_id="cube", side="right", station_revision=1)
        return held, scene
    reader = build_object_transport_reader(observe_held_object=observed, planner_factory=Planner,
        artifact_store=store, runtime_session_id="session:test", current_station_revision=lambda: version[0],
        clock=lambda: 10.5 if failure == "stale" else 10.05)
    request = SimpleNamespace(parameters=params(), expected_station_revision=1)
    if failure in {"stale", "scene_changed"}:
        with pytest.raises(ValueError, match="stale|Scene changed"):
            reader(request, SimpleNamespace(operation_id="operation:1"))
        if failure == "stale":
            assert not requested
        return
    result = reader(request, SimpleNamespace(operation_id="operation:1"))
    validate_object_transport_result(result.execution_result)
    validate_object_transport_result_for_parameters(params(), result.execution_result)
    assert result.execution_result["motion_executed"] is False
    if failure == "collision":
        assert not result.output_artifact_bindings
        assert result.execution_result["reason_code"] == "path_collision_blocked"
    else:
        assert len(result.output_artifact_bindings) == 1
        assert result.execution_result["sample_count"] == 2
        artifact = result.output_artifact_bindings[0].artifact
        with np.load(store.managed_read_path(artifact.artifact_id), allow_pickle=False) as content:
            metadata = json.loads(content["metadata_json"].tobytes())
            assert metadata["result"]["trajectory_hash"] == result.execution_result["trajectory_hash"]
            assert content["left_qpos"].shape == content["right_qpos"].shape == (2, 6)
            np.testing.assert_array_equal(content["time_from_start_sec"], [0, 1])
        np.testing.assert_allclose(np.array(result.execution_result["T_reference_from_tool_goal"]).reshape(4, 4)
                                   @ held.tool_from_object, pose((.4, -.2, .5)), atol=1e-12)
