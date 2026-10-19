"""Native MuJoCo single-owner comparison through the actual Gateway session."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from project_paths import resolve_path
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.transforms import rotation_angle_rad, translation_distance
from sim.mujoco.deploy import MujocoSimHILTargetConfig

CLIENT = "test:mujoco-tcp-comparison"
TASKS = (
    "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json",
    "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json",
)
pytestmark = pytest.mark.integration


def _request(session, adapter, *, skill_id, version, parameters, sequence):
    snapshot = adapter.get_observation_snapshot()
    manifest = session.contract_bundle.route_manifest
    entry = manifest.route(skill_id, version)
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=CLIENT, request_id=f"comparison:{sequence}",
        idempotency_key=f"comparison:{sequence}", client_sequence=sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=snapshot["station_revision"],
        skill_id=skill_id, skill_version=version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id, route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash, parameters=parameters,
        observation_snapshot_id=snapshot["snapshot_id"], requested_mode="sim_active",
        requested_timeout_ms=route.max_operation_duration_ms,
    )


@pytest.mark.parametrize("task", TASKS)
def test_safe_motion_then_fresh_comparison_uses_the_same_owner_and_cut(task, record_property):
    facts = []
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path(Path(task)), realtime_worker=False,
            runtime_session_id="test:mujoco-tcp-comparison",
            control_domain_id="test:mujoco-tcp-comparison",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False,
        ), allowed_client_ids=(CLIENT,), trace_sink=facts.append,
        enable_safe_tcp_pose=True, enable_tcp_pose_comparison=True,
    )
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        readonly = session.wire_adapter(authenticated_client_id=CLIENT, read_only_access=True)
        profile = session.contract_bundle.motion_profile
        state_parameters = {
            "component_ids": ["right_arm"], "reference_frame_id": profile.reference_frame_id,
            "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": 500,
            "target_operation_id": None,
        }
        initial_request = _request(session, adapter, skill_id="state.read_tcp_pose", version="2",
            parameters=state_parameters, sequence=0)
        before_qpos = session.application.scene.data.qpos.copy()
        before_qvel = session.application.scene.data.qvel.copy()
        before_time = float(session.application.scene.data.time)
        admitted = readonly.submit_read_only_skill_request(initial_request.to_wire())
        assert admitted["accepted"], admitted
        assert session.tcp_pose_comparison_executor.wait_all(timeout_sec=5)
        initial_op = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert initial_op.execution_outcome == "completed", initial_op
        assert initial_op.execution_result["target_comparison"] is None
        np.testing.assert_array_equal(session.application.scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(session.application.scene.data.qvel, before_qvel)
        assert float(session.application.scene.data.time) == before_time

        initial = initial_op.execution_result["pose_by_component"]["right_arm"]
        target = list(initial)
        target[2] += .03
        parameters = {key: getattr(profile, key) for key in (
            "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
            "tool_config_hash", "calibration_hash",
        )}
        parameters.update(controlled_frame_id="tool_tcp", pose_targets_by_component={"right_arm": target})
        move = _request(session, adapter, skill_id="motion.move_tcp_to_pose_safe", version="1",
            parameters=parameters, sequence=1)
        admitted = adapter.submit_skill_request(move.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=45)
        moved = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert moved.execution_outcome == "completed", moved
        original_operation = moved.to_wire()
        compared_request = _request(session, adapter, skill_id="state.read_tcp_pose", version="2",
            parameters={**state_parameters, "target_operation_id": moved.operation_id}, sequence=2)
        before_qpos = session.application.scene.data.qpos.copy()
        before_qvel = session.application.scene.data.qvel.copy()
        before_time = float(session.application.scene.data.time)
        before_facts = len(facts)
        admitted = readonly.submit_read_only_skill_request(compared_request.to_wire())
        assert admitted["accepted"], admitted
        assert session.tcp_pose_comparison_executor.wait_all(timeout_sec=5)
        measured = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert measured.execution_outcome == "completed", measured
        result = measured.execution_result
        comparison = result["target_comparison"]
        current_matrix = xyzquat_to_pose_matrix(result["pose_by_component"]["right_arm"])
        target_matrix = xyzquat_to_pose_matrix(target)
        assert comparison["translation_error_m_by_component"]["right_arm"] == pytest.approx(translation_distance(current_matrix, target_matrix))
        assert comparison["rotation_error_rad_by_component"]["right_arm"] == pytest.approx(rotation_angle_rad(current_matrix, target_matrix))
        assert comparison["target_pose_by_component"]["right_arm"] == target
        assert comparison["target_request_hash"] == session.service.operation_store.request_for_operation(moved.operation_id).semantic_content_hash
        assert comparison["target_status"] == moved.status
        assert comparison["target_execution_outcome"] == moved.execution_outcome
        assert comparison["target_task_outcome"] == moved.task_outcome
        assert comparison["translation_error_m_by_component"]["right_arm"] <= profile.translation_tolerance_m
        assert comparison["rotation_error_rad_by_component"]["right_arm"] <= profile.rotation_tolerance_rad
        assert session.service.operation_store.get(moved.operation_id).to_wire() == original_operation
        np.testing.assert_array_equal(session.application.scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(session.application.scene.data.qvel, before_qvel)
        assert float(session.application.scene.data.time) == before_time
        assert len(facts) == before_facts
        record_property("tcp_comparison_native_evidence_json", json.dumps({
            "task": task, "initial_pose": list(initial), "comparison": dict(comparison),
            "current_measured_pose": list(result["pose_by_component"]["right_arm"]),
            "motion_operation_id": moved.operation_id, "state_operation_id": measured.operation_id,
            "read_changed_qpos_qvel_time_or_dispatch": False, "hardware_connected": False,
        }, default=lambda value: dict(value)))
    finally:
        assert session.close().accepted
