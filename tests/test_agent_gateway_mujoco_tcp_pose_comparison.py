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
from agent_gateway.profiles.mujoco.motion_skill import (
    mujoco_agent_skill_hold_evidence_from_transition,
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
@pytest.mark.parametrize("motion_side", ("right", "left"))
def test_safe_motion_then_fresh_comparison_uses_the_same_owner_and_cut(
    task, motion_side, monkeypatch, record_property,
):
    facts = []
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path(Path(task)), realtime_worker=False,
            runtime_session_id="test:mujoco-tcp-comparison",
            control_domain_id="test:mujoco-tcp-comparison",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(
                motion_side=motion_side,
            ),
            exposed_pose_sides=(motion_side,), allow_scripted_task_reset=False,
        ), allowed_client_ids=(CLIENT,), trace_sink=facts.append,
        enable_safe_tcp_pose=True, enable_tcp_pose_comparison=True, motion_side=motion_side,
    )
    try:
        application = session.application
        component = application.component_id
        assert application.motion_side == motion_side
        assert component == motion_side + "_arm"
        captures, holds = [], []
        capture_source = session.observation_producer._capture
        request_hold = application.request_hold

        def record_capture(*, active_operation_id):
            # Observe the real producer return without adding a second read.
            capture = capture_source(active_operation_id=active_operation_id)
            captures.append((active_operation_id, capture))
            return capture

        def record_hold(**kwargs):
            transition = request_hold(**kwargs)
            holds.append((kwargs.get("operation_id"), transition))
            return transition

        monkeypatch.setattr(session.observation_producer, "_capture", record_capture)
        monkeypatch.setattr(application, "request_hold", record_hold)
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        readonly = session.wire_adapter(authenticated_client_id=CLIENT, read_only_access=True)
        profile = session.contract_bundle.motion_profile
        assert profile.component_id == component
        state_parameters = {
            "component_ids": [component], "reference_frame_id": profile.reference_frame_id,
            "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": 500,
            "target_operation_id": None,
        }

        def assert_source_cut(operation):
            owned = [capture for operation_id, capture in captures
                     if operation_id == operation.operation_id]
            assert len(owned) == 1, "Each state operation must use exactly one producer cut."
            capture = owned[0]
            cut, measurement = capture.observation_cut, capture.measurement
            assert cut is not None and measurement is not None
            result = operation.execution_result
            assert tuple(result["component_ids"]) == (component,)
            assert set(cut.selected_sample_by_stream) == {f"robot_state.{component}.tool_tcp_pose"}
            sample = cut.selected_sample_by_stream[f"robot_state.{component}.tool_tcp_pose"]
            assert cut.runtime_session_id == operation.runtime_session_id
            assert cut.control_domain_id == application.runtime.config.control_domain_id
            assert result["observation_cut_id"] == cut.observation_cut_id
            assert result["observation_cut_hash"] == cut.semantic_content_hash
            assert result["cutoff_at"] == cut.cutoff_at
            assert result["cutoff_clock_domain_id"] == cut.cutoff_clock_domain_id
            assert result["clock_domain_id"] == application.runtime.config.clock_domain_id
            assert result["measurement_id"] == measurement.measurement_id == sample.sample_id
            assert result["measurement_observation_hash"] == measurement.semantic_content_hash
            assert result["source_measurement_hash"] == measurement.source_measurement_hash
            assert result["source_measurement_hash"] == capture.snapshot.measurement_summary["measurement_hash"]
            assert result["measured_at"] == measurement.measured_at == capture.snapshot.captured_at == sample.measured_at
            assert result["available_at"] == measurement.available_at == sample.available_at
            assert result["measured_at"] <= result["available_at"] <= result["cutoff_at"]
            assert set(result["pose_by_component"]) == {component}
            assert tuple(result["pose_by_component"][component]) == tuple(measurement.value_by_component[component])
            assert result["pose_schema_hash_by_component"] == {
                component: measurement.schema_hash_by_component[component],
            }
            assert capture.snapshot.measurement_valid
            assert operation.evaluation_status == "not_requested"
            assert operation.task_outcome == "not_assessed"
            return cut

        initial_request = _request(session, adapter, skill_id="state.read_tcp_pose", version="2",
            parameters=state_parameters, sequence=0)
        before_qpos = application.scene.data.qpos.copy()
        before_qvel = application.scene.data.qvel.copy()
        before_time = float(application.scene.data.time)
        before_facts = len(facts)
        admitted = readonly.submit_read_only_skill_request(initial_request.to_wire())
        assert admitted["accepted"], admitted
        assert session.tcp_pose_comparison_executor.wait_all(timeout_sec=5)
        initial_op = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert initial_op.execution_outcome == "completed", initial_op
        assert initial_op.execution_result["target_comparison"] is None
        previous_cut = assert_source_cut(initial_op)
        np.testing.assert_array_equal(application.scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(application.scene.data.qvel, before_qvel)
        assert float(application.scene.data.time) == before_time
        assert len(facts) == before_facts

        initial = list(initial_op.execution_result["pose_by_component"][component])
        raised = list(initial)
        raised[2] += .03
        phases, original_operations = [], {}
        for phase, target in enumerate((raised, list(initial)), start=1):
            parameters = {key: getattr(profile, key) for key in (
                "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
                "tool_config_hash", "calibration_hash",
            )}
            parameters.update(controlled_frame_id="tool_tcp", pose_targets_by_component={component: target})
            move = _request(session, adapter, skill_id="motion.move_tcp_to_pose_safe", version="1",
                parameters=parameters, sequence=2 * phase - 1)
            motion_trace_start = len(facts)
            admitted = adapter.submit_skill_request(move.to_wire())
            assert admitted["accepted"], admitted
            assert session.wait(timeout_sec=45)
            moved = session.service.operation_store.get(admitted["operation"]["operation_id"])
            assert moved.status == "completed" and moved.execution_outcome == "completed", moved
            assert moved.task_outcome == "succeeded", moved
            assert application.held and not application.target.planned_tcp_motion.active
            terminal_holds = [transition for operation_id, transition in holds
                              if operation_id == moved.operation_id]
            assert terminal_holds
            hold = mujoco_agent_skill_hold_evidence_from_transition(
                terminal_holds[-1], lower_operation_id=moved.lower_operation_id,
            )
            assert hold.succeeded and hold.lower_hold_confirmed and hold.evidence_refs
            assert set(hold.evidence_refs).issubset(moved.execution_receipt_refs)
            command_facts = [fact for fact in facts[motion_trace_start:] if fact.event_type == "command"]
            assert command_facts, "Both nonzero phases must execute, not only complete an operation."
            original_operations[moved.operation_id] = moved.to_wire()
            compared_request = _request(session, adapter, skill_id="state.read_tcp_pose", version="2",
                parameters={**state_parameters, "target_operation_id": moved.operation_id}, sequence=2 * phase)
            before_qpos = application.scene.data.qpos.copy()
            before_qvel = application.scene.data.qvel.copy()
            before_time = float(application.scene.data.time)
            before_facts = len(facts)
            admitted = readonly.submit_read_only_skill_request(compared_request.to_wire())
            assert admitted["accepted"], admitted
            assert session.tcp_pose_comparison_executor.wait_all(timeout_sec=5)
            measured = session.service.operation_store.get(admitted["operation"]["operation_id"])
            assert measured.execution_outcome == "completed", measured
            result = measured.execution_result
            cut = assert_source_cut(measured)
            assert cut.observation_cut_id != previous_cut.observation_cut_id
            assert cut.cutoff_at > previous_cut.cutoff_at
            assert result["measured_at"] >= moved.updated_at
            previous_cut = cut
            comparison = result["target_comparison"]
            current_matrix = xyzquat_to_pose_matrix(result["pose_by_component"][component])
            target_matrix = xyzquat_to_pose_matrix(target)
            translation_error = translation_distance(current_matrix, target_matrix)
            rotation_error = rotation_angle_rad(current_matrix, target_matrix)
            assert comparison["translation_error_m_by_component"][component] == pytest.approx(translation_error)
            assert comparison["rotation_error_rad_by_component"][component] == pytest.approx(rotation_error)
            assert translation_error <= profile.translation_tolerance_m
            assert rotation_error <= profile.rotation_tolerance_rad
            retained = session.service.operation_store.request_for_operation(moved.operation_id)
            assert dict(comparison["target_pose_by_component"]) == dict(retained.parameters["pose_targets_by_component"])
            assert list(comparison["target_pose_by_component"][component]) == target
            assert comparison["target_operation_id"] == moved.operation_id
            assert comparison["target_request_hash"] == retained.semantic_content_hash == move.semantic_content_hash
            assert comparison["target_route_hash"] == move.route_hash
            assert comparison["target_skill_version"] == "1"
            assert comparison["target_status"] == moved.status
            assert comparison["target_execution_outcome"] == moved.execution_outcome
            assert comparison["target_task_outcome"] == moved.task_outcome
            for key in ("reference_frame_id", "reference_frame_spec_hash", "controlled_frame_id",
                        "frame_graph_hash", "tool_config_hash", "calibration_hash"):
                assert result[key] == retained.parameters[key]
            for operation_id, original_operation in original_operations.items():
                assert session.service.operation_store.get(operation_id).to_wire() == original_operation
            np.testing.assert_array_equal(application.scene.data.qpos, before_qpos)
            np.testing.assert_array_equal(application.scene.data.qvel, before_qvel)
            assert float(application.scene.data.time) == before_time
            assert len(facts) == before_facts
            phases.append({
                "phase": phase, "motion_operation_id": moved.operation_id,
                "state_operation_id": measured.operation_id, "comparison": dict(comparison),
                "current_measured_pose": list(result["pose_by_component"][component]),
                "translation_error_m": translation_error, "rotation_error_rad": rotation_error,
                "observation_cut_id": cut.observation_cut_id, "observation_cut_hash": cut.semantic_content_hash,
                "measurement_observation_hash": result["measurement_observation_hash"],
                "measured_at": result["measured_at"], "cutoff_at": cut.cutoff_at,
                "command_fact_ids": [fact.fact_id for fact in command_facts],
                "lower_hold_confirmed": hold.lower_hold_confirmed, "hold_evidence_refs": hold.evidence_refs,
            })
        assert len(session.safe_tcp_pose_executor.workers) == len(original_operations) == 2
        assert len({phase["state_operation_id"] for phase in phases}) == 2
        record_property("tcp_comparison_native_evidence_json", json.dumps({
            "task": task, "motion_side": motion_side, "component_id": component,
            "initial_pose": initial, "phases": phases,
            "read_changed_qpos_qvel_time_or_dispatch": False, "hardware_connected": False,
        }, default=lambda value: dict(value)))
    finally:
        assert session.close().accepted
