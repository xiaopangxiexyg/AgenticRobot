from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from sim.mujoco.deploy import MujocoSimHILTargetConfig

TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
)
LONG_MOVE_DISTANCE_M = 0.08


def _max_joint_step_deg(rows: list[tuple[float, ...]]) -> float:
    values = np.asarray(rows, dtype=float)
    assert values.ndim == 2 and values.shape[0] >= 2
    return float(np.degrees(np.max(np.abs(np.diff(values, axis=0)))))


@pytest.mark.integration
def test_move_tcp_to_pose_completes_envelope_derived_eight_cm_mujoco_trace() -> None:
    """Exercise Gateway admission through terminal lower HOLD on real MuJoCo."""

    trace_facts = []
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=TASK_CONFIG,
            realtime_worker=False,
            runtime_session_id="test:mujoco-long-move-trace",
            control_domain_id="test:mujoco-long-move-trace",
            execution_observation_plan=(
                mujoco_agent_tcp_pose_execution_observation_plan_v1()
            ),
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=("test:mujoco-long-move",),
        trace_sink=trace_facts.append,
    )
    closed = False
    measured_pose_path: list[tuple[float, ...]] = []
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(
            authenticated_client_id="test:mujoco-long-move"
        )
        observation = adapter.get_observation_snapshot()
        initial_pose = list(
            observation["measurement_summary"]
            ["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        )
        profile = session.contract_bundle.motion_profile

        # Select the sign from the published workspace rather than pinning one
        # absolute scene coordinate.  The magnitude is intentionally larger
        # than the previously exercised 3 cm primitive and remains below the
        # Route's immutable 10 cm relative envelope.
        z_low, z_high = profile.workspace_bounds[4:6]
        if initial_pose[2] + LONG_MOVE_DISTANCE_M <= z_high:
            signed_distance = LONG_MOVE_DISTANCE_M
        elif initial_pose[2] - LONG_MOVE_DISTANCE_M >= z_low:
            signed_distance = -LONG_MOVE_DISTANCE_M
        else:  # pragma: no cover - would indicate an incompatible scene/profile.
            raise AssertionError("published Z workspace cannot contain an 8 cm move")
        assert 0.03 < abs(signed_distance) <= profile.max_translation_delta_m
        target_pose = list(initial_pose)
        target_pose[2] += signed_distance

        original_capture = session.application.capture_tcp_pose_by_component

        def capture_and_record():
            pose_by_component = original_capture()
            measured_pose_path.append(tuple(pose_by_component["right_arm"]))
            return pose_by_component

        session.application.capture_tcp_pose_by_component = capture_and_record
        entry = session.contract_bundle.route_manifest.route(
            "motion.move_tcp_to_pose",
            "1",
        )
        route = entry.route
        parameter_profile = route.allowed_parameter_profiles[0]
        request = AgentSkillRequest(
            client_id="test:mujoco-long-move",
            request_id="request:mujoco-long-move:1",
            idempotency_key="test:mujoco-long-move:0",
            client_sequence=0,
            runtime_session_id=(
                session.contract_bundle.route_manifest.runtime_session_id
            ),
            skill_route_manifest_hash=(
                session.contract_bundle.route_manifest.semantic_content_hash
            ),
            expected_station_revision=int(observation["station_revision"]),
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=(
                entry.deployment_binding.deployment_binding_hash
            ),
            parameter_profile_id=parameter_profile.parameter_profile_id,
            parameter_profile_hash=parameter_profile.parameter_profile_hash,
            parameters={
                "pose_targets_by_component": {"right_arm": target_pose},
                "reference_frame_id": profile.reference_frame_id,
                "reference_frame_spec_hash": profile.reference_frame_spec_hash,
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": profile.frame_graph_hash,
                "tool_config_hash": profile.tool_config_hash,
                "calibration_hash": profile.calibration_hash,
            },
            observation_snapshot_id=observation["snapshot_id"],
            requested_mode="sim_active",
            requested_timeout_ms=10_000,
        )

        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"] is True
        assert session.wait(timeout_sec=15.0)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation is not None
        assert operation.status == "completed", (
            f"status={operation.status}; reason_code={operation.reason_code}; "
            f"task_outcome={operation.task_outcome}; "
            f"initial_pose={initial_pose}; target_pose={target_pose}; "
            f"first_measured_pose={measured_pose_path[:1]}; "
            f"last_measured_pose={measured_pose_path[-1:]}"
        )
        assert operation.reason_code == "tcp_pose_terminal_rule_succeeded"
        assert operation.task_outcome == "succeeded"

        final_observation = adapter.get_observation_snapshot()
        final_pose = np.asarray(
            final_observation["measurement_summary"]
            ["pose_xyz_quat_xyzw_by_component"]["right_arm"],
            dtype=float,
        )
        translation_error_m = float(
            np.linalg.norm(final_pose[:3] - np.asarray(target_pose[:3]))
        )
        assert translation_error_m <= profile.translation_tolerance_m
        assert len(measured_pose_path) >= profile.stable_measurement_count + 2

        facts_by_type = Counter(fact.event_type for fact in trace_facts)
        command_facts = [fact for fact in trace_facts if fact.event_type == "command"]
        dispatch_facts = [
            fact for fact in trace_facts if fact.event_type == "dispatch_receipt"
        ]
        alignment_facts = [
            fact for fact in trace_facts if fact.event_type == "measurement_alignment"
        ]
        safety_facts = [
            fact for fact in trace_facts if fact.event_type == "shared_safety_result"
        ]
        assert len(command_facts) >= 2
        assert len(dispatch_facts) == len(command_facts)
        assert len(alignment_facts) == len(command_facts)
        assert len(safety_facts) == len(command_facts)
        assert all(bool(fact.payload["all_accepted"]) for fact in dispatch_facts)

        reference_qpos = [
            tuple(
                fact.payload["payload"]["output_values_by_effector"]["right_arm"]
            )
            for fact in trace_facts
            if fact.event_type == "component_execution_target"
        ]
        accepted_qcmd = [
            tuple(
                fact.payload["measurement"]["accepted_qcmd_by_side"]["right"]
            )
            for fact in alignment_facts
        ]
        measured_qpos = [
            tuple(fact.payload["measurement"]["measured_qpos_by_side"]["right"])
            for fact in alignment_facts
        ]
        assert len(reference_qpos) == len(command_facts)
        assert len(accepted_qcmd) == len(command_facts)
        assert len(measured_qpos) == len(command_facts)

        # These calculations mirror the first-divergence report used for real
        # traces: reference -> accepted command -> measured plant state.
        motion_report = {
            "requested_translation_m": abs(signed_distance),
            "final_translation_error_m": translation_error_m,
            "control_cycle_count": len(command_facts),
            "reference_max_joint_step_deg": _max_joint_step_deg(reference_qpos),
            "accepted_qcmd_max_joint_step_deg": _max_joint_step_deg(accepted_qcmd),
            "measured_max_joint_step_deg": _max_joint_step_deg(measured_qpos),
            "trace_fact_counts": dict(facts_by_type),
        }
        assert motion_report["requested_translation_m"] == pytest.approx(0.08)
        assert motion_report["reference_max_joint_step_deg"] > 0.0
        assert motion_report["accepted_qcmd_max_joint_step_deg"] > 0.0
        assert motion_report["measured_max_joint_step_deg"] > 0.0

        gate = session.application.runtime.capture_execution_gate_snapshot(
            now=float(session.application.runtime.clock())
        )
        assert session.application.held is True
        assert gate.status == "hold"
        close_outcome = session.close()
        closed = True
        assert close_outcome.accepted is True
    finally:
        if not closed:
            session.close()
