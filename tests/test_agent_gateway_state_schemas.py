from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from agent_gateway.protocol.state_schemas import (
    TCP_POSE_HISTORY_STATE_SKILL_ID_V1,
    TCP_POSE_STATE_SKILL_ID_V1,
    TcpPoseHistoryLimits,
    agent_state_history_slice_content_hash_v1,
    tcp_pose_history_state_definition_v1,
    tcp_pose_history_state_schema_entry_v1,
    tcp_pose_state_definition_v1,
    tcp_pose_state_schema_entry_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
    validate_agent_state_route_schema_v1,
)

COMPONENTS = ("left_arm", "right_arm")


def _history_limits() -> TcpPoseHistoryLimits:
    return TcpPoseHistoryLimits(
        max_samples=8,
        max_window_ms=2_000,
        max_payload_bytes=32_000,
    )


def _route(entry, *, target_kind: str = "sim") -> SkillExecutionRoute:
    return SkillExecutionRoute(
        skill_id=entry.skill_id,
        skill_version=entry.skill_version,
        skill_definition_hash=entry.skill_definition_hash,
        route_id=(
            "sim.mujoco.rm65.pose_observer"
            if target_kind == "sim"
            else "real.rm65.pose_observer"
        ),
        route_version="1",
        parameter_schema_hash=entry.parameter_schema_hash,
        execution_result_schema_hash=entry.execution_result_schema_hash,
        task_evaluation_schema_hash=entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=entry.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(
            ("sim_active",) if target_kind == "sim" else ("real_observe",)
        ),
        target_kind=target_kind,
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
    )


def _pose_entry():
    definition = tcp_pose_state_definition_v1(
        controlled_component_ids=COMPONENTS,
    )
    return definition, tcp_pose_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="arms-pose-read-v1",
        parameter_profile_hash="profile:arms-pose-read:v1",
        allowed_component_ids=COMPONENTS,
        allowed_reference_frame_ids=("robot_base",),
        max_measurement_age_ms=500,
    )


def _history_entry():
    definition = tcp_pose_history_state_definition_v1(
        controlled_component_ids=COMPONENTS,
    )
    return definition, tcp_pose_history_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="arms-pose-history-v1",
        parameter_profile_hash="profile:arms-pose-history:v1",
        allowed_component_ids=COMPONENTS,
        allowed_reference_frame_ids=("robot_base",),
        limits=_history_limits(),
    )


def _measurement(*, sequence: int, measured_at: float) -> dict[str, object]:
    return {
        "history_sequence": sequence,
        "measurement_id": f"mujoco-fk-pose-measurement:{sequence}",
        "measurement_observation_hash": f"measurement-hash:{sequence}",
        "source_measurement_hash": f"qpos-measurement-hash:{sequence}",
        "measured_at": measured_at,
        "available_at": measured_at + 0.01,
        "clock_domain_id": "sim-clock",
        "pose_by_component": {
            "right_arm": [0.4 + sequence * 0.001, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        },
        "pose_schema_hash_by_component": {
            "right_arm": "schema:cartesian-pose:v1",
        },
        "representation_by_component": {"right_arm": "cartesian_pose"},
        "observation_binding_id": "observation-binding:pose-v1",
        "observation_plan_hash": "observation-plan:pose-v1",
    }


def _frame_and_cut() -> dict[str, object]:
    return {
        "observation_cut_id": "observation-cut:pose:12",
        "observation_cut_hash": "observation-cut-hash:12",
        "cutoff_at": 12.0,
        "cutoff_clock_domain_id": "sim-clock",
        "component_ids": ["right_arm"],
        "reference_frame_id": "robot_base",
        "reference_frame_spec_hash": "frame-spec:robot-base:v1",
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "frame-graph:v1",
        "tool_config_hash": "tool-config:v1",
        "calibration_hash": "calibration:v1",
    }


def test_state_definitions_are_target_independent_inspection_skills() -> None:
    pose_definition, _ = _pose_entry()
    history_definition, _ = _history_entry()

    assert pose_definition.skill_id == TCP_POSE_STATE_SKILL_ID_V1
    assert history_definition.skill_id == TCP_POSE_HISTORY_STATE_SKILL_ID_V1
    assert pose_definition.skill_kind == history_definition.skill_kind == "inspection"
    assert pose_definition.controlled_component_ids == COMPONENTS
    for definition in (pose_definition, history_definition):
        serialized = str(definition.to_wire())
        assert "mujoco" not in serialized
        assert "real" not in serialized


def test_tcp_pose_schema_exposes_only_pose_and_required_lineage() -> None:
    _, entry = _pose_entry()
    result_schema = entry.to_wire()["execution_result_schema"]
    required = set(result_schema["required"])
    assert {
        "observation_cut_id",
        "observation_cut_hash",
        "measurement_id",
        "measurement_observation_hash",
        "source_measurement_hash",
        "pose_by_component",
        "pose_schema_hash_by_component",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
    } <= required
    serialized = str(entry.to_wire())
    for forbidden in (
        "qpos",
        "joint_velocity",
        "joint_effort",
        "controller",
        "sdk_method",
        "raw_telemetry",
    ):
        assert forbidden not in serialized


def test_tcp_pose_result_requires_fresh_causally_available_measurement() -> None:
    parameters = {
        "component_ids": ["right_arm"],
        "reference_frame_id": "robot_base",
        "controlled_frame_id": "tool_tcp",
        "max_measurement_age_ms": 250,
    }
    result = {
        **_frame_and_cut(),
        **{key: value for key, value in _measurement(sequence=7, measured_at=11.9).items() if key != "history_sequence"},
    }
    validate_agent_state_parameters_v1(
        schema_kind="tcp_pose",
        parameters=parameters,
        allowed_component_ids=COMPONENTS,
    )
    validate_agent_state_result_v1(
        schema_kind="tcp_pose",
        parameters=parameters,
        result=result,
        allowed_component_ids=COMPONENTS,
    )

    stale = {**result, "measured_at": 11.0, "available_at": 11.01}
    with pytest.raises(ValueError, match="max_measurement_age_ms"):
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters=parameters,
            result=stale,
            allowed_component_ids=COMPONENTS,
        )
    future = {**result, "available_at": 12.1}
    with pytest.raises(ValueError, match="causally available"):
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters=parameters,
            result=future,
            allowed_component_ids=COMPONENTS,
        )


def test_tcp_pose_history_is_bounded_monotonic_and_content_hashed() -> None:
    limits = _history_limits()
    parameters = {
        "component_ids": ["right_arm"],
        "reference_frame_id": "robot_base",
        "controlled_frame_id": "tool_tcp",
        "history_window_ms": 1_000,
        "max_samples": 4,
    }
    result = {
        **_frame_and_cut(),
        "history_source_id": "causal-history:measured-tcp-pose:v1",
        "history_source_contract_hash": "causal-history-contract:v1",
        "window_begin_at": 11.0,
        "window_end_at": 12.0,
        "sampling_policy": "boundary_plus_uniform",
        "sample_count": 2,
        "samples": [
            _measurement(sequence=20, measured_at=11.2),
            _measurement(sequence=21, measured_at=11.8),
        ],
        "truncated": False,
    }
    result["history_slice_hash"] = agent_state_history_slice_content_hash_v1(result)
    validate_agent_state_result_v1(
        schema_kind="tcp_pose_history",
        parameters=parameters,
        result=result,
        allowed_component_ids=COMPONENTS,
        limits=limits,
    )

    reordered = {
        **result,
        "samples": list(reversed(result["samples"])),
    }
    reordered["history_slice_hash"] = agent_state_history_slice_content_hash_v1(
        reordered
    )
    with pytest.raises(ValueError, match="sequences must be strictly increasing"):
        validate_agent_state_result_v1(
            schema_kind="tcp_pose_history",
            parameters=parameters,
            result=reordered,
            allowed_component_ids=COMPONENTS,
            limits=limits,
        )

    tampered = {**result, "truncated": True}
    with pytest.raises(ValueError, match="history_slice_hash"):
        validate_agent_state_result_v1(
            schema_kind="tcp_pose_history",
            parameters=parameters,
            result=tampered,
            allowed_component_ids=COMPONENTS,
            limits=limits,
        )


def test_state_route_must_remain_t0_read_only_without_task_evaluation() -> None:
    _, entry = _pose_entry()
    route = _route(entry)
    validate_agent_state_route_schema_v1(route=route, schema_entry=entry)

    hold_on_cancel = replace(route, cancellation_policy="hold_on_cancel")
    with pytest.raises(ValueError, match="strictly T0 read-only"):
        validate_agent_state_route_schema_v1(
            route=hold_on_cancel,
            schema_entry=entry,
        )
