from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from agent_gateway.skills.motion.definition import (
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    INLINE_TCP_TRAJECTORY_SKILL_ID_V1,
    REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1,
    TCP_CARDINAL_STEP_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
    cardinal_tcp_step_skill_definition_v1,
    fixed_level_tcp_orientation_skill_definition_v1,
    tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    InlineTcpTrajectoryLimits,
    cardinal_tcp_step_schema_entry_v1,
    fixed_level_tcp_orientation_schema_entry_v1,
    inline_tcp_trajectory_schema_entry_v1,
    registered_tcp_trajectory_schema_entry_v1,
    tcp_pose_target_schema_entry_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)


def _limits() -> InlineTcpTrajectoryLimits:
    return InlineTcpTrajectoryLimits(
        max_waypoints=4,
        max_duration_ms=1_000,
        min_dt_ms=50,
        max_dt_ms=500,
        max_translation_step_m=0.05,
        max_rotation_step_rad=0.25,
        max_payload_bytes=4_096,
    )


def _route(entry, *, target_kind: str) -> SkillExecutionRoute:
    mode = "sim_active" if target_kind == "sim" else "real_active"
    tier = "t1_sim_or_no_motion" if target_kind == "sim" else "t2_real_motion"
    route_id = (
        "sim.mujoco.rm65_inspire.motion_executor"
        if target_kind == "sim"
        else "real.rm65.motion_executor"
    )
    return SkillExecutionRoute(
        skill_id=entry.skill_id,
        skill_version=entry.skill_version,
        skill_definition_hash=entry.skill_definition_hash,
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=entry.parameter_schema_hash,
        execution_result_schema_hash=entry.execution_result_schema_hash,
        task_evaluation_schema_hash=entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=entry.parameter_profiles,
        control_effect_class="motion",
        privilege_tier=tier,
        supported_modes=(mode,),
        target_kind=target_kind,
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=target_kind == "real",
        requires_deadman=False,
        max_operation_duration_ms=5_000,
        cancellation_policy="hold_on_disconnect_or_timeout",
    )


def test_tcp_pose_target_is_absolute_tool_tcp_proposal_without_low_level_api() -> None:
    entry = tcp_pose_target_schema_entry_v1(
        skill_definition_hash="definition:tcp-pose:v1",
        parameter_profile_id="arms-only-safe-v1",
        parameter_profile_hash="profile:arms-only-safe-v1",
        allowed_component_ids=("left_arm", "right_arm"),
        allowed_reference_frame_ids=("robot_base",),
        max_requested_duration_ms=2_000,
        workspace_bounds_by_component={
            "right_arm": (-0.8, 0.8, -0.8, 0.8, 0.0, 1.2),
        },
    )
    assert entry.skill_id == TCP_POSE_TARGET_SKILL_ID_V1
    schema = entry.to_wire()["parameter_schema"]
    assert schema["properties"]["controlled_frame_id"] == {"const": "tool_tcp"}
    serialized = str(schema)
    for forbidden in ("qpos", "joint_velocity", "joint_effort", "sdk_method"):
        assert forbidden not in serialized

    parameters = {
        "controlled_frame_id": "tool_tcp",
        "pose_targets_by_component": {
            "right_arm": [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0],
        },
    }
    validate_agent_motion_parameters_v1(
        schema_kind="tcp_pose_target",
        parameters=parameters,
        allowed_component_ids=("left_arm", "right_arm"),
    )
    with pytest.raises(ValueError, match="normalized quat_xyzw"):
        validate_agent_motion_parameters_v1(
            schema_kind="tcp_pose_target",
            parameters={
                **parameters,
                "pose_targets_by_component": {
                    "right_arm": [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 2.0],
                },
            },
            allowed_component_ids=("left_arm", "right_arm"),
        )


def test_tcp_pose_profile_can_explicitly_exclude_duration_semantics() -> None:
    entry = tcp_pose_target_schema_entry_v1(
        skill_definition_hash="definition:tcp-pose-no-duration:v1",
        parameter_profile_id="right-arm-no-duration-v1",
        parameter_profile_hash="profile:right-arm-no-duration-v1",
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=("right_arm_base",),
        max_requested_duration_ms=None,
        workspace_bounds_by_component={
            "right_arm": (-0.6, 0.6, -0.8, 0.8, 0.0, 0.9),
        },
    )
    schema = entry.to_wire()["parameter_schema"]

    assert "requested_duration_ms" not in schema["properties"]
    assert "requested_duration_ms" not in schema["required"]
    assert schema["additionalProperties"] is False


def test_cardinal_tcp_step_exposes_direction_but_pins_lower_profile_facts() -> None:
    entry = cardinal_tcp_step_schema_entry_v1(
        skill_definition_hash="definition:tcp-cardinal-step:v1",
        parameter_profile_id="right-arm-cardinal-short-step-v1",
        parameter_profile_hash="profile:right-arm-cardinal-short-step-v1",
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=("right_arm_base",),
        reference_frame_spec_hash="frame-spec:right-arm-base",
        frame_graph_hash="frame-graph:fixture",
        tool_config_hash="tool-config:fixture",
        calibration_hash="calibration:fixture",
        primitive_catalog_hash="primitive-catalog:fixture",
    )
    assert entry.skill_id == TCP_CARDINAL_STEP_SKILL_ID_V1
    schema = entry.to_wire()["parameter_schema"]
    assert tuple(schema["properties"]["direction"]["enum"]) == (
        CARDINAL_TCP_STEP_DIRECTIONS_V1
    )
    assert "distance_m" not in schema["properties"]
    assert schema["properties"]["primitive_catalog_hash"] == {
        "const": "primitive-catalog:fixture"
    }
    parameters = {
        "component_id": "right_arm",
        "direction": "up",
        "reference_frame_id": "right_arm_base",
        "reference_frame_spec_hash": "frame-spec:right-arm-base",
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "frame-graph:fixture",
        "tool_config_hash": "tool-config:fixture",
        "calibration_hash": "calibration:fixture",
        "primitive_catalog_hash": "primitive-catalog:fixture",
    }
    validate_agent_motion_parameters_v1(
        schema_kind="cardinal_tcp_step",
        parameters=parameters,
        allowed_component_ids=("right_arm",),
    )
    with pytest.raises(ValueError, match="unsupported direction"):
        validate_agent_motion_parameters_v1(
            schema_kind="cardinal_tcp_step",
            parameters={**parameters, "direction": "diagonal"},
            allowed_component_ids=("right_arm",),
        )


def test_fixed_level_orientation_exposes_no_relative_rotation_parameter() -> None:
    entry = fixed_level_tcp_orientation_schema_entry_v1(
        skill_id=TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
        skill_definition_hash="definition:tcp-level-left-30:v1",
        parameter_profile_id="right-arm-fixed-level-orientation-v1",
        parameter_profile_hash="profile:right-arm-fixed-level-orientation-v1",
        allowed_component_ids=("left_arm", "right_arm"),
        allowed_reference_frame_ids=("left_arm_base", "right_arm_base"),
        reference_frame_spec_hash="frame-spec:right-arm-base",
        frame_graph_hash="frame-graph:fixture",
        tool_config_hash="tool-config:fixture",
        calibration_hash="calibration:fixture",
        primitive_hash="primitive:fixed-level-left-30",
    )
    schema = entry.parameter_schema
    assert entry.skill_id == TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1
    assert schema["additionalProperties"] is False
    assert schema["properties"]["component_id"]["enum"] == (
        "left_arm",
        "right_arm",
    )
    assert {
        "angle",
        "angle_deg",
        "delta_rotation",
        "heading",
        "rotation",
        "yaw",
        "yaw_deg",
    }.isdisjoint(schema["properties"])
    parameters = {
        "component_id": "right_arm",
        "reference_frame_id": "right_arm_base",
        "reference_frame_spec_hash": "frame-spec:right-arm-base",
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "frame-graph:fixture",
        "tool_config_hash": "tool-config:fixture",
        "calibration_hash": "calibration:fixture",
        "primitive_hash": "primitive:fixed-level-left-30",
    }
    validate_agent_motion_parameters_v1(
        schema_kind="fixed_level_tcp_orientation",
        parameters=parameters,
        allowed_component_ids=("left_arm", "right_arm"),
    )
    validate_agent_motion_parameters_v1(
        schema_kind="fixed_level_tcp_orientation",
        parameters={**parameters, "component_id": "left_arm"},
        allowed_component_ids=("left_arm", "right_arm"),
    )
    with pytest.raises(ValueError, match="cannot carry an angle"):
        validate_agent_motion_parameters_v1(
            schema_kind="fixed_level_tcp_orientation",
            parameters={**parameters, "yaw_deg": 5.0},
            allowed_component_ids=("left_arm", "right_arm"),
        )


def test_fixed_level_orientation_definitions_select_either_arm_absolutely() -> None:
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
        definition = fixed_level_tcp_orientation_skill_definition_v1(
            skill_id=skill_id,
        )
        summary = definition.summary.lower()
        assert definition.controlled_component_ids == (
            "left_arm",
            "right_arm",
        )
        assert "selected left or right arm" in summary
        assert "horizontal desk plane" in summary
        assert "fixed absolute" in summary
        assert "not a relative rotation" in summary
        assert "do not accumulate yaw" in summary


def test_motion_skill_definition_hashes_survive_the_package_migration() -> None:
    definitions = (
        tcp_pose_target_skill_definition_v1(
            controlled_component_ids=("right_arm",),
        ),
        cardinal_tcp_step_skill_definition_v1(
            controlled_component_ids=("right_arm",),
        ),
        *(
            fixed_level_tcp_orientation_skill_definition_v1(skill_id=skill_id)
            for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        ),
    )

    assert {
        definition.skill_id: definition.semantic_content_hash
        for definition in definitions
    } == {
        "motion.move_tcp_to_pose": (
            "bbd59b2858b986b695d9a1e47bd057b37be875dc018286050cb9c6d37a19663c"
        ),
        "motion.step_tcp_cardinal": (
            "c6e28fbde6b016a48d7510debf1465d90b6073f5160bd3885f852ab0811bba89"
        ),
        "motion.set_tcp_level_forward": (
            "6e263d603c35da25f11f6ff6cfd0d5a2896e542d04a4bded262fd1bd648de351"
        ),
        "motion.set_tcp_level_forward_left_30": (
            "a1faeff5337562453b64301284a8ba3500b8bf76d822eadaaac28f4c13acba34"
        ),
        "motion.set_tcp_level_forward_right_30": (
            "0d8852424cf94bf5e1b75caedc642be037c2a0393d7137ba9e0a6566cb8133a5"
        ),
    }


def test_registered_real_trajectory_schema_accepts_only_immutable_reference() -> None:
    entry = registered_tcp_trajectory_schema_entry_v1(
        skill_definition_hash="definition:registered-trajectory:v1",
        parameter_profile_id="real-history-safe-v1",
        parameter_profile_hash="profile:real-history-safe-v1",
        allowed_schedule_profile_ids=("native-safe-v1",),
        allowed_time_scale_profile_ids=("half-speed-v1",),
    )
    assert entry.skill_id == REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1
    properties = entry.to_wire()["parameter_schema"]["properties"]
    for forbidden in ("waypoints", "path", "file_path", "raw_actions"):
        assert forbidden not in properties
    required_lineage = {
        "trajectory_id",
        "trajectory_hash",
        "revision",
        "source_contract_id",
        "source_contract_hash",
        "source_frame_contract_hash",
        "source_component_schema_hash",
        "sample_begin",
        "sample_end_exclusive",
        "schedule_profile_id",
        "schedule_profile_hash",
    }
    assert required_lineage <= set(
        entry.to_wire()["parameter_schema"]["required"]
    )
    validate_agent_motion_parameters_v1(
        schema_kind="registered_tcp_pose_trajectory",
        parameters={"sample_begin": 2, "sample_end_exclusive": 8},
    )
    with pytest.raises(ValueError, match="cannot carry inline/path"):
        validate_agent_motion_parameters_v1(
            schema_kind="registered_tcp_pose_trajectory",
            parameters={
                "sample_begin": 2,
                "sample_end_exclusive": 8,
                "waypoints": [],
            },
        )


def test_inline_trajectory_is_separate_bounded_sim_only_schema() -> None:
    limits = _limits()
    entry = inline_tcp_trajectory_schema_entry_v1(
        skill_definition_hash="definition:inline-trajectory:v1",
        parameter_profile_id="sim-inline-safe-v1",
        parameter_profile_hash="profile:sim-inline-safe-v1",
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=("robot_base",),
        limits=limits,
    )
    assert entry.skill_id == INLINE_TCP_TRAJECTORY_SKILL_ID_V1
    waypoint_schema = entry.to_wire()["parameter_schema"]["properties"]["waypoints"]
    assert waypoint_schema["maxItems"] == limits.max_waypoints
    sim_route = _route(entry, target_kind="sim")
    validate_agent_motion_route_schema_v1(route=sim_route, schema_entry=entry)

    real_route = _route(entry, target_kind="real")
    with pytest.raises(ValueError, match="sim-only"):
        validate_agent_motion_route_schema_v1(
            route=real_route,
            schema_entry=entry,
        )


def test_inline_trajectory_cross_field_limits_reject_stream_like_or_jump_input() -> None:
    limits = _limits()
    base = {
        "controlled_frame_id": "tool_tcp",
        "waypoints": [
            {
                "timestamp_ms": 0,
                "pose_targets_by_component": {
                    "right_arm": [0.30, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "timestamp_ms": 100,
                "pose_targets_by_component": {
                    "right_arm": [0.32, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
    }
    validate_agent_motion_parameters_v1(
        schema_kind="inline_tcp_pose_trajectory",
        parameters=base,
        allowed_component_ids=("right_arm",),
        limits=limits,
    )
    too_fast = replace(limits, min_dt_ms=150)
    with pytest.raises(ValueError, match="dt is outside"):
        validate_agent_motion_parameters_v1(
            schema_kind="inline_tcp_pose_trajectory",
            parameters=base,
            allowed_component_ids=("right_arm",),
            limits=too_fast,
        )
    jump = {
        **base,
        "waypoints": [
            base["waypoints"][0],
            {
                "timestamp_ms": 100,
                "pose_targets_by_component": {
                    "right_arm": [0.50, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
    }
    with pytest.raises(ValueError, match="translation step exceeds"):
        validate_agent_motion_parameters_v1(
            schema_kind="inline_tcp_pose_trajectory",
            parameters=jump,
            allowed_component_ids=("right_arm",),
            limits=limits,
        )
