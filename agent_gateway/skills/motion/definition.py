"""Target-independent Agent motion Skill semantics.

This module owns Agent-visible Skill identities and definitions.  It does not
select a Route, device, reference frame, primitive distance, safety envelope,
IK implementation, dispatcher, or execution profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from agent_gateway.protocol.skill_catalog import SkillDefinition

TCP_POSE_TARGET_SKILL_ID_V1 = "motion.move_tcp_to_pose"
TCP_CARDINAL_STEP_SKILL_ID_V1 = "motion.step_tcp_cardinal"
TCP_STEP_FORWARD_SKILL_ID_V1 = "motion.step_tcp_forward"
TCP_STEP_BACKWARD_SKILL_ID_V1 = "motion.step_tcp_backward"
TCP_STEP_LEFT_SKILL_ID_V1 = "motion.step_tcp_left"
TCP_STEP_RIGHT_SKILL_ID_V1 = "motion.step_tcp_right"
TCP_STEP_UP_SKILL_ID_V1 = "motion.step_tcp_up"
TCP_STEP_DOWN_SKILL_ID_V1 = "motion.step_tcp_down"
TCP_LEVEL_FORWARD_SKILL_ID_V1 = "motion.set_tcp_level_forward"
TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1 = "motion.set_tcp_level_forward_left_30"
TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1 = "motion.set_tcp_level_forward_right_30"
REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1 = (
    "motion.replay_registered_tcp_pose_trajectory"
)
INLINE_TCP_TRAJECTORY_SKILL_ID_V1 = "motion.replay_inline_tcp_pose_trajectory"

CARDINAL_TCP_STEP_DIRECTIONS_V1 = (
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
)

FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1: Mapping[str, str] = (
    MappingProxyType(
        {
            "forward": TCP_STEP_FORWARD_SKILL_ID_V1,
            "backward": TCP_STEP_BACKWARD_SKILL_ID_V1,
            "left": TCP_STEP_LEFT_SKILL_ID_V1,
            "right": TCP_STEP_RIGHT_SKILL_ID_V1,
            "up": TCP_STEP_UP_SKILL_ID_V1,
            "down": TCP_STEP_DOWN_SKILL_ID_V1,
        }
    )
)
FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1: Mapping[str, str] = (
    MappingProxyType(
        {
            skill_id: direction
            for direction, skill_id in (
                FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1.items()
            )
        }
    )
)
FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1 = tuple(
    FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1[direction]
    for direction in CARDINAL_TCP_STEP_DIRECTIONS_V1
)

FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1 = (
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
)
FIXED_LEVEL_TCP_ORIENTATION_SKILL_SUMMARY_BY_ID_V1: Mapping[str, str] = (
    MappingProxyType(
        {
            TCP_LEVEL_FORWARD_SKILL_ID_V1: (
                "Keep the selected left or right arm tool_tcp position fixed "
                "and set the gripper in the horizontal desk plane at the "
                "fixed absolute forward heading (0 deg). This is not a "
                "relative rotation; repeated calls do not accumulate yaw."
            ),
            TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: (
                "Keep the selected left or right arm tool_tcp position fixed "
                "and set the gripper in the horizontal desk plane at the "
                "fixed absolute forward-left 30 deg heading. This is not a "
                "relative rotation; repeated calls do not accumulate yaw."
            ),
            TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: (
                "Keep the selected left or right arm tool_tcp position fixed "
                "and set the gripper in the horizontal desk plane at the "
                "fixed absolute forward-right 30 deg heading. This is not a "
                "relative rotation; repeated calls do not accumulate yaw."
            ),
        }
    )
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if normalized != value:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _component_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("controlled_component_ids must be a non-empty tuple.")
    normalized = tuple(
        _required(item, field_name="controlled_component_id") for item in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("controlled_component_ids must be unique.")
    return tuple(sorted(normalized))


def tcp_pose_target_skill_definition_v1(
    *,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Return the bounded absolute TCP-pose proposal Skill definition."""

    components = _component_ids(controlled_component_ids)
    summary = (
        f"Move {components[0]} tool_tcp to one bounded absolute pose proposal."
        if len(components) == 1
        else (
            "Move one selected arm tool_tcp to one bounded absolute pose "
            "proposal."
        )
    )
    return SkillDefinition(
        skill_id=TCP_POSE_TARGET_SKILL_ID_V1,
        skill_version="1",
        skill_kind="motion",
        summary=summary,
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=components,
        parameter_semantics_id="tcp_pose_target.absolute.xyz_quat_xyzw.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def cardinal_tcp_step_skill_definition_v1(
    *,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Return the semantic bounded cardinal TCP-step Skill definition."""

    return SkillDefinition(
        skill_id=TCP_CARDINAL_STEP_SKILL_ID_V1,
        skill_version="1",
        skill_kind="motion",
        summary=(
            "Move one tool_tcp by the reviewed lower short-step primitive in "
            "a semantic workspace direction."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=_component_ids(controlled_component_ids),
        parameter_semantics_id="tcp_cardinal_short_step.direction.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def fixed_cardinal_tcp_step_skill_definition_v1(
    *,
    skill_id: str,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Return one fixed-direction bounded TCP-step Skill definition."""

    selected_skill_id = _required(skill_id, field_name="skill_id")
    try:
        direction = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[
            selected_skill_id
        ]
    except KeyError as exc:
        raise ValueError(
            "skill_id is not a fixed cardinal TCP-step Skill."
        ) from exc
    return SkillDefinition(
        skill_id=selected_skill_id,
        skill_version="1",
        skill_kind="motion",
        summary=(
            "Move one tool_tcp by the reviewed lower 3 cm short-step "
            f"primitive in the fixed semantic {direction} direction. The "
            "caller cannot select another direction or distance."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=_component_ids(controlled_component_ids),
        parameter_semantics_id=(
            f"tcp_fixed_cardinal_short_step.{direction}.no_distance.v1"
        ),
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def fixed_level_tcp_orientation_skill_definition_v1(
    *,
    skill_id: str,
) -> SkillDefinition:
    """Return one arm-selectable fixed absolute orientation definition."""

    selected_skill_id = _required(skill_id, field_name="skill_id")
    try:
        summary = FIXED_LEVEL_TCP_ORIENTATION_SKILL_SUMMARY_BY_ID_V1[
            selected_skill_id
        ]
    except KeyError as exc:
        raise ValueError(
            "skill_id is not a fixed level TCP orientation Skill."
        ) from exc
    return SkillDefinition(
        skill_id=selected_skill_id,
        skill_version="1",
        skill_kind="motion",
        summary=summary,
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=("left_arm", "right_arm"),
        parameter_semantics_id=(
            "tcp_fixed_absolute_horizontal_heading.no_relative_angle.v1"
        ),
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


__all__ = [
    "CARDINAL_TCP_STEP_DIRECTIONS_V1",
    "FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1",
    "FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1",
    "FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1",
    "FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1",
    "FIXED_LEVEL_TCP_ORIENTATION_SKILL_SUMMARY_BY_ID_V1",
    "INLINE_TCP_TRAJECTORY_SKILL_ID_V1",
    "REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1",
    "TCP_CARDINAL_STEP_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_SKILL_ID_V1",
    "TCP_POSE_TARGET_SKILL_ID_V1",
    "TCP_STEP_BACKWARD_SKILL_ID_V1",
    "TCP_STEP_DOWN_SKILL_ID_V1",
    "TCP_STEP_FORWARD_SKILL_ID_V1",
    "TCP_STEP_LEFT_SKILL_ID_V1",
    "TCP_STEP_RIGHT_SKILL_ID_V1",
    "TCP_STEP_UP_SKILL_ID_V1",
    "cardinal_tcp_step_skill_definition_v1",
    "fixed_cardinal_tcp_step_skill_definition_v1",
    "fixed_level_tcp_orientation_skill_definition_v1",
    "tcp_pose_target_skill_definition_v1",
]
