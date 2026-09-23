"""Target-independent semantics for passive robot-state inspection."""

from __future__ import annotations

from agent_gateway.protocol.skill_catalog import SkillDefinition

READ_GRIPPER_STATE_SKILL_ID_V1 = "state.read_gripper_state"


def read_gripper_state_definition_v1(
    *, controlled_component_ids: tuple[str, ...]
) -> SkillDefinition:
    """Describe one fresh, measured gripper-state projection."""

    return SkillDefinition(
        skill_id=READ_GRIPPER_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Read fresh measured gripper close fractions without commanding "
            "motion or changing downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=controlled_component_ids,
        parameter_semantics_id="measured_gripper_state_query.v1",
        execution_result_semantics_id=(
            "measured_normalized_gripper_close_fraction_projection.v1"
        ),
        task_outcome_semantics_id="robot_state_not_evaluated.v1",
        entry_predicate_ids=("measured_gripper_state_available.v1",),
    )


__all__ = ["READ_GRIPPER_STATE_SKILL_ID_V1", "read_gripper_state_definition_v1"]
