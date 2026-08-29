"""Target-independent semantics for read-only perception Skills."""

from __future__ import annotations

from agent_gateway.protocol.skill_catalog import SkillDefinition

CAPTURE_DEPTH_SKILL_ID_V1 = "perception.capture_depth"
CAPTURE_RGB_SKILL_ID_V1 = "perception.capture_rgb"


def capture_depth_skill_definition_v1() -> SkillDefinition:
    """Describe one passive, role-selected depth capture."""

    return SkillDefinition(
        skill_id=CAPTURE_DEPTH_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Capture the latest depth frame for one deployment-registered "
            "semantic role without changing downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="registered_role.latest_depth_frame.v1",
        execution_result_semantics_id=(
            "registered_role.depth_frame_metadata_and_artifact.v1"
        ),
        task_outcome_semantics_id="passive_depth_capture.v1",
        entry_predicate_ids=("registered_depth_role_available.v1",),
        task_outcome_predicate_ids=("depth_frame_captured.v1",),
    )


def capture_depth_skill_definition_v2() -> SkillDefinition:
    """Describe depth capture with an explicit consumer reference frame."""

    return SkillDefinition(
        skill_id=CAPTURE_DEPTH_SKILL_ID_V1,
        skill_version="2",
        skill_kind="inspection",
        summary=(
            "Capture the latest depth frame for one deployment-registered "
            "semantic role and report its camera pose in one explicit "
            "deployment-bound reference frame without changing downstream "
            "control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="registered_role.latest_depth_frame.v1",
        execution_result_semantics_id=(
            "registered_role.depth_frame_artifact_and_reference_pose.v2"
        ),
        task_outcome_semantics_id="passive_depth_capture.v1",
        entry_predicate_ids=("registered_depth_role_available.v1",),
        task_outcome_predicate_ids=("depth_frame_captured.v1",),
    )


def capture_rgb_skill_definition_v1() -> SkillDefinition:
    """Describe one passive, role-selected RGB scene cut."""

    return SkillDefinition(
        skill_id=CAPTURE_RGB_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Capture the latest RGB frames for deployment-registered semantic "
            "roles without changing downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="registered_roles.latest_rgb_scene_cut.v1",
        execution_result_semantics_id=(
            "registered_roles.rgb_frame_metadata_and_artifacts.v1"
        ),
        task_outcome_semantics_id="passive_rgb_capture.v1",
        entry_predicate_ids=("registered_visual_roles_available.v1",),
        task_outcome_predicate_ids=("rgb_frames_captured.v1",),
    )


__all__ = [
    "CAPTURE_DEPTH_SKILL_ID_V1",
    "CAPTURE_RGB_SKILL_ID_V1",
    "capture_depth_skill_definition_v1",
    "capture_depth_skill_definition_v2",
    "capture_rgb_skill_definition_v1",
]
