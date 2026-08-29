"""Target-independent semantics for read-only perception Skills."""

from __future__ import annotations

from agent_gateway.protocol.skill_catalog import SkillDefinition

CAPTURE_DEPTH_SKILL_ID_V1 = "perception.capture_depth"
CAPTURE_RGB_SKILL_ID_V1 = "perception.capture_rgb"
LOCALIZE_OBJECTS_SKILL_ID_V1 = "perception.localize_objects"
PLAN_POINT_GRASP_SKILL_ID_V1 = "perception.plan_point_grasp"


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


def localize_objects_skill_definition_v1() -> SkillDefinition:
    """Describe passive automatic or language-prompted object localization."""

    return SkillDefinition(
        skill_id=LOCALIZE_OBJECTS_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Locate object bounding boxes in one supplied image, either "
            "automatically or using one language prompt, without changing "
            "downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="supplied_image.optional_language_prompt.v1",
        execution_result_semantics_id=(
            "pixel_xyxy_object_boxes_and_annotated_image_artifact.v1"
        ),
        task_outcome_semantics_id="passive_object_localization.v1",
        entry_predicate_ids=("object_localization_model_available.v1",),
        task_outcome_predicate_ids=("object_localization_completed.v1",),
    )


def plan_point_grasp_skill_definition_v1() -> SkillDefinition:
    """Describe passive point-conditioned RGB-D grasp pose planning."""

    return SkillDefinition(
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Plan ranked DMTAC M parallel-jaw tool_tcp poses near one clicked "
            "pixel in an aligned RGB-D frame without moving the robot."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="aligned_rgbd.clicked_region.fixed_camera_calibration.v1",
        execution_result_semantics_id="ranked_camera_and_base_tool_tcp_poses.v1",
        task_outcome_semantics_id="passive_point_grasp_planning.v1",
        entry_predicate_ids=("point_grasp_models_and_calibration_available.v1",),
        task_outcome_predicate_ids=("point_grasp_candidates_generated.v1",),
    )


__all__ = [
    "CAPTURE_DEPTH_SKILL_ID_V1",
    "CAPTURE_RGB_SKILL_ID_V1",
    "LOCALIZE_OBJECTS_SKILL_ID_V1",
    "PLAN_POINT_GRASP_SKILL_ID_V1",
    "capture_depth_skill_definition_v1",
    "capture_rgb_skill_definition_v1",
    "localize_objects_skill_definition_v1",
    "plan_point_grasp_skill_definition_v1",
]
