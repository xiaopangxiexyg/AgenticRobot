"""Target-independent semantics for read-only perception Skills."""

from __future__ import annotations

from agent_gateway.protocol.skill_catalog import SkillDefinition

CAPTURE_DEPTH_SKILL_ID_V1 = "perception.capture_depth"
CAPTURE_RGB_SKILL_ID_V1 = "perception.capture_rgb"
LOCALIZE_OBJECTS_SKILL_ID_V1 = "perception.localize_objects"
PLAN_POINT_GRASP_SKILL_ID_V1 = "perception.plan_point_grasp"
RENDER_POINT_GRASP_CANDIDATE_SKILL_ID_V1 = "perception.render_point_grasp_candidate"
SEGMENT_SKILL_ID_V1 = "perception.segment"


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


def capture_rgb_skill_definition_v2() -> SkillDefinition:
    """Describe one passive, bounded multi-camera RGB capture window."""

    return SkillDefinition(
        skill_id=CAPTURE_RGB_SKILL_ID_V1,
        skill_version="2",
        skill_kind="inspection",
        summary=(
            "Capture deployment-registered RGB roles in one bounded capture "
            "window, preserving ordered roles, per-frame acquisition times, "
            "and the admitted Station revision without changing downstream "
            "control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="registered_roles.latest_rgb_scene_cut.v1",
        execution_result_semantics_id=(
            "registered_roles.rgb_capture_window_and_artifacts.v2"
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


def segment_skill_definition_v1() -> SkillDefinition:
    """Describe passive text- or point-prompted image segmentation."""

    return SkillDefinition(
        skill_id=SEGMENT_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Segment one object in a live deployment-registered camera image "
            "from exactly one language or pixel prompt, without changing "
            "downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id=(
            "registered_rgb_role.exclusive_text_or_pixel_prompt.v1"
        ),
        execution_result_semantics_id=(
            "pixel_mask_metadata_and_mask_overlay_artifacts.v1"
        ),
        task_outcome_semantics_id="passive_object_segmentation.v1",
        entry_predicate_ids=(
            "registered_visual_role_available.v1",
            "segmentation_model_available.v1",
        ),
        task_outcome_predicate_ids=("object_mask_segmented.v1",),
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


def plan_point_grasp_skill_definition_v2() -> SkillDefinition:
    """Describe passive planning from one deployment-bound live RGB-D cut."""

    return SkillDefinition(
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="2",
        skill_kind="inspection",
        summary=(
            "Plan ranked DMTAC M tool_tcp pose proposals from one live, "
            "deployment-bound aligned RGB-D frame in the canonical Station "
            "reference frame without selecting a proposal or moving the robot."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id=(
            "registered_aligned_rgbd_role.clicked_region.bounded_candidates.v2"
        ),
        execution_result_semantics_id=(
            "station_bound_ranked_unprivileged_tool_tcp_proposals.v2"
        ),
        task_outcome_semantics_id="passive_point_grasp_planning.v1",
        entry_predicate_ids=(
            "trusted_live_rgbd_binding_available.v1",
            "point_grasp_models_available.v1",
        ),
        task_outcome_predicate_ids=("point_grasp_candidates_generated.v1",),
    )


def plan_point_grasp_skill_definition_v3() -> SkillDefinition:
    """Describe optional finite-time arm IK annotations on live proposals."""

    return SkillDefinition(
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="3",
        skill_kind="inspection",
        summary=(
            "Plan ranked DMTAC M tool_tcp pose proposals from one live, "
            "deployment-bound aligned RGB-D frame and optionally annotate "
            "each proposal with bounded arm IK reachability without selecting "
            "a proposal or moving the robot."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id=(
            "registered_aligned_rgbd_role.clicked_region.bounded_candidates."
            "optional_arm_ik_reachability.v3"
        ),
        execution_result_semantics_id=(
            "station_bound_ranked_unprivileged_tool_tcp_proposals."
            "optional_finite_time_arm_ik_annotations.v3"
        ),
        task_outcome_semantics_id="passive_point_grasp_planning.v1",
        entry_predicate_ids=(
            "trusted_live_rgbd_binding_available.v1",
            "point_grasp_models_available.v1",
            "deployment_arm_ik_evaluator_available.v1",
        ),
        task_outcome_predicate_ids=("point_grasp_candidates_generated.v1",),
    )


def render_point_grasp_candidate_skill_definition_v1() -> SkillDefinition:
    """Describe one optional view of one retained point-grasp proposal."""

    return SkillDefinition(
        skill_id=RENDER_POINT_GRASP_CANDIDATE_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Render exactly one proposal from one retained Station point-grasp "
            "operation into one deployment-bound aligned RGB-D view without "
            "moving the robot."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id=(
            "retained_station_point_grasp_operation.candidate_ordinal.view_role.v1"
        ),
        execution_result_semantics_id=(
            "single_candidate_station_bound_grasp_render_artifact.v1"
        ),
        task_outcome_semantics_id="passive_point_grasp_rendering.v1",
        entry_predicate_ids=(
            "retained_station_point_grasp_result_available.v1",
            "trusted_live_aligned_rgbd_binding_available.v1",
        ),
        task_outcome_predicate_ids=("point_grasp_candidate_rendered.v1",),
    )


__all__ = [
    "CAPTURE_DEPTH_SKILL_ID_V1",
    "CAPTURE_RGB_SKILL_ID_V1",
    "LOCALIZE_OBJECTS_SKILL_ID_V1",
    "PLAN_POINT_GRASP_SKILL_ID_V1",
    "RENDER_POINT_GRASP_CANDIDATE_SKILL_ID_V1",
    "SEGMENT_SKILL_ID_V1",
    "capture_depth_skill_definition_v1",
    "capture_depth_skill_definition_v2",
    "capture_rgb_skill_definition_v1",
    "capture_rgb_skill_definition_v2",
    "localize_objects_skill_definition_v1",
    "plan_point_grasp_skill_definition_v1",
    "plan_point_grasp_skill_definition_v2",
    "plan_point_grasp_skill_definition_v3",
    "render_point_grasp_candidate_skill_definition_v1",
    "segment_skill_definition_v1",
]
