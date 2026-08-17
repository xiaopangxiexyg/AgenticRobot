"""Target-independent semantics for read-only perception Skills."""

from __future__ import annotations

from agent_gateway.protocol.skill_catalog import SkillDefinition

CAPTURE_DEPTH_SKILL_ID_V1 = "perception.capture_depth"


def capture_depth_skill_definition_v1() -> SkillDefinition:
    """Describe one passive capture from a deployment-registered view."""

    return SkillDefinition(
        skill_id=CAPTURE_DEPTH_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Capture the latest depth frame from one deployment-registered "
            "view without changing downstream control state."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="registered_view.latest_depth_frame.v1",
        execution_result_semantics_id=(
            "registered_view.depth_frame_metadata_and_artifact.v1"
        ),
        task_outcome_semantics_id="passive_depth_capture.v1",
        entry_predicate_ids=("registered_view_available.v1",),
        task_outcome_predicate_ids=("depth_frame_captured.v1",),
    )


__all__ = [
    "CAPTURE_DEPTH_SKILL_ID_V1",
    "capture_depth_skill_definition_v1",
]
