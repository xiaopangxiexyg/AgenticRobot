"""Canonical builder for the first narrow RM65 target execution candidate.

The values in this module freeze the already-validated scoped sim/control
contract.  Building the manifest never creates a runtime owner and never
authorizes hardware.  Hardware authorization remains a separate runtime fact.
"""

from __future__ import annotations

from robot_motion.action import (
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
    ActionComponentCatalog,
    ActionComponentSpec,
    ComponentTargetCapabilitySpec,
    DecisionScheduleSpec,
    build_m6_candidate_policy_arm_qpos_manifest,
    build_robot_action_v1_runtime_schema,
    build_robot_action_v1_target_capability,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.position_control import ExecutionCapabilityProfile


def build_rm65_policy_arm_qpos_candidate_capability() -> ExecutionCapabilityProfile:
    """Return the exact capability bound into the first RM65 candidate."""

    return ExecutionCapabilityProfile(
        profile_id="rm65-arms-only-inventory-capability-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=None,
        configured_coupled_ik_hz=None,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_qpos_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="scoped-scheduled-qpos-evidence-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )


def build_rm65_policy_arm_qpos_candidate_catalog() -> ActionComponentCatalog:
    """Return the exact component catalog bound into the candidate."""

    return ActionComponentCatalog(
        catalog_id="rm65-arms-only-inventory-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
        ),
    )


def build_rm65_policy_arm_qpos_candidate_target_capability(
) -> ComponentTargetCapabilitySpec:
    """Expose the exact target capability previously retained only by hash.

    The candidate manifest already binds this capability hash.  Returning the
    immutable object lets a target-runtime provider reuse the same action
    schema without changing the legacy manifest or execution path.
    """

    catalog = build_rm65_policy_arm_qpos_candidate_catalog()
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    return build_robot_action_v1_target_capability(
        capability_id="m6-candidate-robot-action-v1-target",
        action_schema=action_schema,
        handler_contract_id=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        handler_contract_hash=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        output_target_schema_hash=catalog.semantic_content_hash,
    )


def build_rm65_policy_arm_qpos_candidate_manifest():
    """Return the deterministic ``m6_candidate_policy_arm_qpos_v1`` manifest."""

    catalog = build_rm65_policy_arm_qpos_candidate_catalog()
    capability = build_rm65_policy_arm_qpos_candidate_capability()
    schedule = DecisionScheduleSpec(
        spec_id="rm65-arms-only-inventory-schedule-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    activation = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="one-reference-tick-margin-v1",
    )
    return build_m6_candidate_policy_arm_qpos_manifest(
        catalog=catalog,
        capability=capability,
        activation=activation,
    )


__all__ = [
    "build_rm65_policy_arm_qpos_candidate_capability",
    "build_rm65_policy_arm_qpos_candidate_catalog",
    "build_rm65_policy_arm_qpos_candidate_manifest",
    "build_rm65_policy_arm_qpos_candidate_target_capability",
]
