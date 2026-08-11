from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    DecisionScheduleSpec,
    M6_CANDIDATE_POLICY_ARM_QPOS_V1,
    M6_RUNTIME_POLICY_ARM_QPOS_V2,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
    build_immediate_qref_activation_spec,
    build_m6_candidate_policy_arm_qpos_manifest,
    build_m6_runtime_policy_arm_qpos_manifest,
    build_robot_action_v1_runtime_schema,
    build_robot_action_v1_target_capability,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.position_control import ExecutionCapabilityProfile


def _catalog() -> ActionComponentCatalog:
    return ActionComponentCatalog(
        catalog_id="m6-candidate-arms-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
        ),
    )


def _capability() -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="m6-candidate-sim-evidence-v1",
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
        acceptance_evidence_id="deterministic-scheduled-qpos-fixture-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )


def _scheduled_activation(capability: ExecutionCapabilityProfile):
    schedule = DecisionScheduleSpec(
        spec_id="m6-candidate-scheduled-30hz-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    return build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="one-reference-tick-margin-v1",
    )


def test_first_m6_candidate_freezes_narrow_arms_only_semantics() -> None:
    capability = _capability()
    manifest = build_m6_candidate_policy_arm_qpos_manifest(
        catalog=_catalog(),
        capability=capability,
        activation=_scheduled_activation(capability),
    )

    assert manifest.manifest_id == M6_CANDIDATE_POLICY_ARM_QPOS_V1
    assert manifest.motion_authority_sources == ("policy",)
    assert manifest.human_motion_proposal_status == "not_applicable"
    assert manifest.safety_control_sources == (
        "operator",
        "runtime_safety_supervisor",
    )
    assert manifest.lifecycle_control_sources == (
        "operator_authorized_recovery_state_machine",
    )
    assert manifest.authoritative_decision_hz == 30.0
    assert manifest.reference_timing_semantics_hz == 60.0
    assert manifest.actual_reference_frame_producer == "not_applicable"
    assert manifest.planner_command_hz == 120.0
    assert {item.capability_id for item in manifest.exclusions} == {
        "hand_inspire_v1",
        "hand_dmtac_v1",
        "replay_authority",
        "pose_impedance",
        "fix_arm_human_motion",
        "human_motion_takeover",
    }


def test_manifest_hash_is_deterministic_and_component_order_is_semantic() -> None:
    capability = _capability()
    activation = _scheduled_activation(capability)
    first = build_m6_candidate_policy_arm_qpos_manifest(
        catalog=_catalog(),
        capability=capability,
        activation=activation,
    )
    second = build_m6_candidate_policy_arm_qpos_manifest(
        catalog=_catalog(),
        capability=capability,
        activation=activation,
    )
    assert first.semantic_content_hash == second.semantic_content_hash

    reversed_catalog = replace(
        _catalog(),
        components=tuple(reversed(_catalog().components)),
    )
    reversed_manifest = build_m6_candidate_policy_arm_qpos_manifest(
        catalog=reversed_catalog,
        capability=capability,
        activation=activation,
    )
    assert first.semantic_content_hash != reversed_manifest.semantic_content_hash


def test_runtime_manifest_binds_the_exact_runtime_target_capability() -> None:
    catalog = _catalog()
    capability = _capability()
    activation = _scheduled_activation(capability)
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    target_contract_hash = "target-runtime-action-contract-hash-v1"
    target_capability = build_robot_action_v1_target_capability(
        capability_id="mujoco-component-target:test-domain",
        action_schema=action_schema,
        handler_contract_id=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        handler_contract_hash=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        output_target_schema_hash=target_contract_hash,
    )

    manifest = build_m6_runtime_policy_arm_qpos_manifest(
        catalog=catalog,
        capability=capability,
        activation=activation,
        component_target_capability=target_capability,
        target_runtime_action_contract_hash=target_contract_hash,
    )

    assert manifest.manifest_id == M6_RUNTIME_POLICY_ARM_QPOS_V2
    assert manifest.manifest_version == "v2"
    assert (
        manifest.component_target_capability_hash
        == target_capability.semantic_content_hash
    )


def test_runtime_manifest_rejects_another_target_runtime_contract() -> None:
    catalog = _catalog()
    capability = _capability()
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    target_capability = build_robot_action_v1_target_capability(
        capability_id="mujoco-component-target:test-domain",
        action_schema=action_schema,
        handler_contract_id=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        handler_contract_hash=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        output_target_schema_hash="actual-target-runtime-contract-hash",
    )

    with pytest.raises(ValueError, match="supplied target runtime"):
        build_m6_runtime_policy_arm_qpos_manifest(
            catalog=catalog,
            capability=capability,
            activation=_scheduled_activation(capability),
            component_target_capability=target_capability,
            target_runtime_action_contract_hash="another-contract-hash",
        )


def test_candidate_rejects_immediate_or_non_30hz_profiles() -> None:
    capability = _capability()
    changed_rate = replace(capability, configured_selected_keyframe_hz=20.0)
    with pytest.raises(ValueError, match="30 Hz"):
        build_m6_candidate_policy_arm_qpos_manifest(
            catalog=_catalog(),
            capability=changed_rate,
            activation=_scheduled_activation(changed_rate),
        )

    immediate_schedule = DecisionScheduleSpec(
        spec_id="m6-candidate-immediate-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.0,
    )
    with pytest.raises(ValueError, match="scheduled segment"):
        build_m6_candidate_policy_arm_qpos_manifest(
            catalog=_catalog(),
            capability=capability,
            activation=build_immediate_qref_activation_spec(
                immediate_schedule,
                capability,
            ),
        )


def test_candidate_rejects_uncoupled_components() -> None:
    capability = _capability()
    uncoupled = ActionComponentCatalog(
        catalog_id="uncoupled-arms-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=False),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=False),
        ),
    )
    with pytest.raises(ValueError, match="atomically coupled"):
        build_m6_candidate_policy_arm_qpos_manifest(
            catalog=uncoupled,
            capability=capability,
            activation=_scheduled_activation(capability),
        )
