from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    DecisionScheduleSpec,
    assess_safety_migration_gate,
    build_m6_candidate_policy_arm_qpos_manifest,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.position_control import ExecutionCapabilityProfile
from teleop_runtime.action.rm65_candidate_safety_matrix import (
    build_m6_candidate_policy_arm_qpos_scoped_safety_matrix,
)
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    assess_rm65_arms_only_dispatch_inventory,
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    build_rm65_arms_only_worker_drain_plan,
)


def _manifest():
    catalog = ActionComponentCatalog(
        catalog_id="scoped-safety-matrix-arms-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
        ),
    )
    capability = ExecutionCapabilityProfile(
        profile_id="scoped-safety-matrix-capability-v1",
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
        acceptance_evidence_id="scoped-safety-matrix-sim-evidence-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )
    schedule = DecisionScheduleSpec(
        spec_id="scoped-safety-matrix-schedule-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    activation = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="scoped-safety-matrix-margin-v1",
    )
    return build_m6_candidate_policy_arm_qpos_manifest(
        catalog=catalog,
        capability=capability,
        activation=activation,
    )


def _matrix_inputs():
    manifest = _manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    assessment = assess_rm65_arms_only_dispatch_inventory(inventory)
    worker_drain_plan = build_rm65_arms_only_worker_drain_plan(
        manifest=manifest,
        inventory=inventory,
    )
    return manifest, inventory, assessment, worker_drain_plan


def test_scoped_matrix_signs_only_inventory_readiness() -> None:
    manifest, inventory, assessment, worker_drain_plan = _matrix_inputs()
    matrix = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )

    inventory_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m5_8a_inventory",
    )
    mediation_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m5_8b_legacy_mediation",
    )
    authority_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m6_shared_authority",
    )
    assert inventory_gate.ready
    assert not mediation_gate.ready
    assert not authority_gate.ready
    assert "backend_readiness_receipt_and_final_dispatch" in (
        mediation_gate.blocking_rule_ids
    )
    assert "measured_tracking_divergence" in authority_gate.blocking_rule_ids
    assert not assessment.confers_execution_authorization
    assert not assessment.legacy_mediation_ready


def test_excluded_capabilities_are_not_applicable_at_every_gate() -> None:
    manifest, inventory, assessment, worker_drain_plan = _matrix_inputs()
    matrix = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )
    expected = {
        "excluded_hand_inspire",
        "excluded_hand_dmtac",
        "excluded_human_motion_takeover",
        "excluded_replay_authority",
        "excluded_pose_impedance",
        "excluded_fix_arm_human_motion",
    }

    for stage in (
        "m5_8a_inventory",
        "m5_8b_legacy_mediation",
        "m6_shared_authority",
    ):
        decision = assess_safety_migration_gate(matrix, gate_stage=stage)
        assert set(decision.not_applicable_rule_ids) == expected


def test_matrix_is_deterministic_and_binds_exact_inventory() -> None:
    manifest, inventory, assessment, worker_drain_plan = _matrix_inputs()
    first = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )
    second = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )
    assert first.semantic_content_hash == second.semantic_content_hash

    with pytest.raises(ValueError, match="does not describe this inventory"):
        build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
            manifest=manifest,
            inventory=inventory,
            inventory_assessment=replace(
                assessment,
                inventory_hash="different-inventory",
            ),
            worker_drain_plan=worker_drain_plan,
        )


def test_all_included_rules_retain_explicit_real_mediation_blockers() -> None:
    manifest, inventory, assessment, worker_drain_plan = _matrix_inputs()
    matrix = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )

    included = tuple(
        rule
        for rule in matrix.rules
        if rule.target_disposition != "not_applicable"
    )
    assert included
    assert all(
        rule.gate("m5_8a_inventory").status == "ready"
        and rule.gate("m5_8b_legacy_mediation").status == "blocked"
        and rule.gate("m6_shared_authority").status == "blocked"
        for rule in included
    )


def test_matrix_binds_the_exact_unbound_worker_drain_plan() -> None:
    manifest, inventory, assessment, worker_drain_plan = _matrix_inputs()
    matrix = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=assessment,
        worker_drain_plan=worker_drain_plan,
    )
    drain_rule = next(
        rule
        for rule in matrix.rules
        if rule.rule_id == "execution_cancellation_and_worker_drain"
    )

    assert any(
        worker_drain_plan.semantic_content_hash in evidence_id
        for evidence_id in drain_rule.gate("m5_8a_inventory").evidence_ids
    )
    assert drain_rule.gate("m5_8b_legacy_mediation").status == "blocked"

    with pytest.raises(ValueError, match="not bound"):
        build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
            manifest=manifest,
            inventory=inventory,
            inventory_assessment=assessment,
            worker_drain_plan=replace(
                worker_drain_plan,
                dispatch_inventory_hash="different-inventory",
            ),
        )
