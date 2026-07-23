from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    MeasuredDivergenceGuardSpec,
    RobotAction,
    SafetyAnchorPolicy,
    admit_resolved_execution_target,
    build_resolved_execution_target,
    evaluate_counterfactual_safety_shadow,
    evaluate_counterfactual_target_commit,
    make_uninitialized_safety_anchor_state,
    rebase_safety_anchor_from_measured,
)
from robot_motion.position_control import PositionKeyframeSafetyConfig


def action(value: float) -> RobotAction:
    return RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        timestamp=10.0,
        arm_qpos_by_side={"left": np.full(6, value, dtype=np.float32)},
        active_effectors=("left_arm",),
        source="policy",
    )


def anchor_policy() -> SafetyAnchorPolicy:
    return SafetyAnchorPolicy(
        policy_id="counterfactual-policy-qpos-v1",
        max_measurement_age_sec=0.1,
        max_anchor_age_sec=1.0,
    )


def numeric_profile() -> CanonicalKeyframeSafetyProfile:
    return CanonicalKeyframeSafetyProfile(
        profile_id="counterfactual-qpos-keyframe-v1",
        limits=PositionKeyframeSafetyConfig(
            max_joint_step_rad=0.2,
            joint_lower_rad=(-1.0,) * 6,
            joint_upper_rad=(1.0,) * 6,
        ),
    )


def measured(value: float = 0.0) -> CanonicalActionSafetyAnchor:
    return CanonicalActionSafetyAnchor(
        anchor_id="measurement:left-arm:0",
        anchor_kind="measured_state",
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={"left": np.full(6, value, dtype=np.float32)},
        observed_at=9.99,
    )


def rebased_state():
    policy = anchor_policy()
    uninitialized = make_uninitialized_safety_anchor_state(
        control_domain_id="robot-0",
        selection_group_id="left_arm",
        execution_epoch=0,
        group_segment_generation=0,
        policy=policy,
    )
    state, transition = rebase_safety_anchor_from_measured(
        uninitialized,
        measured_anchor=measured(),
        measured_evidence_id="measurement-event:0",
        occurred_at=10.0,
        policy=policy,
    )
    assert transition.transition_kind == "initialize_from_measured"
    return state


def evaluate(state, *, selected_value: float = 0.8, cutoff_at: float = 10.005):
    return evaluate_counterfactual_safety_shadow(
        source_arbitration_evaluation_id="arbitration-evaluation:0",
        arbitration_result_id="arbitration-result:0",
        directive_id="counterfactual-directive:0",
        selected_action_id="counterfactual-selected:0",
        selected_action=action(selected_value),
        anchor_state=state,
        measured_state_evidence=measured(),
        safety_cutoff_at=cutoff_at,
        anchor_policy=anchor_policy(),
        keyframe_safety_profile=numeric_profile(),
        measured_divergence_guard=MeasuredDivergenceGuardSpec(
            guard_id="disabled-unvalidated-divergence-v1",
            enabled=False,
        ),
        action_schema_hash="robot-action-qpos-hash",
        component_schema_hash="left-arm-component-hash",
    )


def target_and_admission(evaluation, *, accepted: bool = True, kind: str = "qpos_reference"):
    assert evaluation.safety_result.safe_action is not None
    target = build_resolved_execution_target(
        control_domain_id="robot-0",
        selection_group_id="left_arm",
        execution_epoch=0,
        group_segment_generation=1,
        source_safety_result_id=evaluation.safety_result_id,
        safe_action=evaluation.safety_result.safe_action,
        target_kind=kind,
        component_dependencies=("left_arm",),
        resolved_at=10.006,
        scheduled_from=10.0,
        scheduled_until=10.5,
        clock_domain_id="control.monotonic",
        input_action_schema_hash="robot-action-qpos-hash",
        output_target_schema_hash="qpos-reference-hash",
        resolver_contract_id="fixture-qpos-resolver-v1",
        resolver_contract_hash="fixture-qpos-resolver-hash",
        payload={"qpos": np.full(6, 0.2, dtype=np.float32)},
    )
    admission = admit_resolved_execution_target(
        target,
        receiver_id="counterfactual-reference-receiver",
        admitted_at=10.007,
        current_execution_epoch=0,
        expected_group_segment_generation=1,
        receiver_accepts=accepted,
        rejection_reason=None if accepted else "fixture_rejection",
    )
    return target, admission


def test_counterfactual_anchor_advances_only_after_target_admission() -> None:
    initial = rebased_state()
    first = evaluate(initial)
    assert first.non_authoritative
    assert first.anchor_state_before.state_id == initial.state_id
    assert initial.group_segment_generation == 0
    assert first.safety_result.safe_action is not None
    np.testing.assert_allclose(
        first.safety_result.safe_action.arm_qpos_by_side["left"],
        np.full(6, 0.2),
    )

    target, admission = target_and_admission(first)
    committed = evaluate_counterfactual_target_commit(
        first,
        target=target,
        admission=admission,
    )
    assert committed.non_authoritative
    assert committed.committed
    assert committed.anchor_transition is not None
    assert committed.anchor_state_after.group_segment_generation == 1
    assert committed.anchor_state_after.anchor is not None
    np.testing.assert_allclose(
        committed.anchor_state_after.anchor.arm_qpos_by_side["left"],
        np.full(6, 0.2),
    )

    second = evaluate(committed.anchor_state_after, cutoff_at=10.01)
    assert second.safety_result.safe_action is not None
    np.testing.assert_allclose(
        second.safety_result.safe_action.arm_qpos_by_side["left"],
        np.full(6, 0.4),
    )


def test_rejected_shadow_target_does_not_advance_anchor() -> None:
    initial = rebased_state()
    safety = evaluate(initial)
    target, admission = target_and_admission(safety, accepted=False)

    rejected = evaluate_counterfactual_target_commit(
        safety,
        target=target,
        admission=admission,
    )
    assert not rejected.committed
    assert rejected.anchor_transition is None
    assert rejected.anchor_state_after.state_id == initial.state_id
    assert rejected.status == "shadow_target_rejected"


def test_direct_command_cannot_fake_backend_acceptance_in_shadow() -> None:
    safety = evaluate(rebased_state())
    target, admission = target_and_admission(
        safety,
        kind="direct_command_intent",
    )
    with pytest.raises(ValueError, match="backend-acceptance"):
        evaluate_counterfactual_target_commit(
            safety,
            target=target,
            admission=admission,
        )


def test_counterfactual_shadow_rejects_unimplemented_divergence_guard() -> None:
    state = rebased_state()
    guard = MeasuredDivergenceGuardSpec(
        guard_id="enabled-but-unvalidated-v1",
        enabled=True,
        max_joint_error_rad=0.4,
        max_measurement_age_sec=0.1,
    )
    with pytest.raises(ValueError, match="does not yet implement measured divergence"):
        evaluate_counterfactual_safety_shadow(
            source_arbitration_evaluation_id="arbitration-evaluation:0",
            arbitration_result_id="arbitration-result:0",
            directive_id="counterfactual-directive:0",
            selected_action_id="counterfactual-selected:0",
            selected_action=action(0.8),
            anchor_state=state,
            measured_state_evidence=measured(),
            safety_cutoff_at=10.005,
            anchor_policy=anchor_policy(),
            keyframe_safety_profile=numeric_profile(),
            measured_divergence_guard=guard,
            action_schema_hash="robot-action-qpos-hash",
            component_schema_hash="left-arm-component-hash",
        )


def test_counterfactual_hash_is_deterministic_and_target_lineage_is_strict() -> None:
    state = rebased_state()
    first = evaluate(state)
    second = evaluate(state)
    assert first.semantic_content_hash == second.semantic_content_hash

    target, admission = target_and_admission(first)
    wrong_target = replace(target, source_safety_result_id="other-safety-result")
    with pytest.raises(ValueError, match="another safety evaluation"):
        evaluate_counterfactual_target_commit(
            first,
            target=wrong_target,
            admission=admission,
        )
