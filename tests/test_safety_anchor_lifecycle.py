from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    CanonicalActionSafetyAnchor,
    HandActionSpec,
    MeasuredDivergenceGuardSpec,
    RobotAction,
    SafetyAnchorCommitEvidence,
    SafetyAnchorPolicy,
    build_safety_evaluation_input_snapshot,
    commit_safe_action_anchor,
    invalidate_safety_anchor,
    make_uninitialized_safety_anchor_state,
    rebase_safety_anchor_from_measured,
    resolve_safety_anchor,
    safety_anchor_action_hash,
)


def policy() -> SafetyAnchorPolicy:
    return SafetyAnchorPolicy(
        policy_id="canonical-arm-anchor-v1",
        max_anchor_age_sec=0.5,
        max_measurement_age_sec=0.1,
    )


def measured_anchor(*, observed_at: float = 10.0) -> CanonicalActionSafetyAnchor:
    return CanonicalActionSafetyAnchor(
        anchor_id=f"measured-qpos:{observed_at}",
        anchor_kind="measured_state",
        action_space="bimanual_qpos_v1",
        arm_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        observed_at=observed_at,
    )


def safe_action(value: float) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, value),
        right_arm=np.full(6, -value),
        source="shared-safety-fixture",
    )


def initial_state():
    return make_uninitialized_safety_anchor_state(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=0,
        group_segment_generation=0,
        policy=policy(),
    )


def rebased_state():
    return rebase_safety_anchor_from_measured(
        initial_state(),
        measured_anchor=measured_anchor(),
        measured_evidence_id="measurement:0",
        occurred_at=10.02,
        policy=policy(),
    )[0]


def commit_evidence(action: RobotAction, *, accepted: bool = True):
    return SafetyAnchorCommitEvidence(
        evidence_id="target-admission:1",
        evidence_kind="execution_target_admitted",
        target_id="resolved-target:1",
        safety_result_id="shared-safety:1",
        safe_action_hash=safety_anchor_action_hash(action),
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=0,
        group_segment_generation=1,
        occurred_at=10.05,
        accepted=accepted,
        rejection_reason=None if accepted else "planner_rejected",
    )


def test_startup_requires_fresh_measured_rebase() -> None:
    state = initial_state()
    unresolved = resolve_safety_anchor(
        state,
        policy=policy(),
        safety_cutoff_at=10.0,
        execution_epoch=0,
        group_segment_generation=0,
    )
    assert not unresolved.eligible
    assert unresolved.status == "rebase_required"

    active, transition = rebase_safety_anchor_from_measured(
        state,
        measured_anchor=measured_anchor(),
        measured_evidence_id="measurement:0",
        occurred_at=10.02,
        policy=policy(),
    )
    assert active.status == "active"
    assert active.anchor is not None
    assert active.anchor.anchor_kind == "measured_state"
    assert transition.transition_kind == "initialize_from_measured"
    assert resolve_safety_anchor(
        active,
        policy=policy(),
        safety_cutoff_at=10.03,
        execution_epoch=0,
        group_segment_generation=0,
    ).eligible

    with pytest.raises(ValueError, match="stale"):
        rebase_safety_anchor_from_measured(
            state,
            measured_anchor=measured_anchor(observed_at=9.0),
            measured_evidence_id="measurement:stale",
            occurred_at=10.02,
            policy=policy(),
        )
    with pytest.raises(ValueError, match="future"):
        rebase_safety_anchor_from_measured(
            state,
            measured_anchor=measured_anchor(observed_at=10.03),
            measured_evidence_id="measurement:future",
            occurred_at=10.02,
            policy=policy(),
        )


def test_only_explicitly_admitted_safe_action_advances_anchor() -> None:
    previous = rebased_state()
    selected = safe_action(0.8)
    bounded_safe = safe_action(0.2)

    committed, transition = commit_safe_action_anchor(
        previous,
        safe_action=bounded_safe,
        evidence=commit_evidence(bounded_safe),
        valid_until=10.5,
    )

    assert transition.transition_kind == "commit_safe_target"
    assert committed.group_segment_generation == 1
    assert committed.anchor is not None
    assert committed.anchor.anchor_kind == "previous_safe_committed"
    np.testing.assert_allclose(
        committed.anchor.arm_qpos_by_side["left"],
        bounded_safe.arm_qpos_by_side["left"],
    )
    assert not np.array_equal(
        committed.anchor.arm_qpos_by_side["left"],
        selected.arm_qpos_by_side["left"],
    )

    with pytest.raises(ValueError, match="Rejected execution-target"):
        commit_safe_action_anchor(
            previous,
            safe_action=bounded_safe,
            evidence=commit_evidence(bounded_safe, accepted=False),
        )
    wrong_generation = SafetyAnchorCommitEvidence(
        **{
            **commit_evidence(bounded_safe).__dict__,
            "evidence_id": "target-admission:wrong-generation",
            "group_segment_generation": 2,
        }
    )
    with pytest.raises(ValueError, match="next segment generation"):
        commit_safe_action_anchor(
            previous,
            safe_action=bounded_safe,
            evidence=wrong_generation,
        )
    assert previous.group_segment_generation == 0
    assert previous.anchor is not None
    np.testing.assert_allclose(previous.anchor.arm_qpos_by_side["left"], 0.0)


def test_admitted_compound_safe_action_advances_arm_and_hand_anchor_atomically() -> None:
    hand_spec = HandActionSpec.dmtac_gripper("left")
    action = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.1),
        right_arm=np.full(6, -0.1),
        left_hand=np.asarray([0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
        source="compound-safe-fixture",
        metadata={
            "hand_action_specs_by_side": {
                "left": hand_spec.to_metadata(),
            }
        },
    )
    committed, transition = commit_safe_action_anchor(
        rebased_state(),
        safe_action=action,
        evidence=commit_evidence(action),
    )

    assert transition.transition_kind == "commit_safe_target"
    assert committed.anchor is not None
    np.testing.assert_allclose(
        committed.anchor.arm_qpos_by_side["left"],
        np.full(6, 0.1),
    )
    np.testing.assert_allclose(
        committed.anchor.hand_qpos_by_side["left"],
        [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert committed.anchor.hand_action_specs_by_side["left"] == hand_spec


def test_hold_or_epoch_change_invalidates_only_the_explicit_group_state() -> None:
    committed, _ = commit_safe_action_anchor(
        rebased_state(),
        safe_action=safe_action(0.1),
        evidence=commit_evidence(safe_action(0.1)),
    )
    invalid, transition = invalidate_safety_anchor(
        committed,
        transition_kind="invalidate_on_execution_epoch_change",
        occurred_at=10.1,
        causal_evidence_ids=("execution-cancellation:1",),
        next_execution_epoch=1,
        next_group_segment_generation=0,
    )

    assert invalid.status == "rebase_required"
    assert invalid.anchor is None
    assert invalid.execution_epoch == 1
    assert invalid.selection_group_id == "arms"
    assert transition.previous_state_id == committed.state_id
    assert not resolve_safety_anchor(
        invalid,
        policy=policy(),
        safety_cutoff_at=10.11,
        execution_epoch=1,
        group_segment_generation=0,
    ).eligible


def test_safety_evaluation_snapshot_freezes_anchor_measurement_and_selected_action() -> None:
    state = rebased_state()
    action = safe_action(0.3)
    measurement = measured_anchor(observed_at=10.01)
    guard = MeasuredDivergenceGuardSpec(
        guard_id="qpos-divergence-shadow-v1",
        enabled=True,
        max_joint_error_rad=0.4,
        max_measurement_age_sec=0.1,
    )

    first = build_safety_evaluation_input_snapshot(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=0,
        target_group_segment_generation=1,
        arbitration_result_id="arbitration-result:1",
        directive_id="execution-directive:1",
        selected_action_id="selected-action:1",
        selected_action=action,
        anchor_state=state,
        measured_state_evidence=measurement,
        safety_cutoff_at=10.04,
        anchor_policy=policy(),
        keyframe_safety_profile_hash="keyframe-profile-hash",
        measured_divergence_guard=guard,
        action_schema_hash="action-schema-hash",
        component_schema_hash="component-schema-hash",
    )
    second = build_safety_evaluation_input_snapshot(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=0,
        target_group_segment_generation=1,
        arbitration_result_id="arbitration-result:1",
        directive_id="execution-directive:1",
        selected_action_id="selected-action:1",
        selected_action=action,
        anchor_state=state,
        measured_state_evidence=measurement,
        safety_cutoff_at=10.04,
        anchor_policy=policy(),
        keyframe_safety_profile_hash="keyframe-profile-hash",
        measured_divergence_guard=guard,
        action_schema_hash="action-schema-hash",
        component_schema_hash="component-schema-hash",
    )

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.safety_evaluation_input_id == second.safety_evaluation_input_id
    assert first.anchor_group_segment_generation == 0
    assert first.target_group_segment_generation == 1
    with pytest.raises(ValueError):
        first.selected_action.arm_qpos_by_side["left"][0] = 3.0


def test_enabled_measured_guard_requires_threshold_and_freshness_policy() -> None:
    with pytest.raises(ValueError, match="needs a threshold"):
        MeasuredDivergenceGuardSpec(guard_id="invalid", enabled=True)
    with pytest.raises(ValueError, match="max_measurement_age_sec"):
        MeasuredDivergenceGuardSpec(
            guard_id="invalid-freshness",
            enabled=True,
            max_joint_error_rad=0.3,
        )
