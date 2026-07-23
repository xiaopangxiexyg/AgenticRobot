from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    CanonicalActionSafetyAnchor,
    RobotAction,
    SafetyAnchorPolicy,
    activate_group_generation_reservation,
    admission_guarantee_satisfies,
    admit_reference_segment,
    build_reference_endpoint_parent,
    build_pending_group_generation_reservation,
    build_reference_segment,
    build_reference_segment_terminal,
    build_safety_anchor_commit_candidate,
    commit_safe_action_anchor,
    make_uninitialized_safety_anchor_state,
    rebase_safety_anchor_from_measured,
    reference_segment_anchor_commit_evidence,
    resolve_safety_anchor,
    resolve_safety_anchor_commit_timeline,
    safety_anchor_action_hash,
    validate_reference_segment_successor,
)


def safe_action(value: float = 0.2) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, value),
        right_arm=np.full(6, -value),
        source="reference-segment-fixture",
    )


def endpoint_parents(*, end_action: RobotAction | None = None):
    action = safe_action() if end_action is None else end_action
    start = build_reference_endpoint_parent(
        role="start",
        source_kind="measured_rebase",
        source_fact_id="measurement:0",
        projected_group_action_hash="measured-anchor-hash",
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-frame-v1",
        target_effective_at=10.0,
    )
    end = build_reference_endpoint_parent(
        role="end",
        source_kind="current_safe_target",
        source_fact_id="shared-safety:1",
        projected_group_action_hash=safety_anchor_action_hash(action),
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-frame-v1",
        target_effective_at=10.1,
    )
    return start, end


def segment(
    *,
    end_action: RobotAction | None = None,
    created_at: float = 9.98,
    authority_kind: str = "authoritative_shared",
):
    start, end = endpoint_parents(end_action=end_action)
    return build_reference_segment(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=4,
        authority_kind=authority_kind,
        start_parent=start,
        end_parent=end,
        component_dependencies=("left_arm", "right_arm"),
        created_at=created_at,
        clock_domain_id="local_monotonic",
        interpolation_contract_hash="linear-qpos-v1",
        resolver_contract_hash="position-resolver-v1",
        planner_contract_hash="rm65-planner-v1",
        input_action_schema_hash="bimanual-qpos-v1",
        output_reference_schema_hash="bimanual-qref-v1",
        required_admission_guarantee="interpolation_segment_accepted",
        segment_safety_evidence_level="endpoints_only",
    )


def admission(item, **overrides):
    values = {
        "receiver_id": "position-reference-worker:0",
        "admitted_at": 9.99,
        "admission_sequence": 3,
        "current_execution_epoch": 2,
        "expected_group_segment_generation": 4,
        "achieved_admission_guarantee": "interpolation_segment_accepted",
    }
    values.update(overrides)
    return admit_reference_segment(item, **values)


def candidate(*, item=None, receipt=None, action=None):
    resolved_action = safe_action() if action is None else action
    resolved_segment = segment(end_action=resolved_action) if item is None else item
    resolved_admission = (
        admission(resolved_segment) if receipt is None else receipt
    )
    return build_safety_anchor_commit_candidate(
        resolved_segment,
        resolved_admission,
        safety_result_id="shared-safety:1",
        safe_action=resolved_action,
    )


def resolution(item, *, target_time: float, terminals=(), cutoff: float = 10.0):
    return resolve_safety_anchor_commit_timeline(
        (item,),
        terminals=terminals,
        control_domain_id="robot-0",
        selection_group_id="arms",
        safety_cutoff_at=cutoff,
        target_effective_at=target_time,
        current_execution_epoch=2,
        expected_group_segment_generation=4,
        minimum_admission_guarantee="interpolation_segment_accepted",
    )


def test_segment_uses_ordered_time_roles_not_arm_side_names() -> None:
    first = segment()
    second = segment()

    assert first.reference_segment_id == second.reference_segment_id
    assert tuple(parent.role for parent in first.endpoint_parents) == (
        "start",
        "end",
    )
    assert first.start_parent.source_kind == "measured_rebase"
    assert first.end_parent.source_kind == "current_safe_target"
    assert first.scheduled_from == 10.0
    assert first.scheduled_until == 10.1

    with pytest.raises(ValueError, match="START"):
        build_reference_endpoint_parent(
            role="start",
            source_kind="current_safe_target",
            source_fact_id="shared-safety:bad",
            projected_group_action_hash="hash",
            action_schema_hash="schema",
            frame_schema_hash="frame",
            target_effective_at=10.0,
        )


def test_receiver_admission_is_distinct_from_source_queue_and_guarantee() -> None:
    item = segment()
    assert admission(item).accepted
    assert admission(item).admitted_at < item.scheduled_from
    assert admission_guarantee_satisfies(
        "endpoint_resolved",
        "interpolation_segment_accepted",
    )

    weak = admission(item, achieved_admission_guarantee="queue_only")
    assert not weak.accepted
    assert weak.rejection_reason == "insufficient_admission_guarantee"

    stale_epoch = admission(item, current_execution_epoch=3)
    assert not stale_epoch.accepted
    assert stale_epoch.rejection_reason == "execution_epoch_mismatch"

    expired = admission(item, admitted_at=item.scheduled_until)
    assert not expired.accepted
    assert expired.rejection_reason == "segment_interval_expired"


def test_generation_reservation_does_not_activate_before_segment_start() -> None:
    item = segment()
    receipt = admission(item)
    reservation = build_pending_group_generation_reservation(
        item,
        receipt,
        active_generation=3,
        relation_kind="initial",
    )

    assert reservation.active_generation_at_reservation == 3
    assert reservation.reserved_generation == 4
    assert reservation.effective_from == item.scheduled_from
    with pytest.raises(ValueError, match="before effective_from"):
        activate_group_generation_reservation(
            reservation,
            current_execution_epoch=2,
            current_active_generation=3,
            activated_at=9.999,
            activation_sequence=4,
        )

    activation = activate_group_generation_reservation(
        reservation,
        current_execution_epoch=2,
        current_active_generation=3,
        activated_at=10.0,
        activation_sequence=4,
    )
    assert activation.previous_active_generation == 3
    assert activation.active_generation == 4


def test_successor_requires_exact_predecessor_end_parent() -> None:
    predecessor = segment()
    successor_start = build_reference_endpoint_parent(
        role="start",
        source_kind="committed_safe_endpoint",
        source_fact_id=predecessor.reference_segment_id,
        projected_group_action_hash=(
            predecessor.end_parent.projected_group_action_hash
        ),
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-frame-v1",
        target_effective_at=predecessor.scheduled_until,
    )
    successor_end = build_reference_endpoint_parent(
        role="end",
        source_kind="current_safe_target",
        source_fact_id="shared-safety:2",
        projected_group_action_hash=safety_anchor_action_hash(safe_action(0.3)),
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-frame-v1",
        target_effective_at=10.2,
    )
    successor = build_reference_segment(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=5,
        authority_kind="authoritative_shared",
        start_parent=successor_start,
        end_parent=successor_end,
        component_dependencies=("left_arm", "right_arm"),
        created_at=10.02,
        clock_domain_id="local_monotonic",
        interpolation_contract_hash="linear-qpos-v1",
        resolver_contract_hash="position-resolver-v1",
        planner_contract_hash="rm65-planner-v1",
        input_action_schema_hash="bimanual-qpos-v1",
        output_reference_schema_hash="bimanual-qref-v1",
        required_admission_guarantee="interpolation_segment_accepted",
        segment_safety_evidence_level="endpoints_only",
    )

    validate_reference_segment_successor(predecessor, successor)
    bad_start = build_reference_endpoint_parent(
        role="start",
        source_kind="committed_safe_endpoint",
        source_fact_id="reference-segment:wrong",
        projected_group_action_hash=(
            predecessor.end_parent.projected_group_action_hash
        ),
        action_schema_hash="bimanual-qpos-v1",
        frame_schema_hash="joint-frame-v1",
        target_effective_at=predecessor.scheduled_until,
    )
    bad_successor = build_reference_segment(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=5,
        authority_kind="authoritative_shared",
        start_parent=bad_start,
        end_parent=successor_end,
        component_dependencies=("left_arm", "right_arm"),
        created_at=10.02,
        clock_domain_id="local_monotonic",
        interpolation_contract_hash="linear-qpos-v1",
        resolver_contract_hash="position-resolver-v1",
        planner_contract_hash="rm65-planner-v1",
        input_action_schema_hash="bimanual-qpos-v1",
        output_reference_schema_hash="bimanual-qref-v1",
        required_admission_guarantee="interpolation_segment_accepted",
        segment_safety_evidence_level="endpoints_only",
    )
    with pytest.raises(ValueError, match="exactly reference predecessor END"):
        validate_reference_segment_successor(predecessor, bad_successor)


def test_future_endpoint_stays_pending_until_target_effective_time() -> None:
    commit_candidate = candidate()

    pending = resolution(commit_candidate, target_time=10.09)
    assert pending.status == "pending_effective"
    assert pending.selected_candidate is None
    assert pending.assessments[0].status == "pending_effective"

    eligible = resolution(commit_candidate, target_time=10.1)
    assert eligible.status == "eligible"
    assert eligible.selected_candidate_id == commit_candidate.candidate_id
    evidence = reference_segment_anchor_commit_evidence(eligible)
    assert evidence.occurred_at == 9.99
    assert evidence.effective_from == 10.1


def test_candidate_requires_causal_admission_before_safety_cutoff() -> None:
    item = segment(created_at=10.02)
    receipt = admission(item, admitted_at=10.03)
    commit_candidate = candidate(item=item, receipt=receipt)

    unavailable = resolution(
        commit_candidate,
        target_time=10.1,
        cutoff=10.02,
    )
    assert unavailable.status == "not_yet_available"

    available = resolution(
        commit_candidate,
        target_time=10.1,
        cutoff=10.03,
    )
    assert available.status == "eligible"


@pytest.mark.parametrize("status", ("superseded", "canceled", "expired", "faulted"))
def test_terminal_before_endpoint_invalidates_pending_commit(status: str) -> None:
    item = segment()
    commit_candidate = candidate(item=item)
    terminal = build_reference_segment_terminal(
        item,
        status=status,
        available_at=10.04,
        effective_at=10.05,
        terminal_admission_sequence=4,
        replacement_segment_id=("reference-segment:new" if status == "superseded" else None),
        evidence_ids=(f"{status}:1",),
    )

    result = resolution(
        commit_candidate,
        target_time=10.1,
        cutoff=10.06,
        terminals=(terminal,),
    )
    assert result.status == "no_eligible_candidate"
    assert result.assessments[0].status in {"superseded", "invalidated"}


def test_terminal_after_endpoint_does_not_retroactively_remove_active_commit() -> None:
    item = segment()
    commit_candidate = candidate(item=item)
    completed = build_reference_segment_terminal(
        item,
        status="completed",
        available_at=10.11,
        effective_at=10.1,
        terminal_admission_sequence=4,
        last_reference_frame_id="reference-frame:last",
    )
    canceled_later = build_reference_segment_terminal(
        item,
        status="canceled",
        available_at=10.13,
        effective_at=10.12,
        terminal_admission_sequence=5,
        evidence_ids=("cancel:later",),
    )

    result = resolution(
        commit_candidate,
        target_time=10.2,
        cutoff=10.2,
        terminals=(completed, canceled_later),
    )
    assert result.status == "eligible"


def test_terminal_is_not_visible_before_receiver_availability() -> None:
    item = segment()
    commit_candidate = candidate(item=item)
    terminal = build_reference_segment_terminal(
        item,
        status="canceled",
        available_at=10.08,
        effective_at=10.05,
        terminal_admission_sequence=5,
        evidence_ids=("late-terminal",),
    )

    before_available = resolution(
        commit_candidate,
        target_time=10.1,
        cutoff=10.07,
        terminals=(terminal,),
    )
    after_available = resolution(
        commit_candidate,
        target_time=10.1,
        cutoff=10.08 + 1e-6,
        terminals=(terminal,),
    )

    assert before_available.status == "eligible"
    assert after_available.status == "no_eligible_candidate"


def test_same_time_terminal_requires_cutoff_receiver_sequence() -> None:
    item = segment()
    commit_candidate = candidate(item=item)
    terminal = build_reference_segment_terminal(
        item,
        status="canceled",
        available_at=10.06,
        effective_at=10.05,
        terminal_admission_sequence=7,
        evidence_ids=("same-time-terminal",),
    )
    values = {
        "candidates": (commit_candidate,),
        "terminals": (terminal,),
        "control_domain_id": "robot-0",
        "selection_group_id": "arms",
        "safety_cutoff_at": 10.06,
        "target_effective_at": 10.1,
        "current_execution_epoch": 2,
        "expected_group_segment_generation": 4,
        "minimum_admission_guarantee": "interpolation_segment_accepted",
    }

    before_terminal = resolve_safety_anchor_commit_timeline(
        **values,
        safety_cutoff_admission_sequence=6,
    )
    after_terminal = resolve_safety_anchor_commit_timeline(
        **values,
        safety_cutoff_admission_sequence=7,
    )

    assert before_terminal.status == "eligible"
    assert after_terminal.status == "no_eligible_candidate"


def test_multiple_eligible_candidates_without_explicit_supersession_fail_closed() -> None:
    first_segment = segment()
    first = candidate(item=first_segment)
    second_receipt = admission(first_segment, admission_sequence=4)
    second = candidate(item=first_segment, receipt=second_receipt)

    result = resolve_safety_anchor_commit_timeline(
        (first, second),
        control_domain_id="robot-0",
        selection_group_id="arms",
        safety_cutoff_at=10.0,
        target_effective_at=10.1,
        current_execution_epoch=2,
        expected_group_segment_generation=4,
        minimum_admission_guarantee="interpolation_segment_accepted",
    )
    assert result.status == "ambiguous_candidates"
    assert result.selected_candidate is None


def test_anchor_state_tracks_availability_separately_from_effective_time() -> None:
    commit_candidate = candidate()
    eligible = resolution(commit_candidate, target_time=10.1)
    evidence = reference_segment_anchor_commit_evidence(eligible)
    policy = SafetyAnchorPolicy(
        policy_id="canonical-arm-anchor-v1",
        max_measurement_age_sec=0.1,
    )
    initial = make_uninitialized_safety_anchor_state(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=3,
        policy=policy,
    )
    measured = CanonicalActionSafetyAnchor(
        anchor_id="measurement:3",
        anchor_kind="measured_state",
        action_space="bimanual_qpos_v1",
        arm_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        observed_at=9.97,
    )
    active, _ = rebase_safety_anchor_from_measured(
        initial,
        measured_anchor=measured,
        measured_evidence_id="measurement:3",
        occurred_at=9.98,
        policy=policy,
    )
    committed, _ = commit_safe_action_anchor(
        active,
        safe_action=commit_candidate.safe_action,
        evidence=evidence,
    )

    assert committed.available_at == 9.99
    assert committed.committed_at == 10.1
    assert not resolve_safety_anchor(
        committed,
        policy=policy,
        safety_cutoff_at=10.0,
        target_effective_at=10.09,
        execution_epoch=2,
        group_segment_generation=4,
    ).eligible
    assert resolve_safety_anchor(
        committed,
        policy=policy,
        safety_cutoff_at=10.0,
        target_effective_at=10.1,
        execution_epoch=2,
        group_segment_generation=4,
    ).eligible


def test_legacy_observation_segment_cannot_commit_shared_anchor() -> None:
    item = segment(authority_kind="legacy_execution_observation")
    observed_candidate = candidate(item=item)
    eligible = resolution(observed_candidate, target_time=10.1)

    assert eligible.status == "eligible"
    with pytest.raises(ValueError, match="authoritative_shared"):
        reference_segment_anchor_commit_evidence(eligible)
