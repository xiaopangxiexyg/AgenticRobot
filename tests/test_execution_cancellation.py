from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from robot_motion.position_control.capability import ExecutionCapabilityProfile
from robot_motion.position_control.contracts import (
    PositionCommandTick,
    PositionReferenceFrame,
    PositionSideReference,
)
from robot_motion.runtime.cancellation import (
    CANCELLATION_REASONS,
    CancellationAcknowledgementTracker,
    EpochVector,
    ExecutionCancellationRequest,
    ExecutionEpochGuard,
    ExecutionEpochRegistry,
    ExecutionWorkItem,
    is_legal_epoch_transition,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def capability(**overrides) -> ExecutionCapabilityProfile:
    values = {
        "profile_id": "m1c-synthetic-v1",
        "validated_max_selected_keyframe_hz": 60.0,
        "configured_selected_keyframe_hz": 30.0,
        "validated_max_coupled_ik_hz": 60.0,
        "configured_coupled_ik_hz": 60.0,
        "reference_resample_hz": 75.0,
        "command_hz": 150.0,
        "backend_servo_hz": 150.0,
        "validated_action_spaces": ("bimanual_ee_pose_v1",),
        "validated_selection_groups": ("arms", "hands"),
        "supports_local_group_stop": True,
        "acceptance_evidence_id": "synthetic-m1c-only",
        "decision_runtime_budget_sec": 0.005,
        "ik_runtime_budget_sec": 0.012,
        "end_to_end_control_budget_sec": 0.02,
        "cancellation_ack_deadline_sec": 0.1,
        "required_cancellation_ack_levels": (
            "software_queue",
            "sdk_request",
        ),
    }
    values.update(overrides)
    return ExecutionCapabilityProfile(**values)


def registry(clock: ManualClock | None = None) -> ExecutionEpochRegistry:
    return ExecutionEpochRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        clock_domain_id="control-clock",
        initial_execution_epoch=3,
        initial_segment_generation_by_group={"arms": 7, "hands": 2},
        clock=ManualClock() if clock is None else clock,
    )


def cancellation_request(**overrides) -> ExecutionCancellationRequest:
    values = {
        "reason": "hold",
        "authority_epoch": 11,
        "requested_scope": "control_domain",
        "invalidated_decision_ids": ("decision-old",),
        "invalidated_reference_segment_ids": ("segment-old",),
        "invalidated_queued_command_ids": ("command-old",),
        "source_event_ids": ("gate-hold",),
    }
    values.update(overrides)
    return ExecutionCancellationRequest(**values)


def work_item(
    *,
    execution_epoch: int = 3,
    generations: dict[str, int] | None = None,
    kind: str = "backend_command",
) -> ExecutionWorkItem:
    return ExecutionWorkItem(
        work_item_id=f"{kind}-0",
        work_kind=kind,
        control_domain_id="robot-0",
        authority_epoch=11,
        execution_epoch=execution_epoch,
        segment_generation_by_group={"arms": 7} if generations is None else generations,
        source_decision_id="decision-old",
        source_reference_segment_id="segment-old",
    )


def test_domain_cancellation_atomically_advances_execution_epoch() -> None:
    value = registry()
    event = value.cancel(cancellation_request(), capability=capability())
    after = value.snapshot()

    assert event.previous_execution_epoch == 3
    assert event.execution_epoch == 4
    assert after.execution_epoch == 4
    assert event.authority_epoch_evidence == 11
    assert event.invalidated_decision_ids == ("decision-old",)
    assert event.invalidated_reference_segment_ids == ("segment-old",)
    assert event.invalidated_queued_command_ids == ("command-old",)
    assert event.requests_backend_hold
    assert not event.affects_decision_rate
    assert not hasattr(event, "selected_action")


@pytest.mark.parametrize("reason", CANCELLATION_REASONS)
def test_all_cancellation_reasons_are_typed(reason: str) -> None:
    event = registry().cancel(
        cancellation_request(reason=reason),
        capability=capability(),
    )
    assert event.reason == reason


def test_local_cancellation_only_advances_requested_segment_generation() -> None:
    value = registry()
    event = value.cancel(
        cancellation_request(
            requested_scope="selection_groups",
            selection_group_ids=("hands",),
        ),
        capability=capability(supports_local_group_stop=True),
    )

    assert event.effective_scope == "selection_groups"
    assert event.execution_epoch == event.previous_execution_epoch == 3
    assert event.segment_generation_after["arms"] == 7
    assert event.segment_generation_after["hands"] == 3
    assert not event.requests_backend_hold


def test_unsupported_local_stop_escalates_to_domain_cancellation() -> None:
    value = registry()
    event = value.cancel(
        cancellation_request(
            requested_scope="selection_groups",
            selection_group_ids=("arms",),
        ),
        capability=capability(supports_local_group_stop=False),
    )

    assert event.requested_scope == "selection_groups"
    assert event.effective_scope == "control_domain"
    assert event.scope_escalation_reason == "backend_does_not_support_local_group_stop"
    assert event.execution_epoch == 4
    assert event.requests_backend_hold


def test_fault_cannot_remain_a_local_generation_only_cancel() -> None:
    value = registry()
    event = value.cancel(
        cancellation_request(
            reason="fault",
            requested_scope="selection_groups",
            selection_group_ids=("hands",),
        ),
        capability=capability(supports_local_group_stop=True),
    )

    assert event.effective_scope == "control_domain"
    assert event.scope_escalation_reason == "fault_requires_domain_cancellation"
    assert event.execution_epoch == 4
    assert event.fault_latched
    assert value.snapshot().fault_latched


def test_worker_checks_epoch_at_dequeue_and_again_before_backend_call() -> None:
    clock = ManualClock()
    value = registry(clock)
    guard = ExecutionEpochGuard(value, clock=clock)
    item = work_item()

    dequeued = guard.check_at_dequeue(item)
    assert dequeued.accepted

    clock.value = 10.01
    value.cancel(cancellation_request(), capability=capability())
    before_sdk = guard.check_before_backend_call(item)

    assert not before_sdk.accepted
    assert before_sdk.status == "execution_epoch_mismatch"
    assert before_sdk.current_execution_epoch == 4


def test_fault_latch_blocks_motion_but_allows_typed_new_epoch_safety_work() -> None:
    value = registry()
    guard = ExecutionEpochGuard(value)
    event = value.cancel(
        cancellation_request(reason="fault"),
        capability=capability(),
    )
    motion = work_item(execution_epoch=event.execution_epoch)
    safety = ExecutionWorkItem(
        work_item_id="quick-stop-tick-1",
        work_kind="safety_backend_command",
        control_domain_id="robot-0",
        authority_epoch=11,
        execution_epoch=event.execution_epoch,
        segment_generation_by_group={"arms": 7},
        source_safety_operation_id="quick-stop-operation-1",
        source_safety_tick_evidence_id="quick-stop-tick-evidence-1",
    )

    motion_check = guard.check_before_backend_call(motion)
    safety_check = guard.check_before_backend_call(safety)

    assert motion_check.status == "fault_latched"
    assert not motion_check.accepted
    assert safety_check.status == "accepted"
    assert safety_check.accepted


def test_local_generation_invalidates_only_work_covering_that_group() -> None:
    value = registry()
    guard = ExecutionEpochGuard(value)
    event = value.cancel(
        cancellation_request(
            requested_scope="selection_groups",
            selection_group_ids=("hands",),
        ),
        capability=capability(),
    )

    old_hands = work_item(generations={"hands": 2})
    current_arms = work_item(generations={"arms": 7})
    assert guard.check_at_dequeue(old_hands).status == "segment_generation_mismatch"
    assert guard.check_at_dequeue(current_arms).accepted
    assert event.segment_generation_after["hands"] == 3


def test_recovery_creates_new_epoch_and_old_lookahead_cannot_revive() -> None:
    value = registry()
    value.cancel(cancellation_request(reason="fault"), capability=capability())
    recovery = value.recover(authority_epoch=11, source_event_ids=("fault-cleared",))
    guard = ExecutionEpochGuard(value)

    assert recovery.previous_execution_epoch == 4
    assert recovery.execution_epoch == 5
    assert not value.snapshot().fault_latched
    assert guard.check_at_dequeue(work_item(execution_epoch=3)).status == "execution_epoch_mismatch"
    assert guard.check_at_dequeue(work_item(execution_epoch=5)).accepted


def test_acknowledgement_levels_are_distinct_and_deadline_is_enforced() -> None:
    clock = ManualClock()
    value = registry(clock)
    profile = capability(
        required_cancellation_ack_levels=(
            "software_queue",
            "sdk_request",
            "backend_state",
            "measured_motion_stopped",
        ),
        supports_backend_state_ack=True,
        supports_measured_motion_stopped_ack=True,
    )
    event = value.cancel(cancellation_request(), capability=profile)
    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(event)

    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id="control-clock",
        source_id="command-worker",
    )
    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="sdk_request",
        clock_domain_id="control-clock",
        source_id="fake-sdk",
    )
    partial = tracker.receipt(event.cancellation_id)
    assert not partial.complete
    assert partial.pending_levels == ("backend_state", "measured_motion_stopped")

    clock.value = event.acknowledgement_deadline_at + 0.001
    timed_out = tracker.receipt(event.cancellation_id)
    assert timed_out.timed_out
    assert timed_out.requires_fault_escalation
    escalation = tracker.fault_escalation_request(event.cancellation_id)
    assert escalation is not None
    assert escalation.requested_gate_status == "fault"
    assert escalation.reason == "cancellation_ack_deadline_missed"
    assert escalation.missing_ack_levels == (
        "backend_state",
        "measured_motion_stopped",
    )


def test_acknowledgement_after_deadline_cannot_erase_fault_escalation() -> None:
    clock = ManualClock()
    value = registry(clock)
    event = value.cancel(
        cancellation_request(),
        capability=capability(required_cancellation_ack_levels=("software_queue",)),
    )
    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(event)
    clock.value = event.acknowledgement_deadline_at + 0.001

    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id="control-clock",
        source_id="late-worker-drain",
    )
    receipt = tracker.receipt(event.cancellation_id)
    escalation = tracker.fault_escalation_request(event.cancellation_id)

    assert not receipt.complete
    assert receipt.late_levels == ("software_queue",)
    assert receipt.pending_levels == ("software_queue",)
    assert receipt.timed_out
    assert escalation is not None
    assert escalation.reason == "cancellation_ack_deadline_missed"
    assert escalation.missing_ack_levels == ("software_queue",)


def test_cancel_ack_recover_harness_reaches_a_fresh_executable_epoch() -> None:
    clock = ManualClock()
    value = registry(clock)
    guard = ExecutionEpochGuard(value, clock=clock)
    old_reference = work_item(kind="reference_segment")
    old_command = work_item(kind="position_command")
    old_backend = work_item(kind="backend_command")
    assert guard.check_at_dequeue(old_reference).accepted
    assert guard.check_at_dequeue(old_command).accepted
    assert guard.check_at_dequeue(old_backend).accepted

    event = value.cancel(cancellation_request(), capability=capability())
    for item in (old_reference, old_command, old_backend):
        assert not guard.check_before_backend_call(item).accepted

    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(event)
    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id="control-clock",
        source_id="reference-and-command-workers",
    )
    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="sdk_request",
        clock_domain_id="control-clock",
        source_id="fake-backend-hold",
    )
    assert tracker.receipt(event.cancellation_id).complete

    recovery = value.recover(authority_epoch=11, source_event_ids=("manual-rearm",))
    fresh = work_item(execution_epoch=recovery.execution_epoch)
    assert guard.check_at_dequeue(fresh).accepted
    assert guard.check_before_backend_call(fresh).accepted


def test_authority_and_execution_epoch_transition_matrix_is_explicit() -> None:
    start = EpochVector(authority_epoch=4, execution_epoch=7)
    assert is_legal_epoch_transition(
        start,
        EpochVector(authority_epoch=5, execution_epoch=7),
        transition_kind="authority_update",
    )
    assert is_legal_epoch_transition(
        start,
        EpochVector(authority_epoch=4, execution_epoch=8),
        transition_kind="domain_cancellation",
    )
    assert is_legal_epoch_transition(
        start,
        EpochVector(authority_epoch=5, execution_epoch=8),
        transition_kind="authority_handoff_cancellation",
    )
    assert not is_legal_epoch_transition(
        start,
        EpochVector(authority_epoch=5, execution_epoch=8),
        transition_kind="authority_update",
    )


def test_execution_epoch_increment_is_atomic_under_concurrent_cancellation() -> None:
    value = registry()

    def cancel_once(index: int) -> int:
        return value.cancel(
            cancellation_request(source_event_ids=(f"hold-{index}",)),
            capability=capability(),
        ).execution_epoch

    with ThreadPoolExecutor(max_workers=8) as pool:
        epochs = tuple(pool.map(cancel_once, range(32)))

    assert sorted(epochs) == list(range(4, 36))
    assert value.snapshot().execution_epoch == 35


def test_position_reference_and_command_expose_execution_epoch_alias() -> None:
    side = PositionSideReference(
        side="left",
        q_ik_raw=np.zeros(2),
        q_ref=np.zeros(2),
        pose_ref=np.zeros(7),
    )
    frame = PositionReferenceFrame(
        epoch=9,
        sequence=0,
        source_wall_time=1.0,
        generated_wall_time=1.0,
        published_wall_time=1.0,
        generated_monotonic_time=1.0,
        published_monotonic_time=1.0,
        references={"left": side},
    )
    tick = PositionCommandTick(
        epoch=9,
        command_sequence=0,
        reference_sequence=0,
        emitted_monotonic_time=1.0,
        reference_published_monotonic_time=1.0,
        segment_start_monotonic_time=1.0,
        segment_duration_sec=0.01,
        interpolation_alpha=0.0,
        q_cmd_by_side={"left": np.zeros(2)},
        q_target_by_side={"left": np.ones(2)},
    )

    assert frame.execution_epoch == 9
    assert tick.execution_epoch == 9


def test_capability_rejects_required_ack_level_without_evidence_support() -> None:
    with pytest.raises(ValueError, match="backend_state"):
        capability(required_cancellation_ack_levels=("backend_state",))
    with pytest.raises(ValueError, match="measured_motion_stopped"):
        capability(required_cancellation_ack_levels=("measured_motion_stopped",))
