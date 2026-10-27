"""Logical execution authority recovery; no backend or hardware is attached."""

from dataclasses import replace

import pytest

from robot_motion.runtime.cancellation import (
    CancellationAcknowledgementTracker,
    ExecutionEpochGuard,
    ExecutionEpochRegistry,
    ExecutionWorkItem,
)
from robot_motion.runtime.dispatch_owner import DispatchOwnerRegistry
from robot_motion.runtime.execution_checkpoint import (
    capture_execution_authority,
    restore_execution_authority,
)
from tests.test_dispatch_owner import (
    ManualClock,
    cancel,
    execution_registry,
    motion_permit,
    owner_registry,
)

pytestmark = pytest.mark.unit


def _owners():
    clock = ManualClock()
    dispatch, epochs = owner_registry(clock), execution_registry(clock)
    token = dispatch.grant_initial_owner(
        owner_runtime_id="shared-0", owner_kind="shared_arbitration_runtime"
    )
    tracker = CancellationAcknowledgementTracker(clock=clock)
    return clock, dispatch, epochs, tracker, token


def _restore(state, **kwargs):
    options = {
        "runtime_session_id": "session-0",
        "control_domain_id": "robot-0",
        "clock_domain_id": "control-clock",
        "owner_runtime_id": "shared-0",
        "owner_kind": "shared_arbitration_runtime",
        "clock": ManualClock(state.captured_at),
    }
    options.update(kwargs)
    return restore_execution_authority(state, **options)


def _authorize(dispatch, permit):
    snap = dispatch.snapshot()
    return dispatch.authorize(
        permit,
        current_authority_epoch=snap.current_authority_epoch,
        current_execution_epoch=snap.current_execution_epoch,
    )


def _ack(tracker, event, level):
    return tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level=level,
        clock_domain_id="control-clock",
        source_id="deterministic-test-worker",
    )


def test_consumed_permits_and_exact_owner_state_survive_without_regrant(monkeypatch):
    _clock, dispatch, epochs, tracker, token = _owners()
    permit = motion_permit(dispatch, token)
    assert _authorize(dispatch, permit).accepted
    state = capture_execution_authority(dispatch, epochs, tracker)
    old_sequence = dispatch._next_fact_sequence

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery cannot re-grant or replay execution events.")

    with monkeypatch.context() as patch:
        for name in (
            "grant_initial_owner",
            "_grant_owner_locked",
            "issue_motion_permit",
        ):
            patch.setattr(DispatchOwnerRegistry, name, forbidden)
        for name in ("cancel", "recover"):
            patch.setattr(ExecutionEpochRegistry, name, forbidden)
        for name in ("register", "acknowledge"):
            patch.setattr(CancellationAcknowledgementTracker, name, forbidden)
        restored = _restore(state)
    candidate = restored.dispatch_owners
    assert candidate is not dispatch
    assert candidate.snapshot() == dispatch.snapshot()
    assert restored.execution_epochs.snapshot() == epochs.snapshot()
    assert candidate._next_fact_sequence == old_sequence
    replay = _authorize(candidate, permit)
    assert not replay.accepted and replay.status == "permit_already_consumed"
    next_permit = motion_permit(candidate, candidate.snapshot().current_token)
    assert _authorize(candidate, next_permit).accepted
    assert candidate._next_fact_sequence == old_sequence + 1
    assert dispatch._next_fact_sequence == old_sequence
    assert _authorize(dispatch, next_permit).status == "permit_not_issued_by_registry"
    again = _restore(state)
    assert again.dispatch_owners._next_fact_sequence == old_sequence
    assert _authorize(again.dispatch_owners, permit).status == "permit_already_consumed"


def test_live_epochs_advance_without_rewriting_acquisition_epochs_and_pending_ack():
    clock, dispatch, epochs, tracker, token = _owners()
    dispatch.observe_authority_epoch(authority_epoch=2, evidence_event_id="authority-2")
    event = cancel(epochs, reason="hold", authority_epoch=2)
    dispatch.observe_execution_cancellation(event)
    tracker.register(event)
    _ack(tracker, event, "software_queue")
    state = capture_execution_authority(dispatch, epochs, tracker)
    restored = _restore(state)
    snap = restored.dispatch_owners.snapshot()
    assert snap.current_authority_epoch == 2
    assert snap.current_execution_epoch == 4
    assert snap.current_token.authority_epoch_at_acquire == 1
    assert snap.current_token.execution_epoch_at_acquire == 3
    assert snap.current_token.semantic_content_hash == token.semantic_content_hash
    assert restored.cancellation_acknowledgements.receipt(
        event.cancellation_id
    ) == tracker.receipt(event.cancellation_id)
    candidate_clock = restored.execution_epochs._clock
    candidate_clock.value += 1
    candidate = restored.cancellation_acknowledgements.receipt(event.cancellation_id)
    assert not candidate.complete and candidate.timed_out
    assert candidate.pending_levels == ("sdk_request",)
    assert not tracker.receipt(event.cancellation_id).timed_out
    assert clock() == state.captured_at


def test_fault_latch_and_recovery_keep_next_sequence_and_old_cancellation():
    clock, dispatch, epochs, tracker, _token = _owners()
    event = cancel(epochs, reason="fault", authority_epoch=1)
    dispatch.observe_execution_cancellation(event)
    tracker.register(event)
    _ack(tracker, event, "software_queue")
    _ack(tracker, event, "sdk_request")
    faulted = _restore(capture_execution_authority(dispatch, epochs, tracker))
    work = ExecutionWorkItem(
        work_item_id="next-work",
        work_kind="backend_command",
        control_domain_id="robot-0",
        authority_epoch=1,
        execution_epoch=4,
        segment_generation_by_group={},
    )
    assert (
        ExecutionEpochGuard(faulted.execution_epochs, clock=clock)
        .check_before_backend_call(work)
        .status
        == "fault_latched"
    )
    recovery = epochs.recover(
        authority_epoch=1, source_event_ids=("operator-recovery",)
    )
    dispatch.observe_execution_recovery(recovery)
    state = capture_execution_authority(dispatch, epochs, tracker)
    restored = _restore(state)
    assert not restored.execution_epochs.snapshot().fault_latched
    assert (
        restored.execution_epochs.snapshot().last_cancellation_id
        == event.cancellation_id
    )
    assert restored.execution_epochs._next_sequence == 2
    assert restored.cancellation_acknowledgements.receipt(
        event.cancellation_id
    ).complete
    next_event = cancel(restored.execution_epochs, reason="hold", authority_epoch=1)
    assert next_event.sequence == 2
    assert epochs._next_sequence == 2


def test_outstanding_permit_and_incomplete_owner_transition_are_rejected():
    _clock, dispatch, epochs, tracker, token = _owners()
    permit = motion_permit(dispatch, token)
    with pytest.raises(ValueError, match="consumed"):
        capture_execution_authority(dispatch, epochs, tracker)
    assert _authorize(dispatch, permit).accepted
    event = cancel(epochs, reason="owner_rollback", authority_epoch=1)
    dispatch.begin_transition(
        current_token=token,
        transition_kind="owner_rollback",
        cancellation=event,
        gate_status="hold",
        gate_epoch=1,
    )
    tracker.register(event)
    with pytest.raises(ValueError, match="transition"):
        capture_execution_authority(dispatch, epochs, tracker)
    assert dispatch.snapshot().state == "transitioning_hold"
    assert tracker.receipt(event.cancellation_id).pending_levels


def test_destination_scope_owner_clock_and_invalid_records_do_not_relax_checks():
    _clock, dispatch, epochs, tracker, _token = _owners()
    state = capture_execution_authority(dispatch, epochs, tracker)
    for kwargs in (
        {"runtime_session_id": "foreign"},
        {"control_domain_id": "foreign"},
        {"clock_domain_id": "foreign"},
        {"owner_runtime_id": "foreign"},
        {"owner_kind": "legacy_control_path"},
        {"clock": ManualClock(state.captured_at + 1)},
    ):
        with pytest.raises(ValueError, match="Destination"):
            _restore(state, **kwargs)
    for field, value in (
        ("_next_fact_sequence", 0),
        ("_ever_granted", False),
        ("_authority_epoch", 0),
        ("_execution_epoch", 2),
    ):
        bad = dict(state.dispatch)
        bad[field] = value
        with pytest.raises(ValueError):
            _restore(replace(state, dispatch=bad))
    token = state.dispatch["_current_token"]
    bad = dict(state.dispatch)
    bad["_current_token"] = replace(token, semantic_content_hash="corrupt")
    with pytest.raises(ValueError, match="hash"):
        _restore(replace(state, dispatch=bad))
    assert (
        dispatch.snapshot().current_token.semantic_content_hash
        == token.semantic_content_hash
    )


def test_scope_and_unknown_owner_layout_or_ack_index_are_rejected(monkeypatch):
    _clock, dispatch, epochs, tracker, _token = _owners()
    with monkeypatch.context() as patch:
        patch.setattr(epochs, "_clock", ManualClock())
        with pytest.raises(ValueError, match="scope/clock"):
            capture_execution_authority(dispatch, epochs, tracker)
    with monkeypatch.context() as patch:
        patch.setattr(dispatch, "unknown_state", 1, raising=False)
        with pytest.raises(ValueError, match="layout"):
            capture_execution_authority(dispatch, epochs, tracker)
    with monkeypatch.context() as patch:
        patch.setattr(tracker, "_acks", {"unknown-event": {}})
        with pytest.raises(ValueError, match="indexes"):
            capture_execution_authority(dispatch, epochs, tracker)
