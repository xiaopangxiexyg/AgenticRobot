"""Source owner recovery preserves watermarks even after all facts are pruned."""

from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_data
from robot_motion.action.admission_checkpoint import (
    capture_source_admission,
    restore_source_admission,
)
from robot_motion.action.proposal import (
    ProposalBuffer,
    action_proposal_semantic_payload,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import (
    ExecutionGateEventDraft,
    ExecutionGateStore,
)
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
)
from tests.test_action_proposal_contracts import _Clock, _draft, _runtime

pytestmark = pytest.mark.unit


def _restore(state, *, clock=None, **kwargs):
    values = {
        "control_domain_id": state.control_domain_id,
        "clock_domain": state.clock_domain,
        "proposal_capacity": state.proposal_capacity,
        "clock": _Clock(state.captured_at) if clock is None else clock,
    }
    values.update(kwargs)
    return restore_source_admission(state, **values)


def test_complete_source_history_restores_without_replaying_admission(monkeypatch):
    clock, coordinator, epochs, proposals = _runtime()
    for sequence in (0, 4, 12):
        assert proposals.admit(
            _draft(sequence=sequence, audit={"nested": [1, 2]})
        ).accepted
    state = capture_source_admission(coordinator, epochs, proposals)
    before = coordinator.committed_high_watermark

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Restoration cannot replay admissions/activation.")

    with monkeypatch.context() as patch:
        patch.setattr(ControlAdmissionCoordinator, "commit", forbidden)
        patch.setattr(ProducerEpochRegistry, "activate", forbidden)
        patch.setattr(ProposalBuffer, "admit", forbidden)
        restored = _restore(state)
    assert restored.coordinator is not coordinator
    assert restored.producer_epochs.coordinator is restored.coordinator
    assert restored.proposals.coordinator is restored.coordinator
    assert restored.proposals.producer_epochs is restored.producer_epochs
    assert restored.coordinator.committed_high_watermark == before
    assert restored.proposals._last_sequence == proposals._last_sequence
    for saved in state.proposal_records:
        actual = restored.proposals.get(saved.proposal_id)
        original = proposals.get(saved.proposal_id)
        assert actual is not original
        assert actual.semantic_content_hash == original.semantic_content_hash
        assert actual.available_at == original.available_at
        assert canonical_data(
            action_proposal_semantic_payload(actual)
        ) == canonical_data(action_proposal_semantic_payload(original))
        with pytest.raises(TypeError):
            actual.audit_metadata["nested"] = ()
        with pytest.raises(ValueError):
            actual.action.value("left_arm").values.setflags(write=True)
        assert not np.shares_memory(
            actual.action.value("left_arm").values,
            original.action.value("left_arm").values,
        )
    next_receipt = restored.proposals.admit(_draft(sequence=13))
    assert next_receipt.accepted
    assert next_receipt.control_admission_sequence == before + 1
    assert coordinator.committed_high_watermark == before
    assert restored.coordinator.committed_high_watermark == before + 1
    assert clock() == state.captured_at


def test_pruned_history_keeps_sequence_epoch_and_global_admission_high_watermarks():
    _clock, coordinator, epochs, proposals = _runtime()
    for sequence in (0, 5):
        assert proposals.admit(_draft(sequence=sequence)).accepted
    gate = ExecutionGateStore(runtime_session_id="session-0", coordinator=coordinator)
    for sequence, status in ((0, "hold"), (8, "ready")):
        gate.admit(
            ExecutionGateEventDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                sequence=sequence,
                status=status,
                produced_at=10.0,
                produced_clock_domain_id="control.monotonic",
                reason="checkpoint-test",
            )
        )
    before = coordinator.committed_high_watermark
    assert len(proposals.prune_expired(at_or_before=100.0)) == 2
    assert len(proposals) == 0
    state = capture_source_admission(coordinator, epochs, proposals)
    assert state.proposal_records == ()
    assert state.committed_high_watermark > max(
        r.control_admission_sequence for r in state.producer_records
    )
    restored = _restore(state)
    stale = restored.proposals.admit(_draft(sequence=5))
    assert not stale.accepted and stale.rejection_code == "sequence_rollback"
    assert restored.coordinator.committed_high_watermark == before
    accepted = restored.proposals.admit(_draft(sequence=6))
    assert accepted.accepted and accepted.control_admission_sequence == before + 1
    duplicate = restored.proposals.admit(_draft(sequence=6))
    assert not duplicate.accepted
    assert restored.coordinator.committed_high_watermark == before + 1
    restored.producer_epochs.activate(
        producer_id="policy-main",
        producer_epoch=2,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        reason="new source",
    )
    with pytest.raises(ProducerEpochError):
        restored.producer_epochs.activate(
            producer_id="policy-main",
            producer_epoch=1,
            produced_at=10.0,
            produced_clock_domain_id="control.monotonic",
            reason="old source",
        )
    old_epoch = restored.proposals.admit(_draft(sequence=7, epoch=1))
    assert (
        not old_epoch.accepted and old_epoch.rejection_code == "inactive_producer_epoch"
    )
    assert restored.proposals.admit(_draft(sequence=0, epoch=2)).accepted
    # The source snapshot and live original remain independent of all branches.
    assert epochs.current_epoch("policy-main") == 1
    assert coordinator.committed_high_watermark == before
    again = _restore(state)
    assert not again.proposals.admit(_draft(sequence=5)).accepted
    assert len(proposals) == 0


def test_old_epoch_retained_facts_are_preserved_but_not_reactivated():
    _clock, coordinator, epochs, proposals = _runtime()
    first = proposals.admit(_draft(sequence=7)).proposal
    epochs.activate(
        producer_id="policy-main",
        producer_epoch=3,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        reason="replace",
    )
    second = proposals.admit(_draft(sequence=0, epoch=3)).proposal
    state = capture_source_admission(coordinator, epochs, proposals)
    restored = _restore(state)
    cut = restored.coordinator.capture_cut().cut
    assert tuple(
        r.proposal_id for r in restored.proposals.records_at_or_before(cut)
    ) == (first.proposal_id, second.proposal_id)
    assert tuple(
        r.proposal_id for r in restored.proposals.records_for_active_epochs(cut)
    ) == (second.proposal_id,)
    assert restored.producer_epochs._activation_count == {"policy-main": 2}


def test_individually_valid_facts_cannot_rewrite_epoch_active_at_admission():
    _clock, coordinator, epochs, proposals = _runtime()
    assert proposals.admit(_draft(sequence=3)).accepted  # adm2, epoch1 active
    epochs.activate(
        producer_id="policy-main",
        producer_epoch=2,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        reason="replace",
    )  # adm3
    gate = ExecutionGateStore(runtime_session_id="session-0", coordinator=coordinator)
    gate.admit(
        ExecutionGateEventDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            sequence=0,
            status="ready",
            produced_at=10.0,
            produced_clock_domain_id="control.monotonic",
            reason="extra",
        )
    )
    state = capture_source_admission(coordinator, epochs, proposals)
    _other_clock, other_coordinator, _other_epochs, other_proposals = _runtime()
    other_gate = ExecutionGateStore(
        runtime_session_id="session-0", coordinator=other_coordinator
    )
    for seq in (0, 1):
        other_gate.admit(
            ExecutionGateEventDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                sequence=seq,
                status="ready",
                produced_at=10.0,
                produced_clock_domain_id="control.monotonic",
                reason="extra",
            )
        )
    late_epoch1 = other_proposals.admit(
        _draft(sequence=3)
    ).proposal  # adm4, independently valid
    assert late_epoch1.control_admission_sequence == 4
    with pytest.raises(ValueError, match="epoch.*admission"):
        _restore(replace(state, proposal_records=(late_epoch1,)))


def test_proposal_cannot_precede_its_epoch_activation_even_with_valid_atom_hashes():
    _clock, old_coordinator, _old_epochs, old_proposals = _runtime()
    early = old_proposals.admit(_draft(sequence=3)).proposal  # adm2
    clock = _Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0", clock_domain=old_coordinator.clock_domain, clock=clock
    )
    gate = ExecutionGateStore(runtime_session_id="session-0", coordinator=coordinator)
    for sequence in (0, 1):
        gate.admit(
            ExecutionGateEventDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                sequence=sequence,
                status="ready",
                produced_at=10.0,
                produced_clock_domain_id="control.monotonic",
                reason="before source",
            )
        )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0", coordinator=coordinator
    )
    epochs.activate(
        producer_id="policy-main",
        producer_epoch=1,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        reason="late source",
    )  # adm3
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=8,
    )
    state = capture_source_admission(coordinator, epochs, proposals)
    with pytest.raises(ValueError, match="epoch.*admission"):
        _restore(
            replace(
                state,
                proposal_records=(early,),
                last_proposal_sequences=(("policy-main", 1, 3),),
            )
        )


def test_capacity_and_destination_scope_are_not_taken_from_untrusted_saved_fields():
    _clock, coordinator, epochs, proposals = _runtime(capacity=1)
    assert proposals.admit(_draft(sequence=3)).accepted
    state = capture_source_admission(coordinator, epochs, proposals)
    with pytest.raises(ValueError, match="Destination"):
        _restore(state, proposal_capacity=2)
    with pytest.raises(ValueError, match="Destination"):
        _restore(state, control_domain_id="another-control-domain")
    with pytest.raises(ValueError, match="Destination"):
        _restore(state, clock=_Clock(state.captured_at + 1))
    with pytest.raises(ValueError, match="Destination"):
        _restore(
            state, clock_domain=replace(state.clock_domain, boot_id="another-boot")
        )
    restored = _restore(state)
    rejected = restored.proposals.admit(_draft(sequence=4))
    assert not rejected.accepted and rejected.rejection_code == "capacity_exhausted"
    assert (
        restored.coordinator.committed_high_watermark == state.committed_high_watermark
    )


def test_corrupt_records_counters_and_watermarks_fail_before_returning_owners():
    _clock, coordinator, epochs, proposals = _runtime()
    assert proposals.admit(_draft(sequence=3)).accepted
    state = capture_source_admission(coordinator, epochs, proposals)
    record = state.proposal_records[0]
    original = proposals.get(record.proposal_id)
    bad_states = [
        replace(state, committed_high_watermark=0),
        replace(state, committed_high_watermark=True),
        replace(state, captured_at=float("nan")),
        replace(state, active_epoch_by_producer={"policy-main": 99}),
        replace(state, activation_count_by_producer={"policy-main": 0}),
        replace(state, last_proposal_sequences=()),
        replace(state, last_proposal_sequences=(("policy-main", 1, 2),)),
        replace(state, last_proposal_sequences=(("unknown", 1, 9),)),
        replace(state, last_proposal_sequences=(("policy-main", 1, 3),) * 2),
        replace(state, proposal_records=(replace(record, available_at=20.0),)),
        replace(
            state, proposal_records=(replace(record, semantic_content_hash="corrupt"),)
        ),
        replace(state, producer_records=state.producer_records * 2),
        replace(state, proposal_records=state.proposal_records * 2),
    ]
    for bad in bad_states:
        with pytest.raises(ValueError):
            _restore(bad)
    assert coordinator.committed_high_watermark == state.committed_high_watermark
    assert proposals.get(record.proposal_id) is original


def test_capture_rejects_disconnected_owners_and_corrupt_indexes(monkeypatch):
    _clock, coordinator, epochs, proposals = _runtime()
    other = _runtime()[1]
    with pytest.raises(ValueError, match="scope/layout"):
        capture_source_admission(other, epochs, proposals)
    assert proposals.admit(_draft(sequence=1)).accepted
    with monkeypatch.context() as patch:
        patch.setattr(proposals, "_by_id", {})
        with pytest.raises(ValueError, match="retained index"):
            capture_source_admission(coordinator, epochs, proposals)
    with monkeypatch.context() as patch:
        patch.setattr(coordinator, "unknown_state", 1, raising=False)
        with pytest.raises(ValueError, match="scope/layout"):
            capture_source_admission(coordinator, epochs, proposals)
