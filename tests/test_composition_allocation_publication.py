from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from control_composition.activation import build_composition_activation_scope
from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
    ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED,
    ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED,
    ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
    ALLOCATION_PUBLICATION_OUTCOME_LATE,
    ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION,
    ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
    ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN,
    AllocationPublicationBasis,
    AllocationPublicationRuntime,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
    CompositionAllocationSpec,
    empty_composition_allocation_state,
    selected_allocation_modes_from_context,
)
from control_composition.composition import (
    CompositionContextAdmissionReceipt,
    CompositionContextStore,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_station import (
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.composition import (
    StationAllocationPublicationPort,
    StationAllocationPublicationProfile,
)
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class Clock:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _spec() -> CompositionAllocationSpec:
    thresholds = ComponentAllocationEvidencePolicy(
        relative_enter_level=1.0,
        relative_exit_level=0.5,
        absolute_enter_level=3.0,
        absolute_exit_level=2.0,
    )
    return CompositionAllocationSpec(
        allocation_spec_id="tool-allocation",
        base_absolute_role="base",
        relative_overlay_role="operator_relative",
        absolute_overlay_role="operator_absolute",
        component_policies=(
            ComponentAllocationPolicy(
                component_id="tool",
                motion_source_role="operator_activity",
                residual_source_role="operator_relative",
                motion_policy=thresholds,
                residual_magnitude_policy=thresholds,
                minimum_dwell_sec=0.1,
            ),
        ),
    )


def _activation(*, activated_at: float = 10.0, session_epoch: int = 1):
    return build_composition_activation_scope(
        operation_spec_id="allocation-test",
        operation_spec_hash="allocation-test-operation-hash",
        session_epoch=session_epoch,
        activated_at=activated_at,
        clock_domain_id="control.monotonic",
        input_barrier_at=activated_at,
        required_roles=("base", "operator_relative", "operator_absolute"),
    )


def _basis(
    activation,
    *,
    index: int,
    cut_at: float,
    level: float = 1.5,
    valid_until: float = 12.0,
    support_until: float | None = None,
    anchor_request_generation: int = 0,
) -> AllocationPublicationBasis:
    available_at = cut_at - 0.01
    activity = ComponentSourceActivityObservation(
        source_role="operator_activity",
        metric_provider_id="test.activity",
        metric_provider_hash="test-activity-hash",
        component_id="tool",
        representation_family="cartesian_pose",
        window_started_at=cut_at - 0.2,
        window_ended_at=cut_at - 0.02,
        available_at=available_at,
        valid_until=valid_until,
        clock_domain_id="control.monotonic",
        metric_values={"motion": level},
        normalization_scale_by_metric={"motion": 1.0},
        source_fact_ids=(f"human-window:{index}",),
        source_fact_hashes=(f"human-window-hash:{index}",),
    )
    magnitude = ComponentResidualMagnitudeObservation(
        source_role="operator_relative",
        metric_provider_id="test.magnitude",
        metric_provider_hash="test-magnitude-hash",
        component_id="tool",
        representation_family="cartesian_pose",
        sampled_at=cut_at - 0.02,
        available_at=available_at,
        valid_until=valid_until,
        clock_domain_id="control.monotonic",
        anchor_receipt_id=f"anchor-receipt:{anchor_request_generation}",
        anchor_receipt_hash=f"anchor-receipt-hash:{anchor_request_generation}",
        anchor_request_generation=anchor_request_generation,
        residual_action_id=f"residual:{index}",
        residual_action_hash=f"residual-hash:{index}",
        metric_values={"magnitude": level},
        normalization_scale_by_metric={"magnitude": 1.0},
    )
    return AllocationPublicationBasis(
        allocation_scope_id="operator-tool",
        control_cut=ControlCut(
            runtime_session_id="session",
            control_domain_id="robot",
            clock_domain_id="control.monotonic",
            clock_reset_epoch=0,
            cutoff_at=cut_at,
            cutoff_admission_sequence=index,
        ),
        activation_scope=activation,
        activity_by_component={"tool": activity},
        residual_magnitude_by_component={"tool": magnitude},
        requested_effective_from=cut_at + 0.01,
        support_until=(valid_until if support_until is None else support_until),
    )


def _store(clock: Clock):
    coordinator = ControlAdmissionCoordinator(
        "robot",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session",
            boot_id="boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session",
        coordinator=coordinator,
    )
    epochs.activate(
        producer_id="allocation-producer",
        producer_epoch=1,
        produced_at=9.9,
        produced_clock_domain_id="producer.monotonic",
        reason="test_start",
    )
    store = CompositionContextStore(
        runtime_session_id="session",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="allocation-publication-test",
            history_window_sec=0.0,
            max_future_lead_sec=5.0,
            max_effective_duration_sec=5.0,
        ),
    )
    return store, coordinator, epochs


def _publisher(clock: Clock, spec, activation):
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)
    return AllocationPublicationRuntime(
        allocation_scope_id="operator-tool",
        allocation_spec=spec,
        activation_scope=activation,
        runtime_session_id="session",
        control_domain_id="robot",
        control_clock_reset_epoch=0,
        context_channel="source_allocation",
        producer_id="allocation-producer",
        producer_epoch=1,
        initial_reducer_state=state,
        clock=clock,
    )


def _active_station_barrier() -> StationPublicationBarrier:
    barrier = StationPublicationBarrier(source_slot_id="composer")
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:start-composition",
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=10.0,
            completed_clock_domain_id="control.monotonic",
            status="composition_session_ready",
            evidence_refs=("allocation-test",),
        ),
        runtime_session_id="session",
        control_domain_id="robot",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        admission_sequence=0,
    )
    barrier.enable_from_composition_session_result(result)
    return barrier


def test_state_commits_only_after_exact_context_admission() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, coordinator, epochs = _store(clock)
    publisher = _publisher(clock, spec, activation)
    initial_hash = publisher.committed_reducer_state.semantic_content_hash

    prepared = publisher.offer_basis(
        _basis(activation, index=1, cut_at=10.2)
    )
    assert prepared.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    assert publisher.committed_reducer_state.semantic_content_hash == initial_hash
    pending = publisher.snapshot().pending_attempt
    assert pending is not None

    accepted = publisher.attempt_admission(
        store.admit,
        resolve_context=store.record_by_id,
    )
    assert accepted is not None
    assert accepted.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
    assert publisher.committed_reducer_state.revision == 1
    assert publisher.committed_reducer_state.semantic_content_hash == (
        pending.candidate_next_state.semantic_content_hash
    )

    clock.value = 10.3
    cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    snapshot = store.capture_snapshot(
        cut,
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.3,
        required_channels=("source_allocation",),
    )
    context = snapshot.selected_context_by_channel["source_allocation"]
    modes = selected_allocation_modes_from_context(
        context,
        expected_allocation_spec_hash=spec.semantic_content_hash,
        expected_component_ids=("tool",),
        target_effective_at=10.3,
        expected_composition_activation_id=activation.activation_id,
        expected_composition_activation_hash=activation.semantic_content_hash,
    )
    assert modes == {"tool": "relative_overlay"}


def test_unknown_after_store_commit_retries_same_attempt_and_reconciles() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))
    pending_before = publisher.snapshot().pending_attempt
    assert pending_before is not None

    def admit_then_lose_receipt(draft):
        receipt = store.admit(draft)
        assert receipt.accepted
        raise RuntimeError("simulated receipt loss")

    unknown = publisher.attempt_admission(admit_then_lose_receipt)
    assert unknown is not None
    assert unknown.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN
    pending_after = publisher.snapshot().pending_attempt
    assert pending_after is pending_before
    assert pending_after.context_draft is pending_before.context_draft
    assert publisher.committed_reducer_state.revision == 0

    reconciled = publisher.attempt_admission(
        store.admit,
        resolve_context=store.record_by_id,
    )
    assert reconciled is not None
    assert reconciled.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING
    )
    assert reconciled.attempt_id == pending_before.attempt_id
    assert reconciled.publication_sequence == 0
    assert publisher.committed_reducer_state.revision == 1


def test_terminal_rejection_keeps_state_and_consumes_sequence() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))

    rejected = publisher.attempt_admission(
        lambda _draft: CompositionContextAdmissionReceipt(
            accepted=False,
            rejection_code="retention_window_violation",
            message="terminal test rejection",
        )
    )
    assert rejected is not None
    assert rejected.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION
    )
    assert publisher.committed_reducer_state.revision == 0
    assert publisher.snapshot().last_terminal_sequence == 0

    clock.value = 10.41
    next_outcome = publisher.offer_basis(
        _basis(activation, index=2, cut_at=10.4)
    )
    assert next_outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    )
    assert next_outcome.publication_sequence == 1
    accepted_after_gap = publisher.attempt_admission(store.admit)
    assert accepted_after_gap is not None
    assert accepted_after_gap.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
    )


def test_pending_keeps_exact_attempt_and_only_latest_newer_basis() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=2, cut_at=10.2))
    pending = publisher.snapshot().pending_attempt
    assert pending is not None

    older = publisher.offer_basis(_basis(activation, index=1, cut_at=10.19))
    assert older.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED
    second = _basis(activation, index=3, cut_at=10.3)
    third = _basis(activation, index=4, cut_at=10.4)
    assert publisher.offer_basis(second).outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED
    )
    assert publisher.offer_basis(third).outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED
    )
    snapshot = publisher.snapshot()
    assert snapshot.pending_attempt is pending
    assert snapshot.latest_coalesced_basis is third

    publisher.attempt_admission(
        lambda _draft: CompositionContextAdmissionReceipt(
            accepted=False,
            rejection_code="test_terminal",
            message="finish first pending",
        )
    )
    clock.value = 10.41
    next_outcome = publisher.prepare_latest_coalesced()
    assert next_outcome is not None
    assert next_outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    )
    assert publisher.snapshot().pending_attempt.activity_basis is third


def test_slow_admission_does_not_block_latest_basis_coalescing() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))
    entered = threading.Event()
    release = threading.Event()
    outcomes = []

    def slow_admit(_draft):
        entered.set()
        assert release.wait(timeout=2.0)
        return CompositionContextAdmissionReceipt(
            accepted=False,
            rejection_code="test_terminal",
            message="bounded slow admission",
        )

    worker = threading.Thread(
        target=lambda: outcomes.append(publisher.attempt_admission(slow_admit)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=2.0)
    latest = _basis(activation, index=2, cut_at=10.3)
    coalesced = publisher.offer_basis(latest)
    assert coalesced.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED
    )
    assert publisher.snapshot().latest_coalesced_basis is latest
    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcomes[0].outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION
    )


def test_publication_support_is_intersected_and_expired_basis_is_not_reserved() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    publisher = _publisher(clock, spec, activation)
    outcome = publisher.offer_basis(
        _basis(
            activation,
            index=1,
            cut_at=10.2,
            valid_until=12.0,
            support_until=10.5,
        )
    )
    assert outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    pending = publisher.snapshot().pending_attempt
    assert pending is not None
    assert pending.context_draft.effective_until == pytest.approx(10.5)

    other = _publisher(clock, spec, activation)
    delayed = _basis(
        activation,
        index=2,
        cut_at=10.2,
        valid_until=12.0,
        support_until=10.3,
    )
    clock.value = 10.31
    expired = other.offer_basis(delayed)
    assert expired.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION
    assert other.snapshot().last_reserved_sequence == -1
    assert other.snapshot().pending_attempt is None


def test_duplicate_conflict_is_terminal_but_missing_resolver_is_unknown() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))
    pending = publisher.snapshot().pending_attempt
    assert pending is not None
    conflict = replace(
        pending.context_draft,
        semantic_payload={"conflicting": True},
    )
    assert store.admit(conflict).accepted

    unknown = publisher.attempt_admission(store.admit)
    assert unknown is not None
    assert unknown.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN
    assert publisher.snapshot().pending_attempt is pending

    conflict_result = publisher.attempt_admission(
        store.admit,
        resolve_context=store.record_by_id,
    )
    assert conflict_result is not None
    assert conflict_result.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION
    )
    assert publisher.committed_reducer_state.revision == 0


def test_invalidation_fences_pending_and_late_accepted_receipt() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))
    pending = publisher.snapshot().pending_attempt
    assert pending is not None
    receipt = store.admit(pending.context_draft)
    assert receipt.accepted

    publisher.invalidate(reason="composition_activation_closed")
    late = publisher.resolve_admission(pending.attempt_id, receipt)
    assert late.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_LATE
    assert publisher.committed_reducer_state.revision == 0
    assert publisher.snapshot().pending_attempt is None

    next_activation = _activation(activated_at=10.5, session_epoch=2)
    context = receipt.context
    assert context is not None
    with pytest.raises(ValueError, match="another activation"):
        selected_allocation_modes_from_context(
            context,
            expected_allocation_spec_hash=spec.semantic_content_hash,
            expected_component_ids=("tool",),
            target_effective_at=10.3,
            expected_composition_activation_id=next_activation.activation_id,
            expected_composition_activation_hash=(
                next_activation.semantic_content_hash
            ),
        )


def test_anchor_generation_fences_pending_late_receipt_and_old_basis() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    publisher.offer_basis(_basis(activation, index=1, cut_at=10.2))
    pending = publisher.snapshot().pending_attempt
    assert pending is not None
    receipt = store.admit(pending.context_draft)
    assert receipt.accepted

    fenced = publisher.advance_anchor_generation(1)
    assert fenced.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED
    snapshot = publisher.snapshot()
    assert snapshot.active_anchor_request_generation == 1
    assert snapshot.pending_attempt is None
    assert snapshot.latest_coalesced_basis is None
    assert snapshot.committed_reducer_state.revision == 0

    late = publisher.resolve_admission(pending.attempt_id, receipt)
    assert late.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_LATE
    assert late.reason == "receipt_belongs_to_fenced_anchor_generation"
    assert publisher.committed_reducer_state.revision == 0

    old = publisher.offer_basis(_basis(activation, index=2, cut_at=10.3))
    assert old.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED
    assert old.reason == "basis_anchor_generation_is_fenced"

    clock.value = 10.41
    current = publisher.offer_basis(
        _basis(
            activation,
            index=3,
            cut_at=10.4,
            anchor_request_generation=1,
        )
    )
    assert current.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY


def test_station_allocation_port_is_lazy_and_workerless_when_disabled() -> None:
    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(),
    )
    supplier_calls = 0

    def forbidden_supplier():
        nonlocal supplier_calls
        supplier_calls += 1
        raise AssertionError("disabled port must not read a basis")

    assert not port.start().worker_alive
    cycle = port.offer_basis(forbidden_supplier)
    assert cycle.status == "disabled"
    assert supplier_calls == 0
    assert port.close()


def test_station_allocation_port_commits_on_bounded_worker() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="allocation-publication-test",
            enabled=True,
        ),
        allocation_runtime=publisher,
        publication_barrier=_active_station_barrier(),
        admit_context=store.admit,
        resolve_context=store.record_by_id,
    )
    assert port.start().worker_alive
    offered = port.offer_basis(
        lambda: _basis(activation, index=1, cut_at=10.2)
    )
    assert offered.status == "basis_offered"
    assert offered.worker_notified
    assert offered.outcome is not None
    assert offered.outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    )
    assert port.wait_for_outcomes(2, timeout_sec=2.0)
    assert publisher.committed_reducer_state.revision == 1
    assert port.snapshot().latest_outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
    )
    assert port.close()


def test_station_allocation_port_coalesces_while_admission_is_blocked() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    entered = threading.Event()
    release = threading.Event()
    admitted_drafts = []

    def slow_first_admission(draft):
        admitted_drafts.append(draft)
        if len(admitted_drafts) == 1:
            entered.set()
            assert release.wait(timeout=2.0)
        return store.admit(draft)

    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="allocation-publication-coalescing-test",
            enabled=True,
        ),
        allocation_runtime=publisher,
        publication_barrier=_active_station_barrier(),
        admit_context=slow_first_admission,
        resolve_context=store.record_by_id,
    )
    port.start()
    port.offer_basis(lambda: _basis(activation, index=1, cut_at=10.2))
    assert entered.wait(timeout=2.0)
    clock.value = 10.31
    coalesced = port.offer_basis(
        lambda: _basis(activation, index=2, cut_at=10.3)
    )
    assert coalesced.outcome is not None
    assert coalesced.outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED
    )
    assert publisher.committed_reducer_state.revision == 0
    release.set()
    assert port.wait_for_outcomes(5, timeout_sec=2.0)
    assert publisher.committed_reducer_state.revision == 2
    assert len(admitted_drafts) == 2
    assert port.close()


def test_station_allocation_port_retries_exact_attempt_and_fences_close() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    submitted_drafts = []

    def lose_first_outcome(draft):
        submitted_drafts.append(draft)
        if len(submitted_drafts) == 1:
            raise RuntimeError("synthetic admission transport failure")
        return store.admit(draft)

    barrier = _active_station_barrier()
    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="allocation-publication-retry-test",
            enabled=True,
        ),
        allocation_runtime=publisher,
        publication_barrier=barrier,
        admit_context=lose_first_outcome,
        resolve_context=store.record_by_id,
    )
    port.start()
    port.offer_basis(lambda: _basis(activation, index=1, cut_at=10.2))
    assert port.wait_for_outcomes(2, timeout_sec=2.0)
    pending = publisher.snapshot().pending_attempt
    assert pending is not None
    assert port.snapshot().latest_outcome.outcome_kind == (
        ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN
    )
    assert port.request_retry().worker_notified
    assert port.wait_for_outcomes(3, timeout_sec=2.0)
    assert submitted_drafts == [pending.context_draft, pending.context_draft]
    assert publisher.committed_reducer_state.revision == 1

    barrier.disable(reason="composition_session_stopped")
    closed = port.offer_basis(
        lambda: (_ for _ in ()).throw(
            AssertionError("closed barrier must prevent supplier reads")
        )
    )
    assert closed.status == "closed"
    assert publisher.runtime_status == "invalidated"
    assert port.close()


def test_station_allocation_port_fences_inflight_receipt_before_join() -> None:
    clock = Clock(10.21)
    spec = _spec()
    activation = _activation()
    store, _, _ = _store(clock)
    publisher = _publisher(clock, spec, activation)
    entered = threading.Event()
    release = threading.Event()

    def delayed_admission(draft):
        entered.set()
        assert release.wait(timeout=2.0)
        return store.admit(draft)

    barrier = _active_station_barrier()
    port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="allocation-publication-close-test",
            enabled=True,
        ),
        allocation_runtime=publisher,
        publication_barrier=barrier,
        admit_context=delayed_admission,
        resolve_context=store.record_by_id,
    )
    port.start()
    port.offer_basis(lambda: _basis(activation, index=1, cut_at=10.2))
    assert entered.wait(timeout=2.0)
    barrier.disable(reason="composition_session_stopped")
    assert not port.close(timeout_sec=0.01)
    assert publisher.runtime_status == "invalidated"
    assert publisher.committed_reducer_state.revision == 0
    release.set()
    assert port.close(timeout_sec=2.0)
    assert publisher.committed_reducer_state.revision == 0
    assert any(
        outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_LATE
        for outcome in port.drain_outcomes()
    )
