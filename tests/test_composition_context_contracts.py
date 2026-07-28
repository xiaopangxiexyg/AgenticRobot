from __future__ import annotations

from dataclasses import replace

import numpy as np

from control_composition.composition import (
    CompositionContextDraft,
    CompositionContextStore,
    build_composition_context_snapshot,
    project_composition_contexts,
)
from control_composition.retention import TimeWindowRetentionSpec
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from dataset_interface.composition_control_trace import (
    project_composition_context,
    project_composition_context_admission,
    project_composition_context_snapshot,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def runtime():
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0", coordinator=coordinator
    )
    epochs.activate(
        producer_id="slow-policy",
        producer_epoch=1,
        produced_at=1.0,
        produced_clock_domain_id="slow-policy.monotonic",
        reason="startup",
    )
    store = CompositionContextStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="test-context-window-v1",
            history_window_sec=0.0,
            max_future_lead_sec=2.0,
            max_effective_duration_sec=2.0,
            max_late_arrival_sec=0.0,
        ),
    )
    return clock, coordinator, epochs, store


def draft(
    *,
    producer_id: str = "slow-policy",
    sequence: int = 0,
    epoch: int = 1,
    effective_from: float = 10.0,
    effective_until: float = 11.0,
    payload=None,
    audit=None,
) -> CompositionContextDraft:
    return CompositionContextDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer_id,
        producer_epoch=epoch,
        sequence=sequence,
        producer_node_id="slow-vision-policy",
        context_channel="motion-plan",
        context_kind="semantic_subgoal",
        semantic_payload=payload or {"target_xyz": np.asarray([0.3, 0.0, 0.4])},
        produced_at=9.9,
        produced_clock_domain_id="slow-policy.monotonic",
        effective_from=effective_from,
        effective_until=effective_until,
        effective_clock_domain_id="control.monotonic",
        payload_contract_hash="context-contract-hash",
        observation_cut_id="observation-cut-4",
        inference_id="inference-4",
        policy_output_id="policy-output-4",
        audit_metadata=audit or {},
    )


def test_receiver_admission_freezes_payload_and_owns_available_time() -> None:
    clock, coordinator, _, store = runtime()
    mutable = np.asarray([0.3, 0.0, 0.4], dtype=np.float32)
    receipt = store.admit(draft(payload={"target_xyz": mutable}))
    assert receipt.accepted
    assert receipt.available_at == clock.value
    assert receipt.control_admission_sequence == coordinator.committed_high_watermark
    assert receipt.context is not None
    mutable[:] = 9.0
    np.testing.assert_array_equal(
        receipt.context.semantic_payload["target_xyz"],
        np.asarray([0.3, 0.0, 0.4], dtype=np.float32),
    )
    assert not receipt.context.semantic_payload["target_xyz"].flags.writeable


def test_audit_metadata_is_not_semantic_and_opaque_latent_requires_artifact() -> None:
    _, _, _, first_store = runtime()
    _, _, _, second_store = runtime()
    first = first_store.admit(draft(audit={"worker": "one"}))
    second = second_store.admit(draft(audit={"worker": "two"}))
    assert first.semantic_content_hash == second.semantic_content_hash

    rejected = first_store.admit(
        replace(
            draft(sequence=1),
            context_kind="artifact_bound_latent",
            producer_artifact_hash=None,
        )
    )
    assert not rejected.accepted
    assert rejected.rejection_code == "artifact_hash_required"
    accepted = first_store.admit(
        replace(
            draft(sequence=1),
            context_kind="artifact_bound_latent",
            producer_artifact_hash="artifact-sha256",
        )
    )
    assert accepted.accepted


def test_snapshot_is_target_time_aware_and_restart_invalidates_old_context() -> None:
    clock, coordinator, epochs, store = runtime()
    old = store.admit(
        draft(sequence=0, effective_from=10.0, effective_until=12.0)
    ).context
    new = store.admit(
        draft(sequence=1, effective_from=10.5, effective_until=12.0)
    ).context
    assert old is not None and new is not None
    clock.value = 10.75
    cut = coordinator.capture_cut(cutoff_at=10.75).cut
    snapshot = build_composition_context_snapshot(
        project_composition_contexts(store, cut),
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.75,
        required_channels=("motion-plan",),
    )
    assert snapshot.selected_context_id_by_channel["motion-plan"] == new.context_id

    epochs.activate(
        producer_id="slow-policy",
        producer_epoch=2,
        produced_at=10.8,
        produced_clock_domain_id="slow-policy.monotonic",
        reason="restart",
    )
    clock.value = 10.9
    restarted_cut = coordinator.capture_cut(cutoff_at=10.9).cut
    missing = build_composition_context_snapshot(
        project_composition_contexts(store, restarted_cut),
        producer_epochs=epochs.snapshot(restarted_cut),
        target_effective_at=10.9,
        required_channels=("motion-plan",),
    )
    assert missing.missing_channels == ("motion-plan",)
    assert not missing.selected_context_id_by_channel


def test_context_arriving_after_cut_only_changes_next_snapshot() -> None:
    clock, coordinator, epochs, store = runtime()
    first = store.admit(draft(sequence=0)).context
    assert first is not None
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    frozen_projection = project_composition_contexts(store, cut)
    frozen_epochs = epochs.snapshot(cut)

    clock.value = 10.01
    second = store.admit(
        draft(sequence=1, effective_from=10.0, effective_until=11.0)
    ).context
    assert second is not None
    before = build_composition_context_snapshot(
        frozen_projection,
        producer_epochs=frozen_epochs,
        target_effective_at=10.5,
        required_channels=("motion-plan",),
    )
    assert before.selected_context_id_by_channel["motion-plan"] == first.context_id

    next_cut = coordinator.capture_cut(cutoff_at=10.02).cut
    after = build_composition_context_snapshot(
        project_composition_contexts(store, next_cut),
        producer_epochs=epochs.snapshot(next_cut),
        target_effective_at=10.5,
        required_channels=("motion-plan",),
    )
    assert after.selected_context_id_by_channel["motion-plan"] == second.context_id


def test_context_overlap_ranks_by_receiver_admission_across_producers() -> None:
    _, coordinator, epochs, store = runtime()
    epochs.activate(
        producer_id="fast-policy",
        producer_epoch=1,
        produced_at=1.0,
        produced_clock_domain_id="fast-policy.monotonic",
        reason="startup",
    )
    earlier = store.admit(
        draft(
            producer_id="slow-policy",
            sequence=100,
            effective_from=10.0,
            effective_until=11.0,
        )
    ).context
    later = store.admit(
        replace(
            draft(
                producer_id="fast-policy",
                sequence=1,
                effective_from=10.0,
                effective_until=11.0,
            ),
            producer_node_id="fast-vision-policy",
            produced_clock_domain_id="fast-policy.monotonic",
        )
    ).context
    assert earlier is not None and later is not None
    assert earlier.sequence > later.sequence
    assert earlier.control_admission_sequence < later.control_admission_sequence

    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_context_snapshot(
        project_composition_contexts(store, cut),
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.5,
        required_channels=("motion-plan",),
    )
    assert snapshot.selected_context_id_by_channel["motion-plan"] == later.context_id


def test_context_sequence_rollback_and_stale_interval_fail_closed() -> None:
    clock, coordinator, epochs, store = runtime()
    accepted = store.admit(
        draft(sequence=2, effective_from=10.0, effective_until=10.2)
    )
    rollback = store.admit(
        draft(sequence=1, effective_from=10.0, effective_until=11.0)
    )
    assert accepted.accepted
    assert not rollback.accepted
    assert rollback.rejection_code == "sequence_rollback"

    clock.value = 10.3
    cut = coordinator.capture_cut(cutoff_at=10.3).cut
    stale = build_composition_context_snapshot(
        project_composition_contexts(store, cut),
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.3,
        required_channels=("motion-plan",),
    )
    assert stale.missing_channels == ("motion-plan",)


def test_context_trace_records_are_explicit_and_hash_preserving() -> None:
    _, coordinator, epochs, store = runtime()
    receipt = store.admit(draft())
    assert receipt.context is not None
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_context_snapshot(
        project_composition_contexts(store, cut),
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.5,
        required_channels=("motion-plan",),
    )
    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        clock_domain_id="control.monotonic",
    )
    admission_fact = project_composition_context_admission(
        trace_context,
        receipt,
        occurred_at=10.0,
    )
    context_fact = project_composition_context(
        trace_context,
        receipt.context,
    )
    snapshot_fact = project_composition_context_snapshot(
        trace_context,
        snapshot,
    )
    assert admission_fact.event_type == "composition_context_admission"
    assert context_fact.payload["context_semantic_content_hash"] == (
        receipt.context.semantic_content_hash
    )
    assert snapshot_fact.payload["selected_context_id_by_channel"] == {
        "motion-plan": receipt.context.context_id
    }


def test_atomic_snapshot_advances_time_window_and_prunes_full_payload() -> None:
    clock, coordinator, epochs, store = runtime()
    first = store.admit(
        draft(sequence=0, effective_from=10.0, effective_until=10.2)
    )
    assert first.accepted
    clock.value = 10.15
    cut = coordinator.capture_cut(cutoff_at=10.15).cut
    selected = store.capture_snapshot(
        cut,
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=10.15,
        required_channels=("motion-plan",),
    )
    assert selected.selected_context_id_by_channel["motion-plan"] == first.context_id
    assert len(store) == 1

    clock.value = 10.25
    later_cut = coordinator.capture_cut(cutoff_at=10.25).cut
    missing = store.capture_snapshot(
        later_cut,
        producer_epochs=epochs.snapshot(later_cut),
        target_effective_at=10.25,
        required_channels=("motion-plan",),
    )
    assert missing.missing_channels == ("motion-plan",)
    health = store.retention_snapshot()
    assert health.closed_target_through == 10.25
    assert health.retained_record_count == 0
    assert health.pruned_record_count == 1
    assert health.producer_epoch_tombstone_count == 1
    np.testing.assert_allclose(
        selected.strategy_contexts[0].semantic_payload["target_xyz"][0],
        0.3,
    )

    duplicate = store.admit(
        draft(sequence=0, effective_from=10.25, effective_until=10.5)
    )
    assert not duplicate.accepted
    assert duplicate.rejection_code == "duplicate"


def test_retention_rejects_unbounded_future_and_duration() -> None:
    _, _, _, store = runtime()
    future = store.admit(
        draft(sequence=0, effective_from=12.1, effective_until=12.2)
    )
    assert not future.accepted
    assert future.rejection_code == "retention_window_violation"

    too_long = store.admit(
        draft(sequence=0, effective_from=10.0, effective_until=12.1)
    )
    assert not too_long.accepted
    assert too_long.rejection_code == "retention_window_violation"


def test_missing_and_fault_contexts_preserve_lineage_but_not_strategy_payload() -> None:
    clock, coordinator, epochs, store = runtime()
    missing_receipt = store.admit(
        replace(
            draft(sequence=0),
            availability="missing",
            semantic_payload={"reason": "camera_frame_absent"},
        )
    )
    assert missing_receipt.accepted
    clock.value = 10.1
    cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    missing_snapshot = store.capture_snapshot(
        cut,
        producer_epochs=epochs.snapshot(cut),
        target_effective_at=clock.value,
        required_channels=("motion-plan",),
    )
    assert missing_snapshot.missing_channels == ("motion-plan",)
    assert missing_snapshot.unavailable_channels == ("motion-plan",)
    assert missing_snapshot.selected_context_id_by_channel["motion-plan"] == (
        missing_receipt.context_id
    )
    assert missing_snapshot.strategy_contexts == ()

    clock.value = 10.2
    fault_receipt = store.admit(
        replace(
            draft(sequence=1, effective_from=10.2, effective_until=10.8),
            availability="fault",
            fault_code="producer_timeout",
            semantic_payload={"last_valid_context_id": missing_receipt.context_id},
        )
    )
    assert fault_receipt.accepted
    fault_cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    fault_snapshot = store.capture_snapshot(
        fault_cut,
        producer_epochs=epochs.snapshot(fault_cut),
        target_effective_at=clock.value,
        required_channels=("motion-plan",),
    )
    assert fault_snapshot.unavailable_channels == ("motion-plan",)
    assert fault_snapshot.strategy_contexts == ()

    invalid_fault = store.admit(
        replace(
            draft(sequence=2, effective_from=10.2, effective_until=10.8),
            availability="fault",
            fault_code=None,
        )
    )
    assert not invalid_fault.accepted
    assert invalid_fault.rejection_code == "invalid_availability"
