"""Pure ControlTrace projections for Control Station application facts.

This module is an application-layer bridge: ``dataset_contracts`` remains
independent of Control Station, while Control Station may project its stable
contracts into the shared trace envelope.  Projection never admits a fact,
owns an episode, or changes authority.
"""

from __future__ import annotations

from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact

from .contracts import ControlStationRunManifest
from .composition_session import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionTransition,
)
from .publication_barrier import StationPublicationBarrierSnapshot
from .state_machine import (
    ControlStationOperationTransition,
    ControlStationState,
    ControlStationTransition,
)


class StationControlTraceBridge:
    """Project stable Station facts into an injected trace admission seam.

    The bridge owns no queue, journal, episode, or durability policy.  A sink
    rejection remains visible through ``admission_failures`` and therefore
    leaves a required-event/profile gap at seal time; it never grants or
    revokes control authority.
    """

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime_session_id: str,
        embodiment_instance_id: str | None,
        trace_sink: Callable[[ControlTraceFact], object],
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        if not isinstance(manifest, ControlStationRunManifest):
            raise TypeError("manifest must be a ControlStationRunManifest.")
        if not callable(trace_sink) or not callable(clock):
            raise TypeError("trace_sink and clock must be callable.")
        self.manifest = manifest
        self.runtime_session_id = str(runtime_session_id).strip()
        self.embodiment_instance_id = (
            None
            if embodiment_instance_id is None
            else str(embodiment_instance_id).strip()
        )
        self.trace_sink = trace_sink
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if (
            not self.runtime_session_id
            or not self.clock_domain_id
            or self.embodiment_instance_id == ""
        ):
            raise ValueError("Station trace bridge identity/clock is incomplete.")
        self._accepted_fact_ids: list[str] = []
        self._admission_failures: list[tuple[str, str]] = []

    @property
    def accepted_fact_ids(self) -> tuple[str, ...]:
        return tuple(self._accepted_fact_ids)

    @property
    def admission_failures(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._admission_failures)

    def _admit(self, fact: ControlTraceFact) -> ControlTraceFact:
        receipt = self.trace_sink(fact)
        accepted = True if receipt is None else bool(getattr(receipt, "accepted", False))
        if accepted:
            self._accepted_fact_ids.append(fact.fact_id)
        else:
            reason = str(
                getattr(receipt, "rejection_code", "")
                or getattr(receipt, "message", "")
                or "trace_sink_rejected"
            )
            self._admission_failures.append((fact.fact_id, reason))
        return fact

    def record_manifest(self) -> ControlTraceFact:
        return self._admit(
            project_station_run_manifest(
                self.manifest,
                runtime_session_id=self.runtime_session_id,
                occurred_at=float(self.clock()),
                clock_domain_id=self.clock_domain_id,
            )
        )

    def record_intent_transition(
        self,
        transition: ControlStationTransition,
    ) -> ControlTraceFact:
        return self._admit(
            project_station_intent_transition(
                transition,
                embodiment_instance_id=self.embodiment_instance_id,
            )
        )

    def record_operation_transition(
        self,
        transition: ControlStationOperationTransition,
    ) -> ControlTraceFact:
        return self._admit(
            project_station_operation_transition(
                transition,
                embodiment_instance_id=self.embodiment_instance_id,
            )
        )

    def record_publication_barrier_transition(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> ControlTraceFact:
        return self._admit(
            project_station_publication_barrier_transition(
                snapshot,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.manifest.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
                causal_station_fact_id=causal_station_fact_id,
            )
        )

    def record_composition_session_spec(
        self,
        spec: CompositionSessionOperationSpec,
    ) -> ControlTraceFact:
        return self._admit(
            project_composition_session_spec(
                spec,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.manifest.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                occurred_at=float(self.clock()),
                clock_domain_id=self.clock_domain_id,
            )
        )

    def record_composition_session_transition(
        self,
        transition: CompositionSessionTransition,
        *,
        occurred_at: float,
        clock_domain_id: str,
    ) -> ControlTraceFact:
        return self._admit(
            project_composition_session_transition(
                transition,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.manifest.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            )
        )

    def record_composition_session_rearm_spec(
        self,
        spec: CompositionSessionRearmSpec,
    ) -> ControlTraceFact:
        return self._admit(
            project_composition_session_rearm_spec(
                spec,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.manifest.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                occurred_at=float(self.clock()),
                clock_domain_id=self.clock_domain_id,
            )
        )

    def record_composition_session_rearm_receipt(
        self,
        receipt: CompositionSessionRearmReceipt,
    ) -> ControlTraceFact:
        return self._admit(
            project_composition_session_rearm_receipt(
                receipt,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.manifest.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
            )
        )


def _lease_payload(state: ControlStationState) -> dict[str, object] | None:
    lease = state.operator_control_lease
    if lease is None:
        return None
    return {
        "lease_id": lease.lease_id,
        "client_id": lease.client_id,
        "lease_epoch": lease.lease_epoch,
        "acquired_intent_id": lease.acquired_intent_id,
        "acquired_at": lease.acquired_at,
        "expires_at": lease.expires_at,
        "semantic_content_hash": lease.semantic_content_hash,
    }


def _state_payload(state: ControlStationState) -> dict[str, object]:
    return {
        "runtime_session_id": state.runtime_session_id,
        "manifest_hash": state.manifest_hash,
        "control_domain_id": state.control_domain_id,
        "lifecycle_state": state.lifecycle_state,
        "state_revision": state.state_revision,
        "source_state_by_slot": dict(state.source_state_by_slot),
        "target_state": state.target_state,
        "pending_operation_hash_by_subject": {
            subject_id: operation.semantic_content_hash
            for subject_id, operation in state.pending_operation_by_subject.items()
        },
        "last_client_sequence_by_id": dict(state.last_client_sequence_by_id),
        "operator_control_lease": _lease_payload(state),
        "last_intent_id": state.last_intent_id,
        "semantic_content_hash": state.semantic_content_hash,
    }


def _source_slot_payload(slot: object) -> dict[str, object]:
    return {
        "source_slot_id": slot.source_slot_id,
        "source_kind": slot.source_kind,
        "producer_id": slot.producer_id,
        "source_action_contract_hash": slot.source_action_contract_hash,
        "producer_epoch_policy_hash": slot.producer_epoch_policy_hash,
        "supported_uses": slot.supported_uses,
        "expected_emission_seam": slot.expected_emission_seam,
        "required_binding_refs": slot.required_binding_refs,
        "trace_requirement_refs": slot.trace_requirement_refs,
        "required_for_run": slot.required_for_run,
        "explicit_exclusions": slot.explicit_exclusions,
        "semantic_content_hash": slot.semantic_content_hash,
    }


def project_station_run_manifest(
    manifest: ControlStationRunManifest,
    *,
    runtime_session_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(manifest, ControlStationRunManifest):
        raise TypeError("manifest must be a ControlStationRunManifest.")
    binding = manifest.execution_binding
    return ControlTraceFact.create(
        fact_id=f"station-run-manifest:{manifest.semantic_content_hash}",
        event_type="station_run_manifest",
        runtime_session_id=runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        embodiment_instance_id=binding.embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "manifest_id": manifest.manifest_id,
            "manifest_version": manifest.manifest_version,
            "execution_binding": {
                "binding_id": binding.binding_id,
                "target_kind": binding.target_kind,
                "control_domain_id": binding.control_domain_id,
                "embodiment_instance_id": binding.embodiment_instance_id,
                "target_runtime_action_contract_hash": (
                    binding.target_runtime_action_contract_hash
                ),
                "execution_capability_profile_hash": (
                    binding.execution_capability_profile_hash
                ),
                "component_catalog_hash": binding.component_catalog_hash,
                "selection_group_ids": binding.selection_group_ids,
                "owner_registry_id": binding.owner_registry_id,
                "authority_epoch_source_id": binding.authority_epoch_source_id,
                "execution_epoch_source_id": binding.execution_epoch_source_id,
                "trace_requirement_profile_id": (
                    binding.trace_requirement_profile_id
                ),
                "backend_inventory_hash": binding.backend_inventory_hash,
                "explicit_exclusions": binding.explicit_exclusions,
                "semantic_content_hash": binding.semantic_content_hash,
            },
            "source_slots": tuple(
                _source_slot_payload(slot) for slot in manifest.source_slots
            ),
            "authority_policy_hash": manifest.authority_policy_hash,
            "decision_schedule_hash": manifest.decision_schedule_hash,
            "trace_requirement_profile_id": manifest.trace_requirement_profile_id,
            "capture_policy_id": manifest.capture_policy_id,
            "allowed_intent_kinds": manifest.allowed_intent_kinds,
            "operator_control_lease_sec": manifest.operator_control_lease_sec,
            "explicit_exclusions": manifest.explicit_exclusions,
            "semantic_content_hash": manifest.semantic_content_hash,
        },
    )


def project_composition_session_spec(
    spec: CompositionSessionOperationSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(spec, CompositionSessionOperationSpec):
        raise TypeError("spec must be a CompositionSessionOperationSpec.")
    return ControlTraceFact.create(
        fact_id=spec.operation_spec_id,
        event_type="composition_session_spec",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "operation_spec_id": spec.operation_spec_id,
            "session_key": spec.session_key,
            "composer_source_slot_id": spec.composer_source_slot_id,
            "contributor_source_slot_ids": spec.contributor_source_slot_ids,
            "composition_spec_hash": spec.composition_spec_hash,
            "authority_gate_spec_hash": spec.authority_gate_spec_hash,
            "execution_binding_hash": spec.execution_binding_hash,
            "capture_boundary_id": spec.capture_boundary_id,
            "pause_behavior": spec.pause_behavior,
            "end_behavior": spec.end_behavior,
            "resume_behavior": spec.resume_behavior,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_composition_session_transition(
    transition: CompositionSessionTransition,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(transition, CompositionSessionTransition):
        raise TypeError(
            "transition must be a CompositionSessionTransition."
        )
    return ControlTraceFact.create(
        fact_id=transition.transition_id,
        event_type="composition_session_transition",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "transition_id": transition.transition_id,
            "operation_spec_id": transition.operation_spec_id,
            "operation_spec_hash": transition.operation_spec_hash,
            "session_epoch": transition.session_epoch,
            "transition": transition.transition,
            "reason": transition.reason,
            "source_fact_id": transition.source_fact_id,
            "source_fact_hash": transition.source_fact_hash,
            "state_before_hash": transition.state_before_hash,
            "state_after_hash": transition.state_after_hash,
            "semantic_content_hash": transition.semantic_content_hash,
        },
    )


def project_composition_session_rearm_spec(
    spec: CompositionSessionRearmSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(spec, CompositionSessionRearmSpec):
        raise TypeError("spec must be a CompositionSessionRearmSpec.")
    return ControlTraceFact.create(
        fact_id=spec.rearm_spec_id,
        event_type="composition_session_rearm_spec",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "rearm_spec_id": spec.rearm_spec_id,
            "rearm_key": spec.rearm_key,
            "logical_operation_spec_id": spec.logical_operation_spec_id,
            "logical_operation_spec_hash": spec.logical_operation_spec_hash,
            "prior_session_epoch": spec.prior_session_epoch,
            "prior_state_hash": spec.prior_state_hash,
            "execution_binding_hash": spec.execution_binding_hash,
            "readiness_observer_binding_id": (
                spec.readiness_observer_binding_id
            ),
            "readiness_timeout_sec": spec.readiness_timeout_sec,
            "required_readiness_checks": spec.required_readiness_checks,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_composition_session_rearm_receipt(
    receipt: CompositionSessionRearmReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
) -> ControlTraceFact:
    if not isinstance(receipt, CompositionSessionRearmReceipt):
        raise TypeError("receipt must be a CompositionSessionRearmReceipt.")
    return ControlTraceFact.create(
        fact_id=receipt.receipt_id,
        event_type="composition_session_rearm_receipt",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=receipt.completed_at,
        clock_domain_id=receipt.clock_domain_id,
        payload={
            field_name: getattr(receipt, field_name)
            for field_name in (
                "receipt_id",
                "rearm_spec_id",
                "rearm_spec_hash",
                "logical_operation_spec_id",
                "logical_operation_spec_hash",
                "prior_session_epoch",
                "next_session_epoch",
                "prior_composer_producer_epoch",
                "composer_producer_epoch",
                "prior_authority_epoch",
                "authority_epoch",
                "prior_execution_epoch",
                "execution_epoch",
                "cancellation_receipt_id",
                "measurement_id",
                "measurement_hash",
                "producer_activation_id",
                "authority_event_ids",
                "ready_gate_event_id",
                "activation_input_barrier_time",
                "completed_at",
                "clock_domain_id",
                "satisfied_readiness_checks",
                "status",
                "semantic_content_hash",
            )
        },
    )


def project_station_intent_transition(
    transition: ControlStationTransition,
    *,
    embodiment_instance_id: str | None,
) -> ControlTraceFact:
    if not isinstance(transition, ControlStationTransition):
        raise TypeError("transition must be a ControlStationTransition.")
    intent = transition.intent
    receipt = transition.receipt
    return ControlTraceFact.create(
        fact_id=intent.intent_id,
        event_type="station_intent_transition",
        runtime_session_id=intent.runtime_session_id,
        control_domain_id=intent.control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=intent.available_at,
        clock_domain_id=intent.available_clock_domain_id,
        payload={
            "intent_id": intent.intent_id,
            "intent_kind": intent.intent_kind,
            "client_id": intent.client_id,
            "client_sequence": intent.client_sequence,
            "subject_id": intent.subject_id,
            "resource_binding_id": intent.resource_binding_id,
            "operation_spec_id": intent.operation_spec_id,
            "produced_at": intent.produced_at,
            "produced_clock_domain_id": intent.produced_clock_domain_id,
            "available_at": intent.available_at,
            "available_clock_domain_id": intent.available_clock_domain_id,
            "admission_sequence": intent.admission_sequence,
            "expected_state_revision": intent.expected_state_revision,
            "manifest_hash": intent.manifest_hash,
            "reason": intent.reason,
            "accepted": receipt.accepted,
            "status": receipt.status,
            "state_before": _state_payload(transition.state_before),
            "state_after": _state_payload(transition.state_after),
            "intent_semantic_hash": intent.semantic_content_hash,
            "receipt_semantic_hash": receipt.semantic_content_hash,
            "transition_semantic_hash": transition.semantic_content_hash,
        },
    )


def project_station_operation_transition(
    transition: ControlStationOperationTransition,
    *,
    embodiment_instance_id: str | None,
) -> ControlTraceFact:
    if not isinstance(transition, ControlStationOperationTransition):
        raise TypeError(
            "transition must be a ControlStationOperationTransition."
        )
    result = transition.result
    return ControlTraceFact.create(
        fact_id=result.operation_result_id,
        event_type="station_operation_transition",
        runtime_session_id=result.runtime_session_id,
        control_domain_id=result.control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=result.available_at,
        clock_domain_id=result.available_clock_domain_id,
        payload={
            "operation_result_id": result.operation_result_id,
            "source_intent_id": result.source_intent_id,
            "operation_kind": result.operation_kind,
            "subject_id": result.subject_id,
            "succeeded": result.succeeded,
            "completed_at": result.completed_at,
            "completed_clock_domain_id": result.completed_clock_domain_id,
            "available_at": result.available_at,
            "available_clock_domain_id": result.available_clock_domain_id,
            "admission_sequence": result.admission_sequence,
            "result_status": result.status,
            "evidence_refs": result.evidence_refs,
            "transition_accepted": transition.accepted,
            "transition_status": transition.status,
            "state_before": _state_payload(transition.state_before),
            "state_after": _state_payload(transition.state_after),
            "result_semantic_hash": result.semantic_content_hash,
            "transition_semantic_hash": transition.semantic_content_hash,
        },
    )


def project_station_publication_barrier_transition(
    snapshot: StationPublicationBarrierSnapshot,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
    causal_station_fact_id: str,
) -> ControlTraceFact:
    if not isinstance(snapshot, StationPublicationBarrierSnapshot):
        raise TypeError("snapshot must be a StationPublicationBarrierSnapshot.")
    causal_id = str(causal_station_fact_id).strip()
    if not causal_id:
        raise ValueError("causal_station_fact_id must be non-empty.")
    fact_identity_hash = canonical_content_hash(
        {
            "snapshot_semantic_hash": snapshot.semantic_content_hash,
            "causal_station_fact_id": causal_id,
        },
        domain="dexteleop.control_station.publication_barrier_trace_identity.v1",
    )
    return ControlTraceFact.create(
        fact_id=f"station-publication-barrier:{fact_identity_hash}",
        event_type="station_publication_barrier_transition",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "source_slot_id": snapshot.source_slot_id,
            "generation": snapshot.generation,
            "enabled": snapshot.enabled,
            "authority_operation_result_id": (
                snapshot.authority_operation_result_id
            ),
            "authority_evidence_refs": snapshot.authority_evidence_refs,
            "transition_reason": snapshot.transition_reason,
            "causal_station_fact_id": causal_id,
            "snapshot_semantic_hash": snapshot.semantic_content_hash,
        },
    )


__all__ = [
    "StationControlTraceBridge",
    "project_composition_session_rearm_receipt",
    "project_composition_session_rearm_spec",
    "project_composition_session_spec",
    "project_composition_session_transition",
    "project_station_intent_transition",
    "project_station_operation_transition",
    "project_station_publication_barrier_transition",
    "project_station_run_manifest",
]
