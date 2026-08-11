"""ControlTrace projection for residual-anchor lifecycle facts.

This adapter records protocol identity and causal timing only.  The concrete
absolute/relative action payloads remain owned by composition input facts.
"""

from __future__ import annotations

from control_composition.events import CompositionDomainEvent
from control_composition.residual_anchor import (
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
)
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorLifecycleRecord,
    ResidualAnchorLifecycleSnapshot,
    ResidualAnchorLifecycleSpec,
)
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext


RESIDUAL_ANCHOR_LIFECYCLE_EVENT_KINDS = frozenset(
    {
        "residual_anchor_lifecycle_spec",
        "residual_anchor_lifecycle_admission",
        "residual_anchor_lifecycle",
        "residual_anchor_lifecycle_snapshot",
    }
)


def _request_payload(request: ResidualAnchorRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "request_hash": request.semantic_content_hash,
        "request_generation": request.request_generation,
        "source_role": request.source_role,
        "component_ids": request.component_ids,
        "composition_activation_id": request.composition_activation_id,
        "composition_activation_hash": request.composition_activation_hash,
        "source_binding_id": request.source_binding_id,
        "source_binding_hash": request.source_binding_hash,
        "absolute_action_schema_hash": request.absolute_action_schema_hash,
        "relative_action_schema_hash": request.relative_action_schema_hash,
        "requested_at": request.requested_at,
        "clock_domain_id": request.clock_domain_id,
        "input_barrier_at": request.input_barrier_at,
        "previous_request_id": request.previous_request_id,
        "previous_request_hash": request.previous_request_hash,
    }


def _receipt_payload(
    receipt: ResidualAnchorReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "receipt_id": receipt.receipt_id,
        "receipt_hash": receipt.semantic_content_hash,
        "request_id": receipt.request_id,
        "request_hash": receipt.request_hash,
        "request_generation": receipt.request_generation,
        "source_action_id": receipt.source_action_id,
        "source_action_hash": receipt.source_action_hash,
        "source_available_at": receipt.source_available_at,
        "captured_at": receipt.captured_at,
        "clock_domain_id": receipt.clock_domain_id,
        "component_binding_by_id": {
            component_id: {
                "source_sample_id": binding.source_sample_id,
                "source_sample_hash": binding.source_sample_hash,
                "absolute_payload_schema_hash": (
                    binding.absolute_payload_schema_hash
                ),
                "anchor_id": binding.anchor_id,
                "anchor_hash": binding.anchor_hash,
                "binding_hash": binding.semantic_content_hash,
            }
            for component_id, binding in sorted(
                receipt.component_binding_by_id.items()
            )
        },
    }


def project_residual_anchor_lifecycle_spec(
    context: AuthoritativeTraceContext,
    spec: ResidualAnchorLifecycleSpec,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(spec, ResidualAnchorLifecycleSpec):
        raise TypeError("spec must be a ResidualAnchorLifecycleSpec.")
    return context.fact(
        fact_id=f"residual-anchor-lifecycle-spec:{spec.semantic_content_hash}",
        event_type="residual_anchor_lifecycle_spec",
        occurred_at=float(occurred_at),
        payload={
            "spec_id": spec.spec_id,
            "source_role": spec.source_role,
            "component_ids": spec.component_ids,
            "absolute_action_schema_hash": spec.absolute_action_schema_hash,
            "relative_action_schema_hash": spec.relative_action_schema_hash,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_residual_anchor_lifecycle_admission(
    context: AuthoritativeTraceContext,
    receipt: ResidualAnchorLifecycleAdmissionReceipt,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(receipt, ResidualAnchorLifecycleAdmissionReceipt):
        raise TypeError(
            "receipt must be a ResidualAnchorLifecycleAdmissionReceipt."
        )
    record_id = receipt.lifecycle_record_id or "rejected-unidentified"
    return context.fact(
        fact_id=f"residual-anchor-lifecycle-admission:{record_id}",
        event_type="residual_anchor_lifecycle_admission",
        occurred_at=float(occurred_at),
        payload={
            "lifecycle_record_id": receipt.lifecycle_record_id,
            "accepted": receipt.accepted,
            "accepted_existing": receipt.accepted_existing,
            "rejection_code": receipt.rejection_code,
            "message": receipt.message,
            "available_at": receipt.available_at,
            "available_clock_domain_id": receipt.available_clock_domain_id,
            "control_admission_sequence": receipt.control_admission_sequence,
            "lifecycle_record_hash": receipt.semantic_content_hash,
        },
    )


def project_residual_anchor_lifecycle_record(
    context: AuthoritativeTraceContext,
    record: ResidualAnchorLifecycleRecord,
) -> ControlTraceFact:
    if not isinstance(record, ResidualAnchorLifecycleRecord):
        raise TypeError("record must be a ResidualAnchorLifecycleRecord.")
    return context.fact(
        fact_id=record.lifecycle_record_id,
        event_type="residual_anchor_lifecycle",
        occurred_at=record.available_at,
        payload={
            "lifecycle_record_id": record.lifecycle_record_id,
            "producer_id": record.producer_id,
            "producer_epoch": record.producer_epoch,
            "sequence": record.sequence,
            "lifecycle_scope_id": record.lifecycle_scope_id,
            "lifecycle_spec_hash": record.lifecycle_spec_hash,
            "phase": record.phase,
            "effective_for_target_at": record.effective_for_target_at,
            "observed_at": record.observed_at,
            "observed_clock_domain_id": record.observed_clock_domain_id,
            "available_at": record.available_at,
            "available_clock_domain_id": record.available_clock_domain_id,
            "control_admission_sequence": record.control_admission_sequence,
            "composition_activation_id": record.composition_activation_id,
            "composition_activation_hash": record.composition_activation_hash,
            "anchor_request": _request_payload(record.anchor_request),
            "anchor_receipt": _receipt_payload(record.anchor_receipt),
            "semantic_content_hash": record.semantic_content_hash,
        },
    )


def project_residual_anchor_lifecycle_snapshot(
    context: AuthoritativeTraceContext,
    snapshot: ResidualAnchorLifecycleSnapshot,
) -> ControlTraceFact:
    if not isinstance(snapshot, ResidualAnchorLifecycleSnapshot):
        raise TypeError("snapshot must be a ResidualAnchorLifecycleSnapshot.")
    selected = snapshot.selected_record
    return context.fact(
        fact_id=snapshot.lifecycle_snapshot_id,
        event_type="residual_anchor_lifecycle_snapshot",
        occurred_at=snapshot.cutoff_at,
        payload={
            "lifecycle_snapshot_id": snapshot.lifecycle_snapshot_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "target_effective_at": snapshot.target_effective_at,
            "lifecycle_scope_id": snapshot.lifecycle_scope_id,
            "lifecycle_spec_hash": snapshot.lifecycle_spec_hash,
            "status": snapshot.status,
            "reason": snapshot.reason,
            "selected_record_id": (
                None if selected is None else selected.lifecycle_record_id
            ),
            "selected_record_hash": (
                None if selected is None else selected.semantic_content_hash
            ),
            "semantic_content_hash": snapshot.semantic_content_hash,
        },
    )


def project_residual_anchor_lifecycle_domain_event(
    context: AuthoritativeTraceContext,
    event: CompositionDomainEvent,
) -> ControlTraceFact:
    if not isinstance(event, CompositionDomainEvent):
        raise TypeError("event must be a CompositionDomainEvent.")
    if event.event_kind == "residual_anchor_lifecycle_spec":
        return project_residual_anchor_lifecycle_spec(
            context,
            event.subject,
            occurred_at=event.occurred_at,
        )
    if event.event_kind == "residual_anchor_lifecycle_admission":
        return project_residual_anchor_lifecycle_admission(
            context,
            event.subject,
            occurred_at=event.occurred_at,
        )
    if event.event_kind == "residual_anchor_lifecycle":
        return project_residual_anchor_lifecycle_record(context, event.subject)
    if event.event_kind == "residual_anchor_lifecycle_snapshot":
        return project_residual_anchor_lifecycle_snapshot(context, event.subject)
    raise ValueError("Unsupported residual-anchor lifecycle event kind.")


__all__ = [
    "RESIDUAL_ANCHOR_LIFECYCLE_EVENT_KINDS",
    "project_residual_anchor_lifecycle_admission",
    "project_residual_anchor_lifecycle_domain_event",
    "project_residual_anchor_lifecycle_record",
    "project_residual_anchor_lifecycle_snapshot",
    "project_residual_anchor_lifecycle_spec",
]
