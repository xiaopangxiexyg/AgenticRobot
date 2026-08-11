"""Protocol-neutral events for allocation publication lifecycle facts."""

from __future__ import annotations

from control_composition.events import CompositionDomainEvent

from .publication_contracts import AllocationPublicationOutcome


ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND = "allocation_publication_outcome"


def allocation_publication_outcome_semantic_payload(
    outcome: AllocationPublicationOutcome,
) -> dict[str, object]:
    """Return compact actual-data lineage without copying activity payloads."""

    if not isinstance(outcome, AllocationPublicationOutcome):
        raise TypeError("outcome must be AllocationPublicationOutcome.")
    return {
        "outcome_kind": outcome.outcome_kind,
        "allocation_scope_id": outcome.allocation_scope_id,
        "runtime_status": outcome.runtime_status,
        "committed_reducer_state_hash": outcome.committed_reducer_state_hash,
        "reason": outcome.reason,
        "decision_id": outcome.decision_id,
        "decision_hash": outcome.decision_hash,
        "attempt_id": outcome.attempt_id,
        "attempt_hash": outcome.attempt_hash,
        "publication_sequence": outcome.publication_sequence,
        "receipt_context_id": outcome.receipt_context_id,
        "receipt_context_hash": outcome.receipt_context_hash,
        "rejection_code": outcome.rejection_code,
        "semantic_content_hash": outcome.semantic_content_hash,
    }


def allocation_publication_outcome_domain_event(
    outcome: AllocationPublicationOutcome,
) -> CompositionDomainEvent:
    """Wrap one immutable outcome without changing its occurrence time."""

    if not isinstance(outcome, AllocationPublicationOutcome):
        raise TypeError("outcome must be AllocationPublicationOutcome.")
    return CompositionDomainEvent(
        event_kind=ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND,
        subject=outcome,
        occurred_at=outcome.occurred_at,
    )


__all__ = [
    "ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND",
    "allocation_publication_outcome_domain_event",
    "allocation_publication_outcome_semantic_payload",
]
