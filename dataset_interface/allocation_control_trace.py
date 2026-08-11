"""ControlTrace projection for receipt-driven allocation publication facts."""

from __future__ import annotations

from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND,
    AllocationPublicationOutcome,
    allocation_publication_outcome_semantic_payload,
)
from control_composition.events import CompositionDomainEvent
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext


def project_allocation_publication_outcome(
    context: AuthoritativeTraceContext,
    outcome: AllocationPublicationOutcome,
) -> ControlTraceFact:
    if not isinstance(context, AuthoritativeTraceContext):
        raise TypeError("context must be an AuthoritativeTraceContext.")
    if not isinstance(outcome, AllocationPublicationOutcome):
        raise TypeError("outcome must be AllocationPublicationOutcome.")
    return context.fact(
        fact_id=outcome.outcome_id,
        event_type=ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND,
        occurred_at=outcome.occurred_at,
        payload=allocation_publication_outcome_semantic_payload(outcome),
    )


def project_allocation_publication_domain_event(
    context: AuthoritativeTraceContext,
    event: CompositionDomainEvent,
) -> ControlTraceFact:
    if not isinstance(event, CompositionDomainEvent):
        raise TypeError("event must be a CompositionDomainEvent.")
    if event.event_kind != ALLOCATION_PUBLICATION_OUTCOME_EVENT_KIND:
        raise ValueError("Unsupported allocation publication event kind.")
    return project_allocation_publication_outcome(context, event.subject)


__all__ = [
    "project_allocation_publication_domain_event",
    "project_allocation_publication_outcome",
]
