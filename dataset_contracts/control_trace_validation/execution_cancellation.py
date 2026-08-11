"""Execution-cancellation request, acknowledgement, and HOLD lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ExecutionCancellationValidationIndex:
    """Indexes used only by the cancellation lineage validator."""

    cancellation_by_id: FactIndex
    acknowledgement_by_cancellation_id: FactIndex
    dispatch_permit_by_evidence_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ExecutionCancellationValidationIndex":
        return cls(
            cancellation_by_id=context.payload_index(
                event_type="execution_cancellation",
                payload_key="cancellation_id",
            ),
            acknowledgement_by_cancellation_id=context.payload_index(
                event_type="execution_cancellation_ack",
                payload_key="cancellation_id",
            ),
            dispatch_permit_by_evidence_id=context.payload_index(
                event_type="dispatch_owner",
                payload_key="evidence_event_id",
            ),
        )


def validate_execution_cancellation_lineage(
    context: ControlTraceValidationContext,
    indexes: ExecutionCancellationValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate cancellation evidence without owning cancellation or HOLD."""

    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "execution_cancellation":
            cancellation_id = str(payload["cancellation_id"])
            for source_event_id in payload["source_event_ids"]:
                if str(source_event_id) not in context.by_id:
                    issues.add(
                        "execution_cancellation_missing_source_event",
                        fact.fact_id,
                        "Execution cancellation source lineage is absent.",
                        str(source_event_id),
                    )
            if payload["effective_scope"] == "control_domain" and int(
                payload["execution_epoch"]
            ) != int(payload["previous_execution_epoch"]) + 1:
                issues.add(
                    "execution_cancellation_epoch_not_advanced",
                    fact.fact_id,
                    "Domain cancellation must advance execution epoch exactly "
                    "once.",
                )
            acknowledgements = (
                indexes.acknowledgement_by_cancellation_id.get(
                    cancellation_id,
                    (),
                )
            )
            for level in payload["required_ack_levels"]:
                matches = tuple(
                    item
                    for item in acknowledgements
                    if item.payload["acknowledgement_level"] == level
                )
                if len(matches) != 1:
                    issues.add(
                        "execution_cancellation_missing_required_ack",
                        fact.fact_id,
                        "Cancellation must resolve exactly one required "
                        "acknowledgement.",
                        *(item.fact_id for item in matches),
                    )
                elif not bool(matches[0].payload["accepted"]):
                    issues.add(
                        "execution_cancellation_ack_rejected",
                        fact.fact_id,
                        "A required cancellation acknowledgement was rejected.",
                        matches[0].fact_id,
                    )
            if bool(payload["requests_backend_hold"]):
                permits = tuple(
                    item
                    for item in indexes.dispatch_permit_by_evidence_id.get(
                        cancellation_id,
                        (),
                    )
                    if item.payload["permit_kind"] == "safety_control"
                    and item.payload["operation"] == "hold"
                )
                # A control-domain HOLD may fan out to multiple physical final
                # SDK seams. Require at least one accepted permit and reject
                # any failed member; never collapse the fanout into a reusable
                # permit.
                if not permits or any(
                    not bool(item.payload["validation_accepted"])
                    for item in permits
                ):
                    issues.add(
                        "execution_cancellation_missing_backend_hold_permit",
                        fact.fact_id,
                        "Domain hold requires accepted one-shot safety-control "
                        "permits.",
                        *(item.fact_id for item in permits),
                    )
        elif fact.event_type == "execution_cancellation_ack":
            cancellation_id = str(payload["cancellation_id"])
            if len(indexes.cancellation_by_id.get(cancellation_id, ())) != 1:
                issues.add(
                    "execution_cancellation_ack_missing_request",
                    fact.fact_id,
                    "Cancellation acknowledgement must resolve one "
                    "cancellation.",
                    cancellation_id,
                )


__all__ = [
    "ExecutionCancellationValidationIndex",
    "validate_execution_cancellation_lineage",
]
