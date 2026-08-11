"""Publication, adaptation, directive, and shared-safety lineage."""

from __future__ import annotations

from dataclasses import dataclass

from .action_adaptation import ActionAdaptationValidationIndex, FactIndex
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


@dataclass(frozen=True, slots=True)
class ExecutionDirectiveValidationIndex:
    """Directive-owned indexes not shared by another validator family."""

    publication_attempt_by_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ExecutionDirectiveValidationIndex":
        return cls(
            publication_attempt_by_id=context.payload_index(
                event_type="arbitration_publication_attempt",
                payload_key="publication_attempt_id",
            )
        )


def validate_execution_directive_lineage(
    context: ControlTraceValidationContext,
    indexes: ExecutionDirectiveValidationIndex,
    adaptation_indexes: ActionAdaptationValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate execution directives and their adapted shared-safety inputs."""

    for fact in context.facts_of_type("execution_directive"):
        payload = fact.payload
        attempt_id = str(payload["publication_attempt_id"])
        attempts = indexes.publication_attempt_by_id.get(attempt_id, ())
        if len(attempts) != 1:
            issues.add(
                "execution_directive_missing_publication_attempt",
                fact.fact_id,
                "ExecutionDirective must resolve one publication attempt.",
                attempt_id,
            )
        decision_id = payload["decision_id"]
        if decision_id is not None:
            decisions = adaptation_indexes.decision_by_id.get(
                str(decision_id),
                (),
            )
            if len(decisions) != 1:
                issues.add(
                    "execution_directive_missing_decision",
                    fact.fact_id,
                    (
                        "Selected ExecutionDirective must resolve one "
                        "authoritative decision."
                    ),
                    str(decision_id),
                )
            adaptation_receipt_id = payload[
                "action_adaptation_receipt_id"
            ]
            if adaptation_indexes.protocol_active:
                receipts = adaptation_indexes.receipt_by_id.get(
                    str(adaptation_receipt_id),
                    (),
                )
                if len(receipts) != 1 or str(
                    receipts[0].payload["semantic_content_hash"]
                ) != str(payload["action_adaptation_receipt_hash"]):
                    issues.add(
                        "execution_directive_missing_adaptation_receipt",
                        fact.fact_id,
                        "Selected directive must resolve adaptation receipt.",
                        str(adaptation_receipt_id),
                    )
                elif bool(receipts[0].payload["accepted"]) == (
                    payload["action_adaptation_fallback"] is not None
                ):
                    issues.add(
                        "execution_directive_adaptation_fallback_mismatch",
                        fact.fact_id,
                        "Adaptation fallback must exist exactly for rejection.",
                        str(adaptation_receipt_id),
                    )
        elif len(attempts) == 1 and bool(attempts[0].payload["accepted"]):
            issues.add(
                "nonselected_directive_has_accepted_publication",
                fact.fact_id,
                (
                    "A directive without decision_id cannot reference "
                    "accepted publication."
                ),
                attempt_id,
            )

    if not adaptation_indexes.protocol_active:
        return
    for fact in context.facts_of_type("shared_safety_result"):
        payload = fact.payload
        candidate_id = str(payload["input_action_id"])
        candidates = adaptation_indexes.candidate_by_id.get(candidate_id, ())
        if len(candidates) != 1 or str(
            candidates[0].payload["semantic_content_hash"]
        ) != str(payload["input_action_hash"]):
            issues.add(
                "shared_safety_missing_adapted_candidate",
                fact.fact_id,
                "Shared safety must resolve its exact adapted input.",
                candidate_id,
            )


__all__ = [
    "ExecutionDirectiveValidationIndex",
    "validate_execution_directive_lineage",
]
