"""Authoritative decision to target-action adaptation lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ActionAdaptationValidationIndex:
    """Indexes shared with downstream directive and safety checks."""

    decision_by_id: FactIndex
    request_by_id: FactIndex
    capability_assessment_by_id: FactIndex
    candidate_by_id: FactIndex
    receipt_by_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ActionAdaptationValidationIndex":
        return cls(
            decision_by_id=context.payload_index(
                event_type="arbitration_decision",
                payload_key="decision_id",
            ),
            request_by_id=context.payload_index(
                event_type="action_adaptation_request",
                payload_key="request_id",
            ),
            capability_assessment_by_id=context.payload_index(
                event_type="target_capability_assessment",
                payload_key="assessment_id",
            ),
            candidate_by_id=context.payload_index(
                event_type="adapted_action_candidate",
                payload_key="adapted_candidate_id",
            ),
            receipt_by_id=context.payload_index(
                event_type="action_adaptation_receipt",
                payload_key="receipt_id",
            ),
        )

    @property
    def protocol_active(self) -> bool:
        return bool(self.request_by_id or self.receipt_by_id)


def validate_action_adaptation_lineage(
    context: ControlTraceValidationContext,
    indexes: ActionAdaptationValidationIndex,
    execution_manifest: ControlTraceFact | None,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate request, capability, candidate, and receipt bindings."""

    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "action_adaptation_request":
            decision_id = str(payload["authoritative_decision_id"])
            decisions = indexes.decision_by_id.get(decision_id, ())
            if len(decisions) != 1:
                issues.add(
                    "action_adaptation_missing_decision",
                    fact.fact_id,
                    "Action adaptation must resolve one authoritative decision.",
                    decision_id,
                )
            else:
                decision_payload = decisions[0].payload
                if (
                    str(decision_payload["semantic_content_hash"])
                    != str(payload["authoritative_decision_hash"])
                    or str(decision_payload["result_id"])
                    != str(payload["arbitration_result_id"])
                    or str(decision_payload["selected_action_hash"])
                    != str(payload["candidate_action_hash"])
                    or str(
                        decision_payload["selected_action"][
                            "action_schema_hash"
                        ]
                    )
                    != str(payload["action_schema_hash"])
                ):
                    issues.add(
                        "action_adaptation_decision_lineage_mismatch",
                        fact.fact_id,
                        "Action adaptation request and decision disagree.",
                        decision_id,
                    )
            if execution_manifest is not None and (
                str(payload["component_target_capability_hash"])
                != str(
                    execution_manifest.payload[
                        "component_target_capability_hash"
                    ]
                )
            ):
                issues.add(
                    "action_adaptation_manifest_capability_mismatch",
                    fact.fact_id,
                    "Action adaptation uses another target capability.",
                    execution_manifest.fact_id,
                )
        elif fact.event_type == "adapted_action_candidate":
            request_id = str(payload["request_id"])
            requests = indexes.request_by_id.get(request_id, ())
            if (
                len(requests) != 1
                or str(requests[0].payload["semantic_content_hash"])
                != str(payload["request_hash"])
            ):
                issues.add(
                    "adapted_candidate_missing_request",
                    fact.fact_id,
                    "Adapted candidate must resolve its exact request.",
                    request_id,
                )
            elif str(requests[0].payload["candidate_action_hash"]) != str(
                payload["source_candidate_action_hash"]
            ):
                issues.add(
                    "adapted_candidate_source_mismatch",
                    fact.fact_id,
                    "Adapted candidate references another source action.",
                    request_id,
                )
        elif fact.event_type == "action_adaptation_receipt":
            request_id = str(payload["request_id"])
            requests = indexes.request_by_id.get(request_id, ())
            if (
                len(requests) != 1
                or str(requests[0].payload["semantic_content_hash"])
                != str(payload["request_hash"])
            ):
                issues.add(
                    "action_adaptation_receipt_missing_request",
                    fact.fact_id,
                    "Adaptation receipt must resolve its exact request.",
                    request_id,
                )
            assessment_id = str(payload["capability_assessment_id"])
            assessments = indexes.capability_assessment_by_id.get(
                assessment_id,
                (),
            )
            if (
                len(assessments) != 1
                or str(assessments[0].payload["semantic_content_hash"])
                != str(payload["capability_assessment_hash"])
            ):
                issues.add(
                    "action_adaptation_receipt_missing_assessment",
                    fact.fact_id,
                    "Adaptation receipt must resolve its exact assessment.",
                    assessment_id,
                )
            candidate_id = payload["adapted_candidate_id"]
            if bool(payload["accepted"]):
                candidates = indexes.candidate_by_id.get(
                    str(candidate_id),
                    (),
                )
                if (
                    len(candidates) != 1
                    or str(candidates[0].payload["semantic_content_hash"])
                    != str(payload["adapted_candidate_hash"])
                ):
                    issues.add(
                        "action_adaptation_receipt_missing_candidate",
                        fact.fact_id,
                        "Accepted adaptation must resolve one candidate.",
                        str(candidate_id),
                    )
            elif candidate_id is not None:
                issues.add(
                    "rejected_adaptation_has_candidate",
                    fact.fact_id,
                    "Rejected adaptation cannot reference a candidate.",
                    str(candidate_id),
                )


__all__ = [
    "ActionAdaptationValidationIndex",
    "validate_action_adaptation_lineage",
]
