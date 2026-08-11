"""Target-execution request, target, admission, and activation lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class TargetExecutionValidationIndex:
    """Indexes for one target-execution handoff chain."""

    request_by_id: FactIndex
    submission_by_request_id: FactIndex
    completion_by_request_id: FactIndex
    target_by_request_id: FactIndex
    admission_by_target_id: FactIndex
    activation_by_target_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "TargetExecutionValidationIndex":
        return cls(
            request_by_id=context.payload_index(
                event_type="target_execution_request",
                payload_key="request_id",
            ),
            submission_by_request_id=context.payload_index(
                event_type="target_execution_submission",
                payload_key="target_execution_request_id",
            ),
            completion_by_request_id=context.payload_index(
                event_type="target_execution_completion",
                payload_key="target_execution_request_id",
            ),
            target_by_request_id=context.payload_index(
                event_type="component_execution_target",
                payload_key="target_execution_request_id",
            ),
            admission_by_target_id=context.payload_index(
                event_type="component_execution_target_admission",
                payload_key="target_id",
            ),
            activation_by_target_id=context.payload_index(
                event_type="component_execution_target_activation",
                payload_key="target_id",
            ),
        )


def validate_target_execution_lineage(
    indexes: TargetExecutionValidationIndex,
    *,
    enabled: bool,
    activation_required: bool,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate the full target-execution handoff without owning execution."""

    if not enabled:
        return
    for request_id, requests in indexes.request_by_id.items():
        if len(requests) != 1:
            continue
        request = requests[0]
        request_hash = request.payload["semantic_content_hash"]
        submissions = tuple(
            indexes.submission_by_request_id.get(request_id, ())
        )
        if len(submissions) != 1:
            issues.add(
                "component_execution_request_missing_submission",
                request.fact_id,
                "Each component execution request needs exactly one "
                "target submission receipt.",
                *(item.fact_id for item in submissions),
            )
            continue
        submission = submissions[0]
        if submission.payload["target_execution_request_hash"] != request_hash:
            issues.add(
                "component_execution_submission_request_hash_mismatch",
                submission.fact_id,
                "Target submission receipt must bind the exact execution "
                "request hash.",
                request.fact_id,
            )
        if not bool(submission.payload["accepted_for_processing"]):
            continue
        completions = tuple(
            indexes.completion_by_request_id.get(request_id, ())
        )
        if len(completions) != 1:
            issues.add(
                "component_execution_request_missing_completion",
                request.fact_id,
                "Each accepted component execution request needs exactly "
                "one completion receipt.",
                *(item.fact_id for item in completions),
            )
        elif completions[0].payload["status"] != "resolved":
            # A failed or cancelled completion is terminal evidence; it must
            # not be paired with a resolved target.
            unexpected_targets = tuple(
                indexes.target_by_request_id.get(request_id, ())
            )
            if unexpected_targets:
                issues.add(
                    "component_execution_failed_request_has_target",
                    completions[0].fact_id,
                    "A non-resolved completion must not publish a component "
                    "execution target.",
                    *(item.fact_id for item in unexpected_targets),
                )
            continue
        targets = tuple(indexes.target_by_request_id.get(request_id, ()))
        if len(targets) != 1:
            issues.add(
                "component_execution_request_missing_target",
                request.fact_id,
                "Each accepted component execution request needs exactly "
                "one resolved component target.",
                *(item.fact_id for item in targets),
            )
            continue
        target = targets[0]
        if (
            target.payload["target_execution_request_hash"] != request_hash
            or target.payload["submission_receipt_id"] != submission.fact_id
            or target.payload["submission_receipt_hash"]
            != submission.payload["semantic_content_hash"]
        ):
            issues.add(
                "component_execution_target_request_binding_mismatch",
                target.fact_id,
                "Component target must bind the exact request and submission "
                "receipt.",
                request.fact_id,
                submission.fact_id,
            )
        target_id = str(target.payload["target_id"])
        target_hash = target.payload["semantic_content_hash"]
        admissions = tuple(indexes.admission_by_target_id.get(target_id, ()))
        if len(admissions) != 1 or not bool(admissions[0].payload["accepted"]):
            issues.add(
                "component_execution_target_missing_admission",
                target.fact_id,
                "Each resolved component target needs one accepted target "
                "admission receipt.",
                *(item.fact_id for item in admissions),
            )
            continue
        admission = admissions[0]
        if (
            admission.payload["target_hash"] != target_hash
            or admission.payload["target_execution_request_id"] != request_id
            or admission.payload["target_execution_request_hash"] != request_hash
        ):
            issues.add(
                "component_execution_target_admission_binding_mismatch",
                admission.fact_id,
                "Target admission must bind the exact component target and "
                "execution request.",
                target.fact_id,
                request.fact_id,
            )
        activations = tuple(indexes.activation_by_target_id.get(target_id, ()))
        if activation_required and len(activations) != 1:
            issues.add(
                "component_execution_target_missing_activation",
                target.fact_id,
                "Each admitted component target needs exactly one activation "
                "receipt.",
                *(item.fact_id for item in activations),
            )
        elif activation_required:
            activation = activations[0]
            if (
                activation.payload["target_hash"] != target_hash
                or activation.payload["target_execution_request_id"] != request_id
                or activation.payload["target_execution_request_hash"]
                != request_hash
                or activation.payload["target_admission_receipt_id"]
                != admission.fact_id
                or activation.payload["target_admission_receipt_hash"]
                != admission.payload["semantic_content_hash"]
            ):
                issues.add(
                    "component_execution_target_activation_binding_mismatch",
                    activation.fact_id,
                    "Target activation must bind the exact request, target, "
                    "and admission receipt.",
                    request.fact_id,
                    target.fact_id,
                    admission.fact_id,
                )
        elif activations:
            issues.add(
                "component_execution_target_unexpected_activation",
                target.fact_id,
                "Immediate component targets must omit a separate activation "
                "receipt.",
                *(item.fact_id for item in activations),
            )
        if len(completions) == 1:
            completion = completions[0]
            if (
                completion.payload["target_execution_request_hash"]
                != request_hash
                or completion.payload["submission_receipt_id"]
                != submission.fact_id
                or completion.payload["submission_receipt_hash"]
                != submission.payload["semantic_content_hash"]
                or completion.payload["target_id"] != target_id
                or completion.payload["target_hash"] != target_hash
            ):
                issues.add(
                    "component_execution_completion_binding_mismatch",
                    completion.fact_id,
                    "Execution completion must bind the exact request, "
                    "submission, and component target.",
                    request.fact_id,
                    submission.fact_id,
                    target.fact_id,
                )


__all__ = [
    "TargetExecutionValidationIndex",
    "validate_target_execution_lineage",
]
