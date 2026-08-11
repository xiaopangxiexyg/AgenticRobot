"""Causal adapter from one Sim-HIL cycle to passive policy feedback.

This module only joins identities already published by composition and the
downstream execution observer.  It never reconstructs IK/reference state and
never changes authority, safety, or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robot_motion.deploy import PolicyActionEmission
from robot_motion.deploy.execution_feedback import (
    POLICY_EXECUTION_RELATION_COMPOSED,
    POLICY_EXECUTION_RELATION_EXACT,
    POLICY_EXECUTION_RELATION_NOT_SELECTED,
    POLICY_EXECUTION_RELATION_UNKNOWN,
    PolicyExecutionFeedback,
)


@dataclass(frozen=True)
class SimHILPolicyFeedbackProjection:
    """Result of a passive same-cycle projection attempt."""

    status: str
    feedback: PolicyExecutionFeedback | None = None
    detail: str | None = None


def _composition_cut(cycle: Any) -> tuple[Any | None, Any | None]:
    step = getattr(cycle, "composition_step", None)
    compatibility = getattr(step, "evaluation", None)
    if compatibility is not None:
        return (
            getattr(compatibility, "input_snapshot", None),
            getattr(compatibility, "composition_evaluation", None),
        )
    submission = getattr(step, "submission", None)
    if submission is not None:
        return (
            getattr(submission, "input_snapshot", None),
            getattr(submission, "composition_evaluation", None),
        )
    return None, None


def _emission_selected_by_record(
    record: Any,
    emission_by_id: Mapping[str, PolicyActionEmission],
) -> PolicyActionEmission | None:
    fact_ids = tuple(getattr(record, "source_fact_ids", ()))
    fact_hashes = tuple(getattr(record, "source_fact_hashes", ()))
    if len(fact_ids) != len(fact_hashes):
        return None
    hash_by_id = dict(zip(fact_ids, fact_hashes, strict=True))
    for emission_id in fact_ids:
        emission = emission_by_id.get(emission_id)
        if (
            emission is not None
            and hash_by_id[emission_id] == emission.semantic_content_hash
        ):
            return emission
    return None


def _exact_association(
    observer: Any | None,
    *,
    target_admission: Any | None,
) -> Any | None:
    if observer is None or target_admission is None:
        return None
    snapshot = observer.snapshot()
    association = getattr(snapshot, "latest_association", None)
    if association is None or getattr(association, "status", None) != "available":
        return None
    receipt_pairs = {
        (reference.receipt_id, reference.receipt_hash)
        for reference in tuple(getattr(association, "receipt_references", ()))
    }
    expected = (
        target_admission.receipt_id,
        target_admission.semantic_content_hash,
    )
    return association if expected in receipt_pairs else None


def project_sim_hil_policy_execution_feedback(
    *,
    cycle: Any,
    cycle_emission: PolicyActionEmission | None,
    emission_by_id: Mapping[str, PolicyActionEmission],
    execution_difference_observer: Any | None,
    observed_at: float,
    clock_domain_id: str,
    policy_role: str = "policy",
) -> SimHILPolicyFeedbackProjection:
    """Bind selected source, composed output, and exact downstream receipts."""

    input_snapshot, evaluation = _composition_cut(cycle)
    selected = (
        None
        if input_snapshot is None
        else input_snapshot.selected_input_by_role.get(policy_role)
    )
    if selected is None:
        emission = cycle_emission
        relation = POLICY_EXECUTION_RELATION_NOT_SELECTED
    else:
        emission = _emission_selected_by_record(selected, emission_by_id)
        relation = POLICY_EXECUTION_RELATION_UNKNOWN
        candidate = None if evaluation is None else evaluation.composed_candidate
        if candidate is not None:
            relation = (
                POLICY_EXECUTION_RELATION_EXACT
                if candidate.semantic_content_hash
                == selected.action.semantic_content_hash
                else POLICY_EXECUTION_RELATION_COMPOSED
            )
    if emission is None:
        return SimHILPolicyFeedbackProjection(
            status="no_exact_policy_emission",
            detail=(
                "selected policy input has no matching emission lineage"
                if selected is not None
                else "cycle did not produce or select a policy emission"
            ),
        )

    execution = getattr(cycle, "execution", None)
    if execution is None:
        return SimHILPolicyFeedbackProjection(
            status="no_execution_result",
            detail="cycle does not expose downstream execution",
        )
    target_dispatch = getattr(execution, "target_dispatch_result", None)
    target_admission = (
        None
        if target_dispatch is None
        else getattr(target_dispatch, "target_admission", None)
    )
    association = _exact_association(
        execution_difference_observer,
        target_admission=target_admission,
    )
    reference_values: dict[str, tuple[float, ...]] = {}
    measurement_values: dict[str, tuple[float, ...]] = {}
    receipt_references: tuple[str, ...] = ()
    if association is not None:
        for component_id, difference in association.component_difference.items():
            reference = difference.get("reference")
            measured = difference.get("measured")
            if reference is not None and measured is not None:
                reference_values[str(component_id)] = tuple(reference)
                measurement_values[str(component_id)] = tuple(measured)
        receipt_references = tuple(
            reference.receipt_id for reference in association.receipt_references
        )
    safety = getattr(execution, "safety_result", None)
    feedback = PolicyExecutionFeedback(
        policy_emission_id=emission.emission_id,
        policy_emission_hash=emission.semantic_content_hash,
        policy_output_relation=relation,
        execution_status=str(getattr(execution, "status", "unknown")),
        downstream_action_status=(
            None if safety is None else str(getattr(safety, "status", "unknown"))
        ),
        downstream_modified_components=(
            () if safety is None else tuple(getattr(safety, "modified_components", ()))
        ),
        observed_at=float(observed_at),
        clock_domain_id=str(clock_domain_id),
        composition_evaluation_id=(
            None if evaluation is None else evaluation.composition_evaluation_id
        ),
        composition_evaluation_hash=(
            None if evaluation is None else evaluation.semantic_content_hash
        ),
        target_admission_receipt_id=(
            None if target_admission is None else target_admission.receipt_id
        ),
        target_admission_receipt_hash=(
            None
            if target_admission is None
            else target_admission.semantic_content_hash
        ),
        association_id=(
            None if association is None else association.association_id
        ),
        association_hash=(
            None if association is None else association.semantic_content_hash
        ),
        association_status=(
            None if association is None else association.status
        ),
        reference_value_by_component=reference_values,
        measurement_value_by_component=measurement_values,
        receipt_references=receipt_references,
    )
    return SimHILPolicyFeedbackProjection(status="available", feedback=feedback)


__all__ = [
    "SimHILPolicyFeedbackProjection",
    "project_sim_hil_policy_execution_feedback",
]
