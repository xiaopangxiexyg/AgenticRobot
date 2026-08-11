"""Passive downstream feedback for one emitted policy action."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact


POLICY_EXECUTION_RELATION_EXACT = "exact_policy_action"
POLICY_EXECUTION_RELATION_COMPOSED = "composed_with_other_sources"
POLICY_EXECUTION_RELATION_NOT_SELECTED = "policy_not_selected"
POLICY_EXECUTION_RELATION_UNKNOWN = "unknown"
POLICY_EXECUTION_RELATIONS = frozenset(
    {
        POLICY_EXECUTION_RELATION_EXACT,
        POLICY_EXECUTION_RELATION_COMPOSED,
        POLICY_EXECUTION_RELATION_NOT_SELECTED,
        POLICY_EXECUTION_RELATION_UNKNOWN,
    }
)
POLICY_EXECUTION_FEEDBACK_HASH_DOMAIN_V1 = (
    "dexteleop.policy_execution_feedback.v1"
)
POLICY_EXECUTION_FEEDBACK_APPLICATION_HASH_DOMAIN_V1 = (
    "dexteleop.policy_execution_feedback_application.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _component_values(
    values: Mapping[str, Sequence[float]],
) -> Mapping[str, tuple[float, ...]]:
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_component, raw_values in dict(values).items():
        component = _required(raw_component, name="component_id")
        array = np.asarray(raw_values, dtype=float).reshape(-1)
        if array.size == 0 or np.any(~np.isfinite(array)):
            raise ValueError(
                f"Execution feedback component {component!r} must be finite."
            )
        normalized[component] = tuple(float(value) for value in array)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PolicyExecutionFeedback:
    """Causal execution evidence; it never grants authority or runs motion."""

    policy_emission_id: str
    policy_emission_hash: str
    policy_output_relation: str
    execution_status: str
    downstream_action_status: str | None
    downstream_modified_components: tuple[str, ...]
    observed_at: float
    clock_domain_id: str
    composition_evaluation_id: str | None = None
    composition_evaluation_hash: str | None = None
    target_admission_receipt_id: str | None = None
    target_admission_receipt_hash: str | None = None
    association_id: str | None = None
    association_hash: str | None = None
    association_status: str | None = None
    reference_value_by_component: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    measurement_value_by_component: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    receipt_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        relation = _required(
            self.policy_output_relation,
            name="policy_output_relation",
        )
        if relation not in POLICY_EXECUTION_RELATIONS:
            raise ValueError(f"Unsupported policy execution relation: {relation!r}.")
        paired_fields = (
            (self.composition_evaluation_id, self.composition_evaluation_hash),
            (self.target_admission_receipt_id, self.target_admission_receipt_hash),
            (self.association_id, self.association_hash),
        )
        if any((left is None) != (right is None) for left, right in paired_fields):
            raise ValueError("Execution feedback identity pairs must be complete.")
        references = _component_values(self.reference_value_by_component)
        measurements = _component_values(self.measurement_value_by_component)
        if set(references) != set(measurements):
            raise ValueError("Reference and measurement components must match.")
        if references and self.association_id is None:
            raise ValueError("Execution values require an exact association identity.")
        receipt_refs = tuple(
            _required(value, name="receipt_reference")
            for value in self.receipt_references
        )
        if len(receipt_refs) != len(set(receipt_refs)):
            raise ValueError("receipt_references must be unique.")
        object.__setattr__(self, "policy_output_relation", relation)
        object.__setattr__(self, "observed_at", _finite(self.observed_at, name="observed_at"))
        object.__setattr__(
            self,
            "downstream_modified_components",
            tuple(
                _required(value, name="downstream_modified_component")
                for value in self.downstream_modified_components
            ),
        )
        if len(self.downstream_modified_components) != len(
            set(self.downstream_modified_components)
        ):
            raise ValueError("downstream_modified_components must be unique.")
        object.__setattr__(self, "reference_value_by_component", references)
        object.__setattr__(self, "measurement_value_by_component", measurements)
        object.__setattr__(self, "receipt_references", receipt_refs)
        for name in (
            "policy_emission_id",
            "policy_emission_hash",
            "execution_status",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        for name in (
            "composition_evaluation_id",
            "composition_evaluation_hash",
            "target_admission_receipt_id",
            "target_admission_receipt_hash",
            "association_id",
            "association_hash",
            "association_status",
            "downstream_action_status",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name), name=name))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain=POLICY_EXECUTION_FEEDBACK_HASH_DOMAIN_V1,
        )

    @property
    def feedback_id(self) -> str:
        return f"policy-execution-feedback:{self.semantic_content_hash}"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "policy_emission_id": self.policy_emission_id,
            "policy_emission_hash": self.policy_emission_hash,
            "policy_output_relation": self.policy_output_relation,
            "execution_status": self.execution_status,
            "downstream_action_status": self.downstream_action_status,
            "downstream_modified_components": (
                self.downstream_modified_components
            ),
            "observed_at": self.observed_at,
            "clock_domain_id": self.clock_domain_id,
            "composition_evaluation_id": self.composition_evaluation_id,
            "composition_evaluation_hash": self.composition_evaluation_hash,
            "target_admission_receipt_id": self.target_admission_receipt_id,
            "target_admission_receipt_hash": self.target_admission_receipt_hash,
            "association_id": self.association_id,
            "association_hash": self.association_hash,
            "association_status": self.association_status,
            "reference_value_by_component": dict(self.reference_value_by_component),
            "measurement_value_by_component": dict(self.measurement_value_by_component),
            "receipt_references": self.receipt_references,
        }
        if include_hash:
            payload.update(
                {
                    "feedback_id": self.feedback_id,
                    "semantic_content_hash": self.semantic_content_hash,
                }
            )
        return payload


@dataclass(frozen=True)
class PolicyExecutionFeedbackApplication:
    feedback_id: str
    feedback_hash: str
    decision: str
    reason: str
    target_generation_before: int | None = None
    target_generation_after: int | None = None

    def __post_init__(self) -> None:
        for name in ("feedback_id", "feedback_hash", "decision", "reason"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        for name in ("target_generation_before", "target_generation_after"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "feedback_id": self.feedback_id,
                "feedback_hash": self.feedback_hash,
                "decision": self.decision,
                "reason": self.reason,
                "target_generation_before": self.target_generation_before,
                "target_generation_after": self.target_generation_after,
            },
            domain=POLICY_EXECUTION_FEEDBACK_APPLICATION_HASH_DOMAIN_V1,
        )

    @property
    def receipt_id(self) -> str:
        return f"policy-execution-feedback-application:{self.semantic_content_hash}"


def project_policy_execution_feedback(
    feedback: PolicyExecutionFeedback,
    application: PolicyExecutionFeedbackApplication,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    if application.feedback_id != feedback.feedback_id or application.feedback_hash != feedback.semantic_content_hash:
        raise ValueError("Feedback application does not bind the supplied feedback.")
    return ControlTraceFact.create(
        fact_id=application.receipt_id,
        event_type="policy_execution_feedback",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=feedback.observed_at,
        clock_domain_id=feedback.clock_domain_id,
        payload={
            "feedback": feedback.to_dict(),
            "application_receipt_id": application.receipt_id,
            "application_receipt_hash": application.semantic_content_hash,
            "application_decision": application.decision,
            "application_reason": application.reason,
            "target_generation_before": application.target_generation_before,
            "target_generation_after": application.target_generation_after,
        },
    )


__all__ = [
    "POLICY_EXECUTION_RELATION_COMPOSED",
    "POLICY_EXECUTION_RELATION_EXACT",
    "POLICY_EXECUTION_RELATION_NOT_SELECTED",
    "POLICY_EXECUTION_RELATION_UNKNOWN",
    "PolicyExecutionFeedback",
    "PolicyExecutionFeedbackApplication",
    "project_policy_execution_feedback",
]
