"""Pure, pluggable task-evaluation contracts for Agent Skill operations.

Evaluation is deliberately downstream of execution admission and execution
receipts.  Policy self-evaluation is one possible task-outcome input; no value
in this module is a safety permit, authority lease, dispatch receipt, lifecycle
rearm receipt, or proof that the lower runtime executed an operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .skill_catalog import (
    MAX_AGENT_SKILL_TIMEOUT_MS_V1,
    validate_route_id,
    validate_skill_id,
    validate_version,
)

AGENT_SKILL_EVALUATOR_KINDS_V1 = (
    "environment",
    "external",
    "operator",
    "policy_self",
    "rule_based",
)
AGENT_SKILL_EVALUATOR_ROLES_V1 = (
    "advisory",
    "contributing",
    "required",
    "veto",
)
AGENT_SKILL_EVALUATOR_LOCATIONS_V1 = (
    "external_service",
    "gateway_worker",
    "lower_runtime",
    "operator_console",
    "policy_runtime",
)
AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1 = (
    "mark_error",
    "omit",
    "veto",
)
AGENT_SKILL_EVALUATION_VERDICTS_V1 = (
    "error",
    "failed",
    "inconclusive",
    "succeeded",
)
AGENT_SKILL_AGGREGATE_VERDICTS_V1 = (
    "failed",
    "inconclusive",
    "succeeded",
)
AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1 = (
    "fail",
    "inconclusive",
)
AGENT_SKILL_DISAGREEMENT_POLICIES_V1 = (
    "aggregation_policy",
    "inconclusive",
    "veto",
)

AGENT_SKILL_EVALUATOR_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.evaluator_spec.v1"
)
AGENT_SKILL_EVALUATION_PLAN_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.evaluation_plan.v1"
)
AGENT_SKILL_EVALUATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.evaluation_receipt.v1"
)
AGENT_SKILL_AGGREGATE_EVALUATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.aggregate_evaluation_receipt.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if normalized != value:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _optional(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required(value, field_name=field_name)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive_integer(
    value: object,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return value


def _enum(value: object, *, field_name: str, allowed: tuple[str, ...]) -> str:
    normalized = _required(value, field_name=field_name)
    if normalized not in allowed:
        raise ValueError(f"Unsupported {field_name}: {normalized!r}.")
    return normalized


def _unique_sorted(
    values: tuple[str, ...] | list[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list of strings.")
    normalized = tuple(_required(item, field_name=field_name) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SkillEvaluatorSpec:
    """One immutable evaluator registration within a Route-bound plan."""

    evaluator_id: str
    evaluator_version: str
    evaluator_kind: str
    input_evidence_schema_hash: str
    output_schema_hash: str
    role: str
    execution_location: str
    deadline_ms: int
    failure_policy: str
    evidence_domain_id: str
    configuration_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_id",
            validate_route_id(self.evaluator_id),
        )
        object.__setattr__(
            self,
            "evaluator_version",
            validate_version(
                self.evaluator_version,
                field_name="evaluator_version",
            ),
        )
        for field_name in (
            "input_evidence_schema_hash",
            "output_schema_hash",
            "evidence_domain_id",
            "configuration_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "evaluator_kind",
            _enum(
                self.evaluator_kind,
                field_name="evaluator_kind",
                allowed=AGENT_SKILL_EVALUATOR_KINDS_V1,
            ),
        )
        object.__setattr__(
            self,
            "role",
            _enum(
                self.role,
                field_name="role",
                allowed=AGENT_SKILL_EVALUATOR_ROLES_V1,
            ),
        )
        object.__setattr__(
            self,
            "execution_location",
            _enum(
                self.execution_location,
                field_name="execution_location",
                allowed=AGENT_SKILL_EVALUATOR_LOCATIONS_V1,
            ),
        )
        object.__setattr__(
            self,
            "deadline_ms",
            _positive_integer(
                self.deadline_ms,
                field_name="deadline_ms",
                maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            ),
        )
        object.__setattr__(
            self,
            "failure_policy",
            _enum(
                self.failure_policy,
                field_name="failure_policy",
                allowed=AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1,
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_EVALUATOR_SPEC_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "evaluator_kind": self.evaluator_kind,
            "input_evidence_schema_hash": self.input_evidence_schema_hash,
            "output_schema_hash": self.output_schema_hash,
            "role": self.role,
            "execution_location": self.execution_location,
            "deadline_ms": self.deadline_ms,
            "failure_policy": self.failure_policy,
            "evidence_domain_id": self.evidence_domain_id,
            "configuration_hash": self.configuration_hash,
        }
        if include_hash:
            payload["evaluator_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillEvaluationPlan:
    """Route/deployment-bound plan; requests cannot select evaluators."""

    evaluation_plan_id: str
    evaluation_plan_version: str
    skill_definition_hash: str
    route_hash: str
    deployment_binding_hash: str
    evaluator_specs: tuple[SkillEvaluatorSpec, ...]
    aggregation_policy_id: str
    aggregation_policy_hash: str
    required_evaluation_deadline_ms: int
    missing_evidence_policy: str
    disagreement_policy: str
    task_outcome_schema_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_plan_id",
            validate_route_id(self.evaluation_plan_id),
        )
        object.__setattr__(
            self,
            "evaluation_plan_version",
            validate_version(
                self.evaluation_plan_version,
                field_name="evaluation_plan_version",
            ),
        )
        for field_name in (
            "skill_definition_hash",
            "route_hash",
            "deployment_binding_hash",
            "aggregation_policy_id",
            "aggregation_policy_hash",
            "task_outcome_schema_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        specs = tuple(self.evaluator_specs)
        if any(not isinstance(item, SkillEvaluatorSpec) for item in specs):
            raise TypeError(
                "evaluator_specs must contain SkillEvaluatorSpec values."
            )
        identities = tuple(
            (item.evaluator_id, item.evaluator_version) for item in specs
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Evaluator identities must be unique within a plan.")
        object.__setattr__(
            self,
            "evaluator_specs",
            tuple(
                sorted(
                    specs,
                    key=lambda item: (item.evaluator_id, item.evaluator_version),
                )
            ),
        )
        plan_deadline = _positive_integer(
            self.required_evaluation_deadline_ms,
            field_name="required_evaluation_deadline_ms",
            maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
        )
        if any(item.deadline_ms > plan_deadline for item in specs):
            raise ValueError("Evaluator deadline cannot exceed plan deadline.")
        object.__setattr__(
            self,
            "required_evaluation_deadline_ms",
            plan_deadline,
        )
        object.__setattr__(
            self,
            "missing_evidence_policy",
            _enum(
                self.missing_evidence_policy,
                field_name="missing_evidence_policy",
                allowed=AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1,
            ),
        )
        object.__setattr__(
            self,
            "disagreement_policy",
            _enum(
                self.disagreement_policy,
                field_name="disagreement_policy",
                allowed=AGENT_SKILL_DISAGREEMENT_POLICIES_V1,
            ),
        )

    def evaluator(
        self,
        evaluator_id: str,
        evaluator_version: str,
    ) -> SkillEvaluatorSpec:
        identity = (
            validate_route_id(evaluator_id),
            validate_version(evaluator_version, field_name="evaluator_version"),
        )
        for spec in self.evaluator_specs:
            if (spec.evaluator_id, spec.evaluator_version) == identity:
                return spec
        raise KeyError(f"Unknown evaluator: {identity!r}.")

    @property
    def required_evaluator_ids(self) -> tuple[str, ...]:
        return tuple(
            spec.evaluator_id
            for spec in self.evaluator_specs
            if spec.role in {"required", "veto"}
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "evaluation_plan_id": self.evaluation_plan_id,
                "evaluation_plan_version": self.evaluation_plan_version,
                "skill_definition_hash": self.skill_definition_hash,
                "route_hash": self.route_hash,
                "deployment_binding_hash": self.deployment_binding_hash,
                "evaluator_spec_hashes": tuple(
                    item.semantic_content_hash for item in self.evaluator_specs
                ),
                "aggregation_policy_id": self.aggregation_policy_id,
                "aggregation_policy_hash": self.aggregation_policy_hash,
                "required_evaluation_deadline_ms": (
                    self.required_evaluation_deadline_ms
                ),
                "missing_evidence_policy": self.missing_evidence_policy,
                "disagreement_policy": self.disagreement_policy,
                "task_outcome_schema_hash": self.task_outcome_schema_hash,
            },
            domain=AGENT_SKILL_EVALUATION_PLAN_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "evaluation_plan_id": self.evaluation_plan_id,
            "evaluation_plan_version": self.evaluation_plan_version,
            "skill_definition_hash": self.skill_definition_hash,
            "route_hash": self.route_hash,
            "deployment_binding_hash": self.deployment_binding_hash,
            "evaluator_specs": [item.to_wire() for item in self.evaluator_specs],
            "aggregation_policy_id": self.aggregation_policy_id,
            "aggregation_policy_hash": self.aggregation_policy_hash,
            "required_evaluation_deadline_ms": (
                self.required_evaluation_deadline_ms
            ),
            "missing_evidence_policy": self.missing_evidence_policy,
            "disagreement_policy": self.disagreement_policy,
            "task_outcome_schema_hash": self.task_outcome_schema_hash,
            "evaluation_plan_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class SkillEvaluationReceipt:
    """Typed result from exactly one registered evaluator."""

    operation_id: str
    request_hash: str
    skill_id: str
    skill_version: str
    skill_definition_hash: str
    route_id: str
    route_version: str
    route_hash: str
    deployment_binding_hash: str
    evaluation_plan_hash: str
    evaluator_id: str
    evaluator_version: str
    evaluator_hash: str
    evaluator_kind: str
    evidence_domain_id: str
    input_evidence_hash: str
    evaluated_cut_id: str
    evaluated_at: float
    verdict: str
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    score: float | None = None
    score_schema_hash: str | None = None
    confidence: float | None = None
    confidence_semantics_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "operation_id",
            "request_hash",
            "skill_definition_hash",
            "route_hash",
            "deployment_binding_hash",
            "evaluation_plan_hash",
            "evaluator_hash",
            "evidence_domain_id",
            "input_evidence_hash",
            "evaluated_cut_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )
        object.__setattr__(self, "route_id", validate_route_id(self.route_id))
        object.__setattr__(
            self,
            "route_version",
            validate_version(self.route_version, field_name="route_version"),
        )
        object.__setattr__(
            self,
            "evaluator_id",
            validate_route_id(self.evaluator_id),
        )
        object.__setattr__(
            self,
            "evaluator_version",
            validate_version(
                self.evaluator_version,
                field_name="evaluator_version",
            ),
        )
        object.__setattr__(
            self,
            "evaluator_kind",
            _enum(
                self.evaluator_kind,
                field_name="evaluator_kind",
                allowed=AGENT_SKILL_EVALUATOR_KINDS_V1,
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _finite(self.evaluated_at, field_name="evaluated_at"),
        )
        object.__setattr__(
            self,
            "verdict",
            _enum(
                self.verdict,
                field_name="verdict",
                allowed=AGENT_SKILL_EVALUATION_VERDICTS_V1,
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted(self.reason_codes, field_name="reason_codes"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _unique_sorted(self.evidence_refs, field_name="evidence_refs"),
        )
        score_schema_hash = _optional(
            self.score_schema_hash,
            field_name="score_schema_hash",
        )
        if self.score is not None:
            object.__setattr__(self, "score", _finite(self.score, field_name="score"))
            if score_schema_hash is None:
                raise ValueError("score requires score_schema_hash.")
        elif score_schema_hash is not None:
            raise ValueError("score_schema_hash requires score.")
        object.__setattr__(self, "score_schema_hash", score_schema_hash)
        confidence_semantics = _optional(
            self.confidence_semantics_id,
            field_name="confidence_semantics_id",
        )
        if self.confidence is not None:
            confidence = _finite(self.confidence, field_name="confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1].")
            if confidence_semantics is None:
                raise ValueError("confidence requires confidence_semantics_id.")
            object.__setattr__(self, "confidence", confidence)
        elif confidence_semantics is not None:
            raise ValueError("confidence_semantics_id requires confidence.")
        object.__setattr__(
            self,
            "confidence_semantics_id",
            confidence_semantics,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_EVALUATION_RECEIPT_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "operation_id": self.operation_id,
            "request_hash": self.request_hash,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "route_id": self.route_id,
            "route_version": self.route_version,
            "route_hash": self.route_hash,
            "deployment_binding_hash": self.deployment_binding_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "evaluator_hash": self.evaluator_hash,
            "evaluator_kind": self.evaluator_kind,
            "evidence_domain_id": self.evidence_domain_id,
            "input_evidence_hash": self.input_evidence_hash,
            "evaluated_cut_id": self.evaluated_cut_id,
            "evaluated_at": self.evaluated_at,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "evidence_refs": list(self.evidence_refs),
            "score": self.score,
            "score_schema_hash": self.score_schema_hash,
            "confidence": self.confidence,
            "confidence_semantics_id": self.confidence_semantics_id,
        }
        if include_hash:
            payload["evaluation_receipt_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillAggregateEvaluationReceipt:
    """Versioned aggregator output retaining all consumed/missing lineage."""

    operation_id: str
    request_hash: str
    evaluation_plan_hash: str
    aggregation_policy_id: str
    aggregation_policy_hash: str
    consumed_evaluation_receipt_hashes: tuple[str, ...]
    ignored_evaluation_receipt_hashes: tuple[str, ...]
    missing_evaluator_ids: tuple[str, ...]
    verdict: str
    reason_codes: tuple[str, ...]
    generated_at: float

    def __post_init__(self) -> None:
        for field_name in (
            "operation_id",
            "request_hash",
            "evaluation_plan_hash",
            "aggregation_policy_id",
            "aggregation_policy_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "consumed_evaluation_receipt_hashes",
            "ignored_evaluation_receipt_hashes",
            "missing_evaluator_ids",
            "reason_codes",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique_sorted(getattr(self, field_name), field_name=field_name),
            )
        if set(self.consumed_evaluation_receipt_hashes) & set(
            self.ignored_evaluation_receipt_hashes
        ):
            raise ValueError("Consumed and ignored receipt sets must be disjoint.")
        object.__setattr__(
            self,
            "verdict",
            _enum(
                self.verdict,
                field_name="verdict",
                allowed=AGENT_SKILL_AGGREGATE_VERDICTS_V1,
            ),
        )
        object.__setattr__(
            self,
            "generated_at",
            _finite(self.generated_at, field_name="generated_at"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_AGGREGATE_EVALUATION_RECEIPT_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "operation_id": self.operation_id,
            "request_hash": self.request_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "aggregation_policy_id": self.aggregation_policy_id,
            "aggregation_policy_hash": self.aggregation_policy_hash,
            "consumed_evaluation_receipt_hashes": list(
                self.consumed_evaluation_receipt_hashes
            ),
            "ignored_evaluation_receipt_hashes": list(
                self.ignored_evaluation_receipt_hashes
            ),
            "missing_evaluator_ids": list(self.missing_evaluator_ids),
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "generated_at": self.generated_at,
        }
        if include_hash:
            payload["aggregate_evaluation_receipt_hash"] = (
                self.semantic_content_hash
            )
        return payload


@dataclass(frozen=True)
class EvaluationReceiptSetAssessment:
    """Non-aggregating validation of evaluator receipts for one operation."""

    accepted_receipt_hashes: tuple[str, ...]
    ignored_receipt_reasons: Mapping[str, str]
    missing_required_evaluator_ids: tuple[str, ...]
    evidence_domain_evaluator_ids: Mapping[str, tuple[str, ...]]
    has_verdict_disagreement: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_receipt_hashes",
            _unique_sorted(
                self.accepted_receipt_hashes,
                field_name="accepted_receipt_hashes",
            ),
        )
        object.__setattr__(
            self,
            "missing_required_evaluator_ids",
            _unique_sorted(
                self.missing_required_evaluator_ids,
                field_name="missing_required_evaluator_ids",
            ),
        )
        if not isinstance(self.ignored_receipt_reasons, Mapping):
            raise TypeError("ignored_receipt_reasons must be a mapping.")
        ignored = {
            _required(key, field_name="ignored_receipt_hash"): _required(
                value,
                field_name="ignored_receipt_reason",
            )
            for key, value in sorted(self.ignored_receipt_reasons.items())
        }
        object.__setattr__(
            self,
            "ignored_receipt_reasons",
            MappingProxyType(ignored),
        )
        if not isinstance(self.evidence_domain_evaluator_ids, Mapping):
            raise TypeError("evidence_domain_evaluator_ids must be a mapping.")
        domains = {
            _required(key, field_name="evidence_domain_id"): _unique_sorted(
                value,
                field_name="evidence_domain_evaluator_ids",
            )
            for key, value in sorted(self.evidence_domain_evaluator_ids.items())
        }
        object.__setattr__(
            self,
            "evidence_domain_evaluator_ids",
            MappingProxyType(domains),
        )
        if not isinstance(self.has_verdict_disagreement, bool):
            raise TypeError("has_verdict_disagreement must be a boolean.")

    @property
    def complete_for_required(self) -> bool:
        return not self.missing_required_evaluator_ids

    @property
    def independent_evidence_domain_count(self) -> int:
        return len(self.evidence_domain_evaluator_ids)


def assess_evaluation_receipts(
    *,
    plan: SkillEvaluationPlan,
    receipts: tuple[SkillEvaluationReceipt, ...],
    operation_id: str,
    request_hash: str,
    evaluation_deadline_at: float,
) -> EvaluationReceiptSetAssessment:
    """Validate receipt bindings/deadlines without choosing an aggregation rule."""

    if not isinstance(plan, SkillEvaluationPlan):
        raise TypeError("plan must be a SkillEvaluationPlan.")
    expected_operation = _required(operation_id, field_name="operation_id")
    expected_request = _required(request_hash, field_name="request_hash")
    deadline = _finite(evaluation_deadline_at, field_name="evaluation_deadline_at")
    receipt_values = tuple(receipts)
    if any(not isinstance(item, SkillEvaluationReceipt) for item in receipt_values):
        raise TypeError("receipts must contain SkillEvaluationReceipt values.")

    accepted_by_evaluator: dict[str, SkillEvaluationReceipt] = {}
    ignored: dict[str, str] = {}
    for receipt in sorted(
        receipt_values,
        key=lambda item: (item.evaluated_at, item.semantic_content_hash),
    ):
        receipt_hash = receipt.semantic_content_hash
        try:
            spec = plan.evaluator(
                receipt.evaluator_id,
                receipt.evaluator_version,
            )
        except KeyError:
            ignored[receipt_hash] = "evaluator_not_in_plan"
            continue
        checks = (
            (receipt.operation_id == expected_operation, "operation_id_mismatch"),
            (receipt.request_hash == expected_request, "request_hash_mismatch"),
            (
                receipt.skill_definition_hash == plan.skill_definition_hash,
                "skill_definition_hash_mismatch",
            ),
            (receipt.route_hash == plan.route_hash, "route_hash_mismatch"),
            (
                receipt.deployment_binding_hash
                == plan.deployment_binding_hash,
                "deployment_binding_hash_mismatch",
            ),
            (
                receipt.evaluation_plan_hash == plan.semantic_content_hash,
                "evaluation_plan_hash_mismatch",
            ),
            (
                receipt.evaluator_hash == spec.semantic_content_hash,
                "evaluator_hash_mismatch",
            ),
            (
                receipt.evaluator_kind == spec.evaluator_kind,
                "evaluator_kind_mismatch",
            ),
            (
                receipt.evidence_domain_id == spec.evidence_domain_id,
                "evidence_domain_mismatch",
            ),
        )
        mismatch = next((reason for matches, reason in checks if not matches), None)
        if mismatch is not None:
            ignored[receipt_hash] = mismatch
            continue
        if receipt.evaluated_at >= deadline:
            ignored[receipt_hash] = "evaluation_receipt_late"
            continue
        if receipt.evaluator_id in accepted_by_evaluator:
            ignored[receipt_hash] = "duplicate_evaluator_receipt"
            continue
        accepted_by_evaluator[receipt.evaluator_id] = receipt

    missing_required = tuple(
        evaluator_id
        for evaluator_id in plan.required_evaluator_ids
        if evaluator_id not in accepted_by_evaluator
    )
    domains: dict[str, list[str]] = {}
    verdicts: set[str] = set()
    for receipt in accepted_by_evaluator.values():
        domains.setdefault(receipt.evidence_domain_id, []).append(
            receipt.evaluator_id
        )
        if receipt.verdict in {"failed", "succeeded"}:
            verdicts.add(receipt.verdict)
    return EvaluationReceiptSetAssessment(
        accepted_receipt_hashes=tuple(
            item.semantic_content_hash for item in accepted_by_evaluator.values()
        ),
        ignored_receipt_reasons=ignored,
        missing_required_evaluator_ids=missing_required,
        evidence_domain_evaluator_ids={
            key: tuple(values) for key, values in domains.items()
        },
        has_verdict_disagreement=len(verdicts) > 1,
    )


__all__ = [
    "AGENT_SKILL_AGGREGATE_EVALUATION_RECEIPT_HASH_DOMAIN_V1",
    "AGENT_SKILL_AGGREGATE_VERDICTS_V1",
    "AGENT_SKILL_DISAGREEMENT_POLICIES_V1",
    "AGENT_SKILL_EVALUATION_PLAN_HASH_DOMAIN_V1",
    "AGENT_SKILL_EVALUATION_RECEIPT_HASH_DOMAIN_V1",
    "AGENT_SKILL_EVALUATION_VERDICTS_V1",
    "AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1",
    "AGENT_SKILL_EVALUATOR_KINDS_V1",
    "AGENT_SKILL_EVALUATOR_LOCATIONS_V1",
    "AGENT_SKILL_EVALUATOR_ROLES_V1",
    "AGENT_SKILL_EVALUATOR_SPEC_HASH_DOMAIN_V1",
    "AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1",
    "EvaluationReceiptSetAssessment",
    "SkillAggregateEvaluationReceipt",
    "SkillEvaluationPlan",
    "SkillEvaluationReceipt",
    "SkillEvaluatorSpec",
    "assess_evaluation_receipts",
]
