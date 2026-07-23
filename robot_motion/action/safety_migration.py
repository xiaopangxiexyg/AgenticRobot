"""Pure, evidence-bearing gates for safety-owner migration.

This module deliberately does not inspect runtime objects or infer readiness
from code presence.  A profile owner must state each rule's readiness at each
migration gate and cite immutable evidence.  In particular, being ready for a
read-only M5.8-a seam inventory does not imply readiness for real mediation or
shared authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


SAFETY_MIGRATION_MATRIX_HASH_DOMAIN_V1 = "dexteleop.safety_migration_matrix.v1"
SAFETY_MIGRATION_GATE_STAGES_V1 = (
    "m5_8a_inventory",
    "m5_8b_legacy_mediation",
    "m6_shared_authority",
)
SAFETY_MIGRATION_GATE_STATUSES_V1 = frozenset(
    {"ready", "blocked", "not_applicable"}
)
SAFETY_MIGRATION_DISPOSITIONS_V1 = frozenset(
    {
        "preserved_source_validation",
        "moved_shared_safety",
        "moved_reference_planner",
        "moved_backend_guard",
        "reimplemented",
        "intentionally_removed",
        "not_applicable",
    }
)


def _non_empty_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} entries must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique.")
    return normalized


@dataclass(frozen=True)
class SafetyMigrationGateAssessment:
    gate_stage: str
    status: str
    evidence_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stage = str(self.gate_stage).strip()
        status = str(self.status).strip()
        if stage not in SAFETY_MIGRATION_GATE_STAGES_V1:
            raise ValueError(f"Unsupported safety migration gate {stage!r}.")
        if status not in SAFETY_MIGRATION_GATE_STATUSES_V1:
            raise ValueError(f"Unsupported safety migration gate status {status!r}.")
        evidence = _non_empty_tuple(self.evidence_ids, name="evidence_ids")
        reasons = _non_empty_tuple(self.reasons, name="reasons")
        if status == "ready" and not evidence:
            raise ValueError("A ready safety migration gate requires evidence.")
        if status == "ready" and reasons:
            raise ValueError("A ready safety migration gate cannot carry blocking reasons.")
        if status == "blocked" and not reasons:
            raise ValueError("A blocked safety migration gate requires reasons.")
        if status == "not_applicable" and not evidence:
            raise ValueError("A not-applicable gate requires profile exclusion evidence.")
        object.__setattr__(self, "gate_stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class SafetyMigrationRuleAssessment:
    rule_id: str
    current_owner: str
    target_disposition: str | None
    gate_assessments: tuple[SafetyMigrationGateAssessment, ...]
    disposition_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rule_id = str(self.rule_id).strip()
        current_owner = str(self.current_owner).strip()
        if not rule_id or not current_owner:
            raise ValueError("rule_id/current_owner must be non-empty.")
        disposition = (
            None
            if self.target_disposition is None
            else str(self.target_disposition).strip()
        )
        if disposition is not None and disposition not in SAFETY_MIGRATION_DISPOSITIONS_V1:
            raise ValueError(f"Unsupported safety migration disposition {disposition!r}.")
        evidence = _non_empty_tuple(
            self.disposition_evidence_ids,
            name="disposition_evidence_ids",
        )
        if disposition is not None and not evidence:
            raise ValueError("A final target disposition requires evidence.")
        gates = tuple(self.gate_assessments)
        by_stage = {assessment.gate_stage: assessment for assessment in gates}
        if len(by_stage) != len(gates):
            raise ValueError("A rule cannot contain duplicate gate assessments.")
        missing = set(SAFETY_MIGRATION_GATE_STAGES_V1) - set(by_stage)
        extra = set(by_stage) - set(SAFETY_MIGRATION_GATE_STAGES_V1)
        if missing or extra:
            raise ValueError(
                "Every safety rule must assess all migration gates; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )
        m6 = by_stage["m6_shared_authority"]
        if m6.status == "ready" and disposition is None:
            raise ValueError("M6-ready safety rules require a final target disposition.")
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "current_owner", current_owner)
        object.__setattr__(self, "target_disposition", disposition)
        object.__setattr__(
            self,
            "gate_assessments",
            tuple(by_stage[stage] for stage in SAFETY_MIGRATION_GATE_STAGES_V1),
        )
        object.__setattr__(self, "disposition_evidence_ids", evidence)

    def gate(self, stage: str) -> SafetyMigrationGateAssessment:
        requested = str(stage).strip()
        for assessment in self.gate_assessments:
            if assessment.gate_stage == requested:
                return assessment
        raise ValueError(f"Unsupported safety migration gate {requested!r}.")


@dataclass(frozen=True)
class SafetyMigrationProfileMatrix:
    profile_id: str
    matrix_version: str
    rules: tuple[SafetyMigrationRuleAssessment, ...]

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        version = str(self.matrix_version).strip()
        if not profile_id or not version:
            raise ValueError("profile_id/matrix_version must be non-empty.")
        rules = tuple(self.rules)
        if not rules:
            raise ValueError("A safety migration profile must contain at least one rule.")
        rule_ids = tuple(rule.rule_id for rule in rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("Safety migration rule IDs must be unique per profile.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "matrix_version", version)
        object.__setattr__(self, "rules", tuple(sorted(rules, key=lambda rule: rule.rule_id)))

    @property
    def semantic_content_hash(self) -> str:
        rules_payload = []
        for rule in self.rules:
            rules_payload.append(
                {
                    "rule_id": rule.rule_id,
                    "current_owner": rule.current_owner,
                    "target_disposition": rule.target_disposition,
                    "disposition_evidence_ids": rule.disposition_evidence_ids,
                    "gate_assessments": tuple(
                        {
                            "gate_stage": gate.gate_stage,
                            "status": gate.status,
                            "evidence_ids": gate.evidence_ids,
                            "reasons": gate.reasons,
                        }
                        for gate in rule.gate_assessments
                    ),
                }
            )
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "matrix_version": self.matrix_version,
                "rules": tuple(rules_payload),
            },
            domain=SAFETY_MIGRATION_MATRIX_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class SafetyMigrationGateDecision:
    profile_id: str
    matrix_hash: str
    gate_stage: str
    ready: bool
    blocking_rule_ids: tuple[str, ...]
    not_applicable_rule_ids: tuple[str, ...]


def assess_safety_migration_gate(
    matrix: SafetyMigrationProfileMatrix,
    *,
    gate_stage: str,
) -> SafetyMigrationGateDecision:
    if not isinstance(matrix, SafetyMigrationProfileMatrix):
        raise TypeError("matrix must be a SafetyMigrationProfileMatrix.")
    stage = str(gate_stage).strip()
    if stage not in SAFETY_MIGRATION_GATE_STAGES_V1:
        raise ValueError(f"Unsupported safety migration gate {stage!r}.")
    blocking = tuple(
        rule.rule_id for rule in matrix.rules if rule.gate(stage).status == "blocked"
    )
    not_applicable = tuple(
        rule.rule_id
        for rule in matrix.rules
        if rule.gate(stage).status == "not_applicable"
    )
    return SafetyMigrationGateDecision(
        profile_id=matrix.profile_id,
        matrix_hash=matrix.semantic_content_hash,
        gate_stage=stage,
        ready=not blocking,
        blocking_rule_ids=blocking,
        not_applicable_rule_ids=not_applicable,
    )


__all__ = [
    "SAFETY_MIGRATION_DISPOSITIONS_V1",
    "SAFETY_MIGRATION_GATE_STAGES_V1",
    "SAFETY_MIGRATION_GATE_STATUSES_V1",
    "SAFETY_MIGRATION_MATRIX_HASH_DOMAIN_V1",
    "SafetyMigrationGateAssessment",
    "SafetyMigrationGateDecision",
    "SafetyMigrationProfileMatrix",
    "SafetyMigrationRuleAssessment",
    "assess_safety_migration_gate",
]
