"""Pure contracts for non-destructive Progress-to-Skill projection.

The capture-time Progress profile describes one task topology.  A training
Skill is intentionally independent from that topology so equivalent work can
be selected from different task profiles without rewriting either dataset.

This module only defines immutable semantic contracts.  Source-profile
validation and sample-interval projection belong to :mod:`dataset_interface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


SKILL_TRAINING_CONTRACT_SCHEMA_V1 = (
    "dexteleop.skill_training_contract.v1"
)
PROGRESS_SKILL_PROJECTION_SCHEMA_V1 = (
    "dexteleop.progress_skill_projection.v1"
)
SKILL_PROJECTION_INTERVAL_SCHEMA_V1 = (
    "dexteleop.skill_projection_interval.v1"
)
SKILL_PROJECTION_RECEIPT_SCHEMA_V1 = (
    "dexteleop.skill_projection_receipt.v1"
)

SKILL_PROJECTION_KINDS_V1 = frozenset(
    {"EXACT_STEP", "CONDITIONED_STEP", "STAGE_SUBSET"}
)
SKILL_PROJECTION_TRAINING_DOMAINS_V1 = frozenset(
    {"NOMINAL", "FAILURE"}
)

_SKILL_HASH_DOMAIN = "dexteleop.skill_training_contract.semantic.v1"
_PROJECTION_HASH_DOMAIN = (
    "dexteleop.progress_skill_projection.semantic.v1"
)
_RECEIPT_HASH_DOMAIN = "dexteleop.skill_projection_receipt.semantic.v1"


class ProgressSkillProjectionContractError(ValueError):
    """Raised when a Skill or projection contract is ambiguous."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgressSkillProjectionContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ProgressSkillProjectionContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _unique_strings(
    values: object,
    *,
    name: str,
    allow_empty: bool,
    sort_values: bool,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ProgressSkillProjectionContractError(
            f"{name} must be a tuple or list."
        )
    result = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not result:
        raise ProgressSkillProjectionContractError(
            f"{name} must not be empty."
        )
    if len(result) != len(set(result)):
        raise ProgressSkillProjectionContractError(
            f"{name} must contain unique values."
        )
    return tuple(sorted(result)) if sort_values else result


@dataclass(frozen=True)
class SkillTrainingContractV1:
    """Minimal task-topology-independent identity for one trainable Skill.

    Predicates are stable semantic identifiers, not free-form expressions.
    Context identifiers name inputs that a policy may condition on; they do
    not become part of the Skill identity's success definition.
    """

    skill_id: str
    skill_version: str
    semantic_type: str
    controlled_component_ids: tuple[str, ...]
    entry_predicate_ids: tuple[str, ...]
    success_predicate_ids: tuple[str, ...]
    conditionable_context_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=SKILL_TRAINING_CONTRACT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in ("skill_id", "skill_version", "semantic_type"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        for name, allow_empty in (
            ("controlled_component_ids", False),
            ("entry_predicate_ids", False),
            ("success_predicate_ids", False),
            ("conditionable_context_ids", True),
        ):
            object.__setattr__(
                self,
                name,
                _unique_strings(
                    getattr(self, name),
                    name=name,
                    allow_empty=allow_empty,
                    sort_values=True,
                ),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_SKILL_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "semantic_type": self.semantic_type,
            "controlled_component_ids": self.controlled_component_ids,
            "entry_predicate_ids": self.entry_predicate_ids,
            "success_predicate_ids": self.success_predicate_ids,
            "conditionable_context_ids": self.conditionable_context_ids,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "SkillTrainingContractV1":
        if payload.get("schema") != SKILL_TRAINING_CONTRACT_SCHEMA_V1:
            raise ProgressSkillProjectionContractError(
                "Unsupported Skill training contract schema."
            )
        result = cls(
            skill_id=payload.get("skill_id", ""),
            skill_version=payload.get("skill_version", ""),
            semantic_type=payload.get("semantic_type", ""),
            controlled_component_ids=tuple(
                payload.get("controlled_component_ids", ())
            ),
            entry_predicate_ids=tuple(
                payload.get("entry_predicate_ids", ())
            ),
            success_predicate_ids=tuple(
                payload.get("success_predicate_ids", ())
            ),
            conditionable_context_ids=tuple(
                payload.get("conditionable_context_ids", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressSkillProjectionContractError(
                "Skill training contract hash mismatch."
            )
        return result


@dataclass(frozen=True)
class ProgressSkillProjectionV1:
    """Explicitly map one source Progress Step (or stage subset) to a Skill."""

    projection_id: str
    projection_version: str
    source_progress_profile_hash: str
    source_stage_profile_hash: str
    source_progress_step_id: str
    source_stage_ids: tuple[str, ...]
    projection_kind: str
    skill_training_contract_hash: str
    projected_component_ids: tuple[str, ...]
    asserted_entry_predicate_ids: tuple[str, ...]
    asserted_success_predicate_ids: tuple[str, ...]
    conditioned_context_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=PROGRESS_SKILL_PROJECTION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "projection_version",
            "source_progress_step_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        for name in (
            "source_progress_profile_hash",
            "source_stage_profile_hash",
            "skill_training_contract_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        kind = _required(
            self.projection_kind,
            name="projection_kind",
        ).upper()
        if kind not in SKILL_PROJECTION_KINDS_V1:
            raise ProgressSkillProjectionContractError(
                f"Unsupported projection_kind {kind!r}."
            )
        object.__setattr__(self, "projection_kind", kind)
        object.__setattr__(
            self,
            "source_stage_ids",
            _unique_strings(
                self.source_stage_ids,
                name="source_stage_ids",
                allow_empty=False,
                sort_values=False,
            ),
        )
        for name in (
            "projected_component_ids",
            "asserted_entry_predicate_ids",
            "asserted_success_predicate_ids",
        ):
            object.__setattr__(
                self,
                name,
                _unique_strings(
                    getattr(self, name),
                    name=name,
                    allow_empty=False,
                    sort_values=True,
                ),
            )
        object.__setattr__(
            self,
            "conditioned_context_ids",
            _unique_strings(
                self.conditioned_context_ids,
                name="conditioned_context_ids",
                allow_empty=True,
                sort_values=True,
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_PROJECTION_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "source_progress_profile_hash": self.source_progress_profile_hash,
            "source_stage_profile_hash": self.source_stage_profile_hash,
            "source_progress_step_id": self.source_progress_step_id,
            "source_stage_ids": self.source_stage_ids,
            "projection_kind": self.projection_kind,
            "skill_training_contract_hash": (
                self.skill_training_contract_hash
            ),
            "projected_component_ids": self.projected_component_ids,
            "asserted_entry_predicate_ids": (
                self.asserted_entry_predicate_ids
            ),
            "asserted_success_predicate_ids": (
                self.asserted_success_predicate_ids
            ),
            "conditioned_context_ids": self.conditioned_context_ids,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressSkillProjectionV1":
        if payload.get("schema") != PROGRESS_SKILL_PROJECTION_SCHEMA_V1:
            raise ProgressSkillProjectionContractError(
                "Unsupported Progress-to-Skill projection schema."
            )
        result = cls(
            projection_id=payload.get("projection_id", ""),
            projection_version=payload.get("projection_version", ""),
            source_progress_profile_hash=payload.get(
                "source_progress_profile_hash", ""
            ),
            source_stage_profile_hash=payload.get(
                "source_stage_profile_hash", ""
            ),
            source_progress_step_id=payload.get(
                "source_progress_step_id", ""
            ),
            source_stage_ids=tuple(payload.get("source_stage_ids", ())),
            projection_kind=payload.get("projection_kind", ""),
            skill_training_contract_hash=payload.get(
                "skill_training_contract_hash", ""
            ),
            projected_component_ids=tuple(
                payload.get("projected_component_ids", ())
            ),
            asserted_entry_predicate_ids=tuple(
                payload.get("asserted_entry_predicate_ids", ())
            ),
            asserted_success_predicate_ids=tuple(
                payload.get("asserted_success_predicate_ids", ())
            ),
            conditioned_context_ids=tuple(
                payload.get("conditioned_context_ids", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressSkillProjectionContractError(
                "Progress-to-Skill projection hash mismatch."
            )
        return result


@dataclass(frozen=True)
class SkillProjectionIntervalV1:
    """One projected half-open source interval; sample payloads stay in place."""

    sample_begin: int
    sample_end: int
    training_domain: str
    source_attempt_id: str
    source_stage_ids: tuple[str, ...]
    source_occurrence_ids: tuple[str, ...]
    source_failure_fact_id: str | None = None
    schema: str = field(
        default=SKILL_PROJECTION_INTERVAL_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in ("sample_begin", "sample_end"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ProgressSkillProjectionContractError(
                    f"{name} must be a non-negative integer."
                )
        if self.sample_end <= self.sample_begin:
            raise ProgressSkillProjectionContractError(
                "Projected interval must be non-empty."
            )
        domain = _required(
            self.training_domain,
            name="training_domain",
        ).upper()
        if domain not in SKILL_PROJECTION_TRAINING_DOMAINS_V1:
            raise ProgressSkillProjectionContractError(
                f"Unsupported training_domain {domain!r}."
            )
        object.__setattr__(self, "training_domain", domain)
        object.__setattr__(
            self,
            "source_attempt_id",
            _required(self.source_attempt_id, name="source_attempt_id"),
        )
        for name in ("source_stage_ids", "source_occurrence_ids"):
            object.__setattr__(
                self,
                name,
                _unique_strings(
                    getattr(self, name),
                    name=name,
                    allow_empty=False,
                    sort_values=False,
                ),
            )
        if self.source_failure_fact_id is not None:
            object.__setattr__(
                self,
                "source_failure_fact_id",
                _required(
                    self.source_failure_fact_id,
                    name="source_failure_fact_id",
                ),
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sample_begin": self.sample_begin,
            "sample_end": self.sample_end,
            "training_domain": self.training_domain,
            "source_attempt_id": self.source_attempt_id,
            "source_stage_ids": self.source_stage_ids,
            "source_occurrence_ids": self.source_occurrence_ids,
            "source_failure_fact_id": self.source_failure_fact_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "SkillProjectionIntervalV1":
        if payload.get("schema") != SKILL_PROJECTION_INTERVAL_SCHEMA_V1:
            raise ProgressSkillProjectionContractError(
                "Unsupported Skill projection interval schema."
            )
        return cls(
            sample_begin=payload.get("sample_begin", -1),
            sample_end=payload.get("sample_end", -1),
            training_domain=payload.get("training_domain", ""),
            source_attempt_id=payload.get("source_attempt_id", ""),
            source_stage_ids=tuple(payload.get("source_stage_ids", ())),
            source_occurrence_ids=tuple(
                payload.get("source_occurrence_ids", ())
            ),
            source_failure_fact_id=payload.get("source_failure_fact_id"),
        )


@dataclass(frozen=True)
class SkillProjectionReceiptV1:
    """Deterministic provenance for one Episode/projection result."""

    receipt_id: str
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    source_stage_annotation_manifest_hash: str | None
    skill_training_contract_hash: str
    progress_skill_projection_hash: str
    intervals: tuple[SkillProjectionIntervalV1, ...]
    continuity_break_indices: tuple[int, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=SKILL_PROJECTION_RECEIPT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            _required(self.receipt_id, name="receipt_id"),
        )
        for name in (
            "episode_capture_identity_hash",
            "source_progress_manifest_hash",
            "skill_training_contract_hash",
            "progress_skill_projection_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        if self.source_stage_annotation_manifest_hash is not None:
            object.__setattr__(
                self,
                "source_stage_annotation_manifest_hash",
                _sha256(
                    self.source_stage_annotation_manifest_hash,
                    name="source_stage_annotation_manifest_hash",
                ),
            )
        intervals = tuple(self.intervals)
        if any(
            not isinstance(item, SkillProjectionIntervalV1)
            for item in intervals
        ):
            raise ProgressSkillProjectionContractError(
                "intervals must contain SkillProjectionIntervalV1 values."
            )
        ordered = tuple(
            sorted(
                intervals,
                key=lambda item: (item.sample_begin, item.sample_end),
            )
        )
        if ordered != intervals:
            raise ProgressSkillProjectionContractError(
                "intervals must be ordered by sample position."
            )
        object.__setattr__(self, "intervals", intervals)
        breaks = tuple(self.continuity_break_indices)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in breaks
        ):
            raise ProgressSkillProjectionContractError(
                "continuity_break_indices must be non-negative integers."
            )
        if breaks != tuple(sorted(set(breaks))):
            raise ProgressSkillProjectionContractError(
                "continuity_break_indices must be unique and sorted."
            )
        object.__setattr__(self, "continuity_break_indices", breaks)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_RECEIPT_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "source_progress_manifest_hash": (
                self.source_progress_manifest_hash
            ),
            "source_stage_annotation_manifest_hash": (
                self.source_stage_annotation_manifest_hash
            ),
            "skill_training_contract_hash": (
                self.skill_training_contract_hash
            ),
            "progress_skill_projection_hash": (
                self.progress_skill_projection_hash
            ),
            "intervals": tuple(item.to_payload() for item in self.intervals),
            "continuity_break_indices": self.continuity_break_indices,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "SkillProjectionReceiptV1":
        if payload.get("schema") != SKILL_PROJECTION_RECEIPT_SCHEMA_V1:
            raise ProgressSkillProjectionContractError(
                "Unsupported Skill projection receipt schema."
            )
        result = cls(
            receipt_id=payload.get("receipt_id", ""),
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash", ""
            ),
            source_progress_manifest_hash=payload.get(
                "source_progress_manifest_hash", ""
            ),
            source_stage_annotation_manifest_hash=payload.get(
                "source_stage_annotation_manifest_hash"
            ),
            skill_training_contract_hash=payload.get(
                "skill_training_contract_hash", ""
            ),
            progress_skill_projection_hash=payload.get(
                "progress_skill_projection_hash", ""
            ),
            intervals=tuple(
                SkillProjectionIntervalV1.from_payload(item)
                for item in payload.get("intervals", ())
            ),
            continuity_break_indices=tuple(
                payload.get("continuity_break_indices", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressSkillProjectionContractError(
                "Skill projection receipt hash mismatch."
            )
        return result


__all__ = [
    "PROGRESS_SKILL_PROJECTION_SCHEMA_V1",
    "SKILL_PROJECTION_INTERVAL_SCHEMA_V1",
    "SKILL_PROJECTION_KINDS_V1",
    "SKILL_PROJECTION_RECEIPT_SCHEMA_V1",
    "SKILL_PROJECTION_TRAINING_DOMAINS_V1",
    "SKILL_TRAINING_CONTRACT_SCHEMA_V1",
    "ProgressSkillProjectionContractError",
    "ProgressSkillProjectionV1",
    "SkillProjectionIntervalV1",
    "SkillProjectionReceiptV1",
    "SkillTrainingContractV1",
]
