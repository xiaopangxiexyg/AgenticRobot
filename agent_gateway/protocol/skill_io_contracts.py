"""Generic inline and finite-Artifact I/O contracts for every Skill."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from dataset_contracts.canonical_encoding import canonical_content_hash

from ._contract_validation import (
    enum_value,
    nonnegative_integer,
    optional_text,
    positive_integer,
    required_text,
    slot_identifier,
    unique_sorted_text,
)
from .artifact_contracts import AGENT_ARTIFACT_KINDS_V2
from .version import AGENT_SKILL_PROTOCOL_VERSION_V2


AGENT_SKILL_IO_CONTRACT_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.io_contract.v2"
)
AGENT_SKILL_IO_CATALOG_HASH_DOMAIN_V2 = "dexteleop.agent_skill.io_catalog.v2"
AGENT_SKILL_ARTIFACT_SLOT_DIRECTIONS_V2 = ("input", "output")
AGENT_SKILL_ARTIFACT_RETENTION_POLICIES_V2 = (
    "operation_bound",
    "handoff_bound",
)
AGENT_SKILL_ARTIFACT_SENSITIVITY_CLASSES_V2 = (
    "normal",
    "sensitive",
    "restricted",
)

_SKILL_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
)
_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*$")


def _skill_id(value: object) -> str:
    normalized = required_text(value, field_name="skill_id")
    if not _SKILL_ID_PATTERN.fullmatch(normalized):
        raise ValueError("skill_id must be a lowercase namespaced identifier.")
    return normalized


def _version(value: object, *, field_name: str) -> str:
    normalized = required_text(value, field_name=field_name)
    if not _VERSION_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a positive integer string.")
    return normalized


@dataclass(frozen=True)
class SkillArtifactSlotSpec:
    """One bounded semantic Artifact slot independent of transport location."""

    slot_id: str
    direction: str
    min_count: int
    max_count: int
    allowed_artifact_kinds: tuple[str, ...]
    allowed_media_types: tuple[str, ...]
    allowed_schema_ids: tuple[str, ...]
    allowed_encodings: tuple[str, ...]
    allowed_compressions: tuple[str, ...]
    max_artifact_bytes: int
    max_total_bytes: int
    inline_threshold_bytes: int
    retention_policy: str
    sensitivity_class: str = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "slot_id",
            slot_identifier(self.slot_id, field_name="slot_id"),
        )
        object.__setattr__(
            self,
            "direction",
            enum_value(
                self.direction,
                field_name="direction",
                allowed=AGENT_SKILL_ARTIFACT_SLOT_DIRECTIONS_V2,
            ),
        )
        minimum = nonnegative_integer(self.min_count, field_name="min_count")
        maximum = positive_integer(self.max_count, field_name="max_count")
        if minimum > maximum:
            raise ValueError("min_count cannot exceed max_count.")
        object.__setattr__(self, "min_count", minimum)
        object.__setattr__(self, "max_count", maximum)
        kinds = unique_sorted_text(
            self.allowed_artifact_kinds,
            field_name="allowed_artifact_kinds",
            allow_empty=False,
        )
        unsupported = sorted(set(kinds) - set(AGENT_ARTIFACT_KINDS_V2))
        if unsupported:
            raise ValueError(f"Unsupported Artifact kinds: {unsupported}.")
        object.__setattr__(self, "allowed_artifact_kinds", kinds)
        for field_name in (
            "allowed_media_types",
            "allowed_encodings",
            "allowed_compressions",
        ):
            object.__setattr__(
                self,
                field_name,
                unique_sorted_text(
                    getattr(self, field_name),
                    field_name=field_name,
                    allow_empty=False,
                ),
            )
        object.__setattr__(
            self,
            "allowed_schema_ids",
            unique_sorted_text(
                self.allowed_schema_ids,
                field_name="allowed_schema_ids",
            ),
        )
        max_artifact = positive_integer(
            self.max_artifact_bytes,
            field_name="max_artifact_bytes",
        )
        max_total = positive_integer(
            self.max_total_bytes,
            field_name="max_total_bytes",
        )
        if max_total < max_artifact or max_total < max_artifact * minimum:
            raise ValueError(
                "max_total_bytes must cover one Artifact and the required count."
            )
        object.__setattr__(self, "max_artifact_bytes", max_artifact)
        object.__setattr__(self, "max_total_bytes", max_total)
        threshold = nonnegative_integer(
            self.inline_threshold_bytes,
            field_name="inline_threshold_bytes",
        )
        if threshold > max_artifact:
            raise ValueError("inline_threshold_bytes cannot exceed max_artifact_bytes.")
        object.__setattr__(self, "inline_threshold_bytes", threshold)
        object.__setattr__(
            self,
            "retention_policy",
            enum_value(
                self.retention_policy,
                field_name="retention_policy",
                allowed=AGENT_SKILL_ARTIFACT_RETENTION_POLICIES_V2,
            ),
        )
        expected_retention = (
            "operation_bound" if self.direction == "input" else "handoff_bound"
        )
        if self.retention_policy != expected_retention:
            raise ValueError(
                f"{self.direction} slots require {expected_retention!r} retention."
            )
        object.__setattr__(
            self,
            "sensitivity_class",
            enum_value(
                self.sensitivity_class,
                field_name="sensitivity_class",
                allowed=AGENT_SKILL_ARTIFACT_SENSITIVITY_CLASSES_V2,
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "direction": self.direction,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "allowed_artifact_kinds": list(self.allowed_artifact_kinds),
            "allowed_media_types": list(self.allowed_media_types),
            "allowed_schema_ids": list(self.allowed_schema_ids),
            "allowed_encodings": list(self.allowed_encodings),
            "allowed_compressions": list(self.allowed_compressions),
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_total_bytes": self.max_total_bytes,
            "inline_threshold_bytes": self.inline_threshold_bytes,
            "retention_policy": self.retention_policy,
            "sensitivity_class": self.sensitivity_class,
        }


@dataclass(frozen=True)
class SkillIOContract:
    """Complete bounded I/O surface for one semantic Skill definition."""

    skill_id: str
    skill_version: str
    skill_definition_hash: str
    parameter_schema_hash: str
    execution_result_schema_hash: str
    task_evaluation_schema_hash: str
    input_artifact_slots: tuple[SkillArtifactSlotSpec, ...] = ()
    output_artifact_slots: tuple[SkillArtifactSlotSpec, ...] = ()
    progress_schema_hash: str | None = None
    max_inline_parameter_bytes: int = 65_536
    max_inline_result_bytes: int = 65_536
    max_progress_summary_bytes: int = 16_384
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        object.__setattr__(self, "skill_id", _skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            _version(self.skill_version, field_name="skill_version"),
        )
        for field_name in (
            "skill_definition_hash",
            "parameter_schema_hash",
            "execution_result_schema_hash",
            "task_evaluation_schema_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "progress_schema_hash",
            optional_text(
                self.progress_schema_hash,
                field_name="progress_schema_hash",
            ),
        )
        all_slot_ids: set[str] = set()
        for field_name, expected_direction in (
            ("input_artifact_slots", "input"),
            ("output_artifact_slots", "output"),
        ):
            slots = tuple(getattr(self, field_name))
            if any(not isinstance(slot, SkillArtifactSlotSpec) for slot in slots):
                raise TypeError(f"{field_name} must contain SkillArtifactSlotSpec.")
            if any(slot.direction != expected_direction for slot in slots):
                raise ValueError(
                    f"{field_name} contains a slot with the wrong direction."
                )
            ids = tuple(slot.slot_id for slot in slots)
            if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
                raise ValueError(f"{field_name} must be uniquely sorted by slot_id.")
            overlap = all_slot_ids.intersection(ids)
            if overlap:
                raise ValueError(f"Artifact slot IDs must be globally unique: {overlap}.")
            all_slot_ids.update(ids)
            object.__setattr__(self, field_name, slots)
        for field_name in (
            "max_inline_parameter_bytes",
            "max_inline_result_bytes",
            "max_progress_summary_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                positive_integer(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_IO_CONTRACT_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "execution_result_schema_hash": self.execution_result_schema_hash,
            "task_evaluation_schema_hash": self.task_evaluation_schema_hash,
            "progress_schema_hash": self.progress_schema_hash,
            "max_inline_parameter_bytes": self.max_inline_parameter_bytes,
            "max_inline_result_bytes": self.max_inline_result_bytes,
            "max_progress_summary_bytes": self.max_progress_summary_bytes,
            "input_artifact_slots": [
                slot.to_wire() for slot in self.input_artifact_slots
            ],
            "output_artifact_slots": [
                slot.to_wire() for slot in self.output_artifact_slots
            ],
        }
        if include_hash:
            payload["skill_io_contract_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillIOCatalog:
    """One exact I/O contract for every advertised Skill."""

    catalog_id: str
    catalog_version: str
    io_contracts: tuple[SkillIOContract, ...]
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        object.__setattr__(
            self,
            "catalog_id",
            required_text(self.catalog_id, field_name="catalog_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            _version(self.catalog_version, field_name="catalog_version"),
        )
        contracts = tuple(self.io_contracts)
        if any(not isinstance(item, SkillIOContract) for item in contracts):
            raise TypeError("io_contracts must contain SkillIOContract values.")
        keys = tuple((item.skill_id, item.skill_version) for item in contracts)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError(
                "io_contracts must be uniquely sorted by Skill identity."
            )
        object.__setattr__(self, "io_contracts", contracts)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_IO_CATALOG_HASH_DOMAIN_V2,
        )

    def contract(self, skill_id: str, skill_version: str) -> SkillIOContract:
        key = (_skill_id(skill_id), _version(
            skill_version,
            field_name="skill_version",
        ))
        for contract in self.io_contracts:
            if (contract.skill_id, contract.skill_version) == key:
                return contract
        raise KeyError(f"Skill I/O contract is not advertised: {key!r}.")

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "io_contracts": [item.to_wire() for item in self.io_contracts],
        }
        if include_hash:
            payload["skill_io_catalog_hash"] = self.semantic_content_hash
        return payload


def inline_only_skill_io_contract(
    *,
    skill_id: str,
    skill_version: str,
    skill_definition_hash: str,
    parameter_schema_hash: str,
    execution_result_schema_hash: str,
    task_evaluation_schema_hash: str,
) -> SkillIOContract:
    """Build the explicit empty-Artifact contract used by existing Skills."""

    return SkillIOContract(
        skill_id=skill_id,
        skill_version=skill_version,
        skill_definition_hash=skill_definition_hash,
        parameter_schema_hash=parameter_schema_hash,
        execution_result_schema_hash=execution_result_schema_hash,
        task_evaluation_schema_hash=task_evaluation_schema_hash,
    )


def inline_only_skill_io_catalog_for_routes(
    routes: Iterable[object],
    *,
    catalog_id: str = "dexteleop.agent_skill_io",
    catalog_version: str = "1",
) -> SkillIOCatalog:
    """Derive the explicit no-Artifact catalog for existing inline Routes."""

    by_key: dict[tuple[str, str], SkillIOContract] = {}
    for route in routes:
        contract = inline_only_skill_io_contract(
            skill_id=getattr(route, "skill_id"),
            skill_version=getattr(route, "skill_version"),
            skill_definition_hash=getattr(route, "skill_definition_hash"),
            parameter_schema_hash=getattr(route, "parameter_schema_hash"),
            execution_result_schema_hash=getattr(
                route,
                "execution_result_schema_hash",
            ),
            task_evaluation_schema_hash=getattr(
                route,
                "task_evaluation_schema_hash",
            ),
        )
        key = (contract.skill_id, contract.skill_version)
        existing = by_key.get(key)
        if existing is not None and existing != contract:
            raise ValueError(
                "Routes for one Skill disagree on the inline I/O contract."
            )
        by_key[key] = contract
    return SkillIOCatalog(
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        io_contracts=tuple(by_key[key] for key in sorted(by_key)),
    )


def agent_skill_io_wire_schemas_v2() -> dict[str, dict[str, Any]]:
    """Return strict portable JSON schemas for Skill I/O discovery."""

    nullable_string = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "null"},
        ]
    }
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    slot = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
            "direction": {
                "type": "string",
                "enum": list(AGENT_SKILL_ARTIFACT_SLOT_DIRECTIONS_V2),
            },
            "min_count": {"type": "integer", "minimum": 0},
            "max_count": {"type": "integer", "minimum": 1},
            "allowed_artifact_kinds": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(AGENT_ARTIFACT_KINDS_V2)},
            },
            "allowed_media_types": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "allowed_schema_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "allowed_encodings": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "allowed_compressions": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "max_artifact_bytes": {"type": "integer", "minimum": 1},
            "max_total_bytes": {"type": "integer", "minimum": 1},
            "inline_threshold_bytes": {"type": "integer", "minimum": 0},
            "retention_policy": {
                "type": "string",
                "enum": list(AGENT_SKILL_ARTIFACT_RETENTION_POLICIES_V2),
            },
            "sensitivity_class": {
                "type": "string",
                "enum": list(AGENT_SKILL_ARTIFACT_SENSITIVITY_CLASSES_V2),
            },
        },
        "required": [
            "slot_id",
            "direction",
            "min_count",
            "max_count",
            "allowed_artifact_kinds",
            "allowed_media_types",
            "allowed_schema_ids",
            "allowed_encodings",
            "allowed_compressions",
            "max_artifact_bytes",
            "max_total_bytes",
            "inline_threshold_bytes",
            "retention_policy",
            "sensitivity_class",
        ],
    }
    contract = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "skill_id": {"type": "string", "minLength": 1},
            "skill_version": {"type": "string", "pattern": "^[1-9][0-9]*$"},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "parameter_schema_hash": {"type": "string", "minLength": 1},
            "execution_result_schema_hash": {"type": "string", "minLength": 1},
            "task_evaluation_schema_hash": {"type": "string", "minLength": 1},
            "progress_schema_hash": nullable_string,
            "max_inline_parameter_bytes": {"type": "integer", "minimum": 1},
            "max_inline_result_bytes": {"type": "integer", "minimum": 1},
            "max_progress_summary_bytes": {"type": "integer", "minimum": 1},
            "input_artifact_slots": {
                "type": "array",
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/SkillArtifactSlotSpec"},
            },
            "output_artifact_slots": {
                "type": "array",
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/SkillArtifactSlotSpec"},
            },
            "skill_io_contract_hash": digest,
        },
        "required": [
            "protocol_version",
            "skill_id",
            "skill_version",
            "skill_definition_hash",
            "parameter_schema_hash",
            "execution_result_schema_hash",
            "task_evaluation_schema_hash",
            "progress_schema_hash",
            "max_inline_parameter_bytes",
            "max_inline_result_bytes",
            "max_progress_summary_bytes",
            "input_artifact_slots",
            "output_artifact_slots",
            "skill_io_contract_hash",
        ],
    }
    catalog = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "catalog_id": {"type": "string", "minLength": 1},
            "catalog_version": {"type": "string", "pattern": "^[1-9][0-9]*$"},
            "io_contracts": {
                "type": "array",
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/SkillIOContract"},
            },
            "skill_io_catalog_hash": digest,
        },
        "required": [
            "protocol_version",
            "catalog_id",
            "catalog_version",
            "io_contracts",
            "skill_io_catalog_hash",
        ],
        "$defs": {
            "SkillArtifactSlotSpec": slot,
            "SkillIOContract": {
                **contract,
                "$defs": {"SkillArtifactSlotSpec": slot},
            },
        },
    }
    return {
        "skill_io_contract": {
            **contract,
            "$defs": {"SkillArtifactSlotSpec": slot},
        },
        "skill_io_catalog": catalog,
    }


__all__ = [
    "SkillArtifactSlotSpec",
    "SkillIOCatalog",
    "SkillIOContract",
    "agent_skill_io_wire_schemas_v2",
    "inline_only_skill_io_catalog_for_routes",
    "inline_only_skill_io_contract",
]
