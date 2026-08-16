"""Pure Skill definition, schema, route, and deployment-reference contracts.

This module is the canonical, side-effect-free vocabulary published by the
DexTeleop Agent Gateway.  A :class:`SkillDefinition` describes task semantics
and its intended target kind without exclusively binding execution there.  A
:class:`SkillExecutionRoute` describes one stable way to implement that skill.
A :class:`SkillRouteManifestEntry` joins the stable route to one immutable
deployment binding for a runtime session.

None of these records is an authority lease, safety decision, lower permit,
dispatch token, or evidence that a simulator or robot executed anything.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .session_contracts import (
    DEFAULT_AGENT_SKILL_RESOURCE_CLAIMS_V2,
    SkillResourceClaim,
)
from .skill_io_contracts import (
    SkillIOCatalog,
    inline_only_skill_io_catalog_for_routes,
    inline_only_skill_io_contract,
)
from .version import AGENT_SKILL_PROTOCOL_VERSION_V2

AGENT_SKILL_TARGET_KINDS_V1 = ("null", "real", "sim")
AGENT_SKILL_INTENDED_TARGET_KINDS_V1 = (
    "real",
    "sim",
    "target_agnostic",
)
AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1 = (
    "forbidden",
    "not_applicable",
    "review_required",
)
AGENT_SKILL_MODES_V1 = (
    "dry_run",
    "real_active",
    "real_observe",
    "sim_active",
)
AGENT_SKILL_PRIVILEGE_TIERS_V1 = (
    "t0_read_only",
    "t1_sim_or_no_motion",
    "t2_real_motion",
    "t3_lifecycle",
)
AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1 = (
    "lifecycle",
    "motion",
    "read_only",
)
AGENT_SKILL_CANCELLATION_POLICIES_V1 = (
    "cancel_only",
    "hold_on_cancel",
    "hold_on_cancel_disconnect_or_timeout",
    "hold_on_disconnect_or_timeout",
)
AGENT_SKILL_KINDS_V1 = (
    "inspection",
    "lifecycle",
    "motion",
    "policy_task",
)

PRIVILEGED_LIFECYCLE_SKILL_IDS_V1 = (
    "lifecycle.capture_runtime_home",
    "lifecycle.move_to_registered_home",
    "lifecycle.rearm_safety_latch",
    "lifecycle.resume_after_hold",
)
AMBIGUOUS_LIFECYCLE_SKILL_IDS_V1 = (
    "home",
    "home_v1",
    "rearm",
    "rearm_v1",
)

MAX_AGENT_SKILL_TIMEOUT_MS_V1 = 86_400_000
MAX_AGENT_SKILL_JSON_DEPTH_V1 = 16

AGENT_SKILL_DEFINITION_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.definition.v2"
)
AGENT_SKILL_DEFINITION_CATALOG_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.definition_catalog.v2"
)
AGENT_SKILL_SCHEMA_ENTRY_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.schema_entry.v2"
)
AGENT_SKILL_SCHEMA_CATALOG_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.schema_catalog.v2"
)
AGENT_SKILL_SCHEMA_DOCUMENT_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.schema_document.v2"
)
AGENT_SKILL_ROUTE_HASH_DOMAIN_V2 = "dexteleop.agent_skill.route.v2"
AGENT_SKILL_DEPLOYMENT_BINDING_REF_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.deployment_binding_ref.v2"
)
AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_REVIEW_REF_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.cross_target_deployment_review_ref.v2"
)
AGENT_SKILL_ROUTE_MANIFEST_ENTRY_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.route_manifest_entry.v2"
)
AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.route_manifest.v2"
)

_NAMESPACED_IDENTIFIER_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
)
_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*$")


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if normalized != value:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


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
    allowed: tuple[str, ...] | None = None,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list of strings.")
    normalized = tuple(
        _required(value, field_name=field_name) for value in values
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    if allowed is not None:
        unsupported = sorted(set(normalized) - set(allowed))
        if unsupported:
            raise ValueError(
                f"{field_name} contains unsupported values: {unsupported}."
            )
    return tuple(sorted(normalized))


def validate_skill_id(value: object) -> str:
    normalized = _required(value, field_name="skill_id")
    if normalized in AMBIGUOUS_LIFECYCLE_SKILL_IDS_V1:
        raise ValueError(
            "Ambiguous home/rearm skill IDs are forbidden; use a namespaced "
            "typed lifecycle skill ID."
        )
    if _NAMESPACED_IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "skill_id must be a lowercase namespaced identifier; version is "
            "carried separately by skill_version."
        )
    return normalized


def validate_route_id(value: object) -> str:
    normalized = _required(value, field_name="route_id")
    if _NAMESPACED_IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "route_id must be a lowercase namespaced identifier local to a skill."
        )
    return normalized


def validate_version(value: object, *, field_name: str) -> str:
    normalized = _required(value, field_name=field_name)
    if _VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a positive integer string.")
    return normalized


def _freeze_json(value: Any, *, field_name: str, depth: int = 0) -> Any:
    if depth > MAX_AGENT_SKILL_JSON_DEPTH_V1:
        raise ValueError(
            f"{field_name} exceeds maximum JSON depth "
            f"{MAX_AGENT_SKILL_JSON_DEPTH_V1}."
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} rejects non-finite floats.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} requires string object keys.")
        return MappingProxyType(
            {
                key: _freeze_json(
                    value[key],
                    field_name=f"{field_name}.{key}",
                    depth=depth + 1,
                )
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(
                item,
                field_name=f"{field_name}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{field_name} accepts JSON-compatible values only; got "
        f"{type(value).__name__}."
    )


def _freeze_json_object(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object.")
    frozen = _freeze_json(value, field_name=field_name)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise TypeError(f"{field_name} must be a JSON object.")
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _strict_object_schema(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    frozen = _freeze_json_object(value, field_name=field_name)
    if frozen.get("type") != "object":
        raise ValueError(f"{field_name} root type must be 'object'.")
    if frozen.get("additionalProperties") is not False:
        raise ValueError(
            f"{field_name} must set root additionalProperties=false."
        )
    properties = frozen.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError(f"{field_name} must define object properties.")
    required = frozen.get("required", ())
    if not isinstance(required, tuple) or any(
        not isinstance(item, str) for item in required
    ):
        raise ValueError(f"{field_name}.required must be a string array.")
    unknown_required = sorted(set(required) - set(properties))
    if unknown_required:
        raise ValueError(
            f"{field_name}.required names unknown properties: "
            f"{unknown_required}."
        )
    return frozen


def agent_skill_schema_document_hash(
    schema: Mapping[str, Any],
    *,
    schema_kind: str,
) -> str:
    """Hash one strict portable schema in a kind-separated domain."""

    kind = _enum(
        schema_kind,
        field_name="schema_kind",
        allowed=("execution_result", "parameter", "task_evaluation"),
    )
    strict = _strict_object_schema(schema, field_name=f"{kind}_schema")
    return canonical_content_hash(
        _thaw_json(strict),
        domain=f"{AGENT_SKILL_SCHEMA_DOCUMENT_HASH_DOMAIN_V2}.{kind}",
    )


@dataclass(frozen=True)
class SkillDefinition:
    """Task semantics plus intended, non-exclusive execution target."""

    skill_id: str
    skill_version: str
    skill_kind: str
    summary: str
    intended_target_kind: str
    cross_target_deployment_policy: str
    controlled_component_ids: tuple[str, ...]
    parameter_semantics_id: str
    execution_result_semantics_id: str
    task_outcome_semantics_id: str
    entry_predicate_ids: tuple[str, ...] = ()
    task_outcome_predicate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )
        object.__setattr__(
            self,
            "skill_kind",
            _enum(
                self.skill_kind,
                field_name="skill_kind",
                allowed=AGENT_SKILL_KINDS_V1,
            ),
        )
        intended_target_kind = _enum(
            self.intended_target_kind,
            field_name="intended_target_kind",
            allowed=AGENT_SKILL_INTENDED_TARGET_KINDS_V1,
        )
        cross_target_policy = _enum(
            self.cross_target_deployment_policy,
            field_name="cross_target_deployment_policy",
            allowed=AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1,
        )
        if intended_target_kind == "target_agnostic":
            if cross_target_policy != "not_applicable":
                raise ValueError(
                    "target_agnostic Skills require "
                    "cross_target_deployment_policy='not_applicable'."
                )
        elif cross_target_policy == "not_applicable":
            raise ValueError(
                "Real- or sim-intended Skills must explicitly forbid "
                "cross-target deployment or require deployment review."
            )
        object.__setattr__(
            self,
            "intended_target_kind",
            intended_target_kind,
        )
        object.__setattr__(
            self,
            "cross_target_deployment_policy",
            cross_target_policy,
        )
        for field_name in (
            "summary",
            "parameter_semantics_id",
            "execution_result_semantics_id",
            "task_outcome_semantics_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "controlled_component_ids",
            _unique_sorted(
                self.controlled_component_ids,
                field_name="controlled_component_ids",
            ),
        )
        for field_name in ("entry_predicate_ids", "task_outcome_predicate_ids"):
            object.__setattr__(
                self,
                field_name,
                _unique_sorted(getattr(self, field_name), field_name=field_name),
            )
        if self.skill_kind in {"motion", "policy_task"} and not (
            self.controlled_component_ids
        ):
            raise ValueError(
                "Motion and policy-task definitions require controlled components."
            )
        if self.skill_id in PRIVILEGED_LIFECYCLE_SKILL_IDS_V1 and (
            self.skill_kind != "lifecycle"
        ):
            raise ValueError(
                "Frozen lifecycle skill IDs require skill_kind='lifecycle'."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_DEFINITION_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_kind": self.skill_kind,
            "summary": self.summary,
            "intended_target_kind": self.intended_target_kind,
            "cross_target_deployment_policy": (
                self.cross_target_deployment_policy
            ),
            "controlled_component_ids": list(self.controlled_component_ids),
            "parameter_semantics_id": self.parameter_semantics_id,
            "execution_result_semantics_id": (
                self.execution_result_semantics_id
            ),
            "task_outcome_semantics_id": self.task_outcome_semantics_id,
            "entry_predicate_ids": list(self.entry_predicate_ids),
            "task_outcome_predicate_ids": list(
                self.task_outcome_predicate_ids
            ),
        }
        if include_hash:
            payload["skill_definition_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillDefinitionCatalog:
    """Immutable catalog of target-intended, route-independent definitions."""

    catalog_id: str
    catalog_version: str
    skill_definitions: tuple[SkillDefinition, ...]
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(f"Unsupported protocol: {self.protocol_version!r}.")
        object.__setattr__(
            self,
            "catalog_id",
            _required(self.catalog_id, field_name="catalog_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            validate_version(self.catalog_version, field_name="catalog_version"),
        )
        definitions = tuple(self.skill_definitions)
        if any(not isinstance(item, SkillDefinition) for item in definitions):
            raise TypeError(
                "skill_definitions must contain SkillDefinition values."
            )
        identities = tuple(
            (item.skill_id, item.skill_version) for item in definitions
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Skill definition identities must be unique.")
        object.__setattr__(
            self,
            "skill_definitions",
            tuple(
                sorted(
                    definitions,
                    key=lambda item: (item.skill_id, item.skill_version),
                )
            ),
        )

    def definition(self, skill_id: str, skill_version: str) -> SkillDefinition:
        identity = (validate_skill_id(skill_id), validate_version(
            skill_version, field_name="skill_version"
        ))
        for definition in self.skill_definitions:
            if (definition.skill_id, definition.skill_version) == identity:
                return definition
        raise KeyError(f"Unknown Skill definition: {identity!r}.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "protocol_version": self.protocol_version,
                "catalog_id": self.catalog_id,
                "catalog_version": self.catalog_version,
                "skill_definition_hashes": tuple(
                    item.semantic_content_hash for item in self.skill_definitions
                ),
            },
            domain=AGENT_SKILL_DEFINITION_CATALOG_HASH_DOMAIN_V2,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "skill_definitions": [
                item.to_wire() for item in self.skill_definitions
            ],
            "skill_definition_catalog_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class SkillParameterProfileRef:
    """Trusted mapping from a profile identity to one exact parameter schema."""

    parameter_profile_id: str
    parameter_profile_hash: str
    parameter_schema_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "parameter_profile_id",
            "parameter_profile_hash",
            "parameter_schema_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    def to_wire(self) -> dict[str, str]:
        return {
            "parameter_profile_id": self.parameter_profile_id,
            "parameter_profile_hash": self.parameter_profile_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
        }


@dataclass(frozen=True)
class AgentSkillSchemaEntry:
    """Exact portable parameter/result/evaluation schemas for one Skill."""

    skill_id: str
    skill_version: str
    skill_definition_hash: str
    parameter_schema: Mapping[str, Any]
    execution_result_schema: Mapping[str, Any]
    task_evaluation_schema: Mapping[str, Any]
    parameter_profiles: tuple[SkillParameterProfileRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )
        object.__setattr__(
            self,
            "skill_definition_hash",
            _required(
                self.skill_definition_hash,
                field_name="skill_definition_hash",
            ),
        )
        for field_name in (
            "parameter_schema",
            "execution_result_schema",
            "task_evaluation_schema",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_object_schema(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        profiles = tuple(self.parameter_profiles)
        if not profiles:
            raise ValueError("parameter_profiles must not be empty.")
        if any(not isinstance(item, SkillParameterProfileRef) for item in profiles):
            raise TypeError(
                "parameter_profiles must contain SkillParameterProfileRef values."
            )
        identities = tuple(item.parameter_profile_id for item in profiles)
        hashes = tuple(item.parameter_profile_hash for item in profiles)
        if len(set(identities)) != len(identities):
            raise ValueError("parameter_profile_id values must be unique.")
        if len(set(hashes)) != len(hashes):
            raise ValueError("parameter_profile_hash values must be unique.")
        for item in profiles:
            if item.parameter_schema_hash != self.parameter_schema_hash:
                raise ValueError(
                    "Each parameter profile must bind this entry's exact "
                    "parameter schema hash."
                )
        object.__setattr__(
            self,
            "parameter_profiles",
            tuple(sorted(profiles, key=lambda item: item.parameter_profile_id)),
        )

    @property
    def parameter_schema_hash(self) -> str:
        return agent_skill_schema_document_hash(
            self.parameter_schema,
            schema_kind="parameter",
        )

    @property
    def execution_result_schema_hash(self) -> str:
        return agent_skill_schema_document_hash(
            self.execution_result_schema,
            schema_kind="execution_result",
        )

    @property
    def task_evaluation_schema_hash(self) -> str:
        return agent_skill_schema_document_hash(
            self.task_evaluation_schema,
            schema_kind="task_evaluation",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_SCHEMA_ENTRY_HASH_DOMAIN_V2,
        )

    def profile(self, parameter_profile_id: str) -> SkillParameterProfileRef:
        requested = _required(
            parameter_profile_id,
            field_name="parameter_profile_id",
        )
        for profile in self.parameter_profiles:
            if profile.parameter_profile_id == requested:
                return profile
        raise KeyError(f"Unknown parameter profile: {requested!r}.")

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "parameter_schema": _thaw_json(self.parameter_schema),
            "parameter_schema_hash": self.parameter_schema_hash,
            "execution_result_schema": _thaw_json(
                self.execution_result_schema
            ),
            "execution_result_schema_hash": self.execution_result_schema_hash,
            "task_evaluation_schema": _thaw_json(self.task_evaluation_schema),
            "task_evaluation_schema_hash": self.task_evaluation_schema_hash,
            "parameter_profiles": [
                item.to_wire() for item in self.parameter_profiles
            ],
        }
        if include_hash:
            payload["skill_schema_entry_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AgentSkillSchemaCatalog:
    """Immutable portable schema distribution owned by DexTeleop."""

    catalog_id: str
    catalog_version: str
    schema_entries: tuple[AgentSkillSchemaEntry, ...]
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(f"Unsupported protocol: {self.protocol_version!r}.")
        object.__setattr__(
            self,
            "catalog_id",
            _required(self.catalog_id, field_name="catalog_id"),
        )
        object.__setattr__(
            self,
            "catalog_version",
            validate_version(self.catalog_version, field_name="catalog_version"),
        )
        entries = tuple(self.schema_entries)
        if any(not isinstance(item, AgentSkillSchemaEntry) for item in entries):
            raise TypeError(
                "schema_entries must contain AgentSkillSchemaEntry values."
            )
        identities = tuple((item.skill_id, item.skill_version) for item in entries)
        if len(set(identities)) != len(identities):
            raise ValueError("Skill schema entry identities must be unique.")
        object.__setattr__(
            self,
            "schema_entries",
            tuple(sorted(entries, key=lambda item: (item.skill_id, item.skill_version))),
        )

    def entry(self, skill_id: str, skill_version: str) -> AgentSkillSchemaEntry:
        identity = (validate_skill_id(skill_id), validate_version(
            skill_version, field_name="skill_version"
        ))
        for entry in self.schema_entries:
            if (entry.skill_id, entry.skill_version) == identity:
                return entry
        raise KeyError(f"Unknown Skill schema entry: {identity!r}.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "protocol_version": self.protocol_version,
                "catalog_id": self.catalog_id,
                "catalog_version": self.catalog_version,
                "schema_entry_hashes": tuple(
                    item.semantic_content_hash for item in self.schema_entries
                ),
            },
            domain=AGENT_SKILL_SCHEMA_CATALOG_HASH_DOMAIN_V2,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "schema_entries": [item.to_wire() for item in self.schema_entries],
            "skill_schema_catalog_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class SkillExecutionRoute:
    """Stable, deployment-independent execution route for one Skill."""

    skill_id: str
    skill_version: str
    skill_definition_hash: str
    route_id: str
    route_version: str
    parameter_schema_hash: str
    execution_result_schema_hash: str
    task_evaluation_schema_hash: str
    allowed_parameter_profiles: tuple[SkillParameterProfileRef, ...]
    control_effect_class: str
    privilege_tier: str
    supported_modes: tuple[str, ...]
    target_kind: str
    may_move_robot: bool
    may_open_publication: bool
    requires_operator_confirmation: bool
    requires_deadman: bool
    max_operation_duration_ms: int
    cancellation_policy: str
    max_concurrent_operations: int = 1
    requires_task_evaluation: bool = False
    skill_io_contract_hash: str | None = None
    resource_claims: tuple[SkillResourceClaim, ...] = (
        DEFAULT_AGENT_SKILL_RESOURCE_CLAIMS_V2
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )
        object.__setattr__(
            self,
            "route_id",
            validate_route_id(self.route_id),
        )
        object.__setattr__(
            self,
            "route_version",
            validate_version(self.route_version, field_name="route_version"),
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
                _required(getattr(self, field_name), field_name=field_name),
            )
        profiles = tuple(self.allowed_parameter_profiles)
        if not profiles:
            raise ValueError("allowed_parameter_profiles must not be empty.")
        if any(not isinstance(item, SkillParameterProfileRef) for item in profiles):
            raise TypeError(
                "allowed_parameter_profiles must contain "
                "SkillParameterProfileRef values."
            )
        profile_ids = tuple(item.parameter_profile_id for item in profiles)
        profile_hashes = tuple(item.parameter_profile_hash for item in profiles)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Allowed parameter profile IDs must be unique.")
        if len(set(profile_hashes)) != len(profile_hashes):
            raise ValueError("Allowed parameter profile hashes must be unique.")
        for profile in profiles:
            if profile.parameter_schema_hash != self.parameter_schema_hash:
                raise ValueError(
                    "Allowed parameter profiles must bind the Route's exact "
                    "parameter schema hash."
                )
        object.__setattr__(
            self,
            "allowed_parameter_profiles",
            tuple(sorted(profiles, key=lambda item: item.parameter_profile_id)),
        )
        effect = _enum(
            self.control_effect_class,
            field_name="control_effect_class",
            allowed=AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1,
        )
        tier = _enum(
            self.privilege_tier,
            field_name="privilege_tier",
            allowed=AGENT_SKILL_PRIVILEGE_TIERS_V1,
        )
        modes = _unique_sorted(
            self.supported_modes,
            field_name="supported_modes",
            allowed=AGENT_SKILL_MODES_V1,
            allow_empty=False,
        )
        target_kind = _enum(
            self.target_kind,
            field_name="target_kind",
            allowed=AGENT_SKILL_TARGET_KINDS_V1,
        )
        object.__setattr__(self, "control_effect_class", effect)
        object.__setattr__(self, "privilege_tier", tier)
        object.__setattr__(self, "supported_modes", modes)
        object.__setattr__(self, "target_kind", target_kind)
        for field_name in (
            "may_move_robot",
            "may_open_publication",
            "requires_operator_confirmation",
            "requires_deadman",
        ):
            object.__setattr__(
                self,
                field_name,
                _boolean(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "max_operation_duration_ms",
            _positive_integer(
                self.max_operation_duration_ms,
                field_name="max_operation_duration_ms",
                maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            ),
        )
        object.__setattr__(
            self,
            "cancellation_policy",
            _enum(
                self.cancellation_policy,
                field_name="cancellation_policy",
                allowed=AGENT_SKILL_CANCELLATION_POLICIES_V1,
            ),
        )
        concurrency = _positive_integer(
            self.max_concurrent_operations,
            field_name="max_concurrent_operations",
            maximum=64,
        )
        object.__setattr__(self, "max_concurrent_operations", concurrency)
        object.__setattr__(
            self,
            "requires_task_evaluation",
            _boolean(
                self.requires_task_evaluation,
                field_name="requires_task_evaluation",
            ),
        )
        io_contract_hash = self.skill_io_contract_hash
        if io_contract_hash is None:
            io_contract_hash = inline_only_skill_io_contract(
                skill_id=self.skill_id,
                skill_version=self.skill_version,
                skill_definition_hash=self.skill_definition_hash,
                parameter_schema_hash=self.parameter_schema_hash,
                execution_result_schema_hash=self.execution_result_schema_hash,
                task_evaluation_schema_hash=self.task_evaluation_schema_hash,
            ).semantic_content_hash
        object.__setattr__(
            self,
            "skill_io_contract_hash",
            _required(io_contract_hash, field_name="skill_io_contract_hash"),
        )
        claims = tuple(self.resource_claims)
        if not claims or any(
            not isinstance(claim, SkillResourceClaim) for claim in claims
        ):
            raise ValueError(
                "resource_claims must contain at least one SkillResourceClaim."
            )
        claim_ids = tuple(claim.resource_id for claim in claims)
        if claim_ids != tuple(sorted(claim_ids)) or len(set(claim_ids)) != len(
            claim_ids
        ):
            raise ValueError(
                "resource_claims must be uniquely sorted by resource_id."
            )
        object.__setattr__(self, "resource_claims", claims)

        if target_kind == "null" and modes != ("dry_run",):
            raise ValueError("A null Route only supports dry_run mode.")
        if target_kind == "sim" and not set(modes) <= {"dry_run", "sim_active"}:
            raise ValueError("A sim Route cannot advertise a real mode.")
        if target_kind == "real" and not set(modes) <= {
            "dry_run",
            "real_active",
            "real_observe",
        }:
            raise ValueError("A real Route cannot advertise sim_active mode.")
        if effect == "read_only" and (
            self.may_move_robot or self.may_open_publication
        ):
            raise ValueError("Read-only Routes cannot move or open publication.")
        if effect == "motion" and not self.may_move_robot:
            raise ValueError("Motion Routes must declare may_move_robot=true.")
        if tier == "t0_read_only" and (
            effect != "read_only"
            or self.requires_operator_confirmation
            or self.requires_deadman
        ):
            raise ValueError("T0 Routes must be strictly read-only.")
        if tier == "t2_real_motion" and (
            effect != "motion"
            or target_kind != "real"
            or "real_active" not in modes
        ):
            raise ValueError("T2 Routes must describe real-active motion.")
        if tier == "t3_lifecycle" and (
            effect != "lifecycle" or not self.requires_operator_confirmation
        ):
            raise ValueError(
                "T3 lifecycle Routes require lifecycle effect and operator "
                "confirmation."
            )
        if self.requires_deadman and not (
            self.may_move_robot or self.may_open_publication
        ):
            raise ValueError(
                "Deadman can only guard motion or publication-opening Routes."
            )
        if "real_observe" in modes and (
            self.may_move_robot or self.may_open_publication
        ):
            raise ValueError(
                "real_observe Routes cannot move or open publication."
            )
        self._validate_frozen_lifecycle_semantics()

    def _validate_frozen_lifecycle_semantics(self) -> None:
        expected = {
            "lifecycle.capture_runtime_home": (False, False, False),
            "lifecycle.move_to_registered_home": (True, False, True),
            "lifecycle.rearm_safety_latch": (False, False, False),
            "lifecycle.resume_after_hold": (False, True, True),
        }.get(self.skill_id)
        if expected is None:
            return
        actual = (
            self.may_move_robot,
            self.may_open_publication,
            self.requires_deadman,
        )
        if (
            self.control_effect_class != "lifecycle"
            or self.privilege_tier != "t3_lifecycle"
            or not self.requires_operator_confirmation
            or actual != expected
        ):
            raise ValueError(
                f"{self.skill_id} does not match its frozen lifecycle semantics."
            )

    def profile(
        self,
        parameter_profile_id: str,
        parameter_profile_hash: str | None = None,
    ) -> SkillParameterProfileRef:
        requested_id = _required(
            parameter_profile_id,
            field_name="parameter_profile_id",
        )
        for profile in self.allowed_parameter_profiles:
            if profile.parameter_profile_id != requested_id:
                continue
            if (
                parameter_profile_hash is None
                or profile.parameter_profile_hash == parameter_profile_hash
            ):
                return profile
        raise KeyError(
            "Unknown allowed parameter profile: "
            f"{(requested_id, parameter_profile_hash)!r}."
        )

    @property
    def allowed_parameter_profile_hashes(self) -> tuple[str, ...]:
        return tuple(
            item.parameter_profile_hash
            for item in self.allowed_parameter_profiles
        )

    @property
    def allowed_parameter_profile_ids(self) -> tuple[str, ...]:
        return tuple(
            item.parameter_profile_id for item in self.allowed_parameter_profiles
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_ROUTE_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "route_id": self.route_id,
            "route_version": self.route_version,
            "parameter_schema_hash": self.parameter_schema_hash,
            "execution_result_schema_hash": self.execution_result_schema_hash,
            "task_evaluation_schema_hash": self.task_evaluation_schema_hash,
            "allowed_parameter_profiles": [
                item.to_wire() for item in self.allowed_parameter_profiles
            ],
            "control_effect_class": self.control_effect_class,
            "privilege_tier": self.privilege_tier,
            "supported_modes": list(self.supported_modes),
            "target_kind": self.target_kind,
            "may_move_robot": self.may_move_robot,
            "may_open_publication": self.may_open_publication,
            "requires_operator_confirmation": (
                self.requires_operator_confirmation
            ),
            "requires_deadman": self.requires_deadman,
            "max_operation_duration_ms": self.max_operation_duration_ms,
            "cancellation_policy": self.cancellation_policy,
            "max_concurrent_operations": self.max_concurrent_operations,
            "requires_task_evaluation": self.requires_task_evaluation,
            "skill_io_contract_hash": self.skill_io_contract_hash,
            "resource_claims": [
                claim.to_wire() for claim in self.resource_claims
            ],
        }
        if include_hash:
            payload["route_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class CrossTargetDeploymentReviewRef:
    """Immutable deployment-time approval for one non-native target Route."""

    review_id: str
    skill_definition_hash: str
    route_hash: str
    intended_target_kind: str
    deployed_target_kind: str
    compatibility_assessment_hash: str
    approval_receipt_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "review_id",
            "skill_definition_hash",
            "route_hash",
            "compatibility_assessment_hash",
            "approval_receipt_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        intended = _enum(
            self.intended_target_kind,
            field_name="intended_target_kind",
            allowed=("real", "sim"),
        )
        deployed = _enum(
            self.deployed_target_kind,
            field_name="deployed_target_kind",
            allowed=("real", "sim"),
        )
        if intended == deployed:
            raise ValueError(
                "Cross-target review requires different intended and deployed "
                "target kinds."
            )
        object.__setattr__(self, "intended_target_kind", intended)
        object.__setattr__(self, "deployed_target_kind", deployed)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=(
                AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_REVIEW_REF_HASH_DOMAIN_V2
            ),
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, str]:
        payload = {
            "review_id": self.review_id,
            "skill_definition_hash": self.skill_definition_hash,
            "route_hash": self.route_hash,
            "intended_target_kind": self.intended_target_kind,
            "deployed_target_kind": self.deployed_target_kind,
            "compatibility_assessment_hash": (
                self.compatibility_assessment_hash
            ),
            "approval_receipt_hash": self.approval_receipt_hash,
        }
        if include_hash:
            payload["cross_target_deployment_review_ref_hash"] = (
                self.semantic_content_hash
            )
        return payload


@dataclass(frozen=True)
class SkillDeploymentSafetyEnvelopeRef:
    """Public opaque ref to one exact lower-owned safety envelope."""

    object_id: str
    schema: str
    content_hash: str
    constraint_set_hash: str
    allowed_route_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "object_id",
            "schema",
            "content_hash",
            "constraint_set_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.schema != "dexteleop.deployment_safety_envelope.v1":
            raise ValueError("Unsupported deployment safety envelope schema.")
        object.__setattr__(
            self,
            "allowed_route_ids",
            _unique_sorted(
                self.allowed_route_ids,
                field_name="allowed_route_ids",
                allow_empty=False,
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "schema": self.schema,
            "content_hash": self.content_hash,
            "constraint_set_hash": self.constraint_set_hash,
            "allowed_route_ids": list(self.allowed_route_ids),
        }


@dataclass(frozen=True)
class SkillDeploymentBindingRef:
    """Public immutable reference to a lower-owned deployment binding."""

    deployment_binding_id: str
    deployment_binding_hash: str
    target_binding_id: str
    target_kind: str
    execution_manifest_hash: str
    cross_target_deployment_review: (
        CrossTargetDeploymentReviewRef | None
    ) = None
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "deployment_binding_id",
            "deployment_binding_hash",
            "target_binding_id",
            "execution_manifest_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "target_kind",
            _enum(
                self.target_kind,
                field_name="target_kind",
                allowed=AGENT_SKILL_TARGET_KINDS_V1,
            ),
        )
        review = self.cross_target_deployment_review
        if review is not None:
            if not isinstance(review, CrossTargetDeploymentReviewRef):
                raise TypeError(
                    "cross_target_deployment_review must be a "
                    "CrossTargetDeploymentReviewRef or None."
                )
            if review.deployed_target_kind != self.target_kind:
                raise ValueError(
                    "Cross-target review deployed_target_kind must match the "
                    "deployment binding target_kind."
                )
        envelopes = tuple(self.safety_envelope_refs)
        if any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef)
            for item in envelopes
        ):
            raise TypeError(
                "safety_envelope_refs must contain typed envelope refs."
            )
        envelope_ids = tuple(item.object_id for item in envelopes)
        envelope_hashes = tuple(item.content_hash for item in envelopes)
        if len(set(envelope_ids)) != len(envelope_ids) or len(
            set(envelope_hashes)
        ) != len(envelope_hashes):
            raise ValueError("Safety envelope IDs and hashes must be unique.")
        object.__setattr__(
            self,
            "safety_envelope_refs",
            tuple(sorted(envelopes, key=lambda item: item.object_id)),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_reference_hash=False),
            domain=AGENT_SKILL_DEPLOYMENT_BINDING_REF_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_reference_hash: bool = True) -> dict[str, Any]:
        payload = {
            "deployment_binding_id": self.deployment_binding_id,
            "deployment_binding_hash": self.deployment_binding_hash,
            "target_binding_id": self.target_binding_id,
            "target_kind": self.target_kind,
            "execution_manifest_hash": self.execution_manifest_hash,
            "cross_target_deployment_review": (
                None
                if self.cross_target_deployment_review is None
                else self.cross_target_deployment_review.to_wire()
            ),
        }
        if self.safety_envelope_refs:
            payload["safety_envelope_refs"] = [
                item.to_wire() for item in self.safety_envelope_refs
            ]
        if include_reference_hash:
            payload["deployment_binding_ref_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillRouteManifestEntry:
    """One stable Route joined to the current immutable deployment binding."""

    route: SkillExecutionRoute
    deployment_binding: SkillDeploymentBindingRef
    evaluation_plan_hash: str | None
    availability_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.route, SkillExecutionRoute):
            raise TypeError("route must be a SkillExecutionRoute.")
        if not isinstance(self.deployment_binding, SkillDeploymentBindingRef):
            raise TypeError(
                "deployment_binding must be a SkillDeploymentBindingRef."
            )
        if self.route.requires_task_evaluation:
            if self.evaluation_plan_hash is None:
                raise ValueError(
                    "Evaluation-required Route entries must bind an exact "
                    "evaluation_plan_hash."
                )
            object.__setattr__(
                self,
                "evaluation_plan_hash",
                _required(
                    self.evaluation_plan_hash,
                    field_name="evaluation_plan_hash",
                ),
            )
        elif self.evaluation_plan_hash is not None:
            raise ValueError(
                "Routes without task evaluation cannot bind an "
                "evaluation_plan_hash."
            )
        object.__setattr__(
            self,
            "availability_hash",
            _required(self.availability_hash, field_name="availability_hash"),
        )
        if self.route.target_kind != self.deployment_binding.target_kind:
            raise ValueError("Route and deployment target_kind must match.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_ROUTE_MANIFEST_ENTRY_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "route": self.route.to_wire(),
            "deployment_binding": self.deployment_binding.to_wire(),
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "availability_hash": self.availability_hash,
        }
        if include_hash:
            payload["route_manifest_entry_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class SkillRouteManifest:
    """Session-bound advertised Route surface for one control domain."""

    runtime_session_id: str
    control_domain_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    skill_definition_catalog_hash: str
    skill_schema_catalog_hash: str
    target_binding_id: str
    target_kind: str
    current_mode: str
    supported_transport_features: tuple[str, ...]
    skill_routes: tuple[SkillRouteManifestEntry, ...]
    explicit_exclusions: tuple[str, ...] = ()
    skill_io_catalog_hash: str | None = None
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported agent skill protocol: {self.protocol_version!r}."
            )
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "skill_definition_catalog_hash",
            "skill_schema_catalog_hash",
            "target_binding_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        target_kind = _enum(
            self.target_kind,
            field_name="target_kind",
            allowed=AGENT_SKILL_TARGET_KINDS_V1,
        )
        mode = _enum(
            self.current_mode,
            field_name="current_mode",
            allowed=AGENT_SKILL_MODES_V1,
        )
        if target_kind == "null" and mode != "dry_run":
            raise ValueError("A null target only supports dry_run mode.")
        if target_kind == "sim" and mode not in {"dry_run", "sim_active"}:
            raise ValueError("A sim target cannot advertise a real mode.")
        if target_kind == "real" and mode not in {
            "dry_run",
            "real_active",
            "real_observe",
        }:
            raise ValueError("A real target cannot advertise sim_active mode.")
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "current_mode", mode)
        object.__setattr__(
            self,
            "supported_transport_features",
            _unique_sorted(
                self.supported_transport_features,
                field_name="supported_transport_features",
            ),
        )
        entries = tuple(self.skill_routes)
        if any(not isinstance(item, SkillRouteManifestEntry) for item in entries):
            raise TypeError(
                "skill_routes must contain SkillRouteManifestEntry values."
            )
        identities = tuple(
            (
                item.route.skill_id,
                item.route.skill_version,
                item.route.route_id,
                item.route.route_version,
            )
            for item in entries
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Advertised Skill Route identities must be unique.")
        for entry in entries:
            route = entry.route
            binding = entry.deployment_binding
            if route.target_kind != target_kind or binding.target_kind != target_kind:
                raise ValueError("Manifest, Route, and binding target_kind must match.")
            if binding.target_binding_id != self.target_binding_id:
                raise ValueError("Deployment target binding does not match manifest.")
            if binding.execution_manifest_hash != self.execution_manifest_hash:
                raise ValueError(
                    "Deployment execution manifest does not match manifest."
                )
            if mode not in route.supported_modes:
                raise ValueError(
                    f"Route {route.route_id!r} does not support current mode "
                    f"{mode!r}."
                )
            if mode == "real_observe" and (
                route.may_move_robot or route.may_open_publication
            ):
                raise ValueError(
                    "real_observe manifests cannot expose motion or publication "
                    "opening Routes."
                )
            selection_feature = (
                "safety_profile_selection_v1"
                in self.supported_transport_features
            )
            if route.control_effect_class == "motion" and selection_feature:
                if not binding.safety_envelope_refs:
                    raise ValueError(
                        "Safety-selectable motion Route requires lower envelope refs."
                    )
                if not any(
                    route.route_id in item.allowed_route_ids
                    for item in binding.safety_envelope_refs
                ):
                    raise ValueError(
                        "Motion Route is absent from its safety envelope allowlists."
                    )
            if binding.safety_envelope_refs and not selection_feature:
                raise ValueError(
                    "Safety envelope refs require the selection transport feature."
                )
        object.__setattr__(
            self,
            "skill_routes",
            tuple(
                sorted(
                    entries,
                    key=lambda item: (
                        item.route.skill_id,
                        item.route.skill_version,
                        item.route.route_id,
                        item.route.route_version,
                    ),
                )
            ),
        )
        io_catalog_hash = self.skill_io_catalog_hash
        if io_catalog_hash is None:
            inline_catalog = inline_only_skill_io_catalog_for_routes(
                entry.route for entry in self.skill_routes
            )
            for entry in self.skill_routes:
                contract = inline_catalog.contract(
                    entry.route.skill_id,
                    entry.route.skill_version,
                )
                if entry.route.skill_io_contract_hash != (
                    contract.semantic_content_hash
                ):
                    raise ValueError(
                        "Custom Artifact I/O Routes require an explicit "
                        "skill_io_catalog_hash."
                    )
            io_catalog_hash = inline_catalog.semantic_content_hash
        object.__setattr__(
            self,
            "skill_io_catalog_hash",
            _required(io_catalog_hash, field_name="skill_io_catalog_hash"),
        )
        object.__setattr__(
            self,
            "explicit_exclusions",
            _unique_sorted(
                self.explicit_exclusions,
                field_name="explicit_exclusions",
            ),
        )

    def route(
        self,
        skill_id: str,
        skill_version: str,
        route_id: str | None = None,
        route_version: str | None = None,
    ) -> SkillRouteManifestEntry:
        requested_skill = validate_skill_id(skill_id)
        requested_skill_version = validate_version(
            skill_version,
            field_name="skill_version",
        )
        requested_route = None if route_id is None else validate_route_id(route_id)
        requested_route_version = (
            None
            if route_version is None
            else validate_version(route_version, field_name="route_version")
        )
        matches = [
            entry
            for entry in self.skill_routes
            if entry.route.skill_id == requested_skill
            and entry.route.skill_version == requested_skill_version
            and (
                requested_route is None or entry.route.route_id == requested_route
            )
            and (
                requested_route_version is None
                or entry.route.route_version == requested_route_version
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(
                "Unknown advertised Skill Route: "
                f"{(requested_skill, requested_skill_version, requested_route, requested_route_version)!r}."
            )
        raise KeyError("Skill Route reference is ambiguous; route identity required.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "protocol_version": self.protocol_version,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "station_manifest_hash": self.station_manifest_hash,
                "execution_manifest_hash": self.execution_manifest_hash,
                "skill_definition_catalog_hash": (
                    self.skill_definition_catalog_hash
                ),
                "skill_schema_catalog_hash": self.skill_schema_catalog_hash,
                "skill_io_catalog_hash": self.skill_io_catalog_hash,
                "target_binding_id": self.target_binding_id,
                "target_kind": self.target_kind,
                "current_mode": self.current_mode,
                "supported_transport_features": (
                    self.supported_transport_features
                ),
                "skill_route_entry_hashes": tuple(
                    item.semantic_content_hash for item in self.skill_routes
                ),
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V2,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "station_manifest_hash": self.station_manifest_hash,
            "execution_manifest_hash": self.execution_manifest_hash,
            "skill_definition_catalog_hash": self.skill_definition_catalog_hash,
            "skill_schema_catalog_hash": self.skill_schema_catalog_hash,
            "skill_io_catalog_hash": self.skill_io_catalog_hash,
            "target_binding_id": self.target_binding_id,
            "target_kind": self.target_kind,
            "current_mode": self.current_mode,
            "supported_transport_features": list(
                self.supported_transport_features
            ),
            "skill_routes": [item.to_wire() for item in self.skill_routes],
            "explicit_exclusions": list(self.explicit_exclusions),
            "route_manifest_hash": self.semantic_content_hash,
        }


def validate_skill_route_catalog_closure(
    *,
    definition_catalog: SkillDefinitionCatalog,
    schema_catalog: AgentSkillSchemaCatalog,
    route_manifest: SkillRouteManifest,
    io_catalog: SkillIOCatalog | None = None,
) -> None:
    """Fail closed unless every advertised Route closes over exact catalogs."""

    if not isinstance(definition_catalog, SkillDefinitionCatalog):
        raise TypeError("definition_catalog must be a SkillDefinitionCatalog.")
    if not isinstance(schema_catalog, AgentSkillSchemaCatalog):
        raise TypeError("schema_catalog must be an AgentSkillSchemaCatalog.")
    if not isinstance(route_manifest, SkillRouteManifest):
        raise TypeError("route_manifest must be a SkillRouteManifest.")
    if io_catalog is None:
        io_catalog = inline_only_skill_io_catalog_for_routes(
            entry.route for entry in route_manifest.skill_routes
        )
    elif not isinstance(io_catalog, SkillIOCatalog):
        raise TypeError("io_catalog must be a SkillIOCatalog or None.")
    if (
        route_manifest.skill_definition_catalog_hash
        != definition_catalog.semantic_content_hash
    ):
        raise ValueError("Skill definition catalog hash mismatch.")
    if route_manifest.skill_schema_catalog_hash != schema_catalog.semantic_content_hash:
        raise ValueError("Agent Skill schema catalog hash mismatch.")
    if route_manifest.skill_io_catalog_hash != io_catalog.semantic_content_hash:
        raise ValueError("Agent Skill I/O catalog hash mismatch.")
    for advertised in route_manifest.skill_routes:
        route = advertised.route
        try:
            definition = definition_catalog.definition(
                route.skill_id,
                route.skill_version,
            )
        except KeyError as error:
            raise ValueError("Route references an unknown Skill definition.") from error
        if route.skill_definition_hash != definition.semantic_content_hash:
            raise ValueError("Route Skill definition hash mismatch.")
        review = advertised.deployment_binding.cross_target_deployment_review
        intended_target = definition.intended_target_kind
        deployed_target = route.target_kind
        is_cross_target = (
            intended_target in {"real", "sim"}
            and deployed_target in {"real", "sim"}
            and intended_target != deployed_target
        )
        if not is_cross_target:
            if review is not None:
                raise ValueError(
                    "Native-target, target-agnostic, and null Routes cannot "
                    "bind a cross-target deployment review."
                )
        elif definition.cross_target_deployment_policy == "forbidden":
            raise ValueError(
                "Skill Definition forbids deployment to this Route target."
            )
        else:
            if review is None:
                raise ValueError(
                    "Cross-target deployment requires an immutable deployment "
                    "review."
                )
            review_checks = (
                review.skill_definition_hash
                == definition.semantic_content_hash,
                review.route_hash == route.semantic_content_hash,
                review.intended_target_kind == intended_target,
                review.deployed_target_kind == deployed_target,
            )
            if not all(review_checks):
                raise ValueError(
                    "Cross-target deployment review does not bind the exact "
                    "Definition and Route."
                )
        try:
            schema_entry = schema_catalog.entry(
                route.skill_id,
                route.skill_version,
            )
        except KeyError as error:
            raise ValueError("Route references an unknown Skill schema entry.") from error
        schema_checks = (
            schema_entry.skill_definition_hash == route.skill_definition_hash,
            schema_entry.parameter_schema_hash == route.parameter_schema_hash,
            schema_entry.execution_result_schema_hash
            == route.execution_result_schema_hash,
            schema_entry.task_evaluation_schema_hash
            == route.task_evaluation_schema_hash,
        )
        if not all(schema_checks):
            raise ValueError("Route schema hashes do not close over schema catalog.")
        try:
            io_contract = io_catalog.contract(
                route.skill_id,
                route.skill_version,
            )
        except KeyError as error:
            raise ValueError("Route references an unknown Skill I/O contract.") from error
        io_checks = (
            io_contract.skill_definition_hash == route.skill_definition_hash,
            io_contract.parameter_schema_hash == route.parameter_schema_hash,
            io_contract.execution_result_schema_hash
            == route.execution_result_schema_hash,
            io_contract.task_evaluation_schema_hash
            == route.task_evaluation_schema_hash,
            io_contract.semantic_content_hash == route.skill_io_contract_hash,
        )
        if not all(io_checks):
            raise ValueError("Route I/O hashes do not close over I/O catalog.")
        catalog_profiles = {
            (item.parameter_profile_id, item.parameter_profile_hash): item
            for item in schema_entry.parameter_profiles
        }
        for profile in route.allowed_parameter_profiles:
            if catalog_profiles.get(
                (profile.parameter_profile_id, profile.parameter_profile_hash)
            ) != profile:
                raise ValueError(
                    "Route parameter profile does not close over schema catalog."
                )


__all__ = [
    "AGENT_SKILL_CANCELLATION_POLICIES_V1",
    "AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1",
    "AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1",
    "AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_REVIEW_REF_HASH_DOMAIN_V2",
    "AGENT_SKILL_DEFINITION_CATALOG_HASH_DOMAIN_V2",
    "AGENT_SKILL_DEFINITION_HASH_DOMAIN_V2",
    "AGENT_SKILL_DEPLOYMENT_BINDING_REF_HASH_DOMAIN_V2",
    "AGENT_SKILL_KINDS_V1",
    "AGENT_SKILL_INTENDED_TARGET_KINDS_V1",
    "AGENT_SKILL_MODES_V1",
    "AGENT_SKILL_PRIVILEGE_TIERS_V1",
    "AGENT_SKILL_PROTOCOL_VERSION_V2",
    "AGENT_SKILL_ROUTE_HASH_DOMAIN_V2",
    "AGENT_SKILL_ROUTE_MANIFEST_ENTRY_HASH_DOMAIN_V2",
    "AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V2",
    "AGENT_SKILL_SCHEMA_CATALOG_HASH_DOMAIN_V2",
    "AGENT_SKILL_SCHEMA_DOCUMENT_HASH_DOMAIN_V2",
    "AGENT_SKILL_SCHEMA_ENTRY_HASH_DOMAIN_V2",
    "AGENT_SKILL_TARGET_KINDS_V1",
    "AMBIGUOUS_LIFECYCLE_SKILL_IDS_V1",
    "AgentSkillSchemaCatalog",
    "AgentSkillSchemaEntry",
    "CrossTargetDeploymentReviewRef",
    "MAX_AGENT_SKILL_JSON_DEPTH_V1",
    "MAX_AGENT_SKILL_TIMEOUT_MS_V1",
    "PRIVILEGED_LIFECYCLE_SKILL_IDS_V1",
    "SkillDefinition",
    "SkillDefinitionCatalog",
    "SkillDeploymentBindingRef",
    "SkillDeploymentSafetyEnvelopeRef",
    "SkillExecutionRoute",
    "SkillIOCatalog",
    "SkillParameterProfileRef",
    "SkillRouteManifest",
    "SkillRouteManifestEntry",
    "agent_skill_schema_document_hash",
    "validate_route_id",
    "validate_skill_id",
    "validate_skill_route_catalog_closure",
    "validate_version",
]
