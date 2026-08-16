"""Pure external wire contracts for the upper-agent-to-DexTeleop Skill boundary.

The records in this module describe Route discovery, immutable requests,
low-rate observation snapshots, and operation status.  They do not own a
worker, queue, authority lease, safety decision, motion permit, or dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .evaluation_contracts import (
    AGENT_SKILL_AGGREGATE_EVALUATION_RECEIPT_HASH_DOMAIN_V1,
    AGENT_SKILL_AGGREGATE_VERDICTS_V1,
    AGENT_SKILL_DISAGREEMENT_POLICIES_V1,
    AGENT_SKILL_EVALUATION_PLAN_HASH_DOMAIN_V1,
    AGENT_SKILL_EVALUATION_RECEIPT_HASH_DOMAIN_V1,
    AGENT_SKILL_EVALUATION_VERDICTS_V1,
    AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1,
    AGENT_SKILL_EVALUATOR_KINDS_V1,
    AGENT_SKILL_EVALUATOR_LOCATIONS_V1,
    AGENT_SKILL_EVALUATOR_ROLES_V1,
    AGENT_SKILL_EVALUATOR_SPEC_HASH_DOMAIN_V1,
    AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1,
)
from .skill_catalog import (
    AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1,
    AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1,
    AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_REVIEW_REF_HASH_DOMAIN_V1,
    AGENT_SKILL_DEFINITION_CATALOG_HASH_DOMAIN_V1,
    AGENT_SKILL_DEFINITION_HASH_DOMAIN_V1,
    AGENT_SKILL_DEPLOYMENT_BINDING_REF_HASH_DOMAIN_V1,
    AGENT_SKILL_INTENDED_TARGET_KINDS_V1,
    AGENT_SKILL_KINDS_V1,
    AGENT_SKILL_ROUTE_HASH_DOMAIN_V1,
    AGENT_SKILL_ROUTE_MANIFEST_ENTRY_HASH_DOMAIN_V1,
    AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V1,
    AGENT_SKILL_SCHEMA_CATALOG_HASH_DOMAIN_V1,
    AGENT_SKILL_SCHEMA_DOCUMENT_HASH_DOMAIN_V1,
    AGENT_SKILL_SCHEMA_ENTRY_HASH_DOMAIN_V1,
    validate_route_id,
    validate_skill_id,
    validate_version,
)

AGENT_SKILL_PROTOCOL_VERSION_V1 = "dexteleop.agent_skill.v1"
AGENT_SKILL_SCHEMA_SNAPSHOT_V1 = "dexteleop_agent_skill_schema_snapshot_v1"

AGENT_SKILL_TARGET_KINDS_V1 = ("null", "real", "sim")
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
AGENT_SKILL_CANCELLATION_POLICIES_V1 = (
    "cancel_only",
    "hold_on_cancel",
    "hold_on_cancel_disconnect_or_timeout",
    "hold_on_disconnect_or_timeout",
)
AGENT_SKILL_OPERATION_STATUSES_V1 = (
    "received",
    "validated",
    "preflighted",
    "awaiting_operator_confirmation",
    "admitted",
    "activated",
    "running",
    "awaiting_evaluation",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
)
AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1 = (
    "cancelled",
    "completed",
    "failed",
    "timed_out",
)
AGENT_SKILL_EXECUTION_OUTCOMES_V1 = (
    "cancelled",
    "completed",
    "failed",
    "pending",
    "timed_out",
)
AGENT_SKILL_EVALUATION_STATUSES_V1 = (
    "completed",
    "not_requested",
    "pending",
    "timed_out",
)
AGENT_SKILL_TASK_OUTCOMES_V1 = (
    "failed",
    "inconclusive",
    "not_assessed",
    "pending",
    "succeeded",
)
AGENT_CONTROL_REASON_CODES_V1 = (
    "agent_anomaly_detected",
    "agent_replan_requested",
    "agent_requested_pause",
    "agent_task_aborted",
    "agent_uncertain_state",
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

AGENT_SKILL_REQUEST_HASH_DOMAIN_V1 = "dexteleop.agent_skill.request.v1"
AGENT_SKILL_REQUEST_AUTHORIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.request_authorization.v1"
)
AGENT_SKILL_OBSERVATION_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.observation_snapshot.v1"
)
AGENT_SKILL_OPERATION_HASH_DOMAIN_V1 = "dexteleop.agent_skill.operation.v1"
AGENT_SKILL_EXECUTION_GRANT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.execution_grant.v1"
)
AGENT_SKILL_ACCESS_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.access_policy.v1"
)
AGENT_SKILL_ACCESS_DECISION_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.access_decision.v1"
)
AGENT_SKILL_PROTOCOL_SCHEMA_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.protocol_schema.v1"
)
AGENT_OWN_OPERATION_QUERY_HASH_DOMAIN_V1 = (
    "dexteleop.agent_control.own_operation_query.v1"
)
AGENT_OWN_OPERATION_CANCEL_REQUEST_HASH_DOMAIN_V1 = (
    "dexteleop.agent_control.own_operation_cancel_request.v1"
)
AGENT_OWN_DOMAIN_HOLD_REQUEST_HASH_DOMAIN_V1 = (
    "dexteleop.agent_control.own_domain_hold_request.v1"
)
AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.authoritative_port_receipt.v1"
)
AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1 = (
    "authoritative_port_receipt_v1"
)
AGENT_GATEWAY_SAFETY_PROFILE_SELECTION_FEATURE_V1 = (
    "safety_profile_selection_v1"
)
AGENT_GATEWAY_PORT_OPERATION_KINDS_V1 = ("activate", "cancel", "hold")


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


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
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


def _skill_id(value: object) -> str:
    return validate_skill_id(value)


def _skill_version(value: object, *, skill_id: str) -> str:
    del skill_id
    return validate_version(value, field_name="skill_version")


def _skill_id_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$",
    }


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


def _json_schema_object(
    *,
    schema_id: str,
    properties: Mapping[str, Any],
    required: tuple[str, ...],
    definitions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(required),
    }
    if definitions:
        schema["$defs"] = dict(definitions)
    return schema


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


@dataclass(frozen=True)
class AgentSkillRequest:
    """Strict client request for one advertised server-side skill."""

    client_id: str
    request_id: str
    idempotency_key: str
    client_sequence: int
    runtime_session_id: str
    skill_route_manifest_hash: str
    expected_station_revision: int
    skill_id: str
    skill_version: str
    skill_definition_hash: str
    route_id: str
    route_version: str
    route_hash: str
    deployment_binding_hash: str
    parameter_profile_id: str
    parameter_profile_hash: str
    parameters: Mapping[str, Any]
    observation_snapshot_id: str
    requested_mode: str
    requested_timeout_ms: int
    skill_execution_grant_id: str | None = None
    safety_profile_id: str | None = None
    safety_profile_hash: str | None = None
    deployment_safety_envelope_id: str | None = None
    deployment_safety_envelope_hash: str | None = None
    safety_constraint_set_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "client_id",
            "request_id",
            "idempotency_key",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "skill_version",
            "skill_definition_hash",
            "route_hash",
            "deployment_binding_hash",
            "parameter_profile_id",
            "parameter_profile_hash",
            "observation_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "skill_id", _skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            _skill_version(self.skill_version, skill_id=self.skill_id),
        )
        object.__setattr__(self, "route_id", validate_route_id(self.route_id))
        object.__setattr__(
            self,
            "route_version",
            validate_version(self.route_version, field_name="route_version"),
        )
        object.__setattr__(
            self,
            "client_sequence",
            _nonnegative_integer(
                self.client_sequence,
                field_name="client_sequence",
            ),
        )
        object.__setattr__(
            self,
            "expected_station_revision",
            _nonnegative_integer(
                self.expected_station_revision,
                field_name="expected_station_revision",
            ),
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_json_object(self.parameters, field_name="parameters"),
        )
        object.__setattr__(
            self,
            "requested_mode",
            _enum(
                self.requested_mode,
                field_name="requested_mode",
                allowed=AGENT_SKILL_MODES_V1,
            ),
        )
        object.__setattr__(
            self,
            "requested_timeout_ms",
            _positive_integer(
                self.requested_timeout_ms,
                field_name="requested_timeout_ms",
                maximum=MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            ),
        )
        object.__setattr__(
            self,
            "skill_execution_grant_id",
            _optional(
                self.skill_execution_grant_id,
                field_name="skill_execution_grant_id",
            ),
        )
        safety_fields = (
            "safety_profile_id",
            "safety_profile_hash",
            "deployment_safety_envelope_id",
            "deployment_safety_envelope_hash",
            "safety_constraint_set_hash",
        )
        normalized_safety = {
            field_name: _optional(
                getattr(self, field_name),
                field_name=field_name,
            )
            for field_name in safety_fields
        }
        present = {
            field_name
            for field_name, value in normalized_safety.items()
            if value is not None
        }
        if present and len(present) != len(safety_fields):
            raise ValueError(
                "Safety profile selection fields must be supplied together."
            )
        for field_name, value in normalized_safety.items():
            object.__setattr__(self, field_name, value)

    @property
    def has_safety_profile_selection(self) -> bool:
        return self.safety_profile_id is not None

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=AGENT_SKILL_REQUEST_HASH_DOMAIN_V1,
        )

    @property
    def authorization_content_hash(self) -> str:
        """Hash the operator-visible request before a grant ID exists."""

        payload = self.to_wire()
        payload.pop("skill_execution_grant_id")
        return canonical_content_hash(
            payload,
            domain=AGENT_SKILL_REQUEST_AUTHORIZATION_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, Any]:
        payload = {
            "client_id": self.client_id,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "client_sequence": self.client_sequence,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "expected_station_revision": self.expected_station_revision,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_definition_hash": self.skill_definition_hash,
            "route_id": self.route_id,
            "route_version": self.route_version,
            "route_hash": self.route_hash,
            "deployment_binding_hash": self.deployment_binding_hash,
            "parameter_profile_id": self.parameter_profile_id,
            "parameter_profile_hash": self.parameter_profile_hash,
            "parameters": _thaw_json(self.parameters),
            "observation_snapshot_id": self.observation_snapshot_id,
            "requested_mode": self.requested_mode,
            "requested_timeout_ms": self.requested_timeout_ms,
            "skill_execution_grant_id": self.skill_execution_grant_id,
        }
        if self.has_safety_profile_selection:
            payload.update(
                {
                    "safety_profile_id": self.safety_profile_id,
                    "safety_profile_hash": self.safety_profile_hash,
                    "deployment_safety_envelope_id": (
                        self.deployment_safety_envelope_id
                    ),
                    "deployment_safety_envelope_hash": (
                        self.deployment_safety_envelope_hash
                    ),
                    "safety_constraint_set_hash": (
                        self.safety_constraint_set_hash
                    ),
                }
            )
        return payload


_SKILL_REQUEST_FIELDS = frozenset(
    {
        "client_id",
        "request_id",
        "idempotency_key",
        "client_sequence",
        "runtime_session_id",
        "skill_route_manifest_hash",
        "expected_station_revision",
        "skill_id",
        "skill_version",
        "skill_definition_hash",
        "route_id",
        "route_version",
        "route_hash",
        "deployment_binding_hash",
        "parameter_profile_id",
        "parameter_profile_hash",
        "parameters",
        "observation_snapshot_id",
        "requested_mode",
        "requested_timeout_ms",
        "skill_execution_grant_id",
        "safety_profile_id",
        "safety_profile_hash",
        "deployment_safety_envelope_id",
        "deployment_safety_envelope_hash",
        "safety_constraint_set_hash",
    }
)
_REQUIRED_SKILL_REQUEST_FIELDS = _SKILL_REQUEST_FIELDS - {
    "skill_execution_grant_id",
    "safety_profile_id",
    "safety_profile_hash",
    "deployment_safety_envelope_id",
    "deployment_safety_envelope_hash",
    "safety_constraint_set_hash",
}


def agent_skill_request_from_json(payload: Mapping[str, Any]) -> AgentSkillRequest:
    """Parse an exact JSON object without granting or executing the skill."""

    if not isinstance(payload, Mapping):
        raise TypeError("Agent skill request payload must be a JSON object.")
    keys = frozenset(payload)
    unknown = sorted(keys - _SKILL_REQUEST_FIELDS)
    missing = sorted(_REQUIRED_SKILL_REQUEST_FIELDS - keys)
    if unknown:
        raise ValueError(f"Agent skill request contains unknown fields: {unknown}.")
    if missing:
        raise ValueError(f"Agent skill request is missing required fields: {missing}.")
    return AgentSkillRequest(
        client_id=payload["client_id"],
        request_id=payload["request_id"],
        idempotency_key=payload["idempotency_key"],
        client_sequence=payload["client_sequence"],
        runtime_session_id=payload["runtime_session_id"],
        skill_route_manifest_hash=payload["skill_route_manifest_hash"],
        expected_station_revision=payload["expected_station_revision"],
        skill_id=payload["skill_id"],
        skill_version=payload["skill_version"],
        skill_definition_hash=payload["skill_definition_hash"],
        route_id=payload["route_id"],
        route_version=payload["route_version"],
        route_hash=payload["route_hash"],
        deployment_binding_hash=payload["deployment_binding_hash"],
        parameter_profile_id=payload["parameter_profile_id"],
        parameter_profile_hash=payload["parameter_profile_hash"],
        parameters=payload["parameters"],
        observation_snapshot_id=payload["observation_snapshot_id"],
        requested_mode=payload["requested_mode"],
        requested_timeout_ms=payload["requested_timeout_ms"],
        skill_execution_grant_id=payload.get("skill_execution_grant_id"),
        safety_profile_id=payload.get("safety_profile_id"),
        safety_profile_hash=payload.get("safety_profile_hash"),
        deployment_safety_envelope_id=payload.get(
            "deployment_safety_envelope_id"
        ),
        deployment_safety_envelope_hash=payload.get(
            "deployment_safety_envelope_hash"
        ),
        safety_constraint_set_hash=payload.get("safety_constraint_set_hash"),
    )


@dataclass(frozen=True)
class AgentOwnOperationQuery:
    """Read-only query whose ownership is rechecked by the Gateway service."""

    client_id: str
    query_id: str
    runtime_session_id: str
    skill_route_manifest_hash: str
    operation_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "client_id",
            "query_id",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "operation_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=AGENT_OWN_OPERATION_QUERY_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "query_id": self.query_id,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True)
class AgentOwnOperationCancelRequest:
    """One idempotent request to cancel only an operation owned by the client."""

    client_id: str
    control_request_id: str
    idempotency_key: str
    runtime_session_id: str
    skill_route_manifest_hash: str
    operation_id: str
    observation_snapshot_id: str
    expected_station_revision: int
    reason_code: str

    def __post_init__(self) -> None:
        for field_name in (
            "client_id",
            "control_request_id",
            "idempotency_key",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "operation_id",
            "observation_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "expected_station_revision",
            _nonnegative_integer(
                self.expected_station_revision,
                field_name="expected_station_revision",
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(
                self.reason_code,
                field_name="reason_code",
                allowed=AGENT_CONTROL_REASON_CODES_V1,
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=AGENT_OWN_OPERATION_CANCEL_REQUEST_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "control_request_id": self.control_request_id,
            "idempotency_key": self.idempotency_key,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "operation_id": self.operation_id,
            "observation_snapshot_id": self.observation_snapshot_id,
            "expected_station_revision": self.expected_station_revision,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class AgentOwnDomainHoldRequest:
    """One-way HOLD request pinned to the service's exact domain and target."""

    client_id: str
    control_request_id: str
    idempotency_key: str
    runtime_session_id: str
    skill_route_manifest_hash: str
    control_domain_id: str
    target_binding_id: str
    observation_snapshot_id: str
    expected_station_revision: int
    reason_code: str

    def __post_init__(self) -> None:
        for field_name in (
            "client_id",
            "control_request_id",
            "idempotency_key",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "control_domain_id",
            "target_binding_id",
            "observation_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "expected_station_revision",
            _nonnegative_integer(
                self.expected_station_revision,
                field_name="expected_station_revision",
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(
                self.reason_code,
                field_name="reason_code",
                allowed=AGENT_CONTROL_REASON_CODES_V1,
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=AGENT_OWN_DOMAIN_HOLD_REQUEST_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "control_request_id": self.control_request_id,
            "idempotency_key": self.idempotency_key,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "control_domain_id": self.control_domain_id,
            "target_binding_id": self.target_binding_id,
            "observation_snapshot_id": self.observation_snapshot_id,
            "expected_station_revision": self.expected_station_revision,
            "reason_code": self.reason_code,
        }


_OWN_OPERATION_QUERY_FIELDS = frozenset(
    {
        "client_id",
        "query_id",
        "runtime_session_id",
        "skill_route_manifest_hash",
        "operation_id",
    }
)
_OWN_OPERATION_CANCEL_REQUEST_FIELDS = frozenset(
    {
        "client_id",
        "control_request_id",
        "idempotency_key",
        "runtime_session_id",
        "skill_route_manifest_hash",
        "operation_id",
        "observation_snapshot_id",
        "expected_station_revision",
        "reason_code",
    }
)
_OWN_DOMAIN_HOLD_REQUEST_FIELDS = frozenset(
    {
        "client_id",
        "control_request_id",
        "idempotency_key",
        "runtime_session_id",
        "skill_route_manifest_hash",
        "control_domain_id",
        "target_binding_id",
        "observation_snapshot_id",
        "expected_station_revision",
        "reason_code",
    }
)


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    fields: frozenset[str],
    payload_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{payload_name} payload must be a JSON object.")
    unknown = sorted(set(payload) - fields)
    missing = sorted(fields - set(payload))
    if unknown:
        raise ValueError(f"{payload_name} contains unknown fields: {unknown}.")
    if missing:
        raise ValueError(f"{payload_name} is missing required fields: {missing}.")


def agent_own_operation_query_from_json(
    payload: Mapping[str, Any],
) -> AgentOwnOperationQuery:
    _require_exact_fields(
        payload,
        fields=_OWN_OPERATION_QUERY_FIELDS,
        payload_name="Own-operation query",
    )
    return AgentOwnOperationQuery(**payload)


def agent_own_operation_cancel_request_from_json(
    payload: Mapping[str, Any],
) -> AgentOwnOperationCancelRequest:
    _require_exact_fields(
        payload,
        fields=_OWN_OPERATION_CANCEL_REQUEST_FIELDS,
        payload_name="Own-operation cancel request",
    )
    return AgentOwnOperationCancelRequest(**payload)


def agent_own_domain_hold_request_from_json(
    payload: Mapping[str, Any],
) -> AgentOwnDomainHoldRequest:
    _require_exact_fields(
        payload,
        fields=_OWN_DOMAIN_HOLD_REQUEST_FIELDS,
        payload_name="Own-domain HOLD request",
    )
    return AgentOwnDomainHoldRequest(**payload)


@dataclass(frozen=True)
class AgentObservationSnapshot:
    """Low-rate immutable planning snapshot; never a safety measurement."""

    snapshot_id: str
    runtime_session_id: str
    skill_route_manifest_hash: str
    station_revision: int
    captured_at: float
    available_at: float
    clock_domain_id: str
    authority_epoch: int
    execution_epoch: int
    dispatch_owner_generation: int
    lifecycle_state: str
    execution_gate: str
    active_operation_id: str | None
    source_binding_id: str | None
    producer_epoch: int | None
    target_binding_id: str
    execution_manifest_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    measurement_summary: Mapping[str, Any]
    measurement_freshness_ms: int
    measurement_valid: bool
    fault_class: str | None = None
    fault_rearmable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_id",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "clock_domain_id",
            "lifecycle_state",
            "execution_gate",
            "target_binding_id",
            "execution_manifest_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "station_revision",
            "authority_epoch",
            "execution_epoch",
            "dispatch_owner_generation",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        captured = _finite(self.captured_at, field_name="captured_at")
        available = _finite(self.available_at, field_name="available_at")
        if available < captured:
            raise ValueError("available_at cannot precede captured_at.")
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "available_at", available)
        for field_name in ("active_operation_id", "source_binding_id"):
            object.__setattr__(
                self,
                field_name,
                _optional(getattr(self, field_name), field_name=field_name),
            )
        if self.producer_epoch is not None:
            object.__setattr__(
                self,
                "producer_epoch",
                _nonnegative_integer(
                    self.producer_epoch,
                    field_name="producer_epoch",
                ),
            )
        object.__setattr__(
            self,
            "measurement_summary",
            _freeze_json_object(
                self.measurement_summary,
                field_name="measurement_summary",
            ),
        )
        object.__setattr__(
            self,
            "measurement_freshness_ms",
            _nonnegative_integer(
                self.measurement_freshness_ms,
                field_name="measurement_freshness_ms",
            ),
        )
        object.__setattr__(
            self,
            "measurement_valid",
            _boolean(self.measurement_valid, field_name="measurement_valid"),
        )
        fault_class = _optional(self.fault_class, field_name="fault_class")
        fault_rearmable = _boolean(
            self.fault_rearmable,
            field_name="fault_rearmable",
        )
        if fault_class is None and fault_rearmable:
            raise ValueError("fault_rearmable requires a fault_class.")
        object.__setattr__(self, "fault_class", fault_class)
        object.__setattr__(self, "fault_rearmable", fault_rearmable)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_OBSERVATION_SNAPSHOT_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "snapshot_id": self.snapshot_id,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "station_revision": self.station_revision,
            "captured_at": self.captured_at,
            "available_at": self.available_at,
            "clock_domain_id": self.clock_domain_id,
            "authority_epoch": self.authority_epoch,
            "execution_epoch": self.execution_epoch,
            "dispatch_owner_generation": self.dispatch_owner_generation,
            "lifecycle_state": self.lifecycle_state,
            "execution_gate": self.execution_gate,
            "active_operation_id": self.active_operation_id,
            "source_binding_id": self.source_binding_id,
            "producer_epoch": self.producer_epoch,
            "target_binding_id": self.target_binding_id,
            "execution_manifest_hash": self.execution_manifest_hash,
            "frame_graph_hash": self.frame_graph_hash,
            "tool_config_hash": self.tool_config_hash,
            "calibration_hash": self.calibration_hash,
            "measurement_summary": _thaw_json(self.measurement_summary),
            "measurement_freshness_ms": self.measurement_freshness_ms,
            "measurement_valid": self.measurement_valid,
            "fault_class": self.fault_class,
            "fault_rearmable": self.fault_rearmable,
        }
        if include_hash:
            payload["snapshot_hash"] = self.semantic_content_hash
        return payload
@dataclass(frozen=True)
class AgentSkillOperation:
    """Read-only projection separating execution from task evaluation."""

    operation_id: str
    request_hash: str
    runtime_session_id: str
    status: str
    reason_code: str
    station_revision_at_admission: int
    current_station_revision: int
    started_at: float
    updated_at: float
    deadline_at: float
    execution_outcome: str = "pending"
    evaluation_status: str = "not_requested"
    task_outcome: str = "not_assessed"
    lower_operation_id: str | None = None
    execution_receipt_refs: tuple[str, ...] = ()
    evaluator_receipt_refs: tuple[str, ...] = ()
    aggregate_evaluation_receipt_ref: str | None = None
    progress_summary: Mapping[str, Any] = field(default_factory=dict)
    execution_result: Mapping[str, Any] | None = None
    task_evaluation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "operation_id",
            "request_hash",
            "runtime_session_id",
            "reason_code",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "status",
            _enum(
                self.status,
                field_name="status",
                allowed=AGENT_SKILL_OPERATION_STATUSES_V1,
            ),
        )
        object.__setattr__(
            self,
            "execution_outcome",
            _enum(
                self.execution_outcome,
                field_name="execution_outcome",
                allowed=AGENT_SKILL_EXECUTION_OUTCOMES_V1,
            ),
        )
        object.__setattr__(
            self,
            "evaluation_status",
            _enum(
                self.evaluation_status,
                field_name="evaluation_status",
                allowed=AGENT_SKILL_EVALUATION_STATUSES_V1,
            ),
        )
        object.__setattr__(
            self,
            "task_outcome",
            _enum(
                self.task_outcome,
                field_name="task_outcome",
                allowed=AGENT_SKILL_TASK_OUTCOMES_V1,
            ),
        )
        for field_name in (
            "station_revision_at_admission",
            "current_station_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        started = _finite(self.started_at, field_name="started_at")
        updated = _finite(self.updated_at, field_name="updated_at")
        deadline = _finite(self.deadline_at, field_name="deadline_at")
        if updated < started:
            raise ValueError("updated_at cannot precede started_at.")
        if deadline <= started:
            raise ValueError("deadline_at must follow started_at.")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "deadline_at", deadline)
        object.__setattr__(
            self,
            "lower_operation_id",
            _optional(self.lower_operation_id, field_name="lower_operation_id"),
        )
        for field_name in (
            "execution_receipt_refs",
            "evaluator_receipt_refs",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique_sorted(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "aggregate_evaluation_receipt_ref",
            _optional(
                self.aggregate_evaluation_receipt_ref,
                field_name="aggregate_evaluation_receipt_ref",
            ),
        )
        object.__setattr__(
            self,
            "progress_summary",
            _freeze_json_object(
                self.progress_summary,
                field_name="progress_summary",
            ),
        )
        for field_name in ("execution_result", "task_evaluation"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _freeze_json_object(value, field_name=field_name),
                )

        if self.status in {
            "received",
            "validated",
            "preflighted",
            "awaiting_operator_confirmation",
            "admitted",
            "activated",
            "running",
        } and self.execution_outcome != "pending":
            raise ValueError("Pre-completion statuses require pending execution.")
        if self.status == "awaiting_evaluation" and (
            self.execution_outcome != "completed"
            or self.evaluation_status != "pending"
            or self.task_outcome != "pending"
            or self.execution_result is None
        ):
            raise ValueError(
                "awaiting_evaluation requires completed execution, a typed "
                "execution result, and pending evaluation/task outcome."
            )
        terminal_execution = {
            "cancelled": "cancelled",
            "timed_out": "timed_out",
        }.get(self.status)
        if (
            terminal_execution is not None
            and self.execution_outcome not in {
                terminal_execution,
                "completed",
            }
        ):
            raise ValueError(
                f"{self.status} status requires matching or already-completed "
                "execution_outcome."
            )
        if self.status == "completed" and (
            self.execution_outcome != "completed" or self.execution_result is None
        ):
            raise ValueError(
                "A completed operation requires completed execution and a typed "
                "execution_result."
            )
        if self.status == "failed" and self.execution_outcome not in {
            "failed",
            "completed",
        }:
            raise ValueError(
                "A failed operation requires failed or already-completed execution."
            )
        if self.execution_outcome != "completed" and self.execution_result is not None:
            raise ValueError(
                "execution_result requires execution_outcome='completed'."
            )

        if self.evaluation_status == "not_requested":
            if (
                self.task_outcome != "not_assessed"
                or self.evaluator_receipt_refs
                or self.aggregate_evaluation_receipt_ref is not None
                or self.task_evaluation is not None
            ):
                raise ValueError(
                    "not_requested evaluation requires task_outcome=not_assessed "
                    "and no evaluator result fields."
                )
        elif self.evaluation_status == "pending":
            if self.task_outcome != "pending" or self.task_evaluation is not None:
                raise ValueError(
                    "Pending evaluation requires pending task outcome and no "
                    "task_evaluation result."
                )
        elif self.evaluation_status == "completed":
            if self.task_outcome not in {
                "succeeded",
                "failed",
                "inconclusive",
            }:
                raise ValueError(
                    "Completed evaluation requires a terminal task outcome."
                )
            if (
                not self.evaluator_receipt_refs
                or self.aggregate_evaluation_receipt_ref is None
                or self.task_evaluation is None
            ):
                raise ValueError(
                    "Completed evaluation requires raw evaluator refs, an aggregate "
                    "receipt ref, and typed task_evaluation."
                )
        elif self.evaluation_status == "timed_out":
            if (
                self.task_outcome != "inconclusive"
                or self.aggregate_evaluation_receipt_ref is None
                or self.task_evaluation is None
            ):
                raise ValueError(
                    "Timed-out evaluation requires inconclusive task outcome and "
                    "typed aggregate evaluation evidence."
                )

    @property
    def terminal(self) -> bool:
        return self.status in AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_OPERATION_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "operation_id": self.operation_id,
            "request_hash": self.request_hash,
            "runtime_session_id": self.runtime_session_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "terminal": self.terminal,
            "execution_outcome": self.execution_outcome,
            "evaluation_status": self.evaluation_status,
            "task_outcome": self.task_outcome,
            "station_revision_at_admission": (
                self.station_revision_at_admission
            ),
            "current_station_revision": self.current_station_revision,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "deadline_at": self.deadline_at,
            "lower_operation_id": self.lower_operation_id,
            "execution_receipt_refs": list(self.execution_receipt_refs),
            "evaluator_receipt_refs": list(self.evaluator_receipt_refs),
            "aggregate_evaluation_receipt_ref": (
                self.aggregate_evaluation_receipt_ref
            ),
            "progress_summary": _thaw_json(self.progress_summary),
            "execution_result": (
                None
                if self.execution_result is None
                else _thaw_json(self.execution_result)
            ),
            "task_evaluation": (
                None
                if self.task_evaluation is None
                else _thaw_json(self.task_evaluation)
            ),
        }
        if include_hash:
            payload["operation_hash"] = self.semantic_content_hash
        return payload



def _agent_skill_route_wire_schemas() -> dict[str, Any]:
    """Return the post-capability Skill/Route wire surface."""

    nullable_string = _nullable({"type": "string", "minLength": 1})
    nullable_integer = _nullable({"type": "integer", "minimum": 0})
    authoritative_port_receipt_properties = {
        "receipt_id": {"type": "string", "minLength": 1},
        "port_id": {"type": "string", "minLength": 1},
        "operation_kind": {
            "type": "string",
            "enum": list(AGENT_GATEWAY_PORT_OPERATION_KINDS_V1),
        },
        "operation_id": nullable_string,
        "succeeded": {"type": "boolean"},
        "status": {"type": "string", "minLength": 1},
        "completed_at": {"type": "number"},
        "call_sequence": {"type": "integer", "minimum": 0},
        "lower_operation_id": nullable_string,
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "evidence_scope": {"const": "authoritative_lower_reference"},
        "authoritative_lower_receipt": {"const": True},
        "proves_motion_or_hold": {"type": "boolean"},
        "receipt_hash": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    }
    authoritative_port_receipt_body = {
        "type": "object",
        "additionalProperties": False,
        "properties": authoritative_port_receipt_properties,
        "required": list(authoritative_port_receipt_properties),
    }
    authoritative_port_receipt = _json_schema_object(
        schema_id="urn:dexteleop:agent-gateway:authoritative-port-receipt:v1",
        properties=authoritative_port_receipt_properties,
        required=tuple(authoritative_port_receipt_properties),
    )
    profile_ref = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "parameter_profile_id": {"type": "string", "minLength": 1},
            "parameter_profile_hash": {"type": "string", "minLength": 1},
            "parameter_schema_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "parameter_profile_id",
            "parameter_profile_hash",
            "parameter_schema_hash",
        ],
    }
    skill_definition = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": _skill_id_schema(),
            "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_kind": {
                "type": "string",
                "enum": ["inspection", "lifecycle", "motion", "policy_task"],
            },
            "summary": {"type": "string", "minLength": 1},
            "intended_target_kind": {
                "type": "string",
                "enum": list(AGENT_SKILL_INTENDED_TARGET_KINDS_V1),
            },
            "cross_target_deployment_policy": {
                "type": "string",
                "enum": list(
                    AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1
                ),
            },
            "controlled_component_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "parameter_semantics_id": {"type": "string", "minLength": 1},
            "execution_result_semantics_id": {
                "type": "string",
                "minLength": 1,
            },
            "task_outcome_semantics_id": {"type": "string", "minLength": 1},
            "entry_predicate_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "task_outcome_predicate_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "skill_definition_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "skill_id",
            "skill_version",
            "skill_kind",
            "summary",
            "intended_target_kind",
            "cross_target_deployment_policy",
            "controlled_component_ids",
            "parameter_semantics_id",
            "execution_result_semantics_id",
            "task_outcome_semantics_id",
            "entry_predicate_ids",
            "task_outcome_predicate_ids",
            "skill_definition_hash",
        ],
    }
    definition_catalog = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:definition-catalog:v1",
        properties={
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V1},
            "catalog_id": {"type": "string", "minLength": 1},
            "catalog_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_definitions": {
                "type": "array",
                "items": {"$ref": "#/$defs/SkillDefinition"},
            },
            "skill_definition_catalog_hash": {
                "type": "string",
                "minLength": 1,
            },
        },
        required=(
            "protocol_version",
            "catalog_id",
            "catalog_version",
            "skill_definitions",
            "skill_definition_catalog_hash",
        ),
        definitions={"SkillDefinition": skill_definition},
    )
    schema_entry = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": _skill_id_schema(),
            "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "parameter_schema": {"type": "object"},
            "parameter_schema_hash": {"type": "string", "minLength": 1},
            "execution_result_schema": {"type": "object"},
            "execution_result_schema_hash": {
                "type": "string",
                "minLength": 1,
            },
            "task_evaluation_schema": {"type": "object"},
            "task_evaluation_schema_hash": {
                "type": "string",
                "minLength": 1,
            },
            "parameter_profiles": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/SkillParameterProfileRef"},
            },
            "skill_schema_entry_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "skill_id",
            "skill_version",
            "skill_definition_hash",
            "parameter_schema",
            "parameter_schema_hash",
            "execution_result_schema",
            "execution_result_schema_hash",
            "task_evaluation_schema",
            "task_evaluation_schema_hash",
            "parameter_profiles",
            "skill_schema_entry_hash",
        ],
    }
    schema_catalog = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:schema-catalog:v1",
        properties={
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V1},
            "catalog_id": {"type": "string", "minLength": 1},
            "catalog_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "schema_entries": {
                "type": "array",
                "items": {"$ref": "#/$defs/AgentSkillSchemaEntry"},
            },
            "skill_schema_catalog_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "protocol_version",
            "catalog_id",
            "catalog_version",
            "schema_entries",
            "skill_schema_catalog_hash",
        ),
        definitions={
            "AgentSkillSchemaEntry": schema_entry,
            "SkillParameterProfileRef": profile_ref,
        },
    )
    route = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": _skill_id_schema(),
            "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "route_id": _skill_id_schema(),
            "route_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "parameter_schema_hash": {"type": "string", "minLength": 1},
            "execution_result_schema_hash": {"type": "string", "minLength": 1},
            "task_evaluation_schema_hash": {"type": "string", "minLength": 1},
            "allowed_parameter_profiles": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/SkillParameterProfileRef"},
            },
            "control_effect_class": {
                "type": "string",
                "enum": ["lifecycle", "motion", "read_only"],
            },
            "privilege_tier": {
                "type": "string",
                "enum": list(AGENT_SKILL_PRIVILEGE_TIERS_V1),
            },
            "supported_modes": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(AGENT_SKILL_MODES_V1)},
            },
            "target_kind": {
                "type": "string",
                "enum": list(AGENT_SKILL_TARGET_KINDS_V1),
            },
            "may_move_robot": {"type": "boolean"},
            "may_open_publication": {"type": "boolean"},
            "requires_operator_confirmation": {"type": "boolean"},
            "requires_deadman": {"type": "boolean"},
            "max_operation_duration_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            },
            "cancellation_policy": {
                "type": "string",
                "enum": list(AGENT_SKILL_CANCELLATION_POLICIES_V1),
            },
            "max_concurrent_operations": {"const": 1},
            "requires_task_evaluation": {"type": "boolean"},
            "route_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "skill_id",
            "skill_version",
            "skill_definition_hash",
            "route_id",
            "route_version",
            "parameter_schema_hash",
            "execution_result_schema_hash",
            "task_evaluation_schema_hash",
            "allowed_parameter_profiles",
            "control_effect_class",
            "privilege_tier",
            "supported_modes",
            "target_kind",
            "may_move_robot",
            "may_open_publication",
            "requires_operator_confirmation",
            "requires_deadman",
            "max_operation_duration_ms",
            "cancellation_policy",
            "max_concurrent_operations",
            "requires_task_evaluation",
            "route_hash",
        ],
    }
    cross_target_deployment_review = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "review_id": {"type": "string", "minLength": 1},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "route_hash": {"type": "string", "minLength": 1},
            "intended_target_kind": {
                "type": "string",
                "enum": ["real", "sim"],
            },
            "deployed_target_kind": {
                "type": "string",
                "enum": ["real", "sim"],
            },
            "compatibility_assessment_hash": {
                "type": "string",
                "minLength": 1,
            },
            "approval_receipt_hash": {"type": "string", "minLength": 1},
            "cross_target_deployment_review_ref_hash": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "review_id",
            "skill_definition_hash",
            "route_hash",
            "intended_target_kind",
            "deployed_target_kind",
            "compatibility_assessment_hash",
            "approval_receipt_hash",
            "cross_target_deployment_review_ref_hash",
        ],
    }
    binding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "deployment_binding_id": {"type": "string", "minLength": 1},
            "deployment_binding_hash": {"type": "string", "minLength": 1},
            "target_binding_id": {"type": "string", "minLength": 1},
            "target_kind": {
                "type": "string",
                "enum": list(AGENT_SKILL_TARGET_KINDS_V1),
            },
            "execution_manifest_hash": {"type": "string", "minLength": 1},
            "cross_target_deployment_review": {
                "oneOf": [
                    {"type": "null"},
                    {"$ref": "#/$defs/CrossTargetDeploymentReviewRef"},
                ],
            },
            "deployment_binding_ref_hash": {"type": "string", "minLength": 1},
            "safety_envelope_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "object_id": {"type": "string", "minLength": 1},
                        "schema": {
                            "const": "dexteleop.deployment_safety_envelope.v1"
                        },
                        "content_hash": {"type": "string", "minLength": 1},
                        "constraint_set_hash": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "allowed_route_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "required": [
                        "object_id",
                        "schema",
                        "content_hash",
                        "constraint_set_hash",
                        "allowed_route_ids",
                    ],
                },
            },
        },
        "required": [
            "deployment_binding_id",
            "deployment_binding_hash",
            "target_binding_id",
            "target_kind",
            "execution_manifest_hash",
            "cross_target_deployment_review",
            "deployment_binding_ref_hash",
        ],
    }
    route_entry = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "route": {"$ref": "#/$defs/SkillExecutionRoute"},
            "deployment_binding": {
                "$ref": "#/$defs/SkillDeploymentBindingRef"
            },
            "evaluation_plan_hash": nullable_string,
            "availability_hash": {"type": "string", "minLength": 1},
            "route_manifest_entry_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "route",
            "deployment_binding",
            "evaluation_plan_hash",
            "availability_hash",
            "route_manifest_entry_hash",
        ],
    }
    route_manifest = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:route-manifest:v1",
        properties={
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "control_domain_id": {"type": "string", "minLength": 1},
            "station_manifest_hash": {"type": "string", "minLength": 1},
            "execution_manifest_hash": {"type": "string", "minLength": 1},
            "skill_definition_catalog_hash": {"type": "string", "minLength": 1},
            "skill_schema_catalog_hash": {"type": "string", "minLength": 1},
            "target_binding_id": {"type": "string", "minLength": 1},
            "target_kind": {
                "type": "string",
                "enum": list(AGENT_SKILL_TARGET_KINDS_V1),
            },
            "current_mode": {
                "type": "string",
                "enum": list(AGENT_SKILL_MODES_V1),
            },
            "supported_transport_features": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "skill_routes": {
                "type": "array",
                "items": {"$ref": "#/$defs/SkillRouteManifestEntry"},
            },
            "explicit_exclusions": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "route_manifest_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "protocol_version",
            "runtime_session_id",
            "control_domain_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "skill_definition_catalog_hash",
            "skill_schema_catalog_hash",
            "target_binding_id",
            "target_kind",
            "current_mode",
            "supported_transport_features",
            "skill_routes",
            "explicit_exclusions",
            "route_manifest_hash",
        ),
        definitions={
            "SkillParameterProfileRef": profile_ref,
            "SkillExecutionRoute": route,
            "CrossTargetDeploymentReviewRef": cross_target_deployment_review,
            "SkillDeploymentBindingRef": binding,
            "SkillRouteManifestEntry": route_entry,
        },
    )
    request = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:request:v1",
        properties={
            "client_id": {"type": "string", "minLength": 1},
            "request_id": {"type": "string", "minLength": 1},
            "idempotency_key": {"type": "string", "minLength": 1},
            "client_sequence": {"type": "integer", "minimum": 0},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "expected_station_revision": {"type": "integer", "minimum": 0},
            "skill_id": _skill_id_schema(),
            "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "route_id": _skill_id_schema(),
            "route_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "route_hash": {"type": "string", "minLength": 1},
            "deployment_binding_hash": {"type": "string", "minLength": 1},
            "parameter_profile_id": {"type": "string", "minLength": 1},
            "parameter_profile_hash": {"type": "string", "minLength": 1},
            "parameters": {"type": "object"},
            "observation_snapshot_id": {"type": "string", "minLength": 1},
            "requested_mode": {
                "type": "string",
                "enum": list(AGENT_SKILL_MODES_V1),
            },
            "requested_timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            },
            "skill_execution_grant_id": nullable_string,
            "safety_profile_id": {"type": "string", "minLength": 1},
            "safety_profile_hash": {"type": "string", "minLength": 1},
            "deployment_safety_envelope_id": {
                "type": "string",
                "minLength": 1,
            },
            "deployment_safety_envelope_hash": {
                "type": "string",
                "minLength": 1,
            },
            "safety_constraint_set_hash": {
                "type": "string",
                "minLength": 1,
            },
        },
        required=tuple(sorted(_REQUIRED_SKILL_REQUEST_FIELDS)),
    )
    own_operation_query = _json_schema_object(
        schema_id="urn:dexteleop:agent-control:own-operation-query:v1",
        properties={
            "client_id": {"type": "string", "minLength": 1},
            "query_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "operation_id": {"type": "string", "minLength": 1},
        },
        required=tuple(sorted(_OWN_OPERATION_QUERY_FIELDS)),
    )
    own_operation_cancel_request = _json_schema_object(
        schema_id="urn:dexteleop:agent-control:own-operation-cancel:v1",
        properties={
            "client_id": {"type": "string", "minLength": 1},
            "control_request_id": {"type": "string", "minLength": 1},
            "idempotency_key": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "operation_id": {"type": "string", "minLength": 1},
            "observation_snapshot_id": {"type": "string", "minLength": 1},
            "expected_station_revision": {"type": "integer", "minimum": 0},
            "reason_code": {
                "type": "string",
                "enum": list(AGENT_CONTROL_REASON_CODES_V1),
            },
        },
        required=tuple(sorted(_OWN_OPERATION_CANCEL_REQUEST_FIELDS)),
    )
    own_domain_hold_request = _json_schema_object(
        schema_id="urn:dexteleop:agent-control:own-domain-hold:v1",
        properties={
            "client_id": {"type": "string", "minLength": 1},
            "control_request_id": {"type": "string", "minLength": 1},
            "idempotency_key": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "control_domain_id": {"type": "string", "minLength": 1},
            "target_binding_id": {"type": "string", "minLength": 1},
            "observation_snapshot_id": {"type": "string", "minLength": 1},
            "expected_station_revision": {"type": "integer", "minimum": 0},
            "reason_code": {
                "type": "string",
                "enum": list(AGENT_CONTROL_REASON_CODES_V1),
            },
        },
        required=tuple(sorted(_OWN_DOMAIN_HOLD_REQUEST_FIELDS)),
    )
    snapshot = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:observation-snapshot:v1",
        properties={
            "snapshot_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "station_revision": {"type": "integer", "minimum": 0},
            "captured_at": {"type": "number"},
            "available_at": {"type": "number"},
            "clock_domain_id": {"type": "string", "minLength": 1},
            "authority_epoch": {"type": "integer", "minimum": 0},
            "execution_epoch": {"type": "integer", "minimum": 0},
            "dispatch_owner_generation": {"type": "integer", "minimum": 0},
            "lifecycle_state": {"type": "string", "minLength": 1},
            "execution_gate": {"type": "string", "minLength": 1},
            "active_operation_id": nullable_string,
            "source_binding_id": nullable_string,
            "producer_epoch": nullable_integer,
            "target_binding_id": {"type": "string", "minLength": 1},
            "execution_manifest_hash": {"type": "string", "minLength": 1},
            "frame_graph_hash": {"type": "string", "minLength": 1},
            "tool_config_hash": {"type": "string", "minLength": 1},
            "calibration_hash": {"type": "string", "minLength": 1},
            "measurement_summary": {"type": "object"},
            "measurement_freshness_ms": {"type": "integer", "minimum": 0},
            "measurement_valid": {"type": "boolean"},
            "fault_class": nullable_string,
            "fault_rearmable": {"type": "boolean"},
            "snapshot_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "snapshot_id",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "station_revision",
            "captured_at",
            "available_at",
            "clock_domain_id",
            "authority_epoch",
            "execution_epoch",
            "dispatch_owner_generation",
            "lifecycle_state",
            "execution_gate",
            "active_operation_id",
            "source_binding_id",
            "producer_epoch",
            "target_binding_id",
            "execution_manifest_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "measurement_summary",
            "measurement_freshness_ms",
            "measurement_valid",
            "fault_class",
            "fault_rearmable",
            "snapshot_hash",
        ),
    )
    operation = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:operation:v1",
        properties={
            "operation_id": {"type": "string", "minLength": 1},
            "request_hash": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "status": {
                "type": "string",
                "enum": list(AGENT_SKILL_OPERATION_STATUSES_V1),
            },
            "reason_code": {"type": "string", "minLength": 1},
            "terminal": {"type": "boolean"},
            "execution_outcome": {
                "type": "string",
                "enum": list(AGENT_SKILL_EXECUTION_OUTCOMES_V1),
            },
            "evaluation_status": {
                "type": "string",
                "enum": list(AGENT_SKILL_EVALUATION_STATUSES_V1),
            },
            "task_outcome": {
                "type": "string",
                "enum": list(AGENT_SKILL_TASK_OUTCOMES_V1),
            },
            "station_revision_at_admission": {
                "type": "integer",
                "minimum": 0,
            },
            "current_station_revision": {"type": "integer", "minimum": 0},
            "started_at": {"type": "number"},
            "updated_at": {"type": "number"},
            "deadline_at": {"type": "number"},
            "lower_operation_id": nullable_string,
            "execution_receipt_refs": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "evaluator_receipt_refs": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "aggregate_evaluation_receipt_ref": nullable_string,
            "progress_summary": {"type": "object"},
            "execution_result": _nullable({"type": "object"}),
            "task_evaluation": _nullable({"type": "object"}),
            "operation_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "operation_id",
            "request_hash",
            "runtime_session_id",
            "status",
            "reason_code",
            "terminal",
            "execution_outcome",
            "evaluation_status",
            "task_outcome",
            "station_revision_at_admission",
            "current_station_revision",
            "started_at",
            "updated_at",
            "deadline_at",
            "lower_operation_id",
            "execution_receipt_refs",
            "evaluator_receipt_refs",
            "aggregate_evaluation_receipt_ref",
            "progress_summary",
            "execution_result",
            "task_evaluation",
            "operation_hash",
        ),
    )
    control_plane_response = _json_schema_object(
        schema_id="urn:dexteleop:agent-control:response:v1",
        properties={
            "accepted": {"type": "boolean"},
            "reason_code": {"type": "string", "minLength": 1},
            "request_hash": {"type": "string", "minLength": 1},
            "idempotent_replay": {"type": "boolean"},
            "operation": _nullable({"type": "object"}),
            "port_receipt": _nullable(
                {"$ref": "#/$defs/authoritative_port_receipt"}
            ),
        },
        required=(
            "accepted",
            "reason_code",
            "request_hash",
            "idempotent_replay",
            "operation",
        ),
        definitions={
            "authoritative_port_receipt": authoritative_port_receipt_body,
        },
    )
    grant = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:execution-grant:v1",
        properties={
            "grant_id": {"type": "string", "minLength": 1},
            "operator_authorization_receipt_id": {"type": "string", "minLength": 1},
            "client_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "skill_route_manifest_hash": {"type": "string", "minLength": 1},
            "control_domain_id": {"type": "string", "minLength": 1},
            "target_binding_id": {"type": "string", "minLength": 1},
            "skill_id": _skill_id_schema(),
            "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "route_id": _skill_id_schema(),
            "route_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
            "route_hash": {"type": "string", "minLength": 1},
            "deployment_binding_hash": {"type": "string", "minLength": 1},
            "parameter_profile_id": {"type": "string", "minLength": 1},
            "parameter_profile_hash": {"type": "string", "minLength": 1},
            "authorized_request_hash": nullable_string,
            "allowed_mode": {
                "type": "string",
                "enum": list(AGENT_SKILL_MODES_V1),
            },
            "control_effect_class": {
                "type": "string",
                "enum": ["lifecycle", "motion", "read_only"],
            },
            "may_move_robot": {"type": "boolean"},
            "max_invocations": {"type": "integer", "minimum": 1},
            "issued_at": {"type": "number"},
            "expires_at": {"type": "number"},
            "grant_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "grant_id",
            "operator_authorization_receipt_id",
            "client_id",
            "runtime_session_id",
            "skill_route_manifest_hash",
            "control_domain_id",
            "target_binding_id",
            "skill_id",
            "skill_version",
            "skill_definition_hash",
            "route_id",
            "route_version",
            "route_hash",
            "deployment_binding_hash",
            "parameter_profile_id",
            "parameter_profile_hash",
            "authorized_request_hash",
            "allowed_mode",
            "control_effect_class",
            "may_move_robot",
            "max_invocations",
            "issued_at",
            "expires_at",
            "grant_hash",
        ),
    )
    evaluator_spec = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evaluator_id": _skill_id_schema(),
            "evaluator_version": {
                "type": "string",
                "pattern": r"^[1-9][0-9]*$",
            },
            "evaluator_kind": {
                "type": "string",
                "enum": list(AGENT_SKILL_EVALUATOR_KINDS_V1),
            },
            "input_evidence_schema_hash": {"type": "string", "minLength": 1},
            "output_schema_hash": {"type": "string", "minLength": 1},
            "role": {
                "type": "string",
                "enum": list(AGENT_SKILL_EVALUATOR_ROLES_V1),
            },
            "execution_location": {
                "type": "string",
                "enum": list(AGENT_SKILL_EVALUATOR_LOCATIONS_V1),
            },
            "deadline_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            },
            "failure_policy": {
                "type": "string",
                "enum": list(AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1),
            },
            "evidence_domain_id": {"type": "string", "minLength": 1},
            "configuration_hash": {"type": "string", "minLength": 1},
            "evaluator_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "evaluator_id",
            "evaluator_version",
            "evaluator_kind",
            "input_evidence_schema_hash",
            "output_schema_hash",
            "role",
            "execution_location",
            "deadline_ms",
            "failure_policy",
            "evidence_domain_id",
            "configuration_hash",
            "evaluator_hash",
        ],
    }
    evaluation_plan = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:evaluation-plan:v1",
        properties={
            "evaluation_plan_id": _skill_id_schema(),
            "evaluation_plan_version": {
                "type": "string",
                "pattern": r"^[1-9][0-9]*$",
            },
            "skill_definition_hash": {"type": "string", "minLength": 1},
            "route_hash": {"type": "string", "minLength": 1},
            "deployment_binding_hash": {"type": "string", "minLength": 1},
            "evaluator_specs": {
                "type": "array",
                "items": {"$ref": "#/$defs/SkillEvaluatorSpec"},
            },
            "aggregation_policy_id": {"type": "string", "minLength": 1},
            "aggregation_policy_hash": {"type": "string", "minLength": 1},
            "required_evaluation_deadline_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_AGENT_SKILL_TIMEOUT_MS_V1,
            },
            "missing_evidence_policy": {
                "type": "string",
                "enum": list(AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1),
            },
            "disagreement_policy": {
                "type": "string",
                "enum": list(AGENT_SKILL_DISAGREEMENT_POLICIES_V1),
            },
            "task_outcome_schema_hash": {"type": "string", "minLength": 1},
            "evaluation_plan_hash": {"type": "string", "minLength": 1},
        },
        required=(
            "evaluation_plan_id",
            "evaluation_plan_version",
            "skill_definition_hash",
            "route_hash",
            "deployment_binding_hash",
            "evaluator_specs",
            "aggregation_policy_id",
            "aggregation_policy_hash",
            "required_evaluation_deadline_ms",
            "missing_evidence_policy",
            "disagreement_policy",
            "task_outcome_schema_hash",
            "evaluation_plan_hash",
        ),
        definitions={"SkillEvaluatorSpec": evaluator_spec},
    )
    evaluation_receipt_properties = {
        "operation_id": {"type": "string", "minLength": 1},
        "request_hash": {"type": "string", "minLength": 1},
        "skill_id": _skill_id_schema(),
        "skill_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
        "skill_definition_hash": {"type": "string", "minLength": 1},
        "route_id": _skill_id_schema(),
        "route_version": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
        "route_hash": {"type": "string", "minLength": 1},
        "deployment_binding_hash": {"type": "string", "minLength": 1},
        "evaluation_plan_hash": {"type": "string", "minLength": 1},
        "evaluator_id": _skill_id_schema(),
        "evaluator_version": {
            "type": "string",
            "pattern": r"^[1-9][0-9]*$",
        },
        "evaluator_hash": {"type": "string", "minLength": 1},
        "evaluator_kind": {
            "type": "string",
            "enum": list(AGENT_SKILL_EVALUATOR_KINDS_V1),
        },
        "evidence_domain_id": {"type": "string", "minLength": 1},
        "input_evidence_hash": {"type": "string", "minLength": 1},
        "evaluated_cut_id": {"type": "string", "minLength": 1},
        "evaluated_at": {"type": "number"},
        "verdict": {
            "type": "string",
            "enum": list(AGENT_SKILL_EVALUATION_VERDICTS_V1),
        },
        "reason_codes": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "evidence_refs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "score": _nullable({"type": "number"}),
        "score_schema_hash": nullable_string,
        "confidence": _nullable({"type": "number", "minimum": 0, "maximum": 1}),
        "confidence_semantics_id": nullable_string,
        "evaluation_receipt_hash": {"type": "string", "minLength": 1},
    }
    evaluation_receipt = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:evaluation-receipt:v1",
        properties=evaluation_receipt_properties,
        required=tuple(evaluation_receipt_properties),
    )
    string_array = {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    }
    aggregate_properties = {
        "operation_id": {"type": "string", "minLength": 1},
        "request_hash": {"type": "string", "minLength": 1},
        "evaluation_plan_hash": {"type": "string", "minLength": 1},
        "aggregation_policy_id": {"type": "string", "minLength": 1},
        "aggregation_policy_hash": {"type": "string", "minLength": 1},
        "consumed_evaluation_receipt_hashes": string_array,
        "ignored_evaluation_receipt_hashes": string_array,
        "missing_evaluator_ids": string_array,
        "verdict": {
            "type": "string",
            "enum": list(AGENT_SKILL_AGGREGATE_VERDICTS_V1),
        },
        "reason_codes": string_array,
        "generated_at": {"type": "number"},
        "aggregate_evaluation_receipt_hash": {
            "type": "string",
            "minLength": 1,
        },
    }
    aggregate_evaluation_receipt = _json_schema_object(
        schema_id="urn:dexteleop:agent-skill:aggregate-evaluation-receipt:v1",
        properties=aggregate_properties,
        required=tuple(aggregate_properties),
    )
    return {
        "skill_definition_catalog": definition_catalog,
        "agent_skill_schema_catalog": schema_catalog,
        "skill_route_manifest": route_manifest,
        "skill_request": request,
        "own_operation_query": own_operation_query,
        "own_operation_cancel_request": own_operation_cancel_request,
        "own_domain_hold_request": own_domain_hold_request,
        "control_plane_response": control_plane_response,
        "authoritative_port_receipt": authoritative_port_receipt,
        "observation_snapshot": snapshot,
        "skill_operation": operation,
        "skill_execution_grant": grant,
        "skill_evaluation_plan": evaluation_plan,
        "skill_evaluation_receipt": evaluation_receipt,
        "skill_aggregate_evaluation_receipt": aggregate_evaluation_receipt,
    }


def agent_skill_schema_snapshot_document() -> dict[str, Any]:
    """Return the portable schema snapshot consumed by upper Agent clients."""

    enums = {
        "ambiguous_lifecycle_skill_ids": list(
            AMBIGUOUS_LIFECYCLE_SKILL_IDS_V1
        ),
        "agent_control_reason_codes": list(AGENT_CONTROL_REASON_CODES_V1),
        "cancellation_policies": list(AGENT_SKILL_CANCELLATION_POLICIES_V1),
        "cross_target_deployment_policies": list(
            AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1
        ),
        "control_effect_classes": list(
            AGENT_SKILL_CONTROL_EFFECT_CLASSES_V1
        ),
        "aggregate_evaluation_verdicts": list(
            AGENT_SKILL_AGGREGATE_VERDICTS_V1
        ),
        "disagreement_policies": list(
            AGENT_SKILL_DISAGREEMENT_POLICIES_V1
        ),
        "evaluation_statuses": list(AGENT_SKILL_EVALUATION_STATUSES_V1),
        "evaluation_verdicts": list(AGENT_SKILL_EVALUATION_VERDICTS_V1),
        "evaluator_failure_policies": list(
            AGENT_SKILL_EVALUATOR_FAILURE_POLICIES_V1
        ),
        "evaluator_kinds": list(AGENT_SKILL_EVALUATOR_KINDS_V1),
        "evaluator_locations": list(AGENT_SKILL_EVALUATOR_LOCATIONS_V1),
        "evaluator_roles": list(AGENT_SKILL_EVALUATOR_ROLES_V1),
        "execution_outcomes": list(AGENT_SKILL_EXECUTION_OUTCOMES_V1),
        "missing_evidence_policies": list(
            AGENT_SKILL_MISSING_EVIDENCE_POLICIES_V1
        ),
        "modes": list(AGENT_SKILL_MODES_V1),
        "intended_target_kinds": list(
            AGENT_SKILL_INTENDED_TARGET_KINDS_V1
        ),
        "operation_statuses": list(AGENT_SKILL_OPERATION_STATUSES_V1),
        "privilege_tiers": list(AGENT_SKILL_PRIVILEGE_TIERS_V1),
        "privileged_lifecycle_skill_ids": list(
            PRIVILEGED_LIFECYCLE_SKILL_IDS_V1
        ),
        "target_kinds": list(AGENT_SKILL_TARGET_KINDS_V1),
        "skill_kinds": list(AGENT_SKILL_KINDS_V1),
        "task_outcomes": list(AGENT_SKILL_TASK_OUTCOMES_V1),
        "terminal_operation_statuses": list(
            AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
        ),
    }
    hash_domains = {
        "access_decision": AGENT_SKILL_ACCESS_DECISION_HASH_DOMAIN_V1,
        "access_policy": AGENT_SKILL_ACCESS_POLICY_HASH_DOMAIN_V1,
        "aggregate_evaluation_receipt": (
            AGENT_SKILL_AGGREGATE_EVALUATION_RECEIPT_HASH_DOMAIN_V1
        ),
        "definition": AGENT_SKILL_DEFINITION_HASH_DOMAIN_V1,
        "definition_catalog": AGENT_SKILL_DEFINITION_CATALOG_HASH_DOMAIN_V1,
        "cross_target_deployment_review_ref": (
            AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_REVIEW_REF_HASH_DOMAIN_V1
        ),
        "deployment_binding_ref": (
            AGENT_SKILL_DEPLOYMENT_BINDING_REF_HASH_DOMAIN_V1
        ),
        "execution_grant": AGENT_SKILL_EXECUTION_GRANT_HASH_DOMAIN_V1,
        "evaluation_plan": AGENT_SKILL_EVALUATION_PLAN_HASH_DOMAIN_V1,
        "evaluation_receipt": AGENT_SKILL_EVALUATION_RECEIPT_HASH_DOMAIN_V1,
        "evaluator_spec": AGENT_SKILL_EVALUATOR_SPEC_HASH_DOMAIN_V1,
        "observation_snapshot": (
            AGENT_SKILL_OBSERVATION_SNAPSHOT_HASH_DOMAIN_V1
        ),
        "own_domain_hold_request": (
            AGENT_OWN_DOMAIN_HOLD_REQUEST_HASH_DOMAIN_V1
        ),
        "own_operation_cancel_request": (
            AGENT_OWN_OPERATION_CANCEL_REQUEST_HASH_DOMAIN_V1
        ),
        "own_operation_query": AGENT_OWN_OPERATION_QUERY_HASH_DOMAIN_V1,
        "authoritative_port_receipt": (
            AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1
        ),
        "operation": AGENT_SKILL_OPERATION_HASH_DOMAIN_V1,
        "protocol_schema": AGENT_SKILL_PROTOCOL_SCHEMA_HASH_DOMAIN_V1,
        "request": AGENT_SKILL_REQUEST_HASH_DOMAIN_V1,
        "request_authorization": AGENT_SKILL_REQUEST_AUTHORIZATION_HASH_DOMAIN_V1,
        "route": AGENT_SKILL_ROUTE_HASH_DOMAIN_V1,
        "route_manifest": AGENT_SKILL_ROUTE_MANIFEST_HASH_DOMAIN_V1,
        "route_manifest_entry": (
            AGENT_SKILL_ROUTE_MANIFEST_ENTRY_HASH_DOMAIN_V1
        ),
        "schema_catalog": AGENT_SKILL_SCHEMA_CATALOG_HASH_DOMAIN_V1,
        "schema_document": AGENT_SKILL_SCHEMA_DOCUMENT_HASH_DOMAIN_V1,
        "schema_entry": AGENT_SKILL_SCHEMA_ENTRY_HASH_DOMAIN_V1,
    }
    schemas = _agent_skill_route_wire_schemas()
    protocol_surface = {
        "protocol_version": AGENT_SKILL_PROTOCOL_VERSION_V1,
        "enums": enums,
        "hash_domains": hash_domains,
        "limits": {
            "max_json_depth": MAX_AGENT_SKILL_JSON_DEPTH_V1,
            "max_timeout_ms": MAX_AGENT_SKILL_TIMEOUT_MS_V1,
        },
        "schemas": schemas,
    }
    document = {
        "schema": AGENT_SKILL_SCHEMA_SNAPSHOT_V1,
        **protocol_surface,
        "protocol_semantic_hash": canonical_content_hash(
            protocol_surface,
            domain=AGENT_SKILL_PROTOCOL_SCHEMA_HASH_DOMAIN_V1,
        ),
        "snapshot_hash_algorithm": "sha256_canonical_json_v1",
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    document["snapshot_hash"] = hashlib.sha256(encoded).hexdigest()
    return document


def agent_skill_schema_snapshot_json() -> str:
    """Serialize the portable schema snapshot deterministically."""

    return (
        json.dumps(
            agent_skill_schema_snapshot_document(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


__all__ = [
    "AGENT_CONTROL_REASON_CODES_V1",
    "AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1",
    "AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_HASH_DOMAIN_V1",
    "AGENT_GATEWAY_SAFETY_PROFILE_SELECTION_FEATURE_V1",
    "AGENT_GATEWAY_PORT_OPERATION_KINDS_V1",
    "AGENT_OWN_DOMAIN_HOLD_REQUEST_HASH_DOMAIN_V1",
    "AGENT_OWN_OPERATION_CANCEL_REQUEST_HASH_DOMAIN_V1",
    "AGENT_OWN_OPERATION_QUERY_HASH_DOMAIN_V1",
    "AGENT_SKILL_CANCELLATION_POLICIES_V1",
    "AGENT_SKILL_CROSS_TARGET_DEPLOYMENT_POLICIES_V1",
    "AGENT_SKILL_EVALUATION_STATUSES_V1",
    "AGENT_SKILL_EXECUTION_OUTCOMES_V1",
    "AGENT_SKILL_MODES_V1",
    "AGENT_SKILL_INTENDED_TARGET_KINDS_V1",
    "AGENT_SKILL_OPERATION_STATUSES_V1",
    "AGENT_SKILL_PRIVILEGE_TIERS_V1",
    "AGENT_SKILL_PROTOCOL_VERSION_V1",
    "AGENT_SKILL_SCHEMA_SNAPSHOT_V1",
    "AGENT_SKILL_TARGET_KINDS_V1",
    "AGENT_SKILL_TASK_OUTCOMES_V1",
    "AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1",
    "AMBIGUOUS_LIFECYCLE_SKILL_IDS_V1",
    "AgentOwnDomainHoldRequest",
    "AgentOwnOperationCancelRequest",
    "AgentOwnOperationQuery",
    "AgentObservationSnapshot",
    "AgentSkillOperation",
    "AgentSkillRequest",
    "MAX_AGENT_SKILL_JSON_DEPTH_V1",
    "MAX_AGENT_SKILL_TIMEOUT_MS_V1",
    "PRIVILEGED_LIFECYCLE_SKILL_IDS_V1",
    "agent_skill_request_from_json",
    "agent_own_domain_hold_request_from_json",
    "agent_own_operation_cancel_request_from_json",
    "agent_own_operation_query_from_json",
    "agent_skill_schema_snapshot_document",
    "agent_skill_schema_snapshot_json",
]
