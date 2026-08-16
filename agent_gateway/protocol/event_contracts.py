"""Replayable, bounded notification contracts for Agent Skill protocol v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from ._contract_validation import (
    canonical_json_size,
    enum_value,
    finite_number,
    freeze_json_object,
    identifier,
    nonnegative_integer,
    optional_text,
    thaw_json,
)
from .artifact_contracts import AgentArtifactBinding
from .version import AGENT_SKILL_PROTOCOL_VERSION_V2


AGENT_GATEWAY_EVENT_HASH_DOMAIN_V2 = "dexteleop.agent_gateway.event.v2"
AGENT_GATEWAY_EVENT_BATCH_HASH_DOMAIN_V2 = (
    "dexteleop.agent_gateway.event_batch.v2"
)
AGENT_GATEWAY_EVENT_KINDS_V2 = (
    "artifact_ready",
    "capabilities_changed",
    "manifest_changed",
    "operation_progress",
    "operation_terminal",
    "operation_updated",
    "session_closing",
)
MAX_AGENT_GATEWAY_EVENT_PAYLOAD_BYTES_V2 = 65_536


@dataclass(frozen=True)
class AgentGatewayEvent:
    """One at-least-once notification; operation query remains authoritative."""

    runtime_session_id: str
    sequence: int
    event_kind: str
    emitted_at: float
    payload: Mapping[str, Any]
    operation_id: str | None = None
    operation_hash: str | None = None
    artifact_binding: AgentArtifactBinding | None = None
    payload_schema_id: str | None = None
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        object.__setattr__(
            self,
            "runtime_session_id",
            identifier(self.runtime_session_id, field_name="runtime_session_id"),
        )
        object.__setattr__(
            self,
            "sequence",
            nonnegative_integer(self.sequence, field_name="sequence"),
        )
        object.__setattr__(
            self,
            "event_kind",
            enum_value(
                self.event_kind,
                field_name="event_kind",
                allowed=AGENT_GATEWAY_EVENT_KINDS_V2,
            ),
        )
        object.__setattr__(
            self,
            "emitted_at",
            finite_number(self.emitted_at, field_name="emitted_at"),
        )
        operation_id = optional_text(self.operation_id, field_name="operation_id")
        operation_hash = optional_text(
            self.operation_hash,
            field_name="operation_hash",
        )
        if (operation_id is None) != (operation_hash is None):
            raise ValueError("operation_id and operation_hash must be paired.")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "operation_hash", operation_hash)
        if self.artifact_binding is not None and not isinstance(
            self.artifact_binding,
            AgentArtifactBinding,
        ):
            raise TypeError("artifact_binding must be AgentArtifactBinding or None.")
        if self.event_kind == "artifact_ready" and self.artifact_binding is None:
            raise ValueError("artifact_ready requires an Artifact binding.")
        if self.event_kind != "artifact_ready" and self.artifact_binding is not None:
            raise ValueError("Only artifact_ready may carry an Artifact binding.")
        object.__setattr__(
            self,
            "payload_schema_id",
            optional_text(
                self.payload_schema_id,
                field_name="payload_schema_id",
            ),
        )
        payload = freeze_json_object(self.payload, field_name="payload")
        if canonical_json_size(payload, field_name="payload") > (
            MAX_AGENT_GATEWAY_EVENT_PAYLOAD_BYTES_V2
        ):
            raise ValueError("Event payload exceeds the v2 control-plane bound.")
        object.__setattr__(self, "payload", payload)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_identity=False),
            domain=AGENT_GATEWAY_EVENT_HASH_DOMAIN_V2,
        )

    @property
    def event_id(self) -> str:
        return f"event:{self.semantic_content_hash}"

    def to_wire(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "runtime_session_id": self.runtime_session_id,
            "sequence": self.sequence,
            "event_kind": self.event_kind,
            "emitted_at": self.emitted_at,
            "operation_id": self.operation_id,
            "operation_hash": self.operation_hash,
            "artifact_binding": (
                None
                if self.artifact_binding is None
                else self.artifact_binding.to_wire()
            ),
            "payload_schema_id": self.payload_schema_id,
            "payload": thaw_json(self.payload),
        }
        if include_identity:
            payload["event_id"] = self.event_id
            payload["event_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AgentGatewayEventBatch:
    """Bounded replay response with explicit cursor-gap semantics."""

    runtime_session_id: str
    requested_after_sequence: int
    oldest_available_sequence: int
    latest_sequence: int
    gap_detected: bool
    events: tuple[AgentGatewayEvent, ...]
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        object.__setattr__(
            self,
            "runtime_session_id",
            identifier(self.runtime_session_id, field_name="runtime_session_id"),
        )
        for field_name in (
            "requested_after_sequence",
            "oldest_available_sequence",
            "latest_sequence",
        ):
            object.__setattr__(
                self,
                field_name,
                nonnegative_integer(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.gap_detected, bool):
            raise TypeError("gap_detected must be a boolean.")
        if self.oldest_available_sequence > self.latest_sequence + 1:
            raise ValueError("oldest_available_sequence is inconsistent.")
        if self.requested_after_sequence > self.latest_sequence:
            raise ValueError(
                "requested_after_sequence cannot exceed latest_sequence."
            )
        events = tuple(self.events)
        if any(not isinstance(item, AgentGatewayEvent) for item in events):
            raise TypeError("events must contain AgentGatewayEvent values.")
        sequences = tuple(item.sequence for item in events)
        if sequences != tuple(sorted(sequences)) or len(set(sequences)) != len(
            sequences
        ):
            raise ValueError("events must have unique, increasing sequences.")
        if any(
            item.runtime_session_id != self.runtime_session_id for item in events
        ):
            raise ValueError("All events must belong to the batch session.")
        if events and (
            events[0].sequence <= self.requested_after_sequence
            or events[-1].sequence > self.latest_sequence
        ):
            raise ValueError("Event sequences lie outside the requested batch.")
        expected_gap = self.requested_after_sequence + 1 < (
            self.oldest_available_sequence
        )
        if self.gap_detected != expected_gap:
            raise ValueError("gap_detected does not match the retained cursor window.")
        object.__setattr__(self, "events", events)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_GATEWAY_EVENT_BATCH_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "runtime_session_id": self.runtime_session_id,
            "requested_after_sequence": self.requested_after_sequence,
            "oldest_available_sequence": self.oldest_available_sequence,
            "latest_sequence": self.latest_sequence,
            "gap_detected": self.gap_detected,
            "events": [event.to_wire() for event in self.events],
        }
        if include_hash:
            payload["event_batch_hash"] = self.semantic_content_hash
        return payload


def agent_event_wire_schemas_v2() -> dict[str, dict[str, Any]]:
    """Return strict portable schemas for bounded replayable events."""

    nullable_string = {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "null"},
        ]
    }
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    artifact_ref = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "artifact_id": {"type": "string", "minLength": 1},
            "artifact_kind": {"type": "string", "enum": ["blob", "bundle_manifest"]},
            "media_type": {"type": "string", "minLength": 1},
            "schema_id": nullable_string,
            "encoding": {"type": "string", "minLength": 1},
            "compression": {"type": "string", "minLength": 1},
            "byte_length": {"type": "integer", "minimum": 0},
            "sha256": digest,
            "artifact_ref_hash": digest,
        },
        "required": [
            "protocol_version",
            "artifact_id",
            "artifact_kind",
            "media_type",
            "schema_id",
            "encoding",
            "compression",
            "byte_length",
            "sha256",
            "artifact_ref_hash",
        ],
    }
    artifact_binding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot_id": {"type": "string", "minLength": 1},
            "ordinal": {"type": "integer", "minimum": 0},
            "artifact": {"$ref": "#/$defs/AgentArtifactRef"},
            "production_receipt_ref": nullable_string,
            "artifact_binding_hash": digest,
        },
        "required": [
            "slot_id",
            "ordinal",
            "artifact",
            "production_receipt_ref",
            "artifact_binding_hash",
        ],
    }
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "sequence": {"type": "integer", "minimum": 0},
            "event_kind": {
                "type": "string",
                "enum": list(AGENT_GATEWAY_EVENT_KINDS_V2),
            },
            "emitted_at": {"type": "number"},
            "operation_id": nullable_string,
            "operation_hash": nullable_string,
            "artifact_binding": {
                "anyOf": [
                    {"$ref": "#/$defs/AgentArtifactBinding"},
                    {"type": "null"},
                ]
            },
            "payload_schema_id": nullable_string,
            "payload": {"type": "object"},
            "event_id": {"type": "string", "minLength": 1},
            "event_hash": digest,
        },
        "required": [
            "protocol_version",
            "runtime_session_id",
            "sequence",
            "event_kind",
            "emitted_at",
            "operation_id",
            "operation_hash",
            "artifact_binding",
            "payload_schema_id",
            "payload",
            "event_id",
            "event_hash",
        ],
    }
    event_with_defs = {
        **event,
        "$defs": {
            "AgentArtifactRef": artifact_ref,
            "AgentArtifactBinding": artifact_binding,
        },
    }
    batch = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "requested_after_sequence": {"type": "integer", "minimum": 0},
            "oldest_available_sequence": {"type": "integer", "minimum": 0},
            "latest_sequence": {"type": "integer", "minimum": 0},
            "gap_detected": {"type": "boolean"},
            "events": {
                "type": "array",
                "items": {"$ref": "#/$defs/AgentGatewayEvent"},
            },
            "event_batch_hash": digest,
        },
        "required": [
            "protocol_version",
            "runtime_session_id",
            "requested_after_sequence",
            "oldest_available_sequence",
            "latest_sequence",
            "gap_detected",
            "events",
            "event_batch_hash",
        ],
        "$defs": {
            "AgentArtifactRef": artifact_ref,
            "AgentArtifactBinding": artifact_binding,
            "AgentGatewayEvent": event,
        },
    }
    return {
        "gateway_event": event_with_defs,
        "gateway_event_batch": batch,
    }


__all__ = [
    "AgentGatewayEvent",
    "AgentGatewayEventBatch",
    "agent_event_wire_schemas_v2",
]
