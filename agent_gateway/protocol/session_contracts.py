"""Typed session capabilities and profile-owned resource claims for v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from ._contract_validation import (
    enum_value,
    identifier,
    positive_integer,
    required_text,
    unique_sorted_text,
)
from .artifact_contracts import AGENT_ARTIFACT_ACCESS_MECHANISMS_V2
from .event_contracts import MAX_AGENT_GATEWAY_EVENT_PAYLOAD_BYTES_V2
from .version import AGENT_SKILL_PROTOCOL_VERSION_V2


AGENT_GATEWAY_SESSION_CAPABILITIES_HASH_DOMAIN_V2 = (
    "dexteleop.agent_gateway.session_capabilities.v2"
)
AGENT_SKILL_RESOURCE_CLAIM_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.resource_claim.v2"
)
AGENT_SKILL_RESOURCE_ACCESS_MODES_V2 = (
    "shared_read",
    "shared_write",
    "exclusive",
)


@dataclass(frozen=True)
class SkillResourceClaim:
    """Trusted Route/profile declaration used by admission, never Agent input."""

    resource_id: str
    access_mode: str
    units: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            identifier(self.resource_id, field_name="resource_id"),
        )
        object.__setattr__(
            self,
            "access_mode",
            enum_value(
                self.access_mode,
                field_name="access_mode",
                allowed=AGENT_SKILL_RESOURCE_ACCESS_MODES_V2,
            ),
        )
        object.__setattr__(
            self,
            "units",
            positive_integer(self.units, field_name="units"),
        )
        if self.access_mode == "exclusive" and self.units != 1:
            raise ValueError("exclusive resource claims require units=1.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_SKILL_RESOURCE_CLAIM_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "resource_id": self.resource_id,
            "access_mode": self.access_mode,
            "units": self.units,
        }
        if include_hash:
            payload["resource_claim_hash"] = self.semantic_content_hash
        return payload


DEFAULT_AGENT_SKILL_RESOURCE_CLAIMS_V2 = (
    SkillResourceClaim(
        resource_id="gateway.operation",
        access_mode="exclusive",
    ),
)


@dataclass(frozen=True)
class AgentGatewaySessionCapabilities:
    """Typed, bounded capability document for one authenticated session."""

    runtime_session_id: str
    artifact_access_mechanisms: tuple[str, ...]
    max_control_request_bytes: int
    max_control_response_bytes: int
    max_inline_artifact_bytes: int
    max_artifact_bytes: int
    max_bundle_files: int
    max_artifact_refs_per_operation: int
    max_concurrent_uploads: int
    max_concurrent_downloads: int
    artifact_store_quota_bytes: int
    event_payload_max_bytes: int
    event_retention_count: int
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
        mechanisms = unique_sorted_text(
            self.artifact_access_mechanisms,
            field_name="artifact_access_mechanisms",
            allow_empty=False,
        )
        unsupported = sorted(
            set(mechanisms) - set(AGENT_ARTIFACT_ACCESS_MECHANISMS_V2)
        )
        if unsupported:
            raise ValueError(f"Unsupported Artifact mechanisms: {unsupported}.")
        if "chunked_rpc" not in mechanisms:
            raise ValueError("v2 sessions require chunked_rpc as the bounded fallback.")
        object.__setattr__(self, "artifact_access_mechanisms", mechanisms)
        for field_name in (
            "max_control_request_bytes",
            "max_control_response_bytes",
            "max_inline_artifact_bytes",
            "max_artifact_bytes",
            "max_bundle_files",
            "max_artifact_refs_per_operation",
            "max_concurrent_uploads",
            "max_concurrent_downloads",
            "artifact_store_quota_bytes",
            "event_payload_max_bytes",
            "event_retention_count",
        ):
            object.__setattr__(
                self,
                field_name,
                positive_integer(getattr(self, field_name), field_name=field_name),
            )
        if self.max_inline_artifact_bytes > self.max_artifact_bytes:
            raise ValueError(
                "max_inline_artifact_bytes cannot exceed max_artifact_bytes."
            )
        if self.max_artifact_bytes > self.artifact_store_quota_bytes:
            raise ValueError(
                "max_artifact_bytes cannot exceed artifact_store_quota_bytes."
            )
        if self.event_payload_max_bytes > MAX_AGENT_GATEWAY_EVENT_PAYLOAD_BYTES_V2:
            raise ValueError(
                "event_payload_max_bytes exceeds the protocol event bound."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_GATEWAY_SESSION_CAPABILITIES_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "runtime_session_id": self.runtime_session_id,
            "artifact_access_mechanisms": list(
                self.artifact_access_mechanisms
            ),
            "max_control_request_bytes": self.max_control_request_bytes,
            "max_control_response_bytes": self.max_control_response_bytes,
            "max_inline_artifact_bytes": self.max_inline_artifact_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_bundle_files": self.max_bundle_files,
            "max_artifact_refs_per_operation": (
                self.max_artifact_refs_per_operation
            ),
            "max_concurrent_uploads": self.max_concurrent_uploads,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "artifact_store_quota_bytes": self.artifact_store_quota_bytes,
            "event_payload_max_bytes": self.event_payload_max_bytes,
            "event_retention_count": self.event_retention_count,
        }
        if include_hash:
            payload["session_capabilities_hash"] = self.semantic_content_hash
        return payload


def agent_session_wire_schemas_v2() -> dict[str, dict[str, Any]]:
    """Return strict portable schemas for capabilities and resource claims."""

    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    resource_claim = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resource_id": {"type": "string", "minLength": 1},
            "access_mode": {
                "type": "string",
                "enum": list(AGENT_SKILL_RESOURCE_ACCESS_MODES_V2),
            },
            "units": {"type": "integer", "minimum": 1},
            "resource_claim_hash": digest,
        },
        "required": [
            "resource_id",
            "access_mode",
            "units",
            "resource_claim_hash",
        ],
    }
    capabilities = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "artifact_access_mechanisms": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": list(AGENT_ARTIFACT_ACCESS_MECHANISMS_V2),
                },
            },
            "max_control_request_bytes": {"type": "integer", "minimum": 1},
            "max_control_response_bytes": {"type": "integer", "minimum": 1},
            "max_inline_artifact_bytes": {"type": "integer", "minimum": 1},
            "max_artifact_bytes": {"type": "integer", "minimum": 1},
            "max_bundle_files": {"type": "integer", "minimum": 1},
            "max_artifact_refs_per_operation": {
                "type": "integer",
                "minimum": 1,
            },
            "max_concurrent_uploads": {"type": "integer", "minimum": 1},
            "max_concurrent_downloads": {"type": "integer", "minimum": 1},
            "artifact_store_quota_bytes": {"type": "integer", "minimum": 1},
            "event_payload_max_bytes": {"type": "integer", "minimum": 1},
            "event_retention_count": {"type": "integer", "minimum": 1},
            "session_capabilities_hash": digest,
        },
        "required": [
            "protocol_version",
            "runtime_session_id",
            "artifact_access_mechanisms",
            "max_control_request_bytes",
            "max_control_response_bytes",
            "max_inline_artifact_bytes",
            "max_artifact_bytes",
            "max_bundle_files",
            "max_artifact_refs_per_operation",
            "max_concurrent_uploads",
            "max_concurrent_downloads",
            "artifact_store_quota_bytes",
            "event_payload_max_bytes",
            "event_retention_count",
            "session_capabilities_hash",
        ],
    }
    return {
        "skill_resource_claim": resource_claim,
        "gateway_session_capabilities": capabilities,
    }


__all__ = [
    "AgentGatewaySessionCapabilities",
    "DEFAULT_AGENT_SKILL_RESOURCE_CLAIMS_V2",
    "SkillResourceClaim",
    "agent_session_wire_schemas_v2",
]
