"""Transport-neutral finite Artifact contracts for Agent Skill protocol v2.

Artifact identity is immutable and contains no filesystem path or URL.  A
transport-specific location may appear only in a short-lived access grant.
These records describe data-plane facts; they do not own dataset, Experience,
Memory, device, or executor lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from ._contract_validation import (
    enum_value,
    finite_number,
    freeze_json_object,
    identifier,
    nonnegative_integer,
    normalized_relative_path,
    optional_text,
    positive_integer,
    required_text,
    sha256_digest,
    slot_identifier,
    thaw_json,
)
from .version import AGENT_SKILL_PROTOCOL_VERSION_V2


AGENT_ARTIFACT_REF_HASH_DOMAIN_V2 = "dexteleop.agent_skill.artifact_ref.v2"
AGENT_ARTIFACT_BUNDLE_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.artifact_bundle_manifest.v2"
)
AGENT_ARTIFACT_BINDING_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.artifact_binding.v2"
)
AGENT_ARTIFACT_ACCESS_GRANT_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.artifact_access_grant.v2"
)
AGENT_ARTIFACT_TRANSFER_RECEIPT_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.artifact_transfer_receipt.v2"
)
AGENT_ARTIFACT_MATERIALIZATION_RECEIPT_HASH_DOMAIN_V2 = (
    "dexteleop.agent_skill.artifact_materialization_receipt.v2"
)

AGENT_ARTIFACT_KINDS_V2 = ("blob", "bundle_manifest")
AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2 = ("read", "write")
AGENT_ARTIFACT_ACCESS_MECHANISMS_V2 = (
    "chunked_rpc",
    "local_path",
    "shared_fs",
    "unix_fd",
    "https",
    "object_store",
)
AGENT_ARTIFACT_TRANSFER_STATUSES_V2 = (
    "completed",
    "failed",
)


@dataclass(frozen=True)
class AgentArtifactRef:
    """Immutable, portable identity for one finite Artifact."""

    artifact_kind: str
    media_type: str
    encoding: str
    compression: str
    byte_length: int
    sha256: str
    schema_id: str | None = None
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        object.__setattr__(
            self,
            "artifact_kind",
            enum_value(
                self.artifact_kind,
                field_name="artifact_kind",
                allowed=AGENT_ARTIFACT_KINDS_V2,
            ),
        )
        for field_name in ("media_type", "encoding", "compression"):
            object.__setattr__(
                self,
                field_name,
                required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "schema_id",
            optional_text(self.schema_id, field_name="schema_id"),
        )
        object.__setattr__(
            self,
            "byte_length",
            nonnegative_integer(self.byte_length, field_name="byte_length"),
        )
        object.__setattr__(
            self,
            "sha256",
            sha256_digest(self.sha256, field_name="sha256"),
        )
        if self.artifact_kind == "bundle_manifest" and self.media_type != (
            "application/vnd.dexteleop.agent-artifact-bundle+json"
        ):
            raise ValueError(
                "bundle_manifest Artifacts require the canonical bundle media type."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_identity=False),
            domain=AGENT_ARTIFACT_REF_HASH_DOMAIN_V2,
        )

    @property
    def artifact_id(self) -> str:
        return f"artifact:{self.semantic_content_hash}"

    def to_wire(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "artifact_kind": self.artifact_kind,
            "media_type": self.media_type,
            "schema_id": self.schema_id,
            "encoding": self.encoding,
            "compression": self.compression,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }
        if include_identity:
            payload["artifact_id"] = self.artifact_id
            payload["artifact_ref_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "AgentArtifactRef":
        expected = {
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
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("ArtifactRef fields must match the v2 contract exactly.")
        artifact = cls(
            protocol_version=payload["protocol_version"],
            artifact_kind=payload["artifact_kind"],
            media_type=payload["media_type"],
            schema_id=payload["schema_id"],
            encoding=payload["encoding"],
            compression=payload["compression"],
            byte_length=payload["byte_length"],
            sha256=payload["sha256"],
        )
        if payload["artifact_id"] != artifact.artifact_id:
            raise ValueError("artifact_id does not match the immutable descriptor.")
        if payload["artifact_ref_hash"] != artifact.semantic_content_hash:
            raise ValueError("artifact_ref_hash does not match the descriptor.")
        return artifact


@dataclass(frozen=True)
class AgentArtifactBundleMember:
    """One normalized member of a finite Artifact bundle."""

    relative_path: str
    byte_length: int
    sha256: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            normalized_relative_path(
                self.relative_path,
                field_name="relative_path",
            ),
        )
        object.__setattr__(
            self,
            "byte_length",
            nonnegative_integer(self.byte_length, field_name="byte_length"),
        )
        object.__setattr__(
            self,
            "sha256",
            sha256_digest(self.sha256, field_name="sha256"),
        )
        object.__setattr__(
            self,
            "media_type",
            optional_text(self.media_type, field_name="media_type"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class AgentArtifactBundleManifest:
    """Canonical, content-addressed inventory for a multi-file Artifact."""

    members: tuple[AgentArtifactBundleMember, ...]
    protocol_version: str = AGENT_SKILL_PROTOCOL_VERSION_V2

    def __post_init__(self) -> None:
        if self.protocol_version != AGENT_SKILL_PROTOCOL_VERSION_V2:
            raise ValueError(
                f"Unsupported Agent Skill protocol: {self.protocol_version!r}."
            )
        members = tuple(self.members)
        if not members or any(
            not isinstance(member, AgentArtifactBundleMember) for member in members
        ):
            raise ValueError("members must contain AgentArtifactBundleMember values.")
        paths = tuple(member.relative_path for member in members)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("Bundle members must have unique, sorted paths.")
        object.__setattr__(self, "members", members)

    @property
    def file_count(self) -> int:
        return len(self.members)

    @property
    def total_bytes(self) -> int:
        return sum(member.byte_length for member in self.members)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_ARTIFACT_BUNDLE_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "members": [member.to_wire() for member in self.members],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }
        if include_hash:
            payload["bundle_manifest_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AgentArtifactBinding:
    """Bind one immutable Artifact to one semantic Skill I/O slot."""

    slot_id: str
    ordinal: int
    artifact: AgentArtifactRef
    production_receipt_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "slot_id",
            slot_identifier(self.slot_id, field_name="slot_id"),
        )
        object.__setattr__(
            self,
            "ordinal",
            nonnegative_integer(self.ordinal, field_name="ordinal"),
        )
        if not isinstance(self.artifact, AgentArtifactRef):
            raise TypeError("artifact must be an AgentArtifactRef.")
        object.__setattr__(
            self,
            "production_receipt_ref",
            optional_text(
                self.production_receipt_ref,
                field_name="production_receipt_ref",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_ARTIFACT_BINDING_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slot_id": self.slot_id,
            "ordinal": self.ordinal,
            "artifact": self.artifact.to_wire(),
            "production_receipt_ref": self.production_receipt_ref,
        }
        if include_hash:
            payload["artifact_binding_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "AgentArtifactBinding":
        expected = {
            "slot_id",
            "ordinal",
            "artifact",
            "production_receipt_ref",
            "artifact_binding_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError(
                "Artifact binding fields must match the v2 contract exactly."
            )
        binding = cls(
            slot_id=payload["slot_id"],
            ordinal=payload["ordinal"],
            artifact=AgentArtifactRef.from_wire(payload["artifact"]),
            production_receipt_ref=payload["production_receipt_ref"],
        )
        if payload["artifact_binding_hash"] != binding.semantic_content_hash:
            raise ValueError("artifact_binding_hash does not match its content.")
        return binding


@dataclass(frozen=True)
class AgentArtifactAccessGrant:
    """Short-lived, session-bound authorization for one transfer mechanism."""

    grant_id: str
    artifact: AgentArtifactRef
    direction: str
    mechanism: str
    client_id: str
    runtime_session_id: str
    issued_at: float
    expires_at: float
    max_uses: int
    locator: Mapping[str, Any]
    byte_offset: int = 0
    max_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("grant_id", "client_id", "runtime_session_id"):
            object.__setattr__(
                self,
                field_name,
                identifier(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.artifact, AgentArtifactRef):
            raise TypeError("artifact must be an AgentArtifactRef.")
        object.__setattr__(
            self,
            "direction",
            enum_value(
                self.direction,
                field_name="direction",
                allowed=AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2,
            ),
        )
        object.__setattr__(
            self,
            "mechanism",
            enum_value(
                self.mechanism,
                field_name="mechanism",
                allowed=AGENT_ARTIFACT_ACCESS_MECHANISMS_V2,
            ),
        )
        issued = finite_number(self.issued_at, field_name="issued_at")
        expires = finite_number(self.expires_at, field_name="expires_at")
        if expires <= issued:
            raise ValueError("expires_at must follow issued_at.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self,
            "max_uses",
            positive_integer(self.max_uses, field_name="max_uses"),
        )
        object.__setattr__(
            self,
            "byte_offset",
            nonnegative_integer(self.byte_offset, field_name="byte_offset"),
        )
        remaining = self.artifact.byte_length - self.byte_offset
        if remaining < 0:
            raise ValueError("byte_offset cannot exceed Artifact byte_length.")
        max_bytes = (
            remaining
            if self.max_bytes is None
            else nonnegative_integer(self.max_bytes, field_name="max_bytes")
        )
        if max_bytes > remaining:
            raise ValueError("max_bytes cannot exceed the remaining Artifact bytes.")
        object.__setattr__(self, "max_bytes", max_bytes)
        locator = freeze_json_object(self.locator, field_name="locator")
        if not locator:
            raise ValueError("locator must not be empty.")
        object.__setattr__(self, "locator", locator)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_ARTIFACT_ACCESS_GRANT_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "grant_id": self.grant_id,
            "artifact": self.artifact.to_wire(),
            "direction": self.direction,
            "mechanism": self.mechanism,
            "client_id": self.client_id,
            "runtime_session_id": self.runtime_session_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "max_uses": self.max_uses,
            "locator": thaw_json(self.locator),
            "byte_offset": self.byte_offset,
            "max_bytes": self.max_bytes,
        }
        if include_hash:
            payload["access_grant_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AgentArtifactTransferReceipt:
    """Evidence that one bounded transfer completed or failed."""

    receipt_id: str
    grant_id: str
    artifact_id: str
    client_id: str
    runtime_session_id: str
    direction: str
    status: str
    transferred_bytes: int
    verified_sha256: str | None
    completed_at: float
    reason_code: str

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_id",
            "grant_id",
            "artifact_id",
            "client_id",
            "runtime_session_id",
        ):
            object.__setattr__(
                self,
                field_name,
                identifier(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "direction",
            enum_value(
                self.direction,
                field_name="direction",
                allowed=AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2,
            ),
        )
        object.__setattr__(
            self,
            "status",
            enum_value(
                self.status,
                field_name="status",
                allowed=AGENT_ARTIFACT_TRANSFER_STATUSES_V2,
            ),
        )
        object.__setattr__(
            self,
            "transferred_bytes",
            nonnegative_integer(
                self.transferred_bytes,
                field_name="transferred_bytes",
            ),
        )
        object.__setattr__(
            self,
            "verified_sha256",
            (
                None
                if self.verified_sha256 is None
                else sha256_digest(
                    self.verified_sha256,
                    field_name="verified_sha256",
                )
            ),
        )
        object.__setattr__(
            self,
            "completed_at",
            finite_number(self.completed_at, field_name="completed_at"),
        )
        object.__setattr__(
            self,
            "reason_code",
            identifier(self.reason_code, field_name="reason_code"),
        )
        if self.status == "completed" and self.verified_sha256 is None:
            raise ValueError("A completed transfer requires verified_sha256.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_ARTIFACT_TRANSFER_RECEIPT_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "receipt_id": self.receipt_id,
            "grant_id": self.grant_id,
            "artifact_id": self.artifact_id,
            "client_id": self.client_id,
            "runtime_session_id": self.runtime_session_id,
            "direction": self.direction,
            "status": self.status,
            "transferred_bytes": self.transferred_bytes,
            "verified_sha256": self.verified_sha256,
            "completed_at": self.completed_at,
            "reason_code": self.reason_code,
        }
        if include_hash:
            payload["transfer_receipt_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class AgentArtifactMaterializationReceipt:
    """Upper-owned evidence that an Artifact has a separate durable copy."""

    receipt_id: str
    artifact_id: str
    client_id: str
    runtime_session_id: str
    durable_store_id: str
    durable_object_ref: str
    materialized_at: float
    verified_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_id",
            "artifact_id",
            "client_id",
            "runtime_session_id",
            "durable_store_id",
        ):
            object.__setattr__(
                self,
                field_name,
                identifier(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "durable_object_ref",
            required_text(
                self.durable_object_ref,
                field_name="durable_object_ref",
                maximum_length=2048,
            ),
        )
        object.__setattr__(
            self,
            "materialized_at",
            finite_number(self.materialized_at, field_name="materialized_at"),
        )
        object.__setattr__(
            self,
            "verified_sha256",
            sha256_digest(self.verified_sha256, field_name="verified_sha256"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=AGENT_ARTIFACT_MATERIALIZATION_RECEIPT_HASH_DOMAIN_V2,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "receipt_id": self.receipt_id,
            "artifact_id": self.artifact_id,
            "client_id": self.client_id,
            "runtime_session_id": self.runtime_session_id,
            "durable_store_id": self.durable_store_id,
            "durable_object_ref": self.durable_object_ref,
            "materialized_at": self.materialized_at,
            "verified_sha256": self.verified_sha256,
        }
        if include_hash:
            payload["materialization_receipt_hash"] = self.semantic_content_hash
        return payload


def agent_artifact_wire_schemas_v2() -> dict[str, dict[str, Any]]:
    """Return strict portable JSON schemas for the finite Artifact surface."""

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
            "artifact_kind": {"type": "string", "enum": list(AGENT_ARTIFACT_KINDS_V2)},
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
            "slot_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
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
    bundle_member = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {"type": "string", "minLength": 1},
            "byte_length": {"type": "integer", "minimum": 0},
            "sha256": digest,
            "media_type": nullable_string,
        },
        "required": ["relative_path", "byte_length", "sha256", "media_type"],
    }
    bundle_manifest = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"const": AGENT_SKILL_PROTOCOL_VERSION_V2},
            "members": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/AgentArtifactBundleMember"},
            },
            "file_count": {"type": "integer", "minimum": 1},
            "total_bytes": {"type": "integer", "minimum": 0},
            "bundle_manifest_hash": digest,
        },
        "required": [
            "protocol_version",
            "members",
            "file_count",
            "total_bytes",
            "bundle_manifest_hash",
        ],
        "$defs": {"AgentArtifactBundleMember": bundle_member},
    }
    access_grant = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "grant_id": {"type": "string", "minLength": 1},
            "artifact": {"$ref": "#/$defs/AgentArtifactRef"},
            "direction": {
                "type": "string",
                "enum": list(AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2),
            },
            "mechanism": {
                "type": "string",
                "enum": list(AGENT_ARTIFACT_ACCESS_MECHANISMS_V2),
            },
            "client_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "issued_at": {"type": "number"},
            "expires_at": {"type": "number"},
            "max_uses": {"type": "integer", "minimum": 1},
            "locator": {"type": "object"},
            "byte_offset": {"type": "integer", "minimum": 0},
            "max_bytes": {"type": "integer", "minimum": 0},
            "access_grant_hash": digest,
        },
        "required": [
            "grant_id",
            "artifact",
            "direction",
            "mechanism",
            "client_id",
            "runtime_session_id",
            "issued_at",
            "expires_at",
            "max_uses",
            "locator",
            "byte_offset",
            "max_bytes",
            "access_grant_hash",
        ],
        "$defs": {"AgentArtifactRef": artifact_ref},
    }
    transfer_receipt = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "receipt_id": {"type": "string", "minLength": 1},
            "grant_id": {"type": "string", "minLength": 1},
            "artifact_id": {"type": "string", "minLength": 1},
            "client_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "direction": {
                "type": "string",
                "enum": list(AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2),
            },
            "status": {
                "type": "string",
                "enum": list(AGENT_ARTIFACT_TRANSFER_STATUSES_V2),
            },
            "transferred_bytes": {"type": "integer", "minimum": 0},
            "verified_sha256": {
                "anyOf": [digest, {"type": "null"}],
            },
            "completed_at": {"type": "number"},
            "reason_code": {"type": "string", "minLength": 1},
            "transfer_receipt_hash": digest,
        },
        "required": [
            "receipt_id",
            "grant_id",
            "artifact_id",
            "client_id",
            "runtime_session_id",
            "direction",
            "status",
            "transferred_bytes",
            "verified_sha256",
            "completed_at",
            "reason_code",
            "transfer_receipt_hash",
        ],
    }
    materialization_receipt = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "receipt_id": {"type": "string", "minLength": 1},
            "artifact_id": {"type": "string", "minLength": 1},
            "client_id": {"type": "string", "minLength": 1},
            "runtime_session_id": {"type": "string", "minLength": 1},
            "durable_store_id": {"type": "string", "minLength": 1},
            "durable_object_ref": {"type": "string", "minLength": 1},
            "materialized_at": {"type": "number"},
            "verified_sha256": digest,
            "materialization_receipt_hash": digest,
        },
        "required": [
            "receipt_id",
            "artifact_id",
            "client_id",
            "runtime_session_id",
            "durable_store_id",
            "durable_object_ref",
            "materialized_at",
            "verified_sha256",
            "materialization_receipt_hash",
        ],
    }
    return {
        "agent_artifact_ref": artifact_ref,
        "agent_artifact_binding": {
            **artifact_binding,
            "$defs": {"AgentArtifactRef": artifact_ref},
        },
        "agent_artifact_bundle_manifest": bundle_manifest,
        "agent_artifact_access_grant": access_grant,
        "agent_artifact_transfer_receipt": transfer_receipt,
        "agent_artifact_materialization_receipt": materialization_receipt,
    }


__all__ = [
    "AGENT_ARTIFACT_ACCESS_DIRECTIONS_V2",
    "AGENT_ARTIFACT_ACCESS_MECHANISMS_V2",
    "AGENT_ARTIFACT_KINDS_V2",
    "AgentArtifactAccessGrant",
    "AgentArtifactBinding",
    "AgentArtifactBundleManifest",
    "AgentArtifactBundleMember",
    "AgentArtifactMaterializationReceipt",
    "AgentArtifactRef",
    "AgentArtifactTransferReceipt",
    "agent_artifact_wire_schemas_v2",
]
