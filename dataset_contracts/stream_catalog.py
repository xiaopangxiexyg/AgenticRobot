"""Pure v2.4 native-stream contracts.

The records in this module are independent from a device SDK, simulator,
runtime bundle, and physical storage implementation.  Payload bytes live in a
stream store; samples and episode manifests only bind stable identities,
timing, validity, and integrity evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .canonical_encoding import canonical_content_hash


NATIVE_STREAM_CONTRACT_SCHEMA_V1 = "dexteleop.native_stream_contract.v1"
NATIVE_STREAM_SAMPLE_SCHEMA_V1 = "dexteleop.native_stream_sample.v1"
NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_SCHEMA_V1 = (
    "dexteleop.native_stream_external_sample_intent.v1"
)
NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_SCHEMA_V1 = (
    "dexteleop.native_stream_external_artifact_receipt.v1"
)
NATIVE_STREAM_EPISODE_MANIFEST_SCHEMA_V1 = (
    "dexteleop.native_stream_episode_manifest.v1"
)
NATIVE_STREAM_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.native_stream_contract.semantic.v1"
)
NATIVE_STREAM_SAMPLE_HASH_DOMAIN_V1 = (
    "dexteleop.native_stream_sample.semantic.v1"
)
NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_HASH_DOMAIN_V1 = (
    "dexteleop.native_stream_external_sample_intent.semantic.v1"
)
NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.native_stream_external_artifact_receipt.semantic.v1"
)
NATIVE_STREAM_PAYLOAD_DOMAIN_V1 = "dexteleop.native_stream_payload.v1"
NATIVE_STREAM_EPISODE_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.native_stream_episode_manifest.semantic.v1"
)
NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1 = (
    "dexteleop.native_stream_contract.file.v1"
)
NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1 = (
    "dexteleop.native_stream_sample.file.v1"
)
NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.native_stream_episode_manifest.file.v1"
)
NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1 = Path(
    "meta/v24/native_stream/contracts"
)
NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1 = Path(
    "meta/v24/native_stream/episodes"
)

_SAFE_STREAM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ACCESS_ROLES = frozenset(
    {
        "actor_input",
        "actor_target",
        "critic_only",
        "control_private",
        "audit",
    }
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _optional_rate(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    normalized = _finite(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive when provided.")
    return normalized


def _sha256(value: object, *, field_name: str) -> str:
    normalized = _required(value, field_name=field_name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return normalized


def _dataset_relative_path(value: object, *, field_name: str) -> str:
    path = _required(value, field_name=field_name)
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"{field_name} must stay inside the dataset root.")
    return path


@dataclass(frozen=True)
class NativeStreamContract:
    stream_id: str
    modality: str
    producer_id: str
    value_schema: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    frame: str | None
    measured_clock_domain_id: str
    available_clock_domain_id: str
    nominal_rate_hz: float | None = None
    min_expected_rate_hz: float | None = None
    max_expected_rate_hz: float | None = None
    encoding: str = "canonical_cjson_v1"
    compression: str = "none"
    retention_policy: str = "retain_all_v1"
    validity_policy: str = "explicit_valid_stale_v1"
    access_role: str = "audit"
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=NATIVE_STREAM_CONTRACT_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "stream_id",
            "modality",
            "producer_id",
            "value_schema",
            "dtype",
            "units",
            "measured_clock_domain_id",
            "available_clock_domain_id",
            "encoding",
            "compression",
            "retention_policy",
            "validity_policy",
            "access_role",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        if not _SAFE_STREAM_ID.fullmatch(self.stream_id):
            raise ValueError(
                "stream_id must be path-safe and contain only letters, digits, "
                "dot, underscore, or hyphen."
            )
        shape = tuple(int(value) for value in self.shape)
        if any(value < 0 for value in shape):
            raise ValueError("Native stream shape dimensions must be non-negative.")
        object.__setattr__(self, "shape", shape)
        frame = None if self.frame is None else _required(self.frame, field_name="frame")
        object.__setattr__(self, "frame", frame)
        nominal = _optional_rate(
            self.nominal_rate_hz,
            field_name="nominal_rate_hz",
        )
        minimum = _optional_rate(
            self.min_expected_rate_hz,
            field_name="min_expected_rate_hz",
        )
        maximum = _optional_rate(
            self.max_expected_rate_hz,
            field_name="max_expected_rate_hz",
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("min_expected_rate_hz cannot exceed max_expected_rate_hz.")
        if nominal is not None:
            if minimum is not None and nominal < minimum:
                raise ValueError("nominal_rate_hz cannot be below the minimum.")
            if maximum is not None and nominal > maximum:
                raise ValueError("nominal_rate_hz cannot exceed the maximum.")
        object.__setattr__(self, "nominal_rate_hz", nominal)
        object.__setattr__(self, "min_expected_rate_hz", minimum)
        object.__setattr__(self, "max_expected_rate_hz", maximum)
        if self.access_role not in _ACCESS_ROLES:
            raise ValueError(f"Unsupported native stream access_role {self.access_role!r}.")
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=NATIVE_STREAM_CONTRACT_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "modality": self.modality,
            "producer_id": self.producer_id,
            "value_schema": self.value_schema,
            "shape": self.shape,
            "dtype": self.dtype,
            "units": self.units,
            "frame": self.frame,
            "measured_clock_domain_id": self.measured_clock_domain_id,
            "available_clock_domain_id": self.available_clock_domain_id,
            "nominal_rate_hz": self.nominal_rate_hz,
            "min_expected_rate_hz": self.min_expected_rate_hz,
            "max_expected_rate_hz": self.max_expected_rate_hz,
            "encoding": self.encoding,
            "compression": self.compression,
            "retention_policy": self.retention_policy,
            "validity_policy": self.validity_policy,
            "access_role": self.access_role,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class NativeStreamExternalSampleIntent:
    """Capture-time identity for a payload finalized by another writer."""

    stream_id: str
    stream_contract_hash: str
    sample_id: str
    sequence: int
    measured_at: float
    measured_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    source_payload_object_id: str
    payload_content_hash: str
    valid: bool
    stale: bool
    gap_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "stream_id",
            "sample_id",
            "measured_clock_domain_id",
            "available_clock_domain_id",
            "source_payload_object_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        if not _SAFE_STREAM_ID.fullmatch(self.stream_id):
            raise ValueError("external sample intent stream_id is not path-safe.")
        object.__setattr__(
            self,
            "stream_contract_hash",
            _sha256(
                self.stream_contract_hash,
                field_name="stream_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "payload_content_hash",
            _sha256(
                self.payload_content_hash,
                field_name="payload_content_hash",
            ),
        )
        sequence = int(self.sequence)
        if sequence < 0:
            raise ValueError("External sample sequence must be non-negative.")
        object.__setattr__(self, "sequence", sequence)
        for name in ("measured_at", "available_at"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), field_name=name),
            )
        gap_reason = (
            None
            if self.gap_reason is None
            else _required(self.gap_reason, field_name="gap_reason")
        )
        if gap_reason is not None and self.valid:
            raise ValueError("A valid external sample cannot carry gap_reason.")
        object.__setattr__(self, "gap_reason", gap_reason)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=(
                    NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_HASH_DOMAIN_V1
                ),
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        contract: NativeStreamContract,
        sample_id: str,
        sequence: int,
        measured_at: float,
        measured_clock_domain_id: str,
        available_at: float,
        available_clock_domain_id: str,
        source_payload_object_id: str,
        payload: Any,
        valid: bool = True,
        stale: bool = False,
        gap_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "NativeStreamExternalSampleIntent":
        if not isinstance(contract, NativeStreamContract):
            raise TypeError("contract must be NativeStreamContract.")
        measured_clock = _required(
            measured_clock_domain_id,
            field_name="measured_clock_domain_id",
        )
        available_clock = _required(
            available_clock_domain_id,
            field_name="available_clock_domain_id",
        )
        if measured_clock != contract.measured_clock_domain_id:
            raise ValueError(
                "External sample measured clock does not match stream contract."
            )
        if available_clock != contract.available_clock_domain_id:
            raise ValueError(
                "External sample available clock does not match stream contract."
            )
        try:
            expected_dtype = np.dtype(contract.dtype)
        except (TypeError, ValueError):
            expected_dtype = None
        if expected_dtype is not None:
            value = np.asarray(payload)
            if tuple(int(size) for size in value.shape) != contract.shape:
                raise ValueError(
                    "External stream payload shape does not match contract."
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    "External stream payload dtype does not match contract."
                )
        return cls(
            stream_id=contract.stream_id,
            stream_contract_hash=contract.semantic_content_hash,
            sample_id=sample_id,
            sequence=sequence,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock,
            available_at=available_at,
            available_clock_domain_id=available_clock,
            source_payload_object_id=source_payload_object_id,
            payload_content_hash=canonical_content_hash(
                payload,
                domain=NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
            ),
            valid=bool(valid),
            stale=bool(stale),
            gap_reason=gap_reason,
            metadata=dict(metadata or {}),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "stream_contract_hash": self.stream_contract_hash,
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "measured_at": self.measured_at,
            "measured_clock_domain_id": self.measured_clock_domain_id,
            "available_at": self.available_at,
            "available_clock_domain_id": self.available_clock_domain_id,
            "source_payload_object_id": self.source_payload_object_id,
            "payload_content_hash": self.payload_content_hash,
            "valid": bool(self.valid),
            "stale": bool(self.stale),
            "gap_reason": self.gap_reason,
            "metadata": self.metadata,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class NativeStreamExternalArtifactReceipt:
    """Durable dataset-relative artifact receipt for one external sample."""

    episode_uuid: str
    legacy_episode_index: int
    stream_id: str
    sample_id: str
    payload_content_hash: str
    stored_payload_object_id: str
    artifact_relative_path: str
    artifact_sha256: str
    artifact_byte_count: int
    encoding: str
    compression: str
    durable_at: float
    durable_clock_domain_id: str
    durable: bool = True
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_uuid",
            "stream_id",
            "sample_id",
            "stored_payload_object_id",
            "encoding",
            "compression",
            "durable_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        if "/" in self.episode_uuid or self.episode_uuid in {".", ".."}:
            raise ValueError("episode_uuid is not safe for external storage.")
        if not _SAFE_STREAM_ID.fullmatch(self.stream_id):
            raise ValueError("external artifact stream_id is not path-safe.")
        index = int(self.legacy_episode_index)
        if index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(
            self,
            "artifact_relative_path",
            _dataset_relative_path(
                self.artifact_relative_path,
                field_name="artifact_relative_path",
            ),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, field_name="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "payload_content_hash",
            _sha256(
                self.payload_content_hash,
                field_name="payload_content_hash",
            ),
        )
        byte_count = int(self.artifact_byte_count)
        if byte_count < 0:
            raise ValueError("artifact_byte_count must be non-negative.")
        object.__setattr__(self, "artifact_byte_count", byte_count)
        object.__setattr__(
            self,
            "durable_at",
            _finite(self.durable_at, field_name="durable_at"),
        )
        if not bool(self.durable):
            raise ValueError("External artifact receipt must be durable.")
        object.__setattr__(self, "durable", True)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=(
                    NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_HASH_DOMAIN_V1
                ),
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "stream_id": self.stream_id,
            "sample_id": self.sample_id,
            "payload_content_hash": self.payload_content_hash,
            "stored_payload_object_id": self.stored_payload_object_id,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_byte_count": self.artifact_byte_count,
            "encoding": self.encoding,
            "compression": self.compression,
            "durable_at": self.durable_at,
            "durable_clock_domain_id": self.durable_clock_domain_id,
            "durable": self.durable,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class NativeStreamSampleRecord:
    stream_id: str
    stream_contract_hash: str
    sample_id: str
    sequence: int
    measured_at: float
    measured_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    source_payload_object_id: str
    stored_payload_hash: str
    stored_payload_object_id: str
    valid: bool
    stale: bool
    gap_reason: str | None
    written_at: float
    written_clock_domain_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    external_artifact_relative_path: str | None = None
    external_artifact_sha256: str | None = None
    external_artifact_byte_count: int | None = None
    external_artifact_receipt_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=NATIVE_STREAM_SAMPLE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "stream_id",
            "stream_contract_hash",
            "sample_id",
            "measured_clock_domain_id",
            "available_clock_domain_id",
            "source_payload_object_id",
            "stored_payload_hash",
            "stored_payload_object_id",
            "written_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        if not _SAFE_STREAM_ID.fullmatch(self.stream_id):
            raise ValueError("sample stream_id is not path-safe.")
        object.__setattr__(
            self,
            "stream_contract_hash",
            _sha256(self.stream_contract_hash, field_name="stream_contract_hash"),
        )
        object.__setattr__(
            self,
            "stored_payload_hash",
            _sha256(self.stored_payload_hash, field_name="stored_payload_hash"),
        )
        sequence = int(self.sequence)
        if sequence < 0:
            raise ValueError("Native stream sequence must be non-negative.")
        object.__setattr__(self, "sequence", sequence)
        for name in ("measured_at", "available_at", "written_at"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), field_name=name),
            )
        gap_reason = (
            None
            if self.gap_reason is None
            else _required(self.gap_reason, field_name="gap_reason")
        )
        if gap_reason is not None and self.valid:
            raise ValueError("A valid stream sample cannot carry gap_reason.")
        object.__setattr__(self, "gap_reason", gap_reason)
        metadata = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "metadata", metadata)
        external_values = (
            self.external_artifact_relative_path,
            self.external_artifact_sha256,
            self.external_artifact_byte_count,
            self.external_artifact_receipt_hash,
        )
        if any(value is not None for value in external_values):
            if any(value is None for value in external_values):
                raise ValueError(
                    "External artifact sample fields must be provided together."
                )
            object.__setattr__(
                self,
                "external_artifact_relative_path",
                _dataset_relative_path(
                    self.external_artifact_relative_path,
                    field_name="external_artifact_relative_path",
                ),
            )
            object.__setattr__(
                self,
                "external_artifact_sha256",
                _sha256(
                    self.external_artifact_sha256,
                    field_name="external_artifact_sha256",
                ),
            )
            byte_count = int(self.external_artifact_byte_count)
            if byte_count < 0:
                raise ValueError(
                    "external_artifact_byte_count must be non-negative."
                )
            object.__setattr__(
                self,
                "external_artifact_byte_count",
                byte_count,
            )
            object.__setattr__(
                self,
                "external_artifact_receipt_hash",
                _sha256(
                    self.external_artifact_receipt_hash,
                    field_name="external_artifact_receipt_hash",
                ),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=NATIVE_STREAM_SAMPLE_HASH_DOMAIN_V1,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        contract: NativeStreamContract,
        sample_id: str,
        sequence: int,
        measured_at: float,
        measured_clock_domain_id: str,
        available_at: float,
        available_clock_domain_id: str,
        source_payload_object_id: str,
        payload: Any,
        valid: bool = True,
        stale: bool = False,
        gap_reason: str | None = None,
        written_at: float,
        written_clock_domain_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "NativeStreamSampleRecord":
        if not isinstance(contract, NativeStreamContract):
            raise TypeError("contract must be NativeStreamContract.")
        measured_clock = _required(
            measured_clock_domain_id,
            field_name="measured_clock_domain_id",
        )
        available_clock = _required(
            available_clock_domain_id,
            field_name="available_clock_domain_id",
        )
        if measured_clock != contract.measured_clock_domain_id:
            raise ValueError("Sample measured clock does not match stream contract.")
        if available_clock != contract.available_clock_domain_id:
            raise ValueError("Sample available clock does not match stream contract.")
        try:
            expected_dtype = np.dtype(contract.dtype)
        except (TypeError, ValueError):
            expected_dtype = None
        if expected_dtype is not None:
            value = np.asarray(payload)
            if tuple(int(size) for size in value.shape) != contract.shape:
                raise ValueError("Native stream payload shape does not match contract.")
            if value.dtype != expected_dtype:
                raise ValueError("Native stream payload dtype does not match contract.")
        payload_hash = canonical_content_hash(
            payload,
            domain=NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
        )
        return cls(
            stream_id=contract.stream_id,
            stream_contract_hash=contract.semantic_content_hash,
            sample_id=sample_id,
            sequence=sequence,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock,
            available_at=available_at,
            available_clock_domain_id=available_clock,
            source_payload_object_id=source_payload_object_id,
            stored_payload_hash=payload_hash,
            stored_payload_object_id=f"native-payload:{payload_hash}",
            valid=bool(valid),
            stale=bool(stale),
            gap_reason=gap_reason,
            written_at=written_at,
            written_clock_domain_id=written_clock_domain_id,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_external(
        cls,
        *,
        contract: NativeStreamContract,
        intent: NativeStreamExternalSampleIntent,
        receipt: NativeStreamExternalArtifactReceipt,
    ) -> "NativeStreamSampleRecord":
        if not isinstance(contract, NativeStreamContract):
            raise TypeError("contract must be NativeStreamContract.")
        if not isinstance(intent, NativeStreamExternalSampleIntent):
            raise TypeError("intent must be NativeStreamExternalSampleIntent.")
        if not isinstance(receipt, NativeStreamExternalArtifactReceipt):
            raise TypeError(
                "receipt must be NativeStreamExternalArtifactReceipt."
            )
        if (
            intent.stream_id != contract.stream_id
            or intent.stream_contract_hash != contract.semantic_content_hash
        ):
            raise ValueError(
                "External sample intent does not match its stream contract."
            )
        if (
            receipt.stream_id != intent.stream_id
            or receipt.sample_id != intent.sample_id
            or receipt.payload_content_hash != intent.payload_content_hash
        ):
            raise ValueError(
                "External artifact receipt does not match the sample intent."
            )
        if (
            receipt.encoding != contract.encoding
            or receipt.compression != contract.compression
        ):
            raise ValueError(
                "External artifact encoding/compression does not match "
                "the stream contract."
            )
        return cls(
            stream_id=intent.stream_id,
            stream_contract_hash=intent.stream_contract_hash,
            sample_id=intent.sample_id,
            sequence=intent.sequence,
            measured_at=intent.measured_at,
            measured_clock_domain_id=intent.measured_clock_domain_id,
            available_at=intent.available_at,
            available_clock_domain_id=intent.available_clock_domain_id,
            source_payload_object_id=intent.source_payload_object_id,
            stored_payload_hash=intent.payload_content_hash,
            stored_payload_object_id=receipt.stored_payload_object_id,
            valid=intent.valid,
            stale=intent.stale,
            gap_reason=intent.gap_reason,
            written_at=receipt.durable_at,
            written_clock_domain_id=receipt.durable_clock_domain_id,
            metadata=dict(intent.metadata),
            external_artifact_relative_path=receipt.artifact_relative_path,
            external_artifact_sha256=receipt.artifact_sha256,
            external_artifact_byte_count=receipt.artifact_byte_count,
            external_artifact_receipt_hash=receipt.semantic_content_hash,
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "stream_contract_hash": self.stream_contract_hash,
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "measured_at": self.measured_at,
            "measured_clock_domain_id": self.measured_clock_domain_id,
            "available_at": self.available_at,
            "available_clock_domain_id": self.available_clock_domain_id,
            "source_payload_object_id": self.source_payload_object_id,
            "stored_payload_hash": self.stored_payload_hash,
            "stored_payload_object_id": self.stored_payload_object_id,
            "valid": bool(self.valid),
            "stale": bool(self.stale),
            "gap_reason": self.gap_reason,
            "written_at": self.written_at,
            "written_clock_domain_id": self.written_clock_domain_id,
            "metadata": self.metadata,
        }
        if self.external_artifact_relative_path is not None:
            payload.update(
                {
                    "external_artifact_relative_path": (
                        self.external_artifact_relative_path
                    ),
                    "external_artifact_sha256": self.external_artifact_sha256,
                    "external_artifact_byte_count": (
                        self.external_artifact_byte_count
                    ),
                    "external_artifact_receipt_hash": (
                        self.external_artifact_receipt_hash
                    ),
                }
            )
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class NativeStreamEpisodeRange:
    stream_id: str
    stream_contract_hash: str
    index_relative_path: str
    index_sha256: str
    sample_count: int
    first_sequence: int | None
    last_sequence: int | None
    first_measured_at: float | None
    last_measured_at: float | None
    first_available_at: float | None
    last_available_at: float | None
    dropped_count: int = 0
    gap_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _required(self.stream_id, field_name="stream_id"))
        object.__setattr__(
            self,
            "stream_contract_hash",
            _sha256(self.stream_contract_hash, field_name="stream_contract_hash"),
        )
        path = _required(self.index_relative_path, field_name="index_relative_path")
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError("index_relative_path must stay inside the dataset root.")
        object.__setattr__(self, "index_relative_path", path)
        object.__setattr__(
            self,
            "index_sha256",
            _sha256(self.index_sha256, field_name="index_sha256"),
        )
        count = int(self.sample_count)
        dropped = int(self.dropped_count)
        if count < 0 or dropped < 0:
            raise ValueError("Stream sample/drop counts must be non-negative.")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "dropped_count", dropped)
        if count == 0:
            if any(
                value is not None
                for value in (
                    self.first_sequence,
                    self.last_sequence,
                    self.first_measured_at,
                    self.last_measured_at,
                    self.first_available_at,
                    self.last_available_at,
                )
            ):
                raise ValueError("An empty stream range cannot declare endpoints.")
        else:
            if self.first_sequence is None or self.last_sequence is None:
                raise ValueError("A non-empty stream range requires sequence endpoints.")
            if int(self.first_sequence) < 0 or int(self.last_sequence) < int(
                self.first_sequence
            ):
                raise ValueError("Stream sequence endpoints are invalid.")
            object.__setattr__(self, "first_sequence", int(self.first_sequence))
            object.__setattr__(self, "last_sequence", int(self.last_sequence))
            for name in (
                "first_measured_at",
                "last_measured_at",
                "first_available_at",
                "last_available_at",
            ):
                if getattr(self, name) is None:
                    raise ValueError("A non-empty stream range requires time endpoints.")
                object.__setattr__(
                    self,
                    name,
                    _finite(getattr(self, name), field_name=name),
                )
        object.__setattr__(
            self,
            "gap_reasons",
            tuple(sorted({_required(value, field_name="gap_reason") for value in self.gap_reasons})),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "stream_contract_hash": self.stream_contract_hash,
            "index_relative_path": self.index_relative_path,
            "index_sha256": self.index_sha256,
            "sample_count": self.sample_count,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "first_measured_at": self.first_measured_at,
            "last_measured_at": self.last_measured_at,
            "first_available_at": self.first_available_at,
            "last_available_at": self.last_available_at,
            "dropped_count": self.dropped_count,
            "gap_reasons": self.gap_reasons,
        }


@dataclass(frozen=True)
class NativeStreamEpisodeManifest:
    episode_uuid: str
    legacy_episode_index: int
    stream_ranges: tuple[NativeStreamEpisodeRange, ...]
    sealed_at: float
    sealed_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=NATIVE_STREAM_EPISODE_MANIFEST_SCHEMA_V1,
        init=False,
    )
    status: str = field(default="sealed", init=False)

    def __post_init__(self) -> None:
        episode = _required(self.episode_uuid, field_name="episode_uuid")
        if "/" in episode or episode in {".", ".."}:
            raise ValueError("episode_uuid is not safe for storage.")
        object.__setattr__(self, "episode_uuid", episode)
        index = int(self.legacy_episode_index)
        if index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        object.__setattr__(self, "legacy_episode_index", index)
        ranges = tuple(sorted(self.stream_ranges, key=lambda item: item.stream_id))
        if any(not isinstance(item, NativeStreamEpisodeRange) for item in ranges):
            raise TypeError("stream_ranges must contain NativeStreamEpisodeRange.")
        if len({item.stream_id for item in ranges}) != len(ranges):
            raise ValueError("Native stream ranges must have unique stream IDs.")
        if not ranges:
            raise ValueError("A native-stream episode manifest cannot be empty.")
        object.__setattr__(self, "stream_ranges", ranges)
        object.__setattr__(
            self,
            "sealed_at",
            _finite(self.sealed_at, field_name="sealed_at"),
        )
        object.__setattr__(
            self,
            "sealed_clock_domain_id",
            _required(
                self.sealed_clock_domain_id,
                field_name="sealed_clock_domain_id",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=NATIVE_STREAM_EPISODE_MANIFEST_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "status": self.status,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "stream_ranges": tuple(
                item.to_payload() for item in self.stream_ranges
            ),
            "sealed_at": self.sealed_at,
            "sealed_clock_domain_id": self.sealed_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


__all__ = [
    "NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1",
    "NATIVE_STREAM_CONTRACT_HASH_DOMAIN_V1",
    "NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1",
    "NATIVE_STREAM_CONTRACT_SCHEMA_V1",
    "NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_HASH_DOMAIN_V1",
    "NATIVE_STREAM_EXTERNAL_ARTIFACT_RECEIPT_SCHEMA_V1",
    "NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_HASH_DOMAIN_V1",
    "NATIVE_STREAM_EXTERNAL_SAMPLE_INTENT_SCHEMA_V1",
    "NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1",
    "NATIVE_STREAM_EPISODE_MANIFEST_HASH_DOMAIN_V1",
    "NATIVE_STREAM_EPISODE_MANIFEST_SCHEMA_V1",
    "NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1",
    "NATIVE_STREAM_PAYLOAD_DOMAIN_V1",
    "NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1",
    "NATIVE_STREAM_SAMPLE_HASH_DOMAIN_V1",
    "NATIVE_STREAM_SAMPLE_SCHEMA_V1",
    "NativeStreamContract",
    "NativeStreamExternalArtifactReceipt",
    "NativeStreamExternalSampleIntent",
    "NativeStreamEpisodeManifest",
    "NativeStreamEpisodeRange",
    "NativeStreamSampleRecord",
]
