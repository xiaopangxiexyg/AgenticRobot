"""Pure contracts for detachable dataset-modality artifacts.

The contracts describe content identity and verified storage operations only.
They intentionally contain no local filesystem paths, reader policy, or
collection lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


MODALITY_ARTIFACT_INVENTORY_SCHEMA_V1 = (
    "dexteleop_modality_artifact_inventory_v1"
)
MODALITY_STORAGE_TRANSFER_RECEIPT_SCHEMA_V1 = (
    "dexteleop_modality_storage_transfer_receipt_v1"
)
MODALITY_ARTIFACT_INVENTORY_HASH_DOMAIN_V1 = (
    "dexteleop.modality_artifact_inventory.v1"
)
MODALITY_STORAGE_TRANSFER_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.modality_storage_transfer_receipt.v1"
)

MODALITY_ARTIFACT_ROLES_V1 = ("manifest", "payload")
MODALITY_STORAGE_OPERATIONS_V1 = (
    "export_shadow",
    "restore_drill",
    "retire_hot_payloads",
    "retire_hot_module_tree",
    "rehydrate",
    "verify_external",
)
MODALITY_STORAGE_TRANSFER_STATUSES_V1 = (
    "planned",
    "complete",
    "failed",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ModalityArtifactContractError(ValueError):
    """Raised when an inventory or transfer receipt is malformed."""


def _sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_RE.fullmatch(text):
        raise ModalityArtifactContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return text


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise ModalityArtifactContractError(f"{field} is not a valid identifier")
    return text


def _logical_path(value: Any, *, field: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or text != path.as_posix()
        or path.as_posix() in {"", "."}
    ):
        raise ModalityArtifactContractError(
            f"{field} must be a normalized relative POSIX path"
        )
    return path.as_posix()


@dataclass(frozen=True)
class ModalityArtifactRecordV1:
    logical_path: str
    role: str
    size_bytes: int
    sha256: str
    episode_index: int | None = None
    side: str | None = None
    sensor_key: str | None = None
    frame_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_path",
            _logical_path(self.logical_path, field="logical_path"),
        )
        role = str(self.role or "").strip()
        if role not in MODALITY_ARTIFACT_ROLES_V1:
            raise ModalityArtifactContractError(f"unsupported artifact role: {role!r}")
        object.__setattr__(self, "role", role)
        size = int(self.size_bytes)
        if size < 0:
            raise ModalityArtifactContractError("size_bytes must be non-negative")
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "sha256", _sha256(self.sha256, field="sha256"))
        if self.episode_index is not None:
            index = int(self.episode_index)
            if index < 0:
                raise ModalityArtifactContractError(
                    "episode_index must be non-negative"
                )
            object.__setattr__(self, "episode_index", index)
        for field_name in ("side", "sensor_key"):
            value = getattr(self, field_name)
            if value is not None:
                text = str(value).strip()
                if not text:
                    raise ModalityArtifactContractError(
                        f"{field_name} cannot be empty when present"
                    )
                object.__setattr__(self, field_name, text)
        if self.frame_count is not None:
            count = int(self.frame_count)
            if count < 0:
                raise ModalityArtifactContractError(
                    "frame_count must be non-negative"
                )
            object.__setattr__(self, "frame_count", count)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModalityArtifactRecordV1":
        if not isinstance(payload, Mapping):
            raise ModalityArtifactContractError("artifact record must be an object")
        return cls(
            logical_path=payload.get("logical_path", ""),
            role=payload.get("role", ""),
            size_bytes=payload.get("size_bytes", -1),
            sha256=payload.get("sha256", ""),
            episode_index=payload.get("episode_index"),
            side=payload.get("side"),
            sensor_key=payload.get("sensor_key"),
            frame_count=payload.get("frame_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "episode_index": self.episode_index,
            "side": self.side,
            "sensor_key": self.sensor_key,
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class ModalityArtifactInventoryV1:
    modality_key: str
    source_dataset_base_fingerprint: str
    source_episode_catalog_sha256: str
    source_episode_count: int
    logical_prefixes: tuple[str, ...]
    artifacts: tuple[ModalityArtifactRecordV1, ...]
    schema_version: str = MODALITY_ARTIFACT_INVENTORY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != MODALITY_ARTIFACT_INVENTORY_SCHEMA_V1:
            raise ModalityArtifactContractError(
                f"unsupported inventory schema: {self.schema_version!r}"
            )
        object.__setattr__(
            self, "modality_key", _identifier(self.modality_key, field="modality_key")
        )
        object.__setattr__(
            self,
            "source_dataset_base_fingerprint",
            _sha256(
                self.source_dataset_base_fingerprint,
                field="source_dataset_base_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "source_episode_catalog_sha256",
            _sha256(
                self.source_episode_catalog_sha256,
                field="source_episode_catalog_sha256",
            ),
        )
        episode_count = int(self.source_episode_count)
        if episode_count < 0:
            raise ModalityArtifactContractError(
                "source_episode_count must be non-negative"
            )
        object.__setattr__(self, "source_episode_count", episode_count)
        prefixes = tuple(
            _logical_path(value, field="logical_prefixes")
            for value in self.logical_prefixes
        )
        if not prefixes or len(set(prefixes)) != len(prefixes):
            raise ModalityArtifactContractError(
                "logical_prefixes must be non-empty and unique"
            )
        object.__setattr__(self, "logical_prefixes", prefixes)
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ModalityArtifactContractError("inventory must contain artifacts")
        paths = [item.logical_path for item in artifacts]
        if len(set(paths)) != len(paths):
            raise ModalityArtifactContractError(
                "artifact logical_path values must be unique"
            )
        if paths != sorted(paths):
            raise ModalityArtifactContractError(
                "artifacts must be sorted by logical_path"
            )
        for item in artifacts:
            if not any(
                item.logical_path == prefix
                or item.logical_path.startswith(prefix.rstrip("/") + "/")
                for prefix in prefixes
            ):
                raise ModalityArtifactContractError(
                    f"artifact lies outside logical_prefixes: {item.logical_path}"
                )
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def file_count(self) -> int:
        return len(self.artifacts)

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.artifacts)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "modality_key": self.modality_key,
            "source_dataset_base_fingerprint": self.source_dataset_base_fingerprint,
            "source_episode_catalog_sha256": self.source_episode_catalog_sha256,
            "source_episode_count": self.source_episode_count,
            "logical_prefixes": list(self.logical_prefixes),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=MODALITY_ARTIFACT_INVENTORY_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModalityArtifactInventoryV1":
        if not isinstance(payload, Mapping):
            raise ModalityArtifactContractError("inventory must be an object")
        raw_artifacts = payload.get("artifacts", ())
        if not isinstance(raw_artifacts, (list, tuple)):
            raise ModalityArtifactContractError("artifacts must be a list")
        inventory = cls(
            schema_version=str(payload.get("schema_version", "")),
            modality_key=payload.get("modality_key", ""),
            source_dataset_base_fingerprint=payload.get(
                "source_dataset_base_fingerprint", ""
            ),
            source_episode_catalog_sha256=payload.get(
                "source_episode_catalog_sha256", ""
            ),
            source_episode_count=payload.get("source_episode_count", -1),
            logical_prefixes=tuple(payload.get("logical_prefixes", ())),
            artifacts=tuple(
                ModalityArtifactRecordV1.from_dict(item)
                for item in raw_artifacts
            ),
        )
        if int(payload.get("file_count", -1)) != inventory.file_count:
            raise ModalityArtifactContractError("inventory file_count mismatch")
        if int(payload.get("total_bytes", -1)) != inventory.total_bytes:
            raise ModalityArtifactContractError("inventory total_bytes mismatch")
        supplied_hash = _sha256(
            payload.get("semantic_content_hash", ""),
            field="semantic_content_hash",
        )
        if supplied_hash != inventory.semantic_content_hash:
            raise ModalityArtifactContractError(
                "inventory semantic_content_hash mismatch"
            )
        return inventory


@dataclass(frozen=True)
class ModalityStorageTransferReceiptV1:
    operation_id: str
    operation: str
    inventory_hash: str
    source_storage_id: str
    target_storage_id: str
    file_count: int
    total_bytes: int
    status: str
    independent_copy_verified: bool
    completed_at: str
    error: str = ""
    schema_version: str = MODALITY_STORAGE_TRANSFER_RECEIPT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != MODALITY_STORAGE_TRANSFER_RECEIPT_SCHEMA_V1:
            raise ModalityArtifactContractError(
                f"unsupported transfer receipt schema: {self.schema_version!r}"
            )
        for field_name in ("operation_id", "source_storage_id", "target_storage_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field=field_name),
            )
        operation = str(self.operation or "").strip()
        if operation not in MODALITY_STORAGE_OPERATIONS_V1:
            raise ModalityArtifactContractError(
                f"unsupported storage operation: {operation!r}"
            )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self,
            "inventory_hash",
            _sha256(self.inventory_hash, field="inventory_hash"),
        )
        for field_name in ("file_count", "total_bytes"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ModalityArtifactContractError(
                    f"{field_name} must be non-negative"
                )
            object.__setattr__(self, field_name, value)
        status = str(self.status or "").strip()
        if status not in MODALITY_STORAGE_TRANSFER_STATUSES_V1:
            raise ModalityArtifactContractError(
                f"unsupported transfer status: {status!r}"
            )
        object.__setattr__(self, "status", status)
        completed_at = str(self.completed_at or "").strip()
        if status == "complete" and not completed_at:
            raise ModalityArtifactContractError(
                "complete transfer receipt requires completed_at"
            )
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "error", str(self.error or "").strip())
        if status == "complete" and self.error:
            raise ModalityArtifactContractError(
                "complete transfer receipt cannot contain error"
            )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "inventory_hash": self.inventory_hash,
            "source_storage_id": self.source_storage_id,
            "target_storage_id": self.target_storage_id,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "status": self.status,
            "independent_copy_verified": bool(self.independent_copy_verified),
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=MODALITY_STORAGE_TRANSFER_RECEIPT_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ModalityStorageTransferReceiptV1":
        if not isinstance(payload, Mapping):
            raise ModalityArtifactContractError(
                "transfer receipt must be an object"
            )
        receipt = cls(
            schema_version=str(payload.get("schema_version", "")),
            operation_id=payload.get("operation_id", ""),
            operation=payload.get("operation", ""),
            inventory_hash=payload.get("inventory_hash", ""),
            source_storage_id=payload.get("source_storage_id", ""),
            target_storage_id=payload.get("target_storage_id", ""),
            file_count=payload.get("file_count", -1),
            total_bytes=payload.get("total_bytes", -1),
            status=payload.get("status", ""),
            independent_copy_verified=bool(
                payload.get("independent_copy_verified", False)
            ),
            completed_at=payload.get("completed_at", ""),
            error=payload.get("error", ""),
        )
        supplied_hash = _sha256(
            payload.get("semantic_content_hash", ""),
            field="semantic_content_hash",
        )
        if supplied_hash != receipt.semantic_content_hash:
            raise ModalityArtifactContractError(
                "transfer receipt semantic_content_hash mismatch"
            )
        return receipt


__all__ = [
    "MODALITY_ARTIFACT_INVENTORY_HASH_DOMAIN_V1",
    "MODALITY_ARTIFACT_INVENTORY_SCHEMA_V1",
    "MODALITY_ARTIFACT_ROLES_V1",
    "MODALITY_STORAGE_OPERATIONS_V1",
    "MODALITY_STORAGE_TRANSFER_RECEIPT_HASH_DOMAIN_V1",
    "MODALITY_STORAGE_TRANSFER_RECEIPT_SCHEMA_V1",
    "MODALITY_STORAGE_TRANSFER_STATUSES_V1",
    "ModalityArtifactContractError",
    "ModalityArtifactInventoryV1",
    "ModalityArtifactRecordV1",
    "ModalityStorageTransferReceiptV1",
]
