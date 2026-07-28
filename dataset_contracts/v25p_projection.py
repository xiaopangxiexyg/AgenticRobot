"""Explicit compatibility-projection contracts for native v2.5-p episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash


V25P_COMPATIBILITY_PROJECTION_RECEIPT_SCHEMA_V1 = (
    "dexteleop.v25p.compatibility_projection_receipt.v1"
)
V25P_COMPATIBILITY_PROJECTION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.compatibility_projection_receipt.semantic.v1"
)
V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1 = (
    "dexteleop.v25p.compatibility_projection_receipt.file.v1"
)
V25P_COMPATIBILITY_PROJECTION_STATUSES_V1 = frozenset(
    {"complete", "failed", "skipped"}
)


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _hash(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return result


def _optional_hash(value: object | None, *, name: str) -> str | None:
    return None if value is None else _hash(value, name=name)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _unique(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


@dataclass(frozen=True)
class V25pCompatibilityProjectionReceipt:
    projection_id: str
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    native_capture_manifest_hash: str
    projection_profile_hash: str
    target_dataset_schema: str
    status: str
    completed_at: float
    completed_clock_domain_id: str
    target_episode_record_hash: str | None = None
    target_episode_locator: str | None = None
    lossy: bool = False
    omitted_stream_ids: tuple[str, ...] = ()
    message: str = ""
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=V25P_COMPATIBILITY_PROJECTION_RECEIPT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "episode_uuid",
            "target_dataset_schema",
            "completed_clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        index = int(self.legacy_episode_index)
        if isinstance(self.legacy_episode_index, bool) or index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        object.__setattr__(self, "legacy_episode_index", index)
        for name in (
            "episode_identity_hash",
            "native_capture_manifest_hash",
            "projection_profile_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "target_episode_record_hash",
            _optional_hash(
                self.target_episode_record_hash,
                name="target_episode_record_hash",
            ),
        )
        locator = (
            None
            if self.target_episode_locator is None
            else _required(self.target_episode_locator, name="target_episode_locator")
        )
        object.__setattr__(self, "target_episode_locator", locator)
        status = _required(self.status, name="status")
        if status not in V25P_COMPATIBILITY_PROJECTION_STATUSES_V1:
            raise ValueError(f"Unsupported compatibility projection status {status!r}.")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "completed_at",
            _finite(self.completed_at, name="completed_at"),
        )
        omitted = _unique(self.omitted_stream_ids, name="omitted_stream_id")
        lossy = bool(self.lossy)
        message = str(self.message).strip()
        if status == "complete":
            if self.target_episode_record_hash is None or locator is None:
                raise ValueError(
                    "A complete projection requires a target record hash and locator."
                )
            if bool(omitted) != lossy:
                raise ValueError(
                    "A complete projection is lossy exactly when streams were omitted."
                )
        elif self.target_episode_record_hash is not None or locator is not None:
            raise ValueError("A failed/skipped projection cannot bind a target episode.")
        if status == "failed" and not message:
            raise ValueError("A failed projection requires an explanatory message.")
        object.__setattr__(self, "lossy", lossy)
        object.__setattr__(self, "omitted_stream_ids", omitted)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_COMPATIBILITY_PROJECTION_RECEIPT_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "native_capture_manifest_hash": self.native_capture_manifest_hash,
            "projection_profile_hash": self.projection_profile_hash,
            "target_dataset_schema": self.target_dataset_schema,
            "status": self.status,
            "target_episode_record_hash": self.target_episode_record_hash,
            "target_episode_locator": self.target_episode_locator,
            "lossy": self.lossy,
            "omitted_stream_ids": self.omitted_stream_ids,
            "completed_at": self.completed_at,
            "completed_clock_domain_id": self.completed_clock_domain_id,
            "message": self.message,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "V25pCompatibilityProjectionReceipt":
        if payload.get("schema") != V25P_COMPATIBILITY_PROJECTION_RECEIPT_SCHEMA_V1:
            raise ValueError("Unsupported compatibility projection receipt schema.")
        result = cls(
            projection_id=payload.get("projection_id", ""),
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            native_capture_manifest_hash=payload.get("native_capture_manifest_hash", ""),
            projection_profile_hash=payload.get("projection_profile_hash", ""),
            target_dataset_schema=payload.get("target_dataset_schema", ""),
            status=payload.get("status", ""),
            target_episode_record_hash=payload.get("target_episode_record_hash"),
            target_episode_locator=payload.get("target_episode_locator"),
            lossy=payload.get("lossy", False),
            omitted_stream_ids=tuple(payload.get("omitted_stream_ids", ())),
            completed_at=payload.get("completed_at", float("nan")),
            completed_clock_domain_id=payload.get("completed_clock_domain_id", ""),
            message=payload.get("message", ""),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Compatibility projection receipt hash mismatch.")
        return result


__all__ = [
    "V25P_COMPATIBILITY_PROJECTION_RECEIPT_FILE_DOMAIN_V1",
    "V25P_COMPATIBILITY_PROJECTION_RECEIPT_HASH_DOMAIN_V1",
    "V25P_COMPATIBILITY_PROJECTION_RECEIPT_SCHEMA_V1",
    "V25P_COMPATIBILITY_PROJECTION_STATUSES_V1",
    "V25pCompatibilityProjectionReceipt",
]
