"""Frozen v1 compatibility manifests built on finalized v2.3 rows.

This module preserves the first v2.5-p experiment and its hashes for existing
datasets.  It is now the explicit v2.3 compatibility branch, not the native
v2.5-p storage model.  Native embodiment-neutral streams are defined in
``dataset_contracts.v25p_native`` and may optionally project into this format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash
from .v23 import DEXTELEOP_DATASET_SCHEMA_V23


V25P_DATASET_MANIFEST_SCHEMA_V1 = "dexteleop_v25p_dataset_manifest_v1"
V25P_EPISODE_MANIFEST_SCHEMA_V1 = "dexteleop_v25p_episode_manifest_v1"
V25P_STATUS_PROVISIONAL_V1 = "provisional_on_v23"
V25P_EPISODE_STATUS_SEALED_V1 = "sealed_provisional"

V25P_CAPABILITY_EMBODIMENT_V1 = "embodiment_capture_profile_v1"
V25P_CAPABILITY_CAPTURE_PROJECTION_V1 = "capture_projection_profile_v1"
V25P_CAPABILITY_TASK_SEMANTICS_V1 = "long_horizon_task_semantics_v1"
V25P_CAPABILITY_V24_CAPTURE_COMMIT_V1 = "v24_capture_commit_v1"

_DATASET_HASH_DOMAIN = "dexteleop.v25p_dataset_manifest.semantic.v1"
_EPISODE_HASH_DOMAIN = "dexteleop.v25p_episode_manifest.semantic.v1"
V25P_BASE_EPISODE_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.v25p_base_episode_manifest.semantic.v1"
)
V25P_DATASET_MANIFEST_FILE_DOMAIN_V1 = "dexteleop.v25p_dataset_manifest.file.v1"
V25P_PROFILE_FILE_DOMAIN_V1 = "dexteleop.v25p_profile.file.v1"
V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1 = "dexteleop.v25p_episode_identity.file.v1"
V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.v25p_embodiment_binding.file.v1"
)
V25P_TASK_EVENT_FILE_DOMAIN_V1 = "dexteleop.v25p_task_event.file.v1"
V25P_TASK_MANIFEST_FILE_DOMAIN_V1 = "dexteleop.v25p_task_manifest.file.v1"
V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1 = "dexteleop.v25p_episode_manifest.file.v1"
V25P_RELATIVE_ROOT_V1 = Path("meta/v25p")
V25P_PROFILE_RELATIVE_ROOT_V1 = V25P_RELATIVE_ROOT_V1 / "profiles"
V25P_EPISODE_RELATIVE_ROOT_V1 = V25P_RELATIVE_ROOT_V1 / "episodes"


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _hash(value: object, *, name: str) -> str:
    text = _required(value, name=name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _optional_hash(value: object | None, *, name: str) -> str | None:
    return None if value is None else _hash(value, name=name)


def _hashes(values: Sequence[object], *, name: str, required: bool) -> tuple[str, ...]:
    result = tuple(_hash(value, name=name) for value in values)
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique.")
    return result


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def v25p_base_episode_manifest_hash(payload: Mapping[str, Any]) -> str:
    """Hash the exact finalized legacy episode record used by v2.5-p."""

    return canonical_content_hash(
        dict(payload),
        domain=V25P_BASE_EPISODE_MANIFEST_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class V25pDatasetManifest:
    dataset_id: str
    policy_io_layout_hash: str
    pose_frame_contract_hash: str
    embodiment_profile_hashes: tuple[str, ...]
    capture_projection_profile_hashes: tuple[str, ...]
    created_at: float
    created_clock_domain_id: str
    long_horizon_task_profile_hashes: tuple[str, ...] = ()
    base_dataset_schema: str = DEXTELEOP_DATASET_SCHEMA_V23
    status: str = V25P_STATUS_PROVISIONAL_V1
    native_v25_eligible: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_DATASET_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required(self.dataset_id, name="dataset_id"))
        if self.base_dataset_schema != DEXTELEOP_DATASET_SCHEMA_V23:
            raise ValueError("v2.5-p must preserve dexteleop_dataset_v2.3 rows.")
        if self.status != V25P_STATUS_PROVISIONAL_V1:
            raise ValueError("Unsupported v2.5-p dataset status.")
        if bool(self.native_v25_eligible):
            raise ValueError("A provisional v2.5-p manifest cannot claim native v2.5.")
        object.__setattr__(self, "native_v25_eligible", False)
        for name in ("policy_io_layout_hash", "pose_frame_contract_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "embodiment_profile_hashes",
            _hashes(
                self.embodiment_profile_hashes,
                name="embodiment_profile_hashes",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "capture_projection_profile_hashes",
            _hashes(
                self.capture_projection_profile_hashes,
                name="capture_projection_profile_hashes",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "long_horizon_task_profile_hashes",
            _hashes(
                self.long_horizon_task_profile_hashes,
                name="long_horizon_task_profile_hashes",
                required=False,
            ),
        )
        object.__setattr__(self, "created_at", _finite(self.created_at, name="created_at"))
        object.__setattr__(
            self,
            "created_clock_domain_id",
            _required(self.created_clock_domain_id, name="created_clock_domain_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_DATASET_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "base_dataset_schema": self.base_dataset_schema,
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "embodiment_profile_hashes": self.embodiment_profile_hashes,
            "capture_projection_profile_hashes": self.capture_projection_profile_hashes,
            "long_horizon_task_profile_hashes": self.long_horizon_task_profile_hashes,
            "created_at": self.created_at,
            "created_clock_domain_id": self.created_clock_domain_id,
            "status": self.status,
            "native_v25_eligible": self.native_v25_eligible,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pDatasetManifest":
        if payload.get("schema") != V25P_DATASET_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported v2.5-p dataset manifest schema.")
        result = cls(
            dataset_id=payload.get("dataset_id", ""),
            base_dataset_schema=payload.get("base_dataset_schema", ""),
            policy_io_layout_hash=payload.get("policy_io_layout_hash", ""),
            pose_frame_contract_hash=payload.get("pose_frame_contract_hash", ""),
            embodiment_profile_hashes=tuple(
                payload.get("embodiment_profile_hashes", ())
            ),
            capture_projection_profile_hashes=tuple(
                payload.get("capture_projection_profile_hashes", ())
            ),
            long_horizon_task_profile_hashes=tuple(
                payload.get("long_horizon_task_profile_hashes", ())
            ),
            created_at=payload.get("created_at", float("nan")),
            created_clock_domain_id=payload.get("created_clock_domain_id", ""),
            status=payload.get("status", ""),
            native_v25_eligible=payload.get("native_v25_eligible", False),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("v2.5-p dataset manifest hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pEpisodeManifest:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    embodiment_binding_hash: str
    base_episode_manifest_hash: str
    sealed_at: float
    sealed_clock_domain_id: str
    task_semantics_manifest_hash: str | None = None
    v24_capture_commit_hash: str | None = None
    status: str = V25P_EPISODE_STATUS_SEALED_V1
    native_v25_eligible: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_EPISODE_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        index = int(self.legacy_episode_index)
        if isinstance(self.legacy_episode_index, bool) or index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        object.__setattr__(self, "legacy_episode_index", index)
        for name in (
            "episode_identity_hash",
            "embodiment_binding_hash",
            "base_episode_manifest_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "task_semantics_manifest_hash",
            _optional_hash(
                self.task_semantics_manifest_hash,
                name="task_semantics_manifest_hash",
            ),
        )
        object.__setattr__(
            self,
            "v24_capture_commit_hash",
            _optional_hash(self.v24_capture_commit_hash, name="v24_capture_commit_hash"),
        )
        if self.status != V25P_EPISODE_STATUS_SEALED_V1:
            raise ValueError("Unsupported v2.5-p episode status.")
        if bool(self.native_v25_eligible):
            raise ValueError("A provisional episode cannot claim native v2.5 eligibility.")
        object.__setattr__(self, "native_v25_eligible", False)
        object.__setattr__(self, "sealed_at", _finite(self.sealed_at, name="sealed_at"))
        object.__setattr__(
            self,
            "sealed_clock_domain_id",
            _required(self.sealed_clock_domain_id, name="sealed_clock_domain_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_EPISODE_HASH_DOMAIN
            ),
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        values = [
            V25P_CAPABILITY_EMBODIMENT_V1,
            V25P_CAPABILITY_CAPTURE_PROJECTION_V1,
        ]
        if self.task_semantics_manifest_hash is not None:
            values.append(V25P_CAPABILITY_TASK_SEMANTICS_V1)
        if self.v24_capture_commit_hash is not None:
            values.append(V25P_CAPABILITY_V24_CAPTURE_COMMIT_V1)
        return tuple(values)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "embodiment_binding_hash": self.embodiment_binding_hash,
            "base_episode_manifest_hash": self.base_episode_manifest_hash,
            "task_semantics_manifest_hash": self.task_semantics_manifest_hash,
            "v24_capture_commit_hash": self.v24_capture_commit_hash,
            "capabilities": self.capabilities,
            "sealed_at": self.sealed_at,
            "sealed_clock_domain_id": self.sealed_clock_domain_id,
            "status": self.status,
            "native_v25_eligible": self.native_v25_eligible,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pEpisodeManifest":
        if payload.get("schema") != V25P_EPISODE_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported v2.5-p episode manifest schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            embodiment_binding_hash=payload.get("embodiment_binding_hash", ""),
            base_episode_manifest_hash=payload.get("base_episode_manifest_hash", ""),
            task_semantics_manifest_hash=payload.get("task_semantics_manifest_hash"),
            v24_capture_commit_hash=payload.get("v24_capture_commit_hash"),
            sealed_at=payload.get("sealed_at", float("nan")),
            sealed_clock_domain_id=payload.get("sealed_clock_domain_id", ""),
            status=payload.get("status", ""),
            native_v25_eligible=payload.get("native_v25_eligible", False),
        )
        declared_capabilities = payload.get("capabilities")
        if (
            declared_capabilities is not None
            and tuple(declared_capabilities) != result.capabilities
        ):
            raise ValueError("v2.5-p episode capabilities do not match bound artifacts.")
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("v2.5-p episode manifest hash mismatch.")
        return result


__all__ = [
    "V25P_BASE_EPISODE_MANIFEST_HASH_DOMAIN_V1",
    "V25P_CAPABILITY_CAPTURE_PROJECTION_V1",
    "V25P_CAPABILITY_EMBODIMENT_V1",
    "V25P_CAPABILITY_TASK_SEMANTICS_V1",
    "V25P_CAPABILITY_V24_CAPTURE_COMMIT_V1",
    "V25P_DATASET_MANIFEST_SCHEMA_V1",
    "V25P_DATASET_MANIFEST_FILE_DOMAIN_V1",
    "V25P_EMBODIMENT_BINDING_FILE_DOMAIN_V1",
    "V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1",
    "V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1",
    "V25P_EPISODE_RELATIVE_ROOT_V1",
    "V25P_EPISODE_MANIFEST_SCHEMA_V1",
    "V25P_EPISODE_STATUS_SEALED_V1",
    "V25P_PROFILE_FILE_DOMAIN_V1",
    "V25P_PROFILE_RELATIVE_ROOT_V1",
    "V25P_RELATIVE_ROOT_V1",
    "V25P_STATUS_PROVISIONAL_V1",
    "V25P_TASK_EVENT_FILE_DOMAIN_V1",
    "V25pDatasetManifest",
    "V25pEpisodeManifest",
    "V25P_TASK_MANIFEST_FILE_DOMAIN_V1",
    "v25p_base_episode_manifest_hash",
]
