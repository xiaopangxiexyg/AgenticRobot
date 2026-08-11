"""Pure camera identity and physical-role contracts.

The contracts deliberately separate immutable device serials, canonical
physical roles, and the keys used by a stored dataset or policy artifact.
They contain no filesystem or runtime-device behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash


CAMERA_ROLE_BINDING_SCHEMA_V1 = "dexteleop.camera_role_binding.v1"
CAMERA_ROLE_BINDING_HASH_DOMAIN_V1 = "dexteleop.camera_role_binding.v1"
CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1 = (
    "dexteleop.camera_semantic_resolution.v1"
)
CAMERA_SEMANTIC_RESOLUTION_HASH_DOMAIN_V1 = (
    "dexteleop.camera_semantic_resolution.v1"
)


class CameraSemanticContractError(ValueError):
    """Raised when camera identity evidence is missing or contradictory."""


def _nonempty(value: object, *, name: str) -> str:
    if value is None:
        raise CameraSemanticContractError(f"{name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise CameraSemanticContractError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class CameraRole:
    canonical_role: str
    serial: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_role",
            _nonempty(self.canonical_role, name="canonical_role"),
        )
        object.__setattr__(self, "serial", _nonempty(self.serial, name="serial"))

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_role": self.canonical_role,
            "serial": self.serial,
        }


@dataclass(frozen=True)
class CameraRoleBinding:
    """Versioned mapping from canonical physical roles to device serials."""

    roles: tuple[CameraRole, ...]
    schema: str = CAMERA_ROLE_BINDING_SCHEMA_V1

    def __post_init__(self) -> None:
        if str(self.schema) != CAMERA_ROLE_BINDING_SCHEMA_V1:
            raise CameraSemanticContractError(
                f"Unsupported camera role binding schema {self.schema!r}."
            )
        roles = tuple(
            sorted(self.roles, key=lambda item: str(item.canonical_role))
        )
        if not roles:
            raise CameraSemanticContractError(
                "Camera role binding must contain at least one role."
            )
        if any(not isinstance(item, CameraRole) for item in roles):
            raise TypeError("Camera role binding entries must be CameraRole values.")
        canonical_roles = tuple(item.canonical_role for item in roles)
        serials = tuple(item.serial for item in roles)
        if len(set(canonical_roles)) != len(canonical_roles):
            raise CameraSemanticContractError(
                "Camera role binding canonical roles must be unique."
            )
        if len(set(serials)) != len(serials):
            raise CameraSemanticContractError(
                "Camera role binding serials must be unique."
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "schema", CAMERA_ROLE_BINDING_SCHEMA_V1)

    @property
    def canonical_roles(self) -> tuple[str, ...]:
        return tuple(item.canonical_role for item in self.roles)

    @property
    def role_to_serial(self) -> dict[str, str]:
        return {item.canonical_role: item.serial for item in self.roles}

    @property
    def serial_to_role(self) -> dict[str, str]:
        return {item.serial: item.canonical_role for item in self.roles}

    @property
    def contract_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=CAMERA_ROLE_BINDING_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_ROLE_BINDING_SCHEMA_V1,
            "roles": {
                item.canonical_role: {"serial": item.serial}
                for item in self.roles
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraRoleBinding":
        if not isinstance(payload, Mapping):
            raise CameraSemanticContractError(
                "Camera role binding must be an object."
            )
        raw_roles = payload.get("roles")
        if not isinstance(raw_roles, Mapping):
            raise CameraSemanticContractError(
                "Camera role binding must contain a roles object."
            )
        roles: list[CameraRole] = []
        for canonical_role, raw_value in raw_roles.items():
            if not isinstance(raw_value, Mapping):
                raise CameraSemanticContractError(
                    f"Camera role {canonical_role!r} must be an object."
                )
            roles.append(
                CameraRole(
                    canonical_role=str(canonical_role),
                    serial=str(raw_value.get("serial", "")),
                )
            )
        return cls(
            roles=tuple(roles),
            schema=str(payload.get("schema", "")),
        )


@dataclass(frozen=True)
class ResolvedCameraRole:
    canonical_role: str
    stored_key: str
    serial: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_role",
            _nonempty(self.canonical_role, name="canonical_role"),
        )
        object.__setattr__(
            self,
            "stored_key",
            _nonempty(self.stored_key, name="stored_key"),
        )
        object.__setattr__(self, "serial", _nonempty(self.serial, name="serial"))

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_role": self.canonical_role,
            "stored_key": self.stored_key,
            "serial": self.serial,
        }


@dataclass(frozen=True)
class CameraSemanticResolution:
    """Read-only interpretation of one source device snapshot."""

    source_id: str
    source_device_snapshot_hash: str
    binding_hash: str
    roles: tuple[ResolvedCameraRole, ...]
    status: str
    schema: str = CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1

    def __post_init__(self) -> None:
        if str(self.schema) != CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1:
            raise CameraSemanticContractError(
                f"Unsupported camera semantic resolution schema {self.schema!r}."
            )
        source_id = _nonempty(self.source_id, name="source_id")
        snapshot_hash = _nonempty(
            self.source_device_snapshot_hash,
            name="source_device_snapshot_hash",
        )
        binding_hash = _nonempty(self.binding_hash, name="binding_hash")
        roles = tuple(
            sorted(self.roles, key=lambda item: str(item.canonical_role))
        )
        if not roles:
            raise CameraSemanticContractError(
                "Camera semantic resolution must contain at least one role."
            )
        if any(not isinstance(item, ResolvedCameraRole) for item in roles):
            raise TypeError(
                "Camera semantic resolution roles must be ResolvedCameraRole values."
            )
        canonical = tuple(item.canonical_role for item in roles)
        stored = tuple(item.stored_key for item in roles)
        serials = tuple(item.serial for item in roles)
        for label, values in (
            ("canonical roles", canonical),
            ("stored keys", stored),
            ("serials", serials),
        ):
            if len(set(values)) != len(values):
                raise CameraSemanticContractError(
                    f"Camera semantic resolution {label} must be unique."
                )
        status = str(self.status).strip().lower()
        if status not in {"identity", "swapped", "remapped"}:
            raise CameraSemanticContractError(
                f"Unsupported camera semantic resolution status {self.status!r}."
            )
        mapping = {item.canonical_role: item.stored_key for item in roles}
        expected_status = (
            "identity"
            if all(role == stored_key for role, stored_key in mapping.items())
            else "swapped"
            if mapping
            == {
                "left_wrist": "right_wrist",
                "right_wrist": "left_wrist",
            }
            else "remapped"
        )
        if status != expected_status:
            raise CameraSemanticContractError(
                "Camera semantic resolution status disagrees with its role "
                f"mapping: declared={status!r}, expected={expected_status!r}."
            )
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(
            self,
            "source_device_snapshot_hash",
            snapshot_hash,
        )
        object.__setattr__(self, "binding_hash", binding_hash)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "schema",
            CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1,
        )

    @property
    def canonical_to_stored_key(self) -> dict[str, str]:
        return {item.canonical_role: item.stored_key for item in self.roles}

    def stored_key_for(self, canonical_role: str) -> str:
        key = str(canonical_role)
        try:
            return self.canonical_to_stored_key[key]
        except KeyError as exc:
            raise CameraSemanticContractError(
                f"Camera role {key!r} is not resolved for source {self.source_id!r}."
            ) from exc

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=CAMERA_SEMANTIC_RESOLUTION_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1,
            "source_id": self.source_id,
            "source_device_snapshot_hash": self.source_device_snapshot_hash,
            "binding_hash": self.binding_hash,
            "status": self.status,
            "roles": [item.to_dict() for item in self.roles],
        }


def camera_role_binding_from_pairs(
    pairs: Sequence[tuple[str, str]],
) -> CameraRoleBinding:
    return CameraRoleBinding(
        roles=tuple(
            CameraRole(canonical_role=role, serial=serial)
            for role, serial in pairs
        )
    )


__all__ = [
    "CAMERA_ROLE_BINDING_HASH_DOMAIN_V1",
    "CAMERA_ROLE_BINDING_SCHEMA_V1",
    "CAMERA_SEMANTIC_RESOLUTION_HASH_DOMAIN_V1",
    "CAMERA_SEMANTIC_RESOLUTION_SCHEMA_V1",
    "CameraRole",
    "CameraRoleBinding",
    "CameraSemanticContractError",
    "CameraSemanticResolution",
    "ResolvedCameraRole",
    "camera_role_binding_from_pairs",
]
