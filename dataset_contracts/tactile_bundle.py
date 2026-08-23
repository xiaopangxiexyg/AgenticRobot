"""Portable contracts for natively separated tactile dataset bundles.

The contracts in this module contain content identity and logical paths only.
They intentionally exclude mount points, absolute paths, filesystem device
numbers, NAS locations, and runtime attachment policy so a tactile bundle can
be moved independently and rebound to the same core dataset later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence
import unicodedata
import uuid

from .canonical_encoding import canonical_content_hash


TACTILE_BUNDLE_DATASET_IDENTITY_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.dataset_identity.v1"
)
TACTILE_BUNDLE_ARTIFACT_SCHEMA_V1 = "dexteleop.tactile_bundle.artifact.v1"
TACTILE_BUNDLE_CORE_REFERENCE_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.core_reference.v1"
)
TACTILE_BUNDLE_MANIFEST_BINDING_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.manifest_binding.v1"
)
TACTILE_BUNDLE_CORE_COMPONENT_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.core_component.v1"
)
TACTILE_BUNDLE_TACTILE_COMPONENT_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.tactile_component.v1"
)
TACTILE_BUNDLE_EPISODE_COMMIT_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.episode_commit.v1"
)
TACTILE_BUNDLE_DATASET_BINDING_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.dataset_binding.v1"
)
TACTILE_BUNDLE_DATASET_LOCATOR_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.dataset_locator.v1"
)
TACTILE_BUNDLE_EPISODE_BINDING_SCHEMA_V1 = (
    "dexteleop.tactile_bundle.episode_binding.v1"
)

TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1 = (
    "dexteleop.tactile_bundle.dataset_identity.file.v1"
)
TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1 = (
    "dexteleop.tactile_bundle.episode_identity.file.v1"
)
TACTILE_BUNDLE_CORE_COMPONENT_FILE_DOMAIN_V1 = (
    "dexteleop.tactile_bundle.core_component.file.v1"
)
TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1 = (
    "dexteleop.tactile_bundle.tactile_component.file.v1"
)
TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1 = (
    "dexteleop.tactile_bundle.episode_commit.file.v1"
)

_DATASET_IDENTITY_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.dataset_identity.semantic.v1"
)
_ARTIFACT_HASH_DOMAIN = "dexteleop.tactile_bundle.artifact.semantic.v1"
_CORE_REFERENCE_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.core_reference.semantic.v1"
)
_MANIFEST_BINDING_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.manifest_binding.semantic.v1"
)
_CORE_COMPONENT_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.core_component.semantic.v1"
)
_TACTILE_COMPONENT_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.tactile_component.semantic.v1"
)
_EPISODE_COMMIT_HASH_DOMAIN = (
    "dexteleop.tactile_bundle.episode_commit.semantic.v1"
)

TACTILE_BUNDLE_LOGICAL_PREFIX_V1 = "tactile"
TACTILE_BUNDLE_CORE_RELATIVE_ROOT_V1 = PurePosixPath("meta/tactile_bundle_v1")
TACTILE_BUNDLE_STORAGE_RELATIVE_ROOT_V1 = PurePosixPath("meta/tactile_bundle_v1")
TACTILE_BUNDLE_DATASET_LOCATOR_RELATIVE_ROOT_V1 = PurePosixPath(
    "datasets_by_name"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIDE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class TactileBundleContractError(ValueError):
    """Raised when a native tactile bundle contract is malformed."""


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TactileBundleContractError(f"{field_name} must be non-empty")
    return text


def _sha256(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name).lower()
    if not _SHA256_RE.fullmatch(text):
        raise TactileBundleContractError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return text


def _dataset_uuid(value: object) -> str:
    text = _required_text(value, field_name="dataset_uuid")
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TactileBundleContractError("dataset_uuid must be a UUID") from exc
    canonical = str(parsed)
    if text.lower() != canonical:
        raise TactileBundleContractError(
            "dataset_uuid must use canonical lowercase UUID spelling"
        )
    return canonical


def _episode_uuid(value: object) -> str:
    text = _required_text(value, field_name="episode_uuid")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise TactileBundleContractError(
            "episode_uuid must be safe for relative storage"
        )
    return text


def _dataset_label_path_token(value: object) -> str:
    """Return a readable, bounded path token without changing label identity."""

    label = _required_text(value, field_name="dataset_label")
    normalized = unicodedata.normalize("NFKC", label).strip()
    token = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in normalized
    )
    token = re.sub(r"_+", "_", token).strip("._-") or "dataset"
    changed = token != normalized
    while len(token.encode("utf-8")) > 96:
        token = token[:-1]
        changed = True
    token = token.rstrip("._-") or "dataset"
    if changed:
        suffix = canonical_content_hash(
            label,
            domain="dexteleop.tactile_bundle.dataset_label_path.v1",
        )[:8]
        token = f"{token}--{suffix}"
    return token


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TactileBundleContractError(
            f"{field_name} must be a non-negative integer"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TactileBundleContractError(
            f"{field_name} must be a non-negative integer"
        ) from exc
    if result < 0 or result != value:
        raise TactileBundleContractError(
            f"{field_name} must be a non-negative integer"
        )
    return result


def normalize_bundle_relative_path(
    value: object,
    *,
    field_name: str,
    required_prefix: str | None = None,
) -> str:
    """Return a canonical safe POSIX path without filesystem interpretation."""

    text = _required_text(value, field_name=field_name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or text.startswith("/"):
        raise TactileBundleContractError(f"{field_name} must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise TactileBundleContractError(
            f"{field_name} contains an unsafe path component"
        )
    normalized = path.as_posix()
    if required_prefix is not None:
        prefix = PurePosixPath(required_prefix).as_posix()
        if normalized != prefix and not normalized.startswith(f"{prefix}/"):
            raise TactileBundleContractError(
                f"{field_name} must remain under {prefix!r}"
            )
    return normalized


@dataclass(frozen=True)
class TactileBundleDatasetIdentityV1:
    dataset_uuid: str
    dataset_label: str
    logical_prefix: str = TACTILE_BUNDLE_LOGICAL_PREFIX_V1
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_DATASET_IDENTITY_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_uuid", _dataset_uuid(self.dataset_uuid))
        object.__setattr__(
            self,
            "dataset_label",
            _required_text(self.dataset_label, field_name="dataset_label"),
        )
        prefix = normalize_bundle_relative_path(
            self.logical_prefix,
            field_name="logical_prefix",
        )
        if prefix != TACTILE_BUNDLE_LOGICAL_PREFIX_V1:
            raise TactileBundleContractError(
                f"v1 logical_prefix must be {TACTILE_BUNDLE_LOGICAL_PREFIX_V1!r}"
            )
        object.__setattr__(self, "logical_prefix", prefix)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_DATASET_IDENTITY_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset_uuid": self.dataset_uuid,
            "dataset_label": self.dataset_label,
            "logical_prefix": self.logical_prefix,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "TactileBundleDatasetIdentityV1":
        if payload.get("schema") != TACTILE_BUNDLE_DATASET_IDENTITY_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported tactile bundle dataset identity schema"
            )
        result = cls(
            dataset_uuid=payload.get("dataset_uuid", ""),
            dataset_label=payload.get("dataset_label", ""),
            logical_prefix=payload.get("logical_prefix", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError(
                "tactile bundle dataset identity hash mismatch"
            )
        return result


@dataclass(frozen=True)
class TactileBundleDatasetLocatorV1:
    """Rebuildable human-facing pointer to one UUID-addressed bundle."""

    dataset_uuid: str
    dataset_label: str
    dataset_identity_hash: str
    bundle_relative_path: str = field(init=False)
    core_identity_ref: str = field(init=False)
    core_episode_catalog_ref: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_DATASET_LOCATOR_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        dataset_uuid = _dataset_uuid(self.dataset_uuid)
        object.__setattr__(self, "dataset_uuid", dataset_uuid)
        object.__setattr__(
            self,
            "dataset_label",
            _required_text(self.dataset_label, field_name="dataset_label"),
        )
        object.__setattr__(
            self,
            "dataset_identity_hash",
            _sha256(
                self.dataset_identity_hash,
                field_name="dataset_identity_hash",
            ),
        )
        object.__setattr__(
            self,
            "bundle_relative_path",
            (PurePosixPath("datasets") / dataset_uuid).as_posix(),
        )
        object.__setattr__(
            self,
            "core_identity_ref",
            (TACTILE_BUNDLE_CORE_RELATIVE_ROOT_V1 / "dataset_identity.cjson").as_posix(),
        )
        object.__setattr__(
            self,
            "core_episode_catalog_ref",
            "meta/episodes.jsonl",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_uuid": self.dataset_uuid,
            "core_dataset_name": self.dataset_label,
            "dataset_identity_hash": self.dataset_identity_hash,
            "bundle_relative_path": self.bundle_relative_path,
            "core_identity_ref": self.core_identity_ref,
            "core_episode_catalog_ref": self.core_episode_catalog_ref,
            "episode_join_fields": [
                "episode_index",
                "tactile_bundle.dataset_uuid",
                "tactile_bundle.episode_uuid",
            ],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> TactileBundleDatasetLocatorV1:
        if payload.get("schema") != TACTILE_BUNDLE_DATASET_LOCATOR_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported tactile bundle dataset locator schema"
            )
        result = cls(
            dataset_uuid=payload.get("dataset_uuid", ""),
            dataset_label=payload.get("core_dataset_name", ""),
            dataset_identity_hash=payload.get("dataset_identity_hash", ""),
        )
        if dict(payload) != result.to_payload():
            raise TactileBundleContractError(
                "tactile bundle dataset locator fields do not match its identity"
            )
        return result


@dataclass(frozen=True)
class TactileBundleArtifactV1:
    logical_path: str
    role: str
    size_bytes: int
    sha256: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=TACTILE_BUNDLE_ARTIFACT_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_path",
            normalize_bundle_relative_path(
                self.logical_path,
                field_name="logical_path",
                required_prefix=TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
            ),
        )
        role = _required_text(self.role, field_name="role")
        if role not in {"manifest", "payload"}:
            raise TactileBundleContractError(
                "artifact role must be 'manifest' or 'payload'"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "size_bytes",
            _non_negative_int(self.size_bytes, field_name="size_bytes"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, field_name="sha256"))
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_ARTIFACT_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "logical_path": self.logical_path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TactileBundleArtifactV1":
        if payload.get("schema") != TACTILE_BUNDLE_ARTIFACT_SCHEMA_V1:
            raise TactileBundleContractError("unsupported tactile artifact schema")
        result = cls(
            logical_path=payload.get("logical_path", ""),
            role=payload.get("role", ""),
            size_bytes=payload.get("size_bytes", -1),
            sha256=payload.get("sha256", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("tactile artifact hash mismatch")
        return result


@dataclass(frozen=True)
class CoreTactileReferenceV1:
    side: str
    sample_frame_index: int
    manifest_logical_path: str
    tactile_frame_index: int
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_CORE_REFERENCE_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        side = _required_text(self.side, field_name="side")
        if not _SIDE_RE.fullmatch(side):
            raise TactileBundleContractError("side contains unsupported characters")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "sample_frame_index",
            _non_negative_int(
                self.sample_frame_index, field_name="sample_frame_index"
            ),
        )
        object.__setattr__(
            self,
            "manifest_logical_path",
            normalize_bundle_relative_path(
                self.manifest_logical_path,
                field_name="manifest_logical_path",
                required_prefix=TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
            ),
        )
        object.__setattr__(
            self,
            "tactile_frame_index",
            _non_negative_int(
                self.tactile_frame_index, field_name="tactile_frame_index"
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_CORE_REFERENCE_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "side": self.side,
            "sample_frame_index": self.sample_frame_index,
            "manifest_logical_path": self.manifest_logical_path,
            "tactile_frame_index": self.tactile_frame_index,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoreTactileReferenceV1":
        if payload.get("schema") != TACTILE_BUNDLE_CORE_REFERENCE_SCHEMA_V1:
            raise TactileBundleContractError("unsupported core tactile reference schema")
        result = cls(
            side=payload.get("side", ""),
            sample_frame_index=payload.get("sample_frame_index", -1),
            manifest_logical_path=payload.get("manifest_logical_path", ""),
            tactile_frame_index=payload.get("tactile_frame_index", -1),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("core tactile reference hash mismatch")
        return result


@dataclass(frozen=True)
class TactileManifestBindingV1:
    side: str
    manifest_logical_path: str
    frame_count: int
    frame_alignment_hash: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_MANIFEST_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        side = _required_text(self.side, field_name="side")
        if not _SIDE_RE.fullmatch(side):
            raise TactileBundleContractError("side contains unsupported characters")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "manifest_logical_path",
            normalize_bundle_relative_path(
                self.manifest_logical_path,
                field_name="manifest_logical_path",
                required_prefix=TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
            ),
        )
        object.__setattr__(
            self,
            "frame_count",
            _non_negative_int(self.frame_count, field_name="frame_count"),
        )
        object.__setattr__(
            self,
            "frame_alignment_hash",
            _sha256(self.frame_alignment_hash, field_name="frame_alignment_hash"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_MANIFEST_BINDING_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "side": self.side,
            "manifest_logical_path": self.manifest_logical_path,
            "frame_count": self.frame_count,
            "frame_alignment_hash": self.frame_alignment_hash,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "TactileManifestBindingV1":
        if payload.get("schema") != TACTILE_BUNDLE_MANIFEST_BINDING_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported tactile manifest binding schema"
            )
        result = cls(
            side=payload.get("side", ""),
            manifest_logical_path=payload.get("manifest_logical_path", ""),
            frame_count=payload.get("frame_count", -1),
            frame_alignment_hash=payload.get("frame_alignment_hash", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("tactile manifest binding hash mismatch")
        return result


def _normalize_component_identity(
    *,
    dataset_uuid: object,
    episode_uuid: object,
    legacy_episode_index: object,
    episode_identity_hash: object,
    sample_count: object,
) -> tuple[str, str, int, str, int]:
    return (
        _dataset_uuid(dataset_uuid),
        _episode_uuid(episode_uuid),
        _non_negative_int(legacy_episode_index, field_name="legacy_episode_index"),
        _sha256(episode_identity_hash, field_name="episode_identity_hash"),
        _non_negative_int(sample_count, field_name="sample_count"),
    )


@dataclass(frozen=True)
class CoreTactileEpisodeComponentV1:
    dataset_uuid: str
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    sample_count: int
    references: tuple[CoreTactileReferenceV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_CORE_COMPONENT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        values = _normalize_component_identity(
            dataset_uuid=self.dataset_uuid,
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            episode_identity_hash=self.episode_identity_hash,
            sample_count=self.sample_count,
        )
        for name, value in zip(
            (
                "dataset_uuid",
                "episode_uuid",
                "legacy_episode_index",
                "episode_identity_hash",
                "sample_count",
            ),
            values,
        ):
            object.__setattr__(self, name, value)
        references = tuple(self.references)
        if not references:
            raise TactileBundleContractError(
                "native tactile core component must contain frame references"
            )
        keys = [
            (item.side, item.sample_frame_index)
            for item in references
        ]
        if len(keys) != len(set(keys)):
            raise TactileBundleContractError(
                "core tactile references must be unique per side/sample"
            )
        if any(item.sample_frame_index >= self.sample_count for item in references):
            raise TactileBundleContractError(
                "core tactile reference exceeds sample_count"
            )
        references = tuple(
            sorted(references, key=lambda item: (item.side, item.sample_frame_index))
        )
        object.__setattr__(self, "references", references)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_CORE_COMPONENT_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset_uuid": self.dataset_uuid,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "sample_count": self.sample_count,
            "references": [item.to_payload() for item in self.references],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "CoreTactileEpisodeComponentV1":
        if payload.get("schema") != TACTILE_BUNDLE_CORE_COMPONENT_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported core tactile component schema"
            )
        result = cls(
            dataset_uuid=payload.get("dataset_uuid", ""),
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            sample_count=payload.get("sample_count", -1),
            references=tuple(
                CoreTactileReferenceV1.from_payload(item)
                for item in payload.get("references", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("core tactile component hash mismatch")
        return result


@dataclass(frozen=True)
class TactileEpisodeComponentV1:
    dataset_uuid: str
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    sample_count: int
    artifacts: tuple[TactileBundleArtifactV1, ...]
    manifests: tuple[TactileManifestBindingV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_TACTILE_COMPONENT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        values = _normalize_component_identity(
            dataset_uuid=self.dataset_uuid,
            episode_uuid=self.episode_uuid,
            legacy_episode_index=self.legacy_episode_index,
            episode_identity_hash=self.episode_identity_hash,
            sample_count=self.sample_count,
        )
        for name, value in zip(
            (
                "dataset_uuid",
                "episode_uuid",
                "legacy_episode_index",
                "episode_identity_hash",
                "sample_count",
            ),
            values,
        ):
            object.__setattr__(self, name, value)
        artifacts = tuple(sorted(self.artifacts, key=lambda item: item.logical_path))
        if not artifacts:
            raise TactileBundleContractError(
                "native tactile component must contain artifacts"
            )
        artifact_paths = [item.logical_path for item in artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise TactileBundleContractError("tactile artifact paths must be unique")
        manifests = tuple(sorted(self.manifests, key=lambda item: item.side))
        if not manifests:
            raise TactileBundleContractError(
                "native tactile component must contain manifests"
            )
        if any(item.frame_count <= 0 for item in manifests):
            raise TactileBundleContractError(
                "native tactile manifests must contain at least one frame"
            )
        if len({item.side for item in manifests}) != len(manifests):
            raise TactileBundleContractError("tactile manifest sides must be unique")
        manifest_paths = {item.manifest_logical_path for item in manifests}
        declared_manifest_paths = {
            item.logical_path for item in artifacts if item.role == "manifest"
        }
        if manifest_paths != declared_manifest_paths:
            raise TactileBundleContractError(
                "manifest bindings and manifest artifacts must match exactly"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "manifests", manifests)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_TACTILE_COMPONENT_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset_uuid": self.dataset_uuid,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "sample_count": self.sample_count,
            "artifacts": [item.to_payload() for item in self.artifacts],
            "manifests": [item.to_payload() for item in self.manifests],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "TactileEpisodeComponentV1":
        if payload.get("schema") != TACTILE_BUNDLE_TACTILE_COMPONENT_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported tactile episode component schema"
            )
        result = cls(
            dataset_uuid=payload.get("dataset_uuid", ""),
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            sample_count=payload.get("sample_count", -1),
            artifacts=tuple(
                TactileBundleArtifactV1.from_payload(item)
                for item in payload.get("artifacts", ())
            ),
            manifests=tuple(
                TactileManifestBindingV1.from_payload(item)
                for item in payload.get("manifests", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("tactile component hash mismatch")
        return result


@dataclass(frozen=True)
class TactileEpisodeBundleCommitV1:
    dataset_uuid: str
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    core_component_hash: str
    tactile_component_hash: str
    core_component_ref: str
    tactile_component_ref: str
    tactile_required: bool = True
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TACTILE_BUNDLE_EPISODE_COMMIT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_uuid", _dataset_uuid(self.dataset_uuid))
        object.__setattr__(self, "episode_uuid", _episode_uuid(self.episode_uuid))
        object.__setattr__(
            self,
            "legacy_episode_index",
            _non_negative_int(
                self.legacy_episode_index, field_name="legacy_episode_index"
            ),
        )
        for name in (
            "episode_identity_hash",
            "core_component_hash",
            "tactile_component_hash",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "core_component_ref",
            normalize_bundle_relative_path(
                self.core_component_ref,
                field_name="core_component_ref",
            ),
        )
        object.__setattr__(
            self,
            "tactile_component_ref",
            normalize_bundle_relative_path(
                self.tactile_component_ref,
                field_name="tactile_component_ref",
            ),
        )
        expected_core_ref = (
            TACTILE_BUNDLE_CORE_RELATIVE_ROOT_V1
            / "episodes"
            / self.episode_uuid
            / "core_component.cjson"
        ).as_posix()
        expected_tactile_ref = (
            TACTILE_BUNDLE_STORAGE_RELATIVE_ROOT_V1
            / "episodes"
            / self.episode_uuid
            / "tactile_component.cjson"
        ).as_posix()
        if self.core_component_ref != expected_core_ref:
            raise TactileBundleContractError(
                "core_component_ref does not match the episode identity"
            )
        if self.tactile_component_ref != expected_tactile_ref:
            raise TactileBundleContractError(
                "tactile_component_ref does not match the episode identity"
            )
        if self.tactile_required is not True:
            raise TactileBundleContractError(
                "native tactile bundle v1 requires tactile_required=true"
            )
        object.__setattr__(self, "tactile_required", True)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_EPISODE_COMMIT_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset_uuid": self.dataset_uuid,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "core_component_hash": self.core_component_hash,
            "tactile_component_hash": self.tactile_component_hash,
            "core_component_ref": self.core_component_ref,
            "tactile_component_ref": self.tactile_component_ref,
            "tactile_required": self.tactile_required,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "TactileEpisodeBundleCommitV1":
        if payload.get("schema") != TACTILE_BUNDLE_EPISODE_COMMIT_SCHEMA_V1:
            raise TactileBundleContractError(
                "unsupported tactile episode commit schema"
            )
        result = cls(
            dataset_uuid=payload.get("dataset_uuid", ""),
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            core_component_hash=payload.get("core_component_hash", ""),
            tactile_component_hash=payload.get("tactile_component_hash", ""),
            core_component_ref=payload.get("core_component_ref", ""),
            tactile_component_ref=payload.get("tactile_component_ref", ""),
            tactile_required=payload.get("tactile_required", False),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and declared != result.semantic_content_hash:
            raise TactileBundleContractError("tactile episode commit hash mismatch")
        return result


def tactile_frame_alignment_hash(frames: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact causal frame table declared by one tactile manifest."""

    return canonical_content_hash(
        [dict(item) for item in frames],
        domain="dexteleop.tactile_bundle.frame_alignment.v1",
    )


def core_dataset_identity_relative_path() -> str:
    return (TACTILE_BUNDLE_CORE_RELATIVE_ROOT_V1 / "dataset_identity.cjson").as_posix()


def storage_dataset_identity_relative_path() -> str:
    return (TACTILE_BUNDLE_STORAGE_RELATIVE_ROOT_V1 / "dataset_identity.cjson").as_posix()


def tactile_dataset_locator_relative_path(
    *,
    dataset_label: str,
    dataset_uuid: str,
) -> str:
    return (
        TACTILE_BUNDLE_DATASET_LOCATOR_RELATIVE_ROOT_V1
        / _dataset_label_path_token(dataset_label)
        / f"{_dataset_uuid(dataset_uuid)}.json"
    ).as_posix()


def core_episode_relative_dir(episode_uuid: str) -> str:
    return (
        TACTILE_BUNDLE_CORE_RELATIVE_ROOT_V1 / "episodes" / _episode_uuid(episode_uuid)
    ).as_posix()


def storage_episode_relative_dir(episode_uuid: str) -> str:
    return (
        TACTILE_BUNDLE_STORAGE_RELATIVE_ROOT_V1
        / "episodes"
        / _episode_uuid(episode_uuid)
    ).as_posix()


__all__ = [
    "CoreTactileEpisodeComponentV1",
    "CoreTactileReferenceV1",
    "TACTILE_BUNDLE_CORE_COMPONENT_FILE_DOMAIN_V1",
    "TACTILE_BUNDLE_DATASET_BINDING_SCHEMA_V1",
    "TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1",
    "TACTILE_BUNDLE_DATASET_LOCATOR_RELATIVE_ROOT_V1",
    "TACTILE_BUNDLE_DATASET_LOCATOR_SCHEMA_V1",
    "TACTILE_BUNDLE_EPISODE_BINDING_SCHEMA_V1",
    "TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1",
    "TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1",
    "TACTILE_BUNDLE_LOGICAL_PREFIX_V1",
    "TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1",
    "TactileBundleArtifactV1",
    "TactileBundleContractError",
    "TactileBundleDatasetIdentityV1",
    "TactileBundleDatasetLocatorV1",
    "TactileEpisodeBundleCommitV1",
    "TactileEpisodeComponentV1",
    "TactileManifestBindingV1",
    "core_dataset_identity_relative_path",
    "core_episode_relative_dir",
    "normalize_bundle_relative_path",
    "storage_dataset_identity_relative_path",
    "storage_episode_relative_dir",
    "tactile_dataset_locator_relative_path",
    "tactile_frame_alignment_hash",
]
