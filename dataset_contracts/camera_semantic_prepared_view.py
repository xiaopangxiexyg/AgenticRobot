"""Pure contract for camera-corrected, non-tactile dataset projections."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical_encoding import canonical_content_hash

CAMERA_SEMANTIC_PREPARED_VIEW_SCHEMA_V1 = (
    "dexteleop.camera_semantic_prepared_view.v1"
)
CAMERA_SEMANTIC_PREPARED_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.camera_semantic_prepared_view_manifest.v1"
)
CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH = (
    "meta/prepared_views/camera_semantics.json"
)
CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_SCHEMA_V1 = (
    "dexteleop.camera_semantic_video_integrity_exclusions.v1"
)
CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_HASH_DOMAIN_V1 = (
    "dexteleop.camera_semantic_video_integrity_exclusions_manifest.v1"
)
CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH = (
    "meta/prepared_views/camera_semantics_video_integrity_exclusions.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CameraSemanticPreparedViewContractError(ValueError):
    """Raised when a prepared camera projection is incomplete or ambiguous."""


def _sha256(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise CameraSemanticPreparedViewContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


@dataclass(frozen=True)
class CameraSemanticPreparedViewFileV1:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = str(self.relative_path or "").strip()
        parts = tuple(part for part in path.split("/") if part)
        if not path or path.startswith("/") or ".." in parts:
            raise CameraSemanticPreparedViewContractError(
                f"Unsafe prepared-view file path: {path!r}."
            )
        size = int(self.size_bytes)
        if size < 0:
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view file size must be non-negative."
            )
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "sha256", _sha256(self.sha256, name="sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> CameraSemanticPreparedViewFileV1:
        return cls(
            relative_path=str(value.get("relative_path") or ""),
            size_bytes=int(value.get("size_bytes", -1)),
            sha256=str(value.get("sha256") or ""),
        )


@dataclass(frozen=True)
class CameraSemanticVideoIntegrityIssueV1:
    episode_index: int
    camera_key: str
    relative_path: str
    reason: str
    declared_frame_count: int | None = None
    decoded_frame_count: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        episode_index = int(self.episode_index)
        camera_key = str(self.camera_key or "").strip()
        relative_path = str(self.relative_path or "").strip()
        reason = str(self.reason or "").strip()
        detail = str(self.detail or "").strip()
        parts = tuple(part for part in relative_path.split("/") if part)
        if episode_index < 0:
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity issue episode_index must be non-negative."
            )
        if not camera_key or not reason or not detail:
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity issue camera_key, reason and detail must be non-empty."
            )
        if relative_path.startswith("/") or ".." in parts:
            raise CameraSemanticPreparedViewContractError(
                f"Unsafe video-integrity issue path: {relative_path!r}."
            )
        for name in ("declared_frame_count", "decoded_frame_count"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise CameraSemanticPreparedViewContractError(
                    f"{name} must be non-negative when present."
                )
            object.__setattr__(self, name, None if value is None else int(value))
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "camera_key", camera_key)
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "detail", detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "camera_key": self.camera_key,
            "relative_path": self.relative_path,
            "reason": self.reason,
            "declared_frame_count": self.declared_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> CameraSemanticVideoIntegrityIssueV1:
        return cls(
            episode_index=int(value.get("episode_index", -1)),
            camera_key=str(value.get("camera_key") or ""),
            relative_path=str(value.get("relative_path") or ""),
            reason=str(value.get("reason") or ""),
            declared_frame_count=(
                None
                if value.get("declared_frame_count") is None
                else int(value["declared_frame_count"])
            ),
            decoded_frame_count=(
                None
                if value.get("decoded_frame_count") is None
                else int(value["decoded_frame_count"])
            ),
            detail=str(value.get("detail") or ""),
        )


@dataclass(frozen=True)
class CameraSemanticVideoIntegrityExclusionsV1:
    source_dataset_name: str
    source_dataset_identity_hash: str
    selected_source_inventory_hash: str
    source_episode_count: int
    included_episode_indices: tuple[int, ...]
    excluded_episode_indices: tuple[int, ...]
    issues: tuple[CameraSemanticVideoIntegrityIssueV1, ...]
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        name = str(self.source_dataset_name or "").strip()
        included = tuple(int(value) for value in self.included_episode_indices)
        excluded = tuple(int(value) for value in self.excluded_episode_indices)
        issues = tuple(self.issues)
        if not name:
            raise CameraSemanticPreparedViewContractError(
                "source_dataset_name must be non-empty."
            )
        if included != tuple(sorted(set(included))):
            raise CameraSemanticPreparedViewContractError(
                "included_episode_indices must be a sorted unique sequence."
            )
        if not excluded or excluded != tuple(sorted(set(excluded))):
            raise CameraSemanticPreparedViewContractError(
                "excluded_episode_indices must be a non-empty sorted unique sequence."
            )
        if set(included) & set(excluded):
            raise CameraSemanticPreparedViewContractError(
                "Included and excluded episode indices must be disjoint."
            )
        if int(self.source_episode_count) != len(included) + len(excluded):
            raise CameraSemanticPreparedViewContractError(
                "source_episode_count must equal included plus excluded episodes."
            )
        if not issues or {item.episode_index for item in issues} != set(excluded):
            raise CameraSemanticPreparedViewContractError(
                "Every excluded episode must have at least one video-integrity issue."
            )
        issue_keys = tuple(
            (item.episode_index, item.camera_key, item.relative_path, item.reason)
            for item in issues
        )
        if issue_keys != tuple(sorted(set(issue_keys))):
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity issues must be sorted and unique."
            )
        object.__setattr__(self, "source_dataset_name", name)
        object.__setattr__(
            self,
            "source_dataset_identity_hash",
            _sha256(
                self.source_dataset_identity_hash,
                name="source_dataset_identity_hash",
            ),
        )
        object.__setattr__(
            self,
            "selected_source_inventory_hash",
            _sha256(
                self.selected_source_inventory_hash,
                name="selected_source_inventory_hash",
            ),
        )
        object.__setattr__(self, "source_episode_count", int(self.source_episode_count))
        object.__setattr__(self, "included_episode_indices", included)
        object.__setattr__(self, "excluded_episode_indices", excluded)
        object.__setattr__(self, "issues", issues)
        expected = self.semantic_content_hash
        supplied = str(self.receipt_hash or "").strip().lower()
        if supplied and supplied != expected:
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity exclusions receipt hash mismatch."
            )
        object.__setattr__(self, "receipt_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_SCHEMA_V1,
            "complete": True,
            "policy": "exclude_entire_episode_if_any_video_is_invalid",
            "source_dataset_name": self.source_dataset_name,
            "source_dataset_identity_hash": self.source_dataset_identity_hash,
            "selected_source_inventory_hash": self.selected_source_inventory_hash,
            "source_episode_count": self.source_episode_count,
            "included_episode_indices": list(self.included_episode_indices),
            "excluded_episode_indices": list(self.excluded_episode_indices),
            "issues": [item.to_dict() for item in self.issues],
            "source_mutated": False,
        }
        if include_hash:
            payload["receipt_hash"] = self.receipt_hash
        return payload

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain=CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> CameraSemanticVideoIntegrityExclusionsV1:
        if (
            str(value.get("schema") or "")
            != CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_SCHEMA_V1
            or value.get("complete") is not True
            or value.get("policy")
            != "exclude_entire_episode_if_any_video_is_invalid"
            or value.get("source_mutated") is not False
        ):
            raise CameraSemanticPreparedViewContractError(
                "Unsupported or incomplete video-integrity exclusions receipt."
            )
        return cls(
            source_dataset_name=str(value.get("source_dataset_name") or ""),
            source_dataset_identity_hash=str(
                value.get("source_dataset_identity_hash") or ""
            ),
            selected_source_inventory_hash=str(
                value.get("selected_source_inventory_hash") or ""
            ),
            source_episode_count=int(value.get("source_episode_count", -1)),
            included_episode_indices=tuple(
                value.get("included_episode_indices") or ()
            ),
            excluded_episode_indices=tuple(
                value.get("excluded_episode_indices") or ()
            ),
            issues=tuple(
                CameraSemanticVideoIntegrityIssueV1.from_dict(item)
                for item in value.get("issues") or ()
            ),
            receipt_hash=str(value.get("receipt_hash") or ""),
        )


@dataclass(frozen=True)
class CameraSemanticPreparedViewManifestV1:
    source_dataset_name: str
    source_dataset_identity_hash: str
    selected_source_inventory_hash: str
    camera_binding_hash: str
    camera_materialization_receipt_hash: str
    plan_hash: str
    episode_indices: tuple[int, ...]
    files: tuple[CameraSemanticPreparedViewFileV1, ...]
    excluded_episode_indices: tuple[int, ...] = ()
    video_integrity_exclusions_hash: str | None = None
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        name = str(self.source_dataset_name or "").strip()
        if not name:
            raise CameraSemanticPreparedViewContractError(
                "source_dataset_name must be non-empty."
            )
        episodes = tuple(int(value) for value in self.episode_indices)
        if episodes != tuple(sorted(set(episodes))):
            raise CameraSemanticPreparedViewContractError(
                "episode_indices must be a sorted unique sequence."
            )
        files = tuple(self.files)
        excluded = tuple(int(value) for value in self.excluded_episode_indices)
        paths = tuple(item.relative_path for item in files)
        if not files or paths != tuple(sorted(set(paths))):
            raise CameraSemanticPreparedViewContractError(
                "files must be a non-empty path-sorted unique sequence."
            )
        if excluded != tuple(sorted(set(excluded))) or set(episodes) & set(excluded):
            raise CameraSemanticPreparedViewContractError(
                "excluded_episode_indices must be sorted, unique and disjoint from episode_indices."
            )
        exclusions_hash = self.video_integrity_exclusions_hash
        if bool(excluded) != bool(exclusions_hash):
            raise CameraSemanticPreparedViewContractError(
                "Excluded episodes require exactly one exclusions receipt hash."
            )
        if not episodes and not excluded:
            raise CameraSemanticPreparedViewContractError(
                "An empty prepared view must document excluded source episodes."
            )
        forbidden = tuple(
            path
            for path in paths
            if path == "tactile"
            or path.startswith(
                (
                    "tactile/",
                    "meta/modality_artifacts/tactile/",
                    "meta/tactile_bundle_v1/",
                )
            )
            or any(part.endswith(".lock") for part in path.split("/"))
        )
        if forbidden:
            raise CameraSemanticPreparedViewContractError(
                "A camera-only prepared view cannot contain tactile artifacts "
                "or runtime lock files."
            )
        object.__setattr__(self, "source_dataset_name", name)
        object.__setattr__(
            self,
            "source_dataset_identity_hash",
            _sha256(
                self.source_dataset_identity_hash,
                name="source_dataset_identity_hash",
            ),
        )
        object.__setattr__(
            self,
            "selected_source_inventory_hash",
            _sha256(
                self.selected_source_inventory_hash,
                name="selected_source_inventory_hash",
            ),
        )
        object.__setattr__(
            self,
            "camera_binding_hash",
            _sha256(self.camera_binding_hash, name="camera_binding_hash"),
        )
        object.__setattr__(
            self,
            "camera_materialization_receipt_hash",
            _sha256(
                self.camera_materialization_receipt_hash,
                name="camera_materialization_receipt_hash",
            ),
        )
        object.__setattr__(self, "plan_hash", _sha256(self.plan_hash, name="plan_hash"))
        object.__setattr__(self, "episode_indices", episodes)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "excluded_episode_indices", excluded)
        object.__setattr__(
            self,
            "video_integrity_exclusions_hash",
            None
            if exclusions_hash is None
            else _sha256(
                exclusions_hash,
                name="video_integrity_exclusions_hash",
            ),
        )
        expected = self.semantic_content_hash
        supplied = str(self.manifest_hash or "").strip().lower()
        if supplied and supplied != expected:
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view manifest hash mismatch."
            )
        object.__setattr__(self, "manifest_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": CAMERA_SEMANTIC_PREPARED_VIEW_SCHEMA_V1,
            "complete": True,
            "view_kind": "camera_corrected_numeric_video_projection",
            "source_dataset_name": self.source_dataset_name,
            "source_dataset_identity_hash": self.source_dataset_identity_hash,
            "selected_source_inventory_hash": self.selected_source_inventory_hash,
            "camera_binding_hash": self.camera_binding_hash,
            "camera_materialization_receipt_hash": (
                self.camera_materialization_receipt_hash
            ),
            "plan_hash": self.plan_hash,
            "episode_indices": list(self.episode_indices),
            "modalities": ["numeric", "video"],
            "excluded_modalities": ["tactile"],
            "tactile_payloads_inspected": False,
            "full_v23_dataset": False,
            "requires_explicit_consumer_opt_in": True,
            "source_mutated": False,
            "output_uses_fresh_inodes": True,
            "frame_gap_policy": "preserve_on_physical_camera",
            "files": [item.to_dict() for item in self.files],
        }
        if self.excluded_episode_indices:
            payload["excluded_episode_indices"] = list(
                self.excluded_episode_indices
            )
            payload["video_integrity_exclusions_path"] = (
                CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH
            )
            payload["video_integrity_exclusions_hash"] = (
                self.video_integrity_exclusions_hash
            )
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain=CAMERA_SEMANTIC_PREPARED_VIEW_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> CameraSemanticPreparedViewManifestV1:
        if str(value.get("schema") or "") != CAMERA_SEMANTIC_PREPARED_VIEW_SCHEMA_V1:
            raise CameraSemanticPreparedViewContractError(
                "Unsupported camera-semantic prepared-view schema."
            )
        if value.get("complete") is not True:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view is incomplete."
            )
        if value.get("modalities") != ["numeric", "video"]:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view must contain numeric and video modalities."
            )
        if value.get("excluded_modalities") != ["tactile"]:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view must explicitly exclude tactile."
            )
        if value.get("tactile_payloads_inspected") is not False:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view must not inspect tactile payloads."
            )
        if value.get("full_v23_dataset") is not False:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view must not claim to be a full v2.3 dataset."
            )
        if value.get("requires_explicit_consumer_opt_in") is not True:
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view requires explicit consumer opt-in."
            )
        return cls(
            source_dataset_name=str(value.get("source_dataset_name") or ""),
            source_dataset_identity_hash=str(
                value.get("source_dataset_identity_hash") or ""
            ),
            selected_source_inventory_hash=str(
                value.get("selected_source_inventory_hash") or ""
            ),
            camera_binding_hash=str(value.get("camera_binding_hash") or ""),
            camera_materialization_receipt_hash=str(
                value.get("camera_materialization_receipt_hash") or ""
            ),
            plan_hash=str(value.get("plan_hash") or ""),
            episode_indices=tuple(value.get("episode_indices") or ()),
            files=tuple(
                CameraSemanticPreparedViewFileV1.from_dict(item)
                for item in value.get("files") or ()
            ),
            excluded_episode_indices=tuple(
                value.get("excluded_episode_indices") or ()
            ),
            video_integrity_exclusions_hash=(
                None
                if value.get("video_integrity_exclusions_hash") is None
                else str(value.get("video_integrity_exclusions_hash"))
            ),
            manifest_hash=str(value.get("manifest_hash") or ""),
        )


__all__ = [
    "CAMERA_SEMANTIC_PREPARED_VIEW_HASH_DOMAIN_V1",
    "CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH",
    "CAMERA_SEMANTIC_PREPARED_VIEW_SCHEMA_V1",
    "CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_HASH_DOMAIN_V1",
    "CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH",
    "CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_SCHEMA_V1",
    "CameraSemanticPreparedViewContractError",
    "CameraSemanticPreparedViewFileV1",
    "CameraSemanticPreparedViewManifestV1",
    "CameraSemanticVideoIntegrityExclusionsV1",
    "CameraSemanticVideoIntegrityIssueV1",
]
