"""Crash-safe persistence for additive detail-stage annotation overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.progress_detail_annotations import (
    DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
    DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    DetailStageAnnotationBundleV1,
    DetailStageAnnotationManifestV1,
    DetailStageAnnotationSetBindingV1,
    validate_detail_stage_annotation_set,
)
from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_stages import TaskStageProfileV1
from dataset_interface.progress_detail_annotation_artifacts import (
    DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set as _load_detail_stage_annotation_set,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
)

from ..immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)


class DetailStageAnnotationStoreError(RuntimeError):
    """Raised when a detail overlay is incomplete or conflicts with its base."""


@dataclass(frozen=True)
class DetailStageAnnotationCommitReport:
    annotation_dir: Path
    manifest: DetailStageAnnotationManifestV1
    idempotent: bool


def _safe_component(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
    ):
        raise ValueError(f"{name} is not path-safe.")
    return normalized


def commit_detail_stage_annotation_set(
    *,
    dataset_root: str | Path,
    episode_uuid: str,
    source_progress_manifest: ProgressEpisodeManifestV1,
    base_annotation_set: LoadedStageAnnotationSet,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
    binding: DetailStageAnnotationSetBindingV1,
    bundle: DetailStageAnnotationBundleV1,
) -> DetailStageAnnotationCommitReport:
    """Publish one immutable overlay; its manifest is the visibility marker."""

    normalized_episode_uuid = _safe_component(
        episode_uuid,
        name="episode_uuid",
    )
    annotation_set_id = _safe_component(
        binding.annotation_set_id,
        name="annotation_set_id",
    )
    if not isinstance(base_annotation_set, LoadedStageAnnotationSet):
        raise TypeError(
            "base_annotation_set must be LoadedStageAnnotationSet."
        )
    validate_detail_stage_annotation_set(
        source_progress_manifest=source_progress_manifest,
        base_binding=base_annotation_set.binding,
        base_bundle=base_annotation_set.bundle,
        base_manifest=base_annotation_set.manifest,
        base_stage_profile=base_stage_profile,
        detail_stage_profile=detail_stage_profile,
        binding=binding,
        bundle=bundle,
    )
    root = Path(dataset_root).expanduser().resolve()
    annotation_dir = (
        root
        / DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT
        / normalized_episode_uuid
        / annotation_set_id
    )
    expected_parent = (
        root / DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT / normalized_episode_uuid
    )
    if annotation_dir.parent != expected_parent:
        raise ValueError("Detail annotation path escaped its owned root.")
    manifest = DetailStageAnnotationManifestV1(
        annotation_set_binding_hash=binding.semantic_content_hash,
        annotation_bundle_hash=bundle.semantic_content_hash,
    )
    try:
        binding_idempotent = publish_immutable_canonical(
            annotation_dir / "binding.cjson",
            binding.to_payload(),
            domain=DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
            sync_parent=False,
        )
        bundle_idempotent = publish_immutable_canonical(
            annotation_dir / "annotations.cjson",
            bundle.to_payload(),
            domain=DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
            sync_parent=False,
        )
        fsync_directory(annotation_dir)
        manifest_idempotent = publish_immutable_canonical(
            annotation_dir / "manifest.cjson",
            manifest.to_payload(),
            domain=DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DetailStageAnnotationStoreError(
            f"Cannot publish immutable detail annotation set: {exc}"
        ) from exc
    return DetailStageAnnotationCommitReport(
        annotation_dir=annotation_dir,
        manifest=manifest,
        idempotent=(
            binding_idempotent
            and bundle_idempotent
            and manifest_idempotent
        ),
    )


def load_detail_stage_annotation_set(
    annotation_dir: str | Path,
) -> LoadedDetailStageAnnotationSet:
    """Compatibility reader for the dataset-interface-owned loader."""

    try:
        return _load_detail_stage_annotation_set(annotation_dir)
    except ValueError as exc:
        raise DetailStageAnnotationStoreError(str(exc)) from exc


__all__ = [
    "DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT",
    "DetailStageAnnotationCommitReport",
    "DetailStageAnnotationStoreError",
    "commit_detail_stage_annotation_set",
    "load_detail_stage_annotation_set",
]
