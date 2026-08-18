"""Resolve legacy stage annotations with an optional detail overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_contracts.progress_detail_annotations import (
    DetailStageAnnotationBundleV1,
    DetailStageAnnotationSetBindingV1,
    validate_detail_stage_annotation_set,
)
from dataset_contracts.progress_stage_annotations import StageAnnotationBundleV1
from dataset_contracts.progress_stages import TaskStageProfileV1

from .progress_detail_annotation_artifacts import (
    DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
    DetailStageAnnotationArtifactReadError,
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set,
)
from .progress_stage_annotation_artifacts import LoadedStageAnnotationSet


DETAIL_ANNOTATION_RESOLUTION_LEVELS_V1 = frozenset(
    {"COARSE_FALLBACK", "DETAIL_COMPLETE"}
)


class DetailStageAnnotationResolutionError(ValueError):
    """Raised when a present detail overlay cannot be used safely."""


@dataclass(frozen=True)
class ResolvedDetailStageAnnotationView:
    """Effective bundle plus an explicit resolution level.

    ``COARSE_FALLBACK`` means that no detail overlay was published.  It is
    deliberately distinct from ``DETAIL_COMPLETE`` so downstream analysis
    cannot mistake an old coarse annotation for a newly inferred detail one.
    """

    resolution_level: str
    bundle: StageAnnotationBundleV1
    detail_annotation_set: LoadedDetailStageAnnotationSet | None


def _merged_bundle(
    *,
    base_bundle: StageAnnotationBundleV1,
    target_progress_step_id: str,
    detail_bundle: DetailStageAnnotationBundleV1,
) -> StageAnnotationBundleV1:
    coarse_occurrences = tuple(
        item
        for item in base_bundle.occurrences
        if item.progress_step_id != target_progress_step_id
    )
    return StageAnnotationBundleV1(
        occurrences=coarse_occurrences + detail_bundle.occurrences,
        abnormal_intervals=base_bundle.abnormal_intervals,
        progress_boundary_overrides=base_bundle.progress_boundary_overrides,
        capture_end_step=base_bundle.capture_end_step,
        episode_abnormal=base_bundle.episode_abnormal,
        episode_review_request=base_bundle.episode_review_request,
    )


def resolve_detail_stage_annotation_view(
    *,
    source_progress_manifest: Any,
    base_annotation_set: LoadedStageAnnotationSet,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
    detail_annotation_set: LoadedDetailStageAnnotationSet | None = None,
) -> ResolvedDetailStageAnnotationView:
    """Return coarse fallback or a validated effective detail bundle.

    A missing overlay is compatible with every existing legacy dataset.  A
    present but malformed overlay is an error instead of a fallback: silently
    hiding corruption would make analysis appear successful while using the
    wrong annotation level.
    """

    if not isinstance(base_annotation_set, LoadedStageAnnotationSet):
        raise TypeError(
            "base_annotation_set must be LoadedStageAnnotationSet."
        )
    if detail_annotation_set is None:
        return ResolvedDetailStageAnnotationView(
            resolution_level="COARSE_FALLBACK",
            bundle=base_annotation_set.bundle,
            detail_annotation_set=None,
        )
    if not isinstance(
        detail_annotation_set,
        LoadedDetailStageAnnotationSet,
    ):
        raise TypeError(
            "detail_annotation_set must be LoadedDetailStageAnnotationSet."
        )
    try:
        validate_detail_stage_annotation_set(
            source_progress_manifest=source_progress_manifest,
            base_binding=base_annotation_set.binding,
            base_bundle=base_annotation_set.bundle,
            base_manifest=base_annotation_set.manifest,
            base_stage_profile=base_stage_profile,
            detail_stage_profile=detail_stage_profile,
            binding=detail_annotation_set.binding,
            bundle=detail_annotation_set.bundle,
        )
        bundle = _merged_bundle(
            base_bundle=base_annotation_set.bundle,
            target_progress_step_id=(
                detail_annotation_set.binding.target_progress_step_id
            ),
            detail_bundle=detail_annotation_set.bundle,
        )
    except (TypeError, ValueError) as exc:
        raise DetailStageAnnotationResolutionError(str(exc)) from exc
    return ResolvedDetailStageAnnotationView(
        resolution_level="DETAIL_COMPLETE",
        bundle=bundle,
        detail_annotation_set=detail_annotation_set,
    )


def load_latest_detail_stage_annotation_set(
    *,
    dataset_root: str | Path,
    episode_uuid: str,
    source_progress_manifest: Any,
    base_annotation_set: LoadedStageAnnotationSet,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
) -> LoadedDetailStageAnnotationSet | None:
    """Find the newest valid overlay for one exact coarse annotation set.

    Older overlays remain immutable history and are skipped when they bind to
    a different coarse annotation version.  A directory that claims to be a
    visible detail set but is internally malformed raises instead of being
    silently downgraded to coarse analysis.
    """

    normalized_episode_uuid = str(episode_uuid).strip()
    if (
        not normalized_episode_uuid
        or normalized_episode_uuid in {".", ".."}
        or Path(normalized_episode_uuid).name != normalized_episode_uuid
    ):
        raise DetailStageAnnotationResolutionError(
            "episode_uuid is not a safe detail annotation path component."
        )
    root = (
        Path(dataset_root).expanduser().resolve()
        / DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT
        / normalized_episode_uuid
    )
    if not root.is_dir():
        return None
    matches: list[LoadedDetailStageAnnotationSet] = []
    try:
        children = tuple(sorted(root.iterdir(), key=lambda value: value.name))
    except OSError as exc:
        raise DetailStageAnnotationResolutionError(
            f"Cannot inspect detail annotation root {root}: {exc}"
        ) from exc
    for child in children:
        if not child.is_dir() or not (child / "manifest.cjson").is_file():
            continue
        try:
            loaded = load_detail_stage_annotation_set(child)
        except DetailStageAnnotationArtifactReadError as exc:
            raise DetailStageAnnotationResolutionError(str(exc)) from exc
        binding = loaded.binding
        if (
            binding.episode_capture_identity_hash
            != base_annotation_set.binding.episode_capture_identity_hash
            or binding.source_progress_manifest_hash
            != base_annotation_set.binding.source_progress_manifest_hash
            or binding.base_annotation_set_id
            != base_annotation_set.binding.annotation_set_id
            or binding.base_annotation_set_manifest_hash
            != base_annotation_set.manifest.semantic_content_hash
            or binding.base_task_stage_profile_hash
            != base_stage_profile.semantic_content_hash
            or binding.detail_task_stage_profile_hash
            != detail_stage_profile.semantic_content_hash
        ):
            continue
        try:
            validate_detail_stage_annotation_set(
                source_progress_manifest=source_progress_manifest,
                base_binding=base_annotation_set.binding,
                base_bundle=base_annotation_set.bundle,
                base_manifest=base_annotation_set.manifest,
                base_stage_profile=base_stage_profile,
                detail_stage_profile=detail_stage_profile,
                binding=binding,
                bundle=loaded.bundle,
            )
        except (TypeError, ValueError) as exc:
            raise DetailStageAnnotationResolutionError(str(exc)) from exc
        matches.append(loaded)
    if not matches:
        return None
    return max(
        matches,
        key=lambda value: (
            value.binding.created_at,
            value.binding.annotation_set_id,
        ),
    )


def resolve_latest_detail_stage_annotation_view(
    *,
    dataset_root: str | Path,
    episode_uuid: str,
    source_progress_manifest: Any,
    base_annotation_set: LoadedStageAnnotationSet,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
) -> ResolvedDetailStageAnnotationView:
    """Resolve the exact current overlay, or explicitly fall back to coarse."""

    detail_annotation_set = load_latest_detail_stage_annotation_set(
        dataset_root=dataset_root,
        episode_uuid=episode_uuid,
        source_progress_manifest=source_progress_manifest,
        base_annotation_set=base_annotation_set,
        base_stage_profile=base_stage_profile,
        detail_stage_profile=detail_stage_profile,
    )
    return resolve_detail_stage_annotation_view(
        source_progress_manifest=source_progress_manifest,
        base_annotation_set=base_annotation_set,
        base_stage_profile=base_stage_profile,
        detail_stage_profile=detail_stage_profile,
        detail_annotation_set=detail_annotation_set,
    )


__all__ = [
    "DETAIL_ANNOTATION_RESOLUTION_LEVELS_V1",
    "DetailStageAnnotationResolutionError",
    "ResolvedDetailStageAnnotationView",
    "load_latest_detail_stage_annotation_set",
    "resolve_detail_stage_annotation_view",
    "resolve_latest_detail_stage_annotation_view",
]
