"""Strict source-Episode and annotation resolution for prepared views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..episode_curation import source_episode_manifest_hash
from ..progress_annotation_jobs import (
    ProgressAnnotationWorkItem,
    load_episode_sample_axis,
)
from ..progress_failure_trigger_artifacts import (
    FAILURE_TRIGGER_RELATIVE_PATH,
    load_optional_failure_trigger_annotation_set,
)
from ..progress_failure_triggers import failure_trigger_sample_axis_hash
from ..progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
    STAGE_ANNOTATION_RELATIVE_ROOT,
    load_stage_annotation_set,
)
from ..progress_stages import resolve_stage_training_index
from ..training_semantics import (
    EpisodeTrainingSemanticIndex,
    build_episode_training_semantic_index,
)
from .io import sha256_file, source_relative_path
from .models import (
    MANIFEST_FILE,
    PREPARED_SOURCE_EPISODE_HASH_DOMAIN_V1,
    PreparedSourceDataset,
    PreparedTrainingViewError,
)


@dataclass(frozen=True)
class PreparedEpisode:
    source: PreparedSourceDataset
    item: ProgressAnnotationWorkItem
    source_dataset_fingerprint: str
    source_episode_record_hash: str
    source_episode_content_hash: str
    parquet_sha256: str
    video_sha256_by_key: Mapping[str, str]
    annotation: LoadedStageAnnotationSet | None
    failure_trigger_hash: str | None
    semantic_index: EpisodeTrainingSemanticIndex
    projection_schema: tuple[tuple[str, str, bool], ...]
    inventory: tuple[Mapping[str, Any], ...]


def _annotation_candidates(
    item: ProgressAnnotationWorkItem,
) -> tuple[LoadedStageAnnotationSet, ...]:
    root = (
        item.dataset_root
        / STAGE_ANNOTATION_RELATIVE_ROOT
        / item.episode_uuid
    )
    if not root.is_dir():
        return ()
    candidates: list[LoadedStageAnnotationSet] = []
    for child in sorted((path for path in root.iterdir() if path.is_dir())):
        if not (child / MANIFEST_FILE).is_file():
            continue
        loaded = load_stage_annotation_set(child)
        if (
            loaded.binding.episode_capture_identity_hash
            != item.episode_identity_hash
            or loaded.binding.source_progress_manifest_hash
            != item.progress_manifest_hash
            or item.progress.stage_profile is None
            or loaded.binding.task_stage_profile_hash
            != item.progress.stage_profile.semantic_content_hash
            or loaded.binding.bound_clock_domain_id
            != item.progress.timeline.clock_domain_id
        ):
            raise PreparedTrainingViewError(
                f"Stage annotation {child} does not bind its source Episode."
            )
        candidates.append(loaded)
    return tuple(candidates)


def _select_annotation(
    item: ProgressAnnotationWorkItem,
    selector: str | None,
) -> LoadedStageAnnotationSet | None:
    candidates = _annotation_candidates(item)
    if selector is None:
        if len(candidates) > 1:
            raise PreparedTrainingViewError(
                "Episode has multiple annotation sets; annotation_selection "
                f"must bind one explicitly: {item.episode_identity_hash}."
            )
        return None if not candidates else candidates[0]
    selected = tuple(
        candidate
        for candidate in candidates
        if selector
        in {
            candidate.binding.annotation_set_id,
            candidate.manifest.semantic_content_hash,
        }
    )
    if len(selected) != 1:
        raise PreparedTrainingViewError(
            "Explicit annotation selector must match exactly one sealed set "
            f"for Episode {item.episode_identity_hash}."
        )
    return selected[0]


def _projection_schema(
    parquet_path: Path,
    column_projection: tuple[str, ...],
) -> tuple[tuple[str, str, bool], ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PreparedTrainingViewError(
            "pyarrow is required to build a prepared training view."
        ) from exc
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
    except Exception as exc:
        raise PreparedTrainingViewError(
            f"Cannot inspect source Parquet {parquet_path}: {exc}"
        ) from exc
    fields: list[tuple[str, str, bool]] = []
    for name in column_projection:
        index = schema.get_field_index(name)
        if index < 0:
            raise PreparedTrainingViewError(
                f"Source Parquet is missing projected column {name!r}: "
                f"{parquet_path}."
            )
        field = schema.field(index)
        fields.append((name, str(field.type), bool(field.nullable)))
    return tuple(fields)


def _inventory_record(
    *,
    source: PreparedSourceDataset,
    path: Path,
    role: str,
    known_sha256: str | None = None,
) -> Mapping[str, Any]:
    return {
        "source_dataset_id": source.source_dataset_id,
        "relative_path": source_relative_path(source.dataset_root, path),
        "role": role,
        "requirement": "REQUIRED_FOR_VIEW",
        "size_bytes": path.stat().st_size,
        "sha256": known_sha256 or sha256_file(path),
    }


def prepare_episode(
    *,
    source: PreparedSourceDataset,
    item: ProgressAnnotationWorkItem,
    dataset_fingerprint: str,
    selector: str | None,
    column_projection: tuple[str, ...],
    modality_selection: tuple[str, ...],
) -> PreparedEpisode:
    record_hash = source_episode_manifest_hash(
        source.dataset_root,
        item.episode_index,
    )
    if record_hash is None:
        raise PreparedTrainingViewError(
            f"Episode {item.episode_index} is absent from episodes.jsonl."
        )
    parquet_hash = sha256_file(item.parquet_path)
    video_hashes = {
        key: sha256_file(path)
        for key, path in sorted(item.video_paths.items())
    }
    annotation = _select_annotation(item, selector)
    sample_axis = load_episode_sample_axis(item)
    stage_index = None
    failure_triggers = ()
    failure_trigger_hash = None
    if annotation is not None:
        assert item.progress.stage_profile is not None
        stage_index = resolve_stage_training_index(
            progress_profile=item.progress.profile,
            progress_binding=item.progress.binding,
            stage_profile=item.progress.stage_profile,
            stage_entry_binding=item.progress.stage_entry_binding,
            timeline=item.progress.timeline,
            annotations=annotation.bundle,
            sample_axis=sample_axis,
        )
        trigger_set = load_optional_failure_trigger_annotation_set(
            annotation.annotation_dir,
            expected_episode_capture_identity_hash=item.episode_identity_hash,
            expected_source_progress_manifest_hash=item.progress_manifest_hash,
            expected_task_stage_profile_hash=(
                item.progress.stage_profile.semantic_content_hash
            ),
            expected_stage_annotation_set_id=(
                annotation.binding.annotation_set_id
            ),
            expected_stage_annotation_bundle_hash=(
                annotation.bundle.semantic_content_hash
            ),
            expected_sample_axis_hash=failure_trigger_sample_axis_hash(sample_axis),
            expected_clock_domain_id=sample_axis.clock_domain_id,
        )
        if trigger_set is not None:
            failure_triggers = trigger_set.annotations
            failure_trigger_hash = trigger_set.semantic_content_hash
    semantic_index = build_episode_training_semantic_index(
        timeline=item.progress.timeline,
        sample_axis=sample_axis,
        stage_index=stage_index,
        failure_triggers=failure_triggers,
        episode_entry_condition=(
            "NOMINAL"
            if item.progress.stage_entry_binding is None
            else item.progress.stage_entry_binding.entry_condition
        ),
    )
    projection = _projection_schema(item.parquet_path, column_projection)

    inventory: list[Mapping[str, Any]] = [
        _inventory_record(
            source=source,
            path=source.dataset_root / "meta" / "episodes.jsonl",
            role="EPISODE_CATALOG",
        ),
        _inventory_record(
            source=source,
            path=item.parquet_path,
            role="EPISODE_PARQUET",
            known_sha256=parquet_hash,
        ),
    ]
    if "video" in modality_selection:
        inventory.extend(
            _inventory_record(
                source=source,
                path=path,
                role=f"EPISODE_VIDEO:{key}",
                known_sha256=video_hashes[key],
            )
            for key, path in sorted(item.video_paths.items())
        )
    for path in sorted(item.progress_episode_dir.iterdir()):
        if path.is_file():
            inventory.append(
                _inventory_record(
                    source=source,
                    path=path,
                    role="PROGRESS_ARTIFACT",
                )
            )
    if annotation is not None:
        for path in sorted(annotation.annotation_dir.iterdir()):
            if path.is_file():
                inventory.append(
                    _inventory_record(
                        source=source,
                        path=path,
                        role=(
                            "FAILURE_TRIGGER_ANNOTATION"
                            if path.name == FAILURE_TRIGGER_RELATIVE_PATH
                            else "STAGE_ANNOTATION_ARTIFACT"
                        ),
                    )
                )

    source_content_hash = canonical_content_hash(
        {
            "episode_capture_identity_hash": item.episode_identity_hash,
            "source_episode_record_hash": record_hash,
            "parquet_sha256": parquet_hash,
            "video_sha256_by_key": dict(video_hashes),
            "progress_manifest_hash": item.progress_manifest_hash,
        },
        domain=PREPARED_SOURCE_EPISODE_HASH_DOMAIN_V1,
    )
    return PreparedEpisode(
        source=source,
        item=item,
        source_dataset_fingerprint=dataset_fingerprint,
        source_episode_record_hash=record_hash,
        source_episode_content_hash=source_content_hash,
        parquet_sha256=parquet_hash,
        video_sha256_by_key=video_hashes,
        annotation=annotation,
        failure_trigger_hash=failure_trigger_hash,
        semantic_index=semantic_index,
        projection_schema=projection,
        inventory=tuple(inventory),
    )


__all__ = ["PreparedEpisode", "prepare_episode"]
