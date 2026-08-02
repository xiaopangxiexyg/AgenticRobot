"""Atomic prepared-view construction outside the live collection path."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
)

from ..episode_curation import source_dataset_base_fingerprint
from ..progress_annotation_jobs import list_progress_annotation_work_items
from ..training_semantics import (
    TRAINING_DOMAIN_FAILURE,
    TRAINING_DOMAIN_NOMINAL,
)
from .io import (
    fsync_directory,
    normalized_modalities,
    normalized_projection,
    relative_locator,
    sha256_file,
    source_relative_path,
    write_bytes,
    write_cjsonl,
)
from .models import (
    ANCHORS_FILE,
    DERIVED_FILES,
    EPISODES_FILE,
    INTERVALS_FILE,
    MANIFEST_FILE,
    PREPARED_SELECTED_EPISODE_DOMAIN_V1,
    PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    PREPARED_SOURCE_INVENTORY_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_SCHEMA_V1,
    SELECTED_EPISODES_FILE,
    SOURCE_DATASETS_FILE,
    SOURCE_INVENTORY_FILE,
    VIEW_RELATIVE_ROOT,
    PreparedSourceDataset,
    PreparedTrainingViewBuildResult,
    PreparedTrainingViewError,
    arrow_schemas,
)
from .reader import load_prepared_training_view
from .sources import PreparedEpisode, prepare_episode


def _table_rows(
    episodes: Sequence[PreparedEpisode],
    *,
    modality_selection: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for ordinal, episode in enumerate(episodes):
        item = episode.item
        annotation = episode.annotation
        camera_keys = (
            list(sorted(item.video_paths))
            if "video" in modality_selection
            else []
        )
        video_paths = (
            [
                source_relative_path(item.dataset_root, item.video_paths[key])
                for key in camera_keys
            ]
            if "video" in modality_selection
            else []
        )
        episode_rows.append(
            {
                "organized_episode_ordinal": ordinal,
                "source_dataset_id": episode.source.source_dataset_id,
                "source_episode_index": item.episode_index,
                "episode_uuid": item.episode_uuid,
                "episode_capture_identity_hash": item.episode_identity_hash,
                "source_episode_content_hash": episode.source_episode_content_hash,
                "frame_count": item.frame_count,
                "clock_domain_id": item.progress.timeline.clock_domain_id,
                "start_timestamp": item.start_timestamp,
                "end_timestamp": item.end_timestamp,
                "parquet_relative_path": source_relative_path(
                    item.dataset_root, item.parquet_path
                ),
                "camera_keys": camera_keys,
                "video_relative_paths": video_paths,
                "progress_profile_hash": item.progress.profile.semantic_content_hash,
                "stage_profile_hash": (
                    None
                    if item.progress.stage_profile is None
                    else item.progress.stage_profile.semantic_content_hash
                ),
                "progress_manifest_hash": item.progress_manifest_hash,
                "annotation_set_id": (
                    None if annotation is None else annotation.binding.annotation_set_id
                ),
                "annotation_manifest_hash": (
                    None
                    if annotation is None
                    else annotation.manifest.semantic_content_hash
                ),
                "annotation_status": (
                    "UNANNOTATED" if annotation is None else "SELECTED"
                ),
                "failure_trigger_hash": episode.failure_trigger_hash,
            }
        )
        for span in episode.semantic_index.spans:
            interval_rows.append(
                {
                    "organized_episode_ordinal": ordinal,
                    "episode_capture_identity_hash": item.episode_identity_hash,
                    "sample_begin": span.sample_interval.sample_index_begin,
                    "sample_end_exclusive": (
                        span.sample_interval.sample_index_end_exclusive
                    ),
                    "semantic_kind": span.semantic_kind,
                    "training_domain": None,
                    "progress_step_id": span.progress_step_id,
                    "attempt_id": span.attempt_id,
                    "stage_id": span.stage_id,
                    "outcome": span.outcome,
                    "entry_condition": span.entry_condition,
                    "source_id": span.source_id,
                    "source_failure_fact_id": span.source_failure_fact_id,
                    "resolution_kind": span.resolution_kind,
                }
            )
        for domain, intervals in (
            (TRAINING_DOMAIN_NOMINAL, episode.semantic_index.nominal_intervals),
            (TRAINING_DOMAIN_FAILURE, episode.semantic_index.failure_intervals),
        ):
            for interval in intervals:
                interval_rows.append(
                    {
                        "organized_episode_ordinal": ordinal,
                        "episode_capture_identity_hash": item.episode_identity_hash,
                        "sample_begin": interval.sample_index_begin,
                        "sample_end_exclusive": interval.sample_index_end_exclusive,
                        "semantic_kind": "TRAINING_ELIGIBLE_INTERVAL",
                        "training_domain": domain,
                        "progress_step_id": None,
                        "attempt_id": None,
                        "stage_id": None,
                        "outcome": None,
                        "entry_condition": None,
                        "source_id": None,
                        "source_failure_fact_id": None,
                        "resolution_kind": "COMPOSED_V1",
                    }
                )
        for anchor in episode.semantic_index.anchors:
            anchor_rows.append(
                {
                    "organized_episode_ordinal": ordinal,
                    "episode_capture_identity_hash": item.episode_identity_hash,
                    "sample_index": anchor.sample_index,
                    "semantic_kind": anchor.semantic_kind,
                    "progress_step_id": anchor.progress_step_id,
                    "attempt_id": anchor.attempt_id,
                    "stage_id": anchor.stage_id,
                    "source_id": anchor.source_id,
                    "source_failure_fact_id": anchor.source_failure_fact_id,
                }
            )
    return episode_rows, interval_rows, anchor_rows


def _resolve_inputs(
    *,
    organized_root: Path,
    sources: Sequence[PreparedSourceDataset],
    selection: Mapping[str, str],
    included: frozenset[str] | None,
    projection: tuple[str, ...],
    modalities: tuple[str, ...],
) -> tuple[tuple[PreparedEpisode, ...], list[dict[str, Any]]]:
    prepared: list[PreparedEpisode] = []
    source_rows: list[dict[str, Any]] = []
    seen_input_identities: set[str] = set()
    for source in sorted(sources, key=lambda item: item.source_dataset_id):
        fingerprint = source_dataset_base_fingerprint(source.dataset_root)
        source_rows.append(
            {
                "source_dataset_id": source.source_dataset_id,
                "source_root_locator": relative_locator(
                    root=organized_root,
                    target=source.dataset_root,
                ),
                "source_dataset_fingerprint": fingerprint,
            }
        )
        for item in list_progress_annotation_work_items(source.dataset_root):
            if included is not None and item.episode_identity_hash not in included:
                continue
            seen_input_identities.add(item.episode_identity_hash)
            prepared.append(
                prepare_episode(
                    source=source,
                    item=item,
                    dataset_fingerprint=fingerprint,
                    selector=selection.get(item.episode_identity_hash),
                    column_projection=projection,
                    modality_selection=modalities,
                )
            )
    if not prepared:
        raise PreparedTrainingViewError(
            "No sealed stage-profiled Episodes matched the selection."
        )
    if included is not None and seen_input_identities != included:
        raise PreparedTrainingViewError(
            "Requested Episode identities were not found: "
            f"{sorted(included - seen_input_identities)}."
        )
    unused = sorted(set(selection) - seen_input_identities)
    if unused:
        raise PreparedTrainingViewError(
            "annotation_selection contains Episodes outside the selected input: "
            f"{unused}."
        )

    by_identity: dict[str, PreparedEpisode] = {}
    for episode in sorted(
        prepared,
        key=lambda value: (
            value.item.episode_identity_hash,
            value.source.source_dataset_id,
            value.item.episode_index,
        ),
    ):
        previous = by_identity.get(episode.item.episode_identity_hash)
        if previous is None:
            by_identity[episode.item.episode_identity_hash] = episode
            continue
        previous_annotation_hash = (
            None
            if previous.annotation is None
            else previous.annotation.manifest.semantic_content_hash
        )
        annotation_hash = (
            None
            if episode.annotation is None
            else episode.annotation.manifest.semantic_content_hash
        )
        if (
            previous.source_episode_content_hash
            != episode.source_episode_content_hash
            or previous_annotation_hash != annotation_hash
            or previous.failure_trigger_hash != episode.failure_trigger_hash
        ):
            raise PreparedTrainingViewError(
                "The same Episode identity resolves to conflicting source or "
                "annotation content."
            )
    return (
        tuple(by_identity[key] for key in sorted(by_identity)),
        source_rows,
    )


def build_prepared_training_view(
    organized_dataset_root: str | Path,
    sources: Sequence[PreparedSourceDataset],
    *,
    annotation_selection: Mapping[str, str] | None = None,
    include_episode_identities: Iterable[str] | None = None,
    modality_selection: Iterable[str] = ("numeric", "video"),
    column_projection: Iterable[str] = ("timestamp",),
    resolver_policy_id: str = "dataset_interface.training_semantics.v1",
) -> PreparedTrainingViewBuildResult:
    """Build one immutable logical view from sealed date-based source roots."""

    organized_root = Path(organized_dataset_root).expanduser().resolve()
    normalized_sources = tuple(sources)
    if not normalized_sources or any(
        not isinstance(source, PreparedSourceDataset)
        for source in normalized_sources
    ):
        raise PreparedTrainingViewError(
            "sources must contain at least one PreparedSourceDataset."
        )
    source_ids = tuple(source.source_dataset_id for source in normalized_sources)
    if len(source_ids) != len(set(source_ids)):
        raise PreparedTrainingViewError("source_dataset_id values must be unique.")
    if any(not source.dataset_root.is_dir() for source in normalized_sources):
        raise PreparedTrainingViewError("Every source dataset root must exist.")
    modalities = normalized_modalities(modality_selection)
    projection = normalized_projection(column_projection)
    policy = str(resolver_policy_id).strip()
    if not policy:
        raise PreparedTrainingViewError("resolver_policy_id must be non-empty.")
    selection = {
        str(key).strip(): str(value).strip()
        for key, value in (annotation_selection or {}).items()
    }
    if any(not key or not value for key, value in selection.items()):
        raise PreparedTrainingViewError(
            "annotation_selection keys and values must be non-empty."
        )
    included = (
        None
        if include_episode_identities is None
        else frozenset(str(value).strip() for value in include_episode_identities)
    )
    if included is not None and (not included or "" in included):
        raise PreparedTrainingViewError(
            "include_episode_identities must contain non-empty identities."
        )
    episodes, source_rows = _resolve_inputs(
        organized_root=organized_root,
        sources=normalized_sources,
        selection=selection,
        included=included,
        projection=projection,
        modalities=modalities,
    )

    progress_hashes = {
        episode.item.progress.profile.semantic_content_hash for episode in episodes
    }
    stage_hashes = {
        episode.item.progress.stage_profile.semantic_content_hash
        for episode in episodes
        if episode.item.progress.stage_profile is not None
    }
    if len(progress_hashes) != 1 or len(stage_hashes) != 1:
        raise PreparedTrainingViewError(
            "An organized dataset requires exactly one progress profile and "
            "one stage profile."
        )
    projection_schemas = {episode.projection_schema for episode in episodes}
    if len(projection_schemas) != 1:
        raise PreparedTrainingViewError(
            "Projected Parquet field schemas differ across selected Episodes."
        )

    selected_rows: list[dict[str, Any]] = []
    inventory_rows: list[Mapping[str, Any]] = []
    for ordinal, episode in enumerate(episodes):
        annotation = episode.annotation
        selected_rows.append(
            {
                "organized_episode_ordinal": ordinal,
                "source_dataset_id": episode.source.source_dataset_id,
                "source_episode_index": episode.item.episode_index,
                "episode_capture_identity_hash": episode.item.episode_identity_hash,
                "source_episode_record_hash": episode.source_episode_record_hash,
                "source_episode_content_hash": episode.source_episode_content_hash,
                "annotation_set_id": (
                    None if annotation is None else annotation.binding.annotation_set_id
                ),
                "annotation_manifest_hash": (
                    None
                    if annotation is None
                    else annotation.manifest.semantic_content_hash
                ),
                "failure_trigger_hash": episode.failure_trigger_hash,
            }
        )
        inventory_rows.extend(episode.inventory)
    inventory_rows = sorted(
        {
            (
                str(row["source_dataset_id"]),
                str(row["relative_path"]),
                str(row["role"]),
            ): row
            for row in inventory_rows
        }.values(),
        key=lambda row: (
            str(row["source_dataset_id"]),
            str(row["relative_path"]),
            str(row["role"]),
        ),
    )
    episode_rows, interval_rows, anchor_rows = _table_rows(
        episodes,
        modality_selection=modalities,
    )

    view_parent = organized_root / VIEW_RELATIVE_ROOT
    view_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=str(view_parent)))
    try:
        write_bytes(
            staging / SOURCE_DATASETS_FILE,
            canonical_bytes(
                {
                    "schema": PREPARED_TRAINING_VIEW_SCHEMA_V1,
                    "sources": source_rows,
                },
                domain=PREPARED_SOURCE_DATASETS_DOMAIN_V1,
            ),
        )
        write_cjsonl(
            staging / SELECTED_EPISODES_FILE,
            selected_rows,
            domain=PREPARED_SELECTED_EPISODE_DOMAIN_V1,
        )
        write_cjsonl(
            staging / SOURCE_INVENTORY_FILE,
            inventory_rows,
            domain=PREPARED_SOURCE_INVENTORY_DOMAIN_V1,
        )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise PreparedTrainingViewError(
                "pyarrow is required to build a prepared training view."
            ) from exc
        episode_schema, interval_schema, anchor_schema = arrow_schemas()
        tables = (
            (EPISODES_FILE, pa.Table.from_pylist(episode_rows, schema=episode_schema)),
            (
                INTERVALS_FILE,
                pa.Table.from_pylist(interval_rows, schema=interval_schema),
            ),
            (ANCHORS_FILE, pa.Table.from_pylist(anchor_rows, schema=anchor_schema)),
        )
        for filename, table in tables:
            pq.write_table(table, staging / filename, compression="NONE")
            with (staging / filename).open("rb") as handle:
                os.fsync(handle.fileno())

        derived_files = {
            filename: {
                "sha256": sha256_file(staging / filename),
                "size_bytes": (staging / filename).stat().st_size,
            }
            for filename in DERIVED_FILES
        }
        derived_files[EPISODES_FILE]["row_count"] = len(episode_rows)
        derived_files[INTERVALS_FILE]["row_count"] = len(interval_rows)
        derived_files[ANCHORS_FILE]["row_count"] = len(anchor_rows)
        manifest_payload: dict[str, Any] = {
            "schema": PREPARED_TRAINING_VIEW_SCHEMA_V1,
            "source_dataset_ids": [row["source_dataset_id"] for row in source_rows],
            "progress_profile_hash": next(iter(progress_hashes)),
            "stage_profile_hash": next(iter(stage_hashes)),
            "resolver_policy_id": policy,
            "modality_selection": list(modalities),
            "column_projection": list(projection),
            "projection_schema": [
                {"name": name, "type": kind, "nullable": nullable}
                for name, kind, nullable in next(iter(projection_schemas))
            ],
            "episode_count": len(episode_rows),
            "interval_count": len(interval_rows),
            "anchor_count": len(anchor_rows),
            "derived_files": derived_files,
            "implementation_id": "dataset_interface.prepared_training_view.v1",
        }
        training_view_hash = canonical_content_hash(
            manifest_payload,
            domain=PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1,
        )
        manifest_payload["training_view_hash"] = training_view_hash
        write_bytes(
            staging / MANIFEST_FILE,
            canonical_bytes(
                manifest_payload,
                domain=PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1,
            ),
        )
        fsync_directory(staging)
        final = view_parent / training_view_hash
        if final.exists():
            loaded = load_prepared_training_view(final)
            return PreparedTrainingViewBuildResult(
                view_root=loaded.view_root,
                training_view_hash=loaded.training_view_hash,
                episode_count=loaded.episodes.num_rows,
                interval_count=loaded.semantic_intervals.num_rows,
                anchor_count=loaded.semantic_anchors.num_rows,
                idempotent=True,
            )
        os.rename(staging, final)
        fsync_directory(view_parent)
        staging = final
        loaded = load_prepared_training_view(final)
        return PreparedTrainingViewBuildResult(
            view_root=loaded.view_root,
            training_view_hash=loaded.training_view_hash,
            episode_count=loaded.episodes.num_rows,
            interval_count=loaded.semantic_intervals.num_rows,
            anchor_count=loaded.semantic_anchors.num_rows,
            idempotent=False,
        )
    finally:
        if staging.exists() and staging.name.startswith(".building-"):
            shutil.rmtree(staging)


__all__ = ["build_prepared_training_view"]
