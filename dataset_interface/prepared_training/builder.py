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
from ..metadata import DexTeleopDatasetIndex
from ..progress_annotation_jobs import list_progress_annotation_work_items
from ..progress_skill_projection import (
    SkillProjectionCatalog,
    project_episode_skill_view,
)
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
    PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1,
    PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    PREPARED_SOURCE_INVENTORY_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1,
    PREPARED_TRAINING_VIEW_SCHEMA_V1,
    SELECTED_EPISODES_FILE,
    SOURCE_DATASETS_FILE,
    SOURCE_INVENTORY_FILE,
    SKILL_INTERVALS_FILE,
    SKILL_PROJECTION_RECEIPTS_FILE,
    VIEW_RELATIVE_ROOT,
    PreparedSourceDataset,
    PreparedTrainingViewBuildResult,
    PreparedTrainingViewError,
    arrow_schemas,
    skill_interval_arrow_schema,
)
from .reader import load_prepared_training_view
from .sources import PreparedEpisode, prepare_episode


def _table_rows(
    episodes: Sequence[PreparedEpisode],
    *,
    modality_selection: tuple[str, ...],
    camera_selection: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for ordinal, episode in enumerate(episodes):
        item = episode.item
        annotation = episode.annotation
        camera_keys = (
            list(
                sorted(item.video_paths)
                if camera_selection is None
                else camera_selection
            )
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
    camera_selection: tuple[str, ...] | None,
) -> tuple[tuple[PreparedEpisode, ...], list[dict[str, Any]]]:
    prepared: list[PreparedEpisode] = []
    source_rows: list[dict[str, Any]] = []
    seen_input_identities: set[str] = set()
    for source in sorted(sources, key=lambda item: item.source_dataset_id):
        fingerprint = source_dataset_base_fingerprint(source.dataset_root)
        dataset_index = DexTeleopDatasetIndex(
            source.dataset_root,
            parse_capture_contract=False,
        )
        excluded_episode_indices: list[int] = []
        for item in list_progress_annotation_work_items(source.dataset_root):
            requested_cameras = (
                ()
                if "video" not in modalities
                else (
                    tuple(sorted(item.video_paths))
                    if camera_selection is None
                    else camera_selection
                )
            )
            eligibility = dataset_index.training_eligibility(
                item.episode_index,
                camera_keys=requested_cameras,
            )
            if not eligibility.eligible:
                excluded_episode_indices.append(item.episode_index)
                continue
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
                    camera_selection=camera_selection,
                )
            )
        source_rows.append(
            {
                "source_dataset_id": source.source_dataset_id,
                "source_root_locator": relative_locator(
                    root=organized_root,
                    target=source.dataset_root,
                ),
                "source_dataset_fingerprint": fingerprint,
                "training_validity_excluded_episode_indices": sorted(
                    set(excluded_episode_indices)
                ),
            }
        )
    if included is not None and seen_input_identities != included:
        raise PreparedTrainingViewError(
            "Requested Episode identities were not found or are excluded by "
            "the source training-validity policy: "
            f"{sorted(included - seen_input_identities)}."
        )
    if not prepared:
        raise PreparedTrainingViewError(
            "No sealed stage-profiled Episodes matched the selection."
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


def _skill_projection_rows(
    episodes: Sequence[PreparedEpisode],
    *,
    catalog: SkillProjectionCatalog,
    target_skill_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    skill = catalog.skill_by_id.get(target_skill_id)
    if skill is None:
        raise PreparedTrainingViewError(
            f"Unknown target Skill {target_skill_id!r}."
        )
    candidate_projections = catalog.projections_for_skill(target_skill_id)
    interval_rows: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    used_projections: dict[str, Any] = {}
    for ordinal, episode in enumerate(episodes):
        stage_profile = episode.item.progress.stage_profile
        if stage_profile is None:
            raise PreparedTrainingViewError(
                "Skill projection requires an ordered stage profile."
            )
        matches = tuple(
            projection
            for projection in candidate_projections
            if (
                projection.source_progress_profile_hash
                == episode.item.progress.profile.semantic_content_hash
                and projection.source_stage_profile_hash
                == stage_profile.semantic_content_hash
            )
        )
        if len(matches) != 1:
            raise PreparedTrainingViewError(
                "Every selected Episode must resolve exactly one projection "
                f"for Skill {target_skill_id!r}; episode="
                f"{episode.item.episode_identity_hash} matches={len(matches)}."
            )
        projection = matches[0]
        annotation_manifest_hash = (
            None
            if episode.annotation is None
            else episode.annotation.manifest.semantic_content_hash
        )
        view = project_episode_skill_view(
            progress_profile=episode.item.progress.profile,
            stage_profile=stage_profile,
            skill_contract=skill,
            projection=projection,
            semantic_index=episode.semantic_index,
            source_stage_annotation_manifest_hash=annotation_manifest_hash,
        )
        used_projections[projection.semantic_content_hash] = projection
        receipt_rows.append(
            {
                "organized_episode_ordinal": ordinal,
                "receipt": view.receipt.to_payload(),
            }
        )
        for interval in view.receipt.intervals:
            interval_rows.append(
                {
                    "organized_episode_ordinal": ordinal,
                    "episode_capture_identity_hash": (
                        episode.item.episode_identity_hash
                    ),
                    "skill_id": skill.skill_id,
                    "skill_training_contract_hash": (
                        skill.semantic_content_hash
                    ),
                    "progress_skill_projection_hash": (
                        projection.semantic_content_hash
                    ),
                    "sample_begin": interval.sample_begin,
                    "sample_end_exclusive": interval.sample_end,
                    "training_domain": interval.training_domain,
                    "source_attempt_id": interval.source_attempt_id,
                    "source_stage_ids": list(interval.source_stage_ids),
                    "source_occurrence_ids": list(
                        interval.source_occurrence_ids
                    ),
                    "source_failure_fact_id": (
                        interval.source_failure_fact_id
                    ),
                }
            )
    if not interval_rows:
        raise PreparedTrainingViewError(
            "The selected Skill has no annotated eligible intervals."
        )
    capability = {
        "schema": "dexteleop.prepared_skill_projection.provisional.v1",
        "skill_contract": skill.to_payload(),
        "projection_hashes": sorted(used_projections),
        "projections": [
            used_projections[key].to_payload()
            for key in sorted(used_projections)
        ],
        "interval_count": len(interval_rows),
        "receipt_count": len(receipt_rows),
        "implementation_id": "dataset_interface.progress_skill_projection.v1",
    }
    return interval_rows, receipt_rows, capability


def build_prepared_training_view(
    organized_dataset_root: str | Path,
    sources: Sequence[PreparedSourceDataset],
    *,
    annotation_selection: Mapping[str, str] | None = None,
    include_episode_identities: Iterable[str] | None = None,
    modality_selection: Iterable[str] = ("numeric", "video"),
    camera_selection: Iterable[str] | None = None,
    column_projection: Iterable[str] = ("timestamp",),
    resolver_policy_id: str = "dataset_interface.training_semantics.v1",
    skill_projection_catalog: SkillProjectionCatalog | None = None,
    target_skill_id: str | None = None,
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
    cameras = (
        None
        if camera_selection is None
        else tuple(str(value).strip() for value in camera_selection)
    )
    if cameras is not None:
        if not cameras or "" in cameras or len(cameras) != len(set(cameras)):
            raise PreparedTrainingViewError(
                "camera_selection must contain unique non-empty camera keys."
            )
        if "video" not in modalities:
            raise PreparedTrainingViewError(
                "camera_selection requires the video modality."
            )
    projection = normalized_projection(column_projection)
    policy = str(resolver_policy_id).strip()
    if not policy:
        raise PreparedTrainingViewError("resolver_policy_id must be non-empty.")
    normalized_target_skill_id = (
        None if target_skill_id is None else str(target_skill_id).strip()
    )
    if (skill_projection_catalog is None) != (
        normalized_target_skill_id is None
    ):
        raise PreparedTrainingViewError(
            "skill_projection_catalog and target_skill_id must be provided "
            "together."
        )
    if skill_projection_catalog is not None and not isinstance(
        skill_projection_catalog,
        SkillProjectionCatalog,
    ):
        raise PreparedTrainingViewError(
            "skill_projection_catalog must be SkillProjectionCatalog."
        )
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
        camera_selection=cameras,
    )

    progress_hashes = {
        episode.item.progress.profile.semantic_content_hash for episode in episodes
    }
    stage_hashes = {
        episode.item.progress.stage_profile.semantic_content_hash
        for episode in episodes
        if episode.item.progress.stage_profile is not None
    }
    if (
        skill_projection_catalog is None
        and (len(progress_hashes) != 1 or len(stage_hashes) != 1)
    ):
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
        camera_selection=cameras,
    )
    skill_interval_rows: list[dict[str, Any]] = []
    skill_receipt_rows: list[dict[str, Any]] = []
    skill_capability: dict[str, Any] | None = None
    if skill_projection_catalog is not None:
        assert normalized_target_skill_id is not None
        (
            skill_interval_rows,
            skill_receipt_rows,
            skill_capability,
        ) = _skill_projection_rows(
            episodes,
            catalog=skill_projection_catalog,
            target_skill_id=normalized_target_skill_id,
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
        if skill_capability is not None:
            write_cjsonl(
                staging / SKILL_PROJECTION_RECEIPTS_FILE,
                skill_receipt_rows,
                domain=PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1,
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

        if skill_capability is not None:
            skill_table = pa.Table.from_pylist(
                skill_interval_rows,
                schema=skill_interval_arrow_schema(),
            )
            pq.write_table(
                skill_table,
                staging / SKILL_INTERVALS_FILE,
                compression="NONE",
            )
            with (staging / SKILL_INTERVALS_FILE).open("rb") as handle:
                os.fsync(handle.fileno())

        derived_filenames = list(DERIVED_FILES)
        if skill_capability is not None:
            derived_filenames.extend(
                (SKILL_INTERVALS_FILE, SKILL_PROJECTION_RECEIPTS_FILE)
            )
        derived_files = {
            filename: {
                "sha256": sha256_file(staging / filename),
                "size_bytes": (staging / filename).stat().st_size,
            }
            for filename in derived_filenames
        }
        derived_files[EPISODES_FILE]["row_count"] = len(episode_rows)
        derived_files[INTERVALS_FILE]["row_count"] = len(interval_rows)
        derived_files[ANCHORS_FILE]["row_count"] = len(anchor_rows)
        if skill_capability is not None:
            derived_files[SKILL_INTERVALS_FILE]["row_count"] = len(
                skill_interval_rows
            )
            derived_files[SKILL_PROJECTION_RECEIPTS_FILE]["row_count"] = len(
                skill_receipt_rows
            )
        manifest_payload: dict[str, Any] = {
            "schema": PREPARED_TRAINING_VIEW_SCHEMA_V1,
            "source_dataset_ids": [row["source_dataset_id"] for row in source_rows],
            "progress_profile_hash": (
                next(iter(progress_hashes))
                if len(progress_hashes) == 1
                else None
            ),
            "stage_profile_hash": (
                next(iter(stage_hashes)) if len(stage_hashes) == 1 else None
            ),
            "resolver_policy_id": policy,
            "modality_selection": list(modalities),
            "camera_selection": (
                None if cameras is None else list(cameras)
            ),
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
        if skill_capability is not None:
            manifest_payload["skill_projection"] = skill_capability
            manifest_payload["progress_profile_hashes"] = sorted(
                progress_hashes
            )
            manifest_payload["stage_profile_hashes"] = sorted(stage_hashes)
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
                skill_interval_count=(
                    0
                    if loaded.skill_intervals is None
                    else loaded.skill_intervals.num_rows
                ),
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
            skill_interval_count=(
                0
                if loaded.skill_intervals is None
                else loaded.skill_intervals.num_rows
            ),
        )
    finally:
        if staging.exists() and staging.name.startswith(".building-"):
            shutil.rmtree(staging)


__all__ = ["build_prepared_training_view"]
