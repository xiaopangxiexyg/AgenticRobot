"""Preflight and optionally publish writing-stage1 pilot semantic layers.

Without a publication option this command is strictly read-only.  The formal
path writes immutable semantic artifacts below each source dataset's own
``semantic_layers`` directory; raw capture artifacts remain read-only.  The
external workspace option remains available only for compatibility testing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collection_runtime.progress.profiles import load_managed_task_profile
from collection_runtime.progress.retrospective_store import (
    commit_retrospective_progress_episode,
    prepare_retrospective_campaign_definition,
    publish_retrospective_campaign_definition,
    resolve_retrospective_migration_identity,
    seal_retrospective_campaign_publication,
)
from dataset_contracts.progress_profile import ProgressEpisodeBindingV1
from dataset_contracts.retrospective_campaign import (
    RetrospectiveCampaignEpisodeSealV1,
)
from dataset_contracts.retrospective_progress import (
    RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
    RetrospectiveVideoBindingV1,
)
from dataset_interface.episode_curation import (
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)
from dataset_interface.retrospective_progress import retrospective_sample_axis_hash
from dataset_interface.sample_axis import EpisodeSampleAxis
from dataset_interface.writing_stage1_cohort import (
    WritingStage1EpisodeRef,
    build_writing_stage1_cohort_plan,
    select_writing_stage1_pilot,
)
from dataset_interface.writing_stage1_v23 import (
    WRITING_STAGE1_V23_DEFAULT_LAYER_ID,
    resolve_writing_stage1_v23_campaign_batches,
    writing_stage1_v23_semantic_layer_root,
)


_PROFILE_ID = "dmtac_bimanual_writing_acquire_present_window_v1"
_CAMERA_KEY_BY_SLOT = {
    "base_0_rgb": "front",
    "left_wrist_0_rgb": "left_wrist",
    "right_wrist_0_rgb": "right_wrist",
}
_MODALITY_ID_BY_CAMERA = {
    "front": "front_rgb",
    "left_wrist": "left_wrist_rgb",
    "right_wrist": "right_wrist_rgb",
}


@dataclass(frozen=True)
class _PreparedPilotEpisode:
    source_dataset_root: Path
    report: dict[str, Any]
    axis: EpisodeSampleAxis
    progress_binding: ProgressEpisodeBindingV1
    source_binding: RetrospectiveSourceBindingV1
    window: RetrospectiveProgressWindowV1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Source artifact escapes the bound dataset root: {relative}"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"Source artifact is not a regular file: {candidate}")
    return candidate


def _episode_rows(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        index = int(value["episode_index"])
        if index in rows:
            raise ValueError(f"{path}:{line_number} duplicates Episode {index}")
        rows[index] = value
    return rows


def _source_schema(info: dict[str, Any]) -> str:
    candidates = (
        info.get("dexteleop_dataset_schema"),
        info.get("codebase_version"),
        (info.get("metadata") or {}).get("dexteleop_dataset_schema")
        if isinstance(info.get("metadata"), dict)
        else None,
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Dataset info does not declare its source schema")


def _preflight_episode(
    episode: WritingStage1EpisodeRef,
    *,
    rows_by_root: dict[Path, dict[int, dict[str, Any]]],
    dataset_local_source: bool,
) -> _PreparedPilotEpisode:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "pyarrow is required to preflight writing-stage1 Parquet files."
        ) from exc

    root = Path(episode.source_dataset_root).resolve(strict=True)
    if root not in rows_by_root:
        rows_by_root[root] = _episode_rows(root)
    rows = rows_by_root[root]
    try:
        record = rows[episode.source_episode_index]
    except KeyError as exc:
        raise ValueError(
            f"Missing source Episode {episode.source_episode_index} under {root}"
        ) from exc
    if int(record.get("frame_count", -1)) != episode.frame_count:
        raise ValueError("Logical/source Episode frame_count mismatch")
    info_path = _inside(root, "meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError(f"Dataset info must be an object: {info_path}")

    parquet_relative = str(record["parquet_path"])
    parquet_path = _inside(root, parquet_relative)
    timestamps = tuple(
        float(value.as_py())
        for value in pq.read_table(parquet_path, columns=["timestamp"])["timestamp"]
    )
    if len(timestamps) != episode.frame_count:
        raise ValueError("Parquet timestamp count differs from logical frame_count")
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=timestamps,
    )

    raw_video_paths = record.get("video_paths")
    if not isinstance(raw_video_paths, dict):
        raise ValueError("Episode record does not contain video_paths")
    video_bindings: list[RetrospectiveVideoBindingV1] = []
    artifact_stats: list[dict[str, Any]] = []
    for slot in episode.admitted_camera_slots:
        camera = _CAMERA_KEY_BY_SLOT[slot]
        relative = str(raw_video_paths[camera])
        path = _inside(root, relative)
        stat = path.stat()
        digest = _sha256(path)
        video_bindings.append(
            RetrospectiveVideoBindingV1(
                modality_id=_MODALITY_ID_BY_CAMERA[camera],
                source_relative_path=relative,
                content_hash=digest,
                byte_count=stat.st_size,
                declared_frame_count=episode.frame_count,
                decoded_frame_count=episode.frame_count,
            )
        )
        artifact_stats.append(
            {
                "kind": f"video:{camera}",
                "relative_path": relative,
                "sha256": digest,
                "size": stat.st_size,
                "inode": stat.st_ino,
                "link_count": stat.st_nlink,
            }
        )

    parquet_stat = parquet_path.stat()
    parquet_hash = _sha256(parquet_path)
    artifact_stats.insert(
        0,
        {
            "kind": "parquet",
            "relative_path": parquet_relative,
            "sha256": parquet_hash,
            "size": parquet_stat.st_size,
            "inode": parquet_stat.st_ino,
            "link_count": parquet_stat.st_nlink,
        },
    )
    record_hash = source_episode_manifest_hash(
        root,
        episode.source_episode_index,
    )
    if record_hash is None:
        raise ValueError("Source Episode record hash is unavailable")
    source_binding = RetrospectiveSourceBindingV1(
        episode_capture_identity_hash=episode.episode_identity_hash,
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(root),
        source_dataset_locator=(
            RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1
            if dataset_local_source
            else str(root)
        ),
        source_schema=_source_schema(info),
        source_batch_id=episode.batch_id,
        source_episode_index=episode.source_episode_index,
        provenance_source_episode_index=episode.provenance_episode_index,
        source_info_hash=_sha256(info_path),
        source_episode_record_hash=record_hash,
        source_parquet_relative_path=parquet_relative,
        source_parquet_hash=parquet_hash,
        source_parquet_byte_count=parquet_stat.st_size,
        admitted_videos=tuple(video_bindings),
    )

    bundle = load_managed_task_profile(_PROFILE_ID)
    assert bundle.stage_profile is not None
    progress_binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=episode.episode_identity_hash,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        bound_clock_domain_id=axis.clock_domain_id,
        recording_scope="PROGRESS_STEP_WINDOW",
        entry_step_id="right_acquire_brush",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
        truncated_at_start=False,
    )
    window = RetrospectiveProgressWindowV1(
        episode_capture_identity_hash=episode.episode_identity_hash,
        source_binding_hash=source_binding.semantic_content_hash,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        progress_episode_binding_hash=progress_binding.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        sample_count=axis.sample_count,
        clock_domain_id=axis.clock_domain_id,
        progress_step_id="right_acquire_brush",
        attempt_id=f"retrospective:writing-stage1:{episode.episode_identity_hash[:16]}",
        start_at=axis.cut_times[0],
        end_at=axis.cut_times[-1],
    )
    return _PreparedPilotEpisode(
        source_dataset_root=root,
        report={
            "episode": episode.to_payload(),
            "source_binding": source_binding.to_payload(),
            "progress_episode_binding": progress_binding.to_payload(),
            "retrospective_window": window.to_payload(),
            "artifacts": artifact_stats,
        },
        axis=axis,
        progress_binding=progress_binding,
        source_binding=source_binding,
        window=window,
    )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach writing-stage1 pilot arguments to an existing parser."""

    parser.add_argument("--front-manifest", required=True)
    parser.add_argument("--three-view-manifest", required=True)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--three-view-provenance-root")
    provenance.add_argument("--three-view-provenance-marker")
    parser.add_argument("--per-batch", type=int, default=6)
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument(
        "--workspace-root",
        help=(
            "optional new/existing semantic workspace to publish after all "
            "pilot source checks pass"
        ),
    )
    publication.add_argument(
        "--publish-dataset-local",
        action="store_true",
        help=(
            "publish each 6-Episode pilot into the owning v2.3 dataset's "
            "semantic layer"
        ),
    )
    parser.add_argument(
        "--semantic-layer-id",
        default=WRITING_STAGE1_V23_DEFAULT_LAYER_ID,
        help="dataset-local semantic layer id",
    )
    parser.add_argument(
        "--campaign-id",
        default="pilot_v1",
        help="immutable campaign id within each dataset-local layer",
    )
    parser.add_argument(
        "--migration-actor-id",
        default="",
        help=(
            "stable actor id required for the first publication; later "
            "reruns recover it from the campaign or migration seals"
        ),
    )
    parser.add_argument(
        "--migration-created-at",
        type=float,
        help=(
            "optional fixed annotation-clock timestamp; omitted values are "
            "chosen once and recovered from existing migration seals"
        ),
    )
    parser.set_defaults(func=run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return parser


def _provenance_root(args: argparse.Namespace) -> str:
    if args.three_view_provenance_root:
        return str(args.three_view_provenance_root)
    marker = str(args.three_view_provenance_marker)
    roots: set[str] = set()
    for line in Path(args.three_view_manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        provenance = row.get("source_provenance")
        if not isinstance(provenance, dict):
            continue
        root = provenance.get("source_dataset_root")
        if isinstance(root, str) and marker in root:
            roots.add(root)
    if len(roots) != 1:
        raise ValueError(
            "Provenance marker must resolve exactly one source root; "
            f"marker={marker!r} roots={sorted(roots)!r}"
        )
    return next(iter(roots))


def _migration_manifest(
    value: _PreparedPilotEpisode,
    *,
    actor_id: str,
    created_at: float,
) -> RetrospectiveProgressMigrationManifestV1:
    bundle = load_managed_task_profile(_PROFILE_ID)
    assert bundle.stage_profile is not None
    return RetrospectiveProgressMigrationManifestV1(
        episode_capture_identity_hash=(
            value.source_binding.episode_capture_identity_hash
        ),
        source_binding_hash=value.source_binding.semantic_content_hash,
        retrospective_window_hash=value.window.semantic_content_hash,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        progress_episode_binding_hash=(
            value.progress_binding.semantic_content_hash
        ),
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(value.axis),
        migration_actor_id=actor_id,
        migration_created_at=created_at,
        migration_clock_domain_id="annotation_wall_clock",
    )


def run(args: argparse.Namespace) -> int:
    plan = build_writing_stage1_cohort_plan(
        front_manifest_path=args.front_manifest,
        three_view_manifest_path=args.three_view_manifest,
        three_view_required_provenance_root=_provenance_root(args),
    )
    pilot = select_writing_stage1_pilot(plan, per_batch=args.per_batch)
    dataset_local_batches = (
        resolve_writing_stage1_v23_campaign_batches(
            plan,
            pilot,
            expected_per_batch=args.per_batch,
        )
        if args.publish_dataset_local
        else ()
    )
    dataset_local_layer_roots = {
        batch.batch_id: writing_stage1_v23_semantic_layer_root(
            batch.source_dataset_root,
            semantic_layer_id=args.semantic_layer_id,
            require_exists=False,
        )
        for batch in dataset_local_batches
    }
    rows_by_root: dict[Path, dict[int, dict[str, Any]]] = {}
    prepared = tuple(
        _preflight_episode(
            item,
            rows_by_root=rows_by_root,
            dataset_local_source=bool(args.publish_dataset_local),
        )
        for item in pilot
    )
    workspace_root: Path | None = None
    publish_reports: list[dict[str, Any]] = []
    semantic_layer_reports: list[dict[str, Any]] = []
    migration_identity_recovered = False
    migration_actor_id: str | None = None
    migration_created_at: float | None = None
    if args.workspace_root:
        workspace_root = Path(args.workspace_root).expanduser().resolve()
        source_roots = {
            value.source_dataset_root
            for value in prepared
        }
        if any(
            workspace_root == source_root
            or source_root in workspace_root.parents
            for source_root in source_roots
        ):
            raise ValueError(
                "Semantic workspace must be outside every read-only source "
                "dataset root."
            )
        identity = resolve_retrospective_migration_identity(
            workspace_root,
            requested_actor_id=args.migration_actor_id,
            requested_created_at=args.migration_created_at,
        )
        actor_id = identity.actor_id
        created_at = identity.created_at
        migration_actor_id = actor_id
        migration_created_at = created_at
        migration_identity_recovered = identity.recovered
        bundle = load_managed_task_profile(_PROFILE_ID)
        assert bundle.stage_profile is not None
        for value in prepared:
            manifest = _migration_manifest(
                value,
                actor_id=actor_id,
                created_at=created_at,
            )
            commit = commit_retrospective_progress_episode(
                workspace_root=workspace_root,
                progress_profile=bundle.progress_profile,
                progress_binding=value.progress_binding,
                stage_profile=bundle.stage_profile,
                sample_axis=value.axis,
                source_binding=value.source_binding,
                window=value.window,
                migration_manifest=manifest,
            )
            publish_reports.append(
                {
                    "episode_identity_hash": (
                        value.source_binding.episode_capture_identity_hash
                    ),
                    "migration_manifest_hash": (
                        manifest.semantic_content_hash
                    ),
                    "episode_dir": str(commit.episode_dir),
                    "idempotent": commit.idempotent,
                }
            )
    elif args.publish_dataset_local:
        prepared_by_identity: dict[str, _PreparedPilotEpisode] = {}
        for value in prepared:
            episode_identity = (
                value.source_binding.episode_capture_identity_hash
            )
            if episode_identity in prepared_by_identity:
                raise ValueError("Prepared pilot contains duplicate identities.")
            prepared_by_identity[episode_identity] = value

        bundle = load_managed_task_profile(_PROFILE_ID)
        assert bundle.stage_profile is not None
        planned_campaigns = []
        for batch in dataset_local_batches:
            layer_root = dataset_local_layer_roots[batch.batch_id]
            selected_values = tuple(
                prepared_by_identity[value.episode_identity_hash]
                for value in batch.selected
            )
            planned = prepare_retrospective_campaign_definition(
                semantic_layer_root=layer_root,
                semantic_layer_id=args.semantic_layer_id,
                campaign_id=args.campaign_id,
                campaign_kind="PILOT_QA",
                source_dataset_base_fingerprint=(
                    source_dataset_base_fingerprint(batch.source_dataset_root)
                ),
                cohort_plan_hash=plan.semantic_content_hash,
                eligible_episode_identity_hashes=tuple(
                    item.episode_identity_hash for item in batch.eligible
                ),
                selected_episode_identity_hashes=tuple(
                    value.source_binding.episode_capture_identity_hash
                    for value in selected_values
                ),
                requested_actor_id=args.migration_actor_id,
                requested_created_at=args.migration_created_at,
            )
            planned_campaigns.append((batch, layer_root, selected_values, planned))

        campaign_identities: list[tuple[str, float, bool]] = []
        for batch, layer_root, selected_values, planned in planned_campaigns:
            campaign_report = publish_retrospective_campaign_definition(
                semantic_layer_root=layer_root,
                semantic_layer_id=args.semantic_layer_id,
                campaign_id=args.campaign_id,
                campaign_kind="PILOT_QA",
                source_dataset_base_fingerprint=(
                    source_dataset_base_fingerprint(batch.source_dataset_root)
                ),
                cohort_plan_hash=plan.semantic_content_hash,
                eligible_episode_identity_hashes=tuple(
                    item.episode_identity_hash for item in batch.eligible
                ),
                selected_episode_identity_hashes=tuple(
                    value.source_binding.episode_capture_identity_hash
                    for value in selected_values
                ),
                requested_actor_id=planned.campaign.migration_actor_id,
                requested_created_at=planned.campaign.migration_created_at,
            )
            actor_id = campaign_report.campaign.migration_actor_id
            created_at = campaign_report.campaign.migration_created_at
            campaign_identities.append(
                (actor_id, created_at, campaign_report.identity_recovered)
            )
            episode_reports: list[dict[str, Any]] = []
            seals: list[RetrospectiveCampaignEpisodeSealV1] = []
            for value in selected_values:
                manifest = _migration_manifest(
                    value,
                    actor_id=actor_id,
                    created_at=created_at,
                )
                commit = commit_retrospective_progress_episode(
                    workspace_root=layer_root,
                    progress_profile=bundle.progress_profile,
                    progress_binding=value.progress_binding,
                    stage_profile=bundle.stage_profile,
                    sample_axis=value.axis,
                    source_binding=value.source_binding,
                    window=value.window,
                    migration_manifest=manifest,
                )
                seals.append(
                    RetrospectiveCampaignEpisodeSealV1(
                        episode_capture_identity_hash=(
                            value.source_binding.episode_capture_identity_hash
                        ),
                        migration_manifest_hash=manifest.semantic_content_hash,
                    )
                )
                episode_reports.append(
                    {
                        "episode_identity_hash": (
                            value.source_binding.episode_capture_identity_hash
                        ),
                        "migration_manifest_hash": (
                            manifest.semantic_content_hash
                        ),
                        "episode_dir": str(commit.episode_dir),
                        "idempotent": commit.idempotent,
                    }
                )
            receipt_report = seal_retrospective_campaign_publication(
                semantic_layer_root=layer_root,
                campaign=campaign_report.campaign,
                episode_seals=tuple(seals),
            )
            semantic_layer_reports.append(
                {
                    "source_dataset_root": str(batch.source_dataset_root),
                    "source_batch_id": batch.batch_id,
                    "semantic_layer_root": str(layer_root),
                    "semantic_layer_id": args.semantic_layer_id,
                    "campaign_id": args.campaign_id,
                    "campaign_definition_hash": (
                        campaign_report.campaign.semantic_content_hash
                    ),
                    "campaign_definition_idempotent": (
                        campaign_report.idempotent
                    ),
                    "eligible_episode_count": len(batch.eligible),
                    "selected_episode_count": len(selected_values),
                    "migration_actor_id": actor_id,
                    "migration_created_at": created_at,
                    "migration_identity_recovered": (
                        campaign_report.identity_recovered
                    ),
                    "publication_receipt": str(receipt_report.receipt_path),
                    "publication_receipt_hash": (
                        receipt_report.receipt.semantic_content_hash
                    ),
                    "publication_receipt_idempotent": (
                        receipt_report.idempotent
                    ),
                    "published_episodes": tuple(episode_reports),
                }
            )
        distinct_identities = {
            (actor, created_at) for actor, created_at, _ in campaign_identities
        }
        if len(distinct_identities) == 1:
            migration_actor_id, migration_created_at = next(
                iter(distinct_identities)
            )
        migration_identity_recovered = all(
            recovered for _, _, recovered in campaign_identities
        )
    result = {
        "schema": "dexteleop.writing_stage1_pilot_preflight.v1",
        "read_only": workspace_root is None and not args.publish_dataset_local,
        "cohort_hash": plan.semantic_content_hash,
        "cohort_counts": plan.counts,
        "pilot_count": len(pilot),
        "workspace_root": (
            None if workspace_root is None else str(workspace_root)
        ),
        "migration_actor_id": migration_actor_id,
        "migration_created_at": migration_created_at,
        "migration_identity_recovered": migration_identity_recovered,
        "published_episodes": tuple(publish_reports),
        "dataset_local_semantic_layers": tuple(semantic_layer_reports),
        "episodes": tuple(value.report for value in prepared),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
