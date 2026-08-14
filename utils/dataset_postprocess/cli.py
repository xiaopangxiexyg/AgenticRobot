from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from collection_runtime.tactile_postprocess import (
    TACTILE_MAINTENANCE_TARGETS,
    TACTILE_TARGET_COMPRESSED_HOT,
    apply_tactile_postprocess,
    plan_tactile_postprocess,
)
from collection_runtime.tactile_bundle_materialize import (
    materialize_tactile_bundle,
    plan_tactile_bundle_materialization,
)
from dataset_interface.tactile_bundle import attach_tactile_bundle
from dataset_interface.validation import validate_v23_dataset

from .causal_align import causal_align_dataset, plan_causal_align
from .camera_semantic_materialize import (
    materialize_camera_semantics,
    plan_camera_semantic_materialization,
)
from .camera_semantic_prepared_view import (
    materialize_camera_semantic_prepared_view,
    plan_camera_semantic_prepared_view,
)
from .delete_episodes import delete_episodes, plan_episode_deletion
from .fault_nan_tail import detect_fault_nan_tail
from .force_collision_labels import label_force_collisions, plan_force_collision_labels
from .hardlink_dedupe import apply_hardlink_dedupe, plan_hardlink_dedupe
from .io import episode_records, parquet_path_for_record, read_parquet_rows
from .modality_storage import (
    DEFAULT_MIN_FREE_BYTES as DEFAULT_MODALITY_STORAGE_MIN_FREE_BYTES,
    detach_modality,
    export_modality,
    modality_storage_status,
    move_modality,
    plan_modality_detach,
    plan_modality_export,
    plan_modality_rehydrate,
    rehydrate_modality,
)
from .split_episodes import plan_episode_split, split_episodes
from .stutter_segments import (
    clean_stutters,
    detect_stutter_segments,
    plan_stutter_clean,
)
from .tactile_compact import (
    DEFAULT_CHUNK_FRAMES as DEFAULT_TACTILE_COMPACT_CHUNK_FRAMES,
    DEFAULT_MIN_FREE_BYTES as DEFAULT_TACTILE_COMPACT_MIN_FREE_BYTES,
    compact_tactile_dataset_in_place,
    plan_tactile_compaction,
)
from .tactile_recompress import (
    DEFAULT_MIN_FREE_BYTES,
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)
from .umi_next_state_arm_action import migrate_umi_arm_action_to_next_observation
from .trim_nan_tails import plan_nan_tail_trim, trim_nan_tails
from .v21_to_v22 import (
    convert_v21_dataset_to_v22,
    load_v22_layout_json,
    plan_v21_to_v22_conversion,
)
from .v22_to_v23_desk import (
    convert_v22_dataset_to_v23_desk,
    plan_v22_to_v23_desk_conversion,
)
from .workspace_near_boundary import migrate_umi_workspace_near_boundary
from .writing_stage1_pilot import configure_parser as configure_writing_stage1_pilot


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _optional_path(value: str) -> str | None:
    text = str(value or "").strip()
    return None if not text else text


def _cmd_plan_stutters(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    segments = []
    for record in episode_records(root):
        rows = read_parquet_rows(parquet_path_for_record(root, record))
        episode_segments = detect_stutter_segments(
            rows,
            episode_index=int(record["episode_index"]),
            max_gap_sec=float(args.max_gap_sec),
            min_segment_frames=int(args.min_segment_frames),
        )
        if len(episode_segments) > 1:
            segments.extend(segment.to_dict() for segment in episode_segments)
    _print_json({"dataset_root": str(root), "segments": segments})
    return 0


def _cmd_clean_stutters(args: argparse.Namespace) -> int:
    if args.apply:
        if not args.output_root:
            raise ValueError("--output-root is required with --apply.")
        plan = clean_stutters(
            args.dataset_root,
            args.output_root,
            max_gap_sec=float(args.max_gap_sec),
            min_segment_frames=int(args.min_segment_frames),
            overwrite=bool(args.overwrite),
            reset_last_action=bool(args.reset_last_action),
        )
    else:
        plan = plan_stutter_clean(
            args.dataset_root,
            output_root=_optional_path(args.output_root),
            max_gap_sec=float(args.max_gap_sec),
            min_segment_frames=int(args.min_segment_frames),
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_plan_fault_nan(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    tails = []
    for record in episode_records(root):
        rows = read_parquet_rows(parquet_path_for_record(root, record))
        tail = detect_fault_nan_tail(
            rows,
            episode_index=int(record["episode_index"]),
            min_nan_fraction=float(args.min_nan_fraction),
        )
        if tail is not None:
            tails.append(tail.to_dict())
    _print_json({"dataset_root": str(root), "tails": tails})
    return 0


def _cmd_delete_episodes(args: argparse.Namespace) -> int:
    if args.apply:
        if not args.output_root:
            raise ValueError("--output-root is required with --apply.")
        plan = delete_episodes(
            args.dataset_root,
            args.output_root,
            args.episodes,
            overwrite=bool(args.overwrite),
        )
    else:
        plan = plan_episode_deletion(
            args.dataset_root,
            args.episodes,
            output_root=_optional_path(args.output_root),
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_trim_nan_tails(args: argparse.Namespace) -> int:
    if args.apply:
        if not args.output_root:
            raise ValueError("--output-root is required with --apply.")
        plan = trim_nan_tails(
            args.dataset_root,
            args.output_root,
            min_nan_fraction=float(args.min_nan_fraction),
            overwrite=bool(args.overwrite),
        )
    else:
        plan = plan_nan_tail_trim(
            args.dataset_root,
            output_root=_optional_path(args.output_root),
            min_nan_fraction=float(args.min_nan_fraction),
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_split_episodes(args: argparse.Namespace) -> int:
    if args.apply:
        if not args.clean_output_root or not args.extract_output_root:
            raise ValueError(
                "--clean-output-root and --extract-output-root are required with --apply."
            )
        plan = split_episodes(
            args.dataset_root,
            clean_output_root=args.clean_output_root,
            extract_output_root=args.extract_output_root,
            episode_indices=args.episodes,
            overwrite=bool(args.overwrite),
        )
    else:
        plan = plan_episode_split(
            args.dataset_root,
            args.episodes,
            clean_output_root=_optional_path(args.clean_output_root),
            extract_output_root=_optional_path(args.extract_output_root),
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_label_force_collisions(args: argparse.Namespace) -> int:
    if args.apply:
        plan = label_force_collisions(args.dataset_root, override=str(args.override))
    else:
        plan = plan_force_collision_labels(
            args.dataset_root, override=str(args.override)
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_plan_causal_align(args: argparse.Namespace) -> int:
    plan = plan_causal_align(
        args.dataset_root,
        output_root=_optional_path(args.output_root),
        align_cameras=bool(args.align_cameras),
        align_tactile=bool(args.align_tactile),
        no_prior_policy=str(args.no_prior_policy),
    )
    _print_json(plan.to_dict())
    return 0


def _cmd_camera_semantic_materialize(args: argparse.Namespace) -> int:
    plan = plan_camera_semantic_materialization(
        args.dataset_root,
        args.output_root,
        args.registry,
    )
    if bool(args.apply):
        result = materialize_camera_semantics(
            args.dataset_root,
            args.output_root,
            args.registry,
            expected_plan_hash=str(args.expected_plan_hash or ""),
        )
        _print_json(result)
    else:
        _print_json(plan.to_dict())
    return 0


def _cmd_camera_semantic_prepare_view(args: argparse.Namespace) -> int:
    if bool(args.apply) and bool(args.dry_run):
        raise ValueError("--apply and --dry-run are mutually exclusive.")
    plan = plan_camera_semantic_prepared_view(
        args.dataset_root,
        args.output_root,
        args.registry,
    )
    if bool(args.apply):
        result = materialize_camera_semantic_prepared_view(
            args.dataset_root,
            args.output_root,
            args.registry,
            expected_plan_hash=str(args.expected_plan_hash or ""),
        )
        _print_json(result)
    else:
        _print_json(plan.to_dict())
    return 0


def _cmd_causal_align(args: argparse.Namespace) -> int:
    if not args.output_root:
        raise ValueError("--output-root is required.")
    plan = causal_align_dataset(
        args.dataset_root,
        args.output_root,
        align_cameras=bool(args.align_cameras),
        align_tactile=bool(args.align_tactile),
        no_prior_policy=str(args.no_prior_policy),
        overwrite=bool(args.overwrite),
    )
    _print_json(plan.to_dict())
    return 0


def _cmd_v21_to_v22(args: argparse.Namespace) -> int:
    layout = load_v22_layout_json(args.layout_json)
    plan = plan_v21_to_v22_conversion(args.dataset_root, args.output_root, layout)
    if bool(args.apply):
        plan = convert_v21_dataset_to_v22(
            args.dataset_root,
            args.output_root,
            layout,
            hardlink_external=not bool(args.copy_external),
        )
    _print_json(plan.to_dict())
    return 0


def _cmd_validate_v23(args: argparse.Namespace) -> int:
    report = validate_v23_dataset(args.dataset_root, max_errors=int(args.max_errors))
    _print_json(report.to_dict())
    return 0 if report.ok else 1


def _cmd_modality_status(args: argparse.Namespace) -> int:
    _print_json(
        modality_storage_status(
            args.dataset_root,
            inventory_hash=_optional_path(args.inventory_hash),
        )
    )
    return 0


def _cmd_tactile_bundle_status(args: argparse.Namespace) -> int:
    attached = attach_tactile_bundle(
        args.dataset_root,
        args.tactile_root,
        verify_payload_hashes=bool(args.verify_payload_hashes),
    )
    _print_json(attached.report.to_dict())
    return 0


def _cmd_tactile_bundle_materialize(args: argparse.Namespace) -> int:
    plan = plan_tactile_bundle_materialization(
        args.dataset_root,
        args.tactile_root,
    )
    if not bool(args.apply):
        _print_json(plan.to_dict())
        return 0
    if (
        str(args.plan_hash) != plan.plan_hash
        or str(args.confirm_dataset_uuid) != plan.dataset_uuid
        or not bool(args.confirm_no_active_writers)
        or not bool(args.confirm_no_active_readers)
    ):
        raise ValueError(
            "--apply requires the current --plan-hash, matching "
            "--confirm-dataset-uuid, --confirm-no-active-writers, and "
            "--confirm-no-active-readers"
        )
    report = materialize_tactile_bundle(
        args.dataset_root,
        args.tactile_root,
        expected_plan_hash=plan.plan_hash,
    )
    _print_json(report.to_dict())
    return 0


def _cmd_modality_detach(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    if bool(args.apply):
        if (
            not args.plan_hash
            or not args.confirm_dataset_fingerprint
            or not args.confirm_no_active_readers
        ):
            raise ValueError(
                "--plan-hash, --confirm-dataset-fingerprint, and "
                "--confirm-no-active-readers are required with --apply"
            )
        result: Any = detach_modality(
            args.dataset_root,
            args.cold_root,
            expected_plan_hash=str(args.plan_hash),
            confirm_dataset_fingerprint=str(args.confirm_dataset_fingerprint),
            detachment_scope=str(args.detach_scope),
            min_free_bytes=min_free_bytes,
            source_storage_id=str(args.source_storage_id),
            target_storage_id=str(args.target_storage_id),
            confirm_no_active_readers=bool(args.confirm_no_active_readers),
            allow_legacy_hardlinked_source=bool(
                args.allow_legacy_hardlinked_source
            ),
            legacy_scan_payload_hashes=bool(args.legacy_scan_payload_hashes),
        )
    else:
        result = plan_modality_detach(
            args.dataset_root,
            args.cold_root,
            detachment_scope=str(args.detach_scope),
            min_free_bytes=min_free_bytes,
            allow_legacy_hardlinked_source=bool(
                args.allow_legacy_hardlinked_source
            ),
            legacy_scan_payload_hashes=bool(args.legacy_scan_payload_hashes),
        )
    _print_json(result.to_dict())
    return 0


def _cmd_modality_export(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    if bool(args.apply):
        if (
            not args.plan_hash
            or not args.confirm_dataset_fingerprint
            or not args.confirm_no_active_readers
        ):
            raise ValueError(
                "--plan-hash, --confirm-dataset-fingerprint, and "
                "--confirm-no-active-readers are required with --apply"
            )
        result: Any = export_modality(
            args.dataset_root,
            args.cold_root,
            expected_plan_hash=str(args.plan_hash),
            confirm_dataset_fingerprint=str(args.confirm_dataset_fingerprint),
            min_free_bytes=min_free_bytes,
            source_storage_id=str(args.source_storage_id),
            target_storage_id=str(args.target_storage_id),
            confirm_no_active_readers=True,
            allow_legacy_source=bool(args.allow_legacy_source),
            legacy_scan_payload_hashes=bool(args.legacy_scan_payload_hashes),
        )
    else:
        result = plan_modality_export(
            args.dataset_root,
            args.cold_root,
            min_free_bytes=min_free_bytes,
            allow_legacy_source=bool(args.allow_legacy_source),
            legacy_scan_payload_hashes=bool(args.legacy_scan_payload_hashes),
        )
    _print_json(result.to_dict())
    return 0


def _cmd_modality_move(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    if bool(args.apply):
        if not args.confirm_no_active_readers:
            raise ValueError(
                "--confirm-no-active-readers is required with --apply"
            )
        result: Any = move_modality(
            args.dataset_root,
            args.cold_root,
            min_free_bytes=min_free_bytes,
            source_storage_id=str(args.source_storage_id),
            target_storage_id=str(args.target_storage_id),
            confirm_no_active_readers=True,
        )
    else:
        result = plan_modality_detach(
            args.dataset_root,
            args.cold_root,
            detachment_scope="whole-prefix",
            min_free_bytes=min_free_bytes,
        )
    _print_json(result.to_dict())
    return 0


def _cmd_modality_rehydrate(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    if bool(args.apply):
        if (
            not args.plan_hash
            or not args.confirm_dataset_fingerprint
            or not args.confirm_no_active_readers
        ):
            raise ValueError(
                "--plan-hash, --confirm-dataset-fingerprint, and "
                "--confirm-no-active-readers are required with --apply"
            )
        result: Any = rehydrate_modality(
            args.dataset_root,
            args.cold_root,
            str(args.inventory_hash),
            expected_plan_hash=str(args.plan_hash),
            confirm_dataset_fingerprint=str(args.confirm_dataset_fingerprint),
            min_free_bytes=min_free_bytes,
            source_storage_id=str(args.source_storage_id),
            target_storage_id=str(args.target_storage_id),
            confirm_no_active_readers=bool(args.confirm_no_active_readers),
        )
    else:
        result = plan_modality_rehydrate(
            args.dataset_root,
            args.cold_root,
            str(args.inventory_hash),
            min_free_bytes=min_free_bytes,
        )
    _print_json(result.to_dict())
    return 0


def _cmd_v22_to_v23_desk(args: argparse.Namespace) -> int:
    kwargs = {"target_layout_path": args.target_layout_json}
    if bool(args.apply_v22_desk_conversion):
        plan = convert_v22_dataset_to_v23_desk(
            args.dataset_root,
            args.output_root,
            hardlink_external=not bool(args.copy_external),
            **kwargs,
        )
        _print_json(plan.to_dict(mode="applied"))
    else:
        plan = plan_v22_to_v23_desk_conversion(
            args.dataset_root,
            args.output_root,
            **kwargs,
        )
        _print_json(plan.to_dict())
    return 0


def _cmd_umi_workspace_near_boundary_shift(args: argparse.Namespace) -> int:
    result = migrate_umi_workspace_near_boundary(
        args.dataset_root,
        new_origin_in_old_x_m=float(args.new_origin_in_old_x_m),
        backup_root=_optional_path(args.backup_root),
        apply_workspace_near_boundary_shift=bool(
            args.apply_workspace_near_boundary_shift
        ),
    )
    _print_json(result)
    return 0


def _cmd_umi_arm_action_next_observation(args: argparse.Namespace) -> int:
    result = migrate_umi_arm_action_to_next_observation(
        args.dataset_root,
        backup_root=_optional_path(args.backup_root),
        apply_umi_arm_alignment=bool(args.apply_umi_arm_alignment),
    )
    _print_json(result.to_dict())
    return 0


def _cmd_tactile_recompress_zstd(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    max_manifests = None if args.max_manifests is None else int(args.max_manifests)
    if bool(args.apply):
        result = recompress_tactile_zstd_in_place(
            args.dataset_root,
            level=int(args.level),
            chunk_frames=int(args.chunk_frames),
            min_free_bytes=min_free_bytes,
            max_manifests=max_manifests,
            delete_originals=bool(args.delete_originals),
            backend=str(args.backend),
        )
        _print_json(result.to_dict())
    else:
        plan = plan_tactile_zstd_recompression(
            args.dataset_root,
            max_manifests=max_manifests,
            min_free_bytes=min_free_bytes,
        )
        _print_json(plan.to_dict())
    return 0


def _cmd_tactile_maintain(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    kwargs = {
        "target": str(args.target),
        "cold_root": _optional_path(args.cold_root),
        "min_free_bytes": min_free_bytes,
        "compress_before_detach": bool(args.compress_before_detach),
    }
    if bool(args.apply):
        outcome: Any = apply_tactile_postprocess(
            args.dataset_root,
            level=int(args.level),
            chunk_frames=int(args.chunk_frames),
            backend=str(args.backend),
            confirm_no_active_readers=bool(args.confirm_no_active_readers),
            **kwargs,
        )
    else:
        outcome = plan_tactile_postprocess(args.dataset_root, **kwargs)
    _print_json(outcome.to_dict())
    return 0


def _cmd_postprocess(args: argparse.Namespace) -> int:
    if not bool(args.tactile_compact):
        _print_json(
            {"dataset_root": str(Path(args.dataset_root).resolve()), "steps": []}
        )
        return 0
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    if bool(args.apply):
        outcome: Any = compact_tactile_dataset_in_place(
            args.dataset_root,
            level=int(args.level),
            chunk_frames=int(args.chunk_frames),
            min_free_bytes=min_free_bytes,
            delete_originals=True,
            backend=str(args.backend),
        )
        mode = "applied"
    else:
        outcome = plan_tactile_compaction(
            args.dataset_root,
            min_free_bytes=min_free_bytes,
        )
        mode = "dry-run"
    _print_json(
        {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "mode": mode,
            "steps": [
                {"name": "tactile-compact-v1", "default": True, "result": outcome.to_dict()}
            ],
        }
    )
    return 0


def _cmd_tactile_compact(args: argparse.Namespace) -> int:
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    max_manifests = None if args.max_manifests is None else int(args.max_manifests)
    if bool(args.apply):
        outcome: Any = compact_tactile_dataset_in_place(
            args.dataset_root,
            level=int(args.level),
            chunk_frames=int(args.chunk_frames),
            min_free_bytes=min_free_bytes,
            max_manifests=max_manifests,
            delete_originals=bool(args.delete_originals),
            backend=str(args.backend),
        )
    else:
        outcome = plan_tactile_compaction(
            args.dataset_root,
            min_free_bytes=min_free_bytes,
        )
    _print_json(outcome.to_dict())
    return 0


def _cmd_hardlink_dedupe(args: argparse.Namespace) -> int:
    plan = plan_hardlink_dedupe(
        args.target_root,
        args.reference_root,
        min_file_bytes=int(float(args.min_file_mb) * 1024**2),
        include_prefixes=args.include_prefix or ("videos", "tactile", "wrist_depth"),
    )
    if bool(args.apply):
        result = apply_hardlink_dedupe(
            plan,
            confirm_replace_targets=bool(args.confirm_replace_targets),
        )
        _print_json(result.to_dict())
    else:
        _print_json(plan.to_dict(include_replacements=bool(args.show_files)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess DexTeleop LeRobot-style datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera_materialize = subparsers.add_parser(
        "camera-semantics-materialize",
        help=(
            "Plan or materialize serial-resolved wrist-camera namespaces into "
            "a fresh ordinary-loader dataset without modifying the source."
        ),
    )
    camera_materialize.add_argument("dataset_root")
    camera_materialize.add_argument("--output-root", required=True)
    camera_materialize.add_argument("--registry", required=True)
    camera_materialize.add_argument(
        "--expected-plan-hash",
        default="",
        help="Required with --apply; copy from the immediately preceding dry-run.",
    )
    camera_materialize.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Publish a fresh-inode output only after Parquet, receipt, decoded "
            "frame-count, and full/tail video validation."
        ),
    )
    camera_materialize.set_defaults(func=_cmd_camera_semantic_materialize)

    camera_prepare_view = subparsers.add_parser(
        "camera-semantics-prepare-view",
        help=(
            "Plan or build a camera-corrected numeric/video prepared view "
            "without scanning or copying tactile payloads."
        ),
    )
    camera_prepare_view.add_argument("dataset_root")
    camera_prepare_view.add_argument("--output-root", required=True)
    camera_prepare_view.add_argument("--registry", required=True)
    camera_prepare_view.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Explicit no-write marker; planning is already the default and "
            "this flag is useful for remote safety wrappers."
        ),
    )
    camera_prepare_view.add_argument(
        "--expected-plan-hash",
        default="",
        help="Required with --apply; copy from the immediately preceding dry-run.",
    )
    camera_prepare_view.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Publish a fresh-inode, non-tactile prepared view after strict "
            "Parquet, camera receipt and full video validation."
        ),
    )
    camera_prepare_view.set_defaults(func=_cmd_camera_semantic_prepare_view)

    default_postprocess = subparsers.add_parser(
        "postprocess",
        help="Run the default finalized-dataset postprocess (tactile compact v1 is enabled).",
    )
    default_postprocess.add_argument("dataset_root")
    default_postprocess.add_argument(
        "--tactile-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standard 120x160/quantized offline tactile compaction (default: enabled).",
    )
    default_postprocess.add_argument("--level", type=int, default=1)
    default_postprocess.add_argument(
        "--chunk-frames", type=int, default=DEFAULT_TACTILE_COMPACT_CHUNK_FRAMES
    )
    default_postprocess.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_TACTILE_COMPACT_MIN_FREE_BYTES / 1024**3,
    )
    default_postprocess.add_argument(
        "--backend", choices=("auto", "python", "cli"), default="auto"
    )
    default_postprocess.add_argument(
        "--apply",
        action="store_true",
        help="Apply all enabled default steps; otherwise only print their plans.",
    )
    default_postprocess.set_defaults(func=_cmd_postprocess)

    stutters = subparsers.add_parser("plan-stutters")
    stutters.add_argument("dataset_root")
    stutters.add_argument("--max-gap-sec", type=float, default=0.2)
    stutters.add_argument("--min-segment-frames", type=int, default=1)
    stutters.set_defaults(func=_cmd_plan_stutters)

    clean_stutter_parser = subparsers.add_parser("clean-stutters")
    clean_stutter_parser.add_argument("dataset_root")
    clean_stutter_parser.add_argument("--output-root", default="")
    clean_stutter_parser.add_argument("--max-gap-sec", type=float, default=0.2)
    clean_stutter_parser.add_argument("--min-segment-frames", type=int, default=1)
    clean_stutter_parser.add_argument(
        "--reset-last-action", action=argparse.BooleanOptionalAction, default=True
    )
    clean_stutter_parser.add_argument("--apply", action="store_true")
    clean_stutter_parser.add_argument("--overwrite", action="store_true")
    clean_stutter_parser.set_defaults(func=_cmd_clean_stutters)

    fault_nan = subparsers.add_parser("plan-fault-nan")
    fault_nan.add_argument("dataset_root")
    fault_nan.add_argument("--min-nan-fraction", type=float, default=0.8)
    fault_nan.set_defaults(func=_cmd_plan_fault_nan)

    delete = subparsers.add_parser("delete-episodes")
    delete.add_argument("dataset_root")
    delete.add_argument("--episodes", required=True, help="Selection such as 1,3,5-8.")
    delete.add_argument("--output-root", default="")
    delete.add_argument("--apply", action="store_true")
    delete.add_argument("--overwrite", action="store_true")
    delete.set_defaults(func=_cmd_delete_episodes)

    trim = subparsers.add_parser("trim-nan-tails")
    trim.add_argument("dataset_root")
    trim.add_argument("--output-root", default="")
    trim.add_argument("--min-nan-fraction", type=float, default=0.8)
    trim.add_argument("--apply", action="store_true")
    trim.add_argument("--overwrite", action="store_true")
    trim.set_defaults(func=_cmd_trim_nan_tails)

    split = subparsers.add_parser("split-episodes")
    split.add_argument("dataset_root")
    split.add_argument("--episodes", required=True, help="Selection such as 1,3,5-8.")
    split.add_argument("--clean-output-root", default="")
    split.add_argument("--extract-output-root", default="")
    split.add_argument("--apply", action="store_true")
    split.add_argument("--overwrite", action="store_true")
    split.set_defaults(func=_cmd_split_episodes)

    force_collision = subparsers.add_parser("label-force-collisions")
    force_collision.add_argument("dataset_root")
    force_collision.add_argument(
        "--override",
        choices=("auto", "all-detected", "all-clean"),
        default="auto",
        help="Override force collision labels instead of deriving them from diagnostics.",
    )
    force_collision.add_argument("--apply", action="store_true")
    force_collision.set_defaults(func=_cmd_label_force_collisions)

    causal_plan = subparsers.add_parser("plan-causal-align")
    causal_plan.add_argument("dataset_root")
    causal_plan.add_argument("--output-root", default="")
    causal_plan.add_argument(
        "--align-cameras", action=argparse.BooleanOptionalAction, default=True
    )
    causal_plan.add_argument(
        "--align-tactile", action=argparse.BooleanOptionalAction, default=True
    )
    causal_plan.add_argument(
        "--no-prior-policy",
        choices=("keep_current", "first", "null"),
        default="keep_current",
        help="Policy when an episode has no modality frame at or before a sample timestamp.",
    )
    causal_plan.set_defaults(func=_cmd_plan_causal_align)

    causal = subparsers.add_parser("causal-align")
    causal.add_argument("dataset_root")
    causal.add_argument("--output-root", required=True)
    causal.add_argument(
        "--align-cameras", action=argparse.BooleanOptionalAction, default=True
    )
    causal.add_argument(
        "--align-tactile", action=argparse.BooleanOptionalAction, default=True
    )
    causal.add_argument(
        "--no-prior-policy",
        choices=("keep_current", "first", "null"),
        default="keep_current",
        help="Policy when an episode has no modality frame at or before a sample timestamp.",
    )
    causal.add_argument("--overwrite", action="store_true")
    causal.set_defaults(func=_cmd_causal_align)

    v21_to_v22 = subparsers.add_parser(
        "v21-to-v22",
        help="Plan or create a copy-on-write DexTeleop v2.2 policy-I/O conversion.",
    )
    v21_to_v22.add_argument("dataset_root")
    v21_to_v22.add_argument("--output-root", required=True)
    v21_to_v22.add_argument(
        "--layout-json",
        required=True,
        help="Canonical v2.2 layout JSON, or metadata JSON containing policy_io_layout.",
    )
    v21_to_v22.add_argument(
        "--apply",
        action="store_true",
        help="Create the new output root. Without this flag the command only prints the plan.",
    )
    v21_to_v22.add_argument(
        "--copy-external",
        action="store_true",
        help="Copy videos/tactile artifacts instead of hard-linking when possible.",
    )
    v21_to_v22.set_defaults(func=_cmd_v21_to_v22)

    v22_to_v23_desk = subparsers.add_parser(
        "v22-to-v23-desk",
        help=(
            "Plan or create an RM65 v2.2 canonical-desk v2.3 compatibility "
            "derivative; this does not claim native v2.3 causal snapshots."
        ),
    )
    v22_to_v23_desk.add_argument("dataset_root")
    v22_to_v23_desk.add_argument("--output-root", required=True)
    v22_to_v23_desk.add_argument(
        "--target-layout-json",
        default="",
        help=(
            "Optional canonical desk PolicyIOLayout JSON. By default the source "
            "layout is preserved except for its arm reference frame/provenance."
        ),
    )
    v22_to_v23_desk.add_argument(
        "--apply-v22-desk-conversion",
        action="store_true",
        help=(
            "Create the copy-on-write output dataset. Without this exact flag, "
            "the command performs a full read-only scan."
        ),
    )
    v22_to_v23_desk.add_argument(
        "--copy-external",
        action="store_true",
        help="Copy video/tactile artifacts instead of hard-linking when possible.",
    )
    v22_to_v23_desk.set_defaults(func=_cmd_v22_to_v23_desk)

    validate_v23 = subparsers.add_parser(
        "validate-v23",
        help="Read-only acceptance scan for a finalized native DexTeleop v2.3 dataset.",
    )
    validate_v23.add_argument("dataset_root")
    validate_v23.add_argument("--max-errors", type=int, default=200)
    validate_v23.set_defaults(func=_cmd_validate_v23)

    modality_status = subparsers.add_parser(
        "modality-status",
        help="Read-only detachable-modality availability and receipt status.",
    )
    modality_status.add_argument("dataset_root")
    modality_status.add_argument("--modality", choices=("tactile",), default="tactile")
    modality_status.add_argument("--inventory-hash", default="")
    modality_status.set_defaults(func=_cmd_modality_status)

    tactile_bundle_status = subparsers.add_parser(
        "tactile-bundle-status",
        help=(
            "Attach a native tactile bundle read-only and verify dataset, Episode, "
            "commit, artifact, and frame-alignment identities."
        ),
    )
    tactile_bundle_status.add_argument("dataset_root")
    tactile_bundle_status.add_argument("--tactile-root", required=True)
    tactile_bundle_status.add_argument(
        "--verify-payload-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hash every tactile payload (default). --no-verify-payload-hashes "
            "still validates manifests, sizes, identities, and alignment."
        ),
    )
    tactile_bundle_status.set_defaults(func=_cmd_tactile_bundle_status)

    tactile_bundle_materialize = subparsers.add_parser(
        "tactile-bundle-materialize",
        help=(
            "Plan or atomically restore an attached native tactile bundle as a "
            "self-contained dataset tactile/ tree using fresh inodes."
        ),
    )
    tactile_bundle_materialize.add_argument("dataset_root")
    tactile_bundle_materialize.add_argument("--tactile-root", required=True)
    tactile_bundle_materialize.add_argument("--plan-hash", default="")
    tactile_bundle_materialize.add_argument("--confirm-dataset-uuid", default="")
    tactile_bundle_materialize.add_argument(
        "--confirm-no-active-writers",
        action="store_true",
        help=(
            "Required with --apply; confirms collection and every other dataset "
            "writer are stopped."
        ),
    )
    tactile_bundle_materialize.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help="Required with --apply before the tactile/ namespace is published.",
    )
    tactile_bundle_materialize.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the exact fresh plan. Without this flag the command is a "
            "read-only full-hash proposal."
        ),
    )
    tactile_bundle_materialize.set_defaults(func=_cmd_tactile_bundle_materialize)

    modality_export = subparsers.add_parser(
        "modality-export",
        help=(
            "Plan or copy and verify tactile artifacts on independent cold "
            "storage while preserving every hot source file."
        ),
    )
    modality_export.add_argument("dataset_root")
    modality_export.add_argument("--cold-root", required=True)
    modality_export.add_argument("--modality", choices=("tactile",), default="tactile")
    modality_export.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MODALITY_STORAGE_MIN_FREE_BYTES / 1024**3,
    )
    modality_export.add_argument("--source-storage-id", default="hot.local")
    modality_export.add_argument("--target-storage-id", default="cold.archive")
    modality_export.add_argument(
        "--allow-legacy-source",
        action="store_true",
        help=(
            "Allow legacy dataset metadata only for source-preserving export; "
            "the tactile graph still requires exact inventory/hash and reader checks."
        ),
    )
    modality_export.add_argument(
        "--legacy-scan-payload-hashes",
        action="store_true",
        help=(
            "Break-glass legacy export mode: hash actual payload bytes instead "
            "of trusting stale sealed hashes. Requires --allow-legacy-source."
        ),
    )
    modality_export.add_argument("--plan-hash", default="")
    modality_export.add_argument("--confirm-dataset-fingerprint", default="")
    modality_export.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help=(
            "Required with --apply until all external readers participate in a "
            "shared dataset lease."
        ),
    )
    modality_export.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Copy and verify the frozen proposal without retiring or renaming "
            "any hot artifact. Without this flag the command is read-only."
        ),
    )
    modality_export.set_defaults(func=_cmd_modality_export)

    modality_detach = subparsers.add_parser(
        "modality-detach",
        help=(
            "Plan or transactionally export, restore-drill, and retire tactile "
            "payloads. Apply requires a distinct filesystem and frozen confirmations."
        ),
    )
    modality_detach.add_argument("dataset_root")
    modality_detach.add_argument("--cold-root", required=True)
    modality_detach.add_argument("--modality", choices=("tactile",), default="tactile")
    modality_detach.add_argument(
        "--detach-scope",
        choices=("payload-only", "whole-prefix"),
        default="payload-only",
        help=(
            "Retain hot manifests (payload-only, compatibility default) or "
            "transactionally detach the complete tactile/ tree (whole-prefix)."
        ),
    )
    modality_detach.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MODALITY_STORAGE_MIN_FREE_BYTES / 1024**3,
    )
    modality_detach.add_argument("--source-storage-id", default="hot.local")
    modality_detach.add_argument("--target-storage-id", default="cold.archive")
    modality_detach.add_argument("--plan-hash", default="")
    modality_detach.add_argument("--confirm-dataset-fingerprint", default="")
    modality_detach.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help=(
            "Required with --apply until all external readers participate in a "
            "shared dataset lease."
        ),
    )
    modality_detach.add_argument(
        "--allow-legacy-hardlinked-source",
        action="store_true",
        help=(
            "Break-glass support for historical datasets whose tactile paths "
            "share inodes. The default remains fail-closed; apply still requires "
            "an independent verified cold copy and restore drill before unlinking "
            "only this dataset's paths."
        ),
    )
    modality_detach.add_argument(
        "--legacy-scan-payload-hashes",
        action="store_true",
        help=(
            "Break-glass legacy detach mode: hash actual payload bytes instead "
            "of trusting stale sealed hashes. Requires "
            "--allow-legacy-hardlinked-source."
        ),
    )
    modality_detach.add_argument(
        "--apply",
        action="store_true",
        help="Apply exactly the preceding proposal; otherwise this command is read-only.",
    )
    modality_detach.set_defaults(func=_cmd_modality_detach)

    modality_move = subparsers.add_parser(
        "modality-move",
        help=(
            "Safely move the complete tactile/ module to cold storage with a "
            "resumable, verified transfer. Re-run the same command after interruption."
        ),
    )
    modality_move.add_argument("dataset_root")
    modality_move.add_argument("--cold-root", required=True)
    modality_move.add_argument("--modality", choices=("tactile",), default="tactile")
    modality_move.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MODALITY_STORAGE_MIN_FREE_BYTES / 1024**3,
    )
    modality_move.add_argument("--source-storage-id", default="hot.local")
    modality_move.add_argument("--target-storage-id", default="cold.archive")
    modality_move.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help="Required with --apply; confirms the dataset is not being read.",
    )
    modality_move.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Execute a freshly frozen whole-prefix plan. Without this flag the "
            "command remains a read-only proposal."
        ),
    )
    modality_move.set_defaults(func=_cmd_modality_move)

    modality_rehydrate = subparsers.add_parser(
        "modality-rehydrate",
        help=(
            "Plan or transactionally rebuild a complete hot tactile tree from a "
            "verified cold inventory."
        ),
    )
    modality_rehydrate.add_argument("dataset_root")
    modality_rehydrate.add_argument("--cold-root", required=True)
    modality_rehydrate.add_argument("--modality", choices=("tactile",), default="tactile")
    modality_rehydrate.add_argument("--inventory-hash", required=True)
    modality_rehydrate.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MODALITY_STORAGE_MIN_FREE_BYTES / 1024**3,
    )
    modality_rehydrate.add_argument("--source-storage-id", default="cold.archive")
    modality_rehydrate.add_argument("--target-storage-id", default="hot.local")
    modality_rehydrate.add_argument("--plan-hash", default="")
    modality_rehydrate.add_argument("--confirm-dataset-fingerprint", default="")
    modality_rehydrate.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help=(
            "Required with --apply until all external readers participate in a "
            "shared dataset lease."
        ),
    )
    modality_rehydrate.add_argument(
        "--apply",
        action="store_true",
        help="Apply exactly the preceding proposal; otherwise this command is read-only.",
    )
    modality_rehydrate.set_defaults(func=_cmd_modality_rehydrate)

    workspace_shift = subparsers.add_parser(
        "umi-workspace-near-boundary-shift",
        help=(
            "Explicitly migrate legacy UMI table-center coordinates; disabled and "
            "read-only unless its migration-specific apply flag is present."
        ),
    )
    workspace_shift.add_argument("dataset_root")
    workspace_shift.add_argument(
        "--new-origin-in-old-x-m",
        type=float,
        default=-0.15,
        help=(
            "New origin X in the legacy table-center frame. Default -0.15 means "
            "stored pose X is shifted by +0.15 m."
        ),
    )
    workspace_shift.add_argument(
        "--backup-root",
        default="",
        help="Optional explicit backup path; the default is an adjacent timestamped path.",
    )
    workspace_shift.add_argument(
        "--apply-workspace-near-boundary-shift",
        action="store_true",
        help=(
            "Enable this exact in-place migration after backup. Without this flag "
            "the command performs a full read-only dry-run."
        ),
    )
    workspace_shift.set_defaults(func=_cmd_umi_workspace_near_boundary_shift)

    umi_arm_alignment = subparsers.add_parser(
        "umi-arm-action-next-observation",
        help=(
            "Explicitly migrate coupled historical UMI arm actions to the next "
            "accepted observation; disabled unless its dedicated apply flag is present."
        ),
    )
    umi_arm_alignment.add_argument("dataset_root")
    umi_arm_alignment.add_argument(
        "--backup-root",
        default="",
        help="Optional adjacent backup path; a timestamped path is used by default.",
    )
    umi_arm_alignment.add_argument(
        "--apply-umi-arm-alignment",
        action="store_true",
        help=(
            "Create a hard-link snapshot and atomically replace the dataset. Without "
            "this exact flag, the command performs a read-only full preflight."
        ),
    )
    umi_arm_alignment.set_defaults(func=_cmd_umi_arm_action_next_observation)

    tactile_zstd = subparsers.add_parser(
        "tactile-recompress-zstd",
        help="Plan or atomically recompress finalized raw tactile streams in place.",
    )
    tactile_zstd.add_argument("dataset_root")
    tactile_zstd.add_argument("--level", type=int, default=1)
    tactile_zstd.add_argument("--chunk-frames", type=int, default=8)
    tactile_zstd.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_BYTES / 1024**3,
        help="Free-space reserve that must remain while staging one manifest.",
    )
    tactile_zstd.add_argument(
        "--max-manifests",
        type=int,
        default=None,
        help="Limit work for a staged rollout; manifests are finalized episode-side records.",
    )
    tactile_zstd.add_argument(
        "--backend",
        choices=("auto", "python", "cli"),
        default="auto",
    )
    tactile_zstd.add_argument(
        "--apply",
        action="store_true",
        help="Perform the in-place conversion. Without this flag the command is read-only.",
    )
    tactile_zstd.add_argument(
        "--delete-originals",
        action="store_true",
        help="Required with --apply; originals are deleted only after manifest commit.",
    )
    tactile_zstd.set_defaults(func=_cmd_tactile_recompress_zstd)

    tactile_maintain = subparsers.add_parser(
        "tactile-maintain",
        help=(
            "State-aware dry-run/apply interface for mixed raw, compressed, "
            "and detached tactile datasets."
        ),
    )
    tactile_maintain.add_argument("dataset_root")
    tactile_maintain.add_argument(
        "--target",
        choices=TACTILE_MAINTENANCE_TARGETS,
        default=TACTILE_TARGET_COMPRESSED_HOT,
        help=(
            "Desired physical state: hot keeps existing encodings, compressed-hot "
            "losslessly converts only raw streams, detached compresses raw streams "
            "then safely moves the whole tactile prefix."
        ),
    )
    tactile_maintain.add_argument(
        "--cold-root",
        default="",
        help="Required when the target needs detach or rehydrate.",
    )
    tactile_maintain.add_argument("--level", type=int, default=1)
    tactile_maintain.add_argument("--chunk-frames", type=int, default=8)
    tactile_maintain.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_BYTES / 1024**3,
    )
    tactile_maintain.add_argument(
        "--backend", choices=("auto", "python", "cli"), default="auto"
    )
    tactile_maintain.add_argument(
        "--compress-before-detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Losslessly compress remaining raw streams before whole-prefix detach.",
    )
    tactile_maintain.add_argument(
        "--confirm-no-active-readers",
        action="store_true",
        help="Required when apply performs detach or rehydrate.",
    )
    tactile_maintain.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned target; without this flag the command is read-only.",
    )
    tactile_maintain.set_defaults(func=_cmd_tactile_maintain)

    tactile_compact = subparsers.add_parser(
        "tactile-compact",
        help=(
            "Plan or atomically migrate tactile streams to the canonical compact v1 "
            "profiles without changing frame frequency."
        ),
    )
    tactile_compact.add_argument("dataset_root")
    tactile_compact.add_argument("--level", type=int, default=1)
    tactile_compact.add_argument(
        "--chunk-frames", type=int, default=DEFAULT_TACTILE_COMPACT_CHUNK_FRAMES
    )
    tactile_compact.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_TACTILE_COMPACT_MIN_FREE_BYTES / 1024**3,
    )
    tactile_compact.add_argument("--max-manifests", type=int, default=None)
    tactile_compact.add_argument(
        "--backend", choices=("auto", "python", "cli"), default="auto"
    )
    tactile_compact.add_argument("--apply", action="store_true")
    tactile_compact.add_argument(
        "--delete-originals",
        action="store_true",
        help="Required with --apply; source streams are removed only after manifest commit.",
    )
    tactile_compact.set_defaults(func=_cmd_tactile_compact)

    hardlink_dedupe = subparsers.add_parser(
        "hardlink-dedupe",
        help="SHA-256 audit or target-only atomic hardlink dedupe.",
    )
    hardlink_dedupe.add_argument("target_root")
    hardlink_dedupe.add_argument(
        "--reference-root",
        action="append",
        required=True,
        help="Repeat for multiple roots.",
    )
    hardlink_dedupe.add_argument("--min-file-mb", type=float, default=1.0)
    hardlink_dedupe.add_argument(
        "--include-prefix",
        action="append",
        help="Immutable artifact directory component; defaults to videos/tactile/wrist_depth.",
    )
    hardlink_dedupe.add_argument("--show-files", action="store_true")
    hardlink_dedupe.add_argument("--apply", action="store_true")
    hardlink_dedupe.add_argument(
        "--confirm-replace-targets",
        action="store_true",
        help="Required with --apply; source/reference files are never replaced.",
    )
    hardlink_dedupe.set_defaults(func=_cmd_hardlink_dedupe)

    writing_stage1_pilot = subparsers.add_parser(
        "writing-stage1-pilot",
        help=(
            "Preflight or publish the dataset-local 7.23/7.24 v2.3 "
            "writing-stage1 annotation pilot."
        ),
    )
    configure_writing_stage1_pilot(writing_stage1_pilot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
