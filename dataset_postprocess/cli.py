from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .causal_align import causal_align_dataset, plan_causal_align
from .delete_episodes import delete_episodes, plan_episode_deletion
from .fault_nan_tail import detect_fault_nan_tail
from .force_collision_labels import label_force_collisions, plan_force_collision_labels
from .io import episode_records, parquet_path_for_record, read_parquet_rows
from .split_episodes import plan_episode_split, split_episodes
from .stutter_segments import clean_stutters, detect_stutter_segments, plan_stutter_clean
from .trim_nan_tails import plan_nan_tail_trim, trim_nan_tails


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
            raise ValueError("--clean-output-root and --extract-output-root are required with --apply.")
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
        plan = plan_force_collision_labels(args.dataset_root, override=str(args.override))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess DexTeleop LeRobot-style datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    clean_stutter_parser.add_argument("--reset-last-action", action=argparse.BooleanOptionalAction, default=True)
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
    causal_plan.add_argument("--align-cameras", action=argparse.BooleanOptionalAction, default=True)
    causal_plan.add_argument("--align-tactile", action=argparse.BooleanOptionalAction, default=True)
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
    causal.add_argument("--align-cameras", action=argparse.BooleanOptionalAction, default=True)
    causal.add_argument("--align-tactile", action=argparse.BooleanOptionalAction, default=True)
    causal.add_argument(
        "--no-prior-policy",
        choices=("keep_current", "first", "null"),
        default="keep_current",
        help="Policy when an episode has no modality frame at or before a sample timestamp.",
    )
    causal.add_argument("--overwrite", action="store_true")
    causal.set_defaults(func=_cmd_causal_align)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
