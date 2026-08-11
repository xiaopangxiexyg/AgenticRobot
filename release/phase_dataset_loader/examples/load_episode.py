#!/usr/bin/env python3
"""Inspect one Episode without modifying the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from dexteleop_dataset_loader import (  # noqa: E402
    DexTeleopReleaseDataset,
    SelectionRequiredError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="camera to decode; repeat as needed (default: front)",
    )
    parser.add_argument("--annotation-set-id", default="")
    parser.add_argument("--skip-rgb", action="store_true")
    parser.add_argument("--skip-tactile", action="store_true")
    parser.add_argument("--skip-depth", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = DexTeleopReleaseDataset(args.dataset_root)
    episode = dataset.episode(args.episode)
    print(json.dumps(dataset.summary(), ensure_ascii=False, indent=2))
    print(
        f"episode={episode.episode_index} id={episode.episode_id} "
        f"frames={episode.frame_count} requested_frame={args.frame}"
    )

    table = dataset.load_parquet(args.episode)
    print(f"parquet_rows={table.num_rows} columns={table.column_names}")
    print(f"action={table.column('action')[args.frame].as_py()}")
    print(f"observation.state={table.column('observation.state')[args.frame].as_py()}")

    cameras = tuple(args.camera) if args.camera else ("front",)
    if not args.skip_rgb:
        for camera in cameras:
            image = dataset.load_rgb_frame(args.episode, camera, args.frame)
            print(f"rgb[{camera}] shape={image.shape} dtype={image.dtype}")

    if not args.skip_tactile:
        for side in ("left", "right"):
            tactile = dataset.load_tactile_frame(args.episode, side, args.frame)
            shapes = None if tactile is None else {key: value.shape for key, value in tactile.items()}
            print(f"tactile[{side}]={shapes}")

    if not args.skip_depth:
        for camera in ("left_wrist", "right_wrist"):
            mask = dataset.load_wrist_depth_mask(args.episode, camera, args.frame)
            detail = None if mask is None else {
                "shape": mask.shape,
                "valid_fraction": float(mask.mean()),
            }
            print(f"depth_mask[{camera}]={detail}")

    progress = dataset.load_progress_episode(args.episode)
    manifest = progress["manifest"]
    print(
        f"progress_facts={len(progress['facts'])} "
        f"progress_manifest_hash={manifest.get('semantic_content_hash')}"
    )

    annotation_ids = dataset.stage_annotation_set_ids(args.episode)
    print(f"stage_annotation_set_ids={annotation_ids}")
    try:
        annotation = dataset.load_stage_annotation(
            args.episode,
            annotation_set_id=args.annotation_set_id or None,
        )
    except SelectionRequiredError as exc:
        print(f"stage_annotation_selection_required={exc}")
    else:
        occurrences = annotation.annotations.get("occurrences", ())
        abnormal = annotation.annotations.get("episode_abnormal")
        print(
            f"selected_annotation={annotation.annotation_set_id} "
            f"occurrences={len(occurrences)} episode_abnormal={abnormal is not None}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
