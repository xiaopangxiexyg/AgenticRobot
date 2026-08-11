"""CLI for the independent fine-stage annotation application."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
)
from dataset_interface.writing_stage1_v23 import (
    writing_stage1_v23_semantic_layer_root,
)

from .server import RefinementAnnotationGui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_station refinement-annotation",
        description=(
            "Annotate ordered fine stages for finalized DexTeleop Episodes. "
            "This application never starts devices or robot control."
        ),
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help="optional dataset path used to prefill the GUI launcher",
    )
    parser.add_argument(
        "--annotator-id",
        default="",
        help=(
            "optional stable annotator identity used to prefill the GUI "
            "launcher"
        ),
    )
    parser.add_argument(
        "--second-review",
        action="store_true",
        help=(
            "prefill/open the workspace in second-review annotation mode; "
            "opening an Episode clears its pending-review label in the "
            "working draft"
        ),
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "explicit read-only source dataset root allowed by an external "
            "retrospective workspace; repeat for multiple roots"
        ),
    )
    parser.add_argument(
        "--writing-v23-semantic-layer",
        metavar="LAYER_ID",
        help=(
            "open one dataset-local retrospective writing layer; the "
            "positional path must be its owning v2.3 dataset root"
        ),
    )
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=0)
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="start the loopback service without opening a browser",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only list eligible finalized Episodes; do not acquire the writer lock",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    annotation_root = args.dataset_root
    allowed_source_roots = tuple(args.source_root)
    if args.writing_v23_semantic_layer:
        if args.dataset_root is None:
            parser.error(
                "dataset_root is required with --writing-v23-semantic-layer"
            )
        if args.source_root:
            parser.error(
                "--source-root cannot be combined with the dataset-local "
                "--writing-v23-semantic-layer adapter"
            )
        annotation_root = writing_stage1_v23_semantic_layer_root(
            args.dataset_root,
            semantic_layer_id=args.writing_v23_semantic_layer,
        )
        allowed_source_roots = (args.dataset_root.expanduser().resolve(),)
    if args.dry_run:
        if annotation_root is None:
            parser.error(
                "dataset_root is required with --dry-run because no GUI is "
                "started"
            )
        items = list_progress_annotation_work_items(
            annotation_root,
            allowed_source_roots=allowed_source_roots,
        )
        print(
            f"eligible_finalized_episodes={len(items)} "
            f"dataset_root={args.dataset_root.expanduser().resolve()} "
            f"annotation_root={Path(annotation_root).expanduser().resolve()}"
        )
        for item in items:
            print(
                f"episode_index={item.episode_index} "
                f"episode_uuid={item.episode_uuid} "
                f"frames={item.frame_count} cameras="
                f"{','.join(sorted(item.video_paths))}"
            )
        return 0
    gui = RefinementAnnotationGui(
        annotation_root,
        annotator_id=args.annotator_id,
        second_review=args.second_review,
        allowed_source_roots=allowed_source_roots,
        host=args.ui_host,
        port=args.ui_port,
        open_browser=not args.no_open_browser,
    )
    try:
        print(f"[RefinementAnnotation] ready url={gui.start()}")
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[RefinementAnnotation] stopping")
    finally:
        gui.close()
    return 0


__all__ = ["build_parser", "main"]
