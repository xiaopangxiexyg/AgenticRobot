"""CLI for the independent fine-stage annotation application."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
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
    args = build_parser().parse_args(argv)
    if args.dry_run:
        if args.dataset_root is None:
            build_parser().error(
                "dataset_root is required with --dry-run because no GUI is "
                "started"
            )
        items = list_progress_annotation_work_items(args.dataset_root)
        print(
            f"eligible_finalized_episodes={len(items)} "
            f"dataset_root={args.dataset_root.expanduser().resolve()}"
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
        args.dataset_root,
        annotator_id=args.annotator_id,
        second_review=args.second_review,
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
