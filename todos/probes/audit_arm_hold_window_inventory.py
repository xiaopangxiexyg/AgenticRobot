"""Print no-write H-window coverage for a frozen arm-hold split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_interface.arm_hold_split_manifest import load_arm_hold_split_manifest
from dataset_interface.arm_hold_windows import build_arm_hold_window_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit legal H-window and transition coverage without decoding RGB."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation"),
        help="Repeat to select splits; defaults to both.",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Required acknowledgement: this audit must never start devices or motion.",
    )
    args = parser.parse_args()
    if not args.no_motion:
        parser.error("--no-motion is required")

    manifest = load_arm_hold_split_manifest(args.manifest)
    splits = tuple(dict.fromkeys(args.split or ("train", "validation")))
    summaries = [
        build_arm_hold_window_inventory(
            manifest,
            args.workspace_root,
            split=split,
        ).summary_payload(debug=False)
        for split in splits
    ]
    print(
        json.dumps(
            {
                "audit": "arm_hold_window_inventory_v1",
                "writes_artifacts": False,
                "summaries": summaries,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
