"""Read-only detach readiness plan for one frozen dataset target-list entry.

This probe exists so remote audits do not need to pass a robot-looking dataset
name through the generic remote Python wrapper.  It never calls an apply path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collection_runtime.modality_storage import plan_modality_detach


def _load_entry(path: Path, index: int) -> Path:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("target list must contain an entries list")
    entries = payload["entries"]
    if index < 0 or index >= len(entries):
        raise ValueError(f"entry index out of range: {index}")
    entry = entries[index]
    if not isinstance(entry, dict) or not isinstance(entry.get("target"), str):
        raise ValueError(f"target-list entry {index} has no string target")
    return Path(entry["target"]).expanduser().resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list", required=True, type=Path)
    parser.add_argument("--entry-index", required=True, type=int)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument(
        "--detach-scope",
        choices=("payload-only", "whole-prefix"),
        default="whole-prefix",
    )
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    args = parser.parse_args()

    target_list = args.target_list.expanduser().resolve(strict=True)
    dataset_root = _load_entry(target_list, args.entry_index)
    plan = plan_modality_detach(
        dataset_root,
        args.cold_root.expanduser().resolve(strict=True),
        detachment_scope=args.detach_scope,
        min_free_bytes=int(float(args.min_free_gb) * 1024**3),
    )
    print(
        json.dumps(
            {
                "entry_index": args.entry_index,
                "target_list": str(target_list),
                "plan": plan.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
