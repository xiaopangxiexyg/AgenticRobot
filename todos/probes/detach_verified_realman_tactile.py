"""Plan or transactionally detach one relocation-plan tactile dataset.

The dataset is selected by immutable relocation-plan index so remote safety
wrappers never need a hardware-looking dataset leaf in the command line.  The
apply path delegates all copying, independent verification, restore drill,
quiescence checks, quarantine swap, and receipt publication to
``collection_runtime.modality_storage``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collection_runtime.modality_storage import (
    active_collection_dataset_roots_for_date,
    detach_modality,
    plan_modality_detach,
)


def _dataset_root(plan_path: Path, entry_index: int) -> Path:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    if entry_index < 0 or entry_index >= len(entries):
        raise ValueError(f"entry index out of range: {entry_index}")
    entry = entries[entry_index]
    if not isinstance(entry, dict) or not entry.get("target"):
        raise ValueError(f"invalid relocation plan entry: {entry_index}")
    return Path(str(entry["target"])).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--entry-index", required=True, type=int)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--exclude-active-collection-date", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-legacy-hardlink-detach",
        action="store_true",
        help="Required with --apply; only this dataset's paths are retired.",
    )
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve(strict=True)
    root = _dataset_root(plan_path, int(args.entry_index))
    cold_root = args.cold_root.expanduser().resolve(strict=True)
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    active_roots = (
        active_collection_dataset_roots_for_date(
            str(args.exclude_active_collection_date)
        )
        if args.exclude_active_collection_date
        else ()
    )
    if args.exclude_active_collection_date and not active_roots:
        raise RuntimeError("no exact active collection dataset leaf was resolved")
    proposal = plan_modality_detach(
        root,
        cold_root,
        min_free_bytes=min_free_bytes,
        allow_legacy_hardlinked_source=True,
    )
    if not args.apply:
        print(json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if proposal.ready else 2
    if not args.confirm_legacy_hardlink_detach:
        raise ValueError("--apply requires --confirm-legacy-hardlink-detach")
    if not proposal.ready:
        raise RuntimeError("detach proposal is not ready: " + "; ".join(proposal.errors))
    result = detach_modality(
        root,
        cold_root,
        expected_plan_hash=proposal.plan_hash,
        confirm_dataset_fingerprint=proposal.dataset_fingerprint,
        min_free_bytes=min_free_bytes,
        source_storage_id="hot.realman_dmtac",
        target_storage_id="nas2.tactile_archive",
        confirm_no_active_readers=True,
        allow_legacy_hardlinked_source=True,
        allow_disjoint_active_collection=bool(active_roots),
        active_collection_dataset_roots=active_roots,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
