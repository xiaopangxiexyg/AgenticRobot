"""Plan or run a fail-closed batch of verified tactile recompressions.

This is an offline maintenance probe, not a collection entrypoint.  Apply mode
requires the caller to echo the SHA-256 of the exact read-only plan so that a
new or changed dataset cannot silently enter the batch between review and use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from collection_runtime.tactile_recompression import (
    _active_collection_processes,
    _finalized_manifest_paths,
    _raw_manifest_streams,
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)


def _dataset_roots(scan_root: Path) -> list[Path]:
    return sorted(
        {
            path.parent.parent.resolve()
            for path in scan_root.rglob("meta/episodes.jsonl")
            if path.is_file()
        }
    )


def _safety_counts(dataset_root: Path) -> tuple[int, int]:
    external_hardlinks = 0
    for manifest_path in _finalized_manifest_paths(dataset_root):
        _, raw_streams = _raw_manifest_streams(dataset_root, manifest_path)
        external_hardlinks += sum(
            1 for _, _, source in raw_streams if source.stat().st_nlink > 1
        )
    committed_v24 = len(
        list(
            (dataset_root / "meta" / "v24" / "control_trace" / "episodes").glob(
                "*/capture_commit.cjson"
            )
        )
    )
    return external_hardlinks, committed_v24


def _build_plan(
    scan_root: Path,
    *,
    entry_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    scan_root = scan_root.expanduser().resolve()
    datasets: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for dataset_root in _dataset_roots(scan_root):
        try:
            plan = plan_tactile_zstd_recompression(dataset_root, min_free_bytes=0)
            external_hardlinks, committed_v24 = _safety_counts(dataset_root)
        except (FileNotFoundError, ValueError) as exc:
            rejected.append(
                {
                    "dataset_root": str(dataset_root),
                    "reason": str(exc),
                }
            )
            continue
        if plan.stream_count <= 0:
            continue
        datasets.append(
            {
                "dataset_root": str(dataset_root),
                "stream_count": int(plan.stream_count),
                "raw_bytes": int(plan.raw_bytes),
                "manifest_count": int(plan.manifest_count),
                "external_hardlink_stream_count": int(external_hardlinks),
                "committed_v24_capture_count": int(committed_v24),
            }
        )
    available_dataset_count = len(datasets)
    selected_entry_indices = tuple(sorted(set(entry_indices)))
    if len(selected_entry_indices) != len(entry_indices):
        raise ValueError("--entry-index values must be unique")
    invalid_indices = [
        index
        for index in selected_entry_indices
        if index < 0 or index >= available_dataset_count
    ]
    if invalid_indices:
        raise ValueError(
            "--entry-index is outside the eligible dataset plan: "
            f"invalid={invalid_indices} available=0..{available_dataset_count - 1}"
        )
    if selected_entry_indices:
        datasets = [datasets[index] for index in selected_entry_indices]
        canonical_payload: Any = {
            "entry_indices": list(selected_entry_indices),
            "datasets": datasets,
        }
    else:
        canonical_payload = datasets
    canonical = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "scan_root": str(scan_root),
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "available_dataset_count": available_dataset_count,
        "selected_entry_indices": list(selected_entry_indices),
        "dataset_count": len(datasets),
        "stream_count": sum(row["stream_count"] for row in datasets),
        "raw_bytes": sum(row["raw_bytes"] for row in datasets),
        "datasets": datasets,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument(
        "--entry-index",
        action="append",
        type=int,
        default=[],
        help=(
            "Restrict the frozen plan to a zero-based eligible dataset index from an "
            "unfiltered plan. Repeat to select multiple entries. Selection is included "
            "in the plan SHA-256."
        ),
    )
    args = parser.parse_args()

    entry_indices = tuple(args.entry_index)
    plan = _build_plan(args.scan_root, entry_indices=entry_indices)
    print(json.dumps({"phase": "plan", **plan}, ensure_ascii=False, sort_keys=True))
    if not args.apply:
        return
    expected = args.expected_plan_sha256.strip().lower()
    if not expected:
        raise RuntimeError("Apply mode requires --expected-plan-sha256.")
    if expected != plan["plan_sha256"]:
        raise RuntimeError(
            "Tactile batch plan changed; refusing apply: "
            f"expected={expected} actual={plan['plan_sha256']}"
        )
    active = _active_collection_processes()
    if active:
        raise RuntimeError(
            "Formal collection is active; refusing tactile batch: "
            + "; ".join(active[:3])
        )
    for index, row in enumerate(plan["datasets"], start=1):
        if row["external_hardlink_stream_count"]:
            raise RuntimeError(
                "Refusing source with external hardlinks: " + row["dataset_root"]
            )
        if row["committed_v24_capture_count"]:
            raise RuntimeError(
                "Refusing committed v2.4 capture: " + row["dataset_root"]
            )
        print(
            json.dumps(
                {
                    "phase": "dataset_start",
                    "index": index,
                    "dataset_count": plan["dataset_count"],
                    **row,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        result = recompress_tactile_zstd_in_place(
            row["dataset_root"],
            level=args.level,
            chunk_frames=args.chunk_frames,
            delete_originals=True,
        )
        print(
            json.dumps(
                {"phase": "dataset_complete", **result.to_dict()},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    final_plan = _build_plan(args.scan_root)
    print(
        json.dumps(
            {"phase": "complete", "remaining": final_plan},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
