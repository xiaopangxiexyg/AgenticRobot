"""Read-only tactile compression and detachable-artifact readiness audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from collection_runtime.modality_inventory import modality_inventory_path
from collection_runtime.tactile_recompression import plan_tactile_zstd_recompression
from dataset_interface.modality_artifacts import (
    PAYLOAD_HASH_REQUIRE_SEALED,
    build_modality_artifact_inventory,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _audit(root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_root": str(root),
        "compression_status": "unknown",
        "raw_stream_count": None,
        "compressed_stream_count": None,
        "detach_preflight_ready": False,
        "current_inventory_published": False,
        "inventory_hash": None,
        "pending_storage_transaction": (
            root / "meta/modality_artifacts/tactile/active_transaction.json"
        ).is_file(),
        "errors": [],
    }
    try:
        plan = plan_tactile_zstd_recompression(root, min_free_bytes=0)
        row["raw_stream_count"] = int(plan.stream_count)
        row["compressed_stream_count"] = int(plan.skipped_compressed_streams)
        row["compression_status"] = (
            "compressed" if plan.stream_count == 0 else "raw_or_mixed"
        )
    except Exception as exc:
        row["compression_status"] = "invalid_or_unsupported"
        row["errors"].append(f"compression: {type(exc).__name__}: {exc}")

    try:
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=PAYLOAD_HASH_REQUIRE_SEALED,
        )
        row["detach_preflight_ready"] = True
        row["inventory_hash"] = inventory.semantic_content_hash
        published = modality_inventory_path(root, inventory.semantic_content_hash)
        if published.is_file():
            row["current_inventory_published"] = (
                _read_object(published) == inventory.to_dict()
            )
            if not row["current_inventory_published"]:
                row["errors"].append("published current inventory differs")
    except Exception as exc:
        row["errors"].append(f"detach: {type(exc).__name__}: {exc}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = _read_object(args.plan.expanduser().resolve())
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    rows = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("relocation plan entry must be an object")
        rows.append(_audit(Path(str(entry["target"])).resolve(strict=True)))
    compression = Counter(str(row["compression_status"]) for row in rows)
    payload = {
        "dataset_count": len(rows),
        "compression_status_counts": dict(sorted(compression.items())),
        "raw_stream_count": sum(
            int(row["raw_stream_count"] or 0) for row in rows
        ),
        "compressed_stream_count": sum(
            int(row["compressed_stream_count"] or 0) for row in rows
        ),
        "detach_preflight_ready_count": sum(
            bool(row["detach_preflight_ready"]) for row in rows
        ),
        "current_inventory_published_count": sum(
            bool(row["current_inventory_published"]) for row in rows
        ),
        "pending_storage_transaction_count": sum(
            bool(row["pending_storage_transaction"]) for row in rows
        ),
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
