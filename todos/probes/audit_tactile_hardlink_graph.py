"""Read-only hardlink graph audit for a relocated tactile dataset.

The probe starts from the payload inodes of one relocation-plan entry, finds
every plan-owned dataset path sharing those inodes, and reports whether each
dataset's current inventory has a matching independently verified NAS export
receipt.  It never creates inventories, receipts, locks, or transactions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from dataset_contracts.modality_artifacts import (
    ModalityArtifactInventoryV1,
    ModalityStorageTransferReceiptV1,
)
from dataset_interface.modality_artifacts import (
    PAYLOAD_HASH_PREFER_SEALED,
    build_modality_artifact_inventory,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _entries(path: Path) -> list[Path]:
    value = _object(path)
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    return [Path(str(entry["target"])).resolve(strict=False) for entry in entries]


def _cold_binding(root: Path, cold_root: Path) -> dict[str, Any]:
    try:
        inventory = build_modality_artifact_inventory(
            root,
            payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
            require_single_link=False,
        )
    except Exception as exc:
        return {"status": "source_inventory_error", "error": f"{type(exc).__name__}: {exc}"}
    artifact_root = (
        cold_root
        / inventory.source_dataset_base_fingerprint
        / inventory.semantic_content_hash
    )
    inventory_path = artifact_root / "inventory.json"
    receipt_path = artifact_root / "export_receipt.json"
    try:
        cold_inventory = ModalityArtifactInventoryV1.from_dict(_object(inventory_path))
        receipt = ModalityStorageTransferReceiptV1.from_dict(_object(receipt_path))
    except Exception as exc:
        return {
            "status": "missing_or_invalid_cold_binding",
            "artifact_root": str(artifact_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
    valid = bool(
        cold_inventory == inventory
        and receipt.status == "complete"
        and receipt.independent_copy_verified
        and receipt.inventory_hash == inventory.semantic_content_hash
        and receipt.file_count == inventory.file_count
        and receipt.total_bytes == inventory.total_bytes
    )
    return {
        "status": "verified_export_receipt" if valid else "cold_binding_mismatch",
        "artifact_root": str(artifact_root),
        "inventory_hash": inventory.semantic_content_hash,
        "payload_file_count": sum(
            artifact.role == "payload" for artifact in inventory.artifacts
        ),
        "payload_bytes": sum(
            artifact.size_bytes
            for artifact in inventory.artifacts
            if artifact.role == "payload"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--entry-index", required=True, type=int)
    parser.add_argument("--cold-root", required=True, type=Path)
    args = parser.parse_args()

    roots = _entries(args.plan.expanduser().resolve(strict=True))
    if args.entry_index < 0 or args.entry_index >= len(roots):
        raise ValueError(f"entry index out of range: {args.entry_index}")
    target = roots[args.entry_index].resolve(strict=True)
    inventory = build_modality_artifact_inventory(
        target,
        payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
        require_single_link=False,
    )
    payload_paths = tuple(
        target / artifact.logical_path
        for artifact in inventory.artifacts
        if artifact.role == "payload"
    )
    target_inodes = {
        (int(path.stat().st_dev), int(path.stat().st_ino)): int(path.stat().st_nlink)
        for path in payload_paths
    }
    matches: dict[Path, list[tuple[Path, int]]] = defaultdict(list)
    observed_counts: dict[tuple[int, int], int] = defaultdict(int)
    for root in roots:
        tactile = root / "tactile"
        if not tactile.is_dir() or root.is_symlink():
            continue
        for path in tactile.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            key = (int(stat.st_dev), int(stat.st_ino))
            if key not in target_inodes:
                continue
            observed_counts[key] += 1
            matches[root].append((path, int(stat.st_size)))

    cold_root = args.cold_root.expanduser().resolve(strict=True)
    rows = []
    for root, paths in sorted(matches.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "dataset_root": str(root),
                "shared_path_count": len(paths),
                "shared_logical_bytes": sum(size for _, size in paths),
                "cold_binding": _cold_binding(root, cold_root),
            }
        )
    uncovered = [
        {
            "device": key[0],
            "inode": key[1],
            "st_nlink": target_inodes[key],
            "plan_paths_seen": observed_counts.get(key, 0),
        }
        for key in sorted(target_inodes)
        if observed_counts.get(key, 0) != target_inodes[key]
    ]
    print(
        json.dumps(
            {
                "target_entry_index": args.entry_index,
                "target_dataset_root": str(target),
                "target_payload_file_count": len(payload_paths),
                "target_payload_bytes": sum(path.stat().st_size for path in payload_paths),
                "target_unique_inode_count": len(target_inodes),
                "graph_closed_within_plan": not uncovered,
                "uncovered_inode_count": len(uncovered),
                "uncovered_inode_samples": uncovered[:20],
                "dataset_nodes": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
