"""Read-only inventory for finalized tactile streams under dataset roots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from utils.dataset_postprocess.tactile_recompress import (
    _active_collection_processes,
    _finalized_manifest_paths,
    _raw_manifest_streams,
    plan_tactile_zstd_recompression,
)


def _dataset_roots(scan_root: Path) -> list[Path]:
    candidates = {
        path.parent.parent.resolve()
        for path in scan_root.rglob("meta/episodes.jsonl")
        if path.is_file()
    }
    return sorted(candidates)


def _audit_dataset(root: Path) -> dict[str, Any]:
    try:
        plan = plan_tactile_zstd_recompression(root, min_free_bytes=0)
        manifests = _finalized_manifest_paths(root)
    except (FileNotFoundError, ValueError) as exc:
        return {"dataset_root": str(root), "status": "not_applicable", "reason": str(exc)}

    raw_streams: list[dict[str, Any]] = []
    inode_owners: dict[tuple[int, int], list[str]] = {}
    compressed_stream_count = 0
    for manifest_path in manifests:
        manifest, raw = _raw_manifest_streams(root, manifest_path)
        compressed_stream_count += len(manifest.get("streams", {})) - len(raw)
        for sensor_key, spec, source in raw:
            stat = source.stat()
            relative = source.relative_to(root).as_posix()
            inode_owners.setdefault((int(stat.st_dev), int(stat.st_ino)), []).append(relative)
            raw_streams.append(
                {
                    "manifest": manifest_path.relative_to(root).as_posix(),
                    "sensor_key": sensor_key,
                    "path": relative,
                    "bytes": int(stat.st_size),
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "link_count": int(stat.st_nlink),
                    "encoding": str(spec.get("encoding") or ""),
                    "compression": str(spec.get("compression") or ""),
                }
            )

    shared_inodes = [
        {"device": dev, "inode": ino, "paths": paths}
        for (dev, ino), paths in sorted(inode_owners.items())
        if len(paths) > 1
    ]
    external_hardlink_streams = [row for row in raw_streams if row["link_count"] > 1]
    committed_v24 = sorted(
        (root / "meta" / "v24" / "control_trace" / "episodes").glob(
            "*/capture_commit.cjson"
        )
    )
    return {
        "dataset_root": str(root),
        "status": "raw_tactile" if plan.stream_count else "already_compressed_or_empty",
        "plan": plan.to_dict(),
        "compressed_stream_count": compressed_stream_count,
        "raw_streams": raw_streams,
        "shared_inode_groups": shared_inodes,
        "external_hardlink_stream_count": len(external_hardlink_streams),
        "committed_v24_capture_count": len(committed_v24),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_roots", nargs="+", type=Path)
    parser.add_argument("--include-streams", action="store_true")
    args = parser.parse_args()

    roots: set[Path] = set()
    for value in args.scan_roots:
        scan_root = value.expanduser().resolve()
        if scan_root.is_dir():
            roots.update(_dataset_roots(scan_root))

    datasets = [_audit_dataset(root) for root in sorted(roots)]
    if not args.include_streams:
        for dataset in datasets:
            dataset.pop("raw_streams", None)
    tactile_datasets = [row for row in datasets if row.get("status") == "raw_tactile"]
    free_by_device: dict[str, dict[str, int]] = {}
    for row in tactile_datasets:
        root = Path(str(row["dataset_root"]))
        stat = root.stat()
        usage = shutil.disk_usage(root)
        free_by_device.setdefault(
            str(stat.st_dev),
            {"device": int(stat.st_dev), "free_bytes": int(usage.free)},
        )
    payload = {
        "scan_roots": [str(path.expanduser().resolve()) for path in args.scan_roots],
        "active_collection_processes": list(_active_collection_processes()),
        "dataset_count": len(datasets),
        "raw_tactile_dataset_count": len(tactile_datasets),
        "raw_tactile_bytes": sum(
            int(row["plan"]["raw_bytes"]) for row in tactile_datasets
        ),
        "raw_tactile_stream_count": sum(
            int(row["plan"]["stream_count"]) for row in tactile_datasets
        ),
        "free_by_device": list(free_by_device.values()),
        "datasets": datasets,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
