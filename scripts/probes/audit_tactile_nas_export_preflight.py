"""Read-only preflight for exporting canonical tactile artifacts to NAS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def _mount_for(path: Path) -> dict[str, object] | None:
    resolved = path.resolve(strict=False)
    rows: list[tuple[int, str, str, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        separator = fields.index("-")
        mount_point = fields[4].replace("\\040", " ")
        prefix = mount_point.rstrip("/") + "/"
        if str(resolved) == mount_point or str(resolved).startswith(prefix):
            rows.append(
                (
                    len(mount_point),
                    mount_point,
                    fields[separator + 1],
                    fields[separator + 2],
                )
            )
    if not rows:
        return None
    _, mount_point, fs_type, source = max(rows)
    return {"mount_point": mount_point, "fs_type": fs_type, "source": source}


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def _active_processes() -> list[dict[str, object]]:
    needles = (
        "run_vive_rm65_real_teleop",
        "teleop_runtime.collection",
        "tactile_compact",
        "modality-detach",
        "modality-move",
        "modality-rehydrate",
    )
    matches = []
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            cmdline = (process_root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if any(needle in cmdline for needle in needles):
            matches.append({"pid": int(process_root.name), "cmdline": cmdline[:1000]})
    return sorted(matches, key=lambda row: int(row["pid"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--nas-root", required=True, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    plan_path = args.plan.expanduser().resolve(strict=True)
    nas_root = args.nas_root.expanduser().resolve(strict=False)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")

    rows = []
    total_bytes = 0
    for entry in entries:
        root = Path(str(entry["target"])).expanduser().resolve(strict=False)
        tactile = root / "tactile"
        file_count = 0
        byte_count = 0
        hardlinked_files = 0
        symlink_count = 0
        if tactile.is_dir():
            for path in tactile.rglob("*"):
                if path.is_symlink():
                    symlink_count += 1
                elif path.is_file():
                    stat = path.stat()
                    file_count += 1
                    byte_count += stat.st_size
                    hardlinked_files += int(stat.st_nlink > 1)
        total_bytes += byte_count
        rows.append(
            {
                "dataset_root": str(root),
                "exists": root.is_dir(),
                "tactile_exists": tactile.is_dir(),
                "tactile_files": file_count,
                "tactile_bytes": byte_count,
                "hardlinked_files": hardlinked_files,
                "symlinks": symlink_count,
                "active_transaction": (
                    root / "meta/modality_artifacts/tactile/active_transaction.json"
                ).is_file(),
            }
        )

    nas_parent = nas_root.parent
    payload = {
        "source_root": str(source_root),
        "source_stat_dev": source_root.stat().st_dev,
        "source_mount": _mount_for(source_root),
        "source_disk": _disk(source_root),
        "nas_root": str(nas_root),
        "nas_root_exists": nas_root.exists(),
        "nas_parent_exists": nas_parent.is_dir(),
        "nas_parent_writable": os.access(nas_parent, os.W_OK),
        "nas_parent_stat_dev": nas_parent.stat().st_dev if nas_parent.exists() else None,
        "nas_mount": _mount_for(nas_parent),
        "nas_disk": _disk(nas_parent) if nas_parent.exists() else None,
        "same_device": (
            source_root.stat().st_dev == nas_parent.stat().st_dev
            if nas_parent.exists()
            else None
        ),
        "dataset_count": len(rows),
        "tactile_dataset_count": sum(bool(row["tactile_exists"]) for row in rows),
        "tactile_total_bytes": total_bytes,
        "hardlinked_file_count": sum(int(row["hardlinked_files"]) for row in rows),
        "symlink_count": sum(int(row["symlinks"]) for row in rows),
        "pending_transaction_count": sum(
            bool(row["active_transaction"]) for row in rows
        ),
        "active_processes": _active_processes(),
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
