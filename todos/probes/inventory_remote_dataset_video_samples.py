"""Read-only inventory and deterministic video sampling for dataset handoff.

This probe deliberately never writes below the scanned dataset root.  It emits
one JSON manifest on stdout for later retrieval through ``codex_remote``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return tasks
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            tasks.append(value)
    return tasks


def _dataset_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if not Path(current, name).is_symlink()
        )
        current_path = Path(current)
        if current_path.name == "meta" and "info.json" in files:
            roots.append(current_path.parent)
            directories[:] = []
    return sorted(set(roots))


def _videos(dataset_root: Path) -> list[Path]:
    videos_root = dataset_root / "videos"
    if not videos_root.is_dir() or videos_root.is_symlink():
        return []
    candidates: list[Path] = []
    for current, directories, files in os.walk(videos_root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if not Path(current, name).is_symlink()
        )
        for name in sorted(files):
            path = Path(current, name)
            if path.suffix.lower() not in VIDEO_SUFFIXES or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > 0:
                candidates.append(path)
    return candidates


def _front_videos(videos: list[Path]) -> list[Path]:
    front = [
        path
        for path in videos
        if any(
            part in {"front", "observation.images.front"}
            for part in path.parts
        )
    ]
    return front or videos


def _namespace(root: Path, dataset_root: Path) -> str:
    relative = dataset_root.relative_to(root)
    parts = relative.parts[:-1]
    return "__".join(parts) if parts else "root"


def _safe_name(value: str) -> str:
    clean = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value.strip()
    ).strip("._")
    return clean or "dataset"


def build_manifest(root: Path, *, seed: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for dataset_root in _dataset_roots(root):
        info = _read_json(dataset_root / "meta" / "info.json")
        videos = _videos(dataset_root)
        episode_count = int(info.get("episode_count") or 0)
        if episode_count <= 0 or not videos:
            skipped.append(
                {
                    "path": str(dataset_root),
                    "reason": (
                        "no_episodes" if episode_count <= 0 else "no_nonempty_video"
                    ),
                }
            )
            continue
        dataset_name = str(info.get("dataset_name") or dataset_root.name).strip()
        tasks = _read_tasks(dataset_root / "meta" / "tasks.jsonl")
        task_id = str(info.get("task_id") or "").strip()
        task_description = str(info.get("task_description") or "").strip()
        if tasks:
            task_id = str(tasks[0].get("task_id") or task_id).strip()
            task_description = str(
                tasks[0].get("description") or task_description
            ).strip()
        selectable = _front_videos(videos)
        path_seed = hashlib.sha256(
            f"{seed}\0{dataset_root}".encode("utf-8")
        ).digest()
        selected = random.Random(path_seed).choice(selectable)
        selected_stat = selected.stat()
        info_stat = (dataset_root / "meta" / "info.json").stat()
        records.append(
            {
                "dataset_name": dataset_name,
                "dataset_directory_name": dataset_root.name,
                "dataset_root": str(dataset_root),
                "namespace": _namespace(root, dataset_root),
                "schema": str(info.get("schema") or info.get("format") or ""),
                "episode_count": episode_count,
                "frame_count": int(info.get("frame_count") or 0),
                "task_id": task_id,
                "task_description": task_description,
                "task_records": tasks,
                "video_count": len(videos),
                "selection_pool": "front" if selectable != videos else "all",
                "selected_video": str(selected),
                "selected_video_relative": str(selected.relative_to(dataset_root)),
                "selected_video_size_bytes": selected_stat.st_size,
                "selected_video_device": selected_stat.st_dev,
                "selected_video_inode": selected_stat.st_ino,
                "selected_video_nlink": selected_stat.st_nlink,
                "info_device": info_stat.st_dev,
                "info_inode": info_stat.st_ino,
                "info_nlink": info_stat.st_nlink,
            }
        )

    name_counts: dict[str, int] = {}
    for record in records:
        directory_name = record["dataset_directory_name"]
        name_counts[directory_name] = (
            name_counts.get(directory_name, 0) + 1
        )
    for record in records:
        directory_name = record["dataset_directory_name"]
        stem = _safe_name(directory_name)
        if name_counts[directory_name] > 1:
            stem += "__" + _safe_name(record["namespace"])
        suffix = Path(record["selected_video"]).suffix.lower()
        record["local_filename"] = stem + suffix

    inode_groups: dict[tuple[int, int], list[str]] = {}
    for record in records:
        identity = (
            int(record["selected_video_device"]),
            int(record["selected_video_inode"]),
        )
        inode_groups.setdefault(identity, []).append(record["dataset_root"])
    shared = [
        {
            "device": identity[0],
            "inode": identity[1],
            "dataset_roots": paths,
        }
        for identity, paths in sorted(inode_groups.items())
        if len(paths) > 1
    ]
    return {
        "schema": "dexteleop_remote_dataset_video_sample_manifest_v1",
        "scan_root": str(root),
        "selection_seed": seed,
        "dataset_count": len(records),
        "skipped_count": len(skipped),
        "total_selected_bytes": sum(
            int(record["selected_video_size_bytes"]) for record in records
        ),
        "datasets": records,
        "shared_selected_video_inode_groups": shared,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", default="dexteleop-2026-08-12")
    args = parser.parse_args()
    print(
        json.dumps(
            build_manifest(args.root, seed=str(args.seed)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
