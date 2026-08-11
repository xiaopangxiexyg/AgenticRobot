"""Audit or normalize source task IDs declared by one training profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_metadata_payloads(
    dataset_root: Path,
    *,
    task_id: str,
) -> tuple[tuple[Path, bytes, bytes], ...]:
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    info_original = info_path.read_bytes()
    tasks_original = tasks_path.read_bytes()
    info = json.loads(info_original)
    task_rows = [
        json.loads(line)
        for line in tasks_original.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not isinstance(info, dict) or not task_rows or any(
        not isinstance(row, dict) for row in task_rows
    ):
        raise ValueError(f"Invalid source task metadata: {dataset_root}.")
    if str(info.get("task_id", "")).strip() == task_id:
        info_payload = info_original
    else:
        info["task_id"] = task_id
        info_payload = (
            json.dumps(info, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
    if all(str(row.get("task_id", "")).strip() == task_id for row in task_rows):
        tasks_payload = tasks_original
    else:
        for row in task_rows:
            row["task_id"] = task_id
        tasks_payload = (
            "\n".join(json.dumps(row, ensure_ascii=False) for row in task_rows)
            + "\n"
        ).encode("utf-8")
    return (
        (info_path, info_original, info_payload),
        (tasks_path, tasks_original, tasks_payload),
    )


def normalize_profile_task_metadata(
    profile: Mapping[str, Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    expected = str(profile.get("expected_task_id", "")).strip()
    if not expected:
        raise ValueError("Profile has no expected_task_id.")
    source_values = profile.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("Profile has no source datasets.")

    updates: list[tuple[Path, bytes, bytes]] = []
    for source in source_values:
        if not isinstance(source, Mapping):
            raise ValueError("Profile source is not an object.")
        root = Path(str(source["dataset_root"])).expanduser().resolve()
        for path, original, payload in _task_metadata_payloads(
            root,
            task_id=expected,
        ):
            if original != payload:
                updates.append((path, original, payload))

    report: dict[str, Any] = {
        "expected_task_id": expected,
        "apply": bool(apply),
        "changed_file_count": len(updates),
        "changes": [
            {
                "path": str(path),
                "old_sha256": _sha256(original),
                "new_sha256": _sha256(payload),
            }
            for path, original, payload in updates
        ],
    }
    if not apply or not updates:
        return report

    prepared: list[tuple[Path, Path, bytes, int]] = []
    try:
        for path, original, payload in updates:
            if path.read_bytes() != original:
                raise RuntimeError(f"Source metadata changed during update: {path}.")
            old_stat = path.stat()
            temp = path.with_name(
                f".{path.name}.task-metadata-{os.getpid()}.tmp"
            )
            fd = os.open(
                temp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IMODE(old_stat.st_mode),
            )
            try:
                with os.fdopen(fd, "wb", closefd=True) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
            prepared.append((path, temp, original, old_stat.st_ino))

        for path, _temp, original, _old_inode in prepared:
            if path.read_bytes() != original:
                raise RuntimeError(f"Source metadata changed before commit: {path}.")
        for path, temp, _original, _old_inode in prepared:
            os.replace(temp, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        for _path, temp, _original, _old_inode in prepared:
            temp.unlink(missing_ok=True)

    report["changes"] = [
        {
            **change,
            "old_inode": old_inode,
            "new_inode": path.stat().st_ino,
        }
        for change, (path, _temp, _original, old_inode) in zip(
            report["changes"], prepared
        )
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "apply"))
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)

    from .prepared_skill_launch import (
        _load_profile,
        _sources,
        _validate_source_task_ids,
    )

    profile = _load_profile(args.profile)
    report = normalize_profile_task_metadata(
        profile,
        apply=args.command == "apply",
    )
    if args.command == "apply":
        _validate_source_task_ids(profile, _sources(profile))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
