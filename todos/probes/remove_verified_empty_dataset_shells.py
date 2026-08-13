"""Audit and remove an exact allowlist of zero-Episode legacy dataset shells.

This maintenance probe deliberately refuses arbitrary paths.  Apply mode
repeats every audit immediately before removal and only accepts plain
directories with an empty Episode catalog and no dataset payload files,
symlinks, hardlinks, or open file descriptors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


LEGACY_ROOT = Path("/home/user/zc/DexTeleop/datasets")
EMPTY_SHELL_NAMES = (
    "2026-07-23_debug_dmtac_rm65_position_state_only",
    "2026-07-26_dmtac_rm65_bimanual_position_v1",
    "2026-07-31_dmtac_impedance_15_40hz",
    "2026-08-03_dmtac_rm65_bimanual_position_v2_balanced",
    "2026-08-03_dmtac_rm65_bimanual_position_v2_boundary_hold",
    "2026-08-03_dmtac_rm65_bimanual_position_v3_vive_velocity_debug",
    "2026-08-03_rm65_v2_boundary_hold_motion_audit",
    "2026-08-04_dmtac_rm65_bimanual_position_v2_balanced",
)
PAYLOAD_TOP_LEVEL_NAMES = frozenset(
    {
        "data",
        "videos",
        "tactile",
        "wrist_depth",
        "depth",
        "images",
        "episode_records",
        "session_records",
    }
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _episode_count_from_info(info: dict[str, Any]) -> int | None:
    for key in ("episode_count", "total_episodes", "num_episodes"):
        value = info.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _nonempty_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _open_fds_below(root: Path) -> list[str]:
    matches: list[str] = []
    root_prefix = str(root) + os.sep
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            descriptors = tuple(fd_root.iterdir())
        except (OSError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (OSError, PermissionError):
                continue
            normalized = target.removesuffix(" (deleted)")
            if normalized == str(root) or normalized.startswith(root_prefix):
                matches.append(f"pid={process.name}:fd={descriptor.name}:{target}")
    return sorted(matches)


def _audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    payload_files: list[str] = []
    symlinks: list[str] = []
    hardlinks: list[str] = []

    if root.parent != LEGACY_ROOT:
        errors.append("target is outside the exact legacy root")
    if root.name not in EMPTY_SHELL_NAMES:
        errors.append("target is not in the exact empty-shell allowlist")
    if root.is_symlink():
        errors.append("target is a symlink")
    if not root.is_dir():
        errors.append("target is not a directory")
        return {
            "dataset_root": str(root),
            "ready": False,
            "errors": errors,
            "files": files,
        }

    info_path = root / "meta/info.json"
    if not info_path.is_file():
        errors.append("meta/info.json is missing")
        info_episode_count = None
    else:
        try:
            info_episode_count = _episode_count_from_info(_json_object(info_path))
        except Exception as exc:
            errors.append(f"invalid meta/info.json: {type(exc).__name__}: {exc}")
            info_episode_count = None
    if info_episode_count != 0:
        errors.append(f"metadata Episode count is not zero: {info_episode_count!r}")

    catalog_path = root / "meta/episodes.jsonl"
    try:
        catalog_episode_count = _nonempty_line_count(catalog_path)
    except Exception as exc:
        errors.append(f"invalid Episode catalog: {type(exc).__name__}: {exc}")
        catalog_episode_count = None
    if catalog_episode_count != 0:
        errors.append(f"Episode catalog is not empty: {catalog_episode_count!r}")

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "link_count": int(stat.st_nlink),
            }
        )
        if relative.split("/", 1)[0] in PAYLOAD_TOP_LEVEL_NAMES:
            payload_files.append(relative)
        if stat.st_nlink > 1:
            hardlinks.append(relative)

    if payload_files:
        errors.append(f"dataset payload files exist: {payload_files[:10]!r}")
    if symlinks:
        errors.append(f"symlinks exist below target: {symlinks[:10]!r}")
    if hardlinks:
        errors.append(f"hardlinked files exist below target: {hardlinks[:10]!r}")
    open_fds = _open_fds_below(root)
    if open_fds:
        errors.append(f"open file descriptors exist below target: {open_fds[:10]!r}")

    return {
        "dataset_root": str(root),
        "ready": not errors,
        "metadata_episode_count": info_episode_count,
        "catalog_episode_count": catalog_episode_count,
        "file_count": len(files),
        "total_file_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "errors": errors,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-remove-empty-shells", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.confirm_remove_empty_shells:
        raise ValueError("--apply requires --confirm-remove-empty-shells")

    roots = tuple(LEGACY_ROOT / name for name in EMPTY_SHELL_NAMES)
    audits = [_audit(root) for root in roots]
    payload: dict[str, Any] = {
        "operation": "remove_verified_empty_dataset_shells",
        "apply": bool(args.apply),
        "all_ready": all(bool(row["ready"]) for row in audits),
        "rows": audits,
    }
    if not payload["all_ready"]:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    if not args.apply:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    removed: list[str] = []
    for root in roots:
        fresh = _audit(root)
        if not fresh["ready"]:
            raise RuntimeError(
                "empty-shell target changed before removal: "
                + json.dumps(fresh, ensure_ascii=False, sort_keys=True)
            )
        shutil.rmtree(root)
        if root.exists() or root.is_symlink():
            raise RuntimeError(f"empty-shell removal did not complete: {root}")
        _fsync_directory(LEGACY_ROOT)
        removed.append(str(root))
    payload["removed"] = removed
    payload["postcondition_all_absent"] = all(not root.exists() for root in roots)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
