"""Phase-A copy-only archive primitives for legacy runtime-artifact roots.

This module never deletes, truncates, renames, or rewrites a source.  A copy is
assembled under a fresh partial directory, verified file-by-file, and then
published by one same-filesystem directory rename.  Project-local ``legacy/``
storage is treated as an archive, not as an independent backup device.

This module freezes one or more complete old roots for rollback.  It does not
classify their contents or publish them into the canonical semantic layout;
``scripts/runtime_artifact_migration.py`` owns that later, per-family phase.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


MIGRATION_INVENTORY_SCHEMA = "dexteleop.runtime_artifact_inventory.v1"
MIGRATION_ARCHIVE_SCHEMA = "dexteleop.runtime_artifact_archive.v1"
MIGRATION_VERIFY_SCHEMA = "dexteleop.runtime_artifact_archive_verify.v1"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class MigrationSafetyError(RuntimeError):
    """Raised when a copy cannot preserve the migration safety contract."""


def _safe_component(value: str, *, field: str) -> str:
    text = str(value)
    if (
        not text
        or text != text.strip()
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
    ):
        raise ValueError(f"{field} must be one non-empty safe path component")
    return text


@dataclass(frozen=True)
class MigrationSource:
    label: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _safe_component(self.label, field="label"))
        root = Path(self.root).expanduser()
        object.__setattr__(self, "root", Path(os.path.abspath(os.fspath(root))))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _allocated_bytes(metadata: os.stat_result) -> int:
    return int(getattr(metadata, "st_blocks", 0)) * 512


def _entry_kind(metadata: os.stat_result) -> str:
    mode = metadata.st_mode
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _iter_tree(root: Path) -> Iterable[Path]:
    for current_text, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        traversed_directories: list[str] = []
        for name in directory_names:
            path = current / name
            yield path
            if not path.is_symlink():
                traversed_directories.append(name)
        directory_names[:] = traversed_directories
        for name in file_names:
            yield current / name


def _snapshot_entry(
    path: Path,
    *,
    root: Path,
    include_hash: bool,
) -> dict[str, Any]:
    metadata = path.lstat()
    kind = _entry_kind(metadata)
    relative = path.relative_to(root).as_posix()
    entry: dict[str, Any] = {
        "relative_path": relative,
        "kind": kind,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "link_count": int(metadata.st_nlink),
        "mode": stat.S_IMODE(metadata.st_mode),
        "size_bytes": int(metadata.st_size),
        "allocated_bytes": _allocated_bytes(metadata),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }
    if kind == "file":
        entry["sparse"] = bool(
            int(metadata.st_size) > 0
            and _allocated_bytes(metadata) < int(metadata.st_size)
        )
        if include_hash:
            entry["sha256"] = _sha256_file(path)
    elif kind == "symlink":
        target = os.readlink(path)
        entry["symlink_target"] = target
        entry["symlink_target_sha256"] = hashlib.sha256(
            target.encode("utf-8", errors="surrogateescape")
        ).hexdigest()
    return entry


def _recovery_hazard(path: Path) -> dict[str, str] | None:
    if not path.name.endswith(".recovery.json") or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed recovery state is unsafe.
        return {
            "path": str(path),
            "status": "unreadable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "status": "invalid",
            "reason": "recovery payload is not a JSON object",
        }
    status_text = str(payload.get("status", "") or "").strip()
    if status_text == "restored":
        return None
    return {
        "path": str(path),
        "status": status_text or "missing",
        "reason": "recovery journal is not durably marked restored",
    }


def _validate_sources(sources: Sequence[MigrationSource]) -> tuple[MigrationSource, ...]:
    if not sources:
        raise ValueError("at least one migration source is required")
    normalized = tuple(sources)
    labels = [source.label for source in normalized]
    if len(set(labels)) != len(labels):
        raise ValueError(f"migration source labels must be unique: {labels}")
    roots = [source.root for source in normalized]
    if len(set(roots)) != len(roots):
        raise ValueError("migration source roots must be unique")
    for source in normalized:
        if source.root.is_symlink():
            raise MigrationSafetyError(
                f"migration source root must not be a symlink: {source.root}"
            )
        if not source.root.is_dir():
            raise FileNotFoundError(f"migration source root is not a directory: {source.root}")
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise MigrationSafetyError(
                    f"migration source roots must not overlap: {left} and {right}"
                )
    return normalized


def inventory_sources(
    sources: Sequence[MigrationSource],
    *,
    include_hashes: bool = False,
    observed_at: float | None = None,
) -> dict[str, Any]:
    normalized = _validate_sources(sources)
    source_reports: list[dict[str, Any]] = []
    hazards: list[dict[str, str]] = []
    for source in normalized:
        root_metadata = source.root.stat()
        entries: list[dict[str, Any]] = []
        logical_bytes = 0
        allocated_bytes = 0
        newest_mtime_ns = int(root_metadata.st_mtime_ns)
        counts = {kind: 0 for kind in ("file", "directory", "symlink", "special")}
        for path in _iter_tree(source.root):
            entry = _snapshot_entry(
                path,
                root=source.root,
                include_hash=include_hashes,
            )
            entries.append(entry)
            kind = str(entry["kind"])
            counts[kind] = counts.get(kind, 0) + 1
            newest_mtime_ns = max(newest_mtime_ns, int(entry["mtime_ns"]))
            if kind == "file":
                logical_bytes += int(entry["size_bytes"])
                allocated_bytes += int(entry["allocated_bytes"])
                hazard = _recovery_hazard(path)
                if hazard is not None:
                    hazards.append({"source_label": source.label, **hazard})
        source_reports.append(
            {
                "label": source.label,
                "root": str(source.root),
                "root_device": int(root_metadata.st_dev),
                "root_inode": int(root_metadata.st_ino),
                "root_link_count": int(root_metadata.st_nlink),
                "root_mode": stat.S_IMODE(root_metadata.st_mode),
                "root_mtime_ns": int(root_metadata.st_mtime_ns),
                "root_ctime_ns": int(root_metadata.st_ctime_ns),
                "newest_mtime_ns": newest_mtime_ns,
                "counts": counts,
                "logical_bytes": logical_bytes,
                "allocated_bytes": allocated_bytes,
                "entries": entries,
            }
        )
    content = {
        "schema": MIGRATION_INVENTORY_SCHEMA,
        "include_hashes": bool(include_hashes),
        "sources": source_reports,
        "recovery_hazards": hazards,
    }
    return {
        **content,
        "observed_at": time.time() if observed_at is None else float(observed_at),
        "inventory_sha256": _json_sha256(content),
    }


def _validate_archive_target(
    *,
    sources: Sequence[MigrationSource],
    archive_base: Path,
    migration_id: str,
) -> tuple[Path, Path]:
    migration_name = _safe_component(migration_id, field="migration_id")
    base = Path(archive_base).expanduser().resolve()
    final = base / migration_name
    partial = base / f".{migration_name}.partial"
    for source in sources:
        if (
            base == source.root
            or base.is_relative_to(source.root)
            or source.root.is_relative_to(base)
        ):
            raise MigrationSafetyError(
                "archive base and source roots must be disjoint: "
                f"source={source.root} archive={base}"
            )
    if final.exists() or partial.exists():
        raise FileExistsError(
            f"fresh archive targets are required: final={final} partial={partial}"
        )
    return final, partial


def plan_archive_copy(
    sources: Sequence[MigrationSource],
    *,
    archive_base: Path,
    migration_id: str,
    quiescence_sec: float = 300.0,
    now: float | None = None,
) -> dict[str, Any]:
    normalized = _validate_sources(sources)
    final, partial = _validate_archive_target(
        sources=normalized,
        archive_base=archive_base,
        migration_id=migration_id,
    )
    current_time = time.time() if now is None else float(now)
    inventory = inventory_sources(normalized, observed_at=current_time)
    special_entries = [
        {
            "source_label": report["label"],
            "relative_path": entry["relative_path"],
        }
        for report in inventory["sources"]
        for entry in report["entries"]
        if entry["kind"] == "special"
    ]
    recent_entries = [
        {
            "source_label": report["label"],
            "relative_path": entry["relative_path"],
            "mtime_ns": entry["mtime_ns"],
            "ctime_ns": entry["ctime_ns"],
        }
        for report in inventory["sources"]
        for entry in report["entries"]
        if (
            current_time
            - max(int(entry["mtime_ns"]), int(entry["ctime_ns"]))
            / 1_000_000_000
            < float(quiescence_sec)
        )
    ]
    blockers: list[dict[str, Any]] = []
    if inventory["recovery_hazards"]:
        blockers.append(
            {
                "code": "unfinished_recovery_journal",
                "items": inventory["recovery_hazards"],
            }
        )
    if special_entries:
        blockers.append({"code": "special_files_present", "items": special_entries})
    if recent_entries:
        blockers.append(
            {
                "code": "source_not_quiescent",
                "quiescence_sec": float(quiescence_sec),
                "items": recent_entries,
            }
        )
    allocated = sum(
        int(report["allocated_bytes"]) for report in inventory["sources"]
    )
    metadata_allowance = sum(
        sum(int(value) for value in report["counts"].values()) * 4096
        for report in inventory["sources"]
    )
    estimated_copy_bytes = allocated + metadata_allowance
    return {
        "schema": MIGRATION_ARCHIVE_SCHEMA,
        "mode": "copy_plan",
        "migration_id": _safe_component(migration_id, field="migration_id"),
        "archive_base": str(Path(archive_base).expanduser().resolve()),
        "final_archive_root": str(final),
        "partial_archive_root": str(partial),
        "quiescence_sec": float(quiescence_sec),
        "estimated_copy_bytes": estimated_copy_bytes,
        "inventory": inventory,
        "blockers": blockers,
        "copy_allowed": not blockers,
        "deletion_supported": False,
        "project_local_archive_is_independent_backup": False,
    }


def _same_source_snapshot(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    ) == (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )


def _copy_regular_file(source: Path, destination: Path) -> None:
    completed = subprocess.run(
        [
            "cp",
            "--sparse=always",
            "--reflink=auto",
            "--preserve=mode,timestamps,xattr",
            "--no-preserve=ownership",
            "--",
            str(source),
            str(destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise MigrationSafetyError(
            f"sparse-preserving copy failed for {source}: {detail}"
        )
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MigrationSafetyError(
            "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise MigrationSafetyError(f"archive target appeared before publication: {target}")
    raise MigrationSafetyError(
        f"atomic no-replace publication failed: {os.strerror(error_number)}"
    )


def _write_json_durable(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def copy_sources_to_archive(
    sources: Sequence[MigrationSource],
    *,
    archive_base: Path,
    migration_id: str,
    quiescence_sec: float = 300.0,
    now: float | None = None,
) -> Path:
    """Copy and verify sources, then publish a fresh archive; never delete."""

    normalized = _validate_sources(sources)
    plan = plan_archive_copy(
        normalized,
        archive_base=archive_base,
        migration_id=migration_id,
        quiescence_sec=quiescence_sec,
        now=now,
    )
    if plan["blockers"]:
        raise MigrationSafetyError(
            "archive copy blocked by preflight: "
            + json.dumps(plan["blockers"], ensure_ascii=False, sort_keys=True)
        )
    final = Path(plan["final_archive_root"])
    partial = Path(plan["partial_archive_root"])
    base = Path(plan["archive_base"])
    base.mkdir(parents=True, exist_ok=True)
    archive_device = int(base.stat().st_dev)
    free_bytes = shutil.disk_usage(base).free
    required_bytes = int(plan["estimated_copy_bytes"])
    margin_bytes = max(256 << 20, required_bytes // 10)
    if free_bytes < required_bytes + margin_bytes:
        raise MigrationSafetyError(
            "insufficient archive capacity: "
            f"free={free_bytes} required={required_bytes} margin={margin_bytes}"
        )
    partial.mkdir(exist_ok=False)
    payload_root = partial / "payload"
    payload_root.mkdir()
    file_receipts: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    relocation_mappings: list[dict[str, str]] = []
    destination_hardlinks: dict[tuple[int, int], Path] = {}

    for source in normalized:
        destination_root = payload_root / source.label
        destination_root.mkdir()
        source_root_metadata = source.root.stat()
        directory_metadata: list[tuple[Path, os.stat_result]] = [
            (destination_root, source_root_metadata)
        ]
        for path in _iter_tree(source.root):
            relative = path.relative_to(source.root)
            destination = destination_root / relative
            before = path.lstat()
            kind = _entry_kind(before)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if kind == "directory":
                destination.mkdir(exist_ok=False)
                directory_metadata.append((destination, before))
                continue
            if kind == "symlink":
                target = os.readlink(path)
                os.symlink(target, destination)
                after = path.lstat()
                if not _same_source_snapshot(before, after):
                    raise MigrationSafetyError(f"source symlink changed during copy: {path}")
                copied = destination.lstat()
                if not stat.S_ISLNK(copied.st_mode) or os.readlink(destination) != target:
                    raise MigrationSafetyError(f"symlink verification failed: {destination}")
                file_receipts.append(
                    {
                        "source_label": source.label,
                        "relative_path": relative.as_posix(),
                        "kind": "symlink",
                        "symlink_target": target,
                        "source_device": int(before.st_dev),
                        "source_inode": int(before.st_ino),
                        "destination_device": int(copied.st_dev),
                        "destination_inode": int(copied.st_ino),
                    }
                )
                continue
            if kind != "file":
                raise MigrationSafetyError(f"special source file is not copyable: {path}")

            source_hash = _sha256_file(path)
            hardlink_key = (int(before.st_dev), int(before.st_ino))
            existing_destination = destination_hardlinks.get(hardlink_key)
            if existing_destination is None:
                _copy_regular_file(path, destination)
                destination_hardlinks[hardlink_key] = destination
            else:
                os.link(existing_destination, destination)
            copied = destination.stat()
            destination_hash = _sha256_file(destination)
            after = path.lstat()
            if not _same_source_snapshot(before, after):
                raise MigrationSafetyError(f"source file changed during copy: {path}")
            if source_hash != destination_hash or int(before.st_size) != int(copied.st_size):
                raise MigrationSafetyError(f"copied file verification failed: {path}")
            if (
                int(before.st_dev) == int(copied.st_dev)
                and int(before.st_ino) == int(copied.st_ino)
            ):
                raise MigrationSafetyError(f"archive file aliases source inode: {path}")
            file_receipts.append(
                {
                    "source_label": source.label,
                    "relative_path": relative.as_posix(),
                    "kind": "file",
                    "size_bytes": int(before.st_size),
                    "allocated_bytes_source": _allocated_bytes(before),
                    "allocated_bytes_destination": _allocated_bytes(copied),
                    "sparse_source": bool(
                        int(before.st_size) > 0
                        and _allocated_bytes(before) < int(before.st_size)
                    ),
                    "sha256": source_hash,
                    "source_device": int(before.st_dev),
                    "source_inode": int(before.st_ino),
                    "source_link_count": int(before.st_nlink),
                    "source_mode": stat.S_IMODE(before.st_mode),
                    "source_mtime_ns": int(before.st_mtime_ns),
                    "source_ctime_ns": int(before.st_ctime_ns),
                    "destination_device": int(copied.st_dev),
                    "destination_inode": int(copied.st_ino),
                    "destination_link_count": int(copied.st_nlink),
                    "hardlink_group": f"{before.st_dev}:{before.st_ino}",
                }
            )

        for directory, metadata in reversed(directory_metadata):
            os.chmod(directory, stat.S_IMODE(metadata.st_mode))
            os.utime(
                directory,
                ns=(int(metadata.st_atime_ns), int(metadata.st_mtime_ns)),
            )
            _fsync_directory(directory)
        source_receipts.append(
            {
                "label": source.label,
                "source_root": str(source.root),
                "source_device": int(source_root_metadata.st_dev),
                "source_inode": int(source_root_metadata.st_ino),
                "source_link_count": int(source_root_metadata.st_nlink),
                "archive_relative_root": f"payload/{source.label}",
            }
        )
        relocation_mappings.append(
            {
                "source_root": str(source.root),
                "archive_relative_root": f"payload/{source.label}",
            }
        )

    post_copy_inventory = inventory_sources(normalized)
    if (
        post_copy_inventory["inventory_sha256"]
        != plan["inventory"]["inventory_sha256"]
    ):
        raise MigrationSafetyError(
            "source tree changed during archive copy; partial evidence was retained"
        )

    archive_is_independent_device = all(
        int(receipt["source_device"]) != archive_device
        for receipt in source_receipts
    )
    manifest: dict[str, Any] = {
        "schema": MIGRATION_ARCHIVE_SCHEMA,
        "migration_id": _safe_component(migration_id, field="migration_id"),
        "created_at": time.time() if now is None else float(now),
        "copy_completed": True,
        "verification_completed": True,
        "source_deletion_permitted": False,
        "source_deletion_tool_available": False,
        "archive_device": archive_device,
        "archive_is_independent_device": archive_is_independent_device,
        "archive_role": (
            "independent_device_copy"
            if archive_is_independent_device
            else "project_local_archive"
        ),
        "independent_backup_still_required_before_deletion": (
            not archive_is_independent_device
        ),
        "source_inventory_sha256": plan["inventory"]["inventory_sha256"],
        "post_copy_source_inventory_sha256": post_copy_inventory[
            "inventory_sha256"
        ],
        "sources": source_receipts,
        "relocations": relocation_mappings,
        "files": file_receipts,
    }
    manifest_path = partial / "manifest.json"
    _write_json_durable(manifest_path, manifest)
    manifest_hash = _sha256_file(manifest_path)
    checksum_path = partial / "manifest.sha256"
    with checksum_path.open("x", encoding="ascii") as stream:
        stream.write(manifest_hash + "  manifest.json\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(partial)
    _rename_noreplace(partial, final)
    _fsync_directory(base)
    return final / "manifest.json"


def verify_archive(
    manifest_path: Path,
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    archive_root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != MIGRATION_ARCHIVE_SCHEMA:
        raise MigrationSafetyError(f"unsupported migration archive manifest: {path}")
    checksum_path = archive_root / "manifest.sha256"
    expected_manifest_hash = checksum_path.read_text(encoding="ascii").split()[0]
    observed_manifest_hash = _sha256_file(path)
    failures: list[dict[str, str]] = []
    if expected_manifest_hash != observed_manifest_hash:
        failures.append(
            {
                "code": "manifest_hash_mismatch",
                "path": str(path),
            }
        )
    sources_by_label = {
        str(item["label"]): Path(str(item["source_root"]))
        for item in payload.get("sources", [])
    }
    hardlink_destinations: dict[str, tuple[int, int]] = {}
    checked_files = 0
    checked_sources = 0
    for receipt in payload.get("files", []):
        label = str(receipt["source_label"])
        relative = Path(str(receipt["relative_path"]))
        destination = archive_root / "payload" / label / relative
        kind = str(receipt["kind"])
        if kind == "symlink":
            if not destination.is_symlink() or os.readlink(destination) != receipt.get(
                "symlink_target"
            ):
                failures.append(
                    {"code": "destination_symlink_mismatch", "path": str(destination)}
                )
            continue
        if not destination.is_file():
            failures.append({"code": "destination_missing", "path": str(destination)})
            continue
        checked_files += 1
        destination_metadata = destination.stat()
        if _sha256_file(destination) != receipt.get("sha256"):
            failures.append(
                {"code": "destination_hash_mismatch", "path": str(destination)}
            )
        if int(destination_metadata.st_size) != int(receipt.get("size_bytes", -1)):
            failures.append(
                {"code": "destination_size_mismatch", "path": str(destination)}
            )
        if "source_mode" in receipt and stat.S_IMODE(
            destination_metadata.st_mode
        ) != int(receipt["source_mode"]):
            failures.append(
                {"code": "destination_mode_mismatch", "path": str(destination)}
            )
        if int(destination_metadata.st_mtime_ns) != int(
            receipt.get("source_mtime_ns", -1)
        ):
            failures.append(
                {"code": "destination_mtime_mismatch", "path": str(destination)}
            )
        if bool(receipt.get("sparse_source")) and not (
            int(destination_metadata.st_size) > 0
            and _allocated_bytes(destination_metadata)
            < int(destination_metadata.st_size)
        ):
            failures.append(
                {"code": "destination_sparse_layout_lost", "path": str(destination)}
            )
        group = str(receipt.get("hardlink_group", ""))
        destination_identity = (
            int(destination_metadata.st_dev),
            int(destination_metadata.st_ino),
        )
        prior_identity = hardlink_destinations.setdefault(group, destination_identity)
        if group and prior_identity != destination_identity:
            failures.append(
                {"code": "destination_hardlink_topology_mismatch", "path": str(destination)}
            )
        if verify_sources:
            source = sources_by_label[label] / relative
            if not source.is_file():
                failures.append({"code": "source_missing", "path": str(source)})
            else:
                checked_sources += 1
                source_metadata = source.stat()
                if _sha256_file(source) != receipt.get("sha256"):
                    failures.append(
                        {"code": "source_hash_changed", "path": str(source)}
                    )
                if (
                    int(source_metadata.st_dev) == int(destination_metadata.st_dev)
                    and int(source_metadata.st_ino) == int(destination_metadata.st_ino)
                ):
                    failures.append(
                        {"code": "destination_aliases_source_inode", "path": str(source)}
                    )
    return {
        "schema": MIGRATION_VERIFY_SCHEMA,
        "manifest": str(path),
        "verified_at": time.time(),
        "verify_sources": bool(verify_sources),
        "checked_destination_files": checked_files,
        "checked_source_files": checked_sources,
        "failures": failures,
        "passed": not failures,
        "source_deletion_permitted": False,
    }


__all__ = [
    "MIGRATION_ARCHIVE_SCHEMA",
    "MIGRATION_INVENTORY_SCHEMA",
    "MIGRATION_VERIFY_SCHEMA",
    "MigrationSafetyError",
    "MigrationSource",
    "copy_sources_to_archive",
    "inventory_sources",
    "plan_archive_copy",
    "verify_archive",
]
