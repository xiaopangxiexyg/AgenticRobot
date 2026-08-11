#!/usr/bin/env python3
"""Phase-B migrate one classified artifact family without deleting its source.

The transaction is intentionally one-way and compatibility-first.  It can
freeze an exact source/target plan, publish a verified copy at an absent target,
and later verify both trees.  It has no source deletion, move, symlink, or
reader-switch operation.

GNU ``cp`` is used with archive, sparse, and reflink preservation.  Publication
uses Linux ``renameat2(RENAME_NOREPLACE)`` so an unexpected target is never
overwritten.  The source is hashed before and after copying, while the copied
tree is checked for logical content, metadata, hard-link topology, xattrs, and
sparse-file preservation before it is made visible at the final path.

Use ``scripts/runtime_artifact_archive.py`` first when a retained, whole-root
rollback snapshot under ``legacy/`` is required.  This command deliberately
accepts one exact semantic source and one exact canonical target per plan.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_artifact_paths import RuntimeArtifactLayout


PLAN_SCHEMA = "dexteleop.runtime_artifact_copy_plan.v1"
JOURNAL_SCHEMA = "dexteleop.runtime_artifact_copy_journal.v1"
VERIFY_SCHEMA = "dexteleop.runtime_artifact_copy_verification.v1"
PLAN_FILENAME = "plan.json"
JOURNAL_FILENAME = "journal.jsonl"
LOCK_FILENAME = "transaction.lock"
_HASH_CHUNK_BYTES = 4 << 20
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class MigrationSafetyError(RuntimeError):
    """A fail-closed migration invariant was not satisfied."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_lexical_path(value: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise MigrationSafetyError(f"{field} must be an absolute path: {candidate}")
    return Path(os.path.abspath(os.fspath(candidate)))


def _assert_no_symlink_prefix(path: Path, *, include_leaf: bool) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise MigrationSafetyError(
                f"path traverses a symlink component and is not exact: {current}"
            )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_disjoint_paths(
    *, source: Path, target: Path, plan_dir: Path
) -> None:
    if _is_within(source, target) or _is_within(target, source):
        raise MigrationSafetyError(
            f"source and target must not contain one another: {source} ; {target}"
        )
    if _is_within(plan_dir, source) or _is_within(source, plan_dir):
        raise MigrationSafetyError(
            f"plan directory and source must be disjoint: {plan_dir} ; {source}"
        )
    if _is_within(plan_dir, target) or _is_within(target, plan_dir):
        raise MigrationSafetyError(
            f"plan directory and target must be disjoint: {plan_dir} ; {target}"
        )


def _safe_migration_id(value: str) -> str:
    migration_id = str(value).strip()
    if (
        not migration_id
        or migration_id != value
        or migration_id in {".", ".."}
        or "/" in migration_id
        or "\\" in migration_id
        or "\x00" in migration_id
    ):
        raise MigrationSafetyError(
            "migration_id must be one non-empty, trimmed path component"
        )
    return migration_id


def _xattr_fingerprints(path_or_fd: Path | int) -> list[dict[str, Any]]:
    try:
        if isinstance(path_or_fd, int):
            names = os.listxattr(path_or_fd)
        else:
            names = os.listxattr(path_or_fd, follow_symlinks=False)
    except OSError as error:
        raise MigrationSafetyError(
            f"cannot enumerate extended attributes for {path_or_fd}: {error}"
        ) from error
    result: list[dict[str, Any]] = []
    for name in sorted(names):
        try:
            if isinstance(path_or_fd, int):
                value = os.getxattr(path_or_fd, name)
            else:
                value = os.getxattr(
                    path_or_fd,
                    name,
                    follow_symlinks=False,
                )
        except OSError as error:
            raise MigrationSafetyError(
                f"cannot read extended attribute {name!r} for {path_or_fd}: {error}"
            ) from error
        result.append(
            {
                "name_b64": base64.b64encode(os.fsencode(name)).decode("ascii"),
                "size": len(value),
                "sha256": _sha256_bytes(value),
            }
        )
    return result


def _stable_regular_file_record(
    path: Path,
    *,
    relative_path: str,
    initial_lstat: os.stat_result,
) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MigrationSafetyError(f"cannot open regular file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            before.st_dev != initial_lstat.st_dev
            or before.st_ino != initial_lstat.st_ino
            or not stat.S_ISREG(before.st_mode)
        ):
            raise MigrationSafetyError(f"file identity changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
        xattrs = _xattr_fingerprints(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final_lstat = path.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_nlink",
        "st_blocks",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise MigrationSafetyError(f"file changed while hashing: {path}")
    if any(
        getattr(after, name) != getattr(final_lstat, name)
        for name in stable_fields
    ):
        raise MigrationSafetyError(f"file path changed while hashing: {path}")
    allocated_bytes = int(after.st_blocks) * 512
    return {
        "relative_path": relative_path,
        "kind": "file",
        "mode": stat.S_IMODE(after.st_mode),
        "uid": int(after.st_uid),
        "gid": int(after.st_gid),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "link_count": int(after.st_nlink),
        "allocated_bytes": allocated_bytes,
        "sparse": bool(after.st_size > 0 and allocated_bytes < after.st_size),
        "sha256": digest.hexdigest(),
        "xattrs": xattrs,
    }


def _non_file_record(
    path: Path,
    *,
    relative_path: str,
    metadata: os.stat_result,
) -> dict[str, Any]:
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
        link_target = None
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        link_target = os.readlink(path)
    else:
        raise MigrationSafetyError(
            f"unsupported FIFO/socket/device entry in artifact tree: {path}"
        )
    final_lstat = path.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_nlink",
    )
    if any(
        getattr(metadata, name) != getattr(final_lstat, name)
        for name in stable_fields
    ):
        raise MigrationSafetyError(f"entry changed while inspecting: {path}")
    return {
        "relative_path": relative_path,
        "kind": kind,
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": int(metadata.st_uid),
        "gid": int(metadata.st_gid),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "link_count": int(metadata.st_nlink),
        "allocated_bytes": int(metadata.st_blocks) * 512,
        "sparse": False,
        "sha256": None,
        "link_target": link_target,
        "xattrs": _xattr_fingerprints(path),
    }


def _scan_tree(root: Path) -> dict[str, Any]:
    root = Path(root)
    root_lstat = root.lstat()
    if stat.S_ISLNK(root_lstat.st_mode):
        raise MigrationSafetyError(f"artifact root must not be a symlink: {root}")
    if not (
        stat.S_ISREG(root_lstat.st_mode) or stat.S_ISDIR(root_lstat.st_mode)
    ):
        raise MigrationSafetyError(
            f"artifact root must be a regular file or directory: {root}"
        )

    records: list[dict[str, Any]] = []
    pending: list[tuple[Path, str]] = [(root, ".")]
    while pending:
        path, relative_path = pending.pop()
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            record = _stable_regular_file_record(
                path,
                relative_path=relative_path,
                initial_lstat=metadata,
            )
        else:
            record = _non_file_record(
                path,
                relative_path=relative_path,
                metadata=metadata,
            )
        records.append(record)
        if record["kind"] != "directory":
            continue
        try:
            with os.scandir(path) as entries:
                children = sorted(Path(entry.path) for entry in entries)
        except OSError as error:
            raise MigrationSafetyError(f"cannot enumerate directory {path}: {error}") from error
        for child in reversed(children):
            relative = child.relative_to(root).as_posix()
            pending.append((child, relative))

    records.sort(key=lambda item: os.fsencode(str(item["relative_path"])))
    inode_paths: dict[tuple[int, int], list[str]] = {}
    for record in records:
        if record["kind"] != "file":
            continue
        key = (int(record["device"]), int(record["inode"]))
        inode_paths.setdefault(key, []).append(str(record["relative_path"]))
    for paths in inode_paths.values():
        paths.sort(key=os.fsencode)
    for record in records:
        if record["kind"] != "file":
            continue
        paths = inode_paths[(int(record["device"]), int(record["inode"]))]
        group_hash = _sha256_bytes(_canonical_json_bytes(paths))
        record["internal_link_group"] = group_hash
        record["internal_link_count"] = len(paths)
        record["external_link_count"] = int(record["link_count"]) - len(paths)
        if int(record["external_link_count"]) < 0:
            raise MigrationSafetyError(
                f"invalid hard-link count while scanning {root}: {paths}"
            )

    summary = {
        "entry_count": len(records),
        "file_count": sum(item["kind"] == "file" for item in records),
        "directory_count": sum(
            item["kind"] == "directory" for item in records
        ),
        "symlink_count": sum(item["kind"] == "symlink" for item in records),
        "logical_bytes": sum(
            int(item["size"]) for item in records if item["kind"] == "file"
        ),
        "allocated_bytes": sum(
            int(item["allocated_bytes"])
            for item in records
            if item["kind"] == "file"
        ),
        "sparse_file_count": sum(
            bool(item["sparse"]) for item in records if item["kind"] == "file"
        ),
        "hardlinked_path_count": sum(
            int(item.get("internal_link_count", 1)) > 1
            for item in records
            if item["kind"] == "file"
        ),
        "externally_linked_path_count": sum(
            int(item.get("external_link_count", 0)) > 0
            for item in records
            if item["kind"] == "file"
        ),
    }
    identity_payload = {"entries": records, "summary": summary}
    publication_entries = []
    for record in records:
        stable_after_rename = dict(record)
        # Renaming a directory changes its ctime even though the inode and all
        # copied bytes/metadata remain the same.  Keep ctime in the frozen
        # source identity, but exclude it from the physical identity used only
        # across the final no-replace rename.
        stable_after_rename.pop("ctime_ns", None)
        publication_entries.append(stable_after_rename)
    publication_payload = {
        "entries": publication_entries,
        "summary": summary,
    }
    return {
        **identity_payload,
        "identity_hash": _sha256_bytes(_canonical_json_bytes(identity_payload)),
        "publication_identity_hash": _sha256_bytes(
            _canonical_json_bytes(publication_payload)
        ),
    }


def _logical_projection(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in snapshot["entries"]:
        record = dict(raw)
        kind = str(record["kind"])
        projected = {
            "relative_path": record["relative_path"],
            "kind": kind,
            "mode": record["mode"],
            "uid": record["uid"],
            "gid": record["gid"],
            "mtime_ns": record["mtime_ns"],
            "xattrs": record["xattrs"],
        }
        if kind == "file":
            projected.update(
                {
                    "size": record["size"],
                    "sha256": record["sha256"],
                    "internal_link_group": record["internal_link_group"],
                    "internal_link_count": record["internal_link_count"],
                }
            )
        elif kind == "symlink":
            projected["link_target"] = record["link_target"]
        result.append(projected)
    return result


def _verify_copy_equivalence(
    source_snapshot: Mapping[str, Any],
    copied_snapshot: Mapping[str, Any],
) -> None:
    if _logical_projection(source_snapshot) != _logical_projection(copied_snapshot):
        raise MigrationSafetyError(
            "copied tree differs in content, metadata, xattrs, or hard-link topology"
        )
    source_by_path = {
        str(item["relative_path"]): item for item in source_snapshot["entries"]
    }
    copied_by_path = {
        str(item["relative_path"]): item for item in copied_snapshot["entries"]
    }
    for relative_path, source in source_by_path.items():
        if source["kind"] != "file":
            continue
        copied = copied_by_path[relative_path]
        if int(copied["external_link_count"]) != 0:
            raise MigrationSafetyError(
                f"copied file has an unexpected external hard link: {relative_path}"
            )
        if bool(source["sparse"]) and not bool(copied["sparse"]):
            raise MigrationSafetyError(
                f"sparse allocation was not preserved: {relative_path}"
            )


def _open_writable_fds(root: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise MigrationSafetyError("/proc is required for writable-FD preflight")
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                link_value = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError, ProcessLookupError):
                continue
            if not link_value.startswith("/"):
                continue
            resolved = Path(link_value.removesuffix(" (deleted)"))
            if not _is_within(resolved, root):
                continue
            writable: bool | None = None
            try:
                fields = {}
                for line in (process_dir / "fdinfo" / descriptor.name).read_text(
                    encoding="utf-8"
                ).splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                flags = int(fields["flags"], 8)
                writable = (flags & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR)
            except (FileNotFoundError, KeyError, PermissionError, ValueError):
                pass
            if writable is not False:
                matches.append(
                    {
                        "pid": int(process_dir.name),
                        "fd": int(descriptor.name),
                        "path": str(resolved),
                        "writable": writable,
                    }
                )
    return sorted(matches, key=lambda item: (item["pid"], item["fd"]))


def _require_no_writable_fds(source: Path) -> None:
    matches = _open_writable_fds(source)
    if matches:
        raise MigrationSafetyError(
            "source has open or indeterminate writable file descriptors: "
            + json.dumps(matches[:20], ensure_ascii=False, sort_keys=True)
        )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MigrationSafetyError(f"refusing to overwrite JSON artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _rename_noreplace(temporary, path)
    _fsync_directory(path.parent)


def _event_hash(event: Mapping[str, Any]) -> str:
    unsigned = dict(event)
    unsigned.pop("event_hash", None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise MigrationSafetyError(f"journal is missing or not a real file: {path}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise MigrationSafetyError(f"journal has a torn final line: {path}")
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise MigrationSafetyError(f"journal contains a blank line: {path}:{line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise MigrationSafetyError(
                f"journal contains invalid JSON: {path}:{line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise MigrationSafetyError(
                f"journal event must be an object: {path}:{line_number}"
            )
        if event.get("schema") != JOURNAL_SCHEMA:
            raise MigrationSafetyError(
                f"unsupported journal schema at {path}:{line_number}"
            )
        if event.get("previous_event_hash") != previous_hash:
            raise MigrationSafetyError(
                f"journal hash-chain predecessor mismatch at {path}:{line_number}"
            )
        actual_hash = _event_hash(event)
        if event.get("event_hash") != actual_hash:
            raise MigrationSafetyError(
                f"journal event hash mismatch at {path}:{line_number}"
            )
        previous_hash = actual_hash
        events.append(event)
    if not events:
        raise MigrationSafetyError(f"journal is empty: {path}")
    return events


def _append_event(
    plan_dir: Path,
    *,
    plan_hash: str,
    state: str,
    details: Mapping[str, Any] | None = None,
    initialize: bool = False,
) -> dict[str, Any]:
    journal_path = plan_dir / JOURNAL_FILENAME
    if initialize:
        if journal_path.exists() or journal_path.is_symlink():
            raise MigrationSafetyError(f"refusing to overwrite journal: {journal_path}")
        previous_hash = None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC
    else:
        events = _read_journal(journal_path)
        previous_hash = str(events[-1]["event_hash"])
        flags = os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC
    event: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA,
        "event_id": uuid.uuid4().hex,
        "recorded_at_unix_ns": time.time_ns(),
        "plan_hash": plan_hash,
        "previous_event_hash": previous_hash,
        "state": str(state),
        "details": dict(details or {}),
    }
    event["event_hash"] = _event_hash(event)
    payload = _canonical_json_bytes(event) + b"\n"
    descriptor = os.open(journal_path, flags, 0o664)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise MigrationSafetyError(f"short append to journal: {journal_path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(plan_dir)
    return event


@contextlib.contextmanager
def _transaction_lock(plan_dir: Path) -> Iterator[None]:
    lock_path = plan_dir / LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o664)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MigrationSafetyError(
                f"another migration process owns the transaction lock: {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_plan(plan_dir: Path) -> dict[str, Any]:
    plan_path = plan_dir / PLAN_FILENAME
    if not plan_path.is_file() or plan_path.is_symlink():
        raise MigrationSafetyError(f"plan is missing or not a real file: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MigrationSafetyError(f"invalid plan JSON: {plan_path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise MigrationSafetyError(f"unsupported plan schema: {plan_path}")
    expected_hash = str(plan.get("plan_hash") or "")
    unsigned = dict(plan)
    unsigned.pop("plan_hash", None)
    actual_hash = _sha256_bytes(_canonical_json_bytes(unsigned))
    if not expected_hash or expected_hash != actual_hash:
        raise MigrationSafetyError(
            f"plan hash mismatch: expected={expected_hash!r} actual={actual_hash!r}"
        )
    events = _read_journal(plan_dir / JOURNAL_FILENAME)
    if any(str(event.get("plan_hash")) != expected_hash for event in events):
        raise MigrationSafetyError("journal contains an event for another plan hash")
    return plan


def _temporary_target(target: Path, migration_id: str) -> Path:
    return target.with_name(f".{target.name}.migration-{migration_id}.copying")


def _run_cp(source: Path, temporary_target: Path) -> None:
    command = [
        "cp",
        "--archive",
        "--sparse=always",
        "--reflink=auto",
        "--",
        os.fspath(source),
        os.fspath(temporary_target),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise MigrationSafetyError("GNU cp is required but was not found") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4000:]
        raise MigrationSafetyError(
            f"copy command failed with exit {completed.returncode}: {detail}"
        )


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        elif not stat.S_ISLNK(metadata.st_mode):
            raise MigrationSafetyError(f"unsupported copied entry: {path}")
    for directory in reversed(directories):
        _fsync_directory(directory)


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
        raise MigrationSafetyError(f"target appeared before publication: {target}")
    raise MigrationSafetyError(
        f"atomic no-replace publication failed: {os.strerror(error_number)}"
    )


def create_plan(
    *,
    source: Path,
    target: Path,
    plan_dir: Path,
    migration_id: str,
) -> dict[str, Any]:
    migration_id = _safe_migration_id(migration_id)
    source = _absolute_lexical_path(source, field="source")
    target = _absolute_lexical_path(target, field="target")
    plan_dir = _absolute_lexical_path(plan_dir, field="plan_dir")
    _assert_no_symlink_prefix(source, include_leaf=True)
    _assert_no_symlink_prefix(target, include_leaf=False)
    _assert_no_symlink_prefix(plan_dir, include_leaf=False)
    if not source.exists():
        raise MigrationSafetyError(f"source does not exist: {source}")
    if target.exists() or target.is_symlink():
        raise MigrationSafetyError(f"target must be absent: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise MigrationSafetyError(
            f"exact target parent must already be a real directory: {target.parent}"
        )
    if plan_dir.exists() or plan_dir.is_symlink():
        raise MigrationSafetyError(f"plan directory must be absent: {plan_dir}")
    _require_disjoint_paths(source=source, target=target, plan_dir=plan_dir)
    _require_no_writable_fds(source)
    first_snapshot = _scan_tree(source)
    _require_no_writable_fds(source)
    second_snapshot = _scan_tree(source)
    if first_snapshot["identity_hash"] != second_snapshot["identity_hash"]:
        raise MigrationSafetyError("source changed while the plan was being frozen")

    plan_dir.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_prefix(plan_dir, include_leaf=False)
    plan_dir.mkdir(mode=0o775)
    _fsync_directory(plan_dir.parent)
    unsigned: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "migration_id": migration_id,
        "created_at_unix_ns": time.time_ns(),
        "source": str(source),
        "target": str(target),
        "plan_dir": str(plan_dir),
        "source_root_device": int(source.lstat().st_dev),
        "target_parent_device": int(target.parent.stat().st_dev),
        "target_parent_inode": int(target.parent.stat().st_ino),
        "same_source_target_device": bool(
            source.lstat().st_dev == target.parent.stat().st_dev
        ),
        "backup_assessment": {
            "independent_backup_proven": False,
            "reason": (
                "a verified copy is not by itself proof of an independent backup"
            ),
        },
        "source_snapshot": second_snapshot,
        "capabilities": {
            "copy_and_verify": True,
            "verify": True,
            "delete_source": False,
            "move_source": False,
            "install_compatibility_link": False,
            "switch_reader_default": False,
        },
    }
    plan = dict(unsigned)
    plan["plan_hash"] = _sha256_bytes(_canonical_json_bytes(unsigned))
    try:
        _write_json_exclusive(plan_dir / PLAN_FILENAME, plan)
        _append_event(
            plan_dir,
            plan_hash=str(plan["plan_hash"]),
            state="planned",
            details={
                "source_identity_hash": second_snapshot["identity_hash"],
                "target_absent": True,
                "source_deletion_supported": False,
            },
            initialize=True,
        )
    except Exception:
        # The directory is intentionally left in place as evidence.  This tool
        # never recursively removes even its own partially written transaction.
        raise
    return plan


def _validate_live_paths(plan: Mapping[str, Any], plan_dir: Path) -> tuple[Path, Path]:
    source = _absolute_lexical_path(str(plan["source"]), field="source")
    target = _absolute_lexical_path(str(plan["target"]), field="target")
    recorded_plan_dir = _absolute_lexical_path(
        str(plan["plan_dir"]), field="recorded plan_dir"
    )
    if recorded_plan_dir != plan_dir:
        raise MigrationSafetyError(
            f"plan directory moved or aliased: {recorded_plan_dir} != {plan_dir}"
        )
    _assert_no_symlink_prefix(source, include_leaf=True)
    _assert_no_symlink_prefix(target, include_leaf=False)
    _assert_no_symlink_prefix(plan_dir, include_leaf=True)
    _require_disjoint_paths(source=source, target=target, plan_dir=plan_dir)
    if not source.exists() or source.is_symlink():
        raise MigrationSafetyError(f"source is missing or became a symlink: {source}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise MigrationSafetyError(f"target parent changed: {target.parent}")
    parent_stat = target.parent.stat()
    if (
        int(parent_stat.st_dev) != int(plan["target_parent_device"])
        or int(parent_stat.st_ino) != int(plan["target_parent_inode"])
    ):
        raise MigrationSafetyError(
            f"target parent identity changed since planning: {target.parent}"
        )
    return source, target


def _require_source_matches_plan(source: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    _require_no_writable_fds(source)
    snapshot = _scan_tree(source)
    expected = str(plan["source_snapshot"]["identity_hash"])
    if snapshot["identity_hash"] != expected:
        raise MigrationSafetyError(
            f"source identity changed: {snapshot['identity_hash']} != {expected}"
        )
    _require_no_writable_fds(source)
    return snapshot


def copy_verify(plan_dir: Path) -> dict[str, Any]:
    plan_dir = _absolute_lexical_path(plan_dir, field="plan_dir")
    _assert_no_symlink_prefix(plan_dir, include_leaf=True)
    if not plan_dir.is_dir() or plan_dir.is_symlink():
        raise MigrationSafetyError(f"plan directory is not a real directory: {plan_dir}")
    with _transaction_lock(plan_dir):
        plan = _load_plan(plan_dir)
        plan_hash = str(plan["plan_hash"])
        source, target = _validate_live_paths(plan, plan_dir)
        migration_id = _safe_migration_id(str(plan["migration_id"]))
        temporary = _temporary_target(target, migration_id)
        events = _read_journal(plan_dir / JOURNAL_FILENAME)
        last_state = str(events[-1]["state"])

        if target.exists() or target.is_symlink():
            if last_state not in {"published", "published_recovered"}:
                verified_event = next(
                    (
                        event
                        for event in reversed(events)
                        if event.get("state") == "copy_verified"
                    ),
                    None,
                )
                if verified_event is None or temporary.exists() or temporary.is_symlink():
                    raise MigrationSafetyError(
                        f"target exists without a recoverable verified publication: {target}"
                    )
                source_snapshot = _require_source_matches_plan(source, plan)
                target_snapshot = _scan_tree(target)
                _verify_copy_equivalence(source_snapshot, target_snapshot)
                expected_identity = str(
                    verified_event.get("details", {}).get(
                        "temporary_publication_identity_hash", ""
                    )
                )
                observed_identity = str(
                    target_snapshot["publication_identity_hash"]
                )
                failure_details = events[-1].get("details", {})
                recoverable_post_publish_identity_failure = bool(
                    last_state == "copy_failed"
                    and isinstance(failure_details, dict)
                    and failure_details.get("error_type") == "MigrationSafetyError"
                    and failure_details.get("error")
                    == "target identity changed during atomic publication"
                    and failure_details.get("source_retained") is True
                    and failure_details.get("target_exists") is True
                    and failure_details.get("temporary_exists") is False
                )
                if (
                    observed_identity != expected_identity
                    and not recoverable_post_publish_identity_failure
                ):
                    raise MigrationSafetyError(
                        "published target does not match the pre-publication inode identity"
                    )
                recovery_details: dict[str, Any] = {
                    "target_identity_hash": target_snapshot["identity_hash"],
                }
                if observed_identity != expected_identity:
                    recovery_details.update(
                        {
                            "recovery_reason": (
                                "verified_post_fsync_publication_identity_reconciliation"
                            ),
                            "recorded_publication_identity_hash": expected_identity,
                            "observed_publication_identity_hash": observed_identity,
                        }
                    )
                _append_event(
                    plan_dir,
                    plan_hash=plan_hash,
                    state="published_recovered",
                    details=recovery_details,
                )
            return verify_plan(plan_dir)

        try:
            source_snapshot = _require_source_matches_plan(source, plan)
            if temporary.exists() or temporary.is_symlink():
                if last_state not in {"copy_started", "copy_failed"}:
                    raise MigrationSafetyError(
                        f"unexpected temporary target exists: {temporary}"
                    )
            else:
                _append_event(
                    plan_dir,
                    plan_hash=plan_hash,
                    state="copy_started",
                    details={"temporary_target": str(temporary)},
                )
                _run_cp(source, temporary)

            # Delayed allocation can change st_blocks when sparse WAL files are
            # flushed.  Freeze the publication identity only after every file
            # and directory is durable, otherwise a correct atomic rename can
            # look like an inode substitution.
            _fsync_tree(temporary)
            copied_snapshot = _scan_tree(temporary)
            _verify_copy_equivalence(source_snapshot, copied_snapshot)
            source_after = _require_source_matches_plan(source, plan)
            if source_after["identity_hash"] != source_snapshot["identity_hash"]:
                raise MigrationSafetyError("source changed during the copy transaction")
            _append_event(
                plan_dir,
                plan_hash=plan_hash,
                state="copy_verified",
                details={
                    "temporary_target": str(temporary),
                    "temporary_identity_hash": copied_snapshot["identity_hash"],
                    "temporary_publication_identity_hash": (
                        copied_snapshot["publication_identity_hash"]
                    ),
                    "source_identity_hash": source_after["identity_hash"],
                },
            )
            _rename_noreplace(temporary, target)
            _fsync_directory(target.parent)
            target_snapshot = _scan_tree(target)
            _verify_copy_equivalence(source_snapshot, target_snapshot)
            if (
                target_snapshot["publication_identity_hash"]
                != copied_snapshot["publication_identity_hash"]
            ):
                raise MigrationSafetyError("target identity changed during atomic publication")
            _append_event(
                plan_dir,
                plan_hash=plan_hash,
                state="published",
                details={
                    "target": str(target),
                    "target_identity_hash": target_snapshot["identity_hash"],
                    "source_retained": source.exists(),
                },
            )
        except Exception as error:
            try:
                _append_event(
                    plan_dir,
                    plan_hash=plan_hash,
                    state="copy_failed",
                    details={
                        "error_type": type(error).__name__,
                        "error": str(error)[:4000],
                        "source_retained": source.exists(),
                        "target_exists": target.exists() or target.is_symlink(),
                        "temporary_exists": temporary.exists() or temporary.is_symlink(),
                    },
                )
            except Exception:
                pass
            raise
        return verify_plan(plan_dir)


def verify_plan(plan_dir: Path) -> dict[str, Any]:
    plan_dir = _absolute_lexical_path(plan_dir, field="plan_dir")
    _assert_no_symlink_prefix(plan_dir, include_leaf=True)
    plan = _load_plan(plan_dir)
    source, target = _validate_live_paths(plan, plan_dir)
    if not target.exists() or target.is_symlink():
        raise MigrationSafetyError(f"published target is missing or a symlink: {target}")
    source_snapshot = _require_source_matches_plan(source, plan)
    target_snapshot = _scan_tree(target)
    _verify_copy_equivalence(source_snapshot, target_snapshot)
    return {
        "schema": VERIFY_SCHEMA,
        "verified_at_unix_ns": time.time_ns(),
        "plan_hash": plan["plan_hash"],
        "source": str(source),
        "target": str(target),
        "source_identity_hash": source_snapshot["identity_hash"],
        "target_identity_hash": target_snapshot["identity_hash"],
        "source_retained": True,
        "source_deletion_supported": False,
        "summary": source_snapshot["summary"],
    }


def _default_plan_dir(migration_id: str) -> Path:
    return RuntimeArtifactLayout.configured().migration_dir(
        migration_id=_safe_migration_id(migration_id)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="Freeze a content-hashed copy plan; source and target stay unchanged.",
    )
    plan.add_argument("--migration-id", required=True)
    plan.add_argument("--source", required=True, type=Path)
    plan.add_argument("--target", required=True, type=Path)
    plan.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
        help="Exact transaction metadata directory; defaults under the configured migration root.",
    )

    copy = subparsers.add_parser(
        "copy-verify",
        help="Copy to a temporary inode, verify, and atomically publish without deleting source.",
    )
    copy.add_argument("--plan-dir", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify",
        help="Read-only verification of a published target and its retained source.",
    )
    verify.add_argument("--plan-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            migration_id = _safe_migration_id(args.migration_id)
            plan_dir = (
                _default_plan_dir(migration_id)
                if args.plan_dir is None
                else args.plan_dir
            )
            result = create_plan(
                source=args.source,
                target=args.target,
                plan_dir=plan_dir,
                migration_id=migration_id,
            )
        elif args.command == "copy-verify":
            result = copy_verify(args.plan_dir)
        elif args.command == "verify":
            result = verify_plan(args.plan_dir)
        else:  # pragma: no cover - argparse owns the command choices.
            raise MigrationSafetyError(f"unsupported command: {args.command}")
    except MigrationSafetyError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
