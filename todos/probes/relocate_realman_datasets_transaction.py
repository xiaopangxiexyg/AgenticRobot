#!/usr/bin/env python3
"""Journaled same-filesystem relocation for the 2026-08-06 dataset consolidation.

This is an operator probe, not a production entrypoint.  It discovers complete,
non-empty dataset roots below the approved source namespaces, freezes a plan with
tree/inode fingerprints, and then atomically renames each root before installing
directory symlinks at the old locations.  Any failure triggers a best-effort full
rollback and leaves a durable journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_MAP_KEYS = (
    "video_paths",
    "tactile_paths",
    "wrist_depth_mask_paths",
    "wrist_depth_raw_paths",
)
PRUNED_DIR_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
    "anaconda3",
    "data",
    "miniconda3",
    "node_modules",
    "tactile",
    "videos",
    "wrist_depth",
}
ACTIVE_PROCESS_MARKERS = (
    "run_vive_rm65_real_teleop",
    "teleop_runtime.collection",
    "tactile-maintain",
    "tactile-recompress-zstd",
    "idle_tactile_compression",
)


class RelocationError(RuntimeError):
    """Raised when a fail-closed relocation invariant is not satisfied."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RelocationError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RelocationError(f"{path}:{line_number} is not an object")
        records.append(value)
    return records


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _dataset_reference_errors(root: Path) -> tuple[list[str], int, str | None]:
    errors: list[str] = []
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    try:
        info = _read_json(info_path)
    except Exception as exc:
        return [f"invalid info.json: {type(exc).__name__}: {exc}"], 0, None
    try:
        episodes = _read_jsonl(episodes_path)
    except Exception as exc:
        return [f"invalid episodes.jsonl: {type(exc).__name__}: {exc}"], 0, None

    declared = info.get(
        "episode_count",
        info.get("total_episodes", info.get("num_episodes")),
    )
    if declared is not None and int(declared) != len(episodes):
        errors.append(f"episode count mismatch: {declared} != {len(episodes)}")

    for record in episodes:
        episode_index = record.get("episode_index", "?")
        parquet_relative = record.get("parquet_path")
        if not parquet_relative or not (root / str(parquet_relative)).is_file():
            errors.append(f"episode {episode_index}: missing parquet {parquet_relative}")
        for map_key in ARTIFACT_MAP_KEYS:
            mapping = record.get(map_key, {})
            if not isinstance(mapping, dict):
                errors.append(f"episode {episode_index}: {map_key} is not an object")
                continue
            for name, relative in mapping.items():
                artifact = root / str(relative)
                if not artifact.is_file():
                    errors.append(
                        f"episode {episode_index}: missing {map_key}.{name} {relative}"
                    )
                    continue
                if map_key == "video_paths":
                    continue
                try:
                    manifest = _read_json(artifact)
                except Exception as exc:
                    errors.append(
                        f"episode {episode_index}: invalid manifest {relative}: {exc}"
                    )
                    continue
                streams = manifest.get("streams")
                if isinstance(streams, dict):
                    for stream_name, stream in streams.items():
                        if not isinstance(stream, dict) or not stream.get("path"):
                            continue
                        if not (root / str(stream["path"])).is_file():
                            errors.append(
                                f"episode {episode_index}: {relative} stream "
                                f"{stream_name!r} is missing"
                            )
                elif manifest.get("stream_path") and not (
                    root / str(manifest["stream_path"])
                ).is_file():
                    errors.append(
                        f"episode {episode_index}: {relative} stream is missing"
                    )
    schema = info.get(
        "dexteleop_dataset_schema",
        info.get("schema_version", info.get("codebase_version", info.get("version"))),
    )
    return errors, len(episodes), None if schema is None else str(schema)


def _discover_dataset_roots(
    *, scan_root: Path, target_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=False):
        path = Path(dirpath)
        if path == target_root or target_root in path.parents:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in PRUNED_DIR_NAMES]
        if path.name != "meta" or "info.json" not in filenames:
            continue
        root = path.parent
        errors, episode_count, schema = _dataset_reference_errors(root)
        record = {
            "source": str(root),
            "schema": schema,
            "episode_count": episode_count,
            "errors": errors,
        }
        if (
            episode_count > 0
            and not errors
            and ".invalid-hardlink-write-through-" not in str(root)
        ):
            valid.append(record)
        else:
            excluded.append(record)
        dirnames[:] = []
    valid.sort(key=lambda item: item["source"])
    excluded.sort(key=lambda item: item["source"])
    return valid, excluded


def _target_for_source(
    source: Path, *, source_namespaces: tuple[Path, ...], target_root: Path
) -> Path:
    matches: list[tuple[Path, Path]] = []
    for namespace in source_namespaces:
        try:
            relative = source.relative_to(namespace)
        except ValueError:
            continue
        matches.append((namespace, relative))
    if len(matches) != 1:
        raise RelocationError(
            f"source must belong to exactly one approved namespace: {source}"
        )
    namespace, relative = matches[0]
    return target_root / namespace.name / relative


def _tree_identity(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RelocationError(f"dataset root must be a real directory: {root}")
    digest = hashlib.sha256()
    path_count = 0
    file_count = 0
    directory_count = 0
    logical_bytes = 0
    hardlinked_path_count = 0
    symlink_count = 0
    special_count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        dirnames.sort()
        filenames.sort()
        for name in dirnames + filenames:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                symlink_count += 1
                kind = "symlink"
                extra = os.readlink(path)
            elif stat.S_ISREG(mode):
                file_count += 1
                logical_bytes += metadata.st_size
                hardlinked_path_count += int(metadata.st_nlink > 1)
                kind = "file"
                extra = ""
            elif stat.S_ISDIR(mode):
                directory_count += 1
                kind = "directory"
                extra = ""
            else:
                special_count += 1
                kind = "special"
                extra = ""
            path_count += 1
            line = (
                f"{relative}\0{kind}\0{metadata.st_dev}\0{metadata.st_ino}\0"
                f"{metadata.st_size}\0{metadata.st_nlink}\0{stat.S_IMODE(mode)}\0{extra}\n"
            )
            digest.update(line.encode("utf-8", errors="surrogateescape"))
    if symlink_count or special_count:
        raise RelocationError(
            f"internal symlink/special file found in {root}: "
            f"symlinks={symlink_count} special={special_count}"
        )
    root_stat = root.stat()
    return {
        "root_device": root_stat.st_dev,
        "root_inode": root_stat.st_ino,
        "tree_identity_hash": digest.hexdigest(),
        "path_count": path_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "logical_bytes": logical_bytes,
        "hardlinked_path_count": hardlinked_path_count,
        "symlink_count": symlink_count,
        "special_count": special_count,
        "info_sha256": _sha256_file(root / "meta" / "info.json"),
        "episodes_sha256": _sha256_file(root / "meta" / "episodes.jsonl"),
    }


def _active_processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(marker in command for marker in ACTIVE_PROCESS_MARKERS):
            matches.append({"pid": int(process_dir.name), "command": command})
    return sorted(matches, key=lambda item: item["pid"])


def _open_dataset_fds(roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                resolved = descriptor.resolve(strict=True)
            except (FileNotFoundError, PermissionError, OSError, ProcessLookupError):
                continue
            if not _inside(resolved, roots):
                continue
            writable = None
            try:
                fields = {}
                for line in (process_dir / "fdinfo" / descriptor.name).read_text().splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                flags = int(fields["flags"], 8)
                writable = (flags & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR)
            except (FileNotFoundError, KeyError, PermissionError, ValueError):
                pass
            matches.append(
                {
                    "pid": int(process_dir.name),
                    "fd": int(descriptor.name),
                    "path": str(resolved),
                    "writable": writable,
                }
            )
    return sorted(matches, key=lambda item: (item["pid"], item["fd"]))


def _validate_target_root(target_root: Path, *, allow_migration_dir: bool) -> None:
    if target_root.is_symlink() or not target_root.is_dir():
        raise RelocationError(f"target root must be a real directory: {target_root}")
    allowed = {".migration"} if allow_migration_dir else set()
    unexpected = sorted(path.name for path in target_root.iterdir() if path.name not in allowed)
    if unexpected:
        raise RelocationError(
            f"target root contains unexpected entries: {unexpected[:20]}"
        )


def _plan(args: argparse.Namespace) -> None:
    scan_root = Path(args.scan_root).expanduser().resolve(strict=True)
    target_root = Path(args.target_root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().absolute()
    source_namespaces = tuple(
        Path(value).expanduser().resolve(strict=True) for value in args.source_namespace
    )
    _validate_target_root(target_root, allow_migration_dir=False)
    valid, excluded = _discover_dataset_roots(scan_root=scan_root, target_root=target_root)
    if len(valid) != int(args.expected_count):
        raise RelocationError(
            f"valid dataset count changed: {len(valid)} != {args.expected_count}"
        )
    if _active_processes():
        raise RelocationError(f"active collection/postprocess processes: {_active_processes()}")

    sources = tuple(Path(item["source"]) for item in valid)
    open_fds = _open_dataset_fds(sources)
    writable_fds = [item for item in open_fds if item["writable"] is not False]
    if writable_fds:
        raise RelocationError(f"dataset file descriptors may be writable: {writable_fds}")

    target_device = target_root.stat().st_dev
    entries: list[dict[str, Any]] = []
    seen_targets: set[Path] = set()
    for item in valid:
        source = Path(item["source"])
        target = _target_for_source(
            source,
            source_namespaces=source_namespaces,
            target_root=target_root,
        )
        if target in seen_targets:
            raise RelocationError(f"duplicate target mapping: {target}")
        seen_targets.add(target)
        if target.exists() or target.is_symlink():
            raise RelocationError(f"target already exists: {target}")
        identity = _tree_identity(source)
        if int(identity["root_device"]) != target_device:
            raise RelocationError(
                f"cross-device relocation is forbidden: {source} -> {target}"
            )
        entries.append(
            {
                "source": str(source),
                "target": str(target),
                "schema": item["schema"],
                "episode_count": item["episode_count"],
                "identity": identity,
            }
        )

    plan: dict[str, Any] = {
        "schema": "dexteleop_dataset_relocation_plan_v1",
        "transaction_id": uuid.uuid4().hex,
        "created_at_unix_ns": time.time_ns(),
        "scan_root": str(scan_root),
        "target_root": str(target_root),
        "target_device": target_device,
        "source_namespaces": [str(path) for path in source_namespaces],
        "entry_count": len(entries),
        "entries": entries,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "open_read_fds": [item for item in open_fds if item["writable"] is False],
    }
    plan["plan_hash"] = _sha256_bytes(_canonical_json_bytes(plan))
    if output.exists() or output.is_symlink():
        raise RelocationError(f"refusing to overwrite plan: {output}")
    _write_json_atomic(output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


def _load_and_verify_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    expected_hash = str(plan.get("plan_hash") or "")
    unsigned = dict(plan)
    unsigned.pop("plan_hash", None)
    actual_hash = _sha256_bytes(_canonical_json_bytes(unsigned))
    if not expected_hash or actual_hash != expected_hash:
        raise RelocationError(
            f"plan hash mismatch: expected={expected_hash!r} actual={actual_hash!r}"
        )
    if plan.get("schema") != "dexteleop_dataset_relocation_plan_v1":
        raise RelocationError(f"unsupported plan schema: {plan.get('schema')!r}")
    return plan


def _verify_entry_at_target(entry: dict[str, Any]) -> None:
    target = Path(entry["target"])
    identity = _tree_identity(target)
    if identity != entry["identity"]:
        raise RelocationError(f"tree identity changed after relocation: {target}")


def _verify_complete(plan: dict[str, Any]) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        source = Path(entry["source"])
        target = Path(entry["target"])
        if not source.is_symlink():
            raise RelocationError(f"source compatibility path is not a symlink: {source}")
        if source.resolve(strict=True) != target.resolve(strict=True):
            raise RelocationError(f"source symlink target mismatch: {source} -> {target}")
        _verify_entry_at_target(entry)
        if _sha256_file(source / "meta" / "info.json") != entry["identity"]["info_sha256"]:
            raise RelocationError(f"info.json differs through compatibility path: {source}")
        if (
            _sha256_file(source / "meta" / "episodes.jsonl")
            != entry["identity"]["episodes_sha256"]
        ):
            raise RelocationError(f"episodes.jsonl differs through compatibility path: {source}")
        verified.append(
            {
                "source": str(source),
                "target": str(target),
                "root_inode": target.stat().st_ino,
            }
        )
    return {
        "verified_at_unix_ns": time.time_ns(),
        "verified_count": len(verified),
        "entries": verified,
    }


def _rollback(
    *,
    plan: dict[str, Any],
    moved_targets: list[Path],
    created_symlinks: list[Path],
) -> list[str]:
    errors: list[str] = []
    expected_targets = {
        Path(entry["source"]): Path(entry["target"]) for entry in plan["entries"]
    }
    for source in reversed(created_symlinks):
        try:
            if not source.is_symlink():
                raise RelocationError(f"rollback source is not a symlink: {source}")
            if Path(os.readlink(source)) != expected_targets[source]:
                raise RelocationError(f"rollback symlink target changed: {source}")
            source.unlink()
            _fsync_directory(source.parent)
        except Exception as exc:
            errors.append(f"remove symlink {source}: {type(exc).__name__}: {exc}")
    entry_by_target = {Path(entry["target"]): entry for entry in plan["entries"]}
    for target in reversed(moved_targets):
        entry = entry_by_target[target]
        source = Path(entry["source"])
        try:
            if source.exists() or source.is_symlink():
                raise RelocationError(f"rollback source is occupied: {source}")
            if target.is_symlink() or not target.is_dir():
                raise RelocationError(f"rollback target is not a directory: {target}")
            os.rename(target, source)
            _fsync_directory(target.parent)
            _fsync_directory(source.parent)
            if _tree_identity(source) != entry["identity"]:
                raise RelocationError(f"rollback identity mismatch: {source}")
        except Exception as exc:
            errors.append(f"restore {target}: {type(exc).__name__}: {exc}")
    return errors


def _apply(args: argparse.Namespace) -> None:
    if not args.confirm_atomic_relocation:
        raise RelocationError("--confirm-atomic-relocation is required")
    plan_path = Path(args.plan).expanduser().absolute()
    plan = _load_and_verify_plan(plan_path)
    target_root = Path(plan["target_root"])
    _validate_target_root(target_root, allow_migration_dir=True)
    entries = list(plan["entries"])
    if len(entries) != int(plan["entry_count"]):
        raise RelocationError("plan entry_count mismatch")
    sources = tuple(Path(entry["source"]) for entry in entries)
    active = _active_processes()
    if active:
        raise RelocationError(f"active collection/postprocess processes: {active}")
    open_fds = _open_dataset_fds(sources)
    writable_fds = [item for item in open_fds if item["writable"] is not False]
    if writable_fds:
        raise RelocationError(f"dataset file descriptors may be writable: {writable_fds}")

    for entry in entries:
        source = Path(entry["source"])
        target = Path(entry["target"])
        if source.is_symlink() or not source.is_dir():
            raise RelocationError(f"source changed since plan: {source}")
        if target.exists() or target.is_symlink():
            raise RelocationError(f"target appeared after plan: {target}")
        if _tree_identity(source) != entry["identity"]:
            raise RelocationError(f"source identity changed since plan: {source}")
        if source.stat().st_dev != int(plan["target_device"]):
            raise RelocationError(f"source device changed since plan: {source}")

    journal_path = plan_path.with_name("journal.json")
    if journal_path.exists() or journal_path.is_symlink():
        raise RelocationError(f"refusing to overwrite journal: {journal_path}")
    journal: dict[str, Any] = {
        "schema": "dexteleop_dataset_relocation_journal_v1",
        "transaction_id": plan["transaction_id"],
        "plan_hash": plan["plan_hash"],
        "state": "applying",
        "started_at_unix_ns": time.time_ns(),
        "moved_targets": [],
        "created_symlinks": [],
        "open_read_fds": [item for item in open_fds if item["writable"] is False],
    }
    _write_json_atomic(journal_path, journal)
    moved_targets: list[Path] = []
    created_symlinks: list[Path] = []
    try:
        for entry in entries:
            source = Path(entry["source"])
            target = Path(entry["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise RelocationError(f"target collision during apply: {target}")
            os.rename(source, target)
            _fsync_directory(source.parent)
            _fsync_directory(target.parent)
            moved_targets.append(target)
            journal["moved_targets"] = [str(path) for path in moved_targets]
            _write_json_atomic(journal_path, journal)

        for entry in entries:
            _verify_entry_at_target(entry)

        for entry in entries:
            source = Path(entry["source"])
            target = Path(entry["target"])
            if source.exists() or source.is_symlink():
                raise RelocationError(f"source path unexpectedly occupied: {source}")
            source.symlink_to(target, target_is_directory=True)
            _fsync_directory(source.parent)
            created_symlinks.append(source)
            journal["created_symlinks"] = [str(path) for path in created_symlinks]
            _write_json_atomic(journal_path, journal)

        verification = _verify_complete(plan)
        journal.update(
            {
                "state": "complete",
                "completed_at_unix_ns": time.time_ns(),
                "verification": verification,
            }
        )
        _write_json_atomic(journal_path, journal)
        receipt_path = plan_path.with_name("receipt.json")
        receipt = {
            "schema": "dexteleop_dataset_relocation_receipt_v1",
            "transaction_id": plan["transaction_id"],
            "plan_hash": plan["plan_hash"],
            "journal_sha256": _sha256_file(journal_path),
            "entry_count": len(entries),
            "completed_at_unix_ns": journal["completed_at_unix_ns"],
            "verification": verification,
        }
        if receipt_path.exists() or receipt_path.is_symlink():
            raise RelocationError(f"refusing to overwrite receipt: {receipt_path}")
        _write_json_atomic(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    except BaseException as exc:
        rollback_errors = _rollback(
            plan=plan,
            moved_targets=moved_targets,
            created_symlinks=created_symlinks,
        )
        journal.update(
            {
                "state": "rolled_back" if not rollback_errors else "rollback_failed",
                "failed_at_unix_ns": time.time_ns(),
                "failure": f"{type(exc).__name__}: {exc}",
                "rollback_errors": rollback_errors,
            }
        )
        _write_json_atomic(journal_path, journal)
        if rollback_errors:
            raise RelocationError(
                f"relocation failed and rollback was incomplete: {exc}; {rollback_errors}"
            ) from exc
        raise


def _verify(args: argparse.Namespace) -> None:
    plan = _load_and_verify_plan(Path(args.plan).expanduser().absolute())
    result = _verify_complete(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Freeze a read-only relocation plan.")
    plan.add_argument("--scan-root", required=True)
    plan.add_argument("--target-root", required=True)
    plan.add_argument("--source-namespace", action="append", required=True)
    plan.add_argument("--expected-count", type=int, required=True)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=_plan)

    apply = subparsers.add_parser("apply", help="Apply a frozen relocation plan.")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--confirm-atomic-relocation", action="store_true")
    apply.set_defaults(func=_apply)

    verify = subparsers.add_parser("verify", help="Verify a completed relocation.")
    verify.add_argument("--plan", required=True)
    verify.set_defaults(func=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        args.func(args)
    except RelocationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
