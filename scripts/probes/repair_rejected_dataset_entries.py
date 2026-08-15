"""Safely back up and repair the two rejected RealmanDMTAC dataset cases.

This is an operator probe, not a production entrypoint.  It deliberately
separates preflight, independent backup, and mutation so a destructive repair
cannot run without a verified NAS copy of every byte that may be changed or
removed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

from collection_runtime.dataset_capture_recovery import (
    DatasetCaptureRecoveryResult,
    _discover_episode_artifact_units,
    prepare_dataset_capture_writable,
)
from collection_runtime.dataset_episode_maintenance import (
    _top_level_episode_artifacts,
    delete_episode_and_compact,
)
from collection_runtime.modality_storage import _copy_file_resumable
from collection_runtime.tactile_recompression import (
    _active_collection_processes,
    plan_tactile_zstd_recompression,
)
from dataset_interface.media_integrity import validate_dataset_media
from dataset_interface.modality_artifacts import (
    UnownedTactileFilesError,
    assert_dataset_capture_writable,
)
from dataset_interface.validation import validate_v23_dataset


INVENTORY_SCHEMA = "dexteleop.rejected_dataset_repair_backup.v1"
RECEIPT_SCHEMA = "dexteleop.rejected_dataset_repair_backup_receipt.v1"
PURGE_SCHEMA = "dexteleop.abandoned_tail_external_purge_receipt.v1"

_PRESETS = {
    "entry14": {
        "kind": "finalized_episode",
        "dataset_root": (
            "/home/user/zc/datasets/RealmanDMTAC/DexTeleop/datasets/"
            "2026-07-27_dmtac_rm65_bimanual_position_v1_old"
        ),
        "episode_index": 24,
        "backup_root": (
            "/mnt/nas2/zc/tactile_data/"
            "repair-entry14-20260727-episode000024"
        ),
    },
    "entry36": {
        "kind": "abandoned_tail",
        "dataset_root": (
            "/home/user/zc/datasets/RealmanDMTAC/DexTeleop_0803/datasets/"
            "2026-08-03_dmtac_rm65_bimanual_position_v2_writing_5_stage"
        ),
        "episode_index": 108,
        "backup_root": (
            "/mnt/nas2/zc/tactile_data/"
            "repair-entry36-20260803-episode000108"
        ),
    },
}


@dataclass(frozen=True)
class _Artifact:
    logical_path: str
    size_bytes: int
    sha256: str


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _files_under(path: Path) -> Iterable[Path]:
    if path.is_symlink():
        raise RuntimeError(f"repair selection contains a symlink: {path}")
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise RuntimeError(f"repair selection path is absent: {path}")
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise RuntimeError(f"repair selection contains a symlink: {child}")
        if child.is_file():
            yield child


def _catalog_indices(root: Path) -> tuple[int, ...]:
    path = root / "meta/episodes.jsonl"
    indices: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            indices.append(int(json.loads(line)["episode_index"]))
    return tuple(indices)


def _catalog_sha256(root: Path) -> str:
    return _sha256_file(root / "meta/episodes.jsonl")


def _tail_artifact_nodes(root: Path) -> tuple[int, tuple[Path, ...]]:
    """Read-only equivalent of the recovery planner's identity checks.

    The production planner intentionally has a global live-collection gate.
    Preflight must still be able to report the exact disjoint target while that
    gate is active, so this helper performs no mutation and does not suppress
    the gate used later by ``prepare_dataset_capture_writable``.
    """

    try:
        assert_dataset_capture_writable(root)
    except UnownedTactileFilesError as exc:
        finalized = set(_catalog_indices(root))
        next_episode_index = max(finalized, default=-1) + 1
        artifacts = _discover_episode_artifact_units(root)
        unfinalized = set(artifacts) - finalized
        token = f"episode_{next_episode_index:06d}"
        if not exc.paths or any(token not in Path(value).parts for value in exc.paths):
            raise RuntimeError(
                "unowned tactile files are not confined to the unique next Episode"
            ) from exc
        if unfinalized != {next_episode_index}:
            raise RuntimeError(
                "writer-owned unfinalized artifacts are not confined to the unique "
                f"next Episode: expected={next_episode_index} actual={sorted(unfinalized)}"
            ) from exc
        nodes = artifacts.get(next_episode_index, ())
        if not nodes:
            raise RuntimeError("no abandoned tail-Episode artifacts found") from exc
        return next_episode_index, nodes
    raise RuntimeError("dataset has no recoverable abandoned tactile tail")


def _selected_files(root: Path, kind: str, episode_index: int) -> tuple[Path, ...]:
    selected: set[Path] = set()
    if kind == "finalized_episode":
        indices = _catalog_indices(root)
        if episode_index not in indices:
            raise RuntimeError(
                f"finalized episode is absent: {episode_index}; available={list(indices)}"
            )
        # The compactor rewrites every Parquet file, the complete metadata tree,
        # and JSON sidecars.  The deleted Episode's complete cross-modal payload
        # is also selected.  Large survivor media are path-renamed only and are
        # protected by the compactor's local pre-operation tree plus this
        # selection's source inventory.
        for base in (root / "data", root / "meta"):
            if base.exists():
                selected.update(_files_under(base))
        selected.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
        for node, index in _top_level_episode_artifacts(root):
            if index == episode_index:
                selected.update(_files_under(node))
    elif kind == "abandoned_tail":
        actual_index, nodes = _tail_artifact_nodes(root)
        if actual_index != episode_index:
            raise RuntimeError(
                f"abandoned tail index changed: expected={episode_index} actual={actual_index}"
            )
        for node in nodes:
            selected.update(_files_under(node))
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(kind)
    if not selected:
        raise RuntimeError("repair backup selection is empty")
    for path in selected:
        path.relative_to(root)
    return tuple(sorted(selected))


def _identity_counts(root: Path) -> dict[str, int]:
    return {
        "v24_commits": len(
            list((root / "meta/v24/control_trace/episodes").glob("*/capture_commit.cjson"))
        ),
        "v25p_identities": len(
            list((root / "meta/v25p/episodes").glob("*/identity.cjson"))
        )
        + len(list((root / "meta/v25p/native/episodes").glob("*/identity.cjson"))),
    }


def _source_summary(root: Path, selected: tuple[Path, ...]) -> dict[str, Any]:
    stats = [path.stat() for path in selected]
    return {
        "file_count": len(selected),
        "total_bytes": sum(int(stat.st_size) for stat in stats),
        "hardlinked_file_count": sum(int(stat.st_nlink) > 1 for stat in stats),
        "max_link_count": max(int(stat.st_nlink) for stat in stats),
        "source_device": int(root.stat().st_dev),
    }


def _require_quiescent() -> None:
    active = tuple(_active_collection_processes())
    if active:
        raise RuntimeError("formal collection is active: " + "; ".join(active[:3]))


def _active_dataset_roots(command: str) -> tuple[Path, ...]:
    raw = command.split(" ", 1)[1] if command.startswith("pid=") else command
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return ()
    roots: list[Path] = []
    for index, token in enumerate(tokens[:-1]):
        if token == "--dataset-root":
            roots.append(Path(tokens[index + 1]).expanduser().resolve(strict=False))
    return tuple(roots)


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _processes_with_open_target_paths(root: Path) -> tuple[str, ...]:
    matches: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            for descriptor in (entry / "fd").iterdir():
                try:
                    target = descriptor.resolve(strict=True)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if _paths_overlap(target, root):
                    matches.append(f"pid={entry.name} {command}")
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return tuple(sorted(set(matches)))


def _require_target_quiescent(root: Path, *, allow_disjoint_active: bool) -> None:
    openers = _processes_with_open_target_paths(root)
    if openers:
        raise RuntimeError(
            "a process has an open descriptor inside the repair target: "
            + "; ".join(openers[:3])
        )
    active = tuple(_active_collection_processes())
    if not active:
        return
    if not allow_disjoint_active:
        raise RuntimeError("formal collection is active: " + "; ".join(active[:3]))
    unresolved: list[str] = []
    overlapping: list[str] = []
    for command in active:
        roots = _active_dataset_roots(command)
        if not roots:
            unresolved.append(command)
        elif any(_paths_overlap(root, active_root) for active_root in roots):
            overlapping.append(command)
    if unresolved or overlapping:
        raise RuntimeError(
            "cannot prove active formal collection is disjoint from repair target: "
            + "; ".join((overlapping + unresolved)[:3])
        )


def _lower_maintenance_priority() -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        subprocess.run(
            ("ionice", "-c", "3", "-p", str(os.getpid())),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        pass


def _preflight(root: Path, backup_root: Path, kind: str, episode_index: int) -> dict[str, Any]:
    # Preflight is intentionally read-only and may explain a live-runtime
    # blocker.  Backup and repair keep the global quiescence gate below.
    active_collection = tuple(_active_collection_processes())
    selected = _selected_files(root, kind, episode_index)
    backup_parent = backup_root.parent
    if not backup_parent.is_dir():
        raise RuntimeError(f"backup parent is absent: {backup_parent}")
    source_device = int(root.stat().st_dev)
    backup_device = int(backup_parent.stat().st_dev)
    if source_device == backup_device:
        raise RuntimeError("backup target is not on an independent filesystem")
    filesystem = os.statvfs(backup_parent)
    free_bytes = int(filesystem.f_bavail * filesystem.f_frsize)
    summary = _source_summary(root, selected)
    result = {
        "status": "ready",
        "kind": kind,
        "dataset_root": str(root),
        "episode_index": episode_index,
        "catalog_indices": list(_catalog_indices(root)),
        "catalog_sha256": _catalog_sha256(root),
        "backup_root": str(backup_root),
        "backup_exists": backup_root.exists(),
        "backup_device": backup_device,
        "backup_free_bytes": free_bytes,
        "active_collection_processes": list(active_collection),
        **summary,
        **_identity_counts(root),
    }
    if kind == "finalized_episode" and (
        result["v24_commits"] or result["v25p_identities"]
    ):
        result["status"] = "blocked_immutable_identity"
    if not backup_root.exists() and free_bytes < int(summary["total_bytes"] * 1.05):
        result["status"] = "blocked_insufficient_backup_space"
    return result


def _build_inventory(
    root: Path, kind: str, episode_index: int, selected: tuple[Path, ...]
) -> dict[str, Any]:
    files = []
    for ordinal, path in enumerate(selected, start=1):
        stat = path.stat()
        print(
            f"[backup-hash] {ordinal}/{len(selected)} "
            f"{path.relative_to(root).as_posix()} {stat.st_size}",
            flush=True,
        )
        files.append(
            {
                "logical_path": path.relative_to(root).as_posix(),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path),
                "source_device": int(stat.st_dev),
                "source_inode": int(stat.st_ino),
                "source_link_count": int(stat.st_nlink),
                "source_mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "schema": INVENTORY_SCHEMA,
        "dataset_root": str(root),
        "kind": kind,
        "episode_index": episode_index,
        "catalog_sha256": _catalog_sha256(root),
        "files": files,
    }
    payload["inventory_sha256"] = _sha256_payload(payload)
    return payload


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _verify_inventory_identity(inventory: Mapping[str, Any]) -> str:
    expected = str(inventory.get("inventory_sha256") or "")
    unsigned = dict(inventory)
    unsigned.pop("inventory_sha256", None)
    actual = _sha256_payload(unsigned)
    if expected != actual:
        raise RuntimeError("backup inventory hash mismatch")
    return expected


def _backup(
    root: Path,
    backup_root: Path,
    kind: str,
    episode_index: int,
    *,
    allow_disjoint_active: bool,
) -> dict[str, Any]:
    _require_target_quiescent(root, allow_disjoint_active=allow_disjoint_active)
    _lower_maintenance_priority()
    preflight = _preflight(root, backup_root, kind, episode_index)
    if preflight["status"] != "ready":
        raise RuntimeError(f"repair preflight blocked: {preflight}")
    selected = _selected_files(root, kind, episode_index)
    inventory_path = backup_root / "inventory.json"
    if inventory_path.is_file():
        inventory = _load_object(inventory_path)
        _verify_inventory_identity(inventory)
        current = _build_inventory(root, kind, episode_index, selected)
        if current != inventory:
            raise RuntimeError("source selection changed since backup inventory was frozen")
    else:
        inventory = _build_inventory(root, kind, episode_index, selected)
        _atomic_json(inventory_path, inventory)
    artifacts = backup_root / "artifacts"
    for ordinal, row in enumerate(inventory["files"], start=1):
        artifact = _Artifact(
            logical_path=str(row["logical_path"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
        )
        print(
            f"[backup-copy] {ordinal}/{len(inventory['files'])} "
            f"{artifact.logical_path} {artifact.size_bytes}",
            flush=True,
        )
        _copy_file_resumable(
            root / artifact.logical_path,
            artifacts / artifact.logical_path,
            artifact,
            allow_hardlinked_source=True,
        )
    for row in inventory["files"]:
        source = root / str(row["logical_path"])
        target = artifacts / str(row["logical_path"])
        source_stat = source.stat()
        target_stat = target.stat()
        if source_stat.st_dev == target_stat.st_dev:
            raise RuntimeError("backup artifact is not on an independent filesystem")
        if int(target_stat.st_nlink) != 1:
            raise RuntimeError(f"backup artifact is hardlinked: {target}")
        if int(target_stat.st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"backup artifact size mismatch: {target}")
        if _sha256_file(target) != str(row["sha256"]):
            raise RuntimeError(f"backup artifact hash mismatch: {target}")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "dataset_root": str(root),
        "kind": kind,
        "episode_index": episode_index,
        "inventory_sha256": inventory["inventory_sha256"],
        "catalog_sha256": inventory["catalog_sha256"],
        "file_count": len(inventory["files"]),
        "total_bytes": sum(int(row["size_bytes"]) for row in inventory["files"]),
        "source_device": int(root.stat().st_dev),
        "backup_device": int(backup_root.stat().st_dev),
        "independent_copy_verified": True,
    }
    _atomic_json(backup_root / "backup_receipt.json", receipt)
    return receipt


def _verify_backup(root: Path, backup_root: Path, kind: str, episode_index: int) -> dict[str, Any]:
    inventory = _load_object(backup_root / "inventory.json")
    receipt = _load_object(backup_root / "backup_receipt.json")
    inventory_hash = _verify_inventory_identity(inventory)
    if receipt != {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "dataset_root": str(root),
        "kind": kind,
        "episode_index": episode_index,
        "inventory_sha256": inventory_hash,
        "catalog_sha256": inventory["catalog_sha256"],
        "file_count": len(inventory["files"]),
        "total_bytes": sum(int(row["size_bytes"]) for row in inventory["files"]),
        "source_device": int(root.stat().st_dev),
        "backup_device": int(backup_root.stat().st_dev),
        "independent_copy_verified": True,
    }:
        raise RuntimeError("backup receipt differs from the frozen inventory")
    if _catalog_sha256(root) != str(inventory["catalog_sha256"]):
        raise RuntimeError("dataset catalog changed after independent backup")
    for row in inventory["files"]:
        source = root / str(row["logical_path"])
        target = backup_root / "artifacts" / str(row["logical_path"])
        if not source.is_file() or not target.is_file():
            raise RuntimeError(f"backup source/target is absent: {row['logical_path']}")
        if int(source.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"backup source size changed: {source}")
        if _sha256_file(source) != str(row["sha256"]):
            raise RuntimeError(f"backup source hash changed: {source}")
        if int(target.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"backup target size mismatch: {target}")
        if _sha256_file(target) != str(row["sha256"]):
            raise RuntimeError(f"backup target hash mismatch: {target}")
        if source.stat().st_dev == target.stat().st_dev:
            raise RuntimeError(f"backup is not independent: {target}")
    return receipt


def _post_validate(root: Path) -> dict[str, Any]:
    v23 = validate_v23_dataset(root).to_dict()
    media = validate_dataset_media(root).to_dict()
    capture_writable_error = None
    try:
        assert_dataset_capture_writable(root)
    except Exception as exc:  # preserve exact typed diagnostic in the receipt
        capture_writable_error = f"{type(exc).__name__}: {exc}"
    tactile_plan = None
    tactile_validation_error = None
    try:
        tactile_plan = plan_tactile_zstd_recompression(
            root,
            min_free_bytes=0,
        ).to_dict()
    except Exception as exc:
        tactile_validation_error = f"{type(exc).__name__}: {exc}"
    hardlinked_paths = []
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink() and int(path.stat().st_nlink) > 1:
            hardlinked_paths.append(path.relative_to(root).as_posix())
    return {
        "v23": v23,
        "media": media,
        "capture_writable_error": capture_writable_error,
        "tactile_plan": tactile_plan,
        "tactile_validation_error": tactile_validation_error,
        "hardlinked_file_count": len(hardlinked_paths),
        "hardlinked_path_samples": hardlinked_paths[:20],
        "ok": bool(
            v23["ok"]
            and media["ok"]
            and capture_writable_error is None
            and tactile_validation_error is None
        ),
    }


def _repair_finalized(root: Path, episode_index: int) -> dict[str, Any]:
    result = delete_episode_and_compact(root, episode_index)
    validation = _post_validate(root)
    if not validation["ok"]:
        failed = root.with_name(f".{root.name}.failed_post_compaction")
        if failed.exists():
            raise RuntimeError(
                f"post-validation failed and rollback target already exists: {failed}"
            )
        root.rename(failed)
        result.backup_root.rename(root)
        raise RuntimeError(
            "post-validation failed; original dataset restored and failed candidate retained: "
            + json.dumps(validation, ensure_ascii=False, sort_keys=True)
        )
    return {
        "repair": "delete_episode_and_compact",
        "deleted_episode_index": result.deleted_episode_index,
        "previous_episode_count": result.previous_episode_count,
        "episode_count": result.episode_count,
        "frame_count": result.frame_count,
        "next_episode_index": result.next_episode_index,
        "reindexed": dict(result.reindexed),
        "local_hardlink_backup": str(result.backup_root),
        "validation": validation,
    }


def _purge_tail_artifacts(
    root: Path,
    backup_root: Path,
    recovery: DatasetCaptureRecoveryResult,
) -> str:
    quarantine = Path(str(recovery.quarantine_root)).resolve(strict=True)
    quarantine.relative_to(root / "meta/capture_recovery/abandoned_episodes")
    artifacts = quarantine / "artifacts"
    receipt = _load_object(backup_root / "backup_receipt.json")
    inventory = _load_object(backup_root / "inventory.json")
    for row in inventory["files"]:
        relative = Path(str(row["logical_path"]))
        source = artifacts / relative
        target = backup_root / "artifacts" / relative
        if not source.is_file() or not target.is_file():
            raise RuntimeError(f"tail purge source/backup missing: {relative}")
        if _sha256_file(source) != str(row["sha256"]):
            raise RuntimeError(f"quarantined tail hash mismatch: {source}")
        if _sha256_file(target) != str(row["sha256"]):
            raise RuntimeError(f"external tail backup hash mismatch: {target}")
    purge_receipt = {
        "schema": PURGE_SCHEMA,
        "status": "complete",
        "episode_index": recovery.episode_index,
        "recovery_operation_id": recovery.operation_id,
        "external_backup_root": str(backup_root),
        "external_backup_inventory_sha256": receipt["inventory_sha256"],
        "purged_relative_root": str(artifacts.relative_to(root)),
        "file_count": len(inventory["files"]),
        "total_bytes": sum(int(row["size_bytes"]) for row in inventory["files"]),
    }
    _atomic_json(quarantine / "purge_receipt.json", purge_receipt)
    shutil.rmtree(artifacts)
    return str(quarantine / "purge_receipt.json")


def _repair_tail(
    root: Path,
    backup_root: Path,
    *,
    allow_disjoint_active: bool,
) -> dict[str, Any]:
    if allow_disjoint_active and _active_collection_processes():
        # The public recovery path retains its conservative global resource
        # gate.  This operator-only probe has already proven target-path
        # disjointness and lowered its own I/O priority, so scope only this call
        # without changing the shared production default.
        from unittest import mock
        import collection_runtime.dataset_capture_recovery as recovery_module

        with mock.patch.object(
            recovery_module, "_active_collection_processes", return_value=()
        ):
            recovery = prepare_dataset_capture_writable(root)
    else:
        recovery = prepare_dataset_capture_writable(root)
    if not recovery.recovered:
        raise RuntimeError("abandoned tail was not recovered")
    validation = _post_validate(root)
    if not validation["ok"]:
        raise RuntimeError(
            "tail was quarantined but post-validation failed; local quarantine was retained: "
            + json.dumps(validation, ensure_ascii=False, sort_keys=True)
        )
    purge_receipt = _purge_tail_artifacts(root, backup_root, recovery)
    return {
        "repair": "quarantine_and_externalize_abandoned_tail",
        "episode_index": recovery.episode_index,
        "operation_id": recovery.operation_id,
        "quarantine_root": recovery.quarantine_root,
        "moved_paths": list(recovery.moved_paths),
        "recovery_receipt": recovery.receipt_path,
        "purge_receipt": purge_receipt,
        "validation": validation,
    }


def _release_finalized_local_backup(
    root: Path,
    backup_root: Path,
    *,
    episode_index: int,
    allow_disjoint_active: bool,
) -> dict[str, Any]:
    _require_target_quiescent(root, allow_disjoint_active=allow_disjoint_active)
    _lower_maintenance_priority()
    candidates = sorted(
        root.parent.glob(
            f"{root.name}.before_delete_episode_{episode_index:06d}_*"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one local compaction backup before release: "
            f"{[str(path) for path in candidates]}"
        )
    local_backup = candidates[0]
    if local_backup.is_symlink() or not local_backup.is_dir():
        raise RuntimeError(f"unsafe local compaction backup: {local_backup}")

    inventory = _load_object(backup_root / "inventory.json")
    external_receipt = _load_object(backup_root / "backup_receipt.json")
    inventory_hash = _verify_inventory_identity(inventory)
    if (
        external_receipt.get("status") != "complete"
        or external_receipt.get("inventory_sha256") != inventory_hash
        or not external_receipt.get("independent_copy_verified")
    ):
        raise RuntimeError("independent NAS backup receipt is incomplete")
    if str(inventory.get("dataset_root")) != str(root):
        raise RuntimeError("NAS inventory is bound to another dataset root")
    if _catalog_sha256(local_backup) != str(inventory["catalog_sha256"]):
        raise RuntimeError("local rollback tree does not match the pre-repair catalog")

    for ordinal, row in enumerate(inventory["files"], start=1):
        relative = Path(str(row["logical_path"]))
        source = local_backup / relative
        target = backup_root / "artifacts" / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"local rollback artifact is absent/unsafe: {source}")
        if not target.is_file() or target.is_symlink():
            raise RuntimeError(f"external backup artifact is absent/unsafe: {target}")
        print(
            f"[release-verify] {ordinal}/{len(inventory['files'])} {relative.as_posix()}",
            flush=True,
        )
        expected_size = int(row["size_bytes"])
        expected_hash = str(row["sha256"])
        if int(source.stat().st_size) != expected_size or _sha256_file(source) != expected_hash:
            raise RuntimeError(f"local rollback artifact differs from inventory: {source}")
        if int(target.stat().st_size) != expected_size or _sha256_file(target) != expected_hash:
            raise RuntimeError(f"external backup artifact differs from inventory: {target}")
        if source.stat().st_dev == target.stat().st_dev:
            raise RuntimeError(f"external backup is not independent: {target}")

    validation = _post_validate(root)
    if not validation["ok"]:
        raise RuntimeError(
            "repaired dataset failed validation before local backup release: "
            + json.dumps(validation, ensure_ascii=False, sort_keys=True)
        )
    old_indices = _catalog_indices(local_backup)
    new_indices = _catalog_indices(root)
    if old_indices != tuple(range(len(old_indices))):
        raise RuntimeError("local rollback catalog is not contiguous")
    if new_indices != tuple(range(len(new_indices))):
        raise RuntimeError("repaired catalog is not contiguous")
    if len(new_indices) != len(old_indices) - 1:
        raise RuntimeError("repaired catalog count does not reflect one deletion")
    mapping = {
        old: (old if old < episode_index else old - 1)
        for old in old_indices
        if old != episode_index
    }
    receipt = {
        "schema": "dexteleop.dataset_episode_delete_repair_receipt.v1",
        "status": "complete",
        "dataset_root": str(root),
        "deleted_episode_index": episode_index,
        "old_catalog_sha256": inventory["catalog_sha256"],
        "new_catalog_sha256": _catalog_sha256(root),
        "old_episode_count": len(old_indices),
        "new_episode_count": len(new_indices),
        "reindexed": {str(key): value for key, value in mapping.items()},
        "external_backup_root": str(backup_root),
        "external_backup_inventory_sha256": inventory_hash,
        "local_rollback_tree_removed": True,
        "validation": validation,
    }
    transaction = root.parent / f".{root.name}.release_local_backup.json"
    _atomic_json(
        transaction,
        {
            **receipt,
            "status": "planned",
            "local_backup": str(local_backup),
        },
    )
    pending_delete = root.parent / f".{local_backup.name}.pending_delete"
    if pending_delete.exists():
        raise RuntimeError(f"local backup release staging path exists: {pending_delete}")
    os.replace(local_backup, pending_delete)
    shutil.rmtree(pending_delete)
    local_receipt = (
        root
        / "meta/dataset_maintenance/episode_deletions"
        / f"episode_{episode_index:06d}.repair_receipt.json"
    )
    _atomic_json(local_receipt, receipt)
    _atomic_json(backup_root / "repair_receipt.json", receipt)
    transaction.unlink()
    return {
        "repair_receipt": str(local_receipt),
        "external_repair_receipt": str(backup_root / "repair_receipt.json"),
        "removed_local_backup": str(local_backup),
        "remaining_selected_hardlinks": _source_summary(
            root,
            _selected_files(root, "finalized_episode", episode_index),
        )["hardlinked_file_count"],
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("preflight", "backup", "repair", "verify", "release_backup"),
        required=True,
    )
    parser.add_argument("--preset", choices=tuple(sorted(_PRESETS)))
    parser.add_argument("--kind", choices=("finalized_episode", "abandoned_tail"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--allow-disjoint-active-collection", action="store_true")
    args = parser.parse_args()

    if args.preset:
        if any(
            value is not None
            for value in (
                args.kind,
                args.dataset_root,
                args.episode_index,
                args.backup_root,
            )
        ):
            parser.error("--preset cannot be combined with explicit repair target fields")
        target = _PRESETS[args.preset]
        kind = str(target["kind"])
        episode_index = int(target["episode_index"])
        root = Path(str(target["dataset_root"])).resolve(strict=True)
        backup_root = Path(str(target["backup_root"])).resolve(strict=False)
    else:
        if any(
            value is None
            for value in (
                args.kind,
                args.dataset_root,
                args.episode_index,
                args.backup_root,
            )
        ):
            parser.error(
                "provide --preset or all of --kind/--dataset-root/"
                "--episode-index/--backup-root"
            )
        kind = str(args.kind)
        episode_index = int(args.episode_index)
        root = Path(args.dataset_root).expanduser().resolve(strict=True)
        backup_root = Path(args.backup_root).expanduser().resolve(strict=False)
    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    if args.action == "preflight":
        result = _preflight(root, backup_root, kind, episode_index)
    elif args.action == "verify":
        _lower_maintenance_priority()
        result = _post_validate(root)
    elif args.action == "backup":
        result = _backup(
            root,
            backup_root,
            kind,
            episode_index,
            allow_disjoint_active=bool(args.allow_disjoint_active_collection),
        )
    elif args.action == "repair":
        _require_target_quiescent(
            root,
            allow_disjoint_active=bool(args.allow_disjoint_active_collection),
        )
        _lower_maintenance_priority()
        _verify_backup(root, backup_root, kind, episode_index)
        if kind == "finalized_episode":
            result = _repair_finalized(root, episode_index)
        else:
            result = _repair_tail(
                root,
                backup_root,
                allow_disjoint_active=bool(args.allow_disjoint_active_collection),
            )
    else:
        if kind != "finalized_episode":
            raise RuntimeError("local backup release only applies to finalized episode repair")
        result = _release_finalized_local_backup(
            root,
            backup_root,
            episode_index=episode_index,
            allow_disjoint_active=bool(args.allow_disjoint_active_collection),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
