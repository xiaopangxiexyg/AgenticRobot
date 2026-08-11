"""Recover provably abandoned tail-Episode artifacts before capture append.

The read-side admission check remains strictly non-mutating.  This module owns
the collection-time recovery transaction: it only quarantines artifacts for
the unique next legacy Episode, never deletes them, and publishes a receipt so
the operation is reversible and auditable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping
import uuid
import warnings

from dataset_interface.modality_artifacts import (
    UnownedTactileFilesError,
    assert_dataset_capture_writable,
)

from .process_quiescence import (
    active_formal_collection_processes as _active_collection_processes,
)


CAPTURE_RECOVERY_SCHEMA_V1 = "dexteleop.dataset_capture_recovery.v1"
CAPTURE_RECOVERY_RELATIVE_ROOT = Path("meta/capture_recovery")
_ACTIVE_TRANSACTION_FILENAME = "active_transaction.json"
_OWNED_ARTIFACT_ROOTS = ("data", "videos", "tactile", "wrist_depth")
_EPISODE_TOKEN_RE = re.compile(r"^\.?episode_(\d{6})(?:\..+)?$")


class DatasetCaptureRecoveryError(RuntimeError):
    """Raised when an abandoned capture cannot be recovered without guessing."""


@dataclass(frozen=True)
class DatasetCaptureRecoveryResult:
    recovered: bool
    episode_index: int | None = None
    operation_id: str | None = None
    quarantine_root: str | None = None
    moved_paths: tuple[str, ...] = ()
    receipt_path: str | None = None


def _require_no_active_collection() -> None:
    active = _active_collection_processes()
    if active:
        raise DatasetCaptureRecoveryError(
            "formal collection is still active; refusing to quarantine tail artifacts: "
            + "; ".join(active[:3])
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for payload in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if not path.is_dir():
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetCaptureRecoveryError(
            f"cannot read capture recovery transaction {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetCaptureRecoveryError(
            f"capture recovery transaction is not an object: {path}"
        )
    return payload


def _catalog_indices(catalog_path: Path) -> tuple[int, ...]:
    if not catalog_path.is_file():
        return ()
    indices: list[int] = []
    try:
        for line_number, line in enumerate(
            catalog_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"record {line_number} is not an object")
            index = int(record["episode_index"])
            if index < 0:
                raise ValueError(f"record {line_number} has a negative episode index")
            indices.append(index)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetCaptureRecoveryError(
            f"cannot inspect finalized Episode catalog: {exc}"
        ) from exc
    if len(indices) != len(set(indices)):
        raise DatasetCaptureRecoveryError(
            "finalized Episode catalog contains duplicate episode indices"
        )
    return tuple(indices)


def _episode_index_from_part(name: str) -> int | None:
    match = _EPISODE_TOKEN_RE.fullmatch(str(name))
    return None if match is None else int(match.group(1))


def _artifact_episode_unit(
    path: Path,
    *,
    owner_root: Path,
) -> tuple[int, Path] | None:
    relative_parts = path.relative_to(owner_root).parts
    for ordinal, part in enumerate(relative_parts):
        index = _episode_index_from_part(part)
        if index is not None:
            return index, owner_root.joinpath(*relative_parts[: ordinal + 1])
    return None


def _discover_episode_artifact_units(root: Path) -> dict[int, tuple[Path, ...]]:
    """Return top-level writer-owned artifact nodes grouped by Episode index."""

    grouped: dict[int, list[Path]] = {}
    for owner_name in _OWNED_ARTIFACT_ROOTS:
        owner_root = root / owner_name
        if not owner_root.exists():
            continue
        if owner_root.is_symlink() or not owner_root.is_dir():
            raise DatasetCaptureRecoveryError(
                f"capture artifact owner root is unsafe: {owner_root}"
            )
        candidates: set[Path] = set()
        for path in owner_root.rglob("*"):
            if path.is_symlink():
                raise DatasetCaptureRecoveryError(
                    f"capture artifact tree contains a symlink: {path}"
                )
            resolved = _artifact_episode_unit(path, owner_root=owner_root)
            if resolved is None:
                continue
            index, unit = resolved
            if unit not in candidates:
                candidates.add(unit)
                grouped.setdefault(index, []).append(unit)
    return {
        index: tuple(sorted(paths, key=lambda path: path.as_posix()))
        for index, paths in grouped.items()
    }


def _unowned_tactile_indices(paths: tuple[str, ...]) -> set[int]:
    indices: set[int] = set()
    for value in paths:
        parts = Path(value).parts
        matches = [
            index
            for part in parts
            if (index := _episode_index_from_part(part)) is not None
        ]
        if len(matches) != 1:
            raise DatasetCaptureRecoveryError(
                "unowned tactile path is not attributable to one Episode: "
                f"{value}"
            )
        indices.add(matches[0])
    return indices


def _node_summary(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, int(path.stat().st_size)
    file_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise DatasetCaptureRecoveryError(
                f"capture artifact contains a symlink: {child}"
            )
        if child.is_file():
            file_count += 1
            total_bytes += int(child.stat().st_size)
    return file_count, total_bytes


@contextmanager
def _recovery_lock(root: Path) -> Iterator[None]:
    lock_path = root.parent / f".{root.name}.capture-recovery.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DatasetCaptureRecoveryError(
                f"another capture preparation holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _apply_transaction(root: Path, transaction: Mapping[str, Any]) -> DatasetCaptureRecoveryResult:
    _require_no_active_collection()
    if str(transaction.get("schema")) != CAPTURE_RECOVERY_SCHEMA_V1:
        raise DatasetCaptureRecoveryError("unsupported capture recovery transaction schema")
    operation_id = str(transaction.get("operation_id") or "")
    episode_index = int(transaction.get("episode_index", -1))
    quarantine_relative = Path(str(transaction.get("quarantine_relative") or ""))
    expected_quarantine_relative = (
        CAPTURE_RECOVERY_RELATIVE_ROOT
        / "abandoned_episodes"
        / f"episode_{episode_index:06d}_{operation_id}"
    )
    if (
        re.fullmatch(r"[0-9a-f]{32}", operation_id) is None
        or episode_index < 0
        or quarantine_relative.is_absolute()
        or ".." in quarantine_relative.parts
        or quarantine_relative != expected_quarantine_relative
    ):
        raise DatasetCaptureRecoveryError("invalid capture recovery transaction identity")
    catalog_path = root / "meta/episodes.jsonl"
    current_catalog_hash = (
        _sha256_file(catalog_path) if catalog_path.is_file() else None
    )
    if transaction.get("catalog_sha256") != current_catalog_hash:
        raise DatasetCaptureRecoveryError(
            "finalized Episode catalog changed during capture recovery"
        )
    moves = transaction.get("moves")
    if not isinstance(moves, list) or not moves:
        raise DatasetCaptureRecoveryError("capture recovery transaction has no moves")
    quarantine_root = root / quarantine_relative
    moved_paths: list[str] = []
    for raw_move in moves:
        if not isinstance(raw_move, Mapping):
            raise DatasetCaptureRecoveryError("capture recovery move is not an object")
        source_relative = Path(str(raw_move.get("source_relative") or ""))
        target_relative = Path(str(raw_move.get("target_relative") or ""))
        expected_target_relative = quarantine_relative / "artifacts" / source_relative
        if (
            source_relative.is_absolute()
            or target_relative.is_absolute()
            or ".." in source_relative.parts
            or ".." in target_relative.parts
            or not source_relative.parts
            or source_relative.parts[0] not in _OWNED_ARTIFACT_ROOTS
            or target_relative != expected_target_relative
        ):
            raise DatasetCaptureRecoveryError("capture recovery move escapes its owner roots")
        source = root / source_relative
        target = root / target_relative
        for candidate in (source, target):
            cursor = root
            for part in candidate.relative_to(root).parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise DatasetCaptureRecoveryError(
                        f"capture recovery path contains a symlink: {cursor}"
                    )
        if source.exists() and target.exists():
            raise DatasetCaptureRecoveryError(
                f"capture recovery has both source and quarantine target: {source_relative}"
            )
        if not source.exists() and not target.exists():
            raise DatasetCaptureRecoveryError(
                f"capture recovery lost both source and quarantine target: {source_relative}"
            )
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            _fsync_directory(source.parent)
            _fsync_directory(target.parent)
        moved_paths.append(source_relative.as_posix())
    receipt_path = quarantine_root / "receipt.json"
    receipt = dict(transaction)
    receipt.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
        }
    )
    _atomic_write_json(receipt_path, receipt)
    journal = root / CAPTURE_RECOVERY_RELATIVE_ROOT / _ACTIVE_TRANSACTION_FILENAME
    journal.unlink(missing_ok=True)
    _fsync_directory(journal.parent)
    return DatasetCaptureRecoveryResult(
        recovered=True,
        episode_index=episode_index,
        operation_id=operation_id,
        quarantine_root=str(quarantine_root),
        moved_paths=tuple(moved_paths),
        receipt_path=str(receipt_path),
    )


def _plan_transaction(
    root: Path,
    *,
    unowned_tactile_paths: tuple[str, ...],
) -> dict[str, Any]:
    catalog_path = root / "meta/episodes.jsonl"
    finalized = set(_catalog_indices(catalog_path))
    next_episode_index = max(finalized, default=-1) + 1
    artifacts = _discover_episode_artifact_units(root)
    unfinalized = set(artifacts) - finalized
    tactile_indices = _unowned_tactile_indices(unowned_tactile_paths)
    if tactile_indices != {next_episode_index}:
        raise DatasetCaptureRecoveryError(
            "unowned tactile files are not confined to the unique next Episode: "
            f"expected={next_episode_index} actual={sorted(tactile_indices)}"
        )
    if unfinalized != {next_episode_index}:
        raise DatasetCaptureRecoveryError(
            "writer-owned unfinalized artifacts are not confined to the unique "
            f"next Episode: expected={next_episode_index} actual={sorted(unfinalized)}"
        )
    nodes = artifacts.get(next_episode_index, ())
    if not nodes:
        raise DatasetCaptureRecoveryError("no abandoned tail-Episode artifacts found")
    _require_no_active_collection()
    operation_id = uuid.uuid4().hex
    quarantine_relative = (
        CAPTURE_RECOVERY_RELATIVE_ROOT
        / "abandoned_episodes"
        / f"episode_{next_episode_index:06d}_{operation_id}"
    )
    moves: list[dict[str, Any]] = []
    for node in nodes:
        relative = node.relative_to(root)
        file_count, total_bytes = _node_summary(node)
        moves.append(
            {
                "source_relative": relative.as_posix(),
                "target_relative": (quarantine_relative / "artifacts" / relative).as_posix(),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        )
    return {
        "schema": CAPTURE_RECOVERY_SCHEMA_V1,
        "operation_id": operation_id,
        "status": "planned",
        "created_at": _utc_now(),
        "dataset_root": str(root),
        "episode_index": next_episode_index,
        "catalog_sha256": (
            _sha256_file(catalog_path) if catalog_path.is_file() else None
        ),
        "unowned_tactile_paths": list(unowned_tactile_paths),
        "quarantine_relative": quarantine_relative.as_posix(),
        "moves": moves,
    }


def prepare_dataset_capture_writable(
    dataset_root: str | Path,
) -> DatasetCaptureRecoveryResult:
    """Validate capture append, quarantining only a provable abandoned tail.

    Unknown or non-tail inconsistencies remain hard failures.  A torn recovery
    transaction is resumed idempotently before admission is retried.
    """

    root = Path(dataset_root).expanduser().resolve()
    journal = root / CAPTURE_RECOVERY_RELATIVE_ROOT / _ACTIVE_TRANSACTION_FILENAME
    if journal.is_file():
        with _recovery_lock(root):
            result = _apply_transaction(root, _read_json(journal))
        assert_dataset_capture_writable(root)
        warnings.warn(
            "resumed abandoned dataset Episode recovery; artifacts remain available at "
            f"{result.quarantine_root}",
            RuntimeWarning,
            stacklevel=2,
        )
        return result
    try:
        assert_dataset_capture_writable(root)
        return DatasetCaptureRecoveryResult(recovered=False)
    except UnownedTactileFilesError:
        pass
    with _recovery_lock(root):
        if journal.is_file():
            result = _apply_transaction(root, _read_json(journal))
        else:
            try:
                assert_dataset_capture_writable(root)
                return DatasetCaptureRecoveryResult(recovered=False)
            except UnownedTactileFilesError as exc:
                transaction = _plan_transaction(
                    root,
                    unowned_tactile_paths=exc.paths,
                )
            _atomic_write_json(journal, transaction)
            result = _apply_transaction(root, transaction)
    assert_dataset_capture_writable(root)
    warnings.warn(
        "quarantined abandoned dataset Episode artifacts before capture append; "
        f"episode={result.episode_index} recovery={result.quarantine_root}",
        RuntimeWarning,
        stacklevel=2,
    )
    return result


__all__ = [
    "CAPTURE_RECOVERY_RELATIVE_ROOT",
    "CAPTURE_RECOVERY_SCHEMA_V1",
    "DatasetCaptureRecoveryError",
    "DatasetCaptureRecoveryResult",
    "prepare_dataset_capture_writable",
]
