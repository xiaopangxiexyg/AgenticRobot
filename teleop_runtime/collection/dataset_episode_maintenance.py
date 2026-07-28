"""Offline maintenance for finalized LeRobot-style collection episodes.

The operation in this module is intentionally dataset-root scoped and must only
be called after every writer/finalizer for that root has become idle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Any, Iterable, Mapping


_EPISODE_TOKEN_RE = re.compile(r"episode_(\d{6})")
_CHUNK_TOKEN_RE = re.compile(r"^chunk-(\d{3})$")


@dataclass(frozen=True)
class EpisodeDeleteResult:
    deleted_episode_index: int
    previous_episode_count: int
    episode_count: int
    frame_count: int
    next_episode_index: int
    reindexed: Mapping[int, int]
    backup_root: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _episode_index(row: Mapping[str, Any], *, path: Path) -> int:
    try:
        index = int(row["episode_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing/invalid episode_index in {path}") from exc
    if index < 0:
        raise RuntimeError(f"Negative episode_index {index} in {path}")
    return index


def _replace_episode_references(value: Any, old_index: int, new_index: int, chunk_size: int) -> Any:
    old_token = f"episode_{old_index:06d}"
    new_token = f"episode_{new_index:06d}"
    old_chunk = f"chunk-{old_index // chunk_size:03d}"
    new_chunk = f"chunk-{new_index // chunk_size:03d}"
    if isinstance(value, str):
        return value.replace(old_chunk, new_chunk).replace(old_token, new_token)
    if isinstance(value, list):
        return [_replace_episode_references(item, old_index, new_index, chunk_size) for item in value]
    if isinstance(value, dict):
        rewritten = {
            key: _replace_episode_references(item, old_index, new_index, chunk_size)
            for key, item in value.items()
        }
        for key in ("episode_index", "legacy_episode_index", "source_episode_index"):
            try:
                if int(rewritten.get(key, -1)) == old_index:
                    rewritten[key] = new_index
            except (TypeError, ValueError):
                pass
        return rewritten
    return value


def _top_level_episode_artifacts(root: Path) -> list[tuple[Path, int]]:
    matches: list[tuple[Path, int]] = []
    for path in root.rglob("*"):
        match = _EPISODE_TOKEN_RE.search(path.name)
        if match is None:
            continue
        if any(_EPISODE_TOKEN_RE.search(parent.name) for parent in path.parents if parent != root):
            continue
        matches.append((path, int(match.group(1))))
    return sorted(matches, key=lambda item: (len(item[0].parts), str(item[0])))


def _artifact_destination(relative: Path, old_index: int, new_index: int, chunk_size: int) -> Path:
    old_chunk = old_index // chunk_size
    new_chunk = new_index // chunk_size
    parts: list[str] = []
    for part in relative.parts:
        chunk_match = _CHUNK_TOKEN_RE.match(part)
        if chunk_match and int(chunk_match.group(1)) == old_chunk:
            part = f"chunk-{new_chunk:03d}"
        part = part.replace(f"episode_{old_index:06d}", f"episode_{new_index:06d}")
        parts.append(part)
    return Path(*parts)


def _move_or_delete_artifacts(
    root: Path,
    *,
    deleted_index: int,
    mapping: Mapping[int, int],
    chunk_size: int,
) -> None:
    move_root = root / f".episode_compaction_moves_{uuid.uuid4().hex}"
    moves: list[tuple[Path, Path]] = []
    try:
        for ordinal, (path, old_index) in enumerate(_top_level_episode_artifacts(root)):
            if old_index == deleted_index:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                continue
            new_index = mapping.get(old_index)
            if new_index is None or new_index == old_index:
                continue
            relative = path.relative_to(root)
            destination = root / _artifact_destination(relative, old_index, new_index, chunk_size)
            staged = move_root / f"{ordinal:08d}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            path.rename(staged)
            moves.append((staged, destination))
        for staged, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"Episode compaction destination already exists: {destination}")
            staged.rename(destination)
    finally:
        shutil.rmtree(move_root, ignore_errors=True)
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _rewrite_parquet(
    path: Path,
    *,
    old_episode_index: int,
    new_episode_index: int,
    global_start_index: int,
    chunk_size: int,
) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("pyarrow is required to compact a dataset containing Parquet files") from exc
    table = pq.read_table(path)
    row_count = int(table.num_rows)
    for column_name, values in (
        ("episode_index", [new_episode_index] * row_count),
        ("index", list(range(global_start_index, global_start_index + row_count))),
    ):
        if column_name not in table.column_names:
            continue
        column_index = table.schema.get_field_index(column_name)
        target_type = table.schema.field(column_index).type
        table = table.set_column(column_index, column_name, pa.array(values, type=target_type))
    for column_index, field in enumerate(table.schema):
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            continue
        values = table.column(column_index).to_pylist()
        rewritten_values = [
            _replace_episode_references(value, old_episode_index, new_episode_index, chunk_size)
            if value is not None
            else None
            for value in values
        ]
        if rewritten_values != values:
            table = table.set_column(
                column_index,
                field.name,
                pa.array(rewritten_values, type=field.type),
            )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return row_count


def _rewrite_json_sidecars(root: Path, mapping: Mapping[int, int], chunk_size: int) -> None:
    excluded = {root / "meta" / "episodes.jsonl", root / "meta" / "episodes_stats.jsonl"}
    for path in root.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rewritten = value
        for old_index, new_index in mapping.items():
            rewritten = _replace_episode_references(rewritten, old_index, new_index, chunk_size)
        if rewritten != value:
            _write_json(path, rewritten)
    for path in root.rglob("*.jsonl"):
        if path in excluded:
            continue
        try:
            rows = _read_jsonl(path)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue
        rewritten_rows = []
        changed = False
        for row in rows:
            rewritten: Any = row
            for old_index, new_index in mapping.items():
                rewritten = _replace_episode_references(rewritten, old_index, new_index, chunk_size)
            rewritten_rows.append(rewritten)
            changed = changed or rewritten != row
        if changed:
            _write_jsonl(path, rewritten_rows)


def _maintain_v24_control_trace(
    root: Path,
    *,
    deleted_row: Mapping[str, Any],
    survivor_rows: Iterable[tuple[int, Mapping[str, Any]]],
    mapping: Mapping[int, int],
) -> None:
    """Maintain uncommitted v2.4 staging facts during legacy compaction.

    Committed captures are rejected by the public operation before staging.
    Keep the same check here as defense in depth: this helper must never
    rewrite an immutable ``capture_commit.cjson`` identity.
    """
    trace_root = root / "meta" / "v24" / "control_trace" / "episodes"
    if not trace_root.is_dir():
        return
    committed = sorted(trace_root.glob("*/capture_commit.cjson"))
    if committed:
        raise RuntimeError(
            "Refusing to maintain committed v2.4 ControlTrace in place; "
            "create a copy-on-write derived dataset instead."
        )
    deleted_binding = dict(deleted_row.get("metadata") or {}).get("control_trace_episode")
    if isinstance(deleted_binding, Mapping):
        deleted_uuid = str(deleted_binding.get("episode_uuid", "")).strip()
        if deleted_uuid:
            shutil.rmtree(trace_root / deleted_uuid, ignore_errors=True)
    # Uncommitted staging has no immutable commit identity to rewrite.  The
    # following artifact/path pass updates its legacy episode-index bindings.
    del survivor_rows, mapping


def _compact_staged_dataset(root: Path, deleted_index: int) -> tuple[int, int, dict[int, int]]:
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"
    info_path = root / "meta" / "info.json"
    episode_rows = _read_jsonl(episodes_path)
    if not episode_rows:
        raise RuntimeError(f"Dataset contains no finalized episodes: {root}")
    indexed_rows = [(_episode_index(row, path=episodes_path), row) for row in episode_rows]
    indices = [index for index, _ in indexed_rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate episode_index values in {episodes_path}")
    if deleted_index not in indices:
        raise IndexError(f"Episode {deleted_index} does not exist; available={sorted(indices)}")
    sorted_indices = sorted(indices)
    if sorted_indices != list(range(len(sorted_indices))):
        raise RuntimeError(
            "Dataset episode indices are not contiguous before deletion; repair it with an explicit migration first"
        )
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if not isinstance(info, dict):
        raise RuntimeError(f"Dataset info is not a JSON object: {info_path}")
    chunk_size = max(1, int(info.get("chunk_size", 1000)))
    survivors = [index for index in sorted_indices if index != deleted_index]
    mapping = {old_index: new_index for new_index, old_index in enumerate(survivors)}
    deleted_row = next(row for index, row in indexed_rows if index == deleted_index)
    survivor_indexed_rows = [(index, row) for index, row in indexed_rows if index != deleted_index]
    _maintain_v24_control_trace(
        root,
        deleted_row=deleted_row,
        survivor_rows=survivor_indexed_rows,
        mapping=mapping,
    )
    _move_or_delete_artifacts(
        root,
        deleted_index=deleted_index,
        mapping=mapping,
        chunk_size=chunk_size,
    )
    compacted_rows: list[dict[str, Any]] = []
    frame_count = 0
    for old_index, row in sorted(indexed_rows):
        if old_index == deleted_index:
            continue
        new_index = mapping[old_index]
        rewritten = _replace_episode_references(row, old_index, new_index, chunk_size)
        rewritten["episode_index"] = new_index
        parquet_value = rewritten.get("parquet_path")
        row_frame_count = int(rewritten.get("frame_count", 0) or 0)
        if parquet_value:
            parquet_path = root / str(parquet_value)
            if not parquet_path.exists():
                raise RuntimeError(f"Missing Parquet file referenced by episode {new_index}: {parquet_path}")
            row_frame_count = _rewrite_parquet(
                parquet_path,
                old_episode_index=old_index,
                new_episode_index=new_index,
                global_start_index=frame_count,
                chunk_size=chunk_size,
            )
            rewritten["frame_count"] = row_frame_count
        frame_count += row_frame_count
        compacted_rows.append(rewritten)
    stats_by_index = {
        _episode_index(row, path=stats_path): row for row in _read_jsonl(stats_path)
    }
    compacted_stats: list[dict[str, Any]] = []
    for old_index in survivors:
        if old_index not in stats_by_index:
            continue
        rewritten = _replace_episode_references(
            stats_by_index[old_index], old_index, mapping[old_index], chunk_size
        )
        rewritten["episode_index"] = mapping[old_index]
        compacted_stats.append(rewritten)
    _write_jsonl(episodes_path, compacted_rows)
    _write_jsonl(stats_path, compacted_stats)
    _rewrite_json_sidecars(root, mapping, chunk_size)
    info["episode_count"] = len(compacted_rows)
    info["frame_count"] = frame_count
    _write_json(info_path, info)
    return len(compacted_rows), frame_count, mapping


def _delete_episode_and_compact_unlocked(
    root_dir: str | Path, episode_index: int
) -> EpisodeDeleteResult:
    """Delete one finalized episode and atomically install a compacted dataset.

    A hard-linked pre-operation backup is retained beside the dataset root. It
    consumes little additional space for unchanged large media files and makes
    the destructive user action recoverable.
    """

    root = Path(root_dir).expanduser().resolve()
    target = int(episode_index)
    if target < 0:
        raise ValueError("episode_index must be non-negative")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not (root / "meta" / "episodes.jsonl").is_file():
        raise RuntimeError(f"Not a finalized LeRobot-style dataset: {root}")
    committed_v24 = sorted(
        (root / "meta" / "v24" / "control_trace" / "episodes").glob(
            "*/capture_commit.cjson"
        )
    )
    if committed_v24:
        raise RuntimeError(
            "Refusing in-place episode compaction for a committed v2.4 capture; "
            "create a copy-on-write derived dataset with new commit identities instead."
        )
    v25p_episode_identities = sorted(
        (root / "meta" / "v25p" / "episodes").glob("*/identity.cjson")
    ) + sorted(
        (root / "meta" / "v25p" / "native" / "episodes").glob(
            "*/identity.cjson"
        )
    )
    if v25p_episode_identities:
        raise RuntimeError(
            "Refusing in-place episode compaction for a v2.5-p capture because "
            "episode UUID/index bindings and base-record hashes are immutable; "
            "create a copy-on-write derived dataset with new identities instead."
        )
    operation_id = uuid.uuid4().hex[:10]
    stage = root.with_name(f".{root.name}.episode_delete_stage_{operation_id}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root.with_name(f"{root.name}.before_delete_episode_{target:06d}_{timestamp}")
    if stage.exists() or backup.exists():
        raise RuntimeError("Dataset maintenance staging/backup path already exists")
    previous_count = len(_read_jsonl(root / "meta" / "episodes.jsonl"))
    installed = False
    try:
        shutil.copytree(root, stage, copy_function=_hardlink_or_copy, symlinks=True)
        episode_count, frame_count, mapping = _compact_staged_dataset(stage, target)
        root.rename(backup)
        try:
            stage.rename(root)
            installed = True
        except BaseException:
            backup.rename(root)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    if not installed:
        raise RuntimeError("Dataset compaction was not installed")
    return EpisodeDeleteResult(
        deleted_episode_index=target,
        previous_episode_count=previous_count,
        episode_count=episode_count,
        frame_count=frame_count,
        next_episode_index=episode_count,
        reindexed=dict(mapping),
        backup_root=backup,
    )


def delete_episode_and_compact(root_dir: str | Path, episode_index: int) -> EpisodeDeleteResult:
    """Serialize maintenance per dataset root and compact one finalized episode."""

    root = Path(root_dir).expanduser().resolve()
    lock_path = root.with_name(f".{root.name}.episode_maintenance.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+b")
    try:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another episode deletion is already running for dataset: {root}"
            ) from exc
        return _delete_episode_and_compact_unlocked(root, int(episode_index))
    finally:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_handle.close()


__all__ = ["EpisodeDeleteResult", "delete_episode_and_compact"]
