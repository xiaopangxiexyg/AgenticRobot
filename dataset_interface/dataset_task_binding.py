"""Read and validate one physical dataset's canonical task binding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetTaskBindingError(RuntimeError):
    """Raised when dataset-level task metadata is missing or inconsistent."""


@dataclass(frozen=True)
class DatasetTaskBinding:
    task_id: str
    task_description: str


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetTaskBindingError(
            f"Cannot read dataset task metadata {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetTaskBindingError(
            f"Dataset task metadata must be a JSON object: {path}."
        )
    return payload


def _read_task_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetTaskBindingError(
            f"Cannot read dataset task manifest {path}: {exc}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise DatasetTaskBindingError(
                f"Invalid JSON in dataset task manifest {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DatasetTaskBindingError(
                f"Dataset task manifest row must be an object: {path}:{line_number}."
            )
        rows.append(payload)
    return tuple(rows)


def _has_committed_episodes(root: Path, info: Mapping[str, Any] | None) -> bool:
    if info is not None:
        for key in ("episode_count", "frame_count"):
            try:
                if int(info.get(key, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                raise DatasetTaskBindingError(
                    f"Dataset metadata field {key!r} must be an integer: "
                    f"{root / 'meta' / 'info.json'}."
                ) from None
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return False
    try:
        return any(
            line.strip()
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
        )
    except OSError as exc:
        raise DatasetTaskBindingError(
            f"Cannot inspect dataset episode catalog {episodes_path}: {exc}"
        ) from exc


def inspect_dataset_task_binding(
    dataset_root: str | Path,
    *,
    require_complete: bool = False,
) -> DatasetTaskBinding | None:
    """Read and cross-check the canonical task binding of one dataset."""

    root = Path(dataset_root).expanduser()
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    info = _read_json_object(info_path) if info_path.exists() else None
    task_rows = _read_task_rows(tasks_path)
    populated = _has_committed_episodes(root, info)

    info_task_id = "" if info is None else str(info.get("task_id", "") or "").strip()
    info_description = (
        "" if info is None else str(info.get("task_description", "") or "").strip()
    )
    row_bindings = tuple(
        DatasetTaskBinding(
            task_id=str(row.get("task_id", "") or "").strip(),
            task_description=str(row.get("description", "") or "").strip(),
        )
        for row in task_rows
    )

    has_any_binding = bool(info_task_id or row_bindings)
    must_be_complete = bool(require_complete or populated)
    if not has_any_binding:
        if must_be_complete:
            raise DatasetTaskBindingError(
                f"Dataset has committed Episodes but no canonical task binding: {root}."
            )
        return None

    if not info_task_id:
        raise DatasetTaskBindingError(
            f"Dataset task manifest exists but meta/info.json has no task_id: {root}."
        )
    info_binding = DatasetTaskBinding(info_task_id, info_description)
    if not row_bindings:
        if must_be_complete:
            raise DatasetTaskBindingError(
                f"Dataset task binding is incomplete; meta/tasks.jsonl has no task row: {root}."
            )
        return info_binding

    for row_number, row_binding in enumerate(row_bindings, start=1):
        if not row_binding.task_id:
            raise DatasetTaskBindingError(
                f"Dataset task manifest row {row_number} has no task_id: {tasks_path}."
            )
        if row_binding != info_binding:
            raise DatasetTaskBindingError(
                "Dataset task metadata is inconsistent between meta/info.json and "
                f"meta/tasks.jsonl row {row_number}: info={info_binding!r}, "
                f"row={row_binding!r}."
            )
    return info_binding


__all__ = [
    "DatasetTaskBinding",
    "DatasetTaskBindingError",
    "inspect_dataset_task_binding",
]
