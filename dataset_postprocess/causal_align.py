from __future__ import annotations

import bisect
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from teleop_runtime.collection.schema_v21 import write_rows_to_parquet

from .io import episode_records, parquet_path_for_record, read_json, read_parquet_rows, write_json


@dataclass(frozen=True)
class CausalAlignModalityStats:
    episode_index: int
    modality: str
    rows: int
    current_future_count: int
    aligned_future_count: int
    no_prior_count: int
    changed_count: int
    current_max_future_sec: float
    aligned_max_future_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalAlignPlan:
    dataset_root: str
    output_root: str | None
    modalities: tuple[CausalAlignModalityStats, ...]

    @property
    def changed_rows(self) -> int:
        return sum(item.changed_count for item in self.modalities)

    @property
    def current_future_count(self) -> int:
        return sum(item.current_future_count for item in self.modalities)

    @property
    def aligned_future_count(self) -> int:
        return sum(item.aligned_future_count for item in self.modalities)

    @property
    def no_prior_count(self) -> int:
        return sum(item.no_prior_count for item in self.modalities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "changed_rows": int(self.changed_rows),
            "current_future_count": int(self.current_future_count),
            "aligned_future_count": int(self.aligned_future_count),
            "no_prior_count": int(self.no_prior_count),
            "modalities": [item.to_dict() for item in self.modalities],
        }


@dataclass(frozen=True)
class _ModalitySpec:
    name: str
    timestamp_column: str
    frame_index_column: str
    copy_prefix: str


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _valid_frame_index(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _discover_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    align_cameras: bool,
    align_tactile: bool,
) -> tuple[_ModalitySpec, ...]:
    if not rows:
        return ()
    keys: set[str] = set()
    for row in rows[: min(len(rows), 8)]:
        keys.update(str(key) for key in row)
    specs: list[_ModalitySpec] = []
    if align_cameras:
        camera_keys: set[str] = set()
        prefix = "observation.images."
        suffix = ".source_timestamp"
        for key in keys:
            if key.startswith(prefix) and key.endswith(suffix):
                camera_keys.add(key[len(prefix) : -len(suffix)])
        for camera_key in sorted(camera_keys):
            specs.append(
                _ModalitySpec(
                    name=f"camera:{camera_key}",
                    timestamp_column=f"observation.images.{camera_key}.source_timestamp",
                    frame_index_column=f"observation.images.{camera_key}.video_frame_index",
                    copy_prefix=f"observation.images.{camera_key}.",
                )
            )
    if align_tactile:
        for side in ("left", "right"):
            timestamp_column = f"observation.hand.{side}.tactile_timestamp"
            frame_index_column = f"observation.hand.{side}.tactile_frame_index"
            if timestamp_column in keys and frame_index_column in keys:
                specs.append(
                    _ModalitySpec(
                        name=f"tactile:{side}",
                        timestamp_column=timestamp_column,
                        frame_index_column=frame_index_column,
                        copy_prefix=f"observation.hand.{side}.tactile",
                    )
                )
    return tuple(specs)


def _copy_columns(rows: Sequence[Mapping[str, Any]], prefix: str) -> tuple[str, ...]:
    if not rows:
        return ()
    keys: set[str] = set()
    for row in rows[: min(len(rows), 8)]:
        keys.update(str(key) for key in row)
    return tuple(sorted(key for key in keys if key.startswith(prefix)))


def _candidate_indices(
    rows: Sequence[Mapping[str, Any]],
    spec: _ModalitySpec,
) -> tuple[list[float], list[int]]:
    candidates: list[tuple[float, int]] = []
    for row_index, row in enumerate(rows):
        selected_time = _finite_float(row.get(spec.timestamp_column))
        frame_index = _valid_frame_index(row.get(spec.frame_index_column))
        if selected_time is None or frame_index is None:
            continue
        candidates.append((selected_time, row_index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[0] for item in candidates], [item[1] for item in candidates]


def _selected_source_time(
    row: Mapping[str, Any],
    spec: _ModalitySpec,
) -> float | None:
    return _finite_float(row.get(spec.timestamp_column))


def _align_rows_for_spec(
    rows: list[dict[str, Any]],
    *,
    episode_index: int,
    spec: _ModalitySpec,
    mutate: bool,
    no_prior_policy: str,
) -> CausalAlignModalityStats:
    original_rows = [dict(row) for row in rows]
    sample_times = [_finite_float(row.get("timestamp")) for row in rows]
    source_times, source_row_indices = _candidate_indices(original_rows, spec)
    columns = _copy_columns(original_rows, spec.copy_prefix)
    current_future_count = 0
    aligned_future_count = 0
    no_prior_count = 0
    changed_count = 0
    current_max_future = 0.0
    aligned_max_future = 0.0

    if not source_times:
        return CausalAlignModalityStats(
            episode_index=int(episode_index),
            modality=spec.name,
            rows=len(rows),
            current_future_count=0,
            aligned_future_count=0,
            no_prior_count=len(rows),
            changed_count=0,
            current_max_future_sec=0.0,
            aligned_max_future_sec=0.0,
        )

    for row_index, row in enumerate(rows):
        sample_time = sample_times[row_index]
        current_time = _selected_source_time(original_rows[row_index], spec)
        if sample_time is not None and current_time is not None:
            current_offset = current_time - sample_time
            if current_offset > 0.0:
                current_future_count += 1
                current_max_future = max(current_max_future, float(current_offset))

        selected_index: int | None = None
        if sample_time is not None:
            position = bisect.bisect_right(source_times, sample_time) - 1
            if position >= 0:
                selected_index = source_row_indices[position]
            else:
                no_prior_count += 1

        if selected_index is None:
            if no_prior_policy == "first":
                selected_index = source_row_indices[0]
            elif no_prior_policy == "null":
                selected_index = None
            else:
                selected_index = row_index

        if selected_index is not None:
            aligned_time = _selected_source_time(original_rows[selected_index], spec)
            if sample_time is not None and aligned_time is not None:
                aligned_offset = aligned_time - sample_time
                if aligned_offset > 0.0:
                    aligned_future_count += 1
                    aligned_max_future = max(aligned_max_future, float(aligned_offset))
        else:
            aligned_time = None

        if selected_index != row_index:
            changed_count += 1
            if mutate:
                if selected_index is None:
                    for column in columns:
                        row[column] = None
                else:
                    selected_row = original_rows[selected_index]
                    for column in columns:
                        row[column] = selected_row.get(column)

        if mutate and selected_index is not None and sample_time is not None and aligned_time is not None:
            age_column = f"{spec.copy_prefix}age_sec"
            if age_column in row:
                row[age_column] = max(0.0, float(sample_time) - float(aligned_time))

    return CausalAlignModalityStats(
        episode_index=int(episode_index),
        modality=spec.name,
        rows=len(rows),
        current_future_count=int(current_future_count),
        aligned_future_count=int(aligned_future_count),
        no_prior_count=int(no_prior_count),
        changed_count=int(changed_count),
        current_max_future_sec=float(current_max_future),
        aligned_max_future_sec=float(aligned_max_future),
    )


def _align_episode_rows(
    rows: list[dict[str, Any]],
    *,
    episode_index: int,
    align_cameras: bool,
    align_tactile: bool,
    no_prior_policy: str,
    mutate: bool,
) -> tuple[CausalAlignModalityStats, ...]:
    specs = _discover_specs(rows, align_cameras=align_cameras, align_tactile=align_tactile)
    return tuple(
        _align_rows_for_spec(
            rows,
            episode_index=episode_index,
            spec=spec,
            mutate=mutate,
            no_prior_policy=no_prior_policy,
        )
        for spec in specs
    )


def plan_causal_align(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    align_cameras: bool = True,
    align_tactile: bool = True,
    no_prior_policy: str = "keep_current",
) -> CausalAlignPlan:
    if no_prior_policy not in {"keep_current", "first", "null"}:
        raise ValueError("no_prior_policy must be keep_current, first, or null.")
    root = Path(dataset_root)
    stats: list[CausalAlignModalityStats] = []
    for record in episode_records(root):
        rows = read_parquet_rows(parquet_path_for_record(root, record))
        stats.extend(
            _align_episode_rows(
                rows,
                episode_index=int(record["episode_index"]),
                align_cameras=bool(align_cameras),
                align_tactile=bool(align_tactile),
                no_prior_policy=no_prior_policy,
                mutate=False,
            )
        )
    return CausalAlignPlan(
        dataset_root=str(root),
        output_root=None if output_root is None else str(Path(output_root)),
        modalities=tuple(stats),
    )


def causal_align_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    align_cameras: bool = True,
    align_tactile: bool = True,
    no_prior_policy: str = "keep_current",
    overwrite: bool = False,
) -> CausalAlignPlan:
    if no_prior_policy not in {"keep_current", "first", "null"}:
        raise ValueError("no_prior_policy must be keep_current, first, or null.")
    root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("causal_align_dataset writes to a new output_root; in-place alignment is disabled.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(root, output)
    stats: list[CausalAlignModalityStats] = []
    global_index = 0
    for record in episode_records(output):
        parquet_path = parquet_path_for_record(output, record)
        rows = read_parquet_rows(parquet_path)
        stats.extend(
            _align_episode_rows(
                rows,
                episode_index=int(record["episode_index"]),
                align_cameras=bool(align_cameras),
                align_tactile=bool(align_tactile),
                no_prior_policy=no_prior_policy,
                mutate=True,
            )
        )
        for row in rows:
            row["index"] = int(global_index)
            global_index += 1
        write_rows_to_parquet(rows, parquet_path)

    info_path = output / "meta" / "info.json"
    info = read_json(info_path)
    if info:
        for key in ("dataset_name", "dataset_dir_name"):
            if key in info:
                info[key] = output.name
        history = info.get("postprocess_history")
        if not isinstance(history, list):
            history = []
        plan = CausalAlignPlan(
            dataset_root=str(root),
            output_root=str(output),
            modalities=tuple(stats),
        )
        history.append(
            {
                "tool": "dataset_postprocess.causal_align",
                "updated_at": time.time(),
                "align_cameras": bool(align_cameras),
                "align_tactile": bool(align_tactile),
                "no_prior_policy": str(no_prior_policy),
                "changed_rows": int(plan.changed_rows),
                "current_future_count": int(plan.current_future_count),
                "aligned_future_count": int(plan.aligned_future_count),
                "no_prior_count": int(plan.no_prior_count),
                "source_dataset_root": str(root),
                "reindexed": True,
            }
        )
        info["postprocess_history"] = history
        write_json(info_path, info)

    return CausalAlignPlan(
        dataset_root=str(root),
        output_root=str(output),
        modalities=tuple(stats),
    )
