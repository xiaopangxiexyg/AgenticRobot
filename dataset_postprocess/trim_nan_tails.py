from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fault_nan_tail import FaultNanTail, detect_fault_nan_tail
from .io import (
    episode_records,
    episode_stats_records,
    parquet_path_for_record,
    read_json,
    read_parquet_rows,
    read_parquet_table,
    write_json,
    write_jsonl,
    write_parquet_table,
)


@dataclass(frozen=True)
class TailTrim:
    episode_index: int
    parquet_path: str
    original_frame_count: int
    kept_frame_count: int
    trimmed_frame_count: int
    tail: FaultNanTail

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tail"] = self.tail.to_dict()
        return payload


@dataclass(frozen=True)
class TailTrimPlan:
    dataset_root: str
    output_root: str | None
    trims: tuple[TailTrim, ...]
    total_trimmed_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "trims": [trim.to_dict() for trim in self.trims],
            "total_trimmed_frames": self.total_trimmed_frames,
        }


def plan_nan_tail_trim(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    min_nan_fraction: float = 0.8,
) -> TailTrimPlan:
    root = Path(dataset_root)
    trims: list[TailTrim] = []
    for record in episode_records(root):
        rows = read_parquet_rows(parquet_path_for_record(root, record))
        tail = detect_fault_nan_tail(
            rows,
            episode_index=int(record["episode_index"]),
            min_nan_fraction=float(min_nan_fraction),
        )
        if tail is None:
            continue
        parquet_path = str(record["parquet_path"])
        trims.append(
            TailTrim(
                episode_index=int(record["episode_index"]),
                parquet_path=parquet_path,
                original_frame_count=len(rows),
                kept_frame_count=int(tail.start_row),
                trimmed_frame_count=len(rows) - int(tail.start_row),
                tail=tail,
            )
        )
    return TailTrimPlan(
        dataset_root=str(root),
        output_root=None if output_root is None else str(Path(output_root)),
        trims=tuple(trims),
        total_trimmed_frames=sum(trim.trimmed_frame_count for trim in trims),
    )


def _timestamp(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    try:
        value = float(row.get("timestamp"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _update_episode_record(record: dict[str, Any], rows: list[dict[str, Any]], trim: TailTrim) -> None:
    record["frame_count"] = int(trim.kept_frame_count)
    if rows:
        start_timestamp = _timestamp(rows[0])
        end_timestamp = _timestamp(rows[-1])
        if start_timestamp is not None:
            record["start_timestamp"] = start_timestamp
        if end_timestamp is not None:
            record["end_timestamp"] = end_timestamp
        if start_timestamp is not None and end_timestamp is not None:
            record["duration_sec"] = max(0.0, float(end_timestamp) - float(start_timestamp))
    postprocess = record.get("postprocess")
    if not isinstance(postprocess, dict):
        postprocess = {}
    postprocess["nan_tail_trim"] = trim.to_dict()
    record["postprocess"] = postprocess


def _update_stats_record(record: dict[str, Any], rows: list[dict[str, Any]], trim: TailTrim) -> None:
    record["frame_count"] = int(trim.kept_frame_count)
    if rows:
        start_timestamp = _timestamp(rows[0])
        end_timestamp = _timestamp(rows[-1])
        if start_timestamp is not None:
            record["start_timestamp"] = start_timestamp
        if end_timestamp is not None:
            record["end_timestamp"] = end_timestamp
        if start_timestamp is not None and end_timestamp is not None:
            duration = max(0.0, float(end_timestamp) - float(start_timestamp))
            record["duration_sec"] = duration
            record["average_dt_sec"] = duration / max(1, len(rows) - 1)
            record["average_hz"] = 0.0 if duration <= 0.0 else max(0, len(rows) - 1) / duration


def _write_updated_meta(output_root: Path, trims: Mapping[int, TailTrim]) -> None:
    episodes = episode_records(output_root)
    stats = episode_stats_records(output_root)
    for record in episodes:
        episode_index = int(record.get("episode_index", -1))
        trim = trims.get(episode_index)
        if trim is None:
            continue
        rows = read_parquet_rows(output_root / trim.parquet_path)
        _update_episode_record(record, rows, trim)
    for record in stats:
        episode_index = int(record.get("episode_index", -1))
        trim = trims.get(episode_index)
        if trim is None:
            continue
        rows = read_parquet_rows(output_root / trim.parquet_path)
        _update_stats_record(record, rows, trim)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", stats)
    info_path = output_root / "meta" / "info.json"
    info = read_json(info_path)
    if info:
        frame_count = sum(int(record.get("frame_count", 0) or 0) for record in episodes)
        for key in ("total_frames", "num_frames", "frame_count"):
            if key in info:
                info[key] = frame_count
        for key in ("dataset_name", "dataset_dir_name"):
            if key in info:
                info[key] = output_root.name
        history = info.get("postprocess_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "tool": "dataset_postprocess.trim_nan_tails",
                "trimmed_episode_indices": sorted(trims),
                "total_trimmed_frames": sum(trim.trimmed_frame_count for trim in trims.values()),
                "updated_at": time.time(),
            }
        )
        info["postprocess_history"] = history
        write_json(info_path, info)


def trim_nan_tails(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    min_nan_fraction: float = 0.8,
    overwrite: bool = False,
) -> TailTrimPlan:
    root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("trim_nan_tails writes to a new output_root; in-place trimming is disabled.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output}")
        shutil.rmtree(output)
    plan = plan_nan_tail_trim(
        root,
        output_root=output,
        min_nan_fraction=float(min_nan_fraction),
    )
    shutil.copytree(root, output)
    trim_by_episode = {trim.episode_index: trim for trim in plan.trims}
    for trim in plan.trims:
        parquet_path = output / trim.parquet_path
        table = read_parquet_table(parquet_path)
        write_parquet_table(parquet_path, table.slice(0, int(trim.kept_frame_count)))
    _write_updated_meta(output, trim_by_episode)
    return plan
