from __future__ import annotations

import copy
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset_contracts.layout import (
    chunk_id_from_episode_index,
    episode_id_from_index,
)
from dataset_contracts.parquet import write_rows_to_parquet
from dataset_contracts.stats import EpisodeStatsAccumulator

from .io import clone_dataset_tree, episode_records, read_json, read_parquet_rows, write_json, write_jsonl


@dataclass(frozen=True)
class StutterSegment:
    episode_index: int
    start_frame: int
    end_frame_exclusive: int
    start_row: int
    end_row_exclusive: int
    reason: str
    max_gap_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StutterCleanSegment:
    source_episode_index: int
    output_episode_index: int
    source_start_frame: int
    source_end_frame_exclusive: int
    source_start_row: int
    source_end_row_exclusive: int
    output_frame_count: int
    parquet_path: str
    reason: str
    max_gap_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StutterCleanPlan:
    dataset_root: str
    output_root: str | None
    max_gap_sec: float
    min_segment_frames: int
    original_episode_count: int
    output_episode_count: int
    original_frame_count: int
    output_frame_count: int
    dropped_frame_count: int
    segments: tuple[StutterCleanSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "output_root": self.output_root,
            "max_gap_sec": float(self.max_gap_sec),
            "min_segment_frames": int(self.min_segment_frames),
            "original_episode_count": int(self.original_episode_count),
            "output_episode_count": int(self.output_episode_count),
            "original_frame_count": int(self.original_frame_count),
            "output_frame_count": int(self.output_frame_count),
            "dropped_frame_count": int(self.dropped_frame_count),
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _row_value(row: Mapping[str, Any], key: str, fallback: Any = None) -> Any:
    return row.get(key, fallback)


def detect_stutter_segments(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_index: int,
    max_gap_sec: float = 0.2,
    timestamp_column: str = "timestamp",
    frame_index_column: str = "frame_index",
    min_segment_frames: int = 1,
) -> tuple[StutterSegment, ...]:
    """Split an episode into contiguous segments separated by long timestamp gaps."""
    if not rows:
        return ()
    timestamps = np.asarray([float(_row_value(row, timestamp_column)) for row in rows], dtype=float)
    if timestamps.ndim != 1 or timestamps.size != len(rows):
        raise ValueError("Invalid timestamp vector.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Timestamps must be finite for stutter detection.")
    max_gap = float(max_gap_sec)
    if max_gap <= 0.0:
        raise ValueError("max_gap_sec must be positive.")
    min_frames = max(1, int(min_segment_frames))
    split_after = [index for index, dt in enumerate(np.diff(timestamps)) if float(dt) > max_gap]
    boundaries = [0, *(index + 1 for index in split_after), len(rows)]
    segments: list[StutterSegment] = []
    for segment_index, (start_row, end_row) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end_row - start_row < min_frames:
            continue
        segment_rows = rows[start_row:end_row]
        start_frame = int(_row_value(segment_rows[0], frame_index_column, start_row))
        end_frame = int(_row_value(segment_rows[-1], frame_index_column, end_row - 1)) + 1
        if segment_index == 0 and len(boundaries) == 2:
            reason = "no_stutter"
        elif segment_index == 0:
            reason = "before_stutter"
        elif segment_index == len(boundaries) - 2:
            reason = "after_stutter"
        else:
            reason = "between_stutters"
        segment_max_gap = 0.0
        if end_row - start_row > 1:
            segment_max_gap = float(np.max(np.diff(timestamps[start_row:end_row])))
        segments.append(
            StutterSegment(
                episode_index=int(episode_index),
                start_frame=start_frame,
                end_frame_exclusive=end_frame,
                start_row=int(start_row),
                end_row_exclusive=int(end_row),
                reason=reason,
                max_gap_sec=segment_max_gap,
            )
        )
    return tuple(segments)


def _parquet_relative_path(episode_index: int, *, chunk_size: int) -> str:
    return str(
        Path("data")
        / chunk_id_from_episode_index(int(episode_index), chunk_size=int(chunk_size))
        / f"{episode_id_from_index(int(episode_index))}.parquet"
    )


def _timestamp(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    try:
        value = float(row.get("timestamp"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _nominal_fps(info: Mapping[str, Any]) -> float:
    value = info.get("nominal_fps", info.get("fps", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _chunk_size(info: Mapping[str, Any]) -> int:
    try:
        value = int(info.get("chunk_size", 1000))
    except (TypeError, ValueError):
        value = 1000
    return max(1, value)


def _reset_first_last_action(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    value = rows[0].get("observation.last_action")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows[0]["observation.last_action"] = [float("nan")] * len(value)


def _rewrite_segment_rows(
    rows: Sequence[Mapping[str, Any]],
    segment: StutterSegment,
    *,
    output_episode_index: int,
    start_global_index: int,
    reset_last_action: bool,
) -> list[dict[str, Any]]:
    out = [copy.deepcopy(dict(row)) for row in rows[segment.start_row : segment.end_row_exclusive]]
    for offset, row in enumerate(out):
        row["index"] = int(start_global_index + offset)
        row["episode_index"] = int(output_episode_index)
        row["frame_index"] = int(offset)
    if reset_last_action:
        _reset_first_last_action(out)
    return out


def _stats_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_index: int,
    nominal_fps: float,
) -> dict[str, Any]:
    accumulator = EpisodeStatsAccumulator(int(episode_index))
    for row in rows:
        accumulator.update(row)
    stats = accumulator.finalize(nominal_fps=float(nominal_fps))
    return {
        "episode_index": int(stats.episode_index),
        "frame_count": int(stats.frame_count),
        "start_timestamp": stats.start_timestamp,
        "end_timestamp": stats.end_timestamp,
        "duration_sec": float(stats.duration_sec),
        "average_dt_sec": stats.average_dt_sec,
        "average_hz": stats.average_hz,
        "mean_control_hz": stats.mean_control_hz,
        "nominal_fps": float(stats.nominal_fps),
        "active_sides": list(stats.active_sides),
        "arm_state_source": stats.arm_state_source,
        "camera_keys": list(stats.camera_keys),
        "fault_count": int(stats.fault_count),
        "pose_quality_status_counts": dict(stats.pose_quality_status_counts),
        "pose_quality_bad_count": int(stats.pose_quality_bad_count),
        "pose_quality_warning_count": int(stats.pose_quality_warning_count),
        "pose_quality_min_score": stats.pose_quality_min_score,
        "pose_quality_mean_score": stats.pose_quality_mean_score,
        "force_collision_detected": bool(stats.force_collision_detected),
        "force_collision_label": stats.force_collision_label,
    }


def _episode_record(
    source_record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    output_episode_index: int,
    parquet_path: str,
    segment: StutterSegment,
) -> dict[str, Any]:
    record = copy.deepcopy(dict(source_record))
    source_episode_index = int(source_record.get("episode_index", segment.episode_index))
    record["episode_index"] = int(output_episode_index)
    record["episode_id"] = episode_id_from_index(int(output_episode_index))
    record["frame_count"] = int(len(rows))
    record["parquet_path"] = str(parquet_path)
    start_timestamp = _timestamp(rows[0] if rows else None)
    end_timestamp = _timestamp(rows[-1] if rows else None)
    if start_timestamp is not None:
        record["start_timestamp"] = start_timestamp
    if end_timestamp is not None:
        record["end_timestamp"] = end_timestamp
    if start_timestamp is not None and end_timestamp is not None:
        record["duration_sec"] = max(0.0, float(end_timestamp) - float(start_timestamp))

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["source_episode_index"] = source_episode_index
    metadata["source_episode_id"] = source_record.get("episode_id", episode_id_from_index(source_episode_index))
    metadata["source_frame_start"] = int(segment.start_frame)
    metadata["source_frame_end_exclusive"] = int(segment.end_frame_exclusive)
    metadata["source_row_start"] = int(segment.start_row)
    metadata["source_row_end_exclusive"] = int(segment.end_row_exclusive)
    record["metadata"] = metadata

    summary = record.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    summary["episode_frame_count"] = int(len(rows))
    summary["last_step_index"] = int(len(rows) - 1)
    summary["stutter_clean"] = {
        "source_episode_index": source_episode_index,
        "source_frame_start": int(segment.start_frame),
        "source_frame_end_exclusive": int(segment.end_frame_exclusive),
        "source_row_start": int(segment.start_row),
        "source_row_end_exclusive": int(segment.end_row_exclusive),
        "reason": str(segment.reason),
        "max_gap_sec": float(segment.max_gap_sec),
    }
    record["summary"] = summary
    return record


def _segments_for_record(
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    max_gap_sec: float,
    min_segment_frames: int,
) -> tuple[StutterSegment, ...]:
    return detect_stutter_segments(
        rows,
        episode_index=int(record["episode_index"]),
        max_gap_sec=float(max_gap_sec),
        min_segment_frames=int(min_segment_frames),
    )


def plan_stutter_clean(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_gap_sec: float = 0.2,
    min_segment_frames: int = 1,
) -> StutterCleanPlan:
    root = Path(dataset_root)
    info = read_json(root / "meta" / "info.json")
    chunk_size = _chunk_size(info)
    records = episode_records(root)
    output_segments: list[StutterCleanSegment] = []
    original_frame_count = 0
    output_frame_count = 0
    output_episode_index = 0
    for record in records:
        parquet_path = record.get("parquet_path")
        if not isinstance(parquet_path, str) or not parquet_path:
            continue
        rows = read_parquet_rows(root / parquet_path)
        original_frame_count += len(rows)
        segments = _segments_for_record(
            record,
            rows,
            max_gap_sec=float(max_gap_sec),
            min_segment_frames=int(min_segment_frames),
        )
        for segment in segments:
            kept_frames = int(segment.end_row_exclusive - segment.start_row)
            if kept_frames <= 0:
                continue
            relative_path = _parquet_relative_path(output_episode_index, chunk_size=chunk_size)
            output_segments.append(
                StutterCleanSegment(
                    source_episode_index=int(record["episode_index"]),
                    output_episode_index=int(output_episode_index),
                    source_start_frame=int(segment.start_frame),
                    source_end_frame_exclusive=int(segment.end_frame_exclusive),
                    source_start_row=int(segment.start_row),
                    source_end_row_exclusive=int(segment.end_row_exclusive),
                    output_frame_count=kept_frames,
                    parquet_path=relative_path,
                    reason=str(segment.reason),
                    max_gap_sec=float(segment.max_gap_sec),
                )
            )
            output_frame_count += kept_frames
            output_episode_index += 1
    return StutterCleanPlan(
        dataset_root=str(root),
        output_root=None if output_root is None else str(Path(output_root)),
        max_gap_sec=float(max_gap_sec),
        min_segment_frames=int(min_segment_frames),
        original_episode_count=len(records),
        output_episode_count=len(output_segments),
        original_frame_count=int(original_frame_count),
        output_frame_count=int(output_frame_count),
        dropped_frame_count=int(original_frame_count - output_frame_count),
        segments=tuple(output_segments),
    )


def clean_stutters(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    max_gap_sec: float = 0.2,
    min_segment_frames: int = 1,
    overwrite: bool = False,
    reset_last_action: bool = True,
) -> StutterCleanPlan:
    root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("clean_stutters writes to a new output_root; in-place cleaning is disabled.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output}")
        shutil.rmtree(output)
    plan = plan_stutter_clean(
        root,
        output_root=output,
        max_gap_sec=float(max_gap_sec),
        min_segment_frames=int(min_segment_frames),
    )
    clone_dataset_tree(root, output, copy_prefixes=("meta",), skip_prefixes=("data",))
    output_data = output / "data"
    output_data.mkdir(parents=True, exist_ok=True)

    info = read_json(root / "meta" / "info.json")
    nominal_fps = _nominal_fps(info)
    source_records = episode_records(root)
    source_by_index = {int(record["episode_index"]): record for record in source_records}

    episodes: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    global_index = 0
    for segment_record in plan.segments:
        source_record = source_by_index[int(segment_record.source_episode_index)]
        source_rows = read_parquet_rows(root / str(source_record["parquet_path"]))
        source_segment = StutterSegment(
            episode_index=int(segment_record.source_episode_index),
            start_frame=int(segment_record.source_start_frame),
            end_frame_exclusive=int(segment_record.source_end_frame_exclusive),
            start_row=int(segment_record.source_start_row),
            end_row_exclusive=int(segment_record.source_end_row_exclusive),
            reason=str(segment_record.reason),
            max_gap_sec=float(segment_record.max_gap_sec),
        )
        rows = _rewrite_segment_rows(
            source_rows,
            source_segment,
            output_episode_index=int(segment_record.output_episode_index),
            start_global_index=int(global_index),
            reset_last_action=bool(reset_last_action),
        )
        global_index += len(rows)
        write_rows_to_parquet(rows, output / segment_record.parquet_path)
        episodes.append(
            _episode_record(
                source_record,
                rows,
                output_episode_index=int(segment_record.output_episode_index),
                parquet_path=str(segment_record.parquet_path),
                segment=source_segment,
            )
        )
        stats_records.append(
            _stats_record(
                rows,
                episode_index=int(segment_record.output_episode_index),
                nominal_fps=nominal_fps,
            )
        )

    write_jsonl(output / "meta" / "episodes.jsonl", episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", stats_records)
    info_path = output / "meta" / "info.json"
    info = read_json(info_path)
    if info:
        for key in ("total_episodes", "num_episodes", "episode_count"):
            if key in info:
                info[key] = len(episodes)
        for key in ("total_frames", "num_frames", "frame_count"):
            if key in info:
                info[key] = int(global_index)
        for key in ("dataset_name", "dataset_dir_name"):
            if key in info:
                info[key] = output.name
        history = info.get("postprocess_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "tool": "utils.dataset_postprocess.clean_stutters",
                "updated_at": time.time(),
                "source_dataset_root": str(root),
                "max_gap_sec": float(max_gap_sec),
                "min_segment_frames": int(min_segment_frames),
                "reset_last_action": bool(reset_last_action),
                "original_episode_count": int(plan.original_episode_count),
                "output_episode_count": int(plan.output_episode_count),
                "original_frame_count": int(plan.original_frame_count),
                "output_frame_count": int(global_index),
                "dropped_frame_count": int(plan.dropped_frame_count),
                "reindexed": True,
            }
        )
        info["postprocess_history"] = history
        write_json(info_path, info)
    return plan
