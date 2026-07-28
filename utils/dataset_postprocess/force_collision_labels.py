from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_contracts.stats import (
    FORCE_COLLISION_LABEL_DETECTED,
    EpisodeStatsAccumulator,
    summary_has_force_collision,
)

from .io import episode_records, episode_stats_records, read_json, write_json, write_jsonl


_QUALITY_COLUMNS = (
    "timestamp",
    "control_hz",
    "active_sides",
    "arm_state_source",
    "camera_keys",
    "fault_codes",
    "teleop.pose_quality.status",
    "teleop.pose_quality.bad",
    "teleop.pose_quality.score",
    "teleop.pose_quality.special_code",
    "teleop.force_limit_exceeded",
    "teleop.force_collision_detected",
    "faults_json",
    "observation.arm.left.hardware_result_text",
    "observation.arm.left.send_result_text",
    "observation.arm.left.fault",
    "observation.arm.right.hardware_result_text",
    "observation.arm.right.send_result_text",
    "observation.arm.right.fault",
    "observation.arm.left.actual_hardware_result_text",
    "observation.arm.left.actual_state_error",
    "observation.arm.right.actual_hardware_result_text",
    "observation.arm.right.actual_state_error",
)

_OVERRIDE_CHOICES = ("auto", "all-detected", "all-clean")


@dataclass(frozen=True)
class ForceCollisionEpisodeLabel:
    episode_index: int
    force_collision_detected: bool
    force_collision_label: str
    pose_quality_status_counts: Mapping[str, int]
    pose_quality_bad_count: int
    pose_quality_warning_count: int
    pose_quality_min_score: float | None
    pose_quality_mean_score: float | None
    pose_quality_special_count: int = 0
    pose_quality_singularity_count: int = 0
    pose_quality_valid_score_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForceCollisionLabelPlan:
    dataset_root: str
    labels: tuple[ForceCollisionEpisodeLabel, ...]
    detected_indices: tuple[int, ...]
    clean_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "labels": [label.to_dict() for label in self.labels],
            "detected_indices": list(self.detected_indices),
            "clean_indices": list(self.clean_indices),
        }


def _read_parquet_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Force collision label scans require pyarrow.") from exc
    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in set(parquet.schema_arrow.names)]
    if not available:
        return []
    return pq.read_table(path, columns=available).to_pylist()


def _nominal_fps(info: Mapping[str, Any]) -> float:
    value = info.get("nominal_fps", info.get("fps", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_label(
    dataset_root: Path,
    record: Mapping[str, Any],
    *,
    nominal_fps: float,
    override: str = "auto",
) -> ForceCollisionEpisodeLabel:
    if override not in _OVERRIDE_CHOICES:
        raise ValueError(f"override must be one of {_OVERRIDE_CHOICES}.")
    episode_index = int(record["episode_index"])
    parquet_path = record.get("parquet_path")
    if not isinstance(parquet_path, str) or not parquet_path:
        raise ValueError(f"Episode {episode_index} has no parquet_path.")
    rows = _read_parquet_rows(dataset_root / parquet_path, _QUALITY_COLUMNS)
    accumulator = EpisodeStatsAccumulator(episode_index)
    for row in rows:
        accumulator.update(row)
    stats = accumulator.finalize(nominal_fps=nominal_fps)
    detected = bool(
        stats.force_collision_detected
        or summary_has_force_collision(record.get("summary"))
        or summary_has_force_collision(record.get("metadata"))
    )
    label = str(stats.force_collision_label)
    if detected:
        label = FORCE_COLLISION_LABEL_DETECTED
    if override == "all-detected":
        detected = True
        label = "detected_collision"
    elif override == "all-clean":
        detected = False
        label = "clean"
    return ForceCollisionEpisodeLabel(
        episode_index=episode_index,
        force_collision_detected=detected,
        force_collision_label=label,
        pose_quality_status_counts=dict(stats.pose_quality_status_counts),
        pose_quality_bad_count=int(stats.pose_quality_bad_count),
        pose_quality_warning_count=int(stats.pose_quality_warning_count),
        pose_quality_special_count=int(stats.pose_quality_special_count),
        pose_quality_singularity_count=int(stats.pose_quality_singularity_count),
        pose_quality_valid_score_count=int(stats.pose_quality_valid_score_count),
        pose_quality_min_score=stats.pose_quality_min_score,
        pose_quality_mean_score=stats.pose_quality_mean_score,
    )


def plan_force_collision_labels(
    dataset_root: str | Path,
    *,
    override: str = "auto",
) -> ForceCollisionLabelPlan:
    if override not in _OVERRIDE_CHOICES:
        raise ValueError(f"override must be one of {_OVERRIDE_CHOICES}.")
    root = Path(dataset_root)
    info = read_json(root / "meta" / "info.json")
    labels = tuple(
        _compute_label(root, record, nominal_fps=_nominal_fps(info), override=override)
        for record in episode_records(root)
    )
    detected = tuple(
        label.episode_index for label in labels if label.force_collision_detected
    )
    clean = tuple(label.episode_index for label in labels if not label.force_collision_detected)
    return ForceCollisionLabelPlan(
        dataset_root=str(root),
        labels=labels,
        detected_indices=detected,
        clean_indices=clean,
    )


def _quality_payload(label: ForceCollisionEpisodeLabel) -> dict[str, Any]:
    return {
        "pose_quality_status_counts": dict(label.pose_quality_status_counts),
        "pose_quality_bad_count": int(label.pose_quality_bad_count),
        "pose_quality_warning_count": int(label.pose_quality_warning_count),
        "pose_quality_special_count": int(label.pose_quality_special_count),
        "pose_quality_singularity_count": int(label.pose_quality_singularity_count),
        "pose_quality_valid_score_count": int(label.pose_quality_valid_score_count),
        "pose_quality_min_score": label.pose_quality_min_score,
        "pose_quality_mean_score": label.pose_quality_mean_score,
        "force_collision_detected": bool(label.force_collision_detected),
        "force_collision_label": label.force_collision_label,
    }


def _update_episode_record(
    record: dict[str, Any],
    label_by_index: Mapping[int, ForceCollisionEpisodeLabel],
) -> None:
    label = label_by_index.get(int(record.get("episode_index", -1)))
    if label is None:
        return
    summary = record.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    trajectory_quality = summary.get("trajectory_quality")
    if not isinstance(trajectory_quality, dict):
        trajectory_quality = {}
    trajectory_quality.update(_quality_payload(label))
    summary["trajectory_quality"] = trajectory_quality
    summary["force_collision_detected"] = bool(label.force_collision_detected)
    summary["force_collision_label"] = label.force_collision_label
    record["summary"] = summary
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_quality = metadata.get("trajectory_quality")
    if not isinstance(metadata_quality, dict):
        metadata_quality = {}
    metadata_quality.update(_quality_payload(label))
    metadata["trajectory_quality"] = metadata_quality
    metadata["force_collision_detected"] = bool(label.force_collision_detected)
    metadata["force_collision_label"] = label.force_collision_label
    record["metadata"] = metadata


def _update_stats_record(
    record: dict[str, Any],
    label_by_index: Mapping[int, ForceCollisionEpisodeLabel],
) -> None:
    label = label_by_index.get(int(record.get("episode_index", -1)))
    if label is None:
        return
    record.update(_quality_payload(label))


def label_force_collisions(
    dataset_root: str | Path,
    *,
    override: str = "auto",
) -> ForceCollisionLabelPlan:
    root = Path(dataset_root)
    plan = plan_force_collision_labels(root, override=override)
    label_by_index = {label.episode_index: label for label in plan.labels}
    episodes = episode_records(root)
    stats = episode_stats_records(root)
    for record in episodes:
        _update_episode_record(record, label_by_index)
    for record in stats:
        _update_stats_record(record, label_by_index)
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    info_path = root / "meta" / "info.json"
    info = read_json(info_path)
    history = info.get("postprocess_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "tool": "utils.dataset_postprocess.label_force_collisions",
            "updated_at": time.time(),
            "detected_indices": list(plan.detected_indices),
            "clean_indices": list(plan.clean_indices),
            "source": (
                "parquet_faults_json_and_hardware_diagnostics"
                if override == "auto"
                else f"manual_override_{override}"
            ),
            "override": str(override),
        }
    )
    info["postprocess_history"] = history
    write_json(info_path, info)
    return plan
