from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SIDE_ORDER: tuple[str, ...] = ("left", "right")
PACKED_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
LEGACY_V21_STATE_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}
POSE_DIM = 7
PACKED_ARM_POSE_DIM = 14
LEGACY_V21_STATE_DIM = 38

RELATION_SAME_STATE = "same_state"
RELATION_NEXT_STATE = "next_state"
RELATION_LAST_ACTION = "last_action"
RELATIONS: tuple[str, ...] = (
    RELATION_SAME_STATE,
    RELATION_NEXT_STATE,
    RELATION_LAST_ACTION,
)

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "pose_action_next_state_reports"
DEFAULT_TRANSLATION_THRESHOLD_M = 0.01
DEFAULT_ROTATION_THRESHOLD_DEG = 5.0


@dataclass
class MetricAccumulator:
    translation_m: list[float] = field(default_factory=list)
    rotation_deg: list[float] = field(default_factory=list)
    abs_xyz_m: list[list[float]] = field(default_factory=list)
    dt_sec: list[float] = field(default_factory=list)
    skipped_nonfinite: int = 0
    skipped_missing: int = 0

    def add(
        self,
        action_pose: np.ndarray | None,
        state_pose: np.ndarray | None,
        *,
        dt_sec: float | None = None,
    ) -> tuple[float, float] | None:
        if action_pose is None or state_pose is None:
            self.skipped_missing += 1
            return None
        errors = pose_error(action_pose, state_pose)
        if errors is None:
            self.skipped_nonfinite += 1
            return None
        translation_m, rotation_deg, abs_xyz_m = errors
        self.translation_m.append(float(translation_m))
        self.rotation_deg.append(float(rotation_deg))
        self.abs_xyz_m.append(abs_xyz_m.astype(float).tolist())
        if dt_sec is not None and math.isfinite(float(dt_sec)):
            self.dt_sec.append(float(dt_sec))
        return float(translation_m), float(rotation_deg)

    @property
    def pair_count(self) -> int:
        return len(self.translation_m)

    def to_summary(
        self,
        *,
        translation_threshold_m: float,
        rotation_threshold_deg: float,
    ) -> dict[str, Any]:
        translation = np.asarray(self.translation_m, dtype=float)
        rotation = np.asarray(self.rotation_deg, dtype=float)
        abs_xyz = np.asarray(self.abs_xyz_m, dtype=float)
        dt = np.asarray(self.dt_sec, dtype=float)
        summary: dict[str, Any] = {
            "pair_count": int(self.pair_count),
            "skipped_missing": int(self.skipped_missing),
            "skipped_nonfinite": int(self.skipped_nonfinite),
            "translation_error_m": numeric_summary(translation),
            "rotation_error_deg": numeric_summary(rotation),
            "abs_xyz_error_m": {
                "x": numeric_summary(abs_xyz[:, 0]) if abs_xyz.size else numeric_summary([]),
                "y": numeric_summary(abs_xyz[:, 1]) if abs_xyz.size else numeric_summary([]),
                "z": numeric_summary(abs_xyz[:, 2]) if abs_xyz.size else numeric_summary([]),
            },
            "dt_sec": numeric_summary(dt),
        }
        if translation.size:
            summary["translation_gt_threshold_fraction"] = float(
                np.mean(translation > float(translation_threshold_m))
            )
        else:
            summary["translation_gt_threshold_fraction"] = None
        if rotation.size:
            summary["rotation_gt_threshold_fraction"] = float(
                np.mean(rotation > float(rotation_threshold_deg))
            )
        else:
            summary["rotation_gt_threshold_fraction"] = None
        return summary


@dataclass
class DatasetAnalysisConfig:
    dataset_root: Path
    output_dir: Path
    max_episodes: int | None = None
    dataset_glob: str = "*"
    include_dataset: tuple[str, ...] = ()
    exclude_dataset: tuple[str, ...] = ()
    translation_threshold_m: float = DEFAULT_TRANSLATION_THRESHOLD_M
    rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG
    max_outliers: int = 50


def require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pose_action_next_state requires pyarrow. "
            "Activate the DexTeleop environment with "
            "`conda activate dexteleop310`, then run "
            "`python utils/statistic/pose_action_next_state.py`."
        ) from exc
    return pq


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            payload = json.loads(text)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def discover_dataset_roots(
    root: Path,
    *,
    dataset_glob: str = "*",
    include_dataset: Sequence[str] = (),
    exclude_dataset: Sequence[str] = (),
) -> list[Path]:
    root = root.expanduser().resolve()
    include = tuple(str(item) for item in include_dataset if str(item))
    exclude = tuple(str(item) for item in exclude_dataset if str(item))

    candidates: list[Path]
    if (root / "meta" / "episodes.jsonl").exists() or (root / "data").exists():
        candidates = [root]
    else:
        candidates = [
            item
            for item in sorted(root.glob(str(dataset_glob)))
            if item.is_dir() and ((item / "meta" / "episodes.jsonl").exists() or (item / "data").exists())
        ]

    def selected(path: Path) -> bool:
        name = path.name
        if include and not any(pattern in name for pattern in include):
            return False
        if exclude and any(pattern in name for pattern in exclude):
            return False
        return True

    return [path for path in candidates if selected(path)]


def episode_parquet_records(dataset_root: Path) -> list[dict[str, Any]]:
    records = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    out: list[dict[str, Any]] = []
    for record in records:
        relative = record.get("parquet_path")
        if not isinstance(relative, str) or not relative:
            continue
        path = dataset_root / relative
        if path.exists():
            item = dict(record)
            item["parquet_abs_path"] = str(path)
            out.append(item)
    if out:
        return out

    fallback: list[dict[str, Any]] = []
    for path in sorted(dataset_root.glob("data/chunk-*/episode_*.parquet")):
        episode_index = parse_episode_index(path)
        fallback.append(
            {
                "episode_index": episode_index,
                "episode_id": path.stem,
                "parquet_path": str(path.relative_to(dataset_root)),
                "parquet_abs_path": str(path),
            }
        )
    return fallback


def parse_episode_index(path: Path) -> int | None:
    stem = path.stem
    prefix = "episode_"
    if stem.startswith(prefix):
        try:
            return int(stem[len(prefix) :])
        except ValueError:
            return None
    return None


def numeric_summary(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (int(length),):
        return None
    return vector


def vector_with_minimum_length_or_none(value: Any, *, minimum_length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < int(minimum_length):
        return None
    return vector


def packed_arm_pose_from_row(
    row: Mapping[str, Any],
    *,
    key: str,
    side: str,
) -> np.ndarray | None:
    vector = vector_with_minimum_length_or_none(
        row.get(key),
        minimum_length=PACKED_ARM_POSE_DIM,
    )
    if vector is None:
        return None
    return vector[PACKED_ARM_POSE_SLICES[side]]


def pose_from_row(row: Mapping[str, Any], *, side: str, source: str) -> np.ndarray | None:
    if source == "action":
        aliases = (
            f"action.{side}_arm.pose",
            f"action.{side}.arm.pose",
            f"action.{side}_arm_pose",
            f"action.{side}.pose",
        )
        for key in aliases:
            pose = vector_or_none(row.get(key), length=POSE_DIM)
            if pose is not None:
                return pose
        combined_pose = packed_arm_pose_from_row(row, key="action.arm.pose", side=side)
        if combined_pose is not None:
            return combined_pose
        return packed_arm_pose_from_row(row, key="action", side=side)

    if source == "state":
        aliases = (
            f"observation.{side}_arm.pose",
            f"observation.{side}.arm.pose",
            f"observation.{side}_arm_pose",
            f"observation.{side}.pose",
            f"observation.arm.{side}.current_tcp_pose",
            f"observation.arm.{side}.actual_tcp_pose",
        )
        for key in aliases:
            pose = vector_or_none(row.get(key), length=POSE_DIM)
            if pose is not None:
                return pose
        combined_pose = packed_arm_pose_from_row(
            row,
            key="observation.arm.pose",
            side=side,
        )
        if combined_pose is not None:
            return combined_pose
        state = vector_with_minimum_length_or_none(
            row.get("observation.state"),
            minimum_length=PACKED_ARM_POSE_DIM,
        )
        if state is None:
            return None
        if state.shape == (LEGACY_V21_STATE_DIM,):
            return state[LEGACY_V21_STATE_ARM_POSE_SLICES[side]]
        return state[PACKED_ARM_POSE_SLICES[side]]

    if source == "last_action":
        return packed_arm_pose_from_row(
            row,
            key="observation.last_action",
            side=side,
        )

    raise ValueError(f"Unsupported pose source: {source!r}")


def pose_error(action_pose: np.ndarray, state_pose: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    a = np.asarray(action_pose, dtype=float).reshape(-1)
    b = np.asarray(state_pose, dtype=float).reshape(-1)
    if a.shape != (POSE_DIM,) or b.shape != (POSE_DIM,):
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    qa = a[3:7]
    qb = b[3:7]
    norm_a = float(np.linalg.norm(qa))
    norm_b = float(np.linalg.norm(qb))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    abs_xyz_m = np.abs(a[:3] - b[:3])
    translation_m = float(np.linalg.norm(abs_xyz_m))
    dot = float(np.dot(qa / norm_a, qb / norm_b))
    dot = min(1.0, max(-1.0, abs(dot)))
    rotation_deg = float(math.degrees(2.0 * math.acos(dot)))
    return translation_m, rotation_deg, abs_xyz_m


def needed_columns(available: set[str]) -> list[str]:
    preferred = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "action",
        "observation.state",
        "observation.last_action",
        "action.arm.pose",
        "action.left_arm.pose",
        "action.right_arm.pose",
        "observation.arm.pose",
        "observation.left_arm.pose",
        "observation.right_arm.pose",
        "observation.arm.left.current_tcp_pose",
        "observation.arm.right.current_tcp_pose",
        "observation.arm.left.actual_tcp_pose",
        "observation.arm.right.actual_tcp_pose",
        "teleop.pose_quality.status",
        "teleop.pose_quality.bad",
        "teleop.pose_quality.score",
        "teleop.pose_quality.special_code",
        "teleop.arm_mode.left.label",
        "teleop.arm_mode.right.label",
    }
    return sorted(preferred & available)


def load_episode_rows(parquet_path: Path) -> list[dict[str, Any]]:
    pq = require_pyarrow_parquet()
    parquet = pq.ParquetFile(parquet_path)
    columns = needed_columns(set(parquet.schema_arrow.names))
    missing_core = {"action", "observation.state"} - set(columns)
    if missing_core:
        raise ValueError(
            f"{parquet_path} is missing required column(s): {sorted(missing_core)}"
        )
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def analyze_dataset(
    dataset_root: Path,
    *,
    max_episodes: int | None = None,
    translation_threshold_m: float = DEFAULT_TRANSLATION_THRESHOLD_M,
    rotation_threshold_deg: float = DEFAULT_ROTATION_THRESHOLD_DEG,
    max_outliers: int = 50,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    records = episode_parquet_records(dataset_root)
    if max_episodes is not None:
        records = records[: max(0, int(max_episodes))]

    accumulators = {
        relation: {side: MetricAccumulator() for side in SIDE_ORDER}
        for relation in RELATIONS
    }
    episode_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    total_rows = 0

    for record in records:
        parquet_path = Path(str(record["parquet_abs_path"]))
        episode_index = record.get("episode_index")
        try:
            rows = load_episode_rows(parquet_path)
        except Exception as exc:  # noqa: BLE001 - keep scanning the remaining dataset.
            errors.append(
                {
                    "episode_index": episode_index,
                    "parquet_path": str(parquet_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        total_rows += len(rows)
        episode_pair_counts = {
            relation: {side: 0 for side in SIDE_ORDER}
            for relation in RELATIONS
        }

        for index, row in enumerate(rows):
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            previous_row = rows[index - 1] if index > 0 else None
            timestamp = float(row.get("timestamp")) if row.get("timestamp") is not None else None
            next_timestamp = (
                float(next_row.get("timestamp"))
                if next_row is not None and next_row.get("timestamp") is not None
                else None
            )
            for side in SIDE_ORDER:
                action_pose = pose_from_row(row, side=side, source="action")
                same_state_pose = pose_from_row(row, side=side, source="state")
                same_result = accumulators[RELATION_SAME_STATE][side].add(
                    action_pose,
                    same_state_pose,
                    dt_sec=0.0,
                )
                if same_result is not None:
                    episode_pair_counts[RELATION_SAME_STATE][side] += 1
                    maybe_add_outlier(
                        outliers,
                        relation=RELATION_SAME_STATE,
                        side=side,
                        dataset_root=dataset_root,
                        record=record,
                        row=row,
                        result=same_result,
                        translation_threshold_m=translation_threshold_m,
                        rotation_threshold_deg=rotation_threshold_deg,
                        max_outliers=max_outliers,
                    )

                if next_row is not None:
                    next_state_pose = pose_from_row(next_row, side=side, source="state")
                    dt = (
                        float(next_timestamp - timestamp)
                        if timestamp is not None and next_timestamp is not None
                        else None
                    )
                    next_result = accumulators[RELATION_NEXT_STATE][side].add(
                        action_pose,
                        next_state_pose,
                        dt_sec=dt,
                    )
                    if next_result is not None:
                        episode_pair_counts[RELATION_NEXT_STATE][side] += 1
                        maybe_add_outlier(
                            outliers,
                            relation=RELATION_NEXT_STATE,
                            side=side,
                            dataset_root=dataset_root,
                            record=record,
                            row=row,
                            result=next_result,
                            translation_threshold_m=translation_threshold_m,
                            rotation_threshold_deg=rotation_threshold_deg,
                            max_outliers=max_outliers,
                        )

                if previous_row is not None:
                    previous_action_pose = pose_from_row(previous_row, side=side, source="action")
                    last_action_pose = pose_from_row(row, side=side, source="last_action")
                    last_result = accumulators[RELATION_LAST_ACTION][side].add(
                        previous_action_pose,
                        last_action_pose,
                        dt_sec=None,
                    )
                    if last_result is not None:
                        episode_pair_counts[RELATION_LAST_ACTION][side] += 1
                        maybe_add_outlier(
                            outliers,
                            relation=RELATION_LAST_ACTION,
                            side=side,
                            dataset_root=dataset_root,
                            record=record,
                            row=row,
                            result=last_result,
                            translation_threshold_m=translation_threshold_m,
                            rotation_threshold_deg=rotation_threshold_deg,
                            max_outliers=max_outliers,
                        )

        episode_summaries.append(
            {
                "episode_index": episode_index,
                "episode_id": record.get("episode_id"),
                "parquet_path": record.get("parquet_path"),
                "row_count": int(len(rows)),
                "pair_counts": episode_pair_counts,
            }
        )

    relation_summary = {
        relation: {
            side: accumulators[relation][side].to_summary(
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_deg=rotation_threshold_deg,
            )
            for side in SIDE_ORDER
        }
        for relation in RELATIONS
    }
    dataset_name = str(info.get("dataset_name") or dataset_root.name)
    return {
        "dataset_name": dataset_name,
        "dataset_dir_name": dataset_root.name,
        "dataset_label": dataset_root.name
        if dataset_name == dataset_root.name
        else f"{dataset_root.name} (meta: {dataset_name})",
        "dataset_root": str(dataset_root),
        "info_frame_count": info.get("frame_count"),
        "info_episode_count": info.get("episode_count"),
        "dataset_schema": info.get("dexteleop_dataset_schema"),
        "pose_storage_frame": info.get("pose_storage_frame"),
        "policy_io_layout_hash": info.get("policy_io_layout_hash"),
        "scanned_episode_count": int(len(records)),
        "scanned_row_count": int(total_rows),
        "thresholds": {
            "translation_m": float(translation_threshold_m),
            "rotation_deg": float(rotation_threshold_deg),
        },
        "features": info.get("features", {}),
        "arm_action_specs_by_side": info.get("arm_action_specs_by_side", {}),
        "relations": relation_summary,
        "best_pose_relation_by_translation_mean": best_pose_relation(relation_summary),
        "episodes": episode_summaries,
        "outliers": outliers,
        "errors": errors,
    }


def maybe_add_outlier(
    outliers: list[dict[str, Any]],
    *,
    relation: str,
    side: str,
    dataset_root: Path,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    result: tuple[float, float],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    max_outliers: int,
) -> None:
    if len(outliers) >= int(max_outliers):
        return
    translation_m, rotation_deg = result
    if translation_m <= float(translation_threshold_m) and rotation_deg <= float(rotation_threshold_deg):
        return
    outliers.append(
        {
            "dataset_name": dataset_root.name,
            "episode_index": record.get("episode_index"),
            "episode_id": record.get("episode_id"),
            "parquet_path": record.get("parquet_path"),
            "frame_index": row.get("frame_index"),
            "index": row.get("index"),
            "timestamp": row.get("timestamp"),
            "relation": relation,
            "side": side,
            "translation_m": float(translation_m),
            "rotation_deg": float(rotation_deg),
        }
    )


def best_pose_relation(relation_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (RELATION_SAME_STATE, RELATION_NEXT_STATE)
    best: dict[str, Any] = {}
    for side in SIDE_ORDER:
        values: list[tuple[str, float]] = []
        for relation in candidates:
            summary = relation_summary.get(relation, {}).get(side, {})
            mean = summary.get("translation_error_m", {}).get("mean")
            if mean is not None:
                values.append((relation, float(mean)))
        if not values:
            best[side] = None
        else:
            relation, mean = min(values, key=lambda item: item[1])
            best[side] = {
                "relation": relation,
                "translation_mean_m": mean,
            }
    return best


def analyze_all(config: DatasetAnalysisConfig) -> dict[str, Any]:
    dataset_roots = discover_dataset_roots(
        config.dataset_root,
        dataset_glob=config.dataset_glob,
        include_dataset=config.include_dataset,
        exclude_dataset=config.exclude_dataset,
    )
    datasets = [
        analyze_dataset(
            dataset_root,
            max_episodes=config.max_episodes,
            translation_threshold_m=config.translation_threshold_m,
            rotation_threshold_deg=config.rotation_threshold_deg,
            max_outliers=config.max_outliers,
        )
        for dataset_root in dataset_roots
    ]
    return {
        "schema": "dexteleop_pose_action_next_state_summary_v1",
        "dataset_root": str(config.dataset_root.expanduser().resolve()),
        "dataset_count": int(len(datasets)),
        "datasets": datasets,
        "overall": overall_summary(datasets),
    }


def overall_summary(datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scanned_episode_count": int(sum(int(item.get("scanned_episode_count", 0)) for item in datasets)),
        "scanned_row_count": int(sum(int(item.get("scanned_row_count", 0)) for item in datasets)),
        "dataset_count": int(len(datasets)),
        "relations": {},
    }
    for relation in RELATIONS:
        summary["relations"][relation] = {}
        for side in SIDE_ORDER:
            translations: list[float] = []
            rotations: list[float] = []
            pair_count = 0
            for dataset in datasets:
                stats = dataset.get("relations", {}).get(relation, {}).get(side, {})
                pair_count += int(stats.get("pair_count", 0) or 0)
                translation_stats = stats.get("translation_error_m", {})
                rotation_stats = stats.get("rotation_error_deg", {})
                mean_translation = translation_stats.get("mean")
                mean_rotation = rotation_stats.get("mean")
                count = int(translation_stats.get("count", 0) or 0)
                if mean_translation is not None and count > 0:
                    translations.extend([float(mean_translation)] * count)
                if mean_rotation is not None and count > 0:
                    rotations.extend([float(mean_rotation)] * count)
            summary["relations"][relation][side] = {
                "pair_count": int(pair_count),
                "translation_error_m": numeric_summary(translations),
                "rotation_error_deg": numeric_summary(rotations),
            }
    return summary


def write_reports(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pose_action_next_state_summary.json"
    md_path = output_dir / "pose_action_next_state_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# Pose Action vs Next State Summary",
        "",
        f"- Dataset root: `{report.get('dataset_root')}`",
        f"- Dataset count: {report.get('dataset_count')}",
        f"- Scanned episodes: {report.get('overall', {}).get('scanned_episode_count')}",
        f"- Scanned rows: {report.get('overall', {}).get('scanned_row_count')}",
        "",
        "## Dataset Summary",
        "",
        "| dataset | rows | left same mean mm/deg | left next mean mm/deg | right same mean mm/deg | right next mean mm/deg | best left | best right |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset in report.get("datasets", []):
        relations = dataset.get("relations", {})
        best = dataset.get("best_pose_relation_by_translation_mean", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset.get("dataset_label") or dataset.get("dataset_name")),
                    str(dataset.get("scanned_row_count")),
                    relation_cell(relations, RELATION_SAME_STATE, "left"),
                    relation_cell(relations, RELATION_NEXT_STATE, "left"),
                    relation_cell(relations, RELATION_SAME_STATE, "right"),
                    relation_cell(relations, RELATION_NEXT_STATE, "right"),
                    best_cell(best.get("left")),
                    best_cell(best.get("right")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Last Action Check",
            "",
            "| dataset | left mean mm/deg | right mean mm/deg |",
            "|---|---:|---:|",
        ]
    )
    for dataset in report.get("datasets", []):
        relations = dataset.get("relations", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset.get("dataset_label") or dataset.get("dataset_name")),
                    relation_cell(relations, RELATION_LAST_ACTION, "left"),
                    relation_cell(relations, RELATION_LAST_ACTION, "right"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `same_state` compares `action_t` with `observation.state_t` arm pose.",
            "- `next_state` compares `action_t` with `observation.state_{t+1}` arm pose within the same episode.",
            "- `last_action` checks `observation.last_action_t` against `action_{t-1}`.",
            "- Pose layout is `[x, y, z, qx, qy, qz, qw]`; quaternion sign is treated as equivalent.",
            "",
        ]
    )
    return "\n".join(lines)


def relation_cell(relations: Mapping[str, Any], relation: str, side: str) -> str:
    stats = relations.get(relation, {}).get(side, {})
    t = stats.get("translation_error_m", {}).get("mean")
    r = stats.get("rotation_error_deg", {}).get("mean")
    if t is None or r is None:
        return "n/a"
    return f"{float(t) * 1000.0:.3f}/{float(r):.3f}"


def best_cell(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    relation = value.get("relation")
    mean = value.get("translation_mean_m")
    if relation is None or mean is None:
        return "n/a"
    return f"{relation} ({float(mean) * 1000.0:.3f} mm)"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check pose action against same-frame state, next-frame state, "
            "and observation.last_action for DexTeleop LeRobot-style datasets."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Dataset root or directory containing multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for JSON and Markdown summaries.",
    )
    parser.add_argument(
        "--dataset-glob",
        default="*",
        help="Glob used when --dataset-root contains multiple datasets.",
    )
    parser.add_argument(
        "--include-dataset",
        action="append",
        default=[],
        help="Substring filter for dataset directory names. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=["pi_template"],
        help="Substring exclusion for dataset directory names. Can be repeated.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional per-dataset episode scan limit for quick checks.",
    )
    parser.add_argument(
        "--translation-threshold-m",
        type=float,
        default=DEFAULT_TRANSLATION_THRESHOLD_M,
        help="Outlier threshold for translation error in meters.",
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=DEFAULT_ROTATION_THRESHOLD_DEG,
        help="Outlier threshold for rotation error in degrees.",
    )
    parser.add_argument(
        "--max-outliers",
        type=int,
        default=50,
        help="Maximum outlier rows to keep per dataset.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary only; do not write report files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = DatasetAnalysisConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        max_episodes=args.max_episodes,
        dataset_glob=str(args.dataset_glob),
        include_dataset=tuple(args.include_dataset or ()),
        exclude_dataset=tuple(args.exclude_dataset or ()),
        translation_threshold_m=float(args.translation_threshold_m),
        rotation_threshold_deg=float(args.rotation_threshold_deg),
        max_outliers=int(args.max_outliers),
    )
    report = analyze_all(config)
    if not bool(args.no_write):
        json_path, md_path = write_reports(report, config.output_dir)
        print(f"Wrote JSON: {json_path}")
        print(f"Wrote Markdown: {md_path}")
    print(render_console_summary(report))
    return 0


def render_console_summary(report: Mapping[str, Any]) -> str:
    lines = [
        (
            f"Scanned {report.get('dataset_count')} dataset(s), "
            f"{report.get('overall', {}).get('scanned_episode_count')} episode(s), "
            f"{report.get('overall', {}).get('scanned_row_count')} row(s)."
        )
    ]
    for dataset in report.get("datasets", []):
        relations = dataset.get("relations", {})
        label = dataset.get("dataset_label") or dataset.get("dataset_name")
        lines.append(f"- {label}: rows={dataset.get('scanned_row_count')}")
        for side in SIDE_ORDER:
            same = relation_cell(relations, RELATION_SAME_STATE, side)
            next_ = relation_cell(relations, RELATION_NEXT_STATE, side)
            last = relation_cell(relations, RELATION_LAST_ACTION, side)
            lines.append(f"  {side}: same={same}, next={next_}, last_action={last}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
