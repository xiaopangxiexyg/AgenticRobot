from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.statistic.pose_action_next_state import (
    SIDE_ORDER,
    discover_dataset_roots,
    episode_parquet_records,
    load_episode_rows,
    numeric_summary,
    pose_error,
    pose_from_row,
    read_json,
)


DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "pose_action_lag_scan_reports"
DEFAULT_MAX_LAG = 15


@dataclass
class LagAccumulator:
    translation_m: list[float] = field(default_factory=list)
    rotation_deg: list[float] = field(default_factory=list)
    dt_sec: list[float] = field(default_factory=list)
    skipped_missing: int = 0
    skipped_nonfinite: int = 0

    def add(
        self,
        action_pose: np.ndarray | None,
        state_pose: np.ndarray | None,
        *,
        dt_sec: float | None,
    ) -> None:
        if action_pose is None or state_pose is None:
            self.skipped_missing += 1
            return
        errors = pose_error(action_pose, state_pose)
        if errors is None:
            self.skipped_nonfinite += 1
            return
        translation_m, rotation_deg, _abs_xyz = errors
        self.translation_m.append(float(translation_m))
        self.rotation_deg.append(float(rotation_deg))
        if dt_sec is not None and math.isfinite(float(dt_sec)):
            self.dt_sec.append(float(dt_sec))

    def to_summary(self) -> dict[str, Any]:
        return {
            "pair_count": int(len(self.translation_m)),
            "skipped_missing": int(self.skipped_missing),
            "skipped_nonfinite": int(self.skipped_nonfinite),
            "translation_error_m": numeric_summary(np.asarray(self.translation_m, dtype=float)),
            "rotation_error_deg": numeric_summary(np.asarray(self.rotation_deg, dtype=float)),
            "dt_sec": numeric_summary(np.asarray(self.dt_sec, dtype=float)),
        }


@dataclass(frozen=True)
class LagScanConfig:
    dataset_root: Path
    output_dir: Path
    max_lag: int = DEFAULT_MAX_LAG
    max_episodes: int | None = None
    episode_indices: tuple[int, ...] = ()
    latest_episode: bool = False
    dataset_glob: str = "*"
    include_dataset: tuple[str, ...] = ()
    exclude_dataset: tuple[str, ...] = ("pi_template",)
    dedupe: bool = True
    max_best_lag_samples: int = 200


@dataclass
class BestLagAccumulator:
    best_lag: list[int] = field(default_factory=list)
    best_translation_m: list[float] = field(default_factory=list)
    best_rotation_deg: list[float] = field(default_factory=list)
    best_dt_sec: list[float] = field(default_factory=list)
    margin_to_second_m: list[float] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    skipped_incomplete_horizon: int = 0

    def add(
        self,
        *,
        episode_index: Any,
        frame_index: Any,
        timestamp: float | None,
        lag: int,
        translation_m: float,
        rotation_deg: float,
        dt_sec: float | None,
        margin_to_second_m: float | None,
        max_samples: int,
    ) -> None:
        self.best_lag.append(int(lag))
        self.best_translation_m.append(float(translation_m))
        self.best_rotation_deg.append(float(rotation_deg))
        if dt_sec is not None and math.isfinite(float(dt_sec)):
            self.best_dt_sec.append(float(dt_sec))
        if margin_to_second_m is not None and math.isfinite(float(margin_to_second_m)):
            self.margin_to_second_m.append(float(margin_to_second_m))
        if len(self.samples) < int(max_samples):
            self.samples.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "best_lag": int(lag),
                    "best_translation_m": float(translation_m),
                    "best_rotation_deg": float(rotation_deg),
                    "best_dt_sec": None if dt_sec is None else float(dt_sec),
                    "margin_to_second_m": (
                        None if margin_to_second_m is None else float(margin_to_second_m)
                    ),
                }
            )

    def to_summary(self, *, max_lag: int) -> dict[str, Any]:
        lag_array = np.asarray(self.best_lag, dtype=int)
        histogram = {
            str(lag): int(np.sum(lag_array == lag))
            for lag in range(int(max_lag) + 1)
        }
        total = int(lag_array.size)
        mode_lag: int | None = None
        mode_fraction: float | None = None
        if total > 0:
            mode_lag = max(range(int(max_lag) + 1), key=lambda lag: histogram[str(lag)])
            mode_fraction = float(histogram[str(mode_lag)] / total)
        return {
            "sample_count": total,
            "best_lag": numeric_summary(lag_array.astype(float)),
            "best_lag_histogram": histogram,
            "mode_lag": mode_lag,
            "mode_lag_fraction": mode_fraction,
            "best_dt_sec": numeric_summary(np.asarray(self.best_dt_sec, dtype=float)),
            "best_translation_error_m": numeric_summary(
                np.asarray(self.best_translation_m, dtype=float)
            ),
            "best_rotation_error_deg": numeric_summary(
                np.asarray(self.best_rotation_deg, dtype=float)
            ),
            "margin_to_second_best_translation_m": numeric_summary(
                np.asarray(self.margin_to_second_m, dtype=float)
            ),
            "transition_count": _lag_transition_count(self.best_lag),
            "longest_constant_run": _longest_constant_lag_run(self.best_lag),
            "skipped_incomplete_horizon": int(self.skipped_incomplete_horizon),
            "samples": list(self.samples),
        }


def _dataset_key(dataset_root: Path) -> str:
    info = read_json(dataset_root / "meta" / "info.json")
    return str(info.get("dataset_name") or dataset_root.name)


def _dedupe_dataset_roots(dataset_roots: Sequence[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: list[Path] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for root in dataset_roots:
        key = _dataset_key(root)
        if key in seen:
            skipped.append(
                {
                    "dataset_root": str(root),
                    "dedupe_key": key,
                    "kept_dataset_root": str(seen[key]),
                }
            )
            continue
        seen[key] = root
        selected.append(root)
    return selected, skipped


def _timestamp(row: Mapping[str, Any]) -> float | None:
    value = row.get("timestamp")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _record_episode_index(record: Mapping[str, Any]) -> int:
    value = record.get("episode_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _record_parquet_mtime(record: Mapping[str, Any]) -> float:
    path = record.get("parquet_abs_path")
    if not isinstance(path, str) or not path:
        return 0.0
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


def _select_episode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    episode_indices: Sequence[int] = (),
    latest_episode: bool = False,
    max_episodes: int | None = None,
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records]
    if episode_indices:
        wanted = {int(index) for index in episode_indices}
        selected = [
            record for record in selected
            if _record_episode_index(record) in wanted
        ]
    if latest_episode and selected:
        selected = [
            sorted(
                selected,
                key=lambda record: (
                    _record_episode_index(record),
                    _record_parquet_mtime(record),
                    str(record.get("parquet_path", "")),
                ),
            )[-1]
        ]
    if max_episodes is not None:
        selected = selected[: max(0, int(max_episodes))]
    return selected


def _lag_transition_count(lags: Sequence[int]) -> int:
    if not lags:
        return 0
    transitions = 0
    previous = int(lags[0])
    for lag in lags[1:]:
        current = int(lag)
        if current != previous:
            transitions += 1
        previous = current
    return transitions


def _longest_constant_lag_run(lags: Sequence[int]) -> dict[str, Any] | None:
    if not lags:
        return None
    best_lag = int(lags[0])
    best_start = 0
    best_length = 1
    current_lag = int(lags[0])
    current_start = 0
    current_length = 1
    for index, lag_value in enumerate(lags[1:], start=1):
        lag = int(lag_value)
        if lag == current_lag:
            current_length += 1
        else:
            if current_length > best_length:
                best_lag = current_lag
                best_start = current_start
                best_length = current_length
            current_lag = lag
            current_start = index
            current_length = 1
    if current_length > best_length:
        best_lag = current_lag
        best_start = current_start
        best_length = current_length
    return {
        "lag": int(best_lag),
        "start_sample": int(best_start),
        "length": int(best_length),
    }


def _best_lag(lag_summaries: Mapping[str, Any], side: str) -> dict[str, Any] | None:
    candidates: list[tuple[int, float, float | None, int]] = []
    for lag_text, by_side in lag_summaries.items():
        side_summary = by_side.get(side, {})
        translation = side_summary.get("translation_error_m", {})
        rotation = side_summary.get("rotation_error_deg", {})
        mean = translation.get("mean")
        count = int(translation.get("count", 0) or 0)
        if mean is None or count <= 0:
            continue
        candidates.append((int(lag_text), float(mean), rotation.get("mean"), count))
    if not candidates:
        return None
    lag, translation_mean, rotation_mean, count = min(candidates, key=lambda item: item[1])
    return {
        "lag": int(lag),
        "translation_mean_m": float(translation_mean),
        "rotation_mean_deg": None if rotation_mean is None else float(rotation_mean),
        "pair_count": int(count),
    }


def _best_average_lag(lag_summaries: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[int, float, float, int]] = []
    for lag_text, by_side in lag_summaries.items():
        means: list[float] = []
        rotations: list[float] = []
        count = 0
        for side in SIDE_ORDER:
            side_summary = by_side.get(side, {})
            translation = side_summary.get("translation_error_m", {})
            rotation = side_summary.get("rotation_error_deg", {})
            mean = translation.get("mean")
            if mean is None:
                continue
            means.append(float(mean))
            count += int(translation.get("count", 0) or 0)
            rot = rotation.get("mean")
            if rot is not None:
                rotations.append(float(rot))
        if means:
            candidates.append(
                (
                    int(lag_text),
                    float(np.mean(means)),
                    float(np.mean(rotations)) if rotations else float("nan"),
                    int(count),
                )
            )
    if not candidates:
        return None
    lag, translation_mean, rotation_mean, count = min(candidates, key=lambda item: item[1])
    return {
        "lag": int(lag),
        "translation_mean_m": float(translation_mean),
        "rotation_mean_deg": None if not math.isfinite(rotation_mean) else float(rotation_mean),
        "pair_count": int(count),
    }


def analyze_dataset_lags(
    dataset_root: Path,
    *,
    max_lag: int,
    max_episodes: int | None = None,
    episode_indices: Sequence[int] = (),
    latest_episode: bool = False,
    max_best_lag_samples: int = 200,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    all_records = episode_parquet_records(dataset_root)
    records = _select_episode_records(
        all_records,
        episode_indices=episode_indices,
        latest_episode=latest_episode,
        max_episodes=max_episodes,
    )

    accumulators = {
        lag: {side: LagAccumulator() for side in SIDE_ORDER}
        for lag in range(int(max_lag) + 1)
    }
    best_lag_accumulators = {
        side: BestLagAccumulator()
        for side in SIDE_ORDER
    }
    errors: list[dict[str, Any]] = []
    scanned_rows = 0
    scanned_episodes = 0
    episode_summaries: list[dict[str, Any]] = []

    for record in records:
        parquet_path = Path(str(record["parquet_abs_path"]))
        try:
            rows = load_episode_rows(parquet_path)
        except Exception as exc:  # noqa: BLE001 - keep scanning remaining episodes.
            errors.append(
                {
                    "episode_index": record.get("episode_index"),
                    "parquet_path": str(parquet_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not rows:
            continue
        scanned_rows += len(rows)
        scanned_episodes += 1
        episode_summaries.append(
            {
                "episode_index": record.get("episode_index"),
                "episode_id": record.get("episode_id"),
                "parquet_path": record.get("parquet_path"),
                "row_count": int(len(rows)),
            }
        )

        for index, row in enumerate(rows):
            row_time = _timestamp(row)
            for side in SIDE_ORDER:
                action_pose = pose_from_row(row, side=side, source="action")
                frame_candidates: list[tuple[int, float, float, float | None]] = []
                for lag in range(int(max_lag) + 1):
                    target_index = index + lag
                    if target_index >= len(rows):
                        continue
                    target_row = rows[target_index]
                    target_time = _timestamp(target_row)
                    dt_sec = (
                        float(target_time - row_time)
                        if row_time is not None and target_time is not None
                        else None
                    )
                    state_pose = pose_from_row(target_row, side=side, source="state")
                    accumulators[lag][side].add(action_pose, state_pose, dt_sec=dt_sec)
                    if action_pose is None or state_pose is None:
                        continue
                    errors_for_lag = pose_error(action_pose, state_pose)
                    if errors_for_lag is None:
                        continue
                    translation_m, rotation_deg, _abs_xyz = errors_for_lag
                    frame_candidates.append(
                        (int(lag), float(translation_m), float(rotation_deg), dt_sec)
                    )
                if index + int(max_lag) >= len(rows):
                    best_lag_accumulators[side].skipped_incomplete_horizon += 1
                    continue
                if len(frame_candidates) < int(max_lag) + 1:
                    best_lag_accumulators[side].skipped_incomplete_horizon += 1
                    continue
                if frame_candidates:
                    frame_candidates.sort(key=lambda item: (item[1], item[0]))
                    best_lag, best_translation, best_rotation, best_dt = frame_candidates[0]
                    second_translation = (
                        frame_candidates[1][1] if len(frame_candidates) > 1 else None
                    )
                    best_lag_accumulators[side].add(
                        episode_index=record.get("episode_index"),
                        frame_index=row.get("frame_index", row.get("index")),
                        timestamp=row_time,
                        lag=best_lag,
                        translation_m=best_translation,
                        rotation_deg=best_rotation,
                        dt_sec=best_dt,
                        margin_to_second_m=(
                            None
                            if second_translation is None
                            else float(second_translation - best_translation)
                        ),
                        max_samples=max_best_lag_samples,
                    )

    lag_summaries = {
        str(lag): {
            side: accumulators[lag][side].to_summary()
            for side in SIDE_ORDER
        }
        for lag in range(int(max_lag) + 1)
    }
    dataset_name = str(info.get("dataset_name") or dataset_root.name)
    best_by_side = {
        side: _best_lag(lag_summaries, side)
        for side in SIDE_ORDER
    }
    return {
        "dataset_name": dataset_name,
        "dataset_dir_name": dataset_root.name,
        "dataset_label": dataset_root.name
        if dataset_name == dataset_root.name
        else f"{dataset_root.name} (meta: {dataset_name})",
        "dataset_root": str(dataset_root),
        "scanned_episode_count": int(scanned_episodes),
        "scanned_row_count": int(scanned_rows),
        "max_lag": int(max_lag),
        "lags": lag_summaries,
        "best_by_side": best_by_side,
        "best_average_lag": _best_average_lag(lag_summaries),
        "best_lag_variability": {
            side: best_lag_accumulators[side].to_summary(max_lag=int(max_lag))
            for side in SIDE_ORDER
        },
        "episodes": episode_summaries,
        "errors": errors,
    }


def analyze_all_lags(config: LagScanConfig) -> dict[str, Any]:
    roots = discover_dataset_roots(
        config.dataset_root,
        dataset_glob=config.dataset_glob,
        include_dataset=config.include_dataset,
        exclude_dataset=config.exclude_dataset,
    )
    skipped_duplicates: list[dict[str, Any]] = []
    if config.dedupe:
        roots, skipped_duplicates = _dedupe_dataset_roots(roots)
    datasets = [
        analyze_dataset_lags(
            root,
            max_lag=int(config.max_lag),
            max_episodes=config.max_episodes,
            episode_indices=config.episode_indices,
            latest_episode=bool(config.latest_episode),
            max_best_lag_samples=int(config.max_best_lag_samples),
        )
        for root in roots
    ]
    return {
        "schema": "dexteleop_pose_action_state_lag_scan_v1",
        "dataset_root": str(config.dataset_root.expanduser().resolve()),
        "max_lag": int(config.max_lag),
        "dedupe": bool(config.dedupe),
        "dataset_count": int(len(datasets)),
        "datasets": datasets,
        "skipped_duplicates": skipped_duplicates,
        "overall": _overall_best_lags(datasets, max_lag=int(config.max_lag)),
    }


def _overall_best_lags(datasets: Sequence[Mapping[str, Any]], *, max_lag: int) -> dict[str, Any]:
    pooled: dict[int, dict[str, dict[str, float]]] = {
        lag: {side: {"weighted_translation_sum": 0.0, "weighted_rotation_sum": 0.0, "count": 0.0} for side in SIDE_ORDER}
        for lag in range(int(max_lag) + 1)
    }
    for dataset in datasets:
        if int(dataset.get("scanned_row_count", 0) or 0) <= 0:
            continue
        lags = dataset.get("lags", {})
        for lag in range(int(max_lag) + 1):
            by_side = lags.get(str(lag), {})
            for side in SIDE_ORDER:
                summary = by_side.get(side, {})
                t = summary.get("translation_error_m", {})
                r = summary.get("rotation_error_deg", {})
                mean_t = t.get("mean")
                count = int(t.get("count", 0) or 0)
                if mean_t is None or count <= 0:
                    continue
                pooled[lag][side]["weighted_translation_sum"] += float(mean_t) * count
                mean_r = r.get("mean")
                if mean_r is not None:
                    pooled[lag][side]["weighted_rotation_sum"] += float(mean_r) * count
                pooled[lag][side]["count"] += float(count)
    summary_by_lag: dict[str, Any] = {}
    for lag, by_side in pooled.items():
        summary_by_lag[str(lag)] = {}
        for side, values in by_side.items():
            count = int(values["count"])
            if count <= 0:
                summary_by_lag[str(lag)][side] = {
                    "pair_count": 0,
                    "translation_mean_m": None,
                    "rotation_mean_deg": None,
                }
                continue
            summary_by_lag[str(lag)][side] = {
                "pair_count": count,
                "translation_mean_m": values["weighted_translation_sum"] / count,
                "rotation_mean_deg": values["weighted_rotation_sum"] / count,
            }
    return {
        "lags": summary_by_lag,
        "best_by_side": {
            side: _best_overall_side(summary_by_lag, side)
            for side in SIDE_ORDER
        },
        "best_average_lag": _best_overall_average(summary_by_lag),
        "scanned_episode_count": int(sum(int(d.get("scanned_episode_count", 0) or 0) for d in datasets)),
        "scanned_row_count": int(sum(int(d.get("scanned_row_count", 0) or 0) for d in datasets)),
    }


def _best_overall_side(summary_by_lag: Mapping[str, Any], side: str) -> dict[str, Any] | None:
    candidates: list[tuple[int, float, float | None, int]] = []
    for lag_text, by_side in summary_by_lag.items():
        item = by_side.get(side, {})
        mean = item.get("translation_mean_m")
        count = int(item.get("pair_count", 0) or 0)
        if mean is None or count <= 0:
            continue
        candidates.append((int(lag_text), float(mean), item.get("rotation_mean_deg"), count))
    if not candidates:
        return None
    lag, mean, rotation, count = min(candidates, key=lambda item: item[1])
    return {
        "lag": int(lag),
        "translation_mean_m": float(mean),
        "rotation_mean_deg": None if rotation is None else float(rotation),
        "pair_count": int(count),
    }


def _best_overall_average(summary_by_lag: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[int, float, float | None, int]] = []
    for lag_text, by_side in summary_by_lag.items():
        means: list[float] = []
        rotations: list[float] = []
        count = 0
        for side in SIDE_ORDER:
            item = by_side.get(side, {})
            mean = item.get("translation_mean_m")
            side_count = int(item.get("pair_count", 0) or 0)
            if mean is None or side_count <= 0:
                continue
            means.append(float(mean))
            count += side_count
            rotation = item.get("rotation_mean_deg")
            if rotation is not None:
                rotations.append(float(rotation))
        if means:
            candidates.append(
                (
                    int(lag_text),
                    float(np.mean(means)),
                    float(np.mean(rotations)) if rotations else None,
                    count,
                )
            )
    if not candidates:
        return None
    lag, mean, rotation, count = min(candidates, key=lambda item: item[1])
    return {
        "lag": int(lag),
        "translation_mean_m": float(mean),
        "rotation_mean_deg": None if rotation is None else float(rotation),
        "pair_count": int(count),
    }


def _mm_deg(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    lag = value.get("lag")
    translation = value.get("translation_mean_m")
    rotation = value.get("rotation_mean_deg")
    if lag is None or translation is None:
        return "n/a"
    rot_text = "n/a" if rotation is None else f"{float(rotation):.2f}deg"
    return f"n+{int(lag)} ({float(translation) * 1000.0:.2f}mm/{rot_text})"


def _lag_cell(dataset: Mapping[str, Any], side: str, lag: int) -> str:
    summary = dataset.get("lags", {}).get(str(lag), {}).get(side, {})
    mean = summary.get("translation_error_m", {}).get("mean")
    if mean is None:
        return "n/a"
    return f"{float(mean) * 1000.0:.1f}"


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# Pose Action State Lag Scan",
        "",
        f"- Dataset root: `{report.get('dataset_root')}`",
        f"- Max lag: n+{report.get('max_lag')}",
        f"- Dedupe enabled: {report.get('dedupe')}",
        f"- Scanned datasets: {report.get('dataset_count')}",
        f"- Scanned episodes: {report.get('overall', {}).get('scanned_episode_count')}",
        f"- Scanned rows: {report.get('overall', {}).get('scanned_row_count')}",
        f"- Skipped duplicate datasets: {len(report.get('skipped_duplicates', []))}",
        "",
        "## Overall Best Lag",
        "",
        "| side | best lag by mean translation |",
        "|---|---:|",
    ]
    overall = report.get("overall", {})
    for side in SIDE_ORDER:
        lines.append(f"| {side} | {_mm_deg(overall.get('best_by_side', {}).get(side))} |")
    lines.append(f"| average(left,right) | {_mm_deg(overall.get('best_average_lag'))} |")
    lines.extend(
        [
            "",
            "## Dataset Best Lag",
            "",
            "| dataset | rows | left best | right best | avg best |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in report.get("datasets", []):
        if int(dataset.get("scanned_row_count", 0) or 0) <= 0:
            continue
        label = dataset.get("dataset_label") or dataset.get("dataset_name")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(label),
                    str(dataset.get("scanned_row_count")),
                    _mm_deg(dataset.get("best_by_side", {}).get("left")),
                    _mm_deg(dataset.get("best_by_side", {}).get("right")),
                    _mm_deg(dataset.get("best_average_lag")),
                ]
            )
            + " |"
        )
    sample_lags = [0, 1, 2, 3, 5, 8, 10, 12, 15]
    sample_lags = [lag for lag in sample_lags if lag <= int(report.get("max_lag", 0))]
    lines.extend(
        [
            "",
            "## Best Lag Variability",
            "",
            "| dataset | side | mode lag | mode fraction | lag mean | lag std | transitions | longest run | best error mean mm | margin mean mm |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for dataset in report.get("datasets", []):
        if int(dataset.get("scanned_row_count", 0) or 0) <= 0:
            continue
        label = str(dataset.get("dataset_label") or dataset.get("dataset_name"))
        variability = dataset.get("best_lag_variability", {})
        for side in SIDE_ORDER:
            item = variability.get(side, {})
            lag_summary = item.get("best_lag", {})
            best_error = item.get("best_translation_error_m", {})
            margin = item.get("margin_to_second_best_translation_m", {})
            longest = item.get("longest_constant_run")
            longest_text = (
                "n/a"
                if not isinstance(longest, Mapping)
                else f"n+{longest.get('lag')} x {longest.get('length')}"
            )
            mode = item.get("mode_lag")
            mode_text = "n/a" if mode is None else f"n+{int(mode)}"
            mode_fraction = item.get("mode_lag_fraction")
            row = [
                label,
                side,
                mode_text,
                "n/a" if mode_fraction is None else f"{float(mode_fraction):.3f}",
                "n/a" if lag_summary.get("mean") is None else f"{float(lag_summary['mean']):.2f}",
                "n/a" if lag_summary.get("std") is None else f"{float(lag_summary['std']):.2f}",
                str(item.get("transition_count")),
                longest_text,
                "n/a" if best_error.get("mean") is None else f"{float(best_error['mean']) * 1000.0:.2f}",
                "n/a" if margin.get("mean") is None else f"{float(margin['mean']) * 1000.0:.2f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Mean Translation Error By Lag (mm)",
            "",
        ]
    )
    header = ["dataset", "side", *(f"n+{lag}" for lag in sample_lags)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---|---" + "|---:" * len(sample_lags) + "|")
    for dataset in report.get("datasets", []):
        if int(dataset.get("scanned_row_count", 0) or 0) <= 0:
            continue
        label = str(dataset.get("dataset_label") or dataset.get("dataset_name"))
        for side in SIDE_ORDER:
            row = [label, side, *(_lag_cell(dataset, side, lag) for lag in sample_lags)]
            lines.append("| " + " | ".join(row) + " |")
    if report.get("skipped_duplicates"):
        lines.extend(
            [
                "",
                "## Skipped Duplicates",
                "",
                "| skipped dataset | dedupe key | kept dataset |",
                "|---|---|---|",
            ]
        )
        for item in report.get("skipped_duplicates", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(Path(str(item.get("dataset_root"))).name),
                        str(item.get("dedupe_key")),
                        str(Path(str(item.get("kept_dataset_root"))).name),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `n+k` compares `action_t` with `observation.state_{t+k}` inside the same episode.",
            "- Best lag is selected by mean translation error; rotation mean is shown for that same lag.",
            "- Duplicate filtering uses `meta/info.json.dataset_name` as the default key.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pose_action_lag_scan_summary.json"
    md_path = output_dir / "pose_action_lag_scan_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan action_t vs observation.state_{t+k}, k=0..max_lag, for DexTeleop datasets."
    )
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-index",
        action="append",
        type=int,
        default=[],
        help="Only scan the given episode index. May be repeated.",
    )
    parser.add_argument(
        "--latest-episode",
        action="store_true",
        help="Only scan the latest episode by episode_index, mtime, then parquet path.",
    )
    parser.add_argument(
        "--max-best-lag-samples",
        type=int,
        default=200,
        help="Maximum per-frame best-lag sample rows to include in JSON per side.",
    )
    parser.add_argument("--dataset-glob", default="*")
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--exclude-dataset", action="append", default=["pi_template"])
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def _console_summary(report: Mapping[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        (
            f"Scanned {report.get('dataset_count')} dataset(s), "
            f"{overall.get('scanned_episode_count')} episode(s), "
            f"{overall.get('scanned_row_count')} row(s); "
            f"skipped duplicates={len(report.get('skipped_duplicates', []))}."
        ),
        "Overall best: "
        + ", ".join(
            f"{side}={_mm_deg(overall.get('best_by_side', {}).get(side))}"
            for side in SIDE_ORDER
        )
        + f", avg={_mm_deg(overall.get('best_average_lag'))}",
    ]
    for dataset in report.get("datasets", []):
        if int(dataset.get("scanned_row_count", 0) or 0) <= 0:
            continue
        label = dataset.get("dataset_label") or dataset.get("dataset_name")
        lines.append(
            f"- {label}: "
            + ", ".join(
                f"{side}={_mm_deg(dataset.get('best_by_side', {}).get(side))}"
                for side in SIDE_ORDER
            )
            + f", avg={_mm_deg(dataset.get('best_average_lag'))}"
        )
        variability = dataset.get("best_lag_variability", {})
        for side in SIDE_ORDER:
            item = variability.get(side, {})
            mode_lag = item.get("mode_lag")
            mode_fraction = item.get("mode_lag_fraction")
            transitions = item.get("transition_count")
            longest = item.get("longest_constant_run")
            longest_text = (
                "n/a"
                if not isinstance(longest, Mapping)
                else f"n+{longest.get('lag')}x{longest.get('length')}"
            )
            if mode_lag is None:
                continue
            lines.append(
                f"  {side} frame-best lag: mode=n+{int(mode_lag)} "
                f"({float(mode_fraction or 0.0):.1%}), "
                f"transitions={transitions}, longest={longest_text}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = LagScanConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        max_lag=max(0, int(args.max_lag)),
        max_episodes=args.max_episodes,
        episode_indices=tuple(int(index) for index in (args.episode_index or ())),
        latest_episode=bool(args.latest_episode),
        dataset_glob=str(args.dataset_glob),
        include_dataset=tuple(args.include_dataset or ()),
        exclude_dataset=tuple(args.exclude_dataset or ()),
        dedupe=not bool(args.no_dedupe),
        max_best_lag_samples=max(0, int(args.max_best_lag_samples)),
    )
    report = analyze_all_lags(config)
    if not bool(args.no_write):
        json_path, md_path = write_reports(report, config.output_dir)
        print(f"Wrote JSON: {json_path}")
        print(f"Wrote Markdown: {md_path}")
    print(_console_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
