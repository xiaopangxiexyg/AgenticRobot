from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..common import nan_vector
from ..utils.writer import (
    ACTION_DIM,
    ARM_DOF,
    HAND_DOF,
    HAND_VECTOR_ORDER,
    chunk_id_from_episode_index,
    episode_id_from_index,
)
from .observation import DeploymentObservation
from .policy import LoadedPolicy


_ARM_SLICE = slice(0, ARM_DOF)
_HAND_SLICES = {
    "right": slice(ARM_DOF, ARM_DOF + HAND_DOF),
    "left": slice(ARM_DOF + HAND_DOF, ACTION_DIM),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _positive_float_or_none(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0.0:
        return None
    return float(number)


def _episode_stats_record(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    for row in _read_jsonl(dataset_root / "meta" / "episodes_stats.jsonl"):
        try:
            row_episode_index = int(row.get("episode_index", -1))
        except (TypeError, ValueError):
            continue
        if row_episode_index == int(episode_index):
            return row
    return {}


def _estimate_hz_from_timestamps(values: Sequence[Any]) -> dict[str, Any] | None:
    timestamps = np.asarray(
        [float(value) for value in values if _float_or_none(value) is not None],
        dtype=float,
    )
    if timestamps.shape[0] < 2:
        return None
    dts = np.diff(timestamps)
    dts = dts[np.isfinite(dts) & (dts > 1e-9)]
    if dts.size <= 0:
        return None
    median_dt = float(np.median(dts))
    if median_dt <= 0.0:
        return None
    return {
        "hz": float(1.0 / median_dt),
        "source": "parquet_timestamp_median",
        "sample_count": int(timestamps.shape[0]),
        "interval_count": int(dts.size),
        "median_dt_sec": median_dt,
        "mean_dt_sec": float(np.mean(dts)),
        "min_dt_sec": float(np.min(dts)),
        "max_dt_sec": float(np.max(dts)),
    }


def _estimate_hz_from_control_hz(values: Sequence[Any]) -> dict[str, Any] | None:
    hz_values = np.asarray(
        [float(value) for value in values if _positive_float_or_none(value) is not None],
        dtype=float,
    )
    hz_values = hz_values[np.isfinite(hz_values) & (hz_values > 0.0)]
    if hz_values.size <= 0:
        return None
    return {
        "hz": float(np.median(hz_values)),
        "source": "parquet_control_hz_median",
        "sample_count": int(hz_values.size),
        "mean_hz": float(np.mean(hz_values)),
        "min_hz": float(np.min(hz_values)),
        "max_hz": float(np.max(hz_values)),
    }


def _estimate_hz_from_duration(frame_count: Any, duration_sec: Any, *, source: str) -> dict[str, Any] | None:
    count = _float_or_none(frame_count)
    duration = _positive_float_or_none(duration_sec)
    if count is None or count < 2.0 or duration is None:
        return None
    hz = float((count - 1.0) / duration)
    if hz <= 0.0 or not np.isfinite(hz):
        return None
    return {
        "hz": hz,
        "source": source,
        "frame_count": int(count),
        "duration_sec": float(duration),
    }


def _estimate_hz_from_dataset_metadata(
    *,
    dataset_root: str | Path | None,
    episode_index: int,
    resolved_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None and str(dataset_root) else None
    stats = _episode_stats_record(root, int(episode_index)) if root is not None and root.exists() else {}

    for key in ("average_hz", "nominal_fps", "mean_control_hz"):
        value = _positive_float_or_none(stats.get(key))
        if value is not None:
            return {
                "hz": value,
                "source": f"episodes_stats.{key}",
                "episode_index": int(episode_index),
            }

    estimate = _estimate_hz_from_duration(
        resolved_metadata.get("frame_count"),
        resolved_metadata.get("duration_sec"),
        source="episodes.duration_sec",
    )
    if estimate is not None:
        estimate["episode_index"] = int(episode_index)
        return estimate

    episode_metadata = resolved_metadata.get("episode_metadata")
    if isinstance(episode_metadata, Mapping):
        for key in ("hz", "nominal_fps"):
            value = _positive_float_or_none(episode_metadata.get(key))
            if value is not None:
                return {
                    "hz": value,
                    "source": f"episodes.metadata.{key}",
                    "episode_index": int(episode_index),
                }

    dataset_info = resolved_metadata.get("dataset_info")
    if not isinstance(dataset_info, Mapping) and root is not None and root.exists():
        dataset_info = _load_dataset_info(root)
    if isinstance(dataset_info, Mapping):
        value = _positive_float_or_none(dataset_info.get("nominal_fps"))
        if value is not None:
            return {
                "hz": value,
                "source": "info.nominal_fps",
                "episode_index": int(episode_index),
            }
    return None


def resolve_replay_parquet_path(
    *,
    dataset_root: str | Path | None,
    parquet_path: str | Path | None,
    episode_index: int,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a collected episode parquet from a dataset root or direct path."""

    if parquet_path is not None and str(parquet_path):
        path = Path(parquet_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Replay parquet does not exist: {path}")
        return path, {"parquet_path": str(path), "source": "direct_parquet"}

    if dataset_root is None or not str(dataset_root):
        raise ValueError("Replay mode requires --debug-replay-dataset or --debug-replay-parquet.")

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Replay dataset root does not exist: {root}")

    info = _load_dataset_info(root)
    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")
    for row in episode_rows:
        if int(row.get("episode_index", -1)) != int(episode_index):
            continue
        relative = row.get("parquet_path")
        if not relative:
            raise ValueError(f"Episode {episode_index} has no parquet_path in meta/episodes.jsonl.")
        path = (root / str(relative)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Replay episode parquet does not exist: {path}")
        metadata = {
            "dataset_root": str(root),
            "episode_index": int(episode_index),
            "episode_id": row.get("episode_id"),
            "frame_count": row.get("frame_count"),
            "duration_sec": row.get("duration_sec"),
            "episode_metadata": row.get("metadata", {}),
            "dataset_info": info,
            "source": "dataset_meta",
        }
        return path, metadata

    chunk_size = int(info.get("chunk_size", 1000) or 1000)
    fallback = (
        root
        / "data"
        / chunk_id_from_episode_index(int(episode_index), chunk_size=chunk_size)
        / f"{episode_id_from_index(int(episode_index))}.parquet"
    ).resolve()
    if fallback.exists():
        return fallback, {
            "dataset_root": str(root),
            "episode_index": int(episode_index),
            "episode_id": episode_id_from_index(int(episode_index)),
            "dataset_info": info,
            "source": "dataset_fallback_path",
        }
    raise FileNotFoundError(
        f"Could not find episode {episode_index} in {root / 'meta' / 'episodes.jsonl'} "
        f"or at fallback path {fallback}."
    )


def _require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dataset replay requires pyarrow in the active Python environment."
        ) from exc
    return pq


def _full_action_from_value(value: Any, *, row_index: int, column: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(
            f"Replay row {row_index} column {column!r} must have shape ({ACTION_DIM},), "
            f"got {vector.shape}."
        )
    return vector.astype(np.float32, copy=True)


def _component_from_row(
    row: Mapping[str, Any],
    key: str,
    *,
    expected_dim: int,
    row_index: int,
) -> np.ndarray | None:
    if key not in row or row[key] is None:
        return None
    vector = np.asarray(row[key], dtype=np.float32).reshape(-1)
    if vector.shape != (expected_dim,):
        raise ValueError(
            f"Replay row {row_index} column {key!r} must have shape ({expected_dim},), "
            f"got {vector.shape}."
        )
    return vector.astype(np.float32, copy=True)


def _action_from_component_columns(row: Mapping[str, Any], *, row_index: int) -> np.ndarray:
    action = nan_vector(ACTION_DIM)
    arm = _component_from_row(
        row,
        "action.arm.joint_target",
        expected_dim=ARM_DOF,
        row_index=row_index,
    )
    if arm is not None:
        action[_ARM_SLICE] = arm
    for side in HAND_VECTOR_ORDER:
        hand = _component_from_row(
            row,
            f"action.{side}_hand.joint_target",
            expected_dim=HAND_DOF,
            row_index=row_index,
        )
        if hand is not None:
            action[_HAND_SLICES[side]] = hand
    if not np.any(np.isfinite(action)):
        raise ValueError(
            "Replay parquet does not contain an action column or any supported "
            "action.<effector>.joint_target component columns."
        )
    return action


def load_replay_actions_from_parquet(
    parquet_path: str | Path,
    *,
    action_column: str = "action",
    start_index: int = 0,
    end_index: int | None = None,
) -> np.ndarray:
    """Load absolute action vectors from a collected episode parquet."""

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    component_columns = [
        "action.arm.joint_target",
        "action.right_hand.joint_target",
        "action.left_hand.joint_target",
    ]
    requested_action_column = str(action_column or "action")
    if requested_action_column in available_columns:
        columns = [requested_action_column]
        use_full_action = True
    else:
        columns = [column for column in component_columns if column in available_columns]
        use_full_action = False
    if not columns:
        raise ValueError(
            f"Replay parquet {path} does not contain {requested_action_column!r} or "
            "supported component action columns."
        )

    rows = pq.read_table(path, columns=columns).to_pylist()
    if not rows:
        raise ValueError(f"Replay parquet contains no rows: {path}")
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            f"Replay frame range is empty: start_index={start}, end_index={end_index}, "
            f"row_count={len(rows)}."
        )

    actions: list[np.ndarray] = []
    for row_index, row in enumerate(rows[start:stop], start=start):
        if use_full_action:
            actions.append(
                _full_action_from_value(
                    row[requested_action_column],
                    row_index=row_index,
                    column=requested_action_column,
                )
            )
        else:
            actions.append(_action_from_component_columns(row, row_index=row_index))
    return np.stack(actions, axis=0).astype(np.float32, copy=False)


def infer_replay_action_hz(
    *,
    parquet_path: str | Path,
    dataset_root: str | Path | None = None,
    episode_index: int = 0,
    start_index: int = 0,
    end_index: int | None = None,
    resolved_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {} if resolved_metadata is None else dict(resolved_metadata)
    estimate = _estimate_hz_from_dataset_metadata(
        dataset_root=dataset_root,
        episode_index=int(episode_index),
        resolved_metadata=metadata,
    )

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    timing_columns = [column for column in ("control_hz", "timestamp") if column in available_columns]
    if not timing_columns:
        if estimate is not None:
            return estimate
        raise ValueError(
            f"Replay parquet {path} does not contain 'control_hz' or 'timestamp', "
            "and no metadata-derived action hz was available."
        )

    rows = pq.read_table(path, columns=timing_columns).to_pylist()
    if not rows:
        raise ValueError(f"Replay parquet contains no rows: {path}")
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            "Replay frame range is empty for hz inference: "
            f"start_index={start}, end_index={end_index}, row_count={len(rows)}."
        )
    selected_rows = rows[start:stop]
    timestamp_estimate = (
        _estimate_hz_from_timestamps([row.get("timestamp") for row in selected_rows])
        if "timestamp" in timing_columns
        else None
    )
    if timestamp_estimate is not None:
        timestamp_estimate["episode_index"] = int(episode_index)
        timestamp_estimate["frame_range"] = [int(start), int(stop)]
        return timestamp_estimate
    control_hz_estimate = (
        _estimate_hz_from_control_hz([row.get("control_hz") for row in selected_rows])
        if "control_hz" in timing_columns
        else None
    )
    if control_hz_estimate is not None:
        control_hz_estimate["episode_index"] = int(episode_index)
        control_hz_estimate["frame_range"] = [int(start), int(stop)]
        return control_hz_estimate
    if estimate is not None:
        estimate = dict(estimate)
        estimate["frame_range"] = [int(start), int(stop)]
        return estimate
    raise ValueError(f"Replay parquet {path} does not contain enough valid timestamps to infer action hz.")


@dataclass
class DatasetReplayPolicy:
    actions: np.ndarray
    source_path: Path
    chunk_size: int = 32
    loop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    cursor: int = 0
    completed_passes: int = 0
    finished: bool = False

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"DatasetReplayPolicy actions must have shape (N, {ACTION_DIM}), got {actions.shape}."
            )
        if actions.shape[0] <= 0:
            raise ValueError("DatasetReplayPolicy requires at least one action.")
        self.actions = actions
        self.source_path = Path(self.source_path).expanduser().resolve()
        self.chunk_size = max(1, int(self.chunk_size))

    @property
    def frame_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def status_text(self) -> str:
        state = "loop" if self.loop else ("done" if self.finished else "run")
        return (
            f"replay={state}:{min(self.cursor, self.frame_count)}/{self.frame_count}"
            f",passes={self.completed_passes}"
        )

    def reset(self) -> None:
        self.cursor = 0
        self.finished = False

    def skip_startup_frames(self, count: int = 1) -> int:
        skipped = min(max(0, int(count)), max(0, self.frame_count - int(self.cursor)))
        if skipped <= 0:
            return 0
        self.cursor += skipped
        if self.cursor >= self.frame_count:
            self.finished = not self.loop
        return int(skipped)

    def _next_action_chunk(self) -> np.ndarray:
        if self.cursor >= self.frame_count:
            if self.loop:
                self.cursor = 0
                self.completed_passes += 1
                self.finished = False
            else:
                self.finished = True
                return self.actions[-1:].copy()
        end = min(self.frame_count, self.cursor + self.chunk_size)
        chunk = self.actions[self.cursor : end].copy()
        self.cursor = end
        if self.cursor >= self.frame_count:
            self.finished = not self.loop
        return chunk

    def predict(self, observation: DeploymentObservation) -> dict[str, Any]:
        return {
            "action_chunk": self._next_action_chunk(),
            "metadata": {
                "policy": "dataset_replay",
                "source_path": str(self.source_path),
                "cursor": int(self.cursor),
                "frame_count": self.frame_count,
                "loop": bool(self.loop),
                "finished": bool(self.finished),
                "observation_step_index": int(observation.step_index),
            },
        }


def load_dataset_replay_policy(
    *,
    dataset_root: str | Path | None,
    parquet_path: str | Path | None,
    episode_index: int,
    action_column: str = "action",
    start_index: int = 0,
    end_index: int | None = None,
    chunk_size: int = 32,
    loop: bool = False,
) -> LoadedPolicy:
    path, metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=int(episode_index),
    )
    actions = load_replay_actions_from_parquet(
        path,
        action_column=action_column,
        start_index=int(start_index),
        end_index=end_index,
    )
    try:
        action_hz_estimate = infer_replay_action_hz(
            parquet_path=path,
            dataset_root=dataset_root,
            episode_index=int(episode_index),
            start_index=int(start_index),
            end_index=end_index,
            resolved_metadata=metadata,
        )
    except ValueError as exc:
        action_hz_estimate = {
            "hz": 13.0,
            "source": "fallback.no_timing_metadata",
            "warning": str(exc),
        }
    policy = DatasetReplayPolicy(
        actions=actions,
        source_path=path,
        chunk_size=int(chunk_size),
        loop=bool(loop),
        metadata={**dict(metadata), "action_hz_estimate": dict(action_hz_estimate)},
    )
    return LoadedPolicy(
        policy=policy,
        name=f"dataset_replay:{path}",
        metadata={
            "builtin": True,
            "debug_replay": True,
            "source_path": str(path),
            "frame_count": int(actions.shape[0]),
            "action_column": str(action_column or "action"),
            "action_hz": float(action_hz_estimate["hz"]),
            "action_hz_source": str(action_hz_estimate.get("source", "unknown")),
            "action_hz_estimate": dict(action_hz_estimate),
            **dict(metadata),
        },
    )
