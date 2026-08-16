from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    PolicyIOLayout,
    policy_io_layout_from_metadata,
)
from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError
from dataset_contracts.v23 import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    resolve_v23_dataset_contract,
)
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.pose_adapter import (
    PoseDeploymentError,
    convert_normalized_action_chunk_to_relative,
)
from robot_motion.deploy.v22 import V22DeploymentAdapter
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ARM_JOINT_DOF,
    ARM_POSE_DOF,
    HAND_DOF,
    nan_vector,
)
from robot_motion.replay.action_frame import replay_end_effector_targets_at_index
from robot_motion.replay.trajectory import (
    ReplayTrajectoryGroup,
    replay_action_chunk_round_trip_metrics,
)

from ..collection.schema_v21 import chunk_id_from_episode_index, episode_id_from_index
from .policy import LoadedPolicy


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
    return payload if isinstance(payload, dict) else {}


def _discover_dataset_root_for_parquet(path: Path) -> Path | None:
    """Find an enclosing LeRobot-style root for a directly supplied parquet."""

    resolved = path.expanduser().resolve()
    for parent in resolved.parents:
        if (parent / "meta" / "info.json").is_file():
            return parent
    return None


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
    if parquet_path is not None and str(parquet_path):
        path = Path(parquet_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Replay parquet does not exist: {path}")
        discovered_root = _discover_dataset_root_for_parquet(path)
        metadata: dict[str, Any] = {
            "parquet_path": str(path),
            "source": "direct_parquet",
        }
        if discovered_root is not None:
            metadata["dataset_root"] = str(discovered_root)
            metadata["dataset_info"] = _load_dataset_info(discovered_root)
            metadata["source"] = "direct_parquet_with_dataset_meta"
        return path, metadata

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
        return path, {
            "dataset_root": str(root),
            "episode_index": int(episode_index),
            "episode_id": row.get("episode_id"),
            "frame_count": row.get("frame_count"),
            "duration_sec": row.get("duration_sec"),
            "episode_metadata": row.get("metadata", {}),
            "dataset_info": info,
            "source": "dataset_meta",
        }

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
    for side in ("left", "right"):
        arm = _component_from_row(
            row,
            f"action.{side}_arm.pose",
            expected_dim=ARM_POSE_DOF,
            row_index=row_index,
        )
        if arm is not None:
            action[ACTION_ARM_POSE_SLICES[side]] = arm
        hand = _component_from_row(
            row,
            f"action.{side}_hand.joint_target",
            expected_dim=HAND_DOF,
            row_index=row_index,
        )
        if hand is not None:
            action[ACTION_HAND_SLICES[side]] = hand
    if not np.any(np.isfinite(action)):
        raise ValueError(
            "Replay parquet does not contain an action column or supported "
            "teleop_runtime component action columns."
        )
    return action


def load_replay_actions_from_parquet(
    parquet_path: str | Path,
    *,
    action_column: str = "action",
    start_index: int = 0,
    end_index: int | None = None,
) -> np.ndarray:
    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    component_columns = [
        "action.left_arm.pose",
        "action.right_arm.pose",
        "action.left_hand.joint_target",
        "action.right_hand.joint_target",
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


def v22_replay_rows_to_legacy_actions(
    rows: Sequence[Mapping[str, Any]],
    layout: PolicyIOLayout,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt typed v2.2 replay rows through an explicit legacy-q6 bridge.

    This pure helper is shared by parquet replay loading and tests.  It does
    not issue a command; the returned vectors still pass through the normal
    deployment safety filter before real dispatch.
    """

    adapter = V22DeploymentAdapter(layout)
    actions: list[np.ndarray] = []
    action_metadata: dict[str, Any] | None = None
    for row_index, row in enumerate(rows):
        output = {
            "action": row.get("action"),
            "action.discrete": row.get("action.discrete"),
            "policy_io_layout_hash": row.get("policy_io_layout_hash"),
            "metadata": {"replay_row_index": int(row_index)},
        }
        normalized = adapter.to_legacy_normalized_actions(output, source="dataset_replay_v22")
        if len(normalized) != 1:
            raise ValueError("A single v2.2 replay row must produce exactly one action.")
        actions.append(normalized[0].vector.astype(np.float32, copy=True))
        if action_metadata is None:
            action_metadata = dict(normalized[0].metadata)
    if not actions:
        raise ValueError("v2.2 replay contains no selected rows.")
    return np.stack(actions, axis=0).astype(np.float32, copy=False), (action_metadata or {})


def load_v22_replay_actions_from_parquet(
    parquet_path: str | Path,
    *,
    layout: PolicyIOLayout,
    start_index: int = 0,
    end_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load v2.2 action + int64 token fields from parquet for safe replay."""

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    available_columns = set(pq.read_schema(path).names)
    required = {"action", "policy_io_layout_hash"}
    if layout.action_discrete_dim:
        required.add("action.discrete")
    missing = sorted(required - available_columns)
    if missing:
        raise ValueError(
            f"v2.2 replay parquet {path} is missing required column(s): {', '.join(missing)}."
        )
    rows = pq.read_table(path, columns=sorted(required)).to_pylist()
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            f"Replay frame range is empty: start_index={start}, end_index={end_index}, "
            f"row_count={len(rows)}."
        )
    return v22_replay_rows_to_legacy_actions(rows[start:stop], layout)


def _parquet_declares_v22_policy_io(path: Path) -> bool:
    """Return whether row-level v2.2 identity columns are present."""

    names = set(_require_pyarrow_parquet().read_schema(path).names)
    return "policy_io_layout_hash" in names and "action" in names


def load_contract_replay_actions_from_parquet(
    parquet_path: str | Path,
    *,
    resolved_metadata: Mapping[str, Any],
    action_column: str = "action",
    start_index: int = 0,
    end_index: int | None = None,
    policy_io_layout: PolicyIOLayout | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], PoseFrameContract | None, bool]:
    """Load replay actions using the dataset's serialized action contract.

    Legacy datasets store canonical 26D actions directly.  v2.2/v2.3 datasets
    instead store a layout-bound compact action (16D for dual-wrist DMTAC), so
    every replay entrypoint must bridge it before constructing a canonical
    ``ReplayTrajectoryGroup``.
    """

    path = Path(parquet_path).expanduser().resolve()
    dataset_info = resolved_metadata.get("dataset_info")
    dataset_info = dataset_info if isinstance(dataset_info, Mapping) else {}
    explicit_layout = policy_io_layout
    if isinstance(explicit_layout, Mapping):
        explicit_layout = PolicyIOLayout.from_dict(explicit_layout)
    elif explicit_layout is not None and not isinstance(explicit_layout, PolicyIOLayout):
        raise TypeError("policy_io_layout must be a PolicyIOLayout, mapping, or None.")

    dataset_schema = str(dataset_info.get("dexteleop_dataset_schema") or "")
    metadata_is_v22 = dataset_schema == DEXTELEOP_DATASET_SCHEMA_V22
    metadata_is_v23 = dataset_schema == DEXTELEOP_DATASET_SCHEMA_V23
    row_contract_is_v22 = path.is_file() and _parquet_declares_v22_policy_io(path)
    is_typed = metadata_is_v22 or metadata_is_v23 or row_contract_is_v22
    if not is_typed:
        if explicit_layout is not None:
            raise ValueError(
                "A v2.2 replay layout was supplied for a non-v2.2 dataset; convert the dataset first."
            )
        actions = load_replay_actions_from_parquet(
            path,
            action_column=action_column,
            start_index=int(start_index),
            end_index=end_index,
        )
        return actions, {}, None, False

    resolved_pose_contract: PoseFrameContract | None = None
    if metadata_is_v23:
        v23_contract = resolve_v23_dataset_contract(dataset_info)
        layout = v23_contract.policy_io_layout
        resolved_pose_contract = v23_contract.pose_frames
        if explicit_layout is not None and explicit_layout.layout_hash != layout.layout_hash:
            raise ValueError("Explicit replay policy_io_layout does not match dataset metadata.")
    elif metadata_is_v22:
        layout = policy_io_layout_from_metadata(dataset_info, allow_legacy_v21=False)
        if explicit_layout is not None and explicit_layout.layout_hash != layout.layout_hash:
            raise ValueError("Explicit replay policy_io_layout does not match dataset metadata.")
    elif explicit_layout is not None:
        layout = explicit_layout
    else:
        raise ValueError(
            "Direct parquet declares v2.2 policy I/O but its layout cannot be reconstructed. "
            "Pass the enclosing dataset root or an explicit policy_io_layout."
        )

    actions, action_metadata = load_v22_replay_actions_from_parquet(
        path,
        layout=layout,
        start_index=int(start_index),
        end_index=end_index,
    )
    if resolved_pose_contract is not None:
        for side in resolved_pose_contract.source.active_sides:
            pose_slice = ACTION_ARM_POSE_SLICES[side]
            try:
                actions[:, pose_slice] = np.stack(
                    [
                        resolved_pose_contract.storage_pose_vector_to_canonical_desk(
                            row[pose_slice], side=side
                        )
                        for row in actions
                    ],
                    axis=0,
                )
            except (PoseFrameContractError, ValueError) as exc:
                raise ValueError(
                    f"v2.3 replay action for {side!r} cannot be converted to canonical desk: {exc}"
                ) from exc
        action_metadata.update(
            {
                "pose_frame_contract": resolved_pose_contract.to_dict(),
                "pose_storage_frame": resolved_pose_contract.pose_storage_frame,
                "pose_source_id": resolved_pose_contract.source_id,
                "pose_action_absolute_frame": "canonical_desk",
            }
        )
    return actions, action_metadata, resolved_pose_contract, True


def load_replay_frame_times_from_parquet(
    parquet_path: str | Path,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    fallback_hz: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Load strictly increasing relative source times, with an explicit fallback.

    Position replay should preserve real dataset jitter when a complete valid
    timestamp column exists.  Missing, non-finite, duplicate, or backward
    timestamps fall back for the whole selected range to the nominal action
    grid; mixing real and synthetic intervals would create a harder-to-audit
    trajectory.
    """

    rate = float(fallback_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("fallback_hz must be finite and positive")
    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    timestamp_column = next(
        (
            column
            for column in ("timestamp", "observation.timestamp")
            if column in available_columns
        ),
        None,
    )

    if timestamp_column is None:
        row_count = int(pq.read_metadata(path).num_rows)
        selected_values: list[Any] = []
        fallback_reason = "timestamp_column_missing"
    else:
        rows = pq.read_table(path, columns=[timestamp_column]).to_pylist()
        row_count = len(rows)
        selected_values = [row.get(timestamp_column) for row in rows]
        fallback_reason = "timestamp_values_invalid"

    start = max(0, int(start_index))
    stop = row_count if end_index is None else min(row_count, int(end_index))
    if stop <= start:
        raise ValueError(
            "Replay frame range is empty for timestamp loading: "
            f"start_index={start}, end_index={end_index}, row_count={row_count}."
        )
    count = stop - start
    if timestamp_column is not None:
        raw = np.asarray(
            [
                np.nan if _float_or_none(value) is None else float(value)
                for value in selected_values[start:stop]
            ],
            dtype=float,
        )
        if np.all(np.isfinite(raw)) and (
            raw.size <= 1 or np.all(np.diff(raw) > 0.0)
        ):
            relative = raw - raw[0]
            return (
                tuple(float(value) for value in relative),
                {
                    "source_timestamp_mode": "dataset",
                    "source_timestamp_column": str(timestamp_column),
                    "source_timestamp_start": float(raw[0]),
                    "source_timestamp_duration_sec": float(relative[-1]),
                    "frame_range": [int(start), int(stop)],
                },
            )

    fallback = tuple(float(index) / rate for index in range(count))
    return (
        fallback,
        {
            "source_timestamp_mode": "nominal_fallback",
            "source_timestamp_column": timestamp_column,
            "source_timestamp_fallback_reason": fallback_reason,
            "source_timestamp_fallback_hz": rate,
            "frame_range": [int(start), int(stop)],
        },
    )


def load_replay_arm_joint_targets_from_parquet(
    parquet_path: str | Path,
    *,
    active_sides: Sequence[str],
    start_index: int = 0,
    end_index: int | None = None,
    required: bool = True,
) -> dict[str, np.ndarray]:
    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    requested_columns = [
        f"action.replay.{side}_arm_joint_target"
        for side in ("left", "right")
        if side in set(active_sides)
    ]
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    missing = [column for column in requested_columns if column not in available_columns]
    if missing:
        if not required:
            columns = [column for column in requested_columns if column in available_columns]
            if not columns:
                return {}
        else:
            raise ValueError(
                f"Replay parquet {path} is missing required arm joint target column(s): {missing}"
            )
    else:
        columns = requested_columns
    if not columns:
        if required:
            raise ValueError(
                f"Replay parquet {path} has no active-side arm joint target columns."
            )
        return {}

    loaded_sides = [
        side
        for side in active_sides
        if f"action.replay.{side}_arm_joint_target" in columns
    ]
    if required and set(loaded_sides) != set(active_sides):
        raise ValueError(
            f"Replay parquet {path} is missing required arm joint targets for "
            f"{sorted(set(active_sides) - set(loaded_sides))}."
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
    targets_by_side: dict[str, list[np.ndarray]] = {side: [] for side in loaded_sides}
    invalid_sides: set[str] = set()
    for row_index, row in enumerate(rows[start:stop], start=start):
        for side in loaded_sides:
            if side in invalid_sides:
                continue
            key = f"action.replay.{side}_arm_joint_target"
            value = row.get(key)
            if value is None:
                message = (
                    f"Replay row {row_index} column {key!r} is missing; cannot use qpos init."
                )
                if required:
                    raise ValueError(message)
                invalid_sides.add(side)
                continue
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            if vector.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(vector)):
                message = (
                    f"Replay row {row_index} column {key!r} must be finite shape ({ARM_JOINT_DOF},), "
                    f"got {vector.shape}."
                )
                if required:
                    raise ValueError(message)
                invalid_sides.add(side)
                continue
            targets_by_side[side].append(vector.astype(np.float32, copy=True))
    for side in invalid_sides:
        targets_by_side.pop(side, None)
    return {
        side: np.stack(values, axis=0).astype(np.float32, copy=False)
        for side, values in targets_by_side.items()
        if values
    }


def load_replay_hand_joint_targets_from_parquet(
    parquet_path: str | Path,
    *,
    active_sides: Sequence[str],
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Load explicit, fully resolved hand targets recorded for replay.

    Formal collection intentionally stores ``action.replay.*_hand_joint_target``
    from the backend's last accepted command.  It remains valid across a
    temporary tracker hold where the high-level 26D action may omit the hand
    effector and therefore contain NaN hand slots.
    """

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    requested_columns = [
        f"action.replay.{side}_hand_joint_target"
        for side in ("left", "right")
        if side in set(active_sides)
    ]
    available_columns = set(pq.read_schema(path).names)
    columns = [column for column in requested_columns if column in available_columns]
    if not columns:
        return {}

    loaded_sides = [
        side
        for side in active_sides
        if f"action.replay.{side}_hand_joint_target" in columns
    ]
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

    targets_by_side: dict[str, list[np.ndarray]] = {side: [] for side in loaded_sides}
    invalid_sides: set[str] = set()
    for row in rows[start:stop]:
        for side in loaded_sides:
            if side in invalid_sides:
                continue
            key = f"action.replay.{side}_hand_joint_target"
            try:
                vector = np.asarray(row.get(key), dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                invalid_sides.add(side)
                continue
            if vector.shape != (HAND_DOF,) or not np.all(np.isfinite(vector)):
                invalid_sides.add(side)
                continue
            targets_by_side[side].append(vector.astype(np.float32, copy=True))
    for side in invalid_sides:
        targets_by_side.pop(side, None)
    return {
        side: np.stack(values, axis=0).astype(np.float32, copy=False)
        for side, values in targets_by_side.items()
        if len(values) == stop - start
    }


def load_replay_arm_observation_qpos_from_parquet(
    parquet_path: str | Path,
    *,
    active_sides: Sequence[str],
    field_name: str,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Load observation arm qpos columns for replay start initialization.

    These columns are state observations, not trajectory commands. Missing or
    malformed rows are represented as NaN vectors so the caller can choose a
    valid initialization frame without treating the whole episode as a qpos
    target replay.
    """

    field = str(field_name or "").strip()
    if not field:
        raise ValueError("field_name must be non-empty.")
    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    requested_columns = [
        f"observation.arm.{side}.{field}"
        for side in ("left", "right")
        if side in set(active_sides)
    ]
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    columns = [column for column in requested_columns if column in available_columns]
    if not columns:
        return {}

    loaded_sides = [
        side
        for side in active_sides
        if f"observation.arm.{side}.{field}" in columns
    ]
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

    invalid = nan_vector(ARM_JOINT_DOF)
    qpos_by_side: dict[str, list[np.ndarray]] = {side: [] for side in loaded_sides}
    for row in rows[start:stop]:
        for side in loaded_sides:
            key = f"observation.arm.{side}.{field}"
            value = row.get(key)
            try:
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                vector = invalid
            if vector.shape != (ARM_JOINT_DOF,):
                vector = invalid
            qpos_by_side[side].append(vector.astype(np.float32, copy=True))
    return {
        side: np.stack(values, axis=0).astype(np.float32, copy=False)
        for side, values in qpos_by_side.items()
        if values
    }


def load_replay_arm_qpos_audit_from_parquet(
    parquet_path: str | Path,
    *,
    active_sides: Sequence[str],
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Load the v2.2 independent 12D arm-qpos audit field by logical side."""

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    available = set(pq.read_schema(path).names)
    qpos_key = "observation.arm.qpos"
    if qpos_key not in available:
        return {}
    columns = [qpos_key]
    valid_key = "observation.arm.qpos_valid"
    if valid_key in available:
        columns.append(valid_key)
    rows = pq.read_table(path, columns=columns).to_pylist()
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            f"Replay qpos audit frame range is empty: start_index={start}, "
            f"end_index={end_index}, row_count={len(rows)}."
        )
    active = tuple(side for side in ("left", "right") if side in set(active_sides))
    output: dict[str, list[np.ndarray]] = {side: [] for side in active}
    side_slices = {"left": slice(0, ARM_JOINT_DOF), "right": slice(ARM_JOINT_DOF, 12)}
    invalid = nan_vector(ARM_JOINT_DOF)
    for row in rows[start:stop]:
        try:
            qpos = np.asarray(row.get(qpos_key), dtype=np.float32).reshape(-1)
        except Exception:
            qpos = np.empty((0,), dtype=np.float32)
        validity = row.get(valid_key)
        try:
            valid = None if validity is None else np.asarray(validity, dtype=bool).reshape(-1)
        except Exception:
            valid = np.empty((0,), dtype=bool)
        for side in active:
            target_slice = side_slices[side]
            value = qpos[target_slice] if qpos.shape == (12,) else invalid
            if valid is not None and (valid.shape != (12,) or not np.all(valid[target_slice])):
                value = invalid
            if value.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(value)):
                value = invalid
            output[side].append(np.asarray(value, dtype=np.float32).copy())
    return {
        side: np.stack(values, axis=0).astype(np.float32, copy=False)
        for side, values in output.items()
    }


def load_replay_arm_actual_tcp_pose_from_parquet(
    parquet_path: str | Path,
    *,
    active_sides: Sequence[str],
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Load actual TCP anchor history for v2.2/v2.3 pose replay checks."""

    path = Path(parquet_path).expanduser().resolve()
    pq = _require_pyarrow_parquet()
    requested = {
        side: f"observation.arm.{side}.actual_tcp_pose"
        for side in ("left", "right")
        if side in {str(value).strip().lower() for value in active_sides}
    }
    available = set(pq.read_schema(path).names)
    columns = [column for column in requested.values() if column in available]
    if not columns:
        return {}
    rows = pq.read_table(path, columns=columns).to_pylist()
    if not rows:
        raise ValueError(f"Replay parquet contains no rows: {path}")
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            f"Replay actual TCP frame range is empty: start_index={start}, end_index={end_index}, "
            f"row_count={len(rows)}."
        )
    invalid = np.full((7,), np.nan, dtype=np.float32)
    output: dict[str, list[np.ndarray]] = {
        side: [] for side, column in requested.items() if column in columns
    }
    for row in rows[start:stop]:
        for side, column in requested.items():
            if column not in columns:
                continue
            try:
                value = np.asarray(row.get(column), dtype=np.float32).reshape(-1)
            except Exception:
                value = invalid
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                value = invalid
            output[side].append(np.asarray(value, dtype=np.float32).copy())
    return {
        side: np.stack(values, axis=0).astype(np.float32, copy=False)
        for side, values in output.items()
    }


def compare_v22_replay_absolute_relative_chunks(
    parquet_path: str | Path,
    *,
    layout: PolicyIOLayout,
    active_sides: Sequence[str] = ("left", "right"),
    start_index: int = 0,
    end_index: int | None = None,
    chunk_size: int = 32,
    pose_frame_contract: PoseFrameContract | None = None,
) -> dict[str, Any]:
    """Compare absolute and relative round-trip replay poses from one parquet.

    This is a hardware-free diagnostic.  It loads the v2.2 model action and
    actual-TCP history, uses each chunk's first actual TCP as its fixed anchor,
    and returns per-chunk SE(3) reconstruction metrics.  It never rewrites the
    source dataset and does not change deployment's default absolute mode.
    """

    actions, action_metadata = load_v22_replay_actions_from_parquet(
        parquet_path,
        layout=layout,
        start_index=int(start_index),
        end_index=end_index,
    )
    actual_tcp = load_replay_arm_actual_tcp_pose_from_parquet(
        parquet_path,
        active_sides=active_sides,
        start_index=int(start_index),
        end_index=end_index,
    )
    missing = [side for side in active_sides if str(side).strip().lower() not in actual_tcp]
    if missing:
        raise ValueError(
            "v2.2 relative replay comparison requires actual TCP columns for side(s): "
            f"{sorted(set(str(side).strip().lower() for side in missing))}."
        )
    frame_contract = (
        PoseFrameContract.legacy_v22_raw(active_sides=active_sides)
        if pose_frame_contract is None
        else pose_frame_contract
    )
    # Replay actions and measured anchors must enter the relative converter in
    # one canonical desk frame.  The default legacy v2.2 contract is raw with
    # identity transforms, so this remains a no-op for existing datasets.
    for side in active_sides:
        normalized_side = str(side).strip().lower()
        pose_slice = ACTION_ARM_POSE_SLICES[normalized_side]
        try:
            actions[:, pose_slice] = np.stack(
                [
                    frame_contract.storage_pose_vector_to_desk(
                        row[pose_slice], side=normalized_side
                    )
                    for row in actions
                ],
                axis=0,
            )
            actual_tcp[normalized_side] = np.stack(
                [
                    frame_contract.storage_pose_vector_to_desk(
                        row, side=normalized_side
                    )
                    for row in actual_tcp[normalized_side]
                ],
                axis=0,
            )
        except (PoseFrameContractError, ValueError) as exc:
            raise ValueError(
                f"Replay pose frame conversion failed for side {normalized_side!r}: {exc}"
            ) from exc
    group = ReplayTrajectoryGroup(
        key="v22_pose_comparison",
        label="v22_pose_comparison",
        source_path=str(Path(parquet_path).expanduser().resolve()),
        source_episode_index=0,
        start_index=int(start_index),
        end_index=int(start_index) + int(actions.shape[0]),
        frame_count=int(actions.shape[0]),
        action_hz=0.0,
        actions=actions,
        arm_observation_actual_tcp_pose=actual_tcp,
        metadata={"action_metadata": dict(action_metadata)},
    )
    reports: list[dict[str, Any]] = []
    for chunk_start in range(0, int(actions.shape[0]), max(1, int(chunk_size))):
        reports.append(
            replay_action_chunk_round_trip_metrics(
                group,
                active_sides,
                start_index=chunk_start,
                chunk_size=int(chunk_size),
            )
        )
    return {
        "source_path": str(Path(parquet_path).expanduser().resolve()),
        "frame_count": int(actions.shape[0]),
        "chunk_size": max(1, int(chunk_size)),
        "active_sides": tuple(str(side).strip().lower() for side in active_sides),
        "pose_frame_contract": frame_contract.to_dict(),
        "reports": reports,
        "action_metadata": dict(action_metadata),
    }


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
    try:
        pq = _require_pyarrow_parquet()
    except ModuleNotFoundError:
        if estimate is not None:
            estimate = dict(estimate)
            estimate["timestamp_scan_skipped"] = "pyarrow_unavailable"
            return estimate
        raise
    schema = pq.read_schema(path)
    available_columns = set(schema.names)
    timing_columns = [
        column
        for column in ("control_hz", "timestamp", "observation.timestamp")
        if column in available_columns
    ]
    if not timing_columns:
        if estimate is not None:
            return estimate
        raise ValueError(
            f"Replay parquet {path} does not contain 'control_hz', 'timestamp', or "
            "'observation.timestamp', "
            "and no metadata-derived action hz was available."
        )

    rows = pq.read_table(path, columns=timing_columns).to_pylist()
    if not rows:
        raise ValueError(f"Replay parquet contains no rows: {path}")
    start = max(0, int(start_index))
    stop = len(rows) if end_index is None else min(len(rows), int(end_index))
    if stop <= start:
        raise ValueError(
            f"Replay frame range is empty for hz inference: start_index={start}, end_index={end_index}, "
            f"row_count={len(rows)}."
        )
    selected_rows = rows[start:stop]
    timestamp_column = next(
        (
            column
            for column in ("timestamp", "observation.timestamp")
            if column in timing_columns
        ),
        None,
    )
    timestamp_estimate = (
        _estimate_hz_from_timestamps(
            [row.get(timestamp_column) for row in selected_rows]
        )
        if timestamp_column is not None
        else None
    )
    if timestamp_estimate is not None:
        timestamp_estimate["timestamp_column"] = str(timestamp_column)
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
    raise ValueError(
        f"Replay parquet {path} does not contain enough valid timestamps to infer action hz."
    )


@dataclass(frozen=True)
class _DatasetReplayPredictionCheckpoint:
    cursor: int
    completed_passes: int
    finished: bool


@dataclass
class DatasetReplayPolicy:
    actions: np.ndarray
    source_path: Path
    chunk_size: int = 32
    loop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    action_metadata: dict[str, Any] = field(default_factory=dict)
    output_pose_mode: str = "absolute"
    active_sides: tuple[str, ...] = ("left", "right")
    source_start_index: int = 0
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
        self.output_pose_mode = str(self.output_pose_mode).strip().lower()
        if self.output_pose_mode not in {"absolute", "relative"}:
            raise ValueError(
                "DatasetReplayPolicy output_pose_mode must be 'absolute' or 'relative'."
            )
        self.active_sides = tuple(
            dict.fromkeys(str(side).strip().lower() for side in self.active_sides)
        )
        self.source_start_index = max(0, int(self.source_start_index))

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

    def begin_prediction_transaction(self) -> _DatasetReplayPredictionCheckpoint:
        """Checkpoint every field mutated while selecting the next chunk."""

        return _DatasetReplayPredictionCheckpoint(
            cursor=int(self.cursor),
            completed_passes=int(self.completed_passes),
            finished=bool(self.finished),
        )

    def commit_prediction_transaction(
        self,
        checkpoint: _DatasetReplayPredictionCheckpoint,
    ) -> None:
        if not isinstance(checkpoint, _DatasetReplayPredictionCheckpoint):
            raise TypeError("Invalid dataset replay prediction checkpoint.")

    def rollback_prediction_transaction(
        self,
        checkpoint: _DatasetReplayPredictionCheckpoint,
    ) -> None:
        if not isinstance(checkpoint, _DatasetReplayPredictionCheckpoint):
            raise TypeError("Invalid dataset replay prediction checkpoint.")
        self.cursor = int(checkpoint.cursor)
        self.completed_passes = int(checkpoint.completed_passes)
        self.finished = bool(checkpoint.finished)

    def _next_action_chunk(self) -> tuple[np.ndarray, tuple[int, ...]]:
        if self.cursor >= self.frame_count:
            if self.loop:
                self.cursor = 0
                self.completed_passes += 1
                self.finished = False
            else:
                self.finished = True
                local_index = self.frame_count - 1
                return self.actions[-1:].copy(), (local_index,)
        start = int(self.cursor)
        end = min(self.frame_count, self.cursor + self.chunk_size)
        chunk = self.actions[self.cursor : end].copy()
        self.cursor = end
        if self.cursor >= self.frame_count:
            self.finished = not self.loop
        return chunk, tuple(range(start, end))

    def predict(self, observation: DeploymentObservation) -> dict[str, Any]:
        chunk, local_indices = self._next_action_chunk()
        actions = [
            NormalizedAction(
                vector=action,
                metadata={
                    **dict(self.action_metadata),
                    "debug_replay_local_action_index": int(local_index),
                    "debug_replay_source_action_index": int(
                        self.source_start_index + local_index
                    ),
                },
            )
            for action, local_index in zip(chunk, local_indices, strict=True)
        ]
        if self.output_pose_mode == "relative":
            try:
                actions = convert_normalized_action_chunk_to_relative(
                    actions,
                    observation,
                    active_sides=self.active_sides,
                )
            except PoseDeploymentError as exc:
                raise RuntimeError(
                    f"Dataset replay relative output requires measured actual TCP anchors: {exc}"
                ) from exc
        return {
            "action_chunk": actions,
            "metadata": {
                "policy": "dataset_replay",
                "source_path": str(self.source_path),
                "cursor": int(self.cursor),
                "frame_count": self.frame_count,
                "loop": bool(self.loop),
                "finished": bool(self.finished),
                "observation_step_index": int(observation.step_index),
                "output_pose_mode": self.output_pose_mode,
            },
        }


def debug_replay_execution_exhausted(
    policy: Any,
    *,
    queued_actions: int,
) -> bool:
    """Stop non-loop replay after its decoded final chunk is consumed."""

    wrapped_policy = getattr(policy, "policy", policy)
    if not isinstance(wrapped_policy, DatasetReplayPolicy):
        return False
    return (
        not bool(wrapped_policy.loop)
        and bool(wrapped_policy.finished)
        and int(queued_actions) <= 0
    )


def skip_startup_replay_frames(policy: Any, count: int = 1) -> int:
    """Advance a replay source when it supports startup-frame skipping."""

    wrapped_policy = getattr(policy, "policy", policy)
    skip = getattr(wrapped_policy, "skip_startup_frames", None)
    if not callable(skip):
        return 0
    return int(skip(count))


def _restrict_replay_end_effector_metadata(
    metadata: Mapping[str, Any],
    end_effector_sides: Sequence[str],
) -> dict[str, Any]:
    """Bind replay action metadata to the runtime end-effector selection.

    v2.2 rows carry the end-effector sides declared by the dataset layout.  A
    deployment may intentionally run the same arm-only dataset with
    ``--end-effector-sides none`` (for example when the legacy hand hardware is
    absent).  In that case the dataset declaration must not be allowed to
    manufacture a hand request at the common safety boundary.
    """

    requested = tuple(dict.fromkeys(str(side).strip().lower() for side in end_effector_sides))
    invalid = set(requested) - set(ACTION_HAND_SLICES)
    if invalid:
        raise ValueError(f"Unsupported replay end-effector side(s): {sorted(invalid)}")
    restricted = dict(metadata)
    restricted["end_effector_enabled_sides"] = list(requested)
    for key in ("hand_action_specs_by_side", "hand_active_mask_by_side"):
        value = restricted.get(key)
        if isinstance(value, Mapping):
            restricted[key] = {
                side: item
                for side, item in value.items()
                if str(side).strip().lower() in set(requested)
            }
        elif not requested:
            restricted[key] = {}
    return restricted


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
    active_sides: Sequence[str] = ("left", "right"),
    end_effector_sides: Sequence[str] | None = None,
    output_pose_mode: str = "absolute",
    policy_io_layout: PolicyIOLayout | Mapping[str, Any] | None = None,
) -> LoadedPolicy:
    path, metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=int(episode_index),
    )
    actions, v22_action_metadata, resolved_pose_contract, is_typed = (
        load_contract_replay_actions_from_parquet(
            path,
            resolved_metadata=metadata,
            action_column=action_column,
            start_index=int(start_index),
            end_index=end_index,
            policy_io_layout=policy_io_layout,
        )
    )
    dataset_info = metadata.get("dataset_info")
    dataset_info = dataset_info if isinstance(dataset_info, Mapping) else {}
    metadata_is_v23 = (
        str(dataset_info.get("dexteleop_dataset_schema") or "")
        == DEXTELEOP_DATASET_SCHEMA_V23
    )
    policy_io_layout_hash = (
        dataset_info.get("policy_io_layout_hash")
        or v22_action_metadata.get("policy_io_layout_hash")
    )
    if end_effector_sides is not None:
        v22_action_metadata = _restrict_replay_end_effector_metadata(
            v22_action_metadata,
            end_effector_sides,
        )
    active = tuple(dict.fromkeys(str(side).strip().lower() for side in active_sides))
    unknown_sides = [side for side in active if side not in ACTION_HAND_SLICES]
    if unknown_sides:
        raise ValueError(f"Unsupported dataset replay side(s): {unknown_sides}.")
    hand_joint_targets = (
        {}
        if is_typed
        else load_replay_hand_joint_targets_from_parquet(
            path,
            active_sides=active,
            start_index=int(start_index),
            end_index=end_index,
        )
    )
    if active and hand_joint_targets:
        replay_group = ReplayTrajectoryGroup(
            key="deploy_debug_replay",
            label="deploy_debug_replay",
            source_path=str(path),
            source_episode_index=int(metadata.get("episode_index", int(episode_index))),
            start_index=int(start_index),
            end_index=int(start_index) + int(actions.shape[0]),
            frame_count=int(actions.shape[0]),
            action_hz=0.0,
            actions=actions,
            hand_joint_targets=hand_joint_targets,
            metadata={"source_metadata": dict(metadata)},
        )
        normalized_actions = np.asarray(actions, dtype=np.float32).copy()
        for index in range(int(normalized_actions.shape[0])):
            resolved_hands = replay_end_effector_targets_at_index(
                replay_group,
                active,
                index=index,
            )
            for side, target in resolved_hands.items():
                normalized_actions[index, ACTION_HAND_SLICES[side]] = target
        actions = normalized_actions
    action_hz_estimate = infer_replay_action_hz(
        parquet_path=path,
        dataset_root=dataset_root,
        episode_index=int(episode_index),
        start_index=int(start_index),
        end_index=end_index,
        resolved_metadata=metadata,
    )
    policy = DatasetReplayPolicy(
        actions=actions,
        source_path=path,
        chunk_size=int(chunk_size),
        loop=bool(loop),
        metadata={**dict(metadata), "action_hz_estimate": dict(action_hz_estimate)},
        action_metadata=v22_action_metadata,
        output_pose_mode=str(output_pose_mode),
        active_sides=active,
        source_start_index=int(start_index),
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
            "policy_io_layout_hash": None if not is_typed else policy_io_layout_hash,
            "explicit_hand_replay_sides": sorted(hand_joint_targets),
            "action_hz": float(action_hz_estimate["hz"]),
            "action_hz_source": str(action_hz_estimate.get("source", "unknown")),
            "action_hz_estimate": dict(action_hz_estimate),
            "output_pose_mode": str(output_pose_mode).strip().lower(),
            "policy_absolute_frame": (
                "canonical_desk" if metadata_is_v23 else "runtime_raw"
            ),
            **dict(metadata),
        },
    )
