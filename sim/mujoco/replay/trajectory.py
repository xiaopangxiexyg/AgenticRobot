from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .dataset import Row, row_timestamp


QposSource = Literal["auto", "actual", "current", "action"]
GripperSource = Literal[
    "auto",
    "action",
    "state",
    "measured",
    "commanded",
    "hand_action",
    "hand_observation",
    "teleop",
]


@dataclass(frozen=True)
class QposSample:
    frame_index: int
    timestamp: float | None
    relative_time_sec: float
    qpos_by_side: dict[str, np.ndarray]
    source_by_side: dict[str, str]


@dataclass(frozen=True)
class GripperReplaySample:
    frame_index: int
    timestamp: float | None
    relative_time_sec: float
    close_by_name: dict[str, float]
    close_by_side: dict[str, float]
    source_by_name: dict[str, str]
    source_by_side: dict[str, str]


def qpos_candidate_columns(side: str, source: QposSource = "auto") -> tuple[str, ...]:
    actual = (
        f"observation.arm.{side}.actual_qpos",
        f"observation.arm.{side}.actual_hardware_qpos",
    )
    current = (
        f"observation.arm.{side}.current_qpos",
        f"observation.{side}_arm.qpos",
    )
    action = (
        f"action.replay.{side}_arm_joint_target",
        f"observation.arm.{side}.last_target_qpos",
    )
    if source == "actual":
        return actual
    if source == "current":
        return current
    if source == "action":
        return action
    return actual + current + action


def qpos_read_columns(active_sides: Sequence[str], source: QposSource = "auto") -> tuple[str, ...]:
    columns = ["frame_index", "index", "timestamp"]
    for side in active_sides:
        columns.extend(qpos_candidate_columns(side, source))
    return tuple(dict.fromkeys(columns))


def gripper_candidate_columns(side: str, source: GripperSource = "auto") -> tuple[str, ...]:
    action_scalar = (f"action.gripper.{side}.target",)
    state = (f"observation.gripper.{side}.state",)
    measured = (f"observation.gripper.{side}.measured",)
    commanded = (
        f"observation.gripper.{side}.commanded",
        f"observation.hand.{side}.commanded_joint_pos",
    )
    hand_action = (
        f"action.replay.{side}_hand_joint_target",
        f"action.{side}_hand.joint_target",
    )
    hand_observation = (
        f"observation.hand.{side}.joint_pos",
        f"observation.hand.{side}.measured_joint_pos",
        f"observation.{side}_hand.qpos",
    )
    teleop = (f"teleop.action_source.{side}_hand_q6",)
    if source == "action":
        return action_scalar + hand_action
    if source == "state":
        return state
    if source == "measured":
        return measured
    if source == "commanded":
        return commanded
    if source == "hand_action":
        return hand_action
    if source == "hand_observation":
        return hand_observation
    if source == "teleop":
        return teleop
    return action_scalar + state + measured + commanded + hand_action + hand_observation


def gripper_read_columns(
    sides: Sequence[str],
    source: GripperSource = "auto",
) -> tuple[str, ...]:
    columns = ["frame_index", "index", "timestamp"]
    for side in sides:
        columns.extend(gripper_candidate_columns(str(side), source))
    return tuple(dict.fromkeys(columns))


def _as_finite_qpos(value: Any, *, length: int = 6) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.shape != (length,):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _as_finite_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.shape != (1,):
        return None
    scalar = float(array[0])
    if not np.isfinite(scalar):
        return None
    return scalar


def _finite_close_from_column(
    value: Any,
    *,
    command_dim: int,
    vector_length: int = 6,
) -> float | None:
    scalar = _as_finite_scalar(value)
    if scalar is not None:
        return float(np.clip(scalar, 0.0, 1.0))
    vector = _as_finite_qpos(value, length=vector_length)
    if vector is None:
        return None
    index = int(command_dim)
    if index < 0 or index >= vector.size:
        return None
    scalar = float(vector[index])
    if not np.isfinite(scalar):
        return None
    return float(np.clip(scalar, 0.0, 1.0))


def qpos_for_side(row: Row, side: str, source: QposSource = "auto") -> tuple[np.ndarray, str]:
    for column in qpos_candidate_columns(side, source):
        qpos = _as_finite_qpos(row.get(column))
        if qpos is not None:
            return qpos, column
    raise KeyError(
        f"Could not find a finite 6-DoF qpos for {side!r} using source {source!r}. "
        f"Tried: {list(qpos_candidate_columns(side, source))}"
    )


def gripper_close_for_side(
    row: Row,
    side: str,
    source: GripperSource = "auto",
    *,
    command_dim: int = 1,
) -> tuple[float, str]:
    for column in gripper_candidate_columns(side, source):
        close = _finite_close_from_column(row.get(column), command_dim=command_dim)
        if close is not None:
            suffix = f"[{int(command_dim)}]" if isinstance(row.get(column), (list, tuple, np.ndarray)) else ""
            return close, f"{column}{suffix}"
    raise KeyError(
        f"Could not find a finite gripper close scalar for {side!r} using source {source!r}. "
        f"Tried: {list(gripper_candidate_columns(side, source))}"
    )


def _row_frame_index(row: Mapping[str, Any], fallback: int) -> int:
    value = row.get("frame_index", row.get("index", fallback))
    try:
        return int(value)
    except Exception:
        return int(fallback)


def extract_qpos_sample(
    row: Row,
    *,
    active_sides: Sequence[str],
    source: QposSource = "auto",
    first_timestamp: float | None = None,
    fallback_hz: float = 30.0,
    row_offset: int = 0,
) -> QposSample:
    timestamp = row_timestamp(row)
    if timestamp is not None and first_timestamp is not None:
        relative_time = max(0.0, float(timestamp) - float(first_timestamp))
    else:
        relative_time = float(row_offset) / max(float(fallback_hz), 1e-6)

    qpos_by_side: dict[str, np.ndarray] = {}
    source_by_side: dict[str, str] = {}
    for side in active_sides:
        qpos, column = qpos_for_side(row, str(side), source)
        qpos_by_side[str(side)] = qpos
        source_by_side[str(side)] = column

    return QposSample(
        frame_index=_row_frame_index(row, row_offset),
        timestamp=timestamp,
        relative_time_sec=relative_time,
        qpos_by_side=qpos_by_side,
        source_by_side=source_by_side,
    )


def extract_gripper_sample(
    row: Row,
    *,
    gripper_side_by_name: Mapping[str, str],
    source: GripperSource = "auto",
    command_dim: int = 1,
    first_timestamp: float | None = None,
    fallback_hz: float = 30.0,
    row_offset: int = 0,
) -> GripperReplaySample:
    timestamp = row_timestamp(row)
    if timestamp is not None and first_timestamp is not None:
        relative_time = max(0.0, float(timestamp) - float(first_timestamp))
    else:
        relative_time = float(row_offset) / max(float(fallback_hz), 1e-6)

    close_by_name: dict[str, float] = {}
    close_by_side: dict[str, float] = {}
    source_by_name: dict[str, str] = {}
    source_by_side: dict[str, str] = {}
    for name, side in gripper_side_by_name.items():
        close, column = gripper_close_for_side(
            row,
            str(side),
            source,
            command_dim=int(command_dim),
        )
        close_by_name[str(name)] = close
        close_by_side[str(side)] = close
        source_by_name[str(name)] = column
        source_by_side[str(side)] = column

    return GripperReplaySample(
        frame_index=_row_frame_index(row, row_offset),
        timestamp=timestamp,
        relative_time_sec=relative_time,
        close_by_name=close_by_name,
        close_by_side=close_by_side,
        source_by_name=source_by_name,
        source_by_side=source_by_side,
    )
