"""Canonical Parquet physical types and row IO.

The column sets are deliberately public: both online writers and offline
migrations must use the same physical Arrow contract without importing one
another.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FLOAT32_LIST_COLUMNS = frozenset(
    {
        "action",
        "action.left_arm.pose",
        "action.right_arm.pose",
        "action.left_hand.joint_target",
        "action.right_hand.joint_target",
        "observation.state",
        "observation.last_action",
        "observation.left_arm.qpos",
        "observation.left_arm.pose",
        "observation.left_hand.qpos",
        "observation.right_arm.qpos",
        "observation.right_arm.pose",
        "observation.right_hand.qpos",
    }
    | {
        template.format(side=side)
        for side in ("left", "right")
        for template in (
            "action.reference.{side}_arm.q_ref",
            "action.reference.{side}_arm.q_ik_raw",
            "action.reference.{side}_arm.pose_ref",
            "observation.arm.{side}.current_qpos",
            "observation.arm.{side}.current_tcp_pose",
            "observation.arm.{side}.commanded_qpos",
            "observation.arm.{side}.commanded_tcp_pose",
            "observation.arm.{side}.last_target_qpos",
            "position.transport.{side}.last_attempted_qpos",
            "position.transport.{side}.last_accepted_qpos",
            "position.transport.{side}.measured_qpos",
            "observation.arm.{side}.actual_qpos",
            "observation.arm.{side}.actual_tcp_pose",
            "observation.arm.{side}.actual_hardware_qpos",
            "observation.arm.{side}.actual_hardware_pose_6d",
            "observation.wrist_camera.{side}.pose_base",
            "action.replay.{side}_arm_joint_target",
            "teleop.action_source.{side}_wrist_pose_operator",
            "teleop.action_source.{side}_wrist_pose_desk",
            "teleop.action_source.{side}_wrist_pose_base",
            "observation.hand.{side}.joint_pos",
            "observation.hand.{side}.measured_joint_pos",
            "observation.hand.{side}.commanded_joint_pos",
            "observation.hand.{side}.tactile",
            "action.replay.{side}_hand_joint_target",
            "teleop.action_source.{side}_hand_q6",
        )
    }
)

# v2.2 canonical policy-I/O component fields.  Their exact fixed lengths are
# supplied by PolicyIOLayout at write time; this set keeps standalone row
# conversions typed correctly as float32 even without a fixed-list contract.
V22_FLOAT32_LIST_COLUMNS = frozenset(
    {
        "action.arm.pose",
        "observation.arm.pose",
        "observation.arm.qpos",
        "observation.last_action",
    }
    | {
        f"{prefix}.end_effector.{side}.{field_name}"
        for prefix, field_name in (("action", "target"), ("observation", "state"))
        for side in ("left", "right")
    }
)

STRING_LIST_COLUMNS = frozenset(
    {
        "active_sides",
        "camera_keys",
        "fault_codes",
        "teleop.arm_mode.left.tags",
        "teleop.arm_mode.right.tags",
    }
    | {
        template.format(side=side)
        for side in ("left", "right")
        for template in (
            "observation.hand.{side}.tactile_fields",
            "observation.hand.{side}.tactile_regions",
            "observation.hand.{side}.tactile_sensor_keys",
        )
    }
)

INT64_LIST_COLUMNS = frozenset(
    {
        template.format(side=side)
        for side in ("left", "right")
        for template in (
            "observation.hand.{side}.tactile_process_cpu_affinity",
            "observation.hand.{side}.tactile_raw_shape_per_sensor",
        )
    }
)

V22_INT64_LIST_COLUMNS = frozenset(
    {
        "action.discrete",
        "observation.state.discrete",
        "observation.last_action.discrete",
    }
    | {
        f"{prefix}.end_effector.{side}.{field_name}_token"
        for prefix, field_name in (("action", "target"), ("observation", "state"))
        for side in ("left", "right")
    }
)

BOOL_LIST_COLUMNS = frozenset({"observation.arm.qpos_valid"})

ARM_MODE_INT64_COLUMNS = frozenset(
    {
        "teleop.arm_mode.available",
        "teleop.arm_mode.left.id",
        "teleop.arm_mode.left.is_special",
        "teleop.arm_mode.left.init_phase_valid",
        "teleop.arm_mode.right.id",
        "teleop.arm_mode.right.is_special",
        "teleop.arm_mode.right.init_phase_valid",
    }
)

ARM_MODE_STRING_COLUMNS = frozenset(
    {
        "teleop.arm_mode.schema",
        "teleop.arm_mode.left.label",
        "teleop.arm_mode.left.source",
        "teleop.arm_mode.left.scope",
        "teleop.arm_mode.left.init_phase",
        "teleop.arm_mode.right.label",
        "teleop.arm_mode.right.source",
        "teleop.arm_mode.right.scope",
        "teleop.arm_mode.right.init_phase",
        "teleop.arm_mode_json",
    }
)

ARM_MODE_FLOAT64_COLUMNS = frozenset(
    {
        template.format(side=side)
        for side in ("left", "right")
        for template in (
            "teleop.arm_mode.{side}.init_target_position_error_m",
            "teleop.arm_mode.{side}.init_target_rotation_error_deg",
            "teleop.arm_mode.{side}.init_translation_speed_m_s",
            "teleop.arm_mode.{side}.init_rotation_speed_deg_s",
            "teleop.arm_mode.{side}.init_dwell_sec",
        )
    }
)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _to_serializable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def write_rows_to_parquet(
    rows: list[Mapping[str, Any]],
    path: str | Path,
    *,
    compression: str = "zstd",
    fixed_size_list_columns: Mapping[str, tuple[str, int]] | None = None,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LeRobot parquet export requires pyarrow to be installed."
        ) from exc

    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [_to_serializable(dict(row)) for row in rows]
    column_names = list(dict.fromkeys(name for row in normalized_rows for name in row))
    fixed_contracts = {} if fixed_size_list_columns is None else dict(fixed_size_list_columns)

    def fixed_list_type(dtype: str, length: int):
        normalized_dtype = str(dtype).strip().lower()
        normalized_length = int(length)
        if normalized_length < 0:
            raise ValueError("fixed-size parquet list lengths must be non-negative.")
        scalar = {
            "float32": pa.float32(),
            "int64": pa.int64(),
            "bool": pa.bool_(),
        }.get(normalized_dtype)
        if scalar is None:
            raise ValueError(
                "fixed-size parquet list dtype must be float32, int64, or bool; "
                f"got {dtype!r}."
            )
        return pa.list_(scalar, normalized_length)

    arrays = []
    for name in column_names:
        values = [row.get(name) for row in normalized_rows]
        fixed_contract = fixed_contracts.get(name)
        if fixed_contract is not None:
            arrays.append(pa.array(values, type=fixed_list_type(*fixed_contract)))
        elif name in ARM_MODE_INT64_COLUMNS:
            arrays.append(pa.array(values, type=pa.int64()))
        elif name in ARM_MODE_STRING_COLUMNS:
            arrays.append(pa.array(values, type=pa.string()))
        elif name in ARM_MODE_FLOAT64_COLUMNS:
            arrays.append(pa.array(values, type=pa.float64()))
        elif name in FLOAT32_LIST_COLUMNS or name in V22_FLOAT32_LIST_COLUMNS:
            arrays.append(pa.array(values, type=pa.list_(pa.float32())))
        elif name in STRING_LIST_COLUMNS:
            arrays.append(pa.array(values, type=pa.list_(pa.string())))
        elif name in INT64_LIST_COLUMNS or name in V22_INT64_LIST_COLUMNS:
            arrays.append(pa.array(values, type=pa.list_(pa.int64())))
        elif name in BOOL_LIST_COLUMNS:
            arrays.append(pa.array(values, type=pa.list_(pa.bool_())))
        else:
            arrays.append(pa.array(values))
    table = pa.Table.from_arrays(arrays, names=column_names)
    pq.write_table(table, parquet_path, compression=compression)


__all__ = [
    "ARM_MODE_INT64_COLUMNS",
    "ARM_MODE_STRING_COLUMNS",
    "ARM_MODE_FLOAT64_COLUMNS",
    "FLOAT32_LIST_COLUMNS",
    "INT64_LIST_COLUMNS",
    "BOOL_LIST_COLUMNS",
    "STRING_LIST_COLUMNS",
    "V22_FLOAT32_LIST_COLUMNS",
    "V22_INT64_LIST_COLUMNS",
    "write_rows_to_parquet",
]
