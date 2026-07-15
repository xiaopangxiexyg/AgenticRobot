"""Explicit v2.1 fixed-q6 to v2.2 policy-I/O compatibility adapters.

These helpers never guess a device-specific meaning for legacy q6 padding.
Callers must supply a :class:`PolicyIOLayout`; a 1D gripper conversion is only
performed when that layout explicitly selects its source slot.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from .v22 import (
    ARM_ACTION_DIM,
    DEXTELEOP_DATASET_SCHEMA_V22,
    EndEffectorChannelSpec,
    PolicyIOLayout,
    SIDES,
    bind_policy_io_layout,
)


LEGACY_V21_ACTION_DIM = 26
LEGACY_V21_STATE_DIM = 38
_ACTION_ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
_ACTION_HAND_SLICES = {"left": slice(14, 20), "right": slice(20, 26)}
_STATE_ARM_QPOS_SLICES = {"left": slice(0, 6), "right": slice(19, 25)}
_STATE_ARM_POSE_SLICES = {"left": slice(6, 13), "right": slice(25, 32)}
_STATE_HAND_SLICES = {"left": slice(13, 19), "right": slice(32, 38)}


def _vector_or_none(value: Any, *, length: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if length is not None and vector.shape != (int(length),):
        return None
    return vector.astype(np.float32, copy=True)


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _sample_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("sample_metadata_json")
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _command_metadata(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    root = _sample_metadata(row).get("command_metadata_by_effector")
    if not isinstance(root, Mapping):
        return {}
    value = root.get(f"{side}_hand")
    return value if isinstance(value, Mapping) else {}


def _select_numeric(channel: EndEffectorChannelSpec, raw: Any) -> np.ndarray | None:
    vector = _vector_or_none(raw)
    if vector is None:
        return None
    if channel.source_indices is not None:
        indices = np.asarray(channel.source_indices, dtype=int)
        if indices.size != channel.dim or np.any(indices >= vector.size):
            return None
        return vector[indices].astype(np.float32, copy=True)
    return vector if vector.shape == (channel.dim,) else None


def _select_discrete(
    channel: EndEffectorChannelSpec,
    *,
    raw: Any,
    metadata: Mapping[str, Any],
) -> list[Any] | None:
    if channel.source_indices is not None:
        selected = _select_numeric(channel, raw)
        if selected is None:
            return None
        rounded = np.rint(selected)
        if not np.all(np.isfinite(selected)) or not np.allclose(selected, rounded):
            return None
        return [int(value) for value in rounded]
    values: list[Any] = []
    for token in channel.tokens:
        if token.source_metadata_key is None:
            return None
        value = _mapping_value(metadata, token.source_metadata_key)
        if value is None:
            return None
        values.append(value)
    return values


def _legacy_action(row: Mapping[str, Any], *, last: bool = False) -> np.ndarray | None:
    return _vector_or_none(row.get("observation.last_action" if last else "action"), length=LEGACY_V21_ACTION_DIM)


def _legacy_state(row: Mapping[str, Any]) -> np.ndarray | None:
    return _vector_or_none(row.get("observation.state"), length=LEGACY_V21_STATE_DIM)


def _arm_poses_from_action(row: Mapping[str, Any], *, last: bool = False) -> dict[str, np.ndarray]:
    action = _legacy_action(row, last=last)
    if action is not None:
        return {
            side: action[_ACTION_ARM_SLICES[side]].copy()
            for side in SIDES
            if np.all(np.isfinite(action[_ACTION_ARM_SLICES[side]]))
        }
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = _vector_or_none(row.get(f"action.{side}_arm.pose"), length=7)
        if value is not None and np.all(np.isfinite(value)):
            result[side] = value
    return result


def _arm_poses_from_state(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = _legacy_state(row)
    if state is not None:
        return {
            side: state[_STATE_ARM_POSE_SLICES[side]].copy()
            for side in SIDES
            if np.all(np.isfinite(state[_STATE_ARM_POSE_SLICES[side]]))
        }
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = _vector_or_none(row.get(f"observation.{side}_arm.pose"), length=7)
        if value is not None and np.all(np.isfinite(value)):
            result[side] = value
    return result


def _arm_qpos_from_state(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = _legacy_state(row)
    if state is not None:
        return {
            side: state[_STATE_ARM_QPOS_SLICES[side]].copy()
            for side in SIDES
            if np.all(np.isfinite(state[_STATE_ARM_QPOS_SLICES[side]]))
        }
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = _vector_or_none(row.get(f"observation.{side}_arm.qpos"), length=6)
        if value is not None and np.all(np.isfinite(value)):
            result[side] = value
    return result


def _raw_eef_value(
    row: Mapping[str, Any],
    side: str,
    *,
    role: str,
    last_action: bool = False,
    state_source_field: str = "joint_pos",
) -> tuple[Any, Mapping[str, Any]]:
    if role == "action":
        action = _legacy_action(row, last=last_action)
        if action is not None:
            raw = action[_ACTION_HAND_SLICES[side]]
        else:
            raw = row.get(f"action.{side}_hand.joint_target")
        return raw, {} if last_action else _command_metadata(row, side)
    field = str(state_source_field or "joint_pos")
    raw = row.get(f"observation.hand.{side}.{field}")
    if raw is None and field in {"gripper_state_scalar", "gripper_state", "state"}:
        raw = row.get(f"observation.gripper.{side}.state")
    state = _legacy_state(row)
    if raw is None and state is not None:
        raw = state[_STATE_HAND_SLICES[side]]
    if raw is None:
        raw = row.get(f"observation.{side}_hand.qpos")
    return raw, {}


def _end_effector_values(
    row: Mapping[str, Any],
    layout: PolicyIOLayout,
    *,
    role: str,
    last_action: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    layout_role = "action" if role == "action" else "state"
    for side in SIDES:
        channel = layout.channel(layout_role, side)
        if channel.value_type == "none":
            continue
        raw, metadata = _raw_eef_value(
            row,
            side,
            role=role,
            last_action=last_action,
            state_source_field=layout.end_effector(side).state_source_field,
        )
        if channel.value_type == "continuous":
            selected = _select_numeric(channel, raw)
        else:
            selected = (
                None
                if last_action and role == "action" and channel.source_indices is None
                else _select_discrete(channel, raw=raw, metadata=metadata)
            )
        if selected is not None:
            values[side] = selected
    return values


def _write_component_fields(
    output: dict[str, Any],
    layout: PolicyIOLayout,
    *,
    role: str,
    continuous: np.ndarray,
    discrete: np.ndarray,
) -> None:
    layout_role = "action" if role == "action" else "state"
    prefix = "action" if role == "action" else "observation"
    suffix = "target" if role == "action" else "state"
    for component in layout.components(layout_role, "continuous"):
        if component.name.endswith(".arm.pose"):
            continue
        assert component.side is not None
        output[f"{prefix}.end_effector.{component.side}.{suffix}"] = (
            continuous[component.offset : component.stop].astype(np.float32, copy=False).tolist()
        )
    for component in layout.components(layout_role, "discrete"):
        assert component.side is not None
        output[f"{prefix}.end_effector.{component.side}.{suffix}_token"] = (
            discrete[component.offset : component.stop].astype(np.int64, copy=False).tolist()
        )


def v21_row_to_v22_row(row: Mapping[str, Any], layout: PolicyIOLayout) -> dict[str, Any]:
    """Convert one legacy row without guessing end-effector semantics.

    The returned mapping retains all original columns for provenance.  Its root
    ``action``/``observation.state`` fields become the v2.2 model-facing
    vectors, and the original q6 fields remain available as legacy/audit data.
    """

    output = dict(row)
    action = layout.pack_action(
        arm_pose_by_side=_arm_poses_from_action(row),
        end_effector_by_side=_end_effector_values(row, layout, role="action"),
    )
    state = layout.pack_observation_state(
        arm_pose_by_side=_arm_poses_from_state(row),
        end_effector_by_side=_end_effector_values(row, layout, role="state"),
    )
    last_action = layout.pack_action(
        arm_pose_by_side=_arm_poses_from_action(row, last=True),
        end_effector_by_side=_end_effector_values(
            row,
            layout,
            role="action",
            last_action=True,
        ),
    )
    qpos = layout.pack_arm_qpos_audit(_arm_qpos_from_state(row))
    output["dexteleop.dataset_schema"] = DEXTELEOP_DATASET_SCHEMA_V22
    output["policy_io_layout_hash"] = layout.layout_hash
    output["action"] = action.continuous.tolist()
    output["action.arm.pose"] = action.continuous[:ARM_ACTION_DIM].tolist()
    output["observation.state"] = state.continuous.tolist()
    output["observation.arm.pose"] = state.continuous[:ARM_ACTION_DIM].tolist()
    output["observation.arm.qpos"] = qpos.tolist()
    output["observation.arm.qpos_valid"] = np.isfinite(qpos).tolist()
    output["observation.last_action"] = last_action.continuous.tolist()
    if layout.action_discrete_dim:
        output["action.discrete"] = action.discrete.astype(np.int64, copy=False).tolist()
        output["observation.last_action.discrete"] = last_action.discrete.astype(
            np.int64, copy=False
        ).tolist()
    if layout.state_discrete_dim:
        output["observation.state.discrete"] = state.discrete.astype(np.int64, copy=False).tolist()
    _write_component_fields(
        output,
        layout,
        role="action",
        continuous=action.continuous,
        discrete=action.discrete,
    )
    _write_component_fields(
        output,
        layout,
        role="state",
        continuous=state.continuous,
        discrete=state.discrete,
    )
    return output


def v21_info_to_v22_info(
    info: Mapping[str, Any],
    layout: PolicyIOLayout,
    *,
    conversion_name: str = "v21_to_v22_policy_io",
    conversion_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create v2.2 dataset metadata while preserving LeRobot-v2.1-style IO."""

    output = dict(info)
    existing_hash = output.get("policy_io_layout_hash")
    if existing_hash is not None and str(existing_hash) != layout.layout_hash:
        raise ValueError("Existing info policy_io_layout_hash conflicts with requested layout.")
    output["format"] = str(output.get("format") or "lerobot_v2.1_style")
    output["dexteleop_dataset_schema"] = DEXTELEOP_DATASET_SCHEMA_V22
    output.update(bind_policy_io_layout(None, layout))
    output["features"] = layout.feature_metadata()
    history = output.get("postprocess_history")
    records = list(history) if isinstance(history, list) else []
    record = {
            "operation": str(conversion_name),
            "source_schema": "legacy_v21_or_unversioned",
            "target_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "policy_io_layout_hash": layout.layout_hash,
        }
    if conversion_provenance is not None:
        record["provenance"] = dict(conversion_provenance)
    records.append(record)
    output["postprocess_history"] = records
    return output


__all__ = [
    "LEGACY_V21_ACTION_DIM",
    "LEGACY_V21_STATE_DIM",
    "v21_info_to_v22_info",
    "v21_row_to_v22_row",
]
