"""v2.3 model-facing pose views over an absolute-pose dataset.

The parquet representation remains absolute.  This module creates an immutable
window view for training/replay without rewriting rows: state is always desk
absolute, action can be absolute or chunk-relative, and last_action uses the
current row's measured TCP anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError
from dataset_contracts.v22 import DatasetLayoutBindingError, PolicyIOLayout
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    RelativeActionError,
    convert_arm_pose_chunk,
    convert_last_action_chunk,
)

from .policy_io import _float_vector, _token_vector


ANCHOR_COLUMN_TEMPLATE = "observation.arm.{side}.actual_tcp_pose"


class PosePolicyViewError(DatasetLayoutBindingError):
    """Raised when a requested v2.3 pose view cannot be constructed."""


def _normalize_mode(value: str | None) -> str:
    mode = str(value or ACTION_POSE_MODE_ABSOLUTE).strip().lower()
    if mode not in {ACTION_POSE_MODE_ABSOLUTE, ACTION_POSE_MODE_RELATIVE}:
        raise PosePolicyViewError(
            "pose action mode must be 'absolute' or 'relative', "
            f"got {value!r}."
        )
    return mode


def _pose_slices(layout: PolicyIOLayout) -> dict[str, slice]:
    if layout.action_dim < 14 or layout.state_dim < 14:
        raise PosePolicyViewError("v2.3 pose view requires two 7D arm pose prefixes.")
    return {"left": slice(0, 7), "right": slice(7, 14)}


def _row_pose(row: Mapping[str, Any], *, side: str) -> np.ndarray:
    key = ANCHOR_COLUMN_TEMPLATE.format(side=side)
    try:
        value = np.asarray(row[key], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise PosePolicyViewError(f"row is missing a valid actual TCP anchor: {key!r}.") from exc
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise PosePolicyViewError(f"row actual TCP anchor {key!r} is invalid.")
    return value.copy()


def _storage_vector_to_desk(
    values: np.ndarray,
    *,
    contract: PoseFrameContract,
    slices: Mapping[str, slice],
    name: str,
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    # Fixed v2.2/v2.3 vectors retain both arm slots, while a single-arm source
    # legitimately leaves the inactive slot empty.  Only declared source sides
    # are frame-converted; untouched inactive values preserve their sentinel.
    for side, pose_slice in slices.items():
        if side not in set(contract.source.active_sides):
            continue
        try:
            output[pose_slice] = contract.storage_pose_vector_to_desk(
                output[pose_slice], side=side
            )
        except (PoseFrameContractError, ValueError) as exc:
            raise PosePolicyViewError(f"{name}.{side} cannot be converted to desk frame.") from exc
    return output


def _anchors_to_desk(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: PoseFrameContract,
    active_sides: Sequence[str],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for side in active_sides:
        normalized = str(side).strip().lower()
        values = []
        for row in rows:
            try:
                raw = _row_pose(row, side=normalized)
                values.append(
                    contract.storage_pose_vector_to_desk(raw, side=normalized)
                )
            except (PoseFrameContractError, PosePolicyViewError) as exc:
                raise PosePolicyViewError(
                    f"row actual TCP anchor for {normalized!r} cannot be converted to desk frame."
                ) from exc
        output[normalized] = np.stack(values, axis=0).astype(np.float32, copy=False)
    return output


@dataclass(frozen=True)
class PosePolicyWindow:
    """Immutable model-facing window derived from one or more parquet rows."""

    action: np.ndarray
    action_discrete: np.ndarray
    observation_state: np.ndarray
    observation_state_discrete: np.ndarray
    last_action: np.ndarray
    last_action_discrete: np.ndarray
    actual_tcp_pose_by_side: Mapping[str, np.ndarray]
    layout_hash: str
    action_pose_mode: str
    last_action_pose_mode: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = {
            "action": (self.action, np.float32),
            "action_discrete": (self.action_discrete, np.int64),
            "observation_state": (self.observation_state, np.float32),
            "observation_state_discrete": (self.observation_state_discrete, np.int64),
            "last_action": (self.last_action, np.float32),
            "last_action_discrete": (self.last_action_discrete, np.int64),
        }
        for name, (value, dtype) in arrays.items():
            array = np.asarray(value, dtype=dtype)
            if array.ndim != 2:
                raise PosePolicyViewError(f"{name} must be a 2D [horizon, dim] array.")
            object.__setattr__(self, name, array.copy())
        horizon = self.action.shape[0]
        if any(getattr(self, name).shape[0] != horizon for name in arrays if name != "action"):
            raise PosePolicyViewError("All v2.3 pose window arrays must share a horizon.")
        anchors = {
            str(side): np.asarray(values, dtype=np.float32).copy()
            for side, values in dict(self.actual_tcp_pose_by_side).items()
        }
        for side, values in anchors.items():
            if values.shape != (horizon, 7) or not np.all(np.isfinite(values)):
                raise PosePolicyViewError(
                    f"actual_tcp_pose_by_side[{side!r}] must have shape ({horizon}, 7) and be finite."
                )
        object.__setattr__(self, "actual_tcp_pose_by_side", anchors)
        object.__setattr__(self, "layout_hash", str(self.layout_hash))
        object.__setattr__(self, "action_pose_mode", _normalize_mode(self.action_pose_mode))
        object.__setattr__(self, "last_action_pose_mode", _normalize_mode(self.last_action_pose_mode))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def model_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.copy(),
            "action_discrete": self.action_discrete.copy(),
            "observation.state": self.observation_state.copy(),
            "observation.state.discrete": self.observation_state_discrete.copy(),
            "observation.last_action": self.last_action.copy(),
            "observation.last_action.discrete": self.last_action_discrete.copy(),
            "policy_io_layout_hash": self.layout_hash,
            "pose_representation": {
                "state": "absolute_desk",
                "action": self.action_pose_mode,
                "last_action": self.last_action_pose_mode,
                "anchor": "chunk_start_actual_tcp",
                "last_action_anchor": "current_actual_tcp",
            },
            "metadata": dict(self.metadata),
        }


def pose_policy_window_from_rows(
    rows: Sequence[Mapping[str, Any]],
    layout: PolicyIOLayout,
    *,
    pose_frame_contract: PoseFrameContract,
    active_sides: Sequence[str] = ("left", "right"),
    action_pose_mode: str = ACTION_POSE_MODE_ABSOLUTE,
    last_action_pose_mode: str | None = None,
) -> PosePolicyWindow:
    """Build an absolute/relative v2.3 view without changing source rows."""

    if not rows:
        raise PosePolicyViewError("Cannot build a pose policy window from zero rows.")
    action_mode = _normalize_mode(action_pose_mode)
    last_mode = _normalize_mode(last_action_pose_mode or action_mode)
    sides = tuple(dict.fromkeys(str(side).strip().lower() for side in active_sides))
    if not sides or set(sides) - {"left", "right"}:
        raise PosePolicyViewError(f"Unsupported active sides: {active_sides!r}.")
    slices = _pose_slices(layout)

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    last_actions: list[np.ndarray] = []
    action_tokens: list[np.ndarray] = []
    state_tokens: list[np.ndarray] = []
    last_tokens: list[np.ndarray] = []
    for row in rows:
        action = _float_vector(row, "action", length=layout.action_dim)
        state = _float_vector(row, "observation.state", length=layout.state_dim)
        last = _float_vector(row, "observation.last_action", length=layout.action_dim)
        actions.append(_storage_vector_to_desk(action, contract=pose_frame_contract, slices=slices, name="action"))
        states.append(_storage_vector_to_desk(state, contract=pose_frame_contract, slices=slices, name="observation.state"))
        last_actions.append(_storage_vector_to_desk(last, contract=pose_frame_contract, slices=slices, name="observation.last_action"))
        action_tokens.append(_token_vector(row, "action.discrete", length=layout.action_discrete_dim))
        state_tokens.append(_token_vector(row, "observation.state.discrete", length=layout.state_discrete_dim))
        last_tokens.append(_token_vector(row, "observation.last_action.discrete", length=layout.action_discrete_dim and layout.action_discrete_dim))

    action_array = np.stack(actions, axis=0)
    state_array = np.stack(states, axis=0)
    last_array = np.stack(last_actions, axis=0)
    anchors = _anchors_to_desk(rows, contract=pose_frame_contract, active_sides=sides)
    try:
        if action_mode == ACTION_POSE_MODE_RELATIVE:
            action_array = convert_arm_pose_chunk(
                action_array,
                anchors_by_side={side: anchors[side][0] for side in sides},
                source_mode=ACTION_POSE_MODE_ABSOLUTE,
                target_mode=ACTION_POSE_MODE_RELATIVE,
                active_sides=sides,
            )
        if last_mode == ACTION_POSE_MODE_RELATIVE:
            last_array = convert_last_action_chunk(
                last_array,
                current_anchors_by_side={side: anchors[side] for side in sides},
                source_mode=ACTION_POSE_MODE_ABSOLUTE,
                target_mode=ACTION_POSE_MODE_RELATIVE,
                active_sides=sides,
            )
    except RelativeActionError as exc:
        raise PosePolicyViewError(f"Could not construct requested relative pose view: {exc}") from exc

    return PosePolicyWindow(
        action=action_array,
        action_discrete=np.stack(action_tokens, axis=0),
        observation_state=state_array,
        observation_state_discrete=np.stack(state_tokens, axis=0),
        last_action=last_array,
        last_action_discrete=np.stack(last_tokens, axis=0),
        actual_tcp_pose_by_side=anchors,
        layout_hash=layout.layout_hash,
        action_pose_mode=action_mode,
        last_action_pose_mode=last_mode,
        metadata={
            "state_pose_frame": "desk",
            "anchor_policy": "chunk_start_actual_tcp",
            "last_action_anchor_policy": "current_actual_tcp",
            "source_pose_storage_frame": pose_frame_contract.pose_storage_frame,
            "source_id": pose_frame_contract.source_id,
        },
    )


__all__ = [
    "ANCHOR_COLUMN_TEMPLATE",
    "PosePolicyViewError",
    "PosePolicyWindow",
    "pose_policy_window_from_rows",
]
