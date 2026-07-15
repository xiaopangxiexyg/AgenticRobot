"""Stable RobotAction vector codecs.

These dimensions describe serialized ``RobotAction`` action spaces. The 26D
``bimanual_ee_pose_v1`` codec is numerically identical to the deploy policy
vector layout, but the semantic layer is different: codec vectors are tied to a
RobotAction.action_space, while deploy vectors are policy/runtime inputs that
may also carry arm qpos overrides in side metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .schema import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    RobotAction,
    effector_name,
)


EE_POSE_VECTOR_DIM = 26
QPOS_VECTOR_DIM = 24
EE_POSE_HAND16_VECTOR_DIM = 46
QPOS_HAND16_VECTOR_DIM = 44
ROBOT_ACTION_VECTOR_LAYOUT_VERSION = "robot_action_vector.v1"
ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION = "robot_action_vector.hand16.v1"

EE_POSE_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_hand": slice(14, 20),
    "right_hand": slice(20, 26),
}

QPOS_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 6),
    "right_arm": slice(6, 12),
    "left_hand": slice(12, 18),
    "right_hand": slice(18, 24),
}

EE_POSE_HAND16_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_hand": slice(14, 30),
    "right_hand": slice(30, 46),
}

QPOS_HAND16_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 6),
    "right_arm": slice(6, 12),
    "left_hand": slice(12, 28),
    "right_hand": slice(28, 44),
}


@dataclass(frozen=True)
class ActionVectorEncoding:
    action_space: str
    vector: np.ndarray
    component_mask: np.ndarray
    effector_mask: Mapping[str, bool] = field(default_factory=dict)
    layout_version: str = ROBOT_ACTION_VECTOR_LAYOUT_VERSION

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        mask = np.asarray(self.component_mask, dtype=bool).reshape(-1)
        if vector.shape != mask.shape:
            raise ValueError(
                f"vector and component_mask must have the same shape, got {vector.shape} and {mask.shape}."
            )
        object.__setattr__(self, "action_space", str(self.action_space))
        object.__setattr__(self, "vector", vector.astype(np.float32, copy=True))
        object.__setattr__(self, "component_mask", mask.copy())
        object.__setattr__(
            self,
            "effector_mask",
            {str(key): bool(value) for key, value in dict(self.effector_mask).items()},
        )
        object.__setattr__(self, "layout_version", str(self.layout_version))


def _encoding_for_space(action_space: str) -> tuple[int, dict[str, slice]]:
    if action_space == BIMANUAL_EE_POSE_V1:
        return EE_POSE_VECTOR_DIM, EE_POSE_SLICES
    if action_space == BIMANUAL_QPOS_V1:
        return QPOS_VECTOR_DIM, QPOS_SLICES
    if action_space == BIMANUAL_EE_POSE_HAND16_V1:
        return EE_POSE_HAND16_VECTOR_DIM, EE_POSE_HAND16_SLICES
    if action_space == BIMANUAL_QPOS_HAND16_V1:
        return QPOS_HAND16_VECTOR_DIM, QPOS_HAND16_SLICES
    raise ValueError(f"Action space '{action_space}' does not have a stable vector codec.")


def _layout_version_for_space(action_space: str) -> str:
    if action_space in {BIMANUAL_EE_POSE_HAND16_V1, BIMANUAL_QPOS_HAND16_V1}:
        return ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION
    return ROBOT_ACTION_VECTOR_LAYOUT_VERSION


def _is_ee_pose_space(action_space: str) -> bool:
    return action_space in {BIMANUAL_EE_POSE_V1, BIMANUAL_EE_POSE_HAND16_V1}


def _is_hand16_space(action_space: str) -> bool:
    return action_space in {BIMANUAL_EE_POSE_HAND16_V1, BIMANUAL_QPOS_HAND16_V1}


def encode_robot_action(action: RobotAction, *, fill_value: float = np.nan) -> ActionVectorEncoding:
    dim, slices = _encoding_for_space(action.action_space)
    vector = np.full((dim,), float(fill_value), dtype=np.float32)
    component_mask = np.zeros((dim,), dtype=bool)
    effector_mask = {name: False for name in slices}
    active = set(action.active_effectors)

    for side in ("left", "right"):
        arm_effector = effector_name(side, "arm")
        hand_effector = effector_name(side, "hand")
        if _is_ee_pose_space(action.action_space) and arm_effector in active:
            target_slice = slices[arm_effector]
            vector[target_slice] = action.arm_pose_by_side[side]
            component_mask[target_slice] = True
            effector_mask[arm_effector] = True
        if not _is_ee_pose_space(action.action_space) and arm_effector in active:
            target_slice = slices[arm_effector]
            vector[target_slice] = action.arm_qpos_by_side[side]
            component_mask[target_slice] = True
            effector_mask[arm_effector] = True
        if hand_effector in active:
            target_slice = slices[hand_effector]
            hand_values = action.hand_qpos_by_side[side]
            vector[target_slice] = hand_values
            if _is_hand16_space(action.action_space):
                slot_mask = np.isfinite(hand_values)
                component_mask[target_slice] = slot_mask
                effector_mask[hand_effector] = bool(np.any(slot_mask))
            else:
                component_mask[target_slice] = True
                effector_mask[hand_effector] = True

    return ActionVectorEncoding(
        action_space=action.action_space,
        vector=vector,
        component_mask=component_mask,
        effector_mask=effector_mask,
        layout_version=_layout_version_for_space(action.action_space),
    )


def decode_robot_action(
    encoding: ActionVectorEncoding,
    *,
    timestamp: float | None = None,
    source: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> RobotAction:
    _, slices = _encoding_for_space(encoding.action_space)
    vector = np.asarray(encoding.vector, dtype=np.float32).reshape(-1)
    mask = np.asarray(encoding.component_mask, dtype=bool).reshape(-1)
    arm_pose_by_side: dict[str, np.ndarray] = {}
    arm_qpos_by_side: dict[str, np.ndarray] = {}
    hand_qpos_by_side: dict[str, np.ndarray] = {}
    active_effectors: list[str] = []

    for side in ("left", "right"):
        arm_effector = effector_name(side, "arm")
        hand_effector = effector_name(side, "hand")
        arm_slice = slices[arm_effector]
        hand_slice = slices[hand_effector]
        if bool(np.all(mask[arm_slice])):
            if _is_ee_pose_space(encoding.action_space):
                arm_pose_by_side[side] = vector[arm_slice].copy()
            else:
                arm_qpos_by_side[side] = vector[arm_slice].copy()
            active_effectors.append(arm_effector)
        if _is_hand16_space(encoding.action_space):
            hand_active = bool(np.any(mask[hand_slice]))
        else:
            hand_active = bool(np.all(mask[hand_slice]))
        if hand_active:
            hand_qpos_by_side[side] = vector[hand_slice].copy()
            active_effectors.append(hand_effector)

    return RobotAction(
        action_space=encoding.action_space,
        timestamp=timestamp,
        arm_pose_by_side=arm_pose_by_side,
        arm_qpos_by_side=arm_qpos_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        active_effectors=tuple(active_effectors),
        source=source,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "EE_POSE_HAND16_SLICES",
    "EE_POSE_HAND16_VECTOR_DIM",
    "EE_POSE_SLICES",
    "EE_POSE_VECTOR_DIM",
    "QPOS_HAND16_SLICES",
    "QPOS_HAND16_VECTOR_DIM",
    "QPOS_SLICES",
    "QPOS_VECTOR_DIM",
    "ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION",
    "ROBOT_ACTION_VECTOR_LAYOUT_VERSION",
    "ActionVectorEncoding",
    "decode_robot_action",
    "encode_robot_action",
]
