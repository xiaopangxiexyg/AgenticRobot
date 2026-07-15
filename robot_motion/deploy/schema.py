from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.action import (
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    HandActionSpec,
    arm_action_specs_metadata,
    hand_action_specs_metadata,
    normalize_arm_action_specs,
    normalize_hand_action_specs,
)


ARM_VECTOR_ORDER: tuple[str, ...] = ("left", "right")
HAND_VECTOR_ORDER: tuple[str, ...] = ("left", "right")
ALL_SIDES: tuple[str, ...] = ("left", "right")

ARM_POSE_DOF = 7
ARM_JOINT_DOF = 6
HAND_DOF = 6
ACTION_DIM = 26
OBSERVATION_STATE_DIM = 38
DEPLOY_ACTION_DIM = ACTION_DIM
DEPLOY_OBSERVATION_STATE_DIM = OBSERVATION_STATE_DIM
DEPLOY_ACTION_VECTOR_LAYOUT_VERSION = "deploy_action_vector.ee_pose_q6.v1"
DEPLOY_ACTION_VECTOR_ACTION_SPACE = BIMANUAL_EE_POSE_V1

ACTION_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
ACTION_HAND_SLICES: dict[str, slice] = {
    "left": slice(14, 20),
    "right": slice(20, 26),
}
DEPLOY_ACTION_VECTOR_COMPONENTS: tuple[tuple[str, str, str, slice, int], ...] = (
    ("left_arm_pose", "left", "arm_pose_xyzquat_xyzw", ACTION_ARM_POSE_SLICES["left"], ARM_POSE_DOF),
    ("right_arm_pose", "right", "arm_pose_xyzquat_xyzw", ACTION_ARM_POSE_SLICES["right"], ARM_POSE_DOF),
    ("left_hand_q6", "left", "hand_q6", ACTION_HAND_SLICES["left"], HAND_DOF),
    ("right_hand_q6", "right", "hand_q6", ACTION_HAND_SLICES["right"], HAND_DOF),
)
STATE_ARM_QPOS_SLICES: dict[str, slice] = {
    "left": slice(0, 6),
    "right": slice(19, 25),
}
STATE_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}
STATE_HAND_SLICES: dict[str, slice] = {
    "left": slice(13, 19),
    "right": slice(32, 38),
}

DEFAULT_HAND_ACTION_SPECS_BY_SIDE: dict[str, HandActionSpec] = normalize_hand_action_specs(None)
DEFAULT_HAND_ACTION_SPECS_METADATA: dict[str, Any] = hand_action_specs_metadata(
    DEFAULT_HAND_ACTION_SPECS_BY_SIDE
)
DEFAULT_ARM_ACTION_SPECS_BY_SIDE: dict[str, ArmActionSpec] = normalize_arm_action_specs(None)
DEFAULT_ARM_ACTION_SPECS_METADATA: dict[str, Any] = arm_action_specs_metadata(
    DEFAULT_ARM_ACTION_SPECS_BY_SIDE
)


def normalize_deploy_arm_action_specs(
    specs: Mapping[str, Any] | None,
) -> dict[str, ArmActionSpec]:
    return normalize_arm_action_specs(specs)


def deploy_arm_specs_metadata(specs: Mapping[str, ArmActionSpec] | None = None) -> dict[str, Any]:
    return arm_action_specs_metadata(DEFAULT_ARM_ACTION_SPECS_BY_SIDE if specs is None else specs)


def normalize_deploy_hand_action_specs(
    specs: Mapping[str, Any] | None,
) -> dict[str, HandActionSpec]:
    return normalize_hand_action_specs(specs)


def deploy_hand_specs_metadata(specs: Mapping[str, HandActionSpec] | None = None) -> dict[str, Any]:
    return hand_action_specs_metadata(DEFAULT_HAND_ACTION_SPECS_BY_SIDE if specs is None else specs)


def nan_vector(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=np.float32)


def vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.shape != (int(length),) or not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_serializable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def normalize_active_sides(active_sides: object) -> tuple[str, ...]:
    requested = tuple(str(side).strip().lower() for side in active_sides)
    invalid = sorted(set(requested) - set(ALL_SIDES))
    if invalid:
        raise ValueError(f"Unsupported active sides: {invalid}")
    return tuple(side for side in ALL_SIDES if side in set(requested))


def pose_matrix_to_xyzquat(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        # Position-control contracts intentionally expose immutable snapshots.
        # Some supported SciPy builds reject read-only buffers in
        # ``Rotation.from_matrix``, so detach the small matrix at this API
        # boundary instead of requiring callers to make their data mutable.
        pose = np.array(value, dtype=float, copy=True).reshape(4, 4)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    xyz = pose[:3, 3].astype(np.float32, copy=True)
    quat = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32, copy=False)
    return np.concatenate((xyz, quat), dtype=np.float32)


def xyzquat_to_pose_matrix(value: Any) -> np.ndarray | None:
    vector = vector_or_none(value, length=ARM_POSE_DOF)
    if vector is None:
        return None
    quat = np.asarray(vector[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(vector[:3], dtype=float)
    pose[:3, :3] = R.from_quat(quat / norm).as_matrix()
    return pose


def mapping_get(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any | None:
    for key in aliases:
        if key in mapping:
            return mapping[key]
    return None


__all__ = [
    "ACTION_ARM_POSE_SLICES",
    "ACTION_DIM",
    "ACTION_HAND_SLICES",
    "ALL_SIDES",
    "ARM_JOINT_DOF",
    "ARM_POSE_DOF",
    "ARM_VECTOR_ORDER",
    "ArmActionSpec",
    "DEPLOY_ACTION_DIM",
    "DEPLOY_OBSERVATION_STATE_DIM",
    "DEPLOY_ACTION_VECTOR_ACTION_SPACE",
    "DEPLOY_ACTION_VECTOR_COMPONENTS",
    "DEPLOY_ACTION_VECTOR_LAYOUT_VERSION",
    "DEFAULT_ARM_ACTION_SPECS_BY_SIDE",
    "DEFAULT_ARM_ACTION_SPECS_METADATA",
    "DEFAULT_HAND_ACTION_SPECS_BY_SIDE",
    "DEFAULT_HAND_ACTION_SPECS_METADATA",
    "HAND_DOF",
    "HAND_VECTOR_ORDER",
    "HandActionSpec",
    "OBSERVATION_STATE_DIM",
    "STATE_ARM_POSE_SLICES",
    "STATE_ARM_QPOS_SLICES",
    "STATE_HAND_SLICES",
    "deploy_arm_specs_metadata",
    "deploy_hand_specs_metadata",
    "mapping_get",
    "nan_vector",
    "normalize_active_sides",
    "normalize_deploy_arm_action_specs",
    "normalize_deploy_hand_action_specs",
    "pose_matrix_to_xyzquat",
    "to_serializable",
    "vector_or_none",
    "xyzquat_to_pose_matrix",
]
