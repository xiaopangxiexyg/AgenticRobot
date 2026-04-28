from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..recording import _to_serializable


ARM_VECTOR_ORDER: tuple[str, ...] = ("left", "right")
HAND_VECTOR_ORDER: tuple[str, ...] = ("left", "right")
ALL_SIDES: tuple[str, ...] = ("left", "right")

ARM_POSE_DOF = 7
ARM_JOINT_DOF = 6
HAND_DOF = 6
ACTION_DIM = 26
OBSERVATION_STATE_DIM = 38

ACTION_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
ACTION_HAND_SLICES: dict[str, slice] = {
    "left": slice(14, 20),
    "right": slice(20, 26),
}
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
    return _to_serializable(value)


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
        pose = np.asarray(value, dtype=float).reshape(4, 4)
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
