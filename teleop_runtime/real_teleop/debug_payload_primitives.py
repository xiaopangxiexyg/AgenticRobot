"""Pure compact serializers for real-teleop debug payloads.

The helpers format already-read poses, joint vectors, and metadata.  They do
not access a backend, publish to a UI, or mutate runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from ..transforms import as_pose_matrix


def _pose_matrix_debug(matrix: object) -> dict[str, object] | None:
    try:
        pose = as_pose_matrix(matrix)
    except Exception:
        return None
    try:
        rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except Exception:
        rpy_deg = np.full(3, np.nan)
    return {
        "matrix": pose.copy(),
        "xyz_m": pose[:3, 3].copy(),
        "rpy_deg": rpy_deg,
    }


def _qpos_debug(qpos: object) -> dict[str, object] | None:
    if qpos is None:
        return None
    try:
        values = np.asarray(qpos, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    return {
        "rad": values.copy(),
        "deg": np.rad2deg(values).copy(),
    }


def _compact_numeric_list(
    values: object,
    *,
    max_items: int = 12,
) -> list[float] | dict[str, object] | None:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if arr.size > int(max_items):
        return {
            "omitted": True,
            "reason": "compact_debug_log",
            "shape": list(np.asarray(values).shape),
            "size": int(arr.size),
        }
    return [float(value) for value in arr.tolist()]


def _compact_debug_value(
    value: object,
    *,
    max_items: int = 12,
    depth: int = 0,
) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _compact_numeric_list(value, max_items=max_items)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_debug_log_depth",
                "type": "mapping",
                "size": len(value),
            }
        out: dict[str, object] = {}
        items = list(value.items())
        for raw_key, item in items[:max_items]:
            key = str(raw_key)
            if key == "ik_resolver_init":
                continue
            out[key] = _compact_debug_value(item, max_items=max_items, depth=depth + 1)
        if len(items) > max_items:
            out["_omitted_items"] = int(len(items) - max_items)
        return out
    if isinstance(value, (list, tuple)):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_debug_log_depth",
                "type": "sequence",
                "size": len(value),
            }
        out = [
            _compact_debug_value(item, max_items=max_items, depth=depth + 1)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            out.append({"_omitted_items": int(len(value) - max_items)})
        return out
    return str(value)


def _pose_compact_debug(matrix: object) -> dict[str, object] | None:
    try:
        pose = as_pose_matrix(matrix)
    except Exception:
        return None
    try:
        rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except Exception:
        rpy_deg = np.full(3, np.nan)
    return {
        "xyz_m": [float(value) for value in pose[:3, 3].tolist()],
        "rpy_deg": [float(value) for value in rpy_deg.tolist()],
    }


def _qpos_compact_debug(qpos: object) -> dict[str, object] | None:
    if qpos is None:
        return None
    try:
        values = np.asarray(qpos, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)):
        return None
    return {"deg": [float(value) for value in np.rad2deg(values).tolist()]}


def _pose_delta_compact_debug(
    current_pose: object,
    target_pose: object,
) -> dict[str, float] | None:
    try:
        current = as_pose_matrix(current_pose)
        target = as_pose_matrix(target_pose)
    except Exception:
        return None
    try:
        delta_rot = SciPyRotation.from_matrix(target[:3, :3]) * SciPyRotation.from_matrix(
            current[:3, :3]
        ).inv()
        angular_deg = float(np.rad2deg(delta_rot.magnitude()))
    except Exception:
        angular_deg = float("nan")
    return {
        "translation_m": float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
        "rotation_deg": angular_deg,
    }



__all__ = [
    "_pose_matrix_debug",
    "_qpos_debug",
    "_compact_numeric_list",
    "_compact_debug_value",
    "_pose_compact_debug",
    "_qpos_compact_debug",
    "_pose_delta_compact_debug",
]
