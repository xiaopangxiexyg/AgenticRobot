from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_TCP_SAFETY_MAX_ENDPOINT_DISTANCE_M = 0.20
DEFAULT_TCP_SAFETY_MAX_ROTATION_DEG = 90.0


def tcp_endpoint_delta(pose_flange_m_rad: Iterable[float]) -> dict[str, float]:
    """Return device-neutral flange-to-TCP translation and rotation magnitudes."""

    pose = np.asarray(tuple(pose_flange_m_rad), dtype=float).reshape(-1)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("tcp.pose_flange_m_rad must contain six finite values.")
    return {
        "translation_m": float(np.linalg.norm(pose[:3])),
        "rotation_deg": float(np.rad2deg(R.from_euler("xyz", pose[3:]).magnitude())),
    }


def tcp_safety_warnings(
    pose_flange_m_rad: Iterable[float],
    *,
    max_endpoint_distance_m: float = DEFAULT_TCP_SAFETY_MAX_ENDPOINT_DISTANCE_M,
    max_rotation_deg: float = DEFAULT_TCP_SAFETY_MAX_ROTATION_DEG,
    label: str = "tool TCP",
) -> list[str]:
    """Describe suspicious tool offsets without importing a vendor backend."""

    distance_limit = float(max_endpoint_distance_m)
    rotation_limit = float(max_rotation_deg)
    if not math.isfinite(distance_limit) or distance_limit < 0.0:
        raise ValueError("max_endpoint_distance_m must be finite and non-negative.")
    if not math.isfinite(rotation_limit) or rotation_limit < 0.0:
        raise ValueError("max_rotation_deg must be finite and non-negative.")
    metrics = tcp_endpoint_delta(pose_flange_m_rad)
    warnings: list[str] = []
    if metrics["translation_m"] > distance_limit:
        warnings.append(
            f"{label} is {metrics['translation_m']:.6f}m from arm_endpoint, "
            f"exceeding {distance_limit:.6f}m"
        )
    if metrics["rotation_deg"] > rotation_limit:
        warnings.append(
            f"{label} rotation differs from arm_endpoint by {metrics['rotation_deg']:.3f}deg, "
            f"exceeding {rotation_limit:.3f}deg"
        )
    return warnings


__all__ = [
    "DEFAULT_TCP_SAFETY_MAX_ENDPOINT_DISTANCE_M",
    "DEFAULT_TCP_SAFETY_MAX_ROTATION_DEG",
    "tcp_endpoint_delta",
    "tcp_safety_warnings",
]
