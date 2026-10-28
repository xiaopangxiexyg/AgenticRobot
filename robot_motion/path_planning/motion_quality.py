"""Shared kinematic path quality; never collision or execution authority."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PathMotionQualityLimits:
    maximum_proximal_joint_travel_rad: float = math.pi
    maximum_tcp_detour_ratio: float = 4.0
    maximum_tcp_detour_excess_m: float = 0.5

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def exceeded(self, quality: Mapping[str, Any]) -> bool:
        ratio = quality["tcp_detour_ratio"]
        excessive_tcp = (
            quality["tcp_path_length_m"] - quality["tcp_start_goal_distance_m"]
            > self.maximum_tcp_detour_excess_m
            and (ratio is None or ratio > self.maximum_tcp_detour_ratio)
        )
        return bool(
            max(quality["joint_total_travel_deg"][:3])
            > math.degrees(self.maximum_proximal_joint_travel_rad)
            or excessive_tcp
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_proximal_joint_travel_deg": math.degrees(
                self.maximum_proximal_joint_travel_rad
            ),
            "maximum_tcp_detour_ratio": self.maximum_tcp_detour_ratio,
            "maximum_tcp_detour_excess_m": self.maximum_tcp_detour_excess_m,
        }

    def endpoint_joint_travel_exceeded(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Necessary rejection under the existing final proximal-travel limit.

        Total variation of each numeric joint path is at least its net change.
        Do not wrap angles: the executed/quality-checked joint trajectory uses
        these bounded joint coordinates, not equivalent quaternion poses.
        This says nothing about collision, IK, wrist travel or TCP detours.
        """
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if (a.ndim != 1 or a.shape != b.shape or len(a) < 3
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Endpoint quality requires aligned finite joint vectors.")
        lower_bound_deg = np.rad2deg(np.abs(b[:3] - a[:3]))
        return bool(np.max(lower_bound_deg) > math.degrees(self.maximum_proximal_joint_travel_rad))


DEFAULT_PATH_MOTION_QUALITY_LIMITS = PathMotionQualityLimits()


def sampled_path_motion_quality(qpos: np.ndarray, tcp_xyz: np.ndarray) -> dict[str, Any]:
    q = np.asarray(qpos, dtype=float)
    xyz = np.asarray(tcp_xyz, dtype=float)
    if (
        q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 3
        or xyz.shape != (len(q), 3)
        or not np.isfinite(q).all() or not np.isfinite(xyz).all()
    ):
        raise ValueError("Quality requires aligned finite joint and TCP samples.")
    travel = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    direct = float(np.linalg.norm(xyz[-1] - xyz[0]))
    return {
        "scope": "sampled_planned_kinematics_not_realized_motion_or_grasp_success",
        "joint_net_delta_deg": np.rad2deg(q[-1] - q[0]).tolist(),
        "joint_total_travel_deg": np.rad2deg(np.abs(np.diff(q, axis=0)).sum(axis=0)).tolist(),
        "joint_path_length_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
        "tcp_path_length_m": travel,
        "tcp_start_goal_distance_m": direct,
        "tcp_detour_ratio": travel / direct if direct > 1e-9 else None,
        "tcp_max_excursion_from_start_m": float(np.linalg.norm(xyz - xyz[0], axis=1).max()),
        "tcp_world_z_range_m": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
    }


def tcp_path_edge_cost(
    start: Mapping[str, np.ndarray], end: Mapping[str, np.ndarray], *, side: str,
    pose_for_pair: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    step_rad: float,
) -> float:
    """Sample FK arc length for existing shortcut cost, independent of validity."""
    if not math.isfinite(step_rad) or step_rad <= 0:
        raise ValueError("step_rad must be finite and positive.")
    count = max(1, math.ceil(float(np.max(np.abs(end[side] - start[side]))) / step_rad))
    positions = [
        pose_for_pair({name: start[name] + alpha * (end[name] - start[name]) for name in start})[:3, 3]
        for alpha in np.linspace(0.0, 1.0, count + 1)
    ]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
