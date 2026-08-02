from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_motion.pose import pose6_to_matrix
from robot_motion.rm65.position_collision import (
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    RM65PositionPlannerProtocol,
)

from .official_all_ik_governor import RM65OfficialAllIKGovernorPlanner


def _six_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain 6 positive finite values.")
    return tuple(float(value) for value in array)


def endpoint_to_tool_transforms_from_started_backends(
    arm_backends: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Read the active flange/endpoint-to-tool transforms after backend start.

    RM65 may auto-load its active tool profile in ``backend.start()``.  Reading
    the public state here, rather than copying CLI profile arguments earlier,
    ensures coupled IK and the final real backend use the same TCP definition.
    A complete dual-arm mapping is required because collision geometry retains
    both physical arms even when only one target pose changes.
    """

    transforms: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        backend = arm_backends.get(side)
        if backend is None:
            raise RuntimeError(
                "High-rate dual-arm collision planning requires started backends for both arms; "
                f"missing {side!r}. A passive arm qpos/TCP must never be guessed."
            )
        get_state = getattr(backend, "get_state", None)
        if not callable(get_state):
            raise RuntimeError(f"{side} arm backend does not expose get_state().")
        state = get_state()
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{side} arm backend state is not a mapping.")
        raw_pose = state.get("tool_tcp_pose_flange_m_rad")
        try:
            pose6 = np.asarray(raw_pose, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - annotate backend state failures.
            raise RuntimeError(
                f"{side} active tool TCP is unavailable after backend start."
            ) from exc
        if pose6.shape != (6,) or not np.all(np.isfinite(pose6)):
            raise RuntimeError(
                f"{side} active tool TCP must contain 6 finite flange-relative values."
            )
        transforms[side] = pose6_to_matrix(pose6)
    return transforms


def build_position_collision_planner_from_started_backends(
    arm_backends: Mapping[str, Any],
    *,
    reference_hz: float,
    scene_profile: str,
    collision_min_distance_m: float,
    collision_distance_epsilon_m: float,
    max_joint_velocity_rad_s: Sequence[float],
    max_joint_acceleration_rad_s2: Sequence[float] | None = None,
    max_joint_step_rad: float | Sequence[float] | None = None,
    damping: float = 0.01,
    solve_max_iterations: int = 1,
    solve_min_iterations: int = 1,
    solve_adaptive_iterations_enabled: bool = False,
    solve_iteration_reserve_ms: float = 5.0,
    pose_lift_profile: str | Any | None = None,
    debug_profile: bool = False,
    ik_tuning: Any | None = None,
) -> RM65PositionPlannerProtocol:
    """Build the common pose-to-qpos planner at the real-runtime boundary."""

    max_step: tuple[float, ...] | None
    if max_joint_step_rad is None:
        max_step = None
    elif np.asarray(max_joint_step_rad).ndim == 0:
        scalar = float(max_joint_step_rad)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("max_joint_step_rad must be positive when provided.")
        max_step = (scalar,) * 6
    else:
        max_step = _six_positive(max_joint_step_rad, name="max_joint_step_rad")

    planner = RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=float(reference_hz),
            scene_profile=str(scene_profile),
            collision_min_distance_m=float(collision_min_distance_m),
            collision_distance_epsilon_m=float(collision_distance_epsilon_m),
            max_joint_velocity_rad_s=_six_positive(
                max_joint_velocity_rad_s,
                name="max_joint_velocity_rad_s",
            ),
            **(
                {}
                if max_joint_acceleration_rad_s2 is None
                else {
                    "max_joint_acceleration_rad_s2": _six_positive(
                        max_joint_acceleration_rad_s2,
                        name="max_joint_acceleration_rad_s2",
                    )
                }
            ),
            max_joint_step_rad=max_step,
            damping=float(damping),
            solve_max_iterations=int(solve_max_iterations),
            solve_min_iterations=int(solve_min_iterations),
            solve_adaptive_iterations_enabled=bool(
                solve_adaptive_iterations_enabled
            ),
            solve_iteration_reserve_ms=float(solve_iteration_reserve_ms),
            **(
                {}
                if pose_lift_profile is None
                else {"pose_lift_profile": pose_lift_profile}
            ),
            debug_profile=bool(debug_profile),
            ik_tuning=ik_tuning,
        ),
        endpoint_to_tool_by_side=endpoint_to_tool_transforms_from_started_backends(
            arm_backends
        ),
    )
    if int(planner.collision_pair_count) <= 0:
        raise RuntimeError(
            "required high-rate collision planning refuses a scene with no collision pairs."
        )
    return planner


def build_official_all_ik_governor_from_started_backends(
    arm_backends: Mapping[str, Any],
    *,
    source_hz: float,
    command_hz: float,
    lookahead_source_steps: int,
    jerk_ramp_sec: float,
    scene_profile: str,
    collision_min_distance_m: float,
    collision_distance_epsilon_m: float,
    max_joint_velocity_rad_s: Sequence[float],
    max_joint_acceleration_rad_s2: Sequence[float],
    debug_profile: bool = False,
) -> RM65OfficialAllIKGovernorPlanner:
    """Build the promoted source all-IK + causal qref governor backend."""

    endpoint_to_tool = endpoint_to_tool_transforms_from_started_backends(
        arm_backends
    )
    collision_planner = RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=float(command_hz),
            scene_profile=str(scene_profile),
            collision_min_distance_m=float(collision_min_distance_m),
            collision_distance_epsilon_m=float(collision_distance_epsilon_m),
            max_joint_velocity_rad_s=_six_positive(
                max_joint_velocity_rad_s,
                name="max_joint_velocity_rad_s",
            ),
            max_joint_acceleration_rad_s2=_six_positive(
                max_joint_acceleration_rad_s2,
                name="max_joint_acceleration_rad_s2",
            ),
            solve_max_iterations=1,
            solve_min_iterations=1,
            debug_profile=bool(debug_profile),
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )
    if int(collision_planner.collision_pair_count) <= 0:
        raise RuntimeError(
            "required official all-IK governor refuses a scene with no collision pairs."
        )
    return RM65OfficialAllIKGovernorPlanner(
        collision_planner,
        endpoint_to_tool_by_side=endpoint_to_tool,
        source_hz=float(source_hz),
        command_hz=float(command_hz),
        lookahead_source_steps=int(lookahead_source_steps),
        jerk_ramp_sec=float(jerk_ramp_sec),
        max_velocity_rad_s=max_joint_velocity_rad_s,
        max_acceleration_rad_s2=max_joint_acceleration_rad_s2,
    )


__all__ = [
    "build_official_all_ik_governor_from_started_backends",
    "build_position_collision_planner_from_started_backends",
    "endpoint_to_tool_transforms_from_started_backends",
]
