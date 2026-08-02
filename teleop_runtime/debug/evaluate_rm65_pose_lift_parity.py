"""Offline RM65 Pose->QPOS parity evaluator.

This entrypoint never opens a device or dispatches motion.  It compares a
versioned deployment Pose Lift against the accepted q_ref stored in a dataset.
The saved q_ref is the teacher; q_ik_raw is deliberately never used as truth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_motion.position_control import TimestampedMotionTarget, resample_motion_targets
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V1_BASELINE_NAME,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
    resolve_rm65_pose_lift_profile,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
)
from robot_motion.replay import ReplayGroupSpec
from robot_motion.transforms import parse_pose_matrix
from teleop_runtime.replay.executor import (
    REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
    REPLAY_POSITION_IK_DAMPING,
    REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
    REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
    REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS,
    REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS,
)
from teleop_runtime.replay.pose_frames import (
    replay_action_poses_in_hardware_base_at_index,
)
from teleop_runtime.replay.trajectory import build_replay_group_for_episode
from teleop_runtime.replay.trajectory import read_replay_dataset_episode_indices


MODES = ("teacher_forced", "closed_loop", "runtime_causal")
_DIRECTION_EPSILON = 1e-6
_WRIST_NULL_REGION_DEG = 8.0
_WRIST_NULL_MIN_STEP_DEG = 2.0
_WRIST_BRANCH_JUMP_DEG = 10.0
_WRIST_NULL_EQUIVALENT_RATIO = 0.25


def _unwrap_error(candidate: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    delta = np.asarray(candidate, dtype=float) - np.asarray(teacher, dtype=float)
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=float)[:3, :3].T @ np.asarray(right, dtype=float)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _plain_observation(value: Any) -> Any:
    """Copy immutable observer payloads into JSON-native containers."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain_observation(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_observation(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _summary_with_nonfinite(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    raw = np.asarray(tuple(values), dtype=float)
    result = _summary(values)
    result["nonfinite_count"] = int(np.count_nonzero(~np.isfinite(raw)))
    return result


def _sigma_exposure_summary(
    values: Sequence[float],
    *,
    guard_sigma: float = 0.04,
) -> dict[str, Any]:
    """Summarize guarded-band exposure without over-weighting one sampled min."""

    raw = np.asarray(tuple(values), dtype=float)
    array = raw[np.isfinite(raw)]
    guard = float(guard_sigma)
    if not math.isfinite(guard) or guard <= 0.0:
        raise ValueError("sigma exposure guard must be finite and positive.")
    if array.size == 0:
        return {
            "guard_sigma": guard,
            "sample_count": 0,
            "below_guard_count": 0,
            "below_guard_ratio": None,
            "below_0p01_count": 0,
            "below_0p01_ratio": None,
            "below_0p001_count": 0,
            "below_0p001_ratio": None,
            "mean_guard_deficit": None,
            "max_guard_deficit": None,
            "guarded_sigma": _summary(()),
        }
    below_guard = array < guard
    below_0p01 = array < 0.01
    below_0p001 = array < 0.001
    deficit = np.maximum(0.0, guard - array)
    return {
        "guard_sigma": guard,
        "sample_count": int(array.size),
        "below_guard_count": int(np.count_nonzero(below_guard)),
        "below_guard_ratio": float(np.mean(below_guard)),
        "below_0p01_count": int(np.count_nonzero(below_0p01)),
        "below_0p01_ratio": float(np.mean(below_0p01)),
        "below_0p001_count": int(np.count_nonzero(below_0p001)),
        "below_0p001_ratio": float(np.mean(below_0p001)),
        "mean_guard_deficit": float(np.mean(deficit)),
        "max_guard_deficit": float(np.max(deficit)),
        "guarded_sigma": _summary(array[below_guard]),
        "sampling_note": (
            "counts_and_min_are_reference-clock samples; a sign-changing "
            "structural singularity can pass between adjacent samples"
        ),
    }


def _direction_label(delta: float) -> str:
    if float(delta) > _DIRECTION_EPSILON:
        return "away_higher_sigma"
    if float(delta) < -_DIRECTION_EPSILON:
        return "toward_lower_sigma"
    return "flat"


def _wrist_null_redistribution(
    previous: np.ndarray,
    current: np.ndarray,
) -> dict[str, float | bool]:
    """Classify a large q4/q6 redistribution near the coaxial wrist band."""

    before = np.asarray(previous, dtype=float).reshape(6)
    after = np.asarray(current, dtype=float).reshape(6)
    step = _unwrap_error(after, before)
    q4_deg = abs(float(math.degrees(step[3])))
    q6_deg = abs(float(math.degrees(step[5])))
    max_step_deg = max(q4_deg, q6_deg)
    min_step_deg = min(q4_deg, q6_deg)
    sum_motion_deg = abs(float(math.degrees(_unwrap_error(
        np.asarray([after[3] + after[5]]),
        np.asarray([before[3] + before[5]]),
    )[0])))
    difference_motion_deg = abs(float(math.degrees(_unwrap_error(
        np.asarray([after[3] - after[5]]),
        np.asarray([before[3] - before[5]]),
    )[0])))
    equivalent_motion_deg = min(sum_motion_deg, difference_motion_deg)
    near_coaxial = min(
        abs(float(math.degrees(before[4]))),
        abs(float(math.degrees(after[4]))),
    ) <= _WRIST_NULL_REGION_DEG
    redistribution = bool(
        near_coaxial
        and min_step_deg >= _WRIST_NULL_MIN_STEP_DEG
        and equivalent_motion_deg
        <= _WRIST_NULL_EQUIVALENT_RATIO * max(max_step_deg, 1e-12)
    )
    return {
        "near_coaxial": near_coaxial,
        "redistribution": redistribution,
        "branch_jump": bool(
            redistribution and max_step_deg >= _WRIST_BRANCH_JUMP_DEG
        ),
        "max_q4_q6_step_deg": max_step_deg,
        "equivalent_motion_deg": equivalent_motion_deg,
    }


def _configuration_diagnostics(
    planner: Any,
    qpos_by_side: Mapping[str, np.ndarray],
    *,
    characteristic_length_m: float,
) -> Mapping[str, Any] | None:
    analyze = getattr(planner, "analyze_configuration", None)
    if not callable(analyze):
        return None
    result = analyze(
        qpos_by_side,
        characteristic_length_m=float(characteristic_length_m),
    )
    return result if isinstance(result, Mapping) else None


def _task_conditioning_diagnostics(
    planner: Any,
    qpos_by_side: Mapping[str, np.ndarray],
    target_endpoint_pose_world_by_side: Mapping[str, np.ndarray],
    *,
    dt_sec: float,
) -> Mapping[str, Any] | None:
    analyze = getattr(planner, "analyze_task_conditioning", None)
    if not callable(analyze):
        return None
    result = analyze(
        qpos_by_side,
        target_endpoint_pose_world_by_side,
        dt_sec=float(dt_sec),
    )
    if not isinstance(result, Mapping) or not bool(result.get("available", True)):
        return None
    return result


def _constrained_feasibility_diagnostics(
    planner: Any,
    qpos_by_side: Mapping[str, np.ndarray],
    target_endpoint_pose_world_by_side: Mapping[str, np.ndarray],
    *,
    dt_sec: float,
) -> Mapping[str, Any] | None:
    analyze = getattr(planner, "analyze_constrained_task_feasibility", None)
    if not callable(analyze):
        return None
    result = analyze(
        qpos_by_side,
        target_endpoint_pose_world_by_side,
        dt_sec=float(dt_sec),
    )
    if not isinstance(result, Mapping) or not bool(result.get("available", False)):
        return None
    return result


class _AdaptivePostureCandidatePlanner:
    """Offline-only Pareto selector over two fully certified planners.

    Both planners are reset to the same formally selected qref before every
    reference.  Each candidate therefore passes its own joint and collision
    admission before this observer selects one.  The selector is deliberately
    baseline-biased and is not a production planner or dispatch path.
    """

    def __init__(
        self,
        baseline: RM65PositionCollisionPlanner,
        alternate: RM65PositionCollisionPlanner,
        shadow_baseline: RM65PositionCollisionPlanner,
        *,
        guard_sigma: float,
        sigma_tolerance: float,
        sigma_improvement: float,
        pose_score_slack: float,
        pose_score_improvement: float = 0.05,
        max_step_slack_deg: float,
        characteristic_length_m: float,
        dynamic_relative_slack: float = 0.0,
        rollout_baseline: RM65PositionCollisionPlanner | None = None,
        rollout_alternate: RM65PositionCollisionPlanner | None = None,
        rollout_shadow_baseline: RM65PositionCollisionPlanner | None = None,
        horizon_steps: int = 1,
        lock_steps: int = 1,
        terminal_shadow_delta_deg: float = 0.5,
        terminal_mode: str = "joint_shadow",
        terminal_translation_m: float = 0.002,
        terminal_rotation_deg: float = 0.5,
    ) -> None:
        self.baseline = baseline
        self.alternate = alternate
        self.shadow_baseline = shadow_baseline
        self.guard_sigma = float(guard_sigma)
        self.sigma_tolerance = float(sigma_tolerance)
        self.sigma_improvement = float(sigma_improvement)
        self.pose_score_slack = float(pose_score_slack)
        self.pose_score_improvement = float(pose_score_improvement)
        self.max_step_slack_deg = float(max_step_slack_deg)
        self.dynamic_relative_slack = float(dynamic_relative_slack)
        if (
            not math.isfinite(self.dynamic_relative_slack)
            or self.dynamic_relative_slack < 0.0
        ):
            raise ValueError("dynamic_relative_slack must be non-negative.")
        self.characteristic_length_m = float(characteristic_length_m)
        self.rollout_baseline = rollout_baseline
        self.rollout_alternate = rollout_alternate
        self.rollout_shadow_baseline = rollout_shadow_baseline
        self.horizon_steps = max(1, int(horizon_steps))
        self.lock_steps = max(1, int(lock_steps))
        self.terminal_shadow_delta_deg = float(terminal_shadow_delta_deg)
        self.terminal_mode = str(terminal_mode).strip().lower()
        self.terminal_translation_m = float(terminal_translation_m)
        self.terminal_rotation_deg = float(terminal_rotation_deg)
        if self.terminal_mode not in {"joint_shadow", "task_space_tube"}:
            raise ValueError(
                "terminal_mode must be 'joint_shadow' or 'task_space_tube'."
            )
        if (
            not math.isfinite(self.terminal_translation_m)
            or self.terminal_translation_m < 0.0
            or not math.isfinite(self.terminal_rotation_deg)
            or self.terminal_rotation_deg < 0.0
        ):
            raise ValueError("task-space terminal tolerances must be non-negative.")
        rollout_planners = (
            self.rollout_baseline,
            self.rollout_alternate,
            self.rollout_shadow_baseline,
        )
        if self.horizon_steps > 1 and any(
            planner is None for planner in rollout_planners
        ):
            raise ValueError(
                "finite-horizon adaptive posture evaluation requires three "
                "detached rollout planners"
            )
        self._anchor: dict[str, np.ndarray] | None = None
        self._shadow_anchor: dict[str, np.ndarray] | None = None
        self._previous_timestamp: float | None = None
        self._previous_velocity: dict[str, np.ndarray] = {}
        self._previous_acceleration: dict[str, np.ndarray] = {}
        self._selected_dynamic_max = {
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
        }
        self._shadow_dynamic_max = {
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
        }
        self._shadow_previous_timestamp: float | None = None
        self._shadow_previous_velocity: dict[str, np.ndarray] = {}
        self._shadow_previous_acceleration: dict[str, np.ndarray] = {}
        self._reference_window: tuple[Mapping[str, Any], ...] = ()
        self._committed_selection: str | None = None
        self._committed_path: list[dict[str, Any]] = []
        self._baseline_lock_remaining = 0
        self._lock_remaining = 0
        self._plan_id = 0

    @property
    def collision_pair_count(self) -> int:
        return int(self.baseline.collision_pair_count)

    def reset(self, qpos_by_side: Mapping[str, Any]) -> Mapping[str, Any]:
        anchor = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }
        baseline_reset = self.baseline.reset(anchor)
        self.alternate.reset(anchor)
        self.shadow_baseline.reset(anchor)
        self._anchor = anchor
        self._shadow_anchor = {
            side: value.copy() for side, value in anchor.items()
        }
        self._previous_timestamp = None
        self._previous_velocity = {}
        self._previous_acceleration = {}
        self._selected_dynamic_max = {
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
        }
        self._shadow_dynamic_max = {
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
        }
        self._shadow_previous_timestamp = None
        self._shadow_previous_velocity = {}
        self._shadow_previous_acceleration = {}
        self._reference_window = ()
        self._committed_selection = None
        self._committed_path = []
        self._baseline_lock_remaining = 0
        self._lock_remaining = 0
        self._plan_id = 0
        return baseline_reset

    def prepare_reference_window(
        self,
        references: Sequence[Mapping[str, Any]],
    ) -> None:
        """Attach bounded future references for the next offline decision.

        This is intentionally an evaluator-only hook.  Production planners do
        not receive future policy references and no runtime contract is changed.
        """

        self._reference_window = tuple(references[: self.horizon_steps])

    def analyze_configuration(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return self.baseline.analyze_configuration(*args, **kwargs)

    def _metrics(
        self,
        planner: RM65PositionCollisionPlanner,
        result: Any,
        *,
        source_timestamp: float | None,
        anchor: Mapping[str, np.ndarray] | None = None,
        previous_timestamp: float | None = None,
        previous_velocity: Mapping[str, np.ndarray] | None = None,
        previous_acceleration: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        valid = bool(getattr(result, "valid", False)) and not bool(
            getattr(result, "hold", True)
        )
        if not valid:
            return {
                "valid": False,
                "sigma_min": float("-inf"),
                "pose_score": float("inf"),
                "max_step_deg": float("inf"),
            }
        diagnostics = planner.analyze_configuration(
            result.q_ref_by_side,
            characteristic_length_m=self.characteristic_length_m,
        )
        sigma_min = min(
            float(diagnostics["sides"][side]["sigma_min"])
            for side in ("left", "right")
        )
        translation = 0.0
        rotation = 0.0
        for side in result.active_sides:
            target = np.asarray(
                result.target_endpoint_pose_world_by_side[side], dtype=float
            ).reshape(4, 4)
            realized = np.asarray(
                result.realized_endpoint_pose_world_by_side[side], dtype=float
            ).reshape(4, 4)
            translation = max(
                translation,
                float(np.linalg.norm(target[:3, 3] - realized[:3, 3])),
            )
            rotation = max(rotation, _rotation_deg(target, realized))
        resolved_anchor = self._anchor if anchor is None else anchor
        if resolved_anchor is None:
            raise RuntimeError("candidate metrics require an admitted anchor")
        resolved_previous_timestamp = (
            self._previous_timestamp
            if previous_timestamp is None
            else previous_timestamp
        )
        resolved_previous_velocity = (
            self._previous_velocity
            if previous_velocity is None
            else previous_velocity
        )
        resolved_previous_acceleration = (
            self._previous_acceleration
            if previous_acceleration is None
            else previous_acceleration
        )
        max_step_deg = max(
            float(
                np.max(
                    np.abs(
                        np.rad2deg(
                            _unwrap_error(
                                np.asarray(result.q_ref_by_side[side], dtype=float),
                                resolved_anchor[side],
                            )
                        )
                    )
                )
            )
            for side in ("left", "right")
        )
        velocity_by_side: dict[str, np.ndarray] = {}
        acceleration_by_side: dict[str, np.ndarray] = {}
        max_velocity_deg_s: float | None = None
        max_acceleration_deg_s2: float | None = None
        max_jerk_deg_s3: float | None = None
        if source_timestamp is not None and resolved_previous_timestamp is not None:
            dt = float(source_timestamp) - float(resolved_previous_timestamp)
            if math.isfinite(dt) and dt > 0.0:
                max_velocity_deg_s = 0.0
                for side in ("left", "right"):
                    velocity = _unwrap_error(
                        np.asarray(result.q_ref_by_side[side], dtype=float),
                        resolved_anchor[side],
                    ) / dt
                    velocity_by_side[side] = velocity
                    max_velocity_deg_s = max(
                        max_velocity_deg_s,
                        float(np.max(np.abs(np.rad2deg(velocity)))),
                    )
                    if side in resolved_previous_velocity:
                        if max_acceleration_deg_s2 is None:
                            max_acceleration_deg_s2 = 0.0
                        acceleration = (
                            velocity - resolved_previous_velocity[side]
                        ) / dt
                        acceleration_by_side[side] = acceleration
                        max_acceleration_deg_s2 = max(
                            max_acceleration_deg_s2,
                            float(np.max(np.abs(np.rad2deg(acceleration)))),
                        )
                        if side in resolved_previous_acceleration:
                            if max_jerk_deg_s3 is None:
                                max_jerk_deg_s3 = 0.0
                            jerk = (
                                acceleration - resolved_previous_acceleration[side]
                            ) / dt
                            max_jerk_deg_s3 = max(
                                max_jerk_deg_s3,
                                float(np.max(np.abs(np.rad2deg(jerk)))),
                            )
        return {
            "valid": True,
            "sigma_min": sigma_min,
            "translation_m": translation,
            "rotation_deg": rotation,
            "pose_score": max(translation / 0.002, rotation / 0.5),
            "max_step_deg": max_step_deg,
            "max_velocity_deg_s": max_velocity_deg_s,
            "max_acceleration_deg_s2": max_acceleration_deg_s2,
            "max_jerk_deg_s3": max_jerk_deg_s3,
            "_velocity_by_side": velocity_by_side,
            "_acceleration_by_side": acceleration_by_side,
        }

    @staticmethod
    def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metrics.items()
            if not str(key).startswith("_")
        }

    def _rollout_metrics(
        self,
        planner: RM65PositionCollisionPlanner,
        *,
        anchor: Mapping[str, np.ndarray],
        previous_timestamp: float | None,
        previous_velocity: Mapping[str, np.ndarray],
        previous_acceleration: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        planner.reset(anchor)
        current_anchor = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in anchor.items()
        }
        current_timestamp = previous_timestamp
        current_velocity = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in previous_velocity.items()
        }
        current_acceleration = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in previous_acceleration.items()
        }
        aggregate: dict[str, Any] = {
            "valid": True,
            "step_count": 0,
            "hold_count": 0,
            "sigma_min": float("inf"),
            "pose_score": 0.0,
            "pose_score_mean": 0.0,
            "max_step_deg": 0.0,
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
            "max_velocity_deg_s_mean": 0.0,
            "max_acceleration_deg_s2_mean": 0.0,
            "max_jerk_deg_s3_mean": 0.0,
            "terminal_translation_m": float("inf"),
            "terminal_rotation_deg": float("inf"),
            "_results": [],
        }
        pose_score_sum = 0.0
        dynamic_sums = {
            "max_velocity_deg_s": 0.0,
            "max_acceleration_deg_s2": 0.0,
            "max_jerk_deg_s3": 0.0,
        }
        dynamic_counts = {name: 0 for name in dynamic_sums}
        for reference in self._reference_window:
            timestamp = float(reference["source_timestamp"])
            result = planner.plan_pose_reference(
                reference["pose_by_side"],
                target_reference_frame="arm_base",
                target_controlled_frame="tool_tcp",
                source_timestamp=timestamp,
                source="pose_lift_parity:finite_horizon",
                metadata=dict(reference.get("metadata") or {}),
            )
            metrics = self._metrics(
                planner,
                result,
                source_timestamp=timestamp,
                anchor=current_anchor,
                previous_timestamp=current_timestamp,
                previous_velocity=current_velocity,
                previous_acceleration=current_acceleration,
            )
            aggregate["_results"].append(
                {"source_timestamp": timestamp, "result": result}
            )
            aggregate["step_count"] += 1
            if not metrics["valid"]:
                aggregate["valid"] = False
                aggregate["hold_count"] += 1
                break
            aggregate["sigma_min"] = min(
                float(aggregate["sigma_min"]),
                float(metrics["sigma_min"]),
            )
            pose_score = float(metrics["pose_score"])
            pose_score_sum += pose_score
            aggregate["pose_score"] = max(
                float(aggregate["pose_score"]), pose_score
            )
            aggregate["max_step_deg"] = max(
                float(aggregate["max_step_deg"]),
                float(metrics["max_step_deg"]),
            )
            aggregate["terminal_translation_m"] = float(
                metrics["translation_m"]
            )
            aggregate["terminal_rotation_deg"] = float(
                metrics["rotation_deg"]
            )
            for metric_name in (
                "max_velocity_deg_s",
                "max_acceleration_deg_s2",
                "max_jerk_deg_s3",
            ):
                value = metrics.get(metric_name)
                if value is not None:
                    aggregate[metric_name] = max(
                        float(aggregate[metric_name]), float(value)
                    )
                    dynamic_sums[metric_name] += float(value)
                    dynamic_counts[metric_name] += 1
            current_anchor = {
                side: np.asarray(result.q_ref_by_side[side], dtype=float).copy()
                for side in ("left", "right")
            }
            current_timestamp = timestamp
            current_velocity = {
                side: np.asarray(value, dtype=float).copy()
                for side, value in dict(
                    metrics.get("_velocity_by_side") or {}
                ).items()
            }
            current_acceleration = {
                side: np.asarray(value, dtype=float).copy()
                for side, value in dict(
                    metrics.get("_acceleration_by_side") or {}
                ).items()
            }
        completed = int(aggregate["step_count"])
        if completed > 0:
            aggregate["pose_score_mean"] = pose_score_sum / completed
        for metric_name, total in dynamic_sums.items():
            count = dynamic_counts[metric_name]
            aggregate[f"{metric_name}_mean"] = (
                total / count if count > 0 else 0.0
            )
        if not math.isfinite(float(aggregate["sigma_min"])):
            aggregate["sigma_min"] = float("-inf")
        return aggregate

    def _finite_horizon_metrics(self) -> dict[str, dict[str, Any]] | None:
        if self.horizon_steps <= 1 or len(self._reference_window) < 2:
            return None
        if self._anchor is None or self._shadow_anchor is None:
            return None
        if (
            self.rollout_baseline is None
            or self.rollout_alternate is None
            or self.rollout_shadow_baseline is None
        ):
            return None
        metrics = {
            "baseline": self._rollout_metrics(
                self.rollout_baseline,
                anchor=self._anchor,
                previous_timestamp=self._previous_timestamp,
                previous_velocity=self._previous_velocity,
                previous_acceleration=self._previous_acceleration,
            ),
            "alternate": self._rollout_metrics(
                self.rollout_alternate,
                anchor=self._anchor,
                previous_timestamp=self._previous_timestamp,
                previous_velocity=self._previous_velocity,
                previous_acceleration=self._previous_acceleration,
            ),
            "shadow_baseline": self._rollout_metrics(
                self.rollout_shadow_baseline,
                anchor=self._shadow_anchor,
                previous_timestamp=self._shadow_previous_timestamp,
                previous_velocity=self._shadow_previous_velocity,
                previous_acceleration=self._shadow_previous_acceleration,
            ),
        }
        shadow_results = metrics["shadow_baseline"].get("_results") or []
        if shadow_results:
            shadow_q = shadow_results[-1]["result"].q_ref_by_side
            for name in ("baseline", "alternate"):
                candidate_results = metrics[name].get("_results") or []
                if not candidate_results:
                    metrics[name]["terminal_shadow_joint_delta_deg"] = float(
                        "inf"
                    )
                    continue
                candidate_q = candidate_results[-1]["result"].q_ref_by_side
                metrics[name]["terminal_shadow_joint_delta_deg"] = max(
                    float(
                        np.max(
                            np.abs(
                                np.rad2deg(
                                    _unwrap_error(
                                        np.asarray(candidate_q[side], dtype=float),
                                        np.asarray(shadow_q[side], dtype=float),
                                    )
                                )
                            )
                        )
                    )
                    for side in ("left", "right")
                )
        metrics["shadow_baseline"]["terminal_shadow_joint_delta_deg"] = 0.0
        return metrics

    def _select_from_metrics(
        self,
        *,
        baseline_metrics: Mapping[str, Any],
        alternate_metrics: Mapping[str, Any],
        shadow_metrics: Mapping[str, Any],
        anchor_sigma: float,
        horizon: bool,
    ) -> tuple[str, str, dict[str, bool]]:
        if not baseline_metrics["valid"] and alternate_metrics["valid"]:
            return (
                "alternate",
                "baseline_invalid_alternate_certified",
                {
                    "baseline_valid": False,
                    "alternate_valid": True,
                },
            )
        if not (baseline_metrics["valid"] and alternate_metrics["valid"]):
            return (
                "baseline",
                "baseline_default",
                {
                    "baseline_valid": bool(baseline_metrics["valid"]),
                    "alternate_valid": bool(alternate_metrics["valid"]),
                },
            )
        guarded = min(
            anchor_sigma,
            float(baseline_metrics["sigma_min"]),
            float(alternate_metrics["sigma_min"]),
        ) < self.guard_sigma
        step_ok = float(alternate_metrics["max_step_deg"]) <= (
            float(baseline_metrics["max_step_deg"])
            + self.max_step_slack_deg
        )
        sigma_reference = max(
            float(baseline_metrics["sigma_min"]),
            float(shadow_metrics["sigma_min"]),
        )
        sigma_not_worse = float(alternate_metrics["sigma_min"]) >= (
            sigma_reference - self.sigma_tolerance
        )
        pose_better = float(alternate_metrics["pose_score"]) < (
            float(baseline_metrics["pose_score"])
            - self.pose_score_improvement
        )
        sigma_better = float(alternate_metrics["sigma_min"]) >= (
            sigma_reference + self.sigma_improvement
        )
        pose_not_worse = float(alternate_metrics["pose_score"]) <= (
            float(baseline_metrics["pose_score"])
            + self.pose_score_slack
        )
        terminal_delta = alternate_metrics.get("terminal_shadow_joint_delta_deg")
        if not horizon:
            terminal_ok = True
        elif self.terminal_mode == "joint_shadow":
            terminal_ok = bool(
                terminal_delta is not None
                and float(terminal_delta)
                <= self.terminal_shadow_delta_deg
            )
        else:
            terminal_ok = bool(
                float(
                    alternate_metrics.get(
                        "terminal_translation_m",
                        float("inf"),
                    )
                )
                <= self.terminal_translation_m
                and float(
                    alternate_metrics.get(
                        "terminal_rotation_deg",
                        float("inf"),
                    )
                )
                <= self.terminal_rotation_deg
            )
        dynamic_ok = True
        for metric_name in (
            "max_velocity_deg_s",
            "max_acceleration_deg_s2",
            "max_jerk_deg_s3",
        ):
            baseline_value = baseline_metrics.get(metric_name)
            alternate_value = alternate_metrics.get(metric_name)
            if baseline_value is None or alternate_value is None:
                continue
            ceiling = max(
                float(self._shadow_dynamic_max[metric_name]),
                float(shadow_metrics.get(metric_name) or 0.0),
            )
            dynamic_ok = bool(
                dynamic_ok
                and float(alternate_value)
                <= ceiling * (1.0 + self.dynamic_relative_slack) + 1e-9
            )
            if horizon:
                alternate_mean = alternate_metrics.get(f"{metric_name}_mean")
                shadow_mean = shadow_metrics.get(f"{metric_name}_mean")
                if alternate_mean is not None and shadow_mean is not None:
                    dynamic_ok = bool(
                        dynamic_ok
                        and float(alternate_mean)
                        <= float(shadow_mean)
                        * (1.0 + self.dynamic_relative_slack)
                        + 1e-9
                    )
        suffix = "horizon" if horizon else "step"
        decision_checks = {
            "baseline_valid": bool(baseline_metrics["valid"]),
            "alternate_valid": bool(alternate_metrics["valid"]),
            "guarded": bool(guarded),
            "step_ok": bool(step_ok),
            "dynamic_ok": bool(dynamic_ok),
            "terminal_ok": bool(terminal_ok),
            "sigma_not_worse": bool(sigma_not_worse),
            "pose_better": bool(pose_better),
            "sigma_better": bool(sigma_better),
            "pose_not_worse": bool(pose_not_worse),
        }
        if (
            guarded
            and step_ok
            and dynamic_ok
            and terminal_ok
            and sigma_not_worse
            and pose_better
        ):
            return (
                "alternate",
                f"guarded_{suffix}_pose_improvement_sigma_dynamic_nonregression",
                decision_checks,
            )
        if (
            guarded
            and step_ok
            and dynamic_ok
            and terminal_ok
            and sigma_better
            and pose_not_worse
        ):
            return (
                "alternate",
                f"guarded_{suffix}_sigma_improvement_pose_dynamic_bounded",
                decision_checks,
            )
        return "baseline", f"{suffix}_baseline_default", decision_checks

    def plan_pose_reference(self, *args: Any, **kwargs: Any) -> Any:
        if self._anchor is None:
            raise RuntimeError("adaptive posture evaluator planner is not reset")
        source_timestamp_value = kwargs.get("source_timestamp")
        source_timestamp = (
            None
            if source_timestamp_value is None
            else float(source_timestamp_value)
        )
        baseline_result = self.baseline.plan_pose_reference(*args, **kwargs)
        alternate_result = self.alternate.plan_pose_reference(*args, **kwargs)
        shadow_result = self.shadow_baseline.plan_pose_reference(*args, **kwargs)
        baseline_metrics = self._metrics(
            self.baseline,
            baseline_result,
            source_timestamp=source_timestamp,
        )
        alternate_metrics = self._metrics(
            self.alternate,
            alternate_result,
            source_timestamp=source_timestamp,
        )
        if self._shadow_anchor is None:
            raise RuntimeError("adaptive posture shadow planner is not reset")
        shadow_metrics = self._metrics(
            self.shadow_baseline,
            shadow_result,
            source_timestamp=source_timestamp,
            anchor=self._shadow_anchor,
            previous_timestamp=self._shadow_previous_timestamp,
            previous_velocity=self._shadow_previous_velocity,
            previous_acceleration=self._shadow_previous_acceleration,
        )
        anchor_diagnostics = self.baseline.analyze_configuration(
            self._anchor,
            characteristic_length_m=self.characteristic_length_m,
        )
        anchor_sigma = min(
            float(anchor_diagnostics["sides"][side]["sigma_min"])
            for side in ("left", "right")
        )

        horizon_metrics = None
        lock_cancelled = False
        chosen = None
        chosen_metrics = None
        selected = "baseline"
        reason = "baseline_default"
        decision_checks: dict[str, bool] | None = None
        if (
            self._baseline_lock_remaining > 0
            and self._committed_selection == "baseline"
            and baseline_metrics["valid"]
        ):
            selected = "baseline"
            reason = "plan_lock"
            chosen = baseline_result
            chosen_metrics = baseline_metrics
            self._baseline_lock_remaining -= 1
        elif self._committed_path and self._committed_selection is not None:
            entry = self._committed_path.pop(0)
            expected_timestamp = float(entry["source_timestamp"])
            timestamp_matches = bool(
                source_timestamp is not None
                and math.isclose(
                    source_timestamp,
                    expected_timestamp,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            queued_result = entry["result"]
            queued_valid = bool(getattr(queued_result, "valid", False)) and not bool(
                getattr(queued_result, "hold", True)
            )
            if timestamp_matches and queued_valid:
                selected = self._committed_selection
                reason = "plan_lock"
                chosen = queued_result
                chosen_metrics = self._metrics(
                    self.alternate if selected == "alternate" else self.baseline,
                    chosen,
                    source_timestamp=source_timestamp,
                )
            else:
                self._committed_path = []
                self._committed_selection = None
                lock_cancelled = True
        if chosen is None:
            if lock_cancelled and baseline_metrics["valid"]:
                selected = "baseline"
                reason = "plan_lock_cancelled:baseline_certified"
                chosen = baseline_result
                chosen_metrics = baseline_metrics
                self._committed_path = []
                self._committed_selection = selected
                self._baseline_lock_remaining = self.lock_steps - 1
                self._plan_id += 1
            else:
                horizon_metrics = self._finite_horizon_metrics()
                decision_metrics = horizon_metrics or {
                    "baseline": baseline_metrics,
                    "alternate": alternate_metrics,
                    "shadow_baseline": shadow_metrics,
                }
                selected, reason, decision_checks = self._select_from_metrics(
                    baseline_metrics=decision_metrics["baseline"],
                    alternate_metrics=decision_metrics["alternate"],
                    shadow_metrics=decision_metrics["shadow_baseline"],
                    anchor_sigma=anchor_sigma,
                    horizon=horizon_metrics is not None,
                )
                self._plan_id += 1
                self._committed_selection = selected
                if horizon_metrics is not None and selected == "alternate":
                    self._baseline_lock_remaining = 0
                    selected_path = list(
                        horizon_metrics[selected].get("_results") or []
                    )[: self.lock_steps]
                    if selected_path:
                        first_entry = selected_path.pop(0)
                        chosen = first_entry["result"]
                        self._committed_path = selected_path
                        chosen_metrics = self._metrics(
                            self.alternate
                            if selected == "alternate"
                            else self.baseline,
                            chosen,
                            source_timestamp=source_timestamp,
                        )
                elif selected == "baseline":
                    # The rollback path must stay on the authoritative V1
                    # instance. A detached rollout is evidence only and cannot
                    # replace V1 state even when its qpos is numerically close.
                    chosen = baseline_result
                    chosen_metrics = baseline_metrics
                    self._committed_path = []
                    self._baseline_lock_remaining = self.lock_steps - 1
                if chosen is None:
                    chosen = (
                        alternate_result
                        if selected == "alternate"
                        else baseline_result
                    )
                    chosen_metrics = (
                        alternate_metrics
                        if selected == "alternate"
                        else baseline_metrics
                    )
                    self._committed_path = []
                if lock_cancelled:
                    reason = f"plan_lock_cancelled:{reason}"
        self._lock_remaining = max(
            len(self._committed_path), self._baseline_lock_remaining
        )
        if chosen_metrics is None:
            raise RuntimeError("adaptive posture selected path has no metrics")
        selected_q = {
            side: np.asarray(chosen.q_ref_by_side[side], dtype=float).copy()
            for side in ("left", "right")
        }
        # Preserve the selected main planner's exact internal state. Detached
        # rollout results require both candidates to rebase; a main baseline
        # result must never be reset or replaced, otherwise "baseline selected"
        # ceases to be a strict rollback path.
        if chosen is baseline_result:
            self.alternate.reset(selected_q)
        elif chosen is alternate_result:
            self.baseline.reset(selected_q)
        else:
            self.baseline.reset(selected_q)
            self.alternate.reset(selected_q)
        self._anchor = selected_q
        if source_timestamp is not None:
            self._previous_timestamp = source_timestamp
        self._previous_velocity = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in dict(
                chosen_metrics.get("_velocity_by_side") or {}
            ).items()
        }
        self._previous_acceleration = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in dict(
                chosen_metrics.get("_acceleration_by_side") or {}
            ).items()
        }
        for metric_name in self._selected_dynamic_max:
            value = chosen_metrics.get(metric_name)
            if value is not None:
                self._selected_dynamic_max[metric_name] = max(
                    float(self._selected_dynamic_max[metric_name]),
                    float(value),
                )
        shadow_q = {
            side: np.asarray(shadow_result.q_ref_by_side[side], dtype=float).copy()
            for side in ("left", "right")
        }
        self._shadow_anchor = shadow_q
        if source_timestamp is not None:
            self._shadow_previous_timestamp = source_timestamp
        self._shadow_previous_velocity = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in dict(
                shadow_metrics.get("_velocity_by_side") or {}
            ).items()
        }
        self._shadow_previous_acceleration = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in dict(
                shadow_metrics.get("_acceleration_by_side") or {}
            ).items()
        }
        for metric_name in self._shadow_dynamic_max:
            value = shadow_metrics.get(metric_name)
            if value is not None:
                self._shadow_dynamic_max[metric_name] = max(
                    float(self._shadow_dynamic_max[metric_name]),
                    float(value),
                )
        public_baseline_metrics = self._public_metrics(baseline_metrics)
        public_alternate_metrics = self._public_metrics(alternate_metrics)
        public_shadow_metrics = self._public_metrics(shadow_metrics)
        metadata = dict(getattr(chosen, "metadata", {}) or {})
        metadata["adaptive_posture_candidate_selection"] = {
            "offline_only": True,
            "selected": selected,
            "reason": reason,
            "anchor_sigma_min": anchor_sigma,
            "baseline": public_baseline_metrics,
            "alternate": public_alternate_metrics,
            "shadow_baseline": public_shadow_metrics,
            "selected_dynamic_max": dict(self._selected_dynamic_max),
            "shadow_dynamic_max": dict(self._shadow_dynamic_max),
            "horizon_steps": self.horizon_steps,
            "terminal_shadow_delta_deg": self.terminal_shadow_delta_deg,
            "terminal_mode": self.terminal_mode,
            "terminal_translation_m": self.terminal_translation_m,
            "terminal_rotation_deg": self.terminal_rotation_deg,
            "pose_score_improvement": self.pose_score_improvement,
            "dynamic_relative_slack": self.dynamic_relative_slack,
            "horizon": (
                None
                if horizon_metrics is None
                else {
                    name: self._public_metrics(metrics)
                    for name, metrics in horizon_metrics.items()
                }
            ),
            "plan_id": self._plan_id,
            "plan_lock_remaining": self._lock_remaining,
            "plan_lock_cancelled": lock_cancelled,
            "decision_checks": decision_checks,
        }
        return replace(chosen, metadata=metadata)


def _truth_at(group: Any, index: int) -> dict[str, np.ndarray]:
    return {
        side: np.asarray(group.arm_joint_targets[side][int(index)], dtype=float).reshape(6)
        for side in ("left", "right")
    }


def _pose_at(group: Any, index: int) -> dict[str, np.ndarray]:
    raw = replay_action_poses_in_hardware_base_at_index(
        group,
        ("left", "right"),
        index=int(index),
    )
    return {side: parse_pose_matrix(value) for side, value in raw.items()}


def _runtime_references(group: Any, reference_hz: float) -> list[Any]:
    samples = [
        TimestampedMotionTarget(
            timestamp=float(group.relative_time_sec(index)),
            source_index=int(index),
            pose_by_side=_pose_at(group, index),
        )
        for index in range(int(group.frame_count))
    ]
    return resample_motion_targets(
        samples,
        reference_hz=float(reference_hz),
        include_final=True,
    )


def _runtime_truth(group: Any, reference: Any) -> dict[str, np.ndarray]:
    left = _truth_at(group, int(reference.left_source_index))
    right = _truth_at(group, int(reference.right_source_index))
    alpha = float(reference.interpolation_alpha)
    return {
        side: left[side] + alpha * _unwrap_error(right[side], left[side])
        for side in ("left", "right")
    }


def evaluate_group(
    group: Any,
    *,
    planner_factory: Callable[[], Any],
    modes: Sequence[str] = MODES,
    reference_hz: float = 60.0,
    include_frames: bool = False,
    include_constrained_feasibility: bool = False,
    jacobian_characteristic_length_m: float = (
        DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M
    ),
) -> dict[str, Any]:
    """Evaluate one detached replay group without device or transport access."""

    missing = [side for side in ("left", "right") if side not in group.arm_joint_targets]
    if missing:
        raise ValueError(f"parity evaluation requires saved accepted q_ref for {missing}.")
    if int(group.frame_count) < 2:
        raise ValueError("parity evaluation requires at least two frames.")
    characteristic_length_m = float(jacobian_characteristic_length_m)
    if not math.isfinite(characteristic_length_m) or characteristic_length_m <= 0.0:
        raise ValueError("jacobian_characteristic_length_m must be finite and positive.")
    output: dict[str, Any] = {
        "episode_index": int(group.source_episode_index),
        "frame_count": int(group.frame_count),
        "first_frame_contract": "qpos_init_from_saved_accepted_qref",
        "truth_contract": "saved_accepted_qref_not_q_ik_raw",
        "reference_clock_contract": (
            "all_modes_resampled_to_deploy_reference_hz"
        ),
        "modes": {},
    }
    runtime_references = [
        reference
        for reference in _runtime_references(group, reference_hz)
        if int(reference.sequence) > 0
    ]
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown parity mode {mode!r}; expected one of {MODES}.")
        planner = planner_factory()
        planner.reset(_truth_at(group, 0))
        # All three modes exercise the same deployment reference clock.  The
        # distinction is the planner state contract, not the sample rate:
        # teacher-forced resets to the preceding interpolated accepted qref,
        # while closed-loop and runtime-causal retain their admitted state.
        # Running the first two modes at the stored dataset rate would give a
        # time-aware QP only one 60 Hz motion interval for a 15 Hz source
        # interval and manufacture fourfold lag that deployment never sees.
        work = [
            (
                reference.pose_by_side,
                int(reference.right_source_index),
                float(reference.timestamp),
                _runtime_truth(group, reference),
            )
            for reference in runtime_references
        ]

        joint_error_deg: list[float] = []
        translation_error_m: list[float] = []
        rotation_error_deg: list[float] = []
        wrist_equivalent_error_deg: list[float] = []
        q5_branch_mismatch = 0
        holds = 0
        longest_hold = 0
        hold_run = 0
        statuses: dict[str, int] = {}
        failure_codes: dict[str, int] = {}
        pose_lift_reasons: dict[str, int] = {}
        pose_lift_applied = 0
        pose_lift_selected_scales: dict[str, dict[str, int]] = {}
        adaptive_posture_selections: dict[str, int] = {}
        adaptive_posture_reasons: dict[str, int] = {}
        adaptive_posture_failed_checks: dict[str, int] = {}
        adaptive_posture_guarded_failed_checks: dict[str, int] = {}
        adaptive_posture_guarded_decisions = 0
        adaptive_posture_guarded_route_counts = {
            "pose_quality_eligible": 0,
            "sigma_quality_eligible": 0,
            "sigma_quality_terminal_eligible": 0,
            "pose_route_eligible": 0,
            "sigma_route_eligible": 0,
        }
        previous_output = _truth_at(group, 0)
        previous_truth = _truth_at(group, 0)
        previous_timestamp = 0.0
        previous_velocity: dict[str, np.ndarray] = {}
        previous_acceleration: dict[str, np.ndarray] = {}
        max_step_deg: list[float] = []
        max_velocity_deg_s: list[float] = []
        max_acceleration_deg_s2: list[float] = []
        max_jerk_deg_s3: list[float] = []
        post_qp_admission_delta_deg: list[float] = []
        raw_velocity_utilization: list[float] = []
        admitted_velocity_utilization: list[float] = []
        raw_acceleration_utilization: list[float] = []
        admitted_acceleration_utilization: list[float] = []
        motion_limit_violation_counts = {
            "raw_velocity_side_frames": 0,
            "admitted_velocity_side_frames": 0,
            "raw_acceleration_side_frames": 0,
            "admitted_acceleration_side_frames": 0,
        }
        velocity_limit = np.asarray(
            REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
            dtype=float,
        )
        acceleration_limit = np.asarray(
            RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
            dtype=float,
        )
        kinematic_dt_skipped_count = 0
        wrist_motion: dict[str, dict[str, dict[str, Any]]] = {
            stream: {
                side: {
                    "redistribution_count": 0,
                    "branch_jump_count": 0,
                    "max_q4_q6_step_deg": [],
                    "equivalent_motion_deg": [],
                }
                for side in ("left", "right")
            }
            for stream in ("truth", "raw_ik", "admitted")
        }
        jacobian_available = False
        jacobian_values: dict[str, dict[str, dict[str, list[float]]]] = {
            stream: {
                side: {
                    "sigma_min": [],
                    "condition_number": [],
                    "min_abs_q3_or_q5_deg": [],
                }
                for side in ("left", "right")
            }
            for stream in ("truth", "raw_ik", "admitted")
        }
        jacobian_direction_counts: dict[str, dict[str, dict[str, int]]] = {
            comparison: {
                side: {
                    "away_higher_sigma": 0,
                    "toward_lower_sigma": 0,
                    "flat": 0,
                }
                for side in ("left", "right")
            }
            for comparison in (
                "admitted_vs_previous_admitted",
                "truth_vs_previous_truth",
                "raw_candidate_vs_previous_admitted",
            )
        }
        task_conditioning_available = False
        task_conditioning_values: dict[
            str, dict[str, dict[str, list[float]]]
        ] = {
            stream: {
                side: {
                    "sigma_task_velocity_min": [],
                    "condition_number_task_velocity": [],
                    "max_required_velocity_utilization": [],
                    "weakest_direction_abs_demand": [],
                    "weakest_direction_required_velocity_utilization": [],
                }
                for side in ("left", "right")
            }
            for stream in ("pre_solve", "truth", "raw_ik", "admitted")
        }
        constrained_feasibility_available = False
        constrained_global_alpha: list[float] = []
        constrained_side_alpha: dict[str, list[float]] = {
            "left": [],
            "right": [],
        }
        constrained_side_tracking_residual: dict[str, list[float]] = {
            "left": [],
            "right": [],
        }
        constrained_active_source_frame_counts: dict[str, dict[str, int]] = {
            "global": {},
            "left": {},
            "right": {},
        }
        previous_admitted_diagnostics = _configuration_diagnostics(
            planner,
            previous_output,
            characteristic_length_m=characteristic_length_m,
        )
        previous_truth_diagnostics = _configuration_diagnostics(
            planner,
            previous_truth,
            characteristic_length_m=characteristic_length_m,
        )
        jacobian_available = bool(
            previous_admitted_diagnostics is not None
            and previous_truth_diagnostics is not None
        )
        joint_error_by_side_joint: dict[str, list[list[float]]] = {
            side: [[] for _ in range(6)] for side in ("left", "right")
        }
        raw_joint_error_by_side_joint: dict[str, list[list[float]]] = {
            side: [[] for _ in range(6)] for side in ("left", "right")
        }
        admitted_from_raw_by_side_joint: dict[str, list[list[float]]] = {
            side: [[] for _ in range(6)] for side in ("left", "right")
        }
        worst_joint_error_frame: dict[str, Any] | None = None
        first_joint_error_above_5deg: dict[str, Any] | None = None
        frames: list[dict[str, Any]] = []
        for work_offset, (
            pose_by_side,
            truth_index,
            timestamp,
            truth,
        ) in enumerate(work):
            sequence = work_offset + 1
            dt = 1.0 / float(reference_hz)
            if mode == "teacher_forced":
                planner.reset(previous_truth)
            prepare_reference_window = getattr(
                planner, "prepare_reference_window", None
            )
            if callable(prepare_reference_window):
                window_steps = max(
                    1, int(getattr(planner, "horizon_steps", 1))
                )
                prepare_reference_window(
                    [
                        {
                            "pose_by_side": future_pose,
                            "source_timestamp": float(future_timestamp),
                            "metadata": {
                                "truth_frame_index": int(future_truth_index)
                            },
                        }
                        for (
                            future_pose,
                            future_truth_index,
                            future_timestamp,
                            _future_truth,
                        ) in work[work_offset : work_offset + window_steps]
                    ]
                )
            pre_solve_target_endpoint_world: Mapping[str, np.ndarray] | None = None
            resolve_targets = getattr(
                planner,
                "resolve_target_endpoint_pose_world",
                None,
            )
            if callable(resolve_targets):
                resolved_targets = resolve_targets(
                    pose_by_side,
                    target_reference_frame="arm_base",
                    target_controlled_frame="tool_tcp",
                )
                if isinstance(resolved_targets, Mapping):
                    pre_solve_target_endpoint_world = resolved_targets
            pre_solve_task_conditioning = (
                _task_conditioning_diagnostics(
                    planner,
                    previous_output,
                    pre_solve_target_endpoint_world,
                    dt_sec=dt,
                )
                if pre_solve_target_endpoint_world is not None
                else None
            )
            pre_solve_constrained_feasibility = (
                _constrained_feasibility_diagnostics(
                    planner,
                    previous_output,
                    pre_solve_target_endpoint_world,
                    dt_sec=dt,
                )
                if (
                    include_constrained_feasibility
                    and pre_solve_target_endpoint_world is not None
                )
                else None
            )
            result = planner.plan_pose_reference(
                pose_by_side,
                target_reference_frame="arm_base",
                target_controlled_frame="tool_tcp",
                source_timestamp=float(timestamp),
                source=f"pose_lift_parity:{mode}",
                metadata={"truth_frame_index": int(truth_index)},
            )
            status = str(getattr(result, "status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            hold = bool(getattr(result, "hold", True)) or not bool(
                getattr(result, "valid", False)
            )
            result_metadata = dict(getattr(result, "metadata", {}) or {})
            failure = result_metadata.get("failure")
            if isinstance(failure, Mapping):
                failure_code = str(failure.get("code") or "unknown")
                failure_codes[failure_code] = failure_codes.get(failure_code, 0) + 1
            pose_lift = result_metadata.get("pose_lift")
            if isinstance(pose_lift, Mapping):
                lift_reason = str(pose_lift.get("reason") or "unknown")
                pose_lift_reasons[lift_reason] = pose_lift_reasons.get(lift_reason, 0) + 1
                pose_lift_applied += int(bool(pose_lift.get("applied", False)))
                selected_scales = pose_lift.get("selected_scales")
                if isinstance(selected_scales, Mapping):
                    for side, raw_scale in selected_scales.items():
                        try:
                            scale_key = f"{float(raw_scale):g}"
                        except (TypeError, ValueError):
                            scale_key = "invalid"
                        side_counts = pose_lift_selected_scales.setdefault(
                            str(side), {}
                        )
                        side_counts[scale_key] = side_counts.get(scale_key, 0) + 1
            adaptive_posture = result_metadata.get(
                "adaptive_posture_candidate_selection"
            )
            if isinstance(adaptive_posture, Mapping):
                selection = str(adaptive_posture.get("selected") or "unknown")
                selection_reason = str(adaptive_posture.get("reason") or "unknown")
                adaptive_posture_selections[selection] = (
                    adaptive_posture_selections.get(selection, 0) + 1
                )
                adaptive_posture_reasons[selection_reason] = (
                    adaptive_posture_reasons.get(selection_reason, 0) + 1
                )
                decision_checks = adaptive_posture.get("decision_checks")
                if isinstance(decision_checks, Mapping):
                    for check_name, passed in decision_checks.items():
                        if bool(passed):
                            continue
                        key = str(check_name)
                        adaptive_posture_failed_checks[key] = (
                            adaptive_posture_failed_checks.get(key, 0) + 1
                        )
                    if bool(decision_checks.get("guarded", False)):
                        adaptive_posture_guarded_decisions += 1
                        for check_name, passed in decision_checks.items():
                            if bool(passed):
                                continue
                            key = str(check_name)
                            adaptive_posture_guarded_failed_checks[key] = (
                                adaptive_posture_guarded_failed_checks.get(key, 0)
                                + 1
                            )
                        common_ready = all(
                            bool(decision_checks.get(name, False))
                            for name in (
                                "baseline_valid",
                                "alternate_valid",
                                "step_ok",
                                "dynamic_ok",
                                "terminal_ok",
                            )
                        )
                        if (
                            bool(
                                decision_checks.get("sigma_not_worse", False)
                            )
                            and bool(decision_checks.get("pose_better", False))
                        ):
                            adaptive_posture_guarded_route_counts[
                                "pose_quality_eligible"
                            ] += 1
                        sigma_quality_eligible = bool(
                            decision_checks.get("sigma_better", False)
                        ) and bool(
                            decision_checks.get("pose_not_worse", False)
                        )
                        if sigma_quality_eligible:
                            adaptive_posture_guarded_route_counts[
                                "sigma_quality_eligible"
                            ] += 1
                        if sigma_quality_eligible and bool(
                            decision_checks.get("terminal_ok", False)
                        ):
                            adaptive_posture_guarded_route_counts[
                                "sigma_quality_terminal_eligible"
                            ] += 1
                        if (
                            common_ready
                            and bool(
                                decision_checks.get("sigma_not_worse", False)
                            )
                            and bool(decision_checks.get("pose_better", False))
                        ):
                            adaptive_posture_guarded_route_counts[
                                "pose_route_eligible"
                            ] += 1
                        if (
                            common_ready
                            and bool(decision_checks.get("sigma_better", False))
                            and bool(
                                decision_checks.get("pose_not_worse", False)
                            )
                        ):
                            adaptive_posture_guarded_route_counts[
                                "sigma_route_eligible"
                            ] += 1
            holds += int(hold)
            hold_run = hold_run + 1 if hold else 0
            longest_hold = max(longest_hold, hold_run)
            prior_output = {
                side: np.asarray(previous_output[side], dtype=float).copy()
                for side in ("left", "right")
            }
            prior_truth = {
                side: np.asarray(previous_truth[side], dtype=float).copy()
                for side in ("left", "right")
            }
            source_dt = float(timestamp) - float(previous_timestamp)
            if not math.isfinite(source_dt) or source_dt <= 0.0:
                raise ValueError(
                    f"parity references must have increasing timestamps; "
                    f"previous={previous_timestamp}, current={timestamp}."
                )
            # The planner integrates every emitted reference with its fixed
            # deploy control period.  ``resample_motion_targets`` may append a
            # non-grid-aligned final source timestamp, but that final target is
            # still one physical control tick, not a fractional servo cycle.
            # Dynamic certificates must therefore use the same fixed period as
            # the planner instead of the source-time interpolation interval.
            frame_max_joint_error = 0.0
            frame_max_step = 0.0
            frame_sides: dict[str, Any] = {}
            actual_by_side: dict[str, np.ndarray] = {}
            raw_by_side: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                actual = np.asarray(result.q_ref_by_side[side], dtype=float).reshape(6)
                actual_by_side[side] = actual.copy()
                delta = _unwrap_error(actual, truth[side])
                delta_deg = np.rad2deg(delta)
                for joint_index, value in enumerate(delta_deg):
                    joint_error_by_side_joint[side][joint_index].append(abs(float(value)))
                raw_value = getattr(result, "q_ik_raw_by_side", {}).get(side)
                raw = (
                    None
                    if raw_value is None
                    else np.asarray(raw_value, dtype=float).reshape(6)
                )
                if raw is not None:
                    raw_by_side[side] = raw.copy()
                    raw_delta_deg = np.rad2deg(_unwrap_error(raw, truth[side]))
                    admitted_from_raw_deg = np.rad2deg(_unwrap_error(actual, raw))
                    for joint_index in range(6):
                        raw_joint_error_by_side_joint[side][joint_index].append(
                            abs(float(raw_delta_deg[joint_index]))
                        )
                        admitted_from_raw_by_side_joint[side][joint_index].append(
                            abs(float(admitted_from_raw_deg[joint_index]))
                        )
                frame_max_joint_error = max(
                    frame_max_joint_error,
                    float(np.max(np.abs(delta_deg))),
                )
                frame_max_step = max(
                    frame_max_step,
                    float(
                        np.max(
                            np.abs(
                                np.rad2deg(
                                    _unwrap_error(actual, previous_output[side])
                                )
                            )
                        )
                    ),
                )
                q5_branch_mismatch += int(
                    abs(float(truth[side][4])) > math.radians(3.0)
                    and np.sign(actual[4]) != np.sign(truth[side][4])
                )
                sum_error = abs(float(_unwrap_error(
                    np.asarray([actual[3] + actual[5]]),
                    np.asarray([truth[side][3] + truth[side][5]]),
                )[0]))
                diff_error = abs(float(_unwrap_error(
                    np.asarray([actual[3] - actual[5]]),
                    np.asarray([truth[side][3] - truth[side][5]]),
                )[0]))
                wrist_equivalent_error_deg.append(
                    float(math.degrees(min(sum_error, diff_error)))
                )
                target_value = result.target_endpoint_pose_world_by_side.get(side)
                realized_value = result.realized_endpoint_pose_world_by_side.get(side)
                side_translation_error_m: float | None = None
                side_rotation_error_deg: float | None = None
                if target_value is not None and realized_value is not None:
                    target_world = np.asarray(target_value, dtype=float).reshape(4, 4)
                    realized_world = np.asarray(realized_value, dtype=float).reshape(4, 4)
                    side_translation_error_m = float(
                        np.linalg.norm(
                            target_world[:3, 3] - realized_world[:3, 3]
                        )
                    )
                    side_rotation_error_deg = _rotation_deg(
                        target_world, realized_world
                    )
                    translation_error_m.append(side_translation_error_m)
                    rotation_error_deg.append(side_rotation_error_deg)
                frame_sides[side] = {
                    "truth_q_deg": np.rad2deg(truth[side]).tolist(),
                    "admitted_q_deg": np.rad2deg(actual).tolist(),
                    "raw_ik_q_deg": None if raw is None else np.rad2deg(raw).tolist(),
                    "error_deg": delta_deg.tolist(),
                    "requested_to_admitted_translation_m": (
                        side_translation_error_m
                    ),
                    "requested_to_admitted_rotation_deg": (
                        side_rotation_error_deg
                    ),
                }

            frame_velocity = 0.0
            frame_acceleration = 0.0
            frame_jerk = 0.0
            frame_motion_certificate: dict[str, Any] = {}
            acceleration_computed = False
            jerk_computed = False
            kinematic_dt_valid = True
            for side in ("left", "right"):
                velocity = _unwrap_error(
                    actual_by_side[side], prior_output[side]
                ) / dt
                raw_velocity = (
                    None
                    if side not in raw_by_side
                    else _unwrap_error(
                        raw_by_side[side], prior_output[side]
                    )
                    / dt
                )
                post_qp_delta = (
                    None
                    if side not in raw_by_side
                    else float(
                        np.max(
                            np.abs(
                                np.rad2deg(
                                    _unwrap_error(
                                        actual_by_side[side], raw_by_side[side]
                                    )
                                )
                            )
                        )
                    )
                )
                if post_qp_delta is not None:
                    post_qp_admission_delta_deg.append(post_qp_delta)
                side_certificate: dict[str, Any] = {
                    "post_qp_admission_delta_deg": post_qp_delta,
                }
                if kinematic_dt_valid:
                    frame_velocity = max(
                        frame_velocity,
                        float(np.max(np.abs(np.rad2deg(velocity)))),
                    )
                    admitted_velocity_ratio = float(
                        np.max(np.abs(velocity) / velocity_limit)
                    )
                    admitted_velocity_utilization.append(
                        admitted_velocity_ratio
                    )
                    motion_limit_violation_counts[
                        "admitted_velocity_side_frames"
                    ] += int(admitted_velocity_ratio > 1.0 + 1e-5)
                    side_certificate["admitted_velocity_utilization"] = (
                        admitted_velocity_ratio
                    )
                    if raw_velocity is not None:
                        raw_velocity_ratio = float(
                            np.max(np.abs(raw_velocity) / velocity_limit)
                        )
                        raw_velocity_utilization.append(raw_velocity_ratio)
                        motion_limit_violation_counts[
                            "raw_velocity_side_frames"
                        ] += int(raw_velocity_ratio > 1.0 + 1e-5)
                        side_certificate["raw_velocity_utilization"] = (
                            raw_velocity_ratio
                        )
                if kinematic_dt_valid and side in previous_velocity:
                    acceleration_computed = True
                    acceleration = (velocity - previous_velocity[side]) / dt
                    frame_acceleration = max(
                        frame_acceleration,
                        float(np.max(np.abs(np.rad2deg(acceleration)))),
                    )
                    admitted_acceleration_ratio = float(
                        np.max(np.abs(acceleration) / acceleration_limit)
                    )
                    admitted_acceleration_utilization.append(
                        admitted_acceleration_ratio
                    )
                    motion_limit_violation_counts[
                        "admitted_acceleration_side_frames"
                    ] += int(admitted_acceleration_ratio > 1.0 + 1e-5)
                    side_certificate["admitted_acceleration_utilization"] = (
                        admitted_acceleration_ratio
                    )
                    if raw_velocity is not None:
                        raw_acceleration = (
                            raw_velocity - previous_velocity[side]
                        ) / dt
                        raw_acceleration_ratio = float(
                            np.max(
                                np.abs(raw_acceleration) / acceleration_limit
                            )
                        )
                        raw_acceleration_utilization.append(
                            raw_acceleration_ratio
                        )
                        motion_limit_violation_counts[
                            "raw_acceleration_side_frames"
                        ] += int(raw_acceleration_ratio > 1.0 + 1e-5)
                        side_certificate["raw_acceleration_utilization"] = (
                            raw_acceleration_ratio
                        )
                    if side in previous_acceleration:
                        jerk_computed = True
                        jerk = (acceleration - previous_acceleration[side]) / dt
                        frame_jerk = max(
                            frame_jerk,
                            float(np.max(np.abs(np.rad2deg(jerk)))),
                        )
                    previous_acceleration[side] = acceleration.copy()
                if kinematic_dt_valid:
                    previous_velocity[side] = velocity.copy()
                frame_motion_certificate[side] = side_certificate

                stream_pairs = {
                    "truth": (prior_truth[side], truth[side]),
                    "admitted": (prior_output[side], actual_by_side[side]),
                }
                if side in raw_by_side:
                    stream_pairs["raw_ik"] = (
                        prior_output[side],
                        raw_by_side[side],
                    )
                for stream, (before_q, after_q) in stream_pairs.items():
                    wrist = _wrist_null_redistribution(before_q, after_q)
                    stream_metrics = wrist_motion[stream][side]
                    stream_metrics["redistribution_count"] += int(
                        bool(wrist["redistribution"])
                    )
                    stream_metrics["branch_jump_count"] += int(
                        bool(wrist["branch_jump"])
                    )
                    stream_metrics["max_q4_q6_step_deg"].append(
                        float(wrist["max_q4_q6_step_deg"])
                    )
                    stream_metrics["equivalent_motion_deg"].append(
                        float(wrist["equivalent_motion_deg"])
                    )

            if kinematic_dt_valid:
                max_velocity_deg_s.append(frame_velocity)
            if acceleration_computed:
                max_acceleration_deg_s2.append(frame_acceleration)
            if jerk_computed:
                max_jerk_deg_s3.append(frame_jerk)

            diagnostics_by_stream: dict[str, Mapping[str, Any] | None] = {
                "truth": _configuration_diagnostics(
                    planner,
                    truth,
                    characteristic_length_m=characteristic_length_m,
                ),
                "admitted": _configuration_diagnostics(
                    planner,
                    actual_by_side,
                    characteristic_length_m=characteristic_length_m,
                ),
                "raw_ik": (
                    _configuration_diagnostics(
                        planner,
                        raw_by_side,
                        characteristic_length_m=characteristic_length_m,
                    )
                    if set(raw_by_side) == {"left", "right"}
                    else None
                ),
            }
            target_endpoint_world = dict(
                getattr(
                    result,
                    "target_endpoint_pose_world_by_side",
                    {},
                )
                or {}
            )
            task_conditioning_by_stream: dict[
                str, Mapping[str, Any] | None
            ] = {
                "pre_solve": pre_solve_task_conditioning,
                "truth": (
                    _task_conditioning_diagnostics(
                        planner,
                        truth,
                        target_endpoint_world,
                        dt_sec=dt,
                    )
                    if set(target_endpoint_world) == {"left", "right"}
                    else None
                ),
                "admitted": (
                    _task_conditioning_diagnostics(
                        planner,
                        actual_by_side,
                        target_endpoint_world,
                        dt_sec=dt,
                    )
                    if set(target_endpoint_world) == {"left", "right"}
                    else None
                ),
                "raw_ik": (
                    _task_conditioning_diagnostics(
                        planner,
                        raw_by_side,
                        target_endpoint_world,
                        dt_sec=dt,
                    )
                    if (
                        set(raw_by_side) == {"left", "right"}
                        and set(target_endpoint_world) == {"left", "right"}
                    )
                    else None
                ),
            }
            task_conditioning_available = bool(
                task_conditioning_available
                or any(
                    diagnostics is not None
                    for diagnostics in task_conditioning_by_stream.values()
                )
            )
            for stream, diagnostics in task_conditioning_by_stream.items():
                if diagnostics is None:
                    continue
                sides = diagnostics.get("sides")
                if not isinstance(sides, Mapping):
                    continue
                for side in ("left", "right"):
                    side_metrics = sides.get(side)
                    if not isinstance(side_metrics, Mapping):
                        continue
                    for name in (
                        "sigma_task_velocity_min",
                        "condition_number_task_velocity",
                        "max_required_velocity_utilization",
                        "weakest_direction_abs_demand",
                        "weakest_direction_required_velocity_utilization",
                    ):
                        value = side_metrics.get(name)
                        if value is not None:
                            task_conditioning_values[stream][side][name].append(
                                float(value)
                            )
            frame_constrained_feasibility = pre_solve_constrained_feasibility
            if frame_constrained_feasibility is not None:
                constrained_feasibility_available = True
                strict = frame_constrained_feasibility.get(
                    "strict_directional_alpha"
                )
                if isinstance(strict, Mapping):
                    global_metrics = strict.get("global")
                    if isinstance(global_metrics, Mapping) and bool(
                        global_metrics.get("available", False)
                    ):
                        constrained_global_alpha.append(
                            float(global_metrics["alpha"])
                        )
                        for source_metrics in (
                            global_metrics.get("active_inequality_sources") or ()
                        ):
                            if not isinstance(source_metrics, Mapping):
                                continue
                            source = str(source_metrics.get("source") or "")
                            if source and int(
                                source_metrics.get("active_count") or 0
                            ) > 0:
                                counts = constrained_active_source_frame_counts[
                                    "global"
                                ]
                                counts[source] = counts.get(source, 0) + 1
                    frame_metrics = strict.get("frames")
                    if isinstance(frame_metrics, Sequence):
                        for side, side_metrics in zip(
                            ("left", "right"),
                            frame_metrics,
                        ):
                            if isinstance(side_metrics, Mapping) and bool(
                                side_metrics.get("available", False)
                            ):
                                constrained_side_alpha[side].append(
                                    float(side_metrics["alpha"])
                                )
                                for source_metrics in (
                                    side_metrics.get(
                                        "active_inequality_sources"
                                    )
                                    or ()
                                ):
                                    if not isinstance(source_metrics, Mapping):
                                        continue
                                    source = str(
                                        source_metrics.get("source") or ""
                                    )
                                    if source and int(
                                        source_metrics.get("active_count") or 0
                                    ) > 0:
                                        counts = (
                                            constrained_active_source_frame_counts[
                                                side
                                            ]
                                        )
                                        counts[source] = (
                                            counts.get(source, 0) + 1
                                        )
                tracking = frame_constrained_feasibility.get(
                    "best_constrained_tracking"
                )
                if isinstance(tracking, Mapping):
                    frame_metrics = tracking.get("frames")
                    if isinstance(frame_metrics, Sequence):
                        for side, side_metrics in zip(
                            ("left", "right"),
                            frame_metrics,
                        ):
                            if isinstance(side_metrics, Mapping):
                                constrained_side_tracking_residual[side].append(
                                    float(
                                        side_metrics[
                                            "weighted_residual_norm"
                                        ]
                                    )
                                )
            jacobian_available = bool(
                jacobian_available
                or diagnostics_by_stream["truth"] is not None
                or diagnostics_by_stream["admitted"] is not None
            )
            for stream, diagnostics in diagnostics_by_stream.items():
                if diagnostics is None:
                    continue
                sides = diagnostics.get("sides")
                if not isinstance(sides, Mapping):
                    continue
                for side in ("left", "right"):
                    side_metrics = sides.get(side)
                    if not isinstance(side_metrics, Mapping):
                        continue
                    for name in (
                        "sigma_min",
                        "condition_number",
                        "min_abs_q3_or_q5_deg",
                    ):
                        value = side_metrics.get(name)
                        if value is not None:
                            jacobian_values[stream][side][name].append(float(value))

            def record_direction(
                comparison: str,
                side: str,
                current_diagnostics: Mapping[str, Any] | None,
                prior_diagnostics: Mapping[str, Any] | None,
            ) -> None:
                if current_diagnostics is None or prior_diagnostics is None:
                    return
                current_sides = current_diagnostics.get("sides")
                prior_sides = prior_diagnostics.get("sides")
                if not isinstance(current_sides, Mapping) or not isinstance(
                    prior_sides, Mapping
                ):
                    return
                current_side = current_sides.get(side)
                prior_side = prior_sides.get(side)
                if not isinstance(current_side, Mapping) or not isinstance(
                    prior_side, Mapping
                ):
                    return
                delta_sigma = float(current_side["sigma_min"]) - float(
                    prior_side["sigma_min"]
                )
                label = _direction_label(delta_sigma)
                jacobian_direction_counts[comparison][side][label] += 1

            for side in ("left", "right"):
                record_direction(
                    "admitted_vs_previous_admitted",
                    side,
                    diagnostics_by_stream["admitted"],
                    previous_admitted_diagnostics,
                )
                record_direction(
                    "truth_vs_previous_truth",
                    side,
                    diagnostics_by_stream["truth"],
                    previous_truth_diagnostics,
                )
                record_direction(
                    "raw_candidate_vs_previous_admitted",
                    side,
                    diagnostics_by_stream["raw_ik"],
                    previous_admitted_diagnostics,
                )
            previous_admitted_diagnostics = diagnostics_by_stream["admitted"]
            previous_truth_diagnostics = diagnostics_by_stream["truth"]
            previous_output = {
                side: actual_by_side[side].copy() for side in ("left", "right")
            }
            previous_truth = {
                side: np.asarray(truth[side], dtype=float).copy()
                for side in ("left", "right")
            }
            previous_timestamp = float(timestamp)
            joint_error_deg.append(frame_max_joint_error)
            max_step_deg.append(frame_max_step)
            frame_diagnostic = {
                "sequence": sequence,
                "truth_frame_index": int(truth_index),
                "timestamp": float(timestamp),
                "status": status,
                "hold": hold,
                "max_joint_error_deg": frame_max_joint_error,
                "pose_lift_reason": (
                    None
                    if not isinstance(pose_lift, Mapping)
                    else str(pose_lift.get("reason") or "unknown")
                ),
                "pose_lift_selected_scales": (
                    {}
                    if not isinstance(pose_lift, Mapping)
                    else dict(pose_lift.get("selected_scales") or {})
                ),
                "sides": frame_sides,
            }
            if (
                worst_joint_error_frame is None
                or frame_max_joint_error
                > float(worst_joint_error_frame["max_joint_error_deg"])
            ):
                worst_joint_error_frame = frame_diagnostic
            if first_joint_error_above_5deg is None and frame_max_joint_error > 5.0:
                first_joint_error_above_5deg = frame_diagnostic
            if include_frames:
                frame_singularity: dict[str, Any] = {}
                for stream, diagnostics in diagnostics_by_stream.items():
                    if not isinstance(diagnostics, Mapping):
                        continue
                    diagnostic_sides = diagnostics.get("sides")
                    if not isinstance(diagnostic_sides, Mapping):
                        continue
                    frame_singularity[stream] = {
                        side: {
                            name: float(side_metrics[name])
                            for name in (
                                "sigma_min",
                                "condition_number",
                                "min_abs_q3_or_q5_deg",
                            )
                            if name in side_metrics
                        }
                        for side in ("left", "right")
                        if isinstance(
                            side_metrics := diagnostic_sides.get(side), Mapping
                        )
                    }
                frame_task_conditioning: dict[str, Any] = {}
                for stream, diagnostics in task_conditioning_by_stream.items():
                    if not isinstance(diagnostics, Mapping):
                        continue
                    diagnostic_sides = diagnostics.get("sides")
                    if not isinstance(diagnostic_sides, Mapping):
                        continue
                    frame_task_conditioning[stream] = {
                        side: {
                            name: (
                                int(side_metrics[name])
                                if name == "weakest_direction_index"
                                else float(side_metrics[name])
                            )
                            for name in (
                                "sigma_task_velocity_min",
                                "condition_number_task_velocity",
                                "max_required_velocity_utilization",
                                "weakest_direction_index",
                                "weakest_direction_abs_demand",
                                "weakest_direction_required_velocity_utilization",
                            )
                            if side_metrics.get(name) is not None
                        }
                        for side in ("left", "right")
                        if isinstance(
                            side_metrics := diagnostic_sides.get(side), Mapping
                        )
                    }
                frame_feasibility = {}
                if isinstance(frame_constrained_feasibility, Mapping):
                    strict = frame_constrained_feasibility.get(
                        "strict_directional_alpha"
                    )
                    tracking = frame_constrained_feasibility.get(
                        "best_constrained_tracking"
                    )
                    frame_feasibility = {
                        "strict_directional_alpha": (
                            {}
                            if not isinstance(strict, Mapping)
                            else _plain_observation(strict)
                        ),
                        "best_constrained_tracking": (
                            {}
                            if not isinstance(tracking, Mapping)
                            else _plain_observation(tracking)
                        ),
                    }
                frames.append(
                    {
                        "sequence": sequence,
                        "truth_frame_index": int(truth_index),
                        "timestamp": float(timestamp),
                        "status": status,
                        "hold": hold,
                        "pose_lift_reason": frame_diagnostic["pose_lift_reason"],
                        "pose_lift_selected_scales": frame_diagnostic[
                            "pose_lift_selected_scales"
                        ],
                        "adaptive_posture_candidate_selection": (
                            None
                            if not isinstance(adaptive_posture, Mapping)
                            else _plain_observation(adaptive_posture)
                        ),
                        "max_joint_error_deg": frame_max_joint_error,
                        "max_step_deg": frame_max_step,
                        "max_velocity_deg_s": frame_velocity,
                        "max_acceleration_deg_s2": (
                            frame_acceleration if acceleration_computed else None
                        ),
                        "max_jerk_deg_s3": frame_jerk if jerk_computed else None,
                        "sides": frame_sides,
                        "singularity": frame_singularity,
                        "task_conditioning": frame_task_conditioning,
                        "constrained_task_feasibility": frame_feasibility,
                        "motion_constraint_certificate": frame_motion_certificate,
                    }
                )
        wrist_motion_report = {
            stream: {
                side: {
                    "redistribution_count": int(values["redistribution_count"]),
                    "branch_jump_count": int(values["branch_jump_count"]),
                    "max_q4_q6_step_deg": _summary(
                        values["max_q4_q6_step_deg"]
                    ),
                    "equivalent_motion_deg": _summary(
                        values["equivalent_motion_deg"]
                    ),
                }
                for side, values in sides.items()
            }
            for stream, sides in wrist_motion.items()
        }
        jacobian_report = {
            "available": bool(jacobian_available),
            "observational_only": True,
            "characteristic_length_m": characteristic_length_m,
            "row_order": "linear_xyz_then_angular_xyz",
            "normalization": "linear_rows_divided_by_characteristic_length",
            "threshold_contract": (
                "no universal sigma or condition threshold assigned; compare "
                "like-for-like profiles and retain q3/q5 structural metrics"
            ),
            "streams": {
                stream: {
                    side: {
                        "sigma_min": _summary(values["sigma_min"]),
                        "guarded_exposure": _sigma_exposure_summary(
                            values["sigma_min"]
                        ),
                        "condition_number": _summary_with_nonfinite(
                            values["condition_number"]
                        ),
                        "min_abs_q3_or_q5_deg": _summary(
                            values["min_abs_q3_or_q5_deg"]
                        ),
                    }
                    for side, values in sides.items()
                }
                for stream, sides in jacobian_values.items()
            },
            "direction_counts": jacobian_direction_counts,
        }
        task_conditioning_report = {
            "available": bool(task_conditioning_available),
            "observational_only": True,
            "control_authority": "none",
            "jacobian_contract": "pink_frame_task_se3_log_local",
            "task_weighting": "pink_cost_left_multiply",
            "joint_scaling": (
                "columns_multiplied_by_formal_joint_velocity_limit_rad_s"
            ),
            "demand_contract": (
                "pink_deadbeat_minus_gain_error_divided_by_reference_dt"
            ),
            "threshold_contract": (
                "no trigger threshold assigned; retain geometric sigma and "
                "q3/q5 structural margin as separate diagnostics"
            ),
            "streams": {
                stream: {
                    side: {
                        "sigma_task_velocity_min": _summary(
                            values["sigma_task_velocity_min"]
                        ),
                        "condition_number_task_velocity": _summary_with_nonfinite(
                            values["condition_number_task_velocity"]
                        ),
                        "max_required_velocity_utilization": (
                            _summary_with_nonfinite(
                                values["max_required_velocity_utilization"]
                            )
                        ),
                        "weakest_direction_abs_demand": _summary(
                            values["weakest_direction_abs_demand"]
                        ),
                        "weakest_direction_required_velocity_utilization": (
                            _summary_with_nonfinite(
                                values[
                                    "weakest_direction_required_velocity_utilization"
                                ]
                            )
                        ),
                    }
                    for side, values in sides.items()
                }
                for stream, sides in task_conditioning_values.items()
            },
        }
        constrained_feasibility_report = {
            "enabled": bool(include_constrained_feasibility),
            "available": bool(constrained_feasibility_available),
            "observational_only": True,
            "control_authority": "none",
            "strict_directional_alpha_contract": (
                "max_exact_requested_direction_scale_under_current_qp_inequalities"
            ),
            "global_alpha": _summary(constrained_global_alpha),
            "active_inequality_source_frame_counts": {
                scope: dict(sorted(counts.items()))
                for scope, counts in (
                    constrained_active_source_frame_counts.items()
                )
            },
            "sides": {
                side: {
                    "strict_directional_alpha": _summary(
                        constrained_side_alpha[side]
                    ),
                    "best_tracking_weighted_residual_norm": _summary(
                        constrained_side_tracking_residual[side]
                    ),
                }
                for side in ("left", "right")
            },
        }
        mode_report = {
            "evaluated_references": len(work),
            "hold_count": holds,
            "longest_hold_run": longest_hold,
            "status_counts": statuses,
            "failure_code_counts": failure_codes,
            "pose_lift_applied_count": pose_lift_applied,
            "pose_lift_reason_counts": pose_lift_reasons,
            "pose_lift_selected_scale_counts": pose_lift_selected_scales,
            "adaptive_posture_selection_counts": adaptive_posture_selections,
            "adaptive_posture_reason_counts": adaptive_posture_reasons,
            "adaptive_posture_failed_check_counts": (
                adaptive_posture_failed_checks
            ),
            "adaptive_posture_guarded_decision_count": (
                adaptive_posture_guarded_decisions
            ),
            "adaptive_posture_guarded_failed_check_counts": (
                adaptive_posture_guarded_failed_checks
            ),
            "adaptive_posture_guarded_route_counts": (
                adaptive_posture_guarded_route_counts
            ),
            "q5_branch_mismatch_side_frames": q5_branch_mismatch,
            "max_joint_error_deg": _summary(joint_error_deg),
            "requested_to_admitted_translation_m": _summary(translation_error_m),
            "requested_to_admitted_rotation_deg": _summary(rotation_error_deg),
            "wrist_q4_sum_or_difference_error_deg": _summary(
                wrist_equivalent_error_deg
            ),
            "admitted_max_step_deg": _summary(max_step_deg),
            "admitted_max_velocity_deg_s": _summary(max_velocity_deg_s),
            "admitted_max_acceleration_deg_s2": _summary(
                max_acceleration_deg_s2
            ),
            "admitted_max_jerk_deg_s3": _summary(max_jerk_deg_s3),
            "kinematic_dt_skipped_count": int(kinematic_dt_skipped_count),
            "motion_constraint_certificate": {
                "observational_only": True,
                "reference_clock_hz": float(reference_hz),
                "velocity_limit_rad_s": velocity_limit.tolist(),
                "acceleration_limit_rad_s2": acceleration_limit.tolist(),
                "violation_ratio_epsilon": 1e-5,
                "violation_counts": dict(motion_limit_violation_counts),
                "raw_ik_velocity_utilization": _summary(
                    raw_velocity_utilization
                ),
                "admitted_velocity_utilization": _summary(
                    admitted_velocity_utilization
                ),
                "raw_ik_acceleration_utilization": _summary(
                    raw_acceleration_utilization
                ),
                "admitted_acceleration_utilization": _summary(
                    admitted_acceleration_utilization
                ),
                "post_qp_admission_delta_deg": _summary(
                    post_qp_admission_delta_deg
                ),
                "post_qp_scope": (
                    "pose_lift_canonicalization_joint_certificate_collision_backoff"
                ),
            },
            "wrist_branch_continuity": wrist_motion_report,
            "singularity_diagnostics": jacobian_report,
            "task_conditioning_diagnostics": task_conditioning_report,
            "constrained_task_feasibility": constrained_feasibility_report,
            "joint_error_deg_by_side_joint": {
                side: {
                    f"q{joint_index + 1}": _summary(values)
                    for joint_index, values in enumerate(per_joint)
                }
                for side, per_joint in joint_error_by_side_joint.items()
            },
            "raw_ik_joint_error_deg_by_side_joint": {
                side: {
                    f"q{joint_index + 1}": _summary(values)
                    for joint_index, values in enumerate(per_joint)
                }
                for side, per_joint in raw_joint_error_by_side_joint.items()
            },
            "admitted_from_raw_ik_deg_by_side_joint": {
                side: {
                    f"q{joint_index + 1}": _summary(values)
                    for joint_index, values in enumerate(per_joint)
                }
                for side, per_joint in admitted_from_raw_by_side_joint.items()
            },
            "first_joint_error_above_5deg": first_joint_error_above_5deg,
            "worst_joint_error_frame": worst_joint_error_frame,
        }
        if include_frames:
            mode_report["frames"] = frames
        output["modes"][mode] = mode_report
    return output


def rank_singular_groups(groups: Sequence[Any], *, limit: int) -> list[dict[str, Any]]:
    """Rank episodes by saved accepted q_ref proximity to q3/q5 singular bands."""

    rows: list[dict[str, Any]] = []
    for group in groups:
        side_metrics: dict[str, Any] = {}
        episode_min = float("inf")
        for side in ("left", "right"):
            values = np.asarray(group.arm_joint_targets[side], dtype=float).reshape(-1, 6)
            q3_abs_deg = np.abs(np.rad2deg(values[:, 2]))
            q5_abs_deg = np.abs(np.rad2deg(values[:, 4]))
            minimum = float(min(np.min(q3_abs_deg), np.min(q5_abs_deg)))
            episode_min = min(episode_min, minimum)
            side_metrics[side] = {
                "min_abs_q3_deg": float(np.min(q3_abs_deg)),
                "min_abs_q5_deg": float(np.min(q5_abs_deg)),
                "side_frames_inside_6deg": int(
                    np.count_nonzero((q3_abs_deg < 6.0) | (q5_abs_deg < 6.0))
                ),
                "side_frames_inside_3_25deg": int(
                    np.count_nonzero((q3_abs_deg < 3.25) | (q5_abs_deg < 3.25))
                ),
            }
        rows.append(
            {
                "episode_index": int(group.source_episode_index),
                "frame_count": int(group.frame_count),
                "min_abs_q3_or_q5_deg": episode_min,
                "sides": side_metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["min_abs_q3_or_q5_deg"]),
            -sum(
                int(metrics["side_frames_inside_6deg"])
                for metrics in row["sides"].values()
            ),
            int(row["episode_index"]),
        )
    )
    return rows[: max(1, int(limit))]


def rank_jacobian_singular_groups(
    groups: Sequence[Any],
    *,
    planner: Any,
    limit: int,
    characteristic_length_m: float,
) -> list[dict[str, Any]]:
    """Rank saved accepted qref using the same normalized Jacobian as Pose Lift."""

    rows: list[dict[str, Any]] = []
    for group in groups:
        sigma_by_side = {"left": [], "right": []}
        frame_count = int(group.frame_count)
        for frame_index in range(frame_count):
            diagnostics = planner.analyze_configuration(
                {
                    side: np.asarray(
                        group.arm_joint_targets[side][frame_index],
                        dtype=float,
                    )
                    for side in ("left", "right")
                },
                characteristic_length_m=float(characteristic_length_m),
            )
            for side in ("left", "right"):
                sigma_by_side[side].append(
                    float(diagnostics["sides"][side]["sigma_min"])
                )
        side_metrics = {
            side: {
                "sigma_min": _summary(values),
                "guarded_exposure": _sigma_exposure_summary(values),
            }
            for side, values in sigma_by_side.items()
        }
        rows.append(
            {
                "episode_index": int(group.source_episode_index),
                "frame_count": frame_count,
                "min_sigma": min(
                    float(side_metrics[side]["sigma_min"]["min"])
                    for side in ("left", "right")
                ),
                "sides": side_metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["min_sigma"]),
            -sum(
                int(metrics["guarded_exposure"]["below_guard_count"])
                for metrics in row["sides"].values()
            ),
            int(row["episode_index"]),
        )
    )
    return rows[: max(1, int(limit))]


def compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep sweep output bounded while retaining parity decision metrics."""

    compact_modes: dict[str, Any] = {}
    for mode, values in dict(report.get("modes") or {}).items():
        joint_summaries = values.get("joint_error_deg_by_side_joint") or {}
        compact_modes[str(mode)] = {
            key: values.get(key)
            for key in (
                "evaluated_references",
                "hold_count",
                "status_counts",
                "pose_lift_applied_count",
                "pose_lift_selected_scale_counts",
                "adaptive_posture_selection_counts",
                "adaptive_posture_reason_counts",
                "adaptive_posture_failed_check_counts",
                "adaptive_posture_guarded_decision_count",
                "adaptive_posture_guarded_failed_check_counts",
                "adaptive_posture_guarded_route_counts",
                "max_joint_error_deg",
                "requested_to_admitted_translation_m",
                "requested_to_admitted_rotation_deg",
                "wrist_q4_sum_or_difference_error_deg",
                "admitted_max_step_deg",
                "admitted_max_velocity_deg_s",
                "admitted_max_acceleration_deg_s2",
                "admitted_max_jerk_deg_s3",
                "kinematic_dt_skipped_count",
                "motion_constraint_certificate",
            )
        }
        singularity = dict(values.get("singularity_diagnostics") or {})
        admitted_singularity = dict(
            dict(singularity.get("streams") or {}).get("admitted") or {}
        )
        direction_counts = dict(singularity.get("direction_counts") or {})
        wrist = dict(values.get("wrist_branch_continuity") or {})
        compact_modes[str(mode)]["singularity_safety"] = {
            "available": bool(singularity.get("available", False)),
            "admitted_by_side": {
                side: {
                    "sigma_min": dict(
                        dict(admitted_singularity.get(side) or {}).get("sigma_min")
                        or {}
                    ),
                    "guarded_exposure": dict(
                        dict(admitted_singularity.get(side) or {}).get(
                            "guarded_exposure"
                        )
                        or {}
                    ),
                    "condition_number": dict(
                        dict(admitted_singularity.get(side) or {}).get(
                            "condition_number"
                        )
                        or {}
                    ),
                    "min_abs_q3_or_q5_deg": dict(
                        dict(admitted_singularity.get(side) or {}).get(
                            "min_abs_q3_or_q5_deg"
                        )
                        or {}
                    ),
                }
                for side in ("left", "right")
            },
            "direction_counts": direction_counts,
            "admitted_wrist": dict(wrist.get("admitted") or {}),
        }
        task_conditioning = dict(
            values.get("task_conditioning_diagnostics") or {}
        )
        compact_modes[str(mode)]["task_conditioning"] = {
            "available": bool(task_conditioning.get("available", False)),
            "pre_solve_by_side": dict(
                dict(task_conditioning.get("streams") or {}).get("pre_solve")
                or {}
            ),
            "admitted_by_side": dict(
                dict(task_conditioning.get("streams") or {}).get("admitted")
                or {}
            ),
        }
        compact_modes[str(mode)]["constrained_task_feasibility"] = dict(
            values.get("constrained_task_feasibility") or {}
        )
        compact_modes[str(mode)]["right_q4_error_deg"] = (
            dict(joint_summaries.get("right") or {}).get("q4")
        )
        compact_modes[str(mode)]["right_q6_error_deg"] = (
            dict(joint_summaries.get("right") or {}).get("q6")
        )
    return {
        "episode_index": report.get("episode_index"),
        "frame_count": report.get("frame_count"),
        "modes": compact_modes,
    }


def compare_profile_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
    sigma_min_tolerance: float = 1e-6,
    singularity_guard_sigma: float = 0.04,
    translation_p95_slack_m: float = 2.5e-4,
    rotation_p95_slack_deg: float = 0.25,
) -> dict[str, Any]:
    """Require parity improvement without singularity/dynamics regression."""

    if int(baseline.get("episode_index", -1)) != int(
        candidate.get("episode_index", -2)
    ):
        raise ValueError("profile comparison requires the same episode.")
    tolerance = float(absolute_tolerance)
    sigma_tolerance = float(sigma_min_tolerance)
    guard_sigma = float(singularity_guard_sigma)
    if tolerance < 0.0 or sigma_tolerance < 0.0 or guard_sigma <= 0.0:
        raise ValueError("profile comparison tolerances must be non-negative.")

    checks: list[dict[str, Any]] = []

    def lower_or_equal(
        *,
        mode: str,
        metric: str,
        baseline_value: float | int | None,
        candidate_value: float | int | None,
        allowed_regression: float = 0.0,
    ) -> None:
        passed = bool(
            baseline_value is None
            and candidate_value is None
            or (
                baseline_value is not None
                and candidate_value is not None
                and math.isfinite(float(baseline_value))
                and math.isfinite(float(candidate_value))
                and float(candidate_value)
                <= float(baseline_value) + tolerance + float(allowed_regression)
            )
        )
        checks.append(
            {
                "mode": mode,
                "metric": metric,
                "requirement": "candidate<=baseline",
                "baseline": baseline_value,
                "candidate": candidate_value,
                "allowed_regression": float(allowed_regression),
                "passed": passed,
            }
        )

    baseline_modes = dict(baseline.get("modes") or {})
    candidate_modes = dict(candidate.get("modes") or {})
    if set(baseline_modes) != set(candidate_modes):
        raise ValueError("profile comparison requires identical evaluator modes.")
    for mode in sorted(baseline_modes):
        base = dict(baseline_modes[mode])
        cand = dict(candidate_modes[mode])
        lower_or_equal(
            mode=mode,
            metric="hold_count",
            baseline_value=base.get("hold_count"),
            candidate_value=cand.get("hold_count"),
        )
        for metric in (
            "max_joint_error_deg",
            "requested_to_admitted_translation_m",
            "requested_to_admitted_rotation_deg",
        ):
            base_summary = dict(base.get(metric) or {})
            cand_summary = dict(cand.get(metric) or {})
            for statistic in ("mean", "p95"):
                metric_slack = 0.0
                if statistic == "p95" and metric.endswith("translation_m"):
                    metric_slack = float(translation_p95_slack_m)
                elif statistic == "p95" and metric.endswith("rotation_deg"):
                    metric_slack = float(rotation_p95_slack_deg)
                lower_or_equal(
                    mode=mode,
                    metric=f"{metric}.{statistic}",
                    baseline_value=base_summary.get(statistic),
                    candidate_value=cand_summary.get(statistic),
                    allowed_regression=metric_slack,
                )
        for metric in (
            "admitted_max_step_deg",
            "admitted_max_velocity_deg_s",
            "admitted_max_acceleration_deg_s2",
            "admitted_max_jerk_deg_s3",
        ):
            lower_or_equal(
                mode=mode,
                metric=f"{metric}.max",
                baseline_value=dict(base.get(metric) or {}).get("max"),
                candidate_value=dict(cand.get(metric) or {}).get("max"),
            )
        base_violations = dict(
            dict(base.get("motion_constraint_certificate") or {}).get(
                "violation_counts"
            )
            or {}
        )
        cand_violations = dict(
            dict(cand.get("motion_constraint_certificate") or {}).get(
                "violation_counts"
            )
            or {}
        )
        for metric in (
            "raw_velocity_side_frames",
            "admitted_velocity_side_frames",
            "raw_acceleration_side_frames",
            "admitted_acceleration_side_frames",
        ):
            lower_or_equal(
                mode=mode,
                metric=f"motion_constraint_certificate.{metric}",
                baseline_value=int(base_violations.get(metric) or 0),
                candidate_value=int(cand_violations.get(metric) or 0),
            )

        base_singularity = dict(base.get("singularity_diagnostics") or {})
        cand_singularity = dict(cand.get("singularity_diagnostics") or {})
        base_streams = dict(base_singularity.get("streams") or {})
        cand_streams = dict(cand_singularity.get("streams") or {})
        base_admitted = dict(base_streams.get("admitted") or {})
        cand_admitted = dict(cand_streams.get("admitted") or {})
        for side in ("left", "right"):
            base_side = dict(base_admitted.get(side) or {})
            cand_side = dict(cand_admitted.get(side) or {})
            base_sigma = dict(base_side.get("sigma_min") or {}).get("min")
            cand_sigma = dict(cand_side.get("sigma_min") or {}).get("min")
            guarded = bool(
                base_sigma is not None
                and cand_sigma is not None
                and min(float(base_sigma), float(cand_sigma)) < guard_sigma
            )
            sigma_passed = bool(
                base_sigma is not None
                and cand_sigma is not None
                and math.isfinite(float(base_sigma))
                and math.isfinite(float(cand_sigma))
                and (
                    not guarded
                    or float(cand_sigma) + sigma_tolerance >= float(base_sigma)
                )
            )
            checks.append(
                {
                    "mode": mode,
                    "metric": f"admitted.{side}.sigma_min.min",
                    "requirement": (
                        "candidate>=baseline_in_guarded_band"
                    ),
                    "baseline": base_sigma,
                    "candidate": cand_sigma,
                    "guarded": guarded,
                    "passed": sigma_passed,
                }
            )
            base_condition = dict(base_side.get("condition_number") or {})
            cand_condition = dict(cand_side.get("condition_number") or {})
            if guarded:
                lower_or_equal(
                    mode=mode,
                    metric=f"admitted.{side}.condition_number.max",
                    baseline_value=base_condition.get("max"),
                    candidate_value=cand_condition.get("max"),
                )
            nonfinite = int(cand_condition.get("nonfinite_count") or 0)
            checks.append(
                {
                    "mode": mode,
                    "metric": f"admitted.{side}.condition_number.nonfinite_count",
                    "requirement": "candidate==0",
                    "baseline": int(base_condition.get("nonfinite_count") or 0),
                    "candidate": nonfinite,
                    "passed": nonfinite == 0,
                }
            )
        for side in ("left", "right"):
            base_wrist = dict(
                dict(base.get("wrist_branch_continuity") or {})
                .get("admitted", {})
                .get(side, {})
            )
            cand_wrist = dict(
                dict(cand.get("wrist_branch_continuity") or {})
                .get("admitted", {})
                .get(side, {})
            )
            lower_or_equal(
                mode=mode,
                metric=f"admitted.{side}.branch_jump_count",
                baseline_value=base_wrist.get("branch_jump_count", 0),
                candidate_value=cand_wrist.get("branch_jump_count", 0),
            )

    failed = [check for check in checks if not bool(check["passed"])]
    return {
        "schema": "teleop_runtime.rm65_pose_lift_profile_comparison/v1",
        "episode_index": int(candidate["episode_index"]),
        "accepted": not failed,
        "contract": (
            "parity_mean_p95_nonworse_singularity_dynamics_motion_certificate_"
            "and_branch_nonregression"
        ),
        "absolute_tolerance": tolerance,
        "sigma_min_tolerance": sigma_tolerance,
        "singularity_guard_sigma": guard_sigma,
        "translation_p95_slack_m": float(translation_p95_slack_m),
        "rotation_p95_slack_deg": float(rotation_p95_slack_deg),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


def assess_sigma_escape_research_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    singularity_guard_sigma: float = 0.04,
    minimum_sigma_improvement_ratio: float = 1e-3,
    minimum_sigma_improvement_absolute: float = 1e-5,
    max_joint_error_mean_regression_deg: float = 0.15,
    max_joint_error_p95_regression_deg: float = 0.15,
    max_translation_mean_regression_m: float = 2e-5,
    max_translation_p95_regression_m: float = 2e-5,
    max_rotation_mean_regression_deg: float = 0.005,
    max_rotation_p95_regression_deg: float = 0.005,
    max_jerk_regression_ratio: float = 1e-3,
    max_guarded_sample_count_regression: int = 0,
    max_deep_sample_count_regression: int = 0,
    max_mean_guard_deficit_regression: float = 0.0,
) -> dict[str, Any]:
    """Assess an offline singularity tradeoff without weakening production gates.

    This contract is intentionally separate from :func:`compare_profile_reports`.
    It accepts a bounded loss of teacher-qpos/pose parity only when every
    guarded admitted arm gains a material normalized Jacobian margin, no hard
    motion certificate is violated, and hold/branch behavior is non-regressive.
    It is research evidence, never hardware admission authority.
    """

    if int(baseline.get("episode_index", -1)) != int(
        candidate.get("episode_index", -2)
    ):
        raise ValueError("research comparison requires the same episode.")
    thresholds = {
        "singularity_guard_sigma": float(singularity_guard_sigma),
        "minimum_sigma_improvement_ratio": float(minimum_sigma_improvement_ratio),
        "minimum_sigma_improvement_absolute": float(
            minimum_sigma_improvement_absolute
        ),
        "max_joint_error_mean_regression_deg": float(
            max_joint_error_mean_regression_deg
        ),
        "max_joint_error_p95_regression_deg": float(
            max_joint_error_p95_regression_deg
        ),
        "max_translation_mean_regression_m": float(
            max_translation_mean_regression_m
        ),
        "max_translation_p95_regression_m": float(
            max_translation_p95_regression_m
        ),
        "max_rotation_mean_regression_deg": float(
            max_rotation_mean_regression_deg
        ),
        "max_rotation_p95_regression_deg": float(
            max_rotation_p95_regression_deg
        ),
        "max_jerk_regression_ratio": float(max_jerk_regression_ratio),
        "max_guarded_sample_count_regression": float(
            max_guarded_sample_count_regression
        ),
        "max_deep_sample_count_regression": float(
            max_deep_sample_count_regression
        ),
        "max_mean_guard_deficit_regression": float(
            max_mean_guard_deficit_regression
        ),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ValueError("research comparison thresholds must be finite and non-negative.")
    if thresholds["singularity_guard_sigma"] <= 0.0:
        raise ValueError("singularity_guard_sigma must be positive.")

    baseline_modes = dict(baseline.get("modes") or {})
    candidate_modes = dict(candidate.get("modes") or {})
    if set(baseline_modes) != set(candidate_modes):
        raise ValueError("research comparison requires identical evaluator modes.")

    checks: list[dict[str, Any]] = []

    def append_bounded_regression(
        *,
        mode: str,
        metric: str,
        baseline_value: Any,
        candidate_value: Any,
        allowance: float,
    ) -> None:
        passed = bool(
            baseline_value is not None
            and candidate_value is not None
            and math.isfinite(float(baseline_value))
            and math.isfinite(float(candidate_value))
            and float(candidate_value) <= float(baseline_value) + float(allowance)
        )
        checks.append(
            {
                "mode": mode,
                "metric": metric,
                "requirement": "candidate<=baseline+research_allowance",
                "baseline": baseline_value,
                "candidate": candidate_value,
                "allowed_regression": float(allowance),
                "passed": passed,
            }
        )

    guarded_side_count = 0
    for mode in sorted(baseline_modes):
        base = dict(baseline_modes[mode])
        cand = dict(candidate_modes[mode])
        append_bounded_regression(
            mode=mode,
            metric="hold_count",
            baseline_value=base.get("hold_count"),
            candidate_value=cand.get("hold_count"),
            allowance=0.0,
        )
        parity_limits = (
            (
                "max_joint_error_deg",
                "mean",
                thresholds["max_joint_error_mean_regression_deg"],
            ),
            (
                "max_joint_error_deg",
                "p95",
                thresholds["max_joint_error_p95_regression_deg"],
            ),
            (
                "requested_to_admitted_translation_m",
                "mean",
                thresholds["max_translation_mean_regression_m"],
            ),
            (
                "requested_to_admitted_translation_m",
                "p95",
                thresholds["max_translation_p95_regression_m"],
            ),
            (
                "requested_to_admitted_rotation_deg",
                "mean",
                thresholds["max_rotation_mean_regression_deg"],
            ),
            (
                "requested_to_admitted_rotation_deg",
                "p95",
                thresholds["max_rotation_p95_regression_deg"],
            ),
        )
        for metric, statistic, allowance in parity_limits:
            append_bounded_regression(
                mode=mode,
                metric=f"{metric}.{statistic}",
                baseline_value=dict(base.get(metric) or {}).get(statistic),
                candidate_value=dict(cand.get(metric) or {}).get(statistic),
                allowance=allowance,
            )

        base_jerk = dict(base.get("admitted_max_jerk_deg_s3") or {}).get("max")
        jerk_allowance = (
            0.0
            if base_jerk is None
            else abs(float(base_jerk)) * thresholds["max_jerk_regression_ratio"]
        )
        append_bounded_regression(
            mode=mode,
            metric="admitted_max_jerk_deg_s3.max",
            baseline_value=base_jerk,
            candidate_value=dict(cand.get("admitted_max_jerk_deg_s3") or {}).get(
                "max"
            ),
            allowance=jerk_allowance,
        )

        cand_violations = dict(
            dict(cand.get("motion_constraint_certificate") or {}).get(
                "violation_counts"
            )
            or {}
        )
        for metric in (
            "raw_velocity_side_frames",
            "admitted_velocity_side_frames",
            "raw_acceleration_side_frames",
            "admitted_acceleration_side_frames",
        ):
            value = int(cand_violations.get(metric) or 0)
            checks.append(
                {
                    "mode": mode,
                    "metric": f"motion_constraint_certificate.{metric}",
                    "requirement": "candidate==0",
                    "baseline": None,
                    "candidate": value,
                    "passed": value == 0,
                }
            )

        base_admitted = dict(
            dict(base.get("singularity_diagnostics") or {})
            .get("streams", {})
            .get("admitted", {})
        )
        cand_admitted = dict(
            dict(cand.get("singularity_diagnostics") or {})
            .get("streams", {})
            .get("admitted", {})
        )
        for side in ("left", "right"):
            base_side = dict(base_admitted.get(side) or {})
            cand_side = dict(cand_admitted.get(side) or {})
            base_sigma = dict(base_side.get("sigma_min") or {}).get("min")
            cand_sigma = dict(cand_side.get("sigma_min") or {}).get("min")
            guarded = bool(
                base_sigma is not None
                and math.isfinite(float(base_sigma))
                and float(base_sigma) < thresholds["singularity_guard_sigma"]
            )
            if guarded:
                guarded_side_count += 1
                required_gain = max(
                    thresholds["minimum_sigma_improvement_absolute"],
                    abs(float(base_sigma))
                    * thresholds["minimum_sigma_improvement_ratio"],
                )
                sigma_passed = bool(
                    cand_sigma is not None
                    and math.isfinite(float(cand_sigma))
                    and float(cand_sigma) >= float(base_sigma) + required_gain
                )
                checks.append(
                    {
                        "mode": mode,
                        "metric": f"admitted.{side}.sigma_min.min",
                        "requirement": "material_improvement_in_guarded_band",
                        "baseline": base_sigma,
                        "candidate": cand_sigma,
                        "required_gain": required_gain,
                        "passed": sigma_passed,
                    }
                )
                base_exposure = dict(base_side.get("guarded_exposure") or {})
                cand_exposure = dict(cand_side.get("guarded_exposure") or {})
                base_sample_count = base_exposure.get("sample_count")
                cand_sample_count = cand_exposure.get("sample_count")
                checks.append(
                    {
                        "mode": mode,
                        "metric": f"admitted.{side}.guarded_exposure.sample_count",
                        "requirement": "candidate==baseline",
                        "baseline": base_sample_count,
                        "candidate": cand_sample_count,
                        "passed": bool(
                            base_sample_count is not None
                            and cand_sample_count is not None
                            and int(cand_sample_count) == int(base_sample_count)
                        ),
                    }
                )
                for metric, allowance in (
                    (
                        "below_guard_count",
                        thresholds["max_guarded_sample_count_regression"],
                    ),
                    (
                        "below_0p01_count",
                        thresholds["max_deep_sample_count_regression"],
                    ),
                    (
                        "below_0p001_count",
                        thresholds["max_deep_sample_count_regression"],
                    ),
                    (
                        "mean_guard_deficit",
                        thresholds["max_mean_guard_deficit_regression"],
                    ),
                ):
                    append_bounded_regression(
                        mode=mode,
                        metric=f"admitted.{side}.guarded_exposure.{metric}",
                        baseline_value=base_exposure.get(metric),
                        candidate_value=cand_exposure.get(metric),
                        allowance=allowance,
                    )
            cand_condition = dict(cand_side.get("condition_number") or {})
            nonfinite = int(cand_condition.get("nonfinite_count") or 0)
            checks.append(
                {
                    "mode": mode,
                    "metric": f"admitted.{side}.condition_number.nonfinite_count",
                    "requirement": "candidate==0",
                    "baseline": None,
                    "candidate": nonfinite,
                    "passed": nonfinite == 0,
                }
            )
            base_wrist = dict(
                dict(base.get("wrist_branch_continuity") or {})
                .get("admitted", {})
                .get(side, {})
            )
            cand_wrist = dict(
                dict(cand.get("wrist_branch_continuity") or {})
                .get("admitted", {})
                .get(side, {})
            )
            append_bounded_regression(
                mode=mode,
                metric=f"admitted.{side}.branch_jump_count",
                baseline_value=base_wrist.get("branch_jump_count", 0),
                candidate_value=cand_wrist.get("branch_jump_count", 0),
                allowance=0.0,
            )

    checks.append(
        {
            "mode": "all",
            "metric": "guarded_side_count",
            "requirement": "candidate_evaluation_contains_guarded_evidence",
            "baseline": None,
            "candidate": guarded_side_count,
            "passed": guarded_side_count > 0,
        }
    )
    failed = [check for check in checks if not bool(check["passed"])]
    return {
        "schema": "teleop_runtime.rm65_pose_lift_sigma_escape_research/v1",
        "episode_index": int(candidate["episode_index"]),
        "accepted": not failed,
        "authority": "offline_research_only_not_hardware_admission",
        "production_comparison_unchanged": True,
        "guarded_side_count": guarded_side_count,
        "thresholds": thresholds,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-motion",
        action="store_true",
        required=True,
        help="Required acknowledgement: this evaluator must remain offline and dispatch-free.",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help=(
            "Evaluate every committed episode in --dataset-root. Offline-only; "
            "mutually exclusive with --episode and ranking modes."
        ),
    )
    parser.add_argument("--left-tool-profile", required=True)
    parser.add_argument("--right-tool-profile", required=True)
    parser.add_argument("--scene-profile", default=DEFAULT_COLLISION_SCENE)
    parser.add_argument(
        "--pose-lift-profile",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        default=RM65_POSE_LIFT_V1_BASELINE_NAME,
    )
    parser.add_argument(
        "--compare-baseline-profile",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        default=None,
        help=(
            "Evaluate a second rollback profile on the same inputs and emit a "
            "non-regression verdict covering parity, Jacobian margin, dynamics, "
            "holds, and branch jumps."
        ),
    )
    parser.add_argument(
        "--comparison-contract",
        choices=("production", "sigma_escape_research"),
        default="production",
        help=(
            "Keep the production non-regression gate by default. The separate "
            "sigma_escape_research contract permits tightly bounded parity "
            "tradeoffs but remains offline-only and has no hardware authority."
        ),
    )
    parser.add_argument(
        "--comparison-sigma-min-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--comparison-singularity-guard-sigma",
        type=float,
        default=0.04,
    )
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--reference-hz", type=float, default=60.0)
    parser.add_argument(
        "--jacobian-characteristic-length-m",
        type=float,
        default=DEFAULT_RM65_JACOBIAN_CHARACTERISTIC_LENGTH_M,
        help=(
            "Characteristic length used only to normalize offline Jacobian "
            "linear rows; keep fixed when comparing profiles."
        ),
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=None,
        help="Offline-only override for the IK continuity/posture cost sweep.",
    )
    parser.add_argument(
        "--offline-sigma-escape-cost",
        type=float,
        default=None,
        help=(
            "Offline research only: override the selected profile's normalized "
            "sigma escape soft-objective cost without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-sigma-escape-rate-per-sec",
        type=float,
        default=None,
        help=(
            "Offline research only: override the selected profile's bounded "
            "sigma target rate without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-structural-branch-margin-deg",
        type=float,
        default=None,
        help=(
            "Offline research only: override the V6 one-sided q3/q5 zero "
            "margin in degrees without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-structural-branch-guard-sigma",
        type=float,
        default=None,
        help=(
            "Offline research only: override the V6 structural branch guard "
            "sigma threshold without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-structural-branch-release-sigma",
        type=float,
        default=None,
        help=(
            "Offline research only: override the V6 structural branch release "
            "threshold without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-structural-branch-barrier-gain",
        type=float,
        default=None,
        help=(
            "Offline research only: override a barrier-mode structural "
            "profile's CBF gain without changing its registry/default."
        ),
    )
    parser.add_argument(
        "--offline-structural-branch-prediction-horizon-sec",
        type=float,
        default=None,
        help=(
            "Offline research only: override a predictive structural "
            "profile's previous-velocity horizon without changing its "
            "registry/default."
        ),
    )
    parser.add_argument(
        "--alternate-posture-cost",
        type=float,
        default=None,
        help=(
            "Offline-only: solve a second fully constrained candidate from the "
            "same admitted anchor and apply guarded Pareto selection."
        ),
    )
    parser.add_argument(
        "--alternate-pose-lift-profile",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        default=None,
        help=(
            "Offline-only horizon research: use a second registered Pose Lift "
            "as the alternate certified rollout. This does not change deploy "
            "or register a production selector."
        ),
    )
    parser.add_argument(
        "--offline-fixed-solve-iterations",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Offline research only: force min=max=N and disable wall-clock "
            "adaptive iteration stopping. Required for finite-horizon rollouts "
            "so observer workload cannot change the candidate trajectory."
        ),
    )
    parser.add_argument("--adaptive-posture-guard-sigma", type=float, default=0.04)
    parser.add_argument("--adaptive-posture-sigma-tolerance", type=float, default=0.0)
    parser.add_argument("--adaptive-posture-sigma-improvement", type=float, default=1e-4)
    parser.add_argument("--adaptive-posture-pose-score-slack", type=float, default=0.05)
    parser.add_argument(
        "--adaptive-posture-pose-score-improvement",
        type=float,
        default=0.05,
        help=(
            "Minimum material reduction in the normalized worst pose score "
            "before an alternate horizon can be selected for pose benefit."
        ),
    )
    parser.add_argument("--adaptive-posture-max-step-slack-deg", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-posture-dynamic-relative-slack",
        type=float,
        default=0.0,
        help=(
            "Offline horizon research only: bounded relative allowance for "
            "candidate velocity/acceleration/jerk maxima and means. Default "
            "zero preserves strict dynamic non-regression."
        ),
    )
    parser.add_argument(
        "--adaptive-posture-horizon-steps",
        type=int,
        default=1,
        help=(
            "Offline-only bounded look-ahead. Values above one evaluate each "
            "candidate over future recorded pose references before selecting "
            "the current step."
        ),
    )
    parser.add_argument(
        "--adaptive-posture-lock-steps",
        type=int,
        default=1,
        help=(
            "Offline-only number of certified steps kept under one plan id; "
            "an invalid committed candidate cancels the lock immediately."
        ),
    )
    parser.add_argument(
        "--adaptive-posture-terminal-shadow-delta-deg",
        type=float,
        default=0.5,
        help=(
            "Offline-only maximum joint-space distance between a candidate "
            "horizon endpoint and the independent V1 shadow endpoint."
        ),
    )
    parser.add_argument(
        "--adaptive-posture-terminal-mode",
        choices=("joint_shadow", "task_space_tube"),
        default="joint_shadow",
        help=(
            "Offline horizon terminal contract. task_space_tube permits a "
            "different legal joint lift when its terminal FK rejoins the "
            "requested task-space tube."
        ),
    )
    parser.add_argument(
        "--adaptive-posture-terminal-translation-m",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--adaptive-posture-terminal-rotation-deg",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--rank-singular-episodes",
        type=int,
        default=0,
        metavar="N",
        help="Only rank all episodes by saved q3/q5 singular proximity and print top N.",
    )
    parser.add_argument(
        "--rank-jacobian-episodes",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Only rank saved accepted qref by the normalized RM65 Jacobian used "
            "by Pose Lift. This is observational and performs no IK or dispatch."
        ),
    )
    parser.add_argument(
        "--include-frames",
        action="store_true",
        help="Include per-reference rows; default output is bounded aggregate metrics.",
    )
    parser.add_argument(
        "--include-constrained-feasibility",
        action="store_true",
        help=(
            "Run additional detached QPs for strict directional alpha and "
            "best constrained residual. Offline-only and intentionally "
            "disabled by default because it adds four observer QPs/reference."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print bounded sweep metrics instead of full per-joint diagnostics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in (
        "posture_cost",
        "alternate_posture_cost",
        "offline_sigma_escape_cost",
        "offline_sigma_escape_rate_per_sec",
        "offline_structural_branch_margin_deg",
        "offline_structural_branch_guard_sigma",
        "offline_structural_branch_release_sigma",
        "offline_structural_branch_barrier_gain",
        "offline_structural_branch_prediction_horizon_sec",
        "adaptive_posture_guard_sigma",
        "adaptive_posture_sigma_tolerance",
        "adaptive_posture_sigma_improvement",
        "adaptive_posture_pose_score_slack",
        "adaptive_posture_pose_score_improvement",
        "adaptive_posture_max_step_slack_deg",
        "adaptive_posture_dynamic_relative_slack",
        "adaptive_posture_terminal_shadow_delta_deg",
        "adaptive_posture_terminal_translation_m",
        "adaptive_posture_terminal_rotation_deg",
    ):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in (
        "adaptive_posture_horizon_steps",
        "adaptive_posture_lock_steps",
    ):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if int(args.offline_fixed_solve_iterations) < 0:
        raise SystemExit("--offline-fixed-solve-iterations must be non-negative")
    if (
        args.offline_sigma_escape_cost is not None
        or args.offline_sigma_escape_rate_per_sec is not None
    ) and resolve_rm65_pose_lift_profile(
        str(args.pose_lift_profile)
    ).qp_sigma_escape_cost <= 0.0:
        raise SystemExit(
            "offline sigma escape overrides require a selected profile that "
            "already owns the sigma escape algorithm"
        )
    if (
        args.offline_structural_branch_margin_deg is not None
        or args.offline_structural_branch_guard_sigma is not None
        or args.offline_structural_branch_release_sigma is not None
        or args.offline_structural_branch_barrier_gain is not None
        or args.offline_structural_branch_prediction_horizon_sec is not None
    ) and resolve_rm65_pose_lift_profile(
        str(args.pose_lift_profile)
    ).qp_structural_branch_margin_deg <= 0.0:
        raise SystemExit(
            "offline structural branch overrides require a selected profile "
            "that already owns the structural branch algorithm"
        )
    if (
        args.offline_structural_branch_barrier_gain is not None
        or args.offline_structural_branch_prediction_horizon_sec is not None
    ) and resolve_rm65_pose_lift_profile(
        str(args.pose_lift_profile)
    ).qp_structural_branch_constraint_mode != "barrier":
        raise SystemExit(
            "offline structural barrier overrides require a selected "
            "barrier-mode profile"
        )
    if args.offline_structural_branch_barrier_gain is not None and float(
        args.offline_structural_branch_barrier_gain
    ) <= 0.0:
        raise SystemExit(
            "--offline-structural-branch-barrier-gain must be positive"
        )
    if (
        int(args.adaptive_posture_horizon_steps) > 1
        and int(args.offline_fixed_solve_iterations) < 1
    ):
        raise SystemExit(
            "finite-horizon rollout requires --offline-fixed-solve-iterations N; "
            "wall-clock adaptive iteration stopping is observer-dependent"
        )
    if (
        int(args.adaptive_posture_horizon_steps) > 1
        and int(args.adaptive_posture_lock_steps)
        != int(args.adaptive_posture_horizon_steps)
    ):
        raise SystemExit(
            "finite-horizon rollout requires --adaptive-posture-lock-steps "
            "to equal --adaptive-posture-horizon-steps so the certified path "
            "is executed through its terminal shadow-rejoin check"
        )
    group_args = argparse.Namespace(
        replay_dataset=str(Path(args.dataset_root).expanduser().resolve()),
        replay_parquet="",
        replay_episode=0,
        replay_action_column="action",
        replay_all_episodes=False,
        active_sides=("left", "right"),
    )
    ranking_requested = bool(
        int(args.rank_singular_episodes) > 0
        or int(args.rank_jacobian_episodes) > 0
    )
    if bool(args.all_episodes) and (args.episode or ranking_requested):
        raise SystemExit(
            "--all-episodes is mutually exclusive with --episode and ranking modes"
        )
    episode_indices = (
        read_replay_dataset_episode_indices(args.dataset_root)
        if (bool(args.all_episodes) or ranking_requested)
        else list(args.episode or [])
    )
    if not episode_indices:
        raise SystemExit(
            "provide --episode, --all-episodes, or a ranking mode"
        )
    groups = [
        build_replay_group_for_episode(
            group_args,
            episode_index=int(episode),
            group_spec=ReplayGroupSpec(
                key="parity",
                start_index=0,
                end_index=-1,
                episode_range=str(episode),
            ),
            balanced_episode_count=1,
            require_arm_joint_targets=True,
        )
        for episode in episode_indices
    ]
    if int(args.rank_singular_episodes) > 0:
        print(
            json.dumps(
                {
                    "schema": "teleop_runtime.rm65_pose_lift_singular_rank/v1",
                    "device_access": False,
                    "dispatch": False,
                    "episodes_scanned": len(groups),
                    "ranking": rank_singular_groups(
                        groups,
                        limit=int(args.rank_singular_episodes),
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    endpoint_to_tool = {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(getattr(args, f"{side}_tool_profile"), side=side)
        )
        for side in ("left", "right")
    }

    def build_planner(
        profile_name: str,
        posture_cost: float | None,
    ) -> RM65PositionCollisionPlanner:
        fixed_iterations = int(args.offline_fixed_solve_iterations)
        resolved_profile = resolve_rm65_pose_lift_profile(profile_name)
        profile_overrides: dict[str, float] = {}
        apply_sigma_overrides = str(profile_name) == str(args.pose_lift_profile)
        if apply_sigma_overrides and args.offline_sigma_escape_cost is not None:
            profile_overrides["qp_sigma_escape_cost"] = float(
                args.offline_sigma_escape_cost
            )
        if (
            apply_sigma_overrides
            and args.offline_sigma_escape_rate_per_sec is not None
        ):
            profile_overrides["qp_sigma_escape_rate_per_sec"] = float(
                args.offline_sigma_escape_rate_per_sec
            )
        if (
            apply_sigma_overrides
            and args.offline_structural_branch_margin_deg is not None
        ):
            profile_overrides["qp_structural_branch_margin_deg"] = float(
                args.offline_structural_branch_margin_deg
            )
        if (
            apply_sigma_overrides
            and args.offline_structural_branch_guard_sigma is not None
        ):
            profile_overrides["qp_structural_branch_guard_sigma"] = float(
                args.offline_structural_branch_guard_sigma
            )
        if (
            apply_sigma_overrides
            and args.offline_structural_branch_release_sigma is not None
        ):
            profile_overrides["qp_structural_branch_release_sigma"] = float(
                args.offline_structural_branch_release_sigma
            )
        if (
            apply_sigma_overrides
            and args.offline_structural_branch_barrier_gain is not None
        ):
            profile_overrides["qp_structural_branch_barrier_gain"] = float(
                args.offline_structural_branch_barrier_gain
            )
        if (
            apply_sigma_overrides
            and args.offline_structural_branch_prediction_horizon_sec
            is not None
        ):
            profile_overrides[
                "qp_structural_branch_prediction_horizon_sec"
            ] = float(
                args.offline_structural_branch_prediction_horizon_sec
            )
        effective_profile = (
            resolved_profile
            if not profile_overrides
            else replace(resolved_profile, **profile_overrides)
        )
        return RM65PositionCollisionPlanner(
            RM65PositionCollisionPlannerConfig(
                reference_hz=float(args.reference_hz),
                scene_profile=str(args.scene_profile),
                max_joint_velocity_rad_s=REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
                **(
                    {}
                    if posture_cost is None
                    else {"posture_cost": float(posture_cost)}
                ),
                damping=REPLAY_POSITION_IK_DAMPING,
                solve_max_iterations=(
                    fixed_iterations
                    if fixed_iterations > 0
                    else REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS
                ),
                solve_min_iterations=(
                    fixed_iterations
                    if fixed_iterations > 0
                    else REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS
                ),
                solve_adaptive_iterations_enabled=fixed_iterations <= 0,
                solve_iteration_reserve_ms=REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
                pose_lift_profile=effective_profile,
                ik_tuning=build_rm65_pose_lift_ik_tuning(
                    position_control_profile="rm65_bimanual_position_v1",
                    pose_lift_profile=effective_profile,
                    orientation_weight=REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
                ),
            ),
            endpoint_to_tool_by_side=endpoint_to_tool,
        )

    def planner_factory_for(
        profile_name: str,
        *,
        adaptive: bool,
    ) -> Callable[[], Any]:
        def factory() -> Any:
            baseline = build_planner(profile_name, args.posture_cost)
            if not adaptive:
                return baseline
            alternate_profile_name = str(
                args.alternate_pose_lift_profile or profile_name
            )
            alternate_posture_cost = (
                args.posture_cost
                if args.alternate_posture_cost is None
                else float(args.alternate_posture_cost)
            )
            alternate = build_planner(
                alternate_profile_name,
                alternate_posture_cost,
            )
            shadow_baseline = build_planner(profile_name, args.posture_cost)
            finite_horizon = int(args.adaptive_posture_horizon_steps) > 1
            return _AdaptivePostureCandidatePlanner(
                baseline,
                alternate,
                shadow_baseline,
                guard_sigma=float(args.adaptive_posture_guard_sigma),
                sigma_tolerance=float(args.adaptive_posture_sigma_tolerance),
                sigma_improvement=float(args.adaptive_posture_sigma_improvement),
                pose_score_slack=float(args.adaptive_posture_pose_score_slack),
                pose_score_improvement=float(
                    args.adaptive_posture_pose_score_improvement
                ),
                max_step_slack_deg=float(args.adaptive_posture_max_step_slack_deg),
                dynamic_relative_slack=float(
                    args.adaptive_posture_dynamic_relative_slack
                ),
                characteristic_length_m=float(
                    args.jacobian_characteristic_length_m
                ),
                rollout_baseline=(
                    build_planner(profile_name, args.posture_cost)
                    if finite_horizon
                    else None
                ),
                rollout_alternate=(
                    build_planner(
                        alternate_profile_name,
                        alternate_posture_cost,
                    )
                    if finite_horizon
                    else None
                ),
                rollout_shadow_baseline=(
                    build_planner(profile_name, args.posture_cost)
                    if finite_horizon
                    else None
                ),
                horizon_steps=int(args.adaptive_posture_horizon_steps),
                lock_steps=int(args.adaptive_posture_lock_steps),
                terminal_shadow_delta_deg=float(
                    args.adaptive_posture_terminal_shadow_delta_deg
                ),
                terminal_mode=str(
                    args.adaptive_posture_terminal_mode
                ),
                terminal_translation_m=float(
                    args.adaptive_posture_terminal_translation_m
                ),
                terminal_rotation_deg=float(
                    args.adaptive_posture_terminal_rotation_deg
                ),
            )

        return factory

    if int(args.rank_jacobian_episodes) > 0:
        ranking_planner = build_planner(
            RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
            args.posture_cost,
        )
        print(
            json.dumps(
                {
                    "schema": (
                        "teleop_runtime.rm65_pose_lift_jacobian_singular_rank/v1"
                    ),
                    "device_access": False,
                    "dispatch": False,
                    "episodes_scanned": len(groups),
                    "profile": RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
                    "characteristic_length_m": float(
                        args.jacobian_characteristic_length_m
                    ),
                    "ranking": rank_jacobian_singular_groups(
                        groups,
                        planner=ranking_planner,
                        limit=int(args.rank_jacobian_episodes),
                        characteristic_length_m=float(
                            args.jacobian_characteristic_length_m
                        ),
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    reports = []
    adaptive_candidate_enabled = bool(
        args.alternate_posture_cost is not None
        or args.alternate_pose_lift_profile is not None
    )
    planner_factory = planner_factory_for(
        str(args.pose_lift_profile),
        adaptive=adaptive_candidate_enabled,
    )
    for group in groups:
        reports.append(
            evaluate_group(
                group,
                planner_factory=planner_factory,
                modes=tuple(args.mode or MODES),
                reference_hz=float(args.reference_hz),
                include_frames=bool(args.include_frames),
                include_constrained_feasibility=bool(
                    args.include_constrained_feasibility
                ),
                jacobian_characteristic_length_m=float(
                    args.jacobian_characteristic_length_m
                ),
            )
        )
    baseline_reports: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    if args.compare_baseline_profile is not None:
        baseline_factory = planner_factory_for(
            str(args.compare_baseline_profile),
            adaptive=False,
        )
        for group, candidate_report in zip(groups, reports, strict=True):
            baseline_report = evaluate_group(
                group,
                planner_factory=baseline_factory,
                modes=tuple(args.mode or MODES),
                reference_hz=float(args.reference_hz),
                include_frames=False,
                include_constrained_feasibility=bool(
                    args.include_constrained_feasibility
                ),
                jacobian_characteristic_length_m=float(
                    args.jacobian_characteristic_length_m
                ),
            )
            baseline_reports.append(baseline_report)
            if str(args.comparison_contract) == "sigma_escape_research":
                comparisons.append(
                    assess_sigma_escape_research_candidate(
                        baseline_report,
                        candidate_report,
                        singularity_guard_sigma=float(
                            args.comparison_singularity_guard_sigma
                        ),
                    )
                )
            else:
                comparisons.append(
                    compare_profile_reports(
                        baseline_report,
                        candidate_report,
                        sigma_min_tolerance=float(
                            args.comparison_sigma_min_tolerance
                        ),
                        singularity_guard_sigma=float(
                            args.comparison_singularity_guard_sigma
                        ),
                    )
                )
    payload = {
        "schema": "teleop_runtime.rm65_pose_lift_parity/v2",
        "device_access": False,
        "dispatch": False,
        "no_motion_acknowledged": bool(args.no_motion),
        "pose_lift_profile": str(args.pose_lift_profile),
        "compare_baseline_profile": args.compare_baseline_profile,
        "comparison_contract": str(args.comparison_contract),
        "posture_cost_override": args.posture_cost,
        "offline_sigma_escape_overrides": {
            "cost": args.offline_sigma_escape_cost,
            "rate_per_sec": args.offline_sigma_escape_rate_per_sec,
            "registry_profile_unchanged": True,
        },
        "offline_structural_branch_overrides": {
            "margin_deg": args.offline_structural_branch_margin_deg,
            "guard_sigma": args.offline_structural_branch_guard_sigma,
            "release_sigma": args.offline_structural_branch_release_sigma,
            "barrier_gain": args.offline_structural_branch_barrier_gain,
            "prediction_horizon_sec": (
                args.offline_structural_branch_prediction_horizon_sec
            ),
            "registry_profile_unchanged": True,
        },
        "alternate_posture_cost": args.alternate_posture_cost,
        "alternate_pose_lift_profile": args.alternate_pose_lift_profile,
        "offline_fixed_solve_iterations": (
            None
            if int(args.offline_fixed_solve_iterations) <= 0
            else int(args.offline_fixed_solve_iterations)
        ),
        "reference_hz": float(args.reference_hz),
        "reference_clock_contract": (
            "all_modes_resampled_to_deploy_reference_hz"
        ),
        "adaptive_posture_candidate_selection": (
            None
            if not adaptive_candidate_enabled
            else {
                "offline_only": True,
                "alternate_pose_lift_profile": (
                    args.alternate_pose_lift_profile
                ),
                "guard_sigma": float(args.adaptive_posture_guard_sigma),
                "sigma_tolerance": float(args.adaptive_posture_sigma_tolerance),
                "sigma_improvement": float(args.adaptive_posture_sigma_improvement),
                "pose_score_slack": float(args.adaptive_posture_pose_score_slack),
                "pose_score_improvement": float(
                    args.adaptive_posture_pose_score_improvement
                ),
                "max_step_slack_deg": float(
                    args.adaptive_posture_max_step_slack_deg
                ),
                "dynamic_relative_slack": float(
                    args.adaptive_posture_dynamic_relative_slack
                ),
                "horizon_steps": int(args.adaptive_posture_horizon_steps),
                "lock_steps": int(args.adaptive_posture_lock_steps),
                "terminal_shadow_delta_deg": float(
                    args.adaptive_posture_terminal_shadow_delta_deg
                ),
                "terminal_mode": str(
                    args.adaptive_posture_terminal_mode
                ),
                "terminal_translation_m": float(
                    args.adaptive_posture_terminal_translation_m
                ),
                "terminal_rotation_deg": float(
                    args.adaptive_posture_terminal_rotation_deg
                ),
            }
        ),
        "jacobian_characteristic_length_m": float(
            args.jacobian_characteristic_length_m
        ),
        "episodes": (
            [compact_report(report) for report in reports]
            if bool(args.compact)
            else reports
        ),
        "baseline_episodes": (
            [compact_report(report) for report in baseline_reports]
            if bool(args.compact)
            else baseline_reports
        ),
        "comparisons": comparisons,
        "all_comparisons_accepted": bool(comparisons) and all(
            bool(comparison["accepted"]) for comparison in comparisons
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
        print(output)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
