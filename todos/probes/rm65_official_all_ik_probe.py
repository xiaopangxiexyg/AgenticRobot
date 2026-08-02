"""Read-only RM65 official all-solution IK probe for recorded Pose/q_ref.

This file is intentionally a research probe under ``todos/``.  It initializes
only the vendor algorithm library with an empty robot handle, opens no device,
and owns no motion or dispatch authority.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import (
    rm_force_type_e,
    rm_inverse_kinematics_params_t,
    rm_robot_arm_model_e,
)
from hardware.rm65.Robotic_Arm.rm_robot_interface import Algo
from robot_motion.replay.trajectory import ReplayGroupSpec
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.position_collision import (
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
)
from robot_motion.rm65.tool_profiles import (
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.pose import interpolate_pose
from teleop_runtime.debug.evaluate_rm65_pose_lift_parity import (
    _pose_at,
    _runtime_references,
    _runtime_truth,
    _truth_at,
)
from teleop_runtime.replay.trajectory import (
    build_replay_group_for_episode,
    read_replay_dataset_episode_indices,
)

_SIDES = ("left", "right")
_LEFT_TOOL = "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
_RIGHT_TOOL = "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"


def _matrix_to_sdk_pose(matrix: np.ndarray) -> list[float]:
    pose = np.asarray(matrix, dtype=float).reshape(4, 4)
    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return [
        *pose[:3, 3].astype(float).tolist(),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def _sdk_pose_to_matrix(values: list[float]) -> np.ndarray:
    pose = np.asarray(values, dtype=float).reshape(7)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = Rotation.from_quat(
        [pose[4], pose[5], pose[6], pose[3]]
    ).as_matrix()
    return matrix


def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual_pose = np.asarray(actual, dtype=float).reshape(4, 4)
    target_pose = np.asarray(target, dtype=float).reshape(4, 4)
    translation_m = float(
        np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
    )
    rotation_deg = float(
        np.degrees(
            Rotation.from_matrix(
                actual_pose[:3, :3].T @ target_pose[:3, :3]
            ).magnitude()
        )
    )
    return translation_m, rotation_deg


def _normalize_candidate_near_seed(
    candidate_deg: np.ndarray,
    seed_deg: np.ndarray,
    *,
    limit_tolerance_deg: float = 0.002,
) -> np.ndarray | None:
    lower = np.degrees(np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float))
    upper = np.degrees(np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float))
    normalized = np.empty(6, dtype=float)
    for joint_index, raw in enumerate(np.asarray(candidate_deg, dtype=float).reshape(6)):
        equivalents = np.asarray(
            [float(raw) + 360.0 * offset for offset in range(-2, 3)],
            dtype=float,
        )
        valid = equivalents[
            (equivalents >= lower[joint_index] - float(limit_tolerance_deg))
            & (equivalents <= upper[joint_index] + float(limit_tolerance_deg))
        ]
        if valid.size == 0:
            return None
        selected = valid[int(np.argmin(np.abs(valid - seed_deg[joint_index])))]
        # The SDK and canonical URDF differ by a few 1e-4 degrees at exact
        # limits.  Treat that only as numerical boundary tolerance, clamp back
        # into the authoritative project interval, and re-check FK downstream.
        normalized[joint_index] = np.clip(
            selected,
            lower[joint_index],
            upper[joint_index],
        )
    return normalized


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def _build_group(dataset_root: Path, episode_index: int) -> Any:
    args = SimpleNamespace(
        replay_dataset=str(dataset_root),
        replay_parquet="",
        replay_action_column="action",
        replay_all_episodes=False,
        active_sides=_SIDES,
    )
    return build_replay_group_for_episode(
        args,
        episode_index=int(episode_index),
        group_spec=ReplayGroupSpec(
            key="official_all_ik_probe",
            start_index=0,
            end_index=-1,
            episode_range=str(episode_index),
        ),
        balanced_episode_count=1,
        require_arm_joint_targets=True,
    )


def _build_planner(
    endpoint_to_tool: dict[str, np.ndarray],
) -> RM65PositionCollisionPlanner:
    return RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(reference_hz=60.0),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )


def _solve_valid_candidates(
    *,
    algo: Algo,
    seed_rad: np.ndarray,
    target_endpoint: np.ndarray,
) -> tuple[list[np.ndarray], int, int, float]:
    seed_deg = np.degrees(np.asarray(seed_rad, dtype=float).reshape(6))
    params = rm_inverse_kinematics_params_t(
        seed_deg.tolist(),
        _matrix_to_sdk_pose(target_endpoint),
        0,
    )
    started_ns = time.perf_counter_ns()
    result = algo.rm_algo_inverse_kinematics_all(params)
    elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0
    candidates: list[np.ndarray] = []
    if int(result.result) == 0 and int(result.num) > 0:
        for candidate_index in range(int(result.num)):
            normalized = _normalize_candidate_near_seed(
                np.asarray(result.q_solve[candidate_index][:6], dtype=float),
                seed_deg,
            )
            if normalized is not None:
                candidates.append(np.radians(normalized))
    return candidates, int(result.result), int(result.num), float(elapsed_us)


def _run_closed_loop(
    *,
    group: Any,
    algo: Algo,
    planner: RM65PositionCollisionPlanner,
    endpoint_to_tool: dict[str, np.ndarray],
    reference_hz: float = 60.0,
) -> dict[str, Any]:
    dt = 1.0 / float(reference_hz)
    current = _truth_at(group, 0)
    previous_velocity = {
        side: np.zeros(6, dtype=float)
        for side in _SIDES
    }
    previous_truth = _truth_at(group, 0)
    previous_truth_velocity = {
        side: np.zeros(6, dtype=float)
        for side in _SIDES
    }
    previous_q3_q5_sign = {
        side: (
            int(np.sign(float(current[side][2]))),
            int(np.sign(float(current[side][4]))),
        )
        for side in _SIDES
    }
    previous_truth_q3_q5_sign = dict(previous_q3_q5_sign)
    side_metrics: dict[str, dict[str, Any]] = {
        side: {
            "failures": 0,
            "latency_us": [],
            "joint_l2_deg": [],
            "joint_max_deg": [],
            "sigma_min": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "truth_acceleration_max_deg_s2": [],
            "acceleration_error_max_deg_s2": [],
            "branch_sign_changes": 0,
            "truth_branch_sign_changes": 0,
            "branch_sign_mismatch_frames": 0,
            "first_joint_max_gt_1deg": None,
            "min_sigma_reference_sequence": None,
            "min_sigma": math.inf,
            "events": [],
        }
        for side in _SIDES
    }
    references = [
        reference
        for reference in _runtime_references(group, reference_hz)
        if int(reference.sequence) > 0
    ]
    for reference in references:
        truth = _runtime_truth(group, reference)
        targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        next_q: dict[str, np.ndarray] = {}
        next_velocity: dict[str, np.ndarray] = {}
        next_truth_velocity: dict[str, np.ndarray] = {}
        for side in _SIDES:
            candidates, result_code, raw_count, elapsed_us = _solve_valid_candidates(
                algo=algo,
                seed_rad=current[side],
                target_endpoint=targets[side],
            )
            metrics = side_metrics[side]
            metrics["latency_us"].append(elapsed_us)
            if result_code != 0 or not candidates:
                metrics["failures"] += 1
                if len(metrics["events"]) < 32:
                    metrics["events"].append(
                        {
                            "type": "no_project_valid_candidate",
                            "reference_sequence": int(reference.sequence),
                            "left_source_index": int(
                                reference.left_source_index
                            ),
                            "right_source_index": int(
                                reference.right_source_index
                            ),
                            "interpolation_alpha": float(
                                reference.interpolation_alpha
                            ),
                            "sdk_result": int(result_code),
                            "sdk_raw_candidate_count": int(raw_count),
                            "current_q_deg": np.degrees(current[side]).tolist(),
                            "truth_q_deg": np.degrees(truth[side]).tolist(),
                        }
                    )
                next_q[side] = current[side].copy()
                next_velocity[side] = np.zeros(6, dtype=float)
                continue
            # This baseline intentionally uses only causal continuity.  It does
            # not see saved q_ref, future poses, collision, or singularity.
            chosen = min(
                candidates,
                key=lambda value: float(
                    np.linalg.norm(
                        np.asarray(value, dtype=float) - current[side]
                    )
                ),
            )
            velocity = (chosen - current[side]) / dt
            acceleration = (velocity - previous_velocity[side]) / dt
            truth_velocity = (truth[side] - previous_truth[side]) / dt
            truth_acceleration = (
                truth_velocity - previous_truth_velocity[side]
            ) / dt
            next_truth_velocity[side] = truth_velocity
            next_q[side] = chosen
            next_velocity[side] = velocity
            joint_delta_deg = np.degrees(chosen - truth[side])
            joint_l2_deg = float(np.linalg.norm(joint_delta_deg))
            joint_max_deg = float(np.max(np.abs(joint_delta_deg)))
            metrics["joint_l2_deg"].append(joint_l2_deg)
            metrics["joint_max_deg"].append(joint_max_deg)
            metrics["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            metrics["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            metrics["truth_acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(truth_acceleration))))
            )
            metrics["acceleration_error_max_deg_s2"].append(
                float(
                    np.max(
                        np.abs(np.degrees(acceleration - truth_acceleration))
                    )
                )
            )
            if (
                metrics["first_joint_max_gt_1deg"] is None
                and joint_max_deg > 1.0
            ):
                metrics["first_joint_max_gt_1deg"] = int(reference.sequence)
                if len(metrics["events"]) < 32:
                    metrics["events"].append(
                        {
                            "type": "joint_parity_gt_1deg",
                            "reference_sequence": int(reference.sequence),
                            "left_source_index": int(
                                reference.left_source_index
                            ),
                            "right_source_index": int(
                                reference.right_source_index
                            ),
                            "interpolation_alpha": float(
                                reference.interpolation_alpha
                            ),
                            "joint_max_deg": joint_max_deg,
                            "chosen_q_deg": np.degrees(chosen).tolist(),
                            "truth_q_deg": np.degrees(truth[side]).tolist(),
                        }
                    )
            current_sign = (
                int(np.sign(float(chosen[2]))),
                int(np.sign(float(chosen[4]))),
            )
            truth_sign = (
                int(np.sign(float(truth[side][2]))),
                int(np.sign(float(truth[side][4]))),
            )
            if current_sign != previous_q3_q5_sign[side]:
                metrics["branch_sign_changes"] += 1
            if truth_sign != previous_truth_q3_q5_sign[side]:
                metrics["truth_branch_sign_changes"] += 1
            if current_sign != truth_sign:
                metrics["branch_sign_mismatch_frames"] += 1
                if (
                    metrics["branch_sign_mismatch_frames"] == 1
                    and len(metrics["events"]) < 32
                ):
                    metrics["events"].append(
                        {
                            "type": "branch_sign_mismatch",
                            "reference_sequence": int(reference.sequence),
                            "chosen_q3_q5_sign": list(current_sign),
                            "truth_q3_q5_sign": list(truth_sign),
                            "chosen_q_deg": np.degrees(chosen).tolist(),
                            "truth_q_deg": np.degrees(truth[side]).tolist(),
                        }
                    )
            previous_q3_q5_sign[side] = current_sign
            previous_truth_q3_q5_sign[side] = truth_sign

        current = {side: next_q[side].copy() for side in _SIDES}
        previous_velocity = {
            side: next_velocity[side].copy()
            for side in _SIDES
        }
        previous_truth = {
            side: np.asarray(truth[side], dtype=float).copy()
            for side in _SIDES
        }
        previous_truth_velocity = {
            side: next_truth_velocity.get(side, np.zeros(6, dtype=float)).copy()
            for side in _SIDES
        }
        diagnostics = planner.analyze_configuration(current)
        for side in _SIDES:
            sigma = float(diagnostics["sides"][side]["sigma_min"])
            side_metrics[side]["sigma_min"].append(sigma)
            if sigma < float(side_metrics[side]["min_sigma"]):
                side_metrics[side]["min_sigma"] = sigma
                side_metrics[side]["min_sigma_reference_sequence"] = int(
                    reference.sequence
                )

    return {
        "selector": "causal_nearest_joint_exact_pose",
        "reference_hz": float(reference_hz),
        "runtime_reference_count": len(references),
        "sides": {
            side: {
                "failures": int(metrics["failures"]),
                "latency_us": _summary(metrics["latency_us"]),
                "joint_l2_deg": _summary(metrics["joint_l2_deg"]),
                "joint_max_deg": _summary(metrics["joint_max_deg"]),
                "sigma_min": _summary(metrics["sigma_min"]),
                "velocity_max_deg_s": _summary(metrics["velocity_max_deg_s"]),
                "acceleration_max_deg_s2": _summary(
                    metrics["acceleration_max_deg_s2"]
                ),
                "truth_acceleration_max_deg_s2": _summary(
                    metrics["truth_acceleration_max_deg_s2"]
                ),
                "acceleration_error_max_deg_s2": _summary(
                    metrics["acceleration_error_max_deg_s2"]
                ),
                "branch_sign_changes": int(metrics["branch_sign_changes"]),
                "truth_branch_sign_changes": int(
                    metrics["truth_branch_sign_changes"]
                ),
                "branch_sign_mismatch_frames": int(
                    metrics["branch_sign_mismatch_frames"]
                ),
                "first_joint_max_gt_1deg": metrics["first_joint_max_gt_1deg"],
                "min_sigma_reference_sequence": metrics[
                    "min_sigma_reference_sequence"
                ],
                "events": list(metrics["events"]),
            }
            for side, metrics in side_metrics.items()
        },
    }


def _run_governed_closed_loop(
    *,
    group: Any,
    algo: Algo,
    planner: RM65PositionCollisionPlanner,
    endpoint_to_tool: dict[str, np.ndarray],
    reference_hz: float = 60.0,
) -> dict[str, Any]:
    """Prototype shared-alpha governor with exact all-IK endpoint candidates."""

    dt = 1.0 / float(reference_hz)
    current = _truth_at(group, 0)
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    max_velocity = np.asarray(
        planner.config.max_joint_velocity_rad_s,
        dtype=float,
    )
    max_acceleration = np.asarray(
        planner.config.max_joint_acceleration_rad_s2,
        dtype=float,
    )
    references = [
        reference
        for reference in _runtime_references(group, reference_hz)
        if int(reference.sequence) > 0
    ]
    alpha_values: list[float] = []
    step_latency_ms: list[float] = []
    failures = 0
    collision_rejections = 0
    per_side: dict[str, dict[str, list[float] | int]] = {
        side: {
            "joint_max_deg": [],
            "sigma_min": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
        }
        for side in _SIDES
    }

    def evaluate_alpha(
        alpha: float,
        *,
        desired_targets: dict[str, np.ndarray],
        current_endpoints: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray] | None, int]:
        if alpha <= 1e-12:
            candidate_lists = {
                side: [np.asarray(current[side], dtype=float).copy()]
                for side in _SIDES
            }
        else:
            candidate_lists: dict[str, list[np.ndarray]] = {}
            for side in _SIDES:
                governed_target = interpolate_pose(
                    current_endpoints[side],
                    desired_targets[side],
                    float(alpha),
                )
                candidates, result_code, _raw_count, _latency_us = (
                    _solve_valid_candidates(
                        algo=algo,
                        seed_rad=current[side],
                        target_endpoint=governed_target,
                    )
                )
                if result_code != 0:
                    return None, 0
                feasible: list[np.ndarray] = []
                for candidate in candidates:
                    velocity = (candidate - current[side]) / dt
                    acceleration = (
                        velocity - previous_velocity[side]
                    ) / dt
                    if np.any(np.abs(velocity) > max_velocity + 1e-9):
                        continue
                    if np.any(
                        np.abs(acceleration) > max_acceleration + 1e-9
                    ):
                        continue
                    feasible.append(candidate)
                if not feasible:
                    return None, 0
                candidate_lists[side] = feasible

        planner.reset(current)
        best: dict[str, np.ndarray] | None = None
        best_cost = math.inf
        rejected = 0
        for left, right in itertools.product(
            candidate_lists["left"],
            candidate_lists["right"],
        ):
            pair = {
                "left": np.asarray(left, dtype=float),
                "right": np.asarray(right, dtype=float),
            }
            collision = planner.validate_qpos_reference(
                pair,
                source="official_all_ik_reference_governor_probe",
            )
            if not collision.safe:
                rejected += 1
                continue
            cost = 0.0
            for side in _SIDES:
                velocity = (pair[side] - current[side]) / dt
                acceleration = (
                    velocity - previous_velocity[side]
                ) / dt
                cost += float(
                    np.dot(velocity / max_velocity, velocity / max_velocity)
                    + np.dot(
                        acceleration / max_acceleration,
                        acceleration / max_acceleration,
                    )
                )
            if cost < best_cost:
                best = pair
                best_cost = cost
        return best, rejected

    for reference in references:
        step_started = time.perf_counter()
        truth = _runtime_truth(group, reference)
        desired_targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        current_endpoints = {
            side: _sdk_pose_to_matrix(
                algo.rm_algo_forward_kinematics(
                    np.degrees(current[side]).tolist(),
                    flag=0,
                )
            )
            for side in _SIDES
        }

        evaluated: dict[float, dict[str, np.ndarray] | None] = {}
        rejected_total = 0

        def probe(alpha: float) -> dict[str, np.ndarray] | None:
            nonlocal rejected_total
            key = round(float(alpha), 12)
            if key not in evaluated:
                candidate, rejected = evaluate_alpha(
                    float(alpha),
                    desired_targets=desired_targets,
                    current_endpoints=current_endpoints,
                )
                rejected_total += int(rejected)
                evaluated[key] = candidate
            return evaluated[key]

        selected_alpha = 0.0
        selected = probe(0.0)
        previous_infeasible = 1.0
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
            candidate = probe(alpha)
            if candidate is not None:
                selected_alpha = float(alpha)
                selected = candidate
                if alpha < previous_infeasible:
                    low = float(alpha)
                    high = float(previous_infeasible)
                    for _ in range(6):
                        midpoint = 0.5 * (low + high)
                        midpoint_candidate = probe(midpoint)
                        if midpoint_candidate is None:
                            high = midpoint
                        else:
                            low = midpoint
                            selected_alpha = midpoint
                            selected = midpoint_candidate
                break
            previous_infeasible = float(alpha)

        collision_rejections += rejected_total
        if selected is None:
            failures += 1
            selected = {
                side: np.asarray(current[side], dtype=float).copy()
                for side in _SIDES
            }
            selected_alpha = 0.0
        alpha_values.append(float(selected_alpha))

        next_velocity: dict[str, np.ndarray] = {}
        for side in _SIDES:
            velocity = (selected[side] - current[side]) / dt
            acceleration = (velocity - previous_velocity[side]) / dt
            next_velocity[side] = velocity
            joint_error = np.degrees(selected[side] - truth[side])
            per_side[side]["joint_max_deg"].append(
                float(np.max(np.abs(joint_error)))
            )
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            realized_endpoint = _sdk_pose_to_matrix(
                algo.rm_algo_forward_kinematics(
                    np.degrees(selected[side]).tolist(),
                    flag=0,
                )
            )
            translation_error, rotation_error = _pose_error(
                realized_endpoint,
                desired_targets[side],
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)

        current = {
            side: np.asarray(selected[side], dtype=float).copy()
            for side in _SIDES
        }
        previous_velocity = {
            side: next_velocity[side].copy()
            for side in _SIDES
        }
        diagnostics = planner.analyze_configuration(current)
        for side in _SIDES:
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
        step_latency_ms.append(
            (time.perf_counter() - step_started) * 1000.0
        )

    alpha_array = np.asarray(alpha_values, dtype=float)
    return {
        "selector": "shared_alpha_all_ik_dynamic_collision_governor",
        "reference_hz": float(reference_hz),
        "runtime_reference_count": len(references),
        "failures": int(failures),
        "collision_rejections": int(collision_rejections),
        "alpha": _summary(alpha_values),
        "alpha_lt_1_frames": int(np.sum(alpha_array < 1.0 - 1e-9)),
        "alpha_eq_0_frames": int(np.sum(alpha_array <= 1e-12)),
        "step_latency_ms": _summary(step_latency_ms),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
                if isinstance(value, list)
            }
            for side, metrics in per_side.items()
        },
    }


def _run_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    planner: RM65PositionCollisionPlanner,
    endpoint_to_tool: dict[str, np.ndarray],
    max_frames: int,
    analyze_all_candidates: bool,
    include_governor: bool,
) -> dict[str, Any]:
    group = _build_group(dataset_root, episode_index)
    frame_count = int(group.frame_count)
    if max_frames > 0:
        frame_count = min(frame_count, int(max_frames))

    metrics: dict[str, dict[str, Any]] = {
        side: {
            "calls": 0,
            "failures": 0,
            "candidate_count": [],
            "project_valid_candidate_count": [],
            "latency_us": [],
            "teacher_fk_translation_m": [],
            "teacher_fk_rotation_deg": [],
            "closest_teacher_joint_l2_deg": [],
            "closest_teacher_joint_max_deg": [],
            "closest_candidate_fk_translation_m": [],
            "closest_candidate_fk_rotation_deg": [],
            "teacher_candidate_present_0p05deg": 0,
            "teacher_sigma_min": [],
            "best_exact_candidate_sigma_min": [],
            "best_sigma_gain": [],
            "q3_sign_count": {"negative": 0, "zero": 0, "positive": 0},
            "q5_sign_count": {"negative": 0, "zero": 0, "positive": 0},
        }
        for side in _SIDES
    }

    for frame_index in range(frame_count):
        q_ref = _truth_at(group, frame_index)
        tcp_targets = _pose_at(group, frame_index)
        endpoint_targets = {
            side: np.asarray(tcp_targets[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        teacher_diagnostics = planner.analyze_configuration(q_ref)

        for side in _SIDES:
            side_metrics = metrics[side]
            side_metrics["calls"] += 1
            seed_deg = np.degrees(np.asarray(q_ref[side], dtype=float))
            teacher_fk = _sdk_pose_to_matrix(
                algo.rm_algo_forward_kinematics(seed_deg.tolist(), flag=0)
            )
            teacher_translation, teacher_rotation = _pose_error(
                teacher_fk,
                endpoint_targets[side],
            )
            side_metrics["teacher_fk_translation_m"].append(teacher_translation)
            side_metrics["teacher_fk_rotation_deg"].append(teacher_rotation)
            teacher_sigma = float(
                teacher_diagnostics["sides"][side]["sigma_min"]
            )
            side_metrics["teacher_sigma_min"].append(teacher_sigma)

            params = rm_inverse_kinematics_params_t(
                seed_deg.tolist(),
                _matrix_to_sdk_pose(endpoint_targets[side]),
                0,
            )
            started_ns = time.perf_counter_ns()
            result = algo.rm_algo_inverse_kinematics_all(params)
            elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0
            side_metrics["latency_us"].append(float(elapsed_us))
            side_metrics["candidate_count"].append(int(result.num))
            if int(result.result) != 0 or int(result.num) <= 0:
                side_metrics["failures"] += 1
                side_metrics["project_valid_candidate_count"].append(0)
                continue

            candidates: list[np.ndarray] = []
            for candidate_index in range(int(result.num)):
                normalized = _normalize_candidate_near_seed(
                    np.asarray(result.q_solve[candidate_index][:6], dtype=float),
                    seed_deg,
                )
                if normalized is not None:
                    candidates.append(normalized)
                    for joint_key, joint_index in (("q3", 2), ("q5", 4)):
                        value = float(normalized[joint_index])
                        sign = (
                            "positive"
                            if value > 1e-4
                            else "negative"
                            if value < -1e-4
                            else "zero"
                        )
                        side_metrics[f"{joint_key}_sign_count"][sign] += 1
            side_metrics["project_valid_candidate_count"].append(len(candidates))
            if not candidates:
                continue

            joint_l2_errors = [
                float(np.linalg.norm(candidate - seed_deg))
                for candidate in candidates
            ]
            closest_index = int(np.argmin(joint_l2_errors))
            closest = candidates[closest_index]
            closest_delta = np.abs(closest - seed_deg)
            side_metrics["closest_teacher_joint_l2_deg"].append(
                float(np.linalg.norm(closest_delta))
            )
            side_metrics["closest_teacher_joint_max_deg"].append(
                float(np.max(closest_delta))
            )
            if float(np.max(closest_delta)) <= 0.05:
                side_metrics["teacher_candidate_present_0p05deg"] += 1
            closest_fk = _sdk_pose_to_matrix(
                algo.rm_algo_forward_kinematics(closest.tolist(), flag=0)
            )
            candidate_translation, candidate_rotation = _pose_error(
                closest_fk,
                endpoint_targets[side],
            )
            side_metrics["closest_candidate_fk_translation_m"].append(
                candidate_translation
            )
            side_metrics["closest_candidate_fk_rotation_deg"].append(
                candidate_rotation
            )

            if analyze_all_candidates:
                best_sigma = -math.inf
                for candidate in candidates:
                    candidate_q = {
                        other_side: np.asarray(
                            q_ref[other_side], dtype=float
                        ).copy()
                        for other_side in _SIDES
                    }
                    candidate_q[side] = np.radians(candidate)
                    diagnostics = planner.analyze_configuration(candidate_q)
                    best_sigma = max(
                        best_sigma,
                        float(diagnostics["sides"][side]["sigma_min"]),
                    )
                if math.isfinite(best_sigma):
                    side_metrics["best_exact_candidate_sigma_min"].append(
                        best_sigma
                    )
                    side_metrics["best_sigma_gain"].append(
                        best_sigma - teacher_sigma
                    )

    summarized_sides: dict[str, Any] = {}
    for side, side_metrics in metrics.items():
        summarized_sides[side] = {
            "calls": int(side_metrics["calls"]),
            "failures": int(side_metrics["failures"]),
            "teacher_candidate_present_0p05deg": int(
                side_metrics["teacher_candidate_present_0p05deg"]
            ),
            "candidate_count": _summary(side_metrics["candidate_count"]),
            "project_valid_candidate_count": _summary(
                side_metrics["project_valid_candidate_count"]
            ),
            "latency_us": _summary(side_metrics["latency_us"]),
            "teacher_fk_translation_m": _summary(
                side_metrics["teacher_fk_translation_m"]
            ),
            "teacher_fk_rotation_deg": _summary(
                side_metrics["teacher_fk_rotation_deg"]
            ),
            "closest_teacher_joint_l2_deg": _summary(
                side_metrics["closest_teacher_joint_l2_deg"]
            ),
            "closest_teacher_joint_max_deg": _summary(
                side_metrics["closest_teacher_joint_max_deg"]
            ),
            "closest_candidate_fk_translation_m": _summary(
                side_metrics["closest_candidate_fk_translation_m"]
            ),
            "closest_candidate_fk_rotation_deg": _summary(
                side_metrics["closest_candidate_fk_rotation_deg"]
            ),
            "teacher_sigma_min": _summary(side_metrics["teacher_sigma_min"]),
            "best_exact_candidate_sigma_min": _summary(
                side_metrics["best_exact_candidate_sigma_min"]
            ),
            "best_sigma_gain": _summary(side_metrics["best_sigma_gain"]),
            "q3_sign_count": dict(side_metrics["q3_sign_count"]),
            "q5_sign_count": dict(side_metrics["q5_sign_count"]),
        }
    output = {
        "episode_index": int(episode_index),
        "source_frame_count": int(frame_count),
        "source_action_hz": float(group.action_hz),
        "sides": summarized_sides,
        "closed_loop": _run_closed_loop(
            group=group,
            algo=algo,
            planner=planner,
            endpoint_to_tool=endpoint_to_tool,
        ),
    }
    if include_governor:
        output["closed_loop_governed"] = _run_governed_closed_loop(
            group=group,
            algo=algo,
            planner=planner,
            endpoint_to_tool=endpoint_to_tool,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pure-offline RealMan RM65 full-solution IK contract probe. "
            "No robot connection, device access, motion, or dispatch."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Episode index; repeat for multiple. Omit to scan all episodes.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit one compact row per episode instead of full metric summaries.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Emit only an aggregate across selected episodes.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-candidate Jacobian scans; all-IK and closed-loop checks remain.",
    )
    parser.add_argument(
        "--governed",
        action="store_true",
        help="Also run the shared-alpha dynamic/collision reference governor prototype.",
    )
    parser.add_argument("--left-tool-profile", default=_LEFT_TOOL)
    parser.add_argument("--right-tool-profile", default=_RIGHT_TOOL)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    episode_indices = (
        [int(value) for value in args.episode]
        if args.episode
        else read_replay_dataset_episode_indices(dataset_root)
    )
    endpoint_to_tool = {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(
                getattr(args, f"{side}_tool_profile"),
                side=side,
            )
        )
        for side in _SIDES
    }
    algo = Algo(
        rm_robot_arm_model_e.RM_MODEL_RM_65_E,
        rm_force_type_e.RM_MODEL_RM_B_E,
    )
    planner = _build_planner(endpoint_to_tool)
    episodes = [
        _run_episode(
            dataset_root=dataset_root,
            episode_index=episode_index,
            algo=algo,
            planner=planner,
            endpoint_to_tool=endpoint_to_tool,
            max_frames=int(args.max_frames),
            analyze_all_candidates=not bool(args.fast),
            include_governor=bool(args.governed),
        )
        for episode_index in episode_indices
    ]
    if args.compact or args.aggregate_only:
        episodes = [
            {
                "episode_index": int(episode["episode_index"]),
                "source_frame_count": int(episode["source_frame_count"]),
                "sides": {
                    side: {
                        "source_failures": int(episode["sides"][side]["failures"]),
                        "teacher_candidates_present": int(
                            episode["sides"][side][
                                "teacher_candidate_present_0p05deg"
                            ]
                        ),
                        "teacher_fk_translation_max_m": episode["sides"][side][
                            "teacher_fk_translation_m"
                        ]["max"],
                        "teacher_fk_rotation_max_deg": episode["sides"][side][
                            "teacher_fk_rotation_deg"
                        ]["max"],
                        "source_latency_p99_us": episode["sides"][side][
                            "latency_us"
                        ]["p99"],
                        "closed_loop_failures": int(
                            episode["closed_loop"]["sides"][side]["failures"]
                        ),
                        "closed_loop_joint_max_deg": episode["closed_loop"][
                            "sides"
                        ][side]["joint_max_deg"]["max"],
                        "closed_loop_sigma_min": episode["closed_loop"]["sides"][
                            side
                        ]["sigma_min"]["min"],
                        "closed_loop_acceleration_max_deg_s2": episode[
                            "closed_loop"
                        ]["sides"][side]["acceleration_max_deg_s2"]["max"],
                        "truth_acceleration_max_deg_s2": episode["closed_loop"][
                            "sides"
                        ][side]["truth_acceleration_max_deg_s2"]["max"],
                        "acceleration_error_max_deg_s2": episode["closed_loop"][
                            "sides"
                        ][side]["acceleration_error_max_deg_s2"]["max"],
                        "closed_loop_branch_sign_changes": int(
                            episode["closed_loop"]["sides"][side][
                                "branch_sign_changes"
                            ]
                        ),
                        "truth_branch_sign_changes": int(
                            episode["closed_loop"]["sides"][side][
                                "truth_branch_sign_changes"
                            ]
                        ),
                        "branch_sign_mismatch_frames": int(
                            episode["closed_loop"]["sides"][side][
                                "branch_sign_mismatch_frames"
                            ]
                        ),
                        **(
                            {}
                            if "closed_loop_governed" not in episode
                            else {
                                "governed_failures": int(
                                    episode["closed_loop_governed"]["failures"]
                                ),
                                "governed_alpha_min": episode[
                                    "closed_loop_governed"
                                ]["alpha"]["min"],
                                "governed_alpha_mean": episode[
                                    "closed_loop_governed"
                                ]["alpha"]["mean"],
                                "governed_alpha_lt_1_frames": int(
                                    episode["closed_loop_governed"][
                                        "alpha_lt_1_frames"
                                    ]
                                ),
                                "governed_joint_max_deg": episode[
                                    "closed_loop_governed"
                                ]["sides"][side]["joint_max_deg"]["max"],
                                "governed_sigma_min": episode[
                                    "closed_loop_governed"
                                ]["sides"][side]["sigma_min"]["min"],
                                "governed_acceleration_max_deg_s2": episode[
                                    "closed_loop_governed"
                                ]["sides"][side][
                                    "acceleration_max_deg_s2"
                                ]["max"],
                                "governed_translation_error_max_m": episode[
                                    "closed_loop_governed"
                                ]["sides"][side][
                                    "target_translation_error_m"
                                ]["max"],
                                "governed_rotation_error_max_deg": episode[
                                    "closed_loop_governed"
                                ]["sides"][side][
                                    "target_rotation_error_deg"
                                ]["max"],
                                "governed_step_latency_p99_ms": episode[
                                    "closed_loop_governed"
                                ]["step_latency_ms"]["p99"],
                            }
                        ),
                    }
                    for side in _SIDES
                },
            }
            for episode in episodes
        ]
    aggregate: dict[str, Any] | None = None
    if args.aggregate_only:
        aggregate = {
            "episode_count": len(episodes),
            "source_frame_count": int(
                sum(int(episode["source_frame_count"]) for episode in episodes)
            ),
            "sides": {},
        }
        for side in _SIDES:
            aggregate["sides"][side] = {
                "source_failures": int(
                    sum(
                        int(episode["sides"][side]["source_failures"])
                        for episode in episodes
                    )
                ),
                "closed_loop_failures": int(
                    sum(
                        int(episode["sides"][side]["closed_loop_failures"])
                        for episode in episodes
                    )
                ),
                "closed_loop_branch_sign_changes": int(
                    sum(
                        int(
                            episode["sides"][side][
                                "closed_loop_branch_sign_changes"
                            ]
                        )
                        for episode in episodes
                    )
                ),
                "truth_branch_sign_changes": int(
                    sum(
                        int(
                            episode["sides"][side][
                                "truth_branch_sign_changes"
                            ]
                        )
                        for episode in episodes
                    )
                ),
                "branch_sign_mismatch_frames": int(
                    sum(
                        int(
                            episode["sides"][side][
                                "branch_sign_mismatch_frames"
                            ]
                        )
                        for episode in episodes
                    )
                ),
            }
            for metric, reducer in (
                ("teacher_fk_translation_max_m", max),
                ("teacher_fk_rotation_max_deg", max),
                ("source_latency_p99_us", max),
                ("closed_loop_joint_max_deg", max),
                ("closed_loop_sigma_min", min),
                ("closed_loop_acceleration_max_deg_s2", max),
                ("truth_acceleration_max_deg_s2", max),
                ("acceleration_error_max_deg_s2", max),
            ):
                valid = [
                    (
                        float(episode["sides"][side][metric]),
                        int(episode["episode_index"]),
                    )
                    for episode in episodes
                    if episode["sides"][side][metric] is not None
                ]
                if not valid:
                    aggregate["sides"][side][metric] = None
                    continue
                selected_value = reducer(value for value, _episode in valid)
                selected_episode = next(
                    episode
                    for value, episode in valid
                    if value == selected_value
                )
                aggregate["sides"][side][metric] = {
                    "value": float(selected_value),
                    "episode_index": int(selected_episode),
                }
        episodes = []
    print(
        json.dumps(
            {
                "schema": "todos.rm65_official_all_ik_probe/v1",
                "device_access": False,
                "dispatch": False,
                "motion": False,
                "sdk_algorithm_version": str(algo.rm_algo_version()),
                "dataset_root": str(dataset_root),
                "episodes": episodes,
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
