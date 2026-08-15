"""Offline RM65 qref-to-qcmd dynamics probe.

This probe reads only recorded parquet files.  It compares the materialized
15 Hz accepted qref, the recorded latest transport-accepted qcmd snapshot and
an offline 120 Hz projection of the qref path onto explicit joint velocity,
acceleration and jerk bounds.  It never initializes a device or dispatches a
command.

The constrained projection is deliberately future-aware.  It is an offline
feasibility/trade-off oracle, not a runtime implementation.  A causal deploy
governor must later reproduce the accepted trade-off over a bounded lookahead
window.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import osqp
import pyarrow.parquet as pq
from scipy import sparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
)
from robot_motion.rm65.pose_lift_profile import (
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
)
from robot_motion.rm65.position_collision import (
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
)
from robot_motion.rm65.tool_profiles import (
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.position_control import (
    JointHorizonLimits,
    JointHorizonState,
    plan_jerk_limited_joint_horizon,
)
from scripts.probes.rm65_official_all_ik_probe import (
    _LEFT_TOOL,
    _RIGHT_TOOL,
    _pose_error,
)


_SIDES = ("left", "right")
_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")
_SOURCE_HZ = 15.0
_COMMAND_HZ = 120.0
_COMMANDS_PER_SOURCE_INTERVAL = int(_COMMAND_HZ / _SOURCE_HZ)
_QP_BOUND_GUARD = 0.999


def _as_matrix(column: Any) -> np.ndarray:
    values = column.to_pylist()
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError(f"expected an Nx6 qpos column, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("qpos column contains non-finite values.")
    return matrix


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _uniform_joint_reference(source_q: np.ndarray) -> np.ndarray:
    if source_q.shape[0] < 2:
        return source_q.copy()
    rows: list[np.ndarray] = []
    for index in range(source_q.shape[0] - 1):
        for step in range(_COMMANDS_PER_SOURCE_INTERVAL):
            alpha = step / float(_COMMANDS_PER_SOURCE_INTERVAL)
            rows.append(
                (1.0 - alpha) * source_q[index]
                + alpha * source_q[index + 1]
            )
    rows.append(source_q[-1].copy())
    return np.asarray(rows, dtype=float)


def _continuous_equivalent_source(
    source_q: np.ndarray,
    *,
    enforce_limits: bool = True,
) -> np.ndarray:
    """Choose the closest legal 2π-equivalent representation per source frame."""

    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    continuous = np.asarray(source_q, dtype=float).copy()
    for index in range(1, continuous.shape[0]):
        for joint in range(6):
            raw = float(continuous[index, joint])
            equivalents = [
                raw + 2.0 * math.pi * turns
                for turns in range(-2, 3)
                if lower[joint] - 1e-10
                <= raw + 2.0 * math.pi * turns
                <= upper[joint] + 1e-10
            ]
            if equivalents:
                continuous[index, joint] = min(
                    equivalents,
                    key=lambda value: abs(value - continuous[index - 1, joint]),
                )
    below = lower - continuous
    above = continuous - upper
    if enforce_limits and (
        np.any(below > 5e-6) or np.any(above > 5e-6)
    ):
        raise ValueError(
            "stored qref exceeds RM65 absolute limits beyond float32 tolerance."
        )
    if enforce_limits:
        continuous = np.clip(continuous, lower, upper)
    return continuous


def _difference_matrix(order: int, count: int, dt: float) -> sparse.csc_matrix:
    if order == 1:
        coefficients = (-1.0, 1.0)
    elif order == 2:
        coefficients = (1.0, -2.0, 1.0)
    elif order == 3:
        coefficients = (-1.0, 3.0, -3.0, 1.0)
    else:
        raise ValueError(f"unsupported finite-difference order {order}.")
    row_count = count - order
    if row_count <= 0:
        return sparse.csc_matrix((0, count), dtype=float)
    diagonals = [
        np.full(row_count, coefficient / (dt**order), dtype=float)
        for coefficient in coefficients
    ]
    return sparse.diags(
        diagonals,
        offsets=tuple(range(order + 1)),
        shape=(row_count, count),
        format="csc",
    )


def _project_joint_path(
    reference: np.ndarray,
    *,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the closest whole-episode path satisfying box dynamics limits."""

    count = int(reference.shape[0])
    dt = 1.0 / _COMMAND_HZ
    identity = sparse.eye(count, format="csc")
    d1 = _difference_matrix(1, count, dt)
    d2 = _difference_matrix(2, count, dt)
    d3 = _difference_matrix(3, count, dt)
    constraints = sparse.vstack((identity, d1, d2, d3), format="csc")
    projected = np.empty_like(reference)
    receipts: list[dict[str, Any]] = []
    lower_joint = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper_joint = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)

    for joint in range(6):
        lower = np.concatenate(
            (
                np.full(count, lower_joint[joint], dtype=float),
                np.full(
                    d1.shape[0],
                    -_QP_BOUND_GUARD * max_velocity[joint],
                    dtype=float,
                ),
                np.full(
                    d2.shape[0],
                    -_QP_BOUND_GUARD * max_acceleration[joint],
                    dtype=float,
                ),
                np.full(
                    d3.shape[0],
                    -_QP_BOUND_GUARD * max_jerk[joint],
                    dtype=float,
                ),
            )
        )
        upper = np.concatenate(
            (
                np.full(count, upper_joint[joint], dtype=float),
                np.full(
                    d1.shape[0],
                    _QP_BOUND_GUARD * max_velocity[joint],
                    dtype=float,
                ),
                np.full(
                    d2.shape[0],
                    _QP_BOUND_GUARD * max_acceleration[joint],
                    dtype=float,
                ),
                np.full(
                    d3.shape[0],
                    _QP_BOUND_GUARD * max_jerk[joint],
                    dtype=float,
                ),
            )
        )
        # The first materialized qpos is the replay/deploy initialization
        # contract.  Keep it exact without forcing zero initial velocity.
        lower[0] = reference[0, joint]
        upper[0] = reference[0, joint]
        solver = osqp.OSQP()
        solver.setup(
            P=2.0 * identity,
            q=-2.0 * reference[:, joint],
            A=constraints,
            l=lower,
            u=upper,
            verbose=False,
            polishing=True,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=100_000,
        )
        result = solver.solve()
        if result.x is None or result.info.status_val not in {1, 2}:
            raise RuntimeError(
                f"OSQP failed for joint {joint + 1}: {result.info.status}."
            )
        projected[:, joint] = np.asarray(result.x, dtype=float)
        receipts.append(
            {
                "joint_index": joint + 1,
                "status": str(result.info.status),
                "iterations": int(result.info.iter),
                "solve_time_sec": float(result.info.solve_time),
            }
        )
    return projected, receipts


def _solve_state_horizon(
    reference: np.ndarray,
    *,
    initial_q: np.ndarray,
    initial_velocity: np.ndarray,
    initial_acceleration: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Solve one bounded-lookahead third-order joint tracking horizon."""

    count = int(reference.shape[0])
    if count < 2:
        return (
            np.asarray(initial_q, dtype=float).reshape(1, 6),
            np.asarray(initial_velocity, dtype=float).reshape(1, 6),
            np.asarray(initial_acceleration, dtype=float).reshape(1, 6),
            [],
        )
    dt = 1.0 / _COMMAND_HZ
    plan = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=JointHorizonState(
            q_rad=initial_q,
            qd_rad_s=initial_velocity,
            qdd_rad_s2=initial_acceleration,
        ),
        limits=JointHorizonLimits(
            min_position_rad=RM65_ABSOLUTE_Q_MIN_RAD,
            max_position_rad=RM65_ABSOLUTE_Q_MAX_RAD,
            max_velocity_rad_s=max_velocity,
            max_acceleration_rad_s2=max_acceleration,
            max_jerk_rad_s3=max_jerk,
        ),
        dt_sec=dt,
        commit_count=min(_COMMANDS_PER_SOURCE_INTERVAL, count - 1),
        bound_guard=_QP_BOUND_GUARD,
    )
    return (
        np.asarray(plan.q_path_rad),
        np.asarray(plan.qd_path_rad_s),
        np.asarray(plan.qdd_path_rad_s2),
        [dict(receipt) for receipt in plan.metadata["solver_receipts"]],
    )


def _causal_project_joint_path(
    reference: np.ndarray,
    *,
    lookahead_source_steps: int,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replan every source interval and commit only the first control cut."""

    if lookahead_source_steps < 1:
        raise ValueError("lookahead_source_steps must be positive.")
    if reference.shape[0] <= 1:
        return reference.copy(), {
            "replan_count": 0,
            "solve_time_sec": 0.0,
            "max_iterations": 0,
        }
    command = np.empty_like(reference)
    command[0] = reference[0]
    current_q = reference[0].copy()
    current_velocity = np.zeros(6, dtype=float)
    current_acceleration = np.zeros(6, dtype=float)
    replan_count = 0
    total_solve_time = 0.0
    max_iterations = 0
    final_index = int(reference.shape[0] - 1)
    start = 0
    while start < final_index:
        horizon_end = min(
            final_index,
            start
            + lookahead_source_steps * _COMMANDS_PER_SOURCE_INTERVAL,
        )
        horizon_reference = reference[start : horizon_end + 1]
        q_path, velocity_path, acceleration_path, receipts = (
            _solve_state_horizon(
                horizon_reference,
                initial_q=current_q,
                initial_velocity=current_velocity,
                initial_acceleration=current_acceleration,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                max_jerk=max_jerk,
            )
        )
        commit_count = min(
            _COMMANDS_PER_SOURCE_INTERVAL,
            horizon_end - start,
        )
        command[start + 1 : start + commit_count + 1] = q_path[
            1 : commit_count + 1
        ]
        current_q = q_path[commit_count].copy()
        current_velocity = velocity_path[commit_count].copy()
        current_acceleration = acceleration_path[commit_count].copy()
        replan_count += 1
        total_solve_time += sum(
            float(receipt["solve_time_sec"]) for receipt in receipts
        )
        max_iterations = max(
            max_iterations,
            max(
                (int(receipt["iterations"]) for receipt in receipts),
                default=0,
            ),
        )
        start += commit_count
    return command, {
        "replan_count": int(replan_count),
        "solve_time_sec": float(total_solve_time),
        "max_iterations": int(max_iterations),
    }


def _dynamics(path: np.ndarray) -> dict[str, Any]:
    dt = 1.0 / _COMMAND_HZ
    velocity = np.diff(path, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt

    def maxima(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return [0.0] * 6
        return np.max(np.abs(np.degrees(values)), axis=0).tolist()

    return {
        "max_velocity_deg_s_by_joint": maxima(velocity),
        "max_acceleration_deg_s2_by_joint": maxima(acceleration),
        "max_jerk_deg_s3_by_joint": maxima(jerk),
        "max_velocity_deg_s": max(maxima(velocity)),
        "max_acceleration_deg_s2": max(maxima(acceleration)),
        "max_jerk_deg_s3": max(maxima(jerk)),
    }


def _combined_dynamics(paths: list[np.ndarray]) -> dict[str, Any]:
    reports = [_dynamics(path) for path in paths if path.shape[0] >= 1]
    if not reports:
        return _dynamics(np.zeros((1, 6), dtype=float))
    keys = (
        "max_velocity_deg_s_by_joint",
        "max_acceleration_deg_s2_by_joint",
        "max_jerk_deg_s3_by_joint",
    )
    combined: dict[str, Any] = {}
    for key in keys:
        combined[key] = np.max(
            np.asarray([report[key] for report in reports], dtype=float),
            axis=0,
        ).tolist()
    combined["max_velocity_deg_s"] = max(combined[keys[0]])
    combined["max_acceleration_deg_s2"] = max(combined[keys[1]])
    combined["max_jerk_deg_s3"] = max(combined[keys[2]])
    return combined


def _build_collision_planner(
    tool_profiles: dict[str, str],
) -> RM65PositionCollisionPlanner:
    endpoint_to_tool = {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(tool_profiles[side], side=side)
        )
        for side in _SIDES
    }
    return RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            pose_lift_profile=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )


def _validate_kinematics_and_sampled_collision(
    *,
    planner: RM65PositionCollisionPlanner,
    reference_segments_by_side: dict[str, list[np.ndarray]],
    command_segments_by_side: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    translation_error = {side: [] for side in _SIDES}
    rotation_error = {side: [] for side in _SIDES}
    sigma_min = {side: [] for side in _SIDES}
    endpoint_collision_frames = 0
    sampled_segment_collision_intervals = 0
    endpoint_collision_examples: list[dict[str, Any]] = []
    sampled_segment_collision_examples: list[dict[str, Any]] = []
    minimum_collision_distance_m = math.inf
    checked_frames = 0
    checked_intervals = 0
    segment_count = len(reference_segments_by_side["left"])
    if segment_count:
        planner.reset(
            {
                side: command_segments_by_side[side][0][0]
                for side in _SIDES
            }
        )
    for segment_index in range(segment_count):
        references = {
            side: reference_segments_by_side[side][segment_index]
            for side in _SIDES
        }
        commands = {
            side: command_segments_by_side[side][segment_index]
            for side in _SIDES
        }
        count = int(commands["left"].shape[0])
        if any(
            values.shape[0] != count
            for values in (*references.values(), *commands.values())
        ):
            raise ValueError("left/right reference and command segments misalign.")
        for frame_index in range(count):
            command_pair = {
                side: commands[side][frame_index] for side in _SIDES
            }
            reference_pair = {
                side: references[side][frame_index] for side in _SIDES
            }
            validation = planner.validate_qpos_reference(
                command_pair,
                source="offline_qref_dynamics_causal_projection",
            )
            endpoint_collision_frames += int(not validation.safe)
            if validation.collision_distance_m is not None:
                minimum_collision_distance_m = min(
                    minimum_collision_distance_m,
                    float(validation.collision_distance_m),
                )
            if not validation.safe and len(endpoint_collision_examples) < 10:
                endpoint_collision_examples.append(
                    {
                        "continuity_segment_ordinal": segment_index,
                        "frame_index_120hz": frame_index,
                        "reason": validation.reason,
                        "collision_distance_m": (
                            None
                            if validation.collision_distance_m is None
                            else float(validation.collision_distance_m)
                        ),
                    }
                )
            checked_frames += 1
            reference_pose = planner.observe_endpoint_poses_world(
                reference_pair
            )
            command_pose = planner.observe_endpoint_poses_world(command_pair)
            diagnostics = planner.analyze_configuration(command_pair)
            for side in _SIDES:
                translation, rotation = _pose_error(
                    command_pose[side],
                    reference_pose[side],
                )
                translation_error[side].append(float(translation))
                rotation_error[side].append(float(rotation))
                sigma_min[side].append(
                    float(diagnostics["sides"][side]["sigma_min"])
                )
            if frame_index == 0:
                continue
            checked_intervals += 1
            sampled_unsafe = False
            sampled_failure: dict[str, Any] | None = None
            for alpha in (0.25, 0.50, 0.75):
                sample = {
                    side: (
                        (1.0 - alpha) * commands[side][frame_index - 1]
                        + alpha * commands[side][frame_index]
                    )
                    for side in _SIDES
                }
                sampled_validation = planner.validate_qpos_reference(
                    sample,
                    source=(
                        "offline_qref_dynamics_causal_projection_"
                        "sampled_segment"
                    ),
                )
                if sampled_validation.collision_distance_m is not None:
                    minimum_collision_distance_m = min(
                        minimum_collision_distance_m,
                        float(sampled_validation.collision_distance_m),
                    )
                if not sampled_validation.safe:
                    sampled_unsafe = True
                    sampled_failure = {
                        "continuity_segment_ordinal": segment_index,
                        "interval_end_frame_index_120hz": frame_index,
                        "alpha": float(alpha),
                        "reason": sampled_validation.reason,
                        "collision_distance_m": (
                            None
                            if sampled_validation.collision_distance_m is None
                            else float(
                                sampled_validation.collision_distance_m
                            )
                        ),
                    }
                    break
            sampled_segment_collision_intervals += int(sampled_unsafe)
            if (
                sampled_failure is not None
                and len(sampled_segment_collision_examples) < 10
            ):
                sampled_segment_collision_examples.append(sampled_failure)
    return {
        "checked_frames": int(checked_frames),
        "endpoint_collision_frames": int(endpoint_collision_frames),
        "checked_intervals": int(checked_intervals),
        "sampled_segment_collision_intervals": int(
            sampled_segment_collision_intervals
        ),
        "minimum_sampled_collision_distance_m": (
            None
            if not math.isfinite(minimum_collision_distance_m)
            else float(minimum_collision_distance_m)
        ),
        "endpoint_collision_examples": endpoint_collision_examples,
        "sampled_segment_collision_examples": (
            sampled_segment_collision_examples
        ),
        "segment_sampling_alphas": [0.25, 0.50, 0.75],
        "continuous_segment_collision_certified": False,
        "sides": {
            side: {
                "translation_error_m": _summary(translation_error[side]),
                "rotation_error_deg": _summary(rotation_error[side]),
                "sigma_min": _summary(sigma_min[side]),
            }
            for side in _SIDES
        },
    }


def _episode_report(
    parquet_path: Path,
    *,
    jerk_ramp_sec: float,
    run_projection: bool,
    causal_lookahead_source_steps: tuple[int, ...],
    kinematic_validation: bool,
    tool_profiles: dict[str, str],
) -> dict[str, Any]:
    match = _EPISODE_PATTERN.search(parquet_path.name)
    if match is None:
        raise ValueError(f"cannot parse episode index from {parquet_path}.")
    episode_index = int(match.group(1))
    columns = ["collection.continuity_segment_index"]
    for side in _SIDES:
        columns.extend(
            (
                f"action.reference.{side}_arm.q_ref",
                f"position.transport.{side}.last_accepted_qpos",
            )
        )
    table = pq.read_table(parquet_path, columns=columns)
    continuity_segments = np.asarray(
        table["collection.continuity_segment_index"].to_numpy(),
        dtype=int,
    )
    max_velocity = np.asarray(
        RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        dtype=float,
    )
    max_acceleration = np.asarray(
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        dtype=float,
    )
    max_jerk = max_acceleration / float(jerk_ramp_sec)
    report: dict[str, Any] = {
        "episode_index": episode_index,
        "source_frames": int(table.num_rows),
        "jerk_ramp_sec": float(jerk_ramp_sec),
        "limits": {
            "max_velocity_deg_s_by_joint": np.degrees(max_velocity).tolist(),
            "max_acceleration_deg_s2_by_joint": np.degrees(
                max_acceleration
            ).tolist(),
            "max_jerk_deg_s3_by_joint": np.degrees(max_jerk).tolist(),
        },
        "sides": {},
    }
    reference_segments_by_side: dict[str, list[np.ndarray]] = {}
    causal_segments_by_lookahead: dict[
        int,
        dict[str, list[np.ndarray]],
    ] = {
        lookahead: {} for lookahead in causal_lookahead_source_steps
    }
    for side in _SIDES:
        raw_source = _as_matrix(table[f"action.reference.{side}_arm.q_ref"])
        source = _continuous_equivalent_source(raw_source)
        recorded_qcmd = _as_matrix(
            table[f"position.transport.{side}.last_accepted_qpos"]
        )
        segment_source = [
            source[continuity_segments == segment]
            for segment in np.unique(continuity_segments)
        ]
        segment_references = [
            _uniform_joint_reference(values)
            for values in segment_source
            if values.shape[0] >= 1
        ]
        reference_segments_by_side[side] = segment_references
        side_report: dict[str, Any] = {
            "continuity_segment_count": len(segment_references),
            "reference_dynamics": _combined_dynamics(segment_references),
            "recorded_qcmd_source_sample_error_deg": {
                "mean": float(
                    np.mean(
                        np.abs(
                            np.degrees(
                                _continuous_equivalent_source(
                                    recorded_qcmd,
                                    enforce_limits=False,
                                )
                                - source
                            )
                        )
                    )
                ),
                "p95": float(
                    np.percentile(
                        np.abs(
                            np.degrees(
                                _continuous_equivalent_source(
                                    recorded_qcmd,
                                    enforce_limits=False,
                                )
                                - source
                            )
                        ),
                        95.0,
                    )
                ),
                "max": float(
                    np.max(
                        np.abs(
                            np.degrees(
                                _continuous_equivalent_source(
                                    recorded_qcmd,
                                    enforce_limits=False,
                                )
                                - source
                            )
                        )
                    )
                ),
            },
        }
        if run_projection:
            projected_segments: list[np.ndarray] = []
            error_parts: list[np.ndarray] = []
            source_error_parts: list[np.ndarray] = []
            receipts: list[dict[str, Any]] = []
            for segment_index, (source_values, reference) in enumerate(
                zip(segment_source, segment_references, strict=True)
            ):
                projected, segment_receipts = _project_joint_path(
                    reference,
                    max_velocity=max_velocity,
                    max_acceleration=max_acceleration,
                    max_jerk=max_jerk,
                )
                projected_segments.append(projected)
                error_parts.append(
                    np.abs(np.degrees(projected - reference))
                )
                source_indices = np.arange(source_values.shape[0]) * (
                    _COMMANDS_PER_SOURCE_INTERVAL
                )
                source_error_parts.append(
                    np.abs(
                        np.degrees(
                            projected[source_indices] - source_values
                        )
                    )
                )
                receipts.extend(
                    {
                        **receipt,
                        "continuity_segment_ordinal": segment_index,
                    }
                    for receipt in segment_receipts
                )
            error_deg = np.concatenate(error_parts, axis=0)
            source_error_deg = np.concatenate(source_error_parts, axis=0)
            side_report["projected"] = {
                "dynamics": _combined_dynamics(projected_segments),
                "reference_error_deg": {
                    "mean": float(np.mean(error_deg)),
                    "p95": float(np.percentile(error_deg, 95.0)),
                    "max": float(np.max(error_deg)),
                },
                "source_keyframe_error_deg": {
                    "mean": float(np.mean(source_error_deg)),
                    "p95": float(np.percentile(source_error_deg, 95.0)),
                    "max": float(np.max(source_error_deg)),
                },
                "solver_receipts": receipts,
            }
        if causal_lookahead_source_steps:
            causal_reports: dict[str, Any] = {}
            for lookahead in causal_lookahead_source_steps:
                causal_segments: list[np.ndarray] = []
                causal_error_parts: list[np.ndarray] = []
                causal_source_error_parts: list[np.ndarray] = []
                causal_receipts: list[dict[str, Any]] = []
                for source_values, reference in zip(
                    segment_source,
                    segment_references,
                    strict=True,
                ):
                    causal, receipt = _causal_project_joint_path(
                        reference,
                        lookahead_source_steps=lookahead,
                        max_velocity=max_velocity,
                        max_acceleration=max_acceleration,
                        max_jerk=max_jerk,
                    )
                    causal_segments.append(causal)
                    causal_error_parts.append(
                        np.abs(np.degrees(causal - reference))
                    )
                    source_indices = np.arange(source_values.shape[0]) * (
                        _COMMANDS_PER_SOURCE_INTERVAL
                    )
                    causal_source_error_parts.append(
                        np.abs(
                            np.degrees(
                                causal[source_indices] - source_values
                            )
                        )
                    )
                    causal_receipts.append(receipt)
                causal_error = np.concatenate(
                    causal_error_parts,
                    axis=0,
                )
                causal_source_error = np.concatenate(
                    causal_source_error_parts,
                    axis=0,
                )
                causal_reports[str(lookahead)] = {
                    "dynamics": _combined_dynamics(causal_segments),
                    "reference_error_deg": {
                        "mean": float(np.mean(causal_error)),
                        "p95": float(
                            np.percentile(causal_error, 95.0)
                        ),
                        "max": float(np.max(causal_error)),
                    },
                    "source_keyframe_error_deg": {
                        "mean": float(np.mean(causal_source_error)),
                        "p95": float(
                            np.percentile(causal_source_error, 95.0)
                        ),
                        "max": float(np.max(causal_source_error)),
                    },
                    "replan_count": sum(
                        int(receipt["replan_count"])
                        for receipt in causal_receipts
                    ),
                    "solve_time_sec": sum(
                        float(receipt["solve_time_sec"])
                        for receipt in causal_receipts
                    ),
                    "max_iterations": max(
                        (
                            int(receipt["max_iterations"])
                            for receipt in causal_receipts
                        ),
                        default=0,
                    ),
                }
                causal_segments_by_lookahead[lookahead][side] = (
                    causal_segments
                )
            side_report["causal_projection_by_lookahead_source_steps"] = (
                causal_reports
            )
        report["sides"][side] = side_report
    if kinematic_validation:
        if not causal_lookahead_source_steps:
            raise ValueError(
                "--kinematic-validation requires at least one "
                "--causal-lookahead-source-steps value."
            )
        planner = _build_collision_planner(tool_profiles)
        report["kinematic_validation_by_lookahead_source_steps"] = {
            str(lookahead): _validate_kinematics_and_sampled_collision(
                planner=planner,
                reference_segments_by_side=reference_segments_by_side,
                command_segments_by_side=causal_segments_by_lookahead[
                    lookahead
                ],
            )
            for lookahead in causal_lookahead_source_steps
        }
    return report


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "episode_count": len(reports),
        "source_frame_count": sum(
            int(report["source_frames"]) for report in reports
        ),
        "sides": {},
    }
    for side in _SIDES:
        keys = (
            "reference_max_acceleration_deg_s2",
            "reference_max_jerk_deg_s3",
            "recorded_qcmd_source_sample_error_max_deg",
            "projected_reference_error_max_deg",
            "projected_source_keyframe_error_max_deg",
            "projected_max_acceleration_deg_s2",
            "projected_max_jerk_deg_s3",
        )
        values: dict[str, list[float]] = {key: [] for key in keys}
        worst: dict[str, dict[str, float | int]] = {}
        for report in reports:
            side_report = report["sides"][side]
            extracted = {
                "reference_max_acceleration_deg_s2": side_report[
                    "reference_dynamics"
                ]["max_acceleration_deg_s2"],
                "reference_max_jerk_deg_s3": side_report[
                    "reference_dynamics"
                ]["max_jerk_deg_s3"],
                "recorded_qcmd_source_sample_error_max_deg": side_report[
                    "recorded_qcmd_source_sample_error_deg"
                ]["max"],
            }
            projected = side_report.get("projected")
            if projected is not None:
                extracted.update(
                    {
                        "projected_reference_error_max_deg": projected[
                            "reference_error_deg"
                        ]["max"],
                        "projected_source_keyframe_error_max_deg": projected[
                            "source_keyframe_error_deg"
                        ]["max"],
                        "projected_max_acceleration_deg_s2": projected[
                            "dynamics"
                        ]["max_acceleration_deg_s2"],
                        "projected_max_jerk_deg_s3": projected["dynamics"][
                            "max_jerk_deg_s3"
                        ],
                    }
                )
            for key, value in extracted.items():
                normalized = float(value)
                values[key].append(normalized)
                if key not in worst or normalized > float(worst[key]["value"]):
                    worst[key] = {
                        "episode_index": int(report["episode_index"]),
                        "value": normalized,
                    }
        aggregate["sides"][side] = {
            "metrics": {
                key: _summary(metric_values)
                for key, metric_values in values.items()
                if metric_values
            },
            "worst_episode": worst,
        }
        lookaheads = sorted(
            {
                int(value)
                for report in reports
                for value in (
                    report["sides"][side]
                    .get("causal_projection_by_lookahead_source_steps", {})
                    .keys()
                )
            }
        )
        if lookaheads:
            causal_aggregate: dict[str, Any] = {}
            for lookahead in lookaheads:
                metric_values = {
                    "reference_error_max_deg": [],
                    "source_keyframe_error_max_deg": [],
                    "max_acceleration_deg_s2": [],
                    "max_jerk_deg_s3": [],
                    "solve_time_sec": [],
                }
                causal_worst: dict[str, dict[str, float | int]] = {}
                for report in reports:
                    causal = (
                        report["sides"][side]
                        .get(
                            "causal_projection_by_lookahead_source_steps",
                            {},
                        )
                        .get(str(lookahead))
                    )
                    if causal is None:
                        continue
                    extracted = {
                        "reference_error_max_deg": causal[
                            "reference_error_deg"
                        ]["max"],
                        "source_keyframe_error_max_deg": causal[
                            "source_keyframe_error_deg"
                        ]["max"],
                        "max_acceleration_deg_s2": causal["dynamics"][
                            "max_acceleration_deg_s2"
                        ],
                        "max_jerk_deg_s3": causal["dynamics"][
                            "max_jerk_deg_s3"
                        ],
                        "solve_time_sec": causal["solve_time_sec"],
                    }
                    for key, value in extracted.items():
                        normalized = float(value)
                        metric_values[key].append(normalized)
                        if (
                            key not in causal_worst
                            or normalized
                            > float(causal_worst[key]["value"])
                        ):
                            causal_worst[key] = {
                                "episode_index": int(
                                    report["episode_index"]
                                ),
                                "value": normalized,
                            }
                causal_aggregate[str(lookahead)] = {
                    "metrics": {
                        key: _summary(values)
                        for key, values in metric_values.items()
                    },
                    "worst_episode": causal_worst,
                }
            aggregate["sides"][side][
                "causal_projection_by_lookahead_source_steps"
            ] = causal_aggregate
    kinematic_lookaheads = sorted(
        {
            int(value)
            for report in reports
            for value in report.get(
                "kinematic_validation_by_lookahead_source_steps",
                {},
            ).keys()
        }
    )
    if kinematic_lookaheads:
        aggregate["kinematic_validation_by_lookahead_source_steps"] = {}
        for lookahead in kinematic_lookaheads:
            entries = [
                (
                    int(report["episode_index"]),
                    report[
                        "kinematic_validation_by_lookahead_source_steps"
                    ][str(lookahead)],
                )
                for report in reports
                if str(lookahead)
                in report.get(
                    "kinematic_validation_by_lookahead_source_steps",
                    {},
                )
            ]
            aggregate_entry: dict[str, Any] = {
                "checked_frames": sum(
                    int(entry["checked_frames"]) for _, entry in entries
                ),
                "endpoint_collision_frames": sum(
                    int(entry["endpoint_collision_frames"])
                    for _, entry in entries
                ),
                "checked_intervals": sum(
                    int(entry["checked_intervals"]) for _, entry in entries
                ),
                "sampled_segment_collision_intervals": sum(
                    int(entry["sampled_segment_collision_intervals"])
                    for _, entry in entries
                ),
                "continuous_segment_collision_certified": False,
                "endpoint_collision_examples": [
                    {
                        "episode_index": episode_index,
                        **example,
                    }
                    for episode_index, entry in entries
                    for example in entry["endpoint_collision_examples"]
                ][:10],
                "sampled_segment_collision_examples": [
                    {
                        "episode_index": episode_index,
                        **example,
                    }
                    for episode_index, entry in entries
                    for example in entry[
                        "sampled_segment_collision_examples"
                    ]
                ][:10],
                "sides": {},
            }
            for side in _SIDES:
                for metric in (
                    "translation_error_m",
                    "rotation_error_deg",
                ):
                    values = [
                        float(entry["sides"][side][metric]["max"])
                        for _, entry in entries
                    ]
                    aggregate_entry["sides"].setdefault(side, {})[
                        metric
                    ] = _summary(values)
                    if values:
                        worst_index = int(np.argmax(values))
                        aggregate_entry["sides"][side][
                            f"{metric}_worst_episode"
                        ] = {
                            "episode_index": entries[worst_index][0],
                            "value": values[worst_index],
                        }
            aggregate[
                "kinematic_validation_by_lookahead_source_steps"
            ][str(lookahead)] = aggregate_entry
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only RM65 qref/qcmd dynamics probe. No device access, "
            "motion, or dispatch."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--jerk-ramp-sec",
        type=float,
        default=0.10,
        help=(
            "Derive per-joint jerk limits as max_acceleration/ramp_sec. "
            "Default: 0.10."
        ),
    )
    parser.add_argument(
        "--no-projection",
        action="store_true",
        help="Only report recorded/reference dynamics; skip the offline QP.",
    )
    parser.add_argument(
        "--causal-lookahead-source-steps",
        type=int,
        action="append",
        help=(
            "Also run a source-rate receding-horizon 120 Hz governor with "
            "this many future 15 Hz intervals. Repeat to compare values."
        ),
    )
    parser.add_argument(
        "--kinematic-validation",
        action="store_true",
        help=(
            "Also compute FK error, endpoint collision and three-point "
            "sampled segment collision for causal projections."
        ),
    )
    parser.add_argument("--left-tool-profile", default=_LEFT_TOOL)
    parser.add_argument("--right-tool-profile", default=_RIGHT_TOOL)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.jerk_ramp_sec) or args.jerk_ramp_sec <= 0.0:
        raise ValueError("--jerk-ramp-sec must be finite and positive.")
    causal_lookahead_source_steps = tuple(
        sorted(set(args.causal_lookahead_source_steps or ()))
    )
    if any(value < 1 for value in causal_lookahead_source_steps):
        raise ValueError(
            "--causal-lookahead-source-steps values must be positive."
        )
    root = Path(args.dataset_root).expanduser().resolve()
    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    by_index = {
        int(_EPISODE_PATTERN.search(path.name).group(1)): path
        for path in files
        if _EPISODE_PATTERN.search(path.name)
    }
    if args.all_episodes and args.episode:
        raise ValueError("--all-episodes and --episode are mutually exclusive.")
    indices = sorted(by_index) if args.all_episodes else list(args.episode or [])
    if not indices:
        raise ValueError("provide --episode or --all-episodes.")
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"dataset is missing episodes {missing}.")
    reports = [
        _episode_report(
            by_index[index],
            jerk_ramp_sec=float(args.jerk_ramp_sec),
            run_projection=not bool(args.no_projection),
            causal_lookahead_source_steps=causal_lookahead_source_steps,
            kinematic_validation=bool(args.kinematic_validation),
            tool_profiles={
                side: str(getattr(args, f"{side}_tool_profile"))
                for side in _SIDES
            },
        )
        for index in indices
    ]
    if not args.aggregate_only:
        for report in reports:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(
        json.dumps(
            _aggregate(reports),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
