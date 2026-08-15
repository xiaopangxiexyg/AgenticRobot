"""Offline RM65 official-IK path-lift and execution-layer comparison probe.

The vendor algorithm library is initialized without a robot handle.  Official
all-solution candidates are compared as a Pink soft anchor, a causal
source-rate branch path, a 60 Hz exact projection and a target for the existing
qcmd planner.  The vendor continuous-teleoperation solver is also available as
an isolated baseline.  Nothing in this file opens a device or dispatches a
command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from ctypes import c_float
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import (
    rm_Mat_t,
    rm_force_type_e,
    rm_robot_arm_model_e,
)
from hardware.rm65.Robotic_Arm.rm_robot_interface import Algo
from robot_motion.position_control import (
    AccelerationLimitedJointTrajectoryPlanner,
    JointHorizonLimits,
    JointHorizonState,
    JointTrajectoryLimits,
    plan_jerk_limited_joint_horizon,
)
from robot_motion.replay.trajectory import ReplayGroupSpec
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
)
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.pose_lift_profile import (
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
)
from robot_motion.rm65.position_collision import (
    DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
)
from robot_motion.rm65.source_ik_selection import (
    select_nearest_collision_safe_rm65_ik_pair,
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

from todos.probes.rm65_official_all_ik_probe import (
    _LEFT_TOOL,
    _RIGHT_TOOL,
    _SIDES,
    _pose_error,
    _solve_valid_candidates,
    _summary,
)


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
            key="official_branch_anchor_probe",
            start_index=0,
            end_index=-1,
            episode_range=str(episode_index),
        ),
        balanced_episode_count=1,
        require_arm_joint_targets=True,
    )


def _continuity_segment_ranges(
    dataset_root: Path,
    episode_index: int,
    *,
    expected_frame_count: int,
) -> list[tuple[int, int]]:
    candidates = sorted(
        dataset_root.glob(
            f"data/chunk-*/episode_{int(episode_index):06d}.parquet"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one parquet file for "
            f"episode {episode_index}, found {len(candidates)}."
        )
    table = pq.read_table(
        candidates[0],
        columns=["collection.continuity_segment_index"],
    )
    segment_ids = np.asarray(
        table["collection.continuity_segment_index"].to_numpy(),
        dtype=int,
    )
    if segment_ids.shape != (int(expected_frame_count),):
        raise ValueError(
            "continuity segment row count does not match replay group: "
            f"{segment_ids.size} != {expected_frame_count}."
        )
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                segment_ids[1:] != segment_ids[:-1],
            )
        )
    )
    ends = np.concatenate((starts[1:], np.array([segment_ids.size])))
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _clip_numeric_absolute_limit_noise(value: Any, *, name: str) -> np.ndarray:
    qpos = np.asarray(value, dtype=float).reshape(-1).copy()
    clipped = np.clip(
        qpos,
        np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float),
        np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float),
    )
    if float(np.max(np.abs(clipped - qpos))) > 1e-7:
        raise ValueError(f"{name} is materially outside RM65 absolute limits.")
    return clipped


def _build_planner(
    endpoint_to_tool: dict[str, np.ndarray],
    *,
    anchor_cost: float,
) -> RM65PositionCollisionPlanner:
    planner = RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            pose_lift_profile=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )
    if anchor_cost > 0.0:
        planner.solver.configure_analytic_branch_anchor_task(cost=anchor_cost)
    return planner


def _select_anchor_pair(
    *,
    planner: RM65PositionCollisionPlanner,
    current: dict[str, np.ndarray],
    previous_anchor: dict[str, np.ndarray],
    candidate_lists: dict[str, list[np.ndarray]],
) -> tuple[dict[str, np.ndarray], int, bool]:
    receipt = select_nearest_collision_safe_rm65_ik_pair(
        candidate_lists,
        anchor_qpos_by_side=current,
        validate_pair=lambda pair: planner.validate_qpos_reference(
            pair,
            source="official_all_ik_soft_branch_anchor_probe",
        ),
    )
    if not receipt.accepted or receipt.qpos_by_side is None:
        return (
            {
                side: np.asarray(previous_anchor[side], dtype=float).copy()
                for side in _SIDES
            },
            int(receipt.collision_rejection_count),
            False,
        )
    return (
        {
            side: np.asarray(receipt.qpos_by_side[side], dtype=float).copy()
            for side in _SIDES
        },
        int(receipt.collision_rejection_count),
        True,
    )


def _evaluate_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
    anchor_cost: float,
) -> dict[str, Any]:
    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=anchor_cost)
    current = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    previous_anchor = {
        side: np.asarray(current[side], dtype=float).copy() for side in _SIDES
    }
    planner.reset(current)
    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    per_side = {
        side: {
            "anchor_missing_frames": 0,
            "anchor_sdk_latency_us": [],
            "joint_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "jerk_max_deg_s3": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    hold_frames = 0
    collision_rejections = 0
    no_collision_safe_anchor_frames = 0
    plan_latency_ms: list[float] = []

    for reference in references:
        truth = _runtime_truth(group, reference)
        endpoint_targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=current[side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            per_side[side]["anchor_sdk_latency_us"].append(latency_us)
            if result_code != 0 or not candidates:
                per_side[side]["anchor_missing_frames"] += 1
                candidates = []
            candidate_lists[side] = candidates

        anchor, rejected, anchor_safe = _select_anchor_pair(
            planner=planner,
            current=current,
            previous_anchor=previous_anchor,
            candidate_lists=candidate_lists,
        )
        collision_rejections += int(rejected)
        if not anchor_safe:
            no_collision_safe_anchor_frames += 1
        previous_anchor = {
            side: np.asarray(anchor[side], dtype=float).copy()
            for side in _SIDES
        }
        if anchor_cost > 0.0:
            full_anchor = planner._embed_arm_qpos(  # noqa: SLF001 - offline probe
                planner._solver_current_q(),  # noqa: SLF001 - offline probe
                previous_anchor,
            )
            planner.solver.set_analytic_branch_anchor_q(full_anchor)

        started = time.perf_counter()
        result = planner.plan_pose_reference(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
            source="official_all_ik_soft_branch_anchor_probe",
        )
        plan_latency_ms.append((time.perf_counter() - started) * 1000.0)
        if result.hold:
            hold_frames += 1
        next_q = {
            side: np.asarray(result.q_ref_by_side[side], dtype=float).copy()
            for side in _SIDES
        }
        diagnostics = planner.analyze_configuration(next_q)
        for side in _SIDES:
            velocity = (next_q[side] - current[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            joint_error = np.degrees(next_q[side] - truth[side])
            translation_error, rotation_error = _pose_error(
                result.realized_endpoint_pose_world_by_side[side],
                result.target_endpoint_pose_world_by_side[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(np.max(np.abs(joint_error)))
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            previous_velocity[side] = velocity
        current = next_q

    return {
        "episode_index": int(episode_index),
        "profile": (
            "v4_baseline"
            if anchor_cost <= 0.0
            else "v4_plus_official_all_ik_soft_branch_anchor"
        ),
        "anchor_cost": float(anchor_cost),
        "runtime_reference_count": len(references),
        "hold_frames": int(hold_frames),
        "collision_rejections": int(collision_rejections),
        "no_collision_safe_anchor_frames": int(no_collision_safe_anchor_frames),
        "plan_latency_ms": _summary(plan_latency_ms),
        "sides": {
            side: {
                "anchor_missing_frames": int(metrics["anchor_missing_frames"]),
                **{
                    key: _summary(value)
                    for key, value in metrics.items()
                    if isinstance(value, list)
                },
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_joint_governed_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Track an exact all-IK branch with the existing joint governor.

    When an interpolated Pose has no project-valid exact IK, the target is
    frozen at the current command so inherited velocity decelerates under the
    same acceleration contract.  This deliberately tests the non-convex
    alternative to Cartesian straight-line alpha scaling.
    """

    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    current = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    previous_anchor = {
        side: np.asarray(current[side], dtype=float).copy() for side in _SIDES
    }
    planner.reset(current)
    trajectory = {
        side: AccelerationLimitedJointTrajectoryPlanner(
            JointTrajectoryLimits(
                max_velocity_rad_s=np.asarray(
                    DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
                    dtype=float,
                ),
                max_acceleration_rad_s2=np.asarray(
                    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
                    dtype=float,
                ),
                max_command_step_rad=np.full(6, np.deg2rad(0.675)),
            ),
            velocity_ramp_sec=0.10,
            target_velocity_filter_tau_sec=0.03,
            reference_period_sec=1.0 / 60.0,
        )
        for side in _SIDES
    }
    for side in _SIDES:
        trajectory[side].reset(current[side], monotonic_time=0.0)

    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    per_side = {
        side: {
            "exact_target_missing_frames": 0,
            "joint_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    collision_rejections = 0
    held_for_missing_exact_frames = 0
    command_collision_rejections = 0
    now = 0.0

    for reference in references:
        now += 1.0 / 60.0
        truth = _runtime_truth(group, reference)
        endpoint_targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        exact_available = True
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=previous_anchor[side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            if result_code != 0 or not candidates:
                per_side[side]["exact_target_missing_frames"] += 1
                exact_available = False
                candidates = []
            candidate_lists[side] = candidates

        if exact_available:
            anchor, rejected, anchor_safe = _select_anchor_pair(
                planner=planner,
                current=previous_anchor,
                previous_anchor=previous_anchor,
                candidate_lists=candidate_lists,
            )
            collision_rejections += int(rejected)
            exact_available = bool(anchor_safe)
            if anchor_safe:
                previous_anchor = {
                    side: np.asarray(anchor[side], dtype=float).copy()
                    for side in _SIDES
                }
        if not exact_available:
            held_for_missing_exact_frames += 1

        target = (
            previous_anchor
            if exact_available
            else {
                side: np.asarray(current[side], dtype=float).copy()
                for side in _SIDES
            }
        )
        proposals = {}
        for side in _SIDES:
            trajectory[side].accept_target(
                target[side],
                sequence=int(reference.sequence),
                published_monotonic_time=now,
                monotonic_time=now,
                mode="tracking" if exact_available else "degraded",
                reason=(
                    "official_exact_branch_target"
                    if exact_available
                    else "no_project_valid_exact_ik"
                ),
            )
            proposals[side] = trajectory[side].propose(monotonic_time=now)
        proposed_q = {
            side: np.asarray(proposals[side].q_command, dtype=float).copy()
            for side in _SIDES
        }
        validation = planner.validate_qpos_reference(
            proposed_q,
            source="official_all_ik_joint_governor_probe",
        )
        if validation.safe:
            admission = planner.admit_qpos_reference(
                proposed_q,
                source="official_all_ik_joint_governor_probe",
            )
            if not admission.safe:
                raise RuntimeError("validated joint-governor proposal was not admitted.")
            for side in _SIDES:
                trajectory[side].commit(
                    proposals[side],
                    accepted_monotonic_time=now,
                )
            next_q = proposed_q
        else:
            command_collision_rejections += 1
            for side in _SIDES:
                trajectory[side].reject(proposals[side])
            next_q = {
                side: np.asarray(current[side], dtype=float).copy()
                for side in _SIDES
            }

        diagnostics = planner.analyze_configuration(next_q)
        target_endpoint_world = planner.resolve_target_endpoint_pose_world(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = planner.observe_endpoint_poses_world(next_q)
        for side in _SIDES:
            velocity = (next_q[side] - current[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            joint_error = np.degrees(next_q[side] - truth[side])
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(np.max(np.abs(joint_error)))
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            previous_velocity[side] = velocity
        current = next_q

    return {
        "episode_index": int(episode_index),
        "profile": "official_all_ik_plus_existing_joint_governor",
        "runtime_reference_count": len(references),
        "collision_rejections": int(collision_rejections),
        "held_for_missing_exact_frames": int(held_for_missing_exact_frames),
        "command_collision_rejections": int(command_collision_rejections),
        "sides": {
            side: {
                "exact_target_missing_frames": int(
                    metrics["exact_target_missing_frames"]
                ),
                **{
                    key: _summary(value)
                    for key, value in metrics.items()
                    if isinstance(value, list)
                },
            }
            for side, metrics in per_side.items()
        },
    }


def _quintic_hermite_sample(
    q0: np.ndarray,
    q1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    *,
    duration_sec: float,
    alpha: float,
) -> np.ndarray:
    duration = float(duration_sec)
    s = float(alpha)
    c0 = q0
    c1 = v0 * duration
    c2 = 0.5 * a0 * duration**2
    rhs = np.stack(
        (
            q1 - c0 - c1 - c2,
            v1 * duration - c1 - 2.0 * c2,
            a1 * duration**2 - 2.0 * c2,
        ),
        axis=0,
    )
    c3, c4, c5 = np.linalg.solve(
        np.array(
            (
                (1.0, 1.0, 1.0),
                (3.0, 4.0, 5.0),
                (6.0, 12.0, 20.0),
            ),
            dtype=float,
        ),
        rhs,
    )
    return (
        c0
        + c1 * s
        + c2 * s**2
        + c3 * s**3
        + c4 * s**4
        + c5 * s**5
    )


def _source_derivatives_by_continuity_segment(
    source: np.ndarray,
    continuity_ranges: list[tuple[int, int]],
    *,
    source_dt_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(source)
    acceleration = np.zeros_like(source)
    for start, end in continuity_ranges:
        values = source[start:end]
        if values.shape[0] == 1:
            continue
        edge_order = 2 if values.shape[0] >= 3 else 1
        segment_velocity = np.gradient(
            values,
            source_dt_sec,
            axis=0,
            edge_order=edge_order,
        )
        velocity[start:end] = segment_velocity
        if values.shape[0] >= 3:
            acceleration[start:end] = np.gradient(
                segment_velocity,
                source_dt_sec,
                axis=0,
                edge_order=edge_order,
            )
    return velocity, acceleration


def _evaluate_source_all_ik_joint_interpolation(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
    interpolation: str = "linear",
    quintic_blend: float = 1.0,
) -> dict[str, Any]:
    """Solve only materialized source Poses, then interpolate in joint space."""

    if interpolation not in {"linear", "quintic"}:
        raise ValueError("interpolation must be 'linear' or 'quintic'.")
    blend = float(quintic_blend)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("quintic_blend must be finite and in [0, 1].")
    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    continuity_ranges = _continuity_segment_ranges(
        dataset_root,
        episode_index,
        expected_frame_count=int(group.frame_count),
    )
    source_segment_id = np.empty(int(group.frame_count), dtype=int)
    for segment_ordinal, (start, end) in enumerate(continuity_ranges):
        source_segment_id[start:end] = segment_ordinal
    continuity_starts = {start for start, _end in continuity_ranges}
    source_q: list[dict[str, np.ndarray]] = []
    source_failures = 0
    source_collision_rejections = 0
    for frame_index in range(int(group.frame_count)):
        if frame_index in continuity_starts:
            restart = {
                side: _clip_numeric_absolute_limit_noise(
                    value,
                    name=(
                        f"episode {episode_index} frame {frame_index} "
                        f"{side} restart qpos"
                    ),
                )
                for side, value in _truth_at(group, frame_index).items()
            }
            planner.reset(restart)
            source_q.append(restart)
            continue
        tcp_targets = _pose_at(group, frame_index)
        endpoint_targets = {
            side: np.asarray(tcp_targets[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=source_q[-1][side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            if result_code != 0 or not candidates:
                candidates = []
            candidate_lists[side] = candidates
        if any(not candidate_lists[side] for side in _SIDES):
            source_failures += 1
            source_q.append(
                {
                    side: np.asarray(source_q[-1][side], dtype=float).copy()
                    for side in _SIDES
                }
            )
            continue
        selected, rejected, safe = _select_anchor_pair(
            planner=planner,
            current=source_q[-1],
            previous_anchor=source_q[-1],
            candidate_lists=candidate_lists,
        )
        source_collision_rejections += int(rejected)
        if not safe:
            source_failures += 1
            selected = {
                side: np.asarray(source_q[-1][side], dtype=float).copy()
                for side in _SIDES
            }
        source_q.append(selected)

    first = source_q[0]
    source_matrix = {
        side: np.asarray(
            [value[side] for value in source_q],
            dtype=float,
        )
        for side in _SIDES
    }
    source_velocity = {}
    source_acceleration = {}
    for side in _SIDES:
        (
            source_velocity[side],
            source_acceleration[side],
        ) = _source_derivatives_by_continuity_segment(
            source_matrix[side],
            continuity_ranges,
            source_dt_sec=1.0 / 15.0,
        )
    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    current = {
        side: np.asarray(first[side], dtype=float).copy() for side in _SIDES
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    previous_acceleration = {side: np.zeros(6, dtype=float) for side in _SIDES}
    runtime_collision_failures = 0
    sampled_interval_collision_failures = 0
    quintic_collision_fallback_frames = 0
    dynamic_violation_frames = {
        side: {
            "velocity": 0,
            "acceleration": 0,
            "jerk": 0,
        }
        for side in _SIDES
    }
    max_velocity_limit = np.asarray(
        RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        dtype=float,
    )
    max_acceleration_limit = np.asarray(
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        dtype=float,
    )
    max_jerk_limit = max_acceleration_limit / 0.10
    per_side = {
        side: {
            "joint_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "jerk_max_deg_s3": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    for reference in references:
        alpha = float(reference.interpolation_alpha)
        left_index = int(reference.left_source_index)
        right_index = int(reference.right_source_index)
        continuity_break = bool(
            source_segment_id[left_index] != source_segment_id[right_index]
        )
        linear_candidate = {
            side: (
                (1.0 - alpha) * source_matrix[side][left_index]
                + alpha * source_matrix[side][right_index]
            )
            for side in _SIDES
        }
        if continuity_break:
            candidate = {
                side: source_matrix[side][right_index].copy()
                for side in _SIDES
            }
            current = {
                side: value.copy() for side, value in candidate.items()
            }
            previous_velocity = {
                side: np.zeros(6, dtype=float) for side in _SIDES
            }
            previous_acceleration = {
                side: np.zeros(6, dtype=float) for side in _SIDES
            }
            # A continuity cut has no physical interpolation interval.  The
            # first sample of the new segment is an initialization anchor, not
            # a command transition or Pose-tracking observation.
            continue
        elif interpolation == "quintic":
            candidate = {}
            for side in _SIDES:
                quintic_candidate = _quintic_hermite_sample(
                    source_matrix[side][left_index],
                    source_matrix[side][right_index],
                    source_velocity[side][left_index],
                    source_velocity[side][right_index],
                    source_acceleration[side][left_index],
                    source_acceleration[side][right_index],
                    duration_sec=1.0 / 15.0,
                    alpha=alpha,
                )
                candidate[side] = (
                    linear_candidate[side]
                    + blend * (quintic_candidate - linear_candidate[side])
                )
        else:
            candidate = linear_candidate
        validation = planner.validate_qpos_reference(
            candidate,
            source="official_source_all_ik_joint_interpolation_probe",
        )
        if interpolation == "quintic" and not validation.safe:
            candidate = {
                side: value.copy()
                for side, value in linear_candidate.items()
            }
            validation = planner.validate_qpos_reference(
                candidate,
                source=(
                    "official_source_all_ik_quintic_collision_fallback_probe"
                ),
            )
            quintic_collision_fallback_frames += 1
        if not validation.safe:
            runtime_collision_failures += 1
        interval_unsafe = False
        for sample_alpha in (0.25, 0.5, 0.75):
            sample = {
                side: (
                    (1.0 - sample_alpha) * current[side]
                    + sample_alpha * candidate[side]
                )
                for side in _SIDES
            }
            if not planner.validate_qpos_reference(
                sample,
                source=(
                    "official_source_all_ik_joint_interpolation_interval_probe"
                ),
            ).safe:
                interval_unsafe = True
                break
        sampled_interval_collision_failures += int(interval_unsafe)
        truth = _runtime_truth(group, reference)
        diagnostics = planner.analyze_configuration(candidate)
        target_endpoint_world = planner.resolve_target_endpoint_pose_world(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = planner.observe_endpoint_poses_world(candidate)
        for side in _SIDES:
            velocity = (candidate[side] - current[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            jerk = (acceleration - previous_acceleration[side]) * 60.0
            dynamic_violation_frames[side]["velocity"] += int(
                np.any(np.abs(velocity) > max_velocity_limit + 1e-9)
            )
            dynamic_violation_frames[side]["acceleration"] += int(
                np.any(
                    np.abs(acceleration)
                    > max_acceleration_limit + 1e-9
                )
            )
            dynamic_violation_frames[side]["jerk"] += int(
                np.any(np.abs(jerk) > max_jerk_limit + 1e-9)
            )
            joint_error = np.degrees(candidate[side] - truth[side])
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(np.max(np.abs(joint_error)))
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["jerk_max_deg_s3"].append(
                float(np.max(np.abs(np.degrees(jerk))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            if not continuity_break:
                previous_velocity[side] = velocity
                previous_acceleration[side] = acceleration
        current = {
            side: np.asarray(candidate[side], dtype=float).copy()
            for side in _SIDES
        }

    return {
        "episode_index": int(episode_index),
        "profile": (
            "official_source_all_ik_then_joint_interpolation_60hz"
            if interpolation == "linear"
            else "official_source_all_ik_then_quintic_joint_path_60hz"
        ),
        "interpolation": interpolation,
        "quintic_blend": blend if interpolation == "quintic" else 0.0,
        "continuity_segment_count": len(continuity_ranges),
        "source_frame_count": int(group.frame_count),
        "runtime_reference_count": len(references),
        "source_failures": int(source_failures),
        "source_collision_rejections": int(source_collision_rejections),
        "runtime_collision_failures": int(runtime_collision_failures),
        "sampled_interval_collision_failures": int(
            sampled_interval_collision_failures
        ),
        "quintic_collision_fallback_frames": int(
            quintic_collision_fallback_frames
        ),
        "dynamic_violation_frames": dynamic_violation_frames,
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_source_path_through_jerk_horizon_governor(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
    lookahead_source_steps: int = 2,
    jerk_ramp_sec: float = 0.10,
    reference_interpolation: str = "linear",
) -> dict[str, Any]:
    """Run source all-IK through the shared third-order horizon governor."""

    if reference_interpolation not in {"linear", "quintic"}:
        raise ValueError(
            "reference_interpolation must be 'linear' or 'quintic'."
        )
    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    continuity_ranges = _continuity_segment_ranges(
        dataset_root,
        episode_index,
        expected_frame_count=int(group.frame_count),
    )
    continuity_starts = {start for start, _end in continuity_ranges}
    source_q: list[dict[str, np.ndarray]] = []
    source_failures = 0
    source_collision_rejections = 0
    source_pose = [_pose_at(group, index) for index in range(int(group.frame_count))]
    for frame_index in range(int(group.frame_count)):
        if frame_index in continuity_starts:
            restart = {
                side: _clip_numeric_absolute_limit_noise(
                    value,
                    name=(
                        f"episode {episode_index} frame {frame_index} "
                        f"{side} restart qpos"
                    ),
                )
                for side, value in _truth_at(group, frame_index).items()
            }
            planner.reset(restart)
            source_q.append(restart)
            continue
        endpoint_targets = {
            side: np.asarray(source_pose[frame_index][side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=source_q[-1][side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            candidate_lists[side] = candidates if result_code == 0 else []
        if any(not candidate_lists[side] for side in _SIDES):
            source_failures += 1
            source_q.append(
                {side: source_q[-1][side].copy() for side in _SIDES}
            )
            continue
        selected, rejected, safe = _select_anchor_pair(
            planner=planner,
            current=source_q[-1],
            previous_anchor=source_q[-1],
            candidate_lists=candidate_lists,
        )
        source_collision_rejections += int(rejected)
        if not safe:
            source_failures += 1
            selected = {side: source_q[-1][side].copy() for side in _SIDES}
        source_q.append(selected)

    commands_per_source = 8
    source_truth = [_truth_at(group, index) for index in range(int(group.frame_count))]
    source_matrix = {
        side: np.asarray(
            [value[side] for value in source_q],
            dtype=float,
        )
        for side in _SIDES
    }
    source_velocity = {}
    source_acceleration = {}
    for side in _SIDES:
        (
            source_velocity[side],
            source_acceleration[side],
        ) = _source_derivatives_by_continuity_segment(
            source_matrix[side],
            continuity_ranges,
            source_dt_sec=1.0 / 15.0,
        )
    q_reference_segments: dict[str, list[np.ndarray]] = {
        side: [] for side in _SIDES
    }
    q_truth_segments: dict[str, list[np.ndarray]] = {
        side: [] for side in _SIDES
    }
    desired_pose_segments: list[list[dict[str, np.ndarray]]] = []
    for source_start, source_end in continuity_ranges:
        segment_reference = {side: [] for side in _SIDES}
        segment_truth = {side: [] for side in _SIDES}
        segment_pose: list[dict[str, np.ndarray]] = []
        for source_index in range(source_start, source_end - 1):
            for step in range(commands_per_source):
                alpha = step / float(commands_per_source)
                segment_pose.append(
                    {
                        side: interpolate_pose(
                            source_pose[source_index][side],
                            source_pose[source_index + 1][side],
                            alpha,
                        )
                        for side in _SIDES
                    }
                )
                for side in _SIDES:
                    if reference_interpolation == "quintic":
                        reference_q = _quintic_hermite_sample(
                            source_matrix[side][source_index],
                            source_matrix[side][source_index + 1],
                            source_velocity[side][source_index],
                            source_velocity[side][source_index + 1],
                            source_acceleration[side][source_index],
                            source_acceleration[side][source_index + 1],
                            duration_sec=1.0 / 15.0,
                            alpha=alpha,
                        )
                    else:
                        reference_q = (
                            (1.0 - alpha) * source_q[source_index][side]
                            + alpha * source_q[source_index + 1][side]
                        )
                    segment_reference[side].append(reference_q)
                    segment_truth[side].append(
                        (1.0 - alpha) * source_truth[source_index][side]
                        + alpha * source_truth[source_index + 1][side]
                    )
        segment_pose.append(source_pose[source_end - 1])
        desired_pose_segments.append(segment_pose)
        for side in _SIDES:
            segment_reference[side].append(
                source_q[source_end - 1][side].copy()
            )
            segment_truth[side].append(
                source_truth[source_end - 1][side].copy()
            )
            q_reference_segments[side].append(
                np.asarray(segment_reference[side], dtype=float)
            )
            q_truth_segments[side].append(
                np.asarray(segment_truth[side], dtype=float)
            )
    desired_pose = [
        pose
        for segment in desired_pose_segments
        for pose in segment
    ]
    q_reference: dict[str, np.ndarray] = {}
    q_truth: dict[str, np.ndarray] = {}
    for side in _SIDES:
        q_reference[side] = np.concatenate(
            q_reference_segments[side],
            axis=0,
        )
        q_truth[side] = np.concatenate(q_truth_segments[side], axis=0)

    max_velocity = np.asarray(
        RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        dtype=float,
    )
    max_acceleration = np.asarray(
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        dtype=float,
    )
    max_jerk = max_acceleration / float(jerk_ramp_sec)
    limits = JointHorizonLimits(
        min_position_rad=RM65_ABSOLUTE_Q_MIN_RAD,
        max_position_rad=RM65_ABSOLUTE_Q_MAX_RAD,
        max_velocity_rad_s=max_velocity,
        max_acceleration_rad_s2=max_acceleration,
        max_jerk_rad_s3=max_jerk,
    )
    q_command_segments: dict[str, list[np.ndarray]] = {
        side: [] for side in _SIDES
    }
    solve_time_sec = 0.0
    segment_command_ranges: list[tuple[int, int]] = []
    command_offset = 0
    for segment_ordinal in range(len(continuity_ranges)):
        segment_length = int(
            q_reference_segments["left"][segment_ordinal].shape[0]
        )
        segment_command_ranges.append(
            (command_offset, command_offset + segment_length)
        )
        command_offset += segment_length
        segment_commands = {
            side: np.empty_like(
                q_reference_segments[side][segment_ordinal]
            )
            for side in _SIDES
        }
        state = {}
        for side in _SIDES:
            segment_commands[side][0] = q_reference_segments[side][
                segment_ordinal
            ][0]
            state[side] = JointHorizonState(
                q_rad=segment_commands[side][0],
                qd_rad_s=np.zeros(6),
                qdd_rad_s2=np.zeros(6),
            )
        final_segment_index = segment_length - 1
        start = 0
        while start < final_segment_index:
            horizon_end = min(
                final_segment_index,
                start + int(lookahead_source_steps) * commands_per_source,
            )
            commit_count = min(
                commands_per_source,
                horizon_end - start,
            )
            for side in _SIDES:
                try:
                    plan = plan_jerk_limited_joint_horizon(
                        q_reference_segments[side][segment_ordinal][
                            start : horizon_end + 1
                        ],
                        initial_state=state[side],
                        limits=limits,
                        dt_sec=1.0 / 120.0,
                        commit_count=commit_count,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "source-path jerk governor failed "
                        f"episode={episode_index} side={side} "
                        f"segment={segment_ordinal} start={start} "
                        f"horizon_end={horizon_end} "
                        f"q_deg={np.degrees(state[side].q_rad).tolist()} "
                        f"qd_deg_s={np.degrees(state[side].qd_rad_s).tolist()} "
                        f"qdd_deg_s2="
                        f"{np.degrees(state[side].qdd_rad_s2).tolist()}"
                    ) from exc
                segment_commands[side][
                    start + 1 : start + commit_count + 1
                ] = plan.q_control_cut_rad
                solve_time_sec += sum(
                    float(receipt["solve_time_sec"])
                    for receipt in plan.metadata["solver_receipts"]
                )
                state[side] = JointHorizonState(
                    q_rad=plan.q_path_rad[commit_count],
                    qd_rad_s=plan.qd_path_rad_s[commit_count],
                    qdd_rad_s2=plan.qdd_path_rad_s2[commit_count],
                )
            start += commit_count
        for side in _SIDES:
            q_command_segments[side].append(segment_commands[side])
    q_command = {
        side: np.concatenate(q_command_segments[side], axis=0)
        for side in _SIDES
    }
    final_index = int(q_reference["left"].shape[0] - 1)
    segment_command_starts = {
        start for start, _end in segment_command_ranges[1:]
    }

    endpoint_failures = 0
    endpoint_failure_details: list[dict[str, Any]] = []
    sampled_interval_failures = 0
    per_side = {
        side: {
            "joint_max_deg": [],
            "qcmd_qref_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    for index in range(final_index + 1):
        pair = {side: q_command[side][index] for side in _SIDES}
        validation = planner.validate_qpos_reference(
            pair,
            source="official_source_path_jerk_horizon_governor_probe",
        )
        endpoint_failures += int(not validation.safe)
        if not validation.safe:
            endpoint_failure_details.append(
                {
                    "frame_index": int(index),
                    "reason": validation.reason,
                    "collision_distance_m": validation.collision_distance_m,
                    "qcmd_deg": {
                        side: np.degrees(pair[side]).tolist()
                        for side in _SIDES
                    },
                    "qref_deg": {
                        side: np.degrees(q_reference[side][index]).tolist()
                        for side in _SIDES
                    },
                }
            )
        target_endpoint_world = planner.resolve_target_endpoint_pose_world(
            desired_pose[index],
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = planner.observe_endpoint_poses_world(pair)
        diagnostics = planner.analyze_configuration(pair)
        for side in _SIDES:
            translation, rotation = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(pair[side] - q_truth[side][index]))
                    )
                )
            )
            per_side[side]["qcmd_qref_max_deg"].append(
                float(
                    np.max(
                        np.abs(
                            np.degrees(pair[side] - q_reference[side][index])
                        )
                    )
                )
            )
            per_side[side]["target_translation_error_m"].append(translation)
            per_side[side]["target_rotation_error_deg"].append(rotation)
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
        if index == 0 or index in segment_command_starts:
            continue
        interval_unsafe = False
        for alpha in (0.25, 0.5, 0.75):
            sample = {
                side: (
                    (1.0 - alpha) * q_command[side][index - 1]
                    + alpha * q_command[side][index]
                )
                for side in _SIDES
            }
            if not planner.validate_qpos_reference(
                sample,
                source="official_source_path_jerk_horizon_interval_probe",
            ).safe:
                interval_unsafe = True
                break
        sampled_interval_failures += int(interval_unsafe)

    dt = 1.0 / 120.0
    for side in _SIDES:
        velocity_parts = []
        acceleration_parts = []
        jerk_parts = []
        for segment in q_command_segments[side]:
            velocity = np.diff(segment, axis=0) / dt
            acceleration = np.diff(velocity, axis=0) / dt
            jerk = np.diff(acceleration, axis=0) / dt
            if velocity.size:
                velocity_parts.append(velocity)
            if acceleration.size:
                acceleration_parts.append(acceleration)
            if jerk.size:
                jerk_parts.append(jerk)
        per_side[side]["velocity_max_deg_s"] = [
            float(
                max(
                    np.max(np.abs(np.degrees(value)))
                    for value in velocity_parts
                )
            )
        ]
        per_side[side]["acceleration_max_deg_s2"] = [
            float(
                max(
                    np.max(np.abs(np.degrees(value)))
                    for value in acceleration_parts
                )
            )
        ]
        per_side[side]["jerk_max_deg_s3"] = [
            float(
                max(
                    np.max(np.abs(np.degrees(value)))
                    for value in jerk_parts
                )
            )
        ]
    max_qcmd_qref_detail = {}
    for side in _SIDES:
        error_deg = np.abs(np.degrees(q_command[side] - q_reference[side]))
        flat_index = int(np.argmax(error_deg))
        frame_index, joint_index = np.unravel_index(
            flat_index,
            error_deg.shape,
        )
        max_qcmd_qref_detail[side] = {
            "frame_index": int(frame_index),
            "joint_index": int(joint_index + 1),
            "error_deg": float(error_deg[frame_index, joint_index]),
            "qcmd_deg": float(
                np.degrees(q_command[side][frame_index, joint_index])
            ),
            "qref_deg": float(
                np.degrees(q_reference[side][frame_index, joint_index])
            ),
        }
    return {
        "episode_index": int(episode_index),
        "profile": (
            "official_source_all_ik_plus_jerk_horizon_governor"
            if reference_interpolation == "linear"
            else (
                "official_source_all_ik_quintic_plus_"
                "jerk_horizon_governor"
            )
        ),
        "reference_interpolation": reference_interpolation,
        "source_failures": int(source_failures),
        "source_collision_rejections": int(source_collision_rejections),
        "endpoint_collision_failures": int(endpoint_failures),
        "endpoint_collision_failure_details": endpoint_failure_details,
        "sampled_interval_collision_failures": int(sampled_interval_failures),
        "continuous_segment_collision_certified": False,
        "max_qcmd_qref_detail": max_qcmd_qref_detail,
        "solve_time_sec": float(solve_time_sec),
        "source_frame_count": int(group.frame_count),
        "continuity_segment_count": len(continuity_ranges),
        "command_frame_count": int(final_index + 1),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_source_joint_interpolation_with_pink_residual(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Apply frame-local Pink residual correction without closed-loop drift."""

    group = _build_group(dataset_root, episode_index)
    path_planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    first = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    path_planner.reset(first)
    source_q: list[dict[str, np.ndarray]] = [first]
    source_failures = 0
    for frame_index in range(1, int(group.frame_count)):
        tcp_targets = _pose_at(group, frame_index)
        endpoint_targets = {
            side: np.asarray(tcp_targets[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=source_q[-1][side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            candidate_lists[side] = (
                candidates if result_code == 0 else []
            )
        if any(not candidate_lists[side] for side in _SIDES):
            source_failures += 1
            source_q.append(
                {
                    side: np.asarray(source_q[-1][side], dtype=float).copy()
                    for side in _SIDES
                }
            )
            continue
        selected, _rejected, safe = _select_anchor_pair(
            planner=path_planner,
            current=source_q[-1],
            previous_anchor=source_q[-1],
            candidate_lists=candidate_lists,
        )
        if not safe:
            source_failures += 1
            selected = {
                side: np.asarray(source_q[-1][side], dtype=float).copy()
                for side in _SIDES
            }
        source_q.append(selected)

    residual_planner = RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=60.0,
            solve_max_iterations=2,
            solve_min_iterations=2,
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )
    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    previous = {
        side: np.asarray(first[side], dtype=float).copy() for side in _SIDES
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    hold_frames = 0
    per_side = {
        side: {
            "joint_max_deg": [],
            "seed_correction_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    for reference in references:
        alpha = float(reference.interpolation_alpha)
        left_index = int(reference.left_source_index)
        right_index = int(reference.right_source_index)
        seed = {
            side: (
                (1.0 - alpha) * source_q[left_index][side]
                + alpha * source_q[right_index][side]
            )
            for side in _SIDES
        }
        residual_planner.reset(seed)
        result = residual_planner.plan_pose_reference(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
            source="official_source_joint_interpolation_pink_residual_probe",
        )
        if result.hold:
            hold_frames += 1
            candidate = seed
        else:
            candidate = {
                side: np.asarray(
                    result.q_ref_by_side[side],
                    dtype=float,
                ).copy()
                for side in _SIDES
            }
        truth = _runtime_truth(group, reference)
        diagnostics = residual_planner.analyze_configuration(candidate)
        target_endpoint_world = residual_planner.resolve_target_endpoint_pose_world(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = residual_planner.observe_endpoint_poses_world(
            candidate
        )
        for side in _SIDES:
            velocity = (candidate[side] - previous[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(
                    np.max(
                        np.abs(
                            np.degrees(candidate[side] - truth[side])
                        )
                    )
                )
            )
            per_side[side]["seed_correction_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(candidate[side] - seed[side]))
                    )
                )
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            previous_velocity[side] = velocity
        previous = {
            side: np.asarray(candidate[side], dtype=float).copy()
            for side in _SIDES
        }

    return {
        "episode_index": int(episode_index),
        "profile": "official_source_joint_interpolation_plus_frame_local_pink_residual",
        "source_failures": int(source_failures),
        "hold_frames": int(hold_frames),
        "runtime_reference_count": len(references),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_source_path_with_intermediate_all_ik_projection(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Project each Cartesian reference from a non-drifting source-path seed."""

    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    first = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    planner.reset(first)
    source_q: list[dict[str, np.ndarray]] = [first]
    source_failures = 0
    for frame_index in range(1, int(group.frame_count)):
        tcp_targets = _pose_at(group, frame_index)
        endpoint_targets = {
            side: np.asarray(tcp_targets[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=source_q[-1][side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            candidate_lists[side] = (
                candidates if result_code == 0 else []
            )
        if any(not candidate_lists[side] for side in _SIDES):
            source_failures += 1
            source_q.append(
                {
                    side: np.asarray(source_q[-1][side], dtype=float).copy()
                    for side in _SIDES
                }
            )
            continue
        selected, _rejected, safe = _select_anchor_pair(
            planner=planner,
            current=source_q[-1],
            previous_anchor=source_q[-1],
            candidate_lists=candidate_lists,
        )
        if not safe:
            source_failures += 1
            selected = {
                side: np.asarray(source_q[-1][side], dtype=float).copy()
                for side in _SIDES
            }
        source_q.append(selected)

    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    previous = {
        side: np.asarray(first[side], dtype=float).copy() for side in _SIDES
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    intermediate_missing_by_side = {side: 0 for side in _SIDES}
    no_collision_safe_projection_frames = 0
    per_side = {
        side: {
            "joint_max_deg": [],
            "seed_projection_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    for reference in references:
        alpha = float(reference.interpolation_alpha)
        left_index = int(reference.left_source_index)
        right_index = int(reference.right_source_index)
        seed = {
            side: (
                (1.0 - alpha) * source_q[left_index][side]
                + alpha * source_q[right_index][side]
            )
            for side in _SIDES
        }
        endpoint_targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=seed[side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            if result_code != 0 or not candidates:
                intermediate_missing_by_side[side] += 1
                candidates = []
            candidate_lists[side] = candidates
        candidate, _rejected, safe = _select_anchor_pair(
            planner=planner,
            current=seed,
            previous_anchor=seed,
            candidate_lists=candidate_lists,
        )
        if not safe:
            no_collision_safe_projection_frames += 1
            candidate = seed

        truth = _runtime_truth(group, reference)
        diagnostics = planner.analyze_configuration(candidate)
        target_endpoint_world = planner.resolve_target_endpoint_pose_world(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = planner.observe_endpoint_poses_world(candidate)
        for side in _SIDES:
            velocity = (candidate[side] - previous[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(
                    np.max(
                        np.abs(
                            np.degrees(candidate[side] - truth[side])
                        )
                    )
                )
            )
            per_side[side]["seed_projection_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(candidate[side] - seed[side]))
                    )
                )
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(rotation_error)
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            previous_velocity[side] = velocity
        previous = {
            side: np.asarray(candidate[side], dtype=float).copy()
            for side in _SIDES
        }

    return {
        "episode_index": int(episode_index),
        "profile": "official_source_path_seed_plus_intermediate_all_ik_projection",
        "source_failures": int(source_failures),
        "intermediate_missing_by_side": dict(intermediate_missing_by_side),
        "no_collision_safe_projection_frames": int(
            no_collision_safe_projection_frames
        ),
        "runtime_reference_count": len(references),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_source_path_through_downstream_planner(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run source-rate all-IK q references through the production qcmd planner.

    This keeps the ownership split explicit: official all-IK selects the
    causal source-frame branch, joint interpolation produces the 60 Hz qref,
    and the existing acceleration-limited planner alone produces the 120 Hz
    command.  The probe never treats the raw all-IK result as a hardware
    command.
    """

    group = _build_group(dataset_root, episode_index)
    collision_planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    first = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    collision_planner.reset(first)
    source_q: list[dict[str, np.ndarray]] = [first]
    source_failures = 0
    source_collision_rejections = 0
    for frame_index in range(1, int(group.frame_count)):
        tcp_targets = _pose_at(group, frame_index)
        endpoint_targets = {
            side: np.asarray(tcp_targets[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate_lists: dict[str, list[np.ndarray]] = {}
        for side in _SIDES:
            candidates, result_code, _raw_count, _latency_us = (
                _solve_valid_candidates(
                    algo=algo,
                    seed_rad=source_q[-1][side],
                    target_endpoint=endpoint_targets[side],
                )
            )
            candidate_lists[side] = (
                candidates if result_code == 0 else []
            )
        if any(not candidate_lists[side] for side in _SIDES):
            source_failures += 1
            source_q.append(
                {
                    side: np.asarray(source_q[-1][side], dtype=float).copy()
                    for side in _SIDES
                }
            )
            continue
        selected, rejected, safe = _select_anchor_pair(
            planner=collision_planner,
            current=source_q[-1],
            previous_anchor=source_q[-1],
            candidate_lists=candidate_lists,
        )
        source_collision_rejections += int(rejected)
        if not safe:
            source_failures += 1
            selected = {
                side: np.asarray(source_q[-1][side], dtype=float).copy()
                for side in _SIDES
            }
        source_q.append(selected)

    command_planners = {
        side: AccelerationLimitedJointTrajectoryPlanner(
            JointTrajectoryLimits(
                max_velocity_rad_s=np.asarray(
                    DEFAULT_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
                    dtype=float,
                ),
                max_acceleration_rad_s2=np.asarray(
                    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
                    dtype=float,
                ),
                max_command_step_rad=np.full(6, np.deg2rad(0.675)),
            ),
            velocity_ramp_sec=0.10,
            target_velocity_filter_tau_sec=0.03,
            reference_period_sec=1.0 / 60.0,
        )
        for side in _SIDES
    }
    for side in _SIDES:
        command_planners[side].reset(first[side], monotonic_time=0.0)

    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    current = {
        side: np.asarray(first[side], dtype=float).copy() for side in _SIDES
    }
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    command_collision_rejections = 0
    per_side = {
        side: {
            "joint_max_deg": [],
            "qcmd_qref_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "sigma_min": [],
        }
        for side in _SIDES
    }
    outer_time = 0.0
    inner_dt = 1.0 / 120.0
    for reference in references:
        alpha = float(reference.interpolation_alpha)
        left_index = int(reference.left_source_index)
        right_index = int(reference.right_source_index)
        qref = {
            side: (
                (1.0 - alpha) * source_q[left_index][side]
                + alpha * source_q[right_index][side]
            )
            for side in _SIDES
        }
        for side in _SIDES:
            command_planners[side].accept_target(
                qref[side],
                sequence=int(reference.sequence),
                published_monotonic_time=outer_time,
                monotonic_time=outer_time,
                mode="tracking",
                reason="official_source_path_qref",
            )

        interval_velocity: dict[str, list[float]] = {
            side: [] for side in _SIDES
        }
        interval_acceleration: dict[str, list[float]] = {
            side: [] for side in _SIDES
        }
        for inner_index in range(2):
            tick_time = outer_time + (inner_index + 1) * inner_dt
            proposals = {
                side: command_planners[side].propose(
                    monotonic_time=tick_time
                )
                for side in _SIDES
            }
            proposed = {
                side: np.asarray(
                    proposals[side].q_command,
                    dtype=float,
                ).copy()
                for side in _SIDES
            }
            validation = collision_planner.validate_qpos_reference(
                proposed,
                source="official_source_path_downstream_qcmd_probe",
            )
            if not validation.safe:
                command_collision_rejections += 1
                for side in _SIDES:
                    command_planners[side].reject(proposals[side])
                continue
            admission = collision_planner.admit_qpos_reference(
                proposed,
                source="official_source_path_downstream_qcmd_probe",
            )
            if not admission.safe:
                raise RuntimeError(
                    "validated downstream qcmd proposal was not admitted."
                )
            for side in _SIDES:
                velocity = (proposed[side] - current[side]) / inner_dt
                acceleration = (
                    velocity - previous_velocity[side]
                ) / inner_dt
                interval_velocity[side].append(
                    float(np.max(np.abs(np.degrees(velocity))))
                )
                interval_acceleration[side].append(
                    float(np.max(np.abs(np.degrees(acceleration))))
                )
                command_planners[side].commit(
                    proposals[side],
                    accepted_monotonic_time=tick_time,
                )
                previous_velocity[side] = velocity
            current = proposed

        outer_time += 1.0 / 60.0
        truth = _runtime_truth(group, reference)
        diagnostics = collision_planner.analyze_configuration(current)
        target_endpoint_world = (
            collision_planner.resolve_target_endpoint_pose_world(
                reference.pose_by_side,
                target_reference_frame="arm_base",
                target_controlled_frame="tool_tcp",
            )
        )
        realized_endpoint_world = (
            collision_planner.observe_endpoint_poses_world(current)
        )
        for side in _SIDES:
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(current[side] - truth[side]))
                    )
                )
            )
            per_side[side]["qcmd_qref_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(current[side] - qref[side]))
                    )
                )
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(
                rotation_error
            )
            per_side[side]["velocity_max_deg_s"].append(
                max(interval_velocity[side], default=0.0)
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                max(interval_acceleration[side], default=0.0)
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )

    return {
        "episode_index": int(episode_index),
        "profile": "official_source_path_plus_existing_downstream_qcmd_planner",
        "source_failures": int(source_failures),
        "source_collision_rejections": int(source_collision_rejections),
        "command_collision_rejections": int(command_collision_rejections),
        "runtime_reference_count": len(references),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def _evaluate_official_remote_ik_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    algo: Algo,
    endpoint_to_tool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate the vendor's continuous Cartesian teleoperation solver."""

    group = _build_group(dataset_root, episode_index)
    planner = _build_planner(endpoint_to_tool, anchor_cost=0.0)
    current = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in _truth_at(group, 0).items()
    }
    planner.reset(current)
    algo.rm_algo_ik_remote_init(1.0 / 60.0, 1)
    algo.rm_algo_set_error_weight([1.0] * 6)
    algo.rm_algo_set_dq_weight([1.0] * 6)
    algo.rm_algo_set_enable_limit_holdon(0)

    references = [
        reference
        for reference in _runtime_references(group, 60.0)
        if int(reference.sequence) > 0
    ]
    previous_velocity = {side: np.zeros(6, dtype=float) for side in _SIDES}
    result_codes: dict[int, int] = {}
    collision_rejections = 0
    per_side = {
        side: {
            "joint_max_deg": [],
            "target_translation_error_m": [],
            "target_rotation_error_deg": [],
            "velocity_max_deg_s": [],
            "acceleration_max_deg_s2": [],
            "sigma_min": [],
            "solve_latency_us": [],
        }
        for side in _SIDES
    }
    for reference in references:
        endpoint_targets = {
            side: np.asarray(reference.pose_by_side[side], dtype=float)
            @ np.linalg.inv(endpoint_to_tool[side])
            for side in _SIDES
        }
        candidate: dict[str, np.ndarray] = {}
        frame_ok = True
        for side in _SIDES:
            target = rm_Mat_t(
                4,
                4,
                np.asarray(endpoint_targets[side], dtype=float).tolist(),
            )
            q_out = (c_float * 6)()
            started = time.perf_counter_ns()
            code = int(
                algo.rm_algo_ik_remote(
                    target,
                    np.degrees(current[side]).tolist(),
                    q_out,
                )
            )
            elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            per_side[side]["solve_latency_us"].append(float(elapsed_us))
            result_codes[code] = result_codes.get(code, 0) + 1
            if code != 0:
                frame_ok = False
                candidate[side] = np.asarray(current[side], dtype=float).copy()
            else:
                candidate[side] = np.radians(
                    np.asarray(list(q_out), dtype=float)
                )

        if frame_ok:
            validation = planner.validate_qpos_reference(
                candidate,
                source="official_remote_ik_probe",
            )
            if not validation.safe:
                collision_rejections += 1
                frame_ok = False
        if not frame_ok:
            candidate = {
                side: np.asarray(current[side], dtype=float).copy()
                for side in _SIDES
            }
        else:
            admission = planner.admit_qpos_reference(
                candidate,
                source="official_remote_ik_probe",
            )
            if not admission.safe:
                raise RuntimeError(
                    "validated official remote-IK proposal was not admitted."
                )

        truth = _runtime_truth(group, reference)
        diagnostics = planner.analyze_configuration(candidate)
        target_endpoint_world = planner.resolve_target_endpoint_pose_world(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
        )
        realized_endpoint_world = planner.observe_endpoint_poses_world(candidate)
        for side in _SIDES:
            velocity = (candidate[side] - current[side]) * 60.0
            acceleration = (velocity - previous_velocity[side]) * 60.0
            translation_error, rotation_error = _pose_error(
                realized_endpoint_world[side],
                target_endpoint_world[side],
            )
            per_side[side]["joint_max_deg"].append(
                float(
                    np.max(
                        np.abs(np.degrees(candidate[side] - truth[side]))
                    )
                )
            )
            per_side[side]["target_translation_error_m"].append(
                translation_error
            )
            per_side[side]["target_rotation_error_deg"].append(
                rotation_error
            )
            per_side[side]["velocity_max_deg_s"].append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            per_side[side]["acceleration_max_deg_s2"].append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            per_side[side]["sigma_min"].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
            previous_velocity[side] = velocity
        current = candidate

    return {
        "episode_index": int(episode_index),
        "profile": "official_rm_algo_ik_remote_60hz",
        "result_codes": {
            str(code): int(count) for code, count in sorted(result_codes.items())
        },
        "collision_rejections": int(collision_rejections),
        "runtime_reference_count": len(references),
        "sides": {
            side: {
                key: _summary(value)
                for key, value in metrics.items()
            }
            for side, metrics in per_side.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only official all-IK soft branch-anchor probe. "
            "No device access, motion, or dispatch."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument(
        "--anchor-cost",
        type=float,
        action="append",
        help=(
            "Pink posture cost; repeat for multiple. Defaults to "
            "0 (V4), 0.005, 0.02 and 0.05."
        ),
    )
    parser.add_argument("--left-tool-profile", default=_LEFT_TOOL)
    parser.add_argument("--right-tool-profile", default=_RIGHT_TOOL)
    parser.add_argument(
        "--joint-governor",
        action="store_true",
        help=(
            "Also evaluate official all-IK targets through the existing "
            "acceleration-limited joint trajectory planner."
        ),
    )
    parser.add_argument(
        "--source-joint-interpolation",
        action="store_true",
        help=(
            "Also solve only source-rate Poses with official all-IK and "
            "interpolate the selected q targets at 60 Hz."
        ),
    )
    parser.add_argument(
        "--source-quintic-interpolation",
        action="store_true",
        help=(
            "Also solve source-rate Poses with official all-IK and evaluate "
            "a continuity-segment-aware C2 quintic joint path at 60 Hz."
        ),
    )
    parser.add_argument(
        "--source-quintic-blend",
        type=float,
        default=1.0,
        help=(
            "Blend in [0,1] from collision-certified linear source path to "
            "the C2 quintic source path."
        ),
    )
    parser.add_argument(
        "--pink-residual",
        action="store_true",
        help=(
            "Also apply a frame-local two-iteration Pink residual correction "
            "from each source-all-IK joint interpolation seed."
        ),
    )
    parser.add_argument(
        "--intermediate-all-ik-projection",
        action="store_true",
        help=(
            "Also project each 60 Hz Cartesian target to the nearest official "
            "all-IK solution from the source-joint interpolation seed."
        ),
    )
    parser.add_argument(
        "--source-path-downstream-planner",
        action="store_true",
        help=(
            "Also feed the 60 Hz source-all-IK joint path through the existing "
            "120 Hz acceleration-limited qcmd planner."
        ),
    )
    parser.add_argument(
        "--source-path-jerk-governor",
        action="store_true",
        help=(
            "Also feed the 15 Hz official-all-IK source joint path through "
            "the shared bounded-lookahead third-order horizon governor."
        ),
    )
    parser.add_argument(
        "--jerk-governor-lookahead-source-steps",
        type=int,
        default=2,
        help="Source-rate lookahead used by --source-path-jerk-governor.",
    )
    parser.add_argument(
        "--jerk-governor-ramp-sec",
        type=float,
        default=0.10,
        help=(
            "Derive per-joint jerk limits as max_acceleration/ramp_sec for "
            "--source-path-jerk-governor."
        ),
    )
    parser.add_argument(
        "--jerk-governor-reference-interpolation",
        choices=("linear", "quintic"),
        default="linear",
        help=(
            "Joint reference geometry used inside "
            "--source-path-jerk-governor."
        ),
    )
    parser.add_argument(
        "--official-remote-ik",
        action="store_true",
        help=(
            "Also evaluate the vendor rm_algo_ik_remote continuous Cartesian "
            "solver at 60 Hz."
        ),
    )
    parser.add_argument(
        "--skip-soft-anchor",
        action="store_true",
        help="Skip V4 and soft-anchor reports.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Emit one aggregate per evaluated profile.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optionally write the exact JSON report to this path.",
    )
    args = parser.parse_args()
    if args.jerk_governor_lookahead_source_steps < 2:
        raise ValueError(
            "--jerk-governor-lookahead-source-steps must be at least 2."
        )
    if (
        not math.isfinite(args.jerk_governor_ramp_sec)
        or args.jerk_governor_ramp_sec <= 0.0
    ):
        raise ValueError("--jerk-governor-ramp-sec must be finite and positive.")
    if (
        not math.isfinite(args.source_quintic_blend)
        or not 0.0 <= args.source_quintic_blend <= 1.0
    ):
        raise ValueError("--source-quintic-blend must be in [0, 1].")

    costs = (
        [float(value) for value in args.anchor_cost]
        if args.anchor_cost
        else [0.0, 0.005, 0.02, 0.05]
    )
    if any(not math.isfinite(value) or value < 0.0 for value in costs):
        raise ValueError("--anchor-cost values must be finite and non-negative.")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if args.all_episodes and args.episode:
        raise ValueError("--all-episodes and --episode are mutually exclusive.")
    episode_indices = (
        read_replay_dataset_episode_indices(dataset_root)
        if args.all_episodes
        else [int(value) for value in (args.episode or [])]
    )
    if not episode_indices:
        raise ValueError("provide --episode or --all-episodes.")
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
    reports = (
        []
        if args.skip_soft_anchor
        else [
            _evaluate_episode(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
                anchor_cost=float(cost),
            )
            for episode_index in episode_indices
            for cost in costs
        ]
    )
    if args.joint_governor:
        reports.extend(
            _evaluate_joint_governed_episode(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.source_joint_interpolation:
        reports.extend(
            _evaluate_source_all_ik_joint_interpolation(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.source_quintic_interpolation:
        reports.extend(
            _evaluate_source_all_ik_joint_interpolation(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
                interpolation="quintic",
                quintic_blend=float(args.source_quintic_blend),
            )
            for episode_index in episode_indices
        )
    if args.pink_residual:
        reports.extend(
            _evaluate_source_joint_interpolation_with_pink_residual(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.intermediate_all_ik_projection:
        reports.extend(
            _evaluate_source_path_with_intermediate_all_ik_projection(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.source_path_downstream_planner:
        reports.extend(
            _evaluate_source_path_through_downstream_planner(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.source_path_jerk_governor:
        reports.extend(
            _evaluate_source_path_through_jerk_horizon_governor(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
                lookahead_source_steps=int(
                    args.jerk_governor_lookahead_source_steps
                ),
                jerk_ramp_sec=float(args.jerk_governor_ramp_sec),
                reference_interpolation=str(
                    args.jerk_governor_reference_interpolation
                ),
            )
            for episode_index in episode_indices
        )
    if args.official_remote_ik:
        reports.extend(
            _evaluate_official_remote_ik_episode(
                dataset_root=dataset_root,
                episode_index=int(episode_index),
                algo=algo,
                endpoint_to_tool=endpoint_to_tool,
            )
            for episode_index in episode_indices
        )
    if args.aggregate_only:
        aggregate: dict[str, Any] = {}
        for report in reports:
            profile = str(report["profile"])
            profile_row = aggregate.setdefault(
                profile,
                {
                    "episode_count": 0,
                    "source_failures": 0,
                    "runtime_collision_failures": 0,
                    "command_collision_rejections": 0,
                    "endpoint_collision_failures": 0,
                    "sampled_interval_collision_failures": 0,
                    "endpoint_collision_episodes": [],
                    "sampled_interval_collision_episodes": [],
                    "quintic_collision_fallback_frames": 0,
                    "dynamic_violation_frames": {
                        side: {
                            "velocity": 0,
                            "acceleration": 0,
                            "jerk": 0,
                        }
                        for side in _SIDES
                    },
                    "collision_rejections": 0,
                    "no_collision_safe_projection_frames": 0,
                    "intermediate_missing_by_side": {
                        side: 0 for side in _SIDES
                    },
                    "sides": {
                        side: {
                            "joint_max_deg": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "target_translation_error_max_m": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "target_rotation_error_max_deg": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "velocity_max_deg_s": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "acceleration_max_deg_s2": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "jerk_max_deg_s3": {
                                "value": -math.inf,
                                "episode_index": None,
                            },
                            "sigma_min": {
                                "value": math.inf,
                                "episode_index": None,
                            },
                        }
                        for side in _SIDES
                    },
                },
            )
            profile_row["episode_count"] += 1
            profile_row["source_failures"] += int(
                report.get("source_failures", 0)
            )
            profile_row["runtime_collision_failures"] += int(
                report.get("runtime_collision_failures", 0)
            )
            profile_row["command_collision_rejections"] += int(
                report.get("command_collision_rejections", 0)
            )
            profile_row["endpoint_collision_failures"] += int(
                report.get("endpoint_collision_failures", 0)
            )
            if int(report.get("endpoint_collision_failures", 0)) > 0:
                profile_row["endpoint_collision_episodes"].append(
                    {
                        "episode_index": int(report["episode_index"]),
                        "failure_count": int(
                            report["endpoint_collision_failures"]
                        ),
                    }
                )
            profile_row["sampled_interval_collision_failures"] += int(
                report.get("sampled_interval_collision_failures", 0)
            )
            if (
                int(report.get("sampled_interval_collision_failures", 0))
                > 0
            ):
                profile_row["sampled_interval_collision_episodes"].append(
                    {
                        "episode_index": int(report["episode_index"]),
                        "failure_count": int(
                            report["sampled_interval_collision_failures"]
                        ),
                    }
                )
            profile_row["quintic_collision_fallback_frames"] += int(
                report.get("quintic_collision_fallback_frames", 0)
            )
            report_dynamic_violations = dict(
                report.get("dynamic_violation_frames") or {}
            )
            for side in _SIDES:
                side_violations = dict(
                    report_dynamic_violations.get(side) or {}
                )
                for metric in ("velocity", "acceleration", "jerk"):
                    profile_row["dynamic_violation_frames"][side][
                        metric
                    ] += int(side_violations.get(metric) or 0)
            profile_row["collision_rejections"] += int(
                report.get("collision_rejections", 0)
            )
            profile_row["no_collision_safe_projection_frames"] += int(
                report.get("no_collision_safe_projection_frames", 0)
            )
            intermediate_missing = dict(
                report.get("intermediate_missing_by_side", {})
            )
            for side in _SIDES:
                profile_row["intermediate_missing_by_side"][side] += int(
                    intermediate_missing.get(side, 0)
                )
            for side in _SIDES:
                metrics = report["sides"][side]
                mappings = {
                    "joint_max_deg": ("joint_max_deg", "max"),
                    "target_translation_error_max_m": (
                        "target_translation_error_m",
                        "max",
                    ),
                    "target_rotation_error_max_deg": (
                        "target_rotation_error_deg",
                        "max",
                    ),
                    "velocity_max_deg_s": ("velocity_max_deg_s", "max"),
                    "acceleration_max_deg_s2": (
                        "acceleration_max_deg_s2",
                        "max",
                    ),
                    "jerk_max_deg_s3": ("jerk_max_deg_s3", "max"),
                    "sigma_min": ("sigma_min", "min"),
                }
                for output_key, (metric_key, reduction) in mappings.items():
                    value = metrics.get(metric_key, {}).get(reduction)
                    if value is None:
                        continue
                    target = profile_row["sides"][side][output_key]
                    better = (
                        float(value) < float(target["value"])
                        if output_key == "sigma_min"
                        else float(value) > float(target["value"])
                    )
                    if better:
                        target["value"] = float(value)
                        target["episode_index"] = int(
                            report["episode_index"]
                        )
        reports_output: Any = aggregate
    else:
        reports_output = reports
    payload = {
        "schema": "todos.rm65_official_branch_anchor_probe/v1",
        "device_access": False,
        "motion": False,
        "dispatch": False,
        "sdk_algorithm_version": str(algo.rm_algo_version()),
        "reports": reports_output,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
