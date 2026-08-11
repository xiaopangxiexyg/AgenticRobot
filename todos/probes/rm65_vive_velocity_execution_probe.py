"""Offline VIVE-velocity-assisted RM65 15/30 Hz execution probe.

This script is strictly read-only with respect to recorded actuator traces.  It
does not initialize devices and never dispatches robot commands.  It estimates
a causal 60 Hz Cartesian twist from the recorded, calibrated VIVE target pose,
maps that twist through the RM65 Jacobian with damped least squares, and then
compares:

* the production 15 Hz position-only V2/V3 execution baselines;
* 15 Hz action-authoritative Hermite references using VIVE-derived velocity;
* a 30 Hz velocity refresh that keeps the already published 15 Hz action
  endpoint fixed.

The calibrated ``ik_shadow.target_world_pose`` is used for the commanded twist
direction.  Its timestamps and source validity come from the corresponding
``vive_source`` rows.  This is the robot-frame image of VIVE motion, not a
finite difference of IK ``q_ref``.  Every estimate is causal: row ``i`` uses
only rows ``<= i``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROBE_ROOT = Path(__file__).resolve().parent
for _path in (_PROJECT_ROOT, _PROBE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rm65_teleop_smoothing_probe import (  # noqa: E402
    Candidate,
    _BOUND_GUARD,
    _COMMAND_HZ,
    _FORMAL_ACTION_HZ,
    _HORIZON_TICKS,
    _REFERENCE_HZ,
    _aggregate,
    _evaluate_simulation,
    _finite_matrix,
    _print_ranking,
    _reference_horizon,
    _simulate,
)
from robot_motion.position_control import (  # noqa: E402
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonState,
)
from robot_motion.rm65 import (  # noqa: E402
    RM65DualArmCollisionIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
)
from robot_motion.rm65.constants import (  # noqa: E402
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.defaults import (  # noqa: E402
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
)


_SIDES = ("left", "right")
_MISSING_NS = np.iinfo(np.int64).min


@dataclass(frozen=True)
class ViveTrace:
    name: str
    date: str
    path: Path
    qref_full_rad: np.ndarray
    qref_by_side: dict[str, np.ndarray]
    target_pose_by_side: dict[str, np.ndarray]
    raw_vive_pose_by_side: dict[str, np.ndarray]
    sample_time_sec_by_side: dict[str, np.ndarray]
    valid_by_side: dict[str, np.ndarray]
    velocity_pose_source: str


def _load_vive_trace(path: Path) -> ViveTrace:
    data_path = path / "command_trace.npz" if path.is_dir() else path
    with np.load(data_path, allow_pickle=False) as trace:
        if "ik_shadow" not in trace.files:
            raise ValueError(f"{data_path} must contain an ik_shadow array.")
        shadow = trace["ik_shadow"]
        required_shadow = {"admitted_q_rad", "target_world_pose"}
        required_vive = {
            "pose_vive_world",
            "valid_mask",
            "pose_sample_monotonic_time_ns",
            "control_monotonic_time_ns",
        }
        if not required_shadow.issubset(shadow.dtype.names or ()):
            raise ValueError(f"{data_path} has an old ik_shadow schema.")
        vive = trace["vive_source"] if "vive_source" in trace.files else None
        if vive is not None and not required_vive.issubset(vive.dtype.names or ()):
            raise ValueError(f"{data_path} has an incompatible vive_source schema.")
        count = len(shadow) if vive is None else min(len(shadow), len(vive))
        if count < 8:
            raise ValueError(f"{data_path} has too few aligned 60 Hz rows.")
        qref_full = _finite_matrix(shadow["admitted_q_rad"][:count], columns=12)
        target_pose = np.asarray(shadow["target_world_pose"][:count], dtype=float)
        raw_pose = (
            target_pose.copy()
            if vive is None
            else np.asarray(vive["pose_vive_world"][:count], dtype=float)
        )
        if target_pose.shape != (count, 2, 4, 4):
            raise ValueError("target_world_pose must have shape Nx2x4x4.")
        if raw_pose.shape != (count, 2, 4, 4):
            raise ValueError("pose_vive_world must have shape Nx2x4x4.")
        if vive is None:
            shadow_time = np.asarray(
                shadow["monotonic_time_ns"][:count], dtype=np.int64
            )
            sample_ns = np.repeat(shadow_time[:, None], 2, axis=1)
            control_ns = shadow_time.copy()
            valid_mask = np.full(count, 3, dtype=np.uint8)
            velocity_pose_source = "calibrated_vive_target_pose_ik_shadow_60hz"
        else:
            sample_ns = np.asarray(
                vive["pose_sample_monotonic_time_ns"][:count], dtype=np.int64
            )
            control_ns = np.asarray(
                vive["control_monotonic_time_ns"][:count], dtype=np.int64
            )
            valid_mask = np.asarray(vive["valid_mask"][:count], dtype=np.uint8)
            velocity_pose_source = "calibrated_vive_target_pose_with_vive_source_timestamps"

    qref_by_side = {
        "left": qref_full[:, :6].copy(),
        "right": qref_full[:, 6:].copy(),
    }
    target_by_side: dict[str, np.ndarray] = {}
    raw_by_side: dict[str, np.ndarray] = {}
    time_by_side: dict[str, np.ndarray] = {}
    valid_by_side: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(_SIDES):
        side_sample_ns = sample_ns[:, side_index].copy()
        missing = side_sample_ns == _MISSING_NS
        side_sample_ns[missing] = control_ns[missing]
        time_sec = side_sample_ns.astype(np.float64) * 1e-9
        valid = (
            ((valid_mask >> side_index) & 1).astype(bool)
            & np.all(np.isfinite(target_pose[:, side_index]), axis=(1, 2))
            & np.all(np.isfinite(raw_pose[:, side_index]), axis=(1, 2))
            & np.isfinite(time_sec)
        )
        target_by_side[side] = target_pose[:, side_index].copy()
        raw_by_side[side] = raw_pose[:, side_index].copy()
        time_by_side[side] = time_sec
        valid_by_side[side] = valid
    return ViveTrace(
        name=data_path.parent.name,
        date=data_path.parent.name[:8],
        path=data_path,
        qref_full_rad=qref_full.copy(),
        qref_by_side=qref_by_side,
        target_pose_by_side=target_by_side,
        raw_vive_pose_by_side=raw_by_side,
        sample_time_sec_by_side=time_by_side,
        valid_by_side=valid_by_side,
        velocity_pose_source=velocity_pose_source,
    )


def _high_motion_window(trace: ViveTrace, rows: int) -> ViveTrace:
    count = trace.qref_full_rad.shape[0]
    if rows <= 0:
        raise ValueError("window rows must be positive.")
    if count <= rows:
        return trace
    movement = np.max(np.abs(np.diff(trace.qref_full_rad, axis=0)), axis=1)
    score = np.convolve(movement, np.ones(rows - 1, dtype=float), mode="valid")
    start = int(np.argmax(score))
    stop = start + rows
    return ViveTrace(
        name=f"{trace.name}:high_motion[{start}:{stop}]",
        date=trace.date,
        path=trace.path,
        qref_full_rad=trace.qref_full_rad[start:stop].copy(),
        qref_by_side={
            side: values[start:stop].copy()
            for side, values in trace.qref_by_side.items()
        },
        target_pose_by_side={
            side: values[start:stop].copy()
            for side, values in trace.target_pose_by_side.items()
        },
        raw_vive_pose_by_side={
            side: values[start:stop].copy()
            for side, values in trace.raw_vive_pose_by_side.items()
        },
        sample_time_sec_by_side={
            side: values[start:stop].copy()
            for side, values in trace.sample_time_sec_by_side.items()
        },
        valid_by_side={
            side: values[start:stop].copy()
            for side, values in trace.valid_by_side.items()
        },
        velocity_pose_source=trace.velocity_pose_source,
    )


def _causal_pose_twist_60hz(
    pose: np.ndarray,
    sample_time_sec: np.ndarray,
    valid: np.ndarray,
    *,
    window_rows: int,
    efficiency_power: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate world-frame [linear, angular] twist without future samples."""

    poses = np.asarray(pose, dtype=float)
    times = np.asarray(sample_time_sec, dtype=float).reshape(-1)
    validity = np.asarray(valid, dtype=bool).reshape(-1)
    if poses.shape != (times.size, 4, 4) or validity.shape != times.shape:
        raise ValueError("pose, time and validity shapes do not match.")
    if window_rows < 3:
        raise ValueError("window_rows must be >= 3.")
    output = np.zeros((times.size, 6), dtype=float)
    estimated = np.zeros(times.size, dtype=bool)
    interval_count = np.zeros(times.size, dtype=np.int64)
    for end in range(times.size):
        # Never carry a historical estimate across an invalid current source
        # row.  Live integration will use the same fail-closed contract.
        if not validity[end]:
            continue
        start = max(0, end - window_rows + 1)
        linear_steps: list[np.ndarray] = []
        angular_steps: list[np.ndarray] = []
        translations: list[float] = []
        rotations: list[float] = []
        accepted_indices: list[int] = []
        for index in range(start + 1, end + 1):
            previous = index - 1
            if not validity[previous] or not validity[index]:
                continue
            dt = float(times[index] - times[previous])
            # Match live 60 Hz timing while rejecting clock resets and stalls.
            if not 0.005 <= dt <= 0.050:
                continue
            delta_position = poses[index, :3, 3] - poses[previous, :3, 3]
            delta_rotation = Rotation.from_matrix(
                poses[index, :3, :3] @ poses[previous, :3, :3].T
            ).as_rotvec()
            linear_steps.append(delta_position / dt)
            angular_steps.append(delta_rotation / dt)
            translations.append(float(np.linalg.norm(delta_position)))
            rotations.append(float(np.linalg.norm(delta_rotation)))
            if not accepted_indices:
                accepted_indices.append(previous)
            accepted_indices.append(index)
        if len(linear_steps) < 2:
            continue
        linear = np.median(np.asarray(linear_steps), axis=0)
        angular = np.median(np.asarray(angular_steps), axis=0)
        first = accepted_indices[0]
        last = accepted_indices[-1]
        net_translation = float(
            np.linalg.norm(poses[last, :3, 3] - poses[first, :3, 3])
        )
        net_rotation = float(
            np.linalg.norm(
                Rotation.from_matrix(
                    poses[last, :3, :3] @ poses[first, :3, :3].T
                ).as_rotvec()
            )
        )
        translation_efficiency = net_translation / max(sum(translations), 1e-12)
        rotation_efficiency = net_rotation / max(sum(rotations), 1e-12)
        output[end, :3] = linear * translation_efficiency**efficiency_power
        output[end, 3:] = angular * rotation_efficiency**efficiency_power
        estimated[end] = True
        interval_count[end] = len(linear_steps)
    return output, estimated, {
        "estimated_ratio": float(np.mean(estimated)),
        "interval_count_p50": float(np.percentile(interval_count, 50.0)),
        "interval_count_p95": float(np.percentile(interval_count, 95.0)),
        "linear_speed_p95_m_s": float(
            np.percentile(np.linalg.norm(output[:, :3], axis=1), 95.0)
        ),
        "angular_speed_p95_rad_s": float(
            np.percentile(np.linalg.norm(output[:, 3:], axis=1), 95.0)
        ),
    }


def _map_twist_to_joint_velocity(
    *,
    solver: Any,
    frame_name: str,
    arm_indices: np.ndarray,
    qref_full_rad: np.ndarray,
    twist_world: np.ndarray,
    twist_valid: np.ndarray,
    characteristic_length_m: float,
    damping: float,
    velocity_cap_scale: float,
    residual_ratio_guard: float,
    utilization_guard: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q_full = _finite_matrix(qref_full_rad, columns=12)
    twist = _finite_matrix(twist_world, columns=6)
    if q_full.shape[0] != twist.shape[0]:
        raise ValueError("qref and twist row counts must match.")
    valid = np.asarray(twist_valid, dtype=bool).reshape(-1)
    if valid.shape != (q_full.shape[0],):
        raise ValueError("twist_valid must match qref rows.")
    if characteristic_length_m <= 0.0 or damping <= 0.0:
        raise ValueError("DLS scale and damping must be positive.")
    if residual_ratio_guard <= 0.0 or not 0.0 < utilization_guard <= 1.0:
        raise ValueError("DLS residual/utilization guards are invalid.")
    effective_velocity = np.minimum(
        np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S, dtype=float),
        math.radians(RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG) * _COMMAND_HZ,
    )
    cap = float(velocity_cap_scale) * effective_velocity
    task_weight = np.asarray(
        [
            1.0 / characteristic_length_m,
            1.0 / characteristic_length_m,
            1.0 / characteristic_length_m,
            1.0,
            1.0,
            1.0,
        ],
        dtype=float,
    )
    scale = np.diag(effective_velocity)
    joint_velocity = np.zeros((q_full.shape[0], 6), dtype=float)
    residual_ratio = np.zeros(q_full.shape[0], dtype=float)
    utilization = np.zeros(q_full.shape[0], dtype=float)
    rejected = np.zeros(q_full.shape[0], dtype=bool)
    for index, (q, desired_twist) in enumerate(zip(q_full, twist)):
        if not valid[index]:
            continue
        if not np.any(np.abs(desired_twist) > 1e-12):
            continue
        jacobian_full = np.asarray(
            solver.compute_frame_jacobian(frame_name, q), dtype=float
        )
        jacobian = jacobian_full[:, arm_indices]
        weighted_jacobian = task_weight[:, None] * jacobian @ scale
        weighted_twist = task_weight * desired_twist
        normal = weighted_jacobian.T @ weighted_jacobian
        rhs = weighted_jacobian.T @ weighted_twist
        normalized_velocity = np.linalg.solve(
            normal + float(damping) ** 2 * np.eye(6, dtype=float), rhs
        )
        candidate = effective_velocity * normalized_velocity
        candidate = np.clip(candidate, -cap, cap)
        realized = jacobian @ candidate
        residual_ratio[index] = float(
            np.linalg.norm(task_weight * (realized - desired_twist))
            / max(np.linalg.norm(weighted_twist), 1e-12)
        )
        utilization[index] = float(
            np.max(np.abs(candidate) / np.maximum(effective_velocity, 1e-12))
        )
        if (
            residual_ratio[index] > float(residual_ratio_guard)
            or utilization[index] > float(utilization_guard)
        ):
            rejected[index] = True
            continue
        joint_velocity[index] = candidate
    mapped_valid = valid & ~rejected
    return joint_velocity, mapped_valid, {
        "weighted_residual_ratio_p50": float(np.percentile(residual_ratio, 50.0)),
        "weighted_residual_ratio_p95": float(np.percentile(residual_ratio, 95.0)),
        "joint_velocity_abs_p95_deg_s": float(
            np.percentile(np.max(np.abs(np.rad2deg(joint_velocity)), axis=1), 95.0)
        ),
        "joint_velocity_abs_max_deg_s": float(
            np.max(np.abs(np.rad2deg(joint_velocity)))
        ),
        "velocity_utilization_p95": float(np.percentile(utilization, 95.0)),
        "guard_rejected_ratio": float(np.mean(rejected)),
        "mapped_valid_ratio": float(np.mean(mapped_valid)),
        "residual_ratio_guard": float(residual_ratio_guard),
        "utilization_guard": float(utilization_guard),
    }


def _make_limits(candidate: Candidate) -> JointHorizonLimits:
    effective_velocity = np.minimum(
        np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S, dtype=float),
        math.radians(RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG) * _COMMAND_HZ,
    )
    return JointHorizonLimits(
        min_position_rad=np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float) + 1e-4,
        max_position_rad=np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float) - 1e-4,
        max_velocity_rad_s=effective_velocity,
        max_acceleration_rad_s2=RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        max_jerk_rad_s3=(
            np.asarray(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float)
            / candidate.jerk_ramp_sec
        ),
    )


def _simulate_30hz_velocity_refresh_fixed_action(
    source_qref_60hz: np.ndarray,
    external_velocity_60hz: np.ndarray,
    external_velocity_valid_60hz: np.ndarray,
    candidate: Candidate,
) -> dict[str, Any]:
    """Refresh velocity at 30 Hz without replacing the 15 Hz action endpoint."""

    qref = _finite_matrix(source_qref_60hz, columns=6)
    external = _finite_matrix(external_velocity_60hz, columns=6)
    if qref.shape != external.shape:
        raise ValueError("qref and external velocity must have matching shapes.")
    external_valid = np.asarray(
        external_velocity_valid_60hz, dtype=bool
    ).reshape(-1)
    if external_valid.shape != (qref.shape[0],):
        raise ValueError("external velocity validity must match qref rows.")
    if candidate.replan_hz != 30:
        raise ValueError("fixed-action refresh candidate must run at 30 Hz.")
    limits = _make_limits(candidate)
    workspace = JointHorizonBatchedWorkspace(
        limits=limits,
        dt_sec=1.0 / _COMMAND_HZ,
        horizon_transitions=_HORIZON_TICKS,
        bound_guard=_BOUND_GUARD,
        solver_tolerance=1e-7,
        jerk_regularization=candidate.jerk_regularization,
        jerk_delta_regularization=candidate.jerk_delta_regularization,
        initial_jerk_delta_regularization=candidate.initial_jerk_delta_regularization,
    )
    state = JointHorizonState(
        q_rad=qref[0].copy(),
        qd_rad_s=np.zeros(6, dtype=float),
        qdd_rad_s2=np.zeros(6, dtype=float),
    )
    command_rows = [state.q_rad.copy()]
    velocity_rows = [state.qd_rad_s.copy()]
    acceleration_rows = [state.qdd_rad_s2.copy()]
    tracking_rows = [qref[0].copy()]
    command_source_index = [0]
    previous_jerk = np.zeros(6, dtype=float)
    filtered_velocity = np.zeros(6, dtype=float)
    previous_raw_velocity = np.zeros(6, dtype=float)
    active_action_endpoint: np.ndarray | None = None
    formal_forecasts: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    effective_velocity = np.asarray(limits.max_velocity_rad_s, dtype=float)
    cap = candidate.velocity_cap_scale * effective_velocity

    for source_index in range(0, qref.shape[0] - 2, 2):
        phase = source_index % 4
        if phase == 0:
            target = qref[source_index].copy()
            commands_to_target = 8
        else:
            if active_action_endpoint is None:
                raise RuntimeError("30 Hz midpoint has no active action endpoint.")
            target = active_action_endpoint.copy()
            commands_to_target = 4
        raw_velocity = external[source_index].copy()
        if not external_valid[source_index]:
            raw_velocity.fill(0.0)
            filtered_velocity.fill(0.0)
        filtered_velocity = (
            (1.0 - candidate.velocity_ema_alpha) * filtered_velocity
            + candidate.velocity_ema_alpha * raw_velocity
        )
        filtered_velocity = np.clip(filtered_velocity, -cap, cap)
        displacement = target - state.q_rad
        duration = commands_to_target / _COMMAND_HZ
        terminal_velocity = filtered_velocity.copy()
        wrong_direction = terminal_velocity * displacement <= 0.0
        monotone_cap = (
            candidate.terminal_velocity_displacement_scale
            * np.abs(displacement)
            / max(duration, 1e-9)
        )
        terminal_velocity = np.sign(terminal_velocity) * np.minimum(
            np.abs(terminal_velocity), monotone_cap
        )
        terminal_velocity[wrong_direction] = 0.0
        previous_norm = float(np.linalg.norm(previous_raw_velocity))
        current_norm = float(np.linalg.norm(raw_velocity))
        coherence = float(
            np.dot(previous_raw_velocity, raw_velocity)
            / max(previous_norm * current_norm, 1e-12)
        )
        if (
            previous_norm <= 1e-6
            or current_norm <= 1e-6
            or coherence < candidate.velocity_coherence_min
        ):
            terminal_velocity.fill(0.0)
        previous_raw_velocity = raw_velocity.copy()
        reference = _reference_horizon(
            candidate=candidate,
            state=state,
            target=target,
            target_velocity=terminal_velocity,
            commands_per_replan=commands_to_target,
            effective_reference_mode="coherent_hermite",
        )
        plan = workspace.solve(
            reference,
            initial_state=state,
            commit_count=4,
            previous_jerk_rad_s3=previous_jerk,
        )
        command_start = len(command_rows) - 1
        if phase == 0:
            active_action_endpoint = np.asarray(plan.q_path_rad[8], dtype=float).copy()
            formal_forecasts.append(
                (
                    source_index,
                    command_start,
                    active_action_endpoint.copy(),
                    qref[source_index].copy(),
                )
            )
        for step in range(1, 5):
            command_rows.append(np.asarray(plan.q_path_rad[step], dtype=float).copy())
            velocity_rows.append(
                np.asarray(plan.qd_path_rad_s[step], dtype=float).copy()
            )
            acceleration_rows.append(
                np.asarray(plan.qdd_path_rad_s2[step], dtype=float).copy()
            )
            tracking_rows.append(qref[source_index].copy())
            command_source_index.append(source_index)
        state = JointHorizonState(
            q_rad=np.asarray(plan.q_path_rad[4], dtype=float).copy(),
            qd_rad_s=np.asarray(plan.qd_path_rad_s[4], dtype=float).copy(),
            qdd_rad_s2=np.asarray(plan.qdd_path_rad_s2[4], dtype=float).copy(),
        )
        previous_jerk = (
            np.asarray(plan.qdd_path_rad_s2[4], dtype=float)
            - np.asarray(plan.qdd_path_rad_s2[3], dtype=float)
        ) * _COMMAND_HZ

    command = np.asarray(command_rows, dtype=float)
    action_error: list[np.ndarray] = []
    action_state_error: list[np.ndarray] = []
    requested_next_state_error: list[np.ndarray] = []
    for _source_index, command_start, forecast, requested in formal_forecasts:
        next_index = command_start + 8
        if next_index >= command.shape[0]:
            continue
        next_state = command[next_index]
        action_error.append(forecast - requested)
        action_state_error.append(next_state - forecast)
        requested_next_state_error.append(next_state - requested)
    return {
        "q_rad": command,
        "qd_rad_s": np.asarray(velocity_rows, dtype=float),
        "qdd_rad_s2": np.asarray(acceleration_rows, dtype=float),
        "tracking_reference_rad": np.asarray(tracking_rows, dtype=float),
        "command_source_index": np.asarray(command_source_index, dtype=np.int64),
        "action_error_rad": np.asarray(action_error, dtype=float),
        "action_state_error_rad": np.asarray(action_state_error, dtype=float),
        "requested_next_state_error_rad": np.asarray(
            requested_next_state_error, dtype=float
        ),
        "max_jerk_rad_s3": np.asarray(limits.max_jerk_rad_s3, dtype=float),
        "selection_counts": {"hold": 0, "hermite": len(formal_forecasts)},
    }


def _candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("15_runtime_v2", 15, "runtime_v2", 0.05),
        Candidate("15_runtime_v2_nullspace", 15, "runtime_v2_nullspace", 0.05),
        Candidate(
            "15_vive_dls_selective_a25_c25",
            15,
            "runtime_vive_selective",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.25,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "15_vive_dls_selective_a50_c35",
            15,
            "runtime_vive_selective",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "15_vive_dls_nullspace_a15_c15_h060_m050",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050_xj1e5",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050_xj1e5_slew10",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
            velocity_slew_accel_scale=0.10,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050_xj1e5_slew20",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
            velocity_slew_accel_scale=0.20,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050_xj1e4",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-4,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_nullspace_a15_c15_h060_m050_xj1e4",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            initial_jerk_delta_regularization=1e-4,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_nullspace_a05_c10_h070_m025_xj1e5",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.05,
            velocity_cap_scale=0.10,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.25,
        ),
        Candidate(
            "15_vive_dls_nullspace_a10_c10_h070_m025_xj1e5",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.10,
            velocity_cap_scale=0.10,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.25,
        ),
        Candidate(
            "15_vive_dls_nullspace_a15_c10_h070_m025_xj1e5",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.10,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.25,
        ),
        Candidate(
            "15_vive_dls_nullspace_a25_c20_h070_m050_xj1e3",
            15,
            "runtime_vive_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-3,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_endpoint_selective_a15_c15_h060_m050",
            15,
            "runtime_vive_endpoint_selective",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_endpoint_selective_a25_c20_h070_m050",
            15,
            "runtime_vive_endpoint_selective",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_endpoint_selective_nullspace_a15_c15_h060_m050",
            15,
            "runtime_vive_endpoint_selective_nullspace",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_endpoint_selective_nullspace_a25_c20_h070_m050",
            15,
            "runtime_vive_endpoint_selective_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            observer_speed_guard_scale=0.50,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_coherent_a25_c25",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.25,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "15_vive_dls_coherent_a15_c15_h060_m050",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_coherent_a25_c20_h070_m050",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_coherent_a15_c15_h060_m050_xj1e5",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.15,
            velocity_cap_scale=0.15,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.60,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_coherent_a25_c20_h070_m050_xj1e5",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            terminal_velocity_displacement_scale=0.50,
        ),
        Candidate(
            "15_vive_dls_coherent_a50_c35",
            15,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "30_vive_dls_fixed_action_a25_c25",
            30,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.25,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "30_vive_dls_midcut_nullspace_a25_c20",
            15,
            "runtime_vive_midcut_nullspace",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "30_vive_dls_midcut_nullspace_a50_c35",
            15,
            "runtime_vive_midcut_nullspace",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "30_vive_dls_midcut_q_anchor_a25_c20",
            15,
            "runtime_vive_midcut_q_anchor",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "30_vive_dls_midcut_q_anchor_a50_c35",
            15,
            "runtime_vive_midcut_q_anchor",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "30_vive_dls_midcut_q_qdd_anchor_a25_c20",
            15,
            "runtime_vive_midcut_q_qdd_anchor",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "30_vive_dls_midcut_q_qdd_anchor_a50_c35",
            15,
            "runtime_vive_midcut_q_qdd_anchor",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "15_vive_dls_cut_q_anchor_a25_c20",
            15,
            "runtime_vive_cut_q_anchor",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "15_vive_dls_cut_q_anchor_a50_c35",
            15,
            "runtime_vive_cut_q_anchor",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
        ),
        Candidate(
            "15_vive_dls_predictive_action_tick1_a25_c20_slew10",
            15,
            "runtime_vive_predictive_action",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            velocity_slew_accel_scale=0.10,
            velocity_prediction_sec=1.0 / 60.0,
        ),
        Candidate(
            "15_vive_dls_predictive_action_tick2_a25_c20_slew10",
            15,
            "runtime_vive_predictive_action",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.20,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.70,
            velocity_slew_accel_scale=0.10,
            velocity_prediction_sec=2.0 / 60.0,
        ),
        Candidate(
            "15_vive_dls_predictive_action_tick025_a10_c10_slew05",
            15,
            "runtime_vive_predictive_action",
            0.05,
            velocity_ema_alpha=0.10,
            velocity_cap_scale=0.10,
            velocity_observer="vive_dls_60hz",
            velocity_observer_window=8,
            velocity_coherence_min=0.90,
            velocity_slew_accel_scale=0.05,
            velocity_prediction_sec=0.25 / 60.0,
        ),
        Candidate(
            "15_vive_dls_predictive_action_tick050_a10_c10_slew05",
            15,
            "runtime_vive_predictive_action",
            0.05,
            velocity_ema_alpha=0.10,
            velocity_cap_scale=0.10,
            velocity_observer="vive_dls_60hz",
            velocity_observer_window=8,
            velocity_coherence_min=0.90,
            velocity_slew_accel_scale=0.05,
            velocity_prediction_sec=0.50 / 60.0,
        ),
        Candidate(
            "30_vive_dls_fixed_action_a50_c35",
            30,
            "coherent_hermite",
            0.05,
            velocity_ema_alpha=0.50,
            velocity_cap_scale=0.35,
            velocity_observer="vive_dls_60hz",
            velocity_coherence_min=0.40,
            terminal_velocity_displacement_scale=0.75,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--twist-window-rows", type=int, default=5)
    parser.add_argument("--twist-efficiency-power", type=float, default=1.0)
    parser.add_argument("--characteristic-length-m", type=float, default=0.5)
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--dls-velocity-cap-scale", type=float, default=0.5)
    parser.add_argument("--dls-residual-ratio-guard", type=float, default=0.35)
    parser.add_argument("--dls-utilization-guard", type=float, default=0.35)
    parser.add_argument(
        "--dls-anchor",
        choices=("qref", "baseline_v3_qcmd"),
        default="baseline_v3_qcmd",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        help="Evaluate only the highest-qref-motion 60 Hz window per trace.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    traces = tuple(_load_vive_trace(path) for path in args.trace)
    if args.window_rows is not None:
        traces = tuple(_high_motion_window(trace, args.window_rows) for trace in traces)
    candidates = _candidates()
    if args.candidate:
        requested = set(args.candidate)
        candidates = tuple(c for c in candidates if c.name in requested)
        missing = requested - {candidate.name for candidate in candidates}
        if missing:
            raise SystemExit(f"unknown candidate(s): {sorted(missing)}")

    bundle = build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(solver_frequency=_REFERENCE_HZ)
    )
    frame_by_side = {
        "left": bundle.left_frame_name,
        "right": bundle.right_frame_name,
    }
    indices_by_side = {
        "left": np.arange(0, 6, dtype=np.int64),
        "right": np.arange(6, 12, dtype=np.int64),
    }
    velocity_by_trace_side: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray]
    ] = {}
    velocity_diagnostics: list[dict[str, Any]] = []
    baseline_candidate = next(
        candidate
        for candidate in _candidates()
        if candidate.name == "15_runtime_v2_nullspace"
    )
    for trace in traces:
        mapping_anchor = trace.qref_full_rad
        if args.dls_anchor == "baseline_v3_qcmd":
            mapping_anchor = trace.qref_full_rad.copy()
            for anchor_side in _SIDES:
                baseline = _simulate(
                    trace.qref_by_side[anchor_side],
                    baseline_candidate,
                )
                command = np.asarray(baseline["q_rad"], dtype=float)
                command_index = np.minimum(
                    2 * np.arange(trace.qref_full_rad.shape[0], dtype=np.int64),
                    command.shape[0] - 1,
                )
                mapping_anchor[:, indices_by_side[anchor_side]] = command[
                    command_index
                ]
        for side in _SIDES:
            target_twist, target_twist_valid, twist_metadata = _causal_pose_twist_60hz(
                trace.target_pose_by_side[side],
                trace.sample_time_sec_by_side[side],
                trace.valid_by_side[side],
                window_rows=args.twist_window_rows,
                efficiency_power=args.twist_efficiency_power,
            )
            # Raw VIVE is estimated separately as a diagnostic.  The command
            # uses the calibrated target pose so mount/home frame transforms
            # are applied exactly once before the Jacobian mapping.
            raw_twist, _raw_twist_valid, raw_metadata = _causal_pose_twist_60hz(
                trace.raw_vive_pose_by_side[side],
                trace.sample_time_sec_by_side[side],
                trace.valid_by_side[side],
                window_rows=args.twist_window_rows,
                efficiency_power=args.twist_efficiency_power,
            )
            joint_velocity, joint_velocity_valid, mapping_metadata = (
                _map_twist_to_joint_velocity(
                    solver=bundle.solver,
                    frame_name=frame_by_side[side],
                    arm_indices=indices_by_side[side],
                    qref_full_rad=mapping_anchor,
                    twist_world=target_twist,
                    twist_valid=target_twist_valid,
                    characteristic_length_m=args.characteristic_length_m,
                    damping=args.dls_damping,
                    velocity_cap_scale=args.dls_velocity_cap_scale,
                    residual_ratio_guard=args.dls_residual_ratio_guard,
                    utilization_guard=args.dls_utilization_guard,
                )
            )
            velocity_by_trace_side[(trace.name, side)] = (
                joint_velocity,
                joint_velocity_valid,
            )
            velocity_diagnostics.append(
                {
                    "trace": trace.name,
                    "date": trace.date,
                    "side": side,
                    "source_rate_hz": _REFERENCE_HZ,
                    "velocity_pose_source": trace.velocity_pose_source,
                    "causal": True,
                    "dls_anchor": args.dls_anchor,
                    "target_twist": twist_metadata,
                    "raw_vive_twist": raw_metadata,
                    "joint_mapping": mapping_metadata,
                }
            )

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for trace in traces:
            for side in _SIDES:
                try:
                    external_velocity, external_velocity_valid = (
                        velocity_by_trace_side[(trace.name, side)]
                    )
                    if candidate.name.startswith("30_vive_dls_fixed_action"):
                        result = _simulate_30hz_velocity_refresh_fixed_action(
                            trace.qref_by_side[side],
                            external_velocity,
                            external_velocity_valid,
                            candidate,
                        )
                    else:
                        result = _simulate(
                            trace.qref_by_side[side],
                            candidate,
                            external_velocity_60hz=(
                                external_velocity
                                if candidate.velocity_observer == "vive_dls_60hz"
                                else None
                            ),
                            external_velocity_valid_60hz=(
                                external_velocity_valid
                                if candidate.velocity_observer == "vive_dls_60hz"
                                else None
                            ),
                        )
                    metrics = _evaluate_simulation(result)
                except Exception as exc:
                    failure = {
                        "trace": trace.name,
                        "date": trace.date,
                        "side": side,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    details.append({"candidate": candidate.name, **failure})
                    continue
                rows.append(metrics)
                details.append(
                    {
                        "candidate": candidate.name,
                        "trace": trace.name,
                        "date": trace.date,
                        "side": side,
                        "metrics": metrics,
                    }
                )
        summaries.append(_aggregate(rows, candidate=candidate, failures=failures))

    _print_ranking(summaries)
    payload = {
        "schema": "todos.rm65_vive_velocity_execution_probe/v1",
        "trace_paths": [str(trace.path) for trace in traces],
        "estimator": {
            "source": "60hz_causal_calibrated_vive_target_pose",
            "twist_window_rows": args.twist_window_rows,
            "twist_efficiency_power": args.twist_efficiency_power,
            "characteristic_length_m": args.characteristic_length_m,
            "dls_damping": args.dls_damping,
            "dls_velocity_cap_scale": args.dls_velocity_cap_scale,
            "dls_residual_ratio_guard": args.dls_residual_ratio_guard,
            "dls_utilization_guard": args.dls_utilization_guard,
            "future_frames_used": 0,
        },
        "velocity_diagnostics": velocity_diagnostics,
        "summaries": summaries,
        "detail": details,
        "notes": {
            "device_access": False,
            "dispatch": False,
            "15hz_contract": "action endpoint remains authoritative",
            "30hz_contract": (
                "midpoint refresh may update velocity only; it must keep the "
                "published 15 Hz endpoint"
            ),
        },
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
