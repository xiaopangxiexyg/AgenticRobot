"""Offline RM65 teleop qref-to-qcmd smoothing experiments.

This probe consumes actuator ``command_trace.npz`` files only.  It never
initializes a device and never dispatches a command.  The recorded 60 Hz
``ik_shadow.admitted_q_rad`` stream is treated as a common upstream input so
that 15/30/60 Hz execution time laws can be compared without changing IK.

The probe deliberately reports the 15 Hz action forecast error introduced by
30/60 Hz replanning.  A high-rate candidate is not considered equivalent to
the action-authoritative 15 Hz contract merely because its qcmd is smoother.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from robot_motion.position_control import (  # noqa: E402
    AccelerationLimitedJointTrajectoryPlanner,
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonPlan,
    JointHorizonState,
    JointTrajectoryLimits,
    SampleAlignedJerkLimitedJointTrajectoryPlanner,
    final_follow_qpos_target,
)
from robot_motion.rm65 import (  # noqa: E402
    RM65_BIMANUAL_POSITION_V2_RESPONSIVE,
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


_REFERENCE_HZ = 60.0
_COMMAND_HZ = 120.0
_FORMAL_ACTION_HZ = 15.0
_HORIZON_TICKS = 16
_BOUND_GUARD = 0.999
_SIDES = ("left", "right")


@dataclass(frozen=True)
class Candidate:
    name: str
    replan_hz: int
    reference_mode: str
    jerk_ramp_sec: float
    velocity_ema_alpha: float = 0.5
    velocity_cap_scale: float = 0.5
    stop_velocity_decay: float = 0.0
    jerk_regularization: float = 1e-6
    jerk_delta_regularization: float = 1e-5
    initial_jerk_delta_regularization: float = 0.0
    velocity_observer: str = "action_delta"
    velocity_observer_window: int = 4
    observer_efficiency_power: float = 2.0
    velocity_coherence_min: float = 0.5
    terminal_velocity_displacement_scale: float = 1.5
    observer_speed_guard_scale: float = 0.4
    observer_stop_threshold_deg: float = 0.01
    observer_stop_velocity_decay: float = 0.0
    velocity_slew_accel_scale: float | None = None
    velocity_prediction_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.replan_hz not in {15, 30, 60}:
            raise ValueError("replan_hz must be one of 15, 30 or 60.")
        if self.reference_mode not in {
            "hold",
            "velocity_tail",
            "hermite",
            "guarded_hermite",
            "coherent_hermite",
            "selective_hermite",
            "coherent_quintic",
            "runtime_v2",
            "runtime_v2_nullspace",
            "runtime_v3",
            "runtime_vive_selective",
            "runtime_vive_nullspace",
            "runtime_vive_endpoint_selective",
            "runtime_vive_endpoint_selective_nullspace",
            "runtime_vive_midcut_nullspace",
            "runtime_vive_midcut_q_anchor",
            "runtime_vive_midcut_q_qdd_anchor",
            "runtime_vive_cut_q_anchor",
            "runtime_vive_predictive_action",
            "runtime_60hz_v2_responsive",
        }:
            raise ValueError(f"unsupported reference mode {self.reference_mode!r}.")
        if self.jerk_ramp_sec <= 0.0:
            raise ValueError("jerk_ramp_sec must be positive.")
        if self.velocity_slew_accel_scale is not None and not (
            0.0 < float(self.velocity_slew_accel_scale) <= 1.0
        ):
            raise ValueError("velocity_slew_accel_scale must be in (0, 1].")
        if not 0.0 <= float(self.velocity_prediction_sec) <= 1.0 / 15.0:
            raise ValueError("velocity_prediction_sec must be in [0, 1/15].")
        if self.velocity_observer not in {
            "action_delta",
            "source_median",
            "vive_dls_60hz",
        }:
            raise ValueError("unsupported velocity_observer.")
        if self.velocity_observer_window < 2:
            raise ValueError("velocity_observer_window must be at least 2.")
        if self.observer_efficiency_power < 0.0:
            raise ValueError("observer_efficiency_power must be non-negative.")
        if not -1.0 <= self.velocity_coherence_min <= 1.0:
            raise ValueError("velocity_coherence_min must be in [-1, 1].")
        if self.terminal_velocity_displacement_scale <= 0.0:
            raise ValueError(
                "terminal_velocity_displacement_scale must be positive."
            )
        if not 0.0 <= self.observer_speed_guard_scale <= 1.0:
            raise ValueError("observer_speed_guard_scale must be in [0, 1].")
        if self.observer_stop_threshold_deg < 0.0:
            raise ValueError("observer_stop_threshold_deg must be non-negative.")
        if not 0.0 <= self.observer_stop_velocity_decay <= 1.0:
            raise ValueError("observer_stop_velocity_decay must be in [0, 1].")


@dataclass(frozen=True)
class TraceInput:
    name: str
    date: str
    path: Path
    qref_by_side: dict[str, np.ndarray]
    recorded_qcmd_by_side: dict[str, np.ndarray]
    recorded_target_sequence_by_side: dict[str, np.ndarray]


def _finite_matrix(value: Any, *, columns: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"expected Nx{columns} matrix, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values.")
    return matrix


def _load_trace(path: Path) -> TraceInput:
    data_path = path / "command_trace.npz" if path.is_dir() else path
    with np.load(data_path, allow_pickle=False) as trace:
        shadow = trace["ik_shadow"]
        admitted = _finite_matrix(shadow["admitted_q_rad"], columns=12)
        qref = {
            "left": admitted[:, :6].copy(),
            "right": admitted[:, 6:].copy(),
        }
        qcmd = {
            side: _finite_matrix(trace[side]["q_cmd_rad"], columns=6).copy()
            for side in _SIDES
        }
        target_sequence = {
            side: np.asarray(trace[side]["target_sequence"], dtype=np.int64).copy()
            for side in _SIDES
        }
    return TraceInput(
        name=data_path.parent.name,
        date=data_path.parent.name[:8],
        path=data_path,
        qref_by_side=qref,
        recorded_qcmd_by_side=qcmd,
        recorded_target_sequence_by_side=target_sequence,
    )


def _build_candidates(preset: str) -> tuple[Candidate, ...]:
    baseline = [
        Candidate(
            "60_runtime_v2_responsive",
            60,
            "runtime_60hz_v2_responsive",
            0.05,
        ),
        Candidate("15_hold_r050", 15, "hold", 0.05),
        Candidate("15_runtime_v2", 15, "runtime_v2", 0.05),
        Candidate(
            "15_runtime_v2_nullspace",
            15,
            "runtime_v2_nullspace",
            0.05,
        ),
        Candidate(
            "15_runtime_v3",
            15,
            "runtime_v3",
            0.05,
            initial_jerk_delta_regularization=0.0,
            velocity_observer="source_median",
            velocity_observer_window=4,
            observer_efficiency_power=2.0,
            velocity_coherence_min=0.5,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "15_runtime_v3_conservative_a25_c25_m050_g025",
            15,
            "runtime_v3",
            0.05,
            velocity_ema_alpha=0.25,
            velocity_cap_scale=0.25,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
            observer_efficiency_power=2.0,
            velocity_coherence_min=0.5,
            terminal_velocity_displacement_scale=0.5,
            observer_speed_guard_scale=0.25,
        ),
        Candidate(
            "15_runtime_v3_conservative_a35_c35_m050_g030",
            15,
            "runtime_v3",
            0.05,
            velocity_ema_alpha=0.35,
            velocity_cap_scale=0.35,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
            observer_efficiency_power=2.0,
            velocity_coherence_min=0.6,
            terminal_velocity_displacement_scale=0.5,
            observer_speed_guard_scale=0.30,
        ),
        Candidate("15_hold_r075", 15, "hold", 0.075),
        Candidate("15_hold_r100", 15, "hold", 0.10),
    ]
    velocity = [
        Candidate("15_tail_a25_c50_r050", 15, "velocity_tail", 0.05, 0.25, 0.50),
        Candidate("15_tail_a50_c50_r050", 15, "velocity_tail", 0.05, 0.50, 0.50),
        Candidate("15_tail_a75_c50_r050", 15, "velocity_tail", 0.05, 0.75, 0.50),
        Candidate("15_tail_a50_c25_r075", 15, "velocity_tail", 0.075, 0.50, 0.25),
        Candidate("15_tail_a50_c50_r075", 15, "velocity_tail", 0.075, 0.50, 0.50),
        Candidate("15_tail_a50_c100_r075", 15, "velocity_tail", 0.075, 0.50, 1.00),
        Candidate("15_tail_a50_c50_r100", 15, "velocity_tail", 0.10, 0.50, 0.50),
        Candidate("15_hermite_a50_c50_r050", 15, "hermite", 0.05, 0.50, 0.50),
        Candidate("15_hermite_a50_c50_r075", 15, "hermite", 0.075, 0.50, 0.50),
        Candidate("15_hermite_a50_c50_r100", 15, "hermite", 0.10, 0.50, 0.50),
        Candidate(
            "15_guarded_hermite_a50_c50_r050",
            15,
            "guarded_hermite",
            0.05,
            0.50,
            0.50,
        ),
        Candidate(
            "15_guarded_hermite_a50_c50_r075",
            15,
            "guarded_hermite",
            0.075,
            0.50,
            0.50,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r075",
            15,
            "coherent_hermite",
            0.075,
            0.50,
            0.50,
        ),
        Candidate(
            "15_selective_hermite_a50_c50_r050",
            15,
            "selective_hermite",
            0.05,
            0.50,
            0.50,
        ),
        Candidate(
            "15_selective_hermite_a50_c50_r075",
            15,
            "selective_hermite",
            0.075,
            0.50,
            0.50,
        ),
    ]
    high_rate = [
        Candidate("30_hold_r050", 30, "hold", 0.05),
        Candidate("30_tail_a50_c50_r050", 30, "velocity_tail", 0.05, 0.50, 0.50),
        Candidate("30_hermite_a50_c50_r075", 30, "hermite", 0.075, 0.50, 0.50),
        Candidate("60_hold_r050", 60, "hold", 0.05),
        Candidate("60_tail_a50_c50_r050", 60, "velocity_tail", 0.05, 0.50, 0.50),
        Candidate("60_hermite_a50_c50_r075", 60, "hermite", 0.075, 0.50, 0.50),
    ]
    cross_cut_jerk = [
        Candidate(
            "15_coherent_hermite_a50_c50_r050_xj000001",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-6,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r050_xj00001",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
        ),
        Candidate(
            "15_hold_r050_xj001",
            15,
            "hold",
            0.05,
            initial_jerk_delta_regularization=1e-3,
        ),
        Candidate(
            "15_hermite_a50_c50_r050_xj001",
            15,
            "hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-3,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r050_xj0001",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-4,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r050_xj001",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-3,
        ),
        Candidate(
            "15_coherent_hermite_a50_c50_r050_xj010",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-2,
        ),
    ]
    source_observer = [
        Candidate(
            "15_observer4_coherent_hermite_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            velocity_observer="source_median",
            velocity_observer_window=4,
        ),
        Candidate(
            "15_observer8_coherent_hermite_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            velocity_observer="source_median",
            velocity_observer_window=8,
        ),
        Candidate(
            "15_observer4_coherent_hermite_r050_xj00001",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
        ),
        Candidate(
            "15_observer4_coherent_quintic_r050",
            15,
            "coherent_quintic",
            0.05,
            0.50,
            0.50,
            velocity_observer="source_median",
            velocity_observer_window=4,
        ),
        Candidate(
            "15_observer4_coherent_quintic_r050_xj00001",
            15,
            "coherent_quintic",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
        ),
        Candidate(
            "15_observer4_coherent_hermite_m075_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
            terminal_velocity_displacement_scale=0.75,
        ),
        Candidate(
            "15_observer4_coherent_hermite_m100_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
            terminal_velocity_displacement_scale=1.0,
        ),
        Candidate(
            "15_observer4_coherent_hermite_p3_r050",
            15,
            "coherent_hermite",
            0.05,
            0.50,
            0.50,
            initial_jerk_delta_regularization=1e-5,
            velocity_observer="source_median",
            velocity_observer_window=4,
            observer_efficiency_power=3.0,
        ),
    ]
    if preset == "quick":
        return tuple(
            baseline[:6]
            + [velocity[1], velocity[8], velocity[12]]
            + [cross_cut_jerk[0], cross_cut_jerk[1]]
            + source_observer
            + [high_rate[0], high_rate[3]]
        )
    if preset == "full":
        return tuple(
            baseline + velocity + cross_cut_jerk + source_observer + high_rate
        )
    raise ValueError(f"unknown preset {preset!r}.")


def _hermite_segment(
    q0: np.ndarray,
    v0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    *,
    duration_sec: float,
    steps: int,
) -> np.ndarray:
    rows = np.empty((steps + 1, q0.size), dtype=float)
    for step in range(steps + 1):
        s = step / float(steps)
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        rows[step] = (
            h00 * q0
            + h10 * duration_sec * v0
            + h01 * q1
            + h11 * duration_sec * v1
        )
    return rows


def _quintic_segment(
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    *,
    duration_sec: float,
    steps: int,
) -> np.ndarray:
    rows = np.empty((steps + 1, q0.size), dtype=float)
    for step in range(steps + 1):
        s = step / float(steps)
        h00 = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
        h10 = s - 6.0 * s**3 + 8.0 * s**4 - 3.0 * s**5
        h20 = 0.5 * s**2 - 1.5 * s**3 + 1.5 * s**4 - 0.5 * s**5
        h01 = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        h11 = -4.0 * s**3 + 7.0 * s**4 - 3.0 * s**5
        h21 = 0.5 * s**3 - s**4 + 0.5 * s**5
        rows[step] = (
            h00 * q0
            + h10 * duration_sec * v0
            + h20 * duration_sec**2 * a0
            + h01 * q1
            + h11 * duration_sec * v1
            + h21 * duration_sec**2 * a1
        )
    return rows


def _reference_horizon(
    *,
    candidate: Candidate,
    state: JointHorizonState,
    target: np.ndarray,
    target_velocity: np.ndarray,
    commands_per_replan: int,
    effective_reference_mode: str | None = None,
) -> np.ndarray:
    dt = 1.0 / _COMMAND_HZ
    reference = np.empty((_HORIZON_TICKS + 1, 6), dtype=float)
    mode = candidate.reference_mode if effective_reference_mode is None else str(
        effective_reference_mode
    )
    if mode == "coherent_quintic":
        prefix = _quintic_segment(
            state.q_rad,
            state.qd_rad_s,
            state.qdd_rad_s2,
            target,
            target_velocity,
            np.zeros(6, dtype=float),
            duration_sec=commands_per_replan * dt,
            steps=commands_per_replan,
        )
    elif mode in {
        "hermite",
        "guarded_hermite",
        "coherent_hermite",
        "selective_hermite",
    }:
        prefix = _hermite_segment(
            state.q_rad,
            state.qd_rad_s,
            target,
            target_velocity,
            duration_sec=commands_per_replan * dt,
            steps=commands_per_replan,
        )
    else:
        prefix = np.empty((commands_per_replan + 1, 6), dtype=float)
        for step in range(commands_per_replan + 1):
            alpha = step / float(commands_per_replan)
            prefix[step] = (1.0 - alpha) * state.q_rad + alpha * target
    reference[: commands_per_replan + 1] = prefix
    tail_velocity = (
        np.zeros(6, dtype=float)
        if mode == "hold"
        else target_velocity
    )
    for step in range(commands_per_replan + 1, _HORIZON_TICKS + 1):
        reference[step] = target + (
            (step - commands_per_replan) * dt * tail_velocity
        )
    # Match the production planner/QP interior exactly.  A looser reference
    # clip makes offline Hermite candidates look artificially better on real
    # J3/J6 boundary traces and eventually changes the closed-loop anchor.
    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float) + 1e-4
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float) - 1e-4
    return np.clip(reference, lower, upper)


def _simulate(
    source_qref_60hz: np.ndarray,
    candidate: Candidate,
    *,
    external_velocity_60hz: np.ndarray | None = None,
    external_velocity_valid_60hz: np.ndarray | None = None,
) -> dict[str, Any]:
    qref = _finite_matrix(source_qref_60hz, columns=6)
    external_velocity = None
    if external_velocity_60hz is not None:
        external_velocity = _finite_matrix(external_velocity_60hz, columns=6)
        if external_velocity.shape != qref.shape:
            raise ValueError(
                "external_velocity_60hz must match the 60 Hz qref shape."
            )
        if candidate.velocity_slew_accel_scale is not None:
            governed = np.empty_like(external_velocity)
            governed[0] = external_velocity[0]
            delta_limit = (
                float(candidate.velocity_slew_accel_scale)
                * np.asarray(
                    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
                    dtype=float,
                )
                / _REFERENCE_HZ
            )
            for index in range(1, external_velocity.shape[0]):
                delta = np.clip(
                    external_velocity[index] - governed[index - 1],
                    -delta_limit,
                    delta_limit,
                )
                governed[index] = governed[index - 1] + delta
            external_velocity = governed
    external_velocity_valid = None
    if external_velocity_valid_60hz is not None:
        external_velocity_valid = np.asarray(
            external_velocity_valid_60hz, dtype=bool
        ).reshape(-1)
        if external_velocity_valid.shape != (qref.shape[0],):
            raise ValueError(
                "external_velocity_valid_60hz must match the 60 Hz qref rows."
            )
        if external_velocity is None:
            raise ValueError(
                "external_velocity_valid_60hz requires external_velocity_60hz."
            )
    if candidate.velocity_observer == "vive_dls_60hz" and external_velocity is None:
        raise ValueError(
            "vive_dls_60hz requires an explicit causal 60 Hz velocity stream."
        )
    if candidate.reference_mode in {
        "runtime_v2",
        "runtime_v2_nullspace",
        "runtime_v3",
        "runtime_vive_selective",
        "runtime_vive_nullspace",
        "runtime_vive_endpoint_selective",
        "runtime_vive_endpoint_selective_nullspace",
        "runtime_vive_midcut_nullspace",
        "runtime_vive_midcut_q_anchor",
        "runtime_vive_midcut_q_qdd_anchor",
        "runtime_vive_cut_q_anchor",
        "runtime_vive_predictive_action",
    }:
        return _simulate_runtime_sample_aligned(
            qref,
            candidate,
            external_velocity_60hz=external_velocity,
            external_velocity_valid_60hz=external_velocity_valid,
        )
    if candidate.reference_mode == "runtime_60hz_v2_responsive":
        return _simulate_runtime_60hz_v2_responsive(qref)
    stride_60 = int(round(_REFERENCE_HZ / candidate.replan_hz))
    commands_per_replan = int(round(_COMMAND_HZ / candidate.replan_hz))
    if qref.shape[0] < stride_60 + 1:
        raise ValueError("qref trace is too short for the selected replan rate.")
    # Keep an explicit 1e-4 rad offline interior.  Several real traces run J3
    # close to its absolute boundary; the experiment must not rank a candidate
    # by relying on an OSQP residual at the exact hardware limit.
    position_interior = 1e-4
    effective_velocity = np.minimum(
        np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S, dtype=float),
        math.radians(RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG) * _COMMAND_HZ,
    )
    limits = JointHorizonLimits(
        min_position_rad=(
            np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
            + position_interior
        ),
        max_position_rad=(
            np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
            - position_interior
        ),
        max_velocity_rad_s=effective_velocity,
        max_acceleration_rad_s2=RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        max_jerk_rad_s3=(
            np.asarray(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float)
            / candidate.jerk_ramp_sec
        ),
    )
    workspace = JointHorizonBatchedWorkspace(
        limits=limits,
        dt_sec=1.0 / _COMMAND_HZ,
        horizon_transitions=_HORIZON_TICKS,
        bound_guard=_BOUND_GUARD,
        # Use the stricter offline tolerance.  The returned path is also
        # independently checked against the hard limits below.
        solver_tolerance=1e-7,
        jerk_regularization=candidate.jerk_regularization,
        jerk_delta_regularization=candidate.jerk_delta_regularization,
        initial_jerk_delta_regularization=(
            candidate.initial_jerk_delta_regularization
        ),
    )
    state = JointHorizonState(
        q_rad=qref[0],
        qd_rad_s=np.zeros(6, dtype=float),
        qdd_rad_s2=np.zeros(6, dtype=float),
    )
    command_rows = [np.asarray(state.q_rad, dtype=float).copy()]
    tracking_reference_rows = [np.asarray(state.q_rad, dtype=float).copy()]
    velocity_rows = [np.asarray(state.qd_rad_s, dtype=float).copy()]
    acceleration_rows = [np.asarray(state.qdd_rad_s2, dtype=float).copy()]
    command_source_index = [0]
    formal_forecasts: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    previous_target: np.ndarray | None = None
    target_velocity = np.zeros(6, dtype=float)
    raw_target_velocity = np.zeros(6, dtype=float)
    previous_raw_target_velocity = np.zeros(6, dtype=float)
    previous_jerk = np.zeros(6, dtype=float)
    selection_counts = {"hold": 0, "hermite": 0}
    source_dt = 1.0 / candidate.replan_hz
    max_target_velocity = (
        candidate.velocity_cap_scale
        * effective_velocity
    )
    for source_index in range(0, qref.shape[0] - stride_60, stride_60):
        target = qref[source_index]
        if previous_target is not None:
            action_delta_velocity = (target - previous_target) / source_dt
            raw_velocity = action_delta_velocity
            if candidate.velocity_observer == "vive_dls_60hz":
                assert external_velocity is not None
                raw_velocity = external_velocity[source_index].copy()
            elif candidate.velocity_observer == "source_median":
                window_start = max(
                    0,
                    source_index - candidate.velocity_observer_window,
                )
                source_steps = np.diff(
                    qref[window_start : source_index + 1],
                    axis=0,
                )
                if source_steps.shape[0] >= 2:
                    path_length = np.sum(np.abs(source_steps), axis=0)
                    path_efficiency = np.abs(
                        np.sum(source_steps, axis=0)
                    ) / np.maximum(path_length, 1e-12)
                    raw_velocity = (
                        np.median(source_steps, axis=0)
                        * _REFERENCE_HZ
                        * np.power(
                            path_efficiency,
                            candidate.observer_efficiency_power,
                        )
                    )
            previous_raw_target_velocity = raw_target_velocity.copy()
            raw_target_velocity = raw_velocity.copy()
            if np.max(np.abs(target - previous_target)) < math.radians(0.01):
                raw_velocity = candidate.stop_velocity_decay * target_velocity
            target_velocity = (
                (1.0 - candidate.velocity_ema_alpha) * target_velocity
                + candidate.velocity_ema_alpha * raw_velocity
            )
            target_velocity = np.clip(
                target_velocity,
                -max_target_velocity,
                max_target_velocity,
            )
        previous_target = target.copy()
        effective_mode = candidate.reference_mode
        effective_target_velocity = target_velocity.copy()
        if candidate.reference_mode in {
            "guarded_hermite",
            "coherent_hermite",
            "selective_hermite",
            "coherent_quintic",
        }:
            displacement = target - state.q_rad
            duration = commands_per_replan / _COMMAND_HZ
            # Hermite is only useful when its terminal velocity continues
            # toward the current endpoint and remains monotone over the cut.
            wrong_direction = effective_target_velocity * displacement <= 0.0
            monotone_cap = (
                candidate.terminal_velocity_displacement_scale
                * np.abs(displacement)
                / max(duration, 1e-9)
            )
            effective_target_velocity = np.sign(effective_target_velocity) * np.minimum(
                np.abs(effective_target_velocity), monotone_cap
            )
            effective_target_velocity[wrong_direction] = 0.0
            source_speed_ratio = float(
                np.max(
                    np.abs(raw_target_velocity)
                    / np.maximum(
                        np.asarray(
                            effective_velocity,
                            dtype=float,
                        ),
                        1e-9,
                    )
                )
            )
            if (
                candidate.reference_mode != "selective_hermite"
                and source_speed_ratio > 0.40
            ):
                effective_mode = "hold"
                effective_target_velocity.fill(0.0)
            if candidate.reference_mode in {
                "coherent_hermite",
                "coherent_quintic",
            }:
                previous_norm = float(np.linalg.norm(previous_raw_target_velocity))
                current_norm = float(np.linalg.norm(raw_target_velocity))
                coherence = float(
                    np.dot(
                        previous_raw_target_velocity,
                        raw_target_velocity,
                    )
                    / max(previous_norm * current_norm, 1e-12)
                )
                if (
                    previous_norm <= 1e-6
                    or current_norm <= 1e-6
                    or coherence < candidate.velocity_coherence_min
                ):
                    effective_mode = "hold"
                    effective_target_velocity.fill(0.0)
        reference = _reference_horizon(
            candidate=candidate,
            state=state,
            target=target,
            target_velocity=effective_target_velocity,
            commands_per_replan=commands_per_replan,
            effective_reference_mode=effective_mode,
        )
        try:
            if candidate.reference_mode == "selective_hermite":
                hold_reference = _reference_horizon(
                    candidate=candidate,
                    state=state,
                    target=target,
                    target_velocity=np.zeros(6, dtype=float),
                    commands_per_replan=commands_per_replan,
                    effective_reference_mode="hold",
                )
                hold_plan = workspace.solve(
                    hold_reference,
                    initial_state=state,
                    commit_count=commands_per_replan,
                    previous_jerk_rad_s3=previous_jerk,
                )
                hermite_plan = workspace.solve(
                    reference,
                    initial_state=state,
                    commit_count=commands_per_replan,
                    previous_jerk_rad_s3=previous_jerk,
                )

                def _smoothness(plan: Any) -> float:
                    jerk = np.diff(plan.qdd_path_rad_s2, axis=0) * _COMMAND_HZ
                    normalized = jerk / np.asarray(
                        limits.max_jerk_rad_s3, dtype=float
                    )
                    chain = np.vstack(
                        (
                            previous_jerk
                            / np.asarray(limits.max_jerk_rad_s3, dtype=float),
                            normalized,
                        )
                    )
                    return float(
                        np.mean(np.square(normalized))
                        + 0.25 * np.mean(np.square(np.diff(chain, axis=0)))
                    )

                hold_error = float(
                    np.max(
                        np.abs(
                            hold_plan.q_path_rad[commands_per_replan] - target
                        )
                    )
                )
                hermite_error = float(
                    np.max(
                        np.abs(
                            hermite_plan.q_path_rad[commands_per_replan] - target
                        )
                    )
                )
                choose_hermite = bool(
                    hermite_error <= hold_error + 1e-9
                    and hermite_error <= math.radians(0.15)
                    and _smoothness(hermite_plan) <= 0.98 * _smoothness(hold_plan)
                )
                plan = hermite_plan if choose_hermite else hold_plan
                selection_counts["hermite" if choose_hermite else "hold"] += 1
            else:
                plan = workspace.solve(
                    reference,
                    initial_state=state,
                    commit_count=commands_per_replan,
                    previous_jerk_rad_s3=previous_jerk,
                )
        except Exception as exc:
            raise RuntimeError(
                f"candidate={candidate.name} source_index={source_index} "
                f"failed: {exc}"
            ) from exc
        command_start = len(command_rows) - 1
        if source_index % int(_REFERENCE_HZ / _FORMAL_ACTION_HZ) == 0:
            formal_forecasts.append(
                (
                    source_index,
                    command_start,
                    np.asarray(plan.q_path_rad[8], dtype=float).copy(),
                    target.copy(),
                )
            )
        for step in range(1, commands_per_replan + 1):
            command_rows.append(np.asarray(plan.q_path_rad[step], dtype=float).copy())
            tracking_reference_rows.append(target.copy())
            velocity_rows.append(
                np.asarray(plan.qd_path_rad_s[step], dtype=float).copy()
            )
            acceleration_rows.append(
                np.asarray(plan.qdd_path_rad_s2[step], dtype=float).copy()
            )
            command_source_index.append(source_index)
        state = JointHorizonState(
            q_rad=np.asarray(plan.q_path_rad[commands_per_replan], dtype=float),
            qd_rad_s=np.asarray(
                plan.qd_path_rad_s[commands_per_replan], dtype=float
            ),
            qdd_rad_s2=np.asarray(
                plan.qdd_path_rad_s2[commands_per_replan], dtype=float
            ),
        )
        previous_jerk = (
            np.asarray(plan.qdd_path_rad_s2[commands_per_replan], dtype=float)
            - np.asarray(
                plan.qdd_path_rad_s2[commands_per_replan - 1], dtype=float
            )
        ) * _COMMAND_HZ
    command = np.asarray(command_rows, dtype=float)
    velocity = np.asarray(velocity_rows, dtype=float)
    acceleration = np.asarray(acceleration_rows, dtype=float)
    action_error = []
    action_state_error = []
    requested_next_state_error = []
    for _source_index, command_start, forecast, requested in formal_forecasts:
        next_index = command_start + int(_COMMAND_HZ / _FORMAL_ACTION_HZ)
        if next_index >= command.shape[0]:
            continue
        next_state = command[next_index]
        action_error.append(forecast - requested)
        action_state_error.append(next_state - forecast)
        requested_next_state_error.append(next_state - requested)
    return {
        "q_rad": command,
        "qd_rad_s": velocity,
        "qdd_rad_s2": acceleration,
        "tracking_reference_rad": np.asarray(
            tracking_reference_rows,
            dtype=float,
        ),
        "command_source_index": np.asarray(command_source_index, dtype=np.int64),
        "action_error_rad": np.asarray(action_error, dtype=float),
        "action_state_error_rad": np.asarray(action_state_error, dtype=float),
        "requested_next_state_error_rad": np.asarray(
            requested_next_state_error, dtype=float
        ),
        "max_jerk_rad_s3": np.asarray(limits.max_jerk_rad_s3, dtype=float),
        "selection_counts": selection_counts,
    }


def _midcut_velocity_nullspace_plan(
    plan: JointHorizonPlan,
    *,
    target_velocity_rad_s: np.ndarray,
    limits: JointHorizonLimits,
) -> tuple[JointHorizonPlan, dict[str, Any]]:
    """Use a 30 Hz velocity refresh without changing the 15 Hz q/v/a anchor."""

    commit = int(plan.commit_count)
    midpoint = commit // 2
    remaining = commit - midpoint
    if commit % 2 or remaining < 4:
        return plan, {"midcut_selected_joint_count": 0, "midcut_reason": "shape"}
    dt = float(plan.dt_sec)
    q = np.asarray(plan.q_path_rad, dtype=float).copy()
    qd = np.asarray(plan.qd_path_rad_s, dtype=float).copy()
    qdd = np.asarray(plan.qdd_path_rad_s2, dtype=float).copy()
    target_velocity = np.asarray(target_velocity_rad_s, dtype=float).reshape(-1)
    if target_velocity.shape != (q.shape[1],):
        raise ValueError("midcut target velocity dimension mismatch.")

    ticks = np.arange(remaining, 0, -1, dtype=float)
    terminal_map = np.vstack(
        (
            dt * np.ones(remaining, dtype=float),
            dt**2 * ticks,
            dt**3 * ticks * (ticks + 1.0) / 2.0,
        )
    )
    _u, _s, vh = np.linalg.svd(terminal_map, full_matrices=True)
    null_direction = np.asarray(vh[-1], dtype=float)
    null_direction /= max(float(np.linalg.norm(null_direction)), 1e-12)
    selected_joints = 0
    rejection_counts = {
        "terminal": 0,
        "bounds": 0,
        "saturation": 0,
        "smoothness": 0,
        "active_set": 0,
        "direction": 0,
        "velocity_gain": 0,
    }
    velocity_improvement_by_joint = np.zeros(q.shape[1], dtype=float)
    alpha_by_joint = np.zeros(q.shape[1], dtype=float)

    def integrate(
        *,
        joint: int,
        jerk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_q = np.empty(remaining + 1, dtype=float)
        out_qd = np.empty(remaining + 1, dtype=float)
        out_qdd = np.empty(remaining + 1, dtype=float)
        out_q[0] = q[midpoint, joint]
        out_qd[0] = qd[midpoint, joint]
        out_qdd[0] = qdd[midpoint, joint]
        for step in range(remaining):
            out_qdd[step + 1] = out_qdd[step] + dt * jerk[step]
            out_qd[step + 1] = out_qd[step] + dt * out_qdd[step + 1]
            out_q[step + 1] = out_q[step] + dt * out_qd[step + 1]
        return out_q, out_qd, out_qdd

    for joint in range(q.shape[1]):
        baseline_jerk = (
            np.diff(qdd[midpoint : commit + 1, joint]) / dt
        )
        baseline_q = q[midpoint : commit + 1, joint]
        baseline_qd = qd[midpoint : commit + 1, joint]
        baseline_qdd = qdd[midpoint : commit + 1, joint]
        _, basis_qd, _ = integrate(joint=joint, jerk=null_direction)
        _, zero_qd, _ = integrate(
            joint=joint,
            jerk=np.zeros(remaining, dtype=float),
        )
        velocity_basis = basis_qd - zero_qd
        # The terminal velocity is an exact anchor, so only the three interior
        # 120 Hz states can respond to the new 30 Hz observation.
        interior_basis = velocity_basis[1:remaining]
        baseline_error = (
            baseline_qd[1:remaining] - target_velocity[joint]
        )
        denominator = float(np.dot(interior_basis, interior_basis))
        if denominator <= 1e-14:
            continue
        ideal_alpha = -float(np.dot(interior_basis, baseline_error)) / denominator
        if not math.isfinite(ideal_alpha):
            continue
        jerk_limit = max(float(limits.max_jerk_rad_s3[joint]), 1e-12)
        previous_jerk = (
            float(qdd[midpoint, joint] - qdd[midpoint - 1, joint]) / dt
        )

        def score(jerk: np.ndarray) -> float:
            chain = np.r_[previous_jerk, jerk] / jerk_limit
            return float(
                np.mean(np.square(jerk / jerk_limit))
                + 0.25 * np.mean(np.square(np.diff(chain)))
            )

        baseline_score = score(baseline_jerk)
        baseline_saturation = int(
            np.sum(np.abs(baseline_jerk / jerk_limit) >= 0.98)
        )
        baseline_velocity_error = float(np.mean(np.square(baseline_error)))
        if baseline_velocity_error <= 1e-12:
            continue
        baseline_steps = np.diff(baseline_q)
        previous_step = float(qd[midpoint, joint]) * dt
        baseline_chain = np.r_[previous_step, baseline_steps]
        active_threshold = math.radians(0.002)
        baseline_active = np.abs(baseline_chain) >= active_threshold

        for scale in (1.0, 0.75, 0.5, 0.25, 0.125):
            alpha = scale * ideal_alpha
            candidate_jerk = baseline_jerk + alpha * null_direction
            candidate_q, candidate_qd, candidate_qdd = integrate(
                joint=joint,
                jerk=candidate_jerk,
            )
            terminal_error = max(
                abs(candidate_q[-1] - baseline_q[-1]),
                abs(candidate_qd[-1] - baseline_qd[-1]),
                abs(candidate_qdd[-1] - baseline_qdd[-1]),
            )
            if terminal_error > 1e-9:
                rejection_counts["terminal"] += 1
                continue
            candidate_q[-1] = baseline_q[-1]
            candidate_qd[-1] = baseline_qd[-1]
            candidate_qdd[-1] = baseline_qdd[-1]
            candidate_jerk = np.diff(candidate_qdd) / dt
            if np.any(candidate_q < limits.min_position_rad[joint] - 1e-10):
                rejection_counts["bounds"] += 1
                continue
            if np.any(candidate_q > limits.max_position_rad[joint] + 1e-10):
                rejection_counts["bounds"] += 1
                continue
            if np.any(
                np.abs(candidate_qd)
                > limits.max_velocity_rad_s[joint] + 1e-10
            ):
                rejection_counts["bounds"] += 1
                continue
            if np.any(
                np.abs(candidate_qdd)
                > limits.max_acceleration_rad_s2[joint] + 1e-9
            ):
                rejection_counts["bounds"] += 1
                continue
            if np.any(
                np.abs(candidate_jerk)
                > limits.max_jerk_rad_s3[joint] + 1e-8
            ):
                rejection_counts["bounds"] += 1
                continue
            candidate_saturation = int(
                np.sum(np.abs(candidate_jerk / jerk_limit) >= 0.98)
            )
            if candidate_saturation > baseline_saturation:
                rejection_counts["saturation"] += 1
                continue
            if score(candidate_jerk) > 1.10 * baseline_score + 1e-12:
                rejection_counts["smoothness"] += 1
                continue
            candidate_chain = np.r_[previous_step, np.diff(candidate_q)]
            candidate_active = np.abs(candidate_chain) >= active_threshold
            if np.any(candidate_active != baseline_active):
                rejection_counts["active_set"] += 1
                continue
            if np.any(
                np.sign(candidate_chain[baseline_active])
                != np.sign(baseline_chain[baseline_active])
            ):
                rejection_counts["direction"] += 1
                continue
            candidate_velocity_error = float(
                np.mean(
                    np.square(
                        candidate_qd[1:remaining] - target_velocity[joint]
                    )
                )
            )
            if candidate_velocity_error > 0.98 * baseline_velocity_error:
                rejection_counts["velocity_gain"] += 1
                continue
            q[midpoint : commit + 1, joint] = candidate_q
            qd[midpoint : commit + 1, joint] = candidate_qd
            qdd[midpoint : commit + 1, joint] = candidate_qdd
            selected_joints += 1
            velocity_improvement_by_joint[joint] = (
                1.0 - candidate_velocity_error / baseline_velocity_error
            )
            alpha_by_joint[joint] = alpha
            break

    if selected_joints == 0:
        return plan, {
            "midcut_selected_joint_count": 0,
            "midcut_reason": "no_certified_improvement",
            "midcut_rejection_counts": rejection_counts,
        }
    metadata = {
        **dict(plan.metadata),
        "midcut_velocity_nullspace_selected": True,
        "midcut_selected_joint_count": selected_joints,
        "midcut_velocity_improvement_by_joint": (
            velocity_improvement_by_joint.tolist()
        ),
        "midcut_alpha_by_joint": alpha_by_joint.tolist(),
        "midcut_terminal_anchor": "exact_q_qd_qdd",
        "midcut_rejection_counts": rejection_counts,
    }
    return JointHorizonPlan(
        q_path_rad=q,
        qd_path_rad_s=qd,
        qdd_path_rad_s2=qdd,
        commit_count=plan.commit_count,
        dt_sec=plan.dt_sec,
        metadata=metadata,
    ), metadata


def _midcut_velocity_q_anchor_plan(
    plan: JointHorizonPlan,
    *,
    target_velocity_rad_s: np.ndarray,
    limits: JointHorizonLimits,
    start_row: int | None = None,
    previous_jerk_rad_s3: np.ndarray | None = None,
    velocity_joint_valid: np.ndarray | None = None,
    anchor_terminal_acceleration: bool = False,
    max_smoothness_ratio: float = 1.02,
) -> tuple[JointHorizonPlan, dict[str, Any]]:
    """Refresh a cut while preserving formal q and optionally terminal qdd."""

    commit = int(plan.commit_count)
    midpoint = commit // 2 if start_row is None else int(start_row)
    remaining = commit - midpoint
    if midpoint < 0 or midpoint >= commit or remaining < 4:
        return plan, {"midcut_selected_joint_count": 0, "midcut_reason": "shape"}
    dt = float(plan.dt_sec)
    q = np.asarray(plan.q_path_rad, dtype=float).copy()
    qd = np.asarray(plan.qd_path_rad_s, dtype=float).copy()
    qdd = np.asarray(plan.qdd_path_rad_s2, dtype=float).copy()
    target_velocity = np.asarray(target_velocity_rad_s, dtype=float).reshape(-1)
    if target_velocity.shape != (q.shape[1],):
        raise ValueError("midcut target velocity dimension mismatch.")
    smoothness_ratio = float(max_smoothness_ratio)
    if not math.isfinite(smoothness_ratio) or smoothness_ratio <= 0.0:
        raise ValueError("max_smoothness_ratio must be finite and positive.")
    joint_valid = (
        np.ones(q.shape[1], dtype=bool)
        if velocity_joint_valid is None
        else np.asarray(velocity_joint_valid, dtype=bool).reshape(q.shape[1])
    )

    ticks = np.arange(remaining, 0, -1, dtype=float)
    terminal_q_map = (
        dt**3 * ticks * (ticks + 1.0) / 2.0
    ).reshape(1, -1)
    terminal_maps = [terminal_q_map]
    if anchor_terminal_acceleration:
        terminal_maps.append(
            (dt * np.ones(remaining, dtype=float)).reshape(1, -1)
        )
    terminal_map = np.vstack(terminal_maps)
    _u, singular_values, vh = np.linalg.svd(terminal_map, full_matrices=True)
    rank = int(
        np.sum(
            singular_values
            > max(terminal_map.shape)
            * np.finfo(float).eps
            * max(float(singular_values[0]), 1.0)
        )
    )
    null_basis = np.asarray(vh[rank:].T, dtype=float)
    if null_basis.shape[1] == 0:
        return plan, {
            "midcut_selected_joint_count": 0,
            "midcut_reason": "no_terminal_nullspace",
        }

    def integrate(
        *,
        joint: int,
        jerk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_q = np.empty(remaining + 1, dtype=float)
        out_qd = np.empty(remaining + 1, dtype=float)
        out_qdd = np.empty(remaining + 1, dtype=float)
        out_q[0] = q[midpoint, joint]
        out_qd[0] = qd[midpoint, joint]
        out_qdd[0] = qdd[midpoint, joint]
        for step in range(remaining):
            out_qdd[step + 1] = out_qdd[step] + dt * jerk[step]
            out_qd[step + 1] = out_qd[step] + dt * out_qdd[step + 1]
            out_q[step + 1] = out_q[step] + dt * out_qd[step + 1]
        return out_q, out_qd, out_qdd

    # Map a jerk perturbation to the four remaining velocity states.
    velocity_sensitivity = np.empty((remaining, remaining), dtype=float)
    for column in range(remaining):
        basis_jerk = np.zeros(remaining, dtype=float)
        basis_jerk[column] = 1.0
        _, basis_qd, _ = integrate(joint=0, jerk=basis_jerk)
        _, zero_qd, _ = integrate(
            joint=0,
            jerk=np.zeros(remaining, dtype=float),
        )
        velocity_sensitivity[:, column] = basis_qd[1:] - zero_qd[1:]
    reduced_velocity_map = velocity_sensitivity @ null_basis

    selected_joints = 0
    selected_alpha = np.zeros(q.shape[1], dtype=float)
    velocity_improvement = np.zeros(q.shape[1], dtype=float)
    rejection_counts = {
        "bounds": 0,
        "saturation": 0,
        "smoothness": 0,
        "active_set": 0,
        "direction": 0,
        "velocity_gain": 0,
        "terminal_dynamics": 0,
    }
    for joint in range(q.shape[1]):
        if not joint_valid[joint]:
            continue
        baseline_q = q[midpoint : commit + 1, joint]
        baseline_qd = qd[midpoint : commit + 1, joint]
        baseline_qdd = qdd[midpoint : commit + 1, joint]
        baseline_jerk = np.diff(baseline_qdd) / dt
        displacement = float(baseline_q[-1] - baseline_q[0])
        if (
            abs(displacement) < math.radians(0.002)
            or target_velocity[joint] * displacement <= 0.0
        ):
            continue
        desired = np.full(remaining, target_velocity[joint], dtype=float)
        residual = desired - baseline_qd[1:]
        baseline_velocity_error = float(np.mean(np.square(residual)))
        if baseline_velocity_error <= 1e-12:
            continue
        normal = reduced_velocity_map.T @ reduced_velocity_map
        ridge = 1e-3 * max(float(np.trace(normal)), 1e-12)
        reduced_step = np.linalg.solve(
            normal + ridge * np.eye(null_basis.shape[1], dtype=float),
            reduced_velocity_map.T @ residual,
        )
        jerk_delta = null_basis @ reduced_step
        jerk_limit = max(float(limits.max_jerk_rad_s3[joint]), 1e-12)
        previous_jerk = (
            float(qdd[midpoint, joint] - qdd[midpoint - 1, joint]) / dt
            if midpoint > 0
            else (
                float(np.asarray(previous_jerk_rad_s3, dtype=float)[joint])
                if previous_jerk_rad_s3 is not None
                else 0.0
            )
        )

        def score(jerk: np.ndarray) -> float:
            chain = np.r_[previous_jerk, jerk] / jerk_limit
            return float(
                np.mean(np.square(jerk / jerk_limit))
                + 0.25 * np.mean(np.square(np.diff(chain)))
            )

        baseline_score = score(baseline_jerk)
        baseline_saturation = int(
            np.sum(np.abs(baseline_jerk / jerk_limit) >= 0.98)
        )
        previous_step = float(qd[midpoint, joint]) * dt
        baseline_chain = np.r_[previous_step, np.diff(baseline_q)]
        active_threshold = math.radians(0.002)
        baseline_active = np.abs(baseline_chain) >= active_threshold

        for alpha in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
            candidate_jerk = baseline_jerk + alpha * jerk_delta
            candidate_q, candidate_qd, candidate_qdd = integrate(
                joint=joint,
                jerk=candidate_jerk,
            )
            if abs(candidate_q[-1] - baseline_q[-1]) > 1e-9:
                rejection_counts["bounds"] += 1
                continue
            candidate_q[-1] = baseline_q[-1]
            if anchor_terminal_acceleration:
                if abs(candidate_qdd[-1] - baseline_qdd[-1]) > 1e-9:
                    rejection_counts["terminal_dynamics"] += 1
                    continue
                candidate_qdd[-1] = baseline_qdd[-1]
            candidate_jerk = np.diff(candidate_qdd) / dt
            if (
                abs(candidate_qd[-1] - baseline_qd[-1])
                > math.radians(2.0)
                or (
                    not anchor_terminal_acceleration
                    and abs(candidate_qdd[-1] - baseline_qdd[-1])
                    > math.radians(40.0)
                )
            ):
                rejection_counts["terminal_dynamics"] += 1
                continue
            if (
                np.any(candidate_q < limits.min_position_rad[joint] - 1e-10)
                or np.any(candidate_q > limits.max_position_rad[joint] + 1e-10)
                or np.any(
                    np.abs(candidate_qd)
                    > limits.max_velocity_rad_s[joint] + 1e-10
                )
                or np.any(
                    np.abs(candidate_qdd)
                    > limits.max_acceleration_rad_s2[joint] + 1e-9
                )
                or np.any(
                    np.abs(candidate_jerk)
                    > limits.max_jerk_rad_s3[joint] + 1e-8
                )
            ):
                rejection_counts["bounds"] += 1
                continue
            candidate_saturation = int(
                np.sum(np.abs(candidate_jerk / jerk_limit) >= 0.98)
            )
            if candidate_saturation > baseline_saturation:
                rejection_counts["saturation"] += 1
                continue
            if score(candidate_jerk) > smoothness_ratio * baseline_score + 1e-12:
                rejection_counts["smoothness"] += 1
                continue
            candidate_chain = np.r_[previous_step, np.diff(candidate_q)]
            candidate_active = np.abs(candidate_chain) >= active_threshold
            if np.any(candidate_active != baseline_active):
                rejection_counts["active_set"] += 1
                continue
            if np.any(
                np.sign(candidate_chain[baseline_active])
                != np.sign(baseline_chain[baseline_active])
            ):
                rejection_counts["direction"] += 1
                continue
            candidate_velocity_error = float(
                np.mean(np.square(candidate_qd[1:] - desired))
            )
            if candidate_velocity_error > 0.98 * baseline_velocity_error:
                rejection_counts["velocity_gain"] += 1
                continue
            q[midpoint : commit + 1, joint] = candidate_q
            qd[midpoint : commit + 1, joint] = candidate_qd
            qdd[midpoint : commit + 1, joint] = candidate_qdd
            selected_joints += 1
            selected_alpha[joint] = alpha
            velocity_improvement[joint] = (
                1.0 - candidate_velocity_error / baseline_velocity_error
            )
            break

    if selected_joints == 0:
        return plan, {
            "midcut_selected_joint_count": 0,
            "midcut_reason": "no_certified_improvement",
            "midcut_rejection_counts": rejection_counts,
        }
    metadata = {
        **dict(plan.metadata),
        "midcut_velocity_q_anchor_selected": True,
        "midcut_selected_joint_count": selected_joints,
        "midcut_velocity_improvement_by_joint": velocity_improvement.tolist(),
        "midcut_alpha_by_joint": selected_alpha.tolist(),
        "midcut_terminal_anchor": (
            "exact_q_qdd" if anchor_terminal_acceleration else "exact_q"
        ),
        "midcut_max_smoothness_ratio": smoothness_ratio,
        "midcut_rejection_counts": rejection_counts,
    }
    return JointHorizonPlan(
        q_path_rad=q,
        qd_path_rad_s=qd,
        qdd_path_rad_s2=qdd,
        commit_count=plan.commit_count,
        dt_sec=plan.dt_sec,
        metadata=metadata,
    ), metadata


def _simulate_runtime_sample_aligned(
    qref: np.ndarray,
    candidate: Candidate,
    *,
    external_velocity_60hz: np.ndarray | None = None,
    external_velocity_valid_60hz: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay the production V2/V3 planner at its real 60/120 Hz cadence."""

    if qref.shape[0] < 2:
        raise ValueError("runtime V3 replay requires at least two 60 Hz rows.")
    external_velocity = None
    if external_velocity_60hz is not None:
        external_velocity = _finite_matrix(external_velocity_60hz, columns=6)
        if external_velocity.shape != qref.shape:
            raise ValueError(
                "external_velocity_60hz must match the runtime qref shape."
            )
    external_velocity_valid = None
    if external_velocity_valid_60hz is not None:
        external_velocity_valid = np.asarray(
            external_velocity_valid_60hz, dtype=bool
        ).reshape(-1)
        if external_velocity_valid.shape != (qref.shape[0],):
            raise ValueError(
                "external_velocity_valid_60hz must match the runtime qref rows."
            )
        if external_velocity is None:
            raise ValueError(
                "external_velocity_valid_60hz requires external_velocity_60hz."
            )
    if (
        candidate.reference_mode in {
            "runtime_vive_selective",
            "runtime_vive_nullspace",
            "runtime_vive_endpoint_selective",
            "runtime_vive_endpoint_selective_nullspace",
            "runtime_vive_midcut_nullspace",
            "runtime_vive_midcut_q_anchor",
            "runtime_vive_midcut_q_qdd_anchor",
            "runtime_vive_cut_q_anchor",
            "runtime_vive_predictive_action",
        }
        and external_velocity is None
    ):
        raise ValueError(
            f"{candidate.reference_mode} requires external 60 Hz velocity."
        )
    effective_velocity = np.minimum(
        np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S, dtype=float),
        math.radians(RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG) * _COMMAND_HZ,
    )
    position_interior = 1e-4
    limits = JointHorizonLimits(
        min_position_rad=(
            np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float) + position_interior
        ),
        max_position_rad=(
            np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float) - position_interior
        ),
        max_velocity_rad_s=effective_velocity,
        max_acceleration_rad_s2=RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        max_jerk_rad_s3=(
            np.asarray(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float)
            / candidate.jerk_ramp_sec
        ),
    )
    planner = SampleAlignedJerkLimitedJointTrajectoryPlanner(
        limits,
        source_hz=_FORMAL_ACTION_HZ,
        command_hz=_COMMAND_HZ,
        horizon_source_steps=2,
        solver_tolerance=1e-7,
        scale_degraded_target=False,
        jerk_regularization=1e-6,
        jerk_delta_regularization=1e-5,
        initial_jerk_delta_regularization=(
            candidate.initial_jerk_delta_regularization
        ),
        reference_mode=(
            "linear_hold"
            if candidate.reference_mode == "runtime_v2"
            else (
                "linear_hold_nullspace_smooth"
                if candidate.reference_mode
                in {
                    "runtime_v2_nullspace",
                    "runtime_vive_midcut_nullspace",
                    "runtime_vive_midcut_q_anchor",
                    "runtime_vive_midcut_q_qdd_anchor",
                    "runtime_vive_cut_q_anchor",
                    "runtime_vive_predictive_action",
                }
                else (
                    "causal_hermite_nullspace_smooth"
                    if candidate.reference_mode == "runtime_vive_nullspace"
                    else (
                        "causal_hermite_endpoint_selective_nullspace"
                        if candidate.reference_mode
                        == "runtime_vive_endpoint_selective_nullspace"
                        else (
                            "causal_hermite_endpoint_selective"
                        if candidate.reference_mode
                        == "runtime_vive_endpoint_selective"
                        else "causal_hermite_selective"
                        )
                    )
                )
            )
        ),
        observer_window=candidate.velocity_observer_window,
        observer_efficiency_power=candidate.observer_efficiency_power,
        observer_velocity_ema_alpha=candidate.velocity_ema_alpha,
        observer_velocity_cap_scale=candidate.velocity_cap_scale,
        observer_coherence_min=candidate.velocity_coherence_min,
        observer_speed_guard_scale=candidate.observer_speed_guard_scale,
        terminal_velocity_displacement_scale=(
            candidate.terminal_velocity_displacement_scale
        ),
        observer_stop_threshold_rad=math.radians(
            candidate.observer_stop_threshold_deg
        ),
        observer_stop_velocity_decay=candidate.observer_stop_velocity_decay,
    )
    planner.reset(qref[0])
    command_rows = [qref[0].copy()]
    tracking_reference_rows = [qref[0].copy()]
    velocity_rows = [np.zeros(6, dtype=float)]
    acceleration_rows = [np.zeros(6, dtype=float)]
    command_source_index = [0]
    formal_forecasts: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    selection_counts = {
        "hold": 0,
        "hermite": 0,
        "midcut": 0,
        "midcut_selected_joints": 0,
    }
    midcut_filtered_velocity = np.zeros(6, dtype=float)
    predictive_offset = np.zeros(6, dtype=float)

    def velocity_joint_valid_at(index: int) -> np.ndarray:
        if external_velocity is None:
            return np.zeros(6, dtype=bool)
        required_rows = max(3, int(candidate.velocity_observer_window))
        start = max(0, int(index) - required_rows + 1)
        window = np.asarray(external_velocity[start : index + 1], dtype=float)
        if window.shape[0] < required_rows:
            return np.zeros(6, dtype=bool)
        if external_velocity_valid is not None and not np.all(
            external_velocity_valid[start : index + 1]
        ):
            return np.zeros(6, dtype=bool)
        path = np.sum(np.abs(window), axis=0)
        efficiency = np.abs(np.sum(window, axis=0)) / np.maximum(path, 1e-12)
        current = window[-1]
        return (
            (efficiency >= candidate.velocity_coherence_min)
            & (np.abs(current) >= math.radians(0.05))
        )

    sequence = 0
    for source_index, requested in enumerate(qref[:-1]):
        planner_requested = requested
        if (
            candidate.reference_mode == "runtime_vive_predictive_action"
            and source_index % 4 == 0
        ):
            valid = bool(
                external_velocity_valid is None
                or external_velocity_valid[source_index]
            )
            if valid:
                raw_velocity = np.asarray(
                    external_velocity[source_index], dtype=float
                )
                midcut_filtered_velocity = (
                    (1.0 - candidate.velocity_ema_alpha)
                    * midcut_filtered_velocity
                    + candidate.velocity_ema_alpha * raw_velocity
                )
                velocity_cap = (
                    candidate.velocity_cap_scale * limits.max_velocity_rad_s
                )
                midcut_filtered_velocity = np.clip(
                    midcut_filtered_velocity,
                    -velocity_cap,
                    velocity_cap,
                )
                coherent = velocity_joint_valid_at(source_index)
                if source_index >= 4:
                    action_velocity = (
                        requested - qref[source_index - 4]
                    ) / (4.0 / _REFERENCE_HZ)
                    velocity_ratio = np.abs(midcut_filtered_velocity) / np.maximum(
                        np.abs(action_velocity),
                        math.radians(0.05),
                    )
                    coherent &= (
                        midcut_filtered_velocity * action_velocity > 0.0
                    ) & (
                        np.abs(action_velocity) >= math.radians(0.10)
                    ) & (
                        velocity_ratio >= 0.25
                    ) & (
                        velocity_ratio <= 4.0
                    )
                else:
                    coherent.fill(False)
                desired_prediction = (
                    float(candidate.velocity_prediction_sec)
                    * midcut_filtered_velocity
                )
                desired_prediction = np.clip(
                    desired_prediction,
                    -math.radians(0.25),
                    math.radians(0.25),
                )
                desired_prediction[~coherent] = 0.0
                predictive_offset += np.clip(
                    desired_prediction - predictive_offset,
                    -math.radians(0.02),
                    math.radians(0.02),
                )
                planner_requested = np.clip(
                    requested + predictive_offset,
                    limits.min_position_rad,
                    limits.max_position_rad,
                )
            else:
                midcut_filtered_velocity.fill(0.0)
                predictive_offset.fill(0.0)
        admission = planner.prepare_target(
            planner_requested,
            source_timestamp=source_index / _REFERENCE_HZ,
            speed_scale=1.0,
            mode="tracking",
            reason="offline_replay",
            requested_velocity_rad_s=(
                None
                if candidate.reference_mode
                not in {
                    "runtime_vive_selective",
                    "runtime_vive_nullspace",
                    "runtime_vive_endpoint_selective",
                    "runtime_vive_endpoint_selective_nullspace",
                }
                else external_velocity[source_index]
            ),
            requested_velocity_valid=(
                None
                if candidate.reference_mode
                not in {
                    "runtime_vive_selective",
                    "runtime_vive_nullspace",
                    "runtime_vive_endpoint_selective",
                    "runtime_vive_endpoint_selective_nullspace",
                }
                or external_velocity_valid is None
                else bool(external_velocity_valid[source_index])
            ),
        )
        if admission.new_source_bucket:
            sequence += 1
            planner.activate_target(admission, sequence=sequence)
            if (
                candidate.reference_mode == "runtime_vive_cut_q_anchor"
                and planner._active_plan is not None
            ):
                valid = bool(
                    external_velocity_valid is None
                    or external_velocity_valid[source_index]
                )
                if valid:
                    raw_velocity = np.asarray(
                        external_velocity[source_index], dtype=float
                    )
                    midcut_filtered_velocity = (
                        (1.0 - candidate.velocity_ema_alpha)
                        * midcut_filtered_velocity
                        + candidate.velocity_ema_alpha * raw_velocity
                    )
                    velocity_cap = (
                        candidate.velocity_cap_scale
                        * limits.max_velocity_rad_s
                    )
                    midcut_filtered_velocity = np.clip(
                        midcut_filtered_velocity,
                        -velocity_cap,
                        velocity_cap,
                    )
                    refreshed, refresh_metadata = (
                        _midcut_velocity_q_anchor_plan(
                            planner._active_plan,
                            target_velocity_rad_s=midcut_filtered_velocity,
                            limits=limits,
                            start_row=0,
                            previous_jerk_rad_s3=(
                                planner._previous_jerk_rad_s3
                            ),
                            velocity_joint_valid=velocity_joint_valid_at(
                                source_index
                            ),
                        )
                    )
                    planner._active_plan = refreshed
                    if int(
                        refresh_metadata.get(
                            "midcut_selected_joint_count", 0
                        )
                    ):
                        selection_counts["midcut"] += 1
                        selection_counts["midcut_selected_joints"] += int(
                            refresh_metadata["midcut_selected_joint_count"]
                        )
                    for rejection, count in dict(
                        refresh_metadata.get(
                            "midcut_rejection_counts", {}
                        )
                    ).items():
                        key = f"midcut_reject_{rejection}"
                        selection_counts[key] = selection_counts.get(
                            key, 0
                        ) + int(count)
                else:
                    midcut_filtered_velocity.fill(0.0)
            formal_forecasts.append(
                (
                    source_index,
                    len(command_rows) - 1,
                    admission.admitted_qpos.copy(),
                    requested.copy(),
                )
            )
            selected = (
                "hermite"
                if admission.metadata.get(
                    "sample_aligned_reference_selected"
                )
                in {"causal_hermite", "linear_hold_nullspace_smooth"}
                else "hold"
            )
            selection_counts[selected] += 1
        if (
            candidate.reference_mode
            in {
                "runtime_vive_midcut_nullspace",
                "runtime_vive_midcut_q_anchor",
                "runtime_vive_midcut_q_qdd_anchor",
            }
            and source_index % 4 == 2
            and planner._active_plan is not None
        ):
            valid = bool(
                external_velocity_valid is None
                or external_velocity_valid[source_index]
            )
            if valid:
                raw_velocity = np.asarray(
                    external_velocity[source_index], dtype=float
                )
                midcut_filtered_velocity = (
                    (1.0 - candidate.velocity_ema_alpha)
                    * midcut_filtered_velocity
                    + candidate.velocity_ema_alpha * raw_velocity
                )
                velocity_cap = (
                    candidate.velocity_cap_scale * limits.max_velocity_rad_s
                )
                midcut_filtered_velocity = np.clip(
                    midcut_filtered_velocity,
                    -velocity_cap,
                    velocity_cap,
                )
                if candidate.reference_mode in {
                    "runtime_vive_midcut_q_anchor",
                    "runtime_vive_midcut_q_qdd_anchor",
                }:
                    refreshed, refresh_metadata = (
                        _midcut_velocity_q_anchor_plan(
                            planner._active_plan,
                            target_velocity_rad_s=midcut_filtered_velocity,
                            limits=limits,
                            velocity_joint_valid=velocity_joint_valid_at(
                                source_index
                            ),
                            anchor_terminal_acceleration=(
                                candidate.reference_mode
                                == "runtime_vive_midcut_q_qdd_anchor"
                            ),
                            max_smoothness_ratio=(
                                1.0
                                if candidate.reference_mode
                                == "runtime_vive_midcut_q_qdd_anchor"
                                else 1.02
                            ),
                        )
                    )
                else:
                    refreshed, refresh_metadata = (
                        _midcut_velocity_nullspace_plan(
                            planner._active_plan,
                            target_velocity_rad_s=midcut_filtered_velocity,
                            limits=limits,
                        )
                    )
                planner._active_plan = refreshed
                if int(refresh_metadata.get("midcut_selected_joint_count", 0)):
                    selection_counts["midcut"] += 1
                    selection_counts["midcut_selected_joints"] += int(
                        refresh_metadata["midcut_selected_joint_count"]
                    )
                for rejection, count in dict(
                    refresh_metadata.get("midcut_rejection_counts", {})
                ).items():
                    key = f"midcut_reject_{rejection}"
                    selection_counts[key] = selection_counts.get(key, 0) + int(
                        count
                    )
            else:
                midcut_filtered_velocity.fill(0.0)
        for substep in range(2):
            proposal = planner.propose(
                monotonic_time=(2 * source_index + substep) / _COMMAND_HZ
            )
            command_rows.append(np.asarray(proposal.q_command, dtype=float).copy())
            tracking_reference_rows.append(requested.copy())
            velocity_rows.append(
                np.asarray(proposal.qd_command_rad_s, dtype=float).copy()
            )
            acceleration_rows.append(
                np.asarray(proposal.qdd_command_rad_s2, dtype=float).copy()
            )
            command_source_index.append(source_index)
            planner.commit(proposal)

    command = np.asarray(command_rows, dtype=float)
    action_error = []
    action_state_error = []
    requested_next_state_error = []
    for _source_index, command_start, forecast, requested in formal_forecasts:
        next_index = command_start + int(_COMMAND_HZ / _FORMAL_ACTION_HZ)
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
        "tracking_reference_rad": np.asarray(
            tracking_reference_rows,
            dtype=float,
        ),
        "command_source_index": np.asarray(command_source_index, dtype=np.int64),
        "action_error_rad": np.asarray(action_error, dtype=float),
        "action_state_error_rad": np.asarray(action_state_error, dtype=float),
        "requested_next_state_error_rad": np.asarray(
            requested_next_state_error, dtype=float
        ),
        "max_jerk_rad_s3": np.asarray(limits.max_jerk_rad_s3, dtype=float),
        "selection_counts": selection_counts,
    }


def _simulate_runtime_60hz_v2_responsive(qref: np.ndarray) -> dict[str, Any]:
    """Replay the former production 60 Hz V2 qref-to-qcmd chain exactly."""

    if qref.shape[0] < 2:
        raise ValueError("runtime 60 Hz V2 replay requires at least two qref rows.")
    execution = RM65_BIMANUAL_POSITION_V2_RESPONSIVE.command_execution
    planner = AccelerationLimitedJointTrajectoryPlanner(
        JointTrajectoryLimits(
            max_velocity_rad_s=np.asarray(
                execution.max_velocity_rad_s,
                dtype=float,
            ),
            max_acceleration_rad_s2=np.asarray(
                execution.max_acceleration_rad_s2,
                dtype=float,
            ),
            max_command_step_rad=np.full(
                6,
                math.radians(execution.max_command_step_deg),
                dtype=float,
            ),
        ),
        velocity_ramp_sec=execution.velocity_ramp_sec,
        target_velocity_filter_tau_sec=(
            execution.target_velocity_filter_tau_sec
        ),
        reference_period_sec=1.0 / _REFERENCE_HZ,
    )
    planner.reset(qref[0], monotonic_time=0.0)
    command_rows = [qref[0].copy()]
    tracking_reference_rows = [qref[0].copy()]
    velocity_rows = [np.zeros(6, dtype=float)]
    acceleration_rows = [np.zeros(6, dtype=float)]
    command_source_index = [0]
    formal_forecasts: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for source_index, requested in enumerate(qref[:-1]):
        source_time = source_index / _REFERENCE_HZ
        planner.accept_target(
            requested,
            sequence=source_index + 1,
            published_monotonic_time=source_time,
            monotonic_time=source_time,
            speed_scale=1.0,
            mode="tracking",
            reason="offline_replay",
        )
        if source_index % int(_REFERENCE_HZ / _FORMAL_ACTION_HZ) == 0:
            formal_forecasts.append(
                (
                    source_index,
                    len(command_rows) - 1,
                    requested.copy(),
                    requested.copy(),
                )
            )
        for substep in range(2):
            sample_time = (2 * source_index + substep) / _COMMAND_HZ
            proposal = planner.propose(monotonic_time=sample_time)
            accepted, _smoothed = final_follow_qpos_target(
                proposal.q_command,
                command_rows[-1],
                alpha=execution.follow_qpos_alpha,
                bypass=False,
            )
            planner.commit(
                proposal,
                accepted_monotonic_time=sample_time,
                accepted_q_command=accepted,
                allow_bounded_accepted_adjustment=True,
            )
            command_rows.append(np.asarray(accepted, dtype=float).copy())
            snapshot = planner.snapshot()
            velocity_rows.append(
                np.asarray(snapshot["qd_command_rad_s"], dtype=float)
            )
            acceleration_rows.append(
                np.asarray(snapshot["qdd_command_rad_s2"], dtype=float)
            )
            tracking_reference_rows.append(requested.copy())
            command_source_index.append(source_index)

    command = np.asarray(command_rows, dtype=float)
    action_error = []
    action_state_error = []
    requested_next_state_error = []
    for _source_index, command_start, forecast, requested in formal_forecasts:
        next_index = command_start + int(_COMMAND_HZ / _FORMAL_ACTION_HZ)
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
        "tracking_reference_rad": np.asarray(
            tracking_reference_rows,
            dtype=float,
        ),
        "command_source_index": np.asarray(command_source_index, dtype=np.int64),
        "action_error_rad": np.asarray(action_error, dtype=float),
        "action_state_error_rad": np.asarray(action_state_error, dtype=float),
        "requested_next_state_error_rad": np.asarray(
            requested_next_state_error,
            dtype=float,
        ),
        # The historical acceleration-limited planner has no jerk hard bound.
        "max_jerk_rad_s3": np.full(6, np.inf, dtype=float),
        "selection_counts": {"hold": 0, "hermite": 0},
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def _degrees_stats(matrix_rad: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix_rad, dtype=float)
    if matrix.size == 0:
        return _percentiles(np.empty(0))
    return _percentiles(np.max(np.abs(np.rad2deg(matrix)), axis=1))


def _evaluate_simulation(result: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray(result["q_rad"], dtype=float)
    qd = np.asarray(result["qd_rad_s"], dtype=float)
    qdd = np.asarray(result["qdd_rad_s2"], dtype=float)
    jerk = np.diff(qdd, axis=0) * _COMMAND_HZ
    step = np.diff(q, axis=0)
    jerk_limit = np.asarray(result["max_jerk_rad_s3"], dtype=float)
    saturation_90 = np.abs(jerk) >= 0.90 * jerk_limit
    saturation_98 = np.abs(jerk) >= 0.98 * jerk_limit
    source_index = np.asarray(result["command_source_index"], dtype=np.int64)
    cut = np.r_[True, source_index[1:] != source_index[:-1]]
    jerk_command_index = np.arange(1, qdd.shape[0])
    near_cut = np.array(
        [np.any(cut[max(0, index - 1) : index + 1]) for index in jerk_command_index],
        dtype=bool,
    )
    jerk_norm = np.max(np.abs(np.rad2deg(jerk)), axis=1)
    reversal_counts = []
    adjacent_reversal_amplitudes_deg: list[float] = []
    adjacent_reversal_counts = {"0.01": 0, "0.05": 0, "0.10": 0}
    for joint in range(6):
        delta_deg = np.rad2deg(step[:, joint])
        active = np.flatnonzero(np.abs(delta_deg) > 0.002)
        reversal_counts.append(
            int(
                np.sum(
                    np.sign(delta_deg[active][1:])
                    * np.sign(delta_deg[active][:-1])
                    < 0.0
                )
            )
            if active.size > 1
            else 0
        )
        if delta_deg.size > 1:
            reversal_mask = delta_deg[1:] * delta_deg[:-1] < 0.0
            reversal_amplitude = np.minimum(
                np.abs(delta_deg[1:]),
                np.abs(delta_deg[:-1]),
            )[reversal_mask]
            adjacent_reversal_amplitudes_deg.extend(reversal_amplitude.tolist())
            for threshold in (0.01, 0.05, 0.10):
                adjacent_reversal_counts[f"{threshold:.2f}"] += int(
                    np.sum(reversal_amplitude >= threshold)
                )
    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    max_velocity = np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S, dtype=float)
    max_acceleration = np.asarray(
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float
    )
    tracking_reference = np.asarray(
        result.get("tracking_reference_rad", q),
        dtype=float,
    )
    if tracking_reference.shape != q.shape:
        raise ValueError("tracking_reference_rad must match the qcmd path shape.")
    tracking_error = q - tracking_reference
    best_lag_ticks = 0
    best_lag_rms_rad = math.inf
    # Ignore startup and score only changing-reference samples.  This avoids a
    # long stationary tail making every lag look equally good.
    reference_step = np.linalg.norm(np.diff(tracking_reference, axis=0), axis=1)
    moving = np.r_[False, reference_step > math.radians(0.002)]
    for lag_ticks in range(25):
        if lag_ticks == 0:
            command_view = q
            reference_view = tracking_reference
            moving_view = moving
        else:
            command_view = q[lag_ticks:]
            reference_view = tracking_reference[:-lag_ticks]
            moving_view = moving[:-lag_ticks]
        moving_view = moving_view.copy()
        moving_view[: min(16, moving_view.size)] = False
        if not np.any(moving_view):
            continue
        residual = command_view[moving_view] - reference_view[moving_view]
        rms = float(np.sqrt(np.mean(np.square(residual))))
        if rms < best_lag_rms_rad:
            best_lag_rms_rad = rms
            best_lag_ticks = lag_ticks
    return {
        "command_rows": int(q.shape[0]),
        "step_deg": _degrees_stats(step),
        "velocity_deg_s": _degrees_stats(qd),
        "acceleration_deg_s2": _degrees_stats(qdd),
        "jerk_deg_s3": _degrees_stats(jerk),
        "jerk_near_cut_deg_s3": _percentiles(jerk_norm[near_cut]),
        "jerk_inside_cut_deg_s3": _percentiles(jerk_norm[~near_cut]),
        "jerk_saturation_90_count_by_joint": np.sum(
            saturation_90, axis=0
        ).astype(int).tolist(),
        "jerk_saturation_98_count_by_joint": np.sum(
            saturation_98, axis=0
        ).astype(int).tolist(),
        "meaningful_reversal_count_by_joint": reversal_counts,
        "adjacent_reversal_amplitude_deg": _percentiles(
            np.asarray(adjacent_reversal_amplitudes_deg, dtype=float)
        ),
        "adjacent_reversal_count_by_threshold_deg": adjacent_reversal_counts,
        "selection_counts": dict(result["selection_counts"]),
        "action_requested_error_deg": _degrees_stats(result["action_error_rad"]),
        "action_next_state_error_deg": _degrees_stats(
            result["action_state_error_rad"]
        ),
        "requested_next_state_error_deg": _degrees_stats(
            result["requested_next_state_error_rad"]
        ),
        "instantaneous_qref_tracking_error_deg": _degrees_stats(tracking_error),
        "best_tracking_lag_ms": 1000.0 * best_lag_ticks / _COMMAND_HZ,
        "best_tracking_lag_rms_deg": (
            0.0
            if not math.isfinite(best_lag_rms_rad)
            else math.degrees(best_lag_rms_rad)
        ),
        "position_violation_count": int(
            np.sum(np.any((q < lower - 1e-8) | (q > upper + 1e-8), axis=1))
        ),
        "velocity_violation_count": int(
            np.sum(np.any(np.abs(qd) > max_velocity + 1e-8, axis=1))
        ),
        "acceleration_violation_count": int(
            np.sum(np.any(np.abs(qdd) > max_acceleration + 1e-8, axis=1))
        ),
        "jerk_violation_count": int(
            np.sum(np.any(np.abs(jerk) > jerk_limit + 1e-6, axis=1))
        ),
    }


def _evaluate_recorded(
    qcmd_rad: np.ndarray,
    target_sequence: np.ndarray,
) -> dict[str, Any]:
    q = _finite_matrix(qcmd_rad, columns=6)
    step = np.diff(q, axis=0)
    qd = step * _COMMAND_HZ
    qdd = np.diff(qd, axis=0) * _COMMAND_HZ
    jerk = np.diff(qdd, axis=0) * _COMMAND_HZ
    cut = np.r_[True, target_sequence[1:] != target_sequence[:-1]]
    jerk_qindex = np.arange(3, q.shape[0])
    near_cut = np.array(
        [np.any(cut[max(0, index - 1) : index + 1]) for index in jerk_qindex],
        dtype=bool,
    )
    jerk_norm = np.max(np.abs(np.rad2deg(jerk)), axis=1)
    return {
        "command_rows": int(q.shape[0]),
        "step_deg": _degrees_stats(step),
        "velocity_deg_s": _degrees_stats(qd),
        "acceleration_deg_s2": _degrees_stats(qdd),
        "jerk_deg_s3": _degrees_stats(jerk),
        "jerk_near_cut_deg_s3": _percentiles(jerk_norm[near_cut]),
        "jerk_inside_cut_deg_s3": _percentiles(jerk_norm[~near_cut]),
    }


def _aggregate(
    rows: Iterable[dict[str, Any]],
    *,
    candidate: Candidate,
    failures: Sequence[dict[str, str]],
) -> dict[str, Any]:
    materialized = list(rows)
    scalar_paths = (
        ("step_deg", "max"),
        ("velocity_deg_s", "max"),
        ("acceleration_deg_s2", "max"),
        ("jerk_deg_s3", "max"),
        ("jerk_deg_s3", "p95"),
        ("jerk_near_cut_deg_s3", "p95"),
        ("jerk_inside_cut_deg_s3", "p95"),
        ("action_requested_error_deg", "p95"),
        ("action_requested_error_deg", "max"),
        ("action_next_state_error_deg", "p95"),
        ("action_next_state_error_deg", "max"),
        ("requested_next_state_error_deg", "p95"),
        ("requested_next_state_error_deg", "max"),
        ("instantaneous_qref_tracking_error_deg", "p95"),
        ("instantaneous_qref_tracking_error_deg", "max"),
        ("adjacent_reversal_amplitude_deg", "p95"),
        ("adjacent_reversal_amplitude_deg", "max"),
    )
    summary: dict[str, Any] = {
        "candidate": candidate.name,
        "replan_hz": candidate.replan_hz,
        "reference_mode": candidate.reference_mode,
        "jerk_ramp_sec": candidate.jerk_ramp_sec,
        "initial_jerk_delta_regularization": (
            candidate.initial_jerk_delta_regularization
        ),
        "velocity_observer": candidate.velocity_observer,
        "velocity_observer_window": candidate.velocity_observer_window,
        "observer_efficiency_power": candidate.observer_efficiency_power,
        "velocity_coherence_min": candidate.velocity_coherence_min,
        "terminal_velocity_displacement_scale": (
            candidate.terminal_velocity_displacement_scale
        ),
        "trace_side_count": len(materialized),
        "failure_count": len(failures),
        "failures": list(failures),
    }
    for group, key in scalar_paths:
        values = [float(row[group][key]) for row in materialized]
        summary[f"{group}_{key}_worst"] = max(values, default=0.0)
        summary[f"{group}_{key}_mean"] = float(np.mean(values)) if values else 0.0
    for name in (
        "position_violation_count",
        "velocity_violation_count",
        "acceleration_violation_count",
        "jerk_violation_count",
    ):
        summary[name] = int(sum(int(row[name]) for row in materialized))
    lag_values = [float(row["best_tracking_lag_ms"]) for row in materialized]
    lag_rms_values = [
        float(row["best_tracking_lag_rms_deg"]) for row in materialized
    ]
    summary["best_tracking_lag_ms_worst"] = max(lag_values, default=0.0)
    summary["best_tracking_lag_ms_mean"] = (
        float(np.mean(lag_values)) if lag_values else 0.0
    )
    summary["best_tracking_lag_rms_deg_worst"] = max(
        lag_rms_values,
        default=0.0,
    )
    summary["jerk_saturation_98_total"] = int(
        sum(sum(row["jerk_saturation_98_count_by_joint"]) for row in materialized)
    )
    summary["meaningful_reversal_total"] = int(
        sum(sum(row["meaningful_reversal_count_by_joint"]) for row in materialized)
    )
    summary["adjacent_reversal_count_by_threshold_deg"] = {
        threshold: int(
            sum(
                row["adjacent_reversal_count_by_threshold_deg"][threshold]
                for row in materialized
            )
        )
        for threshold in ("0.01", "0.05", "0.10")
    }
    summary["selection_counts"] = {
        mode: int(
            sum(int(row["selection_counts"].get(mode, 0)) for row in materialized)
        )
        for mode in ("hold", "hermite")
    }
    return summary


def _print_ranking(summaries: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(
        summaries,
        key=lambda row: (
            row["failure_count"],
            row["position_violation_count"],
            row["velocity_violation_count"],
            row["acceleration_violation_count"],
            row["jerk_violation_count"],
            row["action_next_state_error_deg_p95_worst"],
            row["jerk_near_cut_deg_s3_p95_worst"],
            row["requested_next_state_error_deg_p95_worst"],
        ),
    )
    print(
        "candidate rate mode ramp failures action_state_p95 action_req_p95 "
        "next_req_p95 qref_track_p95 lag_ms jerk_p95 cut_jerk_p95 "
        "sat98 rev_gt_005 reversals"
    )
    for row in ordered:
        print(
            row["candidate"],
            row["replan_hz"],
            row["reference_mode"],
            f'{row["jerk_ramp_sec"]:.3f}',
            row["failure_count"],
            f'{row["action_next_state_error_deg_p95_worst"]:.4f}',
            f'{row["action_requested_error_deg_p95_worst"]:.4f}',
            f'{row["requested_next_state_error_deg_p95_worst"]:.4f}',
            f'{row["instantaneous_qref_tracking_error_deg_p95_worst"]:.4f}',
            f'{row["best_tracking_lag_ms_mean"]:.1f}',
            f'{row["jerk_deg_s3_p95_worst"]:.1f}',
            f'{row["jerk_near_cut_deg_s3_p95_worst"]:.1f}',
            row["jerk_saturation_98_total"],
            row["adjacent_reversal_count_by_threshold_deg"]["0.05"],
            row["meaningful_reversal_total"],
        )


def _high_motion_window(trace: TraceInput, rows: int) -> TraceInput:
    available = min(
        trace.qref_by_side["left"].shape[0],
        trace.qref_by_side["right"].shape[0],
    )
    if rows >= available:
        return trace
    if rows < 5:
        raise ValueError("window_rows must be at least 5.")
    combined = np.hstack(
        (
            trace.qref_by_side["left"][:available],
            trace.qref_by_side["right"][:available],
        )
    )
    step_energy = np.sum(np.square(np.diff(combined, axis=0)), axis=1)
    width = rows - 1
    rolling = np.convolve(
        step_energy,
        np.ones(width, dtype=float),
        mode="valid",
    )
    start = int(np.argmax(rolling))
    stop = start + rows
    return TraceInput(
        name=f"{trace.name}:high_motion[{start}:{stop}]",
        date=trace.date,
        path=trace.path,
        qref_by_side={
            side: value[start:stop].copy()
            for side, value in trace.qref_by_side.items()
        },
        recorded_qcmd_by_side=trace.recorded_qcmd_by_side,
        recorded_target_sequence_by_side=(
            trace.recorded_target_sequence_by_side
        ),
    )


def _validate_collision_pair(
    result_by_side: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from scripts.probes.rm65_qref_dynamics_probe import (  # noqa: PLC0415
        _build_collision_planner,
    )
    from scripts.probes.rm65_official_all_ik_probe import (  # noqa: PLC0415
        _LEFT_TOOL,
        _RIGHT_TOOL,
    )

    command = {
        side: _finite_matrix(result_by_side[side]["q_rad"], columns=6)
        for side in _SIDES
    }
    count = min(value.shape[0] for value in command.values())
    planner = _build_collision_planner(
        {"left": _LEFT_TOOL, "right": _RIGHT_TOOL}
    )
    planner.reset({side: value[0] for side, value in command.items()})
    endpoint_collision_frames = 0
    sampled_segment_collision_intervals = 0
    minimum_distance = math.inf
    for index in range(count):
        pair = {side: command[side][index] for side in _SIDES}
        validation = planner.validate_qpos_reference(
            pair,
            source="offline_teleop_smoothing_probe",
        )
        endpoint_collision_frames += int(not validation.safe)
        if validation.collision_distance_m is not None:
            minimum_distance = min(
                minimum_distance,
                float(validation.collision_distance_m),
            )
        if index == 0:
            continue
        sampled_unsafe = False
        for alpha in (0.25, 0.5, 0.75):
            sample = {
                side: (
                    (1.0 - alpha) * command[side][index - 1]
                    + alpha * command[side][index]
                )
                for side in _SIDES
            }
            sampled = planner.validate_qpos_reference(
                sample,
                source="offline_teleop_smoothing_probe_segment",
            )
            if sampled.collision_distance_m is not None:
                minimum_distance = min(
                    minimum_distance,
                    float(sampled.collision_distance_m),
                )
            if not sampled.safe:
                sampled_unsafe = True
                break
        sampled_segment_collision_intervals += int(sampled_unsafe)
    return {
        "checked_frames": count,
        "endpoint_collision_frames": endpoint_collision_frames,
        "checked_intervals": max(0, count - 1),
        "sampled_segment_collision_intervals": (
            sampled_segment_collision_intervals
        ),
        "minimum_sampled_collision_distance_m": (
            None if not math.isfinite(minimum_distance) else minimum_distance
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace",
        nargs="+",
        type=Path,
        help="Actuator trace directory or command_trace.npz path.",
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Run only the named candidate; repeat to select more than one.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--window-rows",
        type=int,
        help=(
            "Evaluate the highest-motion window with this many 60 Hz rows "
            "from each trace; omit to evaluate full traces."
        ),
    )
    parser.add_argument(
        "--validate-collision",
        action="store_true",
        help="Validate paired dual-arm endpoints and 0.25/0.5/0.75 segments.",
    )
    args = parser.parse_args()

    traces = tuple(_load_trace(path) for path in args.trace)
    if args.window_rows is not None:
        traces = tuple(
            _high_motion_window(trace, args.window_rows) for trace in traces
        )
    candidates = _build_candidates(args.preset)
    if args.candidate:
        requested = set(args.candidate)
        candidates = tuple(candidate for candidate in candidates if candidate.name in requested)
        missing = requested - {candidate.name for candidate in candidates}
        if missing:
            raise SystemExit(f"unknown candidate(s): {sorted(missing)}")
    detail: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []
    for trace in traces:
        for side in _SIDES:
            recorded.append(
                {
                    "trace": trace.name,
                    "date": trace.date,
                    "side": side,
                    "metrics": _evaluate_recorded(
                        trace.recorded_qcmd_by_side[side],
                        trace.recorded_target_sequence_by_side[side],
                    ),
                }
            )
    summaries = []
    paired_validation: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_rows = []
        failures: list[dict[str, str]] = []
        for trace in traces:
            result_by_side: dict[str, dict[str, Any]] = {}
            for side in _SIDES:
                try:
                    result = _simulate(trace.qref_by_side[side], candidate)
                except Exception as exc:
                    failure = {
                        "trace": trace.name,
                        "date": trace.date,
                        "side": side,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    detail.append(
                        {
                            "candidate": candidate.name,
                            "trace": trace.name,
                            "date": trace.date,
                            "side": side,
                            "failure": failure["error"],
                        }
                    )
                    continue
                metrics = _evaluate_simulation(result)
                result_by_side[side] = result
                candidate_rows.append(metrics)
                detail.append(
                    {
                        "candidate": candidate.name,
                        "trace": trace.name,
                        "date": trace.date,
                        "side": side,
                        "metrics": metrics,
                    }
                )
            if args.validate_collision and set(result_by_side) == set(_SIDES):
                try:
                    validation = _validate_collision_pair(result_by_side)
                except Exception as exc:
                    validation = {
                        "validation_error": f"{type(exc).__name__}: {exc}"
                    }
                paired_validation.append(
                    {
                        "candidate": candidate.name,
                        "trace": trace.name,
                        "date": trace.date,
                        **validation,
                    }
                )
        summaries.append(
            _aggregate(
                candidate_rows,
                candidate=candidate,
                failures=failures,
            )
        )
    _print_ranking(summaries)
    payload = {
        "schema": "todos.rm65_teleop_smoothing_probe/v1",
        "trace_paths": [str(trace.path) for trace in traces],
        "recorded": recorded,
        "summaries": summaries,
        "detail": detail,
        "paired_collision_validation": paired_validation,
        "notes": {
            "device_access": False,
            "dispatch": False,
            "common_input": "ik_shadow.admitted_q_rad at 60 Hz",
            "formal_action_hz": _FORMAL_ACTION_HZ,
            "command_hz": _COMMAND_HZ,
            "high_rate_action_warning": (
                "30/60 Hz plans may preempt the 15 Hz action forecast; inspect "
                "action_next_state_error before treating them as collection-safe."
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
