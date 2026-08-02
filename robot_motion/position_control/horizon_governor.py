"""Backend-neutral finite-horizon qref-to-qcmd projection.

The governor owns no clock, authority, dispatch, or committed state.  A caller
supplies the last admitted joint position, velocity and acceleration together
with a finite qref horizon.  The returned control cut is only a candidate; the
canonical collision/admission layer remains responsible for accepting it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import osqp
from scipy import sparse


_POSITION_NUMERIC_TOLERANCE_RAD = 1e-7
# Keep future QP states a microscopic distance inside absolute joint bounds.
# OSQP's normalized primal tolerance is otherwise large enough to return a
# mathematically "solved inaccurate" point a few microradians outside a hard
# bound when the reference itself lies exactly on that bound.
_POSITION_CONSTRAINT_INTERIOR_RAD = 1e-6
_POSITION_CONSTRAINT_MIN_INTERIOR_RAD = 1e-7
_SOLVER_CERTIFICATE_TOLERANCE_RAD = 1e-6


def _vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1).copy()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return result


def _positive_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = _vector(value, name=name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return result


@dataclass(frozen=True)
class JointHorizonLimits:
    min_position_rad: Sequence[float] | np.ndarray
    max_position_rad: Sequence[float] | np.ndarray
    max_velocity_rad_s: Sequence[float] | np.ndarray
    max_acceleration_rad_s2: Sequence[float] | np.ndarray
    max_jerk_rad_s3: Sequence[float] | np.ndarray

    def __post_init__(self) -> None:
        lower = _vector(self.min_position_rad, name="min_position_rad")
        upper = _vector(self.max_position_rad, name="max_position_rad")
        velocity = _positive_vector(
            self.max_velocity_rad_s,
            name="max_velocity_rad_s",
        )
        acceleration = _positive_vector(
            self.max_acceleration_rad_s2,
            name="max_acceleration_rad_s2",
        )
        jerk = _positive_vector(self.max_jerk_rad_s3, name="max_jerk_rad_s3")
        shape = lower.shape
        if any(
            value.shape != shape
            for value in (upper, velocity, acceleration, jerk)
        ):
            raise ValueError("all horizon limit vectors must have matching shapes.")
        if np.any(lower >= upper):
            raise ValueError("min_position_rad must be below max_position_rad.")
        for name, value in (
            ("min_position_rad", lower),
            ("max_position_rad", upper),
            ("max_velocity_rad_s", velocity),
            ("max_acceleration_rad_s2", acceleration),
            ("max_jerk_rad_s3", jerk),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.min_position_rad).size)


@dataclass(frozen=True)
class JointHorizonState:
    q_rad: Sequence[float] | np.ndarray
    qd_rad_s: Sequence[float] | np.ndarray
    qdd_rad_s2: Sequence[float] | np.ndarray

    def __post_init__(self) -> None:
        values = {
            name: _vector(getattr(self, name), name=name)
            for name in ("q_rad", "qd_rad_s", "qdd_rad_s2")
        }
        if len({value.shape for value in values.values()}) != 1:
            raise ValueError("joint horizon state vectors must have matching shapes.")
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class JointHorizonPlan:
    q_path_rad: np.ndarray
    qd_path_rad_s: np.ndarray
    qdd_path_rad_s2: np.ndarray
    commit_count: int
    dt_sec: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        paths = {}
        for name in ("q_path_rad", "qd_path_rad_s", "qdd_path_rad_s2"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.ndim != 2 or value.shape[0] < 2 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite horizon matrix.")
            value.setflags(write=False)
            paths[name] = value
        if len({value.shape for value in paths.values()}) != 1:
            raise ValueError("joint horizon plan matrices must have matching shapes.")
        count = int(self.commit_count)
        if count < 1 or count >= paths["q_path_rad"].shape[0]:
            raise ValueError("commit_count must select at least one horizon transition.")
        dt = float(self.dt_sec)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be finite and positive.")
        for name, value in paths.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "commit_count", count)
        object.__setattr__(self, "dt_sec", dt)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def q_control_cut_rad(self) -> np.ndarray:
        return self.q_path_rad[1 : self.commit_count + 1]


def plan_jerk_limited_joint_horizon(
    reference_qpos: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_state: JointHorizonState,
    limits: JointHorizonLimits,
    dt_sec: float,
    commit_count: int,
    bound_guard: float = 0.999,
    solver_tolerance: float = 1e-7,
) -> JointHorizonPlan:
    """Project one qref horizon under q/qdot/qddot/jerk box constraints."""

    if not isinstance(initial_state, JointHorizonState):
        raise TypeError("initial_state must be JointHorizonState.")
    if not isinstance(limits, JointHorizonLimits):
        raise TypeError("limits must be JointHorizonLimits.")
    dt = float(dt_sec)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_sec must be finite and positive.")
    guard = float(bound_guard)
    if not math.isfinite(guard) or not 0.0 < guard <= 1.0:
        raise ValueError("bound_guard must be finite and in (0, 1].")
    solver_eps = float(solver_tolerance)
    if not math.isfinite(solver_eps) or not 0.0 < solver_eps <= 1e-4:
        raise ValueError("solver_tolerance must be finite and in (0, 1e-4].")
    reference = np.asarray(reference_qpos, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[0] < 2
        or reference.shape[1] != limits.dimension
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "reference_qpos must be a finite NxD matrix with N >= 2 and "
            "D matching limits."
        )
    if any(
        np.asarray(value).shape != (limits.dimension,)
        for value in (
            initial_state.q_rad,
            initial_state.qd_rad_s,
            initial_state.qdd_rad_s2,
        )
    ):
        raise ValueError("initial_state dimension does not match limits.")
    transitions = reference.shape[0] - 1
    committed = int(commit_count)
    if isinstance(commit_count, bool) or committed != commit_count:
        raise ValueError("commit_count must be an integer.")
    if committed < 1 or committed > transitions:
        raise ValueError("commit_count must be in [1, horizon transitions].")
    below = limits.min_position_rad - initial_state.q_rad
    above = initial_state.q_rad - limits.max_position_rad
    if np.any(below > _POSITION_NUMERIC_TOLERANCE_RAD) or np.any(
        above > _POSITION_NUMERIC_TOLERANCE_RAD
    ):
        raise ValueError("initial position is outside the configured joint limits.")
    initial_q = np.clip(
        initial_state.q_rad,
        limits.min_position_rad,
        limits.max_position_rad,
    )

    count = reference.shape[0]
    q_path = np.empty_like(reference)
    qd_path = np.empty_like(reference)
    qdd_path = np.empty_like(reference)
    receipts: list[dict[str, Any]] = []
    for joint in range(limits.dimension):
        # Optimize normalized jerk only.  Future q/v/a are exact affine
        # functions of that jerk sequence and the last admitted state.  This
        # avoids both the poorly conditioned third-difference q-only form and
        # the redundant-state equality residuals of a q/v/a/jerk form.
        horizon = transitions
        q_scale = max(
            1.0,
            float(limits.max_position_rad[joint] - limits.min_position_rad[joint]),
        )
        velocity_scale = float(limits.max_velocity_rad_s[joint])
        acceleration_scale = float(limits.max_acceleration_rad_s2[joint])
        jerk_scale = float(limits.max_jerk_rad_s3[joint])
        q_velocity_ratio = dt * velocity_scale / q_scale
        velocity_acceleration_ratio = dt * acceleration_scale / velocity_scale
        acceleration_jerk_ratio = dt * jerk_scale / acceleration_scale
        initial_q_normalized = float(initial_q[joint] / q_scale)
        initial_velocity_normalized = float(
            initial_state.qd_rad_s[joint] / velocity_scale
        )
        initial_acceleration_normalized = float(
            initial_state.qdd_rad_s2[joint] / acceleration_scale
        )
        integral = np.tril(np.ones((horizon, horizon), dtype=float))
        acceleration_response = acceleration_jerk_ratio * integral
        acceleration_bias = np.full(
            horizon,
            initial_acceleration_normalized,
            dtype=float,
        )
        velocity_response = (
            velocity_acceleration_ratio
            * integral
            @ acceleration_response
        )
        velocity_bias = (
            np.full(horizon, initial_velocity_normalized, dtype=float)
            + velocity_acceleration_ratio
            * integral
            @ acceleration_bias
        )
        position_response = q_velocity_ratio * integral @ velocity_response
        position_bias = (
            np.full(horizon, initial_q_normalized, dtype=float)
            + q_velocity_ratio * integral @ velocity_bias
        )
        # Conservative jerk-braking viability envelope.  For acceleration
        # magnitude A and jerk limit J, A/(2J) * |a| upper-bounds the velocity
        # consumed while bringing acceleration back to zero.  Applying the
        # signed form keeps accelerating states away from the velocity bound
        # without penalizing acceleration that is already braking.
        braking_time_sec = acceleration_scale / (2.0 * jerk_scale)
        braking_acceleration_ratio = (
            braking_time_sec * acceleration_scale / velocity_scale
        )
        viability_response = (
            velocity_response
            + braking_acceleration_ratio * acceleration_response
        )
        viability_bias = (
            velocity_bias
            + braking_acceleration_ratio * acceleration_bias
        )

        position_response_rad = q_scale * position_response
        position_error_bias_rad = (
            q_scale * position_bias - reference[1:, joint]
        )
        lower_position_interior = min(
            _POSITION_CONSTRAINT_INTERIOR_RAD,
            max(
                _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                float(initial_q[joint] - limits.min_position_rad[joint])
                * 0.5,
            ),
        )
        upper_position_interior = min(
            _POSITION_CONSTRAINT_INTERIOR_RAD,
            max(
                _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                float(limits.max_position_rad[joint] - initial_q[joint])
                * 0.5,
            ),
        )
        jerk_regularization = 1e-9
        objective = sparse.csc_matrix(
            2.0
            * (
                position_response_rad.T @ position_response_rad
                + jerk_regularization * np.eye(horizon)
            )
        )
        linear = 2.0 * (
            position_response_rad.T @ position_error_bias_rad
        )

        constraint = sparse.vstack(
            (
                sparse.eye(horizon, format="csc"),
                sparse.csc_matrix(acceleration_response),
                sparse.csc_matrix(velocity_response),
                sparse.csc_matrix(viability_response),
                sparse.csc_matrix(position_response_rad),
            ),
            format="csc",
        )
        lower = np.concatenate(
            (
                np.full(horizon, -guard),
                np.full(horizon, -guard) - acceleration_bias,
                np.full(horizon, -guard) - velocity_bias,
                np.full(horizon, -guard) - viability_bias,
                np.full(
                    horizon,
                    limits.min_position_rad[joint]
                    + lower_position_interior,
                )
                - q_scale * position_bias,
            )
        )
        upper = np.concatenate(
            (
                np.full(horizon, guard),
                np.full(horizon, guard) - acceleration_bias,
                np.full(horizon, guard) - velocity_bias,
                np.full(horizon, guard) - viability_bias,
                np.full(
                    horizon,
                    limits.max_position_rad[joint]
                    - upper_position_interior,
                )
                - q_scale * position_bias,
            )
        )
        solver = osqp.OSQP()
        solver.setup(
            P=objective,
            q=linear,
            A=constraint,
            l=lower,
            u=upper,
            verbose=False,
            polishing=False,
            eps_abs=solver_eps,
            eps_rel=solver_eps,
            max_iter=100_000,
            adaptive_rho=True,
            adaptive_rho_interval=25,
        )
        result = solver.solve()
        if result.x is None:
            raise RuntimeError(
                f"joint horizon QP failed for joint {joint + 1}: "
                f"{result.info.status}; "
                f"primal_residual={result.info.prim_res}; "
                f"dual_residual={result.info.dual_res}."
            )
        status_acceptable = result.info.status_val in {1, 2, 7}
        if not status_acceptable:
            raise RuntimeError(
                f"joint horizon QP failed for joint {joint + 1}: "
                f"{result.info.status}; "
                f"primal_residual={result.info.prim_res}; "
                f"dual_residual={result.info.dual_res}."
            )
        solution = np.asarray(result.x, dtype=float)
        joint_q = np.empty(count)
        joint_qd = np.empty(count)
        joint_qdd = np.empty(count)
        joint_q[0] = initial_q[joint]
        joint_qd[0] = initial_state.qd_rad_s[joint]
        joint_qdd[0] = initial_state.qdd_rad_s2[joint]
        joint_jerk_command = solution * jerk_scale
        # Jerk is the authoritative control, and these equations are the same
        # semi-implicit integration used to build the affine QP constraints.
        for step in range(1, count):
            joint_qdd[step] = (
                joint_qdd[step - 1]
                + joint_jerk_command[step - 1] * dt
            )
            joint_qd[step] = joint_qd[step - 1] + joint_qdd[step] * dt
            joint_q[step] = joint_q[step - 1] + joint_qd[step] * dt
        joint_jerk = np.diff(joint_qdd) / dt
        certificate_tolerance = _SOLVER_CERTIFICATE_TOLERANCE_RAD
        certificate_passed = bool(
            np.all(
                joint_q
                >= limits.min_position_rad[joint] - certificate_tolerance
            )
            and np.all(
                joint_q
                <= limits.max_position_rad[joint] + certificate_tolerance
            )
            and np.all(
                np.abs(joint_qd[1:])
                <= limits.max_velocity_rad_s[joint] + certificate_tolerance
            )
            and np.all(
                np.abs(joint_qdd[1:])
                <= (
                    limits.max_acceleration_rad_s2[joint]
                    + certificate_tolerance
                )
            )
            and np.all(
                np.abs(joint_jerk)
                <= limits.max_jerk_rad_s3[joint] + certificate_tolerance
            )
        )
        if not certificate_passed:
            min_position_margin = float(
                np.min(joint_q - limits.min_position_rad[joint])
            )
            max_position_margin = float(
                np.min(limits.max_position_rad[joint] - joint_q)
            )
            raise RuntimeError(
                f"joint horizon QP candidate failed certification for joint "
                f"{joint + 1}: status={result.info.status}; "
                f"certificate_passed={certificate_passed}; "
                f"primal_residual={result.info.prim_res}; "
                f"dual_residual={result.info.dual_res}; "
                f"min_position_margin={min_position_margin}; "
                f"max_position_margin={max_position_margin}; "
                f"max_velocity={float(np.max(np.abs(joint_qd[1:])))}; "
                f"max_acceleration={float(np.max(np.abs(joint_qdd[1:])))}; "
                f"max_jerk={float(np.max(np.abs(joint_jerk)))}."
            )
        q_path[:, joint] = joint_q
        qd_path[:, joint] = joint_qd
        qdd_path[:, joint] = joint_qdd
        receipts.append(
            {
                "joint_index": joint + 1,
                "status": str(result.info.status),
                "iterations": int(result.info.iter),
                "solve_time_sec": float(result.info.solve_time),
                "hard_constraint_certificate_passed": certificate_passed,
                "accepted_suboptimal": bool(result.info.status_val == 7),
                "lower_position_interior_rad": lower_position_interior,
                "upper_position_interior_rad": upper_position_interior,
            }
        )
    return JointHorizonPlan(
        q_path_rad=q_path,
        qd_path_rad_s=qd_path,
        qdd_path_rad_s2=qdd_path,
        commit_count=committed,
        dt_sec=dt,
        metadata={
            "planner": "jerk_limited_joint_horizon_qp_v2_jerk_only",
            "bound_guard": guard,
            "solver_tolerance": solver_eps,
            "horizon_transitions": transitions,
            "jerk_braking_viability": True,
            "position_constraint_interior_rad": (
                _POSITION_CONSTRAINT_INTERIOR_RAD
            ),
            "solver_receipts": tuple(receipts),
        },
    )


def plan_jerk_limited_joint_horizon_batched(
    reference_qpos: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_state: JointHorizonState,
    limits: JointHorizonLimits,
    dt_sec: float,
    commit_count: int,
    bound_guard: float = 0.999,
    solver_tolerance: float = 1e-7,
) -> JointHorizonPlan:
    """Solve the same independent per-joint QPs in one block-diagonal OSQP.

    This is numerically equivalent to :func:`plan_jerk_limited_joint_horizon`
    but pays Python/OSQP setup overhead once per source cut.  It is intended
    for the 15 Hz source-cut / 120 Hz command governor.
    """

    if not isinstance(initial_state, JointHorizonState):
        raise TypeError("initial_state must be JointHorizonState.")
    if not isinstance(limits, JointHorizonLimits):
        raise TypeError("limits must be JointHorizonLimits.")
    dt = float(dt_sec)
    guard = float(bound_guard)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_sec must be finite and positive.")
    if not math.isfinite(guard) or not 0.0 < guard <= 1.0:
        raise ValueError("bound_guard must be finite and in (0, 1].")
    solver_eps = float(solver_tolerance)
    if not math.isfinite(solver_eps) or not 0.0 < solver_eps <= 1e-4:
        raise ValueError("solver_tolerance must be finite and in (0, 1e-4].")
    reference = np.asarray(reference_qpos, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[0] < 2
        or reference.shape[1] != limits.dimension
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "reference_qpos must be a finite NxD matrix with N >= 2 and "
            "D matching limits."
        )
    if any(
        np.asarray(value).shape != (limits.dimension,)
        for value in (
            initial_state.q_rad,
            initial_state.qd_rad_s,
            initial_state.qdd_rad_s2,
        )
    ):
        raise ValueError("initial_state dimension does not match limits.")
    transitions = reference.shape[0] - 1
    committed = int(commit_count)
    if isinstance(commit_count, bool) or committed != commit_count:
        raise ValueError("commit_count must be an integer.")
    if committed < 1 or committed > transitions:
        raise ValueError("commit_count must be in [1, horizon transitions].")
    below = limits.min_position_rad - initial_state.q_rad
    above = initial_state.q_rad - limits.max_position_rad
    if np.any(below > _POSITION_NUMERIC_TOLERANCE_RAD) or np.any(
        above > _POSITION_NUMERIC_TOLERANCE_RAD
    ):
        raise ValueError("initial position is outside the configured joint limits.")
    initial_q = np.clip(
        initial_state.q_rad,
        limits.min_position_rad,
        limits.max_position_rad,
    )
    horizon = transitions
    integral = np.tril(np.ones((horizon, horizon), dtype=float))
    identity = sparse.eye(horizon, format="csc")
    problems: list[dict[str, Any]] = []
    objectives = []
    constraints = []
    linears = []
    lowers = []
    uppers = []
    for joint in range(limits.dimension):
        q_scale = max(
            1.0,
            float(limits.max_position_rad[joint] - limits.min_position_rad[joint]),
        )
        velocity_scale = float(limits.max_velocity_rad_s[joint])
        acceleration_scale = float(limits.max_acceleration_rad_s2[joint])
        jerk_scale = float(limits.max_jerk_rad_s3[joint])
        q_velocity_ratio = dt * velocity_scale / q_scale
        velocity_acceleration_ratio = dt * acceleration_scale / velocity_scale
        acceleration_jerk_ratio = dt * jerk_scale / acceleration_scale
        acceleration_response = acceleration_jerk_ratio * integral
        acceleration_bias = np.full(
            horizon,
            float(initial_state.qdd_rad_s2[joint] / acceleration_scale),
        )
        velocity_response = (
            velocity_acceleration_ratio * integral @ acceleration_response
        )
        velocity_bias = (
            np.full(
                horizon,
                float(initial_state.qd_rad_s[joint] / velocity_scale),
            )
            + velocity_acceleration_ratio * integral @ acceleration_bias
        )
        position_response = q_velocity_ratio * integral @ velocity_response
        position_bias = (
            np.full(horizon, float(initial_q[joint] / q_scale))
            + q_velocity_ratio * integral @ velocity_bias
        )
        braking_time_sec = acceleration_scale / (2.0 * jerk_scale)
        braking_ratio = braking_time_sec * acceleration_scale / velocity_scale
        viability_response = (
            velocity_response + braking_ratio * acceleration_response
        )
        viability_bias = velocity_bias + braking_ratio * acceleration_bias
        position_response_rad = q_scale * position_response
        position_error_bias_rad = (
            q_scale * position_bias - reference[1:, joint]
        )
        lower_interior = min(
            _POSITION_CONSTRAINT_INTERIOR_RAD,
            max(
                _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                float(initial_q[joint] - limits.min_position_rad[joint]) * 0.5,
            ),
        )
        upper_interior = min(
            _POSITION_CONSTRAINT_INTERIOR_RAD,
            max(
                _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                float(limits.max_position_rad[joint] - initial_q[joint]) * 0.5,
            ),
        )
        objectives.append(
            sparse.csc_matrix(
                2.0
                * (
                    position_response_rad.T @ position_response_rad
                    + 1e-9 * np.eye(horizon)
                )
            )
        )
        linears.append(
            2.0 * position_response_rad.T @ position_error_bias_rad
        )
        constraints.append(
            sparse.vstack(
                (
                    identity,
                    sparse.csc_matrix(acceleration_response),
                    sparse.csc_matrix(velocity_response),
                    sparse.csc_matrix(viability_response),
                    sparse.csc_matrix(position_response_rad),
                ),
                format="csc",
            )
        )
        lowers.append(
            np.concatenate(
                (
                    np.full(horizon, -guard),
                    np.full(horizon, -guard) - acceleration_bias,
                    np.full(horizon, -guard) - velocity_bias,
                    np.full(horizon, -guard) - viability_bias,
                    np.full(
                        horizon,
                        limits.min_position_rad[joint] + lower_interior,
                    )
                    - q_scale * position_bias,
                )
            )
        )
        uppers.append(
            np.concatenate(
                (
                    np.full(horizon, guard),
                    np.full(horizon, guard) - acceleration_bias,
                    np.full(horizon, guard) - velocity_bias,
                    np.full(horizon, guard) - viability_bias,
                    np.full(
                        horizon,
                        limits.max_position_rad[joint] - upper_interior,
                    )
                    - q_scale * position_bias,
                )
            )
        )
        problems.append(
            {
                "jerk_scale": jerk_scale,
                "lower_interior": lower_interior,
                "upper_interior": upper_interior,
            }
        )
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.block_diag(objectives, format="csc"),
        q=np.concatenate(linears),
        A=sparse.block_diag(constraints, format="csc"),
        l=np.concatenate(lowers),
        u=np.concatenate(uppers),
        verbose=False,
        polishing=False,
        eps_abs=solver_eps,
        eps_rel=solver_eps,
        max_iter=100_000,
        adaptive_rho=True,
        adaptive_rho_interval=25,
    )
    result = solver.solve()
    if result.x is None or result.info.status_val not in {1, 2, 7}:
        raise RuntimeError(
            "batched joint horizon QP failed: "
            f"{result.info.status}; primal_residual={result.info.prim_res}; "
            f"dual_residual={result.info.dual_res}."
        )
    solution = np.asarray(result.x, dtype=float).reshape(limits.dimension, horizon)
    count = reference.shape[0]
    q_path = np.empty_like(reference)
    qd_path = np.empty_like(reference)
    qdd_path = np.empty_like(reference)
    receipts: list[dict[str, Any]] = []
    for joint, problem in enumerate(problems):
        joint_q = np.empty(count)
        joint_qd = np.empty(count)
        joint_qdd = np.empty(count)
        joint_q[0] = initial_q[joint]
        joint_qd[0] = initial_state.qd_rad_s[joint]
        joint_qdd[0] = initial_state.qdd_rad_s2[joint]
        joint_jerk_command = solution[joint] * problem["jerk_scale"]
        for step in range(1, count):
            joint_qdd[step] = (
                joint_qdd[step - 1] + joint_jerk_command[step - 1] * dt
            )
            joint_qd[step] = joint_qd[step - 1] + joint_qdd[step] * dt
            joint_q[step] = joint_q[step - 1] + joint_qd[step] * dt
        joint_jerk = np.diff(joint_qdd) / dt
        tolerance = _SOLVER_CERTIFICATE_TOLERANCE_RAD
        passed = bool(
            np.all(joint_q >= limits.min_position_rad[joint] - tolerance)
            and np.all(joint_q <= limits.max_position_rad[joint] + tolerance)
            and np.all(
                np.abs(joint_qd[1:])
                <= limits.max_velocity_rad_s[joint] + tolerance
            )
            and np.all(
                np.abs(joint_qdd[1:])
                <= limits.max_acceleration_rad_s2[joint] + tolerance
            )
            and np.all(
                np.abs(joint_jerk)
                <= limits.max_jerk_rad_s3[joint] + tolerance
            )
        )
        if not passed:
            raise RuntimeError(
                f"batched joint horizon candidate failed certification for "
                f"joint {joint + 1}."
            )
        q_path[:, joint] = joint_q
        qd_path[:, joint] = joint_qd
        qdd_path[:, joint] = joint_qdd
        receipts.append(
            {
                "joint_index": joint + 1,
                "status": str(result.info.status),
                "iterations": int(result.info.iter),
                "solve_time_sec": float(result.info.solve_time),
                "hard_constraint_certificate_passed": True,
                "accepted_suboptimal": bool(result.info.status_val == 7),
                "lower_position_interior_rad": problem["lower_interior"],
                "upper_position_interior_rad": problem["upper_interior"],
            }
        )
    return JointHorizonPlan(
        q_path_rad=q_path,
        qd_path_rad_s=qd_path,
        qdd_path_rad_s2=qdd_path,
        commit_count=committed,
        dt_sec=dt,
        metadata={
            "planner": "jerk_limited_joint_horizon_qp_v3_batched",
            "bound_guard": guard,
            "solver_tolerance": solver_eps,
            "horizon_transitions": transitions,
            "jerk_braking_viability": True,
            "position_constraint_interior_rad": _POSITION_CONSTRAINT_INTERIOR_RAD,
            "solver_receipts": tuple(receipts),
        },
    )


class JointHorizonBatchedWorkspace:
    """Reusable block-diagonal OSQP workspace for a fixed horizon contract.

    Matrix sparsity and factorization depend only on limits, ``dt_sec`` and
    horizon length.  Source-rate callers can therefore build this workspace
    before the command loop, then update only the reference/state vectors at
    each source cut.  The workspace is serialized because OSQP warm-start
    state is mutable.
    """

    def __init__(
        self,
        *,
        limits: JointHorizonLimits,
        dt_sec: float,
        horizon_transitions: int,
        bound_guard: float = 0.999,
        solver_tolerance: float = 1e-7,
        jerk_regularization: float = 1e-9,
        jerk_delta_regularization: float = 0.0,
        terminal_velocity_regularization: float = 0.0,
        terminal_acceleration_regularization: float = 0.0,
    ) -> None:
        if not isinstance(limits, JointHorizonLimits):
            raise TypeError("limits must be JointHorizonLimits.")
        dt = float(dt_sec)
        guard = float(bound_guard)
        solver_eps = float(solver_tolerance)
        regularization = {
            "jerk": float(jerk_regularization),
            "jerk_delta": float(jerk_delta_regularization),
            "terminal_velocity": float(terminal_velocity_regularization),
            "terminal_acceleration": float(
                terminal_acceleration_regularization
            ),
        }
        transitions = int(horizon_transitions)
        if (
            isinstance(horizon_transitions, bool)
            or transitions != horizon_transitions
            or transitions < 1
        ):
            raise ValueError("horizon_transitions must be a positive integer.")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be finite and positive.")
        if not math.isfinite(guard) or not 0.0 < guard <= 1.0:
            raise ValueError("bound_guard must be finite and in (0, 1].")
        if not math.isfinite(solver_eps) or not 0.0 < solver_eps <= 1e-4:
            raise ValueError("solver_tolerance must be finite and in (0, 1e-4].")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in regularization.values()
        ):
            raise ValueError("horizon regularization weights must be finite and >= 0.")

        self.limits = limits
        self.dt_sec = dt
        self.horizon_transitions = transitions
        self.bound_guard = guard
        self.solver_tolerance = solver_eps
        self._regularization = MappingProxyType(regularization)
        integral = np.tril(np.ones((transitions, transitions), dtype=float))
        identity = sparse.eye(transitions, format="csc")
        if transitions > 1:
            jerk_delta = sparse.diags(
                diagonals=(-np.ones(transitions - 1), np.ones(transitions - 1)),
                offsets=(0, 1),
                shape=(transitions - 1, transitions),
                format="csc",
            )
            jerk_delta_quadratic = (jerk_delta.T @ jerk_delta).toarray()
        else:
            jerk_delta_quadratic = np.zeros((1, 1), dtype=float)
        self._problems: list[dict[str, Any]] = []
        objectives = []
        constraints = []
        for joint in range(limits.dimension):
            q_scale = max(
                1.0,
                float(
                    limits.max_position_rad[joint]
                    - limits.min_position_rad[joint]
                ),
            )
            velocity_scale = float(limits.max_velocity_rad_s[joint])
            acceleration_scale = float(limits.max_acceleration_rad_s2[joint])
            jerk_scale = float(limits.max_jerk_rad_s3[joint])
            q_velocity_ratio = dt * velocity_scale / q_scale
            velocity_acceleration_ratio = (
                dt * acceleration_scale / velocity_scale
            )
            acceleration_jerk_ratio = (
                dt * jerk_scale / acceleration_scale
            )
            acceleration_response = acceleration_jerk_ratio * integral
            velocity_response = (
                velocity_acceleration_ratio
                * integral
                @ acceleration_response
            )
            position_response = (
                q_velocity_ratio * integral @ velocity_response
            )
            braking_time_sec = acceleration_scale / (2.0 * jerk_scale)
            braking_ratio = (
                braking_time_sec * acceleration_scale / velocity_scale
            )
            viability_response = (
                velocity_response + braking_ratio * acceleration_response
            )
            position_response_rad = q_scale * position_response
            terminal_velocity_response = velocity_response[-1]
            terminal_acceleration_response = acceleration_response[-1]
            smoothness_quadratic = (
                regularization["jerk"] * np.eye(transitions)
                + regularization["jerk_delta"] * jerk_delta_quadratic
                + regularization["terminal_velocity"]
                * np.outer(
                    terminal_velocity_response,
                    terminal_velocity_response,
                )
                + regularization["terminal_acceleration"]
                * np.outer(
                    terminal_acceleration_response,
                    terminal_acceleration_response,
                )
            )
            objectives.append(
                sparse.csc_matrix(
                    2.0
                    * (
                        position_response_rad.T @ position_response_rad
                        + smoothness_quadratic
                    )
                )
            )
            constraints.append(
                sparse.vstack(
                    (
                        identity,
                        sparse.csc_matrix(acceleration_response),
                        sparse.csc_matrix(velocity_response),
                        sparse.csc_matrix(viability_response),
                        sparse.csc_matrix(position_response_rad),
                    ),
                    format="csc",
                )
            )
            self._problems.append(
                {
                    "q_scale": q_scale,
                    "velocity_scale": velocity_scale,
                    "acceleration_scale": acceleration_scale,
                    "jerk_scale": jerk_scale,
                    "acceleration_response": acceleration_response,
                    "velocity_response": velocity_response,
                    "viability_response": viability_response,
                    "position_response_rad": position_response_rad,
                    "integral": integral,
                    "q_velocity_ratio": q_velocity_ratio,
                    "velocity_acceleration_ratio": velocity_acceleration_ratio,
                    "acceleration_jerk_ratio": acceleration_jerk_ratio,
                    "braking_ratio": braking_ratio,
                    "terminal_velocity_response": terminal_velocity_response,
                    "terminal_acceleration_response": (
                        terminal_acceleration_response
                    ),
                }
            )

        objective = sparse.block_diag(objectives, format="csc")
        constraint = sparse.block_diag(constraints, format="csc")
        row_count = int(constraint.shape[0])
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=objective,
            q=np.zeros(limits.dimension * transitions, dtype=float),
            A=constraint,
            l=np.full(row_count, -np.inf, dtype=float),
            u=np.full(row_count, np.inf, dtype=float),
            verbose=False,
            polishing=False,
            eps_abs=solver_eps,
            eps_rel=solver_eps,
            max_iter=100_000,
            adaptive_rho=True,
            adaptive_rho_interval=25,
        )
        self._lock = threading.Lock()

    @property
    def regularization_metadata(self) -> dict[str, float]:
        return dict(self._regularization)

    def solve(
        self,
        reference_qpos: Sequence[Sequence[float]] | np.ndarray,
        *,
        initial_state: JointHorizonState,
        commit_count: int,
    ) -> JointHorizonPlan:
        """Update vectors, warm-start, solve and certify one source cut."""

        if not isinstance(initial_state, JointHorizonState):
            raise TypeError("initial_state must be JointHorizonState.")
        limits = self.limits
        horizon = self.horizon_transitions
        reference = np.asarray(reference_qpos, dtype=float)
        if (
            reference.shape != (horizon + 1, limits.dimension)
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError(
                "reference_qpos must be a finite matrix shaped "
                f"({horizon + 1}, {limits.dimension})."
            )
        if any(
            np.asarray(value).shape != (limits.dimension,)
            for value in (
                initial_state.q_rad,
                initial_state.qd_rad_s,
                initial_state.qdd_rad_s2,
            )
        ):
            raise ValueError("initial_state dimension does not match limits.")
        committed = int(commit_count)
        if (
            isinstance(commit_count, bool)
            or committed != commit_count
            or committed < 1
            or committed > horizon
        ):
            raise ValueError(
                "commit_count must be an integer in [1, horizon transitions]."
            )
        below = limits.min_position_rad - initial_state.q_rad
        above = initial_state.q_rad - limits.max_position_rad
        if np.any(below > _POSITION_NUMERIC_TOLERANCE_RAD) or np.any(
            above > _POSITION_NUMERIC_TOLERANCE_RAD
        ):
            raise ValueError(
                "initial position is outside the configured joint limits."
            )
        initial_q = np.clip(
            initial_state.q_rad,
            limits.min_position_rad,
            limits.max_position_rad,
        )

        linears = []
        lowers = []
        uppers = []
        solve_data: list[dict[str, Any]] = []
        guard = self.bound_guard
        for joint, problem in enumerate(self._problems):
            acceleration_bias = np.full(
                horizon,
                float(
                    initial_state.qdd_rad_s2[joint]
                    / problem["acceleration_scale"]
                ),
            )
            velocity_bias = (
                np.full(
                    horizon,
                    float(
                        initial_state.qd_rad_s[joint]
                        / problem["velocity_scale"]
                    ),
                )
                + problem["velocity_acceleration_ratio"]
                * problem["integral"]
                @ acceleration_bias
            )
            position_bias = (
                np.full(
                    horizon,
                    float(initial_q[joint] / problem["q_scale"]),
                )
                + problem["q_velocity_ratio"]
                * problem["integral"]
                @ velocity_bias
            )
            viability_bias = (
                velocity_bias + problem["braking_ratio"] * acceleration_bias
            )
            position_error_bias_rad = (
                problem["q_scale"] * position_bias
                - reference[1:, joint]
            )
            lower_interior = min(
                _POSITION_CONSTRAINT_INTERIOR_RAD,
                max(
                    _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                    float(
                        initial_q[joint]
                        - limits.min_position_rad[joint]
                    )
                    * 0.5,
                ),
            )
            upper_interior = min(
                _POSITION_CONSTRAINT_INTERIOR_RAD,
                max(
                    _POSITION_CONSTRAINT_MIN_INTERIOR_RAD,
                    float(
                        limits.max_position_rad[joint]
                        - initial_q[joint]
                    )
                    * 0.5,
                ),
            )
            linears.append(
                2.0
                * (
                    problem["position_response_rad"].T
                    @ position_error_bias_rad
                    + self._regularization["terminal_velocity"]
                    * problem["terminal_velocity_response"]
                    * velocity_bias[-1]
                    + self._regularization["terminal_acceleration"]
                    * problem["terminal_acceleration_response"]
                    * acceleration_bias[-1]
                )
            )
            lowers.append(
                np.concatenate(
                    (
                        np.full(horizon, -guard),
                        np.full(horizon, -guard) - acceleration_bias,
                        np.full(horizon, -guard) - velocity_bias,
                        np.full(horizon, -guard) - viability_bias,
                        np.full(
                            horizon,
                            limits.min_position_rad[joint] + lower_interior,
                        )
                        - problem["q_scale"] * position_bias,
                    )
                )
            )
            uppers.append(
                np.concatenate(
                    (
                        np.full(horizon, guard),
                        np.full(horizon, guard) - acceleration_bias,
                        np.full(horizon, guard) - velocity_bias,
                        np.full(horizon, guard) - viability_bias,
                        np.full(
                            horizon,
                            limits.max_position_rad[joint] - upper_interior,
                        )
                        - problem["q_scale"] * position_bias,
                    )
                )
            )
            solve_data.append(
                {
                    "lower_interior": lower_interior,
                    "upper_interior": upper_interior,
                }
            )

        with self._lock:
            self._solver.update(
                q=np.concatenate(linears),
                l=np.concatenate(lowers),
                u=np.concatenate(uppers),
            )
            result = self._solver.solve()
        if result.x is None or result.info.status_val not in {1, 2, 7}:
            raise RuntimeError(
                "reused batched joint horizon QP failed: "
                f"{result.info.status}; "
                f"primal_residual={result.info.prim_res}; "
                f"dual_residual={result.info.dual_res}."
            )

        solution = np.asarray(result.x, dtype=float).reshape(
            limits.dimension,
            horizon,
        )
        q_path = np.empty_like(reference)
        qd_path = np.empty_like(reference)
        qdd_path = np.empty_like(reference)
        receipts: list[dict[str, Any]] = []
        for joint, problem in enumerate(self._problems):
            joint_q = np.empty(horizon + 1)
            joint_qd = np.empty(horizon + 1)
            joint_qdd = np.empty(horizon + 1)
            joint_q[0] = initial_q[joint]
            joint_qd[0] = initial_state.qd_rad_s[joint]
            joint_qdd[0] = initial_state.qdd_rad_s2[joint]
            joint_jerk_command = solution[joint] * problem["jerk_scale"]
            for step in range(1, horizon + 1):
                joint_qdd[step] = (
                    joint_qdd[step - 1]
                    + joint_jerk_command[step - 1] * self.dt_sec
                )
                joint_qd[step] = (
                    joint_qd[step - 1]
                    + joint_qdd[step] * self.dt_sec
                )
                joint_q[step] = (
                    joint_q[step - 1] + joint_qd[step] * self.dt_sec
                )
            joint_jerk = np.diff(joint_qdd) / self.dt_sec
            tolerance = _SOLVER_CERTIFICATE_TOLERANCE_RAD
            certificate_passed = bool(
                np.all(
                    joint_q
                    >= limits.min_position_rad[joint] - tolerance
                )
                and np.all(
                    joint_q
                    <= limits.max_position_rad[joint] + tolerance
                )
                and np.all(
                    np.abs(joint_qd[1:])
                    <= limits.max_velocity_rad_s[joint] + tolerance
                )
                and np.all(
                    np.abs(joint_qdd[1:])
                    <= limits.max_acceleration_rad_s2[joint] + tolerance
                )
                and np.all(
                    np.abs(joint_jerk)
                    <= limits.max_jerk_rad_s3[joint] + tolerance
                )
            )
            if not certificate_passed:
                raise RuntimeError(
                    "reused batched joint horizon candidate failed "
                    f"certification for joint {joint + 1}."
                )
            q_path[:, joint] = joint_q
            qd_path[:, joint] = joint_qd
            qdd_path[:, joint] = joint_qdd
            receipts.append(
                {
                    "joint_index": joint + 1,
                    "status": str(result.info.status),
                    "iterations": int(result.info.iter),
                    "solve_time_sec": float(result.info.solve_time),
                    "hard_constraint_certificate_passed": True,
                    "accepted_suboptimal": bool(result.info.status_val == 7),
                    **solve_data[joint],
                }
            )
        return JointHorizonPlan(
            q_path_rad=q_path,
            qd_path_rad_s=qd_path,
            qdd_path_rad_s2=qdd_path,
            commit_count=committed,
            dt_sec=self.dt_sec,
            metadata={
                "planner": "jerk_limited_joint_horizon_qp_v4_reused_batched",
                "bound_guard": guard,
                "solver_tolerance": self.solver_tolerance,
                "horizon_transitions": horizon,
                "jerk_braking_viability": True,
                "position_constraint_interior_rad": (
                    _POSITION_CONSTRAINT_INTERIOR_RAD
                ),
                "solver_workspace_reused": True,
                "regularization": dict(self._regularization),
                "solver_receipts": tuple(receipts),
            },
        )


__all__ = [
    "JointHorizonBatchedWorkspace",
    "JointHorizonLimits",
    "JointHorizonPlan",
    "JointHorizonState",
    "plan_jerk_limited_joint_horizon",
    "plan_jerk_limited_joint_horizon_batched",
]
