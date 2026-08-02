from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def _task_conditioned_jacobian_metrics(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    task_cost: np.ndarray,
    task_gain: float,
    joint_velocity_limit_rad_s: np.ndarray,
    dt_sec: float,
) -> dict[str, Any]:
    """Measure one Pink task in the same weighted local-SE(3) metric.

    Pink optimizes a configuration displacement against ``-gain * error``.
    This observer writes the equivalent velocity problem and normalizes the
    Jacobian columns by the declared joint-velocity capabilities.  Its
    singular values therefore describe task authority per unit joint-velocity
    utilization, while ``directional_required_velocity_utilization`` measures
    how much of that normalized authority the current dead-beat task request
    would consume.

    The helper is intentionally pure and has no threshold or control authority.
    """

    raw_jacobian = np.asarray(jacobian, dtype=float)
    raw_error = np.asarray(error, dtype=float).reshape(-1)
    cost = np.asarray(task_cost, dtype=float).reshape(-1)
    velocity_limit = np.asarray(
        joint_velocity_limit_rad_s,
        dtype=float,
    ).reshape(-1)
    gain = float(task_gain)
    period = float(dt_sec)
    if raw_jacobian.ndim != 2:
        raise ValueError("task Jacobian must be a matrix.")
    task_dim, joint_dim = raw_jacobian.shape
    if raw_error.shape != (task_dim,) or cost.shape != (task_dim,):
        raise ValueError(
            "task error and cost must match the task Jacobian row count."
        )
    if velocity_limit.shape != (joint_dim,):
        raise ValueError(
            "joint velocity limits must match the task Jacobian column count."
        )
    if (
        not np.all(np.isfinite(raw_jacobian))
        or not np.all(np.isfinite(raw_error))
        or not np.all(np.isfinite(cost))
        or not np.all(np.isfinite(velocity_limit))
    ):
        raise ValueError("task-conditioning inputs must be finite.")
    if np.any(cost <= 0.0) or np.any(velocity_limit <= 0.0):
        raise ValueError("task costs and joint velocity limits must be positive.")
    if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("task gain must be finite and in [0, 1].")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("dt_sec must be finite and positive.")

    weighted_jacobian = cost[:, None] * raw_jacobian
    task_velocity_jacobian = weighted_jacobian * velocity_limit[None, :]
    desired_twist = (-gain / period) * raw_error
    weighted_desired_twist = cost * desired_twist
    left_vectors, singular_values, right_vector_rows = np.linalg.svd(
        task_velocity_jacobian,
        full_matrices=False,
    )
    directional_demand = left_vectors.T @ weighted_desired_twist
    epsilon = 1e-12
    required_utilization = np.divide(
        np.abs(directional_demand),
        singular_values,
        out=np.full_like(directional_demand, float("inf")),
        where=singular_values > epsilon,
    )
    signed_modal_utilization = np.divide(
        directional_demand,
        singular_values,
        out=np.zeros_like(directional_demand),
        where=singular_values > epsilon,
    )
    least_squares_joint_utilization = (
        right_vector_rows.T @ signed_modal_utilization
    )
    least_squares_joint_velocity = (
        velocity_limit * least_squares_joint_utilization
    )
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    weakest_index = (
        int(np.argmin(singular_values)) if singular_values.size else None
    )
    return {
        "jacobian_type": "pink_frame_task_se3_log_local",
        "jacobian_row_order": "local_linear_xyz_then_local_angular_xyz",
        "task_weighting": "pink_cost_left_multiply",
        "joint_scaling": "columns_multiplied_by_joint_velocity_limit_rad_s",
        "task_cost": cost.astype(float).tolist(),
        "task_gain": gain,
        "dt_sec": period,
        "joint_velocity_limit_rad_s": velocity_limit.astype(float).tolist(),
        "task_error_local": raw_error.astype(float).tolist(),
        "desired_twist_local_per_sec": desired_twist.astype(float).tolist(),
        "weighted_desired_twist_per_sec": (
            weighted_desired_twist.astype(float).tolist()
        ),
        "singular_values_task_velocity": singular_values.astype(float).tolist(),
        "sigma_task_velocity_min": sigma_min,
        "sigma_task_velocity_max": sigma_max,
        "condition_number_task_velocity": (
            float("inf") if sigma_min <= epsilon else float(sigma_max / sigma_min)
        ),
        "task_direction_rows": left_vectors.T.astype(float).tolist(),
        "joint_utilization_direction_rows": (
            right_vector_rows.astype(float).tolist()
        ),
        "joint_utilization_direction_contract": (
            "right_singular_vectors_in_joint_velocity_limit_normalized_coordinates"
        ),
        "directional_demand": directional_demand.astype(float).tolist(),
        "directional_required_velocity_utilization": (
            required_utilization.astype(float).tolist()
        ),
        "least_squares_joint_velocity_utilization": (
            least_squares_joint_utilization.astype(float).tolist()
        ),
        "least_squares_joint_velocity_rad_s": (
            least_squares_joint_velocity.astype(float).tolist()
        ),
        "least_squares_joint_velocity_contract": (
            "unconstrained_weighted_pseudoinverse_diagnostic_not_command"
        ),
        "max_required_velocity_utilization": (
            float(np.max(required_utilization))
            if required_utilization.size
            else 0.0
        ),
        "weakest_direction_index": weakest_index,
        "weakest_direction_abs_demand": (
            None
            if weakest_index is None
            else float(abs(directional_demand[weakest_index]))
        ),
        "weakest_direction_required_velocity_utilization": (
            None
            if weakest_index is None
            else float(required_utilization[weakest_index])
        ),
    }


class _ExplicitVelocityLimit:
    """Pink-compatible tangent limit backed by an explicit velocity vector.

    Pink's built-in ``VelocityLimit`` reads ``model.velocityLimit`` from the
    URDF.  Production qref profiles are a separate, versioned contract, so a
    solver needs an explicit limit without mutating the shared Pinocchio
    model.  The class deliberately implements only Pink's small Limit
    protocol: ``compute_qp_inequalities(configuration, dt)``.
    """

    def __init__(self, velocity_limit_rad_s: np.ndarray) -> None:
        velocity = np.asarray(velocity_limit_rad_s, dtype=float).reshape(-1)
        if velocity.size == 0:
            raise ValueError("velocity_limit_rad_s must be non-empty.")
        if np.any(np.isnan(velocity)) or np.any(velocity <= 0.0):
            raise ValueError(
                "velocity_limit_rad_s must contain positive values or +inf."
            )
        active = np.isfinite(velocity) & (velocity < 1e20)
        self.indices = np.flatnonzero(active)
        self.velocity_limit_rad_s = velocity.copy()
        self.projection_matrix = (
            np.eye(velocity.size, dtype=float)[self.indices]
            if self.indices.size
            else None
        )

    def compute_qp_inequalities(
        self,
        configuration: Any,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del configuration
        if self.projection_matrix is None:
            return None
        step = float(dt) * self.velocity_limit_rad_s[self.indices]
        return (
            np.vstack([self.projection_matrix, -self.projection_matrix]),
            np.hstack([step, step]),
        )


class _AdmittedReferenceMotionLimit:
    """Bound every numerical IK iterate inside one admitted qref interval.

    Multiple Pink calls may be used to converge one nonlinear pose solve.  They
    are numerical iterations, not additional physical control periods.  This
    limit therefore constrains the *final configuration relative to the outer
    admitted anchor*, rather than granting a fresh velocity/acceleration step
    to every Pink call.
    """

    def __init__(
        self,
        model: Any,
        pin: Any,
        velocity_limit_rad_s: np.ndarray,
        acceleration_limit_rad_s2: np.ndarray | None,
    ) -> None:
        self.model = model
        self.pin = pin
        velocity = np.asarray(velocity_limit_rad_s, dtype=float).reshape(-1)
        if velocity.shape != (int(model.nv),):
            raise ValueError("velocity_limit_rad_s must match model.nv.")
        if np.any(np.isnan(velocity)) or np.any(velocity <= 0.0):
            raise ValueError(
                "velocity_limit_rad_s must contain positive values or +inf."
            )
        acceleration = (
            None
            if acceleration_limit_rad_s2 is None
            else np.asarray(acceleration_limit_rad_s2, dtype=float).reshape(-1)
        )
        if acceleration is not None:
            if acceleration.shape != (int(model.nv),):
                raise ValueError("acceleration_limit_rad_s2 must match model.nv.")
            if np.any(np.isnan(acceleration)) or np.any(acceleration <= 0.0):
                raise ValueError(
                    "acceleration_limit_rad_s2 must contain positive values or +inf."
                )
        active = np.isfinite(velocity) & (velocity < 1e20)
        self.indices = np.flatnonzero(active)
        self.projection_matrix = (
            np.eye(int(model.nv), dtype=float)[self.indices]
            if self.indices.size
            else None
        )
        self.velocity_limit_rad_s = velocity.copy()
        self.acceleration_limit_rad_s2 = (
            None if acceleration is None else acceleration.copy()
        )
        self.anchor_q: np.ndarray | None = None
        self.previous_velocity_rad_s = np.zeros(int(model.nv), dtype=float)
        self.dt = 1.0

    def set_reference(
        self,
        anchor_q: np.ndarray,
        previous_velocity_rad_s: np.ndarray,
        dt_sec: float,
    ) -> None:
        anchor = np.asarray(anchor_q, dtype=float).reshape(-1)
        previous = np.asarray(previous_velocity_rad_s, dtype=float).reshape(-1)
        dt = float(dt_sec)
        if anchor.shape != (int(self.model.nq),):
            raise ValueError("anchor_q must match model.nq.")
        if previous.shape != (int(self.model.nv),):
            raise ValueError("previous_velocity_rad_s must match model.nv.")
        if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(previous)):
            raise ValueError("admitted motion reference must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be finite and positive.")
        self.anchor_q = anchor.copy()
        self.previous_velocity_rad_s = previous.copy()
        self.dt = dt

    def compute_qp_inequalities(
        self,
        configuration: Any,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del dt  # The interval is owned by the outer reference clock.
        if self.projection_matrix is None:
            return None
        anchor = (
            np.asarray(configuration.q, dtype=float).copy()
            if self.anchor_q is None
            else self.anchor_q
        )
        indices = self.indices
        period = float(self.dt)
        velocity_max = self.velocity_limit_rad_s[indices].copy()
        velocity_min = -velocity_max
        acceleration = self.acceleration_limit_rad_s2
        if acceleration is not None:
            accel = acceleration[indices]
            previous = self.previous_velocity_rad_s[indices]
            velocity_max = np.minimum(velocity_max, previous + accel * period)
            velocity_min = np.maximum(velocity_min, previous - accel * period)

            upper_distance = np.asarray(
                self.pin.difference(
                    self.model,
                    anchor,
                    self.model.upperPositionLimit,
                ),
                dtype=float,
            )[indices]
            lower_distance = np.asarray(
                self.pin.difference(
                    self.model,
                    self.model.lowerPositionLimit,
                    anchor,
                ),
                dtype=float,
            )[indices]
            velocity_max = np.minimum(
                velocity_max,
                np.sqrt(2.0 * accel * np.maximum(0.0, upper_distance)),
            )
            velocity_min = np.maximum(
                velocity_min,
                -np.sqrt(2.0 * accel * np.maximum(0.0, lower_distance)),
            )

        anchor_to_current = np.asarray(
            self.pin.difference(
                self.model,
                anchor,
                np.asarray(configuration.q, dtype=float),
            ),
            dtype=float,
        )[indices]
        remaining_upper = period * velocity_max - anchor_to_current
        remaining_lower = period * velocity_min - anchor_to_current
        return (
            np.vstack([self.projection_matrix, -self.projection_matrix]),
            np.hstack([remaining_upper, -remaining_lower]),
        )


class _NormalizedSigmaEscapeTask:
    """Increase guarded minimum singular values along an anchor-local gradient.

    The objective is intentionally soft.  Joint position, qdot, qddot and
    collision constraints remain authoritative.  A central finite-difference
    gradient is computed once per outer admitted reference, not once per Pink
    nonlinear iteration, so every numerical call follows the same bounded
    escape direction.
    """

    def __init__(
        self,
        *,
        frame_tasks: list[Any],
        model: Any,
        pin: Any,
        cost: float,
        guard_sigma: float,
        rate_per_sec: float,
        characteristic_length_m: float = 0.5,
        finite_difference_rad: float = 1e-4,
    ) -> None:
        if not frame_tasks:
            raise ValueError("sigma escape requires at least one frame task.")
        self.frame_tasks = tuple(frame_tasks)
        self.model = model
        self.pin = pin
        self.cost = float(cost)
        self.guard_sigma = float(guard_sigma)
        self.rate_per_sec = float(rate_per_sec)
        self.characteristic_length_m = float(characteristic_length_m)
        self.finite_difference_rad = float(finite_difference_rad)
        self.data = model.createData()
        self.entries: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "active_frame_count": 0,
            "reason": "reference_not_initialized",
        }

    def set_target_from_configuration(self, configuration: Any) -> None:
        del configuration

    def _geometric_jacobian(self, frame_id: int, q: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                int(frame_id),
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ),
            dtype=float,
        )

    def _sigma_min(self, frame_id: int, q: np.ndarray) -> float:
        jacobian = self._geometric_jacobian(frame_id, q).copy()
        jacobian[:3, :] /= self.characteristic_length_m
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        return float(np.min(singular_values)) if singular_values.size else 0.0

    def set_reference(self, anchor_q: np.ndarray, dt_sec: float) -> None:
        anchor = np.asarray(anchor_q, dtype=float).reshape(int(self.model.nq))
        period = float(dt_sec)
        entries: list[dict[str, Any]] = []
        metadata_frames: dict[str, Any] = {}
        for task in self.frame_tasks:
            frame_name = str(task.frame)
            frame_id = int(self.model.getFrameId(frame_name))
            base_jacobian = self._geometric_jacobian(frame_id, anchor)
            sigma = self._sigma_min(frame_id, anchor)
            if sigma >= self.guard_sigma:
                metadata_frames[frame_name] = {
                    "active": False,
                    "sigma_min": sigma,
                    "reason": "outside_guarded_band",
                }
                continue
            active_indices = np.flatnonzero(
                np.linalg.norm(base_jacobian, axis=0) > 1e-10
            )
            gradient = np.zeros(int(self.model.nv), dtype=float)
            epsilon = self.finite_difference_rad
            for index in active_indices:
                tangent = np.zeros(int(self.model.nv), dtype=float)
                tangent[int(index)] = epsilon
                q_plus = self.pin.integrate(self.model, anchor, tangent)
                q_minus = self.pin.integrate(self.model, anchor, -tangent)
                gradient[int(index)] = (
                    self._sigma_min(frame_id, q_plus)
                    - self._sigma_min(frame_id, q_minus)
                ) / (2.0 * epsilon)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= 1e-10:
                metadata_frames[frame_name] = {
                    "active": False,
                    "sigma_min": sigma,
                    "gradient_norm": gradient_norm,
                    "reason": "degenerate_sigma_gradient",
                }
                continue
            target_sigma = min(
                self.guard_sigma,
                sigma + self.rate_per_sec * period,
            )
            entries.append(
                {
                    "frame_name": frame_name,
                    "frame_id": frame_id,
                    "gradient": gradient,
                    "target_sigma": target_sigma,
                }
            )
            metadata_frames[frame_name] = {
                "active": True,
                "sigma_min": sigma,
                "target_sigma": target_sigma,
                "gradient_norm": gradient_norm,
                "active_tangent_indices": active_indices.astype(int).tolist(),
            }
        self.entries = entries
        self.metadata = {
            "active_frame_count": len(entries),
            "reason": "guarded_gradient_ready" if entries else "no_guarded_gradient",
            "frames": metadata_frames,
        }

    def compute_qp_objective(
        self,
        configuration: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        nv = int(self.model.nv)
        if not self.entries:
            return np.zeros((nv, nv), dtype=float), np.zeros(nv, dtype=float)
        current_q = np.asarray(configuration.q, dtype=float)
        gradients: list[np.ndarray] = []
        desired_steps: list[float] = []
        for entry in self.entries:
            current_sigma = self._sigma_min(int(entry["frame_id"]), current_q)
            desired_steps.append(
                max(0.0, float(entry["target_sigma"]) - current_sigma)
            )
            gradients.append(np.asarray(entry["gradient"], dtype=float))
        gradient_matrix = np.vstack(gradients)
        desired = np.asarray(desired_steps, dtype=float)
        weighted_gradient = self.cost * gradient_matrix
        weighted_desired = self.cost * desired
        return (
            weighted_gradient.T @ weighted_gradient,
            -weighted_desired.T @ weighted_gradient,
        )


class _StructuralBranchGuardLimit:
    """Keep an RM65 q3/q5 structural branch on one side of zero.

    RM65's structural singularity is locally sign-changing in q3/q5.  A
    sigma-min objective alone is therefore non-smooth at the crossing and can
    alternate sides between numerical solves.  This limit is only installed
    by an explicit offline profile.  It latches the sign at the outer
    admitted anchor, keeps a small positive margin from zero, and releases the
    latch after the normalized Jacobian has left the guarded band.  Research
    profiles may replace the one-step hard margin with a control-barrier
    inequality and may arm it from a bounded previous-velocity prediction.
    """

    def __init__(
        self,
        *,
        frame_tasks: list[Any],
        model: Any,
        pin: Any,
        guard_sigma: float,
        margin_deg: float,
        release_sigma: float,
        characteristic_length_m: float = 0.5,
        escape_cost: float = 0.0,
        escape_target_deg: float = 0.0,
        escape_guard_sigma: float | None = None,
        constraint_mode: str = "hard_margin",
        barrier_gain: float = 0.0,
        barrier_activation_sigma: float = 0.0,
        prediction_horizon_sec: float = 0.0,
    ) -> None:
        if not frame_tasks:
            raise ValueError("structural branch guard requires frame tasks.")
        if not np.isfinite(guard_sigma) or guard_sigma <= 0.0:
            raise ValueError("structural branch guard_sigma must be positive.")
        if not np.isfinite(margin_deg) or margin_deg < 0.0:
            raise ValueError("structural branch margin_deg must be non-negative.")
        if not np.isfinite(release_sigma) or release_sigma < guard_sigma:
            raise ValueError(
                "structural branch release_sigma must be >= guard_sigma."
            )
        self.frame_tasks = tuple(frame_tasks)
        self.model = model
        self.pin = pin
        self.guard_sigma = float(guard_sigma)
        self.margin_rad = float(np.deg2rad(margin_deg))
        self.release_sigma = float(release_sigma)
        self.characteristic_length_m = float(characteristic_length_m)
        if not np.isfinite(escape_cost) or escape_cost < 0.0:
            raise ValueError("structural branch escape_cost must be non-negative.")
        if not np.isfinite(escape_target_deg) or escape_target_deg < 0.0:
            raise ValueError(
                "structural branch escape_target_deg must be non-negative."
            )
        self.escape_cost = float(escape_cost)
        self.escape_target_rad = float(np.deg2rad(escape_target_deg))
        resolved_escape_guard = (
            guard_sigma if escape_guard_sigma is None else float(escape_guard_sigma)
        )
        if not np.isfinite(resolved_escape_guard) or resolved_escape_guard <= 0.0:
            raise ValueError("structural branch escape_guard_sigma must be positive.")
        if resolved_escape_guard > float(guard_sigma) + 1e-12:
            raise ValueError(
                "structural branch escape_guard_sigma must be <= guard_sigma."
            )
        self.escape_guard_sigma = resolved_escape_guard
        resolved_constraint_mode = str(constraint_mode).strip().lower()
        if resolved_constraint_mode not in {
            "hard_margin",
            "barrier",
            "hybrid_deep_barrier",
        }:
            raise ValueError(
                "structural branch constraint_mode must be "
                "'hard_margin', 'barrier' or 'hybrid_deep_barrier'."
            )
        if not np.isfinite(barrier_gain) or barrier_gain < 0.0:
            raise ValueError(
                "structural branch barrier_gain must be non-negative."
            )
        if (
            resolved_constraint_mode in {"barrier", "hybrid_deep_barrier"}
            and float(barrier_gain) <= 0.0
        ):
            raise ValueError(
                "structural branch barrier_gain must be positive in barrier mode."
            )
        if (
            not np.isfinite(barrier_activation_sigma)
            or barrier_activation_sigma < 0.0
        ):
            raise ValueError(
                "structural branch barrier_activation_sigma must be "
                "non-negative."
            )
        if (
            resolved_constraint_mode == "hybrid_deep_barrier"
            and not 0.0 < float(barrier_activation_sigma) <= float(guard_sigma)
        ):
            raise ValueError(
                "hybrid structural branch barrier_activation_sigma must be "
                "in (0, guard_sigma]."
            )
        if (
            not np.isfinite(prediction_horizon_sec)
            or prediction_horizon_sec < 0.0
        ):
            raise ValueError(
                "structural branch prediction_horizon_sec must be non-negative."
            )
        self.constraint_mode = resolved_constraint_mode
        self.barrier_gain = float(barrier_gain)
        self.barrier_activation_sigma = float(barrier_activation_sigma)
        self.prediction_horizon_sec = float(prediction_horizon_sec)
        self.data = model.createData()
        self.latched_sign_by_frame: dict[str, tuple[int, int]] = {}
        self.entries: list[dict[str, Any]] = []
        self.anchor_q: np.ndarray | None = None
        self.metadata: dict[str, Any] = {
            "active_entry_count": 0,
            "reason": "reference_not_initialized",
        }

    def reset_reference_state(self) -> None:
        """Forget branch state from the previous authoritative motion epoch.

        A planner reset may install a measured configuration on the opposite
        side of a structural boundary.  Keeping a sign latched from the
        constructor seed or from an earlier episode would then turn the guard
        into an unauthorized branch-crossing command.  The next
        :meth:`set_reference` call re-latches from the new authoritative q.
        """

        self.latched_sign_by_frame.clear()
        self.entries = []
        self.anchor_q = None
        self.metadata = {
            "active_entry_count": 0,
            "reason": "reference_state_reset",
        }

    def _geometric_jacobian(self, frame_id: int, q: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                int(frame_id),
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ),
            dtype=float,
        )

    def _sigma_min(self, frame_id: int, q: np.ndarray) -> float:
        jacobian = self._geometric_jacobian(frame_id, q).copy()
        jacobian[:3, :] /= self.characteristic_length_m
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        return float(np.min(singular_values)) if singular_values.size else 0.0

    def _structural_indices(self, frame_id: int, q: np.ndarray) -> tuple[int, int]:
        jacobian = self._geometric_jacobian(frame_id, q)
        active = np.flatnonzero(np.linalg.norm(jacobian, axis=0) > 1e-10)
        if active.size < 6:
            raise ValueError(
                "RM65 structural branch guard could not identify six active arm joints."
            )
        # RM65's structural metrics define q3/q5 as local 0-based offsets 2/4.
        return int(active[2]), int(active[4])

    def set_reference(
        self,
        anchor_q: np.ndarray,
        dt_sec: float,
        previous_velocity_rad_s: np.ndarray | None = None,
    ) -> None:
        anchor = np.asarray(anchor_q, dtype=float).reshape(int(self.model.nq))
        period = float(dt_sec)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("structural branch dt_sec must be positive.")
        previous_velocity = (
            np.zeros(int(self.model.nv), dtype=float)
            if previous_velocity_rad_s is None
            else np.asarray(previous_velocity_rad_s, dtype=float).reshape(
                int(self.model.nv)
            )
        )
        if not np.all(np.isfinite(previous_velocity)):
            raise ValueError(
                "structural branch previous velocity must contain finite values."
            )
        predicted = anchor
        if self.prediction_horizon_sec > 0.0:
            predicted = np.asarray(
                self.pin.integrate(
                    self.model,
                    anchor,
                    previous_velocity * self.prediction_horizon_sec,
                ),
                dtype=float,
            ).reshape(int(self.model.nq))
        self.anchor_q = anchor.copy()
        entries: list[dict[str, Any]] = []
        frames: dict[str, Any] = {}
        current_frames = {str(task.frame): task for task in self.frame_tasks}
        for frame_name, task in current_frames.items():
            del task
            frame_id = int(self.model.getFrameId(frame_name))
            sigma = self._sigma_min(frame_id, anchor)
            predicted_sigma = self._sigma_min(frame_id, predicted)
            prediction_guard_sigma = (
                self.barrier_activation_sigma
                if self.constraint_mode == "hybrid_deep_barrier"
                else self.guard_sigma
            )
            prediction_arms_guard = bool(
                self.prediction_horizon_sec > 0.0
                and predicted_sigma < prediction_guard_sigma
                and predicted_sigma < sigma - 1e-6
            )
            q3_index, q5_index = self._structural_indices(frame_id, anchor)
            structural = np.asarray([q3_index, q5_index], dtype=int)
            selection_q = predicted if prediction_arms_guard else anchor
            abs_values = np.abs(selection_q[structural])
            previous = self.latched_sign_by_frame.get(frame_name)
            if sigma >= self.release_sigma and not prediction_arms_guard:
                self.latched_sign_by_frame.pop(frame_name, None)
                frames[frame_name] = {
                    "active": False,
                    "sigma_min": sigma,
                    "predicted_sigma_min": predicted_sigma,
                    "reason": "outside_release_band",
                }
                continue
            if previous is None and (
                sigma < self.guard_sigma or prediction_arms_guard
            ):
                selected = int(np.argmin(abs_values))
                selected_index = int(structural[selected])
                sign = 1 if float(anchor[selected_index]) >= 0.0 else -1
                if abs(float(anchor[selected_index])) < 1e-8:
                    sign = 1
                self.latched_sign_by_frame[frame_name] = (selected_index, sign)
                previous = (selected_index, sign)
            if previous is None:
                frames[frame_name] = {
                    "active": False,
                    "sigma_min": sigma,
                    "predicted_sigma_min": predicted_sigma,
                    "reason": "waiting_for_guard_entry",
                }
                continue
            selected_index, sign = previous
            barrier_active = bool(
                self.constraint_mode == "barrier"
                or (
                    self.constraint_mode == "hybrid_deep_barrier"
                    and min(sigma, predicted_sigma)
                    < self.barrier_activation_sigma
                )
            )
            entries.append(
                {
                    "frame_name": frame_name,
                    "joint_index": int(selected_index),
                    "sign": int(sign),
                    "barrier_active": barrier_active,
                    "escape_strength": float(
                        np.clip(
                            (self.escape_guard_sigma - sigma)
                            / self.escape_guard_sigma,
                            0.0,
                            1.0,
                        )
                    ),
                }
            )
            frames[frame_name] = {
                "active": True,
                "sigma_min": sigma,
                "predicted_sigma_min": predicted_sigma,
                "prediction_armed": prediction_arms_guard,
                "barrier_active": barrier_active,
                "joint_index": int(selected_index),
                "sign": int(sign),
                "margin_deg": float(np.rad2deg(self.margin_rad)),
                "q_deg": float(np.rad2deg(anchor[selected_index])),
                "structural_indices": [q3_index, q5_index],
            }
        self.entries = entries
        self.metadata = {
            "active_entry_count": len(entries),
            "reason": "latched_branch_guard_ready" if entries else "no_active_branch_guard",
            "guard_sigma": self.guard_sigma,
            "release_sigma": self.release_sigma,
            "margin_deg": float(np.rad2deg(self.margin_rad)),
            "constraint_mode": self.constraint_mode,
            "barrier_gain": self.barrier_gain,
            "barrier_activation_sigma": self.barrier_activation_sigma,
            "prediction_horizon_sec": self.prediction_horizon_sec,
            "dt_sec": period,
            "escape_cost": self.escape_cost,
            "escape_target_deg": float(np.rad2deg(self.escape_target_rad)),
            "escape_guard_sigma": self.escape_guard_sigma,
            "frames": frames,
        }

    def set_target_from_configuration(self, configuration: Any) -> None:
        """Keep the branch target anchored to the outer admitted reference."""

        del configuration

    def compute_qp_objective(
        self,
        configuration: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Softly move the latched structural joint away from zero.

        The hard one-sided limit prevents a numerical solve from crossing the
        latched branch.  This companion objective is deliberately bounded to
        a small branch target, so it can follow a pose-derived qref instead of
        pinning the structural joint at the hard margin.  It is only installed
        by an explicit offline research profile.
        """

        nv = int(self.model.nv)
        if self.escape_cost <= 0.0 or self.anchor_q is None or not self.entries:
            return np.zeros((nv, nv), dtype=float), np.zeros(nv, dtype=float)
        current_q = np.asarray(configuration.q, dtype=float).reshape(int(self.model.nq))
        rows: list[np.ndarray] = []
        desired: list[float] = []
        weights: list[float] = []
        for entry in self.entries:
            strength = float(entry.get("escape_strength", 0.0))
            if strength <= 0.0:
                continue
            index = int(entry["joint_index"])
            sign = float(entry["sign"])
            row = np.zeros(nv, dtype=float)
            row[index] = 1.0
            anchor_signed = sign * float(self.anchor_q[index])
            target_signed = max(anchor_signed, self.escape_target_rad)
            current_signed = sign * float(current_q[index])
            desired_step = max(0.0, target_signed - current_signed)
            # Convert the signed coordinate back to the original tangent
            # coordinate while keeping the objective one-sided.
            rows.append(sign * row)
            desired.append(desired_step)
            weights.append(strength)
        if not rows:
            return np.zeros((nv, nv), dtype=float), np.zeros(nv, dtype=float)
        jacobian = np.vstack(rows)
        target = np.asarray(desired, dtype=float)
        weighted_jacobian = self.escape_cost * np.asarray(weights)[:, None] * jacobian
        weighted_target = self.escape_cost * np.asarray(weights) * target
        return (
            weighted_jacobian.T @ weighted_jacobian,
            -weighted_target.T @ weighted_jacobian,
        )

    def compute_qp_inequalities(
        self,
        configuration: Any,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.entries:
            return None
        period = float(dt)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("structural branch QP dt must be positive.")
        current_q = np.asarray(configuration.q, dtype=float).reshape(int(self.model.nq))
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        for entry in self.entries:
            row = np.zeros(int(self.model.nv), dtype=float)
            index = int(entry["joint_index"])
            sign = float(entry["sign"])
            row[index] = -sign
            rows.append(row)
            branch_margin = sign * float(current_q[index]) - self.margin_rad
            if bool(entry.get("barrier_active", False)):
                # Pink optimizes delta_q.  This is the discrete form of
                # sign*qdot + gain*(sign*q - margin) >= 0: inward motion
                # tapers continuously near the boundary, while tangential and
                # outward motion remain unconstrained.
                bounds.append(self.barrier_gain * period * branch_margin)
            else:
                # Enforce sign * (q + delta_q) >= margin in one QP step.
                bounds.append(branch_margin)
        return np.vstack(rows), np.asarray(bounds, dtype=float)


class MultiTaskIKSolver:
    """Shared dual-arm Pink IK solver without runtime clock ownership.

    The solver computes one bounded reference step per call.  Sampling,
    command and transport clocks belong to the caller; this class never sleeps
    and never starts a viewer.
    """

    def __init__(
        self,
        robot_urdf_path: str | Path,
        robot_package_dirs: list[str | Path] | None = None,
        robot_root_joint: Any = None,
        robot_list_of_joints_to_lock: list[str] | None = None,
        robot_collision_pair_names: list[tuple[str, str]] | None = None,
        solver: str | None = None,
        q_ref: np.ndarray | None = None,
        frequency: float = 7.0,
        debug_viz: bool = False,
        debug_profile: bool = False,
        sleep: bool = True,
        collision_barrier_gain: float = 10.0,
        collision_safe_displacement_gain: float = 0.001,
        collision_min_distance_m: float = 0.03,
        damping: float = 0.01,
    ):
        try:
            import pinocchio as pin
            import pink
            from pink import solve_ik
            from pink.barriers import SelfCollisionBarrier
            from pink.limits import AccelerationLimit
            from pink.tasks import FrameTask, PostureTask
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="no QP solver found on your system.*",
                    category=UserWarning,
                )
                import qpsolvers
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MultiTaskIKSolver requires pinocchio, pink, and qpsolvers."
            ) from exc

        self.pin = pin
        self.pink = pink
        self.solve_ik_fn = solve_ik
        self.FrameTask = FrameTask
        self.PostureTask = PostureTask
        self.AccelerationLimit = AccelerationLimit
        self.SelfCollisionBarrier = SelfCollisionBarrier
        self.qpsolvers = qpsolvers
        self.debug_profile = bool(debug_profile)
        self.debug_viz_requested = bool(debug_viz)
        self.collision_barrier_gain = float(collision_barrier_gain)
        self.collision_safe_displacement_gain = float(collision_safe_displacement_gain)
        self.collision_min_distance_m = float(collision_min_distance_m)
        if not np.isfinite(float(damping)) or float(damping) <= 0.0:
            raise ValueError("damping must be finite and positive.")
        self.damping = float(damping)
        if float(frequency) <= 0.0:
            raise ValueError("frequency must be positive.")
        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency
        self.last_solve_profile: dict[str, float | int | str] = {}
        self.last_collision_distance_profile: dict[str, float | int | bool | None] = {}

        if robot_root_joint is None:
            robot_root_joint = None

        self.robot = self._initialize_robot(
            urdf_path=str(robot_urdf_path),
            package_dirs=None
            if robot_package_dirs is None
            else [str(path) for path in robot_package_dirs],
            root_joint=robot_root_joint,
            list_of_joints_to_lock=robot_list_of_joints_to_lock,
            collision_pair_names=robot_collision_pair_names,
        )
        self.configuration = self._initialize_configuration(q_ref=q_ref)
        self.tasks: list[Any] = []
        self.limits: list[Any] | None = None
        self.explicit_velocity_limit: _ExplicitVelocityLimit | None = None
        self.admitted_reference_motion_limit: _AdmittedReferenceMotionLimit | None = None
        self.normalized_sigma_escape_task: _NormalizedSigmaEscapeTask | None = None
        self.structural_branch_guard_limit: _StructuralBranchGuardLimit | None = None
        self.analytic_branch_anchor_task: Any | None = None
        self._analytic_branch_anchor_q: np.ndarray | None = None
        self.branch_continuity_anchor_task: Any | None = None
        self._branch_continuity_anchor_q: np.ndarray | None = None
        self._branch_continuity_anchor_cost_spec: float | np.ndarray | None = None
        self._branch_continuity_anchor_cost_vector: np.ndarray | None = None
        self.acceleration_limit: Any | None = None
        self.solver = self._select_solver(solver)
        self.viz = None

    def _soft_clamp_configuration_to_limits(
        self,
        q: np.ndarray,
        *,
        tolerance_rad: float = 2e-3,
    ) -> np.ndarray:
        q_array = np.asarray(q, dtype=float).reshape(-1).copy()
        lower = np.asarray(self.robot.model.lowerPositionLimit, dtype=float).reshape(-1)
        upper = np.asarray(self.robot.model.upperPositionLimit, dtype=float).reshape(-1)
        if q_array.shape != lower.shape or q_array.shape != upper.shape:
            return q_array

        limited = np.isfinite(lower) & np.isfinite(upper) & (upper > lower + 1e-9)
        slightly_below = limited & (q_array < lower) & (q_array >= lower - tolerance_rad)
        slightly_above = limited & (q_array > upper) & (q_array <= upper + tolerance_rad)
        if np.any(slightly_below):
            q_array[slightly_below] = lower[slightly_below]
        if np.any(slightly_above):
            q_array[slightly_above] = upper[slightly_above]
        return q_array

    def _clamp_integrated_configuration_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp solver-generated integration output before Pink updates state.

        Pink validates limits inside ``Configuration.update``.  QP integration
        can overshoot an active bound by a few milliradians, which previously
        installed an invalid configuration long enough to emit repeated limit
        warnings (and exposed native geometry code to out-of-range state).
        External feedback keeps the narrower soft-clamp policy in
        :meth:`set_configuration_q`; solver-owned integration is always bounded.
        """

        q_array = np.asarray(q, dtype=float).reshape(-1).copy()
        lower = np.asarray(self.robot.model.lowerPositionLimit, dtype=float).reshape(-1)
        upper = np.asarray(self.robot.model.upperPositionLimit, dtype=float).reshape(-1)
        if q_array.shape != lower.shape or q_array.shape != upper.shape:
            raise ValueError(
                "Integrated configuration shape does not match robot joint limits: "
                f"q={q_array.shape} lower={lower.shape} upper={upper.shape}."
            )
        limited = np.isfinite(lower) & np.isfinite(upper) & (upper > lower + 1e-9)
        q_array[limited] = np.clip(q_array[limited], lower[limited], upper[limited])
        return q_array

    def _initialize_robot(
        self,
        urdf_path: str,
        package_dirs: list[str] | None,
        root_joint: Any,
        list_of_joints_to_lock: list[str] | None,
        collision_pair_names: list[tuple[str, str]] | None,
    ) -> Any:
        robot = self.pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=package_dirs,
            root_joint=root_joint,
        )
        if list_of_joints_to_lock is not None:
            robot = robot.buildReducedRobot(list_of_joints_to_lock=list_of_joints_to_lock)

        if collision_pair_names is not None:
            geometry_count = len(robot.collision_model.geometryObjects)
            missing_geometries = sorted(
                {
                    geometry_name
                    for pair in collision_pair_names
                    for geometry_name in pair
                    if robot.collision_model.getGeometryId(geometry_name) >= geometry_count
                }
            )
            if missing_geometries:
                preview = ", ".join(missing_geometries[:8])
                if len(missing_geometries) > 8:
                    preview += ", ..."
                raise ValueError(
                    "Collision geometry names were not found in the URDF collision model "
                    f"for {urdf_path}: {preview}"
                )
            for link1, link2 in collision_pair_names:
                robot.collision_model.addCollisionPair(
                    self.pin.CollisionPair(
                        robot.collision_model.getGeometryId(link1),
                        robot.collision_model.getGeometryId(link2),
                    )
                )
            robot.collision_data = self.pin.GeometryData(robot.collision_model)
            self.barriers = [
                self.SelfCollisionBarrier(
                    n_collision_pairs=len(robot.collision_model.collisionPairs),
                    gain=self.collision_barrier_gain,
                    safe_displacement_gain=self.collision_safe_displacement_gain,
                    d_min=self.collision_min_distance_m,
                )
            ]
        else:
            robot.collision_data = self.pin.GeometryData(robot.collision_model)
        return robot

    def _initialize_configuration(self, q_ref: np.ndarray | None) -> Any:
        if q_ref is None:
            q_ref = np.zeros(self.robot.nq, dtype=float)
        return self.pink.Configuration(
            self.robot.model,
            self.robot.data,
            q_ref,
            collision_model=self.robot.collision_model,
            collision_data=self.robot.collision_data,
        )

    def _select_solver(self, solver: str | None) -> str:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="no QP solver found on your system.*",
                category=UserWarning,
            )
            available_solvers = tuple(self.qpsolvers.available_solvers)
        if solver is not None:
            if solver not in available_solvers:
                if not available_solvers:
                    raise RuntimeError(
                        "No QP solver backend is installed for qpsolvers. "
                        "Install one or disable dual-arm collision avoidance."
                    )
                raise RuntimeError(
                    f"Requested QP solver '{solver}' is unavailable. "
                    f"Available solvers: {', '.join(available_solvers)}."
                )
            return solver
        if "osqp" in available_solvers:
            return "osqp"
        if available_solvers:
            return available_solvers[0]
        raise RuntimeError(
            "No QP solver backend is installed for qpsolvers. "
            "Install one or disable dual-arm collision avoidance."
        )

    def add_frame_task(
        self, frame_name: str, position_cost: float = 1.0, orientation_cost: float = 1.0
    ) -> None:
        self.tasks.append(
            self.FrameTask(
                frame_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
            )
        )

    def add_posture_task(self, cost: float = 1e-1) -> None:
        self.tasks.append(self.PostureTask(cost=cost))

    def configure_analytic_branch_anchor_task(self, *, cost: float) -> None:
        """Add an opt-in soft posture target supplied by an external IK oracle.

        The anchor is only a QP objective.  Existing configuration, velocity,
        acceleration and collision constraints remain authoritative, and the
        anchor is never integrated or dispatched directly.  Leaving this
        method unused preserves the historical task list and solve behavior.
        """

        resolved_cost = float(cost)
        if not np.isfinite(resolved_cost) or resolved_cost <= 0.0:
            raise ValueError("analytic branch anchor cost must be positive.")
        if self.analytic_branch_anchor_task is not None:
            raise RuntimeError("analytic branch anchor task is already configured.")
        task = self.PostureTask(cost=resolved_cost)
        task.set_target_from_configuration(self.configuration)
        self.tasks.append(task)
        self.analytic_branch_anchor_task = task
        self._analytic_branch_anchor_q = None

    def set_analytic_branch_anchor_q(self, q: np.ndarray | None) -> None:
        """Set or clear the external branch anchor for subsequent solves."""

        task = self.analytic_branch_anchor_task
        if task is None:
            raise RuntimeError(
                "analytic branch anchor task must be configured before setting a target."
            )
        if q is None:
            self._analytic_branch_anchor_q = None
            task.set_target_from_configuration(self.configuration)
            return
        target = np.asarray(q, dtype=float).reshape(-1)
        expected_shape = (int(self.robot.model.nq),)
        if target.shape != expected_shape or not np.all(np.isfinite(target)):
            raise ValueError(
                "analytic branch anchor q must be finite and match model.nq: "
                f"{target.shape} != {expected_shape}."
            )
        target = self._soft_clamp_configuration_to_limits(target)
        lower = np.asarray(
            self.robot.model.lowerPositionLimit,
            dtype=float,
        ).reshape(-1)
        upper = np.asarray(
            self.robot.model.upperPositionLimit,
            dtype=float,
        ).reshape(-1)
        limited = np.isfinite(lower) & np.isfinite(upper) & (upper > lower + 1e-9)
        if np.any(target[limited] < lower[limited]) or np.any(
            target[limited] > upper[limited]
        ):
            raise ValueError("analytic branch anchor q exceeds model position limits.")
        self._analytic_branch_anchor_q = target.copy()
        task.set_target(target)

    def configure_branch_continuity_anchor_task(
        self,
        *,
        cost: float | np.ndarray,
    ) -> None:
        """Add an opt-in posture objective anchored to the admitted qref.

        The ordinary posture task is reset to the current Pink configuration
        during every solve.  That is correct for generic IK, but it is not
        sufficient when the coupled position task has several locally valid
        RM65 branches and the caller replaces the solver configuration with a
        post-selection qref.  This task lets a position runtime supply that
        admitted qref explicitly, optionally with zero cost on non-structural
        joints.  It remains a soft QP objective; limits, collision barriers,
        candidate admission, and execution caps remain authoritative.
        """

        if self.branch_continuity_anchor_task is not None:
            raise RuntimeError(
                "branch continuity anchor task is already configured."
            )
        raw_cost = np.asarray(cost, dtype=float)
        if raw_cost.ndim == 0:
            resolved_cost: float | np.ndarray = float(raw_cost)
            if not np.isfinite(resolved_cost) or resolved_cost <= 0.0:
                raise ValueError(
                    "branch continuity anchor cost must be positive."
                )
        else:
            resolved_cost = raw_cost.reshape(-1)
            expected_shape = (int(self.robot.model.nv),)
            if resolved_cost.shape != expected_shape:
                raise ValueError(
                    "branch continuity anchor cost must be scalar or match "
                    f"model.nv: {resolved_cost.shape} != {expected_shape}."
                )
            if (
                not np.all(np.isfinite(resolved_cost))
                or np.any(resolved_cost < 0.0)
                or not np.any(resolved_cost > 0.0)
            ):
                raise ValueError(
                    "branch continuity anchor cost vector must be finite, "
                    "non-negative, and non-zero."
                )
            resolved_cost = resolved_cost.copy()
        cost_vector = np.full(
            int(self.robot.model.nv),
            float(resolved_cost),
            dtype=float,
        ) if np.ndim(resolved_cost) == 0 else np.asarray(
            resolved_cost,
            dtype=float,
        ).copy()
        task = self.PostureTask(cost=resolved_cost)
        task.set_target_from_configuration(self.configuration)
        self.tasks.append(task)
        self.branch_continuity_anchor_task = task
        self._branch_continuity_anchor_q = None
        self._branch_continuity_anchor_cost_spec = (
            float(resolved_cost)
            if np.ndim(resolved_cost) == 0
            else np.asarray(resolved_cost, dtype=float).copy()
        )
        self._branch_continuity_anchor_cost_vector = cost_vector

    def set_branch_continuity_anchor_q(
        self,
        q: np.ndarray | None,
        *,
        active_joint_mask: np.ndarray | None = None,
    ) -> None:
        """Set or clear the admitted-qref anchor for subsequent solves."""

        task = self.branch_continuity_anchor_task
        if task is None:
            raise RuntimeError(
                "branch continuity anchor task must be configured before "
                "setting a target."
            )
        if active_joint_mask is None:
            cost_spec = self._branch_continuity_anchor_cost_spec
            if cost_spec is not None:
                task.cost = (
                    float(cost_spec)
                    if np.ndim(cost_spec) == 0
                    else np.asarray(cost_spec, dtype=float).copy()
                )
        else:
            mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1)
            expected_shape = (int(self.robot.model.nv),)
            if mask.shape != expected_shape:
                raise ValueError(
                    "branch continuity anchor active_joint_mask must match "
                    f"model.nv: {mask.shape} != {expected_shape}."
                )
            base_cost = self._branch_continuity_anchor_cost_vector
            if base_cost is None:
                raise RuntimeError(
                    "branch continuity anchor cost is unavailable."
                )
            task.cost = base_cost * mask.astype(float)
        if q is None:
            self._branch_continuity_anchor_q = None
            task.set_target_from_configuration(self.configuration)
            return
        target = np.asarray(q, dtype=float).reshape(-1)
        expected_shape = (int(self.robot.model.nq),)
        if target.shape != expected_shape or not np.all(np.isfinite(target)):
            raise ValueError(
                "branch continuity anchor q must be finite and match "
                f"model.nq: {target.shape} != {expected_shape}."
            )
        target = self._soft_clamp_configuration_to_limits(target)
        lower = np.asarray(
            self.robot.model.lowerPositionLimit,
            dtype=float,
        ).reshape(-1)
        upper = np.asarray(
            self.robot.model.upperPositionLimit,
            dtype=float,
        ).reshape(-1)
        limited = np.isfinite(lower) & np.isfinite(upper) & (upper > lower + 1e-9)
        if np.any(target[limited] < lower[limited]) or np.any(
            target[limited] > upper[limited]
        ):
            raise ValueError(
                "branch continuity anchor q exceeds model position limits."
            )
        self._branch_continuity_anchor_q = target.copy()
        task.set_target(target)

    def configure_acceleration_limit(
        self,
        acceleration_limit_rad_s2: np.ndarray,
    ) -> None:
        """Add a stateful Pink acceleration constraint to the default limits.

        This method is deliberately opt-in.  Leaving it unused preserves the
        historical Pink configuration/velocity limits exactly, which is the
        rollback path for the legacy iterative solver.
        """

        acceleration = np.asarray(
            acceleration_limit_rad_s2,
            dtype=float,
        ).reshape(-1)
        if acceleration.shape != (int(self.robot.model.nv),):
            raise ValueError(
                "acceleration_limit_rad_s2 must match model.nv: "
                f"{acceleration.shape} != {(int(self.robot.model.nv),)}."
            )
        if np.any(np.isnan(acceleration)) or np.any(acceleration <= 0.0):
            raise ValueError(
                "acceleration_limit_rad_s2 must contain positive values or +inf."
            )
        acceleration_limit = self.AccelerationLimit(
            self.robot.model,
            acceleration,
        )
        self.acceleration_limit = acceleration_limit
        self._refresh_motion_limits()

    def configure_velocity_limit(
        self,
        velocity_limit_rad_s: np.ndarray,
    ) -> None:
        """Replace the URDF velocity bound with an explicit qref contract.

        This method is opt-in.  When unused, Pink keeps its historical URDF
        configuration/velocity limits.  ``+inf`` entries leave non-controlled
        tangent coordinates unbounded by this explicit velocity limit.
        """

        velocity = np.asarray(velocity_limit_rad_s, dtype=float).reshape(-1)
        if velocity.shape != (int(self.robot.model.nv),):
            raise ValueError(
                "velocity_limit_rad_s must match model.nv: "
                f"{velocity.shape} != {(int(self.robot.model.nv),)}."
            )
        self.explicit_velocity_limit = _ExplicitVelocityLimit(velocity)
        self._refresh_motion_limits()

    def configure_admitted_reference_motion_limit(
        self,
        velocity_limit_rad_s: np.ndarray,
        acceleration_limit_rad_s2: np.ndarray | None = None,
    ) -> None:
        """Constrain all IK iterations to one outer admitted-qref interval."""

        self.admitted_reference_motion_limit = _AdmittedReferenceMotionLimit(
            self.robot.model,
            self.pin,
            velocity_limit_rad_s,
            acceleration_limit_rad_s2,
        )
        self._refresh_motion_limits()

    def configure_normalized_sigma_escape_task(
        self,
        *,
        cost: float,
        guard_sigma: float,
        rate_per_sec: float,
    ) -> None:
        """Add one offline-research soft task aligned with sigma-min evidence."""

        if self.normalized_sigma_escape_task is not None:
            raise RuntimeError("normalized sigma escape task is already configured.")
        task = _NormalizedSigmaEscapeTask(
            frame_tasks=[
                task for task in self.tasks if isinstance(task, self.FrameTask)
            ],
            model=self.robot.model,
            pin=self.pin,
            cost=float(cost),
            guard_sigma=float(guard_sigma),
            rate_per_sec=float(rate_per_sec),
        )
        self.normalized_sigma_escape_task = task
        self.tasks.append(task)

    def configure_structural_branch_guard(
        self,
        *,
        guard_sigma: float,
        margin_deg: float,
        release_sigma: float,
        escape_cost: float = 0.0,
        escape_target_deg: float = 0.0,
        escape_guard_sigma: float | None = None,
        constraint_mode: str = "hard_margin",
        barrier_gain: float = 0.0,
        barrier_activation_sigma: float = 0.0,
        prediction_horizon_sec: float = 0.0,
    ) -> None:
        """Install an explicit RM65 q3/q5 branch guard for offline research."""

        if self.structural_branch_guard_limit is not None:
            raise RuntimeError("structural branch guard is already configured.")
        guard = _StructuralBranchGuardLimit(
            frame_tasks=[
                task for task in self.tasks if isinstance(task, self.FrameTask)
            ],
            model=self.robot.model,
            pin=self.pin,
            guard_sigma=float(guard_sigma),
            margin_deg=float(margin_deg),
            release_sigma=float(release_sigma),
            escape_cost=float(escape_cost),
            escape_target_deg=float(escape_target_deg),
            escape_guard_sigma=escape_guard_sigma,
            constraint_mode=constraint_mode,
            barrier_gain=float(barrier_gain),
            barrier_activation_sigma=float(barrier_activation_sigma),
            prediction_horizon_sec=float(prediction_horizon_sec),
        )
        self.structural_branch_guard_limit = guard
        if float(escape_cost) > 0.0:
            self.tasks.append(guard)
        self._refresh_motion_limits()

    def set_admitted_reference_motion(
        self,
        anchor_q: np.ndarray,
        previous_velocity_rad_s: np.ndarray,
        dt_sec: float,
    ) -> None:
        if self.admitted_reference_motion_limit is not None:
            self.admitted_reference_motion_limit.set_reference(
                anchor_q,
                previous_velocity_rad_s,
                dt_sec,
            )
        if self.normalized_sigma_escape_task is not None:
            self.normalized_sigma_escape_task.set_reference(anchor_q, dt_sec)
        if self.structural_branch_guard_limit is not None:
            self.structural_branch_guard_limit.set_reference(
                anchor_q,
                dt_sec,
                previous_velocity_rad_s,
            )

    def reset_admitted_reference_motion_state(self) -> None:
        """Start a new authoritative reference epoch without stale branch state."""

        if self.structural_branch_guard_limit is not None:
            self.structural_branch_guard_limit.reset_reference_state()

    def _refresh_motion_limits(self) -> None:
        velocity_limit = self.admitted_reference_motion_limit
        if velocity_limit is None:
            velocity_limit = (
                self.robot.model.velocity_limit
                if self.explicit_velocity_limit is None
                else self.explicit_velocity_limit
            )
        limits = [
            self.robot.model.configuration_limit,
            velocity_limit,
        ]
        floating_base_limit = getattr(
            self.robot.model,
            "floating_base_velocity_limit",
            None,
        )
        if floating_base_limit is not None:
            limits.append(floating_base_limit)
        if self.acceleration_limit is not None:
            limits.append(self.acceleration_limit)
        if self.structural_branch_guard_limit is not None:
            limits.append(self.structural_branch_guard_limit)
        self.limits = limits

    def set_last_integration_velocity(
        self,
        velocity_rad_s: np.ndarray,
    ) -> None:
        """Anchor the next acceleration constraint to admitted motion only."""

        if self.acceleration_limit is None:
            return
        velocity = np.asarray(velocity_rad_s, dtype=float).reshape(-1)
        if velocity.shape != (int(self.robot.model.nv),):
            raise ValueError(
                "velocity_rad_s must match model.nv: "
                f"{velocity.shape} != {(int(self.robot.model.nv),)}."
            )
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_rad_s must contain finite values.")
        self.acceleration_limit.set_last_integration(velocity, self.dt)

    @property
    def current_q(self) -> np.ndarray:
        return np.asarray(self.configuration.q, dtype=float).copy()

    def set_integration_dt(self, dt_sec: float) -> None:
        """Set the caller-owned physical integration period for one QP step."""

        dt = float(dt_sec)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be finite and positive.")
        self.dt = dt
        self.frequency = 1.0 / dt

    def set_configuration_q(self, q: np.ndarray) -> None:
        q_array = self._soft_clamp_configuration_to_limits(q)
        if q_array.shape != self.configuration.q.shape:
            raise ValueError(
                f"Expected configuration shape {self.configuration.q.shape}, got {q_array.shape}."
            )
        try:
            self.configuration.q[:] = q_array
        except Exception:
            self.configuration.q = q_array.copy()
        if self.viz is not None:
            self.viz.display(self.configuration.q)

    def compute_frame_pose(self, frame_name: str, q: np.ndarray | None = None) -> np.ndarray:
        q_eval = self.current_q if q is None else np.asarray(q, dtype=float).reshape(-1)
        frame_id = self.robot.model.getFrameId(str(frame_name))
        if frame_id >= len(self.robot.model.frames):
            raise ValueError(f"Frame '{frame_name}' was not found in the multitask IK model.")
        self.pin.forwardKinematics(self.robot.model, self.robot.data, q_eval)
        self.pin.updateFramePlacements(self.robot.model, self.robot.data)
        transform = self.robot.data.oMf[frame_id]
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = np.asarray(transform.rotation, dtype=float)
        pose[:3, 3] = np.asarray(transform.translation, dtype=float)
        return pose

    def compute_frame_jacobian(
        self,
        frame_name: str,
        q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a local-world-aligned frame Jacobian for diagnostics."""

        q_eval = self.current_q if q is None else np.asarray(q, dtype=float).reshape(-1)
        frame_id = self.robot.model.getFrameId(str(frame_name))
        if frame_id >= len(self.robot.model.frames):
            raise ValueError(f"Frame '{frame_name}' was not found in the multitask IK model.")
        jacobian = self.pin.computeFrameJacobian(
            self.robot.model,
            self.robot.data,
            q_eval,
            frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(jacobian, dtype=float).copy()

    def compute_frame_task_conditioning(
        self,
        targets: list[np.ndarray],
        *,
        q: np.ndarray | None = None,
        joint_indices_by_target: list[np.ndarray] | None = None,
        joint_velocity_limit_rad_s: np.ndarray,
        dt_sec: float,
    ) -> list[dict[str, Any]]:
        """Observe Pink FrameTask conditioning without changing solver state."""

        frame_tasks = [
            task for task in self.tasks if isinstance(task, self.FrameTask)
        ]
        if len(targets) != len(frame_tasks):
            raise ValueError(
                f"Expected {len(frame_tasks)} frame targets, got {len(targets)}."
            )
        if joint_indices_by_target is None:
            indices_by_target = [
                np.arange(int(self.robot.model.nv), dtype=int)
                for _task in frame_tasks
            ]
        else:
            if len(joint_indices_by_target) != len(frame_tasks):
                raise ValueError(
                    "joint_indices_by_target must match the FrameTask count."
                )
            indices_by_target = [
                np.asarray(indices, dtype=int).reshape(-1)
                for indices in joint_indices_by_target
            ]
        velocity_limit = np.asarray(
            joint_velocity_limit_rad_s,
            dtype=float,
        ).reshape(int(self.robot.model.nv))
        q_eval = (
            self.current_q
            if q is None
            else np.asarray(q, dtype=float).reshape(int(self.robot.model.nq))
        )
        diagnostic_configuration = self.pink.Configuration(
            self.robot.model,
            self.robot.model.createData(),
            q_eval,
        )
        results: list[dict[str, Any]] = []
        for task_index, (source_task, target, indices) in enumerate(
            zip(frame_tasks, targets, indices_by_target)
        ):
            if (
                indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= int(self.robot.model.nv))
            ):
                raise ValueError(
                    f"Invalid joint indices for FrameTask {task_index}."
                )
            target_array = np.asarray(target, dtype=float)
            if target_array.shape != (7,) or not np.all(np.isfinite(target_array)):
                raise ValueError(
                    f"Frame task targets should be finite with shape (7,), "
                    f"got {target_array.shape}."
                )
            source_cost = np.asarray(source_task.cost, dtype=float).reshape(6)
            diagnostic_task = self.FrameTask(
                str(source_task.frame),
                position_cost=source_cost[:3],
                orientation_cost=source_cost[3:],
                lm_damping=float(source_task.lm_damping),
                gain=float(source_task.gain),
            )
            diagnostic_task.set_target(
                self.pin.SE3(
                    self.pin.Quaternion(
                        target_array[3:7]
                    ).toRotationMatrix(),
                    target_array[:3],
                )
            )
            jacobian = np.asarray(
                diagnostic_task.compute_jacobian(diagnostic_configuration),
                dtype=float,
            )[:, indices]
            error = np.asarray(
                diagnostic_task.compute_error(diagnostic_configuration),
                dtype=float,
            )
            metrics = _task_conditioned_jacobian_metrics(
                jacobian,
                error,
                task_cost=source_cost,
                task_gain=float(source_task.gain),
                joint_velocity_limit_rad_s=velocity_limit[indices],
                dt_sec=float(dt_sec),
            )
            metrics.update(
                {
                    "frame_name": str(source_task.frame),
                    "joint_indices": indices.astype(int).tolist(),
                    "observational_only": True,
                    "threshold_classification": "not_assigned",
                }
            )
            results.append(metrics)
        return results

    def compute_constrained_task_feasibility(
        self,
        targets: list[np.ndarray],
        *,
        q: np.ndarray | None = None,
        dt_sec: float,
    ) -> dict[str, Any]:
        """Solve read-only QPs that measure constrained task feasibility.

        ``strict_directional_alpha`` is the largest scalar in ``[0, 1]`` for
        which the weighted Pink task displacement can be produced in exactly
        the requested direction under the currently configured QP
        inequalities.  A second tracking QP keeps alpha fixed at one and
        reports the residual direction.  Both use isolated Pinocchio/Pink data;
        neither integrates a configuration nor updates a limit history.
        """

        frame_tasks = [
            task for task in self.tasks if isinstance(task, self.FrameTask)
        ]
        if len(targets) != len(frame_tasks):
            raise ValueError(
                f"Expected {len(frame_tasks)} frame targets, got {len(targets)}."
            )
        period = float(dt_sec)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("dt_sec must be finite and positive.")
        q_eval = (
            self.current_q
            if q is None
            else np.asarray(q, dtype=float).reshape(int(self.robot.model.nq))
        )
        collision_model = self.robot.collision_model
        diagnostic_configuration = self.pink.Configuration(
            self.robot.model,
            self.robot.model.createData(),
            q_eval,
            collision_model=collision_model,
            collision_data=self.pin.GeometryData(collision_model),
        )
        diagnostic_tasks: list[Any] = []
        weighted_jacobians: list[np.ndarray] = []
        weighted_displacements: list[np.ndarray] = []
        task_slices: list[slice] = []
        row_start = 0
        for source_task, target in zip(frame_tasks, targets):
            target_array = np.asarray(target, dtype=float)
            if target_array.shape != (7,) or not np.all(np.isfinite(target_array)):
                raise ValueError(
                    "Frame task targets must be finite with shape (7,)."
                )
            source_cost = np.asarray(source_task.cost, dtype=float).reshape(6)
            diagnostic_task = self.FrameTask(
                str(source_task.frame),
                position_cost=source_cost[:3],
                orientation_cost=source_cost[3:],
                lm_damping=float(source_task.lm_damping),
                gain=float(source_task.gain),
            )
            diagnostic_task.set_target(
                self.pin.SE3(
                    self.pin.Quaternion(
                        target_array[3:7]
                    ).toRotationMatrix(),
                    target_array[:3],
                )
            )
            jacobian = np.asarray(
                diagnostic_task.compute_jacobian(diagnostic_configuration),
                dtype=float,
            )
            error = np.asarray(
                diagnostic_task.compute_error(diagnostic_configuration),
                dtype=float,
            ).reshape(6)
            weighted_jacobians.append(source_cost[:, None] * jacobian)
            weighted_displacements.append(
                source_cost * (-float(source_task.gain) * error)
            )
            task_slices.append(slice(row_start, row_start + 6))
            row_start += 6
            diagnostic_tasks.append(diagnostic_task)

        weighted_jacobian = np.vstack(weighted_jacobians)
        weighted_displacement = np.hstack(weighted_displacements)
        constraint_problem = self.pink.build_ik(
            diagnostic_configuration,
            [],
            period,
            damping=0.0,
            limits=self.limits,
            barriers=getattr(self, "barriers", None),
        )
        tangent_dim = int(self.robot.model.nv)
        constraint_blocks: list[dict[str, Any]] = []
        constraint_block_errors: list[str] = []
        constraint_row_cursor = 0
        for constraint_kind, constraint_objects in (
            ("limit", self.limits or []),
            ("barrier", getattr(self, "barriers", None) or []),
        ):
            for constraint_index, constraint_object in enumerate(
                constraint_objects
            ):
                source = (
                    f"{constraint_kind}:{constraint_index}:"
                    f"{type(constraint_object).__name__}"
                )
                try:
                    matvec = constraint_object.compute_qp_inequalities(
                        diagnostic_configuration,
                        period,
                    )
                except Exception as exc:  # diagnostic metadata must not block IK
                    constraint_block_errors.append(
                        f"{source}:{type(exc).__name__}:{exc}"
                    )
                    continue
                if matvec is None:
                    continue
                block_g = np.asarray(matvec[0], dtype=float)
                block_h = np.asarray(matvec[1], dtype=float).reshape(-1)
                if (
                    block_g.ndim != 2
                    or block_g.shape[0] != block_h.size
                    or block_g.shape[1] != tangent_dim
                ):
                    constraint_block_errors.append(
                        f"{source}:invalid_shape:G={block_g.shape}:h={block_h.shape}"
                    )
                    continue
                next_cursor = constraint_row_cursor + block_h.size
                constraint_blocks.append(
                    {
                        "source": source,
                        "kind": constraint_kind,
                        "row_start": constraint_row_cursor,
                        "row_stop": next_cursor,
                    }
                )
                constraint_row_cursor = next_cursor
        expected_constraint_rows = (
            0
            if constraint_problem.G is None
            else int(np.asarray(constraint_problem.G).shape[0])
        )
        constraint_blocks_match_problem = bool(
            not constraint_block_errors
            and constraint_row_cursor == expected_constraint_rows
        )

        def solve_strict_alpha(rows: np.ndarray) -> dict[str, Any]:
            from scipy.optimize import linprog

            selected_jacobian = weighted_jacobian[rows, :]
            selected_displacement = weighted_displacement[rows]
            variable_dim = tangent_dim + 1
            base_g = constraint_problem.G
            base_h = constraint_problem.h
            problem_g: np.ndarray | None = None
            problem_h: np.ndarray | None = None
            if base_g is not None and base_h is not None:
                base_g_array = np.asarray(base_g, dtype=float)
                problem_g = np.hstack(
                    [
                        base_g_array,
                        np.zeros((base_g_array.shape[0], 1), dtype=float),
                    ]
                )
                problem_h = np.asarray(base_h, dtype=float)
            equality_rows = [
                np.hstack(
                    [
                        selected_jacobian,
                        -selected_displacement[:, None],
                    ]
                )
            ]
            equality_bounds = [np.zeros(selected_jacobian.shape[0], dtype=float)]
            if constraint_problem.A is not None and constraint_problem.b is not None:
                base_a = np.asarray(constraint_problem.A, dtype=float)
                equality_rows.append(
                    np.hstack(
                        [
                            base_a,
                            np.zeros((base_a.shape[0], 1), dtype=float),
                        ]
                    )
                )
                equality_bounds.append(np.asarray(constraint_problem.b, dtype=float))
            objective = np.zeros(variable_dim, dtype=float)
            objective[-1] = -1.0
            result = linprog(
                objective,
                A_ub=problem_g,
                b_ub=problem_h,
                A_eq=np.vstack(equality_rows),
                b_eq=np.hstack(equality_bounds),
                bounds=[(None, None)] * tangent_dim + [(0.0, 1.0)],
                method="highs",
            )
            if not bool(result.success) or result.x is None:
                return {
                    "available": False,
                    "reason": "lp_no_solution",
                    "solver_status": int(result.status),
                    "solver_message": str(result.message),
                }
            solution = np.asarray(result.x, dtype=float).reshape(variable_dim)
            delta_q = solution[:tangent_dim]
            alpha = float(np.clip(solution[-1], 0.0, 1.0))
            constraint_slack = (
                np.asarray([], dtype=float)
                if problem_g is None or problem_h is None
                else problem_h - problem_g @ solution
            )
            active_sources: list[dict[str, Any]] = []
            if constraint_blocks_match_problem and constraint_slack.size:
                for block in constraint_blocks:
                    block_slack = constraint_slack[
                        int(block["row_start"]) : int(block["row_stop"])
                    ]
                    active_count = int(
                        np.count_nonzero(block_slack <= 1e-6)
                    )
                    if active_count:
                        active_sources.append(
                            {
                                "source": str(block["source"]),
                                "kind": str(block["kind"]),
                                "active_count": active_count,
                                "minimum_slack": float(np.min(block_slack)),
                            }
                        )
            return {
                "available": True,
                "alpha": alpha,
                "delta_q_norm": float(np.linalg.norm(delta_q)),
                "minimum_inequality_slack": (
                    None
                    if constraint_slack.size == 0
                    else float(np.min(constraint_slack))
                ),
                "active_inequality_count": int(
                    np.count_nonzero(constraint_slack <= 1e-6)
                ),
                "active_inequality_sources": active_sources,
            }

        all_rows = np.arange(weighted_jacobian.shape[0], dtype=int)
        strict_global = solve_strict_alpha(all_rows)
        strict_by_frame = [
            solve_strict_alpha(
                np.arange(task_slice.start, task_slice.stop, dtype=int)
            )
            for task_slice in task_slices
        ]

        tracking_problem = self.pink.build_ik(
            diagnostic_configuration,
            diagnostic_tasks,
            period,
            damping=self.damping,
            limits=self.limits,
            barriers=getattr(self, "barriers", None),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Converted matrix '.*' of your problem to scipy\.sparse\.csc_matrix.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"OSQP exited with status '.*'",
            )
            tracking_result = self.qpsolvers.solve_problem(
                tracking_problem,
                solver=self.solver,
            )
        tracking: dict[str, Any]
        if not bool(tracking_result.found) or tracking_result.x is None:
            tracking = {
                "available": False,
                "reason": "qp_no_solution",
                "frames": [],
            }
        else:
            tracking_delta_q = np.asarray(
                tracking_result.x,
                dtype=float,
            ).reshape(tangent_dim)
            weighted_residual = (
                weighted_jacobian @ tracking_delta_q
                - weighted_displacement
            )
            frame_residuals: list[dict[str, Any]] = []
            for task_slice, task_jacobian in zip(
                task_slices,
                weighted_jacobians,
            ):
                residual = weighted_residual[task_slice]
                left_vectors = np.linalg.svd(
                    task_jacobian,
                    full_matrices=False,
                )[0]
                directional_residual = left_vectors.T @ residual
                frame_residuals.append(
                    {
                        "weighted_residual": residual.astype(float).tolist(),
                        "weighted_residual_norm": float(np.linalg.norm(residual)),
                        "directional_residual": (
                            directional_residual.astype(float).tolist()
                        ),
                        "max_abs_directional_residual": float(
                            np.max(np.abs(directional_residual))
                        ),
                    }
                )
            tracking = {
                "available": True,
                "delta_q_norm": float(np.linalg.norm(tracking_delta_q)),
                "weighted_residual_norm": float(
                    np.linalg.norm(weighted_residual)
                ),
                "frames": frame_residuals,
            }

        return {
            "schema": "robot_motion.rm65.constrained_task_feasibility/v1",
            "available": True,
            "observational_only": True,
            "control_authority": "none",
            "dt_sec": period,
            "constraint_source_diagnostics": {
                "available": constraint_blocks_match_problem,
                "expected_row_count": expected_constraint_rows,
                "classified_row_count": constraint_row_cursor,
                "blocks": constraint_blocks,
                "errors": constraint_block_errors,
            },
            "strict_directional_alpha": {
                "contract": (
                    "max_exact_requested_direction_scale_under_current_qp_inequalities"
                ),
                "global": strict_global,
                "frames": strict_by_frame,
            },
            "best_constrained_tracking": tracking,
        }

    def _scan_loaded_min_collision_distance(self) -> float | None:
        min_distance = float("inf")
        for result in self.robot.collision_data.distanceResults:
            distance_value = getattr(result, "min_distance", None)
            if distance_value is None:
                distance_value = getattr(result, "minimum_distance", None)
            if distance_value is None:
                continue
            distance_float = float(distance_value)
            if np.isfinite(distance_float):
                min_distance = min(min_distance, distance_float)
        if not np.isfinite(min_distance):
            return None
        return min_distance

    def compute_min_collision_distance(self, q: np.ndarray | None = None) -> float | None:
        start_time = time.perf_counter() if self.debug_profile else 0.0
        if len(self.robot.collision_model.collisionPairs) == 0:
            if self.debug_profile:
                self.last_collision_distance_profile = {
                    "recomputed": False,
                    "pair_count": 0,
                    "total_ms": (time.perf_counter() - start_time) * 1000.0,
                    "min_distance_m": None,
                }
            return None
        q_eval = self.current_q if q is None else np.asarray(q, dtype=float).reshape(-1)
        self.pin.computeDistances(
            self.robot.model,
            self.robot.data,
            self.robot.collision_model,
            self.robot.collision_data,
            q_eval,
        )
        min_distance = self._scan_loaded_min_collision_distance()
        if self.debug_profile:
            self.last_collision_distance_profile = {
                "recomputed": True,
                "pair_count": len(self.robot.collision_model.collisionPairs),
                "total_ms": (time.perf_counter() - start_time) * 1000.0,
                "min_distance_m": min_distance,
            }
        return min_distance

    def solve(self, targets: list[np.ndarray] | None = None) -> np.ndarray:
        solve_start = time.perf_counter() if self.debug_profile else 0.0
        target_setup_total = 0.0
        if targets is not None:
            frame_tasks = [task for task in self.tasks if isinstance(task, self.FrameTask)]
            if len(targets) != len(frame_tasks):
                raise ValueError(
                    f"Expected {len(frame_tasks)} frame targets, got {len(targets)}."
                )

            frame_task_index = 0
            branch_anchor_task = getattr(self, "branch_continuity_anchor_task", None)
            branch_anchor_q = getattr(self, "_branch_continuity_anchor_q", None)
            analytic_anchor_task = getattr(self, "analytic_branch_anchor_task", None)
            analytic_anchor_q = getattr(self, "_analytic_branch_anchor_q", None)
            for task in self.tasks:
                task_start = time.perf_counter() if self.debug_profile else 0.0
                if (
                    task is branch_anchor_task
                    and branch_anchor_q is not None
                ):
                    task.set_target(branch_anchor_q)
                elif (
                    task is analytic_anchor_task
                    and analytic_anchor_q is not None
                ):
                    task.set_target(analytic_anchor_q)
                else:
                    task.set_target_from_configuration(self.configuration)
                if isinstance(task, self.FrameTask):
                    target = np.asarray(targets[frame_task_index], dtype=float)
                    if target.shape != (7,):
                        raise ValueError(
                            f"Frame task targets should have shape (7,), got {target.shape}."
                        )
                    task.transform_target_to_world.translation = target[:3]
                    task.transform_target_to_world.rotation = self.pin.Quaternion(
                        target[3:7]
                    ).toRotationMatrix()
                    frame_task_index += 1
                if self.debug_profile:
                    target_setup_total += time.perf_counter() - task_start

        qp_start = time.perf_counter() if self.debug_profile else 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Converted matrix '.*' of your problem to scipy\.sparse\.csc_matrix.*",
            )
            velocity = self.solve_ik_fn(
                self.configuration,
                self.tasks,
                self.dt,
                solver=self.solver,
                limits=self.limits,
                barriers=getattr(self, "barriers", None),
                damping=self.damping,
                safety_break=False,
            )
        qp_elapsed = time.perf_counter() - qp_start if self.debug_profile else 0.0
        integrate_start = time.perf_counter() if self.debug_profile else 0.0
        integrated_q = self.pin.integrate(
            self.robot.model,
            self.configuration.q,
            velocity * self.dt,
        )
        integrated_q = self._clamp_integrated_configuration_to_limits(integrated_q)
        self.configuration.update(integrated_q)
        integrate_elapsed = time.perf_counter() - integrate_start if self.debug_profile else 0.0
        set_elapsed = 0.0
        if self.debug_profile:
            self.last_solve_profile = {
                "solver": self.solver,
                "dt_s": float(self.dt),
                "damping": float(self.damping),
                "task_count": len(self.tasks),
                "explicit_velocity_limit_enabled": (
                    self.explicit_velocity_limit is not None
                ),
                "admitted_reference_motion_limit_enabled": (
                    self.admitted_reference_motion_limit is not None
                ),
                "acceleration_limit_enabled": self.acceleration_limit is not None,
                "normalized_sigma_escape_enabled": (
                    self.normalized_sigma_escape_task is not None
                ),
                "normalized_sigma_escape": (
                    None
                    if self.normalized_sigma_escape_task is None
                    else dict(self.normalized_sigma_escape_task.metadata)
                ),
                "structural_branch_guard_enabled": (
                    self.structural_branch_guard_limit is not None
                ),
                "structural_branch_guard": (
                    None
                    if self.structural_branch_guard_limit is None
                    else dict(self.structural_branch_guard_limit.metadata)
                ),
                "analytic_branch_anchor_enabled": (
                    self.analytic_branch_anchor_task is not None
                ),
                "analytic_branch_anchor_target_active": (
                    self._analytic_branch_anchor_q is not None
                ),
                "branch_continuity_anchor_enabled": (
                    self.branch_continuity_anchor_task is not None
                ),
                "branch_continuity_anchor_target_active": (
                    self._branch_continuity_anchor_q is not None
                ),
                "collision_pair_count": len(self.robot.collision_model.collisionPairs),
                "target_setup_ms": target_setup_total * 1000.0,
                "qp_ms": qp_elapsed * 1000.0,
                "integrate_ms": integrate_elapsed * 1000.0,
                "set_configuration_ms": set_elapsed * 1000.0,
                "total_ms": (time.perf_counter() - solve_start) * 1000.0,
            }
        return self.configuration.q.copy()


DualArmIKSolver = MultiTaskIKSolver


__all__ = [
    "DualArmIKSolver",
    "MultiTaskIKSolver",
    "_task_conditioned_jacobian_metrics",
]
