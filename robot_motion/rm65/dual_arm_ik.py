from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


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


class _AdmittedCartesianVelocityContinuityTask:
    """Soft Cartesian qdot continuity that does not constrain joint null-space.

    The desired displacement is derived from the previous formally admitted
    qdot, but projected through each controlled-frame Jacobian.  Consequently,
    an equivalent wrist redistribution or posture escape with negligible TCP
    motion remains available to the IK optimizer.
    """

    def __init__(
        self,
        *,
        frame_tasks: list[Any],
        reference: _AdmittedReferenceMotionLimit,
        pin: Any,
        model: Any,
        cost: float,
    ) -> None:
        if not frame_tasks:
            raise ValueError("Cartesian velocity continuity requires frame tasks.")
        self.frame_tasks = tuple(frame_tasks)
        self.reference = reference
        self.pin = pin
        self.model = model
        self.cost = float(cost)
        self.active = True
        self.activation_metadata: dict[str, Any] = {
            "active": True,
            "reason": "reference_not_initialized",
        }
        self.direction_data = model.createData()

    @staticmethod
    def _normalized_sigma_min(
        jacobian: np.ndarray,
        *,
        characteristic_length_m: float = 0.5,
    ) -> float:
        normalized = np.asarray(jacobian, dtype=float).copy()
        normalized[:3, :] /= float(characteristic_length_m)
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        return float(np.min(singular_values)) if singular_values.size else 0.0

    def update_activation(self) -> None:
        reference = self.reference
        if reference.anchor_q is None:
            self.active = True
            self.activation_metadata = {
                "active": True,
                "reason": "reference_not_initialized",
            }
            return
        anchor = np.asarray(reference.anchor_q, dtype=float)
        predicted = self.pin.integrate(
            self.model,
            anchor,
            reference.previous_velocity_rad_s * float(reference.dt),
        )
        sides: dict[str, dict[str, float | bool]] = {}
        entering_guarded = False
        for task in self.frame_tasks:
            frame_id = self.model.getFrameId(str(task.frame))
            current_jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.direction_data,
                anchor,
                frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            predicted_jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.direction_data,
                predicted,
                frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            current_sigma = self._normalized_sigma_min(current_jacobian)
            predicted_sigma = self._normalized_sigma_min(predicted_jacobian)
            guarded_and_decreasing = bool(
                current_sigma < 0.04 and predicted_sigma < current_sigma - 1e-6
            )
            entering_guarded = bool(entering_guarded or guarded_and_decreasing)
            sides[str(task.frame)] = {
                "current_sigma_min": current_sigma,
                "predicted_sigma_min": predicted_sigma,
                "guarded_and_decreasing": guarded_and_decreasing,
            }
        self.active = not entering_guarded
        self.activation_metadata = {
            "active": self.active,
            "reason": (
                "enabled"
                if self.active
                else "prior_qdot_predicts_deeper_guarded_singularity"
            ),
            "sides": sides,
        }

    def set_target_from_configuration(self, configuration: Any) -> None:
        del configuration

    def compute_qp_objective(
        self,
        configuration: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        nv = int(self.model.nv)
        reference = self.reference
        if reference.anchor_q is None or not self.active:
            return np.zeros((nv, nv), dtype=float), np.zeros(nv, dtype=float)
        current_q = np.asarray(configuration.q, dtype=float)
        anchor_to_current = np.asarray(
            self.pin.difference(self.model, reference.anchor_q, current_q),
            dtype=float,
        ).reshape(nv)
        desired_outer_delta = (
            reference.previous_velocity_rad_s * float(reference.dt)
        )
        remaining_joint_delta = desired_outer_delta - anchor_to_current
        jacobians = [
            np.asarray(task.compute_jacobian(configuration), dtype=float)
            for task in self.frame_tasks
        ]
        jacobian = np.vstack(jacobians)
        desired_cartesian_delta = jacobian @ remaining_joint_delta
        weighted_jacobian = self.cost * jacobian
        weighted_delta = self.cost * desired_cartesian_delta
        return (
            weighted_jacobian.T @ weighted_jacobian,
            -weighted_delta.T @ weighted_jacobian,
        )


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
        self.admitted_cartesian_velocity_continuity_task: Any | None = None
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

    def configure_admitted_cartesian_velocity_continuity_task(
        self,
        cost: float,
    ) -> None:
        """Prefer previous admitted Cartesian motion while leaving null-space free."""

        resolved_cost = float(cost)
        if not np.isfinite(resolved_cost) or resolved_cost <= 0.0:
            raise ValueError("admitted Cartesian velocity continuity cost must be positive.")
        if self.admitted_cartesian_velocity_continuity_task is not None:
            raise RuntimeError(
                "admitted Cartesian velocity continuity task is already configured."
            )
        reference = self.admitted_reference_motion_limit
        if reference is None:
            raise RuntimeError(
                "Cartesian velocity continuity requires an admitted motion anchor."
            )
        frame_tasks = [task for task in self.tasks if isinstance(task, self.FrameTask)]
        task = _AdmittedCartesianVelocityContinuityTask(
            frame_tasks=frame_tasks,
            reference=reference,
            pin=self.pin,
            model=self.robot.model,
            cost=resolved_cost,
        )
        self.admitted_cartesian_velocity_continuity_task = task
        self.tasks.append(task)

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
        continuity_task = self.admitted_cartesian_velocity_continuity_task
        if continuity_task is not None:
            continuity_task.update_activation()

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
            for task in self.tasks:
                task_start = time.perf_counter() if self.debug_profile else 0.0
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
                "admitted_cartesian_velocity_continuity_enabled": (
                    self.admitted_cartesian_velocity_continuity_task is not None
                ),
                "admitted_cartesian_velocity_continuity": (
                    None
                    if self.admitted_cartesian_velocity_continuity_task is None
                    else dict(
                        self.admitted_cartesian_velocity_continuity_task.activation_metadata
                    )
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


__all__ = ["DualArmIKSolver", "MultiTaskIKSolver"]
