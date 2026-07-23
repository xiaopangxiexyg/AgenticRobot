from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


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

    @property
    def current_q(self) -> np.ndarray:
        return np.asarray(self.configuration.q, dtype=float).copy()

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
