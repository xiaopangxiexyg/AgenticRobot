from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _SimpleRateLimiter:
    frequency: float

    def __post_init__(self) -> None:
        self.period = 1.0 / self.frequency
        self._last_time = time.time()

    def sleep(self) -> None:
        now = time.time()
        remaining = self.period - (now - self._last_time)
        if remaining > 0:
            time.sleep(remaining)
        self._last_time = time.time()


class MultiTaskIKSolver:
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
    ):
        try:
            import pinocchio as pin
            import pink
            import qpsolvers
            from pink import solve_ik
            from pink.barriers import SelfCollisionBarrier
            from pink.tasks import FrameTask, PostureTask
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
        self.rate = self._make_rate_limiter(frequency)
        self.dt = self.rate.period
        self.viz = self._build_visualizer(debug_viz)

    def _make_rate_limiter(self, frequency: float) -> Any:
        try:
            from loop_rate_limiters import RateLimiter

            return RateLimiter(frequency=frequency, warn=False)
        except ModuleNotFoundError:
            return _SimpleRateLimiter(frequency=frequency)

    def _build_visualizer(self, debug_viz: bool) -> Any:
        if not debug_viz:
            return None
        try:
            from pink.visualization import start_meshcat_visualizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "debug_viz=True requires pink.visualization and meshcat."
            ) from exc
        return start_meshcat_visualizer(self.robot)

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
                    gain=10,
                    safe_displacement_gain=0.001,
                    d_min=0.03,
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
        if solver is not None:
            return solver
        if "osqp" in self.qpsolvers.available_solvers:
            return "osqp"
        return self.qpsolvers.available_solvers[0]

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

    def solve(self, targets: list[np.ndarray] | None = None) -> np.ndarray:
        if targets is not None:
            frame_tasks = [task for task in self.tasks if isinstance(task, self.FrameTask)]
            if len(targets) != len(frame_tasks):
                raise ValueError(
                    f"Expected {len(frame_tasks)} frame targets, got {len(targets)}."
                )

            frame_task_index = 0
            for task in self.tasks:
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

        velocity = self.solve_ik_fn(
            self.configuration,
            self.tasks,
            self.dt,
            solver=self.solver,
            barriers=getattr(self, "barriers", None),
            damping=0.01,
            safety_break=False,
        )
        self.configuration.integrate_inplace(velocity, self.dt)
        if self.viz is not None:
            self.viz.display(self.configuration.q)
        self.rate.sleep()
        return self.configuration.q.copy()
