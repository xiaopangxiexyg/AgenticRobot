"""Collision-aware TCP planning and exact joint-reference resolution for MuJoCo.

This lower adapter reuses RM65 IK, BiRRT-Connect and certified C2 retiming.
It owns neither authority nor dispatch: references still pass through the
deploy runtime's canonical safety, execution gate and position rig. Planning
and collision queries use private MjData, never the live plant.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import RobotAction
from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    plan_collision_aware_joint_path,
)
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import (
    OfflineJointWaypointTrajectory,
    build_offline_quintic_waypoint_trajectory,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix as xyzquat_to_matrix
from robot_motion.rm65 import (
    RM65SingleArmIKConfig,
    build_rm65_single_arm_ik_solver,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from sim.mujoco.collision_safety import (
    CollisionSafetyPolicy,
    KinematicCollisionProjector,
)
from sim.mujoco.ik import IKResult


class MujocoTcpPlanRejected(RuntimeError):
    def __init__(self, reason_code: str, details: Mapping[str, Any] | None = None):
        self.reason_code = reason_code
        self.details = dict(details or {})
        super().__init__(f"{reason_code}: {self.details}")


ConfigurationConstraint = Callable[
    [np.ndarray, Mapping[str, np.ndarray]], JointConfigurationCheck
]


@dataclass(frozen=True)
class _NonContactMovePolicy(CollisionSafetyPolicy):
    @property
    def distance_query_limit_m(self) -> float:
        # Match the existing exact-shadow admission query: broad-radius GJK
        # diagnostics may report zero for separated convex CAD meshes. This
        # is a clipped clearance lower bound, not a measured distance. The
        # required prevention margin itself is unchanged.
        return float(self.minimum_clearance_m)

    def pair_is_forbidden(self, first_group: int, second_group: int) -> bool:
        # A free-space move has no grasp-stage permission to touch an object.
        return super().pair_is_forbidden(first_group, second_group) or frozenset(
            (int(first_group), int(second_group))
        ) == frozenset((4, 8))


class MujocoJointSceneOracle:
    """Bind the existing MuJoCo projector to one frozen complete scene cut."""

    def __init__(
        self,
        scene: Any,
        model_qpos: np.ndarray,
        *,
        clearance_padding_m: float = 0.0,
        configuration_constraint: ConfigurationConstraint | None = None,
        constraint_identity: str | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ):
        if (configuration_constraint is None) != (constraint_identity is None):
            raise ValueError(
                "Configuration constraint and identity must be bound together."
            )
        if configuration_constraint is not None and (
            not callable(configuration_constraint)
            or not isinstance(constraint_identity, str)
            or not constraint_identity.strip()
        ):
            raise ValueError(
                "Configuration constraint requires a callable and identity."
            )
        self.configuration_constraint = configuration_constraint
        self.constraint_identity = constraint_identity
        self.cancel_requested = cancel_requested
        clearance_padding_m = float(clearance_padding_m)
        if not math.isfinite(clearance_padding_m) or clearance_padding_m < 0:
            raise ValueError("clearance_padding_m must be finite and non-negative.")
        self.scene = scene
        self.source = scene.mujoco.MjData(scene.model)
        self.source.qpos[:] = model_qpos
        self.addresses = {
            side: np.asarray(
                [
                    scene.joint_qpos_addr(name)
                    for name in scene.arm_specs[side].joint_names
                ],
                dtype=int,
            )
            for side in ("left", "right")
        }
        original = scene.collision_projector()
        if original is None or not original.oracle.forbidden_pair_count:
            raise MujocoTcpPlanRejected("safe_tcp_collision_geometry_unavailable")
        self.projector = KinematicCollisionProjector(
            scene.mujoco,
            scene.model,
            np.concatenate(tuple(self.addresses.values())),
            _NonContactMovePolicy(
                **{
                    **vars(original.policy),
                    "minimum_clearance_m": original.policy.minimum_clearance_m
                    + clearance_padding_m,
                }
            ),
        )
        payload = {
            "model_qpos": np.asarray(model_qpos).tolist(),
            "geom_size": np.asarray(scene.model.geom_size).tolist(),
            "geom_pos": np.asarray(scene.model.geom_pos).tolist(),
            "policy": vars(self.projector.policy),
            "query_policy": "noncontact_threshold_clipped_v1",
        }
        if constraint_identity is not None:
            payload["configuration_constraint"] = constraint_identity
        self.scene_hash = canonical_content_hash(
            payload,
            domain="dexteleop.mujoco.safe_tcp_scene.v1",
        )

    def check(
        self, qpos_by_component: Mapping[str, np.ndarray]
    ) -> JointConfigurationCheck:
        if self.cancel_requested is not None and self.cancel_requested():
            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
        q = np.concatenate([qpos_by_component[side] for side in ("left", "right")])
        answer = self.projector.check_qpos(self.source, q)
        if answer.safe and self.configuration_constraint is not None:
            constraint = self.configuration_constraint(
                self.source.qpos, qpos_by_component
            )
            if not constraint.safe:
                return constraint
        distance = float(answer.minimum_distance_m)
        return JointConfigurationCheck(
            safe=answer.safe,
            minimum_distance_m=distance if math.isfinite(distance) else None,
            reason=None if answer.safe else "mujoco_configured_scene_collision",
            metadata={**answer.to_dict(), "distance_is_threshold_clipped": True},
        )


@dataclass(frozen=True)
class MujocoTcpMotionPlan:
    path: JointPathPlanResult
    trajectory: OfflineJointWaypointTrajectory
    source_model_qpos: np.ndarray
    reference_poses: tuple[tuple[float, ...], ...]
    # Binds the exact retimed references, not just the pre-smoothing path.
    trajectory_hash: str
    configuration_constraint: ConfigurationConstraint | None = None
    constraint_identity: str | None = None


class MujocoPlannedTcpMotion:
    """Lower-owned plan/reference state, consumed by the existing pose resolver."""

    def __init__(
        self,
        *,
        rig: Any,
        task_config: Any,
        pose_projector: Any,
        reference_hz: float,
        planning_clearance_padding_m: float = 0.002,
    ):
        self.rig = rig
        self.scene = rig.scene
        self.task_config = task_config
        self.pose_projector = pose_projector
        self.reference_hz = float(reference_hz)
        self.planning_clearance_padding_m = float(planning_clearance_padding_m)
        if (
            not math.isfinite(self.planning_clearance_padding_m)
            or self.planning_clearance_padding_m < 0
        ):
            raise ValueError(
                "planning_clearance_padding_m must be finite and non-negative."
            )
        self._lock = threading.RLock()
        self._solver = None
        self._active: MujocoTcpMotionPlan | None = None
        self._pending_plan: MujocoTcpMotionPlan | None = None
        self._reference_index = -1
        self._last_cycle = -1
        self._live_oracle: MujocoJointSceneOracle | None = None

    def _model_qpos(self) -> np.ndarray:
        return self.rig.capture_scene(lambda scene: scene.data.qpos.copy())

    def pose_for_configuration(
        self, model_qpos: np.ndarray, pair: Mapping[str, np.ndarray]
    ) -> tuple[float, ...]:
        """Read FK in the bound execution frame using private query data."""
        return self.pose_projector.project(
            model_qpos=model_qpos,
            qpos_by_source=pair,
            component_ids=("right_arm",),
        )["right_arm"]

    def plan(
        self,
        target_pose: tuple[float, ...],
        *,
        translation_tolerance_m: float,
        rotation_tolerance_rad: float,
        cancel_requested: Callable[[], bool],
        configuration_constraint: ConfigurationConstraint | None = None,
        constraint_identity: str | None = None,
        path_search_time_budget_sec: float = 3.0,
    ) -> MujocoTcpMotionPlan:
        if (
            isinstance(path_search_time_budget_sec, bool)
            or not math.isfinite(path_search_time_budget_sec)
            or path_search_time_budget_sec <= 0
        ):
            raise ValueError("path_search_time_budget_sec must be finite and positive.")
        with self._lock:
            if self._active is not None:
                raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
            self._pending_plan = None
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            model_qpos = self._model_qpos()
            # Keep an explicit planning reserve for the physical gripper's
            # small joint/servo motion. Live admission keeps the original
            # scene margin; no collision threshold is reduced or bypassed.
            oracle = MujocoJointSceneOracle(
                self.scene,
                model_qpos,
                clearance_padding_m=self.planning_clearance_padding_m,
                configuration_constraint=configuration_constraint,
                constraint_identity=constraint_identity,
                cancel_requested=cancel_requested,
            )
            start = {
                side: model_qpos[address].copy()
                for side, address in oracle.addresses.items()
            }
            initial = oracle.check(start)
            if not initial.safe:
                raise MujocoTcpPlanRejected("safe_tcp_start_blocked", initial.to_dict())
            if self._solver is None:
                self._solver = build_rm65_single_arm_ik_solver(
                    RM65SingleArmIKConfig(controlled_frame="arm_endpoint")
                )
            solver = self._solver
            offset = endpoint_to_tool_transform(
                load_rm65_tool_profile(
                    self.task_config.right_tool_profile_path, side="right"
                )
            )
            measured = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, start))
            if not np.allclose(
                solver.compute_pose(start["right"]) @ offset, measured, atol=1e-5
            ):
                raise MujocoTcpPlanRejected("safe_tcp_ik_model_tool_mismatch")
            goal = xyzquat_to_matrix(target_pose)
            endpoint_goal = goal @ np.linalg.inv(offset)
            rng = np.random.default_rng(0)
            seeds = [start["right"], np.clip(np.zeros(6), solver.q_min, solver.q_max)]
            seeds.extend(rng.uniform(solver.q_min, solver.q_max) for _ in range(6))
            targets = []
            deadline = time.perf_counter() + 2.0
            for seed in seeds:
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if time.perf_counter() >= deadline:
                    break
                q = solver.solve_iterative(
                    target_pose=endpoint_goal,
                    q_init=seed,
                    max_iterations=100,
                    max_joint_step=0.2,
                    damping=0.01,
                    tolerance=1e-5,
                    deadline_monotonic=deadline,
                )
                pair = {"left": start["left"], "right": np.asarray(q)}
                actual = xyzquat_to_matrix(
                    self.pose_for_configuration(model_qpos, pair)
                )
                position_error = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
                angle = float(
                    np.arccos(
                        np.clip(
                            (np.trace(actual[:3, :3].T @ goal[:3, :3]) - 1) / 2, -1, 1
                        )
                    )
                )
                if (
                    position_error <= translation_tolerance_m
                    and angle <= rotation_tolerance_rad
                ):
                    targets.append(pair)
                    # A nearby valid branch is preferable to remote random seeds.
                    if oracle.check(pair).safe:
                        break
            if not targets:
                raise MujocoTcpPlanRejected("safe_tcp_endpoint_search_exhausted")
            path = plan_collision_aware_joint_path(
                start_qpos_by_component=start,
                target_qpos_by_component=targets[0],
                alternative_target_qpos_by_component=targets[1:],
                movable_components=("right",),
                oracle=oracle,
                config=JointPathPlannerConfig(
                    time_budget_sec=path_search_time_budget_sec,
                    edge_resolution_rad=0.01,
                    adaptive_edge_resolution_rad=0.002,
                    adaptive_refinement_distance_m=0.01,
                ),
                joint_lower_by_component={"right": solver.q_min},
                joint_upper_by_component={"right": solver.q_max},
                scene_hash=oracle.scene_hash,
            )
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if not path.path_found:
                raise MujocoTcpPlanRejected(
                    f"safe_tcp_{path.status.value}", path.to_dict()
                )
            q = np.asarray([point["right"] for point in path.waypoints])
            if q.shape[0] == 1:
                q = np.repeat(q, 2, axis=0)
            velocity = np.minimum(
                np.asarray(self.rig.config.max_velocity_rad_s),
                np.deg2rad(self.rig.config.max_command_step_deg)
                * self.rig.config.timing.command_hz,
            )
            acceleration = np.asarray(self.rig.config.max_acceleration_rad_s2)
            intervals = np.maximum(
                0.2, np.max(np.abs(np.diff(q, axis=0)) / velocity, axis=1)
            )
            limits = JointHorizonLimits(
                min_position_rad=solver.q_min,
                max_position_rad=solver.q_max,
                max_velocity_rad_s=velocity,
                max_acceleration_rad_s2=acceleration,
                max_jerk_rad_s3=acceleration / self.rig.config.velocity_ramp_sec,
            )
            retimed = build_offline_quintic_waypoint_trajectory(
                np.concatenate(([0.0], np.cumsum(intervals))),
                q,
                command_hz=self.reference_hz,
                limits=limits,
                timing_mode="safe_retime_global",
                max_time_scale=100.0,
                stop_at_waypoints=True,
            )
            # Keep RRT corners on the actual reference clock. Otherwise an
            # executor could connect the samples on opposite sides of a knot
            # and cut the verified corner even with zero knot velocities.
            ticks = np.ceil(np.diff(retimed.waypoint_timestamps) * self.reference_hz)
            trajectory = build_offline_quintic_waypoint_trajectory(
                np.concatenate(([0.0], np.cumsum(ticks) / self.reference_hz)),
                q,
                command_hz=self.reference_hz,
                limits=limits,
                stop_at_waypoints=True,
            )
            poses = []
            for index, current in enumerate(trajectory.q_rad):
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                pair = {"left": start["left"], "right": current}
                # Smoothing may leave the polyline. Recheck each reference and
                # every intervening joint-space edge, not only the waypoints.
                previous = trajectory.q_rad[max(0, index - 1)]
                count = max(
                    1, math.ceil(float(np.max(np.abs(current - previous))) / 0.01)
                )
                for alpha in np.linspace(0, 1, count + 1):
                    check = oracle.check(
                        {**pair, "right": previous + alpha * (current - previous)}
                    )
                    if not check.safe:
                        raise MujocoTcpPlanRejected(
                            "safe_tcp_retimed_path_blocked",
                            {
                                "reference_index": index,
                                "edge_alpha": float(alpha),
                                "check": check.to_dict(),
                                "path_metadata": dict(path.metadata),
                            },
                        )
                poses.append(
                    tuple(
                        float(item)
                        for item in self.pose_for_configuration(model_qpos, pair)
                    )
                )
            identity = canonical_content_hash(
                {
                    "path": path.semantic_content_hash,
                    "timestamps": trajectory.timestamps.tolist(),
                    "q": trajectory.q_rad.tolist(),
                },
                domain="dexteleop.mujoco.safe_tcp_joint_trajectory.v1",
            )
            model_qpos.setflags(write=False)
            plan = MujocoTcpMotionPlan(
                path,
                trajectory,
                model_qpos,
                tuple(poses),
                identity,
                configuration_constraint,
                constraint_identity,
            )
            self._pending_plan = plan
            return plan

    def activate(self, plan: MujocoTcpMotionPlan) -> None:
        with self._lock:
            if self._active is not None:
                raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
            if plan is not self._pending_plan:
                raise MujocoTcpPlanRejected("safe_tcp_plan_not_owned_by_runtime")
            live = self.rig.capture_scene(lambda scene: scene.arm_qpos_by_side())
            if any(
                not np.allclose(
                    live[side],
                    plan.path.start_qpos_by_component[side],
                    atol=1e-3,
                    rtol=0,
                )
                for side in ("left", "right")
            ):
                raise MujocoTcpPlanRejected("safe_tcp_start_state_changed")
            self._active = plan
            self._pending_plan = None
            self._reference_index = -1
            self._last_cycle = -1
            self._live_oracle = MujocoJointSceneOracle(
                self.scene,
                self._model_qpos(),
                configuration_constraint=plan.configuration_constraint,
                constraint_identity=plan.constraint_identity,
            )

    def clear(self) -> None:
        with self._lock:
            self._active = None
            self._pending_plan = None
            self._reference_index = -1
            self._last_cycle = -1
            self._live_oracle = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def final_reference_selected(self) -> bool:
        with self._lock:
            return (
                self._active is not None
                and self._reference_index == len(self._active.reference_poses) - 1
            )

    def reference(self, cycle: int) -> tuple[float, ...]:
        if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
            raise ValueError("cycle must be a non-negative integer.")
        with self._lock:
            if self._active is None:
                raise MujocoTcpPlanRejected("safe_tcp_no_active_plan")
            if cycle not in {self._last_cycle, self._last_cycle + 1}:
                raise MujocoTcpPlanRejected("safe_tcp_reference_sequence_changed")
            if cycle == self._last_cycle:
                return self._active.reference_poses[self._reference_index]
            self._last_cycle = cycle
            index = self._reference_index
            if index > 0 and np.any(
                np.isclose(
                    self._active.trajectory.timestamps[index],
                    self._active.trajectory.waypoint_timestamps[1:-1],
                    atol=1e-9,
                    rtol=0,
                )
            ):
                measured = self.rig.measured_qpos_by_side()["right"]
                if np.max(
                    np.abs(measured - self._active.trajectory.q_rad[index])
                ) > np.deg2rad(self.rig.config.max_command_step_deg):
                    # A dispatched reference is not proof that the plant has
                    # reached the corner. Keep that same planned reference;
                    # do not start the next leg or catch up by skipping points.
                    return self._active.reference_poses[index]
            self._reference_index = min(
                index + 1, len(self._active.reference_poses) - 1
            )
            return self._active.reference_poses[self._reference_index]

    def resolve(self, action: RobotAction) -> IKResult:
        with self._lock:
            plan = self._active
            index = self._reference_index
            if plan is None or index < 0:
                raise MujocoTcpPlanRejected("safe_tcp_reference_not_selected")
            requested = np.asarray(action.arm_pose_by_side["right"])
            expected = np.asarray(plan.reference_poses[index], dtype=np.float32)
            if not np.allclose(
                xyzquat_to_matrix(requested),
                xyzquat_to_matrix(expected),
                atol=2e-6,
                rtol=0,
            ):
                raise MujocoTcpPlanRejected("safe_tcp_reference_changed_after_safety")
            q = {
                "left": plan.path.start_qpos_by_component["left"],
                "right": plan.trajectory.q_rad[index],
            }
            # Recheck the prescribed reference against today's scene; no
            # projection or replacement IK branch may silently change it.
            oracle = self._live_oracle
            if oracle is None:
                raise MujocoTcpPlanRejected("safe_tcp_live_scene_unavailable")
            oracle.source.qpos[:] = self._model_qpos()
            if not np.allclose(
                oracle.source.qpos[oracle.addresses["left"]],
                q["left"],
                atol=1e-3,
                rtol=0,
            ):
                raise MujocoTcpPlanRejected("safe_tcp_passive_arm_changed")
            check = oracle.check(q)
            if not check.safe:
                background = set(range(oracle.source.qpos.size)) - set(
                    np.concatenate(tuple(oracle.addresses.values())).tolist()
                )
                changed = {
                    str(i): [
                        float(plan.source_model_qpos[i]),
                        float(oracle.source.qpos[i]),
                    ]
                    for i in sorted(background)
                    if abs(plan.source_model_qpos[i] - oracle.source.qpos[i]) > 1e-8
                }
                raise MujocoTcpPlanRejected(
                    "safe_tcp_live_scene_blocked",
                    {
                        "check": check.to_dict(),
                        "reference_index": index,
                        "changed_background_qpos": changed,
                    },
                )
            start_q = np.concatenate(
                [
                    oracle.source.qpos[oracle.addresses[side]]
                    for side in ("left", "right")
                ]
            )
            requested_q = np.concatenate([q[side] for side in ("left", "right")])
            if oracle.configuration_constraint is not None:
                count = max(
                    1, math.ceil(float(np.max(np.abs(requested_q - start_q))) / 0.01)
                )
                for alpha in np.linspace(0, 1, count + 1):
                    point = start_q + alpha * (requested_q - start_q)
                    constraint = oracle.configuration_constraint(
                        oracle.source.qpos,
                        {
                            "left": point[: len(oracle.addresses["left"])],
                            "right": point[len(oracle.addresses["left"]) :],
                        },
                    )
                    if not constraint.safe:
                        raise MujocoTcpPlanRejected(
                            "safe_tcp_live_reference_constraint_blocked",
                            {"edge_alpha": float(alpha), "check": constraint.to_dict()},
                        )
            sweep = oracle.projector.project(
                oracle.source, start_qpos=start_q, requested_qpos=requested_q
            )
            if sweep.modified or not sweep.check.safe:
                raise MujocoTcpPlanRejected(
                    "safe_tcp_live_reference_sweep_blocked", sweep.to_dict()
                )
            return IKResult(
                success=True,
                iterations=0,
                qpos_by_side={side: values.tolist() for side, values in q.items()},
                position_error_m={"right": 0.0},
                orientation_error_rad={"right": 0.0},
                status="planned_joint_reference",
                solver="mujoco.registered_collision_checked_joint_path",
                metadata={
                    "trajectory_hash": plan.trajectory_hash,
                    "reference_index": index,
                },
            )
