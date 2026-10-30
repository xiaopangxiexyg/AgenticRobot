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
from types import MappingProxyType
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import RobotAction
from robot_motion.observed_planning import ObservedJointPlanningInput
from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    plan_collision_aware_joint_path,
)
from robot_motion.path_planning.motion_quality import (
    DEFAULT_PATH_MOTION_QUALITY_LIMITS,
    sampled_path_motion_quality,
    tcp_path_edge_cost,
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
from robot_motion.rm65.ik_tuning import load_ik_tuning_config, with_rm65_position_ik_objective
from robot_motion.rm65.obstacle_collision import (
    RM65InheritedStartBlocked,
    RM65ObstacleCollisionOracle,
)
from sim.mujoco.collision_safety import (
    KinematicCollisionProjector,
)
from sim.mujoco.collision_safety.policy import NonContactObjectMovePolicy
from sim.mujoco.ik import IKResult

from .grasp_contact import MujocoGraspContactBinding
from .payload_motion import MujocoPayloadBinding, MujocoPayloadProjector


class MujocoTcpPlanRejected(RuntimeError):
    def __init__(self, reason_code: str, details: Mapping[str, Any] | None = None):
        self.reason_code = reason_code
        self.details = dict(details or {})
        super().__init__(f"{reason_code}: {self.details}")


ConfigurationConstraint = Callable[
    [np.ndarray, Mapping[str, np.ndarray]], JointConfigurationCheck
]


ENDPOINT_IK_MAX_ITERATIONS = 100

CANDIDATE_SCHEDULING_POLICY = {
    "id": "bounded_current_seed_ik_hint.v1",
    "preview_budget_fraction": 0.1,
    "preview_budget_cap_sec": 0.5,
    "max_seed_iterations": ENDPOINT_IK_MAX_ITERATIONS,
    "score": "qualified_then_total_joint_excursion_then_input_index",
    "preview_can_prune": False,
}


def _ik_orientation_weight(normalized, translation_tolerance_m, rotation_tolerance_rad):
    if not isinstance(normalized, bool):
        raise TypeError("normalize_ik_objective must be a bool.")
    if not normalized:
        return None
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
           for value in (translation_tolerance_m, rotation_tolerance_rad)):
        raise ValueError("Normalized IK requires positive finite pose tolerances.")
    return translation_tolerance_m / rotation_tolerance_rad


def _solve_endpoint_seed(solver, goal, offset, seed, *, normalized, deadline,
                         check_cancelled, max_iterations=ENDPOINT_IK_MAX_ITERATIONS):
    """Keep preview and full planning on the same calibrated IK objective."""
    return solver.solve_iterative(
        target_pose=goal if normalized else goal @ np.linalg.inv(offset),
        q_init=seed, max_iterations=max_iterations,
        max_joint_step=0.2, damping=0.01,
        tolerance=1e-5, deadline_monotonic=deadline,
        check_cancelled=check_cancelled,
        **({"eef_from_controlled_frame": offset} if normalized else {}),
    )


OBSERVED_START_CLEARANCE_STRICT = "strict"
OBSERVED_START_CLEARANCE_INHERITED = "inherited-positive-v1"
OBSERVED_START_CLEARANCE_MODES = (OBSERVED_START_CLEARANCE_STRICT, OBSERVED_START_CLEARANCE_INHERITED)
_INHERITED_START_REASONS = frozenset({"collision_clearance_violated", "observed_obstacle_clearance_blocked"})


@dataclass(frozen=True)
class MujocoObservedJointPlanningInput(ObservedJointPlanningInput):
    """An observed cut plus its reviewed start-clearance admission mode.

    ``inherited-positive-v1`` mirrors the RealMan planner: a measured start
    below the ordinary margin, at a strictly positive distance, may only plan
    a path whose dense references prove egress. Contact and penetration stay
    blocked. A plain ObservedJointPlanningInput remains strict.
    """

    start_clearance: str = OBSERVED_START_CLEARANCE_STRICT

    def __post_init__(self):
        super().__post_init__()
        if type(self.start_clearance) is not str or self.start_clearance not in OBSERVED_START_CLEARANCE_MODES:
            raise ValueError("start_clearance must be 'strict' or 'inherited-positive-v1'.")


class _ObservedPlanningOracle:
    """Keep the same constraint/search contract without a native scene oracle."""

    def __init__(self, source, model_qpos, *, constraint, constraint_identity, cancel_requested,
                 query=None):
        if (constraint is None) != (constraint_identity is None):
            raise ValueError("Configuration constraint and identity must be bound together.")
        self.source = source
        # A request-owned inherited start oracle replaces only the collision
        # answer; the envelope constraint below still applies on top of it.
        self.query = source.oracle if query is None else query
        self.model_qpos = model_qpos
        self.constraint = constraint
        self.constraint_identity = constraint_identity
        self.cancel_requested = cancel_requested
        self.scene_hash = canonical_content_hash(
            {"observed_scene": self.query.scene_hash, "constraint": constraint_identity},
            domain="dexteleop.mujoco.observed_planning_query.v1")

    def check(self, pair):
        if self.cancel_requested():
            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
        check = self.query.check(pair)
        if check.safe and self.constraint is not None:
            constrained = self.constraint(self.model_qpos, pair)
            if not constrained.safe:
                return constrained
        return check

    def with_inherited_start(self, start, side, details):
        """Return this query on a request-owned inherited start oracle, or None.

        Same admission as the reviewed RealMan planner: only a positive-distance
        clearance shortfall of the native RM65 observed oracle is eligible.
        Contact, penetration, unavailable queries and envelope refusals keep
        strict start admission. Admission faults reject without a fallback.
        """
        source = self.source.oracle
        if type(source) is not RM65ObstacleCollisionOracle:
            return None
        baseline = source.check(start)
        if (baseline.safe or not baseline.collision_check_available
                or baseline.reason not in _INHERITED_START_REASONS
                or baseline.minimum_distance_m is None or not baseline.minimum_distance_m > 0.0):
            return None
        rejected = {**baseline.to_dict(), **details}
        try:
            query = source.inherited_start_clearance(start, active_component=side)
        except RM65InheritedStartBlocked as error:
            raise MujocoTcpPlanRejected("safe_tcp_start_blocked", {**rejected, **error.details}) from error
        except Exception as error:  # noqa: BLE001 - no fallback from native admission faults
            raise MujocoTcpPlanRejected("safe_tcp_start_blocked",
                {**rejected, "exception_type": type(error).__name__}) from error
        return _ObservedPlanningOracle(self.source, self.model_qpos, constraint=self.constraint,
            constraint_identity=self.constraint_identity, cancel_requested=self.cancel_requested,
            query=query)


def _observed_start_clearance(source) -> str:
    if isinstance(source, MujocoObservedJointPlanningInput):
        return source.start_clearance
    return OBSERVED_START_CLEARANCE_STRICT


class _InheritedStartEgress:
    """Ordered dense proof that inherited near pairs/boxes leave the margin."""

    def __init__(self, start_clearance):
        self.start_clearance = start_clearance
        self.progress = start_clearance.new_dense_progress()
        self.observed_progress = start_clearance.new_observed_dense_progress()

    def advance(self, qpos, location):
        clearance = self.start_clearance
        for names, distances, progress, source in (
            (clearance.active_inherited_pairs, clearance.inherited_active_distances,
             self.progress, None),
            (clearance.observed_active_names, clearance.observed_active_distances,
             self.observed_progress, "observed_rgbd"),
        ):
            if not names:
                continue
            try:
                values = distances(qpos)
            except Exception as error:  # noqa: BLE001 - native/observed queries fail closed
                check = JointConfigurationCheck(safe=False, collision_check_available=False,
                    reason="inherited_start_clearance_distance_unavailable",
                    metadata={"exception_type": type(error).__name__,
                              "source": source or "fixed_scene"})
                raise MujocoTcpPlanRejected("safe_tcp_retimed_path_blocked",
                    {**location, "check": check.to_dict()}) from error
            regression = progress.advance(values)
            if regression is not None:
                raise MujocoTcpPlanRejected("safe_tcp_inherited_start_clearance_regressed",
                    {**location, **({"source": source} if source else {}), **regression})

    def require_exited(self):
        clearance = self.start_clearance
        if not self.progress.all_exited:
            raise MujocoTcpPlanRejected("safe_tcp_inherited_start_clearance_not_exited", {
                "reason": "inherited_start_clearance_not_exited",
                "active_inherited_pair_indices": clearance.active_inherited_pairs})
        if not self.observed_progress.all_exited:
            raise MujocoTcpPlanRejected("safe_tcp_inherited_start_clearance_not_exited", {
                "reason": "inherited_observed_clearance_not_exited",
                "active_inherited_observed_boxes": clearance.observed_active_names})

    def summary(self):
        clearance = self.start_clearance
        return {
            "static_pair_count": len(clearance.static_inherited_pairs),
            "active_pair_count": len(clearance.active_inherited_pairs),
            "active_pairs_exited": self.progress.all_exited,
            "observed_static_box_count": len(clearance.observed_static_names),
            "observed_active_box_count": len(clearance.observed_active_names),
            "observed_active_boxes_exited": self.observed_progress.all_exited,
        }


# Preserve the historical private import as the same canonical policy type.
_NonContactMovePolicy = NonContactObjectMovePolicy


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
        contact_binding: MujocoGraspContactBinding | None = None,
        payload_binding: MujocoPayloadBinding | None = None,
    ):
        if payload_binding is not None and not isinstance(payload_binding, MujocoPayloadBinding):
            raise TypeError("payload_binding must be a measured MujocoPayloadBinding.")
        if payload_binding is not None and contact_binding is not None:
            raise ValueError("Terminal grasp contact and attached payload are distinct operation bindings.")
        if payload_binding is not None:
            payload_binding.validate_planning_source(scene, model_qpos)
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
        self.contact_binding = contact_binding
        self.payload_binding = payload_binding
        clearance_padding_m = float(clearance_padding_m)
        if not math.isfinite(clearance_padding_m) or clearance_padding_m < 0:
            raise ValueError("clearance_padding_m must be finite and non-negative.")
        self.scene = scene
        self.source = scene.mujoco.MjData(scene.model)
        self.source.qpos[:] = model_qpos
        if payload_binding is not None:
            payload_binding._restore_planning_auxiliary_state(self.source)
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
        policy = _NonContactMovePolicy(
            **{
                **vars(original.policy),
                "minimum_clearance_m": original.policy.minimum_clearance_m + clearance_padding_m,
            }
        )
        indices = np.concatenate(tuple(self.addresses.values()))
        self.projector = (
            KinematicCollisionProjector(scene.mujoco, scene.model, indices, policy,
                pair_clearance_m=(None if contact_binding is None else contact_binding.pair_clearances(scene)))
            if payload_binding is None else MujocoPayloadProjector(scene, indices, policy, payload_binding)
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
        if contact_binding is not None:
            payload["contact_binding"] = contact_binding.to_payload()
        if payload_binding is not None:
            payload["payload_binding"] = payload_binding.to_payload()
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
            metadata={
                **answer.to_dict(), "distance_is_threshold_clipped": True,
                # Accepted fingertip contact can have a negative signed
                # distance. Zero requests the planner's finest existing edge
                # sampling; it does not reinterpret contact as free space.
                **({"adaptive_edge_distance_m": max(0.0, distance)}
                   if self.contact_binding is not None or self.payload_binding is not None else {}),
            },
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
    contact_binding: MujocoGraspContactBinding | None = None
    payload_binding: MujocoPayloadBinding | None = None
    # The lower cursor owns stage changes and their original contact policy.
    stages: tuple[tuple[int, MujocoTcpMotionPlan], ...] = ()
    endpoint_search_diagnostics: Mapping[str, Any] | None = None
    observed_planning_input: ObservedJointPlanningInput | None = None
    # Set only when the trajectory starts at the rig's held joint command
    # instead of the measured arm cut (held-object plans): the measured cut
    # that every start-state gate compares against.
    measured_start_qpos_by_component: Mapping[str, np.ndarray] | None = None


def _measured_start(plan) -> Mapping[str, np.ndarray]:
    """The measured arm cut a plan was searched from (its start unless held-command)."""
    measured = getattr(plan, "measured_start_qpos_by_component", None)
    return plan.path.start_qpos_by_component if measured is None else measured


def _frozen_copy(value) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


# A held-object plan starts at the rig's held joint command when it is this
# close to the measured arms. The P-servo arms sag below their command under
# gravity and grasp load (about 1.4 mrad on joint 2, 0.66 mm at the wrist);
# starting at the sagged cut stepped the command down by that sag and pressed
# a still-supported object into its support at every lift start.
HELD_COMMAND_START_TOLERANCE_RAD = 0.005
# Joint speed scale (acceleration scales with its square) for every plan that
# carries a held object. At full speed a carry to the far part of a mat drove
# the soft DMTAC pad contact on the cube from ~0.4 mm to 1.007 mm, past the
# 1 mm hard-contact fault; a half-speed carry to the same far region of the
# mat stayed below 0.1 mm.
HELD_OBJECT_SPEED_SCALE = 0.5


@dataclass(frozen=True)
class MujocoTcpCandidateSelection:
    candidate_index: int
    plan: MujocoTcpMotionPlan
    diagnostics: tuple[Mapping[str, Any], ...]
    all_candidates_checked: bool


# Bound fixed-rate materialization before allocating a native grid.
_MAX_TRAJECTORY_DURATION_SEC = 120.0
_MAX_TRAJECTORY_SAMPLES = 100_000


def _trajectory_duration_limit(reference_hz: float) -> float:
    if not math.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("reference_hz must be finite and positive.")
    # Reserve one tick for floating-point endpoint rounding in the fixed grid.
    return min(_MAX_TRAJECTORY_DURATION_SEC, (_MAX_TRAJECTORY_SAMPLES - 2) / reference_hz)



def _require_bounded_grid(duration: float, reference_hz: float) -> None:
    if not math.isfinite(duration) or not 0 < duration <= _trajectory_duration_limit(reference_hz):
        raise MujocoTcpPlanRejected("safe_tcp_trajectory_budget_exhausted", {
            "duration_sec": duration if math.isfinite(duration) else None,
            "maximum_duration_sec": _trajectory_duration_limit(reference_hz),
            "maximum_samples": _MAX_TRAJECTORY_SAMPLES,
        })



@dataclass(frozen=True)
class MujocoTcpPlanGoal:
    """One explicitly bound goal in a complete, lower-owned approach chain."""

    target_pose: tuple[float, ...]
    translation_tolerance_m: float
    rotation_tolerance_rad: float
    configuration_constraint: ConfigurationConstraint | None = None
    constraint_identity: str | None = None
    contact_binding: MujocoGraspContactBinding | None = None
    speed_scale: float = 1.0

    def __post_init__(self):
        pose = np.asarray(self.target_pose, dtype=float)
        if (
            pose.shape != (7,)
            or not np.isfinite(pose).all()
            or abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1e-5
        ):
            raise ValueError("Chain goals require finite normalized xyz+quat_xyzw.")
        object.__setattr__(self, "target_pose", tuple(float(v) for v in pose))
        for name in ("translation_tolerance_m", "rotation_tolerance_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if (self.configuration_constraint is None) != (
            self.constraint_identity is None
        ):
            raise ValueError(
                "Configuration constraint and identity must be bound together."
            )
        if self.configuration_constraint is not None and (
            not callable(self.configuration_constraint)
            or not isinstance(self.constraint_identity, str)
            or not self.constraint_identity.strip()
        ):
            raise ValueError("Chain constraints require a callable and identity.")
        if self.contact_binding is not None and not isinstance(
            self.contact_binding, MujocoGraspContactBinding
        ):
            raise TypeError("Chain contact policy must be a bound contact policy.")
        if (
            isinstance(self.speed_scale, bool)
            or not math.isfinite(self.speed_scale)
            or not 0 < self.speed_scale <= 1
        ):
            raise ValueError("speed_scale must be finite and in (0, 1].")



def _join_motion_plans(plans: tuple[MujocoTcpMotionPlan, ...]) -> MujocoTcpMotionPlan:
    """Concatenate certified zero-velocity/acceleration knots, without retiming.

    These stages were produced under one lower planner lock from the preceding
    stage's exact terminal joint state. Joining does not smooth across a policy
    boundary, cut a corner, or rerun IK. Each continuous stage certificate remains
    applicable because the join has the same q, qd=0 and qdd=0 on both sides.
    """
    for previous, following in zip(plans, plans[1:]):
        if any(
            not np.array_equal(
                previous.path.target_qpos_by_component[side],
                following.path.start_qpos_by_component[side],
            )
            for side in ("left", "right")
        ):
            raise ValueError("Motion chain joint configurations are not continuous.")
    for plan in plans:
        trajectory = plan.trajectory
        if (
            trajectory.timestamps[0] != 0
            or np.any(np.diff(trajectory.timestamps) <= 0)
            or not np.allclose(trajectory.qd_rad_s[[0, -1]], 0, atol=1e-10, rtol=0)
            or not np.allclose(trajectory.qdd_rad_s2[[0, -1]], 0, atol=1e-10, rtol=0)
        ):
            raise ValueError(
                "Motion chain requires certified stopped stage boundaries."
            )
    offsets = np.cumsum(
        [0.0] + [float(p.trajectory.timestamps[-1]) for p in plans[:-1]]
    )

    def joined(name, *, offset=False):
        return np.concatenate(
            [
                (getattr(p.trajectory, name) + (offsets[i] if offset else 0))[
                    1 if i else 0 :
                ]
                for i, p in enumerate(plans)
            ]
        )

    trajectory = OfflineJointWaypointTrajectory(
        timestamps=joined("timestamps", offset=True),
        q_rad=joined("q_rad"),
        qd_rad_s=joined("qd_rad_s"),
        qdd_rad_s2=joined("qdd_rad_s2"),
        jerk_rad_s3=joined("jerk_rad_s3"),
        source_waypoint_timestamps=joined("source_waypoint_timestamps", offset=True),
        waypoint_timestamps=joined("waypoint_timestamps", offset=True),
        waypoint_q_rad=joined("waypoint_q_rad"),
        knot_qd_rad_s=joined("knot_qd_rad_s"),
        knot_qdd_rad_s2=joined("knot_qdd_rad_s2"),
        metadata={
            "join_policy": "certified_stopped_stage_knots_v1",
            "stage_trajectory_hashes": tuple(p.trajectory_hash for p in plans),
        },
    )
    path = replace(
        plans[0].path,
        target_qpos_by_component=plans[-1].path.target_qpos_by_component,
        waypoints=tuple(
            point
            for i, p in enumerate(plans)
            for point in p.path.waypoints[1 if i else 0 :]
        ),
        scene_hash=canonical_content_hash(
            tuple(p.path.scene_hash for p in plans),
            domain="dexteleop.mujoco.tcp_chain_scenes.v1",
        ),
        metadata={
            "stage_path_hashes": tuple(p.path.semantic_content_hash for p in plans)
        },
    )
    starts = np.cumsum([0] + [len(p.reference_poses) - 1 for p in plans[:-1]])
    identity = canonical_content_hash(
        {
            "stages": tuple((int(i), p.trajectory_hash) for i, p in zip(starts, plans)),
            "timestamps": trajectory.timestamps.tolist(),
            "q": trajectory.q_rad.tolist(),
        },
        domain="dexteleop.mujoco.tcp_approach_chain_trajectory.v1",
    )
    return MujocoTcpMotionPlan(
        path=path,
        trajectory=trajectory,
        source_model_qpos=plans[0].source_model_qpos,
        reference_poses=tuple(
            pose
            for i, p in enumerate(plans)
            for pose in p.reference_poses[1 if i else 0 :]
        ),
        trajectory_hash=identity,
        stages=tuple((int(i), p) for i, p in zip(starts, plans)),
    )



def _scene_serialized(*, total_budget=False):
    """Use the rig -> planner lock order already used by live resolution.

    Planning is a frozen-scene operation, not concurrent trajectory execution.
    Holding this existing reentrant scene boundary also prevents a new motion
    from racing a native query; no second mutex or physics owner is introduced.
    """

    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            clock = kwargs.get("monotonic", time.perf_counter)
            if total_budget:
                budget = kwargs["time_budget_sec"]
                if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
                    raise ValueError("time_budget_sec must be finite and positive.")
            began = float(clock()) if total_budget else None
            if total_budget and not math.isfinite(began):
                raise ValueError("Motion chain planning clock must be finite.")

            def invoke(_scene):
                if total_budget:
                    remaining = kwargs["time_budget_sec"] - (float(clock()) - began)
                    if not math.isfinite(remaining):
                        raise ValueError(
                            "Motion chain planning clock and budget must be finite."
                        )
                    if remaining <= 0:
                        raise MujocoTcpPlanRejected(
                            "safe_tcp_planning_budget_exhausted"
                        )
                    return method(
                        self, *args, **{**kwargs, "time_budget_sec": remaining}
                    )
                return method(self, *args, **kwargs)

            return self.rig.capture_scene(invoke)

        return call

    return decorate



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
        motion_side: str = "right",
        source_model_qpos: np.ndarray | None = None,
    ):
        if type(motion_side) is not str or motion_side not in ("left", "right"):
            raise ValueError("motion_side must be exactly 'left' or 'right'.")
        self._motion_side = motion_side
        self._inactive_side = "left" if motion_side == "right" else "right"
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
        self._solver_orientation_weight = None
        self._source_model_qpos = None if source_model_qpos is None else np.asarray(source_model_qpos, dtype=float).copy()
        if self._source_model_qpos is not None:
            if self._source_model_qpos.shape != (self.scene.model.nq,) or not np.isfinite(self._source_model_qpos).all():
                raise ValueError("Planning snapshot must contain one finite complete model qpos.")
            self._source_model_qpos.setflags(write=False)
        self._active: MujocoTcpMotionPlan | None = None
        self._pending_plan: MujocoTcpMotionPlan | None = None
        self._reference_index = -1
        self._last_cycle = -1
        self._live_oracle: MujocoJointSceneOracle | None = None
        self._live_stage: MujocoTcpMotionPlan | None = None
        self._contact_guard_bound = False

    @property
    def motion_side(self) -> str:
        return self._motion_side

    @property
    def component_id(self) -> str:
        return self.motion_side + "_arm"

    def _model_qpos(self) -> np.ndarray:
        if self._source_model_qpos is not None:
            return self._source_model_qpos.copy()
        return self.rig.capture_scene(lambda scene: scene.data.qpos.copy())

    def pose_for_configuration(
        self, model_qpos: np.ndarray, pair: Mapping[str, np.ndarray]
    ) -> tuple[float, ...]:
        """Read FK in the bound execution frame using private query data."""
        return self.pose_projector.project(
            model_qpos=model_qpos,
            qpos_by_source=pair,
            component_ids=(self.component_id,),
        )[self.component_id]

    def _solver_for_ik(self, orientation_weight):
        if self._solver is None or self._solver_orientation_weight != orientation_weight:
            tuning = None if orientation_weight is None else with_rm65_position_ik_objective(
                load_ik_tuning_config(), position_weight=1.0,
                orientation_weight=orientation_weight, mode_name="safe_tcp_tolerance_normalized")
            self._solver = build_rm65_single_arm_ik_solver(
                RM65SingleArmIKConfig(controlled_frame="arm_endpoint", ik_tuning=tuning))
            self._solver_orientation_weight = orientation_weight
        return self._solver


    def _preview_candidate_order(
        self, target_poses, *, translation_tolerance_m, rotation_tolerance_rad,
        normalize_ik_objective, deadline, cancel_requested,
    ):
        """Estimate current-seed travel only; never prune or certify a path."""
        side = self.motion_side
        if self._source_model_qpos is None:
            raise ValueError("Candidate scheduling preview requires a frozen query snapshot.")
        weight = _ik_orientation_weight(
            normalize_ik_objective, translation_tolerance_m, rotation_tolerance_rad)
        model_qpos = self._model_qpos()
        start = {side: np.array([model_qpos[self.scene.joint_qpos_addr(name)]
                 for name in self.scene.arm_specs[side].joint_names])
                 for side in ("left", "right")}
        estimates = {}

        def check_cancelled():
            if cancel_requested() or time.perf_counter() >= deadline:
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")

        if cancel_requested():
            check_cancelled()
        if time.perf_counter() >= deadline:
            return tuple(range(len(target_poses))), estimates
        solver = self._solver_for_ik(weight)
        offset = endpoint_to_tool_transform(
            load_rm65_tool_profile(getattr(self.task_config, side + "_tool_profile_path"), side=side))
        measured = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, start))
        if not np.allclose(solver.compute_pose(start[side]) @ offset, measured, atol=1e-5):
            raise MujocoTcpPlanRejected("safe_tcp_ik_model_tool_mismatch")
        for index, target in enumerate(target_poses):
            if cancel_requested():
                check_cancelled()
            if time.perf_counter() >= deadline:
                break
            started = time.perf_counter()
            goal = xyzquat_to_matrix(target)
            try:
                q = np.asarray(_solve_endpoint_seed(
                    solver, goal, offset, start[side],
                    normalized=normalize_ik_objective, deadline=deadline,
                    max_iterations=CANDIDATE_SCHEDULING_POLICY["max_seed_iterations"],
                    check_cancelled=check_cancelled))
                check_cancelled()
            except MujocoTcpPlanRejected as error:
                if (error.reason_code != "safe_tcp_planning_cancelled" or cancel_requested()
                        or time.perf_counter() < deadline):
                    raise
                break
            valid = (q.shape == start[side].shape and np.isfinite(q).all()
                     and np.all(q >= solver.q_min) and np.all(q <= solver.q_max))
            qualified = False
            if valid:
                actual = xyzquat_to_matrix(self.pose_for_configuration(
                    model_qpos, {**start, side: q}))
                position = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
                angle = float(np.arccos(np.clip(
                    (np.trace(actual[:3, :3].T @ goal[:3, :3])-1)/2, -1, 1)))
                qualified = position <= translation_tolerance_m and angle <= rotation_tolerance_rad
            estimates[index] = {
                "current_seed_ik_qualified": bool(qualified),
                "estimated_max_joint_excursion_rad": float(np.max(np.abs(q-start[side])))
                    if qualified else None,
                "estimated_total_joint_excursion_rad": float(np.sum(np.abs(q-start[side])))
                    if qualified else None,
                "wall_sec": time.perf_counter()-started,
                "collision_checked": False,
            }
        # Unknown/failed previews retain input order and remain eligible. Full
        # multi-seed IK and collision/path checks alone establish a valid plan.
        order = sorted(range(len(target_poses)), key=lambda index: (
            not estimates.get(index, {}).get("current_seed_ik_qualified", False),
            estimates.get(index, {}).get("estimated_total_joint_excursion_rad") or 0.0,
            index))
        return tuple(order), estimates


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
        contact_binding: MujocoGraspContactBinding | None = None,
        payload_binding: MujocoPayloadBinding | None = None,
        preview_model_qpos: np.ndarray | None = None,
        speed_scale: float = 1.0,
        observed_planning_input: ObservedJointPlanningInput | None = None,
        deadline_monotonic: float | None = None,
        incumbent_max_joint_excursion_rad: float | None = None,
        normalize_ik_objective: bool = False,
    ) -> MujocoTcpMotionPlan:
        """Plan without dispatch; a payload preview retains its binding's cut.

        An arbitrary preview qpos is supported only for ordinary TCP queries.
        Payload queries require their frozen complete source. A later native
        cut needs a new binding carrying the original tool-object transform,
        not a bypass of source validation or a new grasp anchor.
        """
        side = self.motion_side
        orientation_weight = _ik_orientation_weight(normalize_ik_objective,
            translation_tolerance_m, rotation_tolerance_rad)
        if deadline_monotonic is not None and (isinstance(deadline_monotonic, bool) or not math.isfinite(deadline_monotonic)):
            raise ValueError("Planning deadline must be finite.")
        if incumbent_max_joint_excursion_rad is not None and (
            isinstance(incumbent_max_joint_excursion_rad, bool)
            or not math.isfinite(incumbent_max_joint_excursion_rad)
            or incumbent_max_joint_excursion_rad < 0):
            raise ValueError("Incumbent joint excursion must be finite and non-negative.")
        if self._source_model_qpos is not None and observed_planning_input is not None:
            raise ValueError("A frozen native query cannot use a live observed-scene cut.")
        incoming_cancel_requested = cancel_requested
        if deadline_monotonic is not None:
            cancel_requested = lambda: incoming_cancel_requested() or time.perf_counter() >= deadline_monotonic

        def check_cancelled():
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")

        if observed_planning_input is not None:
            if not isinstance(observed_planning_input, ObservedJointPlanningInput):
                raise TypeError("observed_planning_input must be a measured observed-scene cut.")
            if any(value is not None for value in (contact_binding, payload_binding, preview_model_qpos)):
                raise ValueError("Observed ordinary TCP planning cannot use native contact/payload/preview state.")
        if payload_binding is not None:
            if not isinstance(payload_binding, MujocoPayloadBinding):
                raise TypeError("payload_binding must be a measured MujocoPayloadBinding.")
            if contact_binding is not None:
                raise ValueError("Terminal grasp contact and attached payload are distinct operation bindings.")
        if side == "left" and contact_binding is not None:
            # Existing contact bindings resolve right-hand sensors only.
            raise MujocoTcpPlanRejected("safe_tcp_left_grasp_contact_unsupported")
        if (
            isinstance(path_search_time_budget_sec, bool)
            or not math.isfinite(path_search_time_budget_sec)
            or path_search_time_budget_sec <= 0
        ):
            raise ValueError("path_search_time_budget_sec must be finite and positive.")
        if isinstance(speed_scale, bool) or not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("speed_scale must be finite and in (0, 1].")
        if payload_binding is not None:
            # Carried-object paths (live motions and their queries alike) run
            # with bounded joint speed and acceleration; see the constant.
            speed_scale = min(speed_scale, HELD_OBJECT_SPEED_SCALE)
        def capture(_scene):
            nonlocal configuration_constraint, constraint_identity
            with self._lock:
                if self._active is not None:
                    raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
                preview = preview_model_qpos is not None
                if not preview:
                    self._pending_plan = None
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                source_model_qpos = None
                if observed_planning_input is not None:
                    addresses = {
                        arm: np.asarray([self.scene.joint_qpos_addr(name)
                            for name in self.scene.arm_specs[arm].joint_names], dtype=int)
                        for arm in ("left", "right")
                    }
                    measured_pair = _scene.arm_qpos_by_side()
                    try:
                        observed_planning_input.require_current(measured_pair)
                    except ValueError as error:
                        raise MujocoTcpPlanRejected(str(error)) from error
                    # Retain the opaque native cut for original lifecycle
                    # diagnostics; only nominal scratch and measured arms enter FK.
                    source_model_qpos = self._model_qpos()
                    model_qpos = np.asarray(self.scene.model.qpos0, dtype=float).copy()
                    for arm, indices in addresses.items():
                        model_qpos[indices] = measured_pair[arm]
                elif payload_binding is not None:
                    # Search is a frozen query; activation and every dispatch
                    # separately validate the current measured relation/geometry.
                    model_qpos = np.asarray(
                        payload_binding.source_model_qpos if not preview else preview_model_qpos,
                        dtype=float,
                    ).copy()
                else:
                    model_qpos = (
                        self._model_qpos() if not preview
                        else np.asarray(preview_model_qpos, dtype=float).copy()
                    )
                if model_qpos.shape != (self.scene.model.nq,) or not np.isfinite(model_qpos).all():
                    raise ValueError("Planning requires a finite complete model qpos cut.")
                if contact_binding is not None:
                    if not isinstance(contact_binding, MujocoGraspContactBinding):
                        raise TypeError("contact_binding must be a bound grasp contact policy.")
                    target_matrix = xyzquat_to_matrix(target_pose)
                    outer_constraint = configuration_constraint
                    constraint_identity = canonical_content_hash(
                        {"envelope": constraint_identity, "contact": contact_binding.semantic_content_hash,
                         "target_pose": target_pose}, domain="mujoco.grasp_terminal_constraint.v1")

                    def configuration_constraint(model_qpos, pair):
                        if outer_constraint is not None:
                            check = outer_constraint(model_qpos, pair)
                            if not check.safe:
                                return check
                        pose = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, pair))
                        inside = contact_binding.in_corridor(pose, target_matrix)
                        return JointConfigurationCheck(
                            safe=inside, minimum_distance_m=None,
                            reason=None if inside else "grasp_contact_outside_terminal_corridor",
                            metadata=contact_binding.to_payload())
                # Keep an explicit planning reserve for the physical gripper's
                # small joint/servo motion. Live admission keeps the original
                # scene margin; no collision threshold is reduced or bypassed.
                if observed_planning_input is None:
                    oracle = MujocoJointSceneOracle(
                        self.scene,
                        model_qpos,
                        clearance_padding_m=self.planning_clearance_padding_m,
                        configuration_constraint=configuration_constraint,
                        constraint_identity=constraint_identity,
                        cancel_requested=cancel_requested,
                        contact_binding=contact_binding,
                        payload_binding=payload_binding,
                    )
                    addresses = oracle.addresses
                else:
                    oracle = _ObservedPlanningOracle(observed_planning_input, model_qpos,
                        constraint=configuration_constraint, constraint_identity=constraint_identity,
                        cancel_requested=cancel_requested)
                start = {
                    side: model_qpos[address].copy()
                    for side, address in addresses.items()
                }
                measured_start = start
                held_start = (self._held_command_start(start)
                              if payload_binding is not None and not preview else None)
                if held_start is not None:
                    start = held_start
                initial = oracle.check(start)
                egress = None
                if (not initial.safe and observed_planning_input is not None
                        and _observed_start_clearance(observed_planning_input)
                        == OBSERVED_START_CLEARANCE_INHERITED):
                    # Strict starts and ineligible inherited starts share the
                    # original reason; details record the admission decision.
                    admission = {"planning_phase": "inherited_start_clearance_admission",
                                 "start_clearance": OBSERVED_START_CLEARANCE_INHERITED,
                                 "inherited_start_clearance_admitted": False}
                    inherited = oracle.with_inherited_start(start, side, admission)
                    if inherited is not None:
                        admission.update(inherited_start_clearance_admitted=True,
                                         baseline_check=initial.to_dict())
                        oracle, egress = inherited, _InheritedStartEgress(inherited.query)
                        initial = oracle.check(start)
                    if not initial.safe:
                        raise MujocoTcpPlanRejected("safe_tcp_start_blocked",
                                                    {**initial.to_dict(), **admission})
                if not initial.safe:
                    raise MujocoTcpPlanRejected("safe_tcp_start_blocked", initial.to_dict())
                solver = self._solver_for_ik(orientation_weight)
                offset = endpoint_to_tool_transform(
                    load_rm65_tool_profile(
                        getattr(self.task_config, side + "_tool_profile_path"), side=side
                    )
                )
                measured = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, measured_start))
                if payload_binding is not None:
                    self.scene.mujoco.mj_kinematics(self.scene.model, oracle.source)
                    world_tcp = payload_binding._tcp(oracle.source)
                    if not np.allclose(
                        world_tcp, self.scene.body_pose(side + "_arm_base_link") @ measured,
                        atol=1e-5, rtol=0,
                    ):
                        raise MujocoTcpPlanRejected("safe_tcp_payload_tool_frame_mismatch")
                if not np.allclose(
                    solver.compute_pose(measured_start[side]) @ offset, measured, atol=1e-5
                ):
                    raise MujocoTcpPlanRejected("safe_tcp_ik_model_tool_mismatch")
                goal = xyzquat_to_matrix(target_pose)
                endpoint_goal = goal @ np.linalg.inv(offset)
                rng = np.random.default_rng(0)
                seeds = [start[side], np.clip(np.zeros(6), solver.q_min, solver.q_max)]
                seeds.extend(rng.uniform(solver.q_min, solver.q_max) for _ in range(6))
                targets = []
                endpoint_attempts = []
                endpoint_check_reasons = []
                endpoint_checks = []
                ik_seed_attempts = []
                best_pose_miss = None
                quality_limits = DEFAULT_PATH_MOTION_QUALITY_LIMITS
                endpoint_started_at = time.perf_counter()
                deadline = endpoint_started_at + 2.0
                if deadline_monotonic is not None:
                    deadline = min(deadline, deadline_monotonic)

                def consider_endpoint(q, seed_index, search_method):
                    nonlocal best_pose_miss
                    pair = {**start, side: np.asarray(q)}
                    actual = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, pair))
                    position_error = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
                    angle = float(np.arccos(np.clip(
                        (np.trace(actual[:3, :3].T @ goal[:3, :3]) - 1) / 2, -1, 1)))
                    within = position_error <= translation_tolerance_m and angle <= rotation_tolerance_rad
                    ik_seed_attempts.append({
                        "seed_index": seed_index, "search_method": search_method,
                        "position_error_m": position_error, "rotation_error_rad": angle,
                        "pose_within_tolerance": within,
                        "iterations": getattr(solver, "last_solve_iterations", None),
                        "solver_converged": getattr(solver, "last_solve_converged", None),
                        "solver_deadline_exceeded": getattr(solver, "last_solve_deadline_exceeded", None),
                        "residual_evaluations": (getattr(solver, "last_solve_residual_evaluations", None)
                                                 if search_method == "bounded_least_squares" else None),
                    })
                    if not within:
                        score = max(position_error / translation_tolerance_m, angle / rotation_tolerance_rad)
                        if best_pose_miss is None or score < best_pose_miss[0]:
                            best_pose_miss = (score, pair[side].copy(), seed_index)
                        return False
                    delta = np.abs(pair[side] - start[side])
                    compact = float(np.max(delta[:3])) <= quality_limits.maximum_proximal_joint_travel_rad
                    endpoint_check = oracle.check(pair)
                    endpoint_check_reasons.append(endpoint_check.reason)
                    endpoint_checks.append(endpoint_check)
                    safe = endpoint_check.safe
                    endpoint_attempts.append({
                        "seed_index": seed_index,
                        **({"search_method": search_method} if search_method != "iterative" else {}),
                        "joint_net_delta_deg": np.rad2deg(pair[side] - start[side]).tolist(),
                        "position_error_m": position_error, "rotation_error_rad": angle,
                        "collision_and_constraint_safe": safe,
                        "proximal_travel_within_limit": compact,
                        **({"collision_check": endpoint_check.to_dict()}
                           if observed_planning_input is not None else {}),
                    })
                    if safe and compact and not any(
                        np.max(np.abs(other[side] - pair[side])) < 0.001 for other in targets
                    ):
                        targets.append(pair)
                    return safe and float(np.max(delta)) < 1e-8

                for seed_index, seed in enumerate(seeds):
                    if cancel_requested():
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                    if time.perf_counter() >= deadline:
                        break
                    if normalize_ik_objective or self._source_model_qpos is not None:
                        q = _solve_endpoint_seed(solver, goal, offset, seed,
                            normalized=normalize_ik_objective, deadline=deadline,
                            check_cancelled=check_cancelled)
                    else:
                        q = solver.solve_iterative(
                            target_pose=endpoint_goal, q_init=seed, max_iterations=100,
                            max_joint_step=0.2, damping=0.01, tolerance=1e-5,
                            deadline_monotonic=deadline,
                        )
                    # Zero travel is the only provable minimum; a converged
                    # current-state seed is not generally the closest branch.
                    if consider_endpoint(q, seed_index, "iterative"):
                        break
                if not endpoint_attempts and best_pose_miss is not None and time.perf_counter() < deadline:
                    from robot_motion.rm65.single_arm_ik import BoundedIKNumericalError

                    # Only a complete pose miss may use the remaining endpoint
                    # budget. A collision/quality refusal is not a solver failure.
                    try:
                        q = solver.solve_bounded_pose(
                            goal, best_pose_miss[1], controlled_frame_offset=offset,
                            translation_tolerance_m=translation_tolerance_m,
                            rotation_tolerance_rad=rotation_tolerance_rad,
                            deadline_monotonic=deadline, cancel_requested=cancel_requested,
                            maximum_evaluations=100,
                        )
                    except InterruptedError as error:
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled") from error
                    except BoundedIKNumericalError as error:
                        raise MujocoTcpPlanRejected("safe_tcp_endpoint_numerical_failure",
                            {"seed_index": best_pose_miss[2], "search_method": "bounded_least_squares",
                             "endpoint_attempts": endpoint_attempts,
                             "ik_seed_attempts": ik_seed_attempts}) from error
                    consider_endpoint(q, best_pose_miss[2], "bounded_least_squares")
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                endpoint_search_diagnostics = {
                    "seed_count": len(seeds), "attempted_seed_count": len(ik_seed_attempts),
                    "elapsed_sec": time.perf_counter() - endpoint_started_at, "budget_sec": 2.0,
                    "deadline_exceeded": time.perf_counter() >= deadline,
                    "translation_tolerance_m": translation_tolerance_m,
                    "rotation_tolerance_rad": rotation_tolerance_rad,
                    "attempts": ik_seed_attempts,
                }
                if not targets:
                    quality_rejected = any(
                        item["collision_and_constraint_safe"] and not item["proximal_travel_within_limit"]
                        for item in endpoint_attempts
                    )
                    raise MujocoTcpPlanRejected(
                        "safe_tcp_motion_quality_exceeded" if quality_rejected
                        else "safe_tcp_target_envelope_blocked"
                        if endpoint_check_reasons and all(
                            reason == "tcp_envelope_blocked" for reason in endpoint_check_reasons
                        )
                        else "safe_tcp_target_blocked" if endpoint_attempts
                        else "safe_tcp_endpoint_search_exhausted",
                        {"endpoint_attempts": endpoint_attempts,
                         "endpoint_check_reasons": endpoint_check_reasons,
                         # Rejection-only evidence: do not change successful
                         # path metadata, hashes, or perform another query.
                         "endpoint_checks": [check.to_dict() for check in endpoint_checks],
                         "endpoint_search_diagnostics": endpoint_search_diagnostics,
                         "motion_quality_limits": quality_limits.to_dict()},
                    )
                if incumbent_max_joint_excursion_rad is not None:
                    lower_bound = min(float(np.max(np.abs(target[side] - start[side]))) for target in targets)
                    rank_margin = 1e-9 * max(1.0, lower_bound, incumbent_max_joint_excursion_rad)
                    if lower_bound > incumbent_max_joint_excursion_rad + rank_margin:
                        check_cancelled()
                        raise MujocoTcpPlanRejected("safe_tcp_candidate_rank_pruned", {
                            "lower_bound_max_joint_excursion_rad": lower_bound,
                            "incumbent_max_joint_excursion_rad": incumbent_max_joint_excursion_rad,
                            "rank_margin_rad": rank_margin, "path_search_skipped": True})
                velocity = np.minimum(
                    np.asarray(self.rig.config.max_velocity_rad_s),
                    np.deg2rad(self.rig.config.max_command_step_deg) * self.rig.config.timing.command_hz,
                )
                targets.sort(key=lambda pair: (
                    float(np.max(np.abs(pair[side] - start[side]) / velocity)),
                    float(np.max(np.abs(pair[side][:3] - start[side][:3]))),
                    float(np.linalg.norm(pair[side] - start[side])),
                ))
                world_from_base = self.scene.body_pose(side + "_arm_base_link")

                def world_pose(pair):
                    return world_from_base @ xyzquat_to_matrix(self.pose_for_configuration(model_qpos, pair))

                search_deadline = time.perf_counter() + path_search_time_budget_sec
                if deadline_monotonic is not None:
                    search_deadline = min(search_deadline, deadline_monotonic)
                rejected_paths = []
                while targets:
                    remaining = search_deadline - time.perf_counter()
                    if remaining <= 0:
                        raise MujocoTcpPlanRejected("safe_tcp_path_search_exhausted", {"rejected_paths": rejected_paths})
                    path = plan_collision_aware_joint_path(
                        start_qpos_by_component=start,
                        target_qpos_by_component=targets[0],
                        alternative_target_qpos_by_component=targets[1:],
                        movable_components=(side,), oracle=oracle,
                        config=JointPathPlannerConfig(
                            time_budget_sec=remaining, edge_resolution_rad=0.01,
                            adaptive_edge_resolution_rad=0.002, adaptive_refinement_distance_m=0.01,
                        ),
                        joint_lower_by_component={side: solver.q_min},
                        joint_upper_by_component={side: solver.q_max}, scene_hash=oracle.scene_hash,
                        edge_cost=lambda a, b: tcp_path_edge_cost(a, b, side=side,
                            pose_for_pair=world_pose, step_rad=0.1),
                    )
                    if not path.path_found:
                        break
                    samples = [path.waypoints[0][side]]
                    for a, b in zip(path.waypoints, path.waypoints[1:]):
                        count = max(1, math.ceil(float(np.max(np.abs(b[side] - a[side]))) / 0.01))
                        samples.extend(a[side] + alpha * (b[side] - a[side])
                            for alpha in np.linspace(0, 1, count + 1)[1:])
                    xyz = []
                    for sample in samples:
                        if cancel_requested():
                            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                        xyz.append(world_pose({**start, side: sample})[:3, 3])
                    quality = sampled_path_motion_quality(np.asarray(samples), np.asarray(xyz))
                    if not quality_limits.exceeded(quality):
                        break
                    rejected_paths.append({"motion_quality": quality, "path_hash": path.semantic_content_hash})
                    targets = [item for item in targets if not np.array_equal(
                        item[side], path.target_qpos_by_component[side])]
                else:
                    raise MujocoTcpPlanRejected("safe_tcp_motion_quality_exceeded",
                        {"rejected_paths": rejected_paths, "motion_quality_limits": quality_limits.to_dict()})
                if cancel_requested():
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if not path.path_found:
                    raise MujocoTcpPlanRejected(
                        f"safe_tcp_{path.status.value}", path.to_dict()
                    )
                q = np.asarray([point[side] for point in path.waypoints])
                if q.shape[0] == 1:
                    q = np.repeat(q, 2, axis=0)
                velocity = np.minimum(
                    np.asarray(self.rig.config.max_velocity_rad_s),
                    np.deg2rad(self.rig.config.max_command_step_deg)
                    * self.rig.config.timing.command_hz,
                )
                velocity = velocity * speed_scale
                acceleration = np.asarray(self.rig.config.max_acceleration_rad_s2) * speed_scale**2
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    intervals = np.maximum(
                        0.2, np.max(np.abs(np.diff(q, axis=0)) / velocity, axis=1)
                    )
                source_duration = float(np.sum(intervals))
                _require_bounded_grid(source_duration, self.reference_hz)
                limits = JointHorizonLimits(
                    min_position_rad=solver.q_min,
                    max_position_rad=solver.q_max,
                    max_velocity_rad_s=velocity,
                    max_acceleration_rad_s2=acceleration,
                    max_jerk_rad_s3=acceleration * speed_scale / self.rig.config.velocity_ramp_sec,
                )
                retimed = build_offline_quintic_waypoint_trajectory(
                    np.concatenate(([0.0], np.cumsum(intervals))),
                    q,
                    command_hz=self.reference_hz,
                    limits=limits,
                    timing_mode="safe_retime_global",
                    max_time_scale=min(100.0, _trajectory_duration_limit(self.reference_hz) / source_duration),
                    stop_at_waypoints=True,
                    **({"check_cancelled": check_cancelled} if deadline_monotonic is not None else {}),
                )
                # Keep RRT corners on the actual reference clock. Otherwise an
                # executor could connect the samples on opposite sides of a knot
                # and cut the verified corner even with zero knot velocities.
                ticks = np.ceil(np.diff(retimed.waypoint_timestamps) * self.reference_hz)
                _require_bounded_grid(float(np.sum(ticks)) / self.reference_hz, self.reference_hz)
                trajectory = build_offline_quintic_waypoint_trajectory(
                    np.concatenate(([0.0], np.cumsum(ticks) / self.reference_hz)),
                    q,
                    command_hz=self.reference_hz,
                    limits=limits,
                    stop_at_waypoints=True,
                    **({"check_cancelled": check_cancelled} if deadline_monotonic is not None else {}),
                )
                poses = []
                for index, current in enumerate(trajectory.q_rad):
                    if cancel_requested():
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                    pair = {**start, side: current}
                    # Smoothing may leave the polyline. Recheck each reference and
                    # every intervening joint-space edge, not only the waypoints.
                    previous = trajectory.q_rad[max(0, index - 1)]
                    count = max(
                        1, math.ceil(float(np.max(np.abs(current - previous))) / 0.01)
                    )
                    for alpha in np.linspace(0, 1, count + 1):
                        sample = {**pair, side: previous + alpha * (current - previous)}
                        check = oracle.check(sample)
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
                        if egress is not None:
                            # Inherited near pairs must move out monotonically
                            # along these exact ordered executed samples.
                            egress.advance(sample, {"reference_index": index,
                                                    "edge_alpha": float(alpha)})
                    poses.append(
                        tuple(
                            float(item)
                            for item in self.pose_for_configuration(model_qpos, pair)
                        )
                    )
                if egress is not None:
                    egress.require_exited()
                quality = sampled_path_motion_quality(trajectory.q_rad, np.asarray([
                    (world_from_base @ xyzquat_to_matrix(pose))[:3, 3] for pose in poses
                ]))
                if quality_limits.exceeded(quality):
                    raise MujocoTcpPlanRejected("safe_tcp_motion_quality_exceeded", {"motion_quality": quality})
                path = replace(path, metadata={**dict(path.metadata), "motion_quality": quality,
                    "motion_quality_limits": quality_limits.to_dict(), "endpoint_attempts": endpoint_attempts,
                    "quality_rejected_paths": rejected_paths,
                    **({"planning_scene_source": "calibrated_rgbd_and_measured_joints",
                        "observed_scene_hash": observed_planning_input.oracle.scene_hash}
                       if observed_planning_input is not None else {}),
                    **({"inherited_start_clearance": egress.summary()} if egress is not None else {}),
                    # Only a held-command start adds metadata; measured-start
                    # plans keep their previous path identity.
                    **({"start_source": "held_command", "held_command_minus_measured_rad": {
                        arm: (held_start[arm] - measured_start[arm]).tolist() for arm in held_start}}
                       if held_start is not None else {})})
                identity = canonical_content_hash(
                    {
                        "path": path.semantic_content_hash,
                        "timestamps": trajectory.timestamps.tolist(),
                        "q": trajectory.q_rad.tolist(),
                    },
                    domain="dexteleop.mujoco.safe_tcp_joint_trajectory.v1",
                )
                model_qpos.setflags(write=False)
                retained_source = model_qpos if source_model_qpos is None else source_model_qpos
                retained_source.setflags(write=False)
                if observed_planning_input is not None:
                    try:
                        observed_planning_input.require_current(_scene.arm_qpos_by_side())
                    except ValueError as error:
                        raise MujocoTcpPlanRejected(str(error)) from error
                    if cancel_requested():
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                plan = MujocoTcpMotionPlan(
                    path,
                    trajectory,
                    retained_source,
                    tuple(poses),
                    identity,
                    configuration_constraint,
                    constraint_identity,
                    contact_binding,
                    payload_binding,
                    endpoint_search_diagnostics=MappingProxyType({
                        **endpoint_search_diagnostics,
                        "attempts": tuple(MappingProxyType(row) for row in ik_seed_attempts),
                    }),
                    observed_planning_input=observed_planning_input,
                    measured_start_qpos_by_component=(
                        None if held_start is None else MappingProxyType(
                            {arm: _frozen_copy(value) for arm, value in measured_start.items()})),
                )
                check_cancelled()
                if not preview:
                    self._pending_plan = plan
                return plan


        return self.rig.capture_scene(capture)

    def plan_candidates(
        self,
        target_poses: Sequence[tuple[float, ...]],
        *,
        translation_tolerance_m: float,
        rotation_tolerance_rad: float,
        cancel_requested: Callable[[], bool],
        time_budget_sec: float = 10.0,
        path_search_time_budget_sec: float = 3.0,
        max_reference_count: int | None = None,
        configuration_constraint: ConfigurationConstraint | None = None,
        constraint_identity: str | None = None,
        normalize_ik_objective: bool = False,
        prioritize_current_seed: bool = False,
        payload_binding: MujocoPayloadBinding | None = None,
    ) -> MujocoTcpCandidateSelection:
        """Rank sampled candidates without motion, retaining the selected owned plan.

        Prefer smaller worst-joint excursion, then certified execution duration,
        then total joint travel. This bounded sampled comparison is not a global
        optimum. A conservative endpoint bound may skip path search only after
        a qualified incumbent exists. Such candidates count toward diagnostic
        coverage in all_candidates_checked, without certifying a complete path.
        Unpruned candidates use the same existing IK/path/retiming chain.
        Frozen queries can opt into a bounded, current-seed IK scheduling hint.
        This hint changes order only, never feasibility or the ranking criterion.
        """
        started_at = time.perf_counter()
        if not target_poses or len(target_poses) > 64:
            raise ValueError("Candidate planning requires 1..64 explicit poses.")
        if not isinstance(prioritize_current_seed, bool):
            raise TypeError("prioritize_current_seed must be a bool.")
        if payload_binding is not None and not isinstance(payload_binding, MujocoPayloadBinding):
            raise TypeError("payload_binding must be a measured MujocoPayloadBinding.")
        if payload_binding is not None and self._source_model_qpos is not None:
            raise ValueError("Payload candidate planning requires the live owned planner.")
        if payload_binding is not None and prioritize_current_seed:
            raise ValueError("Payload candidate scheduling requires full guarded plans.")
        if max_reference_count is not None and (type(max_reference_count) is not int or max_reference_count <= 0):
            raise ValueError("Reference count bound must be a positive integer.")
        if isinstance(time_budget_sec, bool) or not math.isfinite(time_budget_sec) or time_budget_sec <= 0:
            raise ValueError("Candidate planning needs a positive finite time budget.")
        if (isinstance(path_search_time_budget_sec, bool) or not math.isfinite(path_search_time_budget_sec)
                or path_search_time_budget_sec <= 0):
            raise ValueError("Each candidate needs a positive finite path search budget.")
        # Both scene and planner lock waits consume the original query budget.
        deadline = started_at + time_budget_sec
        def capture(_scene):
            with self._lock:
                if self._active is not None:
                    raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
                selection = None
                diagnostics = []
                previews = {}
                try:
                    best = None
                    self._pending_plan = None

                    def cancelled_or_expired():
                        return cancel_requested() or time.perf_counter() >= deadline
                    order, previews = tuple(range(len(target_poses))), {}
                    if prioritize_current_seed and len(target_poses) > 1:
                        order, previews = self._preview_candidate_order(
                            target_poses, translation_tolerance_m=translation_tolerance_m,
                            rotation_tolerance_rad=rotation_tolerance_rad,
                            normalize_ik_objective=normalize_ik_objective,
                            deadline=min(deadline, time.perf_counter()+min(
                                CANDIDATE_SCHEDULING_POLICY["preview_budget_cap_sec"],
                                time_budget_sec*CANDIDATE_SCHEDULING_POLICY["preview_budget_fraction"])),
                            cancel_requested=cancel_requested)
                    for index in order:
                        target = target_poses[index]
                        if cancel_requested():
                            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        started = time.perf_counter()
                        try:
                            rank_bound = {} if best is None else {
                                "incumbent_max_joint_excursion_rad": best[0][0]
                            }
                            plan = self.plan(target,
                                translation_tolerance_m=translation_tolerance_m,
                                rotation_tolerance_rad=rotation_tolerance_rad,
                                normalize_ik_objective=normalize_ik_objective,
                                cancel_requested=cancelled_or_expired,
                                configuration_constraint=configuration_constraint,
                                constraint_identity=constraint_identity,
                                payload_binding=payload_binding,
                                deadline_monotonic=deadline,
                                path_search_time_budget_sec=min(path_search_time_budget_sec, remaining),
                                **rank_bound)
                        except MujocoTcpPlanRejected as error:
                            if error.reason_code == "safe_tcp_planning_cancelled":
                                if cancel_requested():
                                    raise
                                break
                            if error.reason_code not in {
                            "safe_tcp_endpoint_search_exhausted", "safe_tcp_target_blocked",
                            "safe_tcp_target_envelope_blocked",
                                "safe_tcp_path_search_exhausted", "safe_tcp_retimed_path_blocked",
                                "safe_tcp_candidate_rank_pruned",
                                "safe_tcp_motion_quality_exceeded",
                            }:
                                raise
                            diagnostics.append({"candidate_index": index, "reason_code": error.reason_code,
                                                "details": error.details, "wall_sec": time.perf_counter()-started})
                            continue
                        if max_reference_count is not None and len(plan.reference_poses) >= max_reference_count:
                            diagnostics.append({"candidate_index": index, "reason_code": "safe_tcp_path_exceeds_control_budget",
                                "wall_sec": time.perf_counter()-started})
                            continue
                        q = plan.trajectory.q_rad
                        key = (float(np.max(np.abs(q-q[0]))),
                               float(plan.trajectory.timestamps[-1]-plan.trajectory.timestamps[0]),
                               float(np.sum(np.abs(np.diff(q, axis=0)))))
                        diagnostics.append({"candidate_index": index, "reason_code": "path_found",
                            "ranking_cost": key, "trajectory_hash": plan.trajectory_hash,
                            "phase_timing_sec": getattr(plan, "phase_timing_sec", None),
                            "wall_sec": time.perf_counter()-started})
                        if best is None or key < best[0]:
                            best = (key, index, plan)
                    if best is None:
                        raise MujocoTcpPlanRejected("safe_tcp_no_candidate_path",
                            {"candidates": diagnostics, "all_candidates_checked": len(diagnostics) == len(target_poses)})
                    if cancel_requested():
                        raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                    selection = MujocoTcpCandidateSelection(best[1], best[2], tuple(diagnostics),
                                                           len(diagnostics) == len(target_poses))
                    return selection
                finally:
                    for item in diagnostics:
                        if item["candidate_index"] in previews:
                            item["scheduling_preview"] = previews[item["candidate_index"]]
                    self._pending_plan = None if selection is None else selection.plan


        return self.rig.capture_scene(capture)


    @_scene_serialized(total_budget=True)
    def plan_chain(
        self,
        goals: tuple[MujocoTcpPlanGoal, ...],
        *,
        cancel_requested: Callable[[], bool],
        time_budget_sec: float,
        preview_model_qpos: np.ndarray | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> MujocoTcpMotionPlan:
        """Plan all approach stages before any execution, with one total budget.

        Preview remains unowned and cannot replace a pending plan. A normal
        query retains only the complete successful chain; rejection, timeout,
        scene drift or cancellation cannot leave a partial executable prefix.
        """
        goals = tuple(goals)
        if not 1 <= len(goals) <= 8 or any(
            not isinstance(g, MujocoTcpPlanGoal) for g in goals
        ):
            raise ValueError("A motion chain requires one to eight typed goals.")
        if (
            isinstance(time_budget_sec, bool)
            or not math.isfinite(time_budget_sec)
            or time_budget_sec <= 0
        ):
            raise ValueError("time_budget_sec must be finite and positive.")

        def now():
            value = float(monotonic())
            if not math.isfinite(value):
                raise ValueError("Motion chain planning clock must be finite.")
            return value

        deadline = now() + time_budget_sec

        def stopped():
            return cancel_requested() or now() >= deadline

        def check_request():
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if now() >= deadline:
                raise MujocoTcpPlanRejected("safe_tcp_planning_budget_exhausted")

        with self._lock:
            if self._active is not None:
                raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
            preview = preview_model_qpos is not None
            if not preview:
                self._pending_plan = None
            check_request()
            source = (
                self._model_qpos()
                if not preview
                else np.asarray(preview_model_qpos, dtype=float).copy()
            )
            if source.shape != (self.scene.model.nq,) or not np.isfinite(source).all():
                raise ValueError("Planning requires a finite complete model qpos cut.")
            cut = source.copy()
            addresses = {
                side: [
                    self.scene.joint_qpos_addr(n)
                    for n in self.scene.arm_specs[side].joint_names
                ]
                for side in ("left", "right")
            }
            plans = []
            for index, goal in enumerate(goals):
                check_request()
                try:
                    stage = self.plan(
                        goal.target_pose,
                        translation_tolerance_m=goal.translation_tolerance_m,
                        rotation_tolerance_rad=goal.rotation_tolerance_rad,
                        cancel_requested=stopped,
                        configuration_constraint=goal.configuration_constraint,
                        constraint_identity=goal.constraint_identity,
                        path_search_time_budget_sec=max(0.001, deadline - now()),
                        contact_binding=goal.contact_binding,
                        preview_model_qpos=cut,
                        speed_scale=goal.speed_scale,
                    )
                except MujocoTcpPlanRejected as error:
                    check_request()
                    raise MujocoTcpPlanRejected(
                        error.reason_code,
                        {
                            **error.details,
                            "chain_stage_index": index,
                            "completed_stage_count": len(plans),
                        },
                    ) from error
                plans.append(stage)
                _require_bounded_grid(sum(float(p.trajectory.timestamps[-1]) for p in plans), self.reference_hz)
                for side, address in addresses.items():
                    cut[address] = stage.path.target_qpos_by_component[side]
            check_request()
            combined = _join_motion_plans(tuple(plans))
            world_from_base = self.scene.body_pose(self.motion_side + "_arm_base_link")
            quality = sampled_path_motion_quality(
                combined.trajectory.q_rad,
                np.asarray(
                    [
                        (world_from_base @ xyzquat_to_matrix(p))[:3, 3]
                        for p in combined.reference_poses
                    ]
                ),
            )
            if DEFAULT_PATH_MOTION_QUALITY_LIMITS.exceeded(quality):
                raise MujocoTcpPlanRejected(
                    "safe_tcp_motion_quality_exceeded",
                    {
                        "motion_quality": quality,
                        "scope": "complete_approach_chain",
                    },
                )
            combined = replace(
                combined,
                path=replace(
                    combined.path,
                    metadata={
                        **dict(combined.path.metadata),
                        "motion_quality": quality,
                        "motion_quality_limits": DEFAULT_PATH_MOTION_QUALITY_LIMITS.to_dict(),
                    },
                ),
            )
            check_request()
            if not preview:
                if not np.array_equal(source, self._model_qpos()):
                    raise MujocoTcpPlanRejected(
                        "safe_tcp_scene_changed_during_planning"
                    )
                check_request()
                self._pending_plan = combined
            return combined


    @_scene_serialized(total_budget=True)
    def refine_pending_chain(
        self, original: MujocoTcpMotionPlan, goals: tuple[MujocoTcpPlanGoal, ...], *,
        cancel_requested: Callable[[], bool], time_budget_sec: float,
        validate_plan: Callable[[MujocoTcpMotionPlan], None], deadline_at: float,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> MujocoTcpMotionPlan:
        """Atomically replace an unconsumed chain only after full refinement.

        Preview uses the same lower planner without clearing the valid handle.
        Rejection leaves that exact handle intact; cancellation and scene drift
        remain errors and never authorize consuming either plan.
        """
        if (isinstance(time_budget_sec, bool) or not math.isfinite(time_budget_sec)
                or time_budget_sec <= 0 or not math.isfinite(deadline_at)
                or not callable(validate_plan)):
            raise ValueError("Refinement requires a finite budget, deadline and validation port.")
        started = float(monotonic())
        if not math.isfinite(started):
            raise ValueError("Refinement clock must be finite.")
        # The decorator already deducted rig-lock wait; this slice must also
        # include planner-lock wait and the final measured-scene read.
        deadline = min(started + time_budget_sec, deadline_at)

        def check_request():
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            timestamp = float(monotonic())
            if not math.isfinite(timestamp):
                raise ValueError("Refinement clock must be finite.")
            if timestamp >= deadline:
                raise MujocoTcpPlanRejected("safe_tcp_planning_budget_exhausted")
            return timestamp

        with self._lock:
            if self._active is not None or self._pending_plan is not original:
                raise MujocoTcpPlanRejected("safe_tcp_plan_not_owned_by_runtime")
            if original.observed_planning_input is not None:
                raise MujocoTcpPlanRejected("safe_tcp_observed_chain_refinement_unsupported")
            if not np.array_equal(original.source_model_qpos, self._model_qpos()):
                raise MujocoTcpPlanRejected("safe_tcp_scene_changed_during_planning")
            checked_at = check_request()
            refined = self.plan_chain(goals, cancel_requested=cancel_requested,
                time_budget_sec=deadline - checked_at, preview_model_qpos=original.source_model_qpos,
                monotonic=monotonic)
            validate_plan(refined)
            check_request()
            if (self._pending_plan is not original or self._active is not None
                    or not np.array_equal(original.source_model_qpos, self._model_qpos())):
                raise MujocoTcpPlanRejected("safe_tcp_scene_changed_during_planning")
            check_request()
            self._pending_plan = refined
            return refined


    @staticmethod
    def _stage_at(plan: MujocoTcpMotionPlan, index: int) -> MujocoTcpMotionPlan:
        for start, stage in reversed(plan.stages):
            # A stopped join belongs to the incoming stage. The next stage's
            # policy takes effect only when its first outgoing sample is used.
            if index > start or start == 0:
                return stage
        return plan


    @_scene_serialized()
    def owns_pending_plan(self, plan: MujocoTcpMotionPlan) -> bool:
        """Inspect ownership without registering a preview or activating motion."""
        with self._lock:
            return self._active is None and self._pending_plan is plan


    @_scene_serialized()
    def discard_pending_plan(self, plan: MujocoTcpMotionPlan) -> None:
        """Withdraw only this unpublished handle, never HOLD or clear active motion."""
        with self._lock:
            if self._pending_plan is plan:
                self._pending_plan = None


    def _oracle_for_stage(self, stage: MujocoTcpMotionPlan) -> MujocoJointSceneOracle:
        return MujocoJointSceneOracle(
            self.scene,
            stage.source_model_qpos if stage.payload_binding is not None else self._model_qpos(),
            configuration_constraint=stage.configuration_constraint,
            constraint_identity=stage.constraint_identity,
            contact_binding=stage.contact_binding,
            payload_binding=stage.payload_binding,
        )

    def _held_command_start(self, measured_start):
        """The rig's held joint command for both arms, if it is near the measured cut."""
        accepted = getattr(self.rig, "accepted_qcmd_by_side", None)
        if not callable(accepted):
            return None   # private query rigs keep the measured start
        command = accepted()
        if set(command) != set(measured_start):
            return None
        held = {}
        for arm, measured in measured_start.items():
            value = np.asarray(command[arm], dtype=float)
            if (value.shape != measured.shape or not np.isfinite(value).all()
                    or float(np.max(np.abs(value - measured))) > HELD_COMMAND_START_TOLERANCE_RAD):
                return None
            held[arm] = value.copy()
        return held

    def _require_held_command_start(self, plan) -> None:
        """A held-command start is valid only while the rig still holds that command."""
        if getattr(plan, "measured_start_qpos_by_component", None) is None:
            return
        command = self.rig.accepted_qcmd_by_side()
        if any(not np.array_equal(np.asarray(command[arm], dtype=float),
                                  np.asarray(plan.path.start_qpos_by_component[arm], dtype=float))
               for arm in ("left", "right")):
            raise MujocoTcpPlanRejected("safe_tcp_start_state_changed")

    def _bind_live_stage(self, stage, oracle, *, bind_noncontact=True):
        if stage.payload_binding is not None:
            def bind_payload(scene):
                pair = scene.arm_qpos_by_side()
                measured = _measured_start(stage)
                if any(not np.allclose(pair[side], measured[side], atol=1e-3, rtol=0)
                       for side in ("left", "right")):
                    raise MujocoTcpPlanRejected("safe_tcp_start_state_changed")
                self._require_held_command_start(stage)
                self._capture_current_payload(stage, oracle, scene)
                self.rig.set_additional_contact_oracle(stage.payload_binding.contact_oracle(scene))

            self.rig.capture_scene(bind_payload)
            self._contact_guard_bound = True
        elif stage.contact_binding is not None or bind_noncontact:
            # Noncontact stages retain their stricter hand/object policy too.
            self.rig.set_additional_contact_oracle(oracle.projector.oracle)
            self._contact_guard_bound = True
        self._live_stage = stage
        self._live_oracle = oracle

    @_scene_serialized()
    def activate(self, plan: MujocoTcpMotionPlan, *, deadline_at: float | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if self._source_model_qpos is not None:
            raise MujocoTcpPlanRejected("safe_tcp_query_only_plan")
        with self._lock:
            if deadline_at is not None and (
                isinstance(deadline_at, bool) or not math.isfinite(deadline_at) or not callable(clock)
            ):
                raise ValueError("Native activation deadline requires a finite same-domain clock binding.")
            if self._active is not None:
                raise MujocoTcpPlanRejected("safe_tcp_another_plan_active")
            if plan is not self._pending_plan:
                raise MujocoTcpPlanRejected("safe_tcp_plan_not_owned_by_runtime")
            live = self.rig.capture_scene(lambda scene: scene.arm_qpos_by_side())
            if plan.observed_planning_input is not None:
                try:
                    plan.observed_planning_input.require_current(live)
                except ValueError as error:
                    raise MujocoTcpPlanRejected(str(error)) from error
            measured = _measured_start(plan)
            if any(
                not np.allclose(
                    live[side],
                    measured[side],
                    atol=1e-3,
                    rtol=0,
                )
                for side in ("left", "right")
            ):
                raise MujocoTcpPlanRejected("safe_tcp_start_state_changed")
            self._require_held_command_start(plan)
            stage = self._stage_at(plan, 0)
            oracle = self._oracle_for_stage(stage)
            # Validate at the consumption boundary, after scene/planner lock
            # wait and oracle construction. No caller/app callback may acquire
            # another owner here; the injected clock is a read-only time port.
            if plan.observed_planning_input is not None:
                try:
                    plan.observed_planning_input.require_current(self.scene.arm_qpos_by_side())
                except ValueError as error:
                    raise MujocoTcpPlanRejected(str(error)) from error
            if deadline_at is not None:
                timestamp = float(clock())
                if not math.isfinite(timestamp) or timestamp >= deadline_at:
                    raise MujocoTcpPlanRejected("grasp_path_plan_expired")
            self._bind_live_stage(stage, oracle, bind_noncontact=bool(plan.stages))
            self._active = plan
            self._pending_plan = None
            self._reference_index = -1
            self._last_cycle = -1

    @_scene_serialized()
    def clear(self) -> None:
        with self._lock:
            if self._contact_guard_bound:
                self.rig.set_additional_contact_oracle(None)
                self._contact_guard_bound = False
            self._active = None
            self._pending_plan = None
            self._reference_index = -1
            self._last_cycle = -1
            self._live_oracle = None

            self._live_stage = None
    def _capture_current_payload(self, plan, oracle, scene) -> None:
        """Caller holds the rig scene lock through the following live sweep."""
        check = plan.payload_binding.check_measured(scene)
        if not check.safe:
            raise MujocoTcpPlanRejected("safe_tcp_payload_measured_blocked", check.to_dict())
        # qpos alone omits time/velocity and mocap/equality geometry inputs.
        # Never regenerate the frozen hand-object relation from this cut.
        scene.mujoco.mj_copyData(oracle.source, scene.model, scene.data)

    def check_measured_contact(self, binding, target_pose) -> JointConfigurationCheck:
        """Check right-hand geometry, including after HOLD cleared the plan."""
        if self.motion_side == "left":
            raise MujocoTcpPlanRejected("safe_tcp_left_grasp_contact_unsupported")
        def capture(_scene):
            with self._lock:
                model_qpos = self._model_qpos()
                oracle = MujocoJointSceneOracle(self.scene, model_qpos, contact_binding=binding)
                pair = {side: model_qpos[address] for side, address in oracle.addresses.items()}
                check = oracle.check(pair)
                if not check.safe:
                    return check
                pose = xyzquat_to_matrix(self.pose_for_configuration(model_qpos, pair))
                if not binding.in_corridor(pose, xyzquat_to_matrix(target_pose)):
                    return JointConfigurationCheck(
                        safe=False, minimum_distance_m=None,
                        reason="grasp_contact_measured_outside_terminal_corridor",
                        metadata=binding.to_payload())
                return check


        return self.rig.capture_scene(capture)

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

    @_scene_serialized()
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
            next_index = min(index + 1, len(self._active.reference_poses) - 1)
            current_stage = self._stage_at(self._active, max(0, index))
            next_stage = self._stage_at(self._active, next_index)
            if next_stage is not current_stage and next_stage.configuration_constraint is not None:
                # A commanded near-pose is not measured arrival. Keep the
                # already selected stopped knot until the plant enters the
                # original terminal corridor; never widen it or skip ahead.
                model_qpos = self._model_qpos()
                pair = {side: model_qpos[[self.scene.joint_qpos_addr(n)
                        for n in self.scene.arm_specs[side].joint_names]]
                        for side in ("left", "right")}
                check = next_stage.configuration_constraint(model_qpos, pair)
                if not check.safe:
                    if check.reason == "grasp_contact_outside_terminal_corridor":
                        return self._active.reference_poses[index]
                    raise MujocoTcpPlanRejected("safe_tcp_stage_transition_blocked", check.to_dict())
            if index > 0 and np.any(
                np.isclose(
                    self._active.trajectory.timestamps[index],
                    self._active.trajectory.waypoint_timestamps[1:-1],
                    atol=1e-9,
                    rtol=0,
                )
            ):
                measured = self.rig.measured_qpos_by_side()[self.motion_side]
                if np.max(
                    np.abs(measured - self._active.trajectory.q_rad[index])
                ) > np.deg2rad(self.rig.config.max_command_step_deg):
                    # A dispatched reference is not proof that the plant has
                    # reached the corner. Keep that same planned reference;
                    # do not start the next leg or catch up by skipping points.
                    return self._active.reference_poses[index]
            self._reference_index = next_index
            return self._active.reference_poses[self._reference_index]

    @_scene_serialized()
    def resolve(self, action: RobotAction) -> IKResult:
        with self._lock:
            plan = self._active
            index = self._reference_index
            if plan is None or index < 0:
                raise MujocoTcpPlanRejected("safe_tcp_reference_not_selected")
            requested = np.asarray(action.arm_pose_by_side[self.motion_side])
            expected = np.asarray(plan.reference_poses[index], dtype=np.float32)
            if not np.allclose(
                xyzquat_to_matrix(requested),
                xyzquat_to_matrix(expected),
                atol=2e-6,
                rtol=0,
            ):
                raise MujocoTcpPlanRejected("safe_tcp_reference_changed_after_safety")
            q = {**plan.path.start_qpos_by_component,
                 self.motion_side: plan.trajectory.q_rad[index]}
            # Recheck the prescribed reference against today's scene; no
            # projection or replacement IK branch may silently change it.
            stage = self._stage_at(plan, index)
            oracle = self._live_oracle if stage is self._live_stage else self._oracle_for_stage(stage)
            if oracle is None:
                raise MujocoTcpPlanRejected("safe_tcp_live_scene_unavailable")
            if plan.payload_binding is not None:
                def resolve_payload(scene):
                    self._capture_current_payload(plan, oracle, scene)
                    return self._resolve_checked_reference(plan, index, q, oracle)

                # Keep the current-cut test, measured observer and complete
                # candidate sweep under the one existing physics-owner lock.
                return self.rig.capture_scene(resolve_payload)
            oracle.source.qpos[:] = self._model_qpos()
            result = self._resolve_checked_reference(plan, index, q, oracle)
            if stage is not self._live_stage:
                self._bind_live_stage(stage, oracle, bind_noncontact=bool(plan.stages))
            return result

    def _resolve_checked_reference(self, plan, index, q, oracle) -> IKResult:
        if not np.allclose(
            oracle.source.qpos[oracle.addresses[self._inactive_side]],
            _measured_start(plan)[self._inactive_side], atol=1e-3, rtol=0,
        ):
            raise MujocoTcpPlanRejected("safe_tcp_passive_arm_changed")
        # The payload sweep checks the whole live edge including its endpoint.
        # Its source must be today's cut, not the frozen search-only qpos.
        if plan.payload_binding is None:
            check = oracle.check(q)
            if not check.safe:
                background = set(range(oracle.source.qpos.size)) - set(
                    np.concatenate(tuple(oracle.addresses.values())).tolist()
                )
                changed = {
                    str(i): [float(plan.source_model_qpos[i]), float(oracle.source.qpos[i])]
                    for i in sorted(background)
                    if abs(plan.source_model_qpos[i] - oracle.source.qpos[i]) > 1e-8
                }
                raise MujocoTcpPlanRejected("safe_tcp_live_scene_blocked", {
                    "check": check.to_dict(), "reference_index": index,
                    "changed_background_qpos": changed,
                })
        start_q = np.concatenate([
            oracle.source.qpos[oracle.addresses[side]] for side in ("left", "right")
        ])
        requested_q = np.concatenate([q[side] for side in ("left", "right")])
        if oracle.configuration_constraint is not None:
            count = max(1, math.ceil(float(np.max(np.abs(requested_q - start_q))) / 0.01))
            for alpha in np.linspace(0, 1, count + 1):
                point = start_q + alpha * (requested_q - start_q)
                constraint = oracle.configuration_constraint(oracle.source.qpos, {
                    "left": point[:len(oracle.addresses["left"])],
                    "right": point[len(oracle.addresses["left"]):],
                })
                if not constraint.safe:
                    raise MujocoTcpPlanRejected("safe_tcp_live_reference_constraint_blocked", {
                        "edge_alpha": float(alpha), "check": constraint.to_dict(),
                    })
        sweep = oracle.projector.project(
            oracle.source, start_qpos=start_q, requested_qpos=requested_q
        )
        if sweep.modified or not sweep.check.safe:
            raise MujocoTcpPlanRejected(
                "safe_tcp_live_reference_sweep_blocked", sweep.to_dict()
            )
        return IKResult(
            success=True, iterations=0,
            qpos_by_side={side: values.tolist() for side, values in q.items()},
            position_error_m={self.motion_side: 0.0},
            orientation_error_rad={self.motion_side: 0.0},
            status="planned_joint_reference",
            solver="mujoco.registered_collision_checked_joint_path",
            metadata={"trajectory_hash": plan.trajectory_hash, "reference_index": index},
        )
