"""URDF/FCL and optional observed-gripper planning for one RM65 approach Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import NoReturn
from zipfile import BadZipFile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.grasp_visible_scene import VisibleGraspScene
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
    plan_collision_aware_joint_path,
)
from robot_motion.pose import pose_delta_metrics
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import (
    OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
    OfflineWaypointTrajectoryInfeasibleError,
    build_offline_quintic_waypoint_trajectory,
)
from robot_motion.rm65 import (
    DEFAULT_GRASP_PATH_COLLISION_SCENE,
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    build_dual_arm_scene_profile,
)
from robot_motion.rm65.obstacle_collision import (
    RM65_D405_ATTACHED_COLLISION_PROXIES,
    RM65ObstacleCollisionOracle,
    build_rm65_profile_obstacle_collision_oracle,
)
from robot_motion.rm65.scene_profiles import fixed_scene_aabb_obstacles
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from teleop_runtime.target_safety.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeTargetFilter,
)

from .grasp_trajectory_record import write_grasp_trajectory_record
from .motion_port import RealAgentJointTrajectory
from .rm65_ports import RM65SkillTargetBuilder
from .rm65_profile import ResolvedExclusiveRM65SkillDeploymentProfile

_SIDES = ("left", "right")
_PATH_COLLISION_SCENE_PROFILE = DEFAULT_GRASP_PATH_COLLISION_SCENE
_DMTAC_PATH_COLLISION_SCENE_PROFILE = "dual_arm_hand_proxy_dmtac_strict"
RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC = 120.0
# The capture-time CAD/depth filter is accurate to a bounded image/depth
# neighbourhood, but DMTAC points at the pad-height silhouette can still land
# just outside the analytic open-gripper proxy after reprojection.  Six
# millimetres covers the measured 5.2--6.0 mm boundary residual while remaining
# below the 19.9 mm finger thickness.  This margin applies only at the captured
# robot pose, never to arbitrary world points along the planned path, and target
# labels remain immutable.
RM65_GRASP_CAPTURE_SELF_OBSERVATION_SURFACE_MARGIN_M = 0.006
RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M = 0.006
RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG = 3.0
RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP = "segmented_stop"
RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2 = "continuous_c2"
RM65_GRASP_TRAJECTORY_PARAMETERIZATION_MODES = (
    RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP,
    RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2,
)
RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM = "uniform"
RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X = (
    "far_4x_approach_1x"
)
RM65_GRASP_TRAJECTORY_SPEED_PROFILE_MODES = (
    RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM,
    RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X,
)
RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE = "joint_distance"
RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME = (
    "estimated_execution_time"
)
RM65_GRASP_ENDPOINT_ORDER_MODES = (
    RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE,
    RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME,
)
# A failed fixed-orientation pregrasp is a local-route rejection, not proof
# that the terminal grasp target is unreachable.  These failures may fall
# back to the existing global target search; unavailable checks, a blocked
# measured start, and exhausted global time must remain terminal.
_RM65_GLOBAL_GRASP_TARGET_FALLBACK_REASONS = frozenset(
    {
        "ik_search_exhausted",
        "approach_ik_search_exhausted",
        "approach_branch_discontinuous",
        "endpoint_collision_blocked",
        "endpoint_clearance_blocked",
        "endpoint_safety_envelope_blocked",
        "approach_endpoint_collision_blocked",
        "approach_endpoint_clearance_blocked",
        "approach_endpoint_safety_envelope_blocked",
        "path_search_exhausted",
        "path_target_collision_blocked",
        "path_collision_blocked",
        "path_target_safety_envelope_blocked",
        "target_safety_envelope_blocked",
    }
)


def _joint_proposals_match(first: np.ndarray, second: np.ndarray, *, atol: float) -> bool:
    """Compare normalized finite joint proposals at zero relative tolerance."""
    return bool(np.all(np.abs(first - second) <= atol))


class RM65GraspTrajectoryPlanningError(RuntimeError):
    """Typed zero-motion failure produced before lower activation."""

    def __init__(
        self, reason_code: str, *, details: Mapping[str, object] | None = None
    ):
        self.reason_code = str(reason_code).strip()
        if not self.reason_code:
            raise ValueError("reason_code must be non-empty.")
        self.details = dict(details or {})
        super().__init__(self.reason_code)


@dataclass(frozen=True)
class RM65GraspTrajectoryPlanningConfig:
    minimum_clearance_m: float = 0.02
    linear_speed_m_s: float = 0.12
    angular_speed_deg_s: float = 45.0
    minimum_segment_duration_sec: float = 0.25
    pregrasp_distance_m: float = 0.06
    minimum_pregrasp_distance_m: float = 0.02
    pregrasp_distance_step_m: float = 0.01
    approach_waypoint_spacing_m: float = 0.005
    maximum_approach_joint_step_rad: float = float(np.deg2rad(12.0))
    maximum_approach_lateral_error_m: float = 0.002
    pregrasp_orientation_tolerance_deg: float = 3.0
    maximum_approach_orientation_error_deg: float = 3.0
    approach_progress_tolerance_m: float = 0.0005
    maximum_endpoint_solve_cycles: int = 180
    endpoint_seed_count: int = 24
    endpoint_seed_perturbation_rad: float = float(np.deg2rad(25.0))
    maximum_endpoint_candidates: int = 8
    endpoint_position_residual_scale_m: float = 0.01
    endpoint_orientation_residual_scale_deg: float = 10.0
    endpoint_tolerance_margin_ratio: float = 0.90
    path_search_attempts_per_endpoint: int = 4
    path_planning_time_budget_sec: float = RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC
    endpoint_position_tolerance_m: float = 0.003
    endpoint_orientation_tolerance_deg: float = 1.0
    final_edge_resolution_rad: float = float(np.deg2rad(0.1))
    maximum_path_envelope_ingress_distance_m: float = 0.16
    path_envelope_ingress_progress_tolerance_m: float = 0.002
    path_envelope_ingress_tangent_tolerance_m: float = 0.0005
    singularity_soft_margin_deg: float = 6.0
    singularity_replay_slowdown: float = 0.5
    trajectory_parameterization: str = (
        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP
    )
    trajectory_speed_profile: str = RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM
    endpoint_candidate_ordering: str = RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE
    allow_unconstrained_terminal_fallback: bool = False
    path_envelope_ingress_waypoint_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.35,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.allow_unconstrained_terminal_fallback, bool):
            raise ValueError("allow_unconstrained_terminal_fallback must be bool.")
        for name in (
            "minimum_clearance_m",
            "linear_speed_m_s",
            "angular_speed_deg_s",
            "minimum_segment_duration_sec",
            "pregrasp_distance_m",
            "minimum_pregrasp_distance_m",
            "pregrasp_distance_step_m",
            "approach_waypoint_spacing_m",
            "maximum_approach_joint_step_rad",
            "maximum_approach_lateral_error_m",
            "pregrasp_orientation_tolerance_deg",
            "maximum_approach_orientation_error_deg",
            "approach_progress_tolerance_m",
            "endpoint_position_tolerance_m",
            "endpoint_orientation_tolerance_deg",
            "endpoint_seed_perturbation_rad",
            "endpoint_position_residual_scale_m",
            "endpoint_orientation_residual_scale_deg",
            "path_planning_time_budget_sec",
            "final_edge_resolution_rad",
            "maximum_path_envelope_ingress_distance_m",
            "path_envelope_ingress_progress_tolerance_m",
            "path_envelope_ingress_tangent_tolerance_m",
            "singularity_soft_margin_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if (
            self.pregrasp_orientation_tolerance_deg
            > self.maximum_approach_orientation_error_deg
        ):
            raise ValueError(
                "pregrasp orientation tolerance must fit the approach bound."
            )
        if (
            self.pregrasp_orientation_tolerance_deg
            < self.endpoint_orientation_tolerance_deg
        ):
            raise ValueError(
                "pregrasp orientation tolerance must not be stricter than "
                "the terminal endpoint tolerance."
            )
        if self.singularity_soft_margin_deg <= RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG:
            raise ValueError(
                "singularity_soft_margin_deg must exceed the hard threshold."
            )
        slowdown = float(self.singularity_replay_slowdown)
        if not math.isfinite(slowdown) or not 0.0 < slowdown <= 1.0:
            raise ValueError("singularity_replay_slowdown must be in (0, 1].")
        object.__setattr__(self, "singularity_replay_slowdown", slowdown)
        parameterization = str(self.trajectory_parameterization).strip().lower()
        if parameterization not in RM65_GRASP_TRAJECTORY_PARAMETERIZATION_MODES:
            raise ValueError(
                "trajectory_parameterization must be one of "
                f"{RM65_GRASP_TRAJECTORY_PARAMETERIZATION_MODES}."
            )
        object.__setattr__(
            self,
            "trajectory_parameterization",
            parameterization,
        )
        speed_profile = str(self.trajectory_speed_profile).strip().lower()
        if speed_profile not in RM65_GRASP_TRAJECTORY_SPEED_PROFILE_MODES:
            raise ValueError(
                "trajectory_speed_profile must be one of "
                f"{RM65_GRASP_TRAJECTORY_SPEED_PROFILE_MODES}."
            )
        object.__setattr__(self, "trajectory_speed_profile", speed_profile)
        endpoint_ordering = str(self.endpoint_candidate_ordering).strip().lower()
        if endpoint_ordering not in RM65_GRASP_ENDPOINT_ORDER_MODES:
            raise ValueError(
                "endpoint_candidate_ordering must be one of "
                f"{RM65_GRASP_ENDPOINT_ORDER_MODES}."
            )
        object.__setattr__(
            self,
            "endpoint_candidate_ordering",
            endpoint_ordering,
        )
        if self.minimum_pregrasp_distance_m > self.pregrasp_distance_m:
            raise ValueError(
                "minimum_pregrasp_distance_m must not exceed pregrasp_distance_m."
            )
        margin = float(self.endpoint_tolerance_margin_ratio)
        if not math.isfinite(margin) or not 0.0 < margin < 1.0:
            raise ValueError("endpoint_tolerance_margin_ratio must be in (0, 1).")
        object.__setattr__(self, "endpoint_tolerance_margin_ratio", margin)
        cycles = int(self.maximum_endpoint_solve_cycles)
        if isinstance(self.maximum_endpoint_solve_cycles, bool) or cycles < 1:
            raise ValueError("maximum_endpoint_solve_cycles must be positive.")
        object.__setattr__(self, "maximum_endpoint_solve_cycles", cycles)
        seed_count = int(self.endpoint_seed_count)
        if isinstance(self.endpoint_seed_count, bool) or not 1 <= seed_count <= 64:
            raise ValueError("endpoint_seed_count must be in [1, 64].")
        object.__setattr__(self, "endpoint_seed_count", seed_count)
        endpoint_candidates = int(self.maximum_endpoint_candidates)
        if (
            isinstance(self.maximum_endpoint_candidates, bool)
            or not 1 <= endpoint_candidates <= 16
        ):
            raise ValueError("maximum_endpoint_candidates must be in [1, 16].")
        object.__setattr__(
            self,
            "maximum_endpoint_candidates",
            endpoint_candidates,
        )
        path_attempts = int(self.path_search_attempts_per_endpoint)
        if (
            isinstance(self.path_search_attempts_per_endpoint, bool)
            or not 1 <= path_attempts <= 8
        ):
            raise ValueError("path_search_attempts_per_endpoint must be in [1, 8].")
        object.__setattr__(
            self,
            "path_search_attempts_per_endpoint",
            path_attempts,
        )
        fractions = tuple(
            float(value) for value in self.path_envelope_ingress_waypoint_fractions
        )
        if (
            not fractions
            or len(fractions) > 8
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions
            )
            or any(current <= previous for previous, current in pairwise(fractions))
        ):
            raise ValueError(
                "path_envelope_ingress_waypoint_fractions must contain 1-8 "
                "strictly increasing finite values in (0, 1)."
            )
        object.__setattr__(
            self,
            "path_envelope_ingress_waypoint_fractions",
            fractions,
        )


@dataclass(frozen=True)
class RM65GraspReachabilityConfig:
    """Finite endpoint-IK budget used only for optional candidate annotation."""

    time_budget_ms: float = 1_000.0
    max_solve_cycles: int = 2_880
    max_solve_evaluations_per_seed: int | None = None
    seed_count: int = 16
    standard_position_tolerance_m: float = 0.003
    standard_orientation_tolerance_deg: float = 1.0
    basic_position_tolerance_m: float = 0.05
    basic_orientation_tolerance_deg: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "time_budget_ms",
            "standard_position_tolerance_m",
            "standard_orientation_tolerance_deg",
            "basic_position_tolerance_m",
            "basic_orientation_tolerance_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        cycles = int(self.max_solve_cycles)
        if isinstance(self.max_solve_cycles, bool) or cycles < 1:
            raise ValueError("max_solve_cycles must be positive.")
        object.__setattr__(self, "max_solve_cycles", cycles)
        configured_per_seed = self.max_solve_evaluations_per_seed
        per_seed = (
            min(180, cycles)
            if configured_per_seed is None
            else int(configured_per_seed)
        )
        if isinstance(configured_per_seed, bool) or not 1 <= per_seed <= cycles:
            raise ValueError(
                "max_solve_evaluations_per_seed must be within the total budget."
            )
        object.__setattr__(self, "max_solve_evaluations_per_seed", per_seed)
        seed_count = int(self.seed_count)
        if isinstance(self.seed_count, bool) or not 1 <= seed_count <= 64:
            raise ValueError("seed_count must be in [1, 64].")
        object.__setattr__(self, "seed_count", seed_count)
        if (
            self.standard_position_tolerance_m >= self.basic_position_tolerance_m
            or self.standard_orientation_tolerance_deg
            >= self.basic_orientation_tolerance_deg
        ):
            raise ValueError("Standard reachability tolerances must be stricter.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "evaluation_scope": (
                    "finite_time_endpoint_ik_not_path_or_grasp_success"
                )
            },
            domain="dexteleop.real_skill.rm65_grasp_reachability.v1",
        )


@dataclass(frozen=True)
class RM65GraspReachabilityResult:
    classification: str
    position_residual_m: float
    orientation_residual_deg: float
    solve_cycles: int
    elapsed_ms: float
    termination_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RM65GraspLocalChainResult:
    """Read-only admission result for the production pregrasp and approach."""

    feasible: bool
    selected_pregrasp_distance_m: float | None
    branch_count: int
    elapsed_ms: float
    reason_code: str | None
    attempts: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "selected_pregrasp_distance_m": self.selected_pregrasp_distance_m,
            "branch_count": self.branch_count,
            "elapsed_ms": self.elapsed_ms,
            "reason_code": self.reason_code,
            "attempts": [dict(item) for item in self.attempts],
        }


@dataclass(frozen=True)
class RM65GraspEndpointProbeResult:
    """Advisory terminal IK/clearance evidence used to order candidates.

    A negative probe is never a physical rejection: its small wall-time budget
    can miss an IK branch.  The production local-chain validator remains the
    admission authority.
    """

    classification: str
    elapsed_ms: float
    reason_code: str | None
    best_minimum_distance_m: float | None
    attempted_seed_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class _EndpointIKCandidate:
    qpos_by_side: Mapping[str, np.ndarray]
    seed_index: int
    solve_cycles: int
    position_residual_m: float
    orientation_residual_deg: float


@dataclass(frozen=True)
class _PlannedPathLeg:
    endpoint: _EndpointIKCandidate
    path: JointPathPlanResult
    dense_revalidation: Mapping[str, object]
    search_attempts: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _PreparedGraspApproaches:
    candidates: tuple[tuple[_EndpointIKCandidate, _PlannedPathLeg], ...]
    validation_attempts: tuple[Mapping[str, object], ...]


class _CollisionSearchOracle:
    """Use clearance guards only in search, not endpoint or final validation."""

    def __init__(self, oracle: RM65ObstacleCollisionOracle, refinement_distance_m: float) -> None:
        self.oracle = oracle
        self.refinement_distance_m = refinement_distance_m
        self.scene_hash = oracle.scene_hash

    def check(self, qpos_by_component: Mapping[str, np.ndarray]) -> JointConfigurationCheck:
        return self.oracle.check_for_search(
            qpos_by_component, refinement_distance_m=self.refinement_distance_m,
        )


class _TargetEnvelopePathOracle:
    """Compose collision and TCP-envelope checks for grasp path search."""

    def __init__(
        self,
        *,
        collision_oracle: object,
        envelope_check: Callable[[Mapping[str, np.ndarray]], None],
    ) -> None:
        self._collision_oracle = collision_oracle
        self._envelope_check = envelope_check
        self.scene_hash = collision_oracle.scene_hash

    def check(
        self,
        qpos_by_component: Mapping[str, np.ndarray],
    ) -> JointConfigurationCheck:
        collision_check = self._collision_oracle.check(qpos_by_component)
        if not isinstance(collision_check, JointConfigurationCheck):
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_oracle_returned_invalid_type",
                metadata={"returned_type": type(collision_check).__name__},
            )
        if not collision_check.collision_check_available or not collision_check.safe:
            return collision_check
        try:
            self._envelope_check(qpos_by_component)
        except RM65GraspTrajectoryPlanningError as error:
            available = error.reason_code == "path_target_safety_envelope_blocked"
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=available,
                reason=error.reason_code,
                minimum_distance_m=collision_check.minimum_distance_m,
                metadata={
                    "collision_check": collision_check.to_dict(),
                    "target_envelope": error.details,
                },
            )
        except Exception as error:  # noqa: BLE001 - search must fail closed.
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="path_target_safety_envelope_unavailable",
                minimum_distance_m=collision_check.minimum_distance_m,
                metadata={
                    "collision_check": collision_check.to_dict(),
                    "exception_type": type(error).__name__,
                },
            )
        return collision_check


def _terminal_target_contact_region(
    *,
    tool_tcp_world: np.ndarray,
    terminal_tool_tcp_world: np.ndarray,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> tuple[bool, float, float]:
    position_error_m, orientation_error_deg = pose_delta_metrics(
        tool_tcp_world,
        terminal_tool_tcp_world,
    )
    return (
        position_error_m <= position_tolerance_m
        and orientation_error_deg <= orientation_tolerance_deg,
        position_error_m,
        orientation_error_deg,
    )


class _VisibleGripperPathOracle:
    """Compose observed gripper solids with, never replace, canonical safety."""
    def __init__(self, *, oracle: object, visible_scene: VisibleGraspScene,
                 target_builder: RM65SkillTargetBuilder, side: str,
                 terminal_tool_tcp_world: np.ndarray | None = None,
                 terminal_position_tolerance_m: float = (
                     RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M
                 ),
                 terminal_orientation_tolerance_deg: float = 1.0) -> None:
        self.oracle = oracle
        self.visible_scene = visible_scene
        self.target_builder = target_builder
        self.side = side
        self.terminal_tool_tcp_world = (
            None
            if terminal_tool_tcp_world is None
            else np.asarray(terminal_tool_tcp_world, dtype=float).copy()
        )
        self.terminal_position_tolerance_m = float(
            terminal_position_tolerance_m
        )
        self.terminal_orientation_tolerance_deg = float(
            terminal_orientation_tolerance_deg
        )
        self.scene_hash = canonical_content_hash({
            "production_scene_hash": oracle.scene_hash,
            "visible_scene_hash": visible_scene.semantic_content_hash,
            "terminal_contact_target": (
                None
                if self.terminal_tool_tcp_world is None
                else self.terminal_tool_tcp_world.reshape(-1).tolist()
            ),
            "terminal_contact_depth_m": (
                RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M
            ),
            "terminal_contact_position_tolerance_m": (
                self.terminal_position_tolerance_m
            ),
            "terminal_contact_orientation_tolerance_deg": (
                self.terminal_orientation_tolerance_deg
            ),
        }, domain="dexteleop.motion.composed_grasp_scene.v1")

    def check(self, qpos_by_component: Mapping[str, np.ndarray]) -> JointConfigurationCheck:
        check = self.oracle.check(qpos_by_component)
        if (not isinstance(check, JointConfigurationCheck) or not check.safe
                or not check.collision_check_available):
            return check
        pose = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos_by_component["left"], right_qpos=qpos_by_component["right"]
        )[self.side]
        terminal_contact_points = 0
        terminal_contact_allowed = False
        if self.terminal_tool_tcp_world is not None:
            terminal_contact_allowed, _, _ = _terminal_target_contact_region(
                tool_tcp_world=pose,
                terminal_tool_tcp_world=self.terminal_tool_tcp_world,
                position_tolerance_m=self.terminal_position_tolerance_m,
                orientation_tolerance_deg=(
                    self.terminal_orientation_tolerance_deg
                ),
            )
        if terminal_contact_allowed:
            (
                target_hits,
                terminal_contact_points,
                background_hits,
            ) = self.visible_scene.collision_counts_with_terminal_target_contact(
                pose,
                opening_m=self.visible_scene.geometry.maximum_opening_m,
                contact_depth_m=RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
            )
        else:
            target_hits, background_hits = self.visible_scene.collision_counts(
                pose, opening_m=self.visible_scene.geometry.maximum_opening_m
            )
        if not target_hits and not background_hits:
            return check
        return JointConfigurationCheck(
            safe=False, collision_check_available=True,
            reason="visible_gripper_path_collision",
            metadata={"production_collision_check": check.to_dict(),
                      "target_collision_points": target_hits,
                      "background_collision_points": background_hits,
                      "terminal_target_contact_points": terminal_contact_points,
                      "terminal_contact_allowed": terminal_contact_allowed},
        )


class RM65GraspJointTrajectoryPlanner:
    """Resolve endpoint IK, plan around fixed table/poles, then time-sample qpos."""

    def __init__(
        self,
        *,
        profile: ResolvedExclusiveRM65SkillDeploymentProfile,
        planning_target_builder: object,
        target_envelope_filter: (
            Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray] | None] | None
        ),
        config: RM65GraspTrajectoryPlanningConfig | None = None,
        planner_config: JointPathPlannerConfig | None = None,
        reachability_config: RM65GraspReachabilityConfig | None = None,
        trajectory_record_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(profile, ResolvedExclusiveRM65SkillDeploymentProfile):
            raise TypeError("profile must be resolved and typed.")
        for name in (
            "set_current_qpos_pair",
            "solver_world_targets_for_qpos_pair",
            "solver_world_target_from_tool_tcp_world_target",
            "tool_tcp_world_targets_for_qpos_pair",
            "tool_tcp_world_target_from_solver_world_target",
            "solve_target_pair",
        ):
            if not callable(getattr(planning_target_builder, name, None)):
                raise TypeError(f"planning_target_builder must expose {name}().")
        if target_envelope_filter is not None and not callable(target_envelope_filter):
            raise TypeError("target_envelope_filter must be callable or None.")
        self.profile = profile
        self.target_builder = planning_target_builder
        self.target_envelope_filter = target_envelope_filter
        self.config = config or RM65GraspTrajectoryPlanningConfig()
        self.reachability_config = reachability_config or RM65GraspReachabilityConfig()
        if trajectory_record_dir is None:
            self.trajectory_record_dir = None
        else:
            record_dir = Path(trajectory_record_dir)
            if not record_dir.is_dir() or record_dir.is_symlink():
                raise ValueError("trajectory_record_dir must be an existing real directory.")
            self.trajectory_record_dir = record_dir
        self._solver_lock = threading.Lock()
        self.planner_config = planner_config or JointPathPlannerConfig(
            shortcut_attempts=80,
            adaptive_edge_resolution_rad=self.config.final_edge_resolution_rad,
            adaptive_refinement_distance_m=0.03,
        )
        self.path_collision_scene_profile = (
            _DMTAC_PATH_COLLISION_SCENE_PROFILE
            if profile.collision_scene_profile
            == _DMTAC_PATH_COLLISION_SCENE_PROFILE
            else _PATH_COLLISION_SCENE_PROFILE
        )
        path_profile = build_dual_arm_scene_profile(
            self.path_collision_scene_profile
        )
        fixed_obstacles = fixed_scene_aabb_obstacles(path_profile)
        required = {
            "fixed-scene:camera_pole",
            "fixed-scene:left_camera_pole",
            "fixed-scene:right_camera_pole",
            "fixed-scene:table",
        }
        if {item.obstacle_id for item in fixed_obstacles} != required:
            raise ValueError(
                "strict RM65 URDF must expose table and three camera poles."
            )
        frame_hash = canonical_content_hash(
            {
                "deployment_profile_hash": profile.semantic_content_hash,
                "collision_scene_profile": self.path_collision_scene_profile,
                "arm_collision_representation": "ik_sparse_proxy_pairs_v1",
                "fixed_obstacles": [item.to_dict() for item in fixed_obstacles],
            },
            domain="dexteleop.real_skill.rm65_fixed_obstacle_frame.v1",
        )
        self.scene = ObstacleSceneSnapshot(
            reference_frame_id=profile.station_world_frame_id,
            reference_frame_spec_hash=frame_hash,
            revision=1,
            obstacles=fixed_obstacles,
            minimum_clearance_m=self.config.minimum_clearance_m,
        )
        # The reviewed IK profile's sparse pairs are authoritative for both
        # search and final path admission. This deliberately does not certify
        # full arm-mesh or finger-mesh clearance.
        self.oracle = build_rm65_profile_obstacle_collision_oracle(
            scene_profile=self.path_collision_scene_profile,
            scene=self.scene,
            expected_reference_frame_id=self.scene.reference_frame_id,
            expected_reference_frame_spec_hash=self.scene.reference_frame_spec_hash,
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
            use_profile_proxy_geometry=True,
        )

    def _remaining_path_planning_time_sec(
        self,
        *,
        deadline: float,
        monotonic: Callable[[], float],
        planning_phase: str,
        details: Mapping[str, object] | None = None,
    ) -> float:
        remaining = float(deadline) - float(monotonic())
        if remaining <= 0.0:
            raise RM65GraspTrajectoryPlanningError(
                "path_planning_time_budget_exhausted",
                details={
                    "planning_phase": planning_phase,
                    "time_budget_sec": self.config.path_planning_time_budget_sec,
                    **dict(details or {}),
                },
            )
        return remaining

    def _target_envelope_projection_details(
        self,
        modified_sides: tuple[str, ...],
    ) -> dict[str, object]:
        """Return compact diagnostics without weakening a fail-closed decision."""

        snapshot_fn = getattr(self.target_envelope_filter, "snapshot", None)
        if not callable(snapshot_fn):
            return {}
        try:
            projection = getattr(snapshot_fn(), "projection", None)
        except Exception:  # noqa: BLE001 - diagnostics must not mask the block.
            return {}
        if not isinstance(projection, Mapping):
            return {}
        raw_pose_by_side = projection.get("raw_tcp_pose_by_side")
        effective_pose_by_side = projection.get("effective_tcp_pose_by_side")
        side_projection_by_side = projection.get("side_projection_by_side")
        if not all(
            isinstance(value, Mapping)
            for value in (
                raw_pose_by_side,
                effective_pose_by_side,
                side_projection_by_side,
            )
        ):
            return {}

        def xyz_from(mapping: Mapping[str, object], side: str) -> list[float] | None:
            try:
                pose = np.asarray(mapping[side], dtype=float)
            except (KeyError, TypeError, ValueError):
                return None
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                return None
            return [float(value) for value in pose[:3, 3]]

        details_by_side: dict[str, object] = {}
        for side in modified_sides:
            side_projection = side_projection_by_side.get(side)
            if not isinstance(side_projection, Mapping):
                continue
            details: dict[str, object] = {}
            for field_name in ("translation_modified", "rotation_modified"):
                value = side_projection.get(field_name)
                if type(value) is bool:
                    details[field_name] = value
            for source_name, output_name, scale in (
                ("translation_adjustment_m", "translation_adjustment_m", 1.0),
                ("rotation_adjustment_rad", "rotation_adjustment_deg", 180.0 / math.pi),
            ):
                value = side_projection.get(source_name)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    details[output_name] = float(value) * scale
            for field_name in (
                "violated_translation_constraints",
                "active_translation_constraints",
            ):
                value = side_projection.get(field_name)
                if isinstance(value, (tuple, list)) and all(
                    isinstance(item, str) and item for item in value
                ):
                    details[field_name] = list(value)
            raw_xyz = xyz_from(raw_pose_by_side, side)
            effective_xyz = xyz_from(effective_pose_by_side, side)
            if raw_xyz is not None:
                details["raw_target_xyz_m"] = raw_xyz
            if effective_xyz is not None:
                details["effective_target_xyz_m"] = effective_xyz
            if details:
                details_by_side[side] = details
        return (
            {"envelope_projection_by_side": details_by_side} if details_by_side else {}
        )

    def _require_unmodified_target_envelope(
        self,
        solver_targets: Mapping[str, np.ndarray],
        *,
        sides: tuple[str, ...],
        phase: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if self.target_envelope_filter is None:
            return
        projected = self._project_target_envelope(
            solver_targets,
            phase=phase,
            details=details,
        )
        modified_sides = tuple(
            side
            for side in sides
            if not np.allclose(
                np.asarray(projected[side], dtype=float),
                np.asarray(solver_targets[side], dtype=float),
                atol=1e-9,
                rtol=0.0,
            )
        )
        if modified_sides:
            raise RM65GraspTrajectoryPlanningError(
                f"{phase}_safety_envelope_blocked",
                details={
                    **dict(details or {}),
                    "modified_sides": modified_sides,
                    **self._target_envelope_projection_details(modified_sides),
                },
            )

    def _project_target_envelope(
        self,
        solver_targets: Mapping[str, np.ndarray],
        *,
        phase: str,
        details: Mapping[str, object] | None = None,
    ) -> Mapping[str, np.ndarray]:
        if self.target_envelope_filter is None:
            return {
                name: np.asarray(value, dtype=float).copy()
                for name, value in solver_targets.items()
            }
        try:
            # Only the native filter's read-only planning port omits report
            # construction. Injected/subclassed filters retain their contract.
            project = (
                self.target_envelope_filter.project_for_planning
                if type(self.target_envelope_filter) is TCPPoseEnvelopeTargetFilter
                else self.target_envelope_filter
            )
            projected = project(
                {
                    name: np.asarray(value, dtype=float).copy()
                    for name, value in solver_targets.items()
                }
            )
        except Exception as exc:
            raise RM65GraspTrajectoryPlanningError(
                f"{phase}_safety_envelope_unavailable",
                details={
                    **dict(details or {}),
                    "exception_type": type(exc).__name__,
                },
            ) from exc
        if not isinstance(projected, Mapping) or set(projected) != set(_SIDES):
            raise RM65GraspTrajectoryPlanningError(
                f"{phase}_safety_envelope_unavailable",
                details=dict(details or {}),
            )
        normalized: dict[str, np.ndarray] = {}
        for side in _SIDES:
            value = np.asarray(projected[side], dtype=float)
            if value.shape != (4, 4) or not np.all(np.isfinite(value)):
                raise RM65GraspTrajectoryPlanningError(
                    f"{phase}_safety_envelope_unavailable",
                    details=dict(details or {}),
                )
            normalized[side] = value.copy()
        return normalized

    def _require_qpos_within_target_envelope(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        *,
        sides: tuple[str, ...],
        phase: str,
    ) -> None:
        solver_targets = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
        )
        self._require_unmodified_target_envelope(
            solver_targets,
            sides=sides,
            phase=phase,
        )

    def _qpos_is_within_target_envelope(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        *,
        sides: tuple[str, ...],
        phase: str,
    ) -> bool:
        try:
            self._require_qpos_within_target_envelope(
                qpos_by_side,
                sides=sides,
                phase=phase,
            )
        except RM65GraspTrajectoryPlanningError as error:
            if error.reason_code == f"{phase}_safety_envelope_blocked":
                return False
            raise
        return True

    def _path_search_oracle(self, *, planning_phase: str,
                            visible_scene: VisibleGraspScene | None = None,
                            side: str | None = None,
                            start_qpos_by_side: Mapping[str, np.ndarray] | None = None,
                            terminal_tool_tcp_world: np.ndarray | None = None) -> object:
        collision_oracle = self.oracle
        if isinstance(collision_oracle, RM65ObstacleCollisionOracle):
            refinement = (
                self.planner_config.adaptive_refinement_distance_m
                if self.planner_config.adaptive_edge_resolution_rad is not None
                else self.scene.minimum_clearance_m
            )
            collision_oracle = _CollisionSearchOracle(collision_oracle, refinement)
        # The measured staging pose may intentionally start just outside the
        # task envelope.  Its dedicated ingress leg is bounded, monotonic, and
        # densely revalidated.  Once inside, every grasp-search sample must
        # satisfy both collision and TCP-envelope constraints.
        if planning_phase == "envelope_ingress" and self.target_envelope_filter is not None:
            if side not in _SIDES or start_qpos_by_side is None:
                raise ValueError("envelope ingress search requires its moving side and start qpos")
            oracle = _TargetEnvelopePathOracle(
                collision_oracle=collision_oracle,
                envelope_check=self._ingress_search_envelope_check(
                    qpos_by_side=start_qpos_by_side, side=side,
                ),
            )
        elif (
            planning_phase not in {"pregrasp_target", "grasp_target"}
            or self.target_envelope_filter is None
        ):
            oracle = collision_oracle
        else:
            oracle = _TargetEnvelopePathOracle(
                collision_oracle=collision_oracle,
                envelope_check=lambda qpos: self._require_qpos_within_target_envelope(
                    qpos, sides=_SIDES, phase="path_target",
                ),
            )
        if visible_scene is not None:
            if side not in _SIDES:
                raise ValueError("visible scene path search requires a moving side")
            return _VisibleGripperPathOracle(
                oracle=oracle, visible_scene=visible_scene,
                target_builder=self.target_builder, side=side,
                terminal_tool_tcp_world=terminal_tool_tcp_world,
                terminal_position_tolerance_m=(
                    max(
                        self.config.endpoint_position_tolerance_m,
                        RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
                    )
                ),
                terminal_orientation_tolerance_deg=(
                    self.config.endpoint_orientation_tolerance_deg
                ),
            )
        return oracle

    def _pregrasp_target_pose(
        self,
        target_tool_tcp_world: np.ndarray,
        *,
        distance_m: float | None = None,
    ) -> np.ndarray:
        """Back away from the grasp along the tool's configured +Z approach axis."""

        target = np.asarray(target_tool_tcp_world, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_tool_tcp_world must be one finite 4x4 pose.")
        rotation = target[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
            rtol=0.0,
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(
                "target_tool_tcp_world rotation must be right-handed and orthonormal."
            )
        if not np.allclose(
            target[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError("target_tool_tcp_world must be a rigid transform.")
        distance = (
            self.config.pregrasp_distance_m
            if distance_m is None
            else float(distance_m)
        )
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_m must be finite and positive.")
        pregrasp = target.copy()
        pregrasp[:3, 3] -= distance * rotation[:, 2]
        return pregrasp

    def _pregrasp_distance_candidates(self) -> tuple[float, ...]:
        """Prefer the longest local approach that remains fully feasible."""

        maximum = self.config.pregrasp_distance_m
        minimum = self.config.minimum_pregrasp_distance_m
        step = self.config.pregrasp_distance_step_m
        values: list[float] = []
        current = maximum
        while current >= minimum - 1e-12:
            values.append(max(minimum, current))
            current -= step
        if values[-1] > minimum + 1e-12:
            values.append(minimum)
        return tuple(values)

    def pregrasp_distance_candidates(self) -> tuple[float, ...]:
        """Expose the exact production approach distances for read-only admission."""

        return self._pregrasp_distance_candidates()

    def pregrasp_target_pose(
        self,
        target_tool_tcp_world: np.ndarray,
        *,
        distance_m: float,
    ) -> np.ndarray:
        """Build the same pregrasp pose used by production path planning."""

        return self._pregrasp_target_pose(
            target_tool_tcp_world,
            distance_m=distance_m,
        )

    def _singularity_interval_scale(
        self,
        start_qpos: np.ndarray,
        end_qpos: np.ndarray,
    ) -> float:
        """Return the established soft singularity slowdown for one path edge."""

        start = np.asarray(start_qpos, dtype=float).reshape(6)
        end = np.asarray(end_qpos, dtype=float).reshape(6)
        distances: list[float] = []
        for joint_index in (2, 4):
            left = float(start[joint_index])
            right = float(end[joint_index])
            distances.append(
                0.0
                if left == 0.0 or right == 0.0 or left * right < 0.0
                else min(abs(left), abs(right))
            )
        distance_rad = min(distances)
        hard_rad = math.radians(RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG)
        soft_rad = math.radians(self.config.singularity_soft_margin_deg)
        maximum_scale = 1.0 / self.config.singularity_replay_slowdown
        if distance_rad <= hard_rad:
            return maximum_scale
        if distance_rad >= soft_rad:
            return 1.0
        fraction = (soft_rad - distance_rad) / (soft_rad - hard_rad)
        return 1.0 + fraction * (maximum_scale - 1.0)

    def _ordered_endpoint_candidates(
        self,
        *,
        candidates: tuple[_EndpointIKCandidate, ...],
        current_qpos: np.ndarray,
        side: str,
    ) -> tuple[_EndpointIKCandidate, ...]:
        """Order valid IK branches without adding another path-search pass."""

        current = np.asarray(current_qpos, dtype=float).reshape(6)
        if self.config.endpoint_candidate_ordering == (
            RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE
        ):
            return tuple(
                sorted(
                    candidates,
                    key=lambda item: float(
                        np.linalg.norm(
                            np.asarray(item.qpos_by_side[side], dtype=float)
                            - current
                        )
                    ),
                )
            )

        velocity_profile_name = (
            self.profile.position_profile.ik_execution.joint_velocity_limit_profile
        )
        velocity_profile = load_joint_velocity_limit_profiles()[
            velocity_profile_name
        ]
        velocity_limit = np.deg2rad(
            np.asarray(velocity_profile.max_velocity_deg_s, dtype=float)
        )
        if velocity_limit.shape != (6,) or not np.all(velocity_limit > 0.0):
            raise ValueError(
                "joint velocity profile must expose six positive limits."
            )

        def key(item: _EndpointIKCandidate) -> tuple[float, float, int]:
            target = np.asarray(item.qpos_by_side[side], dtype=float)
            delta = np.abs(target - current)
            minimum_joint_time = float(np.max(delta / velocity_limit))
            estimated_time = minimum_joint_time * self._singularity_interval_scale(
                current,
                target,
            )
            return estimated_time, float(np.linalg.norm(delta)), item.seed_index

        return tuple(sorted(candidates, key=key))

    def _segment_duration_diagnostics(
        self,
        *,
        start: Mapping[str, np.ndarray],
        end: Mapping[str, np.ndarray],
        moving_side: str,
        speed_scale: float,
        joint_velocity_rad_s: np.ndarray,
    ) -> tuple[float, dict[str, object]]:
        """Compute the legacy conservative duration and name its active limiter."""

        start_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=start["left"], right_qpos=start["right"]
        )[moving_side]
        end_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=end["left"], right_qpos=end["right"]
        )[moving_side]
        translation_m, rotation_deg = pose_delta_metrics(start_tcp, end_tcp)
        delta = np.abs(
            np.asarray(end[moving_side], dtype=float)
            - np.asarray(start[moving_side], dtype=float)
        )
        candidates = {
            "minimum_segment_duration": self.config.minimum_segment_duration_sec,
            "tcp_translation": (
                1.875
                * translation_m
                / (self.config.linear_speed_m_s * speed_scale)
            ),
            "tcp_rotation": (
                1.875
                * rotation_deg
                / (self.config.angular_speed_deg_s * speed_scale)
            ),
            "joint_velocity": 1.875
            * float(np.max(delta / joint_velocity_rad_s)),
        }
        duration = max(candidates.values())
        active_limiters = tuple(
            name
            for name, value in candidates.items()
            if math.isclose(value, duration, rel_tol=1e-9, abs_tol=1e-12)
        )
        return duration, {
            "duration_sec": float(duration),
            "active_limiters": active_limiters,
            "limiter_duration_sec": {
                name: float(value) for name, value in candidates.items()
            },
            "translation_m": float(translation_m),
            "rotation_deg": float(rotation_deg),
            "maximum_joint_delta_rad": float(np.max(delta)),
        }

    def _effective_trajectory_speed_scale(
        self,
        *,
        base_speed_scale: float,
        planning_phase: str,
    ) -> float:
        """Return the configured speed for a semantic trajectory phase.

        The accelerated profile only affects the collision-checked far-field
        ingress and pregrasp legs.  The final constrained approach and the
        terminal grasp leg retain the caller's conservative base scale.
        """

        if (
            self.config.trajectory_speed_profile
            == RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X
            and planning_phase in {"envelope_ingress", "pregrasp_target"}
        ):
            return min(1.0, 4.0 * float(base_speed_scale))
        return float(base_speed_scale)

    def _continuous_leg_trajectory(
        self,
        *,
        waypoints: tuple[Mapping[str, np.ndarray], ...],
        moving_side: str,
        speed_scale: float,
        joint_velocity_rad_s: np.ndarray,
        reference_hz: float,
    ):
        """Build one stop-to-stop C2 leg with non-zero internal knot velocities."""

        if len(waypoints) < 2:
            raise ValueError("continuous trajectory leg requires at least two waypoints.")
        source_times = [0.0]
        segment_diagnostics: list[dict[str, object]] = []
        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            base_duration, diagnostic = self._segment_duration_diagnostics(
                start=start,
                end=end,
                moving_side=moving_side,
                speed_scale=speed_scale,
                joint_velocity_rad_s=joint_velocity_rad_s,
            )
            singularity_scale = self._singularity_interval_scale(
                np.asarray(start[moving_side], dtype=float),
                np.asarray(end[moving_side], dtype=float),
            )
            duration = base_duration * singularity_scale
            diagnostic.update(
                {
                    "segment_index": segment_index,
                    "singularity_time_scale": float(singularity_scale),
                    "scaled_duration_sec": float(duration),
                }
            )
            segment_diagnostics.append(diagnostic)
            source_times.append(source_times[-1] + duration)

        source_q = np.stack(
            [
                np.concatenate(
                    (
                        np.asarray(waypoint["left"], dtype=float),
                        np.asarray(waypoint["right"], dtype=float),
                    )
                )
                for waypoint in waypoints
            ],
            axis=0,
        )
        acceleration = np.tile(
            np.asarray(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2, dtype=float),
            2,
        )
        limits = JointHorizonLimits(
            min_position_rad=np.tile(RM65_ABSOLUTE_Q_MIN_RAD, 2),
            max_position_rad=np.tile(RM65_ABSOLUTE_Q_MAX_RAD, 2),
            max_velocity_rad_s=np.tile(joint_velocity_rad_s, 2),
            max_acceleration_rad_s2=acceleration,
            max_jerk_rad_s3=acceleration / 0.05,
        )

        def build(timestamps: np.ndarray):
            return build_offline_quintic_waypoint_trajectory(
                timestamps,
                source_q,
                command_hz=reference_hz,
                limits=limits,
                timing_mode=OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
                max_time_scale=16.0,
            )

        try:
            continuous = build(np.asarray(source_times, dtype=float))
        except OfflineWaypointTrajectoryInfeasibleError as error:
            raise RM65GraspTrajectoryPlanningError(
                "trajectory_parameterization_failed",
                details={
                    "parameterization": (
                        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
                    ),
                    "error": str(error),
                },
            ) from error

        # Joint-space extrema are certified analytically by the shared C2
        # builder.  TCP speed is nonlinear in q, so certify the sampled curve
        # and conservatively retime the complete leg if needed.
        def required_tcp_time_scale(candidate) -> float:
            tcp_poses = [
                self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=np.asarray(row[:6], dtype=float).copy(),
                    right_qpos=np.asarray(row[6:], dtype=float).copy(),
                )[moving_side]
                for row in candidate.q_rad
            ]
            required = 1.0
            for index, (start_tcp, end_tcp) in enumerate(pairwise(tcp_poses)):
                dt = float(
                    candidate.timestamps[index + 1]
                    - candidate.timestamps[index]
                )
                translation_m, rotation_deg = pose_delta_metrics(
                    start_tcp,
                    end_tcp,
                )
                required = max(
                    required,
                    translation_m
                    / (dt * self.config.linear_speed_m_s * speed_scale),
                    rotation_deg
                    / (dt * self.config.angular_speed_deg_s * speed_scale),
                )
            return float(required)

        tcp_time_scale = 1.0
        for _ in range(3):
            required_tcp_scale = required_tcp_time_scale(continuous)
            if required_tcp_scale <= 1.0 + 1e-6:
                break
            incremental_scale = required_tcp_scale * 1.01
            tcp_time_scale *= incremental_scale
            continuous = build(
                np.asarray(continuous.waypoint_timestamps, dtype=float)
                * incremental_scale
            )
        else:
            raise RM65GraspTrajectoryPlanningError(
                "trajectory_parameterization_failed",
                details={
                    "parameterization": (
                        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
                    ),
                    "reason": "sampled_tcp_speed_limit_not_certified",
                    "required_tcp_time_scale": required_tcp_time_scale(
                        continuous
                    ),
                },
            )
        return continuous, tuple(segment_diagnostics), float(tcp_time_scale)

    def _segmented_leg_trajectory(
        self,
        *,
        waypoints: tuple[Mapping[str, np.ndarray], ...],
        moving_side: str,
        speed_scale: float,
        joint_velocity_rad_s: np.ndarray,
        reference_hz: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        """Build a path-preserving stop-to-stop trajectory for one leg.

        This is the bounded fallback when C2 corner rounding leaves a route
        that was already admitted in joint space.  Every output sample stays
        on one validated joint-space edge; only the affected semantic leg is
        downgraded, so unrelated approach legs remain continuous.
        """

        if len(waypoints) < 2:
            raise ValueError("segmented trajectory leg requires at least two waypoints.")
        times = [0.0]
        q_rows = [
            np.concatenate(
                (
                    np.asarray(waypoints[0]["left"], dtype=float),
                    np.asarray(waypoints[0]["right"], dtype=float),
                )
            )
        ]
        diagnostics: list[dict[str, object]] = []
        elapsed = 0.0
        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            base_duration, diagnostic = self._segment_duration_diagnostics(
                start=start,
                end=end,
                moving_side=moving_side,
                speed_scale=speed_scale,
                joint_velocity_rad_s=joint_velocity_rad_s,
            )
            singularity_scale = self._singularity_interval_scale(
                np.asarray(start[moving_side], dtype=float),
                np.asarray(end[moving_side], dtype=float),
            )
            duration = base_duration * singularity_scale
            diagnostic.update(
                {
                    "segment_index": segment_index,
                    "singularity_time_scale": float(singularity_scale),
                    "scaled_duration_sec": float(duration),
                }
            )
            diagnostics.append(diagnostic)
            maximum_delta = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(end[side], dtype=float)
                            - np.asarray(start[side], dtype=float)
                        )
                    )
                )
                for side in _SIDES
            )
            frame_count = max(
                2,
                math.ceil(duration * reference_hz) + 1,
                math.ceil(maximum_delta / self.config.final_edge_resolution_rad) + 1,
            )
            for frame_index in range(1, frame_count):
                unit = float(frame_index) / float(frame_count - 1)
                progress = unit**3 * (10.0 - 15.0 * unit + 6.0 * unit**2)
                elapsed += duration / float(frame_count - 1)
                times.append(elapsed)
                q_rows.append(
                    np.concatenate(
                        tuple(
                            np.asarray(start[side], dtype=float)
                            + progress
                            * (
                                np.asarray(end[side], dtype=float)
                                - np.asarray(start[side], dtype=float)
                            )
                            for side in _SIDES
                        )
                    )
                )
        return (
            np.asarray(times, dtype=float),
            np.asarray(q_rows, dtype=float),
            tuple(diagnostics),
        )

    def _apply_singularity_replay_time_scaling(
        self,
        *,
        times: list[float],
        samples: Mapping[str, list[np.ndarray]],
        moving_side: str,
    ) -> tuple[list[float], dict[str, object]]:
        """Slow only the local singular traversal; never reject it.

        The spatial path has already passed the production collision checks.
        This pass changes timing only, mirroring the real-teleop soft
        singularity policy while keeping the path and endpoint unchanged.
        """
        if moving_side not in _SIDES:
            raise ValueError("moving_side must be left or right.")
        if len(times) < 2:
            return list(times), {
                "policy": "real_teleop_soft_traversal_v1",
                "scaled_interval_count": 0,
                "spatial_path_unchanged": True,
                "collision_rechecked": True,
            }
        qpos = np.asarray(samples[moving_side], dtype=float)
        if qpos.shape != (len(times), 6) or not np.all(np.isfinite(qpos)):
            raise ValueError("singularity timing samples must be finite sample_count x 6.")
        hard_deg = float(RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG)
        soft_deg = float(self.config.singularity_soft_margin_deg)
        slowdown = float(self.config.singularity_replay_slowdown)
        hard_rad = np.deg2rad(hard_deg)
        soft_rad = np.deg2rad(soft_deg)
        max_scale = 1.0 / slowdown
        scaled_intervals = 0
        maximum_scale = 1.0
        scaled_times = [0.0]
        for index in range(1, len(times)):
            distance_rad = min(
                abs(float(qpos[index - 1, 2])),
                abs(float(qpos[index - 1, 4])),
                abs(float(qpos[index, 2])),
                abs(float(qpos[index, 4])),
            )
            if distance_rad <= hard_rad:
                interval_scale = max_scale
            elif distance_rad < soft_rad:
                fraction = (soft_rad - distance_rad) / (soft_rad - hard_rad)
                interval_scale = 1.0 + fraction * (max_scale - 1.0)
            else:
                interval_scale = 1.0
            if interval_scale > 1.0 + 1e-12:
                scaled_intervals += 1
                maximum_scale = max(maximum_scale, interval_scale)
            delta = float(times[index]) - float(times[index - 1])
            if not math.isfinite(delta) or delta <= 0.0:
                raise ValueError("trajectory times must be finite and increasing.")
            scaled_times.append(scaled_times[-1] + delta * interval_scale)
        return scaled_times, {
            "policy": "real_teleop_soft_traversal_v1",
            "hard_threshold_deg": hard_deg,
            "soft_margin_deg": soft_deg,
            "replay_slowdown": slowdown,
            "scaled_interval_count": int(scaled_intervals),
            "maximum_interval_scale": float(maximum_scale),
            "spatial_path_unchanged": True,
            "collision_rechecked": True,
        }

    def _visible_grasp_collision_diagnostics(
        self,
        visible_scene: VisibleGraspScene,
        *,
        grasp_tool_tcp_world: np.ndarray,
        distance_m: float,
    ) -> dict[str, object]:
        """Classify terminal and straight-ingress observations without routing.

        The analytic straight sweep is a useful local route diagnostic, but it
        is not a proof that every route to the same grasp target is blocked.
        Keep the terminal goal check hard while allowing the joint path search
        to look for a different ingress when only the fixed Cartesian sweep
        intersects an observed point.
        """

        opening_m = visible_scene.geometry.maximum_opening_m
        (
            terminal_target_noncontact,
            terminal_target_contact,
            terminal_background,
        ) = visible_scene.collision_counts_with_terminal_target_contact(
            grasp_tool_tcp_world,
            opening_m=opening_m,
            contact_depth_m=RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
        )
        approach_target, approach_background = (
            visible_scene.approach_collision_counts(
                grasp_tool_tcp_world,
                opening_m=opening_m,
                distance_m=distance_m,
                allow_terminal_target_contact=True,
                terminal_target_contact_depth_m=(
                    RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M
                ),
            )
        )
        return {
            "distance_m": float(distance_m),
            "opening_m": float(opening_m),
            "terminal_target_noncontact_points": int(terminal_target_noncontact),
            "terminal_target_contact_points": int(terminal_target_contact),
            "terminal_background_points": int(terminal_background),
            "approach_target_noncontact_points": int(approach_target),
            "approach_background_points": int(approach_background),
            "straight_ingress_target_noncontact_points": int(
                max(0, approach_target - terminal_target_noncontact)
            ),
            "straight_ingress_background_points": int(
                max(0, approach_background - terminal_background)
            ),
            "terminal_collision_blocked": bool(
                terminal_target_noncontact or terminal_background
            ),
            "straight_ingress_blocked": bool(approach_target or approach_background),
            "straight_ingress_only": bool(
                (approach_target or approach_background)
                and not (terminal_target_noncontact or terminal_background)
            ),
        }

    def _constrained_approach_targets(
        self,
        *,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return fixed-orientation Cartesian targets after the pregrasp pose."""

        pregrasp = np.asarray(pregrasp_tool_tcp_world, dtype=float)
        grasp = np.asarray(grasp_tool_tcp_world, dtype=float)
        distance = float(np.linalg.norm(grasp[:3, 3] - pregrasp[:3, 3]))
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("pregrasp and grasp positions must be distinct and finite.")
        steps = max(
            1,
            math.ceil(distance / self.config.approach_waypoint_spacing_m),
        )
        targets: list[np.ndarray] = []
        for index in range(1, steps + 1):
            fraction = float(index) / float(steps)
            waypoint = grasp.copy()
            waypoint[:3, 3] = (
                pregrasp[:3, 3]
                + fraction * (grasp[:3, 3] - pregrasp[:3, 3])
            )
            targets.append(waypoint)
        return tuple(targets)

    def _validate_constrained_approach_geometry(
        self,
        waypoints: tuple[Mapping[str, np.ndarray], ...],
        *,
        moving_side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> dict[str, object]:
        """Fail closed unless the realized TCP stays on the intended approach ray."""

        pregrasp = np.asarray(pregrasp_tool_tcp_world, dtype=float)
        grasp = np.asarray(grasp_tool_tcp_world, dtype=float)
        delta = grasp[:3, 3] - pregrasp[:3, 3]
        distance = float(np.linalg.norm(delta))
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("pregrasp and grasp positions must be distinct and finite.")
        approach_axis = delta / distance
        sample_count = 0
        maximum_lateral_error_m = 0.0
        maximum_orientation_error_deg = 0.0
        maximum_reverse_progress_m = 0.0
        previous_progress_m: float | None = None

        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            maximum_delta = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(end[side], dtype=float)
                            - np.asarray(start[side], dtype=float)
                        )
                    )
                )
                for side in _SIDES
            )
            steps = max(
                1,
                math.ceil(maximum_delta / self.config.final_edge_resolution_rad),
            )
            for edge_sample_index in range(steps + 1):
                if segment_index > 0 and edge_sample_index == 0:
                    continue
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="approach_geometry_revalidation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                        },
                    )
                alpha = float(edge_sample_index) / float(steps)
                qpos = {
                    side: np.asarray(start[side], dtype=float)
                    + alpha
                    * (
                        np.asarray(end[side], dtype=float)
                        - np.asarray(start[side], dtype=float)
                    )
                    for side in _SIDES
                }
                tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=qpos["left"],
                    right_qpos=qpos["right"],
                )[moving_side]
                relative = np.asarray(tcp[:3, 3], dtype=float) - pregrasp[:3, 3]
                progress_m = float(np.dot(relative, approach_axis))
                lateral_error_m = float(
                    np.linalg.norm(relative - progress_m * approach_axis)
                )
                orientation_error_deg = math.degrees(
                    float(
                        Rotation.from_matrix(
                            grasp[:3, :3]
                            @ np.asarray(tcp[:3, :3], dtype=float).T
                        ).magnitude()
                    )
                )
                sample_count += 1
                maximum_lateral_error_m = max(
                    maximum_lateral_error_m,
                    lateral_error_m,
                )
                maximum_orientation_error_deg = max(
                    maximum_orientation_error_deg,
                    orientation_error_deg,
                )
                if (
                    progress_m < -self.config.approach_progress_tolerance_m
                    or progress_m
                    > distance + self.config.approach_progress_tolerance_m
                ):
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_progress_out_of_bounds",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                            "progress_m": progress_m,
                            "approach_distance_m": distance,
                            "tolerance_m": (
                                self.config.approach_progress_tolerance_m
                            ),
                        },
                    )
                if previous_progress_m is not None:
                    reverse_progress_m = previous_progress_m - progress_m
                    maximum_reverse_progress_m = max(
                        maximum_reverse_progress_m,
                        reverse_progress_m,
                    )
                    if reverse_progress_m > self.config.approach_progress_tolerance_m:
                        raise RM65GraspTrajectoryPlanningError(
                            "approach_progress_reversed",
                            details={
                                "segment_index": segment_index,
                                "edge_sample_index": edge_sample_index,
                                "previous_progress_m": previous_progress_m,
                                "progress_m": progress_m,
                                "reverse_progress_m": reverse_progress_m,
                                "tolerance_m": (
                                    self.config.approach_progress_tolerance_m
                                ),
                            },
                        )
                previous_progress_m = progress_m
                if (
                    lateral_error_m
                    > self.config.maximum_approach_lateral_error_m
                ):
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_lateral_deviation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                            "lateral_error_m": lateral_error_m,
                            "maximum_lateral_error_m": (
                                self.config.maximum_approach_lateral_error_m
                            ),
                        },
                    )
                if (
                    orientation_error_deg
                    > self.config.maximum_approach_orientation_error_deg
                ):
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_orientation_deviation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                            "orientation_error_deg": orientation_error_deg,
                            "maximum_orientation_error_deg": (
                                self.config.maximum_approach_orientation_error_deg
                            ),
                        },
                    )
        return {
            "sample_count": sample_count,
            "approach_distance_m": distance,
            "approach_axis_world": [float(value) for value in approach_axis],
            "maximum_lateral_error_m": maximum_lateral_error_m,
            "maximum_orientation_error_deg": maximum_orientation_error_deg,
            "maximum_reverse_progress_m": maximum_reverse_progress_m,
        }

    @staticmethod
    def _raise_ingress_rejection(
        details: Mapping[str, object],
        reason: str,
    ) -> NoReturn:
        raise RM65GraspTrajectoryPlanningError(
            "path_target_safety_envelope_blocked",
            details={
                **dict(details),
                "envelope_ingress_rejection": reason,
            },
        )

    def _ingress_projection(
        self,
        *,
        moving_side: str,
        details: Mapping[str, object],
    ) -> tuple[float, frozenset[str]]:
        if tuple(details.get("modified_sides", ())) != (moving_side,):
            self._raise_ingress_rejection(
                details,
                "modified_sides_are_not_the_moving_side_only",
            )
        projection_by_side = details.get("envelope_projection_by_side")
        projection = (
            projection_by_side.get(moving_side)
            if isinstance(projection_by_side, Mapping)
            else None
        )
        if not isinstance(projection, Mapping):
            self._raise_ingress_rejection(
                details,
                "projection_diagnostics_unavailable",
            )
        if (
            projection.get("translation_modified") is not True
            or projection.get("rotation_modified") is not False
        ):
            self._raise_ingress_rejection(
                details,
                "violation_is_not_translation_only",
            )
        adjustment = projection.get("translation_adjustment_m")
        if (
            isinstance(adjustment, bool)
            or not isinstance(adjustment, (int, float))
            or not math.isfinite(float(adjustment))
            or float(adjustment) <= 0.0
        ):
            self._raise_ingress_rejection(
                details,
                "translation_adjustment_is_invalid",
            )
        raw_constraints = projection.get("violated_translation_constraints")
        if (
            not isinstance(raw_constraints, (tuple, list))
            or not raw_constraints
            or any(
                not isinstance(item, str) or not item.startswith("absolute:")
                for item in raw_constraints
            )
        ):
            self._raise_ingress_rejection(
                details,
                "violation_is_not_an_absolute_translation_bound",
            )
        return float(adjustment), frozenset(raw_constraints)

    def _bounded_ingress_projection(
        self,
        *,
        moving_side: str,
        details: Mapping[str, object],
    ) -> tuple[float, frozenset[str]]:
        adjustment, constraints = self._ingress_projection(
            moving_side=moving_side, details=details,
        )
        if len(constraints) != 1:
            self._raise_ingress_rejection(
                details,
                "multiple_initial_constraints_are_not_supported",
            )
        if float(adjustment) > self.config.maximum_path_envelope_ingress_distance_m:
            self._raise_ingress_rejection(
                {
                    **dict(details),
                    "maximum_path_envelope_ingress_distance_m": (
                        self.config.maximum_path_envelope_ingress_distance_m
                    ),
                    "path_envelope_ingress_distance_excess_m": (
                        float(adjustment)
                        - self.config.maximum_path_envelope_ingress_distance_m
                    ),
                },
                "initial_distance_exceeds_limit",
            )
        return adjustment, constraints

    def _require_ingress_progress(
        self,
        *,
        details: Mapping[str, object],
        adjustment_m: float,
        constraints: frozenset[str],
        initial_adjustment_m: float,
        initial_constraints: frozenset[str],
        best_adjustment_m: float,
    ) -> float:
        if (constraints - initial_constraints
                and adjustment_m > self.config.path_envelope_ingress_tangent_tolerance_m):
            self._raise_ingress_rejection(details, "violation_constraint_set_expanded")
        tolerance = self.config.path_envelope_ingress_progress_tolerance_m
        if adjustment_m > initial_adjustment_m + tolerance:
            self._raise_ingress_rejection(details, "path_exceeded_initial_outer_boundary")
        if adjustment_m > best_adjustment_m + tolerance:
            self._raise_ingress_rejection(details, "path_moved_away_from_envelope")
        return min(best_adjustment_m, adjustment_m)

    def _ingress_search_envelope_check(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
    ) -> Callable[[Mapping[str, np.ndarray]], None]:
        initial: tuple[float, frozenset[str]] | None = None
        try:
            self._require_qpos_within_target_envelope(qpos_by_side, sides=_SIDES, phase="path_target")
        except RM65GraspTrajectoryPlanningError as error:
            if error.reason_code != "path_target_safety_envelope_blocked":
                raise
            initial = self._bounded_ingress_projection(moving_side=side, details=error.details)

        def check(qpos: Mapping[str, np.ndarray]) -> None:
            try:
                self._require_qpos_within_target_envelope(qpos, sides=_SIDES, phase="path_target")
            except RM65GraspTrajectoryPlanningError as error:
                if error.reason_code != "path_target_safety_envelope_blocked":
                    raise
                if initial is None:
                    self._raise_ingress_rejection(error.details, "path_did_not_start_at_the_violation")
                adjustment, constraints = self._ingress_projection(moving_side=side, details=error.details)
                initial_adjustment, initial_constraints = initial
                # BiRRT queries are unordered. Bound samples against the fixed
                # start, never mutate progress on a random oracle query. Global
                # best progress and no re-exit remain final path-order checks.
                self._require_ingress_progress(
                    details=error.details, adjustment_m=adjustment, constraints=constraints,
                    initial_adjustment_m=initial_adjustment, initial_constraints=initial_constraints,
                    best_adjustment_m=initial_adjustment,
                )

        return check

    def _preflight_start_envelope(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
    ) -> None:
        """Reject a candidate-independent start violation before searching IK."""
        if self.target_envelope_filter is None:
            return
        targets = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"], right_qpos=qpos_by_side["right"]
        )
        projected = self._project_target_envelope(targets, phase="path_target")
        modified = tuple(
            name for name in _SIDES
            if not np.allclose(projected[name], targets[name], atol=1e-9, rtol=0.0)
        )
        if not modified:
            return
        details = {
            "modified_sides": modified,
            "envelope_projection_subject": "current_measured_tool_tcp_in_envelope_reference_frame",
            **self._target_envelope_projection_details(modified),
        }
        try:
            self._bounded_ingress_projection(moving_side=side, details=details)
        except RM65GraspTrajectoryPlanningError as error:
            raise RM65GraspTrajectoryPlanningError(
                "start_state_safety_envelope_blocked", details=error.details
            ) from error

    def _envelope_ingress_targets(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        final_tool_tcp_world: np.ndarray,
    ) -> tuple[tuple[float, np.ndarray], ...]:
        if self.target_envelope_filter is None:
            return ()
        solver_targets = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
        )
        projected = self._project_target_envelope(
            solver_targets,
            phase="path_target",
        )
        modified_sides = tuple(
            name
            for name in _SIDES
            if not np.allclose(
                projected[name],
                np.asarray(solver_targets[name], dtype=float),
                atol=1e-9,
                rtol=0.0,
            )
        )
        if not modified_sides:
            return ()
        projection_details = {
            "modified_sides": modified_sides,
            "envelope_projection_subject": "current_measured_tool_tcp_in_envelope_reference_frame",
            **self._target_envelope_projection_details(modified_sides),
        }
        self._bounded_ingress_projection(
            moving_side=side,
            details=projection_details,
        )

        final_solver_targets = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in solver_targets.items()
        }
        final_solver_targets[side] = (
            self.target_builder.solver_world_target_from_tool_tcp_world_target(
                side=side,
                tool_tcp_world_target=final_tool_tcp_world,
            )
        )
        self._require_unmodified_target_envelope(
            final_solver_targets,
            sides=(side,),
            phase="target",
        )

        boundary_solver = np.asarray(projected[side], dtype=float).reshape(4, 4).copy()
        boundary_tool = np.asarray(
            self.target_builder.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=boundary_solver,
            ),
            dtype=float,
        ).reshape(4, 4)
        final_tool = np.asarray(final_tool_tcp_world, dtype=float).reshape(4, 4).copy()
        targets: list[tuple[float, np.ndarray]] = []
        for fraction in self.config.path_envelope_ingress_waypoint_fractions:
            # The envelope is evaluated on the actual tool TCP. Interpolating
            # solver-endpoint translation is incorrect when the fixed tool
            # offset rotates with the endpoint orientation: the TCP can leave
            # the envelope even though both endpoint poses are admissible.
            ingress_tool = boundary_tool.copy()
            ingress_tool[:3, 3] = (
                boundary_tool[:3, 3]
                + fraction * (final_tool[:3, 3] - boundary_tool[:3, 3])
            )
            ingress_solver = np.asarray(
                self.target_builder.solver_world_target_from_tool_tcp_world_target(
                    side=side,
                    tool_tcp_world_target=ingress_tool,
                ),
                dtype=float,
            ).reshape(4, 4)
            ingress_pair = {
                name: np.asarray(projected[name], dtype=float).copy() for name in _SIDES
            }
            ingress_pair[side] = ingress_solver
            try:
                self._require_unmodified_target_envelope(
                    ingress_pair,
                    sides=_SIDES,
                    phase="ingress_waypoint",
                    details={"inward_fraction": fraction},
                )
            except RM65GraspTrajectoryPlanningError as error:
                if error.reason_code == "ingress_waypoint_safety_envelope_blocked":
                    continue
                raise
            targets.append((fraction, ingress_tool.copy()))
        if not targets:
            raise RM65GraspTrajectoryPlanningError(
                "path_target_safety_envelope_blocked",
                details={
                    **projection_details,
                    "modified_sides": modified_sides,
                    "envelope_ingress_rejection": (
                        "no_strictly_inside_ingress_waypoint"
                    ),
                },
            )
        return tuple(targets)

    def _endpoint_qpos_candidates(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        prefer_closest_tolerance: bool = True,
        orientation_tolerance_deg: float | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> tuple[_EndpointIKCandidate, ...]:
        with self._solver_lock:
            return self._endpoint_qpos_candidates_locked(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_tool_tcp_world,
                prefer_closest_tolerance=prefer_closest_tolerance,
                orientation_tolerance_deg=orientation_tolerance_deg,
                deadline=deadline,
                monotonic=monotonic,
            )

    def _endpoint_seed_qpos(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        *,
        side: str,
        seed_count: int | None = None,
    ) -> tuple[dict[str, np.ndarray], ...]:
        # Retain the measured posture and nearby basins first so the shortest
        # solution wins when it exists.  The deterministic Halton tail covers
        # the full bounded RM65 joint domain; arbitrary grasp poses routinely
        # have valid branches far outside a 25-degree neighborhood of the live
        # posture.
        patterns = np.asarray(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, -1.0, 0.0, 0.0, 0.0),
                (0.0, -1.0, 1.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0, -1.0, 0.0, 1.0),
                (0.0, 0.0, -1.0, 1.0, 0.0, -1.0),
            ),
            dtype=float,
        )
        lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
        upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
        baseline = {
            name: np.asarray(qpos_by_side[name], dtype=float).copy() for name in _SIDES
        }
        maximum = (
            self.config.endpoint_seed_count if seed_count is None else int(seed_count)
        )
        if not 1 <= maximum <= 64:
            raise ValueError("seed_count must be in [1, 64].")
        seeds: list[dict[str, np.ndarray]] = []
        local_values = [
            np.clip(
                baseline[side] + pattern * self.config.endpoint_seed_perturbation_rad,
                lower,
                upper,
            )
            for pattern in patterns
        ]
        default_by_side = {
            "left": np.asarray(DEFAULT_LEFT_ARM_QPOS, dtype=float),
            "right": np.asarray(DEFAULT_RIGHT_ARM_QPOS, dtype=float),
        }
        local_values.insert(1, np.clip(default_by_side[side], lower, upper))
        for values in local_values:
            candidate = {name: value.copy() for name, value in baseline.items()}
            candidate[side] = np.asarray(values, dtype=float).copy()
            if any(
                _joint_proposals_match(candidate[side], existing[side], atol=1e-9)
                for existing in seeds
            ):
                continue
            seeds.append(candidate)
            if len(seeds) == maximum:
                return tuple(seeds)

        for sample_index in range(1, maximum * 2 + 1):
            unit = np.asarray(
                [
                    self._radical_inverse(sample_index, prime)
                    for prime in (2, 3, 5, 7, 11, 13)
                ],
                dtype=float,
            )
            candidate = {name: value.copy() for name, value in baseline.items()}
            candidate[side] = lower + unit * (upper - lower)
            if any(
                _joint_proposals_match(candidate[side], existing[side], atol=1e-9)
                for existing in seeds
            ):
                continue
            seeds.append(candidate)
            if len(seeds) == maximum:
                break
        return tuple(seeds)

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        value = 0.0
        denominator = 1.0
        current = int(index)
        while current > 0:
            current, remainder = divmod(current, int(base))
            denominator *= float(base)
            value += float(remainder) / denominator
        return value

    def _least_squares_endpoint_from_seed(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        seed_qpos: np.ndarray,
        maximum_evaluations: int,
        prefer_closest_tolerance: bool = True,
    ) -> tuple[dict[str, np.ndarray], int, float, float]:
        target_pose = np.asarray(target_tool_tcp_world, dtype=float)
        lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
        upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
        position_scale = self.config.endpoint_position_residual_scale_m
        orientation_scale = math.radians(
            self.config.endpoint_orientation_residual_scale_deg
        )

        def qpos_pair(values: np.ndarray) -> dict[str, np.ndarray]:
            pair = {
                name: np.asarray(qpos_by_side[name], dtype=float).copy()
                for name in _SIDES
            }
            pair[side] = np.asarray(values, dtype=float).copy()
            return pair

        def residual(values: np.ndarray) -> np.ndarray:
            pair = qpos_pair(values)
            tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=pair["left"],
                right_qpos=pair["right"],
            )[side]
            rotation_error = Rotation.from_matrix(
                target_pose[:3, :3] @ tcp[:3, :3].T
            ).as_rotvec()
            return np.concatenate(
                (
                    (target_pose[:3, 3] - tcp[:3, 3]) / position_scale,
                    rotation_error / orientation_scale,
                )
            )

        solved = least_squares(
            residual,
            np.asarray(seed_qpos, dtype=float),
            bounds=(lower, upper),
            max_nfev=int(maximum_evaluations),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        candidate = qpos_pair(np.asarray(solved.x, dtype=float))
        tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=candidate["left"],
            right_qpos=candidate["right"],
        )[side]
        position, orientation = pose_delta_metrics(tcp, target_pose)
        if prefer_closest_tolerance and (
            position <= self.config.endpoint_position_tolerance_m
            and orientation <= self.config.endpoint_orientation_tolerance_deg
        ):
            candidate, position, orientation = self._closest_tolerance_endpoint(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_pose,
                seed_qpos=np.asarray(seed_qpos, dtype=float),
                solved_qpos=np.asarray(candidate[side], dtype=float),
            )
        return candidate, int(solved.nfev), position, orientation

    def _closest_tolerance_endpoint(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        seed_qpos: np.ndarray,
        solved_qpos: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, float]:
        """Keep the earliest seed-to-solution point inside a strict margin.

        A least-squares optimum commonly drives numerical error to zero even
        though the motion contract only requires a bounded endpoint.  Backing
        toward the seed preserves the same IK branch and materially shortens
        the subsequent collision-aware path without relaxing acceptance.
        """

        position_limit = (
            self.config.endpoint_position_tolerance_m
            * self.config.endpoint_tolerance_margin_ratio
        )
        orientation_limit = (
            self.config.endpoint_orientation_tolerance_deg
            * self.config.endpoint_tolerance_margin_ratio
        )
        target_pose = np.asarray(target_tool_tcp_world, dtype=float)
        seed = np.asarray(seed_qpos, dtype=float)
        solved = np.asarray(solved_qpos, dtype=float)

        def evaluate(alpha: float) -> tuple[dict[str, np.ndarray], float, float]:
            pair = {
                name: np.asarray(qpos_by_side[name], dtype=float).copy()
                for name in _SIDES
            }
            pair[side] = seed + float(alpha) * (solved - seed)
            tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=pair["left"],
                right_qpos=pair["right"],
            )[side]
            position, orientation = pose_delta_metrics(tcp, target_pose)
            return pair, position, orientation

        def acceptable(
            pair: Mapping[str, np.ndarray],
            position: float,
            orientation: float,
        ) -> bool:
            return (
                position <= position_limit
                and orientation <= orientation_limit
                and self._qpos_is_within_target_envelope(
                    pair,
                    sides=(side,),
                    phase="endpoint",
                )
            )

        previous_alpha = 0.0
        bracket: tuple[float, float] | None = None
        for alpha in np.linspace(0.0, 1.0, 33)[1:]:
            pair, position, orientation = evaluate(float(alpha))
            if acceptable(pair, position, orientation):
                bracket = (previous_alpha, float(alpha))
                break
            previous_alpha = float(alpha)
        if bracket is None:
            return evaluate(1.0)
        lower, upper = bracket
        for _ in range(32):
            midpoint = (lower + upper) * 0.5
            pair, position, orientation = evaluate(midpoint)
            if acceptable(pair, position, orientation):
                upper = midpoint
            else:
                lower = midpoint
        return evaluate(upper)

    def _endpoint_qpos_candidates_locked(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        prefer_closest_tolerance: bool = True,
        orientation_tolerance_deg: float | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> tuple[_EndpointIKCandidate, ...]:
        solver_targets = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
        )
        solver_targets[side] = (
            self.target_builder.solver_world_target_from_tool_tcp_world_target(
                side=side,
                tool_tcp_world_target=target_tool_tcp_world,
            )
        )
        self._require_unmodified_target_envelope(
            solver_targets,
            sides=(side,),
            phase="target",
        )
        effective_orientation_tolerance_deg = (
            self.config.endpoint_orientation_tolerance_deg
            if orientation_tolerance_deg is None
            else float(orientation_tolerance_deg)
        )
        if (
            not math.isfinite(effective_orientation_tolerance_deg)
            or effective_orientation_tolerance_deg <= 0.0
        ):
            raise ValueError(
                "orientation_tolerance_deg must be finite and positive."
            )
        inactive = "right" if side == "left" else "left"
        attempts: list[dict[str, object]] = []
        accepted: list[_EndpointIKCandidate] = []
        clearance_rescue_seeds: list[
            tuple[float, int, dict[str, np.ndarray]]
        ] = []
        collision_unavailable = False
        collision_blocked = False
        clearance_blocked = False
        endpoint_envelope_blocked = False
        for seed_index, seed in enumerate(
            self._endpoint_seed_qpos(qpos_by_side, side=side)
        ):
            if deadline is not None and float(monotonic()) >= deadline:
                if accepted:
                    break
                raise RM65GraspTrajectoryPlanningError(
                    "path_planning_time_budget_exhausted",
                    details={
                        "planning_phase": "endpoint_ik",
                        "time_budget_sec": (self.config.path_planning_time_budget_sec),
                        "endpoint_seed_count_attempted": len(attempts),
                        "attempts": attempts,
                    },
                )
            outcome = "ik_search_exhausted"
            evaluations = 0
            collision_detail: dict[str, object] = {}
            try:
                candidate, evaluations, position, orientation = (
                    self._least_squares_endpoint_from_seed(
                        qpos_by_side=qpos_by_side,
                        side=side,
                        target_tool_tcp_world=target_tool_tcp_world,
                        seed_qpos=seed[side],
                        maximum_evaluations=(self.config.maximum_endpoint_solve_cycles),
                        prefer_closest_tolerance=prefer_closest_tolerance,
                    )
                )
                candidate[inactive] = np.asarray(
                    qpos_by_side[inactive], dtype=float
                ).copy()
            except RM65GraspTrajectoryPlanningError:
                raise
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                attempts.append(
                    {
                        "seed_index": seed_index,
                        "solve_cycles": evaluations,
                        "outcome": "ik_solver_error",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            if (
                position <= self.config.endpoint_position_tolerance_m
                and orientation <= effective_orientation_tolerance_deg
            ):
                if not self._qpos_is_within_target_envelope(
                    candidate,
                    sides=(side,),
                    phase="endpoint",
                ):
                    endpoint_envelope_blocked = True
                    outcome = "endpoint_safety_envelope_blocked"
                elif any(
                    _joint_proposals_match(
                        candidate[side],
                        item.qpos_by_side[side],
                        atol=1e-5,
                    )
                    for item in accepted
                ):
                    outcome = "duplicate_exact_solution"
                else:
                    check = self.oracle.check(candidate)
                    if not check.collision_check_available:
                        collision_unavailable = True
                        outcome = "endpoint_collision_check_unavailable"
                    elif not check.safe:
                        positive_gap = (
                            check.minimum_distance_m is not None
                            and check.minimum_distance_m > 0.0
                        )
                        if positive_gap:
                            clearance_blocked = True
                            outcome = "endpoint_clearance_blocked"
                            clearance_rescue_seeds.append(
                                (
                                    float(check.minimum_distance_m),
                                    seed_index,
                                    {
                                        name: np.asarray(values, dtype=float).copy()
                                        for name, values in candidate.items()
                                    },
                                )
                            )
                        else:
                            collision_blocked = True
                            outcome = "endpoint_collision_blocked"
                        collision_detail = {
                            "closest_geometry_names": check.metadata.get(
                                "closest_geometry_names"
                            ),
                            "closest_obstacle_id": check.metadata.get(
                                "closest_obstacle_id"
                            ),
                            "minimum_distance_m": check.minimum_distance_m,
                        }
                    else:
                        accepted.append(
                            _EndpointIKCandidate(
                                qpos_by_side=candidate,
                                seed_index=seed_index,
                                solve_cycles=evaluations,
                                position_residual_m=position,
                                orientation_residual_deg=orientation,
                            )
                        )
                        outcome = "accepted"
            attempts.append(
                {
                    "seed_index": seed_index,
                    "solve_cycles": evaluations,
                    "outcome": outcome,
                    "best_position_residual_m": position,
                    "best_orientation_residual_deg": orientation,
                    **collision_detail,
                }
            )
        # Pose-only least squares converges to the exact target and therefore
        # cannot use the already-authorized endpoint tolerance to improve a
        # positive but insufficient clearance.  Starting from the best such
        # branch, probe a deterministic set of nearby target poses.  Every
        # accepted result is still measured against the original requested
        # pose, the same envelope, and the unchanged production collision
        # clearance.  True penetrations do not activate this rescue.
        if accepted == [] and clearance_rescue_seeds and (
            deadline is None or float(monotonic()) < deadline
        ):
            _distance, source_seed_index, rescue_seed = max(
                clearance_rescue_seeds,
                key=lambda item: item[0],
            )
            for rescue_index, (rescue_kind, rescue_target) in enumerate(
                self._clearance_rescue_target_poses(
                    target_tool_tcp_world,
                    orientation_tolerance_deg=(
                        effective_orientation_tolerance_deg
                    ),
                )
            ):
                if deadline is not None and float(monotonic()) >= deadline:
                    break
                rescue_attempt: dict[str, object] = {
                    "seed_index": source_seed_index,
                    "clearance_rescue_index": rescue_index,
                    "clearance_rescue_kind": rescue_kind,
                }
                try:
                    candidate, evaluations, _position, _orientation = (
                        self._least_squares_endpoint_from_seed(
                            qpos_by_side=qpos_by_side,
                            side=side,
                            target_tool_tcp_world=rescue_target,
                            seed_qpos=rescue_seed[side],
                            maximum_evaluations=min(
                                self.config.maximum_endpoint_solve_cycles,
                                60,
                            ),
                            prefer_closest_tolerance=False,
                        )
                    )
                    candidate[inactive] = np.asarray(
                        qpos_by_side[inactive], dtype=float
                    ).copy()
                    realized = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                        left_qpos=candidate["left"],
                        right_qpos=candidate["right"],
                    )[side]
                    position, orientation = pose_delta_metrics(
                        realized,
                        target_tool_tcp_world,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    rescue_attempt.update(
                        {
                            "solve_cycles": 0,
                            "outcome": "clearance_rescue_solver_error",
                            "error_type": type(exc).__name__,
                        }
                    )
                    attempts.append(rescue_attempt)
                    continue
                rescue_attempt.update(
                    {
                        "solve_cycles": evaluations,
                        "best_position_residual_m": position,
                        "best_orientation_residual_deg": orientation,
                    }
                )
                if (
                    position > self.config.endpoint_position_tolerance_m
                    or orientation > effective_orientation_tolerance_deg
                ):
                    rescue_attempt["outcome"] = (
                        "clearance_rescue_pose_tolerance_exceeded"
                    )
                    attempts.append(rescue_attempt)
                    continue
                if not self._qpos_is_within_target_envelope(
                    candidate,
                    sides=(side,),
                    phase="endpoint",
                ):
                    endpoint_envelope_blocked = True
                    rescue_attempt["outcome"] = (
                        "endpoint_safety_envelope_blocked"
                    )
                    attempts.append(rescue_attempt)
                    continue
                check = self.oracle.check(candidate)
                rescue_attempt.update(
                    {
                        "closest_geometry_names": check.metadata.get(
                            "closest_geometry_names"
                        ),
                        "closest_obstacle_id": check.metadata.get(
                            "closest_obstacle_id"
                        ),
                        "minimum_distance_m": check.minimum_distance_m,
                    }
                )
                if not check.collision_check_available:
                    collision_unavailable = True
                    rescue_attempt["outcome"] = (
                        "endpoint_collision_check_unavailable"
                    )
                elif not check.safe:
                    positive_gap = (
                        check.minimum_distance_m is not None
                        and check.minimum_distance_m > 0.0
                    )
                    if positive_gap:
                        clearance_blocked = True
                        rescue_attempt["outcome"] = (
                            "endpoint_clearance_blocked"
                        )
                    else:
                        collision_blocked = True
                        rescue_attempt["outcome"] = (
                            "endpoint_collision_blocked"
                        )
                else:
                    accepted.append(
                        _EndpointIKCandidate(
                            qpos_by_side=candidate,
                            seed_index=source_seed_index,
                            solve_cycles=evaluations,
                            position_residual_m=position,
                            orientation_residual_deg=orientation,
                        )
                    )
                    rescue_attempt["outcome"] = (
                        "accepted_clearance_rescue"
                    )
                attempts.append(rescue_attempt)
                if accepted:
                    break
        if deadline is not None and not accepted and float(monotonic()) >= deadline:
            raise RM65GraspTrajectoryPlanningError(
                "path_planning_time_budget_exhausted",
                details={
                    "planning_phase": "endpoint_ik",
                    "time_budget_sec": self.config.path_planning_time_budget_sec,
                    "endpoint_seed_count_attempted": len(attempts),
                    "attempts": attempts,
                },
            )
        if accepted:
            current = np.asarray(qpos_by_side[side], dtype=float)
            ordered = self._ordered_endpoint_candidates(
                candidates=tuple(accepted),
                current_qpos=current,
                side=side,
            )
            return ordered[: self.config.maximum_endpoint_candidates]
        if collision_unavailable:
            reason = "endpoint_collision_check_unavailable"
        elif endpoint_envelope_blocked:
            reason = "endpoint_safety_envelope_blocked"
        elif clearance_blocked:
            reason = "endpoint_clearance_blocked"
        elif collision_blocked:
            reason = "endpoint_collision_blocked"
        else:
            reason = "ik_search_exhausted"
        residual_attempts = [
            item
            for item in attempts
            if item.get("best_position_residual_m") is not None
            and item.get("best_orientation_residual_deg") is not None
        ]
        if not residual_attempts:
            raise RM65GraspTrajectoryPlanningError(
                "ik_solver_error",
                details={
                    "cycles": sum(int(item["solve_cycles"]) for item in attempts),
                    "endpoint_seed_count": len(attempts),
                    "maximum_cycles_per_seed": (
                        self.config.maximum_endpoint_solve_cycles
                    ),
                    "endpoint_solver": "scipy_least_squares_fk_multistart",
                    "solve_unit": "function_evaluations",
                    "attempts": attempts,
                },
            )
        best_attempt = min(
            residual_attempts,
            key=lambda item: max(
                float(item["best_position_residual_m"])
                / self.config.endpoint_position_tolerance_m,
                float(item["best_orientation_residual_deg"])
                / effective_orientation_tolerance_deg,
            ),
        )
        raise RM65GraspTrajectoryPlanningError(
            reason,
            details={
                "cycles": sum(int(item["solve_cycles"]) for item in attempts),
                "best_position_residual_m": best_attempt["best_position_residual_m"],
                "best_orientation_residual_deg": best_attempt[
                    "best_orientation_residual_deg"
                ],
                "endpoint_seed_count": len(attempts),
                "maximum_cycles_per_seed": (self.config.maximum_endpoint_solve_cycles),
                "endpoint_solver": "scipy_least_squares_fk_multistart",
                "solve_unit": "function_evaluations",
                "position_tolerance_m": self.config.endpoint_position_tolerance_m,
                "orientation_tolerance_deg": effective_orientation_tolerance_deg,
                "attempts": attempts,
                "collision_examples": [
                    {
                        "seed_index": item["seed_index"],
                        "closest_geometry_names": item["closest_geometry_names"],
                        "closest_obstacle_id": item["closest_obstacle_id"],
                        "minimum_distance_m": item["minimum_distance_m"],
                    }
                    for item in attempts
                    if item.get("outcome") in {
                        "endpoint_collision_blocked",
                        "endpoint_clearance_blocked",
                    }
                ][:3],
            },
        )

    def _clearance_rescue_target_poses(
        self,
        target_tool_tcp_world: np.ndarray,
        *,
        orientation_tolerance_deg: float,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Return deterministic poses strictly inside endpoint tolerances."""

        target = np.asarray(target_tool_tcp_world, dtype=float).reshape(4, 4)
        orientation_rad = math.radians(orientation_tolerance_deg * 0.8)
        position_offset_m = self.config.endpoint_position_tolerance_m * 0.8
        variants: list[tuple[str, np.ndarray]] = []
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            axis = np.zeros(3, dtype=float)
            axis[axis_index] = 1.0
            for sign in (-1.0, 1.0):
                pose = target.copy()
                pose[:3, :3] = target[:3, :3] @ Rotation.from_rotvec(
                    sign * orientation_rad * axis
                ).as_matrix()
                variants.append(
                    (f"tool_rotation_{axis_name}_{'neg' if sign < 0 else 'pos'}", pose)
                )
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            for sign in (-1.0, 1.0):
                pose = target.copy()
                pose[axis_index, 3] += sign * position_offset_m
                variants.append(
                    (f"world_translation_{axis_name}_{'neg' if sign < 0 else 'pos'}", pose)
                )
        return tuple(variants)

    def assess_endpoint_ik(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> RM65GraspReachabilityResult:
        """Classify one endpoint with bounded read-only FK IK, without a path."""

        if side not in _SIDES:
            raise ValueError("side must be left or right.")
        target_pose = np.asarray(target_tool_tcp_world, dtype=float)
        if target_pose.shape != (4, 4) or not np.all(np.isfinite(target_pose)):
            raise ValueError("target_tool_tcp_world must be one finite 4x4 pose.")
        config = self.reachability_config
        with self._solver_lock:
            started = float(monotonic())
            deadline = started + config.time_budget_ms / 1_000.0
            best: tuple[float, float, float] | None = None
            cycles = 0
            termination = "seed_budget_exhausted"
            for seed_index, seed in enumerate(
                self._endpoint_seed_qpos(
                    qpos_by_side,
                    side=side,
                    seed_count=config.seed_count,
                )
            ):
                if seed_index > 0 and float(monotonic()) >= deadline:
                    termination = "time_budget_exhausted"
                    break
                remaining_evaluations = config.max_solve_cycles - cycles
                if remaining_evaluations <= 0:
                    termination = "cycle_budget_exhausted"
                    break
                try:
                    _candidate, evaluations, position, orientation = (
                        self._least_squares_endpoint_from_seed(
                            qpos_by_side=qpos_by_side,
                            side=side,
                            target_tool_tcp_world=target_pose,
                            seed_qpos=seed[side],
                            maximum_evaluations=min(
                                config.max_solve_evaluations_per_seed,
                                remaining_evaluations,
                            ),
                        )
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                cycles += evaluations
                if cycles > config.max_solve_cycles:
                    raise RuntimeError(
                        "RM65 reachability solver exceeded its total evaluation budget."
                    )
                normalized_error = max(
                    position / config.basic_position_tolerance_m,
                    orientation / config.basic_orientation_tolerance_deg,
                )
                if best is None or normalized_error < best[0]:
                    best = (normalized_error, position, orientation)
                if (
                    position <= config.standard_position_tolerance_m
                    and orientation <= config.standard_orientation_tolerance_deg
                ):
                    elapsed_ms = max(0.0, (float(monotonic()) - started) * 1_000.0)
                    return RM65GraspReachabilityResult(
                        classification="standard_reachable",
                        position_residual_m=position,
                        orientation_residual_deg=orientation,
                        solve_cycles=cycles,
                        elapsed_ms=elapsed_ms,
                        termination_reason="standard_tolerance_met",
                    )
            if best is None:
                raise RuntimeError("RM65 reachability solver produced no candidate.")
            if float(monotonic()) >= deadline:
                termination = "time_budget_exhausted"
            elif cycles >= config.max_solve_cycles:
                termination = "cycle_budget_exhausted"
            _normalized, position, orientation = best
            classification = (
                "basically_reachable"
                if position <= config.basic_position_tolerance_m
                and orientation <= config.basic_orientation_tolerance_deg
                else "difficult_to_reach"
            )
            return RM65GraspReachabilityResult(
                classification=classification,
                position_residual_m=position,
                orientation_residual_deg=orientation,
                solve_cycles=cycles,
                elapsed_ms=max(0.0, (float(monotonic()) - started) * 1_000.0),
                termination_reason=termination,
            )

    def probe_grasp_endpoint_clearance(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        time_budget_sec: float,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> RM65GraspEndpointProbeResult:
        """Quickly rank terminal branches without deciding candidate validity.

        This uses the same endpoint solver, envelope and sparse URDF/FCL oracle
        as production planning.  The deadline is intentionally small and is
        checked between deterministic seeds, so every non-success result is
        advisory and must not remove the candidate from the later full-chain
        search.
        """

        if side not in _SIDES:
            raise ValueError("side must be left or right.")
        budget = float(time_budget_sec)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("time_budget_sec must be finite and positive.")
        target = np.asarray(target_tool_tcp_world, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_tool_tcp_world must be one finite 4x4 pose.")
        started = float(monotonic())
        attempts: tuple[Mapping[str, object], ...] = ()
        reason_code: str | None = None
        try:
            candidates = self._endpoint_qpos_candidates(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target,
                deadline=started + budget,
                monotonic=monotonic,
            )
        except RM65GraspTrajectoryPlanningError as error:
            reason_code = error.reason_code
            raw_attempts = error.details.get("attempts", ())
            if isinstance(raw_attempts, (tuple, list)):
                attempts = tuple(
                    dict(item) for item in raw_attempts if isinstance(item, Mapping)
                )
            outcomes = {
                str(item.get("outcome"))
                for item in attempts
                if item.get("outcome") is not None
            }
            if "endpoint_clearance_blocked" in outcomes:
                classification = "clearance_blocked"
            elif "endpoint_collision_blocked" in outcomes:
                classification = "collision_blocked"
            elif error.reason_code in {
                "endpoint_collision_check_unavailable",
                "target_safety_envelope_unavailable",
            }:
                classification = "check_unavailable"
            else:
                classification = "unresolved"
        else:
            classification = "clearance_safe"
            attempts = tuple(
                {
                    "outcome": "accepted",
                    "seed_index": candidate.seed_index,
                }
                for candidate in candidates
            )
        distances = [
            float(item["minimum_distance_m"])
            for item in attempts
            if item.get("minimum_distance_m") is not None
            and math.isfinite(float(item["minimum_distance_m"]))
        ]
        return RM65GraspEndpointProbeResult(
            classification=classification,
            elapsed_ms=max(0.0, (float(monotonic()) - started) * 1_000.0),
            reason_code=reason_code,
            best_minimum_distance_m=(max(distances) if distances else None),
            attempted_seed_count=len(attempts),
        )

    def _revalidate(
        self,
        waypoints: tuple[Mapping[str, np.ndarray], ...],
        *,
        moving_side: str,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> dict[str, object]:
        checks = 0
        minimum_distance = math.inf
        minimum_distance_lower_bound = math.inf
        clearance_only = type(self.oracle) is RM65ObstacleCollisionOracle
        ingress_constraints: frozenset[str] | None = None
        ingress_initial_adjustment_m: float | None = None
        ingress_best_adjustment_m: float | None = None
        ingress_observed_constraints: set[str] = set()
        ingress_entered_at: dict[str, int] | None = None

        def reject_ingress(
            error: RM65GraspTrajectoryPlanningError,
            reason: str,
        ) -> None:
            raise RM65GraspTrajectoryPlanningError(
                error.reason_code,
                details={
                    **error.details,
                    "envelope_ingress_rejection": reason,
                },
            ) from error

        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            maximum_delta = max(
                float(np.max(np.abs(np.asarray(end[side]) - np.asarray(start[side]))))
                for side in _SIDES
            )
            steps = max(
                1, math.ceil(maximum_delta / self.config.final_edge_resolution_rad)
            )
            for index in range(steps + 1):
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="dense_revalidation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                        },
                    )
                alpha = float(index) / float(steps)
                qpos = {
                    side: np.asarray(start[side])
                    + alpha * (np.asarray(end[side]) - np.asarray(start[side]))
                    for side in _SIDES
                }
                solver_targets = self.target_builder.solver_world_targets_for_qpos_pair(
                    left_qpos=qpos["left"],
                    right_qpos=qpos["right"],
                )
                try:
                    self._require_unmodified_target_envelope(
                        solver_targets,
                        sides=_SIDES,
                        phase="path_target",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                        },
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    if error.reason_code != "path_target_safety_envelope_blocked":
                        raise
                    adjustment, constraints = self._ingress_projection(
                        moving_side=moving_side, details=error.details,
                    )
                    if ingress_entered_at is not None:
                        reject_ingress(error, "path_left_envelope_after_entry")
                    elif ingress_constraints is None:
                        if segment_index != 0 or index != 0:
                            reject_ingress(error, "path_did_not_start_at_the_violation")
                        if len(constraints) != 1:
                            reject_ingress(
                                error,
                                "multiple_initial_constraints_are_not_supported",
                            )
                        if (
                            adjustment
                            > self.config.maximum_path_envelope_ingress_distance_m
                        ):
                            reject_ingress(error, "initial_distance_exceeds_limit")
                        ingress_constraints = constraints
                        ingress_observed_constraints.update(constraints)
                        ingress_initial_adjustment_m = adjustment
                        ingress_best_adjustment_m = adjustment
                    else:
                        ingress_observed_constraints.update(constraints)
                        assert ingress_initial_adjustment_m is not None
                        assert ingress_best_adjustment_m is not None
                        ingress_best_adjustment_m = self._require_ingress_progress(
                            details=error.details, adjustment_m=adjustment, constraints=constraints,
                            initial_adjustment_m=ingress_initial_adjustment_m,
                            initial_constraints=ingress_constraints,
                            best_adjustment_m=ingress_best_adjustment_m,
                        )
                else:
                    if ingress_constraints is not None and ingress_entered_at is None:
                        ingress_entered_at = {
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                        }
                # Validate every sample against the selected sparse proxy
                # pairs at the same clearance threshold. Exact distances are
                # unnecessary for this Boolean proxy-clearance decision.
                check = (
                    self.oracle.check_clearance_only(qpos)
                    if clearance_only else self.oracle.check(qpos)
                )
                checks += 1
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="dense_revalidation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                        },
                    )
                if check.minimum_distance_m is not None:
                    minimum_distance_lower_bound = min(
                        minimum_distance_lower_bound, check.minimum_distance_m
                    )
                    if not clearance_only:
                        minimum_distance = min(minimum_distance, check.minimum_distance_m)
                if not check.collision_check_available or not check.safe:
                    raise RM65GraspTrajectoryPlanningError(
                        (
                            "path_collision_check_unavailable"
                            if not check.collision_check_available
                            else "path_collision_blocked"
                        ),
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                            "collision_check": check.to_dict(),
                        },
                    )
        waypoint_minimum_distance = math.inf
        if clearance_only:
            # Preserve exact endpoint diagnostics and fail closed if the two
            # FCL query modes ever disagree at a waypoint.
            for waypoint_index, waypoint in enumerate(waypoints):
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="dense_revalidation",
                        details={"waypoint_index": waypoint_index},
                    )
                exact = self.oracle.check(waypoint)
                if not exact.collision_check_available or not exact.safe:
                    raise RM65GraspTrajectoryPlanningError(
                        "path_collision_check_unavailable"
                        if not exact.collision_check_available else "path_collision_blocked",
                        details={
                            "waypoint_index": waypoint_index,
                            "collision_check": exact.to_dict(),
                        },
                    )
                if exact.minimum_distance_m is not None:
                    waypoint_minimum_distance = min(
                        waypoint_minimum_distance, exact.minimum_distance_m
                    )
        if ingress_constraints is not None and ingress_entered_at is None:
            raise RM65GraspTrajectoryPlanningError(
                "path_target_safety_envelope_blocked",
                details={
                    "envelope_ingress_rejection": "path_never_entered_envelope",
                    "modified_sides": (moving_side,),
                },
            )
        return {
            "sample_count": checks,
            "minimum_distance_m": (
                None if not math.isfinite(minimum_distance) else minimum_distance
            ),
            "minimum_distance_lower_bound_m": (
                None
                if not math.isfinite(minimum_distance_lower_bound)
                else minimum_distance_lower_bound
            ),
            "waypoint_minimum_distance_m": (
                None
                if not math.isfinite(waypoint_minimum_distance)
                else waypoint_minimum_distance
            ),
            "collision_query_mode": (
                "clearance_guard_with_exact_waypoints"
                if clearance_only else "all_pair_minimum_distance"
            ),
            "collision_scene_profile": self.path_collision_scene_profile,
            "envelope_ingress": (
                None
                if ingress_constraints is None
                else {
                    "side": moving_side,
                    "initial_translation_adjustment_m": (ingress_initial_adjustment_m),
                    "violated_translation_constraints": sorted(ingress_constraints),
                    "observed_translation_constraints": sorted(
                        ingress_observed_constraints
                    ),
                    "entered_at": ingress_entered_at,
                }
            ),
        }

    def _revalidate_visible_gripper_path(
        self,
        waypoints: tuple[Mapping[str, np.ndarray], ...],
        *,
        moving_side: str,
        visible_scene: VisibleGraspScene,
        terminal_tool_tcp_world: np.ndarray | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> dict[str, object]:
        """Densely admit the observed gripper sweep of a found joint path.

        The sampling planner checks the composed visible-scene oracle while it
        grows its trees, but its adaptive refinement is driven by the sparse
        URDF/FCL clearance.  A very small observed-point collision can
        therefore lie between two coarse tree-edge samples even when the FCL
        proxy is far from a fixed obstacle.  Recheck every found edge at the
        final 0.1-degree joint resolution before accepting the route.
        """

        terminal = (
            None
            if terminal_tool_tcp_world is None
            else np.asarray(terminal_tool_tcp_world, dtype=float).reshape(4, 4)
        )
        checks = 0
        terminal_contact_points = 0
        terminal_contact_sample_count = 0
        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            maximum_delta = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(end[side], dtype=float)
                            - np.asarray(start[side], dtype=float)
                        )
                    )
                )
                for side in _SIDES
            )
            steps = max(
                1,
                math.ceil(maximum_delta / self.config.final_edge_resolution_rad),
            )
            for edge_sample_index in range(steps + 1):
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="visible_scene_dense_revalidation",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                        },
                    )
                alpha = float(edge_sample_index) / float(steps)
                qpos = {
                    side: np.asarray(start[side], dtype=float)
                    + alpha
                    * (
                        np.asarray(end[side], dtype=float)
                        - np.asarray(start[side], dtype=float)
                    )
                    for side in _SIDES
                }
                pose = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=qpos["left"],
                    right_qpos=qpos["right"],
                )[moving_side]
                contact_allowed = False
                position_error_m: float | None = None
                orientation_error_deg: float | None = None
                if terminal is not None:
                    (
                        contact_allowed,
                        position_error_m,
                        orientation_error_deg,
                    ) = _terminal_target_contact_region(
                        tool_tcp_world=pose,
                        terminal_tool_tcp_world=terminal,
                        position_tolerance_m=max(
                            self.config.endpoint_position_tolerance_m,
                            RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
                        ),
                        orientation_tolerance_deg=(
                            self.config.endpoint_orientation_tolerance_deg
                        ),
                    )
                if contact_allowed:
                    (
                        target_hits,
                        sample_terminal_contact_points,
                        background_hits,
                    ) = visible_scene.collision_counts_with_terminal_target_contact(
                        pose,
                        opening_m=visible_scene.geometry.maximum_opening_m,
                        contact_depth_m=RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
                    )
                    terminal_contact_points = max(
                        terminal_contact_points,
                        int(sample_terminal_contact_points),
                    )
                    terminal_contact_sample_count += int(
                        sample_terminal_contact_points > 0
                    )
                else:
                    sample_terminal_contact_points = 0
                    target_hits, background_hits = visible_scene.collision_counts(
                        pose,
                        opening_m=visible_scene.geometry.maximum_opening_m,
                    )
                checks += 1
                if target_hits or background_hits:
                    raise RM65GraspTrajectoryPlanningError(
                        "visible_gripper_path_collision",
                        details={
                            "planning_phase": "visible_scene_dense_revalidation",
                            "segment_index": segment_index,
                            "edge_sample_index": edge_sample_index,
                            "target_collision_points": int(target_hits),
                            "background_collision_points": int(background_hits),
                            "terminal_target_contact_points": int(
                                sample_terminal_contact_points
                            ),
                            "terminal_contact_allowed": contact_allowed,
                            "terminal_position_error_m": position_error_m,
                            "terminal_orientation_error_deg": (
                                orientation_error_deg
                            ),
                        },
                    )
        return {
            "sample_count": checks,
            "scene_hash": visible_scene.semantic_content_hash,
            "opening_m": visible_scene.geometry.maximum_opening_m,
            "joint_edge_resolution_rad": self.config.final_edge_resolution_rad,
            "terminal_target_contact_points": terminal_contact_points,
            "terminal_target_contact_sample_count": (
                terminal_contact_sample_count
            ),
        }

    def _plan_path_leg(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        planning_phase: str,
        visible_scene: VisibleGraspScene | None = None,
        prefer_closest_tolerance: bool = True,
        endpoint_candidates: tuple[_EndpointIKCandidate, ...] | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> _PlannedPathLeg:
        if deadline is not None:
            self._remaining_path_planning_time_sec(
                deadline=deadline,
                monotonic=monotonic,
                planning_phase=planning_phase,
            )
        if endpoint_candidates is None:
            endpoint_candidates = self._endpoint_qpos_candidates(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_tool_tcp_world,
                prefer_closest_tolerance=prefer_closest_tolerance,
                deadline=deadline,
                monotonic=monotonic,
            )
        lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
        upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
        cfg = self.planner_config
        search_oracle = self._path_search_oracle(
            planning_phase=planning_phase, visible_scene=visible_scene, side=side,
            start_qpos_by_side=qpos_by_side,
            terminal_tool_tcp_world=(
                target_tool_tcp_world
                if planning_phase == "grasp_target"
                else None
            ),
        )
        search_attempts: list[dict[str, object]] = []
        terminal_error: RM65GraspTrajectoryPlanningError | None = None
        if not endpoint_candidates:
            raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")
        target_values = np.stack([endpoint.qpos_by_side[side] for endpoint in endpoint_candidates])
        search_candidates = endpoint_candidates
        # All locally valid branches share one tree and query cache per corridor.
        for path_attempt in range(self.config.path_search_attempts_per_endpoint):
            target = search_candidates[0].qpos_by_side
            full_joint_range = path_attempt == self.config.path_search_attempts_per_endpoint - 1
            padding = cfg.sample_padding_rad * float(2**path_attempt)
            corridor_lower = lower if full_joint_range else np.maximum(
                lower, np.minimum(qpos_by_side[side], target_values.min(axis=0)) - padding,
            )
            corridor_upper = upper if full_joint_range else np.minimum(
                upper, np.maximum(qpos_by_side[side], target_values.max(axis=0)) + padding,
            )
            attempt_config = replace(
                cfg,
                seed=cfg.seed + path_attempt,
                time_budget_sec=(
                    cfg.time_budget_sec if deadline is None else min(
                        self._remaining_path_planning_time_sec(
                            deadline=deadline, monotonic=monotonic,
                            planning_phase=planning_phase,
                            details={"path_attempt": path_attempt,
                                     "endpoint_seed_indices": tuple(e.seed_index for e in search_candidates)},
                        ),
                        math.inf if cfg.time_budget_sec is None else cfg.time_budget_sec,
                    )
                ),
            )
            path = plan_collision_aware_joint_path(
                start_qpos_by_component=qpos_by_side,
                target_qpos_by_component=target,
                alternative_target_qpos_by_component=tuple(
                    endpoint.qpos_by_side for endpoint in search_candidates[1:]
                ),
                movable_components=(side,), oracle=search_oracle,
                config=attempt_config,
                joint_lower_by_component={side: corridor_lower},
                joint_upper_by_component={side: corridor_upper},
                monotonic=monotonic,
            )
            attempt_record: dict[str, object] = {
                "planning_phase": planning_phase,
                "search_collision_model": "ik_sparse_proxy_pairs",
                "endpoint_candidate_ordering": (
                    self.config.endpoint_candidate_ordering
                ),
                "endpoint_seed_indices": tuple(e.seed_index for e in search_candidates),
                "path_attempt": path_attempt,
                "search_seed": attempt_config.seed,
                "full_joint_range": full_joint_range,
                "sample_padding_rad": None if full_joint_range else padding,
                "status": path.status.value,
                "reason": path.reason,
            }
            search_attempts.append(attempt_record)
            if path.status is not JointPathStatus.PATH_FOUND:
                if path.status is JointPathStatus.START_BLOCKED:
                    raise RM65GraspTrajectoryPlanningError(
                        "path_start_collision_blocked",
                        details={"planning_phase": planning_phase,
                                 "path": path.to_dict(), "search_attempts": search_attempts},
                    )
                if path.status is JointPathStatus.INVALID_REQUEST:
                    raise RM65GraspTrajectoryPlanningError(
                        "path_planner_error",
                        details={"planning_phase": planning_phase,
                                 "path": path.to_dict(), "search_attempts": search_attempts},
                    )
                if path.reason == "time_budget_exhausted":
                    terminal_error = RM65GraspTrajectoryPlanningError(
                        "path_planning_time_budget_exhausted",
                        details={"planning_phase": planning_phase,
                                 "time_budget_sec": self.config.path_planning_time_budget_sec,
                                 "path": path.to_dict(), "search_attempts": search_attempts},
                    )
                    break
                if path.status is JointPathStatus.COLLISION_CHECK_UNAVAILABLE:
                    terminal_error = RM65GraspTrajectoryPlanningError(
                        "path_collision_check_unavailable",
                        details={"planning_phase": planning_phase,
                                 "path": path.to_dict(), "search_attempts": search_attempts},
                    )
                    break
                continue
            selected_index = int(path.metadata.get("selected_target_index", 0))
            if not 0 <= selected_index < len(endpoint_candidates):
                raise RM65GraspTrajectoryPlanningError(
                    "path_planner_error", details={"planning_phase": planning_phase,
                                                   "rejection": "selected_endpoint_index_out_of_range"},
                )
            endpoint = search_candidates[selected_index]
            if any(
                not np.array_equal(endpoint.qpos_by_side[name], path.target_qpos_by_component[name])
                for name in _SIDES
            ):
                raise RM65GraspTrajectoryPlanningError(
                    "path_planner_error", details={"planning_phase": planning_phase,
                                                   "rejection": "selected_endpoint_mismatch"},
                )
            attempt_record["endpoint_seed_index"] = endpoint.seed_index
            try:
                dense = self._revalidate(
                    path.waypoints, moving_side=side, deadline=deadline, monotonic=monotonic,
                )
                if visible_scene is not None:
                    dense = {
                        **dense,
                        "visible_scene": self._revalidate_visible_gripper_path(
                            path.waypoints,
                            moving_side=side,
                            visible_scene=visible_scene,
                            terminal_tool_tcp_world=(
                                target_tool_tcp_world
                                if planning_phase == "grasp_target"
                                else None
                            ),
                            deadline=deadline,
                            monotonic=monotonic,
                        ),
                    }
            except RM65GraspTrajectoryPlanningError as error:
                attempt_record["dense_revalidation_error"] = error.reason_code
                attempt_record["dense_revalidation_details"] = dict(error.details)
                terminal_error = error
                if error.reason_code in {
                    "path_collision_check_unavailable", "path_planning_time_budget_exhausted",
                    "path_target_safety_envelope_unavailable",
                }:
                    break
                # Keep this goal available, but do not let a rejected direct
                # connection hide the other local branches in the next round.
                search_candidates = tuple(e for e in search_candidates if e is not endpoint) + (endpoint,)
                continue
            return _PlannedPathLeg(
                endpoint=endpoint, path=path, dense_revalidation=dense,
                search_attempts=tuple(search_attempts),
            )
        if terminal_error is not None and terminal_error.reason_code in {
            "path_collision_check_unavailable",
            "path_planning_time_budget_exhausted",
            "path_target_safety_envelope_blocked",
            "path_target_safety_envelope_unavailable",
        }:
            reason = terminal_error.reason_code
        elif any(
            item["status"] == JointPathStatus.PATH_SEARCH_EXHAUSTED.value
            for item in search_attempts
        ):
            reason = "path_search_exhausted"
        elif any(
            item["status"] == JointPathStatus.START_BLOCKED.value
            for item in search_attempts
        ):
            reason = "path_start_collision_blocked"
        elif any(
            item["status"] == JointPathStatus.TARGET_BLOCKED.value
            for item in search_attempts
        ):
            reason = "path_target_collision_blocked"
        elif terminal_error is not None:
            reason = terminal_error.reason_code
        else:
            reason = "path_planner_error"
        raise RM65GraspTrajectoryPlanningError(
            reason,
            details={
                "planning_phase": planning_phase,
                "endpoint_candidate_count": len(endpoint_candidates),
                "search_attempts": search_attempts,
                "last_dense_revalidation_error": (
                    None
                    if terminal_error is None
                    else {
                        "reason_code": terminal_error.reason_code,
                        "details": terminal_error.details,
                    }
                ),
            },
        )

    def _plan_constrained_approach_leg(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        visible_scene: VisibleGraspScene | None = None,
        terminal_endpoint: _EndpointIKCandidate | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> _PlannedPathLeg:
        """Resolve a fixed-orientation, same-branch Cartesian grasp approach.

        Normal planning propagates a pregrasp IK branch toward the grasp.  If
        ``terminal_endpoint`` is supplied, the same corridor is solved in
        reverse from an already collision-safe terminal branch, then returned
        in execution order.  The reverse form prevents an independently solved
        pregrasp branch from hiding a valid terminal branch; all envelope,
        collision, continuity, dense-path, and realized-TCP checks are shared.
        """

        targets = self._constrained_approach_targets(
            pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
            grasp_tool_tcp_world=grasp_tool_tcp_world,
        )
        reverse_from_terminal = terminal_endpoint is not None
        inactive = "right" if side == "left" else "left"
        current = {
            name: np.asarray(qpos_by_side[name], dtype=float).copy()
            for name in _SIDES
        }
        solve_start = {name: value.copy() for name, value in current.items()}
        waypoints: list[Mapping[str, np.ndarray]] = [solve_start]
        attempts: list[dict[str, object]] = []
        total_cycles = 0
        final_position_residual_m = math.inf
        final_orientation_residual_deg = math.inf
        if reverse_from_terminal:
            assert terminal_endpoint is not None
            forward_targets = tuple(
                (target, float(index) / float(len(targets)))
                for index, target in enumerate(targets, start=1)
            )
            solve_targets = (
                *tuple(reversed(forward_targets[:-1])),
                (np.asarray(pregrasp_tool_tcp_world, dtype=float), 0.0),
            )
        else:
            solve_targets = tuple(
                (target, float(index) / float(len(targets)))
                for index, target in enumerate(targets, start=1)
            )

        with self._solver_lock:
            for waypoint_index, (target, approach_progress) in enumerate(
                solve_targets,
                start=1,
            ):
                orientation_tolerance_deg = (
                    self.config.endpoint_orientation_tolerance_deg
                    + (1.0 - approach_progress)
                    * (
                        self.config.pregrasp_orientation_tolerance_deg
                        - self.config.endpoint_orientation_tolerance_deg
                    )
                )
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="constrained_approach_ik",
                        details={
                            "approach_waypoint_index": waypoint_index,
                            "approach_waypoint_count": len(solve_targets),
                            "solve_direction": (
                                "reverse_from_terminal"
                                if reverse_from_terminal
                                else "forward_from_pregrasp"
                            ),
                        },
                    )
                try:
                    candidate, cycles, position, orientation = (
                        self._least_squares_endpoint_from_seed(
                            qpos_by_side=current,
                            side=side,
                            target_tool_tcp_world=target,
                            seed_qpos=current[side],
                            maximum_evaluations=(
                                self.config.maximum_endpoint_solve_cycles
                            ),
                            prefer_closest_tolerance=False,
                        )
                    )
                except RM65GraspTrajectoryPlanningError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_ik_solver_error",
                        details={
                            "approach_waypoint_index": waypoint_index,
                            "approach_waypoint_count": len(solve_targets),
                            "solve_direction": (
                                "reverse_from_terminal"
                                if reverse_from_terminal
                                else "forward_from_pregrasp"
                            ),
                            "exception_type": type(exc).__name__,
                            "attempts": attempts,
                        },
                    ) from exc
                total_cycles += cycles
                candidate[inactive] = solve_start[inactive].copy()
                attempt = {
                    "planning_phase": "constrained_approach",
                    "approach_waypoint_index": waypoint_index,
                    "approach_waypoint_count": len(solve_targets),
                    "approach_progress": approach_progress,
                    "solve_direction": (
                        "reverse_from_terminal"
                        if reverse_from_terminal
                        else "forward_from_pregrasp"
                    ),
                    "solve_cycles": cycles,
                    "position_residual_m": position,
                    "orientation_residual_deg": orientation,
                    "orientation_tolerance_deg": orientation_tolerance_deg,
                }
                attempts.append(attempt)
                if (
                    position > self.config.endpoint_position_tolerance_m
                    or orientation > orientation_tolerance_deg
                ):
                    attempt["outcome"] = "ik_search_exhausted"
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_ik_search_exhausted",
                        details={
                            **attempt,
                            "endpoint_position_tolerance_m": (
                                self.config.endpoint_position_tolerance_m
                            ),
                            "endpoint_orientation_tolerance_deg": (
                                orientation_tolerance_deg
                            ),
                            "attempts": attempts,
                        },
                    )
                try:
                    self._require_qpos_within_target_envelope(
                        candidate,
                        sides=(side,),
                        phase="approach_endpoint",
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    attempt["outcome"] = error.reason_code
                    raise RM65GraspTrajectoryPlanningError(
                        error.reason_code,
                        details={
                            **error.details,
                            **attempt,
                            "attempts": attempts,
                        },
                    ) from error
                check = self.oracle.check(candidate)
                if not isinstance(check, JointConfigurationCheck):
                    attempt["outcome"] = "collision_oracle_returned_invalid_type"
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_endpoint_collision_check_unavailable",
                        details={
                            **attempt,
                            "returned_type": type(check).__name__,
                            "attempts": attempts,
                        },
                    )
                if not check.collision_check_available:
                    attempt["outcome"] = "collision_check_unavailable"
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_endpoint_collision_check_unavailable",
                        details={
                            **attempt,
                            "collision_check": check.to_dict(),
                            "attempts": attempts,
                        },
                    )
                if not check.safe:
                    positive_gap = (
                        check.minimum_distance_m is not None
                        and check.minimum_distance_m > 0.0
                    )
                    attempt["outcome"] = (
                        "clearance_blocked" if positive_gap else "collision_blocked"
                    )
                    raise RM65GraspTrajectoryPlanningError(
                        (
                            "approach_endpoint_clearance_blocked"
                            if positive_gap
                            else "approach_endpoint_collision_blocked"
                        ),
                        details={
                            **attempt,
                            "collision_check": check.to_dict(),
                            "attempts": attempts,
                        },
                    )
                maximum_joint_step_rad = float(
                    np.max(
                        np.abs(
                            np.asarray(candidate[side], dtype=float)
                            - np.asarray(current[side], dtype=float)
                        )
                    )
                )
                attempt["maximum_joint_step_rad"] = maximum_joint_step_rad
                if (
                    maximum_joint_step_rad
                    > self.config.maximum_approach_joint_step_rad
                ):
                    attempt["outcome"] = "branch_discontinuous"
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_branch_discontinuous",
                        details={
                            **attempt,
                            "maximum_allowed_joint_step_rad": (
                                self.config.maximum_approach_joint_step_rad
                            ),
                            "attempts": attempts,
                        },
                    )
                attempt["outcome"] = "accepted"
                current = {
                    name: np.asarray(candidate[name], dtype=float).copy()
                    for name in _SIDES
                }
                waypoints.append(current)
                final_position_residual_m = position
                final_orientation_residual_deg = orientation

        waypoint_tuple = (
            tuple(reversed(waypoints))
            if reverse_from_terminal
            else tuple(waypoints)
        )
        dense = self._revalidate(
            waypoint_tuple,
            moving_side=side,
            deadline=deadline,
            monotonic=monotonic,
        )
        if visible_scene is not None:
            dense = {
                **dense,
                "visible_scene": self._revalidate_visible_gripper_path(
                    waypoint_tuple,
                    moving_side=side,
                    visible_scene=visible_scene,
                    terminal_tool_tcp_world=grasp_tool_tcp_world,
                    deadline=deadline,
                    monotonic=monotonic,
                ),
            }
        approach_geometry = self._validate_constrained_approach_geometry(
            waypoint_tuple,
            moving_side=side,
            pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
            grasp_tool_tcp_world=grasp_tool_tcp_world,
            deadline=deadline,
            monotonic=monotonic,
        )
        dense = {**dense, "approach_geometry": approach_geometry}
        path = JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND,
            start_qpos_by_component=waypoint_tuple[0],
            target_qpos_by_component=waypoint_tuple[-1],
            movable_components=(side,),
            waypoints=waypoint_tuple,
            scene_hash=self.scene.semantic_content_hash,
            metadata={
                "planning_phase": "constrained_approach",
                "approach_waypoint_count": len(targets),
                "solve_direction": (
                    "reverse_from_terminal"
                    if reverse_from_terminal
                    else "forward_from_pregrasp"
                ),
                "approach_geometry": approach_geometry,
            },
        )
        endpoint = (
            terminal_endpoint
            if terminal_endpoint is not None
            else _EndpointIKCandidate(
                qpos_by_side=current,
                seed_index=-1,
                solve_cycles=total_cycles,
                position_residual_m=final_position_residual_m,
                orientation_residual_deg=final_orientation_residual_deg,
            )
        )
        return _PlannedPathLeg(
            endpoint=endpoint,
            path=path,
            dense_revalidation=dense,
            search_attempts=tuple(attempts),
        )

    def _prepare_grasp_approaches(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        visible_scene: VisibleGraspScene | None = None,
        deadline: float,
        monotonic: Callable[[], float],
        terminal_endpoint_search: Callable[
            [], tuple[_EndpointIKCandidate, ...]
        ] | None = None,
    ) -> _PreparedGraspApproaches:
        """Validate local branches before spending global/ingress budget.

        A pregrasp-first solve remains the fast path.  If that independently
        selected IK branch cannot enter the grasp, retry by solving collision-
        safe terminal branches and propagating them backward to the pregrasp.
        This mirrors bidirectional task planning: a failed local seed is not
        treated as proof that every terminal branch lacks an approach.
        """

        fatal_reasons = {
            "path_collision_check_unavailable",
            "path_planning_time_budget_exhausted",
            "endpoint_collision_check_unavailable",
            "approach_endpoint_collision_check_unavailable",
            "path_target_safety_envelope_unavailable",
            "approach_endpoint_safety_envelope_unavailable",
            "target_safety_envelope_unavailable",
        }
        candidates: list[tuple[_EndpointIKCandidate, _PlannedPathLeg]] = []
        attempts: list[dict[str, object]] = []
        last_error: RM65GraspTrajectoryPlanningError | None = None
        try:
            endpoints = self._endpoint_qpos_candidates(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=pregrasp_tool_tcp_world,
                prefer_closest_tolerance=False,
                orientation_tolerance_deg=(
                    self.config.pregrasp_orientation_tolerance_deg
                ),
                deadline=deadline,
                monotonic=monotonic,
            )
        except RM65GraspTrajectoryPlanningError as error:
            last_error = RM65GraspTrajectoryPlanningError(
                error.reason_code,
                details={
                    "planning_phase": "local_pregrasp_ik",
                    **error.details,
                },
            )
            attempts.append(
                {
                    "planning_phase": "local_pregrasp_ik",
                    "solve_direction": "forward_from_pregrasp",
                    "status": "rejected",
                    "reason_code": error.reason_code,
                }
            )
            if error.reason_code in fatal_reasons:
                raise RM65GraspTrajectoryPlanningError(
                    error.reason_code,
                    details={
                        "planning_phase": "local_pregrasp_ik",
                        **error.details,
                        "pregrasp_branch_attempts": attempts,
                    },
                ) from error
        else:
            for endpoint in endpoints:
                attempt: dict[str, object] = {
                    "planning_phase": "pregrasp_branch_selection",
                    "solve_direction": "forward_from_pregrasp",
                    "endpoint_seed_index": endpoint.seed_index,
                }
                attempts.append(attempt)
                try:
                    approach = self._plan_constrained_approach_leg(
                        qpos_by_side=endpoint.qpos_by_side,
                        side=side,
                        pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                        grasp_tool_tcp_world=grasp_tool_tcp_world,
                        visible_scene=visible_scene,
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    last_error = error
                    attempt.update(
                        {"status": "rejected", "reason_code": error.reason_code}
                    )
                    if error.reason_code in fatal_reasons:
                        raise RM65GraspTrajectoryPlanningError(
                            error.reason_code,
                            details={
                                **error.details,
                                "pregrasp_branch_attempts": attempts,
                            },
                        ) from error
                    continue
                attempt["status"] = "local_chain_valid"
                candidates.append((endpoint, approach))
        if candidates:
            return _PreparedGraspApproaches(tuple(candidates), tuple(attempts))

        # A forward solve can converge to a convenient pregrasp branch that
        # has no continuous route to any valid terminal IK branch.  Search the
        # necessary endpoint first, then propagate the exact branch backward.
        try:
            terminal_endpoints = (
                terminal_endpoint_search()
                if terminal_endpoint_search is not None
                else self._endpoint_qpos_candidates(
                    qpos_by_side=qpos_by_side,
                    side=side,
                    target_tool_tcp_world=grasp_tool_tcp_world,
                    prefer_closest_tolerance=False,
                    deadline=deadline,
                    monotonic=monotonic,
                )
            )
        except RM65GraspTrajectoryPlanningError as error:
            last_error = RM65GraspTrajectoryPlanningError(
                error.reason_code,
                details={
                    "planning_phase": "terminal_branch_selection",
                    **error.details,
                },
            )
            attempts.append(
                {
                    "planning_phase": "terminal_branch_selection",
                    "solve_direction": "reverse_from_terminal",
                    "status": "rejected",
                    "reason_code": error.reason_code,
                }
            )
            if error.reason_code in fatal_reasons:
                raise RM65GraspTrajectoryPlanningError(
                    error.reason_code,
                    details={
                        "planning_phase": "terminal_branch_selection",
                        **error.details,
                        "pregrasp_branch_attempts": attempts,
                    },
                ) from error
        else:
            for terminal_endpoint in terminal_endpoints:
                attempt = {
                    "planning_phase": "terminal_branch_selection",
                    "solve_direction": "reverse_from_terminal",
                    "endpoint_seed_index": terminal_endpoint.seed_index,
                }
                attempts.append(attempt)
                try:
                    approach = self._plan_constrained_approach_leg(
                        qpos_by_side=terminal_endpoint.qpos_by_side,
                        side=side,
                        pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                        grasp_tool_tcp_world=grasp_tool_tcp_world,
                        visible_scene=visible_scene,
                        terminal_endpoint=terminal_endpoint,
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    last_error = error
                    attempt.update(
                        {"status": "rejected", "reason_code": error.reason_code}
                    )
                    if error.reason_code in fatal_reasons:
                        raise RM65GraspTrajectoryPlanningError(
                            error.reason_code,
                            details={
                                **error.details,
                                "pregrasp_branch_attempts": attempts,
                            },
                        ) from error
                    continue
                pregrasp_qpos = {
                    name: np.asarray(values, dtype=float).copy()
                    for name, values in approach.path.waypoints[0].items()
                }
                realized_pregrasp = (
                    self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                        left_qpos=pregrasp_qpos["left"],
                        right_qpos=pregrasp_qpos["right"],
                    )[side]
                )
                position, orientation = pose_delta_metrics(
                    realized_pregrasp,
                    pregrasp_tool_tcp_world,
                )
                pregrasp_endpoint = _EndpointIKCandidate(
                    qpos_by_side=pregrasp_qpos,
                    seed_index=terminal_endpoint.seed_index,
                    solve_cycles=sum(
                        int(item.get("solve_cycles", 0))
                        for item in approach.search_attempts
                    ),
                    position_residual_m=position,
                    orientation_residual_deg=orientation,
                )
                attempt["status"] = "local_chain_valid"
                candidates.append((pregrasp_endpoint, approach))
        if candidates:
            return _PreparedGraspApproaches(tuple(candidates), tuple(attempts))
        assert last_error is not None
        raise RM65GraspTrajectoryPlanningError(
            last_error.reason_code,
            details={"planning_phase": "local_approach_validation", **last_error.details,
                     "pregrasp_branch_attempts": attempts},
        ) from last_error

    def assess_local_grasp_chain(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        time_budget_sec: float,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> RM65GraspLocalChainResult:
        """Run the production terminal/pregrasp IK and local approach checks.

        This is a read-only admission primitive for perception shortlisting. It
        deliberately stops before global path search, but shares the exact
        endpoint tolerances, envelope checks, sparse URDF/FCL collision oracle,
        branch continuity, and constrained-approach implementation used by
        :meth:`plan`.
        """

        if side not in _SIDES:
            raise ValueError("side must be left or right.")
        target = np.asarray(target_tool_tcp_world, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_tool_tcp_world must be one finite 4x4 pose.")
        budget = float(time_budget_sec)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("time_budget_sec must be finite and positive.")
        started = float(monotonic())
        deadline = started + min(budget, self.config.path_planning_time_budget_sec)
        attempts: list[Mapping[str, object]] = []
        last_error: RM65GraspTrajectoryPlanningError | None = None
        terminal_endpoints: tuple[_EndpointIKCandidate, ...] | None = None
        terminal_error: RM65GraspTrajectoryPlanningError | None = None

        def terminal_endpoint_search() -> tuple[_EndpointIKCandidate, ...]:
            # Distance changes the pregrasp, not this fixed terminal request.
            # Share its deterministic search only within this one assessment;
            # every distance still attempts its own forward approach first.
            nonlocal terminal_endpoints, terminal_error
            if float(monotonic()) >= deadline:
                raise RM65GraspTrajectoryPlanningError(
                    "path_planning_time_budget_exhausted",
                    details={"planning_phase": "terminal_branch_selection"},
                )
            if terminal_error is not None:
                raise terminal_error
            if terminal_endpoints is None:
                try:
                    terminal_endpoints = self._endpoint_qpos_candidates(
                        qpos_by_side=qpos_by_side,
                        side=side,
                        target_tool_tcp_world=target,
                        prefer_closest_tolerance=False,
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    terminal_error = error
                    raise
            return terminal_endpoints

        for distance_m in self._pregrasp_distance_candidates():
            if float(monotonic()) >= deadline:
                last_error = RM65GraspTrajectoryPlanningError(
                    "path_planning_time_budget_exhausted",
                    details={"planning_phase": "local_grasp_chain_admission"},
                )
                break
            pregrasp = self._pregrasp_target_pose(
                target,
                distance_m=distance_m,
            )
            try:
                prepared = self._prepare_grasp_approaches(
                    qpos_by_side=qpos_by_side,
                    side=side,
                    pregrasp_tool_tcp_world=pregrasp,
                    grasp_tool_tcp_world=target,
                    deadline=deadline,
                    monotonic=monotonic,
                    terminal_endpoint_search=terminal_endpoint_search,
                )
            except RM65GraspTrajectoryPlanningError as error:
                last_error = error
                attempts.append(
                    {
                        "distance_m": float(distance_m),
                        "status": "rejected",
                        "reason_code": error.reason_code,
                        "planning_phase": error.details.get("planning_phase"),
                        "best_position_residual_m": error.details.get(
                            "best_position_residual_m"
                        ),
                        "best_orientation_residual_deg": error.details.get(
                            "best_orientation_residual_deg"
                        ),
                        "endpoint_seed_count": error.details.get(
                            "endpoint_seed_count"
                        ),
                        "collision_examples": error.details.get(
                            "collision_examples", []
                        ),
                    }
                )
                if error.reason_code in {
                    "path_planning_time_budget_exhausted",
                    "path_collision_check_unavailable",
                    "endpoint_collision_check_unavailable",
                    "approach_endpoint_collision_check_unavailable",
                    "approach_endpoint_safety_envelope_unavailable",
                    "path_target_safety_envelope_unavailable",
                    "target_safety_envelope_unavailable",
                }:
                    break
                continue
            attempts.append(
                {
                    "distance_m": float(distance_m),
                    "status": "local_chain_valid",
                    "branch_count": len(prepared.candidates),
                }
            )
            return RM65GraspLocalChainResult(
                feasible=True,
                selected_pregrasp_distance_m=float(distance_m),
                branch_count=len(prepared.candidates),
                elapsed_ms=max(0.0, (float(monotonic()) - started) * 1_000.0),
                reason_code=None,
                attempts=tuple(attempts),
            )
        return RM65GraspLocalChainResult(
            feasible=False,
            selected_pregrasp_distance_m=None,
            branch_count=0,
            elapsed_ms=max(0.0, (float(monotonic()) - started) * 1_000.0),
            reason_code=(
                "local_grasp_chain_rejected"
                if last_error is None
                else last_error.reason_code
            ),
            attempts=tuple(attempts),
        )

    def _plan_pregrasp_and_approach(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        deadline: float,
        monotonic: Callable[[], float],
        visible_scene: VisibleGraspScene | None = None,
        prepared_approaches: _PreparedGraspApproaches | None = None,
    ) -> tuple[_PlannedPathLeg, _PlannedPathLeg]:
        """Connect to an exact prevalidated local branch, without re-solving IK."""

        prepared = prepared_approaches if prepared_approaches is not None else self._prepare_grasp_approaches(
            qpos_by_side=qpos_by_side, side=side,
            pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
            grasp_tool_tcp_world=grasp_tool_tcp_world,
            visible_scene=visible_scene,
            deadline=deadline, monotonic=monotonic,
        )
        try:
            pregrasp = self._plan_path_leg(
                qpos_by_side=qpos_by_side, side=side,
                target_tool_tcp_world=pregrasp_tool_tcp_world,
                planning_phase="pregrasp_target", visible_scene=visible_scene,
                prefer_closest_tolerance=False,
                endpoint_candidates=tuple(endpoint for endpoint, _ in prepared.candidates),
                deadline=deadline, monotonic=monotonic,
            )
        except RM65GraspTrajectoryPlanningError as error:
            raise RM65GraspTrajectoryPlanningError(
                error.reason_code,
                details={**error.details,
                         "local_approach_validation_attempts": prepared.validation_attempts},
            ) from error
        approach = next(
            approach for endpoint, approach in prepared.candidates
            if endpoint is pregrasp.endpoint
        )
        return (
            replace(pregrasp, search_attempts=prepared.validation_attempts + pregrasp.search_attempts),
            approach,
        )

    def _plan_chain_for_pregrasp(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        deadline: float,
        monotonic: Callable[[], float],
        visible_scene: VisibleGraspScene | None = None,
    ) -> tuple[
        tuple[_PlannedPathLeg, ...],
        dict[str, object] | None,
        list[dict[str, object]],
    ]:
        """Validate the local chain, then plan ingress and its exact connection."""

        prepared = self._prepare_grasp_approaches(
            qpos_by_side=qpos_by_side, side=side,
            pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
            grasp_tool_tcp_world=grasp_tool_tcp_world,
            visible_scene=visible_scene,
            deadline=deadline, monotonic=monotonic,
        )
        ingress_targets = self._envelope_ingress_targets(
            qpos_by_side=qpos_by_side,
            side=side,
            final_tool_tcp_world=pregrasp_tool_tcp_world,
        )
        if not ingress_targets:
            pregrasp_leg, approach_leg = self._plan_pregrasp_and_approach(
                qpos_by_side=qpos_by_side,
                side=side,
                pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                grasp_tool_tcp_world=grasp_tool_tcp_world,
                deadline=deadline,
                monotonic=monotonic,
                visible_scene=visible_scene,
                prepared_approaches=prepared,
            )
            return (pregrasp_leg, approach_leg), None, []

        ingress_route_attempts: list[dict[str, object]] = []
        route_error: RM65GraspTrajectoryPlanningError | None = None
        for route_index, (inward_fraction, ingress_target) in enumerate(ingress_targets):
            remaining = self._remaining_path_planning_time_sec(
                deadline=deadline, monotonic=monotonic, planning_phase="envelope_ingress",
                details={"envelope_ingress_route_attempts": ingress_route_attempts},
            )
            # An unsuccessful entry must not consume every other route's time.
            # The same slice includes connection and exact combined revalidation.
            route_budget = remaining / (len(ingress_targets) - route_index)
            route_deadline = min(deadline, float(monotonic()) + route_budget)
            route_attempt: dict[str, object] = {
                "inward_fraction": inward_fraction,
                "route_time_budget_sec": route_budget,
            }
            ingress_route_attempts.append(route_attempt)
            try:
                ingress_leg = self._plan_path_leg(
                    qpos_by_side=qpos_by_side,
                    side=side,
                    target_tool_tcp_world=ingress_target,
                    planning_phase="envelope_ingress",
                    visible_scene=visible_scene,
                    deadline=route_deadline,
                    monotonic=monotonic,
                )
                pregrasp_leg, approach_leg = self._plan_pregrasp_and_approach(
                    qpos_by_side=ingress_leg.endpoint.qpos_by_side,
                    side=side,
                    pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                    grasp_tool_tcp_world=grasp_tool_tcp_world,
                    deadline=route_deadline,
                    monotonic=monotonic,
                    visible_scene=visible_scene,
                    prepared_approaches=prepared,
                )
                combined_waypoints = (
                    *ingress_leg.path.waypoints,
                    *pregrasp_leg.path.waypoints[1:],
                    *approach_leg.path.waypoints[1:],
                )
                self._revalidate(
                    combined_waypoints,
                    moving_side=side,
                    deadline=route_deadline,
                    monotonic=monotonic,
                )
                if visible_scene is not None:
                    self._revalidate_visible_gripper_path(
                        combined_waypoints,
                        moving_side=side,
                        visible_scene=visible_scene,
                        terminal_tool_tcp_world=grasp_tool_tcp_world,
                        deadline=route_deadline,
                        monotonic=monotonic,
                    )
            except RM65GraspTrajectoryPlanningError as error:
                route_error = error
                route_attempt.update(
                    {
                        "status": "rejected",
                        "reason_code": error.reason_code,
                        "planning_phase": error.details.get("planning_phase"),
                        "details": dict(error.details),
                    }
                )
                if error.reason_code in {
                    "path_collision_check_unavailable",
                    "endpoint_collision_check_unavailable",
                    "approach_endpoint_collision_check_unavailable",
                    "path_target_safety_envelope_unavailable",
                    "target_safety_envelope_unavailable",
                }:
                    break
                if error.reason_code == "path_planning_time_budget_exhausted":
                    global_exhausted = float(monotonic()) >= deadline
                    route_attempt["budget_scope"] = "global" if global_exhausted else "ingress_route"
                    if global_exhausted:
                        break
                continue
            route_attempt["status"] = "path_found"
            return (
                (ingress_leg, pregrasp_leg, approach_leg),
                {
                    "inward_fraction": inward_fraction,
                    "ingress_target_tool_tcp_world": ingress_target,
                    "ingress_endpoint_seed_index": ingress_leg.endpoint.seed_index,
                    "ingress_path_hash": ingress_leg.path.semantic_content_hash,
                },
                ingress_route_attempts,
            )
        if route_error is None:
            route_error = RM65GraspTrajectoryPlanningError("path_planner_error")
        raise RM65GraspTrajectoryPlanningError(
            route_error.reason_code,
            details={
                **route_error.details,
                "envelope_ingress_route_attempts": ingress_route_attempts,
            },
        ) from route_error

    @staticmethod
    def _may_fallback_to_global_grasp_target_search(
        error: RM65GraspTrajectoryPlanningError,
    ) -> bool:
        """Return whether a local grasp-chain rejection may try the terminal target.

        The fallback is deliberately limited to failures produced while
        searching an artificial pregrasp/approach route.  The final target
        envelope is checked before entering this loop, and the global target
        search performs its own endpoint IK, sparse-proxy path checks, and
        dense revalidation.  This therefore removes a false veto from the
        fixed local approach without weakening the safety gates.
        """

        if error.reason_code not in _RM65_GLOBAL_GRASP_TARGET_FALLBACK_REASONS:
            return False
        if error.reason_code != "target_safety_envelope_blocked":
            return True
        # A target-safety error from the final target is already raised before
        # the pregrasp loop.  Only an artificial local target may be bypassed.
        phase = str(error.details.get("planning_phase", ""))
        return phase in {
            "local_pregrasp_ik",
            "local_approach_validation",
            "pregrasp_target",
            "constrained_approach",
            "constrained_approach_ik",
            "envelope_ingress",
            "dense_revalidation",
        }

    def __call__(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        speed_scale: float,
        visible_scene: VisibleGraspScene | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> RealAgentJointTrajectory:
        selected_side = str(side).strip().lower()
        if selected_side not in _SIDES:
            raise ValueError("side must be left or right.")
        current = {
            name: np.asarray(dict(qpos_by_side).get(name), dtype=float)
            .reshape(6)
            .copy()
            for name in _SIDES
        }
        target_pose = np.asarray(target_tool_tcp_world, dtype=float).reshape(4, 4)
        planning_started_at = time.perf_counter()
        capture_self_observation_evidence = (
            None
            if visible_scene is None
            or visible_scene.capture_self_observation_evidence is None
            else dict(visible_scene.capture_self_observation_evidence)
        )
        depth_preprocessing_evidence = (
            None
            if visible_scene is None
            or visible_scene.depth_preprocessing_evidence is None
            else dict(visible_scene.depth_preprocessing_evidence)
        )
        excluded_gripper_observation_count = 0
        if visible_scene is not None and visible_scene.capture_qpos_by_side is not None:
            capture_qpos = visible_scene.capture_qpos_by_side
            with self._solver_lock:
                capture_poses = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=capture_qpos["left"], right_qpos=capture_qpos["right"],
                )
            visible_scene, excluded_gripper_observation_count = (
                visible_scene.without_captured_gripper_observations(
                    capture_poses,
                    surface_margin_m=(
                        RM65_GRASP_CAPTURE_SELF_OBSERVATION_SURFACE_MARGIN_M
                    ),
                )
            )
        pregrasp_pose = self._pregrasp_target_pose(target_pose)
        final_solver_targets = (
            self.target_builder.solver_world_targets_for_qpos_pair(
                left_qpos=current["left"],
                right_qpos=current["right"],
            )
        )
        final_solver_targets[selected_side] = (
            self.target_builder.solver_world_target_from_tool_tcp_world_target(
                side=selected_side,
                tool_tcp_world_target=target_pose,
            )
        )
        self._require_unmodified_target_envelope(
            final_solver_targets,
            sides=(selected_side,),
            phase="target",
        )
        current_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=current["left"],
            right_qpos=current["right"],
        )[selected_side]
        (
            current_position_error_m,
            current_orientation_error_deg,
        ) = pose_delta_metrics(current_tcp, target_pose)
        already_at_grasp = (
            current_position_error_m <= 1e-9
            and current_orientation_error_deg <= 1e-7
        )
        scale = float(speed_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale must satisfy 0 < speed_scale <= 1.")
        self._preflight_start_envelope(qpos_by_side=current, side=selected_side)
        deadline = float(monotonic()) + self.config.path_planning_time_budget_sec
        ingress_route_attempts: list[dict[str, object]] = []
        pregrasp_route_attempts: list[dict[str, object]] = []
        selected_legs: tuple[_PlannedPathLeg, ...] | None = None
        ingress_route: dict[str, object] | None = None
        route_error: RM65GraspTrajectoryPlanningError | None = None
        route_mode: str | None = "already_at_grasp" if already_at_grasp else None
        direct_fallback_attempt: dict[str, object] | None = None
        direct_fallback_tried = False
        path_search_started_at = time.perf_counter()
        if already_at_grasp:
            selected_legs = (
                self._plan_path_leg(
                    qpos_by_side=current,
                    side=selected_side,
                    target_tool_tcp_world=target_pose,
                    planning_phase="grasp_target",
                    visible_scene=visible_scene,
                    deadline=deadline,
                    monotonic=monotonic,
                ),
            )
        else:
            for distance_m in self._pregrasp_distance_candidates():
                candidate_pregrasp = self._pregrasp_target_pose(
                    target_pose,
                    distance_m=distance_m,
                )
                route_attempt: dict[str, object] = {"distance_m": distance_m}
                pregrasp_route_attempts.append(route_attempt)
                try:
                    if visible_scene is not None:
                        visible_diagnostics = self._visible_grasp_collision_diagnostics(
                            visible_scene,
                            grasp_tool_tcp_world=target_pose,
                            distance_m=distance_m,
                        )
                        route_attempt["visible_gripper_diagnostics"] = (
                            visible_diagnostics
                        )
                        if visible_diagnostics["terminal_collision_blocked"]:
                            raise RM65GraspTrajectoryPlanningError(
                                "visible_gripper_terminal_collision",
                                details={
                                    "planning_phase": "terminal_grasp",
                                    **visible_diagnostics,
                                },
                            )
                        if visible_diagnostics["straight_ingress_only"]:
                            # A collision on one nominal ingress distance is not
                            # proof that every shorter constrained approach is
                            # blocked.  Keep searching those approaches first.
                            # Production never substitutes an unconstrained
                            # direct terminal path for this grasp-chain check.
                            route_error = RM65GraspTrajectoryPlanningError(
                                "visible_gripper_approach_collision",
                                details={
                                    "planning_phase": "constrained_approach",
                                    **visible_diagnostics,
                                },
                            )
                            route_attempt.update(
                                {
                                    "status": "rejected",
                                    "reason_code": (
                                        "straight_ingress_collision"
                                    ),
                                }
                            )
                            if (
                                self.config.allow_unconstrained_terminal_fallback
                                and not direct_fallback_tried
                            ):
                                direct_fallback_tried = True
                                direct_fallback_attempt = {
                                    "planning_phase": "global_grasp_target_fallback",
                                    "trigger_distance_m": float(distance_m),
                                    "trigger_diagnostics": visible_diagnostics,
                                }
                                try:
                                    direct_leg = self._plan_path_leg(
                                        qpos_by_side=current,
                                        side=selected_side,
                                        target_tool_tcp_world=target_pose,
                                        planning_phase="grasp_target",
                                        visible_scene=visible_scene,
                                        prefer_closest_tolerance=False,
                                        deadline=deadline,
                                        monotonic=monotonic,
                                    )
                                except RM65GraspTrajectoryPlanningError as error:
                                    direct_fallback_attempt.update(
                                        {
                                            "status": "rejected",
                                            "reason_code": error.reason_code,
                                            "details": dict(error.details),
                                        }
                                    )
                                    route_error = error
                                else:
                                    direct_fallback_attempt["status"] = "path_found"
                                    route_attempt["status"] = "path_found"
                                    route_attempt["fallback"] = (
                                        "global_grasp_target"
                                    )
                                    selected_legs = (direct_leg,)
                                    ingress_route = None
                                    ingress_route_attempts = []
                                    pregrasp_pose = target_pose.copy()
                                    route_mode = "global_grasp_target_fallback"
                                    break
                            continue
                    (
                        candidate_legs,
                        candidate_ingress_route,
                        candidate_ingress_attempts,
                    ) = self._plan_chain_for_pregrasp(
                        qpos_by_side=current,
                        side=selected_side,
                        pregrasp_tool_tcp_world=candidate_pregrasp,
                        grasp_tool_tcp_world=target_pose,
                        deadline=deadline,
                        monotonic=monotonic,
                        visible_scene=visible_scene,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    route_error = error
                    route_attempt.update(
                        {
                            "status": "rejected",
                            "reason_code": error.reason_code,
                            "planning_phase": error.details.get("planning_phase"),
                            "approach_waypoint_index": error.details.get(
                                "approach_waypoint_index"
                            ),
                            "details": dict(error.details),
                        }
                    )
                    if error.reason_code in {
                        "path_collision_check_unavailable",
                        "path_planning_time_budget_exhausted",
                        "endpoint_collision_check_unavailable",
                        "approach_endpoint_collision_check_unavailable",
                        "approach_endpoint_safety_envelope_unavailable",
                        "path_target_safety_envelope_unavailable",
                        "target_safety_envelope_unavailable",
                    }:
                        break
                    continue
                route_attempt["status"] = "path_found"
                selected_legs = candidate_legs
                ingress_route = candidate_ingress_route
                ingress_route_attempts = candidate_ingress_attempts
                pregrasp_pose = candidate_pregrasp
                break
            if selected_legs is None:
                if route_error is None:
                    route_error = RM65GraspTrajectoryPlanningError("path_planner_error")
                local_route_error = route_error
                if (
                    self.config.allow_unconstrained_terminal_fallback
                    and not direct_fallback_tried
                    and self._may_fallback_to_global_grasp_target_search(
                        local_route_error
                    )
                ):
                    direct_fallback_tried = True
                    direct_fallback_attempt = {
                        "planning_phase": "global_grasp_target_fallback",
                        "trigger_reason_code": local_route_error.reason_code,
                        "trigger_details": dict(local_route_error.details),
                        "trigger_scope": "local_pregrasp_chain",
                    }
                    try:
                        direct_leg = self._plan_path_leg(
                            qpos_by_side=current,
                            side=selected_side,
                            target_tool_tcp_world=target_pose,
                            planning_phase="grasp_target",
                            visible_scene=visible_scene,
                            prefer_closest_tolerance=False,
                            deadline=deadline,
                            monotonic=monotonic,
                        )
                    except RM65GraspTrajectoryPlanningError as error:
                        direct_fallback_attempt.update(
                            {
                                "status": "rejected",
                                "reason_code": error.reason_code,
                                "details": dict(error.details),
                            }
                        )
                        route_error = RM65GraspTrajectoryPlanningError(
                            error.reason_code,
                            details={
                                **error.details,
                                "local_pregrasp_failure": {
                                    "reason_code": local_route_error.reason_code,
                                    "details": dict(local_route_error.details),
                                },
                                "direct_grasp_target_fallback": (
                                    direct_fallback_attempt
                                ),
                            },
                        )
                    else:
                        direct_fallback_attempt["status"] = "path_found"
                        selected_legs = (direct_leg,)
                        ingress_route = None
                        ingress_route_attempts = []
                        pregrasp_pose = target_pose.copy()
                        route_mode = "global_grasp_target_fallback"
                if selected_legs is None:
                    raise RM65GraspTrajectoryPlanningError(
                        route_error.reason_code,
                        details={
                            **route_error.details,
                            "pregrasp_route_attempts": pregrasp_route_attempts,
                            "direct_grasp_target_fallback": direct_fallback_attempt,
                        },
                    ) from route_error

        assert selected_legs is not None
        path_search_finished_at = time.perf_counter()
        self._remaining_path_planning_time_sec(
            deadline=deadline,
            monotonic=monotonic,
            planning_phase="trajectory_parameterization",
        )
        endpoint = selected_legs[-1].endpoint
        waypoints = tuple(
            waypoint
            for leg_index, leg in enumerate(selected_legs)
            for waypoint in (
                leg.path.waypoints if leg_index == 0 else leg.path.waypoints[1:]
            )
        )
        leg_end_waypoint_indices: dict[int, str] = {}
        leg_phases = {
            1: ("grasp_target",),
            2: ("pregrasp_target", "constrained_approach"),
            3: ("envelope_ingress", "pregrasp_target", "constrained_approach"),
        }[len(selected_legs)]
        waypoint_count = 0
        for leg_index, leg in enumerate(selected_legs):
            waypoint_count += len(leg.path.waypoints) - (1 if leg_index else 0)
            leg_end_waypoint_indices[waypoint_count - 1] = str(
                leg.path.metadata.get("planning_phase", leg_phases[leg_index])
            )
        dense = self._revalidate(
            waypoints,
            moving_side=selected_side,
            deadline=deadline,
            monotonic=monotonic,
        )
        dense["stage_revalidation"] = [
            {
                "planning_phase": leg.path.metadata.get("planning_phase"),
                "path_hash": leg.path.semantic_content_hash,
                "dense_revalidation": dict(leg.dense_revalidation),
            }
            for leg in selected_legs
        ]
        search_attempts = [
            dict(attempt) for leg in selected_legs for attempt in leg.search_attempts
        ]
        velocity_profile = load_joint_velocity_limit_profiles()[
            self.profile.position_profile.ik_execution.joint_velocity_limit_profile
        ]
        joint_velocity_limit = np.deg2rad(
            np.asarray(velocity_profile.max_velocity_deg_s, dtype=float)
        )
        reference_hz = float(self.profile.position_profile.rates.ik_hz)
        times = [0.0]
        samples = {name: [current[name].copy()] for name in _SIDES}
        stage_boundaries: list[dict[str, object]] = []
        parameterization_metadata: dict[str, object]
        segment_diagnostics: list[dict[str, object]] = []
        continuous_per_leg_coverage_complete = False
        if self.config.trajectory_parameterization == (
            RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
        ):
            leg_parameterization: list[dict[str, object]] = []
            fallback_leg_count = 0
            for leg_index, leg in enumerate(selected_legs):
                self._remaining_path_planning_time_sec(
                    deadline=deadline,
                    monotonic=monotonic,
                    planning_phase="trajectory_parameterization",
                )
                phase = str(
                    leg.path.metadata.get("planning_phase", leg_phases[leg_index])
                )
                effective_speed_scale = self._effective_trajectory_speed_scale(
                    base_speed_scale=scale,
                    planning_phase=phase,
                )
                effective_joint_velocity = (
                    joint_velocity_limit * effective_speed_scale
                )
                leg_waypoints = tuple(leg.path.waypoints)
                if len(leg_waypoints) >= 2:
                    continuous, leg_segments, tcp_time_scale = (
                        self._continuous_leg_trajectory(
                            waypoints=leg_waypoints,
                            moving_side=selected_side,
                            speed_scale=effective_speed_scale,
                            joint_velocity_rad_s=effective_joint_velocity,
                            reference_hz=reference_hz,
                        )
                    )
                    leg_times = np.asarray(continuous.timestamps, dtype=float)
                    leg_q = np.asarray(continuous.q_rad, dtype=float)
                    execution_mode = (
                        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
                    )
                    fallback: dict[str, object] | None = None
                    parameterized_waypoints = tuple(
                        {
                            "left": np.asarray(row[:6], dtype=float),
                            "right": np.asarray(row[6:], dtype=float),
                        }
                        for row in leg_q
                    )
                    try:
                        collision_revalidation = self._revalidate(
                            parameterized_waypoints,
                            moving_side=selected_side,
                            deadline=deadline,
                            monotonic=monotonic,
                        )
                        visible_revalidation = None
                        if visible_scene is not None:
                            visible_revalidation = (
                                self._revalidate_visible_gripper_path(
                                    parameterized_waypoints,
                                    moving_side=selected_side,
                                    visible_scene=visible_scene,
                                    terminal_tool_tcp_world=(
                                        target_pose
                                        if phase
                                        in {
                                            "grasp_target",
                                            "constrained_approach",
                                        }
                                        else None
                                    ),
                                    deadline=deadline,
                                    monotonic=monotonic,
                                )
                            )
                    except RM65GraspTrajectoryPlanningError as error:
                        if error.reason_code not in {
                            "path_collision_blocked",
                            "path_target_safety_envelope_blocked",
                            "visible_gripper_path_collision",
                        }:
                            raise
                        fallback_leg_count += 1
                        execution_mode = (
                            "segmented_stop_collision_preserving_fallback"
                        )
                        fallback = {
                            "reason_code": error.reason_code,
                            "planning_phase": error.details.get(
                                "planning_phase"
                            ),
                            "segment_index": error.details.get("segment_index"),
                            "edge_sample_index": error.details.get(
                                "edge_sample_index"
                            ),
                            "target_collision_points": error.details.get(
                                "target_collision_points"
                            ),
                            "background_collision_points": error.details.get(
                                "background_collision_points"
                            ),
                        }
                        leg_times, leg_q, leg_segments = (
                            self._segmented_leg_trajectory(
                                waypoints=leg_waypoints,
                                moving_side=selected_side,
                                speed_scale=effective_speed_scale,
                                joint_velocity_rad_s=effective_joint_velocity,
                                reference_hz=reference_hz,
                            )
                        )
                        fallback_waypoints = tuple(
                            {
                                "left": np.asarray(row[:6], dtype=float),
                                "right": np.asarray(row[6:], dtype=float),
                            }
                            for row in leg_q
                        )
                        collision_revalidation = self._revalidate(
                            fallback_waypoints,
                            moving_side=selected_side,
                            deadline=deadline,
                            monotonic=monotonic,
                        )
                        visible_revalidation = None
                        if visible_scene is not None:
                            visible_revalidation = (
                                self._revalidate_visible_gripper_path(
                                    fallback_waypoints,
                                    moving_side=selected_side,
                                    visible_scene=visible_scene,
                                    terminal_tool_tcp_world=(
                                        target_pose
                                        if phase
                                        in {
                                            "grasp_target",
                                            "constrained_approach",
                                        }
                                        else None
                                    ),
                                    deadline=deadline,
                                    monotonic=monotonic,
                                )
                            )
                    offset = float(times[-1])
                    for sample_index in range(1, leg_times.size):
                        times.append(
                            offset
                            + float(
                                leg_times[sample_index]
                                - leg_times[0]
                            )
                        )
                        samples["left"].append(
                            np.asarray(leg_q[sample_index, :6], dtype=float)
                        )
                        samples["right"].append(
                            np.asarray(leg_q[sample_index, 6:], dtype=float)
                        )
                    segment_diagnostics.extend(
                        {
                            **dict(item),
                            "planning_phase": phase,
                            "effective_speed_scale": float(effective_speed_scale),
                            "leg_index": leg_index,
                        }
                        for item in leg_segments
                    )
                    leg_parameterization.append(
                        {
                            "planning_phase": phase,
                            "source_waypoint_count": len(leg_waypoints),
                            "sample_count": int(leg_times.size),
                            "execution_mode": execution_mode,
                            "collision_preserving_fallback": fallback,
                            "source_duration_sec": float(
                                continuous.metadata["source_duration_sec"]
                            ),
                            "execution_duration_sec": float(
                                leg_times[-1] - leg_times[0]
                            ),
                            "joint_limit_time_scale": float(
                                continuous.metadata["time_scale"]
                            ),
                            "tcp_limit_time_scale": tcp_time_scale,
                            "internal_knot_zero_velocity_required": bool(
                                fallback is not None
                            ),
                            "collision_revalidation": collision_revalidation,
                            "visible_scene_revalidation": visible_revalidation,
                            "continuous_hard_limit_certificate": (
                                continuous.metadata[
                                    "continuous_hard_limit_certificate"
                                ]
                            ),
                        }
                    )
                stage_boundaries.append(
                    {
                        "planning_phase": phase,
                        "end_sample_index": len(times) - 1,
                        "end_time_sec": float(times[-1]),
                    }
                )
            singularity_scales = [
                float(item["singularity_time_scale"])
                for item in segment_diagnostics
            ]
            singularity_replay_metadata = {
                "policy": "real_teleop_soft_traversal_v1",
                "hard_threshold_deg": float(
                    RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG
                ),
                "soft_margin_deg": float(self.config.singularity_soft_margin_deg),
                "replay_slowdown": float(
                    self.config.singularity_replay_slowdown
                ),
                "scaled_interval_count": sum(
                    value > 1.0 + 1e-12 for value in singularity_scales
                ),
                "maximum_interval_scale": max(singularity_scales, default=1.0),
                "spatial_path_unchanged": True,
                "collision_rechecked": True,
                "applied_before_continuous_parameterization": True,
            }
            parameterization_metadata = {
                "mode": RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2,
                "internal_path_waypoint_stops": bool(fallback_leg_count),
                "stage_boundary_stops": True,
                "collision_preserving_fallback_leg_count": fallback_leg_count,
                "speed_profile": self.config.trajectory_speed_profile,
                "base_speed_scale": float(scale),
                "legs": tuple(leg_parameterization),
                "segments": tuple(segment_diagnostics),
            }
            per_leg_collision = tuple(
                {
                    "planning_phase": item["planning_phase"],
                    "execution_mode": item["execution_mode"],
                    "revalidation": item["collision_revalidation"],
                }
                for item in leg_parameterization
            )
            continuous_per_leg_coverage_complete = (
                len(leg_parameterization) == len(selected_legs)
            )
            # Each leg was checked after C2 parameterization, including the
            # exact stage-boundary state shared by adjacent legs.  Singularity
            # replay changes timestamps only, so concatenating those samples
            # does not create a new spatial edge that needs a second full pass.
            dense["parameterized_trajectory_revalidation"] = {
                "coverage": "per_parameterized_leg",
                "complete": continuous_per_leg_coverage_complete,
                "execution_sample_count": len(times),
                "collision_query_count": sum(
                    int(item["collision_revalidation"].get("sample_count", 0))
                    for item in leg_parameterization
                ),
                "legs": per_leg_collision,
            }
        else:
            elapsed = 0.0
            for end_waypoint_index, (start, end) in enumerate(
                pairwise(waypoints), start=1
            ):
                self._remaining_path_planning_time_sec(
                    deadline=deadline,
                    monotonic=monotonic,
                    planning_phase="trajectory_parameterization",
                )
                segment_phase = next(
                    (
                        phase
                        for waypoint_index, phase in sorted(
                            leg_end_waypoint_indices.items()
                        )
                        if end_waypoint_index <= waypoint_index
                    ),
                    leg_phases[-1],
                )
                effective_speed_scale = self._effective_trajectory_speed_scale(
                    base_speed_scale=scale,
                    planning_phase=segment_phase,
                )
                duration, diagnostic = self._segment_duration_diagnostics(
                    start=start,
                    end=end,
                    moving_side=selected_side,
                    speed_scale=effective_speed_scale,
                    joint_velocity_rad_s=joint_velocity_limit * effective_speed_scale,
                )
                diagnostic.update(
                    {
                        "segment_index": end_waypoint_index - 1,
                        "planning_phase": segment_phase,
                        "effective_speed_scale": float(effective_speed_scale),
                    }
                )
                segment_diagnostics.append(diagnostic)
                delta = np.abs(
                    np.asarray(end[selected_side])
                    - np.asarray(start[selected_side])
                )
                frame_count = max(
                    2,
                    math.ceil(duration * reference_hz) + 1,
                    math.ceil(
                        1.875
                        * float(np.max(delta))
                        / self.config.final_edge_resolution_rad
                    )
                    + 1,
                )
                for frame_index in range(1, frame_count):
                    unit = float(frame_index) / float(frame_count - 1)
                    progress = unit**3 * (10.0 - 15.0 * unit + 6.0 * unit**2)
                    elapsed += duration / float(frame_count - 1)
                    times.append(elapsed)
                    for name in _SIDES:
                        samples[name].append(
                            np.asarray(start[name])
                            + progress
                            * (np.asarray(end[name]) - np.asarray(start[name]))
                        )
                if end_waypoint_index in leg_end_waypoint_indices:
                    stage_boundaries.append(
                        {
                            "planning_phase": leg_end_waypoint_indices[
                                end_waypoint_index
                            ],
                            "end_sample_index": len(times) - 1,
                            "end_time_sec": float(times[-1]),
                        }
                    )
            times, singularity_replay_metadata = (
                self._apply_singularity_replay_time_scaling(
                    times=times,
                    samples=samples,
                    moving_side=selected_side,
                )
            )
            parameterization_metadata = {
                "mode": RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP,
                "internal_path_waypoint_stops": True,
                "stage_boundary_stops": True,
                "speed_profile": self.config.trajectory_speed_profile,
                "base_speed_scale": float(scale),
                "segments": tuple(segment_diagnostics),
            }
        if len(times) == 1:
            elapsed = self.config.minimum_segment_duration_sec
            times.append(elapsed)
            for name in _SIDES:
                samples[name].append(current[name].copy())
        for boundary in stage_boundaries:
            boundary["end_time_sec"] = float(
                times[int(boundary["end_sample_index"])]
            )
        if (
            self.config.trajectory_parameterization
            == RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
            and not continuous_per_leg_coverage_complete
        ):
            dense["parameterized_trajectory_revalidation"] = self._revalidate(
                tuple(
                    {
                        side: np.asarray(samples[side][index], dtype=float)
                        for side in _SIDES
                    }
                    for index in range(len(times))
                ),
                moving_side=selected_side,
                deadline=deadline,
                monotonic=monotonic,
            )
        if visible_scene is not None:
            terminal_target_contact_points = 0
            terminal_target_contact_sample_count = 0
            if self.config.trajectory_parameterization == (
                RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
            ) and continuous_per_leg_coverage_complete:
                visible_revalidations = tuple(
                    item["visible_scene_revalidation"]
                    for item in leg_parameterization
                    if item["visible_scene_revalidation"] is not None
                )
                terminal_target_contact_points = max(
                    (
                        int(item["terminal_target_contact_points"])
                        for item in visible_revalidations
                    ),
                    default=0,
                )
                terminal_target_contact_sample_count = sum(
                    int(item["terminal_target_contact_sample_count"])
                    for item in visible_revalidations
                )
                visible_collision_query_count = sum(
                    int(item["sample_count"])
                    for item in visible_revalidations
                )
                visible_validation_mode = "per_parameterized_leg"
            else:
                for index in range(len(times)):
                    self._remaining_path_planning_time_sec(
                        deadline=deadline, monotonic=monotonic,
                        planning_phase="visible_scene_final_trajectory_revalidation",
                    )
                    pose = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                        left_qpos=samples["left"][index],
                        right_qpos=samples["right"][index],
                    )[selected_side]
                    (
                        terminal_contact_allowed,
                        position_error_m,
                        orientation_error_deg,
                    ) = _terminal_target_contact_region(
                        tool_tcp_world=pose,
                        terminal_tool_tcp_world=target_pose,
                        position_tolerance_m=max(
                            self.config.endpoint_position_tolerance_m,
                            RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
                        ),
                        orientation_tolerance_deg=(
                            self.config.endpoint_orientation_tolerance_deg
                        ),
                    )
                    if terminal_contact_allowed:
                        (
                            target_hits,
                            sample_terminal_target_contact_points,
                            background_hits,
                        ) = visible_scene.collision_counts_with_terminal_target_contact(
                            pose,
                            opening_m=visible_scene.geometry.maximum_opening_m,
                            contact_depth_m=(
                                RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M
                            ),
                        )
                        terminal_target_contact_points = max(
                            terminal_target_contact_points,
                            int(sample_terminal_target_contact_points),
                        )
                        terminal_target_contact_sample_count += int(
                            sample_terminal_target_contact_points > 0
                        )
                    else:
                        sample_terminal_target_contact_points = 0
                        target_hits, background_hits = visible_scene.collision_counts(
                            pose,
                            opening_m=visible_scene.geometry.maximum_opening_m,
                        )
                    if target_hits or background_hits:
                        raise RM65GraspTrajectoryPlanningError(
                            "visible_gripper_path_collision",
                            details={"edge_sample_index": index,
                                     "target_collision_points": target_hits,
                                     "background_collision_points": background_hits,
                                     "terminal_target_contact_points": (
                                         sample_terminal_target_contact_points
                                     ),
                                     "terminal_contact_allowed": terminal_contact_allowed,
                                     "terminal_position_error_m": position_error_m,
                                     "terminal_orientation_error_deg": (
                                         orientation_error_deg
                                     )},
                        )
                visible_validation_mode = "whole_parameterized_trajectory"
                visible_collision_query_count = len(times)
            dense["visible_scene_final_trajectory"] = {
                "scene_hash": visible_scene.semantic_content_hash,
                "excluded_captured_gripper_background_points": excluded_gripper_observation_count,
                "excluded_captured_robot_background_points": (
                    excluded_gripper_observation_count
                    + int(
                        0
                        if capture_self_observation_evidence is None
                        else capture_self_observation_evidence[
                            "removed_background_points"
                        ]
                    )
                ),
                "captured_robot_observation_filter": (
                    capture_self_observation_evidence
                ),
                "visible_depth_preprocessing": depth_preprocessing_evidence,
                "self_filter_opening_basis": "nominal_maximum_opening",
                "self_filter_surface_margin_m": (
                    RM65_GRASP_CAPTURE_SELF_OBSERVATION_SURFACE_MARGIN_M
                ),
                "sample_count": visible_collision_query_count,
                "execution_sample_count": len(times),
                "validation_mode": visible_validation_mode,
                "opening_m": visible_scene.geometry.maximum_opening_m,
                "terminal_target_contact_depth_m": (
                    RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M
                ),
                "terminal_target_contact_points": int(
                    terminal_target_contact_points
                ),
                "terminal_target_contact_sample_count": int(
                    terminal_target_contact_sample_count
                ),
                "terminal_contact_position_tolerance_m": float(
                    max(
                        self.config.endpoint_position_tolerance_m,
                        RM65_GRASP_TERMINAL_TARGET_CONTACT_DEPTH_M,
                    )
                ),
                "terminal_contact_orientation_tolerance_deg": float(
                    self.config.endpoint_orientation_tolerance_deg
                ),
                "scope": "observed_points_nominal_parallel_jaw_solids_only",
            }
        planning_timing = {
            "clock": "host_perf_counter",
            "pre_path_search_wall_duration_sec": float(
                path_search_started_at - planning_started_at
            ),
            "path_search_wall_duration_sec": float(
                path_search_finished_at - path_search_started_at
            ),
            "trajectory_parameterization_wall_duration_sec": float(
                time.perf_counter() - path_search_finished_at
            ),
            "planning_wall_duration_sec": float(
                time.perf_counter() - planning_started_at
            ),
            "path_planning_time_budget_sec": float(
                self.config.path_planning_time_budget_sec
            ),
        }
        evidence_hash = canonical_content_hash(
            {
                "scene_hash": self.scene.semantic_content_hash,
                "path_hash": selected_legs[-1].path.semantic_content_hash,
                "path_leg_hashes": [
                    leg.path.semantic_content_hash for leg in selected_legs
                ],
                "dense_revalidation": dense,
                "singularity_replay": singularity_replay_metadata,
                "trajectory_parameterization": parameterization_metadata,
                "envelope_ingress_route": ingress_route,
                "envelope_ingress_route_attempts": ingress_route_attempts,
                "pregrasp_route_attempts": pregrasp_route_attempts,
                "endpoint_seed_index": endpoint.seed_index,
                "endpoint_solve_cycles": endpoint.solve_cycles,
                "endpoint_position_residual_m": endpoint.position_residual_m,
                "endpoint_orientation_residual_deg": (
                    endpoint.orientation_residual_deg
                ),
                "path_search_attempts": search_attempts,
                "pregrasp_tool_tcp_world": pregrasp_pose,
                "route_mode": route_mode or "pregrasp_and_fixed_approach",
                "direct_grasp_target_fallback": direct_fallback_attempt,
                "unconstrained_terminal_fallback_enabled": (
                    self.config.allow_unconstrained_terminal_fallback
                ),
                "pregrasp_applied": (
                    not already_at_grasp
                    and route_mode != "global_grasp_target_fallback"
                ),
                "selected_pregrasp_distance_m": (
                    0.0
                    if already_at_grasp
                    or route_mode == "global_grasp_target_fallback"
                    else float(
                        np.linalg.norm(
                            target_pose[:3, 3] - pregrasp_pose[:3, 3]
                        )
                    )
                ),
                "maximum_pregrasp_distance_m": self.config.pregrasp_distance_m,
                "minimum_pregrasp_distance_m": (
                    self.config.minimum_pregrasp_distance_m
                ),
                "approach_waypoint_spacing_m": (
                    self.config.approach_waypoint_spacing_m
                ),
                "target_tool_tcp_world": target_pose,
                "speed_scale": scale,
                "trajectory_speed_profile": self.config.trajectory_speed_profile,
                "reference_hz": reference_hz,
            },
            domain="dexteleop.real_agent_motion.rm65_grasp_path.v1",
        )
        trajectory = RealAgentJointTrajectory(
            time_from_start_sec=np.asarray(times, dtype=float),
            qpos_by_side={name: np.stack(samples[name], axis=0) for name in _SIDES},
            planner_evidence_refs=("rm65-grasp-path:" + evidence_hash,),
        )
        if self.trajectory_record_dir is not None:
            try:
                write_grasp_trajectory_record(
                    self.trajectory_record_dir,
                    trajectory,
                    moving_side=selected_side,
                    scene_hash=self.scene.semantic_content_hash,
                    collision_scope="profile_sparse_proxy_pairs",
                    collision_scene_profile=self.path_collision_scene_profile,
                    target_tool_tcp_world=target_pose,
                    speed_scale=scale,
                    stage_boundaries=tuple(stage_boundaries),
                    parameterization=parameterization_metadata,
                    planning_timing=planning_timing,
                    source_waypoints_by_side={
                        side: np.stack(
                            [
                                np.asarray(waypoint[side], dtype=float)
                                for waypoint in waypoints
                            ],
                            axis=0,
                        )
                        for side in _SIDES
                    },
                )
            except (OSError, ValueError, BadZipFile) as error:
                raise RM65GraspTrajectoryPlanningError(
                    "trajectory_record_failed",
                    details={"exception_type": type(error).__name__},
                ) from error
        return trajectory


__all__ = [
    "RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME",
    "RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE",
    "RM65_GRASP_ENDPOINT_ORDER_MODES",
    "RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC",
    "RM65_GRASP_SINGULARITY_HARD_THRESHOLD_DEG",
    "RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2",
    "RM65_GRASP_TRAJECTORY_PARAMETERIZATION_MODES",
    "RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP",
    "RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X",
    "RM65_GRASP_TRAJECTORY_SPEED_PROFILE_MODES",
    "RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM",
    "RM65GraspJointTrajectoryPlanner",
    "RM65GraspEndpointProbeResult",
    "RM65GraspLocalChainResult",
    "RM65GraspReachabilityConfig",
    "RM65GraspReachabilityResult",
    "RM65GraspTrajectoryPlanningConfig",
    "RM65GraspTrajectoryPlanningError",
]
