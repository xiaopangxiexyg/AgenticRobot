"""URDF/FCL-only planning for one real RM65 grasp-approach Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import NoReturn

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.path_planning import (
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
    plan_collision_aware_joint_path,
)
from robot_motion.pose import pose_delta_metrics
from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65DualArmCollisionIKConfig,
    build_dual_arm_scene_profile,
    build_rm65_dual_arm_collision_ik_solver,
)
from robot_motion.rm65.obstacle_collision import (
    RM65_D405_ATTACHED_COLLISION_PROXIES,
    RM65ObstacleCollisionOracle,
)
from robot_motion.rm65.scene_profiles import fixed_scene_aabb_obstacles
from robot_motion.scene_obstacles import ObstacleSceneSnapshot

from .motion_port import RealAgentJointTrajectory
from .rm65_profile import ResolvedExclusiveRM65SkillDeploymentProfile

_SIDES = ("left", "right")
_STRICT_SCENE_PROFILE = "dual_arm_hand_proxy_strict"


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
    maximum_endpoint_solve_cycles: int = 180
    endpoint_seed_count: int = 24
    endpoint_seed_perturbation_rad: float = float(np.deg2rad(25.0))
    maximum_endpoint_candidates: int = 8
    endpoint_position_residual_scale_m: float = 0.01
    endpoint_orientation_residual_scale_deg: float = 10.0
    endpoint_tolerance_margin_ratio: float = 0.90
    path_search_attempts_per_endpoint: int = 4
    path_planning_time_budget_sec: float = 120.0
    endpoint_position_tolerance_m: float = 0.003
    endpoint_orientation_tolerance_deg: float = 1.0
    final_edge_resolution_rad: float = float(np.deg2rad(0.1))
    maximum_path_envelope_ingress_distance_m: float = 0.15
    path_envelope_ingress_progress_tolerance_m: float = 0.002
    path_envelope_ingress_tangent_tolerance_m: float = 0.0005
    path_envelope_ingress_waypoint_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.35,
    )

    def __post_init__(self) -> None:
        for name in (
            "minimum_clearance_m",
            "linear_speed_m_s",
            "angular_speed_deg_s",
            "minimum_segment_duration_sec",
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
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
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


class RM65GraspJointTrajectoryPlanner:
    """Resolve endpoint IK, plan around fixed table/poles, then time-sample qpos."""

    def __init__(
        self,
        *,
        profile: ResolvedExclusiveRM65SkillDeploymentProfile,
        planning_target_builder: object,
        target_envelope_filter: Callable[
            [Mapping[str, np.ndarray]], Mapping[str, np.ndarray] | None
        ]
        | None,
        config: RM65GraspTrajectoryPlanningConfig | None = None,
        planner_config: JointPathPlannerConfig | None = None,
        reachability_config: RM65GraspReachabilityConfig | None = None,
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
        self._solver_lock = threading.Lock()
        self.planner_config = planner_config or JointPathPlannerConfig(
            shortcut_attempts=8,
            adaptive_edge_resolution_rad=self.config.final_edge_resolution_rad,
            adaptive_refinement_distance_m=0.03,
        )
        strict_profile = build_dual_arm_scene_profile(_STRICT_SCENE_PROFILE)
        fixed_obstacles = fixed_scene_aabb_obstacles(strict_profile)
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
                "collision_scene_profile": _STRICT_SCENE_PROFILE,
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
        collision_bundle = build_rm65_dual_arm_collision_ik_solver(
            RM65DualArmCollisionIKConfig(scene_profile=_STRICT_SCENE_PROFILE)
        )
        self.oracle = RM65ObstacleCollisionOracle.from_solver(
            collision_bundle.solver,
            scene=self.scene,
            solution_layout=collision_bundle.scene_profile.solution_layout,
            expected_reference_frame_id=self.scene.reference_frame_id,
            expected_reference_frame_spec_hash=self.scene.reference_frame_spec_hash,
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
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
            projected = self.target_envelope_filter(
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

    def _bounded_ingress_projection(
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
        if len(raw_constraints) != 1:
            self._raise_ingress_rejection(
                details,
                "multiple_initial_constraints_are_not_supported",
            )
        if float(adjustment) > self.config.maximum_path_envelope_ingress_distance_m:
            self._raise_ingress_rejection(
                details,
                "initial_distance_exceeds_limit",
            )
        return float(adjustment), frozenset(raw_constraints)

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

        boundary = np.asarray(projected[side], dtype=float)
        final_solver = np.asarray(final_solver_targets[side], dtype=float)
        targets: list[tuple[float, np.ndarray]] = []
        for fraction in self.config.path_envelope_ingress_waypoint_fractions:
            ingress_solver = boundary.copy()
            ingress_solver[:3, 3] += fraction * (final_solver[:3, 3] - boundary[:3, 3])
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
            target = self.target_builder.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=ingress_solver,
            )
            targets.append((fraction, np.asarray(target, dtype=float).copy()))
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
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> tuple[_EndpointIKCandidate, ...]:
        with self._solver_lock:
            return self._endpoint_qpos_candidates_locked(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_tool_tcp_world,
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
                np.allclose(candidate[side], existing[side], atol=1e-9, rtol=0.0)
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
                np.allclose(candidate[side], existing[side], atol=1e-9, rtol=0.0)
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
        if (
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

        previous_alpha = 0.0
        bracket: tuple[float, float] | None = None
        for alpha in np.linspace(0.0, 1.0, 33)[1:]:
            _pair, position, orientation = evaluate(float(alpha))
            if position <= position_limit and orientation <= orientation_limit:
                bracket = (previous_alpha, float(alpha))
                break
            previous_alpha = float(alpha)
        if bracket is None:
            return evaluate(1.0)
        lower, upper = bracket
        for _ in range(32):
            midpoint = (lower + upper) * 0.5
            _pair, position, orientation = evaluate(midpoint)
            if position <= position_limit and orientation <= orientation_limit:
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
        inactive = "right" if side == "left" else "left"
        attempts: list[dict[str, object]] = []
        accepted: list[_EndpointIKCandidate] = []
        collision_unavailable = False
        collision_blocked = False
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
                        "time_budget_sec": (
                            self.config.path_planning_time_budget_sec
                        ),
                        "endpoint_seed_count_attempted": len(attempts),
                        "attempts": attempts,
                    },
                )
            outcome = "ik_search_exhausted"
            evaluations = 0
            try:
                candidate, evaluations, position, orientation = (
                    self._least_squares_endpoint_from_seed(
                        qpos_by_side=qpos_by_side,
                        side=side,
                        target_tool_tcp_world=target_tool_tcp_world,
                        seed_qpos=seed[side],
                        maximum_evaluations=(self.config.maximum_endpoint_solve_cycles),
                    )
                )
                candidate[inactive] = np.asarray(
                    qpos_by_side[inactive], dtype=float
                ).copy()
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
                and orientation <= self.config.endpoint_orientation_tolerance_deg
            ):
                if any(
                    np.allclose(
                        candidate[side],
                        item.qpos_by_side[side],
                        atol=1e-5,
                        rtol=0.0,
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
                        collision_blocked = True
                        outcome = "endpoint_collision_blocked"
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
                }
            )
        if (
            deadline is not None
            and not accepted
            and float(monotonic()) >= deadline
        ):
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
            accepted.sort(
                key=lambda item: float(
                    np.linalg.norm(np.asarray(item.qpos_by_side[side]) - current)
                )
            )
            return tuple(accepted[: self.config.maximum_endpoint_candidates])
        if collision_unavailable:
            reason = "endpoint_collision_check_unavailable"
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
                / self.config.endpoint_orientation_tolerance_deg,
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
                "attempts": attempts,
            },
        )

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

        def blocked_projection(
            error: RM65GraspTrajectoryPlanningError,
        ) -> tuple[float, frozenset[str]]:
            if tuple(error.details.get("modified_sides", ())) != (moving_side,):
                reject_ingress(error, "modified_sides_are_not_the_moving_side_only")
            projection_by_side = error.details.get("envelope_projection_by_side")
            projection = (
                projection_by_side.get(moving_side)
                if isinstance(projection_by_side, Mapping)
                else None
            )
            if not isinstance(projection, Mapping):
                reject_ingress(error, "projection_diagnostics_unavailable")
            if (
                projection.get("translation_modified") is not True
                or projection.get("rotation_modified") is not False
            ):
                reject_ingress(error, "violation_is_not_translation_only")
            adjustment = projection.get("translation_adjustment_m")
            if (
                isinstance(adjustment, bool)
                or not isinstance(adjustment, (int, float))
                or not math.isfinite(float(adjustment))
                or float(adjustment) <= 0.0
            ):
                reject_ingress(error, "translation_adjustment_is_invalid")
            raw_constraints = projection.get("violated_translation_constraints")
            if (
                not isinstance(raw_constraints, (tuple, list))
                or not raw_constraints
                or any(
                    not isinstance(item, str) or not item.startswith("absolute:")
                    for item in raw_constraints
                )
            ):
                reject_ingress(error, "violation_is_not_an_absolute_translation_bound")
            return float(adjustment), frozenset(raw_constraints)

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
                    if ingress_entered_at is not None:
                        reject_ingress(error, "path_left_envelope_after_entry")
                    adjustment, constraints = blocked_projection(error)
                    if ingress_constraints is None:
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
                        unexpected_constraints = constraints - ingress_constraints
                        if (
                            unexpected_constraints
                            and adjustment
                            > self.config.path_envelope_ingress_tangent_tolerance_m
                        ):
                            reject_ingress(error, "violation_constraint_set_expanded")
                        ingress_observed_constraints.update(constraints)
                        assert ingress_initial_adjustment_m is not None
                        assert ingress_best_adjustment_m is not None
                        tolerance = (
                            self.config.path_envelope_ingress_progress_tolerance_m
                        )
                        if adjustment > ingress_initial_adjustment_m + tolerance:
                            reject_ingress(
                                error, "path_exceeded_initial_outer_boundary"
                            )
                        if adjustment > ingress_best_adjustment_m + tolerance:
                            reject_ingress(error, "path_moved_away_from_envelope")
                        ingress_best_adjustment_m = min(
                            ingress_best_adjustment_m,
                            adjustment,
                        )
                else:
                    if ingress_constraints is not None and ingress_entered_at is None:
                        ingress_entered_at = {
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                        }
                check = self.oracle.check(qpos)
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
                    minimum_distance = min(minimum_distance, check.minimum_distance_m)
                if not check.collision_check_available or not check.safe:
                    raise RM65GraspTrajectoryPlanningError(
                        "path_collision_check_unavailable"
                        if not check.collision_check_available
                        else "path_collision_blocked",
                        details={
                            "segment_index": segment_index,
                            "edge_sample_index": index,
                            "collision_check": check.to_dict(),
                        },
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

    def _plan_path_leg(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        planning_phase: str,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> _PlannedPathLeg:
        if deadline is not None:
            self._remaining_path_planning_time_sec(
                deadline=deadline,
                monotonic=monotonic,
                planning_phase=planning_phase,
            )
        endpoint_candidates = self._endpoint_qpos_candidates(
            qpos_by_side=qpos_by_side,
            side=side,
            target_tool_tcp_world=target_tool_tcp_world,
            deadline=deadline,
            monotonic=monotonic,
        )
        lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
        upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
        cfg = self.planner_config
        search_attempts: list[dict[str, object]] = []
        terminal_error: RM65GraspTrajectoryPlanningError | None = None
        for endpoint_index, endpoint in enumerate(endpoint_candidates):
            target = endpoint.qpos_by_side
            for path_attempt in range(self.config.path_search_attempts_per_endpoint):
                full_joint_range = (
                    path_attempt == self.config.path_search_attempts_per_endpoint - 1
                )
                padding = cfg.sample_padding_rad * float(2**path_attempt)
                corridor_lower = (
                    lower
                    if full_joint_range
                    else np.maximum(
                        lower,
                        np.minimum(qpos_by_side[side], target[side]) - padding,
                    )
                )
                corridor_upper = (
                    upper
                    if full_joint_range
                    else np.minimum(
                        upper,
                        np.maximum(qpos_by_side[side], target[side]) + padding,
                    )
                )
                attempt_config = replace(
                    cfg,
                    seed=(
                        cfg.seed
                        + endpoint_index * self.config.path_search_attempts_per_endpoint
                        + path_attempt
                    ),
                    time_budget_sec=(
                        cfg.time_budget_sec
                        if deadline is None
                        else min(
                            self._remaining_path_planning_time_sec(
                                deadline=deadline,
                                monotonic=monotonic,
                                planning_phase=planning_phase,
                                details={
                                    "endpoint_seed_index": endpoint.seed_index,
                                    "path_attempt": path_attempt,
                                },
                            ),
                            (
                                math.inf
                                if cfg.time_budget_sec is None
                                else cfg.time_budget_sec
                            ),
                        )
                    ),
                )
                path = plan_collision_aware_joint_path(
                    start_qpos_by_component=qpos_by_side,
                    target_qpos_by_component=target,
                    movable_components=(side,),
                    oracle=self.oracle,
                    config=attempt_config,
                    joint_lower_by_component={side: corridor_lower},
                    joint_upper_by_component={side: corridor_upper},
                    scene_hash=self.scene.semantic_content_hash,
                    monotonic=monotonic,
                )
                attempt_record: dict[str, object] = {
                    "planning_phase": planning_phase,
                    "endpoint_seed_index": endpoint.seed_index,
                    "path_attempt": path_attempt,
                    "search_seed": attempt_config.seed,
                    "full_joint_range": full_joint_range,
                    "sample_padding_rad": None if full_joint_range else padding,
                    "status": path.status.value,
                    "reason": path.reason,
                }
                search_attempts.append(attempt_record)
                if path.status is not JointPathStatus.PATH_FOUND:
                    if path.reason == "time_budget_exhausted":
                        terminal_error = RM65GraspTrajectoryPlanningError(
                            "path_planning_time_budget_exhausted",
                            details={
                                "planning_phase": planning_phase,
                                "time_budget_sec": (
                                    self.config.path_planning_time_budget_sec
                                ),
                                "path": path.to_dict(),
                                "search_attempts": search_attempts,
                            },
                        )
                        break
                    if path.status is JointPathStatus.COLLISION_CHECK_UNAVAILABLE:
                        terminal_error = RM65GraspTrajectoryPlanningError(
                            "path_collision_check_unavailable",
                            details={
                                "planning_phase": planning_phase,
                                "path": path.to_dict(),
                                "search_attempts": search_attempts,
                            },
                        )
                        break
                    continue
                try:
                    dense = self._revalidate(
                        path.waypoints,
                        moving_side=side,
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    attempt_record["dense_revalidation_error"] = error.reason_code
                    terminal_error = error
                    if error.reason_code in {
                        "path_collision_check_unavailable",
                        "path_planning_time_budget_exhausted",
                    }:
                        break
                    continue
                return _PlannedPathLeg(
                    endpoint=endpoint,
                    path=path,
                    dense_revalidation=dense,
                    search_attempts=tuple(search_attempts),
                )
            if (
                terminal_error is not None
                and terminal_error.reason_code
                in {
                    "path_collision_check_unavailable",
                    "path_planning_time_budget_exhausted",
                }
            ):
                break

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

    def __call__(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        speed_scale: float,
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
        scale = float(speed_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale must satisfy 0 < speed_scale <= 1.")
        deadline = (
            float(monotonic()) + self.config.path_planning_time_budget_sec
        )
        ingress_targets = self._envelope_ingress_targets(
            qpos_by_side=current,
            side=selected_side,
            final_tool_tcp_world=target_pose,
        )
        ingress_route_attempts: list[dict[str, object]] = []
        selected_legs: tuple[_PlannedPathLeg, ...] | None = None
        ingress_route: dict[str, object] | None = None
        route_error: RM65GraspTrajectoryPlanningError | None = None
        if not ingress_targets:
            selected_legs = (
                self._plan_path_leg(
                    qpos_by_side=current,
                    side=selected_side,
                    target_tool_tcp_world=target_pose,
                    planning_phase="grasp_target",
                    deadline=deadline,
                    monotonic=monotonic,
                ),
            )
        else:
            for inward_fraction, ingress_target in ingress_targets:
                route_attempt: dict[str, object] = {
                    "inward_fraction": inward_fraction,
                }
                ingress_route_attempts.append(route_attempt)
                try:
                    ingress_leg = self._plan_path_leg(
                        qpos_by_side=current,
                        side=selected_side,
                        target_tool_tcp_world=ingress_target,
                        planning_phase="envelope_ingress",
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                    ingress_qpos = ingress_leg.endpoint.qpos_by_side
                    grasp_leg = self._plan_path_leg(
                        qpos_by_side=ingress_qpos,
                        side=selected_side,
                        target_tool_tcp_world=target_pose,
                        planning_phase="grasp_target",
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                    combined_waypoints = (
                        *ingress_leg.path.waypoints,
                        *grasp_leg.path.waypoints[1:],
                    )
                    combined_dense = self._revalidate(
                        combined_waypoints,
                        moving_side=selected_side,
                        deadline=deadline,
                        monotonic=monotonic,
                    )
                except RM65GraspTrajectoryPlanningError as error:
                    route_error = error
                    route_attempt.update(
                        {
                            "status": "rejected",
                            "reason_code": error.reason_code,
                            "planning_phase": error.details.get("planning_phase"),
                        }
                    )
                    if error.reason_code in {
                        "path_collision_check_unavailable",
                        "path_planning_time_budget_exhausted",
                    }:
                        break
                    continue
                route_attempt["status"] = "path_found"
                selected_legs = (ingress_leg, grasp_leg)
                ingress_route = {
                    "inward_fraction": inward_fraction,
                    "ingress_target_tool_tcp_world": ingress_target,
                    "ingress_endpoint_seed_index": ingress_leg.endpoint.seed_index,
                    "ingress_path_hash": ingress_leg.path.semantic_content_hash,
                }
                dense = combined_dense
                break
            if selected_legs is None:
                if route_error is None:
                    route_error = RM65GraspTrajectoryPlanningError("path_planner_error")
                raise RM65GraspTrajectoryPlanningError(
                    route_error.reason_code,
                    details={
                        **route_error.details,
                        "envelope_ingress_route_attempts": ingress_route_attempts,
                    },
                ) from route_error

        assert selected_legs is not None
        self._remaining_path_planning_time_sec(
            deadline=deadline,
            monotonic=monotonic,
            planning_phase="trajectory_parameterization",
        )
        if ingress_route is None:
            dense = dict(selected_legs[0].dense_revalidation)
        endpoint = selected_legs[-1].endpoint
        waypoints = tuple(
            waypoint
            for leg_index, leg in enumerate(selected_legs)
            for waypoint in (
                leg.path.waypoints if leg_index == 0 else leg.path.waypoints[1:]
            )
        )
        search_attempts = [
            dict(attempt) for leg in selected_legs for attempt in leg.search_attempts
        ]
        velocity_profile = load_joint_velocity_limit_profiles()[
            self.profile.position_profile.ik_execution.joint_velocity_limit_profile
        ]
        joint_velocity = (
            np.deg2rad(np.asarray(velocity_profile.max_velocity_deg_s, dtype=float))
            * scale
        )
        reference_hz = float(self.profile.position_profile.rates.ik_hz)
        times = [0.0]
        samples = {name: [current[name].copy()] for name in _SIDES}
        elapsed = 0.0
        for start, end in pairwise(waypoints):
            self._remaining_path_planning_time_sec(
                deadline=deadline,
                monotonic=monotonic,
                planning_phase="trajectory_parameterization",
            )
            start_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=start["left"], right_qpos=start["right"]
            )[selected_side]
            end_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=end["left"], right_qpos=end["right"]
            )[selected_side]
            translation, rotation_deg = pose_delta_metrics(start_tcp, end_tcp)
            delta = np.abs(
                np.asarray(end[selected_side]) - np.asarray(start[selected_side])
            )
            duration = max(
                self.config.minimum_segment_duration_sec,
                1.875 * translation / (self.config.linear_speed_m_s * scale),
                1.875 * rotation_deg / (self.config.angular_speed_deg_s * scale),
                1.875 * float(np.max(delta / joint_velocity)),
            )
            frame_count = max(2, math.ceil(duration * reference_hz) + 1)
            for frame_index in range(1, frame_count):
                unit = float(frame_index) / float(frame_count - 1)
                progress = unit**3 * (10.0 - 15.0 * unit + 6.0 * unit**2)
                elapsed += duration / float(frame_count - 1)
                times.append(elapsed)
                for name in _SIDES:
                    samples[name].append(
                        np.asarray(start[name])
                        + progress * (np.asarray(end[name]) - np.asarray(start[name]))
                    )
        if len(times) == 1:
            elapsed = self.config.minimum_segment_duration_sec
            times.append(elapsed)
            for name in _SIDES:
                samples[name].append(current[name].copy())
        evidence_hash = canonical_content_hash(
            {
                "scene_hash": self.scene.semantic_content_hash,
                "path_hash": selected_legs[-1].path.semantic_content_hash,
                "path_leg_hashes": [
                    leg.path.semantic_content_hash for leg in selected_legs
                ],
                "dense_revalidation": dense,
                "envelope_ingress_route": ingress_route,
                "envelope_ingress_route_attempts": ingress_route_attempts,
                "endpoint_seed_index": endpoint.seed_index,
                "endpoint_solve_cycles": endpoint.solve_cycles,
                "endpoint_position_residual_m": endpoint.position_residual_m,
                "endpoint_orientation_residual_deg": (
                    endpoint.orientation_residual_deg
                ),
                "path_search_attempts": search_attempts,
                "target_tool_tcp_world": target_pose,
                "speed_scale": scale,
                "reference_hz": reference_hz,
            },
            domain="dexteleop.real_agent_motion.rm65_grasp_path.v1",
        )
        return RealAgentJointTrajectory(
            time_from_start_sec=np.asarray(times, dtype=float),
            qpos_by_side={name: np.stack(samples[name], axis=0) for name in _SIDES},
            planner_evidence_refs=("rm65-grasp-path:" + evidence_hash,),
        )


__all__ = [
    "RM65GraspJointTrajectoryPlanner",
    "RM65GraspReachabilityConfig",
    "RM65GraspReachabilityResult",
    "RM65GraspTrajectoryPlanningConfig",
    "RM65GraspTrajectoryPlanningError",
]
