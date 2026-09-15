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
    JointConfigurationCheck,
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
_STRICT_SCENE_PROFILE = "dual_arm_hand_proxy_grasp_full"
RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC = 120.0


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
    maximum_approach_orientation_error_deg: float = 1.0
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
            "pregrasp_distance_m",
            "minimum_pregrasp_distance_m",
            "pregrasp_distance_step_m",
            "approach_waypoint_spacing_m",
            "maximum_approach_joint_step_rad",
            "maximum_approach_lateral_error_m",
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
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
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
        self.scene_hash = getattr(collision_oracle, "scene_hash")

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
            shortcut_attempts=80,
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
                "arm_collision_representation": "cad_convex_hulls_v1",
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
        self._use_convex_arm_meshes(collision_bundle.solver.robot.collision_model)
        self.oracle = RM65ObstacleCollisionOracle.from_solver(
            collision_bundle.solver,
            scene=self.scene,
            solution_layout=collision_bundle.scene_profile.solution_layout,
            expected_reference_frame_id=self.scene.reference_frame_id,
            expected_reference_frame_spec_hash=self.scene.reference_frame_spec_hash,
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
        )

    @staticmethod
    def _use_convex_arm_meshes(collision_model: object) -> None:
        """Match MuJoCo's conservative CAD collision representation, not triangles.

        The collision bundle is private to this planner.  The oracle clones it
        afterward; the live IK solver and on-disk meshes are not modified.
        """

        expected = {
            f"{prefix}arm_link{index}_0"
            for prefix in ("", "right_")
            for index in range(1, 7)
        }
        geometries = {
            str(geometry.name): geometry
            for geometry in collision_model.geometryObjects
            if str(geometry.name) in expected
        }
        if set(geometries) != expected:
            raise RM65GraspTrajectoryPlanningError(
                "arm_collision_model_unavailable",
                details={"missing_geometries": sorted(expected - set(geometries))},
            )
        for name, geometry in geometries.items():
            mesh = geometry.geometry
            try:
                # A false return can mean the hull was already cached.  The
                # actual postcondition is a retained, non-null convex shape.
                mesh.buildConvexHull(True, "Qt")
                hull = mesh.convex
            except (AttributeError, RuntimeError, ValueError) as error:
                raise RM65GraspTrajectoryPlanningError(
                    "arm_collision_model_unavailable",
                    details={
                        "geometry_name": name,
                        "exception_type": type(error).__name__,
                    },
                ) from error
            if hull is None:
                raise RM65GraspTrajectoryPlanningError(
                    "arm_collision_model_unavailable",
                    details={"geometry_name": name, "reason": "convex_hull_missing"},
                )
            geometry.geometry = hull

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

    def _path_search_oracle(self, *, planning_phase: str) -> object:
        # The measured staging pose may intentionally start just outside the
        # task envelope.  Its dedicated ingress leg is bounded, monotonic, and
        # densely revalidated.  Once inside, every grasp-search sample must
        # satisfy both collision and TCP-envelope constraints.
        if (
            planning_phase not in {"pregrasp_target", "grasp_target"}
            or self.target_envelope_filter is None
        ):
            return self.oracle
        return _TargetEnvelopePathOracle(
            collision_oracle=self.oracle,
            envelope_check=lambda qpos: self._require_qpos_within_target_envelope(
                qpos,
                sides=_SIDES,
                phase="path_target",
            ),
        )

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
            "envelope_projection_subject": "current_measured_solver_tcp",
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
        prefer_closest_tolerance: bool = True,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> tuple[_EndpointIKCandidate, ...]:
        with self._solver_lock:
            return self._endpoint_qpos_candidates_locked(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_tool_tcp_world,
                prefer_closest_tolerance=prefer_closest_tolerance,
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
                and orientation <= self.config.endpoint_orientation_tolerance_deg
            ):
                if not self._qpos_is_within_target_envelope(
                    candidate,
                    sides=(side,),
                    phase="endpoint",
                ):
                    endpoint_envelope_blocked = True
                    outcome = "endpoint_safety_envelope_blocked"
                elif any(
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
            accepted.sort(
                key=lambda item: float(
                    np.linalg.norm(np.asarray(item.qpos_by_side[side]) - current)
                )
            )
            return tuple(accepted[: self.config.maximum_endpoint_candidates])
        if collision_unavailable:
            reason = "endpoint_collision_check_unavailable"
        elif endpoint_envelope_blocked:
            reason = "endpoint_safety_envelope_blocked"
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
                    adjustment, constraints = blocked_projection(error)
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
        search_oracle = self._path_search_oracle(planning_phase=planning_phase)
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
                    oracle=search_oracle,
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
            if terminal_error is not None and terminal_error.reason_code in {
                "path_collision_check_unavailable",
                "path_planning_time_budget_exhausted",
            }:
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

    def _plan_constrained_approach_leg(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> _PlannedPathLeg:
        """Resolve a fixed-orientation, same-branch Cartesian grasp approach."""

        targets = self._constrained_approach_targets(
            pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
            grasp_tool_tcp_world=grasp_tool_tcp_world,
        )
        inactive = "right" if side == "left" else "left"
        current = {
            name: np.asarray(qpos_by_side[name], dtype=float).copy()
            for name in _SIDES
        }
        start = {name: value.copy() for name, value in current.items()}
        waypoints: list[Mapping[str, np.ndarray]] = [start]
        attempts: list[dict[str, object]] = []
        total_cycles = 0
        final_position_residual_m = math.inf
        final_orientation_residual_deg = math.inf

        with self._solver_lock:
            for waypoint_index, target in enumerate(targets, start=1):
                if deadline is not None:
                    self._remaining_path_planning_time_sec(
                        deadline=deadline,
                        monotonic=monotonic,
                        planning_phase="constrained_approach_ik",
                        details={
                            "approach_waypoint_index": waypoint_index,
                            "approach_waypoint_count": len(targets),
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
                            "approach_waypoint_count": len(targets),
                            "exception_type": type(exc).__name__,
                            "attempts": attempts,
                        },
                    ) from exc
                total_cycles += cycles
                candidate[inactive] = start[inactive].copy()
                attempt = {
                    "planning_phase": "constrained_approach",
                    "approach_waypoint_index": waypoint_index,
                    "approach_waypoint_count": len(targets),
                    "solve_cycles": cycles,
                    "position_residual_m": position,
                    "orientation_residual_deg": orientation,
                }
                attempts.append(attempt)
                if (
                    position > self.config.endpoint_position_tolerance_m
                    or orientation > self.config.endpoint_orientation_tolerance_deg
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
                                self.config.endpoint_orientation_tolerance_deg
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
                    attempt["outcome"] = "collision_blocked"
                    raise RM65GraspTrajectoryPlanningError(
                        "approach_endpoint_collision_blocked",
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

        waypoint_tuple = tuple(waypoints)
        dense = self._revalidate(
            waypoint_tuple,
            moving_side=side,
            deadline=deadline,
            monotonic=monotonic,
        )
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
            start_qpos_by_component=start,
            target_qpos_by_component=current,
            movable_components=(side,),
            waypoints=waypoint_tuple,
            scene_hash=self.scene.semantic_content_hash,
            metadata={
                "planning_phase": "constrained_approach",
                "approach_waypoint_count": len(targets),
                "approach_geometry": approach_geometry,
            },
        )
        endpoint = _EndpointIKCandidate(
            qpos_by_side=current,
            seed_index=-1,
            solve_cycles=total_cycles,
            position_residual_m=final_position_residual_m,
            orientation_residual_deg=final_orientation_residual_deg,
        )
        return _PlannedPathLeg(
            endpoint=endpoint,
            path=path,
            dense_revalidation=dense,
            search_attempts=tuple(attempts),
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
    ) -> tuple[_PlannedPathLeg, _PlannedPathLeg]:
        """Validate each local IK branch before its exact global connection."""

        endpoints = self._endpoint_qpos_candidates(
            qpos_by_side=qpos_by_side,
            side=side,
            target_tool_tcp_world=pregrasp_tool_tcp_world,
            prefer_closest_tolerance=False,
            deadline=deadline,
            monotonic=monotonic,
        )
        attempts: list[dict[str, object]] = []
        last_error: RM65GraspTrajectoryPlanningError | None = None
        for endpoint in endpoints:
            attempt: dict[str, object] = {
                "planning_phase": "pregrasp_branch_selection",
                "endpoint_seed_index": endpoint.seed_index,
            }
            attempts.append(attempt)
            try:
                approach = self._plan_constrained_approach_leg(
                    qpos_by_side=endpoint.qpos_by_side,
                    side=side,
                    pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                    grasp_tool_tcp_world=grasp_tool_tcp_world,
                    deadline=deadline,
                    monotonic=monotonic,
                )
                pregrasp = self._plan_path_leg(
                    qpos_by_side=qpos_by_side,
                    side=side,
                    target_tool_tcp_world=pregrasp_tool_tcp_world,
                    planning_phase="pregrasp_target",
                    prefer_closest_tolerance=False,
                    endpoint_candidates=(endpoint,),
                    deadline=deadline,
                    monotonic=monotonic,
                )
            except RM65GraspTrajectoryPlanningError as error:
                last_error = error
                attempt.update(
                    {"status": "rejected", "reason_code": error.reason_code}
                )
                if error.reason_code in {
                    "path_collision_check_unavailable",
                    "path_planning_time_budget_exhausted",
                    "approach_endpoint_collision_check_unavailable",
                    "path_target_safety_envelope_unavailable",
                    "approach_endpoint_safety_envelope_unavailable",
                }:
                    break
                continue
            attempt["status"] = "path_found"
            return (
                replace(
                    pregrasp,
                    search_attempts=tuple(attempts) + pregrasp.search_attempts,
                ),
                approach,
            )
        assert last_error is not None
        raise RM65GraspTrajectoryPlanningError(
            last_error.reason_code,
            details={**last_error.details, "pregrasp_branch_attempts": attempts},
        ) from last_error

    def _plan_chain_for_pregrasp(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        pregrasp_tool_tcp_world: np.ndarray,
        grasp_tool_tcp_world: np.ndarray,
        deadline: float,
        monotonic: Callable[[], float],
    ) -> tuple[
        tuple[_PlannedPathLeg, ...],
        dict[str, object] | None,
        list[dict[str, object]],
    ]:
        """Plan optional envelope ingress, global pregrasp, and local approach."""

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
            )
            return (pregrasp_leg, approach_leg), None, []

        ingress_route_attempts: list[dict[str, object]] = []
        route_error: RM65GraspTrajectoryPlanningError | None = None
        for inward_fraction, ingress_target in ingress_targets:
            route_attempt: dict[str, object] = {"inward_fraction": inward_fraction}
            ingress_route_attempts.append(route_attempt)
            try:
                ingress_leg = self._plan_path_leg(
                    qpos_by_side=qpos_by_side,
                    side=side,
                    target_tool_tcp_world=ingress_target,
                    planning_phase="envelope_ingress",
                    deadline=deadline,
                    monotonic=monotonic,
                )
                pregrasp_leg, approach_leg = self._plan_pregrasp_and_approach(
                    qpos_by_side=ingress_leg.endpoint.qpos_by_side,
                    side=side,
                    pregrasp_tool_tcp_world=pregrasp_tool_tcp_world,
                    grasp_tool_tcp_world=grasp_tool_tcp_world,
                    deadline=deadline,
                    monotonic=monotonic,
                )
                combined_waypoints = (
                    *ingress_leg.path.waypoints,
                    *pregrasp_leg.path.waypoints[1:],
                    *approach_leg.path.waypoints[1:],
                )
                self._revalidate(
                    combined_waypoints,
                    moving_side=side,
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
                    "approach_endpoint_collision_check_unavailable",
                }:
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
        deadline = float(monotonic()) + self.config.path_planning_time_budget_sec
        ingress_route_attempts: list[dict[str, object]] = []
        pregrasp_route_attempts: list[dict[str, object]] = []
        selected_legs: tuple[_PlannedPathLeg, ...] | None = None
        ingress_route: dict[str, object] | None = None
        route_error: RM65GraspTrajectoryPlanningError | None = None
        if already_at_grasp:
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
            for distance_m in self._pregrasp_distance_candidates():
                candidate_pregrasp = self._pregrasp_target_pose(
                    target_pose,
                    distance_m=distance_m,
                )
                route_attempt: dict[str, object] = {"distance_m": distance_m}
                pregrasp_route_attempts.append(route_attempt)
                try:
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
                        }
                    )
                    if error.reason_code in {
                        "path_collision_check_unavailable",
                        "path_planning_time_budget_exhausted",
                        "approach_endpoint_collision_check_unavailable",
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
                raise RM65GraspTrajectoryPlanningError(
                    route_error.reason_code,
                    details={
                        **route_error.details,
                        "pregrasp_route_attempts": pregrasp_route_attempts,
                    },
                ) from route_error

        assert selected_legs is not None
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
                "pregrasp_route_attempts": pregrasp_route_attempts,
                "endpoint_seed_index": endpoint.seed_index,
                "endpoint_solve_cycles": endpoint.solve_cycles,
                "endpoint_position_residual_m": endpoint.position_residual_m,
                "endpoint_orientation_residual_deg": (
                    endpoint.orientation_residual_deg
                ),
                "path_search_attempts": search_attempts,
                "pregrasp_tool_tcp_world": pregrasp_pose,
                "pregrasp_applied": not already_at_grasp,
                "selected_pregrasp_distance_m": (
                    0.0
                    if already_at_grasp
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
    "RM65_GRASP_PATH_PLANNING_TIME_BUDGET_SEC",
    "RM65GraspTrajectoryPlanningConfig",
    "RM65GraspTrajectoryPlanningError",
]
