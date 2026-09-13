"""URDF/FCL-only planning for one real RM65 grasp-approach Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import pairwise

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

    def __init__(self, reason_code: str, *, details: Mapping[str, object] | None = None):
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
    endpoint_position_tolerance_m: float = 0.003
    endpoint_orientation_tolerance_deg: float = 1.0
    final_edge_resolution_rad: float = float(np.deg2rad(0.1))

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
            "final_edge_resolution_rad",
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


@dataclass(frozen=True)
class RM65GraspReachabilityConfig:
    """Finite endpoint-IK budget used only for optional candidate annotation."""

    time_budget_ms: float = 1_000.0
    max_solve_cycles: int = 180
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


class RM65GraspJointTrajectoryPlanner:
    """Resolve endpoint IK, plan around fixed table/poles, then time-sample qpos."""

    def __init__(
        self,
        *,
        profile: ResolvedExclusiveRM65SkillDeploymentProfile,
        planning_target_builder: object,
        target_envelope_filter: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray] | None]
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
            raise ValueError("strict RM65 URDF must expose table and three camera poles.")
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
        try:
            projected = self.target_envelope_filter(
                {name: np.asarray(value, dtype=float).copy() for name, value in solver_targets.items()}
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
                },
            )

    def _endpoint_qpos_candidates(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
    ) -> tuple[_EndpointIKCandidate, ...]:
        with self._solver_lock:
            return self._endpoint_qpos_candidates_locked(
                qpos_by_side=qpos_by_side,
                side=side,
                target_tool_tcp_world=target_tool_tcp_world,
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
            name: np.asarray(qpos_by_side[name], dtype=float).copy()
            for name in _SIDES
        }
        maximum = (
            self.config.endpoint_seed_count
            if seed_count is None
            else int(seed_count)
        )
        if not 1 <= maximum <= 64:
            raise ValueError("seed_count must be in [1, 64].")
        seeds: list[dict[str, np.ndarray]] = []
        local_values = [
            np.clip(
                baseline[side]
                + pattern * self.config.endpoint_seed_perturbation_rad,
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
            outcome = "ik_search_exhausted"
            evaluations = 0
            try:
                candidate, evaluations, position, orientation = (
                    self._least_squares_endpoint_from_seed(
                        qpos_by_side=qpos_by_side,
                        side=side,
                        target_tool_tcp_world=target_tool_tcp_world,
                        seed_qpos=seed[side],
                        maximum_evaluations=(
                            self.config.maximum_endpoint_solve_cycles
                        ),
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
                "best_position_residual_m": best_attempt[
                    "best_position_residual_m"
                ],
                "best_orientation_residual_deg": best_attempt[
                    "best_orientation_residual_deg"
                ],
                "endpoint_seed_count": len(attempts),
                "maximum_cycles_per_seed": (
                    self.config.maximum_endpoint_solve_cycles
                ),
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
                try:
                    _candidate, evaluations, position, orientation = (
                        self._least_squares_endpoint_from_seed(
                            qpos_by_side=qpos_by_side,
                            side=side,
                            target_tool_tcp_world=target_pose,
                            seed_qpos=seed[side],
                            maximum_evaluations=config.max_solve_cycles,
                        )
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                cycles += evaluations
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

    def _revalidate(self, waypoints: tuple[Mapping[str, np.ndarray], ...]) -> dict[str, object]:
        checks = 0
        minimum_distance = math.inf
        for segment_index, (start, end) in enumerate(pairwise(waypoints)):
            maximum_delta = max(
                float(np.max(np.abs(np.asarray(end[side]) - np.asarray(start[side]))))
                for side in _SIDES
            )
            steps = max(1, math.ceil(maximum_delta / self.config.final_edge_resolution_rad))
            for index in range(steps + 1):
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
                self._require_unmodified_target_envelope(
                    solver_targets,
                    sides=_SIDES,
                    phase="path_target",
                    details={
                        "segment_index": segment_index,
                        "edge_sample_index": index,
                    },
                )
                check = self.oracle.check(qpos)
                checks += 1
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
        return {
            "sample_count": checks,
            "minimum_distance_m": (
                None if not math.isfinite(minimum_distance) else minimum_distance
            ),
        }

    def __call__(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        side: str,
        target_tool_tcp_world: np.ndarray,
        speed_scale: float,
    ) -> RealAgentJointTrajectory:
        selected_side = str(side).strip().lower()
        if selected_side not in _SIDES:
            raise ValueError("side must be left or right.")
        current = {
            name: np.asarray(dict(qpos_by_side).get(name), dtype=float).reshape(6).copy()
            for name in _SIDES
        }
        target_pose = np.asarray(target_tool_tcp_world, dtype=float).reshape(4, 4)
        scale = float(speed_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale must satisfy 0 < speed_scale <= 1.")
        endpoint_candidates = self._endpoint_qpos_candidates(
            qpos_by_side=current,
            side=selected_side,
            target_tool_tcp_world=target_pose,
        )
        lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
        upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
        cfg = self.planner_config
        search_attempts: list[dict[str, object]] = []
        selected: (
            tuple[_EndpointIKCandidate, JointPathPlanResult, dict[str, object]]
            | None
        ) = None
        terminal_error: RM65GraspTrajectoryPlanningError | None = None
        for endpoint_index, endpoint in enumerate(endpoint_candidates):
            target = endpoint.qpos_by_side
            for path_attempt in range(
                self.config.path_search_attempts_per_endpoint
            ):
                full_joint_range = (
                    path_attempt
                    == self.config.path_search_attempts_per_endpoint - 1
                )
                padding = cfg.sample_padding_rad * float(2**path_attempt)
                corridor_lower = (
                    lower
                    if full_joint_range
                    else np.maximum(
                        lower,
                        np.minimum(current[selected_side], target[selected_side])
                        - padding,
                    )
                )
                corridor_upper = (
                    upper
                    if full_joint_range
                    else np.minimum(
                        upper,
                        np.maximum(current[selected_side], target[selected_side])
                        + padding,
                    )
                )
                attempt_config = replace(
                    cfg,
                    seed=(
                        cfg.seed
                        + endpoint_index
                        * self.config.path_search_attempts_per_endpoint
                        + path_attempt
                    ),
                )
                path = plan_collision_aware_joint_path(
                    start_qpos_by_component=current,
                    target_qpos_by_component=target,
                    movable_components=(selected_side,),
                    oracle=self.oracle,
                    config=attempt_config,
                    joint_lower_by_component={selected_side: corridor_lower},
                    joint_upper_by_component={selected_side: corridor_upper},
                    scene_hash=self.scene.semantic_content_hash,
                )
                attempt_record: dict[str, object] = {
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
                    if path.status is JointPathStatus.COLLISION_CHECK_UNAVAILABLE:
                        terminal_error = RM65GraspTrajectoryPlanningError(
                            "path_collision_check_unavailable",
                            details={
                                "path": path.to_dict(),
                                "search_attempts": search_attempts,
                            },
                        )
                        break
                    continue
                try:
                    dense = self._revalidate(path.waypoints)
                except RM65GraspTrajectoryPlanningError as exc:
                    attempt_record["dense_revalidation_error"] = exc.reason_code
                    terminal_error = exc
                    if exc.reason_code == "path_collision_check_unavailable":
                        break
                    continue
                selected = (endpoint, path, dense)
                break
            if selected is not None or (
                terminal_error is not None
                and terminal_error.reason_code == "path_collision_check_unavailable"
            ):
                break
        if selected is None:
            if terminal_error is not None and terminal_error.reason_code in {
                "path_collision_check_unavailable",
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
        endpoint, path, dense = selected
        target = endpoint.qpos_by_side
        velocity_profile = load_joint_velocity_limit_profiles()[
            self.profile.position_profile.ik_execution.joint_velocity_limit_profile
        ]
        joint_velocity = np.deg2rad(
            np.asarray(velocity_profile.max_velocity_deg_s, dtype=float)
        ) * scale
        reference_hz = float(self.profile.position_profile.rates.ik_hz)
        times = [0.0]
        samples = {name: [current[name].copy()] for name in _SIDES}
        elapsed = 0.0
        for start, end in pairwise(path.waypoints):
            start_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=start["left"], right_qpos=start["right"]
            )[selected_side]
            end_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=end["left"], right_qpos=end["right"]
            )[selected_side]
            translation, rotation_deg = pose_delta_metrics(start_tcp, end_tcp)
            delta = np.abs(np.asarray(end[selected_side]) - np.asarray(start[selected_side]))
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
                "path_hash": path.semantic_content_hash,
                "dense_revalidation": dense,
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
