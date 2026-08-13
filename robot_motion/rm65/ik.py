from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import ARM_ENDPOINT_FRAME, BARE_ENDPOINT_FRAME, DEFAULT_COLLISION_SCENE, TOOL_TCP_FRAME
from .scene_profiles import build_rm65_dual_arm_scene_profile


HAND_BASE_FRAME = "hand_base"
ARM_LINK6_FRAME = "arm_link6"
DEFAULT_SINGLE_ARM_DRY_RUN_MAX_ITERATIONS = 48
DEFAULT_SINGLE_ARM_DRY_RUN_MAX_JOINT_STEP_RAD = (0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
DEFAULT_SINGLE_ARM_DRY_RUN_DAMPING = 0.03
DEFAULT_DUAL_ARM_SOLVER_FREQUENCY_HZ = 30.0
DEFAULT_DUAL_ARM_POSTURE_COST = 1e-1
DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M = 0.03
DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M = 1e-4
DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN = 10.0
DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN = 0.001


@dataclass(frozen=True)
class RM65SingleArmIKConfig:
    """Shared RM65 single-arm IK construction settings.

    The solve-time defaults mirror ``RM65ArmBackendConfig`` so sim and real can
    share one set of RM65 IK assumptions while keeping hardware dispatch outside
    ``robot_motion``.
    """

    arm_urdf_path: str | Path | None = None
    controlled_frame: str = TOOL_TCP_FRAME
    ik_tuning: Any | None = field(default=None, repr=False)
    dry_run_max_iterations: int = DEFAULT_SINGLE_ARM_DRY_RUN_MAX_ITERATIONS
    dry_run_max_joint_step_rad: tuple[float, ...] = DEFAULT_SINGLE_ARM_DRY_RUN_MAX_JOINT_STEP_RAD
    dry_run_damping: float = DEFAULT_SINGLE_ARM_DRY_RUN_DAMPING

    def __post_init__(self) -> None:
        if int(self.dry_run_max_iterations) <= 0:
            raise ValueError("dry_run_max_iterations must be positive.")
        joint_step = tuple(float(value) for value in self.dry_run_max_joint_step_rad)
        if len(joint_step) != 6 or any(value <= 0.0 for value in joint_step):
            raise ValueError("dry_run_max_joint_step_rad must contain 6 positive values.")
        if float(self.dry_run_damping) <= 0.0:
            raise ValueError("dry_run_damping must be positive.")
        object.__setattr__(self, "controlled_frame", canonical_rm65_controlled_frame(self.controlled_frame))
        object.__setattr__(self, "dry_run_max_iterations", int(self.dry_run_max_iterations))
        object.__setattr__(self, "dry_run_max_joint_step_rad", joint_step)
        object.__setattr__(self, "dry_run_damping", float(self.dry_run_damping))


@dataclass(frozen=True)
class RM65DualArmCollisionIKConfig:
    """Shared RM65 dual-arm collision-aware IK construction settings."""

    scene_profile: str = DEFAULT_COLLISION_SCENE
    solver: str | None = None
    solver_frequency: float = DEFAULT_DUAL_ARM_SOLVER_FREQUENCY_HZ
    damping: float = 0.01
    posture_cost: float = DEFAULT_DUAL_ARM_POSTURE_COST
    collision_min_distance_m: float = DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M
    collision_distance_epsilon_m: float = DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M
    collision_barrier_gain: float = DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN
    collision_safe_displacement_gain: float = DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN
    collision_constraints_enabled: bool = True
    debug_viz: bool = False
    debug_profile: bool = False
    ik_tuning: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.collision_constraints_enabled, bool):
            raise TypeError("collision_constraints_enabled must be a bool.")
        if float(self.solver_frequency) <= 0.0:
            raise ValueError("solver_frequency must be positive.")
        if not np.isfinite(float(self.damping)) or float(self.damping) <= 0.0:
            raise ValueError("damping must be finite and positive.")
        if float(self.posture_cost) < 0.0:
            raise ValueError("posture_cost must be >= 0.")
        if float(self.collision_min_distance_m) < 0.0:
            raise ValueError("collision_min_distance_m must be >= 0.")
        if float(self.collision_distance_epsilon_m) < 0.0:
            raise ValueError("collision_distance_epsilon_m must be >= 0.")
        object.__setattr__(self, "scene_profile", str(self.scene_profile or DEFAULT_COLLISION_SCENE))
        object.__setattr__(self, "solver_frequency", float(self.solver_frequency))
        object.__setattr__(self, "damping", float(self.damping))
        object.__setattr__(self, "posture_cost", float(self.posture_cost))
        object.__setattr__(self, "collision_min_distance_m", float(self.collision_min_distance_m))
        object.__setattr__(self, "collision_distance_epsilon_m", float(self.collision_distance_epsilon_m))
        object.__setattr__(self, "collision_barrier_gain", float(self.collision_barrier_gain))
        object.__setattr__(
            self,
            "collision_safe_displacement_gain",
            float(self.collision_safe_displacement_gain),
        )


@dataclass(frozen=True)
class RM65DualArmCollisionIKBundle:
    solver: Any
    scene_profile: Any
    config: RM65DualArmCollisionIKConfig

    @property
    def left_frame_name(self) -> str:
        return str(self.scene_profile.left_frame_name)

    @property
    def right_frame_name(self) -> str:
        return str(self.scene_profile.right_frame_name)

    @property
    def collision_pair_count(self) -> int:
        return len(self.scene_profile.collision_pairs)


def canonical_rm65_controlled_frame(frame: str | None = None) -> str:
    text = "" if frame is None else str(frame).strip()
    if text in {"", "tcp", TOOL_TCP_FRAME}:
        return TOOL_TCP_FRAME
    if text in {BARE_ENDPOINT_FRAME, ARM_ENDPOINT_FRAME, ARM_LINK6_FRAME, "empty"}:
        return ARM_ENDPOINT_FRAME
    if text == HAND_BASE_FRAME:
        return HAND_BASE_FRAME
    return text


def rm65_urdf_frame_for_controlled_frame(frame: str | None = None) -> str:
    controlled = canonical_rm65_controlled_frame(frame)
    if controlled == ARM_ENDPOINT_FRAME:
        return ARM_LINK6_FRAME
    if controlled == TOOL_TCP_FRAME:
        return HAND_BASE_FRAME
    return controlled


def rm65_single_arm_candidate_frames(frame: str | None = None) -> tuple[str, ...]:
    controlled = canonical_rm65_controlled_frame(frame)
    if controlled == ARM_ENDPOINT_FRAME:
        return (ARM_LINK6_FRAME, HAND_BASE_FRAME)
    if controlled in {TOOL_TCP_FRAME, HAND_BASE_FRAME}:
        return (HAND_BASE_FRAME, ARM_LINK6_FRAME)
    return (controlled, HAND_BASE_FRAME, ARM_LINK6_FRAME)


def build_rm65_single_arm_ik_solver(config: RM65SingleArmIKConfig):
    """Build a SingleArmIKSolver from shared RM65 defaults.

    The optional Pinocchio implementation is imported lazily so this module
    remains importable in lightweight test environments.
    """

    from .single_arm_ik import SingleArmIKSolver

    last_error: Exception | None = None
    for frame_name in rm65_single_arm_candidate_frames(config.controlled_frame):
        try:
            return SingleArmIKSolver(
                urdf_path=config.arm_urdf_path,
                eef_name=frame_name,
                ik_tuning=config.ik_tuning,
            )
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Could not resolve RM65 IK frame {config.controlled_frame!r}; "
        f"tried={rm65_single_arm_candidate_frames(config.controlled_frame)}."
    ) from last_error


def solve_rm65_single_arm_ik(
    solver: Any,
    *,
    target_pose: Sequence[Sequence[float]] | np.ndarray,
    current_qpos: Sequence[float] | np.ndarray,
    fast_iterative: bool = False,
    max_joint_step_rad: Sequence[float] | np.ndarray | None = None,
    dry_run_max_iterations: int = DEFAULT_SINGLE_ARM_DRY_RUN_MAX_ITERATIONS,
    dry_run_max_joint_step_rad: Sequence[float] | np.ndarray = DEFAULT_SINGLE_ARM_DRY_RUN_MAX_JOINT_STEP_RAD,
    dry_run_damping: float = DEFAULT_SINGLE_ARM_DRY_RUN_DAMPING,
    iterative_tolerance: float = 1e-5,
) -> np.ndarray:
    """Solve one RM65 single-arm IK target with shared sim/real defaults.

    Hardware dispatch, TCP frame conversion, and jump gates intentionally stay
    outside this helper; those are runtime concerns. This function only decides
    between the iterative dry-run solver used by virtual preview and the
    bounded single-step solver used for normal command generation.
    """

    target = np.asarray(target_pose, dtype=float).reshape(4, 4)
    qpos = np.asarray(current_qpos, dtype=float).reshape(6)
    if fast_iterative:
        return np.asarray(
            solver.solve_iterative(
                target_pose=target,
                q_init=qpos,
                max_iterations=int(dry_run_max_iterations),
                max_joint_step=np.asarray(as_joint_step_tuple(dry_run_max_joint_step_rad), dtype=float),
                damping=float(dry_run_damping),
                tolerance=float(iterative_tolerance),
            ),
            dtype=float,
        ).reshape(6)

    if max_joint_step_rad is None:
        raise ValueError("max_joint_step_rad is required when fast_iterative=False.")
    return np.asarray(
        solver.solve_step(
            target_pose=target,
            current_qpos=qpos,
            max_joint_step=np.asarray(as_joint_step_tuple(max_joint_step_rad), dtype=float),
        ),
        dtype=float,
    ).reshape(6)


def build_rm65_dual_arm_collision_ik_solver(
    config: RM65DualArmCollisionIKConfig | None = None,
) -> RM65DualArmCollisionIKBundle:
    """Build and task-configure the shared RM65 dual-arm collision IK solver."""

    resolved_config = config if config is not None else RM65DualArmCollisionIKConfig()
    scene_profile = build_rm65_dual_arm_scene_profile(resolved_config.scene_profile)
    if not resolved_config.collision_constraints_enabled:
        # Preserve the exact kinematic model and frame layout while omitting
        # only the optional collision pairs.  MuJoCo contact physics remains
        # independent from this pose-to-qpos planning constraint.
        scene_profile = replace(scene_profile, collision_pairs=())
    try:
        from .dual_arm_ik import MultiTaskIKSolver
        from .ik_tuning import load_ik_tuning_config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RM65 dual-arm collision IK requires pinocchio, pink, and qpsolvers."
        ) from exc

    ik_tuning = (
        resolved_config.ik_tuning
        if resolved_config.ik_tuning is not None
        else load_ik_tuning_config()
    )
    solver = MultiTaskIKSolver(
        robot_urdf_path=scene_profile.urdf_path,
        robot_package_dirs=list(scene_profile.package_dirs),
        robot_root_joint=None,
        robot_list_of_joints_to_lock=None,
        robot_collision_pair_names=list(scene_profile.collision_pairs),
        solver=resolved_config.solver,
        frequency=float(resolved_config.solver_frequency),
        damping=float(resolved_config.damping),
        debug_viz=bool(resolved_config.debug_viz),
        debug_profile=bool(resolved_config.debug_profile),
        collision_barrier_gain=float(resolved_config.collision_barrier_gain),
        collision_safe_displacement_gain=float(resolved_config.collision_safe_displacement_gain),
        collision_min_distance_m=float(resolved_config.collision_min_distance_m),
    )
    objective = ik_tuning.objective_weights()
    solver.add_frame_task(
        scene_profile.left_frame_name,
        position_cost=float(objective.dual_arm_position_cost),
        orientation_cost=float(objective.dual_arm_orientation_cost),
    )
    solver.add_frame_task(
        scene_profile.right_frame_name,
        position_cost=float(objective.dual_arm_position_cost),
        orientation_cost=float(objective.dual_arm_orientation_cost),
    )
    solver.add_posture_task(cost=float(resolved_config.posture_cost))
    return RM65DualArmCollisionIKBundle(
        solver=solver,
        scene_profile=scene_profile,
        config=resolved_config,
    )


def rm65_dual_arm_default_initial_qpos_by_side() -> dict[str, np.ndarray]:
    from .defaults import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS

    return {
        "left": np.asarray(DEFAULT_LEFT_ARM_QPOS, dtype=float).reshape(6).copy(),
        "right": np.asarray(DEFAULT_RIGHT_ARM_QPOS, dtype=float).reshape(6).copy(),
    }


def as_joint_step_tuple(values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("joint step must contain 6 positive finite values.")
    return tuple(float(value) for value in array)


__all__ = [
    "ARM_LINK6_FRAME",
    "DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN",
    "DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M",
    "DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M",
    "DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN",
    "DEFAULT_DUAL_ARM_POSTURE_COST",
    "DEFAULT_DUAL_ARM_SOLVER_FREQUENCY_HZ",
    "DEFAULT_SINGLE_ARM_DRY_RUN_DAMPING",
    "DEFAULT_SINGLE_ARM_DRY_RUN_MAX_ITERATIONS",
    "DEFAULT_SINGLE_ARM_DRY_RUN_MAX_JOINT_STEP_RAD",
    "HAND_BASE_FRAME",
    "RM65DualArmCollisionIKBundle",
    "RM65DualArmCollisionIKConfig",
    "RM65SingleArmIKConfig",
    "as_joint_step_tuple",
    "build_rm65_dual_arm_collision_ik_solver",
    "build_rm65_single_arm_ik_solver",
    "canonical_rm65_controlled_frame",
    "rm65_dual_arm_default_initial_qpos_by_side",
    "rm65_single_arm_candidate_frames",
    "rm65_urdf_frame_for_controlled_frame",
    "solve_rm65_single_arm_ik",
]
