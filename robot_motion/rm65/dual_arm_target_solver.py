from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.frames import (
    ARM_ENDPOINT_FRAME,
    TOOL_TCP_FRAME,
    TOOL_TCP_SEMANTICS,
    canonical_arm_controlled_frame,
)
from robot_motion.rm65.ik_tuning import (
    RM65IKTuningConfig,
    hard_branch_jump_exceeds,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
    preferred_wrist_flip_candidate,
    weighted_joint_distance_score,
)
from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
)
from robot_motion.joint_limits import (
    JointVelocityLimitProfile,
    load_joint_velocity_limit_profiles,
)
from robot_motion.rm65 import (
    DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN,
    DEFAULT_DUAL_ARM_POSTURE_COST,
    DEFAULT_DUAL_ARM_SOLVER_FREQUENCY_HZ,
    RM65DualArmCollisionIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
    build_rm65_dual_arm_scene_profile,
)
from robot_motion.posture_preferences import JointPosturePreference, posture_preference_from_metadata
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles
from robot_motion.runtime import EffectorCommand, Retargeter, TeleopCommand, TeleopObservation
from robot_motion.tcp import HAND_BASE_TCP_FRAME, canonical_tcp_frame
from robot_motion.transforms import as_pose_matrix, interpolate_pose, invert_pose

from .tool_safety import tcp_safety_warnings


_HAND_BASE_FROM_ARM_LINK6 = np.eye(4, dtype=float)
_HAND_BASE_FROM_ARM_LINK6[2, 3] = 0.009
_ARM_LINK6_FROM_HAND_BASE = np.linalg.inv(_HAND_BASE_FROM_ARM_LINK6)
_CURRENT_TARGET_HOLD_TRANSLATION_M = 1e-3
_CURRENT_TARGET_HOLD_ROTATION_DEG = 0.5
IK_FRAMEWORK_ROBOT_MOTION = "robot_motion"
IK_FRAMEWORK_CHOICES = (IK_FRAMEWORK_ROBOT_MOTION,)


def _normalize_ik_framework(value: object) -> str:
    text = str(value or IK_FRAMEWORK_ROBOT_MOTION).strip().lower().replace("-", "_")
    aliases = {
        "default": IK_FRAMEWORK_ROBOT_MOTION,
        "shared": IK_FRAMEWORK_ROBOT_MOTION,
        "robotmotion": IK_FRAMEWORK_ROBOT_MOTION,
        "robot_motion": IK_FRAMEWORK_ROBOT_MOTION,
    }
    resolved = aliases.get(text, text)
    if resolved not in IK_FRAMEWORK_CHOICES:
        raise ValueError(
            f"Unsupported dual-arm IK framework {value!r}. Expected one of {IK_FRAMEWORK_CHOICES}."
        )
    return resolved


@dataclass(frozen=True)
class _DualArmCandidate:
    name: str
    qpos: np.ndarray
    source: str
    method: str
    raw_score_deg: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DualArmQposCollisionCheck:
    """Read-only endpoint check for a concrete pair of arm joint targets."""

    safe: bool
    reason: str = ""
    min_distance_m: float | None = None
    collision_pair_count: int = 0
    required_min_distance_m: float = 0.0


@dataclass(frozen=True)
class _DualArmEvaluatedCandidate:
    candidate: _DualArmCandidate
    output_qpos: np.ndarray
    score: float
    metadata: dict[str, object]


def _optional_qpos(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    qpos = np.asarray(value, dtype=float).reshape(-1)
    if qpos.shape != (6,):
        raise ValueError(f"Expected a 6D joint vector, got shape {qpos.shape}.")
    return qpos.copy()


def _optional_joint_step(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    joint_step = np.asarray(value, dtype=float).reshape(-1)
    if joint_step.shape != (6,):
        raise ValueError(f"Expected a 6D joint-step vector, got shape {joint_step.shape}.")
    if np.any(joint_step <= 0.0):
        raise ValueError("Execution joint-step limits must be strictly positive.")
    return joint_step.copy()


def _optional_pose6(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    pose = np.asarray(value, dtype=float).reshape(-1)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Expected a finite 6D pose, got shape {pose.shape}.")
    return pose.copy()


def _linear_release_weight(
    *,
    value: float,
    static_threshold: float,
    full_threshold: float,
) -> float:
    """Map an error to [0, 1] between validated static/full thresholds."""
    if not np.isfinite(value):
        return 0.0
    return float(
        np.clip(
            (float(value) - float(static_threshold))
            / (float(full_threshold) - float(static_threshold)),
            0.0,
            1.0,
        )
    )


def _pose6_to_matrix(pose6: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if pose6 is None:
        return np.eye(4, dtype=float)
    values = np.asarray(pose6, dtype=float).reshape(6)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = R.from_euler("xyz", values[3:6], degrees=False).as_matrix()
    return matrix


def _warn_tcp_safety(side: str, pose6: Sequence[float] | np.ndarray | None) -> None:
    if pose6 is None:
        return
    for warning in tcp_safety_warnings(pose6, label=f"{side} collision TCP"):
        print(f"WARNING: {warning}; verify tool profile before enabling dual-arm collision retargeting.")


def _load_safety_profile(path: str | None, profile_name: str) -> IKSafetyProfile:
    profiles = load_ik_safety_profiles(path)
    name = str(profile_name).strip()
    if name not in profiles:
        raise ValueError(
            f"Unknown IK safety profile {name!r}. Available profiles: {sorted(profiles)}"
        )
    return profiles[name]


def _load_joint_velocity_profile(
    path: str | None,
    profile_name: str,
) -> JointVelocityLimitProfile | None:
    name = str(profile_name).strip()
    if not name:
        return None
    profiles = load_joint_velocity_limit_profiles(path)
    if name not in profiles:
        raise ValueError(
            f"Unknown joint velocity limit profile {name!r}. Available profiles: {sorted(profiles)}"
        )
    profile = profiles[name]
    if profile.dof != 6:
        raise ValueError(
            f"Dual-arm retargeting expects a per-arm 6-DOF velocity profile, got {profile.dof}."
        )
    return profile


def _dual_arm_step_scale_factors(profile: IKSafetyProfile) -> tuple[float, ...]:
    raw_scales = tuple(float(value) for value in getattr(profile, "local_step_scale_factors", ()) or ())
    if not raw_scales:
        raw_scales = (1.0,)
    scales: list[float] = []
    for value in raw_scales:
        if not np.isfinite(value) or value <= 0.0:
            continue
        scale = max(1e-6, min(1.0, float(value)))
        if any(abs(scale - existing) < 1e-9 for existing in scales):
            continue
        scales.append(scale)
    if not scales:
        scales = [1.0]
    scales.sort(reverse=True)
    return tuple(scales)


@dataclass(frozen=True)
class DualArmTargetContext:
    timestamp: float | None
    left_target: np.ndarray
    right_target: np.ndarray
    left_world_to_base: np.ndarray | None
    right_world_to_base: np.ndarray | None
    left_cartesian: EffectorCommand | None
    right_cartesian: EffectorCommand | None
    left_input_skipped: str | None
    right_input_skipped: str | None
    target_smoothing_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BimanualPoseTargetSource:
    """Pure-data description of one upstream Cartesian source command.

    This deliberately mirrors only the data needed by the coupled planner.  It
    does not depend on a teleoperation command container, tracker source, or
    calibration object.
    """

    command_type: str
    value: np.ndarray
    frame_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    _data_is_owned: InitVar[bool] = False

    def __post_init__(self, _data_is_owned: bool) -> None:
        value = np.asarray(self.value, dtype=float)
        if not _data_is_owned:
            value = value.copy()
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("BimanualPoseTargetSource.value must be finite and non-empty.")
        object.__setattr__(self, "command_type", str(self.command_type))
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "frame_name",
            None if self.frame_name is None else str(self.frame_name),
        )
        object.__setattr__(
            self,
            "metadata",
            (
                dict(self.metadata)
                if _data_is_owned
                else copy.deepcopy(dict(self.metadata))
            ),
        )


@dataclass(frozen=True)
class BimanualPoseTarget:
    """Solver-frame bimanual pose target independent of teleop runtime types."""

    timestamp: float | None
    target_pose_by_side: Mapping[str, np.ndarray]
    active_sides: tuple[str, ...] = ("left", "right")
    world_to_base_by_side: Mapping[str, np.ndarray | None] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    source_by_side: Mapping[str, BimanualPoseTargetSource | None] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    input_skipped_by_side: Mapping[str, str | None] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    target_smoothing_metadata: dict[str, object] = field(default_factory=dict)
    _data_is_owned: InitVar[bool] = False

    def __post_init__(self, _data_is_owned: bool) -> None:
        expected = {"left", "right"}
        target_keys = set(self.target_pose_by_side)
        if target_keys != expected:
            raise ValueError(
                "target_pose_by_side must contain exactly left and right; "
                f"got {sorted(str(key) for key in target_keys)}."
            )
        targets = {}
        for side in ("left", "right"):
            pose = as_pose_matrix(self.target_pose_by_side[side])
            targets[side] = pose if _data_is_owned else pose.copy()
        active_sides = tuple(str(side).strip().lower() for side in self.active_sides)
        if len(set(active_sides)) != len(active_sides) or not set(active_sides) <= expected:
            raise ValueError("active_sides must be a unique subset of ('left', 'right').")
        world_to_base: dict[str, np.ndarray | None] = {}
        sources: dict[str, BimanualPoseTargetSource | None] = {}
        skipped: dict[str, str | None] = {}
        for side in ("left", "right"):
            transform = self.world_to_base_by_side.get(side)
            if transform is None:
                world_to_base[side] = None
            else:
                pose = as_pose_matrix(transform)
                world_to_base[side] = pose if _data_is_owned else pose.copy()
            source = self.source_by_side.get(side)
            if source is not None and not isinstance(source, BimanualPoseTargetSource):
                raise TypeError(
                    f"source_by_side[{side!r}] must be BimanualPoseTargetSource or None."
                )
            sources[side] = source
            reason = self.input_skipped_by_side.get(side)
            skipped[side] = None if reason is None else str(reason)
        object.__setattr__(self, "target_pose_by_side", targets)
        object.__setattr__(self, "active_sides", active_sides)
        object.__setattr__(self, "world_to_base_by_side", world_to_base)
        object.__setattr__(self, "source_by_side", sources)
        object.__setattr__(self, "input_skipped_by_side", skipped)
        object.__setattr__(
            self,
            "target_smoothing_metadata",
            (
                dict(self.target_smoothing_metadata)
                if _data_is_owned
                else copy.deepcopy(dict(self.target_smoothing_metadata))
            ),
        )


@dataclass(frozen=True)
class BimanualIKSideResult:
    """One side of an atomic coupled-IK result, expressed as pure data."""

    side: str
    qpos: np.ndarray
    raw_qpos: np.ndarray | None
    solver_pose_world: np.ndarray
    tool_tcp_pose_world: np.ndarray
    command_type: str
    command_value: np.ndarray
    command_frame_name: str | None
    metadata: dict[str, object] = field(default_factory=dict)
    _metadata_is_owned: InitVar[bool] = False

    def __post_init__(self, _metadata_is_owned: bool) -> None:
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported bimanual IK result side {self.side!r}.")
        qpos = np.asarray(self.qpos, dtype=float).reshape(-1).copy()
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError("BimanualIKSideResult.qpos must contain 6 finite values.")
        raw_qpos = None
        if self.raw_qpos is not None:
            raw_qpos = np.asarray(self.raw_qpos, dtype=float).reshape(-1).copy()
            if raw_qpos.shape != (6,) or not np.all(np.isfinite(raw_qpos)):
                raise ValueError(
                    "BimanualIKSideResult.raw_qpos must contain 6 finite values."
                )
        command_value = np.asarray(self.command_value, dtype=float).copy()
        if command_value.size == 0 or not np.all(np.isfinite(command_value)):
            raise ValueError(
                "BimanualIKSideResult.command_value must be finite and non-empty."
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "raw_qpos", raw_qpos)
        object.__setattr__(
            self, "solver_pose_world", as_pose_matrix(self.solver_pose_world).copy()
        )
        object.__setattr__(
            self, "tool_tcp_pose_world", as_pose_matrix(self.tool_tcp_pose_world).copy()
        )
        object.__setattr__(self, "command_type", str(self.command_type))
        object.__setattr__(self, "command_value", command_value)
        object.__setattr__(
            self,
            "command_frame_name",
            None if self.command_frame_name is None else str(self.command_frame_name),
        )
        object.__setattr__(
            self,
            "metadata",
            (
                dict(self.metadata)
                if _metadata_is_owned
                else copy.deepcopy(dict(self.metadata))
            ),
        )


@dataclass(frozen=True)
class BimanualIKResult:
    """Atomic coupled-IK result that can be consumed by real, sim, or replay."""

    timestamp: float | None
    status: str
    fallback_reason: str | None
    side_results: Mapping[str, BimanualIKSideResult]
    metadata: dict[str, object] = field(default_factory=dict)
    _metadata_is_owned: InitVar[bool] = False

    def __post_init__(self, _metadata_is_owned: bool) -> None:
        expected = {"left", "right"}
        if set(self.side_results) != expected:
            raise ValueError("side_results must contain exactly left and right.")
        results = {
            side: self.side_results[side]
            for side in ("left", "right")
        }
        for side, result in results.items():
            if not isinstance(result, BimanualIKSideResult) or result.side != side:
                raise TypeError(f"side_results[{side!r}] has an invalid side result.")
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self,
            "fallback_reason",
            None if self.fallback_reason is None else str(self.fallback_reason),
        )
        object.__setattr__(self, "side_results", results)
        object.__setattr__(
            self,
            "metadata",
            (
                dict(self.metadata)
                if _metadata_is_owned
                else copy.deepcopy(dict(self.metadata))
            ),
        )

    def qpos_by_side(self) -> dict[str, np.ndarray]:
        return {
            side: self.side_results[side].qpos.copy()
            for side in ("left", "right")
        }


@dataclass
class DualArmCollisionAwareTargetSolverConfig:
    left_input_channel_name: str = "left_wrist_pose"
    right_input_channel_name: str = "right_wrist_pose"
    left_effector_name: str = "left_arm"
    right_effector_name: str = "right_arm"
    # Absolute retargeting is the canonical default. The legacy delta-clutch
    # mode remains accepted only when callers explicitly request it.
    control_mode: str = "absolute"
    orientation_mode: str = "full"
    translation_scale: float = 1.0
    workspace_clip: bool = True
    scene_profile: str = "dual_arm_hand_proxy_open"
    contact_mode: Literal["boundary_slide", "freeze_on_contact"] = "boundary_slide"
    solver: str | None = None
    solver_frequency: float = DEFAULT_DUAL_ARM_SOLVER_FREQUENCY_HZ
    damping: float = 0.01
    posture_cost: float = DEFAULT_DUAL_ARM_POSTURE_COST
    collision_min_distance_m: float = DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M
    collision_distance_epsilon_m: float = DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M
    collision_barrier_gain: float = DEFAULT_DUAL_ARM_COLLISION_BARRIER_GAIN
    collision_safe_displacement_gain: float = DEFAULT_DUAL_ARM_COLLISION_SAFE_DISPLACEMENT_GAIN
    hold_last_safe_on_failure: bool = True
    ik_framework: str = IK_FRAMEWORK_ROBOT_MOTION
    debug_viz: bool = False
    debug_profile: bool = False
    debug_profile_print_every: int = 0
    solve_time_budget_ms: float | None = None
    output_command_type: Literal["joint_vector", "cartesian_pose"] = "joint_vector"
    # Keep the generic/force-facing facade on its historical command output by
    # default. Position/simulation composition roots may opt into native result
    # output; solve_target_pair() always uses the pure contract directly.
    bimanual_result_contract_enabled: bool = False
    solve_max_iterations: int = 1
    solve_min_iterations: int = 1
    # Position control may opt into running up to solve_max_iterations while
    # there is still time left in the frame budget.  The generic retargeter
    # default remains fixed-one-iteration so force/impedance callers retain
    # their historical behavior.
    solve_adaptive_iterations_enabled: bool = False
    solve_iteration_reserve_ms: float = 5.0
    solve_translation_tolerance_m: float = 2e-3
    solve_orientation_tolerance_deg: float = 0.5
    enforce_realistic_joint_step: bool = True
    left_execution_max_joint_step_rad: np.ndarray | Sequence[float] | None = None
    right_execution_max_joint_step_rad: np.ndarray | Sequence[float] | None = None
    left_tool_tcp_pose_flange_m_rad: np.ndarray | Sequence[float] | None = None
    right_tool_tcp_pose_flange_m_rad: np.ndarray | Sequence[float] | None = None
    left_initial_qpos: np.ndarray | Sequence[float] | None = None
    right_initial_qpos: np.ndarray | Sequence[float] | None = None
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)
    ik_safety_profiles_json: str | None = None
    ik_safety_profile_name: str = "dual_arm_teleop"
    ik_safety_profile: IKSafetyProfile | None = field(default=None, repr=False)
    joint_velocity_limits_json: str | None = None
    joint_velocity_limit_profile: str = "rm65"
    joint_velocity_profile: JointVelocityLimitProfile | None = field(default=None, repr=False)
    target_pose_smoother: Any | None = field(default=None, repr=False)
    output_qpos_smoothing_alpha: float = 1.0
    output_qpos_smoothing_adaptive_enabled: bool = False
    output_qpos_smoothing_fast_alpha: float = 0.7
    output_qpos_smoothing_linear_speed_low_m_s: float = 0.015
    output_qpos_smoothing_linear_speed_high_m_s: float = 0.25
    output_qpos_smoothing_angular_speed_low_rad_s: float = 0.08
    output_qpos_smoothing_angular_speed_high_rad_s: float = 1.5
    output_qpos_smoothing_attack_sec: float = 0.05
    output_qpos_smoothing_release_sec: float = 0.20
    # Zero preserves the historical hard-limit-only behavior. Runtime may set
    # a mode-specific interior envelope for force control.
    joint_operating_margin_deg: float = 0.0
    current_target_hold_enabled: bool = True
    local_step_candidate_selection_enabled: bool = True
    # Position teleop may prefer an evaluated branch whose RM65 wrist joint is
    # outside the SDK wrist-singularity threshold. This is a soft preference:
    # when every feasible branch is singular, the best singular branch remains
    # executable. The generic default is off so force/impedance behavior stays
    # unchanged.
    prefer_non_wrist_singular_branch: bool = False
    wrist_singularity_threshold_deg: float = 3.0
    # Position teleop may also prefer candidates outside the RM65 elbow
    # singularity neighborhood.  Candidate selection uses a small conservative
    # margin because the vendor SDK is the final authority and can classify the
    # boundary slightly differently from the local J3 proxy.  The generic
    # default remains off so force/impedance callers are unchanged.
    prefer_non_elbow_singular_branch: bool = False
    elbow_singularity_threshold_deg: float = 3.0
    elbow_singularity_selection_margin_deg: float = 1.0
    # Multiplies only the candidate-local clip used while scoring/selecting an
    # IK branch. The final qref execution velocity cap remains independent.
    local_step_candidate_clip_scale: float = 1.0
    inactive_side_hold_enabled: bool = True
    inactive_side_hold_translation_deadband_m: float = 0.003
    inactive_side_hold_rotation_deadband_deg: float = 0.5
    inactive_side_hold_max_unexplained_position_m: float = 0.025
    inactive_side_hold_max_unexplained_rotation_deg: float = 8.0
    inactive_side_hold_score_weight: float = 1_000_000.0
    inactive_side_hold_execution_cap_enabled: bool = True
    inactive_side_hold_execution_cap_deg: float = 0.5
    # This is opt-in at the position-control entrypoint.  Keeping the generic
    # retargeter default false guarantees force/impedance callers retain their
    # existing inactive-side behavior.
    inactive_side_hold_execution_release_enabled: bool = False
    # While a VIVE tracker is confirmed inactive, let the robot converge toward
    # its frozen Cartesian target instead of freezing qref unconditionally.
    # Translation and rotation releases are linearly normalized independently
    # and added with equal weight; see _tracker_inactive_execution_release().
    inactive_side_hold_execution_release_translation_static_m: float = 0.03
    inactive_side_hold_execution_release_translation_full_m: float = 0.10
    inactive_side_hold_execution_release_rotation_static_deg: float = 5.0
    inactive_side_hold_execution_release_rotation_full_deg: float = 30.0

    def __post_init__(self) -> None:
        if self.control_mode != "absolute":
            raise ValueError(
                "delta_clutch has been hard-archived; the dual-arm target solver only "
                "supports 'absolute'."
            )
        if self.contact_mode not in {"boundary_slide", "freeze_on_contact"}:
            raise ValueError(
                f"Unsupported contact_mode: {self.contact_mode}. "
                "Expected 'boundary_slide' or 'freeze_on_contact'."
            )
        self.ik_framework = _normalize_ik_framework(self.ik_framework)
        if self.solver_frequency <= 0.0:
            raise ValueError("solver_frequency must be positive.")
        if not np.isfinite(float(self.damping)) or float(self.damping) <= 0.0:
            raise ValueError("damping must be finite and positive.")
        self.damping = float(self.damping)
        if self.collision_min_distance_m < 0.0:
            raise ValueError("collision_min_distance_m must be >= 0.")
        if self.collision_distance_epsilon_m < 0.0:
            raise ValueError("collision_distance_epsilon_m must be >= 0.")
        if self.solve_max_iterations <= 0:
            raise ValueError("solve_max_iterations must be positive.")
        if self.solve_min_iterations <= 0:
            raise ValueError("solve_min_iterations must be positive.")
        if self.solve_min_iterations > self.solve_max_iterations:
            raise ValueError("solve_min_iterations must be <= solve_max_iterations.")
        self.solve_adaptive_iterations_enabled = bool(self.solve_adaptive_iterations_enabled)
        if not np.isfinite(float(self.solve_iteration_reserve_ms)) or self.solve_iteration_reserve_ms < 0.0:
            raise ValueError("solve_iteration_reserve_ms must be finite and >= 0.")
        self.solve_iteration_reserve_ms = float(self.solve_iteration_reserve_ms)
        if self.solve_translation_tolerance_m < 0.0:
            raise ValueError("solve_translation_tolerance_m must be >= 0.")
        if self.solve_orientation_tolerance_deg < 0.0:
            raise ValueError("solve_orientation_tolerance_deg must be >= 0.")
        if self.debug_profile_print_every < 0:
            raise ValueError("debug_profile_print_every must be >= 0.")
        if self.solve_time_budget_ms is None:
            self.solve_time_budget_ms = 1000.0 / float(self.solver_frequency)
        elif self.solve_time_budget_ms < 0.0:
            raise ValueError("solve_time_budget_ms must be >= 0.")
        if self.output_command_type not in {"joint_vector", "cartesian_pose"}:
            raise ValueError("output_command_type must be joint_vector or cartesian_pose.")
        self.bimanual_result_contract_enabled = bool(
            self.bimanual_result_contract_enabled
        )
        self.local_step_candidate_selection_enabled = bool(
            self.local_step_candidate_selection_enabled
        )
        self.prefer_non_wrist_singular_branch = bool(
            self.prefer_non_wrist_singular_branch
        )
        self.prefer_non_elbow_singular_branch = bool(
            self.prefer_non_elbow_singular_branch
        )
        if (
            not np.isfinite(float(self.wrist_singularity_threshold_deg))
            or float(self.wrist_singularity_threshold_deg) < 0.0
        ):
            raise ValueError("wrist_singularity_threshold_deg must be finite and >= 0.")
        self.wrist_singularity_threshold_deg = float(
            self.wrist_singularity_threshold_deg
        )
        for name in (
            "elbow_singularity_threshold_deg",
            "elbow_singularity_selection_margin_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
            setattr(self, name, value)
        if (
            not np.isfinite(float(self.local_step_candidate_clip_scale))
            or float(self.local_step_candidate_clip_scale) < 1.0
        ):
            raise ValueError("local_step_candidate_clip_scale must be finite and >= 1.")
        self.local_step_candidate_clip_scale = float(self.local_step_candidate_clip_scale)
        self.current_target_hold_enabled = bool(self.current_target_hold_enabled)
        self.inactive_side_hold_execution_release_enabled = bool(
            self.inactive_side_hold_execution_release_enabled
        )
        if self.target_pose_smoother is not None and not callable(
            getattr(self.target_pose_smoother, "smooth_pair", None)
        ):
            raise ValueError("target_pose_smoother must provide a smooth_pair method.")
        if not 0.0 <= float(self.output_qpos_smoothing_alpha) <= 1.0:
            raise ValueError("output_qpos_smoothing_alpha must be in [0, 1].")
        if not 0.0 <= float(self.output_qpos_smoothing_fast_alpha) <= 1.0:
            raise ValueError("output_qpos_smoothing_fast_alpha must be in [0, 1].")
        if bool(self.output_qpos_smoothing_adaptive_enabled) and float(
            self.output_qpos_smoothing_fast_alpha
        ) < float(self.output_qpos_smoothing_alpha):
            raise ValueError("output_qpos_smoothing_fast_alpha must be >= the slow alpha.")
        for low_name, high_name in (
            ("output_qpos_smoothing_linear_speed_low_m_s", "output_qpos_smoothing_linear_speed_high_m_s"),
            ("output_qpos_smoothing_angular_speed_low_rad_s", "output_qpos_smoothing_angular_speed_high_rad_s"),
        ):
            low = float(getattr(self, low_name))
            high = float(getattr(self, high_name))
            if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high <= low:
                raise ValueError(f"{high_name} must be finite and greater than non-negative {low_name}.")
        if float(self.output_qpos_smoothing_attack_sec) <= 0.0:
            raise ValueError("output_qpos_smoothing_attack_sec must be positive.")
        if float(self.output_qpos_smoothing_release_sec) <= 0.0:
            raise ValueError("output_qpos_smoothing_release_sec must be positive.")
        if not np.isfinite(float(self.joint_operating_margin_deg)) or float(
            self.joint_operating_margin_deg
        ) < 0.0:
            raise ValueError("joint_operating_margin_deg must be finite and >= 0.")
        if self.inactive_side_hold_translation_deadband_m < 0.0:
            raise ValueError("inactive_side_hold_translation_deadband_m must be >= 0.")
        if self.inactive_side_hold_rotation_deadband_deg < 0.0:
            raise ValueError("inactive_side_hold_rotation_deadband_deg must be >= 0.")
        if self.inactive_side_hold_max_unexplained_position_m < 0.0:
            raise ValueError("inactive_side_hold_max_unexplained_position_m must be >= 0.")
        if self.inactive_side_hold_max_unexplained_rotation_deg < 0.0:
            raise ValueError("inactive_side_hold_max_unexplained_rotation_deg must be >= 0.")
        if self.inactive_side_hold_score_weight < 0.0:
            raise ValueError("inactive_side_hold_score_weight must be >= 0.")
        if self.inactive_side_hold_execution_cap_deg < 0.0:
            raise ValueError("inactive_side_hold_execution_cap_deg must be >= 0.")
        for static_name, full_name in (
            (
                "inactive_side_hold_execution_release_translation_static_m",
                "inactive_side_hold_execution_release_translation_full_m",
            ),
            (
                "inactive_side_hold_execution_release_rotation_static_deg",
                "inactive_side_hold_execution_release_rotation_full_deg",
            ),
        ):
            static_value = float(getattr(self, static_name))
            full_value = float(getattr(self, full_name))
            if (
                not np.isfinite(static_value)
                or not np.isfinite(full_value)
                or static_value < 0.0
                or full_value <= static_value
            ):
                raise ValueError(
                    f"{full_name} must be finite and greater than non-negative {static_name}."
                )
        self.left_execution_max_joint_step_rad = _optional_joint_step(
            self.left_execution_max_joint_step_rad
        )
        self.right_execution_max_joint_step_rad = _optional_joint_step(
            self.right_execution_max_joint_step_rad
        )
        self.left_tool_tcp_pose_flange_m_rad = _optional_pose6(
            self.left_tool_tcp_pose_flange_m_rad
        )
        self.right_tool_tcp_pose_flange_m_rad = _optional_pose6(
            self.right_tool_tcp_pose_flange_m_rad
        )
        _warn_tcp_safety("left", self.left_tool_tcp_pose_flange_m_rad)
        _warn_tcp_safety("right", self.right_tool_tcp_pose_flange_m_rad)
        if self.ik_safety_profile is None:
            self.ik_safety_profile = _load_safety_profile(
                self.ik_safety_profiles_json,
                self.ik_safety_profile_name,
            )
        if self.joint_velocity_profile is None:
            self.joint_velocity_profile = _load_joint_velocity_profile(
                self.joint_velocity_limits_json,
                self.joint_velocity_limit_profile,
            )
        elif self.joint_velocity_profile.dof != 6:
            raise ValueError(
                f"Dual-arm retargeting expects a per-arm 6-DOF velocity profile, got {self.joint_velocity_profile.dof}."
            )
        if self.enforce_realistic_joint_step:
            if self.left_execution_max_joint_step_rad is None and self.joint_velocity_profile is None:
                raise ValueError(
                    "enforce_realistic_joint_step=True requires either left_execution_max_joint_step_rad "
                    "or a joint_velocity_profile."
                )
            if self.right_execution_max_joint_step_rad is None and self.joint_velocity_profile is None:
                raise ValueError(
                    "enforce_realistic_joint_step=True requires either right_execution_max_joint_step_rad "
                    "or a joint_velocity_profile."
                )
        self.left_initial_qpos = _optional_qpos(self.left_initial_qpos)
        self.right_initial_qpos = _optional_qpos(self.right_initial_qpos)
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)


class DualArmCollisionAwareTargetSolver(Retargeter):
    """Device-neutral coupled target-pose to safe-qref controller.

    Operator-specific pose adapters may be injected by ``teleop_runtime``.
    Simulation and replay callers should bypass them and call
    :meth:`solve_target_pair` with a :class:`BimanualPoseTarget`.
    """

    def __init__(
        self,
        config: DualArmCollisionAwareTargetSolverConfig,
        *,
        left_pose_retargeter: Any | None = None,
        right_pose_retargeter: Any | None = None,
        ik_solver: Any | None = None,
        ik_solver_builder: Callable[..., Any] | None = None,
        scene_profile_builder: Callable[..., Any] | None = None,
    ):
        self.config = config
        self.ik_tuning = config.ik_tuning
        assert self.ik_tuning is not None
        self.ik_safety_profile = config.ik_safety_profile
        assert self.ik_safety_profile is not None
        self.posture_preference: JointPosturePreference | None = posture_preference_from_metadata(
            self.ik_safety_profile.metadata
        )
        self.joint_velocity_profile = config.joint_velocity_profile
        self.profile = None
        self.left_pose_retargeter = left_pose_retargeter
        self.right_pose_retargeter = right_pose_retargeter
        selected_ik_builder = (
            build_rm65_dual_arm_collision_ik_solver
            if ik_solver_builder is None
            else ik_solver_builder
        )
        selected_scene_builder = (
            build_rm65_dual_arm_scene_profile
            if scene_profile_builder is None
            else scene_profile_builder
        )
        self.target_override_filter: Callable[
            [dict[str, np.ndarray], TeleopObservation, Any | None],
            dict[str, np.ndarray] | None,
        ] | None = None
        if ik_solver is not None:
            self.ik_solver = ik_solver
            self.profile = selected_scene_builder(config.scene_profile)
            solver_tasks_configured = False
        else:
            try:
                bundle = selected_ik_builder(
                    RM65DualArmCollisionIKConfig(
                        scene_profile=config.scene_profile,
                        solver=config.solver,
                        solver_frequency=float(config.solver_frequency),
                        damping=float(config.damping),
                        posture_cost=float(config.posture_cost),
                        collision_min_distance_m=float(config.collision_min_distance_m),
                        collision_distance_epsilon_m=float(config.collision_distance_epsilon_m),
                        collision_barrier_gain=float(config.collision_barrier_gain),
                        collision_safe_displacement_gain=float(config.collision_safe_displacement_gain),
                        debug_viz=bool(config.debug_viz),
                        debug_profile=bool(config.debug_profile),
                        ik_tuning=self.ik_tuning,
                    )
                )
                self.ik_solver = bundle.solver
                self.profile = bundle.scene_profile
                solver_tasks_configured = True
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "DualArmCollisionAwareRetargeter requires pinocchio, pink, and qpsolvers "
                    "for joint-space collision avoidance."
                ) from exc
        assert self.profile is not None

        if not solver_tasks_configured:
            objective = self.ik_tuning.objective_weights()
            self.ik_solver.add_frame_task(
                self.profile.left_frame_name,
                position_cost=float(objective.dual_arm_position_cost),
                orientation_cost=float(objective.dual_arm_orientation_cost),
            )
            self.ik_solver.add_frame_task(
                self.profile.right_frame_name,
                position_cost=float(objective.dual_arm_position_cost),
                orientation_cost=float(objective.dual_arm_orientation_cost),
            )
            self.ik_solver.add_posture_task(cost=float(config.posture_cost))

        self._arm_joint_weights = np.asarray(
            self.ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        )
        self._full_joint_weights = self._make_full_joint_weights(self.ik_solver.current_q)
        self._execution_dt_sec = 1.0 / float(config.solver_frequency)
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )
        self._left_arm_limits = self._split_solution(
            np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float)
        )[0], self._split_solution(
            np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float)
        )[0]
        self._right_arm_limits = self._split_solution(
            np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float)
        )[1], self._split_solution(
            np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float)
        )[1]
        self._last_target_poses: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._tool_tcp_from_solver = {
            "left": _pose6_to_matrix(config.left_tool_tcp_pose_flange_m_rad),
            "right": _pose6_to_matrix(config.right_tool_tcp_pose_flange_m_rad),
        }
        self._solver_from_tool_tcp = {
            "left": np.linalg.inv(self._tool_tcp_from_solver["left"]),
            "right": np.linalg.inv(self._tool_tcp_from_solver["right"]),
        }
        self._jump_hold_active = False
        self._jump_hold_full_q: np.ndarray | None = None
        self._jump_hold_anchor_targets: dict[str, np.ndarray] | None = None
        self._jump_hold_frames = 0
        self._profile_frame_index = 0
        self._time_budget_warning_frames = 0
        self._last_solve_iteration_budget_exhausted = False
        self._last_solve_iteration_stop_reason: str | None = None
        self._target_pose_smoother = config.target_pose_smoother
        self._last_output_qpos_smoothing_full_q: np.ndarray | None = None
        self._output_qpos_smoothing_effective_alpha = {
            side: float(config.output_qpos_smoothing_alpha) for side in ("left", "right")
        }
        self._output_qpos_smoothing_update_time: dict[str, float | None] = {
            "left": None,
            "right": None,
        }

        self.set_current_qpos_pair(
            left_qpos=config.left_initial_qpos,
            right_qpos=config.right_initial_qpos,
        )
        self._last_safe_full_q = self.ik_solver.current_q.copy()

    @property
    def name(self) -> str:
        return "dual_arm_collision_aware"

    def reset_target_pose_smoothing(self) -> None:
        reset = getattr(self._target_pose_smoother, "reset", None)
        if callable(reset):
            reset()
        self._last_target_poses = {"left": None, "right": None}
        self._last_output_qpos_smoothing_full_q = None
        self._reset_output_qpos_smoothing_alpha_state()
        self._clear_hold()

    def _reset_output_qpos_smoothing_alpha_state(self) -> None:
        slow_alpha = float(self.config.output_qpos_smoothing_alpha)
        self._output_qpos_smoothing_effective_alpha = {
            "left": slow_alpha,
            "right": slow_alpha,
        }
        self._output_qpos_smoothing_update_time = {"left": None, "right": None}

    def set_current_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray | None = None,
        right_qpos: Sequence[float] | np.ndarray | None = None,
        reset_tracking: bool = True,
    ) -> None:
        left = (
            DEFAULT_LEFT_ARM_QPOS.copy()
            if left_qpos is None
            else np.asarray(left_qpos, dtype=float).reshape(-1)
        )
        right = (
            DEFAULT_RIGHT_ARM_QPOS.copy()
            if right_qpos is None
            else np.asarray(right_qpos, dtype=float).reshape(-1)
        )
        if left.shape != (6,) or right.shape != (6,):
            raise ValueError(
                f"DualArmCollisionAwareRetargeter expects 6D arm qpos per side, got {left.shape} and {right.shape}."
            )
        full_q = self._embed_arm_qpos(
            current_full_q=self.ik_solver.current_q,
            left_qpos=left,
            right_qpos=right,
        )
        self.ik_solver.set_configuration_q(full_q)
        self._last_safe_full_q = full_q.copy()
        if reset_tracking:
            self._last_output_qpos_smoothing_full_q = None
            self._reset_output_qpos_smoothing_alpha_state()
            self._clear_hold()

    def current_world_targets(self) -> dict[str, np.ndarray]:
        # Preserve the established solver protocol for current-state FK: older
        # injected diagnostic solvers may only accept the frame name here.
        return {
            "left": as_pose_matrix(
                self.ik_solver.compute_frame_pose(self.profile.left_frame_name)
            ),
            "right": as_pose_matrix(
                self.ik_solver.compute_frame_pose(self.profile.right_frame_name)
            ),
        }

    def current_qpos_by_side(self) -> dict[str, np.ndarray]:
        """Return a detached arm-qpos snapshot without exposing solver layout."""

        left, right = self._split_solution(self.ik_solver.current_q)
        return {"left": left.copy(), "right": right.copy()}

    def solver_q_from_qpos_by_side(
        self,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray],
        *,
        base_solver_q: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed one complete bimanual qpos pair into the active solver layout.

        The returned configuration is detached and this method never mutates
        the solver.  Requiring both sides prevents callers from accidentally
        combining a requested arm with an implicit or stale opposite arm.
        """

        if not isinstance(qpos_by_side, Mapping):
            raise TypeError("qpos_by_side must be a mapping containing left and right.")
        missing = [side for side in ("left", "right") if side not in qpos_by_side]
        extra = sorted(str(side) for side in qpos_by_side if side not in {"left", "right"})
        if missing or extra:
            raise ValueError(
                "qpos_by_side must contain exactly left and right; "
                f"missing={missing} extra={extra}"
            )
        normalized: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            qpos = np.asarray(qpos_by_side[side], dtype=float).reshape(-1)
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError(f"qpos_by_side[{side!r}] must contain 6 finite values.")
            normalized[side] = qpos.copy()
        base = (
            np.asarray(self.ik_solver.current_q, dtype=float).reshape(-1).copy()
            if base_solver_q is None
            else np.asarray(base_solver_q, dtype=float).reshape(-1).copy()
        )
        if not np.all(np.isfinite(base)):
            raise ValueError("base_solver_q must contain finite values.")
        return self._embed_arm_qpos(
            current_full_q=base,
            left_qpos=normalized["left"],
            right_qpos=normalized["right"],
        )

    def solver_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray,
        right_qpos: Sequence[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute solver-frame FK for a qpos pair without changing IK state."""

        full_q = self.solver_q_from_qpos_by_side(
            {"left": left_qpos, "right": right_qpos}
        )
        return self._solver_world_targets_for_full_q(full_q)

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray,
        right_qpos: Sequence[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute tool-TCP FK for a qpos pair without changing IK state."""

        solver_targets = self.solver_world_targets_for_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
        )
        return {
            side: self.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=solver_targets[side],
            )
            for side in ("left", "right")
        }

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side: str,
        tool_tcp_world_target: np.ndarray,
    ) -> np.ndarray:
        """Convert one public tool-TCP target to the configured solver frame."""

        normalized_side = self._validated_side(side)
        return (
            as_pose_matrix(tool_tcp_world_target)
            @ self._solver_from_tool_tcp[normalized_side]
        )

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side: str,
        solver_world_target: np.ndarray,
    ) -> np.ndarray:
        """Convert one solver-frame target to the configured tool-TCP frame."""

        normalized_side = self._validated_side(side)
        return (
            as_pose_matrix(solver_world_target)
            @ self._tool_tcp_from_solver[normalized_side]
        )

    def current_tool_tcp_world_targets(self) -> dict[str, np.ndarray]:
        """Return FK tool-TCP poses for the currently accepted qpos pair.

        The underlying solver frames are flange/link frames.  Runtime-home
        calibration and simulation feedback need the same tool-TCP semantics
        as the retarget target, so callers should use this public view instead
        of reaching into the solver's private tool transform.
        """

        solver_targets = self.current_world_targets()
        return {
            side: self.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=solver_targets[side],
            )
            for side in ("left", "right")
        }

    def _solver_world_targets_for_full_q(
        self,
        full_q: Sequence[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        q = np.asarray(full_q, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q)):
            raise ValueError("solver configuration must contain finite values.")
        return {
            "left": as_pose_matrix(
                self.ik_solver.compute_frame_pose(self.profile.left_frame_name, q)
            ),
            "right": as_pose_matrix(
                self.ik_solver.compute_frame_pose(self.profile.right_frame_name, q)
            ),
        }

    @staticmethod
    def _validated_side(side: str) -> str:
        normalized = str(side).strip().lower()
        if normalized not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        return normalized

    def check_qpos_pair_collision(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray,
        right_qpos: Sequence[float] | np.ndarray,
    ) -> DualArmQposCollisionCheck:
        """Check the configured collision pairs without changing solver state.

        This is used for a final pre-dispatch check when a later command filter
        replaces one side of a collision-aware IK result (for example fix-arm).
        A missing, malformed, or non-finite distance is intentionally unsafe.
        """

        pair_count = len(tuple(getattr(self.profile, "collision_pairs", ()) or ()))
        required_min_distance_m = float(self.config.collision_min_distance_m)
        try:
            left = np.asarray(left_qpos, dtype=float).reshape(6)
            right = np.asarray(right_qpos, dtype=float).reshape(6)
        except (TypeError, ValueError):
            return DualArmQposCollisionCheck(
                safe=False,
                reason="invalid_qpos",
                collision_pair_count=pair_count,
                required_min_distance_m=required_min_distance_m,
            )
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return DualArmQposCollisionCheck(
                safe=False,
                reason="nonfinite_qpos",
                collision_pair_count=pair_count,
                required_min_distance_m=required_min_distance_m,
            )
        if pair_count <= 0:
            return DualArmQposCollisionCheck(
                safe=False,
                reason="collision_pairs_unavailable",
                collision_pair_count=pair_count,
                required_min_distance_m=required_min_distance_m,
            )
        try:
            full_q = self._embed_arm_qpos(
                current_full_q=np.asarray(self.ik_solver.current_q, dtype=float),
                left_qpos=left,
                right_qpos=right,
            )
            min_distance_m = float(self.ik_solver.compute_min_collision_distance(full_q))
        except Exception as exc:  # noqa: BLE001 - pre-dispatch safety must fail closed.
            return DualArmQposCollisionCheck(
                safe=False,
                reason=f"collision_check_exception:{type(exc).__name__}",
                collision_pair_count=pair_count,
                required_min_distance_m=required_min_distance_m,
            )
        if not np.isfinite(min_distance_m):
            return DualArmQposCollisionCheck(
                safe=False,
                reason="collision_distance_unavailable",
                collision_pair_count=pair_count,
                required_min_distance_m=required_min_distance_m,
            )
        reason = self._fallback_reason_for_distance(min_distance_m)
        return DualArmQposCollisionCheck(
            safe=reason is None,
            reason="" if reason is None else str(reason),
            min_distance_m=min_distance_m,
            collision_pair_count=pair_count,
            required_min_distance_m=required_min_distance_m,
        )

    def _profile_template(self) -> dict[str, object] | None:
        if not self.config.debug_profile:
            return None
        return {
            "solve_calls": 0,
            "solve_total_ms": 0.0,
            "solve_target_setup_ms": 0.0,
            "solve_qp_ms": 0.0,
            "solve_integrate_ms": 0.0,
            "solve_set_configuration_ms": 0.0,
            "solve_iteration_mode": (
                "adaptive_budget"
                if bool(self.config.solve_adaptive_iterations_enabled)
                else "fixed_max"
            ),
            "solve_iteration_max": int(self.config.solve_max_iterations),
            "solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_damping": float(self.config.damping),
            "solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "solve_iteration_budget_ms": max(
                0.0,
                float(self.config.solve_time_budget_ms or 0.0)
                - float(self.config.solve_iteration_reserve_ms),
            ),
            "solve_iteration_budget_exhausted": False,
            "solve_iteration_stop_reason": None,
            "min_distance_calls": 0,
            "min_distance_total_ms": 0.0,
            "min_distance_phase_ms": {},
            "branch_ms": 0.0,
            "jump_gate_ms": 0.0,
            "hold_recovery_ms": 0.0,
            "target_already_reached": False,
            "frame_total_ms": 0.0,
            "time_budget_ms": float(self.config.solve_time_budget_ms or 0.0),
            "time_budget_exceeded": False,
        }

    def _record_solver_profile(self, profile: dict[str, object] | None) -> None:
        if profile is None:
            return
        solver_profile = dict(self.ik_solver.last_solve_profile)
        if not solver_profile:
            return
        profile["solve_calls"] = int(profile.get("solve_calls", 0)) + 1
        for source_key, target_key in (
            ("total_ms", "solve_total_ms"),
            ("target_setup_ms", "solve_target_setup_ms"),
            ("qp_ms", "solve_qp_ms"),
            ("integrate_ms", "solve_integrate_ms"),
            ("set_configuration_ms", "solve_set_configuration_ms"),
        ):
            profile[target_key] = float(profile.get(target_key, 0.0)) + float(
                solver_profile.get(source_key, 0.0)
            )

    def _profiled_min_collision_distance(
        self,
        q: np.ndarray,
        profile: dict[str, object] | None,
        *,
        phase: str,
    ) -> float | None:
        start_time = time.perf_counter()
        min_distance = self.ik_solver.compute_min_collision_distance(q)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if profile is not None:
            profile["min_distance_calls"] = int(profile.get("min_distance_calls", 0)) + 1
            profile["min_distance_total_ms"] = float(
                profile.get("min_distance_total_ms", 0.0)
            ) + elapsed_ms
            phase_totals = profile.get("min_distance_phase_ms")
            if not isinstance(phase_totals, dict):
                phase_totals = {}
                profile["min_distance_phase_ms"] = phase_totals
            phase_totals[phase] = float(phase_totals.get(phase, 0.0)) + elapsed_ms
        return min_distance

    def _emit_profile(
        self,
        *,
        profile: dict[str, object] | None,
        output_metadata: dict[str, object],
        status: str,
        iterations: int,
        branch_candidate: str,
    ) -> None:
        if profile is None:
            return
        self._profile_frame_index += 1
        output_metadata["ik_debug_profile"] = dict(profile)
        print_every = int(self.config.debug_profile_print_every)
        if print_every <= 0 or self._profile_frame_index % print_every != 0:
            return
        print(
            "[dual_arm_collision_aware profile] "
            f"frame={self._profile_frame_index} "
            f"total_ms={float(profile.get('frame_total_ms', 0.0)):.3f} "
            f"solve_calls={int(profile.get('solve_calls', 0))} "
            f"solve_ms={float(profile.get('solve_total_ms', 0.0)):.3f} "
            f"qp_ms={float(profile.get('solve_qp_ms', 0.0)):.3f} "
            f"integrate_ms={float(profile.get('solve_integrate_ms', 0.0)):.3f} "
            f"min_dist_ms={float(profile.get('min_distance_total_ms', 0.0)):.3f} "
            f"iters={iterations} status={status} branch={branch_candidate}"
        )

    def _maybe_warn_time_budget(self, *, elapsed_ms: float, budget_ms: float) -> None:
        if budget_ms <= 0.0 or elapsed_ms <= budget_ms:
            self._time_budget_warning_frames = 0
            return
        self._time_budget_warning_frames += 1
        if self._time_budget_warning_frames == 1 or self._time_budget_warning_frames % 30 == 0:
            print(
                "[dual_arm_collision_aware WARNING] "
                f"IK frame exceeded time budget: elapsed={elapsed_ms:.3f}ms "
                f"budget={budget_ms:.3f}ms status=warning_only"
            )

    def interpolate_world_targets(
        self,
        start_targets: dict[str, np.ndarray],
        end_targets: dict[str, np.ndarray],
        alpha: float,
    ) -> dict[str, np.ndarray]:
        return {
            "left": interpolate_pose(start_targets["left"], end_targets["left"], alpha),
            "right": interpolate_pose(start_targets["right"], end_targets["right"], alpha),
        }

    def build_target_context(
        self,
        observation: TeleopObservation,
        calibration: Any | None = None,
    ) -> DualArmTargetContext:
        if calibration is None:
            raise RuntimeError(f"{self.name} requires a TeleopCalibration instance.")
        if self.left_pose_retargeter is None or self.right_pose_retargeter is None:
            raise RuntimeError(
                f"{self.name} requires injected operator pose adapters for observation retargeting; "
                "target-pose callers should use build_command_for_context()."
            )

        left_command = self.left_pose_retargeter.retarget(observation, calibration)
        right_command = self.right_pose_retargeter.retarget(observation, calibration)
        left_cartesian = left_command.commands.get(self.config.left_effector_name)
        right_cartesian = right_command.commands.get(self.config.right_effector_name)

        left_target = self._target_pose_world(
            command=left_cartesian,
            fallback_frame_name=self.profile.left_frame_name,
            side="left",
            calibration=calibration,
        )
        right_target = self._target_pose_world(
            command=right_cartesian,
            fallback_frame_name=self.profile.right_frame_name,
            side="right",
            calibration=calibration,
        )
        target_smoothing_metadata: dict[str, object] = {"enabled": False}
        if self._target_pose_smoother is not None:
            smoothed_targets, target_smoothing_metadata = (
                self._target_pose_smoother.smooth_pair(
                    left_target=left_target,
                    right_target=right_target,
                    left_active=left_cartesian is not None,
                    right_active=right_cartesian is not None,
                    timestamp=observation.timestamp,
                )
            )
            left_target = as_pose_matrix(smoothed_targets["left"])
            right_target = as_pose_matrix(smoothed_targets["right"])
        return DualArmTargetContext(
            timestamp=observation.timestamp,
            left_target=left_target.copy(),
            right_target=right_target.copy(),
            left_world_to_base=(
                None
                if calibration.get_arm_for_side("left").T_D_B is None
                else invert_pose(calibration.get_arm_for_side("left").T_D_B)
            ),
            right_world_to_base=(
                None
                if calibration.get_arm_for_side("right").T_D_B is None
                else invert_pose(calibration.get_arm_for_side("right").T_D_B)
            ),
            left_cartesian=left_cartesian.copy() if left_cartesian is not None else None,
            right_cartesian=right_cartesian.copy() if right_cartesian is not None else None,
            left_input_skipped=left_command.metadata.get("skipped"),
            right_input_skipped=right_command.metadata.get("skipped"),
            target_smoothing_metadata=dict(target_smoothing_metadata),
        )

    def _bimanual_target_from_context(
        self,
        context: DualArmTargetContext,
        *,
        targets: Mapping[str, np.ndarray] | None = None,
    ) -> BimanualPoseTarget:
        target_pose_by_side = {
            "left": as_pose_matrix(
                context.left_target if targets is None else targets["left"]
            ),
            "right": as_pose_matrix(
                context.right_target if targets is None else targets["right"]
            ),
        }

        def source_from_command(
            command: EffectorCommand | None,
        ) -> BimanualPoseTargetSource | None:
            if command is None:
                return None
            return BimanualPoseTargetSource(
                command_type=command.command_type,
                value=np.asarray(command.value, dtype=float),
                frame_name=command.frame_name,
                metadata=dict(command.metadata),
                _data_is_owned=True,
            )

        source_by_side = {
            "left": source_from_command(context.left_cartesian),
            "right": source_from_command(context.right_cartesian),
        }
        return BimanualPoseTarget(
            timestamp=context.timestamp,
            target_pose_by_side=target_pose_by_side,
            active_sides=tuple(
                side for side in ("left", "right") if source_by_side[side] is not None
            ),
            world_to_base_by_side={
                "left": context.left_world_to_base,
                "right": context.right_world_to_base,
            },
            source_by_side=source_by_side,
            input_skipped_by_side={
                "left": context.left_input_skipped,
                "right": context.right_input_skipped,
            },
            target_smoothing_metadata=dict(context.target_smoothing_metadata),
            _data_is_owned=True,
        )

    def solve_target_pair(
        self,
        target: BimanualPoseTarget,
    ) -> BimanualIKResult:
        """Solve one atomic bimanual target through a runtime-independent API.

        The shared state machine emits this result directly; no intermediate
        ``TeleopCommand`` is constructed or decoded on this path.
        """

        result = self._solve_bimanual_core(target, output_kind="result")
        assert isinstance(result, BimanualIKResult)
        return result

    def _command_from_bimanual_result(
        self,
        result: BimanualIKResult,
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=result.timestamp)
        command.metadata.update(dict(result.metadata))
        for side, effector_name in (
            ("left", self.config.left_effector_name),
            ("right", self.config.right_effector_name),
        ):
            side_result = result.side_results[side]
            command.add_command(
                EffectorCommand(
                    effector_name=effector_name,
                    side=side,
                    command_type=side_result.command_type,
                    value=side_result.command_value.copy(),
                    frame_name=side_result.command_frame_name,
                    metadata=dict(side_result.metadata),
                )
            )
        return command

    def build_command_for_context(
        self,
        context: DualArmTargetContext,
        *,
        targets: dict[str, np.ndarray] | None = None,
    ) -> TeleopCommand:
        if not self.config.bimanual_result_contract_enabled:
            return self._build_command_for_context_compat(context, targets=targets)
        target = self._bimanual_target_from_context(context, targets=targets)
        result = self._solve_bimanual_core(
            target,
            output_kind="result",
        )
        assert isinstance(result, BimanualIKResult)
        return self._command_from_bimanual_result(result)

    def _build_command_for_context_compat(
        self,
        context: DualArmTargetContext,
        *,
        targets: dict[str, np.ndarray] | None = None,
    ) -> TeleopCommand:
        command = self._solve_bimanual_core(
            context,
            targets=targets,
            output_kind="command",
        )
        assert isinstance(command, TeleopCommand)
        return command

    def _solve_bimanual_core(
        self,
        solve_input: DualArmTargetContext | BimanualPoseTarget,
        *,
        targets: dict[str, np.ndarray] | None = None,
        output_kind: Literal["command", "result"],
    ) -> TeleopCommand | BimanualIKResult:
        if isinstance(solve_input, BimanualPoseTarget):
            if targets is not None:
                raise ValueError("targets override is only valid for DualArmTargetContext.")
            timestamp = solve_input.timestamp
            left_input_target = solve_input.target_pose_by_side["left"]
            right_input_target = solve_input.target_pose_by_side["right"]
            left_world_to_base = solve_input.world_to_base_by_side["left"]
            right_world_to_base = solve_input.world_to_base_by_side["right"]
            left_source = solve_input.source_by_side["left"]
            right_source = solve_input.source_by_side["right"]
            left_source_command_type = (
                None if left_source is None else left_source.command_type
            )
            right_source_command_type = (
                None if right_source is None else right_source.command_type
            )
            left_source_value = None if left_source is None else left_source.value
            right_source_value = None if right_source is None else right_source.value
            left_source_metadata = {} if left_source is None else left_source.metadata
            right_source_metadata = {} if right_source is None else right_source.metadata
            left_input_skipped = solve_input.input_skipped_by_side["left"]
            right_input_skipped = solve_input.input_skipped_by_side["right"]
            target_smoothing_metadata = solve_input.target_smoothing_metadata
        else:
            timestamp = solve_input.timestamp
            left_input_target = (
                solve_input.left_target if targets is None else targets["left"]
            )
            right_input_target = (
                solve_input.right_target if targets is None else targets["right"]
            )
            left_world_to_base = solve_input.left_world_to_base
            right_world_to_base = solve_input.right_world_to_base
            left_source_command_type = (
                None
                if solve_input.left_cartesian is None
                else solve_input.left_cartesian.command_type
            )
            right_source_command_type = (
                None
                if solve_input.right_cartesian is None
                else solve_input.right_cartesian.command_type
            )
            left_source_value = (
                None
                if solve_input.left_cartesian is None
                else solve_input.left_cartesian.value
            )
            right_source_value = (
                None
                if solve_input.right_cartesian is None
                else solve_input.right_cartesian.value
            )
            left_source_metadata = (
                {} if solve_input.left_cartesian is None else solve_input.left_cartesian.metadata
            )
            right_source_metadata = (
                {}
                if solve_input.right_cartesian is None
                else solve_input.right_cartesian.metadata
            )
            left_input_skipped = solve_input.left_input_skipped
            right_input_skipped = solve_input.right_input_skipped
            target_smoothing_metadata = solve_input.target_smoothing_metadata

        frame_start = time.perf_counter()
        profile = self._profile_template()
        left_target = as_pose_matrix(left_input_target)
        right_target = as_pose_matrix(right_input_target)
        current_targets = {"left": left_target.copy(), "right": right_target.copy()}
        solver_targets_before_solve = self.current_world_targets()
        target_activity_metadata = self._target_activity_metadata(
            current_targets,
            source_metadata_by_side={
                "left": left_source_metadata,
                "right": right_source_metadata,
            },
        )
        left_source_target_pose_base = (
            None
            if left_source_command_type != "cartesian_pose" or left_source_value is None
            else as_pose_matrix(left_source_value)
        )
        right_source_target_pose_base = (
            None
            if right_source_command_type != "cartesian_pose" or right_source_value is None
            else as_pose_matrix(right_source_value)
        )
        left_source_target_pose_desk = left_source_metadata.get("target_pose_desk")
        right_source_target_pose_desk = right_source_metadata.get("target_pose_desk")
        left_source_target_pose_operator = left_source_metadata.get(
            "target_pose_operator"
        )
        right_source_target_pose_operator = right_source_metadata.get(
            "target_pose_operator"
        )

        output_metadata: dict[str, object] = {}
        output_metadata.update(
            {
                "controller": self.name,
                "scene_profile": self.profile.name,
                "collision_pair_count": len(self.profile.collision_pairs),
                "collision_avoidance_enabled": True,
                "collision_avoidance_mode": self.config.contact_mode,
                "collision_min_distance_m": float(self.config.collision_min_distance_m),
                "left_input_skipped": left_input_skipped,
                "right_input_skipped": right_input_skipped,
                "ik_framework": self.config.ik_framework,
                "ik_objective_mode": self.ik_tuning.ik_objective_mode,
                "ik_tuning_source": self.ik_tuning.source_path,
                "ik_safety_profile": self.ik_safety_profile.name,
                "joint_velocity_limit_profile": (
                    None if self.joint_velocity_profile is None else self.joint_velocity_profile.name
                ),
                "source_target_pose_base": {
                    "left": left_source_target_pose_base,
                    "right": right_source_target_pose_base,
                },
                "source_target_pose_desk": {
                    "left": left_source_target_pose_desk,
                    "right": right_source_target_pose_desk,
                },
                "source_target_pose_operator": {
                    "left": left_source_target_pose_operator,
                    "right": right_source_target_pose_operator,
                },
                "target_pose_smoothing": dict(target_smoothing_metadata),
                "target_activity": dict(target_activity_metadata),
            }
        )

        status = "tracking"
        fallback_reason: str | None = None
        min_distance_m: float | None = None
        branch_candidate = "original"
        branch_score_deg: float | None = None
        jump_gate_status = "pass"
        jump_gate_reason: str | None = None
        jump_gate_release_ready = False
        jump_gate_hold_frames = self._jump_hold_frames
        jump_gate_metrics: dict[str, object] = {}
        candidate_selection_metadata: dict[str, object] = {}
        execution_step_metadata: dict[str, object] = {
            "enabled": bool(self.config.enforce_realistic_joint_step),
            "step_limited": False,
            "weighted_delta_deg": 0.0,
            "left_max_abs_delta_deg": 0.0,
            "right_max_abs_delta_deg": 0.0,
        }
        solve_iteration_count = 0
        solve_pose_error: dict[str, dict[str, float]] = {}
        raw_solver_q: np.ndarray | None = None
        self._last_solve_iteration_budget_exhausted = False
        self._last_solve_iteration_stop_reason = "not_started"

        pre_solve_q = self.ik_solver.current_q.copy()
        active_q = pre_solve_q.copy()
        current_target_hold_enabled = bool(self.config.current_target_hold_enabled)
        target_already_reached = current_target_hold_enabled
        for side in ("left", "right"):
            translation_m, rotation_deg = pose_delta_metrics(
                solver_targets_before_solve[side],
                current_targets[side],
            )
            solve_pose_error[side] = {
                "translation_m": float(translation_m),
                "rotation_deg": float(rotation_deg),
            }
            if current_target_hold_enabled and (
                translation_m > _CURRENT_TARGET_HOLD_TRANSLATION_M
                or rotation_deg > _CURRENT_TARGET_HOLD_ROTATION_DEG
            ):
                target_already_reached = False
        if profile is not None:
            profile["current_target_hold_enabled"] = bool(current_target_hold_enabled)
            profile["target_already_reached"] = bool(target_already_reached)

        try:
            if target_already_reached:
                candidate_q = pre_solve_q.copy()
                branch_candidate = "current_pose_hold"
                branch_score_deg = 0.0
                branch_min_distance = self._profiled_min_collision_distance(
                    candidate_q,
                    profile,
                    phase="current_pose_hold",
                )
            else:
                solved_q, solve_iteration_count, solve_pose_error = self._solve_toward_targets(
                    left_target=left_target,
                    right_target=right_target,
                    profile=profile,
                    frame_start_time=frame_start,
                )
                raw_solver_q = np.asarray(solved_q, dtype=float).copy()
                branch_start = time.perf_counter()
                elapsed_before_selection_ms = (branch_start - frame_start) * 1000.0
                candidate_q, branch_candidate, branch_score_deg, branch_min_distance, candidate_selection_metadata = (
                    self._select_safe_tracking_candidate(
                        raw_full_q=solved_q,
                        reference_full_q=pre_solve_q,
                        current_targets=current_targets,
                        target_activity=target_activity_metadata,
                        profile=profile,
                        elapsed_before_selection_ms=elapsed_before_selection_ms,
                    )
                )
                if profile is not None:
                    profile["branch_ms"] = float(profile.get("branch_ms", 0.0)) + (
                        time.perf_counter() - branch_start
                    ) * 1000.0
                gate_start = time.perf_counter()
                gate_result = self._evaluate_jump_gate(
                    reference_full_q=pre_solve_q,
                    candidate_full_q=candidate_q,
                    current_targets=current_targets,
                )
                if profile is not None:
                    profile["jump_gate_ms"] = float(profile.get("jump_gate_ms", 0.0)) + (
                        time.perf_counter() - gate_start
                    ) * 1000.0
                jump_gate_status = str(gate_result["status"])
                jump_gate_reason = (
                    None if gate_result["reason"] is None else str(gate_result["reason"])
                )
                jump_gate_release_ready = bool(gate_result["release_ready"])
                jump_gate_metrics = dict(gate_result["metrics"])

                if bool(gate_result["hold"]) and not jump_gate_release_ready:
                    self._enter_hold(hold_full_q=pre_solve_q, current_targets=current_targets)
                    jump_gate_hold_frames = self._jump_hold_frames
                    active_q = (
                        self._jump_hold_full_q.copy()
                        if self._jump_hold_full_q is not None
                        else pre_solve_q.copy()
                    )
                    self.ik_solver.set_configuration_q(active_q)
                    status = "jump_gate_hold"
                    fallback_reason = jump_gate_reason
                    recovery_start = time.perf_counter()
                    recovery_result = self._attempt_hold_recovery(
                        current_targets=current_targets,
                        profile=profile,
                    )
                    if profile is not None:
                        profile["hold_recovery_ms"] = float(
                            profile.get("hold_recovery_ms", 0.0)
                        ) + (time.perf_counter() - recovery_start) * 1000.0
                    if recovery_result is not None:
                        active_q = recovery_result["q"].copy()
                        min_distance_m = recovery_result["min_distance_m"]
                        branch_candidate = str(recovery_result["candidate_name"])
                        branch_score_deg = float(recovery_result["score_deg"])
                        jump_gate_status = str(recovery_result["jump_gate_status"])
                        jump_gate_reason = (
                            None
                            if recovery_result["jump_gate_reason"] is None
                            else str(recovery_result["jump_gate_reason"])
                        )
                        jump_gate_release_ready = bool(recovery_result["release_ready"])
                        jump_gate_metrics = dict(recovery_result["metrics"])
                        near_boundary = (
                            min_distance_m is not None
                            and min_distance_m
                            <= float(self.config.collision_min_distance_m)
                            + float(self.config.collision_distance_epsilon_m)
                        )
                        status = "boundary_active" if near_boundary else "tracking"
                        fallback_reason = None
                        self.ik_solver.set_configuration_q(active_q)
                        self._last_safe_full_q = active_q.copy()
                        self._clear_hold()
                else:
                    min_distance_m = (
                        branch_min_distance
                        if branch_min_distance is not None
                        else self._profiled_min_collision_distance(
                            candidate_q,
                            profile,
                            phase="final_candidate",
                        )
                    )
                    fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                    if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                        self.ik_solver.set_configuration_q(self._last_safe_full_q)
                        active_q = self._last_safe_full_q.copy()
                        status = "hold_last_safe"
                    else:
                        active_q = candidate_q.copy()
                        self.ik_solver.set_configuration_q(active_q)
                        near_boundary = (
                            min_distance_m is not None
                            and min_distance_m
                            <= float(self.config.collision_min_distance_m)
                            + float(self.config.collision_distance_epsilon_m)
                        )
                        status = "boundary_active" if near_boundary else "tracking"
                        self._last_safe_full_q = active_q.copy()
                    self._clear_hold()
            if target_already_reached:
                min_distance_m = (
                    branch_min_distance
                    if branch_min_distance is not None
                    else self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="target_hold_finalize",
                    )
                )
                fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                active_q = candidate_q.copy()
                self.ik_solver.set_configuration_q(active_q)
                if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                    active_q = self._last_safe_full_q.copy()
                    self.ik_solver.set_configuration_q(active_q)
                    status = "hold_last_safe"
                else:
                    near_boundary = (
                        min_distance_m is not None
                        and min_distance_m
                        <= float(self.config.collision_min_distance_m)
                        + float(self.config.collision_distance_epsilon_m)
                    )
                    status = "boundary_active" if near_boundary else "tracking"
                    self._last_safe_full_q = active_q.copy()
                self._clear_hold()
        except Exception as exc:
            fallback_reason = f"ik_exception:{type(exc).__name__}"
            if self.config.hold_last_safe_on_failure:
                self.ik_solver.set_configuration_q(self._last_safe_full_q)
                active_q = self._last_safe_full_q.copy()
                status = "hold_last_safe"
            else:
                self.ik_solver.set_configuration_q(pre_solve_q)
                active_q = pre_solve_q.copy()
                status = "unsafe_candidate"

        if status not in {"jump_gate_hold", "hold_last_safe", "unsafe_candidate"}:
            active_q, execution_step_metadata = self._limit_to_execution_step(
                reference_full_q=pre_solve_q,
                candidate_full_q=active_q,
                target_activity=target_activity_metadata,
                current_targets=current_targets,
            )
            self.ik_solver.set_configuration_q(active_q)
            if bool(execution_step_metadata.get("enabled")):
                min_distance_m = self._profiled_min_collision_distance(
                    active_q,
                    profile,
                    phase="execution_step",
                )
                fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                    self.ik_solver.set_configuration_q(self._last_safe_full_q)
                    active_q = self._last_safe_full_q.copy()
                    status = "hold_last_safe"
                else:
                    near_boundary = (
                        min_distance_m is not None
                        and min_distance_m
                        <= float(self.config.collision_min_distance_m)
                        + float(self.config.collision_distance_epsilon_m)
                    )
                    status = "boundary_active" if near_boundary else "tracking"
                    self._last_safe_full_q = active_q.copy()

        left_qpos, right_qpos = self._split_solution(active_q)
        left_inactive_jitter_hold = self._inactive_jitter_hold_qpos(
            metadata=left_source_metadata
        )
        if left_inactive_jitter_hold is not None:
            left_qpos = left_inactive_jitter_hold
            active_q = self._embed_arm_qpos(
                current_full_q=active_q,
                left_qpos=left_qpos,
                right_qpos=right_qpos,
            )
            self.ik_solver.set_configuration_q(active_q)
            self._last_safe_full_q = active_q.copy()
        active_q, left_qpos, right_qpos, output_qpos_smoothing = (
            self._smooth_output_qpos_before_action(
                active_q=active_q,
                left_qpos=left_qpos,
                right_qpos=right_qpos,
                target_activity=target_activity_metadata,
            )
        )
        if output_qpos_smoothing.get("enabled", False):
            self.ik_solver.set_configuration_q(active_q)
            self._last_safe_full_q = active_q.copy()
        left_limit_margin = self._joint_limit_margin_metadata("left", left_qpos)
        right_limit_margin = self._joint_limit_margin_metadata("right", right_qpos)
        raw_left_qpos = None
        raw_right_qpos = None
        if raw_solver_q is not None:
            raw_left_qpos, raw_right_qpos = self._split_solution(raw_solver_q)
        frame_total_ms = (time.perf_counter() - frame_start) * 1000.0
        time_budget_ms = float(self.config.solve_time_budget_ms or 0.0)
        time_budget_exceeded = bool(time_budget_ms > 0.0 and frame_total_ms > time_budget_ms)
        self._maybe_warn_time_budget(elapsed_ms=frame_total_ms, budget_ms=time_budget_ms)
        if profile is not None:
            profile["frame_total_ms"] = frame_total_ms
            profile["time_budget_ms"] = time_budget_ms
            profile["time_budget_exceeded"] = time_budget_exceeded
        output_metadata.update(
            {
                "collision_avoidance_status": status,
                "collision_avoidance_fallback_reason": fallback_reason,
                "collision_distance_m": min_distance_m,
                "ik_branch_candidate": branch_candidate,
                "ik_branch_distance_score_deg": branch_score_deg,
                "ik_solve_iterations": int(solve_iteration_count),
                "ik_solve_iteration_mode": (
                    "adaptive_budget"
                    if bool(self.config.solve_adaptive_iterations_enabled)
                    else "fixed_max"
                ),
                "ik_solve_iteration_max": int(self.config.solve_max_iterations),
                "ik_solve_iteration_min": int(self.config.solve_min_iterations),
                "ik_damping": float(self.config.damping),
                "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
                "ik_solve_iteration_budget_ms": max(
                    0.0,
                    float(self.config.solve_time_budget_ms or 0.0)
                    - float(self.config.solve_iteration_reserve_ms),
                ),
                "ik_solve_iteration_budget_exhausted": bool(
                    self._last_solve_iteration_budget_exhausted
                ),
                "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
                "ik_solve_pose_error": {
                    side: dict(metrics) for side, metrics in solve_pose_error.items()
                },
                "ik_jump_gate_status": jump_gate_status,
                "ik_jump_gate_reason": jump_gate_reason,
                "ik_jump_gate_release_ready": jump_gate_release_ready,
                "ik_jump_gate_hold_frames": jump_gate_hold_frames,
                "ik_jump_gate_metrics": jump_gate_metrics,
                "ik_candidate_selection": dict(candidate_selection_metadata),
                "joint_limit_margin": {
                    "left": dict(left_limit_margin),
                    "right": dict(right_limit_margin),
                },
                "ik_frame_total_ms": frame_total_ms,
                "ik_time_budget_ms": time_budget_ms,
                "ik_time_budget_exceeded": time_budget_exceeded,
                "execution_step_metadata": dict(execution_step_metadata),
                "output_qpos_smoothing": dict(output_qpos_smoothing),
                "left_inactive_jitter_hold": left_inactive_jitter_hold is not None,
                "target_pose_smoothing": dict(target_smoothing_metadata),
                "target_activity": dict(target_activity_metadata),
            }
        )
        left_metadata = {
            "controller": self.name,
            "scene_profile": self.profile.name,
            "collision_pair_count": len(self.profile.collision_pairs),
            "collision_avoidance_mode": self.config.contact_mode,
            "collision_avoidance_status": status,
            "collision_avoidance_fallback_reason": fallback_reason,
            "collision_distance_m": min_distance_m,
            "source_command_type": (
                left_source_command_type
            ),
            "source_target_pose_base": left_source_target_pose_base,
            "source_target_pose_desk": left_source_target_pose_desk,
            "source_target_pose_operator": left_source_target_pose_operator,
            "ik_framework": self.config.ik_framework,
            "ik_objective_mode": self.ik_tuning.ik_objective_mode,
            "ik_branch_candidate": branch_candidate,
            "ik_branch_distance_score_deg": branch_score_deg,
            "ik_solve_iterations": int(solve_iteration_count),
            "ik_solve_iteration_mode": (
                "adaptive_budget"
                if bool(self.config.solve_adaptive_iterations_enabled)
                else "fixed_max"
            ),
            "ik_solve_iteration_max": int(self.config.solve_max_iterations),
            "ik_solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_damping": float(self.config.damping),
            "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "ik_solve_iteration_budget_ms": max(
                0.0,
                float(self.config.solve_time_budget_ms or 0.0)
                - float(self.config.solve_iteration_reserve_ms),
            ),
            "ik_solve_iteration_budget_exhausted": bool(
                self._last_solve_iteration_budget_exhausted
            ),
            "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
            "ik_solve_pose_error": dict(solve_pose_error.get("left", {})),
            "ik_jump_gate_status": jump_gate_status,
            "ik_jump_gate_reason": jump_gate_reason,
            "ik_jump_gate_release_ready": jump_gate_release_ready,
            "ik_jump_gate_hold_frames": jump_gate_hold_frames,
            "ik_jump_gate_metrics": dict(jump_gate_metrics.get("left", {})),
            "ik_candidate_selection": dict(candidate_selection_metadata),
            "joint_limit_margin": dict(left_limit_margin),
            "ik_frame_total_ms": frame_total_ms,
            "ik_time_budget_ms": time_budget_ms,
            "ik_time_budget_exceeded": time_budget_exceeded,
            "execution_step_metadata": dict(execution_step_metadata),
            "output_qpos_smoothing": dict(output_qpos_smoothing.get("left", {})),
            "inactive_jitter_hold": left_inactive_jitter_hold is not None,
            "target_pose_smoothing": dict(
                target_smoothing_metadata.get("left", {})
            ),
            "target_activity": dict(target_activity_metadata.get("left", {})),
        }
        right_metadata = {
            "controller": self.name,
            "scene_profile": self.profile.name,
            "collision_pair_count": len(self.profile.collision_pairs),
            "collision_avoidance_mode": self.config.contact_mode,
            "collision_avoidance_status": status,
            "collision_avoidance_fallback_reason": fallback_reason,
            "collision_distance_m": min_distance_m,
            "source_command_type": (
                right_source_command_type
            ),
            "source_target_pose_base": right_source_target_pose_base,
            "source_target_pose_desk": right_source_target_pose_desk,
            "source_target_pose_operator": right_source_target_pose_operator,
            "ik_framework": self.config.ik_framework,
            "ik_objective_mode": self.ik_tuning.ik_objective_mode,
            "ik_branch_candidate": branch_candidate,
            "ik_branch_distance_score_deg": branch_score_deg,
            "ik_solve_iterations": int(solve_iteration_count),
            "ik_solve_iteration_mode": (
                "adaptive_budget"
                if bool(self.config.solve_adaptive_iterations_enabled)
                else "fixed_max"
            ),
            "ik_solve_iteration_max": int(self.config.solve_max_iterations),
            "ik_solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_damping": float(self.config.damping),
            "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "ik_solve_iteration_budget_ms": max(
                0.0,
                float(self.config.solve_time_budget_ms or 0.0)
                - float(self.config.solve_iteration_reserve_ms),
            ),
            "ik_solve_iteration_budget_exhausted": bool(
                self._last_solve_iteration_budget_exhausted
            ),
            "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
            "ik_solve_pose_error": dict(solve_pose_error.get("right", {})),
            "ik_jump_gate_status": jump_gate_status,
            "ik_jump_gate_reason": jump_gate_reason,
            "ik_jump_gate_release_ready": jump_gate_release_ready,
            "ik_jump_gate_hold_frames": jump_gate_hold_frames,
            "ik_jump_gate_metrics": dict(jump_gate_metrics.get("right", {})),
            "ik_candidate_selection": dict(candidate_selection_metadata),
            "joint_limit_margin": dict(right_limit_margin),
            "ik_frame_total_ms": frame_total_ms,
            "ik_time_budget_ms": time_budget_ms,
            "ik_time_budget_exceeded": time_budget_exceeded,
            "execution_step_metadata": dict(execution_step_metadata),
            "output_qpos_smoothing": dict(output_qpos_smoothing.get("right", {})),
            "target_pose_smoothing": dict(
                target_smoothing_metadata.get("right", {})
            ),
            "target_activity": dict(target_activity_metadata.get("right", {})),
        }
        left_world_pose = self.ik_solver.compute_frame_pose(self.profile.left_frame_name, active_q)
        right_world_pose = self.ik_solver.compute_frame_pose(self.profile.right_frame_name, active_q)
        left_base_pose = (
            left_world_pose
            if left_world_to_base is None
            else left_world_to_base @ left_world_pose
        )
        right_base_pose = (
            right_world_pose
            if right_world_to_base is None
            else right_world_to_base @ right_world_pose
        )
        left_tcp_base_pose = left_base_pose @ self._tool_tcp_from_solver["left"]
        right_tcp_base_pose = right_base_pose @ self._tool_tcp_from_solver["right"]
        left_tcp_world_pose = left_world_pose @ self._tool_tcp_from_solver["left"]
        right_tcp_world_pose = right_world_pose @ self._tool_tcp_from_solver["right"]
        left_metadata.update(
            {
                "target_pose_base": left_tcp_base_pose.copy(),
                "target_pose_tcp_base": left_tcp_base_pose.copy(),
                "target_pose_controlled_frame": TOOL_TCP_FRAME,
                "solver_output_endpoint_pose_base": left_base_pose.copy(),
                "solver_output_qpos": left_qpos.copy(),
                "q_ik_raw": None if raw_left_qpos is None else raw_left_qpos.copy(),
                "action_pose_source": "limited_solver_output_qpos",
            }
        )
        right_metadata.update(
            {
                "target_pose_base": right_tcp_base_pose.copy(),
                "target_pose_tcp_base": right_tcp_base_pose.copy(),
                "target_pose_controlled_frame": TOOL_TCP_FRAME,
                "solver_output_endpoint_pose_base": right_base_pose.copy(),
                "solver_output_qpos": right_qpos.copy(),
                "q_ik_raw": None if raw_right_qpos is None else raw_right_qpos.copy(),
                "action_pose_source": "limited_solver_output_qpos",
            }
        )
        if self.config.output_command_type == "cartesian_pose":
            left_metadata.update(
                {
                    "arm_controlled_frame": TOOL_TCP_FRAME,
                    "arm_reference_frame": "arm_base",
                    "arm_action_spec": {
                        "side": "left",
                        "controlled_frame": TOOL_TCP_FRAME,
                        "reference_frame": "arm_base",
                        "representation": "xyz_quat",
                        "dof": 7,
                        "semantics": TOOL_TCP_SEMANTICS,
                        "metadata": {"controller": self.name},
                    },
                    "target_pose_desk": left_tcp_world_pose.copy(),
                    "target_pose_base": left_tcp_base_pose.copy(),
                    "target_pose_tcp_base": left_tcp_base_pose.copy(),
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                    "solver_output_endpoint_pose_base": left_base_pose.copy(),
                    "solver_output_qpos": left_qpos.copy(),
                }
            )
            right_metadata.update(
                {
                    "arm_controlled_frame": TOOL_TCP_FRAME,
                    "arm_reference_frame": "arm_base",
                    "arm_action_spec": {
                        "side": "right",
                        "controlled_frame": TOOL_TCP_FRAME,
                        "reference_frame": "arm_base",
                        "representation": "xyz_quat",
                        "dof": 7,
                        "semantics": TOOL_TCP_SEMANTICS,
                        "metadata": {"controller": self.name},
                    },
                    "target_pose_desk": right_tcp_world_pose.copy(),
                    "target_pose_base": right_tcp_base_pose.copy(),
                    "target_pose_tcp_base": right_tcp_base_pose.copy(),
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                    "solver_output_endpoint_pose_base": right_base_pose.copy(),
                    "solver_output_qpos": right_qpos.copy(),
                }
            )
            left_command_type = "cartesian_pose"
            right_command_type = "cartesian_pose"
            left_command_value = left_tcp_base_pose
            right_command_value = right_tcp_base_pose
            left_command_frame_name = "arm_base"
            right_command_frame_name = "arm_base"
        else:
            left_command_type = "joint_vector"
            right_command_type = "joint_vector"
            left_command_value = left_qpos
            right_command_value = right_qpos
            left_command_frame_name = None
            right_command_frame_name = None
        self._last_target_poses = {
            "left": left_target.copy(),
            "right": right_target.copy(),
        }
        self._emit_profile(
            profile=profile,
            output_metadata=output_metadata,
            status=status,
            iterations=solve_iteration_count,
            branch_candidate=branch_candidate,
        )
        if output_kind == "result":
            return BimanualIKResult(
                timestamp=timestamp,
                status=status,
                fallback_reason=fallback_reason,
                side_results={
                    "left": BimanualIKSideResult(
                        side="left",
                        qpos=left_qpos,
                        raw_qpos=raw_left_qpos,
                        solver_pose_world=left_world_pose,
                        tool_tcp_pose_world=left_tcp_world_pose,
                        command_type=left_command_type,
                        command_value=left_command_value,
                        command_frame_name=left_command_frame_name,
                        metadata=left_metadata,
                        _metadata_is_owned=True,
                    ),
                    "right": BimanualIKSideResult(
                        side="right",
                        qpos=right_qpos,
                        raw_qpos=raw_right_qpos,
                        solver_pose_world=right_world_pose,
                        tool_tcp_pose_world=right_tcp_world_pose,
                        command_type=right_command_type,
                        command_value=right_command_value,
                        command_frame_name=right_command_frame_name,
                        metadata=right_metadata,
                        _metadata_is_owned=True,
                    ),
                },
                metadata=output_metadata,
                _metadata_is_owned=True,
            )

        command = TeleopCommand(timestamp=timestamp)
        command.metadata.update(output_metadata)
        command.add_command(
            EffectorCommand(
                effector_name=self.config.left_effector_name,
                side="left",
                command_type=left_command_type,
                value=left_command_value,
                frame_name=left_command_frame_name,
                metadata=left_metadata,
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name=self.config.right_effector_name,
                side="right",
                command_type=right_command_type,
                value=right_command_value,
                frame_name=right_command_frame_name,
                metadata=right_metadata,
            )
        )
        return command

    def retarget(
        self, observation: TeleopObservation, calibration: Any | None = None
    ) -> TeleopCommand:
        context = self.build_target_context(observation, calibration)
        targets = None
        if self.target_override_filter is not None:
            maybe_targets = self.target_override_filter(
                {
                    "left": context.left_target.copy(),
                    "right": context.right_target.copy(),
                },
                observation,
                calibration,
            )
            if maybe_targets is not None:
                targets = {
                    "left": as_pose_matrix(maybe_targets["left"]).copy(),
                    "right": as_pose_matrix(maybe_targets["right"]).copy(),
                }
        return self.build_command_for_context(context, targets=targets)

    def _target_pose_world(
        self,
        *,
        command: EffectorCommand | None,
        fallback_frame_name: str,
        side: Literal["left", "right"],
        calibration: Any | None = None,
    ) -> np.ndarray:
        if command is None:
            return self.ik_solver.compute_frame_pose(fallback_frame_name)
        if command.command_type != "cartesian_pose":
            raise ValueError(
                f"{self.name} expected cartesian_pose from the composed arm retargeter, got {command.command_type}."
            )
        target_pose_desk = command.metadata.get("target_pose_desk")
        if target_pose_desk is not None:
            return self._solver_frame_target_pose(
                pose_world=as_pose_matrix(target_pose_desk),
                command=command,
                fallback_frame_name=fallback_frame_name,
                side=side,
            )
        pose_base = as_pose_matrix(command.value)
        if calibration is not None:
            arm = calibration.get_arm_for_side(side)
            if arm.T_D_B is not None:
                return self._solver_frame_target_pose(
                    pose_world=arm.T_D_B @ pose_base,
                    command=command,
                    fallback_frame_name=fallback_frame_name,
                    side=side,
                )
        pose_transform = (
            self.profile.left_pose_transform
            if side == "left"
            else self.profile.right_pose_transform
        )
        return self._solver_frame_target_pose(
            pose_world=pose_transform @ pose_base,
            command=command,
            fallback_frame_name=fallback_frame_name,
            side=side,
        )

    def _inactive_jitter_hold_qpos(
        self,
        command: EffectorCommand | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> np.ndarray | None:
        if metadata is None:
            if command is None:
                return None
            metadata = command.metadata or {}
        if not bool(metadata.get("jitter_arm")):
            return None
        if bool(metadata.get("jitter_active", True)):
            return None
        hold_qpos = metadata.get("hold_qpos")
        if hold_qpos is None:
            return None
        qpos = np.asarray(hold_qpos, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            return None
        return qpos.copy()

    def _solver_frame_target_pose(
        self,
        *,
        pose_world: np.ndarray,
        command: EffectorCommand,
        fallback_frame_name: str,
        side: Literal["left", "right"],
    ) -> np.ndarray:
        pose_world = as_pose_matrix(pose_world)
        controlled_frame = canonical_arm_controlled_frame(
            command.metadata.get("arm_controlled_frame")
            or command.metadata.get("target_pose_controlled_frame")
            or command.metadata.get("controlled_frame")
        )
        if controlled_frame == TOOL_TCP_FRAME:
            return pose_world @ self._solver_from_tool_tcp[side]
        if controlled_frame == ARM_ENDPOINT_FRAME:
            return pose_world
        tcp_frame = canonical_tcp_frame(command.metadata.get("tcp_frame", HAND_BASE_TCP_FRAME))
        if tcp_frame != HAND_BASE_TCP_FRAME:
            return pose_world
        if fallback_frame_name not in {"arm_link6", "left_arm_link6", "right_arm_link6"}:
            return pose_world
        return pose_world @ _ARM_LINK6_FROM_HAND_BASE

    def _solve_toward_targets(
        self,
        *,
        left_target: np.ndarray,
        right_target: np.ndarray,
        profile: dict[str, object] | None = None,
        frame_start_time: float | None = None,
    ) -> tuple[np.ndarray, int, dict[str, dict[str, float]]]:
        targets = [
            self._pose_to_target(left_target),
            self._pose_to_target(right_target),
        ]
        last_q = self.ik_solver.current_q.copy()
        pose_error: dict[str, dict[str, float]] = {}
        iterations = 0
        self._last_solve_iteration_budget_exhausted = False
        self._last_solve_iteration_stop_reason = None
        adaptive = bool(self.config.solve_adaptive_iterations_enabled)
        time_budget_ms = float(self.config.solve_time_budget_ms or 0.0)
        iteration_budget_ms = (
            max(0.0, time_budget_ms - float(self.config.solve_iteration_reserve_ms))
            if adaptive and time_budget_ms > 0.0
            else None
        )
        budget_start = time.perf_counter() if frame_start_time is None else float(frame_start_time)
        previous_solve_ms: float | None = None
        min_iterations = int(self.config.solve_min_iterations)
        for iteration in range(1, int(self.config.solve_max_iterations) + 1):
            if iteration > min_iterations and iteration_budget_ms is not None:
                elapsed_ms = (time.perf_counter() - budget_start) * 1000.0
                estimated_next_ms = max(0.05, float(previous_solve_ms or 0.0))
                if elapsed_ms + estimated_next_ms > iteration_budget_ms:
                    self._last_solve_iteration_budget_exhausted = True
                    self._last_solve_iteration_stop_reason = "iteration_budget_estimate"
                    break
            solve_start = time.perf_counter()
            last_q = self.ik_solver.solve(targets=targets)
            self._record_solver_profile(profile)
            measured_solve_ms = (time.perf_counter() - solve_start) * 1000.0
            solver_profile = getattr(self.ik_solver, "last_solve_profile", {})
            profiled_solve_ms = (
                float(solver_profile.get("total_ms", 0.0))
                if isinstance(solver_profile, dict)
                else 0.0
            )
            previous_solve_ms = max(
                0.05,
                profiled_solve_ms if np.isfinite(profiled_solve_ms) and profiled_solve_ms > 0.0 else measured_solve_ms,
            )
            iterations = iteration
            current_left = self.ik_solver.compute_frame_pose(
                self.profile.left_frame_name,
                last_q,
            )
            current_right = self.ik_solver.compute_frame_pose(
                self.profile.right_frame_name,
                last_q,
            )
            left_translation_m, left_rotation_deg = pose_delta_metrics(current_left, left_target)
            right_translation_m, right_rotation_deg = pose_delta_metrics(current_right, right_target)
            pose_error = {
                "left": {
                    "translation_m": float(left_translation_m),
                    "rotation_deg": float(left_rotation_deg),
                },
                "right": {
                    "translation_m": float(right_translation_m),
                    "rotation_deg": float(right_rotation_deg),
                },
            }
            if iteration >= min_iterations and (
                left_translation_m <= float(self.config.solve_translation_tolerance_m)
                and left_rotation_deg <= float(self.config.solve_orientation_tolerance_deg)
                and right_translation_m <= float(self.config.solve_translation_tolerance_m)
                and right_rotation_deg <= float(self.config.solve_orientation_tolerance_deg)
            ):
                self._last_solve_iteration_stop_reason = "pose_tolerance"
                break
            if iteration >= min_iterations and iteration_budget_ms is not None:
                elapsed_ms = (time.perf_counter() - budget_start) * 1000.0
                if elapsed_ms >= iteration_budget_ms:
                    self._last_solve_iteration_budget_exhausted = True
                    self._last_solve_iteration_stop_reason = "iteration_budget_elapsed"
                    break
        if iterations >= int(self.config.solve_max_iterations) and self._last_solve_iteration_stop_reason is None:
            self._last_solve_iteration_stop_reason = "max_iterations"
        if profile is not None:
            profile["solve_iteration_budget_exhausted"] = bool(
                self._last_solve_iteration_budget_exhausted
            )
            profile["solve_iteration_stop_reason"] = self._last_solve_iteration_stop_reason
        return last_q.copy(), iterations, pose_error

    def _fallback_reason_for_distance(self, min_distance_m: float | None) -> str | None:
        if min_distance_m is None:
            return None
        safe_distance = float(self.config.collision_min_distance_m)
        epsilon = float(self.config.collision_distance_epsilon_m)
        if self.config.contact_mode == "freeze_on_contact":
            if min_distance_m <= safe_distance + epsilon:
                return "freeze_on_contact"
            return None
        if min_distance_m < safe_distance - epsilon:
            return "collision_margin_low"
        return None

    def _embed_arm_qpos(
        self,
        *,
        current_full_q: np.ndarray,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> np.ndarray:
        full_q = np.asarray(current_full_q, dtype=float).copy()
        if self.profile.solution_layout == "dual_arm_hand" and full_q.shape[0] >= 24:
            full_q[0:6] = left_qpos
            full_q[18:24] = right_qpos
            return full_q
        if full_q.shape[0] == 14:
            full_q[1:7] = left_qpos
            full_q[8:14] = right_qpos
            return full_q
        if full_q.shape[0] == 12:
            full_q[0:6] = left_qpos
            full_q[6:12] = right_qpos
            return full_q
        raise ValueError(
            f"{self.name} cannot embed arm qpos into solver configuration shape {full_q.shape}."
        )

    def _split_solution(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float)
        if self.profile.solution_layout == "dual_arm_hand" and qpos.shape[0] >= 24:
            return qpos[0:6].copy(), qpos[18:24].copy()
        if qpos.shape[0] == 14:
            return qpos[1:7].copy(), qpos[8:14].copy()
        if qpos.shape[0] == 12:
            return qpos[0:6].copy(), qpos[6:12].copy()
        raise ValueError(f"{self.name} received unsupported solver output shape: {qpos.shape}")

    def _smooth_output_qpos_before_action(
        self,
        *,
        active_q: np.ndarray,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        target_activity: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        alpha_by_side, alpha_metadata = self._effective_output_qpos_smoothing_alpha(
            target_activity
        )
        raw_full_q = self._embed_arm_qpos(
            current_full_q=active_q,
            left_qpos=np.asarray(left_qpos, dtype=float).reshape(6),
            right_qpos=np.asarray(right_qpos, dtype=float).reshape(6),
        )
        enabled_by_side = {
            side: bool(self.config.output_command_type == "joint_vector" and alpha < 1.0)
            for side, alpha in alpha_by_side.items()
        }
        enabled = any(enabled_by_side.values())
        metadata: dict[str, object] = {
            "enabled": enabled,
            "applied": False,
            "initialized": False,
            "alpha": max(alpha_by_side.values()),
            "alpha_by_side": dict(alpha_by_side),
            "adaptive": alpha_metadata,
            "stage": "post_ik_pre_action_fk",
            "raw_qpos_source": "post_ik_step_limited_qpos",
            "final_action_realized_from_joint": True,
            "downstream_command_stage": "interpolation_only",
        }
        if self.config.output_command_type != "joint_vector":
            metadata["reason"] = "non_joint_vector_output"
            self._last_output_qpos_smoothing_full_q = None
            return raw_full_q.copy(), left_qpos.copy(), right_qpos.copy(), metadata
        if not enabled:
            metadata["reason"] = "alpha_one"
            self._last_output_qpos_smoothing_full_q = raw_full_q.copy()
            return raw_full_q.copy(), left_qpos.copy(), right_qpos.copy(), metadata

        previous = self._last_output_qpos_smoothing_full_q
        if previous is None or previous.shape != raw_full_q.shape:
            smoothed_full_q = raw_full_q.copy()
            metadata["initialized"] = True
        else:
            previous = np.asarray(previous, dtype=float).reshape(raw_full_q.shape)
            previous_left, previous_right = self._split_solution(previous)
            smoothed_left = previous_left + alpha_by_side["left"] * (
                np.asarray(left_qpos, dtype=float).reshape(6) - previous_left
            )
            smoothed_right = previous_right + alpha_by_side["right"] * (
                np.asarray(right_qpos, dtype=float).reshape(6) - previous_right
            )
            smoothed_full_q = self._embed_arm_qpos(
                current_full_q=raw_full_q,
                left_qpos=smoothed_left,
                right_qpos=smoothed_right,
            )
            metadata["applied"] = not np.allclose(smoothed_full_q, raw_full_q, atol=1e-10)
            raw_delta_deg = np.rad2deg(raw_full_q - previous)
            output_delta_deg = np.rad2deg(smoothed_full_q - previous)
            metadata["raw_max_step_deg"] = float(np.max(np.abs(raw_delta_deg))) if raw_delta_deg.size else 0.0
            metadata["output_max_step_deg"] = (
                float(np.max(np.abs(output_delta_deg))) if output_delta_deg.size else 0.0
            )
            metadata["raw_to_output_max_delta_deg"] = (
                float(np.max(np.abs(np.rad2deg(raw_full_q - smoothed_full_q))))
                if raw_full_q.size
                else 0.0
            )
        self._last_output_qpos_smoothing_full_q = smoothed_full_q.copy()
        smoothed_left, smoothed_right = self._split_solution(smoothed_full_q)
        for side, raw_qpos, output_qpos in (
            ("left", left_qpos, smoothed_left),
            ("right", right_qpos, smoothed_right),
        ):
            raw_q = np.asarray(raw_qpos, dtype=float).reshape(6)
            out_q = np.asarray(output_qpos, dtype=float).reshape(6)
            side_applied = not np.allclose(out_q, raw_q, atol=1e-10)
            metadata[side] = {
                "enabled": enabled_by_side[side],
                "applied": side_applied,
                "initialized": bool(metadata.get("initialized", False)),
                "alpha": alpha_by_side[side],
                "adaptive": dict(alpha_metadata.get(side, {})),
                "stage": "post_ik_pre_action_fk",
                "raw_target_joint_deg": np.rad2deg(raw_q).astype(float).tolist(),
                "output_target_joint_deg": np.rad2deg(out_q).astype(float).tolist(),
                "raw_to_output_delta_deg": np.rad2deg(out_q - raw_q).astype(float).tolist(),
                "raw_to_output_max_delta_deg": float(np.rad2deg(np.max(np.abs(out_q - raw_q)))),
            }
        return smoothed_full_q.copy(), smoothed_left, smoothed_right, metadata

    def _effective_output_qpos_smoothing_alpha(
        self,
        target_activity: dict[str, object] | None,
    ) -> tuple[dict[str, float], dict[str, object]]:
        slow_alpha = float(self.config.output_qpos_smoothing_alpha)
        if not bool(self.config.output_qpos_smoothing_adaptive_enabled):
            alphas = {"left": slow_alpha, "right": slow_alpha}
            return alphas, {
                "enabled": False,
                "effective_alpha": dict(alphas),
                "reason": "disabled",
            }

        now = time.monotonic()
        effective_by_side: dict[str, float] = {}
        side_results: dict[str, object] = {}
        for side in ("left", "right"):
            side_metadata = (
                dict(target_activity.get(side, {}) or {})
                if isinstance(target_activity, dict)
                else {}
            )
            evidence = side_metadata.get("evidence")
            evidence = dict(evidence) if isinstance(evidence, dict) else {}
            motion_window = side_metadata.get("motion_window")
            motion_window = (
                dict(motion_window) if isinstance(motion_window, dict) else {}
            )
            # Tracker motion evidence intentionally carries the speed estimates,
            # while the validity bit lives in the associated motion-window
            # diagnostic.  Accept the older combined shape as well.
            quality_valid = bool(
                evidence.get("quality_valid", motion_window.get("quality_valid", False))
            )
            linear = evidence.get(
                "linear_speed_p90_m_s",
                motion_window.get("linear_speed_p90_m_s"),
            )
            angular = evidence.get(
                "angular_speed_p90_rad_s",
                motion_window.get("angular_speed_p90_rad_s"),
            )
            speed_valid = quality_valid and linear is not None and angular is not None
            linear_raw = float(linear or 0.0)
            angular_raw = float(angular or 0.0)
            speed_valid = bool(
                speed_valid and np.isfinite(linear_raw) and np.isfinite(angular_raw)
            )
            linear_value = max(0.0, linear_raw) if np.isfinite(linear_raw) else 0.0
            angular_value = max(0.0, angular_raw) if np.isfinite(angular_raw) else 0.0

            def normalized(value: float, low: float, high: float) -> float:
                return float(np.clip((value - low) / (high - low), 0.0, 1.0))

            linear_ratio = normalized(
                linear_value,
                float(self.config.output_qpos_smoothing_linear_speed_low_m_s),
                float(self.config.output_qpos_smoothing_linear_speed_high_m_s),
            )
            angular_ratio = normalized(
                angular_value,
                float(self.config.output_qpos_smoothing_angular_speed_low_rad_s),
                float(self.config.output_qpos_smoothing_angular_speed_high_rad_s),
            )
            if speed_valid:
                raw_ratio = max(linear_ratio, angular_ratio)
                motion_ratio = raw_ratio * raw_ratio * (3.0 - 2.0 * raw_ratio)
                requested_alpha = slow_alpha + motion_ratio * (
                    float(self.config.output_qpos_smoothing_fast_alpha) - slow_alpha
                )
                reason = "tracker_speed"
            else:
                linear_ratio = angular_ratio = motion_ratio = 0.0
                requested_alpha = slow_alpha
                reason = "invalid_or_missing_speed"

            previous = float(self._output_qpos_smoothing_effective_alpha[side])
            previous_time = self._output_qpos_smoothing_update_time[side]
            dt = self._execution_dt_sec if previous_time is None else max(0.0, now - previous_time)
            tau = (
                float(self.config.output_qpos_smoothing_attack_sec)
                if requested_alpha >= previous
                else float(self.config.output_qpos_smoothing_release_sec)
            )
            blend = 1.0 - float(np.exp(-min(dt, 0.25) / tau))
            effective = previous + blend * (requested_alpha - previous)
            self._output_qpos_smoothing_effective_alpha[side] = effective
            self._output_qpos_smoothing_update_time[side] = now
            effective_by_side[side] = effective
            side_results[side] = {
                "reason": reason,
                "quality_valid": speed_valid,
                "linear_speed_m_s": linear_value,
                "angular_speed_rad_s": angular_value,
                "linear_ratio": linear_ratio,
                "angular_ratio": angular_ratio,
                "motion_ratio": motion_ratio,
                "requested_alpha": requested_alpha,
                "effective_alpha": effective,
                "transition": "attack" if requested_alpha >= previous else "release",
                "transition_dt_sec": dt,
                "transition_tau_sec": tau,
            }

        return effective_by_side, {
            "enabled": True,
            "slow_alpha": slow_alpha,
            "fast_alpha": float(self.config.output_qpos_smoothing_fast_alpha),
            "selection": "per_side_tracker_speed",
            "effective_alpha": dict(effective_by_side),
            **side_results,
        }

    def _joint_limit_margin_metadata(self, side: str, qpos: np.ndarray) -> dict[str, object]:
        lower, upper = self._left_arm_limits if side == "left" else self._right_arm_limits
        q = np.asarray(qpos, dtype=float).reshape(6)
        lower = np.asarray(lower, dtype=float).reshape(6)
        upper = np.asarray(upper, dtype=float).reshape(6)
        lower_margin = q - lower
        upper_margin = upper - q
        min_margin = np.minimum(lower_margin, upper_margin)
        nearest_index = int(np.argmin(min_margin))
        return {
            "lower_margin_deg": np.rad2deg(lower_margin).copy(),
            "upper_margin_deg": np.rad2deg(upper_margin).copy(),
            "min_margin_deg": np.rad2deg(min_margin).copy(),
            "nearest_joint": int(nearest_index + 1),
            "nearest_margin_deg": float(np.rad2deg(min_margin[nearest_index])),
            "near_limit": bool(min_margin[nearest_index] < np.deg2rad(5.0)),
        }

    def _joint_operating_margin_metadata(
        self,
        *,
        reference_full_q: np.ndarray,
        output_full_q: np.ndarray,
    ) -> dict[str, object]:
        """Check an optional interior envelope before collision evaluation.

        A qpos can be inside the physical URDF envelope but still too close to
        a joint stop for force control. Once a candidate enters the optional
        interior envelope, it must strictly improve every offending margin. No
        qpos is clipped here: recovery remains collision-checked through the
        normal auxiliary joint-limit-escape candidates.
        """

        required_margin_deg = float(self.config.joint_operating_margin_deg)
        if required_margin_deg <= 0.0:
            return {
                "enabled": False,
                "status": "disabled",
                "required_margin_deg": 0.0,
                "recovery_only": False,
                "reject": False,
                "violations": [],
            }
        required_margin_rad = float(np.deg2rad(required_margin_deg))
        reference_left, reference_right = self._split_solution(reference_full_q)
        output_left, output_right = self._split_solution(output_full_q)
        violations: list[dict[str, object]] = []
        tolerance_rad = 1e-9
        for side, reference_q, output_q, limits in (
            ("left", reference_left, output_left, self._left_arm_limits),
            ("right", reference_right, output_right, self._right_arm_limits),
        ):
            lower, upper = (np.asarray(value, dtype=float).reshape(6) for value in limits)
            reference_margin = np.minimum(reference_q - lower, upper - reference_q)
            output_margin = np.minimum(output_q - lower, upper - output_q)
            for local_index in np.flatnonzero(output_margin < required_margin_rad).tolist():
                before_rad = float(reference_margin[local_index])
                after_rad = float(output_margin[local_index])
                progress_rad = after_rad - before_rad
                violations.append(
                    {
                        "side": side,
                        "joint": int(local_index + 1),
                        "reference_margin_deg": float(np.rad2deg(before_rad)),
                        "output_margin_deg": float(np.rad2deg(after_rad)),
                        "margin_progress_deg": float(np.rad2deg(progress_rad)),
                        "moving_inward": bool(progress_rad > tolerance_rad),
                    }
                )
        recovery_only = bool(violations)
        reject = bool(recovery_only and any(not item["moving_inward"] for item in violations))
        return {
            "enabled": True,
            "status": "rejected" if reject else "recovery_only" if recovery_only else "pass",
            "required_margin_deg": required_margin_deg,
            "recovery_only": recovery_only,
            "reject": reject,
            "violations": violations,
        }

    def _pose_to_target(self, pose: np.ndarray) -> np.ndarray:
        quat = R.from_matrix(as_pose_matrix(pose)[:3, :3]).as_quat()
        pose_matrix = as_pose_matrix(pose)
        return np.array(
            [
                pose_matrix[0, 3],
                pose_matrix[1, 3],
                pose_matrix[2, 3],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ],
            dtype=float,
        )

    def _make_full_joint_weights(self, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        weights = np.zeros_like(q)
        if self.profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
            weights[0:6] = self._arm_joint_weights
            weights[18:24] = self._arm_joint_weights
            return weights
        if q.shape[0] == 14:
            weights[1:7] = self._arm_joint_weights
            weights[8:14] = self._arm_joint_weights
            return weights
        if q.shape[0] == 12:
            weights[0:6] = self._arm_joint_weights
            weights[6:12] = self._arm_joint_weights
            return weights
        raise ValueError(f"{self.name} cannot build weights for solver configuration shape {q.shape}.")

    def _make_full_execution_joint_step(self, qpos: np.ndarray) -> np.ndarray | None:
        if not self.config.enforce_realistic_joint_step:
            return None
        profile_step: np.ndarray | None = None
        if self.joint_velocity_profile is not None:
            dt_sec = max(1e-6, float(self._execution_dt_sec))
            profile_step = np.deg2rad(
                np.asarray(self.joint_velocity_profile.max_step_deg(dt_sec), dtype=float).reshape(6)
            )
        left_step = self.config.left_execution_max_joint_step_rad
        right_step = self.config.right_execution_max_joint_step_rad
        if profile_step is not None:
            left_effective = (
                profile_step
                if left_step is None
                else np.minimum(profile_step, np.asarray(left_step, dtype=float).reshape(6))
            )
            right_effective = (
                profile_step
                if right_step is None
                else np.minimum(profile_step, np.asarray(right_step, dtype=float).reshape(6))
            )
            return self._make_full_step_from_arm_steps(
                qpos,
                left_step=left_effective,
                right_step=right_effective,
            )
        if left_step is None or right_step is None:
            return None
        return self._make_full_step_from_arm_steps(
            qpos,
            left_step=left_step,
            right_step=right_step,
        )

    def _make_full_step_from_arm_steps(
        self,
        qpos: np.ndarray,
        *,
        left_step: np.ndarray | Sequence[float],
        right_step: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        left_step = np.asarray(left_step, dtype=float).reshape(6)
        right_step = np.asarray(right_step, dtype=float).reshape(6)
        max_step = np.zeros_like(q)
        if self.profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
            max_step[0:6] = left_step
            max_step[18:24] = right_step
            return max_step
        if q.shape[0] == 14:
            max_step[1:7] = left_step
            max_step[8:14] = right_step
            return max_step
        if q.shape[0] == 12:
            max_step[0:6] = left_step
            max_step[6:12] = right_step
            return max_step
        raise ValueError(
            f"{self.name} cannot build execution joint-step limits for solver shape {q.shape}."
        )

    def _execution_step_source(self) -> str | None:
        has_profile = self.joint_velocity_profile is not None
        has_explicit = (
            self.config.left_execution_max_joint_step_rad is not None
            or self.config.right_execution_max_joint_step_rad is not None
        )
        if has_profile and has_explicit:
            return "joint_velocity_profile+explicit_execution_max_joint_step"
        if has_profile:
            return "joint_velocity_profile"
        if has_explicit:
            return "explicit_execution_max_joint_step"
        return None

    def set_execution_max_joint_step(
        self,
        *,
        left_step_rad: Sequence[float] | np.ndarray,
        right_step_rad: Sequence[float] | np.ndarray,
    ) -> None:
        self.config.left_execution_max_joint_step_rad = _optional_joint_step(left_step_rad)
        self.config.right_execution_max_joint_step_rad = _optional_joint_step(right_step_rad)
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )

    def set_execution_dt_sec(self, dt_sec: float) -> None:
        self._execution_dt_sec = max(1e-6, float(dt_sec))
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )

    def _limit_to_execution_step(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        target_activity: dict[str, object] | None = None,
        current_targets: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        reference_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate_q = np.asarray(candidate_full_q, dtype=float).reshape(-1)
        metadata = {
            "enabled": bool(self.config.enforce_realistic_joint_step),
            "step_limited": False,
            "weighted_delta_deg": 0.0,
            "left_max_abs_delta_deg": 0.0,
            "right_max_abs_delta_deg": 0.0,
            "execution_step_source": self._execution_step_source(),
            "joint_velocity_limit_profile": (
                None if self.joint_velocity_profile is None else self.joint_velocity_profile.name
            ),
            "joint_velocity_limit_dt_sec": float(self._execution_dt_sec),
            "joint_velocity_limit_max_dt_sec": (
                None if self.joint_velocity_profile is None else float(self.joint_velocity_profile.max_dt_sec)
            ),
        }
        if not self.config.enforce_realistic_joint_step or self._full_execution_joint_step is None:
            return candidate_q.copy(), metadata
        limited_q = reference_q + np.clip(
            candidate_q - reference_q,
            -self._full_execution_joint_step,
            self._full_execution_joint_step,
        )
        inactive_capped_q, inactive_cap_metadata = self._apply_inactive_side_execution_cap(
            reference_full_q=reference_q,
            candidate_full_q=limited_q,
            target_activity=target_activity,
            current_targets=current_targets,
        )
        step_limited = bool(
            not np.allclose(limited_q, candidate_q, atol=1e-10)
            or inactive_cap_metadata.get("applied", False)
        )
        limited_q = inactive_capped_q
        left_delta, right_delta = self._split_solution(limited_q - reference_q)
        left_max_step, right_max_step = self._split_solution(self._full_execution_joint_step)
        requested_delta_deg = np.rad2deg(candidate_q - reference_q)
        max_step_deg = np.rad2deg(np.asarray(self._full_execution_joint_step, dtype=float).reshape(reference_q.shape))
        ratios = np.divide(
            np.abs(requested_delta_deg),
            max_step_deg,
            out=np.zeros_like(requested_delta_deg, dtype=float),
            where=max_step_deg > 0.0,
        )
        metadata.update(
            {
                "step_limited": bool(step_limited),
                "weighted_delta_deg": float(
                    weighted_joint_distance_score(
                        reference_q=np.zeros_like(limited_q),
                        candidate_q=limited_q - reference_q,
                        joint_distance_weights=self._full_joint_weights,
                    )
                ),
                "left_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(left_delta)))),
                "right_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(right_delta)))),
                "joint_velocity_limit_requested_delta_deg": requested_delta_deg.astype(float).tolist(),
                "joint_velocity_limit_delta_deg": np.rad2deg(limited_q - reference_q).astype(float).tolist(),
                "joint_velocity_limit_max_step_deg": max_step_deg.astype(float).tolist(),
                "joint_velocity_limit_ratio": float(np.max(ratios)) if ratios.size else 0.0,
                "left_max_step_deg": np.rad2deg(left_max_step).astype(float).tolist(),
                "right_max_step_deg": np.rad2deg(right_max_step).astype(float).tolist(),
                "inactive_side_hold_execution_cap": inactive_cap_metadata,
            }
        )
        return limited_q.copy(), metadata

    def _apply_inactive_side_execution_cap(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        target_activity: dict[str, object] | None,
        current_targets: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        reference_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        output_q = np.asarray(candidate_full_q, dtype=float).reshape(reference_q.shape).copy()
        cap_deg = max(0.0, float(self.config.inactive_side_hold_execution_cap_deg))
        enabled = bool(
            self.config.inactive_side_hold_enabled
            and self.config.inactive_side_hold_execution_cap_enabled
            and isinstance(target_activity, dict)
            and target_activity.get("enabled", False)
        )
        metadata: dict[str, object] = {
            "enabled": enabled,
            "cap_deg": cap_deg,
            "applied": False,
            "sides": {},
        }
        if not enabled:
            return output_q.copy(), metadata

        ref_left, ref_right = self._split_solution(reference_q)
        out_left, out_right = self._split_solution(output_q)
        applied_any = False
        sides: dict[str, object] = {}
        for side in ("left", "right"):
            side_activity_raw = target_activity.get(side, {})
            side_activity = dict(side_activity_raw) if isinstance(side_activity_raw, dict) else {}
            hold_weight = max(0.0, min(1.0, float(side_activity.get("hold_weight", 0.0) or 0.0)))
            inactive = bool(side_activity.get("inactive", False) or hold_weight > 0.0)
            tracker_confirmed_inactive = bool(
                side_activity.get("source") == "tracker_motion_state_machine"
                and side_activity.get("state") == "INACTIVE"
                and hold_weight >= 0.999
            )
            ref_arm = ref_left if side == "left" else ref_right
            out_arm = out_left if side == "left" else out_right
            requested_delta = out_arm - ref_arm
            requested_max_deg = float(np.rad2deg(np.max(np.abs(requested_delta))))
            release_metadata = self._tracker_inactive_execution_release(
                side=side,
                reference_full_q=reference_q,
                current_targets=current_targets,
            ) if (
                tracker_confirmed_inactive
                and self.config.inactive_side_hold_execution_release_enabled
            ) else None
            release_scale = (
                None
                if release_metadata is None
                else float(release_metadata["velocity_release_scale"])
            )
            effective_cap_deg = (
                cap_deg
                if not tracker_confirmed_inactive
                else (
                    0.0
                    if release_scale is None
                    else requested_max_deg * release_scale
                )
            )
            side_metadata = {
                "inactive": inactive,
                "hold_weight": float(hold_weight),
                "source": side_activity.get("source"),
                "state": side_activity.get("state"),
                "tracker_inactive": tracker_confirmed_inactive,
                "tracker_inactive_execution_release_enabled": bool(
                    self.config.inactive_side_hold_execution_release_enabled
                ),
                "tracker_inactive_freeze": bool(
                    tracker_confirmed_inactive
                    and (release_scale is None or release_scale <= 1e-12)
                ),
                "tracker_inactive_execution_release": release_metadata,
                "effective_cap_deg": effective_cap_deg,
                "requested_max_abs_delta_deg": requested_max_deg,
                "applied": False,
            }
            if (
                tracker_confirmed_inactive
                and release_scale is not None
                and release_scale < 1.0 - 1e-12
            ):
                capped_delta = requested_delta * release_scale
                capped_arm = ref_arm + capped_delta
                side_metadata.update(
                    {
                        "applied": True,
                        "capped_max_abs_delta_deg": float(
                            np.rad2deg(np.max(np.abs(capped_delta)))
                        ),
                    }
                )
                applied_any = True
                if side == "left":
                    out_left = capped_arm
                else:
                    out_right = capped_arm
            elif (
                inactive
                and hold_weight > 0.0
                and requested_max_deg > effective_cap_deg + 1e-9
            ):
                effective_cap_rad = np.deg2rad(effective_cap_deg)
                capped_delta = np.clip(
                    requested_delta,
                    -effective_cap_rad,
                    effective_cap_rad,
                )
                capped_arm = ref_arm + capped_delta
                side_metadata.update(
                    {
                        "applied": True,
                        "capped_max_abs_delta_deg": float(
                            np.rad2deg(np.max(np.abs(capped_delta)))
                        ),
                    }
                )
                applied_any = True
                if side == "left":
                    out_left = capped_arm
                else:
                    out_right = capped_arm
            sides[side] = side_metadata

        if applied_any:
            output_q = self._embed_arm_qpos(
                current_full_q=output_q,
                left_qpos=out_left,
                right_qpos=out_right,
            )
        metadata["applied"] = bool(applied_any)
        metadata["sides"] = sides
        return output_q.copy(), metadata

    def _tracker_inactive_execution_release(
        self,
        *,
        side: str,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray] | None,
    ) -> dict[str, object]:
        """Compute the inactive convergence release from execution-reference FK.

        ``reference_full_q`` is the qcmd/qref execution anchor selected by the
        runtime before this IK cycle.  The release is intentionally a pure
        observation of its FK versus the frozen target: it cannot loosen the
        normal joint-velocity, collision, jump-gate, or joint-limit checks that
        ran before this inactive-only cap.
        """
        metadata: dict[str, object] = {
            "available": False,
            "reference_qpos_source": "execution_reference_qcmd_or_qref",
            "translation_static_m": float(
                self.config.inactive_side_hold_execution_release_translation_static_m
            ),
            "translation_full_m": float(
                self.config.inactive_side_hold_execution_release_translation_full_m
            ),
            "rotation_static_deg": float(
                self.config.inactive_side_hold_execution_release_rotation_static_deg
            ),
            "rotation_full_deg": float(
                self.config.inactive_side_hold_execution_release_rotation_full_deg
            ),
            "translation_weight": 0.0,
            "rotation_weight": 0.0,
            "velocity_release_scale": 0.0,
        }
        if current_targets is None or side not in current_targets:
            metadata["unavailable_reason"] = "missing_current_target"
            return metadata
        frame_name = (
            self.profile.left_frame_name if side == "left" else self.profile.right_frame_name
        )
        try:
            reference_pose = self.ik_solver.compute_frame_pose(
                frame_name,
                np.asarray(reference_full_q, dtype=float).reshape(-1),
            )
            target_pose = as_pose_matrix(current_targets[side])
            translation_m, rotation_deg = pose_delta_metrics(reference_pose, target_pose)
        except Exception as exc:  # noqa: BLE001 - preserve a fail-closed inactive hold.
            metadata["unavailable_reason"] = f"fk_error:{type(exc).__name__}"
            return metadata

        translation_weight = _linear_release_weight(
            value=float(translation_m),
            static_threshold=float(
                self.config.inactive_side_hold_execution_release_translation_static_m
            ),
            full_threshold=float(
                self.config.inactive_side_hold_execution_release_translation_full_m
            ),
        )
        rotation_weight = _linear_release_weight(
            value=float(rotation_deg),
            static_threshold=float(
                self.config.inactive_side_hold_execution_release_rotation_static_deg
            ),
            full_threshold=float(
                self.config.inactive_side_hold_execution_release_rotation_full_deg
            ),
        )
        metadata.update(
            {
                "available": True,
                "reference_pose_frame": frame_name,
                "translation_error_m": float(translation_m),
                "rotation_error_deg": float(rotation_deg),
                "translation_weight": float(translation_weight),
                "rotation_weight": float(rotation_weight),
                "velocity_release_scale": float(
                    min(1.0, translation_weight + rotation_weight)
                ),
            }
        )
        return metadata

    def _full_joint_distance_score(self, reference_full_q: np.ndarray, candidate_full_q: np.ndarray) -> float:
        return weighted_joint_distance_score(
            reference_q=reference_full_q,
            candidate_q=candidate_full_q,
            joint_distance_weights=self._full_joint_weights,
        )

    def _dual_posture_penalty_deg(self, full_q: np.ndarray) -> float:
        if self.posture_preference is None:
            return 0.0
        left_q, right_q = self._split_solution(np.asarray(full_q, dtype=float).reshape(-1))
        return float(
            self.posture_preference.evaluate(left_q).score_deg
            + self.posture_preference.evaluate(right_q).score_deg
        )

    def _dual_posture_comparison_metadata(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
    ) -> dict[str, object] | None:
        if self.posture_preference is None:
            return None
        ref_left, ref_right = self._split_solution(reference_full_q)
        cand_left, cand_right = self._split_solution(candidate_full_q)
        left = self.posture_preference.compare(ref_left, cand_left)
        right = self.posture_preference.compare(ref_right, cand_right)
        return {
            "name": self.posture_preference.name,
            "left": left.metadata(),
            "right": right.metadata(),
            "score_penalty_deg": float(left.candidate.score_deg + right.candidate.score_deg),
            "improvement_deg": float(left.improvement_deg + right.improvement_deg),
            "any_reference_bad": bool(left.reference.bad or right.reference.bad),
            "any_candidate_bad": bool(left.candidate.bad or right.candidate.bad),
            "improving_sides": [
                side
                for side, comparison in (("left", left), ("right", right))
                if comparison.improvement_deg > 0.0
            ],
        }

    def _posture_hard_branch_escape_metadata(
        self,
        *,
        side: str,
        reference_q: np.ndarray,
        candidate_q: np.ndarray,
        exceeded_joint_indices: tuple[int, ...],
    ) -> dict[str, object] | None:
        if self.posture_preference is None:
            return None
        metadata = self.posture_preference.hard_branch_escape_metadata(
            reference_qpos=reference_q,
            candidate_qpos=candidate_q,
            exceeded_joint_indices=exceeded_joint_indices,
        )
        metadata["side"] = side
        return metadata

    def _branch_candidates(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float]]:
        raw_full = np.asarray(raw_full_q, dtype=float).reshape(-1)
        reference_full = np.asarray(reference_full_q, dtype=float).reshape(-1)
        raw_left, raw_right = self._split_solution(raw_full)
        ref_left, ref_right = self._split_solution(reference_full)
        left_min, left_max = self._left_arm_limits
        right_min, right_max = self._right_arm_limits
        left_alt = preferred_wrist_flip_candidate(
            raw_left,
            reference_q=ref_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        right_alt = preferred_wrist_flip_candidate(
            raw_right,
            reference_q=ref_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )

        candidates: list[tuple[str, np.ndarray]] = [("original", raw_full.copy())]
        if left_alt is not None:
            candidates.append(
                (
                    "left_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=left_alt,
                        right_qpos=raw_right,
                    ),
                )
            )
        if right_alt is not None:
            candidates.append(
                (
                    "right_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=raw_left,
                        right_qpos=right_alt,
                    ),
                )
            )
        if left_alt is not None and right_alt is not None:
            candidates.append(
                (
                    "both_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=left_alt,
                        right_qpos=right_alt,
                    ),
                )
            )

        ranked: list[tuple[str, np.ndarray, float]] = []
        seen: list[np.ndarray] = []
        for name, candidate_q in candidates:
            if any(np.allclose(candidate_q, existing, atol=1e-8) for existing in seen):
                continue
            seen.append(candidate_q.copy())
            posture_penalty = self._dual_posture_penalty_deg(candidate_q)
            ranked.append(
                (
                    name,
                    candidate_q.copy(),
                    self._full_joint_distance_score(reference_full, candidate_q) + posture_penalty,
                )
            )
        if self.config.prefer_non_elbow_singular_branch:
            ranked.sort(
                key=lambda item: (
                    int(self._position_singularity_metadata(item[1])["singular_side_count"]),
                    item[2],
                )
            )
        elif self.config.prefer_non_wrist_singular_branch:
            ranked.sort(
                key=lambda item: (
                    int(self._wrist_singularity_metadata(item[1])["singular_side_count"]),
                    item[2],
                )
            )
        else:
            ranked.sort(key=lambda item: item[2])
        return ranked

    def _wrist_singularity_metadata(self, full_qpos: np.ndarray) -> dict[str, object]:
        """Classify the RM65 wrist (J5) for branch preference and diagnostics."""

        left_q, right_q = self._split_solution(
            np.asarray(full_qpos, dtype=float).reshape(-1)
        )
        threshold_deg = float(self.config.wrist_singularity_threshold_deg)
        sides: dict[str, dict[str, object]] = {}
        for side, arm_q in (("left", left_q), ("right", right_q)):
            j5_deg = float(np.rad2deg(np.asarray(arm_q, dtype=float).reshape(6)[4]))
            singular = bool(abs(j5_deg) < threshold_deg) if threshold_deg > 0.0 else False
            sides[side] = {
                "j5_deg": j5_deg,
                "abs_j5_deg": abs(j5_deg),
                "margin_deg": abs(j5_deg) - threshold_deg,
                "is_wrist_singular": singular,
            }
        singular_sides = [
            side for side, values in sides.items() if bool(values["is_wrist_singular"])
        ]
        return {
            "threshold_deg": threshold_deg,
            "criterion": "abs(j5_deg) < threshold_deg",
            "sides": sides,
            "singular_sides": singular_sides,
            "singular_side_count": len(singular_sides),
            "any_wrist_singular": bool(singular_sides),
        }

    def _position_singularity_metadata(self, full_qpos: np.ndarray) -> dict[str, object]:
        """Classify position candidates before the RM65 SDK final gate.

        Wrist classification preserves the existing J5 proxy.  Elbow
        classification uses J3 plus a conservative selection-only margin; the
        backend SDK remains authoritative and can still reject a target that
        this lightweight candidate classifier considers normal.
        """

        left_q, right_q = self._split_solution(
            np.asarray(full_qpos, dtype=float).reshape(-1)
        )
        wrist_threshold_deg = float(self.config.wrist_singularity_threshold_deg)
        elbow_threshold_deg = float(self.config.elbow_singularity_threshold_deg)
        elbow_margin_deg = float(self.config.elbow_singularity_selection_margin_deg)
        elbow_selection_threshold_deg = elbow_threshold_deg + elbow_margin_deg
        sides: dict[str, dict[str, object]] = {}
        for side, arm_q in (("left", left_q), ("right", right_q)):
            joint_deg = np.rad2deg(np.asarray(arm_q, dtype=float).reshape(6))
            j3_deg = float(joint_deg[2])
            j5_deg = float(joint_deg[4])
            elbow_singular = bool(
                self.config.prefer_non_elbow_singular_branch
                and elbow_selection_threshold_deg > 0.0
                and abs(j3_deg) < elbow_selection_threshold_deg
            )
            wrist_singular = bool(
                self.config.prefer_non_wrist_singular_branch
                and wrist_threshold_deg > 0.0
                and abs(j5_deg) < wrist_threshold_deg
            )
            sides[side] = {
                "j3_deg": j3_deg,
                "abs_j3_deg": abs(j3_deg),
                "elbow_margin_deg": abs(j3_deg) - elbow_selection_threshold_deg,
                "is_elbow_singular": elbow_singular,
                "j5_deg": j5_deg,
                "abs_j5_deg": abs(j5_deg),
                "wrist_margin_deg": abs(j5_deg) - wrist_threshold_deg,
                "is_wrist_singular": wrist_singular,
                "is_singular": bool(elbow_singular or wrist_singular),
            }
        elbow_sides = [
            side for side, values in sides.items() if bool(values["is_elbow_singular"])
        ]
        wrist_sides = [
            side for side, values in sides.items() if bool(values["is_wrist_singular"])
        ]
        singular_sides = [
            side for side, values in sides.items() if bool(values["is_singular"])
        ]
        return {
            "enabled": bool(
                self.config.prefer_non_elbow_singular_branch
                or self.config.prefer_non_wrist_singular_branch
            ),
            "criterion": "J3 elbow proxy or J5 wrist proxy",
            "elbow_threshold_deg": elbow_threshold_deg,
            "elbow_selection_margin_deg": elbow_margin_deg,
            "elbow_selection_threshold_deg": elbow_selection_threshold_deg,
            "wrist_threshold_deg": wrist_threshold_deg,
            "sides": sides,
            "elbow_singular_sides": elbow_sides,
            "elbow_singular_side_count": len(elbow_sides),
            "any_elbow_singular": bool(elbow_sides),
            "wrist_singular_sides": wrist_sides,
            "wrist_singular_side_count": len(wrist_sides),
            "any_wrist_singular": bool(wrist_sides),
            "singular_sides": singular_sides,
            "singular_side_count": len(singular_sides),
            "any_singular": bool(singular_sides),
        }

    def _position_singularity_selection_metadata(
        self,
        evaluated: Sequence[_DualArmEvaluatedCandidate],
        *,
        selected: _DualArmEvaluatedCandidate | None = None,
    ) -> dict[str, object]:
        classifications = [
            self._position_singularity_metadata(item.output_qpos) for item in evaluated
        ]
        counts = [int(item["singular_side_count"]) for item in classifications]
        selected_classification = (
            None
            if selected is None
            else self._position_singularity_metadata(selected.output_qpos)
        )
        fully_non_singular_count = sum(count == 0 for count in counts)
        return {
            "enabled": bool(
                self.config.prefer_non_elbow_singular_branch
                or self.config.prefer_non_wrist_singular_branch
            ),
            "prefer_non_elbow_singular_branch": bool(
                self.config.prefer_non_elbow_singular_branch
            ),
            "prefer_non_wrist_singular_branch": bool(
                self.config.prefer_non_wrist_singular_branch
            ),
            "evaluated_feasible_count": len(evaluated),
            "fully_non_singular_count": fully_non_singular_count,
            "minimum_singular_side_count": min(counts) if counts else None,
            "fallback_to_singular": bool(
                selected_classification is not None
                and bool(selected_classification["any_singular"])
                and fully_non_singular_count == 0
            ),
            "selected": selected_classification,
            "candidates": [
                {
                    "name": item.candidate.name,
                    "source": item.candidate.source,
                    "method": item.candidate.method,
                    "score": float(item.score),
                    **classification,
                }
                for item, classification in zip(evaluated, classifications, strict=True)
            ],
        }

    def _wrist_singularity_selection_metadata(
        self,
        evaluated: Sequence[_DualArmEvaluatedCandidate],
        *,
        selected: _DualArmEvaluatedCandidate | None = None,
    ) -> dict[str, object]:
        classifications = [
            self._wrist_singularity_metadata(item.output_qpos) for item in evaluated
        ]
        counts = [int(item["singular_side_count"]) for item in classifications]
        minimum = min(counts) if counts else None
        selected_classification = (
            None
            if selected is None
            else self._wrist_singularity_metadata(selected.output_qpos)
        )
        return {
            "enabled": bool(self.config.prefer_non_wrist_singular_branch),
            "threshold_deg": float(self.config.wrist_singularity_threshold_deg),
            "evaluated_feasible_count": len(evaluated),
            "fully_non_singular_count": sum(count == 0 for count in counts),
            "minimum_singular_side_count": minimum,
            "fallback_to_singular": bool(
                self.config.prefer_non_wrist_singular_branch
                and selected_classification is not None
                and bool(selected_classification["any_wrist_singular"])
            ),
            "selected": selected_classification,
            "candidates": [
                {
                    "name": item.candidate.name,
                    "source": item.candidate.source,
                    "method": item.candidate.method,
                    "score": float(item.score),
                    **classification,
                }
                for item, classification in zip(evaluated, classifications, strict=True)
            ],
        }

    def _select_normalized_candidate(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        profile: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, str, float, float | None]:
        ranked = self._branch_candidates(
            raw_full_q=raw_full_q,
            reference_full_q=reference_full_q,
        )
        candidate_name, candidate_q, score_deg = ranked[0]
        candidate_min_distance: float | None = None
        if (
            candidate_name != "original"
            and self.ik_tuning.dual_arm_recovery.check_collision_on_branch_switch
        ):
            candidate_min_distance = self._profiled_min_collision_distance(
                candidate_q,
                profile,
                phase="branch_switch",
            )
            if self._fallback_reason_for_distance(candidate_min_distance) is not None:
                for fallback_name, fallback_q, fallback_score in ranked:
                    if fallback_name == "original":
                        return fallback_q.copy(), fallback_name, float(fallback_score), None
        return candidate_q.copy(), candidate_name, float(score_deg), candidate_min_distance

    def _select_safe_tracking_candidate(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
        target_activity: dict[str, object] | None = None,
        profile: dict[str, object] | None = None,
        elapsed_before_selection_ms: float | None = None,
    ) -> tuple[np.ndarray, str, float, float | None, dict[str, object]]:
        safety_profile = self.ik_safety_profile
        if (
            safety_profile is None
            or not bool(safety_profile.local_step_enabled)
            or not bool(self.config.enforce_realistic_joint_step)
            or not bool(self.config.local_step_candidate_selection_enabled)
        ):
            q, name, score, min_distance = self._select_normalized_candidate(
                raw_full_q=raw_full_q,
                reference_full_q=reference_full_q,
                profile=profile,
            )
            selected_position_singularity = self._position_singularity_metadata(q)
            return q, name, score, min_distance, {
                "enabled": False,
                "reason": (
                    "execution_step_limit_disabled"
                    if not bool(self.config.enforce_realistic_joint_step)
                    else "local_step_candidate_selection_disabled"
                    if not bool(self.config.local_step_candidate_selection_enabled)
                    else "ik_safety_profile_local_step_disabled"
                ),
                "selected": name,
                "position_singularity_preference": {
                    "enabled": bool(selected_position_singularity["enabled"]),
                    "prefer_non_elbow_singular_branch": bool(
                        self.config.prefer_non_elbow_singular_branch
                    ),
                    "prefer_non_wrist_singular_branch": bool(
                        self.config.prefer_non_wrist_singular_branch
                    ),
                    "evaluated_feasible_count": 1,
                    "fully_non_singular_count": int(
                        not bool(selected_position_singularity["any_singular"])
                    ),
                    "minimum_singular_side_count": int(
                        selected_position_singularity["singular_side_count"]
                    ),
                    "fallback_to_singular": bool(
                        selected_position_singularity["enabled"]
                        and selected_position_singularity["any_singular"]
                    ),
                    "selected": selected_position_singularity,
                },
                "wrist_singularity_preference": {
                    "enabled": bool(self.config.prefer_non_wrist_singular_branch),
                    "threshold_deg": float(self.config.wrist_singularity_threshold_deg),
                    "fallback_to_singular": bool(
                        self.config.prefer_non_wrist_singular_branch
                        and self._wrist_singularity_metadata(q)["any_wrist_singular"]
                    ),
                    "selected": self._wrist_singularity_metadata(q),
                },
            }

        reference_full = np.asarray(reference_full_q, dtype=float).reshape(-1)
        target_activity_metadata = (
            self._target_activity_metadata(current_targets)
            if target_activity is None
            else dict(target_activity)
        )
        inactive_hold_active = bool(
            target_activity_metadata.get("enabled", False)
            and any(
                float(
                    dict(target_activity_metadata.get(side, {}) or {}).get(
                        "hold_weight", 0.0
                    )
                    or 0.0
                )
                > 0.0
                for side in ("left", "right")
            )
        )
        selection_start = time.perf_counter()
        time_budget_ms = float(self.config.solve_time_budget_ms or 0.0)
        pre_selection_elapsed_ms = (
            max(0.0, float(elapsed_before_selection_ms))
            if elapsed_before_selection_ms is not None
            else float(profile.get("solve_total_ms", 0.0))
            if isinstance(profile, dict)
            else 0.0
        )
        deadline_enabled = bool(time_budget_ms > 0.0)
        branch_budget_ms = (
            max(0.0, time_budget_ms - pre_selection_elapsed_ms - 5.0)
            if deadline_enabled
            else 0.0
        )

        evaluated: list[_DualArmEvaluatedCandidate] = []
        collision_safe_evaluated: list[_DualArmEvaluatedCandidate] = []
        rejections: list[dict[str, object]] = []
        collision_rejections: list[dict[str, object]] = []
        candidate_eval_stopped_early = False
        candidate_eval_stop_reason: str | None = None
        stage_summaries: list[dict[str, object]] = []
        reference_min_distance: float | None = None

        def deadline_reached() -> bool:
            if not deadline_enabled:
                return False
            elapsed_ms = (time.perf_counter() - selection_start) * 1000.0
            return elapsed_ms >= branch_budget_ms

        def stop_for_deadline(reason: str) -> None:
            nonlocal candidate_eval_stopped_early, candidate_eval_stop_reason
            candidate_eval_stopped_early = True
            if candidate_eval_stop_reason is None:
                candidate_eval_stop_reason = reason

        # A deadline fallback must never launch a new collision query after the
        # candidate budget has elapsed. Certify the current reference up front,
        # then reuse this value if no budgeted candidate is safe.
        if deadline_enabled:
            reference_min_distance = self._profiled_min_collision_distance(
                reference_full,
                profile,
                phase="candidate_selection_reference",
            )
            if deadline_reached():
                stop_for_deadline("branch_budget_exhausted_before_candidates")

        base_candidates, limit_escape_metadata = self._dual_arm_tracking_candidates(
            raw_full_q=raw_full_q,
            reference_full_q=reference_full,
            current_targets=current_targets,
            include_limit_escape=False,
            include_posture_escape=False,
            include_wrist_unload=False,
        )
        scale_factors = _dual_arm_step_scale_factors(safety_profile)

        def evaluate_stage(
            *,
            stage: str,
            candidates: list[_DualArmCandidate],
            scales: tuple[float, ...],
            stop_after_first_accept_per_candidate: bool = True,
        ) -> None:
            nonlocal candidate_eval_stopped_early
            accepted_before = len(evaluated)
            collision_safe_before = len(collision_safe_evaluated)
            rejected_before = len(rejections)
            attempts = 0
            for candidate in candidates:
                if deadline_reached():
                    stop_for_deadline("branch_budget_exhausted_before_candidate")
                    break
                for scale in scales:
                    if deadline_reached():
                        stop_for_deadline("branch_budget_exhausted_before_scale")
                        break
                    attempts += 1
                    accepted, output_qpos, metadata = self._evaluate_dual_arm_step_candidate(
                        candidate=candidate,
                        reference_full_q=reference_full,
                        current_targets=current_targets,
                        target_activity=target_activity_metadata,
                        step_scale=scale,
                    )
                    metadata["candidate_eval_stage"] = stage
                    if accepted:
                        # Do not start a collision query for a candidate that
                        # completed only after its budget had expired.
                        if deadline_reached():
                            metadata["candidate_eval_discarded_due_deadline"] = True
                            stop_for_deadline("branch_budget_exhausted_after_candidate")
                            break
                        min_distance = self._profiled_min_collision_distance(
                            output_qpos,
                            profile,
                            phase="candidate_selection",
                        )
                        metadata["collision_distance_m"] = min_distance
                        metadata["wrist_singularity"] = self._wrist_singularity_metadata(
                            output_qpos
                        )
                        metadata["position_singularity"] = (
                            self._position_singularity_metadata(output_qpos)
                        )
                        fallback_reason = self._fallback_reason_for_distance(min_distance)
                        evaluated.append(
                            _DualArmEvaluatedCandidate(
                                candidate=candidate,
                                output_qpos=output_qpos,
                                score=float(metadata.get("score", 1e18)),
                                metadata=metadata,
                            )
                        )
                        if fallback_reason is None:
                            collision_safe_evaluated.append(evaluated[-1])
                        else:
                            collision_rejections.append(
                                {
                                    "candidate": candidate.name,
                                    "method": candidate.method,
                                    "scale": metadata.get("step_scale"),
                                    "collision_distance_m": min_distance,
                                    "reject_reason": fallback_reason,
                                    "candidate_eval_stage": stage,
                                }
                            )
                        if deadline_reached():
                            stop_for_deadline("branch_budget_exhausted_after_collision_check")
                            break
                        if stop_after_first_accept_per_candidate:
                            break
                    else:
                        rejections.append(metadata)
                if candidate_eval_stopped_early:
                    break
            stage_summaries.append(
                {
                    "stage": stage,
                    "candidate_count": len(candidates),
                    "scale_factors": list(scales),
                    "attempt_count": attempts,
                    "accepted_count": len(evaluated) - accepted_before,
                    "collision_safe_count": len(collision_safe_evaluated) - collision_safe_before,
                    "rejected_count": len(rejections) - rejected_before,
                    "stopped_early": bool(candidate_eval_stopped_early),
                }
            )

        def selection_metadata(
            *,
            candidates: list[_DualArmCandidate],
            auxiliary_metadata: dict[str, object] | None = None,
        ) -> dict[str, object]:
            combined = dict(limit_escape_metadata)
            if auxiliary_metadata:
                combined.update(auxiliary_metadata)
            combined.setdefault("posture_escape", {"enabled": False, "status": "deferred"})
            combined.setdefault("wrist_unload", {"enabled": False, "status": "deferred"})
            elapsed_ms = (time.perf_counter() - selection_start) * 1000.0
            return {
                "enabled": True,
                "profile": safety_profile.name,
                "joint_velocity_limit_profile": (
                    None if self.joint_velocity_profile is None else self.joint_velocity_profile.name
                ),
                "execution_dt_sec": float(self._execution_dt_sec),
                "candidate_count": len(candidates),
                "evaluated_count": len(evaluated),
                "collision_safe_count": len(collision_safe_evaluated),
                "collision_checked_count": len(evaluated),
                "rejection_count": len(rejections),
                "last_rejections": rejections[-8:],
                "collision_rejection_count": len(collision_rejections),
                "collision_rejections": collision_rejections[-8:],
                "scale_factors": list(scale_factors),
                "candidate_eval_stages": list(stage_summaries),
                "candidate_eval_elapsed_ms": float(elapsed_ms),
                "candidate_eval_pre_selection_elapsed_ms": float(pre_selection_elapsed_ms),
                "candidate_eval_branch_budget_ms": (
                    float(branch_budget_ms) if deadline_enabled else None
                ),
                "candidate_eval_stopped_early": bool(candidate_eval_stopped_early),
                "candidate_eval_stop_reason": candidate_eval_stop_reason,
                "reference_collision_distance_m": reference_min_distance,
                "target_activity": dict(target_activity_metadata),
                "inactive_side_hold_active": bool(inactive_hold_active),
                "limit_escape": dict(combined),
                "posture_escape": dict(combined.get("posture_escape", {})),
                "wrist_unload": dict(combined.get("wrist_unload", {})),
            }

        def choose_from_evaluated(
            *,
            metadata: dict[str, object],
        ) -> tuple[np.ndarray, str, float, float | None, dict[str, object]] | None:
            if not collision_safe_evaluated:
                return None
            wrist_pool = list(collision_safe_evaluated)
            if (
                self.config.prefer_non_elbow_singular_branch
                or self.config.prefer_non_wrist_singular_branch
            ):
                singular_counts = [
                    int(
                        dict(item.metadata.get("position_singularity", {})).get(
                            "singular_side_count", 2
                        )
                    )
                    for item in wrist_pool
                ]
                minimum_singular_sides = min(singular_counts)
                wrist_pool = [
                    item
                    for item, count in zip(
                        wrist_pool, singular_counts, strict=True
                    )
                    if count == minimum_singular_sides
                ]
            selection_pool, selection_policy = self._dual_stable_selection_pool(wrist_pool)
            metadata["selection_policy"] = selection_policy
            selection_pool = list(selection_pool)
            selection_pool.sort(key=lambda item: item.score)
            item = selection_pool[0]
            metadata["wrist_singularity_preference"] = (
                self._wrist_singularity_selection_metadata(
                    collision_safe_evaluated,
                    selected=item,
                )
            )
            metadata["position_singularity_preference"] = (
                self._position_singularity_selection_metadata(
                    collision_safe_evaluated,
                    selected=item,
                )
            )
            selected_metadata = dict(item.metadata)
            min_distance = selected_metadata.get("collision_distance_m")
            metadata.update(
                {
                    "collision_rejection_count": len(collision_rejections),
                    "collision_rejections": collision_rejections[-8:],
                    "status": "selected",
                    "selected": item.candidate.name,
                    "selected_source": item.candidate.source,
                    "selected_method": item.candidate.method,
                    "selected_score": float(item.score),
                    "selected_step_scale": selected_metadata.get("step_scale"),
                    "collision_distance_m": min_distance,
                    "selected_metadata": selected_metadata,
                }
            )
            return (
                item.output_qpos.copy(),
                item.candidate.name,
                float(item.score),
                min_distance,
                metadata,
            )

        primary_scales = scale_factors if inactive_hold_active else scale_factors[:1]
        fallback_scales = () if inactive_hold_active else scale_factors[1:]
        evaluate_stage(
            stage="base_primary",
            candidates=base_candidates,
            scales=primary_scales,
            stop_after_first_accept_per_candidate=not inactive_hold_active,
        )
        if not evaluated and fallback_scales and not candidate_eval_stopped_early:
            evaluate_stage(stage="base_fallback", candidates=base_candidates, scales=fallback_scales)

        base_metadata = selection_metadata(candidates=base_candidates)
        base_selection = choose_from_evaluated(metadata=base_metadata)
        base_selected_singularity = (
            {}
            if base_selection is None
            else dict(
                dict(
                    base_selection[4].get("position_singularity_preference", {})
                ).get("selected", {})
                or {}
            )
        )
        expand_for_elbow_singularity = bool(
            base_selection is not None
            and self.config.prefer_non_elbow_singular_branch
            and base_selected_singularity.get("any_elbow_singular", False)
            and not candidate_eval_stopped_early
        )
        if base_selection is not None and not expand_for_elbow_singularity:
            base_selection[4]["candidate_eval_mode"] = (
                "deadline_partial" if candidate_eval_stopped_early else "base_only"
            )
            return base_selection

        if candidate_eval_stopped_early:
            base_metadata.update(
                {
                    "status": "hold_reference",
                    "selected": "local_step_hold",
                    "reason": "branch_budget_exhausted_no_safe_candidate",
                    "collision_distance_m": reference_min_distance,
                    "candidate_eval_mode": "deadline_hold",
                }
            )
            return reference_full.copy(), "local_step_hold", 0.0, reference_min_distance, base_metadata

        auxiliary_candidates, auxiliary_metadata = self._dual_arm_tracking_candidates(
            raw_full_q=raw_full_q,
            reference_full_q=reference_full,
            current_targets=current_targets,
            include_base=False,
            include_limit_escape=True,
            include_posture_escape=not candidate_eval_stopped_early,
            include_wrist_unload=not candidate_eval_stopped_early,
        )
        all_candidates = list(base_candidates) + list(auxiliary_candidates)
        if auxiliary_candidates and not candidate_eval_stopped_early:
            evaluated_before_auxiliary = len(evaluated)
            evaluate_stage(
                stage="auxiliary_primary",
                candidates=auxiliary_candidates,
                scales=primary_scales,
                stop_after_first_accept_per_candidate=not inactive_hold_active,
            )
            if (
                len(evaluated) == evaluated_before_auxiliary
                and fallback_scales
                and not candidate_eval_stopped_early
            ):
                evaluate_stage(
                    stage="auxiliary_fallback",
                    candidates=auxiliary_candidates,
                    scales=fallback_scales,
                )

        metadata = selection_metadata(
            candidates=all_candidates,
            auxiliary_metadata=auxiliary_metadata,
        )
        auxiliary_selection = choose_from_evaluated(metadata=metadata)
        if auxiliary_selection is not None:
            if candidate_eval_stopped_early:
                candidate_eval_mode = "deadline_partial"
            elif expand_for_elbow_singularity:
                candidate_eval_mode = "base_then_auxiliary_singularity_search"
            else:
                candidate_eval_mode = "base_then_auxiliary"
            auxiliary_selection[4]["candidate_eval_mode"] = candidate_eval_mode
            auxiliary_selection[4]["expanded_for_elbow_singularity"] = bool(
                expand_for_elbow_singularity
            )
            return auxiliary_selection

        if candidate_eval_stopped_early:
            metadata.update(
                {
                    "status": "hold_reference",
                    "selected": "local_step_hold",
                    "reason": "branch_budget_exhausted_no_safe_candidate",
                    "collision_distance_m": reference_min_distance,
                    "candidate_eval_mode": "deadline_hold",
                }
            )
            return reference_full.copy(), "local_step_hold", 0.0, reference_min_distance, metadata

        if not evaluated:
            hold_q = reference_full.copy()
            min_distance = (
                reference_min_distance
                if reference_min_distance is not None
                else self._profiled_min_collision_distance(
                    hold_q,
                    profile,
                    phase="candidate_selection_hold",
                )
            )
            metadata.update(
                {
                    "status": "hold_reference",
                    "selected": "local_step_hold",
                    "reason": "no_progress_candidate",
                    "collision_distance_m": min_distance,
                }
            )
            return hold_q, "local_step_hold", 0.0, min_distance, metadata

        hold_q = reference_full.copy()
        min_distance = (
            reference_min_distance
            if reference_min_distance is not None
            else self._profiled_min_collision_distance(
                hold_q,
                profile,
                phase="candidate_selection_collision_hold",
            )
        )
        metadata.update(
            {
                "status": "hold_reference",
                "selected": "local_step_hold",
                "reason": "candidate_collision_rejected",
                "collision_distance_m": min_distance,
                "collision_rejection_count": len(metadata.get("collision_rejections", [])),
                "collision_rejections": metadata.get("collision_rejections", [])[-8:],
            }
        )
        return hold_q, "local_step_hold", 0.0, min_distance, metadata

    def _dual_arm_tracking_candidates(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
        include_base: bool = True,
        include_limit_escape: bool = True,
        include_posture_escape: bool = True,
        include_wrist_unload: bool = True,
    ) -> tuple[list[_DualArmCandidate], dict[str, object]]:
        candidates: list[_DualArmCandidate] = []
        seen: list[np.ndarray] = []
        if include_base:
            for name, qpos, score_deg in self._branch_candidates(
                raw_full_q=raw_full_q,
                reference_full_q=reference_full_q,
            ):
                posture_metadata = self._dual_posture_comparison_metadata(
                    reference_full_q=reference_full_q,
                    candidate_full_q=qpos,
                )
                candidates.append(
                    _DualArmCandidate(
                        name=name,
                        qpos=qpos.copy(),
                        source="multitask",
                        method=name,
                        raw_score_deg=float(score_deg),
                        metadata=(
                            {}
                            if posture_metadata is None
                            else {"posture_preference": posture_metadata}
                        ),
                    )
                )
                seen.append(qpos.copy())

        if include_limit_escape:
            limit_candidates, limit_metadata = self._dual_arm_limit_escape_candidates(
                reference_full_q=reference_full_q,
                current_targets=current_targets,
            )
        else:
            limit_candidates = []
            limit_metadata = {"enabled": False, "status": "deferred", "candidate_count": 0}
        for candidate in limit_candidates:
            if any(np.allclose(candidate.qpos, existing, atol=1e-9) for existing in seen):
                continue
            candidates.append(candidate)
            seen.append(candidate.qpos.copy())

        if include_posture_escape:
            posture_candidates, posture_metadata = self._dual_arm_posture_escape_candidates(
                reference_full_q=reference_full_q,
            )
        else:
            posture_candidates = []
            posture_metadata = {"enabled": False, "status": "deferred", "candidate_count": 0}
        for candidate in posture_candidates:
            if any(np.allclose(candidate.qpos, existing, atol=1e-9) for existing in seen):
                continue
            candidates.append(candidate)
            seen.append(candidate.qpos.copy())

        if include_wrist_unload:
            wrist_candidates, wrist_metadata = self._dual_arm_wrist_unload_candidates(
                reference_full_q=reference_full_q,
            )
        else:
            wrist_candidates = []
            wrist_metadata = {"enabled": False, "status": "deferred", "candidate_count": 0}
        for candidate in wrist_candidates:
            if any(np.allclose(candidate.qpos, existing, atol=1e-9) for existing in seen):
                continue
            candidates.append(candidate)
            seen.append(candidate.qpos.copy())

        combined_metadata = dict(limit_metadata)
        combined_metadata["posture_escape"] = dict(posture_metadata)
        combined_metadata["wrist_unload"] = dict(wrist_metadata)
        return candidates, combined_metadata

    def _dual_arm_posture_escape_candidates(
        self,
        *,
        reference_full_q: np.ndarray,
    ) -> tuple[list[_DualArmCandidate], dict[str, object]]:
        metadata: dict[str, object] = {
            "enabled": bool(
                self.posture_preference is not None
                and self.posture_preference.enabled
                and self.posture_preference.local_escape_enabled
            ),
            "candidate_count": 0,
        }
        if self.posture_preference is None or not self.posture_preference.local_escape_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        if self._full_execution_joint_step is None:
            metadata["status"] = "joint_step_unavailable"
            return [], metadata

        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        left_q, right_q = self._split_solution(q)
        left_step, right_step = self._split_solution(self._full_execution_joint_step)
        left_targets, left_metadata = self.posture_preference.local_escape_targets(
            reference_qpos=left_q,
            max_step_rad=left_step,
            q_min=self._left_arm_limits[0],
            q_max=self._left_arm_limits[1],
        )
        right_targets, right_metadata = self.posture_preference.local_escape_targets(
            reference_qpos=right_q,
            max_step_rad=right_step,
            q_min=self._right_arm_limits[0],
            q_max=self._right_arm_limits[1],
        )
        candidates: list[_DualArmCandidate] = []
        seen: list[np.ndarray] = []
        max_candidates = max(0, int(self.posture_preference.local_escape_max_candidates))
        for side, targets in (("left", left_targets), ("right", right_targets)):
            for label, arm_q, arm_metadata in targets:
                if len(candidates) >= max_candidates:
                    break
                candidate_q = self._embed_arm_qpos(
                    current_full_q=q,
                    left_qpos=arm_q if side == "left" else left_q,
                    right_qpos=arm_q if side == "right" else right_q,
                )
                if any(np.allclose(candidate_q, existing, atol=1e-9) for existing in seen):
                    continue
                seen.append(candidate_q.copy())
                posture_metadata = self._dual_posture_comparison_metadata(
                    reference_full_q=q,
                    candidate_full_q=candidate_q,
                )
                candidates.append(
                    _DualArmCandidate(
                        name=f"posture_escape_{side}_{label}",
                        qpos=candidate_q.copy(),
                        source="posture_escape",
                        method=f"posture_escape_{side}_{label}",
                        metadata={
                            "local_step_posture_escape_candidate": True,
                            "local_step_posture_escape_side": side,
                            "local_step_posture_escape_label": label,
                            "local_step_posture_escape_arm_metadata": dict(arm_metadata),
                            **(
                                {}
                                if posture_metadata is None
                                else {"posture_preference": posture_metadata}
                            ),
                        },
                    )
                )
            if len(candidates) >= max_candidates:
                break
        metadata.update(
            {
                "status": "active" if candidates else "no_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "left": dict(left_metadata),
                "right": dict(right_metadata),
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

    def _dual_arm_wrist_unload_candidates(
        self,
        *,
        reference_full_q: np.ndarray,
    ) -> tuple[list[_DualArmCandidate], dict[str, object]]:
        metadata: dict[str, object] = {
            "enabled": bool(
                self.posture_preference is not None
                and self.posture_preference.enabled
                and self.posture_preference.wrist_unload_enabled
            ),
            "candidate_count": 0,
        }
        if self.posture_preference is None or not self.posture_preference.wrist_unload_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        if self._full_execution_joint_step is None:
            metadata["status"] = "joint_step_unavailable"
            return [], metadata

        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        left_q, right_q = self._split_solution(q)
        left_step, right_step = self._split_solution(self._full_execution_joint_step)
        left_targets, left_metadata = self.posture_preference.wrist_unload_targets(
            reference_qpos=left_q,
            max_step_rad=left_step,
            q_min=self._left_arm_limits[0],
            q_max=self._left_arm_limits[1],
        )
        right_targets, right_metadata = self.posture_preference.wrist_unload_targets(
            reference_qpos=right_q,
            max_step_rad=right_step,
            q_min=self._right_arm_limits[0],
            q_max=self._right_arm_limits[1],
        )
        candidates: list[_DualArmCandidate] = []
        seen: list[np.ndarray] = []
        max_candidates = max(0, int(self.posture_preference.wrist_unload_max_candidates))
        for side, targets in (("left", left_targets), ("right", right_targets)):
            for label, arm_q, arm_metadata in targets:
                if len(candidates) >= max_candidates:
                    break
                candidate_q = self._embed_arm_qpos(
                    current_full_q=q,
                    left_qpos=arm_q if side == "left" else left_q,
                    right_qpos=arm_q if side == "right" else right_q,
                )
                if any(np.allclose(candidate_q, existing, atol=1e-9) for existing in seen):
                    continue
                seen.append(candidate_q.copy())
                posture_metadata = self._dual_posture_comparison_metadata(
                    reference_full_q=q,
                    candidate_full_q=candidate_q,
                )
                candidates.append(
                    _DualArmCandidate(
                        name=f"wrist_unload_{side}_{label}",
                        qpos=candidate_q.copy(),
                        source="wrist_unload",
                        method=f"wrist_unload_{side}_{label}",
                        metadata={
                            "local_step_wrist_unload_candidate": True,
                            "local_step_wrist_unload_side": side,
                            "local_step_wrist_unload_label": label,
                            "local_step_wrist_unload_arm_metadata": dict(arm_metadata),
                            "local_step_wrist_unload_cost_improvement_deg": float(
                                arm_metadata.get("local_step_wrist_unload_cost_improvement_deg", 0.0) or 0.0
                            ),
                            **(
                                {}
                                if posture_metadata is None
                                else {"posture_preference": posture_metadata}
                            ),
                        },
                    )
                )
            if len(candidates) >= max_candidates:
                break
        metadata.update(
            {
                "status": "active" if candidates else "no_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "left": dict(left_metadata),
                "right": dict(right_metadata),
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

    def _dual_arm_limit_escape_candidates(
        self,
        *,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> tuple[list[_DualArmCandidate], dict[str, object]]:
        safety_profile = self.ik_safety_profile
        metadata: dict[str, object] = {
            "enabled": bool(
                safety_profile is not None
                and getattr(safety_profile, "local_step_limit_escape_enabled", False)
            ),
            "candidate_count": 0,
        }
        if safety_profile is None or not bool(getattr(safety_profile, "local_step_limit_escape_enabled", False)):
            metadata["status"] = "disabled"
            return [], metadata
        if self._full_execution_joint_step is None:
            metadata["status"] = "joint_step_unavailable"
            return [], metadata

        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        q_min = np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float).reshape(q.shape)
        q_max = np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float).reshape(q.shape)
        if np.any(q_max <= q_min):
            metadata["status"] = "invalid_joint_position_limits"
            return [], metadata

        arm_entries = self._arm_joint_entries(q)
        arm_indices = np.array([entry[2] for entry in arm_entries], dtype=int)
        lower_margin = q - q_min
        upper_margin = q_max - q
        min_margin = np.minimum(lower_margin, upper_margin)
        activation_margin_rad = np.deg2rad(
            max(0.0, float(getattr(safety_profile, "local_step_limit_escape_activation_margin_deg", 2.0)))
        )
        active_full_indices = [
            int(index)
            for index in arm_indices.tolist()
            if min_margin[int(index)] <= activation_margin_rad
        ]
        metadata.update(
            {
                "status": "inactive" if not active_full_indices else "active",
                "activation_margin_deg": float(np.rad2deg(activation_margin_rad)),
                "active_full_joint_indices": active_full_indices,
                "active_joints": [
                    f"{side}_j{local_index + 1}"
                    for side, local_index, full_index in arm_entries
                    if int(full_index) in set(active_full_indices)
                ],
                "min_margin_deg": np.rad2deg(min_margin[arm_indices]).astype(float).tolist(),
            }
        )
        if not active_full_indices:
            return [], metadata

        max_candidates = max(0, int(getattr(safety_profile, "local_step_limit_escape_max_candidates", 12)))
        if max_candidates <= 0:
            metadata["status"] = "max_candidates_zero"
            return [], metadata
        step_fraction = max(
            0.0,
            min(1.0, float(getattr(safety_profile, "local_step_limit_escape_step_fraction", 1.0))),
        )
        if step_fraction <= 0.0:
            metadata["status"] = "step_fraction_zero"
            return [], metadata

        before_metrics = self._dual_pose_metrics(q, current_targets)
        before_score = self._dual_pose_score(before_metrics)
        full_step = np.asarray(self._full_execution_joint_step, dtype=float).reshape(q.shape) * step_fraction
        active_set = set(active_full_indices)
        scored: list[tuple[float, float, int, int, str, int, np.ndarray, dict[str, object]]] = []
        eps = 1e-9
        for side, local_index, full_index in arm_entries:
            for direction in (-1, 1):
                if direction < 0 and q[full_index] <= q_min[full_index] + eps:
                    continue
                if direction > 0 and q[full_index] >= q_max[full_index] - eps:
                    continue
                if full_index in active_set:
                    near_upper = upper_margin[full_index] <= lower_margin[full_index]
                    if near_upper and direction > 0:
                        continue
                    if (not near_upper) and direction < 0:
                        continue
                step = min(
                    float(full_step[full_index]),
                    float(upper_margin[full_index] if direction > 0 else lower_margin[full_index]),
                )
                if step <= 1e-12:
                    continue
                candidate_q = q.copy()
                candidate_q[full_index] += float(direction) * step
                after_metrics = self._dual_pose_metrics(candidate_q, current_targets)
                after_score = self._dual_pose_score(after_metrics)
                pose_progress = before_score - after_score
                candidate_lower_margin = candidate_q - q_min
                candidate_upper_margin = q_max - candidate_q
                margin_before = float(np.min(min_margin[arm_indices]))
                margin_after = float(np.min(np.minimum(candidate_lower_margin, candidate_upper_margin)[arm_indices]))
                margin_progress = margin_after - margin_before
                candidate_metadata = {
                    "local_step_limit_escape_candidate": True,
                    "local_step_limit_escape_side": side,
                    "local_step_limit_escape_joint_index": int(local_index),
                    "local_step_limit_escape_full_joint_index": int(full_index),
                    "local_step_limit_escape_direction": int(direction),
                    "local_step_limit_escape_step_deg": float(np.rad2deg(step)),
                    "local_step_limit_escape_step_fraction": float(step_fraction),
                    "local_step_limit_escape_min_margin_before_deg": float(np.rad2deg(margin_before)),
                    "local_step_limit_escape_min_margin_after_deg": float(np.rad2deg(margin_after)),
                    "local_step_limit_escape_margin_progress_deg": float(np.rad2deg(margin_progress)),
                    "local_step_limit_escape_pose_progress": float(pose_progress),
                }
                scored.append(
                    (
                        int(full_index not in active_set),
                        -float(margin_progress),
                        -float(pose_progress),
                        int(local_index),
                        side,
                        int(direction),
                        candidate_q,
                        candidate_metadata,
                    )
                )

        scored.sort(key=lambda item: item[:6])
        candidates: list[_DualArmCandidate] = []
        seen: list[np.ndarray] = []
        for _non_active, _margin_score, _pose_score, local_index, side, direction, candidate_q, candidate_metadata in scored:
            if len(candidates) >= max_candidates:
                break
            if any(np.allclose(candidate_q, existing, atol=1e-9) for existing in seen):
                continue
            seen.append(candidate_q.copy())
            candidates.append(
                _DualArmCandidate(
                    name=f"joint_limit_escape_{side}_j{int(local_index) + 1}_{'pos' if int(direction) > 0 else 'neg'}",
                    qpos=candidate_q.copy(),
                    source="joint_limit_escape",
                    method=f"joint_limit_escape_{side}_j{int(local_index) + 1}_{'pos' if int(direction) > 0 else 'neg'}",
                    metadata=candidate_metadata,
                )
            )
        metadata.update(
            {
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "step_fraction": float(step_fraction),
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

    def _evaluate_dual_arm_step_candidate(
        self,
        *,
        candidate: _DualArmCandidate,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
        target_activity: dict[str, object] | None = None,
        step_scale: float,
    ) -> tuple[bool, np.ndarray, dict[str, object]]:
        safety_profile = self.ik_safety_profile
        assert safety_profile is not None
        reference_full = np.asarray(reference_full_q, dtype=float).reshape(-1)
        raw_full = np.asarray(candidate.qpos, dtype=float).reshape(reference_full.shape)
        scale = max(1e-6, min(1.0, float(step_scale)))
        output_q, limit_metadata = self._limit_full_joint_step(
            reference_full_q=reference_full,
            candidate_full_q=raw_full,
            step_scale=scale,
            target_activity=target_activity,
            current_targets=current_targets,
        )
        metadata: dict[str, object] = {
            "candidate": candidate.name,
            "source": candidate.source,
            "method": candidate.method,
            "candidate_metadata": dict(candidate.metadata),
            "step_scale": scale,
            **limit_metadata,
        }

        raw_left, raw_right = self._split_solution(raw_full)
        ref_left, ref_right = self._split_solution(reference_full)
        for side, ref_q, raw_q in (
            ("left", ref_left, raw_left),
            ("right", ref_right, raw_right),
        ):
            raw_delta_deg = np.rad2deg(np.abs(raw_q - ref_q))
            branch_indices = tuple(int(index) for index in safety_profile.hard_branch_joint_indices)
            branch_limits = safety_profile.hard_branch_limits_array()
            if branch_indices:
                branch_delta = raw_delta_deg[list(branch_indices)]
                metadata[f"{side}_hard_branch_joint_delta_deg"] = branch_delta.tolist()
                if np.any(branch_delta > branch_limits):
                    exceeded = tuple(
                        int(branch_indices[index])
                        for index, exceeded in enumerate((branch_delta > branch_limits).tolist())
                        if bool(exceeded)
                    )
                    escape_metadata = self._posture_hard_branch_escape_metadata(
                        side=side,
                        reference_q=ref_q,
                        candidate_q=raw_q,
                        exceeded_joint_indices=exceeded,
                    )
                    if escape_metadata is not None:
                        metadata[f"{side}_posture_hard_branch_escape"] = escape_metadata
                    if escape_metadata is None or not bool(escape_metadata.get("allowed", False)):
                        metadata["reject_reason"] = f"hard_branch_jump:{side}"
                        metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()
                        metadata["score"] = 1e18
                        return False, output_q, metadata
                    metadata[f"{side}_hard_branch_escape_allowed"] = True
                    metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()

        if not self._full_qpos_within_limits(output_q):
            metadata["reject_reason"] = "joint_position_limit"
            metadata["score"] = 1e18
            return False, output_q, metadata

        operating_margin = self._joint_operating_margin_metadata(
            reference_full_q=reference_full,
            output_full_q=output_q,
        )
        metadata["joint_operating_margin"] = operating_margin
        if bool(operating_margin["reject"]):
            metadata["reject_reason"] = "joint_operating_margin"
            metadata["score"] = 1e18
            return False, output_q, metadata

        before_metrics = self._dual_pose_metrics(reference_full, current_targets)
        after_metrics = self._dual_pose_metrics(output_q, current_targets)
        output_motion_metrics = self._dual_output_motion_metrics(reference_full, output_q)
        before_score = self._dual_pose_score(before_metrics)
        after_score = self._dual_pose_score(after_metrics)
        before_pose_fraction = self._dual_pose_fraction(before_metrics)
        posture_metadata = self._dual_posture_comparison_metadata(
            reference_full_q=reference_full,
            candidate_full_q=output_q,
        )
        posture_penalty_deg = (
            0.0
            if posture_metadata is None
            else float(posture_metadata.get("score_penalty_deg", 0.0) or 0.0)
        )
        posture_improvement_deg = (
            0.0
            if posture_metadata is None
            else float(posture_metadata.get("improvement_deg", 0.0) or 0.0)
        )
        progress = before_score - after_score
        worsen_limits = self._dual_worsen_limits(before_metrics)
        bounded_worsening = self._dual_bounded_worsening(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_position_worsen_m=float(worsen_limits["position_m"]),
            max_orientation_worsen_deg=float(worsen_limits["orientation_deg"]),
        )
        orientation_recovery_limits = {
            "position_m": max(
                float(worsen_limits["position_m"]),
                max(
                    0.0,
                    float(getattr(safety_profile, "local_step_orientation_recovery_max_position_worsen_m", 0.06)),
                ),
            ),
            "orientation_deg": max(
                float(worsen_limits["orientation_deg"]),
                max(
                    0.0,
                    float(getattr(safety_profile, "local_step_dynamic_worsen_max_orientation_deg", 8.0)),
                ),
            ),
            "after_position_m": max(
                0.0,
                float(getattr(safety_profile, "local_step_orientation_recovery_max_after_position_error_m", 0.25)),
            ),
        }
        orientation_recovery_bounded = self._dual_bounded_worsening(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_position_worsen_m=float(orientation_recovery_limits["position_m"]),
            max_orientation_worsen_deg=float(orientation_recovery_limits["orientation_deg"]),
        )
        max_orientation_progress_deg = max(
            float(before_metrics[side]["rotation_deg"]) - float(after_metrics[side]["rotation_deg"])
            for side in ("left", "right")
        )
        max_after_position_m = max(float(after_metrics[side]["translation_m"]) for side in ("left", "right"))
        orientation_recovery_ok = (
            bool(getattr(safety_profile, "local_step_orientation_recovery_enabled", False))
            and max_orientation_progress_deg
            >= max(0.0, float(getattr(safety_profile, "local_step_orientation_recovery_min_progress_deg", 5.0)))
            and orientation_recovery_bounded
            and (
                float(orientation_recovery_limits["after_position_m"]) <= 0.0
                or max_after_position_m <= float(orientation_recovery_limits["after_position_m"])
            )
        )
        within_tolerance = self._dual_within_tolerance(after_metrics)
        side_position_progress_m = {
            side: float(before_metrics[side]["translation_m"]) - float(after_metrics[side]["translation_m"])
            for side in ("left", "right")
        }
        side_orientation_progress_deg = {
            side: float(before_metrics[side]["rotation_deg"]) - float(after_metrics[side]["rotation_deg"])
            for side in ("left", "right")
        }
        inactive_hold = self._inactive_side_hold_evaluation(
            target_activity=target_activity,
            output_motion_metrics=output_motion_metrics,
            side_position_progress_m=side_position_progress_m,
            side_orientation_progress_deg=side_orientation_progress_deg,
        )
        max_position_progress_m = max(side_position_progress_m.values())
        component_progress_limits = {
            "position_m": max(
                float(worsen_limits["position_m"]),
                max(
                    0.0,
                    float(
                        getattr(
                            safety_profile,
                            "local_step_component_progress_max_position_worsen_m",
                            0.0,
                        )
                    ),
                ),
            ),
            "orientation_deg": max(
                float(worsen_limits["orientation_deg"]),
                max(
                    0.0,
                    float(
                        getattr(
                            safety_profile,
                            "local_step_component_progress_max_orientation_worsen_deg",
                            0.0,
                        )
                    ),
                ),
            ),
            "after_position_m": max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_component_progress_max_after_position_error_m",
                        0.0,
                    )
                ),
            ),
            "max_score_regression": max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_component_progress_max_score_regression",
                        0.0,
                    )
                ),
            ),
            "min_position_progress_m": max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_component_progress_min_position_m",
                        safety_profile.local_step_accept_position_progress_m,
                    )
                ),
            ),
            "min_orientation_progress_deg": max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_component_progress_min_orientation_deg",
                        safety_profile.local_step_accept_orientation_progress_deg,
                    )
                ),
            ),
        }
        component_progress_bounded = self._dual_bounded_worsening(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_position_worsen_m=float(component_progress_limits["position_m"]),
            max_orientation_worsen_deg=float(component_progress_limits["orientation_deg"]),
        )
        component_progress_ok = (
            bool(getattr(safety_profile, "local_step_component_progress_enabled", False))
            and not within_tolerance
            and progress >= -float(component_progress_limits["max_score_regression"])
            and component_progress_bounded
            and (
                float(component_progress_limits["after_position_m"]) <= 0.0
                or max_after_position_m <= float(component_progress_limits["after_position_m"])
            )
            and (
                max_position_progress_m >= float(component_progress_limits["min_position_progress_m"])
                or max_orientation_progress_deg
                >= float(component_progress_limits["min_orientation_progress_deg"])
            )
        )
        if inactive_hold["reject"]:
            reject_sides = ",".join(str(side) for side in inactive_hold["reject_sides"])
            metadata.update(
                {
                    "reject_reason": f"inactive_side_unexplained_motion:{reject_sides}",
                    "score": 1e18,
                    "dual_pose_progress": float(progress),
                    "dual_pose_before": before_metrics,
                    "dual_pose_after": after_metrics,
                    "dual_output_motion": output_motion_metrics,
                    "target_activity": dict(target_activity or {}),
                    "inactive_side_hold": inactive_hold,
                    "dual_bounded_worsening": bool(bounded_worsening),
                    "dual_side_position_progress_m": dict(side_position_progress_m),
                    "dual_side_orientation_progress_deg": dict(side_orientation_progress_deg),
                }
            )
            return False, output_q, metadata
        limit_escape_candidate = bool(candidate.metadata.get("local_step_limit_escape_candidate", False))
        posture_escape_candidate = bool(candidate.metadata.get("local_step_posture_escape_candidate", False))
        wrist_unload_candidate = bool(candidate.metadata.get("local_step_wrist_unload_candidate", False))
        limit_escape_margin_progress_deg = float(
            candidate.metadata.get("local_step_limit_escape_margin_progress_deg", 0.0) or 0.0
        )
        wrist_unload_improvement_deg = float(
            candidate.metadata.get("local_step_wrist_unload_cost_improvement_deg", 0.0) or 0.0
        )
        limit_escape_bounded = self._dual_bounded_worsening(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_position_worsen_m=max(
                float(worsen_limits["position_m"]),
                max(0.0, float(getattr(safety_profile, "local_step_limit_escape_max_position_worsen_m", 0.025))),
            ),
            max_orientation_worsen_deg=max(
                float(worsen_limits["orientation_deg"]),
                max(0.0, float(getattr(safety_profile, "local_step_limit_escape_max_orientation_worsen_deg", 10.0))),
            ),
        )
        posture_escape_bounded = False
        posture_escape_active = False
        if self.posture_preference is not None and posture_metadata is not None:
            posture_escape_active = bool(
                posture_improvement_deg
                >= max(0.0, float(self.posture_preference.local_escape_min_improvement_deg))
                and (
                    posture_escape_candidate
                    or bool(posture_metadata.get("any_reference_bad", False))
                )
            )
            posture_escape_bounded = self._dual_bounded_worsening(
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                max_position_worsen_m=max(
                    float(worsen_limits["position_m"]),
                    max(0.0, float(self.posture_preference.local_escape_max_position_worsen_m)),
                ),
                max_orientation_worsen_deg=max(
                    float(worsen_limits["orientation_deg"]),
                    max(0.0, float(self.posture_preference.local_escape_max_orientation_worsen_deg)),
                ),
            )
        wrist_unload_active = False
        wrist_unload_bounded = False
        if self.posture_preference is not None:
            wrist_unload_active = bool(
                wrist_unload_candidate
                and (
                    wrist_unload_improvement_deg
                    >= max(0.0, float(self.posture_preference.wrist_unload_min_cost_improvement_deg))
                    or progress > 0.0
                )
            )
            wrist_unload_bounded = self._dual_bounded_worsening(
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                max_position_worsen_m=max(
                    float(worsen_limits["position_m"]),
                    max(0.0, float(self.posture_preference.wrist_unload_max_position_worsen_m)),
                ),
                max_orientation_worsen_deg=max(
                    float(worsen_limits["orientation_deg"]),
                    max(0.0, float(self.posture_preference.wrist_unload_max_orientation_worsen_deg)),
                ),
            )
        min_progress = max(0.0, float(safety_profile.local_step_min_progress_fraction))
        progress_accept_reason = ""
        if within_tolerance:
            if bool(getattr(safety_profile, "local_step_accept_worse_within_tolerance", True)):
                progress_accept_reason = "within_pose_tolerance"
            elif progress >= -min_progress:
                progress_accept_reason = "within_pose_tolerance_nonregressing"
            elif (
                limit_escape_candidate
                and bool(getattr(safety_profile, "local_step_limit_escape_accept_worse_within_tolerance", True))
                and limit_escape_margin_progress_deg > 0.0
                and limit_escape_bounded
            ):
                progress_accept_reason = "within_pose_tolerance_joint_limit_escape"
        elif progress >= min_progress and bounded_worsening:
            progress_accept_reason = "dual_pose_score_progress"
        elif orientation_recovery_ok:
            progress_accept_reason = "dual_orientation_recovery_bounded_position_worsening"
        elif component_progress_ok:
            progress_accept_reason = "dual_component_progress_bounded_score_regression"
        elif (
            limit_escape_candidate
            and bool(getattr(safety_profile, "local_step_limit_escape_accept_bounded_worse", True))
            and limit_escape_margin_progress_deg > 0.0
            and limit_escape_bounded
        ):
            progress_accept_reason = "joint_limit_escape_bounded_worsening"
        elif posture_escape_active and posture_escape_bounded:
            progress_accept_reason = "posture_escape_bounded_worsening"
        elif wrist_unload_active and wrist_unload_bounded:
            progress_accept_reason = "wrist_unload_bounded_worsening"
        if not progress_accept_reason:
            metadata.update(
                {
                    "reject_reason": "dual_local_step_no_pose_progress",
                    "score": 1e18,
                    "dual_pose_progress": float(progress),
                    "dual_pose_before": before_metrics,
                    "dual_pose_after": after_metrics,
                    "dual_output_motion": output_motion_metrics,
                    "dual_bounded_worsening": bool(bounded_worsening),
                    "target_activity": dict(target_activity or {}),
                    "inactive_side_hold": inactive_hold,
                    "dual_orientation_recovery_ok": bool(orientation_recovery_ok),
                    "dual_component_progress_ok": bool(component_progress_ok),
                    "dual_component_progress_bounded": bool(component_progress_bounded),
                    "dual_component_progress_limits": dict(component_progress_limits),
                    "dual_side_position_progress_m": dict(side_position_progress_m),
                    "dual_side_orientation_progress_deg": dict(side_orientation_progress_deg),
                    "dual_max_position_progress_m": float(max_position_progress_m),
                    "dual_max_orientation_progress_deg": float(max_orientation_progress_deg),
                    "dual_limit_escape_bounded_worsening": bool(limit_escape_bounded),
                    "dual_posture_escape_active": bool(posture_escape_active),
                    "dual_posture_escape_bounded_worsening": bool(posture_escape_bounded),
                    "dual_wrist_unload_active": bool(wrist_unload_active),
                    "dual_wrist_unload_bounded_worsening": bool(wrist_unload_bounded),
                    "posture_preference": posture_metadata,
                }
            )
            return False, output_q, metadata

        delta_deg = np.rad2deg(output_q - reference_full)
        metadata.update(
            {
                "reject_reason": None,
                "dual_pose_before": before_metrics,
                "dual_pose_after": after_metrics,
                "dual_output_motion": output_motion_metrics,
                "dual_pose_score_before": float(before_score),
                "dual_pose_score_after": float(after_score),
                "dual_pose_fraction_before": float(before_pose_fraction),
                "dual_pose_progress": float(progress),
                "dual_progress_accept_reason": progress_accept_reason,
                "dual_bounded_worsening": bool(bounded_worsening),
                "target_activity": dict(target_activity or {}),
                "inactive_side_hold": inactive_hold,
                "dual_orientation_recovery_ok": bool(orientation_recovery_ok),
                "dual_orientation_recovery_limits": dict(orientation_recovery_limits),
                "dual_component_progress_ok": bool(component_progress_ok),
                "dual_component_progress_bounded": bool(component_progress_bounded),
                "dual_component_progress_limits": dict(component_progress_limits),
                "dual_side_position_progress_m": dict(side_position_progress_m),
                "dual_side_orientation_progress_deg": dict(side_orientation_progress_deg),
                "dual_max_position_progress_m": float(max_position_progress_m),
                "dual_max_orientation_progress_deg": float(max_orientation_progress_deg),
                "dual_limit_escape_bounded_worsening": bool(limit_escape_bounded),
                "dual_posture_escape_active": bool(posture_escape_active),
                "dual_posture_escape_bounded_worsening": bool(posture_escape_bounded),
                "dual_wrist_unload_active": bool(wrist_unload_active),
                "dual_wrist_unload_bounded_worsening": bool(wrist_unload_bounded),
                "dual_wrist_unload_improvement_deg": float(wrist_unload_improvement_deg),
                "posture_preference": posture_metadata,
                "dual_worsen_limits": dict(worsen_limits),
                "within_pose_tolerance": bool(within_tolerance),
                "dual_recovery_state": (
                    "joint_limit_escape"
                    if progress_accept_reason.startswith("joint_limit_escape")
                    else (
                        "orientation_recovery"
                        if progress_accept_reason.startswith("dual_orientation_recovery")
                        else (
                            "component_progress"
                            if progress_accept_reason.startswith("dual_component_progress")
                            else (
                                "posture_escape"
                                if progress_accept_reason.startswith("posture_escape")
                                else (
                                    "wrist_unload"
                                    if progress_accept_reason.startswith("wrist_unload")
                                    else "tracking"
                                )
                            )
                        )
                    )
                ),
                "joint_delta_deg": delta_deg.astype(float).tolist(),
                "joint_delta_norm_deg": float(np.linalg.norm(delta_deg)),
                "score": float(
                    after_score * 1_000_000.0
                    + posture_penalty_deg * 100_000.0
                    + float(inactive_hold["score_penalty"])
                    + np.linalg.norm(delta_deg) * 10.0
                    + (1.0 - scale) * 100.0
                    + (
                        -min(10_000_000.0, 5_000_000.0 + max(0.0, limit_escape_margin_progress_deg) * 1_000_000.0)
                        if progress_accept_reason.startswith("joint_limit_escape")
                        else 0.0
                    )
                    + (
                        -min(5_000_000.0, 100_000.0 + max(0.0, posture_improvement_deg) * 100_000.0)
                        if progress_accept_reason.startswith("posture_escape")
                        else 0.0
                    )
                    + (
                        -min(1_000_000.0, 50_000.0 + max(0.0, wrist_unload_improvement_deg) * 50_000.0)
                        if progress_accept_reason.startswith("wrist_unload")
                        else 0.0
                    )
                    + (-500.0 if progress_accept_reason.startswith("dual_orientation_recovery") else 0.0)
                    + (-250.0 if progress_accept_reason.startswith("dual_component_progress") else 0.0)
                ),
            }
        )
        return True, output_q, metadata

    def _limit_full_joint_step(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        step_scale: float,
        target_activity: dict[str, object] | None = None,
        current_targets: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        reference_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate_q = np.asarray(candidate_full_q, dtype=float).reshape(reference_q.shape)
        metadata: dict[str, object] = {
            "step_limited": False,
            "step_scale": float(step_scale),
            "candidate_local_step_clip_scale": float(
                self.config.local_step_candidate_clip_scale
            ),
            "execution_step_source": self._execution_step_source(),
            "joint_velocity_limit_profile": (
                None if self.joint_velocity_profile is None else self.joint_velocity_profile.name
            ),
            "joint_velocity_limit_dt_sec": float(self._execution_dt_sec),
        }
        if not self.config.enforce_realistic_joint_step or self._full_execution_joint_step is None:
            metadata["step_limit_enabled"] = False
            return candidate_q.copy(), metadata
        max_step = (
            np.asarray(self._full_execution_joint_step, dtype=float).reshape(reference_q.shape)
            * float(self.config.local_step_candidate_clip_scale)
        )
        full_limited = reference_q + np.clip(candidate_q - reference_q, -max_step, max_step)
        full_delta = full_limited - reference_q
        output_q = reference_q + full_delta * max(1e-6, min(1.0, float(step_scale)))
        inactive_capped_q, inactive_cap_metadata = self._apply_inactive_side_execution_cap(
            reference_full_q=reference_q,
            candidate_full_q=output_q,
            target_activity=target_activity,
            current_targets=current_targets,
        )
        limited = bool(
            not np.allclose(full_limited, candidate_q, atol=1e-10)
            or inactive_cap_metadata.get("applied", False)
        )
        output_q = inactive_capped_q
        left_delta, right_delta = self._split_solution(output_q - reference_q)
        left_max_step, right_max_step = self._split_solution(max_step)
        metadata.update(
            {
                "step_limit_enabled": True,
                "step_limited": bool(limited or step_scale < 1.0),
                "joint_velocity_limit_delta_deg": np.rad2deg(output_q - reference_q).astype(float).tolist(),
                "joint_velocity_limit_full_delta_deg": np.rad2deg(full_delta).astype(float).tolist(),
                "joint_velocity_limit_max_step_deg": np.rad2deg(max_step).astype(float).tolist(),
                "left_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(left_delta)))),
                "right_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(right_delta)))),
                "left_max_step_deg": np.rad2deg(left_max_step).astype(float).tolist(),
                "right_max_step_deg": np.rad2deg(right_max_step).astype(float).tolist(),
                "inactive_side_hold_execution_cap": inactive_cap_metadata,
            }
        )
        return output_q.copy(), metadata

    def _dual_pose_metrics(
        self,
        qpos: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        left_pose = self.ik_solver.compute_frame_pose(self.profile.left_frame_name, q)
        right_pose = self.ik_solver.compute_frame_pose(self.profile.right_frame_name, q)
        left_m, left_deg = pose_delta_metrics(left_pose, current_targets["left"])
        right_m, right_deg = pose_delta_metrics(right_pose, current_targets["right"])
        return {
            "left": {"translation_m": float(left_m), "rotation_deg": float(left_deg)},
            "right": {"translation_m": float(right_m), "rotation_deg": float(right_deg)},
        }

    def _dual_pose_score(self, metrics: dict[str, dict[str, float]]) -> float:
        safety_profile = self.ik_safety_profile
        assert safety_profile is not None
        position_tol = max(1e-6, float(safety_profile.target_tolerance.position_m))
        orientation_tol = max(1e-6, float(safety_profile.target_tolerance.orientation_deg))
        side_scores = []
        for side in ("left", "right"):
            values = metrics[side]
            side_scores.append(
                float(values["translation_m"]) / position_tol
                + float(values["rotation_deg"]) / orientation_tol
            )
        return float(max(side_scores) * 10.0 + sum(side_scores))

    def _target_activity_metadata(
        self,
        current_targets: dict[str, np.ndarray],
        *,
        cartesian_commands: dict[str, EffectorCommand | None] | None = None,
        source_metadata_by_side: Mapping[str, Mapping[str, object] | None]
        | None = None,
    ) -> dict[str, object]:
        translation_deadband = float(self.config.inactive_side_hold_translation_deadband_m)
        rotation_deadband = float(self.config.inactive_side_hold_rotation_deadband_deg)
        metadata: dict[str, object] = {
            "enabled": bool(self.config.inactive_side_hold_enabled),
            "translation_deadband_m": translation_deadband,
            "rotation_deadband_deg": rotation_deadband,
        }
        for side in ("left", "right"):
            cartesian_command = (
                None if cartesian_commands is None else cartesian_commands.get(side)
            )
            source_metadata = (
                None
                if source_metadata_by_side is None
                else source_metadata_by_side.get(side)
            )
            tracker_intent_raw = (
                source_metadata.get("tracker_motion_intent")
                if source_metadata is not None
                else (
                    None
                    if cartesian_command is None
                    else cartesian_command.metadata.get("tracker_motion_intent")
                )
            )
            if (
                bool(self.config.inactive_side_hold_enabled)
                and isinstance(tracker_intent_raw, dict)
                and tracker_intent_raw.get("state") != "DISABLED"
            ):
                tracker_intent = dict(tracker_intent_raw)
                hold_weight = max(
                    0.0,
                    min(1.0, float(tracker_intent.get("hold_weight", 0.0) or 0.0)),
                )
                metadata[side] = {
                    "source": "tracker_motion_state_machine",
                    "previous_target_available": self._last_target_poses.get(side)
                    is not None,
                    "translation_m": None,
                    "rotation_deg": None,
                    "activity_ratio": None,
                    "hold_weight": hold_weight,
                    "inactive": bool(tracker_intent.get("inactive", False)),
                    "state": tracker_intent.get("state"),
                    "quiet_dwell_sec": tracker_intent.get("quiet_dwell_sec"),
                    "exit_dwell_sec": tracker_intent.get("exit_dwell_sec"),
                    "transition_reason": tracker_intent.get("transition_reason"),
                    "evidence": tracker_intent.get("evidence"),
                    "motion_window": tracker_intent.get("motion_window"),
                }
                continue
            previous_target = self._last_target_poses.get(side)
            if (
                not bool(self.config.inactive_side_hold_enabled)
                or previous_target is None
            ):
                metadata[side] = {
                    "source": "target_delta_legacy",
                    "previous_target_available": previous_target is not None,
                    "translation_m": None,
                    "rotation_deg": None,
                    "activity_ratio": None,
                    "hold_weight": 0.0,
                    "inactive": False,
                }
                continue
            translation_m, rotation_deg = pose_delta_metrics(
                previous_target,
                current_targets[side],
            )
            if translation_deadband <= 0.0:
                translation_ratio = 0.0 if translation_m <= 0.0 else float("inf")
            else:
                translation_ratio = float(translation_m) / translation_deadband
            if rotation_deadband <= 0.0:
                rotation_ratio = 0.0 if rotation_deg <= 0.0 else float("inf")
            else:
                rotation_ratio = float(rotation_deg) / rotation_deadband
            activity_ratio = max(float(translation_ratio), float(rotation_ratio))
            hold_weight = 0.0
            if np.isfinite(activity_ratio):
                hold_weight = max(0.0, min(1.0, 1.0 - activity_ratio))
            metadata[side] = {
                "source": "target_delta_legacy",
                "previous_target_available": True,
                "translation_m": float(translation_m),
                "rotation_deg": float(rotation_deg),
                "activity_ratio": None if not np.isfinite(activity_ratio) else float(activity_ratio),
                "hold_weight": float(hold_weight),
                "inactive": bool(hold_weight > 0.0),
            }
        return metadata

    def _dual_output_motion_metrics(
        self,
        reference_full_q: np.ndarray,
        output_full_q: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        reference_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        output_q = np.asarray(output_full_q, dtype=float).reshape(reference_q.shape)
        ref_left_q, ref_right_q = self._split_solution(reference_q)
        out_left_q, out_right_q = self._split_solution(output_q)
        result: dict[str, dict[str, float]] = {}
        for side, frame_name, ref_q, out_q in (
            ("left", self.profile.left_frame_name, ref_left_q, out_left_q),
            ("right", self.profile.right_frame_name, ref_right_q, out_right_q),
        ):
            reference_pose = self.ik_solver.compute_frame_pose(frame_name, reference_q)
            output_pose = self.ik_solver.compute_frame_pose(frame_name, output_q)
            translation_m, rotation_deg = pose_delta_metrics(reference_pose, output_pose)
            joint_delta_deg = np.rad2deg(out_q - ref_q)
            result[side] = {
                "translation_m": float(translation_m),
                "rotation_deg": float(rotation_deg),
                "joint_max_abs_delta_deg": float(np.max(np.abs(joint_delta_deg))),
                "joint_weighted_delta_deg": float(
                    np.linalg.norm(joint_delta_deg * self._arm_joint_weights)
                ),
            }
        return result

    def _inactive_side_hold_evaluation(
        self,
        *,
        target_activity: dict[str, object] | None,
        output_motion_metrics: dict[str, dict[str, float]],
        side_position_progress_m: dict[str, float],
        side_orientation_progress_deg: dict[str, float],
    ) -> dict[str, object]:
        enabled = bool(
            self.config.inactive_side_hold_enabled
            and isinstance(target_activity, dict)
            and target_activity.get("enabled", False)
        )
        max_position_m = max(
            1e-9,
            float(self.config.inactive_side_hold_max_unexplained_position_m),
        )
        max_rotation_deg = max(
            1e-9,
            float(self.config.inactive_side_hold_max_unexplained_rotation_deg),
        )
        score_weight = float(self.config.inactive_side_hold_score_weight)
        result: dict[str, object] = {
            "enabled": enabled,
            "max_unexplained_position_m": max_position_m,
            "max_unexplained_rotation_deg": max_rotation_deg,
            "score_weight": score_weight,
            "score_penalty": 0.0,
            "reject": False,
            "reject_sides": [],
            "sides": {},
        }
        if not enabled:
            return result
        assert target_activity is not None
        sides: dict[str, object] = {}
        reject_sides: list[str] = []
        score_penalty = 0.0
        for side in ("left", "right"):
            side_activity_raw = target_activity.get(side, {})
            side_activity = (
                dict(side_activity_raw)
                if isinstance(side_activity_raw, dict)
                else {}
            )
            hold_weight = max(
                0.0,
                min(1.0, float(side_activity.get("hold_weight", 0.0) or 0.0)),
            )
            motion = output_motion_metrics[side]
            position_progress = max(0.0, float(side_position_progress_m.get(side, 0.0)))
            rotation_progress = max(0.0, float(side_orientation_progress_deg.get(side, 0.0)))
            unexplained_position_m = max(
                0.0,
                float(motion["translation_m"]) - position_progress,
            )
            unexplained_rotation_deg = max(
                0.0,
                float(motion["rotation_deg"]) - rotation_progress,
            )
            normalized = (
                unexplained_position_m / max_position_m
                + unexplained_rotation_deg / max_rotation_deg
            )
            side_penalty = hold_weight * normalized * score_weight
            side_reject = bool(
                hold_weight >= 0.999
                and (
                    unexplained_position_m > max_position_m
                    or unexplained_rotation_deg > max_rotation_deg
                )
            )
            if side_reject:
                reject_sides.append(side)
            score_penalty += side_penalty
            sides[side] = {
                "hold_weight": float(hold_weight),
                "output_translation_m": float(motion["translation_m"]),
                "output_rotation_deg": float(motion["rotation_deg"]),
                "position_progress_m": float(position_progress),
                "rotation_progress_deg": float(rotation_progress),
                "unexplained_position_m": float(unexplained_position_m),
                "unexplained_rotation_deg": float(unexplained_rotation_deg),
                "score_penalty": float(side_penalty),
                "reject": bool(side_reject),
            }
        result.update(
            {
                "score_penalty": float(score_penalty),
                "reject": bool(reject_sides),
                "reject_sides": reject_sides,
                "sides": sides,
            }
        )
        return result

    def _dual_pose_fraction(self, metrics: dict[str, dict[str, float]]) -> float:
        safety_profile = self.ik_safety_profile
        assert safety_profile is not None
        position_tol = max(1e-6, float(safety_profile.target_tolerance.position_m))
        orientation_tol = max(1e-6, float(safety_profile.target_tolerance.orientation_deg))
        fractions = []
        for side in ("left", "right"):
            values = metrics[side]
            fractions.append(float(values["translation_m"]) / position_tol)
            fractions.append(float(values["rotation_deg"]) / orientation_tol)
        return float(max(fractions) if fractions else 0.0)

    def _dual_stable_selection_pool(
        self,
        evaluated: list[_DualArmEvaluatedCandidate],
    ) -> tuple[list[_DualArmEvaluatedCandidate], dict[str, object]]:
        preferred = [
            item
            for item in evaluated
            if not self._dual_auxiliary_recovery_without_pose_progress(item.metadata)
        ]
        if preferred:
            return preferred, {
                "name": "prefer_tracking_over_auxiliary_posture_recovery",
                "accepted_count": len(evaluated),
                "selection_count": len(preferred),
                "posture_wrist_auxiliary_deprioritized_count": len(evaluated) - len(preferred),
                "fallback_to_auxiliary_recovery": False,
            }
        return evaluated, {
            "name": "prefer_tracking_over_auxiliary_posture_recovery",
            "accepted_count": len(evaluated),
            "selection_count": len(evaluated),
            "posture_wrist_auxiliary_deprioritized_count": 0,
            "fallback_to_auxiliary_recovery": True,
        }

    def _dual_auxiliary_recovery_without_pose_progress(self, metadata: dict[str, object]) -> bool:
        source = str(metadata.get("source", ""))
        if source not in {"posture_escape", "wrist_unload"}:
            return False
        reason = str(metadata.get("dual_progress_accept_reason", ""))
        if reason in {"dual_pose_score_progress", "dual_orientation_recovery_bounded_position_worsening"}:
            return False
        return True

    def _dual_within_tolerance(self, metrics: dict[str, dict[str, float]]) -> bool:
        safety_profile = self.ik_safety_profile
        assert safety_profile is not None
        return all(
            float(metrics[side]["translation_m"]) <= float(safety_profile.target_tolerance.position_m)
            and float(metrics[side]["rotation_deg"]) <= float(safety_profile.target_tolerance.orientation_deg)
            for side in ("left", "right")
        )

    def _dual_worsen_limits(self, before_metrics: dict[str, dict[str, float]]) -> dict[str, object]:
        safety_profile = self.ik_safety_profile
        assert safety_profile is not None
        base_position_m = max(0.0, float(safety_profile.local_step_max_position_worsen_m))
        base_orientation_deg = max(0.0, float(safety_profile.local_step_max_orientation_worsen_deg))
        before_m = max(float(before_metrics[side]["translation_m"]) for side in ("left", "right"))
        before_deg = max(float(before_metrics[side]["rotation_deg"]) for side in ("left", "right"))
        enabled = bool(getattr(safety_profile, "local_step_dynamic_worsen_enabled", False))
        start_m = max(0.0, float(getattr(safety_profile, "local_step_dynamic_worsen_error_start_m", 0.05)))
        start_deg = max(0.0, float(getattr(safety_profile, "local_step_dynamic_worsen_error_start_deg", 15.0)))
        active = bool(enabled and (before_m >= start_m or before_deg >= start_deg))
        position_m = base_position_m
        orientation_deg = base_orientation_deg
        if active:
            position_fraction = max(
                0.0,
                float(getattr(safety_profile, "local_step_dynamic_worsen_position_fraction", 0.10)),
            )
            orientation_fraction = max(
                0.0,
                float(getattr(safety_profile, "local_step_dynamic_worsen_orientation_fraction", 0.10)),
            )
            max_position_m = max(
                0.0,
                float(getattr(safety_profile, "local_step_dynamic_worsen_max_position_m", 0.02)),
            )
            max_orientation_deg = max(
                0.0,
                float(getattr(safety_profile, "local_step_dynamic_worsen_max_orientation_deg", 8.0)),
            )
            position_m = max(position_m, before_m * position_fraction)
            orientation_deg = max(orientation_deg, before_deg * orientation_fraction)
            if max_position_m > 0.0:
                position_m = min(position_m, max_position_m)
            if max_orientation_deg > 0.0:
                orientation_deg = min(orientation_deg, max_orientation_deg)
        return {
            "position_m": float(position_m),
            "orientation_deg": float(orientation_deg),
            "base_position_m": float(base_position_m),
            "base_orientation_deg": float(base_orientation_deg),
            "active": bool(active),
        }

    def _dual_bounded_worsening(
        self,
        *,
        before_metrics: dict[str, dict[str, float]],
        after_metrics: dict[str, dict[str, float]],
        max_position_worsen_m: float,
        max_orientation_worsen_deg: float,
    ) -> bool:
        for side in ("left", "right"):
            position_worsen = float(after_metrics[side]["translation_m"]) - float(
                before_metrics[side]["translation_m"]
            )
            orientation_worsen = float(after_metrics[side]["rotation_deg"]) - float(
                before_metrics[side]["rotation_deg"]
            )
            if position_worsen > max_position_worsen_m:
                return False
            if orientation_worsen > max_orientation_worsen_deg:
                return False
        return True

    def _full_qpos_within_limits(self, qpos: np.ndarray) -> bool:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        q_min = np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float).reshape(q.shape)
        q_max = np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float).reshape(q.shape)
        return bool(np.all(q >= q_min - 1e-8) and np.all(q <= q_max + 1e-8))

    def _arm_joint_entries(self, qpos: np.ndarray) -> tuple[tuple[str, int, int], ...]:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        if self.profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
            return tuple(
                [("left", index, index) for index in range(6)]
                + [("right", index, 18 + index) for index in range(6)]
            )
        if q.shape[0] == 14:
            return tuple(
                [("left", index, 1 + index) for index in range(6)]
                + [("right", index, 8 + index) for index in range(6)]
            )
        if q.shape[0] == 12:
            return tuple(
                [("left", index, index) for index in range(6)]
                + [("right", index, 6 + index) for index in range(6)]
            )
        raise ValueError(f"{self.name} cannot map arm joint entries for solver shape {q.shape}.")

    def _evaluate_jump_gate(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> dict[str, object]:
        jump_gate = self.ik_tuning.jump_gate
        relaxed = self._jump_hold_active
        per_joint_limit = (
            jump_gate.relaxed_per_joint_jump_deg
            if relaxed
            else jump_gate.per_joint_jump_deg
        )
        weighted_limit = (
            jump_gate.relaxed_weighted_jump_threshold_deg
            if relaxed
            else jump_gate.weighted_jump_threshold_deg
        )
        ref_left_q, ref_right_q = self._split_solution(reference_full_q)
        cand_left_q, cand_right_q = self._split_solution(candidate_full_q)

        metrics: dict[str, object] = {}
        triggered_sides: list[str] = []
        hard_branch_sides: list[str] = []
        soft_jump_warning_sides: list[str] = []
        for side, ref_q, cand_q in (
            ("left", ref_left_q, cand_left_q),
            ("right", ref_right_q, cand_right_q),
        ):
            pose_translation_m = float("inf")
            pose_rotation_deg = float("inf")
            previous_target = self._last_target_poses.get(side)
            if previous_target is not None:
                pose_translation_m, pose_rotation_deg = pose_delta_metrics(
                    previous_target,
                    current_targets[side],
                )
            pose_is_small = bool(
                previous_target is not None
                and pose_translation_m <= float(jump_gate.small_pose_translation_m)
                and pose_rotation_deg <= float(jump_gate.small_pose_rotation_deg)
            )
            exceeded, per_joint_delta_deg, weighted_jump_deg = joint_jump_exceeds(
                reference_q=ref_q,
                candidate_q=cand_q,
                joint_distance_weights=self._arm_joint_weights,
                per_joint_jump_deg=per_joint_limit,
                weighted_jump_threshold_deg=weighted_limit,
            )
            hard_branch_exceeded, hard_branch_delta_deg = hard_branch_jump_exceeds(
                reference_q=ref_q,
                candidate_q=cand_q,
                joint_indices=jump_gate.hard_branch_joint_indices,
                max_delta_deg=jump_gate.hard_branch_max_delta_deg,
            )
            if hard_branch_exceeded:
                hard_branch_sides.append(side)
            if hard_branch_exceeded:
                triggered_sides.append(side)
            elif pose_is_small and exceeded:
                if self.config.enforce_realistic_joint_step:
                    soft_jump_warning_sides.append(side)
                else:
                    triggered_sides.append(side)
            metrics[side] = {
                "pose_delta_m": None if not np.isfinite(pose_translation_m) else pose_translation_m,
                "pose_delta_deg": None if not np.isfinite(pose_rotation_deg) else pose_rotation_deg,
                "pose_is_small": pose_is_small,
                "joint_jump_max_deg": float(np.max(per_joint_delta_deg)),
                "joint_jump_weighted_deg": float(weighted_jump_deg),
                "joint_jump_exceeded": bool(exceeded),
                "hard_branch_joint_indices": list(jump_gate.hard_branch_joint_indices),
                "hard_branch_delta_deg": hard_branch_delta_deg.tolist(),
                "hard_branch_max_delta_deg": list(jump_gate.hard_branch_max_delta_deg),
                "hard_branch_exceeded": bool(hard_branch_exceeded),
            }
        metrics["soft_joint_jump_warning_sides"] = list(soft_jump_warning_sides)

        release_ready = False
        release_metrics: dict[str, object] = {}
        if self._jump_hold_active and self._jump_hold_anchor_targets is not None:
            for side in ("left", "right"):
                release_translation_m, release_rotation_deg = pose_delta_metrics(
                    self._jump_hold_anchor_targets[side],
                    current_targets[side],
                )
                side_ready = bool(
                    release_translation_m >= float(jump_gate.release_translation_m)
                    or release_rotation_deg >= float(jump_gate.release_rotation_deg)
                )
                release_ready = release_ready or side_ready
                release_metrics[side] = {
                    "translation_m": release_translation_m,
                    "rotation_deg": release_rotation_deg,
                    "ready": side_ready,
                }
        metrics["release"] = release_metrics

        hold = bool(triggered_sides)
        if hold:
            status = "holding" if self._jump_hold_active else "hold_entered"
            if hard_branch_sides:
                reason = f"hard_branch_jump:{','.join(hard_branch_sides)}"
            else:
                reason = f"small_pose_large_q_jump:{','.join(triggered_sides)}"
        elif soft_jump_warning_sides:
            status = "warning"
            reason = f"small_pose_large_q_jump_warning_only:{','.join(soft_jump_warning_sides)}"
        elif self._jump_hold_active and release_ready:
            status = "released"
            reason = "hold_anchor_released"
        elif self._jump_hold_active:
            status = "recovering"
            reason = None
        else:
            status = "pass"
            reason = None
        return {
            "hold": hold,
            "status": status,
            "reason": reason,
            "release_ready": release_ready,
            "metrics": metrics,
        }

    def _enter_hold(
        self,
        *,
        hold_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> None:
        if not self._jump_hold_active:
            self._jump_hold_active = True
            self._jump_hold_full_q = np.asarray(hold_full_q, dtype=float).copy()
            self._jump_hold_anchor_targets = {
                side: pose.copy()
                for side, pose in current_targets.items()
            }
            self._jump_hold_frames = 0
        self._jump_hold_frames += 1

    def _clear_hold(self) -> None:
        self._jump_hold_active = False
        self._jump_hold_full_q = None
        self._jump_hold_anchor_targets = None
        self._jump_hold_frames = 0

    def _hold_seed_candidates(self) -> list[tuple[str, np.ndarray]]:
        if self._jump_hold_full_q is None:
            return []
        hold_q = self._jump_hold_full_q.copy()
        hold_left, hold_right = self._split_solution(hold_q)
        left_min, left_max = self._left_arm_limits
        right_min, right_max = self._right_arm_limits
        left_alt = preferred_wrist_flip_candidate(
            hold_left,
            reference_q=hold_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        right_alt = preferred_wrist_flip_candidate(
            hold_right,
            reference_q=hold_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        seed_map = {"current": hold_q.copy()}
        if left_alt is not None:
            seed_map["left_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=left_alt,
                right_qpos=hold_right,
            )
        if right_alt is not None:
            seed_map["right_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=hold_left,
                right_qpos=right_alt,
            )
        if left_alt is not None and right_alt is not None:
            seed_map["both_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=left_alt,
                right_qpos=right_alt,
            )

        ordered: list[tuple[str, np.ndarray]] = []
        for seed_name in self.ik_tuning.dual_arm_recovery.seed_modes:
            candidate = seed_map.get(seed_name)
            if candidate is None:
                continue
            if any(np.allclose(candidate, existing, atol=1e-8) for _, existing in ordered):
                continue
            ordered.append((seed_name, candidate.copy()))
        if not ordered:
            ordered.append(("current", hold_q.copy()))
        return ordered

    def _attempt_hold_recovery(
        self,
        *,
        current_targets: dict[str, np.ndarray],
        profile: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not self._jump_hold_active or self._jump_hold_full_q is None:
            return None
        if not self.ik_tuning.dual_arm_recovery.enable_seed_expansion:
            return None
        if self._jump_hold_frames < int(self.ik_tuning.jump_gate.seed_expansion_after_hold_frames):
            return None

        reference_full_q = self._jump_hold_full_q.copy()
        best_result: dict[str, object] | None = None
        seen_candidates: list[np.ndarray] = []
        try:
            for seed_name, seed_q in self._hold_seed_candidates():
                self.ik_solver.set_configuration_q(seed_q)
                try:
                    raw_q = self.ik_solver.solve(
                        targets=[
                            self._pose_to_target(current_targets["left"]),
                            self._pose_to_target(current_targets["right"]),
                        ]
                    )
                    self._record_solver_profile(profile)
                except Exception:
                    continue
                for branch_name, candidate_q, score_deg in self._branch_candidates(
                    raw_full_q=raw_q,
                    reference_full_q=reference_full_q,
                ):
                    if any(np.allclose(candidate_q, existing, atol=1e-8) for existing in seen_candidates):
                        continue
                    seen_candidates.append(candidate_q.copy())
                    gate_result = self._evaluate_jump_gate(
                        reference_full_q=reference_full_q,
                        candidate_full_q=candidate_q,
                        current_targets=current_targets,
                    )
                    if bool(gate_result["hold"]) and not bool(gate_result["release_ready"]):
                        continue
                    min_distance_m = self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="hold_recovery_candidate",
                    )
                    if self._fallback_reason_for_distance(min_distance_m) is not None:
                        continue
                    candidate_name = (
                        seed_name if branch_name == "original" else f"{seed_name}+{branch_name}"
                    )
                    best_result = {
                        "q": candidate_q.copy(),
                        "min_distance_m": min_distance_m,
                        "candidate_name": candidate_name,
                        "score_deg": float(score_deg),
                        "jump_gate_status": gate_result["status"],
                        "jump_gate_reason": gate_result["reason"],
                        "release_ready": gate_result["release_ready"],
                        "metrics": gate_result["metrics"],
                    }
                    return best_result
        finally:
            self.ik_solver.set_configuration_q(reference_full_q)
        return best_result
