from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field
import math
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
from robot_motion.rm65.wrist_coaxial import (
    canonicalize_rm65_wrist_coaxial_step,
    stabilize_rm65_wrist_q5_step,
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
from robot_motion.position_control import (
    POSITION_QREF_FINGERPRINT_SCHEMA,
    position_qref_fingerprint,
)
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles
from robot_motion.runtime import EffectorCommand, Retargeter, TeleopCommand, TeleopObservation
from robot_motion.tcp import HAND_BASE_TCP_FRAME, canonical_tcp_frame
from robot_motion.transforms import as_pose_matrix, interpolate_pose, invert_pose

from .dual_arm_solver.geometry import (
    arm_joint_entries,
    arm_joint_indices,
    embed_arm_qpos,
    split_arm_solution,
)
from .tool_safety import tcp_safety_warnings


_HAND_BASE_FROM_ARM_LINK6 = np.eye(4, dtype=float)
_HAND_BASE_FROM_ARM_LINK6[2, 3] = 0.009
_ARM_LINK6_FROM_HAND_BASE = np.linalg.inv(_HAND_BASE_FROM_ARM_LINK6)
_CURRENT_TARGET_HOLD_TRANSLATION_M = 1e-3
_CURRENT_TARGET_HOLD_ROTATION_DEG = 0.5
IK_FRAMEWORK_ROBOT_MOTION = "robot_motion"
IK_FRAMEWORK_CHOICES = (IK_FRAMEWORK_ROBOT_MOTION,)
IK_STEP_MODE_LEGACY_ITERATIVE = "legacy_iterative"
IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE = (
    "single_step_singularity_aware"
)
IK_STEP_MODE_CHOICES = (
    IK_STEP_MODE_LEGACY_ITERATIVE,
    IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE,
)


JOINT_LIMIT_GUARD_LEGACY = "legacy"
JOINT_LIMIT_GUARD_BOUNDARY_HOLD = "boundary_hold"
JOINT_LIMIT_GUARD_FORCED_ESCAPE_LATCHED = "forced_escape_latched"
JOINT_LIMIT_GUARD_MODE_CHOICES = (
    JOINT_LIMIT_GUARD_LEGACY,
    JOINT_LIMIT_GUARD_BOUNDARY_HOLD,
    JOINT_LIMIT_GUARD_FORCED_ESCAPE_LATCHED,
)


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
class DualArmCollisionAwareRetargeterConfig:
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
    # Position single-step IK may have several locally equivalent RM65
    # solutions.  This optional task anchors only the declared structural
    # joints to the previously admitted qref so a raw QP branch change does
    # not become a command.  It is disabled for the generic/force facade.
    branch_continuity_anchor_enabled: bool = False
    branch_continuity_anchor_cost: float = 0.5
    branch_continuity_anchor_joint_weights: np.ndarray | Sequence[float] = (
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    # Physical-limit protection is deliberately independent from the IK step
    # mode.  ``legacy`` preserves the stable pre-2026-08-02 selector.
    # ``boundary_hold`` is the reversible production candidate: it removes
    # only the inward component of an otherwise admitted solution near a hard
    # joint limit.  The historical forced escape/latch experiment remains
    # available for offline comparison but is never enabled implicitly.
    joint_limit_guard_mode: str = JOINT_LIMIT_GUARD_LEGACY
    joint_limit_boundary_guard_margin_deg: float = 1.5
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
    # Rollback boundary for the singularity-aware differential path.  Legacy
    # mode preserves the historical min/max repeated integration exactly.
    # The new mode advances physical configuration exactly once per outer
    # reference tick; additional numerical refinement must not masquerade as
    # additional elapsed time.
    solve_step_mode: str = IK_STEP_MODE_LEGACY_ITERATIVE
    joint_acceleration_limit_rad_s2: np.ndarray | Sequence[float] | None = None
    # Opt-in successor to the acceleration-only Pink limit.  It constrains the
    # numerical IK result to the same admitted-qref velocity and acceleration
    # interval already enforced after IK.  Keep this false by default so the
    # stable live-teleop profile remains an exact rollback.
    admitted_reference_motion_limit_enabled: bool = False
    wrist_coaxial_continuity_enabled: bool = True
    # The coaxial null-space projection is deliberately confined to the
    # immediate wrist-singularity neighborhood.  The previous 2--8 degree
    # band made ordinary wrist motion feel sticky long before the RM65 SDK
    # gate was relevant.  Keep a small hard floor for numerical conditioning,
    # but restore most q4/q6 response outside four degrees.
    wrist_coaxial_soft_threshold_deg: float = 4.0
    wrist_coaxial_hard_threshold_deg: float = 3.0
    # Within the narrow band, only a large q4-q6 null-direction step receives
    # the full attenuation.  Small operator inputs stay responsive while the
    # branch-jump envelope remains suppressed.
    wrist_coaxial_null_motion_soft_limit_deg: float = 5.0
    # Near the wrist coaxial neighborhood, a mathematically equivalent
    # +/-pi wrist-flip candidate must not compete with the locally continuous
    # Pink result.  The guard is only active in the opt-in single-step path;
    # zero disables it and preserves the historical branch ordering.
    wrist_branch_switch_guard_deg: float = 0.0
    # Position control may opt into running up to solve_max_iterations while
    # there is still time left in the frame budget.  The generic retargeter
    # default remains fixed-one-iteration so force/impedance callers retain
    # their historical behavior.
    solve_adaptive_iterations_enabled: bool = False
    solve_iteration_reserve_ms: float = 5.0
    # Portion of ``solve_iteration_reserve_ms`` assigned to certifying at
    # least the base/original candidate.  Zero exactly restores the historical
    # fixed 5 ms post-selection subtraction.
    candidate_certification_reserve_ms: float = 0.0
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
    output_qpos_normal_fast_path_enabled: bool = False
    output_qpos_normal_fast_path_alpha: float = 1.0
    output_qpos_normal_fast_path_guard_deg: float = 4.0
    # A valid tracker motion window can disappear for one scheduling frame
    # even though the authoritative channel and admission state remain
    # healthy.  Keep only that soft eligibility bit alive briefly; joint,
    # null-projection, q5-latch, and admission guards always fail closed.
    output_qpos_normal_fast_path_quality_grace_sec: float = 0.0
    # Optional endpoint for a nonzero, pre-IK bounded Cartesian admittance
    # modifier.  It is separate from ordinary tracker smoothing so profiles
    # can close modifier/qref lag without changing no-force teleoperation.
    output_qpos_admittance_fast_path_enabled: bool = False
    output_qpos_admittance_fast_path_alpha: float = 1.0
    # Default-off, one-tick confirmation for tiny modifier-only qpos direction
    # reversals.  A reversal is held only when either the previous admitted
    # step or the proposed step is within this threshold; a persistent reversal
    # is admitted on the next reference tick.  Zero preserves the historical
    # Active fast-path behavior.
    output_qpos_admittance_micro_reversal_hold_deg: float = 0.0
    # Optional nonzero step admitted on the confirmation tick instead of an
    # exact joint hold.  It is capped by the hold threshold and is force-path
    # metadata only; zero preserves the original one-tick stop.
    output_qpos_admittance_micro_reversal_creep_deg: float = 0.0
    # When q5 is in the narrow escape band, use the already safety-bounded
    # fast EMA endpoint instead of the normal slow alpha.  This only changes
    # the q5 singular neighborhood; all other joints retain the normal
    # adaptive smoothing policy.
    output_qpos_smoothing_singularity_alpha: float | None = None
    # Keep only a small margin above the 3 deg SDK wrist gate.  A larger
    # margin makes q5 sit visibly on an artificial boundary and contributes
    # to the post-singularity "sticky" feel.
    wrist_q5_escape_margin_deg: float = 0.25
    # Hysteresis for the position-only q5 escape.  While the previous and
    # candidate q5 values remain on the same side inside this release band,
    # an inward candidate keeps the previous escaped value instead of
    # toggling between the raw IK result and a one-step singularity escape.
    # Zero is the atomic rollback to the pre-hysteresis behavior.
    wrist_q5_hysteresis_release_deg: float = 6.0
    wrist_q5_hysteresis_requires_escape_latch: bool = False
    # A far Cartesian target should be allowed to pull the wrist out of the
    # release band instead of waiting for the operator to reverse the hand.
    # This only relaxes the q5 hysteresis hold; the 3.25 deg escape boundary,
    # per-cycle velocity cap, collision check, and final admission remain in
    # force.  Disable the option for an exact rollback to fixed-band behavior.
    wrist_q5_hysteresis_relax_for_far_target: bool = True
    wrist_q5_hysteresis_far_target_position_m: float = 0.10
    wrist_q5_hysteresis_far_target_orientation_deg: float = 20.0
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
    # Shoulder singularity cannot be classified reliably from one joint angle.
    # Position runtimes may inject the vendor/local kinematics analyzer through
    # set_position_singularity_analyzers().  The default stays off so force and
    # compatibility callers neither acquire a hardware dependency nor change
    # their branch ordering.
    prefer_non_shoulder_singular_branch: bool = False
    shoulder_singularity_threshold_m: float = 0.05
    position_singularity_candidate_certificate_enabled: bool = False
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
        self.solve_step_mode = str(self.solve_step_mode).strip().lower()
        if self.solve_step_mode not in IK_STEP_MODE_CHOICES:
            raise ValueError(
                f"solve_step_mode must be one of {IK_STEP_MODE_CHOICES}, "
                f"got {self.solve_step_mode!r}."
            )
        self.branch_continuity_anchor_enabled = bool(
            self.branch_continuity_anchor_enabled
        )
        self.branch_continuity_anchor_cost = float(
            self.branch_continuity_anchor_cost
        )
        if (
            not np.isfinite(self.branch_continuity_anchor_cost)
            or self.branch_continuity_anchor_cost <= 0.0
        ):
            raise ValueError("branch_continuity_anchor_cost must be positive.")
        branch_weights = np.asarray(
            self.branch_continuity_anchor_joint_weights,
            dtype=float,
        ).reshape(-1)
        if branch_weights.shape != (6,) or not np.all(np.isfinite(branch_weights)):
            raise ValueError(
                "branch_continuity_anchor_joint_weights must contain 6 finite values."
            )
        if np.any(branch_weights < 0.0):
            raise ValueError(
                "branch_continuity_anchor_joint_weights must be non-negative."
            )
        if self.branch_continuity_anchor_enabled and not np.any(branch_weights > 0.0):
            raise ValueError(
                "branch_continuity_anchor_joint_weights must activate at least one joint."
            )
        self.branch_continuity_anchor_joint_weights = tuple(
            float(value) for value in branch_weights
        )
        if self.branch_continuity_anchor_enabled and (
            self.solve_step_mode != IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        ):
            raise ValueError(
                "branch_continuity_anchor_enabled requires "
                "single_step_singularity_aware."
            )
        self.joint_limit_guard_mode = str(
            self.joint_limit_guard_mode
        ).strip().lower()
        if self.joint_limit_guard_mode not in JOINT_LIMIT_GUARD_MODE_CHOICES:
            raise ValueError(
                "joint_limit_guard_mode must be one of "
                f"{JOINT_LIMIT_GUARD_MODE_CHOICES}, got "
                f"{self.joint_limit_guard_mode!r}."
            )
        boundary_margin_deg = float(
            self.joint_limit_boundary_guard_margin_deg
        )
        if not np.isfinite(boundary_margin_deg) or boundary_margin_deg < 0.0:
            raise ValueError(
                "joint_limit_boundary_guard_margin_deg must be finite and >= 0."
            )
        self.joint_limit_boundary_guard_margin_deg = boundary_margin_deg
        self.wrist_coaxial_continuity_enabled = bool(
            self.wrist_coaxial_continuity_enabled
        )
        for threshold_name in (
            "wrist_coaxial_soft_threshold_deg",
            "wrist_coaxial_hard_threshold_deg",
            "wrist_coaxial_null_motion_soft_limit_deg",
            "wrist_branch_switch_guard_deg",
        ):
            threshold = float(getattr(self, threshold_name))
            if not np.isfinite(threshold) or (
                threshold <= 0.0
                if threshold_name == "wrist_coaxial_null_motion_soft_limit_deg"
                else threshold < 0.0
            ):
                comparison = (
                    "> 0"
                    if threshold_name == "wrist_coaxial_null_motion_soft_limit_deg"
                    else ">= 0"
                )
                raise ValueError(f"{threshold_name} must be finite and {comparison}.")
            setattr(self, threshold_name, threshold)
        if (
            self.wrist_coaxial_hard_threshold_deg
            > self.wrist_coaxial_soft_threshold_deg
        ):
            raise ValueError(
                "wrist_coaxial_hard_threshold_deg must be <= "
                "wrist_coaxial_soft_threshold_deg."
            )
        if self.joint_acceleration_limit_rad_s2 is not None:
            acceleration = np.asarray(
                self.joint_acceleration_limit_rad_s2,
                dtype=float,
            ).reshape(-1)
            if acceleration.shape != (6,) or not np.all(np.isfinite(acceleration)):
                raise ValueError(
                    "joint_acceleration_limit_rad_s2 must contain 6 finite values."
                )
            if np.any(acceleration <= 0.0):
                raise ValueError(
                    "joint_acceleration_limit_rad_s2 values must be positive."
                )
            self.joint_acceleration_limit_rad_s2 = acceleration.copy()
        self.admitted_reference_motion_limit_enabled = bool(
            self.admitted_reference_motion_limit_enabled
        )
        if self.admitted_reference_motion_limit_enabled and (
            self.solve_step_mode
            != IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        ):
            raise ValueError(
                "admitted_reference_motion_limit_enabled requires "
                "single_step_singularity_aware."
            )
        if self.admitted_reference_motion_limit_enabled and (
            self.joint_acceleration_limit_rad_s2 is None
        ):
            raise ValueError(
                "admitted_reference_motion_limit_enabled requires "
                "joint_acceleration_limit_rad_s2."
            )
        self.solve_adaptive_iterations_enabled = bool(self.solve_adaptive_iterations_enabled)
        if not np.isfinite(float(self.solve_iteration_reserve_ms)) or self.solve_iteration_reserve_ms < 0.0:
            raise ValueError("solve_iteration_reserve_ms must be finite and >= 0.")
        self.solve_iteration_reserve_ms = float(self.solve_iteration_reserve_ms)
        if (
            not np.isfinite(float(self.candidate_certification_reserve_ms))
            or float(self.candidate_certification_reserve_ms) < 0.0
        ):
            raise ValueError(
                "candidate_certification_reserve_ms must be finite and >= 0."
            )
        self.candidate_certification_reserve_ms = float(
            self.candidate_certification_reserve_ms
        )
        if self.candidate_certification_reserve_ms > self.solve_iteration_reserve_ms:
            raise ValueError(
                "candidate_certification_reserve_ms must be <= "
                "solve_iteration_reserve_ms."
            )
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
        self.prefer_non_shoulder_singular_branch = bool(
            self.prefer_non_shoulder_singular_branch
        )
        self.position_singularity_candidate_certificate_enabled = bool(
            self.position_singularity_candidate_certificate_enabled
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
            "shoulder_singularity_threshold_m",
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
        self.output_qpos_normal_fast_path_enabled = bool(
            self.output_qpos_normal_fast_path_enabled
        )
        normal_fast_alpha = float(self.output_qpos_normal_fast_path_alpha)
        if not np.isfinite(normal_fast_alpha) or not 0.0 <= normal_fast_alpha <= 1.0:
            raise ValueError("output_qpos_normal_fast_path_alpha must be in [0, 1].")
        if normal_fast_alpha < float(self.output_qpos_smoothing_fast_alpha):
            raise ValueError(
                "output_qpos_normal_fast_path_alpha must be >= the adaptive fast alpha."
            )
        self.output_qpos_normal_fast_path_alpha = normal_fast_alpha
        normal_fast_guard = float(self.output_qpos_normal_fast_path_guard_deg)
        if not np.isfinite(normal_fast_guard) or normal_fast_guard < 0.0:
            raise ValueError(
                "output_qpos_normal_fast_path_guard_deg must be finite and >= 0."
            )
        self.output_qpos_normal_fast_path_guard_deg = normal_fast_guard
        normal_fast_quality_grace_sec = float(
            self.output_qpos_normal_fast_path_quality_grace_sec
        )
        if (
            not np.isfinite(normal_fast_quality_grace_sec)
            or normal_fast_quality_grace_sec < 0.0
        ):
            raise ValueError(
                "output_qpos_normal_fast_path_quality_grace_sec must be finite and >= 0."
            )
        self.output_qpos_normal_fast_path_quality_grace_sec = (
            normal_fast_quality_grace_sec
        )
        self.output_qpos_admittance_fast_path_enabled = bool(
            self.output_qpos_admittance_fast_path_enabled
        )
        admittance_fast_alpha = float(
            self.output_qpos_admittance_fast_path_alpha
        )
        if (
            not np.isfinite(admittance_fast_alpha)
            or not 0.0 <= admittance_fast_alpha <= 1.0
        ):
            raise ValueError(
                "output_qpos_admittance_fast_path_alpha must be in [0, 1]."
            )
        if admittance_fast_alpha < float(self.output_qpos_smoothing_alpha):
            raise ValueError(
                "output_qpos_admittance_fast_path_alpha must be >= the slow alpha."
            )
        self.output_qpos_admittance_fast_path_alpha = admittance_fast_alpha
        admittance_reversal_hold_deg = float(
            self.output_qpos_admittance_micro_reversal_hold_deg
        )
        if (
            not np.isfinite(admittance_reversal_hold_deg)
            or admittance_reversal_hold_deg < 0.0
        ):
            raise ValueError(
                "output_qpos_admittance_micro_reversal_hold_deg must be "
                "finite and >= 0."
            )
        self.output_qpos_admittance_micro_reversal_hold_deg = (
            admittance_reversal_hold_deg
        )
        admittance_reversal_creep_deg = float(
            self.output_qpos_admittance_micro_reversal_creep_deg
        )
        if (
            not np.isfinite(admittance_reversal_creep_deg)
            or admittance_reversal_creep_deg < 0.0
            or admittance_reversal_creep_deg
            > admittance_reversal_hold_deg + 1e-12
        ):
            raise ValueError(
                "output_qpos_admittance_micro_reversal_creep_deg must be "
                "finite and in [0, hold_deg]."
            )
        self.output_qpos_admittance_micro_reversal_creep_deg = (
            admittance_reversal_creep_deg
        )
        if self.output_qpos_smoothing_singularity_alpha is not None:
            singularity_alpha = float(self.output_qpos_smoothing_singularity_alpha)
            if not np.isfinite(singularity_alpha) or not 0.0 <= singularity_alpha <= 1.0:
                raise ValueError(
                    "output_qpos_smoothing_singularity_alpha must be in [0, 1] when set."
                )
            if singularity_alpha < float(self.output_qpos_smoothing_alpha):
                raise ValueError(
                    "output_qpos_smoothing_singularity_alpha must be >= the slow alpha."
                )
            self.output_qpos_smoothing_singularity_alpha = singularity_alpha
        escape_margin = float(self.wrist_q5_escape_margin_deg)
        if not np.isfinite(escape_margin) or escape_margin < 0.0:
            raise ValueError("wrist_q5_escape_margin_deg must be finite and >= 0.")
        self.wrist_q5_escape_margin_deg = escape_margin
        hysteresis_release = float(self.wrist_q5_hysteresis_release_deg)
        if not np.isfinite(hysteresis_release) or hysteresis_release < 0.0:
            raise ValueError(
                "wrist_q5_hysteresis_release_deg must be finite and >= 0."
            )
        self.wrist_q5_hysteresis_release_deg = hysteresis_release
        self.wrist_q5_hysteresis_requires_escape_latch = bool(
            self.wrist_q5_hysteresis_requires_escape_latch
        )
        self.wrist_q5_hysteresis_relax_for_far_target = bool(
            self.wrist_q5_hysteresis_relax_for_far_target
        )
        for threshold_name in (
            "wrist_q5_hysteresis_far_target_position_m",
            "wrist_q5_hysteresis_far_target_orientation_deg",
        ):
            threshold = float(getattr(self, threshold_name))
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError(f"{threshold_name} must be finite and >= 0.")
            setattr(self, threshold_name, threshold)
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


class DualArmCollisionAwareRetargeter(Retargeter):
    """Device-neutral coupled target-pose to safe-qref controller.

    Operator-specific pose adapters may be injected by ``teleop_runtime``.
    Simulation and replay callers should bypass them and call
    :meth:`solve_target_pair` with a :class:`BimanualPoseTarget`.
    """

    def __init__(
        self,
        config: DualArmCollisionAwareRetargeterConfig,
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
        # Canonical safety owns a separate, final target boundary.  Unlike
        # ``target_override_filter`` (which composes optional source modifiers),
        # this hook is evaluated inside every bimanual solve path immediately
        # before IK.  Keeping it separate prevents a later-installed modifier
        # or a direct recovery ``build_command_for_context`` call from bypassing
        # an Active target-space safety projector.  The default remains inert.
        self.pre_ik_target_safety_filter: Callable[
            [dict[str, np.ndarray]],
            Mapping[str, np.ndarray] | None,
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
        self._last_admitted_velocity_full = np.zeros_like(
            np.asarray(self.ik_solver.current_q, dtype=float)
        )
        self._branch_continuity_anchor_enabled = False
        if config.branch_continuity_anchor_enabled:
            current_q = np.asarray(self.ik_solver.current_q, dtype=float).reshape(-1)
            branch_weights = np.asarray(
                config.branch_continuity_anchor_joint_weights,
                dtype=float,
            ).reshape(6)
            full_branch_cost = np.zeros_like(current_q)
            for indices in self._arm_joint_indices(current_q.size).values():
                full_branch_cost[indices] = (
                    float(config.branch_continuity_anchor_cost) * branch_weights
                )
            configure_branch_anchor = getattr(
                self.ik_solver,
                "configure_branch_continuity_anchor_task",
                None,
            )
            if not callable(configure_branch_anchor):
                raise ValueError(
                    "branch_continuity_anchor_enabled requires an IK solver "
                    "that implements configure_branch_continuity_anchor_task()."
                )
            configure_branch_anchor(cost=full_branch_cost)
            self._branch_continuity_anchor_enabled = True
        self._acceleration_limit_enabled = False
        self._admitted_reference_motion_limit_enabled = False
        if (
            config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and config.joint_acceleration_limit_rad_s2 is not None
        ):
            full_acceleration_limit = self._make_full_joint_acceleration_limit(
                np.asarray(config.joint_acceleration_limit_rad_s2, dtype=float)
            )
            if config.admitted_reference_motion_limit_enabled:
                configure_motion_limit = getattr(
                    self.ik_solver,
                    "configure_admitted_reference_motion_limit",
                    None,
                )
                if not callable(configure_motion_limit):
                    raise ValueError(
                        "admitted-reference motion limiting requires an IK solver "
                        "that implements configure_admitted_reference_motion_limit()."
                    )
                configure_motion_limit(
                    self._make_full_execution_joint_velocity_limit(
                        np.asarray(self.ik_solver.current_q, dtype=float)
                    ),
                    full_acceleration_limit,
                )
                self._admitted_reference_motion_limit_enabled = True
            else:
                configure_acceleration_limit = getattr(
                    self.ik_solver,
                    "configure_acceleration_limit",
                    None,
                )
                if not callable(configure_acceleration_limit):
                    raise ValueError(
                        "single_step_singularity_aware with acceleration limits "
                        "requires an IK solver that implements configure_acceleration_limit()."
                    )
                configure_acceleration_limit(full_acceleration_limit)
            self._acceleration_limit_enabled = True
            if self._admitted_reference_motion_limit_enabled:
                self._set_solver_motion_reference(
                    np.asarray(self.ik_solver.current_q, dtype=float)
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
        self._last_raw_solver_q: np.ndarray | None = None
        self._last_branch_continuity_anchor_metadata: dict[str, object] = {
            "enabled": bool(self._branch_continuity_anchor_enabled),
            "active_joint_mask": None,
            "disabled_for_limit_escape": [],
        }
        self._last_wrist_coaxial_metadata: dict[str, object] = {
            "enabled": False,
            "applied": False,
            "reason": "not_started",
        }
        self._last_directional_singularity_diagnostics: dict[str, object] = {}
        self._target_pose_smoother = config.target_pose_smoother
        self._last_output_qpos_smoothing_full_q: np.ndarray | None = None
        self._output_qpos_smoothing_effective_alpha = {
            side: float(config.output_qpos_smoothing_alpha) for side in ("left", "right")
        }
        self._output_qpos_smoothing_update_time: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._output_qpos_normal_fast_path_last_quality_valid_time: dict[
            str, float | None
        ] = {"left": None, "right": None}
        self._output_qpos_admittance_pending_reversal_direction = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self._position_singularity_analyzers: dict[
            str, Callable[[np.ndarray], Mapping[str, object]]
        ] = {}
        self._position_singularity_metadata_cache: dict[
            tuple[float, ...], dict[str, object]
        ] = {}
        # Directional singularity escape is intentionally stateful.  A raw
        # per-frame IK solve can alternate sides while q3/q5 is close to zero;
        # retaining the admitted outward direction prevents that solver noise
        # from becoming a physical left/right oscillation.  The latch is only
        # used by profiles that explicitly opt into the new policy.
        self._singularity_escape_direction_latch: dict[tuple[str, int], int] = {}
        self._singularity_escape_direction_failures: dict[tuple[str, int], int] = {}
        # Once one side/axis (or the coupled J3+J5 pair) is admitted, keep
        # that axis plan until it leaves the release band.  Sign latching
        # alone still allowed the selector to alternate between J3 and J5
        # when both candidates had similar Cartesian scores.
        self._singularity_escape_axis_latch: dict[str, tuple[int, ...]] = {}
        # A physical joint-limit escape is a safety transition, not an
        # ordinary pose-scoring alternative.  Keep the admitted outward
        # direction until the joint clears the activation band with a small
        # release margin; otherwise the next raw IK frame can immediately
        # select the opposite (inward) branch and create a qref toggle.
        self._joint_limit_escape_direction_latch: dict[tuple[str, int], int] = {}
        # If the joint has crossed the physical release band but the current
        # raw IK still asks for an inward step, keep the same escape direction
        # until that demand clears.  Without this short release guard, a
        # large raw branch can produce an escape/ordinary sawtooth at the
        # release boundary.
        self._joint_limit_escape_release_guard: dict[tuple[str, int], int] = {}
        self._wrist_q5_hysteresis_escape_latch = {
            "left": False,
            "right": False,
        }
        self._candidate_certificate_sequence = 0
        self._ik_shadow_trace_sink: Callable[..., object] | None = None

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
        self._reset_output_qpos_admittance_reversal_state()
        self._singularity_escape_direction_latch.clear()
        self._singularity_escape_direction_failures.clear()
        self._singularity_escape_axis_latch.clear()
        self._joint_limit_escape_direction_latch.clear()
        self._joint_limit_escape_release_guard.clear()
        self._wrist_q5_hysteresis_escape_latch = {
            "left": False,
            "right": False,
        }
        self._reset_output_qpos_smoothing_alpha_state()
        self._clear_hold()

    def set_position_singularity_analyzers(
        self,
        analyzers: Mapping[str, Callable[[np.ndarray], Mapping[str, object]]] | None,
    ) -> None:
        """Inject per-arm candidate analyzers without coupling robot_motion to hardware.

        This hook is used only by the opt-in position singularity preference.
        Passing ``None`` (or an empty mapping) is the atomic rollback path: the
        existing J3/J5 proxies remain available and shoulder classification is
        reported as unavailable.
        """

        resolved: dict[str, Callable[[np.ndarray], Mapping[str, object]]] = {}
        for side, analyzer in dict(analyzers or {}).items():
            normalized_side = self._validated_side(side)
            if not callable(analyzer):
                raise TypeError(f"position singularity analyzer for {side!r} must be callable.")
            resolved[normalized_side] = analyzer
        self._position_singularity_analyzers = resolved
        self._position_singularity_metadata_cache.clear()

    def set_ik_shadow_trace_sink(
        self,
        sink: Callable[..., object] | None,
    ) -> None:
        """Attach a passive, non-authoritative fixed-field IK trace sink."""

        if sink is not None and not callable(sink):
            raise TypeError("IK shadow trace sink must be callable or None.")
        self._ik_shadow_trace_sink = sink

    def _reset_output_qpos_smoothing_alpha_state(self) -> None:
        slow_alpha = float(self.config.output_qpos_smoothing_alpha)
        self._output_qpos_smoothing_effective_alpha = {
            "left": slow_alpha,
            "right": slow_alpha,
        }
        self._output_qpos_smoothing_update_time = {"left": None, "right": None}
        self._output_qpos_normal_fast_path_last_quality_valid_time = {
            "left": None,
            "right": None,
        }

    def _reset_output_qpos_admittance_reversal_state(
        self,
        side: str | None = None,
    ) -> None:
        sides = (
            (self._validated_side(side),)
            if side is not None
            else ("left", "right")
        )
        for resolved_side in sides:
            self._output_qpos_admittance_pending_reversal_direction[
                resolved_side
            ].fill(0.0)

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
            self._last_admitted_velocity_full = np.zeros_like(full_q)
            self._singularity_escape_direction_latch.clear()
            self._singularity_escape_direction_failures.clear()
            self._singularity_escape_axis_latch.clear()
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
            self._wrist_q5_hysteresis_escape_latch = {
                "left": False,
                "right": False,
            }
            self._set_solver_motion_reference(full_q)
            self._last_output_qpos_smoothing_full_q = None
            self._reset_output_qpos_admittance_reversal_state()
            self._reset_output_qpos_smoothing_alpha_state()
            self._clear_hold()
        elif (
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        ):
            # The real-position loop synchronizes the IK configuration from the
            # latest transport-accepted qcmd before and after every outer tick.
            # That synchronization is not a lifecycle reset: keep the velocity
            # admitted by the previous tick so Pink's acceleration constraint
            # remains stateful.  Rebase the post-IK smoother to the same
            # authoritative qcmd; otherwise a stale, ahead-of-transport qref can
            # reintroduce the q4-q6 null redistribution removed by the coaxial
            # canonicalizer.
            self._last_output_qpos_smoothing_full_q = full_q.copy()
            self._set_solver_motion_reference(full_q)

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

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray,
        right_qpos: Sequence[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return read-only tool-TCP FK for one admitted dual-arm qref pair."""

        left = np.asarray(left_qpos, dtype=float).reshape(-1)
        right = np.asarray(right_qpos, dtype=float).reshape(-1)
        if left.shape != (6,) or not np.all(np.isfinite(left)):
            raise ValueError("left_qpos must contain six finite values.")
        if right.shape != (6,) or not np.all(np.isfinite(right)):
            raise ValueError("right_qpos must contain six finite values.")
        full_q = self._embed_arm_qpos(
            current_full_q=self.ik_solver.current_q,
            left_qpos=left,
            right_qpos=right,
        )
        solver_targets = self._solver_world_targets_for_full_q(full_q)
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
                "single_physical_step"
                if self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                else "adaptive_budget"
                if bool(self.config.solve_adaptive_iterations_enabled)
                else "fixed_max"
            ),
            "solve_step_mode": str(self.config.solve_step_mode),
            "solve_iteration_max": int(self.config.solve_max_iterations),
            "solve_iteration_min": int(self.config.solve_min_iterations),
            "solve_iteration_effective_max": (
                1
                if self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                else int(self.config.solve_max_iterations)
            ),
            "solve_iteration_effective_min": (
                1
                if self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                else int(self.config.solve_min_iterations)
            ),
            "ik_damping": float(self.config.damping),
            "ik_acceleration_limit_enabled": bool(
                self._acceleration_limit_enabled
            ),
            "ik_admitted_reference_motion_limit_enabled": bool(
                self._admitted_reference_motion_limit_enabled
            ),
            "ik_branch_continuity_anchor_enabled": bool(
                self._branch_continuity_anchor_enabled
            ),
            "ik_branch_continuity_anchor_cost": float(
                self.config.branch_continuity_anchor_cost
            ),
            "ik_branch_continuity_anchor_joint_weights": list(
                self.config.branch_continuity_anchor_joint_weights
            ),
            "ik_joint_limit_guard_mode": self._joint_limit_guard_mode(),
            "ik_joint_limit_boundary_guard_margin_deg": float(
                self.config.joint_limit_boundary_guard_margin_deg
            ),
            "solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "candidate_certification_reserve_ms": float(
                self.config.candidate_certification_reserve_ms
            ),
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
        pre_ik_target_safety_metadata: dict[str, object] | None = None
        if self.pre_ik_target_safety_filter is not None:
            safety_result = self.pre_ik_target_safety_filter(
                {
                    "left": left_target.copy(),
                    "right": right_target.copy(),
                }
            )
            if safety_result is not None:
                if not isinstance(safety_result, Mapping):
                    raise TypeError(
                        "pre-IK target safety filter must return a side mapping or None"
                    )
                expected_sides = {"left", "right"}
                actual_sides = set(safety_result)
                if actual_sides != expected_sides:
                    raise ValueError(
                        "pre-IK target safety filter must return exactly left and right; "
                        f"got={sorted(actual_sides)}"
                    )
                left_target = as_pose_matrix(safety_result["left"])
                right_target = as_pose_matrix(safety_result["right"])
            metadata_provider = getattr(
                self.pre_ik_target_safety_filter,
                "command_metadata",
                None,
            )
            if callable(metadata_provider):
                provided_metadata = metadata_provider()
                if not isinstance(provided_metadata, Mapping):
                    raise TypeError(
                        "pre-IK target safety command_metadata must return a mapping"
                    )
                pre_ik_target_safety_metadata = dict(provided_metadata)
        current_targets = {"left": left_target.copy(), "right": right_target.copy()}
        solver_targets_before_solve = self.current_world_targets()
        target_activity_metadata = self._target_activity_metadata(
            current_targets,
            source_metadata_by_side={
                "left": left_source_metadata,
                "right": right_source_metadata,
            },
        )
        solve_iteration_mode = (
            "single_physical_step"
            if self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            else "adaptive_budget"
            if bool(self.config.solve_adaptive_iterations_enabled)
            else "fixed_max"
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
                "ik_branch_continuity_anchor": {
                    "enabled": bool(self._branch_continuity_anchor_enabled),
                    "cost": float(self.config.branch_continuity_anchor_cost),
                    "joint_weights": list(
                        self.config.branch_continuity_anchor_joint_weights
                    ),
                    "anchor": "previous_admitted_qref",
                    "scope": "structural_q3_q5_by_arm",
                },
                "ik_joint_limit_guard": {
                    "mode": self._joint_limit_guard_mode(),
                    "boundary_margin_deg": float(
                        self.config.joint_limit_boundary_guard_margin_deg
                    ),
                },
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
        if pre_ik_target_safety_metadata is not None:
            output_metadata["tcp_pose_envelope"] = pre_ik_target_safety_metadata

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
        canonical_solver_q: np.ndarray | None = None
        self._last_solve_iteration_budget_exhausted = False
        self._last_solve_iteration_stop_reason = "not_started"

        pre_solve_q = self.ik_solver.current_q.copy()
        active_q = pre_solve_q.copy()
        self._last_raw_solver_q = None
        self._last_branch_continuity_anchor_metadata = {
            "enabled": bool(self._branch_continuity_anchor_enabled),
            "active_joint_mask": None,
            "disabled_for_limit_escape": [],
        }
        self._last_wrist_coaxial_metadata = {
            "enabled": bool(
                self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            ),
            "applied": False,
            "reason": "target_hold_or_not_solved",
        }
        self._last_directional_singularity_diagnostics = {}
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
        current_target_hold_metadata = {
            "enabled": bool(current_target_hold_enabled),
            "policy": "shared_ik_unchanged",
            "translation_threshold_m": float(
                _CURRENT_TARGET_HOLD_TRANSLATION_M
            ),
            "rotation_threshold_deg": float(
                _CURRENT_TARGET_HOLD_ROTATION_DEG
            ),
            "target_already_reached": bool(target_already_reached),
        }
        if profile is not None:
            profile["current_target_hold_enabled"] = bool(current_target_hold_enabled)
            profile["current_target_hold_policy"] = "shared_ik_unchanged"
            profile["current_target_hold_translation_m"] = float(
                _CURRENT_TARGET_HOLD_TRANSLATION_M
            )
            profile["current_target_hold_rotation_deg"] = float(
                _CURRENT_TARGET_HOLD_ROTATION_DEG
            )
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
                    target_activity=target_activity_metadata,
                    profile=profile,
                    frame_start_time=frame_start,
                )
                raw_solver_q = (
                    np.asarray(solved_q, dtype=float).copy()
                    if self._last_raw_solver_q is None
                    else self._last_raw_solver_q.copy()
                )
                canonical_solver_q = np.asarray(solved_q, dtype=float).copy()
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
                # Commit directional escape state only after the candidate
                # selector has produced its collision-safe admitted qref.  A
                # rejected probe must not be mistaken for physical progress;
                # repeated fallback is aged by the bounded latch policy.
                self._update_singularity_escape_direction_state(
                    selected_q=candidate_q,
                    selection_metadata=candidate_selection_metadata,
                )
                candidate_q, selected_wrist_coaxial_metadata = (
                    self._canonicalize_wrist_coaxial_solution(
                        reference_full_q=pre_solve_q,
                        candidate_full_q=candidate_q,
                    )
                )
                candidate_q_before_q5_stabilization = candidate_q.copy()
                candidate_q, selected_wrist_q5_metadata = (
                    self._stabilize_wrist_q5_candidate(
                        reference_full_q=pre_solve_q,
                        candidate_full_q=candidate_q,
                        current_targets=current_targets,
                    )
                )
                self._last_wrist_coaxial_metadata = (
                    selected_wrist_coaxial_metadata
                )
                canonical_solver_q = np.asarray(candidate_q, dtype=float).copy()
                candidate_selection_metadata[
                    "singularity_aware_wrist_canonicalization"
                ] = dict(selected_wrist_coaxial_metadata)
                candidate_selection_metadata[
                    "singularity_aware_wrist_q5_candidate_stabilization"
                ] = dict(selected_wrist_q5_metadata)
                if not np.allclose(
                    candidate_q,
                    candidate_q_before_q5_stabilization,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    # q5-only task relaxation happens after candidate scoring;
                    # re-check the coupled collision envelope before the jump
                    # gate so the continuity guard cannot weaken safety.
                    branch_min_distance = self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="post_selection_wrist_q5_stabilization",
                    )
                if bool(selected_wrist_coaxial_metadata.get("applied", False)):
                    # Candidate selection may replace the raw Pink result with
                    # a locally generated branch/escape candidate.  Recompute
                    # collision distance after constraining that selected
                    # candidate; the earlier branch distance belongs to the
                    # uncanonicalized configuration.
                    branch_min_distance = self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="post_selection_wrist_canonicalization",
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
                reference_full_q=pre_solve_q,
                target_activity=target_activity_metadata,
                current_targets=current_targets,
                admission_status=status,
            )
        )
        if status in {
            "jump_gate_hold",
            "hold_last_safe",
            "unsafe_candidate",
        }:
            final_canonical_q = np.asarray(active_q, dtype=float).copy()
            final_wrist_coaxial_metadata = {
                "enabled": bool(
                    self.config.solve_step_mode
                    == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                ),
                "applied": False,
                "reason": "hold_status_preserved",
                "sides": {},
            }
        else:
            final_canonical_q, final_wrist_coaxial_metadata = (
                self._canonicalize_wrist_coaxial_solution(
                    reference_full_q=pre_solve_q,
                    candidate_full_q=active_q,
                )
            )
        final_canonicalization_changed = not np.allclose(
            final_canonical_q,
            active_q,
            rtol=0.0,
            atol=1e-12,
        )
        active_q = final_canonical_q
        left_qpos, right_qpos = self._split_solution(active_q)
        self._last_wrist_coaxial_metadata = final_wrist_coaxial_metadata
        output_qpos_smoothing["post_smoothing_wrist_canonicalization"] = dict(
            final_wrist_coaxial_metadata
        )
        if final_canonicalization_changed:
            min_distance_m = self._profiled_min_collision_distance(
                active_q,
                profile,
                phase="post_smoothing_wrist_canonicalization",
            )
            final_canonical_fallback = self._fallback_reason_for_distance(
                min_distance_m
            )
            if (
                final_canonical_fallback is not None
                and self.config.hold_last_safe_on_failure
            ):
                active_q = pre_solve_q.copy()
                left_qpos, right_qpos = self._split_solution(active_q)
                status = "hold_last_safe"
                fallback_reason = final_canonical_fallback
            else:
                near_boundary = (
                    min_distance_m is not None
                    and min_distance_m
                    <= float(self.config.collision_min_distance_m)
                    + float(self.config.collision_distance_epsilon_m)
                )
                status = "boundary_active" if near_boundary else "tracking"
                fallback_reason = None
        q5_candidate_metadata = candidate_selection_metadata.get(
            "singularity_aware_wrist_q5_candidate_stabilization"
        )
        q5_output_metadata = output_qpos_smoothing.get("wrist_q5_escape")
        q5_hysteresis_latch_metadata = (
            self._update_wrist_q5_hysteresis_escape_latch(
                reference_full_q=pre_solve_q,
                admitted_full_q=active_q,
                status=status,
                candidate_metadata=(
                    q5_candidate_metadata
                    if isinstance(q5_candidate_metadata, Mapping)
                    else None
                ),
                output_metadata=(
                    q5_output_metadata
                    if isinstance(q5_output_metadata, Mapping)
                    else None
                ),
            )
        )
        output_qpos_smoothing["wrist_q5_hysteresis_latch"] = (
            q5_hysteresis_latch_metadata
        )
        joint_limit_escape_hysteresis_metadata = (
            self._update_joint_limit_escape_direction_latch(
                reference_full_q=pre_solve_q,
                admitted_full_q=active_q,
                raw_full_q=(
                    raw_solver_q.copy()
                    if raw_solver_q is not None
                    else None
                ),
                status=status,
                selection_metadata=(
                    candidate_selection_metadata
                    if isinstance(candidate_selection_metadata, Mapping)
                    else None
                ),
            )
        )
        output_qpos_smoothing["joint_limit_escape_hysteresis"] = (
            joint_limit_escape_hysteresis_metadata
        )
        if (
            output_qpos_smoothing.get("enabled", False)
            or final_canonicalization_changed
            or bool(
                isinstance(output_qpos_smoothing.get("wrist_q5_escape"), Mapping)
                and output_qpos_smoothing["wrist_q5_escape"].get("applied", False)
            )
        ):
            self.ik_solver.set_configuration_q(active_q)
            self._last_safe_full_q = active_q.copy()
            # The smoother's internal state must describe the qref that is
            # actually admitted after the final singularity-aware projection.
            self._last_output_qpos_smoothing_full_q = active_q.copy()
        admitted_velocity_metadata = self._commit_admitted_velocity(
            reference_full_q=pre_solve_q,
            admitted_full_q=active_q,
        )
        self._record_ik_shadow_trace(
            monotonic_time=time.monotonic(),
            pre_solve_q=pre_solve_q,
            raw_solver_q=raw_solver_q,
            canonical_solver_q=canonical_solver_q,
            admitted_q=active_q,
            current_targets=current_targets,
            status=status,
        )
        if self.config.position_singularity_candidate_certificate_enabled and (
            candidate_selection_metadata
            or self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        ):
            # ``current_pose_hold`` has no branch-selection payload, but it is
            # still a formal position qref.  The certified single-step path
            # must keep carrying its qref/traversal certificate while the
            # robot remains in the wrist coaxial band; otherwise the backend
            # would clear the fallback latch on an otherwise harmless no-op
            # frame and immediately re-enter SDK hold.
            if not isinstance(candidate_selection_metadata, dict):
                candidate_selection_metadata = {}
            self._candidate_certificate_sequence += 1
            position_preference = dict(
                candidate_selection_metadata.get(
                    "position_singularity_preference",
                    {},
                )
                or {}
            )
            wrist_preference = dict(
                candidate_selection_metadata.get(
                    "wrist_singularity_preference",
                    {},
                )
                or {}
            )
            singular_fallback_claimed = bool(
                position_preference.get("fallback_to_singular", False)
                or wrist_preference.get("fallback_to_singular", False)
            )
            wrist_metadata_sides = self._last_wrist_coaxial_metadata.get(
                "sides",
                {},
            )
            wrist_metadata_sides = (
                wrist_metadata_sides
                if isinstance(wrist_metadata_sides, Mapping)
                else {}
            )
            singularity_aware_wrist_traversal_by_side: dict[str, bool] = {}
            for side in ("left", "right"):
                side_metadata = wrist_metadata_sides.get(side, {})
                side_metadata = (
                    side_metadata
                    if isinstance(side_metadata, Mapping)
                    else {}
                )
                null_scale = side_metadata.get("null_scale")
                try:
                    within_coaxial_band = (
                        null_scale is not None
                        and np.isfinite(float(null_scale))
                        and float(null_scale) < 1.0 - 1e-12
                    )
                except (TypeError, ValueError):
                    within_coaxial_band = False
                singularity_aware_wrist_traversal_by_side[side] = bool(
                    self.config.solve_step_mode
                    == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                    and bool(
                        self._last_wrist_coaxial_metadata.get(
                            "enabled",
                            False,
                        )
                    )
                    and status
                    not in {
                        "jump_gate_hold",
                        "hold_last_safe",
                        "unsafe_candidate",
                    }
                    and within_coaxial_band
                )
            elbow_traversal_metadata = self._singularity_escape_traversal_metadata(
                reference_full_q=pre_solve_q,
                admitted_full_q=active_q,
                selection_metadata=candidate_selection_metadata,
            )
            singularity_aware_elbow_traversal_by_side = {
                side: bool(
                    dict(elbow_traversal_metadata.get("by_side", {})).get(
                        side, False
                    )
                )
                for side in ("left", "right")
            }
            candidate_selected_singular_fallback_claimed = (
                singular_fallback_claimed
            )
            certificate: dict[str, object] = {
                "schema": "robot_motion.rm65.position_candidate_certificate/v1",
                "sequence": int(self._candidate_certificate_sequence),
                "qref_fingerprint_schema": POSITION_QREF_FINGERPRINT_SCHEMA,
                "qref_fingerprint_by_side": {
                    "left": position_qref_fingerprint(left_qpos),
                    "right": position_qref_fingerprint(right_qpos),
                },
                "selected_branch": str(branch_candidate),
                "selection_status": candidate_selection_metadata.get("status"),
                "selection_reason": candidate_selection_metadata.get("reason"),
                "singular_fallback_claimed": singular_fallback_claimed,
                "candidate_selected_singular_fallback_claimed": (
                    candidate_selected_singular_fallback_claimed
                ),
                "singularity_aware_wrist_traversal_by_side": dict(
                    singularity_aware_wrist_traversal_by_side
                ),
                "singularity_aware_elbow_traversal_by_side": dict(
                    singularity_aware_elbow_traversal_by_side
                ),
                "singularity_aware_elbow_traversal": dict(
                    elbow_traversal_metadata
                ),
                "singularity_aware_wrist_traversal_mode": (
                    str(self.config.solve_step_mode)
                ),
            }
            if singular_fallback_claimed or any(
                singularity_aware_wrist_traversal_by_side.values()
            ) or any(singularity_aware_elbow_traversal_by_side.values()):
                # The SDK-backed shoulder classification is intentionally
                # repeated only on the rare fallback path.  Normal 60 Hz
                # tracking pays only the fixed 48-byte qref fingerprint cost.
                certificate["final_qref_position_singularity"] = (
                    self._position_singularity_metadata(active_q)
                )
                certificate["final_qref_wrist_singularity"] = (
                    self._wrist_singularity_metadata(active_q)
                )
            candidate_selection_metadata["candidate_certificate"] = certificate
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
                "ik_branch_continuity_anchor_frame": dict(
                    self._last_branch_continuity_anchor_metadata
                ),
                "ik_solve_iterations": int(solve_iteration_count),
                "ik_solve_step_mode": str(self.config.solve_step_mode),
                "ik_solve_iteration_mode": solve_iteration_mode,
                "ik_solve_iteration_max": int(self.config.solve_max_iterations),
                "ik_solve_iteration_min": int(self.config.solve_min_iterations),
                "ik_damping": float(self.config.damping),
                "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
                "ik_candidate_certification_reserve_ms": float(
                    self.config.candidate_certification_reserve_ms
                ),
                "ik_solve_iteration_budget_ms": max(
                    0.0,
                    float(self.config.solve_time_budget_ms or 0.0)
                    - float(self.config.solve_iteration_reserve_ms),
                ),
                "ik_solve_iteration_budget_exhausted": bool(
                    self._last_solve_iteration_budget_exhausted
                ),
                "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
                "ik_wrist_coaxial_continuity": dict(
                    self._last_wrist_coaxial_metadata
                ),
                "ik_directional_singularity": dict(
                    self._last_directional_singularity_diagnostics
                ),
                "ik_admitted_velocity_state": dict(admitted_velocity_metadata),
                "ik_solve_pose_error": {
                    side: dict(metrics) for side, metrics in solve_pose_error.items()
                },
                "ik_current_target_hold": dict(
                    current_target_hold_metadata
                ),
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
            "ik_solve_step_mode": str(self.config.solve_step_mode),
            "ik_solve_iteration_mode": solve_iteration_mode,
            "ik_solve_iteration_max": int(self.config.solve_max_iterations),
            "ik_solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_damping": float(self.config.damping),
            "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "ik_candidate_certification_reserve_ms": float(
                self.config.candidate_certification_reserve_ms
            ),
            "ik_solve_iteration_budget_ms": max(
                0.0,
                float(self.config.solve_time_budget_ms or 0.0)
                - float(self.config.solve_iteration_reserve_ms),
            ),
            "ik_solve_iteration_budget_exhausted": bool(
                self._last_solve_iteration_budget_exhausted
            ),
            "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
            "ik_wrist_coaxial_continuity": dict(
                self._last_wrist_coaxial_metadata
            ),
            "ik_directional_singularity": dict(
                self._last_directional_singularity_diagnostics.get(
                    "sides",
                    {},
                ).get("left", {})
                if isinstance(
                    self._last_directional_singularity_diagnostics.get("sides"),
                    dict,
                )
                else {}
            ),
            "ik_admitted_velocity_state": {
                "enabled": bool(admitted_velocity_metadata["enabled"]),
                "source": admitted_velocity_metadata["source"],
                "dt_s": admitted_velocity_metadata["dt_s"],
                "qdot_rad_s": admitted_velocity_metadata["left_qdot_rad_s"],
                "max_abs_qdot_rad_s": admitted_velocity_metadata[
                    "left_max_abs_qdot_rad_s"
                ],
            },
            "ik_solve_pose_error": dict(solve_pose_error.get("left", {})),
            "ik_current_target_hold": dict(current_target_hold_metadata),
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
            "ik_solve_step_mode": str(self.config.solve_step_mode),
            "ik_solve_iteration_mode": solve_iteration_mode,
            "ik_solve_iteration_max": int(self.config.solve_max_iterations),
            "ik_solve_iteration_min": int(self.config.solve_min_iterations),
            "ik_damping": float(self.config.damping),
            "ik_solve_iteration_reserve_ms": float(self.config.solve_iteration_reserve_ms),
            "ik_candidate_certification_reserve_ms": float(
                self.config.candidate_certification_reserve_ms
            ),
            "ik_solve_iteration_budget_ms": max(
                0.0,
                float(self.config.solve_time_budget_ms or 0.0)
                - float(self.config.solve_iteration_reserve_ms),
            ),
            "ik_solve_iteration_budget_exhausted": bool(
                self._last_solve_iteration_budget_exhausted
            ),
            "ik_solve_iteration_stop_reason": self._last_solve_iteration_stop_reason,
            "ik_wrist_coaxial_continuity": dict(
                self._last_wrist_coaxial_metadata
            ),
            "ik_directional_singularity": dict(
                self._last_directional_singularity_diagnostics.get(
                    "sides",
                    {},
                ).get("right", {})
                if isinstance(
                    self._last_directional_singularity_diagnostics.get("sides"),
                    dict,
                )
                else {}
            ),
            "ik_admitted_velocity_state": {
                "enabled": bool(admitted_velocity_metadata["enabled"]),
                "source": admitted_velocity_metadata["source"],
                "dt_s": admitted_velocity_metadata["dt_s"],
                "qdot_rad_s": admitted_velocity_metadata["right_qdot_rad_s"],
                "max_abs_qdot_rad_s": admitted_velocity_metadata[
                    "right_max_abs_qdot_rad_s"
                ],
            },
            "ik_solve_pose_error": dict(solve_pose_error.get("right", {})),
            "ik_current_target_hold": dict(current_target_hold_metadata),
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
                self._apply_target_override_activity_metadata(context)
        return self.build_command_for_context(context, targets=targets)

    def _apply_target_override_activity_metadata(
        self,
        context: DualArmTargetContext,
    ) -> None:
        """Project optional modifier motion intent into owned command copies."""

        provider = getattr(
            self.target_override_filter,
            "target_activity_override_by_side",
            None,
        )
        if not callable(provider):
            return
        raw_overrides = provider()
        if raw_overrides is None:
            return
        if not isinstance(raw_overrides, Mapping):
            raise TypeError(
                "target activity override provider must return a side mapping"
            )
        commands = {
            "left": context.left_cartesian,
            "right": context.right_cartesian,
        }
        for raw_side, raw_override in raw_overrides.items():
            side = self._validated_side(str(raw_side))
            if not isinstance(raw_override, Mapping):
                raise TypeError(
                    f"target activity override for {side} must be a mapping"
                )
            command = commands[side]
            if command is None:
                raise ValueError(
                    f"target activity override for {side} has no source command"
                )
            override = dict(raw_override)
            hold_weight = float(override.get("hold_weight", 0.0))
            if (
                not math.isfinite(hold_weight)
                or hold_weight < 0.0
                or hold_weight > 1.0
            ):
                raise ValueError(
                    f"target activity override for {side} has invalid hold_weight"
                )
            tracker_intent = command.metadata.get("tracker_motion_intent")
            if isinstance(tracker_intent, Mapping):
                # The modifier owns activity/hold authority, but the tracker
                # remains the source of per-frame motion-window quality used
                # by the frozen V2 output-smoothing profile.  Dropping these
                # fields silently forces Balanced from its ordinary 0.55 path
                # back to the legacy 0.14 endpoint while force is active.
                for key in ("evidence", "motion_window"):
                    value = tracker_intent.get(key)
                    if key not in override and isinstance(value, Mapping):
                        override[key] = dict(value)
            command.metadata["pose_action_motion_intent"] = override

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

    def _branch_continuity_anchor_joint_mask(
        self,
        *,
        reference_full_q: np.ndarray,
        target_activity: Mapping[str, object] | None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Disable only structural anchor rows inside the hard-limit escape band.

        The continuity task is useful for ordinary multi-branch IK, but it
        must not resist a safety transition away from a physical joint limit.
        The normal limit-escape selector remains authoritative; this mask only
        removes the soft q3/q5 preference for the affected side/joint.
        """

        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        mask = np.ones(reference.shape, dtype=bool)
        disabled: list[str] = []
        safety_profile = self.ik_safety_profile
        activation_margin_deg = (
            0.0
            if safety_profile is None
            else max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_activation_margin_deg",
                        2.0,
                    )
                ),
            )
        )
        activation_margin_rad = np.deg2rad(activation_margin_deg)
        lower = np.asarray(
            self.ik_solver.robot.model.lowerPositionLimit,
            dtype=float,
        ).reshape(reference.shape)
        upper = np.asarray(
            self.ik_solver.robot.model.upperPositionLimit,
            dtype=float,
        ).reshape(reference.shape)
        fixed_sides = self._authority_fixed_sides(target_activity)
        for side, local_index, full_index in self._arm_joint_entries(reference):
            if local_index not in {2, 4} or side in fixed_sides:
                continue
            if not np.isfinite(reference[full_index]):
                continue
            margin = min(
                float(reference[full_index] - lower[full_index]),
                float(upper[full_index] - reference[full_index]),
            )
            if margin <= activation_margin_rad + 1e-9:
                mask[full_index] = False
                disabled.append(f"{side}.q{local_index + 1}")
        return mask, tuple(disabled)

    def _solve_toward_targets(
        self,
        *,
        left_target: np.ndarray,
        right_target: np.ndarray,
        target_activity: Mapping[str, object] | None = None,
        profile: dict[str, object] | None = None,
        frame_start_time: float | None = None,
    ) -> tuple[np.ndarray, int, dict[str, dict[str, float]]]:
        targets = [
            self._pose_to_target(left_target),
            self._pose_to_target(right_target),
        ]
        initial_q = self.ik_solver.current_q.copy()
        last_q = initial_q.copy()
        self._last_raw_solver_q = None
        self._last_wrist_coaxial_metadata = {
            "enabled": False,
            "applied": False,
            "reason": "not_solved",
        }
        self._last_directional_singularity_diagnostics = (
            self._directional_singularity_diagnostics(
                current_full_q=initial_q,
            )
        )
        self._set_solver_motion_reference(initial_q)
        if self._branch_continuity_anchor_enabled:
            anchor_mask, disabled_for_limit_escape = (
                self._branch_continuity_anchor_joint_mask(
                    reference_full_q=initial_q,
                    target_activity=target_activity,
                )
            )
            set_branch_anchor = getattr(
                self.ik_solver,
                "set_branch_continuity_anchor_q",
            )
            set_branch_anchor(
                initial_q,
                active_joint_mask=anchor_mask,
            )
            self._last_branch_continuity_anchor_metadata = {
                "enabled": True,
                "active_joint_mask": anchor_mask.astype(bool).tolist(),
                "disabled_for_limit_escape": list(disabled_for_limit_escape),
            }
        else:
            self._last_branch_continuity_anchor_metadata = {
                "enabled": False,
                "active_joint_mask": None,
                "disabled_for_limit_escape": [],
            }
        pose_error: dict[str, dict[str, float]] = {}
        iterations = 0
        self._last_solve_iteration_budget_exhausted = False
        self._last_solve_iteration_stop_reason = None
        single_step = bool(
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        )
        adaptive = bool(
            self.config.solve_adaptive_iterations_enabled and not single_step
        )
        time_budget_ms = float(self.config.solve_time_budget_ms or 0.0)
        iteration_budget_ms = (
            max(0.0, time_budget_ms - float(self.config.solve_iteration_reserve_ms))
            if adaptive and time_budget_ms > 0.0
            else None
        )
        budget_start = time.perf_counter() if frame_start_time is None else float(frame_start_time)
        previous_solve_ms: float | None = None
        min_iterations = 1 if single_step else int(self.config.solve_min_iterations)
        max_iterations = 1 if single_step else int(self.config.solve_max_iterations)
        for iteration in range(1, max_iterations + 1):
            if iteration > min_iterations and iteration_budget_ms is not None:
                elapsed_ms = (time.perf_counter() - budget_start) * 1000.0
                estimated_next_ms = max(0.05, float(previous_solve_ms or 0.0))
                if elapsed_ms + estimated_next_ms > iteration_budget_ms:
                    self._last_solve_iteration_budget_exhausted = True
                    self._last_solve_iteration_stop_reason = "iteration_budget_estimate"
                    break
            solve_start = time.perf_counter()
            last_q = self.ik_solver.solve(targets=targets)
            self._last_raw_solver_q = np.asarray(last_q, dtype=float).copy()
            if single_step:
                last_q, self._last_wrist_coaxial_metadata = (
                    self._canonicalize_wrist_coaxial_solution(
                        reference_full_q=initial_q,
                        candidate_full_q=last_q,
                    )
                )
                self.ik_solver.set_configuration_q(last_q)
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
        if iterations >= max_iterations and self._last_solve_iteration_stop_reason is None:
            self._last_solve_iteration_stop_reason = "max_iterations"
        if profile is not None:
            profile["solve_step_mode"] = str(self.config.solve_step_mode)
            profile["solve_iteration_mode"] = (
                "single_physical_step"
                if single_step
                else "adaptive_budget"
                if adaptive
                else "fixed_max"
            )
            profile["solve_iteration_effective_max"] = int(max_iterations)
            profile["solve_iteration_effective_min"] = int(min_iterations)
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
        return embed_arm_qpos(
            solution_layout=self.profile.solution_layout,
            current_full_q=current_full_q,
            left_qpos=left_qpos,
            right_qpos=right_qpos,
            owner_name=self.name,
        )

    def _split_solution(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return split_arm_solution(
            solution_layout=self.profile.solution_layout,
            qpos=qpos,
            owner_name=self.name,
        )

    def _arm_joint_indices(self, vector_size: int) -> dict[str, np.ndarray]:
        return arm_joint_indices(
            solution_layout=self.profile.solution_layout,
            vector_size=vector_size,
            owner_name=self.name,
        )

    def _make_full_joint_acceleration_limit(
        self,
        arm_limit_rad_s2: np.ndarray,
    ) -> np.ndarray:
        arm_limit = np.asarray(arm_limit_rad_s2, dtype=float).reshape(-1)
        if arm_limit.shape != (6,) or not np.all(np.isfinite(arm_limit)):
            raise ValueError("arm acceleration limits must contain 6 finite values.")
        if np.any(arm_limit <= 0.0):
            raise ValueError("arm acceleration limits must be positive.")
        current_q = np.asarray(self.ik_solver.current_q, dtype=float).reshape(-1)
        model_nv = int(getattr(self.ik_solver.robot.model, "nv", current_q.size))
        if current_q.size != model_nv:
            raise ValueError(
                "single-step acceleration constraints currently require nq == nv; "
                f"got nq={current_q.size} nv={model_nv}."
            )
        full_limit = np.full(model_nv, np.inf, dtype=float)
        for indices in self._arm_joint_indices(model_nv).values():
            full_limit[indices] = arm_limit
        return full_limit

    def _canonicalize_wrist_coaxial_solution(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Remove only the unobservable q4-q6 redistribution near q5 == 0.

        For the RM65 spherical wrist, q4 and q6 become coaxial at q5 == 0 and
        the observable combination is q4 + q6.  This projection keeps that sum
        exactly unchanged while continuously suppressing the null-direction
        increment q4 - q6.  It is therefore not a per-joint clip and does not
        alter the first-order wrist motion that remains observable.
        """

        metadata: dict[str, object] = {
            "enabled": bool(
                self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
                and self.config.wrist_coaxial_continuity_enabled
            ),
            "applied": False,
            "observable_combination": "q4_plus_q6",
            "null_combination": "q4_minus_q6",
            "soft_threshold_deg": float(
                self.config.wrist_coaxial_soft_threshold_deg
            ),
            "hard_threshold_deg": float(
                self.config.wrist_coaxial_hard_threshold_deg
            ),
            "null_motion_soft_limit_deg": float(
                self.config.wrist_coaxial_null_motion_soft_limit_deg
            ),
            "branch_switch_guard_deg": float(
                self.config.wrist_branch_switch_guard_deg
            ),
            "sides": {},
        }
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate = np.asarray(candidate_full_q, dtype=float).reshape(-1).copy()
        if not metadata["enabled"]:
            metadata["reason"] = "disabled_or_legacy_mode"
            return candidate, metadata
        if reference.shape != candidate.shape:
            raise ValueError(
                "reference and candidate configurations must have matching shapes."
            )

        side_metadata: dict[str, object] = {}
        reference_by_side = dict(zip(("left", "right"), self._split_solution(reference)))
        candidate_by_side = dict(zip(("left", "right"), self._split_solution(candidate)))
        canonical_by_side: dict[str, np.ndarray] = {}
        any_applied = False
        for side in ("left", "right"):
            ref = reference_by_side[side]
            raw = candidate_by_side[side]
            out, side_result = canonicalize_rm65_wrist_coaxial_step(
                ref,
                raw,
                soft_threshold_deg=float(
                    self.config.wrist_coaxial_soft_threshold_deg
                ),
                hard_threshold_deg=float(
                    self.config.wrist_coaxial_hard_threshold_deg
                ),
                null_motion_soft_limit_deg=float(
                    self.config.wrist_coaxial_null_motion_soft_limit_deg
                ),
            )
            applied = bool(side_result["applied"])
            any_applied = any_applied or applied
            canonical_by_side[side] = out
            side_metadata[side] = side_result
        candidate = self._embed_arm_qpos(
            current_full_q=candidate,
            left_qpos=canonical_by_side["left"],
            right_qpos=canonical_by_side["right"],
        )
        metadata["applied"] = any_applied
        metadata["reason"] = "coaxial_projection" if any_applied else "outside_soft_band"
        metadata["sides"] = side_metadata
        return candidate, metadata

    def _directional_singularity_diagnostics(
        self,
        *,
        current_full_q: np.ndarray,
    ) -> dict[str, object]:
        if (
            self.config.solve_step_mode
            != IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
        ):
            return {"enabled": False, "reason": "legacy_mode"}
        compute_jacobian = getattr(self.ik_solver, "compute_frame_jacobian", None)
        if not callable(compute_jacobian):
            return {"enabled": False, "reason": "solver_jacobian_unavailable"}
        q = np.asarray(current_full_q, dtype=float).reshape(-1)
        try:
            indices_by_side = self._arm_joint_indices(q.size)
            frames = {
                "left": self.profile.left_frame_name,
                "right": self.profile.right_frame_name,
            }
            result: dict[str, object] = {
                "enabled": True,
                "advisory_only": True,
                "jacobian_scaling": "raw_pinocchio_unscaled",
                "threshold_classification": "disabled",
                "sides": {},
            }
            sides: dict[str, object] = {}
            for side in ("left", "right"):
                jacobian_full = np.asarray(
                    compute_jacobian(frames[side], q),
                    dtype=float,
                )
                jacobian = jacobian_full[:, indices_by_side[side]]
                singular_values = np.linalg.svd(
                    jacobian,
                    compute_uv=False,
                )
                sigma_min = float(np.min(singular_values))
                sigma_max = float(np.max(singular_values))
                sides[side] = {
                    "singular_values": singular_values.astype(float).tolist(),
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "condition_number": (
                        float("inf")
                        if sigma_min <= 1e-12
                        else float(sigma_max / sigma_min)
                    ),
                }
            result["sides"] = sides
            return result
        except Exception as exc:  # diagnostics must never change admission.
            return {
                "enabled": False,
                "reason": f"diagnostic_exception:{type(exc).__name__}",
            }

    def _commit_admitted_velocity(
        self,
        *,
        reference_full_q: np.ndarray,
        admitted_full_q: np.ndarray,
    ) -> dict[str, object]:
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        admitted = np.asarray(admitted_full_q, dtype=float).reshape(-1)
        velocity = (admitted - reference) / float(self._execution_dt_sec)
        self._last_admitted_velocity_full = velocity.copy()
        self._set_solver_motion_reference(admitted)
        left_velocity, right_velocity = self._split_solution(velocity)
        return {
            "enabled": bool(self._acceleration_limit_enabled),
            "admitted_reference_motion_limit_enabled": bool(
                self._admitted_reference_motion_limit_enabled
            ),
            "source": "final_admitted_qref",
            "dt_s": float(self._execution_dt_sec),
            "left_qdot_rad_s": left_velocity.astype(float).tolist(),
            "right_qdot_rad_s": right_velocity.astype(float).tolist(),
            "left_max_abs_qdot_rad_s": float(np.max(np.abs(left_velocity))),
            "right_max_abs_qdot_rad_s": float(np.max(np.abs(right_velocity))),
        }

    def _record_ik_shadow_trace(
        self,
        *,
        monotonic_time: float,
        pre_solve_q: np.ndarray,
        raw_solver_q: np.ndarray | None,
        canonical_solver_q: np.ndarray | None,
        admitted_q: np.ndarray,
        current_targets: Mapping[str, np.ndarray],
        status: str,
    ) -> None:
        sink = self._ik_shadow_trace_sink
        if sink is None:
            return

        def arm_pair(q: np.ndarray | None) -> np.ndarray | None:
            if q is None:
                return None
            left, right = self._split_solution(np.asarray(q, dtype=float))
            return np.concatenate((left, right))

        directional_sides = self._last_directional_singularity_diagnostics.get(
            "sides",
            {},
        )
        directional_sides = (
            directional_sides if isinstance(directional_sides, Mapping) else {}
        )
        wrist_sides = self._last_wrist_coaxial_metadata.get("sides", {})
        wrist_sides = wrist_sides if isinstance(wrist_sides, Mapping) else {}
        try:
            sink(
                float(monotonic_time),
                arm_pair(pre_solve_q),
                arm_pair(raw_solver_q),
                arm_pair(canonical_solver_q),
                arm_pair(admitted_q),
                np.asarray(current_targets["left"], dtype=float),
                np.asarray(current_targets["right"], dtype=float),
                (
                    directional_sides.get("left", {}).get("singular_values")
                    if isinstance(directional_sides.get("left"), Mapping)
                    else None
                ),
                (
                    directional_sides.get("right", {}).get("singular_values")
                    if isinstance(directional_sides.get("right"), Mapping)
                    else None
                ),
                (
                    (
                        wrist_sides.get("left", {}).get("effective_null_scale")
                        if wrist_sides.get("left", {}).get("effective_null_scale")
                        is not None
                        else wrist_sides.get("left", {}).get("null_scale")
                    )
                    if isinstance(wrist_sides.get("left"), Mapping)
                    else None
                ),
                (
                    (
                        wrist_sides.get("right", {}).get("effective_null_scale")
                        if wrist_sides.get("right", {}).get("effective_null_scale")
                        is not None
                        else wrist_sides.get("right", {}).get("null_scale")
                    )
                    if isinstance(wrist_sides.get("right"), Mapping)
                    else None
                ),
                str(status),
            )
        except Exception:
            # Debug tracing is strictly observational and cannot affect
            # admission, collision handling, or command dispatch.
            return

    def _smooth_output_qpos_before_action(
        self,
        *,
        active_q: np.ndarray,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        reference_full_q: np.ndarray | None = None,
        target_activity: dict[str, object] | None = None,
        current_targets: dict[str, np.ndarray] | None = None,
        admission_status: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        alpha_by_side, alpha_metadata = self._effective_output_qpos_smoothing_alpha(
            target_activity
        )
        raw_full_q = self._embed_arm_qpos(
            current_full_q=active_q,
            left_qpos=np.asarray(left_qpos, dtype=float).reshape(6),
            right_qpos=np.asarray(right_qpos, dtype=float).reshape(6),
        )
        metadata: dict[str, object] = {
            "enabled": False,
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
        previous = self._last_output_qpos_smoothing_full_q
        smoothing_state_missing = previous is None
        if previous is not None and previous.shape != raw_full_q.shape:
            previous = None
            smoothing_state_missing = True
        if previous is None:
            previous = (
                None
                if reference_full_q is None
                else np.asarray(reference_full_q, dtype=float).reshape(raw_full_q.shape)
            )
        if previous is None:
            previous = raw_full_q.copy()

        # A slow (0.14 by default) post-IK EMA is useful for ordinary tracker
        # noise, but it becomes visibly sticky when the q5 escape path is
        # active: the smoother keeps feeding a stale wrist target while the
        # canonicalizer deliberately holds the observable q4+q6 coordinate.
        # Raise only the affected side to the already configured fast endpoint
        # (or an explicit singularity alpha).  The upstream one-step and
        # downstream transport limits remain unchanged, so this is a tracking
        # responsiveness change, not a speed-limit bypass.
        singularity_band_threshold_deg = max(
            float(self.config.wrist_singularity_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
            float(self.config.wrist_coaxial_hard_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
        )
        elbow_singularity_band_threshold_deg = max(
            float(self.config.elbow_singularity_threshold_deg)
            + float(self.config.elbow_singularity_selection_margin_deg),
            float(self.config.elbow_singularity_threshold_deg),
        )
        singularity_alpha = (
            float(self.config.output_qpos_smoothing_singularity_alpha)
            if self.config.output_qpos_smoothing_singularity_alpha is not None
            else float(self.config.output_qpos_smoothing_fast_alpha)
        )
        previous_by_side = dict(zip(("left", "right"), self._split_solution(previous)))
        raw_target_by_side = {
            "left": np.asarray(left_qpos, dtype=float).reshape(6),
            "right": np.asarray(right_qpos, dtype=float).reshape(6),
        }
        previous_admitted_step_by_side = dict(
            zip(
                ("left", "right"),
                self._split_solution(
                    np.asarray(
                        self._last_admitted_velocity_full,
                        dtype=float,
                    )
                    * float(self._execution_dt_sec)
                ),
            )
        )
        admittance_micro_reversal_hold_deg = float(
            self.config.output_qpos_admittance_micro_reversal_hold_deg
        )
        admittance_micro_reversal_creep_deg = float(
            self.config.output_qpos_admittance_micro_reversal_creep_deg
        )
        singularity_boost_by_side: dict[str, dict[str, object]] = {}
        normal_fast_path_by_side: dict[str, dict[str, object]] = {}
        admittance_fast_path_by_side: dict[str, dict[str, object]] = {}
        wrist_metadata_sides = self._last_wrist_coaxial_metadata.get("sides", {})
        wrist_metadata_sides = (
            wrist_metadata_sides if isinstance(wrist_metadata_sides, Mapping) else {}
        )
        singularity_mode_enabled = bool(
            self.config.output_command_type == "joint_vector"
            and self.config.solve_step_mode == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
        )
        normal_fast_now = time.monotonic()
        for side in ("left", "right"):
            reference_q5 = float(previous_by_side[side][4])
            candidate_q5 = float(raw_target_by_side[side][4])
            reference_q3 = float(previous_by_side[side][2])
            candidate_q3 = float(raw_target_by_side[side][2])
            near_wrist_band = bool(
                abs(np.rad2deg(reference_q5)) <= singularity_band_threshold_deg
                or abs(np.rad2deg(candidate_q5)) <= singularity_band_threshold_deg
                or reference_q5 * candidate_q5 < 0.0
            )
            near_elbow_band = bool(
                abs(np.rad2deg(reference_q3)) <= elbow_singularity_band_threshold_deg
                or abs(np.rad2deg(candidate_q3)) <= elbow_singularity_band_threshold_deg
            )
            near_band = bool(near_wrist_band or near_elbow_band)
            before_alpha = float(alpha_by_side[side])
            adaptive_side = alpha_metadata.get(side, {})
            adaptive_side = (
                adaptive_side if isinstance(adaptive_side, Mapping) else {}
            )
            wrist_side = wrist_metadata_sides.get(side, {})
            wrist_side = wrist_side if isinstance(wrist_side, Mapping) else {}
            effective_null_scale = wrist_side.get(
                "effective_null_scale",
                wrist_side.get("null_scale"),
            )
            null_projection_inactive = bool(
                effective_null_scale is not None
                and np.isfinite(float(effective_null_scale))
                and float(effective_null_scale) >= 1.0 - 1e-9
            )
            normal_guard_deg = float(
                self.config.output_qpos_normal_fast_path_guard_deg
            )
            outside_normal_guard = bool(
                min(
                    abs(float(np.rad2deg(reference_q3))),
                    abs(float(np.rad2deg(candidate_q3))),
                    abs(float(np.rad2deg(reference_q5))),
                    abs(float(np.rad2deg(candidate_q5))),
                )
                >= normal_guard_deg
            )
            tracker_quality_valid = bool(adaptive_side.get("quality_valid", False))
            if (
                self.config.output_qpos_normal_fast_path_enabled
                and tracker_quality_valid
            ):
                self._output_qpos_normal_fast_path_last_quality_valid_time[side] = (
                    normal_fast_now
                )
            last_quality_valid_time = (
                self._output_qpos_normal_fast_path_last_quality_valid_time[side]
            )
            tracker_quality_age_sec = (
                None
                if last_quality_valid_time is None
                else max(0.0, normal_fast_now - last_quality_valid_time)
            )
            tracker_quality_grace_sec = float(
                self.config.output_qpos_normal_fast_path_quality_grace_sec
            )
            tracker_quality_grace_active = bool(
                not tracker_quality_valid
                and tracker_quality_age_sec is not None
                and tracker_quality_grace_sec > 0.0
                and tracker_quality_age_sec <= tracker_quality_grace_sec
            )
            tracker_quality_valid_or_grace = bool(
                tracker_quality_valid or tracker_quality_grace_active
            )
            q5_hysteresis_latch_clear = not bool(
                self._wrist_q5_hysteresis_escape_latch[side]
            )
            normal_fast_eligible = bool(
                self.config.output_qpos_normal_fast_path_enabled
                and singularity_mode_enabled
                and str(admission_status) == "tracking"
                and tracker_quality_valid_or_grace
                and outside_normal_guard
                and null_projection_inactive
                and q5_hysteresis_latch_clear
            )
            if normal_fast_eligible:
                alpha_by_side[side] = float(
                    self.config.output_qpos_normal_fast_path_alpha
                )
                # The adaptive alpha is evaluated before this opt-in override.
                # Keep its state aligned with the alpha that was actually
                # admitted; otherwise one soft dropout uses a stale low alpha
                # and the following frame snaps back to the fast target.
                self._output_qpos_smoothing_effective_alpha[side] = float(
                    alpha_by_side[side]
                )
                self._output_qpos_smoothing_update_time[side] = normal_fast_now
            normal_fast_path_by_side[side] = {
                "enabled": bool(self.config.output_qpos_normal_fast_path_enabled),
                "eligible": normal_fast_eligible,
                "admission_status": admission_status,
                "tracker_quality_valid": tracker_quality_valid,
                "tracker_quality_valid_or_grace": tracker_quality_valid_or_grace,
                "tracker_quality_grace_active": tracker_quality_grace_active,
                "tracker_quality_grace_sec": tracker_quality_grace_sec,
                "tracker_quality_age_sec": tracker_quality_age_sec,
                "outside_joint_guard": outside_normal_guard,
                "guard_deg": normal_guard_deg,
                "null_projection_inactive": null_projection_inactive,
                "effective_null_scale": effective_null_scale,
                "q5_hysteresis_latch_clear": q5_hysteresis_latch_clear,
                "alpha_before": before_alpha,
                "alpha_after": float(alpha_by_side[side]),
            }
            side_activity = (
                target_activity.get(side)
                if isinstance(target_activity, Mapping)
                else None
            )
            side_activity = (
                side_activity if isinstance(side_activity, Mapping) else {}
            )
            modifier_active = bool(
                str(side_activity.get("source", ""))
                == "position_admittance_active_modifier"
                and not bool(side_activity.get("inactive", False))
                and bool(side_activity.get("modifier_active", True))
            )
            rotational_modifier = bool(
                side_activity.get("modifier_rotation_offset_norm_rad") is not None
            )
            # The modifier owns its own offset/slew/stale admission and may be
            # active while the operator's tracker is intentionally still, so
            # tracker-speed quality is not an eligibility input here.  Keep
            # every hard solver guard: tracking admission, ordinary joint
            # region, inactive null projection, and a clear q5 latch.
            admittance_fast_eligible = bool(
                self.config.output_qpos_admittance_fast_path_enabled
                and modifier_active
                and singularity_mode_enabled
                and str(admission_status) == "tracking"
                and outside_normal_guard
                and null_projection_inactive
                and q5_hysteresis_latch_clear
            )
            alpha_before_admittance = float(alpha_by_side[side])
            requested_admittance_alpha = float(
                self.config.output_qpos_admittance_fast_path_alpha
            )
            if (
                self.config.output_qpos_admittance_fast_path_enabled
                and modifier_active
            ):
                requested_activity_alpha = side_activity.get(
                    "modifier_qpos_fast_path_alpha"
                )
                requested_admittance_alpha = float(
                    self.config.output_qpos_admittance_fast_path_alpha
                    if requested_activity_alpha is None
                    else requested_activity_alpha
                )
                if (
                    not np.isfinite(requested_admittance_alpha)
                    or requested_admittance_alpha
                    < float(self.config.output_qpos_smoothing_alpha)
                    or requested_admittance_alpha
                    > float(self.config.output_qpos_admittance_fast_path_alpha)
                    + 1e-12
                ):
                    raise ValueError(
                        "position admittance requested qpos alpha is outside the "
                        "configured slow-to-fast envelope"
                    )
            if admittance_fast_eligible:
                alpha_by_side[side] = (
                    requested_admittance_alpha
                    if rotational_modifier
                    else max(
                        alpha_before_admittance,
                        requested_admittance_alpha,
                    )
                )
                self._output_qpos_smoothing_effective_alpha[side] = float(
                    alpha_by_side[side]
                )
                self._output_qpos_smoothing_update_time[side] = normal_fast_now
            else:
                self._reset_output_qpos_admittance_reversal_state(side)
            admittance_fast_path_by_side[side] = {
                "enabled": bool(
                    self.config.output_qpos_admittance_fast_path_enabled
                ),
                "eligible": admittance_fast_eligible,
                "modifier_active": modifier_active,
                "rotational_modifier": rotational_modifier,
                "modifier_phase": side_activity.get("modifier_phase"),
                "requested_alpha": requested_admittance_alpha,
                "configured_alpha_cap": float(
                    self.config.output_qpos_admittance_fast_path_alpha
                ),
                "tracker_quality_required": False,
                "micro_reversal_hold_enabled": bool(
                    admittance_fast_eligible
                    and admittance_micro_reversal_hold_deg > 0.0
                ),
                "micro_reversal_hold_threshold_deg": (
                    admittance_micro_reversal_hold_deg
                ),
                "micro_reversal_creep_deg": (
                    admittance_micro_reversal_creep_deg
                ),
                "micro_reversal_held_joint_indices": [],
                "micro_reversal_softened_joint_indices": [],
                "micro_reversal_pending_joint_indices": [],
                "hard_solver_guards_satisfied": bool(
                    singularity_mode_enabled
                    and str(admission_status) == "tracking"
                    and outside_normal_guard
                    and null_projection_inactive
                    and q5_hysteresis_latch_clear
                ),
                "alpha_before": alpha_before_admittance,
                "alpha_after": float(alpha_by_side[side]),
            }
            boosted = bool(singularity_mode_enabled and near_band and before_alpha < singularity_alpha)
            if boosted:
                alpha_by_side[side] = singularity_alpha
            singularity_boost_by_side[side] = {
                "enabled": singularity_mode_enabled,
                "near_band": near_band,
                "boosted": boosted,
                "reference_q5_deg": float(np.rad2deg(reference_q5)),
                "candidate_q5_deg": float(np.rad2deg(candidate_q5)),
                "reference_q3_deg": float(np.rad2deg(reference_q3)),
                "candidate_q3_deg": float(np.rad2deg(candidate_q3)),
                "near_wrist_band": near_wrist_band,
                "near_elbow_band": near_elbow_band,
                "band_threshold_deg": singularity_band_threshold_deg,
                "elbow_band_threshold_deg": elbow_singularity_band_threshold_deg,
                "alpha_before": before_alpha,
                "alpha_after": float(alpha_by_side[side]),
                "alpha_target": singularity_alpha,
            }
        enabled_by_side = {
            side: bool(
                self.config.output_command_type == "joint_vector"
                and (
                    alpha < 1.0
                    or (
                        bool(admittance_fast_path_by_side[side]["eligible"])
                        and admittance_micro_reversal_hold_deg > 0.0
                    )
                )
            )
            for side, alpha in alpha_by_side.items()
        }
        enabled = any(enabled_by_side.values())
        metadata["enabled"] = enabled
        metadata["alpha"] = max(alpha_by_side.values())
        metadata["alpha_by_side"] = dict(alpha_by_side)
        metadata["singularity_band_threshold_deg"] = singularity_band_threshold_deg
        metadata["singularity_smoothing"] = singularity_boost_by_side
        metadata["normal_fast_path"] = normal_fast_path_by_side
        metadata["admittance_fast_path"] = admittance_fast_path_by_side
        if self.config.output_command_type != "joint_vector":
            metadata["reason"] = "non_joint_vector_output"
            self._last_output_qpos_smoothing_full_q = None
            self._reset_output_qpos_admittance_reversal_state()
            return raw_full_q.copy(), left_qpos.copy(), right_qpos.copy(), metadata
        if not enabled:
            metadata["reason"] = "alpha_one"
            smoothed_full_q = raw_full_q.copy()
            metadata["initialized"] = bool(smoothing_state_missing)
        else:
            metadata["initialized"] = bool(smoothing_state_missing)
            previous = np.asarray(previous, dtype=float).reshape(raw_full_q.shape)
            previous_left, previous_right = self._split_solution(previous)
            smoothed_by_side: dict[str, np.ndarray] = {}
            for side, raw_qpos, previous_qpos in (
                ("left", left_qpos, previous_left),
                ("right", right_qpos, previous_right),
            ):
                proposed_step = float(alpha_by_side[side]) * (
                    np.asarray(raw_qpos, dtype=float).reshape(6) - previous_qpos
                )
                held_joint_indices: list[int] = []
                softened_joint_indices: list[int] = []
                pending = self._output_qpos_admittance_pending_reversal_direction[
                    side
                ]
                if bool(admittance_fast_path_by_side[side]["eligible"]):
                    threshold_rad = float(
                        np.deg2rad(admittance_micro_reversal_hold_deg)
                    )
                    previous_step = np.asarray(
                        previous_admitted_step_by_side[side],
                        dtype=float,
                    ).reshape(6)
                    for joint_index in range(6):
                        proposed = float(proposed_step[joint_index])
                        proposed_direction = float(np.sign(proposed))
                        pending_direction = float(pending[joint_index])
                        if pending_direction != 0.0:
                            # One repeated sample confirms a real direction
                            # change.  A return to the prior direction clears
                            # the pending reversal without ever emitting it.
                            pending[joint_index] = 0.0
                            continue
                        previous_value = float(previous_step[joint_index])
                        if (
                            threshold_rad > 0.0
                            and proposed_direction != 0.0
                            and previous_value * proposed < 0.0
                            and min(abs(previous_value), abs(proposed))
                            <= threshold_rad + 1e-12
                        ):
                            creep_rad = float(
                                np.deg2rad(admittance_micro_reversal_creep_deg)
                            )
                            proposed_step[joint_index] = math.copysign(
                                min(abs(proposed), creep_rad),
                                proposed,
                            )
                            pending[joint_index] = proposed_direction
                            if creep_rad <= 0.0:
                                held_joint_indices.append(joint_index)
                            else:
                                softened_joint_indices.append(joint_index)
                else:
                    pending.fill(0.0)
                smoothed_by_side[side] = previous_qpos + proposed_step
                admittance_fast_path_by_side[side][
                    "micro_reversal_held_joint_indices"
                ] = held_joint_indices
                admittance_fast_path_by_side[side][
                    "micro_reversal_softened_joint_indices"
                ] = softened_joint_indices
                admittance_fast_path_by_side[side][
                    "micro_reversal_pending_joint_indices"
                ] = np.flatnonzero(pending).astype(int).tolist()
            smoothed_left = smoothed_by_side["left"]
            smoothed_right = smoothed_by_side["right"]
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

        # The q4/q6 null projection cannot prevent a later qpos smoother from
        # blending q5 through the vendor SDK's wrist-singularity band. Apply a
        # directional, one-step q5 escape after smoothing and before the final
        # action FK. This path is enabled only for the explicit position-mode
        # single-step solver; legacy/impedance callers remain unchanged.
        smoothed_full_q, wrist_q5_escape = self._stabilize_wrist_q5_full_step(
            reference_full_q=np.asarray(previous, dtype=float).reshape(raw_full_q.shape),
            candidate_full_q=smoothed_full_q,
            current_targets=current_targets,
        )
        metadata["wrist_q5_escape"] = dict(wrist_q5_escape)
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
                "singularity_smoothing": dict(singularity_boost_by_side[side]),
                "admittance_fast_path": dict(
                    admittance_fast_path_by_side[side]
                ),
                "stage": "post_ik_pre_action_fk",
                "raw_target_joint_deg": np.rad2deg(raw_q).astype(float).tolist(),
                "output_target_joint_deg": np.rad2deg(out_q).astype(float).tolist(),
                "raw_to_output_delta_deg": np.rad2deg(out_q - raw_q).astype(float).tolist(),
                "raw_to_output_max_delta_deg": float(np.rad2deg(np.max(np.abs(out_q - raw_q)))),
            }
        return smoothed_full_q.copy(), smoothed_left, smoothed_right, metadata

    def _wrist_q5_far_target_sides(
        self,
        *,
        current_targets: dict[str, np.ndarray] | None,
        reference_full_q: np.ndarray,
    ) -> tuple[dict[str, bool], dict[str, object]]:
        """Return sides for which a far target should relax q5 hysteresis.

        This is an admission-side relief only.  It never changes the q5
        escape boundary, the execution velocity cap, collision checks, or the
        final jump gate.  If pose metrics are unavailable, the conservative
        fixed release band remains active.
        """

        sides = {"left": False, "right": False}
        metadata: dict[str, object] = {
            "enabled": bool(self.config.wrist_q5_hysteresis_relax_for_far_target),
            "position_threshold_m": float(
                self.config.wrist_q5_hysteresis_far_target_position_m
            ),
            "orientation_threshold_deg": float(
                self.config.wrist_q5_hysteresis_far_target_orientation_deg
            ),
            "sides": {},
        }
        if not metadata["enabled"] or current_targets is None:
            metadata["reason"] = "disabled_or_targets_unavailable"
            return sides, metadata
        try:
            metrics = self._dual_pose_metrics(
                np.asarray(reference_full_q, dtype=float).reshape(-1),
                current_targets,
            )
        except Exception as exc:  # noqa: BLE001 - relief is optional.
            metadata["reason"] = f"metrics_unavailable:{type(exc).__name__}"
            return sides, metadata
        position_threshold = float(self.config.wrist_q5_hysteresis_far_target_position_m)
        orientation_threshold = float(
            self.config.wrist_q5_hysteresis_far_target_orientation_deg
        )
        for side in ("left", "right"):
            side_metrics = dict(metrics.get(side, {}) or {})
            position_error = float(side_metrics.get("translation_m", 0.0) or 0.0)
            orientation_error = float(side_metrics.get("rotation_deg", 0.0) or 0.0)
            far = bool(
                position_error >= position_threshold
                or orientation_error >= orientation_threshold
            )
            sides[side] = far
            metadata["sides"][side] = {
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "far_target": far,
            }
        metadata["reason"] = "target_error"
        return sides, metadata

    def _stabilize_wrist_q5_full_step(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Keep the final position qref outside the RM65 wrist gate.

        This is intentionally separate from q4/q6 canonicalization: q5 is an
        observable wrist bend, so the only safe intervention is a small,
        signed task relaxation toward the previous side of the coaxial point.
        The existing execution profile supplies the per-cycle q5 cap.
        """

        enabled = bool(
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
        )
        metadata: dict[str, object] = {
            "enabled": enabled,
            "applied": False,
            "reason": "disabled_or_legacy_mode" if not enabled else "outside_escape_band",
            "escape_threshold_deg": None,
            "sides": {},
        }
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate = np.asarray(candidate_full_q, dtype=float).reshape(-1).copy()
        if not enabled:
            return candidate, metadata

        # The SDK gate is configured at 3 deg in the real position backend.
        # Keep a small explicit margin while respecting a user-specified
        # coaxial hard threshold if it is larger.  The previous fixed 1 deg
        # margin made the final qref stick at 4 deg even for benign motion;
        # the default 0.5 deg margin leaves the SDK's 3 deg gate protected
        # without widening the recovery band unnecessarily.
        escape_threshold_deg = max(
            float(self.config.wrist_singularity_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
            float(self.config.wrist_coaxial_hard_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
        )
        metadata["escape_threshold_deg"] = escape_threshold_deg
        metadata["hysteresis_requires_escape_latch"] = bool(
            self.config.wrist_q5_hysteresis_requires_escape_latch
        )
        step_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
        if self._full_execution_joint_step is not None:
            left_step, right_step = self._split_solution(self._full_execution_joint_step)
            step_by_side = {"left": left_step, "right": right_step}
        reference_by_side = dict(zip(("left", "right"), self._split_solution(reference)))
        candidate_by_side = dict(zip(("left", "right"), self._split_solution(candidate)))
        far_target_sides, far_target_metadata = self._wrist_q5_far_target_sides(
            current_targets=current_targets,
            reference_full_q=reference,
        )
        metadata["far_target_hysteresis_relaxation"] = far_target_metadata
        output_by_side: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            max_step = step_by_side[side]
            latch_armed = bool(self._wrist_q5_hysteresis_escape_latch[side])
            hysteresis_allowed = bool(
                not self.config.wrist_q5_hysteresis_requires_escape_latch
                or latch_armed
            )
            release_threshold = (
                None
                if far_target_sides.get(side, False) or not hysteresis_allowed
                else (
                    None
                    if float(self.config.wrist_q5_hysteresis_release_deg) <= 0.0
                    else max(
                        escape_threshold_deg,
                        float(self.config.wrist_q5_hysteresis_release_deg),
                    )
                )
            )
            output_side, side_metadata = stabilize_rm65_wrist_q5_step(
                reference_by_side[side],
                candidate_by_side[side],
                escape_threshold_deg=escape_threshold_deg,
                max_step_rad=None if max_step is None else float(max_step[4]),
                hysteresis_release_threshold_deg=release_threshold,
            )
            side_metadata["far_target_hysteresis_relaxed"] = bool(
                far_target_sides.get(side, False)
            )
            side_metadata["hysteresis_latch_required"] = bool(
                self.config.wrist_q5_hysteresis_requires_escape_latch
            )
            side_metadata["hysteresis_latch_armed"] = latch_armed
            side_metadata["hysteresis_allowed"] = hysteresis_allowed
            output_by_side[side] = output_side
            metadata["sides"][side] = side_metadata
        output = self._embed_arm_qpos(
            current_full_q=candidate,
            left_qpos=output_by_side["left"],
            right_qpos=output_by_side["right"],
        )
        metadata["applied"] = any(
            bool(side_metadata.get("applied", False))
            for side_metadata in metadata["sides"].values()
        )
        metadata["reason"] = (
            "q5_escape" if metadata["applied"] else "outside_escape_band"
        )
        return output, metadata

    def _stabilize_wrist_q5_candidate(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply q5 escape hysteresis before jump-gate admission.

        The post-smoothing q5 guard protects the emitted qref, but the
        candidate trace can still alternate between a raw near-boundary
        result and a one-step singularity-escape result.  Apply the same
        q5-only, step-capped rule to the selected candidate first so the
        canonical IK trace and the later qref smoother see one continuous
        side.  This is limited to the opt-in single-step position path and
        never changes q4/q6 or the ordinary/legacy branch.
        """

        enabled = bool(
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
        )
        metadata: dict[str, object] = {
            "enabled": enabled,
            "applied": False,
            "reason": "disabled_or_legacy_mode" if not enabled else "outside_escape_band",
            "escape_threshold_deg": None,
            "hysteresis_release_threshold_deg": None,
            "sides": {},
        }
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate = np.asarray(candidate_full_q, dtype=float).reshape(-1).copy()
        if not enabled:
            return candidate, metadata

        escape_threshold_deg = max(
            float(self.config.wrist_singularity_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
            float(self.config.wrist_coaxial_hard_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
        )
        release_threshold_deg = (
            None
            if float(self.config.wrist_q5_hysteresis_release_deg) <= 0.0
            else max(
                escape_threshold_deg,
                float(self.config.wrist_q5_hysteresis_release_deg),
            )
        )
        metadata["escape_threshold_deg"] = escape_threshold_deg
        metadata["hysteresis_release_threshold_deg"] = release_threshold_deg
        metadata["hysteresis_requires_escape_latch"] = bool(
            self.config.wrist_q5_hysteresis_requires_escape_latch
        )
        step_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
        if self._full_execution_joint_step is not None:
            left_step, right_step = self._split_solution(self._full_execution_joint_step)
            step_by_side = {"left": left_step, "right": right_step}
        reference_by_side = dict(zip(("left", "right"), self._split_solution(reference)))
        candidate_by_side = dict(zip(("left", "right"), self._split_solution(candidate)))
        far_target_sides, far_target_metadata = self._wrist_q5_far_target_sides(
            current_targets=current_targets,
            reference_full_q=reference,
        )
        metadata["far_target_hysteresis_relaxation"] = far_target_metadata
        output_by_side: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            max_step = step_by_side[side]
            latch_armed = bool(self._wrist_q5_hysteresis_escape_latch[side])
            hysteresis_allowed = bool(
                not self.config.wrist_q5_hysteresis_requires_escape_latch
                or latch_armed
            )
            side_release_threshold = (
                None
                if far_target_sides.get(side, False) or not hysteresis_allowed
                else release_threshold_deg
            )
            output_side, side_metadata = stabilize_rm65_wrist_q5_step(
                reference_by_side[side],
                candidate_by_side[side],
                escape_threshold_deg=escape_threshold_deg,
                max_step_rad=None if max_step is None else float(max_step[4]),
                hysteresis_release_threshold_deg=side_release_threshold,
            )
            side_metadata["far_target_hysteresis_relaxed"] = bool(
                far_target_sides.get(side, False)
            )
            side_metadata["hysteresis_latch_required"] = bool(
                self.config.wrist_q5_hysteresis_requires_escape_latch
            )
            side_metadata["hysteresis_latch_armed"] = latch_armed
            side_metadata["hysteresis_allowed"] = hysteresis_allowed
            output_by_side[side] = output_side
            metadata["sides"][side] = side_metadata
        output = self._embed_arm_qpos(
            current_full_q=candidate,
            left_qpos=output_by_side["left"],
            right_qpos=output_by_side["right"],
        )
        metadata["applied"] = bool(
            not np.allclose(output, candidate, rtol=0.0, atol=1e-12)
        )
        metadata["reason"] = (
            "q5_candidate_hysteresis_or_escape"
            if metadata["applied"]
            else "outside_escape_band"
        )
        return output, metadata

    def _update_wrist_q5_hysteresis_escape_latch(
        self,
        *,
        reference_full_q: np.ndarray,
        admitted_full_q: np.ndarray,
        status: str,
        candidate_metadata: Mapping[str, object] | None,
        output_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Commit q5 hysteresis state only from the final admitted qref.

        Candidate probes are intentionally insufficient: collision, jump, or
        final canonicalization may still reject them.  The latch is armed only
        when an explicit q5 escape correction is visible in the candidate or
        post-smoothing metadata *and* the final qref either reached the escape
        boundary or made real outward progress.  It is released after the
        admitted q5 leaves the configured release band.
        """

        required = bool(self.config.wrist_q5_hysteresis_requires_escape_latch)
        metadata: dict[str, object] = {
            "enabled": required,
            "status": str(status),
            "sides": {},
        }
        if not required:
            metadata["reason"] = "fixed_band_v1"
            return metadata

        reference_by_side = dict(
            zip(
                ("left", "right"),
                self._split_solution(np.asarray(reference_full_q, dtype=float).reshape(-1)),
            )
        )
        admitted_by_side = dict(
            zip(
                ("left", "right"),
                self._split_solution(np.asarray(admitted_full_q, dtype=float).reshape(-1)),
            )
        )
        escape_threshold_deg = max(
            float(self.config.wrist_singularity_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
            float(self.config.wrist_coaxial_hard_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
        )
        release_threshold_deg = max(
            escape_threshold_deg,
            float(self.config.wrist_q5_hysteresis_release_deg),
        )

        def side_reason(payload: Mapping[str, object] | None, side: str) -> str:
            if not isinstance(payload, Mapping):
                return ""
            sides = payload.get("sides")
            if not isinstance(sides, Mapping):
                return ""
            side_payload = sides.get(side)
            return (
                str(side_payload.get("reason") or "")
                if isinstance(side_payload, Mapping)
                else ""
            )

        admitted_status = str(status) in {"tracking", "boundary_active"}
        for side in ("left", "right"):
            before = bool(self._wrist_q5_hysteresis_escape_latch[side])
            reference_q5_deg = float(np.rad2deg(reference_by_side[side][4]))
            admitted_q5_deg = float(np.rad2deg(admitted_by_side[side][4]))
            reasons = (
                side_reason(candidate_metadata, side),
                side_reason(output_metadata, side),
            )
            explicit_escape = any(
                reason.startswith("escape_previous_side") for reason in reasons
            )
            outward_progress = bool(
                abs(admitted_q5_deg) > abs(reference_q5_deg) + 1e-9
            )
            reached_boundary = bool(
                abs(admitted_q5_deg) >= escape_threshold_deg - 1e-9
            )
            released = bool(
                admitted_status
                and abs(admitted_q5_deg) >= release_threshold_deg - 1e-9
            )
            armed = before
            transition = "unchanged"
            if released:
                armed = False
                transition = "released_outside_band"
            elif admitted_status and explicit_escape and (
                outward_progress or reached_boundary
            ):
                armed = True
                transition = "armed_from_admitted_escape"
            self._wrist_q5_hysteresis_escape_latch[side] = armed
            metadata["sides"][side] = {
                "before": before,
                "after": armed,
                "transition": transition,
                "reference_q5_deg": reference_q5_deg,
                "admitted_q5_deg": admitted_q5_deg,
                "candidate_reason": reasons[0],
                "output_reason": reasons[1],
                "explicit_escape": explicit_escape,
                "outward_progress": outward_progress,
                "reached_escape_boundary": reached_boundary,
                "release_threshold_deg": release_threshold_deg,
            }
        metadata["reason"] = "final_admitted_qref"
        return metadata

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

    def _make_full_execution_joint_velocity_limit(
        self,
        qpos: np.ndarray,
    ) -> np.ndarray:
        step = self._make_full_execution_joint_step(qpos)
        if step is None:
            raise ValueError(
                "admitted-reference motion limiting requires a finite execution "
                "joint-step contract."
            )
        velocity = np.full(np.asarray(step).shape, np.inf, dtype=float)
        active = np.asarray(step, dtype=float) > 0.0
        velocity[active] = np.asarray(step, dtype=float)[active] / float(
            self._execution_dt_sec
        )
        return velocity

    def _refresh_admitted_reference_motion_limit(self) -> None:
        if not self._admitted_reference_motion_limit_enabled:
            return
        configure = getattr(
            self.ik_solver,
            "configure_admitted_reference_motion_limit",
            None,
        )
        if not callable(configure):
            raise RuntimeError(
                "configured admitted-reference motion limit lost its configurator."
            )
        current_q = np.asarray(self.ik_solver.current_q, dtype=float).reshape(-1)
        configure(
            self._make_full_execution_joint_velocity_limit(current_q),
            self._make_full_joint_acceleration_limit(
                np.asarray(
                    self.config.joint_acceleration_limit_rad_s2,
                    dtype=float,
                )
            ),
        )
        self._set_solver_motion_reference(current_q)

    def _set_solver_motion_reference(self, anchor_full_q: np.ndarray) -> None:
        if not self._acceleration_limit_enabled:
            return
        if self._admitted_reference_motion_limit_enabled:
            setter = getattr(
                self.ik_solver,
                "set_admitted_reference_motion",
                None,
            )
            if not callable(setter):
                raise RuntimeError(
                    "configured admitted-reference motion limit lost its state setter."
                )
            setter(
                np.asarray(anchor_full_q, dtype=float).reshape(-1),
                self._last_admitted_velocity_full,
                float(self._execution_dt_sec),
            )
            return
        setter = getattr(self.ik_solver, "set_last_integration_velocity", None)
        if not callable(setter):
            raise RuntimeError(
                "configured acceleration-limited solver lost its velocity-state setter."
            )
        setter(self._last_admitted_velocity_full)

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
        self._refresh_admitted_reference_motion_limit()

    def set_execution_dt_sec(self, dt_sec: float) -> None:
        self._execution_dt_sec = max(1e-6, float(dt_sec))
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )
        self._refresh_admitted_reference_motion_limit()

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

    def _wrist_branch_switch_guard_active(
        self,
        *,
        raw_q: np.ndarray,
        reference_q: np.ndarray,
        candidate_q: np.ndarray | None = None,
        side: str | None = None,
    ) -> bool:
        """Return whether a wrist +/-pi branch switch must be suppressed.

        Pink's raw result is already a locally integrated configuration.  The
        explicit flip candidates are useful for recovery and for ordinary
        poses, but around the RM65 q5 coaxial neighborhood they are equivalent
        representations of the same tool orientation and can alternate when
        the candidate scorer applies the per-cycle clip.  Once clipped, the
        two representations are no longer equivalent and produce the large
        q4/q5/q6 oscillation visible in the IK shadow trace.  Keep the raw
        continuous branch while either endpoint is inside this guard band.

        The guard is deliberately opt-in and only applies to the new
        single-step position path.  A zero threshold is an exact rollback to
        the historical branch candidate set.
        """

        if not (
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
        ):
            return False
        guard_deg = float(self.config.wrist_branch_switch_guard_deg)
        if guard_deg <= 0.0:
            return False
        raw = np.asarray(raw_q, dtype=float).reshape(6)
        reference = np.asarray(reference_q, dtype=float).reshape(6)
        if min(
            abs(float(np.rad2deg(raw[4]))),
            abs(float(np.rad2deg(reference[4]))),
        ) <= guard_deg:
            return True

        # A branch candidate may start outside the guard on both endpoints,
        # then become a near-zero q5 configuration after the deliberately
        # enlarged local candidate clip (the formal position profile uses
        # 8x).  That was the remaining failure seen in the 2026-07-26 trace:
        # a +/-pi wrist flip was clipped to roughly q5=+4 deg, immediately
        # entering the SDK coaxial gate, and the next frame flipped back to
        # the continuous raw branch.  Evaluate the candidate *after the same
        # one-cycle candidate clip* and suppress only this hazardous crossing.
        if candidate_q is None:
            return False
        candidate = np.asarray(candidate_q, dtype=float).reshape(6)

        # A +/-pi wrist alternative can remain on the same q5 side after the
        # enlarged candidate clip, while still redistributing q4/q5/q6 by
        # tens of degrees.  The ordinary raw IK step is the continuity
        # reference: if it is small but the alternate's *clipped* wrist step
        # is much larger, suppress the alternate.  Deliberate, large operator
        # motion is left available (raw wrist motion above the guard), which
        # preserves an escape route instead of turning the guard into a
        # blanket hold.
        if (
            side in {"left", "right"}
            and bool(self.config.enforce_realistic_joint_step)
            and self._full_execution_joint_step is not None
        ):
            left_step, right_step = self._split_solution(
                np.asarray(self._full_execution_joint_step, dtype=float)
            )
            arm_step = left_step if str(side) == "left" else right_step
            max_step_rad = np.asarray(arm_step, dtype=float).reshape(6) * max(
                1.0,
                float(self.config.local_step_candidate_clip_scale),
            )
            raw_projected = reference + np.clip(
                raw - reference,
                -max_step_rad,
                max_step_rad,
            )
            candidate_projected = reference + np.clip(
                candidate - reference,
                -max_step_rad,
                max_step_rad,
            )
            wrist_indices = np.asarray((3, 4, 5), dtype=int)
            raw_wrist_step_deg = float(
                np.linalg.norm(np.rad2deg(raw_projected[wrist_indices] - reference[wrist_indices]))
            )
            candidate_wrist_step_deg = float(
                np.linalg.norm(
                    np.rad2deg(
                        candidate_projected[wrist_indices] - reference[wrist_indices]
                    )
                )
            )
            if (
                raw_wrist_step_deg <= guard_deg
                and candidate_wrist_step_deg > raw_wrist_step_deg + guard_deg
            ):
                return True

        reference_q5 = float(reference[4])
        candidate_q5 = float(candidate[4])
        if reference_q5 * candidate_q5 >= 0.0:
            return False
        if abs(float(np.rad2deg(candidate_q5))) <= guard_deg:
            return True
        if (
            not bool(self.config.enforce_realistic_joint_step)
            or self._full_execution_joint_step is None
        ):
            return False
        if side not in {"left", "right"}:
            return False
        left_step, right_step = self._split_solution(
            np.asarray(self._full_execution_joint_step, dtype=float)
        )
        # This helper is called with one arm's six-joint vector.  The branch
        # candidate is scored at scale=1 first; use the largest local clip
        # envelope so a candidate that can enter the guard in that primary
        # evaluation is never offered as an alternate seed.
        arm_step = left_step if str(side) == "left" else right_step
        max_step_rad = float(arm_step[4]) * max(
            1.0,
            float(self.config.local_step_candidate_clip_scale),
        )
        if not np.isfinite(max_step_rad) or max_step_rad <= 0.0:
            return False
        projected_q5 = reference_q5 + float(
            np.clip(candidate_q5 - reference_q5, -max_step_rad, max_step_rad)
        )
        return bool(abs(float(np.rad2deg(projected_q5))) <= guard_deg)

    def _wrist_branch_guard_output_crossing_sides(
        self,
        *,
        reference_full_q: np.ndarray,
        output_full_q: np.ndarray,
    ) -> tuple[str, ...]:
        """Find a guarded branch crossing after candidate-local clipping.

        Candidate selection intentionally uses a wider local clip than the
        final qref admission.  A mathematically equivalent wrist branch can
        therefore be clipped to the *opposite* side of q5=0 while still
        looking like a feasible alternate candidate to the pose scorer.
        Reject that alternate before canonicalization; the locally integrated
        original result remains available for deliberate operator crossings.
        Otherwise the canonical trace can contain a large q4/q5/q6 jump even
        though the final admitted qref is later rate-limited.
        """

        if not (
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
            and float(self.config.wrist_branch_switch_guard_deg) > 0.0
        ):
            return ()
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        output = np.asarray(output_full_q, dtype=float).reshape(reference.shape)
        guard_deg = float(self.config.wrist_branch_switch_guard_deg)
        reference_left, reference_right = self._split_solution(reference)
        output_left, output_right = self._split_solution(output)
        sides: list[str] = []
        for side, reference_arm, output_arm in (
            ("left", reference_left, output_left),
            ("right", reference_right, output_right),
        ):
            reference_q5 = float(reference_arm[4])
            output_q5 = float(output_arm[4])
            if not np.isfinite(reference_q5) or not np.isfinite(output_q5):
                continue
            if (
                abs(float(np.rad2deg(reference_q5))) > guard_deg
                and abs(float(np.rad2deg(output_q5))) <= guard_deg
                and reference_q5 * output_q5 <= 0.0
            ):
                sides.append(side)
        return tuple(sides)

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
        left_alt_candidate = preferred_wrist_flip_candidate(
            raw_left,
            reference_q=ref_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(
                self.ik_tuning.branch_normalization.limit_tolerance_rad
            ),
        )
        left_alt = (
            None
            if self._wrist_branch_switch_guard_active(
                raw_q=raw_left,
                reference_q=ref_left,
                candidate_q=left_alt_candidate,
                side="left",
            )
            else left_alt_candidate
        )
        right_alt_candidate = preferred_wrist_flip_candidate(
            raw_right,
            reference_q=ref_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(
                self.ik_tuning.branch_normalization.limit_tolerance_rad
            ),
        )
        right_alt = (
            None
            if self._wrist_branch_switch_guard_active(
                raw_q=raw_right,
                reference_q=ref_right,
                candidate_q=right_alt_candidate,
                side="right",
            )
            else right_alt_candidate
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
        if (
            self.config.prefer_non_elbow_singular_branch
            or self.config.prefer_non_wrist_singular_branch
            or self.config.prefer_non_shoulder_singular_branch
        ):
            ranked.sort(
                key=lambda item: (
                    int(self._position_singularity_metadata(item[1])["singular_side_count"]),
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
        injected analyzer is authoritative for shoulder singularity and also
        augments the two joint proxies when available.  The analyzer boundary
        is vendor-neutral so this shared solver remains usable in simulation.
        """

        normalized_q = np.asarray(full_qpos, dtype=float).reshape(-1)
        cache_key = (
            float(self.config.prefer_non_elbow_singular_branch),
            float(self.config.prefer_non_wrist_singular_branch),
            float(self.config.prefer_non_shoulder_singular_branch),
            float(self.config.elbow_singularity_threshold_deg),
            float(self.config.elbow_singularity_selection_margin_deg),
            float(self.config.wrist_singularity_threshold_deg),
            float(self.config.shoulder_singularity_threshold_m),
            *(float(value) for value in normalized_q),
        )
        cached = self._position_singularity_metadata_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        left_q, right_q = self._split_solution(normalized_q)
        wrist_threshold_deg = float(self.config.wrist_singularity_threshold_deg)
        elbow_threshold_deg = float(self.config.elbow_singularity_threshold_deg)
        elbow_margin_deg = float(self.config.elbow_singularity_selection_margin_deg)
        elbow_selection_threshold_deg = elbow_threshold_deg + elbow_margin_deg
        shoulder_threshold_m = float(self.config.shoulder_singularity_threshold_m)
        sides: dict[str, dict[str, object]] = {}
        for side, arm_q in (("left", left_q), ("right", right_q)):
            arm_q = np.asarray(arm_q, dtype=float).reshape(6)
            joint_deg = np.rad2deg(arm_q)
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
            analyzer_metadata: dict[str, object] = {
                "status_code": None,
                "label": "unavailable",
                "reason": "analyzer_not_configured",
            }
            analyzer = self._position_singularity_analyzers.get(side)
            if analyzer is not None and self.config.prefer_non_shoulder_singular_branch:
                try:
                    raw_analysis = analyzer(arm_q.copy())
                    if isinstance(raw_analysis, Mapping):
                        analyzer_metadata = dict(raw_analysis)
                    else:
                        analyzer_metadata = {
                            "status_code": None,
                            "label": "unavailable",
                            "reason": f"invalid_result_type:{type(raw_analysis).__name__}",
                        }
                except Exception as exc:  # noqa: BLE001 - preference must fail open.
                    analyzer_metadata = {
                        "status_code": None,
                        "label": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
            analyzer_label = str(analyzer_metadata.get("label") or "unknown")
            analyzer_code = analyzer_metadata.get("status_code")
            analyzer_available = analyzer_code is not None
            if self.config.prefer_non_elbow_singular_branch and analyzer_label == "elbow_singularity":
                elbow_singular = True
            if self.config.prefer_non_wrist_singular_branch and analyzer_label == "wrist_singularity":
                wrist_singular = True
            shoulder_singular = bool(
                self.config.prefer_non_shoulder_singular_branch
                and analyzer_label == "shoulder_singularity"
            )
            shoulder_distance: float | None = None
            raw_distance = analyzer_metadata.get("wrist_center_distance_m")
            try:
                shoulder_distance = abs(float(raw_distance))
            except (TypeError, ValueError):
                pass

            elbow_depth = (
                max(0.0, elbow_selection_threshold_deg - abs(j3_deg))
                / max(elbow_selection_threshold_deg, 1e-12)
                if elbow_singular
                else 0.0
            )
            wrist_depth = (
                max(0.0, wrist_threshold_deg - abs(j5_deg))
                / max(wrist_threshold_deg, 1e-12)
                if wrist_singular
                else 0.0
            )
            shoulder_depth = (
                max(0.0, shoulder_threshold_m - shoulder_distance)
                / max(shoulder_threshold_m, 1e-12)
                if shoulder_singular and shoulder_distance is not None
                else float(shoulder_singular)
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
                "is_shoulder_singular": shoulder_singular,
                "shoulder_wrist_center_distance_m": shoulder_distance,
                "shoulder_margin_m": (
                    None
                    if shoulder_distance is None
                    else shoulder_distance - shoulder_threshold_m
                ),
                "analyzer_available": bool(analyzer_available),
                "analyzer": analyzer_metadata,
                "singularity_depth_score": float(
                    elbow_depth + wrist_depth + shoulder_depth
                ),
                "is_singular": bool(
                    elbow_singular or wrist_singular or shoulder_singular
                ),
            }
        elbow_sides = [
            side for side, values in sides.items() if bool(values["is_elbow_singular"])
        ]
        wrist_sides = [
            side for side, values in sides.items() if bool(values["is_wrist_singular"])
        ]
        shoulder_sides = [
            side for side, values in sides.items() if bool(values["is_shoulder_singular"])
        ]
        singular_sides = [
            side for side, values in sides.items() if bool(values["is_singular"])
        ]
        result: dict[str, object] = {
            "enabled": bool(
                self.config.prefer_non_elbow_singular_branch
                or self.config.prefer_non_wrist_singular_branch
                or self.config.prefer_non_shoulder_singular_branch
            ),
            "criterion": "J3 elbow proxy, J5 wrist proxy, or injected analyzer",
            "elbow_threshold_deg": elbow_threshold_deg,
            "elbow_selection_margin_deg": elbow_margin_deg,
            "elbow_selection_threshold_deg": elbow_selection_threshold_deg,
            "wrist_threshold_deg": wrist_threshold_deg,
            "shoulder_threshold_m": shoulder_threshold_m,
            "sides": sides,
            "elbow_singular_sides": elbow_sides,
            "elbow_singular_side_count": len(elbow_sides),
            "any_elbow_singular": bool(elbow_sides),
            "wrist_singular_sides": wrist_sides,
            "wrist_singular_side_count": len(wrist_sides),
            "any_wrist_singular": bool(wrist_sides),
            "shoulder_singular_sides": shoulder_sides,
            "shoulder_singular_side_count": len(shoulder_sides),
            "any_shoulder_singular": bool(shoulder_sides),
            "singular_sides": singular_sides,
            "singular_side_count": len(singular_sides),
            "singularity_depth_score": float(
                sum(float(values["singularity_depth_score"]) for values in sides.values())
            ),
            "any_singular": bool(singular_sides),
        }
        analyzer_results_cacheable = bool(
            not self.config.prefer_non_shoulder_singular_branch
            or all(
                bool(sides[side]["analyzer_available"])
                for side in self._position_singularity_analyzers
            )
        )
        if analyzer_results_cacheable:
            if len(self._position_singularity_metadata_cache) >= 128:
                self._position_singularity_metadata_cache.clear()
            self._position_singularity_metadata_cache[cache_key] = copy.deepcopy(result)
        return result

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
        depth_scores = [float(item["singularity_depth_score"]) for item in classifications]
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
                or self.config.prefer_non_shoulder_singular_branch
            ),
            "prefer_non_elbow_singular_branch": bool(
                self.config.prefer_non_elbow_singular_branch
            ),
            "prefer_non_wrist_singular_branch": bool(
                self.config.prefer_non_wrist_singular_branch
            ),
            "prefer_non_shoulder_singular_branch": bool(
                self.config.prefer_non_shoulder_singular_branch
            ),
            "evaluated_feasible_count": len(evaluated),
            "fully_non_singular_count": fully_non_singular_count,
            "minimum_singular_side_count": min(counts) if counts else None,
            "minimum_singularity_depth_score": min(depth_scores) if depth_scores else None,
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
                and sum(count == 0 for count in counts) == 0
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
                    "prefer_non_shoulder_singular_branch": bool(
                        self.config.prefer_non_shoulder_singular_branch
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
        authority_fixed_sides = self._authority_fixed_sides(
            target_activity_metadata
        )
        forced_escape_latched = (
            self._joint_limit_forced_escape_latch_enabled()
        )
        boundary_hold_enabled = self._joint_limit_boundary_hold_enabled()
        if forced_escape_latched:
            self._refresh_joint_limit_escape_release_guards(
                raw_full_q=raw_full_q,
                reference_full_q=reference_full,
                fixed_sides=authority_fixed_sides,
            )
            active_limit_escape_latches = {
                key: direction
                for key, direction in self._refresh_joint_limit_escape_latches(
                    reference_full_q=reference_full,
                ).items()
                if key[0] not in authority_fixed_sides
            }
        else:
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
            active_limit_escape_latches = {}
        # A latched escape is a separate control mode.  The current limit-band
        # directions are used only to decide whether the ordinary result would
        # move inward; they must not filter the ordinary candidate pool before
        # it has been evaluated, otherwise every near-limit frame enters the
        # expensive auxiliary search path.
        limit_escape_required_directions = (
            self._active_joint_limit_escape_directions(
                reference_full_q=reference_full,
                target_activity=target_activity_metadata,
            )
            if forced_escape_latched
            else {}
        )
        boundary_hold_directions = (
            self._active_joint_limit_boundary_directions(
                reference_full_q=reference_full,
                target_activity=target_activity_metadata,
            )
            if boundary_hold_enabled
            else {}
        )
        limit_escape_active = bool(active_limit_escape_latches)
        limit_escape_mode = "active" if limit_escape_active else "normal"
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
        post_candidate_reserve_ms = max(
            0.0,
            float(self.config.solve_iteration_reserve_ms)
            - float(self.config.candidate_certification_reserve_ms),
        )
        branch_budget_ms = (
            max(
                0.0,
                time_budget_ms
                - pre_selection_elapsed_ms
                - post_candidate_reserve_ms,
            )
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

        if limit_escape_active:
            # An active escape never competes with ordinary tracking or the
            # other auxiliary branches.  Generate its one bounded candidate
            # only after the common deadline checks below.
            base_candidates = []
            limit_escape_metadata = {
                "enabled": True,
                "status": "active_latch_deferred",
                "candidate_count": 0,
            }
        else:
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
                "candidate_certification_reserve_ms": float(
                    self.config.candidate_certification_reserve_ms
                ),
                "candidate_post_selection_reserve_ms": float(
                    post_candidate_reserve_ms
                ),
                "candidate_eval_stopped_early": bool(candidate_eval_stopped_early),
                "candidate_eval_stop_reason": candidate_eval_stop_reason,
                "reference_collision_distance_m": reference_min_distance,
                "target_activity": dict(target_activity_metadata),
                "inactive_side_hold_active": bool(inactive_hold_active),
                "joint_limit_guard": {
                    "mode": self._joint_limit_guard_mode(),
                    "boundary_margin_deg": float(
                        self.config.joint_limit_boundary_guard_margin_deg
                    ),
                    "boundary_directions": self._joint_limit_escape_latch_metadata(
                        boundary_hold_directions
                    ),
                },
                "joint_limit_escape_hysteresis": {
                    "status": (
                        "latched_escape_only"
                        if limit_escape_mode == "active"
                        else "pending_single_escape"
                        if limit_escape_mode == "pending"
                        else "normal_tracking"
                    ),
                    "active": self._joint_limit_escape_latch_metadata(
                        active_limit_escape_latches
                    ),
                    "required": self._joint_limit_escape_latch_metadata(
                        limit_escape_required_directions
                    ),
                },
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
            directional_escape_pool: list[_DualArmEvaluatedCandidate] = []
            # In the directional policy an admitted escape is a safety
            # transition, not merely another pose-scoring alternative.  If a
            # candidate demonstrably reduces the local J3/J5 singularity
            # depth, prefer that pool over ordinary tracking candidates;
            # otherwise the pose score can keep selecting a visually better
            # but singular target and leave the arm parked in the pocket.
            if bool(
                getattr(
                    safety_profile,
                    "local_step_singularity_escape_directional_enabled",
                    False,
                )
            ):
                directional_escape_pool = [
                    item
                    for item in wrist_pool
                    if item.candidate.source == "singularity_escape"
                    and bool(
                        item.metadata.get(
                            "dual_singularity_escape_active",
                            False,
                        )
                    )
                    and (
                        int(
                            item.metadata.get(
                                "dual_singularity_escape_count_progress",
                                0,
                            )
                            or 0
                        )
                        > 0
                        or float(
                            item.metadata.get(
                                "dual_singularity_escape_depth_progress",
                                0.0,
                            )
                            or 0.0
                        )
                        > 0.0
                        or bool(
                            item.metadata.get(
                                "dual_singularity_escape_wide_plan_active",
                                False,
                            )
                        )
                        or bool(
                            item.metadata.get(
                                "dual_singularity_escape_directional_outward",
                                False,
                            )
                        )
                    )
                ]
                if directional_escape_pool:
                    wrist_pool = directional_escape_pool
            if (
                self.config.prefer_non_elbow_singular_branch
                or self.config.prefer_non_wrist_singular_branch
                or self.config.prefer_non_shoulder_singular_branch
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
            if directional_escape_pool:
                # Once an escape is active, Cartesian score alone is not a
                # stable selector: two equally good axes can alternate every
                # frame while the arm remains in the same pocket.  Keep the
                # direction with the strongest *actual* local progress, then
                # retain the already latched sign/axis, and use pose score
                # only as the final tie breaker.
                def directional_escape_key(item: _DualArmEvaluatedCandidate) -> tuple[object, ...]:
                    item_metadata = item.metadata
                    candidate_metadata = dict(
                        item_metadata.get("candidate_metadata", {}) or {}
                    )
                    direction_map = dict(
                        candidate_metadata.get(
                            "local_step_singularity_escape_direction_by_joint", {}
                        )
                        or {}
                    )
                    latch_match = 0
                    for label, direction in direction_map.items():
                        try:
                            side_text, joint_text = str(label).split(":j", 1)
                            key = (side_text, int(joint_text) - 1)
                            if int(direction) == int(
                                self._singularity_escape_direction_latch.get(key, 0)
                            ):
                                latch_match += 1
                        except (TypeError, ValueError):
                            continue
                    return (
                        -int(
                            item_metadata.get(
                                "dual_singularity_escape_count_progress", 0
                            )
                            or 0
                        ),
                        -float(
                            item_metadata.get(
                                "dual_singularity_escape_depth_progress", 0.0
                            )
                            or 0.0
                        ),
                        -int(latch_match),
                        0
                        if bool(
                            item_metadata.get(
                                "dual_singularity_escape_directional_outward", False
                            )
                        )
                        else 1,
                        float(item.score),
                        str(item.candidate.name),
                    )

                selection_pool.sort(key=directional_escape_key)
            else:
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

        # A near-limit transition is still a single bounded path even when an
        # inactive-side hold would normally request all configured scales.
        # Otherwise the first rejected ordinary candidate can consume the
        # whole scale ladder before the escape decision is made.
        primary_scales = (
            scale_factors
            if inactive_hold_active and not limit_escape_required_directions
            else scale_factors[:1]
        )
        fallback_scales = (
            ()
            if inactive_hold_active or limit_escape_required_directions
            else scale_factors[1:]
        )
        evaluate_stage(
            stage="base_primary",
            candidates=base_candidates,
            scales=primary_scales,
            stop_after_first_accept_per_candidate=not inactive_hold_active,
        )
        if (
            not evaluated
            and fallback_scales
            and not limit_escape_required_directions
            and not candidate_eval_stopped_early
        ):
            evaluate_stage(stage="base_fallback", candidates=base_candidates, scales=fallback_scales)

        base_metadata = selection_metadata(candidates=base_candidates)
        base_selection = choose_from_evaluated(metadata=base_metadata)
        if boundary_hold_enabled and base_selection is not None:
            # Catch a candidate that enters the guard band from just outside
            # it; waiting until the following frame would admit one avoidable
            # inward step at the hard boundary.
            boundary_hold_directions = (
                self._active_joint_limit_boundary_directions(
                    reference_full_q=reference_full,
                    candidate_full_q=base_selection[0],
                    target_activity=target_activity_metadata,
                )
            )
        boundary_hold_candidate = (
            None
            if base_selection is None
            else self._joint_limit_boundary_hold_candidate(
                ordinary_q=base_selection[0],
                reference_full_q=reference_full,
                directions=boundary_hold_directions,
            )
        )
        if boundary_hold_candidate is not None:
            # The ordinary candidate has already passed the normal admission
            # checks, but one or more near-limit components still point into
            # the hard stop.  Re-evaluate the projected candidate and its
            # collision distance; never substitute the historical fixed-size
            # outward kick here.
            evaluated.clear()
            collision_safe_evaluated.clear()
            evaluate_stage(
                stage="joint_limit_boundary_hold",
                candidates=[boundary_hold_candidate],
                scales=(1.0,),
            )
            boundary_metadata = selection_metadata(
                candidates=[boundary_hold_candidate]
            )
            boundary_selection = choose_from_evaluated(
                metadata=boundary_metadata
            )
            if boundary_selection is not None:
                boundary_selection[4]["candidate_eval_mode"] = (
                    "joint_limit_boundary_hold"
                )
                return boundary_selection
            boundary_metadata.update(
                {
                    "status": "hold_reference",
                    "selected": "joint_limit_boundary_hold_reference",
                    "selected_source": "joint_limit_boundary_hold",
                    "reason": "projected_boundary_candidate_not_admissible",
                    "candidate_eval_mode": "joint_limit_boundary_hold",
                    "collision_distance_m": reference_min_distance,
                }
            )
            return (
                reference_full.copy(),
                "joint_limit_boundary_hold_reference",
                0.0,
                reference_min_distance,
                boundary_metadata,
            )
        wrist_q5_boundary_hold_candidate = (
            None
            if base_selection is None
            else self._wrist_q5_entry_boundary_hold_candidate(
                ordinary_q=base_selection[0],
                reference_full_q=reference_full,
                target_activity=target_activity_metadata,
            )
        )
        if wrist_q5_boundary_hold_candidate is not None:
            # V2's latch-required wrist policy must not convert the first
            # inward step at the SDK q5 boundary into a full-size synthetic
            # escape step.  Preserve the already admitted q5 and continue
            # with the ordinary motion on every other joint.  Returning here
            # also avoids launching the expensive singularity candidate
            # expansion for a boundary that was handled directionally.
            evaluated.clear()
            collision_safe_evaluated.clear()
            evaluate_stage(
                stage="wrist_q5_entry_boundary_hold",
                candidates=[wrist_q5_boundary_hold_candidate],
                scales=(1.0,),
            )
            wrist_boundary_metadata = selection_metadata(
                candidates=[wrist_q5_boundary_hold_candidate]
            )
            wrist_boundary_selection = choose_from_evaluated(
                metadata=wrist_boundary_metadata
            )
            if wrist_boundary_selection is not None:
                wrist_boundary_selection[4]["candidate_eval_mode"] = (
                    "wrist_q5_entry_boundary_hold"
                )
                return wrist_boundary_selection
            wrist_boundary_metadata.update(
                {
                    "status": "hold_reference",
                    "selected": "wrist_q5_entry_boundary_hold_reference",
                    "selected_source": "wrist_q5_boundary_hold",
                    "reason": "projected_wrist_boundary_candidate_not_admissible",
                    "candidate_eval_mode": "wrist_q5_entry_boundary_hold",
                    "collision_distance_m": reference_min_distance,
                }
            )
            return (
                reference_full.copy(),
                "wrist_q5_entry_boundary_hold_reference",
                0.0,
                reference_min_distance,
                wrist_boundary_metadata,
            )
        limit_escape_pending = bool(limit_escape_active)
        if not limit_escape_pending and limit_escape_required_directions:
            if base_selection is None:
                # A near-limit frame with no safe ordinary result gets one
                # bounded escape probe, provided the deadline has not already
                # forced a fail-closed hold.
                limit_escape_pending = not candidate_eval_stopped_early
            else:
                # Do not let an ordinary/canonical branch step back toward a
                # physical limit.  This is the only first-entry trigger; mere
                # proximity to a limit does not launch auxiliary search.
                limit_escape_pending = not self._joint_limit_escape_output_moves_outward(
                    output_q=base_selection[0],
                    reference_full_q=reference_full,
                    directions=limit_escape_required_directions,
                )
        if limit_escape_pending:
            limit_escape_mode = "active" if limit_escape_active else "pending"
            # The ordinary result has already been proven to be an inward
            # transition (or an active latch is forcing escape).  Keep its
            # diagnostics in the stage summary, but remove it from the final
            # selection pool so it cannot compete with the bounded escape
            # candidate below.
            evaluated.clear()
            collision_safe_evaluated.clear()
            base_metadata["joint_limit_escape_hysteresis"] = {
                "status": (
                    "latched_escape_only"
                    if limit_escape_active
                    else "pending_single_escape"
                ),
                "active": self._joint_limit_escape_latch_metadata(
                    active_limit_escape_latches
                ),
                "required": self._joint_limit_escape_latch_metadata(
                    limit_escape_required_directions
                ),
            }
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
        expand_for_shoulder_singularity = bool(
            base_selection is not None
            and self.config.prefer_non_shoulder_singular_branch
            and base_selected_singularity.get("any_shoulder_singular", False)
            and not candidate_eval_stopped_early
        )
        expand_for_wrist_singularity = bool(
            base_selection is not None
            and self.config.prefer_non_wrist_singular_branch
            and base_selected_singularity.get("any_wrist_singular", False)
            and not candidate_eval_stopped_early
        )
        base_local_singularity = (
            {}
            if base_selection is None
            else self._local_proxy_singularity_metadata(base_selection[0])
        )
        raw_local_singularity = self._local_proxy_singularity_metadata(
            np.asarray(raw_full_q, dtype=float).reshape(-1)
        )
        expand_for_local_singularity_escape = bool(
            base_selection is not None
            and bool(
                getattr(
                    safety_profile,
                    "local_step_singularity_escape_enabled",
                    False,
                )
            )
            and bool(
                base_local_singularity.get("any_singular", False)
                or raw_local_singularity.get("any_singular", False)
            )
            and not candidate_eval_stopped_early
        )
        expand_for_position_singularity = bool(
            expand_for_elbow_singularity
            or expand_for_wrist_singularity
            or expand_for_shoulder_singularity
            or expand_for_local_singularity_escape
        )
        if limit_escape_pending:
            # Limit escape is mutually exclusive with posture, wrist, and
            # singularity expansion.  Combining them was the source of the
            # repeated IK budget overruns in the bad revision.
            expand_for_elbow_singularity = False
            expand_for_shoulder_singularity = False
            expand_for_wrist_singularity = False
            expand_for_local_singularity_escape = False
            expand_for_position_singularity = False
        if (
            base_selection is not None
            and not limit_escape_pending
            and not expand_for_position_singularity
        ):
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
            target_activity=target_activity_metadata,
            include_base=False,
            # Keep the historical auxiliary limit branch available for the
            # normal singularity/fallback path.  In the pending/active mode,
            # the other branches are disabled and this is the only branch
            # actually evaluated.
            include_limit_escape=not boundary_hold_enabled,
            include_posture_escape=(
                not limit_escape_pending and not candidate_eval_stopped_early
            ),
            include_wrist_unload=(
                not limit_escape_pending and not candidate_eval_stopped_early
            ),
            include_singularity_escape=(
                not limit_escape_pending and expand_for_local_singularity_escape
            ),
            limit_escape_max_candidates=1 if limit_escape_pending else None,
            # Use the raw one-step IK result as the escape probe.  The base
            # candidate is already locally clipped and may sit just outside
            # the proxy band even while the raw target is trying to cross it;
            # seeding from the base output was the source of the alternating
            # "ordinary / escape" frames seen in the 2026-07-26 trace.
            singularity_seed_qpos=raw_full_q,
        )
        all_candidates = list(base_candidates) + list(auxiliary_candidates)
        if auxiliary_candidates and not candidate_eval_stopped_early:
            evaluated_before_auxiliary = len(evaluated)
            evaluate_stage(
                stage="auxiliary_primary",
                candidates=auxiliary_candidates,
                # A limit transition has one candidate and one scale attempt.
                # If that probe is not admissible, holding the last qref is
                # safer and cheaper than trying posture/wrist alternatives or
                # repeatedly certifying the same escape at smaller scales.
                scales=scale_factors[:1] if limit_escape_pending else primary_scales,
                stop_after_first_accept_per_candidate=not inactive_hold_active,
            )
            if (
                len(evaluated) == evaluated_before_auxiliary
                and fallback_scales
                and not limit_escape_pending
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
            elif expand_for_position_singularity:
                candidate_eval_mode = "base_then_auxiliary_singularity_search"
            else:
                candidate_eval_mode = "base_then_auxiliary"
            auxiliary_selection[4]["candidate_eval_mode"] = candidate_eval_mode
            auxiliary_selection[4]["expanded_for_elbow_singularity"] = bool(
                expand_for_elbow_singularity
            )
            auxiliary_selection[4]["expanded_for_shoulder_singularity"] = bool(
                expand_for_shoulder_singularity
            )
            auxiliary_selection[4]["expanded_for_wrist_singularity"] = bool(
                expand_for_wrist_singularity
            )
            auxiliary_selection[4]["expanded_for_local_singularity_escape"] = bool(
                expand_for_local_singularity_escape
            )
            auxiliary_selection[4]["expanded_for_position_singularity"] = bool(
                expand_for_position_singularity
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
        target_activity: dict[str, object] | None = None,
        limit_escape_max_candidates: int | None = None,
        include_base: bool = True,
        include_limit_escape: bool = True,
        include_posture_escape: bool = True,
        include_wrist_unload: bool = True,
        include_singularity_escape: bool = False,
        singularity_seed_qpos: np.ndarray | None = None,
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
                target_activity=target_activity,
                max_candidates_override=limit_escape_max_candidates,
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

        if include_singularity_escape:
            singularity_candidates, singularity_metadata = (
                self._dual_arm_singularity_escape_candidates(
                    reference_full_q=reference_full_q,
                    current_targets=current_targets,
                    seed_full_q=singularity_seed_qpos,
                )
            )
        else:
            singularity_candidates = []
            singularity_metadata = {
                "enabled": False,
                "status": "deferred",
                "candidate_count": 0,
            }
        for candidate in singularity_candidates:
            if any(np.allclose(candidate.qpos, existing, atol=1e-9) for existing in seen):
                continue
            candidates.append(candidate)
            seen.append(candidate.qpos.copy())

        combined_metadata = dict(limit_metadata)
        combined_metadata["posture_escape"] = dict(posture_metadata)
        combined_metadata["wrist_unload"] = dict(wrist_metadata)
        combined_metadata["singularity_escape"] = dict(singularity_metadata)
        return candidates, combined_metadata

    def _local_proxy_singularity_metadata(self, full_qpos: np.ndarray) -> dict[str, object]:
        """Return the cheap, vendor-independent J3/J5 singularity proxy.

        The SDK analyzer remains a diagnostic/final-gate input.  Local escape
        candidates deliberately use only these two joint proxies so a vendor
        boundary classification cannot prevent a one-step move away from a
        known local singular neighborhood.
        """

        left_q, right_q = self._split_solution(np.asarray(full_qpos, dtype=float).reshape(-1))
        elbow_threshold = float(
            self.config.elbow_singularity_threshold_deg
            + self.config.elbow_singularity_selection_margin_deg
        )
        wrist_threshold = float(self.config.wrist_singularity_threshold_deg)
        sides: dict[str, dict[str, object]] = {}
        for side, arm_q in (("left", left_q), ("right", right_q)):
            joint_deg = np.rad2deg(np.asarray(arm_q, dtype=float).reshape(6))
            j3_deg = float(joint_deg[2])
            j5_deg = float(joint_deg[4])
            elbow_singular = bool(
                self.config.prefer_non_elbow_singular_branch
                and elbow_threshold > 0.0
                and abs(j3_deg) < elbow_threshold
            )
            wrist_singular = bool(
                self.config.prefer_non_wrist_singular_branch
                and wrist_threshold > 0.0
                and abs(j5_deg) < wrist_threshold
            )
            elbow_depth = (
                max(0.0, elbow_threshold - abs(j3_deg)) / max(elbow_threshold, 1e-12)
                if elbow_singular
                else 0.0
            )
            wrist_depth = (
                max(0.0, wrist_threshold - abs(j5_deg)) / max(wrist_threshold, 1e-12)
                if wrist_singular
                else 0.0
            )
            sides[side] = {
                "j3_deg": j3_deg,
                "j5_deg": j5_deg,
                "elbow_margin_deg": abs(j3_deg) - elbow_threshold,
                "wrist_margin_deg": abs(j5_deg) - wrist_threshold,
                "is_elbow_singular": elbow_singular,
                "is_wrist_singular": wrist_singular,
                "is_singular": bool(elbow_singular or wrist_singular),
                "singularity_depth_score": float(elbow_depth + wrist_depth),
            }
        singular_sides = [side for side, values in sides.items() if values["is_singular"]]
        return {
            "criterion": "local J3 elbow proxy and J5 wrist proxy",
            "elbow_threshold_deg": elbow_threshold,
            "wrist_threshold_deg": wrist_threshold,
            "sides": sides,
            "singular_sides": singular_sides,
            "singular_side_count": len(singular_sides),
            "singularity_depth_score": float(
                sum(float(values["singularity_depth_score"]) for values in sides.values())
            ),
            "any_singular": bool(singular_sides),
        }

    def _singularity_escape_traversal_metadata(
        self,
        *,
        reference_full_q: np.ndarray,
        admitted_full_q: np.ndarray,
        selection_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Describe a certified outward J3 traversal for the RM65 backend.

        The SDK gate normally holds a command whenever both the current and
        requested q3 are inside its elbow-singularity neighborhood.  A local
        escape candidate is safe to continue only when the *emitted* qref
        actually moves in the latched outward direction.  This helper keeps
        that exception narrow: a raw/wide probe, a smoothed no-op, or an
        ordinary target never receives an elbow-traversal certificate.
        """

        result: dict[str, object] = {
            "enabled": bool(
                self.config.solve_step_mode
                == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            ),
            "by_side": {"left": False, "right": False},
            "direction_by_side": {},
            "delta_q3_deg_by_side": {},
            "final_local_singularity": {},
        }
        if not result["enabled"]:
            return result
        metadata = dict(selection_metadata or {})
        if str(metadata.get("selected_source", "") or "") != "singularity_escape":
            result["reason"] = "selected_source_not_singularity_escape"
            return result
        selected_metadata = dict(metadata.get("selected_metadata", {}) or {})
        candidate_metadata = dict(
            selected_metadata.get("candidate_metadata", {}) or {}
        )
        direction_map = dict(
            candidate_metadata.get(
                "local_step_singularity_escape_direction_by_joint", {}
            )
            or {}
        )
        if not direction_map:
            result["reason"] = "escape_direction_metadata_missing"
            return result
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        admitted = np.asarray(admitted_full_q, dtype=float).reshape(reference.shape)
        reference_by_side = dict(zip(("left", "right"), self._split_solution(reference)))
        admitted_by_side = dict(zip(("left", "right"), self._split_solution(admitted)))
        reference_local = self._local_proxy_singularity_metadata(reference)
        final_local = self._local_proxy_singularity_metadata(admitted)
        release_band_deg = max(
            0.0,
            float(
                getattr(
                    self.ik_safety_profile,
                    "local_step_singularity_escape_direction_hysteresis_deg",
                    1.0,
                )
            ),
        )
        for side in ("left", "right"):
            label = f"{side}:j3"
            try:
                direction = int(direction_map.get(label, 0))
            except (TypeError, ValueError):
                direction = 0
            if direction not in (-1, 1):
                continue
            delta_q3 = float(admitted_by_side[side][2] - reference_by_side[side][2])
            reference_side = dict(reference_local.get("sides", {}).get(side, {}) or {})
            side_local = dict(final_local.get("sides", {}).get(side, {}) or {})
            outward = bool(direction * delta_q3 > 1e-6)
            # The certificate is about a bounded outward traversal, not only
            # the final sample's local classification.  Permit the last
            # step(s) that cross the local proxy while the reference remains
            # inside its release band; the backend still checks the SDK
            # target/current labels and requires the exact qref fingerprint.
            singular = bool(
                reference_side.get("is_elbow_singular", False)
                or float(reference_side.get("elbow_margin_deg", -1.0))
                < release_band_deg
                or side_local.get("is_elbow_singular", False)
            )
            result["direction_by_side"][side] = direction
            result["delta_q3_deg_by_side"][side] = float(np.rad2deg(delta_q3))
            result["final_local_singularity"][side] = side_local
            result["by_side"][side] = bool(outward and singular)
        result["reason"] = (
            "outward_emitted_qref" if any(result["by_side"].values()) else "no_outward_singular_q3"
        )
        return result

    def _singularity_escape_direction_for_axis(
        self,
        *,
        side: str,
        local_index: int,
        full_index: int,
        seed: np.ndarray,
        current_targets: dict[str, np.ndarray],
        before_score: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        step: float,
    ) -> tuple[int, str]:
        """Choose one deterministic outward direction for a singular axis.

        The old escape layer emitted both signs on every frame and then let a
        global singularity-depth score choose between them.  Near q3/q5=0 the
        vendor IK can change sign by a fraction of a degree, so that policy
        converted harmless solver noise into a large physical reversal.  A
        latch is preferred while the local proxy remains singular.  If the
        coordinate is numerically on zero, the first direction is selected by
        actual Cartesian target progress (with a deterministic positive tie
        break), rather than by an arbitrary candidate ordering.
        """

        key = (str(side), int(local_index))
        latched = int(self._singularity_escape_direction_latch.get(key, 0))
        if latched in (-1, 1):
            return latched, "latched"

        value_deg = float(np.rad2deg(seed[full_index]))
        # Outside a small numerical band, moving in the sign of the joint
        # increases |q3|/|q5| and is the only locally meaningful escape
        # direction.  The old 1e-4 rad cut-over was too small for encoder
        # noise: a joint reported at +/-0.01 deg could still make the probe
        # choose the opposite side on successive frames.  Keep the sign
        # locked unless the joint is genuinely close enough to zero that both
        # directions have indistinguishable local singularity depth.
        direction_probe_band_deg = 0.50
        if abs(value_deg) > direction_probe_band_deg:
            direction = 1 if value_deg > 0.0 else -1
            self._singularity_escape_direction_latch[key] = direction
            self._singularity_escape_direction_failures[key] = 0
            return direction, "outward_from_zero"

        # Near zero use a one-step probe.  Singularity depth is the primary
        # objective; Cartesian target progress is only a tie breaker.  The
        # previous implementation selected by Cartesian progress first, so a
        # direction that improved the target while moving *deeper* into the
        # singular pocket could win and then be reversed by the next frame.
        probe_scores: dict[int, tuple[float, float]] = {}
        before_local = self._local_proxy_singularity_metadata(seed)
        before_depth = float(before_local.get("singularity_depth_score", 0.0))
        before_count = int(before_local.get("singular_side_count", 0) or 0)
        for direction in (-1, 1):
            probe_q = seed.copy()
            probe_q[full_index] += float(direction) * float(step)
            if (
                probe_q[full_index] < q_min[full_index] - 1e-9
                or probe_q[full_index] > q_max[full_index] + 1e-9
            ):
                continue
            probe_local = self._local_proxy_singularity_metadata(probe_q)
            depth_progress = before_depth - float(
                probe_local.get("singularity_depth_score", 0.0)
            )
            count_progress = before_count - int(
                probe_local.get("singular_side_count", 0) or 0
            )
            pose_progress = before_score - self._dual_pose_score(
                self._dual_pose_metrics(probe_q, current_targets)
            )
            # Count progress is deliberately a large first-class component:
            # leaving the pocket is more important than a small Cartesian
            # improvement.  The actual candidate evaluator still applies all
            # pose, collision, and joint limits before admission.
            probe_scores[direction] = (
                float(count_progress) * 10.0 + float(depth_progress),
                float(pose_progress),
            )
        if not probe_scores:
            direction = 1
            source = "deterministic_fallback"
        else:
            direction = max(
                probe_scores,
                key=lambda item: (
                    float(probe_scores[item][0]),
                    float(probe_scores[item][1]),
                    int(item),
                ),
            )
            source = "singularity_depth_probe"
        self._singularity_escape_direction_latch[key] = int(direction)
        self._singularity_escape_direction_failures[key] = 0
        return int(direction), source

    def _update_singularity_escape_direction_state(
        self,
        *,
        selected_q: np.ndarray,
        selection_metadata: Mapping[str, object] | None,
    ) -> None:
        """Commit or age the directional escape latch after admission.

        A proposed direction is not considered successful until it is the
        selected collision-safe candidate.  If the selector repeatedly falls
        back to a non-escape candidate while the pocket remains singular, the
        latch is released after a bounded number of frames so a blocked side
        can be reconsidered.  Once a side leaves the local singular proxy its
        latch is removed immediately.
        """

        profile = self.ik_safety_profile
        if profile is None or not bool(
            getattr(profile, "local_step_singularity_escape_directional_enabled", False)
        ):
            return
        metadata = dict(selection_metadata or {})
        selected_metadata = dict(metadata.get("selected_metadata", {}) or {})
        candidate_metadata = dict(selected_metadata.get("candidate_metadata", {}) or {})
        selected_source = str(
            metadata.get("selected_source", selected_metadata.get("source", ""))
            or ""
        )
        if selected_source == "singularity_escape":
            direction_map = dict(
                candidate_metadata.get(
                    "local_step_singularity_escape_direction_by_joint", {}
                )
                or {}
            )
            if not direction_map:
                side = candidate_metadata.get("local_step_singularity_escape_side")
                local_index = candidate_metadata.get(
                    "local_step_singularity_escape_joint_index"
                )
                direction = candidate_metadata.get(
                    "local_step_singularity_escape_direction"
                )
                if side is not None and local_index is not None and direction in (-1, 1):
                    direction_map[f"{side}:j{int(local_index) + 1}"] = int(direction)
            for label, direction in direction_map.items():
                try:
                    side_text, joint_text = str(label).split(":j", 1)
                    key = (side_text, int(joint_text) - 1)
                    value = int(direction)
                except (TypeError, ValueError):
                    continue
                if value in (-1, 1):
                    self._singularity_escape_direction_latch[key] = value
                    self._singularity_escape_direction_failures[key] = 0
            # Lock the selected axis set as well as its signs.  The coupled
            # J3+J5 candidate therefore remains coupled until the side exits
            # the release band; a pose-score tie cannot alternate J3/J5 on
            # successive frames.
            axis_by_side: dict[str, list[int]] = {}
            for label in direction_map:
                try:
                    side_text, joint_text = str(label).split(":j", 1)
                    local_index = int(joint_text) - 1
                except (TypeError, ValueError):
                    continue
                if local_index in (2, 4):
                    axis_by_side.setdefault(side_text, []).append(local_index)
            for side_text, indices in axis_by_side.items():
                self._singularity_escape_axis_latch[side_text] = tuple(
                    sorted(set(int(index) for index in indices))
                )

        local = self._local_proxy_singularity_metadata(
            np.asarray(selected_q, dtype=float).reshape(-1)
        )
        release_margin_deg = max(
            0.0,
            float(
                getattr(
                    profile,
                    "local_step_singularity_escape_direction_hysteresis_deg",
                    1.0,
                )
            ),
        )
        for key in tuple(self._singularity_escape_direction_latch):
            side, local_index = key
            side_metadata = dict(local.get("sides", {}).get(side, {}) or {})
            margin_key = "elbow_margin_deg" if local_index == 2 else "wrist_margin_deg"
            if not bool(side_metadata.get("is_singular", False)) and float(
                side_metadata.get(margin_key, -1.0)
            ) >= release_margin_deg:
                self._singularity_escape_direction_latch.pop(key, None)
                self._singularity_escape_direction_failures.pop(key, None)
                continue
            if selected_source != "singularity_escape":
                failures = int(self._singularity_escape_direction_failures.get(key, 0)) + 1
                self._singularity_escape_direction_failures[key] = failures
                if failures >= 6:
                    self._singularity_escape_direction_latch.pop(key, None)
                    self._singularity_escape_direction_failures.pop(key, None)

        for side, axis_plan in tuple(self._singularity_escape_axis_latch.items()):
            side_metadata = dict(local.get("sides", {}).get(side, {}) or {})
            if not axis_plan or (
                not bool(side_metadata.get("is_singular", False))
                and all(
                    float(
                        side_metadata.get(
                            "elbow_margin_deg" if index == 2 else "wrist_margin_deg",
                            -1.0,
                        )
                    )
                    >= release_margin_deg
                    for index in axis_plan
                )
            ):
                self._singularity_escape_axis_latch.pop(side, None)
            elif selected_source != "singularity_escape":
                # Keep a blocked plan for a few frames so a transient
                # collision query or branch-budget miss cannot immediately
                # flip the commanded escape axis.
                if all(
                    int(self._singularity_escape_direction_failures.get((side, index), 0))
                    >= 6
                    for index in axis_plan
                ):
                    self._singularity_escape_axis_latch.pop(side, None)

        if not bool(local.get("any_singular", False)):
            self._singularity_escape_direction_latch.clear()
            self._singularity_escape_direction_failures.clear()
            self._singularity_escape_axis_latch.clear()

    def _dual_arm_singularity_escape_candidates(
        self,
        *,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
        seed_full_q: np.ndarray | None = None,
    ) -> tuple[list[_DualArmCandidate], dict[str, object]]:
        safety_profile = self.ik_safety_profile
        metadata: dict[str, object] = {
            "enabled": bool(
                safety_profile is not None
                and getattr(safety_profile, "local_step_singularity_escape_enabled", False)
            ),
            "candidate_count": 0,
        }
        if safety_profile is None or not bool(
            getattr(safety_profile, "local_step_singularity_escape_enabled", False)
        ):
            metadata["status"] = "disabled"
            return [], metadata
        if self._full_execution_joint_step is None:
            metadata["status"] = "joint_step_unavailable"
            return [], metadata

        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        seed = (
            reference
            if seed_full_q is None
            else np.asarray(seed_full_q, dtype=float).reshape(reference.shape)
        )
        # ``reference`` is the last admitted qref and is the only state from
        # which an escape command may be integrated.  The raw IK result is a
        # useful crossing hint, but must not redefine the singularity depth
        # used by admission: doing so made a raw 3--4 degree branch appear
        # already safe and silently removed the escape candidate on the next
        # frame.  Keep both views in the metadata for offline diagnosis.
        reference_local = self._local_proxy_singularity_metadata(reference)
        seed_local = self._local_proxy_singularity_metadata(seed)
        metadata["reference_singularity"] = reference_local
        metadata["seed_singularity"] = seed_local
        release_band_deg = max(
            0.0,
            float(
                getattr(
                    safety_profile,
                    "local_step_singularity_escape_direction_hysteresis_deg",
                    1.0,
                )
            ),
        )
        latched_band_active = False
        for side in ("left", "right"):
            side_reference = dict(reference_local.get("sides", {}).get(side, {}) or {})
            for local_index, margin_key in ((2, "elbow_margin_deg"), (4, "wrist_margin_deg")):
                if (side, local_index) not in self._singularity_escape_direction_latch:
                    continue
                if float(side_reference.get(margin_key, -1.0)) < release_band_deg:
                    latched_band_active = True
                    break
            if latched_band_active:
                break
        if not bool(reference_local["any_singular"] or seed_local["any_singular"] or latched_band_active):
            metadata["status"] = "inactive_reference_and_seed_not_singular"
            return [], metadata
        max_candidates = max(
            0,
            int(getattr(safety_profile, "local_step_singularity_escape_max_candidates", 6)),
        )
        step_fraction = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_step_fraction",
                        1.0,
                    )
                ),
            ),
        )
        if max_candidates <= 0 or step_fraction <= 0.0:
            metadata["status"] = "max_candidates_zero" if max_candidates <= 0 else "step_fraction_zero"
            return [], metadata

        q_min = np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float).reshape(reference.shape)
        q_max = np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float).reshape(reference.shape)
        full_step = np.asarray(self._full_execution_joint_step, dtype=float).reshape(reference.shape) * step_fraction
        # Progress is measured from the actual admitted qref.  Measuring from
        # the raw IK branch lets a solver branch flip manufacture artificial
        # progress and is the main mechanism behind the historical back-and-
        # forth escape trace.
        before_metrics = self._dual_pose_metrics(reference, current_targets)
        before_score = self._dual_pose_score(before_metrics)
        max_position_error_m = max(
            float(before_metrics[side]["translation_m"]) for side in ("left", "right")
        )
        max_orientation_error_deg = max(
            float(before_metrics[side]["rotation_deg"]) for side in ("left", "right")
        )
        wide_search_position_threshold_m = max(
            0.0,
            float(
                getattr(
                    safety_profile,
                    "local_step_singularity_escape_wide_search_position_m",
                    0.10,
                )
            ),
        )
        wide_search_orientation_threshold_deg = max(
            0.0,
            float(
                getattr(
                    safety_profile,
                    "local_step_singularity_escape_wide_search_orientation_deg",
                    20.0,
                )
            ),
        )
        wide_search_far_target_sides = {
            side: bool(
                float(before_metrics[side]["translation_m"])
                >= wide_search_position_threshold_m
                or float(before_metrics[side]["rotation_deg"])
                >= wide_search_orientation_threshold_deg
            )
            for side in ("left", "right")
        }
        wide_search_enabled = bool(
            getattr(
                safety_profile,
                "local_step_singularity_escape_wide_search_enabled",
                False,
            )
            and any(wide_search_far_target_sides.values())
        )
        raw_radius_factors = (
            getattr(
                safety_profile,
                "local_step_singularity_escape_wide_search_radius_factors",
                (1.0,),
            )
            if wide_search_enabled
            else (1.0,)
        )
        radius_factors: list[float] = []
        max_radius = max(
            1.0,
            float(getattr(self.config, "local_step_candidate_clip_scale", 1.0)),
        )
        for raw_factor in tuple(raw_radius_factors or ()):
            try:
                factor = float(raw_factor)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(factor) or factor <= 0.0:
                continue
            factor = min(max_radius, factor)
            if not any(abs(factor - existing) <= 1e-9 for existing in radius_factors):
                radius_factors.append(factor)
        if not radius_factors:
            radius_factors = [1.0]
        radius_factors.sort()
        # Build one outward direction per active singular coordinate.  In the
        # opt-in policy we never emit both signs for the same axis in one
        # frame.  This is the key difference from the previous implementation:
        # selection noise cannot turn into an alternating physical command.
        axis_by_side: dict[str, list[tuple[int, int, int, float, str]]] = {
            "left": [],
            "right": [],
        }
        active_axis_plan_by_side = {
            side: tuple(
                int(index)
                for index in self._singularity_escape_axis_latch.get(side, ())
                if int(index) in (2, 4)
            )
            for side in ("left", "right")
        }
        directional_enabled = bool(
            getattr(
                safety_profile,
                "local_step_singularity_escape_directional_enabled",
                False,
            )
        )
        for side, local_index, full_index in self._arm_joint_entries(seed):
            if local_index not in (2, 4):
                continue
            active_axis_plan = active_axis_plan_by_side.get(side, ())
            side_reference_for_plan = dict(reference_local["sides"].get(side, {}) or {})
            side_seed_for_plan = dict(seed_local["sides"].get(side, {}) or {})
            planned_axis_still_active = bool(
                active_axis_plan
                and any(
                    bool(
                        side_reference_for_plan.get(
                            "is_elbow_singular" if planned_index == 2 else "is_wrist_singular",
                            False,
                        )
                    )
                    or bool(
                        side_seed_for_plan.get(
                            "is_elbow_singular" if planned_index == 2 else "is_wrist_singular",
                            False,
                        )
                    )
                    or (
                        (side, planned_index) in self._singularity_escape_direction_latch
                        and float(
                            side_reference_for_plan.get(
                                "elbow_margin_deg" if planned_index == 2 else "wrist_margin_deg",
                                -1.0,
                            )
                        )
                        < release_band_deg
                    )
                    for planned_index in active_axis_plan
                )
            )
            if planned_axis_still_active and int(local_index) not in active_axis_plan:
                continue
            side_reference = dict(reference_local["sides"][side])
            side_seed = dict(seed_local["sides"].get(side, {}) or {})
            singular_key = "is_elbow_singular" if local_index == 2 else "is_wrist_singular"
            margin_key = "elbow_margin_deg" if local_index == 2 else "wrist_margin_deg"
            latched = (side, local_index) in self._singularity_escape_direction_latch
            reference_in_release_band = bool(
                float(side_reference.get(margin_key, -1.0)) < release_band_deg
            )
            seed_singular = bool(side_seed.get(singular_key, False))
            # Keep an already admitted outward plan alive through the small
            # band between the local proxy and the SDK boundary.  Also create
            # a pre-emptive candidate when the raw IK result is trying to
            # cross into the pocket while the admitted qref is still just
            # outside it.  Neither path can bypass the final step/collision
            # admission or the certificate-bound RM65 gate.
            if not bool(
                side_reference.get(singular_key, False)
                or (latched and reference_in_release_band)
                or seed_singular
            ):
                continue
            base_step = float(full_step[full_index])
            if base_step <= 1e-12:
                continue
            if directional_enabled:
                direction_probe = reference.copy()
                # The direction must be tied to the admitted/reference qref,
                # not to a raw IK branch that may cross zero between frames.
                # Other joints are deliberately kept at the admitted qref as
                # well: direction selection must not be changed by a branch
                # flip in an unrelated joint.
                direction_probe[full_index] = reference[full_index]
                direction, direction_source = self._singularity_escape_direction_for_axis(
                    side=side,
                    local_index=local_index,
                    full_index=full_index,
                    seed=direction_probe,
                    current_targets=current_targets,
                    before_score=before_score,
                    q_min=q_min,
                    q_max=q_max,
                    step=base_step,
                )
            else:
                # Exact rollback: retain the historical two-sided candidate
                # generator when the profile does not opt into directionality.
                direction, direction_source = 0, "legacy_two_sided"
            axis_by_side[side].append(
                (int(local_index), int(full_index), int(direction), base_step, direction_source)
            )

        metadata["active_axis_plan_by_side"] = {
            side: list(indices)
            for side, indices in active_axis_plan_by_side.items()
            if indices
        }

        scored: list[
            tuple[int, float, float, int, float, int, int, str, np.ndarray, dict[str, object]]
        ] = []

        def append_scored_candidate(
            *,
            side: str,
            radius_factor: float,
            axes: Sequence[tuple[int, int, int, float, str]],
            direction_override: int | None = None,
            directions_override: Mapping[int, int] | None = None,
        ) -> None:
            # Directional escape is a safety transition, not a second full
            # IK branch.  Start from the admitted qref for *all* joints so a
            # raw branch change on the inactive arm (especially in single-arm
            # fix mode) cannot make the candidate fail the inactive-side hold
            # or reintroduce a hidden multi-joint jump.
            candidate_q = reference.copy() if directional_enabled else seed.copy()
            # A wide search is a *direction probe*, not a permission to send
            # a multi-step joint jump.  The previous implementation used the
            # wide radius as the candidate itself; the normal local-step clip
            # then either erased the probe (clip scale=1) or, when relaxed,
            # let the escape path jump too far.  Keep the emitted candidate at
            # one ordinary execution step and carry the farther probe's
            # singularity progress as a bounded plan certificate.
            wide_probe = bool(
                directional_enabled
                and wide_search_enabled
                and wide_search_far_target_sides.get(side, False)
                and float(radius_factor) > 1.0
            )
            command_radius_factor = 1.0 if wide_probe else float(radius_factor)
            probe_q = reference.copy() if directional_enabled else seed.copy()
            direction_map: dict[str, int] = {}
            step_degrees: dict[str, float] = {}
            probe_step_degrees: dict[str, float] = {}
            for local_index, full_index, direction, base_step, _direction_source in axes:
                effective_direction = (
                    int(directions_override.get(int(local_index), direction))
                    if directions_override is not None
                    else (
                        int(direction_override)
                        if direction_override in (-1, 1)
                        else int(direction)
                    )
                )
                if effective_direction not in (-1, 1):
                    return
                probe_step = float(base_step) * float(radius_factor)
                command_step = float(base_step) * float(command_radius_factor)
                if probe_step <= 1e-12 or command_step <= 1e-12:
                    return
                if directional_enabled:
                    # Candidate motion is integrated from the last admitted
                    # qref.  Starting from a raw branch that has crossed the
                    # singular plane would otherwise make an outward sign
                    # become an inward clipped command after
                    # ``_limit_full_joint_step``.
                    candidate_q[full_index] = reference[full_index]
                    probe_q[full_index] = reference[full_index]
                candidate_q[full_index] += float(effective_direction) * command_step
                probe_q[full_index] += float(effective_direction) * probe_step
                label = f"{side}:j{int(local_index) + 1}"
                direction_map[label] = effective_direction
                step_degrees[label] = float(np.rad2deg(command_step))
                probe_step_degrees[label] = float(np.rad2deg(probe_step))
            if any(
                candidate_q[full_index] < q_min[full_index] - 1e-9
                or candidate_q[full_index] > q_max[full_index] + 1e-9
                for _local_index, full_index, _direction, _base_step, _source in axes
            ):
                return
            candidate_local = self._local_proxy_singularity_metadata(candidate_q)
            candidate_side = dict(candidate_local["sides"][side])
            before_side = dict(reference_local["sides"][side])
            # Only the side/joints this candidate is escaping may certify
            # singularity progress.  A raw branch flip on the other arm must
            # not make a left-arm escape look better (or worse) and cause the
            # selector to alternate sides.
            count_progress = int(bool(before_side.get("is_singular", False))) - int(
                bool(candidate_side.get("is_singular", False))
            )
            depth_progress = float(before_side.get("singularity_depth_score", 0.0)) - float(
                candidate_side.get("singularity_depth_score", 0.0)
            )
            probe_valid = not any(
                probe_q[full_index] < q_min[full_index] - 1e-9
                or probe_q[full_index] > q_max[full_index] + 1e-9
                for _local_index, full_index, _direction, _base_step, _source in axes
            )
            if probe_valid:
                probe_local = self._local_proxy_singularity_metadata(probe_q)
                probe_side = dict(probe_local["sides"].get(side, {}) or {})
                probe_count_progress = int(bool(before_side.get("is_singular", False))) - int(
                    bool(probe_side.get("is_singular", False))
                )
                probe_depth_progress = float(
                    before_side.get("singularity_depth_score", 0.0)
                ) - float(probe_side.get("singularity_depth_score", 0.0))
                probe_metrics = self._dual_pose_metrics(probe_q, current_targets)
                probe_pose_progress = before_score - self._dual_pose_score(probe_metrics)
            else:
                probe_local = {}
                probe_side = {}
                probe_count_progress = 0
                probe_depth_progress = 0.0
                probe_pose_progress = 0.0
            wide_plan_active = bool(
                wide_probe
                and probe_valid
                and (
                    probe_count_progress > 0
                    or probe_depth_progress > 0.0
                )
            )
            after_metrics = self._dual_pose_metrics(candidate_q, current_targets)
            pose_progress = before_score - self._dual_pose_score(after_metrics)
            combined = len(axes) > 1
            local_index = min(int(axis[0]) for axis in axes)
            signed_direction = (
                int(axes[0][2]) if len(axes) == 1 else 0
            )
            scored.append(
                (
                    0 if not bool(candidate_side.get("is_singular", False)) else 1,
                    -float(pose_progress),
                    -float(depth_progress),
                    -int(count_progress),
                    -float(radius_factor),
                    -int(combined),
                    int(local_index),
                    side,
                    candidate_q,
                    {
                        "local_step_singularity_escape_candidate": True,
                        "local_step_singularity_escape_side": side,
                        "local_step_singularity_escape_joint_index": (
                            int(local_index) if not combined else None
                        ),
                        "local_step_singularity_escape_full_joint_index": (
                            int(axes[0][1]) if not combined else None
                        ),
                        "local_step_singularity_escape_direction": (
                            signed_direction if not combined else None
                        ),
                        "local_step_singularity_escape_direction_by_joint": direction_map,
                        "local_step_singularity_escape_directional": bool(directional_enabled),
                        "local_step_singularity_escape_combined": bool(combined),
                        "local_step_singularity_escape_step_deg": (
                            float(step_degrees[next(iter(step_degrees))])
                            if not combined
                            else None
                        ),
                        "local_step_singularity_escape_step_degrees_by_joint": step_degrees,
                        "local_step_singularity_escape_probe_step_degrees_by_joint": (
                            probe_step_degrees
                        ),
                        "local_step_singularity_escape_step_fraction": float(step_fraction),
                        "local_step_singularity_escape_radius_factor": float(radius_factor),
                        "local_step_singularity_escape_command_radius_factor": float(
                            command_radius_factor
                        ),
                        "local_step_singularity_escape_wide_search": bool(
                            wide_search_enabled
                            and wide_search_far_target_sides.get(side, False)
                        ),
                        "local_step_singularity_escape_wide_probe": bool(wide_probe),
                        "local_step_singularity_escape_wide_probe_valid": bool(
                            probe_valid
                        ),
                        "local_step_singularity_escape_wide_probe_q": (
                            probe_q.copy() if wide_probe and probe_valid else None
                        ),
                        "local_step_singularity_escape_wide_probe_local": (
                            probe_local if wide_probe and probe_valid else {}
                        ),
                        "local_step_singularity_escape_wide_probe_count_progress": int(
                            probe_count_progress
                        ),
                        "local_step_singularity_escape_wide_probe_depth_progress": float(
                            probe_depth_progress
                        ),
                        "local_step_singularity_escape_wide_probe_pose_progress": float(
                            probe_pose_progress
                        ),
                        "local_step_singularity_escape_wide_plan_active": bool(
                            wide_plan_active
                        ),
                        "local_step_singularity_escape_target_error_m": float(
                            before_metrics[side]["translation_m"]
                        ),
                        "local_step_singularity_escape_target_error_deg": float(
                            before_metrics[side]["rotation_deg"]
                        ),
                        "local_step_singularity_escape_reference": reference_local,
                        "local_step_singularity_escape_candidate_local": candidate_local,
                        "local_step_singularity_escape_side_reference": before_side,
                        "local_step_singularity_escape_side_candidate": candidate_side,
                        "local_step_singularity_escape_singular_side_count_progress": int(count_progress),
                        "local_step_singularity_escape_depth_progress": float(depth_progress),
                        "local_step_singularity_escape_side_count_progress": int(
                            count_progress
                        ),
                        "local_step_singularity_escape_side_depth_progress": float(
                            depth_progress
                        ),
                        "local_step_singularity_escape_seed_active": bool(
                            seed_local.get("sides", {})
                            .get(side, {})
                            .get("is_singular", False)
                        ),
                        "local_step_singularity_escape_predicted_pose_progress": float(pose_progress),
                        "local_step_singularity_escape_direction_source": {
                            f"{side}:j{int(axis[0]) + 1}": str(axis[4]) for axis in axes
                        },
                    },
                )
            )

        def choose_combined_directions(
            *,
            side: str,
            axes: Sequence[tuple[int, int, int, float, str]],
            radius_factor: float,
        ) -> dict[int, int] | None:
            """Choose one coupled J3/J5 sign pair before scoring candidates.

            A coupled wrist/elbow pocket can require the two coordinates to
            move with opposite signs.  Emitting all four sign combinations
            each frame, however, recreates the old oscillating selector.  We
            therefore evaluate the four bounded FK probes once, rank them by
            singularity-depth/count progress first and Cartesian progress
            second, and emit exactly one pair for this radius.  The selected
            pair is recorded in candidate metadata and latched only after a
            candidate is admitted.
            """

            if len(axes) < 2:
                return None
            active_axis_plan = tuple(
                int(index)
                for index in self._singularity_escape_axis_latch.get(side, ())
            )
            if set(active_axis_plan) >= {int(axis[0]) for axis in axes}:
                latched_directions: dict[int, int] = {}
                for axis in axes:
                    local_index = int(axis[0])
                    latched = int(
                        self._singularity_escape_direction_latch.get(
                            (side, local_index),
                            0,
                        )
                    )
                    if latched not in (-1, 1):
                        break
                    latched_directions[local_index] = latched
                else:
                    # Once a coupled plan has been admitted, never recompute
                    # its sign pair from a noisy raw target.
                    return latched_directions
            options: list[tuple[tuple[float, float, float, tuple[int, ...]], dict[int, int]]] = []
            before_side = dict(reference_local.get("sides", {}).get(side, {}) or {})
            before_depth = float(before_side.get("singularity_depth_score", 0.0) or 0.0)
            before_count = int(bool(before_side.get("is_singular", False)))
            for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                probe_q = reference.copy()
                direction_map: dict[int, int] = {}
                valid = True
                for axis, sign in zip(axes, signs, strict=True):
                    local_index, full_index, _direction, base_step, _source = axis
                    value = float(reference[full_index]) + float(sign) * float(base_step) * float(radius_factor)
                    if value < q_min[full_index] - 1e-9 or value > q_max[full_index] + 1e-9:
                        valid = False
                        break
                    probe_q[full_index] = value
                    direction_map[int(local_index)] = int(sign)
                if not valid:
                    continue
                probe_local = self._local_proxy_singularity_metadata(probe_q)
                probe_side = dict(probe_local.get("sides", {}).get(side, {}) or {})
                depth_progress = before_depth - float(
                    probe_side.get("singularity_depth_score", 0.0)
                )
                count_progress = before_count - int(bool(probe_side.get("is_singular", False)))
                pose_progress = before_score - self._dual_pose_score(
                    self._dual_pose_metrics(probe_q, current_targets)
                )
                # Prefer actually leaving the local pocket, then choose the
                # direction that follows the operator target.  The final
                # candidate evaluator remains authoritative for bounded pose
                # worsening and collision safety.
                key = (
                    0.0 if count_progress > 0 else 1.0,
                    -float(depth_progress),
                    -float(pose_progress),
                    tuple(-int(sign) for sign in signs),
                )
                options.append((key, direction_map))
            if not options:
                return None
            options.sort(key=lambda item: item[0])
            return dict(options[0][1])

        for side in ("left", "right"):
            axes = axis_by_side[side]
            if not axes:
                continue
            side_radius_factors = (
                radius_factors
                if wide_search_enabled and wide_search_far_target_sides.get(side, False)
                else [1.0]
            )
            for radius_factor in side_radius_factors:
                # A wrist+elbow pocket often cannot be escaped by changing
                # either coordinate alone.  Search the coupled outward move
                # as an additional candidate, still under the same final
                # execution step/collision/admission limits.
                if bool(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_combined_enabled",
                        False,
                    )
                ) and len(axes) > 1:
                    combined_directions = (
                        choose_combined_directions(
                            side=side,
                            axes=axes,
                            radius_factor=radius_factor,
                        )
                        if directional_enabled
                        else None
                    )
                    append_scored_candidate(
                        side=side,
                        radius_factor=radius_factor,
                        axes=axes,
                        directions_override=combined_directions,
                    )
                for axis in axes:
                    if directional_enabled:
                        append_scored_candidate(
                            side=side,
                            radius_factor=radius_factor,
                            axes=(axis,),
                        )
                    else:
                        append_scored_candidate(
                            side=side,
                            radius_factor=radius_factor,
                            axes=(axis,),
                            direction_override=-1,
                        )
                        append_scored_candidate(
                            side=side,
                            radius_factor=radius_factor,
                            axes=(axis,),
                            direction_override=1,
                        )
        scored.sort(key=lambda item: item[:8])
        candidates: list[_DualArmCandidate] = []
        seen: list[np.ndarray] = []
        for (
            _rank,
            _pose,
            _depth,
            _count,
            _radius,
            _combined,
            local_index,
            side,
            candidate_q,
            candidate_metadata,
        ) in scored:
            if len(candidates) >= max_candidates:
                break
            if any(np.allclose(candidate_q, existing, atol=1e-9) for existing in seen):
                continue
            seen.append(candidate_q.copy())
            combined = bool(candidate_metadata.get("local_step_singularity_escape_combined", False))
            direction = candidate_metadata.get("local_step_singularity_escape_direction")
            direction_label = (
                "combined"
                if combined
                else "pos"
                if int(direction or 0) > 0
                else "neg"
            )
            joint_label = (
                "j3_j5"
                if combined
                else f"j{int(local_index) + 1}"
            )
            candidates.append(
                _DualArmCandidate(
                    name=f"singularity_escape_{side}_{joint_label}_{direction_label}",
                    qpos=candidate_q.copy(),
                    source="singularity_escape",
                    method=f"singularity_escape_{side}_{joint_label}_{direction_label}",
                    metadata=candidate_metadata,
                )
            )
        metadata.update(
            {
                "status": "active" if candidates else "no_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "step_fraction": float(step_fraction),
                "wide_search_enabled": bool(wide_search_enabled),
                "wide_search_target_error_m": float(max_position_error_m),
                "wide_search_target_error_deg": float(max_orientation_error_deg),
                "wide_search_far_target_sides": [
                    side for side, enabled in wide_search_far_target_sides.items() if enabled
                ],
                "wide_search_position_threshold_m": float(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_wide_search_position_m",
                        0.10,
                    )
                ),
                "wide_search_orientation_threshold_deg": float(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_wide_search_orientation_deg",
                        20.0,
                    )
                ),
                "wide_search_radius_factors": [float(value) for value in radius_factors],
                "directional_enabled": bool(directional_enabled),
                "combined_enabled": bool(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_combined_enabled",
                        False,
                    )
                ),
                "direction_latches": {
                    f"{side}:j{int(local_index) + 1}": int(direction)
                    for (side, local_index), direction in self._singularity_escape_direction_latch.items()
                },
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

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

    def _joint_limit_guard_mode(self) -> str:
        return str(
            getattr(
                self.config,
                "joint_limit_guard_mode",
                JOINT_LIMIT_GUARD_LEGACY,
            )
        )

    def _joint_limit_forced_escape_latch_enabled(self) -> bool:
        return bool(
            self._joint_limit_guard_mode()
            == JOINT_LIMIT_GUARD_FORCED_ESCAPE_LATCHED
        )

    def _joint_limit_boundary_hold_enabled(self) -> bool:
        return bool(
            self._joint_limit_guard_mode()
            == JOINT_LIMIT_GUARD_BOUNDARY_HOLD
        )

    def _joint_limit_escape_candidate_key(
        self,
        metadata: Mapping[str, object],
    ) -> tuple[str, int] | None:
        side = str(metadata.get("local_step_limit_escape_side", "")).strip().lower()
        try:
            local_index = int(metadata.get("local_step_limit_escape_joint_index", -1))
            direction = int(metadata.get("local_step_limit_escape_direction", 0))
        except (TypeError, ValueError):
            return None
        if side not in {"left", "right"}:
            return None
        if local_index < 0 or local_index >= 6 or direction not in {-1, 1}:
            return None
        return side, local_index

    def _joint_limit_escape_raw_moves_inward(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        key: tuple[str, int],
        direction: int,
    ) -> bool:
        if int(direction) not in {-1, 1}:
            return False
        raw = np.asarray(raw_full_q, dtype=float).reshape(-1)
        reference = np.asarray(reference_full_q, dtype=float).reshape(raw.shape)
        entry_by_key = {
            (side, int(local_index)): int(full_index)
            for side, local_index, full_index in self._arm_joint_entries(reference)
        }
        full_index = entry_by_key.get(key)
        if full_index is None or not np.isfinite(raw[full_index]):
            return False
        # ``direction`` is the outward direction.  A negative signed delta
        # therefore means that raw IK is asking to move inward.
        return float(direction) * float(raw[full_index] - reference[full_index]) < -1e-9

    def _refresh_joint_limit_escape_release_guards(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        fixed_sides: set[str] | None = None,
    ) -> dict[tuple[str, int], int]:
        """Keep a released direction only while raw IK still asks inward."""

        safety_profile = self.ik_safety_profile
        if (
            not self._joint_limit_forced_escape_latch_enabled()
            or safety_profile is None
            or not bool(
            getattr(safety_profile, "local_step_limit_escape_enabled", False)
            )
        ):
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
            return {}
        fixed = set(fixed_sides or ())
        for key, direction in list(self._joint_limit_escape_release_guard.items()):
            if (
                key[0] in fixed
                or not self._joint_limit_escape_raw_moves_inward(
                    raw_full_q=raw_full_q,
                    reference_full_q=reference_full_q,
                    key=key,
                    direction=direction,
                )
            ):
                self._joint_limit_escape_release_guard.pop(key, None)
                self._joint_limit_escape_direction_latch.pop(key, None)
                continue
            # The normal latch refresh path is intentionally unaware of raw
            # target intent; mirror the guarded direction into its existing
            # authoritative latch map before it evaluates the release band.
            self._joint_limit_escape_direction_latch[key] = int(direction)
        return dict(self._joint_limit_escape_release_guard)

    def _refresh_joint_limit_escape_latches(
        self,
        *,
        reference_full_q: np.ndarray,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ) -> dict[tuple[str, int], int]:
        """Keep admitted limit-escape directions until the release band clears.

        This state is deliberately narrow: it only constrains a direction that
        was already admitted as a physical limit escape, and it is cleared on
        reset or when the corresponding joint is safely away from that limit.
        """

        safety_profile = self.ik_safety_profile
        if (
            not self._joint_limit_forced_escape_latch_enabled()
            or safety_profile is None
            or not bool(
            getattr(safety_profile, "local_step_limit_escape_enabled", False)
            )
        ):
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
            return {}

        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        lower = (
            np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float).reshape(q.shape)
            if q_min is None
            else np.asarray(q_min, dtype=float).reshape(q.shape)
        )
        upper = (
            np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float).reshape(q.shape)
            if q_max is None
            else np.asarray(q_max, dtype=float).reshape(q.shape)
        )
        if np.any(upper <= lower):
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
            return {}

        activation_margin_rad = np.deg2rad(
            max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_activation_margin_deg",
                        2.0,
                    )
                ),
            )
        )
        hysteresis_rad = np.deg2rad(
            max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_direction_hysteresis_deg",
                        0.0,
                    )
                ),
            )
        )
        release_margin_rad = activation_margin_rad + hysteresis_rad
        entry_by_key = {
            (side, int(local_index)): int(full_index)
            for side, local_index, full_index in self._arm_joint_entries(q)
        }
        active: dict[tuple[str, int], int] = {}
        for key, direction_value in list(self._joint_limit_escape_direction_latch.items()):
            full_index = entry_by_key.get(key)
            direction = int(direction_value)
            if (
                full_index is None
                or direction not in {-1, 1}
                or not np.isfinite(q[full_index])
            ):
                self._joint_limit_escape_direction_latch.pop(key, None)
                continue
            lower_margin = float(q[full_index] - lower[full_index])
            upper_margin = float(upper[full_index] - q[full_index])
            min_margin = min(lower_margin, upper_margin)
            outward_direction = -1 if upper_margin <= lower_margin else 1
            if direction != outward_direction:
                self._joint_limit_escape_direction_latch.pop(key, None)
                self._joint_limit_escape_release_guard.pop(key, None)
                continue
            if (
                min_margin >= release_margin_rad - 1e-9
                and key not in self._joint_limit_escape_release_guard
            ):
                self._joint_limit_escape_direction_latch.pop(key, None)
                continue
            active[key] = direction
        return active

    def _active_joint_limit_escape_directions(
        self,
        *,
        reference_full_q: np.ndarray,
        target_activity: Mapping[str, object] | None = None,
    ) -> dict[tuple[str, int], int]:
        """Return latched or currently required outward limit directions.

        A direction latch only exists after a limit-escape candidate has been
        admitted.  The current reference can nevertheless already be inside
        the activation band on the first protected frame.  Include that
        provisional direction here so ordinary IK cannot reselect an inward
        step before the latch is armed.
        """

        safety_profile = self.ik_safety_profile
        if (
            not self._joint_limit_forced_escape_latch_enabled()
            or safety_profile is None
            or not bool(
            getattr(safety_profile, "local_step_limit_escape_enabled", False)
            )
        ):
            return {}

        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        q_min = np.asarray(
            self.ik_solver.robot.model.lowerPositionLimit,
            dtype=float,
        ).reshape(q.shape)
        q_max = np.asarray(
            self.ik_solver.robot.model.upperPositionLimit,
            dtype=float,
        ).reshape(q.shape)
        if np.any(q_max <= q_min):
            return {}

        fixed_sides = self._authority_fixed_sides(target_activity)
        directions = {
            key: direction
            for key, direction in self._refresh_joint_limit_escape_latches(
                reference_full_q=q,
                q_min=q_min,
                q_max=q_max,
            ).items()
            if key[0] not in fixed_sides
        }
        activation_margin_rad = np.deg2rad(
            max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_activation_margin_deg",
                        2.0,
                    )
                ),
            )
        )
        for side, local_index, full_index in self._arm_joint_entries(q):
            if side in fixed_sides:
                continue
            if not np.isfinite(q[full_index]):
                continue
            lower_margin = float(q[full_index] - q_min[full_index])
            upper_margin = float(q_max[full_index] - q[full_index])
            if not np.isfinite(lower_margin) or not np.isfinite(upper_margin):
                continue
            if min(lower_margin, upper_margin) > activation_margin_rad:
                continue
            key = (side, int(local_index))
            directions.setdefault(
                key,
                -1 if upper_margin <= lower_margin else 1,
            )
        return directions

    def _active_joint_limit_boundary_directions(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray | None = None,
        target_activity: Mapping[str, object] | None = None,
    ) -> dict[tuple[str, int], int]:
        """Return outward directions inside the narrow hard-limit guard band.

        Unlike the legacy escape candidate this helper is stateless and does
        not manufacture an outward joint step.  It exists only for the
        opt-in ``boundary_hold`` mode, where an inward component is projected
        back to the already admitted reference for that joint.
        """

        if not self._joint_limit_boundary_hold_enabled():
            return {}
        q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate = (
            q
            if candidate_full_q is None
            else np.asarray(candidate_full_q, dtype=float).reshape(q.shape)
        )
        lower = np.asarray(
            self.ik_solver.robot.model.lowerPositionLimit,
            dtype=float,
        ).reshape(q.shape)
        upper = np.asarray(
            self.ik_solver.robot.model.upperPositionLimit,
            dtype=float,
        ).reshape(q.shape)
        if np.any(upper <= lower):
            return {}
        margin_rad = np.deg2rad(
            max(
                0.0,
                float(self.config.joint_limit_boundary_guard_margin_deg),
            )
        )
        fixed_sides = self._authority_fixed_sides(target_activity)
        directions: dict[tuple[str, int], int] = {}
        for side, local_index, full_index in self._arm_joint_entries(q):
            if side in fixed_sides or not np.isfinite(q[full_index]):
                continue
            reference_lower_margin = float(q[full_index] - lower[full_index])
            reference_upper_margin = float(upper[full_index] - q[full_index])
            candidate_lower_margin = float(
                candidate[full_index] - lower[full_index]
            )
            candidate_upper_margin = float(
                upper[full_index] - candidate[full_index]
            )
            lower_margin = min(reference_lower_margin, candidate_lower_margin)
            upper_margin = min(reference_upper_margin, candidate_upper_margin)
            if min(lower_margin, upper_margin) > margin_rad:
                continue
            directions[(side, int(local_index))] = (
                -1 if upper_margin <= lower_margin else 1
            )
        return directions

    def _joint_limit_boundary_hold_candidate(
        self,
        *,
        ordinary_q: np.ndarray,
        reference_full_q: np.ndarray,
        directions: Mapping[tuple[str, int], int],
    ) -> _DualArmCandidate | None:
        """Project only inward near-limit joint components to zero motion."""

        if not directions:
            return None
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        projected = np.asarray(ordinary_q, dtype=float).reshape(
            reference.shape
        ).copy()
        entry_by_key = {
            (side, int(local_index)): int(full_index)
            for side, local_index, full_index in self._arm_joint_entries(
                reference
            )
        }
        held: dict[str, int] = {}
        for key, outward_direction in directions.items():
            full_index = entry_by_key.get(key)
            if full_index is None:
                continue
            delta = float(projected[full_index] - reference[full_index])
            if float(outward_direction) * delta >= -1e-9:
                continue
            projected[full_index] = reference[full_index]
            held[f"{key[0]}:j{int(key[1]) + 1}"] = int(outward_direction)
        if not held:
            return None
        return _DualArmCandidate(
            name="joint_limit_boundary_hold",
            qpos=projected,
            source="joint_limit_boundary_hold",
            method="project_inward_limit_components",
            metadata={
                "joint_limit_boundary_hold_candidate": True,
                "held_outward_direction_by_joint": held,
                "guard_margin_deg": float(
                    self.config.joint_limit_boundary_guard_margin_deg
                ),
            },
        )

    def _wrist_q5_entry_boundary_hold_candidate(
        self,
        *,
        ordinary_q: np.ndarray,
        reference_full_q: np.ndarray,
        target_activity: Mapping[str, object] | None = None,
    ) -> _DualArmCandidate | None:
        """Project only a first inward q5 boundary crossing to zero motion.

        This behavior is intentionally limited to the V2 latch-required
        wrist policy.  V1 keeps its historical escape behavior as the atomic
        rollback.  Once the admitted q5 is already inside the boundary, the
        existing bounded escape remains available; this helper only prevents
        a safe outside reference from being replaced by a synthetic 4-degree
        outward pulse when the ordinary solution first requests the coaxial
        pocket.  Outward motion is never held.
        """

        enabled = bool(
            self.config.solve_step_mode
            == IK_STEP_MODE_SINGLE_STEP_SINGULARITY_AWARE
            and self.config.wrist_coaxial_continuity_enabled
            and self.config.wrist_q5_hysteresis_requires_escape_latch
        )
        if not enabled:
            return None
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        projected = np.asarray(ordinary_q, dtype=float).reshape(
            reference.shape
        ).copy()
        threshold_deg = max(
            float(self.config.wrist_singularity_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
            float(self.config.wrist_coaxial_hard_threshold_deg)
            + float(self.config.wrist_q5_escape_margin_deg),
        )
        threshold_rad = float(np.deg2rad(threshold_deg))
        step_by_side: dict[str, float] = {"left": 0.0, "right": 0.0}
        if self._full_execution_joint_step is not None:
            left_step, right_step = self._split_solution(
                np.asarray(self._full_execution_joint_step, dtype=float)
            )
            step_by_side = {
                "left": abs(float(left_step[4])),
                "right": abs(float(right_step[4])),
            }
        fixed_sides = self._authority_fixed_sides(target_activity)
        held: dict[str, dict[str, float | str]] = {}
        for side, _local_index, full_index in self._arm_joint_entries(reference):
            if side in fixed_sides or int(_local_index) != 4:
                continue
            reference_q5 = float(reference[full_index])
            candidate_q5 = float(projected[full_index])
            if not np.isfinite(reference_q5) or not np.isfinite(candidate_q5):
                continue
            # If the reference is already inside the protected boundary, an
            # outward escape is still required and the established path owns
            # that recovery.  This entry guard must never trap an unsafe
            # startup state in place.
            if abs(reference_q5) < threshold_rad - 1e-12:
                continue
            same_side = bool(reference_q5 * candidate_q5 > 0.0)
            inward = bool(
                (same_side and abs(candidate_q5) < abs(reference_q5) - 1e-12)
                or reference_q5 * candidate_q5 <= 0.0
            )
            crossing = bool(reference_q5 * candidate_q5 <= 0.0)
            crossing_near_boundary = bool(
                crossing
                and abs(reference_q5)
                <= threshold_rad + float(step_by_side.get(side, 0.0)) + 1e-12
            )
            enters_boundary = bool(
                abs(candidate_q5) < threshold_rad or crossing_near_boundary
            )
            if not inward or not enters_boundary:
                continue
            projected[full_index] = reference_q5
            held[side] = {
                "joint": "j5",
                "reference_q5_deg": float(np.rad2deg(reference_q5)),
                "ordinary_q5_deg": float(np.rad2deg(candidate_q5)),
                "threshold_deg": float(threshold_deg),
                "reason": "inward_entry_or_near_crossing",
            }
        if not held:
            return None
        return _DualArmCandidate(
            name="wrist_q5_entry_boundary_hold",
            qpos=projected,
            source="wrist_q5_boundary_hold",
            method="project_inward_wrist_q5_entry",
            metadata={
                "wrist_q5_boundary_hold_candidate": True,
                "held_sides": held,
                "escape_threshold_deg": float(threshold_deg),
            },
        )

    def _joint_limit_escape_latch_metadata(
        self,
        latches: Mapping[tuple[str, int], int],
    ) -> dict[str, int]:
        return {
            f"{side}:j{int(local_index) + 1}": int(direction)
            for (side, local_index), direction in sorted(latches.items())
        }

    def _joint_limit_escape_candidate_moves_outward(
        self,
        *,
        item: _DualArmEvaluatedCandidate,
        reference_full_q: np.ndarray,
        latches: Mapping[tuple[str, int], int],
    ) -> bool:
        return self._joint_limit_escape_output_moves_outward(
            output_q=item.output_qpos,
            reference_full_q=reference_full_q,
            directions=latches,
        )

    def _joint_limit_escape_output_moves_outward(
        self,
        *,
        output_q: np.ndarray,
        reference_full_q: np.ndarray,
        directions: Mapping[tuple[str, int], int],
    ) -> bool:
        if not directions:
            return True
        reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
        entry_by_key = {
            (side, int(local_index)): int(full_index)
            for side, local_index, full_index in self._arm_joint_entries(reference)
        }
        output = np.asarray(output_q, dtype=float).reshape(reference.shape)
        for key, direction in directions.items():
            full_index = entry_by_key.get(key)
            if full_index is None:
                return False
            if float(direction) * float(output[full_index] - reference[full_index]) < -1e-9:
                return False
        return True

    def _update_joint_limit_escape_direction_latch(
        self,
        *,
        reference_full_q: np.ndarray,
        admitted_full_q: np.ndarray,
        raw_full_q: np.ndarray | None = None,
        status: str,
        selection_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Arm a physical-limit direction only after its qref is admitted."""

        safety_profile = self.ik_safety_profile
        enabled = bool(
            self._joint_limit_forced_escape_latch_enabled()
            and safety_profile is not None
            and getattr(safety_profile, "local_step_limit_escape_enabled", False)
        )
        if not enabled:
            self._joint_limit_escape_direction_latch.clear()
            self._joint_limit_escape_release_guard.clear()
        before = dict(self._joint_limit_escape_direction_latch)
        if raw_full_q is not None:
            self._refresh_joint_limit_escape_release_guards(
                raw_full_q=np.asarray(raw_full_q, dtype=float),
                reference_full_q=np.asarray(reference_full_q, dtype=float),
            )
        active_before = self._refresh_joint_limit_escape_latches(
            reference_full_q=np.asarray(admitted_full_q, dtype=float),
        )
        armed_key: tuple[str, int] | None = None
        if enabled and status in {"tracking", "boundary_active"} and selection_metadata:
            selected_source = str(selection_metadata.get("selected_source", ""))
            selected_metadata = selection_metadata.get("selected_metadata", {})
            selected_metadata = (
                selected_metadata
                if isinstance(selected_metadata, Mapping)
                else {}
            )
            candidate_metadata = selected_metadata.get("candidate_metadata", {})
            candidate_metadata = (
                candidate_metadata
                if isinstance(candidate_metadata, Mapping)
                else {}
            )
            key = self._joint_limit_escape_candidate_key(candidate_metadata)
            direction = int(
                candidate_metadata.get("local_step_limit_escape_direction", 0) or 0
            )
            full_index = candidate_metadata.get("local_step_limit_escape_full_joint_index")
            try:
                full_index_int = int(full_index)
            except (TypeError, ValueError):
                full_index_int = -1
            reference = np.asarray(reference_full_q, dtype=float).reshape(-1)
            admitted = np.asarray(admitted_full_q, dtype=float).reshape(reference.shape)
            margin_progress_deg = float(
                candidate_metadata.get(
                    "local_step_limit_escape_joint_margin_progress_deg",
                    candidate_metadata.get(
                        "local_step_limit_escape_margin_progress_deg",
                        0.0,
                    ),
                )
                or 0.0
            )
            if (
                selected_source == "joint_limit_escape"
                and key is not None
                and direction in {-1, 1}
                and 0 <= full_index_int < reference.shape[0]
                and margin_progress_deg > 0.0
                and float(direction) * float(admitted[full_index_int] - reference[full_index_int])
                > 1e-9
            ):
                self._joint_limit_escape_direction_latch[key] = direction
                armed_key = key
                if raw_full_q is not None:
                    raw = np.asarray(raw_full_q, dtype=float).reshape(reference.shape)
                    lower = np.asarray(
                        self.ik_solver.robot.model.lowerPositionLimit,
                        dtype=float,
                    ).reshape(reference.shape)
                    upper = np.asarray(
                        self.ik_solver.robot.model.upperPositionLimit,
                        dtype=float,
                    ).reshape(reference.shape)
                    release_margin_rad = np.deg2rad(
                        max(
                            0.0,
                            float(
                                getattr(
                                    safety_profile,
                                    "local_step_limit_escape_activation_margin_deg",
                                    2.0,
                                )
                            ),
                        )
                        + max(
                            0.0,
                            float(
                                getattr(
                                    safety_profile,
                                    "local_step_limit_escape_direction_hysteresis_deg",
                                    0.0,
                                )
                            ),
                        )
                    )
                    full_lower_margin = float(admitted[full_index_int] - lower[full_index_int])
                    full_upper_margin = float(upper[full_index_int] - admitted[full_index_int])
                    if (
                        min(full_lower_margin, full_upper_margin)
                        >= release_margin_rad - 1e-9
                        and self._joint_limit_escape_raw_moves_inward(
                            raw_full_q=raw,
                            reference_full_q=reference,
                            key=key,
                            direction=direction,
                        )
                    ):
                        self._joint_limit_escape_release_guard[key] = direction

        active_after = self._refresh_joint_limit_escape_latches(
            reference_full_q=np.asarray(admitted_full_q, dtype=float),
        )
        activation_margin_deg = (
            0.0
            if safety_profile is None
            else max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_activation_margin_deg",
                        2.0,
                    )
                ),
            )
        )
        hysteresis_deg = (
            0.0
            if safety_profile is None
            else max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_limit_escape_direction_hysteresis_deg",
                        0.0,
                    )
                ),
            )
        )
        return {
            "enabled": enabled,
            "activation_margin_deg": activation_margin_deg,
            "hysteresis_deg": hysteresis_deg,
            "release_margin_deg": activation_margin_deg + hysteresis_deg,
            "before": self._joint_limit_escape_latch_metadata(before),
            "active_before": self._joint_limit_escape_latch_metadata(active_before),
            "armed": (
                None
                if armed_key is None
                else {
                    "joint": f"{armed_key[0]}:j{int(armed_key[1]) + 1}",
                    "direction": int(self._joint_limit_escape_direction_latch.get(armed_key, 0)),
                }
            ),
            "released": sorted(
                set(self._joint_limit_escape_latch_metadata(before))
                - set(self._joint_limit_escape_latch_metadata(active_after))
            ),
            "active": self._joint_limit_escape_latch_metadata(active_after),
        }

    def _dual_arm_limit_escape_candidates(
        self,
        *,
        reference_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
        target_activity: dict[str, object] | None = None,
        max_candidates_override: int | None = None,
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

        authority_fixed_sides = self._authority_fixed_sides(target_activity)
        arm_entries = [
            entry
            for entry in self._arm_joint_entries(q)
            if entry[0] not in authority_fixed_sides
        ]
        metadata["authority_fixed_sides"] = sorted(authority_fixed_sides)
        if not arm_entries:
            metadata["status"] = "all_sides_fixed_by_authority"
            return [], metadata
        arm_indices = np.array([entry[2] for entry in arm_entries], dtype=int)
        lower_margin = q - q_min
        upper_margin = q_max - q
        min_margin = np.minimum(lower_margin, upper_margin)
        activation_margin_rad = np.deg2rad(
            max(0.0, float(getattr(safety_profile, "local_step_limit_escape_activation_margin_deg", 2.0)))
        )
        active_latches = {
            key: direction
            for key, direction in self._refresh_joint_limit_escape_latches(
                reference_full_q=q,
                q_min=q_min,
                q_max=q_max,
            ).items()
            if key[0] not in authority_fixed_sides
        }
        hysteresis_deg = max(
            0.0,
            float(
                getattr(
                    safety_profile,
                    "local_step_limit_escape_direction_hysteresis_deg",
                    0.0,
                )
            ),
        )
        release_margin_rad = activation_margin_rad + np.deg2rad(hysteresis_deg)
        entry_by_key = {
            (side, int(local_index)): int(full_index)
            for side, local_index, full_index in arm_entries
        }
        latched_full_indices = {
            entry_by_key[key]
            for key in active_latches
            if key in entry_by_key
        }
        active_full_indices = [
            int(index)
            for index in arm_indices.tolist()
            if min_margin[int(index)] <= activation_margin_rad
            or int(index) in latched_full_indices
        ]
        metadata.update(
            {
                "status": "inactive" if not active_full_indices else "active",
                "activation_margin_deg": float(np.rad2deg(activation_margin_rad)),
                "hysteresis_deg": float(hysteresis_deg),
                "release_margin_deg": float(np.rad2deg(release_margin_rad)),
                "active_full_joint_indices": active_full_indices,
                "active_joints": [
                    f"{side}_j{local_index + 1}"
                    for side, local_index, full_index in arm_entries
                    if int(full_index) in set(active_full_indices)
                ],
                "latched_direction_by_joint": self._joint_limit_escape_latch_metadata(
                    active_latches
                ),
                "min_margin_deg": np.rad2deg(min_margin[arm_indices]).astype(float).tolist(),
            }
        )
        if not active_full_indices:
            return [], metadata

        configured_max_candidates = max(
            0,
            int(getattr(safety_profile, "local_step_limit_escape_max_candidates", 12)),
        )
        max_candidates = configured_max_candidates
        if max_candidates_override is not None:
            max_candidates = min(
                max_candidates,
                max(0, int(max_candidates_override)),
            )
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

        full_step = np.asarray(self._full_execution_joint_step, dtype=float).reshape(q.shape) * step_fraction
        active_set = set(active_full_indices)
        # In the bounded hysteresis path only one candidate can be admitted.
        # Preserve the deterministic margin/latch ordering, but defer the
        # expensive FK pose score for candidates that will not be emitted.
        defer_pose_scoring = max_candidates == 1
        before_score = (
            None
            if defer_pose_scoring
            else self._dual_pose_score(self._dual_pose_metrics(q, current_targets))
        )
        scored: list[
            tuple[int, int, float, float, int, str, int, np.ndarray, dict[str, object]]
        ] = []
        eps = 1e-9
        for side, local_index, full_index in arm_entries:
            latch_key = (side, int(local_index))
            latched_direction = active_latches.get(latch_key)
            directions = (
                (int(latched_direction),)
                if latched_direction is not None and int(full_index) in latched_full_indices
                else (-1, 1)
            )
            for direction in directions:
                if direction < 0 and q[full_index] <= q_min[full_index] + eps:
                    continue
                if direction > 0 and q[full_index] >= q_max[full_index] - eps:
                    continue
                if full_index in active_set and latched_direction is None:
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
                if defer_pose_scoring:
                    pose_progress = 0.0
                else:
                    assert before_score is not None
                    after_metrics = self._dual_pose_metrics(candidate_q, current_targets)
                    after_score = self._dual_pose_score(after_metrics)
                    pose_progress = before_score - after_score
                candidate_lower_margin = candidate_q - q_min
                candidate_upper_margin = q_max - candidate_q
                joint_margin_before = min(
                    float(lower_margin[full_index]),
                    float(upper_margin[full_index]),
                )
                joint_margin_after = min(
                    float(candidate_lower_margin[full_index]),
                    float(candidate_upper_margin[full_index]),
                )
                joint_margin_progress = joint_margin_after - joint_margin_before
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
                    "local_step_limit_escape_joint_margin_before_deg": float(
                        np.rad2deg(joint_margin_before)
                    ),
                    "local_step_limit_escape_joint_margin_after_deg": float(
                        np.rad2deg(joint_margin_after)
                    ),
                    "local_step_limit_escape_joint_margin_progress_deg": float(
                        np.rad2deg(joint_margin_progress)
                    ),
                    "local_step_limit_escape_pose_progress": float(pose_progress),
                    "local_step_limit_escape_pose_progress_deferred": bool(
                        defer_pose_scoring
                    ),
                    "local_step_limit_escape_latched": bool(latched_direction is not None),
                    "local_step_limit_escape_direction_source": (
                        "latch" if latched_direction is not None else "margin"
                    ),
                }
                scored.append(
                    (
                        # Once a direction is latched, service that joint
                        # before a newly-near joint.  Otherwise a one-candidate
                        # cap could keep servicing the newcomer while the
                        # already active escape remains parked.
                        int(latched_direction is None),
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
        for (
            _latched_priority,
            _non_active,
            _margin_score,
            _pose_score,
            local_index,
            side,
            direction,
            candidate_q,
            candidate_metadata,
        ) in scored:
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

        guarded_crossing_sides = self._wrist_branch_guard_output_crossing_sides(
            reference_full_q=reference_full,
            output_full_q=output_q,
        )
        metadata["wrist_branch_guard_output_crossing_sides"] = list(
            guarded_crossing_sides
        )
        # Keep the locally integrated/original Pink result available so an
        # operator can deliberately leave the old q5 side.  The guard is for
        # alternate branches and auxiliary posture candidates, which are the
        # paths that fabricate the +/-pi jump; a raw crossing is handled by
        # the existing q5 escape and final SDK admission.
        if guarded_crossing_sides and candidate.name != "original":
            metadata["reject_reason"] = (
                "wrist_branch_guard_output_crossing:"
                + ",".join(guarded_crossing_sides)
            )
            metadata["score"] = 1e18
            return False, output_q, metadata

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
        joint_limit_boundary_hold_candidate = bool(
            candidate.metadata.get("joint_limit_boundary_hold_candidate", False)
        )
        wrist_q5_boundary_hold_candidate = bool(
            candidate.metadata.get("wrist_q5_boundary_hold_candidate", False)
        )
        boundary_hold_candidate = bool(
            joint_limit_boundary_hold_candidate
            or wrist_q5_boundary_hold_candidate
        )
        singularity_escape_candidate = bool(
            candidate.metadata.get("local_step_singularity_escape_candidate", False)
        )
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
        singularity_escape_local_before = dict(
            candidate.metadata.get("local_step_singularity_escape_reference", {}) or {}
        )
        singularity_escape_local_after = (
            self._local_proxy_singularity_metadata(output_q)
            if singularity_escape_candidate
            else {}
        )
        singularity_escape_side = str(
            candidate.metadata.get("local_step_singularity_escape_side", "") or ""
        )
        if not singularity_escape_side and singularity_escape_local_before:
            singular_before_sides = [
                str(side)
                for side, side_metadata in dict(
                    singularity_escape_local_before.get("sides", {}) or {}
                ).items()
                if bool(dict(side_metadata or {}).get("is_singular", False))
            ]
            if len(singular_before_sides) == 1:
                # Compatibility for recorded/hand-built candidates created
                # before the side-specific escape metadata was introduced.
                singularity_escape_side = singular_before_sides[0]
        singularity_escape_side_before = dict(
            singularity_escape_local_before.get("sides", {})
            .get(singularity_escape_side, {})
            or {}
        )
        singularity_escape_side_after = dict(
            singularity_escape_local_after.get("sides", {})
            .get(singularity_escape_side, {})
            or {}
        )
        singularity_escape_wide_search = bool(
            singularity_escape_candidate
            and candidate.metadata.get(
                "local_step_singularity_escape_wide_search", False
            )
        )
        singularity_escape_count_progress = (
            int(bool(singularity_escape_side_before.get("is_singular", False)))
            - int(bool(singularity_escape_side_after.get("is_singular", False)))
            if singularity_escape_candidate
            else 0
        )
        singularity_escape_depth_progress = (
            float(singularity_escape_side_before.get("singularity_depth_score", 0.0) or 0.0)
            - float(singularity_escape_side_after.get("singularity_depth_score", 0.0) or 0.0)
            if singularity_escape_candidate
            else 0.0
        )
        if singularity_escape_candidate and not singularity_escape_side:
            # Legacy/hand-built candidate payloads may not identify a side.
            # Fall back to the aggregate proxy only for that compatibility
            # shape; all new directional candidates carry a side and are
            # evaluated side-locally above.
            singularity_escape_count_progress = (
                int(singularity_escape_local_before.get("singular_side_count", 0) or 0)
                - int(singularity_escape_local_after.get("singular_side_count", 0) or 0)
            )
            singularity_escape_depth_progress = (
                float(singularity_escape_local_before.get("singularity_depth_score", 0.0) or 0.0)
                - float(singularity_escape_local_after.get("singularity_depth_score", 0.0) or 0.0)
            )
        direction_map = dict(
            candidate.metadata.get(
                "local_step_singularity_escape_direction_by_joint", {}
            )
            or {}
        )
        directional_outward = False
        if singularity_escape_candidate and direction_map:
            reference_by_side = dict(
                zip(("left", "right"), self._split_solution(reference_full))
            )
            output_by_side = dict(
                zip(("left", "right"), self._split_solution(output_q))
            )
            for label, direction in direction_map.items():
                try:
                    side_text, joint_text = str(label).split(":j", 1)
                    local_index = int(joint_text) - 1
                    direction_int = int(direction)
                except (TypeError, ValueError):
                    continue
                if side_text not in reference_by_side or direction_int not in (-1, 1):
                    continue
                delta = float(
                    output_by_side[side_text][local_index]
                    - reference_by_side[side_text][local_index]
                )
                if direction_int * delta > 1e-7:
                    directional_outward = True
                    break
        candidate_escape_metadata = dict(candidate.metadata)
        singularity_escape_wide_probe_count_progress = int(
            candidate_escape_metadata.get(
                "local_step_singularity_escape_wide_probe_count_progress",
                0,
            )
            or 0
        )
        singularity_escape_wide_probe_depth_progress = float(
            candidate_escape_metadata.get(
                "local_step_singularity_escape_wide_probe_depth_progress",
                0.0,
            )
            or 0.0
        )
        singularity_escape_wide_plan_active = bool(
            singularity_escape_candidate
            and singularity_escape_wide_search
            and candidate_escape_metadata.get(
                "local_step_singularity_escape_wide_plan_active",
                False,
            )
            and (
                singularity_escape_wide_probe_count_progress > 0
                or singularity_escape_wide_probe_depth_progress > 0.0
            )
        )
        singularity_escape_max_position_worsen_m = max(
            float(worsen_limits["position_m"]),
            max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_max_position_worsen_m",
                        0.02,
                    )
                ),
            ),
        )
        singularity_escape_max_orientation_worsen_deg = max(
            float(worsen_limits["orientation_deg"]),
            max(
                0.0,
                float(
                    getattr(
                        safety_profile,
                        "local_step_singularity_escape_max_orientation_worsen_deg",
                        8.0,
                    )
                ),
            ),
        )
        if singularity_escape_wide_search:
            singularity_escape_max_position_worsen_m = max(
                singularity_escape_max_position_worsen_m,
                max(
                    0.0,
                    float(
                        getattr(
                            safety_profile,
                            "local_step_singularity_escape_wide_search_max_position_worsen_m",
                            0.05,
                        )
                    ),
                ),
            )
            singularity_escape_max_orientation_worsen_deg = max(
                singularity_escape_max_orientation_worsen_deg,
                max(
                    0.0,
                    float(
                        getattr(
                            safety_profile,
                            "local_step_singularity_escape_wide_search_max_orientation_worsen_deg",
                            18.0,
                        )
                    ),
                ),
            )
        singularity_escape_bounded = self._dual_bounded_worsening(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            max_position_worsen_m=singularity_escape_max_position_worsen_m,
            max_orientation_worsen_deg=singularity_escape_max_orientation_worsen_deg,
        )
        singularity_escape_active = bool(
            singularity_escape_candidate
            and (
                singularity_escape_count_progress > 0
                or singularity_escape_depth_progress > 0.0
                or singularity_escape_wide_plan_active
                or directional_outward
            )
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
        elif boundary_hold_candidate and bounded_worsening:
            # The boundary candidate is derived from an ordinary candidate
            # that already passed the normal local-step gate, with only the
            # inward component of a hard-limit joint or the protected wrist
            # q5 entry projected back to the admitted reference.  Requiring
            # fresh positive Cartesian
            # progress after that projection can turn a one-joint boundary
            # stop into an unnecessary bimanual hold when the requested pose
            # itself lies beyond the reachable boundary.  Preserve the other
            # admitted components only while they remain inside the ordinary
            # (not escape-relaxed) bounded-worsening contract; collision and
            # final joint admission are still rechecked by the caller.
            progress_accept_reason = (
                "wrist_q5_boundary_hold_bounded_worsening"
                if wrist_q5_boundary_hold_candidate
                else "joint_limit_boundary_hold_bounded_worsening"
            )
        elif (
            limit_escape_candidate
            and bool(getattr(safety_profile, "local_step_limit_escape_accept_bounded_worse", True))
            and limit_escape_margin_progress_deg > 0.0
            and limit_escape_bounded
        ):
            progress_accept_reason = "joint_limit_escape_bounded_worsening"
        elif singularity_escape_active and singularity_escape_bounded:
            progress_accept_reason = (
                "singularity_escape_wide_search_bounded_worsening"
                if singularity_escape_wide_plan_active
                and singularity_escape_count_progress <= 0
                and singularity_escape_depth_progress <= 0.0
                else "singularity_escape_bounded_worsening"
            )
            if (
                directional_outward
                and singularity_escape_count_progress <= 0
                and singularity_escape_depth_progress <= 0.0
                and not singularity_escape_wide_plan_active
            ):
                progress_accept_reason = "singularity_escape_outward_traversal"
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
                    "dual_boundary_hold_candidate": bool(boundary_hold_candidate),
                    "dual_joint_limit_boundary_hold_candidate": bool(
                        joint_limit_boundary_hold_candidate
                    ),
                    "dual_wrist_q5_boundary_hold_candidate": bool(
                        wrist_q5_boundary_hold_candidate
                    ),
                    "dual_boundary_hold_bounded_worsening": bool(
                        boundary_hold_candidate and bounded_worsening
                    ),
                    "dual_singularity_escape_active": bool(singularity_escape_active),
                    "dual_singularity_escape_wide_search": bool(
                        singularity_escape_wide_search
                    ),
                    "dual_singularity_escape_max_position_worsen_m": float(
                        singularity_escape_max_position_worsen_m
                    ),
                    "dual_singularity_escape_max_orientation_worsen_deg": float(
                        singularity_escape_max_orientation_worsen_deg
                    ),
                    "dual_singularity_escape_bounded_worsening": bool(
                        singularity_escape_bounded
                    ),
                    "dual_singularity_escape_count_progress": int(
                        singularity_escape_count_progress
                    ),
                    "dual_singularity_escape_depth_progress": float(
                        singularity_escape_depth_progress
                    ),
                    "dual_singularity_escape_wide_probe_count_progress": int(
                        singularity_escape_wide_probe_count_progress
                    ),
                    "dual_singularity_escape_wide_probe_depth_progress": float(
                        singularity_escape_wide_probe_depth_progress
                    ),
                    "dual_singularity_escape_wide_plan_active": bool(
                        singularity_escape_wide_plan_active
                    ),
                    "dual_singularity_escape_directional_outward": bool(
                        directional_outward
                    ),
                    "dual_singularity_escape_directional_pose_progress": float(
                        progress
                    ),
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
                "dual_boundary_hold_candidate": bool(boundary_hold_candidate),
                "dual_joint_limit_boundary_hold_candidate": bool(
                    joint_limit_boundary_hold_candidate
                ),
                "dual_wrist_q5_boundary_hold_candidate": bool(
                    wrist_q5_boundary_hold_candidate
                ),
                "dual_boundary_hold_bounded_worsening": bool(
                    boundary_hold_candidate and bounded_worsening
                ),
                "dual_singularity_escape_active": bool(singularity_escape_active),
                "dual_singularity_escape_wide_search": bool(
                    singularity_escape_wide_search
                ),
                "dual_singularity_escape_max_position_worsen_m": float(
                    singularity_escape_max_position_worsen_m
                ),
                "dual_singularity_escape_max_orientation_worsen_deg": float(
                    singularity_escape_max_orientation_worsen_deg
                ),
                "dual_singularity_escape_bounded_worsening": bool(
                    singularity_escape_bounded
                ),
                "dual_singularity_escape_count_progress": int(
                    singularity_escape_count_progress
                ),
                "dual_singularity_escape_depth_progress": float(
                    singularity_escape_depth_progress
                ),
                "dual_singularity_escape_wide_probe_count_progress": int(
                    singularity_escape_wide_probe_count_progress
                ),
                "dual_singularity_escape_wide_probe_depth_progress": float(
                    singularity_escape_wide_probe_depth_progress
                ),
                "dual_singularity_escape_wide_plan_active": bool(
                    singularity_escape_wide_plan_active
                ),
                "dual_singularity_escape_directional_outward": bool(
                    directional_outward
                ),
                "dual_singularity_escape_directional_pose_progress": float(
                    progress
                ),
                "dual_singularity_escape_local_before": singularity_escape_local_before,
                "dual_singularity_escape_local_after": singularity_escape_local_after,
                "dual_posture_escape_active": bool(posture_escape_active),
                "dual_posture_escape_bounded_worsening": bool(posture_escape_bounded),
                "dual_wrist_unload_active": bool(wrist_unload_active),
                "dual_wrist_unload_bounded_worsening": bool(wrist_unload_bounded),
                "dual_wrist_unload_improvement_deg": float(wrist_unload_improvement_deg),
                "posture_preference": posture_metadata,
                "dual_worsen_limits": dict(worsen_limits),
                "within_pose_tolerance": bool(within_tolerance),
                "dual_recovery_state": (
                    "wrist_q5_boundary_hold"
                    if progress_accept_reason.startswith("wrist_q5_boundary_hold")
                    else "joint_limit_boundary_hold"
                    if progress_accept_reason.startswith("joint_limit_boundary_hold")
                    else (
                        "joint_limit_escape"
                        if progress_accept_reason.startswith("joint_limit_escape")
                        else (
                            "singularity_escape"
                            if progress_accept_reason.startswith("singularity_escape")
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
                        -min(
                            750_000.0,
                            # Do not give every singularity candidate an
                            # unconditional priority bonus.  The previous
                            # 150k floor made an escape that worsened the
                            # target outrank a smooth tracking candidate and
                            # was a direct source of hesitation.  Depth/count
                            # only break ties; actual admitted pose progress
                            # is the dominant directional signal.
                            max(0.0, progress) * 500_000.0
                            + max(0.0, singularity_escape_depth_progress) * 100_000.0
                            + max(0, singularity_escape_count_progress) * 250_000.0,
                        )
                        if progress_accept_reason.startswith("singularity_escape")
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
            authority_metadata = (
                source_metadata
                if source_metadata is not None
                else (
                    None
                    if cartesian_command is None
                    else cartesian_command.metadata
                )
            )
            fix_arm_fixed_side = str(
                (
                    authority_metadata.get("fix_arm_fixed_side", side)
                    if authority_metadata is not None
                    else side
                )
                or ""
            ).strip().lower()
            if (
                authority_metadata is not None
                and bool(authority_metadata.get("fix_arm_frozen", False))
                and fix_arm_fixed_side == side
            ):
                metadata[side] = {
                    "source": "fix_arm_controller",
                    "previous_target_available": self._last_target_poses.get(side)
                    is not None,
                    "translation_m": 0.0,
                    "rotation_deg": 0.0,
                    "activity_ratio": 0.0,
                    "hold_weight": 1.0,
                    "inactive": True,
                    "state": "FIXED",
                    "fixed_by_authority": True,
                    "fixed_side": side,
                }
                continue
            pose_action_intent_raw = (
                source_metadata.get("pose_action_motion_intent")
                if source_metadata is not None
                else (
                    None
                    if cartesian_command is None
                    else cartesian_command.metadata.get(
                        "pose_action_motion_intent"
                    )
                )
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
            motion_intent_raw = (
                pose_action_intent_raw
                if isinstance(pose_action_intent_raw, dict)
                else tracker_intent_raw
            )
            if (
                bool(self.config.inactive_side_hold_enabled)
                and isinstance(motion_intent_raw, dict)
                and motion_intent_raw.get("state") != "DISABLED"
            ):
                motion_intent = dict(motion_intent_raw)
                hold_weight = max(
                    0.0,
                    min(1.0, float(motion_intent.get("hold_weight", 0.0) or 0.0)),
                )
                modifier_rotation_offset = motion_intent.get(
                    "modifier_rotation_offset_norm_rad"
                )
                metadata[side] = {
                    "source": str(
                        motion_intent.get(
                            "source",
                            (
                                "pose_action_interval_velocity_state_machine"
                                if isinstance(pose_action_intent_raw, dict)
                                else "tracker_motion_state_machine"
                            ),
                        )
                    ),
                    "previous_target_available": self._last_target_poses.get(side)
                    is not None,
                    "translation_m": None,
                    "rotation_deg": None,
                    "activity_ratio": None,
                    "hold_weight": hold_weight,
                    "inactive": bool(motion_intent.get("inactive", False)),
                    "state": motion_intent.get("state"),
                    "quiet_dwell_sec": motion_intent.get("quiet_dwell_sec"),
                    "exit_dwell_sec": motion_intent.get("exit_dwell_sec"),
                    "transition_reason": motion_intent.get("transition_reason"),
                    "evidence": motion_intent.get("evidence"),
                    "motion_window": motion_intent.get("motion_window"),
                    "modifier_active": bool(
                        motion_intent.get("modifier_active", False)
                    ),
                    "modifier_offset_norm_m": motion_intent.get(
                        "modifier_offset_norm_m"
                    ),
                    "modifier_rotation_offset_norm_rad": modifier_rotation_offset,
                    "modifier_qpos_fast_path_alpha": (
                        motion_intent.get("modifier_qpos_fast_path_alpha")
                        if modifier_rotation_offset is not None
                        else None
                    ),
                    "modifier_phase": (
                        motion_intent.get("modifier_phase")
                        if modifier_rotation_offset is not None
                        else None
                    ),
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

    @staticmethod
    def _authority_fixed_sides(
        target_activity: Mapping[str, object] | None,
    ) -> frozenset[str]:
        if not isinstance(target_activity, Mapping):
            return frozenset()
        return frozenset(
            side
            for side in ("left", "right")
            if isinstance(target_activity.get(side), Mapping)
            and bool(target_activity[side].get("fixed_by_authority", False))
        )

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
        return arm_joint_entries(
            solution_layout=self.profile.solution_layout,
            vector_size=q.size,
            owner_name=self.name,
        )

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
        left_alt_candidate = preferred_wrist_flip_candidate(
            hold_left,
            reference_q=hold_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(
                self.ik_tuning.branch_normalization.limit_tolerance_rad
            ),
        )
        left_alt = (
            None
            if self._wrist_branch_switch_guard_active(
                raw_q=hold_left,
                reference_q=hold_left,
                candidate_q=left_alt_candidate,
                side="left",
            )
            else left_alt_candidate
        )
        right_alt_candidate = preferred_wrist_flip_candidate(
            hold_right,
            reference_q=hold_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(
                self.ik_tuning.branch_normalization.limit_tolerance_rad
            ),
        )
        right_alt = (
            None
            if self._wrist_branch_switch_guard_active(
                raw_q=hold_right,
                reference_q=hold_right,
                candidate_q=right_alt_candidate,
                side="right",
            )
            else right_alt_candidate
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


# Compatibility names for callers that use the narrower target-solver
# terminology.  The object is a complete robot_motion Retargeter and the
# retargeter name is canonical for new composition roots, including sim.
DualArmCollisionAwareTargetSolverConfig = DualArmCollisionAwareRetargeterConfig
DualArmCollisionAwareTargetSolver = DualArmCollisionAwareRetargeter
