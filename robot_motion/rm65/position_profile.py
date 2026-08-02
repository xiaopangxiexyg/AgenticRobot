"""Versioned RM65 bimanual position-control profiles.

The profile deliberately separates source shaping from execution of an already
materialized main action.  Collection stores each arm action as the FK pose of
the last accepted qref, so tracker filtering and post-IK adaptive qref
smoothing are already represented by that action.  Replay and deployment must
not apply those stages a second time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from robot_motion.position_control.config import PositionControlRates
from robot_motion.position_control.trajectory import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
)


RM65_BIMANUAL_POSITION_PROFILE_SCHEMA = "robot_motion.rm65_bimanual_position_profile/v1"
RM65_BIMANUAL_POSITION_V1_NAME = "rm65_bimanual_position_v1"
RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME = (
    "rm65_bimanual_position_v2_responsive"
)
RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME = (
    "rm65_bimanual_position_sample_aligned_jerk_v1"
)
RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME = (
    "rm65_bimanual_position_sample_aligned_jerk_v2"
)
RM65_BIMANUAL_POSITION_V1_QREF_PROFILE = "rm65_bimanual_position_v1_qref"
LEGACY_POSITION_CONTROL_PROFILE_NAME = "legacy"


def _finite(value: float, *, name: str, minimum: float | None = None) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return resolved


def _six_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != 6 or any(not math.isfinite(value) or value <= 0.0 for value in resolved):
        raise ValueError(f"{name} must contain six positive finite values.")
    return resolved


@dataclass(frozen=True)
class RM65PositionSourceShapingProfile:
    """Human-source stages that are baked into the collected main action."""

    inactive_enter_threshold_scale: float = 0.5
    post_ik_qpos_smoothing_alpha: float = 0.14
    post_ik_qpos_smoothing_fast_alpha: float = 0.45
    post_ik_qpos_smoothing_adaptive: bool = True
    # V1 deliberately leaves these off.  A later profile may bypass the
    # broad post-IK EMA only for a per-side, locally ordinary tracking step;
    # the final qref velocity/acceleration/collision admission is downstream
    # and remains authoritative.
    post_ik_qpos_normal_fast_path: bool = False
    post_ik_qpos_normal_fast_path_alpha: float = 1.0
    post_ik_qpos_normal_fast_path_guard_deg: float = 4.0
    post_ik_qpos_normal_fast_path_quality_grace_sec: float = 0.0
    # V1 preserves the historical fixed release band.  Responsive profiles
    # require evidence that an escape correction reached admitted qref before
    # the release-band hysteresis is allowed to hold an inward q5 candidate.
    wrist_q5_hysteresis_requires_escape_latch: bool = False
    # A stationary tracker freezes the Cartesian target, not the robot.  These
    # thresholds decide when the robot-side execution reference is close
    # enough to that frozen target to re-enter the inactive jitter hold.  The
    # broad historical defaults are preserved for V1; successor profiles may
    # use task-scale tolerances so an inactive hand does not leave centimetres
    # of permanent tracking error.
    inactive_execution_release_translation_static_m: float = 0.03
    inactive_execution_release_translation_full_m: float = 0.10
    inactive_execution_release_rotation_static_deg: float = 5.0
    inactive_execution_release_rotation_full_deg: float = 30.0

    def __post_init__(self) -> None:
        scale = _finite(
            self.inactive_enter_threshold_scale,
            name="inactive_enter_threshold_scale",
            minimum=0.0,
        )
        slow = _finite(
            self.post_ik_qpos_smoothing_alpha,
            name="post_ik_qpos_smoothing_alpha",
            minimum=0.0,
        )
        fast = _finite(
            self.post_ik_qpos_smoothing_fast_alpha,
            name="post_ik_qpos_smoothing_fast_alpha",
            minimum=0.0,
        )
        normal_fast = _finite(
            self.post_ik_qpos_normal_fast_path_alpha,
            name="post_ik_qpos_normal_fast_path_alpha",
            minimum=0.0,
        )
        normal_guard = _finite(
            self.post_ik_qpos_normal_fast_path_guard_deg,
            name="post_ik_qpos_normal_fast_path_guard_deg",
            minimum=0.0,
        )
        normal_quality_grace = _finite(
            self.post_ik_qpos_normal_fast_path_quality_grace_sec,
            name="post_ik_qpos_normal_fast_path_quality_grace_sec",
            minimum=0.0,
        )
        inactive_translation_static = _finite(
            self.inactive_execution_release_translation_static_m,
            name="inactive_execution_release_translation_static_m",
            minimum=0.0,
        )
        inactive_translation_full = _finite(
            self.inactive_execution_release_translation_full_m,
            name="inactive_execution_release_translation_full_m",
            minimum=0.0,
        )
        inactive_rotation_static = _finite(
            self.inactive_execution_release_rotation_static_deg,
            name="inactive_execution_release_rotation_static_deg",
            minimum=0.0,
        )
        inactive_rotation_full = _finite(
            self.inactive_execution_release_rotation_full_deg,
            name="inactive_execution_release_rotation_full_deg",
            minimum=0.0,
        )
        if scale > 1.0:
            raise ValueError("inactive_enter_threshold_scale must be <= 1.")
        if slow > 1.0 or fast > 1.0 or normal_fast > 1.0:
            raise ValueError("post-IK smoothing alphas must be <= 1.")
        if fast < slow:
            raise ValueError("fast post-IK smoothing alpha must be >= slow alpha.")
        if inactive_translation_full <= inactive_translation_static:
            raise ValueError(
                "inactive execution translation full threshold must exceed the "
                "static threshold."
            )
        if inactive_rotation_full <= inactive_rotation_static:
            raise ValueError(
                "inactive execution rotation full threshold must exceed the "
                "static threshold."
            )
        object.__setattr__(self, "inactive_enter_threshold_scale", scale)
        object.__setattr__(self, "post_ik_qpos_smoothing_alpha", slow)
        object.__setattr__(self, "post_ik_qpos_smoothing_fast_alpha", fast)
        object.__setattr__(
            self,
            "post_ik_qpos_normal_fast_path_alpha",
            normal_fast,
        )
        object.__setattr__(
            self,
            "post_ik_qpos_normal_fast_path_guard_deg",
            normal_guard,
        )
        object.__setattr__(
            self,
            "post_ik_qpos_normal_fast_path_quality_grace_sec",
            normal_quality_grace,
        )
        object.__setattr__(
            self,
            "post_ik_qpos_smoothing_adaptive",
            bool(self.post_ik_qpos_smoothing_adaptive),
        )
        object.__setattr__(
            self,
            "post_ik_qpos_normal_fast_path",
            bool(self.post_ik_qpos_normal_fast_path),
        )
        object.__setattr__(
            self,
            "wrist_q5_hysteresis_requires_escape_latch",
            bool(self.wrist_q5_hysteresis_requires_escape_latch),
        )
        object.__setattr__(
            self,
            "inactive_execution_release_translation_static_m",
            inactive_translation_static,
        )
        object.__setattr__(
            self,
            "inactive_execution_release_translation_full_m",
            inactive_translation_full,
        )
        object.__setattr__(
            self,
            "inactive_execution_release_rotation_static_deg",
            inactive_rotation_static,
        )
        object.__setattr__(
            self,
            "inactive_execution_release_rotation_full_deg",
            inactive_rotation_full,
        )

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "inactive_enter_threshold_scale": self.inactive_enter_threshold_scale,
            "post_ik_qpos_smoothing_alpha": self.post_ik_qpos_smoothing_alpha,
            "post_ik_qpos_smoothing_fast_alpha": self.post_ik_qpos_smoothing_fast_alpha,
            "post_ik_qpos_smoothing_adaptive": self.post_ik_qpos_smoothing_adaptive,
            "application_scope": "live_human_source_before_main_action",
            "replay_or_deploy_reapply": False,
        }
        # Preserve the established V1 metadata/hash exactly.  New source
        # semantics become part of the fingerprint only for a profile that
        # explicitly enables them.
        if self.post_ik_qpos_normal_fast_path:
            payload.update(
                {
                    "post_ik_qpos_normal_fast_path": True,
                    "post_ik_qpos_normal_fast_path_alpha": (
                        self.post_ik_qpos_normal_fast_path_alpha
                    ),
                    "post_ik_qpos_normal_fast_path_guard_deg": (
                        self.post_ik_qpos_normal_fast_path_guard_deg
                    ),
                    "post_ik_qpos_normal_fast_path_quality_grace_sec": (
                        self.post_ik_qpos_normal_fast_path_quality_grace_sec
                    ),
                }
            )
        if self.wrist_q5_hysteresis_requires_escape_latch:
            payload["wrist_q5_hysteresis_requires_escape_latch"] = True
        inactive_release = (
            self.inactive_execution_release_translation_static_m,
            self.inactive_execution_release_translation_full_m,
            self.inactive_execution_release_rotation_static_deg,
            self.inactive_execution_release_rotation_full_deg,
        )
        if inactive_release != (0.03, 0.10, 5.0, 30.0):
            payload["inactive_execution_release"] = {
                "translation_static_m": inactive_release[0],
                "translation_full_m": inactive_release[1],
                "rotation_static_deg": inactive_release[2],
                "rotation_full_deg": inactive_release[3],
                "semantics": "freeze_target_then_converge_to_task_tolerance",
            }
        return payload


@dataclass(frozen=True)
class RM65PositionIKExecutionProfile:
    """Pose-action to accepted-qref IK and safety settings."""

    position_weight: float = 1.0
    orientation_weight: float = 0.35
    damping: float = 0.03
    solve_min_iterations: int = 2
    solve_max_iterations: int = 4
    adaptive_iterations: bool = True
    iteration_reserve_ms: float = 5.0
    candidate_certification_reserve_ms: float = 3.5
    local_step_candidate_clip_scale: float = 8.0
    joint_velocity_limit_profile: str = RM65_BIMANUAL_POSITION_V1_QREF_PROFILE
    prefer_non_wrist_singular_branch: bool = True
    prefer_non_elbow_singular_branch: bool = True
    prefer_non_shoulder_singular_branch: bool = True
    require_singularity_candidate_certificate: bool = True

    def __post_init__(self) -> None:
        position = _finite(self.position_weight, name="position_weight", minimum=0.0)
        orientation = _finite(
            self.orientation_weight,
            name="orientation_weight",
            minimum=0.0,
        )
        damping = _finite(self.damping, name="damping", minimum=0.0)
        reserve = _finite(
            self.iteration_reserve_ms,
            name="iteration_reserve_ms",
            minimum=0.0,
        )
        candidate_reserve = _finite(
            self.candidate_certification_reserve_ms,
            name="candidate_certification_reserve_ms",
            minimum=0.0,
        )
        if candidate_reserve > reserve:
            raise ValueError(
                "candidate_certification_reserve_ms must be <= iteration_reserve_ms."
            )
        local_clip = _finite(
            self.local_step_candidate_clip_scale,
            name="local_step_candidate_clip_scale",
            minimum=1.0,
        )
        minimum = int(self.solve_min_iterations)
        maximum = int(self.solve_max_iterations)
        if minimum <= 0 or maximum < minimum:
            raise ValueError("solve iterations must satisfy 0 < min <= max.")
        profile = str(self.joint_velocity_limit_profile).strip()
        if not profile:
            raise ValueError("joint_velocity_limit_profile must be non-empty.")
        object.__setattr__(self, "position_weight", position)
        object.__setattr__(self, "orientation_weight", orientation)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "solve_min_iterations", minimum)
        object.__setattr__(self, "solve_max_iterations", maximum)
        object.__setattr__(self, "adaptive_iterations", bool(self.adaptive_iterations))
        object.__setattr__(self, "iteration_reserve_ms", reserve)
        object.__setattr__(
            self,
            "candidate_certification_reserve_ms",
            candidate_reserve,
        )
        object.__setattr__(self, "local_step_candidate_clip_scale", local_clip)
        object.__setattr__(self, "joint_velocity_limit_profile", profile)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "position_weight": self.position_weight,
            "orientation_weight": self.orientation_weight,
            "damping": self.damping,
            "solve_min_iterations": self.solve_min_iterations,
            "solve_max_iterations": self.solve_max_iterations,
            "adaptive_iterations": self.adaptive_iterations,
            "iteration_reserve_ms": self.iteration_reserve_ms,
            "candidate_certification_reserve_ms": (
                self.candidate_certification_reserve_ms
            ),
            "local_step_candidate_clip_scale": self.local_step_candidate_clip_scale,
            "joint_velocity_limit_profile": self.joint_velocity_limit_profile,
            "prefer_non_wrist_singular_branch": self.prefer_non_wrist_singular_branch,
            "prefer_non_elbow_singular_branch": self.prefer_non_elbow_singular_branch,
            "prefer_non_shoulder_singular_branch": (
                self.prefer_non_shoulder_singular_branch
            ),
            "require_singularity_candidate_certificate": (
                self.require_singularity_candidate_certificate
            ),
        }


@dataclass(frozen=True)
class RM65PositionCommandExecutionProfile:
    """Accepted-qref to high-rate qcmd execution settings."""

    planner: str = POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    reference_max_age_sec: float = 0.2
    canfd_follow: bool = True
    canfd_trajectory_mode: int = 0
    canfd_radio: int = 0
    max_velocity_rad_s: tuple[float, ...] | Sequence[float] = (
        5.6,
        5.6,
        7.2,
        12.0,
        12.0,
        16.0,
    )
    max_acceleration_rad_s2: tuple[float, ...] | Sequence[float] = (
        56.0,
        56.0,
        72.0,
        120.0,
        120.0,
        160.0,
    )
    target_velocity_filter_tau_sec: float = 0.03
    velocity_ramp_sec: float = 0.10
    max_command_step_deg: float = 0.675
    follow_qpos_alpha: float = 0.9
    degraded_speed_scale: float = 0.5
    # V1 scales the absolute target itself in degraded mode.  A successor may
    # retain the full already-admitted qref and let the hard q/qd/qdd/jerk
    # envelope determine how much is safely reachable in the current cut.
    sample_aligned_scale_degraded_target: bool = True
    # Dimensionless regularizers for the normalized-jerk horizon QP.  They are
    # zero/near-zero in V1 for exact rollback.  V2 uses small positive values
    # to discourage bang-bang jerk without weakening any hard bound.
    sample_aligned_jerk_regularization: float = 1e-9
    sample_aligned_jerk_delta_regularization: float = 0.0
    sample_aligned_terminal_velocity_regularization: float = 0.0
    sample_aligned_terminal_acceleration_regularization: float = 0.0

    def __post_init__(self) -> None:
        planner = str(self.planner).strip()
        if planner not in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }:
            raise ValueError(
                "formal RM65 position execution requires a registered "
                "acceleration/jerk-limited planner."
            )
        reference_max_age = _finite(
            self.reference_max_age_sec,
            name="reference_max_age_sec",
            minimum=0.0,
        )
        if reference_max_age <= 0.0:
            raise ValueError("reference_max_age_sec must be positive.")
        trajectory_mode = int(self.canfd_trajectory_mode)
        radio = int(self.canfd_radio)
        if trajectory_mode < 0 or radio < 0:
            raise ValueError("CANFD trajectory mode and radio must be non-negative.")
        velocity = _six_positive(self.max_velocity_rad_s, name="max_velocity_rad_s")
        acceleration = _six_positive(
            self.max_acceleration_rad_s2,
            name="max_acceleration_rad_s2",
        )
        tau = _finite(
            self.target_velocity_filter_tau_sec,
            name="target_velocity_filter_tau_sec",
            minimum=0.0,
        )
        ramp = _finite(self.velocity_ramp_sec, name="velocity_ramp_sec", minimum=0.0)
        step = _finite(self.max_command_step_deg, name="max_command_step_deg", minimum=0.0)
        follow = _finite(self.follow_qpos_alpha, name="follow_qpos_alpha", minimum=0.0)
        degraded = _finite(
            self.degraded_speed_scale,
            name="degraded_speed_scale",
            minimum=0.0,
        )
        jerk_regularization = _finite(
            self.sample_aligned_jerk_regularization,
            name="sample_aligned_jerk_regularization",
            minimum=0.0,
        )
        jerk_delta_regularization = _finite(
            self.sample_aligned_jerk_delta_regularization,
            name="sample_aligned_jerk_delta_regularization",
            minimum=0.0,
        )
        terminal_velocity_regularization = _finite(
            self.sample_aligned_terminal_velocity_regularization,
            name="sample_aligned_terminal_velocity_regularization",
            minimum=0.0,
        )
        terminal_acceleration_regularization = _finite(
            self.sample_aligned_terminal_acceleration_regularization,
            name="sample_aligned_terminal_acceleration_regularization",
            minimum=0.0,
        )
        if ramp <= 0.0 or step <= 0.0:
            raise ValueError("velocity_ramp_sec and max_command_step_deg must be positive.")
        if follow > 1.0 or degraded > 1.0:
            raise ValueError("follow alpha and degraded scale must be <= 1.")
        object.__setattr__(self, "planner", planner)
        object.__setattr__(self, "reference_max_age_sec", reference_max_age)
        object.__setattr__(self, "canfd_follow", bool(self.canfd_follow))
        object.__setattr__(self, "canfd_trajectory_mode", trajectory_mode)
        object.__setattr__(self, "canfd_radio", radio)
        object.__setattr__(self, "max_velocity_rad_s", velocity)
        object.__setattr__(self, "max_acceleration_rad_s2", acceleration)
        object.__setattr__(self, "target_velocity_filter_tau_sec", tau)
        object.__setattr__(self, "velocity_ramp_sec", ramp)
        object.__setattr__(self, "max_command_step_deg", step)
        object.__setattr__(self, "follow_qpos_alpha", follow)
        object.__setattr__(self, "degraded_speed_scale", degraded)
        object.__setattr__(
            self,
            "sample_aligned_scale_degraded_target",
            bool(self.sample_aligned_scale_degraded_target),
        )
        object.__setattr__(
            self,
            "sample_aligned_jerk_regularization",
            jerk_regularization,
        )
        object.__setattr__(
            self,
            "sample_aligned_jerk_delta_regularization",
            jerk_delta_regularization,
        )
        object.__setattr__(
            self,
            "sample_aligned_terminal_velocity_regularization",
            terminal_velocity_regularization,
        )
        object.__setattr__(
            self,
            "sample_aligned_terminal_acceleration_regularization",
            terminal_acceleration_regularization,
        )

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "planner": self.planner,
            "reference_max_age_sec": self.reference_max_age_sec,
            "canfd_follow": self.canfd_follow,
            "canfd_trajectory_mode": self.canfd_trajectory_mode,
            "canfd_radio": self.canfd_radio,
            "max_velocity_rad_s": list(self.max_velocity_rad_s),
            "max_acceleration_rad_s2": list(self.max_acceleration_rad_s2),
            "target_velocity_filter_tau_sec": self.target_velocity_filter_tau_sec,
            "velocity_ramp_sec": self.velocity_ramp_sec,
            "max_command_step_deg": self.max_command_step_deg,
            "follow_qpos_alpha": self.follow_qpos_alpha,
            "degraded_speed_scale": self.degraded_speed_scale,
        }
        sample_aligned_v2 = (
            not self.sample_aligned_scale_degraded_target
            or self.sample_aligned_jerk_regularization != 1e-9
            or self.sample_aligned_jerk_delta_regularization != 0.0
            or self.sample_aligned_terminal_velocity_regularization != 0.0
            or self.sample_aligned_terminal_acceleration_regularization != 0.0
        )
        if sample_aligned_v2:
            payload["sample_aligned_governor"] = {
                "scale_degraded_target": self.sample_aligned_scale_degraded_target,
                "jerk_regularization": self.sample_aligned_jerk_regularization,
                "jerk_delta_regularization": (
                    self.sample_aligned_jerk_delta_regularization
                ),
                "terminal_velocity_regularization": (
                    self.sample_aligned_terminal_velocity_regularization
                ),
                "terminal_acceleration_regularization": (
                    self.sample_aligned_terminal_acceleration_regularization
                ),
            }
        return payload


@dataclass(frozen=True)
class RM65BimanualPositionProfile:
    name: str
    rates: PositionControlRates
    source_shaping: RM65PositionSourceShapingProfile = field(
        default_factory=RM65PositionSourceShapingProfile
    )
    ik_execution: RM65PositionIKExecutionProfile = field(
        default_factory=RM65PositionIKExecutionProfile
    )
    command_execution: RM65PositionCommandExecutionProfile = field(
        default_factory=RM65PositionCommandExecutionProfile
    )
    schema: str = RM65_BIMANUAL_POSITION_PROFILE_SCHEMA
    main_action_contract: str = "accepted_qref_fk_pose"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("position profile name must be non-empty.")
        if not isinstance(self.rates, PositionControlRates):
            raise TypeError("rates must be PositionControlRates.")
        if str(self.schema) != RM65_BIMANUAL_POSITION_PROFILE_SCHEMA:
            raise ValueError("unsupported RM65 position profile schema.")
        if str(self.main_action_contract) != "accepted_qref_fk_pose":
            raise ValueError("formal RM65 main action must be accepted_qref_fk_pose.")
        object.__setattr__(self, "name", name)

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "name": self.name,
            "main_action_contract": self.main_action_contract,
            "rates": self.rates.to_metadata(),
            "source_shaping": self.source_shaping.to_metadata(),
            "action_execution": {
                "ik": self.ik_execution.to_metadata(),
                "command": self.command_execution.to_metadata(),
                "reapply_source_shaping": False,
            },
        }
        payload["sha256"] = self.sha256(payload)
        return payload

    @staticmethod
    def sha256(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


RM65_BIMANUAL_POSITION_V1 = RM65BimanualPositionProfile(
    name=RM65_BIMANUAL_POSITION_V1_NAME,
    rates=PositionControlRates(
        sample_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
    ),
)

RM65_BIMANUAL_POSITION_V2_RESPONSIVE = RM65BimanualPositionProfile(
    name=RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME,
    rates=PositionControlRates(
        sample_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
    ),
    source_shaping=RM65PositionSourceShapingProfile(
        post_ik_qpos_normal_fast_path=True,
        # The old 0.14 adaptive endpoint was visibly sticky, while alpha=1
        # exposed tracker/IK micro-reversals directly to the 120 Hz transport.
        # 0.70 settles 90% of a step in roughly two 60 Hz reference cycles and
        # still attenuates frame-alternating joint noise.
        post_ik_qpos_normal_fast_path_alpha=0.70,
        post_ik_qpos_normal_fast_path_guard_deg=4.0,
        # Cover transient motion-window invalidation only.  Hard admission and
        # singularity guards in the solver remain immediate and fail closed.
        post_ik_qpos_normal_fast_path_quality_grace_sec=0.05,
        wrist_q5_hysteresis_requires_escape_latch=True,
    ),
)

# Opt-in execution experiment.  Source shaping and IK stay byte-for-byte equal
# to V1; only accepted-qref execution changes.  A 15 Hz source cut is projected
# onto the 120 Hz position/velocity/acceleration/jerk reachable set, and its
# admitted endpoint becomes the recorded main action.  Selecting V1 is the
# complete rollback.
RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1 = (
    RM65BimanualPositionProfile(
        name=RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME,
        rates=PositionControlRates(
            sample_hz=15.0,
            reference_hz=60.0,
            ik_hz=60.0,
            command_hz=120.0,
        ),
        command_execution=RM65PositionCommandExecutionProfile(
            planner=POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
            target_velocity_filter_tau_sec=0.0,
            # The jerk bound is max_acceleration / ramp.  The 50 ms initial
            # value is the only tracking-oriented point already rerun on all
            # 115 offline episodes: zero q/v/a/jerk/collision failures, while
            # the worst stored-action pose lag stayed below 2.756 mm / 0.831°.
            # 75/100 ms remain explicit research overrides for more smoothing.
            velocity_ramp_sec=0.05,
            follow_qpos_alpha=1.0,
        ),
    )
)

# Reversible successor to the first sample-aligned experiment.  It restores
# the responsive normal source path, lets an inactive tracker's frozen target
# converge to task-scale tolerance, preserves the full admitted target in
# degraded mode, and adds only soft QP smoothness costs.  All hard position,
# velocity, acceleration, jerk, collision and singularity contracts remain
# unchanged.  Selecting the V1 profile atomically restores the old behavior.
RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2 = (
    RM65BimanualPositionProfile(
        name=RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME,
        rates=RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1.rates,
        source_shaping=RM65PositionSourceShapingProfile(
            post_ik_qpos_normal_fast_path=True,
            post_ik_qpos_normal_fast_path_alpha=0.70,
            post_ik_qpos_normal_fast_path_guard_deg=4.0,
            post_ik_qpos_normal_fast_path_quality_grace_sec=0.05,
            wrist_q5_hysteresis_requires_escape_latch=True,
            inactive_execution_release_translation_static_m=0.0015,
            inactive_execution_release_translation_full_m=0.012,
            inactive_execution_release_rotation_static_deg=0.25,
            inactive_execution_release_rotation_full_deg=2.0,
        ),
        command_execution=RM65PositionCommandExecutionProfile(
            planner=POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
            target_velocity_filter_tau_sec=0.0,
            velocity_ramp_sec=0.05,
            follow_qpos_alpha=1.0,
            sample_aligned_scale_degraded_target=False,
            sample_aligned_jerk_regularization=1e-6,
            sample_aligned_jerk_delta_regularization=1e-5,
            sample_aligned_terminal_velocity_regularization=0.0,
            sample_aligned_terminal_acceleration_regularization=0.0,
        ),
    )
)

RM65_BIMANUAL_POSITION_PROFILES = {
    RM65_BIMANUAL_POSITION_V1.name: RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V2_RESPONSIVE.name: (
        RM65_BIMANUAL_POSITION_V2_RESPONSIVE
    ),
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1.name: (
        RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1
    ),
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2.name: (
        RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2
    ),
}
RM65_POSITION_CONTROL_PROFILE_CHOICES = (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
)


def resolve_rm65_bimanual_position_profile(name: str) -> RM65BimanualPositionProfile:
    normalized = str(name).strip()
    try:
        return RM65_BIMANUAL_POSITION_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown RM65 bimanual position profile {name!r}; "
            f"expected one of {tuple(RM65_BIMANUAL_POSITION_PROFILES)}."
        ) from exc


__all__ = [
    "LEGACY_POSITION_CONTROL_PROFILE_NAME",
    "RM65_BIMANUAL_POSITION_PROFILE_SCHEMA",
    "RM65_BIMANUAL_POSITION_PROFILES",
    "RM65_BIMANUAL_POSITION_V1",
    "RM65_BIMANUAL_POSITION_V1_NAME",
    "RM65_BIMANUAL_POSITION_V1_QREF_PROFILE",
    "RM65_BIMANUAL_POSITION_V2_RESPONSIVE",
    "RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME",
    "RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1",
    "RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME",
    "RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2",
    "RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME",
    "RM65_POSITION_CONTROL_PROFILE_CHOICES",
    "RM65BimanualPositionProfile",
    "RM65PositionCommandExecutionProfile",
    "RM65PositionIKExecutionProfile",
    "RM65PositionSourceShapingProfile",
    "resolve_rm65_bimanual_position_profile",
]
