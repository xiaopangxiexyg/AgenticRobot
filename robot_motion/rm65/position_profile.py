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
)


RM65_BIMANUAL_POSITION_PROFILE_SCHEMA = "robot_motion.rm65_bimanual_position_profile/v1"
RM65_BIMANUAL_POSITION_V1_NAME = "rm65_bimanual_position_v1"
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
        if scale > 1.0:
            raise ValueError("inactive_enter_threshold_scale must be <= 1.")
        if slow > 1.0 or fast > 1.0:
            raise ValueError("post-IK smoothing alphas must be <= 1.")
        if fast < slow:
            raise ValueError("fast post-IK smoothing alpha must be >= slow alpha.")
        object.__setattr__(self, "inactive_enter_threshold_scale", scale)
        object.__setattr__(self, "post_ik_qpos_smoothing_alpha", slow)
        object.__setattr__(self, "post_ik_qpos_smoothing_fast_alpha", fast)
        object.__setattr__(
            self,
            "post_ik_qpos_smoothing_adaptive",
            bool(self.post_ik_qpos_smoothing_adaptive),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "inactive_enter_threshold_scale": self.inactive_enter_threshold_scale,
            "post_ik_qpos_smoothing_alpha": self.post_ik_qpos_smoothing_alpha,
            "post_ik_qpos_smoothing_fast_alpha": self.post_ik_qpos_smoothing_fast_alpha,
            "post_ik_qpos_smoothing_adaptive": self.post_ik_qpos_smoothing_adaptive,
            "application_scope": "live_human_source_before_main_action",
            "replay_or_deploy_reapply": False,
        }


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

    def __post_init__(self) -> None:
        planner = str(self.planner).strip()
        if planner != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED:
            raise ValueError("formal RM65 position execution requires acceleration_limited.")
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

    def to_metadata(self) -> dict[str, Any]:
        return {
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

RM65_BIMANUAL_POSITION_PROFILES = {
    RM65_BIMANUAL_POSITION_V1.name: RM65_BIMANUAL_POSITION_V1,
}
RM65_POSITION_CONTROL_PROFILE_CHOICES = (
    RM65_BIMANUAL_POSITION_V1_NAME,
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
    "RM65_POSITION_CONTROL_PROFILE_CHOICES",
    "RM65BimanualPositionProfile",
    "RM65PositionCommandExecutionProfile",
    "RM65PositionIKExecutionProfile",
    "RM65PositionSourceShapingProfile",
    "resolve_rm65_bimanual_position_profile",
]
