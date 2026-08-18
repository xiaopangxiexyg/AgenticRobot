"""Concrete lower-only RM65 assembly for an independent Skill process."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
)
from robot_motion.rm65 import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    load_ik_tuning_config,
    with_rm65_position_ik_objective,
)
from teleop_runtime.backends.arms import (
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    RM65_POSITION_SEND_DT_ALPHA,
    RM65_POSITION_SEND_DT_MAX_SCALE,
    RM65_POSITION_SEND_DT_STALL_RESET_SEC,
    RM65ArmBackend,
    RM65ArmBackendConfig,
)
from teleop_runtime.interfaces import EffectorBackend

from .rm65_ports import (
    ExclusiveRM65SkillDeployPorts,
    ExclusiveRM65SkillPortProfile,
    RM65SkillPositionForceGuard,
    RM65SkillTargetBuilder,
    build_exclusive_rm65_skill_position_force_guard,
)
from .rm65_profile import ResolvedExclusiveRM65SkillDeploymentProfile
from .rm65_target_safety import (
    ExclusiveRM65SkillStartupSafetyBinder,
    ExclusiveRM65SkillTargetSafetyRuntime,
)


class RM65SkillBackendFactory(Protocol):
    def __call__(self, config: RM65ArmBackendConfig) -> EffectorBackend: ...


class RM65SkillTargetBuilderFactory(Protocol):
    def __call__(
        self,
        config: DualArmCollisionAwareRetargeterConfig,
    ) -> RM65SkillTargetBuilder: ...


@dataclass(frozen=True)
class ExclusiveRM65SkillLowerAssembly:
    """One exact lower graph; constructing it performs no device I/O."""

    profile: ResolvedExclusiveRM65SkillDeploymentProfile
    arm_backends: dict[str, EffectorBackend]
    target_builder: RM65SkillTargetBuilder
    position_force_guard: RM65SkillPositionForceGuard
    ports: ExclusiveRM65SkillDeployPorts
    target_safety: ExclusiveRM65SkillTargetSafetyRuntime
    startup_admission: ExclusiveRM65SkillStartupSafetyBinder

    def __post_init__(self) -> None:
        if set(self.arm_backends) != {"left", "right"}:
            raise ValueError("lower assembly requires exact left/right backends.")
        if set(self.ports.arm_backends) != set(self.arm_backends) or any(
            self.ports.arm_backends[side] is not self.arm_backends[side]
            for side in self.arm_backends
        ):
            raise ValueError("ports must own the exact assembled backend inventory.")
        if self.ports.target_builder is not self.target_builder:
            raise ValueError("ports and assembly must share one target builder.")
        if self.target_safety.target_builder is not self.target_builder:
            raise ValueError("target safety must share the exact target builder.")
        if self.startup_admission.ports is not self.ports:
            raise ValueError("startup admission must bind the exact ports owner.")
        if self.startup_admission.target_safety is not self.target_safety:
            raise ValueError("startup admission must bind the exact safety owner.")


def _target_builder_config(
    profile: ResolvedExclusiveRM65SkillDeploymentProfile,
) -> DualArmCollisionAwareRetargeterConfig:
    position = profile.position_profile
    ik = position.ik_execution
    tuning = with_rm65_position_ik_objective(
        load_ik_tuning_config(),
        position_weight=ik.position_weight,
        orientation_weight=ik.orientation_weight,
        mode_name="exclusive_rm65_skill_position_profile",
    )
    return DualArmCollisionAwareRetargeterConfig(
        control_mode="absolute",
        orientation_mode="full",
        translation_scale=1.0,
        workspace_clip=True,
        scene_profile=profile.collision_scene_profile,
        contact_mode="boundary_slide",
        solver_frequency=position.rates.ik_hz,
        enforce_realistic_joint_step=True,
        left_tool_tcp_pose_flange_m_rad=(
            profile.tool_tcp_pose_flange_m_rad_by_side["left"]
        ),
        right_tool_tcp_pose_flange_m_rad=(
            profile.tool_tcp_pose_flange_m_rad_by_side["right"]
        ),
        ik_tuning=tuning,
        joint_velocity_limit_profile=ik.joint_velocity_limit_profile,
        damping=ik.damping,
        solve_min_iterations=ik.solve_min_iterations,
        solve_max_iterations=ik.solve_max_iterations,
        solve_adaptive_iterations_enabled=ik.adaptive_iterations,
        solve_iteration_reserve_ms=ik.iteration_reserve_ms,
        candidate_certification_reserve_ms=(
            ik.candidate_certification_reserve_ms
        ),
        local_step_candidate_clip_scale=ik.local_step_candidate_clip_scale,
        prefer_non_wrist_singular_branch=(
            ik.prefer_non_wrist_singular_branch
        ),
        prefer_non_elbow_singular_branch=(
            ik.prefer_non_elbow_singular_branch
        ),
        prefer_non_shoulder_singular_branch=(
            ik.prefer_non_shoulder_singular_branch
        ),
        position_singularity_candidate_certificate_enabled=(
            ik.require_singularity_candidate_certificate
        ),
        wrist_branch_switch_guard_deg=12.0,
        wrist_q5_hysteresis_release_deg=6.0,
        elbow_singularity_selection_margin_deg=0.25,
        inactive_side_hold_enabled=True,
        inactive_side_hold_execution_cap_enabled=True,
        inactive_side_hold_execution_release_enabled=False,
        output_command_type="joint_vector",
    )


def _backend_config(
    profile: ResolvedExclusiveRM65SkillDeploymentProfile,
    *,
    side: str,
) -> RM65ArmBackendConfig:
    position = profile.position_profile
    command = position.command_execution
    if command.planner != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED:
        raise ValueError(
            "independent real Skill v1 currently supports only the reviewed "
            "acceleration-limited RM65 executor; sample-aligned policy clocks "
            "require a separate Skill-source timing review."
        )
    return RM65ArmBackendConfig(
        arm_ip=profile.arm_ip_by_side[side],
        side=side,
        connect_hardware=True,
        control_dt=1.0 / position.rates.command_hz,
        max_angvel=tuple(command.max_velocity_rad_s),
        dry_run_fast=False,
        use_actual_send_dt_scaling=True,
        actual_send_dt_alpha=RM65_POSITION_SEND_DT_ALPHA,
        actual_send_dt_min_scale=1.0,
        actual_send_dt_max_scale=RM65_POSITION_SEND_DT_MAX_SCALE,
        actual_send_dt_stall_reset_sec=(
            RM65_POSITION_SEND_DT_STALL_RESET_SEC
        ),
        background_streaming_enabled=True,
        background_streaming_hz=position.rates.command_hz,
        background_streaming_target_max_age_sec=command.reference_max_age_sec,
        background_streaming_target_admission_mode=(
            RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
        ),
        position_command_planner=command.planner,
        position_velocity_ramp_sec=command.velocity_ramp_sec,
        position_target_velocity_filter_tau_sec=(
            command.target_velocity_filter_tau_sec
        ),
        position_max_acceleration_rad_s2=tuple(
            command.max_acceleration_rad_s2
        ),
        position_max_command_step_deg=command.max_command_step_deg,
        position_degraded_speed_scale=command.degraded_speed_scale,
        arm_follow_qpos_alpha=command.follow_qpos_alpha,
        canfd_follow=command.canfd_follow,
        canfd_trajectory_mode=command.canfd_trajectory_mode,
        canfd_radio=command.canfd_radio,
        tool_profile=str(profile.tool_profile_path_by_side[side]),
        singularity_protection_enabled=True,
        allow_position_wrist_singularity_fallback=(
            position.ik_execution.prefer_non_wrist_singular_branch
        ),
        allow_position_elbow_singularity_fallback=(
            position.ik_execution.prefer_non_elbow_singular_branch
        ),
        allow_position_shoulder_singularity_fallback=(
            position.ik_execution.prefer_non_shoulder_singular_branch
        ),
        require_position_singularity_candidate_certificate=(
            position.ik_execution.require_singularity_candidate_certificate
        ),
        position_sdk_singularity_advisory=(
            position.ik_execution.require_singularity_candidate_certificate
        ),
        ik_tuning=with_rm65_position_ik_objective(
            load_ik_tuning_config(),
            position_weight=position.ik_execution.position_weight,
            orientation_weight=position.ik_execution.orientation_weight,
            mode_name="exclusive_rm65_skill_position_profile",
        ),
    )


def _default_force_guard_factory(
    backends: dict[str, EffectorBackend],
) -> RM65SkillPositionForceGuard:
    return build_exclusive_rm65_skill_position_force_guard(
        arm_backends=backends
    )


def assemble_exclusive_rm65_skill_lower(
    *,
    profile: ResolvedExclusiveRM65SkillDeploymentProfile,
    backend_factory: RM65SkillBackendFactory = RM65ArmBackend,
    target_builder_factory: RM65SkillTargetBuilderFactory = (
        DualArmCollisionAwareRetargeter
    ),
    force_guard_factory: Callable[
        [dict[str, EffectorBackend]],
        RM65SkillPositionForceGuard,
    ] = _default_force_guard_factory,
) -> ExclusiveRM65SkillLowerAssembly:
    """Construct the lower graph without starting a backend or transport."""

    if not isinstance(profile, ResolvedExclusiveRM65SkillDeploymentProfile):
        raise TypeError("profile must be resolved and typed.")
    for name, factory in (
        ("backend_factory", backend_factory),
        ("target_builder_factory", target_builder_factory),
        ("force_guard_factory", force_guard_factory),
    ):
        if not callable(factory):
            raise TypeError(f"{name} must be callable.")
    target_builder = target_builder_factory(_target_builder_config(profile))
    backends = {
        side: backend_factory(_backend_config(profile, side=side))
        for side in ("left", "right")
    }
    for side, backend in backends.items():
        actual_tcp = getattr(
            getattr(backend, "config", None),
            "tcp_pose_flange_m_rad",
            None,
        )
        if actual_tcp is not None and tuple(actual_tcp) != tuple(
            profile.tool_tcp_pose_flange_m_rad_by_side[side]
        ):
            raise ValueError(f"{side} backend tool TCP differs from target solver.")
    force_guard = force_guard_factory(backends)
    ports = ExclusiveRM65SkillDeployPorts(
        arm_backends=backends,
        target_builder=target_builder,
        position_force_guard=force_guard,
        profile=ExclusiveRM65SkillPortProfile(
            state_refresh_hz=30.0,
            max_measurement_age_sec=(
                profile.position_profile.command_execution.reference_max_age_sec
            ),
        ),
    )
    target_safety = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=target_builder,
        station_world_from_arm_base_by_side=(
            profile.scene_world_from_arm_base_by_side
        ),
        station_world_frame_id=profile.station_world_frame_id,
        envelope_profile_path=profile.envelope_profile_path,
    )
    startup_admission = ExclusiveRM65SkillStartupSafetyBinder(
        ports=ports,
        target_safety=target_safety,
    )
    return ExclusiveRM65SkillLowerAssembly(
        profile=profile,
        arm_backends=backends,
        target_builder=target_builder,
        position_force_guard=force_guard,
        ports=ports,
        target_safety=target_safety,
        startup_admission=startup_admission,
    )


__all__ = [
    "ExclusiveRM65SkillLowerAssembly",
    "RM65SkillBackendFactory",
    "RM65SkillTargetBuilderFactory",
    "assemble_exclusive_rm65_skill_lower",
]
