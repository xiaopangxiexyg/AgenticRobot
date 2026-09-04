"""Deployment-owned RM65 backend construction and command timing.

This module translates an already parsed deployment profile into concrete
arm backends and per-command timing metadata.  It does not start or close a
backend, own dispatch, or make authority and safety decisions; those remain
with the deployment application and the lower execution owners.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_ARM_CONTROL_MODE_IMPEDANCE,
    RM65_ARM_CONTROL_MODE_POSITION,
    build_rm65_pose_lift_ik_tuning,
    normalize_rm65_arm_control_mode,
)
from robot_motion.rm65.assets import require_asset_path

from ..backends.arms import (
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    RM65_POSITION_SEND_DT_ALPHA,
    RM65_POSITION_SEND_DT_MAX_SCALE,
    RM65_POSITION_SEND_DT_STALL_RESET_SEC,
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from ..defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..interfaces import EffectorBackend
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding


DEPLOY_RM65_ARM_MAX_ANGVEL = (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
DEPLOY_ARM_STEP_DT_ALPHA = RM65_POSITION_SEND_DT_ALPHA
DEPLOY_ARM_STEP_DT_MAX_SCALE = RM65_POSITION_SEND_DT_MAX_SCALE
DEPLOY_ARM_STEP_DT_STALL_RESET_SEC = RM65_POSITION_SEND_DT_STALL_RESET_SEC
DEPLOY_MAX_ARM_JOINT_STEP_RAD = float(np.deg2rad(6.0))


def _position_deploy_ik_tuning(args: argparse.Namespace):
    """Bind the formal action-to-qref objective without source-side shaping."""

    return build_rm65_pose_lift_ik_tuning(
        position_control_profile=(
            str(getattr(args, "position_control_profile_resolved", ""))
            or RM65_BIMANUAL_POSITION_V1_NAME
        ),
        pose_lift_profile=str(args.pose_lift_profile),
        orientation_weight=float(args.position_ik_orientation_weight),
    )


def effective_arm_control_hz(args: argparse.Namespace) -> float:
    value = getattr(args, "arm_control_hz", None)
    if value is not None:
        return float(value)
    if (
        normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_IMPEDANCE
    ):
        return float(DEFAULT_RM65_ARM_CONTROL_HZ)
    return float(DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)


def effective_position_reference_hz(args: argparse.Namespace) -> float:
    return float(
        getattr(
            args,
            "position_reference_hz",
            DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        )
    )


def resolve_position_reference_and_command_hz(
    args: argparse.Namespace,
    *,
    backend_control_hz: float,
) -> tuple[float, float]:
    """Keep planner-reference and downstream CANFD clocks distinct."""

    reference_hz = effective_position_reference_hz(args)
    command_hz = (
        float(backend_control_hz)
        if effective_arm_canfd_follow(args)
        else float(reference_hz)
    )
    return float(reference_hz), float(command_hz)


def effective_arm_canfd_follow(args: argparse.Namespace) -> bool:
    value = getattr(args, "arm_canfd_follow", None)
    if value is not None:
        return bool(value)
    return (
        normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
        == RM65_ARM_CONTROL_MODE_POSITION
    )


def resolve_arm_step_dt_max_scale(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> tuple[float, bool]:
    explicit = getattr(args, "arm_step_dt_max_scale", None)
    if explicit is not None:
        return float(explicit), False
    target_hz = float(action_hz or 0.0)
    if target_hz <= 0.0:
        target_hz = float(
            getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ)
            or DEFAULT_TELEOP_SAMPLE_HZ
        )
    target_hz = max(1e-6, target_hz)
    required_scale = 1.1 * max(1e-6, float(control_hz)) / target_hz
    return max(DEPLOY_ARM_STEP_DT_MAX_SCALE, required_scale), True


def apply_arm_step_dt_defaults(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> None:
    max_scale, auto = resolve_arm_step_dt_max_scale(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    args.arm_step_dt_max_scale = float(max_scale)
    args.arm_step_dt_max_scale_auto = bool(auto)


def deploy_arm_command_step_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> float:
    base_dt = 1.0 / max(1e-6, float(control_hz))
    if not bool(getattr(args, "arm_step_dt_scaling", True)):
        return base_dt
    action_dt = 1.0 / max(1e-6, float(action_hz))
    max_scale = float(
        getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
    )
    return float(np.clip(action_dt, base_dt, base_dt * max_scale))


def deploy_arm_backend_control_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None,
) -> float:
    arm_control_mode = normalize_rm65_arm_control_mode(
        getattr(args, "arm_control_mode", None)
    )
    if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
        return 1.0 / max(1e-6, float(control_hz))
    del action_hz
    return 1.0 / max(1e-6, effective_position_reference_hz(args))


def deploy_arm_command_max_joint_step(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> np.ndarray:
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    return np.asarray(args.arm_max_angvel, dtype=float).reshape(6) * step_dt


def annotate_deploy_arm_command_limits(
    command: Any,
    *,
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    control_hz: float,
    action_hz: float,
) -> None:
    if command is None:
        return
    commands = getattr(command, "commands", {}) or {}
    if not commands:
        return
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    max_joint_step = deploy_arm_command_max_joint_step(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if effector_command is None:
            continue
        effector_command.metadata["max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_action_step_dt_sec"] = float(step_dt)
        effector_command.metadata["deploy_backend_control_dt_sec"] = float(
            backend_control_dt
        )
        effector_command.metadata["deploy_arm_step_dt_scaling"] = bool(
            getattr(args, "arm_step_dt_scaling", True)
        )
        effector_command.metadata["deploy_arm_step_dt_max_scale"] = float(
            getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
        )


def build_arm_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    control_hz: float | None,
    action_hz: float | None = None,
) -> dict[str, EffectorBackend]:
    """Build deployment RM65 backends without starting device resources."""

    if not bool(args.arms_enabled):
        return {}
    arm_urdf_path = str(args.arm_urdf_path or require_asset_path("rm65_arm.urdf"))
    control_hz = (
        effective_arm_control_hz(args) if control_hz is None else float(control_hz)
    )
    control_dt = 1.0 / float(control_hz)
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    arm_control_mode = normalize_rm65_arm_control_mode(
        getattr(args, "arm_control_mode", None)
    )
    worker_binding = getattr(args, "_rm65_legacy_worker_binding", None)
    if worker_binding is not None:
        if not isinstance(worker_binding, RM65LegacyWorkerExecutionBinding):
            raise TypeError("_rm65_legacy_worker_binding has an unsupported type")
        if tuple(active_sides) != ("left", "right"):
            raise ValueError(
                "RM65 arms-only worker binding requires coupled left/right sides"
            )
        if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
            raise ValueError("RM65 arms-only worker binding excludes impedance control")
    command_input_hz = (
        effective_position_reference_hz(args)
        if arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        else action_hz
    )
    apply_arm_step_dt_defaults(
        args,
        control_hz=float(control_hz),
        action_hz=command_input_hz,
    )
    arm_canfd_follow = effective_arm_canfd_follow(args)
    # A model RTC real-shadow can be either a passive live observation run or
    # an explicitly authorized active run.  Keep the backend observation-only
    # only when the caller did not request dispatch; ``--send-actions`` is the
    # single explicit transition to normal command-capable backend configs.
    observation_only = bool(getattr(args, "rtc_real_shadow", False)) and not bool(
        getattr(args, "send_actions", False)
    )
    if observation_only:
        arm_canfd_follow = False
    backends: dict[str, EffectorBackend] = {}
    for side in active_sides:
        arm_ip = str(getattr(args, f"{side}_arm_ip") or "")
        if arm_control_mode == RM65_ARM_CONTROL_MODE_IMPEDANCE:
            backends[side] = RM65ImpedanceArmBackend(
                RM65ImpedanceArmBackendConfig(
                    arm_ip=arm_ip or "0.0.0.0",
                    side=side,
                    connect_hardware=bool(arm_ip),
                    observation_only=observation_only,
                    tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                    strict_tool_profile=bool(
                        getattr(args, "strict_tool_profile", True)
                    ),
                    impedance_mode="pose",
                    force_tare_sec=float(getattr(args, "force_tare_sec", 3.0)),
                    control_dt=control_dt,
                    action_hz=(None if action_hz is None else float(action_hz)),
                    max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                    actual_send_dt_stall_reset_sec=float(
                        args.arm_step_dt_stall_reset_sec
                    ),
                    debug_timing=bool(getattr(args, "debug", False)),
                    impedance_overrides={
                        "canfd_follow": bool(arm_canfd_follow),
                        "joint_rate_limit_use_nominal_dt": True,
                        "admittance_use_nominal_dt": True,
                        "joint_operating_margin_deg": 3.0,
                        "ik_hard_deadline_enabled": True,
                        "deployment_dual_rate_ik_enabled": True,
                        "deployment_full_ik_hz": 15.0,
                        "deployment_full_ik_deadline_ms": 66.66666666666667,
                        "deployment_local_ik_deadline_ms": 6.0,
                        "deployment_full_ik_preempts_local": bool(
                            getattr(args, "deployment_full_ik_preempts_local", False)
                        ),
                        "ik_resolver_candidate_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_max_position_worsen_m": 0.003,
                        "ik_resolver_local_step_max_orientation_worsen_deg": 0.5,
                        "ik_resolver_local_step_dynamic_worsen_enabled": False,
                        "direct_joint_hint_max_linear_offset_m": 0.0,
                        "direct_joint_hint_max_angular_offset_rad": 0.0,
                    },
                )
            )
            continue
        binding_kwargs = (
            {}
            if worker_binding is None
            else worker_binding.arm_backend_kwargs(
                side,
                connect_hardware=bool(arm_ip),
            )
        )
        backends[side] = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip or "0.0.0.0",
                side=side,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=bool(arm_ip),
                observation_only=observation_only,
                control_dt=backend_control_dt,
                max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                singularity_qpos_init_slowdown=float(
                    args.singularity_qpos_init_slowdown
                ),
                dry_run_fast=not bool(arm_ip),
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(
                    args.rm65_hardware_read_retry_delay_sec
                ),
                initial_qpos=getattr(args, f"{side}_initial_qpos"),
                tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
                use_actual_send_dt_scaling=bool(
                    getattr(args, "arm_step_dt_scaling", True)
                ),
                actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                actual_send_dt_min_scale=1.0,
                actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                background_streaming_enabled=bool(arm_ip) and bool(arm_canfd_follow),
                background_streaming_hz=float(control_hz),
                background_streaming_target_admission_mode=str(
                    getattr(
                        args,
                        "position_background_target_admission",
                        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
                    )
                ),
                position_command_planner=str(args.position_command_planner),
                position_velocity_ramp_sec=float(args.position_velocity_ramp_sec),
                position_target_velocity_filter_tau_sec=float(
                    args.position_target_velocity_filter_tau_sec
                ),
                position_max_acceleration_rad_s2=tuple(
                    float(value) for value in args.position_max_acceleration_rad_s2
                ),
                position_max_command_step_deg=float(args.position_max_command_step_deg),
                position_degraded_speed_scale=float(args.position_degraded_speed_scale),
                arm_follow_qpos_alpha=float(args.arm_follow_qpos_alpha),
                canfd_follow=bool(arm_canfd_follow),
                canfd_trajectory_mode=int(
                    getattr(args, "arm_canfd_trajectory_mode", 0)
                ),
                canfd_radio=int(getattr(args, "arm_canfd_radio", 0)),
            ),
            **binding_kwargs,
        )
    return backends


__all__ = [
    "DEPLOY_ARM_STEP_DT_ALPHA",
    "DEPLOY_ARM_STEP_DT_MAX_SCALE",
    "DEPLOY_ARM_STEP_DT_STALL_RESET_SEC",
    "DEPLOY_MAX_ARM_JOINT_STEP_RAD",
    "DEPLOY_RM65_ARM_MAX_ANGVEL",
    "_position_deploy_ik_tuning",
    "annotate_deploy_arm_command_limits",
    "apply_arm_step_dt_defaults",
    "build_arm_backends",
    "deploy_arm_backend_control_dt_sec",
    "deploy_arm_command_max_joint_step",
    "deploy_arm_command_step_dt_sec",
    "effective_arm_canfd_follow",
    "effective_arm_control_hz",
    "effective_position_reference_hz",
    "resolve_arm_step_dt_max_scale",
    "resolve_position_reference_and_command_hz",
]
