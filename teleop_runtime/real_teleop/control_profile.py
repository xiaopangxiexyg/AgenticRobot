"""Pure resolution of the real-teleop control and CANFD runtime profile.

These functions deliberately contain no scheduler, backend, or motion calls.
They translate parsed real-teleop arguments (or a resolved execution profile)
into the rates and switches consumed by the entrypoint's composition layer.
"""

from __future__ import annotations

import argparse

from ..defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..position_control import RealArmExecutionConfig


def arm_control_mode(args: argparse.Namespace) -> str:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.mode
    return str(getattr(args, "arm_control_mode", "impedance"))


def effective_position_reference_hz(args: argparse.Namespace) -> float:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.position_reference_hz
    return float(
        getattr(
            args,
            "position_reference_hz",
            DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        )
    )


def effective_outer_control_hz(args: argparse.Namespace) -> float:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.outer_control_hz
    if arm_control_mode(args) == "position":
        return effective_position_reference_hz(args)
    return float(getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ))


def effective_dual_arm_ik_hz(args: argparse.Namespace) -> float:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.dual_arm_ik_hz
    value = getattr(args, "dual_arm_ik_hz", None)
    return effective_outer_control_hz(args) if value is None else float(value)


def effective_arm_control_hz(args: argparse.Namespace) -> float:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.arm_control_hz
    value = getattr(args, "arm_control_hz", None)
    if value is not None:
        return float(value)
    if arm_control_mode(args) == "position":
        return float(DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)
    return float(DEFAULT_RM65_ARM_CONTROL_HZ)


def effective_arm_canfd_follow(args: argparse.Namespace) -> bool:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.arm_canfd_follow
    value = getattr(args, "arm_canfd_follow", None)
    if value is not None:
        return bool(value)
    if arm_control_mode(args) == "position":
        return True
    return False


def effective_position_post_ik_qpos_smoothing_alpha(args: argparse.Namespace) -> float:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.position_post_ik_qpos_smoothing_alpha
    canfd_runtime_config = canfd_follow_runtime_config(args)
    position_canfd_passthrough = bool(
        arm_control_mode(args) == "position"
        and canfd_runtime_config["position_background_streaming_enabled"]
    )
    if not position_canfd_passthrough:
        return 1.0
    value = getattr(args, "position_post_ik_qpos_smoothing_alpha", None)
    return 0.9 if value is None else float(value)


def output_qpos_smoothing_singularity_alpha(
    arm_control_mode: str,
    *,
    slow_alpha: float,
    fast_alpha: float,
) -> float | None:
    """Resolve the position-only singularity smoothing endpoint.

    Impedance mode emits Cartesian poses and does not use the position-mode
    post-IK qpos smoothing path.  Keep the optional solver field unset there;
    otherwise its cross-field validation would compare an irrelevant 0.65
    endpoint against the impedance identity slow alpha of 1.0.
    """

    if str(arm_control_mode) != "position":
        return None
    return max(float(slow_alpha), float(fast_alpha), 0.65)


def effective_arm_follow_qpos_alpha(args: argparse.Namespace) -> float:
    """Return final high-rate transport EMA alpha, or 1 outside position streaming."""

    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return resolved.arm_follow_qpos_alpha
    canfd_runtime_config = canfd_follow_runtime_config(args)
    position_streaming = bool(
        arm_control_mode(args) == "position"
        and canfd_runtime_config["position_background_streaming_enabled"]
    )
    if not position_streaming:
        return 1.0
    value = getattr(args, "arm_follow_qpos_alpha", None)
    return 0.1 if value is None else float(value)


def effective_dual_arm_temporary_safety(
    args: argparse.Namespace,
    *,
    position_canfd_passthrough: bool,
) -> dict[str, object]:
    requested = bool(getattr(args, "dual_arm_temporary_safety", True))
    if not requested:
        return {
            "requested": False,
            "enabled": False,
            "disabled_reason": "disabled_by_cli",
        }
    if bool(position_canfd_passthrough):
        return {
            "requested": True,
            "enabled": False,
            "disabled_reason": "position_canfd_passthrough_interpolation_only",
        }
    if arm_control_mode(args) == "impedance":
        # The limiter operates after coupled IK.  In impedance mode that can
        # turn a fresh Cartesian command into a pose without its matching
        # solver qpos, which creates a stateful target chase at the outer
        # (typically 15 Hz) rate.  Coupled IK and the execution joint limits
        # remain active; degraded IK candidates are bounded before dispatch.
        return {
            "requested": True,
            "enabled": False,
            "disabled_reason": "impedance_uses_pre_ik_coupled_safety_bounds",
        }
    return {
        "requested": True,
        "enabled": True,
        "disabled_reason": None,
    }


def position_retargeter_stability_switches(
    args: argparse.Namespace,
    *,
    position_canfd_passthrough: bool,
) -> dict[str, bool]:
    """Resolve live-teleop retargeter stability independently of transport mode.

    The 120 Hz CANFD transport only replaces the final qpos sink.  It must not
    disable the high-rate target smoother, local safe-candidate selection, or the
    inactive-side hold that prevents a stationary arm from following nullspace
    changes caused by the moving arm's coupled IK solve.
    """

    hold_disabled = bool(
        position_canfd_passthrough
        and getattr(args, "safe_debug", False)
        and getattr(args, "disable_hold", False)
    )
    return {
        "target_pose_smoothing_enabled": True,
        "local_step_candidate_selection_enabled": True,
        "inactive_side_hold_enabled": not hold_disabled,
        "inactive_side_hold_execution_cap_enabled": not hold_disabled,
        "hold_disabled": hold_disabled,
    }


def canfd_follow_runtime_config(
    args: argparse.Namespace,
    *,
    arm_control_hz: float | None = None,
    arm_canfd_follow: bool | None = None,
) -> dict[str, object]:
    resolved = getattr(args, "arm_execution_config", None)
    resolved_arm_control_hz = (
        float(effective_arm_control_hz(args))
        if arm_control_hz is None
        else float(arm_control_hz)
    )
    resolved_arm_canfd_follow = (
        bool(effective_arm_canfd_follow(args))
        if arm_canfd_follow is None
        else bool(arm_canfd_follow)
    )
    requested_stream_hz = (
        resolved.arm_canfd_follow_stream_hz
        if isinstance(resolved, RealArmExecutionConfig)
        else getattr(args, "arm_canfd_follow_stream_hz", None)
    )
    requested_stream_hz_float = (
        None
        if requested_stream_hz is None
        else float(requested_stream_hz)
    )
    background_streaming_enabled = resolved_arm_canfd_follow and (
        requested_stream_hz_float is None
        or requested_stream_hz_float > 0.0
    )
    background_streaming_hz = (
        resolved_arm_control_hz
        if requested_stream_hz_float is None
        or requested_stream_hz_float <= 0.0
        else requested_stream_hz_float
    )
    warning_ratio = (
        resolved.arm_canfd_follow_warning_ratio
        if isinstance(resolved, RealArmExecutionConfig)
        else 0.9
        if getattr(args, "arm_canfd_follow_warning_ratio", None) is None
        else float(args.arm_canfd_follow_warning_ratio)
    )
    trajectory_mode = (
        resolved.arm_canfd_trajectory_mode
        if isinstance(resolved, RealArmExecutionConfig)
        else 0
        if getattr(args, "arm_canfd_trajectory_mode", None) is None
        else int(args.arm_canfd_trajectory_mode)
    )
    radio = (
        resolved.arm_canfd_radio
        if isinstance(resolved, RealArmExecutionConfig)
        else 0
        if getattr(args, "arm_canfd_radio", None) is None
        else int(args.arm_canfd_radio)
    )
    return {
        "position_background_streaming_enabled": bool(background_streaming_enabled),
        "position_background_streaming_hz": float(background_streaming_hz),
        "canfd_follow_warning_ratio": float(warning_ratio),
        "canfd_trajectory_mode": int(trajectory_mode),
        "canfd_radio": int(radio),
    }


__all__ = [
    "arm_control_mode",
    "canfd_follow_runtime_config",
    "effective_arm_canfd_follow",
    "effective_arm_control_hz",
    "effective_dual_arm_ik_hz",
    "effective_dual_arm_temporary_safety",
    "effective_outer_control_hz",
    "effective_position_post_ik_qpos_smoothing_alpha",
    "output_qpos_smoothing_singularity_alpha",
    "effective_arm_follow_qpos_alpha",
    "effective_position_reference_hz",
    "position_retargeter_stability_switches",
]
