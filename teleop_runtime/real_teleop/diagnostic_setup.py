"""Construction and attachment of observational real-teleop diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from robot_motion.position_control import (
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    PositionAuditFanout,
)

from ..cli_value_parsers import _active_sides_from_args
from ..position_control import (
    PositionDebugRecorder,
    PositionDebugRecorderConfig,
    RealArmExecutionConfig,
)
from .actuator_command_trace import (
    ActuatorCommandTraceConfig,
    ActuatorCommandTraceRecorder,
)
from .components import RealTeleopComponents
from .control_profile import (
    arm_control_mode as _arm_control_mode_arg,
    canfd_follow_runtime_config as _canfd_follow_runtime_config,
    effective_arm_canfd_follow as _effective_arm_canfd_follow,
    effective_position_post_ik_qpos_smoothing_alpha as _effective_position_post_ik_qpos_smoothing_alpha,
    effective_position_reference_hz as _effective_position_reference_hz,
)
from .runtime_metadata import (
    arm_execution_profile_metadata as _arm_execution_profile_metadata,
    position_ik_iteration_metadata as _position_ik_iteration_metadata,
)


def _attach_position_audit_sink(
    components: RealTeleopComponents,
    sink: PositionAuditFanout | None,
) -> None:
    """Attach one observational sink to the reference publisher and command workers."""

    publisher = components.position_reference_publisher
    if publisher is not None:
        publisher.set_audit_observer(sink)
    for backend in (
        components.left_command_backend,
        components.right_command_backend,
    ):
        setter = getattr(backend, "set_position_audit_sink", None)
        if callable(setter):
            setter(sink)


def _attach_actuator_command_trace_sink(
    components: RealTeleopComponents,
    sink: Callable[..., object] | None,
) -> int:
    """Attach only to the post-SDK fixed-field command trace hook."""

    attached = 0
    for backend in (
        components.left_command_backend,
        components.right_command_backend,
    ):
        setter = getattr(backend, "set_actuator_command_trace_sink", None)
        if callable(setter):
            setter(sink)
            attached += 1
    return attached


def _attach_singularity_transition_trace_sink(
    components: RealTeleopComponents,
    sink: Callable[..., object] | None,
) -> int:
    """Attach the sparse position singularity trace beside the qcmd trace."""

    attached = 0
    for backend in (
        components.left_command_backend,
        components.right_command_backend,
    ):
        setter = getattr(backend, "set_singularity_transition_trace_sink", None)
        if callable(setter):
            setter(sink)
            attached += 1
    return attached


def _new_actuator_command_trace_recorder(
    args: argparse.Namespace,
    *,
    components: RealTeleopComponents,
    output_dir: Path,
) -> ActuatorCommandTraceRecorder:
    resolved = getattr(args, "arm_execution_config", None)
    command_hz = float(
        getattr(resolved, "position_background_streaming_hz", 0.0)
    )
    metadata = {
        "entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "active_sides": list(components.active_sides),
        "logical_arm_ip_mapping": dict(components.logical_arm_ip_mapping),
        **_arm_execution_profile_metadata(args),
        "effective_transport": _canfd_follow_runtime_config(args),
        "position_command_planner": str(
            getattr(args, "position_command_planner", "")
        ),
        "control_debug_level": str(args.control_debug_config.level),
    }
    return ActuatorCommandTraceRecorder(
        ActuatorCommandTraceConfig(
            output_dir=output_dir,
            active_sides=tuple(components.active_sides),
            command_hz=command_hz,
            reference_hz=float(_effective_position_reference_hz(args)),
            max_duration_sec=float(args.actuator_trace_max_duration_sec),
            metadata=metadata,
        )
    )


def _new_position_debug_recorder(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    thread_name: str,
    components: RealTeleopComponents | None = None,
    recorder_scope: str = "session",
    episode_index: int | None = None,
) -> PositionDebugRecorder:
    resolved = getattr(args, "arm_execution_config", None)
    execution = (
        resolved.to_metadata()
        if isinstance(resolved, RealArmExecutionConfig)
        else {"mode": _arm_control_mode_arg(args)}
    )
    canfd_runtime = _canfd_follow_runtime_config(args)
    collision_context = {
        "requested": bool(
            getattr(components, "collision_guard_requested", False)
        ),
        "enabled": bool(
            components is not None
            and components.dual_arm_retargeter is not None
            and components.collision_guard_setup_error is None
        ),
        "scene_profile": str(getattr(args, "collision_scene_profile", "")),
        "setup_error": (
            None
            if components is None
            else components.collision_guard_setup_error
        ),
    }
    context = {
        "recorder_scope": str(recorder_scope),
        "episode_index": None if episode_index is None else int(episode_index),
        "active_sides": list(_active_sides_from_args(args)),
        "arm_execution": execution,
        **_position_ik_iteration_metadata(args),
        "position_max_command_step_deg": (
            None
            if getattr(args, "position_max_command_step_deg", None) is None
            else float(args.position_max_command_step_deg)
        ),
        "position_target_velocity_filter_tau_sec": float(
            getattr(
                args,
                "position_target_velocity_filter_tau_sec",
                DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
            )
        ),
        "effective_transport": {
            "position_background_streaming_enabled": bool(
                canfd_runtime["position_background_streaming_enabled"]
            ),
            "position_background_streaming_hz": float(
                canfd_runtime["position_background_streaming_hz"]
            ),
            "canfd_follow": bool(_effective_arm_canfd_follow(args)),
            "canfd_trajectory_mode": int(canfd_runtime["canfd_trajectory_mode"]),
            "canfd_radio": int(canfd_runtime["canfd_radio"]),
        },
        "semantic_switches": {
            "position_post_ik_qpos_smoothing_alpha": float(
                _effective_position_post_ik_qpos_smoothing_alpha(args)
            ),
            "position_post_ik_qpos_smoothing_adaptive": bool(
                getattr(args, "position_post_ik_qpos_smoothing_adaptive", False)
            ),
            "position_post_ik_qpos_smoothing_fast_alpha": float(
                getattr(args, "position_post_ik_qpos_smoothing_fast_alpha", 0.45)
            ),
            "safe_debug": bool(getattr(args, "safe_debug", False)),
            "disable_hold": bool(getattr(args, "disable_hold", False)),
            "singularity_protection_enabled": bool(
                getattr(args, "arm_singularity_protection", True)
            ),
            "collision_guard": collision_context,
        },
    }
    config = PositionDebugRecorderConfig.from_control_debug_config(
        args.control_debug_config,
        output_dir=output_dir,
        thread_name=thread_name,
        context=context,
    )
    return PositionDebugRecorder(config, auto_start=True)


__all__ = [
    "_attach_position_audit_sink",
    "_attach_actuator_command_trace_sink",
    "_attach_singularity_transition_trace_sink",
    "_new_actuator_command_trace_recorder",
    "_new_position_debug_recorder",
]

