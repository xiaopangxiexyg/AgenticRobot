"""Construction and attachment of observational real-teleop diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

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
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC,
)


class _StartupCleanupPort(Protocol):
    def add(self, label: str, callback: Callable[[], None]) -> None: ...


@dataclass(frozen=True)
class ActuatorTraceSetup:
    """Result of the optional all-sinks actuator trace setup transaction."""

    recorder: ActuatorCommandTraceRecorder | None
    session_dir: Path | None


@dataclass(frozen=True)
class ControlDebugSetup:
    """Result of the optional position-audit observer setup transaction."""

    audit_fanout: PositionAuditFanout | None
    recorder: PositionDebugRecorder | None
    session_dir: Path | None


def initialize_control_debug_runtime_metadata(
    args: argparse.Namespace,
    *,
    merge_runtime_metadata: Callable[..., dict[str, object]],
) -> dict[str, object]:
    """Publish the complete unattached diagnostic metadata baseline."""

    requested = bool(args.control_debug_config.enabled)
    return merge_runtime_metadata(
        args,
        attached=False,
        session_sidecar_requested=requested,
        session_sidecar_attached=False,
        session_sidecar_error=None,
        session_sidecar_output_dir=None,
        session_sidecar_attached_wall_time=None,
        session_sidecar_attached_monotonic_time=None,
        episode_sidecar_requested=bool(
            requested
            and bool(getattr(args, "control_debug_episode_sidecar", True))
            and not bool(getattr(args, "no_collection", False))
        ),
        episode_sidecar_attached=False,
        episode_sidecar_error=None,
        episode_sidecar_output_dir=None,
        episode_sidecar_episode_index=None,
        episode_sidecar_attached_wall_time=None,
        episode_sidecar_attached_monotonic_time=None,
        episode_sidecars=False,
    )


def setup_control_debug_runtime(
    args: argparse.Namespace,
    *,
    components: RealTeleopComponents,
    startup_cleanup: _StartupCleanupPort,
    timestamp: str,
    expand_cli_path: Callable[[str | Path], Path],
    new_recorder: Callable[..., PositionDebugRecorder],
    audit_fanout_type: type[PositionAuditFanout],
    attach_position_sink: Callable[..., None],
    update_runtime_metadata: Callable[..., dict[str, object]],
    wall_time: Callable[[], float],
    monotonic_time: Callable[[], float],
    log: Callable[..., None],
) -> ControlDebugSetup:
    """Attach the optional position audit observer or fully isolate failure."""

    if not bool(args.control_debug_config.enabled):
        return ControlDebugSetup(
            audit_fanout=None,
            recorder=None,
            session_dir=None,
        )

    audit_fanout: PositionAuditFanout | None = None
    recorder: PositionDebugRecorder | None = None
    session_dir: Path | None = None
    try:
        session_dir = (
            expand_cli_path(args.control_debug_config.root)
            / f"{timestamp}_vive_rm65_real_teleop"
        )
        recorder = new_recorder(
            args,
            output_dir=session_dir,
            thread_name="position-debug-session-writer",
            components=components,
            recorder_scope="session",
        )
        startup_cleanup.add(
            "position_debug_session_recorder",
            lambda recorder=recorder: recorder.close(timeout_sec=1.0),
        )
        audit_fanout = audit_fanout_type((recorder,))
        attach_position_sink(components, audit_fanout)
        attached_wall_time = wall_time()
        attached_monotonic_time = monotonic_time()
        update_runtime_metadata(
            attached=True,
            session_output_dir=str(session_dir),
            session_sidecar_attached=True,
            session_sidecar_error=None,
            session_sidecar_output_dir=str(session_dir),
            session_sidecar_attached_wall_time=attached_wall_time,
            session_sidecar_attached_monotonic_time=attached_monotonic_time,
        )
        log(
            "ControlDebug",
            f"async audit attached output={session_dir} "
            f"mode={args.control_debug_config.level}",
        )
        return ControlDebugSetup(
            audit_fanout=audit_fanout,
            recorder=recorder,
            session_dir=session_dir,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot alter control behavior.
        cleanup_errors: list[str] = []
        if recorder is not None:
            try:
                recorder.close(timeout_sec=1.0)
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_errors.append(
                    f"recorder_close:{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        try:
            attach_position_sink(components, None)
        except Exception as cleanup_exc:  # noqa: BLE001
            cleanup_errors.append(
                f"sink_detach:{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        setup_error = f"{type(exc).__name__}: {exc}"
        if cleanup_errors:
            setup_error += "; cleanup=" + "; ".join(cleanup_errors)
        update_runtime_metadata(
            attached=False,
            error=setup_error,
            session_sidecar_attached=False,
            session_sidecar_error=setup_error,
            session_sidecar_output_dir=(
                None if session_dir is None else str(session_dir)
            ),
            session_sidecar_attached_wall_time=None,
            session_sidecar_attached_monotonic_time=None,
        )
        log(
            "ControlDebug",
            "WARNING: async audit setup failed and was isolated: "
            f"{type(exc).__name__}: {exc}",
        )
        return ControlDebugSetup(
            audit_fanout=None,
            recorder=None,
            session_dir=session_dir,
        )


def setup_actuator_command_trace(
    args: argparse.Namespace,
    *,
    components: RealTeleopComponents,
    startup_cleanup: _StartupCleanupPort,
    timestamp: str,
    expand_cli_path: Callable[[str | Path], Path],
    new_recorder: Callable[..., ActuatorCommandTraceRecorder],
    attach_command_sink: Callable[..., int],
    attach_singularity_sink: Callable[..., int],
    attach_ik_shadow_sink: Callable[..., int],
    log: Callable[..., None],
) -> ActuatorTraceSetup:
    """Attach the optional trace atomically, or isolate and roll it back."""

    if not bool(getattr(args, "actuator_trace", False)):
        return ActuatorTraceSetup(recorder=None, session_dir=None)

    recorder: ActuatorCommandTraceRecorder | None = None
    session_dir: Path | None = None
    try:
        session_dir = (
            expand_cli_path(args.actuator_trace_root)
            / f"{timestamp}_vive_rm65_actuator_trace"
        )
        recorder = new_recorder(
            args,
            components=components,
            output_dir=session_dir,
        )
        startup_cleanup.add("actuator_trace_recorder", recorder.close)
        attached_count = attach_command_sink(
            components,
            recorder.record_command,
        )
        if attached_count != len(components.active_sides):
            raise RuntimeError(
                "actuator trace did not attach to every active command backend: "
                f"attached={attached_count} active={len(components.active_sides)}"
            )
        singularity_attached_count = 0
        if bool(getattr(args, "actuator_singularity_trace", True)):
            singularity_attached_count = attach_singularity_sink(
                components,
                recorder.record_singularity_transition,
            )
            if singularity_attached_count != len(components.active_sides):
                raise RuntimeError(
                    "singularity transition trace did not attach to every active "
                    "command backend: "
                    f"attached={singularity_attached_count} "
                    f"active={len(components.active_sides)}"
                )
        ik_shadow_attached_count = attach_ik_shadow_sink(
            components,
            recorder.record_ik_shadow,
        )
        log(
            "ActuatorTrace",
            "preallocated fixed-field capture attached "
            f"output={session_dir} "
            f"capacity_per_side={recorder.config.capacity_per_side} "
            f"singularity_trace={bool(singularity_attached_count)} "
            f"ik_shadow_trace={bool(ik_shadow_attached_count)} "
            "vive_source_trace=true "
            "hot_path_disk_write=false",
        )
        return ActuatorTraceSetup(recorder=recorder, session_dir=session_dir)
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot alter control behavior.
        cleanup_errors: list[str] = []
        for label, detach in (
            ("sink_detach", attach_command_sink),
            ("singularity_sink_detach", attach_singularity_sink),
            ("ik_shadow_sink_detach", attach_ik_shadow_sink),
        ):
            try:
                detach(components, None)
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_errors.append(
                    f"{label}:{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        if recorder is not None:
            try:
                recorder.close()
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_errors.append(
                    f"recorder_close:{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        detail = f"{type(exc).__name__}: {exc}"
        if cleanup_errors:
            detail += "; cleanup=" + "; ".join(cleanup_errors)
        log(
            "ActuatorTrace",
            f"WARNING: setup failed and was isolated: {detail}",
        )
        return ActuatorTraceSetup(recorder=None, session_dir=session_dir)


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


def _attach_ik_shadow_trace_sink(
    components: RealTeleopComponents,
    sink: Callable[..., object] | None,
) -> int:
    """Attach one passive 60 Hz IK trace sink to the shared retargeter."""

    retargeter = components.dual_arm_retargeter
    setter = getattr(retargeter, "set_ik_shadow_trace_sink", None)
    if not callable(setter):
        return 0
    setter(sink)
    return 1


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
        **_position_ik_iteration_metadata(args),
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
            "position_post_ik_qpos_normal_fast_path": bool(
                getattr(args, "position_post_ik_qpos_normal_fast_path", False)
            ),
            "position_post_ik_qpos_normal_fast_path_alpha": float(
                getattr(args, "position_post_ik_qpos_normal_fast_path_alpha", 1.0)
            ),
            "position_post_ik_qpos_normal_fast_path_guard_deg": float(
                getattr(
                    args,
                    "position_post_ik_qpos_normal_fast_path_guard_deg",
                    4.0,
                )
            ),
            "position_post_ik_qpos_normal_fast_path_quality_grace_sec": float(
                getattr(
                    args,
                    "position_post_ik_qpos_normal_fast_path_quality_grace_sec",
                    0.0,
                )
            ),
            "position_admittance_active_qpos_fast_path": bool(
                getattr(args, "position_admittance_active", False)
            ),
            "position_admittance_active_envelope": str(
                getattr(args, "position_admittance_active_envelope", "cautious")
            ),
            "position_admittance_active_max_tool_excursion_m": float(
                getattr(
                    args,
                    "position_admittance_active_max_tool_excursion_m",
                    0.08,
                )
            ),
            "position_admittance_active_qpos_fast_path_alpha": float(
                POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA
            ),
            "position_admittance_rotation_active_qpos_fast_path_alpha": float(
                POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA
            ),
            "position_admittance_rotation_release_qpos_fast_path_alpha": float(
                POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA
            ),
            "position_admittance_rotation_release_qpos_ramp_sec": float(
                POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC
            ),
            "position_admittance_active_qpos_micro_reversal_hold_deg": float(
                POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG
            ),
            "position_wrist_q5_hysteresis_requires_escape_latch": bool(
                getattr(
                    args,
                    "position_wrist_q5_hysteresis_requires_escape_latch",
                    False,
                )
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
    "ActuatorTraceSetup",
    "ControlDebugSetup",
    "_attach_position_audit_sink",
    "_attach_actuator_command_trace_sink",
    "_attach_singularity_transition_trace_sink",
    "_attach_ik_shadow_trace_sink",
    "_new_actuator_command_trace_recorder",
    "_new_position_debug_recorder",
    "setup_actuator_command_trace",
    "setup_control_debug_runtime",
]
