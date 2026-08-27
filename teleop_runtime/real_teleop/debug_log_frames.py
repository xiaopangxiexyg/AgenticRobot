"""Read-only debug frame assembly for the real-teleop runtime.

The functions in this module project already-read command, backend, camera,
and session snapshots into JSON-ready dictionaries. They do not read a
device, dispatch a command, publish a UI update, or mutate runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from robot_motion.rm65 import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from robot_motion.runtime import TeleopCommand

from ..collection.camera_preview import CameraPreviewer
from ..sources.camera import RealSenseRGBSource
from .arm_state_sync import _state_qpos
from .components import RealTeleopComponents
from .debug_ik_payload import (
    _backend_compact_debug_payload,
    _backend_debug_payload,
    _backend_ik_metadata_compact_debug,
    _command_compact_debug_payload,
    _command_debug_payload,
    _impedance_record_compact_debug,
    _observation_pose_compact_debug,
    _observation_pose_debug,
    _teleop_command_metadata_compact_debug,
    _teleop_command_metadata_debug,
)
from .debug_payload_primitives import (
    _compact_debug_value,
    _qpos_compact_debug,
    _qpos_debug,
)
from .debug_recorders import state_value as _state_value
from .debug_timing_payload import (
    _channels_log_payload,
    _fault_signature,
    _faults_log_payload,
    _latency_log_payload,
    _phase_timings_log_payload,
)
from .gui_special_arm_status import (
    fix_arm_debug_state as _fix_arm_debug_state,
    init_arm_debug_state as _init_arm_debug_state,
)
from .runtime_health import _short_status_text
from .status_payloads import _build_camera_status


def _command_for_side(active_command: TeleopCommand | None, side: str):
    if active_command is None:
        return None
    return active_command.commands.get(f"{side}_arm")


def _build_ik_debug_log_frame(
    *,
    components: RealTeleopComponents,
    step_index: int,
    loop_start: float,
    step_result,
    active_command: TeleopCommand | None,
    left_command_state: Mapping[str, object] | None,
    right_command_state: Mapping[str, object] | None,
    left_hw_state: Mapping[str, object] | None,
    right_hw_state: Mapping[str, object] | None,
    ik_timing_payload: Mapping[str, object] | None,
    target_step_scale: float,
    left_clutch: bool,
    right_clutch: bool,
    full: bool = False,
    diagnostic_profile: str | None = None,
    camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    loop_phases: Sequence[tuple[str, float]] | None = None,
    loop_elapsed_sec: float | None = None,
) -> dict[str, object]:
    fault_codes = [
        f"{fault.code}:{fault.channel_name}" if fault.channel_name else fault.code
        for fault in getattr(step_result, "faults", ())
    ]
    base = {
        "schema": "dexteleop_ik_debug_v1",
        "event": "frame",
        "step_index": int(step_index),
        "timestamp": float(loop_start),
        "state": getattr(getattr(step_result, "state", None), "value", str(getattr(step_result, "state", ""))),
        "control_hz": float(getattr(step_result, "control_hz", 0.0) or 0.0),
        "faults": fault_codes,
        "clutch": {"left": bool(left_clutch), "right": bool(right_clutch)},
        "fix_arm": _fix_arm_debug_state(components),
        "init_arm": _init_arm_debug_state(components),
        "target_step_scale": float(target_step_scale),
        "ik_timing": dict(ik_timing_payload or {}),
    }
    if diagnostic_profile:
        # The 0827 investigation deliberately extends the existing compact
        # IK frame with read-only timing/observation snapshots.  Keep this
        # branch opt-in so historical GUI logging and its payload shape stay
        # unchanged when no investigation profile is requested.
        base["recording_profile"] = str(diagnostic_profile)
        base["latency"] = _latency_log_payload(step_result, now=loop_start)
        base["channels"] = _channels_log_payload(step_result, now=loop_start)
        base["camera_status"] = _camera_status_log_payload(
            camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=None,
            now=loop_start,
        )
    if loop_elapsed_sec is not None or loop_phases:
        base["outer_loop_timing"] = {
            "elapsed_ms": (
                None
                if loop_elapsed_sec is None
                else float(loop_elapsed_sec) * 1000.0
            ),
            "phases_ms": [
                {"name": str(name), "elapsed_ms": float(elapsed) * 1000.0}
                for name, elapsed in tuple(loop_phases or ())
            ],
        }
    if full:
        return {
            **base,
            "log_mode": "full",
            "retargeter_command": {
                "metadata": _teleop_command_metadata_debug(active_command),
                "left": _command_debug_payload(_command_for_side(active_command, "left")),
                "right": _command_debug_payload(_command_for_side(active_command, "right")),
            },
            "backend": {
                "left_command": _backend_debug_payload(left_command_state),
                "right_command": _backend_debug_payload(right_command_state),
                "left_hardware": _backend_debug_payload(left_hw_state),
                "right_hardware": _backend_debug_payload(right_hw_state),
            },
            "preview_qpos": {
                "left": _qpos_debug(_state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)),
                "right": _qpos_debug(_state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)),
            },
            "observation": _observation_pose_debug(step_result),
        }
    return {
        **base,
        "log_mode": "compact",
        "retargeter_command": {
            "metadata": _teleop_command_metadata_compact_debug(active_command),
            "left": _command_compact_debug_payload(_command_for_side(active_command, "left")),
            "right": _command_compact_debug_payload(_command_for_side(active_command, "right")),
        },
        "backend": {
            "left_command": _backend_compact_debug_payload(left_command_state),
            "right_command": _backend_compact_debug_payload(right_command_state),
            "left_hardware": _backend_compact_debug_payload(left_hw_state),
            "right_hardware": _backend_compact_debug_payload(right_hw_state),
        },
        "preview_qpos": {
            "left": _qpos_compact_debug(_state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)),
            "right": _qpos_compact_debug(_state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)),
        },
        "observation": _observation_pose_compact_debug(step_result),
    }


def _camera_status_log_payload(
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    *,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None,
    camera_previewer: CameraPreviewer | None,
    now: float,
) -> dict[str, object]:
    statuses = _build_camera_status(
        camera_sources,
        preview_camera_sources=preview_camera_sources,
        camera_previewer=camera_previewer,
        now=now,
    )
    selected_keys = (
        "status_level",
        "status_label",
        "status_message",
        "serial",
        "width",
        "height",
        "fps",
        "frame_count",
        "last_frame_age_sec",
        "stale_after_sec",
        "timeout_count",
        "consecutive_timeout_count",
        "restart_count",
        "last_error",
        "last_runtime_error",
        "preview_active",
        "depth_required",
        "depth_status_level",
        "depth_status_label",
        "depth_status_message",
        "depth_frame_number",
        "depth_camera_timestamp_sec",
        "depth_sensor_timestamp_ms",
        "depth_signature_unique_count",
        "depth_signature_repeat_count",
        "depth_timestamp_age_sec",
    )
    return {
        str(camera_key): {
            key: _compact_debug_value(status.get(key), max_items=8)
            for key in selected_keys
            if key in status
        }
        for camera_key, status in statuses.items()
        if isinstance(status, Mapping)
    }


def _backend_timing_log_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    selected_keys = (
        "hardware_result_code",
        "hardware_result_text",
        "send_result_code",
        "send_result_text",
        "effective_control_dt_sec",
        "effective_step_scale",
        "impedance_enabled",
        "impedance_loop_state",
        "active_tool_name",
        "active_tool_profile_path",
        "active_tool_frame",
        "profile_tcp",
        "last_command_controlled_frame",
        "latest_target_tcp_pose6",
        "latest_target_joint_deg",
        "fault",
    )
    out: dict[str, object] = {
        key: _compact_debug_value(state.get(key), max_items=12)
        for key in selected_keys
        if key in state and state.get(key) is not None
    }
    if state.get("last_send_command_timing") is not None:
        out["last_send_command_timing"] = _compact_debug_value(
            state.get("last_send_command_timing"),
            max_items=24,
        )
    if state.get("last_impedance_loop_timing") is not None:
        out["last_impedance_loop_timing"] = _compact_debug_value(
            state.get("last_impedance_loop_timing"),
            max_items=24,
        )
    if state.get("active_impedance_config") is not None:
        out["active_impedance_config"] = _compact_debug_value(
            state.get("active_impedance_config"),
            max_items=80,
        )
    impedance_record = _impedance_record_compact_debug(state.get("last_impedance_record"))
    if impedance_record:
        out["last_impedance_record"] = impedance_record
    recent_records = [
        item
        for item in (
            _impedance_record_compact_debug(record)
            for record in list(state.get("recent_impedance_records") or [])[-4:]
        )
        if item
    ]
    if recent_records:
        out["recent_impedance_records"] = recent_records
    ik_metadata = _backend_ik_metadata_compact_debug(state.get("ik_metadata"))
    if ik_metadata:
        out["ik_metadata"] = ik_metadata
    return out


def _build_debug_timing_log_frame(
    *,
    components: RealTeleopComponents,
    step_index: int,
    loop_start: float,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    reason: str,
    step_result: object,
    active_command: TeleopCommand | None,
    left_command_state: Mapping[str, object] | None,
    right_command_state: Mapping[str, object] | None,
    left_hw_state: Mapping[str, object] | None,
    right_hw_state: Mapping[str, object] | None,
    ik_timing_payload: Mapping[str, object] | None,
    target_step_scale: float,
    left_clutch: bool,
    right_clutch: bool,
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None,
    camera_previewer: CameraPreviewer | None,
    hand_input_status: object,
    gripper_status: object,
    recording_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    faults = tuple(getattr(step_result, "faults", ()) or ())
    return {
        "schema": "dexteleop_debug_timing_v1",
        "event": "frame",
        "reason": str(reason),
        "step_index": int(step_index),
        "timestamp": float(loop_start),
        "loop_elapsed_ms": float(elapsed_sec) * 1000.0,
        "control_hz": float(getattr(step_result, "control_hz", 0.0) or 0.0),
        "state": _state_value(getattr(step_result, "state", "")),
        "session_fault_reason": _short_status_text(
            getattr(getattr(components, "session", None), "last_fault_reason", ""),
            max_len=400,
        ),
        "fault_codes": list(_fault_signature(faults)),
        "faults": _faults_log_payload(step_result),
        "clutch": {"left": bool(left_clutch), "right": bool(right_clutch)},
        "fix_arm": _fix_arm_debug_state(components),
        "init_arm": _init_arm_debug_state(components),
        "target_step_scale": float(target_step_scale),
        "recording": _compact_debug_value(dict(recording_status or {}), max_items=16),
        "phases": _phase_timings_log_payload(phases),
        "latency": _latency_log_payload(step_result, now=loop_start),
        "channels": _channels_log_payload(step_result, now=loop_start),
        "camera_status": _camera_status_log_payload(
            camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=loop_start,
        ),
        "ik_timing": _compact_debug_value(dict(ik_timing_payload or {}), max_items=24),
        "retargeter_command": {
            "metadata": _teleop_command_metadata_compact_debug(active_command),
        },
        "backend": {
            "left_command": _backend_timing_log_payload(left_command_state),
            "right_command": _backend_timing_log_payload(right_command_state),
            "left_hardware": _backend_timing_log_payload(left_hw_state),
            "right_hardware": _backend_timing_log_payload(right_hw_state),
        },
        "hand_input_status": _compact_debug_value(hand_input_status, max_items=16),
        "gripper_status": _compact_debug_value(gripper_status, max_items=16),
        "active_sides": list(getattr(components, "active_sides", ()) or ()),
    }


__all__ = [
    "_command_for_side",
    "_build_ik_debug_log_frame",
    "_camera_status_log_payload",
    "_backend_timing_log_payload",
    "_build_debug_timing_log_frame",
]
