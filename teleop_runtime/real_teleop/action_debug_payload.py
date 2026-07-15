"""Action-debug payload construction for the real teleoperation runtime.

This module owns only snapshot-to-payload serialization and aggregation.  It
does not decide when to read a backend.  The legacy entrypoint keeps the
backend reader seam (including get_state() calls) and binds it explicitly to
preserve timing and compatibility behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import time
from typing import Any


_SUPPORT_NAMES = (
    "_backend_compact_debug_payload",
    "_channels_log_payload",
    "_command_compact_debug_payload",
    "_compact_debug_value",
    "_faults_log_payload",
    "_float_or_none",
    "_phase_timings_log_payload",
    "_pose_delta_compact_debug",
    "_preloaded_action_debug_backend_snapshot",
)


class _LegacyFunctionProxy:
    """Resolve legacy helpers dynamically so old patch points remain live."""

    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive action-debug support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    """Bind readers/compact serializers owned by the runtime entrypoint."""

    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "action-debug payload support is incomplete: "
            + ", ".join(sorted(missing))
        )
    globals().update(
        {
            name: (
                _LegacyFunctionProxy(namespace, name)
                if inspect.isroutine(namespace[name])
                else namespace[name]
            )
            for name in _SUPPORT_NAMES
        }
    )


def _action_debug_active_sides(components: Any | None) -> tuple[str, ...]:
    raw_sides = tuple(str(side) for side in (getattr(components, "active_sides", ()) or ()))
    if raw_sides:
        return raw_sides
    inferred: list[str] = []
    for side in ("left", "right"):
        if (
            getattr(components, f"{side}_command_backend", None) is not None
            or getattr(components, f"{side}_state_backend", None) is not None
        ):
            inferred.append(side)
    return tuple(inferred or ("left", "right"))

def _action_debug_backend_snapshot_from_state(
    state: object,
    *,
    backend: Any | None,
    side: str,
    role: str,
    sample_timestamp: float | None,
    record_wall_time: float,
    read_start: float | None = None,
    read_end: float | None = None,
) -> dict[str, Any]:
    start = float(time.time() if read_start is None else read_start)
    end = float(start if read_end is None else read_end)
    if backend is None:
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "reason": "missing_backend",
        }
    if not isinstance(state, Mapping):
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "backend_class": type(backend).__name__,
            "reason": "state_not_mapping",
            "state_type": type(state).__name__,
            "read_start_wall_time": start,
            "read_end_wall_time": end,
            "read_elapsed_s": float(end - start),
        }

    sample_ts = _float_or_none(sample_timestamp)
    record_ts = _float_or_none(record_wall_time)
    last_command_ts = _float_or_none(state.get("last_command_timestamp"))
    pose_tracking_error = _pose_delta_compact_debug(
        state.get("current_tcp_pose"),
        state.get("last_target_tcp_pose"),
    )
    compact_state = _backend_compact_debug_payload(state)
    out: dict[str, Any] = {
        "available": True,
        "side": str(side),
        "role": str(role),
        "backend_class": type(backend).__name__,
        "arm_state_source": state.get("arm_state_source"),
        "read_start_wall_time": start,
        "read_end_wall_time": end,
        "read_elapsed_s": float(end - start),
        "sample_timestamp": sample_ts,
        "record_wall_time": record_ts,
        "last_command_timestamp": last_command_ts,
        "last_command_sequence": compact_state.get("last_command_sequence"),
        "command_sequence": compact_state.get("command_sequence"),
        "latest_target_sequence": compact_state.get("latest_target_sequence"),
        "hardware_result_code": state.get("hardware_result_code"),
        "hardware_result_text": state.get("hardware_result_text"),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": state.get("send_result_text"),
        "effective_control_dt_sec": state.get("effective_control_dt_sec"),
        "effective_step_scale": state.get("effective_step_scale"),
        "latest_target_tcp_pose6": compact_state.get("latest_target_tcp_pose6"),
        "latest_target_joint_deg": compact_state.get("latest_target_joint_deg"),
        "current_qpos": compact_state.get("current_qpos"),
        "last_target_qpos": compact_state.get("last_target_qpos"),
        "current_tcp_pose": compact_state.get("current_tcp_pose"),
        "last_target_tcp_pose": compact_state.get("last_target_tcp_pose"),
        "pose_tracking_error": pose_tracking_error,
        "last_send_command_timing": compact_state.get("last_send_command_timing"),
        "last_impedance_loop_timing": compact_state.get("last_impedance_loop_timing"),
        "last_impedance_record": compact_state.get("last_impedance_record"),
        "recent_impedance_records": compact_state.get("recent_impedance_records"),
        "backend_state_compact": compact_state,
    }
    if last_command_ts is not None and sample_ts is not None:
        out["last_command_age_sec_at_sample"] = float(sample_ts - last_command_ts)
        out["sample_to_last_command_dt_sec"] = float(last_command_ts - sample_ts)
    else:
        out["last_command_age_sec_at_sample"] = None
        out["sample_to_last_command_dt_sec"] = None
    if last_command_ts is not None and record_ts is not None:
        out["last_command_age_sec_at_record"] = float(record_ts - last_command_ts)
        out["record_to_last_command_dt_sec"] = float(last_command_ts - record_ts)
    else:
        out["last_command_age_sec_at_record"] = None
        out["record_to_last_command_dt_sec"] = None
    if pose_tracking_error is not None:
        out["target_current_translation_m"] = pose_tracking_error.get("translation_m")
        out["target_current_rotation_deg"] = pose_tracking_error.get("rotation_deg")
    return out

def _action_debug_jitter_focus(
    command_backend_by_side: Mapping[str, Any],
) -> dict[str, Any]:
    focus_by_side: dict[str, Any] = {}
    for side, snapshot in command_backend_by_side.items():
        if not isinstance(snapshot, Mapping):
            continue
        compact_state = snapshot.get("backend_state_compact")
        if not isinstance(compact_state, Mapping):
            compact_state = {}
        record = compact_state.get("last_impedance_record")
        if not isinstance(record, Mapping):
            record = {}
        stream = record.get("canfd_follow_stream")
        if not isinstance(stream, Mapping):
            stream = {}
        position_stream = compact_state.get("background_streaming")
        if not isinstance(position_stream, Mapping):
            position_stream = {}
        stream_keys = (
            "enabled",
            "hz",
            "warning_ratio",
            "warning_threshold_hz",
            "thread_alive",
            "target_active",
            "target_sequence",
            "target_age_sec",
            "last_send_sequence",
            "send_count",
            "last_send_dt_sec",
            "actual_hz",
            "last_call_ms",
            "last_deadline_lag_ms",
            "last_sdk_lock_wait_ms",
            "last_interpolation_alpha",
            "last_sent_joint_delta_deg",
            "last_sent_joint_delta_norm_deg",
            "last_sent_joint_step_deg",
            "deadline_miss_count",
            "low_rate_count",
            "warning",
            "fault",
        )
        focus_by_side[str(side)] = {
            "available": bool(snapshot.get("available", False)),
            "backend_class": snapshot.get("backend_class"),
            "latest_target_sequence": snapshot.get("latest_target_sequence"),
            "last_command_sequence": snapshot.get("last_command_sequence"),
            "command_sequence": snapshot.get("command_sequence"),
            "latest_target_tcp_pose6": snapshot.get("latest_target_tcp_pose6"),
            "latest_target_joint_deg": snapshot.get("latest_target_joint_deg"),
            "current_qpos": snapshot.get("current_qpos"),
            "last_target_qpos": snapshot.get("last_target_qpos"),
            "pose_tracking_error": snapshot.get("pose_tracking_error"),
            "backend_target_sequence": record.get("backend_target_sequence"),
            "backend_target_dispatched_this_step": record.get(
                "backend_target_dispatched_this_step"
            ),
            "backend_last_dispatched_target_sequence": record.get(
                "backend_last_dispatched_target_sequence"
            ),
            "backend_target_age_sec_at_snapshot": record.get(
                "backend_target_age_sec_at_snapshot"
            ),
            "backend_target_age_sec_at_step_start": record.get(
                "backend_target_age_sec_at_step_start"
            ),
            "backend_target_age_sec_at_step_end": record.get(
                "backend_target_age_sec_at_step_end"
            ),
            "backend_target_joint_hint_present": record.get(
                "backend_target_joint_hint_present"
            ),
            "reference": record.get("reference"),
            "passthrough_position_control": record.get("passthrough_position_control"),
            "force_admittance_skipped": record.get("force_admittance_skipped"),
            "command_pose": record.get("command_pose"),
            "pre_follow_smoothing_command_pose": record.get(
                "pre_follow_smoothing_command_pose"
            ),
            "command_joint_deg": record.get("command_joint_deg"),
            "pre_follow_smoothing_command_joint_deg": record.get(
                "pre_follow_smoothing_command_joint_deg"
            ),
            "measured_joint_deg": record.get("measured_joint_deg"),
            "command_joint_step_deg": record.get("command_joint_step_deg"),
            "command_to_measured_joint_deg": record.get("command_to_measured_joint_deg"),
            "measured_joint_velocity_deg_s": record.get("measured_joint_velocity_deg_s"),
            "command_to_measured_linear_m": record.get("command_to_measured_linear_m"),
            "command_to_measured_linear_norm_m": record.get(
                "command_to_measured_linear_norm_m"
            ),
            "command_to_measured_angular_norm_rad": record.get(
                "command_to_measured_angular_norm_rad"
            ),
            "force_source": record.get("force_source"),
            "selected_force_wrench": record.get("selected_force_wrench"),
            "tare_corrected_wrench": record.get("tare_corrected_wrench"),
            "filtered_wrench": record.get("filtered_wrench"),
            "raw_deadbanded_wrench": record.get("raw_deadbanded_wrench"),
            "admittance_wrench": record.get("admittance_wrench"),
            "return_activity_wrench": record.get("return_activity_wrench"),
            "active_wrench": record.get("active_wrench"),
            "target_offset": record.get("target_offset"),
            "target_offset_after_offset_limit": record.get(
                "target_offset_after_offset_limit"
            ),
            "offset": record.get("offset"),
            "applied_offset": record.get("applied_offset"),
            "offset_delta_before_velocity_limit": record.get(
                "offset_delta_before_velocity_limit"
            ),
            "offset_step_after_velocity_limit": record.get(
                "offset_step_after_velocity_limit"
            ),
            "applied_offset_source": record.get("applied_offset_source"),
            "admittance_offset_antiwindup_applied": record.get(
                "admittance_offset_antiwindup_applied"
            ),
            "admittance_offset_antiwindup_would_apply": record.get(
                "admittance_offset_antiwindup_would_apply"
            ),
            "admittance_offset_state_update_enabled": record.get(
                "admittance_offset_state_update_enabled"
            ),
            "admittance_offset_antiwindup_skip_reason": record.get(
                "admittance_offset_antiwindup_skip_reason"
            ),
            "admittance_offset_antiwindup_guarded_axes": record.get(
                "admittance_offset_antiwindup_guarded_axes"
            ),
            "admittance_offset_antiwindup_guard_reason": record.get(
                "admittance_offset_antiwindup_guard_reason"
            ),
            "final_command_pose_realized_offset": record.get(
                "final_command_pose_realized_offset"
            ),
            "final_command_pose_realization_source": record.get(
                "final_command_pose_realization_source"
            ),
            "applied_offset_realization_linear_error_m": record.get(
                "applied_offset_realization_linear_error_m"
            ),
            "applied_offset_realization_angular_error_rad": record.get(
                "applied_offset_realization_angular_error_rad"
            ),
            "linear_velocity_limit_applied": record.get("linear_velocity_limit_applied"),
            "angular_velocity_limit_applied": record.get("angular_velocity_limit_applied"),
            "angular_axis_velocity_limit_applied": record.get(
                "angular_axis_velocity_limit_applied"
            ),
            "linear_step_velocity_limit_m_s": record.get("linear_step_velocity_limit_m_s"),
            "angular_step_velocity_limit_rad_s": record.get(
                "angular_step_velocity_limit_rad_s"
            ),
            "joint_rate_limited": record.get("joint_rate_limited"),
            "allowed_command_joint_step_deg": record.get(
                "allowed_command_joint_step_deg"
            ),
            "joint_velocity_limit_ratio": record.get("joint_velocity_limit_ratio"),
            "reference_to_command_linear_m": record.get(
                "reference_to_command_linear_m"
            ),
            "reference_to_command_angular_rotvec_rad": record.get(
                "reference_to_command_angular_rotvec_rad"
            ),
            "commanded_offset_linear_norm_m": record.get(
                "commanded_offset_linear_norm_m"
            ),
            "commanded_offset_angular_norm_rad": record.get(
                "commanded_offset_angular_norm_rad"
            ),
            "admittance_offset_linear_norm_m": record.get(
                "admittance_offset_linear_norm_m"
            ),
            "admittance_offset_angular_norm_rad": record.get(
                "admittance_offset_angular_norm_rad"
            ),
            "ik_method": record.get("ik_method"),
            "ik_metadata": record.get("ik_metadata"),
            "ik_timing": record.get("ik_timing"),
            "follow_qpos_smoothing": record.get("follow_qpos_smoothing"),
            "canfd_follow_stream": {
                key: stream.get(key)
                for key in stream_keys
                if key in stream
            },
            "position_background_streaming": {
                key: position_stream.get(key)
                for key in (
                    "enabled",
                    "hz",
                    "thread_alive",
                    "target_active",
                    "target_source",
                    "target_sequence",
                    "target_age_sec",
                    "target_stale",
                    "command_control_dt_sec",
                    "segment_duration_sec",
                    "tick_sequence",
                    "scheduled_tick_count",
                    "executed_tick_count",
                    "skipped_tick_count",
                    "send_count",
                    "accepted_send_count",
                    "failed_send_count",
                    "deadline_miss_count",
                    "last_send_dt_sec",
                    "current_inter_send_gap_sec",
                    "max_inter_send_gap_sec",
                    "actual_hz",
                    "last_call_ms",
                    "last_deadline_lag_ms",
                    "last_sent_step_deg",
                    "stale_count",
                    "fault",
                    "fault_count",
                    "healthy",
                    "health_reasons",
                    "warning",
                    "timing_percentiles",
                )
                if key in position_stream
            },
        }
    return focus_by_side

def _build_action_debug_metadata(
    step_result: SessionStepResult,
    *,
    components: RealTeleopComponents | Any | None,
    record_wall_time: float,
    step_index: int | None = None,
    loop_start: float | None = None,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
    left_hw_state: Mapping[str, object] | None = None,
    right_hw_state: Mapping[str, object] | None = None,
    target_step_scale: float | None = None,
    left_clutch: bool | None = None,
    right_clutch: bool | None = None,
    ik_timing_payload: Mapping[str, object] | None = None,
    loop_phases: Sequence[tuple[str, float]] | None = None,
    loop_elapsed_sec: float | None = None,
    recording_status: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    sample_timestamp = _float_or_none(getattr(step_result, "timestamp", None))
    command = getattr(step_result, "command", None)
    command_timestamp = _float_or_none(getattr(command, "timestamp", None))
    command_age_sec_at_sample = (
        None
        if sample_timestamp is None or command_timestamp is None
        else float(sample_timestamp - command_timestamp)
    )
    commands = getattr(command, "commands", {}) or {}
    if not isinstance(commands, Mapping):
        commands = {}
    active_sides = _action_debug_active_sides(components)
    command_backend_by_side: dict[str, Any] = {}
    observation_backend_by_side: dict[str, Any] = {}
    command_by_side: dict[str, Any] = {}
    command_states = {
        "left": left_command_state,
        "right": right_command_state,
    }
    hw_states = {
        "left": left_hw_state,
        "right": right_hw_state,
    }
    for side in active_sides:
        command_backend = getattr(components, f"{side}_command_backend", None)
        state_backend = getattr(components, f"{side}_state_backend", None)
        command_backend_by_side[side] = _preloaded_action_debug_backend_snapshot(
            command_backend,
            command_states.get(side),
            side=side,
            role="command_backend",
            sample_timestamp=sample_timestamp,
            record_wall_time=record_wall_time,
        )
        if state_backend is command_backend and command_backend is not None:
            observation_backend_by_side[side] = {
                "available": True,
                "side": str(side),
                "role": "observation_backend",
                "same_as_command_backend": True,
                "backend_class": type(command_backend).__name__,
            }
        else:
            observation_backend_by_side[side] = _preloaded_action_debug_backend_snapshot(
                state_backend,
                hw_states.get(side),
                side=side,
                role="observation_backend",
                sample_timestamp=sample_timestamp,
                record_wall_time=record_wall_time,
            )
        command_by_side[side] = _command_compact_debug_payload(commands.get(f"{side}_arm"))

    out: dict[str, Any] = {
        "schema": "dexteleop_action_debug_v1",
        "event": "frame",
        "enabled": True,
        "sample_timestamp": sample_timestamp,
        "record_wall_time": float(record_wall_time),
        "record_wall_minus_sample_timestamp_sec": (
            None
            if sample_timestamp is None
            else float(record_wall_time - sample_timestamp)
        ),
        "teleop_command_timestamp": command_timestamp,
        "teleop_command_age_sec_at_sample": command_age_sec_at_sample,
        "teleop_command_sample_to_command_dt_sec": (
            None if command_age_sec_at_sample is None else float(-command_age_sec_at_sample)
        ),
        "command_dispatched": bool(getattr(step_result, "command_dispatched", False)),
        "teleop_state": str(
            getattr(getattr(step_result, "state", None), "value", "") or ""
        ),
        "motion_hold_mode": str(
            getattr(step_result, "motion_hold_mode", "") or ""
        ),
        "motion_hold_reason": str(
            getattr(step_result, "motion_hold_reason", "") or ""
        ),
        "tracker_loss_duration_sec": float(
            getattr(step_result, "tracker_loss_duration_sec", 0.0) or 0.0
        ),
        "tracker_loss_channels": list(
            getattr(step_result, "tracker_loss_channels", ()) or ()
        ),
        "home_recalibration_required": bool(
            getattr(step_result, "home_recalibration_required", False)
        ),
        "faults": _faults_log_payload(step_result),
        "tracker_channels": {
            name: payload
            for name, payload in _channels_log_payload(
                step_result,
                now=float(record_wall_time),
            ).items()
            if name in {"left_wrist_pose", "right_wrist_pose"}
        },
        "active_sides": list(active_sides),
        "command_effectors": sorted(str(name) for name in commands),
        "command_by_side": command_by_side,
        "command_backend_by_side": command_backend_by_side,
        "observation_backend_by_side": observation_backend_by_side,
        "jitter_focus_by_side": _action_debug_jitter_focus(command_backend_by_side),
    }
    if step_index is not None:
        out["step_index"] = int(step_index)
    if loop_start is not None:
        out["loop_start_wall_time"] = float(loop_start)
        out["record_wall_minus_loop_start_sec"] = float(record_wall_time - float(loop_start))
    if target_step_scale is not None:
        out["target_step_scale"] = float(target_step_scale)
    if left_clutch is not None or right_clutch is not None:
        out["clutch"] = {
            "left": bool(left_clutch),
            "right": bool(right_clutch),
        }
    if ik_timing_payload is not None:
        out["ik_timing"] = _compact_debug_value(dict(ik_timing_payload), max_items=24)
    if loop_phases is not None:
        out["phases"] = _phase_timings_log_payload(loop_phases)
    if loop_elapsed_sec is not None:
        out["loop_elapsed_ms"] = float(loop_elapsed_sec) * 1000.0
    if recording_status is not None:
        out["recording"] = _compact_debug_value(dict(recording_status), max_items=24)
    return out

__all__ = [
    "bind_legacy_support",
    "_action_debug_active_sides",
    "_action_debug_backend_snapshot_from_state",
    "_action_debug_jitter_focus",
    "_build_action_debug_metadata",
]

