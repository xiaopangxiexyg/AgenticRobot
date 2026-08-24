"""Read-only Q6 and gripper status projections for real teleoperation.

These helpers inspect already-created channels, retargeters, and backend state
and return status payloads/signatures.  They never send commands, change
backend configuration, or publish GUI events.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np

from ..backends.grippers import GRIPPER_ACTION_MAX
from ..retargeters import (
    DEFAULT_GRIPPER_INPUT_MAX,
    DEFAULT_GRIPPER_INPUT_MIN,
    ScalarToGripperRetargeter,
    map_scalar_to_gripper_target,
)
from .status_payloads import (
    _compact_runtime_bool,
    _compact_runtime_float,
    _json_safe_value,
    _safe_float_value,
)
from .components import RealTeleopComponents
from .hand_runtime import (
    _gripper_backend_for_side,
    _gripper_input_channel_name,
    _gripper_retargeter_for_side,
)

def _q6_channel_status(
    channel: object | None,
    *,
    side: str,
    now: float,
    channel_name: str | None = None,
) -> dict[str, object]:
    channel_name = channel_name or f"{side}_hand_q6"
    status: dict[str, object] = {
        "q6_channel_present": channel is not None,
        "q6_channel_name": channel_name,
        "q6_valid": False,
        "q6_age_sec": None,
        "q6_values": None,
        "q6_value_shape": None,
        "q6_port": None,
        "q6_baudrate": None,
        "q6_read_error": None,
        "q6_reader_error_count": None,
        "q6_reader_recovery_count": None,
        "q6_stale_frames_dropped": None,
        "q6_serial_read_stats": None,
    }
    if channel is None:
        return status

    status["q6_channel_name"] = str(getattr(channel, "channel_name", channel_name) or channel_name)
    status["q6_channel_source"] = str(getattr(channel, "source_name", "") or "")
    status["q6_valid"] = bool(getattr(channel, "valid", False))
    timestamp = getattr(channel, "timestamp", None)
    if timestamp is not None:
        status["q6_age_sec"] = round(max(0.0, float(now) - float(timestamp)), 3)

    try:
        value_array = np.asarray(getattr(channel, "value"), dtype=float)
        status["q6_value_shape"] = tuple(int(dim) for dim in value_array.shape)
        flat_values = value_array.reshape(-1)
        status["q6_values"] = flat_values.round(4).tolist()
    except Exception as exc:  # noqa: BLE001
        status["q6_value_error"] = f"{type(exc).__name__}: {exc}"

    metadata = getattr(channel, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        status["q6_port"] = metadata.get("port")
        status["q6_baudrate"] = metadata.get("baudrate")
        status["q6_read_error"] = metadata.get("read_error") or metadata.get("error")
        status["q6_reader_error_count"] = (
            metadata.get("reader_error_count")
            if metadata.get("reader_error_count") is not None
            else metadata.get("error_count")
        )
        status["q6_reader_recovery_count"] = metadata.get("reader_recovery_count")
        status["q6_stale_frames_dropped"] = metadata.get("stale_q6_frames_dropped")
        serial_read_stats = metadata.get("serial_read_stats")
        if isinstance(serial_read_stats, Mapping):
            status["q6_serial_read_stats"] = _json_safe_value(serial_read_stats)
    return status


def _q6_retarget_status(
    components: RealTeleopComponents,
    retargeter: ScalarToGripperRetargeter | None,
    *,
    q6_values: object,
) -> dict[str, object]:
    source_dim = int(components.gripper_policy.get("source_dim", 2))
    output_dim = int(components.gripper_policy.get("output_dim", 1))
    invert = bool(components.gripper_policy.get("invert", False))
    alpha = float(components.gripper_policy.get("alpha", 1.0))
    input_min = float(components.gripper_policy.get("input_min", DEFAULT_GRIPPER_INPUT_MIN))
    input_max = float(components.gripper_policy.get("input_max", DEFAULT_GRIPPER_INPUT_MAX))
    output_min = 0.0
    output_max = float(components.gripper_policy.get("max_action", GRIPPER_ACTION_MAX))
    if retargeter is not None:
        config = getattr(retargeter, "config", None)
        if config is not None:
            configured_source_dim = getattr(config, "source_dim", source_dim)
            source_dim = None if configured_source_dim is None else int(configured_source_dim)
            output_dim = int(getattr(config, "output_dim", output_dim))
            invert = bool(getattr(config, "invert", invert))
            alpha = float(getattr(config, "alpha", alpha))
            input_min = float(getattr(config, "input_min", input_min))
            input_max = float(getattr(config, "input_max", input_max))
            output_min = float(getattr(config, "output_min", output_min))
            output_max = float(getattr(config, "output_max", output_max))

    source_value: float | None = None
    normalized_value: float | None = None
    target_preview: float | None = None
    try:
        q6_array = np.asarray(q6_values, dtype=float).reshape(-1)
        if source_dim is None and q6_array.size == 1:
            raw_source = _safe_float_value(q6_array[0])
        elif source_dim is not None and 0 <= source_dim < q6_array.size:
            raw_source = _safe_float_value(q6_array[source_dim])
        else:
            raw_source = None
        if raw_source is not None:
            source_value = float(np.clip(raw_source, 0.0, 1.0))
            target_preview, normalized_value = map_scalar_to_gripper_target(
                source_value,
                input_min=input_min,
                input_max=input_max,
                invert=invert,
                output_min=output_min,
                output_max=output_max,
            )
    except Exception:
        source_value = None
        normalized_value = None
        target_preview = None

    filtered_target = None
    if retargeter is not None:
        filtered_target = _safe_float_value(getattr(retargeter, "_filtered_value", None))

    return {
        "q6_control_available": retargeter is not None,
        "q6_source_dim": source_dim,
        "q6_output_dim": output_dim,
        "q6_invert": invert,
        "q6_alpha": alpha,
        "q6_input_min": input_min,
        "q6_input_max": input_max,
        "q6_output_min": output_min,
        "q6_output_max": output_max,
        "q6_source_value": source_value,
        "q6_normalized_value": normalized_value,
        "q6_target_preview": target_preview,
        "q6_filtered_target": filtered_target,
    }


def _gripper_status_for_side(
    components: RealTeleopComponents,
    side: str,
    *,
    channels: dict[str, object] | None,
    now: float,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    backend = _gripper_backend_for_side(components, side)
    retargeter = _gripper_retargeter_for_side(components, side)
    input_channel_name = _gripper_input_channel_name(components, side)
    channel = None if channels is None else channels.get(input_channel_name)
    q6_status = _q6_channel_status(channel, side=side, now=now, channel_name=input_channel_name)
    q6_status["input_source"] = str(getattr(components, "gripper_input_source", "q6") or "q6")
    q6_status.update(
        _q6_retarget_status(
            components,
            retargeter,
            q6_values=q6_status.get("q6_values"),
        )
    )
    if command_status is not None:
        q6_status["q6_command"] = _json_safe_value(command_status.get(side) or {})
    if backend is None:
        return {
            "configured": False,
            **q6_status,
        }
    try:
        state = backend.get_state(include_tactile_arrays=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "is_connected": False,
            "telemetry_error": f"state_failed: {exc}",
            **q6_status,
        }
    return {
        "configured": True,
        "backend_type": state.get("backend_type"),
        "is_connected": bool(state.get("is_connected")),
        "connect_hardware": bool(state.get("connect_hardware")),
        "connect_tactile": bool(state.get("connect_tactile")),
        "send_commands": bool(state.get("send_commands")),
        "port": state.get("gripper_port"),
        "id": state.get("gripper_id"),
        "command_dim": state.get("gripper_command_dim"),
        "max_action": state.get("gripper_max_action"),
        "speed": state.get("gripper_speed"),
        "torque": state.get("gripper_torque"),
        "command_hz": state.get("gripper_command_hz"),
        "continuous": state.get("gripper_continuous"),
        "commanded": state.get("gripper_commanded_scalar"),
        "measured": state.get("gripper_measured_scalar"),
        "state": state.get("gripper_state_scalar"),
        "error_code": state.get("gripper_error_code"),
        "status_code": state.get("gripper_status_code"),
        "status_text": state.get("gripper_status_text"),
        "is_moving": state.get("gripper_is_moving"),
        "current": state.get("gripper_current"),
        "temperature_c": state.get("gripper_temperature_c"),
        "last_sent_at": state.get("last_sent_command_timestamp"),
        "joint_at": state.get("joint_timestamp"),
        "telemetry_at": state.get("telemetry_timestamp"),
        "telemetry_error": state.get("telemetry_error"),
        "gripper_read_elapsed_s": state.get("gripper_read_elapsed_s"),
        "gripper_read_skipped": state.get("gripper_read_skipped"),
        "gripper_read_skip_reason": state.get("gripper_read_skip_reason"),
        "gripper_read_forced": state.get("gripper_read_forced"),
        "gripper_telemetry_max_deferral_s": state.get(
            "gripper_telemetry_max_deferral_s"
        ),
        "tactile_read_elapsed_s": state.get("tactile_read_elapsed_s"),
        "tactile_gate_wait_s": state.get("tactile_gate_wait_s"),
        "tactile_timestamp": state.get("tactile_timestamp"),
        "telemetry_elapsed_s": state.get("telemetry_elapsed_s"),
        "tactile_fields": _json_safe_value(state.get("tactile_fields") or ()),
        "tactile_devices": _json_safe_value(state.get("tactile_devices") or ()),
        "tactile_frame_ids": _json_safe_value(state.get("tactile_frame_ids") or {}),
        "tactile_field_by_key": _json_safe_value(state.get("tactile_field_by_key") or {}),
        "tactile_shapes": _json_safe_value(state.get("tactile_shapes") or {}),
        "tactile_raw_shape_per_sensor": _json_safe_value(
            state.get("tactile_raw_shape_per_sensor") or {}
        ),
        "tactile_timings_ms": _json_safe_value(state.get("tactile_timings_ms") or {}),
        "tactile_read_hz": state.get("tactile_read_hz"),
        "tactile_read_skipped": state.get("tactile_read_skipped"),
        "tactile_process_read": state.get("tactile_process_read"),
        "tactile_worker_alive": state.get("tactile_worker_alive"),
        "tactile_worker_receiver_alive": state.get("tactile_worker_receiver_alive"),
        "tactile_worker_restart_pending": state.get("tactile_worker_restart_pending"),
        "tactile_worker_restart_count": state.get("tactile_worker_restart_count"),
        "tactile_worker_failure_count": state.get("tactile_worker_failure_count"),
        "tactile_worker_last_exit_code": state.get("tactile_worker_last_exit_code"),
        "tactile_process_nice": state.get("tactile_process_nice"),
        "tactile_process_cpu_affinity": _json_safe_value(
            state.get("tactile_process_cpu_affinity") or ()
        ),
        "tactile_process_queue_size": state.get("tactile_process_queue_size"),
        "tactile_max_fps": state.get("tactile_max_fps"),
        "tactile_warmup_frames": state.get("tactile_warmup_frames"),
        "tactile_denoise": state.get("tactile_denoise"),
        "tactile_store_flat_vector": state.get("tactile_store_flat_vector"),
        "tactile_global_read_lock": state.get("tactile_global_read_lock"),
        "tactile_ready": _json_safe_value(state.get("tactile_ready") or {}),
        "tactile_version": state.get("tactile_version"),
        **q6_status,
    }


def _build_gripper_status(
    components: RealTeleopComponents,
    *,
    channels: dict[str, object] | None = None,
    now: float | None = None,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    current_time = time.time() if now is None else float(now)
    return {
        side: _gripper_status_for_side(
            components,
            side,
            channels=channels,
            now=current_time,
            command_status=command_status,
        )
        for side in ("left", "right")
    }





def _hand_channel_status(
    components: RealTeleopComponents,
    channels: dict[str, object],
) -> tuple[str, ...]:
    statuses: list[str] = []
    for side in components.active_sides:
        channel = channels.get(_gripper_input_channel_name(components, side))
        if channel is None:
            continue
        valid = bool(getattr(channel, "valid", False))
        if valid:
            statuses.append(f"{side}=ok")
            continue
        metadata = getattr(channel, "metadata", {}) or {}
        read_error = str(metadata.get("read_error") or "invalid")
        statuses.append(f"{side}=invalid:{read_error}")
    return tuple(statuses)


def _gripper_status_signature(components: RealTeleopComponents) -> tuple[str, ...]:
    return _gripper_runtime_status_signature(components)


def _gripper_runtime_status_signature(
    components: RealTeleopComponents,
    *,
    channels: Mapping[str, object] | None = None,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
    now: float | None = None,
) -> tuple[str, ...]:
    statuses: list[str] = []
    current_time = time.time() if now is None else float(now)
    for side in components.active_sides:
        backend = _gripper_backend_for_side(components, side)
        retargeter = _gripper_retargeter_for_side(components, side)
        channel_name = _gripper_input_channel_name(components, side)
        channel = None if channels is None else channels.get(channel_name)
        q6_status = _q6_channel_status(channel, side=side, now=current_time, channel_name=channel_name)
        q6_status.update(
            _q6_retarget_status(
                components,
                retargeter,
                q6_values=q6_status.get("q6_values"),
            )
        )
        cmd_status = dict(command_status.get(side) or {}) if isinstance(command_status, Mapping) else {}
        cmd_text = str(cmd_status.get("status") or "-")
        sent_text = _compact_runtime_bool(cmd_status.get("sent", False))
        cmd_target = _compact_runtime_float(cmd_status.get("target"))
        cmd_source = _compact_runtime_float(cmd_status.get("source_value"))
        cmd_send_ms = _compact_runtime_float(cmd_status.get("send_elapsed_ms"), digits=1)
        cmd_skip = str(cmd_status.get("skip_reason") or "")
        q6_valid = "ok" if bool(q6_status.get("q6_valid")) else (
            "missing" if not bool(q6_status.get("q6_channel_present")) else "invalid"
        )
        q6_source_dim = q6_status.get("q6_source_dim")
        q6_source = _compact_runtime_float(q6_status.get("q6_source_value"))
        q6_target = _compact_runtime_float(q6_status.get("q6_target_preview"))
        q6_filtered = _compact_runtime_float(q6_status.get("q6_filtered_target"))
        q6_values = q6_status.get("q6_values")
        q6_values_text = "-"
        if isinstance(q6_values, Sequence) and not isinstance(q6_values, (str, bytes)):
            q6_values_text = "[" + ",".join(_compact_runtime_float(value, digits=2) for value in q6_values) + "]"
        q6_error = q6_status.get("q6_read_error") or q6_status.get("q6_value_error") or ""

        prefix = f"{side}=no_backend"
        measured_text = "-"
        commanded_text = "-"
        read_text = "-"
        joint_age_ms_text = "-"
        suffix = ""
        try:
            state = backend.get_state(include_tactile_arrays=False) if backend is not None else {}
        except Exception as exc:  # noqa: BLE001
            prefix = f"{side}=state_error:{exc}"
        else:
            if backend is not None:
                if cmd_send_ms == "-":
                    send_elapsed_s = _safe_float_value(state.get("gripper_send_elapsed_s"))
                    if send_elapsed_s is not None:
                        cmd_send_ms = _compact_runtime_float(float(send_elapsed_s) * 1000.0, digits=1)
                if not cmd_skip:
                    cmd_skip = str(state.get("gripper_last_send_skip_reason") or "")
                connected = "connected" if bool(state.get("is_connected")) else "disconnected"
                error = state.get("telemetry_error")
                measured = state.get("gripper_measured_scalar")
                commanded = state.get("gripper_commanded_scalar")
                error_code = state.get("gripper_error_code")
                status_code = state.get("gripper_status_code")
                status_text = str(state.get("gripper_status_text") or "").strip()
                measured_text = _compact_runtime_float(measured)
                commanded_text = _compact_runtime_float(commanded)
                if bool(state.get("gripper_read_forced")):
                    read_text = "forced"
                elif bool(state.get("gripper_read_skipped")):
                    read_text = str(state.get("gripper_read_skip_reason") or "skipped")
                else:
                    read_text = "read"
                joint_at = _safe_float_value(state.get("joint_timestamp"))
                if joint_at is not None:
                    joint_age_ms_text = _compact_runtime_float(
                        max(0.0, current_time - joint_at) * 1000.0,
                        digits=1,
                    )
                q6_mode = str(getattr(components, "gripper_input_source", "q6")) if retargeter is not None else "manual"
                hardware_fault = (
                    ""
                    if error_code in (None, 0)
                    else f" hardware_fault={error_code}"
                )
                hardware_status = ""
                if status_code is not None or status_text:
                    hardware_status = (
                        f" hw_status={status_code if status_code is not None else '-'}"
                        f"{':' + status_text if status_text else ''}"
                    )
                suffix = (
                    (f":{error}" if error else "")
                    + hardware_fault
                    + hardware_status
                )
                prefix = f"{side}={connected}/{q6_mode}/cmd={commanded_text}/meas={measured_text}{suffix}"
        statuses.append(
            f"{prefix} input={getattr(components, 'gripper_input_source', 'q6')} valid={q6_valid} dim={q6_source_dim} src={q6_source} "
            f"target={q6_target} filt={q6_filtered} vals={q6_values_text} "
            f"send={cmd_text}/{sent_text} send_src={cmd_source} send_target={cmd_target} "
            f"send_ms={cmd_send_ms} read={read_text} joint_age_ms={joint_age_ms_text}"
            + (f" skip={cmd_skip}" if cmd_skip else "")
            + (f" q6_error={q6_error}" if q6_error else "")
        )
    return tuple(statuses)


def _gripper_runtime_health_signature(statuses: Sequence[str]) -> tuple[str, ...]:
    """Return only the stable health fields used to decide whether to print status.

    The full status intentionally contains live q6, target, measured, and send-time
    values.  Including those values in the change gate turns ordinary hand motion
    into a terminal log on every control frame.
    """

    health_statuses: list[str] = []
    for status in statuses:
        text = str(status)
        prefix, _, remainder = text.partition(" input=")
        route = "/".join(prefix.split("/")[:2])
        valid = "unknown"
        for token in remainder.split():
            if token.startswith("valid="):
                valid = token.removeprefix("valid=")
                break
        lower_text = text.lower()
        unhealthy = any(
            marker in lower_text
            for marker in (
                "disconnected",
                "no_backend",
                "state_error",
                "invalid",
                "error",
                "failed",
                "hardware_fault=",
            )
        )
        health_statuses.append(f"{route} valid={valid} unhealthy={int(unhealthy)}")
    return tuple(health_statuses)


_DEBUG_TIMING_RED = "\033[1;31;40m"
_DEBUG_TIMING_RESET = "\033[0m"









__all__ = [
    "_build_gripper_status",
    "_gripper_runtime_health_signature",
    "_gripper_runtime_status_signature",
    "_gripper_status_for_side",
    "_gripper_status_signature",
    "_hand_channel_status",
    "_q6_channel_status",
    "_q6_retarget_status",
]
