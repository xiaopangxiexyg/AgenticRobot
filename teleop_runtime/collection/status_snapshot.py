from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def compact_log_value(value: object, *, max_chars: int = 90) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def writer_debug_text(writer_pool: Any) -> str:
    try:
        stats = writer_pool.stats()
    except Exception as exc:  # noqa: BLE001
        return f"writer=status_error:{type(exc).__name__}:{compact_log_value(exc, max_chars=80)}"
    return (
        "writer="
        f"queued:{stats.queued} submitted:{stats.submitted} completed:{stats.completed} "
        f"dropped:{stats.dropped} failures:{stats.failures} workers:{stats.worker_count}"
    )


def camera_debug_text(
    camera_sources: Mapping[str, Any],
    *,
    camera_previewer: Any | None = None,
) -> str:
    if not camera_sources:
        return "camera=off"
    parts: list[str] = []
    if camera_previewer is not None:
        disabled_reason = camera_previewer.disabled_reason
        parts.append(
            "preview=active"
            if not disabled_reason
            else f"preview=disabled:{compact_log_value(disabled_reason, max_chars=80)}"
        )
    for camera_key, source in sorted(camera_sources.items()):
        try:
            info = source.device_info()
        except Exception as exc:  # noqa: BLE001
            parts.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        age = info.get("last_frame_age_sec")
        age_text = "none" if age is None else f"{float(age):.3f}s"
        last_error = info.get("last_error")
        error_text = "ok" if not last_error else compact_log_value(last_error, max_chars=70)
        parts.append(
            f"{camera_key}=frames:{frame_count},age:{age_text},"
            f"timeouts:{int(info.get('timeout_count', 0) or 0)},"
            f"consecutive:{int(info.get('consecutive_timeout_count', 0) or 0)},"
            f"restarts:{int(info.get('restart_count', 0) or 0)},err:{error_text}"
        )
    return "camera=" + "; ".join(parts)


def camera_status_text(
    camera_sources: Mapping[str, Any],
    *,
    camera_previewer: Any | None = None,
) -> str:
    if not camera_sources:
        return " camera=off"
    bits: list[str] = []
    if camera_previewer is not None and camera_previewer.disabled_reason:
        bits.append(
            "preview=disabled:"
            + compact_log_value(camera_previewer.disabled_reason, max_chars=80)
        )
    for camera_key, source in sorted(camera_sources.items()):
        try:
            info = source.device_info()
        except Exception as exc:  # noqa: BLE001
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(
                f"{camera_key}=frames:{frame_count},err:"
                f"{compact_log_value(last_error, max_chars=70)}"
            )
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "" if not bits else f" camera={';'.join(bits)}"


def camera_state_signature_text(
    camera_sources: Mapping[str, Any],
    *,
    camera_previewer: Any | None = None,
) -> tuple[tuple[str, ...], str]:
    if not camera_sources:
        return (), "camera=off"
    signature_parts: list[str] = []
    text_parts: list[str] = []
    if camera_previewer is not None:
        disabled_reason = camera_previewer.disabled_reason
        if disabled_reason:
            reason_text = compact_log_value(disabled_reason, max_chars=80)
            signature_parts.append(f"preview:disabled:{reason_text}")
            text_parts.append(f"preview=disabled:{reason_text}")
        else:
            signature_parts.append("preview:active")
            text_parts.append("preview=active")
    for camera_key, source in sorted(camera_sources.items()):
        try:
            info = source.device_info()
        except Exception as exc:  # noqa: BLE001
            state = f"status_error:{type(exc).__name__}"
            signature_parts.append(f"{camera_key}:{state}")
            text_parts.append(f"{camera_key}={state}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            error_text = compact_log_value(last_error, max_chars=70)
            signature_parts.append(f"{camera_key}:error:{error_text}")
            text_parts.append(f"{camera_key}=error:{error_text}")
        elif frame_count > 0:
            signature_parts.append(f"{camera_key}:streaming")
            text_parts.append(f"{camera_key}=streaming")
        else:
            signature_parts.append(f"{camera_key}:waiting")
            text_parts.append(f"{camera_key}=waiting")
    return tuple(signature_parts), "camera=" + ";".join(text_parts)


def hand_debug_text(
    active_sides: Sequence[str],
    backend_for_side: Callable[[str], Any | None],
    *,
    now: float | None = None,
) -> str:
    parts: list[str] = []
    current_time = time.time() if now is None else float(now)
    for side in active_sides:
        backend = backend_for_side(str(side))
        if backend is None:
            continue
        try:
            state = backend.get_state()
        except Exception as exc:  # noqa: BLE001
            parts.append(f"{side}=state_error:{type(exc).__name__}")
            continue
        telemetry_at = state.get("telemetry_timestamp")
        tactile_at = state.get("tactile_timestamp")
        telemetry_age = (
            "none" if telemetry_at is None else f"{current_time - float(telemetry_at):.3f}s"
        )
        tactile_age = "none" if tactile_at is None else f"{current_time - float(tactile_at):.3f}s"
        error = state.get("telemetry_error")
        error_text = "ok" if not error else compact_log_value(error, max_chars=70)
        parts.append(
            f"{side}=connected:{bool(state.get('is_connected'))},"
            f"telemetry_age:{telemetry_age},tactile_age:{tactile_age},"
            f"reconnects:{int(state.get('reconnect_count', 0) or 0)},err:{error_text}"
        )
    return "hand=none" if not parts else "hand=" + "; ".join(parts)


def arm_debug_text(
    active_sides: Sequence[str],
    backend_for_side: Callable[[str], Any | None],
    *,
    now: float | None = None,
) -> str:
    parts: list[str] = []
    current_time = time.time() if now is None else float(now)
    for side in active_sides:
        backend = backend_for_side(str(side))
        if backend is None:
            continue
        try:
            state = backend.get_state()
        except Exception as exc:  # noqa: BLE001
            parts.append(f"{side}=state_error:{type(exc).__name__}")
            continue
        refresh_at = state.get("hardware_refresh_time")
        refresh_age = "none" if refresh_at is None else f"{current_time - float(refresh_at):.3f}s"
        code = state.get("hardware_result_code")
        code_text = "none" if code is None else str(code)
        result_text = compact_log_value(state.get("hardware_result_text", ""), max_chars=50)
        ik_meta = state.get("ik_metadata")
        ik_text = ""
        if isinstance(ik_meta, Mapping) and ik_meta:
            ik_keys = (
                "collision_avoidance_status",
                "collision_avoidance_fallback_reason",
                "collision_distance_m",
                "ik_branch_selection",
                "ik_jump_gate_status",
                "ik_jump_gate_reason",
                "ik_jump_gate_hold_frames",
            )
            ik_parts = [
                f"{key}:{compact_log_value(ik_meta[key], max_chars=40)}"
                for key in ik_keys
                if key in ik_meta and ik_meta[key] is not None
            ]
            if ik_parts:
                ik_text = ",ik:{" + ",".join(ik_parts) + "}"
        parts.append(
            f"{side}=refresh_age:{refresh_age},code:{code_text},text:{result_text},"
            f"dt:{float(state.get('effective_control_dt_sec', 0.0) or 0.0):.4f},"
            f"scale:{float(state.get('effective_step_scale', 0.0) or 0.0):.3f}"
            f"{ik_text}"
        )
    return "arm=none" if not parts else "arm=" + "; ".join(parts)


def fault_signature(faults: Sequence[Any]) -> tuple[str, ...]:
    parts: list[str] = []
    for fault in faults:
        code = str(getattr(fault, "code", fault))
        channel_name = getattr(fault, "channel_name", None)
        parts.append(f"{code}:{channel_name}" if channel_name else code)
    return tuple(sorted(parts))


def fault_status_text(faults: Sequence[Any]) -> str:
    bits: list[str] = []
    for fault in faults:
        code = str(getattr(fault, "code", fault))
        channel_name = getattr(fault, "channel_name", None)
        message = str(getattr(fault, "message", "") or "")
        if channel_name:
            code = f"{code}:{channel_name}"
        bits.append(code if not message else f"{code}({compact_log_value(message, max_chars=70)})")
    return "; ".join(bits)
