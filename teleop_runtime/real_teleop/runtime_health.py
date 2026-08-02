"""Read-only backend and safety health signatures for real teleoperation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
)

def _short_status_text(value: object, *, max_len: int = 160) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _command_backend_status_text(state: dict[str, object]) -> str:
    if not state:
        return "none"
    if "impedance_enabled" in state:
        bits = [
            f"imp={int(bool(state.get('impedance_enabled')))}",
            f"loop={state.get('impedance_loop_state')}",
            f"send={state.get('send_result_text')}",
        ]
        ik_metadata = state.get("ik_metadata")
        if isinstance(ik_metadata, Mapping):
            ik_status = ik_metadata.get("status") or ik_metadata.get("ik_status")
            if ik_status:
                bits.append(f"ik={_short_status_text(ik_status, max_len=48)}")
        fault = state.get("fault")
        if fault:
            bits.append(f"fault={_short_status_text(fault)}")
        return ",".join(bits)
    code = state.get("send_result_code")
    text = state.get("send_result_text")
    bits = [
        f"send={code}:{_short_status_text(text)}" if text is not None else f"send={code}"
    ]
    stream = state.get("background_streaming")
    if isinstance(stream, Mapping):
        planner = str(stream.get("position_command_planner") or "")
        if planner in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }:
            mode = str(stream.get("trajectory_policy_mode") or "unknown")
            scale = float(stream.get("trajectory_policy_speed_scale", 1.0) or 0.0)
            bits.append(f"traj={mode}/{scale:.2f}")
            last_debug = stream.get("last_trajectory_debug")
            if isinstance(last_debug, Mapping):
                for key, label in (
                    ("trajectory_qd_command_rad_s", "qd"),
                    ("trajectory_qdd_command_rad_s2", "qdd"),
                ):
                    try:
                        values = np.asarray(last_debug.get(key), dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if values.size and np.all(np.isfinite(values)):
                        bits.append(f"{label}max={float(np.max(np.abs(values))):.2f}")
            if bool(stream.get("quick_stop_active", False)):
                bits.append(
                    "hold="
                    + _short_status_text(stream.get("quick_stop_reason"), max_len=64)
                )
    return ",".join(bits)


def _command_backend_health_signature(state: Mapping[str, object] | None) -> tuple[object, ...]:
    """Keep high-rate command diagnostics out of the status change gate."""

    if not state:
        return ("none",)
    if "impedance_enabled" in state:
        ik_metadata = state.get("ik_metadata")
        ik_status = None
        if isinstance(ik_metadata, Mapping):
            ik_status = ik_metadata.get("status") or ik_metadata.get("ik_status")
        return (
            "impedance",
            bool(state.get("impedance_enabled")),
            str(state.get("impedance_loop_state") or ""),
            state.get("send_result_code"),
            str(ik_status or ""),
            bool(state.get("fault")),
        )
    stream = state.get("background_streaming")
    if not isinstance(stream, Mapping):
        stream = {}
    return (
        "position",
        state.get("send_result_code"),
        str(stream.get("position_command_planner") or ""),
        str(stream.get("trajectory_policy_mode") or ""),
        bool(stream.get("quick_stop_active", False)),
        str(stream.get("quick_stop_reason") or ""),
    )


def _temporary_safety_health_signature(status: object) -> tuple[object, ...] | None:
    if not isinstance(status, Mapping):
        return None
    effectors = status.get("effectors")
    effector_statuses: list[tuple[object, ...]] = []
    if isinstance(effectors, Mapping):
        for name, value in sorted(effectors.items(), key=lambda item: str(item[0])):
            details = value if isinstance(value, Mapping) else {}
            effector_statuses.append(
                (
                    str(name),
                    str(details.get("command_type") or ""),
                    bool(details.get("limited", False)),
                    bool(details.get("initialized", False)),
                    str(details.get("reason") or ""),
                )
            )
    return (
        bool(status.get("enabled", False)),
        bool(status.get("limited", False)),
        tuple(effector_statuses),
    )



__all__ = [
    "_command_backend_health_signature",
    "_command_backend_status_text",
    "_short_status_text",
    "_temporary_safety_health_signature",
]
