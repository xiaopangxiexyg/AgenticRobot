"""Pure debug-timing payload projections for already-read snapshots.

This module does not read backends, inspect cameras, publish to a UI, or
dispatch commands.  Serializer helpers are imported directly from their
owning projection modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..collection.diagnostics import build_end_to_end_latency_breakdown
from .status_payloads import _float_or_none
from .debug_payload_primitives import (
    _compact_debug_value,
    _pose_compact_debug,
)


def _fault_signature(faults: Sequence[object]) -> tuple[str, ...]:
    labels: list[str] = []
    for fault in faults:
        code = str(getattr(fault, "code", fault) or "")
        channel_name = str(getattr(fault, "channel_name", "") or "")
        labels.append(f"{code}:{channel_name}" if channel_name else code)
    return tuple(labels)

def _phase_timings_log_payload(phases: Sequence[tuple[str, float]]) -> dict[str, object]:
    items = [
        {"name": str(name), "elapsed_ms": float(elapsed) * 1000.0}
        for name, elapsed in phases
    ]
    top = sorted(items, key=lambda item: float(item["elapsed_ms"]), reverse=True)[:16]
    return {
        "total_marked_ms": sum(float(item["elapsed_ms"]) for item in items),
        "items": items,
        "top": top,
    }

def _latency_items_log_payload(items: Sequence[object]) -> list[dict[str, object]]:
    return [
        {"name": str(getattr(item, "name", "")), "age_ms": float(getattr(item, "age_sec", 0.0)) * 1000.0}
        for item in sorted(items, key=lambda item: float(getattr(item, "age_sec", 0.0)), reverse=True)
    ]

def _latency_log_payload(result: object, *, now: float) -> dict[str, object]:
    breakdown = build_end_to_end_latency_breakdown(result, reference_time=float(now))
    return {
        "reference_time": float(breakdown.reference_time),
        "step_age_ms": None if breakdown.step_age_sec is None else float(breakdown.step_age_sec) * 1000.0,
        "observation_age_ms": (
            None
            if breakdown.observation_age_sec is None
            else float(breakdown.observation_age_sec) * 1000.0
        ),
        "command_age_ms": (
            None if breakdown.command_age_sec is None else float(breakdown.command_age_sec) * 1000.0
        ),
        "channel_age": _latency_items_log_payload(breakdown.channel_age_sec),
        "receiver_age": _latency_items_log_payload(breakdown.receiver_age_sec),
        "sender_age": _latency_items_log_payload(breakdown.sender_age_sec),
    }

def _faults_log_payload(result: object) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for fault in getattr(result, "faults", ()) or ():
        severity = getattr(fault, "severity", None)
        payload.append(
            {
                "code": str(getattr(fault, "code", fault) or ""),
                "channel_name": getattr(fault, "channel_name", None),
                "component": getattr(fault, "component", None),
                "severity": str(getattr(severity, "value", severity) or ""),
                "message": str(getattr(fault, "message", "") or ""),
                "timestamp": _float_or_none(getattr(fault, "timestamp", None)),
                "metadata": _compact_debug_value(
                    getattr(fault, "metadata", {}) or {},
                    max_items=12,
                ),
            }
        )
    return payload

def _channels_log_payload(result: object, *, now: float) -> dict[str, object]:
    channels = getattr(result, "channels", {}) or {}
    if not isinstance(channels, Mapping):
        return {}
    metadata_keys = (
        "sender_timestamp",
        "receiver_timestamp",
        "tracker_name",
        "tracker_role",
        "seq",
        "port",
        "baudrate",
        "stale_q6_frames_dropped",
        "reader_error_count",
        "reader_recovery_count",
        "read_error",
        "error",
    )
    out: dict[str, object] = {}
    for channel_name, channel in sorted(channels.items(), key=lambda item: str(item[0])):
        timestamp = _float_or_none(getattr(channel, "timestamp", None))
        metadata = getattr(channel, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        capability = getattr(channel, "capability", None)
        channel_payload: dict[str, object] = {
            "valid": bool(getattr(channel, "valid", False)),
            "critical": bool(getattr(channel, "critical", False)),
            "capability": capability,
            "side": getattr(channel, "side", None),
            "frame_name": getattr(channel, "frame_name", None),
            "source_name": getattr(channel, "source_name", None),
            "timestamp": timestamp,
            "age_ms": None if timestamp is None else (float(now) - timestamp) * 1000.0,
            "metadata": {
                key: _compact_debug_value(metadata.get(key), max_items=8)
                for key in metadata_keys
                if key in metadata
            },
        }
        if capability == "pose6d":
            channel_payload["pose_stage"] = "observation_channel_before_retarget"
            raw_pose = getattr(channel, "value", None)
            pose_payload = _pose_compact_debug(raw_pose)
            try:
                pose_matrix = np.asarray(raw_pose, dtype=float)
            except (TypeError, ValueError):
                pose_matrix = np.empty((0, 0), dtype=float)
            if (
                pose_payload is not None
                and pose_matrix.shape == (4, 4)
                and np.all(np.isfinite(pose_matrix))
            ):
                # Preserve exact SO(3) evidence; RPY is display-only and can
                # become ambiguous near gimbal lock.
                pose_payload["rotation_matrix"] = [
                    [float(value) for value in row]
                    for row in pose_matrix[:3, :3].tolist()
                ]
            channel_payload["pose"] = pose_payload
            motion_window = metadata.get("tracker_motion_window")
            if isinstance(motion_window, Mapping):
                channel_payload["motion_window"] = _compact_debug_value(
                    dict(motion_window),
                    max_items=24,
                )
            motion_keys = (
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
                "velocity_frame_name",
                "velocity_source",
                "velocity_valid",
                "velocity_error",
            )
            channel_payload["motion"] = {
                key: _compact_debug_value(metadata.get(key), max_items=8)
                for key in motion_keys
                if key in metadata
            }
        out[str(channel_name)] = channel_payload
    return out

__all__ = [
    "_fault_signature",
    "_phase_timings_log_payload",
    "_latency_items_log_payload",
    "_latency_log_payload",
    "_faults_log_payload",
    "_channels_log_payload",
]
