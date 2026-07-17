"""Pure debug-timing payload projections for already-read snapshots.

This module does not read backends, inspect cameras, publish to a UI, or
dispatch commands.  The legacy entrypoint binds the two serializer helpers
dynamically so existing monkeypatch points remain live.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect

from ..collection.diagnostics import build_end_to_end_latency_breakdown
from ..gui.status_payloads import _float_or_none as _default_float_or_none
from .debug_payload_primitives import _compact_debug_value as _default_compact_debug_value


_float_or_none = _default_float_or_none
_compact_debug_value = _default_compact_debug_value
_SUPPORT_NAMES = ("_compact_debug_value", "_float_or_none")


class _LegacyFunctionProxy:
    """Resolve legacy serializer helpers dynamically."""

    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(
                f"recursive debug-timing payload support proxy: {self._name}"
            )
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    """Bind legacy compact serializer patch points."""

    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "debug-timing payload support is incomplete: "
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
        out[str(channel_name)] = {
            "valid": bool(getattr(channel, "valid", False)),
            "critical": bool(getattr(channel, "critical", False)),
            "capability": getattr(channel, "capability", None),
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
    return out

__all__ = [
    "bind_legacy_support",
    "_fault_signature",
    "_phase_timings_log_payload",
    "_latency_items_log_payload",
    "_latency_log_payload",
    "_faults_log_payload",
    "_channels_log_payload",
]

