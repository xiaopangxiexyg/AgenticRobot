"""Small helpers for presenting bounded backend/Rig readiness reports."""

from __future__ import annotations

from collections.abc import Mapping


def readiness_report_blockers(report: Mapping[str, object] | object) -> dict[str, str]:
    """Return ``effector -> reason`` entries for every unready backend."""

    if not isinstance(report, Mapping):
        return {}
    raw_backends = report.get("backends", report.get("effectors", {}))
    if not isinstance(raw_backends, Mapping):
        return {}
    blockers: dict[str, str] = {}
    for effector, raw in raw_backends.items():
        if not isinstance(raw, Mapping) or bool(raw.get("ready", False)):
            continue
        reason = str(raw.get("reason", "") or "").strip()
        details = raw.get("details")
        if not reason and isinstance(details, Mapping):
            error_code = details.get("gripper_error_code")
            if error_code not in (None, 0):
                reason = f"gripper_error_code={error_code}"
            elif details.get("telemetry_error"):
                reason = f"telemetry_error={details['telemetry_error']}"
            elif details.get("hardware_ready") is False:
                reason = "hardware_ready=false"
        blockers[str(effector)] = reason or "backend is not ready"
    return blockers


def format_readiness_report(report: Mapping[str, object] | object) -> str:
    """Format a readiness report for logs and operator-facing errors."""

    blockers = readiness_report_blockers(report)
    if blockers:
        return "; ".join(f"{name} {reason}" for name, reason in blockers.items())
    if isinstance(report, Mapping):
        reason = str(report.get("reason", "") or "").strip()
        if reason:
            return reason
    return "rig is not ready"


__all__ = ["format_readiness_report", "readiness_report_blockers"]
