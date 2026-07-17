"""Observational timing helpers for the real teleoperation runtime.

The functions here construct metadata and emit audit events only.  They do not
start devices, dispatch commands, or alter the control decision.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from robot_motion.position_control import PositionAuditFanout, PositionTimingAuditEvent

from ..collection.diagnostics import PhaseTimer


@dataclass
class RealDebugTimingState:
    enabled: bool = False
    threshold_sec: float = 0.025
    report_interval_sec: float = 1.0
    last_report_at: float = 0.0
    suppressed_reports: int = 0


@dataclass(frozen=True)
class DebugTimingModules:
    legacy_all: bool = False
    console: bool = False
    light_log: bool = False
    rm65_backend: bool = False
    ik_profile: bool = False
    watchdog: bool = False

    @property
    def enabled(self) -> bool:
        return bool(
            self.console
            or self.light_log
            or self.rm65_backend
            or self.ik_profile
            or self.watchdog
        )

    @property
    def needs_loop_phase_timer(self) -> bool:
        return bool(self.console or self.light_log)

    def enabled_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.console:
            names.append("console")
        if self.light_log:
            names.append("log")
        if self.rm65_backend:
            names.append("rm65")
        if self.ik_profile:
            names.append("ik")
        if self.watchdog:
            names.append("watchdog")
        return tuple(names)


def debug_timing_modules(args: argparse.Namespace) -> DebugTimingModules:
    legacy_all = bool(getattr(args, "debug_timing", False))
    timing_rm65 = bool(getattr(args, "debug_timing_rm65", False))
    timing_ik = bool(getattr(args, "debug_timing_ik", False))
    auto_log = bool(getattr(args, "debug_timing_auto_log", True))
    disk_write_disabled = bool(getattr(args, "no_debug_disk_write", False))
    auto_light_log = auto_log and (timing_rm65 or timing_ik)
    return DebugTimingModules(
        legacy_all=legacy_all,
        console=legacy_all or bool(getattr(args, "debug_timing_console", False)),
        light_log=(
            not disk_write_disabled
            and (
                legacy_all
                or bool(getattr(args, "debug_timing_log", False))
                or auto_light_log
            )
        ),
        rm65_backend=legacy_all or timing_rm65,
        ik_profile=legacy_all
        or timing_ik
        or bool(getattr(args, "debug_ik_timing", False)),
        watchdog=legacy_all or bool(getattr(args, "debug_timing_watchdog", False)),
    )


def debug_timing_metadata(args: argparse.Namespace) -> dict[str, object]:
    modules = debug_timing_modules(args)
    control_debug = getattr(args, "control_debug_config", None)
    return {
        "debug_timing": bool(modules.enabled),
        "debug_timing_legacy_all": bool(modules.legacy_all),
        "debug_timing_modules": list(modules.enabled_names()),
        "debug_timing_console": bool(modules.console),
        "debug_timing_light_log": bool(modules.light_log),
        "rm65_backend_debug_timing": bool(modules.rm65_backend),
        "debug_timing_ik_profile": bool(modules.ik_profile),
        "debug_timing_watchdog": bool(modules.watchdog),
        "debug_disk_write_disabled": bool(
            getattr(args, "no_debug_disk_write", False)
        ),
        "debug_timing_threshold_sec": float(getattr(args, "debug_timing_threshold_sec", 0.025)),
        "debug_timing_report_interval_sec": float(
            getattr(args, "debug_timing_report_interval_sec", 1.0)
        ),
        "debug_timing_log_sample_hz": float(
            getattr(args, "debug_timing_log_sample_hz", 0.0)
        ),
        "debug_action_sample_hz": float(
            getattr(args, "debug_action_sample_hz", 0.0)
        ),
        "control_debug": (
            {
                **control_debug.to_metadata(),
                **dict(getattr(args, "control_debug_runtime_metadata", {}) or {}),
            }
            if hasattr(control_debug, "to_metadata")
            else {
                "level": str(getattr(args, "control_debug", "off")),
                "enabled": str(getattr(args, "control_debug", "off")) != "off",
                "recording_requested": str(getattr(args, "control_debug", "off"))
                != "off",
                "observational_only": True,
            }
        ),
    }


def merge_control_debug_runtime_metadata(
    args: argparse.Namespace,
    **updates: object,
) -> dict[str, object]:
    """Atomically replace the runtime audit-status snapshot stored on ``args``."""

    current = dict(getattr(args, "control_debug_runtime_metadata", {}) or {})
    current.update(updates)
    setattr(args, "control_debug_runtime_metadata", current)
    return current


def record_position_loop_timings(
    sink: PositionAuditFanout,
    *,
    timer: PhaseTimer,
    step_index: int,
    state: object,
    outer_control_hz: float,
) -> None:
    """Emit full stage timing rows; the async sink computes summary percentiles."""

    timestamp_wall = time.time()
    timestamp_monotonic = time.monotonic()
    metadata: dict[str, Any] = {
        "step_index": int(step_index),
        "teleop_state": str(getattr(state, "value", state)),
        "outer_control_hz": float(outer_control_hz),
    }
    for name, elapsed_sec in timer.phases:
        sink(
            PositionTimingAuditEvent(
                stage=str(name),
                elapsed_ms=max(0.0, float(elapsed_sec)) * 1000.0,
                timestamp_wall=timestamp_wall,
                timestamp_monotonic=timestamp_monotonic,
                metadata=metadata,
            )
        )
    total_ms = max(0.0, timer.elapsed_sec()) * 1000.0
    deadline_ms = 1000.0 / max(1e-9, float(outer_control_hz))
    sink(
        PositionTimingAuditEvent(
            stage="main_loop_total",
            elapsed_ms=total_ms,
            timestamp_wall=timestamp_wall,
            timestamp_monotonic=timestamp_monotonic,
            deadline_ms=deadline_ms,
            deadline_missed=bool(total_ms > deadline_ms),
            metadata=metadata,
        )
    )


__all__ = [
    "DebugTimingModules",
    "RealDebugTimingState",
    "debug_timing_metadata",
    "debug_timing_modules",
    "merge_control_debug_runtime_metadata",
    "record_position_loop_timings",
]
