"""Foreground coordination for RM65 free-drive to B601 MuJoCo.

The bridge and the existing B601 position driver own all source admission,
retargeting, physics, and target safety.  This module only coordinates one
foreground operator loop with the existing passive MuJoCo viewer.  It never
opens CAN, sends an RM65 command, or writes MuJoCo state directly.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from .rm65_b601_free_drive import (
    RM65FreeDriveAdmissionError,
    RM65ToB601FreeDriveBridge,
)


def _print_status(message: str) -> None:
    print(message, flush=True)


def _default_terminal_commands() -> Callable[[float], str | None]:
    """Reuse the existing immediate-key reader without importing it eagerly."""

    from .rebot_b601_rm65_foreground import B601RM65TerminalCommands

    return B601RM65TerminalCommands()


def _record_event(
    record: Callable[[Mapping[str, object]], None] | None,
    *,
    clock: Callable[[], float],
    bridge: RM65ToB601FreeDriveBridge,
    name: str,
    **details: object,
) -> None:
    if record is not None:
        record(
            {
                "event": name,
                "at_monotonic_sec": float(clock()),
                "bridge_state": bridge.state,
                **details,
            }
        )


def _instructions(emit: Callable[[str], None]) -> None:
    emit(
        "[RM65→B601] Keys (no Enter): a=activate/retry c=clutch r=recenter s=stop q=quit ?=help"
    )
    emit(
        "[RM65→B601] Start inactive; manually settle/free-drive RM65 first, then press a."
    )
    emit(
        "[RM65→B601] A rejected/stale source or IK result stays held; repair source and press a again."
    )
    emit(
        "[RM65→B601] Ctrl-C/EOF: stop following and close MuJoCo; no RM65/B601 hardware command is sent."
    )


def run_rm65_to_b601_foreground(
    bridge: RM65ToB601FreeDriveBridge,
    *,
    viewer: Any,
    rate_hz: float,
    viewer_sync_hz: float | None = None,
    poll_command: Callable[[float], str | None] | None = None,
    emit: Callable[[str], None] = _print_status,
    record: Callable[[Mapping[str, object]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Run the RM65 observation-only reverse-following loop in the foreground.

    ``bridge.step()`` is called at the requested MuJoCo target rate even while
    inactive or clutched, so the existing position driver's physics clock and
    viewer remain live.  RM65 observations are still sampled only at the
    bridge's configured source rate.  The injected ``poll_command`` is
    non-blocking apart from its bounded timeout; a real terminal therefore
    uses the existing immediate-key reader and never requires Enter.
    """

    if not isinstance(bridge, RM65ToB601FreeDriveBridge):
        raise TypeError("bridge must be RM65ToB601FreeDriveBridge.")
    for name in ("start", "sync", "status", "close"):
        if not callable(getattr(viewer, name, None)):
            raise TypeError(f"viewer must provide {name}().")
    if isinstance(rate_hz, bool) or not math.isfinite(float(rate_hz)) or float(rate_hz) <= 0.0:
        raise ValueError("rate_hz must be finite and positive.")
    if viewer_sync_hz is not None and (
        isinstance(viewer_sync_hz, bool)
        or not math.isfinite(float(viewer_sync_hz))
        or float(viewer_sync_hz) <= 0.0
    ):
        raise ValueError("viewer_sync_hz must be finite and positive when provided.")
    if poll_command is not None and not callable(poll_command):
        raise TypeError("poll_command must be callable or None.")
    if not callable(emit) or not callable(clock) or not callable(sleeper):
        raise TypeError("emit, clock, and sleeper must be callable.")
    if record is not None and not callable(record):
        raise TypeError("record must be callable or None.")

    period = 1.0 / float(rate_hz)
    viewer_period = (
        None if viewer_sync_hz is None else 1.0 / float(viewer_sync_hz)
    )
    owns_terminal_reader = poll_command is None
    if poll_command is None:
        poll_command = _default_terminal_commands()

    def event(name: str, **details: object) -> None:
        _record_event(
            record,
            clock=clock,
            bridge=bridge,
            name=name,
            **details,
        )

    def handle_command(command: str) -> bool:
        normalized = str(command).strip().lower()
        if normalized in {"", "?"}:
            _instructions(emit)
            event("operator_help")
            return True
        if normalized == "q":
            event("operator_quit")
            return False
        allowed = {
            "a": {"inactive", "held"},
            "c": {"active", "clutched"},
            "r": {"clutched"},
            "s": {"inactive", "active", "clutched", "held"},
        }
        if normalized not in allowed:
            emit("[RM65→B601] unknown command; ? for help")
            event("operator_command_rejected", command=normalized)
            return True
        if bridge.state not in allowed[normalized]:
            emit(f"[RM65→B601] {normalized} rejected in {bridge.state}")
            event("operator_command_rejected", command=normalized)
            return True
        transition = {
            "a": bridge.activate,
            "c": bridge.engage_clutch,
            "r": bridge.release_clutch,
            "s": bridge.deactivate,
        }[normalized]
        try:
            transition()
        except RM65FreeDriveAdmissionError as error:
            emit(f"[RM65→B601] held: {error}; repair source and press a")
            event(
                "admission_rejected",
                command=normalized,
                error_type=type(error).__name__,
                error=str(error),
            )
            return True
        emit(f"[RM65→B601] {bridge.state}")
        event("operator_command", command=normalized)
        return True

    try:
        bridge.deactivate()
        viewer.start()
        _instructions(emit)
        event("foreground_started")
        next_tick = float(clock()) + period
        next_viewer_sync = float(clock())
        last_hold_reason: str | None = None
        last_hold_message_at = float("-inf")
        while True:
            now = float(clock())
            if viewer_period is None or now + 1e-12 >= next_viewer_sync:
                if not bool(viewer.sync()):
                    status = viewer.status()
                    emit(f"[RM65→B601] viewer stopped: {status}")
                    event("viewer_stopped", viewer_status=status)
                    return
                next_viewer_sync = now if viewer_period is None else now + viewer_period
            timeout = max(0.0, next_tick - float(clock()))
            command = poll_command(timeout)
            if command is not None and not handle_command(command):
                return
            now = float(clock())
            if now < next_tick:
                sleeper(max(0.0, next_tick - now))
                continue
            tick = bridge.step()
            position_tick = tick.position_tick
            retarget_result = tick.retarget_result
            ik_result = None if retarget_result is None else retarget_result.ik_result
            retarget_diagnostics = (
                {}
                if retarget_result is None
                else dict(retarget_result.diagnostics)
            )
            rejected_source = (
                tick.source_step is not None
                and tick.source_step.sample is not None
                and retarget_result is not None
                and not retarget_result.accepted
            )
            source_debug = {}
            if tick.source_step is not None:
                source_state = tick.source_step.state
                source_debug = {
                    # These fields are present only on the low-rate source
                    # event, not on every 1 kHz physics tick.  They make a
                    # real dropout distinguishable from a valid sample that
                    # later failed IK.
                    "source_step_rejection_reason": tick.source_step.rejection_reason,
                    "source_state_measurement_sequence": source_state.get(
                        "hardware_measurement_sequence"
                    ),
                    "source_state_measurement_valid": source_state.get(
                        "hardware_measurement_valid"
                    ),
                    "source_state_read_success": source_state.get(
                        "hardware_read_success"
                    ),
                    "source_state_measurement_timestamp": source_state.get(
                        "hardware_measurement_timestamp"
                    ),
                    "source_state_read_timestamp_monotonic": source_state.get(
                        "hardware_read_timestamp_monotonic"
                    ),
                    "source_state_read_phase_timing": source_state.get(
                        "hardware_read_phase_timing"
                    ),
                    "source_state_background_streaming": source_state.get(
                        "background_streaming"
                    ),
                }
                if rejected_source:
                    assert tick.source_step.sample is not None
                    source_sample = tick.source_step.sample
                    source_debug.update(
                        {
                            "debug_source_sequence": source_sample.sequence,
                            "debug_source_qpos_rad": source_sample.qpos_rad,
                            "debug_source_pose": (
                                source_sample.pose_source_reference_source_controlled
                            ),
                        }
                    )
            position_debug = {}
            if tick.source_step is not None or position_tick.rejection_reason is not None:
                position_debug = {
                    "sim_measured_qpos_rad": position_tick.measured_qpos,
                    "sim_measured_qvel_rad_s": position_tick.measured_qvel,
                    "sim_target_qpos_rad": position_tick.target_qpos,
                    "sim_actuator_target_qpos_rad": position_tick.actuator_target_qpos,
                    "sim_slew_limited": position_tick.slew_limited,
                    "sim_slew_limit_ratio": position_tick.slew_limit_ratio,
                    "sim_accepted_sequence": position_tick.accepted_sequence,
                }
            event(
                "bridge_step",
                source_admitted=(
                    None if tick.source_step is None else tick.source_step.admitted
                ),
                retarget_accepted=(
                    None
                    if tick.retarget_result is None
                    else tick.retarget_result.accepted
                ),
                command_applied=position_tick.command_applied,
                command_rejected=position_tick.rejection_reason,
                bridge_rejection_reason=getattr(tick, "rejection_reason", None),
                retarget_rejection_reason=(
                    None
                    if getattr(tick, "retarget_result", None) is None
                    else tick.retarget_result.rejection_reason
                ),
                ik_status=(
                    None if ik_result is None else ik_result.status
                ),
                ik_iterations=(
                    None if ik_result is None else ik_result.iterations
                ),
                ik_position_error_m=(
                    None if ik_result is None else ik_result.position_error_m
                ),
                ik_orientation_error_rad=(
                    None if ik_result is None else ik_result.orientation_error_rad
                ),
                ik_seed_distance_rad=(
                    None if ik_result is None else ik_result.seed_distance_rad
                ),
                ik_sigma_min=(None if ik_result is None else ik_result.sigma_min),
                ik_condition_number=(
                    None if ik_result is None else ik_result.condition_number
                ),
                ik_limit_projection_applied=(
                    None
                    if ik_result is None
                    else ik_result.limit_projection_applied
                ),
                ik_target_seed_translation_delta_m=retarget_diagnostics.get(
                    "target_seed_translation_delta_m"
                ),
                ik_target_seed_orientation_delta_rad=retarget_diagnostics.get(
                    "target_seed_orientation_delta_rad"
                ),
                ik_target_pose_candidate=retarget_diagnostics.get(
                    "target_pose_candidate"
                ),
                ik_seed_qpos_rad=retarget_diagnostics.get("ik_seed_qpos_rad"),
                ik_result_qpos_rad=retarget_diagnostics.get("ik_result_qpos_rad"),
                source_step_elapsed_sec=getattr(tick, "source_step_elapsed_sec", 0.0),
                retarget_elapsed_sec=getattr(tick, "retarget_elapsed_sec", 0.0),
                position_step_elapsed_sec=getattr(tick, "position_step_elapsed_sec", 0.0),
                step_elapsed_sec=getattr(tick, "step_elapsed_sec", 0.0),
                source_receive_monotonic_sec=getattr(
                    tick, "source_receive_monotonic_sec", None
                ),
                source_step_started_monotonic_sec=getattr(
                    tick, "source_step_started_monotonic_sec", None
                ),
                source_step_completed_monotonic_sec=getattr(
                    tick, "source_step_completed_monotonic_sec", None
                ),
                source_admission_monotonic_sec=getattr(
                    tick, "source_admission_monotonic_sec", None
                ),
                source_consumption_monotonic_sec=getattr(
                    tick, "source_consumption_monotonic_sec", None
                ),
                source_completion_to_consumption_latency_sec=getattr(
                    tick, "source_completion_to_consumption_latency_sec", None
                ),
                target_dispatch_monotonic_sec=getattr(
                    tick, "target_dispatch_monotonic_sec", None
                ),
                source_to_target_dispatch_latency_sec=getattr(
                    tick, "source_to_target_dispatch_latency_sec", None
                ),
                source_admission_to_target_dispatch_latency_sec=getattr(
                    tick,
                    "source_admission_to_target_dispatch_latency_sec",
                    None,
                ),
                source_completion_to_target_dispatch_latency_sec=getattr(
                    tick,
                    "source_completion_to_target_dispatch_latency_sec",
                    None,
                ),
                source_poll_deadline_sec=getattr(
                    tick, "source_poll_deadline_sec", None
                ),
                source_poll_lateness_sec=getattr(
                    tick, "source_poll_lateness_sec", None
                ),
                source_poll_schedule_skipped=getattr(
                    tick, "source_poll_schedule_skipped", False
                ),
                **source_debug,
                **position_debug,
            )
            if tick.state == "held":
                reason = (
                    getattr(tick, "rejection_reason", None)
                    or getattr(bridge, "last_rejection_reason", None)
                    or (
                        None
                        if tick.source_step is None
                        else tick.source_step.rejection_reason
                    )
                )
                now = float(clock())
                if (
                    reason != last_hold_reason
                    or now - last_hold_message_at >= 1.0
                ):
                    emit(
                        f"[RM65→B601] held: {reason or 'bridge rejected the target'}; "
                        "inspect before a"
                    )
                    last_hold_reason = reason
                    last_hold_message_at = now
            else:
                last_hold_reason = None
            next_tick += period
            completed_at = float(clock())
            if next_tick <= completed_at:
                next_tick = completed_at + period
    finally:
        try:
            bridge.deactivate()
        finally:
            try:
                viewer.close()
            finally:
                if owns_terminal_reader:
                    close = getattr(poll_command, "close", None)
                    if callable(close):
                        close()
                event("foreground_stopped")
                emit("[RM65→B601] following stopped; MuJoCo target inactive")


__all__ = ["run_rm65_to_b601_foreground"]
