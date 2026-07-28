"""Real-arm state synchronization and execution-step maintenance."""

from __future__ import annotations

import time

import numpy as np

from robot_motion.rm65 import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS

from ..backends.arms import RM65ArmBackend
from ..episodes import compare_start_qpos_pair
from ..log_utils import log_event as _log
from .arm_runtime import (
    _command_backend_for_side,
    _sync_dual_arm_retargeter_for_real_control,
)
from .arm_tool_config import _alignment_qpos_pair, _start_state_summary
from .components import RealTeleopComponents


def _qpos_summary(qpos: np.ndarray | None) -> str:
    if qpos is None:
        return "None"
    return str(np.round(np.asarray(qpos, dtype=float), 3).tolist())


def _state_qpos(backend: RM65ArmBackend, fallback: np.ndarray) -> np.ndarray:
    state = backend.get_state()
    qpos = state["current_qpos"]
    if qpos is None:
        return fallback.copy()
    return np.asarray(qpos, dtype=float).copy()


def _run_initial_alignment_start_check(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> str:
    recorded_qpos_pair = _alignment_qpos_pair(components.loaded_initial_alignment)
    if recorded_qpos_pair is None:
        if components.loaded_initial_alignment is not None:
            _log(
                "StartCheck",
                "loaded initial alignment does not contain both left/right arm_qpos entries; startup joint check skipped.",
            )
        return "skipped"

    current_qpos_pair = {
        "left": _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS),
        "right": _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS),
    }
    inactive_sides = set(("left", "right")) - set(components.active_sides)
    for side in inactive_sides:
        current_qpos_pair[side] = recorded_qpos_pair[side]
    checks = compare_start_qpos_pair(
        current_qpos_pair=current_qpos_pair,
        recorded_qpos_pair=recorded_qpos_pair,
        joint_distance_weights=components.joint_distance_weights,
        max_abs_threshold_deg=float(args.max_start_joint_error_deg),
        weighted_threshold_deg=float(args.max_start_weighted_error_deg),
    )
    checks = {side: checks[side] for side in components.active_sides}
    _log("StartCheck", f"initial-alignment check: {_start_state_summary(checks)}")
    for side in components.active_sides:
        check = checks[side]
        _log(
            "StartCheck",
            f"{side} current_q={_qpos_summary(check.current_qpos)} "
            f"alignment_q={_qpos_summary(check.recorded_qpos)} "
            f"abs_joint_error_deg={_qpos_summary(check.abs_joint_error_deg)}",
        )
    start_ok = all(check.within_limits for check in checks.values())
    if not start_ok and not args.ignore_start_mismatch:
        raise RuntimeError(
            "Real teleop aborted because the current start-state joint error exceeds the configured threshold."
        )
    if not start_ok:
        return "ignored_mismatch"
    return "passed"


def _sync_preview_from_real_state_real(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in components.active_sides}
    backends = (
        ("left", components.left_state_backend, components.left_preview_backend),
        ("right", components.right_state_backend, components.right_preview_backend),
    )
    active = [
        (side, state_backend, preview_backend)
        for side, state_backend, preview_backend in backends
        if side in components.active_sides and state_backend is not None
    ]
    parallel_submit_failed = False
    if components.arm_state_read_executor is not None and len(active) > 1:
        try:
            future_map = {
                components.arm_state_read_executor.submit(state_backend.refresh_state_from_hardware): (
                    side,
                    preview_backend,
                )
                for side, state_backend, preview_backend in active
            }
        except RuntimeError:
            parallel_submit_failed = True
        else:
            for future, (side, preview_backend) in future_map.items():
                try:
                    state = future.result()
                except Exception:
                    if required:
                        raise
                    continue
                qpos = state.get("current_qpos")
                if qpos is None:
                    if required:
                        raise RuntimeError(
                            f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                        )
                    continue
                preview_backend.set_local_qpos(qpos)
                synced[side] = True
    if components.arm_state_read_executor is None or len(active) <= 1 or parallel_submit_failed:
        for side, state_backend, preview_backend in backends:
            if side not in components.active_sides or state_backend is None:
                continue
            try:
                state = state_backend.refresh_state_from_hardware()
            except Exception:
                if required:
                    raise
                continue
            qpos = state.get("current_qpos")
            if qpos is None:
                if required:
                    raise RuntimeError(
                        f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                    )
                continue
            preview_backend.set_local_qpos(qpos)
            synced[side] = True
    if required or all(synced.values()):
        _sync_dual_arm_retargeter_for_real_control(components)
    return synced


def _reset_runtime_motion_state_from_real_qpos(
    components: RealTeleopComponents,
    *,
    required: bool,
) -> dict[str, bool]:
    reset_start = time.perf_counter()
    synced = {side: False for side in components.active_sides}
    timing_parts: list[str] = []
    for side in components.active_sides:
        side_start = time.perf_counter()
        state_backend = components.left_state_backend if side == "left" else components.right_state_backend
        preview_backend = components.left_preview_backend if side == "left" else components.right_preview_backend
        command_backend = _command_backend_for_side(components, side)
        if state_backend is None:
            if required:
                raise RuntimeError(f"{side}_arm has no real state backend for runtime restart.")
            timing_parts.append(f"{side}:missing_state_backend")
            continue
        refresh_start = time.perf_counter()
        state = state_backend.refresh_state_from_hardware()
        refresh_elapsed_ms = (time.perf_counter() - refresh_start) * 1000.0
        qpos = state.get("current_qpos")
        if qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: no current_qpos available from hardware."
                )
            timing_parts.append(f"{side}:refresh={refresh_elapsed_ms:.1f}ms no_qpos")
            continue
        try:
            qpos_array = np.asarray(qpos, dtype=float).reshape(6)
        except (TypeError, ValueError) as exc:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: invalid current_qpos from hardware."
                ) from exc
            continue
        if not np.all(np.isfinite(qpos_array)):
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: non-finite current_qpos from hardware."
                )
            continue
        reset_side_start = time.perf_counter()
        preview_backend.set_local_qpos(qpos_array)
        preview_anchor_qpos = preview_backend.get_state().get("current_qpos")
        if preview_anchor_qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: preview did not retain current_qpos."
                )
            continue
        preview_anchor_qpos = np.asarray(preview_anchor_qpos, dtype=float).reshape(6)
        if not np.all(np.isfinite(preview_anchor_qpos)):
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: preview produced non-finite current_qpos."
                )
            continue
        if command_backend is not None:
            # A position-force collision leaves the command backend latched
            # and its CANFD worker stopped.  Runtime Home is the explicit
            # operator authorization to clear that latch; never clear it from
            # the normal control loop or from a safety callback.
            rearm = getattr(
                command_backend,
                "manual_rearm_after_emergency_stop",
                None,
            )
            is_stopped = getattr(command_backend, "is_emergency_stopped", None)
            if callable(rearm) and callable(is_stopped) and bool(is_stopped()):
                rearm(preview_anchor_qpos)
            command_backend.reset_motion_state(preview_anchor_qpos)
        reset_side_elapsed_ms = (time.perf_counter() - reset_side_start) * 1000.0
        synced[side] = True
        timing_parts.append(
            f"{side}:total={(time.perf_counter() - side_start) * 1000.0:.1f}ms "
            f"refresh={refresh_elapsed_ms:.1f}ms "
            f"reset={reset_side_elapsed_ms:.1f}ms"
        )
    sync_start = time.perf_counter()
    _sync_dual_arm_retargeter_for_real_control(components)
    sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0
    _log(
        "CalibrationTiming",
        "runtime_restart_state "
        f"total={(time.perf_counter() - reset_start) * 1000.0:.1f}ms "
        f"retargeter_sync={sync_elapsed_ms:.1f}ms "
        + " | ".join(timing_parts),
    )
    return synced


def _sync_real_execution_step_limits(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
) -> None:
    current_time = time.time() if now is None else float(now)
    active_sides = tuple(getattr(components, "active_sides", ()) or ())
    if not active_sides:
        active_sides = tuple(
            side
            for side in ("left", "right")
            if _command_backend_for_side(components, side) is not None
        )
    for side in active_sides:
        backend = _command_backend_for_side(components, side)
        prepare_step = getattr(backend, "prepare_effective_max_joint_step", None)
        if callable(prepare_step):
            prepare_step(now=current_time)

    if components.dual_arm_retargeter is None:
        return
    if not components.dual_arm_retargeter.config.enforce_realistic_joint_step:
        return
    if components.left_command_backend is None or components.right_command_backend is None:
        return
    tracker = components.arm_loop_step_dt_tracker
    command_config = components.left_command_backend.config
    scale_outer_execution_step_by_actual_dt = bool(
        getattr(command_config, "scale_outer_execution_step_by_actual_dt", True)
    )
    backend_control_dt = float(command_config.control_dt)
    action_hz = getattr(command_config, "action_hz", None)
    nominal_outer_dt = (
        1.0 / float(action_hz)
        if action_hz is not None and float(action_hz) > 0.0
        else backend_control_dt
    )
    # Impedance deliberately does not expand a per-frame motion allowance
    # after a slow outer frame.  Its fixed allowance must nevertheless use the
    # nominal *outer* target cadence, not the 40 Hz inner controller.
    # Position backends retain their existing actual-dt policy.
    base_dt = (
        backend_control_dt
        if scale_outer_execution_step_by_actual_dt
        else nominal_outer_dt
    )
    smoothed_loop_dt = float(tracker.get("smoothed_loop_dt", base_dt) or base_dt)
    last_loop_time = tracker.get("last_loop_time")
    if not scale_outer_execution_step_by_actual_dt or last_loop_time is None:
        smoothed_loop_dt = base_dt
    else:
        raw_loop_dt = max(0.0, current_time - float(last_loop_time))
        if raw_loop_dt > float(command_config.actual_send_dt_stall_reset_sec):
            smoothed_loop_dt = base_dt
        else:
            alpha = float(command_config.actual_send_dt_alpha)
            smoothed_loop_dt = alpha * raw_loop_dt + (1.0 - alpha) * smoothed_loop_dt
    tracker["last_loop_time"] = current_time
    tracker["smoothed_loop_dt"] = smoothed_loop_dt
    effective_loop_dt = float(
        np.clip(
            smoothed_loop_dt,
            base_dt * float(command_config.actual_send_dt_min_scale),
            base_dt * float(command_config.actual_send_dt_max_scale),
        )
    )
    explicit_steps_configured = (
        components.dual_arm_retargeter.config.left_execution_max_joint_step_rad is not None
        and components.dual_arm_retargeter.config.right_execution_max_joint_step_rad is not None
    )
    if explicit_steps_configured or components.dual_arm_retargeter.config.joint_velocity_profile is None:
        components.dual_arm_retargeter.set_execution_max_joint_step(
            left_step_rad=np.asarray(components.left_command_backend.config.max_angvel, dtype=float)
            * effective_loop_dt,
            right_step_rad=np.asarray(components.right_command_backend.config.max_angvel, dtype=float)
            * effective_loop_dt,
        )
        return
    components.dual_arm_retargeter.set_execution_dt_sec(effective_loop_dt)

__all__ = [
    "_reset_runtime_motion_state_from_real_qpos",
    "_run_initial_alignment_start_check",
    "_sync_preview_from_real_state_real",
    "_sync_real_execution_step_limits",
]
