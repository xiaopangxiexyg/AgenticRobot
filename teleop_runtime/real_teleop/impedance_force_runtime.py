"""Impedance force-control admission, health guarding, and force-stop summaries."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from dataset_contracts.stats import (
    FORCE_COLLISION_LABEL_DETECTED,
    has_force_limit_collision_text,
)

from ..backends.arms import RM65ImpedanceArmBackend
from ..diagnostics import TeleopRunState
from ..log_utils import log_event as _log
from .arm_runtime import (
    _active_impedance_force_backends,
    _command_backend_for_side,
    _estimate_impedance_force_control_warmup_sec,
    _impedance_force_control_warmup_backends,
)
from .arm_tool_config import _force_tare_sec_arg
from .components import RealTeleopComponents
from .control_profile import arm_control_mode as _arm_control_mode_arg
from .position_force_guard import PositionForceGuardFault
from .runtime_health import _short_status_text
from .special_arm_runtime import (
    _disengage_position_backends_for_manual_interaction,
    _release_gui_qpos_init_transit,
)


def _warm_start_impedance_force_control_before_home(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: "RealTeleopControlInput",
) -> bool:
    backends = _impedance_force_control_warmup_backends(args, components)
    if not backends:
        return True
    total_estimate_sec = _estimate_impedance_force_control_warmup_sec(args, backends)
    active_sides = tuple(side for side, _ in backends)
    warmup_started_perf = time.perf_counter()
    _log(
        "ForceControlWarmup",
        "cold_start required before runtime home "
        f"sides={list(active_sides)} estimated={total_estimate_sec:.2f}s "
        f"tare_sec={_force_tare_sec_arg(args):.2f} "
        f"io_threads={max(1, int(getattr(args, 'arm_io_threads', 1)))}",
    )

    def warm_start_one(item: tuple[str, RM65ImpedanceArmBackend]) -> tuple[str, dict[str, object], dict[str, float]]:
        side, backend = item
        side_start = time.perf_counter()
        warm_start = time.perf_counter()
        backend.warm_start_force_control()
        warm_elapsed_ms = (time.perf_counter() - warm_start) * 1000.0
        return (
            side,
            backend.get_state(),
            {
                "warm_start_ms": warm_elapsed_ms,
                "total_ms": (time.perf_counter() - side_start) * 1000.0,
            },
        )

    def run_warmup() -> list[tuple[str, dict[str, object], dict[str, float]]]:
        if len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1:
            with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="rm65_warmup_arm") as executor:
                return [future.result() for future in (executor.submit(warm_start_one, item) for item in backends)]
        return [warm_start_one(item) for item in backends]

    clutch_input.publish_force_control_warmup_progress(
        elapsed_sec=0.0,
        total_sec=total_estimate_sec,
        active_sides=active_sides,
    )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rm65_warmup_gate") as executor:
        future = executor.submit(run_warmup)
        while not future.done():
            elapsed_sec = time.perf_counter() - warmup_started_perf
            clutch_input.publish_force_control_warmup_progress(
                elapsed_sec=elapsed_sec,
                total_sec=total_estimate_sec,
                active_sides=active_sides,
            )
            clutch_input.terminal.poll()
            clutch_input.terminal.consume_calibration_request()
            clutch_input.terminal.consume_episode_start_request()
            clutch_input.terminal.consume_episode_stop_request()
            if clutch_input.terminal.should_quit:
                clutch_input.should_quit = True
            if clutch_input.ui is not None:
                ui_events = clutch_input.ui.consume_events()
                deferred_ui_events = []
                for event in ui_events:
                    if event == "q":
                        clutch_input.should_quit = True
                    else:
                        deferred_ui_events.append(event)
                defer_ui_events = getattr(clutch_input, "defer_ui_events", None)
                if callable(defer_ui_events):
                    defer_ui_events(deferred_ui_events)
            time.sleep(0.05)
        try:
            results = future.result()
        except Exception as exc:  # noqa: BLE001
            message = f"力控冷启动失败: {type(exc).__name__}: {exc}"
            _log("ForceControlWarmup", message)
            clutch_input.publish_calibration_waiting(message)
            return False

    elapsed_ms = (time.perf_counter() - warmup_started_perf) * 1000.0
    timing_parts = []
    for side, state, timing in results:
        timing_parts.append(
            f"{side}:total={timing['total_ms']:.1f}ms "
            f"warm_start={timing['warm_start_ms']:.1f}ms "
            f"status={state.get('send_result_text')}"
        )
    _log(
        "ForceControlWarmup",
        f"cold_start completed total={elapsed_ms:.1f}ms " + " | ".join(timing_parts),
    )
    clutch_input.publish_force_control_warmup_progress(
        elapsed_sec=total_estimate_sec,
        total_sec=total_estimate_sec,
        active_sides=active_sides,
    )
    if clutch_input.should_quit:
        return False
    return True


def _emergency_stop_real_arms(
    components: RealTeleopComponents,
    *,
    reason: str,
    now: float | None = None,
    raise_on_failure: bool = False,
) -> dict[str, str]:
    stop_time = time.time() if now is None else float(now)

    def stop_one(side: str) -> tuple[str, str | None]:
        label = f"{side}_arm"
        try:
            backend = _command_backend_for_side(components, side)
        except Exception as exc:  # noqa: BLE001 - report one side and continue the other.
            detail = f"{type(exc).__name__}: {exc}"
            _log(
                "EmergencyStop",
                f"{label} stop lookup failed reason={reason}: {detail}",
                now=stop_time,
            )
            return side, detail
        if backend is None:
            detail = "command backend unavailable"
            _log(
                "EmergencyStop",
                f"{label} stop failed reason={reason}: {detail}",
                now=stop_time,
            )
            return side, detail
        try:
            is_stopped = getattr(backend, "is_emergency_stopped", None)
            if callable(is_stopped) and bool(is_stopped()):
                _log(
                    "EmergencyStop",
                    f"{label} stop already latched reason={reason}",
                    now=stop_time,
                )
                return side, None
            stop = getattr(backend, "emergency_stop", None)
            if callable(stop):
                stop()
            else:
                backend.close()
        except Exception as exc:
            _log(
                "EmergencyStop",
                f"{label} stop failed reason={reason}: {type(exc).__name__}: {exc}",
                now=stop_time,
            )
            return side, f"{type(exc).__name__}: {exc}"
        else:
            _log("EmergencyStop", f"{label} stop requested reason={reason}", now=stop_time)
            return side, None

    active_sides = tuple(str(side) for side in components.active_sides)
    results: list[tuple[str, str | None]]
    if len(active_sides) <= 1:
        results = [stop_one(side) for side in active_sides]
    else:
        # A force trip is bimanual by definition.  Do not wait for one
        # controller's Arm_Stop response before requesting the other arm.
        with ThreadPoolExecutor(
            max_workers=len(active_sides),
            thread_name_prefix="rm65_bimanual_emergency_stop",
        ) as executor:
            futures = [executor.submit(stop_one, side) for side in active_sides]
            results = []
            for side, future in zip(active_sides, futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - one side cannot mask the other.
                    detail = f"{type(exc).__name__}: {exc}"
                    _log(
                        "EmergencyStop",
                        f"{side}_arm stop worker failed reason={reason}: {detail}",
                        now=stop_time,
                    )
                    results.append((side, detail))
    errors = {
        side: detail
        for side, detail in results
        if detail is not None
    }
    if errors and raise_on_failure:
        raise RuntimeError(
            "bimanual emergency stop was not acknowledged by all active arms: "
            + "; ".join(f"{side}={detail}" for side, detail in sorted(errors.items()))
        )
    return errors


def _force_collision_recording_summary(
    *,
    source: str,
    control_mode: str,
    reason: str,
    message: str,
    side: str | None = None,
    fault: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "detected": True,
        "source": str(source),
        "control_mode": str(control_mode),
        "reason": str(reason),
        "message": str(message),
    }
    if side:
        event["side"] = str(side)
    if fault is not None:
        event["fault"] = dict(fault)
    return {
        "force_collision_detected": True,
        "force_collision_label": FORCE_COLLISION_LABEL_DETECTED,
        "force_collision_event": event,
    }


def _position_force_guard_collision_summary(
    fault: PositionForceGuardFault | None,
) -> dict[str, object]:
    if fault is None or str(fault.reason) != "force_limit_exceeded":
        return {}
    fault_payload = fault.to_dict()
    return _force_collision_recording_summary(
        source="position_force_guard",
        control_mode="position",
        reason=str(fault.reason),
        message=str(fault.message),
        side=fault.side,
        fault=fault_payload,
    )


def _force_guard_recording_stop_context(
    fault: PositionForceGuardFault | None,
    *,
    fallback_end_reason: str,
    fallback_summary: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Bind an asynchronous force-guard trip before the episode closes."""

    summary = dict(fallback_summary or {})
    if fault is None:
        return str(fallback_end_reason), summary
    summary["reason"] = "position_force_guard"
    summary["position_force_guard_fault"] = fault.to_dict()
    summary.update(_position_force_guard_collision_summary(fault))
    return "position_force_guard", summary


def _exception_chain_messages(error: BaseException | None) -> tuple[str, ...]:
    messages: list[str] = []
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return tuple(messages)


def _terminal_force_collision_summary(
    error: BaseException | None,
    *,
    control_mode: str,
) -> dict[str, object]:
    messages = _exception_chain_messages(error)
    if not messages or not has_force_limit_collision_text(messages):
        return {}
    return _force_collision_recording_summary(
        source="controller_hard_stop",
        control_mode=control_mode,
        reason="force_limit_exceeded",
        message=" <- ".join(messages),
    )


def _handle_init_qpos_emergency_stop(
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    now: float,
) -> None:
    """Abort only the GUI Init-qpos operation and leave the runtime usable.

    This is intentionally not ``_emergency_stop_real_arms``: the GUI Init
    button is an operation-level stop, not a process-wide/backend-latched
    emergency stop.  It releases the exact-qpos lease, cancels the planner,
    sends the normal Hold command, and leaves the operator in the GUI.
    """

    controller = getattr(components, "init_arm_controller", None)
    operation_id = str(getattr(controller, "qpos_operation_id", "") or "")
    release_errors: list[str] = []
    if controller is not None and bool(getattr(controller, "qpos_active", False)):
        _messages, release_errors = _release_gui_qpos_init_transit(
            components,
            operation_id=operation_id,
            now=now,
            sides=("left", "right"),
        )
        try:
            transition = controller.cancel_active(
                now=now,
                reason="gui_init_qpos_emergency_stop",
            )
        except Exception as exc:  # noqa: BLE001 - keep the normal Hold fallback alive.
            release_errors.append(f"Init 状态取消异常: {type(exc).__name__}: {exc}")
        else:
            if not bool(getattr(transition, "ok", False)):
                release_errors.append(
                    "Init 状态取消失败: " + str(getattr(transition, "message", ""))
                )
    clutch_input._cancel_init_arm_qpos_sequence(reason="GUI Init 急停")
    message = "Init 归位已急停并取消；机械臂保持静止，可继续使用 GUI。"
    if release_errors:
        message += " 精确控制权释放存在问题：" + "；".join(release_errors)
    try:
        components.session.hold_hardware(message)
    except Exception as exc:  # noqa: BLE001 - report but never resume motion silently.
        message += f" Hold 下发失败: {type(exc).__name__}: {exc}"
    try:
        _disengage_position_backends_for_manual_interaction(
            components,
            now=now,
            reason="gui_init_qpos_emergency_stop",
        )
    except Exception as exc:  # noqa: BLE001 - keep the Hold/UI path alive.
        message += f" 位置控制流释放失败: {type(exc).__name__}: {exc}"
    if bool(getattr(clutch_input, "_teleop_started", False)):
        clutch_input.request_hold(reason="init_qpos_emergency_stop")
        # The normal Hold command was issued above; avoid a duplicate command
        # on the next loop while retaining the coordinator's Hold/UI state.
        clutch_input._hold_requested = False
    clutch_input._publish_init_arm_state(message=message, control_hint=message)
    _log("InitArmQpos", message, now=now)


def _enable_impedance_force_control_after_runtime_home(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    home_captured: bool = True,
) -> None:
    force_control_start = time.perf_counter()
    backends = _active_impedance_force_backends(args, components)
    if not backends:
        return
    _log(
        "ForceControl",
        f"runtime home {'captured' if home_captured else 'skipped'}; "
        f"hold active tools still for tare_sec={_force_tare_sec_arg(args):.2f}s "
        f"before enabling pose impedance.",
    )

    def enable_one(
        item: tuple[str, RM65ImpedanceArmBackend],
    ) -> tuple[str, dict[str, object], dict[str, float]]:
        side, backend = item
        side_start = time.perf_counter()
        enable_start = time.perf_counter()
        backend.enable_force_control()
        enable_elapsed_ms = (time.perf_counter() - enable_start) * 1000.0
        timeout_sec = max(1.0, 3.0 * float(backend.config.control_dt))
        wait_start = time.perf_counter()
        state = backend.wait_until_force_control_ready(timeout_sec=timeout_sec)
        wait_elapsed_ms = (time.perf_counter() - wait_start) * 1000.0
        return (
            side,
            state,
            {
                "enable_ms": enable_elapsed_ms,
                "wait_ready_ms": wait_elapsed_ms,
                "total_ms": (time.perf_counter() - side_start) * 1000.0,
            },
        )

    if len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1:
        with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="rm65_tare") as executor:
            results = [future.result() for future in (executor.submit(enable_one, item) for item in backends)]
    else:
        results = [enable_one(item) for item in backends]
    timing_parts = []
    for side, state, timing in results:
        timing_parts.append(
            f"{side}:total={timing['total_ms']:.1f}ms "
            f"enable={timing['enable_ms']:.1f}ms "
            f"wait_ready={timing['wait_ready_ms']:.1f}ms"
        )
        record = state.get("last_impedance_record")
        step = record.get("step") if isinstance(record, dict) else None
        force_norm = record.get("force_norm_n") if isinstance(record, dict) else None
        torque_norm = record.get("torque_norm_nm") if isinstance(record, dict) else None
        wrench_text = ""
        if force_norm is not None and torque_norm is not None:
            wrench_text = f" first_tick_force={float(force_norm):.3f}N torque={float(torque_norm):.3f}Nm"
        target_offset = record.get("target_offset") if isinstance(record, dict) else None
        if target_offset is not None:
            offset_values = np.asarray(target_offset, dtype=float).reshape(-1)
            if offset_values.size >= 6 and np.all(np.isfinite(offset_values[:6])):
                linear_m = float(np.linalg.norm(offset_values[:3]))
                angular_rad = float(np.linalg.norm(offset_values[3:6]))
                wrench_text += f" first_tick_offset={linear_m:.5f}m/{angular_rad:.5f}rad"
        _log(
            "ForceControl",
            f"{side}_arm pose impedance enabled "
            f"tool={state.get('active_tool_name')} "
            f"profile={state.get('active_tool_profile_path')} "
            f"tare_sec={state.get('force_tare_sec')} "
            f"loop={state.get('impedance_loop_state')} "
            f"tick={step} "
            f"send={state.get('send_result_text')}"
            f"{wrench_text}",
        )
    _log(
        "CalibrationTiming",
        "force_control_after_home "
        f"total={(time.perf_counter() - force_control_start) * 1000.0:.1f}ms "
        f"tare_sec={_force_tare_sec_arg(args):.2f} "
        + " | ".join(timing_parts),
    )

def _impedance_force_control_issue(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> str | None:
    if _arm_control_mode_arg(args) != "impedance":
        return None
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ImpedanceArmBackend):
            raise RuntimeError(f"{side}_arm command backend is not RM65ImpedanceArmBackend.")
        state = backend.get_state()
        fault = state.get("fault")
        if fault:
            return (
                f"{side}_arm pose impedance fault before command dispatch: "
                f"{_short_status_text(fault, max_len=240)}; status={backend.force_control_status_text()}"
            )
        if not bool(state.get("impedance_enabled")):
            return (
                f"{side}_arm pose impedance is disabled before command dispatch: "
                f"{backend.force_control_status_text()}"
            )
        if state.get("impedance_loop_state") == "stopped":
            return (
                f"{side}_arm pose impedance loop stopped before command dispatch: "
                f"{backend.force_control_status_text()}"
            )
    return None


def hold_motion_after_impedance_guard_issue(
    components: RealTeleopComponents,
    *,
    reason: str,
    now: float,
    log: Callable[..., None],
) -> None:
    rig = components.session.rig
    if rig is None:
        log(
            "ForceControl",
            f"control fault hold_motion skipped because session has no rig: {_short_status_text(reason)}",
            now=now,
        )
        return
    try:
        rig.hold_motion(f"control fault: {reason}")
    except Exception as exc:  # noqa: BLE001
        log(
            "ForceControl",
            "control fault hold_motion failed: "
            f"{type(exc).__name__}: {exc}; reason={_short_status_text(reason)}",
            now=now,
        )
    else:
        log(
            "ForceControl",
            f"control fault hold_motion requested for all rig backends: {_short_status_text(reason)}",
            now=now,
        )


def guard_impedance_force_control(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    context: str,
    now: float,
    issue_provider: Callable[..., str | None],
    hold_handler: Callable[..., None],
    log: Callable[..., None],
    collision_handler: Callable[[str], None] | None = None,
) -> bool:
    if context == "main_loop" and components.session.state_machine.state not in {
        TeleopRunState.ARMED,
        TeleopRunState.RUNNING,
        TeleopRunState.RECOVERING,
    }:
        return True
    try:
        issue = issue_provider(args, components)
    except Exception as exc:  # noqa: BLE001 - classify raced controller stops below.
        issue_text = f"{context}: {type(exc).__name__}: {exc}"
        lowered = issue_text.lower()
        stop_latch_race = any(
            marker in lowered
            for marker in (
                "emergency_stop is latched",
                "emergency stop is latched",
                "emergency_stop_failed",
                "emergency stop was not acknowledged",
            )
        )
        if collision_handler is not None and (
            has_force_limit_collision_text((issue_text,)) or stop_latch_race
        ):
            collision_handler(issue_text)
            return False
        raise
    if issue is None:
        return True
    reason = f"{context}: {issue}"
    # A controller-side force hard-stop is already a safety event, not an
    # ordinary impedance-health fault.  Let the real-teleop control loop
    # perform the bimanual Arm_Stop, close/label the active episode, and enter
    # the normal GUI Hold path.  Keeping this callback before the generic
    # ``FAULT`` transition is important: FAULT is terminal for a session and
    # would strand the GUI even though the hardware stop was acknowledged.
    if (
        collision_handler is not None
        and has_force_limit_collision_text((issue,))
    ):
        collision_handler(reason)
        return False
    if components.session.state_machine.state != TeleopRunState.FAULT:
        log("ForceControl", reason, now=now)
        hold_handler(components, reason=reason, now=now)
        components.session.fault(reason)
    return False

__all__ = [
    "_emergency_stop_real_arms",
    "_enable_impedance_force_control_after_runtime_home",
    "_exception_chain_messages",
    "_force_collision_recording_summary",
    "_force_guard_recording_stop_context",
    "_handle_init_qpos_emergency_stop",
    "_impedance_force_control_issue",
    "_position_force_guard_collision_summary",
    "_terminal_force_collision_summary",
    "_warm_start_impedance_force_control_before_home",
    "guard_impedance_force_control",
    "hold_motion_after_impedance_guard_issue",
]
