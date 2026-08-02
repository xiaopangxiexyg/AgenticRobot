"""Startup hold, exact-qpos maintenance, and runtime-home orchestration.

This module owns only the startup-stage ordering that used to live at the end
of the canonical real-teleop entrypoint.  Its compatibility binding keeps
historical entrypoint monkeypatch points live without importing debug modules.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import inspect
import time
from typing import TYPE_CHECKING

import numpy as np

from robot_motion.runtime import EffectorCommand, TeleopCommand

from ..collection.camera_preview import CameraPreviewer
from ..collection.sample_assembler import DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
from ..log_utils import ChangeGate
from ..sources.camera import RealSenseRGBSource

if TYPE_CHECKING:
    from ..operator_ui_ports import OperatorUiPort
    from .operator_control_runtime import RealTeleopControlInput
    from ..special_tasks.init_arm import InitArmCommandController
    from .components import RealTeleopComponents


CAMERA_STARTUP_SELF_CHECK_DURATION_SEC = 3.0
CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC = 0.2
CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES = 3
CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC

_SUPPORT_NAMES = (
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "_advance_startup_gui_qpos_maintenance",
    "_camera_self_check_payload",
    "_cancel_special_arm_modes_for_hold",
    "_enable_impedance_force_control_after_runtime_home",
    "_expand_cli_path",
    "_finish_startup_gui_qpos_terminal",
    "_handle_ik_debug_log_toggle_request",
    "_handle_init_qpos_emergency_stop",
    "_impedance_force_control_not_ready_sides",
    "_impedance_force_control_warmup_backends",
    "_log",
    "_release_gui_qpos_init_transit",
    "_reset_runtime_motion_state_from_real_qpos",
    "_safe_float_value",
    "_safe_int_value",
    "_startup_gui_qpos_seed_command",
    "_sync_preview_from_real_state_real",
    "_warm_start_impedance_force_control_before_home",
    "_with_expected_wrist_depth_camera_statuses",
)


class _LegacyFunctionProxy:
    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive startup support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "real-teleop startup support is incomplete: "
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


def _run_camera_startup_self_check(
    *,
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    clutch_input: RealTeleopControlInput,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    camera_previewer: CameraPreviewer | None = None,
    duration_sec: float = CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    interval_sec: float = CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
) -> dict[str, object]:
    started_at = time.time()
    duration_sec = max(0.0, float(duration_sec))
    interval_sec = max(0.01, float(interval_sec))
    final_payload: dict[str, object] = {}
    camera_sources = camera_sources or {}
    if not camera_sources:
        expected_status = _with_expected_wrist_depth_camera_statuses(
            {},
            args=clutch_input.args,
            now=started_at,
        )
        final_payload = _camera_self_check_payload(
            expected_status,
            started_at=started_at,
            now=started_at,
            duration_sec=duration_sec,
            completed=True,
        )
        if clutch_input.ui is not None:
            clutch_input.ui.publish(
                camera_self_check=dict(final_payload),
                camera_status=expected_status,
            )
        _log("CameraSelfCheck", str(final_payload.get("message") or final_payload), now=started_at)
        return final_payload
    _log(
        "CameraSelfCheck",
        "startup self-check "
        f"timeout={duration_sec:.1f}s cameras={sorted(camera_sources)} "
        f"early_success={CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES} "
        "consecutive new frames/camera "
        f"age<={CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC:.3f}s",
        now=started_at,
    )
    previous_frame_counts: dict[str, int | None] = {
        str(camera_key): None for camera_key in camera_sources
    }
    fresh_frame_streaks: dict[str, int] = {
        str(camera_key): 0 for camera_key in camera_sources
    }
    while True:
        now = time.time()
        timed_out = now - started_at >= duration_sec
        status = clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=now,
            force=True,
        )
        for camera_key in camera_sources:
            key = str(camera_key)
            item = status.get(key)
            frame_count = _safe_int_value(item.get("frame_count")) if isinstance(item, Mapping) else None
            frame_age_sec = (
                _safe_float_value(item.get("last_frame_age_sec"))
                if isinstance(item, Mapping)
                else None
            )
            status_level = str(item.get("status_level") or "") if isinstance(item, Mapping) else ""
            previous_count = previous_frame_counts[key]
            is_new_fresh_frame = bool(
                frame_count is not None
                and int(frame_count) > 0
                and (previous_count is None or int(frame_count) > previous_count)
                and frame_age_sec is not None
                and 0.0 <= float(frame_age_sec) <= CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC
                and status_level == "ok"
            )
            fresh_frame_streaks[key] = fresh_frame_streaks[key] + 1 if is_new_fresh_frame else 0
            previous_frame_counts[key] = frame_count
        early_success = bool(fresh_frame_streaks) and all(
            streak >= CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES
            for streak in fresh_frame_streaks.values()
        )
        completed = bool(timed_out or early_success)
        final_payload = _camera_self_check_payload(
            status,
            started_at=started_at,
            now=now,
            duration_sec=duration_sec,
            completed=completed,
        )
        final_payload.update(
            {
                "completion_reason": (
                    "fresh_frames" if early_success else "timeout" if timed_out else None
                ),
                "fresh_frame_max_age_sec": CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
                "fresh_frame_required_consecutive_samples": CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
                "fresh_frame_streaks": dict(fresh_frame_streaks),
            }
        )
        if clutch_input.ui is not None:
            updates: dict[str, object] = {"camera_self_check": dict(final_payload)}
            if completed and final_payload.get("status") in {"failed", "warning"}:
                updates["control_hint"] = str(final_payload.get("message") or "")
            clutch_input.ui.publish(**updates)
        if completed:
            break
        clutch_input.poll()
        if clutch_input.should_quit:
            break
        sleep_sec = min(interval_sec, max(0.0, duration_sec - (time.time() - started_at)))
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    finished_at = time.time()
    message = str(final_payload.get("message") or "")
    _log(
        "CameraSelfCheck",
        f"{message or str(final_payload)} "
        f"elapsed={max(0.0, finished_at - started_at):.3f}s "
        f"completion_reason={final_payload.get('completion_reason')}",
        now=finished_at,
    )
    if clutch_input.ui is not None and message:
        clutch_input.ui.append_message("CameraSelfCheck", message)
    return final_payload


def _release_runtime_state_readers(components: RealTeleopComponents) -> None:
    if components.arm_state_read_executor is not None:
        components.arm_state_read_executor.shutdown(wait=True)


def _log_dual_arm_readiness(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> None:
    active_sides = tuple(components.active_sides)
    if set(active_sides) != {"left", "right"}:
        _log(
            "DualArmReady",
            f"active_sides={list(active_sides)}; dual-arm collision guard is not used for single-side teleop.",
        )
        return
    if not components.collision_guard_requested:
        _log(
            "DualArmReady",
            "WARNING: dual-arm collision guard explicitly disabled; "
            "the session will use two independent arm retargeters.",
        )
        return
    retargeter = components.dual_arm_retargeter
    if retargeter is None:
        error = components.collision_guard_setup_error or "unknown setup error"
        _log(
            "DualArmReady",
            "WARNING: dual-arm collision guard was requested but is not active; "
            f"setup_error={error}. Do not run close-range dual-arm tests in this mode.",
        )
        return

    safety_profile = retargeter.ik_safety_profile
    joint_velocity_profile = retargeter.joint_velocity_profile
    hard_indices = tuple(int(index) for index in safety_profile.hard_branch_joint_indices)
    j1_relaxed = 0 not in hard_indices
    time_budget_ms = retargeter.config.solve_time_budget_ms
    _log(
        "DualArmReady",
        "guard=enabled "
        f"scene_profile={retargeter.profile.name} "
        f"collision_pairs={len(retargeter.profile.collision_pairs)} "
        f"mode={retargeter.config.contact_mode} "
        f"output={retargeter.config.output_command_type} "
        f"ik_hz={float(retargeter.config.solver_frequency):.1f} "
        f"time_budget_ms={0.0 if time_budget_ms is None else float(time_budget_ms):.1f} "
        f"safety_profile={safety_profile.name} "
        f"hard_branch_joint_indices={list(hard_indices)} "
        f"j1_hard_branch_relaxed={j1_relaxed}",
    )
    local_bounds = safety_profile.metadata.get(
        "impedance_bimanual_local_step_bounds"
    )
    if isinstance(local_bounds, Mapping):
        _log(
            "DualArmReady",
            "impedance_local_candidate_bounds="
            f"{float(local_bounds.get('max_position_worsen_m', 0.0)) * 1000.0:.1f}mm/"
            f"{float(local_bounds.get('max_orientation_worsen_deg', 0.0)):.1f}deg "
            f"dynamic_worsen={bool(local_bounds.get('dynamic_worsen_enabled', False))} "
            f"scope={local_bounds.get('scope', 'unknown')}",
        )
    if joint_velocity_profile is not None:
        _log(
            "DualArmReady",
            "joint_velocity_limit "
            f"profile={joint_velocity_profile.name} "
            f"max_velocity_deg_s={np.round(np.asarray(joint_velocity_profile.max_velocity_deg_s, dtype=float), 3).tolist()} "
            f"max_dt_sec={float(joint_velocity_profile.max_dt_sec):.3f}",
        )
    posture = retargeter.posture_preference
    if posture is not None:
        _log(
            "DualArmReady",
            "rm65_posture_preference "
            f"name={posture.name} "
            f"local_escape={bool(posture.local_escape_enabled)} "
            f"wrist_unload={bool(posture.wrist_unload_enabled)} "
            f"j2_preferred=[{posture.j2_preferred_min_deg:.1f},{posture.j2_preferred_max_deg:.1f}]deg "
            f"j3_preferred=[{posture.j3_preferred_min_deg:.1f},{posture.j3_preferred_max_deg:.1f}]deg "
            f"hard_escape_allowed={list(posture.hard_branch_escape_allowed_joint_indices)}",
        )
    _log(
        "DualArmReady",
        "runtime_home will be captured from live real-arm qpos after startup; "
        "the loaded calibration profile defines coordinate frames, not reusable wrist home.",
    )


def _startup_gui_qpos_seed_command(
    controller: InitArmCommandController,
    *,
    now: float,
) -> TeleopCommand:
    """Build the complete neutral input required by the shared qpos filter."""

    qpos_by_side = controller.qpos_seed_qpos_by_side()
    if set(qpos_by_side) != {"left", "right"}:
        raise RuntimeError("startup Init qpos requires a complete dual-arm command seed")
    command = TeleopCommand(
        timestamp=float(now),
        metadata={
            "source": "startup_gui_qpos_maintenance",
            "startup_qpos_maintenance": True,
            "dataset_recording_eligible": False,
        },
    )
    for side in ("left", "right"):
        qpos = np.asarray(qpos_by_side[side], dtype=float).reshape(6)
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"startup Init qpos seed for {side} is non-finite")
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=qpos.copy(),
                frame_name="arm_base",
                metadata={
                    "source": "startup_gui_qpos_maintenance_seed",
                    "startup_qpos_maintenance": True,
                },
            )
        )
    return command


def _finish_startup_gui_qpos_terminal(
    *,
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    terminal: Mapping[str, object],
    now: float,
) -> None:
    """Close one GUI qpos-maintenance operation without capturing Home."""

    controller = getattr(components, "init_arm_controller", None)
    if controller is None:
        return
    status = str(terminal.get("status", "failed") or "failed")
    side = str(terminal.get("side", "") or "")
    operation_id = str(getattr(controller, "qpos_operation_id", "") or "")
    sequence_was_active = bool(
        getattr(clutch_input, "_init_qpos_sequence_active", False)
    )
    regular_hold = bool(
        getattr(clutch_input, "_teleop_started", False)
        and clutch_input.hold_active()
    )
    maintenance_label = "Hold 归位" if regular_hold else "启动前归位"
    hold_state_label = "常规 Hold" if regular_hold else "启动前 Hold"

    release_errors: list[str] = []
    if status == "completed" and side in {"left", "right"}:
        support_side = "right" if side == "left" else "left"
        _messages, release_errors = _release_gui_qpos_init_transit(
            components,
            operation_id=operation_id,
            now=now,
            sides=(support_side,),
        )
    if not release_errors:
        clutch_input.handle_init_arm_qpos_terminal(terminal, now=now)
    else:
        clutch_input._cancel_init_arm_qpos_sequence(
            reason="精确控制权释放失败：" + "；".join(release_errors)
        )

    # The first half of a dual sequence has already released its old leases
    # and started a new right-arm operation inside the shared sequence handler.
    if (
        status == "completed"
        and sequence_was_active
        and bool(getattr(clutch_input, "_init_qpos_sequence_active", False))
        and str(getattr(controller, "qpos_operation_id", "") or "") != operation_id
    ):
        return

    _messages, final_release_errors = _release_gui_qpos_init_transit(
        components,
        operation_id=operation_id,
        now=now,
    )
    release_errors.extend(final_release_errors)
    if (
        bool(getattr(controller, "active", False))
        and str(getattr(controller, "qpos_operation_id", "") or "") == operation_id
    ):
        try:
            transition = controller.cancel_active(
                now=now,
                reason=(
                    f"hold_qpos_{status}_without_home_capture"
                    if regular_hold
                    else f"startup_qpos_{status}_without_home_capture"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - startup maintenance fails closed.
            release_errors.append(f"Init 状态释放异常: {type(exc).__name__}: {exc}")
        else:
            if not bool(getattr(transition, "ok", False)):
                release_errors.append(
                    "Init 状态释放失败: " + str(getattr(transition, "message", ""))
                )

    if release_errors:
        message = f"{maintenance_label}结束，但控制权释放失败：" + "；".join(release_errors)
        components.session.hold_hardware(message)
    elif status == "completed":
        if sequence_was_active:
            message = str(
                getattr(clutch_input, "_init_qpos_sequence_status", "")
                or f"双臂顺序归位完成；当前保持在{hold_state_label}。"
            )
        else:
            side_label = "左臂" if side == "left" else "右臂"
            message = (
                f"{side_label}精确归位完成；未采集 tracker Home，"
                f"当前保持在{hold_state_label}。"
            )
    else:
        detail = str(terminal.get("message", "归位失败") or "归位失败")
        message = f"{maintenance_label}失败：{detail}；机械臂保持停止，未采集 tracker Home。"
        components.session.hold_hardware(message)

    _log("InitArmQpos", message, now=now)
    clutch_input._publish_init_arm_state(message=message, control_hint=message)


def _advance_startup_gui_qpos_maintenance(
    *,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    now: float,
) -> bool:
    """Advance one explicit GUI exact-qpos frame outside normal teleop."""

    del args  # Control-mode admission already happened in _start_gui_qpos_init.
    controller = getattr(components, "init_arm_controller", None)
    if controller is None or not bool(getattr(controller, "qpos_active", False)):
        return False
    try:
        command = _startup_gui_qpos_seed_command(controller, now=now)
        command_filter = getattr(components.session, "command_filter", None)
        if not callable(command_filter):
            raise RuntimeError("startup Init qpos shared command filter is unavailable")
        command = command_filter(command, None, float(now))
        guard_fault = components.session.evaluate_command_guard(command, None, float(now))
        if guard_fault is not None:
            raise RuntimeError(
                f"{guard_fault.code}: {guard_fault.message}"
            )
        components.session.dispatch_command(command)
    except Exception as exc:  # noqa: BLE001 - real startup motion must fail closed.
        message = f"启动前归位下发失败：{type(exc).__name__}: {exc}"
        _log("InitArmQpos", message, now=now)
        try:
            components.session.hold_hardware(message)
        finally:
            _cancel_special_arm_modes_for_hold(
                components,
                now=now,
                reason="startup_qpos_dispatch_failed",
            )
            clutch_input._cancel_init_arm_qpos_sequence(reason=message)
            clutch_input._publish_init_arm_state(message=message, control_hint=message)
        return True

    terminal = controller.consume_qpos_terminal_event()
    if terminal is not None:
        _finish_startup_gui_qpos_terminal(
            components=components,
            clutch_input=clutch_input,
            terminal=terminal,
            now=now,
        )
    return True


def _wait_for_real_teleop_start_request(
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    arm_mode: str,
    args: argparse.Namespace,
    control_ui: OperatorUiPort | None,
    ik_debug_log_recorder: IkDebugLogRecorder,
    ik_debug_profile_base_enabled: bool,
    visualization_request_handler: Callable[[float], None] | None = None,
    visualization_update_handler: Callable[[float], None] | None = None,
    runtime_status_update_handler: Callable[[], None] | None = None,
) -> str | None:
    if not clutch_input.enabled:
        _log(
            "Calibration",
            "stdin/gui is not interactive; starting runtime home calibration automatically.",
        )
        return "home"

    _log(
        "Calibration",
        "HOME CAPTURE IS ARMED. Use GUI/keyboard: p=home capture, q=quit. "
        "Pre-Home exact-qpos maintenance is available from the GUI.",
    )
    status_gate = ChangeGate()
    # The startup wait loop is not a control loop.  Before runtime Home is
    # captured, the state backends are still live because the GUI needs a
    # fresh qpos for an explicit Init/hold operation, but polling them every
    # 20 ms needlessly competes with PositionForceGuard's 20 Hz
    # ``Get_Force_Data`` calls on the same RM65 SDK connection.  The vendor
    # request/response API is serialized by the backend read lock; an
    # occasional delayed state response can therefore make the force sample
    # appear stale and trip the fail-closed watchdog.  Five Hz is sufficient
    # for the startup preview, while the explicit Init action still performs
    # its own fresh qpos read when requested.
    startup_state_sync_period_sec = 0.2
    last_startup_state_sync_monotonic: float | None = None
    clutch_input.set_startup_qpos_maintenance_enabled(True)
    try:
        initial_force_sides = _impedance_force_control_not_ready_sides(args, components)
        if initial_force_sides:
            clutch_input.publish_calibration_waiting(
                "力控尚未初始化。请先点击“开始遥操”初始化 "
                + "/".join(initial_force_sides)
                + " 力控；完成后可先执行机械臂精确归位，或再次点击“开始遥操”采集 runtime Home。"
            )
        while True:
            loop_start = time.time()
            clutch_input.poll()
            if bool(
                getattr(
                    clutch_input,
                    "consume_init_qpos_emergency_stop_request",
                    lambda: False,
                )()
            ):
                _handle_init_qpos_emergency_stop(
                    components,
                    clutch_input,
                    now=loop_start,
                )
                # Stay in the startup GUI wait loop.  Init emergency stop is
                # not a process exit and must not latch the arm backend.
                continue
            if runtime_status_update_handler is not None:
                runtime_status_update_handler()
            if visualization_request_handler is not None:
                visualization_request_handler(loop_start)
            _advance_startup_gui_qpos_maintenance(
                args=args,
                components=components,
                clutch_input=clutch_input,
                now=loop_start,
            )
            _handle_ik_debug_log_toggle_request(
                args=args,
                components=components,
                clutch_input=clutch_input,
                ik_debug_log_recorder=ik_debug_log_recorder,
                control_ui=control_ui,
                ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
                now=loop_start,
            )
            if clutch_input.should_quit:
                return None
            if clutch_input.consume_calibration_request():
                init_controller = getattr(components, "init_arm_controller", None)
                if bool(getattr(init_controller, "qpos_active", False)):
                    clutch_input.publish_calibration_waiting(
                        "机械臂归位仍在规划、执行或到位确认中；请等待归位结束后再点击“开始遥操”。"
                    )
                    continue
                force_sides = _impedance_force_control_not_ready_sides(args, components)
                if force_sides:
                    if _impedance_force_control_warmup_backends(args, components):
                        if not _warm_start_impedance_force_control_before_home(
                            args,
                            components,
                            clutch_input,
                        ):
                            if clutch_input.should_quit:
                                return None
                            continue
                    try:
                        _enable_impedance_force_control_after_runtime_home(
                            args,
                            components,
                            home_captured=False,
                        )
                    except Exception as exc:  # noqa: BLE001 - remain in startup Hold.
                        message = f"力控初始化失败：{type(exc).__name__}: {exc}"
                        _log("ForceControl", message, now=loop_start)
                        clutch_input.publish_calibration_waiting(
                            message + "；机械臂保持停止，请检查力控/传感器状态后重试。"
                        )
                    else:
                        clutch_input.publish_calibration_waiting(
                            "力控初始化完成。现在可以执行机械臂精确归位；"
                            "如无需归位，请再次点击“开始遥操”采集 runtime Home 并进入遥操。"
                        )
                    continue
                if clutch_input.before_calibration_start_delay(
                    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                    arm_mode=arm_mode,
                ):
                    return "home"
                if clutch_input.should_quit:
                    return None
                clutch_input.publish_calibration_waiting("Home 标定已取消，回到等待状态。")

            now_monotonic = time.monotonic()
            if (
                last_startup_state_sync_monotonic is None
                or now_monotonic - last_startup_state_sync_monotonic
                >= startup_state_sync_period_sec
            ):
                _sync_preview_from_real_state_real(components, required=False)
                last_startup_state_sync_monotonic = time.monotonic()

            channels, _ = components.session.poll_sources(loop_start)
            clutch_input.publish_raw_tracker_pose_state(channels=channels, now=loop_start)
            ready = {
                name: bool(channels.get(name) is not None and channels[name].valid)
                for name in tuple(f"{side}_wrist_pose" for side in components.active_sides)
            }
            if status_gate.should_emit("wait_for_real_start_status", tuple(sorted(ready.items())), now=loop_start):
                _log(
                    "Calibration",
                    f"waiting_for_gui tracker_ready={ready} ready_key='p'",
                    now=loop_start,
                )
            elif status_gate.should_emit(
                "wait_for_real_start_heartbeat",
                "waiting",
                now=loop_start,
                heartbeat_sec=3.0,
            ):
                _log(
                    "Calibration",
                    f"still_waiting ready={ready} ready_key='p'",
                    now=loop_start,
                )
            if visualization_update_handler is not None:
                visualization_update_handler(loop_start)
            time.sleep(0.02)
    finally:
        clutch_input.set_startup_qpos_maintenance_enabled(False)


def _runtime_home_failure_message(exc: BaseException) -> str:
    return f"runtime home 标定失败: {type(exc).__name__}: {exc}"


def _restore_init_arm_after_foot_pad_fallback(
    args: argparse.Namespace,
    *,
    live_foot_pad_count: int | None,
) -> tuple[bool, str]:
    """Restore a requested Init controller if an older caller disabled it.

    Pedal topology is intentionally ignored: pedals are only an additional
    input mapping and never own Fix/Init feature availability.
    """

    del live_foot_pad_count
    if bool(getattr(args, "init_arm_available", False)):
        return False, ""
    if not bool(getattr(args, "init_arm_requested", False)):
        return False, ""
    profile_text = str(getattr(args, "init_arm_profile", "") or "").strip()
    profile_path = _expand_cli_path(profile_text) if profile_text else None
    if profile_path is None or not profile_path.is_file():
        args.init_arm_disabled_reason = "init_arm_profile_unavailable"
        return (
            False,
            "Init 已请求，但 Init profile 不可用："
            f"{profile_text or '<empty>'}",
        )
    args.init_arm_available = True
    args.init_arm_disabled_reason = ""
    return True, "已恢复请求的机械臂精确归位功能；踏板状态不参与此功能门控。"


def _arm_after_start_calibration(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    home_captured: bool,
    position_force_guard_monitor: object | None = None,
    runtime_home_qpos_snapshot: Mapping[str, np.ndarray] | None = None,
) -> None:
    arm_after_start = time.perf_counter()
    reset_elapsed_ms = 0.0
    if home_captured:
        reset_start = time.perf_counter()
        try:
            synced = _reset_runtime_motion_state_from_real_qpos(
                components,
                required=True,
                qpos_snapshot=runtime_home_qpos_snapshot,
            )
        except Exception as exc:  # noqa: BLE001 - collision recovery remains in GUI Hold.
            guard_fault = (
                None
                if position_force_guard_monitor is None
                else getattr(position_force_guard_monitor, "fault", None)
            )
            if guard_fault is None:
                raise
            message = (
                "位控受力急停后的 runtime Home 复位失败，保持停臂："
                f"{type(exc).__name__}: {exc}"
            )
            _log("PositionForceGuard", message)
            clutch_input.publish_calibration_waiting(
                message + " 请检查机械臂状态后重试。"
            )
            return
        reset_elapsed_ms = (time.perf_counter() - reset_start) * 1000.0
        _log("Calibration", f"runtime restart motion state reset from live real qpos synced={synced}")
        if position_force_guard_monitor is not None:
            guard_fault = getattr(position_force_guard_monitor, "fault", None)
            rearm = getattr(
                position_force_guard_monitor,
                "rearm_after_manual_home",
                None,
            )
            if guard_fault is not None and callable(rearm):
                try:
                    rearm()
                except Exception as exc:  # noqa: BLE001 - remain in safe Hold.
                    message = (
                        "位控受力急停仍保持锁存，不能恢复遥操："
                        f"{type(exc).__name__}: {exc}"
                    )
                    _log("PositionForceGuard", message)
                    try:
                        components.session.hold_hardware(message)
                    except Exception as hold_exc:  # noqa: BLE001 - preserve first cause.
                        message += f"；Hold 下发失败: {type(hold_exc).__name__}: {hold_exc}"
                    clutch_input.publish_calibration_waiting(
                        message + " 请检查急停确认/机械臂状态后再重试。"
                    )
                    return
    state_machine_start = time.perf_counter()
    if home_captured:
        components.session.mark_home_recalibrated("RM65 runtime home captured")
    components.session.set_calibration_ready()
    components.session.mark_debug_passed()
    if components.position_reference_publisher is not None:
        components.position_reference_publisher.reset_epoch()
    state_machine_elapsed_ms = (time.perf_counter() - state_machine_start) * 1000.0
    force_control_start = time.perf_counter()
    _enable_impedance_force_control_after_runtime_home(
        args,
        components,
        home_captured=home_captured,
    )
    force_control_elapsed_ms = (time.perf_counter() - force_control_start) * 1000.0
    arm_hardware_start = time.perf_counter()
    components.session.arm_hardware()
    arm_hardware_elapsed_ms = (time.perf_counter() - arm_hardware_start) * 1000.0
    publish_start = time.perf_counter()
    clutch_input.publish_teleop_running(
        home_ready=home_captured,
        message=(
            "runtime home 完成，已进入真机遥操。"
            if home_captured
            else "runtime home 未就绪，未进入正常真机遥操。"
        ),
    )
    publish_elapsed_ms = (time.perf_counter() - publish_start) * 1000.0
    _log(
        "CalibrationTiming",
        f"arm_after_home total={(time.perf_counter() - arm_after_start) * 1000.0:.1f}ms "
        f"reset_motion={reset_elapsed_ms:.1f}ms "
        f"state_machine={state_machine_elapsed_ms:.1f}ms "
        f"force_control={force_control_elapsed_ms:.1f}ms "
        f"arm_hardware={arm_hardware_elapsed_ms:.1f}ms "
        f"publish_gui={publish_elapsed_ms:.1f}ms",
    )


def _update_init_arm_arrival_from_hardware(
    components: "RealTeleopComponents",
    *,
    hardware_state_by_side: Mapping[str, Mapping[str, object]],
    now: float,
) -> bool:
    """Feed measured TCP feedback into Init arrival and init-fixed promotion."""

    controller = getattr(components, "init_arm_controller", None)
    if controller is None or not bool(getattr(controller, "active", False)):
        return False
    side = str(getattr(controller, "active_side", "") or "")
    state = hardware_state_by_side.get(side)
    backend = (
        components.left_state_backend
        if side == "left"
        else components.right_state_backend
        if side == "right"
        else None
    )
    invalid_reason = ""
    pose_base = None
    timestamp = float(now)
    valid = False
    if not isinstance(state, Mapping) or backend is None:
        invalid_reason = "hardware_state_unavailable"
    elif not bool(state.get("hardware_read_success")):
        invalid_reason = "hardware_read_unsuccessful"
    elif not bool(state.get("hardware_measurement_valid")):
        invalid_reason = "hardware_measurement_invalid"
    else:
        try:
            qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(6)
            measurement_time = float(
                state.get("hardware_measurement_timestamp", state.get("hardware_refresh_time"))
            )
            if not np.all(np.isfinite(qpos)) or not np.isfinite(measurement_time):
                raise ValueError("non-finite measured qpos/timestamp")
            age_sec = max(0.0, float(now) - measurement_time)
            if age_sec > 0.2:
                invalid_reason = f"hardware_measurement_stale:{age_sec:.3f}s"
            else:
                pose_base = backend.compute_tcp_pose(qpos=qpos)
                timestamp = measurement_time
                valid = True
        except Exception as exc:  # noqa: BLE001 - UI-only feedback must never affect control.
            if not invalid_reason:
                invalid_reason = f"actual_tcp_unavailable:{type(exc).__name__}"
    controller.observe_actual_feedback(
        pose_base,
        timestamp=timestamp,
        valid=valid,
        invalid_reason=invalid_reason,
    )
    return True


__all__ = [
    "CAMERA_STARTUP_SELF_CHECK_DURATION_SEC",
    "CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC",
    "CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES",
    "CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC",
    "bind_legacy_support",
    "_update_init_arm_arrival_from_hardware",
    "_run_camera_startup_self_check",
    "_release_runtime_state_readers",
    "_log_dual_arm_readiness",
    "_startup_gui_qpos_seed_command",
    "_finish_startup_gui_qpos_terminal",
    "_advance_startup_gui_qpos_maintenance",
    "_wait_for_real_teleop_start_request",
    "_runtime_home_failure_message",
    "_restore_init_arm_after_foot_pad_fallback",
    "_arm_after_start_calibration",
]
