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
from ..log_utils import ChangeGate
from ..sources.camera import RealSenseRGBSource
from .camera_runtime import (
    CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
    CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
    CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
    CameraStartupFreshnessGate,
)

if TYPE_CHECKING:
    from ..operator_ui_ports import OperatorUiPort
    from .operator_control_runtime import RealTeleopControlInput
    from ..special_tasks.init_arm import InitArmCommandController
    from .components import RealTeleopComponents


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
    "_warm_start_impedance_force_control_before_home",
    "_with_expected_wrist_depth_camera_statuses",
)


_ACTIVE_STARTUP_CLEANUP: "StartupCleanupRegistry | None" = None


class StartupCleanupRegistry:
    """Release resources created before normal runtime shutdown takes ownership.

    The canonical runtime enters its complete shutdown ``finally`` only after
    the initial UI, inputs, and component bundle exist.  This small LIFO owner
    covers that startup preamble and is explicitly disarmed at the handoff to
    normal shutdown.
    """

    def __init__(self, *, logger: Callable[..., None] | None = None) -> None:
        self._steps: list[tuple[str, Callable[[], None]]] = []
        self._active = True
        self._logger = logger

    def add(self, label: str, callback: Callable[[], None]) -> None:
        if self._active and callable(callback):
            self._steps.append((str(label), callback))

    def disarm(self) -> None:
        global _ACTIVE_STARTUP_CLEANUP
        self._active = False
        if _ACTIVE_STARTUP_CLEANUP is self:
            _ACTIVE_STARTUP_CLEANUP = None
        self._steps.clear()

    def close(self) -> None:
        global _ACTIVE_STARTUP_CLEANUP
        if not self._active:
            return
        self._active = False
        if _ACTIVE_STARTUP_CLEANUP is self:
            _ACTIVE_STARTUP_CLEANUP = None
        failures: list[str] = []
        for label, callback in reversed(self._steps):
            try:
                callback()
            except BaseException as exc:  # noqa: BLE001 - never mask Ctrl+C.
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
        self._steps.clear()
        if failures and callable(self._logger):
            self._logger(
                "Shutdown",
                "startup Ctrl+C cleanup completed with recoverable failures: "
                + "; ".join(failures),
            )


def activate_startup_cleanup(
    *,
    logger: Callable[..., None] | None = None,
) -> StartupCleanupRegistry:
    """Activate the sole early-startup cleanup owner for this process."""

    global _ACTIVE_STARTUP_CLEANUP
    if _ACTIVE_STARTUP_CLEANUP is not None:
        _ACTIVE_STARTUP_CLEANUP.close()
    registry = StartupCleanupRegistry(logger=logger)
    _ACTIVE_STARTUP_CLEANUP = registry
    return registry


def deactivate_startup_cleanup() -> None:
    """Transfer all registered resources to normal runtime shutdown."""

    registry = _ACTIVE_STARTUP_CLEANUP
    if registry is not None:
        registry.disarm()


def cleanup_active_startup() -> None:
    """Close resources created before normal runtime shutdown takes ownership."""

    registry = _ACTIVE_STARTUP_CLEANUP
    if registry is not None:
        registry.close()


def register_component_startup_cleanup(
    registry: StartupCleanupRegistry,
    components: object,
) -> None:
    """Register only idempotent component close hooks for early cancellation."""

    for side in ("right", "left"):
        state_backend = getattr(components, f"{side}_state_backend", None)
        if state_backend is not None:
            registry.add(f"{side}_state_backend", state_backend.close)
        hand_backend = getattr(components, f"{side}_hand_backend", None)
        if hand_backend is not None:
            registry.add(f"{side}_hand_backend", hand_backend.close)
    executor = getattr(components, "arm_state_read_executor", None)
    if executor is not None:
        registry.add(
            "arm_state_read_executor",
            lambda executor=executor: executor.shutdown(wait=True),
        )
    session = getattr(components, "session", None)
    if session is not None:
        registry.add("teleop_session", session.close)


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
    poll_input: bool = True,
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
    freshness_gate = CameraStartupFreshnessGate(
        tuple(str(camera_key) for camera_key in camera_sources)
    )
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
        early_success = freshness_gate.update(status)
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
                "fresh_frame_streaks": dict(freshness_gate.fresh_frame_streaks),
            }
        )
        if clutch_input.ui is not None:
            updates: dict[str, object] = {"camera_self_check": dict(final_payload)}
            if completed and final_payload.get("status") in {"failed", "warning"}:
                updates["control_hint"] = str(final_payload.get("message") or "")
            clutch_input.ui.publish(**updates)
        if completed:
            break
        if poll_input:
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
    begin_home_transition_handler: Callable[[], None] | None = None,
    require_home_transition_ready_handler: Callable[[], None] | None = None,
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
    # The startup wait loop is not a control loop.  Keep the preview on the
    # startup snapshot while the operator is idle instead of periodically
    # calling Get_Current_Arm_State on the state connection.  That connection
    # also serves PositionForceGuard's 20 Hz Get_Force_Data calls, and one slow
    # state response can otherwise hold the serialized SDK lock beyond the
    # 250 ms stale deadline and cause a false fail-closed trip.  Explicit Init
    # reads fresh qpos before planning and during feedback checks; runtime Home
    # captures its own final atomic dual-arm hardware snapshot.
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
                if begin_home_transition_handler is not None:
                    try:
                        begin_home_transition_handler()
                    except Exception as exc:  # noqa: BLE001 - remain in startup Hold.
                        message = (
                            "Home 前后台维护停止请求失败："
                            f"{type(exc).__name__}: {exc}"
                        )
                        _log("TactileCompression", message, now=loop_start)
                        clutch_input.publish_calibration_waiting(
                            message + "；保持停臂，请重试。"
                        )
                        continue
                if clutch_input.before_calibration_start_delay(
                    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                    arm_mode=arm_mode,
                ):
                    if require_home_transition_ready_handler is not None:
                        try:
                            require_home_transition_ready_handler()
                        except Exception as exc:  # noqa: BLE001 - remain in startup Hold.
                            message = (
                                "Home 前后台维护未能安全停止："
                                f"{type(exc).__name__}: {exc}"
                            )
                            _log("TactileCompression", message, now=time.time())
                            clutch_input.publish_calibration_waiting(
                                message + "；保持停臂，请重试。"
                            )
                            continue
                    return "home"
                if clutch_input.should_quit:
                    return None
                clutch_input.publish_calibration_waiting("Home 标定已取消，回到等待状态。")

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
    runtime_home_capture_sides: tuple[str, ...] | None = None,
) -> bool:
    """Try to admit real-arm control after Home and report the outcome.

    ``home_captured`` only describes the calibration input.  A ``True``
    receipt additionally means every pre-arm continuity/safety gate completed,
    the Session reached ARMED, and the application projection was published.
    Failures that are deliberately consumed here return ``False`` after
    keeping the runtime in Hold; callers must not commit a profile/progress
    transition from the Home receipt alone.
    """
    arm_after_start = time.perf_counter()
    reset_elapsed_ms = 0.0

    def reject_arm_admission(tag: str, message: str) -> bool:
        """Keep motion blocked when a post-Home admission step fails."""

        _log(tag, message)
        try:
            components.session.hold_hardware(message)
        except Exception as hold_exc:  # noqa: BLE001 - preserve the admission cause.
            _log(
                tag,
                "arm-admission Hold failed after the original error: "
                f"{type(hold_exc).__name__}: {hold_exc}",
            )
        try:
            clutch_input.publish_calibration_waiting(message)
        except Exception as publish_exc:  # noqa: BLE001 - UI cannot reopen motion.
            _log(
                tag,
                "arm-admission failure UI publish failed: "
                f"{type(publish_exc).__name__}: {publish_exc}",
            )
        return False

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
            return False
        reset_elapsed_ms = (time.perf_counter() - reset_start) * 1000.0
        _log("Calibration", f"runtime restart motion state reset from live real qpos synced={synced}")
        startup_continuity_gate = getattr(
            components,
            "startup_home_continuity_gate",
            None,
        )
        if startup_continuity_gate is not None:
            try:
                continuity_snapshot = startup_continuity_gate.arm_from_home(
                    runtime_home_qpos_snapshot or {}
                )
            except Exception as exc:  # noqa: BLE001 - startup admission fails closed.
                message = (
                    "runtime Home 首帧连续性闸门建立失败，保持停臂："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("StartupHomeContinuity", message)
                components.session.hold_hardware(message)
                clutch_input.publish_calibration_waiting(message)
                return False
            _log(
                "StartupHomeContinuity",
                f"armed receipt={continuity_snapshot}",
            )
        endpoint_guard = getattr(
            components,
            "position_admittance_endpoint_guard",
            None,
        )
        if endpoint_guard is not None:
            try:
                endpoint_guard_snapshot = endpoint_guard.reset_anchor_from_qpos(
                    runtime_home_qpos_snapshot
                )
            except Exception as exc:  # noqa: BLE001 - Active guard fails closed.
                return reject_arm_admission(
                    "PositionAdmittance",
                    "Active endpoint excursion guard 无法绑定 runtime Home，"
                    "保持停臂："
                    f"{type(exc).__name__}: {exc}",
                )
            _log(
                "PositionAdmittance",
                "Active endpoint excursion guard armed from runtime Home: "
                f"{endpoint_guard_snapshot}",
            )
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
                    return False
    state_machine_start = time.perf_counter()
    reference_epoch: int | None = None
    if components.position_reference_publisher is not None:
        try:
            reference_epoch = components.position_reference_publisher.reset_epoch()
        except Exception as exc:  # noqa: BLE001 - reference reset is authoritative.
            return reject_arm_admission(
                "PositionReference",
                "runtime Home 后位置参考 epoch 复位失败，保持停臂："
                f"{type(exc).__name__}: {exc}",
            )
    retargeter = getattr(components, "dual_arm_retargeter", None)
    safety_filter = (
        None
        if retargeter is None
        else getattr(retargeter, "pre_ik_target_safety_filter", None)
    )
    anchor_required_sides = tuple(
        getattr(
            getattr(safety_filter, "profile", None),
            "anchor_required_sides",
            (),
        )
    )
    safety_filter_active = bool(
        getattr(safety_filter, "apply_to_target", False)
    )
    if home_captured:
        rebind_from_home = getattr(
            safety_filter,
            "rebind_from_calibration_home",
            None,
        )
        if callable(rebind_from_home):
            safety_epoch = (
                int(reference_epoch)
                if reference_epoch is not None
                else int(getattr(safety_filter, "execution_epoch", 0)) + 1
            )
            try:
                anchor = rebind_from_home(
                    execution_epoch=safety_epoch,
                    captured_at=time.monotonic(),
                    captured_sides=(
                        tuple(components.active_sides)
                        if runtime_home_capture_sides is None
                        else tuple(runtime_home_capture_sides)
                    ),
                    anchor_id=f"runtime_home:{safety_epoch}",
                    clock_domain_id="python_monotonic",
                )
            except Exception as exc:  # noqa: BLE001 - Active fails closed.
                if safety_filter_active:
                    message = (
                        "TCP Pose 安全运行区间绑定 runtime Home 失败，保持停臂："
                        f"{type(exc).__name__}: {exc}"
                    )
                    _log("TCPPoseEnvelope", message)
                    components.session.hold_hardware(message)
                    clutch_input.publish_calibration_waiting(message)
                    return False
                _log(
                    "TCPPoseEnvelope",
                    "Shadow runtime Home binding failed; observation remains "
                    "non-authoritative and hardware lifecycle is unchanged: "
                    f"{type(exc).__name__}: {exc}",
                )
                anchor = None
            _log(
                "TCPPoseEnvelope",
                "runtime Home bound before arm "
                f"epoch={safety_epoch} "
                f"sides={list(runtime_home_capture_sides or components.active_sides)} "
                f"anchor_hash={None if anchor is None else anchor.semantic_content_hash}",
            )
        try:
            components.session.mark_home_recalibrated(
                "RM65 runtime home captured"
            )
        except Exception as exc:  # noqa: BLE001 - Session must own the receipt.
            return reject_arm_admission(
                "Calibration",
                "runtime Home 无法写入 Session 校准回执，保持停臂："
                f"{type(exc).__name__}: {exc}",
            )
    elif anchor_required_sides and safety_filter_active:
        message = (
            "TCP Pose 相对安全运行区间需要本次运行的 runtime Home；"
            "当前未捕获 Home，拒绝真机 arm。"
        )
        _log("TCPPoseEnvelope", message)
        components.session.hold_hardware(message)
        clutch_input.publish_calibration_waiting(message)
        return False
    try:
        components.session.set_calibration_ready()
        components.session.mark_debug_passed()
    except Exception as exc:  # noqa: BLE001 - invalid lifecycle cannot arm.
        return reject_arm_admission(
            "Calibration",
            "runtime Home 后 Session 校准状态迁移失败，保持停臂："
            f"{type(exc).__name__}: {exc}",
        )
    state_machine_elapsed_ms = (time.perf_counter() - state_machine_start) * 1000.0
    force_control_start = time.perf_counter()
    try:
        _enable_impedance_force_control_after_runtime_home(
            args,
            components,
            home_captured=home_captured,
        )
    except Exception as exc:  # noqa: BLE001 - partial force admission stays held.
        return reject_arm_admission(
            "ForceControl",
            "runtime Home 后力控接入失败，保持停臂："
            f"{type(exc).__name__}: {exc}",
        )
    force_control_elapsed_ms = (time.perf_counter() - force_control_start) * 1000.0
    arm_hardware_start = time.perf_counter()
    try:
        components.session.arm_hardware()
    except Exception as exc:  # noqa: BLE001 - rig/Session admission is recoverable Hold.
        return reject_arm_admission(
            "Calibration",
            "runtime Home 已完成但机械臂未通过 Session arm admission，保持停臂："
            f"{type(exc).__name__}: {exc}",
        )
    arm_hardware_elapsed_ms = (time.perf_counter() - arm_hardware_start) * 1000.0
    publish_start = time.perf_counter()
    try:
        clutch_input.publish_teleop_running(
            home_ready=home_captured,
            message=(
                (
                    "runtime home 完成，正在验证首帧连续性；通过前保持不下发机械臂命令。"
                    if getattr(components, "startup_home_continuity_gate", None)
                    is not None
                    else "runtime home 完成，已进入真机遥操。"
                )
                if home_captured
                else "runtime home 未就绪，未进入正常真机遥操。"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - application projection is required.
        return reject_arm_admission(
            "WebGUI",
            "Session 已 arm 但应用遥操状态发布失败，已重新 Hold："
            f"{type(exc).__name__}: {exc}",
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
    return True


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
    "StartupCleanupRegistry",
    "activate_startup_cleanup",
    "bind_legacy_support",
    "cleanup_active_startup",
    "deactivate_startup_cleanup",
    "register_component_startup_cleanup",
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
