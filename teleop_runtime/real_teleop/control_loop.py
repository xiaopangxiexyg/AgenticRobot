"""The canonical real-teleop per-frame control loop.

Resource construction, device startup, and finally cleanup remain in
`runtime.py`.  This module owns only the ordered control-frame cycle.  Mutable
outer lifecycle values are read through explicit accessors so device retry,
recording admission, and preview toggles retain same-frame visibility.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import time
from typing import Any

import numpy as np
from dataclasses import replace


_REQUIRED_SUPPORT_NAMES = (
    "AsyncDMTacDistributedForceVisualizer",
    "DEFAULT_LEFT_ARM_QPOS",
    "DEFAULT_RIGHT_ARM_QPOS",
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "DMTacDistributedForceVisualizer",
    "PhaseTimer",
    "TRACKER_HOLD_AUTOMATIC",
    "TRACKER_HOLD_GRACE",
    "TeleopRunState",
    "_active_command_backend",
    "_advance_recovery_plan",
    "_advance_startup_gui_qpos_maintenance",
    "_apply_active_arm_command_to_preview_real",
    "_arm_after_start_calibration",
    "_build_action_debug_metadata",
    "_build_debug_timing_log_frame",
    "_build_episode_frame",
    "_build_ik_debug_log_frame",
    "_build_ik_timing_payload",
    "_build_ik_warning_payload",
    "_build_pose_quality_display_payload",
    "_build_renderer_debug_boxes",
    "_build_renderer_debug_frames",
    "_build_runtime_hand_command_for_recovery",
    "_cancel_special_arm_modes_for_hold",
    "_channel_pose_in_desk",
    "_channels_log_payload",
    "_collection_sample_step_result",
    "_collection_step_result",
    "_command_backend_health_signature",
    "_command_backend_status_text",
    "_copy_teleop_command",
    "_debug_timing_maybe_report",
    "_disengage_position_backends_for_manual_interaction",
    "_dispatch_q6_gripper_control",
    "_emergency_stop_real_arms",
    "_emit_home_alignment_logs",
    "_emit_runtime_orientation_logs",
    "_emit_tactile_sum_only_status",
    "_enable_gui_tactile_solving",
    "_extract_hand_command_from_command",
    "_faults_log_payload",
    "_filter_manual_real_command",
    "_fix_arm_debug_state",
    "_fix_arm_status_text",
    "_fix_arm_ui_state",
    "_force_guard_recording_stop_context",
    "_gripper_backend_for_side",
    "_gripper_runtime_health_signature",
    "_gripper_runtime_status_signature",
    "_guard_impedance_force_control",
    "_hand_backend_for_side",
    "_hand_channel_status",
    "_hand_tactile_sum",
    "_handle_ik_debug_log_toggle_request",
    "_handle_init_qpos_emergency_stop",
    "_hold_requested_pause_reason",
    "_ik_timing_status_text",
    "_init_arm_debug_state",
    "_init_arm_status_text",
    "_init_arm_ui_state",
    "_log",
    "_mean_pose",
    "_merge_runtime_hand_command",
    "_movement_multiple_ui_state",
    "_movement_multiple_value",
    "_pose_quality_reason_list",
    "_pose_quality_status_text",
    "_real_hand_render_qpos",
    "_record_position_loop_timings",
    "_release_gui_qpos_init_transit",
    "_resolve_robot_home_for_side",
    "_runtime_arm_backend_state",
    "_runtime_fault_label",
    "_runtime_home_calibration",
    "_runtime_home_failure_message",
    "_short_status_text",
    "_side_retargeter",
    "_start_recovery_plan",
    "_state_qpos",
    "_sync_dual_arm_retargeter_for_real_control",
    "_sync_real_execution_step_limits",
    "_tactile_solving_active",
    "_tactile_visualization_available",
    "_temporary_safety_health_signature",
    "_update_init_arm_arrival_from_hardware",
    "advance_fix_arm_home_capture",
    "advance_init_arm_home_capture",
    "build_fix_arm_pose_snapshot_payload",
    "dmtac_visualization_snapshot_from_backends",
    "merge_teleop_commands",
    "tactile_by_side_from_backends",
)


class _LegacyFunctionProxy:
    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive control-loop support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    missing = [name for name in _REQUIRED_SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "real-teleop control-loop support is incomplete: "
            + ", ".join(sorted(missing))
        )
    globals().update(
        {
            name: (
                _LegacyFunctionProxy(namespace, name)
                if inspect.isroutine(namespace[name])
                else namespace[name]
            )
            for name in _REQUIRED_SUPPORT_NAMES
        }
    )


@dataclass(frozen=True)
class RealTeleopControlLoopContext:
    action_debug_log_recorder: Any
    advance_recording_countdown: Callable[..., None]
    args: Any
    camera_preview_rate: Any
    camera_sources: Mapping[str, Any]
    cancel_recording_countdown: Callable[..., None]
    close_camera_preview: Callable[..., None]
    close_tactile_preview: Callable[..., None]
    clutch_input: Any
    components: Any
    console_stream: Any
    control_ui: Any
    debug_timing_log_recorder: Any
    debug_timing_modules: Any
    debug_timing_state: Any
    episode_recorder: Any
    fix_hold_coordinator: Any
    gripper_warmup_probe_running: Callable[[], bool]
    ik_debug_log_recorder: Any
    ik_debug_profile_base_enabled: bool
    main_loop_watchdog: Any
    open_camera_preview: Callable[[float], None]
    open_tactile_preview: Callable[[float], None]
    orientation_gate: Any
    outer_control_hz: float
    pause_recording: Callable[..., None]
    poll_dataset_finalize: Callable[[float], None]
    pose_snapshot_recorder: Any
    position_audit_fanout: Any
    position_force_guard_monitor: Any
    preview_camera_sources: Mapping[str, Any]
    publish_collection_frequency_warning: Callable[..., None]
    recording_active: Callable[[], bool]
    recording_auto_stop_reason: Callable[[Any], str | None]
    recording_frame_count: Callable[[], int]
    recording_frequency_status: Callable[[], Mapping[str, object]]
    recording_path: Callable[[], str]
    recording_pause_reason: Callable[[], str]
    recording_paused: Callable[[], bool]
    recording_warmup_status: Callable[..., tuple[bool, str]]
    resume_recording_if_ready: Callable[..., None]
    retry_failed_devices: Callable[[], None]
    sample_rate_gate: Any
    schedule_recording_countdown: Callable[..., None]
    status_gate: Any
    status_heartbeat_sec: float
    stop_recording: Callable[..., None]
    sync_fix_hold: Callable[[float], None]
    sync_foot_pad_runtime_status: Callable[[], None]
    tactile_preview_rate: Any
    tactile_sum_gate: Any
    dataset_recorder: Callable[[], Any]
    visualization_state: Callable[[], tuple[Any, bool, Any]]
    episode_start_countdown_deadline: Callable[[], float | None]
    episode_last_state: Callable[[], str | None]
    set_episode_last_state: Callable[[str | None], None]
    step_index: Callable[[], int]
    set_step_index: Callable[[int], None]


def run_real_teleop_control_loop(context: RealTeleopControlLoopContext) -> None:
    action_debug_log_recorder = context.action_debug_log_recorder
    advance_recording_countdown = context.advance_recording_countdown
    args = context.args
    camera_preview_rate = context.camera_preview_rate
    camera_sources = context.camera_sources
    cancel_recording_countdown = context.cancel_recording_countdown
    close_camera_preview = context.close_camera_preview
    close_tactile_preview = context.close_tactile_preview
    clutch_input = context.clutch_input
    components = context.components
    console_stream = context.console_stream
    control_ui = context.control_ui
    debug_timing_log_recorder = context.debug_timing_log_recorder
    debug_timing_modules = context.debug_timing_modules
    debug_timing_state = context.debug_timing_state
    episode_recorder = context.episode_recorder
    fix_hold_coordinator = context.fix_hold_coordinator
    gripper_warmup_probe_running = context.gripper_warmup_probe_running
    ik_debug_log_recorder = context.ik_debug_log_recorder
    ik_debug_profile_base_enabled = context.ik_debug_profile_base_enabled
    main_loop_watchdog = context.main_loop_watchdog
    open_camera_preview = context.open_camera_preview
    open_tactile_preview = context.open_tactile_preview
    orientation_gate = context.orientation_gate
    outer_control_hz = context.outer_control_hz
    pause_recording = context.pause_recording
    poll_dataset_finalize = context.poll_dataset_finalize
    pose_snapshot_recorder = context.pose_snapshot_recorder
    position_audit_fanout = context.position_audit_fanout
    position_force_guard_monitor = context.position_force_guard_monitor
    preview_camera_sources = context.preview_camera_sources
    publish_collection_frequency_warning = context.publish_collection_frequency_warning
    recording_active = context.recording_active
    recording_auto_stop_reason = context.recording_auto_stop_reason
    recording_frame_count = context.recording_frame_count
    recording_frequency_status = context.recording_frequency_status
    recording_path = context.recording_path
    recording_pause_reason = context.recording_pause_reason
    recording_paused = context.recording_paused
    recording_warmup_status = context.recording_warmup_status
    resume_recording_if_ready = context.resume_recording_if_ready
    retry_failed_devices = context.retry_failed_devices
    sample_rate_gate = context.sample_rate_gate
    schedule_recording_countdown = context.schedule_recording_countdown
    status_gate = context.status_gate
    status_heartbeat_sec = context.status_heartbeat_sec
    stop_recording = context.stop_recording
    sync_fix_hold = context.sync_fix_hold
    sync_foot_pad_runtime_status = context.sync_foot_pad_runtime_status
    tactile_preview_rate = context.tactile_preview_rate
    tactile_sum_gate = context.tactile_sum_gate

    dataset_recorder = context.dataset_recorder()
    camera_previewer, camera_preview_active, tactile_visualizer = (
        context.visualization_state()
    )
    episode_last_state = context.episode_last_state()
    step_index = context.step_index()
    recovery_plan = None
    last_dispatched_main_command = None

    main_loop_watchdog.start()
    emergency_stop_sent = False
    while True:
        loop_start = time.time()
        dataset_recorder = context.dataset_recorder()
        camera_previewer, camera_preview_active, tactile_visualizer = (
            context.visualization_state()
        )
        episode_last_state = context.episode_last_state()
        main_loop_watchdog.beat("loop.start", step_index)
        loop_phase_timer = (
            PhaseTimer()
            if (
                debug_timing_modules.needs_loop_phase_timer
                or recording_active()
                or action_debug_log_recorder.active
                or ik_debug_log_recorder.active
                or position_audit_fanout is not None
            )
            else None
        )

        def mark_loop_phase(name: str) -> None:
            main_loop_watchdog.beat(name, step_index)
            if loop_phase_timer is not None:
                loop_phase_timer.mark(name)

        clutch_input.poll()
        mark_loop_phase("loop.control_input_poll")
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
            # Do not run a normal control frame after the Init operation
            # has been stopped by the operator.
            continue
        sync_foot_pad_runtime_status()
        poll_dataset_finalize(loop_start)
        plus_request = clutch_input.consume_plus_curation_request()
        if plus_request is not None:
            # This is a tiny post-finalize JSON sidecar write.  It never
            # enters the raw writer or issues a hardware/control call.
            clutch_input.save_plus_curation(
                episode_index=int(plus_request.get("episode_index")),
                pending_token=str(plus_request.get("pending_token", "") or ""),
                disposition=str(plus_request.get("disposition", "unknown")),
                reason_code=str(plus_request.get("reason_code", "unknown")),
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
        if clutch_input.consume_device_retry_request():
            retry_failed_devices()
        dataset_recorder = context.dataset_recorder()
        if clutch_input.consume_camera_visualization_toggle_request():
            if clutch_input._episode_busy():
                message = (
                    "数采倒计时、录制、暂停或保存期间不能切换图像窗口；"
                    "请等待 episode 完全结束。"
                )
                clutch_input.set_visualization_state(status=message)
                if control_ui is not None:
                    control_ui.append_message("Camera", message)
            elif camera_preview_active or camera_previewer is not None:
                close_camera_preview("图像窗口已关闭")
                if control_ui is not None:
                    control_ui.append_message("Camera", "图像窗口已关闭。")
            else:
                open_camera_preview(loop_start)
        if clutch_input.consume_tactile_solving_enable_request():
            if bool(getattr(clutch_input, "_hold_requested", False)):
                message = "Hold 正在生效，暂不重启触觉 worker；请在停臂完成后重试。"
                clutch_input.set_visualization_state(status=message)
                if control_ui is not None:
                    control_ui.append_message("TactileSolve", message)
            elif recording_active() or context.episode_start_countdown_deadline() is not None:
                message = "数采或开始倒计时期间不能切换触觉解算；请停止数采并进入 Hold。"
                clutch_input.set_visualization_state(status=message)
                if control_ui is not None:
                    control_ui.append_message("TactileSolve", message)
            elif not clutch_input.hold_active() and bool(
                getattr(clutch_input, "_teleop_started", False)
            ):
                message = "开启触觉解算需要先进入 Hold，避免重启触觉 worker 影响控制周期。"
                clutch_input.set_visualization_state(status=message)
                if control_ui is not None:
                    control_ui.append_message("TactileSolve", message)
            else:
                try:
                    changed = _enable_gui_tactile_solving(args, components)
                    message = (
                        f"触觉解算已开启：{changed} 个 worker 已按固定传感器标定重启；"
                        "解算保持开启，关闭窗口不会改变采集字段。"
                    )
                    clutch_input.set_visualization_state(
                        tactile_solving_active=True,
                        status=message,
                    )
                    if control_ui is not None:
                        control_ui.append_message("TactileSolve", message)
                    _log("TactileSolve", message, now=loop_start)
                except Exception as exc:  # noqa: BLE001
                    message = f"触觉解算开启失败: {type(exc).__name__}: {exc}"
                    args.tactile_solving_active = _tactile_solving_active(args, components)
                    args.tactile_visualization_available = _tactile_visualization_available(
                        args, components
                    )
                    clutch_input.set_visualization_state(
                        tactile_solving_active=args.tactile_solving_active,
                        status=message,
                    )
                    if control_ui is not None:
                        control_ui.append_message("TactileSolve", message)
        if clutch_input.consume_tactile_visualization_toggle_request():
            if clutch_input._episode_busy():
                message = (
                    "数采倒计时、录制、暂停或保存期间不能切换触觉窗口；"
                    "请等待 episode 完全结束。"
                )
                clutch_input.set_visualization_state(status=message)
                if control_ui is not None:
                    control_ui.append_message("TactileVis", message)
            elif tactile_visualizer is not None:
                close_tactile_preview("触觉窗口已关闭")
                if control_ui is not None:
                    control_ui.append_message("TactileVis", "触觉窗口已关闭。")
            else:
                open_tactile_preview(loop_start)
        camera_previewer, camera_preview_active, tactile_visualizer = (
            context.visualization_state()
        )
        sync_fix_hold(loop_start)
        if clutch_input.should_quit:
            if not emergency_stop_sent:
                _emergency_stop_real_arms(
                    components,
                    reason="quit_requested",
                    now=loop_start,
                )
                emergency_stop_sent = True
            cancel_recording_countdown(loop_start, reason="quit_requested")
            if recording_active():
                force_guard_fault = (
                    None
                    if position_force_guard_monitor is None
                    else position_force_guard_monitor.fault
                )
                recording_end_reason, recording_summary = (
                    _force_guard_recording_stop_context(
                        force_guard_fault,
                        fallback_end_reason="quit_requested",
                        fallback_summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                )
                stop_recording(
                    loop_start,
                    end_reason=recording_end_reason,
                    summary=recording_summary,
                )
            break
        if clutch_input.consume_hold_request():
            hold_reason = clutch_input.hold_request_reason()
            fix_hold_escalation = fix_hold_coordinator.escalate_to_regular_hold()
            if fix_hold_escalation.changed:
                clutch_input.set_fix_hold_active(
                    False,
                    message=(
                        "Fix-Hold 已由用户升级为常规 Hold；"
                        "特殊固定状态将清除，之后必须重采双臂 Home。"
                    ),
                )
                if episode_recorder.is_recording:
                    episode_recorder.record_event(
                        "fix_hold_escalated_to_regular_hold",
                        payload={
                            "step_index": int(step_index),
                            "locked_sides": list(fix_hold_escalation.locked_sides),
                            "source_by_side": dict(
                                fix_hold_escalation.source_by_side
                            ),
                        },
                        timestamp=loop_start,
                    )
            if hold_reason == "q6_gesture":
                hold_hardware_reason = "q6 gesture hold requested"
            elif hold_reason == "tracker_loss_timeout":
                hold_hardware_reason = "tracker target loss requires runtime home recalibration"
            elif hold_reason.startswith("init_qpos_"):
                hold_hardware_reason = (
                    "Init qpos operation completed; runtime home recalibration required"
                    if hold_reason in {
                        "init_qpos_completed",
                        "init_qpos_sequence_completed",
                    }
                    else "Init qpos operation failed; safe Hold and runtime home recalibration required"
                )
            else:
                hold_hardware_reason = "gui hold requested"
            if recording_active():
                pause_recording(
                    loop_start,
                    reason=_hold_requested_pause_reason(hold_reason),
                )
            special_cancel_messages = _cancel_special_arm_modes_for_hold(
                components,
                now=loop_start,
                reason="hold_requested",
            )
            cancel_recording_countdown(loop_start, reason="hold_requested")
            recovery_plan = None
            if not components.session.home_recalibration_required:
                components.session.hold_hardware(hold_hardware_reason)
            _disengage_position_backends_for_manual_interaction(
                components,
                now=loop_start,
                reason=hold_hardware_reason,
            )
            _log(
                "Hold",
                f"{hold_hardware_reason}; waiting for explicit runtime home calibration",
                now=loop_start,
            )
            if control_ui is not None:
                if hold_reason == "q6_gesture":
                    message = "机械臂已进入保持状态；松开双手后三维到释放区间 1 秒，再次握紧 1 秒后会重新采集 runtime home。"
                elif hold_reason == "tracker_loss_timeout":
                    message = "Tracker 目标持续丢失已进入手动 Hold；必须点击“开始遥操”重新采集 runtime home 后才能恢复。"
                elif hold_reason.startswith("init_qpos_"):
                    message = (
                        "Init qpos 已完成并进入 Hold；请摆好 VIVE/Q6 后点击“开始遥操”重新采集 runtime Home。"
                        if hold_reason in {
                            "init_qpos_completed",
                            "init_qpos_sequence_completed",
                        }
                        else "Init qpos 未安全完成，已进入 Hold；请检查提示和机械臂状态，确认后重新采集 runtime Home。"
                    )
                else:
                    message = "机械臂已进入保持状态；点击“开始遥操”后才会重新采集 runtime home。"
                control_ui.append_message("Hold", message)
                if special_cancel_messages:
                    control_ui.append_message(
                        "Hold",
                        "；".join(special_cancel_messages)
                        + "；重新采集 runtime home 后恢复普通遥操。",
                    )
            continue
        fast_two_pedal_home, fast_two_pedal_fixed_side = (
            clutch_input.consume_two_pedal_fast_home_target(now=time.time())
        )
        if fast_two_pedal_home:
            if recording_active():
                pause_recording(
                    loop_start,
                    reason="two_pedal_fast_home_transition",
                )
            _cancel_special_arm_modes_for_hold(
                components,
                now=loop_start,
                reason="two_pedal_fast_home_transition",
            )
            recovery_plan = None
            try:
                _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=clutch_input.two_pedal_fast_home_timeout_sec(),
                    num_samples=1,
                    progress_callback=clutch_input.publish_runtime_home_progress,
                    require_stable=True,
                    # When entering a selected single-arm mode from Hold,
                    # only the arm that will become fixed needs a fresh Home.
                    # The other arm keeps its existing runtime anchor and must
                    # not block this transition merely because its tracker is
                    # still moving.
                    capture_sides=(
                        (fast_two_pedal_fixed_side,)
                        if fast_two_pedal_fixed_side is not None
                        else None
                    ),
                    stability_profile="transition",
                )
            except Exception as exc:  # noqa: BLE001
                message = (
                    "双踏板切换失败：1 秒内未获得稳定、同步的 runtime Home；"
                    f"{type(exc).__name__}: {exc}"
                )
                components.session.hold_hardware(message)
                _disengage_position_backends_for_manual_interaction(
                    components,
                    now=loop_start,
                    reason=message,
                )
                clutch_input.finish_two_pedal_fast_home(
                    success=False,
                    now=loop_start,
                    message=message,
                )
                continue
            _emit_home_alignment_logs(components)
            # Preserve the requested single-arm target until the first
            # fresh post-Home control frame has populated Fix snapshots.
            clutch_input._two_pedal_transition_target_fixed_side = (
                fast_two_pedal_fixed_side
            )
            _arm_after_start_calibration(
                args,
                components,
                clutch_input,
                home_captured=True,
            )
            clutch_input.finish_two_pedal_fast_home(
                success=True,
                now=loop_start,
                message="双踏板快速 Home 已完成，正在无缝接入目标遥操状态。",
            )
            continue
        if clutch_input.consume_calibration_request():
            if recording_active():
                pause_recording(loop_start, reason="runtime_recalibration_requested")
            special_cancel_messages = _cancel_special_arm_modes_for_hold(
                components,
                now=loop_start,
                reason="runtime_recalibration_requested",
            )
            cancel_recording_countdown(loop_start, reason="runtime_recalibration_requested")
            recovery_plan = None
            if components.session.state_machine.state != TeleopRunState.DEBUG_ONLY:
                components.session.hold_hardware("runtime home recalibration requested")
            _disengage_position_backends_for_manual_interaction(
                components,
                now=loop_start,
                reason="runtime home recalibration requested",
            )
            _log("Calibration", "explicit runtime home recalibration requested from hold", now=loop_start)
            if control_ui is not None and special_cancel_messages:
                control_ui.append_message(
                    "Calibration",
                    "；".join(special_cancel_messages)
                    + "；本次 runtime home 后特殊状态保持关闭。",
                )
            if not clutch_input.before_calibration_start_delay(
                DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                arm_mode="real",
            ):
                continue
            try:
                _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=args.calibration_timeout,
                    num_samples=args.calibration_samples,
                    progress_callback=clutch_input.publish_runtime_home_progress,
                )
            except Exception as exc:  # noqa: BLE001
                message = _runtime_home_failure_message(exc)
                _log("Calibration", message, now=loop_start)
                clutch_input.publish_calibration_waiting(
                    f"{message}\n请检查 VIVE sender/tracker_role 后重新选择操作。"
                )
                continue
            _emit_home_alignment_logs(components)
            _arm_after_start_calibration(
                args,
                components,
                clutch_input,
                home_captured=True,
            )
            continue
        if clutch_input.consume_episode_stop_request():
            if recording_active():
                stop_recording(
                    loop_start,
                    end_reason="manual_key_stop",
                    summary={
                        "last_step_index": step_index,
                        "last_state": episode_last_state,
                    },
                )
            else:
                cancel_recording_countdown(loop_start, reason="manual_key_stop")
        episode_delete_index = clutch_input.consume_episode_delete_request()
        if episode_delete_index is not None:
            if clutch_input.plus_curation_save_in_flight():
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message="删除失败：Plus 整理旁路正在写入，请等待写入完成后重试。",
                )
            elif dataset_recorder is None:
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message=(
                        "删除失败：当前运行未启用正式 dataset collection；"
                        "debug episode 不支持此删除入口。"
                    ),
                )
            elif (
                recording_active()
                or context.episode_start_countdown_deadline() is not None
                or dataset_recorder.finalizing
            ):
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message="删除失败：数采、开始倒计时或后台保存仍在进行，请等待空闲后重试。",
                )
            elif not clutch_input.hold_active():
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message="真机删除需要先进入停臂保持 / Hold，避免数据压紧阻塞活动中的控制循环。",
                )
            else:
                clutch_input.set_episode_finalize_state(
                    status="finalizing",
                    message=f"正在删除 episode {episode_delete_index} 并连续重编号，请勿开始数采。",
                )
                try:
                    deletion = dataset_recorder.delete_episode(episode_delete_index)
                except (ValueError, IndexError, OSError, RuntimeError) as exc:
                    message = f"删除 episode {episode_delete_index} 失败: {type(exc).__name__}: {exc}"
                    _log("Dataset", message, now=loop_start)
                    clutch_input.set_episode_finalize_state(status="error", message=message)
                else:
                    message = (
                        f"已删除 episode {episode_delete_index}；剩余 {deletion.episode_count} 个，"
                        f"下次从 {deletion.next_episode_index} 开始。备份: {deletion.backup_root}"
                    )
                    _log("Dataset", message, now=loop_start)
                    clutch_input.clear_plus_curation_pending(
                        message="episode 删除/重编号后，待整理 Plus 绑定已清除；请重新复核保存后的 episode。"
                    )
                    clutch_input.set_episode_finalize_state(
                        status="deleted",
                        message=message,
                        episode_index=(
                            None if deletion.episode_count <= 0 else deletion.episode_count - 1
                        ),
                    )
        if clutch_input.consume_episode_start_request():
            if not recording_active():
                schedule_recording_countdown(loop_start, start_reason="manual_key_start")
        mark_loop_phase("loop.keys")

        if (
            clutch_input.hold_active()
            and components.session.state_machine.state == TeleopRunState.DEBUG_ONLY
        ):
            _advance_startup_gui_qpos_maintenance(
                args=args,
                components=components,
                clutch_input=clutch_input,
                now=loop_start,
            )
            if control_ui is not None:
                control_ui.publish(
                    stage="Hold / 停臂保持",
                    active_controls=clutch_input._hold_controls(),
                    control_hint="机械臂已保持停止；可手动调整机械臂，之后重新采集 runtime Home 再遥操。",
                    capture_stage="runtime_home",
                    capture_phase="ready",
                    teleop_running=False,
                    capturing=False,
                    capture_waiting_for_samples=False,
                    debug_ik_timing=clutch_input.ik_debug_timing_enabled(),
                    debug_ik_logging=clutch_input._ik_debug_logging_active,
                    ik_debug_log_path=clutch_input._ik_debug_log_path,
                    gripper_control_enabled=False,
                    **_movement_multiple_ui_state(args),
                    **_fix_arm_ui_state(components),
                    **_init_arm_ui_state(components),
                )
            if loop_phase_timer is not None:
                mark_loop_phase("loop.hold_idle")
            clutch_input.publish_camera_status(
                camera_sources=camera_sources,
                preview_camera_sources=preview_camera_sources,
                camera_previewer=camera_previewer,
                now=loop_start,
            )
            time.sleep(max(0.0, min(0.02, (1.0 / outer_control_hz) - (time.time() - loop_start))))
            continue

        _sync_dual_arm_retargeter_for_real_control(
            components,
            reset_tracking=False,
        )
        mark_loop_phase("loop.sync_preview_before")
        _sync_real_execution_step_limits(components, now=loop_start)
        mark_loop_phase("loop.sync_step_limits")

        _guard_impedance_force_control(
            args,
            components,
            context="main_loop",
            now=loop_start,
        )
        mark_loop_phase("loop.impedance_guard")

        phase_recorder = loop_phase_timer.record if loop_phase_timer is not None else None
        step_result = components.session.run_step(
            now=loop_start,
            phase_recorder=phase_recorder,
        )
        mark_loop_phase("loop.session_run_step")
        init_controller = getattr(components, "init_arm_controller", None)
        qpos_terminal = (
            init_controller.consume_qpos_terminal_event()
            if init_controller is not None
            and callable(getattr(init_controller, "consume_qpos_terminal_event", None))
            else None
        )
        if qpos_terminal is not None:
            qpos_status = str(qpos_terminal.get("status", "failed") or "failed")
            qpos_message = str(qpos_terminal.get("message", "Init qpos finished") or "")
            _log("InitArmQpos", qpos_message, now=loop_start)
            if control_ui is not None:
                control_ui.append_message("InitArmQpos", qpos_message)
                control_ui.publish(
                    control_hint=qpos_message,
                    **_init_arm_ui_state(components),
                )
            if qpos_status == "completed":
                target_side = str(qpos_terminal.get("side", "") or "")
                support_side = "right" if target_side == "left" else "left"
                _release_messages, release_errors = _release_gui_qpos_init_transit(
                    components,
                    operation_id=str(
                        getattr(init_controller, "qpos_operation_id", "") or ""
                    ),
                    now=loop_start,
                    sides=(support_side,),
                )
                if release_errors:
                    clutch_input.request_hold(
                        reason="init_qpos_support_release_failed"
                    )
                    continue
                clutch_input.handle_init_arm_qpos_terminal(
                    qpos_terminal,
                    now=loop_start,
                )
                if clutch_input.hold_active():
                    # A completed dual-arm sequence requests regular Hold.
                    # Do not let its terminal control frame reach collection;
                    # the next loop iteration consumes the Hold transition.
                    continue
            else:
                clutch_input.handle_init_arm_qpos_terminal(
                    qpos_terminal,
                    now=loop_start,
                )
                clutch_input.request_hold(reason=f"init_qpos_{qpos_status}")
                continue
        raw_step_channels = (
            step_result.raw_channels
            if step_result.raw_channels
            else step_result.channels
        )
        clutch_input.publish_raw_tracker_pose_state(
            channels=raw_step_channels,
            now=loop_start,
        )
        post_home_fixed_side = (
            clutch_input.consume_two_pedal_post_home_fixed_side()
        )
        if post_home_fixed_side is not None:
            fix_controller = getattr(components, "fix_arm_controller", None)
            transition = (
                fix_controller.select_side(post_home_fixed_side)
                if fix_controller is not None
                else None
            )
            if transition is not None and transition.ok:
                transition = fix_controller.freeze_selected(now=loop_start)
            if transition is None or not transition.ok:
                message = (
                    "双踏板切换失败：快速 Home 完成后无法建立目标 Fix 快照；"
                    + (
                        str(getattr(transition, "message", "fix controller unavailable"))
                        if transition is not None
                        else "fix controller unavailable"
                    )
                )
                clutch_input.request_hold(reason="two_pedal_post_home_fix_failed")
                clutch_input.finish_two_pedal_fast_home(
                    success=False,
                    now=loop_start,
                    message=message,
                )
                continue
            clutch_input._clear_two_pedal_alert()
        if (
            step_result.home_recalibration_required
            and not clutch_input.hold_active()
        ):
            tracker_hold_payload = {
                "step_index": int(step_index),
                "teleop_state": str(step_result.state.value),
                "motion_hold_mode": str(step_result.motion_hold_mode),
                "motion_hold_reason": str(step_result.motion_hold_reason),
                "tracker_loss_duration_sec": float(
                    step_result.tracker_loss_duration_sec
                ),
                "tracker_loss_channels": list(step_result.tracker_loss_channels),
                "home_recalibration_required": True,
                "faults": _faults_log_payload(step_result),
                "tracker_channels": {
                    name: payload
                    for name, payload in _channels_log_payload(
                        step_result,
                        now=float(loop_start),
                    ).items()
                    if name in {"left_wrist_pose", "right_wrist_pose"}
                },
            }
            if action_debug_log_recorder.active:
                action_debug_log_recorder.write_event(
                    "tracker_manual_hold",
                    tracker_hold_payload,
                    now=loop_start,
                )
            if recording_active():
                pause_recording(
                    loop_start,
                    reason="tracker_loss_manual_hold",
                )
            _log(
                "TrackerHold",
                "tracker target loss exceeded manual-hold threshold; "
                "waiting for explicit runtime home recalibration; "
                f"channels={list(step_result.tracker_loss_channels)} "
                f"duration={step_result.tracker_loss_duration_sec:.3f}s "
                f"reason={step_result.motion_hold_reason}",
                now=loop_start,
            )
            clutch_input.request_hold(reason="tracker_loss_timeout")
            continue
        gripper_command = None
        gripper_command_status: dict[str, dict[str, object]] = {}
        if step_result.motion_hold_mode in {
            TRACKER_HOLD_GRACE,
            TRACKER_HOLD_AUTOMATIC,
        }:
            for side in components.active_sides:
                gripper_command_status[side] = {
                    "status": "tracker_motion_hold",
                    "sent": False,
                    "reason": step_result.motion_hold_reason,
                    "timestamp": float(loop_start),
                }
        elif clutch_input.gripper_control_enabled() and gripper_warmup_probe_running():
            for side in components.active_sides:
                gripper_command_status[side] = {
                    "status": "warmup_probe",
                    "sent": False,
                    "reason": "gripper_warmup_probe",
                    "timestamp": float(loop_start),
                }
        elif clutch_input.gripper_control_enabled():
            gripper_command = _dispatch_q6_gripper_control(
                components,
                step_result.observation,
                now=loop_start,
                status_out=gripper_command_status,
            )
        else:
            for side in components.active_sides:
                gripper_command_status[side] = {
                    "status": "control_disabled",
                    "sent": False,
                    "timestamp": float(loop_start),
                }
        clutch_input.publish_gripper_status(
            channels=step_result.channels,
            command_status=gripper_command_status,
            now=loop_start,
        )
        fix_controller = getattr(components, "fix_arm_controller", None)
        if (
            fix_controller is not None
            and clutch_input._two_pedal_mode_active()
            and not bool(getattr(fix_controller, "recapturing", False))
            and components.session.calibration is not None
            and bool(getattr(fix_controller, "fixed_sides", ()) or ())
        ):
            # Keep fresh candidates for both trackers during single-arm
            # control.  A newly fixed side can then be released again
            # immediately without starting its cache from an empty dwell
            # window.  Applying Home remains restricted to the side that
            # the atomic transition is actually releasing.
            for cache_side in tuple(
                getattr(fix_controller, "active_sides", ()) or ()
            ):
                tracker_channel = raw_step_channels.get(
                    f"{cache_side}_wrist_pose"
                )
                source_timestamp = float(
                    getattr(tracker_channel, "timestamp", float("nan"))
                )
                if (
                    tracker_channel is None
                    or not bool(getattr(tracker_channel, "valid", False))
                    or not np.isfinite(source_timestamp)
                    or not -0.05 <= loop_start - source_timestamp <= 0.5
                ):
                    if tracker_channel is None:
                        unavailable_reason = "tracker_channel_missing"
                    elif not bool(getattr(tracker_channel, "valid", False)):
                        unavailable_reason = "tracker_channel_invalid"
                    elif not np.isfinite(source_timestamp):
                        unavailable_reason = "tracker_timestamp_invalid"
                    else:
                        unavailable_reason = (
                            "tracker_sample_age_out_of_range:"
                            f"{loop_start - source_timestamp:.3f}s"
                        )
                    fix_controller.note_two_pedal_home_unavailable(
                        cache_side,
                        reason=unavailable_reason,
                        observed_at=loop_start,
                    )
                    continue
                try:
                    cached_pose = _channel_pose_in_desk(
                        tracker_channel,
                        components.session.calibration,
                    )
                    fix_controller.observe_two_pedal_home_cache(
                        cache_side,
                        cached_pose,
                        source_timestamp=source_timestamp,
                    )
                except Exception as exc:  # noqa: BLE001
                    fix_controller.note_two_pedal_home_unavailable(
                        cache_side,
                        reason=f"tracker_pose_rejected:{type(exc).__name__}",
                        observed_at=loop_start,
                    )
                    _log(
                        "TwoPedal",
                        "passive Home cache sample rejected: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
        fix_arm_progress = advance_fix_arm_home_capture(
            controller=fix_controller,
            channels=raw_step_channels,
            calibration=components.session.calibration,
            now=loop_start,
            pose_from_channel=_channel_pose_in_desk,
            mean_pose=_mean_pose,
            resolve_robot_home=lambda side: _resolve_robot_home_for_side(components, side=side),
            side_retargeter=lambda side: _side_retargeter(components, side),
            tcp_mode=args.tcp_mode,
            log=lambda tag, message: _log(tag, message, now=loop_start),
        )
        if fix_arm_progress.active:
            if fix_arm_progress.completed and fix_arm_progress.side:
                try:
                    components.session.safety_monitor.request_pose_baseline_reset_many(
                        (f"{fix_arm_progress.side}_wrist_pose",)
                    )
                except Exception as exc:  # noqa: BLE001 - keep the loop live.
                    _log(
                        "FixArm",
                        "Home was applied but safety baseline reset failed; requesting Hold: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
                    clutch_input.request_hold(
                        reason="fix_arm_home_baseline_reset_failed"
                    )
            if fix_arm_progress.failed and bool(
                getattr(fix_arm_progress, "requires_hold", False)
            ):
                clutch_input.request_hold(
                    reason="fix_arm_home_recapture_apply_failed"
                )
            clutch_input.publish_fix_arm_home_capture_progress(fix_arm_progress)
        init_arm_progress = advance_init_arm_home_capture(
            controller=getattr(components, "init_arm_controller", None),
            channels=raw_step_channels,
            calibration=components.session.calibration,
            now=loop_start,
            pose_from_channel=_channel_pose_in_desk,
            mean_pose=_mean_pose,
            resolve_robot_home=lambda side: _resolve_robot_home_for_side(components, side=side),
            side_retargeter=lambda side: _side_retargeter(components, side),
            tcp_mode=args.tcp_mode,
            log=lambda tag, message: _log(tag, message, now=loop_start),
        )
        if init_arm_progress.active:
            if init_arm_progress.completed and init_arm_progress.side:
                reset_sides = (
                    init_arm_progress.sides
                    if init_arm_progress.sides
                    else (init_arm_progress.side,)
                )
                try:
                    components.session.safety_monitor.request_pose_baseline_reset_many(
                        tuple(f"{side}_wrist_pose" for side in reset_sides)
                    )
                except Exception as exc:  # noqa: BLE001 - keep the loop live.
                    _log(
                        "InitArm",
                        "Home was applied but safety baseline reset failed; requesting Hold: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
                    clutch_input.request_hold(
                        reason="init_arm_home_baseline_reset_failed"
                    )
                reset_smoothing = getattr(
                    getattr(components, "dual_arm_retargeter", None),
                    "reset_target_pose_smoothing",
                    None,
                )
                if callable(reset_smoothing):
                    try:
                        reset_smoothing()
                    except Exception as exc:
                        _log(
                            "InitArm",
                            "WARNING: retargeter target smoothing reset failed after home recapture: "
                            f"{type(exc).__name__}: {exc}",
                            now=loop_start,
                        )
                if init_arm_progress.qpos_operation_id:
                    release_messages, release_errors = _release_gui_qpos_init_transit(
                        components,
                        operation_id=init_arm_progress.qpos_operation_id,
                        now=loop_start,
                        sides=(init_arm_progress.side,),
                    )
                    if control_ui is not None and release_messages:
                        control_ui.append_message(
                            "InitArmQpos",
                            "；".join(release_messages) + "；已恢复普通遥操。",
                        )
                    if release_errors:
                        message = (
                            "Init qpos 已完成 Home 重采，但柔顺子模式释放失败；"
                            "为避免控制权冲突已请求 Hold：" + "；".join(release_errors)
                        )
                        if control_ui is not None:
                            control_ui.append_message("InitArmQpos", message)
                        clutch_input.request_hold(
                            reason="init_qpos_transit_release_failed"
                        )
            if init_arm_progress.failed and bool(
                getattr(init_arm_progress, "requires_hold", False)
            ):
                clutch_input.request_hold(
                    reason="init_arm_home_recapture_apply_failed"
                )
            clutch_input.publish_init_arm_home_capture_progress(
                init_arm_progress,
                now=loop_start,
            )
        sync_fix_hold(loop_start)
        clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=loop_start,
        )
        advance_recording_countdown(loop_start, step_result=step_result)
        dataset_recorder = context.dataset_recorder()
        episode_last_state = context.episode_last_state()
        mark_loop_phase("loop.gripper_control")
        active_command = step_result.command
        manual_recovery_command_dispatched = False
        hand_command = _extract_hand_command_from_command(step_result.command)
        if step_result.state == TeleopRunState.RECOVERING:
            hand_command = _build_runtime_hand_command_for_recovery(
                components,
                step_result.observation,
            )
            if step_result.observation is None:
                active_command = None
            elif recovery_plan is None:
                recovery_plan, active_command, recovery_done = _start_recovery_plan(
                    components,
                    step_result.observation,
                    now=loop_start,
                    arm_mode="real",
                    control_mode=args.control_mode,
                    ramp_sec=args.recovery_ramp_sec,
                )
                if not recording_active():
                    if recovery_plan is None:
                        _log(
                            "Recovery",
                            f"direct mode control_mode={args.control_mode}",
                            now=loop_start,
                        )
                    else:
                        _log(
                            "Recovery",
                            f"start mode={recovery_plan.mode} duration={recovery_plan.duration_sec:.2f}s",
                            now=loop_start,
                        )
                active_command = _merge_runtime_hand_command(active_command, hand_command)
                active_command = _filter_manual_real_command(
                    components,
                    active_command,
                    step_result.observation,
                    loop_start,
                )
                if active_command is not None:
                    command_guard_fault = components.session.evaluate_command_guard(
                        active_command,
                        step_result.observation,
                        loop_start,
                    )
                    if command_guard_fault is not None:
                        components.session.reject_command_for_guard_fault(command_guard_fault)
                        _log(
                            "Safety",
                            "recovery_start command rejected by pre-dispatch guard: "
                            f"{command_guard_fault.code}",
                            now=loop_start,
                        )
                        active_command = None
                    elif _guard_impedance_force_control(
                        args,
                        components,
                        context="recovery_start",
                        now=loop_start,
                    ):
                        assert components.session.rig is not None
                        components.session.dispatch_command(active_command)
                        manual_recovery_command_dispatched = True
                        if components.temporary_safety_limiter is not None:
                            components.temporary_safety_limiter.sync_from_command(active_command)
                    else:
                        active_command = None
                if recovery_done:
                    components.session.finish_recovery("recovery completed")
                    recovery_plan = None
            else:
                active_command, recovery_done = _advance_recovery_plan(
                    components,
                    step_result.observation,
                    recovery_plan,
                    now=loop_start,
                )
                active_command = _merge_runtime_hand_command(active_command, hand_command)
                active_command = _filter_manual_real_command(
                    components,
                    active_command,
                    step_result.observation,
                    loop_start,
                )
                if active_command is not None:
                    command_guard_fault = components.session.evaluate_command_guard(
                        active_command,
                        step_result.observation,
                        loop_start,
                    )
                    if command_guard_fault is not None:
                        components.session.reject_command_for_guard_fault(command_guard_fault)
                        _log(
                            "Safety",
                            "recovery_advance command rejected by pre-dispatch guard: "
                            f"{command_guard_fault.code}",
                            now=loop_start,
                        )
                        active_command = None
                    elif _guard_impedance_force_control(
                        args,
                        components,
                        context="recovery_advance",
                        now=loop_start,
                    ):
                        assert components.session.rig is not None
                        components.session.dispatch_command(active_command)
                        manual_recovery_command_dispatched = True
                        if components.temporary_safety_limiter is not None:
                            components.temporary_safety_limiter.sync_from_command(active_command)
                    else:
                        active_command = None
                if recovery_done:
                    components.session.finish_recovery("recovery completed")
                    if not recording_active():
                        _log("Recovery", "completed", now=loop_start)
                    recovery_plan = None
        else:
            recovery_plan = None
        mark_loop_phase("loop.recovery")
        if gripper_command is not None:
            hand_command = (
                gripper_command
                if hand_command is None
                else merge_teleop_commands((hand_command, gripper_command))
            )

        # Promote measured Init arrival to the semantic init-fixed state
        # before collection admission.  This prevents one dual-fixed frame
        # from leaking into the episode before Fix-Hold pauses the writer.
        # These are the same reads used by the later GUI diagnostics; do
        # not add a second synchronous state-backend read to the loop.
        debug_derived_pose_requested = bool(
            ik_debug_log_recorder.active
            or action_debug_log_recorder.active
            or debug_timing_log_recorder.active
        )
        left_hw_state = _runtime_arm_backend_state(
            components.left_state_backend,
            include_derived_pose=debug_derived_pose_requested,
        )
        right_hw_state = _runtime_arm_backend_state(
            components.right_state_backend,
            include_derived_pose=debug_derived_pose_requested,
        )
        if _update_init_arm_arrival_from_hardware(
            components,
            hardware_state_by_side={
                "left": left_hw_state,
                "right": right_hw_state,
            },
            now=loop_start,
        ):
            clutch_input.publish_init_arm_arrival_state()
        sync_fix_hold(loop_start)

        collection_step_result = _collection_step_result(
            step_result,
            active_command=active_command,
            gripper_command=gripper_command,
        )
        if (
            bool(step_result.command_dispatched or manual_recovery_command_dispatched)
            and collection_step_result.command is not None
            and any(
                name in collection_step_result.command.commands
                for name in ("left_arm", "right_arm")
            )
        ):
            last_dispatched_main_command = _copy_teleop_command(
                collection_step_result.command
            )
        if components.position_reference_publisher is not None:
            components.position_reference_publisher.publish_step(collection_step_result)
        mark_loop_phase("loop.publish_position_reference")

        _apply_active_arm_command_to_preview_real(components, active_command)
        mark_loop_phase("loop.preview_apply")
        _sync_dual_arm_retargeter_for_real_control(
            components,
            reset_tracking=False,
        )
        mark_loop_phase("loop.sync_preview_after")

        if args.print_tactile_sum_only:
            tactile_snapshot = (
                _hand_tactile_sum(components.left_hand_backend),
                _hand_tactile_sum(components.right_hand_backend),
            )
            if tactile_sum_gate.should_emit(
                "tactile_sum_only_status",
                tactile_snapshot,
                now=loop_start,
                heartbeat_sec=0.1,
            ):
                _emit_tactile_sum_only_status(
                    components,
                    now=loop_start,
                    stream=console_stream,
                )
        mark_loop_phase("loop.tactile_sum")

        if tactile_visualizer is not None and tactile_preview_rate.should_update(loop_start):
            try:
                if isinstance(
                    tactile_visualizer,
                    (DMTacDistributedForceVisualizer, AsyncDMTacDistributedForceVisualizer),
                ):
                    tactile_force, tactile_health = dmtac_visualization_snapshot_from_backends(
                        components.active_sides,
                        lambda side: _gripper_backend_for_side(components, side),
                        now=loop_start,
                    )
                    tactile_keys = tactile_visualizer.update(
                        tactile_force,
                        health_by_side=tactile_health,
                    )
                else:
                    tactile_visualizer.update(
                        tactile_by_side_from_backends(
                            components.active_sides,
                            lambda side: _hand_backend_for_side(components, side),
                        )
                    )
                    tactile_keys = []
                if any(key in {"q", "\x1b"} for key in tactile_keys):
                    close_tactile_preview("触觉窗口已关闭")
            except Exception as exc:  # noqa: BLE001
                close_tactile_preview(f"触觉窗口刷新失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("TactileVis", f"触觉窗口刷新失败：{exc}")
        mark_loop_phase("loop.tactile_visualizer")

        if camera_preview_active and camera_previewer is not None and camera_preview_rate.should_update(loop_start):
            try:
                camera_keys = camera_previewer.update(preview_camera_sources)
                if any(key in {"q", "\x1b"} for key in camera_keys):
                    close_camera_preview("图像窗口已关闭")
            except Exception as exc:  # noqa: BLE001
                close_camera_preview(f"图像窗口刷新失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("Camera", f"图像窗口刷新失败：{exc}")
        mark_loop_phase("loop.camera_previewer")
        camera_previewer, camera_preview_active, tactile_visualizer = (
            context.visualization_state()
        )

        left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
        right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
        mark_loop_phase("loop.preview_state")
        if components.renderer is not None:
            components.renderer.set_debug_frames(
                _build_renderer_debug_frames(
                    components,
                    channels=step_result.channels,
                )
            )
            components.renderer.set_debug_boxes(_build_renderer_debug_boxes(components))
            components.renderer.render_data(
                left_qpos_arm=left_qpos,
                left_qpos_hand=_real_hand_render_qpos(components.left_hand_backend, "left"),
                right_qpos_arm=right_qpos,
                right_qpos_hand=_real_hand_render_qpos(components.right_hand_backend, "right"),
                capture_image=False,
            )
        mark_loop_phase("loop.render")
        if bool(args.debug_orientation_console) and not recording_active():
            _emit_runtime_orientation_logs(
                components,
                channels=step_result.channels,
                now=loop_start,
                gate=orientation_gate,
            )
        mark_loop_phase("loop.orientation_logs")

        fault_codes = tuple(_runtime_fault_label(fault) for fault in step_result.faults)
        channel_valid = tuple(
            sorted((name, bool(channel.valid)) for name, channel in step_result.channels.items())
        )
        hand_input_status = _hand_channel_status(components, step_result.channels)
        gripper_status = _gripper_runtime_status_signature(
            components,
            channels=step_result.channels,
            command_status=gripper_command_status,
            now=loop_start,
        )
        gripper_health_status = _gripper_runtime_health_signature(gripper_status)
        clutch_channels = components.keyboard_source.get_latest_channels()
        left_clutch_channel = clutch_channels.get("left_clutch_button")
        right_clutch_channel = clutch_channels.get("right_clutch_button")
        left_clutch = bool(left_clutch_channel.value) if left_clutch_channel is not None else False
        right_clutch = bool(right_clutch_channel.value) if right_clutch_channel is not None else False
        position_reference_enabled = components.position_reference_publisher is not None
        sample_due = sample_rate_gate.is_due(time.monotonic())
        auto_stop_reason = recording_auto_stop_reason(step_result.state)
        if auto_stop_reason is not None:
            cancel_recording_countdown(loop_start, reason=auto_stop_reason)
            if recording_active():
                stop_recording(
                    loop_start,
                    end_reason=auto_stop_reason,
                    summary={
                        "last_step_index": step_index,
                        "last_state": step_result.state.value,
                    },
                )
        if recording_active() and not recording_paused() and sample_due:
            modalities_ready, modality_message = recording_warmup_status(
                loop_start,
                step_result=step_result,
            )
            if not modalities_ready:
                pause_recording(
                    loop_start,
                    reason=f"modality_health:{modality_message}",
                )
        if recording_active() and recording_paused():
            resume_recording_if_ready(loop_start, step_result=step_result)
        if recording_active():
            episode_last_state = context.episode_last_state()
            if episode_last_state != step_result.state.value:
                if episode_recorder.is_recording:
                    episode_recorder.record_event(
                        "state_change",
                        payload={
                            "old_state": episode_last_state,
                            "new_state": step_result.state.value,
                        },
                        timestamp=loop_start,
                    )
                episode_last_state = step_result.state.value
                context.set_episode_last_state(episode_last_state)
            if (
                dataset_recorder is not None
                and dataset_recorder.is_recording
                and dataset_recorder.can_record_samples
                and sample_due
            ):
                collection_metadata_extra = {
                    "collection_continuity_segment_index": int(
                        dataset_recorder.continuity_segment_index
                    )
                }
                if loop_phase_timer is not None:
                    collection_metadata_extra.update({
                        "loop_phase_timings_ms": [
                            {"name": name, "elapsed_ms": elapsed * 1000.0}
                            for name, elapsed in loop_phase_timer.phases
                        ],
                        "loop_elapsed_before_record_s": loop_phase_timer.elapsed_sec(),
                    })
                if position_reference_enabled:
                    dataset_step_result = replace(
                        collection_step_result,
                        # ``collection_step_result.timestamp`` is T0: the
                        # action/control information cut.  Sample assembly
                        # may execute asynchronously after this point, so
                        # do not move T0 to the later record call.
                        timestamp=float(collection_step_result.timestamp),
                        action_record=None,
                    )
                elif last_dispatched_main_command is not None:
                    dataset_step_result = _collection_sample_step_result(
                        collection_step_result,
                        dispatched_command=last_dispatched_main_command,
                    )
                    collection_metadata_extra = {
                        **dict(collection_metadata_extra or {}),
                        "collection_action_source": "latest_dispatched_main_command",
                        "collection_action_timestamp": float(
                            last_dispatched_main_command.timestamp
                        ),
                        "collection_action_age_sec": max(
                            0.0,
                            float(collection_step_result.timestamp)
                            - float(last_dispatched_main_command.timestamp),
                        ),
                    }
                else:
                    dataset_step_result = None
                if dataset_step_result is not None:
                    sample_accepted = dataset_recorder.record_step_async(
                        dataset_step_result,
                        include_debug_metadata=bool(ik_debug_log_recorder.active),
                        include_action_debug_metadata=bool(getattr(args, "debug_action", False)),
                        metadata_extra=collection_metadata_extra,
                    )
                    overload_reason = dataset_recorder.collection_overload_reason
                    if not sample_accepted and overload_reason:
                        pause_recording(
                            loop_start,
                            reason=f"collection_overload:{overload_reason}",
                        )
                    modality_reason = dataset_recorder.collection_modality_reason
                    if not sample_accepted and modality_reason:
                        pause_recording(
                            loop_start,
                            reason=f"modality_health:{modality_reason}",
                        )
            if dataset_recorder is not None and not recording_paused():
                overload_reason = dataset_recorder.poll_collection_overload()
                if overload_reason:
                    pause_recording(
                        loop_start,
                        reason=f"collection_overload:{overload_reason}",
                    )
                modality_reason = dataset_recorder.collection_modality_reason
                if modality_reason:
                    pause_recording(
                        loop_start,
                        reason=f"modality_health:{modality_reason}",
                    )
            frequency_status = recording_frequency_status()
            publish_collection_frequency_warning(frequency_status, now=loop_start)
            if (
                episode_recorder.is_recording
                and not recording_paused()
                and sample_due
                and (position_reference_enabled or last_dispatched_main_command is not None)
            ):
                episode_recorder.record_frame(
                    _build_episode_frame(
                        components,
                        step_index=step_index,
                        step_result=step_result,
                        active_command=(
                            active_command
                            if position_reference_enabled
                            else last_dispatched_main_command
                        ),
                        hand_command=hand_command,
                        recovery_plan=recovery_plan,
                        left_clutch=left_clutch,
                        right_clutch=right_clutch,
                    ),
                    timestamp=loop_start,
                )
            paused_now = recording_paused()
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=frequency_status,
                event_status="resume_pending" if paused_now else "recording",
                event_message=(
                    f"数采暂停中：{recording_pause_reason() or 'paused'}；等待遥操恢复稳定后继续写入。"
                    if paused_now
                    else "正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。"
                ),
            )
        mark_loop_phase("loop.episode_record")

        collision_status = None
        collision_fallback = None
        jump_gate_status = None
        branch_candidate = None
        temporary_safety_status = None
        if active_command is not None:
            collision_status = active_command.metadata.get("collision_avoidance_status")
            collision_fallback = active_command.metadata.get(
                "collision_avoidance_fallback_reason"
            )
            jump_gate_status = active_command.metadata.get("ik_jump_gate_status")
            branch_candidate = active_command.metadata.get("ik_branch_candidate")
            temporary_safety_status = active_command.metadata.get("temporary_dual_arm_safety")
        left_command_state = _runtime_arm_backend_state(
            components.left_command_backend,
            include_derived_pose=debug_derived_pose_requested,
        )
        right_command_state = _runtime_arm_backend_state(
            components.right_command_backend,
            include_derived_pose=debug_derived_pose_requested,
        )
        ik_warning_payload = _build_ik_warning_payload(
            active_command=active_command,
            left_command_state=left_command_state,
            right_command_state=right_command_state,
        )
        clutch_input.publish_ik_warning_state(
            ik_warning_payload,
            now=loop_start,
        )
        pose_quality_payload = _build_pose_quality_display_payload(active_command)
        clutch_input.publish_pose_quality_state(
            pose_quality_payload,
            now=loop_start,
        )
        ik_timing_payload = None
        if clutch_input.ik_debug_timing_enabled():
            ik_timing_payload = _build_ik_timing_payload(
                active_command=active_command,
                left_command_state=left_command_state,
                right_command_state=right_command_state,
            )
            clutch_input.publish_ik_timing_state(
                ik_timing_payload,
                now=loop_start,
            )
        active_command_backend = _active_command_backend(components)
        base_control_dt = (
            float(active_command_backend.config.control_dt)
            if active_command_backend is not None
            else 1.0 / outer_control_hz
        )
        target_step_scale = (
            float(components.arm_loop_step_dt_tracker.get("smoothed_loop_dt", base_control_dt))
            / base_control_dt
        )
        if action_debug_log_recorder.should_record_frame(now=loop_start):
            try:
                if ik_timing_payload is None:
                    ik_timing_payload = _build_ik_timing_payload(
                        active_command=active_command,
                        left_command_state=left_command_state,
                        right_command_state=right_command_state,
                    )
                action_debug_recording_status: dict[str, object] = {
                    "active": bool(recording_active()),
                    "path": recording_path(),
                    "frame_count": int(recording_frame_count()),
                }
                if recording_active():
                    action_debug_recording_status["frequency"] = recording_frequency_status()
                action_debug_log_recorder.record_frame(
                    _build_action_debug_metadata(
                        _collection_step_result(
                            step_result,
                            active_command=active_command,
                            gripper_command=gripper_command,
                        ),
                        components=components,
                        record_wall_time=time.time(),
                        step_index=step_index,
                        loop_start=loop_start,
                        left_command_state=left_command_state,
                        right_command_state=right_command_state,
                        left_hw_state=left_hw_state,
                        right_hw_state=right_hw_state,
                        target_step_scale=target_step_scale,
                        left_clutch=left_clutch,
                        right_clutch=right_clutch,
                        ik_timing_payload=ik_timing_payload,
                        loop_phases=(
                            loop_phase_timer.phases
                            if loop_phase_timer is not None
                            else None
                        ),
                        loop_elapsed_sec=(
                            loop_phase_timer.elapsed_sec()
                            if loop_phase_timer is not None
                            else None
                        ),
                        recording_status=action_debug_recording_status,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics must not break teleop.
                action_debug_log_recorder.write_event(
                    "frame_error",
                    {
                        "step_index": int(step_index),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    now=loop_start,
                )
        mark_loop_phase("loop.action_debug_log")
        if clutch_input.consume_pose_snapshot_request():
            if pose_snapshot_recorder is None:
                message = "Pose 抽帧不可用：当前未启用 --fix-arm。"
                clutch_input.set_pose_snapshot_state(status=message, append_message=True)
            elif recording_active() or context.episode_start_countdown_deadline() is not None:
                message = "Pose 抽帧与正式数采互斥；本次请求已忽略。"
                clutch_input.set_pose_snapshot_state(status=message, append_message=True)
            else:
                try:
                    payload = build_fix_arm_pose_snapshot_payload(
                        timestamp=loop_start,
                        monotonic_time=time.monotonic(),
                        step_index=step_index,
                        active_sides=components.active_sides,
                        teleop_state=step_result.state.value,
                        control_hz=float(step_result.control_hz),
                        movement_multiple=_movement_multiple_value(args),
                        tcp_mode=str(args.tcp_mode),
                        calibration=components.session.calibration,
                        raw_channels=raw_step_channels,
                        filtered_channels=step_result.channels,
                        observation=step_result.observation,
                        step_command=step_result.command,
                        active_command=active_command,
                        hand_command=hand_command,
                        gripper_command=gripper_command,
                        command_dispatched=bool(
                            step_result.command_dispatched or active_command is not None
                        ),
                        faults=step_result.faults,
                        fix_arm_state=_fix_arm_ui_state(components),
                        left_retargeter=components.left_retargeter,
                        right_retargeter=components.right_retargeter,
                        left_preview_qpos=left_qpos,
                        right_preview_qpos=right_qpos,
                        left_hardware_state=left_hw_state,
                        right_hardware_state=right_hw_state,
                        left_command_state=left_command_state,
                        right_command_state=right_command_state,
                        gripper_status=gripper_status,
                        pose_quality=pose_quality_payload,
                        ik_timing=ik_timing_payload,
                        extra={
                            "recording_active": False,
                            "episode_countdown_active": False,
                        },
                    )
                    record = pose_snapshot_recorder.record(payload, now=loop_start)
                    message = f"Pose 抽帧已保存 #{record.sequence}: {record.path}"
                    _log("PoseSnapshot", message, now=loop_start)
                    clutch_input.set_pose_snapshot_state(
                        last_path=record.path,
                        status=message,
                        append_message=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    message = f"Pose 抽帧保存失败: {type(exc).__name__}: {exc}"
                    _log("PoseSnapshot", message, now=loop_start)
                    clutch_input.set_pose_snapshot_state(
                        status=message,
                        append_message=True,
                    )
        if ik_debug_log_recorder.active:
            if ik_timing_payload is None:
                ik_timing_payload = _build_ik_timing_payload(
                    active_command=active_command,
                    left_command_state=left_command_state,
                    right_command_state=right_command_state,
                )
            ik_debug_log_recorder.record_frame(
                _build_ik_debug_log_frame(
                    components=components,
                    step_index=step_index,
                    loop_start=loop_start,
                    step_result=step_result,
                    active_command=active_command,
                    left_command_state=left_command_state,
                    right_command_state=right_command_state,
                    left_hw_state=left_hw_state,
                    right_hw_state=right_hw_state,
                    ik_timing_payload=ik_timing_payload,
                    target_step_scale=target_step_scale,
                    left_clutch=left_clutch,
                    right_clutch=right_clutch,
                    full=bool(getattr(args, "ik_debug_log_full", False)),
                    loop_phases=(
                        loop_phase_timer.phases
                        if loop_phase_timer is not None
                        else None
                    ),
                    loop_elapsed_sec=(
                        loop_phase_timer.elapsed_sec()
                        if loop_phase_timer is not None
                        else None
                    ),
                )
            )
        hardware_snapshot = (
            left_hw_state.get("hardware_result_code"),
            right_hw_state.get("hardware_result_code"),
        )
        command_backend_snapshot = (
            _command_backend_health_signature(left_command_state),
            _command_backend_health_signature(right_command_state),
        )
        pose_quality_snapshot = (
            pose_quality_payload.get("status"),
            bool(pose_quality_payload.get("bad", False)),
            tuple(_pose_quality_reason_list(pose_quality_payload.get("reasons"))[:3]),
        )
        fix_arm_snapshot = _fix_arm_debug_state(components)
        init_arm_snapshot = _init_arm_debug_state(components)
        status_snapshot = (
            step_result.state.value,
            fault_codes,
            channel_valid,
            left_clutch,
            right_clutch,
            collision_status,
            collision_fallback,
            jump_gate_status,
            _temporary_safety_health_signature(temporary_safety_status),
            hardware_snapshot,
            command_backend_snapshot,
            pose_quality_snapshot,
            hand_input_status,
            gripper_health_status,
            clutch_input.gripper_control_enabled(),
            tuple(sorted(fix_arm_snapshot.items())),
            tuple(sorted(init_arm_snapshot.items())),
        )
        mark_loop_phase("loop.status_snapshot")
        if status_gate.should_emit(
            "real_teleop_status",
            status_snapshot,
            now=loop_start,
            heartbeat_sec=status_heartbeat_sec,
        ):
            collision_bits = ""
            if collision_status is not None:
                collision_bits = f" collision={collision_status}"
                if collision_fallback:
                    collision_bits += f" fallback={collision_fallback}"
            if jump_gate_status is not None:
                collision_bits += f" ik={jump_gate_status}"
            if branch_candidate is not None:
                collision_bits += f" branch={branch_candidate}"
            if isinstance(temporary_safety_status, Mapping):
                temp_limited = bool(temporary_safety_status.get("limited", False))
                collision_bits += f" temp_safety={'limited' if temp_limited else 'on'}"
            if clutch_input.ik_debug_timing_enabled():
                collision_bits += f" {_ik_timing_status_text(ik_timing_payload)}"
            fault_reason_bits = ""
            fault_reason = getattr(components.session, "last_fault_reason", "")
            if fault_reason:
                fault_reason_bits = f" fault_reason={_short_status_text(fault_reason, max_len=160)}"
            fix_arm_bits = ""
            fix_arm_text = _fix_arm_status_text(fix_arm_snapshot)
            if fix_arm_text:
                fix_arm_bits = f" fix_arm={fix_arm_text.strip()}"
            init_arm_bits = ""
            init_arm_text = _init_arm_status_text(init_arm_snapshot)
            if init_arm_text:
                init_arm_bits = f" init_arm={init_arm_text.strip()}"
            pose_quality_bits = f" {_pose_quality_status_text(pose_quality_payload)}"
            hand_bits = "" if not hand_input_status else f" hand_q6={list(hand_input_status)}"
            gripper_bits = ""
            if gripper_status:
                gripper_bits = (
                    f" gripper_control={'on' if clutch_input.gripper_control_enabled() else 'off'}"
                    f" gripper={list(gripper_status)}"
                )
            _log(
                "RealTeleop",
                f"step={step_index} hz={step_result.control_hz:.1f} "
                f"state={step_result.state.value} "
                f"left_hw_init={left_hw_state.get('hardware_result_code')}:{left_hw_state.get('hardware_result_text')} "
                f"right_hw_init={right_hw_state.get('hardware_result_code')}:{right_hw_state.get('hardware_result_text')} "
                f"left_cmd={_command_backend_status_text(left_command_state)} "
                f"right_cmd={_command_backend_status_text(right_command_state)} "
                f"left_step_scale={float(left_command_state.get('effective_step_scale', 1.0)):.2f} "
                f"right_step_scale={float(right_command_state.get('effective_step_scale', 1.0)):.2f} "
                f"target_step_scale={target_step_scale:.2f} "
                f"left_clutch={left_clutch} "
                f"right_clutch={right_clutch} "
                f"faults={list(fault_codes)}"
                f"{fault_reason_bits}"
                f"{fix_arm_bits}"
                f"{init_arm_bits}"
                f"{collision_bits}"
                f"{pose_quality_bits}"
                f"{hand_bits}"
                f"{gripper_bits}",
                now=loop_start,
            )
        mark_loop_phase("loop.status_emit")
        if loop_phase_timer is not None:
            active_elapsed_sec = loop_phase_timer.elapsed_sec()
            if debug_timing_log_recorder.active:
                timing_log_reason = debug_timing_log_recorder.select_frame_reason(
                    now=loop_start,
                    elapsed_sec=active_elapsed_sec,
                    phases=loop_phase_timer.phases,
                    state=step_result.state,
                    fault_codes=fault_codes,
                    threshold_sec=float(debug_timing_state.threshold_sec),
                )
                if timing_log_reason is not None:
                    if ik_timing_payload is None:
                        ik_timing_payload = _build_ik_timing_payload(
                            active_command=active_command,
                            left_command_state=left_command_state,
                            right_command_state=right_command_state,
                        )
                    debug_recording_status: dict[str, object] = {
                        "active": bool(recording_active()),
                        "path": recording_path(),
                        "frame_count": int(recording_frame_count()),
                    }
                    if recording_active():
                        debug_recording_status["frequency"] = recording_frequency_status()
                    debug_timing_log_recorder.record_frame(
                        _build_debug_timing_log_frame(
                            components=components,
                            step_index=step_index,
                            loop_start=loop_start,
                            elapsed_sec=active_elapsed_sec,
                            phases=loop_phase_timer.phases,
                            reason=timing_log_reason,
                            step_result=step_result,
                            active_command=active_command,
                            left_command_state=left_command_state,
                            right_command_state=right_command_state,
                            left_hw_state=left_hw_state,
                            right_hw_state=right_hw_state,
                            ik_timing_payload=ik_timing_payload,
                            target_step_scale=target_step_scale,
                            left_clutch=left_clutch,
                            right_clutch=right_clutch,
                            camera_sources=camera_sources,
                            preview_camera_sources=preview_camera_sources,
                            camera_previewer=camera_previewer,
                            hand_input_status=hand_input_status,
                            gripper_status=gripper_status,
                            recording_status=debug_recording_status,
                        )
                    )
            _debug_timing_maybe_report(
                debug_timing_state,
                reason="main_loop",
                elapsed_sec=active_elapsed_sec,
                phases=loop_phase_timer.phases,
                step_index=step_index,
                result=step_result,
                active_command=active_command,
                left_hw_state=left_hw_state,
                right_hw_state=right_hw_state,
                left_command_state=left_command_state,
                right_command_state=right_command_state,
                target_step_scale=target_step_scale,
            )
            if position_audit_fanout is not None:
                try:
                    _record_position_loop_timings(
                        position_audit_fanout,
                        timer=loop_phase_timer,
                        step_index=step_index,
                        state=step_result.state,
                        outer_control_hz=outer_control_hz,
                    )
                except Exception as exc:  # noqa: BLE001 - debug must not interrupt control.
                    _log(
                        "ControlDebug",
                        f"WARNING: loop timing audit event failed: {type(exc).__name__}: {exc}",
                        now=loop_start,
                    )

        if dataset_recorder is not None:
            for task in dataset_recorder.start_pending_finalize_tasks():
                _log(
                    "Dataset",
                    "starting deferred episode finalization after the completed "
                    f"control frame: episode {task.episode_index}",
                    now=loop_start,
                )

        step_index += 1
        context.set_step_index(step_index)
        if args.max_steps > 0 and step_index >= args.max_steps:
            break

        remaining = (1.0 / outer_control_hz) - (time.time() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

__all__ = [
    "RealTeleopControlLoopContext",
    "bind_legacy_support",
    "run_real_teleop_control_loop",
]
