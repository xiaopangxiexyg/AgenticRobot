"""GUI control coordinator for the real VIVE-to-RM65 teleoperation runtime.

The legacy debug entrypoint owns hardware/runtime assembly.  It binds the
small, explicit support namespace below after importing this file, so this
module can own the coordinator without importing the legacy entrypoint back.
The binding is intentionally explicit: adding a new runtime dependency to the
GUI coordinator requires reviewing its ownership and side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect


_SUPPORT_NAMES = (
    "CalibrationWebUi",
    "CameraPreviewer",
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "DMTacGripperBackend",
    "EPISODE_GUI_EVENTS",
    "FixArmPoseSnapshotRecorder",
    "FootPadSpecialArmInput",
    "GuiControlEvents",
    "GuiControlState",
    "MOVEMENT_MULTIPLE_DEFAULT",
    "MOVEMENT_MULTIPLE_GUI_EVENTS",
    "MOVEMENT_MULTIPLE_GUI_MIN",
    "MOVEMENT_MULTIPLE_GUI_STEP",
    "Mapping",
    "POSE_SNAPSHOT_GUI_EVENT",
    "Q6HoldRecalibrationGesture",
    "Q6HoldRecalibrationGestureConfig",
    "RealSenseRGBSource",
    "RealTeleopComponents",
    "SPECIAL_ARM_CONTROL_MODE_FOOT_PAD",
    "SPECIAL_ARM_CONTROL_MODE_MANUAL",
    "Sequence",
    "TerminalClutchInput",
    "VIVE_CALIBRATION_GUI_EVENTS",
    "WristDepthFreshnessMonitor",
    "_apply_movement_multiple",
    "_apply_vive_tracker_mapping",
    "_build_camera_status",
    "_build_gripper_status",
    "_build_ik_warning_payload",
    "_build_pose_quality_display_payload",
    "_build_tactile_camera_status",
    "_calibrate_gui_vive_coordinate_profile",
    "_clear_gripper_policy_cache",
    "_collection_task_ui_state",
    "_debug_timing_modules",
    "_default_task_id_from_name",
    "_expand_cli_path",
    "_fix_arm_ui_state",
    "_gripper_backend_for_side",
    "_gripper_gui_available",
    "_gripper_input_source_from_components",
    "_gripper_retargeter_for_side",
    "_gui_collection_task_controls",
    "_gui_current_active_controls",
    "_gui_episode_busy",
    "_gui_episode_controls",
    "_gui_episode_countdown_controls",
    "_gui_finalizing_controls",
    "_gui_fix_arm_controls",
    "_gui_fix_init_allowed_in_controls",
    "_gui_hold_controls",
    "_gui_init_arm_controls",
    "_gui_movement_multiple_controls",
    "_gui_pose_snapshot_controls",
    "_gui_pre_home_controls",
    "_gui_recording_controls",
    "_gui_single_arm_tracker_swap_allowed_in_hold",
    "_gui_teleop_controls",
    "_gui_visualization_controls",
    "_gui_vive_calibration_controls",
    "_hand_backend_for_side",
    "_hand_q6_source_from_components",
    "_has_q6_gripper_control",
    "_init_arm_ui_state",
    "_load_gui_vive_calibration_profile",
    "_log",
    "_movement_multiple_ui_state",
    "_movement_multiple_value",
    "_publish_raw_tracker_pose_state",
    "_refresh_formal_dataset_layout",
    "_resolved_gripper_policies_by_side",
    "_resolved_gripper_policy",
    "_save_gui_vive_calibration_profile",
    "_task_definition_for_key",
    "_toggle_collection_round_trip_phase",
    "_tracker_source_side_for_logical_side",
    "_vive_calibration_profile_ui_state",
    "_vive_pose_source_from_components",
    "_with_expected_wrist_depth_camera_statuses",
    "argparse",
    "gui_collection_tasks",
    "json",
    "np",
    "Path",
    "run_best_effort_cleanup",
    "time",
)


def bind_legacy_support(namespace: Mapping[str, object]) -> None:
    """Bind legacy runtime helpers while preserving the old import contract."""

    missing = [name for name in _SUPPORT_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "real_teleop GUI coordinator support is incomplete: "
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


class _LegacyFunctionProxy:
    """Keep old-entrypoint monkeypatch points live after the class moves."""

    __slots__ = ("_namespace", "_name")

    def __init__(self, namespace: Mapping[str, object], name: str) -> None:
        self._namespace = namespace
        self._name = str(name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        target = self._namespace[self._name]
        if target is self:
            raise RuntimeError(f"recursive GUI support proxy: {self._name}")
        return target(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._namespace[self._name], attribute)


class RealTeleopControlInput:
    def __init__(
        self,
        *,
        terminal: TerminalClutchInput,
        ui: CalibrationWebUi | None,
        args: argparse.Namespace,
        components: RealTeleopComponents,
        pose_snapshot_recorder: FixArmPoseSnapshotRecorder | None = None,
        foot_pad_input: FootPadSpecialArmInput | None = None,
    ) -> None:
        self.terminal = terminal
        self.ui = ui
        self.args = args
        self.components = components
        self.foot_pad_input = foot_pad_input
        self.enabled = bool(terminal.enabled or ui is not None)
        self.should_quit = False
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        self._hold_requested = False
        self._hold_active = False
        self._tracker_swap_locked = False
        self._teleop_started = False
        self._last_raw_pose_publish_at = 0.0
        self._raw_pose_publish_interval_sec = 0.2
        self._last_capture_progress_publish_at = 0.0
        self._capture_progress_publish_interval_sec = 0.1
        self._last_ik_timing_publish_at = 0.0
        self._ik_timing_publish_interval_sec = 0.2
        self._last_ik_warning_publish_at = 0.0
        self._ik_warning_publish_interval_sec = 0.2
        self._last_ik_warning_signature: tuple[object, ...] | None = None
        self._last_pose_quality_publish_at = 0.0
        self._pose_quality_publish_interval_sec = 0.2
        self._ik_debug_log_toggle_requested = False
        self._ik_debug_logging_active = False
        self._ik_debug_log_path = ""
        self._gripper_control_enabled = False
        self._hold_request_reason = "manual"
        self._q6_hold_recalibration_gesture = Q6HoldRecalibrationGesture(
            Q6HoldRecalibrationGestureConfig(
                enabled=bool(getattr(args, "q6_hold_recalibration_gesture", False)),
                hold_sec=float(getattr(args, "q6_gesture_hold_sec", 1.0)),
                close_threshold=float(getattr(args, "q6_gesture_close_threshold", 0.95)),
                open_threshold=float(getattr(args, "q6_gesture_open_threshold", 0.85)),
            )
        )
        self._last_gripper_status_publish_at = 0.0
        self._gripper_status_publish_interval_sec = 0.2
        self._last_camera_status_publish_at = 0.0
        self._camera_status_publish_interval_sec = 0.2
        self._camera_depth_monitor = WristDepthFreshnessMonitor()
        self._episode_start_requested = False
        self._episode_stop_requested = False
        self._episode_last_episode_index: int | None = None
        self._episode_last_frame_count: int | None = None
        self._episode_recording_ui_active = False
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = False
        self._manual_gripper_force_by_side: dict[str, float] = {}
        self._camera_visualization_toggle_requested = False
        self._tactile_visualization_toggle_requested = False
        self._camera_visualization_active = bool(getattr(args, "vis_cam", False))
        self._tactile_visualization_active = bool(getattr(args, "visualize_tactile", False))
        self._pose_snapshot_recorder = pose_snapshot_recorder
        self._pose_snapshot_requested = False
        self._pose_snapshot_last_path = ""
        self._pose_snapshot_status = ""
        self._gui_control_events = GuiControlEvents(
            episode_events=tuple(EPISODE_GUI_EVENTS),
            pose_snapshot_event=POSE_SNAPSHOT_GUI_EVENT,
            vive_calibration_events=tuple(VIVE_CALIBRATION_GUI_EVENTS),
            movement_multiple_events=tuple(MOVEMENT_MULTIPLE_GUI_EVENTS),
        )

    def _gui_control_state(self) -> GuiControlState:
        return GuiControlState(
            teleop_started=bool(getattr(self, "_teleop_started", False)),
            hold_active=bool(getattr(self, "_hold_active", False)),
            episode_recording_available=(
                not bool(getattr(self.args, "no_collection", False))
                or not bool(getattr(self.args, "no_episode_recording", False))
            ),
            episode_recording_active=bool(
                getattr(self, "_episode_recording_ui_active", False)
            ),
            episode_countdown_active=bool(
                getattr(self, "_episode_countdown_ui_active", False)
            ),
            episode_finalizing_active=bool(
                getattr(self, "_episode_finalizing_ui_active", False)
            ),
            camera_visualization_available=bool(
                getattr(self.args, "camera_visualization_available", False)
            ),
            tactile_visualization_available=bool(
                getattr(self.args, "tactile_visualization_available", False)
            ),
            fix_arm_available=getattr(self.components, "fix_arm_controller", None) is not None,
            init_arm_available=getattr(self.components, "init_arm_controller", None) is not None,
            pose_snapshot_available=(
                getattr(self.components, "fix_arm_controller", None) is not None
                and getattr(self, "_pose_snapshot_recorder", None) is not None
            ),
            active_side_count=len(
                tuple(getattr(self.components, "active_sides", ()) or ())
            ),
            special_arm_control_mode=str(
                getattr(self.args, "special_arm_control_mode", SPECIAL_ARM_CONTROL_MODE_MANUAL)
                or SPECIAL_ARM_CONTROL_MODE_MANUAL
            ),
        )

    def _foot_pad_mode_enabled(self) -> bool:
        return str(
            getattr(self.args, "special_arm_control_mode", SPECIAL_ARM_CONTROL_MODE_MANUAL)
            or SPECIAL_ARM_CONTROL_MODE_MANUAL
        ) == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD

    def _manual_special_arm_controls_enabled(self) -> bool:
        return not self._foot_pad_mode_enabled()

    def _gui_events(self) -> GuiControlEvents:
        events = getattr(self, "_gui_control_events", None)
        if isinstance(events, GuiControlEvents):
            return events
        return GuiControlEvents(
            episode_events=tuple(EPISODE_GUI_EVENTS),
            pose_snapshot_event=POSE_SNAPSHOT_GUI_EVENT,
            vive_calibration_events=tuple(VIVE_CALIBRATION_GUI_EVENTS),
            movement_multiple_events=tuple(MOVEMENT_MULTIPLE_GUI_EVENTS),
        )

    def _episode_recording_available(self) -> bool:
        return self._gui_control_state().episode_recording_available

    def _camera_visualization_available(self) -> bool:
        return self._gui_control_state().camera_visualization_available

    def _tactile_visualization_available(self) -> bool:
        return self._gui_control_state().tactile_visualization_available

    def _visualization_controls(self) -> list[str]:
        return _gui_visualization_controls(self._gui_control_state())

    def _vive_calibration_controls(self, *, allow_load: bool) -> list[str]:
        return _gui_vive_calibration_controls(self._gui_events(), allow_load=allow_load)

    def _movement_multiple_controls(self) -> list[str]:
        return _gui_movement_multiple_controls(self._gui_control_state(), self._gui_events())

    def _collection_task_controls(self) -> list[str]:
        return _gui_collection_task_controls(self._gui_control_state())

    def _episode_busy(self) -> bool:
        return _gui_episode_busy(self._gui_control_state())

    def _episode_controls(self) -> list[str]:
        return _gui_episode_controls(self._gui_control_state(), self._gui_events())

    def _fix_init_allowed_in_controls(self) -> bool:
        return _gui_fix_init_allowed_in_controls(self._gui_control_state())

    def _recording_controls(self, *, hold: bool = False) -> list[str]:
        return _gui_recording_controls(
            self._gui_control_state(),
            self._gui_events(),
            hold=hold,
        )

    def _finalizing_controls(self) -> list[str]:
        return _gui_finalizing_controls(self._gui_control_state())

    def _current_active_controls(self) -> list[str]:
        return _gui_current_active_controls(self._gui_control_state(), self._gui_events())

    def _fix_arm_available(self) -> bool:
        return self._gui_control_state().fix_arm_available

    def _fix_arm_controls(self) -> list[str]:
        return _gui_fix_arm_controls(self._gui_control_state())

    def _init_arm_available(self) -> bool:
        return self._gui_control_state().init_arm_available

    def _init_arm_controls(self) -> list[str]:
        return _gui_init_arm_controls(self._gui_control_state())

    def _pose_snapshot_available(self) -> bool:
        return self._gui_control_state().pose_snapshot_available

    def _pose_snapshot_controls(self) -> list[str]:
        return _gui_pose_snapshot_controls(self._gui_control_state(), self._gui_events())

    def _pose_snapshot_ui_state(self) -> dict[str, object]:
        recorder = getattr(self, "_pose_snapshot_recorder", None)
        return {
            "pose_snapshot_available": self._pose_snapshot_available(),
            "pose_snapshot_dir": "" if recorder is None else str(recorder.root_dir),
            "pose_snapshot_count": 0 if recorder is None else int(recorder.count),
            "pose_snapshot_last_path": getattr(self, "_pose_snapshot_last_path", ""),
            "pose_snapshot_status": getattr(self, "_pose_snapshot_status", ""),
        }

    def _teleop_controls(self) -> list[str]:
        return _gui_teleop_controls(self._gui_control_state(), self._gui_events())

    def _single_arm_tracker_swap_allowed_in_hold(self) -> bool:
        return _gui_single_arm_tracker_swap_allowed_in_hold(self._gui_control_state())

    def _pre_home_controls(self) -> list[str]:
        return _gui_pre_home_controls(self._gui_control_state(), self._gui_events())

    def _hold_controls(self) -> list[str]:
        return _gui_hold_controls(self._gui_control_state(), self._gui_events())

    def start(self) -> None:
        self.terminal.start()
        if self.ui is None:
            self.enabled = bool(self.terminal.enabled)
            return
        gripper_status = _build_gripper_status(self.components)
        self.ui.publish(
            stage="RM65 真机遥操准备",
            selected_side="",
            host=str(self.args.host),
            port=int(self.args.port),
            load_profile=str(getattr(self.args, "load_profile", "") or ""),
            active_controls=self._pre_home_controls(),
            control_hint=self._control_hint(),
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            debug_ik_timing=bool(getattr(self.args, "debug_ik_timing", False)),
            debug_ik_logging=False,
            ik_debug_log_path="",
            ik_timing={
                "available": False,
                "message": "等待遥操启动后接收 IK timing 元数据。",
            },
            ik_warning=_build_ik_warning_payload(active_command=None),
            pose_quality=_build_pose_quality_display_payload(None),
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=False,
            gripper_status=gripper_status,
            tactile_camera_status=_build_tactile_camera_status(gripper_status),
            camera_self_check={
                "enabled": False,
                "status": "idle",
                "status_level": "off",
                "status_label": "未开始",
                "message": "",
            },
            episode_recording_available=self._episode_recording_available(),
            episode_recording=False,
            episode_dir="",
            episode_frame_count=None,
            episode_sample_hz=None,
            episode_target_hz=float(getattr(self.args, "collection_warning_hz", 15.0)),
            episode_frequency_warning={"active": False, "message": ""},
            episode_event_status=(
                "idle" if self._episode_recording_available() else "unavailable"
            ),
            episode_event_message=(
                "点击“开始数采”或按 A 开始；再次按 A 停止。"
                if self._episode_recording_available()
                else "当前启动参数禁用了正式数据集和 debug episode 录制。"
            ),
            episode_last_episode_index=None,
            episode_last_frame_count=None,
            **_collection_task_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            visualization_status="",
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
            capture_waiting_for_samples=False,
            special_arm_control_mode=str(
                getattr(self.args, "special_arm_control_mode", SPECIAL_ARM_CONTROL_MODE_MANUAL)
            ),
        )
        self.ui.start()
        ui_url = str(getattr(self.ui, "url", "") or "")
        _log(
            "WebGUI",
            f"ready url={ui_url or '<unknown>'} state_url={(ui_url.rstrip('/') + '/state.json') if ui_url else '<unknown>'} "
            f"debug_url={(ui_url.rstrip('/') + '/debug.json') if ui_url else '<unknown>'} "
            "capture_stage=runtime_home capture_phase=ready",
        )
        self.ui.append_message("WebGUI", f"后端状态已发布；如果页面不刷新，请打开 {ui_url or '当前 Web GUI URL'}。")
        self.publish_raw_tracker_pose_state(force=True)
        self.enabled = True

    def close(self) -> None:
        steps = []
        if self.ui is not None:
            steps.append(("web_ui", self.ui.close))
        steps.append(("terminal", self.terminal.close))
        if self.foot_pad_input is not None:
            steps.append(("foot_pad", self.foot_pad_input.close))
        run_best_effort_cleanup("real_teleop_control_input.close", steps)

    def poll(self) -> None:
        current_time = time.time()
        self.terminal.poll()
        if self.foot_pad_input is not None:
            for event in self.foot_pad_input.consume_events():
                self._handle_foot_pad_special_arm_event(event, now=current_time)
        if self.terminal.should_quit:
            self.should_quit = True
        if self.terminal.consume_calibration_request():
            self._calibration_requested = True
            self._calibration_request_from_hold = bool(self._hold_active)
        if self.ui is not None:
            for event in self.ui.consume_events():
                if event == "q":
                    self.should_quit = True
                    if self._teleop_started:
                        self.ui.publish(
                            stage="退出 / 停臂中",
                            active_controls=[],
                            control_hint="已收到网页退出；后端正在停止机械臂并关闭遥操。",
                            capture_stage="",
                            capture_phase="idle",
                            teleop_running=False,
                            capturing=False,
                            capture_waiting_for_samples=False,
                        )
                        self.ui.append_message("EmergencyStop", "网页退出已触发机械臂停臂请求。")
                elif event == "p":
                    self._calibration_requested = True
                    self._calibration_request_from_hold = bool(self._hold_active)
                elif event in {"c", "fix_arm_toggle"}:
                    if (
                        self._manual_special_arm_controls_enabled()
                        and
                        self._fix_arm_available()
                        and self._teleop_started
                        and not self._hold_active
                    ):
                        if self._handle_fix_arm_c_event(now=current_time):
                            continue
                        message = (
                            "fix-arm C event was received while teleop is running, "
                            "but no fix-arm transition was available."
                        )
                        _log("FixArm", message, now=current_time)
                        self._publish_fix_arm_state(
                            message=message,
                            control_hint=message,
                        )
                    elif self.ui is not None:
                        message = "C 仅用于运行中的 fix-arm；runtime home 请点击“开始遥操”。"
                        self.ui.append_message("Calibration", message)
                        self.ui.publish(control_hint=message)
                elif event == "h":
                    self.request_hold()
                elif event == "w":
                    self._toggle_vive_tracker_swap()
                elif event == "d":
                    self._ik_debug_log_toggle_requested = True
                elif event in {"g", "init_arm_toggle"}:
                    if (
                        self._manual_special_arm_controls_enabled()
                        and self._init_arm_available()
                    ):
                        self._handle_init_arm_g_event(now=current_time)
                elif event == "episode_start" or event.startswith("episode_start:"):
                    if event.startswith("episode_start:"):
                        self._apply_episode_start_payload(event.split(":", 1)[1])
                    self._episode_start_requested = True
                elif event.startswith("task_select:"):
                    self._apply_episode_start_payload(event.split(":", 1)[1])
                elif event.startswith("gripper_force_set:"):
                    self._set_gripper_force_from_gui(event.split(":", 1)[1])
                elif event == "episode_stop":
                    self._episode_stop_requested = True
                elif event == "round_trip_toggle":
                    self._toggle_round_trip_collection_task(source="manual")
                elif event.startswith("fix_arm_select:"):
                    if self._manual_special_arm_controls_enabled():
                        self._set_fix_arm_side_from_gui(event.split(":", 1)[1])
                elif event.startswith("init_arm_select:"):
                    if self._manual_special_arm_controls_enabled():
                        self._set_init_arm_side_from_gui(event.split(":", 1)[1])
                elif event == POSE_SNAPSHOT_GUI_EVENT:
                    self.request_pose_snapshot()
                elif event == "toggle_camera_vis":
                    self._camera_visualization_toggle_requested = True
                elif event == "toggle_tactile_vis":
                    self._tactile_visualization_toggle_requested = True
                elif event == "calibrate_vive_coordinate_profile":
                    self._calibrate_vive_coordinate_profile()
                elif event == "save_vive_calibration_profile":
                    self._save_vive_calibration_profile()
                elif event == "load_vive_calibration_profile":
                    self._load_vive_calibration_profile()
                elif event.startswith("movement_multiple_set:"):
                    self._set_movement_multiple_from_gui(event.split(":", 1)[1])
                elif event in MOVEMENT_MULTIPLE_GUI_EVENTS:
                    self._handle_movement_multiple_event(event)
        self._poll_q6_hold_recalibration_gesture(now=current_time)
        if self.ui is not None:
            self.publish_raw_tracker_pose_state(now=current_time)
            self.publish_gripper_status(now=current_time)

    def _handle_foot_pad_special_arm_event(self, event: str, *, now: float) -> None:
        """Route one physical pedal to the existing GUI special-arm transition."""

        if not self._foot_pad_mode_enabled():
            return
        routes = {
            "fix_arm_left": ("fix", "left"),
            "fix_arm_right": ("fix", "right"),
            "init_arm_left": ("init", "left"),
            "init_arm_right": ("init", "right"),
        }
        route = routes.get(str(event))
        if route is None:
            return
        operation, side = route
        side_label = "左臂" if side == "left" else "右臂"
        if not self._fix_init_allowed_in_controls():
            message = f"foot-pad {event} 已忽略：{side_label}特殊臂操作仅在正常遥操或数采中可用。"
            if operation == "fix":
                self._publish_fix_arm_state(message=message, control_hint=message)
            else:
                self._publish_init_arm_state(message=message, control_hint=message)
            return

        if operation == "fix":
            controller = getattr(self.components, "fix_arm_controller", None)
            if controller is None:
                return
            if controller.fixed and controller.fixed_side != side:
                active_label = "左臂" if controller.fixed_side == "left" else "右臂"
                message = (
                    f"foot-pad {event} 已忽略：当前固定的是{active_label}；"
                    f"请踩对应踏板重采 {active_label} Home。"
                )
                self._publish_fix_arm_state(message=message, control_hint=message)
                return
            if not controller.fixed:
                transition = controller.select_side(side)
                if not transition.ok:
                    message = f"foot-pad 选择{side_label} fix-arm失败: {transition.message}"
                    self._publish_fix_arm_state(message=message, control_hint=message)
                    return
            self._handle_fix_arm_c_event(now=now)
            return

        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None:
            return
        if controller.active and controller.active_side != side:
            active_label = "左臂" if controller.active_side == "left" else "右臂"
            message = (
                f"foot-pad {event} 已忽略：当前归位的是{active_label}；"
                f"请踩对应踏板重采 {active_label} Home。"
            )
            self._publish_init_arm_state(message=message, control_hint=message)
            return
        if not controller.active:
            transition = controller.select_side(side)
            if not transition.ok:
                message = f"foot-pad 选择{side_label} init-arm失败: {transition.message}"
                self._publish_init_arm_state(message=message, control_hint=message)
                return
        self._handle_init_arm_g_event(now=now)

    def _publish_fix_arm_state(self, *, message: str = "", control_hint: str = "") -> None:
        if self.ui is None:
            return
        updates = {
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            "active_controls": (
                self._teleop_controls()
                if self._teleop_started and not self._hold_active
                else self._hold_controls()
                if self._teleop_started
                else self._pre_home_controls()
            ),
        }
        if control_hint:
            updates["control_hint"] = control_hint
        self.ui.publish(**updates)
        if message:
            self.ui.append_message("FixArm", message)

    def _set_fix_arm_side_from_gui(self, side: str) -> None:
        controller = getattr(self.components, "fix_arm_controller", None)
        if controller is None:
            return
        transition = controller.select_side(side)
        side_label = "左臂" if transition.side == "left" else "右臂"
        if transition.ok:
            message = f"固定侧已选择: {side_label}"
        else:
            message = f"固定侧切换失败: {transition.message}"
        self._publish_fix_arm_state(message=message, control_hint=message)

    def _publish_init_arm_state(self, *, message: str = "", control_hint: str = "") -> None:
        if self.ui is None:
            return
        updates = {
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            "active_controls": (
                self._teleop_controls()
                if self._teleop_started and not self._hold_active
                else self._hold_controls()
                if self._teleop_started
                else self._pre_home_controls()
            ),
        }
        if control_hint:
            updates["control_hint"] = control_hint
        self.ui.publish(**updates)
        if message:
            self.ui.append_message("InitArm", message)

    def _set_init_arm_side_from_gui(self, side: str) -> None:
        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None:
            return
        transition = controller.select_side(side)
        side_label = "左臂" if transition.side == "left" else "右臂"
        if transition.ok:
            message = f"归位侧已选择: {side_label}"
        else:
            message = f"归位侧切换失败: {transition.message}"
        self._publish_init_arm_state(message=message, control_hint=message)

    def _reset_runtime_tracking_for_side(self, side: str, *, tag: str, now: float) -> None:
        channel = f"{side}_wrist_pose"
        try:
            self.components.session.safety_monitor.request_pose_baseline_reset_many((channel,))
        except Exception as exc:
            _log(
                tag,
                f"WARNING: safety baseline reset failed for {channel}: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
        reset_smoothing = getattr(
            getattr(self.components, "dual_arm_retargeter", None),
            "reset_target_pose_smoothing",
            None,
        )
        if callable(reset_smoothing):
            try:
                reset_smoothing()
            except Exception as exc:
                _log(
                    tag,
                    "WARNING: retargeter target smoothing reset failed: "
                    f"{type(exc).__name__}: {exc}",
                    now=now,
                )

    def _handle_init_arm_g_event(self, *, now: float) -> bool:
        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None:
            return False
        if not self._teleop_started or self._hold_active:
            return False
        try:
            _log(
                "InitArm",
                "G event received "
                f"selected={getattr(controller, 'selected_side', '') or '-'} "
                f"active={getattr(controller, 'active_side', '') or '-'} "
                f"mode={getattr(controller, 'mode', '') or '-'} "
                f"recapturing={bool(getattr(controller, 'recapturing', False))}",
                now=now,
            )
            if controller.recapturing:
                message = "归位臂正在重采 home，请保持 tracker 直到样本采满。"
                self._publish_init_arm_state(message=message, control_hint=message)
                return True
            if controller.active:
                transition = controller.start_recalibration(
                    now=now,
                    timeout_sec=float(getattr(self.args, "calibration_timeout", 10.0)),
                    min_samples=int(getattr(self.args, "calibration_samples", 20)),
                )
                side_label = "左臂" if transition.side == "left" else "右臂"
                if transition.ok:
                    message = f"{side_label}继续执行 init pose，开始重采该侧 tracker home。"
                else:
                    message = f"归位臂重采失败: {transition.message}"
                self._publish_init_arm_state(message=message, control_hint=message)
                return True

            side = str(getattr(controller, "selected_side", "left") or "left")
            fix_controller = getattr(self.components, "fix_arm_controller", None)
            same_side_fixed = bool(
                fix_controller is not None
                and getattr(fix_controller, "fixed", False)
                and getattr(fix_controller, "fixed_side", "") == side
            )
            if same_side_fixed:
                side_label = "左臂" if side == "left" else "右臂"
                message = f"{side_label}当前已被 fix-arm 固定；请先完成/释放该侧固定后再归位。"
                self._publish_init_arm_state(message=message, control_hint=message)
                return True

            transition = controller.start_init(now=now)
            side_label = "左臂" if transition.side == "left" else "右臂"
            if transition.ok:
                self._reset_runtime_tracking_for_side(transition.side, tag="InitArm", now=now)
                message = f"{side_label}开始执行 init pose；再次按 G 重采该侧 home 并恢复遥操。"
            else:
                message = f"{side_label}归位失败: {transition.message}"
            self._publish_init_arm_state(message=message, control_hint=message)
        except Exception as exc:
            message = (
                "init-arm G event failed before state transition; "
                f"{type(exc).__name__}: {exc}"
            )
            _log("InitArm", message, now=now)
            self._publish_init_arm_state(message=message, control_hint=message)
        return True

    def _handle_fix_arm_c_event(self, *, now: float) -> bool:
        controller = getattr(self.components, "fix_arm_controller", None)
        if controller is None:
            return False
        if not self._teleop_started or self._hold_active:
            return False
        try:
            _log(
                "FixArm",
                "C event received "
                f"selected={getattr(controller, 'selected_side', '') or '-'} "
                f"fixed={getattr(controller, 'fixed_side', '') or '-'} "
                f"active={bool(getattr(controller, 'fixed', False))} "
                f"recapturing={bool(getattr(controller, 'recapturing', False))}",
                now=now,
            )
            if controller.recapturing:
                message = "固定臂正在重采 home，请保持姿态直到样本采满。"
                self._publish_fix_arm_state(message=message, control_hint=message)
                return True
            if controller.fixed:
                transition = controller.start_recalibration(
                    now=now,
                    timeout_sec=float(getattr(self.args, "calibration_timeout", 10.0)),
                    min_samples=int(getattr(self.args, "calibration_samples", 20)),
                )
                side_label = "左臂" if transition.side == "left" else "右臂"
                if transition.ok:
                    message = f"{side_label}保持固定，开始逐帧重采该侧 tracker home。"
                else:
                    message = f"固定臂重采失败: {transition.message}"
                self._publish_fix_arm_state(message=message, control_hint=message)
                return True
            init_controller = getattr(self.components, "init_arm_controller", None)
            init_same_side_active = bool(
                init_controller is not None
                and getattr(init_controller, "active", False)
                and getattr(init_controller, "active_side", "") == getattr(controller, "selected_side", "")
            )
            if init_same_side_active:
                side = str(getattr(controller, "selected_side", "left") or "left")
                side_label = "左臂" if side == "left" else "右臂"
                message = f"{side_label}当前正在 init-arm 归位；请先按 G 完成该侧 home 重采后再固定。"
                self._publish_fix_arm_state(message=message, control_hint=message)
                return True
            transition = controller.freeze_selected(now=now)
            side_label = "左臂" if transition.side == "left" else "右臂"
            if transition.ok:
                fixed_channel = f"{transition.side}_wrist_pose"
                try:
                    self.components.session.safety_monitor.request_pose_baseline_reset_many(
                        (fixed_channel,)
                    )
                except Exception as exc:
                    _log(
                        "FixArm",
                        f"WARNING: safety baseline reset failed for {fixed_channel}: "
                        f"{type(exc).__name__}: {exc}",
                        now=now,
                    )
                reset_smoothing = getattr(
                    getattr(self.components, "dual_arm_retargeter", None),
                    "reset_target_pose_smoothing",
                    None,
                )
                if callable(reset_smoothing):
                    try:
                        reset_smoothing()
                    except Exception as exc:
                        _log(
                            "FixArm",
                            "WARNING: retargeter target smoothing reset failed: "
                            f"{type(exc).__name__}: {exc}",
                            now=now,
                        )
                _log(
                    "FixArm",
                    f"{side_label} fixed; reset safety baseline for {fixed_channel} "
                    "and retargeter target smoothing.",
                    now=now,
                )
                message = f"{side_label}已固定：固定侧输入快照保持不动，IK/防碰撞继续正常运行；再次按 C 重采该侧 home。"
            else:
                missing = ", ".join(transition.missing_effectors)
                missing_channels = ", ".join(getattr(transition, "missing_channels", ()))
                details = []
                if missing:
                    details.append(f"缺少上一帧命令: {missing}")
                if missing_channels:
                    details.append(f"缺少有效输入快照: {missing_channels}")
                detail = " " + "；".join(details) if details else ""
                message = f"{side_label}固定失败: {transition.message}.{detail}"
            self._publish_fix_arm_state(message=message, control_hint=message)
        except Exception as exc:
            message = (
                "fix-arm C event failed before state transition; "
                f"{type(exc).__name__}: {exc}"
            )
            _log("FixArm", message, now=now)
            self._publish_fix_arm_state(message=message, control_hint=message)
        return True

    def request_pose_snapshot(self) -> None:
        if not self._pose_snapshot_available():
            message = "Pose 抽帧仅在 --fix-arm 遥操模式可用。"
            self.set_pose_snapshot_state(status=message, append_message=True)
            return
        if self._episode_recording_ui_active or self._episode_countdown_ui_active:
            message = "Pose 抽帧与正式数采互斥；请先停止数采或取消倒计时。"
            self.set_pose_snapshot_state(status=message, append_message=True)
            return
        if not self._teleop_started or self._hold_active:
            message = "Pose 抽帧只能在遥操运行中执行。"
            self.set_pose_snapshot_state(status=message, append_message=True)
            return
        self._pose_snapshot_requested = True
        self.set_pose_snapshot_state(status="已请求抽帧；等待下一帧遥操状态。")

    def consume_pose_snapshot_request(self) -> bool:
        requested = bool(self._pose_snapshot_requested)
        self._pose_snapshot_requested = False
        return requested

    def set_pose_snapshot_state(
        self,
        *,
        last_path: str | Path | None = None,
        status: str | None = None,
        append_message: bool = False,
    ) -> None:
        if last_path is not None:
            self._pose_snapshot_last_path = str(last_path)
        if status is not None:
            self._pose_snapshot_status = str(status)
        if self.ui is not None:
            self.ui.publish(**self._pose_snapshot_ui_state())
            if append_message and status:
                self.ui.append_message("PoseSnapshot", str(status))

    def consume_calibration_request(self) -> bool:
        requested = bool(self._calibration_requested)
        request_from_hold = bool(self._calibration_request_from_hold or self._hold_active)
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        if requested:
            if self._teleop_started and not self._hold_active:
                if self.ui is not None:
                    self.ui.append_message("Hold", "运行中重采前请先点击 Hold / 冻结。")
                    self.ui.publish(
                        active_controls=self._teleop_controls(),
                        control_hint="运行中重采前请先点击 Hold / 冻结；Hold 只停臂，不会自动重采。",
                        teleop_running=True,
                        gripper_control_enabled=self.gripper_control_enabled(),
                    )
                return False
            if request_from_hold:
                self._hold_active = False
                self._hold_request_reason = "manual"
                self._last_calibration_started_from_hold = True
            self._gripper_control_enabled = False
            self._tracker_swap_locked = True
            if self.ui is not None:
                self.ui.publish(
                    active_controls=["q"],
                    control_hint=(
                        "已确认开始，VIVE tracker 左右映射已锁定；"
                        f"{DEFAULT_RUNTIME_HOME_START_DELAY_SEC:.0f} 秒后采集 runtime home。"
                    ),
                    teleop_running=False,
                    home_ready=False,
                    gripper_control_enabled=False,
                )
        return requested

    def _request_runtime_home_calibration_from_gui(self) -> None:
        if self.ui is None:
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中重新标定前请先点击 Hold / 冻结。"
            self.ui.append_message("Calibration", message)
            self.ui.publish(
                active_controls=self._teleop_controls(),
                control_hint=message,
                teleop_running=True,
                gripper_control_enabled=self.gripper_control_enabled(),
            )
            return
        self._calibration_requested = True
        self._calibration_request_from_hold = bool(self._hold_active)
        self._gripper_control_enabled = False
        self.ui.publish(
            stage="准备重新标定 runtime home",
            active_controls=["q"],
            control_hint="已请求重新进入 Home 标定流程；后端将先倒计时，再采集当前 VIVE/机械臂 home。",
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            capture_waiting_for_samples=False,
        )
        self.ui.append_message("Calibration", "已请求重新进入 runtime home 标定流程。")

    def _publish_movement_multiple(self, *, message: str = "") -> None:
        if self.ui is None:
            return
        self.ui.publish(**_movement_multiple_ui_state(self.args))
        if message:
            self.ui.append_message("MovementMultiple", message)

    def _set_movement_multiple_value(self, value: float, *, source: str) -> float | None:
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不修改 movement_multiple；请先停止当前 episode。"
            _log("MovementMultiple", f"{source}: rejected while episode busy")
            self._publish_movement_multiple(message=message)
            return None
        if self._teleop_started and not self._hold_active:
            message = "运行中不直接修改 movement_multiple；请先 Hold，避免目标位置因倍率变化发生跳变。"
            _log("MovementMultiple", f"{source}: rejected while teleop running")
            self._publish_movement_multiple(message=message)
            return None
        try:
            movement_multiple = _apply_movement_multiple(self.args, self.components, value)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            message = f"移动倍率无效: {exc}"
            _log("MovementMultiple", message)
            self._publish_movement_multiple(message=message)
            return None
        message = (
            f"movement_multiple={movement_multiple:.3f}; "
            "只缩放 tracker 相对 runtime home 的平移，旋转保持原逻辑。"
        )
        _log("MovementMultiple", f"{source}: {message}")
        self._publish_movement_multiple(message=message)
        return movement_multiple

    def _set_movement_multiple_from_gui(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except ValueError:
            self._set_movement_multiple_value(float("nan"), source="gui_set")
            return
        self._set_movement_multiple_value(value, source="gui_set")

    def _handle_movement_multiple_event(self, event: str) -> None:
        current = _movement_multiple_value(self.args)
        step = MOVEMENT_MULTIPLE_GUI_STEP
        if event == "movement_multiple_down":
            value = max(MOVEMENT_MULTIPLE_GUI_MIN, current - step)
        elif event == "movement_multiple_up":
            value = current + step
        elif event == "movement_multiple_reset":
            value = MOVEMENT_MULTIPLE_DEFAULT
        elif event == "movement_multiple_set":
            value = current
        else:
            return
        self._set_movement_multiple_value(value, source=event)

    def _set_gripper_force_from_gui(self, text: str) -> None:
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不修改夹爪力度；当前 episode 保持任务配置。"
            _log("GripperForce", "gui_set: rejected while episode busy")
            if self.ui is not None:
                self.publish_gripper_status(force=True)
                self.ui.append_message("GripperForce", message)
            return
        try:
            payload = json.loads(str(text or "{}"))
        except json.JSONDecodeError as exc:
            payload = None
            parse_error = str(exc)
        else:
            parse_error = ""
        if not isinstance(payload, Mapping):
            message = (
                f"夹爪力度参数无效: {parse_error or '必须是包含 left/right 的 JSON 对象。'}"
            )
            _log("GripperForce", message)
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return

        requested: dict[str, float] = {}
        try:
            for side in ("left", "right"):
                value = payload.get(side)
                if value is None:
                    continue
                if isinstance(value, bool):
                    raise ValueError(f"{side} 必须是 0 到 1 的数字")
                force = float(value)
                if not np.isfinite(force) or not 0.0 <= force <= 1.0:
                    raise ValueError(f"{side} 必须在 [0, 1] 内")
                requested[side] = force
        except (TypeError, ValueError) as exc:
            message = f"夹爪力度参数无效: {exc}。"
            _log("GripperForce", message)
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return
        if not requested:
            message = "未提供可设置的左/右夹爪力度。"
            _log("GripperForce", message)
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return

        active_sides = set(self.components.active_sides)
        updated: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for side, force in requested.items():
            if side not in active_sides:
                skipped.append(f"{side}:未启用")
                continue
            backend = _gripper_backend_for_side(self.components, side)
            if backend is None:
                skipped.append(f"{side}:无 DM-Tac 夹爪")
                continue
            try:
                backend.set_torque(force)
            except Exception as exc:  # noqa: BLE001 - report per-side UI input failure.
                failed.append(f"{side}:{type(exc).__name__}: {exc}")
                continue
            self._manual_gripper_force_by_side[side] = force
            updated.append(f"{side}={force:.3f}")

        parts: list[str] = []
        if updated:
            parts.append("已设置 " + ", ".join(updated))
        if skipped:
            parts.append("未设置 " + ", ".join(skipped))
        if failed:
            parts.append("失败 " + ", ".join(failed))
        message = "；".join(parts) or "夹爪力度未改变。"
        _log("GripperForce", f"gui_set: {message}")
        if self.ui is not None:
            self.publish_gripper_status(force=True)
            self.ui.append_message("GripperForce", message)

    def _restore_gripper_task_policy_for_collection(self) -> None:
        if not self._manual_gripper_force_by_side:
            return
        previous = ", ".join(
            f"{side}={force:.3f}"
            for side, force in sorted(self._manual_gripper_force_by_side.items())
        )
        self._manual_gripper_force_by_side.clear()
        self._apply_current_gripper_task_policy()
        message = f"数采开始前已恢复任务夹爪力度配置（临时设置: {previous}）。"
        _log("GripperForce", message)
        if self.ui is not None:
            self.ui.append_message("GripperForce", message)

    def _apply_episode_start_payload(self, text: str) -> None:
        try:
            payload = json.loads(str(text or "{}"))
        except json.JSONDecodeError as exc:
            if self.ui is not None:
                self.ui.append_message("Dataset", f"数采参数无效: {exc}")
            return
        if not isinstance(payload, Mapping):
            return
        dataset_base_root = str(payload.get("dataset_base_root") or "").strip()
        task_key = str(payload.get("task_key") or "").strip()
        english_name = str(payload.get("task_english_name") or "").strip()
        chinese_name = str(payload.get("task_chinese_name") or "").strip()
        payload_task_type = str(payload.get("task_type") or "").strip()
        payload_round_trip_phase = str(payload.get("round_trip_phase") or "").strip()
        task_def = _task_definition_for_key(self.args, task_key)
        if task_def is not None:
            if not english_name:
                english_name = str(task_def.get("english_name") or task_key)
            if not chinese_name:
                chinese_name = str(task_def.get("chinese_name") or "")
            gripper_task = str(task_def.get("gripper_task") or task_key)
            task_type = (
                gui_collection_tasks.normalize_task_type(payload_task_type)
                if payload_task_type
                else gui_collection_tasks.normalize_task_type(task_def.get("task_type", ""))
            )
        else:
            gripper_task = task_key
            task_type = (
                gui_collection_tasks.normalize_task_type(payload_task_type)
                if payload_task_type
                else gui_collection_tasks.normalize_task_type(getattr(self.args, "collection_task_type", ""))
            )
        if dataset_base_root:
            self.args.dataset_root = dataset_base_root
            self.args.dataset_base_root = str(_expand_cli_path(dataset_base_root))
        if gripper_task:
            self.args.gripper_task = gripper_task
        task_name = (
            english_name
            or gripper_task
            or gui_collection_tasks.strip_round_trip_suffix(
                str(getattr(self.args, "collection_task_name", "") or "")
            )
        )
        if task_name:
            self.args.collection_task_name = task_name
        if chinese_name:
            self.args.collection_task_name_zh = chinese_name
        base_task_id = task_key or _default_task_id_from_name(task_name or gripper_task)
        self.args.collection_task_id = base_task_id
        self.args.collection_task_key = task_key
        self.args.collection_task_type = task_type
        task_description = (
            str(task_def.get("description") or "").strip()
            if task_def is not None
            else ""
        )
        self.args.collection_task_description = task_description or chinese_name or task_name
        if task_type == gui_collection_tasks.ROUND_TRIP_TASK_TYPE:
            phase = (
                gui_collection_tasks.normalize_round_trip_phase(payload_round_trip_phase)
                if payload_round_trip_phase
                else gui_collection_tasks.ROUND_TRIP_IN
            )
            self.args.collection_round_trip_phase = phase
            self.args.collection_round_trip_base_name = task_name
            self.args.collection_round_trip_base_id = base_task_id
            self.args.collection_round_trip_base_name_zh = chinese_name
            self.args.collection_round_trip_base_description = self.args.collection_task_description
        else:
            self.args.collection_round_trip_phase = gui_collection_tasks.ROUND_TRIP_IN
            self.args.collection_round_trip_base_name = gui_collection_tasks.strip_round_trip_suffix(task_name)
            self.args.collection_round_trip_base_id = gui_collection_tasks.strip_round_trip_suffix(base_task_id)
            self.args.collection_round_trip_base_name_zh = chinese_name
            self.args.collection_round_trip_base_description = self.args.collection_task_description
        _refresh_formal_dataset_layout(self.args)
        self._apply_current_gripper_task_policy()
        if self.ui is not None:
            self.ui.publish(**_collection_task_ui_state(self.args))
            self.ui.append_message(
                "Dataset",
                (
                    f"数采任务已更新: {self.args.collection_task_name} "
                    f"({self.args.collection_task_id}) -> {self.args.dataset_root}"
                ),
            )

    def _toggle_round_trip_collection_task(self, *, source: str) -> bool:
        if self._episode_busy():
            if self.ui is not None:
                self.ui.append_message("Dataset", "数采录制、暂停、倒计时或保存中不切换 in/out；请先停止当前 episode。")
                self.ui.publish(**_collection_task_ui_state(self.args))
            return False
        if (
            gui_collection_tasks.normalize_task_type(getattr(self.args, "collection_task_type", ""))
            != gui_collection_tasks.ROUND_TRIP_TASK_TYPE
        ):
            if self.ui is not None:
                self.ui.append_message("Dataset", "当前是一般类任务，不需要切换 in/out。")
                self.ui.publish(**_collection_task_ui_state(self.args))
            return False
        if not _toggle_collection_round_trip_phase(self.args):
            return False
        phase = str(getattr(self.args, "collection_round_trip_phase", "") or "")
        label = "自动" if source == "auto_after_episode" else "手动"
        message = (
            f"来回任务已{label}切换到 _{phase}: "
            f"{self.args.collection_task_name} ({self.args.collection_task_id}) -> {self.args.dataset_root}"
        )
        _log("Dataset", message)
        if self.ui is not None:
            self.ui.publish(**_collection_task_ui_state(self.args))
            self.ui.append_message("Dataset", message)
        return True

    def advance_round_trip_task_after_saved_episode(self) -> bool:
        return self._toggle_round_trip_collection_task(source="auto_after_episode")

    def _apply_current_gripper_task_policy(self) -> None:
        self._manual_gripper_force_by_side.clear()
        _clear_gripper_policy_cache(self.args)
        policies = _resolved_gripper_policies_by_side(self.args, self.components.active_sides)
        self.components.gripper_policy.clear()
        self.components.gripper_policy.update(_resolved_gripper_policy(self.args))
        for side in self.components.active_sides:
            backend = _hand_backend_for_side(self.components, side)
            policy = policies.get(side)
            if not isinstance(backend, DMTacGripperBackend) or policy is None:
                continue
            backend.config.speed = float(policy["speed"])
            backend.config.torque = float(policy["torque"])
            backend.config.max_action = float(policy["max_action"])
            backend.config.send_threshold = float(policy["send_threshold"])
            backend.config.command_hz = float(policy["command_hz"])
            backend.config.continuous_grasp = bool(policy["continuous_grasp"])
            backend.config.telemetry_hz = float(policy["telemetry_hz"])
        if self.ui is not None:
            self.publish_gripper_status(force=True)

    def request_hold(self, *, reason: str = "manual") -> None:
        if not self._teleop_started:
            return
        self._hold_requested = True
        self._hold_active = True
        self._gripper_control_enabled = False
        self._hold_request_reason = str(reason or "manual")
        self._q6_hold_recalibration_gesture.reset_for_hold()
        is_q6_gesture = self._hold_request_reason == "q6_gesture"
        if self.ui is not None:
            self.ui.publish(
                stage="Hold / 停臂保持",
                active_controls=self._hold_controls(),
                control_hint=(
                    "机械臂保持当前位置；请松开双手 Q6 后三维到释放区间 1 秒，再同时握紧 1 秒以重新标定。"
                    if is_q6_gesture
                    else (
                        "机械臂保持当前位置；单臂模式可先点击“交换 Tracker”，"
                            "然后重新点击“开始遥操”。"
                        if len(tuple(self.components.active_sides)) == 1
                        else (
                            "机械臂保持当前位置；需要重新标定时，"
                            "摆好 VIVE/Q6 后点击“开始遥操”。"
                        )
                    )
                ),
                capture_stage="runtime_home",
                capture_phase="ready",
                teleop_running=False,
                capture_waiting_for_samples=False,
                debug_ik_timing=self.ik_debug_timing_enabled(),
                debug_ik_logging=self._ik_debug_logging_active,
                ik_debug_log_path=self._ik_debug_log_path,
                gripper_control_enabled=False,
                **_movement_multiple_ui_state(self.args),
            )
            message = (
                "Q6 手势已请求冻结机械臂；松开双手后三维到释放区间 1 秒，再次握紧 1 秒会重采 runtime home。"
                if is_q6_gesture
                else "已请求冻结机械臂；不会自动重采 runtime home。"
            )
            self.ui.append_message("Hold", message)

    def consume_hold_request(self) -> bool:
        requested = bool(self._hold_requested)
        self._hold_requested = False
        return requested

    def hold_active(self) -> bool:
        return bool(self._hold_active)

    def hold_request_reason(self) -> str:
        return str(self._hold_request_reason or "manual")

    def consume_episode_start_request(self) -> bool:
        requested = bool(self._episode_start_requested)
        self._episode_start_requested = False
        return bool(self.terminal.consume_episode_start_request() or requested)

    def consume_episode_stop_request(self) -> bool:
        requested = bool(self._episode_stop_requested)
        self._episode_stop_requested = False
        return bool(self.terminal.consume_episode_stop_request() or requested)

    def consume_camera_visualization_toggle_request(self) -> bool:
        requested = bool(self._camera_visualization_toggle_requested)
        self._camera_visualization_toggle_requested = False
        return requested

    def consume_tactile_visualization_toggle_request(self) -> bool:
        requested = bool(self._tactile_visualization_toggle_requested)
        self._tactile_visualization_toggle_requested = False
        return requested

    def consume_ik_debug_log_toggle_request(self) -> bool:
        requested = bool(self._ik_debug_log_toggle_requested)
        self._ik_debug_log_toggle_requested = False
        return requested

    def ik_debug_timing_enabled(self) -> bool:
        return bool(_debug_timing_modules(self.args).ik_profile) or bool(
            self._ik_debug_logging_active
        )

    def gripper_control_enabled(self) -> bool:
        return bool(self._gripper_control_enabled and _has_q6_gripper_control(self.components))

    def _episode_countdown_controls(self) -> list[str]:
        return _gui_episode_countdown_controls(
            self._gui_control_state(),
            self._gui_events(),
        )

    def _poll_q6_hold_recalibration_gesture(self, *, now: float) -> None:
        gesture = self._q6_hold_recalibration_gesture
        if not bool(gesture.config.enabled):
            return
        if set(self.components.active_sides) != {"left", "right"}:
            gesture.reset_for_running()
            return
        source = _hand_q6_source_from_components(self.components)
        if source is None:
            gesture.update(
                left_q6=None,
                right_q6=None,
                teleop_started=self._teleop_started,
                hold_active=self._hold_active,
                now=float(now),
            )
            return
        channels = source.get_latest_channels()
        left_q6 = Q6HoldRecalibrationGesture.q6_values_from_channel(channels.get("left_hand_q6"))
        right_q6 = Q6HoldRecalibrationGesture.q6_values_from_channel(channels.get("right_hand_q6"))
        event = gesture.update(
            left_q6=left_q6,
            right_q6=right_q6,
            teleop_started=self._teleop_started,
            hold_active=self._hold_active,
            now=float(now),
        )
        if event == Q6HoldRecalibrationGesture.EVENT_HOLD:
            self.request_hold(reason="q6_gesture")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 closed for hold window; requested hold",
                now=now,
            )
        elif event == Q6HoldRecalibrationGesture.EVENT_REARMED:
            if self.ui is not None:
                self.ui.append_message("Q6Gesture", "已检测到双手后三维进入释放区间；再次同时握紧 1 秒将重采 runtime home。")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 released; waiting for second close to recalibrate",
                now=now,
            )
        elif event == Q6HoldRecalibrationGesture.EVENT_RESTART:
            self._calibration_requested = True
            self._calibration_request_from_hold = True
            self._gripper_control_enabled = False
            if self.ui is not None:
                self.ui.publish(
                    stage="准备重采 runtime home",
                    active_controls=["q", *self._visualization_controls()],
                    control_hint="Q6 手势已确认；即将重采 runtime home 并恢复遥操。",
                    teleop_running=False,
                    gripper_control_enabled=False,
                )
                self.ui.append_message("Q6Gesture", "已检测到第二次同时握紧；准备重采 runtime home。")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 closed again; requested runtime home recalibration",
                now=now,
            )

    def set_ik_debug_logging_state(self, *, active: bool, path: str = "") -> None:
        self._ik_debug_logging_active = bool(active)
        self._ik_debug_log_path = str(path or "")
        if self.ui is not None:
            self.ui.publish(
                debug_ik_timing=self.ik_debug_timing_enabled(),
                debug_ik_logging=self._ik_debug_logging_active,
                ik_debug_log_path=self._ik_debug_log_path,
            )

    def set_episode_start_countdown_state(self, *, remaining_sec: float, message: str | None = None) -> None:
        if self.ui is None:
            return
        remaining = max(0.0, float(remaining_sec))
        self._episode_countdown_ui_active = True
        self._episode_recording_ui_active = False
        self._episode_finalizing_ui_active = False
        self.ui.publish(
            active_controls=self._episode_countdown_controls(),
            episode_recording_available=self._episode_recording_available(),
            episode_recording=False,
            episode_dir="",
            episode_frame_count=None,
            episode_sample_hz=None,
            episode_target_hz=float(getattr(self.args, "collection_warning_hz", 15.0)),
            episode_tactile_write_hz={},
            episode_tactile_write_hz_mean=None,
            episode_tactile_frame_counts={},
            episode_frequency_warning={"active": False, "message": ""},
            episode_event_status="preparing",
            episode_event_message=(
                message
                or f"准备开始数采：{remaining:.1f} 秒后开始写入，请保持当前动作和画面稳定。"
            ),
            episode_last_episode_index=self._episode_last_episode_index,
            episode_last_frame_count=self._episode_last_frame_count,
            **_collection_task_ui_state(self.args),
            **self._pose_snapshot_ui_state(),
        )

    def set_episode_recording_state(
        self,
        *,
        active: bool,
        episode_dir: str | Path | None = None,
        frame_count: int | None = None,
        frequency_status: Mapping[str, object] | None = None,
        event_status: str | None = None,
        event_message: str | None = None,
        episode_index: int | None = None,
    ) -> None:
        if self.ui is None:
            return
        self._episode_recording_ui_active = bool(active)
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = bool((event_status or "") == "finalizing")
        if episode_index is not None:
            self._episode_last_episode_index = int(episode_index)
        if frame_count is not None:
            self._episode_last_frame_count = int(frame_count)
        frequency_payload = dict(frequency_status or {})
        warning_payload = dict(frequency_payload.get("warning") or {})
        if not active:
            warning_payload = {"active": False, "message": ""}
        tactile_write_hz_payload = dict(frequency_payload.get("tactile_write_hz") or {})
        tactile_frame_counts_payload = dict(frequency_payload.get("tactile_frame_counts") or {})
        self.ui.publish(
            active_controls=self._current_active_controls(),
            episode_recording_available=self._episode_recording_available(),
            episode_recording=bool(active),
            episode_dir="" if episode_dir is None else str(episode_dir),
            episode_frame_count=frame_count,
            episode_sample_hz=frequency_payload.get("sample_hz") if active else None,
            episode_target_hz=frequency_payload.get("target_hz") if active else None,
            episode_tactile_write_hz=tactile_write_hz_payload if active else {},
            episode_tactile_write_hz_mean=(
                frequency_payload.get("tactile_write_hz_mean") if active else None
            ),
            episode_tactile_frame_counts=tactile_frame_counts_payload if active else {},
            episode_frequency_warning=warning_payload,
            episode_event_status=event_status or ("recording" if active else "idle"),
            episode_event_message=event_message or (
                "正在写入正式数据集；结束前请确认画面、触觉和机械臂状态正常。"
                if active
                else "点击“开始数采”或按 A 开始；再次按 A 停止。"
            ),
            episode_last_episode_index=self._episode_last_episode_index,
            episode_last_frame_count=self._episode_last_frame_count,
            **_collection_task_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
        )

    def set_episode_finalize_state(
        self,
        *,
        status: str,
        message: str,
        episode_index: int | None = None,
        frame_count: int | None = None,
    ) -> None:
        if self.ui is None:
            return
        self._episode_recording_ui_active = False
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = False
        if episode_index is not None:
            self._episode_last_episode_index = int(episode_index)
        if frame_count is not None:
            self._episode_last_frame_count = int(frame_count)
        self.ui.publish(
            active_controls=self._current_active_controls(),
            episode_recording_available=self._episode_recording_available(),
            episode_recording=False,
            episode_frame_count=frame_count,
            episode_sample_hz=None,
            episode_target_hz=float(getattr(self.args, "collection_warning_hz", 15.0)),
            episode_tactile_write_hz={},
            episode_tactile_write_hz_mean=None,
            episode_tactile_frame_counts={},
            episode_frequency_warning={"active": False, "message": ""},
            episode_event_status=str(status),
            episode_event_message=str(message),
            episode_last_episode_index=self._episode_last_episode_index,
            episode_last_frame_count=self._episode_last_frame_count,
            **_collection_task_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
        )

    def set_visualization_state(
        self,
        *,
        camera_active: bool | None = None,
        tactile_active: bool | None = None,
        status: str | None = None,
    ) -> None:
        if camera_active is not None:
            self._camera_visualization_active = bool(camera_active)
        if tactile_active is not None:
            self._tactile_visualization_active = bool(tactile_active)
        if self.ui is None:
            return
        self.ui.publish(
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            visualization_status="" if status is None else str(status),
            active_controls=(
                self._teleop_controls()
                if self._teleop_started and not self._hold_active
                else self._hold_controls()
                if self._teleop_started
                else self._pre_home_controls()
            ),
            **_movement_multiple_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
        )

    def before_calibration_start_delay(
        self,
        delay_sec: float,
        *,
        arm_mode: str,
    ) -> bool:
        del arm_mode
        total = max(0.0, float(delay_sec))
        deadline = time.time() + total
        while True:
            remaining = max(0.0, deadline - time.time())
            if self.ui is not None:
                self.ui.publish(
                    stage="即将开始 runtime home",
                    active_controls=["q"],
                    control_hint=f"请松开按键并保持姿态，{remaining:.1f}s 后开始。",
                    capture_stage="runtime_home",
                    capture_phase="preparing",
                    teleop_running=False,
                    capture_prep_total_sec=total,
                    capture_prep_remaining_sec=remaining,
                    capture_waiting_for_samples=False,
                )
            self.publish_raw_tracker_pose_state(force=True)
            if remaining <= 0.0:
                break
            self.terminal.poll()
            if self.terminal.should_quit:
                self.should_quit = True
            if self.ui is not None:
                for event in self.ui.consume_events():
                    if event == "q":
                        self.should_quit = True
            if self.should_quit:
                return False
            time.sleep(min(0.05, remaining))
        if self.ui is not None:
            self.ui.publish(
                stage="采集 runtime home",
                active_controls=["q"],
                control_hint="正在采集当前 tracker 和机械臂 home，请保持静止。",
                capture_stage="runtime_home",
                capture_phase="capturing",
                teleop_running=False,
                capture_samples=0,
                capture_left_samples=0,
                capture_right_samples=0,
                capture_min_samples=int(getattr(self.args, "calibration_samples", 0)),
                capture_valid=False,
                capture_left_valid=False,
                capture_right_valid=False,
                capture_elapsed_sec=0.0,
                capture_duration_sec=float(getattr(self.args, "calibration_timeout", 0.0)),
                capture_prep_remaining_sec=0.0,
                capture_waiting_for_samples=False,
            )
        self.publish_raw_tracker_pose_state(force=True)
        return True

    def publish_force_control_warmup_progress(
        self,
        *,
        elapsed_sec: float,
        total_sec: float,
        active_sides: Sequence[str],
    ) -> None:
        if self.ui is None:
            return
        total = max(0.1, float(total_sec))
        elapsed = max(0.0, float(elapsed_sec))
        remaining = max(0.0, total - elapsed)
        side_text = "/".join(str(side) for side in active_sides) or "arm"
        self.ui.publish(
            stage="预热 RM65 力控",
            active_controls=["q"],
            control_hint=(
                f"首次启动正在预热 {side_text} pose impedance；"
                "完成后请再次点击“开始遥操”进入准备和 Home 标定。"
            ),
            capture_stage="force_control_warmup",
            capture_phase="preparing",
            teleop_running=False,
            home_ready=False,
            capturing=False,
            capture_prep_total_sec=total,
            capture_prep_remaining_sec=remaining,
            capture_waiting_for_samples=False,
            gripper_control_enabled=False,
        )

    def publish_teleop_running(self, *, home_ready: bool = True, message: str | None = None) -> None:
        was_hold_active = bool(self._hold_active)
        was_hold_restart = bool(self._last_calibration_started_from_hold)
        self._last_calibration_started_from_hold = False
        self._teleop_started = True
        self._hold_active = False
        self._gripper_control_enabled = _has_q6_gripper_control(self.components)
        if was_hold_active or was_hold_restart:
            self._q6_hold_recalibration_gesture.mark_runtime_restarted()
        else:
            self._q6_hold_recalibration_gesture.reset_for_running()
        if self.ui is None:
            return
        control_hint = "遥操已开始；Q6 夹爪控制会自动开启；Hold 只冻结机械臂，重采需再次点击开始。"
        if self._q6_hold_recalibration_gesture.config.enabled:
            control_hint = (
                "遥操已开始；Q6 手势模式已开启：双手后三维握紧 1 秒进入 Hold，"
                "松开到释放区间 1 秒后再次握紧 1 秒会重采 runtime home。"
            )
        gripper_status = _build_gripper_status(self.components)
        self.ui.publish(
            stage="RM65 真机遥操运行中",
            active_controls=self._teleop_controls(),
            control_hint=control_hint,
            home_ready=bool(home_ready),
            capture_stage="teleop",
            capture_phase="teleop",
            teleop_running=True,
            capturing=False,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=None,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=None,
            capture_waiting_for_samples=False,
            debug_ik_timing=self.ik_debug_timing_enabled(),
            debug_ik_logging=self._ik_debug_logging_active,
            ik_debug_log_path=self._ik_debug_log_path,
            ik_timing={
                "available": False,
                "message": "等待第一帧 IK timing 元数据。",
            },
            ik_warning=_build_ik_warning_payload(active_command=None),
            pose_quality=_build_pose_quality_display_payload(None),
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=self.gripper_control_enabled(),
            gripper_status=gripper_status,
            tactile_camera_status=_build_tactile_camera_status(gripper_status),
            episode_recording_available=self._episode_recording_available(),
            **_collection_task_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
        )
        _log(
            "WebGUI",
            f"teleop state published url={str(getattr(self.ui, 'url', '') or '<unknown>')} "
            f"gripper_control={'on' if self.gripper_control_enabled() else 'off'} "
            f"q6_sides={[side for side in self.components.active_sides if _gripper_retargeter_for_side(self.components, side) is not None]}",
        )
        self.ui.append_message("RealTeleop", message or "runtime home 完成，已进入真机遥操。")
        self.publish_raw_tracker_pose_state(force=True)
        self.publish_gripper_status(force=True)

    def publish_calibration_waiting(self, message: str = "") -> None:
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        if self._teleop_started:
            self._hold_active = True
            self._gripper_control_enabled = False
            self._q6_hold_recalibration_gesture.reset_for_hold()
        self._tracker_swap_locked = bool(self._teleop_started)
        if self.ui is None:
            return
        if message:
            self.ui.append_message("Calibration", message)
        self.ui.publish(
            stage="RM65 真机遥操准备" if not self._teleop_started else "Hold / 停臂保持",
            active_controls=self._hold_controls()
            if self._teleop_started
            else self._pre_home_controls(),
            control_hint=(
                message
                or "请选择：标定/加载/保存 VIVE 坐标系，或点击“开始遥操”采集 runtime Home。"
            ),
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            capturing=False,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=None,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
            capture_waiting_for_samples=False,
            gripper_control_enabled=False,
            **_collection_task_ui_state(self.args),
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
        )

    def publish_runtime_home_progress(
        self,
        *,
        sample_counts: Mapping[str, int],
        valid_by_side: Mapping[str, bool],
        min_samples: int,
        elapsed_sec: float,
        timeout_sec: float,
        waiting_for_samples: bool,
        now: float | None = None,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        complete = all(int(sample_counts.get(side, 0)) >= int(min_samples) for side in self.components.active_sides)
        if (
            not complete
            and current_time - self._last_capture_progress_publish_at < self._capture_progress_publish_interval_sec
        ):
            return
        self._last_capture_progress_publish_at = current_time
        left_samples = int(sample_counts.get("left", 0)) if "left" in self.components.active_sides else None
        right_samples = int(sample_counts.get("right", 0)) if "right" in self.components.active_sides else None
        single_count = None
        if len(self.components.active_sides) == 1:
            single_count = int(sample_counts.get(self.components.active_sides[0], 0))
        self.ui.publish(
            stage="采集 runtime home",
            capture_stage="runtime_home",
            capture_phase="capturing",
            capturing=True,
            capture_samples=single_count,
            capture_left_samples=left_samples,
            capture_right_samples=right_samples,
            capture_min_samples=int(min_samples),
            capture_valid=complete,
            capture_left_valid=valid_by_side.get("left") if "left" in self.components.active_sides else None,
            capture_right_valid=valid_by_side.get("right") if "right" in self.components.active_sides else None,
            capture_elapsed_sec=float(elapsed_sec),
            capture_duration_sec=float(timeout_sec),
            capture_waiting_for_samples=bool(waiting_for_samples and not complete),
            control_hint=(
                "正在采集 runtime home 样本，请保持 VIVE 和机械臂静止。"
                if not complete
                else "runtime home 样本已采满，正在完成标定。"
            ),
        )

    def publish_fix_arm_home_capture_progress(self, progress) -> None:
        if self.ui is None or not progress.active:
            return
        side_label = "左臂" if progress.side == "left" else "右臂"
        if progress.completed:
            message = progress.message or f"{side_label} home 重采完成，已恢复遥操。"
        elif progress.failed:
            message = progress.message or f"{side_label} home 重采失败，保持固定。"
        else:
            message = (
                f"{side_label}保持固定并重采 home: "
                f"{progress.sample_count}/{progress.min_samples} samples"
            )
        self.ui.publish(
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            active_controls=self._current_active_controls(),
            control_hint=message,
        )
        if progress.completed or progress.failed:
            self.ui.append_message("FixArm", message)

    def publish_init_arm_home_capture_progress(self, progress) -> None:
        if self.ui is None or not progress.active:
            return
        side_label = "左臂" if progress.side == "left" else "右臂"
        if progress.completed:
            message = progress.message or f"{side_label} home 重采完成，已恢复遥操。"
        elif progress.failed:
            message = progress.message or f"{side_label} home 重采失败，继续保持 init pose。"
        else:
            message = (
                f"{side_label}保持 init pose 并重采 home: "
                f"{progress.sample_count}/{progress.min_samples} samples"
            )
        self.ui.publish(
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            active_controls=self._current_active_controls(),
            control_hint=message,
        )
        if progress.completed or progress.failed:
            self.ui.append_message("InitArm", message)

    def publish_init_arm_arrival_state(self) -> None:
        """Refresh the display-only measured arrival state while init-arm is active."""

        if self.ui is None:
            return
        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None or not bool(getattr(controller, "active", False)):
            return
        self.ui.publish(**_init_arm_ui_state(self.components))

    def _calibrate_vive_coordinate_profile(self) -> None:
        if self.ui is None:
            return
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不能重新标定 VIVE 坐标系；请先停止当前 episode。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(control_hint=message)
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中不能重新标定 VIVE 坐标系；请先点击 Hold / 冻结。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(control_hint=message)
            return
        try:
            path = _calibrate_gui_vive_coordinate_profile(self.args, self.components, self.ui)
        except Exception as exc:  # noqa: BLE001
            message = f"标定 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                active_controls=[
                    "p",
                    "q",
                    *self._visualization_controls(),
                    *self._vive_calibration_controls(allow_load=True),
                    *self._movement_multiple_controls(),
                ],
                capture_stage="runtime_home",
                capture_phase="ready",
                capturing=False,
                capture_waiting_for_samples=False,
                **_movement_multiple_ui_state(self.args),
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=False,
                )
            )
            return
        message = f"VIVE 坐标系已重新标定并自动保存：{path}；请重新采集 runtime Home 后再遥操。"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            stage="VIVE 坐标系已保存",
                active_controls=[
                    "p",
                    "q",
                *self._visualization_controls(),
                *self._vive_calibration_controls(allow_load=True),
                *self._movement_multiple_controls(),
            ],
            home_ready=False,
            capture_stage="runtime_home",
            capture_phase="ready",
            capturing=False,
            capture_waiting_for_samples=False,
            control_hint="坐标系已保存；请点击“开始遥操”重新采集 runtime Home。",
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="坐标系标定并保存成功",
                path=path,
                loaded=True,
            ),
        )

    def _save_vive_calibration_profile(self) -> None:
        if self.ui is None:
            return
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不保存 VIVE 坐标系 profile；请先停止当前 episode。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(control_hint=message)
            return
        try:
            path = _save_gui_vive_calibration_profile(self.args, self.components)
        except Exception as exc:  # noqa: BLE001
            message = f"保存 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=None,
                )
            )
            return
        message = f"已保存 VIVE 坐标系 profile：{path}"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="保存成功",
                path=path,
                loaded=None,
            )
        )

    def _load_vive_calibration_profile(self) -> None:
        if self.ui is None:
            return
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不能加载 VIVE 坐标系；请先停止当前 episode。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                control_hint=message,
                **_vive_calibration_profile_ui_state(self.args, self.components),
            )
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中不能加载 VIVE 坐标系；请先点击 Hold / 冻结。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                control_hint=message,
                **_vive_calibration_profile_ui_state(self.args, self.components),
            )
            return
        try:
            path = _load_gui_vive_calibration_profile(self.args, self.components)
        except Exception as exc:  # noqa: BLE001
            message = f"加载 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=False,
                )
            )
            return
        self._gripper_control_enabled = False
        message = f"已加载 VIVE 坐标系 profile：{path}；请重新采集 runtime home。"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            stage="已加载 VIVE 坐标系",
                active_controls=[
                    "p",
                    "q",
                *self._visualization_controls(),
                *self._vive_calibration_controls(allow_load=True),
                *self._movement_multiple_controls(),
            ],
            control_hint="VIVE 坐标系已加载；请点击“开始遥操”重新采集 runtime home。",
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="加载成功，等待重采 runtime home",
                path=path,
                loaded=True,
            ),
        )

    def publish_ik_timing_state(
        self,
        payload: Mapping[str, object],
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None or not self.ik_debug_timing_enabled():
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and current_time - self._last_ik_timing_publish_at < self._ik_timing_publish_interval_sec
        ):
            return
        self._last_ik_timing_publish_at = current_time
        self.ui.publish(debug_ik_timing=True, ik_timing=dict(payload))

    def publish_ik_warning_state(
        self,
        payload: Mapping[str, object],
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None:
            return
        warning = dict(payload)
        current_time = time.time() if now is None else float(now)
        codes_value = warning.get("codes")
        sources_value = warning.get("sources")
        codes = (
            codes_value
            if isinstance(codes_value, Sequence) and not isinstance(codes_value, (str, bytes))
            else ()
        )
        sources = (
            sources_value
            if isinstance(sources_value, Sequence) and not isinstance(sources_value, (str, bytes))
            else ()
        )
        signature = (
            bool(warning.get("active", False)),
            tuple(str(code) for code in codes if code is not None),
            tuple(str(source) for source in sources if source is not None),
            str(warning.get("message") or ""),
        )
        changed = signature != self._last_ik_warning_signature
        if (
            not force
            and not changed
            and current_time - self._last_ik_warning_publish_at < self._ik_warning_publish_interval_sec
        ):
            return
        self._last_ik_warning_signature = signature
        self._last_ik_warning_publish_at = current_time
        self.ui.publish(ik_warning=warning)

    def publish_pose_quality_state(
        self,
        payload: Mapping[str, object],
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and current_time - self._last_pose_quality_publish_at < self._pose_quality_publish_interval_sec
        ):
            return
        self._last_pose_quality_publish_at = current_time
        self.ui.publish(pose_quality=dict(payload))

    def publish_gripper_status(
        self,
        *,
        channels: dict[str, object] | None = None,
        command_status: Mapping[str, Mapping[str, object]] | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None or not _gripper_gui_available(self.components):
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and current_time - self._last_gripper_status_publish_at < self._gripper_status_publish_interval_sec
        ):
            return
        if channels is None:
            source = _gripper_input_source_from_components(self.components)
            channels = {} if source is None else source.get_latest_channels()
        self._last_gripper_status_publish_at = current_time
        gripper_status = _build_gripper_status(
            self.components,
            channels=channels,
            command_status=command_status,
            now=current_time,
        )
        self.ui.publish(
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=self.gripper_control_enabled(),
            gripper_status=gripper_status,
            tactile_camera_status=_build_tactile_camera_status(gripper_status, now=current_time),
        )

    def publish_camera_status(
        self,
        *,
        camera_sources: Mapping[str, RealSenseRGBSource] | None,
        preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
        camera_previewer: CameraPreviewer | None = None,
        now: float | None = None,
        force: bool = False,
        camera_self_check: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        current_time = time.time() if now is None else float(now)
        status = _build_camera_status(
            camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=current_time,
            depth_monitor=self._camera_depth_monitor,
        )
        status = _with_expected_wrist_depth_camera_statuses(
            status,
            args=self.args,
            now=current_time,
        )
        if self.ui is None:
            return status
        if (
            not force
            and current_time - self._last_camera_status_publish_at < self._camera_status_publish_interval_sec
        ):
            return status
        self._last_camera_status_publish_at = current_time
        updates: dict[str, object] = {
            "camera_visualization_available": self._camera_visualization_available(),
            "camera_visualization_active": self._camera_visualization_active,
            "camera_status": status,
        }
        if camera_self_check is not None:
            updates["camera_self_check"] = dict(camera_self_check)
        self.ui.publish(**updates)
        return status

    def _toggle_vive_tracker_swap(self) -> None:
        hold_swap_allowed = self._single_arm_tracker_swap_allowed_in_hold()
        if self._tracker_swap_locked and not hold_swap_allowed:
            if self.ui is not None:
                self.ui.append_message(
                    "InputMapping",
                    "已开始遥操准备，运行中不能互换 VIVE tracker。",
                )
            return
        self.args.swap_operator_sides = not bool(
            getattr(self.args, "swap_operator_sides", False)
        )
        _apply_vive_tracker_mapping(self.args, self.components)
        message = self._control_hint()
        if hold_swap_allowed:
            message = (
                f"{message} 单臂 Hold 状态已切换 tracker；"
                "请重新点击“开始遥操”采集 runtime home。"
            )
        _log("InputMapping", message)
        if self.ui is not None:
            payload: dict[str, object] = {"control_hint": message}
            if hold_swap_allowed:
                payload.update(
                    {
                        "active_controls": self._hold_controls(),
                        "home_ready": False,
                        "teleop_running": False,
                    }
                )
            self.ui.publish(**payload)
            self.ui.append_message("InputMapping", message)
            self.publish_raw_tracker_pose_state(force=True)

    def publish_raw_tracker_pose_state(
        self,
        *,
        channels: dict[str, object] | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        if not force and current_time - self._last_raw_pose_publish_at < self._raw_pose_publish_interval_sec:
            return
        if channels is None:
            source = _vive_pose_source_from_components(self.components)
            channels = {} if source is None else source.get_latest_channels()
        _publish_raw_tracker_pose_state(
            self.ui,
            channels,
            active_sides=self.components.active_sides,
            now=current_time,
        )
        self._last_raw_pose_publish_at = current_time

    def _control_hint(self) -> str:
        mapping = ", ".join(
            f"logical_{side}<-{_tracker_source_side_for_logical_side(self.args, side)}_tracker"
            for side in self.components.active_sides
        )
        return (
            f"VIVE tracker 映射: {mapping}. "
            "点击“交换 Tracker”只会交换 VIVE 输入；机械臂、手套串口、真手和 profile 都不变。"
        )
