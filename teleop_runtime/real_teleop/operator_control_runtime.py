"""Entity-side operator-control adapter for the real teleoperation runtime.

The legacy debug entrypoint owns hardware/runtime assembly.  It binds the
small, explicit support namespace below after importing this file, so this
module can own the coordinator without importing the legacy entrypoint back.
The binding is intentionally explicit: adding a new runtime dependency to the
GUI coordinator requires reviewing its ownership and side effects.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import inspect

from ..collection.episode_curation import EpisodeCurationSession
from ..operator_ui_ports import (
    OperatorUiPort,
    RealTeleopInputBatchPrioritizer,
)
from .foot_pad_controls import (
    FOOT_PAD_FIVE_PROGRESS_EVENTS,
    FOOT_PAD_TWO_PEDAL_STYLE_COUNTS,
    persist_five_pedal_test_receipt,
    persisted_five_pedal_test_verified,
)
from .dataset_selection import (
    RealTeleopDatasetSelectionError,
    apply_real_teleop_dataset_selection,
)
from .progress_contracts import (
    PROGRESS_ARM_CONTROL_MODE_STRICT,
    ProgressApplicationPort,
    ProgressArmControlModeSelectionCommand,
    ProgressEntryStepSelectionCommand,
    ProgressMutationCommand,
    ProgressProfileSelectionCommand,
    PromotableHoldPause,
    RealTeleopProgressWorkflowPort,
)
from .gripper_policy import (
    GripperForceRequestError,
    parse_gripper_force_request,
)
from .plus_curation_runtime import RealTeleopPlusCurationRuntime
from .tracker_side_check import evaluate_tracker_side_assignment
from .defaults import DEFAULT_RUNTIME_HOME_FRESH_DATA_TIMEOUT_SEC


_SUPPORT_NAMES = (
    "CameraPreviewer",
    "CaptureModalityContract",
    "DEFAULT_INIT_ARM_QPOS_SPEED_SCALE",
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "DMTacGripperBackend",
    "GripperForceDatasetContract",
    "EPISODE_GUI_EVENTS",
    "FixArmPoseSnapshotRecorder",
    "FootPadSpecialArmInput",
    "GuiControlEvents",
    "GuiControlState",
    "MOVEMENT_MULTIPLE_DEFAULT",
    "MOVEMENT_MULTIPLE_GUI_EVENTS",
    "MOVEMENT_MULTIPLE_GUI_MIN",
    "MOVEMENT_MULTIPLE_GUI_STEP",
    "MAX_INIT_ARM_QPOS_SPEED_SCALE",
    "Mapping",
    "MIN_INIT_ARM_QPOS_SPEED_SCALE",
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
    "_gui_finalizing_controls",
    "_gui_fix_arm_controls",
    "_gui_fix_init_allowed_in_controls",
    "_gui_hold_controls",
    "_gui_init_arm_controls",
    "_gui_movement_multiple_controls",
    "_gui_pose_snapshot_controls",
    "_gui_pre_home_controls",
    "_gui_single_arm_tracker_swap_allowed_in_hold",
    "_gui_teleop_controls",
    "_gui_visualization_controls",
    "_gui_vive_calibration_controls",
    "_hand_backend_for_side",
    "_hand_q6_source_from_components",
    "_has_q6_gripper_control",
    "_init_arm_ui_state",
    "_init_arm_qpos_speed_scale_for_side",
    "_load_gui_vive_calibration_profile",
    "_log",
    "_movement_multiple_ui_state",
    "_movement_multiple_value",
    "_publish_raw_tracker_pose_state",
    "_refresh_formal_dataset_layout",
    "_release_gui_qpos_init_transit",
    "_resolved_gripper_policies_by_side",
    "_resolved_gripper_policy",
    "_save_gui_vive_calibration_profile",
    "_start_gui_qpos_init",
    "_task_definition_for_key",
    "_toggle_collection_round_trip_phase",
    "_tracker_source_side_for_logical_side",
    "_vive_calibration_profile_ui_state",
    "_vive_pose_source_from_components",
    "_with_expected_wrist_depth_camera_statuses",
    "argparse",
    "bind_capture_modality_contract",
    "bind_gripper_force_contract",
    "build_real_teleop_progress_application",
    "gui_collection_tasks",
    "json",
    "np",
    "Path",
    "prioritize_input_batch",
    "run_best_effort_cleanup",
    "time",
)


_FOOT_PAD_TEST_LABELS = {
    "fix_arm_left": "踏板 1 / F13 → 固定左臂",
    "fix_arm_right": "踏板 2 / F14 → 固定右臂",
    "init_arm_left": "踏板 3 / F15 → 左臂精确 Init qpos",
    "init_arm_right": "踏板 4 / F16 → 右臂精确 Init qpos",
    "progress_undo": "踏板 5 / F17 → 撤销上一个子任务进度标注",
}
_TWO_PEDAL_TEST_LABELS = {
    "fix_arm_left": "踏板 1 / F13 → 仅左臂遥操（Fix 右臂）",
    "fix_arm_right": "踏板 2 / F14 → 仅右臂遥操（Fix 左臂）",
    "init_arm_left": "踏板 3 / F15 → 测试键位（双踏板模式不执行 Init）",
    "init_arm_right": "踏板 4 / F16 → 测试键位（双踏板模式不执行 Init）",
}
_FIVE_PEDAL_TEST_LABELS = {
    "fix_arm_left": "踏板 1 / F13 → 仅左臂遥操（Fix 右臂）",
    "fix_arm_right": "踏板 2 / F14 → 仅右臂遥操（Fix 左臂）",
    "init_arm_left": "踏板 3 / F15 → 进入下一子任务阶段",
    "init_arm_right": "踏板 4 / F16 → 当前子任务失败并 Hold",
    "progress_undo": "踏板 5 / F17 → 撤销上一个子任务进度标注",
}


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
        ui: OperatorUiPort | None,
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
        self._last_foot_pad_online = self._foot_pad_online()
        self.enabled = bool(terminal.enabled or ui is not None)
        self.should_quit = False
        self._init_qpos_emergency_stop_requested = False
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        self._tracker_side_check: dict[str, object] = {
            "status": "unavailable",
            "warning_required": False,
            "operator_lateral_axis": "+Y=left",
            "basis": "vive_virtual_desk_before_post_yaw",
            "message": "等待左右 wrist tracker 和 VIVE desk 坐标系。",
            "confirmation_required": False,
            "confirmation_token": "",
            "session_check_completed": False,
            "monitoring_active": True,
        }
        # The left/right spatial check is a startup admission guard, not a
        # runtime safety monitor.  Once the first Home has either passed the
        # check or received explicit operator confirmation, keep that result
        # for this GUI/controller lifetime and stop evaluating tracker sides.
        self._tracker_side_initial_check_completed = False
        self._tracker_side_confirmation_pending = False
        self._tracker_side_confirmation_token_counter = 0
        self._tracker_side_confirmation_from_hold = False
        self._tracker_side_confirmation_request_kind = "normal"
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._hold_requested = False
        self._hold_active = False
        self._fix_hold_active = False
        self._two_pedal_transition_active = False
        self._two_pedal_transition_target_fixed_side: str | None = None
        self._two_pedal_post_home_fixed_side: str | None = None
        self._two_pedal_fast_home_requested = False
        self._two_pedal_fast_home_deadline = 0.0
        self._two_pedal_fast_home_timeout_sec = 1.0
        self._two_pedal_alert_level = ""
        self._two_pedal_alert_message = ""
        self._two_pedal_alert_expires_at = 0.0
        self._gripper_control_before_fix_hold = False
        self._tracker_swap_locked = False
        self._teleop_started = False
        # ``_teleop_started`` means the Session has been armed and therefore
        # owns the ordinary Hold/re-Home lifecycle.  It does not prove that an
        # arm command crossed the final Session dispatch boundary.  Keep that
        # fact separate so the GUI, collection admission, and gripper control
        # cannot report RUNNING while the Session is still ARMED or frozen.
        self._teleop_dispatch_confirmed = False
        self._teleop_dispatch_projection = "idle"
        # Exact-qpos maintenance before runtime Home is enabled only while the
        # real startup wait loop owns and services its dedicated dispatch path.
        # Keeping this explicit prevents a merely enabled browser control from
        # accepting motion when no startup executor is running.
        self._startup_qpos_maintenance_enabled = False
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
        # The dual-arm Init operation is deliberately only an orchestration
        # layer over the existing single-arm exact-qpos lifecycle.  It never
        # allows both arms to own an Init operation at the same time.
        self._init_qpos_sequence_active = False
        self._init_qpos_sequence_side = ""
        self._init_qpos_sequence_phase = "idle"
        self._init_qpos_sequence_restore_side = "left"
        self._init_qpos_sequence_status = ""
        self._init_arm_qpos_speed_scale_by_side = {
            side: _init_arm_qpos_speed_scale_for_side(args, side)
            for side in ("left", "right")
        }
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
        self._dataset_reconfigure_callback: Callable[..., object] | None = None
        self._dataset_reconfigure_transactional = False
        self._dataset_selection_pending = False
        self._progress_operator_intents: deque[ProgressMutationCommand] = deque()
        self._progress_profile_selection_intents: deque[
            ProgressProfileSelectionCommand
        ] = deque()
        self._progress_entry_step_selection_intents: deque[
            ProgressEntryStepSelectionCommand
        ] = deque()
        self._progress_arm_control_mode_selection_intents: deque[
            ProgressArmControlModeSelectionCommand
        ] = deque()
        # Force-control warmup may need to sample the browser queue only for a
        # quit request. Preserve every other event so profile selection and
        # the common GUI/pedal contract are observed by the normal poll path.
        self._deferred_ui_events: deque[str] = deque()
        self._hitl_source_requests: deque[str] = deque()
        self._progress_operator_intent_sequence = 0
        self._progress_ui_transport_revision = 0
        self._progress_ui_payload: dict[str, object] = {"enabled": False}
        self._progress_ui_controls: tuple[str, ...] = ()
        # Physical progress pedals are an application input, not a rendered
        # GUI detail.  Keep their canonical admission state separate from the
        # optional progress panel so a CLI-selected profile remains operable
        # when ``--collection-progress-gui`` is disabled.
        self._progress_runtime_controls: tuple[str, ...] = ()
        self._progress_runtime_state: dict[str, object] = {
            "configured": False,
            "recording_active": False,
            "phase": "IDLE",
            "application_held": False,
            "allowed_next_step_ids": (),
            "blocked_reason": "progress_not_configured",
        }
        self._episode_delete_requested: int | None = None
        self._episode_last_episode_index: int | None = None
        self._episode_last_frame_count: int | None = None
        self._episode_recording_ui_active = False
        self._episode_recording_ui_paused = False
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = False
        self.plus_curation = EpisodeCurationSession.from_args(
            args,
            source_binding=str(
                getattr(args, "plus_source_binding", "v23_compat_content_locator_v1")
            ),
            source_schema="dexteleop_dataset_v2.3",
        )
        self._plus_curation_runtime = RealTeleopPlusCurationRuntime(
            self.plus_curation
        )
        self._foot_pad_test_active = False
        self._foot_pad_test_completed = False
        self._foot_pad_test_seen_events: set[str] = set()
        self._foot_pad_test_last_event = ""
        self._foot_pad_test_status = (
            "测试模式关闭；映射测试是可选诊断，不会阻塞踏板或 A/B/C 控制。"
            if foot_pad_input is not None
            else "脚踏测试不可用：当前未启用 foot-pad 输入。"
        )
        if foot_pad_input is not None and self._foot_pad_count() == 2:
            self._foot_pad_test_status = (
                "已连接 2 个踏板：踏板 1 进入仅左臂遥操，"
                "踏板 2 进入仅右臂遥操；双踏板不执行 Init。"
            )
        elif foot_pad_input is not None and self._foot_pad_count() == 5:
            self._foot_pad_test_status = (
                "已连接 5 个踏板：踏板 1/2 与双踏板一致，分别进入仅左/右臂遥操；"
                "踏板 3/4/5 在 progress profile 就绪后分别推进、失败、撤销。"
            )
            self._restore_persisted_five_pedal_test_state()
        self._manual_gripper_force_by_side: dict[str, float] = {}
        self._camera_visualization_toggle_requested = False
        self._tactile_visualization_toggle_requested = False
        self._tactile_solving_enable_requested = False
        self._device_retry_requested = False
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

    def bind_application_ports(
        self,
        *,
        progress_application: ProgressApplicationPort,
        input_batch_prioritizer: RealTeleopInputBatchPrioritizer,
    ) -> None:
        """Bind Station policy before the runtime begins polling inputs."""

        if not callable(input_batch_prioritizer):
            raise TypeError("input_batch_prioritizer must be callable.")
        existing_application = getattr(self, "_progress_application", None)
        existing_prioritizer = getattr(self, "_input_batch_prioritizer", None)
        if (
            existing_application is not None
            and existing_application is not progress_application
        ) or (
            existing_prioritizer is not None
            and existing_prioritizer is not input_batch_prioritizer
        ):
            raise RuntimeError("RealTeleop application ports are already bound.")
        self._progress_application = progress_application
        self._input_batch_prioritizer = input_batch_prioritizer

    def _gui_control_state(self) -> GuiControlState:
        init_controller = getattr(self.components, "init_arm_controller", None)
        arm_control_mode = str(getattr(self.args, "arm_control_mode", "position"))
        impedance_exact_qpos_available = bool(
            arm_control_mode == "impedance"
            and all(
                callable(getattr(backend, "begin_exact_qpos_transit", None))
                for backend in (
                    getattr(self.components, "left_command_backend", None),
                    getattr(self.components, "right_command_backend", None),
                )
            )
        )
        plus_curation = getattr(self, "plus_curation", None)
        plus_binding_blocker = (
            plus_curation.persisted_binding_lock_reason()
            if plus_curation is not None
            else ""
        )
        return GuiControlState(
            teleop_started=bool(getattr(self, "_teleop_started", False)),
            teleop_dispatch_confirmed=self._teleop_arm_dispatch_confirmed(),
            hold_active=bool(getattr(self, "_hold_active", False)),
            episode_recording_available=(
                not bool(getattr(self.args, "no_collection", False))
                or not bool(getattr(self.args, "no_episode_recording", False))
            ),
            episode_delete_available=not bool(
                getattr(self.args, "no_collection", False)
            ) and not bool(plus_binding_blocker),
            dataset_reconfiguration_available=not bool(plus_binding_blocker),
            episode_recording_active=bool(
                getattr(self, "_episode_recording_ui_active", False)
            ),
            recording_paused=bool(
                getattr(self, "_episode_recording_ui_paused", False)
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
            tactile_solving_available=bool(
                getattr(self.args, "tactile_solving_available", False)
            ),
            tactile_solving_active=bool(
                getattr(self.args, "tactile_solving_active", False)
            ),
            fix_arm_available=getattr(self.components, "fix_arm_controller", None) is not None,
            init_arm_available=init_controller is not None,
            init_arm_qpos_active=bool(
                getattr(init_controller, "qpos_active", False)
            ),
            fix_hold_active=bool(getattr(self, "_fix_hold_active", False)),
            init_arm_qpos_move_available=bool(
                init_controller is not None
                and (
                    arm_control_mode == "position"
                    or impedance_exact_qpos_available
                )
                and bool(getattr(self.components, "collision_guard_requested", False))
                and getattr(self.components, "dual_arm_retargeter", None) is not None
            ),
            startup_qpos_maintenance_enabled=bool(
                getattr(self, "_startup_qpos_maintenance_enabled", False)
            ),
            pose_snapshot_available=(
                getattr(self.components, "fix_arm_controller", None) is not None
                and getattr(self, "_pose_snapshot_recorder", None) is not None
            ),
            active_side_count=len(
                tuple(getattr(self.components, "active_sides", ()) or ())
            ),
            special_arm_control_mode=str(
                SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
                if self._foot_pad_mode_enabled()
                else SPECIAL_ARM_CONTROL_MODE_MANUAL
            ),
            foot_pad_available=self._foot_pad_online(),
            foot_pad_count=int(
                getattr(getattr(self, "foot_pad_input", None), "foot_pad_count", 0) or 0
            ),
            foot_pad_test_active=bool(getattr(self, "_foot_pad_test_active", False)),
            foot_pad_test_completed=bool(
                getattr(self, "_foot_pad_test_completed", False)
            ),
        )

    def _foot_pad_test_ui_state(self) -> dict[str, object]:
        online = self._foot_pad_online()
        count = int(
            getattr(getattr(self, "foot_pad_input", None), "foot_pad_count", 0) or 0
        )
        args = getattr(self, "args", None)
        requested = bool(getattr(args, "foot_pad_requested", False))
        if online and count in {2, 4, 5}:
            status = "online"
            status_label = f"{count} 个已连接"
            status_message = f"已连接并识别 {count} 个踏板。"
        elif requested:
            status = "unavailable"
            status_label = "未连接 / 未就绪"
            startup_error = str(
                getattr(args, "foot_pad_startup_error", "") or ""
            )
            status_message = startup_error or "未找到可用的 2/4/5 踏板虚拟键盘。"
            if getattr(self, "foot_pad_input", None) is not None:
                status_message = (
                    f"{status_message} 启动前可直接点击“重连失败设备”；"
                    "遥操运行中请先进入 Hold，再点击该按钮重新扫描。"
                )
        else:
            status = "disabled"
            status_label = "未启用"
            status_message = "当前配置未请求 foot-pad 控制。"
        return {
            "foot_pad_requested": requested,
            "foot_pad_available": online,
            "foot_pad_count": count,
            "foot_pad_status": status,
            "foot_pad_status_label": status_label,
            "foot_pad_status_message": status_message,
            "foot_pad_test_active": bool(getattr(self, "_foot_pad_test_active", False)),
            "foot_pad_test_completed": bool(
                getattr(self, "_foot_pad_test_completed", False)
            ),
            "foot_pad_test_seen_count": len(
                getattr(self, "_foot_pad_test_seen_events", set())
            ),
            "foot_pad_test_last_event": getattr(self, "_foot_pad_test_last_event", ""),
            "foot_pad_test_status": getattr(self, "_foot_pad_test_status", ""),
        }

    def _foot_pad_online(self) -> bool:
        source = getattr(self, "foot_pad_input", None)
        if source is None:
            return False
        configured_count = getattr(source, "foot_pad_count", None)
        if configured_count is not None:
            try:
                if int(configured_count) not in {2, 4, 5}:
                    return False
            except (TypeError, ValueError):
                return False
        online = getattr(source, "is_online", None)
        # Keep lightweight fakes and compatibility adapters usable in tests;
        # the production adapter exposes the explicit online property.
        return bool(online if online is not None else True)

    def _foot_pad_mode_enabled(self) -> bool:
        mode = str(
            getattr(self.args, "special_arm_control_mode", SPECIAL_ARM_CONTROL_MODE_MANUAL)
            or SPECIAL_ARM_CONTROL_MODE_MANUAL
        )
        return (
            mode == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            and self._foot_pad_online()
            and self._foot_pad_count() in {2, 4, 5}
        )

    def _foot_pad_count(self) -> int:
        return int(getattr(getattr(self, "foot_pad_input", None), "foot_pad_count", 0) or 0)

    def _foot_pad_test_required_count(self) -> int:
        """Return the number of valid pedal edges needed by the test.

        A two-pedal virtual keyboard only has the two Fix mappings.  Keep the
        four-pedal diagnostic contract unchanged, while making the two-pedal
        test finish as soon as both available pedals have been verified.
        """

        count = self._foot_pad_count()
        return 2 if count == 2 else 5 if count == 5 else 4

    def _restore_persisted_five_pedal_test_state(self) -> bool:
        """Restore mapping verification shared by position/impedance runs."""

        verified = bool(
            self._foot_pad_online()
            and self._foot_pad_count() == 5
            and persisted_five_pedal_test_verified()
        )
        self._foot_pad_test_completed = verified
        self._foot_pad_test_seen_events = set()
        self._foot_pad_test_last_event = ""
        if verified:
            self._foot_pad_test_status = (
                "当前五踏板物理映射已在此前完整测试并持久化；"
                "力控/位控共用该结果，无需再次测试。"
            )
        return verified

    def _two_pedal_mode_active(self) -> bool:
        """Return whether the 2/5-pedal side-switch contract is active."""

        return bool(
            self._foot_pad_mode_enabled()
            and self._foot_pad_online()
            and self._foot_pad_count() in FOOT_PAD_TWO_PEDAL_STYLE_COUNTS
        )

    def _strict_profile_arm_control_active(self) -> bool:
        payload = getattr(self, "_progress_ui_payload", {})
        return bool(
            isinstance(payload, Mapping)
            and payload.get("configured", False)
            and str(payload.get("arm_control_mode", ""))
            == PROGRESS_ARM_CONTROL_MODE_STRICT
        )

    def _manual_single_arm_switch_enabled(self) -> bool:
        return not self._strict_profile_arm_control_active()

    def _publish_two_pedal_alert(
        self,
        *,
        level: str,
        message: str,
        now: float,
        duration_sec: float | None = None,
    ) -> None:
        normalized = str(level or "").strip().lower()
        self._two_pedal_alert_level = normalized
        self._two_pedal_alert_message = str(message or "")
        self._two_pedal_alert_expires_at = (
            0.0
            if duration_sec is None
            else float(now) + max(0.0, float(duration_sec))
        )
        if self.ui is not None:
            self.ui.publish(
                two_pedal_alert={
                    "active": bool(normalized and message),
                    "level": normalized,
                    "message": str(message or ""),
                },
                control_hint=str(message or ""),
            )

    def _clear_two_pedal_alert(self) -> None:
        self._two_pedal_alert_level = ""
        self._two_pedal_alert_message = ""
        self._two_pedal_alert_expires_at = 0.0
        if self.ui is not None:
            self.ui.publish(
                two_pedal_alert={},
                control_hint=self._control_hint(),
            )

    def _poll_two_pedal_alert_timeout(self, *, now: float) -> None:
        if (
            float(getattr(self, "_two_pedal_alert_expires_at", 0.0)) > 0.0
            and float(now)
            >= float(getattr(self, "_two_pedal_alert_expires_at", 0.0))
        ):
            self._clear_two_pedal_alert()

    def _manual_fix_arm_controls_enabled(self) -> bool:
        """Foot pads supplement rather than replace keyboard/GUI Fix controls."""

        return self._manual_single_arm_switch_enabled()

    def _manual_init_arm_controls_enabled(self) -> bool:
        """Foot pads supplement rather than replace keyboard/GUI Init controls."""

        return True

    def _gui_init_arm_controls_enabled(self) -> bool:
        """The foot-pad topology never removes explicit GUI Init controls."""

        return True

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

    def plus_curation_ui_state(self) -> dict[str, object]:
        return self.plus_curation.ui_state()

    def begin_plus_curation_episode(self, episode_index: int) -> None:
        if not self.plus_curation.enabled:
            return
        self.plus_curation.begin_episode(
            episode_index=int(episode_index),
            variant_id=self.plus_curation.variant_id or None,
        )
        if self.ui is not None:
            self.ui.publish(**self.plus_curation.ui_state())

    def set_plus_curation_pending(self, *, dataset_root: str | Path, episode_index: int) -> None:
        self.plus_curation.mark_pending(dataset_root=dataset_root, episode_index=int(episode_index))
        if self.ui is not None:
            self.ui.publish(
                **self.plus_curation.ui_state(),
                active_controls=self._current_active_controls(),
            )

    def clear_plus_curation_pending(self, *, message: str = "") -> None:
        self.plus_curation.clear_pending(message=message)
        if self.ui is not None:
            self.ui.publish(
                **self.plus_curation.ui_state(),
                active_controls=self._current_active_controls(),
            )

    def consume_plus_curation_request(self) -> dict[str, object] | None:
        runtime = getattr(self, "_plus_curation_runtime", None)
        if runtime is None:
            return None
        return runtime.consume_request()

    def _plus_curation_save_in_flight(self) -> bool:
        runtime = getattr(self, "_plus_curation_runtime", None)
        if runtime is not None:
            return runtime.save_in_flight()
        plus_curation = getattr(self, "plus_curation", None)
        return bool(
            plus_curation is not None
            and plus_curation.enabled
            and plus_curation.status == "saving"
        )

    def plus_curation_save_in_flight(self) -> bool:
        return self._plus_curation_save_in_flight()

    def save_plus_curation(
        self,
        *,
        episode_index: int,
        pending_token: str,
        disposition: str,
        reason_code: str,
    ) -> None:
        runtime = getattr(self, "_plus_curation_runtime", None)
        if runtime is None:
            runtime = RealTeleopPlusCurationRuntime(self.plus_curation)
            self._plus_curation_runtime = runtime

        def publish_state() -> None:
            if self.ui is not None:
                self.ui.publish(
                    **self.plus_curation.ui_state(),
                    dataset_configuration_blocked_reason=(
                        self._dataset_configuration_blocked_reason()
                    ),
                    active_controls=self._current_active_controls(),
                )

        runtime.start_save(
            episode_index=episode_index,
            pending_token=pending_token,
            disposition=disposition,
            reason_code=reason_code,
            on_state_changed=publish_state,
            on_failure=lambda message: (
                self.ui.append_message("Plus", message)
                if self.ui is not None
                else None
            ),
            on_success=lambda record: (
                self.ui.append_message(
                    "Plus",
                    f"已写入 Plus sidecar: {record.annotation_episode_id}",
                )
                if self.ui is not None
                else None
            ),
        )

    def _camera_visualization_available(self) -> bool:
        return self._gui_control_state().camera_visualization_available

    def _tactile_visualization_available(self) -> bool:
        return self._gui_control_state().tactile_visualization_available

    def _visualization_controls(self) -> list[str]:
        return _gui_visualization_controls(self._gui_control_state())

    def _dataset_configuration_blocked_reason(self) -> str:
        return self.plus_curation.persisted_binding_lock_reason()

    def _episode_busy(self) -> bool:
        return _gui_episode_busy(self._gui_control_state())

    def _teleop_arm_dispatch_confirmed(self) -> bool:
        """Return the production dispatch fact with thin-adapter compatibility."""

        session = getattr(self.components, "session", None)
        if getattr(session, "state_machine", None) is None:
            # Historic offline/test adapters expose no Session state at all.
            # They cannot project the authoritative dispatch boundary, so
            # preserve their established armed-only contract.  The real
            # runtime always has a state_machine and never takes this branch.
            return bool(getattr(self, "_teleop_started", False))
        return bool(getattr(self, "_teleop_dispatch_confirmed", False))

    def _episode_transition_busy(self) -> bool:
        state = self._gui_control_state()
        return bool(state.episode_countdown_active or state.episode_finalizing_active)

    def episode_start_admission(self) -> tuple[bool, str]:
        """Apply one start gate to GUI, browser shortcuts, and terminal input.

        The progress GUI is an optional annotation/control layer.  Showing it
        must not turn an ordinary profile-less Episode into an admission
        failure; a selected profile is consumed only when the operator has
        explicitly enabled the progress workflow for that Episode.
        """
        if bool(getattr(self, "_dataset_selection_pending", False)):
            return False, "数据集任务/profile 正在后台准备，完成后才能开始数采。"
        if not bool(getattr(self, "_teleop_started", False)):
            return False, "遥操尚未进入 RUNNING，不能开始数采。"
        if not self._teleop_arm_dispatch_confirmed():
            return False, "机械臂命令尚未实际派发，不能开始数采。"
        if bool(getattr(self, "_hold_active", False)):
            return False, "常规 Hold 期间不能开始数采；请先重采 runtime Home。"
        init_controller = getattr(self.components, "init_arm_controller", None)
        fix_controller = getattr(self.components, "fix_arm_controller", None)
        fixed_like_sides = set(
            tuple(getattr(fix_controller, "fixed_sides", ()) or ())
        )
        fixed_like_sides.update(
            tuple(getattr(init_controller, "fix_hold_sides", ()) or ())
        )
        if self._fix_hold_active or fixed_like_sides == {"left", "right"}:
            return (
                False,
                "Fix-Hold 期间不能新开始数采；请先完成任一侧 Home 重采以解除特殊固定。",
            )
        if bool(getattr(init_controller, "qpos_active", False)):
            return (
                False,
                "精确 Init qpos 规划、执行或保持期间不能新开始数采；"
                "请先重采目标侧 Home，或进入常规 Hold。",
            )
        if (
            bool(getattr(init_controller, "moving", False))
            or bool(getattr(init_controller, "recapturing", False))
            or str(getattr(init_controller, "mode", "")) == "qpos_failed"
        ):
            return (
                False,
                "Init 移动、Home 重采或失败恢复期间不能新开始数采。",
            )
        return True, ""

    def _publish_episode_start_rejection(self, message: str) -> None:
        if self.ui is None:
            return
        self.ui.append_message("Dataset", message)
        self.ui.publish(
            active_controls=self._current_active_controls(),
            control_hint=message,
        )

    def _fix_init_allowed_in_controls(self) -> bool:
        return _gui_fix_init_allowed_in_controls(self._gui_control_state())

    def _current_active_controls(self) -> list[str]:
        gui_state = self._gui_control_state()
        controls = list(_gui_current_active_controls(
            gui_state, self._gui_events()
        ))
        if not self._manual_single_arm_switch_enabled():
            controls = [
                control
                for control in controls
                if control
                not in {"c", "fix_arm_toggle", "fix_arm_select", "two_pedal_b"}
            ]
        controls.extend(getattr(self, "_progress_ui_controls", ()))
        plus_curation = getattr(self, "plus_curation", None)
        plus_save_in_flight = self._plus_curation_save_in_flight()
        if plus_save_in_flight:
            controls = [control for control in controls if control != "episode_delete"]
        if (
            plus_curation is not None
            and plus_curation.enabled
            and plus_curation.pending_episode_index is not None
            and plus_curation.status not in {"saving"}
        ):
            controls.append("plus_curation_save")
        if self._two_pedal_mode_active() and self._teleop_started:
            if (
                self._manual_single_arm_switch_enabled()
                and not self._two_pedal_switch_blocked_by_init()
                and (
                    bool(getattr(gui_state, "hold_active", False))
                    or bool(
                        getattr(gui_state, "teleop_started", False)
                        and getattr(
                            gui_state,
                            "teleop_dispatch_confirmed",
                            None,
                        )
                        is not False
                    )
                )
            ):
                controls.append("two_pedal_b")
            # C is the unconditional route to regular Hold, including while
            # an Init operation is active.
            controls.append("two_pedal_c")
        tracker_side_gate_pending = bool(
            getattr(self, "_tracker_side_confirmation_pending", False)
            or getattr(self, "_tracker_side_request_pending", False)
        )
        if tracker_side_gate_pending:
            controls = [
                control
                for control in controls
                if control not in {"p", "two_pedal_b"}
            ]
        if bool(getattr(self, "_tracker_side_confirmation_pending", False)):
            controls.extend(
                (
                    "tracker_side_confirm_continue",
                    "tracker_side_confirm_cancel",
                )
            )
        if bool(getattr(self, "_hold_requested", False)):
            controls = [
                control
                for control in controls
                if control
                not in {
                    "device_retry",
                    "calibrate_vive_coordinate_profile",
                    "load_vive_calibration_profile",
                    "movement_multiple_down",
                    "movement_multiple_up",
                    "movement_multiple_reset",
                    "movement_multiple_set",
                    "init_arm_move_to_qpos",
                    "init_arm_move_both_to_qpos",
                    "enable_tactile_solving",
                }
            ]
        return list(dict.fromkeys(controls))

    def _fix_arm_available(self) -> bool:
        return self._gui_control_state().fix_arm_available

    def _init_arm_available(self) -> bool:
        return self._gui_control_state().init_arm_available

    def _pose_snapshot_available(self) -> bool:
        return self._gui_control_state().pose_snapshot_available

    def _pose_snapshot_ui_state(self) -> dict[str, object]:
        recorder = getattr(self, "_pose_snapshot_recorder", None)
        return {
            "pose_snapshot_available": self._pose_snapshot_available(),
            "pose_snapshot_dir": "" if recorder is None else str(recorder.root_dir),
            "pose_snapshot_count": 0 if recorder is None else int(recorder.count),
            "pose_snapshot_last_path": getattr(self, "_pose_snapshot_last_path", ""),
            "pose_snapshot_status": getattr(self, "_pose_snapshot_status", ""),
        }

    def _single_arm_tracker_swap_allowed_in_hold(self) -> bool:
        return _gui_single_arm_tracker_swap_allowed_in_hold(self._gui_control_state())

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
            active_controls=self._current_active_controls(),
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
            dataset_configuration_blocked_reason=(
                self._dataset_configuration_blocked_reason()
            ),
            **self.plus_curation.ui_state(),
            **_collection_task_ui_state(self.args),
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            tactile_solving_available=bool(
                getattr(self.args, "tactile_solving_available", False)
            ),
            tactile_solving_active=bool(
                getattr(self.args, "tactile_solving_active", False)
            ),
            visualization_status="",
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
            capture_waiting_for_samples=False,
            special_arm_control_mode=self._gui_control_state().special_arm_control_mode,
            tracker_side_check=dict(self._tracker_side_check),
            **self._foot_pad_test_ui_state(),
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
        init_controller = getattr(self.components, "init_arm_controller", None)
        if init_controller is not None and callable(getattr(init_controller, "close", None)):
            steps.append(("init_arm_controller", init_controller.close))
        if self.ui is not None:
            steps.append(("web_ui", self.ui.close))
        steps.append(("terminal", self.terminal.close))
        if self.foot_pad_input is not None:
            steps.append(("foot_pad", self.foot_pad_input.close))
        run_best_effort_cleanup("real_teleop_control_input.close", steps)

    def defer_ui_events(self, events: object) -> None:
        """Return non-quit UI events to the next normal input poll.

        The impedance force warmup is a temporary startup worker and cannot
        execute application actions. It may still need to observe ``q`` to
        abort. All other events must remain ordered for the shared position/
        impedance coordinator instead of being silently consumed.
        """

        pending = getattr(self, "_deferred_ui_events", None)
        if pending is None:
            pending = deque()
            self._deferred_ui_events = pending
        if isinstance(events, str):
            events = (events,)
        try:
            for event in events:  # type: ignore[union-attr]
                normalized = str(event)
                if normalized != "q":
                    pending.append(normalized)
        except TypeError:
            return

    def _progress_application_port(self) -> ProgressApplicationPort:
        application = getattr(self, "_progress_application", None)
        if application is None:
            factory = globals().get("build_real_teleop_progress_application")
            if not callable(factory):
                raise RuntimeError(
                    "RealTeleop progress application was not injected."
                )
            application = factory(getattr(self, "args", None))
            self._progress_application = application
        return application

    def _input_batch_policy(self) -> RealTeleopInputBatchPrioritizer:
        policy = getattr(self, "_input_batch_prioritizer", None)
        if policy is None:
            policy = globals().get("prioritize_input_batch")
            if not callable(policy):
                raise RuntimeError(
                    "RealTeleop input-batch prioritizer was not injected."
                )
            self._input_batch_prioritizer = policy
        return policy

    def poll(self) -> None:
        current_time = time.time()
        self._poll_two_pedal_alert_timeout(now=current_time)
        self._poll_tracker_side_request_timeout(now=current_time)
        self.terminal.poll()
        consume_hitl_requests = getattr(
            self.terminal,
            "consume_hitl_source_requests",
            None,
        )
        if callable(consume_hitl_requests) and not self.terminal.should_quit:
            for requested_source in consume_hitl_requests():
                source = str(requested_source).strip().lower()
                if source in {"human", "policy"}:
                    self._hitl_source_requests.append(source)
        foot_pad_state_published = False
        deferred_ui_events = list(getattr(self, "_deferred_ui_events", ()))
        if deferred_ui_events:
            getattr(self, "_deferred_ui_events").clear()
        ui_events = deferred_ui_events + (
            list(self.ui.consume_events()) if self.ui is not None else []
        )
        # Station owns application-level input precedence. Device polling and
        # all resulting entity operations remain in this lower runtime.
        input_batch = self._input_batch_policy()(
            ui_events,
            terminal_quit_requested=bool(self.terminal.should_quit),
            teleop_started=bool(self._teleop_started),
            # Preserve the legacy lazy query: partially assembled startup/test
            # adapters need no mode state unless this exact event was sampled.
            two_pedal_mode_active=bool(
                "two_pedal_c" in ui_events and self._two_pedal_mode_active()
            ),
        )
        terminal_quit_requested = input_batch.terminal_quit_requested
        suppress_foot_pad_events = input_batch.suppress_foot_pad_events
        ui_events = list(input_batch.events)
        if terminal_quit_requested:
            self.should_quit = True
        foot_pad_test_batch = bool(getattr(self, "_foot_pad_test_active", False)) or any(
            event in {"foot_pad_test_start", "foot_pad_test_stop"}
            for event in ui_events
        )
        if self.foot_pad_input is not None:
            consume_foot_pad_events = (
                getattr(self.foot_pad_input, "consume_test_events", None)
                if foot_pad_test_batch
                else None
            )
            if not callable(consume_foot_pad_events):
                consume_foot_pad_events = self.foot_pad_input.consume_events
            for event in consume_foot_pad_events():
                publish_input_activity = getattr(
                    self.ui,
                    "publish_input_activity",
                    None,
                )
                if callable(publish_input_activity):
                    publish_input_activity(
                        source="foot_pad",
                        event=str(event),
                        pedal_count=self._foot_pad_count(),
                    )
                if suppress_foot_pad_events:
                    # Drain the non-blocking device queue so a press that
                    # coincided with the barrier cannot fire on the next loop.
                    continue
                if foot_pad_test_batch:
                    self._record_foot_pad_test_event(event, now=current_time)
                else:
                    self._handle_foot_pad_special_arm_event(event, now=current_time)
            if bool(getattr(self, "_foot_pad_test_active", False)) and not self._foot_pad_online():
                self._foot_pad_test_active = False
                self._foot_pad_test_completed = False
                self._foot_pad_test_seen_events = set()
                self._foot_pad_test_status = (
                    "脚踏设备已断开，测试模式已安全关闭；"
                    "运行中请先进入 Hold，再点击“重连失败设备”；成功连接后可重新测试。"
                )
                self._publish_foot_pad_test_state(
                    message=(
                        "脚踏设备已断开；仅踏板快捷输入暂停，GUI/键盘 Fix/Init "
                        "保持可用。运行中请先进入 Hold，再点击“重连失败设备”重新扫描。"
                    )
                )
                foot_pad_state_published = True
            online_now = self._foot_pad_online()
            previous_online = getattr(self, "_last_foot_pad_online", None)
            if previous_online is None:
                # A partially constructed/test adapter has no prior edge to
                # publish.  Establish the baseline without inventing a
                # reconnect transition.
                self._last_foot_pad_online = online_now
            elif online_now != bool(previous_online) and not foot_pad_state_published:
                self._last_foot_pad_online = online_now
                restored_test = bool(
                    online_now
                    and self._foot_pad_count() == 5
                    and self._restore_persisted_five_pedal_test_state()
                )
                if not restored_test:
                    self._foot_pad_test_completed = False
                    self._foot_pad_test_seen_events = set()
                self._publish_foot_pad_test_state(
                    message=(
                        (
                            "脚踏设备已重连；五踏板映射与已保存测试结果一致，"
                            "无需重测。"
                            if restored_test
                            else "脚踏设备已重连；已恢复对应踏板 fix/init 映射。"
                        )
                        if online_now
                        else (
                            "脚踏设备已断开；仅踏板快捷输入暂停，GUI/键盘 Fix/Init "
                            "保持可用。运行中请先进入 Hold，再点击“重连失败设备”重新扫描。"
                        )
                    )
                )
            else:
                self._last_foot_pad_online = online_now
        if terminal_quit_requested:
            self.should_quit = True
        if not self.should_quit and self.terminal.consume_calibration_request():
            self._request_calibration_with_tracker_side_guard(source="terminal")
        if self.ui is not None:
            for event in ui_events:
                if event == "init_qpos_emergency_stop":
                    self._init_qpos_emergency_stop_requested = True
                    self._cancel_init_arm_qpos_sequence(reason="GUI 急停")
                    message = "已触发 Init 急停请求；本次归位将立即中止并保持机械臂静止，不退出遥操程序。"
                    _log("InitArmQpos", "GUI init_qpos_emergency_stop event received", now=current_time)
                    self.ui.publish(
                        stage="Init 急停处理中",
                        active_controls=[],
                        control_hint=message,
                    )
                    self.ui.append_message("InitArmQpos", message)
                elif event == "q":
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
                elif event == "tracker_side_confirm_continue":
                    self._confirm_tracker_side_warning()
                elif event == "tracker_side_confirm_cancel":
                    self._cancel_tracker_side_warning()
                elif event == "p":
                    self._request_calibration_with_tracker_side_guard(source="gui")
                elif event == "two_pedal_b":
                    self._handle_two_pedal_b_event(now=current_time)
                elif event == "two_pedal_c":
                    if self._two_pedal_mode_active():
                        self.request_hold(reason="two_pedal_c")
                elif event == "foot_pad_test_start":
                    self._start_foot_pad_test()
                elif event == "foot_pad_test_stop":
                    self._stop_foot_pad_test()
                elif event in {"c", "fix_arm_toggle"}:
                    if (
                        self._manual_fix_arm_controls_enabled()
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
                        self._manual_init_arm_controls_enabled()
                        and self._init_arm_available()
                    ):
                        self._handle_init_arm_g_event(now=current_time)
                elif event == "init_arm_move_to_qpos":
                    if (
                        self._gui_init_arm_controls_enabled()
                        and self._init_arm_available()
                    ):
                        self._handle_init_arm_move_to_qpos_event(now=current_time)
                elif event == "init_arm_move_both_to_qpos":
                    if (
                        self._gui_init_arm_controls_enabled()
                        and self._init_arm_available()
                    ):
                        self._handle_init_arm_move_both_to_qpos_event(now=current_time)
                elif event.startswith("plus_curation_save:"):
                    self._plus_curation_runtime.submit_request_text(
                        event.split(":", 1)[1],
                        on_state_changed=lambda: self.ui.publish(
                            **self.plus_curation.ui_state(),
                            active_controls=self._current_active_controls(),
                        ),
                    )
                elif (
                    event in {"progress_failure", "progress_undo"}
                    or event == "progress_advance"
                    or event.startswith("progress_advance:")
                ):
                    if not self.submit_progress_ui_event(event):
                        self.ui.append_message(
                            "Progress",
                            f"无法解析长程进度请求：{event}",
                        )
                    else:
                        self.ui.append_message(
                            "Progress",
                            f"长程进度请求已提交：{event}",
                        )
                elif event.startswith("progress_profile_select:"):
                    if not self.submit_progress_profile_selection_ui_event(event):
                        self.ui.append_message(
                            "Progress",
                            f"无法解析或已有待处理的 profile 选择：{event}",
                        )
                    else:
                        self.ui.append_message(
                            "Progress",
                            f"长程进度 profile 选择已提交：{event}",
                        )
                elif event.startswith("progress_entry_step_select:"):
                    if not self.submit_progress_entry_step_selection_ui_event(
                        event
                    ):
                        self.ui.append_message(
                            "Progress",
                            f"无法解析或已有待处理的起始子任务选择：{event}",
                        )
                    else:
                        self.ui.append_message(
                            "Progress",
                            f"长程进度起始子任务选择已提交：{event}",
                        )
                elif event.startswith("progress_arm_control_mode_select:"):
                    if not self.submit_progress_arm_control_mode_selection_ui_event(
                        event
                    ):
                        self.ui.append_message(
                            "Progress",
                            f"无法解析或已有待处理的臂控制模式选择：{event}",
                        )
                    else:
                        self.ui.append_message(
                            "Progress",
                            f"长程进度臂控制模式选择已提交：{event}",
                        )
                elif event == "episode_start" or event.startswith("episode_start:"):
                    admitted, message = self.episode_start_admission()
                    if not admitted:
                        self._publish_episode_start_rejection(message)
                    else:
                        payload_applied = True
                        if event.startswith("episode_start:"):
                            payload_applied = self._apply_episode_start_payload(
                                event.split(":", 1)[1],
                                bind_dataset_contract=True,
                                start_after_commit=True,
                            )
                        if payload_applied and not self._dataset_selection_pending:
                            self._episode_start_requested = True
                elif event.startswith("task_select:"):
                    blocker = self.plus_curation.persisted_binding_lock_reason()
                    if blocker:
                        if self.ui is not None:
                            self.ui.append_message("Dataset", blocker)
                            self.ui.publish(
                                dataset_configuration_blocked_reason=blocker,
                                active_controls=self._current_active_controls(),
                            )
                    else:
                        self._apply_episode_start_payload(
                            event.split(":", 1)[1],
                            bind_dataset_contract=True,
                        )
                elif event.startswith("gripper_force_set:"):
                    self._set_gripper_force_from_gui(event.split(":", 1)[1])
                elif event.startswith("gripper_force_persist:"):
                    self._persist_gripper_force_from_gui(event.split(":", 1)[1])
                elif event.startswith("init_arm_qpos_speed_set:"):
                    self._set_init_arm_qpos_speed_from_gui(event.split(":", 1)[1])
                elif event == "episode_stop":
                    self._episode_stop_requested = True
                elif event.startswith("episode_delete:"):
                    try:
                        self._episode_delete_requested = int(event.split(":", 1)[1])
                    except ValueError:
                        if self.ui is not None:
                            self.ui.append_message("Dataset", f"无效删除请求: {event}")
                            self.ui.publish(
                                active_controls=self._current_active_controls()
                            )
                elif event == "round_trip_toggle":
                    self._toggle_round_trip_collection_task(source="manual")
                elif event.startswith("fix_arm_select:"):
                    if self._manual_fix_arm_controls_enabled():
                        self._set_fix_arm_side_from_gui(event.split(":", 1)[1])
                elif event.startswith("init_arm_select:"):
                    if self._gui_init_arm_controls_enabled():
                        self._set_init_arm_side_from_gui(event.split(":", 1)[1])
                elif event == POSE_SNAPSHOT_GUI_EVENT:
                    self.request_pose_snapshot()
                elif event == "toggle_camera_vis":
                    self._camera_visualization_toggle_requested = True
                elif event == "toggle_tactile_vis":
                    self._tactile_visualization_toggle_requested = True
                elif event == "enable_tactile_solving":
                    self._tactile_solving_enable_requested = True
                elif event == "device_retry":
                    admitted, message = self.device_retry_admission()
                    if admitted:
                        self._device_retry_requested = True
                    elif self.ui is not None:
                        self.ui.append_message("Device", message)
                        self.ui.publish(
                            active_controls=self._current_active_controls(),
                            control_hint=message,
                        )
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
        # Until the first Home admission, keep the latest tracker-side result
        # available to both GUI and terminal compatibility flows.  After that
        # one-time check is latched, publish_raw_tracker_pose_state keeps only
        # the raw visualization current and skips side evaluation.
        self.publish_raw_tracker_pose_state(now=current_time)
        if self.ui is not None:
            self.publish_gripper_status(now=current_time)

    def _publish_foot_pad_test_state(self, *, message: str = "") -> None:
        if self.ui is None:
            return
        self.ui.publish(
            active_controls=self._current_active_controls(),
            control_hint=(
                self._foot_pad_test_status
                if self._foot_pad_test_active
                else self._control_hint()
            ),
            **self._foot_pad_test_ui_state(),
        )
        if message:
            self.ui.append_message("FootPadTest", message)

    def _start_foot_pad_test(self) -> None:
        if not self._foot_pad_online():
            self._foot_pad_test_status = "脚踏测试不可用：当前没有在线的 foot-pad 输入。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        if self._foot_pad_test_active:
            return
        if self._episode_busy():
            self._foot_pad_test_status = "脚踏测试未开启：请先停止当前数采/倒计时/保存流程。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        if bool(getattr(self, "_hold_requested", False)):
            self._foot_pad_test_status = "脚踏测试未开启：Hold 正在生效，请稍后重试。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        if self._teleop_started and not self._hold_active:
            self._foot_pad_test_status = "脚踏测试未开启：运行中请先 Hold / 冻结。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        self._foot_pad_test_active = True
        self._foot_pad_test_completed = False
        self._foot_pad_test_seen_events = set()
        self._foot_pad_test_last_event = ""
        required_count = self._foot_pad_test_required_count()
        if required_count == 2:
            self._foot_pad_test_status = (
                "双踏板测试进行中：请依次踩踏板 1、2；检测到两个踏板后自动完成。"
                "这里只显示映射，不执行任何臂操作。"
            )
        elif required_count == 5:
            self._foot_pad_test_status = (
                "五踏板测试进行中：请依次踩踏板 1、2、3、4、5；"
                "检测到五个踏板后自动完成。这里只显示映射，不执行任何实际操作。"
            )
        else:
            self._foot_pad_test_status = (
                "四踏板测试进行中：请依次踩踏板 1、2、3、4；"
                "这里只显示映射，不执行任何臂操作。"
            )
        self._publish_foot_pad_test_state(message="已开启脚踏测试模式；踏板事件不会触发 fix/init。")

    def _stop_foot_pad_test(self) -> None:
        if not self._foot_pad_test_active:
            self._foot_pad_test_status = "脚踏测试当前未开启。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        self._foot_pad_test_active = False
        self._foot_pad_test_completed = True
        seen_count = len(getattr(self, "_foot_pad_test_seen_events", set()))
        required_count = self._foot_pad_test_required_count()
        self._foot_pad_test_status = (
            f"脚踏测试已完成（本次检测到 {seen_count}/{required_count} 个键位）；"
            "踏板控制与 A/B/C 快捷键均已恢复。"
        )
        self._publish_foot_pad_test_state(message=self._foot_pad_test_status)

    def _record_foot_pad_test_event(self, event: str, *, now: float) -> None:
        del now
        if not bool(getattr(self, "_foot_pad_test_active", False)):
            return
        required_count = self._foot_pad_test_required_count()
        if required_count == 2 and str(event) not in {"fix_arm_left", "fix_arm_right"}:
            # The diagnostic adapter may expose all four labels for
            # compatibility, but F15/F16 are not pedals in the live two-pedal
            # topology and must not satisfy its completion condition.
            return
        count = self._foot_pad_count()
        labels = (
            _TWO_PEDAL_TEST_LABELS
            if count == 2
            else _FIVE_PEDAL_TEST_LABELS
            if count == 5
            else _FOOT_PAD_TEST_LABELS
        )
        label = labels.get(str(event))
        if label is None:
            return
        seen_events = getattr(self, "_foot_pad_test_seen_events", None)
        if not isinstance(seen_events, set):
            seen_events = set()
            self._foot_pad_test_seen_events = seen_events
        seen_events.add(str(event))
        self._foot_pad_test_last_event = label
        seen_count = len(seen_events)
        if seen_count >= required_count:
            self._foot_pad_test_active = False
            self._foot_pad_test_completed = True
            tested_label = (
                "双踏板 1、2"
                if required_count == 2
                else "踏板 1–5"
                if required_count == 5
                else "踏板 1–4"
            )
            self._foot_pad_test_status = (
                f"{tested_label} 均已检测，测试自动完成；"
                "踏板控制与 A/B/C 快捷键均已恢复。"
            )
            persisted_message = ""
            if required_count == 5:
                try:
                    receipt_path = persist_five_pedal_test_receipt()
                except Exception as exc:  # noqa: BLE001 - test remains valid for this run.
                    persisted_message = (
                        " 五踏板测试结果未能持久化，下次启动仍会要求重测："
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._foot_pad_test_status += persisted_message
                else:
                    persisted_message = (
                        f" 五踏板测试结果已保存到 {receipt_path}；"
                        "后续力控/位控运行共用。"
                    )
                    self._foot_pad_test_status += persisted_message
            self._publish_foot_pad_test_state(
                message=(
                    f"检测到 {label}；{required_count}/{required_count} 键位齐全，"
                    f"测试已自动完成。{persisted_message}"
                )
            )
            return
        self._foot_pad_test_status = (
            f"最近检测：{label}（已检测 {seen_count}/{required_count}）。"
            "测试模式仍开启；可继续测试，或点击“停止测试踏板”确认完成。"
        )
        self._publish_foot_pad_test_state(
            message=f"检测到 {label}；未执行实际操作。"
        )

    def _progress_pedal_routing_enabled(self) -> bool:
        """Whether progress pedals are reserved for the configured progress run."""

        return bool(
            self._foot_pad_mode_enabled()
            and self._foot_pad_count() in {4, 5}
            and getattr(
                getattr(self, "args", None),
                "collection_progress_profile",
                None,
            )
            is not None
        )

    def _publish_progress_pedal_message(
        self,
        message: str,
        *,
        now: float,
    ) -> None:
        logger = globals().get("_log")
        if callable(logger):
            logger("Progress", message, now=now)
        ui = getattr(self, "ui", None)
        if ui is not None:
            ui.append_message("Progress", message)

    def _handle_progress_pedal_event(
        self,
        event: str,
        *,
        now: float,
    ) -> bool:
        """Reuse the admitted GUI progress intents for pedals 3/4/5.

        The physical adapter intentionally keeps its legacy F15/F16 labels so
        the existing four-key diagnostic test is unchanged.  Once a progress
        profile is configured, however, these two runtime edges are reserved:
        pedal 3 advances and pedal 4 reports Failure.  The five-pedal topology
        adds F17 for Undo without changing the four-pedal topology.  An
        unavailable progress control is ignored in-place and never falls
        through to Init or remains queued for a later Episode.
        """

        if not self._progress_pedal_routing_enabled():
            return False
        event_bindings = self._progress_application_port().event_bindings
        progress_event_by_legacy_label = {
            "init_arm_left": event_bindings.advance,
            "init_arm_right": event_bindings.failure,
            "progress_undo": event_bindings.undo,
        }
        progress_event = progress_event_by_legacy_label.get(str(event))
        if progress_event is None:
            return False
        pedal_number_by_event = {
            event_bindings.advance: 3,
            event_bindings.failure: 4,
            event_bindings.undo: 5,
        }
        operation_label_by_event = {
            event_bindings.advance: "进入下一子任务阶段",
            event_bindings.failure: "记录当前子任务失败并 Hold",
            event_bindings.undo: "撤销上一个子任务进度标注",
        }
        pedal_number = pedal_number_by_event[progress_event]
        operation_label = operation_label_by_event[progress_event]
        active_controls = set(
            getattr(
                self,
                "_progress_runtime_controls",
                # Compatibility for small tests/embedders constructed before
                # runtime/UI admission was split.  Normal initialized objects
                # always own ``_progress_runtime_controls``.
                getattr(self, "_progress_ui_controls", ()),
            )
        )
        if progress_event not in active_controls:
            runtime_state = dict(
                getattr(self, "_progress_runtime_state", {}) or {}
            )
            blocked_reason = str(
                runtime_state.get("blocked_reason", "") or ""
            ).strip()
            phase = str(runtime_state.get("phase", "") or "IDLE")
            recording_active = bool(
                runtime_state.get("recording_active", False)
            )
            if getattr(self, "_progress_operator_intents", ()):
                reason = "已有 progress 请求等待控制循环处理"
            elif blocked_reason == "arm_transition_home_pending":
                reason = "上一条阶段切换正在等待目标活动臂 runtime Home"
            elif not bool(runtime_state.get("configured", False)):
                reason = "当前 Episode 未配置 progress profile"
            elif not recording_active or phase == "IDLE":
                reason = "当前没有活动的正式 Episode"
            elif bool(runtime_state.get("application_held", False)):
                reason = f"当前处于 Hold（phase={phase}）"
            elif phase != "ACTIVE_ATTEMPT":
                reason = f"当前 progress phase={phase}"
            elif not tuple(
                runtime_state.get("allowed_next_step_ids", ()) or ()
            ):
                reason = "当前已经是 profile 的末阶段，没有合法后继"
            elif blocked_reason:
                reason = blocked_reason
            else:
                reason = "progress 准入状态尚未刷新"
            message = (
                f"踏板 {pedal_number} 已忽略：当前 progress 状态不允许"
                f"“{operation_label}”（{reason}）；未执行 Init。"
            )
            self._publish_progress_pedal_message(message, now=now)
            return True
        if not self.submit_progress_foot_pad_event(progress_event):
            message = (
                f"踏板 {pedal_number} 请求未接纳：已有 progress 请求等待处理；"
                "未执行 Init。"
            )
            self._publish_progress_pedal_message(message, now=now)
            return True
        message = f"踏板 {pedal_number} 已提交：{operation_label}。"
        self._publish_progress_pedal_message(message, now=now)
        return True

    def _handle_foot_pad_special_arm_event(self, event: str, *, now: float) -> None:
        """Route one physical pedal to the existing GUI special-arm transition."""

        if not self._foot_pad_mode_enabled():
            return
        if (
            self._foot_pad_count() == 5
            and str(event) in FOOT_PAD_FIVE_PROGRESS_EVENTS
            and not self._progress_pedal_routing_enabled()
        ):
            # The five-pedal Fix prefix remains usable while the operator is
            # choosing a profile. Progress-labelled edges are fail-closed and
            # never fall back to Init or queue for a later profile.
            self._publish_progress_pedal_message(
                "五踏板进度踏板已忽略：请先选择可用的 progress profile；未执行 Init。",
                now=now,
            )
            return
        if self._handle_progress_pedal_event(event, now=now):
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
        if self._foot_pad_count() in FOOT_PAD_TWO_PEDAL_STYLE_COUNTS and operation == "init":
            return
        if operation == "fix" and not self._manual_single_arm_switch_enabled():
            self._publish_progress_pedal_message(
                "严格 Profile 模式已忽略手动 Fix/单臂踏板；活动臂仅由当前子任务定义。",
                now=now,
            )
            return
        if self._two_pedal_mode_active() and operation == "fix":
            # Pedal 1 means left-only teleop (fix right); pedal 2 means
            # right-only teleop (fix left).
            moving_side = side
            target_fixed_side = "right" if moving_side == "left" else "left"
            self._request_two_pedal_fixed_side(
                target_fixed_side,
                now=now,
                source=f"pedal_{1 if moving_side == 'left' else 2}",
            )
            return
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
        if controller.active:
            active_label = "左臂" if controller.active_side == "left" else "右臂"
            if controller.active_side != side:
                message = (
                    f"foot-pad {event} 已拒绝：当前是{active_label} Init；"
                    f"请再次踩{active_label}对应的 Init 踏板来暂停并重采 Home。"
                )
                self._publish_init_arm_state(message=message, control_hint=message)
                return
            if bool(getattr(controller, "recapturing", False)):
                message = "Init qpos 已暂停，正在仅重采目标侧 tracker Home；请保持该侧 tracker 稳定。"
                self._publish_init_arm_state(message=message, control_hint=message)
                return
            transition, release_errors = self._start_active_init_home_recapture(
                controller,
                now=now,
            )
            message = (
                f"{active_label} Init qpos 已暂停；Init 数采标签已结束，"
                "只重采目标侧 tracker Home，完成后自动恢复该臂遥操。"
                if transition.ok
                else f"{active_label} Init qpos 暂停/Home 重采失败: {transition.message}"
            )
            if release_errors:
                message += " 非目标臂精确控制权释放失败，已升级为常规 Hold。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return
        transition = controller.select_side(side)
        if not transition.ok:
            message = f"foot-pad 选择{side_label} init-arm失败: {transition.message}"
            self._publish_init_arm_state(message=message, control_hint=message)
            return
        self._handle_init_arm_move_to_qpos_event(now=now)

    def _handle_two_pedal_b_event(self, *, now: float) -> None:
        if not self._two_pedal_mode_active():
            return
        if not self._manual_single_arm_switch_enabled():
            self._publish_progress_pedal_message(
                "严格 Profile 模式已忽略双臂/单臂手动切换；活动臂仅由当前子任务定义。",
                now=now,
            )
            return
        if self._reject_two_pedal_switch_during_init(now=now):
            return
        controller = getattr(self.components, "fix_arm_controller", None)
        if self._hold_active:
            # The runtime consumes this as a fast, no-start-delay Home request.
            self._supersede_tracker_side_home_gate(
                reason="superseded_by_two_pedal_b"
            )
            self._two_pedal_transition_target_fixed_side = None
            self._two_pedal_fast_home_deadline = float(now) + 1.0
            self._request_calibration_with_tracker_side_guard(
                source="two_pedal_b",
                request_kind="two_pedal_fast",
                now=now,
            )
            self._publish_two_pedal_alert(
                level="orange",
                message=(
                    "2/5 踏板切换：正在检查左右 Tracker，并尽快采集双臂稳定 "
                    "Home 后恢复双臂遥操。"
                ),
                now=now,
            )
            return
        if bool(getattr(controller, "fixed", False)):
            # Single-arm -> bimanual is an in-loop atomic Fix release.  Keep
            # the currently moving arm and collection running; the fixed arm
            # remains frozen until its background Home refresh is ready.
            self._request_two_pedal_fixed_side(None, now=now, source="key_b")
            return
        self.request_hold(reason="two_pedal_b")

    def _request_two_pedal_fixed_side(
        self,
        fixed_side: str | None,
        *,
        now: float,
        source: str,
    ) -> None:
        if not self._two_pedal_mode_active():
            return
        if not self._manual_single_arm_switch_enabled():
            self._publish_progress_pedal_message(
                "严格 Profile 模式已忽略单臂切换请求。",
                now=now,
            )
            return
        if self._reject_two_pedal_switch_during_init(now=now):
            return
        if self._hold_active:
            self._supersede_tracker_side_home_gate(
                reason=f"superseded_by_{source}"
            )
            self._two_pedal_transition_target_fixed_side = fixed_side
            self._two_pedal_fast_home_deadline = float(now) + 1.0
            self._request_calibration_with_tracker_side_guard(
                source=source,
                request_kind="two_pedal_fast",
                now=now,
            )
            target_text = (
                "仅左臂遥操"
                if fixed_side == "right"
                else "仅右臂遥操"
            )
            self._publish_two_pedal_alert(
                level="orange",
                message=f"2/5 踏板切换：正在尽快采集稳定 Home，随后进入{target_text}。",
                now=now,
            )
            return
        controller = getattr(self.components, "fix_arm_controller", None)
        if controller is None or not self._teleop_started:
            return
        if not self._teleop_arm_dispatch_confirmed():
            self._publish_two_pedal_alert(
                level="red",
                message=(
                    "2/5 踏板切换已拒绝：当前没有已确认的机械臂命令下发；"
                    "请先进入 Hold，再通过快速 Home 恢复控制。"
                ),
                now=now,
                duration_sec=3.0,
            )
            return
        current_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
        if current_fixed:
            transition = controller.start_two_pedal_transition(
                fixed_side=fixed_side,
                now=now,
                timeout_sec=1.0,
                min_samples=1,
            )
            if transition.ok:
                pending_state = getattr(
                    controller,
                    "two_pedal_transition_pending",
                    None,
                )
                transition_pending = (
                    bool(pending_state)
                    if pending_state is not None
                    else bool(getattr(controller, "recapturing", True))
                )
                self._two_pedal_transition_active = transition_pending
                self._two_pedal_transition_target_fixed_side = (
                    fixed_side if transition_pending else None
                )
                if transition_pending:
                    target_text = (
                        "双臂遥操"
                        if fixed_side is None
                        else (
                            "仅左臂遥操"
                            if fixed_side == "right"
                            else "仅右臂遥操"
                        )
                    )
                    self._publish_two_pedal_alert(
                        level="orange",
                        message=(
                            f"2/5 踏板切换目标已更新为{target_text}；"
                            "当前控制状态保持不变，正在等待稳定 Home，最长 1 秒。"
                            f"\n{transition.message}"
                        ),
                        now=now,
                    )
                else:
                    self._clear_two_pedal_alert()
                    self._publish_fix_arm_state(
                        message=transition.message,
                        control_hint=transition.message,
                    )
            else:
                if not bool(
                    getattr(
                        controller,
                        "two_pedal_transition_pending",
                        self._two_pedal_transition_active,
                    )
                ):
                    self._two_pedal_transition_active = False
                    self._two_pedal_transition_target_fixed_side = None
                self._publish_two_pedal_alert(
                    level="red",
                    message=f"2/5 踏板切换失败：{transition.message}",
                    now=now,
                    duration_sec=3.0,
                )
            return
        if fixed_side is None:
            return
        transition = controller.select_side(fixed_side)
        if transition.ok:
            transition = controller.freeze_selected(now=now)
        if transition.ok:
            self._clear_two_pedal_alert()
            self._reset_runtime_tracking_for_side(
                fixed_side,
                tag="TwoPedal",
                now=now,
            )
            side_text = "右臂" if fixed_side == "right" else "左臂"
            moving_text = "左臂" if fixed_side == "right" else "右臂"
            message = f"2/5 踏板切换完成：{side_text} Fix，仅{moving_text}遥操。"
            self._publish_fix_arm_state(message=message, control_hint=message)
        else:
            self._publish_two_pedal_alert(
                level="red",
                message=f"2/5 踏板切换失败：{transition.message}",
                now=now,
                duration_sec=3.0,
            )

    def _two_pedal_switch_blocked_by_init(self) -> bool:
        controller = getattr(self.components, "init_arm_controller", None)
        return bool(
            controller is not None
            and (
                bool(getattr(controller, "active", False))
                or bool(getattr(controller, "qpos_active", False))
                or bool(getattr(controller, "moving", False))
                or bool(getattr(controller, "recapturing", False))
            )
        )

    def _reject_two_pedal_switch_during_init(self, *, now: float) -> bool:
        if not self._two_pedal_switch_blocked_by_init():
            return False
        self._publish_two_pedal_alert(
            level="red",
            message=(
                "2/5 踏板切换未执行：Init 移动、精确 qpos 或 Home 重采仍在进行；"
                "请先按 C 进入常规 Hold，再选择单臂/双臂遥操。"
            ),
            now=now,
            duration_sec=3.0,
        )
        return True

    def _start_active_init_home_recapture(self, controller, *, now: float):
        """End the Init label and recapture only the active target-side Home."""

        transition = controller.start_recalibration(
            now=now,
            timeout_sec=float(getattr(self.args, "calibration_timeout", 10.0)),
            min_samples=int(getattr(self.args, "calibration_samples", 20)),
        )
        release_errors: list[str] = []
        operation_id = str(getattr(controller, "qpos_operation_id", "") or "")
        if transition.ok and operation_id:
            support_side = "right" if transition.side == "left" else "left"
            _release_messages, release_errors = _release_gui_qpos_init_transit(
                self.components,
                operation_id=operation_id,
                now=now,
                sides=(support_side,),
            )
            if release_errors:
                self.request_hold(reason="init_qpos_support_release_failed")
        return transition, release_errors

    def _publish_fix_arm_state(self, *, message: str = "", control_hint: str = "") -> None:
        if self.ui is None:
            return
        updates = {
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            **self._pose_snapshot_ui_state(),
            "active_controls": self._current_active_controls(),
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
            **self._init_qpos_sequence_ui_state(),
            **self._pose_snapshot_ui_state(),
            "active_controls": self._current_active_controls(),
        }
        if control_hint:
            updates["control_hint"] = control_hint
        self.ui.publish(**updates)
        if message:
            self.ui.append_message("InitArm", message)

    def _init_qpos_sequence_ui_state(self) -> dict[str, object]:
        return {
            "init_arm_sequence_active": bool(
                getattr(self, "_init_qpos_sequence_active", False)
            ),
            "init_arm_sequence_side": str(
                getattr(self, "_init_qpos_sequence_side", "")
            ),
            "init_arm_sequence_phase": str(
                getattr(self, "_init_qpos_sequence_phase", "idle")
            ),
            "init_arm_sequence_status": str(
                getattr(self, "_init_qpos_sequence_status", "")
            ),
            "init_arm_qpos_speed_scale_by_side": dict(
                getattr(self, "_init_arm_qpos_speed_scale_by_side", {})
            ),
            "init_arm_qpos_speed_scale_min": float(
                MIN_INIT_ARM_QPOS_SPEED_SCALE
            ),
            "init_arm_qpos_speed_scale_max": float(
                MAX_INIT_ARM_QPOS_SPEED_SCALE
            ),
            "init_arm_qpos_speed_scale_default": float(
                DEFAULT_INIT_ARM_QPOS_SPEED_SCALE
            ),
        }

    def _set_init_arm_qpos_speed_from_gui(self, text: str) -> None:
        controller = getattr(self.components, "init_arm_controller", None)
        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不能修改归位速度倍率。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return
        if controller is not None and bool(getattr(controller, "active", False)):
            message = "Init 操作进行中不能修改归位速度；请等待完成或先触发 Init 急停。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return
        try:
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("参数必须是包含 left/right 的 JSON 对象")
            requested: dict[str, float] = {}
            for side in ("left", "right"):
                value = payload.get(side)
                if value is None:
                    continue
                numeric = float(value)
                if (
                    not np.isfinite(numeric)
                    or numeric < MIN_INIT_ARM_QPOS_SPEED_SCALE
                    or numeric > MAX_INIT_ARM_QPOS_SPEED_SCALE
                ):
                    raise ValueError(
                        f"{side} 必须在 {MIN_INIT_ARM_QPOS_SPEED_SCALE:g}–"
                        f"{MAX_INIT_ARM_QPOS_SPEED_SCALE:g} 之间"
                    )
                requested[side] = numeric
            if not requested:
                raise ValueError("没有提供 left/right 归位速度倍率")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"归位速度倍率无效：{exc}。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return

        for side, scale in requested.items():
            self._init_arm_qpos_speed_scale_by_side[side] = float(scale)
            setattr(self.args, f"init_arm_{side}_qpos_speed_scale", float(scale))
        left = self._init_arm_qpos_speed_scale_by_side["left"]
        right = self._init_arm_qpos_speed_scale_by_side["right"]
        message = (
            "归位速度倍率已更新："
            f"左臂={left:.3f}×，右臂={right:.3f}×（基线=1.0）；"
            "启动前归位、数采过程归位、踏板归位和双臂顺序归位统一使用该设置。"
        )
        _log("InitArmQpos", message)
        self._publish_init_arm_state(message=message, control_hint=message)

    def _cancel_init_arm_qpos_sequence(self, *, reason: str) -> None:
        if not bool(getattr(self, "_init_qpos_sequence_active", False)):
            return
        self._init_qpos_sequence_active = False
        self._init_qpos_sequence_phase = "aborted"
        self._init_qpos_sequence_status = f"双臂顺序归位已中止：{reason}"

    def _start_init_arm_sequence_side(self, side: str, *, now: float):
        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None:
            return None
        selected = controller.select_side(side)
        if not selected.ok:
            return selected
        transition = _start_gui_qpos_init(self.args, self.components, now=now)
        if transition.ok:
            self._init_qpos_sequence_side = side
            self._init_qpos_sequence_phase = f"{side}_qpos"
            label = "左臂" if side == "left" else "右臂"
            speed_scale = self._init_arm_qpos_speed_scale_by_side.get(
                side,
                DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
            )
            self._init_qpos_sequence_status = (
                f"双臂顺序归位：{label}正在以 {speed_scale:.3f}× "
                "执行精确关节归位。"
            )
        return transition

    def _restore_init_arm_sequence_selection(self) -> None:
        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None or bool(getattr(controller, "active", False)):
            return
        try:
            controller.select_side(self._init_qpos_sequence_restore_side)
        except Exception as exc:  # noqa: BLE001 - display selection is best effort.
            _log(
                "InitArmQpos",
                "WARNING: failed to restore Init-arm selection after sequence: "
                f"{type(exc).__name__}: {exc}",
            )

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
        active_special_operation = bool(
            getattr(controller, "qpos_active", False)
            or getattr(controller, "recapturing", False)
            or getattr(controller, "active", False)
        )
        if (
            not active_special_operation
            and not self._teleop_arm_dispatch_confirmed()
        ):
            message = (
                "Init 被拒绝：当前没有已确认的机械臂命令下发；"
                "请先进入 Hold 并重新采集 runtime Home。"
            )
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        try:
            if bool(getattr(controller, "qpos_active", False)):
                transition, release_errors = self._start_active_init_home_recapture(
                    controller,
                    now=now,
                )
                side_label = "左臂" if transition.side == "left" else "右臂"
                message = (
                    f"{side_label}精确 Init 已结束；正在保持目标臂并重采该侧 tracker Home，"
                    "完成后自动恢复普通遥操。"
                    if transition.ok
                    else f"{side_label}精确 Init/Home 重采启动失败: {transition.message}"
                )
                if release_errors:
                    message += " 非目标臂精确控制权释放失败，已请求常规 Hold。"
                self._publish_init_arm_state(message=message, control_hint=message)
                return True
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
            any_side_fixed = bool(
                fix_controller is not None
                and getattr(fix_controller, "fixed", False)
            )
            if any_side_fixed:
                fixed_sides = tuple(getattr(fix_controller, "fixed_sides", ()) or ())
                fixed_text = "/".join("左臂" if item == "left" else "右臂" for item in fixed_sides)
                message = (
                    f"Init 被拒绝：{fixed_text or '已有机械臂'}处于 Fix；"
                    "请先解除 Fix，避免 Init 移动与另一侧固定形成死锁。"
                )
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

    def _handle_init_arm_move_to_qpos_event(self, *, now: float) -> bool:
        """Start the dedicated collision-planned exact-qpos operation."""

        controller = getattr(self.components, "init_arm_controller", None)
        startup_maintenance = bool(
            getattr(self, "_startup_qpos_maintenance_enabled", False)
            and not self._teleop_started
        )
        hold_maintenance = bool(self._teleop_started and self._hold_active)
        if controller is None or (
            not self._teleop_started and not startup_maintenance
        ):
            return False
        if self._hold_requested:
            message = "Hold 正在生效，尚未启动 Init qpos；请在停臂完成后重试。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if (
            self._teleop_started
            and not self._hold_active
            and not self._teleop_arm_dispatch_confirmed()
        ):
            message = (
                "Init qpos 未启动：当前没有已确认的机械臂命令下发；"
                "请先进入 Hold 后再执行维护操作。"
            )
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if self._hold_active and not hold_maintenance:
            return False
        if self._episode_transition_busy():
            message = "数采开始倒计时或后台保存期间禁止启动 Init qpos；请等待状态稳定。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if bool(getattr(controller, "active", False)):
            side = str(getattr(controller, "active_side", "") or "left")
            side_label = "左臂" if side == "left" else "右臂"
            message = f"{side_label}正在执行 Init 操作；请点击 Hold 取消或等待完成。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        try:
            transition = _start_gui_qpos_init(self.args, self.components, now=now)
        except Exception as exc:  # noqa: BLE001 - planning admission must fail closed.
            message = f"Init qpos 启动失败，未发送轨迹: {type(exc).__name__}: {exc}"
            _log("InitArmQpos", message, now=now)
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        side_label = "左臂" if transition.side == "left" else "右臂"
        speed_scale = self._init_arm_qpos_speed_scale_by_side.get(
            transition.side,
            DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
        )
        message = (
            f"{side_label} Init qpos 已进入后台路径规划（归位速度 {speed_scale:.3f}×）；"
            "规划/执行期间另一臂固定在实测起点。"
            if transition.ok
            else f"Init qpos 请求被拒绝: {transition.message}"
        )
        self._publish_init_arm_state(message=message, control_hint=message)
        return True

    def _handle_init_arm_move_both_to_qpos_event(self, *, now: float) -> bool:
        """Run exact Init for left then right, then enter regular Hold."""

        controller = getattr(self.components, "init_arm_controller", None)
        startup_maintenance = bool(
            getattr(self, "_startup_qpos_maintenance_enabled", False)
            and not self._teleop_started
        )
        hold_maintenance = bool(self._teleop_started and self._hold_active)
        if controller is None or (
            not self._teleop_started and not startup_maintenance
        ):
            return False
        if self._hold_requested:
            message = "Hold 正在生效，尚未启动双臂顺序归位；请在停臂完成后重试。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if (
            self._teleop_started
            and not self._hold_active
            and not self._teleop_arm_dispatch_confirmed()
        ):
            message = (
                "双臂顺序归位未启动：当前没有已确认的机械臂命令下发；"
                "请先进入 Hold 后再执行维护操作。"
            )
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if self._hold_active and not hold_maintenance:
            return False
        if self._episode_transition_busy():
            message = "数采开始倒计时或后台保存期间禁止启动双臂顺序归位。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True
        if bool(getattr(controller, "active", False)):
            message = "已有 Init 操作正在执行；请等待完成或点击 Hold 后重试。"
            self._publish_init_arm_state(message=message, control_hint=message)
            return True

        self._init_qpos_sequence_restore_side = str(
            getattr(controller, "selected_side", "left") or "left"
        )
        self._init_qpos_sequence_active = True
        self._init_qpos_sequence_phase = "starting_left"
        self._init_qpos_sequence_status = "双臂顺序归位正在启动左臂。"
        try:
            transition = self._start_init_arm_sequence_side("left", now=now)
        except Exception as exc:  # noqa: BLE001 - exact-qpos admission fails closed.
            transition = None
            failure = f"{type(exc).__name__}: {exc}"
        else:
            failure = "Init controller unavailable" if transition is None else transition.message
        if transition is None or not transition.ok:
            self._cancel_init_arm_qpos_sequence(reason=f"左臂启动失败：{failure}")
            self._restore_init_arm_sequence_selection()
            message = self._init_qpos_sequence_status
            _log("InitArmQpos", message, now=now)
            self._publish_init_arm_state(message=message, control_hint=message)
            return True

        message = (
            "双臂顺序精确归位已启动：先左臂再右臂；过程不会修改 tracker Home，"
            + (
                "双臂到位后停留在启动前 Hold。"
                if startup_maintenance
                else "双臂到位后自动进入常规 Hold。"
            )
        )
        self._publish_init_arm_state(message=message, control_hint=message)
        return True

    def handle_init_arm_qpos_terminal(self, terminal: Mapping[str, object], *, now: float) -> None:
        """Advance left -> right without changing Home; finish in regular Hold."""

        if not self._init_qpos_sequence_active:
            return
        status = str(terminal.get("status", "failed") or "failed")
        side = str(terminal.get("side", "") or "")
        if side != self._init_qpos_sequence_side or status != "completed":
            reason = str(terminal.get("message", "") or status)
            if side != self._init_qpos_sequence_side:
                reason = f"终止事件侧不匹配（期望 {self._init_qpos_sequence_side}，收到 {side or '-'}）"
            self._cancel_init_arm_qpos_sequence(reason=reason)
            self.request_hold(reason="init_qpos_sequence_terminal_failed")
            return

        controller = getattr(self.components, "init_arm_controller", None)
        if controller is None:
            self._cancel_init_arm_qpos_sequence(reason="Init controller unavailable")
            self.request_hold(reason="init_qpos_sequence_controller_missing")
            return

        if side == "left":
            operation_id = str(getattr(controller, "qpos_operation_id", "") or "")
            _release_messages, release_errors = _release_gui_qpos_init_transit(
                self.components,
                operation_id=operation_id,
                now=now,
                sides=("left",),
            )
            if release_errors:
                self._cancel_init_arm_qpos_sequence(
                    reason="左臂精确控制权释放失败：" + "；".join(release_errors)
                )
                self.request_hold(reason="init_qpos_sequence_left_release_failed")
                return
            try:
                released = controller.cancel_active(
                    now=now,
                    reason="dual_qpos_sequence_left_completed_without_home_recapture",
                )
            except Exception as exc:  # noqa: BLE001 - do not start the peer arm.
                self._cancel_init_arm_qpos_sequence(
                    reason=f"左臂 Init 状态释放异常：{type(exc).__name__}: {exc}"
                )
                self.request_hold(reason="init_qpos_sequence_left_cancel_failed")
                return
            if not bool(getattr(released, "ok", False)):
                self._cancel_init_arm_qpos_sequence(
                    reason=f"左臂 Init 状态释放失败：{getattr(released, 'message', '')}"
                )
                self.request_hold(reason="init_qpos_sequence_left_cancel_failed")
                return

            self._init_qpos_sequence_phase = "starting_right"
            self._init_qpos_sequence_status = (
                "双臂顺序归位：左臂已实测到位；未修改 tracker Home，正在启动右臂。"
            )
            try:
                transition = self._start_init_arm_sequence_side("right", now=now)
            except Exception as exc:  # noqa: BLE001 - admission fails closed.
                transition = None
                failure = f"{type(exc).__name__}: {exc}"
            else:
                failure = (
                    "Init controller unavailable"
                    if transition is None
                    else transition.message
                )
            if transition is None or not transition.ok:
                self._cancel_init_arm_qpos_sequence(
                    reason=f"右臂启动失败：{failure}"
                )
                self.request_hold(reason="init_qpos_sequence_right_start_failed")
            self._publish_init_arm_state(
                message=self._init_qpos_sequence_status,
                control_hint=self._init_qpos_sequence_status,
            )
            return

        self._init_qpos_sequence_active = False
        self._init_qpos_sequence_phase = "completed_hold_requested"
        startup_maintenance = bool(
            getattr(self, "_startup_qpos_maintenance_enabled", False)
            and not self._teleop_started
        )
        self._init_qpos_sequence_status = (
            "双臂顺序精确归位完成；未自动重采 tracker Home，当前保持在启动前 Hold。"
            if startup_maintenance
            else "双臂顺序精确归位完成；未自动重采 tracker Home，正在进入常规 Hold。"
        )
        if not startup_maintenance:
            self.request_hold(reason="init_qpos_sequence_completed")
        self._publish_init_arm_state(
            message=self._init_qpos_sequence_status,
            control_hint=self._init_qpos_sequence_status,
        )

    def set_startup_qpos_maintenance_enabled(self, enabled: bool) -> None:
        """Expose pre-Home exact-qpos controls only with an active executor."""

        self._startup_qpos_maintenance_enabled = bool(enabled)
        if self.ui is not None:
            self.ui.publish(active_controls=self._current_active_controls())

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
            selected_side = str(getattr(controller, "selected_side", "left") or "left")
            selected_is_fixed = bool(
                controller.is_fixed(selected_side)
                if callable(getattr(controller, "is_fixed", None))
                else controller.fixed and controller.fixed_side == selected_side
            )
            if (
                not selected_is_fixed
                and not bool(getattr(controller, "recapturing", False))
                and not self._teleop_arm_dispatch_confirmed()
            ):
                message = (
                    "Fix 被拒绝：当前没有已确认的机械臂命令下发，"
                    "不能从旧命令快照建立新的固定状态；请先进入 Hold 并重新采集 runtime Home。"
                )
                self._publish_fix_arm_state(message=message, control_hint=message)
                return True
            if selected_is_fixed:
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
            if bool(getattr(init_controller, "moving", False)) or bool(
                getattr(init_controller, "recapturing", False)
            ):
                message = "Init qpos 移动/目标侧 Home 重采期间不能新启用 Fix-arm；请等待该流程结束或点击 Hold 取消。"
                self._publish_fix_arm_state(message=message, control_hint=message)
                return True
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
            init_other_side_moving = bool(
                init_controller is not None
                and getattr(init_controller, "moving", False)
                and getattr(init_controller, "active_side", "") != selected_side
            )
            if init_other_side_moving:
                side_label = "左臂" if selected_side == "left" else "右臂"
                message = (
                    f"{side_label} Fix 被拒绝：另一侧 Init 正在移动；"
                    "请等待 Init 到位固定或先取消 Init。"
                )
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
        if (
            not self._teleop_started
            or self._hold_active
            or not self._teleop_arm_dispatch_confirmed()
        ):
            message = "Pose 抽帧只能在机械臂命令已实际派发的遥操帧中执行。"
            self.set_pose_snapshot_state(status=message, append_message=True)
            return
        self._pose_snapshot_requested = True
        self.set_pose_snapshot_state(status="已请求抽帧；等待下一帧遥操状态。")

    def consume_pose_snapshot_request(self) -> bool:
        requested = bool(self._pose_snapshot_requested)
        self._pose_snapshot_requested = False
        return requested

    def _queue_runtime_home_request(
        self,
        *,
        from_hold: bool,
        request_kind: str = "normal",
    ) -> None:
        self._calibration_requested = True
        self._calibration_request_from_hold = bool(from_hold)
        self._two_pedal_fast_home_requested = (
            str(request_kind) == "two_pedal_fast"
        )
        if (
            self._two_pedal_fast_home_requested
            and self._two_pedal_fast_home_deadline <= 0.0
        ):
            self._two_pedal_fast_home_deadline = time.time() + 1.0
        self._tracker_side_guard_request_active = not bool(
            self._tracker_side_initial_check_completed
        )

    def _complete_initial_tracker_side_check(self, *, result: str) -> None:
        """Latch the one-time tracker-side admission result for this GUI."""

        self._tracker_side_initial_check_completed = True
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._tracker_side_check.update(
            {
                "warning_required": False,
                "home_request_pending": False,
                "confirmation_required": False,
                "confirmation_result": str(result),
                "session_check_completed": True,
                "monitoring_active": False,
            }
        )

    def _supersede_tracker_side_home_gate(self, *, reason: str) -> None:
        """Give a later non-interactive Home request sole ownership.

        A two-pedal request may replace an earlier explicit Home action.  If
        both requests stayed alive, a stale confirmation could mutate the
        later transition or make the main loop select the wrong Home path.
        """

        had_pending_gate = bool(
            self._tracker_side_confirmation_pending
            or self._tracker_side_request_pending
            or self._tracker_side_guard_request_active
            or self._tracker_side_override_granted
        )
        if had_pending_gate:
            self._calibration_requested = False
            self._calibration_request_from_hold = False
        self._tracker_side_confirmation_pending = False
        self._tracker_side_confirmation_from_hold = False
        self._tracker_side_confirmation_request_kind = "normal"
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._two_pedal_fast_home_requested = False
        self._two_pedal_fast_home_deadline = 0.0
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": False,
                "confirmation_result": str(reason),
            }
        )
        if had_pending_gate:
            _log(
                "InputMapping",
                "pending explicit tracker-side Home gate "
                f"{reason.replace('_', ' ')}",
            )

    def _tracker_side_swap_available(self) -> bool:
        return bool(
            not self._tracker_swap_locked
            or self._single_arm_tracker_swap_allowed_in_hold()
        )

    def _tracker_side_correction_hint(self) -> str:
        if self._tracker_side_swap_available():
            return "选择“取消”后可点击“交换左右 Tracker”，再重新开始。"
        return (
            "当前双臂运行映射已锁定；请选择“取消”，把左右手恢复到身体对应一侧后重试。"
            "若确认启动映射本身错误，请安全退出并重新启动后交换。"
        )

    def _begin_tracker_side_confirmation(
        self,
        *,
        from_hold: bool,
        source: str,
        request_kind: str = "normal",
    ) -> None:
        self._tracker_side_confirmation_pending = True
        self._tracker_side_confirmation_from_hold = bool(from_hold)
        self._tracker_side_confirmation_request_kind = str(request_kind)
        self._tracker_side_confirmation_token_counter += 1
        token = f"tracker-side-{self._tracker_side_confirmation_token_counter}"
        warning_message = str(
            self._tracker_side_check.get("message")
            or "Tracker 左右位置检查未通过。"
        )
        correction_hint = self._tracker_side_correction_hint()
        confirmation_message = (
            f"{warning_message}\n\n"
            "是否仍要忽略此警告并继续本次 runtime Home？\n\n"
            f"{correction_hint}"
        )
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": True,
                "confirmation_token": token,
                "confirmation_message": confirmation_message,
                "confirmation_result": "pending",
                "swap_available": self._tracker_side_swap_available(),
            }
        )
        _log(
            "InputMapping",
            "runtime Home blocked for tracker-side confirmation: "
            + warning_message,
        )
        if self.ui is not None:
            self.ui.publish(
                tracker_side_check=dict(self._tracker_side_check),
                stage="确认 Tracker 左右位置",
                active_controls=self._current_active_controls(),
                control_hint=(
                    "检测到 Tracker 左右位置异常或无法可靠判定；"
                    "请在弹窗中二次确认。"
                ),
                home_ready=False,
            )
            self.ui.append_message(
                "InputMapping",
                f"{warning_message} {correction_hint}",
            )
        elif source == "terminal":
            _log(
                "InputMapping",
                "再次按 Home 键将忽略左右位置警告并继续；"
                "也可先修正 Tracker 佩戴或映射。",
            )

    def _request_calibration_with_tracker_side_guard(
        self,
        *,
        source: str,
        request_kind: str = "normal",
        now: float | None = None,
    ) -> None:
        """Gate an explicit runtime-Home request on the latest side check."""

        if self._calibration_requested:
            _log(
                "InputMapping",
                "explicit runtime Home ignored because another Home request "
                "is already queued",
            )
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中重采前请先点击 Hold / 冻结。"
            if self.ui is not None:
                self.ui.append_message("Hold", message)
                self.ui.publish(
                    active_controls=self._current_active_controls(),
                    control_hint=message,
                    teleop_running=self._teleop_arm_dispatch_confirmed(),
                    gripper_control_enabled=self.gripper_control_enabled(),
                )
            return
        if self._tracker_side_confirmation_pending:
            if source == "terminal" and self.ui is None:
                self._confirm_tracker_side_warning()
            elif source == "terminal":
                _log(
                    "InputMapping",
                    "terminal Home confirmation ignored while Web GUI confirmation is pending",
                )
            return
        if self._tracker_side_request_pending:
            return

        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        from_hold = bool(self._hold_active)
        if self._tracker_side_initial_check_completed:
            self._tracker_side_check.update(
                {
                    "home_request_pending": False,
                    "confirmation_required": False,
                    "session_check_completed": True,
                    "monitoring_active": False,
                }
            )
            self._queue_runtime_home_request(
                from_hold=from_hold,
                request_kind=request_kind,
            )
            return
        if bool(self._tracker_side_check.get("warning_required", False)):
            self._begin_tracker_side_confirmation(
                from_hold=from_hold,
                source=source,
                request_kind=request_kind,
            )
            return

        status = str(self._tracker_side_check.get("status", "unavailable"))
        dual_side = set(self.components.active_sides) == {"left", "right"}
        if dual_side and status != "correct":
            self._tracker_side_request_pending = True
            self._tracker_side_request_from_hold = from_hold
            self._tracker_side_request_source = str(source)
            self._tracker_side_request_kind = str(request_kind)
            request_time = time.time() if now is None else float(now)
            self._tracker_side_request_deadline = (
                self._two_pedal_fast_home_deadline
                if (
                    str(request_kind) == "two_pedal_fast"
                    and self._two_pedal_fast_home_deadline > 0.0
                )
                else request_time + DEFAULT_RUNTIME_HOME_FRESH_DATA_TIMEOUT_SEC
            )
            self._tracker_side_check.update(
                {
                    "home_request_pending": True,
                    "confirmation_required": False,
                    "confirmation_result": "waiting_for_fresh_check",
                }
            )
            message = (
                "Home 请求已暂缓：正在等待左右 Tracker 同时在线、数据新鲜，"
                "并完成左右空间位置检查。"
            )
            _log("InputMapping", message)
            if self.ui is not None:
                self.ui.publish(
                    tracker_side_check=dict(self._tracker_side_check),
                    stage="等待 Tracker 左右检查",
                    active_controls=self._current_active_controls(),
                    control_hint=message,
                    home_ready=False,
                )
                self.ui.append_message("InputMapping", message)
            return

        self._tracker_side_check["home_request_pending"] = False
        self._queue_runtime_home_request(
            from_hold=from_hold,
            request_kind=request_kind,
        )

    def _confirm_tracker_side_warning(self) -> None:
        if not self._tracker_side_confirmation_pending:
            return
        from_hold = bool(self._tracker_side_confirmation_from_hold)
        request_kind = str(self._tracker_side_confirmation_request_kind)
        self._tracker_side_confirmation_pending = False
        self._tracker_side_confirmation_from_hold = False
        self._tracker_side_confirmation_request_kind = "normal"
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        if request_kind == "two_pedal_fast":
            # Operator deliberation in the mandatory warning popup is not a
            # failed stability acquisition.  Start the one-second switching
            # budget when they explicitly continue.
            self._two_pedal_fast_home_deadline = time.time() + 1.0
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": False,
                "confirmation_result": "continued",
            }
        )
        self._complete_initial_tracker_side_check(
            result="continued_for_gui_session"
        )
        self._queue_runtime_home_request(
            from_hold=from_hold,
            request_kind=request_kind,
        )
        message = (
            "已二次确认忽略 Tracker 左右位置警告；本 GUI 会话后续 Home "
            "不再重复执行左右位置检测。"
        )
        _log("InputMapping", message)
        if self.ui is not None:
            self.ui.publish(
                tracker_side_check=dict(self._tracker_side_check),
                active_controls=self._current_active_controls(),
                control_hint=message,
            )
            self.ui.append_message("InputMapping", message)

    def _cancel_tracker_side_warning(self) -> None:
        if not self._tracker_side_confirmation_pending:
            return
        request_kind = str(self._tracker_side_confirmation_request_kind)
        self._tracker_side_confirmation_pending = False
        self._tracker_side_confirmation_from_hold = False
        self._tracker_side_confirmation_request_kind = "normal"
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._two_pedal_fast_home_requested = False
        self._two_pedal_fast_home_deadline = 0.0
        if request_kind == "two_pedal_fast":
            self._two_pedal_transition_target_fixed_side = None
            self._clear_two_pedal_alert()
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": False,
                "confirmation_result": "cancelled",
            }
        )
        message = (
            "已取消本次 runtime Home；请确认 Tracker 佩戴。"
            f"{self._tracker_side_correction_hint()}"
        )
        _log("InputMapping", message)
        if self.ui is not None:
            self.ui.publish(
                tracker_side_check=dict(self._tracker_side_check),
                active_controls=self._current_active_controls(),
                control_hint=message,
            )
            self.ui.append_message("InputMapping", message)

    def _update_tracker_side_check(self, payload: Mapping[str, object]) -> None:
        if self._tracker_side_initial_check_completed:
            self._tracker_side_check.update(
                {
                    "warning_required": False,
                    "home_request_pending": False,
                    "confirmation_required": False,
                    "session_check_completed": True,
                    "monitoring_active": False,
                }
            )
            return
        next_payload = dict(payload)
        next_payload.update(
            {
                "session_check_completed": False,
                "monitoring_active": True,
            }
        )
        if self._tracker_side_confirmation_pending:
            if bool(next_payload.get("warning_required", False)):
                next_payload.update(
                    {
                        "confirmation_required": True,
                        "confirmation_token": self._tracker_side_check.get(
                            "confirmation_token", ""
                        ),
                        "confirmation_message": (
                            f"{next_payload.get('message', '')}\n\n"
                            "是否仍要忽略此警告并继续本次 runtime Home？\n\n"
                            f"{self._tracker_side_correction_hint()}"
                        ),
                        "confirmation_result": "pending",
                        "swap_available": self._tracker_side_swap_available(),
                    }
                )
            elif str(next_payload.get("status", "")) == "correct":
                resolved_request_kind = str(
                    self._tracker_side_confirmation_request_kind
                )
                self._tracker_side_confirmation_pending = False
                self._tracker_side_confirmation_from_hold = False
                self._tracker_side_confirmation_request_kind = "normal"
                if resolved_request_kind == "two_pedal_fast":
                    self._two_pedal_transition_target_fixed_side = None
                    self._two_pedal_fast_home_deadline = 0.0
                    self._clear_two_pedal_alert()
                next_payload.update(
                    {
                        "confirmation_required": False,
                        "confirmation_token": self._tracker_side_check.get(
                            "confirmation_token", ""
                        ),
                        "confirmation_result": "resolved",
                    }
                )
                if self.ui is not None:
                    self.ui.append_message(
                        "InputMapping",
                        "Tracker 左右位置现已通过检查；已撤销待确认警告，请重新点击“开始遥操”。",
                    )
            else:
                # A transient packet loss must not invalidate a confirmation
                # dialog that the operator is currently reading.  Keep the
                # last concrete warning and expose the live unavailable state
                # only as diagnostic context.
                previous = dict(self._tracker_side_check)
                previous["live_status"] = next_payload.get("status", "")
                previous["live_message"] = next_payload.get("message", "")
                next_payload = previous
        else:
            next_payload.setdefault("confirmation_required", False)
            next_payload.setdefault(
                "confirmation_token",
                self._tracker_side_check.get("confirmation_token", ""),
            )
        self._tracker_side_check = next_payload
        if self._tracker_side_request_pending:
            status = str(self._tracker_side_check.get("status", "unavailable"))
            if bool(self._tracker_side_check.get("warning_required", False)):
                from_hold = bool(self._tracker_side_request_from_hold)
                source = str(self._tracker_side_request_source or "runtime")
                request_kind = str(self._tracker_side_request_kind)
                self._tracker_side_request_pending = False
                self._tracker_side_request_from_hold = False
                self._tracker_side_request_source = ""
                self._tracker_side_request_kind = "normal"
                self._tracker_side_request_deadline = 0.0
                self._begin_tracker_side_confirmation(
                    from_hold=from_hold,
                    source=source,
                    request_kind=request_kind,
                )
                return
            if status == "correct":
                from_hold = bool(self._tracker_side_request_from_hold)
                request_kind = str(self._tracker_side_request_kind)
                self._tracker_side_request_pending = False
                self._tracker_side_request_from_hold = False
                self._tracker_side_request_source = ""
                self._tracker_side_request_kind = "normal"
                self._tracker_side_request_deadline = 0.0
                self._tracker_side_check.update(
                    {
                        "home_request_pending": False,
                        "confirmation_required": False,
                        "confirmation_result": "check_passed",
                    }
                )
                self._queue_runtime_home_request(
                    from_hold=from_hold,
                    request_kind=request_kind,
                )
                message = "Tracker 左右位置检查通过；继续本次 runtime Home。"
                _log("InputMapping", message)
                if self.ui is not None:
                    self.ui.publish(
                        tracker_side_check=dict(self._tracker_side_check),
                        stage="即将开始 runtime Home",
                        active_controls=["q"],
                        control_hint=message,
                        home_ready=False,
                    )
                    self.ui.append_message("InputMapping", message)
                return
            self._tracker_side_check["home_request_pending"] = True
        if self.ui is not None:
            self.ui.publish(
                tracker_side_check=dict(self._tracker_side_check),
                active_controls=self._current_active_controls(),
            )

    def _poll_tracker_side_request_timeout(self, *, now: float) -> None:
        if (
            not self._tracker_side_request_pending
            or self._tracker_side_request_deadline <= 0.0
            or float(now) < self._tracker_side_request_deadline
        ):
            return
        request_kind = str(self._tracker_side_request_kind)
        request_from_hold = bool(self._tracker_side_request_from_hold)
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._two_pedal_fast_home_requested = False
        self._two_pedal_fast_home_deadline = 0.0
        self._two_pedal_transition_target_fixed_side = None
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": False,
                "confirmation_result": (
                    "two_pedal_tracker_timeout"
                    if request_kind == "two_pedal_fast"
                    else "runtime_home_tracker_timeout"
                ),
            }
        )
        if request_kind == "two_pedal_fast":
            message = (
                "双踏板切换失败：1 秒内未获得新鲜且可判定的左右 Tracker；"
                "本次切换已取消。"
            )
            self._publish_two_pedal_alert(
                level="red",
                message=message,
                now=now,
                duration_sec=3.0,
            )
            if self.ui is not None:
                self.ui.publish(
                    tracker_side_check=dict(self._tracker_side_check),
                    active_controls=self._current_active_controls(),
                )
                self.ui.append_message("TwoPedal", message)
            return

        message = (
            "恢复遥操已取消：1 秒内未获得新鲜且可判定的左右 Tracker；"
            "机械臂保持 Hold，可恢复 VIVE 后重新点击“开始遥操”。"
            if request_from_hold
            else (
                "开始遥操已取消：1 秒内未获得新鲜且可判定的左右 Tracker；"
                "已回到未遥操状态，可恢复 VIVE 后重新点击“开始遥操”。"
            )
        )
        _log("InputMapping", message, now=now)
        self.publish_calibration_waiting(message)

    def _final_tracker_side_home_admission(self) -> bool:
        """Recheck the fresh side relation immediately before Home capture."""

        if self._tracker_side_initial_check_completed:
            self._tracker_side_guard_request_active = False
            self._tracker_side_override_granted = False
            return True
        if not self._tracker_side_guard_request_active:
            return True
        self._tracker_side_guard_request_active = False
        override_granted = bool(self._tracker_side_override_granted)
        self._tracker_side_override_granted = False
        if override_granted:
            self._complete_initial_tracker_side_check(
                result="continued_for_gui_session"
            )
            return True
        if set(self.components.active_sides) != {"left", "right"}:
            self._complete_initial_tracker_side_check(result="not_required")
            return True
        if str(self._tracker_side_check.get("status", "")) == "correct":
            self._complete_initial_tracker_side_check(result="check_passed")
            return True
        self._last_calibration_started_from_hold = False
        if self._teleop_started:
            self._hold_active = True
            self._gripper_control_enabled = False
        message = (
            "Home 已取消：准备倒计时结束时没有通过新鲜的 Tracker 左右位置复核。"
            f"{self._tracker_side_check.get('message', '')}"
            "请恢复左右位置/连接后重新点击“开始遥操”。"
        )
        _log("InputMapping", message)
        if self.ui is not None:
            self.ui.publish(
                tracker_side_check=dict(self._tracker_side_check),
                active_controls=self._current_active_controls(),
                control_hint=message,
                home_ready=False,
            )
            self.ui.append_message("InputMapping", message)
        return False

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
                        active_controls=self._current_active_controls(),
                        control_hint="运行中重采前请先点击 Hold / 冻结；Hold 只停臂，不会自动重采。",
                        teleop_running=self._teleop_arm_dispatch_confirmed(),
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

    def consume_hitl_source_requests(self) -> tuple[str, ...]:
        requests = tuple(self._hitl_source_requests)
        self._hitl_source_requests.clear()
        return requests

    def request_runtime_home_for_hitl(self) -> bool:
        """Queue the established guarded runtime-Home flow from HITL HOLD."""

        if not self._teleop_started or not self._hold_active:
            return False
        self._request_calibration_with_tracker_side_guard(source="hitl")
        return bool(
            self._calibration_requested
            or self._tracker_side_request_pending
            or self._tracker_side_confirmation_pending
        )

    def consume_two_pedal_fast_home_target(
        self,
        *,
        now: float | None = None,
    ) -> tuple[bool, str | None]:
        """Return and clear a pending Hold-to-teleop two-pedal transition."""

        if (
            not self._two_pedal_fast_home_requested
            or not self._calibration_requested
            or not self._calibration_request_from_hold
        ):
            return False, None
        current_time = time.time() if now is None else float(now)
        remaining_sec = self._two_pedal_fast_home_deadline - current_time
        if remaining_sec <= 0.0:
            self._two_pedal_fast_home_requested = False
            self._calibration_requested = False
            self._calibration_request_from_hold = False
            self._tracker_side_guard_request_active = False
            self._tracker_side_override_granted = False
            self._two_pedal_transition_target_fixed_side = None
            self._two_pedal_fast_home_deadline = 0.0
            self._publish_two_pedal_alert(
                level="red",
                message="双踏板切换失败：总切换时间已超过 1 秒；本次切换已取消。",
                now=current_time,
                duration_sec=3.0,
            )
            return False, None
        target = self._two_pedal_transition_target_fixed_side
        self._two_pedal_fast_home_timeout_sec = max(0.001, remaining_sec)
        self._two_pedal_fast_home_requested = False
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        if not self._final_tracker_side_home_admission():
            self._two_pedal_transition_target_fixed_side = None
            self._two_pedal_fast_home_deadline = 0.0
            self._publish_two_pedal_alert(
                level="red",
                message=(
                    "双踏板切换失败：执行快速 Home 前的 Tracker 左右复核未通过；"
                    "本次切换已取消。"
                ),
                now=time.time(),
                duration_sec=3.0,
            )
            return False, None
        self._last_calibration_started_from_hold = True
        self._hold_active = False
        self._gripper_control_enabled = False
        return True, target

    def two_pedal_fast_home_timeout_sec(self) -> float:
        return max(
            0.001,
            min(1.0, float(self._two_pedal_fast_home_timeout_sec)),
        )

    def finish_two_pedal_fast_home(
        self,
        *,
        success: bool,
        now: float,
        message: str,
    ) -> None:
        target = self._two_pedal_transition_target_fixed_side
        self._two_pedal_transition_target_fixed_side = None
        self._two_pedal_fast_home_deadline = 0.0
        self._two_pedal_fast_home_timeout_sec = 1.0
        if success:
            self._two_pedal_post_home_fixed_side = target
            if target is None:
                self._clear_two_pedal_alert()
            if self.ui is not None:
                self.ui.append_message("TwoPedal", message)
        else:
            self._hold_active = True
            self._publish_two_pedal_alert(
                level="red",
                message=message,
                now=now,
                duration_sec=3.0,
            )
            if self.ui is not None:
                self.ui.append_message("TwoPedal", message)

    def consume_two_pedal_post_home_fixed_side(self) -> str | None:
        side = self._two_pedal_post_home_fixed_side
        self._two_pedal_post_home_fixed_side = None
        return side

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
        if self._hold_requested:
            message = "Hold 正在生效，暂不修改 movement_multiple；请在停臂完成后重试。"
            _log("MovementMultiple", f"{source}: rejected during hold transition")
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
            requested = parse_gripper_force_request(
                text,
                require_nonempty=True,
            )
        except GripperForceRequestError as exc:
            reason = str(exc)
            message = (
                reason
                if reason == "未提供可设置的左/右夹爪力度。"
                else f"夹爪力度参数无效: {reason}"
            )
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

    def _persist_gripper_force_from_gui(self, text: str) -> None:
        """Make the selected force an immutable contract for a new dataset."""

        if self._episode_busy():
            message = "数采录制、暂停、倒计时或保存中不能修改数据集夹爪力度；请先停止当前流程。"
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return
        dataset_root = str(getattr(self.args, "dataset_root", "") or "").strip()
        if not dataset_root:
            message = "永久夹爪力度未保存：请先选择一个新的数据集目录。"
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return
        try:
            requested = parse_gripper_force_request(
                text,
                require_nonempty=False,
            )
        except GripperForceRequestError as exc:
            message = f"永久夹爪力度参数无效: {exc}"
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return

        active_sides = tuple(
            side
            for side in self.components.active_sides
            if _gripper_backend_for_side(self.components, side) is not None
        )
        if not active_sides:
            message = "永久夹爪力度未保存：没有可用的 DM-Tac 夹爪。"
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return
        policies = _resolved_gripper_policies_by_side(self.args, active_sides)
        force_by_side = {
            side: float(policies[side]["torque"])
            for side in active_sides
            if side in policies
        }
        for side, force in requested.items():
            if side in force_by_side:
                force_by_side[side] = force
        if not force_by_side:
            message = "永久夹爪力度未保存：没有可设置的活动侧。"
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return

        contract = GripperForceDatasetContract(
            force_by_side=tuple(force_by_side.items()),
            profile_name=str(getattr(self.args, "gripper_task_profile", "") or ""),
            task=str(getattr(self.args, "gripper_task", "") or ""),
        )
        try:
            path = bind_gripper_force_contract(Path(dataset_root), contract)
        except Exception as exc:  # noqa: BLE001 - surface dataset-level rejection in GUI.
            message = (
                "永久夹爪力度未保存：只允许新建或空数据集，且一个数据集不能混用不同设置。"
                f" {type(exc).__name__}: {exc}"
            )
            if self.ui is not None:
                self.ui.append_message("GripperForce", message)
            return

        self.args._gripper_force_dataset_contract = contract.to_dict()
        _clear_gripper_policy_cache(self.args)
        self._apply_current_gripper_task_policy()
        applied: list[str] = []
        for side, force in force_by_side.items():
            backend = _gripper_backend_for_side(self.components, side)
            if backend is None:
                continue
            try:
                backend.set_torque(float(force))
            except Exception as exc:  # noqa: BLE001 - config is persisted; report hardware apply separately.
                message = f"数据集永久力度已写入，但 {side} 夹爪即时应用失败: {type(exc).__name__}: {exc}"
                if self.ui is not None:
                    self.ui.append_message("GripperForce", message)
                continue
            applied.append(f"{side}={force:.3f}")
        message = (
            f"已将夹爪力度永久绑定到新数据集：{path}；"
            f"force_by_side={contract.force_by_side_dict}。"
            "后续 episode 必须保持该 profile，不能在同一数据集混用不同力度。"
        )
        if applied:
            message += " 已立即应用：" + ", ".join(applied) + "。"
        _log("GripperForce", message)
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

    def _apply_episode_start_payload(
        self,
        text: str,
        *,
        bind_dataset_contract: bool = False,
        start_after_commit: bool = False,
    ) -> bool:
        args_snapshot = dict(vars(self.args))
        plus_variant_snapshot = str(self.plus_curation.variant_id or "")
        side_effects_started = False
        failure_reported = False

        def restore_original_state() -> None:
            nonlocal side_effects_started
            vars(self.args).clear()
            vars(self.args).update(args_snapshot)
            self.plus_curation.variant_id = plus_variant_snapshot
            if side_effects_started:
                try:
                    self._apply_current_gripper_task_policy()
                except BaseException as exc:
                    _log(
                        "Dataset",
                        "failed to restore gripper task policy after dataset "
                        f"selection rollback: {type(exc).__name__}: {exc}",
                    )
            side_effects_started = False
            self._dataset_selection_pending = False

        def reject(message: str) -> bool:
            nonlocal failure_reported
            restore_original_state()
            failure_reported = True
            if self.ui is not None:
                self.ui.append_message("Dataset", message)
                self.ui.publish(
                    **_collection_task_ui_state(self.args),
                    active_controls=self._current_active_controls(),
                    dataset_modality_contract=CaptureModalityContract(
                        ignored_camera_keys=tuple(
                            getattr(self.args, "ignore_camera", ()) or ()
                        ),
                        ignored_tactile_keys=tuple(
                            getattr(self.args, "ignore_tactile", ()) or ()
                        ),
                    ).to_dict(),
                )
            return False

        if bool(getattr(self, "_dataset_selection_pending", False)):
            return reject(
                "已有数据集任务/profile 切换正在等待后台准备完成；"
                "本次选择未提交。"
            )
        if self._episode_busy():
            return reject(
                "数采录制、暂停、倒计时或保存中不能切换任务、数据目录或忽视模态；"
                "请先停止当前 episode。"
            )
        try:
            transactional_reconfigure = bool(
                self._dataset_reconfigure_callback is not None
                and getattr(self, "_dataset_reconfigure_transactional", False)
            )
            payload = apply_real_teleop_dataset_selection(
                self.args,
                text,
                bind_dataset_contract=(
                    bind_dataset_contract and not transactional_reconfigure
                ),
                task_definition_for_key=_task_definition_for_key,
                refresh_formal_dataset_layout=_refresh_formal_dataset_layout,
                expand_cli_path=_expand_cli_path,
                bind_capture_modality_contract=bind_capture_modality_contract,
            )
        except RealTeleopDatasetSelectionError as exc:
            return reject(str(exc))
        plus_variant_id = str(payload.get("plus_variant_id", "") or "").strip()
        if self.plus_curation.enabled and "plus_variant_id" in payload:
            try:
                if plus_variant_id:
                    profile = self.plus_curation.task_profile
                    if profile is None:
                        raise ValueError(
                            "Plus variant selection requires a task profile"
                        )
                    plus_variant_id = profile.require_variant(plus_variant_id)
            except ValueError as exc:
                return reject(f"Plus 变体选择无效：{exc}")

        selected_args_snapshot = dict(vars(self.args))

        def commit_selected_state(snapshot: Mapping[str, object]) -> None:
            nonlocal side_effects_started
            side_effects_started = True
            vars(self.args).clear()
            vars(self.args).update(dict(snapshot))
            if self.plus_curation.enabled and "plus_variant_id" in payload:
                self.plus_curation.set_variant(plus_variant_id)
            self._apply_current_gripper_task_policy()

        def publish_committed_state() -> None:
            self._dataset_selection_pending = False
            if start_after_commit:
                self._episode_start_requested = True
            if self.ui is not None:
                self.ui.publish(
                    **_collection_task_ui_state(self.args),
                    **self.plus_curation.ui_state(),
                    dataset_modality_contract={
                        "schema_version": "dexteleop.capture_modalities.v1",
                        "scope": "dataset",
                        "ignored_modalities": {
                            "camera": list(
                                getattr(self.args, "ignore_camera", ()) or ()
                            ),
                            "tactile": list(
                                getattr(self.args, "ignore_tactile", ()) or ()
                            ),
                        },
                    },
                )
                self.ui.append_message(
                    "Dataset",
                    (
                        f"数采任务已更新: {self.args.collection_task_name} "
                        f"({self.args.collection_task_id}) -> "
                        f"{self.args.dataset_root}"
                    ),
                )

        def publish_selection_failure(exc: BaseException) -> None:
            reject(
                "数据集 writer 未能按新配置重建，原配置保持不变："
                f"{type(exc).__name__}: {exc}"
            )

        if transactional_reconfigure:
            # Keep the canonical namespace and all application-side effects on
            # the old selection until the private writer candidate is ready.
            vars(self.args).clear()
            vars(self.args).update(args_snapshot)
            self._dataset_selection_pending = True
            try:
                outcome = self._dataset_reconfigure_callback(
                    selected_args_snapshot,
                    bind_dataset_contract=bind_dataset_contract,
                    on_commit=commit_selected_state,
                    on_complete=publish_committed_state,
                    on_failure=publish_selection_failure,
                )
            except Exception as exc:  # noqa: BLE001 - restore the full draft.
                if not failure_reported:
                    return reject(
                        "数据集 writer 未能按新配置重建，原配置保持不变："
                        f"{type(exc).__name__}: {exc}"
                    )
                return False
            if outcome not in {"pending", "committed"}:
                return reject(
                    "数据集 writer 重建返回了无效事务状态，原配置保持不变："
                    f"{outcome!r}"
                )
            if outcome == "pending" and self.ui is not None:
                self.ui.append_message(
                    "Dataset",
                    "正在后台准备所选数据集/profile 的目录、契约及必要的 writer；"
                    "旧配置保持有效，准备完成后再原子切换。",
                )
                self.ui.publish(
                    active_controls=self._current_active_controls(),
                    dataset_configuration_transition_active=True,
                )
            return True

        commit_selected_state(selected_args_snapshot)
        if self._dataset_reconfigure_callback is not None:
            try:
                self._dataset_reconfigure_callback()
            except Exception as exc:  # noqa: BLE001 - restore the full draft.
                return reject(
                    "数据集 writer 未能按新配置重建，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
        publish_committed_state()
        return True

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
        self._cancel_init_arm_qpos_sequence(reason=str(reason or "manual Hold"))
        self._hold_requested = True
        self._hold_active = True
        self._teleop_dispatch_confirmed = False
        self._teleop_dispatch_projection = "hold"
        self._gripper_control_enabled = False
        self._hold_request_reason = str(reason or "manual")
        self._q6_hold_recalibration_gesture.reset_for_hold()
        is_q6_gesture = self._hold_request_reason == "q6_gesture"
        if self.ui is not None:
            self.ui.publish(
                stage="Hold / 停臂保持",
                active_controls=self._current_active_controls(),
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

    def set_fix_hold_active(self, active: bool, *, message: str = "") -> None:
        normalized = bool(active)
        if normalized == self._fix_hold_active:
            return
        if normalized:
            self._gripper_control_before_fix_hold = bool(self._gripper_control_enabled)
            self._gripper_control_enabled = False
        elif not self._hold_active:
            self._gripper_control_enabled = bool(
                self._gripper_control_before_fix_hold
                and self._teleop_arm_dispatch_confirmed()
            )
        self._fix_hold_active = normalized
        if self.ui is not None:
            self.ui.publish(
                stage="Fix-Hold / 双侧特殊固定" if normalized else "遥操运行",
                fix_hold_active=normalized,
                active_controls=self._current_active_controls(),
                control_hint=message,
                gripper_control_enabled=bool(self._gripper_control_enabled),
            )
            if message:
                self.ui.append_message("FixHold", message)

    def fix_hold_active(self) -> bool:
        return bool(self._fix_hold_active)

    def consume_hold_request(self) -> bool:
        requested = bool(self._hold_requested)
        self._hold_requested = False
        return requested

    def consume_init_qpos_emergency_stop_request(self) -> bool:
        requested = bool(self._init_qpos_emergency_stop_requested)
        self._init_qpos_emergency_stop_requested = False
        return requested

    def hold_active(self) -> bool:
        return bool(self._hold_active)

    def hold_request_reason(self) -> str:
        return str(self._hold_request_reason or "manual")

    def consume_episode_start_request(self) -> bool:
        requested = bool(self._episode_start_requested)
        self._episode_start_requested = False
        requested = bool(self.terminal.consume_episode_start_request() or requested)
        if not requested:
            return False
        admitted, message = self.episode_start_admission()
        if not admitted:
            self._publish_episode_start_rejection(message)
            return False
        return True

    def consume_episode_stop_request(self) -> bool:
        requested = bool(self._episode_stop_requested)
        self._episode_stop_requested = False
        return bool(self.terminal.consume_episode_stop_request() or requested)

    def bind_dataset_reconfigure_callback(
        self,
        callback: Callable[..., object] | None,
        *,
        transactional: bool = False,
    ) -> None:
        """Bind the runtime-owned idle dataset/profile reconfiguration hook."""

        if callback is not None and not callable(callback):
            raise TypeError("dataset reconfigure callback must be callable or None")
        self._dataset_reconfigure_callback = callback
        self._dataset_reconfigure_transactional = bool(
            callback is not None and transactional
        )

    def build_progress_workflow(
        self,
        recorder: object,
    ) -> RealTeleopProgressWorkflowPort:
        """Request the Station coordinator through the injected port."""

        return self._progress_application_port().build_progress_workflow(
            recorder
        )

    def build_startup_progress_ui_payload(
        self,
        *,
        dataset_recorder: object | None,
        profile_selection_enabled: bool,
    ) -> dict[str, object]:
        return self._progress_application_port().build_startup_progress_ui_payload(
            dataset_recorder=dataset_recorder,
            profile_selection_enabled=profile_selection_enabled,
        )

    def select_managed_progress_profile(
        self,
        *,
        dataset_recorder: object | None,
        profile_id: str,
    ):
        return self._progress_application_port().select_managed_progress_profile(
            dataset_recorder=dataset_recorder,
            profile_id=profile_id,
        )

    def select_progress_entry_step(
        self,
        *,
        dataset_recorder: object | None,
        progress_step_id: str,
    ) -> str:
        return self._progress_application_port().select_progress_entry_step(
            dataset_recorder=dataset_recorder,
            progress_step_id=progress_step_id,
        )

    def publish_progress_application_state(
        self,
        *,
        dataset_recorder: object | None,
        progress_workflow: object | None,
        promotable_hold_pause: PromotableHoldPause | None,
        arm_transition_pending: bool,
    ) -> None:
        """Project lower progress state through the Station application leaf."""

        projection = (
            self._progress_application_port().project_progress_application_state(
                dataset_recorder=dataset_recorder,
                progress_workflow=progress_workflow,
                promotable_hold_pause=promotable_hold_pause,
                application_held=self.hold_active(),
                arm_transition_pending=arm_transition_pending,
                profile_selection_enabled=not self._episode_busy(),
            )
        )
        self.publish_progress_control_availability(
            configured=projection.configured,
            active_controls=projection.active_controls,
            recording_active=projection.recording_active,
            phase=projection.phase,
            application_held=projection.application_held,
            allowed_next_step_ids=projection.allowed_next_step_ids,
            blocked_reason=projection.blocked_reason,
        )
        self.publish_progress_ui_state(projection.ui_payload)

    def submit_progress_operator_intent(
        self,
        command: ProgressMutationCommand,
    ) -> bool:
        """Admit at most one mutation before rebuilt state is observed."""

        if not isinstance(command, ProgressMutationCommand):
            raise TypeError("command must be ProgressMutationCommand.")
        if self._progress_operator_intents:
            return False
        self._progress_operator_intents.append(command)
        # The HTTP admission layer removes one-shot controls immediately.
        # Force the next canonical projection to be published even when a
        # lower layer rejects the request and the canonical state is
        # unchanged, otherwise the button would remain disabled forever.
        self._progress_ui_payload = {}
        self._progress_ui_controls = ()
        if hasattr(self, "_progress_runtime_controls"):
            self._progress_runtime_controls = ()
        self._progress_ui_transport_revision = int(
            getattr(self, "_progress_ui_transport_revision", 0)
        ) + 1
        return True

    def submit_progress_ui_event(self, event: str) -> bool:
        """Translate one admitted browser event into the typed application port."""

        self._progress_operator_intent_sequence += 1
        command = self._progress_application_port().command_from_event(
            event=str(event),
            command_id=(
                "real-teleop-progress-gui:"
                f"{self._progress_operator_intent_sequence}"
            ),
        )
        if command is None:
            return False
        return self.submit_progress_operator_intent(command)

    def publish_progress_control_availability(
        self,
        *,
        configured: bool,
        active_controls: tuple[str, ...] | list[str],
        recording_active: bool,
        phase: str,
        application_held: bool,
        allowed_next_step_ids: tuple[str, ...] | list[str],
        blocked_reason: str = "",
    ) -> None:
        """Publish backend progress admission independently of GUI rendering."""

        if not isinstance(active_controls, (list, tuple)):
            raise TypeError("progress active_controls must be a list or tuple.")
        if not isinstance(allowed_next_step_ids, (list, tuple)):
            raise TypeError(
                "progress allowed_next_step_ids must be a list or tuple."
            )
        self._progress_runtime_controls = (
            tuple(str(item) for item in active_controls)
            if bool(configured)
            else ()
        )
        self._progress_runtime_state = {
            "configured": bool(configured),
            "recording_active": bool(recording_active),
            "phase": str(phase),
            "application_held": bool(application_held),
            "allowed_next_step_ids": tuple(
                str(item) for item in allowed_next_step_ids
            ),
            "blocked_reason": str(blocked_reason),
        }

    def submit_progress_foot_pad_event(self, event: str) -> bool:
        """Translate one admitted progress-pedal edge into the same typed port."""

        self._progress_operator_intent_sequence += 1
        command = self._progress_application_port().command_from_event(
            event=str(event),
            command_id=(
                "real-teleop-progress-foot-pad:"
                f"{self._progress_operator_intent_sequence}"
            ),
        )
        if command is None:
            return False
        return self.submit_progress_operator_intent(command)

    def submit_progress_profile_selection_intent(
        self,
        command: ProgressProfileSelectionCommand,
    ) -> bool:
        """Admit one configuration selection before rebuilt state is observed."""

        if not isinstance(command, ProgressProfileSelectionCommand):
            raise TypeError(
                "command must be ProgressProfileSelectionCommand."
            )
        if self._progress_profile_selection_intents:
            return False
        self._progress_profile_selection_intents.append(command)
        self._progress_ui_payload = {}
        self._progress_ui_controls = ()
        self._progress_ui_transport_revision = int(
            getattr(self, "_progress_ui_transport_revision", 0)
        ) + 1
        return True

    def submit_progress_profile_selection_ui_event(
        self,
        event: str,
    ) -> bool:
        self._progress_operator_intent_sequence += 1
        command = self._progress_application_port().profile_selection_from_event(
            event=str(event),
            command_id=(
                "real-teleop-progress-profile-gui:"
                f"{self._progress_operator_intent_sequence}"
            ),
        )
        if command is None:
            return False
        return self.submit_progress_profile_selection_intent(command)

    def submit_progress_arm_control_mode_selection_intent(
        self,
        command: ProgressArmControlModeSelectionCommand,
    ) -> bool:
        """Admit one idle-time arm-control enforcement selection."""

        if not isinstance(command, ProgressArmControlModeSelectionCommand):
            raise TypeError(
                "command must be ProgressArmControlModeSelectionCommand."
            )
        pending = getattr(
            self,
            "_progress_arm_control_mode_selection_intents",
            None,
        )
        if pending is None:
            pending = deque()
            self._progress_arm_control_mode_selection_intents = pending
        if pending:
            return False
        pending.append(command)
        # Keep the last backend-published mode authoritative until the control
        # loop consumes this selection. Clearing the payload here would create
        # a brief strict->permissive window for a second event in the same GUI
        # poll batch.
        self._progress_ui_transport_revision = int(
            getattr(self, "_progress_ui_transport_revision", 0)
        ) + 1
        return True

    def submit_progress_entry_step_selection_intent(
        self,
        command: ProgressEntryStepSelectionCommand,
    ) -> bool:
        """Admit one next-Episode coarse entry selection."""

        if not isinstance(command, ProgressEntryStepSelectionCommand):
            raise TypeError(
                "command must be ProgressEntryStepSelectionCommand."
            )
        pending = getattr(
            self,
            "_progress_entry_step_selection_intents",
            None,
        )
        if pending is None:
            pending = deque()
            self._progress_entry_step_selection_intents = pending
        if pending:
            return False
        pending.append(command)
        self._progress_ui_payload = {}
        self._progress_ui_controls = ()
        self._progress_ui_transport_revision = int(
            getattr(self, "_progress_ui_transport_revision", 0)
        ) + 1
        return True

    def submit_progress_entry_step_selection_ui_event(
        self,
        event: str,
    ) -> bool:
        self._progress_operator_intent_sequence += 1
        command = self._progress_application_port().entry_step_selection_from_event(
            event=str(event),
            command_id=(
                "real-teleop-progress-entry-step-gui:"
                f"{self._progress_operator_intent_sequence}"
            ),
        )
        if command is None:
            return False
        return self.submit_progress_entry_step_selection_intent(command)

    def submit_progress_arm_control_mode_selection_ui_event(
        self,
        event: str,
    ) -> bool:
        self._progress_operator_intent_sequence += 1
        command = (
            self._progress_application_port().arm_control_mode_selection_from_event(
                event=str(event),
                command_id=(
                    "real-teleop-progress-arm-mode-gui:"
                    f"{self._progress_operator_intent_sequence}"
                ),
            )
        )
        if command is None:
            return False
        return self.submit_progress_arm_control_mode_selection_intent(command)

    def publish_progress_ui_state(
        self,
        payload: Mapping[str, object],
    ) -> None:
        """Publish one rebuildable Control Station projection.

        The payload is display-only.  Canonical progress state and all writer
        effects remain behind the workflow/recorder ports.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("progress UI payload must be a mapping.")
        normalized = dict(payload)
        normalized["transport_revision"] = int(
            getattr(self, "_progress_ui_transport_revision", 0)
        )
        controls = normalized.get("active_controls", ())
        if not isinstance(controls, (list, tuple)):
            raise TypeError("progress UI active_controls must be a list or tuple.")
        normalized_controls = tuple(str(item) for item in controls)
        effective_controls = (
            normalized_controls
            if bool(normalized.get("enabled", False))
            else ()
        )
        if (
            normalized == getattr(self, "_progress_ui_payload", None)
            and effective_controls
            == getattr(self, "_progress_ui_controls", ())
        ):
            return
        self._progress_ui_payload = normalized
        self._progress_ui_controls = effective_controls
        if self.ui is not None:
            self.ui.publish(
                progress_ui=dict(normalized),
                active_controls=self._current_active_controls(),
            )

    def consume_progress_operator_intents(
        self,
        *,
        max_intents: int = 8,
    ) -> tuple[ProgressMutationCommand, ...]:
        pending = getattr(self, "_progress_operator_intents", None)
        if pending is None:
            return ()
        result: list[ProgressMutationCommand] = []
        for _ in range(max(1, int(max_intents))):
            if not pending:
                break
            result.append(pending.popleft())
        return tuple(result)

    def consume_progress_commands(
        self,
        *,
        max_commands: int = 8,
    ) -> tuple[ProgressMutationCommand, ...]:
        """Consume Station-admitted lower commands exactly once."""

        return self.consume_progress_operator_intents(
            max_intents=max_commands,
        )

    def consume_progress_profile_selection_intents(
        self,
    ) -> tuple[ProgressProfileSelectionCommand, ...]:
        pending = getattr(
            self,
            "_progress_profile_selection_intents",
            None,
        )
        if not pending:
            return ()
        # The browser transport and entity queue each admit one selection per
        # published state. Keep that invariant explicit at the application
        # port as well.
        return (pending.popleft(),)

    def consume_progress_arm_control_mode_selection_intents(
        self,
    ) -> tuple[ProgressArmControlModeSelectionCommand, ...]:
        pending = getattr(
            self,
            "_progress_arm_control_mode_selection_intents",
            None,
        )
        if not pending:
            return ()
        return (pending.popleft(),)

    def consume_progress_entry_step_selection_intents(
        self,
    ) -> tuple[ProgressEntryStepSelectionCommand, ...]:
        pending = getattr(
            self,
            "_progress_entry_step_selection_intents",
            None,
        )
        if not pending:
            return ()
        return (pending.popleft(),)

    def consume_episode_delete_request(self) -> int | None:
        requested = self._episode_delete_requested
        self._episode_delete_requested = None
        return requested

    def consume_camera_visualization_toggle_request(self) -> bool:
        requested = bool(self._camera_visualization_toggle_requested)
        self._camera_visualization_toggle_requested = False
        return requested

    def consume_tactile_visualization_toggle_request(self) -> bool:
        requested = bool(self._tactile_visualization_toggle_requested)
        self._tactile_visualization_toggle_requested = False
        return requested

    def consume_tactile_solving_enable_request(self) -> bool:
        requested = bool(self._tactile_solving_enable_requested)
        self._tactile_solving_enable_requested = False
        return requested

    def device_retry_admission(self) -> tuple[bool, str]:
        """Authoritative backend gate for reconstructing failed devices."""

        if self._episode_busy():
            return (
                False,
                "设备重连未执行：请先停止当前数采、倒计时或后台保存流程。",
            )
        if self._hold_requested:
            return (
                False,
                "设备重连未执行：Hold 正在生效，请等待停臂完成后重试。",
            )
        init_controller = getattr(self.components, "init_arm_controller", None)
        if bool(getattr(init_controller, "qpos_active", False)):
            return (
                False,
                "设备重连未执行：精确 Init qpos 正在规划、执行或保持；"
                "请先停止该流程并进入常规 Hold。",
            )
        if self._teleop_started and not self._hold_active:
            return (
                False,
                "设备重连未执行：运行中请先进入 Hold / 冻结。",
            )
        return True, ""

    def consume_device_retry_request(self) -> bool:
        requested = bool(self._device_retry_requested)
        self._device_retry_requested = False
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
        return bool(
            self._gripper_control_enabled
            and self._teleop_arm_dispatch_confirmed()
            and _has_q6_gripper_control(self.components)
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
            self._request_calibration_with_tracker_side_guard(
                source="q6_restart",
                now=now,
            )
            self._gripper_control_enabled = False
            if self.ui is not None and self._calibration_requested:
                self.ui.publish(
                    stage="准备重采 runtime home",
                    active_controls=["q", *self._visualization_controls()],
                    control_hint="Q6 手势已确认；即将重采 runtime home 并恢复遥操。",
                    teleop_running=False,
                    gripper_control_enabled=False,
                )
                self.ui.append_message(
                    "Q6Gesture",
                    "已检测到第二次同时握紧；准备重采 runtime home。",
                )
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
        remaining = max(0.0, float(remaining_sec))
        self._episode_countdown_ui_active = True
        self._episode_recording_ui_active = False
        self._episode_recording_ui_paused = False
        self._episode_finalizing_ui_active = False
        if self.ui is None:
            return
        self.ui.publish(
            active_controls=self._current_active_controls(),
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
            **self.plus_curation.ui_state(),
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
        self._episode_recording_ui_active = bool(active)
        self._episode_recording_ui_paused = bool(
            active and str(event_status or "") in {"paused", "resume_pending"}
        )
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = bool((event_status or "") == "finalizing")
        if episode_index is not None:
            self._episode_last_episode_index = int(episode_index)
        if frame_count is not None:
            self._episode_last_frame_count = int(frame_count)
        if self.ui is None:
            return
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
            **self.plus_curation.ui_state(),
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
        self._episode_recording_ui_active = False
        self._episode_recording_ui_paused = False
        self._episode_countdown_ui_active = False
        self._episode_finalizing_ui_active = str(status) == "finalizing"
        if episode_index is not None:
            self._episode_last_episode_index = int(episode_index)
        if frame_count is not None:
            self._episode_last_frame_count = int(frame_count)
        if self.ui is None:
            return
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
            **self.plus_curation.ui_state(),
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
        tactile_solving_active: bool | None = None,
        status: str | None = None,
    ) -> None:
        if camera_active is not None:
            self._camera_visualization_active = bool(camera_active)
        if tactile_active is not None:
            self._tactile_visualization_active = bool(tactile_active)
        if tactile_solving_active is not None:
            setattr(self.args, "tactile_solving_active", bool(tactile_solving_active))
        if self.ui is None:
            return
        self.ui.publish(
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            tactile_solving_available=bool(
                getattr(self.args, "tactile_solving_available", False)
            ),
            tactile_solving_active=bool(
                getattr(self.args, "tactile_solving_active", False)
            ),
            visualization_status="" if status is None else str(status),
            active_controls=self._current_active_controls(),
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
            # Key repeat or a pasted newline during the preparation window
            # must not survive as a second Home/episode request after capture.
            self.terminal.consume_calibration_request()
            self.terminal.consume_episode_start_request()
            self.terminal.consume_episode_stop_request()
            if self.ui is not None:
                for event in self.ui.consume_events():
                    if event == "q":
                        self.should_quit = True
            if self.should_quit:
                return False
            time.sleep(min(0.05, remaining))
        self.publish_raw_tracker_pose_state(force=True)
        if not self._final_tracker_side_home_admission():
            return False
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
                capture_duration_sec=float(
                    getattr(self.args, "calibration_timeout", 10.0)
                ),
                capture_prep_remaining_sec=0.0,
                capture_waiting_for_samples=False,
            )
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
        state_machine = getattr(
            getattr(self.components, "session", None),
            "state_machine",
            None,
        )
        session_state = getattr(state_machine, "state", None)
        session_state_value = str(
            getattr(session_state, "value", session_state) or ""
        ).upper()
        # Missing state_machine is retained as a compatibility case for thin
        # offline adapters.  The production Session always exposes ARMED here
        # and only transitions to RUNNING after its first real dispatch.
        dispatch_confirmed = bool(
            home_ready
            and (not session_state_value or session_state_value == "RUNNING")
        )
        self._teleop_dispatch_confirmed = dispatch_confirmed
        self._teleop_dispatch_projection = (
            "running" if dispatch_confirmed else f"suspended:{session_state_value or 'HOME'}"
        )
        self._gripper_control_enabled = bool(
            dispatch_confirmed and _has_q6_gripper_control(self.components)
        )
        if was_hold_active or was_hold_restart:
            self._q6_hold_recalibration_gesture.mark_runtime_restarted()
        else:
            self._q6_hold_recalibration_gesture.reset_for_running()
        if self.ui is None:
            return
        control_hint = (
            "遥操已开始；Q6 夹爪控制会自动开启；Hold 只冻结机械臂，重采需再次点击开始。"
            if dispatch_confirmed
            else "Session 已进入 ARMED；正在等待首条机械臂命令通过最终 admission，期间夹爪控制保持关闭。"
        )
        if dispatch_confirmed and self._q6_hold_recalibration_gesture.config.enabled:
            control_hint = (
                "遥操已开始；Q6 手势模式已开启：双手后三维握紧 1 秒进入 Hold，"
                "松开到释放区间 1 秒后再次握紧 1 秒会重采 runtime home。"
            )
        gripper_status = _build_gripper_status(self.components)
        self.ui.publish(
            stage=(
                "RM65 真机遥操运行中"
                if dispatch_confirmed
                else "RM65 首帧命令验证中"
            ),
            active_controls=self._current_active_controls(),
            control_hint=control_hint,
            home_ready=bool(home_ready),
            capture_stage="teleop",
            capture_phase="teleop",
            teleop_running=dispatch_confirmed,
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
            "teleop state published "
            f"session_state={session_state_value or '<compat>'} "
            f"dispatch_confirmed={dispatch_confirmed} "
            f"url={str(getattr(self.ui, 'url', '') or '<unknown>')} "
            f"gripper_control={'on' if self.gripper_control_enabled() else 'off'} "
            f"q6_sides={[side for side in self.components.active_sides if _gripper_retargeter_for_side(self.components, side) is not None]}",
        )
        self.ui.append_message(
            "RealTeleop",
            message
            or (
                "runtime home 完成，已进入真机遥操。"
                if dispatch_confirmed
                else "runtime home 完成，正在等待首条机械臂命令实际派发。"
            ),
        )
        self.publish_raw_tracker_pose_state(force=True)
        self.publish_gripper_status(force=True)

    def publish_teleop_dispatch_running(self, *, message: str = "") -> bool:
        """Project a Session-confirmed arm dispatch into application state."""

        if not self._teleop_started or self._hold_active:
            return False
        changed = not bool(getattr(self, "_teleop_dispatch_confirmed", False))
        self._teleop_dispatch_confirmed = True
        self._teleop_dispatch_projection = "running"
        self._gripper_control_enabled = _has_q6_gripper_control(self.components)
        if self.ui is None or not changed:
            return changed
        control_hint = (
            "首条机械臂命令已实际派发；遥操与 Q6 夹爪控制现在同步开放。"
        )
        self.ui.publish(
            stage="RM65 真机遥操运行中",
            active_controls=self._current_active_controls(),
            control_hint=control_hint,
            capture_stage="teleop",
            capture_phase="teleop",
            teleop_running=True,
            gripper_control_enabled=self.gripper_control_enabled(),
        )
        self.ui.append_message("RealTeleop", message or control_hint)
        self.publish_gripper_status(force=True)
        return True

    def publish_teleop_dispatch_suspended(
        self,
        *,
        session_state: str,
        reason: str = "",
    ) -> bool:
        """Remove the RUNNING projection when no arm command was dispatched."""

        if not self._teleop_started or self._hold_active:
            return False
        normalized_state = str(session_state or "UNKNOWN").upper()
        normalized_reason = str(reason or "").strip()
        projection = f"suspended:{normalized_state}:{normalized_reason}"
        changed = projection != getattr(self, "_teleop_dispatch_projection", "")
        self._teleop_dispatch_confirmed = False
        self._teleop_dispatch_projection = projection
        self._gripper_control_enabled = False
        if self.ui is None or not changed:
            return changed
        stage_by_state = {
            "ARMED": "RM65 首帧命令验证中",
            "FROZEN": "RM65 命令冻结 / 等待恢复",
            "RECOVERING": "RM65 自动恢复中",
            "FAULT": "RM65 控制故障",
            "ESTOP": "RM65 急停",
        }
        state_label = stage_by_state.get(
            normalized_state,
            "RM65 命令暂未派发",
        )
        detail = normalized_reason or (
            f"Session={normalized_state}，本控制帧没有实际派发机械臂命令；"
            "夹爪控制同步保持关闭。"
        )
        self.ui.publish(
            stage=state_label,
            active_controls=self._current_active_controls(),
            control_hint=detail,
            teleop_running=False,
            gripper_control_enabled=False,
        )
        self.ui.append_message("RealTeleop", detail)
        return True

    def publish_calibration_waiting(self, message: str = "") -> None:
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        self._tracker_side_confirmation_pending = False
        self._tracker_side_confirmation_from_hold = False
        self._tracker_side_confirmation_request_kind = "normal"
        self._tracker_side_request_pending = False
        self._tracker_side_request_from_hold = False
        self._tracker_side_request_source = ""
        self._tracker_side_request_kind = "normal"
        self._tracker_side_request_deadline = 0.0
        self._tracker_side_guard_request_active = False
        self._tracker_side_override_granted = False
        self._two_pedal_fast_home_requested = False
        self._two_pedal_fast_home_deadline = 0.0
        self._two_pedal_fast_home_timeout_sec = 1.0
        self._tracker_side_check.update(
            {
                "home_request_pending": False,
                "confirmation_required": False,
            }
        )
        if self._teleop_started:
            self._hold_active = True
            self._teleop_dispatch_confirmed = False
            self._teleop_dispatch_projection = "calibration_waiting"
            self._gripper_control_enabled = False
            self._q6_hold_recalibration_gesture.reset_for_hold()
        self._tracker_swap_locked = bool(self._teleop_started)
        if self.ui is None:
            return
        if message:
            self.ui.append_message("Calibration", message)
        self.ui.publish(
            stage="RM65 真机遥操准备" if not self._teleop_started else "Hold / 停臂保持",
            active_controls=self._current_active_controls(),
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
        # A two-pedal Hold -> single-arm transition intentionally captures
        # only its target fixed side.  Do not mark that one-sided capture
        # incomplete merely because the other active arm was not requested.
        captured_sides = tuple(
            side
            for side in self.components.active_sides
            if side in sample_counts
        )
        complete = bool(captured_sides) and all(
            int(sample_counts.get(side, 0)) >= int(min_samples)
            for side in captured_sides
        )
        if (
            not complete
            and current_time - self._last_capture_progress_publish_at < self._capture_progress_publish_interval_sec
        ):
            return
        self._last_capture_progress_publish_at = current_time
        left_samples = int(sample_counts["left"]) if "left" in sample_counts else None
        right_samples = int(sample_counts["right"]) if "right" in sample_counts else None
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
            capture_left_valid=valid_by_side.get("left") if "left" in sample_counts else None,
            capture_right_valid=valid_by_side.get("right") if "right" in sample_counts else None,
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
        if not progress.active:
            return
        side_label = "左臂" if progress.side == "left" else "右臂"
        if progress.completed:
            message = progress.message or f"{side_label} home 重采完成，已恢复遥操。"
            if self._two_pedal_transition_active:
                self._two_pedal_transition_active = False
                self._two_pedal_transition_target_fixed_side = None
                self._clear_two_pedal_alert()
        elif progress.failed:
            message = progress.message or f"{side_label} home 重采失败，保持固定。"
            if self._two_pedal_transition_active:
                self._two_pedal_transition_active = False
                self._two_pedal_transition_target_fixed_side = None
                self._publish_two_pedal_alert(
                    level="red",
                    message=f"双踏板切换失败：1 秒内未获得稳定 Home；{message}",
                    now=time.time(),
                    duration_sec=3.0,
                )
        else:
            stability = str(getattr(progress, "stability_reason", "") or "")
            message = (
                f"{side_label}保持固定并等待 tracker 稳定: {stability or 'waiting'}"
                if bool(getattr(progress, "waiting_for_stability", False))
                else f"{side_label}保持固定并重采 home: "
                f"{progress.sample_count}/{progress.min_samples} samples"
            )
        # Transition ownership is runtime state, not presentation state.  The
        # cleanup above must also run in terminal/headless compatibility mode.
        if self.ui is None:
            return
        self.ui.publish(
            **_fix_arm_ui_state(self.components),
            **_init_arm_ui_state(self.components),
            active_controls=self._current_active_controls(),
            control_hint=message,
        )
        if progress.completed or progress.failed:
            self.ui.append_message("FixArm", message)

    def publish_init_arm_home_capture_progress(self, progress, *, now: float | None = None) -> None:
        del now
        if not progress.active:
            return
        side_label = "左臂" if progress.side == "left" else "右臂"
        if progress.completed:
            message = progress.message or f"{side_label} home 重采完成，已恢复遥操。"
        elif progress.failed:
            message = progress.message or f"{side_label} home 重采失败，继续保持 init pose。"
        else:
            stability = str(getattr(progress, "stability_reason", "") or "")
            message = (
                f"{side_label}保持 init pose 并等待 tracker 稳定: {stability or 'waiting'}"
                if bool(getattr(progress, "waiting_for_stability", False))
                else f"{side_label}保持 init pose 并重采 home: "
                f"{progress.sample_count}/{progress.min_samples} samples"
            )

        if self.ui is not None:
            self.ui.publish(
                **_fix_arm_ui_state(self.components),
                **_init_arm_ui_state(self.components),
                **self._init_qpos_sequence_ui_state(),
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
        if self._hold_requested:
            message = "Hold 正在生效，暂不标定 VIVE 坐标系；请在停臂完成后重试。"
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
                active_controls=self._current_active_controls(),
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
            active_controls=self._current_active_controls(),
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
        if self._hold_requested:
            message = "Hold 正在生效，暂不加载 VIVE 坐标系；请在停臂完成后重试。"
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
            active_controls=self._current_active_controls(),
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
        if self._tracker_side_confirmation_pending:
            self._cancel_tracker_side_warning()
        elif self._tracker_side_request_pending:
            self._tracker_side_request_pending = False
            self._tracker_side_request_from_hold = False
            self._tracker_side_request_source = ""
            self._tracker_side_check.update(
                {
                    "home_request_pending": False,
                    "confirmation_required": False,
                    "confirmation_result": "cancelled_for_tracker_swap",
                }
            )
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
        current_time = time.time() if now is None else float(now)
        if not force and current_time - self._last_raw_pose_publish_at < self._raw_pose_publish_interval_sec:
            return
        if channels is None:
            source = _vive_pose_source_from_components(self.components)
            channels = {} if source is None else source.get_latest_channels()
        if not self._tracker_side_initial_check_completed:
            self._update_tracker_side_check(
                evaluate_tracker_side_assignment(
                    channels,
                    calibration=self.components.session.calibration,
                    active_sides=self.components.active_sides,
                    now=current_time,
                    post_yaw_deg=getattr(
                        self.args,
                        "virtual_desk_post_yaw_deg",
                        None,
                    ),
                )
            )
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
