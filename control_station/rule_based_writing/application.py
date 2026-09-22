"""Typed operator intents and rebuildable UI state for rule-based writing.

This application leaf translates the dedicated rule controls to the existing
RealTeleop writer and safety-loop intents.  It does not own robot dispatch,
writer persistence, or the rule cursor.
"""

from __future__ import annotations

import math
import queue
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .task_chain import RuleWritingTaskChain, RuleWritingTaskChainPhase


class RuleBasedWritingIntent(str, Enum):
    START = "rule_start"
    STOP = "rule_stop"
    DEVICE_RETRY = "device_retry"
    RUNTIME_HOME_RETRY = "runtime_home_retry"
    RELEASE_GRIPPERS = "release_grippers"
    RELEASE_RIGHT_GRIPPER = "release_right_gripper"
    CLOSE_GRIPPERS = "close_grippers"
    CLOSE_RIGHT_GRIPPER = "close_right_gripper"
    DEBUG_TOGGLE = "debug_toggle"
    QUIT = "q"


@dataclass
class RuleBasedWritingUiState:
    revision: int = 0
    phase: str = "preparing"
    title: str = "规则采集环境准备中"
    detail: str = "正在连接设备并建立安全控制链路。"
    can_start: bool = False
    can_stop: bool = False
    step_id: str | None = None
    directive_kind: str | None = None
    sequence_index: int | None = None
    stage_label: str | None = None
    stage_index: int | None = None
    stage_frame_index: int | None = None
    stage_frame_total: int | None = None
    recorded_sample_count: int = 0
    planned_sample_count: int = 0
    collection_position: str | None = None
    initial_alignment_speed_scale: float | None = None
    rule_configuration: dict[str, object] = field(default_factory=dict)
    collection_mode: str = "rule_based_writing"
    rule_configuration_batch: list[dict[str, object]] = field(default_factory=list)
    batch_index: int = 0
    batch_count: int = 1
    batch_completed_count: int = 0
    batch_cycle_complete: bool = False
    configuration_available: bool = False
    configuration_pending: bool = False
    configuration_error: str = ""
    can_configure: bool = False
    recording_ready: bool = False
    writer_blocked: bool = False
    writer_warning: str = ""
    writer_retry_pending: bool = False
    episode_event_status: str = ""
    episode_event_message: str = ""
    gripper_sides: list[str] = field(default_factory=list)
    control_topology: str = "双臂（无夹爪）"
    gripper_control_available: bool = False
    gripper_action_pending: bool = False
    gripper_action_id: int | None = None
    gripper_action_intent: str = ""
    gripper_action_result: str = "idle"
    rule_start_available: bool = False
    session_state: str = ""
    readiness_reason: str = ""
    startup_blockers: dict[str, str] = field(default_factory=dict)
    can_retry_devices: bool = False
    device_retry_pending: bool = False
    can_runtime_home: bool = False
    runtime_home_pending: bool = False
    gripper_action_status: str = ""
    can_release_grippers: bool = False
    can_close_grippers: bool = False
    debug_mode_enabled: bool = False
    debug_mode_pending: bool = False
    debug_log_path: str = ""
    can_toggle_debug: bool = False
    height_calibration_enabled: bool = False
    height_calibrated_z_m: float | None = None
    height_lower_tolerance_m: float = 0.001
    height_upper_tolerance_m: float = 0.001
    height_lower_bound_z_m: float | None = None
    height_upper_bound_z_m: float | None = None
    height_calibration_available: bool = False
    height_calibration_clear_available: bool = False
    height_calibration_pending: bool = False
    height_calibration_error: str = ""
    can_capture_height: bool = False
    can_clear_height: bool = False
    task_chain_active: bool = False
    task_chain_phase: str = "idle"
    task_chain_step: str = "0 / 5"
    task_chain_title: str = "任务链未启动"
    task_chain_detail: str = "等待右夹爪、安全门禁和轨迹配置就绪。"
    task_chain_can_advance: bool = False
    task_chain_owner_recovery_required: bool = False
    task_chain_cycle_count: int = 0
    task_chain_selected_configuration: dict[str, object] = field(default_factory=dict)
    task_chain_selected_configuration_index: int | None = None
    held: bool = False
    completed: bool = False
    aborted: bool = False
    shutdown: bool = False
    messages: list[dict[str, str]] = field(default_factory=list)


class RuleBasedWritingApplication:
    """Own the dedicated rule UI state and its typed operator-intent queues."""

    _VISIBLE_MESSAGE_TAGS = frozenset(
        {
            "RuleBased",
            "RuleBasedWriting",
            "Camera",
            "Dataset",
            "DMTAC",
            "Device",
            "Calibration",
            "CameraSelfCheck",
            "TactileCompression",
            "ArmState",
            "StartupHomeContinuity",
            "Hold",
            "Progress",
            "PositionForceGuard",
            "JointHardLimit",
            "JointHorizon",
            "TCPPoseEnvelope",
            "EmergencyStop",
            "Shutdown",
            "IKDebugLog",
        }
    )

    def __init__(
        self,
        *,
        task_chain_choice_index: Callable[[int], int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._state = RuleBasedWritingUiState()
        self._runtime_events: queue.Queue[str] = queue.Queue()
        self._rule_intents: queue.Queue[RuleBasedWritingIntent] = queue.Queue()
        self._configuration_requests: queue.Queue[dict[str, object]] = queue.Queue()
        self._height_calibration_requests: queue.Queue[dict[str, object]] = queue.Queue()
        self._task_chain = RuleWritingTaskChain()
        self._task_chain_cancel_pending = False
        self._last_gripper_action_id: int | None = None
        self._task_chain_configuration_pool: list[dict[str, object]] = []
        self._task_chain_choice_index = task_chain_choice_index or secrets.randbelow
        self._configuration_options: dict[str, object] = {}
        self._runtime_controls: frozenset[str] = frozenset()
        self._runtime_controls_seen = False
        # A failed runtime-Home attempt is a retryable safety state, not a
        # transient preparing state.  Keep this latch separate from the
        # in-flight flag so a lower runtime failure cannot strand the GUI with
        # every control disabled.
        self._runtime_home_retry_available = False
        self._runtime_home_retry_reason = ""
        self._device_retry_waiting_for_hold = False
        self._device_recovery_task_chain_step: str | None = None
        self._device_recovery_home_requested = False
        self._device_recovery_owner_close_requested = False
        self._rule_runtime_bound = False
        # A dedicated rule start owns a short pre-Episode writer-open window.
        # The lower RealTeleop admission may consult this capability while the
        # Session is still ARMED; it is cleared on stop/abort/hold and never
        # applies to the generic teleoperation UI.
        self._pre_episode_start_admission_open = False
        # Binding the extension only wires the UI to the runtime.  It is not a
        # readiness receipt: the extension publishes its first status only
        # after the shared Session/Rig admission path has been evaluated.  Keep
        # the UI in ``preparing`` until that authoritative projection arrives
        # so a transient bind cannot show a misleading ready banner with a
        # disabled start button.
        self._rule_status_seen = False
        self._completion_stop_queued = False
        # Generic runtime publications and the rule extension status are
        # independent projections.  Keep their blocker sets separate so an
        # empty readiness report cannot erase a camera/writer failure, while a
        # later non-empty readiness report can still replace an old hand fault.
        self._runtime_startup_blockers: dict[str, str] = {}
        self._rule_readiness_blockers: dict[str, str] = {}

    def _touch(self) -> None:
        self._state.revision += 1

    def _rebuild_startup_blockers_locked(self) -> None:
        merged = dict(self._runtime_startup_blockers)
        merged.update(self._rule_readiness_blockers)
        self._state.startup_blockers = merged

    def _is_expected_pre_episode_session_blocker_locked(
        self,
        *,
        key: str,
        detail: str,
    ) -> bool:
        """Recognize only the command-inert rule-start admission projection.

        The generic runtime publishes ``rig.session`` while Session is ARMED
        and no generic arm command has been dispatched yet.  A dedicated rule
        START deliberately opens the formal writer before its initial-alignment
        planner emits that first command, so this one diagnostic is expected
        during the bounded pre-Episode window.  Device/backend faults and all
        other Session states remain real blockers.
        """

        rule_start_window_open = bool(
            self._pre_episode_start_admission_open
            or (
                self._task_chain.phase is RuleWritingTaskChainPhase.READY_RULE
                and self._task_chain.rule_start_waiting
            )
        )
        if not (
            rule_start_window_open
            and str(key) == "rig.session"
            and str(self._state.session_state or "").upper()
            in {"", "ARMED", "RUNNING"}
            and not self._state.held
            and not self._state.completed
            and not self._state.aborted
            and not self._state.shutdown
        ):
            return False
        normalized = " ".join(str(detail or "").lower().split())
        return bool(
            "runtime home" in normalized
            and "hardware arm admission" in normalized
            and (
                "no command" in normalized
                or "not dispatched" in normalized
                or "dispatch" in normalized
                or "before rule collection" in normalized
            )
        )

    def _is_expected_pre_episode_lifecycle_blocker_locked(
        self,
        *,
        key: str,
        detail: str,
    ) -> bool:
        """Recognize only the writer-settling transition owned by this START.

        The rule start intentionally prepares the formal writer before the
        first rule arm command can be dispatched.  During that bounded
        command-inert interval the rule runtime reports its collection
        control-plane transition as ``dataset.lifecycle``.  This is not a
        hardware/readiness fault and must not revoke the already accepted
        pre-Episode admission.  The exact settling diagnostic is filtered;
        active/awaiting-discard lifecycle states remain blocking.
        """

        rule_start_window_open = bool(
            self._pre_episode_start_admission_open
            or (
                self._task_chain.phase is RuleWritingTaskChainPhase.READY_RULE
                and self._task_chain.rule_start_waiting
            )
        )
        if not (
            rule_start_window_open
            and str(key) == "dataset.lifecycle"
            and str(self._state.session_state or "").upper()
            in {"", "ARMED", "RUNNING"}
            and not self._state.held
            and not self._state.completed
            and not self._state.aborted
            and not self._state.shutdown
        ):
            return False
        normalized = " ".join(str(detail or "").lower().split())
        return bool(
            normalized.startswith(
                "cancelled rule episode lifecycle is still settling"
            )
            and "awaiting discard" not in normalized
        )

    def _effective_startup_blockers_locked(self) -> dict[str, str]:
        """Return blockers after the narrow accepted-rule-start exceptions."""

        blockers = dict(self._state.startup_blockers)
        for key, detail in tuple(blockers.items()):
            if (
                self._is_expected_pre_episode_session_blocker_locked(
                    key=key,
                    detail=detail,
                )
                or self._is_expected_pre_episode_lifecycle_blocker_locked(
                    key=key,
                    detail=detail,
                )
                or self._is_expected_task_chain_gripper_blocker_locked(key=key)
            ):
                blockers.pop(key, None)
        return blockers

    def _is_expected_task_chain_gripper_blocker_locked(self, *, key: str) -> bool:
        """Keep the pre-rule gripper steps available while owner is absent.

        The writing runtime correctly blocks rule START until the one-shot
        closed right-gripper owner exists.  The task chain itself must still
        be able to perform its first release and close steps, and its final
        release must be allowed to end a cycle.  This exception is limited to
        a configured task-chain pool and never applies in the READY_RULE or
        active-rule phases.
        """

        return bool(
            str(key) == "rule.right_hand_hold_owner"
            and self._task_chain_configuration_pool
            and self._task_chain.phase
            in {
                RuleWritingTaskChainPhase.IDLE,
                RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING,
                RuleWritingTaskChainPhase.READY_CLOSE,
                RuleWritingTaskChainPhase.CLOSE_PENDING,
                RuleWritingTaskChainPhase.READY_HEIGHT,
                RuleWritingTaskChainPhase.HEIGHT_PENDING,
                RuleWritingTaskChainPhase.READY_FINAL_RELEASE,
                RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING,
                RuleWritingTaskChainPhase.COMPLETE,
            }
        )

    def _rule_owner_recovery_window_locked(self) -> bool:
        """Return whether READY_RULE may expose a safe gripper retry."""

        blockers = self._effective_startup_blockers_locked()
        return bool(
            self._task_chain.phase is RuleWritingTaskChainPhase.READY_RULE
            and set(blockers) == {"rule.right_hand_hold_owner"}
            and not self._state.held
            and not self._state.completed
            and not self._state.aborted
            and not self._state.shutdown
        )

    def observe_runtime_publication(self, updates: Mapping[str, object]) -> None:
        """Consume the generic projection without exposing its controls or progress."""

        with self._lock:
            device_retry_hold_rejected = False
            runtime_home_was_pending = bool(self._state.runtime_home_pending)
            capture_stage = str(updates.get("capture_stage", "") or "")
            capture_phase = str(updates.get("capture_phase", "") or "").lower()
            if bool(updates.get("home_ready", False)):
                self._state.runtime_home_pending = False
                self._runtime_home_retry_available = False
                self._runtime_home_retry_reason = ""
            elif capture_stage == "runtime_home":
                if capture_phase in {"preparing", "capturing"}:
                    self._state.runtime_home_pending = True
                    self._runtime_home_retry_available = False
                elif capture_phase in {"ready", "idle", "failed"}:
                    runtime_home_failed = bool(
                        capture_phase == "failed"
                        or bool(updates.get("runtime_home_failure", False))
                    )
                    # A normal Hold projection also uses runtime_home/ready.
                    # Keep an explicit retry pending until the lower runtime
                    # publishes either a Home failure or home_ready=True.
                    if runtime_home_was_pending and not runtime_home_failed:
                        self._state.runtime_home_pending = True
                    else:
                        self._state.runtime_home_pending = False
                    if runtime_home_was_pending and runtime_home_failed:
                        self._runtime_home_retry_available = True
                        self._runtime_home_retry_reason = str(
                            updates.get("control_hint")
                            or updates.get("readiness_reason")
                            or "上一次安全初始化未完成；请确认机械臂处于安全位置后重试。"
                        )
            if "active_controls" in updates:
                controls = updates.get("active_controls") or ()
                self._runtime_controls = frozenset(str(item) for item in controls)
                self._runtime_controls_seen = True
                self._advance_device_retry_transaction_locked()
            if "episode_event_status" in updates:
                self._state.episode_event_status = str(
                    updates.get("episode_event_status", "") or ""
                )
            if "episode_event_message" in updates:
                self._state.episode_event_message = str(
                    updates.get("episode_event_message", "") or ""
                )
            if (
                self._state.device_retry_pending
                and self._device_retry_waiting_for_hold
                and updates.get("hardware_hold_confirmed") is False
            ):
                self._state.device_retry_pending = False
                self._device_retry_waiting_for_hold = False
                self._state.phase = "blocked"
                self._state.title = "安全停臂未确认"
                self._state.detail = str(
                    updates.get("control_hint")
                    or "底层 Hold 或位置命令流释放未获完整确认；设备重连未执行。"
                )
                device_retry_hold_rejected = True
            if "startup_blockers" in updates:
                raw_blockers = updates.get("startup_blockers")
                self._runtime_startup_blockers = (
                    {str(key): str(value) for key, value in raw_blockers.items()}
                    if isinstance(raw_blockers, Mapping)
                    else {}
                )
                self._rebuild_startup_blockers_locked()
            if "debug_ik_logging" in updates:
                self._state.debug_mode_enabled = bool(
                    updates.get("debug_ik_logging", False)
                )
                self._state.debug_mode_pending = False
            if "ik_debug_log_path" in updates:
                self._state.debug_log_path = str(
                    updates.get("ik_debug_log_path", "") or ""
                )
            if device_retry_hold_rejected:
                self._rebuild_controls()
                self._touch()
                return
            if self._state.device_retry_pending:
                self._project_device_retry_transaction_locked()
            elif self._state.runtime_home_pending:
                self._set_runtime_home_pending_locked()
            elif self._effective_startup_blockers_locked():
                self._set_readiness_blocked_locked()
            elif self._state.phase == "preparing" and self._rule_runtime_bound:
                self._project_prestart_readiness_locked()
            self._maybe_continue_device_recovery_locked()
            self._try_start_waiting_task_chain_locked()
            self._rebuild_controls()
            self._touch()

    def bind_rule_runtime(self) -> None:
        with self._lock:
            self._rule_runtime_bound = True
            self._project_prestart_readiness_locked()
            self._rebuild_controls()
            self._touch()

    def allows_pre_episode_armed_warmup(self) -> bool:
        """Admit only this UI's already-accepted rule start before dispatch.

        The normal RealTeleop start gate remains dispatch-based.  Rule-based
        collection has one bounded exception: its formal writer must open
        before the first rule command can be produced, while the Session is
        still ARMED.  This method exposes that application-owned lifecycle
        fact without making the lower runtime infer it from a stale phase.
        """

        with self._lock:
            return bool(
                self._pre_episode_start_admission_open
                and self._rule_runtime_bound
                # The writer-preparation callback and the rule-runtime status
                # heartbeat are asynchronous.  The accepted START latch is
                # the authoritative fact here; requiring the UI phase to be
                # exactly ``armed`` lets a stale/preparing projection cancel
                # a command-inert writer countdown before the rule runtime
                # has a chance to dispatch its first arm command.
                and str(self._state.session_state or "").upper()
                in {"", "ARMED", "RUNNING"}
                and not self._effective_startup_blockers_locked()
                and not self._state.held
                and not self._state.completed
                and not self._state.aborted
                and not self._state.shutdown
                and not self._state.configuration_pending
                and not self._state.writer_retry_pending
                and not self._state.gripper_action_pending
                and not self._state.height_calibration_pending
            )

    def pre_episode_start_admission_reason(self) -> str | None:
        """Explain why the rule-owned pre-Episode window is not admissible.

        The lower RealTeleop gate still owns the actual admission decision.
        This small diagnostic seam only preserves the application-owned
        blocker when a stale/non-dispatch heartbeat closes the window; without
        it the lower layer can only report its generic arm-dispatch message.
        """

        with self._lock:
            effective_blockers = self._effective_startup_blockers_locked()
            if effective_blockers:
                detail_lines = "; ".join(
                    f"{key}: {value}"
                    for key, value in sorted(effective_blockers.items())
                )
                return (
                    "规则采集启动门禁已关闭；请先完成设备恢复和安全初始化。\n"
                    + detail_lines
                )
            if self._state.runtime_home_pending:
                return "正在重新确认 runtime Home；完成前不能开始规则采集。"
            if self._state.held:
                return "机械臂处于 Hold；请重新完成 runtime Home 后再开始规则采集。"
            if self._state.gripper_action_pending:
                return "夹爪动作仍在执行；收到完成回执前不能开始规则采集。"
            if self._state.height_calibration_pending:
                return "高度标定仍在执行；完成后才能开始规则采集。"
            if self._state.detail and self._state.phase in {
                "blocked",
                "preparing",
                "configuring",
                "stopping",
            }:
                return str(self._state.detail)
            return None

    def submit_intent(self, intent: RuleBasedWritingIntent) -> bool:
        if not isinstance(intent, RuleBasedWritingIntent):
            raise TypeError("intent must be RuleBasedWritingIntent.")
        with self._lock:
            if intent is RuleBasedWritingIntent.STOP and self._task_chain.active:
                if self._task_chain_cancel_pending:
                    return False
                self._task_chain_cancel_pending = True
                self._clear_device_recovery_locked()
                self._pre_episode_start_admission_open = False
                self._rule_intents.put(intent)
                self._runtime_events.put("episode_stop:rule_based_hold_cancel")
                self._append_message_locked(
                    "RuleBased",
                    "已收到“停止并 Hold”操作请求；正在取消当前任务链。",
                )
                self._state.phase = "stopping"
                self._state.title = "正在取消任务链并进入 Hold"
                self._state.detail = (
                    "已停止接纳任务链动作；等待底层 Hold 回执后释放任务链控件。"
                )
                self._rebuild_controls()
                self._touch()
                return True
            if self._task_chain.active and intent in {
                RuleBasedWritingIntent.START,
                RuleBasedWritingIntent.RELEASE_GRIPPERS,
                RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER,
                RuleBasedWritingIntent.CLOSE_GRIPPERS,
                RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER,
            }:
                return False
            return self._submit_intent_locked(intent)

    def _submit_intent_locked(
        self,
        intent: RuleBasedWritingIntent,
    ) -> bool:
        available = self._base_control_availability_locked()
        if intent is RuleBasedWritingIntent.START:
            if not available["start"]:
                return False
            retry_writer_only = bool(self._state.writer_blocked)
            if not retry_writer_only:
                self._completion_stop_queued = False
                self._rule_intents.put(intent)
                self._pre_episode_start_admission_open = True
                self._state.episode_event_status = ""
                self._state.episode_event_message = ""
            else:
                self._pre_episode_start_admission_open = False
            self._runtime_events.put("episode_start")
            self._state.phase = "armed"
            self._state.writer_retry_pending = retry_writer_only
            self._state.title = (
                "正在重试正式 Writer" if retry_writer_only else "规则采集启动中"
            )
            self._state.detail = (
                "正在按当前 GUI 配置重新准备 writer；就绪前不会输出规则运动命令。"
                if retry_writer_only
                else "正在准备 writer；就绪前不会输出规则运动命令。"
            )
        elif intent is RuleBasedWritingIntent.STOP:
            if not available["stop"]:
                return False
            self._pre_episode_start_admission_open = False
            self._rule_intents.put(intent)
            self._runtime_events.put("episode_stop:rule_based_hold_cancel")
            self._append_message_locked(
                "RuleBased",
                "已收到“停止并 Hold”操作请求；正在取消当前规则采集。",
            )
            self._state.phase = "stopping"
            self._state.title = "规则采集停止中"
            self._state.detail = "正在进入 Hold 并取消本轮规则采集。"
        elif intent is RuleBasedWritingIntent.DEVICE_RETRY:
            if not available["retry_devices"]:
                return False
            if self._device_recovery_task_chain_step is None:
                self._device_recovery_task_chain_step = (
                    self._task_chain.prepare_device_recovery()
                )
                self._device_recovery_home_requested = False
                self._device_recovery_owner_close_requested = False
            self._state.device_retry_pending = True
            self._state.phase = "preparing"
            if self._device_retry_directly_admitted_locked():
                self._device_retry_waiting_for_hold = False
                self._runtime_events.put("device_retry")
                self._set_device_retry_running_locked()
            else:
                self._device_retry_waiting_for_hold = True
                self._runtime_events.put("h")
                self._state.title = "正在安全停臂后重连"
                self._state.detail = (
                    "已停止接纳规则命令；正在等待底层 Hold 与位置命令流释放回执，"
                    "确认后将自动重连失败设备。"
                )
        elif intent is RuleBasedWritingIntent.RUNTIME_HOME_RETRY:
            if not available["runtime_home"]:
                return False
            self._runtime_home_retry_available = False
            self._runtime_home_retry_reason = ""
            self._runtime_events.put("runtime_home_retry")
            self._state.runtime_home_pending = True
            self._state.phase = "preparing"
            self._state.title = "正在重新确认安全初始化"
            self._state.detail = (
                "请保持机械臂处于安全位置；正在重新采集 runtime Home，"
                "完成前不会开始规则运动。"
            )
        elif intent in {
            RuleBasedWritingIntent.RELEASE_GRIPPERS,
            RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER,
            RuleBasedWritingIntent.CLOSE_GRIPPERS,
            RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER,
        }:
            control = (
                "release_grippers"
                if intent
                in {
                    RuleBasedWritingIntent.RELEASE_GRIPPERS,
                    RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER,
                }
                else (
                    "close_right_gripper"
                    if intent is RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
                    else "close_grippers"
                )
            )
            if not available[control]:
                return False
            self._rule_intents.put(intent)
            self._state.gripper_action_pending = True
            self._state.gripper_action_id = None
            self._state.gripper_action_intent = intent.value
            self._state.gripper_action_result = "requested"
            self._state.gripper_action_status = (
                "正在松开夹爪…"
                if intent is RuleBasedWritingIntent.RELEASE_GRIPPERS
                else (
                    "正在松开右夹爪…"
                    if intent is RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER
                    else (
                        "正在闭紧右夹爪…"
                        if intent is RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
                        else "正在闭紧夹爪…"
                    )
                )
            )
            self._state.detail = self._state.gripper_action_status
        elif intent is RuleBasedWritingIntent.DEBUG_TOGGLE:
            if not available["toggle_debug"]:
                return False
            # Reuse the existing asynchronous real-control diagnostic recorder.
            self._runtime_events.put("d")
            self._state.debug_mode_pending = True
        else:
            self._runtime_events.put("q")
            self._state.phase = "stopping"
            self._state.title = "程序退出中"
            self._state.detail = "正在停止机械臂并关闭运行资源。"
        self._rebuild_controls()
        self._touch()
        return True

    def configure_rule_options(
        self,
        options: Mapping[str, object],
        selected: Mapping[str, object],
    ) -> None:
        """Publish immutable between-attempt choices owned by the rule extension."""

        normalized_options = dict(options)
        normalized_selected = self._validate_configuration(
            selected,
            options=normalized_options,
        )
        with self._lock:
            self._configuration_options = normalized_options
            self._state.collection_mode = str(
                normalized_options.get("collection_mode")
                or self._state.collection_mode
            )
            self._state.rule_configuration = normalized_selected
            self._task_chain_configuration_pool = [dict(normalized_selected)]
            self._state.configuration_error = ""
            self._rebuild_controls()
            self._touch()

    def update_rule_runtime_info(self, values: Mapping[str, object]) -> None:
        """Refresh mutable dataset facts without replacing operator choices."""

        normalized = dict(values)
        with self._lock:
            options = dict(self._configuration_options)
            runtime_info = dict(options.get("runtime_info", {}) or {})
            runtime_info.update(normalized)
            options["runtime_info"] = runtime_info
            self._configuration_options = options
            self._touch()

    def submit_configuration(self, payload: Mapping[str, object]) -> bool:
        """Queue one validated configuration batch while collection is idle."""

        with self._lock:
            if self._task_chain.active:
                return False
            try:
                configuration = self._validate_configuration_request(
                    payload,
                    options=self._configuration_options,
                )
            except (TypeError, ValueError) as exc:
                self._state.configuration_error = str(exc)
                self._touch()
                return False
            return self._submit_configuration_locked(configuration)

    def _submit_configuration_locked(
        self,
        configuration: Mapping[str, object],
    ) -> bool:
        if (
            not self._base_control_availability_locked()["configure"]
            or self._state.configuration_pending
        ):
            return False
        self._configuration_requests.put(dict(configuration))
        self._state.configuration_pending = True
        self._state.configuration_error = ""
        self._state.phase = "configuring"
        self._state.title = "正在更新规则组合"
        self._state.detail = "正在后台读取并验证所选来源；完成前不会输出运动命令。"
        self._rebuild_controls()
        self._touch()
        return True

    def complete_configuration(
        self,
        *,
        selected: Mapping[str, object] | None,
        error: str | None,
    ) -> None:
        with self._lock:
            chain_configuration = (
                self._task_chain.phase
                is RuleWritingTaskChainPhase.CONFIGURATION_PENDING
            )
            self._state.configuration_pending = False
            if error:
                self._state.configuration_error = str(error)
                self._state.title = "规则组合更新失败"
                self._state.detail = str(error)
            else:
                if selected is not None:
                    raw_batch = selected.get("batch")
                    if isinstance(raw_batch, (list, tuple)) and raw_batch:
                        batch = [dict(item) for item in raw_batch]
                        self._state.rule_configuration_batch = batch
                        self._state.rule_configuration = dict(batch[0])
                    else:
                        self._state.rule_configuration_batch = [dict(selected)]
                        self._state.rule_configuration = dict(selected)
                    if not chain_configuration:
                        self._task_chain_configuration_pool = [
                            dict(item)
                            for item in self._state.rule_configuration_batch
                        ]
                self._state.configuration_error = ""
                if self._state.startup_blockers:
                    self._set_readiness_blocked_locked()
                elif self._state.writer_blocked:
                    self._state.title = "正式 Writer 已阻塞"
                    self._state.detail = (
                        "参数已更新并通过验证；点击“按当前配置重试 Writer”。"
                    )
                else:
                    self._project_prestart_readiness_locked()
            if self._state.writer_blocked:
                self._state.phase = "blocked"
            elif not self._state.startup_blockers and self._state.phase not in {
                "armed",
                "running",
                "completed",
                "aborted",
            }:
                self._project_prestart_readiness_locked()
            self._rebuild_controls()
            if chain_configuration:
                if error:
                    self._task_chain.fail_rule_preparation(str(error))
                elif selected is None:
                    self._task_chain.fail_rule_preparation(
                        "配置接口未返回已激活的配置。"
                    )
                elif self._submit_intent_locked(RuleBasedWritingIntent.START):
                    self._task_chain.begin_rule()
                else:
                    # Configuration completion and the lower Session/Rig
                    # admission receipt are asynchronous.  Keep the validated
                    # choice and wait for an explicit operator retry instead
                    # of treating a stale start receipt as a config failure or
                    # bypassing the safety gate.
                    self._task_chain.wait_for_rule_start()
                self._rebuild_controls()
            self._touch()

    def submit_height_calibration(self, payload: Mapping[str, object]) -> bool:
        """Queue one command-inert right-TCP height maintenance request."""

        with self._lock:
            if self._task_chain.active:
                return False
            request = self._height_calibration_request_locked(payload)
            if request is None:
                return False
            return self._submit_height_calibration_locked(request)

    def _height_calibration_request_locked(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, object] | None:
        operation = str(payload.get("operation") or "").strip().lower()
        available = self._base_control_availability_locked()
        if operation == "capture":
            if not available["capture_height"]:
                return None
            try:
                lower_mm = float(payload.get("lower_tolerance_mm", 1.0))
                upper_mm = float(payload.get("upper_tolerance_mm", 1.0))
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(lower_mm)
                or not math.isfinite(upper_mm)
                or not 0.1 <= lower_mm <= 20.0
                or not 0.1 <= upper_mm <= 20.0
            ):
                return None
            return {
                "operation": "capture",
                "lower_tolerance_m": lower_mm / 1000.0,
                "upper_tolerance_m": upper_mm / 1000.0,
            }
        if operation == "clear" and available["clear_height"]:
            return {"operation": "clear"}
        return None

    def _submit_height_calibration_locked(
        self,
        request: Mapping[str, object],
    ) -> bool:
        if self._state.height_calibration_pending:
            return False
        self._height_calibration_requests.put(dict(request))
        self._state.height_calibration_pending = True
        self._state.height_calibration_error = ""
        self._rebuild_controls()
        self._touch()
        return True

    def complete_height_calibration(self, *, error: str | None) -> None:
        with self._lock:
            chain_height = (
                self._task_chain.phase is RuleWritingTaskChainPhase.HEIGHT_PENDING
            )
            self._state.height_calibration_pending = False
            self._state.height_calibration_error = str(error or "")
            if chain_height:
                self._task_chain.finish_height(error=error)
            self._rebuild_controls()
            self._touch()

    def submit_task_chain_advance(self, payload: Mapping[str, object]) -> bool:
        """Advance exactly one externally visible task-chain phase.

        The rule Episode is the only compound phase: after this method chooses
        and submits one prevalidated configuration, the existing asynchronous
        configuration callback starts the existing complete rule lifecycle.
        """

        with self._lock:
            if str(payload.get("event") or "advance") != "advance":
                return False
            phase = self._task_chain.phase
            available = self._base_control_availability_locked()
            if phase in {
                RuleWritingTaskChainPhase.IDLE,
                RuleWritingTaskChainPhase.COMPLETE,
            }:
                if not self._task_chain_initially_available_locked(available):
                    return False
                action_id_baseline = self._last_gripper_action_id
                if not self._submit_intent_locked(
                    RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER
                ):
                    return False
                self._task_chain.begin_initial_release(
                    action_id_baseline=action_id_baseline
                )
            elif phase is RuleWritingTaskChainPhase.READY_CLOSE:
                action_id_baseline = self._last_gripper_action_id
                if not self._submit_intent_locked(
                    RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
                ):
                    return False
                self._task_chain.begin_close(
                    action_id_baseline=action_id_baseline
                )
            elif phase is RuleWritingTaskChainPhase.READY_HEIGHT:
                request = self._height_calibration_request_locked(
                    {
                        "operation": "capture",
                        "lower_tolerance_mm": payload.get("lower_tolerance_mm", 1.0),
                        "upper_tolerance_mm": payload.get("upper_tolerance_mm", 1.0),
                    }
                )
                if request is None or not self._submit_height_calibration_locked(request):
                    return False
                self._task_chain.begin_height()
            elif phase is RuleWritingTaskChainPhase.READY_RULE:
                if self._rule_owner_recovery_window_locked():
                    # A serial reconnect correctly invalidates the previous
                    # close receipt.  Re-establish that receipt through the
                    # same right-only, one-shot task-chain command instead of
                    # exposing the generic continuous-close maintenance path.
                    if not self._submit_intent_locked(
                        RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
                    ):
                        return False
                elif self._task_chain.rule_start_waiting:
                    if not self._submit_intent_locked(RuleBasedWritingIntent.START):
                        return False
                    self._task_chain.begin_rule_after_wait()
                else:
                    if not available["configure"] or not self._task_chain_configuration_pool:
                        return False
                    count = len(self._task_chain_configuration_pool)
                    index = int(self._task_chain_choice_index(count))
                    if not 0 <= index < count:
                        raise ValueError("task_chain_choice_index returned an invalid index.")
                    selected = dict(self._task_chain_configuration_pool[index])
                    # Revalidate at the same trust boundary used by manual GUI
                    # configuration; the chain never promotes cached UI state.
                    selected = self._validate_configuration_request(
                        selected,
                        options=self._configuration_options,
                    )
                    if not self._submit_configuration_locked(selected):
                        return False
                    self._task_chain.begin_configuration(index=index, selected=selected)
            elif phase is RuleWritingTaskChainPhase.READY_FINAL_RELEASE:
                action_id_baseline = self._last_gripper_action_id
                if not self._submit_intent_locked(
                    RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER
                ):
                    return False
                self._task_chain.begin_final_release(
                    action_id_baseline=action_id_baseline
                )
            else:
                return False
            self._rebuild_controls()
            self._touch()
            return True

    @staticmethod
    def _option_values(options: Mapping[str, object], key: str) -> set[str]:
        raw = options.get(key, ())
        if not isinstance(raw, (list, tuple)):
            return set()
        values: set[str] = set()
        for item in raw:
            value = item.get("value") if isinstance(item, Mapping) else item
            if value is not None:
                values.add(str(value))
        return values

    @classmethod
    def _validate_configuration_request(
        cls,
        payload: Mapping[str, object],
        *,
        options: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(payload, Mapping):
            raise TypeError("规则组合必须是对象。")
        if "batch" not in payload:
            return cls._validate_configuration(payload, options=options)
        if set(payload) != {"batch"}:
            raise ValueError("批次请求只能包含 batch 字段。")
        raw_batch = payload.get("batch")
        if not isinstance(raw_batch, (list, tuple)) or not raw_batch:
            raise ValueError("采集批次至少需要一个组合。")
        if len(raw_batch) > 128:
            raise ValueError("单个采集批次最多允许 128 个组合。")
        return {
            "batch": [
                cls._validate_configuration(item, options=options)
                for item in raw_batch
            ]
        }

    @classmethod
    def _validate_configuration(
        cls,
        payload: Mapping[str, object],
        *,
        options: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(payload, Mapping):
            raise TypeError("规则组合必须是对象。")
        configured_core = options.get("configuration_core_fields")
        if configured_core is None:
            core_required = {
                "pen_down_candidate_id",
                "write_candidate_id",
                "pen_up_candidate_id",
                "trajectory_id",
                "direction",
                "writing_size_m",
                "initial_alignment_speed_scale",
            }
        elif not isinstance(configured_core, (list, tuple)):
            raise ValueError("核心配置字段定义尚未就绪。")
        else:
            core_required = {str(key) for key in configured_core}
        extra_fields = options.get("runtime_configuration_fields", ())
        if not isinstance(extra_fields, (list, tuple)):
            raise ValueError("运行参数字段定义尚未就绪。")
        required = core_required | {str(key) for key in extra_fields}
        unknown = sorted(set(payload) - required)
        missing = sorted(required - set(payload))
        if unknown or missing:
            raise ValueError(f"规则组合字段无效：missing={missing}, unknown={unknown}。")
        result: dict[str, object] = {}
        source_options = options.get("source_candidates_by_phase", {})
        if not isinstance(source_options, Mapping):
            raise ValueError("规则来源选项尚未就绪。")
        for phase in ("pen_down", "write", "pen_up"):
            key = f"{phase}_candidate_id"
            if key not in required:
                continue
            value = str(payload[key])
            allowed = cls._option_values(source_options, phase)
            if value not in allowed:
                raise ValueError(f"{phase} 来源不在当前已验证范围内。")
            result[key] = value
        for key, option_key in (
            ("trajectory_id", "trajectory_ids"),
            ("direction", "directions"),
        ):
            if key not in required:
                continue
            value = str(payload[key])
            if value not in cls._option_values(options, option_key):
                raise ValueError(f"{key} 不是可用选项。")
            result[key] = value
        bounds = options.get("numeric_bounds", {})
        if not isinstance(bounds, Mapping):
            raise ValueError("规则数值范围尚未就绪。")
        integer_fields = {str(key) for key in options.get("integer_fields", ())}
        numeric_fields = tuple(
            key for key in required if key in bounds
        )
        for key in numeric_fields:
            try:
                value = float(payload[key])
                lower, upper = bounds[key]
                lower = float(lower)
                upper = float(upper)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{key} 必须是有效数值。") from exc
            if not lower <= value <= upper:
                raise ValueError(f"{key} 必须位于 [{lower}, {upper}]。")
            if key in integer_fields:
                if not value.is_integer():
                    raise ValueError(f"{key} 必须是整数。")
                result[key] = int(value)
            else:
                result[key] = value
        choice_fields = options.get("choice_fields", {})
        if not isinstance(choice_fields, Mapping):
            raise ValueError("运行选项字段定义尚未就绪。")
        for key, option_key in choice_fields.items():
            key = str(key)
            if key not in required:
                continue
            value = str(payload[key])
            if value not in cls._option_values(options, str(option_key)):
                raise ValueError(f"{key} 不是可用选项。")
            result[key] = value
        select_fields = options.get("select_fields", {})
        if not isinstance(select_fields, Mapping):
            raise ValueError("运行枚举参数尚未就绪。")
        for key, values in select_fields.items():
            key = str(key)
            if key not in required:
                continue
            value = str(payload[key])
            allowed = cls._option_values({key: values}, key)
            if value not in allowed:
                raise ValueError(f"{key} 不是可用选项。")
            result[key] = value
        boolean_fields = {str(key) for key in options.get("boolean_fields", ())}
        for key in boolean_fields:
            if key not in required or not isinstance(payload[key], bool):
                raise ValueError(f"{key} 必须是布尔值。")
            result[key] = bool(payload[key])
        text_fields = {str(key) for key in options.get("text_fields", ())}
        for key in text_fields:
            if key not in required:
                continue
            value = str(payload[key]).strip()
            if "\x00" in value or len(value) > 4096:
                raise ValueError(f"{key} 包含无效路径文本。")
            result[key] = value
        unparsed = sorted(required - set(result))
        if unparsed:
            raise ValueError(f"配置字段缺少类型定义：{unparsed}。")
        return result

    def publish_writer_prepare_result(self, *, error: str | None) -> None:
        """Project retryable writer preparation without ending the GUI runtime."""

        with self._lock:
            self._state.writer_retry_pending = False
            if error:
                self._state.writer_blocked = True
                self._state.writer_warning = str(error)
                self._state.recording_ready = False
                self._state.phase = "blocked"
                self._state.title = "正式 Writer 已阻塞"
                self._state.detail = str(error)
            else:
                self._state.writer_blocked = False
                self._state.writer_warning = ""
                if self._state.startup_blockers:
                    self._set_readiness_blocked_locked()
                elif self._state.phase == "blocked":
                    self._state.phase = "armed"
                    self._state.title = "正式 Writer 已就绪"
                    self._state.detail = "正在进入采集倒计时；规则运动尚未开始。"
            self._rebuild_controls()
            self._touch()

    def complete_device_retry(self, *, error: str | None) -> None:
        """Complete the rule UI's bounded Hold-then-retry transaction."""

        with self._lock:
            if not self._state.device_retry_pending:
                return
            self._state.device_retry_pending = False
            self._device_retry_waiting_for_hold = False
            if error:
                self._state.phase = "blocked"
                self._state.title = "设备重连未执行"
                self._state.detail = str(error)
            else:
                if self._device_recovery_task_chain_step is None:
                    self._device_recovery_task_chain_step = (
                        self._task_chain.prepare_device_recovery()
                    )
                self._maybe_continue_device_recovery_locked()
                if self._state.runtime_home_pending:
                    self._set_runtime_home_pending_locked()
                elif self._state.startup_blockers:
                    self._set_readiness_blocked_locked()
                elif self._device_recovery_task_chain_step is None:
                    self._project_prestart_readiness_locked()
            self._rebuild_controls()
            self._touch()

    def observe_rule_status(self, status: Mapping[str, object]) -> None:
        with self._lock:
            self._rule_status_seen = True
            armed = bool(status.get("armed", False))
            begun = bool(status.get("execution_begun", False))
            completed = bool(status.get("completed", False))
            aborted = bool(status.get("aborted", False))
            held = bool(status.get("held", False))
            waiting = str(status.get("waiting_reason", "") or "")
            if "session_state" in status:
                self._state.session_state = str(status.get("session_state") or "").upper()
                # A rule-status heartbeat can lag the generic Home
                # publication.  Do not let that stale session label clear an
                # in-flight safety initialization; only the generic
                # ``home_ready`` receipt may do that.
                if (
                    self._state.session_state in {"ARMED", "RUNNING"}
                    and not self._state.runtime_home_pending
                ):
                    self._state.runtime_home_pending = False
                    # Generic ``home_ready`` is authoritative, but the rule
                    # heartbeat may lag it by one publication.  Keep the
                    # one-shot request latch until this fresh Session receipt
                    # arrives so the stale DEBUG_ONLY label cannot enqueue a
                    # duplicate runtime Home.
                    self._device_recovery_home_requested = False
            if "readiness_reason" in status:
                self._state.readiness_reason = str(
                    status.get("readiness_reason") or ""
                )
            if "readiness_blockers" in status:
                raw_blockers = status.get("readiness_blockers")
                if isinstance(raw_blockers, Mapping):
                    readiness_blockers = {
                        str(key): str(value) for key, value in raw_blockers.items()
                    }
                    self._rule_readiness_blockers = readiness_blockers
                    self._rebuild_startup_blockers_locked()
            self._state.step_id = (
                None if status.get("step_id") is None else str(status["step_id"])
            )
            self._state.directive_kind = (
                None
                if status.get("directive_kind") is None
                else str(status["directive_kind"])
            )
            self._state.sequence_index = (
                None
                if status.get("sequence_index") is None
                else int(status["sequence_index"])
            )
            self._state.stage_label = (
                None if status.get("stage_label") is None else str(status["stage_label"])
            )
            self._state.stage_index = self._optional_int(status.get("stage_index"))
            self._state.stage_frame_index = self._optional_int(
                status.get("stage_frame_index")
            )
            self._state.stage_frame_total = self._optional_int(
                status.get("stage_frame_total")
            )
            self._state.recorded_sample_count = max(
                0, self._optional_int(status.get("recorded_sample_count")) or 0
            )
            self._state.planned_sample_count = max(
                0, self._optional_int(status.get("planned_sample_count")) or 0
            )
            self._state.collection_position = (
                None
                if status.get("collection_position") is None
                else str(status["collection_position"])
            )
            speed = status.get("initial_alignment_speed_scale")
            try:
                self._state.initial_alignment_speed_scale = (
                    None if speed is None else float(speed)
                )
            except (TypeError, ValueError):
                self._state.initial_alignment_speed_scale = None
            raw_gripper_sides = status.get("gripper_sides") or ()
            self._state.gripper_sides = [str(side) for side in raw_gripper_sides]
            self._state.control_topology = str(
                status.get("control_topology", "") or "双臂（无夹爪）"
            )
            self._state.gripper_control_available = bool(
                status.get("gripper_control_available", False)
            )
            self._state.gripper_action_pending = bool(
                status.get(
                    "gripper_action_pending",
                    self._state.gripper_action_pending,
                )
            )
            if "gripper_action_id" in status:
                action_id = self._optional_int(
                    status.get("gripper_action_id")
                )
                self._state.gripper_action_id = action_id
                if action_id is not None:
                    self._last_gripper_action_id = max(
                        action_id,
                        self._last_gripper_action_id or action_id,
                    )
            if "gripper_action_intent" in status:
                self._state.gripper_action_intent = str(
                    status.get("gripper_action_intent") or ""
                )
            if "gripper_action_result" in status:
                self._state.gripper_action_result = str(
                    status.get("gripper_action_result") or "idle"
                )
            self._state.rule_start_available = bool(
                status.get("rule_start_available", False)
            )
            if "configuration_available" in status:
                self._state.configuration_available = bool(
                    status.get("configuration_available", False)
                )
            else:
                # Compatibility for old/in-process status fixtures.  The
                # production rule runtime always publishes the authoritative
                # lifecycle value.
                self._state.configuration_available = bool(
                    self._state.rule_start_available
                    and not completed
                    and not aborted
                )
            self._state.gripper_action_status = str(
                status.get("gripper_action_status", "") or ""
            )
            raw_configuration = status.get("rule_configuration")
            if isinstance(raw_configuration, Mapping):
                self._state.rule_configuration = dict(raw_configuration)
            self._state.collection_mode = str(
                status.get("collection_mode", self._state.collection_mode)
                or "rule_based_writing"
            )
            raw_batch = status.get("rule_configuration_batch")
            if isinstance(raw_batch, (list, tuple)):
                self._state.rule_configuration_batch = [
                    dict(item) for item in raw_batch if isinstance(item, Mapping)
                ]
            self._state.batch_index = max(
                0, self._optional_int(status.get("batch_index")) or 0
            )
            self._state.batch_count = max(
                1, self._optional_int(status.get("batch_count")) or 1
            )
            self._state.batch_completed_count = max(
                0, self._optional_int(status.get("batch_completed_count")) or 0
            )
            self._state.batch_cycle_complete = bool(
                status.get("batch_cycle_complete", False)
            )
            self._state.configuration_pending = bool(
                status.get(
                    "configuration_pending",
                    self._state.configuration_pending,
                )
            )
            self._state.configuration_error = str(
                status.get(
                    "configuration_error",
                    self._state.configuration_error,
                )
                or ""
            )
            self._state.height_calibration_enabled = bool(
                status.get("height_calibration_enabled", False)
            )
            self._state.height_calibrated_z_m = self._optional_float(
                status.get("height_calibrated_z_m")
            )
            self._state.height_lower_tolerance_m = self._optional_float(
                status.get("height_lower_tolerance_m")
            ) or 0.001
            self._state.height_upper_tolerance_m = self._optional_float(
                status.get("height_upper_tolerance_m")
            ) or 0.001
            self._state.height_lower_bound_z_m = self._optional_float(
                status.get("height_lower_bound_z_m")
            )
            self._state.height_upper_bound_z_m = self._optional_float(
                status.get("height_upper_bound_z_m")
            )
            self._state.height_calibration_available = bool(
                status.get("height_calibration_available", False)
            )
            self._state.height_calibration_clear_available = bool(
                status.get("height_calibration_clear_available", False)
            )
            self._state.height_calibration_pending = bool(
                status.get(
                    "height_calibration_pending",
                    self._state.height_calibration_pending,
                )
            )
            self._state.height_calibration_error = str(
                status.get(
                    "height_calibration_error",
                    self._state.height_calibration_error,
                )
                or ""
            )
            self._state.recording_ready = bool(status.get("recording_ready", False))
            self._state.held = held
            self._state.completed = completed
            self._state.aborted = aborted
            if (
                begun
                or held
                or completed
                or aborted
                or self._effective_startup_blockers_locked()
            ):
                # Once the first rule command has actually begun execution,
                # the pre-Episode writer exception has served its purpose and
                # must not survive a later stale heartbeat.
                self._pre_episode_start_admission_open = False
            if self._state.device_retry_pending:
                self._project_device_retry_transaction_locked()
            elif self._state.configuration_pending:
                self._state.phase = "configuring"
                self._state.title = "正在更新规则组合"
                self._state.detail = "正在后台读取并验证所选来源；完成前不会输出运动命令。"
            elif self._state.runtime_home_pending:
                self._set_runtime_home_pending_locked()
            elif self._effective_startup_blockers_locked() and not begun:
                if self._rule_owner_recovery_window_locked():
                    self._set_ready_locked()
                    self._state.detail = (
                        "规则启动暂不可用；请重新执行右夹爪一次性闭合，"
                        "并等待实测闭紧反馈后再继续。"
                    )
                else:
                    self._set_readiness_blocked_locked()
            elif self._state.writer_blocked and not begun:
                self._state.phase = "blocked"
                self._state.title = "正式 Writer 已阻塞"
                self._state.detail = self._state.writer_warning
            elif aborted:
                self._state.phase = "aborted"
                self._state.title = "规则采集已停止"
                self._state.detail = "机械臂已进入 Hold；当前规则任务不会继续执行。"
            elif completed:
                self._state.phase = "completed"
                self._state.title = "规则采集已完成"
                self._state.detail = "规则计划已完成，正在或已经提交 writer。"
            elif held:
                self._state.phase = "blocked"
                self._state.title = "规则采集已暂停"
                self._state.detail = (
                    "本轮 Episode 已取消；确认机械臂安全后点击“重新安全初始化”，"
                    "完成后可重新开始规则采集。"
                )
            elif begun:
                self._state.phase = "running"
                self._state.title = "规则采集运行中"
                if waiting in {
                    "formal_writer_paused",
                    "initial_alignment_tracking_lag",
                }:
                    self._state.detail = self._waiting_detail_locked(waiting)
                else:
                    step = self._state.step_id or "等待下一步"
                    self._state.detail = f"当前规则步骤：{step}"
            elif (
                # A dedicated START intent is accepted before the first rule
                # command is consumed.  During that short interval the rule
                # runtime can legitimately publish an older ``armed=False``
                # heartbeat.  Keep the application-owned admission window
                # visible until an authoritative blocker or terminal state
                # closes it; otherwise the writer countdown is cancelled by
                # its own stale status refresh.
                self._pre_episode_start_admission_open
                and not self._state.writer_blocked
                and not self._state.configuration_pending
                and not self._state.writer_retry_pending
                and not self._state.gripper_action_pending
                and not self._state.height_calibration_pending
                and str(self._state.session_state or "").upper()
                in {"", "ARMED", "RUNNING"}
            ):
                self._state.phase = "armed"
                self._state.title = "规则采集启动中"
                self._state.detail = self._waiting_detail_locked(
                    waiting or "formal_episode_not_active"
                )
            elif armed:
                self._state.phase = "armed"
                self._state.title = "规则采集启动中"
                self._state.detail = self._waiting_detail_locked(waiting)
            elif self._rule_runtime_bound:
                self._project_prestart_readiness_locked(
                    status_started=bool(status.get("started", False))
                )
            if self._state.phase == "ready":
                if self._state.configuration_error:
                    self._state.detail = self._state.configuration_error
                elif self._state.gripper_action_status:
                    self._state.detail = self._state.gripper_action_status
            self._maybe_continue_device_recovery_locked()
            self._try_start_waiting_task_chain_locked()
            self._observe_task_chain_status_locked(
                begun=begun,
                completed=completed,
                aborted=aborted,
                held=held,
            )
            self._rebuild_controls()
            self._touch()

    def _try_start_waiting_task_chain_locked(self) -> bool:
        """Commit a deferred chain start after a fresh lower-layer receipt.

        Configuration completion and Session/Rig readiness are asynchronous.
        If the configuration callback observes a stale start receipt, the
        chain deliberately enters ``READY_RULE`` with ``rule_start_waiting``.
        A later authoritative status update can now complete that same pending
        transition without requiring a second operator click or bypassing the
        normal start admission predicate.

        This method is called only while ``self._lock`` is held.  It must stay
        narrow: it may submit the existing START intents, but it does not
        synthesize readiness or dispatch a robot command.
        """

        if (
            self._task_chain.phase is not RuleWritingTaskChainPhase.READY_RULE
            or not self._task_chain.rule_start_waiting
            or self._state.writer_blocked
            or self._state.configuration_pending
            or self._state.writer_retry_pending
            or self._state.gripper_action_pending
            or self._state.height_calibration_pending
            or self._state.runtime_home_pending
            or self._effective_startup_blockers_locked()
        ):
            return False
        if not self._base_control_availability_locked()["start"]:
            return False
        if not self._submit_intent_locked(RuleBasedWritingIntent.START):
            return False
        self._task_chain.begin_rule_after_wait()
        return True

    def _clear_device_recovery_locked(self) -> None:
        self._device_recovery_task_chain_step = None
        self._device_recovery_home_requested = False
        self._device_recovery_owner_close_requested = False

    def _maybe_continue_device_recovery_locked(self) -> bool:
        """Continue one interrupted task-chain step through canonical gates.

        Reconnection alone is never motion admission.  The existing
        robot-state-only runtime Home must succeed first, then the exact step
        accepted before the device fault is resubmitted once.  Repeated status
        publications are intentionally idempotent.
        """

        step = self._device_recovery_task_chain_step
        if (
            step is None
            or self._state.device_retry_pending
            or self._device_retry_waiting_for_hold
            or self._state.configuration_pending
            or self._state.writer_retry_pending
            or self._state.height_calibration_pending
            or self._state.shutdown
        ):
            return False

        blockers = self._effective_startup_blockers_locked()
        non_home_blockers = {
            key: detail
            for key, detail in blockers.items()
            if not self._is_runtime_home_requirement(key, detail)
            and key != "rule.right_hand_hold_owner"
        }
        if non_home_blockers:
            return False

        if self._state.runtime_home_pending:
            return False
        if self._session_requires_runtime_home_locked():
            if self._device_recovery_home_requested:
                return False
            if not self._submit_intent_locked(
                RuleBasedWritingIntent.RUNTIME_HOME_RETRY
            ):
                return False
            self._device_recovery_home_requested = True
            self._append_message_locked(
                "Progress",
                "失败设备已恢复；正在通过现有 runtime Home 门禁重建任务链安全状态。",
            )
            return True

        if blockers and set(blockers) != {"rule.right_hand_hold_owner"}:
            return False
        if self._state.gripper_action_pending:
            return False

        if step == "rule":
            phase = self._task_chain.phase
            if phase is RuleWritingTaskChainPhase.CONFIGURATION_PENDING:
                return False
            if phase is RuleWritingTaskChainPhase.RULE_PENDING:
                self._clear_device_recovery_locked()
                return True
            if phase is not RuleWritingTaskChainPhase.READY_RULE:
                return False
            if self._rule_owner_recovery_window_locked():
                if self._device_recovery_owner_close_requested:
                    return False
                if not self._submit_intent_locked(
                    RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
                ):
                    return False
                self._device_recovery_owner_close_requested = True
                self._append_message_locked(
                    "Progress",
                    "设备会话恢复后正在重新建立右夹爪一次性闭合 owner；"
                    "实测闭紧前不会恢复规则运动。",
                )
                return True
            if not self._task_chain.selected_configuration:
                return False
            if not self._task_chain.rule_start_waiting:
                self._task_chain.wait_for_recovered_rule_start()
            if not self._try_start_waiting_task_chain_locked():
                return False
            self._clear_device_recovery_locked()
            return True

        action_id_baseline = self._last_gripper_action_id
        if step == "initial_release":
            if self._task_chain.phase is not RuleWritingTaskChainPhase.IDLE:
                return False
            intent = RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER
        elif step == "close":
            if self._task_chain.phase is not RuleWritingTaskChainPhase.READY_CLOSE:
                return False
            intent = RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER
        elif step == "final_release":
            if (
                self._task_chain.phase
                is not RuleWritingTaskChainPhase.READY_FINAL_RELEASE
            ):
                return False
            intent = RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER
        else:
            raise RuntimeError(f"unknown device recovery task-chain step: {step}")
        if not self._submit_intent_locked(intent):
            return False
        if step == "initial_release":
            self._task_chain.begin_initial_release(
                action_id_baseline=action_id_baseline
            )
        elif step == "close":
            self._task_chain.begin_close(action_id_baseline=action_id_baseline)
        else:
            self._task_chain.begin_final_release(
                action_id_baseline=action_id_baseline
            )
        self._clear_device_recovery_locked()
        return True

    def _observe_task_chain_status_locked(
        self,
        *,
        begun: bool,
        completed: bool,
        aborted: bool,
        held: bool,
    ) -> None:
        if self._task_chain_cancel_pending:
            if held or aborted:
                self._task_chain.cancel(
                    "任务链已取消并进入 Hold；重新完成安全初始化后可开始新一轮。"
                )
                self._task_chain_cancel_pending = False
            return

        phase = self._task_chain.phase
        if (
            phase
            in {
                RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING,
                RuleWritingTaskChainPhase.CLOSE_PENDING,
                RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING,
            }
        ):
            expected_intent = (
                RuleBasedWritingIntent.CLOSE_RIGHT_GRIPPER.value
                if phase is RuleWritingTaskChainPhase.CLOSE_PENDING
                else RuleBasedWritingIntent.RELEASE_RIGHT_GRIPPER.value
            )
            action_id = self._state.gripper_action_id
            if (
                action_id is None
                or self._state.gripper_action_intent != expected_intent
                or not self._task_chain.bind_gripper_action(action_id)
            ):
                return
            result = self._state.gripper_action_result
            if result in {"requested", "pending"}:
                return
            if result not in {"succeeded", "failed"}:
                return
            error = (
                None
                if result == "succeeded"
                else self._state.gripper_action_status or "夹爪动作未返回成功回执。"
            )
            if phase is RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING:
                self._task_chain.finish_initial_release(error=error)
            elif phase is RuleWritingTaskChainPhase.CLOSE_PENDING:
                self._task_chain.finish_close(error=error)
            else:
                self._task_chain.finish_final_release(error=error)
            return

        if phase is not RuleWritingTaskChainPhase.RULE_PENDING:
            return
        if begun:
            self._task_chain.observe_rule_execution()
        if aborted or held:
            self._task_chain.fail_rule(
                "规则 Episode 已中止或进入 Hold；完成现有恢复门禁后可重试。"
            )
            return
        if (
            self._task_chain.rule_execution_seen
            and not begun
            and not completed
            and self._state.batch_completed_count >= 1
            and not self._state.configuration_pending
        ):
            # The lower runtime increments this count only after the writer has
            # fully closed the singleton chain Episode and reset to pre-start.
            self._task_chain.finish_rule()

    @staticmethod
    def _waiting_detail(waiting: str) -> str:
        if waiting in {
            "formal_episode_not_active",
            "formal_writer_not_recording",
        }:
            return "正在准备规则采集 writer；就绪前不会输出运动命令。"
        if waiting.startswith("causal_arm_feedback"):
            return "正在等待双臂因果反馈就绪；就绪前不会输出运动命令。"
        if waiting == "initial_alignment_planning":
            return "正在后台规划并验证双臂初始化路径；验证完成前不会输出运动命令。"
        if waiting == "initial_alignment_tracking_lag":
            return "双臂实测状态暂时落后于初始化控制流；保持当前认证目标，跟上后自动继续。"
        if waiting == "startup_home_continuity_dwell":
            return (
                "首条初始化 qref 正在通过 runtime Home 连续性验证；"
                "连续验证完成前不会下发运动或写入样本。"
            )
        if waiting == "right_gripper_hold_owner_pending":
            return (
                "右夹爪尚未建立书写用的一次性闭合 owner；"
                "请先完成“闭紧右夹爪”并等待“实测已闭紧”。"
            )
        if waiting == "initial_alignment_safety_rejected":
            return "双臂初始化路径未通过安全验证，机械臂已进入 Hold。"
        if waiting == "coupled_pose_safety_rejected":
            return "规则动作未通过双臂耦合安全验证，机械臂已进入 Hold。"
        if waiting == "runtime_home_required_after_hold":
            return "本轮规则采集已取消；需要重新完成安全初始化后才能再次开始。"
        if waiting == "formal_writer_paused":
            return "采集模态暂时未就绪；保持最后一个安全目标，连续新鲜后自动恢复。"
        if waiting == "session_readiness_failure":
            return (
                "机械臂尚未通过 Session arm admission；请检查硬件状态并重新完成 runtime Home。"
            )
        return "规则任务已确认，正在通过运行前门禁。"

    def _waiting_detail_locked(self, waiting: str) -> str:
        """Prefer the writer owner's live lifecycle explanation while waiting.

        The generic recording runtime owns writer warmup, countdown and
        Episode-open transitions.  Its publication is more actionable than
        the rule runtime's intentionally coarse ``formal_episode_not_active``
        marker, so project it without copying any writer state machine here.
        """

        if waiting in {
            "formal_episode_not_active",
            "formal_writer_not_recording",
            "formal_writer_paused",
        }:
            detail = str(self._state.episode_event_message or "").strip()
            if detail:
                return detail
        return self._waiting_detail(waiting)

    def request_completion_stop(self) -> None:
        """Ask the existing writer owner to finalize after the plan completes."""

        with self._lock:
            if self._completion_stop_queued:
                return
            self._completion_stop_queued = True
            # Carry an explicit completion reason so the progress adapter can
            # distinguish a successfully completed rule window from an
            # operator-aborted/manual stop.
            self._runtime_events.put("episode_stop:rule_based_complete")
            self._touch()

    def request_failure_stop(self) -> None:
        """Ask the writer owner to discard/seal a failed rule Episode."""

        with self._lock:
            if self._completion_stop_queued:
                return
            self._completion_stop_queued = True
            self._runtime_events.put("episode_stop:rule_based_abort")
            self._touch()

    def append_message(self, tag: str, message: str) -> None:
        normalized_tag = str(tag)
        if normalized_tag not in self._VISIBLE_MESSAGE_TAGS:
            return
        with self._lock:
            self._append_message_locked(normalized_tag, message)
            self._touch()

    def _append_message_locked(self, tag: str, message: str) -> None:
        self._state.messages.append({"tag": str(tag), "message": str(message)})
        self._state.messages = self._state.messages[-32:]

    def mark_shutdown(self, message: str) -> None:
        with self._lock:
            self._state.shutdown = True
            self._state.phase = "shutdown"
            self._state.title = "规则采集程序已退出"
            self._state.detail = str(message)
            self._rebuild_controls()
            self._touch()

    def _rebuild_controls(self) -> None:
        available = self._base_control_availability_locked()
        chain = self._task_chain.presentation()
        self._state.task_chain_active = bool(chain["active"])
        self._state.task_chain_phase = str(chain["phase"])
        self._state.task_chain_step = str(chain["step"])
        self._state.task_chain_title = str(chain["title"])
        self._state.task_chain_cycle_count = int(chain["cycle_count"])
        selected_configuration = chain["selected_configuration"]
        self._state.task_chain_selected_configuration = (
            dict(selected_configuration)
            if isinstance(selected_configuration, Mapping)
            else {}
        )
        selected_index = chain["selected_configuration_index"]
        self._state.task_chain_selected_configuration_index = (
            None if selected_index is None else int(selected_index)
        )
        self._state.task_chain_can_advance = self._task_chain_can_advance_locked(
            available
        )
        owner_recovery_required = self._rule_owner_recovery_window_locked()
        self._state.task_chain_owner_recovery_required = owner_recovery_required
        task_chain_detail = str(chain["detail"])
        if (
            self._task_chain.phase
            in {
                RuleWritingTaskChainPhase.IDLE,
                RuleWritingTaskChainPhase.COMPLETE,
            }
            and not self._state.task_chain_can_advance
        ):
            task_chain_detail = self._task_chain_blocked_detail_locked(available)
        elif (
            self._task_chain.phase is RuleWritingTaskChainPhase.READY_RULE
            and not self._state.task_chain_can_advance
            and (
                self._effective_startup_blockers_locked()
                or self._state.runtime_home_pending
            )
        ):
            task_chain_detail = self._task_chain_blocked_detail_locked(available)
        elif owner_recovery_required:
            task_chain_detail = (
                "右夹爪串口会话已变化，先前闭爪 owner 已安全失效。"
                "点击下一步只会重新发送右夹爪一次性闭合命令；"
                "收到新的实测闭紧反馈后，任务链才会恢复规则采集。"
            )
            if self._state.gripper_action_pending:
                task_chain_detail = (
                    "正在恢复右夹爪一次性闭合 owner；等待新串口会话中的"
                    "实测闭紧反馈。"
                )
            elif self._state.gripper_action_result == "failed":
                task_chain_detail += "\n" + (
                    self._state.gripper_action_status
                    or "右夹爪 owner 恢复未完成，请确认安全后重试。"
                )
        elif (
            self._task_chain.phase is RuleWritingTaskChainPhase.RULE_PENDING
            and self._state.phase == "armed"
        ):
            task_chain_detail = self._waiting_detail_locked(
                "formal_episode_not_active"
            )
        self._state.task_chain_detail = task_chain_detail

        chain_owns_manual_step = self._task_chain.active
        self._state.can_configure = bool(
            available["configure"] and not chain_owns_manual_step
        )
        self._state.can_start = bool(
            available["start"] and not chain_owns_manual_step
        )
        self._state.can_stop = bool(
            available["stop"]
            or (
                self._task_chain.active
                and not self._task_chain_cancel_pending
                and not self._state.shutdown
            )
        )
        self._state.can_retry_devices = available["retry_devices"]
        self._state.can_runtime_home = available["runtime_home"]
        self._state.can_release_grippers = bool(
            available["release_grippers"] and not chain_owns_manual_step
        )
        self._state.can_close_grippers = bool(
            available["close_grippers"] and not chain_owns_manual_step
        )
        self._state.can_toggle_debug = available["toggle_debug"]
        self._state.can_capture_height = bool(
            available["capture_height"] and not chain_owns_manual_step
        )
        self._state.can_clear_height = bool(
            available["clear_height"] and not chain_owns_manual_step
        )

    def _base_control_availability_locked(self) -> dict[str, bool]:
        terminal = self._state.completed or self._state.aborted or self._state.shutdown
        configure = bool(
            self._rule_runtime_bound
            and self._state.configuration_available
            and not self._state.shutdown
            and not self._state.device_retry_pending
            and not self._state.configuration_pending
            and not self._state.writer_retry_pending
            and not self._state.gripper_action_pending
            and not self._state.height_calibration_pending
        )
        start_blockers = self._state.startup_blockers
        if (
            self._task_chain.phase is RuleWritingTaskChainPhase.READY_RULE
            and self._task_chain.rule_start_waiting
        ):
            # Configuration has already been validated for this dedicated
            # chain.  Permit its explicit START retry through the same narrow
            # pre-dispatch Session projection used by the warmup gate; the
            # ordinary manual START path still requires no startup blockers.
            start_blockers = self._effective_startup_blockers_locked()
        start = bool(
            self._rule_runtime_bound
            and (self._state.rule_start_available or self._state.writer_blocked)
            and not terminal
            and self._state.phase in {"ready", "blocked"}
            and not self._state.configuration_pending
            and not self._state.writer_retry_pending
            and not self._state.gripper_action_pending
            and not self._state.height_calibration_pending
            and not start_blockers
        )
        stop = bool(
            not terminal
            and (
                self._state.phase in {"armed", "running"}
                or (self._state.phase == "blocked" and self._state.writer_blocked)
            )
        )
        retry_devices = bool(
            self._rule_runtime_bound
            and not terminal
            and self._state.phase in {"ready", "blocked", "preparing"}
            and self._has_retryable_device_blockers_locked()
            and self._device_retry_request_available_locked()
            and not self._state.device_retry_pending
            and not self._state.configuration_pending
            and not self._state.writer_retry_pending
            and not self._state.gripper_action_pending
            and not self._state.height_calibration_pending
        )
        runtime_home = bool(
            self._rule_runtime_bound
            and not terminal
            and not self._state.runtime_home_pending
            and self._state.phase in {"ready", "blocked"}
            and not self._state.configuration_pending
            and not self._state.writer_retry_pending
            and not self._state.gripper_action_pending
            and not self._has_retryable_device_blockers_locked()
            and self._session_requires_runtime_home_locked()
        )
        grippers_ready = bool(
            self._state.gripper_control_available
            and self._state.gripper_sides
            and not self._state.gripper_action_pending
        )
        release_grippers = bool(
            grippers_ready and not terminal and self._state.phase == "ready"
            and not self._state.height_calibration_pending
        )
        close_grippers = bool(
            grippers_ready and not terminal and self._state.phase == "ready"
            and not self._state.height_calibration_pending
        )
        close_right_gripper = bool(
            close_grippers and "right" in self._state.gripper_sides
        )
        toggle_debug = bool(
            self._rule_runtime_bound
            and not self._state.shutdown
            and not self._state.debug_mode_pending
        )
        capture_height = bool(
            self._state.collection_mode == "rule_based_writing"
            and self._state.height_calibration_available
            and not terminal
            and self._state.phase == "ready"
            and not self._state.height_calibration_pending
            and not self._state.configuration_pending
            and not self._state.gripper_action_pending
        )
        clear_height = bool(
            self._state.collection_mode == "rule_based_writing"
            and self._state.height_calibration_clear_available
            and not terminal
            and self._state.phase == "ready"
            and not self._state.height_calibration_pending
            and not self._state.configuration_pending
            and not self._state.gripper_action_pending
        )
        return {
            "configure": configure,
            "start": start,
            "stop": stop,
            "retry_devices": retry_devices,
            "runtime_home": runtime_home,
            "release_grippers": release_grippers,
            "close_grippers": close_grippers,
            "close_right_gripper": close_right_gripper,
            "toggle_debug": toggle_debug,
            "capture_height": capture_height,
            "clear_height": clear_height,
        }

    def _task_chain_initially_available_locked(
        self,
        available: Mapping[str, bool],
    ) -> bool:
        return bool(
            self._state.collection_mode == "rule_based_writing"
            and self._task_chain_configuration_pool
            and available["configure"]
            and available["release_grippers"]
            and available["close_right_gripper"]
            and available["capture_height"]
        )

    def _task_chain_blocked_detail_locked(
        self,
        available: Mapping[str, bool],
    ) -> str:
        """Explain the first unmet task-chain admission condition.

        The button remains fail-closed.  This projection only tells the
        operator why the first step is unavailable; it must not turn a
        disabled manual gripper action into an unsafe skip.
        """

        if self._state.collection_mode != "rule_based_writing":
            return "当前采集模式不是 rule-based writing，任务链不可用。"
        if not self._rule_runtime_bound:
            return "等待规则运行时完成绑定；就绪前不能开始任务链。"
        effective_blockers = self._effective_startup_blockers_locked()
        if effective_blockers:
            if "rule.right_hand_hold_owner" in effective_blockers:
                return (
                    "规则采集保持禁用；请先完成任务链中的“闭紧右夹爪”，"
                    "并等待右夹爪反馈确认实测闭紧。\n"
                    + effective_blockers["rule.right_hand_hold_owner"]
                )
            detail_lines = "; ".join(
                f"{key}: {value}"
                for key, value in sorted(effective_blockers.items())
            )
            if self._has_retryable_device_blockers_locked():
                if self._device_retry_request_available_locked():
                    action = "请先重连失败设备，并重新完成 runtime Home。"
                else:
                    action = (
                        "底层当前未提供安全重连控制；请保持 Hold，"
                        "等待底层恢复后再重试。"
                    )
            else:
                action = "请重新完成 runtime Home；完成前不能继续任务链。"
            return "规则采集保持禁用；" + action + "\n" + detail_lines
        if self._state.runtime_home_pending:
            return "正在重新确认 runtime Home；完成前不能继续任务链。"
        if not self._state.gripper_control_available:
            return "等待夹爪控制拓扑就绪；任务链第一步需要先实测松开右夹爪。"
        if "right" not in self._state.gripper_sides:
            return "当前没有右夹爪控制；任务链需要右夹爪抓笔，不能跳过松开步骤。"
        if not available["release_grippers"]:
            return "等待右夹爪松开操作的安全门禁就绪；就绪后才能开始任务链。"
        if not available["close_right_gripper"]:
            return "等待右夹爪闭合操作的安全门禁就绪；不能跳过夹爪步骤。"
        if not available["capture_height"]:
            return "等待高度标定能力和安全状态就绪；就绪前不能开始任务链。"
        if not self._task_chain_configuration_pool:
            return "等待可用轨迹组合完成预验证；就绪前不能开始任务链。"
        if not available["configure"]:
            return "等待轨迹配置事务可用；完成前不会输出规则运动命令。"
        if not available["start"]:
            return "等待现有安全门禁放行；安全就绪后才能开始任务链。"
        return "任务链尚未满足全部前置条件；请查看上方设备与安全消息。"

    def _task_chain_can_advance_locked(
        self,
        available: Mapping[str, bool],
    ) -> bool:
        phase = self._task_chain.phase
        if phase in {
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
        }:
            return self._task_chain_initially_available_locked(available)
        if phase is RuleWritingTaskChainPhase.READY_CLOSE:
            return bool(available["close_right_gripper"])
        if phase is RuleWritingTaskChainPhase.READY_HEIGHT:
            return bool(available["capture_height"])
        if phase is RuleWritingTaskChainPhase.READY_RULE:
            if self._rule_owner_recovery_window_locked():
                return bool(available["close_right_gripper"])
            if (
                self._effective_startup_blockers_locked()
                or self._state.runtime_home_pending
            ):
                return False
            if self._task_chain.rule_start_waiting:
                return bool(available["start"])
            return bool(available["configure"] and self._task_chain_configuration_pool)
        if phase is RuleWritingTaskChainPhase.READY_FINAL_RELEASE:
            return bool(available["release_grippers"])
        return False

    def _set_ready_locked(self) -> None:
        self._state.phase = "ready"
        self._state.title = "规则采集已就绪"
        self._state.detail = "确认工作区安全后，点击“开始规则采集”。"

    def _project_prestart_readiness_locked(
        self,
        *,
        status_started: bool | None = None,
    ) -> None:
        """Project only an explicit runtime readiness receipt to the UI.

        ``bind_rule_runtime`` is deliberately earlier than Session arm
        admission.  This helper centralizes the pre-start projection so every
        update path (generic startup publication, rule status, and
        configuration completion) follows the same fail-closed ordering.
        ``status_started`` is supplied by the status callback when available;
        the cached ``_rule_status_seen`` gate still prevents a bind-only ready
        state.
        """

        if self._state.runtime_home_pending:
            self._set_runtime_home_pending_locked()
            return
        if self._effective_startup_blockers_locked():
            if self._rule_owner_recovery_window_locked():
                self._set_ready_locked()
                self._state.detail = (
                    "规则启动暂不可用；请重新执行右夹爪一次性闭合，"
                    "并等待实测闭紧反馈后再继续。"
                )
            else:
                self._set_readiness_blocked_locked()
            return
        if self._session_requires_runtime_home_locked():
            self._set_readiness_blocked_locked()
            return
        if status_started is None:
            status_started = self._rule_status_seen
        session_state = str(self._state.session_state or "").upper()
        if bool(status_started) and session_state not in {"", "ARMED", "RUNNING"}:
            self._set_readiness_blocked_locked()
            return
        if not self._rule_status_seen:
            self._set_preparing_status_locked()
            return
        if (
            self._task_chain_configuration_pool
            and self._is_expected_task_chain_gripper_blocker_locked(
                key="rule.right_hand_hold_owner"
            )
            and "rule.right_hand_hold_owner" in self._state.startup_blockers
        ):
            # The dedicated chain is intentionally ready for its command-inert
            # release/close gates even though rule START itself remains
            # disabled until the close receipt establishes the hold owner.
            self._set_ready_locked()
            return
        if self._state.rule_start_available or self._state.writer_blocked:
            if self._state.writer_blocked:
                self._state.phase = "blocked"
                self._state.title = "正式 Writer 已阻塞"
                self._state.detail = self._state.writer_warning
            else:
                self._set_ready_locked()
            return
        # A status was received, but it has not granted the rule start gate
        # (for example while the runtime is still opening the formal writer or
        # waiting for a safe Session state).  Keep the controls disabled and
        # make the wait explicit instead of claiming readiness.
        self._set_preparing_status_locked()

    def _set_preparing_status_locked(self) -> None:
        self._state.phase = "preparing"
        self._state.title = "规则采集环境准备中"
        self._state.detail = (
            "正在等待规则运行时完成设备、安全门禁和正式 Writer readiness；"
            "确认完成前不会输出规则运动命令。"
        )

    @staticmethod
    def _is_runtime_home_requirement(key: str, detail: str) -> bool:
        if str(key) != "rig.session":
            return False
        text = str(detail or "").lower()
        return any(
            marker in text
            for marker in (
                "runtime home",
                "runtime_home",
                "arm admission",
                "session state=",
            )
        )

    def _has_retryable_device_blockers_locked(self) -> bool:
        return any(
            not self._is_runtime_home_requirement(key, detail)
            and str(key) != "rule.right_hand_hold_owner"
            for key, detail in self._state.startup_blockers.items()
        )

    def _device_retry_directly_admitted_locked(self) -> bool:
        if "device_retry" in self._runtime_controls:
            return True
        # Preserve startup recovery before the generic runtime has published
        # its first control projection.  Once a projection exists, it is the
        # authoritative lower admission receipt.
        return bool(
            not self._runtime_controls_seen
            and str(self._state.session_state or "").upper()
            not in {"ARMED", "RUNNING"}
        )

    def _device_retry_request_available_locked(self) -> bool:
        return bool(
            self._device_retry_directly_admitted_locked()
            or "h" in self._runtime_controls
        )

    def _set_device_retry_running_locked(self) -> None:
        self._state.title = "正在重连失败设备"
        self._state.detail = (
            "底层停机已确认，正在重新检查设备 readiness；"
            "确认通过前规则采集保持禁用。"
        )

    def _project_device_retry_transaction_locked(self) -> None:
        self._state.phase = "preparing"
        if self._device_retry_waiting_for_hold:
            self._state.title = "正在安全停臂后重连"
            self._state.detail = (
                "已停止接纳规则命令；正在等待底层 Hold 与位置命令流释放回执，"
                "确认后将自动重连失败设备。"
            )
            return
        self._set_device_retry_running_locked()

    def _advance_device_retry_transaction_locked(self) -> None:
        if not (
            self._state.device_retry_pending
            and self._device_retry_waiting_for_hold
            and "device_retry" in self._runtime_controls
        ):
            return
        self._device_retry_waiting_for_hold = False
        self._runtime_events.put("device_retry")
        self._set_device_retry_running_locked()

    def _session_requires_runtime_home_locked(self) -> bool:
        blockers = {
            key: detail
            for key, detail in self._effective_startup_blockers_locked().items()
            if key != "rule.right_hand_hold_owner"
        }
        if self._runtime_home_retry_available:
            return True
        state = str(self._state.session_state or "").upper()
        if state in {"", "ARMED", "RUNNING", "FAULT", "ESTOP", "INIT"}:
            return any(
                self._is_runtime_home_requirement(key, detail)
                for key, detail in blockers.items()
            )
        if state not in {"CALIBRATION", "DEBUG_ONLY", "FROZEN", "RECOVERING"}:
            return False
        if not blockers:
            return bool(self._state.readiness_reason) or state != ""
        return all(
            self._is_runtime_home_requirement(key, detail)
            for key, detail in blockers.items()
        )

    def _set_runtime_home_pending_locked(self) -> None:
        self._state.phase = "preparing"
        self._state.title = "正在重新确认安全初始化"
        self._state.detail = (
            "正在重新采集 runtime Home；完成前规则采集保持禁用，"
            "不会输出规则运动命令。"
        )

    def _set_readiness_blocked_locked(self) -> None:
        if self._state.runtime_home_pending:
            self._set_runtime_home_pending_locked()
            return
        self._state.phase = "blocked"
        self._state.title = "机械臂安全接入未完成"
        if self._runtime_home_retry_available:
            self._state.detail = (
                "规则采集保持禁用；上一次安全初始化未完成。"
                "确认机械臂处于安全位置后点击“重新安全初始化”。\n"
                + self._runtime_home_retry_reason
            )
            return
        details = self._state.startup_blockers
        reason = self._state.readiness_reason
        if details and self._has_retryable_device_blockers_locked():
            detail_lines = "; ".join(
                f"{key}: {value}" for key, value in sorted(details.items())
            )
            if self._device_retry_request_available_locked():
                recovery = (
                    "请检查故障设备后点击“重连失败设备”，"
                    "并重新完成 runtime Home。"
                )
            else:
                recovery = (
                    "底层当前未提供安全重连控制；请保持 Hold，"
                    "等待底层恢复后再重试，不能绕过设备故障开始规则采集。"
                )
            self._state.detail = "规则采集保持禁用；" + recovery + "\n" + detail_lines
        elif details:
            detail_lines = "; ".join(
                f"{key}: {value}" for key, value in sorted(details.items())
            )
            self._state.detail = (
                "设备已通过 readiness 检查，但仍需重新完成 runtime Home；"
                "确认机械臂处于安全位置后点击“重新安全初始化”。\n"
                + detail_lines
            )
        elif reason:
            if self._session_requires_runtime_home_locked():
                self._state.detail = (
                    "规则采集保持禁用；请确认机械臂处于安全位置后点击"
                    "“重新安全初始化”。\n"
                    + reason
                )
            else:
                self._state.detail = (
                    "规则采集保持禁用；请先完成机械臂 runtime Home 和安全接入。\n"
                    + reason
                )
        else:
            self._state.detail = (
                "规则采集保持禁用；机械臂尚未通过 Session arm admission。"
            )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        return normalized if math.isfinite(normalized) else None

    def consume_runtime_events(self, *, max_events: int = 64) -> list[str]:
        events: list[str] = []
        for _ in range(max(1, int(max_events))):
            try:
                events.append(self._runtime_events.get_nowait())
            except queue.Empty:
                break
        return events

    def consume_rule_intents(self, *, max_events: int = 16) -> tuple[str, ...]:
        intents: list[str] = []
        for _ in range(max(1, int(max_events))):
            try:
                intents.append(self._rule_intents.get_nowait().value)
            except queue.Empty:
                break
        return tuple(intents)

    def consume_configuration_requests(
        self,
        *,
        max_events: int = 4,
    ) -> tuple[dict[str, object], ...]:
        requests: list[dict[str, object]] = []
        for _ in range(max(1, int(max_events))):
            try:
                requests.append(self._configuration_requests.get_nowait())
            except queue.Empty:
                break
        return tuple(requests)

    def consume_height_calibration_requests(
        self,
        *,
        max_events: int = 4,
    ) -> tuple[dict[str, object], ...]:
        requests: list[dict[str, object]] = []
        for _ in range(max(1, int(max_events))):
            try:
                requests.append(self._height_calibration_requests.get_nowait())
            except queue.Empty:
                break
        return tuple(requests)

    def configuration_options(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._configuration_options)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._state)


__all__ = [
    "RuleBasedWritingApplication",
    "RuleBasedWritingIntent",
    "RuleBasedWritingUiState",
]
