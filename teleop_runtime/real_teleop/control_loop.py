"""The canonical real-teleop per-frame control loop.

Resource construction, device startup, and finally cleanup remain in
`runtime.py`.  This module owns only the ordered control-frame cycle.  Mutable
outer lifecycle values are read through explicit accessors so device retry,
recording admission, and preview toggles retain same-frame visibility.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from dataset_contracts.stats import (
    FORCE_COLLISION_LABEL_DETECTED,
    has_force_limit_collision_text,
)
from robot_motion.rm65 import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from robot_motion.runtime import merge_teleop_commands
from teleop_runtime.collection.diagnostics import PhaseTimer
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.log_utils import log_event as _log
from teleop_runtime.recovery import (
    advance_component_recovery_plan as _advance_recovery_plan,
)
from teleop_runtime.recovery import (
    component_side_retargeter as _side_retargeter,
)
from teleop_runtime.recovery import (
    start_component_recovery_plan as _start_recovery_plan,
)
from teleop_runtime.session import TRACKER_HOLD_AUTOMATIC, TRACKER_HOLD_GRACE
from teleop_runtime.special_tasks.fix_arm import (
    advance_fix_arm_home_capture,
    build_fix_arm_pose_snapshot_payload,
)
from teleop_runtime.special_tasks.init_arm import advance_init_arm_home_capture
from teleop_runtime.tactile_visualizer import (
    AsyncDMTacDistributedForceVisualizer,
    DMTacDistributedForceVisualizer,
    dmtac_visualization_snapshot_from_backends,
    tactile_by_side_from_backends,
)

from ..target_safety.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeRuntimeError,
)
from .action_debug_payload import _build_action_debug_metadata
from .arm_runtime import (
    _active_command_backend,
    _apply_active_arm_command_to_preview_real,
    _runtime_arm_backend_state,
    _sync_dual_arm_retargeter_for_real_control,
)
from .arm_state_sync import _state_qpos, _sync_real_execution_step_limits
from .collection_bridge import (
    collection_sample_step_result as _collection_sample_step_result,
)
from .collection_bridge import (
    collection_step_result as _collection_step_result,
)
from .collection_bridge import (
    copy_teleop_command as _copy_teleop_command,
)
from .command_filters import _filter_manual_real_command
from .debug_log_frames import (
    _build_debug_timing_log_frame,
    _build_ik_debug_log_frame,
)
from .debug_timing import record_position_loop_timings as _record_position_loop_timings
from .debug_timing_payload import _channels_log_payload, _faults_log_payload
from .defaults import runtime_home_fresh_data_timeout_sec
from .episode_runtime import _build_episode_frame
from .gripper_control import _dispatch_q6_gripper_control
from .gripper_status import (
    _gripper_runtime_health_signature,
    _gripper_runtime_status_signature,
    _hand_channel_status,
)
from .gui_special_arm_status import (
    fix_arm_debug_state as _fix_arm_debug_state,
)
from .gui_special_arm_status import (
    fix_arm_status_text as _fix_arm_status_text,
)
from .gui_special_arm_status import (
    fix_arm_ui_state as _fix_arm_ui_state,
)
from .gui_special_arm_status import (
    init_arm_debug_state as _init_arm_debug_state,
)
from .gui_special_arm_status import (
    init_arm_status_text as _init_arm_status_text,
)
from .gui_special_arm_status import (
    init_arm_ui_state as _init_arm_ui_state,
)
from .hand_runtime import (
    _build_runtime_hand_command_for_recovery,
    _emit_tactile_sum_only_status,
    _enable_gui_tactile_solving,
    _extract_hand_command_from_command,
    _gripper_backend_for_side,
    _hand_backend_for_side,
    _hand_tactile_sum,
    _hold_requested_pause_reason,
    _merge_runtime_hand_command,
    _tactile_solving_active,
    _tactile_visualization_available,
)
from .impedance_force_runtime import (
    _emergency_stop_real_arms,
    _force_guard_recording_stop_context,
    _handle_init_qpos_emergency_stop,
)
from .movement import (
    movement_multiple_ui_state as _movement_multiple_ui_state,
)
from .movement import (
    movement_multiple_value as _movement_multiple_value,
)
from .position_admittance_shadow import (
    PositionAdmittanceActiveFault,
    admitted_solver_world_poses_from_joint_command,
)
from .progress_arm_control import (
    PendingProgressArmTransition,
    ProgressArmTransitionPlan,
    build_progress_arm_transition_plan,
    build_progress_entry_arm_transition_plan,
    progress_step_arm_requirements_differ,
)
from .progress_contracts import (
    PROGRESS_ARM_CONTROL_MODE_IGNORE,
    PromotableHoldPause,
    RealTeleopProgressWorkflowPort,
)
from .recording_admission import session_step_arm_dispatch_confirmed
from .runtime_debug import _debug_timing_maybe_report
from .runtime_extension import runtime_extension_dataset_sample_admission
from .runtime_health import (
    _command_backend_health_signature,
    _command_backend_status_text,
    _short_status_text,
    _temporary_safety_health_signature,
)
from .runtime_home import (
    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
    _build_renderer_debug_boxes,
    _build_renderer_debug_frames,
    _channel_pose_in_desk,
    _emit_home_alignment_logs,
    _emit_runtime_orientation_logs,
    _mean_pose,
    _resolve_robot_home_for_side,
    _runtime_home_calibration,
)
from .special_arm_runtime import (
    _cancel_special_arm_modes_for_hold,
    _disengage_position_backends_for_manual_interaction,
    _release_gui_qpos_init_transit,
)
from .startup_runtime import (
    _advance_startup_gui_qpos_maintenance,
    _arm_after_start_calibration,
    _runtime_home_failure_message,
    _update_init_arm_arrival_from_hardware,
)
from .status_payloads import (
    _build_ik_timing_payload,
    _build_ik_warning_payload,
    _build_pose_quality_display_payload,
    _ik_timing_status_text,
    _pose_quality_reason_list,
    _pose_quality_status_text,
    _runtime_fault_label,
)

_POSITION_ADMITTANCE_CONTROL_PLANE_REARM_THRESHOLD_SEC = 0.05

_PROGRESS_FAILURE_PROMOTABLE_HOLD_REASONS = frozenset(
    {"manual", "two_pedal_b", "two_pedal_c"}
)


def _build_promotable_progress_hold_pause(
    *,
    hold_reason: object,
    application_hold_was_active: bool,
    formal_capture_was_paused: bool,
    hold_requested_at: float,
    application_hold_effective_at: float | None,
    formal_capture_pause_effective_at: float | None,
    episode_capture_identity_hash: object,
    progress_phase: object,
) -> PromotableHoldPause | None:
    """Admit only an operator Hold that created this exact formal pause."""

    normalized_reason = str(hold_reason or "").strip()
    identity_hash = str(episode_capture_identity_hash or "").strip()
    if (
        normalized_reason not in _PROGRESS_FAILURE_PROMOTABLE_HOLD_REASONS
        or bool(application_hold_was_active)
        or bool(formal_capture_was_paused)
        or application_hold_effective_at is None
        or formal_capture_pause_effective_at is None
        or not identity_hash
        or str(progress_phase) != "ACTIVE_ATTEMPT"
    ):
        return None
    return PromotableHoldPause(
        episode_capture_identity_hash=identity_hash,
        hold_requested_at=hold_requested_at,
        application_hold_effective_at=application_hold_effective_at,
        formal_capture_pause_effective_at=(
            formal_capture_pause_effective_at
        ),
        hold_reason=normalized_reason,
    )


def _fresh_control_step_timestamp(
    loop_started_at: float,
    *,
    clock: Callable[[], float] = time.time,
) -> float:
    """Return a nondecreasing wall timestamp immediately before source polling."""

    return max(float(loop_started_at), float(clock()))


def _publish_real_agent_motion_measured_cut(
    components: object,
    *,
    hardware_state_by_side: Mapping[str, Mapping[str, object]],
    preview_qpos_by_side: Mapping[str, object],
    available_at: float,
) -> object | None:
    """Publish only fresh hardware qpos already read by this control frame."""

    port = getattr(components, "agent_motion_port", None)
    if port is None:
        return None
    active_sides = tuple(getattr(components, "active_sides", ()))
    qpos_by_side = {
        side: np.asarray(preview_qpos_by_side[side], dtype=float).reshape(6).copy()
        for side in ("left", "right")
    }
    measured_timestamps: list[float] = []
    evidence_refs: list[str] = []
    for side in active_sides:
        state = hardware_state_by_side.get(side, {})
        measured_qpos = state.get("measured_qpos")
        measurement_valid = bool(state.get("hardware_measurement_valid", False))
        measurement_timestamp = state.get("hardware_measurement_timestamp")
        measurement_sequence = state.get("hardware_measurement_sequence")
        if measured_qpos is None:
            measured_qpos = state.get("current_qpos")
            measurement_valid = bool(state.get("measurement_valid", False))
            measurement_timestamp = state.get("measurement_timestamp")
            measurement_sequence = state.get("measurement_sequence")
        if not measurement_valid or measured_qpos is None:
            return None
        try:
            qpos = np.asarray(measured_qpos, dtype=float).reshape(6)
            timestamp = float(measurement_timestamp)
            sequence = int(measurement_sequence)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(qpos)) or not np.isfinite(timestamp):
            return None
        qpos_by_side[side] = qpos.copy()
        measured_timestamps.append(timestamp)
        evidence_refs.append(
            f"realteleop-hardware-qpos:{side}:{sequence}:{timestamp:.9f}"
        )
    if not measured_timestamps:
        return None
    cut = port.publish_measured_qpos_pair(
        qpos_by_side=qpos_by_side,
        measured_at=min(measured_timestamps),
        available_at=max(float(available_at), max(measured_timestamps)),
        source_evidence_refs=tuple(evidence_refs),
    )
    if port.config.exclusive_command_claim:
        port.publish_owner_state(
            session=components.session,
            captured_at=cut.available_at,
        )
    return cut


def _exception_chain_text(error: BaseException) -> tuple[str, ...]:
    """Return the complete exception chain without re-raising any cause."""

    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return tuple(messages)


def _is_emergency_stop_latch_error(error: BaseException) -> bool:
    """Recognize a backend stop race that must enter safety Hold, not crash."""

    texts = _exception_chain_text(error)
    markers = (
        "emergency_stop is latched",
        "emergency stop is latched",
        "emergency_stop_failed",
        "emergency stop was not acknowledged",
        "emergency stop was not confirmed",
    )
    return any(
        marker in text.lower()
        for text in texts
        for marker in markers
    )


_EXCEPTION_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_JOINT_HARD_LIMIT_PATTERN = re.compile(
    rf"(?P<effector>left_arm|right_arm)\s+"
    rf"(?P<label>.+?)\s+joint\s+(?P<joint>\d+)="
    rf"(?P<target>{_EXCEPTION_FLOAT_PATTERN})rad\s+exceeds\s+absolute\s+limits\s+"
    rf"\[(?P<lower>{_EXCEPTION_FLOAT_PATTERN}),\s*"
    rf"(?P<upper>{_EXCEPTION_FLOAT_PATTERN})\]rad\."
)


def _joint_hard_limit_dispatch_context(
    error: BaseException,
) -> dict[str, object] | None:
    """Parse only the RM65 transport hard-limit rejection contract.

    The active control loop may consume this exact fail-closed rejection as a
    safety transition.  Other ``ValueError`` and ``RigDispatchError`` failures
    remain programming/transport errors and must still escape visibly.
    """

    exception_chain = _exception_chain_text(error)
    for text in exception_chain:
        match = _JOINT_HARD_LIMIT_PATTERN.search(text)
        if match is None:
            continue
        target_rad = float(match.group("target"))
        lower_rad = float(match.group("lower"))
        upper_rad = float(match.group("upper"))
        limit_side = "lower" if target_rad <= lower_rad else "upper"
        return {
            "effector_name": match.group("effector"),
            "target_label": match.group("label").strip(),
            "joint_index": int(match.group("joint")),
            "target_rad": target_rad,
            "lower_rad": lower_rad,
            "upper_rad": upper_rad,
            "limit_side": limit_side,
            "exception_chain": exception_chain,
        }
    return None


def _consume_joint_hard_limit_dispatch(
    error: BaseException,
    *,
    components: Any,
    clutch_input: Any,
    recording_active: Callable[[], bool],
    stop_recording: Callable[..., Any],
    cancel_recording_countdown: Callable[..., Any],
    emergency_stop_real_arms: Callable[..., Mapping[str, str]],
    now: float,
    step_index: int,
    last_state: object,
    log: Callable[..., Any],
    control_ui: Any,
) -> bool:
    """Turn one active RM65 hard-limit rejection into a retained safety Hold."""

    context = _joint_hard_limit_dispatch_context(error)
    if context is None:
        return False
    exception_chain = tuple(context.pop("exception_chain"))
    event = {
        "detected": True,
        "source": "rm65_transport_hard_limit_gate",
        **context,
        "message": " <- ".join(exception_chain),
    }
    summary: dict[str, object] = {
        "last_step_index": int(step_index),
        "last_state": last_state,
        "reason": "joint_hard_limit",
        "joint_hard_limit_detected": True,
        "joint_hard_limit_event": event,
    }
    _cancel_recording_countdown_for_hold_best_effort(
        cancel_recording_countdown,
        now=now,
        reason="joint_hard_limit",
        log=log,
    )
    stop_errors: dict[str, str] = {}
    try:
        stop_errors = dict(
            emergency_stop_real_arms(
                components,
                reason=(
                    "joint_hard_limit_dispatch:"
                    f"{event['effector_name']}:joint_{event['joint_index']}"
                ),
                now=now,
                raise_on_failure=False,
            )
            or {}
        )
    except Exception as exc:  # noqa: BLE001 - preserve the fail-closed transition.
        stop_errors = {"__all__": f"{type(exc).__name__}: {exc}"}
    if stop_errors:
        summary["emergency_stop_errors"] = dict(stop_errors)
    _stop_recording_best_effort(
        recording_active=recording_active,
        stop_recording=stop_recording,
        now=now,
        end_reason="joint_hard_limit",
        summary=summary,
        log=log,
        control_ui=control_ui,
        ui_tag="JointHardLimit",
        ui_failure_message=(
            "关节硬限位触发后的数采收尾失败；机械臂保持急停，"
            "请检查数据目录和设备状态。"
        ),
    )
    _request_safety_hold_best_effort(
        clutch_input,
        reason="joint_hard_limit",
        log=log,
        now=now,
    )
    log(
        "JointHardLimit",
        "active RM65 joint hard-limit rejection consumed; bimanual Arm_Stop "
        "requested, episode closed, and GUI/devices retained in Hold: "
        + " <- ".join(exception_chain),
        now=now,
    )
    if control_ui is not None:
        try:
            if stop_errors:
                control_ui.append_message(
                    "JointHardLimit",
                    "关节硬限位已触发，但至少一侧 Arm_Stop 未确认；当前数采已停止，"
                    "GUI/设备连接保持在 Hold。必须检查双臂状态后再决定是否恢复。",
                )
            else:
                control_ui.append_message(
                    "JointHardLimit",
                    "检测到机械臂关节硬限位门禁：双臂已急停并结束当前数采；"
                    "GUI/设备连接保持。请人工调整位置后点击“开始遥操”重新采集 runtime Home。",
                )
        except Exception as exc:  # noqa: BLE001 - UI cannot change safety state.
            log(
                "JointHardLimit",
                "hard-limit Hold message could not be published: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
    return True


_HORIZON_CERTIFICATION_MARKERS = (
    "reused batched joint horizon candidate failed certification",
    "batched joint horizon candidate failed certification",
    "joint horizon QP candidate failed certification",
    "reused batched joint horizon QP failed",
    "joint horizon numeric position projection would violate",
    "joint horizon position output exceeds numeric projection budget",
)


def _joint_horizon_certification_dispatch_context(
    error: BaseException,
) -> dict[str, object] | None:
    """Recognize only planner hard-certificate rejections.

    Horizon certification is a safety failure, but it is not a process
    failure.  The rig's fail-stop path has already held the command streams;
    the real-teleop loop must close the episode and enter its ordinary Hold
    state instead of allowing this expected runtime rejection to terminate
    the GUI process.
    """

    exception_chain = _exception_chain_text(error)
    for text in exception_chain:
        lowered = text.lower()
        for marker in _HORIZON_CERTIFICATION_MARKERS:
            if marker in lowered:
                return {
                    "marker": marker,
                    "exception_chain": exception_chain,
                }
    return None


def _consume_joint_horizon_certification_dispatch(
    error: BaseException,
    *,
    clutch_input: Any,
    recording_active: Callable[[], bool],
    stop_recording: Callable[..., Any],
    cancel_recording_countdown: Callable[..., Any],
    now: float,
    step_index: int,
    last_state: object,
    log: Callable[..., Any],
    control_ui: Any,
) -> bool:
    """Retain the runtime after an uncertified horizon dispatch.

    The candidate was not admitted and must not be retried in the same
    realtime frame.  Recording/countdown state is closed best-effort, then a
    normal safety Hold request is latched.  Cleanup and UI failures are
    intentionally contained so the operator can inspect the still-live GUI
    and devices rather than losing the whole process to a planner exception.
    """

    context = _joint_horizon_certification_dispatch_context(error)
    if context is None:
        return False
    exception_chain = tuple(context["exception_chain"])
    summary: dict[str, object] = {
        "last_step_index": int(step_index),
        "last_state": last_state,
        "reason": "joint_horizon_certification",
        "joint_horizon_certification_detected": True,
        "joint_horizon_certification_marker": str(context["marker"]),
        "message": " <- ".join(exception_chain),
    }
    _cancel_recording_countdown_for_hold_best_effort(
        cancel_recording_countdown,
        now=now,
        reason="joint_horizon_certification",
        log=log,
    )
    _stop_recording_best_effort(
        recording_active=recording_active,
        stop_recording=stop_recording,
        now=now,
        end_reason="joint_horizon_certification",
        summary=summary,
        log=log,
        control_ui=control_ui,
        ui_tag="JointHorizon",
        ui_failure_message=(
            "位控轨迹证书无法确认；当前数采已停止，机械臂保持 Hold，"
            "请调整位置后重新采集 runtime Home。"
        ),
    )
    # BimanualRig fail-stop has already held all backends when enabled.  The
    # latch below routes the next control-loop turn through the normal Hold
    # reducer, which performs session state transition and Home invalidation;
    # it also covers rigs configured without group fail-stop.
    _request_safety_hold_best_effort(
        clutch_input,
        reason="joint_horizon_certification",
        log=log,
        now=now,
    )
    log(
        "JointHorizon",
        "uncertified joint horizon dispatch consumed; episode closed and "
        "GUI/devices retained in Hold: "
        + " <- ".join(exception_chain),
        now=now,
    )
    if control_ui is not None:
        try:
            control_ui.append_message(
                "JointHorizon",
                "位控轨迹未通过关节安全证书：数采已停止，GUI/设备连接保持在 Hold；"
                "请人工调整机械臂后点击“开始遥操”重新采集 runtime Home。",
            )
        except Exception as exc:  # noqa: BLE001 - UI cannot change safety state.
            log(
                "JointHorizon",
                "horizon-certification Hold message could not be published: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
    return True


def _stop_recording_best_effort(
    *,
    recording_active: Callable[[], bool],
    stop_recording: Callable[..., Any],
    now: float,
    end_reason: str,
    summary: Mapping[str, object],
    log: Callable[..., Any],
    control_ui: Any,
    ui_tag: str,
    ui_failure_message: str,
) -> Exception | None:
    """Close a safety episode without allowing writer cleanup to resume motion.

    A writer/finalizer failure is recorded and surfaced in the GUI, but it is
    never allowed to escape a collision path.  The arm stop and Hold request
    are owned by the caller and therefore still execute after this function
    returns.
    """

    try:
        active = bool(recording_active())
    except Exception as exc:  # noqa: BLE001 - safety cleanup must remain bounded.
        active = True
        log(
            ui_tag,
            "cannot determine recording state during safety stop; treating it as active: "
            f"{type(exc).__name__}: {exc}",
            now=now,
        )
    if not active:
        return None
    try:
        stop_recording(
            now,
            end_reason=end_reason,
            summary=summary,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the safety transition.
        log(
            ui_tag,
            "safety episode stop failed; runtime remains fail-closed in Hold: "
            f"{type(exc).__name__}: {exc}",
            now=now,
        )
        if control_ui is not None:
            try:
                control_ui.append_message(ui_tag, ui_failure_message)
            except Exception as ui_exc:  # noqa: BLE001 - UI is not the safety owner.
                log(
                    ui_tag,
                    "safety stop failure message could not be published: "
                    f"{type(ui_exc).__name__}: {ui_exc}",
                    now=now,
                )
        return exc
    return None


def _consume_manual_episode_stop_request(
    *,
    clutch_input: Any,
    recording_active: Callable[[], bool],
    stop_recording: Callable[..., Any],
    cancel_recording_countdown: Callable[..., Any],
    now: float,
    step_index: int,
    episode_last_state: object,
) -> bool:
    """Consume one operator stop even while progress is fail-closed.

    A durable Failure can keep the control loop in
    ``AWAITING_HOLD_RECEIPT``.  Stopping/sealing the Episode is still a valid
    operator action in that state and must not be starved by the normal
    fail-closed ``continue``.
    """

    if not clutch_input.consume_episode_stop_request():
        return False
    if recording_active():
        stop_recording(
            now,
            end_reason="manual_key_stop",
            summary={
                "last_step_index": step_index,
                "last_state": episode_last_state,
            },
        )
    else:
        cancel_recording_countdown(now, reason="manual_key_stop")
    return True


def _request_safety_hold_best_effort(
    clutch_input: Any,
    *,
    reason: str,
    log: Callable[..., Any],
    now: float,
) -> None:
    """Latch Hold even if the normal input/UI callback raises or is a no-op.

    ``OperatorControlRuntime.request_hold`` intentionally ignores requests
    before teleop admission (while it is waiting for the first runtime Home).
    That is useful for ordinary UI events, but it is unsafe for an asynchronous
    safety fault: the next control frame must observe a durable local Hold
    request even during Calibration.  Treat a callback that returns without
    setting both local latches as a no-op and apply the same fail-closed
    fallback used for callback exceptions.
    """

    try:
        clutch_input.request_hold(reason=reason)
    except Exception as exc:  # noqa: BLE001 - safety Hold must still be latched.
        log(
            "Hold",
            "safety Hold request callback failed; forcing the local Hold latch: "
            f"{type(exc).__name__}: {exc}",
            now=now,
        )
    else:
        # The real operator-control implementation sets both fields
        # atomically.  A callback that silently returns (notably during the
        # pre-teleop Calibration wait) did not establish a consumable Hold
        # request and must be promoted to the fallback below.
        try:
            hold_requested = bool(getattr(clutch_input, "_hold_requested"))
            hold_active = bool(getattr(clutch_input, "_hold_active"))
        except Exception:
            hold_requested = False
            hold_active = False
        if hold_requested and hold_active:
            return
        log(
            "Hold",
            "safety Hold request callback returned without latching Hold; "
            "forcing the local Hold latch",
            now=now,
        )
    fallback_values = {
        "_hold_requested": True,
        "_hold_active": True,
        "_hardware_hold_confirmed": False,
        "_hold_request_reason": str(reason),
        "_teleop_dispatch_confirmed": False,
        "_teleop_dispatch_projection": "hold",
        "_gripper_control_enabled": False,
    }
    for attribute, value in fallback_values.items():
        try:
            setattr(clutch_input, attribute, value)
        except Exception as exc:  # noqa: BLE001 - report, never clear the stop latch.
            log(
                "Hold",
                f"safety Hold fallback field {attribute} could not be set: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )


def _latch_safety_hold_and_hold_hardware_best_effort(
    components: Any,
    clutch_input: Any,
    *,
    message: str,
    reason: str,
    log: Callable[..., Any],
    now: float,
    require_home_recalibration: bool = True,
) -> bool:
    """Latch the application stop before attempting the fallible Session Hold.

    The application latch is established before the fallible Session operation.
    It revokes command admission immediately, while a positive physical Hold
    receipt remains a separate fact and cannot be inferred from this helper.
    """

    _request_safety_hold_best_effort(
        clutch_input,
        reason=reason,
        log=log,
        now=now,
    )
    try:
        hold_applied = bool(
            components.session.hold_hardware(
                message,
                require_home_recalibration=require_home_recalibration,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the local safety latch is authoritative.
        log(
            "Hold",
            "safety hardware Hold failed after the local Hold was latched; "
            "motion remains blocked by the application latch: "
            f"{type(exc).__name__}: {exc}",
            now=now,
        )
        return False
    if not hold_applied:
        log(
            "Hold",
            "safety hardware Hold returned without a positive receipt; "
            "motion remains blocked by the application latch",
            now=now,
        )
    return hold_applied


def _position_force_guard_fault_pending(
    fault: Any,
    *,
    handled: bool,
) -> bool:
    """Return whether a latched force fault still needs one Hold transition.

    This predicate deliberately has no ``_teleop_started`` admission gate.
    Force monitoring is active before the first runtime Home, and a trip in
    that Calibration window must still stop the episode/countdown and retain
    the process in Hold.
    """

    return fault is not None and not bool(handled)


def _latch_unreceipted_progress_failure_hold(
    *,
    progress_failure_pending: bool,
    clutch_input: Any,
    pause_recording: Callable[..., Any],
    log: Callable[..., Any],
    now: float,
) -> bool:
    """Keep an unresolved durable Failure aligned with application Hold.

    A promoted Failure can survive process restart while the exact historical
    Hold/Pause receipt it refers to is unavailable.  The progress reducer must
    remain ``AWAITING_HOLD_RECEIPT`` in that case; a new Hold must never be
    recorded as if it were the missing historical receipt.  It must still
    revoke live arm/gripper dispatch, however, otherwise the control loop's
    fail-closed early return leaves the previous RUNNING UI projection stale.

    The distinct reason below deliberately does not match ``progress_failure``
    in the receipt-recording branch.  It is a present-time safety/application
    Hold only, not evidence for the old promoted cut.
    """

    if not bool(progress_failure_pending):
        return False
    if bool(getattr(clutch_input, "_hold_requested", False)) or bool(
        clutch_input.hold_active()
    ):
        return False
    _request_safety_hold_best_effort(
        clutch_input,
        reason="progress_failure_receipt_unavailable",
        log=log,
        now=now,
    )
    try:
        pause_recording(
            now,
            reason="progress_failure_receipt_unavailable",
        )
    except Exception as exc:  # noqa: BLE001 - the Hold remains authoritative.
        log(
            "Progress",
            "unreceipted Failure Hold was latched but formal capture pause "
            f"refresh failed: {type(exc).__name__}: {exc}",
            now=now,
        )
    return True


def _project_session_dispatch_state(
    step_result: Any,
    *,
    clutch_input: Any,
) -> bool:
    """Publish operator arm-control availability, not merely transport work.

    A dual-side Fix-Hold deliberately continues dispatching fixed joint
    commands so the lower position stream remains well defined. Those
    receipts are not evidence that live tracker control is available and must
    not reopen the gripper or formal-capture gates.
    """

    session_state_value = str(
        getattr(step_result.state, "value", step_result.state)
    ).upper()
    dispatched_arm_command = session_step_arm_dispatch_confirmed(step_result)
    fix_hold_state = getattr(clutch_input, "fix_hold_active", False)
    fix_hold_active = bool(
        fix_hold_state() if callable(fix_hold_state) else fix_hold_state
    )
    if (
        session_state_value == "RUNNING"
        and dispatched_arm_command
        and not fix_hold_active
    ):
        publish_running = getattr(
            clutch_input,
            "publish_teleop_dispatch_running",
            None,
        )
        if callable(publish_running):
            publish_running()
        return True

    startup_projection = (
        None
        if step_result.command is None
        else step_result.command.metadata.get("startup_home_continuity")
    )
    startup_reason = (
        str(startup_projection.get("reason", "") or "")
        if isinstance(startup_projection, Mapping)
        else ""
    )
    publish_suspended = getattr(
        clutch_input,
        "publish_teleop_dispatch_suspended",
        None,
    )
    if callable(publish_suspended):
        publish_suspended(
            session_state=session_state_value,
            reason=(
                "dual-side Fix-Hold owns fixed arm commands; live tracker and "
                "gripper control remain suspended"
                if fix_hold_active
                else str(step_result.motion_hold_reason or "") or startup_reason
            ),
        )
    return False


def _request_position_admittance_endpoint_hold(
    command: Any,
    *,
    clutch_input: Any,
    action_debug_log_recorder: Any,
    log: Callable[..., Any],
    now: float,
) -> bool:
    """Promote a held endpoint command into the ordinary GUI Hold flow."""

    metadata = getattr(command, "metadata", None)
    payload = (
        metadata.get("position_admittance_endpoint_hold")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(payload, Mapping):
        return False
    log(
        "PositionAdmittance",
        "endpoint envelope reached; last safe qpos dispatched and "
        "ordinary GUI Hold requested: "
        f"code={payload.get('code')} reason={payload.get('reason')}",
        now=now,
    )
    if bool(getattr(action_debug_log_recorder, "active", False)):
        action_debug_log_recorder.write_event(
            "position_admittance_endpoint_hold",
            dict(payload),
            now=now,
        )
    _request_safety_hold_best_effort(
        clutch_input,
        reason="position_admittance_endpoint_limit",
        log=log,
        now=now,
    )
    return True


def _cancel_recording_countdown_for_hold_best_effort(
    cancel_recording_countdown: Callable[..., Any],
    *,
    now: float,
    reason: str,
    log: Callable[..., Any],
) -> None:
    """Cancel a pending recording start without interrupting a Hold."""

    try:
        cancel_recording_countdown(now, reason=reason)
    except Exception as exc:  # noqa: BLE001 - Hold/stop remains authoritative.
        log(
            "Dataset",
            "recording countdown cancellation failed during Hold; "
            f"continuing with motion blocked: {type(exc).__name__}: {exc}",
            now=now,
        )


def _reconcile_idle_progress_entry_arm_control(
    *,
    profile: Any,
    selected_entry_step_id: object,
    current_fixed_sides: tuple[str, ...],
    enforcement_mode: str,
    apply_direct: Callable[[ProgressArmTransitionPlan], bool],
) -> tuple[
    ProgressArmTransitionPlan | None,
    PendingProgressArmTransition | None,
    bool,
]:
    """Apply or gate the selected entry restriction before Episode start.

    The final boolean reports whether the exact target is ready.  Adding a
    fixed side can briefly lack the first live command snapshot at startup;
    that case must keep Episode admission closed while allowing one ordinary
    control frame to populate the snapshot, rather than entering Home/Hold.
    """

    plan = build_progress_entry_arm_transition_plan(
        profile=profile,
        selected_entry_step_id=selected_entry_step_id,
        current_fixed_sides=current_fixed_sides,
        enforcement_mode=enforcement_mode,
    )
    if plan is None or not plan.changes_arm_control:
        return plan, None, True
    if apply_direct(plan):
        return plan, None, True
    if plan.requires_home:
        return (
            plan,
            PendingProgressArmTransition(
                intent=None,
                plan=plan,
                capture_sides=plan.newly_teleop_sides,
            ),
            False,
        )
    return plan, None, False


def _apply_pending_progress_arm_target_after_home(
    *,
    components: Any,
    pending: PendingProgressArmTransition,
    now: float,
) -> tuple[bool, str]:
    """Stage the profile's exact Fix authority while the Session is held.

    The preceding Hold deliberately removes active Fix authority, but retains
    the last accepted command/channel snapshots for a profile transition.  A
    target of no fixed sides is just as meaningful as a single fixed side and
    must be verified explicitly before the progress mutation can commit.
    """

    controller = getattr(components, "fix_arm_controller", None)
    if controller is None:
        return False, "FixArm controller unavailable"
    target_fixed = tuple(pending.plan.target_fixed_sides)
    actual_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
    if actual_fixed == target_fixed:
        return True, "target Fix authority already staged"

    cancel_active = getattr(controller, "cancel_active", None)
    if actual_fixed:
        if not callable(cancel_active):
            return False, "FixArm controller cannot clear prior authority"
        transition = cancel_active(
            now=now,
            reason="progress arm target replacement",
            clear_snapshots=False,
        )
        if not bool(getattr(transition, "ok", False)):
            return False, str(
                getattr(transition, "message", "prior Fix authority release failed")
            )

    if not target_fixed:
        actual_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
        if actual_fixed:
            return False, f"expected live/live but fixed_sides={actual_fixed!r}"
        return True, "live/live authority staged"
    if len(target_fixed) != 1:
        return False, f"unsupported target fixed_sides={target_fixed!r}"

    select_side = getattr(controller, "select_side", None)
    freeze_selected = getattr(controller, "freeze_selected", None)
    if not callable(select_side) or not callable(freeze_selected):
        return False, "FixArm controller cannot establish target authority"
    transition = select_side(target_fixed[0])
    if bool(getattr(transition, "ok", False)):
        transition = freeze_selected(now=now)
    if not bool(getattr(transition, "ok", False)):
        return False, str(getattr(transition, "message", "target Fix failed"))
    actual_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
    if actual_fixed != target_fixed:
        return False, (
            f"expected fixed_sides={target_fixed!r} but actual={actual_fixed!r}"
        )
    return True, "target Fix authority staged"


@dataclass(frozen=True)
class RealTeleopControlLoopContext:
    action_debug_log_recorder: Any
    actuator_trace_recorder: Any
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
    guard_impedance_force_control: Callable[..., bool]
    handle_ik_debug_log_toggle_request: Callable[..., None]
    ik_debug_log_recorder: Any
    ik_debug_profile_base_enabled: bool
    idle_tactile_compression_tick: Callable[..., None]
    begin_idle_tactile_quiesce: Callable[[str], None]
    require_idle_tactile_quiesced: Callable[[str], None]
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
    position_admittance_shadow: Any
    preview_camera_sources: Mapping[str, Any]
    publish_collection_frequency_warning: Callable[..., None]
    record_debug_event: Callable[..., bool]
    record_debug_frame: Callable[..., bool]
    recording_active: Callable[[], bool]
    recording_auto_stop_reason: Callable[[Any], str | None]
    recording_frame_count: Callable[[], int]
    recording_frequency_status: Callable[[], Mapping[str, object]]
    recording_path: Callable[[], str]
    recording_pause_reason: Callable[[], str]
    recording_paused: Callable[[], bool]
    recording_warmup_status: Callable[..., tuple[bool, str]]
    real_hand_render_qpos: Callable[[Any, str], np.ndarray]
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
    collection_transition_active: Callable[[], bool] = lambda: False
    runtime_extension: object | None = None


def run_real_teleop_control_loop(context: RealTeleopControlLoopContext) -> None:
    action_debug_log_recorder = context.action_debug_log_recorder
    actuator_trace_recorder = context.actuator_trace_recorder
    advance_recording_countdown = context.advance_recording_countdown
    args = context.args
    camera_preview_rate = context.camera_preview_rate
    camera_sources = context.camera_sources
    cancel_recording_countdown = context.cancel_recording_countdown
    close_camera_preview = context.close_camera_preview
    close_tactile_preview = context.close_tactile_preview
    clutch_input = context.clutch_input
    components = context.components
    runtime_extension = context.runtime_extension
    console_stream = context.console_stream
    control_ui = context.control_ui
    debug_timing_log_recorder = context.debug_timing_log_recorder
    debug_timing_modules = context.debug_timing_modules
    debug_timing_state = context.debug_timing_state
    episode_recorder = context.episode_recorder
    fix_hold_coordinator = context.fix_hold_coordinator
    gripper_warmup_probe_running = context.gripper_warmup_probe_running
    _guard_impedance_force_control = context.guard_impedance_force_control
    _handle_ik_debug_log_toggle_request = (
        context.handle_ik_debug_log_toggle_request
    )
    ik_debug_log_recorder = context.ik_debug_log_recorder
    debug_ik_0827_enabled = bool(getattr(args, "debug_ik_0827", False))
    ik_debug_profile_base_enabled = context.ik_debug_profile_base_enabled
    idle_tactile_compression_tick = context.idle_tactile_compression_tick
    begin_idle_tactile_quiesce = context.begin_idle_tactile_quiesce
    require_idle_tactile_quiesced = context.require_idle_tactile_quiesced
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
    position_admittance_shadow = context.position_admittance_shadow
    preview_camera_sources = context.preview_camera_sources
    publish_collection_frequency_warning = context.publish_collection_frequency_warning
    record_debug_event = context.record_debug_event
    record_debug_frame = context.record_debug_frame
    recording_active = context.recording_active
    recording_auto_stop_reason = context.recording_auto_stop_reason
    recording_frame_count = context.recording_frame_count
    recording_frequency_status = context.recording_frequency_status
    recording_path = context.recording_path
    recording_pause_reason = context.recording_pause_reason
    recording_paused = context.recording_paused
    recording_warmup_status = context.recording_warmup_status
    _real_hand_render_qpos = context.real_hand_render_qpos
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
    last_startup_home_continuity_key: tuple[int, str] | None = None
    progress_workflow: RealTeleopProgressWorkflowPort | None = None
    progress_workflow_recorder: object | None = None
    promotable_hold_pause: PromotableHoldPause | None = None
    pending_progress_arm_transition: PendingProgressArmTransition | None = None
    progress_arm_entry_snapshot_wait_step_id = ""
    progress_arm_entry_episode_hash = ""
    next_idle_tactile_compression_tick_at = 0.0
    agent_motion_port = getattr(components, "agent_motion_port", None)
    agent_motion_last_phase = (
        None
        if agent_motion_port is None
        else agent_motion_port.snapshot().phase.value
    )

    def synchronize_agent_motion_owner_state(
        snapshot: object,
        *,
        now: float,
    ) -> None:
        """Mirror a newly terminal lower Agent state into the normal UI Hold."""

        nonlocal agent_motion_last_phase
        phase = str(getattr(getattr(snapshot, "phase", None), "value", ""))
        previous = agent_motion_last_phase
        agent_motion_last_phase = phase
        if phase not in {"held", "faulted"} or phase == previous:
            return
        reason = (
            "agent_motion_owner_fault"
            if phase == "faulted"
            else "agent_motion_owner_hold"
        )
        _request_safety_hold_best_effort(
            clutch_input,
            reason=reason,
            log=_log,
            now=now,
        )

    def reset_position_admittance_shadow(
        *,
        sides: Sequence[str] | None,
        reason: str,
        now: float,
    ) -> bool:
        if position_admittance_shadow is None:
            return True
        active_admittance = bool(
            getattr(position_admittance_shadow, "apply_to_target", False)
        )
        try:
            position_admittance_shadow.reset_from_monitor(sides=sides)
        except Exception as exc:  # noqa: BLE001 - freeze; existing safety remains authoritative.
            try:
                position_admittance_shadow.freeze_all("tare_reset_failed")
            except Exception as freeze_exc:  # noqa: BLE001 - Hold must still be latched.
                _log(
                    "PositionAdmittance",
                    "tare reset failure could not freeze the admittance observer: "
                    f"{type(freeze_exc).__name__}: {freeze_exc}",
                    now=now,
                )
            message = (
                "position admittance tare reset failed: "
                f"{type(exc).__name__}: {exc}"
            )
            _log(
                "PositionAdmittance",
                "tare reset failed; admittance frozen: " + message,
                now=now,
            )
            if active_admittance:
                _latch_safety_hold_and_hold_hardware_best_effort(
                    components,
                    clutch_input,
                    message=message,
                    reason="position_admittance_tare_reset_failed",
                    log=_log,
                    now=now,
                    require_home_recalibration=True,
                )
                return False
            return True
        try:
            admittance_mode = position_admittance_shadow.snapshot().get(
                "mode",
                "shadow",
            )
        except Exception as exc:  # noqa: BLE001 - status projection is observational.
            admittance_mode = "active" if active_admittance else "shadow"
            _log(
                "PositionAdmittance",
                "tare reset succeeded but status snapshot failed: "
                f"{type(exc).__name__}: {exc}",
                now=now,
            )
        _log(
            "PositionAdmittance",
            f"tare reset mode={admittance_mode} after {reason}; sides="
            f"{list(components.active_sides if sides is None else sides)}",
            now=now,
        )
        return True

    def rearm_position_admittance_after_control_plane_pause(
        *,
        started_at: float,
        reason: str,
        now: float,
    ) -> bool:
        """Re-arm Active force clocks after a measured runtime pause.

        Dataset/tactile startup is a named control-plane transition, so its
        blocking duration is known even though it currently runs in this
        thread.  Only pauses above the explicit threshold use the timing-only
        re-arm; an unrelated sensor gap still reaches the normal
        ``force_dt_exceeded`` fail-closed path.
        """

        elapsed_sec = max(0.0, time.perf_counter() - float(started_at))
        if elapsed_sec < _POSITION_ADMITTANCE_CONTROL_PLANE_REARM_THRESHOLD_SEC:
            return True
        shadow = position_admittance_shadow
        rearm = (
            None
            if shadow is None
            else getattr(shadow, "rearm_after_control_plane_pause", None)
        )
        if not callable(rearm):
            return True
        try:
            result = rearm(
                now_sec=time.monotonic(),
                pause_duration_sec=elapsed_sec,
                reason=str(reason),
            )
        except Exception as exc:  # noqa: BLE001 - Active mode remains fail-closed.
            message = (
                "position admittance control-plane re-arm failed: "
                f"{type(exc).__name__}: {exc}"
            )
            _log("PositionAdmittance", message, now=now)
            if bool(getattr(shadow, "apply_to_target", False)):
                _latch_safety_hold_and_hold_hardware_best_effort(
                    components,
                    clutch_input,
                    message=message,
                    reason="position_admittance_control_plane_rearm_failed",
                    log=_log,
                    now=now,
                    require_home_recalibration=True,
                )
                return False
            return True
        rearmed_sides = tuple(
            str(side)
            for side in (
                result.get("sides", ())
                if isinstance(result, Mapping)
                else ()
            )
        )
        if rearmed_sides:
            _log(
                "PositionAdmittance",
                "rearmed force sample clocks after known control-plane pause "
                f"reason={reason} elapsed={elapsed_sec * 1000.0:.1f}ms "
                f"sides={list(rearmed_sides)}",
                now=now,
            )
        return True

    def apply_progress_arm_plan_direct(
        plan: ProgressArmTransitionPlan,
        *,
        now: float,
    ) -> bool:
        """Apply one exact target, or report that runtime input is not ready."""

        controller = getattr(components, "fix_arm_controller", None)
        if controller is None:
            raise RuntimeError(
                "A progress arm_control declaration requires the FixArm controller."
            )
        current_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
        target_fixed = plan.target_fixed_sides
        if current_fixed == target_fixed:
            return True
        for side in target_fixed:
            scope = str(getattr(controller, "scope_for_side")(side))
            if scope != plan.requirement.fixed_scope:
                raise RuntimeError(
                    "Profile fixed_hold requires arm_and_end_effector but the "
                    f"runtime FixArm scope is {scope!r} for {side}."
                )
        newly_teleop = tuple(
            side for side in current_fixed if side not in set(target_fixed)
        )
        if newly_teleop:
            cache_statuses = tuple(
                controller.passive_home_cache_status(
                    side,
                    now=now,
                    allow_fresh_rebase=True,
                )
                for side in newly_teleop
            )
            if not all(bool(status.get("ready", False)) for status in cache_statuses):
                _log(
                    "Progress",
                    "profile arm transition has no fresh target-side Home; "
                    f"statuses={list(cache_statuses)}",
                    now=now,
                )
                return False
            if len(current_fixed) != 1 or len(target_fixed) > 1:
                raise RuntimeError(
                    "Cached profile arm switching only supports the existing "
                    "bimanual/single-arm Fix states."
                )
            target_side = target_fixed[0] if target_fixed else None
            transition = controller.start_two_pedal_transition(
                fixed_side=target_side,
                now=now,
                timeout_sec=1.0,
                min_samples=1,
                allow_fresh_home_rebase=True,
            )
            if not transition.ok:
                raise RuntimeError(transition.message)
            if any(
                str(status.get("quality", "")) == "fresh_rebase"
                for status in cache_statuses
            ):
                _log(
                    "Progress",
                    "profile arm transition used an instantaneous fresh "
                    "target-side tracker rebase; no global Hold/Pause required",
                    now=now,
                )
            home_progress = advance_fix_arm_home_capture(
                controller=controller,
                channels={},
                calibration=components.session.calibration,
                now=now,
                pose_from_channel=_channel_pose_in_desk,
                mean_pose=_mean_pose,
                resolve_robot_home=lambda side: _resolve_robot_home_for_side(
                    components,
                    side=side,
                ),
                side_retargeter=lambda side: _side_retargeter(components, side),
                tcp_mode=args.tcp_mode,
                log=lambda tag, message: _log(tag, message, now=now),
            )
            if not home_progress.completed:
                if home_progress.failed:
                    raise RuntimeError(home_progress.message)
                return False
            components.session.safety_monitor.request_pose_baseline_reset_many(
                tuple(f"{side}_wrist_pose" for side in newly_teleop)
            )
        elif target_fixed:
            target_side = target_fixed[0]
            transition = controller.select_side(target_side)
            if transition.ok:
                transition = controller.freeze_selected(now=now)
            if not transition.ok:
                if tuple(getattr(transition, "missing_effectors", ()) or ()) or tuple(
                    getattr(transition, "missing_channels", ()) or ()
                ):
                    return False
                raise RuntimeError(transition.message)
        else:
            raise RuntimeError(
                "Unexpected profile arm transition without a newly live arm."
            )
        actual_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
        if actual_fixed != target_fixed:
            raise RuntimeError(
                "Profile arm transition did not reach its exact target: "
                f"expected={target_fixed!r} actual={actual_fixed!r}."
            )
        return True

    main_loop_watchdog.start()
    emergency_stop_sent = False
    force_guard_fault_handled = False

    def current_promotable_hold_pause() -> PromotableHoldPause | None:
        """Return only the still-current, same-Episode ordinary Hold cut."""

        candidate = promotable_hold_pause
        episode = (
            None
            if dataset_recorder is None
            else getattr(
                dataset_recorder,
                "active_progress_episode",
                None,
            )
        )
        identity_hash = str(
            getattr(
                None if episode is None else episode.identity,
                "semantic_content_hash",
                "",
            )
        )
        pause_started_at = (
            None
            if dataset_recorder is None
            else dataset_recorder.pause_started_at
        )
        if (
            candidate is None
            or progress_workflow_recorder is not dataset_recorder
            or episode is None
            or episode.state.phase
            not in {"ACTIVE_ATTEMPT", "AWAITING_HOLD_RECEIPT"}
            or identity_hash != candidate.episode_capture_identity_hash
            or dataset_recorder is None
            or not dataset_recorder.is_recording
            or not dataset_recorder.is_paused
            or pause_started_at is None
            or pause_started_at
            != candidate.formal_capture_pause_effective_at
            or not clutch_input.hold_active()
        ):
            return None
        return candidate

    def publish_progress_ui_state(*, now: float) -> None:
        """Publish display-only progress state without affecting control."""

        if context.collection_transition_active():
            return
        publisher = getattr(
            clutch_input,
            "publish_progress_application_state",
            None,
        )
        if not callable(publisher):
            return
        try:
            publisher(
                dataset_recorder=dataset_recorder,
                progress_workflow=progress_workflow,
                promotable_hold_pause=current_promotable_hold_pause(),
                arm_transition_pending=(
                    pending_progress_arm_transition is not None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - UI is observational only.
            _log(
                "Progress",
                "长程进度 GUI 状态发布失败；控制与数采语义不受影响："
                f"{type(exc).__name__}: {exc}",
                now=now,
            )

    def consume_position_force_guard_fault(
        fault: Any,
        *,
        now: float,
    ) -> bool:
        """Consume one latched position-force fault in the normal Hold path."""

        nonlocal force_guard_fault_handled
        if force_guard_fault_handled:
            # The monitor callback is the authoritative Arm_Stop owner.  A
            # dispatch race may reach this function after the episode was
            # already closed; keep the latch and simply prevent another stop
            # or writer finalization attempt.
            _request_safety_hold_best_effort(
                clutch_input,
                reason="position_force_guard_fault",
                log=_log,
                now=now,
            )
            return False
        force_guard_fault_handled = True
        _cancel_recording_countdown_for_hold_best_effort(
            cancel_recording_countdown,
            now=now,
            reason="position_force_guard",
            log=_log,
        )
        recording_end_reason, recording_summary = _force_guard_recording_stop_context(
            fault,
            fallback_end_reason="position_force_guard",
            fallback_summary={
                "last_step_index": step_index,
                "last_state": episode_last_state,
            },
        )
        _stop_recording_best_effort(
            recording_active=recording_active,
            stop_recording=stop_recording,
            now=now,
            end_reason=recording_end_reason,
            summary=recording_summary,
            log=_log,
            control_ui=control_ui,
            ui_tag="PositionForceGuard",
            ui_failure_message=(
                "碰撞数采收尾失败，但机械臂仍保持急停；"
                "请检查数据目录后再重新采集。"
            ),
        )
        fault_reason = str(getattr(fault, "reason", "unknown"))
        collision_fault = fault_reason == "force_limit_exceeded"
        hold_reason = (
            "position_force_guard_collision"
            if collision_fault
            else "position_force_guard_fault"
        )
        _request_safety_hold_best_effort(
            clutch_input,
            reason=hold_reason,
            log=_log,
            now=now,
        )
        _log(
            "PositionForceGuard",
            "active teleop fault consumed; GUI/devices retained in Hold "
            f"reason={fault_reason}",
            now=now,
        )
        if control_ui is not None:
            try:
                if collision_fault:
                    message = (
                        "检测到受力碰撞：数采已停止并保留碰撞标签；"
                        "GUI/设备连接未退出，请人工调整位置后点击“开始遥操”重新采集 Home。"
                    )
                else:
                    message = (
                        "位控受力安全监控触发停机：数采已停止；"
                        "GUI/设备连接保持，但需检查设备后重新采集 runtime Home。"
                    )
                control_ui.append_message("PositionForceGuard", message)
            except Exception as exc:  # noqa: BLE001 - UI cannot change safety state.
                _log(
                    "PositionForceGuard",
                    "fault Hold message failed: "
                    f"{type(exc).__name__}: {exc}",
                    now=now,
                )
        return True

    while True:
        loop_start = time.time()
        dataset_recorder = context.dataset_recorder()
        if loop_start >= next_idle_tactile_compression_tick_at:
            idle_tactile_compression_tick(
                episode_countdown_active=(
                    context.episode_start_countdown_deadline() is not None
                )
            )
            next_idle_tactile_compression_tick_at = loop_start + 0.2
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
        if runtime_extension is not None:
            after_control_input_poll = getattr(
                runtime_extension,
                "after_control_input_poll",
                None,
            )
            if callable(after_control_input_poll):
                after_control_input_poll(now=loop_start)
                mark_loop_phase("loop.runtime_extension_after_control_input")
        consume_profile_selections = getattr(
            clutch_input,
            "consume_progress_profile_selection_intents",
            None,
        )
        profile_selection_intents = (
            ()
            if not callable(consume_profile_selections)
            else consume_profile_selections()
        )
        consume_entry_step_selections = getattr(
            clutch_input,
            "consume_progress_entry_step_selection_intents",
            None,
        )
        entry_step_selection_intents = (
            ()
            if not callable(consume_entry_step_selections)
            else consume_entry_step_selections()
        )
        consume_arm_mode_selections = getattr(
            clutch_input,
            "consume_progress_arm_control_mode_selection_intents",
            None,
        )
        arm_mode_selection_intents = (
            ()
            if not callable(consume_arm_mode_selections)
            else consume_arm_mode_selections()
        )
        for selection_intent in arm_mode_selection_intents:
            try:
                if context.collection_transition_active():
                    raise RuntimeError("Collection lifecycle transition is active")
                if bool(
                    getattr(clutch_input, "_episode_busy", lambda: True)()
                ):
                    raise RuntimeError(
                        "Episode countdown, recording, pause, or finalization "
                        "is active"
                    )
                args.collection_progress_arm_control_mode = (
                    selection_intent.mode
                )
                if control_ui is not None:
                    control_ui.append_message(
                        "Progress",
                        "已选择长程进度臂控制模式："
                        f"{selection_intent.mode}。",
                    )
            except Exception as exc:  # noqa: BLE001 - no control effect.
                message = (
                    "长程进度臂控制模式选择失败，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Progress", message, now=loop_start)
                if control_ui is not None:
                    control_ui.append_message("Progress", message)
        for selection_intent in profile_selection_intents:
            try:
                if context.collection_transition_active():
                    raise RuntimeError("Collection lifecycle transition is active")
                if not bool(
                    getattr(args, "collection_progress_gui", False)
                ):
                    raise RuntimeError(
                        "collection progress GUI selection is disabled"
                    )
                if bool(
                    getattr(clutch_input, "_episode_busy", lambda: True)()
                ):
                    raise RuntimeError(
                        "Episode countdown, recording, pause, or finalization "
                        "is active"
                    )
                current_recorder = context.dataset_recorder()
                select_profile = getattr(
                    clutch_input,
                    "select_managed_progress_profile",
                    None,
                )
                if not callable(select_profile):
                    raise RuntimeError(
                        "Progress application profile selection is unavailable."
                    )
                selected_profile = select_profile(
                    dataset_recorder=current_recorder,
                    profile_id=selection_intent.profile_id,
                )
                if control_ui is not None:
                    control_ui.append_message(
                        "Progress",
                        "已选择长程进度 profile："
                        f"{selected_profile.profile_id}@"
                        f"{selected_profile.profile_version}；"
                        "首个 Episode 启动时将按完整 hash 锁定到数据集。",
                    )
            except Exception as exc:  # noqa: BLE001 - selection has no control effect.
                message = (
                    "长程进度 profile 选择失败，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Progress", message, now=loop_start)
                if control_ui is not None:
                    control_ui.append_message("Progress", message)
        for selection_intent in entry_step_selection_intents:
            try:
                if context.collection_transition_active():
                    raise RuntimeError("Collection lifecycle transition is active")
                if pending_progress_arm_transition is not None:
                    raise RuntimeError(
                        "A profile arm transition is already waiting for "
                        "runtime Home."
                    )
                if bool(
                    getattr(clutch_input, "_episode_busy", lambda: True)()
                ):
                    raise RuntimeError(
                        "Episode countdown, recording, pause, or finalization "
                        "is active"
                    )
                select_entry_step = getattr(
                    clutch_input,
                    "select_progress_entry_step",
                    None,
                )
                if not callable(select_entry_step):
                    raise RuntimeError(
                        "Progress application entry-step selection is unavailable."
                    )
                selected_step_id = select_entry_step(
                    dataset_recorder=context.dataset_recorder(),
                    progress_step_id=selection_intent.progress_step_id,
                )
                if control_ui is not None:
                    control_ui.append_message(
                        "Progress",
                        "下一个 Episode 将从子任务开始："
                        f"{selected_step_id}；之后沿用正常推进与标注流程。",
                    )
            except Exception as exc:  # noqa: BLE001 - selection has no control effect.
                message = (
                    "长程进度起始子任务选择失败，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Progress", message, now=loop_start)
                if control_ui is not None:
                    control_ui.append_message("Progress", message)
        collection_transition_pending = context.collection_transition_active()
        progress_configured = bool(
            dataset_recorder is not None
            and getattr(dataset_recorder, "progress_profile", None) is not None
        )
        if not progress_configured:
            progress_workflow = None
            progress_workflow_recorder = None
            promotable_hold_pause = None
            pending_progress_arm_transition = None
            progress_arm_entry_snapshot_wait_step_id = ""
            progress_arm_entry_episode_hash = ""
        elif progress_workflow_recorder is not dataset_recorder:
            # A recorder identity change invalidates the old workflow and its
            # pending arm transaction.  Build the new binding only after the
            # collection control-plane has published a stable recorder.
            progress_workflow = None
            progress_workflow_recorder = None
            promotable_hold_pause = None
            pending_progress_arm_transition = None
            progress_arm_entry_snapshot_wait_step_id = ""
            progress_arm_entry_episode_hash = ""
            if not collection_transition_pending:
                build_workflow = getattr(
                    clutch_input,
                    "build_progress_workflow",
                    None,
                )
                if not callable(build_workflow):
                    raise RuntimeError(
                        "Progress application workflow binding is unavailable."
                    )
                progress_workflow = build_workflow(dataset_recorder)
                progress_workflow_recorder = dataset_recorder
        progress_failure_pending = False
        if progress_workflow is not None and not collection_transition_pending:
            progress_callbacks = {
                "request_application_hold": (
                    lambda reason: clutch_input.request_hold(reason=reason)
                ),
                "pause_formal_capture": (
                    lambda at, reason: pause_recording(at, reason=reason)
                ),
            }
            consume_progress_commands = getattr(
                clutch_input,
                "consume_progress_commands",
                None,
            )
            progress_intents = (
                ()
                if not callable(consume_progress_commands)
                else consume_progress_commands()
            )
            try:
                active_progress_episode = getattr(
                    dataset_recorder,
                    "active_progress_episode",
                    None,
                )
                if (
                    active_progress_episode is None
                    and pending_progress_arm_transition is None
                ):
                    fix_controller = getattr(
                        components,
                        "fix_arm_controller",
                        None,
                    )
                    (
                        idle_entry_plan,
                        idle_entry_pending,
                        idle_entry_ready,
                    ) = _reconcile_idle_progress_entry_arm_control(
                        profile=dataset_recorder.progress_profile,
                        selected_entry_step_id=getattr(
                            args,
                            "collection_progress_entry_step",
                            "",
                        ),
                        current_fixed_sides=tuple(
                            getattr(fix_controller, "fixed_sides", ()) or ()
                        ),
                        enforcement_mode=str(
                            getattr(
                                args,
                                "collection_progress_arm_control_mode",
                                "profile",
                            )
                        ),
                        apply_direct=lambda plan: apply_progress_arm_plan_direct(
                            plan,
                            now=loop_start,
                        ),
                    )
                    if (
                        idle_entry_plan is not None
                        and idle_entry_plan.changes_arm_control
                    ):
                        if idle_entry_ready:
                            progress_arm_entry_snapshot_wait_step_id = ""
                            if control_ui is not None:
                                control_ui.append_message(
                                    "Progress",
                                    "已在 Episode 启动前应用所选起始子任务的 "
                                    "arm+end-effector Hold 限制："
                                    f"{idle_entry_plan.target_progress_step_id}。",
                                )
                        elif idle_entry_pending is not None:
                            progress_arm_entry_snapshot_wait_step_id = ""
                            pending_progress_arm_transition = idle_entry_pending
                            clutch_input.request_hold(
                                reason="progress_arm_profile_transition"
                            )
                            if control_ui is not None:
                                control_ui.append_message(
                                    "Progress",
                                    "所选起始子任务需要切换活动臂；已在 Episode "
                                    "启动前进入 Hold。请完成目标活动臂 runtime Home，"
                                    "限制同步完成后再开始数采。",
                                )
                            publish_progress_ui_state(now=time.time())
                            continue
                        elif (
                            progress_arm_entry_snapshot_wait_step_id
                            != idle_entry_plan.target_progress_step_id
                        ):
                            progress_arm_entry_snapshot_wait_step_id = (
                                idle_entry_plan.target_progress_step_id
                            )
                            if (
                                context.episode_start_countdown_deadline()
                                is not None
                            ):
                                cancel_recording_countdown(
                                    loop_start,
                                    reason="progress_entry_arm_snapshot_pending",
                                )
                            message = (
                                "所选起始子任务的固定臂正在等待首帧 arm/hand "
                                "控制快照；快照建立前不会启动 Episode。"
                            )
                            _log("Progress", message, now=loop_start)
                            if control_ui is not None:
                                control_ui.append_message("Progress", message)
                    elif idle_entry_ready:
                        progress_arm_entry_snapshot_wait_step_id = ""
                active_identity_hash = str(
                    getattr(
                        None
                        if active_progress_episode is None
                        else active_progress_episode.identity,
                        "semantic_content_hash",
                        "",
                    )
                )
                if (
                    active_progress_episode is not None
                    and active_progress_episode.state.phase == "ACTIVE_ATTEMPT"
                    and active_identity_hash
                    and active_identity_hash != progress_arm_entry_episode_hash
                    and pending_progress_arm_transition is None
                ):
                    entry_step_id = (
                        active_progress_episode.state.current_progress_step_id
                    )
                    assert entry_step_id is not None
                    fix_controller = getattr(
                        components,
                        "fix_arm_controller",
                        None,
                    )
                    entry_plan = build_progress_arm_transition_plan(
                        profile=active_progress_episode.profile,
                        target_progress_step_id=entry_step_id,
                        current_fixed_sides=tuple(
                            getattr(fix_controller, "fixed_sides", ()) or ()
                        ),
                        enforcement_mode=str(
                            getattr(
                                args,
                                "collection_progress_arm_control_mode",
                                "profile",
                            )
                        ),
                    )
                    if entry_plan is None or not entry_plan.changes_arm_control:
                        progress_arm_entry_episode_hash = active_identity_hash
                    elif apply_progress_arm_plan_direct(
                        entry_plan,
                        now=loop_start,
                    ):
                        progress_arm_entry_episode_hash = active_identity_hash
                    else:
                        pending_progress_arm_transition = (
                            PendingProgressArmTransition(
                                intent=None,
                                plan=entry_plan,
                                capture_sides=entry_plan.newly_teleop_sides,
                            )
                        )
                        pause_recording(
                            loop_start,
                            reason="progress_arm_profile_entry_home_required",
                        )
                        clutch_input.request_hold(
                            reason="progress_arm_profile_transition"
                        )
                for progress_intent in progress_intents:
                    if pending_progress_arm_transition is not None:
                        raise RuntimeError(
                            "A profile arm transition is already waiting for "
                            "runtime Home."
                        )
                    episode = getattr(
                        dataset_recorder,
                        "active_progress_episode",
                        None,
                    )
                    mode = str(
                        getattr(
                            args,
                            "collection_progress_arm_control_mode",
                            "profile",
                        )
                    )
                    plan = None
                    undo_profile_arm_change = False
                    if (
                        episode is not None
                        and mode != PROGRESS_ARM_CONTROL_MODE_IGNORE
                        and progress_intent.kind
                        in {"ADVANCE_PROGRESS", "UNDO_LAST_ADVANCE"}
                    ):
                        target_step_id = (
                            episode.preview_advance_target(
                                progress_intent.target_progress_step_id
                            )
                            if progress_intent.kind == "ADVANCE_PROGRESS"
                            else episode.preview_undo_target()
                        )
                        fix_controller = getattr(
                            components,
                            "fix_arm_controller",
                            None,
                        )
                        plan = build_progress_arm_transition_plan(
                            profile=episode.profile,
                            target_progress_step_id=target_step_id,
                            current_fixed_sides=tuple(
                                getattr(fix_controller, "fixed_sides", ()) or ()
                            ),
                            enforcement_mode=mode,
                        )
                        if (
                            progress_intent.kind == "UNDO_LAST_ADVANCE"
                            and plan is not None
                        ):
                            source_step_id = (
                                episode.state.current_progress_step_id
                            )
                            undo_profile_arm_change = bool(
                                source_step_id is not None
                                and progress_step_arm_requirements_differ(
                                    episode.profile,
                                    source_step_id,
                                    target_step_id,
                                )
                            )
                    force_hold_for_undo = bool(
                        plan is not None
                        and progress_intent.kind == "UNDO_LAST_ADVANCE"
                        and undo_profile_arm_change
                    )
                    direct_applied = bool(
                        plan is None
                        or (
                            not force_hold_for_undo
                            and (
                                not plan.changes_arm_control
                                or apply_progress_arm_plan_direct(
                                    plan,
                                    now=loop_start,
                                )
                            )
                        )
                    )
                    if not direct_applied:
                        assert plan is not None
                        capture_sides = (
                            plan.newly_teleop_sides
                            if plan.newly_teleop_sides
                            else plan.requirement.teleop_sides
                        )
                        pending_progress_arm_transition = (
                            PendingProgressArmTransition(
                                intent=progress_intent,
                                plan=plan,
                                capture_sides=capture_sides,
                            )
                        )
                        pause_recording(
                            loop_start,
                            reason="progress_arm_profile_transition_home_required",
                        )
                        clutch_input.request_hold(
                            reason="progress_arm_profile_transition"
                        )
                        if control_ui is not None:
                            control_ui.append_message(
                                "Progress",
                                "目标子任务需要切换活动臂；已把 Hold/Pause "
                                "保留在当前子任务。请摆好目标活动臂 VIVE 后"
                                "点击“开始遥操”。",
                            )
                        continue
                    progress_result = progress_workflow.handle_command(
                        progress_intent,
                        now=loop_start,
                        application_held=clutch_input.hold_active(),
                        promotable_hold_pause=(
                            current_promotable_hold_pause()
                        ),
                        **progress_callbacks,
                    )
                    if control_ui is not None and progress_result.handled:
                        control_ui.append_message(
                            "Progress",
                            "采集进度请求已生效："
                            f"{progress_intent.kind} → {progress_result.phase}",
                        )
                pending_result = progress_workflow.request_pending_failure_effect(
                    now=loop_start,
                    promotable_hold_pause=current_promotable_hold_pause(),
                    **progress_callbacks,
                )
                progress_failure_pending = pending_result.phase == (
                    "AWAITING_HOLD_RECEIPT"
                )
            except Exception as exc:  # noqa: BLE001 - preserve collection fail-close.
                message = (
                    "collection progress 请求处理失败，数采保持暂停并进入 Hold："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Progress", message, now=loop_start)
                if control_ui is not None:
                    control_ui.append_message("Progress", message)
                clutch_input.request_hold(reason="progress_workflow_error")
                pause_recording(loop_start, reason="progress_workflow_error")
                progress_failure_pending = True
        _latch_unreceipted_progress_failure_hold(
            progress_failure_pending=progress_failure_pending,
            clutch_input=clutch_input,
            pause_recording=pause_recording,
            log=_log,
            now=loop_start,
        )
        publish_progress_ui_state(now=loop_start)
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
        # A force-guard trip during an admitted teleop session is a runtime
        # safety transition, not a process-lifecycle failure.  The monitor
        # callback has already latched both command backends and issued the
        # bimanual Arm_Stop.  Close the active episode with the collision
        # summary, request the normal GUI Hold path, and leave the cameras,
        # trackers, hands, and GUI alive for a fresh operator-led Home.
        force_guard_fault = (
            None
            if position_force_guard_monitor is None
            else position_force_guard_monitor.fault
        )
        if force_guard_fault is None:
            # A successful operator-led runtime Home re-arms the monitor;
            # allow the next independent collision in the same process to be
            # consumed exactly once as well.
            force_guard_fault_handled = False
        if _position_force_guard_fault_pending(
            force_guard_fault,
            handled=force_guard_fault_handled,
        ):
            consume_position_force_guard_fault(force_guard_fault, now=loop_start)
            continue
        if clutch_input.should_quit:
            if not emergency_stop_sent:
                try:
                    _emergency_stop_real_arms(
                        components,
                        reason="quit_requested",
                        now=loop_start,
                    )
                except Exception as exc:  # noqa: BLE001 - quit remains fail-closed.
                    _log(
                        "EmergencyStop",
                        "quit emergency stop reported a recoverable cleanup error; "
                        f"runtime will still terminate safely: {type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
            emergency_stop_sent = True
            _cancel_recording_countdown_for_hold_best_effort(
                cancel_recording_countdown,
                now=loop_start,
                reason="quit_requested",
                log=_log,
            )
            force_guard_fault = (
                None
                if position_force_guard_monitor is None
                else position_force_guard_monitor.fault
            )
            recording_end_reason, recording_summary = _force_guard_recording_stop_context(
                force_guard_fault,
                fallback_end_reason="quit_requested",
                fallback_summary={
                    "last_step_index": step_index,
                    "last_state": episode_last_state,
                },
            )
            _stop_recording_best_effort(
                recording_active=recording_active,
                stop_recording=stop_recording,
                now=loop_start,
                end_reason=recording_end_reason,
                summary=recording_summary,
                log=_log,
                control_ui=control_ui,
                ui_tag="Dataset",
                ui_failure_message=(
                    "退出前数采收尾失败；机械臂保持停机，请检查数据目录。"
                ),
            )
            break
        if _consume_manual_episode_stop_request(
            clutch_input=clutch_input,
            recording_active=recording_active,
            stop_recording=stop_recording,
            cancel_recording_countdown=cancel_recording_countdown,
            now=loop_start,
            step_index=step_index,
            episode_last_state=episode_last_state,
        ):
            # A pending Failure outbox may re-issue Hold on every frame.  The
            # Episode stop must win this frame so it cannot be starved by the
            # Hold branch's fail-closed continue.  No control command is
            # dispatched; an unconsumed Hold remains latched for the next
            # frame.
            continue
        if clutch_input.consume_hold_request():
            hold_reason = clutch_input.hold_request_reason()
            promotable_hold_pause = None
            application_hold_effective_at: float | None = None
            application_hold_was_active = bool(
                components.session.home_recalibration_required
            )
            formal_capture_was_paused = bool(
                dataset_recorder is not None
                and dataset_recorder.is_paused
            )
            # Preserve the Session's historical manual-Hold contract while
            # marking hardware/safety cuts as requiring an explicit new Home.
            # Error containment below applies to every Hold independently of
            # this narrower Session lifecycle distinction.
            requires_session_home_recalibration = hold_reason in {
                "position_force_guard_collision",
                "position_force_guard_fault",
                "impedance_force_collision",
                "emergency_stop_latched",
                "joint_hard_limit",
                "joint_horizon_certification",
                "position_admittance_endpoint_limit",
            }
            hold_pause_error: Exception | None = None
            fix_hold_escalation = fix_hold_coordinator.escalate_to_regular_hold()
            if fix_hold_escalation.changed:
                try:
                    clutch_input.set_fix_hold_active(
                        False,
                        message=(
                            "Fix-Hold 已由用户升级为常规 Hold；"
                            "特殊固定状态将清除，之后必须重采双臂 Home。"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - Hold remains fail-closed.
                    _log(
                        "Hold",
                        "Fix-Hold UI transition failed during Hold: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
                if episode_recorder.is_recording:
                    try:
                        record_debug_event(
                            "fix_hold_escalated_to_regular_hold",
                            payload={
                                "step_index": int(step_index),
                                "locked_sides": list(fix_hold_escalation.locked_sides),
                                "source_by_side": dict(
                                    fix_hold_escalation.source_by_side
                                ),
                            },
                            now=loop_start,
                        )
                    except Exception as exc:  # noqa: BLE001 - diagnostics cannot reopen motion.
                        _log(
                            "Hold",
                            "Fix-Hold episode event failed during Hold: "
                            f"{type(exc).__name__}: {exc}",
                            now=loop_start,
                        )
            if hold_reason == "q6_gesture":
                hold_hardware_reason = "q6 gesture hold requested"
            elif hold_reason == "tracker_loss_timeout":
                hold_hardware_reason = "tracker target loss requires runtime home recalibration"
            elif hold_reason == "position_force_guard_collision":
                hold_hardware_reason = (
                    "position force guard collision emergency stop; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "position_force_guard_fault":
                hold_hardware_reason = (
                    "position force guard safety fault; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "impedance_force_collision":
                hold_hardware_reason = (
                    "impedance force collision emergency stop; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "emergency_stop_latched":
                hold_hardware_reason = (
                    "emergency stop latch observed during command dispatch; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "joint_horizon_certification":
                hold_hardware_reason = (
                    "joint horizon candidate could not be certified; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "joint_hard_limit":
                hold_hardware_reason = (
                    "RM65 joint hard limit rejected the command; "
                    "runtime home recalibration required"
                )
            elif hold_reason == "position_admittance_endpoint_limit":
                hold_hardware_reason = (
                    "position admittance endpoint envelope reached; "
                    "holding the last safe qpos until runtime home recalibration"
                )
            elif hold_reason == "progress_failure":
                hold_hardware_reason = (
                    "collection progress failure reported; "
                    "waiting for operator adjustment and runtime home recalibration"
                )
            elif hold_reason == "progress_arm_profile_transition":
                hold_hardware_reason = (
                    "collection progress arm profile requires a newly active arm; "
                    "waiting for target-side runtime home calibration"
                )
            elif hold_reason.startswith("init_qpos_"):
                hold_hardware_reason = (
                    "Init qpos operation completed; runtime home recalibration required"
                    if hold_reason in {
                        "init_qpos_completed",
                        "init_qpos_sequence_completed",
                    }
                    else "Init qpos operation failed; safe Hold and runtime home recalibration required"
                )
            elif hold_reason in {"", "manual", "two_pedal_b", "two_pedal_c"}:
                hold_hardware_reason = "gui hold requested"
            else:
                hold_hardware_reason = (
                    f"runtime Hold requested ({hold_reason}); "
                    "runtime home recalibration required"
                )
            if recording_active():
                try:
                    pause_recording(
                        loop_start,
                        reason=_hold_requested_pause_reason(hold_reason),
                    )
                except Exception as exc:  # noqa: BLE001 - Hold must still complete.
                    hold_pause_error = exc
                    _log(
                        "Dataset",
                        "Hold could not pause the active episode; "
                        f"continuing with arm stop: {type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
                    if control_ui is not None:
                        try:
                            control_ui.append_message(
                                "Dataset",
                                "Hold 时数采暂停失败；机械臂仍将停机，当前 episode 将按失败收尾。",
                            )
                        except Exception as ui_exc:  # noqa: BLE001 - UI cannot change safety state.
                            _log(
                                "Dataset",
                                "safety pause failure message could not be published: "
                                f"{type(ui_exc).__name__}: {ui_exc}",
                                now=loop_start,
                            )
            try:
                special_cancel_messages = _cancel_special_arm_modes_for_hold(
                    components,
                    now=loop_start,
                    reason="hold_requested",
                    # A profile transition freezes the currently-live peer
                    # side at this exact Hold cut after Home. Preserve that
                    # last accepted arm+end-effector snapshot so no ordinary
                    # command is needed before the new Fix is established.
                    clear_fix_snapshots=not (
                        hold_reason == "progress_arm_profile_transition"
                        and pending_progress_arm_transition is not None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - Hold remains fail-closed.
                special_cancel_messages = [
                    "特殊控制模式释放异常，已保持停机"
                ]
                _log(
                    "Hold",
                    "special arm mode cancellation failed during Hold; "
                    f"motion remains blocked: {type(exc).__name__}: {exc}",
                    now=loop_start,
                )
            _cancel_recording_countdown_for_hold_best_effort(
                cancel_recording_countdown,
                now=loop_start,
                reason="hold_requested",
                log=_log,
            )
            recovery_plan = None
            hold_applied = False
            try:
                hold_applied = bool(
                    components.session.hold_hardware(
                        hold_hardware_reason,
                        require_home_recalibration=(
                            requires_session_home_recalibration
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - stay fail-closed in GUI.
                _log(
                    "Hold",
                    "Hold hardware transition failed; local dispatch remains blocked: "
                    f"{type(exc).__name__}: {exc}",
                    now=loop_start,
                )
                if control_ui is not None:
                    try:
                        control_ui.append_message(
                            "Hold",
                            "Hold 下发存在异常；已禁止继续遥操，请检查机械臂状态并重启后再恢复。",
                        )
                    except Exception as ui_exc:  # noqa: BLE001 - UI cannot change safety state.
                        _log(
                            "Hold",
                            "Hold failure message could not be published: "
                            f"{type(ui_exc).__name__}: {ui_exc}",
                            now=loop_start,
                        )
            position_streams_disengaged = False
            try:
                _disengage_position_backends_for_manual_interaction(
                    components,
                    now=loop_start,
                    reason=hold_hardware_reason,
                )
                position_streams_disengaged = True
            except Exception as exc:  # noqa: BLE001 - Hold remains fail-closed.
                _log(
                    "Hold",
                    "Hold position-stream disengage failed; "
                    f"motion remains blocked: {type(exc).__name__}: {exc}",
                    now=loop_start,
                )
                if control_ui is not None:
                    try:
                        control_ui.append_message(
                            "Hold",
                            "Hold 控制流释放存在异常；已禁止继续遥操，请检查设备并重启。",
                        )
                    except Exception as ui_exc:  # noqa: BLE001 - UI cannot change safety state.
                        _log(
                            "Hold",
                            "safety stream-release failure message could not be published: "
                            f"{type(ui_exc).__name__}: {ui_exc}",
                            now=loop_start,
                        )
                try:
                    components.session.fault(
                        "position command streams could not be disengaged after Hold: "
                        f"{type(exc).__name__}: {exc}"
                    )
                except Exception as fault_exc:  # noqa: BLE001 - application latch remains closed.
                    _log(
                        "Hold",
                        "Session fault projection failed after position-stream release error: "
                        f"{type(fault_exc).__name__}: {fault_exc}",
                        now=loop_start,
                    )
            session_state = getattr(
                getattr(components.session, "state_machine", None),
                "state",
                None,
            )
            session_state_value = str(
                getattr(session_state, "value", session_state) or ""
            ).upper()
            hardware_hold_confirmed = bool(
                hold_applied
                and position_streams_disengaged
                and session_state_value not in {"FAULT", "ESTOP"}
            )
            if hardware_hold_confirmed:
                application_hold_effective_at = time.time()
            publish_hold_confirmation = getattr(
                clutch_input,
                "publish_hardware_hold_confirmation",
                None,
            )
            if callable(publish_hold_confirmation):
                publish_hold_confirmation(
                    confirmed=hardware_hold_confirmed,
                    message=(
                        "底层 Hold 与位置命令流释放均已确认；可以重采 Home 或执行维护操作。"
                        if hardware_hold_confirmed
                        else (
                            "应用 Hold 已锁存，但底层停机或位置命令流释放未获完整确认；"
                            "禁止恢复遥操和维护动作，请检查设备并重启。"
                        )
                    ),
                )
            if hold_pause_error is not None:
                _stop_recording_best_effort(
                    recording_active=recording_active,
                    stop_recording=stop_recording,
                    now=loop_start,
                    end_reason="hold_pause_failure",
                    summary={
                        "last_step_index": int(step_index),
                        "last_state": episode_last_state,
                        "hold_reason": str(hold_reason),
                        "pause_error": (
                            f"{type(hold_pause_error).__name__}: "
                            f"{hold_pause_error}"
                        ),
                    },
                    log=_log,
                    control_ui=control_ui,
                    ui_tag="Dataset",
                    ui_failure_message=(
                        "Hold 时暂停与停止当前 episode 均失败；机械臂保持停机，"
                        "请检查 writer 和数据目录后重启。"
                    ),
                )
            _log(
                "Hold",
                f"{hold_hardware_reason}; waiting for explicit runtime home calibration",
                now=loop_start,
            )
            if control_ui is not None:
                if not hardware_hold_confirmed:
                    message = (
                        "应用 Hold 已生效，但机械臂停机回执不完整；"
                        "不要执行 Home、设备重连或维护运动，请检查设备并重启。"
                    )
                elif hold_reason == "q6_gesture":
                    message = "机械臂已进入保持状态；松开双手后三维到释放区间 1 秒，再次握紧 1 秒后会重新采集 runtime home。"
                elif hold_reason == "tracker_loss_timeout":
                    message = "Tracker 目标持续丢失已进入手动 Hold；必须点击“开始遥操”重新采集 runtime home 后才能恢复。"
                elif hold_reason == "position_force_guard_collision":
                    message = (
                        "受力碰撞已急停双臂并结束当前数采；GUI 与设备连接保持。"
                        "请人工调整机械臂位置后点击“开始遥操”重新采集 runtime Home。"
                    )
                elif hold_reason == "position_force_guard_fault":
                    message = (
                        "位控受力安全监控已急停双臂并结束当前数采；"
                        "请检查设备后点击“开始遥操”重新采集 runtime Home。"
                    )
                elif hold_reason == "impedance_force_collision":
                    message = (
                        "力控碰撞已急停双臂并结束当前数采；GUI 与设备连接保持。"
                        "请人工调整机械臂位置后点击“开始遥操”重新采集 runtime Home。"
                    )
                elif hold_reason == "emergency_stop_latched":
                    message = (
                        "检测到急停与控制帧竞态，当前数采已结束并进入 Hold；"
                        "请检查机械臂后重新采集 runtime Home。"
                    )
                elif hold_reason == "joint_hard_limit":
                    message = (
                        "机械臂关节硬限位门禁已拒绝命令并急停双臂；当前数采已停止。"
                        "请人工检查并调整机械臂后重新采集 runtime Home。"
                    )
                elif hold_reason == "position_admittance_endpoint_limit":
                    message = (
                        "力控末端已到达位移或旋转包络，已保持最后安全位置；"
                        "进程与设备连接保持。确认姿态后点击“开始遥操”重新采集 runtime Home。"
                    )
                elif hold_reason == "progress_failure":
                    message = (
                        "当前阶段失败已记录，正式数采已暂停且机械臂进入 Hold。"
                        "人工调整后点击“开始遥操”重新采集 runtime Home；"
                        "writer 接纳恢复后的首帧时会创建同阶段 Retry Attempt。"
                    )
                elif hold_reason == "progress_arm_profile_transition":
                    message = (
                        "当前子任务的正式数采边界已暂停，目标子任务尚未提交。"
                        "请摆好目标活动臂 VIVE 后点击“开始遥操”；Home 与目标 Fix "
                        "状态建立成功后才会切换子任务并继续写入。"
                    )
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
                    message = (
                        "机械臂已进入保持状态；点击“开始遥操”后才会重新采集 "
                        f"runtime home。原因：{hold_reason or 'manual'}。"
                    )
                try:
                    control_ui.append_message("Hold", message)
                    if special_cancel_messages:
                        control_ui.append_message(
                            "Hold",
                            "；".join(special_cancel_messages)
                            + "；重新采集 runtime home 后恢复普通遥操。",
                        )
                except Exception as exc:  # noqa: BLE001 - UI cannot change safety state.
                    _log(
                        "Hold",
                        "Hold message could not be published: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
            if (
                progress_workflow is not None
                and not context.collection_transition_active()
                and progress_failure_pending
                and hold_reason == "progress_failure"
                and application_hold_effective_at is not None
                and dataset_recorder is not None
                and dataset_recorder.is_paused
                and dataset_recorder.pause_started_at is not None
            ):
                try:
                    receipt_result = (
                        progress_workflow.record_pending_failure_effect_receipt(
                            application_hold_effective_at=(
                                application_hold_effective_at
                            ),
                            formal_capture_pause_effective_at=(
                                dataset_recorder.pause_started_at
                            ),
                            available_at=time.time(),
                        )
                    )
                    progress_failure_pending = receipt_result.phase == (
                        "AWAITING_HOLD_RECEIPT"
                    )
                    publish_progress_ui_state(now=time.time())
                except Exception as exc:  # noqa: BLE001 - remain held and paused.
                    _log(
                        "Progress",
                        "Hold/Pause 已执行但 progress receipt 写入失败；"
                        "保持 Hold/Pause 并等待 durable outbox 重放："
                        f"{type(exc).__name__}: {exc}",
                        now=time.time(),
                    )
            active_progress_episode = (
                None
                if dataset_recorder is None
                else getattr(
                    dataset_recorder,
                    "active_progress_episode",
                    None,
                )
            )
            candidate = _build_promotable_progress_hold_pause(
                hold_reason=hold_reason,
                application_hold_was_active=application_hold_was_active,
                formal_capture_was_paused=formal_capture_was_paused,
                hold_requested_at=loop_start,
                application_hold_effective_at=(
                    application_hold_effective_at
                ),
                formal_capture_pause_effective_at=(
                    None
                    if dataset_recorder is None
                    or not dataset_recorder.is_recording
                    or not dataset_recorder.is_paused
                    else dataset_recorder.pause_started_at
                ),
                episode_capture_identity_hash=(
                    ""
                    if active_progress_episode is None
                    else getattr(
                        active_progress_episode.identity,
                        "semantic_content_hash",
                        "",
                    )
                ),
                progress_phase=(
                    ""
                    if active_progress_episode is None
                    else active_progress_episode.state.phase
                ),
            )
            if candidate is not None:
                promotable_hold_pause = candidate
                if control_ui is not None:
                    control_ui.append_message(
                        "Progress",
                        "本次普通 Hold 可直接提升为“当前阶段失败并 Hold”；"
                        "失败边界将沿用原暂停时刻，不包含后续人工调整。",
                    )
            publish_progress_ui_state(now=time.time())
            continue
        if progress_failure_pending:
            # The Failure fact is durable, but no authoritative application
            # Hold receipt exists yet. Never dispatch a normal control frame.
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
                require_idle_tactile_quiesced("two_pedal_fast_runtime_home")
                fast_home_qpos_snapshot = _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=clutch_input.two_pedal_fast_home_timeout_sec(),
                    fresh_data_timeout_sec=runtime_home_fresh_data_timeout_sec(
                        clutch_input.two_pedal_fast_home_timeout_sec()
                    ),
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
                    atomic_robot_home_snapshot=bool(
                        getattr(args, "runtime_home_atomic_arm_snapshot", True)
                    ),
                    late_weighted_human_home=bool(
                        getattr(args, "runtime_home_late_weighted_vive", True)
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                message = (
                    "双踏板切换失败：1 秒内未获得稳定、同步的 runtime Home；"
                    f"{type(exc).__name__}: {exc}"
                )
                _latch_safety_hold_and_hold_hardware_best_effort(
                    components,
                    clutch_input,
                    message=message,
                    reason="two_pedal_fast_runtime_home_failed",
                    log=_log,
                    now=loop_start,
                    require_home_recalibration=True,
                )
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
            arm_admitted = _arm_after_start_calibration(
                args,
                components,
                clutch_input,
                home_captured=True,
                position_force_guard_monitor=position_force_guard_monitor,
                runtime_home_qpos_snapshot=(
                    fast_home_qpos_snapshot
                    if bool(getattr(args, "runtime_home_atomic_arm_snapshot", True))
                    else None
                ),
                runtime_home_capture_sides=(
                    (fast_two_pedal_fixed_side,)
                    if fast_two_pedal_fixed_side is not None
                    else None
                ),
            )
            if not arm_admitted:
                clutch_input.finish_two_pedal_fast_home(
                    success=False,
                    now=loop_start,
                    message=(
                        "双踏板快速 Home 已采集，但机械臂安全接入未完成；"
                        "保持 Hold，未提交目标遥操状态，请检查提示后重试。"
                    ),
                )
                continue
            if not reset_position_admittance_shadow(
                sides=(
                    (fast_two_pedal_fixed_side,)
                    if fast_two_pedal_fixed_side is not None
                    else None
                ),
                reason="two_pedal_fast_runtime_home",
                now=loop_start,
            ):
                clutch_input.finish_two_pedal_fast_home(
                    success=False,
                    now=loop_start,
                    message=(
                        "双踏板快速 Home 已采集，但 Active 位置导纳 tare 复位失败；"
                        "保持 Hold，未提交目标遥操状态，请重新 Home。"
                    ),
                )
                continue
            clutch_input.finish_two_pedal_fast_home(
                success=True,
                now=loop_start,
                message="双踏板快速 Home 已完成，正在无缝接入目标遥操状态。",
            )
            continue
        if clutch_input.consume_calibration_request():
            profile_home_pending = pending_progress_arm_transition
            if recording_active():
                pause_recording(
                    loop_start,
                    reason=(
                        "progress_arm_profile_transition_home"
                        if profile_home_pending is not None
                        else "runtime_recalibration_requested"
                    ),
                )
            special_cancel_messages = _cancel_special_arm_modes_for_hold(
                components,
                now=loop_start,
                reason="runtime_recalibration_requested",
                clear_fix_snapshots=profile_home_pending is None,
            )
            cancel_recording_countdown(loop_start, reason="runtime_recalibration_requested")
            recovery_plan = None
            if components.session.state_machine.state != TeleopRunState.DEBUG_ONLY:
                try:
                    runtime_home_hold_applied = bool(
                        components.session.hold_hardware(
                            "runtime home recalibration requested"
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - return to durable calibration Hold.
                    runtime_home_hold_applied = False
                    runtime_home_hold_failure = f"{type(exc).__name__}: {exc}"
                else:
                    runtime_home_hold_failure = "negative Hold receipt"
                if not runtime_home_hold_applied:
                    message = (
                        "runtime Home 前无法确认机械臂 Hold；本次恢复已取消："
                        f"{runtime_home_hold_failure}"
                    )
                    _log("Calibration", message, now=loop_start)
                    _request_safety_hold_best_effort(
                        clutch_input,
                        reason="runtime_home_prehold_failed",
                        log=_log,
                        now=loop_start,
                    )
                    try:
                        clutch_input.publish_calibration_waiting(message)
                    except Exception as ui_exc:  # noqa: BLE001 - local Hold is already latched.
                        _log(
                            "Calibration",
                            "runtime Home pre-Hold failure UI publish failed: "
                            f"{type(ui_exc).__name__}: {ui_exc}",
                            now=loop_start,
                        )
                    continue
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
                    + (
                        "；本次 runtime Home 后将按目标子任务建立精确 Fix 状态。"
                        if profile_home_pending is not None
                        else "；本次 runtime home 后特殊状态保持关闭。"
                    ),
                )
            try:
                begin_idle_tactile_quiesce("explicit_runtime_home")
            except Exception as exc:  # noqa: BLE001 - remain in Hold.
                message = (
                    "runtime Home 前无法停止触觉后台压缩："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("TactileCompression", message, now=loop_start)
                clutch_input.publish_calibration_waiting(
                    message + "；保持停臂，请重试。"
                )
                continue
            if not clutch_input.before_calibration_start_delay(
                DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                arm_mode="real",
            ):
                continue
            try:
                require_idle_tactile_quiesced("explicit_runtime_home")
                runtime_home_qpos_snapshot = _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=args.calibration_timeout,
                    fresh_data_timeout_sec=runtime_home_fresh_data_timeout_sec(
                        args.calibration_timeout
                    ),
                    num_samples=args.calibration_samples,
                    progress_callback=clutch_input.publish_runtime_home_progress,
                    capture_sides=(
                        profile_home_pending.capture_sides
                        if profile_home_pending is not None
                        else None
                    ),
                    require_stable=(profile_home_pending is not None),
                    stability_profile=(
                        "transition"
                        if profile_home_pending is not None
                        else "strict"
                    ),
                    atomic_robot_home_snapshot=bool(
                        getattr(args, "runtime_home_atomic_arm_snapshot", True)
                    ),
                    late_weighted_human_home=bool(
                        getattr(args, "runtime_home_late_weighted_vive", True)
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                message = _runtime_home_failure_message(exc)
                _log("Calibration", message, now=loop_start)
                clutch_input.publish_calibration_waiting(
                    f"{message}\n本次恢复遥操已取消，机械臂保持 Hold；"
                    "请检查 VIVE sender/tracker_role 后重新点击“开始遥操”。"
                )
                if profile_home_pending is not None:
                    clutch_input.request_hold(
                        reason="progress_arm_profile_transition"
                    )
                continue
            if profile_home_pending is not None:
                progress_transaction_blocker = (
                    "collection lifecycle transition active"
                    if context.collection_transition_active()
                    else (
                        "progress workflow unavailable"
                        if (
                            profile_home_pending.intent is not None
                            and progress_workflow is None
                        )
                        else ""
                    )
                )
                if progress_transaction_blocker:
                    message = (
                        "目标活动臂 Home 已采集，但 progress mutation 当前不可提交；"
                        "保持原 progress 与 Hold/Pause："
                        f"{progress_transaction_blocker}。"
                    )
                    _log("Progress", message, now=loop_start)
                    if control_ui is not None:
                        try:
                            control_ui.append_message("Progress", message)
                        except Exception as exc:  # noqa: BLE001 - UI is observational.
                            _log(
                                "Progress",
                                "progress transaction blocker UI publish failed: "
                                f"{type(exc).__name__}: {exc}",
                                now=loop_start,
                            )
                    clutch_input.request_hold(
                        reason="progress_arm_profile_transition"
                    )
                    continue
                arm_target_ready, arm_target_detail = (
                    _apply_pending_progress_arm_target_after_home(
                        components=components,
                        pending=profile_home_pending,
                        now=loop_start,
                    )
                )
                if not arm_target_ready:
                    message = (
                        "目标子任务 Home 完成后无法从 Hold 边界快照建立精确 "
                        "arm+end-effector 权限；保持原 progress 与 Hold/Pause："
                        f"{arm_target_detail}"
                    )
                    _log("Progress", message, now=loop_start)
                    if control_ui is not None:
                        control_ui.append_message("Progress", message)
                    clutch_input.request_hold(
                        reason="progress_arm_profile_transition"
                    )
                    continue
            _emit_home_alignment_logs(components)
            arm_admitted = _arm_after_start_calibration(
                args,
                components,
                clutch_input,
                home_captured=True,
                position_force_guard_monitor=position_force_guard_monitor,
                runtime_home_qpos_snapshot=(
                    runtime_home_qpos_snapshot
                    if bool(getattr(args, "runtime_home_atomic_arm_snapshot", True))
                    else None
                ),
                runtime_home_capture_sides=(
                    profile_home_pending.capture_sides
                    if profile_home_pending is not None
                    else None
                ),
            )
            if not arm_admitted:
                if profile_home_pending is not None:
                    message = (
                        "目标活动臂 Home 已采集，但机械臂安全接入未完成；"
                        "保持原 progress 与 Hold/Pause，本次切换未提交。"
                    )
                    _log("Progress", message, now=loop_start)
                    if control_ui is not None:
                        control_ui.append_message("Progress", message)
                    clutch_input.request_hold(
                        reason="progress_arm_profile_transition"
                    )
                continue
            if not reset_position_admittance_shadow(
                sides=(
                    profile_home_pending.capture_sides
                    if profile_home_pending is not None
                    else None
                ),
                reason="explicit_runtime_home",
                now=loop_start,
            ):
                if profile_home_pending is not None:
                    _log(
                        "Progress",
                        "目标活动臂 Home 已采集，但 Active 位置导纳 tare 复位失败；"
                        "保持原 progress 与 Hold/Pause，本次切换未提交。",
                        now=loop_start,
                    )
                continue
            if profile_home_pending is not None:
                if profile_home_pending.intent is not None:
                    assert progress_workflow is not None
                    try:
                        progress_result = progress_workflow.handle_command(
                            profile_home_pending.intent,
                            now=time.time(),
                            application_held=False,
                            promotable_hold_pause=None,
                            **progress_callbacks,
                        )
                    except Exception as exc:  # noqa: BLE001 - idempotent intent remains pending.
                        message = (
                            "目标机械臂权限已建立，但 progress mutation 未获得提交回执；"
                            "重新进入 Hold/Pause 并保留同一 intent："
                            f"{type(exc).__name__}: {exc}"
                        )
                        _log("Progress", message, now=loop_start)
                        if control_ui is not None:
                            try:
                                control_ui.append_message("Progress", message)
                            except Exception as ui_exc:  # noqa: BLE001 - preserve transaction failure.
                                _log(
                                    "Progress",
                                    "progress failure UI publish failed: "
                                    f"{type(ui_exc).__name__}: {ui_exc}",
                                    now=loop_start,
                                )
                        clutch_input.request_hold(
                            reason="progress_arm_profile_transition"
                        )
                        continue
                    if not bool(progress_result.handled):
                        message = (
                            "目标机械臂权限已建立，但 progress workflow 未确认处理；"
                            "保持同一 intent 与 Hold/Pause，本次切换未提交："
                            f"phase={progress_result.phase} "
                            f"message={progress_result.message}"
                        )
                        _log("Progress", message, now=loop_start)
                        if control_ui is not None:
                            try:
                                control_ui.append_message("Progress", message)
                            except Exception as exc:  # noqa: BLE001 - UI is observational.
                                _log(
                                    "Progress",
                                    "unhandled progress receipt UI publish failed: "
                                    f"{type(exc).__name__}: {exc}",
                                    now=loop_start,
                                )
                        clutch_input.request_hold(
                            reason="progress_arm_profile_transition"
                        )
                        continue
                    if control_ui is not None:
                        target_label = (
                            "目标 arm+end-effector Fix 与活动臂 Home"
                            if profile_home_pending.plan.target_fixed_sides
                            else "目标活动臂 Home"
                        )
                        try:
                            control_ui.append_message(
                                "Progress",
                                f"{target_label}已建立，子任务切换已提交："
                                f"{progress_result.phase}。",
                            )
                        except Exception as exc:  # noqa: BLE001 - commit is authoritative.
                            _log(
                                "Progress",
                                "committed progress UI publish failed: "
                                f"{type(exc).__name__}: {exc}",
                                now=loop_start,
                            )
                active_episode = getattr(
                    dataset_recorder,
                    "active_progress_episode",
                    None,
                )
                progress_arm_entry_episode_hash = str(
                    getattr(
                        None
                        if active_episode is None
                        else active_episode.identity,
                        "semantic_content_hash",
                        "",
                    )
                )
                pending_progress_arm_transition = None
            continue
        episode_delete_index = clutch_input.consume_episode_delete_request()
        if episode_delete_index is not None:
            plus_binding_blocker = (
                clutch_input.plus_curation.persisted_binding_lock_reason()
            )
            if plus_binding_blocker:
                clutch_input.set_episode_finalize_state(
                    status="error",
                    message=f"删除失败：{plus_binding_blocker}",
                )
            elif clutch_input.plus_curation_save_in_flight():
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
                    require_idle_tactile_quiesced("episode_delete")
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
                if progress_arm_entry_snapshot_wait_step_id:
                    message = (
                        "数采开始被拒绝：所选起始子任务的固定臂尚未建立首帧 "
                        "arm/hand 控制快照，请等待 GUI 提示限制已应用后重试。"
                    )
                    _log("Progress", message, now=loop_start)
                    if control_ui is not None:
                        control_ui.append_message("Progress", message)
                    continue
                # schedule_recording_countdown() transfers the authoritative
                # tactile-maintenance quiesce, dataset contract check, and
                # writer ready handshake to the collection control plane.
                # The former duplicate synchronous quiesce waited for that
                # process from the realtime arm loop before the same check ran
                # again in prepare_writer().
                schedule_recording_countdown(
                    loop_start,
                    start_reason="manual_key_start",
                )
        mark_loop_phase("loop.keys")

        # Control-plane work above may legitimately wait for a writer RPC,
        # filesystem quiescence, or a GUI-triggered reconfiguration.  VIVE is
        # read causally at-or-before ``now``; reusing the timestamp captured
        # before such work can therefore exclude every packet that arrived
        # during the pause and falsely report both trackers missing.  Refresh
        # the authoritative control-frame time immediately before polling the
        # session.  This does not mask a real tracker outage: if no fresh VIVE
        # packet exists at the refreshed cutoff, the unchanged tracker-loss
        # state machine still enters automatic/manual Hold.
        control_step_now = _fresh_control_step_timestamp(loop_start)
        pre_step_control_plane_elapsed = max(0.0, control_step_now - loop_start)
        if pre_step_control_plane_elapsed >= 0.05:
            _log(
                "ControlTiming",
                "pre-step control-plane work delayed the realtime frame "
                f"elapsed={pre_step_control_plane_elapsed * 1000.0:.1f}ms; "
                "refreshed session causal timestamp",
                now=control_step_now,
            )
        loop_start = max(float(loop_start), float(control_step_now))

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

        impedance_collision_consumed = {"value": False}

        def handle_impedance_force_collision(reason: str) -> None:
            """Consume a controller hard-stop without tearing down runtime."""

            if impedance_collision_consumed["value"]:
                return
            impedance_collision_consumed["value"] = True
            collision_summary = {
                "last_step_index": int(step_index),
                "last_state": episode_last_state,
                "reason": "force_collision",
                "force_collision_detected": True,
                "force_collision_label": FORCE_COLLISION_LABEL_DETECTED,
                "force_collision_event": {
                    "detected": True,
                    "source": "controller_hard_stop",
                    "control_mode": str(
                        getattr(args, "arm_control_mode", "impedance")
                    ),
                    "reason": "force_limit_exceeded",
                    "message": str(reason),
                },
            }
            _cancel_recording_countdown_for_hold_best_effort(
                cancel_recording_countdown,
                now=loop_start,
                reason="force_collision",
                log=_log,
            )
            stop_errors: dict[str, str] = {}
            try:
                stop_errors = _emergency_stop_real_arms(
                    components,
                    reason="controller_hard_stop:force_collision",
                    now=loop_start,
                    raise_on_failure=False,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed on stop acknowledgement.
                stop_errors = {"__all__": f"{type(exc).__name__}: {exc}"}
            if stop_errors:
                collision_summary["emergency_stop_errors"] = dict(stop_errors)
            _stop_recording_best_effort(
                recording_active=recording_active,
                stop_recording=stop_recording,
                now=loop_start,
                end_reason="force_collision",
                summary=collision_summary,
                log=_log,
                control_ui=control_ui,
                ui_tag="ForceControl",
                ui_failure_message=(
                    "碰撞数采收尾失败；机械臂保持急停，请检查数据目录。"
                ),
            )
            if stop_errors and control_ui is not None:
                try:
                    control_ui.append_message(
                        "ForceControl",
                        "力控碰撞已触发，但至少一侧 Arm_Stop 未确认；"
                        "已保持 Hold，禁止继续运动；必须检查设备后再决定是否重启。",
                    )
                except Exception as exc:  # noqa: BLE001 - UI cannot change the stop latch.
                    _log(
                        "ForceControl",
                        "Arm_Stop failure message could not be published: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )
            _request_safety_hold_best_effort(
                clutch_input,
                reason="impedance_force_collision",
                log=_log,
                now=loop_start,
            )
            _log(
                "ForceControl",
                "active impedance force collision consumed; GUI/devices retained in Hold "
                f"reason={reason}",
                now=loop_start,
            )
            if control_ui is not None:
                try:
                    control_ui.append_message(
                        "ForceControl",
                        "力控碰撞已急停双臂并结束当前数采；GUI/设备连接保持。"
                        "请人工调整位置后点击“开始遥操”重新采集 runtime Home。",
                    )
                except Exception as exc:  # noqa: BLE001 - UI cannot change the stop latch.
                    _log(
                        "ForceControl",
                        "collision Hold message could not be published: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )

        impedance_guard_ok = _guard_impedance_force_control(
            args,
            components,
            context="main_loop",
            now=loop_start,
            collision_handler=(
                handle_impedance_force_collision
                if bool(getattr(clutch_input, "_teleop_started", False))
                else None
            ),
        )
        mark_loop_phase("loop.impedance_guard")
        if not impedance_guard_ok and impedance_collision_consumed["value"]:
            # The collision handler has already issued Arm_Stop and closed the
            # active episode.  Let the next iteration consume the normal GUI
            # Hold request; no command or hand/gripper dispatch may follow this
            # frame.
            continue

        def consume_latched_safety_dispatch(error: BaseException) -> None:
            """Close an episode after a backend rejected a raced command."""

            _cancel_recording_countdown_for_hold_best_effort(
                cancel_recording_countdown,
                now=loop_start,
                reason="emergency_stop_latched",
                log=_log,
            )
            safety_summary: dict[str, object] = {
                "reason": "emergency_stop_latched",
                "safety_stop_error": " <- ".join(_exception_chain_text(error)),
            }
            stop_errors: dict[str, str] = {}
            try:
                stop_errors = _emergency_stop_real_arms(
                    components,
                    reason="dispatch_safety_race",
                    now=loop_start,
                    raise_on_failure=False,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the fail-closed path.
                stop_errors = {"__all__": f"{type(exc).__name__}: {exc}"}
            if stop_errors:
                safety_summary["emergency_stop_errors"] = dict(stop_errors)
            _stop_recording_best_effort(
                recording_active=recording_active,
                stop_recording=stop_recording,
                now=loop_start,
                end_reason="safety_stop",
                summary=safety_summary,
                log=_log,
                control_ui=control_ui,
                ui_tag="EmergencyStop",
                ui_failure_message=(
                    "急停竞态下数采收尾失败；机械臂保持停机，请检查数据目录。"
                ),
            )
            _request_safety_hold_best_effort(
                clutch_input,
                reason="emergency_stop_latched",
                log=_log,
                now=loop_start,
            )
            _log(
                "EmergencyStop",
                "raced command was rejected after an emergency stop latch; "
                "episode closed and runtime retained in Hold: "
                + " <- ".join(_exception_chain_text(error)),
                now=loop_start,
            )
            if control_ui is not None:
                try:
                    control_ui.append_message(
                        "EmergencyStop",
                        "检测到急停与控制帧竞态：当前数采已停止，GUI/设备连接保持在 Hold；"
                        "请检查机械臂后重新采集 runtime Home。",
                    )
                except Exception as exc:  # noqa: BLE001 - UI cannot change safety state.
                    _log(
                        "EmergencyStop",
                        "safety Hold message could not be published: "
                        f"{type(exc).__name__}: {exc}",
                        now=loop_start,
                    )

        def handle_safety_dispatch_exception(error: BaseException) -> bool:
            """Route classified safety rejections; leave other failures visible."""

            force_guard_fault_now = (
                None
                if position_force_guard_monitor is None
                else position_force_guard_monitor.fault
            )
            if _position_force_guard_fault_pending(
                force_guard_fault_now,
                handled=force_guard_fault_handled,
            ):
                consume_position_force_guard_fault(
                    force_guard_fault_now,
                    now=loop_start,
                )
                return True
            exception_text = _exception_chain_text(error)
            if has_force_limit_collision_text(exception_text):
                handle_impedance_force_collision(
                    "dispatch rejected after controller force collision: "
                    + " <- ".join(exception_text)
                )
                return True
            # Hard-limit/horizon rejections are recoverable application Hold
            # transitions only after the operator has completed the initial
            # real-teleop admission.  Before that boundary, keep surfacing the
            # underlying startup/calibration failure.  This predicate used to
            # be assigned in the force-guard branch; keeping it local to the
            # dispatch classifier prevents the exceptional path from referring
            # to stale or unbound per-frame state.
            teleop_admitted = bool(
                getattr(clutch_input, "_teleop_started", False)
            )
            if teleop_admitted and _consume_joint_hard_limit_dispatch(
                error,
                components=components,
                clutch_input=clutch_input,
                recording_active=recording_active,
                stop_recording=stop_recording,
                cancel_recording_countdown=cancel_recording_countdown,
                emergency_stop_real_arms=_emergency_stop_real_arms,
                now=loop_start,
                step_index=step_index,
                last_state=episode_last_state,
                log=_log,
                control_ui=control_ui,
            ):
                return True
            if teleop_admitted and _consume_joint_horizon_certification_dispatch(
                error,
                clutch_input=clutch_input,
                recording_active=recording_active,
                stop_recording=stop_recording,
                cancel_recording_countdown=cancel_recording_countdown,
                now=loop_start,
                step_index=step_index,
                last_state=episode_last_state,
                log=_log,
                control_ui=control_ui,
            ):
                return True
            if _is_emergency_stop_latch_error(error):
                consume_latched_safety_dispatch(error)
                return True
            return False

        phase_recorder = loop_phase_timer.record if loop_phase_timer is not None else None
        if agent_motion_port is not None:
            agent_motion_snapshot = agent_motion_port.apply_owner_requests(
                session=components.session,
                now=loop_start,
            )
            synchronize_agent_motion_owner_state(
                agent_motion_snapshot,
                now=loop_start,
            )
        try:
            step_result = components.session.run_step(
                now=loop_start,
                phase_recorder=phase_recorder,
            )
        except PositionAdmittanceActiveFault as exc:
            message = f"position admittance pre-IK fault: {exc}"
            if position_admittance_shadow is not None:
                position_admittance_shadow.latch_admission_feedback_fault(message)
            _log(
                "PositionAdmittance",
                "Active target rejected before dispatch; requesting Hold: "
                + message,
                now=loop_start,
            )
            _latch_safety_hold_and_hold_hardware_best_effort(
                components,
                clutch_input,
                message=message,
                reason="position_admittance_active_pre_ik_fault",
                log=_log,
                now=loop_start,
                require_home_recalibration=True,
            )
            continue
        except TCPPoseEnvelopeRuntimeError as exc:
            message = f"TCP Pose envelope pre-IK fault: {exc}"
            _log(
                "TCPPoseEnvelope",
                "Active target rejected before IK/dispatch; requesting Hold: "
                + message,
                now=loop_start,
            )
            _latch_safety_hold_and_hold_hardware_best_effort(
                components,
                clutch_input,
                message=message,
                reason="tcp_pose_envelope_active_pre_ik_fault",
                log=_log,
                now=loop_start,
                require_home_recalibration=True,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - only safety races are consumed here.
            if handle_safety_dispatch_exception(exc):
                continue
            raise
        mark_loop_phase("loop.session_run_step")
        if runtime_extension is not None:
            after_session_step = getattr(
                runtime_extension,
                "after_session_step",
                None,
            )
            if callable(after_session_step):
                after_session_step(step_result=step_result, now=loop_start)
                mark_loop_phase("loop.runtime_extension_after_session_step")
        if agent_motion_port is not None:
            agent_motion_snapshot = agent_motion_port.observe_session_step(
                command=step_result.command,
                command_dispatched=step_result.command_dispatched,
            )
            synchronize_agent_motion_owner_state(
                agent_motion_snapshot,
                now=loop_start,
            )
        startup_continuity = (
            None
            if step_result.command is None
            else step_result.command.metadata.get("startup_home_continuity")
        )
        if isinstance(startup_continuity, Mapping):
            startup_status = str(startup_continuity.get("status", "") or "")
            startup_key = (
                int(startup_continuity.get("epoch", 0)),
                startup_status,
            )
            if startup_status and startup_key != last_startup_home_continuity_key:
                last_startup_home_continuity_key = startup_key
                dispatch_allowed = bool(
                    startup_continuity.get("dispatch_allowed", False)
                )
                if dispatch_allowed:
                    startup_message = (
                        "首帧连续性验证通过；已将当前 qref 交给既有位控规划器。"
                    )
                else:
                    startup_message = (
                        "首帧连续性验证中，机械臂命令保持不下发："
                        + str(
                            startup_continuity.get("reason")
                            or (
                                f"连续帧 {int(startup_continuity.get('stable_frames', 0))}/"
                                f"{int(startup_continuity.get('required_consecutive_frames', 0))}"
                            )
                        )
                    )
                if control_ui is not None:
                    control_ui.append_message(
                        "StartupHomeContinuity",
                        startup_message,
                    )
                    control_ui.publish(control_hint=startup_message)
        if _request_position_admittance_endpoint_hold(
            step_result.command,
            clutch_input=clutch_input,
            action_debug_log_recorder=action_debug_log_recorder,
            log=_log,
            now=loop_start,
        ):
            continue
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
        if actuator_trace_recorder is not None:
            actuator_trace_recorder.record_vive_source(
                control_wall_time=loop_start,
                control_monotonic_time_ns=time.monotonic_ns(),
                raw_channels=raw_step_channels,
                teleop_state=step_result.state.value,
                motion_hold_mode=step_result.motion_hold_mode,
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
        arm_dispatch_confirmed = _project_session_dispatch_state(
            step_result,
            clutch_input=clutch_input,
        )
        if (
            not arm_dispatch_confirmed
            and recording_active()
            and not recording_paused()
        ):
            pause_recording(
                loop_start,
                reason="arm_dispatch_unavailable",
            )
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
            and (
                clutch_input._two_pedal_mode_active()
                or (
                    progress_workflow is not None
                    and not context.collection_transition_active()
                    and str(
                        getattr(
                            args,
                            "collection_progress_arm_control_mode",
                            "profile",
                        )
                    )
                    != PROGRESS_ARM_CONTROL_MODE_IGNORE
                )
            )
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
        countdown_control_plane_started_at = time.perf_counter()
        advance_recording_countdown(loop_start, step_result=step_result)
        if not rearm_position_admittance_after_control_plane_pause(
            started_at=countdown_control_plane_started_at,
            reason="recording_countdown_or_episode_open",
            now=loop_start,
        ):
            _cancel_recording_countdown_for_hold_best_effort(
                cancel_recording_countdown,
                now=loop_start,
                reason="position_admittance_control_plane_rearm_failed",
                log=_log,
            )
            _stop_recording_best_effort(
                recording_active=recording_active,
                stop_recording=stop_recording,
                now=loop_start,
                end_reason="position_admittance_control_plane_rearm_failed",
                summary={
                    "last_step_index": step_index,
                    "last_state": episode_last_state,
                    "reason": "position_admittance_control_plane_rearm_failed",
                },
                log=_log,
                control_ui=control_ui,
                ui_tag="PositionAdmittance",
                ui_failure_message=(
                    "位置导纳重武装失败后的数采收尾未完成；机械臂仍保持 Hold，"
                    "请检查 Dataset 日志并重新 Home。"
                ),
            )
            continue
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
                        collision_handler=handle_impedance_force_collision,
                    ):
                        assert components.session.rig is not None
                        try:
                            components.session.dispatch_command(active_command)
                        except Exception as exc:  # noqa: BLE001 - consume stop races only.
                            if handle_safety_dispatch_exception(exc):
                                continue
                            raise
                        manual_recovery_command_dispatched = True
                        if components.temporary_safety_limiter is not None:
                            components.temporary_safety_limiter.sync_from_command(active_command)
                    elif impedance_collision_consumed["value"]:
                        continue
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
                        collision_handler=handle_impedance_force_collision,
                    ):
                        assert components.session.rig is not None
                        try:
                            components.session.dispatch_command(active_command)
                        except Exception as exc:  # noqa: BLE001 - consume stop races only.
                            if handle_safety_dispatch_exception(exc):
                                continue
                            raise
                        manual_recovery_command_dispatched = True
                        if components.temporary_safety_limiter is not None:
                            components.temporary_safety_limiter.sync_from_command(active_command)
                    elif impedance_collision_consumed["value"]:
                        continue
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
        if agent_motion_port is not None:
            try:
                _publish_real_agent_motion_measured_cut(
                    components,
                    hardware_state_by_side={
                        "left": left_hw_state,
                        "right": right_hw_state,
                    },
                    preview_qpos_by_side={
                        "left": _state_qpos(
                            components.left_preview_backend,
                            DEFAULT_LEFT_ARM_QPOS,
                        ),
                        "right": _state_qpos(
                            components.right_preview_backend,
                            DEFAULT_RIGHT_ARM_QPOS,
                        ),
                    },
                    available_at=loop_start,
                )
            except Exception as exc:  # noqa: BLE001 - optional publication owns no control effect.
                _log(
                    "AgentMotion",
                    "measured TCP publication rejected; active Agent operation "
                    "will fail closed on freshness: "
                    f"{type(exc).__name__}: {exc}",
                    now=loop_start,
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
            active_command_dispatched=manual_recovery_command_dispatched,
        )
        if (
            position_admittance_shadow is not None
            and bool(getattr(position_admittance_shadow, "apply_to_target", False))
            and bool(step_result.command_dispatched)
            and step_result.command is not None
        ):
            try:
                admitted_poses = admitted_solver_world_poses_from_joint_command(
                    retargeter=components.dual_arm_retargeter,
                    command=step_result.command,
                )
                position_admittance_shadow.observe_admitted_poses(
                    admitted_poses,
                    # The pre-IK modifier sampled its output after loop_start.
                    # Commit admitted-qref feedback at the actual observation
                    # time so the next 60 Hz slew interval is never backdated.
                    now_sec=time.monotonic(),
                )
            except Exception as exc:  # noqa: BLE001 - active mode must fail closed.
                message = (
                    "position admittance admitted-qref feedback failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                position_admittance_shadow.latch_admission_feedback_fault(message)
                _log(
                    "PositionAdmittance",
                    "admitted-qref feedback failed; requesting Hold: " + message,
                    now=loop_start,
                )
                _latch_safety_hold_and_hold_hardware_best_effort(
                    components,
                    clutch_input,
                    message=message,
                    reason="position_admittance_admission_feedback_failed",
                    log=_log,
                    now=loop_start,
                    require_home_recalibration=True,
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
        if (
            progress_workflow is not None
            and not context.collection_transition_active()
        ):
            try:
                resume_result = progress_workflow.observe_applied_resume(
                    now=loop_start,
                    application_running=bool(
                        arm_dispatch_confirmed
                        and not clutch_input.hold_active()
                    ),
                    formal_capture_paused=bool(
                        dataset_recorder is not None
                        and dataset_recorder.is_paused
                    ),
                    formal_capture_resume_effective_at=(
                        None
                        if dataset_recorder is None
                        else dataset_recorder.last_resume_at
                    ),
                )
                acceptance_result = progress_workflow.consume_sample_acceptance(
                    available_at=loop_start,
                )
                if control_ui is not None and resume_result.handled:
                    control_ui.append_message(
                        "Progress",
                        "失败后的 Hold 已解除，正式采集已恢复；"
                        "等待第一帧 writer 接纳后创建 Retry Attempt。",
                    )
                if control_ui is not None and acceptance_result.handled:
                    control_ui.append_message(
                        "Progress",
                        "首帧已由正式 writer 接纳；Retry Attempt 已开始。",
                    )
                publish_progress_ui_state(now=loop_start)
            except Exception as exc:  # noqa: BLE001 - semantics fail closed.
                message = (
                    "collection progress Resume/Retry 对齐失败，"
                    "数采保持暂停并进入 Hold："
                    f"{type(exc).__name__}: {exc}"
                )
                _log("Progress", message, now=loop_start)
                if control_ui is not None:
                    control_ui.append_message("Progress", message)
                clutch_input.request_hold(reason="progress_retry_alignment_error")
                pause_recording(
                    loop_start,
                    reason="progress_retry_alignment_error",
                )
                continue
        if recording_active():
            episode_last_state = context.episode_last_state()
            if episode_last_state != step_result.state.value:
                if episode_recorder.is_recording:
                    record_debug_event(
                        "state_change",
                        payload={
                            "old_state": episode_last_state,
                            "new_state": step_result.state.value,
                        },
                        now=loop_start,
                    )
                episode_last_state = step_result.state.value
                context.set_episode_last_state(episode_last_state)
            if (
                dataset_recorder is not None
                and dataset_recorder.is_recording
                and dataset_recorder.can_record_samples
                and not recording_paused()
                and sample_due
            ):
                try:
                    sample_admission = runtime_extension_dataset_sample_admission(
                        runtime_extension,
                        step_result=step_result,
                        now=loop_start,
                    )
                except Exception as exc:  # noqa: BLE001 - capture must fail closed.
                    message = (
                        "runtime extension dataset sample admission failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    _log("Dataset", message, now=loop_start)
                    clutch_input.request_hold(
                        reason="runtime_extension_dataset_admission_error"
                    )
                    pause_recording(
                        loop_start,
                        reason="runtime_extension_dataset_admission_error",
                    )
                    continue
                sample_allowed = bool(sample_admission.allowed)
                collection_metadata_extra = {
                    **(
                        {}
                        if not sample_allowed
                        else dict(sample_admission.metadata)
                    ),
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
                if not sample_allowed:
                    dataset_step_result = None
                elif position_reference_enabled:
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
                        # --debug-ik-0827 is a standalone investigation log.
                        # Do not route its opt-in rows into formal dataset
                        # metadata; the historical GUI toggle retains its
                        # existing include-debug-metadata behavior.
                        include_debug_metadata=bool(
                            ik_debug_log_recorder.active
                            and not debug_ik_0827_enabled
                        ),
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
                record_debug_frame(
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
                    now=loop_start,
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
                action_debug_payload = _build_action_debug_metadata(
                    _collection_step_result(
                        step_result,
                        active_command=active_command,
                        gripper_command=gripper_command,
                        active_command_dispatched=(
                            manual_recovery_command_dispatched
                        ),
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
                if position_admittance_shadow is not None:
                    action_debug_payload["position_admittance"] = (
                        position_admittance_shadow.snapshot()
                    )
                action_debug_log_recorder.record_frame(action_debug_payload)
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
                            collection_step_result.command_dispatched
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
                    diagnostic_profile=(
                        "ik-0827" if debug_ik_0827_enabled else None
                    ),
                    camera_sources=(camera_sources if debug_ik_0827_enabled else None),
                    preview_camera_sources=(
                        preview_camera_sources if debug_ik_0827_enabled else None
                    ),
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
        # Planner policy and pose-quality thresholds can legitimately toggle
        # several times per second near a workspace/singularity boundary. They
        # remain in the full diagnostic snapshot below, but do not make a
        # terminal line urgent. Transport errors and quick-stop state do.
        command_backend_immediate_snapshot = tuple(
            (
                signature[0],
                signature[1],
                signature[4],
                signature[5],
            )
            if signature and signature[0] == "position" and len(signature) >= 6
            else (
                signature[0],
                signature[1],
                signature[2],
                signature[3],
                signature[5],
            )
            if signature and signature[0] == "impedance" and len(signature) >= 6
            else signature
            for signature in command_backend_snapshot
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
        immediate_status_snapshot = (
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
            command_backend_immediate_snapshot,
            hand_input_status,
            gripper_health_status,
            clutch_input.gripper_control_enabled(),
            tuple(sorted(fix_arm_snapshot.items())),
            tuple(sorted(init_arm_snapshot.items())),
        )
        mark_loop_phase("loop.status_snapshot")
        immediate_status_changed = status_gate.should_emit(
            "real_teleop_status_immediate",
            immediate_status_snapshot,
            now=loop_start,
        )
        if status_gate.should_emit(
            "real_teleop_status",
            status_snapshot,
            now=loop_start,
            heartbeat_sec=status_heartbeat_sec,
            min_interval_sec=(
                0.0
                if immediate_status_changed
                else float(getattr(args, "status_change_min_sec", 1.0))
            ),
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
    "run_real_teleop_control_loop",
]
