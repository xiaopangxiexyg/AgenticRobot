"""Readiness checks used before starting or resuming an episode recording."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping

from dataset_contracts.capture_modalities import CaptureModalityContract

from ..collection.sample_assembler import DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
from ..diagnostics import TeleopRunState
from ..session import SessionStepResult
from .components import RealTeleopComponents
from .hand_config import HAND_BACKEND_NONE, hand_backend_mode_for_side
from .hand_runtime import (
    _gripper_backend_for_side,
    _gripper_input_channel_name,
    _hand_backend_for_side,
)


def session_step_arm_dispatch_confirmed(step_result: object) -> bool:
    """Return whether this exact Session step dispatched an arm command.

    ``TeleopRunState.RUNNING`` is a lifecycle state and can remain latched while
    a command source or final admission gate temporarily inhibits dispatch.
    Collection admission must therefore use the per-step dispatch receipt and
    also reject hand-only commands.
    """

    command = getattr(step_result, "command", None)
    commands = getattr(command, "commands", {}) if command is not None else {}
    return bool(
        getattr(step_result, "command_dispatched", False)
        and any(name in commands for name in ("left_arm", "right_arm"))
    )


def recording_auto_stop_reason(state: TeleopRunState | str) -> str | None:
    state_value = str(getattr(state, "value", state)).upper()
    if state_value == TeleopRunState.FAULT.value:
        return "runtime_fault"
    if state_value == TeleopRunState.ESTOP.value:
        return "runtime_estop"
    return None


def recording_warmup_status(
    now: float,
    *,
    step_result: SessionStepResult | None,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: object,
    dataset_recorder: object | None,
    camera_sources: Mapping[str, object],
    preview_camera_sources: Mapping[str, object],
    camera_previewer: object | None,
    recording_auto_stop_reason_fn: Callable[[TeleopRunState | str], str | None],
    gripper_warmup_probe_status: Callable[..., tuple[bool, str]],
    camera_status_blocking_message: Callable[[Mapping[str, object]], str | None],
    allow_pre_episode_armed_warmup: bool = False,
) -> tuple[bool, str]:
    """Return the historical fail-closed recording admission result."""

    recording_auto_stop_reason = recording_auto_stop_reason_fn
    _camera_status_blocking_message = camera_status_blocking_message
    _hand_backend_mode_for_side = hand_backend_mode_for_side

    if step_result is None:
        return False, "等待首个控制帧。"
    stop_reason = recording_auto_stop_reason(step_result.state)
    if stop_reason is not None:
        return False, f"等待恢复，当前状态会自动停止数采：{stop_reason}。"
    if clutch_input.hold_active():
        return False, "等待 hold 解除并重采 runtime home。"
    pre_episode_armed_warmup = bool(
        allow_pre_episode_armed_warmup
        and step_result.state == TeleopRunState.ARMED
    )
    if step_result.state != TeleopRunState.RUNNING and not pre_episode_armed_warmup:
        return False, f"等待遥操进入 RUNNING，当前状态：{step_result.state.value}。"
    if step_result.observation is None:
        return False, "等待 observation 可用。"
    if not pre_episode_armed_warmup:
        if step_result.command is None:
            return False, "等待 action command 可用。"
        if not session_step_arm_dispatch_confirmed(step_result):
            return False, "等待机械臂命令通过最终 admission 并实际派发。"
    if dataset_recorder is not None:
        actual_ready, actual_message = dataset_recorder.actual_feedback_warmup_status(now)
        if not actual_ready:
            return False, actual_message
        worker_ready, worker_message = (
            dataset_recorder.collection_worker_ready_status()
        )
        if not worker_ready:
            return False, worker_message
    if not bool(getattr(args, "no_camera", False)):
        pending_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(
                getattr(args, "ignore_camera", ()) or ()
            ),
            ignored_tactile_keys=tuple(
                getattr(args, "ignore_tactile", ()) or ()
            ),
        )
        ignored_camera_keys = set(pending_contract.ignored_camera_keys)
        for camera_key, source in camera_sources.items():
            if camera_key in ignored_camera_keys:
                continue
            frame = source.frame_buffer().get_latest()
            if frame is None:
                return False, f"等待相机 {camera_key} 首帧。"
            source_timestamp = float(frame.source_timestamp)
            # CollectorSampleBuilder applies the stricter of the source and
            # availability timestamps.  Use the same effective age here so
            # a resume cannot reopen an episode that the very next sample
            # will reject because the frame was not yet available to the
            # collector.
            available_timestamp = float(
                getattr(frame, "available_timestamp", source_timestamp)
            )
            source_age = float(now) - source_timestamp
            available_age = float(now) - available_timestamp
            if source_age < -0.05 or available_age < -0.05:
                return False, f"等待相机 {camera_key} 时间戳稳定。"
            camera_age = max(
                source_age,
                available_age,
            )
            if camera_age > DEFAULT_CAMERA_FRAME_MAX_AGE_SEC:
                return False, f"等待相机 {camera_key} 最新帧，当前 age={camera_age:.2f}s。"
        camera_status = clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=now,
            force=False,
        )
        camera_blocking_message = _camera_status_blocking_message(
            {
                key: value
                for key, value in camera_status.items()
                if key not in ignored_camera_keys
            }
        )
        if camera_blocking_message:
            return False, camera_blocking_message
    gripper_input_source = str(
        getattr(components, "gripper_input_source", "q6") or "q6"
    ).strip().lower()
    if gripper_input_source not in {"none", "q6", "angle_trigger"}:
        return False, f"等待夹爪输入配置恢复：不支持 {gripper_input_source!r}。"
    # The physical hand backend is an output/feedback device.  It must not
    # imply an operator-input dependency: rule control deliberately uses
    # gripper_input_source=none while still commanding and monitoring DMTAC.
    if gripper_input_source != "none" and any(
        _hand_backend_mode_for_side(args, side) != HAND_BACKEND_NONE
        for side in components.active_sides
    ):
        for side in components.active_sides:
            if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_NONE:
                continue
            channel_name = _gripper_input_channel_name(components, side)
            if channel_name is None:
                continue
            channel = step_result.channels.get(channel_name)
            input_label = (
                "hand_q6" if gripper_input_source == "q6" else "angle_trigger"
            )
            if channel is None or not bool(getattr(channel, "valid", False)):
                return False, f"等待 {side} {input_label} 输入有效。"
            hand_age = float(now) - float(getattr(channel, "timestamp", now))
            if hand_age < -0.05:
                return False, f"等待 {side} {input_label} 时间戳稳定。"
            if hand_age > 0.5:
                return False, (
                    f"等待 {side} {input_label} 最新输入，"
                    f"当前 age={hand_age:.2f}s。"
                )
    for side in components.active_sides:
        backend = _gripper_backend_for_side(components, side)
        if backend is None:
            continue
        backend_config = getattr(backend, "config", None)
        if not bool(getattr(backend_config, "connect_hardware", False)):
            continue
        get_state = getattr(backend, "get_state")
        try:
            state = get_state(include_tactile_arrays=False)
        except TypeError:
            state = get_state()
        error_code = state.get("gripper_error_code")
        if error_code not in (None, 0):
            status_code = state.get("gripper_status_code")
            status_text = str(state.get("gripper_status_text") or "").strip()
            status_suffix = ""
            if status_code is not None or status_text:
                status_suffix = (
                    f"，status={status_code if status_code is not None else '-'}"
                    f"{':' + status_text if status_text else ''}"
                )
            return False, (
                f"等待 {side} 夹爪故障恢复：error_code={error_code}{status_suffix}。"
            )
        telemetry_error = str(state.get("telemetry_error") or "").strip()
        # DMTAC shares one telemetry field for gripper transport and tactile
        # worker diagnostics.  The latter is handled by the modality-specific
        # gate below; a serial/command telemetry error must block the episode
        # even when the cached vendor error_code is still zero.
        if telemetry_error and not telemetry_error.lower().startswith("tactile_"):
            return False, f"等待 {side} 夹爪传输故障恢复：{telemetry_error}。"
    for side in components.active_sides:
        if (
            side
            in CaptureModalityContract(
                ignored_camera_keys=tuple(
                    getattr(args, "ignore_camera", ()) or ()
                ),
                ignored_tactile_keys=tuple(
                    getattr(args, "ignore_tactile", ()) or ()
                ),
            ).ignored_tactile_keys
        ):
            continue
        backend = _hand_backend_for_side(components, side)
        if backend is None:
            backend = _gripper_backend_for_side(components, side)
        if backend is None:
            continue
        backend_config = getattr(backend, "config", None)
        tactile_configured = bool(
            getattr(backend_config, "connect_tactile", False)
            or tuple(getattr(backend_config, "tactile_regions", ()) or ())
        )
        if not tactile_configured:
            continue
        get_state = getattr(backend, "get_state")
        try:
            state = get_state(include_tactile_arrays=False)
        except TypeError:
            state = get_state()
        if bool(state.get("tactile_process_read")) and not bool(
            state.get("tactile_worker_alive")
        ):
            return False, f"等待 {side} 触觉采集子进程恢复。"
        tactile_timestamp = state.get("tactile_timestamp")
        if tactile_timestamp is None:
            error = str(state.get("telemetry_error") or "")
            return False, f"等待 {side} 触觉首帧{': ' + error if error else ''}。"
        tactile_age = float(now) - float(tactile_timestamp)
        read_hz = float(
            state.get("tactile_read_hz")
            or getattr(backend_config, "telemetry_hz", 0.0)
            or 0.0
        )
        tactile_max_age = max(0.5, 3.0 / read_hz) if read_hz > 0.0 else 1.0
        if tactile_age < -0.05:
            return False, f"等待 {side} 触觉时间戳稳定。"
        if tactile_age > tactile_max_age:
            return False, (
                f"等待 {side} 最新触觉帧，当前 age={tactile_age:.2f}s "
                f"> {tactile_max_age:.2f}s。"
            )
    probe_ready, probe_message = gripper_warmup_probe_status(now, reason="recording_warmup")
    if not probe_ready:
        return False, probe_message
    return True, "warmup 输入稳定。"


__all__ = [
    "recording_auto_stop_reason",
    "recording_warmup_status",
    "session_step_arm_dispatch_confirmed",
]
