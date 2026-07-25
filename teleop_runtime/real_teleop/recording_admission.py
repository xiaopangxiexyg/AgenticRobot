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
from .hand_runtime import _gripper_backend_for_side, _hand_backend_for_side


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
    if step_result.state != TeleopRunState.RUNNING:
        return False, f"等待遥操进入 RUNNING，当前状态：{step_result.state.value}。"
    if step_result.observation is None:
        return False, "等待 observation 可用。"
    if step_result.command is None:
        return False, "等待 action command 可用。"
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
    if any(
        _hand_backend_mode_for_side(args, side) != HAND_BACKEND_NONE
        for side in components.active_sides
    ):
        for side in components.active_sides:
            if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_NONE:
                continue
            channel = step_result.channels.get(f"{side}_hand_q6")
            if channel is None or not bool(getattr(channel, "valid", False)):
                return False, f"等待 {side} hand_q6 输入有效。"
            hand_age = float(now) - float(getattr(channel, "timestamp", now))
            if hand_age < -0.05:
                return False, f"等待 {side} hand_q6 时间戳稳定。"
            if hand_age > 0.5:
                return False, f"等待 {side} hand_q6 最新输入，当前 age={hand_age:.2f}s。"
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
]
