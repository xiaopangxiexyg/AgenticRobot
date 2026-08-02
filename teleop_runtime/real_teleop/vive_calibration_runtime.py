"""Interactive VIVE coordinate and wrist-offset calibration workflow.

The functions operate on an already-constructed VIVE source and GUI.  They do
not start or close devices.  Profile persistence is supplied explicitly by the
canonical real-teleop coordinator so its legacy path and clock overrides remain
authoritative.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..calibration import TeleopCalibration, VIRTUAL_DESK_SWEEP_CONVENTION_VERSION
from ..calibration_flow import (
    average_pose,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)
from ..log_utils import ChangeGate, log_event as _log
from ..operator_ui_ports import OperatorUiPort
from ..sources.pose import VivePoseSource
from ..tracker_hand import (
    TRACKER_OFFSET_CALIBRATION_VERSION,
    canonical_hand_rpy_default,
)
from ..transforms import as_pose_matrix
from .components import RealTeleopComponents
from .gui_tracker_status import (
    TRACKER_GUI_STALE_AGE_SEC,
    TRACKER_GUI_WARN_AGE_SEC,
    raw_tracker_pose_state_from_channels,
)
from .gui_vive_calibration import (
    vive_calibration_profile_ui_state as _vive_calibration_profile_ui_state,
)


@dataclass
class ViveCoordinateCalibrationContext:
    """Minimal context consumed by the shared VIVE coordinate workflow.

    Real teleop still passes its existing ``RealTeleopComponents`` object to
    the compatibility helpers below.  Simulation and Control Station callers
    can pass this small context instead, without constructing a real-teleop
    component graph or acquiring any execution authority.
    """

    source: object
    calibration: TeleopCalibration
    active_sides: tuple[str, ...]


def _context_calibration(context: object) -> TeleopCalibration | None:
    calibration = getattr(context, "calibration", None)
    if calibration is not None:
        return calibration
    session = getattr(context, "session", None)
    return getattr(session, "calibration", None)


def _context_active_sides(context: object) -> tuple[str, ...]:
    return tuple(getattr(context, "active_sides", ()) or ())


def _context_profile_ui_state(args: argparse.Namespace, context: object) -> dict[str, object]:
    if isinstance(context, ViveCoordinateCalibrationContext):
        return {}
    return _vive_calibration_profile_ui_state(args, context)


def _vive_pose_source_from_components(
    components: RealTeleopComponents,
) -> VivePoseSource | None:
    for source in components.session.sources:
        if isinstance(source, VivePoseSource):
            return source
    return None


def _raw_tracker_pose_state_from_channels(
    channels: dict[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float,
) -> dict[str, object]:
    return raw_tracker_pose_state_from_channels(
        channels,
        active_sides=active_sides,
        now=now,
        warn_age_sec=TRACKER_GUI_WARN_AGE_SEC,
        stale_age_sec=TRACKER_GUI_STALE_AGE_SEC,
    )


def _clear_runtime_home_state(components: object) -> None:
    calibration = _context_calibration(components)
    if calibration is not None:
        calibration.clear_home_state()
    for retargeter_name in ("left_retargeter", "right_retargeter"):
        retargeter = getattr(components, retargeter_name, None)
        clear_home_anchors = getattr(retargeter, "clear_home_anchors", None)
        if callable(clear_home_anchors):
            clear_home_anchors()


def _channel_sample_key(channel: object) -> tuple[object, object, object]:
    metadata = getattr(channel, "metadata", None) or {}
    return (
        metadata.get("seq"),
        metadata.get("receiver_timestamp"),
        getattr(channel, "timestamp", None),
    )


def _format_wall_time(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(float(timestamp)))


def _source_debug_stats(source: object) -> dict[str, object]:
    get_stats = getattr(source, "debug_stats", None)
    if get_stats is None:
        return {}
    stats = get_stats()
    return stats if isinstance(stats, dict) else {}


def _format_vive_source_rates(
    source: object,
    *,
    start_stats: Mapping[str, object],
    channel_names: tuple[str, ...],
    elapsed_sec: float,
    now: float,
) -> str:
    current_stats = _source_debug_stats(source)
    if not current_stats:
        return "source_stats=unavailable"
    elapsed = max(float(elapsed_sec), 1e-6)
    packet_count = int(current_stats.get("packet_count") or 0)
    start_packet_count = int(start_stats.get("packet_count") or 0)
    packet_hz = (packet_count - start_packet_count) / elapsed
    update_counts = current_stats.get("channel_update_counts") or {}
    start_update_counts = start_stats.get("channel_update_counts") or {}
    last_updates = current_stats.get("last_channel_update_at") or {}
    channel_parts: list[str] = []
    if isinstance(update_counts, Mapping) and isinstance(start_update_counts, Mapping):
        for channel_name in channel_names:
            count = int(update_counts.get(channel_name) or 0)
            start_count = int(start_update_counts.get(channel_name) or 0)
            channel_hz = (count - start_count) / elapsed
            last_update_at = last_updates.get(channel_name) if isinstance(last_updates, Mapping) else None
            age = (now - float(last_update_at)) if last_update_at is not None else float("nan")
            channel_parts.append(f"{channel_name}_hz={channel_hz:.1f} age={age:.2f}s")
    last_packet_at = current_stats.get("last_packet_received_at")
    packet_age = (now - float(last_packet_at)) if last_packet_at is not None else float("nan")
    return f"source_udp_hz={packet_hz:.1f} packet_age={packet_age:.2f}s " + " ".join(channel_parts)


def _format_vive_sweep_quality(poses: list[np.ndarray]) -> str:
    try:
        quality = fit_signed_axis_from_poses(poses).quality
    except Exception as exc:  # noqa: BLE001
        return f"quality=unavailable({type(exc).__name__}: {exc})"
    return (
        f"quality extent={quality.extent_m:.3f}m "
        f"variance_ratio={quality.variance_ratio:.3f} "
        f"start_end_distance={quality.start_end_distance_m:.3f}m"
    )


def _virtual_desk_post_rotation_pose(yaw_deg: float) -> np.ndarray:
    return pose_from_rotation_translation(
        euler_xyz_deg_to_matrix([0.0, 0.0, float(yaw_deg)]),
        [0.0, 0.0, 0.0],
    )


def _channel_for_tracker_role(role: str) -> str:
    return {
        "auxiliary": "auxiliary_tracker_pose",
        "left_wrist": "left_wrist_pose",
        "right_wrist": "right_wrist_pose",
    }[str(role)]


def _tracker_role_for_channel(channel_name: str) -> str:
    return {
        "auxiliary_tracker_pose": "auxiliary",
        "left_wrist_pose": "left_wrist",
        "right_wrist_pose": "right_wrist",
    }.get(str(channel_name), str(channel_name))


def _resolve_virtual_desk_channel_name(args: argparse.Namespace, channels: Mapping[str, object]) -> str:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    if requested != "auto":
        return _channel_for_tracker_role(requested)
    for candidate in ("right_wrist_pose", "left_wrist_pose", "auxiliary_tracker_pose"):
        channel = channels.get(candidate)
        if channel is not None and bool(getattr(channel, "valid", False)):
            return candidate
    if str(getattr(args, "aux_tracker_name", "") or "").strip():
        return "auxiliary_tracker_pose"
    return "right_wrist_pose"


def _default_gui_vive_calibration_side(
    args: argparse.Namespace,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> str:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    if requested in {"left_wrist", "right_wrist"}:
        requested_side = requested.removesuffix("_wrist")
        if requested_side in active_sides:
            return requested_side
    if "right" in active_sides:
        return "right"
    if "left" in active_sides:
        return "left"
    return "right"


def _select_gui_vive_coordinate_sweep_hand(
    *,
    ui: OperatorUiPort,
    args: argparse.Namespace,
    components: object,
    source: object,
) -> tuple[str, str]:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    interactive_hand_selection = bool(
        getattr(args, "interactive_virtual_desk_hand_selection", True)
    )
    if requested == "auxiliary" or (
        requested in {"left_wrist", "right_wrist"}
        and not interactive_hand_selection
    ):
        channel_name = _channel_for_tracker_role(requested)
        ui.append_message(
            "ViveCalibration",
            f"已显式指定 {channel_name} 做三轴 sweep，不再进入额外的手侧确认阶段。",
        )
        _log(
            "ViveCalibration",
            f"using explicit tracker role={requested} channel={channel_name}",
        )
        return requested.removesuffix("_wrist"), channel_name

    stage_name = "VIVE 标定手选择"
    stage_id = "ViveCalibrationHand"
    selected_side = _default_gui_vive_calibration_side(
        args,
        _context_active_sides(components) or ("left", "right"),
    )
    last_status_hint = ""
    ui.append_message(
        "ViveCalibration",
        "请选择用于 VIVE 坐标系三轴 sweep 的手：默认右手。点击“左手/右手”或按 1/2，点击“下一步”或按 n 确认。",
    )
    _log(
        "ViveCalibration",
        f"hand selection ready default={selected_side}; press 1/2 then n to continue",
    )
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        for event in events:
            if event == "1":
                selected_side = "left"
                last_status_hint = "已选择左手；点击“下一步”或按 n 开始三轴 sweep。"
                ui.append_message("ViveCalibration", "已选择左手进行 VIVE 坐标系标定。")
            elif event == "2":
                selected_side = "right"
                last_status_hint = "已选择右手；点击“下一步”或按 n 开始三轴 sweep。"
                ui.append_message("ViveCalibration", "已选择右手进行 VIVE 坐标系标定。")

        channel_name = f"{selected_side}_wrist_pose"
        channel = channels.get(channel_name)
        channel_valid = bool(channel is not None and getattr(channel, "valid", False))
        side_text = "左手" if selected_side == "left" else "右手"
        if last_status_hint:
            control_hint = last_status_hint
        else:
            control_hint = f"请选择用于三轴 sweep 的手；默认右手，当前为{side_text}。点击“下一步”或按 n 确认。"
        if not channel_valid:
            control_hint += f" 当前 {channel_name} 暂无有效数据。"

        ui.publish(
            stage=stage_name,
            selected_side=selected_side,
            active_controls=["1", "2", "n", "q"],
            control_hint=control_hint,
            capture_stage=stage_id,
            capture_phase="ready",
            capturing=False,
            capture_channel=channel_name,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=channel_valid,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=float(getattr(args, "capture_prep_sec", 3.0)),
            capture_waiting_for_samples=False,
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            raw_tracker_poses=_raw_tracker_pose_state_from_channels(
                dict(channels),
                active_sides=_context_active_sides(components),
                now=loop_start,
            ),
            **_context_profile_ui_state(args, components),
        )
        if "n" in events:
            if channel_valid:
                ui.append_message(
                    "ViveCalibration",
                    f"将使用{side_text} {channel_name} 做 VIVE 坐标系三轴 sweep。",
                )
                _log(
                    "ViveCalibration",
                    f"selected {selected_side} hand for coordinate sweep channel={channel_name}",
                    now=loop_start,
                )
                return selected_side, channel_name
            last_status_hint = (
                f"{side_text} {channel_name} 当前无有效数据；请等待 tracker 在线，"
                "或选择另一只手后再点“下一步”。"
            )
            ui.append_message("ViveCalibration", last_status_hint)

        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def _publish_vive_capture_state(
    ui: OperatorUiPort,
    args: argparse.Namespace,
    components: object,
    channels: Mapping[str, object],
    *,
    now: float,
    stage: str,
    active_controls: list[str],
    control_hint: str,
    capture_stage: str,
    capture_phase: str,
    capturing: bool,
    capture_channel: str,
    capture_samples: int | None = None,
    capture_left_samples: int | None = None,
    capture_right_samples: int | None = None,
    capture_min_samples: int | None = None,
    capture_valid: bool | None = None,
    capture_left_valid: bool | None = None,
    capture_right_valid: bool | None = None,
    capture_elapsed_sec: float | None = None,
    capture_duration_sec: float | None = None,
    capture_prep_remaining_sec: float | None = None,
    capture_prep_total_sec: float | None = None,
    capture_waiting_for_samples: bool = False,
) -> None:
    ui.publish(
        stage=stage,
        active_controls=active_controls,
        control_hint=control_hint,
        capture_stage=capture_stage,
        capture_phase=capture_phase,
        capturing=capturing,
        capture_channel=capture_channel,
        capture_samples=capture_samples,
        capture_left_samples=capture_left_samples,
        capture_right_samples=capture_right_samples,
        capture_min_samples=capture_min_samples,
        capture_valid=capture_valid,
        capture_left_valid=capture_left_valid,
        capture_right_valid=capture_right_valid,
        capture_elapsed_sec=capture_elapsed_sec,
        capture_duration_sec=capture_duration_sec,
        capture_prep_remaining_sec=capture_prep_remaining_sec,
        capture_prep_total_sec=capture_prep_total_sec,
        capture_waiting_for_samples=capture_waiting_for_samples,
        teleop_running=False,
        home_ready=False,
        gripper_control_enabled=False,
        raw_tracker_poses=_raw_tracker_pose_state_from_channels(
            dict(channels),
            active_sides=_context_active_sides(components),
            now=now,
        ),
        **_context_profile_ui_state(args, components),
    )


def _consume_vive_capture_events(ui: OperatorUiPort) -> list[str]:
    return ui.consume_events()


def _capture_single_vive_channel_from_gui(
    *,
    ui: OperatorUiPort,
    args: argparse.Namespace,
    components: object,
    source: object,
    stage_name: str,
    instructions: str,
    timed_flow_hint: str,
    direction_hint: str,
    channel_name: str,
    min_samples: int,
    capture_duration_sec: float,
    validator,
    pose_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    ui.append_message(stage_name, instructions)
    ui.append_message(stage_name, timed_flow_hint)
    ui.append_message(stage_name, "本阶段需要点击“下一步”或按 n 才开始准备倒计时；r 重置当前阶段，q 取消。")
    _log(
        "ViveCalibration",
        f"{stage_name} ready channel={channel_name} direction={direction_hint} "
        f"prep={float(getattr(args, 'capture_prep_sec', 3.0)):.1f}s "
        f"window={float(capture_duration_sec):.1f}s min_samples={int(min_samples)}; wait_for_next=true; "
        f"{instructions}",
    )
    poses: list[np.ndarray] = []
    preparing = False
    capturing = False
    capture_start_time = 0.0
    capture_deadline = 0.0
    capture_extension_deadline = 0.0
    source_stats_at_start: dict[str, object] = {}
    last_sample_key: tuple[object, object, object] | None = None
    stage_gate = ChangeGate()
    min_samples_reported = False
    waiting_for_samples_reported = False
    last_status_hint = ""
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        if "r" in events:
            poses = []
            preparing = False
            capturing = False
            min_samples_reported = False
            waiting_for_samples_reported = False
            last_sample_key = None
            last_status_hint = f"已重置；点击“下一步”或按 n 重采，方向仍为 {direction_hint}。"
            ui.append_message(stage_name, "当前阶段已重置。")
        start_requested = bool("n" in events)
        if start_requested:
            if not capturing and not preparing:
                poses = []
                preparing = True
                last_status_hint = ""
                capture_start_time = loop_start + float(getattr(args, "capture_prep_sec", 3.0))
                capture_deadline = capture_start_time + float(capture_duration_sec)
                capture_extension_deadline = capture_deadline + float(getattr(args, "capture_extra_wait_sec", 0.0))
                source_stats_at_start = {}
                last_sample_key = None
                min_samples_reported = False
                waiting_for_samples_reported = False
                ui.append_message(
                    stage_name,
                    f"准备 {float(getattr(args, 'capture_prep_sec', 3.0)):.1f}s，"
                    f"{_format_wall_time(capture_start_time)} 开始采集。",
                )
            else:
                ui.append_message(stage_name, "采集已经在进行，请等待自动结束或按 r 重置。")

        channel = channels.get(channel_name)
        channel_valid = bool(channel is not None and getattr(channel, "valid", False))
        if preparing and loop_start >= capture_start_time:
            preparing = False
            capturing = True
            source_stats_at_start = _source_debug_stats(source)
            if channel_valid:
                last_sample_key = _channel_sample_key(channel)
                pose = as_pose_matrix(getattr(channel, "value"))
                poses.append(pose if pose_transform is None else pose_transform(pose))
            ui.append_message(
                stage_name,
                f"采集开始，窗口 {float(capture_duration_sec):.1f}s，"
                f"自动结束时间 {_format_wall_time(capture_deadline)}。",
            )
        elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
        waiting_for_samples = bool(capturing and loop_start >= capture_deadline and len(poses) < int(min_samples))
        capture_active = preparing or capturing
        capture_phase = (
            "preparing"
            if preparing
            else "waiting"
            if waiting_for_samples
            else "capturing"
            if capturing
            else "ready"
        )
        if preparing:
            control_hint = f"{direction_hint}；准备中，{max(0.0, capture_start_time - loop_start):.1f}s 后开始。"
        elif waiting_for_samples:
            control_hint = (
                f"{direction_hint}；样本不足 {len(poses)}/{int(min_samples)}，"
                "继续等待 fresh samples。"
            )
        elif capturing:
            control_hint = f"{direction_hint}；采集中 {len(poses)}/{int(min_samples)}，请按方向扫动。"
        else:
            control_hint = last_status_hint or f"{direction_hint}；点击“下一步”或按 n 开始本阶段。"
        _publish_vive_capture_state(
            ui,
            args,
            components,
            channels,
            now=loop_start,
            stage=stage_name,
            active_controls=["n", "r", "q"] if not capture_active else ["r", "q"],
            control_hint=control_hint,
            capture_stage=stage_name,
            capture_phase=capture_phase,
            capturing=capture_active,
            capture_channel=channel_name,
            capture_samples=len(poses),
            capture_min_samples=int(min_samples),
            capture_valid=channel_valid,
            capture_elapsed_sec=elapsed if capturing else None,
            capture_duration_sec=(
                float(capture_duration_sec) + float(getattr(args, "capture_extra_wait_sec", 0.0))
                if waiting_for_samples
                else float(capture_duration_sec)
            ),
            capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
            capture_prep_total_sec=float(getattr(args, "capture_prep_sec", 3.0)),
            capture_waiting_for_samples=waiting_for_samples,
        )
        if capturing and stage_gate.should_emit(f"{stage_name}_channel_valid", channel_valid, now=loop_start):
            _log("ViveCalibration", f"{stage_name} channel_valid={channel_valid}", now=loop_start)
        if capturing and channel_valid:
            sample_key = _channel_sample_key(channel)
            if sample_key != last_sample_key:
                last_sample_key = sample_key
                pose = as_pose_matrix(getattr(channel, "value"))
                poses.append(pose if pose_transform is None else pose_transform(pose))
            if len(poses) >= int(min_samples) and not min_samples_reported:
                min_samples_reported = True
                ui.append_message(stage_name, f"已达到最小样本数 {len(poses)}/{int(min_samples)}。")
        if capturing and stage_gate.should_emit(
            f"{stage_name}_live",
            channel_valid,
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "ViveCalibration",
                f"{stage_name} live time={elapsed:.1f}/{float(capture_duration_sec):.1f}s "
                f"samples={len(poses)} valid={channel_valid} "
                f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                now=loop_start,
            )
        if capturing and loop_start >= capture_deadline:
            capturing = False
            effective_hz = len(poses) / max(elapsed, 1e-6)
            if len(poses) < int(min_samples):
                if loop_start < capture_extension_deadline:
                    capturing = True
                    if not waiting_for_samples_reported:
                        waiting_for_samples_reported = True
                        ui.append_message(
                            stage_name,
                            f"样本不足 {len(poses)}/{int(min_samples)}，继续等待 fresh samples。",
                        )
                    time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))
                    continue
                last_status_hint = (
                    f"未通过：样本不足 {len(poses)}/{int(min_samples)}。"
                    f"当前仍在 {stage_name}，不会进入下一轴；请重采，方向仍为 {direction_hint}。"
                )
                _log(
                    "ViveCalibration",
                    f"{stage_name} AUTO-END INSUFFICIENT samples={len(poses)} need>={int(min_samples)} "
                    f"effective_hz={effective_hz:.1f}; stay_on={stage_name}; direction={direction_hint}; "
                    f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
                ui.append_message(stage_name, f"{last_status_hint}")
                _publish_vive_capture_state(
                    ui,
                    args,
                    components,
                    channels,
                    now=loop_start,
                    stage=stage_name,
                    active_controls=["n", "r", "q"],
                    control_hint=last_status_hint,
                    capture_stage=stage_name,
                    capture_phase="ready",
                    capturing=False,
                    capture_channel=channel_name,
                    capture_samples=0,
                    capture_min_samples=int(min_samples),
                    capture_valid=channel_valid,
                    capture_waiting_for_samples=False,
                )
                poses = []
                min_samples_reported = False
                waiting_for_samples_reported = False
                last_sample_key = None
            else:
                quality_text = _format_vive_sweep_quality(poses)
                error = validator(poses) if validator is not None else None
                if error is None:
                    _log(
                        "ViveCalibration",
                        f"{stage_name} AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f}; "
                        f"{quality_text}; "
                        f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                        now=loop_start,
                    )
                    ui.append_message(
                        stage_name,
                        f"采集接受：samples={len(poses)} hz={effective_hz:.1f}；{quality_text}。",
                    )
                    ui.publish(
                        stage=stage_name,
                        active_controls=[],
                        control_hint=f"{stage_name} 已接受，正在进入下一阶段。",
                        capture_phase="ready",
                        capturing=False,
                    )
                    return poses
                last_status_hint = (
                    f"未通过：{error}。当前仍在 {stage_name}，"
                    f"不会进入下一轴；请重采，方向仍为 {direction_hint}。{quality_text}。"
                )
                _log(
                    "ViveCalibration",
                    f"{stage_name} AUTO-END REJECTED: {error}; samples={len(poses)} "
                    f"effective_hz={effective_hz:.1f}; {quality_text}; "
                    f"stay_on={stage_name}; direction={direction_hint}; "
                    f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
                ui.append_message(stage_name, f"{last_status_hint}")
                _publish_vive_capture_state(
                    ui,
                    args,
                    components,
                    channels,
                    now=loop_start,
                    stage=stage_name,
                    active_controls=["n", "r", "q"],
                    control_hint=last_status_hint,
                    capture_stage=stage_name,
                    capture_phase="ready",
                    capturing=False,
                    capture_channel=channel_name,
                    capture_samples=0,
                    capture_min_samples=int(min_samples),
                    capture_valid=channel_valid,
                    capture_waiting_for_samples=False,
                )
                poses = []
                min_samples_reported = False
                waiting_for_samples_reported = False
                last_sample_key = None
        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def _capture_dual_wrist_offsets_from_gui(
    *,
    ui: OperatorUiPort,
    args: argparse.Namespace,
    components: object,
    source: object,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stage_id = "WristOffset"
    stage_name = "Wrist Offset Capture"
    ui.append_message(
        stage_id,
        "把左右 wrist tracker 放到标准中立姿态：前臂朝前、手背朝上、手指大致沿 desk +X。"
    )
    ui.append_message(stage_id, "点击“下一步”或按 n 开始定时采集；这里只写 tracker offset，不保存 runtime Home。")
    left_poses: list[np.ndarray] = []
    right_poses: list[np.ndarray] = []
    preparing = False
    capturing = False
    capture_start_time = 0.0
    capture_deadline = 0.0
    capture_extension_deadline = 0.0
    source_stats_at_start: dict[str, object] = {}
    left_last_sample_key: tuple[object, object, object] | None = None
    right_last_sample_key: tuple[object, object, object] | None = None
    stage_gate = ChangeGate()
    left_min_reported = False
    right_min_reported = False
    waiting_for_samples_reported = False
    min_samples = int(getattr(args, "min_hold_samples", 3))
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        if "r" in events:
            left_poses = []
            right_poses = []
            preparing = False
            capturing = False
            left_min_reported = False
            right_min_reported = False
            waiting_for_samples_reported = False
            left_last_sample_key = None
            right_last_sample_key = None
            ui.append_message(stage_id, "当前 wrist offset 阶段已重置。")
        if "n" in events:
            if not capturing and not preparing:
                left_poses = []
                right_poses = []
                preparing = True
                capture_start_time = loop_start + float(getattr(args, "wrist_capture_prep_sec", 5.0))
                capture_deadline = capture_start_time + float(getattr(args, "hold_capture_sec", 4.0))
                capture_extension_deadline = capture_deadline + float(getattr(args, "capture_extra_wait_sec", 0.0))
                source_stats_at_start = {}
                left_last_sample_key = None
                right_last_sample_key = None
                left_min_reported = False
                right_min_reported = False
                waiting_for_samples_reported = False
                ui.append_message(
                    stage_id,
                    f"准备 {float(getattr(args, 'wrist_capture_prep_sec', 5.0)):.1f}s，"
                    f"{_format_wall_time(capture_start_time)} 开始采集。",
                )
            else:
                ui.append_message(stage_id, "采集已经在进行，请等待自动结束或按 r 重置。")

        left_channel = channels.get("left_wrist_pose")
        right_channel = channels.get("right_wrist_pose")
        left_valid = bool(left_channel is not None and getattr(left_channel, "valid", False))
        right_valid = bool(right_channel is not None and getattr(right_channel, "valid", False))
        validity_snapshot = (left_valid, right_valid)
        if preparing and loop_start >= capture_start_time:
            preparing = False
            capturing = True
            source_stats_at_start = _source_debug_stats(source)
            if left_valid:
                left_last_sample_key = _channel_sample_key(left_channel)
                left_poses.append(
                    _context_calibration(components).world_to_desk(as_pose_matrix(getattr(left_channel, "value")))
                )
            if right_valid:
                right_last_sample_key = _channel_sample_key(right_channel)
                right_poses.append(
                    _context_calibration(components).world_to_desk(as_pose_matrix(getattr(right_channel, "value")))
                )
            ui.append_message(stage_id, f"采集开始，窗口 {float(getattr(args, 'hold_capture_sec', 4.0)):.1f}s。")
        elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
        waiting_for_samples = bool(
            capturing
            and loop_start >= capture_deadline
            and (len(left_poses) < min_samples or len(right_poses) < min_samples)
        )
        capture_active = preparing or capturing
        capture_phase = (
            "preparing"
            if preparing
            else "waiting"
            if waiting_for_samples
            else "capturing"
            if capturing
            else "ready"
        )
        _publish_vive_capture_state(
            ui,
            args,
            components,
            channels,
            now=loop_start,
            stage=stage_name,
            active_controls=["n", "r", "q"] if not capture_active else ["r", "q"],
            control_hint=(
                "左右手中立姿态；点击“下一步”或按 n 开始本阶段。"
                if not capture_active
                else "左右手中立姿态；该阶段只标定 tracker offset，不保存 runtime Home。"
            ),
            capture_stage=stage_id,
            capture_phase=capture_phase,
            capturing=capture_active,
            capture_channel="left_wrist_pose,right_wrist_pose",
            capture_left_samples=len(left_poses),
            capture_right_samples=len(right_poses),
            capture_min_samples=min_samples,
            capture_left_valid=left_valid,
            capture_right_valid=right_valid,
            capture_elapsed_sec=elapsed if capturing else None,
            capture_duration_sec=(
                float(getattr(args, "hold_capture_sec", 4.0)) + float(getattr(args, "capture_extra_wait_sec", 0.0))
                if waiting_for_samples
                else float(getattr(args, "hold_capture_sec", 4.0))
            ),
            capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
            capture_prep_total_sec=float(getattr(args, "wrist_capture_prep_sec", 5.0)),
            capture_waiting_for_samples=waiting_for_samples,
        )
        if capturing and stage_gate.should_emit("wrist_offset_validity", validity_snapshot, now=loop_start):
            _log("ViveCalibration", f"{stage_id} channel_valid={{'left': {left_valid}, 'right': {right_valid}}}", now=loop_start)
        if capturing and left_valid:
            left_sample_key = _channel_sample_key(left_channel)
            if left_sample_key != left_last_sample_key:
                left_last_sample_key = left_sample_key
                left_poses.append(_context_calibration(components).world_to_desk(as_pose_matrix(getattr(left_channel, "value"))))
            if len(left_poses) >= min_samples and not left_min_reported:
                left_min_reported = True
                ui.append_message(stage_id, f"左手已达到最小样本数 {len(left_poses)}/{min_samples}。")
        if capturing and right_valid:
            right_sample_key = _channel_sample_key(right_channel)
            if right_sample_key != right_last_sample_key:
                right_last_sample_key = right_sample_key
                right_poses.append(_context_calibration(components).world_to_desk(as_pose_matrix(getattr(right_channel, "value"))))
            if len(right_poses) >= min_samples and not right_min_reported:
                right_min_reported = True
                ui.append_message(stage_id, f"右手已达到最小样本数 {len(right_poses)}/{min_samples}。")
        if capturing and stage_gate.should_emit(
            "wrist_offset_live",
            validity_snapshot,
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "ViveCalibration",
                f"{stage_id} live time={elapsed:.1f}/{float(getattr(args, 'hold_capture_sec', 4.0)):.1f}s "
                f"left={len(left_poses)} right={len(right_poses)} valid={{'left': {left_valid}, 'right': {right_valid}}} "
                f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                now=loop_start,
            )
        if capturing and loop_start >= capture_deadline:
            capturing = False
            left_hz = len(left_poses) / max(elapsed, 1e-6)
            right_hz = len(right_poses) / max(elapsed, 1e-6)
            if len(left_poses) >= min_samples and len(right_poses) >= min_samples:
                ui.append_message(stage_id, f"采集接受：left={len(left_poses)} right={len(right_poses)}。")
                return left_poses, right_poses
            if loop_start < capture_extension_deadline:
                capturing = True
                if not waiting_for_samples_reported:
                    waiting_for_samples_reported = True
                    ui.append_message(stage_id, f"样本不足 left={len(left_poses)} right={len(right_poses)}，继续等待。")
                time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))
                continue
            ui.append_message(
                stage_id,
                f"样本不足 left={len(left_poses)} right={len(right_poses)} "
                f"need>={min_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f}，请按 n 重采。",
            )
            left_poses = []
            right_poses = []
            left_min_reported = False
            right_min_reported = False
            waiting_for_samples_reported = False
            left_last_sample_key = None
            right_last_sample_key = None
        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def calibrate_vive_coordinate_profile(
    args: argparse.Namespace,
    context: ViveCoordinateCalibrationContext | object,
    ui: OperatorUiPort,
    *,
    save_profile: Callable[[argparse.Namespace, object], Path] | None = None,
    save_on_complete: bool = True,
) -> Path | None:
    active_sides = _context_active_sides(context)
    if not active_sides:
        raise RuntimeError("VIVE coordinate calibration requires at least one active side.")
    source = getattr(context, "source", None)
    if source is None and isinstance(context, RealTeleopComponents):
        source = _vive_pose_source_from_components(context)
    if source is None:
        raise RuntimeError("No VIVE pose source is attached to the teleop session.")
    calibration = _context_calibration(context)
    if calibration is None:
        raise RuntimeError("No active calibration is attached to the teleop session.")

    def _sweep_validator(poses: list[np.ndarray]) -> str | None:
        fit = fit_signed_axis_from_poses(poses)
        quality = fit.quality
        if quality.extent_m < float(getattr(args, "min_sweep_extent_m", 0.15)):
            return f"extent {quality.extent_m:.3f}m < {float(getattr(args, 'min_sweep_extent_m', 0.15)):.3f}m"
        if quality.variance_ratio < float(getattr(args, "min_axis_variance_ratio", 0.5)):
            return (
                f"variance_ratio {quality.variance_ratio:.3f} "
                f"< {float(getattr(args, 'min_axis_variance_ratio', 0.5)):.3f}"
            )
        if quality.start_end_distance_m < float(getattr(args, "min_start_end_distance_m", 0.08)):
            return (
                f"start_end_distance {quality.start_end_distance_m:.3f}m "
                f"< {float(getattr(args, 'min_start_end_distance_m', 0.08)):.3f}m"
            )
        return None

    _, desk_channel_name = _select_gui_vive_coordinate_sweep_hand(
        ui=ui,
        args=args,
        components=context,
        source=source,
    )
    desk_tracker_role = _tracker_role_for_channel(desk_channel_name)
    ui.append_message("ViveCalibration", f"使用 {desk_channel_name} ({desk_tracker_role}) 做 VIVE 坐标系三轴 sweep。")

    x_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=context,
        source=source,
        stage_name="VirtualDeskX",
        instructions=(
            f"使用 {desk_tracker_role} 做 X 轴扫动，让拟合后的 +X 指向身体前方："
            "从靠近身体后侧的位置开始，向身体前方移动并结束。"
        ),
        timed_flow_hint="按一次采集后，按“后 -> 前”扫动；采集自动结束前停在身体前方。",
        direction_hint="后 -> 前",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )
    y_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=context,
        source=source,
        stage_name="VirtualDeskY",
        instructions="做 Y 轴扫动，让拟合后的 +Y 指向你的左侧：从身体右侧开始，向身体左侧移动并结束。",
        timed_flow_hint="按一次采集后，按“右 -> 左”扫动；采集自动结束前停在身体左侧。",
        direction_hint="右 -> 左",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )
    z_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=context,
        source=source,
        stage_name="VirtualDeskZ",
        instructions="做 Z 轴扫动，让拟合后的 +Z 指向上方：从低处开始，向上移动并结束。",
        timed_flow_hint="按一次采集后，按“低 -> 高”扫动；采集自动结束前不要向下返回。",
        direction_hint="低 -> 高",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )

    result = fit_virtual_desk_frame(
        x_sweep_poses=x_sweep,
        y_sweep_poses=y_sweep,
        z_sweep_poses=z_sweep,
    )
    max_raw_non_orthogonality = max(
        result.xy_non_orthogonality_deg,
        result.xz_non_orthogonality_deg,
        result.yz_non_orthogonality_deg,
    )
    max_axis_correction = max(
        result.x_axis_correction_deg,
        result.y_axis_correction_deg,
        result.z_axis_correction_deg,
    )
    max_allowed_axis_correction = float(getattr(args, "max_axis_non_orthogonality_deg", 20.0))
    if max_axis_correction > max_allowed_axis_correction:
        worst_axis_name, worst_axis_correction = max(
            (
                ("x", result.x_axis_correction_deg),
                ("y", result.y_axis_correction_deg),
                ("z", result.z_axis_correction_deg),
            ),
            key=lambda item: item[1],
        )
        raise RuntimeError(
            "Virtual desk fit failed axis-correction check: "
            f"worst_axis={worst_axis_name} correction={worst_axis_correction:.2f}deg "
            f"> {max_allowed_axis_correction:.2f}deg. "
            f"raw xy={result.xy_non_orthogonality_deg:.2f}deg "
            f"xz={result.xz_non_orthogonality_deg:.2f}deg "
            f"yz={result.yz_non_orthogonality_deg:.2f}deg. "
            "请按提示方向重新扫 X/Y/Z。"
        )
    if max_raw_non_orthogonality > max_allowed_axis_correction:
        ui.append_message(
            "ViveCalibration",
            f"raw sweep 非正交较大 {max_raw_non_orthogonality:.2f}deg，但正交拟合已接受。",
        )
    base_world_to_desk = result.T_W_V.copy()
    yaw_deg = float(getattr(args, "virtual_desk_post_yaw_deg", 180.0))
    world_to_desk = base_world_to_desk @ _virtual_desk_post_rotation_pose(yaw_deg)
    calibration.set_world_to_desk(world_to_desk)
    calibration.metadata.update(
        {
            "calibration_mode": "vive_rm65_virtual_desk",
            "virtual_desk": result.to_metadata(),
            "virtual_desk_tracker_role": desk_tracker_role,
            "virtual_desk_channel_name": desk_channel_name,
            "virtual_desk_sweep_convention_version": VIRTUAL_DESK_SWEEP_CONVENTION_VERSION,
            "virtual_desk_post_yaw_deg": yaw_deg,
            "runtime_gui_vive_calibrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    ui.append_message(
        "ViveCalibration",
        f"VIVE desk 坐标系拟合完成，post_yaw={yaw_deg:.0f}deg；接下来采左右 wrist tracker offset。",
    )

    if set(active_sides) == {"left", "right"}:
        left_poses, right_poses = _capture_dual_wrist_offsets_from_gui(
            ui=ui,
            args=args,
            components=context,
            source=source,
        )
        per_side_poses = {"left": left_poses, "right": right_poses}
    else:
        per_side_poses = {}
        for side in active_sides:
            per_side_poses[side] = _capture_single_vive_channel_from_gui(
                ui=ui,
                args=args,
                components=context,
                source=source,
                stage_name="WristOffset",
                instructions=(
                    f"把 {side} wrist tracker 放到标准中立姿态：前臂朝前、手背朝上、"
                    "手指大致沿 desk +X。"
                ),
                timed_flow_hint="保持中立姿态，采集自动结束后写入 tracker offset。",
                direction_hint="保持中立姿态",
                channel_name=f"{side}_wrist_pose",
                min_samples=int(getattr(args, "min_hold_samples", 3)),
                capture_duration_sec=float(getattr(args, "hold_capture_sec", 4.0)),
                validator=None,
                pose_transform=lambda pose, calibration=calibration: calibration.world_to_desk(pose),
            )
    tracker_metadata = calibration.metadata.setdefault("tracker_offset_calibration", {})
    if not isinstance(tracker_metadata, dict):
        tracker_metadata = {}
        calibration.metadata["tracker_offset_calibration"] = tracker_metadata
    for side, poses in per_side_poses.items():
        tracker_role = f"{side}_wrist"
        canonical_rpy_deg = canonical_hand_rpy_default()
        hand_rotation_virtual = euler_xyz_deg_to_matrix(canonical_rpy_deg)
        tracker_pose_virtual = average_pose(poses)
        offset = tracker_offset_from_neutral_pose(
            tracker_pose_virtual,
            hand_rotation_virtual=hand_rotation_virtual,
            translation_tracker_to_hand=np.zeros(3, dtype=float),
        )
        calibration.set_tracker_offset(tracker_role, offset)
        tracker_metadata[tracker_role] = {
            "method": "runtime_gui_wrist_neutral_capture",
            "canonical_hand_rpy_deg": canonical_rpy_deg.tolist(),
            "canonical_hand_frame_version": TRACKER_OFFSET_CALIBRATION_VERSION,
            "translation_tracker_to_hand": [0.0, 0.0, 0.0],
            "coarse_offset": offset.tolist(),
            "fine_tune_rpy_deg": [0.0, 0.0, 0.0],
            "final_offset": offset.tolist(),
            "captured_tracker_pose_virtual": tracker_pose_virtual.tolist(),
        }
    _clear_runtime_home_state(context)
    if save_profile is None or not save_on_complete:
        return None
    path = save_profile(args, context)
    setattr(args, "load_profile", str(path))
    return path


def calibrate_gui_vive_coordinate_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    ui: OperatorUiPort,
    *,
    save_profile: Callable[[argparse.Namespace, RealTeleopComponents], Path],
) -> Path:
    """Compatibility wrapper for the existing real-teleop GUI entrypoint."""

    source = _vive_pose_source_from_components(components)
    if source is None:
        raise RuntimeError("No VIVE pose source is attached to the teleop session.")
    calibration = components.session.calibration
    if calibration is None:
        raise RuntimeError("No active calibration is attached to the teleop session.")
    result = calibrate_vive_coordinate_profile(
        args,
        ViveCoordinateCalibrationContext(
            source=source,
            calibration=calibration,
            active_sides=tuple(components.active_sides),
        ),
        ui,
        save_profile=lambda workflow_args, _context: save_profile(workflow_args, components),
    )
    if result is None:
        raise RuntimeError("VIVE coordinate calibration did not produce a profile path.")
    return result

__all__ = [
    "_capture_dual_wrist_offsets_from_gui",
    "_capture_single_vive_channel_from_gui",
    "_channel_for_tracker_role",
    "_channel_sample_key",
    "_clear_runtime_home_state",
    "_consume_vive_capture_events",
    "_default_gui_vive_calibration_side",
    "_format_vive_source_rates",
    "_format_vive_sweep_quality",
    "_format_wall_time",
    "_publish_vive_capture_state",
    "_resolve_virtual_desk_channel_name",
    "_select_gui_vive_coordinate_sweep_hand",
    "_source_debug_stats",
    "_tracker_role_for_channel",
    "_virtual_desk_post_rotation_pose",
    "ViveCoordinateCalibrationContext",
    "calibrate_vive_coordinate_profile",
    "calibrate_gui_vive_coordinate_profile",
]
