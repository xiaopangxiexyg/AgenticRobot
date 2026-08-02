"""Web-GUI workflow for AngleTrigger side assignment and endpoint calibration."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping

from hardware.angle_trigger.angle_trigger import (
    DEFAULT_ASSIGNMENT_MOTION_MIN_DEG,
    DEFAULT_BAUDRATE,
    DEFAULT_CAPTURE_SECONDS,
    DEFAULT_ENDPOINT_MIN_SEPARATION_DEG,
    DEFAULT_SAMPLE_HZ,
    AngleTriggerError,
    SensorCandidate,
    SensorConnection,
    discover_sensors,
    load_angle_calibration_config,
    load_assignment_config,
    load_configured_reader,
    resolve_assignment_with_unique_completion,
    save_angle_calibration_config,
    save_assignment_config,
)
from .topology import SIDES


ANGLE_NEXT_EVENT = "angle_calibration_next"
ANGLE_CONFIRM_ASSIGNMENT_EVENT = "angle_assignment_confirm"


class AngleTriggerCalibrationCancelled(RuntimeError):
    pass


def ensure_default_command_input_from_gui(
    ui: Any,
    config: Any,
    *,
    force: bool = False,
) -> bool:
    """Run the default AngleTrigger bootstrap behind the generic app port."""

    retry_force = bool(force)
    while True:
        try:
            return ensure_angle_trigger_ready(
                ui,
                config_root=config.angle_trigger_config_root,
                rate_hz=config.angle_trigger_rate_hz,
                force=retry_force,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            ui.publish(
                ui_mode="umi_collection",
                stage="AngleTrigger 标定未完成",
                teleop_running=False,
                episode_recording_available=False,
                episode_event_status="error",
                episode_event_message=str(exc),
                active_controls=["angle_recalibrate", "q"],
                control_hint="检查两个 Angle 传感器连接后点击“重新标定 Angle”。",
            )
            while True:
                events = set(ui.consume_events())
                if "q" in events:
                    return False
                if "angle_recalibrate" in events:
                    retry_force = True
                    break
                time.sleep(0.1)


def ensure_angle_trigger_ready(
    ui: Any,
    *,
    config_root: Path | None,
    rate_hz: float,
    force: bool = False,
) -> bool:
    """Validate AngleTrigger config and enter GUI calibration when needed."""

    reason = "操作员请求重新标定 AngleTrigger。"
    if not force:
        try:
            load_configured_reader(config_root=config_root)
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc)
        else:
            return True
        sensors = discover_sensors()
        mapping = None
        inferred = None
        if len(sensors) == 2:
            try:
                assignment = load_assignment_config(config_root=config_root)
                mapping, inferred, _errors = resolve_assignment_with_unique_completion(
                    assignment, sensors
                )
            except (OSError, ValueError, RuntimeError):
                mapping = None
        if mapping is not None:
            if inferred is not None:
                decision = _confirm_completed_assignment(
                    ui,
                    mapping=mapping,
                    inferred=inferred,
                    reason=reason,
                )
                if decision == "quit":
                    return False
                if decision == "recalibrate":
                    mapping = None
                else:
                    save_assignment_config(mapping, config_root=config_root)
            if mapping is not None:
                try:
                    load_angle_calibration_config(config_root=config_root)
                except (OSError, ValueError, RuntimeError):
                    try:
                        _run_endpoint_only_gui_calibration(
                            ui,
                            mapping=mapping,
                            config_root=config_root,
                            sample_hz=float(rate_hz),
                        )
                    except AngleTriggerCalibrationCancelled:
                        return False
                return True
        if mapping is None:
            decision = _wait_for_recalibration_choice(ui, reason=reason)
            if decision == "quit":
                return False
    try:
        run_angle_trigger_gui_calibration(
            ui,
            config_root=config_root,
            sample_hz=float(rate_hz),
            reason=reason,
        )
    except AngleTriggerCalibrationCancelled:
        return False
    return True


def _confirm_completed_assignment(
    ui: Any,
    *,
    mapping: Mapping[str, SensorCandidate],
    inferred: tuple[str, SensorCandidate],
    reason: str,
) -> str:
    inferred_side, inferred_sensor = inferred
    ui.publish(
        ui_mode="umi_collection",
        stage="确认 AngleTrigger 左右自动补全",
        teleop_running=False,
        episode_recording_available=False,
        episode_event_status="calibrating",
        episode_event_message=(
            f"{reason} 已唯一匹配另一侧，因此自动补全 "
            f"{inferred_side}={inferred_sensor.label}。"
            f"完整结果：left={mapping['left'].label}；right={mapping['right'].label}。"
        ),
        active_controls=[
            ANGLE_CONFIRM_ASSIGNMENT_EVENT,
            "angle_recalibrate",
            "q",
        ],
        control_hint="确认自动判断，或选择重新标定 Angle。",
    )
    while True:
        events = set(ui.consume_events())
        if "q" in events:
            return "quit"
        if "angle_recalibrate" in events:
            return "recalibrate"
        if ANGLE_CONFIRM_ASSIGNMENT_EVENT in events:
            return "confirm"
        time.sleep(0.05)


def _wait_for_recalibration_choice(ui: Any, *, reason: str) -> str:
    ui.publish(
        ui_mode="umi_collection",
        stage="AngleTrigger 左右无法自动补全",
        teleop_running=False,
        episode_recording_available=False,
        episode_event_status="error",
        episode_event_message=reason,
        active_controls=["angle_recalibrate", "q"],
        control_hint="无法得到唯一左右映射；请选择重新标定 Angle。",
    )
    while True:
        events = set(ui.consume_events())
        if "q" in events:
            return "quit"
        if "angle_recalibrate" in events:
            return "recalibrate"
        time.sleep(0.05)


def run_angle_trigger_gui_calibration(
    ui: Any,
    *,
    config_root: Path | None,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
    baudrate: int = DEFAULT_BAUDRATE,
    capture_seconds: float = DEFAULT_CAPTURE_SECONDS,
    min_motion_deg: float = DEFAULT_ASSIGNMENT_MOTION_MIN_DEG,
    reason: str = "",
) -> None:
    if sample_hz <= 0.0 or capture_seconds <= 0.0:
        raise ValueError("AngleTrigger sample rate and capture duration must be positive")
    sensors = discover_sensors(baudrate=baudrate)
    if len(sensors) != 2:
        raise AngleTriggerError(
            f"AngleTrigger GUI 标定需要恰好两个传感器，当前发现 {len(sensors)} 个。"
        )
    candidates = {
        f"sensor_{index}": SensorConnection(sensor, baudrate=baudrate)
        for index, sensor in enumerate(sensors, start=1)
    }
    try:
        for connection in candidates.values():
            connection.open()
        mapping = _assign_sides_in_gui(
            ui,
            candidates=candidates,
            sensors=sensors,
            sample_hz=sample_hz,
            min_motion_deg=min_motion_deg,
            reason=reason,
        )
        assignment_file = save_assignment_config(mapping, config_root=config_root)
        profiles = _calibrate_endpoints_in_gui(
            ui,
            mapping=mapping,
            connections={
                side: candidates[
                    next(
                        name
                        for name, sensor in zip(candidates, sensors)
                        if sensor == mapping[side]
                    )
                ]
                for side in SIDES
            },
            sample_hz=sample_hz,
            capture_seconds=capture_seconds,
        )
        angles_file = save_angle_calibration_config(profiles, config_root=config_root)
    finally:
        for connection in candidates.values():
            connection.close()
    ui.publish(
        stage="AngleTrigger 标定完成",
        episode_event_status="ready",
        episode_event_message=(
            f"左右分配：{assignment_file}；角度端点：{angles_file}。正在继续初始化。"
        ),
        active_controls=["q"],
        control_hint="AngleTrigger 配置已保存。",
    )


def _run_endpoint_only_gui_calibration(
    ui: Any,
    *,
    mapping: Mapping[str, SensorCandidate],
    config_root: Path | None,
    sample_hz: float,
    baudrate: int = DEFAULT_BAUDRATE,
    capture_seconds: float = DEFAULT_CAPTURE_SECONDS,
) -> None:
    connections = {
        side: SensorConnection(mapping[side], baudrate=baudrate)
        for side in SIDES
    }
    try:
        for connection in connections.values():
            connection.open()
        profiles = _calibrate_endpoints_in_gui(
            ui,
            mapping=mapping,
            connections=connections,
            sample_hz=sample_hz,
            capture_seconds=capture_seconds,
        )
        save_angle_calibration_config(profiles, config_root=config_root)
    finally:
        for connection in connections.values():
            connection.close()


def _assign_sides_in_gui(
    ui: Any,
    *,
    candidates: Mapping[str, SensorConnection],
    sensors: list[SensorCandidate],
    sample_hz: float,
    min_motion_deg: float,
    reason: str,
) -> dict[str, SensorCandidate]:
    sensor_by_name = dict(zip(candidates, sensors))
    remaining = dict(candidates)
    selected: dict[str, SensorCandidate] = {}
    for side, side_zh in (("left", "左侧"), ("right", "右侧")):
        while True:
            ui.publish(
                ui_mode="umi_collection",
                stage=f"AngleTrigger {side_zh}分配",
                teleop_running=False,
                episode_recording_available=False,
                episode_event_status="calibrating",
                episode_event_message=(
                    f"{reason + ' ' if reason else ''}持续移动{side_zh} Angle 传感器，"
                    "形成明显角度范围后点击“确认 / 采集当前 Angle 步骤”。"
                ),
                active_controls=[ANGLE_NEXT_EVENT, "q"],
                control_hint=f"移动{side_zh}传感器，然后点击确认。",
            )
            samples = _sample_until_gui_event(
                ui,
                connections=remaining,
                sample_hz=sample_hz,
            )
            ranges = {
                name: max(values) - min(values) if len(values) >= 3 else 0.0
                for name, values in samples.items()
            }
            selected_name, selected_range = max(
                ranges.items(), key=lambda item: item[1]
            )
            if selected_range >= min_motion_deg:
                selected[side] = sensor_by_name[selected_name]
                remaining.pop(selected_name)
                break
            reason = (
                f"未检测到足够运动：最大范围 {selected_range:.2f}°，"
                f"至少需要 {min_motion_deg:.2f}°。请重试。"
            )
    return selected


def _sample_until_gui_event(
    ui: Any,
    *,
    connections: Mapping[str, SensorConnection],
    sample_hz: float,
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {name: [] for name in connections}
    period = 1.0 / sample_hz
    while True:
        started = time.monotonic()
        for name, connection in connections.items():
            try:
                samples[name].append(connection.read_angle_deg())
            except AngleTriggerError:
                continue
        events = set(ui.consume_events())
        if "q" in events:
            raise AngleTriggerCalibrationCancelled("AngleTrigger calibration cancelled")
        if ANGLE_NEXT_EVENT in events:
            return samples
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def _calibrate_endpoints_in_gui(
    ui: Any,
    *,
    mapping: Mapping[str, SensorCandidate],
    connections: Mapping[str, SensorConnection],
    sample_hz: float,
    capture_seconds: float,
) -> dict[str, dict[str, float]]:
    del mapping
    profiles: dict[str, dict[str, float]] = {}
    for side, side_zh in (("left", "左侧"), ("right", "右侧")):
        values: dict[str, float] = {}
        for label, label_zh in (("zero_deg", "0（松开）"), ("one_deg", "1（夹紧）")):
            _wait_for_gui_capture(
                ui,
                stage=f"AngleTrigger {side_zh}端点 {label_zh}",
                message=(
                    f"将{side_zh}传感器保持在 {label_zh} 位置，"
                    "点击“确认 / 采集当前 Angle 步骤”后保持不动。"
                ),
            )
            ui.publish(
                stage=f"正在采集 {side_zh}端点 {label_zh}",
                episode_event_status="calibrating",
                episode_event_message=f"采集 {capture_seconds:.1f} 秒，请保持不动。",
                active_controls=["q"],
            )
            mean, std, count = _capture_endpoint(
                connections[side],
                duration_s=capture_seconds,
                sample_hz=sample_hz,
            )
            values[label] = mean
            ui.publish(
                episode_event_message=(
                    f"{side_zh} {label_zh}: {mean:.2f}°，std={std:.2f}°，n={count}。"
                )
            )
        if abs(values["one_deg"] - values["zero_deg"]) < DEFAULT_ENDPOINT_MIN_SEPARATION_DEG:
            raise AngleTriggerError(
                f"{side_zh} 0/1 端点差小于 {DEFAULT_ENDPOINT_MIN_SEPARATION_DEG:.1f}°，"
                "标定未保存。"
            )
        profiles[side] = values
    return profiles


def _wait_for_gui_capture(ui: Any, *, stage: str, message: str) -> None:
    ui.publish(
        ui_mode="umi_collection",
        stage=stage,
        teleop_running=False,
        episode_recording_available=False,
        episode_event_status="calibrating",
        episode_event_message=message,
        active_controls=[ANGLE_NEXT_EVENT, "q"],
        control_hint=message,
    )
    while True:
        events = set(ui.consume_events())
        if "q" in events:
            raise AngleTriggerCalibrationCancelled("AngleTrigger calibration cancelled")
        if ANGLE_NEXT_EVENT in events:
            return
        time.sleep(0.05)


def _capture_endpoint(
    connection: SensorConnection,
    *,
    duration_s: float,
    sample_hz: float,
) -> tuple[float, float, int]:
    samples: list[float] = []
    deadline = time.monotonic() + duration_s
    period = 1.0 / sample_hz
    while time.monotonic() < deadline:
        started = time.monotonic()
        samples.append(connection.read_angle_deg())
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(min(remaining, max(0.0, deadline - time.monotonic())))
    minimum = max(3, int(duration_s * sample_hz * 0.6))
    if len(samples) < minimum:
        raise AngleTriggerError(
            f"Angle endpoint only collected {len(samples)} samples; need {minimum}."
        )
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return mean, math.sqrt(variance), len(samples)


__all__ = [
    "ANGLE_NEXT_EVENT",
    "ANGLE_CONFIRM_ASSIGNMENT_EVENT",
    "ensure_default_command_input_from_gui",
    "ensure_angle_trigger_ready",
    "run_angle_trigger_gui_calibration",
]
