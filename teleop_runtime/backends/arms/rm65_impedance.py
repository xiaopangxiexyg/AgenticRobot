from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from hardware.rm65.calibration_profiles import tcp_endpoint_delta, tcp_safety_warnings
from hardware.rm65.calibration_profiles import load_profile
from hardware.rm65.impedance import RM65ImpedanceConfig, RM65PoseImpedanceController

from ...arm_frames import (
    ARM_ENDPOINT_FRAME,
    TOOL_TCP_FRAME,
    canonical_arm_controlled_frame,
    convert_pose_between_arm_control_frames,
)
from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand
from ...transforms import as_pose_matrix
from .rm65 import _load_rm65_sdk, _parse_rm65_result, _pose6_to_matrix


def _tool_name_from_frame_result(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return ""


def _matrix_to_pose6(matrix: np.ndarray) -> list[float]:
    pose = np.asarray(matrix, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose matrix contains non-finite values.")
    rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    return [
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    ]


def _profile_impedance_section(profile: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    impedance = profile.get("impedance")
    if not isinstance(impedance, Mapping):
        return {}
    section = impedance.get(mode)
    return section if isinstance(section, Mapping) else {}


def _strict_profile_errors(
    profile: Mapping[str, Any],
    *,
    side: str,
    active_tool_name: str | None,
    mode: str,
) -> list[str]:
    errors: list[str] = []
    if str(profile.get("arm") or "") != side:
        errors.append(f"profile arm={profile.get('arm')!r} does not match backend side={side!r}")
    profile_tool = str(profile.get("tool_name") or "")
    if not profile_tool:
        errors.append("profile tool_name is empty")
    if active_tool_name and profile_tool and profile_tool not in active_tool_name:
        errors.append(
            f"active tool {active_tool_name!r} does not contain profile tool_name {profile_tool!r}"
        )
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), Mapping) else {}
    pose = tcp.get("pose_flange_m_rad") if isinstance(tcp, Mapping) else None
    if pose is None:
        errors.append("profile tcp.pose_flange_m_rad is missing")
    else:
        try:
            values = np.asarray(pose, dtype=float).reshape(-1)
        except Exception:
            errors.append("profile tcp.pose_flange_m_rad is not numeric")
        else:
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                errors.append("profile tcp.pose_flange_m_rad must contain 6 finite values")
    if not bool(tcp.get("confirmed")):
        errors.append("profile tcp.confirmed must be true for strict impedance")
    payload = profile.get("payload") if isinstance(profile.get("payload"), Mapping) else {}
    if payload.get("mass_kg") is None:
        errors.append("profile payload.mass_kg is missing")
    if payload.get("com_tool_m") is None:
        errors.append("profile payload.com_tool_m is missing")
    else:
        try:
            com = np.asarray(payload.get("com_tool_m"), dtype=float).reshape(-1)
        except Exception:
            errors.append("profile payload.com_tool_m is not numeric")
        else:
            if com.shape != (3,) or not np.all(np.isfinite(com)):
                errors.append("profile payload.com_tool_m must contain 3 finite values")
    if not _profile_impedance_section(profile, mode):
        errors.append(f"profile impedance.{mode} is missing")
    return errors


def _warn_profile_tcp_safety(
    *,
    owner: str,
    profile: Mapping[str, Any],
) -> None:
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), Mapping) else {}
    pose = tcp.get("pose_flange_m_rad") if isinstance(tcp, Mapping) else None
    if pose is None:
        return
    try:
        warnings = tcp_safety_warnings(pose, label=f"{owner} TCP")
    except Exception:
        return
    for warning in warnings:
        print(f"WARNING: {warning}; profile={profile.get('_profile_path') or profile.get('name')}. Verify before impedance control.")


@dataclass
class RM65ImpedanceArmBackendConfig:
    arm_ip: str
    side: str
    effector_name: str | None = None
    connect_hardware: bool = True
    tool_profile: str | Path | None = None
    auto_load_tool_profile: bool = True
    strict_tool_profile: bool = True
    impedance_mode: str = "pose"
    force_tare_sec: float = 3.0
    control_dt: float = 1.0 / 30.0
    action_hz: float | None = None
    max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    actual_send_dt_alpha: float = 0.25
    actual_send_dt_min_scale: float = 1.0
    actual_send_dt_max_scale: float = 2.0
    actual_send_dt_stall_reset_sec: float = 0.25
    impedance_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if self.impedance_mode != "pose":
            raise ValueError("RM65ImpedanceArmBackend currently supports impedance_mode='pose'.")
        if float(self.force_tare_sec) <= 0.0:
            raise ValueError("force_tare_sec must be positive.")
        if float(self.control_dt) <= 0.0:
            raise ValueError("control_dt must be positive.")
        if self.action_hz is not None and float(self.action_hz) <= 0.0:
            raise ValueError("action_hz must be positive when provided.")
        if len(self.max_angvel) != 6:
            raise ValueError("max_angvel must contain 6 values.")
        if float(self.actual_send_dt_stall_reset_sec) <= 0.0:
            raise ValueError("actual_send_dt_stall_reset_sec must be positive.")


@register_backend("rm65_pose_impedance_arm")
class RM65ImpedanceArmBackend(EffectorBackend):
    def __init__(self, config: RM65ImpedanceArmBackendConfig):
        self.config = config
        self.arm = None
        self._controller: RM65PoseImpedanceController | None = None
        self._profile: dict[str, Any] | None = None
        self._active_tool_name: str | None = None
        self._active_tool_profile_path: str | None = None
        self._endpoint_to_tool_tcp = np.eye(4, dtype=float)
        self._latest_target_pose6: list[float] | None = None
        self._last_command: EffectorCommand | None = None
        self._last_command_timestamp: float | None = None
        self._last_step_record: dict[str, object] | None = None
        self._last_fault: str | None = None
        self._last_send_result_code: int | None = None
        self._last_send_result_text: str | None = None
        self._enabled = False
        self._state_lock = threading.RLock()
        self._loop_stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._last_send_time: float | None = None
        self._smoothed_send_dt = float(config.control_dt)
        self._last_effective_control_dt = float(config.control_dt)
        self._last_effective_step_scale = 1.0

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.arm is not None:
            return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
        socket_code = int(self.arm.Arm_Socket_State())
        if socket_code != 0:
            raise RuntimeError(f"{self.name} failed to connect RM65 ip={self.config.arm_ip}: {socket_code}")
        self._active_tool_name = self._read_active_tool_name()
        self._profile = self._load_profile(self._active_tool_name)
        self._validate_profile(self._profile, self._active_tool_name)
        tcp = self._profile.get("tcp") if isinstance(self._profile.get("tcp"), Mapping) else {}
        self._endpoint_to_tool_tcp = _pose6_to_matrix(tcp.get("pose_flange_m_rad", [0.0] * 6))
        tcp_metrics = tcp_endpoint_delta(tcp.get("pose_flange_m_rad", [0.0] * 6))
        print(
            f"{self.name}: impedance tool_tcp endpoint_delta="
            f"{tcp_metrics['translation_m']:.4f}m/{tcp_metrics['rotation_deg']:.2f}deg"
        )
        self._active_tool_profile_path = str(self._profile.get("_profile_path") or "")
        impedance_config = self._build_impedance_config(self._profile)
        self._controller = RM65PoseImpedanceController(
            arm=self.config.side,
            config=impedance_config,
            robot=self.arm,
            left_ip=self.config.arm_ip,
            right_ip=self.config.arm_ip,
        )

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or self.arm is not None

    def enable_force_control(self) -> None:
        if self._enabled:
            return
        if not self.config.connect_hardware:
            self._enabled = True
            return
        self.start()
        assert self._controller is not None
        self._controller.start()
        self._enabled = True
        self._loop_stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._impedance_loop,
            daemon=True,
            name=f"{self.name}_pose_impedance",
        )
        self._loop_thread.start()

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for effector '{command.effector_name}'.")
        if command.command_type != "cartesian_pose":
            raise ValueError(
                f"{self.name} pose impedance requires cartesian_pose command, got {command.command_type!r}."
            )
        if self.config.connect_hardware and not self._enabled:
            raise RuntimeError(f"{self.name} force control is not enabled; run tare/enable after runtime home first.")
        now = time.time()
        target_pose6 = self._command_to_tool_tcp_pose6(command)
        with self._state_lock:
            self._latest_target_pose6 = list(target_pose6)
            self._last_command = command.copy()
            self._last_command_timestamp = now
            self._update_dt_scale(now)
        if not self.config.connect_hardware:
            self._last_send_result_code = 0
            self._last_send_result_text = "dry_run"

    def _read_active_tool_name(self) -> str | None:
        if self.arm is None:
            return None
        code, frame = self.arm.Get_Current_Tool_Frame()
        if int(code) != 0:
            raise RuntimeError(f"{self.name} Get_Current_Tool_Frame failed: {code}")
        return _tool_name_from_frame_result((code, frame)) or None

    def _load_profile(self, active_tool_name: str | None) -> dict[str, Any]:
        requested = self.config.tool_profile
        if requested:
            return load_profile(requested, arm=self.config.side)
        if active_tool_name and self.config.auto_load_tool_profile:
            return load_profile(active_tool_name, arm=self.config.side)
        raise RuntimeError(
            f"{self.name} needs a tool profile for impedance. "
            "Pass --left/--right-tool-profile or enable a named active RM65 tool frame."
        )

    def _validate_profile(self, profile: Mapping[str, Any], active_tool_name: str | None) -> None:
        _warn_profile_tcp_safety(owner=self.name, profile=profile)
        if not self.config.strict_tool_profile:
            return
        errors = _strict_profile_errors(
            profile,
            side=self.config.side,
            active_tool_name=active_tool_name,
            mode=self.config.impedance_mode,
        )
        if errors:
            raise RuntimeError(f"{self.name} strict impedance profile validation failed: " + "; ".join(errors))

    def _build_impedance_config(self, profile: Mapping[str, Any]) -> RM65ImpedanceConfig:
        section = _profile_impedance_section(profile, self.config.impedance_mode)
        control_hz = float(section.get("control_hz", 40.0) or 40.0)
        requested_action_hz = (
            float(self.config.action_hz)
            if self.config.action_hz is not None
            else 1.0 / float(self.config.control_dt)
        )
        overrides = {
            "tare_sec": float(self.config.force_tare_sec),
            "action_hz": min(requested_action_hz, control_hz),
            **dict(self.config.impedance_overrides),
        }
        return RM65ImpedanceConfig.from_profile(
            dict(profile),
            mode=self.config.impedance_mode,
            **overrides,
        )

    def _command_to_tool_tcp_pose6(self, command: EffectorCommand) -> list[float]:
        controlled_frame = canonical_arm_controlled_frame(
            command.metadata.get("arm_controlled_frame")
            or command.metadata.get("target_pose_controlled_frame")
            or TOOL_TCP_FRAME
        )
        if controlled_frame == TOOL_TCP_FRAME:
            return _matrix_to_pose6(as_pose_matrix(command.value))
        endpoint_pose = convert_pose_between_arm_control_frames(
            as_pose_matrix(command.value),
            from_frame=controlled_frame,
            to_frame=ARM_ENDPOINT_FRAME,
        )
        tool_tcp_pose = endpoint_pose @ self._endpoint_to_tool_tcp
        return _matrix_to_pose6(tool_tcp_pose)

    def _update_dt_scale(self, now: float) -> None:
        if self._last_send_time is None:
            smoothed_dt = float(self.config.control_dt)
        else:
            raw_dt = max(0.0, now - float(self._last_send_time))
            if raw_dt > float(self.config.actual_send_dt_stall_reset_sec):
                smoothed_dt = float(self.config.control_dt)
            else:
                alpha = float(self.config.actual_send_dt_alpha)
                smoothed_dt = alpha * raw_dt + (1.0 - alpha) * float(self._smoothed_send_dt)
        effective_dt = float(
            np.clip(
                smoothed_dt,
                float(self.config.control_dt) * float(self.config.actual_send_dt_min_scale),
                float(self.config.control_dt) * float(self.config.actual_send_dt_max_scale),
            )
        )
        self._last_send_time = now
        self._smoothed_send_dt = smoothed_dt
        self._last_effective_control_dt = effective_dt
        self._last_effective_step_scale = effective_dt / float(self.config.control_dt)

    def _impedance_loop(self) -> None:
        assert self._controller is not None
        period = 1.0 / float(self._controller.config.control_hz)
        next_deadline = time.perf_counter()
        while not self._loop_stop_event.is_set():
            with self._state_lock:
                target = None if self._latest_target_pose6 is None else list(self._latest_target_pose6)
            try:
                record = self._controller.step(target)
            except Exception as exc:
                fault = f"{type(exc).__name__}: {exc}"
                with self._state_lock:
                    self._last_fault = fault
                    self._last_send_result_code = -1
                    self._last_send_result_text = fault
                try:
                    self._controller.stop_motion()
                finally:
                    self._enabled = False
                    self._loop_stop_event.set()
                break
            with self._state_lock:
                self._last_step_record = dict(record)
                self._last_send_result_code = 0
                self._last_send_result_text = "ok"
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                self._loop_stop_event.wait(sleep_sec)

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            command = None if self._last_command is None else self._last_command.copy()
            record = None if self._last_step_record is None else dict(self._last_step_record)
            current_qpos = None
            current_pose = None
            if record is not None:
                joint_deg = record.get("command_joint_deg")
                pose6 = record.get("command_pose")
                if joint_deg is not None:
                    current_qpos = np.asarray(joint_deg, dtype=float).reshape(6) / 180.0 * np.pi
                if pose6 is not None:
                    current_pose = _pose6_to_matrix(pose6)
            return {
                "current_qpos": current_qpos,
                "current_tcp_pose": current_pose,
                "last_target_qpos": current_qpos,
                "last_command": command,
                "last_command_timestamp": self._last_command_timestamp,
                "arm_state_source": "pose_impedance_command",
                "hardware_result_code": 0 if self._last_fault is None else -1,
                "hardware_result_text": self._last_fault or "ok",
                "send_result_code": self._last_send_result_code,
                "send_result_text": self._last_send_result_text,
                "effective_control_dt_sec": float(self._last_effective_control_dt),
                "effective_step_scale": float(self._last_effective_step_scale),
                "impedance_enabled": bool(self._enabled),
                "impedance_mode": self.config.impedance_mode,
                "force_tare_sec": float(self.config.force_tare_sec),
                "active_tool_name": self._active_tool_name,
                "active_tool_profile_path": self._active_tool_profile_path,
                "last_impedance_record": record,
                "fault": self._last_fault,
            }

    def close(self) -> None:
        self._loop_stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.0)
            self._loop_thread = None
        if self._controller is not None:
            try:
                self._controller.stop_motion()
            except Exception:
                pass
            self._controller.close()
            self._controller = None
        if self.arm is not None:
            try:
                self.arm.Arm_Socket_Close()
            finally:
                self.arm = None
        self._enabled = False
