from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from hardware.rm65.calibration_profiles import load_profile
from hardware.rm65.force_compensation import (
    PoseAdmittance,
    WrenchProcessor,
    angle_delta,
    apply_pose_offset,
    finite_list,
    mean_wrench,
    normalize_pose,
    norm,
    parse_axes,
)
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


ActionSource = Callable[[int, float, dict[str, object] | None], Iterable[float] | None]


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(arm: str, left_ip: str = LEFT_ARM_IP, right_ip: str = RIGHT_ARM_IP) -> tuple[str, str]:
    if arm == "left":
        return "left", left_ip
    if arm == "right":
        return "right", right_ip
    raise ValueError("arm must be left or right")


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": finite_list(joint_deg, expected_len=6, label="joint_deg"),
        "pose": finite_list(pose, expected_len=6, label="pose"),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": finite_list(raw, expected_len=6, label="raw"),
        "zero": finite_list(zero, expected_len=6, label="zero"),
        "work": finite_list(work, expected_len=6, label="work"),
        "tool": finite_list(tool, expected_len=6, label="tool"),
    }


def _current_tool_name(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return str(frame_result).strip("\x00")


def _current_tool_pose(frame_result: object) -> list[float] | None:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict) and frame.get("pose") is not None:
        try:
            return finite_list(frame["pose"], expected_len=6, label="tool_pose")
        except ValueError:
            return None
    return None


def _tool_payload(frame_result: object) -> tuple[float, list[float]] | None:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if not isinstance(frame, dict):
        return None
    try:
        payload = float(frame.get("payload"))
        com = [float(frame.get(key)) for key in ("x", "y", "z")]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in [payload, *com]):
        return None
    if norm(com) > 1.0:
        com = [value * 0.001 for value in com]
    return payload, com


def _pose_delta(a: list[float], b: list[float]) -> tuple[float, float]:
    linear = norm(a[index] - b[index] for index in range(3))
    angular = norm(angle_delta(a[index], b[index]) for index in range(3, 6))
    return linear, angular


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _format_vector(values: Iterable[float], *, precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _interpolate_pose(a: list[float], b: list[float], alpha: float) -> list[float]:
    alpha = max(0.0, min(1.0, float(alpha)))
    out = [
        a[index] + (b[index] - a[index]) * alpha
        for index in range(3)
    ]
    out.extend(a[index] + angle_delta(b[index], a[index]) * alpha for index in range(3, 6))
    return normalize_pose(out)


def _interpolate_joint(a: list[float], b: list[float], alpha: float) -> list[float]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return [a[index] + (b[index] - a[index]) * alpha for index in range(6)]


def _scaled_offset(offset: Iterable[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in offset]


@dataclass
class RM65ImpedanceConfig:
    tool_name: str
    control_hz: float = 40.0
    action_hz: float = 20.0
    force_source: str = "work"
    axes: str = "all"
    force_sign: float = 1.0
    angular_force_sign: float = -1.0
    angular_axis_signs: list[float] = field(default_factory=lambda: [1.0, 1.0, -1.0])
    lowpass_alpha: float = 0.95
    force_deadband_n: float = 0.30
    torque_deadband_nm: float = 0.03
    linear_gain_m_s_per_n: float = 0.0
    angular_gain_rad_s_per_nm: float = 0.0
    linear_damping_n_s_per_m: float = 0.0
    angular_damping_nm_s_per_rad: float = 0.0
    linear_compliance_m_per_n: float = 0.004
    angular_compliance_rad_per_nm: float = 0.04
    max_linear_velocity_m_s: float = 0.4
    max_angular_velocity_rad_s: float = 0.4
    max_linear_offset_m: float = 0.05
    max_angular_offset_rad: float = 0.4
    return_rate_s: float = 0.35
    trajectory_mode: int = 0
    radio: int = 0
    hard_stop_force_n: float = 50.0
    hard_stop_torque_nm: float = 10.0
    max_command_joint_step_deg: float = 20.0
    ik_offset_backoff_min_scale: float = 1.0
    debug_arm_endpoint_tcp: bool = False
    debug_tool_frame_name: str = ""
    tare_sec: float = 1.0
    state_read_every: int = 1
    tcp_pose_flange_m_rad: list[float] | None = None
    tcp_confirmed: bool = False
    payload_mass_kg: float | None = None
    payload_com_tool_m: list[float] | None = None

    @classmethod
    def from_profile(cls, profile: dict[str, object], mode: str = "pose", **overrides) -> "RM65ImpedanceConfig":
        impedance = dict(((profile.get("impedance") or {}).get(mode) or {}))
        tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
        payload = profile.get("payload") if isinstance(profile.get("payload"), dict) else {}
        if "tcp_pose_flange_m_rad" not in overrides and tcp.get("pose_flange_m_rad") is not None:
            impedance["tcp_pose_flange_m_rad"] = list(tcp["pose_flange_m_rad"])
            impedance["tcp_confirmed"] = bool(tcp.get("confirmed"))
        if "payload_mass_kg" not in overrides and payload.get("mass_kg") is not None:
            impedance["payload_mass_kg"] = float(payload["mass_kg"])
        if "payload_com_tool_m" not in overrides and payload.get("com_tool_m") is not None:
            impedance["payload_com_tool_m"] = list(payload["com_tool_m"])
        impedance.update(overrides)
        tool_name = str(impedance.pop("tool_name", "") or profile.get("tool_name") or "")
        impedance.pop("mode", None)
        return cls(tool_name=tool_name, **impedance)

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name must not be empty")
        if not 0.0 < self.control_hz <= 40.0:
            raise ValueError("control_hz must be in (0, 40]")
        if self.action_hz <= 0.0:
            raise ValueError("action_hz must be positive")
        if self.action_hz > self.control_hz:
            raise ValueError("action_hz must not exceed control_hz")
        if self.force_source not in ("zero", "work", "tool"):
            raise ValueError("force_source must be zero, work, or tool")
        if self.state_read_every < 1:
            raise ValueError("state_read_every must be >= 1")
        if not 0.0 <= self.ik_offset_backoff_min_scale <= 1.0:
            raise ValueError("ik_offset_backoff_min_scale must be in [0, 1]")
        if self.tcp_pose_flange_m_rad is not None:
            self.tcp_pose_flange_m_rad = finite_list(
                self.tcp_pose_flange_m_rad,
                expected_len=6,
                label="tcp_pose_flange_m_rad",
            )
        if self.payload_com_tool_m is not None:
            self.payload_com_tool_m = finite_list(
                self.payload_com_tool_m,
                expected_len=3,
                label="payload_com_tool_m",
            )
        if self.linear_damping_n_s_per_m > 0.0:
            self.linear_gain_m_s_per_n = 1.0 / self.linear_damping_n_s_per_m
        if self.angular_damping_nm_s_per_rad > 0.0:
            self.angular_gain_rad_s_per_nm = 1.0 / self.angular_damping_nm_s_per_rad

    def make_admittance(self) -> PoseAdmittance:
        return PoseAdmittance(
            enabled_axes=parse_axes(self.axes, allow_rotation=True),
            force_sign=self.force_sign,
            angular_force_sign=self.angular_force_sign,
            angular_axis_signs=self.angular_axis_signs,
            linear_gain_m_s_per_n=self.linear_gain_m_s_per_n,
            angular_gain_rad_s_per_nm=self.angular_gain_rad_s_per_nm,
            linear_compliance_m_per_n=self.linear_compliance_m_per_n,
            angular_compliance_rad_per_nm=self.angular_compliance_rad_per_nm,
            max_linear_velocity_m_s=self.max_linear_velocity_m_s,
            max_angular_velocity_rad_s=self.max_angular_velocity_rad_s,
            max_linear_offset_m=self.max_linear_offset_m,
            max_angular_offset_rad=self.max_angular_offset_rad,
            return_rate_s=self.return_rate_s,
        )

    def saturation_thresholds(self) -> dict[str, float | None]:
        thresholds: dict[str, float | None] = {
            "linear_deadbanded_force_n_for_max_offset": None,
            "linear_filtered_force_n_for_max_offset": None,
            "angular_deadbanded_torque_nm_for_max_offset": None,
            "angular_filtered_torque_nm_for_max_offset": None,
        }
        if self.linear_compliance_m_per_n > 0.0:
            deadbanded_force = self.max_linear_offset_m / self.linear_compliance_m_per_n
            thresholds["linear_deadbanded_force_n_for_max_offset"] = deadbanded_force
            thresholds["linear_filtered_force_n_for_max_offset"] = (
                deadbanded_force + self.force_deadband_n
            )
        if self.angular_compliance_rad_per_nm > 0.0:
            deadbanded_torque = self.max_angular_offset_rad / self.angular_compliance_rad_per_nm
            thresholds["angular_deadbanded_torque_nm_for_max_offset"] = deadbanded_torque
            thresholds["angular_filtered_torque_nm_for_max_offset"] = (
                deadbanded_torque + self.torque_deadband_nm
            )
        return thresholds


class _ReferenceInterpolator:
    def __init__(self, initial: list[float], *, kind: str, action_hz: float):
        self.kind = kind
        self.current = list(initial)
        self.start = list(initial)
        self.target = list(initial)
        self.elapsed = 0.0
        self.duration = 0.0
        self.default_duration = 1.0 / action_hz

    def set_target(self, target: Iterable[float], duration: float | None = None) -> None:
        target_values = finite_list(target, expected_len=6, label="target")
        self.start = list(self.current)
        self.target = normalize_pose(target_values) if self.kind == "pose" else target_values
        self.elapsed = 0.0
        self.duration = max(1e-6, self.default_duration if duration is None else float(duration))

    def advance(self, dt: float) -> list[float]:
        if self.duration <= 0.0:
            self.current = list(self.target)
            return list(self.current)
        self.elapsed = min(self.duration, self.elapsed + max(0.0, dt))
        alpha = self.elapsed / self.duration
        if self.kind == "pose":
            self.current = _interpolate_pose(self.start, self.target, alpha)
        else:
            self.current = _interpolate_joint(self.start, self.target, alpha)
        return list(self.current)


class RM65BaseImpedanceController:
    reference_kind = "pose"

    def __init__(
        self,
        *,
        arm: str,
        config: RM65ImpedanceConfig,
        robot: Arm | None = None,
        left_ip: str = LEFT_ARM_IP,
        right_ip: str = RIGHT_ARM_IP,
    ):
        self.side, self.ip = _selected_ip(arm, left_ip, right_ip)
        self.config = config
        self.robot = robot
        self._owns_robot = robot is None
        self._started = False
        self._state: dict[str, list[float]] | None = None
        self._tare: list[float] | None = None
        self._processor: WrenchProcessor | None = None
        self._admittance: PoseAdmittance | None = None
        self._reference: _ReferenceInterpolator | None = None
        self._previous_command_joint: list[float] | None = None
        self._ik_seed: list[float] | None = None
        self._tool_frame: dict[str, object] | None = None
        self._original_tool_name: str | None = None
        self._debug_tool_frame_active = False
        self._model_fk_pose: list[float] | None = None
        self._latest_wrench_context: dict[str, list[float]] | None = None
        self._last_step_time: float | None = None
        self._step_index = 0

    @classmethod
    def from_profile(cls, profile_name_or_path: str, *, mode: str, arm: str | None = None, **overrides):
        profile = load_profile(profile_name_or_path, arm=arm)
        selected_arm = str(arm or profile["arm"])
        config = RM65ImpedanceConfig.from_profile(profile, mode=mode, **overrides)
        return cls(arm=selected_arm, config=config)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _connect(self) -> None:
        if self.robot is None:
            self.robot = Arm(RM65, self.ip)
        if int(self.robot.Arm_Socket_State()) != 0:
            raise RuntimeError(f"{self.side}: failed to connect ip={self.ip}")

    def _check_tool(self) -> None:
        assert self.robot is not None
        code, frame = self.robot.Get_Current_Tool_Frame()
        _require_zero(code, "Get_Current_Tool_Frame")
        self._tool_frame = dict(frame) if isinstance(frame, dict) else {"raw": frame}
        current = _current_tool_name((code, frame))
        self._original_tool_name = current
        if self.config.tool_name not in current:
            raise RuntimeError(f"expected active tool containing '{self.config.tool_name}', got '{current}'")
        current_pose = _current_tool_pose((code, frame))
        if current_pose is not None and self.config.tcp_pose_flange_m_rad is not None:
            linear, angular = _pose_delta(current_pose, self.config.tcp_pose_flange_m_rad)
            if linear > 0.005 or angular > 0.05:
                print(
                    f"WARNING: active RealMan tool pose differs from profile TCP by "
                    f"{linear:.5f}m/{angular:.5f}rad. "
                    "Use rm65_tool_tcp_adjust.py --allow-controller-write to sync the controller."
                )

    def _debug_tool_frame_name(self) -> str:
        raw = self.config.debug_tool_frame_name or f"{self.config.tool_name}_tcp0"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._-")
        if not name:
            name = "tcp0"
        if len(name) > 10:
            short = name[:10]
            print(
                f"{self.side}: WARNING debug tool frame name '{name}' exceeds "
                f"RealMan's 10-character limit; using '{short}'"
            )
            name = short
        return name

    def _activate_debug_arm_endpoint_tcp(self) -> None:
        if not self.config.debug_arm_endpoint_tcp:
            return
        assert self.robot is not None
        payload = float(self.config.payload_mass_kg or 0.0)
        com = list(self.config.payload_com_tool_m or [0.0, 0.0, 0.0])
        frame_payload = _tool_payload(self._tool_frame)
        if frame_payload is not None:
            payload, com = frame_payload
        tool_name = self._debug_tool_frame_name()
        endpoint_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        code = self.robot.Update_Tool_Frame(tool_name, endpoint_pose, payload, *com)
        if int(code) != 0:
            code = self.robot.Manual_Set_Tool_Frame(tool_name, endpoint_pose, payload, *com)
            _require_zero(code, f"{self.side} Manual_Set_Tool_Frame({tool_name})")
        _require_zero(self.robot.Change_Tool_Frame(tool_name), f"{self.side} Change_Tool_Frame({tool_name})")
        code, frame = self.robot.Get_Current_Tool_Frame()
        _require_zero(code, "Get_Current_Tool_Frame")
        self._tool_frame = dict(frame) if isinstance(frame, dict) else {"raw": frame}
        self._debug_tool_frame_active = True
        print(
            f"{self.side}: DEBUG active tool frame switched to arm endpoint "
            f"name={tool_name} pose={_format_vector(endpoint_pose, precision=6)} "
            f"payload={payload:.6f}kg com={_format_vector(com, precision=6)}"
        )

    def _restore_debug_tool_frame(self) -> None:
        if not self._debug_tool_frame_active or self.robot is None:
            return
        self._debug_tool_frame_active = False
        original = self._original_tool_name
        if not original:
            return
        code = self.robot.Change_Tool_Frame(original)
        if int(code) != 0:
            print(f"{self.side}: WARNING failed to restore original tool frame '{original}' -> {code}")
            return
        print(f"{self.side}: restored original tool frame {original}")

    def _forward_pose(self, joint_deg: Iterable[float]) -> list[float]:
        assert self.robot is not None
        pose = self.robot.Algo_Forward_Kinematics(list(joint_deg), flag=1)
        return normalize_pose(finite_list(pose[:6], expected_len=6, label="fk_pose"))

    def _post_start_preflight(self) -> None:
        """Optional subclass preflight after state/reference initialization."""

    def _sample_tare(self) -> list[float]:
        assert self.robot is not None
        samples = max(1, int(round(self.config.tare_sec * self.config.control_hz)))
        interval = 1.0 / self.config.control_hz
        rows = []
        for index in range(samples):
            rows.append(_read_force(self.robot, self.side)[self.config.force_source])
            if index + 1 < samples:
                time.sleep(interval)
        return mean_wrench(rows)

    def _initial_reference(self, state: dict[str, list[float]]) -> list[float]:
        raise NotImplementedError

    def _command_from_reference(self, reference: list[float], offset: list[float]) -> tuple[list[float], list[float]]:
        raise NotImplementedError

    def _singularity_summary(self, joint_deg: Iterable[float]) -> str | None:
        if self.robot is None:
            return None
        try:
            result = self.robot.Algo_Kin_Singularity_Analyse(list(joint_deg))
        except Exception as exc:  # noqa: BLE001 - diagnostic path must not mask the IK failure.
            return f"unavailable ({exc})"
        if not isinstance(result, tuple) or len(result) < 2:
            return str(result)
        status, distance = result[:2]
        labels = {
            0: "normal",
            -1: "shoulder_singularity",
            -2: "elbow_singularity",
            -3: "wrist_singularity",
        }
        label = labels.get(int(status), f"unknown_status_{int(status)}")
        try:
            return f"{label} wrist_center_distance_m={float(distance):.6f}"
        except (TypeError, ValueError):
            return f"{label} wrist_center_distance_m={distance}"

    def _ik_backoff_scales(self) -> list[float]:
        min_scale = float(self.config.ik_offset_backoff_min_scale)
        if min_scale >= 1.0:
            return []
        candidates = [0.75, 0.5, 0.25, 0.1, min_scale]
        out: list[float] = []
        for scale in candidates:
            bounded = max(min_scale, min(1.0, float(scale)))
            if bounded < 1.0 and all(abs(bounded - previous) > 1e-9 for previous in out):
                out.append(bounded)
        return out

    def _solve_ik_with_current_retry(
        self,
        *,
        seed_joint: list[float],
        command_pose: list[float],
    ) -> tuple[int, list[float], list[float]]:
        assert self.robot is not None
        code, command_joint = self.robot.Algo_Inverse_Kinematics(seed_joint, command_pose, flag=1)
        used_seed = list(seed_joint)
        if int(code) != 0 and self._state is not None:
            current_joint = finite_list(self._state["joint_deg"], expected_len=6, label="current_joint_deg")
            if _joint_delta_deg(used_seed, current_joint) > 1e-6:
                retry_code, retry_joint = self.robot.Algo_Inverse_Kinematics(current_joint, command_pose, flag=1)
                if int(retry_code) == 0:
                    used_seed = current_joint
                    code = retry_code
                    command_joint = retry_joint
        return int(code), list(command_joint), used_seed

    def _command_from_pose_with_ik_backoff(
        self,
        *,
        reference_pose: list[float],
        offset: list[float],
    ) -> tuple[list[float], list[float], list[float], float, list[float], int]:
        assert self._ik_seed is not None
        seed_joint = list(self._ik_seed)
        command_pose = normalize_pose(apply_pose_offset(reference_pose, offset))
        code, command_joint, used_seed = self._solve_ik_with_current_retry(
            seed_joint=seed_joint,
            command_pose=command_pose,
        )
        if int(code) == 0:
            return command_pose, command_joint, used_seed, 1.0, list(offset), int(code)

        last_code = int(code)
        last_seed = list(used_seed)
        if norm(offset) > 1e-12:
            for scale in self._ik_backoff_scales():
                scaled = _scaled_offset(offset, scale)
                scaled_pose = normalize_pose(apply_pose_offset(reference_pose, scaled))
                retry_code, retry_joint, retry_seed = self._solve_ik_with_current_retry(
                    seed_joint=seed_joint,
                    command_pose=scaled_pose,
                )
                last_code = int(retry_code)
                last_seed = list(retry_seed)
                if int(retry_code) == 0:
                    if self._admittance is not None:
                        self._admittance.offset = list(scaled)
                    if self._latest_wrench_context is not None:
                        self._latest_wrench_context["ik_backoff_scale"] = [float(scale)]
                        self._latest_wrench_context["applied_offset"] = list(scaled)
                    return scaled_pose, retry_joint, retry_seed, float(scale), scaled, int(retry_code)

        return command_pose, command_joint, last_seed, 1.0, list(offset), last_code

    def _ik_failure_message(
        self,
        *,
        code: object,
        seed_joint: Iterable[float],
        target_pose: Iterable[float],
        reference: Iterable[float] | None = None,
        offset: Iterable[float] | None = None,
        extra_hint: str | None = None,
    ) -> str:
        lines = [
            f"{self.side} Algo_Inverse_Kinematics failed with return code {int(code)}",
            f"step={int(self._step_index + 1)} tool={self.config.tool_name}",
            f"seed_joint_deg={_format_vector(seed_joint, precision=3)}",
            f"target_pose={_format_vector(target_pose, precision=6)}",
        ]
        if reference is not None:
            lines.append(f"reference={_format_vector(reference, precision=6)}")
        if offset is not None:
            offset_values = list(offset)
            lines.append(f"offset={_format_vector(offset_values, precision=6)}")
            lines.append(
                "offset_norms="
                f"linear:{norm(offset_values[:3]):.6f}m angular:{norm(offset_values[3:]):.6f}rad"
            )
        if self._state is not None:
            lines.append(f"current_joint_deg={_format_vector(self._state['joint_deg'], precision=3)}")
            lines.append(f"current_pose={_format_vector(self._state['pose'], precision=6)}")
        seed_singularity = self._singularity_summary(seed_joint)
        if seed_singularity is not None:
            lines.append(f"seed_singularity={seed_singularity}")
        if self._state is not None:
            current_singularity = self._singularity_summary(self._state["joint_deg"])
            if current_singularity is not None:
                lines.append(f"current_singularity={current_singularity}")
        if self._model_fk_pose is not None and self._state is not None:
            linear, angular = _pose_delta(self._model_fk_pose, self._state["pose"])
            lines.append(f"model_fk_pose={_format_vector(self._model_fk_pose, precision=6)}")
            lines.append(
                f"state_vs_model_fk_delta=linear:{linear:.6f}m angular:{angular:.6f}rad"
            )
        if self._tool_frame is not None:
            tool_name = str(
                self._tool_frame.get("name")
                or self._tool_frame.get("frame_name")
                or self._tool_frame.get("tool_name")
                or ""
            ).strip("\x00")
            if tool_name:
                lines.append(f"active_tool_frame={tool_name}")
            pose = self._tool_frame.get("pose")
            if pose is not None:
                try:
                    lines.append(
                        "active_tool_pose="
                        + _format_vector(finite_list(pose, expected_len=6, label="tool_pose"), precision=6)
                    )
                except ValueError:
                    lines.append(f"active_tool_pose_raw={pose}")
        if self.config.tcp_pose_flange_m_rad is not None:
            lines.append(
                "profile_tcp_pose_flange_m_rad="
                + _format_vector(self.config.tcp_pose_flange_m_rad, precision=6)
            )
        if self._latest_wrench_context is not None:
            tare_corrected = self._latest_wrench_context.get("tare_corrected")
            deadbanded = self._latest_wrench_context.get("deadbanded")
            target_offset = self._latest_wrench_context.get("target_offset")
            applied_offset = self._latest_wrench_context.get("applied_offset")
            ik_backoff_scale = self._latest_wrench_context.get("ik_backoff_scale")
            if tare_corrected is not None:
                lines.append(
                    "tare_corrected_wrench="
                    + _format_vector(tare_corrected, precision=5)
                    + f" |F|={norm(tare_corrected[:3]):.5f}N |T|={norm(tare_corrected[3:]):.5f}Nm"
                )
            if deadbanded is not None:
                lines.append(
                    "deadbanded_wrench="
                    + _format_vector(deadbanded, precision=5)
                    + f" |F|={norm(deadbanded[:3]):.5f}N |T|={norm(deadbanded[3:]):.5f}Nm"
                )
            if target_offset is not None:
                lines.append(
                    "target_offset="
                    + _format_vector(target_offset, precision=6)
                    + (
                        f" linear:{norm(target_offset[:3]):.6f}m"
                        f" angular:{norm(target_offset[3:]):.6f}rad"
                    )
                )
            if applied_offset is not None:
                lines.append(
                    "applied_offset="
                    + _format_vector(applied_offset, precision=6)
                    + (
                        f" linear:{norm(applied_offset[:3]):.6f}m"
                        f" angular:{norm(applied_offset[3:]):.6f}rad"
                    )
                )
            if ik_backoff_scale is not None:
                lines.append(f"ik_backoff_scale={float(ik_backoff_scale[0]):.6f}")
        thresholds = self.config.saturation_thresholds()
        hints: list[str] = []
        linear_force = thresholds.get("linear_filtered_force_n_for_max_offset")
        angular_torque = thresholds.get("angular_filtered_torque_nm_for_max_offset")
        if (
            self._latest_wrench_context is not None
            and self.config.max_linear_offset_m > 0.0
            and linear_force is not None
        ):
            target_offset = self._latest_wrench_context.get("target_offset")
            if target_offset is not None and norm(target_offset[:3]) >= 0.95 * self.config.max_linear_offset_m:
                hints.append(
                    "linear target offset is saturating the configured limit; "
                    f"filtered force above about {linear_force:.3f}N saturates max_linear_offset_m"
                )
        if (
            self._latest_wrench_context is not None
            and self.config.max_angular_offset_rad > 0.0
            and angular_torque is not None
        ):
            target_offset = self._latest_wrench_context.get("target_offset")
            if target_offset is not None and norm(target_offset[3:]) >= 0.95 * self.config.max_angular_offset_rad:
                hints.append(
                    "angular target offset is saturating the configured limit; "
                    f"filtered torque above about {angular_torque:.3f}Nm saturates max_angular_offset_rad"
                )
        if self.reference_kind == "pose":
            hints.append(
                "if zero-offset current pose is inconsistent with local IK, try --mode joint "
                "or inspect tool-frame/TCP sync"
            )
        if extra_hint:
            hints.append(extra_hint)
        if self.config.ik_offset_backoff_min_scale >= 1.0:
            hints.append(
                "to probe whether a smaller Cartesian offset is IK-solvable, rerun with "
                "--ik-offset-backoff-min-scale 0.1"
            )
        for hint in hints:
            lines.append(f"hint={hint}")
        return "\n".join(lines)

    def _raise_ik_failure(
        self,
        *,
        code: object,
        seed_joint: Iterable[float],
        target_pose: Iterable[float],
        reference: Iterable[float] | None = None,
        offset: Iterable[float] | None = None,
        extra_hint: str | None = None,
    ) -> None:
        raise RuntimeError(
            self._ik_failure_message(
                code=code,
                seed_joint=seed_joint,
                target_pose=target_pose,
                reference=reference,
                offset=offset,
                extra_hint=extra_hint,
            )
        )

    def start(self) -> None:
        if self._started:
            return
        self._connect()
        assert self.robot is not None
        self._check_tool()
        self._activate_debug_arm_endpoint_tcp()
        self._state = _read_state(self.robot, self.side)
        self._tare = self._sample_tare()
        self._processor = WrenchProcessor(
            tare=self._tare,
            lowpass_alpha=self.config.lowpass_alpha,
            force_deadband_n=self.config.force_deadband_n,
            torque_deadband_nm=self.config.torque_deadband_nm,
        )
        self._admittance = self.config.make_admittance()
        initial_reference = self._initial_reference(self._state)
        self._reference = _ReferenceInterpolator(
            initial_reference,
            kind=self.reference_kind,
            action_hz=self.config.action_hz,
        )
        self._previous_command_joint = list(self._state["joint_deg"])
        self._ik_seed = list(self._state["joint_deg"])
        self._model_fk_pose = self._forward_pose(self._state["joint_deg"])
        self._latest_wrench_context = None
        self._post_start_preflight()
        self._last_step_time = time.perf_counter()
        self._started = True

    def set_target(self, target: Iterable[float], *, duration_sec: float | None = None) -> None:
        self.start()
        assert self._reference is not None
        self._reference.set_target(target, duration_sec)

    def step(self, target: Iterable[float] | None = None) -> dict[str, object]:
        self.start()
        assert self.robot is not None
        assert self._state is not None
        assert self._processor is not None
        assert self._admittance is not None
        assert self._reference is not None
        assert self._previous_command_joint is not None
        now = time.perf_counter()
        dt = now - self._last_step_time if self._last_step_time is not None else 1.0 / self.config.control_hz
        self._last_step_time = now
        if target is not None:
            self._reference.set_target(target)
        reference = self._reference.advance(dt)

        readings = _read_force(self.robot, self.side)
        processed = self._processor.process(readings[self.config.force_source])
        update = self._admittance.update(processed["deadbanded"], dt)
        self._latest_wrench_context = {
            "tare_corrected": list(processed["tare_corrected"]),
            "deadbanded": list(processed["deadbanded"]),
            "target_offset": list(update["target_offset"]),
            "offset": list(update["offset"]),
        }
        command_pose, command_joint = self._command_from_reference(reference, update["offset"])

        command_step = _joint_delta_deg(self._previous_command_joint, command_joint)
        if command_step > self.config.max_command_joint_step_deg:
            raise RuntimeError(f"{self.side}: command joint step {command_step:.4f}deg > max_command_joint_step_deg")
        force_norm = norm(processed["tare_corrected"][:3])
        torque_norm = norm(processed["tare_corrected"][3:])
        if force_norm > self.config.hard_stop_force_n:
            raise RuntimeError(f"{self.side}: force {force_norm:.4f}N > hard_stop_force_n")
        if torque_norm > self.config.hard_stop_torque_nm:
            raise RuntimeError(f"{self.side}: torque {torque_norm:.4f}Nm > hard_stop_torque_nm")

        self._send(command_pose, command_joint)
        self._previous_command_joint = list(command_joint)
        self._step_index += 1
        if self._step_index % self.config.state_read_every == 0:
            self._state = _read_state(self.robot, self.side)
        pose_delta = _pose_delta(command_pose, self._state["pose"]) if self._state else (math.nan, math.nan)
        return {
            "step": self._step_index,
            "dt_sec": dt,
            "reference": reference,
            "command_pose": command_pose,
            "command_joint_deg": command_joint,
            "offset": update["offset"],
            "target_offset": update["target_offset"],
            "active_wrench": update["active_wrench"],
            "tare_corrected_wrench": processed["tare_corrected"],
            "force_norm_n": force_norm,
            "torque_norm_nm": torque_norm,
            "command_joint_step_deg": command_step,
            "pose_tracking_linear_m": pose_delta[0],
            "pose_tracking_angular_rad": pose_delta[1],
        }

    def _print_progress(self, record: dict[str, object]) -> None:
        active = finite_list(record["active_wrench"], expected_len=6, label="active_wrench")
        target_offset = finite_list(record["target_offset"], expected_len=6, label="target_offset")
        offset = finite_list(record["offset"], expected_len=6, label="offset")
        print(
            f"step={int(record['step'])} "
            f"|F|={float(record['force_norm_n']):.4f}N "
            f"|T|={float(record['torque_norm_nm']):.4f}Nm "
            f"active_wrench={_format_vector(active, precision=4)} "
            f"target_offset={_format_vector(target_offset, precision=5)} "
            f"offset={_format_vector(offset, precision=5)} "
            f"offset_norms=linear:{norm(offset[:3]):.5f}m angular:{norm(offset[3:]):.5f}rad "
            f"tracking=linear:{float(record['pose_tracking_linear_m']):.5f}m "
            f"angular:{float(record['pose_tracking_angular_rad']):.5f}rad"
        )

    def run(
        self,
        duration_sec: float,
        action_source: ActionSource | None = None,
        *,
        print_every: int = 0,
    ) -> list[dict[str, object]]:
        self.start()
        period = 1.0 / self.config.control_hz
        loops = max(1, int(round(duration_sec * self.config.control_hz)))
        progress_every = max(0, int(print_every))
        records = []
        next_deadline = time.perf_counter()
        last_record = None
        try:
            for index in range(loops):
                elapsed = index * period
                target = action_source(index, elapsed, last_record) if action_source else None
                last_record = self.step(target)
                records.append(last_record)
                if progress_every and (index % progress_every == 0 or index + 1 == loops):
                    self._print_progress(last_record)
                next_deadline += period
                sleep_sec = next_deadline - time.perf_counter()
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        finally:
            self.stop_motion()
        return records

    def _send(self, command_pose: list[float], command_joint: list[float]) -> None:
        raise NotImplementedError

    def stop_motion(self) -> None:
        if self.robot is not None:
            code = self.robot.Arm_Slow_Stop()
            if int(code) != 0:
                print(f"{self.side}: Arm_Slow_Stop -> {code}")

    def close(self) -> None:
        if self.robot is not None:
            self._restore_debug_tool_frame()
            if self._owns_robot:
                self.robot.Arm_Socket_Close()
        self._started = False


class RM65PoseImpedanceController(RM65BaseImpedanceController):
    reference_kind = "pose"

    def _initial_reference(self, state: dict[str, list[float]]) -> list[float]:
        return normalize_pose(state["pose"])

    def _post_start_preflight(self) -> None:
        assert self.robot is not None
        assert self._state is not None
        assert self._ik_seed is not None
        controller_pose = normalize_pose(self._state["pose"])
        code, _joint = self.robot.Algo_Inverse_Kinematics(self._ik_seed, controller_pose, flag=1)
        if int(code) != 0:
            self._raise_ik_failure(
                code=code,
                seed_joint=self._ik_seed,
                target_pose=controller_pose,
                reference=controller_pose,
                offset=[0.0] * 6,
                extra_hint=(
                    "zero-offset preflight failed before the impedance loop started; "
                    "the controller-reported pose is not locally IK-consistent for the current seed"
                ),
            )

    def _command_from_reference(self, reference: list[float], offset: list[float]) -> tuple[list[float], list[float]]:
        assert self._ik_seed is not None
        command_pose, command_joint, seed_joint, backoff_scale, applied_offset, code = (
            self._command_from_pose_with_ik_backoff(reference_pose=reference, offset=offset)
        )
        if int(code) != 0:
            self._raise_ik_failure(
                code=code,
                seed_joint=seed_joint,
                target_pose=command_pose,
                reference=reference,
                offset=offset,
            )
        command_joint = finite_list(command_joint[:6], expected_len=6, label="ik_joint_deg")
        self._ik_seed = list(command_joint)
        return command_pose, command_joint

    def _send(self, command_pose: list[float], command_joint: list[float]) -> None:
        del command_joint
        assert self.robot is not None
        _require_zero(
            self.robot.Movep_CANFD(
                command_pose,
                follow=False,
                trajectory_mode=self.config.trajectory_mode,
                radio=self.config.radio,
            ),
            f"{self.side} Movep_CANFD",
        )


class RM65JointImpedanceController(RM65BaseImpedanceController):
    reference_kind = "joint"

    def _initial_reference(self, state: dict[str, list[float]]) -> list[float]:
        return list(state["joint_deg"])

    def _command_from_reference(self, reference: list[float], offset: list[float]) -> tuple[list[float], list[float]]:
        assert self._ik_seed is not None
        ref_joint = finite_list(reference, expected_len=6, label="reference_joint")
        ref_pose = self._forward_pose(ref_joint)
        command_pose, command_joint, seed_joint, backoff_scale, applied_offset, code = (
            self._command_from_pose_with_ik_backoff(reference_pose=ref_pose, offset=offset)
        )
        if int(code) != 0:
            self._raise_ik_failure(
                code=code,
                seed_joint=seed_joint,
                target_pose=command_pose,
                reference=ref_pose,
                offset=offset,
            )
        command_joint = finite_list(command_joint[:6], expected_len=6, label="ik_joint_deg")
        self._ik_seed = list(command_joint)
        return command_pose, command_joint

    def _send(self, command_pose: list[float], command_joint: list[float]) -> None:
        del command_pose
        assert self.robot is not None
        _require_zero(
            self.robot.Movej_CANFD(
                command_joint,
                follow=False,
                trajectory_mode=self.config.trajectory_mode,
                radio=self.config.radio,
            ),
            f"{self.side} Movej_CANFD",
        )
