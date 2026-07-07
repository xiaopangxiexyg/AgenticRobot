from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

from hardware.rm65.calibration_profiles import HAND_BASE_TCP_POSE_FLANGE_M_RAD, load_profile
from hardware.rm65.force_compensation import (
    PoseAdmittance,
    WrenchProcessor,
    admittance_wrench_for_arm,
    angle_delta,
    apply_pose_offset,
    clamp_norm,
    finite_list,
    mean_wrench,
    normalize_pose,
    norm,
    parse_axes,
)
from hardware.rm65.ik_provider import RM65SDKIKProvider
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.safety_diagnostics import check_rm65_joint_safety
from robot_motion.composite_provider import CompositeKinematicsProvider
from robot_motion.ik_resolver import IKSafetyResolver
from robot_motion.joint_limits import (
    JointRateLimiter,
    JointVelocityLimitProfile,
    load_joint_velocity_limit_profiles,
)
from robot_motion.pinocchio_provider import PinocchioIKProvider
from robot_motion.pose import matrix_to_pose6, pose6_to_matrix
from robot_motion.profiles import load_ik_safety_profiles
from robot_motion.seed_expansion import load_seed_expansion_profiles
from robot_motion.types import IKResolveRequest


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
    linear_vector, angular_rotvec = _pose_delta_vectors(source=a, target=b)
    linear = norm(linear_vector)
    angular = norm(angular_rotvec)
    return linear, angular


def _pose_delta_vectors(*, source: Iterable[float], target: Iterable[float]) -> tuple[list[float], list[float]]:
    source_pose = normalize_pose(finite_list(source, expected_len=6, label="source_pose"))
    target_pose = normalize_pose(finite_list(target, expected_len=6, label="target_pose"))
    source_matrix = pose6_to_matrix(source_pose)
    target_matrix = pose6_to_matrix(target_pose)
    linear = target_matrix[:3, 3] - source_matrix[:3, 3]
    angular = (
        R.from_matrix(target_matrix[:3, :3])
        * R.from_matrix(source_matrix[:3, :3]).inv()
    ).as_rotvec()
    return [float(value) for value in linear.tolist()], [float(value) for value in angular.tolist()]


def _pose_offset_from_reference(*, reference_pose: Iterable[float], command_pose: Iterable[float]) -> list[float]:
    linear, angular = _pose_delta_vectors(source=reference_pose, target=command_pose)
    return linear + angular


def _scaled_vector(values: Iterable[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in values]


def _vector_error_norm(a: Iterable[float], b: Iterable[float]) -> float:
    left = list(a)
    right = list(b)
    return norm([float(left[index]) - float(right[index]) for index in range(min(len(left), len(right)))])


def _fraction_or_none(numerator: float, denominator: float, *, eps: float = 1e-9) -> float | None:
    if abs(float(denominator)) <= eps:
        return None
    return float(numerator) / float(denominator)


def _directional_speed(velocity: Iterable[float] | None, error: Iterable[float] | None) -> float | None:
    if velocity is None or error is None:
        return None
    velocity_arr = np.asarray(list(velocity), dtype=float).reshape(-1)
    error_arr = np.asarray(list(error), dtype=float).reshape(-1)
    if velocity_arr.shape != error_arr.shape or not np.all(np.isfinite(velocity_arr)) or not np.all(np.isfinite(error_arr)):
        return None
    error_norm = float(np.linalg.norm(error_arr))
    if error_norm <= 1e-12:
        return 0.0
    return float(np.dot(velocity_arr, error_arr) / error_norm)


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _joint_deltas_deg(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [abs(float(x) - float(y)) for x, y in zip(a, b)]


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


def _pose6_to_quat_pose(pose: Iterable[float]) -> list[float]:
    values = finite_list(pose, expected_len=6, label="pose")
    quat_xyzw = R.from_euler("xyz", values[3:6]).as_quat()
    return values[:3] + [
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def _invert_transform(matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def _joint_deg_to_rad(values: Iterable[float], *, label: str) -> np.ndarray:
    return np.deg2rad(np.asarray(finite_list(values, expected_len=6, label=label), dtype=float))


def _append_unique_seed(
    seeds: list[np.ndarray],
    labels: list[str],
    qpos: np.ndarray | None,
    label: str,
) -> None:
    if qpos is None:
        return
    candidate = np.asarray(qpos, dtype=float).reshape(-1)
    if candidate.shape != (6,) or not np.all(np.isfinite(candidate)):
        return
    if any(np.allclose(candidate, existing, atol=1e-9) for existing in seeds):
        return
    seeds.append(candidate.copy())
    labels.append(str(label))


def _coerce_float_list(values: object, *, expected_len: int) -> list[float] | None:
    if values is None:
        return None
    try:
        out = [float(value) for value in list(values)[:expected_len]]
    except (TypeError, ValueError):
        return None
    if len(out) != expected_len or not all(math.isfinite(value) for value in out):
        return None
    return out


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
    angular_axis_velocity_scales: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    lowpass_alpha: float = 0.95
    force_deadband_n: float = 0.30
    torque_deadband_nm: float = 0.03
    return_force_deadband_n: float | None = None
    return_torque_deadband_nm: float | None = None
    linear_gain_m_s_per_n: float = 0.0
    angular_gain_rad_s_per_nm: float = 0.0
    linear_damping_n_s_per_m: float = 0.0
    angular_damping_nm_s_per_rad: float = 0.0
    linear_compliance_m_per_n: float = 0.004
    angular_compliance_rad_per_nm: float = 0.04
    max_linear_velocity_m_s: float = 0.4
    max_angular_velocity_rad_s: float = 0.4
    max_linear_return_velocity_m_s: float | None = None
    max_angular_return_velocity_rad_s: float | None = None
    max_linear_offset_m: float = 0.05
    max_angular_offset_rad: float = 0.4
    return_rate_s: float = 0.35
    trajectory_mode: int = 0
    radio: int = 0
    hard_stop_force_n: float = 50.0
    hard_stop_torque_nm: float = 10.0
    max_command_joint_step_deg: float = 20.0
    max_command_joint_velocity_deg_s: float = 120.0
    joint_rate_limiter_enabled: bool = True
    joint_velocity_limits_json: str | None = None
    joint_velocity_limit_profile: str = "rm65"
    pose_command_transport: str = "joint"
    ik_offset_backoff_min_scale: float = 1.0
    ik_reference_backoff_min_scale: float = 0.1
    ik_command_step_backoff_min_scale: float = 0.02
    ik_quaternion_fallback: bool = True
    ik_branch_jump_guard_deg: float = 90.0
    ik_branch_jump_guard_joints: int = 3
    ik_resolver_enabled: bool = True
    ik_safety_profile: str = "teleop"
    ik_safety_profile_json: str | None = None
    ik_resolver_include_pinocchio: bool = True
    ik_resolver_use_all_sdk_solutions: bool = True
    ik_resolver_enable_expensive_seed_expansion: bool | None = None
    ik_resolver_allow_parallel: bool | None = None
    ik_resolver_candidate_provider_names: list[str] = field(default_factory=lambda: ["rm65_pinocchio"])
    ik_resolver_pinocchio_eef_name: str = "hand_base"
    ik_resolver_pinocchio_max_iterations: int = 16
    ik_resolver_pinocchio_max_joint_step_rad: float = 0.08
    ik_resolver_pinocchio_damping: float = 0.1
    ik_resolver_pinocchio_tolerance: float = 1e-4
    ik_resolver_local_step_provider_names: list[str] = field(default_factory=lambda: ["rm65_pinocchio"])
    ik_resolver_local_step_max_iterations: int | None = None
    ik_resolver_local_step_trigger_ratio: float | None = None
    ik_resolver_local_step_max_position_worsen_m: float | None = None
    ik_resolver_local_step_component_progress_max_position_worsen_m: float | None = None
    ik_seed_expansion_enabled: bool = True
    ik_seed_expansion_profile: str = "rm65"
    ik_seed_expansion_json: str | None = None
    direct_joint_hint_max_linear_offset_m: float = 1e-3
    direct_joint_hint_max_angular_offset_rad: float = 2e-3
    direct_joint_hint_force_guard_enabled: bool = True
    direct_joint_hint_force_guard_force_n: float = 20.0
    direct_joint_hint_force_guard_torque_nm: float = 5.0
    direct_joint_hint_force_guard_reset_admittance: bool = True
    hold_last_safe_on_ik_failure: bool = False
    debug_arm_endpoint_tcp: bool = False
    debug_timing: bool = False
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
        if self.force_deadband_n < 0.0 or self.torque_deadband_nm < 0.0:
            raise ValueError("force and torque deadbands must be non-negative")
        if self.return_force_deadband_n is not None and self.return_force_deadband_n < 0.0:
            raise ValueError("return_force_deadband_n must be non-negative")
        if self.return_torque_deadband_nm is not None and self.return_torque_deadband_nm < 0.0:
            raise ValueError("return_torque_deadband_nm must be non-negative")
        self.angular_axis_velocity_scales = finite_list(
            self.angular_axis_velocity_scales,
            expected_len=3,
            label="angular_axis_velocity_scales",
        )
        if any(value < 0.0 for value in self.angular_axis_velocity_scales):
            raise ValueError("angular_axis_velocity_scales values must be non-negative")
        if self.max_linear_return_velocity_m_s is not None and self.max_linear_return_velocity_m_s < 0.0:
            raise ValueError("max_linear_return_velocity_m_s must be non-negative")
        if self.max_angular_return_velocity_rad_s is not None and self.max_angular_return_velocity_rad_s < 0.0:
            raise ValueError("max_angular_return_velocity_rad_s must be non-negative")
        if self.state_read_every < 1:
            raise ValueError("state_read_every must be >= 1")
        if self.max_command_joint_step_deg <= 0.0:
            raise ValueError("max_command_joint_step_deg must be positive")
        if self.max_command_joint_velocity_deg_s <= 0.0:
            raise ValueError("max_command_joint_velocity_deg_s must be positive")
        if self.pose_command_transport not in ("joint", "pose"):
            raise ValueError("pose_command_transport must be 'joint' or 'pose'")
        if self.joint_rate_limiter_enabled and self.pose_command_transport != "joint":
            raise ValueError("joint_rate_limiter_enabled requires pose_command_transport='joint'")
        if self.joint_rate_limiter_enabled and not str(self.joint_velocity_limit_profile).strip():
            raise ValueError("joint_velocity_limit_profile must not be empty when joint_rate_limiter_enabled")
        if not 0.0 <= self.ik_offset_backoff_min_scale <= 1.0:
            raise ValueError("ik_offset_backoff_min_scale must be in [0, 1]")
        if not 0.0 <= self.ik_reference_backoff_min_scale <= 1.0:
            raise ValueError("ik_reference_backoff_min_scale must be in [0, 1]")
        if not 0.0 <= self.ik_command_step_backoff_min_scale <= 1.0:
            raise ValueError("ik_command_step_backoff_min_scale must be in [0, 1]")
        if self.ik_branch_jump_guard_deg <= 0.0:
            raise ValueError("ik_branch_jump_guard_deg must be positive")
        if not 1 <= int(self.ik_branch_jump_guard_joints) <= 6:
            raise ValueError("ik_branch_jump_guard_joints must be in [1, 6]")
        if not str(self.ik_safety_profile).strip():
            raise ValueError("ik_safety_profile must not be empty")
        if not str(self.ik_resolver_pinocchio_eef_name).strip():
            raise ValueError("ik_resolver_pinocchio_eef_name must not be empty")
        if int(self.ik_resolver_pinocchio_max_iterations) < 1:
            raise ValueError("ik_resolver_pinocchio_max_iterations must be >= 1")
        if self.ik_resolver_pinocchio_max_joint_step_rad <= 0.0:
            raise ValueError("ik_resolver_pinocchio_max_joint_step_rad must be positive")
        if self.ik_resolver_pinocchio_damping <= 0.0:
            raise ValueError("ik_resolver_pinocchio_damping must be positive")
        if self.ik_resolver_pinocchio_tolerance <= 0.0:
            raise ValueError("ik_resolver_pinocchio_tolerance must be positive")
        self.ik_resolver_local_step_provider_names = [
            str(name).strip()
            for name in list(self.ik_resolver_local_step_provider_names or [])
            if str(name).strip()
        ]
        self.ik_resolver_candidate_provider_names = [
            str(name).strip()
            for name in list(self.ik_resolver_candidate_provider_names or [])
            if str(name).strip()
        ]
        if self.ik_resolver_local_step_max_iterations is not None and int(self.ik_resolver_local_step_max_iterations) < 1:
            raise ValueError("ik_resolver_local_step_max_iterations must be >= 1")
        if self.ik_resolver_local_step_trigger_ratio is not None and float(self.ik_resolver_local_step_trigger_ratio) < 1.0:
            raise ValueError("ik_resolver_local_step_trigger_ratio must be >= 1")
        if (
            self.ik_resolver_local_step_max_position_worsen_m is not None
            and float(self.ik_resolver_local_step_max_position_worsen_m) < 0.0
        ):
            raise ValueError("ik_resolver_local_step_max_position_worsen_m must be non-negative")
        if (
            self.ik_resolver_local_step_component_progress_max_position_worsen_m is not None
            and float(self.ik_resolver_local_step_component_progress_max_position_worsen_m) < 0.0
        ):
            raise ValueError(
                "ik_resolver_local_step_component_progress_max_position_worsen_m must be non-negative"
            )
        if self.ik_seed_expansion_enabled and not str(self.ik_seed_expansion_profile).strip():
            raise ValueError("ik_seed_expansion_profile must not be empty when ik_seed_expansion_enabled")
        if self.direct_joint_hint_max_linear_offset_m < 0.0:
            raise ValueError("direct_joint_hint_max_linear_offset_m must be non-negative")
        if self.direct_joint_hint_max_angular_offset_rad < 0.0:
            raise ValueError("direct_joint_hint_max_angular_offset_rad must be non-negative")
        if self.direct_joint_hint_force_guard_force_n < 0.0:
            raise ValueError("direct_joint_hint_force_guard_force_n must be non-negative")
        if self.direct_joint_hint_force_guard_torque_nm < 0.0:
            raise ValueError("direct_joint_hint_force_guard_torque_nm must be non-negative")
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

    @property
    def effective_return_force_deadband_n(self) -> float:
        if self.return_force_deadband_n is None:
            return float(self.force_deadband_n)
        return float(self.return_force_deadband_n)

    @property
    def effective_return_torque_deadband_nm(self) -> float:
        if self.return_torque_deadband_nm is None:
            return float(self.torque_deadband_nm)
        return float(self.return_torque_deadband_nm)

    def make_admittance(self) -> PoseAdmittance:
        return PoseAdmittance(
            enabled_axes=parse_axes(self.axes, allow_rotation=True),
            force_sign=self.force_sign,
            angular_force_sign=self.angular_force_sign,
            angular_axis_signs=self.angular_axis_signs,
            angular_axis_velocity_scales=self.angular_axis_velocity_scales,
            linear_gain_m_s_per_n=self.linear_gain_m_s_per_n,
            angular_gain_rad_s_per_nm=self.angular_gain_rad_s_per_nm,
            linear_compliance_m_per_n=self.linear_compliance_m_per_n,
            angular_compliance_rad_per_nm=self.angular_compliance_rad_per_nm,
            max_linear_velocity_m_s=self.max_linear_velocity_m_s,
            max_angular_velocity_rad_s=self.max_angular_velocity_rad_s,
            max_linear_return_velocity_m_s=self.max_linear_return_velocity_m_s,
            max_angular_return_velocity_rad_s=self.max_angular_return_velocity_rad_s,
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
        self._joint_hint_reference: _ReferenceInterpolator | None = None
        self._previous_command_joint: list[float] | None = None
        self._ik_seed: list[float] | None = None
        self._tool_frame: dict[str, object] | None = None
        self._original_tool_name: str | None = None
        self._debug_tool_frame_active = False
        self._model_fk_pose: list[float] | None = None
        self._latest_wrench_context: dict[str, object] | None = None
        self._last_step_time: float | None = None
        self._current_step_dt: float | None = None
        self._previous_command_pose: list[float] | None = None
        self._previous_measured_pose: list[float] | None = None
        self._previous_measured_joint: list[float] | None = None
        self._previous_measured_sample_time: float | None = None
        self._step_index = 0
        self._last_ik_hold_warning_time = 0.0
        self._last_ik_hold_warning_key: str | None = None
        self._ik_resolver: IKSafetyResolver | None = None
        self._ik_resolver_provider: CompositeKinematicsProvider | None = None
        self._ik_resolver_init_metadata: dict[str, object] = {}
        self._joint_rate_limiter: JointRateLimiter | None = None
        self._joint_velocity_limit_profile: JointVelocityLimitProfile | None = None
        self._last_joint_rate_limit_warning_time = 0.0
        self._joint_position_limits_deg: tuple[list[float], list[float]] | None = None
        self._joint_position_limits_checked = False
        self._joint_position_limits_error = ""

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

    def _target_tcp_to_hand_base_transform(self) -> np.ndarray | None:
        if self.config.tcp_pose_flange_m_rad is None:
            return None
        t_flange_hand_base = pose6_to_matrix(HAND_BASE_TCP_POSE_FLANGE_M_RAD)
        t_flange_tcp = pose6_to_matrix(self.config.tcp_pose_flange_m_rad)
        return _invert_transform(t_flange_tcp) @ t_flange_hand_base

    def _initialize_ik_resolver(self) -> None:
        self._ik_resolver = None
        self._ik_resolver_provider = None
        self._ik_resolver_init_metadata = {
            "enabled": bool(self.config.ik_resolver_enabled),
            "profile": str(self.config.ik_safety_profile),
            "provider_names": [],
            "provider_errors": [],
        }
        if not bool(self.config.ik_resolver_enabled):
            return
        if self.robot is None:
            self._ik_resolver_init_metadata["provider_errors"] = ["robot_unavailable"]
            return

        try:
            profiles = load_ik_safety_profiles(self.config.ik_safety_profile_json)
            profile = profiles[str(self.config.ik_safety_profile)]
        except Exception as exc:  # noqa: BLE001 - keep impedance usable with legacy IK.
            error = f"load_profile_failed:{type(exc).__name__}: {exc}"
            self._ik_resolver_init_metadata["provider_errors"] = [error]
            print(f"WARNING: {self.side}: IK safety resolver disabled: {error}")
            return

        replace_kwargs: dict[str, object] = {}
        if self.config.ik_resolver_enable_expensive_seed_expansion is not None:
            replace_kwargs["enable_expensive_seed_expansion"] = bool(
                self.config.ik_resolver_enable_expensive_seed_expansion
            )
        if self.config.ik_resolver_allow_parallel is not None:
            replace_kwargs["allow_parallel_candidate_generation"] = bool(self.config.ik_resolver_allow_parallel)
        if self.config.ik_resolver_local_step_max_iterations is not None:
            local_step_iterations = int(self.config.ik_resolver_local_step_max_iterations)
            replace_kwargs["local_step_max_iterations"] = local_step_iterations
            replace_kwargs["local_step_iteration_counts"] = (local_step_iterations,)
        if self.config.ik_resolver_local_step_trigger_ratio is not None:
            replace_kwargs["local_step_trigger_ratio"] = float(self.config.ik_resolver_local_step_trigger_ratio)
        if self.config.ik_resolver_local_step_max_position_worsen_m is not None:
            replace_kwargs["local_step_max_position_worsen_m"] = float(
                self.config.ik_resolver_local_step_max_position_worsen_m
            )
        if self.config.ik_resolver_local_step_component_progress_max_position_worsen_m is not None:
            replace_kwargs["local_step_component_progress_max_position_worsen_m"] = float(
                self.config.ik_resolver_local_step_component_progress_max_position_worsen_m
            )
        if replace_kwargs:
            profile = replace(profile, **replace_kwargs)

        errors: list[str] = []
        seed_expansion_profile = None
        seed_expansion_metadata: dict[str, object] = {
            "enabled": bool(self.config.ik_seed_expansion_enabled),
            "profile": str(self.config.ik_seed_expansion_profile),
            "profile_source": self.config.ik_seed_expansion_json or "default",
        }
        if bool(self.config.ik_seed_expansion_enabled):
            try:
                seed_profiles = load_seed_expansion_profiles(self.config.ik_seed_expansion_json)
                seed_expansion_profile = seed_profiles[str(self.config.ik_seed_expansion_profile)]
                seed_expansion_metadata.update(
                    {
                        "loaded": True,
                        "max_extra_seeds": int(seed_expansion_profile.max_extra_seeds),
                        "perturbation_count": len(seed_expansion_profile.perturbations),
                        "source_label_allowlist": list(seed_expansion_profile.source_label_allowlist),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - optional seed expansion must not break teleop.
                seed_expansion_metadata.update(
                    {
                        "loaded": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                errors.append(
                    f"seed_expansion_failed:{type(exc).__name__}: {exc}"
                )

        providers: list[object] = [
            RM65SDKIKProvider(
                self.robot,
                use_quaternion_fallback=bool(self.config.ik_quaternion_fallback),
                use_all_solutions=bool(self.config.ik_resolver_use_all_sdk_solutions),
                include_wrist_flip_candidates=True,
            )
        ]
        names = ["rm65_sdk"]

        if bool(self.config.ik_resolver_include_pinocchio):
            target_to_eef = self._target_tcp_to_hand_base_transform()
            if target_to_eef is None:
                errors.append("rm65_pinocchio_skipped:tcp_pose_flange_m_rad_unavailable")
            else:
                try:
                    from rm65_inspire_tac.ik import SingleArmIKSolver

                    providers.append(
                        PinocchioIKProvider(
                            solver=SingleArmIKSolver(
                                eef_name=str(self.config.ik_resolver_pinocchio_eef_name),
                            ),
                            source="rm65_pinocchio",
                            target_to_eef_transform=target_to_eef,
                            qpos_size=6,
                        )
                    )
                    names.append("rm65_pinocchio")
                except Exception as exc:  # noqa: BLE001 - optional provider must not break teleop.
                    errors.append(f"rm65_pinocchio_failed:{type(exc).__name__}: {exc}")

        candidate_provider_names = [
            str(name).strip()
            for name in list(self.config.ik_resolver_candidate_provider_names or [])
            if str(name).strip()
        ]
        primary_provider_index = 0
        for provider_name in candidate_provider_names:
            if provider_name in names:
                primary_provider_index = names.index(provider_name)
                break
        self._ik_resolver_provider = CompositeKinematicsProvider(
            providers=tuple(providers),
            names=tuple(names),
            primary_index=primary_provider_index,
        )
        unknown_candidate_provider_names = [
            name for name in candidate_provider_names if name not in names
        ]
        if unknown_candidate_provider_names:
            errors.append(
                "candidate_provider_names_unavailable:"
                + ",".join(unknown_candidate_provider_names)
            )
        self._ik_resolver = IKSafetyResolver(
            profile=profile,
            provider=self._ik_resolver_provider,
            seed_expansion_profile=seed_expansion_profile,
        )
        self._ik_resolver_init_metadata = {
            "enabled": True,
            "profile": profile.name,
            "profile_source": self.config.ik_safety_profile_json or "default",
            "provider_names": list(names),
            "primary_provider_name": names[primary_provider_index],
            "candidate_provider_names": list(candidate_provider_names),
            "rm_sdk_inverse_candidates_enabled": (
                not candidate_provider_names or "rm65_sdk" in candidate_provider_names
            ),
            "provider_errors": list(errors),
            "resolver_policy": "minimize_penalized_max_joint_delta_then_norm",
            "target_tolerance_position_m": float(profile.target_tolerance.position_m),
            "target_tolerance_orientation_deg": float(profile.target_tolerance.orientation_deg),
            "relaxation_fractions": list(profile.relaxation_fractions),
            "reject_singular": bool(profile.reject_singular),
            "provider_time_budget_ms": profile.provider_time_budget_ms,
            "coarse_tracking_enabled": bool(profile.coarse_tracking_enabled),
            "coarse_tracking_start_fraction": float(profile.coarse_tracking_start_fraction),
            "coarse_tracking_max_tolerance_fraction": float(profile.coarse_tracking_max_tolerance_fraction),
            "coarse_tracking_max_position_error_m": float(profile.coarse_tracking_max_position_error_m),
            "coarse_tracking_max_orientation_error_deg": float(profile.coarse_tracking_max_orientation_error_deg),
            "coarse_tracking_min_fraction_progress": float(profile.coarse_tracking_min_fraction_progress),
            "coarse_tracking_orientation_recovery_enabled": bool(
                profile.coarse_tracking_orientation_recovery_enabled
            ),
            "coarse_tracking_orientation_recovery_min_progress_deg": float(
                profile.coarse_tracking_orientation_recovery_min_progress_deg
            ),
            "coarse_tracking_orientation_recovery_max_position_worsen_m": float(
                profile.coarse_tracking_orientation_recovery_max_position_worsen_m
            ),
            "local_step_enabled": bool(profile.local_step_enabled),
            "local_step_first_enabled": bool(profile.local_step_first_enabled),
            "local_step_first_full_solve_interval": int(profile.local_step_first_full_solve_interval),
            "local_step_provider_names": list(self.config.ik_resolver_local_step_provider_names),
            "local_step_trigger_ratio": float(profile.local_step_trigger_ratio),
            "local_step_max_iterations": int(profile.local_step_max_iterations),
            "local_step_iteration_counts": [
                int(value)
                for value in (profile.local_step_iteration_counts or (profile.local_step_max_iterations,))
            ],
            "local_step_scale_factors": [float(value) for value in profile.local_step_scale_factors],
            "local_step_dynamic_worsen_enabled": bool(profile.local_step_dynamic_worsen_enabled),
            "local_step_dynamic_worsen_error_start_m": float(profile.local_step_dynamic_worsen_error_start_m),
            "local_step_dynamic_worsen_error_start_deg": float(profile.local_step_dynamic_worsen_error_start_deg),
            "local_step_dynamic_worsen_position_fraction": float(profile.local_step_dynamic_worsen_position_fraction),
            "local_step_dynamic_worsen_orientation_fraction": float(profile.local_step_dynamic_worsen_orientation_fraction),
            "local_step_dynamic_worsen_max_position_m": float(profile.local_step_dynamic_worsen_max_position_m),
            "local_step_dynamic_worsen_max_orientation_deg": float(profile.local_step_dynamic_worsen_max_orientation_deg),
            "local_step_accept_worse_within_tolerance": bool(profile.local_step_accept_worse_within_tolerance),
            "local_step_orientation_recovery_enabled": bool(profile.local_step_orientation_recovery_enabled),
            "local_step_orientation_recovery_min_progress_deg": float(
                profile.local_step_orientation_recovery_min_progress_deg
            ),
            "local_step_orientation_recovery_max_position_worsen_m": float(
                profile.local_step_orientation_recovery_max_position_worsen_m
            ),
            "local_step_orientation_recovery_max_after_position_error_m": float(
                profile.local_step_orientation_recovery_max_after_position_error_m
            ),
            "local_step_component_progress_enabled": bool(profile.local_step_component_progress_enabled),
            "local_step_component_progress_min_position_m": float(
                profile.local_step_component_progress_min_position_m
            ),
            "local_step_component_progress_min_orientation_deg": float(
                profile.local_step_component_progress_min_orientation_deg
            ),
            "local_step_component_progress_max_score_regression": float(
                profile.local_step_component_progress_max_score_regression
            ),
            "local_step_component_progress_max_position_worsen_m": float(
                profile.local_step_component_progress_max_position_worsen_m
            ),
            "local_step_component_progress_max_orientation_worsen_deg": float(
                profile.local_step_component_progress_max_orientation_worsen_deg
            ),
            "local_step_component_progress_max_after_position_error_m": float(
                profile.local_step_component_progress_max_after_position_error_m
            ),
            "local_step_limit_escape_enabled": bool(profile.local_step_limit_escape_enabled),
            "local_step_limit_escape_activation_margin_deg": float(profile.local_step_limit_escape_activation_margin_deg),
            "local_step_limit_escape_step_fraction": float(profile.local_step_limit_escape_step_fraction),
            "local_step_limit_escape_max_candidates": int(profile.local_step_limit_escape_max_candidates),
            "local_step_limit_escape_accept_worse_within_tolerance": bool(
                profile.local_step_limit_escape_accept_worse_within_tolerance
            ),
            "local_step_limit_escape_accept_bounded_worse": bool(
                profile.local_step_limit_escape_accept_bounded_worse
            ),
            "local_step_limit_escape_max_position_worsen_m": float(
                profile.local_step_limit_escape_max_position_worsen_m
            ),
            "local_step_limit_escape_max_orientation_worsen_deg": float(
                profile.local_step_limit_escape_max_orientation_worsen_deg
            ),
            "local_step_singularity_escape_enabled": bool(profile.local_step_singularity_escape_enabled),
            "local_step_singularity_escape_step_fraction": float(profile.local_step_singularity_escape_step_fraction),
            "local_step_singularity_escape_max_candidates": int(profile.local_step_singularity_escape_max_candidates),
            "local_step_singularity_escape_max_position_worsen_m": float(
                profile.local_step_singularity_escape_max_position_worsen_m
            ),
            "local_step_singularity_escape_max_orientation_worsen_deg": float(
                profile.local_step_singularity_escape_max_orientation_worsen_deg
            ),
            "local_step_full_target_step_enabled": bool(profile.local_step_full_target_step_enabled),
            "local_step_accept_position_progress_m": float(profile.local_step_accept_position_progress_m),
            "local_step_accept_orientation_progress_deg": float(profile.local_step_accept_orientation_progress_deg),
            "local_step_max_position_worsen_m": float(profile.local_step_max_position_worsen_m),
            "local_step_max_orientation_worsen_deg": float(profile.local_step_max_orientation_worsen_deg),
            "seed_expansion": seed_expansion_metadata,
        }
        if errors:
            print(f"WARNING: {self.side}: IK resolver provider warnings: {'; '.join(errors)}")

    def _initialize_joint_rate_limiter(self) -> None:
        self._joint_rate_limiter = None
        self._joint_velocity_limit_profile = None
        if not bool(self.config.joint_rate_limiter_enabled):
            return
        profiles = load_joint_velocity_limit_profiles(self.config.joint_velocity_limits_json)
        profile_name = str(self.config.joint_velocity_limit_profile).strip()
        if profile_name not in profiles:
            raise RuntimeError(
                f"{self.side}: joint velocity limit profile {profile_name!r} not found; "
                f"available={sorted(profiles)}"
            )
        profile = profiles[profile_name]
        if profile.dof != 6:
            raise RuntimeError(
                f"{self.side}: joint velocity limit profile {profile_name!r} has dof={profile.dof}, expected 6"
            )
        self._joint_velocity_limit_profile = profile
        self._joint_rate_limiter = JointRateLimiter(profile)
        if self._previous_command_joint is not None:
            self._joint_rate_limiter.reset(
                _joint_deg_to_rad(self._previous_command_joint, label="previous_command_joint")
            )

    def _joint_position_limits(self) -> tuple[list[float], list[float]] | None:
        if self._joint_position_limits_checked:
            return self._joint_position_limits_deg
        self._joint_position_limits_checked = True
        self._joint_position_limits_error = ""
        candidates = [self.robot, getattr(self.robot, "_robot", None) if self.robot is not None else None]
        getter_pairs = (
            ("rm_algo_get_joint_min_limit", "rm_algo_get_joint_max_limit"),
            ("Algo_Get_Joint_Min_Limit", "Algo_Get_Joint_Max_Limit"),
        )
        for owner in candidates:
            if owner is None:
                continue
            for min_name, max_name in getter_pairs:
                min_getter = getattr(owner, min_name, None)
                max_getter = getattr(owner, max_name, None)
                if min_getter is None or max_getter is None:
                    continue
                try:
                    q_min = _coerce_float_list(min_getter(), expected_len=6)
                    q_max = _coerce_float_list(max_getter(), expected_len=6)
                except Exception as exc:  # noqa: BLE001 - diagnostic path.
                    self._joint_position_limits_error = f"{type(exc).__name__}: {exc}"
                    continue
                if q_min is None or q_max is None:
                    self._joint_position_limits_error = f"{min_name}/{max_name} returned invalid limits"
                    continue
                self._joint_position_limits_deg = (q_min, q_max)
                return self._joint_position_limits_deg
        if not self._joint_position_limits_error:
            self._joint_position_limits_error = "joint limit getters unavailable"
        return None

    def _joint_limit_margin_metadata(self, joint_deg: Iterable[float]) -> dict[str, object]:
        joint = finite_list(joint_deg, expected_len=6, label="joint_deg")
        metadata: dict[str, object] = {"available": False}
        limits = self._joint_position_limits()
        position_limit_code = None
        if self.robot is not None:
            checker = getattr(self.robot, "Algo_IKine_Check_Joint_Position_Limit", None)
            if checker is not None:
                try:
                    position_limit_code = int(checker(joint))
                except Exception as exc:  # noqa: BLE001 - diagnostic path.
                    metadata["position_limit_error"] = f"{type(exc).__name__}: {exc}"
        if position_limit_code is not None:
            metadata["position_limit_code"] = int(position_limit_code)
            metadata["position_limit_ok"] = int(position_limit_code) == 0
        if limits is None:
            metadata["error"] = self._joint_position_limits_error
            return metadata

        q_min, q_max = limits
        lower = [float(joint[index] - q_min[index]) for index in range(6)]
        upper = [float(q_max[index] - joint[index]) for index in range(6)]
        per_joint_min = [min(lower[index], upper[index]) for index in range(6)]
        nearest_index = int(min(range(6), key=lambda index: per_joint_min[index]))
        metadata.update(
            {
                "available": True,
                "lower_margin_deg": lower,
                "upper_margin_deg": upper,
                "min_margin_deg": float(per_joint_min[nearest_index]),
                "nearest_joint": f"J{nearest_index + 1}",
                "nearest_joint_index": nearest_index,
                "nearest_lower_margin_deg": float(lower[nearest_index]),
                "nearest_upper_margin_deg": float(upper[nearest_index]),
                "limit_min_deg": list(q_min),
                "limit_max_deg": list(q_max),
            }
        )
        return metadata

    def _singularity_metadata(self, joint_deg: Iterable[float]) -> dict[str, object]:
        if self.robot is None:
            return {"available": False, "label": "unavailable", "error": "robot_unavailable"}
        joint = finite_list(joint_deg, expected_len=6, label="joint_deg")
        labels = {
            0: "normal",
            -1: "shoulder_singularity",
            -2: "elbow_singularity",
            -3: "wrist_singularity",
        }
        try:
            result = self.robot.Algo_Kin_Singularity_Analyse(joint)
        except Exception as exc:  # noqa: BLE001 - diagnostic path must not mask control.
            return {
                "available": False,
                "label": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(result, tuple) or len(result) < 2:
            return {
                "available": False,
                "label": "unknown",
                "raw": repr(result),
            }
        try:
            status = int(result[0])
        except (TypeError, ValueError):
            status = None
        try:
            metric = float(result[1])
        except (TypeError, ValueError):
            metric = None
        return {
            "available": True,
            "status_code": status,
            "label": labels.get(status, f"unknown_status_{status}"),
            "is_singular": status not in (0, None),
            "wrist_center_distance_m": metric,
            "raw": repr(result),
        }

    def _attach_ik_timing_metadata(
        self,
        metadata: dict[str, object],
        *,
        source: str,
        elapsed_ms: float,
        provider_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        provider_metadata = dict(provider_metadata or {})
        existing_profile = metadata.get("ik_debug_profile")
        ik_debug_profile = dict(existing_profile) if isinstance(existing_profile, dict) else {}
        provider_last_elapsed_ms = provider_metadata.get("provider_elapsed_ms")
        provider_elapsed_ms = ik_debug_profile.get("provider_total_ms", provider_last_elapsed_ms)
        time_budget_ms = ik_debug_profile.get(
            "time_budget_ms",
            provider_metadata.get("provider_time_budget_ms"),
        )
        time_budget_exceeded = bool(
            ik_debug_profile.get(
                "time_budget_exceeded",
                provider_metadata.get("provider_time_budget_exceeded", False),
            )
        )
        ik_debug_profile.setdefault("frame_total_ms", float(elapsed_ms))
        ik_debug_profile.setdefault("solve_total_ms", float(elapsed_ms))
        if provider_elapsed_ms is not None:
            ik_debug_profile.setdefault("provider_total_ms", provider_elapsed_ms)
        if provider_last_elapsed_ms is not None:
            ik_debug_profile["provider_last_ms"] = provider_last_elapsed_ms
        if time_budget_ms is not None:
            ik_debug_profile.setdefault("time_budget_ms", time_budget_ms)
        ik_debug_profile["time_budget_exceeded"] = bool(time_budget_exceeded)
        metadata.update(
            {
                "ik_frame_total_ms": float(elapsed_ms),
                "ik_solve_total_ms": float(elapsed_ms),
                "ik_backend_solve_ms": float(elapsed_ms),
                "ik_resolve_elapsed_ms": float(elapsed_ms),
                "ik_provider_elapsed_ms": provider_elapsed_ms,
                "ik_provider_last_elapsed_ms": provider_last_elapsed_ms,
                "ik_provider_total_ms": ik_debug_profile.get("provider_total_ms"),
                "ik_non_provider_ms": ik_debug_profile.get("non_provider_ms"),
                "ik_fk_ms": ik_debug_profile.get("fk_ms"),
                "ik_singularity_ms": ik_debug_profile.get("singularity_ms"),
                "ik_collision_ms": ik_debug_profile.get("collision_ms"),
                "ik_candidate_eval_ms": ik_debug_profile.get("candidate_eval_ms"),
                "ik_time_budget_ms": time_budget_ms,
                "ik_time_budget_exceeded": time_budget_exceeded,
                "ik_timing_source": source,
                "ik_debug_profile": ik_debug_profile,
            }
        )
        if provider_metadata:
            metadata["provider"] = provider_metadata
        if self._latest_wrench_context is not None:
            self._latest_wrench_context["ik_timing"] = {
                "available": True,
                "source": source,
                "frame_total_ms": float(elapsed_ms),
                "solve_total_ms": float(elapsed_ms),
                "provider_elapsed_ms": provider_elapsed_ms,
                "time_budget_ms": time_budget_ms,
                "time_budget_exceeded": time_budget_exceeded,
            }
        return metadata

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

    def _command_from_reference(
        self,
        reference: list[float],
        offset: list[float],
        *,
        preferred_joint: list[float] | None = None,
    ) -> tuple[list[float], list[float]]:
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

    def _reference_backoff_scales(self) -> list[float]:
        min_scale = float(self.config.ik_reference_backoff_min_scale)
        if min_scale >= 1.0:
            return []
        candidates = [0.75, 0.5, 0.25, 0.1, min_scale]
        out: list[float] = []
        for scale in candidates:
            bounded = max(min_scale, min(1.0, float(scale)))
            if bounded < 1.0 and all(abs(bounded - previous) > 1e-9 for previous in out):
                out.append(bounded)
        return out

    def _command_step_backoff_scales(self, *, command_step: float, max_allowed: float) -> list[float]:
        min_scale = float(self.config.ik_command_step_backoff_min_scale)
        if min_scale >= 1.0:
            return []
        adaptive = 0.0
        if command_step > 1e-9 and max_allowed > 0.0:
            adaptive = 0.9 * float(max_allowed) / float(command_step)
        candidates = [0.75, 0.5, 0.25, adaptive, 0.1, 0.05, 0.02, min_scale]
        out: list[float] = []
        for scale in candidates:
            bounded = max(min_scale, min(1.0, float(scale)))
            if bounded < 1.0 and all(abs(bounded - previous) > 1e-9 for previous in out):
                out.append(bounded)
        out.sort(reverse=True)
        return out

    def _max_allowed_command_joint_step_deg(self) -> float:
        dt_for_velocity = self._observed_control_dt_sec()
        if self._joint_velocity_limit_profile is not None:
            profile_max_step = self._joint_velocity_limit_profile.max_step_deg(dt_for_velocity)
            if self.config.joint_rate_limiter_enabled:
                profile_max_abs = float(np.max(profile_max_step)) if profile_max_step.size else 0.0
                return max(float(self.config.max_command_joint_step_deg), profile_max_abs)
            dt_for_velocity = self._joint_velocity_limit_profile.effective_dt_sec(dt_for_velocity)
        else:
            dt_for_velocity = min(dt_for_velocity, 1.0)
        return min(
            float(self.config.max_command_joint_step_deg),
            float(self.config.max_command_joint_velocity_deg_s) * dt_for_velocity,
        )

    def _validate_ik_solution(
        self,
        *,
        command_joint: Iterable[float],
        reference_joint: Iterable[float],
        target_pose: Iterable[float],
        method: str,
    ) -> list[float]:
        joint = finite_list(list(command_joint)[:6], expected_len=6, label="ik_joint_deg")
        reference = finite_list(list(reference_joint)[:6], expected_len=6, label="reference_joint_deg")
        guard_count = int(self.config.ik_branch_jump_guard_joints)
        deltas = [abs(joint[index] - reference[index]) for index in range(guard_count)]
        max_delta = max(deltas) if deltas else 0.0
        if max_delta > float(self.config.ik_branch_jump_guard_deg):
            raise RuntimeError(
                f"{self.side}: IK branch jump guard tripped on J1-J{guard_count}: "
                f"max_delta={max_delta:.3f}deg > "
                f"ik_branch_jump_guard_deg={float(self.config.ik_branch_jump_guard_deg):.3f}deg "
                f"method={method} "
                f"reference_joint_deg={_format_vector(reference, precision=3)} "
                f"candidate_joint_deg={_format_vector(joint, precision=3)} "
                f"target_pose={_format_vector(target_pose, precision=6)}"
            )
        return joint

    def _solve_ik_with_current_retry(
        self,
        *,
        seed_joint: list[float],
        command_pose: list[float],
    ) -> tuple[int, list[float], list[float], str]:
        assert self.robot is not None
        code, command_joint = self.robot.Algo_Inverse_Kinematics(seed_joint, command_pose, flag=1)
        used_seed = list(seed_joint)
        method = "euler"
        if int(code) != 0 and self.config.ik_quaternion_fallback:
            quat_pose = _pose6_to_quat_pose(command_pose)
            quat_code, quat_joint = self.robot.Algo_Inverse_Kinematics(seed_joint, quat_pose, flag=0)
            if int(quat_code) == 0:
                code = quat_code
                command_joint = quat_joint
                method = "quaternion"
        if int(code) != 0 and self._state is not None:
            current_joint = finite_list(self._state["joint_deg"], expected_len=6, label="current_joint_deg")
            if _joint_delta_deg(used_seed, current_joint) > 1e-6:
                retry_code, retry_joint = self.robot.Algo_Inverse_Kinematics(current_joint, command_pose, flag=1)
                if int(retry_code) == 0:
                    used_seed = current_joint
                    code = retry_code
                    command_joint = retry_joint
                    method = "current_seed_euler"
                elif self.config.ik_quaternion_fallback:
                    quat_pose = _pose6_to_quat_pose(command_pose)
                    retry_quat_code, retry_quat_joint = self.robot.Algo_Inverse_Kinematics(
                        current_joint,
                        quat_pose,
                        flag=0,
                    )
                    if int(retry_quat_code) == 0:
                        used_seed = current_joint
                        code = retry_quat_code
                        command_joint = retry_quat_joint
                        method = "current_seed_quaternion"
        if int(code) == 0:
            reference_joint = self._previous_command_joint or used_seed
            command_joint = self._validate_ik_solution(
                command_joint=command_joint,
                reference_joint=reference_joint,
                target_pose=command_pose,
                method=method,
            )
        return int(code), list(command_joint), used_seed, method

    def _resolver_seed_qpos(
        self,
        *,
        reference_qpos: np.ndarray,
        seed_joint_override: list[float] | None,
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        seeds: list[np.ndarray] = []
        labels: list[str] = []
        if seed_joint_override is not None:
            _append_unique_seed(
                seeds,
                labels,
                _joint_deg_to_rad(seed_joint_override, label="seed_joint_override"),
                "preferred_joint",
            )
        if self._ik_seed is not None:
            _append_unique_seed(
                seeds,
                labels,
                _joint_deg_to_rad(self._ik_seed, label="ik_seed"),
                "ik_seed",
            )
        if self._state is not None:
            _append_unique_seed(
                seeds,
                labels,
                _joint_deg_to_rad(self._state["joint_deg"], label="current_joint_deg"),
                "measured_current",
            )
        if self._previous_command_joint is not None:
            _append_unique_seed(
                seeds,
                labels,
                _joint_deg_to_rad(self._previous_command_joint, label="previous_command_joint_deg"),
                "previous_command",
            )
        _append_unique_seed(seeds, labels, reference_qpos, "reference_duplicate_guard")
        return tuple(seeds), tuple(labels)

    def _command_from_pose_with_resolver(
        self,
        *,
        reference_pose: list[float],
        offset: list[float],
        seed_joint_override: list[float] | None = None,
    ) -> tuple[list[float], list[float], list[float], float, list[float], int]:
        if self._ik_resolver is None:
            raise RuntimeError("IK resolver is not initialized")
        assert self._ik_seed is not None
        command_pose = normalize_pose(apply_pose_offset(reference_pose, offset))
        reference_joint = (
            seed_joint_override
            if seed_joint_override is not None
            else (self._previous_command_joint or self._ik_seed)
        )
        reference_qpos = _joint_deg_to_rad(reference_joint, label="reference_joint")
        seed_qpos, seed_labels = self._resolver_seed_qpos(
            reference_qpos=reference_qpos,
            seed_joint_override=seed_joint_override,
        )
        previous_safe_qpos = (
            None
            if self._previous_command_joint is None
            else _joint_deg_to_rad(self._previous_command_joint, label="previous_command_joint_deg")
        )
        reference_pose_matrix = pose6_to_matrix(self._previous_command_pose or reference_pose)
        target_pose_matrix = pose6_to_matrix(command_pose)
        request = IKResolveRequest(
            target_pose=target_pose_matrix,
            reference_qpos=reference_qpos,
            reference_pose=reference_pose_matrix,
            seed_qpos=seed_qpos,
            seed_labels=seed_labels,
            previous_safe_qpos=previous_safe_qpos,
            metadata={
                "side": self.side,
                "controller": type(self).__name__,
                "tool_name": self.config.tool_name,
                "ik_resolver_init": dict(self._ik_resolver_init_metadata),
                **(
                    {
                        "provider_name_allowlist": list(
                            self.config.ik_resolver_candidate_provider_names
                        )
                    }
                    if self.config.ik_resolver_candidate_provider_names
                    else {}
                ),
                "pinocchio_max_iterations": int(self.config.ik_resolver_pinocchio_max_iterations),
                "pinocchio_max_joint_step_rad": float(self.config.ik_resolver_pinocchio_max_joint_step_rad),
                "pinocchio_damping": float(self.config.ik_resolver_pinocchio_damping),
                "pinocchio_tolerance": float(self.config.ik_resolver_pinocchio_tolerance),
            },
        )

        started_at = time.perf_counter()
        if bool(self.config.joint_rate_limiter_enabled) and self.config.pose_command_transport == "joint":
            if self._joint_velocity_limit_profile is None:
                self._initialize_joint_rate_limiter()
            rate_limit_dt_details = self._joint_rate_limit_dt_details()
            result = self._ik_resolver.resolve_step(
                request,
                joint_velocity_profile=self._joint_velocity_limit_profile,
                dt_sec=float(rate_limit_dt_details["effective_dt_sec"]),
                local_provider_names=tuple(self.config.ik_resolver_local_step_provider_names),
                local_options={
                    "pinocchio_damping": float(self.config.ik_resolver_pinocchio_damping),
                    "pinocchio_tolerance": float(self.config.ik_resolver_pinocchio_tolerance),
                    "joint_velocity_limit_observed_dt_sec": float(rate_limit_dt_details["observed_dt_sec"]),
                    "joint_velocity_limit_effective_dt_sec": float(rate_limit_dt_details["effective_dt_sec"]),
                    "joint_velocity_limit_max_dt_sec": float(rate_limit_dt_details["max_dt_sec"]),
                },
            )
        else:
            result = self._ik_resolver.resolve(request)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        provider_metadata = (
            dict(self._ik_resolver_provider.last_metadata)
            if self._ik_resolver_provider is not None
            else {}
        )
        base_metadata: dict[str, object] = {
            **dict(result.metadata),
            "status": result.status.value,
            "source": "generic_ik_resolver",
            "result_message": result.message,
            "candidate_source": result.candidate_source,
            "accepted": bool(result.accepted),
            "seed_joint_deg": np.rad2deg(reference_qpos).astype(float).tolist(),
            "seed_labels": list(seed_labels),
            "requested_target_pose": list(command_pose),
            "ik_resolver_init": dict(self._ik_resolver_init_metadata),
        }
        self._attach_ik_timing_metadata(
            base_metadata,
            source="generic_ik_resolver",
            elapsed_ms=elapsed_ms,
            provider_metadata=provider_metadata,
        )

        if not result.accepted or result.qpos is None:
            if self._latest_wrench_context is not None:
                self._latest_wrench_context["ik_method"] = ["generic_ik_resolver"]
                self._latest_wrench_context["ik_metadata"] = base_metadata
            return (
                command_pose,
                [0.0] * 6,
                np.rad2deg(reference_qpos).astype(float).tolist(),
                1.0,
                list(offset),
                -100,
            )

        command_joint = np.rad2deg(np.asarray(result.qpos, dtype=float).reshape(6)).astype(float).tolist()
        effective_pose = normalize_pose(matrix_to_pose6(result.effective_target_pose))
        method = str(base_metadata.get("method") or result.candidate_source or "generic_ik_resolver")
        if self._latest_wrench_context is not None:
            self._latest_wrench_context["ik_method"] = [method]
            self._latest_wrench_context["ik_resolver_tolerance_fraction"] = [float(result.tolerance_fraction)]
            self._latest_wrench_context["ik_metadata"] = base_metadata
        return (
            effective_pose,
            command_joint,
            np.rad2deg(reference_qpos).astype(float).tolist(),
            float(result.tolerance_fraction),
            list(offset),
            0,
        )

    def _command_from_pose_with_ik_backoff(
        self,
        *,
        reference_pose: list[float],
        offset: list[float],
        seed_joint_override: list[float] | None = None,
    ) -> tuple[list[float], list[float], list[float], float, list[float], int]:
        assert self._ik_seed is not None
        seed_joint = (
            finite_list(seed_joint_override, expected_len=6, label="seed_joint_override")
            if seed_joint_override is not None
            else list(self._ik_seed)
        )
        legacy_started_at = time.perf_counter()

        def timed_metadata(metadata: dict[str, object]) -> dict[str, object]:
            return self._attach_ik_timing_metadata(
                metadata,
                source="rm_sdk_legacy",
                elapsed_ms=(time.perf_counter() - legacy_started_at) * 1000.0,
            )

        command_pose = normalize_pose(apply_pose_offset(reference_pose, offset))
        code, command_joint, used_seed, method = self._solve_ik_with_current_retry(
            seed_joint=seed_joint,
            command_pose=command_pose,
        )
        if int(code) == 0:
            if self._latest_wrench_context is not None:
                self._latest_wrench_context["ik_method"] = [method]
                self._latest_wrench_context["ik_metadata"] = timed_metadata({
                    "status": "pass",
                    "method": method,
                    "source": "rm_sdk_legacy",
                    "seed_joint_deg": list(used_seed),
                })
            return command_pose, command_joint, used_seed, 1.0, list(offset), int(code)

        last_code = int(code)
        last_seed = list(used_seed)
        if norm(offset) > 1e-12:
            for scale in self._ik_backoff_scales():
                scaled = _scaled_offset(offset, scale)
                scaled_pose = normalize_pose(apply_pose_offset(reference_pose, scaled))
                retry_code, retry_joint, retry_seed, retry_method = self._solve_ik_with_current_retry(
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
                        self._latest_wrench_context["ik_method"] = [retry_method]
                        self._latest_wrench_context["ik_metadata"] = timed_metadata({
                            "status": "relaxed",
                            "method": retry_method,
                            "source": "rm_sdk_legacy",
                            "relaxation_type": "offset_backoff",
                            "scale": float(scale),
                            "seed_joint_deg": list(retry_seed),
                        })
                    return scaled_pose, retry_joint, retry_seed, float(scale), scaled, int(retry_code)

        if self._previous_command_pose is not None:
            for scale in self._reference_backoff_scales():
                scaled_reference = _interpolate_pose(self._previous_command_pose, reference_pose, scale)
                scaled_pose = normalize_pose(apply_pose_offset(scaled_reference, offset))
                retry_code, retry_joint, retry_seed, retry_method = self._solve_ik_with_current_retry(
                    seed_joint=seed_joint,
                    command_pose=scaled_pose,
                )
                last_code = int(retry_code)
                last_seed = list(retry_seed)
                if int(retry_code) == 0:
                    if self._reference is not None and self.reference_kind == "pose":
                        self._reference.current = list(scaled_reference)
                    if self._latest_wrench_context is not None:
                        self._latest_wrench_context["ik_reference_backoff_scale"] = [float(scale)]
                        self._latest_wrench_context["ik_method"] = [retry_method]
                        self._latest_wrench_context["ik_metadata"] = timed_metadata({
                            "status": "relaxed",
                            "method": retry_method,
                            "source": "rm_sdk_legacy",
                            "relaxation_type": "reference_backoff",
                            "scale": float(scale),
                            "seed_joint_deg": list(retry_seed),
                        })
                    return scaled_pose, retry_joint, retry_seed, float(scale), list(offset), int(retry_code)

        if self._latest_wrench_context is not None:
            self._latest_wrench_context["ik_metadata"] = timed_metadata({
                "status": "smoothing_required",
                "method": "rm_sdk_legacy",
                "source": "rm_sdk_legacy",
                "last_code": int(last_code),
                "seed_joint_deg": list(last_seed),
            })
        return command_pose, command_joint, last_seed, 1.0, list(offset), last_code

    def _command_step_failure_message(
        self,
        *,
        command_pose: Iterable[float],
        command_joint: Iterable[float],
        command_step: float,
    ) -> str:
        previous_joint = (
            []
            if self._previous_command_joint is None
            else finite_list(self._previous_command_joint, expected_len=6, label="previous_command_joint_deg")
        )
        candidate_joint = finite_list(list(command_joint)[:6], expected_len=6, label="candidate_joint_deg")
        deltas = _joint_deltas_deg(previous_joint, candidate_joint) if previous_joint else []
        max_index = deltas.index(max(deltas)) if deltas else -1
        lines = [
            f"{self.side}: command joint step {command_step:.4f}deg > allowed_command_joint_step",
            f"allowed_command_joint_step_deg={self._max_allowed_command_joint_step_deg():.4f}",
            f"max_command_joint_step_deg={float(self.config.max_command_joint_step_deg):.4f}",
            f"max_command_joint_velocity_deg_s={float(self.config.max_command_joint_velocity_deg_s):.4f}",
        ]
        if max_index >= 0:
            lines.append(f"max_delta_joint=J{max_index + 1}")
            lines.append(f"joint_delta_deg={_format_vector(deltas, precision=3)}")
            lines.append(f"previous_command_joint_deg={_format_vector(previous_joint, precision=3)}")
        lines.append(f"candidate_joint_deg={_format_vector(candidate_joint, precision=3)}")
        lines.append(f"candidate_pose={_format_vector(command_pose, precision=6)}")
        if self._previous_command_pose is not None:
            lines.append(f"previous_command_pose={_format_vector(self._previous_command_pose, precision=6)}")
        if self._latest_wrench_context is not None:
            for key in (
                "ik_method",
                "ik_backoff_scale",
                "ik_reference_backoff_scale",
                "ik_command_step_backoff_scale",
                "ik_command_step_candidate_deg",
                "ik_command_step_allowed_deg",
                "ik_command_step_applied_deg",
            ):
                value = self._latest_wrench_context.get(key)
                if value is None:
                    continue
                if isinstance(value, list) and len(value) == 1:
                    lines.append(f"{key}={value[0]}")
                else:
                    lines.append(f"{key}={value}")
        return "\n".join(lines)

    def _observed_control_dt_sec(self) -> float:
        dt = self._current_step_dt
        if dt is None or not math.isfinite(float(dt)) or float(dt) <= 0.0:
            return 1.0 / float(self.config.control_hz)
        return float(dt)

    def _joint_rate_limit_dt_details(self) -> dict[str, float]:
        observed_dt = self._observed_control_dt_sec()
        max_dt = 1.0
        effective_dt = min(observed_dt, max_dt)
        if self._joint_velocity_limit_profile is not None:
            max_dt = float(self._joint_velocity_limit_profile.max_dt_sec)
            effective_dt = self._joint_velocity_limit_profile.effective_dt_sec(observed_dt)
        return {
            "observed_dt_sec": float(observed_dt),
            "effective_dt_sec": float(effective_dt),
            "max_dt_sec": float(max_dt),
        }

    def _joint_rate_limit_dt(self) -> float:
        return self._joint_rate_limit_dt_details()["effective_dt_sec"]

    def _emit_joint_rate_limit_warning(self, metadata: dict[str, object]) -> None:
        now = time.time()
        if now - self._last_joint_rate_limit_warning_time < 1.0:
            return
        self._last_joint_rate_limit_warning_time = now
        print(
            f"WARNING: {self.side} joint rate limiter active "
            f"profile={metadata.get('joint_velocity_limit_profile')} "
            f"requested_max_delta={float(metadata.get('joint_velocity_limit_requested_max_abs_delta_deg', 0.0)):.3f}deg "
            f"applied_max_delta={float(metadata.get('joint_velocity_limit_applied_max_abs_delta_deg', 0.0)):.3f}deg "
            f"ratio={float(metadata.get('joint_velocity_limit_ratio', 0.0)):.3f}; "
            "continuing to chase latest IK target."
        )

    def _record_joint_rate_limit_metadata(self, metadata: dict[str, object]) -> None:
        if self._latest_wrench_context is None:
            return
        self._latest_wrench_context.update(metadata)
        ik_metadata = dict(self._latest_wrench_context.get("ik_metadata", {}))
        ik_metadata.update(metadata)
        self._latest_wrench_context["ik_metadata"] = ik_metadata

    def _apply_joint_rate_limiter(
        self,
        *,
        command_pose: list[float],
        command_joint: list[float],
    ) -> tuple[list[float], list[float]]:
        previous_joint = self._previous_command_joint
        if previous_joint is None:
            return command_pose, command_joint
        if self._joint_rate_limiter is None or self._joint_velocity_limit_profile is None:
            self._initialize_joint_rate_limiter()
        if self._joint_rate_limiter is None or self._joint_velocity_limit_profile is None:
            return command_pose, command_joint

        target_qpos = _joint_deg_to_rad(command_joint, label="rate_limit_target_joint")
        previous_qpos = _joint_deg_to_rad(previous_joint, label="rate_limit_previous_joint")
        dt_details = self._joint_rate_limit_dt_details()
        dt_sec = float(dt_details["effective_dt_sec"])
        result = self._joint_rate_limiter.step(
            target_qpos=target_qpos,
            reference_qpos=previous_qpos,
            dt_sec=dt_sec,
        )
        limited_joint = np.rad2deg(result.qpos).astype(float).tolist()
        requested_delta_deg = np.rad2deg(target_qpos - previous_qpos).astype(float)
        requested_max = float(np.max(np.abs(requested_delta_deg))) if requested_delta_deg.size else 0.0
        metadata: dict[str, object] = {
            "joint_rate_limiter_enabled": True,
            "joint_rate_limited": bool(result.limited),
            "joint_velocity_limit_profile": self._joint_velocity_limit_profile.name,
            "joint_velocity_limit_dt_sec": float(dt_sec),
            "joint_velocity_limit_observed_dt_sec": float(dt_details["observed_dt_sec"]),
            "joint_velocity_limit_effective_dt_sec": float(dt_sec),
            "joint_velocity_limit_max_dt_sec": float(dt_details["max_dt_sec"]),
            "joint_velocity_limit_requested_delta_deg": requested_delta_deg.tolist(),
            "joint_velocity_limit_requested_max_abs_delta_deg": requested_max,
            "joint_velocity_limit_delta_deg": result.delta_deg.tolist(),
            "joint_velocity_limit_applied_max_abs_delta_deg": float(result.max_abs_delta_deg),
            "joint_velocity_limit_max_step_deg": result.max_step_deg.tolist(),
            "joint_velocity_limit_ratio": float(result.limit_ratio),
            "joint_velocity_limit_target_joint_deg": list(command_joint),
            "joint_velocity_limit_output_joint_deg": list(limited_joint),
        }

        report = check_rm65_joint_safety(
            self.robot,
            limited_joint,
            side=self.side,
            label=self._step_index + 1,
            previous_joint_deg=previous_joint,
            check_position_limit=True,
            check_self_collision=True,
            context={
                "phase": "impedance_rate_limited_command",
                "requested_pose": list(command_pose),
                "ik_target_pose": list(command_pose),
                "ik_target_joint_deg": list(command_joint),
                "ik_target_delta_from_previous_deg": requested_delta_deg.tolist(),
                "joint_rate_limited": bool(result.limited),
                "joint_velocity_limit_profile": self._joint_velocity_limit_profile.name,
                "joint_velocity_limit_observed_dt_sec": float(dt_details["observed_dt_sec"]),
                "joint_velocity_limit_effective_dt_sec": float(dt_sec),
                "joint_velocity_limit_max_dt_sec": float(dt_details["max_dt_sec"]),
                "joint_velocity_limit_delta_deg": result.delta_deg.tolist(),
                "joint_velocity_limit_max_step_deg": result.max_step_deg.tolist(),
                "joint_velocity_limit_ratio": float(result.limit_ratio),
            },
        )
        metadata["joint_safety"] = {
            "ok": bool(report.ok),
            "position_limit_code": report.position_limit_code,
            "position_limit_available": bool(report.position_limit_available),
            "self_collision_code": report.self_collision_code,
            "self_collision_available": bool(report.self_collision_available),
            "status_text": report.status_text,
        }
        self._record_joint_rate_limit_metadata(metadata)
        report.raise_if_unsafe()

        if result.limited:
            self._emit_joint_rate_limit_warning(metadata)
            return self._forward_pose(limited_joint), limited_joint
        return command_pose, command_joint

    def _apply_joint_step_policy(
        self,
        *,
        command_pose: list[float],
        command_joint: list[float],
    ) -> tuple[list[float], list[float]]:
        if self.config.joint_rate_limiter_enabled:
            return self._apply_joint_rate_limiter(command_pose=command_pose, command_joint=command_joint)
        return self._limit_command_joint_step(command_pose=command_pose, command_joint=command_joint)

    def _limit_command_joint_step(
        self,
        *,
        command_pose: list[float],
        command_joint: list[float],
    ) -> tuple[list[float], list[float]]:
        previous_joint = self._previous_command_joint
        if previous_joint is None:
            return command_pose, command_joint
        max_allowed = self._max_allowed_command_joint_step_deg()
        command_step = _joint_delta_deg(previous_joint, command_joint)
        if command_step <= max_allowed:
            return command_pose, command_joint
        if self._latest_wrench_context is not None:
            self._latest_wrench_context["ik_command_step_candidate_deg"] = [float(command_step)]
            self._latest_wrench_context["ik_command_step_allowed_deg"] = [float(max_allowed)]

        if self._previous_command_pose is not None:
            seed_joint = list(previous_joint)
            step_backoff_started_at = time.perf_counter()
            for scale in self._command_step_backoff_scales(
                command_step=command_step,
                max_allowed=max_allowed,
            ):
                scaled_pose = _interpolate_pose(self._previous_command_pose, command_pose, scale)
                retry_code, retry_joint, _retry_seed, retry_method = self._solve_ik_with_current_retry(
                    seed_joint=seed_joint,
                    command_pose=scaled_pose,
                )
                if int(retry_code) != 0:
                    continue
                retry_joint = finite_list(retry_joint[:6], expected_len=6, label="ik_joint_deg")
                retry_step = _joint_delta_deg(previous_joint, retry_joint)
                if retry_step <= max_allowed:
                    if self._latest_wrench_context is not None:
                        self._latest_wrench_context["ik_command_step_backoff_scale"] = [float(scale)]
                        self._latest_wrench_context["ik_command_step_applied_deg"] = [float(retry_step)]
                        self._latest_wrench_context["ik_method"] = [retry_method]
                        self._latest_wrench_context["ik_metadata"] = self._attach_ik_timing_metadata(
                            {
                                "status": "relaxed",
                                "method": retry_method,
                                "source": "rm_sdk_legacy",
                                "relaxation_type": "command_step_backoff",
                                "scale": float(scale),
                                "candidate_step_deg": float(command_step),
                                "applied_step_deg": float(retry_step),
                            },
                            source="rm_sdk_legacy",
                            elapsed_ms=(time.perf_counter() - step_backoff_started_at) * 1000.0,
                        )
                    if self._reference is not None and self.reference_kind == "pose":
                        offset = None
                        if self._latest_wrench_context is not None:
                            raw_offset = self._latest_wrench_context.get("offset")
                            if raw_offset is not None:
                                offset = finite_list(raw_offset, expected_len=6, label="offset")
                        if offset is None:
                            self._reference.current = list(scaled_pose)
                        else:
                            self._reference.current = normalize_pose(
                                [scaled_pose[index] - offset[index] for index in range(6)]
                            )
                    elif self._reference is not None and self.reference_kind == "joint":
                        self._reference.current = list(retry_joint)
                    return scaled_pose, retry_joint

        raise RuntimeError(
            self._command_step_failure_message(
                command_pose=command_pose,
                command_joint=command_joint,
                command_step=command_step,
            )
        )

    def _hardware_command_failure_message(
        self,
        *,
        exc: Exception,
        phase: str,
        command_pose: Iterable[float],
        command_joint: Iterable[float],
        command_step: float,
        allowed_command_step: float,
    ) -> str:
        lines = [
            f"{self.side}: hardware command failure during {phase}: {type(exc).__name__}: {exc}",
            f"step={int(self._step_index + 1)}",
            f"command_transport={self.config.pose_command_transport if self.reference_kind == 'pose' else 'joint'}",
            f"command_joint_step_deg={float(command_step):.4f}",
            f"allowed_command_joint_step_deg={float(allowed_command_step):.4f}",
            f"command_joint_deg={_format_vector(command_joint, precision=3)}",
            f"command_pose={_format_vector(command_pose, precision=6)}",
        ]
        if self._previous_command_joint is not None:
            previous_joint = finite_list(
                self._previous_command_joint,
                expected_len=6,
                label="previous_command_joint_deg",
            )
            lines.append(f"previous_command_joint_deg={_format_vector(previous_joint, precision=3)}")
            lines.append(
                "joint_delta_deg="
                + _format_vector(_joint_deltas_deg(previous_joint, command_joint), precision=3)
            )
        if self._previous_command_pose is not None:
            lines.append(f"previous_command_pose={_format_vector(self._previous_command_pose, precision=6)}")
        if self._latest_wrench_context is not None:
            ik_method = self._latest_wrench_context.get("ik_method")
            if ik_method is not None:
                lines.append(f"ik_method={ik_method[0]}")
            for key in (
                "ik_backoff_scale",
                "ik_reference_backoff_scale",
                "ik_command_step_backoff_scale",
                "ik_command_step_candidate_deg",
                "ik_command_step_allowed_deg",
                "ik_command_step_applied_deg",
            ):
                value = self._latest_wrench_context.get(key)
                if value is not None:
                    lines.append(f"{key}={value[0] if isinstance(value, list) and len(value) == 1 else value}")
        return "\n".join(lines)

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
            raw_deadbanded = self._latest_wrench_context.get("raw_deadbanded")
            target_offset = self._latest_wrench_context.get("target_offset")
            applied_offset = self._latest_wrench_context.get("applied_offset")
            ik_backoff_scale = self._latest_wrench_context.get("ik_backoff_scale")
            ik_reference_backoff_scale = self._latest_wrench_context.get("ik_reference_backoff_scale")
            ik_method = self._latest_wrench_context.get("ik_method")
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
            if raw_deadbanded is not None:
                lines.append(
                    "raw_deadbanded_wrench="
                    + _format_vector(raw_deadbanded, precision=5)
                    + f" |F|={norm(raw_deadbanded[:3]):.5f}N |T|={norm(raw_deadbanded[3:]):.5f}Nm"
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
            if ik_reference_backoff_scale is not None:
                lines.append(f"ik_reference_backoff_scale={float(ik_reference_backoff_scale[0]):.6f}")
            if ik_method is not None:
                lines.append(f"ik_method={ik_method[0]}")
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

    def _emit_ik_hold_warning(self, message: str) -> None:
        first_line = str(message).splitlines()[0] if str(message).strip() else "IK failure"
        now = time.time()
        if (
            self._last_ik_hold_warning_key == first_line
            and now - self._last_ik_hold_warning_time < 1.0
        ):
            return
        self._last_ik_hold_warning_key = first_line
        self._last_ik_hold_warning_time = now
        print(
            f"WARNING: {self.side} impedance IK hold_last_safe: {first_line}; "
            "holding previous safe command until a later target is solvable."
        )

    def _hold_last_safe_command_after_ik_failure(
        self,
        *,
        reason: str,
        message: str,
        target_pose: Iterable[float],
        seed_joint: Iterable[float] | None = None,
        code: object | None = None,
    ) -> tuple[list[float], list[float]] | None:
        if not bool(self.config.hold_last_safe_on_ik_failure):
            return None
        if self._previous_command_pose is None or self._previous_command_joint is None:
            return None
        self._emit_ik_hold_warning(message)
        if self._reference is not None:
            if self.reference_kind == "pose":
                self._reference.current = list(self._previous_command_pose)
            elif self.reference_kind == "joint":
                self._reference.current = list(self._previous_command_joint)
        if self._latest_wrench_context is not None:
            previous_metadata = dict(self._latest_wrench_context.get("ik_metadata", {}))
            source = str(previous_metadata.get("source") or "rm_sdk_legacy")
            failed_metadata = dict(previous_metadata)
            failed_metadata.pop("failed_ik_metadata", None)
            failed_metadata.pop("hold_previous_ik_metadata", None)
            metadata = {
                "status": "hold_last_safe",
                "method": "hold_previous_command",
                "source": source,
                "reason": str(reason),
                "last_code": None if code is None else int(code),
                "seed_joint_deg": [] if seed_joint is None else list(seed_joint),
                "target_pose": list(target_pose),
                "message": str(message).splitlines()[0] if str(message).strip() else "",
                "previous_ik_status": previous_metadata.get("status"),
                "previous_ik_method": previous_metadata.get("method"),
                "previous_ik_source": previous_metadata.get("source"),
                "failed_ik_metadata": failed_metadata,
            }
            for key in (
                "ik_frame_total_ms",
                "ik_solve_total_ms",
                "ik_backend_solve_ms",
                "ik_resolve_elapsed_ms",
                "ik_provider_elapsed_ms",
                "ik_time_budget_ms",
                "ik_time_budget_exceeded",
                "ik_timing_source",
                "provider",
                "ik_debug_profile",
                "best_rejected",
                "last_rejections",
                "local_step_status",
                "local_step_first_status",
                "local_step_first_attempted",
                "local_step_first_periodic_full_solve",
                "local_step_scale_strategy",
                "local_step_recovery_state",
                "local_step_rejection_count",
                "local_step_last_rejections",
                "local_step_singularity_escape",
                "local_step_provider",
                "local_step_provider_horizons",
                "full_target_result_message",
                "provider_budget_skipped_names",
                "provider_global_time_budget_ms",
                "provider_global_time_budget_scope",
            ):
                if key in previous_metadata:
                    metadata[key] = previous_metadata[key]
            self._latest_wrench_context["ik_metadata"] = metadata
        return list(self._previous_command_pose), list(self._previous_command_joint)

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
        self._joint_hint_reference = _ReferenceInterpolator(
            list(self._state["joint_deg"]),
            kind="joint",
            action_hz=self.config.action_hz,
        )
        self._previous_command_joint = list(self._state["joint_deg"])
        self._previous_command_pose = list(self._state["pose"])
        self._ik_seed = list(self._state["joint_deg"])
        self._model_fk_pose = self._forward_pose(self._state["joint_deg"])
        self._initialize_joint_rate_limiter()
        self._initialize_ik_resolver()
        self._latest_wrench_context = None
        self._post_start_preflight()
        initial_sample_time = time.perf_counter()
        self._previous_measured_pose = list(self._state["pose"])
        self._previous_measured_joint = list(self._state["joint_deg"])
        self._previous_measured_sample_time = initial_sample_time
        self._last_step_time = initial_sample_time
        self._started = True

    def reset_motion_state(self, joint_rad: Iterable[float] | None = None) -> None:
        if joint_rad is None:
            if self.robot is None:
                return
            state = _read_state(self.robot, self.side)
            state["pose"] = normalize_pose(state["pose"])
        else:
            if self.robot is None:
                raise RuntimeError(f"{self.side}: cannot reset motion state without an RM65 robot")
            joint_values = finite_list(joint_rad, expected_len=6, label="joint_rad")
            joint_deg = [float(math.degrees(value)) for value in joint_values]
            state = {
                "joint_deg": joint_deg,
                "pose": self._forward_pose(joint_deg),
            }

        self._state = {
            "joint_deg": list(state["joint_deg"]),
            "pose": normalize_pose(state["pose"]),
        }
        initial_reference = self._initial_reference(self._state)
        self._reference = _ReferenceInterpolator(
            initial_reference,
            kind=self.reference_kind,
            action_hz=self.config.action_hz,
        )
        self._joint_hint_reference = _ReferenceInterpolator(
            list(self._state["joint_deg"]),
            kind="joint",
            action_hz=self.config.action_hz,
        )
        self._previous_command_joint = list(self._state["joint_deg"])
        self._previous_command_pose = list(self._state["pose"])
        self._ik_seed = list(self._state["joint_deg"])
        self._model_fk_pose = self._forward_pose(self._state["joint_deg"])
        self._admittance = self.config.make_admittance()
        if self._joint_rate_limiter is not None:
            self._joint_rate_limiter.reset(
                _joint_deg_to_rad(self._previous_command_joint, label="previous_command_joint")
            )
        self._latest_wrench_context = None
        reset_sample_time = time.perf_counter()
        self._previous_measured_pose = list(self._state["pose"])
        self._previous_measured_joint = list(self._state["joint_deg"])
        self._previous_measured_sample_time = reset_sample_time
        self._last_step_time = reset_sample_time
        self._current_step_dt = None
        self._step_index = 0

    def set_target(self, target: Iterable[float], *, duration_sec: float | None = None) -> None:
        self.start()
        assert self._reference is not None
        self._reference.set_target(target, duration_sec)

    def step(
        self,
        target: Iterable[float] | None = None,
        *,
        target_joint_deg: Iterable[float] | None = None,
    ) -> dict[str, object]:
        self.start()
        assert self.robot is not None
        assert self._state is not None
        assert self._processor is not None
        assert self._admittance is not None
        assert self._reference is not None
        assert self._previous_command_joint is not None
        timing_enabled = bool(self.config.debug_timing)
        timing_started_at = time.perf_counter()
        timing_last_at = timing_started_at
        timing_phases: list[dict[str, object]] = []

        def mark_timing(name: str) -> None:
            nonlocal timing_last_at
            if not timing_enabled:
                return
            now_timing = time.perf_counter()
            timing_phases.append(
                {
                    "name": name,
                    "dt_ms": (now_timing - timing_last_at) * 1000.0,
                    "elapsed_ms": (now_timing - timing_started_at) * 1000.0,
                }
            )
            timing_last_at = now_timing

        def timing_payload(*, state_read_performed: bool) -> dict[str, object] | None:
            if not timing_enabled:
                return None
            now_timing = time.perf_counter()
            return {
                "enabled": True,
                "side": self.side,
                "step": int(self._step_index),
                "total_ms": (now_timing - timing_started_at) * 1000.0,
                "dt_sec": float(self._current_step_dt),
                "state_read_performed": bool(state_read_performed),
                "pose_command_transport": self.config.pose_command_transport,
                "phases": timing_phases,
            }

        now = time.perf_counter()
        dt = now - self._last_step_time if self._last_step_time is not None else 1.0 / self.config.control_hz
        self._last_step_time = now
        self._current_step_dt = dt
        if target is not None:
            self._reference.set_target(target)
        has_preferred_joint = target_joint_deg is not None
        if has_preferred_joint and self._joint_hint_reference is not None:
            self._joint_hint_reference.set_target(target_joint_deg)
        reference = self._reference.advance(dt)
        preferred_joint = (
            None
            if not has_preferred_joint or self._joint_hint_reference is None
            else self._joint_hint_reference.advance(dt)
        )
        mark_timing("reference")

        readings = _read_force(self.robot, self.side)
        mark_timing("read_force")
        processed = self._processor.process(readings[self.config.force_source])
        raw_deadbanded = list(processed["deadbanded"])
        admittance_wrench = admittance_wrench_for_arm(raw_deadbanded, self.side)
        return_activity_wrench = admittance_wrench_for_arm(processed["filtered"], self.side)
        return_force_deadband_n = self.config.effective_return_force_deadband_n
        return_torque_deadband_nm = self.config.effective_return_torque_deadband_nm
        update = self._admittance.update(
            admittance_wrench,
            dt,
            return_activity_wrench=return_activity_wrench,
            return_force_deadband_n=return_force_deadband_n,
            return_torque_deadband_nm=return_torque_deadband_nm,
        )
        self._latest_wrench_context = {
            "tare_corrected": list(processed["tare_corrected"]),
            "active_wrench": list(update["active_wrench"]),
            "deadbanded": list(admittance_wrench),
            "raw_deadbanded": raw_deadbanded,
            "return_activity_wrench": list(return_activity_wrench),
            "return_force_deadband_n": return_force_deadband_n,
            "return_torque_deadband_nm": return_torque_deadband_nm,
            "target_offset_unclamped": list(update["target_offset_unclamped"]),
            "target_offset_after_return": list(update["target_offset_after_return"]),
            "target_offset_after_offset_limit": list(update["target_offset_after_offset_limit"]),
            "target_offset": list(update["target_offset"]),
            "offset": list(update["offset"]),
            "offset_delta_before_velocity_limit": list(update["offset_delta_before_velocity_limit"]),
            "offset_step_after_velocity_limit": list(update["offset_step_after_velocity_limit"]),
            "return_applied_axes": list(update["return_applied_axes"]),
            "return_paused_axes": list(update["return_paused_axes"]),
            "linear_return_paused": bool(update["linear_return_paused"]),
            "angular_return_paused": bool(update["angular_return_paused"]),
            "linear_offset_limit_applied": bool(update["linear_offset_limit_applied"]),
            "angular_offset_limit_applied": bool(update["angular_offset_limit_applied"]),
            "linear_velocity_limit_applied": bool(update["linear_velocity_limit_applied"]),
            "angular_velocity_limit_applied": bool(update["angular_velocity_limit_applied"]),
            "angular_axis_velocity_limit_applied": bool(update["angular_axis_velocity_limit_applied"]),
            "linear_return_velocity_limit_applied": bool(update["linear_return_velocity_limit_applied"]),
            "angular_return_velocity_limit_applied": bool(update["angular_return_velocity_limit_applied"]),
            "linear_step_velocity_limit_m_s": float(update["linear_step_velocity_limit_m_s"]),
            "angular_step_velocity_limit_rad_s": float(update["angular_step_velocity_limit_rad_s"]),
            "angular_axis_velocity_scales": list(update["angular_axis_velocity_scales"]),
            "angular_axis_step_velocity_limits_rad_s": list(update["angular_axis_step_velocity_limits_rad_s"]),
            "preferred_joint_deg": [] if preferred_joint is None else list(preferred_joint),
        }
        mark_timing("process_force_admittance")
        command_pose, command_joint = self._command_from_reference(
            reference,
            update["offset"],
            preferred_joint=preferred_joint,
        )
        realized_offset = (
            None
            if self._latest_wrench_context is None
            else self._latest_wrench_context.get("offset")
        )
        if realized_offset is not None:
            update["offset"] = finite_list(realized_offset, expected_len=6, label="realized_offset")
        mark_timing("command_from_reference")

        command_step = _joint_delta_deg(self._previous_command_joint, command_joint)
        allowed_command_step = self._max_allowed_command_joint_step_deg()
        if command_step > allowed_command_step:
            raise RuntimeError(
                self._command_step_failure_message(
                    command_pose=command_pose,
                    command_joint=command_joint,
                    command_step=command_step,
                )
            )
        force_norm = norm(processed["tare_corrected"][:3])
        torque_norm = norm(processed["tare_corrected"][3:])
        if force_norm > self.config.hard_stop_force_n:
            raise RuntimeError(f"{self.side}: force {force_norm:.4f}N > hard_stop_force_n")
        if torque_norm > self.config.hard_stop_torque_nm:
            raise RuntimeError(f"{self.side}: torque {torque_norm:.4f}Nm > hard_stop_torque_nm")
        mark_timing("safety_checks")

        try:
            self._send(command_pose, command_joint)
        except Exception as exc:
            raise RuntimeError(
                self._hardware_command_failure_message(
                    exc=exc,
                    phase="send",
                    command_pose=command_pose,
                    command_joint=command_joint,
                    command_step=command_step,
                    allowed_command_step=allowed_command_step,
                )
            ) from exc
        mark_timing("send")
        self._step_index += 1
        state_read_performed = self._step_index % self.config.state_read_every == 0
        state_sample_time: float | None = None
        previous_measured_pose = None if self._previous_measured_pose is None else list(self._previous_measured_pose)
        previous_measured_joint = None if self._previous_measured_joint is None else list(self._previous_measured_joint)
        previous_measured_sample_time = self._previous_measured_sample_time
        if state_read_performed:
            try:
                self._state = _read_state(self.robot, self.side)
                state_sample_time = time.perf_counter()
            except Exception as exc:
                raise RuntimeError(
                    self._hardware_command_failure_message(
                        exc=exc,
                        phase="state_read_after_send",
                        command_pose=command_pose,
                        command_joint=command_joint,
                        command_step=command_step,
                        allowed_command_step=allowed_command_step,
                    )
                ) from exc
            mark_timing("state_read_after_send")
        else:
            mark_timing("state_read_skipped")
        self._previous_command_joint = list(command_joint)
        self._previous_command_pose = list(command_pose)
        measured_pose = None if self._state is None else list(self._state["pose"])
        pose_delta = _pose_delta(command_pose, measured_pose) if measured_pose is not None else (math.nan, math.nan)
        measured_joint = None if self._state is None else list(self._state["joint_deg"])
        reference_to_command_linear, reference_to_command_angular = _pose_delta_vectors(
            source=reference,
            target=command_pose,
        )
        applied_offset = finite_list(
            self._latest_wrench_context.get("applied_offset", update["offset"]),
            expected_len=6,
            label="applied_offset",
        )
        target_offset = finite_list(update["target_offset"], expected_len=6, label="target_offset")
        admittance_target_linear_norm = norm(target_offset[:3])
        admittance_target_angular_norm = norm(target_offset[3:])
        admittance_offset_linear_norm = norm(applied_offset[:3])
        admittance_offset_angular_norm = norm(applied_offset[3:])
        commanded_offset_linear_norm = norm(reference_to_command_linear)
        commanded_offset_angular_norm = norm(reference_to_command_angular)
        offset_to_command_linear_error = _vector_error_norm(applied_offset[:3], reference_to_command_linear)
        offset_to_command_angular_error = _vector_error_norm(applied_offset[3:], reference_to_command_angular)
        target_to_command_linear_error = _vector_error_norm(target_offset[:3], reference_to_command_linear)
        target_to_command_angular_error = _vector_error_norm(target_offset[3:], reference_to_command_angular)
        reference_to_measured_linear = None
        reference_to_measured_angular = None
        command_to_measured_linear = None
        command_to_measured_angular = None
        measured_to_command_linear = None
        measured_to_command_angular = None
        measured_pose_velocity_linear = None
        measured_pose_velocity_angular = None
        measured_joint_velocity = None
        measured_velocity_dt_sec = None
        command_to_measured_joint = None
        measured_to_command_joint = None
        if measured_pose is not None:
            reference_to_measured_linear, reference_to_measured_angular = _pose_delta_vectors(
                source=reference,
                target=measured_pose,
            )
            command_to_measured_linear, command_to_measured_angular = _pose_delta_vectors(
                source=command_pose,
                target=measured_pose,
            )
            measured_to_command_linear, measured_to_command_angular = _pose_delta_vectors(
                source=measured_pose,
                target=command_pose,
            )
        if measured_joint is not None:
            command_to_measured_joint = [
                float(measured_joint[index] - command_joint[index])
                for index in range(6)
            ]
            measured_to_command_joint = [
                float(command_joint[index] - measured_joint[index])
                for index in range(6)
            ]
        if (
            state_read_performed
            and state_sample_time is not None
            and measured_pose is not None
            and previous_measured_pose is not None
            and previous_measured_joint is not None
            and previous_measured_sample_time is not None
        ):
            measured_velocity_dt_sec = max(0.0, state_sample_time - previous_measured_sample_time)
            if measured_velocity_dt_sec > 1e-9:
                prev_to_measured_linear, prev_to_measured_angular = _pose_delta_vectors(
                    source=previous_measured_pose,
                    target=measured_pose,
                )
                measured_pose_velocity_linear = _scaled_vector(
                    prev_to_measured_linear,
                    1.0 / measured_velocity_dt_sec,
                )
                measured_pose_velocity_angular = _scaled_vector(
                    prev_to_measured_angular,
                    1.0 / measured_velocity_dt_sec,
                )
                if measured_joint is not None:
                    measured_joint_velocity = [
                        float(measured_joint[index] - previous_measured_joint[index])
                        / measured_velocity_dt_sec
                        for index in range(6)
                    ]
        if state_read_performed and measured_pose is not None and measured_joint is not None:
            self._previous_measured_pose = list(measured_pose)
            self._previous_measured_joint = list(measured_joint)
            self._previous_measured_sample_time = (
                state_sample_time if state_sample_time is not None else time.perf_counter()
            )
        command_singularity = self._singularity_metadata(command_joint)
        measured_singularity = (
            {"available": False, "label": "unavailable", "error": "state_unavailable"}
            if measured_joint is None
            else self._singularity_metadata(measured_joint)
        )
        command_joint_limit_margin = self._joint_limit_margin_metadata(command_joint)
        measured_joint_limit_margin = (
            {"available": False, "error": "state_unavailable"}
            if measured_joint is None
            else self._joint_limit_margin_metadata(measured_joint)
        )
        singularity_metadata = {
            "command": command_singularity,
            "measured": measured_singularity,
        }
        joint_limit_margin_metadata = {
            "command": command_joint_limit_margin,
            "measured": measured_joint_limit_margin,
        }
        ik_metadata = dict(self._latest_wrench_context.get("ik_metadata", {}))
        ik_metadata.setdefault("command_singularity", command_singularity)
        ik_metadata.setdefault("measured_singularity", measured_singularity)
        ik_metadata.setdefault("command_joint_limit_margin", command_joint_limit_margin)
        ik_metadata.setdefault("measured_joint_limit_margin", measured_joint_limit_margin)
        ik_metadata.setdefault("ik_resolver_init", dict(self._ik_resolver_init_metadata))
        self._latest_wrench_context["ik_metadata"] = ik_metadata
        ik_timing = dict(self._latest_wrench_context.get("ik_timing", {}))
        mark_timing("metadata")
        record = {
            "step": self._step_index,
            "dt_sec": dt,
            "reference": reference,
            "force_source": self.config.force_source,
            "force_readings": {key: list(value) for key, value in readings.items()},
            "selected_force_wrench": list(readings[self.config.force_source]),
            "command_pose": command_pose,
            "command_joint_deg": command_joint,
            "measured_pose": None if self._state is None else list(self._state["pose"]),
            "measured_joint_deg": measured_joint,
            "offset": update["offset"],
            "applied_offset": applied_offset,
            "requested_offset_before_command_realization": self._latest_wrench_context.get(
                "requested_offset_before_command_realization"
            ),
            "admittance_offset_before_antiwindup": self._latest_wrench_context.get(
                "admittance_offset_before_antiwindup"
            ),
            "applied_offset_source": self._latest_wrench_context.get("applied_offset_source"),
            "admittance_offset_antiwindup_applied": self._latest_wrench_context.get(
                "admittance_offset_antiwindup_applied"
            ),
            "admittance_offset_antiwindup_skip_reason": self._latest_wrench_context.get(
                "admittance_offset_antiwindup_skip_reason"
            ),
            "admittance_offset_antiwindup_guarded_axes": self._latest_wrench_context.get(
                "admittance_offset_antiwindup_guarded_axes"
            ),
            "admittance_offset_antiwindup_active_axes": self._latest_wrench_context.get(
                "admittance_offset_antiwindup_active_axes"
            ),
            "admittance_offset_antiwindup_guard_reason": self._latest_wrench_context.get(
                "admittance_offset_antiwindup_guard_reason"
            ),
            "admittance_offset_realized_before_guard": self._latest_wrench_context.get(
                "admittance_offset_realized_before_guard"
            ),
            "applied_offset_realization_linear_error_m": self._latest_wrench_context.get(
                "applied_offset_realization_linear_error_m"
            ),
            "applied_offset_realization_angular_error_rad": self._latest_wrench_context.get(
                "applied_offset_realization_angular_error_rad"
            ),
            "target_offset_unclamped": update["target_offset_unclamped"],
            "target_offset_after_return": update["target_offset_after_return"],
            "target_offset_after_offset_limit": update["target_offset_after_offset_limit"],
            "target_offset": update["target_offset"],
            "offset_delta_before_velocity_limit": update["offset_delta_before_velocity_limit"],
            "offset_step_after_velocity_limit": update["offset_step_after_velocity_limit"],
            "return_applied_axes": update["return_applied_axes"],
            "linear_offset_limit_applied": update["linear_offset_limit_applied"],
            "angular_offset_limit_applied": update["angular_offset_limit_applied"],
            "linear_velocity_limit_applied": update["linear_velocity_limit_applied"],
            "angular_velocity_limit_applied": update["angular_velocity_limit_applied"],
            "angular_axis_velocity_limit_applied": update["angular_axis_velocity_limit_applied"],
            "linear_return_velocity_limit_applied": update["linear_return_velocity_limit_applied"],
            "angular_return_velocity_limit_applied": update["angular_return_velocity_limit_applied"],
            "linear_step_velocity_limit_m_s": update["linear_step_velocity_limit_m_s"],
            "angular_step_velocity_limit_rad_s": update["angular_step_velocity_limit_rad_s"],
            "angular_axis_velocity_scales": update["angular_axis_velocity_scales"],
            "angular_axis_step_velocity_limits_rad_s": update["angular_axis_step_velocity_limits_rad_s"],
            "active_wrench": update["active_wrench"],
            "tare_corrected_wrench": processed["tare_corrected"],
            "filtered_wrench": processed["filtered"],
            "raw_deadbanded_wrench": raw_deadbanded,
            "admittance_wrench": admittance_wrench,
            "return_activity_wrench": update["return_activity_wrench"],
            "return_force_deadband_n": return_force_deadband_n,
            "return_torque_deadband_nm": return_torque_deadband_nm,
            "return_paused_axes": update["return_paused_axes"],
            "linear_return_paused": update["linear_return_paused"],
            "angular_return_paused": update["angular_return_paused"],
            "force_norm_n": force_norm,
            "torque_norm_nm": torque_norm,
            "command_joint_step_deg": command_step,
            "allowed_command_joint_step_deg": allowed_command_step,
            "joint_rate_limited": self._latest_wrench_context.get("joint_rate_limited"),
            "joint_velocity_limit_profile": self._latest_wrench_context.get("joint_velocity_limit_profile"),
            "joint_velocity_limit_dt_sec": self._latest_wrench_context.get("joint_velocity_limit_dt_sec"),
            "joint_velocity_limit_observed_dt_sec": self._latest_wrench_context.get(
                "joint_velocity_limit_observed_dt_sec"
            ),
            "joint_velocity_limit_effective_dt_sec": self._latest_wrench_context.get(
                "joint_velocity_limit_effective_dt_sec"
            ),
            "joint_velocity_limit_max_dt_sec": self._latest_wrench_context.get("joint_velocity_limit_max_dt_sec"),
            "joint_velocity_limit_requested_delta_deg": self._latest_wrench_context.get(
                "joint_velocity_limit_requested_delta_deg"
            ),
            "joint_velocity_limit_delta_deg": self._latest_wrench_context.get("joint_velocity_limit_delta_deg"),
            "joint_velocity_limit_max_step_deg": self._latest_wrench_context.get(
                "joint_velocity_limit_max_step_deg"
            ),
            "joint_velocity_limit_ratio": self._latest_wrench_context.get("joint_velocity_limit_ratio"),
            "joint_safety": self._latest_wrench_context.get("joint_safety"),
            "reference_to_command_linear_m": reference_to_command_linear,
            "reference_to_command_angular_rotvec_rad": reference_to_command_angular,
            "admittance_target_linear_norm_m": admittance_target_linear_norm,
            "admittance_target_angular_norm_rad": admittance_target_angular_norm,
            "admittance_offset_linear_norm_m": admittance_offset_linear_norm,
            "admittance_offset_angular_norm_rad": admittance_offset_angular_norm,
            "commanded_offset_linear_norm_m": commanded_offset_linear_norm,
            "commanded_offset_angular_norm_rad": commanded_offset_angular_norm,
            "offset_to_command_linear_error_m": offset_to_command_linear_error,
            "offset_to_command_angular_error_rad": offset_to_command_angular_error,
            "offset_to_command_linear_fraction": _fraction_or_none(
                commanded_offset_linear_norm,
                admittance_offset_linear_norm,
            ),
            "offset_to_command_angular_fraction": _fraction_or_none(
                commanded_offset_angular_norm,
                admittance_offset_angular_norm,
            ),
            "target_to_command_linear_error_m": target_to_command_linear_error,
            "target_to_command_angular_error_rad": target_to_command_angular_error,
            "target_to_command_linear_fraction": _fraction_or_none(
                commanded_offset_linear_norm,
                admittance_target_linear_norm,
            ),
            "target_to_command_angular_fraction": _fraction_or_none(
                commanded_offset_angular_norm,
                admittance_target_angular_norm,
            ),
            "reference_to_measured_linear_m": reference_to_measured_linear,
            "reference_to_measured_angular_rotvec_rad": reference_to_measured_angular,
            "measured_offset_linear_norm_m": (
                None if reference_to_measured_linear is None else norm(reference_to_measured_linear)
            ),
            "measured_offset_angular_norm_rad": (
                None if reference_to_measured_angular is None else norm(reference_to_measured_angular)
            ),
            "command_to_measured_linear_m": command_to_measured_linear,
            "command_to_measured_angular_rotvec_rad": command_to_measured_angular,
            "command_to_measured_linear_norm_m": (
                None if command_to_measured_linear is None else norm(command_to_measured_linear)
            ),
            "command_to_measured_angular_norm_rad": (
                None if command_to_measured_angular is None else norm(command_to_measured_angular)
            ),
            "measured_to_command_linear_m": measured_to_command_linear,
            "measured_to_command_angular_rotvec_rad": measured_to_command_angular,
            "command_to_measured_joint_deg": command_to_measured_joint,
            "measured_to_command_joint_deg": measured_to_command_joint,
            "measured_velocity_dt_sec": measured_velocity_dt_sec,
            "measured_pose_velocity_linear_m_s": measured_pose_velocity_linear,
            "measured_pose_velocity_angular_rad_s": measured_pose_velocity_angular,
            "measured_joint_velocity_deg_s": measured_joint_velocity,
            "tracking_error_closing_linear_m_s": _directional_speed(
                measured_pose_velocity_linear,
                measured_to_command_linear,
            ),
            "tracking_error_closing_angular_rad_s": _directional_speed(
                measured_pose_velocity_angular,
                measured_to_command_angular,
            ),
            "tracking_error_closing_joint_deg_s": _directional_speed(
                measured_joint_velocity,
                measured_to_command_joint,
            ),
            "ik_method": (
                None
                if self._latest_wrench_context.get("ik_method") is None
                else self._latest_wrench_context["ik_method"][0]
            ),
            "ik_backoff_scale": (
                None
                if self._latest_wrench_context.get("ik_backoff_scale") is None
                else self._latest_wrench_context["ik_backoff_scale"][0]
            ),
            "ik_reference_backoff_scale": (
                None
                if self._latest_wrench_context.get("ik_reference_backoff_scale") is None
                else self._latest_wrench_context["ik_reference_backoff_scale"][0]
            ),
            "ik_command_step_backoff_scale": (
                None
                if self._latest_wrench_context.get("ik_command_step_backoff_scale") is None
                else self._latest_wrench_context["ik_command_step_backoff_scale"][0]
            ),
            "ik_command_step_candidate_deg": (
                None
                if self._latest_wrench_context.get("ik_command_step_candidate_deg") is None
                else self._latest_wrench_context["ik_command_step_candidate_deg"][0]
            ),
            "ik_command_step_allowed_deg": (
                None
                if self._latest_wrench_context.get("ik_command_step_allowed_deg") is None
                else self._latest_wrench_context["ik_command_step_allowed_deg"][0]
            ),
            "ik_command_step_applied_deg": (
                None
                if self._latest_wrench_context.get("ik_command_step_applied_deg") is None
                else self._latest_wrench_context["ik_command_step_applied_deg"][0]
            ),
            "ik_resolver_tolerance_fraction": (
                None
                if self._latest_wrench_context.get("ik_resolver_tolerance_fraction") is None
                else self._latest_wrench_context["ik_resolver_tolerance_fraction"][0]
            ),
            "ik_timing": ik_timing,
            "singularity": singularity_metadata,
            "joint_limit_margin": joint_limit_margin_metadata,
            "ik_metadata": ik_metadata,
            "pose_tracking_linear_m": pose_delta[0],
            "pose_tracking_angular_rad": pose_delta[1],
        }
        controller_timing = timing_payload(state_read_performed=state_read_performed)
        if controller_timing is not None:
            record["rm65_controller_timing"] = controller_timing
        return record

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

    def _sync_admittance_to_realized_pose_offset(
        self,
        *,
        reference_pose: Iterable[float],
        command_pose: Iterable[float],
        requested_offset: Iterable[float],
        require_active_wrench: bool = True,
    ) -> list[float]:
        requested = finite_list(requested_offset, expected_len=6, label="requested_offset")
        realized = _pose_offset_from_reference(
            reference_pose=reference_pose,
            command_pose=command_pose,
        )
        realized[:3] = clamp_norm(realized[:3], float(self.config.max_linear_offset_m))
        if float(self.config.max_angular_offset_rad) > 0.0:
            realized[3:] = clamp_norm(realized[3:], float(self.config.max_angular_offset_rad))
        else:
            realized[3:] = [0.0, 0.0, 0.0]
        context = self._latest_wrench_context if isinstance(self._latest_wrench_context, dict) else {}
        active_wrench = _coerce_float_list(context.get("active_wrench"), expected_len=6)
        target_offset = _coerce_float_list(context.get("target_offset_after_offset_limit"), expected_len=6)
        if target_offset is None:
            target_offset = _coerce_float_list(context.get("target_offset"), expected_len=6)
        realized_before_guard = list(realized)
        axis_labels = ("x", "y", "z", "rx", "ry", "rz")
        axis_thresholds = (1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9)
        active_axes = [False] * 6
        if not require_active_wrench:
            active_axes = [
                abs(requested[index]) > axis_thresholds[index]
                for index in range(6)
            ]
        elif active_wrench is not None:
            offset_for_activity = target_offset if target_offset is not None else requested
            active_axes = [
                abs(active_wrench[index]) > 1e-12
                and abs(offset_for_activity[index]) > axis_thresholds[index]
                for index in range(6)
            ]

        for index, active in enumerate(active_axes):
            if not active:
                realized[index] = requested[index]

        guarded_axes: list[str] = []
        if active_wrench is not None and target_offset is not None:
            for index, axis in enumerate(("x", "y", "z")):
                if not active_axes[index]:
                    continue
                if target_offset[index] * realized[index] >= -1e-9:
                    continue
                replacement = requested[index]
                if target_offset[index] * replacement < -1e-9:
                    replacement = 0.0
                realized[index] = replacement
                guarded_axes.append(axis)
        previous = None if self._admittance is None else list(self._admittance.offset)
        linear_error = _vector_error_norm(requested[:3], realized[:3])
        angular_error = _vector_error_norm(requested[3:], realized[3:])
        active_wrench_norm = 0.0 if active_wrench is None else norm(active_wrench)
        returning_to_neutral = bool(
            active_wrench_norm <= 1e-12
            and (
                context.get("return_applied_axes")
                or context.get("linear_return_velocity_limit_applied")
                or context.get("angular_return_velocity_limit_applied")
            )
        )
        if returning_to_neutral:
            active_axes = [False] * 6
        active_for_antiwindup = (
            any(active_axes)
            and not returning_to_neutral
        )
        antiwindup_applied = bool(
            active_for_antiwindup
            and (linear_error > 1e-6 or angular_error > 1e-6)
        )
        skip_reason = ""
        if not active_for_antiwindup:
            skip_reason = (
                "returning_to_neutral"
                if returning_to_neutral
                else (
                    "active_wrench_zero"
                    if require_active_wrench and active_wrench_norm <= 1e-12
                    else "no_active_offset_axis"
                )
            )
        if antiwindup_applied and self._admittance is not None:
            self._admittance.offset = list(realized)
        if self._latest_wrench_context is not None:
            if previous is not None:
                self._latest_wrench_context["admittance_offset_before_antiwindup"] = list(previous)
            self._latest_wrench_context["requested_offset_before_command_realization"] = list(requested)
            if antiwindup_applied:
                self._latest_wrench_context["applied_offset"] = list(realized)
                self._latest_wrench_context["offset"] = list(realized)
                self._latest_wrench_context["applied_offset_source"] = (
                    "guarded_final_command_pose" if guarded_axes else "final_command_pose"
                )
            else:
                self._latest_wrench_context["applied_offset_source"] = (
                    "requested_offset:antiwindup_guarded"
                    if guarded_axes
                    else ("requested_offset" if not skip_reason else f"requested_offset:{skip_reason}")
                )
            self._latest_wrench_context["admittance_offset_antiwindup_applied"] = antiwindup_applied
            self._latest_wrench_context["admittance_offset_antiwindup_skip_reason"] = skip_reason
            self._latest_wrench_context["admittance_offset_antiwindup_guarded_axes"] = list(guarded_axes)
            self._latest_wrench_context["admittance_offset_antiwindup_active_axes"] = [
                axis
                for axis, active in zip(axis_labels, active_axes)
                if active
            ]
            if guarded_axes:
                self._latest_wrench_context["admittance_offset_realized_before_guard"] = list(realized_before_guard)
                self._latest_wrench_context["admittance_offset_antiwindup_guard_reason"] = (
                    "realized_linear_offset_opposes_active_target"
                )
            self._latest_wrench_context["applied_offset_realization_linear_error_m"] = float(linear_error)
            self._latest_wrench_context["applied_offset_realization_angular_error_rad"] = float(angular_error)
        return realized

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

    def _direct_joint_hint_force_guard_state(self) -> tuple[bool, dict[str, object]]:
        enabled = bool(
            self.config.direct_joint_hint_force_guard_enabled
            and self.config.pose_command_transport == "joint"
        )
        metadata: dict[str, object] = {
            "enabled": enabled,
            "active": False,
            "force_threshold_n": float(self.config.direct_joint_hint_force_guard_force_n),
            "torque_threshold_nm": float(self.config.direct_joint_hint_force_guard_torque_nm),
            "force_norm_n": None,
            "torque_norm_nm": None,
        }
        if not enabled:
            return False, metadata
        context = self._latest_wrench_context if isinstance(self._latest_wrench_context, dict) else {}
        wrench = _coerce_float_list(context.get("tare_corrected"), expected_len=6)
        if wrench is None:
            metadata["reason"] = "wrench_unavailable"
            return False, metadata
        force_norm = norm(wrench[:3])
        torque_norm = norm(wrench[3:])
        force_threshold = float(self.config.direct_joint_hint_force_guard_force_n)
        torque_threshold = float(self.config.direct_joint_hint_force_guard_torque_nm)
        force_active = bool(force_threshold > 0.0 and force_norm >= force_threshold)
        torque_active = bool(torque_threshold > 0.0 and torque_norm >= torque_threshold)
        active = bool(force_active or torque_active)
        metadata.update(
            {
                "active": active,
                "force_norm_n": float(force_norm),
                "torque_norm_nm": float(torque_norm),
                "force_active": force_active,
                "torque_active": torque_active,
            }
        )
        return active, metadata

    def _command_from_reference(
        self,
        reference: list[float],
        offset: list[float],
        *,
        preferred_joint: list[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        assert self._ik_seed is not None
        preferred = (
            None
            if preferred_joint is None
            else finite_list(preferred_joint, expected_len=6, label="preferred_joint")
        )
        offset_values = finite_list(offset, expected_len=6, label="pose_offset")
        linear_offset_m = norm(offset_values[:3])
        angular_offset_rad = norm(offset_values[3:])
        force_guard_active, force_guard_metadata = self._direct_joint_hint_force_guard_state()
        offset_within_direct_hint = bool(
            linear_offset_m <= float(self.config.direct_joint_hint_max_linear_offset_m)
            and angular_offset_rad <= float(self.config.direct_joint_hint_max_angular_offset_rad)
        )
        direct_joint_hint = bool(
            preferred is not None
            and (offset_within_direct_hint or force_guard_active)
        )
        if direct_joint_hint:
            command_pose = normalize_pose(reference)
            command_joint = list(preferred)
            applied_offset = [0.0] * 6
            if (
                force_guard_active
                and bool(self.config.direct_joint_hint_force_guard_reset_admittance)
                and self._admittance is not None
            ):
                self._admittance.reset()
            if self._latest_wrench_context is not None:
                self._latest_wrench_context["applied_offset"] = list(applied_offset)
                self._latest_wrench_context["requested_offset_before_command_realization"] = list(offset_values)
                self._latest_wrench_context["applied_offset_source"] = "direct_joint_hint"
                self._latest_wrench_context["admittance_offset_antiwindup_applied"] = False
                self._latest_wrench_context["admittance_offset_antiwindup_skip_reason"] = "direct_joint_hint"
                if force_guard_active and bool(self.config.direct_joint_hint_force_guard_reset_admittance):
                    self._latest_wrench_context["offset"] = list(applied_offset)
                self._latest_wrench_context["ik_method"] = ["dual_arm_joint_hint"]
                self._latest_wrench_context["ik_metadata"] = self._attach_ik_timing_metadata({
                    "status": "pass",
                    "method": "dual_arm_joint_hint",
                    "source": "retargeter_solver_output_qpos",
                    "direct_joint_hint_reason": (
                        "force_guard" if force_guard_active and not offset_within_direct_hint else "offset_threshold"
                    ),
                    "seed_joint_deg": list(preferred),
                    "direct_joint_hint_linear_offset_m": float(linear_offset_m),
                    "direct_joint_hint_angular_offset_rad": float(angular_offset_rad),
                    "direct_joint_hint_max_linear_offset_m": float(
                        self.config.direct_joint_hint_max_linear_offset_m
                    ),
                    "direct_joint_hint_max_angular_offset_rad": float(
                        self.config.direct_joint_hint_max_angular_offset_rad
                    ),
                    "direct_joint_hint_offset_ignored": bool(
                        linear_offset_m > 1e-9 or angular_offset_rad > 1e-9
                    ),
                    "direct_joint_hint_force_guard": dict(force_guard_metadata),
                    "direct_joint_hint_force_guard_reset_admittance": bool(
                        force_guard_active
                        and self.config.direct_joint_hint_force_guard_reset_admittance
                    ),
                }, source="retargeter_solver_output_qpos", elapsed_ms=0.0)
            try:
                command_pose, command_joint = self._apply_joint_step_policy(
                    command_pose=command_pose,
                    command_joint=command_joint,
                )
            except RuntimeError as exc:
                held = self._hold_last_safe_command_after_ik_failure(
                    reason="joint_hint_step_limit",
                    message=str(exc),
                    target_pose=command_pose,
                    seed_joint=preferred,
                )
                if held is not None:
                    self._ik_seed = list(held[1])
                    return held
                raise
            self._ik_seed = list(command_joint)
            return command_pose, command_joint

        try:
            command_solver = (
                self._command_from_pose_with_resolver
                if self._ik_resolver is not None
                else self._command_from_pose_with_ik_backoff
            )
            command_pose, command_joint, seed_joint, backoff_scale, applied_offset, code = (
                command_solver(
                    reference_pose=reference,
                    offset=offset,
                    seed_joint_override=preferred,
                )
            )
            if self._latest_wrench_context is not None:
                self._latest_wrench_context["applied_offset"] = list(applied_offset)
        except RuntimeError as exc:
            held = self._hold_last_safe_command_after_ik_failure(
                reason="ik_exception",
                message=str(exc),
                target_pose=reference,
                seed_joint=self._ik_seed,
            )
            if held is not None:
                self._ik_seed = list(held[1])
                return held
            raise
        if int(code) != 0:
            source = "generic_ik_resolver" if self._ik_resolver is not None else "rm_sdk_legacy"
            message = self._ik_failure_message(
                code=code,
                seed_joint=seed_joint,
                target_pose=command_pose,
                reference=reference,
                offset=offset,
            )
            held = self._hold_last_safe_command_after_ik_failure(
                reason="generic_resolver_return_code" if source == "generic_ik_resolver" else "ik_return_code",
                message=message,
                target_pose=command_pose,
                seed_joint=seed_joint,
                code=code,
            )
            if held is not None:
                self._ik_seed = list(held[1])
                return held
            self._raise_ik_failure(
                code=code,
                seed_joint=seed_joint,
                target_pose=command_pose,
                reference=reference,
                offset=offset,
            )
        command_joint = finite_list(command_joint[:6], expected_len=6, label="ik_joint_deg")
        try:
            command_pose, command_joint = self._apply_joint_step_policy(
                command_pose=command_pose,
                command_joint=command_joint,
            )
        except RuntimeError as exc:
            held = self._hold_last_safe_command_after_ik_failure(
                reason="command_step_limit",
                message=str(exc),
                target_pose=command_pose,
                seed_joint=seed_joint,
            )
            if held is not None:
                self._ik_seed = list(held[1])
                return held
            raise
        self._sync_admittance_to_realized_pose_offset(
            reference_pose=reference,
            command_pose=command_pose,
            requested_offset=applied_offset,
        )
        self._ik_seed = list(command_joint)
        return command_pose, command_joint

    def _send(self, command_pose: list[float], command_joint: list[float]) -> None:
        assert self.robot is not None
        if self.config.pose_command_transport == "pose":
            _require_zero(
                self.robot.Movep_CANFD(
                    command_pose,
                    follow=False,
                    trajectory_mode=self.config.trajectory_mode,
                    radio=self.config.radio,
                ),
                f"{self.side} Movep_CANFD",
            )
            return
        _require_zero(
            self.robot.Movej_CANFD(
                command_joint,
                follow=False,
                trajectory_mode=self.config.trajectory_mode,
                radio=self.config.radio,
            ),
            f"{self.side} Movej_CANFD",
        )


class RM65JointImpedanceController(RM65BaseImpedanceController):
    reference_kind = "joint"

    def _initial_reference(self, state: dict[str, list[float]]) -> list[float]:
        return list(state["joint_deg"])

    def _command_from_reference(
        self,
        reference: list[float],
        offset: list[float],
        *,
        preferred_joint: list[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        assert self._ik_seed is not None
        del preferred_joint
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
        command_pose, command_joint = self._apply_joint_step_policy(
            command_pose=command_pose,
            command_joint=command_joint,
        )
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
