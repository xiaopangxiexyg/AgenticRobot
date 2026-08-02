from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.runtime.lifecycle import (
    LifecycleFailure,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)

from hardware.rm65.calibration_profiles import (
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    is_position_endpoint_profile,
    tcp_endpoint_delta,
    tcp_safety_warnings,
)
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


_IMPEDANCE_CONFIG_DEBUG_KEYS = (
    "tool_name",
    "control_hz",
    "action_hz",
    "hard_stop_force_n",
    "hard_stop_torque_nm",
    "force_source",
    "axes",
    "force_sign",
    "angular_force_sign",
    "angular_axis_signs",
    "angular_axis_velocity_scales",
    "lowpass_alpha",
    "force_deadband_n",
    "torque_deadband_nm",
    "return_force_deadband_n",
    "return_torque_deadband_nm",
    "effective_return_force_deadband_n",
    "effective_return_torque_deadband_nm",
    "linear_gain_m_s_per_n",
    "angular_gain_rad_s_per_nm",
    "linear_damping_n_s_per_m",
    "angular_damping_nm_s_per_rad",
    "linear_compliance_m_per_n",
    "angular_compliance_rad_per_nm",
    "max_linear_velocity_m_s",
    "max_angular_velocity_rad_s",
    "max_linear_return_velocity_m_s",
    "max_angular_return_velocity_rad_s",
    "max_linear_offset_m",
    "max_angular_offset_rad",
    "return_rate_s",
    "canfd_follow",
    "follow_qpos_alpha",
    "canfd_follow_stream_hz",
    "canfd_follow_warning_ratio",
    "passthrough_position_control",
    "trajectory_mode",
    "radio",
    "pose_command_transport",
    "max_command_joint_step_deg",
    "max_command_joint_velocity_deg_s",
    "joint_rate_limiter_enabled",
    "joint_velocity_limit_profile",
    "joint_rate_limit_use_nominal_dt",
    "admittance_use_nominal_dt",
    "joint_operating_margin_deg",
    "observed_joint_tolerance_deg",
    "ik_safety_profile",
    "ik_resolver_local_step_max_iterations",
    "ik_resolver_local_step_trigger_ratio",
    "ik_resolver_local_step_max_position_worsen_m",
    "ik_resolver_local_step_max_orientation_worsen_deg",
    "ik_resolver_local_step_component_progress_max_position_worsen_m",
    "ik_resolver_local_step_dynamic_worsen_enabled",
    "ik_hard_deadline_enabled",
    "ik_hard_deadline_ms",
    "deployment_dual_rate_ik_enabled",
    "deployment_full_ik_hz",
    "deployment_full_ik_deadline_ms",
    "deployment_local_ik_deadline_ms",
    "deployment_full_ik_preempts_local",
    "direct_joint_hint_max_linear_offset_m",
    "direct_joint_hint_max_angular_offset_rad",
    "direct_joint_hint_force_guard_enabled",
    "direct_joint_hint_force_guard_force_n",
    "direct_joint_hint_force_guard_torque_nm",
    "direct_joint_hint_force_guard_reset_admittance",
    "hold_last_safe_on_ik_failure",
    "debug_timing",
    "state_read_every",
    "tcp_pose_flange_m_rad",
    "tcp_confirmed",
    "payload_mass_kg",
    "payload_com_tool_m",
)
_RM65_IMPEDANCE_MAX_CONTROL_HZ = 40.0
_RM65_CANFD_FOLLOW_DEFAULT_STATE_READ_EVERY = 4
_RM65_PASSTHROUGH_POSITION_MAX_COMMAND_STEP_DEG = 1.0e9


def _is_force_hard_stop_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        text = str(current)
        if (
            "hard_stop_force_n" in text
            or "hard_stop_torque_nm" in text
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _debug_config_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_debug_config_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _debug_config_value(item) for key, item in value.items()}
    return str(value)


def _coerce_float_vector(value: object, *, expected_len: int) -> list[float] | None:
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if out.shape != (expected_len,) or not np.all(np.isfinite(out)):
        return None
    return [float(item) for item in out.tolist()]


def _tool_frame_debug_snapshot(frame: object) -> dict[str, object]:
    out: dict[str, object] = {"name": _tool_name_from_frame_result(frame)}
    payload = frame[1] if isinstance(frame, tuple) and len(frame) >= 2 else frame
    if isinstance(payload, Mapping):
        pose = _coerce_float_vector(payload.get("pose"), expected_len=6)
        if pose is not None:
            out["pose_flange_m_rad"] = pose
            out["endpoint_delta"] = tcp_endpoint_delta(pose)
        try:
            mass = float(payload.get("payload"))
        except (TypeError, ValueError):
            mass = None
        if mass is not None and np.isfinite(mass):
            out["payload_mass_kg"] = mass
        com = []
        for key in ("x", "y", "z"):
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                com = []
                break
            com.append(value)
        if len(com) == 3 and all(np.isfinite(value) for value in com):
            if np.linalg.norm(np.asarray(com, dtype=float)) > 1.0:
                com = [value * 0.001 for value in com]
            out["payload_com_tool_m"] = [float(value) for value in com]
    return out


def _profile_tcp_debug_snapshot(profile: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(profile, Mapping):
        return None
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), Mapping) else {}
    pose = _coerce_float_vector(tcp.get("pose_flange_m_rad") if isinstance(tcp, Mapping) else None, expected_len=6)
    if pose is None:
        return None
    payload = profile.get("payload") if isinstance(profile.get("payload"), Mapping) else {}
    out: dict[str, object] = {
        "pose_flange_m_rad": pose,
        "confirmed": bool(tcp.get("confirmed")) if isinstance(tcp, Mapping) else False,
        "endpoint_delta": tcp_endpoint_delta(pose),
        "frame_name": str(tcp.get("frame_name") or "") if isinstance(tcp, Mapping) else "",
    }
    if isinstance(payload, Mapping):
        try:
            mass = float(payload.get("mass_kg"))
        except (TypeError, ValueError):
            mass = None
        if mass is not None and np.isfinite(mass):
            out["payload_mass_kg"] = mass
        com = _coerce_float_vector(payload.get("com_tool_m"), expected_len=3)
        if com is not None:
            out["payload_com_tool_m"] = com
        warnings = payload.get("warnings")
        if isinstance(warnings, (list, tuple)):
            out["payload_warnings"] = [str(item) for item in warnings]
    return out


def _impedance_config_debug_snapshot(
    *,
    profile: Mapping[str, Any],
    mode: str,
    config: RM65ImpedanceConfig,
    overrides: Mapping[str, Any],
    requested_control_hz: float,
    effective_control_hz: float,
    requested_action_hz: float,
) -> dict[str, object]:
    section = _profile_impedance_section(profile, mode)
    out: dict[str, object] = {
        "profile_path": str(profile.get("_profile_path") or ""),
        "profile_name": str(profile.get("name") or ""),
        "profile_arm": str(profile.get("arm") or ""),
        "profile_tool_name": str(profile.get("tool_name") or ""),
        "mode": str(mode),
        "requested_control_hz": float(requested_control_hz),
        "effective_control_hz": float(effective_control_hz),
        "control_hz_clamped": bool(abs(float(requested_control_hz) - float(effective_control_hz)) > 1e-9),
        "requested_action_hz": float(requested_action_hz),
        "profile_section_keys": sorted(str(key) for key in section.keys()),
        "override_keys": sorted(str(key) for key in overrides.keys()),
    }
    for key in _IMPEDANCE_CONFIG_DEBUG_KEYS:
        if hasattr(config, key):
            out[key] = _debug_config_value(getattr(config, key))
    return out


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


def _require_rm65_success(result: object, label: str) -> None:
    code, text = _parse_rm65_result(result)
    if code != 0:
        raise RuntimeError(f"{label} failed with return code {text}")


def _profile_tool_frame_params(profile: Mapping[str, Any]) -> tuple[str, list[float], float, list[float]]:
    tool_name = str(profile.get("tool_name") or "").strip()
    if not tool_name:
        raise RuntimeError("profile tool_name is empty; cannot sync RM65 tool frame")
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), Mapping) else {}
    payload = profile.get("payload") if isinstance(profile.get("payload"), Mapping) else {}
    try:
        tcp_pose = np.asarray(tcp.get("pose_flange_m_rad"), dtype=float).reshape(-1)
    except Exception as exc:
        raise RuntimeError("profile tcp.pose_flange_m_rad is not a numeric 6D pose") from exc
    if tcp_pose.shape != (6,) or not np.all(np.isfinite(tcp_pose)):
        raise RuntimeError("profile tcp.pose_flange_m_rad must contain 6 finite values")
    try:
        payload_mass = float(payload.get("mass_kg"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("profile payload.mass_kg is missing or non-numeric") from exc
    if not np.isfinite(payload_mass) or payload_mass < 0.0:
        raise RuntimeError("profile payload.mass_kg must be a finite non-negative value")
    try:
        payload_com = np.asarray(payload.get("com_tool_m"), dtype=float).reshape(-1)
    except Exception as exc:
        raise RuntimeError("profile payload.com_tool_m is missing or non-numeric") from exc
    if payload_com.shape != (3,) or not np.all(np.isfinite(payload_com)):
        raise RuntimeError("profile payload.com_tool_m must contain 3 finite values")
    return (
        tool_name,
        [float(value) for value in tcp_pose],
        float(payload_mass),
        [float(value) for value in payload_com],
    )


@dataclass
class RM65ImpedanceArmBackendConfig:
    arm_ip: str
    side: str
    effector_name: str | None = None
    connect_hardware: bool = True
    # Deployment-only live observation shadow.  It keeps RM65 feedback and
    # local FK available while forbidding tool-frame writes, force-control
    # startup, periodic output, stop commands, and command dispatch.
    tool_profile: str | Path | None = None
    auto_load_tool_profile: bool = True
    strict_tool_profile: bool = True
    auto_sync_tool_frame: bool = True
    auto_create_tool_frame: bool = True
    impedance_mode: str = "pose"
    force_tare_sec: float = 3.0
    control_dt: float = 1.0 / 30.0
    action_hz: float | None = None
    max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    actual_send_dt_alpha: float = 0.25
    actual_send_dt_min_scale: float = 1.0
    actual_send_dt_max_scale: float = 1.0
    actual_send_dt_stall_reset_sec: float = 0.25
    # A delayed force-control frame must not enlarge the outer IK step and
    # create a catch-up motion. Position backends retain their existing policy.
    scale_outer_execution_step_by_actual_dt: bool = False
    debug_timing: bool = False
    impedance_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Kept after the existing public fields so legacy positional construction
    # keeps its old meaning.
    observation_only: bool = False

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
        self._active_impedance_config: dict[str, object] | None = None
        self._active_tool_frame_snapshot: dict[str, object] | None = None
        self._profile_tcp_snapshot: dict[str, object] | None = None
        self._endpoint_to_tool_tcp = np.eye(4, dtype=float)
        self._latest_target_pose6: list[float] | None = None
        self._latest_target_joint_deg: list[float] | None = None
        self._latest_target_sequence: int | None = None
        self._command_sequence = 0
        self._last_command: EffectorCommand | None = None
        self._last_command_timestamp: float | None = None
        self._last_step_record: dict[str, object] | None = None
        self._recent_step_records: list[dict[str, object]] = []
        self._last_measured_joint_deg: list[float] | None = None
        self._last_measured_pose: list[float] | None = None
        self._last_measurement_timestamp: float | None = None
        self._last_measurement_sequence: int | None = None
        self._observation_state: dict[str, list[float]] | None = None
        self._observation_raw_joint_deg: list[float] | None = None
        self._observation_read_timestamp: float | None = None
        self._observation_read_sequence = 0
        self._observation_read_success = False
        self._observation_read_error: str | None = None
        self._last_fault: str | None = None
        self._last_send_result_code: int | None = None
        self._last_send_result_text: str | None = None
        # Deployment-only inference gap lease.  When the policy queue is
        # empty, the worker freezes a measured joint snapshot instead of
        # continuing the previous Cartesian interpolation.  A fresh command
        # releases this lease atomically under ``_state_lock``.
        self._policy_inference_soft_hold = False
        self._policy_inference_soft_hold_joint_deg: list[float] | None = None
        self._policy_inference_soft_hold_reason = ""
        self._policy_inference_soft_hold_timestamp: float | None = None
        self._enabled = False
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._emergency_stopped = False
        self._force_hard_stop_callback: Callable[[str, str], None] | None = None
        self._loop_stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._last_send_time: float | None = None
        self._smoothed_send_dt = float(config.control_dt)
        self._last_effective_control_dt = float(config.control_dt)
        self._last_effective_step_scale = 1.0
        self._last_send_command_timing: dict[str, object] | None = None
        self._last_impedance_loop_timing: dict[str, object] | None = None
        self._exact_qpos_transit_operation_id: str | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _active_tcp_pose_flange_m_rad(self) -> tuple[float, ...]:
        """Return the TCP that the active impedance controller actually uses.

        Replay preflight compares this value with the dataset tool profile.  A
        position backend stores the value directly on its backend config, while
        the impedance backend resolves it when the calibration profile is
        loaded.  Exposing the same getter prevents replay from silently falling
        back to the zero arm-endpoint TCP for an active impedance tool.
        """

        with self._state_lock:
            controller = self._controller
            profile = self._profile
        candidates: list[object] = []
        if controller is not None:
            candidates.append(
                getattr(controller.config, "tcp_pose_flange_m_rad", None)
            )
        if isinstance(profile, Mapping):
            tcp = profile.get("tcp")
            if isinstance(tcp, Mapping):
                candidates.append(tcp.get("pose_flange_m_rad"))
        candidates.append(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD)
        for candidate in candidates:
            pose = _coerce_float_vector(candidate, expected_len=6)
            if pose is not None:
                return tuple(float(value) for value in pose)
        raise RuntimeError(f"{self.name} active impedance TCP is unavailable.")

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
        self._validate_profile(self._profile, self._active_tool_name, check_active_tool=False)
        if not self.config.observation_only:
            self._active_tool_name = self._sync_profile_tool_frame(
                self._profile,
                self._active_tool_name,
            )
        self._active_tool_frame_snapshot = self._read_active_tool_frame_snapshot()
        self._validate_profile(self._profile, self._active_tool_name)
        tcp = self._profile.get("tcp") if isinstance(self._profile.get("tcp"), Mapping) else {}
        self._endpoint_to_tool_tcp = _pose6_to_matrix(tcp.get("pose_flange_m_rad", [0.0] * 6))
        tcp_metrics = tcp_endpoint_delta(tcp.get("pose_flange_m_rad", [0.0] * 6))
        print(
            f"{self.name}: impedance tool_tcp endpoint_delta="
            f"{tcp_metrics['translation_m']:.4f}m/{tcp_metrics['rotation_deg']:.2f}deg"
        )
        self._active_tool_profile_path = str(self._profile.get("_profile_path") or "")
        self._profile_tcp_snapshot = _profile_tcp_debug_snapshot(self._profile)
        impedance_config = self._build_impedance_config(self._profile)
        self._controller = RM65PoseImpedanceController(
            arm=self.config.side,
            config=impedance_config,
            robot=self.arm,
            left_ip=self.config.arm_ip,
            right_ip=self.config.arm_ip,
        )
        if self.config.observation_only:
            state = self.refresh_state_from_hardware()
            if not bool(state.get("hardware_read_success")):
                raise RuntimeError(
                    f"{self.name} observation-only startup did not obtain a trusted "
                    "RM65 state sample."
                )

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or self.arm is not None

    def set_force_hard_stop_callback(
        self,
        callback: Callable[[str, str], None] | None,
    ) -> None:
        """Install the deployment-owned callback for controller hard stops."""

        if callback is not None and not callable(callback):
            raise TypeError("force hard-stop callback must be callable or None")
        with self._state_lock:
            self._force_hard_stop_callback = callback

    def read_impedance_force_guard_data(self) -> dict[str, object]:
        """Return the latest validated force sample without another SDK read."""

        with self._state_lock:
            controller = self._controller
            record = (
                None
                if self._last_step_record is None
                else dict(self._last_step_record)
            )
            fault = self._last_fault
            emergency_stopped = bool(self._emergency_stopped)
        if controller is None:
            return {
                "sample": None,
                "sample_error": f"{self.name} impedance controller is unavailable",
            }
        if fault is not None:
            return {
                "sample": None,
                "backend_fault": fault,
            }
        if emergency_stopped:
            return {
                "sample": None,
                "backend_fault": f"{self.name} emergency_stop is latched",
            }
        if record is None:
            return {
                "sample": None,
                "sample_error": f"{self.name} has no completed impedance step",
            }
        sample_monotonic = record.get("backend_controller_step_end_monotonic")
        sample_wall = record.get("backend_controller_step_end_wall_time")
        try:
            force_norm = float(record.get("force_norm_n"))
            torque_norm = float(record.get("torque_norm_nm"))
            sample_monotonic = float(sample_monotonic)
            sample_wall = float(sample_wall)
            if not all(
                np.isfinite(value)
                for value in (force_norm, torque_norm, sample_monotonic, sample_wall)
            ):
                raise ValueError("force sample contains non-finite values")
        except (TypeError, ValueError) as exc:
            return {
                "sample": None,
                "sample_error": f"{self.name} force sample is invalid: {exc}",
            }
        config = controller.config
        return {
            "sample": {
                "side": self.config.side,
                "force_source": str(config.force_source),
                "force_norm_n": force_norm,
                "torque_norm_nm": torque_norm,
                "max_force_n": float(config.hard_stop_force_n),
                "max_torque_nm": float(config.hard_stop_torque_nm),
                "monotonic_time_sec": sample_monotonic,
                "wall_time_sec": sample_wall,
                "step": record.get("step"),
            }
        }

    def hold_motion(self, reason: str = "manual hold") -> None:
        del reason
        if self.config.observation_only:
            with self._state_lock:
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._last_send_result_code = 0
                self._last_send_result_text = "observation_only_hold"
            return
        # A force/collision stop is an explicit backend latch.  The outer
        # runtime may still consume the normal GUI Hold transition after it
        # has already issued Arm_Stop; do not send a second Slow_Stop here or
        # race the controller's failed worker teardown.  Runtime Home calls
        # ``manual_rearm_after_emergency_stop`` before enabling impedance
        # again.
        with self._state_lock:
            if self._emergency_stopped:
                self._last_send_result_code = 0
                self._last_send_result_text = "hold_skipped_emergency_latched"
                return
        self._release_exact_qpos_transit_best_effort(reason="hold")
        with self._send_lock:
            self._loop_stop_event.set()
            with self._state_lock:
                self._enabled = False
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._latest_target_pose6 = None
                self._latest_target_joint_deg = None
                self._latest_target_sequence = None
                self._command_sequence = 0
                self._last_command = None
                self._last_command_timestamp = None
                self._last_step_record = None
                self._recent_step_records = []
                self._last_send_command_timing = None
                self._last_impedance_loop_timing = None
                self._last_send_result_code = 0
                self._last_send_result_text = "hold"
            controller = self._controller
            robot = self.arm
            if robot is None and controller is not None:
                robot = controller.robot
            if controller is not None:
                try:
                    controller.stop_motion()
                except Exception as exc:  # noqa: BLE001
                    with self._state_lock:
                        self._last_send_result_code = -1
                        self._last_send_result_text = f"hold:stop_motion:{type(exc).__name__}: {exc}"
            elif robot is not None:
                stop_method = getattr(robot, "Arm_Slow_Stop", None)
                stop_label = "Arm_Slow_Stop"
                if stop_method is None:
                    stop_method = getattr(robot, "Arm_Stop", None)
                    stop_label = "Arm_Stop"
                if stop_method is not None:
                    stop_result = stop_method()
                    stop_code, stop_text = _parse_rm65_result(stop_result)
                    with self._state_lock:
                        self._last_send_result_code = stop_code
                        self._last_send_result_text = f"hold:{stop_label}:{stop_text}"
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.0)
            if not self._loop_thread.is_alive():
                self._loop_thread = None

    def request_policy_inference_soft_hold(
        self,
        reason: str = "policy_inference_wait",
    ) -> dict[str, object]:
        """Freeze the latest valid measured qpos during a synchronous policy gap.

        This is intentionally separate from :meth:`hold_motion`: it does not
        stop the impedance worker, does not issue a robot stop command, and
        keeps force sensing/hard-stop checks active.  The next accepted
        ``send_command`` releases the lease.
        """

        normalized_reason = str(reason or "policy_inference_wait")
        if self.config.observation_only or not self.config.connect_hardware:
            with self._state_lock:
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = normalized_reason
                self._policy_inference_soft_hold_timestamp = None
            return {
                "active": False,
                "status": "skipped",
                "reason": "observation_only_or_dry_run",
            }

        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(
                    f"{self.name} emergency_stop is latched; refusing inference soft hold."
                )
            if self._policy_inference_soft_hold:
                return {
                    "active": True,
                    "status": "already_active",
                    "reason": self._policy_inference_soft_hold_reason,
                    "joint_deg": (
                        None
                        if self._policy_inference_soft_hold_joint_deg is None
                        else list(self._policy_inference_soft_hold_joint_deg)
                    ),
                    "timestamp": self._policy_inference_soft_hold_timestamp,
                }

        state = self.get_state()
        if state.get("fault"):
            raise RuntimeError(
                f"{self.name} cannot enter inference soft hold while faulted: {state['fault']}"
            )
        if not bool(state.get("impedance_enabled", False)):
            raise RuntimeError(
                f"{self.name} cannot enter inference soft hold before impedance output is enabled."
            )
        if not bool(state.get("hardware_measurement_valid", False)):
            raise RuntimeError(
                f"{self.name} cannot enter inference soft hold without valid measured qpos."
            )
        current_qpos = _coerce_float_vector(state.get("current_qpos"), expected_len=6)
        if current_qpos is None:
            raise RuntimeError(
                f"{self.name} cannot enter inference soft hold without finite measured qpos."
            )
        measured_joint_deg = np.rad2deg(np.asarray(current_qpos, dtype=float)).astype(float).tolist()
        measured_at = state.get("hardware_measurement_timestamp")
        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(
                    f"{self.name} emergency_stop was latched while taking inference hold snapshot."
                )
            if self._policy_inference_soft_hold:
                return {
                    "active": True,
                    "status": "already_active",
                    "reason": self._policy_inference_soft_hold_reason,
                    "joint_deg": list(self._policy_inference_soft_hold_joint_deg or measured_joint_deg),
                    "timestamp": self._policy_inference_soft_hold_timestamp,
                }
            self._policy_inference_soft_hold = True
            self._policy_inference_soft_hold_joint_deg = list(measured_joint_deg)
            self._policy_inference_soft_hold_reason = normalized_reason
            self._policy_inference_soft_hold_timestamp = time.time()
            self._last_send_result_code = 0
            self._last_send_result_text = "policy_inference_soft_hold"
            return {
                "active": True,
                "status": "activated",
                "reason": normalized_reason,
                "joint_deg": list(measured_joint_deg),
                "timestamp": self._policy_inference_soft_hold_timestamp,
                "measurement_timestamp": measured_at,
            }

    def release_policy_inference_soft_hold(
        self,
        reason: str = "fresh_policy_action",
    ) -> None:
        """Release the inference-gap lease without changing the target."""

        with self._state_lock:
            was_active = bool(self._policy_inference_soft_hold)
            self._policy_inference_soft_hold = False
            self._policy_inference_soft_hold_joint_deg = None
            self._policy_inference_soft_hold_reason = ""
            self._policy_inference_soft_hold_timestamp = None
            if was_active and self._last_fault is None:
                self._last_send_result_text = str(reason or "fresh_policy_action")

    def reset_motion_state(self, qpos: object | None = None) -> None:
        qpos_array: np.ndarray | None = None
        if qpos is not None:
            qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
            if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
                raise ValueError(f"{self.name} reset_motion_state qpos must be a finite 6-vector.")
        with self._send_lock:
            with self._state_lock:
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._latest_target_pose6 = None
                self._latest_target_joint_deg = None
                self._latest_target_sequence = None
                self._command_sequence = 0
                self._last_command = None
                self._last_command_timestamp = None
                self._last_step_record = None
                self._recent_step_records = []
                self._last_send_command_timing = None
                self._last_impedance_loop_timing = None
                self._last_send_time = None
                self._smoothed_send_dt = float(self.config.control_dt)
                self._last_effective_control_dt = float(self.config.control_dt)
                self._last_effective_step_scale = 1.0
                if not self._enabled:
                    self._last_send_result_text = "reset_motion_state"
            controller = self._controller
            if controller is not None and qpos_array is not None:
                controller.reset_motion_state(qpos_array)

    def is_emergency_stopped(self) -> bool:
        """Return whether an emergency stop latch still blocks dispatch."""

        with self._state_lock:
            return bool(self._emergency_stopped)

    def manual_rearm_after_emergency_stop(
        self,
        qpos: object,
    ) -> dict[str, object]:
        """Clear a force/collision stop latch after an explicit fresh Home.

        The control loop never calls this method.  ``runtime Home`` is the
        operator acknowledgement that the arm was manually repositioned and
        that it is safe to rebuild the impedance controller state.  A failed
        controller/worker teardown keeps the latch set and therefore forces a
        process restart rather than silently resuming after an incomplete
        safety stop.
        """

        qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
            raise ValueError(
                f"{self.name} manual rearm qpos must be a finite 6-vector."
            )
        if self.config.observation_only:
            with self._state_lock:
                self._emergency_stopped = False
                self._last_fault = None
                self._last_send_result_code = 0
                self._last_send_result_text = "observation_only_manual_rearm"
            return {"rearmed": True, "observation_only": True}

        with self._state_lock:
            if self.arm is None or self._controller is None:
                raise RuntimeError(
                    f"{self.name} has no ready RM65 connection for manual re-arm."
                )
            if not self._emergency_stopped:
                return {"rearmed": False, "reason": "not_emergency_stopped"}
            worker = self._loop_thread
            self._loop_stop_event.set()

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
            if worker.is_alive():
                raise RuntimeError(
                    f"{self.name} impedance worker did not stop before manual re-arm."
                )

        # The worker normally called stop_motion while reporting its fault.
        # Repeat that local teardown only after the worker is joined; it is
        # idempotent for the RM65 controller and also stops a follow stream
        # that may have outlived the failed impedance tick.
        with self._send_lock:
            with self._state_lock:
                if not self._emergency_stopped:
                    return {"rearmed": False, "reason": "latch_cleared_concurrently"}
                controller = self._controller
            if controller is None:
                raise RuntimeError(
                    f"{self.name} impedance controller disappeared during manual re-arm."
                )
            try:
                controller.stop_motion()
                controller.reset_motion_state(qpos_array)
            except Exception:
                # Keep ``_emergency_stopped`` and the worker stop event set on
                # every failure.  The next Home attempt can retry, while no
                # command can leak through this partially rebuilt state.
                raise
            with self._state_lock:
                self._loop_thread = None
                self._loop_stop_event.clear()
                self._enabled = False
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._last_fault = None
                self._last_step_record = None
                self._recent_step_records = []
                self._latest_target_pose6 = None
                self._latest_target_joint_deg = None
                self._latest_target_sequence = None
                self._command_sequence = 0
                self._last_command = None
                self._last_command_timestamp = None
                self._last_send_time = None
                self._smoothed_send_dt = float(self.config.control_dt)
                self._last_effective_control_dt = float(self.config.control_dt)
                self._last_effective_step_scale = 1.0
                self._emergency_stopped = False
                self._last_send_result_code = 0
                self._last_send_result_text = "manual_rearm_ready"
        return {"rearmed": True}

    def set_command_action_dt_sec(self, action_dt_sec: float) -> None:
        value = float(action_dt_sec)
        if value <= 0.0:
            raise ValueError(f"{self.name} command action_dt must be positive.")
        requested_action_hz = 1.0 / value
        with self._send_lock:
            controller = self._controller
            effective_action_hz = requested_action_hz
            if controller is not None:
                effective_action_hz = min(
                    requested_action_hz,
                    float(controller.config.control_hz),
                )
                controller.config.action_hz = effective_action_hz
                duration = 1.0 / effective_action_hz
                reference = getattr(controller, "_reference", None)
                if reference is not None:
                    reference.default_duration = duration
                joint_hint_reference = getattr(controller, "_joint_hint_reference", None)
                if joint_hint_reference is not None:
                    joint_hint_reference.default_duration = duration
            with self._state_lock:
                self.config.action_hz = effective_action_hz
                if self._active_impedance_config is not None:
                    self._active_impedance_config["requested_action_hz"] = requested_action_hz
                    self._active_impedance_config["action_hz"] = effective_action_hz

    def enable_force_control(self) -> None:
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; force control is disabled."
            )
        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(
                    f"{self.name} emergency_stop is latched; "
                    "manual re-arm is required before enabling force control."
                )
        if self._enabled:
            return
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.0)
            if self._loop_thread.is_alive():
                raise RuntimeError(f"{self.name} previous pose impedance loop is still alive.")
            self._loop_thread = None
        if not self.config.connect_hardware:
            with self._state_lock:
                self._last_fault = None
                self._last_send_result_code = 0
                self._last_send_result_text = "dry_run"
                self._enabled = True
            return
        self.start()
        assert self._controller is not None
        with self._state_lock:
            self._last_fault = None
            self._last_send_result_code = None
            self._last_send_result_text = "starting"
            self._last_step_record = None
            self._recent_step_records = []
        self._controller.start()
        self._loop_stop_event.clear()
        with self._state_lock:
            self._enabled = True
        self._loop_thread = threading.Thread(
            target=self._impedance_loop,
            daemon=True,
            name=f"{self.name}_pose_impedance",
        )
        self._loop_thread.start()

    def force_control_status_text(self) -> str:
        with self._state_lock:
            if self._loop_thread is None:
                loop_text = "not_started"
            elif self._loop_thread.is_alive():
                loop_text = "alive"
            else:
                loop_text = "stopped"
            bits = [
                f"enabled={bool(self._enabled)}",
                f"loop={loop_text}",
                f"last_step={'yes' if self._last_step_record is not None else 'none'}",
            ]
            if self._last_fault:
                bits.append(f"last_fault={self._last_fault}")
            if self._last_send_result_text:
                bits.append(f"last_send={self._last_send_result_text}")
            return " ".join(bits)

    def force_control_warm_started(self) -> bool:
        controller = self._controller
        return bool(getattr(controller, "_started", False))

    @property
    def exact_qpos_transit_active(self) -> bool:
        with self._state_lock:
            return bool(self._exact_qpos_transit_operation_id)

    @property
    def exact_qpos_transit_operation_id(self) -> str:
        with self._state_lock:
            return str(self._exact_qpos_transit_operation_id or "")

    def begin_exact_qpos_transit(
        self,
        operation_id: str,
        *,
        max_force_n: float = 20.0,
        max_torque_nm: float = 5.0,
        max_record_age_sec: float = 0.25,
    ) -> None:
        """Arm a force-monitored, compliance-suspended qpos transition."""

        normalized = str(operation_id or "").strip()
        if not normalized:
            raise ValueError(f"{self.name} exact qpos transit requires operation_id")
        with self._send_lock:
            with self._state_lock:
                active = self._exact_qpos_transit_operation_id
                if active is not None:
                    if active == normalized:
                        return
                    raise RuntimeError(
                        f"{self.name} exact qpos transit {active!r} is already active"
                    )
                if not self._enabled or self._last_fault is not None:
                    raise RuntimeError(
                        f"{self.name} force control is not healthy: {self.force_control_status_text()}"
                    )
                controller = self._controller
                record = None if self._last_step_record is None else dict(self._last_step_record)
            if controller is None:
                raise RuntimeError(f"{self.name} impedance controller is unavailable")
            if self.config.connect_hardware:
                if record is None:
                    raise RuntimeError(
                        f"{self.name} has no fresh impedance/force sample; retry after the control loop is ready"
                    )
                sampled_at = record.get("backend_controller_step_end_wall_time")
                if sampled_at is None:
                    raise RuntimeError(f"{self.name} force sample timestamp is unavailable")
                age_sec = max(0.0, time.time() - float(sampled_at))
                if age_sec > max(0.01, float(max_record_age_sec)):
                    raise RuntimeError(
                        f"{self.name} force sample is stale ({age_sec:.3f}s)"
                    )
                force_n = float(record.get("force_norm_n", float("inf")))
                torque_nm = float(record.get("torque_norm_nm", float("inf")))
                if not np.isfinite(force_n) or not np.isfinite(torque_nm):
                    raise RuntimeError(f"{self.name} force sample is non-finite")
                if force_n >= float(max_force_n) or torque_nm >= float(max_torque_nm):
                    raise RuntimeError(
                        f"{self.name} refuses exact qpos transit under contact: "
                        f"force={force_n:.3f}N/{float(max_force_n):.3f}N "
                        f"torque={torque_nm:.3f}Nm/{float(max_torque_nm):.3f}Nm"
                    )
            controller.begin_exact_qpos_transit(normalized)
            with self._state_lock:
                self._exact_qpos_transit_operation_id = normalized
                self._last_send_result_text = "exact_qpos_transit_armed"

    def end_exact_qpos_transit(self, operation_id: str | None = None) -> bool:
        """Release the temporary exact-qpos authority without closing hardware."""

        with self._send_lock:
            with self._state_lock:
                active = self._exact_qpos_transit_operation_id
                controller = self._controller
            if active is None:
                return False
            expected = str(operation_id or "").strip()
            if expected and expected != active:
                raise RuntimeError(
                    f"{self.name} exact qpos transit id mismatch: {expected!r} != {active!r}"
                )
            if controller is not None:
                controller.end_exact_qpos_transit(active)
            with self._state_lock:
                self._exact_qpos_transit_operation_id = None
                if self._last_fault is None:
                    self._last_send_result_text = "exact_qpos_transit_released"
            return True

    def _release_exact_qpos_transit_best_effort(self, *, reason: str) -> None:
        try:
            self.end_exact_qpos_transit()
        except Exception as exc:  # noqa: BLE001 - stopping motion has priority.
            with self._state_lock:
                self._exact_qpos_transit_operation_id = None
                self._last_send_result_text = (
                    f"exact_qpos_release_failed:{reason}:{type(exc).__name__}: {exc}"
                )

    def warm_start_force_control(self) -> None:
        if self.config.observation_only:
            return
        if self.force_control_warm_started():
            return
        if not self.config.connect_hardware:
            return
        self.start()
        assert self._controller is not None
        self._controller.start()
        with self._state_lock:
            if not self._enabled:
                self._last_fault = None
                self._last_send_result_code = 0
                self._last_send_result_text = "warm_started"

    def wait_until_force_control_ready(self, *, timeout_sec: float = 1.0) -> dict[str, Any]:
        if not self.config.connect_hardware:
            return self.get_state()
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            state = self.get_state()
            if state.get("fault") or not bool(state.get("impedance_enabled")):
                raise RuntimeError(
                    f"{self.name} pose impedance failed to stay enabled: "
                    f"{self.force_control_status_text()}"
                )
            if state.get("last_impedance_record") is not None and state.get("send_result_code") == 0:
                return state
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{self.name} pose impedance did not report a successful control tick "
                    f"within {timeout_sec:.3f}s: {self.force_control_status_text()}"
                )
            time.sleep(0.005)

    def refresh_state_from_hardware(self) -> dict[str, Any]:
        """Refresh a live observation-only sample without starting impedance output.

        The ordinary impedance backend obtains measured state from its periodic
        controller step.  A raw-RTC shadow must not start that loop, because the
        loop owns force/admittance and can issue output.  The controller's
        private ``_read_state`` helper is deliberately reused here: it performs
        only ``Get_Current_Arm_State`` plus local normalization and does not
        activate force control or CANFD follow.
        """

        if not self.config.observation_only:
            return self.get_state()
        controller = self._controller
        if controller is None or self.arm is None:
            return self.get_state()
        timestamp = time.time()
        try:
            state = dict(controller._read_state())
            joint_deg = np.asarray(state.get("joint_deg"), dtype=float).reshape(6)
            pose = np.asarray(state.get("pose"), dtype=float).reshape(6)
            if not np.all(np.isfinite(joint_deg)) or not np.all(np.isfinite(pose)):
                raise ValueError("RM65 observation contains non-finite state")
            raw_joint = getattr(controller, "_last_raw_observed_joint_deg", None)
            with self._state_lock:
                self._observation_state = {
                    "joint_deg": [float(value) for value in joint_deg.tolist()],
                    "pose": [float(value) for value in pose.tolist()],
                }
                self._observation_raw_joint_deg = (
                    None
                    if raw_joint is None
                    else [float(value) for value in np.asarray(raw_joint, dtype=float).reshape(6).tolist()]
                )
                self._observation_read_timestamp = timestamp
                self._observation_read_sequence += 1
                self._observation_read_success = True
                self._observation_read_error = None
                self._last_fault = None
        except Exception as exc:  # noqa: BLE001 - state becomes invalid, never a command.
            with self._state_lock:
                self._observation_read_timestamp = timestamp
                self._observation_read_sequence += 1
                self._observation_read_success = False
                self._observation_read_error = f"{type(exc).__name__}: {exc}"
                self._last_fault = self._observation_read_error
        return self.get_state()

    def compute_tcp_pose(
        self,
        tcp_frame: str | None = None,
        *,
        qpos: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        frame = canonical_arm_controlled_frame(TOOL_TCP_FRAME if tcp_frame is None else tcp_frame)
        if frame != TOOL_TCP_FRAME:
            raise ValueError(f"{self.name} impedance backend only exposes {TOOL_TCP_FRAME!r} FK.")
        if qpos is None:
            state = self.get_state()
            pose = state.get("current_tcp_pose")
            if pose is None:
                raise RuntimeError(f"{self.name} current TCP pose is unavailable.")
            return as_pose_matrix(pose)
        controller = self._controller
        if controller is None:
            raise RuntimeError(f"{self.name} cannot compute TCP pose before impedance controller is initialized.")
        qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
            raise ValueError(f"{self.name} qpos must be a finite 6D vector.")
        pose6 = controller._forward_pose(np.rad2deg(qpos_array).astype(float).tolist())
        return _pose6_to_matrix(pose6)

    def _joint_vector_command_as_cartesian(self, command: EffectorCommand) -> EffectorCommand:
        qpos = np.asarray(command.value, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{self.name} joint_vector command must be a finite 6D vector.")
        pose = self.compute_tcp_pose(qpos=qpos)
        metadata = dict(command.metadata or {})
        metadata.setdefault("source", metadata.get("source", "joint_vector_fk"))
        metadata.setdefault("solver_output_qpos", qpos.copy())
        metadata.setdefault("action_pose_source", "joint_vector_fk")
        metadata.setdefault("target_pose_base", pose.copy())
        metadata.setdefault("target_pose_tcp_base", pose.copy())
        metadata.setdefault("target_pose_controlled_frame", TOOL_TCP_FRAME)
        metadata.setdefault("arm_controlled_frame", TOOL_TCP_FRAME)
        return EffectorCommand(
            effector_name=command.effector_name,
            side=command.side,
            command_type="cartesian_pose",
            value=pose,
            frame_name=command.frame_name or "arm_base",
            metadata=metadata,
        )

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for effector '{command.effector_name}'.")
        if self.config.observation_only:
            raise RuntimeError(
                f"{self.name} is observation-only; refusing impedance command dispatch."
            )
        metadata = dict(command.metadata or {})
        requested_exact = bool(metadata.get("init_arm_qpos_active", False))
        requested_operation_id = str(
            metadata.get("init_arm_qpos_operation_id", "") or ""
        ).strip()
        with self._state_lock:
            active_operation_id = str(self._exact_qpos_transit_operation_id or "")
        if active_operation_id:
            if not requested_exact or requested_operation_id != active_operation_id:
                raise RuntimeError(
                    f"{self.name} exact qpos transit only accepts commands from "
                    f"operation {active_operation_id!r}"
                )
        elif requested_exact:
            raise RuntimeError(
                f"{self.name} received exact qpos command without an armed impedance transit"
            )
        if command.command_type == "joint_vector":
            command = self._joint_vector_command_as_cartesian(command)
        if command.command_type != "cartesian_pose":
            raise ValueError(
                f"{self.name} pose impedance requires cartesian_pose command, got {command.command_type!r}."
            )
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

        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
        if self.config.connect_hardware:
            with self._state_lock:
                force_control_ready = bool(self._enabled) and self._last_fault is None
            if not force_control_ready:
                raise RuntimeError(
                    f"{self.name} force control is not enabled; "
                    "run tare/enable after runtime home first. "
                    f"status={self.force_control_status_text()}"
                )
        mark_timing("precheck")
        now = time.time()
        target_pose6 = self._command_to_tool_tcp_pose6(command)
        target_joint_deg = self._command_solver_output_joint_deg(command)
        mark_timing("command_conversion")
        send_lock_wait_ms = 0.0
        mark_timing("send_lock_wait")
        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
            self._command_sequence += 1
            command_sequence = int(self._command_sequence)
            self._policy_inference_soft_hold = False
            self._policy_inference_soft_hold_joint_deg = None
            self._policy_inference_soft_hold_reason = ""
            self._policy_inference_soft_hold_timestamp = None
            self._latest_target_pose6 = list(target_pose6)
            self._latest_target_joint_deg = None if target_joint_deg is None else list(target_joint_deg)
            self._latest_target_sequence = command_sequence
            self._last_command = command.copy()
            self._last_command_timestamp = now
            self._update_dt_scale(now)
            if not self.config.connect_hardware:
                self._last_send_result_code = 0
                self._last_send_result_text = "dry_run"
        mark_timing("state_update")
        if timing_enabled:
            timing_finished_at = time.perf_counter()
            timing = {
                "enabled": True,
                "side": self.config.side,
                "effector_name": self.name,
                "timestamp": now,
                "total_ms": (timing_finished_at - timing_started_at) * 1000.0,
                "send_lock_wait_ms": send_lock_wait_ms,
                "target_joint_hint_present": target_joint_deg is not None,
                "phases": timing_phases,
            }
            with self._state_lock:
                self._last_send_command_timing = timing

    def _read_active_tool_name(self) -> str | None:
        if self.arm is None:
            return None
        code, frame = self.arm.Get_Current_Tool_Frame()
        if int(code) != 0:
            raise RuntimeError(f"{self.name} Get_Current_Tool_Frame failed: {code}")
        return _tool_name_from_frame_result((code, frame)) or None

    def _read_active_tool_frame_snapshot(self) -> dict[str, object] | None:
        if self.arm is None:
            return None
        code, frame = self.arm.Get_Current_Tool_Frame()
        if int(code) != 0:
            raise RuntimeError(f"{self.name} Get_Current_Tool_Frame failed: {code}")
        return _tool_frame_debug_snapshot((code, frame))

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

    def _sync_profile_tool_frame(
        self,
        profile: Mapping[str, Any],
        active_tool_name: str | None,
    ) -> str | None:
        if not self.config.auto_sync_tool_frame or self.arm is None:
            return active_tool_name
        tool_name, tcp_pose, payload_mass, payload_com = _profile_tool_frame_params(profile)
        updater = getattr(self.arm, "Update_Tool_Frame", None)
        creator = getattr(self.arm, "Manual_Set_Tool_Frame", None)
        changer = getattr(self.arm, "Change_Tool_Frame", None)
        if changer is None:
            raise RuntimeError(f"{self.name} RM65 SDK does not expose Change_Tool_Frame")
        action = "updated"
        update_result = updater(tool_name, tcp_pose, payload_mass, *payload_com) if updater is not None else -1
        update_code, update_text = _parse_rm65_result(update_result)
        if update_code != 0:
            if not self.config.auto_create_tool_frame:
                raise RuntimeError(
                    f"{self.name} Update_Tool_Frame({tool_name}) failed with return code {update_text}; "
                    "tool frame creation is disabled"
                )
            if creator is None:
                raise RuntimeError(
                    f"{self.name} Update_Tool_Frame({tool_name}) failed with return code {update_text}; "
                    "RM65 SDK does not expose Manual_Set_Tool_Frame"
                )
            _require_rm65_success(
                creator(tool_name, tcp_pose, payload_mass, *payload_com),
                f"{self.name} Manual_Set_Tool_Frame({tool_name})",
            )
            action = "created"
        _require_rm65_success(changer(tool_name), f"{self.name} Change_Tool_Frame({tool_name})")
        time.sleep(0.05)
        synced_tool_name = self._read_active_tool_name()
        if tool_name not in str(synced_tool_name or ""):
            raise RuntimeError(
                f"{self.name} requested active tool frame {tool_name!r}, got {synced_tool_name!r}"
            )
        print(
            f"{self.name}: {action} and activated RM65 tool frame from profile "
            f"tool={tool_name} tcp={tcp_pose} payload={payload_mass:.6f}kg com={payload_com}"
        )
        return synced_tool_name

    def _validate_profile(
        self,
        profile: Mapping[str, Any],
        active_tool_name: str | None,
        *,
        check_active_tool: bool = True,
    ) -> None:
        if is_position_endpoint_profile(dict(profile)):
            raise RuntimeError(
                f"{self.name} received the position-only profile {profile.get('name')!r}. "
                "Use --arm-control-mode position for endpoint-TCP position control, or pass an impedance-capable tool profile."
            )
        _warn_profile_tcp_safety(owner=self.name, profile=profile)
        if not self.config.strict_tool_profile:
            return
        errors = _strict_profile_errors(
            profile,
            side=self.config.side,
            active_tool_name=active_tool_name if check_active_tool else None,
            mode=self.config.impedance_mode,
        )
        if errors:
            raise RuntimeError(f"{self.name} strict impedance profile validation failed: " + "; ".join(errors))

    def _build_impedance_config(self, profile: Mapping[str, Any]) -> RM65ImpedanceConfig:
        user_overrides = dict(self.config.impedance_overrides)
        requested_control_hz = float(
            user_overrides.get("control_hz", 1.0 / float(self.config.control_dt))
        )
        requested_action_hz = (
            float(user_overrides.get("action_hz"))
            if user_overrides.get("action_hz") is not None
            else (
                float(self.config.action_hz)
                if self.config.action_hz is not None
                else requested_control_hz
            )
        )
        passthrough_position_requested = bool(
            user_overrides.get("passthrough_position_control", False)
            and user_overrides.get("canfd_follow", False)
        )
        effective_control_hz = min(requested_control_hz, _RM65_IMPEDANCE_MAX_CONTROL_HZ)
        if passthrough_position_requested:
            effective_control_hz = min(requested_action_hz, _RM65_IMPEDANCE_MAX_CONTROL_HZ)
        overrides = {
            "tare_sec": float(self.config.force_tare_sec),
            "hold_last_safe_on_ik_failure": True,
            "debug_timing": bool(self.config.debug_timing),
            **user_overrides,
        }
        overrides["control_hz"] = effective_control_hz
        overrides["action_hz"] = min(requested_action_hz, effective_control_hz)
        canfd_follow_requested = bool(overrides.get("canfd_follow", False))
        if passthrough_position_requested:
            overrides.setdefault("state_read_every", 1)
            overrides.setdefault("joint_rate_limiter_enabled", False)
            overrides.setdefault(
                "max_command_joint_step_deg",
                _RM65_PASSTHROUGH_POSITION_MAX_COMMAND_STEP_DEG,
            )
            overrides.setdefault(
                "max_command_joint_velocity_deg_s",
                _RM65_PASSTHROUGH_POSITION_MAX_COMMAND_STEP_DEG,
            )
            overrides.setdefault("follow_qpos_alpha", 1.0)
        elif canfd_follow_requested and "state_read_every" not in user_overrides:
            overrides["state_read_every"] = _RM65_CANFD_FOLLOW_DEFAULT_STATE_READ_EVERY
        impedance_config = RM65ImpedanceConfig.from_profile(
            dict(profile),
            mode=self.config.impedance_mode,
            **overrides,
        )
        self._active_impedance_config = _impedance_config_debug_snapshot(
            profile=profile,
            mode=self.config.impedance_mode,
            config=impedance_config,
            overrides=overrides,
            requested_control_hz=requested_control_hz,
            effective_control_hz=effective_control_hz,
            requested_action_hz=requested_action_hz,
        )
        return impedance_config

    def _command_controlled_frame(self, command: EffectorCommand) -> str:
        return canonical_arm_controlled_frame(
            command.metadata.get("arm_controlled_frame")
            or command.metadata.get("target_pose_controlled_frame")
            or command.metadata.get("controlled_frame")
            or TOOL_TCP_FRAME
        )

    def _command_to_tool_tcp_pose6(self, command: EffectorCommand) -> list[float]:
        controlled_frame = self._command_controlled_frame(command)
        if controlled_frame == TOOL_TCP_FRAME:
            return _matrix_to_pose6(as_pose_matrix(command.value))
        endpoint_pose = convert_pose_between_arm_control_frames(
            as_pose_matrix(command.value),
            from_frame=controlled_frame,
            to_frame=ARM_ENDPOINT_FRAME,
        )
        tool_tcp_pose = endpoint_pose @ self._endpoint_to_tool_tcp
        return _matrix_to_pose6(tool_tcp_pose)

    def _command_solver_output_joint_deg(self, command: EffectorCommand) -> list[float] | None:
        raw_qpos = command.metadata.get("solver_output_qpos")
        if raw_qpos is None:
            return None
        try:
            qpos = np.asarray(raw_qpos, dtype=float).reshape(6)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(qpos)):
            return None
        return np.rad2deg(qpos).astype(float).tolist()

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
        previous_loop_start: float | None = None
        last_dispatched_target_sequence: int | None = None
        while not self._loop_stop_event.is_set():
            timing_enabled = bool(self.config.debug_timing)
            timing_started_at = time.perf_counter()
            timing_last_at = timing_started_at
            timing_phases: list[dict[str, object]] = []
            loop_period_ms = (
                None
                if previous_loop_start is None
                else (timing_started_at - previous_loop_start) * 1000.0
            )
            deadline_lag_ms = max(0.0, (timing_started_at - next_deadline) * 1000.0)
            previous_loop_start = timing_started_at

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

            try:
                with self._state_lock:
                    if self._emergency_stopped or self._loop_stop_event.is_set():
                        break
                    target = None if self._latest_target_pose6 is None else list(self._latest_target_pose6)
                    target_joint_deg = (
                        None
                        if self._latest_target_joint_deg is None
                        else list(self._latest_target_joint_deg)
                    )
                    policy_inference_soft_hold = bool(self._policy_inference_soft_hold)
                    policy_inference_soft_hold_joint_deg = (
                        None
                        if self._policy_inference_soft_hold_joint_deg is None
                        else list(self._policy_inference_soft_hold_joint_deg)
                    )
                    policy_inference_soft_hold_reason = str(
                        self._policy_inference_soft_hold_reason
                    )
                    target_sequence = self._latest_target_sequence
                    target_timestamp = self._last_command_timestamp
                target_snapshot_wall_time = time.time()
                target_dispatched_this_step = bool(
                    target is not None and target_sequence != last_dispatched_target_sequence
                )
                step_target = target if target_dispatched_this_step else None
                step_target_joint_deg = target_joint_deg if target_dispatched_this_step else None
                mark_timing("target_snapshot")
                send_lock_wait_started_at = time.perf_counter()
                send_lock_wait_ms = 0.0
                with self._send_lock:
                    send_lock_acquired_at = time.perf_counter()
                    send_lock_wait_ms = (send_lock_acquired_at - send_lock_wait_started_at) * 1000.0
                    mark_timing("send_lock_wait")
                    with self._state_lock:
                        if self._emergency_stopped or self._loop_stop_event.is_set():
                            break
                        # ``send_command`` publishes under ``_state_lock`` but
                        # does not wait for the controller's send lock.  Re-read
                        # the target/soft-hold lease here so a fresh policy
                        # action cannot be followed by one stale hold tick.
                        latest_target = (
                            None
                            if self._latest_target_pose6 is None
                            else list(self._latest_target_pose6)
                        )
                        latest_target_joint_deg = (
                            None
                            if self._latest_target_joint_deg is None
                            else list(self._latest_target_joint_deg)
                        )
                        latest_target_sequence = self._latest_target_sequence
                        if latest_target_sequence != target_sequence:
                            target = latest_target
                            target_joint_deg = latest_target_joint_deg
                            target_sequence = latest_target_sequence
                            target_timestamp = self._last_command_timestamp
                            target_dispatched_this_step = bool(
                                target is not None
                                and target_sequence != last_dispatched_target_sequence
                            )
                            step_target = target if target_dispatched_this_step else None
                            step_target_joint_deg = (
                                target_joint_deg if target_dispatched_this_step else None
                            )
                        policy_inference_soft_hold = bool(
                            self._policy_inference_soft_hold
                        )
                        policy_inference_soft_hold_joint_deg = (
                            None
                            if self._policy_inference_soft_hold_joint_deg is None
                            else list(self._policy_inference_soft_hold_joint_deg)
                        )
                        policy_inference_soft_hold_reason = str(
                            self._policy_inference_soft_hold_reason
                        )
                    mark_timing("pre_step_check")
                    controller_step_start_wall_time = time.time()
                    controller_step_start_monotonic = time.monotonic()
                    if policy_inference_soft_hold and policy_inference_soft_hold_joint_deg is not None:
                        record = dict(
                            self._controller.step(
                                None,
                                soft_hold_joint_deg=policy_inference_soft_hold_joint_deg,
                                soft_hold_reason=policy_inference_soft_hold_reason,
                            )
                        )
                    else:
                        record = dict(
                            self._controller.step(
                                step_target,
                                target_joint_deg=step_target_joint_deg,
                            )
                        )
                    controller_step_end_wall_time = time.time()
                    controller_step_end_monotonic = time.monotonic()
                    if target_dispatched_this_step:
                        last_dispatched_target_sequence = target_sequence
                    record.update(
                        {
                            "backend_target_sequence": target_sequence,
                            "backend_target_timestamp": target_timestamp,
                            "backend_target_present": target is not None,
                            "backend_target_joint_hint_present": target_joint_deg is not None,
                            "backend_target_dispatched_this_step": target_dispatched_this_step,
                            "backend_last_dispatched_target_sequence": last_dispatched_target_sequence,
                            "backend_policy_inference_soft_hold": bool(
                                policy_inference_soft_hold
                            ),
                            "backend_policy_inference_soft_hold_reason": (
                                policy_inference_soft_hold_reason
                                if policy_inference_soft_hold
                                else None
                            ),
                            "backend_target_snapshot_wall_time": float(target_snapshot_wall_time),
                            "backend_controller_step_start_wall_time": float(
                                controller_step_start_wall_time
                            ),
                            "backend_controller_step_end_wall_time": float(
                                controller_step_end_wall_time
                            ),
                            "backend_controller_step_start_monotonic": float(
                                controller_step_start_monotonic
                            ),
                            "backend_controller_step_end_monotonic": float(
                                controller_step_end_monotonic
                            ),
                            "backend_controller_step_wall_duration_sec": float(
                                controller_step_end_wall_time - controller_step_start_wall_time
                            ),
                        }
                    )
                    if target_timestamp is not None:
                        record["backend_target_age_sec_at_snapshot"] = float(
                            target_snapshot_wall_time - float(target_timestamp)
                        )
                        record["backend_target_age_sec_at_step_start"] = float(
                            controller_step_start_wall_time - float(target_timestamp)
                        )
                        record["backend_target_age_sec_at_step_end"] = float(
                            controller_step_end_wall_time - float(target_timestamp)
                        )
                    mark_timing("controller_step")
                    loop_timing = None
                    if timing_enabled:
                        timing_finished_at = time.perf_counter()
                        loop_timing = {
                            "enabled": True,
                            "side": self.config.side,
                            "effector_name": self.name,
                            "period_target_ms": period * 1000.0,
                            "loop_period_ms": loop_period_ms,
                            "deadline_lag_ms": deadline_lag_ms,
                            "total_ms": (timing_finished_at - timing_started_at) * 1000.0,
                            "send_lock_wait_ms": send_lock_wait_ms,
                            "target_present": target is not None,
                            "target_joint_hint_present": target_joint_deg is not None,
                            "phases": timing_phases,
                        }
                        record["rm65_backend_loop_timing"] = loop_timing
                    with self._state_lock:
                        if self._emergency_stopped or self._loop_stop_event.is_set():
                            break
                        self._last_step_record = dict(record)
                        if self.config.debug_timing:
                            self._recent_step_records.append(dict(record))
                            self._recent_step_records = self._recent_step_records[-8:]
                        if loop_timing is not None:
                            self._last_impedance_loop_timing = loop_timing
                        self._last_send_result_code = 0
                        self._last_send_result_text = "ok"
            except Exception as exc:
                fault = f"{type(exc).__name__}: {exc}"
                with self._state_lock:
                    self._last_fault = fault
                    self._last_send_result_code = -1
                    self._last_send_result_text = fault
                    self._enabled = False
                    self._policy_inference_soft_hold = False
                    self._policy_inference_soft_hold_joint_deg = None
                    self._policy_inference_soft_hold_reason = ""
                    self._policy_inference_soft_hold_timestamp = None
                    self._exact_qpos_transit_operation_id = None
                    force_hard_stop_callback = self._force_hard_stop_callback
                hard_stop = _is_force_hard_stop_exception(exc)
                bimanual_stop_completed = False
                try:
                    if hard_stop and force_hard_stop_callback is not None:
                        force_hard_stop_callback(self.config.side, fault)
                        bimanual_stop_completed = self.is_emergency_stopped()
                        if not bimanual_stop_completed:
                            self.emergency_stop()
                            bimanual_stop_completed = True
                    else:
                        try:
                            self._controller.end_exact_qpos_transit()
                        finally:
                            self._controller.stop_motion()
                finally:
                    if hard_stop and force_hard_stop_callback is not None and not bimanual_stop_completed:
                        try:
                            self.emergency_stop()
                        except Exception as stop_exc:  # noqa: BLE001
                            with self._state_lock:
                                self._last_send_result_text = (
                                    f"{fault}; emergency_stop_failed:"
                                    f"{type(stop_exc).__name__}: {stop_exc}"
                                )
                    self._loop_stop_event.set()
                break
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                self._loop_stop_event.wait(sleep_sec)

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            command = None if self._last_command is None else self._last_command.copy()
            latest_target_pose6 = (
                None if self._latest_target_pose6 is None else list(self._latest_target_pose6)
            )
            latest_target_joint_deg = (
                None if self._latest_target_joint_deg is None else list(self._latest_target_joint_deg)
            )
            policy_inference_soft_hold = bool(self._policy_inference_soft_hold)
            policy_inference_soft_hold_joint_deg = (
                None
                if self._policy_inference_soft_hold_joint_deg is None
                else list(self._policy_inference_soft_hold_joint_deg)
            )
            policy_inference_soft_hold_reason = str(
                self._policy_inference_soft_hold_reason
            )
            policy_inference_soft_hold_timestamp = self._policy_inference_soft_hold_timestamp
            latest_target_sequence = self._latest_target_sequence
            command_sequence = int(self._command_sequence)
            last_command_controlled_frame = (
                None if command is None else self._command_controlled_frame(command)
            )
            record = None if self._last_step_record is None else dict(self._last_step_record)
            recent_records = [dict(item) for item in self._recent_step_records]
            send_command_timing = (
                None if self._last_send_command_timing is None else dict(self._last_send_command_timing)
            )
            impedance_loop_timing = (
                None if self._last_impedance_loop_timing is None else dict(self._last_impedance_loop_timing)
            )
            active_impedance_config = (
                None if self._active_impedance_config is None else dict(self._active_impedance_config)
            )
            active_tool_frame = (
                None if self._active_tool_frame_snapshot is None else dict(self._active_tool_frame_snapshot)
            )
            profile_tcp = (
                None if self._profile_tcp_snapshot is None else dict(self._profile_tcp_snapshot)
            )
            observation_state = (
                None
                if self._observation_state is None
                else {
                    "joint_deg": list(self._observation_state.get("joint_deg", ())),
                    "pose": list(self._observation_state.get("pose", ())),
                }
            )
            observation_raw_joint_deg = (
                None
                if self._observation_raw_joint_deg is None
                else list(self._observation_raw_joint_deg)
            )
            observation_read_timestamp = self._observation_read_timestamp
            observation_read_sequence = int(self._observation_read_sequence)
            observation_read_success = bool(self._observation_read_success)
            observation_read_error = self._observation_read_error
            if self._loop_thread is None:
                loop_state = "not_started"
            elif self._loop_thread.is_alive():
                loop_state = "alive"
            else:
                loop_state = "stopped"
            emergency_stopped = bool(self._emergency_stopped)
            measurement_timestamp = (
                None if record is None else record.get("measurement_timestamp")
            )
            measurement_timestamp_monotonic = (
                None
                if record is None
                else record.get("measurement_timestamp_monotonic")
            )
            measurement_sequence = None if record is None else record.get("step")
            measurement_valid = bool(
                record is not None and record.get("measurement_valid", False)
            )
            if measurement_valid:
                self._last_measured_joint_deg = list(record.get("measured_joint_deg"))
                self._last_measured_pose = list(record.get("measured_pose"))
                self._last_measurement_timestamp = float(measurement_timestamp)
                self._last_measurement_sequence = int(measurement_sequence)
            if self._last_measurement_timestamp is not None:
                measurement_timestamp = self._last_measurement_timestamp
                measurement_sequence = self._last_measurement_sequence
                measurement_valid = bool(
                    self._last_measured_joint_deg is not None
                    and self._last_measured_pose is not None
                )
            current_qpos = None
            raw_hardware_qpos = None
            observed_qpos_normalization = None
            current_pose = None
            target_qpos = None
            target_pose = None
            if record is not None:
                measured_joint_deg = record.get("measured_joint_deg")
                measured_joint_deg_raw = record.get("measured_joint_deg_raw")
                measured_pose6 = record.get("measured_pose")
                command_joint_deg = record.get("command_joint_deg")
                command_pose6 = record.get("command_pose")
                joint_deg = measured_joint_deg if measured_joint_deg is not None else command_joint_deg
                pose6 = measured_pose6 if measured_pose6 is not None else command_pose6
                if joint_deg is not None:
                    current_qpos = np.asarray(joint_deg, dtype=float).reshape(6) / 180.0 * np.pi
                if measured_joint_deg_raw is not None:
                    raw_hardware_qpos = (
                        np.asarray(measured_joint_deg_raw, dtype=float).reshape(6) / 180.0 * np.pi
                    )
                observed_qpos_normalization = record.get("measured_joint_normalization")
                if pose6 is not None:
                    current_pose = _pose6_to_matrix(pose6)
                if command_joint_deg is not None:
                    target_qpos = np.asarray(command_joint_deg, dtype=float).reshape(6) / 180.0 * np.pi
                if command_pose6 is not None:
                    target_pose = _pose6_to_matrix(command_pose6)
            if measurement_valid:
                current_qpos = (
                    np.asarray(self._last_measured_joint_deg, dtype=float).reshape(6)
                    / 180.0
                    * np.pi
                )
                current_pose = _pose6_to_matrix(self._last_measured_pose)
            if self.config.observation_only and not measurement_valid and observation_read_success:
                measured_joint_deg = observation_state.get("joint_deg") if observation_state else None
                measured_pose6 = observation_state.get("pose") if observation_state else None
                if measured_joint_deg is not None and measured_pose6 is not None:
                    current_qpos = (
                        np.asarray(measured_joint_deg, dtype=float).reshape(6)
                        / 180.0
                        * np.pi
                    )
                    current_pose = _pose6_to_matrix(measured_pose6)
                    raw_hardware_qpos = (
                        None
                        if observation_raw_joint_deg is None
                        else (
                            np.asarray(observation_raw_joint_deg, dtype=float).reshape(6)
                            / 180.0
                            * np.pi
                        )
                    )
                    controller = self._controller
                    observed_qpos_normalization = (
                        None
                        if controller is None
                        else dict(getattr(controller, "_last_observed_joint_normalization", {}))
                    )
                    measurement_timestamp = observation_read_timestamp
                    measurement_timestamp_monotonic = None
                    measurement_sequence = observation_read_sequence
                    measurement_valid = True
            return {
                "current_qpos": current_qpos,
                # Normalized measured qpos is the only state anchor exposed to
                # replay/deploy planners.  Preserve the raw SDK feedback for
                # diagnosis without allowing it to seed a motion command.
                "hardware_qpos": current_qpos,
                "hardware_qpos_raw": raw_hardware_qpos,
                "local_qpos_normalization": observed_qpos_normalization,
                "current_tcp_pose": current_pose,
                "last_target_qpos": target_qpos if target_qpos is not None else current_qpos,
                "last_target_tcp_pose": target_pose,
                "last_command": command,
                "last_command_controlled_frame": last_command_controlled_frame,
                "last_command_timestamp": self._last_command_timestamp,
                "last_command_sequence": latest_target_sequence,
                "command_sequence": command_sequence,
                "latest_target_tcp_pose6": latest_target_pose6,
                "latest_target_joint_deg": latest_target_joint_deg,
                "latest_target_sequence": latest_target_sequence,
                "policy_inference_soft_hold": policy_inference_soft_hold,
                "policy_inference_soft_hold_joint_deg": policy_inference_soft_hold_joint_deg,
                "policy_inference_soft_hold_reason": (
                    policy_inference_soft_hold_reason if policy_inference_soft_hold else None
                ),
                "policy_inference_soft_hold_timestamp": (
                    policy_inference_soft_hold_timestamp
                    if policy_inference_soft_hold
                    else None
                ),
                "arm_state_source": (
                    "hardware_observation_only"
                    if self.config.observation_only
                    else "pose_impedance_command"
                ),
                "hardware_refresh_time": measurement_timestamp,
                "hardware_measurement_timestamp": measurement_timestamp,
                "hardware_measurement_timestamp_monotonic": measurement_timestamp_monotonic,
                "hardware_measurement_sequence": measurement_sequence,
                "hardware_measurement_valid": measurement_valid,
                "hardware_read_success": bool(
                    observation_read_success
                    if self.config.observation_only
                    else measurement_valid
                ),
                "hardware_pose_6d": (
                    (
                        None
                        if observation_state is None
                        else observation_state.get("pose")
                    )
                    if self.config.observation_only
                    else (None if record is None else record.get("measured_pose"))
                ),
                "hardware_result_code": (
                    0
                    if (observation_read_success if self.config.observation_only else self._last_fault is None)
                    else -1
                ),
                "hardware_result_text": (
                    (observation_read_error or "ok")
                    if self.config.observation_only
                    else (self._last_fault or "ok")
                ),
                "send_result_code": self._last_send_result_code,
                "send_result_text": self._last_send_result_text,
                "emergency_stopped": emergency_stopped,
                "effective_control_dt_sec": float(self._last_effective_control_dt),
                "effective_step_scale": float(self._last_effective_step_scale),
                "impedance_enabled": bool(self._enabled) and not self.config.observation_only,
                "exact_qpos_transit_active": bool(
                    self._exact_qpos_transit_operation_id
                ),
                "exact_qpos_transit_operation_id": str(
                    self._exact_qpos_transit_operation_id or ""
                ),
                "impedance_loop_state": loop_state,
                "impedance_mode": self.config.impedance_mode,
                "observation_only": bool(self.config.observation_only),
                "force_tare_sec": float(self.config.force_tare_sec),
                "active_tool_name": self._active_tool_name,
                "active_tool_profile_path": self._active_tool_profile_path,
                "active_tool_frame": active_tool_frame,
                "profile_tcp": profile_tcp,
                "active_impedance_config": active_impedance_config,
                "last_send_command_timing": send_command_timing,
                "last_impedance_loop_timing": impedance_loop_timing,
                "last_impedance_record": record,
                "recent_impedance_records": recent_records,
                "ik_metadata": (
                    dict(record.get("ik_metadata", {}))
                    if isinstance(record, Mapping)
                    else {}
                ),
                "fault": self._last_fault,
            }

    def emergency_stop(self) -> None:
        if self.config.observation_only:
            with self._state_lock:
                self._enabled = False
                self._emergency_stopped = True
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._last_send_result_code = 0
                self._last_send_result_text = "observation_only_emergency_stop"
            return
        self._release_exact_qpos_transit_best_effort(reason="emergency_stop")
        with self._send_lock:
            self._loop_stop_event.set()
            with self._state_lock:
                self._enabled = False
                self._emergency_stopped = True
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
                self._latest_target_pose6 = None
                self._latest_target_joint_deg = None
                self._latest_target_sequence = None
                self._last_send_result_code = 0
                self._last_send_result_text = "emergency_stop"
            robot = self.arm
            if robot is None and self._controller is not None:
                robot = self._controller.robot
            if robot is None:
                return
            stop_method = getattr(robot, "Arm_Stop", None)
            stop_label = "Arm_Stop"
            if stop_method is None:
                stop_method = getattr(robot, "Arm_Slow_Stop", None)
                stop_label = "Arm_Slow_Stop"
            if stop_method is None:
                raise RuntimeError(f"{self.name} RM65 SDK exposes no Arm_Stop/Arm_Slow_Stop method")
            stop_result = stop_method()
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = f"emergency_stop:{stop_label}:{stop_text}"
            if stop_code not in (0, None):
                raise RuntimeError(f"{self.name} {stop_label} failed with return code {stop_text}")

    def close(self) -> None:
        if not self.config.observation_only:
            self._release_exact_qpos_transit_best_effort(reason="close")
        self._loop_stop_event.set()
        failures: list[LifecycleFailure] = []
        if self._loop_thread is not None:
            worker = self._loop_thread

            def join_worker() -> None:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    raise RuntimeError(
                        f"{self.name} impedance worker did not stop within 1.000s"
                    )

            failures.extend(
                collect_lifecycle_failures(
                    ((f"worker[{worker.name}].join", join_worker),)
                )
            )
            if not worker.is_alive():
                self._loop_thread = None
        controller = self._controller
        arm = self.arm

        def close_controller() -> None:
            assert controller is not None
            controller.close()
            self._controller = None

        def close_arm_socket() -> None:
            assert arm is not None
            if controller is not None and controller.follow_stream_worker_alive():
                raise RuntimeError(
                    f"{self.name} refuses to close the RM65 socket while the CANFD "
                    "follow stream worker is still alive"
                )
            try:
                arm.Arm_Socket_Close()
            finally:
                self.arm = None

        steps = []
        if controller is not None:
            if not self._emergency_stopped and not self.config.observation_only:
                steps.append(("controller.stop_motion", controller.stop_motion))
            steps.append(("controller.close", close_controller))
        if arm is not None:
            steps.append(("arm.socket_close", close_arm_socket))
        try:
            failures.extend(collect_lifecycle_failures(steps))
            raise_lifecycle_failures(f"{self.name}.close", failures)
        finally:
            with self._state_lock:
                self._enabled = False
                self._policy_inference_soft_hold = False
                self._policy_inference_soft_hold_joint_deg = None
                self._policy_inference_soft_hold_reason = ""
                self._policy_inference_soft_hold_timestamp = None
