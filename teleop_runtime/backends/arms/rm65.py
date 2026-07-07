from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Mapping, Sequence

import numpy as np
import time

from hardware.rm65.calibration_profiles import (
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    is_position_endpoint_profile,
    tcp_endpoint_delta,
    tcp_safety_warnings,
)
from ...arm_frames import (
    ARM_ENDPOINT_FRAME,
    EMPTY_TOOL_FRAME,
    TOOL_TCP_FRAME,
    canonical_arm_controlled_frame,
)
from rm65_inspire_tac.ik_tuning import (
    RM65IKTuningConfig,
    hard_branch_jump_exceeds,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
)
from robot_motion.rm65 import (
    RM65SingleArmIKConfig,
    build_rm65_single_arm_ik_solver,
    rm65_single_arm_candidate_frames,
    solve_rm65_single_arm_ik,
)

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand


RM65_IK_FRAMEWORK_ROBOT_MOTION = "robot_motion"
RM65_IK_FRAMEWORK_CHOICES = (RM65_IK_FRAMEWORK_ROBOT_MOTION,)
HAND_BASE_POSE_FLANGE_M_RAD = (0.0, 0.0, 0.009, 0.0, 0.0, 0.0)
_SINGULARITY_WARNING_INTERVAL_SEC = 1.0
DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN = 0.2
_SINGULARITY_QPOS_INIT_SLOW_SOURCES = frozenset(
    {
        "replay_qpos_init",
        "replay_qpos_init_final_hold",
        "deploy_fixed_init_qpos",
    }
)
_SINGULARITY_LABELS = {
    0: "normal",
    -1: "shoulder_singularity",
    -2: "elbow_singularity",
    -3: "wrist_singularity",
}


def normalize_rm65_ik_framework(value: object) -> str:
    text = str(value or RM65_IK_FRAMEWORK_ROBOT_MOTION).strip().lower().replace("-", "_")
    aliases = {
        "default": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "shared": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "robotmotion": RM65_IK_FRAMEWORK_ROBOT_MOTION,
        "robot_motion": RM65_IK_FRAMEWORK_ROBOT_MOTION,
    }
    resolved = aliases.get(text, text)
    if resolved not in RM65_IK_FRAMEWORK_CHOICES:
        raise ValueError(
            f"Unsupported RM65 IK framework {value!r}. Expected one of {RM65_IK_FRAMEWORK_CHOICES}."
        )
    return resolved


def _is_empty_tool_frame_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    stem = Path(text).stem
    normalized = stem if stem else text
    return normalized == EMPTY_TOOL_FRAME or normalized.endswith(f"_{EMPTY_TOOL_FRAME}")


def _load_rm65_sdk():
    try:
        from hardware.rm65.rm_api2_adapter import Arm, RM65
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RM65ArmBackend requires the RM65 SDK to be importable."
        ) from exc
    return Arm, RM65


def _parse_rm65_result(result: object) -> tuple[int | None, str]:
    if isinstance(result, (int, np.integer)):
        code = int(result)
        return code, str(code)
    text = "" if result is None else str(result)
    prefix = text.split(":", 1)[0].strip()
    try:
        code = int(prefix)
    except (TypeError, ValueError):
        code = None
    return code, text


def _singularity_status_label(status_code: int | None) -> str:
    if status_code is None:
        return "unavailable"
    return _SINGULARITY_LABELS.get(int(status_code), f"unknown_status_{int(status_code)}")


def _command_source(command: EffectorCommand) -> str:
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    source_text = str(metadata.get("source") or "").strip()
    action_source_text = str(metadata.get("action_source") or "").strip()
    if source_text == "robot_action" and action_source_text:
        return action_source_text
    return source_text or action_source_text or str(command.command_type)


def _is_qpos_init_slow_singularity_source(source: object) -> bool:
    return str(source or "").strip() in _SINGULARITY_QPOS_INIT_SLOW_SOURCES


def _joint_now_to_qpos(joint_now: object) -> np.ndarray | None:
    if joint_now is None:
        return None
    try:
        qpos = np.asarray(joint_now, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape[0] != 6 or not np.all(np.isfinite(qpos)):
        return None
    return qpos / 180.0 * np.pi


def _state_result_is_usable(
    parsed_code: int | None,
    *,
    joint_now: object,
    pose_now: object,
) -> bool:
    if parsed_code == 0:
        return True
    if parsed_code == 9:
        return _joint_now_to_qpos(joint_now) is not None or pose_now is not None
    return False


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


def _rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(float(rx)), np.cos(float(rx))
    sy, cy = np.sin(float(ry)), np.cos(float(ry))
    sz, cz = np.sin(float(rz)), np.cos(float(rz))
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_mat @ ry_mat @ rx_mat


def _pose6_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=float).reshape(6)
    if not np.all(np.isfinite(values)):
        raise ValueError("TCP pose contains non-finite values.")
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _rotation_matrix_from_euler_xyz(*values[3:6])
    matrix[:3, 3] = values[:3]
    return matrix


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def _as_pose_matrix(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape == (4, 4):
        return pose.copy()
    if pose.shape == (7,):
        x, y, z, qx, qy, qz, qw = pose
        norm = np.linalg.norm(pose[3:7])
        if norm <= 0.0:
            raise ValueError("target quaternion norm must be positive")
        qx, qy, qz, qw = np.asarray([qx, qy, qz, qw], dtype=float) / norm
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = [x, y, z]
        matrix[:3, :3] = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )
        return matrix
    raise ValueError(f"Unsupported pose shape for configured TCP transform: {pose.shape}")


def _warn_if_tcp_outside_safety_limits(
    *,
    owner: str,
    pose_flange_m_rad: Sequence[float],
    source: str,
) -> None:
    warnings = tcp_safety_warnings(pose_flange_m_rad, label=f"{owner} TCP")
    for warning in warnings:
        print(f"WARNING: {warning}; source={source}. Verify this TCP before robot motion.")


@dataclass
class RM65ArmBackendConfig:
    arm_ip: str
    side: str
    effector_name: str | None = None
    default_tcp_frame: str = TOOL_TCP_FRAME
    arm_urdf_path: str | Path | None = None
    connect_hardware: bool = True
    control_dt: float = 0.02
    max_angvel: tuple[float, ...] = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)
    dry_run_fast: bool = True
    dry_run_max_iterations: int = 48
    dry_run_max_joint_step: tuple[float, ...] = (0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
    dry_run_damping: float = 0.03
    hardware_read_retries: int = 2
    hardware_read_retry_delay_sec: float = 0.02
    initial_qpos: tuple[float, ...] | None = None
    use_actual_send_dt_scaling: bool = False
    actual_send_dt_alpha: float = 0.25
    actual_send_dt_min_scale: float = 1.0
    actual_send_dt_max_scale: float = 2.0
    actual_send_dt_stall_reset_sec: float = 0.25
    background_streaming_enabled: bool = False
    background_streaming_hz: float = 30.0
    force_position_control: bool = False
    force_sensor: int = 1
    force_mode: int = 0
    force_dir: int = 0
    force: float = 0.0
    force_follow: bool = False
    force_clear_on_start: bool = True
    auto_load_tool_profile: bool = True
    tool_profile: str | None = None
    tcp_pose_flange_m_rad: tuple[float, ...] | None = None
    singularity_protection_enabled: bool = True
    configure_singularity_thresholds: bool = True
    singularity_elbow_threshold_deg: float = 3.0
    singularity_wrist_threshold_deg: float = 3.0
    singularity_shoulder_threshold_m: float = 0.05
    singularity_qpos_init_slowdown: float = DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)
    ik_framework: str = RM65_IK_FRAMEWORK_ROBOT_MOTION
    builtin_empty_tool_frame: bool = field(default=False, init=False, repr=False)
    position_endpoint_profile: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        self.default_tcp_frame = canonical_arm_controlled_frame(self.default_tcp_frame)
        if len(self.max_angvel) != 6:
            raise ValueError("max_angvel must contain 6 joint velocity limits.")
        if len(self.dry_run_max_joint_step) != 6:
            raise ValueError("dry_run_max_joint_step must contain 6 joint step limits.")
        if self.initial_qpos is not None and len(self.initial_qpos) != 6:
            raise ValueError("initial_qpos must contain exactly 6 joint values.")
        if not 0.0 < float(self.actual_send_dt_alpha) <= 1.0:
            raise ValueError("actual_send_dt_alpha must be in (0, 1].")
        if float(self.actual_send_dt_min_scale) < 1.0:
            raise ValueError("actual_send_dt_min_scale must be >= 1.0.")
        if float(self.actual_send_dt_max_scale) < float(self.actual_send_dt_min_scale):
            raise ValueError("actual_send_dt_max_scale must be >= actual_send_dt_min_scale.")
        if float(self.actual_send_dt_stall_reset_sec) <= 0.0:
            raise ValueError("actual_send_dt_stall_reset_sec must be positive.")
        if float(self.background_streaming_hz) <= 0.0:
            raise ValueError("background_streaming_hz must be positive.")
        if self.force_sensor not in (0, 1):
            raise ValueError("force_sensor must be 0 (one-axis) or 1 (six-axis).")
        if self.force_mode not in (0, 1):
            raise ValueError("force_mode must be 0 (base frame) or 1 (tool frame).")
        if not 0 <= int(self.force_dir) <= 5:
            raise ValueError("force_dir must be in [0, 5].")
        if float(self.singularity_elbow_threshold_deg) < 0.0:
            raise ValueError("singularity_elbow_threshold_deg must be >= 0.")
        if float(self.singularity_wrist_threshold_deg) < 0.0:
            raise ValueError("singularity_wrist_threshold_deg must be >= 0.")
        if float(self.singularity_shoulder_threshold_m) < 0.0:
            raise ValueError("singularity_shoulder_threshold_m must be >= 0.")
        if not 0.0 < float(self.singularity_qpos_init_slowdown) <= 1.0:
            raise ValueError("singularity_qpos_init_slowdown must be in (0, 1].")
        self.ik_framework = normalize_rm65_ik_framework(self.ik_framework)
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)
        if _is_empty_tool_frame_name(self.tool_profile):
            self.builtin_empty_tool_frame = True
            self.tool_profile = None
            self.auto_load_tool_profile = False
            if self.tcp_pose_flange_m_rad is None:
                self.tcp_pose_flange_m_rad = tuple(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD)
        if self.tool_profile:
            from hardware.rm65.calibration_profiles import load_profile

            profile = load_profile(self.tool_profile, arm=self.side if self.side in ("left", "right") else None)
            if str(profile.get("arm")) != self.side:
                raise ValueError(f"tool_profile arm={profile.get('arm')} does not match backend side={self.side}")
            tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
            pose = tcp.get("pose_flange_m_rad")
            if pose is None:
                raise ValueError(f"tool_profile {self.tool_profile!r} does not contain tcp.pose_flange_m_rad")
            self.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
            if is_position_endpoint_profile(profile):
                self.position_endpoint_profile = True
                self.auto_load_tool_profile = False
        if self.tcp_pose_flange_m_rad is not None:
            pose = np.asarray(self.tcp_pose_flange_m_rad, dtype=float).reshape(-1)
            if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise ValueError("tcp_pose_flange_m_rad must contain 6 finite values.")
            self.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
            _warn_if_tcp_outside_safety_limits(
                owner=f"rm65_{self.side}",
                pose_flange_m_rad=self.tcp_pose_flange_m_rad,
                source="RM65ArmBackendConfig.tcp_pose_flange_m_rad",
            )


@register_backend("rm65_arm")
class RM65ArmBackend(EffectorBackend):
    def __init__(self, config: RM65ArmBackendConfig):
        self.config = config
        self.arm = None
        self._solvers: dict[str, object] = {}
        self._current_qpos = (
            None
            if config.initial_qpos is None
            else np.asarray(config.initial_qpos, dtype=float).copy()
        )
        self._last_command: EffectorCommand | None = None
        self._last_target_qpos: np.ndarray | None = None
        self._last_command_timestamp: float | None = None
        self._last_hardware_qpos: np.ndarray | None = None
        self._last_hardware_pose_6d: np.ndarray | None = None
        self._last_hardware_refresh_time: float | None = None
        self._last_hardware_result_code: int | None = None
        self._last_hardware_result_text: str | None = None
        self._last_hardware_arm_err: int | None = None
        self._last_hardware_sys_err: int | None = None
        self._last_send_result_code: int | None = None
        self._last_send_result_text: str | None = None
        self._singularity_config_attempted = False
        self._singularity_protection_ready = False
        self._last_singularity_metadata: dict[str, object] = {
            "singularity_gate_status": "not_checked",
        }
        self._last_singularity_warning_time = 0.0
        self._last_singularity_warning_key: tuple[object, ...] | None = None
        self._base_control_dt = float(config.control_dt)
        self._base_max_joint_step = np.asarray(config.max_angvel, dtype=float) * self._base_control_dt
        self._last_send_time: float | None = None
        self._smoothed_send_dt = self._base_control_dt
        self._last_effective_control_dt = self._base_control_dt
        self._last_effective_step_scale = 1.0
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._emergency_stopped = False
        self._stream_target_qpos: np.ndarray | None = None
        self._stream_target_source = ""
        self._stream_stop_event = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._force_position_active = False
        self._ik_tuning = config.ik_tuning
        assert self._ik_tuning is not None
        self._joint_distance_weights = np.asarray(
            self._ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        )
        self._jump_gate_active = False
        self._jump_hold_qpos: np.ndarray | None = None
        self._jump_hold_anchor_pose: np.ndarray | None = None
        self._jump_hold_frames = 0
        self._last_target_pose: np.ndarray | None = None
        self._last_ik_metadata: dict[str, object] = {}
        self._hand_base_to_configured_tcp = self._build_hand_base_to_configured_tcp()
        self._configured_tcp_enabled = not np.allclose(
            self._hand_base_to_configured_tcp,
            np.eye(4, dtype=float),
            atol=1e-9,
        )
        self._active_tool_profile_path: str | None = None
        self._active_tool_name: str | None = None
        self._builtin_empty_tool_frame = bool(config.builtin_empty_tool_frame)
        self._position_endpoint_profile = bool(config.position_endpoint_profile)

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.arm is not None:
            return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
        self._auto_load_active_tool_profile()
        self._configure_singularity_protection()
        if self.config.force_position_control:
            self._start_force_position_control()
        state = self.refresh_state_from_hardware()
        if state["hardware_result_code"] != 0 and self._current_qpos is None:
            self._current_qpos = np.zeros(6, dtype=float)
        if self.config.background_streaming_enabled and self._stream_thread is None:
            self._stream_stop_event.clear()
            self._stream_thread = threading.Thread(
                target=self._background_stream_loop,
                daemon=True,
                name=f"{self.name}_stream",
            )
            self._stream_thread.start()

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or (self.arm is not None)

    def hold_motion(self, reason: str = "manual hold") -> None:
        del reason
        with self._send_lock:
            with self._state_lock:
                self._stream_target_qpos = None
                self._stream_target_source = ""
                self._last_target_qpos = None
                self._last_target_pose = None
                self._jump_gate_active = False
                self._jump_hold_qpos = None
                self._jump_hold_anchor_pose = None
                self._jump_hold_frames = 0
            if self.arm is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "hold_no_arm"
                return
            stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
            stop_label = "Arm_Slow_Stop"
            if stop_method is None:
                stop_method = getattr(self.arm, "Arm_Stop", None)
                stop_label = "Arm_Stop"
            if stop_method is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "hold_no_stop_method"
                return
            stop_result = stop_method()
            stop_code, stop_text = _parse_rm65_result(stop_result)
            with self._state_lock:
                self._last_send_result_code = stop_code
                self._last_send_result_text = f"hold:{stop_label}:{stop_text}"

    def reset_motion_state(self, qpos: Sequence[float] | np.ndarray | None = None) -> None:
        with self._send_lock:
            with self._state_lock:
                if qpos is not None:
                    qpos_array = np.asarray(qpos, dtype=float).reshape(6)
                    if not np.all(np.isfinite(qpos_array)):
                        raise ValueError(f"{self.name} expected finite qpos for reset_motion_state.")
                    self._current_qpos = qpos_array.copy()
                    self._last_hardware_qpos = qpos_array.copy()
                self._stream_target_qpos = None
                self._stream_target_source = ""
                self._last_target_qpos = None
                self._last_target_pose = None
                self._last_command = None
                self._last_command_timestamp = None
                self._last_send_time = None
                self._smoothed_send_dt = self._base_control_dt
                self._last_effective_control_dt = self._base_control_dt
                self._last_effective_step_scale = 1.0
                self._jump_gate_active = False
                self._jump_hold_qpos = None
                self._jump_hold_anchor_pose = None
                self._jump_hold_frames = 0

    def _configure_singularity_protection(self) -> None:
        if self._singularity_config_attempted:
            return
        self._singularity_config_attempted = True
        self._singularity_protection_ready = False
        if not self.config.singularity_protection_enabled or self.arm is None:
            return
        analyzer = getattr(self.arm, "Algo_Kin_Singularity_Analyse", None)
        if analyzer is None:
            print(f"WARNING: {self.name} RM65 SDK does not expose singularity analysis; backend gate disabled.")
            return
        if not self.config.configure_singularity_thresholds:
            self._singularity_protection_ready = True
            return
        setter = getattr(self.arm, "Algo_Kin_Set_Singularity_Thresholds", None)
        if setter is None:
            print(
                f"WARNING: {self.name} RM65 SDK does not expose singularity threshold configuration; "
                "backend gate disabled to avoid using unknown SDK defaults."
            )
            return
        try:
            setter(
                self.config.singularity_elbow_threshold_deg,
                self.config.singularity_wrist_threshold_deg,
                self.config.singularity_shoulder_threshold_m,
            )
            self._singularity_protection_ready = True
        except Exception as exc:  # noqa: BLE001 - protection should fail open for teleop.
            print(
                f"WARNING: {self.name} failed to configure singularity thresholds: {exc}; "
                "backend singularity gate disabled."
            )
            return
        getter = getattr(self.arm, "Algo_Kin_Get_Singularity_Thresholds", None)
        if getter is None:
            print(
                f"{self.name}: singularity gate enabled "
                f"elbow={self.config.singularity_elbow_threshold_deg:.3f}deg "
                f"wrist={self.config.singularity_wrist_threshold_deg:.3f}deg "
                f"shoulder={self.config.singularity_shoulder_threshold_m:.3f}m"
            )
            return
        try:
            qe, qw, d = getter()
            print(f"{self.name}: singularity gate enabled elbow={qe:.3f}deg wrist={qw:.3f}deg shoulder={d:.3f}m")
        except Exception as exc:  # noqa: BLE001 - diagnostic only.
            print(f"WARNING: {self.name} failed to read singularity thresholds after setting: {exc}")

    def _analyze_singularity(self, qpos_rad: Sequence[float] | np.ndarray) -> dict[str, object]:
        if self.arm is None:
            return {"status_code": None, "label": "unavailable", "reason": "no_arm"}
        analyzer = getattr(self.arm, "Algo_Kin_Singularity_Analyse", None)
        if analyzer is None:
            return {"status_code": None, "label": "unavailable", "reason": "missing_sdk_method"}
        joint_deg = (np.asarray(qpos_rad, dtype=float).reshape(6) / np.pi * 180.0).tolist()
        try:
            result = analyzer(joint_deg)
        except Exception as exc:  # noqa: BLE001 - protection should fail open for teleop.
            return {"status_code": None, "label": "unavailable", "reason": str(exc)}
        if not isinstance(result, tuple) or len(result) < 2:
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_result:{result}",
            }
        status, distance = result[:2]
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            return {
                "status_code": None,
                "label": "unavailable",
                "reason": f"unexpected_status:{status}",
            }
        metadata: dict[str, object] = {
            "status_code": status_code,
            "label": _singularity_status_label(status_code),
        }
        try:
            metadata["wrist_center_distance_m"] = float(distance)
        except (TypeError, ValueError):
            metadata["wrist_center_distance_m"] = str(distance)
        return metadata

    def _record_singularity_metadata(self, metadata: dict[str, object]) -> None:
        with self._state_lock:
            self._last_singularity_metadata = dict(metadata)

    def _emit_singularity_warning(self, metadata: dict[str, object]) -> None:
        now = time.time()
        key = (
            metadata.get("singularity_gate_status"),
            metadata.get("target_singularity_label"),
            metadata.get("source"),
        )
        if (
            key == self._last_singularity_warning_key
            and now - self._last_singularity_warning_time < _SINGULARITY_WARNING_INTERVAL_SEC
        ):
            return
        self._last_singularity_warning_key = key
        self._last_singularity_warning_time = now
        print(
            f"WARNING: {self.name} singularity gate hold "
            f"source={metadata.get('source')} "
            f"target={metadata.get('target_singularity_label')} "
            f"current={metadata.get('current_singularity_label')}"
        )

    def _apply_singularity_protection(
        self,
        qpos_target: np.ndarray,
        *,
        reference_qpos: np.ndarray,
        source: str,
        max_joint_step: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        target = np.asarray(qpos_target, dtype=float).reshape(6)
        reference = np.asarray(reference_qpos, dtype=float).reshape(6)
        if not self.config.singularity_protection_enabled:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "disabled",
                    "source": source,
                }
            )
            return target
        if self.arm is None:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "not_checked",
                    "source": source,
                    "reason": "no_arm",
                }
            )
            return target
        self._configure_singularity_protection()
        if not self._singularity_protection_ready:
            self._record_singularity_metadata(
                {
                    "singularity_gate_status": "unavailable",
                    "source": source,
                }
            )
            return target

        target_status = self._analyze_singularity(target)
        target_code = target_status.get("status_code")
        target_singular = target_code not in (0, None)
        qpos_init_slow_source = _is_qpos_init_slow_singularity_source(source)
        current_status = (
            self._analyze_singularity(reference)
            if target_singular or qpos_init_slow_source
            else {"status_code": 0, "label": "normal"}
        )
        current_code = current_status.get("status_code")
        current_singular = current_code not in (0, None)
        entering_singularity = bool(target_singular and current_code == 0)
        gate_status = "hold_entry" if entering_singularity else "pass"
        slow_qpos_init = bool(
            qpos_init_slow_source
            and target_code is not None
            and (target_singular or current_singular)
        )
        if slow_qpos_init:
            if entering_singularity:
                gate_status = "slow_qpos_init_entering_singularity"
            elif target_singular:
                gate_status = "slow_qpos_init_current_singular"
            else:
                gate_status = "slow_qpos_init_exiting_singularity"
        elif target_singular and current_code not in (0, None):
            gate_status = "allow_current_singular"
        elif target_code is None:
            gate_status = "analysis_unavailable"
        metadata: dict[str, object] = {
            "singularity_gate_status": gate_status,
            "source": source,
            "target_singularity_code": target_code,
            "target_singularity_label": target_status.get("label"),
            "current_singularity_code": current_code,
            "current_singularity_label": current_status.get("label"),
            "configured_elbow_threshold_deg": float(self.config.singularity_elbow_threshold_deg),
            "configured_wrist_threshold_deg": float(self.config.singularity_wrist_threshold_deg),
            "configured_shoulder_threshold_m": float(self.config.singularity_shoulder_threshold_m),
        }
        if slow_qpos_init:
            slowdown = float(self.config.singularity_qpos_init_slowdown)
            if max_joint_step is None:
                slow_target = target.copy()
            else:
                step_limit = np.asarray(max_joint_step, dtype=float).reshape(6)
                step_limit = np.maximum(0.0, step_limit) * slowdown
                slow_target = reference + np.clip(target - reference, -step_limit, step_limit)
            metadata["singularity_qpos_init_slowdown"] = slowdown
            metadata["singularity_qpos_init_slow_source"] = True
            metadata["singularity_qpos_init_requested_target"] = target.tolist()
            metadata["singularity_qpos_init_slow_target"] = slow_target.tolist()
            target = slow_target
        if "wrist_center_distance_m" in target_status:
            metadata["target_wrist_center_distance_m"] = target_status["wrist_center_distance_m"]
        if "wrist_center_distance_m" in current_status:
            metadata["current_wrist_center_distance_m"] = current_status["wrist_center_distance_m"]
        self._record_singularity_metadata(metadata)
        if entering_singularity and not slow_qpos_init:
            self._emit_singularity_warning(metadata)
            return reference.copy()
        return target

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        with self._state_lock:
            if self._emergency_stopped:
                raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
            if self._current_qpos is None:
                self._current_qpos = np.zeros(6, dtype=float)
        command_start_time = time.time()
        command_source = _command_source(command)
        max_joint_step = self._resolve_effective_max_joint_step(
            now=command_start_time,
            update_state=True,
        )
        ik_solve_elapsed_ms: float | None = None

        if command.command_type == "joint_vector":
            qpos_target = np.asarray(command.value, dtype=float)
            if qpos_target.shape != (6,):
                raise ValueError(
                    f"{self.name} expects joint_vector with shape (6,), got {qpos_target.shape}."
                )
            if self.config.connect_hardware or not self.config.dry_run_fast:
                delta = np.clip(
                    qpos_target - self._current_qpos,
                    -max_joint_step,
                    max_joint_step,
                )
                qpos_target = self._current_qpos + delta
        elif command.command_type == "cartesian_pose":
            ik_solve_started = time.perf_counter()
            qpos_target = self._solve_cartesian_command(command, max_joint_step=max_joint_step)
            ik_solve_elapsed_ms = (time.perf_counter() - ik_solve_started) * 1000.0
        else:
            raise ValueError(
                f"{self.name} does not support command_type '{command.command_type}'."
            )
        if ik_solve_elapsed_ms is not None:
            self._last_ik_metadata = {
                **dict(self._last_ik_metadata),
                "ik_status": dict(self._last_ik_metadata).get("ik_jump_gate_status", "pass"),
                "ik_backend_source": "rm65_arm_backend",
                "ik_framework": self.config.ik_framework,
                "ik_backend_solve_ms": float(ik_solve_elapsed_ms),
                "ik_frame_total_ms": float(ik_solve_elapsed_ms),
                "ik_solve_total_ms": float(ik_solve_elapsed_ms),
            }

        with self._state_lock:
            reference_qpos = self._current_qpos.copy()
        qpos_target = self._apply_singularity_protection(
            qpos_target,
            reference_qpos=reference_qpos,
            source=command_source,
            max_joint_step=max_joint_step,
        )

        if self.arm is not None and self.config.background_streaming_enabled:
            with self._send_lock:
                with self._state_lock:
                    if self._emergency_stopped:
                        raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
                    self._stream_target_qpos = qpos_target.copy()
                    self._stream_target_source = command_source
                    self._last_target_qpos = qpos_target.copy()
                    self._last_command = command.copy()
                    self._last_command_timestamp = command_start_time
            return

        with self._send_lock:
            with self._state_lock:
                if self._emergency_stopped:
                    raise RuntimeError(f"{self.name} emergency_stop is latched; refusing command dispatch.")
            if self.arm is not None:
                raw_send_result = self._send_joint_target(qpos_target)
                send_code, send_text = _parse_rm65_result(raw_send_result)
                with self._state_lock:
                    self._last_send_result_code = send_code
                    self._last_send_result_text = send_text
        self._last_send_time = command_start_time

        with self._state_lock:
            self._current_qpos = qpos_target.copy()
            self._last_target_qpos = qpos_target.copy()
            self._last_command = command.copy()
            self._last_command_timestamp = command_start_time

    def _solver_for_tcp_frame(self, tcp_frame: str):
        tcp_frame = canonical_arm_controlled_frame(tcp_frame)
        if tcp_frame not in self._solvers:
            try:
                self._solvers[tcp_frame] = build_rm65_single_arm_ik_solver(
                    RM65SingleArmIKConfig(
                        arm_urdf_path=self.config.arm_urdf_path,
                        controlled_frame=tcp_frame,
                        ik_tuning=self._ik_tuning,
                        dry_run_max_iterations=int(self.config.dry_run_max_iterations),
                        dry_run_max_joint_step_rad=tuple(self.config.dry_run_max_joint_step),
                        dry_run_damping=float(self.config.dry_run_damping),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{self.name} could not resolve tcp_frame '{tcp_frame}' against the arm URDF."
                ) from exc
        return self._solvers[tcp_frame]

    def _candidate_tcp_frames(self, tcp_frame: str) -> tuple[str, ...]:
        return rm65_single_arm_candidate_frames(canonical_arm_controlled_frame(tcp_frame))

    def _build_hand_base_to_configured_tcp(self) -> np.ndarray:
        pose_flange_m_rad = self._active_tcp_pose_flange_m_rad()
        t_flange_hand_base = _pose6_to_matrix(HAND_BASE_POSE_FLANGE_M_RAD)
        t_flange_tcp = _pose6_to_matrix(pose_flange_m_rad)
        return _invert_transform(t_flange_hand_base) @ t_flange_tcp

    def _active_tcp_pose_flange_m_rad(self) -> tuple[float, ...]:
        pose_flange_m_rad = (
            ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
            if self.config.tcp_pose_flange_m_rad is None
            else self.config.tcp_pose_flange_m_rad
        )
        return tuple(float(value) for value in pose_flange_m_rad)

    def _active_tcp_source(self) -> str:
        if self._position_endpoint_profile:
            return "position_endpoint_profile"
        if self._builtin_empty_tool_frame:
            return "builtin_empty_arm_endpoint"
        if self.config.tcp_pose_flange_m_rad is None:
            return "arm_endpoint_default"
        if self._active_tool_profile_path:
            return "active_tool_profile"
        if self.config.tool_profile:
            return "configured_tool_profile"
        return "configured_tcp_pose"

    def _set_configured_tcp_pose(self, pose_flange_m_rad: Sequence[float]) -> None:
        pose = np.asarray(pose_flange_m_rad, dtype=float).reshape(6)
        if not np.all(np.isfinite(pose)):
            raise ValueError("tcp_pose_flange_m_rad must contain finite values.")
        self.config.tcp_pose_flange_m_rad = tuple(float(value) for value in pose)
        _warn_if_tcp_outside_safety_limits(
            owner=self.name,
            pose_flange_m_rad=self.config.tcp_pose_flange_m_rad,
            source="active tool profile",
        )
        self._hand_base_to_configured_tcp = self._build_hand_base_to_configured_tcp()
        self._configured_tcp_enabled = not np.allclose(
            self._hand_base_to_configured_tcp,
            np.eye(4, dtype=float),
            atol=1e-9,
        )

    def _auto_load_active_tool_profile(self) -> None:
        if self.arm is None:
            return
        if self.config.tcp_pose_flange_m_rad is not None:
            return
        if self.config.tool_profile or not self.config.auto_load_tool_profile:
            return
        if self.config.side not in ("left", "right"):
            return
        try:
            code, frame = self.arm.Get_Current_Tool_Frame()
        except Exception as exc:
            print(f"WARNING: {self.name} could not read active RM65 tool frame for TCP profile auto-load: {exc}")
            return
        if int(code) != 0:
            print(f"WARNING: {self.name} Get_Current_Tool_Frame failed during TCP profile auto-load: {code}")
            return
        tool_name = _tool_name_from_frame_result((code, frame))
        self._active_tool_name = tool_name or None
        if not tool_name:
            return
        if _is_empty_tool_frame_name(tool_name):
            self._builtin_empty_tool_frame = True
            print(
                f"{self.name}: active tool frame is '{tool_name}'; using built-in "
                "arm_endpoint TCP and skipping calibration profile auto-load for position control."
            )
            return
        try:
            from hardware.rm65.calibration_profiles import load_profile

            profile = load_profile(tool_name, arm=self.config.side)
            profile_tool = str(profile.get("tool_name") or profile.get("name") or "")
            if profile_tool != tool_name:
                raise ValueError(f"profile tool_name={profile_tool!r} does not match active tool {tool_name!r}")
            tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
            pose = tcp.get("pose_flange_m_rad")
            if pose is None:
                print(
                    f"WARNING: {self.name} active tool profile arm={self.config.side} "
                    f"tool={tool_name} has no tcp.pose_flange_m_rad; using arm_endpoint"
                )
                return
            self._set_configured_tcp_pose(pose)
            metrics = tcp_endpoint_delta(self.config.tcp_pose_flange_m_rad)
            self._active_tool_profile_path = str(profile.get("_profile_path") or "")
            print(
                f"{self.name}: auto-loaded TCP profile "
                f"tool={tool_name} path={self._active_tool_profile_path or '(unknown)'} "
                f"configured_tcp_enabled={self._configured_tcp_enabled} "
                f"tcp_endpoint_delta={metrics['translation_m']:.4f}m/{metrics['rotation_deg']:.2f}deg"
            )
        except Exception as exc:
            print(
                f"WARNING: {self.name} no usable TCP profile for active tool "
                f"arm={self.config.side} tool={tool_name}: {exc}. Using arm_endpoint."
            )

    def _target_pose_for_solver(self, tcp_frame: str, target_pose: object) -> np.ndarray:
        matrix = _as_pose_matrix(target_pose)
        controlled_frame = canonical_arm_controlled_frame(tcp_frame)
        if controlled_frame == ARM_ENDPOINT_FRAME:
            return matrix
        if controlled_frame == TOOL_TCP_FRAME and self._configured_tcp_enabled:
            return matrix @ _invert_transform(self._hand_base_to_configured_tcp)
        return matrix

    def _pose_from_solver_frame(self, tcp_frame: str, solver_pose: np.ndarray) -> np.ndarray:
        matrix = np.asarray(solver_pose, dtype=float).reshape(4, 4)
        controlled_frame = canonical_arm_controlled_frame(tcp_frame)
        if controlled_frame == ARM_ENDPOINT_FRAME:
            return matrix
        if controlled_frame == TOOL_TCP_FRAME and self._configured_tcp_enabled:
            return matrix @ self._hand_base_to_configured_tcp
        return matrix

    def _controlled_frame_for_cartesian_command(self, command: EffectorCommand) -> str:
        for key in ("arm_controlled_frame", "target_pose_controlled_frame", "controlled_frame"):
            value = command.metadata.get(key)
            if value:
                return canonical_arm_controlled_frame(str(value))
        return canonical_arm_controlled_frame(command.metadata.get("tcp_frame", self.config.default_tcp_frame))

    def _solve_cartesian_command(
        self,
        command: EffectorCommand,
        *,
        max_joint_step: np.ndarray | None = None,
    ) -> np.ndarray:
        tcp_frame = self._controlled_frame_for_cartesian_command(command)
        solver = self._solver_for_tcp_frame(tcp_frame)
        target_pose = self._target_pose_for_solver(tcp_frame, command.value)
        current_qpos = np.asarray(self._current_qpos, dtype=float).copy()
        command_max_joint_step = (
            np.asarray(max_joint_step, dtype=float).reshape(6)
            if max_joint_step is not None
            else self.get_effective_max_joint_step()
        )
        solve_kwargs = {
            "target_pose": target_pose,
            "current_qpos": current_qpos,
            "fast_iterative": bool(not self.config.connect_hardware and self.config.dry_run_fast),
            "max_joint_step_rad": command_max_joint_step,
            "dry_run_max_iterations": int(self.config.dry_run_max_iterations),
            "dry_run_max_joint_step_rad": tuple(self.config.dry_run_max_joint_step),
            "dry_run_damping": float(self.config.dry_run_damping),
        }
        qpos_target = solve_rm65_single_arm_ik(solver, **solve_kwargs)
        return self._apply_jump_gate(
            solver=solver,
            target_pose=target_pose,
            current_qpos=current_qpos,
            candidate_qpos=np.asarray(qpos_target, dtype=float),
        )

    def _estimate_effective_control_dt(
        self,
        *,
        now: float | None,
        update_state: bool,
    ) -> float:
        if not self.config.use_actual_send_dt_scaling:
            if update_state:
                self._last_effective_control_dt = self._base_control_dt
                self._last_effective_step_scale = 1.0
            return self._base_control_dt

        current_time = time.time() if now is None else float(now)
        smoothed_dt = float(self._smoothed_send_dt)
        if self._last_send_time is None:
            smoothed_dt = self._base_control_dt
        else:
            raw_dt = max(0.0, current_time - float(self._last_send_time))
            if raw_dt > float(self.config.actual_send_dt_stall_reset_sec):
                smoothed_dt = self._base_control_dt
            else:
                alpha = float(self.config.actual_send_dt_alpha)
                smoothed_dt = alpha * raw_dt + (1.0 - alpha) * float(self._smoothed_send_dt)
        effective_dt = float(
            np.clip(
                smoothed_dt,
                self._base_control_dt * float(self.config.actual_send_dt_min_scale),
                self._base_control_dt * float(self.config.actual_send_dt_max_scale),
            )
        )
        if update_state:
            self._smoothed_send_dt = smoothed_dt
            self._last_effective_control_dt = effective_dt
            self._last_effective_step_scale = (
                effective_dt / self._base_control_dt if self._base_control_dt > 0.0 else 1.0
            )
        return effective_dt

    def _resolve_effective_max_joint_step(
        self,
        *,
        now: float | None = None,
        update_state: bool,
    ) -> np.ndarray:
        effective_dt = self._estimate_effective_control_dt(now=now, update_state=update_state)
        return np.asarray(self.config.max_angvel, dtype=float) * effective_dt

    def get_effective_max_joint_step(self, *, now: float | None = None) -> np.ndarray:
        return self._resolve_effective_max_joint_step(now=now, update_state=False).copy()

    def _background_stream_loop(self) -> None:
        period_sec = 1.0 / float(self.config.background_streaming_hz)
        while not self._stream_stop_event.is_set():
            loop_start = time.time()
            with self._state_lock:
                target_qpos = (
                    None if self._stream_target_qpos is None else self._stream_target_qpos.copy()
                )
                target_source = self._stream_target_source or "background_stream"
                current_qpos = (
                    None if self._current_qpos is None else self._current_qpos.copy()
                )
            if target_qpos is not None:
                if current_qpos is None:
                    current_qpos = np.zeros(6, dtype=float)
                max_joint_step = self._resolve_effective_max_joint_step(
                    now=loop_start,
                    update_state=True,
                )
                delta = np.clip(
                    target_qpos - current_qpos,
                    -max_joint_step,
                    max_joint_step,
                )
                next_qpos = current_qpos + delta
                next_qpos = self._apply_singularity_protection(
                    next_qpos,
                    reference_qpos=current_qpos,
                    source=target_source,
                    max_joint_step=max_joint_step,
                )
                if self.arm is not None and not np.allclose(next_qpos, current_qpos, atol=1e-10):
                    with self._send_lock:
                        with self._state_lock:
                            if self._emergency_stopped or self._stream_stop_event.is_set():
                                break
                        raw_send_result = self._send_joint_target(next_qpos)
                        send_code, send_text = _parse_rm65_result(raw_send_result)
                        with self._state_lock:
                            self._last_send_result_code = send_code
                            self._last_send_result_text = send_text
                            self._current_qpos = next_qpos.copy()
                            self._last_send_time = loop_start
            elapsed = time.time() - loop_start
            remaining = period_sec - elapsed
            if remaining > 0.0:
                self._stream_stop_event.wait(remaining)

    def _apply_jump_gate(
        self,
        *,
        solver,
        target_pose: np.ndarray,
        current_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
    ) -> np.ndarray:
        jump_gate = self._ik_tuning.jump_gate
        target_pose = np.asarray(target_pose, dtype=float)
        candidate_qpos = np.asarray(candidate_qpos, dtype=float).reshape(-1)
        if not jump_gate.enabled:
            self._last_target_pose = target_pose.copy()
            self._last_ik_metadata = {
                "ik_objective_mode": self._ik_tuning.ik_objective_mode,
                "ik_branch_selection": getattr(solver, "last_branch_selection", "unknown"),
                "ik_jump_gate_status": "disabled",
                "configured_tcp_enabled": self._configured_tcp_enabled,
                "ik_framework": self.config.ik_framework,
            }
            return candidate_qpos

        relaxed = self._jump_gate_active
        per_joint_limit = (
            jump_gate.relaxed_per_joint_jump_deg
            if relaxed
            else jump_gate.per_joint_jump_deg
        )
        weighted_limit = (
            jump_gate.relaxed_weighted_jump_threshold_deg
            if relaxed
            else jump_gate.weighted_jump_threshold_deg
        )
        pose_translation_m = float("inf")
        pose_rotation_deg = float("inf")
        if self._last_target_pose is not None:
            pose_translation_m, pose_rotation_deg = pose_delta_metrics(
                self._last_target_pose,
                target_pose,
            )
        small_pose = bool(
            self._last_target_pose is not None
            and pose_translation_m <= float(jump_gate.small_pose_translation_m)
            and pose_rotation_deg <= float(jump_gate.small_pose_rotation_deg)
        )
        moderate_pose_translation_m = max(
            float(jump_gate.small_pose_translation_m),
            float(getattr(jump_gate, "moderate_pose_translation_m", jump_gate.release_translation_m)),
        )
        moderate_pose_rotation_deg = max(
            float(jump_gate.small_pose_rotation_deg),
            float(getattr(jump_gate, "moderate_pose_rotation_deg", jump_gate.release_rotation_deg)),
        )
        moderate_pose = bool(
            self._last_target_pose is not None
            and pose_translation_m <= moderate_pose_translation_m
            and pose_rotation_deg <= moderate_pose_rotation_deg
        )
        jump_exceeded, per_joint_delta_deg, weighted_jump_deg = joint_jump_exceeds(
            reference_q=current_qpos,
            candidate_q=candidate_qpos,
            joint_distance_weights=self._joint_distance_weights,
            per_joint_jump_deg=per_joint_limit,
            weighted_jump_threshold_deg=weighted_limit,
        )
        moderate_jump_scale = max(
            1.0,
            float(getattr(jump_gate, "moderate_pose_joint_jump_scale", 1.25)),
        )
        moderate_per_joint_limit = np.asarray(per_joint_limit, dtype=float).reshape(-1) * moderate_jump_scale
        moderate_weighted_limit = float(weighted_limit) * moderate_jump_scale
        moderate_jump_exceeded = bool(
            np.any(per_joint_delta_deg > moderate_per_joint_limit)
            or weighted_jump_deg > moderate_weighted_limit
        )
        moderate_pose_large_jump = bool(moderate_pose and moderate_jump_exceeded)
        hard_branch_exceeded, hard_branch_delta_deg = hard_branch_jump_exceeds(
            reference_q=current_qpos,
            candidate_q=candidate_qpos,
            joint_indices=jump_gate.hard_branch_joint_indices,
            max_delta_deg=jump_gate.hard_branch_max_delta_deg,
        )
        release_ready = False
        if self._jump_gate_active and self._jump_hold_anchor_pose is not None:
            release_translation_m, release_rotation_deg = pose_delta_metrics(
                self._jump_hold_anchor_pose,
                target_pose,
            )
            release_ready = bool(
                release_translation_m >= float(jump_gate.release_translation_m)
                or release_rotation_deg >= float(jump_gate.release_rotation_deg)
            )

        metadata = {
            "ik_objective_mode": self._ik_tuning.ik_objective_mode,
            "ik_branch_selection": getattr(solver, "last_branch_selection", "unknown"),
            "ik_branch_distance_score_deg": getattr(
                solver,
                "last_branch_distance_score_deg",
                None,
            ),
            "ik_jump_gate_status": "pass",
            "ik_jump_gate_hold_frames": self._jump_hold_frames,
            "ik_pose_delta_m": None if not np.isfinite(pose_translation_m) else pose_translation_m,
            "ik_pose_delta_deg": None if not np.isfinite(pose_rotation_deg) else pose_rotation_deg,
            "ik_pose_is_small": small_pose,
            "ik_pose_is_moderate": moderate_pose,
            "ik_joint_jump_max_deg": float(np.max(per_joint_delta_deg)),
            "ik_joint_jump_weighted_deg": float(weighted_jump_deg),
            "ik_joint_jump_threshold_deg": float(weighted_limit),
            "ik_moderate_pose_translation_m": float(moderate_pose_translation_m),
            "ik_moderate_pose_rotation_deg": float(moderate_pose_rotation_deg),
            "ik_moderate_pose_joint_jump_scale": float(moderate_jump_scale),
            "ik_moderate_pose_joint_jump_exceeded": bool(moderate_jump_exceeded),
            "ik_moderate_pose_large_jump": bool(moderate_pose_large_jump),
            "ik_moderate_pose_per_joint_threshold_deg": moderate_per_joint_limit.tolist(),
            "ik_moderate_pose_weighted_threshold_deg": float(moderate_weighted_limit),
            "ik_hard_branch_joint_indices": list(jump_gate.hard_branch_joint_indices),
            "ik_hard_branch_delta_deg": hard_branch_delta_deg.tolist(),
            "ik_hard_branch_max_delta_deg": list(jump_gate.hard_branch_max_delta_deg),
            "ik_hard_branch_exceeded": bool(hard_branch_exceeded),
            "ik_release_ready": release_ready,
            "configured_tcp_enabled": self._configured_tcp_enabled,
            "ik_framework": self.config.ik_framework,
        }

        should_hold = bool(
            (
                hard_branch_exceeded
                or (small_pose and jump_exceeded)
                or moderate_pose_large_jump
            )
            and not release_ready
        )
        if self._jump_gate_active and not should_hold and release_ready:
            metadata["ik_jump_gate_status"] = "released"
        elif self._jump_gate_active and should_hold:
            metadata["ik_jump_gate_status"] = "holding"
            if hard_branch_exceeded:
                metadata["ik_jump_gate_reason"] = "hard_branch_jump"
            elif small_pose and jump_exceeded:
                metadata["ik_jump_gate_reason"] = "small_pose_large_q_jump"
            else:
                metadata["ik_jump_gate_reason"] = "moderate_pose_large_q_jump"
        elif should_hold:
            metadata["ik_jump_gate_status"] = "hold_entered"
            if hard_branch_exceeded:
                metadata["ik_jump_gate_reason"] = "hard_branch_jump"
            elif small_pose and jump_exceeded:
                metadata["ik_jump_gate_reason"] = "small_pose_large_q_jump"
            else:
                metadata["ik_jump_gate_reason"] = "moderate_pose_large_q_jump"

        if should_hold:
            if not self._jump_gate_active:
                self._jump_gate_active = True
                self._jump_hold_qpos = current_qpos.copy()
                self._jump_hold_anchor_pose = target_pose.copy()
                self._jump_hold_frames = 0
            self._jump_hold_frames += 1
            metadata["ik_jump_gate_hold_frames"] = self._jump_hold_frames
            self._last_target_pose = target_pose.copy()
            self._last_ik_metadata = metadata
            return current_qpos.copy()

        self._jump_gate_active = False
        self._jump_hold_qpos = None
        self._jump_hold_anchor_pose = None
        self._jump_hold_frames = 0
        self._last_target_pose = target_pose.copy()
        self._last_ik_metadata = metadata
        return candidate_qpos.copy()

    def _start_force_position_control(self) -> None:
        if self.arm is None:
            return
        if self.config.force_clear_on_start:
            clear_result = self.arm.Clear_Force_Data(True)
            clear_code, clear_text = _parse_rm65_result(clear_result)
            if clear_code != 0:
                raise RuntimeError(
                    f"{self.name} failed to clear force data before force-position control: {clear_text}"
                )
        start_result = self.arm.Start_Force_Position_Move(True)
        start_code, start_text = _parse_rm65_result(start_result)
        if start_code != 0:
            raise RuntimeError(
                f"{self.name} failed to start force-position control: {start_text}"
            )
        self._force_position_active = True

    def _send_joint_target(self, qpos_rad: np.ndarray) -> object:
        if self.arm is None:
            return None
        joint_deg = np.asarray(qpos_rad, dtype=float).reshape(6) / np.pi * 180.0
        if self.config.force_position_control:
            return self.arm.Force_Position_Move_Joint(
                joint_deg,
                sensor=self.config.force_sensor,
                mode=self.config.force_mode,
                dir=self.config.force_dir,
                force=self.config.force,
                follow=self.config.force_follow,
            )
        return self.arm.Movej_CANFD(joint_deg, False)

    def refresh_state_from_hardware(
        self,
        *,
        hardware_read_retries: int | None = None,
        hardware_read_retry_delay_sec: float | None = None,
    ) -> dict[str, np.ndarray | EffectorCommand | None | int | float | str]:
        if self.arm is None:
            return self.get_state()
        retry_count = max(
            1,
            int(
                self.config.hardware_read_retries
                if hardware_read_retries is None
                else hardware_read_retries
            )
            + 1,
        )
        retry_delay_sec = float(
            self.config.hardware_read_retry_delay_sec
            if hardware_read_retry_delay_sec is None
            else hardware_read_retry_delay_sec
        )
        raw_result = -1
        joint_now = None
        pose_now = None
        arm_err = None
        sys_err = None
        parsed_code: int | None = None
        parsed_text = ""
        qpos_from_joint_now = None
        for attempt in range(retry_count):
            try:
                raw_result, joint_now, pose_now, arm_err, sys_err = self.arm.Get_Current_Arm_State()
            except Exception as exc:
                raw_result = f"EXCEPTION: {exc}"
                joint_now = None
                pose_now = None
                arm_err = None
                sys_err = None
            parsed_code, parsed_text = _parse_rm65_result(raw_result)
            qpos_from_joint_now = _joint_now_to_qpos(joint_now)
            if _state_result_is_usable(
                parsed_code,
                joint_now=joint_now,
                pose_now=pose_now,
            ):
                break
            if attempt + 1 < retry_count:
                time.sleep(retry_delay_sec)
        self._last_hardware_result_code = parsed_code
        self._last_hardware_result_text = parsed_text
        self._last_hardware_arm_err = None if arm_err is None else int(arm_err)
        self._last_hardware_sys_err = None if sys_err is None else int(sys_err)
        self._last_hardware_refresh_time = time.time()
        with self._state_lock:
            if pose_now is not None:
                self._last_hardware_pose_6d = np.asarray(pose_now, dtype=float)
            if qpos_from_joint_now is not None:
                self._last_hardware_qpos = qpos_from_joint_now.copy()
                self._current_qpos = qpos_from_joint_now
        return self.get_state()

    def get_state(self) -> dict[str, np.ndarray | EffectorCommand | None | int | float | str]:
        with self._state_lock:
            current_qpos = None if self._current_qpos is None else self._current_qpos.copy()
            hardware_qpos = (
                None if self._last_hardware_qpos is None else self._last_hardware_qpos.copy()
            )
            last_target_qpos = (
                None if self._last_target_qpos is None else self._last_target_qpos.copy()
            )
            last_command = None if self._last_command is None else self._last_command.copy()
            hardware_pose_6d = (
                None if self._last_hardware_pose_6d is None else self._last_hardware_pose_6d.copy()
            )
            singularity_metadata = dict(self._last_singularity_metadata)
            emergency_stopped = bool(self._emergency_stopped)
        current_tcp_pose = None
        if current_qpos is not None:
            try:
                current_tcp_pose = self.compute_tcp_pose(qpos=current_qpos)
            except Exception:
                current_tcp_pose = None
        tcp_pose_flange = self._active_tcp_pose_flange_m_rad()
        tcp_delta = tcp_endpoint_delta(tcp_pose_flange)
        ik_metadata = dict(self._last_ik_metadata)
        ik_metadata.update(singularity_metadata)
        return {
            "current_qpos": current_qpos,
            "current_tcp_pose": current_tcp_pose,
            "hardware_qpos": hardware_qpos,
            "last_target_qpos": last_target_qpos,
            "last_command": last_command,
            "last_command_timestamp": self._last_command_timestamp,
            "arm_state_source": "last_commanded_action",
            "hardware_pose_6d": hardware_pose_6d,
            "hardware_refresh_time": self._last_hardware_refresh_time,
            "hardware_result_code": self._last_hardware_result_code,
            "hardware_result_text": self._last_hardware_result_text,
            "hardware_arm_err": self._last_hardware_arm_err,
            "hardware_sys_err": self._last_hardware_sys_err,
            "send_result_code": self._last_send_result_code,
            "send_result_text": self._last_send_result_text,
            "emergency_stopped": emergency_stopped,
            "effective_control_dt_sec": float(self._last_effective_control_dt),
            "effective_step_scale": float(self._last_effective_step_scale),
            "ik_metadata": ik_metadata,
            "singularity_metadata": singularity_metadata,
            "singularity_protection_ready": bool(self._singularity_protection_ready),
            "active_tool_name": self._active_tool_name,
            "active_tool_profile_path": self._active_tool_profile_path,
            "configured_tcp_enabled": self._configured_tcp_enabled,
            "tool_tcp_source": self._active_tcp_source(),
            "tool_tcp_pose_flange_m_rad": np.asarray(tcp_pose_flange, dtype=float),
            "tool_tcp_endpoint_delta_m": float(tcp_delta["translation_m"]),
            "tool_tcp_endpoint_delta_deg": float(tcp_delta["rotation_deg"]),
        }

    def set_local_qpos(self, qpos: Sequence[float] | np.ndarray | None) -> None:
        with self._state_lock:
            if qpos is None:
                self._current_qpos = None
                return
            qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
            if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
                raise ValueError(f"{self.name} expected a finite 6D joint vector.")
            self._current_qpos = qpos_array.copy()
            self._jump_gate_active = False
            self._jump_hold_qpos = None
            self._jump_hold_anchor_pose = None
            self._jump_hold_frames = 0

    def compute_tcp_pose(
        self,
        tcp_frame: str | None = None,
        *,
        qpos: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        if qpos is None:
            with self._state_lock:
                if self._current_qpos is None:
                    raise RuntimeError(f"{self.name} does not have a current joint state yet.")
                qpos_array = self._current_qpos.copy()
        else:
            qpos_array = np.asarray(qpos, dtype=float).reshape(-1)
            if qpos_array.shape != (6,) or not np.all(np.isfinite(qpos_array)):
                raise RuntimeError(f"{self.name} does not have a current joint state yet.")
        tcp_frame = canonical_arm_controlled_frame(
            self.config.default_tcp_frame if tcp_frame is None else tcp_frame
        )
        solver = self._solver_for_tcp_frame(tcp_frame)
        solver_pose = np.asarray(solver.compute_pose(qpos_array), dtype=float)
        return self._pose_from_solver_frame(tcp_frame, solver_pose)

    def emergency_stop(self) -> None:
        with self._send_lock:
            self._stream_stop_event.set()
            with self._state_lock:
                self._emergency_stopped = True
                self._stream_target_qpos = None
                self._stream_target_source = ""
            if self.arm is None:
                with self._state_lock:
                    self._last_send_result_code = 0
                    self._last_send_result_text = "emergency_stop_no_arm"
                return
            if self._force_position_active:
                self.arm.Stop_Force_Position_Move(True)
                self._force_position_active = False
            stop_method = getattr(self.arm, "Arm_Stop", None)
            stop_label = "Arm_Stop"
            if stop_method is None:
                stop_method = getattr(self.arm, "Arm_Slow_Stop", None)
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
        if self._stream_thread is not None:
            self._stream_stop_event.set()
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        if self.arm is None:
            return
        try:
            if self._force_position_active:
                self.arm.Stop_Force_Position_Move(True)
                self._force_position_active = False
            self.arm.RM_API_UnInit()
        finally:
            try:
                self.arm.Arm_Socket_Close()
            finally:
                self.arm = None
