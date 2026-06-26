from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
import time

from hardware.rm65.calibration_profiles import (
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    tcp_endpoint_delta,
    tcp_safety_warnings,
)
from ...arm_frames import (
    ARM_ENDPOINT_FRAME,
    TOOL_TCP_FRAME,
    canonical_arm_controlled_frame,
    rm65_urdf_frame_for_controlled_frame,
)
from rm65_inspire_tac.ik_tuning import (
    RM65IKTuningConfig,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
)

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand
from ...tcp import HAND_BASE_TCP_FRAME


HAND_BASE_POSE_FLANGE_M_RAD = (0.0, 0.0, 0.009, 0.0, 0.0, 0.0)


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
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)

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
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)
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
        self._base_control_dt = float(config.control_dt)
        self._base_max_joint_step = np.asarray(config.max_angvel, dtype=float) * self._base_control_dt
        self._last_send_time: float | None = None
        self._smoothed_send_dt = self._base_control_dt
        self._last_effective_control_dt = self._base_control_dt
        self._last_effective_step_scale = 1.0
        self._state_lock = threading.RLock()
        self._stream_target_qpos: np.ndarray | None = None
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

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.arm is not None:
            return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
        self._auto_load_active_tool_profile()
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

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        with self._state_lock:
            if self._current_qpos is None:
                self._current_qpos = np.zeros(6, dtype=float)
        command_start_time = time.time()
        max_joint_step = self._resolve_effective_max_joint_step(
            now=command_start_time,
            update_state=True,
        )

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
            qpos_target = self._solve_cartesian_command(command, max_joint_step=max_joint_step)
        else:
            raise ValueError(
                f"{self.name} does not support command_type '{command.command_type}'."
            )

        if self.arm is not None and self.config.background_streaming_enabled:
            with self._state_lock:
                self._stream_target_qpos = qpos_target.copy()
                self._last_target_qpos = qpos_target.copy()
                self._last_command = command.copy()
                self._last_command_timestamp = command_start_time
            return

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
            from rm65_inspire_tac.ik import SingleArmIKSolver

            last_error: Exception | None = None
            for candidate in self._candidate_tcp_frames(tcp_frame):
                try:
                    self._solvers[tcp_frame] = SingleArmIKSolver(
                        urdf_path=self.config.arm_urdf_path,
                        eef_name=candidate,
                        ik_tuning=self._ik_tuning,
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                raise ValueError(
                    f"{self.name} could not resolve tcp_frame '{tcp_frame}' against the arm URDF."
                ) from last_error
        return self._solvers[tcp_frame]

    def _candidate_tcp_frames(self, tcp_frame: str) -> tuple[str, ...]:
        tcp_frame = canonical_arm_controlled_frame(tcp_frame)
        if tcp_frame == ARM_ENDPOINT_FRAME:
            return (rm65_urdf_frame_for_controlled_frame(tcp_frame), HAND_BASE_TCP_FRAME)
        if tcp_frame in {TOOL_TCP_FRAME, HAND_BASE_TCP_FRAME}:
            return (HAND_BASE_TCP_FRAME, "arm_link6")
        return (tcp_frame, HAND_BASE_TCP_FRAME, "arm_link6")

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
        if not self.config.connect_hardware and self.config.dry_run_fast:
            qpos_target = solver.solve_iterative(
                target_pose=target_pose,
                q_init=current_qpos,
                max_iterations=int(self.config.dry_run_max_iterations),
                max_joint_step=np.asarray(self.config.dry_run_max_joint_step, dtype=float),
                damping=float(self.config.dry_run_damping),
                tolerance=1e-5,
            )
            return self._apply_jump_gate(
                solver=solver,
                target_pose=target_pose,
                current_qpos=current_qpos,
                candidate_qpos=np.asarray(qpos_target, dtype=float),
            )
        qpos_target = solver.solve_step(
            target_pose=target_pose,
            current_qpos=current_qpos,
            max_joint_step=command_max_joint_step,
        )
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
                if self.arm is not None and not np.allclose(next_qpos, current_qpos, atol=1e-10):
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
        jump_exceeded, per_joint_delta_deg, weighted_jump_deg = joint_jump_exceeds(
            reference_q=current_qpos,
            candidate_q=candidate_qpos,
            joint_distance_weights=self._joint_distance_weights,
            per_joint_jump_deg=per_joint_limit,
            weighted_jump_threshold_deg=weighted_limit,
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
            "ik_joint_jump_max_deg": float(np.max(per_joint_delta_deg)),
            "ik_joint_jump_weighted_deg": float(weighted_jump_deg),
            "ik_joint_jump_threshold_deg": float(weighted_limit),
            "ik_release_ready": release_ready,
            "configured_tcp_enabled": self._configured_tcp_enabled,
        }

        should_hold = bool(small_pose and jump_exceeded and not release_ready)
        if self._jump_gate_active and not should_hold and release_ready:
            metadata["ik_jump_gate_status"] = "released"
        elif self._jump_gate_active and should_hold:
            metadata["ik_jump_gate_status"] = "holding"
        elif should_hold:
            metadata["ik_jump_gate_status"] = "hold_entered"

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

    def refresh_state_from_hardware(self) -> dict[str, np.ndarray | EffectorCommand | None | int | float | str]:
        if self.arm is None:
            return self.get_state()
        raw_result = -1
        joint_now = None
        pose_now = None
        arm_err = None
        sys_err = None
        parsed_code: int | None = None
        parsed_text = ""
        qpos_from_joint_now = None
        for attempt in range(max(1, int(self.config.hardware_read_retries) + 1)):
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
            if attempt + 1 < max(1, int(self.config.hardware_read_retries) + 1):
                time.sleep(float(self.config.hardware_read_retry_delay_sec))
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
        current_tcp_pose = None
        if current_qpos is not None:
            try:
                current_tcp_pose = self.compute_tcp_pose(qpos=current_qpos)
            except Exception:
                current_tcp_pose = None
        tcp_pose_flange = self._active_tcp_pose_flange_m_rad()
        tcp_delta = tcp_endpoint_delta(tcp_pose_flange)
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
            "effective_control_dt_sec": float(self._last_effective_control_dt),
            "effective_step_scale": float(self._last_effective_step_scale),
            "ik_metadata": dict(self._last_ik_metadata),
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
