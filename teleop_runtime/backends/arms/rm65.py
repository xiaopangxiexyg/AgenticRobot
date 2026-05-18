from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
import time

from rm65_inspire_tac.ik_tuning import (
    RM65IKTuningConfig,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
)

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand
from ...tcp import HAND_BASE_TCP_FRAME, canonical_tcp_frame


def _load_rm65_sdk():
    try:
        from hardware.rm65.robotic_arm_package.robotic_arm import Arm, RM65
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


@dataclass
class RM65ArmBackendConfig:
    arm_ip: str
    side: str
    effector_name: str | None = None
    default_tcp_frame: str = "hand_base"
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
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        self.default_tcp_frame = canonical_tcp_frame(self.default_tcp_frame)
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
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)


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

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.arm is not None:
            return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
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
            raw_send_result = self.arm.Movej_CANFD(qpos_target / np.pi * 180.0, False)
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
        tcp_frame = canonical_tcp_frame(tcp_frame)
        if tcp_frame == HAND_BASE_TCP_FRAME:
            return (HAND_BASE_TCP_FRAME, "arm_link6")
        return (tcp_frame, HAND_BASE_TCP_FRAME, "arm_link6")

    def _solve_cartesian_command(
        self,
        command: EffectorCommand,
        *,
        max_joint_step: np.ndarray | None = None,
    ) -> np.ndarray:
        tcp_frame = canonical_tcp_frame(command.metadata.get("tcp_frame", self.config.default_tcp_frame))
        solver = self._solver_for_tcp_frame(tcp_frame)
        target_pose = np.asarray(command.value, dtype=float)
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
                    raw_send_result = self.arm.Movej_CANFD(next_qpos / np.pi * 180.0, False)
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
        return {
            "current_qpos": current_qpos,
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
        tcp_frame = canonical_tcp_frame(self.config.default_tcp_frame if tcp_frame is None else tcp_frame)
        solver = self._solver_for_tcp_frame(tcp_frame)
        return np.asarray(solver.compute_pose(qpos_array), dtype=float)

    def close(self) -> None:
        if self._stream_thread is not None:
            self._stream_stop_event.set()
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        if self.arm is None:
            return
        try:
            self.arm.RM_API_UnInit()
        finally:
            try:
                self.arm.Arm_Socket_Close()
            finally:
                self.arm = None
