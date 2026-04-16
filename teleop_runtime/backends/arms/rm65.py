from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand


def _load_rm65_sdk():
    try:
        from hardware.rm65.robotic_arm_package.robotic_arm import Arm, RM65
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RM65ArmBackend requires the RM65 SDK to be importable."
        ) from exc
    return Arm, RM65


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
    initial_qpos: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if len(self.max_angvel) != 6:
            raise ValueError("max_angvel must contain 6 joint velocity limits.")
        if self.initial_qpos is not None and len(self.initial_qpos) != 6:
            raise ValueError("initial_qpos must contain exactly 6 joint values.")


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
        self._max_joint_step = np.asarray(config.max_angvel, dtype=float) * float(config.control_dt)

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.arm is not None:
            return
        Arm, RM65 = _load_rm65_sdk()
        self.arm = Arm(RM65, self.config.arm_ip)
        try:
            result_code, joint_now, _, _, _ = self.arm.Get_Current_Arm_State()
        except Exception:
            result_code = -1
            joint_now = None
        if result_code == 0 and joint_now is not None:
            self._current_qpos = np.asarray(joint_now, dtype=float) / 180.0 * np.pi
        elif self._current_qpos is None:
            self._current_qpos = np.zeros(6, dtype=float)

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or (self.arm is not None)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        if self._current_qpos is None:
            self._current_qpos = np.zeros(6, dtype=float)

        if command.command_type == "joint_vector":
            qpos_target = np.asarray(command.value, dtype=float)
            if qpos_target.shape != (6,):
                raise ValueError(
                    f"{self.name} expects joint_vector with shape (6,), got {qpos_target.shape}."
                )
            delta = np.clip(qpos_target - self._current_qpos, -self._max_joint_step, self._max_joint_step)
            qpos_target = self._current_qpos + delta
        elif command.command_type == "cartesian_pose":
            qpos_target = self._solve_cartesian_command(command)
        else:
            raise ValueError(
                f"{self.name} does not support command_type '{command.command_type}'."
            )

        if self.arm is not None:
            self.arm.Movej_CANFD(qpos_target / np.pi * 180.0, False)

        self._current_qpos = qpos_target.copy()
        self._last_target_qpos = qpos_target.copy()
        self._last_command = command.copy()

    def _solver_for_tcp_frame(self, tcp_frame: str):
        if tcp_frame not in self._solvers:
            from rm65_inspire_tac.ik import SingleArmIKSolver

            last_error: Exception | None = None
            for candidate in self._candidate_tcp_frames(tcp_frame):
                try:
                    self._solvers[tcp_frame] = SingleArmIKSolver(
                        urdf_path=self.config.arm_urdf_path,
                        eef_name=candidate,
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
        if tcp_frame == "rm_eef":
            return ("rm_eef", "arm_link6", "hand_base")
        if tcp_frame == "hand_base":
            return ("hand_base", "arm_link6")
        return (tcp_frame, "hand_base", "arm_link6")

    def _solve_cartesian_command(self, command: EffectorCommand) -> np.ndarray:
        tcp_frame = str(command.metadata.get("tcp_frame", self.config.default_tcp_frame))
        solver = self._solver_for_tcp_frame(tcp_frame)
        qpos_target = solver.solve_step(
            target_pose=np.asarray(command.value, dtype=float),
            current_qpos=self._current_qpos,
            max_joint_step=self._max_joint_step,
        )
        return np.asarray(qpos_target, dtype=float)

    def get_state(self) -> dict[str, np.ndarray | EffectorCommand | None]:
        return {
            "current_qpos": None if self._current_qpos is None else self._current_qpos.copy(),
            "last_target_qpos": None
            if self._last_target_qpos is None
            else self._last_target_qpos.copy(),
            "last_command": None if self._last_command is None else self._last_command.copy(),
        }

    def compute_tcp_pose(self, tcp_frame: str | None = None) -> np.ndarray:
        if self._current_qpos is None:
            raise RuntimeError(f"{self.name} does not have a current joint state yet.")
        tcp_frame = self.config.default_tcp_frame if tcp_frame is None else str(tcp_frame)
        solver = self._solver_for_tcp_frame(tcp_frame)
        return np.asarray(solver.compute_pose(self._current_qpos), dtype=float)

    def close(self) -> None:
        if self.arm is None:
            return
        try:
            self.arm.RM_API_UnInit()
        finally:
            try:
                self.arm.Arm_Socket_Close()
            finally:
                self.arm = None
