from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from robot_motion.runtime import EffectorBackend, EffectorCommand
from robot_motion.rm65 import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)

from .scene import MujocoScene


@dataclass
class MuJoCoArmBackendConfig:
    side: str
    effector_name: str | None = None
    controlled_body: str | None = None
    initial_qpos: tuple[float, ...] | None = None
    tool_profile: str | Path | None = None

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("MuJoCoArmBackendConfig.side must be 'left' or 'right'.")
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if self.initial_qpos is not None and len(self.initial_qpos) != 6:
            raise ValueError("initial_qpos must contain 6 joint values.")
        if self.initial_qpos is not None and not np.all(
            np.isfinite(np.asarray(self.initial_qpos, dtype=float))
        ):
            raise ValueError("initial_qpos must contain finite joint values.")
        if self.initial_qpos is not None:
            initial = np.asarray(self.initial_qpos, dtype=float)
            if np.any(initial < np.asarray(RM65_ABSOLUTE_Q_MIN_RAD)) or np.any(
                initial > np.asarray(RM65_ABSOLUTE_Q_MAX_RAD)
            ):
                raise ValueError("initial_qpos exceeds canonical RM65 joint limits.")


@dataclass
class MuJoCoGripperBackendConfig:
    gripper_name: str | None = None
    effector_name: str = "gripper"
    side: str | None = None
    command_dim: int = 1
    hand_vector_length: int = 6
    initial_close: float | None = None

    def __post_init__(self) -> None:
        if self.side is not None and self.side not in {"left", "right"}:
            raise ValueError("MuJoCoGripperBackendConfig.side must be 'left', 'right', or None.")
        if int(self.command_dim) < 0:
            raise ValueError("command_dim must be non-negative.")
        if int(self.hand_vector_length) <= 0:
            raise ValueError("hand_vector_length must be positive.")
        if int(self.command_dim) >= int(self.hand_vector_length):
            raise ValueError("command_dim must be inside hand_vector_length.")
        if self.initial_close is not None:
            close = float(self.initial_close)
            if close < 0.0 or close > 1.0:
                raise ValueError("initial_close must be in [0, 1].")


class MuJoCoArmBackend(EffectorBackend):
    """Joint-target MuJoCo sink compatible with the common runtime contract.

    Cartesian pose planning belongs to the group-level canonical RM65 planner,
    which has both arm targets and collision state. This per-effector backend
    only applies already-planned joint vectors and reports MuJoCo state.
    """

    def __init__(
        self,
        scene: MujocoScene,
        config: MuJoCoArmBackendConfig,
    ):
        self.scene = scene
        self.config = config
        self._last_command: EffectorCommand | None = None
        self._last_target_qpos: np.ndarray | None = None
        self._last_command_timestamp: float | None = None
        self._endpoint_to_tool: np.ndarray | None = None
        self._tool_tcp_pose_flange_m_rad: np.ndarray | None = None
        if config.side not in self.scene.arm_specs:
            raise KeyError(f"Scene has no arm spec for side '{config.side}'.")
        if config.controlled_body is not None:
            spec = self.scene.arm_specs[config.side]
            self.scene.arm_specs[config.side] = type(spec)(
                side=spec.side,
                joint_names=spec.joint_names,
                controlled_body=str(config.controlled_body),
                effector_name=spec.effector_name,
            )
        if config.tool_profile is not None:
            profile = load_rm65_tool_profile(config.tool_profile, side=config.side)
            self._endpoint_to_tool = endpoint_to_tool_transform(profile)
            self._tool_tcp_pose_flange_m_rad = tool_tcp_pose_flange_m_rad(profile)

    @property
    def name(self) -> str:
        assert self.config.effector_name is not None
        return str(self.config.effector_name)

    def start(self) -> None:
        if self.config.initial_qpos is not None:
            self.scene.set_arm_qpos(self.config.side, self.config.initial_qpos)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for {command.effector_name!r}.")
        if command.command_type == "joint_vector":
            target = np.asarray(command.value, dtype=float).reshape(6)
            if not np.all(np.isfinite(target)):
                raise ValueError(f"{self.name} joint_vector must contain finite values.")
            if np.any(target < np.asarray(RM65_ABSOLUTE_Q_MIN_RAD)) or np.any(
                target > np.asarray(RM65_ABSOLUTE_Q_MAX_RAD)
            ):
                raise ValueError(f"{self.name} joint_vector exceeds canonical RM65 joint limits.")
            self.scene.set_arm_qpos(self.config.side, target)
        elif command.command_type == "cartesian_pose":
            raise ValueError(
                f"{self.name} is a joint-vector sink and cannot solve cartesian_pose per arm. "
                "Run the coupled robot_motion.rm65 canonical planner first, then dispatch its "
                "left/right joint_vector outputs."
            )
        else:
            raise ValueError(f"{self.name} does not support command_type={command.command_type!r}.")

        self._last_command = command.copy()
        self._last_target_qpos = target.copy()
        self._last_command_timestamp = time.time()

    def hold_motion(self, reason: str = "manual hold") -> None:
        del reason
        self._last_target_qpos = None

    def reset_motion_state(self, qpos: Sequence[float] | np.ndarray | None = None) -> None:
        if qpos is not None:
            self.scene.set_arm_qpos(self.config.side, np.asarray(qpos, dtype=float).reshape(6))
        self._last_command = None
        self._last_target_qpos = None
        self._last_command_timestamp = None

    def _arm_base_body(self) -> str:
        candidates = (
            f"{self.config.side}_arm_base_link",
            "arm_base_link",
            "base_link",
        )
        for name in candidates:
            if self.scene.has_body(name):
                return name
        raise KeyError(
            f"Could not find the {self.config.side} arm base body; tried={candidates}."
        )

    def compute_endpoint_pose(
        self,
        qpos: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return arm-endpoint pose in the canonical arm-base frame."""

        body_name = self.scene.arm_specs[self.config.side].controlled_body
        if qpos is None:
            world_from_endpoint = self.scene.body_pose(body_name)
            world_from_base = self.scene.body_pose(self._arm_base_body())
            return np.linalg.inv(world_from_base) @ world_from_endpoint
        current = self.scene.get_arm_qpos(self.config.side)
        try:
            self.scene.set_arm_qpos(self.config.side, np.asarray(qpos, dtype=float).reshape(6))
            world_from_endpoint = self.scene.body_pose(body_name)
            world_from_base = self.scene.body_pose(self._arm_base_body())
            return np.linalg.inv(world_from_base) @ world_from_endpoint
        finally:
            self.scene.set_arm_qpos(self.config.side, current)

    def compute_tcp_pose(
        self,
        qpos: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return tool-TCP pose in arm-base coordinates using an explicit profile."""

        if self._endpoint_to_tool is None:
            raise RuntimeError(
                f"{self.name} has no explicit tool_profile; refusing to label arm_link6 as tool_tcp."
            )
        return self.compute_endpoint_pose(qpos) @ self._endpoint_to_tool

    def get_state(self) -> dict[str, object]:
        qpos = self.scene.get_arm_qpos(self.config.side)
        endpoint_pose = self.compute_endpoint_pose()
        tcp_pose = None if self._endpoint_to_tool is None else endpoint_pose @ self._endpoint_to_tool
        return {
            "current_qpos": qpos.copy(),
            "current_tcp_pose": None if tcp_pose is None else tcp_pose.copy(),
            "current_endpoint_pose": endpoint_pose.copy(),
            "current_endpoint_pose_base": endpoint_pose.copy(),
            "tool_tcp_pose_flange_m_rad": (
                None
                if self._tool_tcp_pose_flange_m_rad is None
                else self._tool_tcp_pose_flange_m_rad.copy()
            ),
            "tcp_feedback_available": tcp_pose is not None,
            "pose_reference_frame": f"{self.config.side}_arm_base",
            "last_target_qpos": None if self._last_target_qpos is None else self._last_target_qpos.copy(),
            "last_command": None if self._last_command is None else self._last_command.copy(),
            "last_command_timestamp": self._last_command_timestamp,
            "backend": "mujoco_joint_sink",
            "controlled_body": self.scene.arm_specs[self.config.side].controlled_body,
            "tool_profile": None if self.config.tool_profile is None else str(self.config.tool_profile),
        }


class MuJoCoGripperBackend(EffectorBackend):
    """Normalized gripper backend backed by a shared :class:`MujocoScene`."""

    def __init__(self, scene: MujocoScene, config: MuJoCoGripperBackendConfig):
        self.scene = scene
        self.config = config
        self._gripper_name = self.scene._resolve_gripper_name(config.gripper_name)
        self._last_command: EffectorCommand | None = None
        self._last_target_close: float | None = None
        self._last_target_qpos: np.ndarray | None = None
        self._last_command_timestamp: float | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    @property
    def gripper_name(self) -> str:
        return self._gripper_name

    def start(self) -> None:
        if self.config.initial_close is not None:
            self.scene.set_gripper_close(self.gripper_name, float(self.config.initial_close))

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for {command.effector_name!r}.")
        if command.command_type == "gripper_scalar":
            close = self._extract_close_scalar(command.value)
            self.scene.set_gripper_close(self.gripper_name, close)
            target_qpos = self.scene.get_gripper_qpos(self.gripper_name)
        elif command.command_type == "joint_vector":
            values = np.asarray(command.value, dtype=float).reshape(-1)
            spec = self.scene.gripper_specs[self.gripper_name]
            if values.shape == (len(spec.joint_names),):
                self.scene.set_gripper_qpos(self.gripper_name, values)
                target_qpos = self.scene.get_gripper_qpos(self.gripper_name)
                close = self.scene.get_gripper_close(self.gripper_name)
            else:
                close = self._extract_close_scalar(values)
                self.scene.set_gripper_close(self.gripper_name, close)
                target_qpos = self.scene.get_gripper_qpos(self.gripper_name)
        else:
            raise ValueError(
                f"{self.name} supports command_type='gripper_scalar' or 'joint_vector', "
                f"got {command.command_type!r}."
            )

        self._last_command = command.copy()
        self._last_target_close = float(close)
        self._last_target_qpos = target_qpos.copy()
        self._last_command_timestamp = time.time()

    def _extract_close_scalar(self, value: object) -> float:
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.shape == (1,):
            return float(np.clip(float(values[0]), 0.0, 1.0))
        if values.shape == (int(self.config.hand_vector_length),):
            return float(
                np.clip(float(values[int(self.config.command_dim)]), 0.0, 1.0)
            )
        raise ValueError(
            f"{self.name} expects scalar, gripper qpos, or hand vector length "
            f"{int(self.config.hand_vector_length)}, got shape {values.shape}."
        )

    def hold_motion(self, reason: str = "manual hold") -> None:
        del reason
        self._last_target_close = None
        self._last_target_qpos = None

    def reset_motion_state(self, qpos: Sequence[float] | np.ndarray | None = None) -> None:
        if qpos is not None:
            values = np.asarray(qpos, dtype=float).reshape(-1)
            if values.shape == (1,):
                self.scene.set_gripper_close(self.gripper_name, float(values[0]))
            else:
                self.scene.set_gripper_qpos(self.gripper_name, values)
        self._last_command = None
        self._last_target_close = None
        self._last_target_qpos = None
        self._last_command_timestamp = None

    def get_state(self) -> dict[str, object]:
        return {
            "gripper_name": self.gripper_name,
            "current_close": float(self.scene.get_gripper_close(self.gripper_name)),
            "current_qpos": self.scene.get_gripper_qpos(self.gripper_name).copy(),
            "last_target_close": self._last_target_close,
            "last_target_qpos": None if self._last_target_qpos is None else self._last_target_qpos.copy(),
            "last_command": None if self._last_command is None else self._last_command.copy(),
            "last_command_timestamp": self._last_command_timestamp,
            "backend": "mujoco_gripper",
        }
