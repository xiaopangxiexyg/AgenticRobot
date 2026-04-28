from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..calibration import TeleopCalibration
from ..interfaces import Retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation
from ..tcp import HAND_BASE_TCP_FRAME
from ..transforms import as_pose_matrix


def _qpos6(value: Sequence[float] | np.ndarray) -> np.ndarray:
    qpos = np.asarray(value, dtype=float).reshape(-1)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"Fixed arm state requires a finite 6D qpos, got {qpos.shape}.")
    return qpos.copy()


@dataclass(frozen=True)
class FixedArmState:
    qpos: np.ndarray
    target_pose_base: np.ndarray
    target_pose_desk: np.ndarray | None = None
    captured_at: float | None = None
    source: str = "hardware_current_qpos_at_home"

    def __post_init__(self) -> None:
        object.__setattr__(self, "qpos", _qpos6(self.qpos))
        object.__setattr__(self, "target_pose_base", as_pose_matrix(self.target_pose_base))
        if self.target_pose_desk is not None:
            object.__setattr__(self, "target_pose_desk", as_pose_matrix(self.target_pose_desk))
        if self.captured_at is not None:
            object.__setattr__(self, "captured_at", float(self.captured_at))


@dataclass
class FixedArmRetargeterConfig:
    side: str = "left"
    effector_name: str | None = None
    output_frame_name: str = "arm_base"
    tcp_frame: str = HAND_BASE_TCP_FRAME

    def __post_init__(self) -> None:
        self.side = str(self.side)
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        self.effector_name = str(self.effector_name)


class FixedArmRetargeter(Retargeter):
    """Retargeter that emits a joint hold command for one arm.

    The command is intentionally shaped like the normal arm command stream: it is
    a `joint_vector` command with source pose metadata, so the collector can fill
    both action pose slots and replay joint-target slots without changing schema.
    """

    def __init__(self, config: FixedArmRetargeterConfig | None = None):
        self.config = config if config is not None else FixedArmRetargeterConfig()
        self._state: FixedArmState | None = None

    @property
    def name(self) -> str:
        return f"fixed_arm_{self.config.side}"

    @property
    def fixed_state(self) -> FixedArmState | None:
        return self._copy_state(self._state)

    def clear_fixed_state(self) -> None:
        self._state = None

    def set_fixed_state(
        self,
        *,
        qpos: Sequence[float] | np.ndarray,
        target_pose_base: Any,
        target_pose_desk: Any | None = None,
        captured_at: float | None = None,
        source: str = "hardware_current_qpos_at_home",
    ) -> None:
        self._state = FixedArmState(
            qpos=np.asarray(qpos, dtype=float),
            target_pose_base=target_pose_base,
            target_pose_desk=target_pose_desk,
            captured_at=captured_at,
            source=str(source),
        )

    def retarget(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration | None = None,
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=observation.timestamp)
        state = self._state
        if state is None:
            command.metadata["skipped"] = f"{self.config.side}_arm_fixed_state_unset"
            return command

        target_pose_desk = state.target_pose_desk
        if target_pose_desk is None and calibration is not None:
            arm = calibration.get_arm_for_side(self.config.side)
            if arm.T_D_B is not None:
                target_pose_desk = arm.T_D_B @ state.target_pose_base

        metadata = {
            "controller": self.name,
            "fixed_arm": True,
            "fixed_side": self.config.side,
            "hold_source": state.source,
            "captured_at": state.captured_at,
            "tcp_mode": HAND_BASE_TCP_FRAME,
            "tcp_frame": self.config.tcp_frame,
            "source_command_type": "fixed_joint_hold",
            "source_target_pose_base": state.target_pose_base.copy(),
            "target_pose_base": state.target_pose_base.copy(),
            "source_target_pose_desk": None if target_pose_desk is None else target_pose_desk.copy(),
            "target_pose_desk": None if target_pose_desk is None else target_pose_desk.copy(),
        }
        command.metadata.update(
            {
                "fixed_arm_sides": [self.config.side],
                "fixed_arm_states": {
                    self.config.side: {
                        "qpos": state.qpos.copy(),
                        "target_pose_base": state.target_pose_base.copy(),
                        "target_pose_desk": None
                        if target_pose_desk is None
                        else target_pose_desk.copy(),
                        "captured_at": state.captured_at,
                        "source": state.source,
                    }
                },
            }
        )
        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="joint_vector",
                value=state.qpos.copy(),
                frame_name=self.config.output_frame_name,
                metadata=metadata,
            )
        )
        return command

    @staticmethod
    def _copy_state(state: FixedArmState | None) -> FixedArmState | None:
        if state is None:
            return None
        return FixedArmState(
            qpos=state.qpos.copy(),
            target_pose_base=state.target_pose_base.copy(),
            target_pose_desk=None if state.target_pose_desk is None else state.target_pose_desk.copy(),
            captured_at=state.captured_at,
            source=state.source,
        )


@dataclass
class SmoothJitterArmRetargeterConfig:
    side: str = "left"
    effector_name: str | None = None
    output_frame_name: str = "arm_base"
    tcp_frame: str = HAND_BASE_TCP_FRAME
    max_translation_m: float = 0.01
    max_rotation_deg: float = 3.0
    target_refresh_sec: float = 1.2
    smoothing_time_constant_sec: float = 0.45
    seed: int | None = None

    def __post_init__(self) -> None:
        self.side = str(self.side)
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        self.effector_name = str(self.effector_name)
        if float(self.max_translation_m) < 0.0:
            raise ValueError("max_translation_m must be >= 0.")
        if float(self.max_rotation_deg) < 0.0:
            raise ValueError("max_rotation_deg must be >= 0.")
        if float(self.target_refresh_sec) <= 0.0:
            raise ValueError("target_refresh_sec must be > 0.")
        if float(self.smoothing_time_constant_sec) < 0.0:
            raise ValueError("smoothing_time_constant_sec must be >= 0.")


class SmoothJitterArmRetargeter(Retargeter):
    """Emit a smooth random cartesian target around a captured arm pose."""

    def __init__(self, config: SmoothJitterArmRetargeterConfig | None = None):
        self.config = config if config is not None else SmoothJitterArmRetargeterConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._anchor_state: FixedArmState | None = None
        self._current_translation = np.zeros(3, dtype=float)
        self._target_translation = np.zeros(3, dtype=float)
        self._current_rotvec = np.zeros(3, dtype=float)
        self._target_rotvec = np.zeros(3, dtype=float)
        self._last_update_at: float | None = None
        self._next_target_at: float | None = None

    @property
    def name(self) -> str:
        return f"smooth_jitter_arm_{self.config.side}"

    @property
    def anchor_state(self) -> FixedArmState | None:
        return FixedArmRetargeter._copy_state(self._anchor_state)

    def clear_anchor_state(self) -> None:
        self._anchor_state = None
        self._current_translation = np.zeros(3, dtype=float)
        self._target_translation = np.zeros(3, dtype=float)
        self._current_rotvec = np.zeros(3, dtype=float)
        self._target_rotvec = np.zeros(3, dtype=float)
        self._last_update_at = None
        self._next_target_at = None

    def set_anchor_state(
        self,
        *,
        qpos: Sequence[float] | np.ndarray,
        target_pose_base: Any,
        target_pose_desk: Any | None = None,
        captured_at: float | None = None,
        source: str = "hardware_current_qpos_at_home",
    ) -> None:
        self._anchor_state = FixedArmState(
            qpos=np.asarray(qpos, dtype=float),
            target_pose_base=target_pose_base,
            target_pose_desk=target_pose_desk,
            captured_at=captured_at,
            source=str(source),
        )
        self._current_translation = np.zeros(3, dtype=float)
        self._target_translation = np.zeros(3, dtype=float)
        self._current_rotvec = np.zeros(3, dtype=float)
        self._target_rotvec = np.zeros(3, dtype=float)
        self._last_update_at = None
        self._next_target_at = None

    def retarget(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration | None = None,
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=observation.timestamp)
        state = self._anchor_state
        if state is None:
            command.metadata["skipped"] = f"{self.config.side}_arm_jitter_anchor_unset"
            return command

        now = time.time() if observation.timestamp is None else float(observation.timestamp)
        target_pose_desk = self._build_jitter_pose_desk(state, now=now, calibration=calibration)
        if calibration is not None:
            target_pose_base = calibration.desk_to_base(target_pose_desk, side=self.config.side)
        else:
            target_pose_base = target_pose_desk.copy()

        metadata = {
            "controller": self.name,
            "jitter_arm": True,
            "fixed_side": self.config.side,
            "hold_source": state.source,
            "captured_at": state.captured_at,
            "tcp_mode": HAND_BASE_TCP_FRAME,
            "tcp_frame": self.config.tcp_frame,
            "source_command_type": "smooth_random_cartesian_jitter",
            "anchor_qpos": state.qpos.copy(),
            "anchor_pose_base": state.target_pose_base.copy(),
            "anchor_pose_desk": None
            if state.target_pose_desk is None
            else state.target_pose_desk.copy(),
            "source_target_pose_base": target_pose_base.copy(),
            "target_pose_base": target_pose_base.copy(),
            "source_target_pose_desk": target_pose_desk.copy(),
            "target_pose_desk": target_pose_desk.copy(),
            "jitter_translation_offset_m": self._current_translation.copy(),
            "jitter_rotation_offset_deg": np.rad2deg(self._current_rotvec).copy(),
            "jitter_target_translation_offset_m": self._target_translation.copy(),
            "jitter_target_rotation_offset_deg": np.rad2deg(self._target_rotvec).copy(),
            "jitter_max_translation_m": float(self.config.max_translation_m),
            "jitter_max_rotation_deg": float(self.config.max_rotation_deg),
            "jitter_target_refresh_sec": float(self.config.target_refresh_sec),
            "jitter_smoothing_time_constant_sec": float(
                self.config.smoothing_time_constant_sec
            ),
        }
        command.metadata.update(
            {
                "jitter_arm_sides": [self.config.side],
                "jitter_arm_states": {
                    self.config.side: {
                        "anchor_qpos": state.qpos.copy(),
                        "anchor_pose_base": state.target_pose_base.copy(),
                        "anchor_pose_desk": None
                        if state.target_pose_desk is None
                        else state.target_pose_desk.copy(),
                        "target_pose_base": target_pose_base.copy(),
                        "target_pose_desk": target_pose_desk.copy(),
                        "translation_offset_m": self._current_translation.copy(),
                        "rotation_offset_deg": np.rad2deg(self._current_rotvec).copy(),
                    }
                },
            }
        )
        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="cartesian_pose",
                value=target_pose_base,
                frame_name=self.config.output_frame_name,
                metadata=metadata,
            )
        )
        return command

    def _build_jitter_pose_desk(
        self,
        state: FixedArmState,
        *,
        now: float,
        calibration: TeleopCalibration | None,
    ) -> np.ndarray:
        if state.target_pose_desk is not None:
            anchor_pose_desk = state.target_pose_desk.copy()
        elif calibration is not None:
            arm = calibration.get_arm_for_side(self.config.side)
            if arm.T_D_B is None:
                raise RuntimeError(f"Arm calibration for side '{self.config.side}' is missing T_D_B.")
            anchor_pose_desk = arm.T_D_B @ state.target_pose_base
        else:
            anchor_pose_desk = state.target_pose_base.copy()
        self._advance_offsets(now)
        return self._apply_offsets(anchor_pose_desk)

    def _advance_offsets(self, now: float) -> None:
        if self._last_update_at is None:
            self._last_update_at = float(now)
            self._next_target_at = float(now)
        if self._next_target_at is None or now >= self._next_target_at:
            self._target_translation = self._sample_translation_offset()
            self._target_rotvec = self._sample_rotation_offset()
            self._next_target_at = float(now) + float(self.config.target_refresh_sec)

        dt = max(0.0, float(now) - float(self._last_update_at))
        tau = float(self.config.smoothing_time_constant_sec)
        alpha = 1.0 if tau <= 1e-9 else 1.0 - float(np.exp(-dt / tau))
        self._current_translation = (
            self._current_translation
            + alpha * (self._target_translation - self._current_translation)
        )
        self._current_rotvec = (
            self._current_rotvec + alpha * (self._target_rotvec - self._current_rotvec)
        )
        self._last_update_at = float(now)

    def _sample_translation_offset(self) -> np.ndarray:
        radius = float(self.config.max_translation_m)
        if radius <= 0.0:
            return np.zeros(3, dtype=float)
        direction = self._rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return np.zeros(3, dtype=float)
        distance = radius * float(self._rng.random() ** (1.0 / 3.0))
        return direction / norm * distance

    def _sample_rotation_offset(self) -> np.ndarray:
        max_angle_rad = float(np.deg2rad(self.config.max_rotation_deg))
        if max_angle_rad <= 0.0:
            return np.zeros(3, dtype=float)
        axis = self._rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return np.zeros(3, dtype=float)
        angle = float(self._rng.uniform(-max_angle_rad, max_angle_rad))
        return axis / norm * angle

    def _apply_offsets(self, anchor_pose: np.ndarray) -> np.ndarray:
        pose = as_pose_matrix(anchor_pose)
        pose[:3, 3] = pose[:3, 3] + self._current_translation
        pose[:3, :3] = pose[:3, :3] @ R.from_rotvec(self._current_rotvec).as_matrix()
        return pose
