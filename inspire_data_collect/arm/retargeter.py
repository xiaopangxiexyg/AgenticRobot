from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exo_teleop.arm_exo.motors_to_robot import dynamixel_to_robot_arm

from ..common import EffectorCommand, ObservationChannel

ARM_DOF = 7
FR3_Q_LOW = np.deg2rad(np.array([-166, -105, -166, -176, -165, 25, -175], dtype=np.float64))
FR3_Q_HIGH = np.deg2rad(np.array([166, 105, 166, -7, 165, 265, 175], dtype=np.float64))


def safe_clip(
    robot_q_rad: np.ndarray,
    *,
    shrink_deg: float = 10.0,
) -> tuple[np.ndarray, bool, tuple[np.ndarray, np.ndarray]]:
    q = np.asarray(robot_q_rad, dtype=np.float64).reshape(-1)
    if q.shape != (ARM_DOF,):
        raise ValueError(f"robot_q_rad dim mismatch: got {q.shape}, expect {(ARM_DOF,)}")
    margin = np.deg2rad(float(shrink_deg))
    q_low_safe = FR3_Q_LOW + margin
    q_high_safe = FR3_Q_HIGH - margin
    if np.any(q_low_safe >= q_high_safe):
        raise ValueError(f"safe limits invalid after shrink_deg={shrink_deg}")
    q_clipped = np.clip(q, q_low_safe, q_high_safe)
    clipped = bool(np.any(np.abs(q_clipped - q) > 1e-12))
    return q_clipped.astype(np.float32), clipped, (q_low_safe.astype(np.float32), q_high_safe.astype(np.float32))


@dataclass(frozen=True)
class FR3ArmRetargeterConfig:
    side: str = "right"
    input_channel_name: str = "arm_exo_joint_qpos"
    effector_name: str = "arm"
    shrink_deg: float = 10.0


class FR3ArmRetargeter:
    def __init__(self, config: FR3ArmRetargeterConfig):
        self.config = config

    def retarget(
        self,
        channels: dict[str, ObservationChannel],
        *,
        timestamp: float,
    ) -> EffectorCommand | None:
        channel = channels.get(self.config.input_channel_name)
        if channel is None or not channel.valid:
            return None
        source_q = np.asarray(channel.value, dtype=np.float32).reshape(-1)
        if source_q.shape != (ARM_DOF,):
            return None
        preclip = np.asarray(dynamixel_to_robot_arm(source_q.tolist()), dtype=np.float32).reshape(-1)
        if preclip.shape != (ARM_DOF,):
            raise ValueError(f"Expected mapped arm q shape {(ARM_DOF,)}, got {preclip.shape}.")
        target, clipped, (safe_q_low, safe_q_high) = safe_clip(
            preclip,
            shrink_deg=float(self.config.shrink_deg),
        )
        clip_mask = np.abs(target - preclip) > 1e-12
        return EffectorCommand(
            effector_name=self.config.effector_name,
            side=self.config.side,
            value=target.astype(np.float32, copy=True),
            timestamp=float(timestamp),
            metadata={
                "input_channel": str(self.config.input_channel_name),
                "source_qpos": source_q.astype(np.float32, copy=True),
                "source_timestamp": float(channel.timestamp),
                "preclip_joint_target": preclip.astype(np.float32, copy=True),
                "clip_mask": clip_mask.astype(bool).tolist(),
                "clipped": bool(clipped),
                "mapping": "dynamixel_to_robot_arm",
                "shrink_deg": float(self.config.shrink_deg),
                "safe_q_low": safe_q_low.astype(np.float32, copy=True),
                "safe_q_high": safe_q_high.astype(np.float32, copy=True),
                "arm_side": str(self.config.side),
            },
        )

