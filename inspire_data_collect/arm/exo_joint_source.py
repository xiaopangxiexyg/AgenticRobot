from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common import ObservationChannel

ARM_DOF = 7


@dataclass(frozen=True)
class ExoJointSourceConfig:
    side: str = "right"
    channel_name: str = "arm_exo_joint_qpos"
    source_name: str = "arm_exo_dynamixel"
    port: str = ""
    baudrate: int = 57600


class ExoJointSource:
    def __init__(self, config: ExoJointSourceConfig):
        self.config = config
        self._driver: Any | None = None
        self._startup_error: str | None = None
        self._last_value = np.zeros(ARM_DOF, dtype=np.float32)
        self._effective_port = str(config.port)
        self._effective_baudrate = int(config.baudrate)

    def start(self) -> None:
        if self._driver is not None:
            return
        from exo_teleop.arm_exo.motors_config import make_default_arm_params
        from exo_teleop.arm_exo.motors_driver import DynamixelDriver

        params = make_default_arm_params()
        if self.config.port:
            params.port = str(self.config.port)
        if self.config.baudrate:
            params.baudrate = int(self.config.baudrate)
        self._effective_port = str(params.port)
        self._effective_baudrate = int(params.baudrate)
        try:
            self._driver = DynamixelDriver(params, read_only=True)
            self._startup_error = None
        except Exception as exc:
            self._driver = None
            self._startup_error = str(exc)

    def is_ready(self) -> bool:
        return self._driver is not None

    def expected_channel_names(self) -> tuple[str, ...]:
        return (self.config.channel_name,)

    def device_info(self) -> dict[str, Any]:
        return {
            "type": "exo_dynamixel",
            "side": str(self.config.side),
            "channel_name": str(self.config.channel_name),
            "source_name": str(self.config.source_name),
            "port": str(self._effective_port),
            "baudrate": int(self._effective_baudrate),
            "arm_dof": int(ARM_DOF),
            "ready": bool(self.is_ready()),
            "startup_error": self._startup_error,
        }

    def _invalid_channel(self, *, error: str) -> ObservationChannel:
        return ObservationChannel(
            channel_name=self.config.channel_name,
            side=self.config.side,
            value=self._last_value.copy(),
            timestamp=time.time(),
            valid=False,
            source_name=self.config.source_name,
            metadata={
                "port": str(self._effective_port),
                "baudrate": int(self._effective_baudrate),
                "read_error": str(error),
            },
        )

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        if self._driver is None:
            return {
                self.config.channel_name: self._invalid_channel(
                    error=self._startup_error or "source_not_started"
                )
            }
        try:
            q, age = self._driver.get_joints_cached()
            value = np.asarray(q, dtype=np.float32).reshape(-1)
            if value.shape != (ARM_DOF,):
                raise ValueError(f"Expected {ARM_DOF} joints, got {value.shape}.")
            self._last_value = value.copy()
            channel = ObservationChannel(
                channel_name=self.config.channel_name,
                side=self.config.side,
                value=value.copy(),
                timestamp=float(time.time() - max(float(age), 0.0)),
                valid=True,
                source_name=self.config.source_name,
                metadata={
                    "port": str(self._effective_port),
                    "baudrate": int(self._effective_baudrate),
                    "cache_age_sec": float(age),
                },
            )
        except Exception as exc:
            channel = self._invalid_channel(error=str(exc))
        return {channel.channel_name: channel}

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None
