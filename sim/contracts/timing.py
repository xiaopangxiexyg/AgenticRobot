from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from robot_motion.position_control import (
    DEFAULT_POSITION_SAMPLE_HZ,
    PositionControlRates,
)

from .versions import TIMING_SPEC_VERSION, require_version


DEFAULT_MUJOCO_PHYSICS_HZ = 1000.0
DEFAULT_SIM_CAMERA_HZ = 30.0


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class TimingSpec:
    """Independent clocks; no integer-frequency-ratio assumption is allowed."""

    schema_version: str = TIMING_SPEC_VERSION
    sample_hz: float = DEFAULT_POSITION_SAMPLE_HZ
    policy_hz: float = DEFAULT_POSITION_SAMPLE_HZ
    reference_hz: float = PositionControlRates().reference_hz
    ik_hz: float = PositionControlRates().ik_hz
    command_hz: float = PositionControlRates().command_hz
    physics_hz: float = DEFAULT_MUJOCO_PHYSICS_HZ
    camera_hz: float = DEFAULT_SIM_CAMERA_HZ
    interpolation: str = "timestamp_linear"
    command_hold: str = "zero_order_hold_between_command_ticks"
    parameter_source: str = "project_default_pending_real_system_id"

    def __post_init__(self) -> None:
        require_version(self.schema_version, TIMING_SPEC_VERSION, field="timing spec version")
        for field in (
            "sample_hz",
            "policy_hz",
            "reference_hz",
            "ik_hz",
            "command_hz",
            "physics_hz",
            "camera_hz",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        PositionControlRates(
            sample_hz=self.sample_hz,
            reference_hz=self.reference_hz,
            ik_hz=self.ik_hz,
            command_hz=self.command_hz,
        )
        if self.physics_hz < self.command_hz:
            raise ValueError("physics_hz must be >= command_hz.")
        if self.interpolation != "timestamp_linear":
            raise ValueError("TimingSpec v1 requires timestamp_linear interpolation.")
        if self.command_hold != "zero_order_hold_between_command_ticks":
            raise ValueError("TimingSpec v1 requires zero-order hold between commands.")

    @property
    def physics_timestep_sec(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def position_control_rates(self) -> PositionControlRates:
        return PositionControlRates(
            sample_hz=self.sample_hz,
            reference_hz=self.reference_hz,
            ik_hz=self.ik_hz,
            command_hz=self.command_hz,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_hz": self.sample_hz,
            "policy_hz": self.policy_hz,
            "reference_hz": self.reference_hz,
            "ik_hz": self.ik_hz,
            "command_hz": self.command_hz,
            "physics_hz": self.physics_hz,
            "physics_timestep_sec": self.physics_timestep_sec,
            "camera_hz": self.camera_hz,
            "interpolation": self.interpolation,
            "command_hold": self.command_hold,
            "parameter_source": self.parameter_source,
            "independent_clocks": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimingSpec":
        rates = PositionControlRates()
        return cls(
            schema_version=str(payload.get("schema_version", TIMING_SPEC_VERSION)),
            sample_hz=float(payload.get("sample_hz", rates.sample_hz)),
            policy_hz=float(payload.get("policy_hz", rates.sample_hz)),
            reference_hz=float(payload.get("reference_hz", rates.reference_hz)),
            ik_hz=float(payload.get("ik_hz", rates.ik_hz)),
            command_hz=float(payload.get("command_hz", rates.command_hz)),
            physics_hz=float(payload.get("physics_hz", DEFAULT_MUJOCO_PHYSICS_HZ)),
            camera_hz=float(payload.get("camera_hz", DEFAULT_SIM_CAMERA_HZ)),
            interpolation=str(payload.get("interpolation", "timestamp_linear")),
            command_hold=str(
                payload.get("command_hold", "zero_order_hold_between_command_ticks")
            ),
            parameter_source=str(
                payload.get("parameter_source", "project_default_pending_real_system_id")
            ),
        )


__all__ = ["DEFAULT_MUJOCO_PHYSICS_HZ", "DEFAULT_SIM_CAMERA_HZ", "TimingSpec"]
