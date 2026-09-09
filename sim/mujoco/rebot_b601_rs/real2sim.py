"""Explicit controller-side calibration for the ideal B601 MuJoCo plant.

This profile does not identify passive friction, armature or motor bandwidth.
The controller offsets intentionally do not modify the plant's URDF.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from robot_motion.rebot_b601_rs.mit import _six_finite, _six_non_negative


@dataclass(frozen=True)
class B601Real2SimControllerProfile:
    """Host-held feedforward plus local zero-reference velocity damping.

    Defaults reproduce the operator-selected 2026-09-09 compensation. Source
    feedback is still ideal simulation state, without a sensor-delay claim.
    """

    name: str = "b601_operator_gravity_20260909_v1"
    calibration_source: str = "b601-gravity-candidate:c8556a6f"
    gravity_scale: float = 1.0
    gravity_joint_scales: tuple[float, ...] = (1.0, 0.95, 1.0, 1.0, 1.0, 1.0)
    gravity_joint_offsets_rad: tuple[float, ...] = (0.0,) * 6
    gravity_joint_bias_nm: tuple[float, ...] = (0.0, -0.105, -0.1, -0.05, 0.0, 0.0)
    gravity_coulomb_friction_nm: tuple[float, ...] = (0.0,) * 6
    coulomb_velocity_threshold_rad_s: float = 0.0
    driver_kd_nm_s_per_rad: tuple[float, ...] = (0.20, 0.30, 0.20, 0.15, 0.10, 0.05)

    def __post_init__(self) -> None:
        for name in ("name", "calibration_source"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("gravity_scale", "coulomb_velocity_threshold_rad_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "gravity_joint_scales",
            "gravity_coulomb_friction_nm",
            "driver_kd_nm_s_per_rad",
        ):
            object.__setattr__(
                self, name, _six_non_negative(getattr(self, name), name=name)
            )
        for name in ("gravity_joint_offsets_rad", "gravity_joint_bias_nm"):
            object.__setattr__(self, name, _six_finite(getattr(self, name), name=name))

    def compensate(self, model_gravity: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Apply compensation to gravity already evaluated at q + offset."""
        gravity = np.asarray(_six_finite(model_gravity, name="model_gravity"))
        velocity = np.asarray(_six_finite(qvel, name="qvel"))
        direction = np.where(
            np.abs(velocity) > self.coulomb_velocity_threshold_rad_s,
            np.sign(velocity),
            0.0,
        )
        return (
            self.gravity_scale * np.asarray(self.gravity_joint_scales) * gravity
            + np.asarray(self.gravity_joint_bias_nm)
            + np.asarray(self.gravity_coulomb_friction_nm) * direction
        )

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "calibration_status": "configured_compensation_not_plant_identification",
            "plant": "unchanged_urdf_ideal_effort",
            "unmodeled": [
                "passive_friction",
                "armature",
                "motor_bandwidth",
                "sensor_delay",
                "quantization",
                "transport_delay",
            ],
            "local_damping_reference": "zero_velocity",
            "startup_state_machine_parity": False,
            "hardware_safety_claim": False,
        }
