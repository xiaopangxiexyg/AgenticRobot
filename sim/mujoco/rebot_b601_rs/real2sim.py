"""Explicit controller-side calibration for the ideal B601 MuJoCo plant.

This profile does not identify passive friction, armature or motor bandwidth.
The controller offsets intentionally do not modify the plant's URDF.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from robot_motion.rebot_b601_rs.mit import _six_finite, _six_non_negative


def audit_b601_probe_trace(payload: Mapping[str, object]) -> dict[str, object]:
    """Audit the existing low-gain probe JSON, without replaying or fitting it.

    This capture format lacks per-sample dispatch receipts and motor feedback
    acquisition timestamps. Even well-formed rows do not qualify it for a
    quantitative paired replay. Never infer motor acquisition time from a poll.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("probe trace must be a mapping")
    if payload.get("mode") not in {"startup-replay", "gravity-calibration"}:
        raise ValueError("only existing startup-replay/gravity-calibration traces are supported")
    if payload.get("active_joint_names") != [f"joint{i}" for i in range(1, 7)]:
        raise ValueError("probe trace must explicitly order all six arm joints")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("probe trace requires a non-empty samples list")
    timestamps: list[float] = []
    previous_sequence = -1
    unverified = 0
    max_velocity = np.zeros(6)
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise TypeError(f"sample {index} must be a mapping")
        sequence = row.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError(f"sample {index}: sequence must be an integer")
        if sequence <= previous_sequence:
            raise ValueError(f"sample {index}: sequence did not advance")
        previous_sequence = sequence
        timestamp = row.get("feedback_timestamp_sec")
        if (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp) or timestamp < 0.0
        ):
            raise ValueError(f"sample {index}: invalid feedback timestamp")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError(f"sample {index}: feedback timestamp did not advance")
        timestamps.append(float(timestamp))
        for field in (
            "qpos_rad", "qvel_fd_rad_s", "conditioned_tau_nm", "dispatch_qref_rad",
        ):
            _six_finite(row.get(field, ()), name=f"sample {index}.{field}")
        for field in ("vendor_kp_nm_per_rad", "vendor_kd_nm_s_per_rad"):
            _six_non_negative(row.get(field, ()), name=f"sample {index}.{field}")
        max_velocity = np.maximum(max_velocity, np.abs(row["qvel_fd_rad_s"]))
        unverified += row.get("freshness_verified") is not True
    gaps = np.diff(timestamps)
    blockers = [
        "probe_format_has_no_per_sample_dispatch_receipt_or_send_timestamp",
        "load_contact_operator_input_and_temperature_are_not_qualified",
        "fit_and_held_out_episode_split_not_established",
    ]
    if unverified or payload.get("freshness_verified") is not True:
        blockers.append("feedback_freshness_unverified")
    identity = {key: payload.get(key) for key in ("config_sha256", "urdf_sha256")}
    if not all(isinstance(v, str) and len(v) == 64 for v in identity.values()):
        blockers.append("missing_config_or_urdf_identity")
    return {
        "status": "QUALITATIVE_TRACE_ONLY",
        "sample_count": len(samples),
        "unverified_sample_count": unverified,
        "poll_observation_span_sec": timestamps[-1] - timestamps[0],
        "poll_interval_sec": {
            "minimum": float(gaps.min()) if gaps.size else None,
            "maximum": float(gaps.max()) if gaps.size else None,
            "median": float(np.median(gaps)) if gaps.size else None,
        },
        "max_abs_fd_velocity_rad_s": max_velocity.tolist(),
        "declared_identity": identity,
        "identity_verified_against_assets": False,
        "source_termination_status": payload.get("status"),
        "blockers": blockers,
        "quantitative_paired_replay_ready": False,
        "hardware_safety_claim": False,
    }


@dataclass(frozen=True)
class B601Real2SimPlantProfile:
    """Explicit experimental plant terms, never inferred from controller gains."""

    name: str = "b601_provisional_plant"
    parameter_source: str = "unidentified_simulation_experiment"
    damping_nm_s_per_rad: tuple[float, ...] = (0.0,) * 6
    frictionloss_nm: tuple[float, ...] = (0.0,) * 6
    armature_kg_m2: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self) -> None:
        for name in ("name", "parameter_source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("damping_nm_s_per_rad", "frictionloss_nm", "armature_kg_m2"):
            object.__setattr__(self, name, _six_non_negative(getattr(self, name), name=name))

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self), "identification_status": "provisional_not_hardware_identified",
            "friction_model": "mujoco_joint_dry_friction_constraint",
            "urdf_mass_com_inertia_modified": False, "hardware_safety_claim": False,
        }


@dataclass(frozen=True)
class B601SimStartupProfile:
    """Explicit simulation lock gains and admission envelope, not hardware defaults."""

    lock_kp_nm_per_rad: tuple[float, ...]
    lock_kd_nm_s_per_rad: tuple[float, ...]
    freeze_sec: float = 0.1
    rise_time_sec: float = 0.1
    max_lock_error_rad: float = 0.02
    settled_velocity_rad_s: float = 0.02

    def __post_init__(self) -> None:
        for name in ("lock_kp_nm_per_rad", "lock_kd_nm_s_per_rad"):
            object.__setattr__(
                self, name, _six_non_negative(getattr(self, name), name=name)
            )
        for name in (
            "freeze_sec",
            "rise_time_sec",
            "max_lock_error_rad",
            "settled_velocity_rad_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


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

    @classmethod
    def raw_model_baseline(cls) -> B601Real2SimControllerProfile:
        """Uncalibrated gravity control; retain the same motor-local damping."""
        return cls(
            name="b601_raw_model_gravity_baseline",
            calibration_source="unmodified_urdf_controller_baseline",
            gravity_joint_scales=(1.0,) * 6,
            gravity_joint_bias_nm=(0.0,) * 6,
        )

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
