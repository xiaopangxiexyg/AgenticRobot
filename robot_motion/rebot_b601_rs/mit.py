"""Backend-neutral B601-RS MIT reference and torque decomposition.

The functions in this module perform no I/O and own no actuator.  Local
backends remain responsible for command admission, final torque limiting,
watchdogs, fault latching and transport-specific MIT semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .model import B601_RS_ARM_JOINT_NAMES

B601_RS_JOINT_MIT_FRAME = "b601_rs/joint_mit_reference"
_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _finite_timestamp(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and non-negative")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _six_finite(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain six finite values")
    return tuple(float(value) for value in array)


def _six_non_negative(
    values: Sequence[float],
    *,
    name: str,
    strictly_positive: bool = False,
) -> tuple[float, ...]:
    normalized = _six_finite(values, name=name)
    array = np.asarray(normalized, dtype=float)
    invalid = array <= 0.0 if strictly_positive else array < 0.0
    if np.any(invalid):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must contain six {qualifier} values")
    return normalized


@dataclass(frozen=True)
class B601MitReference:
    """Timestamped local joint reference plus a torque overlay.

    The intended hardware path is a single-host, wired control process.
    ``source_timestamp_sec`` and ``receive_timestamp_sec`` therefore describe
    producer and driver-ingress observations in a host-local monotonic clock;
    they are not a network clock-synchronization contract.  A backend must use
    its own local acceptance clock for the command watchdog.

    ``frame_id`` identifies the six-joint command convention and ordering.  It
    is retained even for a wired backend so a configuration error cannot be
    mistaken for a valid B601 torque reference.
    """

    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    qpos_rad: tuple[float, ...]
    qvel_rad_s: tuple[float, ...] = _ZERO_SIX
    joint_torque_overlay_nm: tuple[float, ...] = _ZERO_SIX
    frame_id: str = B601_RS_JOINT_MIT_FRAME
    valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        frame_id = str(self.frame_id).strip()
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(
            self,
            "sequence",
            _non_negative_integer(self.sequence, name="sequence"),
        )
        object.__setattr__(
            self,
            "source_timestamp_sec",
            _finite_timestamp(
                self.source_timestamp_sec,
                name="source_timestamp_sec",
            ),
        )
        object.__setattr__(
            self,
            "receive_timestamp_sec",
            _finite_timestamp(
                self.receive_timestamp_sec,
                name="receive_timestamp_sec",
            ),
        )
        object.__setattr__(
            self,
            "qpos_rad",
            _six_finite(self.qpos_rad, name="qpos_rad"),
        )
        object.__setattr__(
            self,
            "qvel_rad_s",
            _six_finite(self.qvel_rad_s, name="qvel_rad_s"),
        )
        object.__setattr__(
            self,
            "joint_torque_overlay_nm",
            _six_finite(
                self.joint_torque_overlay_nm,
                name="joint_torque_overlay_nm",
            ),
        )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "valid", self.valid)


@dataclass(frozen=True)
class B601MitControlProfile:
    """Locally selected gains and bounds for one B601 MIT control loop.

    A profile is deliberately separate from :class:`B601MitReference`; policy
    or teleoperation samples therefore cannot silently retune a local servo.
    """

    name: str
    position_kp_nm_per_rad: tuple[float, ...]
    damping_kd_nm_s_per_rad: tuple[float, ...]
    max_abs_torque_nm: tuple[float, ...]
    max_torque_rate_nm_s: tuple[float, ...]
    max_abs_overlay_torque_nm: tuple[float, ...] = _ZERO_SIX
    max_abs_reference_velocity_rad_s: tuple[float, ...] = (2.0,) * 6
    max_abs_state_velocity_rad_s: tuple[float, ...] = (2.0,) * 6
    max_abs_state_acceleration_rad_s2: tuple[float, ...] = (100.0,) * 6
    joint_names: tuple[str, ...] = B601_RS_ARM_JOINT_NAMES
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("name must not be empty")
        joint_names = tuple(str(value) for value in self.joint_names)
        if joint_names != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joint_names must match the B601-RS arm order")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "position_kp_nm_per_rad",
            _six_non_negative(
                self.position_kp_nm_per_rad,
                name="position_kp_nm_per_rad",
            ),
        )
        object.__setattr__(
            self,
            "damping_kd_nm_s_per_rad",
            _six_non_negative(
                self.damping_kd_nm_s_per_rad,
                name="damping_kd_nm_s_per_rad",
            ),
        )
        object.__setattr__(
            self,
            "max_abs_torque_nm",
            _six_non_negative(
                self.max_abs_torque_nm,
                name="max_abs_torque_nm",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_torque_rate_nm_s",
            _six_non_negative(
                self.max_torque_rate_nm_s,
                name="max_torque_rate_nm_s",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_overlay_torque_nm",
            _six_non_negative(
                self.max_abs_overlay_torque_nm,
                name="max_abs_overlay_torque_nm",
            ),
        )
        object.__setattr__(
            self,
            "max_abs_reference_velocity_rad_s",
            _six_non_negative(
                self.max_abs_reference_velocity_rad_s,
                name="max_abs_reference_velocity_rad_s",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_state_velocity_rad_s",
            _six_non_negative(
                self.max_abs_state_velocity_rad_s,
                name="max_abs_state_velocity_rad_s",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_state_acceleration_rad_s2",
            _six_non_negative(
                self.max_abs_state_acceleration_rad_s2,
                name="max_abs_state_acceleration_rad_s2",
                strictly_positive=True,
            ),
        )
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True)
class B601MitTorqueBreakdown:
    """Auditable components before backend-level total-torque conditioning."""

    position_torque_nm: tuple[float, ...]
    damping_torque_nm: tuple[float, ...]
    gravity_torque_nm: tuple[float, ...]
    overlay_requested_nm: tuple[float, ...]
    overlay_effective_nm: tuple[float, ...]
    requested_torque_nm: tuple[float, ...]
    overlay_limited: bool


def compute_b601_mit_torque(
    reference: B601MitReference,
    profile: B601MitControlProfile,
    *,
    measured_qpos_rad: Sequence[float],
    measured_qvel_rad_s: Sequence[float],
    gravity_torque_nm: Sequence[float],
) -> B601MitTorqueBreakdown:
    """Evaluate the ideal joint-space MIT law without actuator side effects."""

    if not isinstance(reference, B601MitReference):
        raise TypeError("reference must be a B601MitReference")
    if not isinstance(profile, B601MitControlProfile):
        raise TypeError("profile must be a B601MitControlProfile")
    qpos = np.asarray(
        _six_finite(measured_qpos_rad, name="measured_qpos_rad"),
        dtype=float,
    )
    qvel = np.asarray(
        _six_finite(measured_qvel_rad_s, name="measured_qvel_rad_s"),
        dtype=float,
    )
    gravity = np.asarray(
        _six_finite(gravity_torque_nm, name="gravity_torque_nm"),
        dtype=float,
    )
    qpos_reference = np.asarray(reference.qpos_rad, dtype=float)
    qvel_reference = np.asarray(reference.qvel_rad_s, dtype=float)
    kp = np.asarray(profile.position_kp_nm_per_rad, dtype=float)
    kd = np.asarray(profile.damping_kd_nm_s_per_rad, dtype=float)
    overlay_requested = np.asarray(reference.joint_torque_overlay_nm, dtype=float)
    overlay_limit = np.asarray(profile.max_abs_overlay_torque_nm, dtype=float)
    overlay_effective = np.clip(
        overlay_requested,
        -overlay_limit,
        overlay_limit,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        position = kp * (qpos_reference - qpos)
        damping = kd * (qvel_reference - qvel)
        requested = position + damping + gravity + overlay_effective
    if not all(
        np.all(np.isfinite(values)) for values in (position, damping, requested)
    ):
        raise ValueError("computed MIT torque contains non-finite values")
    return B601MitTorqueBreakdown(
        position_torque_nm=tuple(float(value) for value in position),
        damping_torque_nm=tuple(float(value) for value in damping),
        gravity_torque_nm=tuple(float(value) for value in gravity),
        overlay_requested_nm=tuple(float(value) for value in overlay_requested),
        overlay_effective_nm=tuple(float(value) for value in overlay_effective),
        requested_torque_nm=tuple(float(value) for value in requested),
        overlay_limited=not np.array_equal(
            overlay_requested,
            overlay_effective,
        ),
    )


__all__ = [
    "B601_RS_JOINT_MIT_FRAME",
    "B601MitControlProfile",
    "B601MitReference",
    "B601MitTorqueBreakdown",
    "compute_b601_mit_torque",
]
