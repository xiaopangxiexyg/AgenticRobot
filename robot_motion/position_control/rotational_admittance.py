"""Backend-neutral bounded rotational admittance state.

The controller consumes tare-corrected torque exactly once per fresh wrench
sample and exposes a slew-limited rotation-vector offset at the caller's output
clock.  The offset lives in one explicit control frame and is intended to be
composed on SO(3) as ``R_applied = Exp(offset) @ R_raw``.

This module owns no clock, sensor, IK, collision, safety, dispatch, or hardware
authority.  All axes are disabled by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ._bounded_admittance_lifecycle import (
    BoundedVectorLifecycle,
    _clamp_bounded_vector,
    _finite_time,
    _finite_vector,
    _nonnegative_vector,
    _positive_vector,
    _rotation_matrix,
)
from .cartesian_admittance import CartesianAdmittanceWrenchSample


def rotation_matrix_from_rotvec(rotvec_rad: Iterable[float]) -> np.ndarray:
    """Return ``Exp(rotvec)`` without Euler-angle conversion."""

    vector = _finite_vector(rotvec_rad, size=3, name="rotvec_rad")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        return np.eye(3, dtype=float)
    axis = vector / angle
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def compose_world_rotation_offset(
    raw_rotation: np.ndarray,
    rotvec_control_rad: Iterable[float],
) -> np.ndarray:
    """Pre-multiply a control/world-frame rotation-vector offset."""

    raw = _rotation_matrix(raw_rotation, name="raw_rotation")
    return rotation_matrix_from_rotvec(rotvec_control_rad) @ raw


@dataclass(frozen=True)
class RotationalAdmittanceConfig:
    """Explicit bounded profile-compliance parameters for rotational offset."""

    enabled_axes: tuple[bool, bool, bool] = (False, False, False)
    control_frame: str = "arm_base"
    stiffness_nm_rad: tuple[float, float, float] = (10.0, 10.0, 10.0)
    torque_deadband_nm: tuple[float, float, float] = (0.04, 0.04, 0.04)
    torque_filter_tau_sec: float = 0.0
    output_follow_tau_sec: float = 0.0
    max_output_angular_acceleration_rad_s2: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    torque_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_offset_rad: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_velocity_rad_s: tuple[float, float, float] = (0.02, 0.02, 0.02)
    max_return_velocity_rad_s: tuple[float, float, float] = (0.02, 0.02, 0.02)
    max_output_slew_rad_s: tuple[float, float, float] = (0.02, 0.02, 0.02)
    return_torque_deadband_nm: tuple[float, float, float] = (0.04, 0.04, 0.04)
    return_rate_s: float = 1.0
    response_reference_hz: float = 40.0
    stale_after_sec: float = 0.2
    max_torque_dt_sec: float = 0.1

    def __post_init__(self) -> None:
        axes = tuple(self.enabled_axes)
        if len(axes) != 3 or any(type(value) is not bool for value in axes):
            raise ValueError("enabled_axes must contain exactly three bool values")
        object.__setattr__(self, "enabled_axes", axes)
        frame = str(self.control_frame).strip()
        if not frame:
            raise ValueError("control_frame must be non-empty")
        object.__setattr__(self, "control_frame", frame)
        for name in (
            "stiffness_nm_rad",
            "max_offset_rad",
            "max_velocity_rad_s",
            "max_return_velocity_rad_s",
            "max_output_slew_rad_s",
        ):
            object.__setattr__(
                self,
                name,
                _positive_vector(getattr(self, name), name=name),
            )
        for name in (
            "torque_deadband_nm",
            "return_torque_deadband_nm",
            "max_output_angular_acceleration_rad_s2",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_vector(getattr(self, name), name=name),
            )
        acceleration = np.asarray(
            self.max_output_angular_acceleration_rad_s2,
            dtype=float,
        )
        enabled = np.asarray(self.enabled_axes, dtype=bool)
        enabled_acceleration = acceleration[enabled]
        if (
            enabled_acceleration.size
            and np.any(enabled_acceleration > 0.0)
            and np.any(enabled_acceleration == 0.0)
        ):
            raise ValueError(
                "max_output_angular_acceleration_rad_s2 must be zero on every "
                "enabled axis (disabled) or positive on every enabled axis"
            )
        signs = _finite_vector(self.torque_sign, size=3, name="torque_sign")
        if np.any(np.abs(signs) != 1.0):
            raise ValueError("torque_sign values must be exactly -1 or +1")
        object.__setattr__(
            self,
            "torque_sign",
            tuple(float(value) for value in signs),
        )
        for name in (
            "response_reference_hz",
            "stale_after_sec",
            "max_torque_dt_sec",
        ):
            value = _finite_time(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "torque_filter_tau_sec",
            "output_follow_tau_sec",
            "return_rate_s",
        ):
            value = _finite_time(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RotationalAdmittanceSnapshot:
    requested_rotvec_rad: tuple[float, float, float]
    output_rotvec_rad: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]
    last_sample_identity: tuple[int, float] | None
    last_sample_age_sec: float | None
    tare_generation: int | None
    frozen: bool
    reason: str
    new_sample_integrated: bool
    integration_dt_sec: float | None
    deadbanded_torque_control_nm: tuple[float, float, float]
    filtered_torque_control_nm: tuple[float, float, float]


class BoundedRotationalAdmittance:
    """Causal, profile-compliance rotational admittance with bounded output."""

    def __init__(self, config: RotationalAdmittanceConfig | None = None):
        self.config = config if config is not None else RotationalAdmittanceConfig()
        self._lifecycle = BoundedVectorLifecycle()

    @property
    def enabled(self) -> bool:
        return any(self.config.enabled_axes)

    def reset(
        self,
        *,
        tare_sample: CartesianAdmittanceWrenchSample,
        output_time_sec: float | None = None,
    ) -> RotationalAdmittanceSnapshot:
        self._lifecycle.reset(
            tare_values=tare_sample.values_n_nm,
            tare_frame=tare_sample.frame,
            tare_generation=tare_sample.tare_generation,
            sample_identity=tare_sample.identity,
            sample_source_time=tare_sample.source_time_sec,
            output_time_sec=output_time_sec,
        )
        return self.snapshot(now_sec=output_time_sec)

    def freeze(self, reason: str) -> None:
        self._lifecycle.freeze(reason)

    def resume(self) -> None:
        self._lifecycle.resume()

    def rebase_realized_rotation(
        self,
        rotvec_rad: Iterable[float],
        *,
        output_time_sec: float | None = None,
        preserve_requested: bool = False,
    ) -> RotationalAdmittanceSnapshot:
        self._lifecycle.rebase(
            rotvec_rad,
            value_name="rotvec_rad",
            component_limits=self.config.max_offset_rad,
            enabled_axes=self.config.enabled_axes,
            output_time_sec=output_time_sec,
            preserve_requested=preserve_requested,
            backwards_error=(
                "rotation rebase output_time_sec must not move backwards"
            ),
        )
        return self.snapshot(now_sec=output_time_sec)

    def _integrate_profile_substep(
        self,
        *,
        torque_control: np.ndarray,
        dt: float,
    ) -> None:
        """Advance one held-torque reference step.

        The established RM65 profile evaluates its filter, deadband, compliance
        target, and rate limit at 40 Hz.  A 20 Hz wrench producer therefore
        represents two held-input reference steps.  Substepping is necessary
        around the torque deadband: matching only the final continuous-time
        filter pole does not reproduce the intermediate return/load decision.
        """

        state = self._lifecycle
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        if not state.filter_initialized:
            filtered = torque_control.copy()
        elif self.config.torque_filter_tau_sec > 0.0:
            alpha = 1.0 - math.exp(-dt / self.config.torque_filter_tau_sec)
            filtered = state.filtered_input + alpha * (
                torque_control - state.filtered_input
            )
        else:
            filtered = torque_control.copy()
        filtered = np.where(enabled, filtered, 0.0)
        deadband = np.asarray(self.config.torque_deadband_nm, dtype=float)
        deadbanded = np.sign(filtered) * np.maximum(
            np.abs(filtered) - deadband,
            0.0,
        )
        deadbanded = np.where(enabled, deadbanded, 0.0)

        previous = state.requested.copy()
        stiffness = np.asarray(self.config.stiffness_nm_rad, dtype=float)
        target = np.zeros(3, dtype=float)
        target[enabled] = deadbanded[enabled] / stiffness[enabled]
        return_deadband = np.asarray(
            self.config.return_torque_deadband_nm,
            dtype=float,
        )
        return_applied = np.zeros(3, dtype=bool)
        for axis in range(3):
            if not enabled[axis] or abs(deadbanded[axis]) > 1e-12:
                continue
            if abs(filtered[axis]) > return_deadband[axis] + 1e-12:
                target[axis] = previous[axis]
                continue
            reference_decay = max(0.0, 1.0 - self.config.return_rate_s * dt)
            target[axis] = previous[axis] * reference_decay
            return_applied[axis] = True

        target = _clamp_bounded_vector(
            target,
            component_limits=np.asarray(self.config.max_offset_rad, dtype=float),
            enabled_axes=enabled,
        )
        passive_return = bool(
            np.any(return_applied)
            and not np.any(np.abs(deadbanded[enabled]) > 1e-12)
        )
        velocity_limit = np.asarray(self.config.max_velocity_rad_s, dtype=float)
        if passive_return:
            velocity_limit = np.minimum(
                velocity_limit,
                np.asarray(self.config.max_return_velocity_rad_s, dtype=float),
            )
        step = _clamp_bounded_vector(
            target - previous,
            component_limits=velocity_limit * dt,
            enabled_axes=enabled,
        )
        requested = _clamp_bounded_vector(
            previous + step,
            component_limits=np.asarray(self.config.max_offset_rad, dtype=float),
            enabled_axes=enabled,
        )
        state.requested = requested
        state.velocity = (requested - previous) / dt
        state.deadbanded_input = deadbanded
        state.filtered_input = filtered
        state.filter_initialized = True
        state.passive_return_active = passive_return

    def step_new_sample(
        self,
        sample: CartesianAdmittanceWrenchSample,
        *,
        now_sec: float,
        rotation_control_from_sample: np.ndarray | None = None,
        active: bool = True,
    ) -> RotationalAdmittanceSnapshot:
        now = _finite_time(now_sec, name="now_sec")
        state = self._lifecycle
        dt = state.prepare_sample(
            sample_identity=sample.identity,
            sample_source_time=sample.source_time_sec,
            sample_frame=sample.frame,
            sample_tare_generation=sample.tare_generation,
            now_sec=now,
            active=active,
            enabled=self.enabled,
            stale_after_sec=self.config.stale_after_sec,
            max_input_dt_sec=self.config.max_torque_dt_sec,
            dt_exceeded_reason="torque_dt_exceeded",
        )
        if dt is None:
            return self.snapshot(now_sec=now)
        assert state.tare_values is not None

        if sample.frame == self.config.control_frame:
            rotation = np.eye(3)
        elif rotation_control_from_sample is None:
            self.freeze("frame_transform_missing")
            return self.snapshot(now_sec=now)
        else:
            rotation = _rotation_matrix(
                rotation_control_from_sample,
                name="rotation_control_from_sample",
            )

        tared_torque = (
            np.asarray(sample.values_n_nm[3:], dtype=float)
            - state.tare_values[3:]
        )
        torque_control = rotation @ tared_torque
        torque_control *= np.asarray(self.config.torque_sign, dtype=float)
        reference_steps = max(1, int(round(dt * self.config.response_reference_hz)))
        substep_dt = dt / reference_steps
        for _ in range(reference_steps):
            self._integrate_profile_substep(
                torque_control=torque_control,
                dt=substep_dt,
            )
        state.finish_sample(
            sample_identity=sample.identity,
            sample_source_time=sample.source_time_sec,
            integration_dt_sec=dt,
        )
        return self.snapshot(now_sec=now)

    def sample_output(
        self,
        *,
        now_sec: float,
        active: bool = True,
    ) -> RotationalAdmittanceSnapshot:
        now = _finite_time(now_sec, name="now_sec")
        self._lifecycle.sample_output(
            now_sec=now,
            active=active,
            stale_after_sec=self.config.stale_after_sec,
            enabled_axes=self.config.enabled_axes,
            output_follow_tau_sec=self.config.output_follow_tau_sec,
            output_slew_limits=self.config.max_output_slew_rad_s,
            acceleration_limits=(
                self.config.max_output_angular_acceleration_rad_s2
            ),
            absolute_limits=self.config.max_offset_rad,
        )
        return self.snapshot(now_sec=now)

    def snapshot(self, *, now_sec: float | None = None) -> RotationalAdmittanceSnapshot:
        state = self._lifecycle
        return RotationalAdmittanceSnapshot(
            requested_rotvec_rad=tuple(float(value) for value in state.requested),
            output_rotvec_rad=tuple(float(value) for value in state.output),
            angular_velocity_rad_s=tuple(float(value) for value in state.velocity),
            last_sample_identity=state.last_sample_identity,
            last_sample_age_sec=state.sample_age(now_sec=now_sec),
            tare_generation=state.tare_generation,
            frozen=state.frozen,
            reason=state.reason,
            new_sample_integrated=state.new_sample_integrated,
            integration_dt_sec=state.integration_dt_sec,
            deadbanded_torque_control_nm=tuple(
                float(value) for value in state.deadbanded_input
            ),
            filtered_torque_control_nm=tuple(
                float(value) for value in state.filtered_input
            ),
        )


ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG = "cautious-3deg"
ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG = "balanced-5deg"
ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG = (
    "rotation-emphasis-5deg"
)
ROTATIONAL_ADMITTANCE_PROFILE_MODERATED_8DEG = "moderated-8deg"
ROTATIONAL_ADMITTANCE_PROFILE_LEGACY_NORMAL = "legacy-normal"
ROTATIONAL_ADMITTANCE_PROFILE_CHOICES = (
    ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_MODERATED_8DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_LEGACY_NORMAL,
)


def rotational_admittance_config_for_profile(
    profile: str,
) -> RotationalAdmittanceConfig:
    """Resolve a named, immutable rotational response envelope."""

    name = str(profile).strip().lower()
    common = {
        "enabled_axes": (True, True, True),
        "control_frame": "solver_world",
        "torque_deadband_nm": (0.04, 0.04, 0.04),
        "torque_filter_tau_sec": 0.010857362,
        "torque_sign": (1.0, 1.0, 1.0),
        "return_torque_deadband_nm": (0.04, 0.04, 0.04),
        "return_rate_s": 1.0,
        "response_reference_hz": 40.0,
        "stale_after_sec": 0.15,
        "max_torque_dt_sec": 0.15,
    }
    if name == ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG:
        return RotationalAdmittanceConfig(
            **common,
            stiffness_nm_rad=(30.0, 30.0, 30.0),
            max_offset_rad=(math.radians(3.0),) * 3,
            max_velocity_rad_s=(math.radians(4.0),) * 3,
            max_return_velocity_rad_s=(math.radians(3.0),) * 3,
            max_output_slew_rad_s=(math.radians(4.0),) * 3,
        )
    if name == ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG:
        return RotationalAdmittanceConfig(
            **common,
            stiffness_nm_rad=(25.0, 25.0, 25.0),
            max_offset_rad=(math.radians(5.0),) * 3,
            max_velocity_rad_s=(math.radians(8.0),) * 3,
            max_return_velocity_rad_s=(math.radians(4.5),) * 3,
            max_output_slew_rad_s=(math.radians(8.0),) * 3,
        )
    if name == ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG:
        return RotationalAdmittanceConfig(
            **common,
            stiffness_nm_rad=(25.0, 25.0, 25.0),
            max_offset_rad=(math.radians(5.0),) * 3,
            max_velocity_rad_s=(math.radians(9.0),) * 3,
            max_return_velocity_rad_s=(math.radians(4.5),) * 3,
            max_output_slew_rad_s=(math.radians(9.0),) * 3,
        )
    if name == ROTATIONAL_ADMITTANCE_PROFILE_MODERATED_8DEG:
        return RotationalAdmittanceConfig(
            **common,
            stiffness_nm_rad=(25.0, 25.0, 25.0),
            max_offset_rad=(math.radians(8.0),) * 3,
            max_velocity_rad_s=(math.radians(10.0),) * 3,
            max_return_velocity_rad_s=(math.radians(5.7),) * 3,
            max_output_slew_rad_s=(math.radians(10.0),) * 3,
        )
    if name == ROTATIONAL_ADMITTANCE_PROFILE_LEGACY_NORMAL:
        return RotationalAdmittanceConfig(
            **common,
            stiffness_nm_rad=(10.0, 10.0, 10.0),
            max_offset_rad=(0.4, 0.4, 0.4),
            max_velocity_rad_s=(0.35, 0.35, 0.35),
            max_return_velocity_rad_s=(0.1, 0.1, 0.1),
            max_output_slew_rad_s=(0.35, 0.35, 0.35),
        )
    raise ValueError(
        "rotational admittance profile must be one of "
        + ", ".join(ROTATIONAL_ADMITTANCE_PROFILE_CHOICES)
    )


__all__ = [
    "BoundedRotationalAdmittance",
    "RotationalAdmittanceConfig",
    "RotationalAdmittanceSnapshot",
    "ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG",
    "ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG",
    "ROTATIONAL_ADMITTANCE_PROFILE_CHOICES",
    "ROTATIONAL_ADMITTANCE_PROFILE_LEGACY_NORMAL",
    "ROTATIONAL_ADMITTANCE_PROFILE_MODERATED_8DEG",
    "ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG",
    "compose_world_rotation_offset",
    "rotation_matrix_from_rotvec",
    "rotational_admittance_config_for_profile",
]
