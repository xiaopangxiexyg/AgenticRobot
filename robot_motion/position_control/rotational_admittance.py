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

from .cartesian_admittance import CartesianAdmittanceWrenchSample


def _finite_vector(
    values: Iterable[float],
    *,
    size: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=float).reshape(-1)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _positive_vector(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    vector = _finite_vector(values, size=3, name=name)
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain positive values")
    return tuple(float(value) for value in vector)


def _nonnegative_vector(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    vector = _finite_vector(values, size=3, name=name)
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return tuple(float(value) for value in vector)


def _finite_time(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _rotation_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be a proper rotation")
    return rotation


def _clamp_rotvec_bound(
    values: np.ndarray,
    *,
    component_limits: np.ndarray,
    enabled_axes: np.ndarray,
) -> np.ndarray:
    vector = np.where(enabled_axes, np.asarray(values, dtype=float), 0.0)
    limits = np.asarray(component_limits, dtype=float)
    vector = np.where(enabled_axes, np.clip(vector, -limits, limits), 0.0)
    if not np.any(enabled_axes):
        return np.zeros(3, dtype=float)
    norm_limit = float(np.max(limits[enabled_axes]))
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm > norm_limit:
        vector = vector * (norm_limit / vector_norm)
    return vector


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
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_vector(getattr(self, name), name=name),
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
        for name in ("torque_filter_tau_sec", "return_rate_s"):
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
        self._requested = np.zeros(3, dtype=float)
        self._output = np.zeros(3, dtype=float)
        self._velocity = np.zeros(3, dtype=float)
        self._tare_values: np.ndarray | None = None
        self._tare_frame: str | None = None
        self._tare_generation: int | None = None
        self._last_sample_identity: tuple[int, float] | None = None
        self._last_sample_source_time: float | None = None
        self._last_output_time: float | None = None
        self._frozen = True
        self._reason = "not_tared"
        self._new_sample_integrated = False
        self._integration_dt_sec: float | None = None
        self._deadbanded_torque = np.zeros(3, dtype=float)
        self._filtered_torque = np.zeros(3, dtype=float)
        self._filter_initialized = False

    @property
    def enabled(self) -> bool:
        return any(self.config.enabled_axes)

    def reset(
        self,
        *,
        tare_sample: CartesianAdmittanceWrenchSample,
        output_time_sec: float | None = None,
    ) -> RotationalAdmittanceSnapshot:
        self._requested.fill(0.0)
        self._output.fill(0.0)
        self._velocity.fill(0.0)
        self._tare_values = np.asarray(tare_sample.values_n_nm, dtype=float)
        self._tare_frame = tare_sample.frame
        self._tare_generation = tare_sample.tare_generation
        self._last_sample_identity = tare_sample.identity
        self._last_sample_source_time = tare_sample.source_time_sec
        self._last_output_time = (
            None
            if output_time_sec is None
            else _finite_time(output_time_sec, name="output_time_sec")
        )
        self._frozen = False
        self._reason = "reset"
        self._new_sample_integrated = False
        self._integration_dt_sec = None
        self._deadbanded_torque.fill(0.0)
        self._filtered_torque.fill(0.0)
        self._filter_initialized = False
        return self.snapshot(now_sec=output_time_sec)

    def freeze(self, reason: str) -> None:
        self._velocity.fill(0.0)
        self._deadbanded_torque.fill(0.0)
        self._filtered_torque.fill(0.0)
        self._filter_initialized = False
        self._frozen = True
        self._reason = str(reason).strip() or "frozen"
        self._new_sample_integrated = False
        self._integration_dt_sec = None

    def resume(self) -> None:
        if self._tare_generation is None:
            self.freeze("not_tared")
            return
        self._last_sample_identity = None
        self._last_sample_source_time = None
        self._velocity.fill(0.0)
        self._deadbanded_torque.fill(0.0)
        self._filtered_torque.fill(0.0)
        self._filter_initialized = False
        self._frozen = False
        self._reason = "resumed_awaiting_sample"

    def rebase_realized_rotation(
        self,
        rotvec_rad: Iterable[float],
        *,
        output_time_sec: float | None = None,
        preserve_requested: bool = False,
    ) -> RotationalAdmittanceSnapshot:
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        value = _clamp_rotvec_bound(
            _finite_vector(rotvec_rad, size=3, name="rotvec_rad"),
            component_limits=np.asarray(self.config.max_offset_rad, dtype=float),
            enabled_axes=enabled,
        )
        output_time = None
        if output_time_sec is not None:
            output_time = _finite_time(output_time_sec, name="output_time_sec")
            if self._last_output_time is not None and output_time < self._last_output_time:
                raise ValueError("rotation rebase output_time_sec must not move backwards")
        if not preserve_requested:
            self._requested = value.copy()
        self._output = value.copy()
        self._velocity.fill(0.0)
        if output_time is not None:
            self._last_output_time = output_time
        self._reason = "realized_rebase"
        self._new_sample_integrated = False
        self._integration_dt_sec = None
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

        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        if not self._filter_initialized:
            filtered = torque_control.copy()
        elif self.config.torque_filter_tau_sec > 0.0:
            alpha = 1.0 - math.exp(-dt / self.config.torque_filter_tau_sec)
            filtered = self._filtered_torque + alpha * (
                torque_control - self._filtered_torque
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

        previous = self._requested.copy()
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

        target = _clamp_rotvec_bound(
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
        step = _clamp_rotvec_bound(
            target - previous,
            component_limits=velocity_limit * dt,
            enabled_axes=enabled,
        )
        requested = _clamp_rotvec_bound(
            previous + step,
            component_limits=np.asarray(self.config.max_offset_rad, dtype=float),
            enabled_axes=enabled,
        )
        self._requested = requested
        self._velocity = (requested - previous) / dt
        self._deadbanded_torque = deadbanded
        self._filtered_torque = filtered
        self._filter_initialized = True

    def step_new_sample(
        self,
        sample: CartesianAdmittanceWrenchSample,
        *,
        now_sec: float,
        rotation_control_from_sample: np.ndarray | None = None,
        active: bool = True,
    ) -> RotationalAdmittanceSnapshot:
        now = _finite_time(now_sec, name="now_sec")
        self._new_sample_integrated = False
        self._integration_dt_sec = None
        if not active:
            self.freeze("inactive")
            return self.snapshot(now_sec=now)
        if self._frozen:
            return self.snapshot(now_sec=now)
        if not self.enabled:
            self._reason = "disabled"
            return self.snapshot(now_sec=now)
        if self._tare_values is None or self._tare_generation is None:
            self.freeze("not_tared")
            return self.snapshot(now_sec=now)
        if sample.tare_generation != self._tare_generation:
            self.freeze("tare_generation_mismatch")
            return self.snapshot(now_sec=now)
        if sample.frame != self._tare_frame:
            self.freeze("tare_frame_mismatch")
            return self.snapshot(now_sec=now)
        if sample.source_time_sec > now:
            self.freeze("sample_from_future")
            return self.snapshot(now_sec=now)
        if now - sample.source_time_sec > self.config.stale_after_sec:
            self.freeze("sample_stale")
            return self.snapshot(now_sec=now)
        if sample.identity == self._last_sample_identity:
            self._reason = "duplicate_sample"
            return self.snapshot(now_sec=now)
        if self._last_sample_source_time is None:
            self._last_sample_identity = sample.identity
            self._last_sample_source_time = sample.source_time_sec
            self._reason = "sample_clock_rearmed"
            return self.snapshot(now_sec=now)
        if sample.source_time_sec <= self._last_sample_source_time:
            self.freeze("sample_time_not_increasing")
            return self.snapshot(now_sec=now)
        dt = sample.source_time_sec - self._last_sample_source_time
        if dt > self.config.max_torque_dt_sec:
            self.freeze("torque_dt_exceeded")
            return self.snapshot(now_sec=now)

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
            - self._tare_values[3:]
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
        self._last_sample_identity = sample.identity
        self._last_sample_source_time = sample.source_time_sec
        self._reason = "sample_integrated"
        self._new_sample_integrated = True
        self._integration_dt_sec = dt
        return self.snapshot(now_sec=now)

    def sample_output(
        self,
        *,
        now_sec: float,
        active: bool = True,
    ) -> RotationalAdmittanceSnapshot:
        now = _finite_time(now_sec, name="now_sec")
        self._new_sample_integrated = False
        self._integration_dt_sec = None
        if not active:
            self.freeze("inactive")
            return self.snapshot(now_sec=now)
        if self._frozen:
            return self.snapshot(now_sec=now)
        if self._last_sample_source_time is None:
            self._reason = "awaiting_fresh_sample"
            return self.snapshot(now_sec=now)
        if now - self._last_sample_source_time > self.config.stale_after_sec:
            self.freeze("sample_stale")
            return self.snapshot(now_sec=now)
        if self._last_output_time is None:
            self._last_output_time = now
            self._reason = "output_clock_armed"
            return self.snapshot(now_sec=now)
        if now == self._last_output_time:
            self._reason = "duplicate_output_time"
            return self.snapshot(now_sec=now)
        if now < self._last_output_time:
            self.freeze("output_time_not_increasing")
            return self.snapshot(now_sec=now)

        dt = now - self._last_output_time
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        delta = _clamp_rotvec_bound(
            self._requested - self._output,
            component_limits=np.asarray(self.config.max_output_slew_rad_s) * dt,
            enabled_axes=enabled,
        )
        self._output = _clamp_rotvec_bound(
            self._output + delta,
            component_limits=np.asarray(self.config.max_offset_rad, dtype=float),
            enabled_axes=enabled,
        )
        self._last_output_time = now
        self._reason = "output_sampled"
        return self.snapshot(now_sec=now)

    def snapshot(self, *, now_sec: float | None = None) -> RotationalAdmittanceSnapshot:
        age = None
        if now_sec is not None and self._last_sample_source_time is not None:
            now = _finite_time(now_sec, name="now_sec")
            age = max(0.0, now - self._last_sample_source_time)
        return RotationalAdmittanceSnapshot(
            requested_rotvec_rad=tuple(float(value) for value in self._requested),
            output_rotvec_rad=tuple(float(value) for value in self._output),
            angular_velocity_rad_s=tuple(float(value) for value in self._velocity),
            last_sample_identity=self._last_sample_identity,
            last_sample_age_sec=age,
            tare_generation=self._tare_generation,
            frozen=self._frozen,
            reason=self._reason,
            new_sample_integrated=self._new_sample_integrated,
            integration_dt_sec=self._integration_dt_sec,
            deadbanded_torque_control_nm=tuple(
                float(value) for value in self._deadbanded_torque
            ),
            filtered_torque_control_nm=tuple(
                float(value) for value in self._filtered_torque
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
