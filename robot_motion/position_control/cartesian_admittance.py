"""Backend-neutral bounded Cartesian admittance state.

This module owns no clock, thread, sensor, IK, safety, collision, reference, or
dispatch authority.  Callers provide explicit timestamps and consume the
translation offset before their existing IK/admission pipeline.

The first implementation is intentionally translation-only.  Rotational
compliance must not be inferred from the six-dimensional wrench input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


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


def _frame_name(value: str, *, name: str) -> str:
    frame = str(value).strip()
    if not frame:
        raise ValueError(f"{name} must be non-empty")
    return frame


def _finite_time(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def discrete_lowpass_alpha_to_tau(
    alpha: float,
    *,
    sample_hz: float,
) -> float:
    """Convert a discrete new-sample weight to a continuous time constant.

    The established RM65 force controller uses ``y += alpha * (x - y)`` at
    its controller rate.  Expressing the same pole as a time constant lets a
    20 Hz wrench consumer preserve the 40 Hz profile dynamics without
    integrating a duplicated sample twice.
    """

    resolved_alpha = _finite_time(alpha, name="alpha")
    resolved_hz = _finite_time(sample_hz, name="sample_hz")
    if not 0.0 < resolved_alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if resolved_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    if resolved_alpha == 1.0:
        return 0.0
    return -1.0 / (resolved_hz * math.log1p(-resolved_alpha))


def _rotation_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be a proper rotation")
    return rotation


def _clamp_translation_bound(
    values: np.ndarray,
    *,
    component_limits: np.ndarray,
    enabled_axes: np.ndarray,
) -> np.ndarray:
    """Apply component bounds plus one total translation-norm bound.

    The public CLI currently expands one scalar translation limit to all
    three axes.  Treating those three equal values as independent limits
    would permit a diagonal command up to ``sqrt(3)`` times larger than the
    declared limit.  Keep the component bounds for anisotropic callers, while
    also bounding the total norm by the largest enabled component limit.  This
    matches the older RM65 pose-admittance contract, which limits the linear
    offset and velocity by vector norm.
    """

    vector = np.where(enabled_axes, np.asarray(values, dtype=float), 0.0)
    limits = np.asarray(component_limits, dtype=float)
    vector = np.where(
        enabled_axes,
        np.clip(vector, -limits, limits),
        0.0,
    )
    if not np.any(enabled_axes):
        return np.zeros(3, dtype=float)
    norm_limit = float(np.max(limits[enabled_axes]))
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm > norm_limit:
        vector = vector * (norm_limit / vector_norm)
    return vector


@dataclass(frozen=True)
class CartesianAdmittanceConfig:
    """Explicit translation-only admittance limits.

    All axes are disabled by default.  Enabling this configuration alone does
    not grant command authority; it only permits this state machine to produce
    a bounded offset for an upper-owned target composition stage.
    """

    enabled_axes: tuple[bool, bool, bool] = (False, False, False)
    control_frame: str = "arm_base"
    response_model: str = "second_order"
    virtual_mass_kg: tuple[float, float, float] = (1.0, 1.0, 1.0)
    damping_n_s_m: tuple[float, float, float] = (20.0, 20.0, 20.0)
    stiffness_n_m: tuple[float, float, float] = (100.0, 100.0, 100.0)
    force_deadband_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_filter_tau_sec: float = 0.0
    force_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_offset_m: tuple[float, float, float] = (0.005, 0.005, 0.005)
    max_velocity_m_s: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_return_velocity_m_s: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_output_slew_m_s: tuple[float, float, float] = (0.01, 0.01, 0.01)
    return_force_deadband_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_rate_s: float = 1.0
    response_reference_hz: float = 40.0
    stale_after_sec: float = 0.2
    max_force_dt_sec: float = 0.1

    def __post_init__(self) -> None:
        raw_axes = tuple(self.enabled_axes)
        if len(raw_axes) != 3 or any(type(value) is not bool for value in raw_axes):
            raise ValueError("enabled_axes must contain exactly three bool values")
        object.__setattr__(self, "enabled_axes", raw_axes)
        object.__setattr__(
            self,
            "control_frame",
            _frame_name(self.control_frame, name="control_frame"),
        )
        response_model = str(self.response_model).strip().lower()
        if response_model not in {"second_order", "profile_compliance"}:
            raise ValueError(
                "response_model must be 'second_order' or 'profile_compliance'"
            )
        object.__setattr__(self, "response_model", response_model)
        for name in (
            "virtual_mass_kg",
            "max_offset_m",
            "max_velocity_m_s",
            "max_return_velocity_m_s",
            "max_output_slew_m_s",
        ):
            object.__setattr__(
                self,
                name,
                _positive_vector(getattr(self, name), name=name),
            )
        for name in (
            "damping_n_s_m",
            "stiffness_n_m",
            "force_deadband_n",
            "return_force_deadband_n",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_vector(getattr(self, name), name=name),
            )
        force_sign = _finite_vector(self.force_sign, size=3, name="force_sign")
        if np.any(np.abs(force_sign) != 1.0):
            raise ValueError("force_sign values must be exactly -1 or +1")
        object.__setattr__(
            self,
            "force_sign",
            tuple(float(value) for value in force_sign),
        )
        for name in (
            "stale_after_sec",
            "max_force_dt_sec",
            "response_reference_hz",
        ):
            value = _finite_time(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        filter_tau = _finite_time(
            self.force_filter_tau_sec,
            name="force_filter_tau_sec",
        )
        if filter_tau < 0.0:
            raise ValueError("force_filter_tau_sec must be non-negative")
        object.__setattr__(self, "force_filter_tau_sec", filter_tau)
        return_rate = _finite_time(self.return_rate_s, name="return_rate_s")
        if return_rate < 0.0:
            raise ValueError("return_rate_s must be non-negative")
        object.__setattr__(self, "return_rate_s", return_rate)
        if response_model == "profile_compliance":
            stiffness = np.asarray(self.stiffness_n_m, dtype=float)
            enabled = np.asarray(self.enabled_axes, dtype=bool)
            if np.any(stiffness[enabled] <= 0.0):
                raise ValueError(
                    "profile_compliance requires positive stiffness on enabled axes"
                )


@dataclass(frozen=True)
class CartesianAdmittanceWrenchSample:
    """One immutable wrench sample stamped with the active Home generation."""

    values_n_nm: tuple[float, float, float, float, float, float]
    frame: str
    source_time_sec: float
    receive_time_sec: float
    tare_generation: int

    def __post_init__(self) -> None:
        values = _finite_vector(self.values_n_nm, size=6, name="values_n_nm")
        object.__setattr__(
            self,
            "values_n_nm",
            tuple(float(value) for value in values),
        )
        object.__setattr__(self, "frame", _frame_name(self.frame, name="frame"))
        object.__setattr__(
            self,
            "source_time_sec",
            _finite_time(self.source_time_sec, name="source_time_sec"),
        )
        object.__setattr__(
            self,
            "receive_time_sec",
            _finite_time(self.receive_time_sec, name="receive_time_sec"),
        )
        if isinstance(self.tare_generation, bool) or int(self.tare_generation) < 0:
            raise ValueError("tare_generation must be a non-negative integer")
        object.__setattr__(self, "tare_generation", int(self.tare_generation))

    @property
    def identity(self) -> tuple[int, float]:
        return (self.tare_generation, self.source_time_sec)


@dataclass(frozen=True)
class CartesianAdmittanceSnapshot:
    requested_offset_m: tuple[float, float, float]
    output_offset_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    last_sample_identity: tuple[int, float] | None
    last_sample_age_sec: float | None
    tare_generation: int | None
    frozen: bool
    reason: str
    new_sample_integrated: bool
    integration_dt_sec: float | None
    deadbanded_force_control_n: tuple[float, float, float]
    filtered_force_control_n: tuple[float, float, float]


class BoundedCartesianAdmittance:
    """Causal translation-only admittance with explicit freeze/reset semantics."""

    def __init__(self, config: CartesianAdmittanceConfig | None = None):
        self.config = config if config is not None else CartesianAdmittanceConfig()
        self._requested_offset = np.zeros(3, dtype=float)
        self._output_offset = np.zeros(3, dtype=float)
        self._velocity = np.zeros(3, dtype=float)
        self._tare_values: np.ndarray | None = None
        self._tare_frame: str | None = None
        self._tare_generation: int | None = None
        self._last_sample_identity: tuple[int, float] | None = None
        self._last_sample_source_time: float | None = None
        self._last_sample_receive_time: float | None = None
        self._last_output_time: float | None = None
        self._frozen = True
        self._reason = "not_tared"
        self._new_sample_integrated = False
        self._integration_dt_sec: float | None = None
        self._deadbanded_force = np.zeros(3, dtype=float)
        self._filtered_force = np.zeros(3, dtype=float)
        self._force_filter_initialized = False

    @property
    def enabled(self) -> bool:
        return any(self.config.enabled_axes)

    def reset(
        self,
        *,
        tare_sample: CartesianAdmittanceWrenchSample,
        output_time_sec: float | None = None,
    ) -> CartesianAdmittanceSnapshot:
        """Start one Home/tare generation from zero offset and velocity."""

        self._requested_offset.fill(0.0)
        self._output_offset.fill(0.0)
        self._velocity.fill(0.0)
        self._tare_values = np.asarray(tare_sample.values_n_nm, dtype=float)
        self._tare_frame = tare_sample.frame
        self._tare_generation = tare_sample.tare_generation
        self._last_sample_identity = tare_sample.identity
        self._last_sample_source_time = tare_sample.source_time_sec
        self._last_sample_receive_time = tare_sample.receive_time_sec
        self._last_output_time = (
            None
            if output_time_sec is None
            else _finite_time(output_time_sec, name="output_time_sec")
        )
        self._frozen = False
        self._reason = "reset"
        self._new_sample_integrated = False
        self._integration_dt_sec = None
        self._deadbanded_force.fill(0.0)
        self._filtered_force.fill(0.0)
        self._force_filter_initialized = False
        return self.snapshot(now_sec=output_time_sec)

    def freeze(self, reason: str) -> None:
        self._velocity.fill(0.0)
        self._deadbanded_force.fill(0.0)
        self._filtered_force.fill(0.0)
        self._force_filter_initialized = False
        self._frozen = True
        self._reason = str(reason).strip() or "frozen"
        self._new_sample_integrated = False
        self._integration_dt_sec = None

    def resume(self) -> None:
        if self._tare_generation is None:
            self.freeze("not_tared")
            return
        # A frozen interval must never be integrated as one large force dt.
        # The first fresh sample after resume only re-arms the sample clock.
        self._last_sample_identity = None
        self._last_sample_source_time = None
        self._last_sample_receive_time = None
        self._velocity.fill(0.0)
        self._deadbanded_force.fill(0.0)
        self._filtered_force.fill(0.0)
        self._force_filter_initialized = False
        self._frozen = False
        self._reason = "resumed_awaiting_sample"

    def rebase_realized_offset(
        self,
        offset_m: Iterable[float],
        *,
        output_time_sec: float | None = None,
        preserve_requested: bool = False,
    ) -> CartesianAdmittanceSnapshot:
        """Explicitly reset output to one caller-provided bounded offset.

        ``preserve_requested`` keeps the bounded force target distinct from
        the reset output. Callers must already own a valid incremental baseline;
        an absolute target-to-command or target-to-pose residual is not a valid
        input. The default full-state rebase remains available for explicit
        lifecycle resets and isolated controller tests.
        """

        offset = _finite_vector(offset_m, size=3, name="offset_m")
        maximum = np.asarray(self.config.max_offset_m, dtype=float)
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        offset = _clamp_translation_bound(
            offset,
            component_limits=maximum,
            enabled_axes=enabled,
        )
        output_time: float | None = None
        if output_time_sec is not None:
            output_time = _finite_time(
                output_time_sec,
                name="output_time_sec",
            )
            if (
                self._last_output_time is not None
                and output_time < self._last_output_time
            ):
                raise ValueError(
                    "realized-offset rebase output_time_sec must not move "
                    "backwards"
                )
        if not preserve_requested:
            self._requested_offset = offset.copy()
        self._output_offset = offset.copy()
        self._velocity.fill(0.0)
        if output_time is not None:
            self._last_output_time = output_time
        self._reason = "realized_rebase"
        self._new_sample_integrated = False
        self._integration_dt_sec = None
        return self.snapshot(now_sec=output_time_sec)

    def step_new_sample(
        self,
        sample: CartesianAdmittanceWrenchSample,
        *,
        now_sec: float,
        rotation_control_from_sample: np.ndarray | None = None,
        active: bool = True,
    ) -> CartesianAdmittanceSnapshot:
        """Integrate exactly once for one fresh, in-generation wrench sample."""

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
        sample_age = max(0.0, now - sample.source_time_sec)
        if sample_age > self.config.stale_after_sec:
            self.freeze("sample_stale")
            return self.snapshot(now_sec=now)
        if sample.identity == self._last_sample_identity:
            self._reason = "duplicate_sample"
            return self.snapshot(now_sec=now)
        if self._last_sample_source_time is None:
            self._last_sample_identity = sample.identity
            self._last_sample_source_time = sample.source_time_sec
            self._last_sample_receive_time = sample.receive_time_sec
            self._reason = "sample_clock_rearmed"
            return self.snapshot(now_sec=now)
        if sample.source_time_sec <= self._last_sample_source_time:
            self.freeze("sample_time_not_increasing")
            return self.snapshot(now_sec=now)

        dt = sample.source_time_sec - self._last_sample_source_time
        if dt > self.config.max_force_dt_sec:
            self.freeze("force_dt_exceeded")
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

        tared_force_sample = (
            np.asarray(sample.values_n_nm[:3], dtype=float)
            - self._tare_values[:3]
        )
        force_control = rotation @ tared_force_sample
        force_control *= np.asarray(self.config.force_sign, dtype=float)
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        deadband = np.asarray(self.config.force_deadband_n, dtype=float)

        if self.config.response_model == "profile_compliance":
            # Match the established RM65 WrenchProcessor ordering: filter the
            # tare-corrected wrench first, then apply deadband.  The first
            # post-reset sample initializes the filter by identity, as the
            # legacy processor does.
            if not self._force_filter_initialized:
                filtered_force = force_control.copy()
            elif self.config.force_filter_tau_sec > 0.0:
                filter_alpha = 1.0 - math.exp(
                    -dt / self.config.force_filter_tau_sec
                )
                filtered_force = (
                    self._filtered_force
                    + filter_alpha * (force_control - self._filtered_force)
                )
            else:
                filtered_force = force_control.copy()
            filtered_force = np.where(enabled, filtered_force, 0.0)
            deadbanded = np.sign(filtered_force) * np.maximum(
                np.abs(filtered_force) - deadband,
                0.0,
            )
            deadbanded = np.where(enabled, deadbanded, 0.0)
        else:
            # Preserve the original exploratory second-order behavior as an
            # atomic rollback mode: deadband first, then low-pass the force
            # that drives M/B/K integration.
            deadbanded = np.sign(force_control) * np.maximum(
                np.abs(force_control) - deadband,
                0.0,
            )
            deadbanded = np.where(enabled, deadbanded, 0.0)
            if self.config.force_filter_tau_sec > 0.0:
                filter_alpha = 1.0 - math.exp(
                    -dt / self.config.force_filter_tau_sec
                )
                filtered_force = (
                    self._filtered_force
                    + filter_alpha * (deadbanded - self._filtered_force)
                )
            else:
                filtered_force = deadbanded.copy()
            filtered_force = np.where(enabled, filtered_force, 0.0)

        previous_requested = self._requested_offset.copy()
        maximum_velocity = np.asarray(self.config.max_velocity_m_s, dtype=float)
        maximum_offset = np.asarray(self.config.max_offset_m, dtype=float)

        if self.config.response_model == "profile_compliance":
            stiffness = np.asarray(self.config.stiffness_n_m, dtype=float)
            target = np.zeros(3, dtype=float)
            target[enabled] = deadbanded[enabled] / stiffness[enabled]

            return_deadband = np.asarray(
                self.config.return_force_deadband_n,
                dtype=float,
            )
            return_applied = np.zeros(3, dtype=bool)
            for axis in range(3):
                if not enabled[axis] or abs(deadbanded[axis]) > 1e-12:
                    continue
                if abs(filtered_force[axis]) > return_deadband[axis] + 1e-12:
                    target[axis] = previous_requested[axis]
                    continue
                reference_dt = 1.0 / self.config.response_reference_hz
                reference_decay = max(
                    0.0,
                    1.0 - self.config.return_rate_s * reference_dt,
                )
                decay = reference_decay ** (dt / reference_dt)
                target[axis] = previous_requested[axis] * decay
                return_applied[axis] = True

            target = _clamp_translation_bound(
                target,
                component_limits=maximum_offset,
                enabled_axes=enabled,
            )
            passive_return = bool(
                np.any(return_applied)
                and not np.any(np.abs(deadbanded[enabled]) > 1e-12)
            )
            step_velocity = maximum_velocity
            if passive_return:
                step_velocity = np.minimum(
                    maximum_velocity,
                    np.asarray(
                        self.config.max_return_velocity_m_s,
                        dtype=float,
                    ),
                )
            step = _clamp_translation_bound(
                target - previous_requested,
                component_limits=step_velocity * dt,
                enabled_axes=enabled,
            )
            requested = previous_requested + step
        else:
            mass = np.asarray(self.config.virtual_mass_kg, dtype=float)
            damping = np.asarray(self.config.damping_n_s_m, dtype=float)
            stiffness = np.asarray(self.config.stiffness_n_m, dtype=float)
            acceleration = (
                filtered_force
                - damping * self._velocity
                - stiffness * self._requested_offset
            ) / mass
            velocity = self._velocity + acceleration * dt
            velocity = _clamp_translation_bound(
                velocity,
                component_limits=maximum_velocity,
                enabled_axes=enabled,
            )
            requested = previous_requested + velocity * dt

        requested = _clamp_translation_bound(
            requested,
            component_limits=maximum_offset,
            enabled_axes=enabled,
        )
        # Store the actually admitted state velocity.  At the total-norm
        # boundary this removes only the unrealized outward component instead
        # of retaining hidden velocity that would later cause a catch-up.
        velocity = (requested - previous_requested) / dt

        self._requested_offset = requested
        self._velocity = velocity
        self._last_sample_identity = sample.identity
        self._last_sample_source_time = sample.source_time_sec
        self._last_sample_receive_time = sample.receive_time_sec
        self._deadbanded_force = deadbanded
        self._filtered_force = filtered_force
        self._force_filter_initialized = True
        self._reason = "sample_integrated"
        self._new_sample_integrated = True
        self._integration_dt_sec = dt
        return self.snapshot(now_sec=now)

    def sample_output(
        self,
        *,
        now_sec: float,
        active: bool = True,
    ) -> CartesianAdmittanceSnapshot:
        """Read a slew-limited offset without integrating the wrench again."""

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
        max_delta = np.asarray(self.config.max_output_slew_m_s, dtype=float) * dt
        delta = _clamp_translation_bound(
            self._requested_offset - self._output_offset,
            component_limits=max_delta,
            enabled_axes=enabled,
        )
        # Re-assert the absolute envelope at the final 60 Hz output boundary.
        # Requested state is already bounded, but the output state can survive
        # a profile/code transition or a realized-state rebase.  Never rely on
        # convex interpolation alone for a command-authority safety invariant.
        self._output_offset = _clamp_translation_bound(
            self._output_offset + delta,
            component_limits=np.asarray(self.config.max_offset_m, dtype=float),
            enabled_axes=enabled,
        )
        self._last_output_time = now
        self._reason = "output_sampled"
        return self.snapshot(now_sec=now)

    def snapshot(self, *, now_sec: float | None = None) -> CartesianAdmittanceSnapshot:
        age: float | None = None
        if now_sec is not None and self._last_sample_source_time is not None:
            now = _finite_time(now_sec, name="now_sec")
            age = max(0.0, now - self._last_sample_source_time)
        return CartesianAdmittanceSnapshot(
            requested_offset_m=tuple(float(value) for value in self._requested_offset),
            output_offset_m=tuple(float(value) for value in self._output_offset),
            velocity_m_s=tuple(float(value) for value in self._velocity),
            last_sample_identity=self._last_sample_identity,
            last_sample_age_sec=age,
            tare_generation=self._tare_generation,
            frozen=self._frozen,
            reason=self._reason,
            new_sample_integrated=self._new_sample_integrated,
            integration_dt_sec=self._integration_dt_sec,
            deadbanded_force_control_n=tuple(
                float(value) for value in self._deadbanded_force
            ),
            filtered_force_control_n=tuple(
                float(value) for value in self._filtered_force
            ),
        )


__all__ = [
    "BoundedCartesianAdmittance",
    "CartesianAdmittanceConfig",
    "CartesianAdmittanceSnapshot",
    "CartesianAdmittanceWrenchSample",
    "discrete_lowpass_alpha_to_tau",
]
