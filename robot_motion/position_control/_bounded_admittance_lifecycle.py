"""Shared internal lifecycle for bounded three-vector admittance state.

The Cartesian and rotational controllers keep ownership of their physical
input semantics and integration equations.  This module owns the mechanics
that must stay identical between them: tare/sample identity, freeze/resume,
sample and output clocks, and the final bounded-vector output envelope.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

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


def _clamp_bounded_vector(
    values: np.ndarray,
    *,
    component_limits: np.ndarray,
    enabled_axes: np.ndarray,
) -> np.ndarray:
    """Apply enabled-axis component limits and one total vector-norm limit."""

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


class BoundedVectorLifecycle:
    """One authoritative lifecycle for a bounded three-vector modifier."""

    def __init__(self) -> None:
        self.requested = np.zeros(3, dtype=float)
        self.output = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.output_velocity = np.zeros(3, dtype=float)
        self.tare_values: np.ndarray | None = None
        self.tare_frame: str | None = None
        self.tare_generation: int | None = None
        self.last_sample_identity: tuple[int, float] | None = None
        self.last_sample_source_time: float | None = None
        self.last_output_time: float | None = None
        self.frozen = True
        self.reason = "not_tared"
        self.new_sample_integrated = False
        self.integration_dt_sec: float | None = None
        self.deadbanded_input = np.zeros(3, dtype=float)
        self.filtered_input = np.zeros(3, dtype=float)
        self.filter_initialized = False
        self.passive_return_active = False

    def _clear_event(self) -> None:
        self.new_sample_integrated = False
        self.integration_dt_sec = None

    def _clear_motion(self) -> None:
        self.velocity.fill(0.0)
        self.output_velocity.fill(0.0)
        self.deadbanded_input.fill(0.0)
        self.filtered_input.fill(0.0)
        self.filter_initialized = False
        self.passive_return_active = False

    def reset(
        self,
        *,
        tare_values: Iterable[float],
        tare_frame: str,
        tare_generation: int,
        sample_identity: tuple[int, float],
        sample_source_time: float,
        output_time_sec: float | None,
    ) -> None:
        self.requested.fill(0.0)
        self.output.fill(0.0)
        self._clear_motion()
        self.tare_values = np.asarray(tuple(tare_values), dtype=float)
        self.tare_frame = tare_frame
        self.tare_generation = tare_generation
        self.last_sample_identity = sample_identity
        self.last_sample_source_time = sample_source_time
        self.last_output_time = (
            None
            if output_time_sec is None
            else _finite_time(output_time_sec, name="output_time_sec")
        )
        self.frozen = False
        self.reason = "reset"
        self._clear_event()

    def freeze(self, reason: str) -> None:
        self._clear_motion()
        self.frozen = True
        self.reason = str(reason).strip() or "frozen"
        self._clear_event()

    def resume(self) -> None:
        if self.tare_generation is None:
            self.freeze("not_tared")
            return
        self.last_sample_identity = None
        self.last_sample_source_time = None
        self._clear_motion()
        self.frozen = False
        self.reason = "resumed_awaiting_sample"

    def rebase(
        self,
        values: Iterable[float],
        *,
        value_name: str,
        component_limits: Iterable[float],
        enabled_axes: Iterable[bool],
        output_time_sec: float | None,
        preserve_requested: bool,
        backwards_error: str,
    ) -> None:
        value = _clamp_bounded_vector(
            _finite_vector(values, size=3, name=value_name),
            component_limits=np.asarray(tuple(component_limits), dtype=float),
            enabled_axes=np.asarray(tuple(enabled_axes), dtype=bool),
        )
        output_time: float | None = None
        if output_time_sec is not None:
            output_time = _finite_time(output_time_sec, name="output_time_sec")
            if (
                self.last_output_time is not None
                and output_time < self.last_output_time
            ):
                raise ValueError(backwards_error)
        if not preserve_requested:
            self.requested = value.copy()
        self.output = value.copy()
        self.velocity.fill(0.0)
        self.output_velocity.fill(0.0)
        self.passive_return_active = False
        if output_time is not None:
            self.last_output_time = output_time
        self.reason = "realized_rebase"
        self._clear_event()

    def prepare_sample(
        self,
        *,
        sample_identity: tuple[int, float],
        sample_source_time: float,
        sample_frame: str,
        sample_tare_generation: int,
        now_sec: float,
        active: bool,
        enabled: bool,
        stale_after_sec: float,
        max_input_dt_sec: float,
        dt_exceeded_reason: str,
    ) -> float | None:
        """Admit one sample and return its integration dt when it is fresh."""

        now = _finite_time(now_sec, name="now_sec")
        self._clear_event()
        if not active:
            self.freeze("inactive")
            return None
        if self.frozen:
            return None
        if not enabled:
            self.reason = "disabled"
            return None
        if self.tare_values is None or self.tare_generation is None:
            self.freeze("not_tared")
            return None
        if sample_tare_generation != self.tare_generation:
            self.freeze("tare_generation_mismatch")
            return None
        if sample_frame != self.tare_frame:
            self.freeze("tare_frame_mismatch")
            return None
        if sample_source_time > now:
            self.freeze("sample_from_future")
            return None
        if max(0.0, now - sample_source_time) > stale_after_sec:
            self.freeze("sample_stale")
            return None
        if sample_identity == self.last_sample_identity:
            self.reason = "duplicate_sample"
            return None
        if self.last_sample_source_time is None:
            self.last_sample_identity = sample_identity
            self.last_sample_source_time = sample_source_time
            self.reason = "sample_clock_rearmed"
            return None
        if sample_source_time <= self.last_sample_source_time:
            self.freeze("sample_time_not_increasing")
            return None
        dt = sample_source_time - self.last_sample_source_time
        if dt > max_input_dt_sec:
            self.freeze(dt_exceeded_reason)
            return None
        return dt

    def finish_sample(
        self,
        *,
        sample_identity: tuple[int, float],
        sample_source_time: float,
        integration_dt_sec: float,
    ) -> None:
        self.last_sample_identity = sample_identity
        self.last_sample_source_time = sample_source_time
        self.reason = "sample_integrated"
        self.new_sample_integrated = True
        self.integration_dt_sec = integration_dt_sec

    def sample_output(
        self,
        *,
        now_sec: float,
        active: bool,
        stale_after_sec: float,
        enabled_axes: Iterable[bool],
        output_follow_tau_sec: float,
        output_slew_limits: Iterable[float],
        acceleration_limits: Iterable[float],
        absolute_limits: Iterable[float],
    ) -> None:
        """Advance the common output clock and final bounded-vector envelope."""

        now = _finite_time(now_sec, name="now_sec")
        self._clear_event()
        if not active:
            self.freeze("inactive")
            return
        if self.frozen:
            return
        if self.last_sample_source_time is None:
            self.reason = "awaiting_fresh_sample"
            return
        if now - self.last_sample_source_time > stale_after_sec:
            self.freeze("sample_stale")
            return
        if self.last_output_time is None:
            self.last_output_time = now
            self.reason = "output_clock_armed"
            return
        if now == self.last_output_time:
            self.reason = "duplicate_output_time"
            return
        if now < self.last_output_time:
            self.freeze("output_time_not_increasing")
            return

        dt = now - self.last_output_time
        enabled = np.asarray(tuple(enabled_axes), dtype=bool)
        output_slew = np.asarray(tuple(output_slew_limits), dtype=float)
        output_target = self.requested
        if output_follow_tau_sec > 0.0 and self.passive_return_active:
            output_target = np.zeros(3, dtype=float)
            for axis in range(3):
                if not enabled[axis] or abs(self.output[axis]) <= 1e-15:
                    continue
                requested = self.requested[axis]
                current = self.output[axis]
                if requested * current > 0.0:
                    output_target[axis] = math.copysign(
                        min(abs(requested), abs(current)),
                        current,
                    )
        raw_delta = output_target - self.output
        if output_follow_tau_sec > 0.0:
            raw_delta = raw_delta * (
                1.0 - math.exp(-dt / output_follow_tau_sec)
            )
        delta = _clamp_bounded_vector(
            raw_delta,
            component_limits=output_slew * dt,
            enabled_axes=enabled,
        )
        acceleration = np.asarray(tuple(acceleration_limits), dtype=float)
        if np.any(acceleration[enabled] > 0.0):
            desired_velocity = delta / dt
            velocity_delta = _clamp_bounded_vector(
                desired_velocity - self.output_velocity,
                component_limits=acceleration * dt,
                enabled_axes=enabled,
            )
            output_velocity = _clamp_bounded_vector(
                self.output_velocity + velocity_delta,
                component_limits=output_slew,
                enabled_axes=enabled,
            )
            delta = output_velocity * dt
        previous_output = self.output.copy()
        candidate_output = _clamp_bounded_vector(
            self.output + delta,
            component_limits=np.asarray(tuple(absolute_limits), dtype=float),
            enabled_axes=enabled,
        )
        if np.any(acceleration[enabled] > 0.0) and self.passive_return_active:
            candidate_output = np.where(
                candidate_output * previous_output > 0.0,
                np.sign(previous_output)
                * np.minimum(
                    np.abs(candidate_output),
                    np.abs(previous_output),
                ),
                0.0,
            )
            if float(np.linalg.norm(candidate_output)) > float(
                np.linalg.norm(previous_output)
            ):
                candidate_output = previous_output
        self.output = candidate_output
        self.output_velocity = (self.output - previous_output) / dt
        self.last_output_time = now
        self.reason = "output_sampled"

    def sample_age(self, *, now_sec: float | None) -> float | None:
        if now_sec is None or self.last_sample_source_time is None:
            return None
        now = _finite_time(now_sec, name="now_sec")
        return max(0.0, now - self.last_sample_source_time)
