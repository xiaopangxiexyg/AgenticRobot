from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .timeline import OutOfOrderReferenceError


POSITION_COMMAND_PLANNER_LINEAR = "linear"
POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED = "acceleration_limited"
POSITION_COMMAND_PLANNER_CHOICES = (
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
)
DEFAULT_POSITION_VELOCITY_RAMP_SEC = 0.10
DEFAULT_POSITION_DEGRADED_SPEED_SCALE = 0.50


def normalize_position_command_planner(value: object) -> str:
    text = str(value or POSITION_COMMAND_PLANNER_LINEAR).strip().lower().replace("-", "_")
    aliases = {
        "position": POSITION_COMMAND_PLANNER_LINEAR,
        "pure_position": POSITION_COMMAND_PLANNER_LINEAR,
        "velocity": POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        "velocity_planned": POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        "accel_limited": POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    }
    normalized = aliases.get(text, text)
    if normalized not in POSITION_COMMAND_PLANNER_CHOICES:
        raise ValueError(
            f"Unsupported position command planner {value!r}; expected one of "
            f"{POSITION_COMMAND_PLANNER_CHOICES}."
        )
    return normalized


def _readonly_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1).copy()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    vector.setflags(write=False)
    return vector


def _positive_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = _readonly_vector(value, name=name)
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return vector


def _finite_time(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True)
class JointTrajectoryLimits:
    """Per-joint limits for the sim/real-neutral online qpos planner."""

    max_velocity_rad_s: Sequence[float] | np.ndarray
    max_acceleration_rad_s2: Sequence[float] | np.ndarray
    max_command_step_rad: Sequence[float] | np.ndarray | None = None

    def __post_init__(self) -> None:
        max_velocity = _positive_vector(
            self.max_velocity_rad_s,
            name="max_velocity_rad_s",
        )
        max_acceleration = _positive_vector(
            self.max_acceleration_rad_s2,
            name="max_acceleration_rad_s2",
        )
        if max_velocity.shape != max_acceleration.shape:
            raise ValueError("velocity and acceleration limits must have matching shapes.")
        max_step = (
            None
            if self.max_command_step_rad is None
            else _positive_vector(
                self.max_command_step_rad,
                name="max_command_step_rad",
            )
        )
        if max_step is not None and max_step.shape != max_velocity.shape:
            raise ValueError("max_command_step_rad must match the velocity-limit shape.")
        object.__setattr__(self, "max_velocity_rad_s", max_velocity)
        object.__setattr__(self, "max_acceleration_rad_s2", max_acceleration)
        object.__setattr__(self, "max_command_step_rad", max_step)


@dataclass(frozen=True)
class JointTrajectoryProposal:
    """One transactional high-rate command proposal.

    A transport must call ``commit`` only after the exact qpos was accepted by
    its sink. A rejected or superseded send calls ``reject``. This prevents SDK
    failures from advancing the trajectory's position or velocity state.
    """

    token: int
    generation: int
    target_sequence: int
    sample_monotonic_time: float
    integration_dt_sec: float
    q_command: Sequence[float] | np.ndarray
    q_target: Sequence[float] | np.ndarray
    qd_command_rad_s: Sequence[float] | np.ndarray
    qd_reference_rad_s: Sequence[float] | np.ndarray
    qd_desired_rad_s: Sequence[float] | np.ndarray
    qdd_command_rad_s2: Sequence[float] | np.ndarray
    speed_scale: float
    mode: str
    reason: str
    quick_stop: bool
    acceleration_limited: bool
    velocity_limited: bool
    command_step_limited: bool
    safety_envelope_limited: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _non_negative_int(self.token, name="token"))
        object.__setattr__(
            self,
            "generation",
            _non_negative_int(self.generation, name="generation"),
        )
        object.__setattr__(
            self,
            "target_sequence",
            _non_negative_int(self.target_sequence, name="target_sequence"),
        )
        object.__setattr__(
            self,
            "sample_monotonic_time",
            _finite_time(self.sample_monotonic_time, name="sample_monotonic_time"),
        )
        dt = _finite_time(self.integration_dt_sec, name="integration_dt_sec")
        if dt < 0.0:
            raise ValueError("integration_dt_sec must be non-negative.")
        object.__setattr__(self, "integration_dt_sec", dt)
        vectors = {
            name: _readonly_vector(getattr(self, name), name=name)
            for name in (
                "q_command",
                "q_target",
                "qd_command_rad_s",
                "qd_reference_rad_s",
                "qd_desired_rad_s",
                "qdd_command_rad_s2",
            )
        }
        shape = vectors["q_command"].shape
        if any(vector.shape != shape for vector in vectors.values()):
            raise ValueError("all trajectory proposal vectors must have matching shapes.")
        for name, vector in vectors.items():
            object.__setattr__(self, name, vector)
        scale = float(self.speed_scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("speed_scale must be finite and in [0, 1].")
        object.__setattr__(self, "speed_scale", scale)
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "quick_stop", bool(self.quick_stop))
        object.__setattr__(self, "acceleration_limited", bool(self.acceleration_limited))
        object.__setattr__(self, "velocity_limited", bool(self.velocity_limited))
        object.__setattr__(self, "command_step_limited", bool(self.command_step_limited))
        object.__setattr__(
            self,
            "safety_envelope_limited",
            bool(self.safety_envelope_limited),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def debug_metadata(self) -> dict[str, Any]:
        return {
            "planner_mode": POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            "trajectory_proposal_token": int(self.token),
            "trajectory_generation": int(self.generation),
            "trajectory_target_sequence": int(self.target_sequence),
            "trajectory_mode": self.mode,
            "trajectory_reason": self.reason,
            "trajectory_quick_stop": bool(self.quick_stop),
            "trajectory_speed_scale": float(self.speed_scale),
            "trajectory_integration_dt_sec": float(self.integration_dt_sec),
            "trajectory_qd_reference_rad_s": self.qd_reference_rad_s.tolist(),
            "trajectory_qd_desired_rad_s": self.qd_desired_rad_s.tolist(),
            "trajectory_qd_command_rad_s": self.qd_command_rad_s.tolist(),
            "trajectory_qdd_command_rad_s2": self.qdd_command_rad_s2.tolist(),
            "trajectory_position_error_rad": (self.q_target - self.q_command).tolist(),
            "trajectory_acceleration_limited": bool(self.acceleration_limited),
            "trajectory_velocity_limited": bool(self.velocity_limited),
            "trajectory_command_step_limited": bool(self.command_step_limited),
            "trajectory_safety_envelope_limited": bool(self.safety_envelope_limited),
            **dict(self.metadata),
        }


class AccelerationLimitedJointTrajectoryPlanner:
    """Preemptible qpos planner with continuous commanded velocity.

    Safe qpos references are accepted at an arbitrary outer rate (75 Hz in the
    current teleop profile) and sampled at a higher sink rate (150 Hz). Target
    velocity is estimated causally from accepted safe-reference history. A new
    target never resets the current planned velocity; its desired velocity is
    reached under the configured acceleration bound.

    The componentwise safety envelope is the convex interval containing the
    last committed command, the previous safe target, the active safe target,
    and a retained braking endpoint.  The latter keeps the last already-safe
    endpoint in the direction of motion until that joint has stopped or
    reversed.  Without it, rapidly reversing outer references can shrink the
    interval faster than the inherited velocity can decelerate and force a
    one-tick acceleration discontinuity.  The envelope prevents target-
    velocity feed-forward from extrapolating beyond already accepted
    endpoints. It does not claim continuous-space collision certification
    between endpoint configurations.
    """

    def __init__(
        self,
        limits: JointTrajectoryLimits,
        *,
        velocity_ramp_sec: float = DEFAULT_POSITION_VELOCITY_RAMP_SEC,
        reference_period_sec: float | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(limits, JointTrajectoryLimits):
            raise TypeError("limits must be JointTrajectoryLimits.")
        ramp = float(velocity_ramp_sec)
        if not math.isfinite(ramp) or ramp <= 0.0:
            raise ValueError("velocity_ramp_sec must be finite and positive.")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        if reference_period_sec is None:
            normalized_reference_period = None
        else:
            normalized_reference_period = float(reference_period_sec)
            if (
                not math.isfinite(normalized_reference_period)
                or normalized_reference_period <= 0.0
            ):
                raise ValueError("reference_period_sec must be finite and positive, or None.")
        self.limits = limits
        self.velocity_ramp_sec = ramp
        self.reference_period_sec = normalized_reference_period
        self._clock = monotonic_clock
        self._lock = threading.RLock()
        self._q: np.ndarray | None = None
        self._qd: np.ndarray | None = None
        self._qdd: np.ndarray | None = None
        self._last_commit_monotonic: float | None = None
        self._target_q: np.ndarray | None = None
        self._previous_target_q: np.ndarray | None = None
        self._braking_guard_q: np.ndarray | None = None
        self._braking_guard_direction: np.ndarray | None = None
        self._target_qd_raw: np.ndarray | None = None
        self._target_qd_reference: np.ndarray | None = None
        self._target_qd_desired: np.ndarray | None = None
        self._target_qd_desired_raw: np.ndarray | None = None
        self._target_tracking_error: np.ndarray | None = None
        self._target_braking_offset: np.ndarray | None = None
        self._target_reference_lead_reserve: np.ndarray | None = None
        self._target_reference_period_sec = 0.0
        self._target_sequence: int | None = None
        self._target_published_monotonic: float | None = None
        self._last_safe_target_q: np.ndarray | None = None
        self._last_safe_target_published_monotonic: float | None = None
        self._last_observed_reference_dt_sec: float | None = None
        self._speed_scale = 1.0
        self._mode = "waiting_for_target"
        self._reason = "waiting_for_target"
        self._quick_stop = False
        self._anchor_resync_count = 0
        self._generation = 0
        self._next_token = 0
        self._pending: JointTrajectoryProposal | None = None
        self._last_proposed_generation: int | None = None

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.limits.max_velocity_rad_s).size)

    def reset(
        self,
        q_command: Sequence[float] | np.ndarray,
        *,
        monotonic_time: float | None = None,
    ) -> None:
        now = self._resolve_time(monotonic_time)
        qpos = self._vector(q_command, name="q_command")
        with self._lock:
            self._q = qpos
            self._qd = np.zeros_like(qpos)
            self._qdd = np.zeros_like(qpos)
            self._last_commit_monotonic = now
            self._target_q = None
            self._previous_target_q = None
            self._braking_guard_q = qpos.copy()
            self._braking_guard_direction = np.zeros_like(qpos, dtype=np.int8)
            self._target_qd_raw = None
            self._target_qd_reference = None
            self._target_qd_desired = None
            self._target_qd_desired_raw = None
            self._target_tracking_error = None
            self._target_braking_offset = None
            self._target_reference_lead_reserve = None
            self._target_reference_period_sec = 0.0
            self._target_sequence = None
            self._target_published_monotonic = None
            self._last_safe_target_q = None
            self._last_safe_target_published_monotonic = None
            self._last_observed_reference_dt_sec = None
            self._speed_scale = 1.0
            self._mode = "waiting_for_target"
            self._reason = "waiting_for_target"
            self._quick_stop = False
            self._generation += 1
            self._pending = None
            self._last_proposed_generation = None

    def set_max_command_step(
        self,
        max_command_step_rad: Sequence[float] | np.ndarray,
    ) -> None:
        max_step = _positive_vector(max_command_step_rad, name="max_command_step_rad")
        if max_step.shape != np.asarray(self.limits.max_velocity_rad_s).shape:
            raise ValueError("max_command_step_rad has the wrong shape.")
        with self._lock:
            self.limits = JointTrajectoryLimits(
                max_velocity_rad_s=self.limits.max_velocity_rad_s,
                max_acceleration_rad_s2=self.limits.max_acceleration_rad_s2,
                max_command_step_rad=max_step,
            )

    def set_reference_period_sec(self, reference_period_sec: float) -> None:
        period = float(reference_period_sec)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("reference_period_sec must be finite and positive.")
        with self._lock:
            self.reference_period_sec = period

    def accept_target(
        self,
        q_target: Sequence[float] | np.ndarray,
        *,
        sequence: int,
        published_monotonic_time: float,
        monotonic_time: float | None = None,
        speed_scale: float = 1.0,
        mode: str = "tracking",
        reason: str = "normal",
        anchor_q_command: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        now = self._resolve_time(monotonic_time)
        target = self._vector(q_target, name="q_target")
        normalized_sequence = _non_negative_int(sequence, name="sequence")
        published = _finite_time(
            published_monotonic_time,
            name="published_monotonic_time",
        )
        scale = float(speed_scale)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale must be finite and in (0, 1].")
        normalized_mode = str(mode).strip() or "tracking"
        normalized_reason = str(reason).strip() or "normal"
        anchor = (
            None
            if anchor_q_command is None
            else self._vector(anchor_q_command, name="anchor_q_command")
        )
        with self._lock:
            self._require_initialized()
            assert self._q is not None
            assert self._qd is not None
            assert self._qdd is not None
            assert self._last_commit_monotonic is not None
            if published > now:
                raise ValueError("cannot accept a target published in the monotonic future.")
            if now < self._last_commit_monotonic:
                raise ValueError("monotonic_time cannot precede the last committed command.")
            if self._target_sequence is not None and normalized_sequence <= self._target_sequence:
                raise OutOfOrderReferenceError(
                    f"target sequence {normalized_sequence} is not newer than "
                    f"{self._target_sequence}."
                )
            if (
                self._target_published_monotonic is not None
                and published < self._target_published_monotonic
            ):
                raise OutOfOrderReferenceError("target published_monotonic_time moved backwards.")

            anchor_resynced = False
            if anchor is not None and not np.allclose(anchor, self._q, rtol=0.0, atol=1e-10):
                self._q = anchor.copy()
                self._qd = np.zeros_like(anchor)
                self._qdd = np.zeros_like(anchor)
                self._braking_guard_q = anchor.copy()
                self._braking_guard_direction = np.zeros_like(anchor, dtype=np.int8)
                self._last_commit_monotonic = now
                self._last_safe_target_q = None
                self._last_safe_target_published_monotonic = None
                self._last_observed_reference_dt_sec = None
                self._anchor_resync_count += 1
                anchor_resynced = True

            previous_safe_target = (
                None if self._last_safe_target_q is None else self._last_safe_target_q.copy()
            )
            previous_endpoint = (
                self._q.copy()
                if previous_safe_target is None
                else previous_safe_target.copy()
            )

            # Retain an already accepted endpoint in the current velocity
            # direction.  A 75 Hz reference may reverse every other 150 Hz
            # transport tick; replacing the previous endpoint immediately in
            # that case can leave less stopping distance than qdd allows.
            assert self._braking_guard_q is not None
            assert self._braking_guard_direction is not None
            old_guard_q = self._braking_guard_q.copy()
            old_guard_direction = self._braking_guard_direction.copy()
            velocity_epsilon = 1e-12
            moving_positive = self._qd > velocity_epsilon
            moving_negative = self._qd < -velocity_epsilon
            guard_q = self._q.copy()
            guard_direction = np.zeros_like(old_guard_direction, dtype=np.int8)
            positive_candidates = np.maximum.reduce(
                (self._q, previous_endpoint, target)
            )
            negative_candidates = np.minimum.reduce(
                (self._q, previous_endpoint, target)
            )
            preserve_positive = moving_positive & (old_guard_direction > 0)
            preserve_negative = moving_negative & (old_guard_direction < 0)
            positive_candidates = np.where(
                preserve_positive,
                np.maximum(positive_candidates, old_guard_q),
                positive_candidates,
            )
            negative_candidates = np.where(
                preserve_negative,
                np.minimum(negative_candidates, old_guard_q),
                negative_candidates,
            )
            guard_q = np.where(moving_positive, positive_candidates, guard_q)
            guard_q = np.where(moving_negative, negative_candidates, guard_q)
            guard_direction = np.where(moving_positive, 1, guard_direction)
            guard_direction = np.where(moving_negative, -1, guard_direction)
            target_qd_raw = np.zeros_like(target)
            if (
                self._last_safe_target_q is not None
                and self._last_safe_target_published_monotonic is not None
            ):
                reference_dt = published - self._last_safe_target_published_monotonic
                if reference_dt <= 0.0:
                    raise OutOfOrderReferenceError(
                        "safe target published_monotonic_time must advance."
                    )
                target_qd_raw = (
                    target - self._last_safe_target_q
                ) / reference_dt
                self._last_observed_reference_dt_sec = float(reference_dt)
            max_velocity = np.asarray(self.limits.max_velocity_rad_s, dtype=float)
            target_qd_raw = np.clip(target_qd_raw, -max_velocity, max_velocity)
            max_acceleration = np.asarray(
                self.limits.max_acceleration_rad_s2,
                dtype=float,
            )
            scaled_velocity = max_velocity * scale
            target_qd_reference = np.clip(
                target_qd_raw,
                -scaled_velocity,
                scaled_velocity,
            )
            reference_period = self.reference_period_sec
            if reference_period is None:
                observed_period = self._last_observed_reference_dt_sec
                reference_period = (
                    0.0
                    if observed_period is None
                    else min(float(observed_period), self.velocity_ramp_sec)
                )
            braking_offset = (
                np.sign(self._qd)
                * np.square(self._qd)
                / (2.0 * max_acceleration)
            )
            reference_lead_reserve = target_qd_reference * float(reference_period)
            tracking_error = (
                target
                - self._q
                - braking_offset
                - reference_lead_reserve
            )
            target_qd_desired_raw = (
                target_qd_reference + tracking_error / self.velocity_ramp_sec
            )
            target_qd_desired = np.clip(
                target_qd_desired_raw,
                -scaled_velocity,
                scaled_velocity,
            )

            self._previous_target_q = previous_endpoint
            self._braking_guard_q = guard_q
            self._braking_guard_direction = guard_direction.astype(np.int8, copy=False)
            self._target_q = target.copy()
            self._target_qd_raw = target_qd_raw
            self._target_qd_reference = target_qd_reference
            self._target_qd_desired = target_qd_desired
            self._target_qd_desired_raw = target_qd_desired_raw
            self._target_tracking_error = tracking_error
            self._target_braking_offset = braking_offset
            self._target_reference_lead_reserve = reference_lead_reserve
            self._target_reference_period_sec = float(reference_period)
            self._target_sequence = normalized_sequence
            self._target_published_monotonic = published
            self._last_safe_target_q = target.copy()
            self._last_safe_target_published_monotonic = published
            self._speed_scale = scale
            self._mode = normalized_mode
            self._reason = normalized_reason
            self._quick_stop = False
            self._generation += 1
            self._pending = None
            # Kept in snapshot/debug without adding a public mutable field.
            self._last_anchor_resynced = anchor_resynced

    def quick_stop(
        self,
        *,
        sequence: int,
        monotonic_time: float | None = None,
        reason: str,
    ) -> None:
        """Immediately zero planner velocity and hold the last committed qpos."""

        now = self._resolve_time(monotonic_time)
        normalized_sequence = _non_negative_int(sequence, name="sequence")
        normalized_reason = str(reason).strip() or "quick_stop"
        with self._lock:
            self._require_initialized()
            assert self._q is not None
            assert self._qd is not None
            assert self._qdd is not None
            assert self._last_commit_monotonic is not None
            if now < self._last_commit_monotonic:
                raise ValueError("monotonic_time cannot precede the last committed command.")
            if self._target_sequence is not None and normalized_sequence <= self._target_sequence:
                raise OutOfOrderReferenceError(
                    f"quick-stop sequence {normalized_sequence} is not newer than "
                    f"{self._target_sequence}."
                )
            self._target_q = self._q.copy()
            self._previous_target_q = self._q.copy()
            self._braking_guard_q = self._q.copy()
            self._braking_guard_direction = np.zeros_like(self._q, dtype=np.int8)
            self._target_qd_raw = np.zeros_like(self._q)
            self._target_qd_reference = np.zeros_like(self._q)
            self._target_qd_desired = np.zeros_like(self._q)
            self._target_qd_desired_raw = np.zeros_like(self._q)
            self._target_tracking_error = np.zeros_like(self._q)
            self._target_braking_offset = np.zeros_like(self._q)
            self._target_reference_lead_reserve = np.zeros_like(self._q)
            self._target_reference_period_sec = float(
                self.reference_period_sec
                if self.reference_period_sec is not None
                else self._last_observed_reference_dt_sec or 0.0
            )
            self._target_sequence = normalized_sequence
            self._target_published_monotonic = now
            self._qd = np.zeros_like(self._q)
            self._qdd = np.zeros_like(self._q)
            self._last_commit_monotonic = now
            self._speed_scale = 0.0
            self._mode = "quick_stop"
            self._reason = normalized_reason
            self._quick_stop = True
            self._generation += 1
            self._pending = None
            self._last_anchor_resynced = False

    def propose(self, *, monotonic_time: float | None = None) -> JointTrajectoryProposal:
        now = self._resolve_time(monotonic_time)
        with self._lock:
            self._require_target()
            assert self._q is not None
            assert self._qd is not None
            assert self._qdd is not None
            assert self._last_commit_monotonic is not None
            assert self._target_q is not None
            assert self._previous_target_q is not None
            assert self._braking_guard_q is not None
            assert self._braking_guard_direction is not None
            assert self._target_qd_raw is not None
            assert self._target_qd_reference is not None
            assert self._target_qd_desired is not None
            assert self._target_qd_desired_raw is not None
            assert self._target_tracking_error is not None
            assert self._target_braking_offset is not None
            assert self._target_reference_lead_reserve is not None
            assert self._target_sequence is not None
            if now < self._last_commit_monotonic:
                raise ValueError("monotonic_time cannot precede the last committed command.")
            dt = max(0.0, now - self._last_commit_monotonic)
            max_velocity = np.asarray(self.limits.max_velocity_rad_s, dtype=float)
            max_acceleration = np.asarray(
                self.limits.max_acceleration_rad_s2,
                dtype=float,
            )
            envelope_lower = np.minimum.reduce(
                (
                    self._q,
                    self._previous_target_q,
                    self._target_q,
                    self._braking_guard_q,
                )
            )
            envelope_upper = np.maximum.reduce(
                (
                    self._q,
                    self._previous_target_q,
                    self._target_q,
                    self._braking_guard_q,
                )
            )

            if self._quick_stop:
                q_command = self._q.copy()
                qd_reference = np.zeros_like(self._q)
                qd_desired = np.zeros_like(self._q)
                qd_command = np.zeros_like(self._q)
                qdd_command = np.zeros_like(self._q)
                acceleration_limited = False
                velocity_limited = False
                command_step_limited = False
                envelope_limited = False
                endpoint_velocity_limited = False
                safety_velocity_infeasible = False
                reference_period = self.reference_period_sec
                if reference_period is None:
                    reference_period = self._target_reference_period_sec
                braking_offset = np.zeros_like(self._q)
                reference_lead_reserve = np.zeros_like(self._q)
                raw_desired = qd_desired.copy()
            else:
                qd_reference = self._target_qd_reference.copy()
                raw_desired = self._target_qd_desired_raw.copy()
                qd_desired = self._target_qd_desired.copy()
                reference_period = self._target_reference_period_sec
                braking_offset = self._target_braking_offset.copy()
                reference_lead_reserve = self._target_reference_lead_reserve.copy()
                velocity_limited = bool(
                    np.any(np.abs(raw_desired - qd_desired) > 1e-12)
                )
                if dt <= 0.0:
                    qd_command = self._qd.copy()
                    qdd_command = np.zeros_like(self._q)
                    acceleration_limited = bool(
                        np.any(np.abs(qd_desired - self._qd) > 1e-12)
                    )
                    command_step_limited = False
                    envelope_limited = False
                    endpoint_velocity_limited = False
                    safety_velocity_infeasible = False
                else:
                    distance_to_lower = np.maximum(0.0, self._q - envelope_lower)
                    distance_to_upper = np.maximum(0.0, envelope_upper - self._q)

                    # For semi-implicit q_next=q+v_next*dt, reserve enough
                    # distance after this tick to brake from v_next to zero:
                    #   v*dt + v^2/(2*a) <= remaining_distance.
                    # Solving the quadratic gives a velocity interval that
                    # avoids the endpoint hard-clamp/velocity discontinuity of
                    # the previous implementation.
                    accel_dt = max_acceleration * dt
                    safe_positive_velocity = -accel_dt + np.sqrt(
                        accel_dt * accel_dt
                        + 2.0 * max_acceleration * distance_to_upper
                    )
                    safe_negative_speed = -accel_dt + np.sqrt(
                        accel_dt * accel_dt
                        + 2.0 * max_acceleration * distance_to_lower
                    )
                    safety_velocity_lower = -safe_negative_speed
                    safety_velocity_upper = safe_positive_velocity

                    endpoint_bounded_desired = np.clip(
                        qd_desired,
                        safety_velocity_lower,
                        safety_velocity_upper,
                    )
                    endpoint_velocity_limited = bool(
                        np.any(
                            np.abs(endpoint_bounded_desired - qd_desired) > 1e-12
                        )
                    )

                    command_step_limited = False
                    bounded_desired = endpoint_bounded_desired
                    max_step = self.limits.max_command_step_rad
                    if max_step is not None:
                        step_velocity = np.asarray(max_step, dtype=float) / dt
                        step_bounded_desired = np.clip(
                            bounded_desired,
                            -step_velocity,
                            step_velocity,
                        )
                        command_step_limited = bool(
                            np.any(
                                np.abs(step_bounded_desired - bounded_desired) > 1e-12
                            )
                        )
                        bounded_desired = step_bounded_desired
                        safety_velocity_lower = np.maximum(
                            safety_velocity_lower,
                            -step_velocity,
                        )
                        safety_velocity_upper = np.minimum(
                            safety_velocity_upper,
                            step_velocity,
                        )

                    raw_qdd = (bounded_desired - self._qd) / dt
                    qdd_command = np.clip(
                        raw_qdd,
                        -max_acceleration,
                        max_acceleration,
                    )
                    acceleration_limited = bool(
                        np.any(np.abs(raw_qdd - qdd_command) > 1e-12)
                    )
                    qd_command = self._qd + qdd_command * dt
                    # Normally the previous tick's braking invariant makes the
                    # acceleration-reachable velocity fall inside this safety
                    # interval. A suddenly closer preempting endpoint can make
                    # that impossible; endpoint/step safety then overrides qdd
                    # and is reported explicitly instead of being hidden.
                    safe_qd_command = np.clip(
                        qd_command,
                        safety_velocity_lower,
                        safety_velocity_upper,
                    )
                    safety_velocity_infeasible = bool(
                        np.any(np.abs(safe_qd_command - qd_command) > 1e-12)
                    )
                    qd_command = safe_qd_command
                    qdd_command = (qd_command - self._qd) / dt
                    q_command = self._q + qd_command * dt
                    enveloped_q = np.clip(
                        q_command,
                        envelope_lower,
                        envelope_upper,
                    )
                    numerical_envelope_clamp = bool(
                        np.any(np.abs(q_command - enveloped_q) > 1e-12)
                    )
                    if numerical_envelope_clamp:
                        q_command = enveloped_q
                        qd_command = (q_command - self._q) / dt
                        qdd_command = (qd_command - self._qd) / dt
                        safety_velocity_infeasible = True
                    envelope_limited = bool(
                        endpoint_velocity_limited
                        or safety_velocity_infeasible
                        or numerical_envelope_clamp
                    )

                # A reduced speed mode decelerates under qdd rather than
                # discontinuously clipping a previously faster trajectory.
                qd_command = np.clip(qd_command, -max_velocity, max_velocity)
                if dt <= 0.0:
                    q_command = self._q.copy()

            token = self._next_token
            self._next_token += 1
            target_preempted = self._last_proposed_generation != self._generation
            proposal = JointTrajectoryProposal(
                token=token,
                generation=self._generation,
                target_sequence=self._target_sequence,
                sample_monotonic_time=now,
                integration_dt_sec=dt,
                q_command=q_command,
                q_target=self._target_q,
                qd_command_rad_s=qd_command,
                qd_reference_rad_s=qd_reference,
                qd_desired_rad_s=qd_desired,
                qdd_command_rad_s2=qdd_command,
                speed_scale=self._speed_scale,
                mode=self._mode,
                reason=self._reason,
                quick_stop=self._quick_stop,
                acceleration_limited=acceleration_limited,
                velocity_limited=velocity_limited,
                command_step_limited=command_step_limited,
                safety_envelope_limited=envelope_limited,
                metadata={
                    "trajectory_target_velocity_raw_rad_s": self._target_qd_raw.tolist(),
                    "trajectory_desired_velocity_raw_rad_s": raw_desired.tolist(),
                    "trajectory_velocity_ramp_sec": float(self.velocity_ramp_sec),
                    "trajectory_max_velocity_rad_s": max_velocity.tolist(),
                    "trajectory_max_acceleration_rad_s2": max_acceleration.tolist(),
                    "trajectory_reference_period_sec": float(reference_period),
                    "trajectory_braking_offset_rad": braking_offset.tolist(),
                    "trajectory_reference_lead_reserve_rad": (
                        reference_lead_reserve.tolist()
                    ),
                    "trajectory_max_command_step_rad": (
                        None
                        if self.limits.max_command_step_rad is None
                        else self.limits.max_command_step_rad.tolist()
                    ),
                    "trajectory_qdd_limit_exceeded_after_safety_clamp": bool(
                        np.any(np.abs(qdd_command) > max_acceleration + 1e-12)
                    ),
                    "trajectory_endpoint_velocity_limited": bool(
                        endpoint_velocity_limited
                    ),
                    "trajectory_safety_velocity_infeasible": bool(
                        safety_velocity_infeasible
                    ),
                    "trajectory_anchor_resynced": bool(
                        getattr(self, "_last_anchor_resynced", False)
                    ),
                    "trajectory_anchor_resync_count": int(self._anchor_resync_count),
                    "trajectory_target_preempted": bool(target_preempted),
                    "trajectory_braking_guard_qpos": self._braking_guard_q.tolist(),
                    "trajectory_braking_guard_direction": (
                        self._braking_guard_direction.astype(int).tolist()
                    ),
                    "trajectory_safety_envelope_lower_qpos": envelope_lower.tolist(),
                    "trajectory_safety_envelope_upper_qpos": envelope_upper.tolist(),
                },
            )
            self._pending = proposal
            self._last_proposed_generation = self._generation
            return proposal

    def commit(
        self,
        proposal: JointTrajectoryProposal,
        *,
        accepted_monotonic_time: float | None = None,
        accepted_q_command: Sequence[float] | np.ndarray | None = None,
        allow_bounded_accepted_adjustment: bool = False,
    ) -> None:
        accepted_time = self._resolve_time(accepted_monotonic_time)
        with self._lock:
            self._require_matching_pending(proposal)
            if accepted_time < proposal.sample_monotonic_time:
                raise ValueError("accepted_monotonic_time cannot precede proposal sampling.")
            accepted_q = (
                proposal.q_command.copy()
                if accepted_q_command is None
                else self._vector(accepted_q_command, name="accepted_q_command")
            )
            adjusted = not np.allclose(
                accepted_q,
                proposal.q_command,
                rtol=0.0,
                atol=1e-10,
            )
            if adjusted and not bool(allow_bounded_accepted_adjustment):
                raise ValueError("accepted_q_command differs from the transactional proposal.")
            if adjusted:
                assert self._q is not None
                lower = np.minimum(self._q, proposal.q_command) - 1e-12
                upper = np.maximum(self._q, proposal.q_command) + 1e-12
                if np.any(accepted_q < lower) or np.any(accepted_q > upper):
                    raise ValueError(
                        "adjusted accepted_q_command must remain between the previous "
                        "accepted command and the transactional proposal."
                    )
                assert self._qd is not None
                assert self._last_commit_monotonic is not None
                dt = max(
                    0.0,
                    float(proposal.sample_monotonic_time)
                    - float(self._last_commit_monotonic),
                )
                if dt > 0.0:
                    accepted_qd = (accepted_q - self._q) / dt
                    accepted_qdd = (accepted_qd - self._qd) / dt
                else:
                    accepted_qd = self._qd.copy()
                    accepted_qdd = np.zeros_like(self._qd)
            else:
                accepted_qd = proposal.qd_command_rad_s.copy()
                accepted_qdd = proposal.qdd_command_rad_s2.copy()
            self._q = accepted_q.copy()
            self._qd = accepted_qd
            self._qdd = accepted_qdd
            # The state is committed only after sink acceptance, but its
            # trajectory timestamp remains the scheduler sample time. Using
            # SDK-return time here would subtract every SDK call duration from
            # the next integration interval and systematically under-speed the
            # 150 Hz path.
            self._last_commit_monotonic = proposal.sample_monotonic_time
            self._pending = None

    def reject(self, proposal: JointTrajectoryProposal) -> bool:
        """Discard a proposal without advancing state.

        Returns ``False`` when target preemption already invalidated it.
        """

        with self._lock:
            if self._pending is None or self._pending.token != proposal.token:
                return False
            self._pending = None
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "planner": POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
                "initialized": self._q is not None,
                "q_command": None if self._q is None else self._q.tolist(),
                "qd_command_rad_s": None if self._qd is None else self._qd.tolist(),
                "qdd_command_rad_s2": None if self._qdd is None else self._qdd.tolist(),
                "target_q": None if self._target_q is None else self._target_q.tolist(),
                "braking_guard_q": (
                    None
                    if self._braking_guard_q is None
                    else self._braking_guard_q.tolist()
                ),
                "braking_guard_direction": (
                    None
                    if self._braking_guard_direction is None
                    else self._braking_guard_direction.astype(int).tolist()
                ),
                "target_qd_raw_rad_s": (
                    None if self._target_qd_raw is None else self._target_qd_raw.tolist()
                ),
                "target_sequence": self._target_sequence,
                "speed_scale": float(self._speed_scale),
                "mode": self._mode,
                "reason": self._reason,
                "quick_stop": bool(self._quick_stop),
                "velocity_ramp_sec": float(self.velocity_ramp_sec),
                "reference_period_sec": self.reference_period_sec,
                "last_observed_reference_dt_sec": self._last_observed_reference_dt_sec,
                "max_velocity_rad_s": self.limits.max_velocity_rad_s.tolist(),
                "max_acceleration_rad_s2": self.limits.max_acceleration_rad_s2.tolist(),
                "max_command_step_rad": (
                    None
                    if self.limits.max_command_step_rad is None
                    else self.limits.max_command_step_rad.tolist()
                ),
                "anchor_resync_count": int(self._anchor_resync_count),
                "pending_proposal": self._pending is not None,
            }

    def _resolve_time(self, monotonic_time: float | None) -> float:
        value = self._clock() if monotonic_time is None else monotonic_time
        return _finite_time(value, name="monotonic_time")

    def _vector(self, value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must contain exactly {self.dimension} finite joint values."
            )
        return vector.astype(float, copy=True)

    def _require_initialized(self) -> None:
        if self._q is None or self._last_commit_monotonic is None:
            raise RuntimeError("trajectory planner must be reset before use.")

    def _require_target(self) -> None:
        self._require_initialized()
        if self._target_q is None or self._target_sequence is None:
            raise RuntimeError("trajectory planner has no active target.")

    def _require_matching_pending(self, proposal: JointTrajectoryProposal) -> None:
        if not isinstance(proposal, JointTrajectoryProposal):
            raise TypeError("proposal must be JointTrajectoryProposal.")
        if self._pending is None or self._pending.token != proposal.token:
            raise RuntimeError("trajectory proposal is no longer pending.")
        if proposal.generation != self._generation:
            raise RuntimeError("trajectory proposal was superseded by a newer target.")


__all__ = [
    "AccelerationLimitedJointTrajectoryPlanner",
    "DEFAULT_POSITION_DEGRADED_SPEED_SCALE",
    "DEFAULT_POSITION_VELOCITY_RAMP_SEC",
    "JointTrajectoryLimits",
    "JointTrajectoryProposal",
    "POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED",
    "POSITION_COMMAND_PLANNER_CHOICES",
    "POSITION_COMMAND_PLANNER_LINEAR",
    "normalize_position_command_planner",
]
