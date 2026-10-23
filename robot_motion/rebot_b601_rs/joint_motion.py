"""Single-joint reference tracking and stop-point capture, without actuator I/O.

References use the existing transactional trajectory planner. Only a complete
owner dispatch receipt commits their evolution; measured motion is separate.
This is not a force observer, friction model, or hardware safety supervisor.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from robot_motion.position_control.trajectory import (
    AccelerationLimitedJointTrajectoryPlanner,
    JointTrajectoryLimits,
    JointTrajectoryProposal,
)

# Both gates request the same bounded stop. Crossing the absolute SOFT speed
# does not revoke a governed target; raw hard guards and support faults do.
_GOVERNED_SPEED_BRAKES = frozenset({
    "resolved_tracking_speed_braking", "soft_velocity_braking",
})


@dataclass(frozen=True)
class B601JointMotionConfig:
    tracking_kp_nm_per_rad: float = 6.0
    tracking_kd_nm_s_per_rad: float = 1.0
    max_reference_velocity_rad_s: float = 0.015
    max_reference_acceleration_rad_s2: float = 0.03
    max_reference_lag_rad: float = 0.013
    soft_velocity_rad_s: float = 0.03
    braking_kd_nm_s_per_rad: float = 2.0
    braking_slew_nm_s: float = 2.0
    hold_kp_nm_per_rad: float = 2.0
    hold_kd_nm_s_per_rad: float = 1.0
    hold_torque_cap_nm: float = 0.05
    braking_timeout_sec: float = 3.9
    max_control_gap_sec: float = 0.15
    velocity_window_sec: float = 0.08
    # Motor-local velocity damping is separate from software guide torque.
    # Zero preserves the original controller; it is never passive-mode tuning.
    drive_driver_kd_nm_s_per_rad: tuple[float, ...] = (0.0,) * 6
    drive_damping_rise_per_sec: float = 2.0
    minimum_owner_rate_hz: float = 20.0
    windowed_soft_velocity: bool = False
    # Offline-only pacing; the existing lead brake remains the admission gate.
    reference_pacing: bool = False
    # Explicit acquisition-only resolution envelope; None keeps legacy gates.
    position_resolution_rad: float | None = None
    responsive_tracking_brake: bool = False
    # Reserve room for unloading before the unchanged final arrival tolerance.
    arrival_margin_rad: float = 0.0
    speed_governor: bool = False
    arrival_stabilization: bool = False
    arrival_unload_slew_nm_s: float = 0.5
    # Operational headroom applies to the early brake, never the raw guard.
    velocity_headroom_ratio: float = 0.0
    # Explicit positioning only: brake for support soft speed, then require
    # the same six-axis zero/still evidence before a bounded same-goal resume.
    support_soft_speed_recovery: bool = False
    # Explicit continuous mode: regulate at the resolved early gate, stop at
    # this soft ceiling. Acquisition can retain the original absolute soft
    # stop; transfer can admit a higher ceiling below the owner's raw guard.
    continuous_speed_limit_rad_s: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in {"drive_driver_kd_nm_s_per_rad", "windowed_soft_velocity", "reference_pacing", "position_resolution_rad", "responsive_tracking_brake", "arrival_margin_rad", "speed_governor", "support_soft_speed_recovery", "arrival_stabilization", "velocity_headroom_ratio", "continuous_speed_limit_rad_s"}:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        try:
            values = tuple(self.drive_driver_kd_nm_s_per_rad)
        except TypeError as exc:
            raise ValueError("drive damping requires six non-negative gains") from exc
        if len(values) != 6 or any(
            isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 5 for v in values
        ):
            raise ValueError("drive damping requires six finite gains within the 0..5 commissioning envelope")
        object.__setattr__(self, "drive_driver_kd_nm_s_per_rad", values)
        if not isinstance(self.windowed_soft_velocity, bool):
            raise TypeError("windowed_soft_velocity must be boolean")
        if not isinstance(self.reference_pacing, bool):
            raise TypeError("reference_pacing must be boolean")
        if not isinstance(self.responsive_tracking_brake, bool):
            raise TypeError("responsive_tracking_brake must be boolean")
        if self.responsive_tracking_brake and self.position_resolution_rad is None:
            raise ValueError("responsive tracking brake requires resolved window evidence")
        if (isinstance(self.velocity_headroom_ratio, bool)
            or not math.isfinite(self.velocity_headroom_ratio)
            or self.velocity_headroom_ratio < 0):
            raise ValueError("velocity headroom must be finite and non-negative")
        if self.velocity_headroom_ratio and not self.responsive_tracking_brake:
            raise ValueError("velocity headroom requires explicit responsive tracking")
        if type(self.speed_governor) is not bool:
            raise TypeError("speed governor must be boolean")
        if type(self.support_soft_speed_recovery) is not bool:
            raise TypeError("support soft speed recovery must be boolean")
        if self.support_soft_speed_recovery and not self.speed_governor:
            raise ValueError("support soft speed recovery requires governed tracking")
        if self.speed_governor and not (self.responsive_tracking_brake and self.reference_pacing):
            raise ValueError("speed governor requires explicit resolved responsive pacing")
        if type(self.arrival_stabilization) is not bool:
            raise TypeError("arrival stabilization must be boolean")
        if self.arrival_stabilization and not self.speed_governor:
            raise ValueError("arrival stabilization requires governed tracking")
        if isinstance(self.arrival_margin_rad, bool) or not math.isfinite(self.arrival_margin_rad) or self.arrival_margin_rad < 0:
            raise ValueError("arrival margin must be finite and non-negative")
        if self.arrival_margin_rad and self.position_resolution_rad is None:
            raise ValueError("arrival margin requires resolved window evidence")
        if self.max_reference_velocity_rad_s >= self.soft_velocity_rad_s:
            raise ValueError("reference velocity must be below soft velocity")
        if self.position_resolution_rad is not None:
            value = self.position_resolution_rad
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("position resolution must be finite and positive")
            if not self.windowed_soft_velocity:
                raise ValueError("position resolution requires windowed motion control")
            if value >= self.max_reference_lag_rad:
                raise ValueError("position resolution must be below reference lag limit")
        if self.continuous_speed_limit_rad_s is not None:
            limit = self.continuous_speed_limit_rad_s
            if (isinstance(limit, bool) or not math.isfinite(limit)
                or limit < self.soft_velocity_rad_s
                or limit <= self.resolved_soft_velocity_rad_s):
                raise ValueError("continuous speed ceiling must be finite, above the resolved gate and not below the original soft stop")
            if not (self.speed_governor and self.windowed_soft_velocity):
                raise ValueError("continuous speed regulation requires explicit governed windowed tracking")

    @property
    def resolved_soft_velocity_rad_s(self) -> float:
        """One early speed gate shared by breakaway and tracking.

        Add quantization allowance before the operational margin. The absolute
        soft ceiling still caps this value; the owner retains its raw guard.
        """
        resolution = self.position_resolution_rad or 0.0
        return min(self.soft_velocity_rad_s,
            (self.max_reference_velocity_rad_s + resolution / self.velocity_window_sec)
            * (1.0 + self.velocity_headroom_ratio))


def slew_driver_damping(
    previous: tuple[float, ...],
    target: tuple[float, ...],
    *,
    rate_per_sec: float,
    dt_sec: float,
) -> tuple[float, ...]:
    """Bound a six-axis gain transition; callers own dispatch acknowledgement."""
    for values in (previous, target):
        if len(values) != 6 or any(
            isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 5
            for v in values
        ):
            raise ValueError("driver damping requires six finite gains within 0..5")
    if (
        isinstance(rate_per_sec, bool) or not math.isfinite(rate_per_sec) or rate_per_sec <= 0
        or isinstance(dt_sec, bool) or not math.isfinite(dt_sec) or dt_sec < 0
    ):
        raise ValueError("damping slew requires a positive rate and non-negative finite dt")
    delta = rate_per_sec * dt_sec
    # Exact plateaus matter to the dispatch-based phase-transition gate.
    return tuple(
        goal if abs(goal - old) <= delta else old + math.copysign(delta, goal - old)
        for old, goal in zip(previous, target)
    )


class B601WindowVelocityEstimator:
    """Causal least-squares velocity; never replaces the raw hard-guard input."""

    def __init__(self, window_sec: float) -> None:
        if not math.isfinite(window_sec) or window_sec <= 0.0:
            raise ValueError("window_sec must be finite and positive")
        self.window_sec = window_sec
        self._samples: deque[tuple[float, float]] = deque(maxlen=256)
        self.span_sec = 0.0

    def reset(self) -> None:
        self._samples.clear()
        self.span_sec = 0.0

    @property
    def ready(self) -> bool:
        return len(self._samples) >= 3 and self.span_sec >= self.window_sec - 1e-9

    @property
    def latest_observation(self) -> tuple[float, float] | None:
        return self._samples[-1] if self._samples else None

    @property
    def position_span_rad(self) -> float:
        values = [q for _, q in self._samples]
        return max(values) - min(values) if values else 0.0

    def update(
        self, *, timestamp_sec: float, qpos_rad: float, fallback_rad_s: float
    ) -> float:
        if not all(math.isfinite(x) for x in (timestamp_sec, qpos_rad, fallback_rad_s)):
            raise ValueError("velocity input must be finite")
        if self._samples and timestamp_sec <= self._samples[-1][0]:
            raise ValueError("velocity feedback timestamps must advance")
        if self._samples and timestamp_sec - self._samples[-1][0] > self.window_sec:
            self.reset()
        self._samples.append((timestamp_sec, qpos_rad))
        while (
            len(self._samples) > 2
            and self._samples[1][0] < timestamp_sec - self.window_sec
        ):
            self._samples.popleft()
        origin = self._samples[0][0]
        self.span_sec = timestamp_sec - origin
        if len(self._samples) < 2:
            return fallback_rad_s
        # Subtract local time AND position origins. Constant non-binary q
        # otherwise leaves signed round-off after mean subtraction, which can
        # spuriously reset a direction-based onset confirmation at exact rest.
        # This is the same least-squares slope, not a velocity deadband.
        times = [t - origin for t, _ in self._samples]
        positions = [q - self._samples[0][1] for _, q in self._samples]
        mean_t = sum(times) / len(times)
        mean_q = sum(positions) / len(positions)
        denominator = sum((t - mean_t) ** 2 for t in times)
        return (
            sum((t - mean_t) * (q - mean_q) for t, q in zip(times, positions))
            / denominator
        )


def window_stationarity_gate_reason(
    estimators: tuple[B601WindowVelocityEstimator, ...],
    raw_qvel_rad_s: tuple[float, ...],
    window_qvel_rad_s: tuple[float, ...],
    *,
    soft_limit_rad_s: float,
    hard_limit_rad_s: float,
    joint_index: int | None = None,
) -> str:
    """Shared causal preparation gate; raw hard protection is not filtered out."""
    if len(estimators) != 6 or len(raw_qvel_rad_s) != 6 or len(window_qvel_rad_s) != 6:
        raise ValueError("stationarity gate requires six feedback channels")
    if any(isinstance(v, bool) or not math.isfinite(v) for v in (*raw_qvel_rad_s, *window_qvel_rad_s)):
        raise ValueError("stationarity velocities must be finite")
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (soft_limit_rad_s, hard_limit_rad_s)):
        raise ValueError("stationarity limits must be finite and positive")
    if joint_index is not None and (isinstance(joint_index, bool) or not isinstance(joint_index, int) or not 0 <= joint_index < 6):
        raise ValueError("stationarity joint index must be within six channels")
    indices = range(6) if joint_index is None else (joint_index,)
    if not all(e.ready for e in estimators):
        return "velocity_window_not_ready"
    if max(abs(raw_qvel_rad_s[i]) for i in indices) > hard_limit_rad_s:
        return "raw_velocity_hard_guard"
    if max(abs(window_qvel_rad_s[i]) for i in indices) > soft_limit_rad_s:
        return "not_stationary"
    if any(estimators[i].position_span_rad > soft_limit_rad_s * estimators[i].window_sec for i in indices):
        return "velocity_window_displacement_exceeded"
    return "ready"


@dataclass(frozen=True)
class B601WindowStopEvidence:
    """Same-tick six-axis causal evidence, not proof of physical standstill."""

    timestamp_sec: float
    qpos_rad: tuple[float, ...]
    raw_qvel_rad_s: tuple[float, ...]
    window_qvel_rad_s: tuple[float, ...]
    position_span_rad: tuple[float, ...]
    window_span_sec: float
    raw_velocity_limit_rad_s: float
    joint_index: int = 2

    def __post_init__(self) -> None:
        if type(self.joint_index) is not int or not 0 <= self.joint_index < 6:
            raise ValueError("stop evidence joint index must be within six channels")
        for name in ("qpos_rad", "raw_qvel_rad_s", "window_qvel_rad_s", "position_span_rad"):
            values = tuple(getattr(self, name))
            if len(values) != 6 or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError("stop evidence requires six finite feedback channels")
            if name == "position_span_rad" and any(v < 0 for v in values):
                raise ValueError("stop evidence position spans must be non-negative")
            object.__setattr__(self, name, values)
        for name in ("timestamp_sec", "window_span_sec", "raw_velocity_limit_rad_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("stop evidence clocks/guard must be finite and non-negative")
        if self.raw_velocity_limit_rad_s <= 0:
            raise ValueError("stop evidence raw guard must be positive")

    def gate_reason(self, *, window_sec: float, velocity_limit: float, resolution: float) -> str | None:
        if self.window_span_sec < window_sec - 1e-9:
            return "velocity_window_not_ready"
        if max(abs(v) for v in self.raw_qvel_rad_s) > self.raw_velocity_limit_rad_s:
            return "raw_velocity_hard_guard"
        for span, velocity in zip(self.position_span_rad, self.window_qvel_rad_s):
            # Sub-resolution toggling can have a large fitted slope. Outside
            # that small envelope BOTH the old speed and range gates apply.
            if span <= resolution:
                continue
            if abs(velocity) > velocity_limit:
                return "six_axis_window_velocity"
            if span > velocity_limit * window_sec:
                return "six_axis_window_displacement"
        return None


@dataclass(frozen=True)
class B601JointMotionFrame:
    phase: str
    reason: str
    q_reference_rad: float
    qvel_reference_rad_s: float
    control_qvel_rad_s: float
    position_torque_nm: float
    velocity_torque_nm: float
    requested_torque_nm: float
    hold_position_rad: float | None = None
    brake_elapsed_sec: float = 0.0
    target_position_rad: float | None = None
    target_error_rad: float | None = None
    brake_trigger_reference_rad: float | None = None
    brake_trigger_lag_rad: float | None = None
    stop_capture_block_reason: str | None = None
    stop_capture_dwell_elapsed_sec: float = 0.0
    stop_capture_required_dwell_sec: float | None = None
    stop_capture_velocity_limit_rad_s: float | None = None
    reference_speed_scale: float | None = None
    reference_headroom_rad: float | None = None
    stop_position_resolution_rad: float | None = None
    tracking_speed_brake_limit_rad_s: float | None = None
    torque_transition_reason: str | None = None
    torque_transition_target_nm: float | None = None
    torque_transition_rate_nm_s: float | None = None
    arrival_brake_error_rad: float | None = None
    reference_safety_headroom_rad: float | None = None
    reference_position_reserve_rad: float | None = None
    motion_evidence_block_reason: str | None = None
    speed_governor_active: bool = False
    speed_governor_resume_count: int = 0
    speed_governor_resume_block_reason: str | None = None
    tracking_recent_velocity_rad_s: float | None = None
    tracking_recent_position_span_rad: float | None = None
    arrival_stabilization_active: bool = False
    arrival_unloading: bool = False
    arrival_unload_exit_error_rad: float | None = None
    continuous_speed_scale: float | None = None
    continuous_speed_observed_rad_s: float | None = None
    continuous_speed_start_rad_s: float | None = None
    event_severity: str = "info"
    event_action: str = "continue"
    brake_started_at_sec: float | None = None


class B601JointMotionTracker:
    """Owner-tick local tracking, damping-only brake, then measured-point hold."""

    def _governed_speed_brake(self, reason: str) -> bool:
        return reason in _GOVERNED_SPEED_BRAKES or (
            self.config.support_soft_speed_recovery and reason == "support_velocity_braking"
        ) or (
            self.config.continuous_speed_limit_rad_s is not None
            and reason == "reference_pacing_headroom_braking"
        )

    def __init__(self, config: B601JointMotionConfig) -> None:
        self.config = config
        self._planner = AccelerationLimitedJointTrajectoryPlanner(
            JointTrajectoryLimits(
                max_velocity_rad_s=(config.max_reference_velocity_rad_s,),
                max_acceleration_rad_s2=(config.max_reference_acceleration_rad_s2,),
            ),
            velocity_ramp_sec=config.max_reference_velocity_rad_s
            / config.max_reference_acceleration_rad_s2,
        )
        self.reset()

    def reset(self) -> None:
        self._pending: JointTrajectoryProposal | None = None
        self._pending_sequence: int | None = None
        self._started = False
        self._brake_start: float | None = None
        self._stationary_start: float | None = None
        self._stationary_position: float | None = None
        self._stationary_min: tuple[float, ...] | None = None
        self._stationary_max: tuple[float, ...] | None = None
        self._hold_position: float | None = None
        self._target = 0.0
        self._last_tick: float | None = None
        self._brake_trigger_reference: float | None = None
        self._brake_trigger_lag: float | None = None
        self._speed_governor_resume_count = 0
        self._arrival_unloading = False
        self._calibration_probe_active = False
        # The shorter causal window only advances soft withdrawal in explicit
        # governed tracking. It never authorizes start, stop, or hard admission.
        self._recent_velocity = B601WindowVelocityEstimator(self.config.velocity_window_sec / 2)
        self._recent_speed = 0.0

    def observe_feedback(self, *, now: float, q: float, raw_v: float) -> None:
        """Warm the existing short observer; never advance a reference here."""
        if not all(math.isfinite(x) for x in (now, q, raw_v)) or now < 0:
            raise ValueError("motion feedback must have a finite non-negative clock")
        latest = self._recent_velocity.latest_observation
        if latest == (now, q):
            return  # One physical observation, including start/update in this tick.
        self._recent_speed = self._recent_velocity.update(
            timestamp_sec=now, qpos_rad=q, fallback_rad_s=raw_v)

    def start(self, *, now: float, q: float, target: float,
              preserve_velocity_history: bool = False) -> None:
        if not all(math.isfinite(x) for x in (now, q, target)) or now < 0:
            raise ValueError("motion start must be finite with a non-negative clock")
        if type(preserve_velocity_history) is not bool:
            raise TypeError("preserve_velocity_history must be boolean")
        recent, speed = self._recent_velocity, self._recent_speed
        if preserve_velocity_history and (
            self.config.continuous_speed_limit_rad_s is None
            or recent.latest_observation != (now, q)
        ):
            raise ValueError("transfer history requires same-tick feedback in explicit continuous mode")
        self.reset()
        if preserve_velocity_history:
            self._recent_velocity, self._recent_speed = recent, speed
        self._planner.reset((q,), monotonic_time=now)
        self._planner.accept_target(
            (target,), sequence=0, published_monotonic_time=now, monotonic_time=now
        )
        self._target = target
        self._started = True
        self._last_tick = now
        self.observe_feedback(now=now, q=q, raw_v=0.0)

    def record_dispatch(self, control_sequence: int) -> None:
        if self._pending is not None and control_sequence == self._pending_sequence:
            # This acknowledges an internal reference used by the dispatched
            # torque calculation, not realization of that position by the plant.
            self._planner.commit(
                self._pending,
                accepted_monotonic_time=self._pending.sample_monotonic_time,
            )
            self._pending = None
            self._pending_sequence = None

    def _observe_stationary_range(self, qpos: tuple[float, ...]) -> float:
        if self._stationary_min is None:
            self._stationary_min = self._stationary_max = qpos
        else:
            self._stationary_min = tuple(min(a, b) for a, b in zip(self._stationary_min, qpos))
            self._stationary_max = tuple(max(a, b) for a, b in zip(self._stationary_max, qpos))
        return max(hi - lo for lo, hi in zip(self._stationary_min, self._stationary_max))

    def update(
        self,
        *,
        now: float,
        q: float,
        raw_v: float,
        control_v: float,
        control_sequence: int,
        authority: float,
        kp: float,
        kd: float,
        stop_error_rad: float,
        stop_velocity_rad_s: float,
        stop_dwell_sec: float,
        request_brake: bool = False,
        brake_reason: str = "operator_stop_and_hold",
        support_max_velocity_rad_s: float = 0.0,
        support_raw_max_velocity_rad_s: float | None = None,
        stop_evidence: B601WindowStopEvidence | None = None,
        speed_governor_permitted: bool = False,
        zero_dispatch_confirmed: bool = False,
        calibration_hold_enabled: bool = False,
        calibration_probe_nm: float | None = None,
        calibration_target_hold: bool = False,
        calibration_support_dispatch_confirmed: bool | None = None,
    ) -> B601JointMotionFrame:
        if calibration_support_dispatch_confirmed is not None and (
            type(calibration_support_dispatch_confirmed) is not bool or not calibration_target_hold
        ):
            raise ValueError("support receipt requires explicit calibration target holding")
        supported_stop = calibration_support_dispatch_confirmed is not None
        # The coordinated owner may confirm a bounded support frame instead
        # of a zero frame during an explicitly authorized SOFT stop. This is
        # not a motor acknowledgement or permission to recover terminal stops.
        stop_dispatch_confirmed = (calibration_support_dispatch_confirmed if supported_stop
                                   else zero_dispatch_confirmed)
        if type(calibration_target_hold) is not bool:
            raise TypeError("calibration target hold must be boolean")
        if calibration_target_hold and not (calibration_hold_enabled and self.config.speed_governor):
            raise ValueError("calibration target hold requires explicit governed calibration")
        if type(calibration_hold_enabled) is not bool:
            raise TypeError("calibration hold permit must be boolean")
        if calibration_hold_enabled and not (
            self.config.responsive_tracking_brake
            and self.config.continuous_speed_limit_rad_s is not None
        ):
            raise ValueError("holding calibration requires explicit resolved continuous motion")
        if calibration_probe_nm is not None and (
            not calibration_hold_enabled or isinstance(calibration_probe_nm, bool)
            or not math.isfinite(calibration_probe_nm)
        ):
            raise ValueError("fixed holding probe requires explicit finite calibration demand")
        if type(speed_governor_permitted) is not bool or type(zero_dispatch_confirmed) is not bool:
            raise TypeError("governor permit and zero receipt must be boolean")
        if calibration_target_hold and not speed_governor_permitted:
            request_brake, brake_reason = True, "speed_governor_permission_withdrawn"
        support_raw = support_max_velocity_rad_s if support_raw_max_velocity_rad_s is None else support_raw_max_velocity_rad_s
        if not all(
            math.isfinite(x)
            for x in (
                now,
                q,
                raw_v,
                control_v,
                authority,
                kp,
                kd,
                stop_error_rad,
                stop_velocity_rad_s,
                stop_dwell_sec,
                support_max_velocity_rad_s,
                support_raw,
            )
        ):
            raise ValueError("motion input must be finite")
        if not 0 <= authority <= 1 or min(kp, kd) < 0:
            raise ValueError("invalid motion authority or gains")
        if min(support_max_velocity_rad_s, support_raw) < 0:
            raise ValueError("support velocity magnitude must be non-negative")
        if not self._started:
            raise RuntimeError("motion tracker requires a measured start")
        if min(stop_error_rad, stop_velocity_rad_s, stop_dwell_sec) <= 0:
            raise ValueError("motion stop thresholds must be positive")
        arrival_error = stop_error_rad - self.config.arrival_margin_rad
        if arrival_error <= 0:
            raise ValueError("arrival margin must be below final stop tolerance")
        if not isinstance(brake_reason, str) or not brake_reason.strip():
            raise ValueError("brake reason must be a non-empty string")
        if (
            isinstance(control_sequence, bool)
            or not isinstance(control_sequence, int)
            or control_sequence < 0
        ):
            raise ValueError("motion control sequence must be a non-negative integer")
        # The first reference may be computed in the same tick as start.
        if now < self._last_tick:
            raise ValueError("motion control clock must not move backwards")
        tick_advanced = now > self._last_tick
        self._last_tick = now
        if self.config.speed_governor and tick_advanced:
            self.observe_feedback(now=now, q=q, raw_v=raw_v)
        # Window velocity is opt-in for quantized feedback. Raw speed remains
        # an independent owner hard guard and is still required for stop dwell.
        soft_speed = abs(control_v) if self.config.windowed_soft_velocity else max(abs(raw_v), abs(control_v))

        stop_capture_block_reason = None
        resolution = self.config.position_resolution_rad
        evidence_reason = None
        if resolution is not None:
            if stop_evidence is None or stop_evidence.timestamp_sec != now:
                evidence_reason = "same_tick_stop_evidence_unavailable"
            else:
                index = stop_evidence.joint_index
                if (stop_evidence.qpos_rad[index], stop_evidence.raw_qvel_rad_s[index], stop_evidence.window_qvel_rad_s[index]) != (q, raw_v, control_v):
                    raise ValueError("stop evidence must match selected feedback")
                evidence_reason = stop_evidence.gate_reason(
                    window_sec=self.config.velocity_window_sec,
                    velocity_limit=stop_velocity_rad_s, resolution=resolution,
                )
                # Gain restoration can take longer than a single dwell. Keep
                # its six-axis envelope anchored to the captured stop instead
                # of accepting a staircase of quiet short windows.
                if (
                    evidence_reason is None and self._hold_position is not None
                    and self._observe_stationary_range(stop_evidence.qpos_rad)
                    > max(resolution, stop_velocity_rad_s * stop_dwell_sec)
                ):
                    evidence_reason = "measured_position_changed"
        reference_speed_scale = None
        reference_headroom_rad = None
        reference_safety_headroom_rad = None
        reference_position_reserve_rad = None
        tracking_speed_limit = None
        speed_governor_resume_block_reason = None
        if self.config.responsive_tracking_brake:
            # Reserve one encoder-resolution interval above the reference
            # speed. This is an earlier software brake, not a raw hard limit.
            tracking_speed_limit = self.config.resolved_soft_velocity_rad_s
        continuous_limit = self.config.continuous_speed_limit_rad_s
        selected_stop_limit = continuous_limit or self.config.soft_velocity_rad_s
        continuous_scale = None
        continuous_speed = None
        if continuous_limit is not None:
            # Unresolved one-count jitter must not drain maintenance demand.
            # Only travel TOWARD the goal earns a drive reduction. Opposing
            # travel needs the existing corrective demand, not less of it.
            # Absolute speed remains independently bounded by soft/hard stops.
            direction = math.copysign(1.0, self._target - q)
            continuous_speed = max(0.0, direction * control_v) if (
                stop_evidence is not None
                and stop_evidence.position_span_rad[stop_evidence.joint_index] > resolution
            ) else 0.0
            if self._recent_velocity.ready and self._recent_velocity.position_span_rad > resolution:
                continuous_speed = max(continuous_speed, direction * self._recent_speed)
            continuous_scale = max(0.0, min(1.0,
                (continuous_limit - continuous_speed) / (continuous_limit - tracking_speed_limit)))

        def frame(*args, **kwargs) -> B601JointMotionFrame:
            # Preserve the reference that caused braking. The braking reference
            # itself is measured position and must not erase that evidence.
            stabilizing = (
                args[0] == "brake" and self.config.arrival_stabilization
                and self._brake_start is not None
                and self._brake_reason == "target_reached_braking"
                and self._hold_position is None
            )
            governing = (
                self.config.speed_governor and self._brake_start is not None
                and self._hold_position is None
                and self._governed_speed_brake(self._brake_reason)
            )
            return B601JointMotionFrame(
                *args,
                **kwargs,
                target_position_rad=self._target,
                target_error_rad=self._target - q,
                brake_trigger_reference_rad=self._brake_trigger_reference,
                brake_trigger_lag_rad=self._brake_trigger_lag,
                stop_capture_block_reason=stop_capture_block_reason,
                stop_capture_dwell_elapsed_sec=(
                    0.0 if self._stationary_start is None else now - self._stationary_start
                ),
                stop_capture_required_dwell_sec=stop_dwell_sec,
                stop_capture_velocity_limit_rad_s=stop_velocity_rad_s,
                reference_speed_scale=reference_speed_scale,
                reference_headroom_rad=reference_headroom_rad,
                reference_safety_headroom_rad=reference_safety_headroom_rad,
                reference_position_reserve_rad=reference_position_reserve_rad,
                speed_governor_active=governing,
                speed_governor_resume_count=self._speed_governor_resume_count,
                speed_governor_resume_block_reason=speed_governor_resume_block_reason,
                stop_position_resolution_rad=resolution,
                tracking_speed_brake_limit_rad_s=continuous_limit or tracking_speed_limit,
                continuous_speed_scale=continuous_scale if args[0] == "track" else None,
                continuous_speed_observed_rad_s=continuous_speed,
                continuous_speed_start_rad_s=tracking_speed_limit if continuous_limit else None,
                arrival_brake_error_rad=arrival_error,
                tracking_recent_velocity_rad_s=(
                    self._recent_speed if self.config.speed_governor and self._recent_velocity.ready else None),
                tracking_recent_position_span_rad=(
                    self._recent_velocity.position_span_rad if self.config.speed_governor else None),
                arrival_stabilization_active=stabilizing,
                arrival_unloading=stabilizing and self._arrival_unloading,
                arrival_unload_exit_error_rad=stop_error_rad if stabilizing else None,
                event_severity=("error" if args[0] == "authority_exhausted" else "warning" if governing else "info"),
                event_action=("withdraw" if args[0] == "authority_exhausted" else "brake_zero_still_then_resume" if governing else "continue"),
                brake_started_at_sec=self._brake_start,
            )

        if self._pending is not None:
            return frame(
                "authority_exhausted",
                "reference_dispatch_not_confirmed",
                q,
                0.0,
                control_v,
                0.0,
                0.0,
                0.0,
            )
        if self.config.responsive_tracking_brake:
            block = None
            if stop_evidence is None:
                block = "feedback_evidence_missing"
            elif stop_evidence.timestamp_sec != now:
                block = "feedback_evidence_timestamp_mismatch"
            elif stop_evidence.window_span_sec < self.config.velocity_window_sec - 1e-9:
                block = "velocity_window_not_ready"
            elif max(abs(v) for v in stop_evidence.raw_qvel_rad_s) > stop_evidence.raw_velocity_limit_rad_s:
                block = "raw_velocity_hard_guard"
            if block is not None:
                if self.config.speed_governor:
                    self._brake_start = now if self._brake_start is None else self._brake_start
                    self._brake_reason = "tracking_motion_evidence_unavailable_or_unsafe"
                return frame("authority_exhausted", "tracking_motion_evidence_unavailable_or_unsafe",
                             q, 0.0, control_v, 0.0, 0.0, 0.0,
                             motion_evidence_block_reason=block)
        if calibration_probe_nm is not None:
            # This is a fixed-command observation, not position regulation.
            # Raw/evidence admission above and every existing soft brake below
            # still take precedence, including motion away from the goal.
            probe_safe = (
                not request_brake
                and support_max_velocity_rad_s < self.config.soft_velocity_rad_s
                and soft_speed < selected_stop_limit
                and (not self._recent_velocity.ready
                     or self._recent_velocity.position_span_rad <= (resolution or 0.0)
                     or abs(self._recent_speed) < (continuous_limit or tracking_speed_limit or selected_stop_limit))
                and (self._brake_start is None or self._brake_reason == "target_reached_braking")
            )
            if probe_safe:
                self._calibration_probe_active = True
                self._brake_start = self._stationary_start = None
                self._arrival_unloading = False
                return frame("track", "calibration_fixed_command_observation",
                             q, 0.0, control_v, 0.0, 0.0, authority * calibration_probe_nm)
        elif self._calibration_probe_active:
            # Same frozen target, fresh measured reference. Do not reset the
            # governor count or caller's point/command-exposure budgets.
            if not request_brake and self._brake_start is None:
                self._planner.reset((q,), monotonic_time=now)
                self._planner.accept_target((self._target,), sequence=0,
                    published_monotonic_time=now, monotonic_time=now)
            self._calibration_probe_active = False
        active_target_hold = (calibration_target_hold and self._hold_position == self._target
                              and self._brake_reason == "calibration_target_tracking")
        if (
            self._hold_position is not None
            and not request_brake
            and soft_speed < selected_stop_limit
            and support_max_velocity_rad_s < self.config.soft_velocity_rad_s
            and (abs(self._target-q) <= self.config.max_reference_lag_rad if active_target_hold
                 else resolution is None or evidence_reason is None)
        ):
            p = self.config.hold_kp_nm_per_rad * (self._hold_position - q)
            damping_v = control_v if self.config.windowed_soft_velocity else raw_v if raw_v * control_v < 0 else control_v
            d = -self.config.hold_kd_nm_s_per_rad * damping_v
            cap = self.config.hold_torque_cap_nm
            return frame(
                "track" if active_target_hold else "soft_hold",
                "calibration_target_tracking" if active_target_hold else "holding_measured_stop_point",
                self._hold_position,
                0.0,
                control_v,
                p,
                d,
                authority * max(-cap, min(cap, p + d)),
                self._hold_position,
            )
        if self._hold_position is not None:
            hold_reference = self._hold_position
            self._hold_position = None
            self._brake_start = now
            self._stationary_start = None
            self._stationary_position = None
            self._brake_reason = (
                brake_reason if request_brake else
                "reference_lag_braking" if active_target_hold and abs(self._target-q) > self.config.max_reference_lag_rad else
                "support_velocity_braking" if active_target_hold and support_max_velocity_rad_s >= self.config.soft_velocity_rad_s else
                "soft_velocity_braking" if active_target_hold and soft_speed >= selected_stop_limit else
                "soft_hold_motion_braking") if active_target_hold else "soft_hold_motion_braking"
            self._brake_trigger_reference = hold_reference if active_target_hold else q
            self._brake_trigger_lag = hold_reference-q if active_target_hold else 0.0
        if (self.config.speed_governor and self._brake_start is not None
            and (self._governed_speed_brake(self._brake_reason)
                 or self.config.arrival_stabilization
                 and self._brake_reason == "target_reached_braking")):
            was_calibration_arrival = calibration_hold_enabled and self._brake_reason == "target_reached_braking"
            # Once withdrawn, a permit or stop cannot be undone by a later
            # quiet frame. The caller must start a new authorized action.
            if request_brake:
                self._brake_reason = brake_reason
            elif not speed_governor_permitted:
                self._brake_reason = "speed_governor_permission_withdrawn"
            elif support_max_velocity_rad_s >= self.config.soft_velocity_rad_s:
                self._brake_reason = "support_velocity_braking"
            elif soft_speed >= selected_stop_limit:
                self._brake_reason = "soft_velocity_braking"
            elif self._brake_reason == "target_reached_braking":
                if abs(self._target - q) > self.config.max_reference_lag_rad:
                    self._brake_reason = "reference_lag_braking"
                    self._brake_trigger_reference = self._target
                    self._brake_trigger_lag = self._target-q
                elif (
                    abs(control_v) >= (continuous_limit or tracking_speed_limit)
                    and stop_evidence.position_span_rad[stop_evidence.joint_index] > resolution
                    or self._recent_velocity.ready
                    and abs(self._recent_speed) >= (continuous_limit or tracking_speed_limit)
                    and self._recent_velocity.position_span_rad > resolution
                ):
                    self._brake_reason = "resolved_tracking_speed_braking"
            if (was_calibration_arrival and not request_brake and speed_governor_permitted
                and self._brake_reason in {"support_velocity_braking", "soft_velocity_braking",
                                           "resolved_tracking_speed_braking"}):
                # Normal calibration arrival may hold a nonzero command for
                # arbitrarily longer than the brake timeout. A newly triggered
                # speed brake owns a new clock, never that arrival dwell.
                self._brake_start = now
                self._stationary_start = self._stationary_position = None
                self._stationary_min = self._stationary_max = None
                self._brake_trigger_reference = self._target
                self._brake_trigger_lag = self._target - q
            if self._brake_reason != "target_reached_braking":
                self._arrival_unloading = False
        if self._brake_start is None:
            pacing_exhausted = False
            if (
                self.config.reference_pacing
                and tick_advanced
                and not request_brake
                and soft_speed < selected_stop_limit
                and support_max_velocity_rad_s < self.config.soft_velocity_rad_s
                and abs(self._target - q) > stop_error_rad
            ):
                snapshot = self._planner.snapshot()
                committed_q = snapshot["q_command"][0]
                direction = math.copysign(1.0, self._target - committed_q)
                # Reserve the unchanged watchdog interval for observed reverse
                # motion and a deceleration distance for the reference itself.
                # This is not a plant/friction estimate or a force guarantee.
                gap = self.config.max_control_gap_sec
                adverse_velocity = max(0.0, -direction * control_v)
                if (resolution is not None and stop_evidence is not None
                    and stop_evidence.timestamp_sec == now
                    and stop_evidence.window_span_sec >= self.config.velocity_window_sec - 1e-9
                    and stop_evidence.position_span_rad[stop_evidence.joint_index] <= resolution):
                    # One unresolved count is covered by the position reserve
                    # below. Extrapolating its fitted slope as continued reverse
                    # travel would charge the same ambiguity a second time.
                    adverse_velocity = 0.0
                # During explicit transfer the measured joint can run ahead
                # after breakaway. Advancing the reference then REDUCES error;
                # slowing it as if it were ahead compounds the divergence.
                # Do not count that measured lead as extra authority. The
                # independent absolute reference-error brake below is intact.
                measured_ahead = (continuous_limit is not None
                    and direction * (committed_q-q) < 0.0)
                committed_lead = 0.0 if measured_ahead else abs(committed_q-q)
                reference_safety_headroom_rad = (
                    self.config.max_reference_lag_rad - committed_lead
                    - adverse_velocity * gap
                )
                # A causal low-speed window can be zero immediately before
                # one encoder count changes. Reserve that declared uncertainty
                # BEFORE the unchanged hard lead boundary, not after a trip.
                reference_position_reserve_rad = resolution or 0.0
                reference_headroom_rad = reference_safety_headroom_rad - reference_position_reserve_rad
                # A paused reference can consume the reserved encoder count
                # during finite-rate deceleration. Stop before the next count
                # violates the unchanged absolute reference-lag boundary.
                pacing_exhausted = (reference_headroom_rad <= 0.0 if calibration_target_hold
                                    else reference_safety_headroom_rad <= 0.0)
                if not pacing_exhausted:
                    acceleration = self.config.max_reference_acceleration_rad_s2
                    # Positive root of v*gap + v^2/(2*a) = headroom, in a
                    # cancellation-resistant form. Never reset/clip committed q
                    # or velocity; the canonical planner enforces acceleration.
                    planning_headroom = max(0.0, reference_headroom_rad)
                    safe_speed = (
                        2.0 * acceleration * planning_headroom
                        / (math.sqrt((acceleration * gap) ** 2
                                     + 2.0 * acceleration * planning_headroom)
                           + acceleration * gap)
                    )
                    reference_speed_scale = min(
                        1.0, safe_speed / self.config.max_reference_velocity_rad_s
                    )
                    if continuous_scale is not None and not measured_ahead:
                        reference_speed_scale = min(reference_speed_scale, continuous_scale)
                    if reference_speed_scale != snapshot["speed_scale"]:
                        # Renew only a changed budget. Re-publishing an
                        # unchanged endpoint every tick would also alter the
                        # canonical endpoint approach even with scale == 1.
                        if reference_speed_scale == 0.0:
                            self._planner.pause_reference(monotonic_time=now,
                                reason="quantized_lead_reserve_pause")
                        else:
                            self._planner.accept_target(
                                (self._target,), sequence=snapshot["target_sequence"] + 1,
                                published_monotonic_time=now, monotonic_time=now,
                                speed_scale=reference_speed_scale,
                                reason="measured_lead_pacing",
                            )
            proposal = self._planner.propose(monotonic_time=now)
            qr, vr = float(proposal.q_command[0]), float(proposal.qd_command_rad_s[0])
            reason = (
                brake_reason
                if request_brake
                else "support_velocity_braking"
                if support_max_velocity_rad_s >= self.config.soft_velocity_rad_s
                else "soft_velocity_braking"
                if soft_speed >= selected_stop_limit
                else "resolved_tracking_speed_braking"
                if self.config.responsive_tracking_brake
                and (
                    abs(control_v) >= (continuous_limit or tracking_speed_limit)
                    and stop_evidence.position_span_rad[stop_evidence.joint_index] > resolution
                    or self.config.speed_governor and self._recent_velocity.ready
                    and abs(self._recent_speed) >= (continuous_limit or tracking_speed_limit)
                    and self._recent_velocity.position_span_rad > resolution
                )
                else "reference_lag_braking"
                if abs(qr - q) > self.config.max_reference_lag_rad
                else "reference_pacing_headroom_braking"
                if pacing_exhausted
                else "target_reached_braking"
                if abs(self._target - q) <= arrival_error
                else None
            )
            if reason is None:
                self._pending = proposal
                self._pending_sequence = control_sequence
                p, d = kp * (qr - q), kd * (vr - control_v)
                return frame(
                    "track",
                    "bounded_reference_tracking",
                    qr,
                    vr,
                    control_v,
                    p,
                    d,
                    authority * (p + d),
                )
            self._planner.reject(proposal)
            self._brake_trigger_reference = qr
            self._brake_trigger_lag = qr - q
            self._brake_start = now
            self._brake_reason = reason
            if (self.config.speed_governor and self._governed_speed_brake(reason)
                and abs(qr-q) > self.config.max_reference_lag_rad):
                self._brake_reason = "reference_lag_braking"
            self._hold_position = None
        elapsed = now - self._brake_start
        if elapsed >= self.config.braking_timeout_sec and not (
            calibration_hold_enabled and not request_brake
            and self._brake_reason == "target_reached_braking"
        ):
            stop_capture_block_reason = "braking_timeout"
            return frame(
                "authority_exhausted",
                "braking_timeout_assist_withdrawn",
                q,
                0.0,
                control_v,
                0.0,
                0.0,
                0.0,
                brake_elapsed_sec=elapsed,
            )
        if (calibration_hold_enabled and not request_brake
            and self._brake_reason == "target_reached_braking"):
            # Calibration must observe the necessary nonzero command BEFORE
            # normal arrival unloads it. Genuine brakes retain their old exit.
            self._arrival_unloading = False
            return frame("brake", "target_reached_braking", self._target,
                         0.0, control_v, kp * (self._target-q), -kd * control_v,
                         authority * (kp * (self._target-q) - kd * control_v),
                         brake_elapsed_sec=elapsed)
        stationary = (
            evidence_reason is None if resolution is not None else
            max(abs(raw_v), abs(control_v), support_max_velocity_rad_s, support_raw) <= stop_velocity_rad_s
        )
        arrival_active = (
            self.config.arrival_stabilization
            and self._brake_reason == "target_reached_braking"
        )
        # The reserved inner arrival_error starts braking early. Once in the
        # target-arrival brake, use the same configured final-position gate
        # instead of reusing the tighter inner braking trigger.
        arrival_position_limit = stop_error_rad
        if arrival_active and abs(self._target - q) > arrival_position_limit:
            # Quiet outside the configured target band is not arrival. Correct
            # within the SAME braking/leg deadline, without recapturing a
            # displaced endpoint. The separate stationary dwell and final
            # zero-dispatch confirmation still gate arrival capture.
            stationary = False
            evidence_reason = "arrival_position_outside_tolerance"
        if not stationary:
            self._arrival_unloading = False
        elif arrival_active and self._arrival_unloading and not zero_dispatch_confirmed:
            # Final arrival needs a fresh quiet dwell AFTER the final zero,
            # not quiet samples collected while maintenance was still applied.
            self._stationary_start = now
        if stationary:
            if self._stationary_start is None:
                self._stationary_start = now
                self._stationary_position = q
                if resolution is not None:
                    self._stationary_min = self._stationary_max = stop_evidence.qpos_rad
            if resolution is not None:
                changed = self._observe_stationary_range(stop_evidence.qpos_rad) > max(resolution, stop_velocity_rad_s * stop_dwell_sec)
            else:
                changed = self.config.windowed_soft_velocity and abs(q - self._stationary_position) > stop_velocity_rad_s * stop_dwell_sec
            if changed:
                self._arrival_unloading = False
                self._stationary_start = now
                self._stationary_position = q
                if resolution is not None:
                    self._stationary_min = self._stationary_max = stop_evidence.qpos_rad
                stop_capture_block_reason = "measured_position_changed"
            if now - self._stationary_start >= stop_dwell_sec:
                if arrival_active and (not self._arrival_unloading or not zero_dispatch_confirmed):
                    # The staged owner drains only existing maintenance demand.
                    # A zero request is not a complete six-axis zero receipt.
                    self._arrival_unloading = True
                    self._stationary_start = now
                    stop_capture_block_reason = "arrival_unloading_waiting_for_zero_dispatch"
                    return frame("brake", "target_reached_braking", self._target,
                                 0.0, control_v, 0.0, 0.0, 0.0,
                                 brake_elapsed_sec=elapsed)
                if (self.config.speed_governor
                    and self._governed_speed_brake(self._brake_reason)
                    and not request_brake and speed_governor_permitted
                    and abs(self._target - q) > stop_error_rad):
                    if not stop_dispatch_confirmed:
                        # Confirm the owner's selected stop baseline; never
                        # call a nonzero support frame a zero receipt.
                        speed_governor_resume_block_reason = ("support_dispatch_not_confirmed" if supported_stop
                                                             else "native_zero_dispatch_not_confirmed")
                        return frame("brake", "speed_governor_waiting_for_support_dispatch" if supported_stop
                                     else "speed_governor_waiting_for_zero_dispatch",
                                     q, 0.0, control_v, 0.0, 0.0, 0.0,
                                     brake_elapsed_sec=elapsed)
                    # Reuse the authoritative planner at fresh measured state
                    # after a confirmed stop baseline. Keep the SAME goal and the
                    # caller's leg clock/budget; no new onset authorization.
                    self._planner.reset((q,), monotonic_time=now)
                    self._planner.accept_target((self._target,), sequence=0,
                        published_monotonic_time=now, monotonic_time=now)
                    self._speed_governor_resume_count += 1
                    self._brake_start = self._stationary_start = None
                    self._stationary_position = None
                    self._stationary_min = self._stationary_max = None
                    self._brake_trigger_reference = self._brake_trigger_lag = None
                    return frame("track", "speed_governor_resumed_supported_target" if supported_stop
                                 else "speed_governor_resumed_original_target",
                                 q, 0.0, control_v, 0.0, 0.0, 0.0)
                retain_target = (
                    calibration_target_hold and not request_brake and speed_governor_permitted
                    and (self._governed_speed_brake(self._brake_reason)
                         or self._brake_reason in {"soft_hold_motion_braking", "support_velocity_braking"})
                    and abs(self._target-q) <= min(stop_error_rad, self.config.max_reference_lag_rad)
                )
                if retain_target and not stop_dispatch_confirmed:
                    return frame("brake", "speed_governor_waiting_for_support_dispatch" if supported_stop
                                 else "speed_governor_waiting_for_zero_dispatch",
                                 q, 0.0, control_v, 0.0, 0.0, 0.0,
                                 brake_elapsed_sec=elapsed)
                # Only retain an already-admitted nearby calibration target.
                # Terminal reference-lag/operator stops still capture q.
                self._hold_position = self._target if retain_target else q
                if retain_target:
                    # This completed stop now owns an active target servo.
                    # Keep its goal/count/deadline; do not leave a stale brake
                    # latch that would veto the next fixed-command probe.
                    self._brake_start = self._stationary_start = None
                    self._brake_reason = "calibration_target_tracking"
                return frame(
                    "soft_hold",
                    "calibration_target_hold_captured" if retain_target else "measured_stop_point_captured",
                    self._hold_position,
                    0.0,
                    control_v,
                    0.0,
                    0.0,
                    0.0,
                    self._hold_position,
                    elapsed,
                )
            if stop_capture_block_reason is None:
                stop_capture_block_reason = "stationary_dwell_incomplete"
        else:
            self._stationary_start = None
            self._stationary_position = None
            # Explain the existing gate; these fields never authorize capture.
            stop_capture_block_reason = evidence_reason if resolution is not None else (
                "selected_raw_velocity" if abs(raw_v) > stop_velocity_rad_s else
                "selected_window_velocity" if abs(control_v) > stop_velocity_rad_s else
                "support_window_velocity" if support_max_velocity_rad_s > stop_velocity_rad_s else
                "support_raw_velocity"
            )
        # Only an explicitly authorized normal arrival retains the current
        # goal. Withdrawal, faults and speed braking never retain this spring.
        # Legacy braking also opposes raw speed;
        # quantized-feedback mode uses the window and keeps the raw hard guard.
        brake_v = control_v if self.config.windowed_soft_velocity else (
            raw_v if raw_v * control_v < 0.0 or abs(raw_v) > abs(control_v) else control_v
        )
        if self.config.windowed_soft_velocity and continuous_limit is not None:
            # A causal window can retain the old direction after a gearbox
            # recoil. Never turn damping into propulsion in the latest paired
            # position interval. Require agreement for a nonzero request and
            # cap its magnitude by BOTH estimates; one-count raw noise cannot
            # amplify the window brake. Motor-local damping remains active.
            brake_v = (math.copysign(min(abs(raw_v), abs(control_v)), raw_v)
                       if raw_v * control_v > 0.0 else 0.0)
        d = -self.config.braking_kd_nm_s_per_rad * brake_v
        p = kp * (self._target - q) if arrival_active else 0.0
        if arrival_active and self._arrival_unloading:
            p = d = 0.0
        return frame(
            "brake",
            self._brake_reason,
            self._target if arrival_active else q,
            0.0,
            control_v,
            p,
            d,
            authority * (p + d),
            brake_elapsed_sec=elapsed,
        )
