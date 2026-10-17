"""Offline continuous joint trajectories through timestamped waypoints.

This module owns only backend-neutral trajectory construction and
certification.  It owns no replay cursor, clock, safety authority, collision
planner, dispatch, or hardware state.

The trajectory treats source-rate positions as waypoints rather than implicit
zero-velocity stops.  Piecewise quintics share position, velocity and
acceleration at every knot.  The complete polynomial intervals are checked
analytically against joint position, velocity, acceleration and jerk limits;
checking only the output sample grid would miss between-tick extrema.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .horizon_governor import JointHorizonLimits
from .resampling import validate_resample_rate

_TIME_EPSILON_SEC = 1e-12
_ROOT_IMAG_TOLERANCE = 1e-9
# Match the shared position-governor's microradian-scale numerical contract.
# This tolerance only accepts solver/serialization residue; it is not a motion
# margin and no output is silently clipped here.
_CERTIFICATE_TOLERANCE = 1e-7

OFFLINE_WAYPOINT_TIMING_STRICT = "strict_fixed_timing"
OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL = "safe_retime_global"
OFFLINE_WAYPOINT_TIMING_MODES = (
    OFFLINE_WAYPOINT_TIMING_STRICT,
    OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
)


class OfflineWaypointTrajectoryInfeasibleError(RuntimeError):
    """The fixed-time waypoint trajectory failed a continuous hard limit."""


@dataclass(frozen=True)
class OfflineJointWaypointTrajectory:
    """One sampled C2 joint trajectory and its continuous-time certificate."""

    timestamps: np.ndarray
    q_rad: np.ndarray
    qd_rad_s: np.ndarray
    qdd_rad_s2: np.ndarray
    jerk_rad_s3: np.ndarray
    source_waypoint_timestamps: np.ndarray
    waypoint_timestamps: np.ndarray
    waypoint_q_rad: np.ndarray
    knot_qd_rad_s: np.ndarray
    knot_qdd_rad_s2: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        vectors = {}
        for name in (
            "timestamps",
            "q_rad",
            "qd_rad_s",
            "qdd_rad_s2",
            "jerk_rad_s3",
            "source_waypoint_timestamps",
            "waypoint_timestamps",
            "waypoint_q_rad",
            "knot_qd_rad_s",
            "knot_qdd_rad_s2",
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
            value.setflags(write=False)
            vectors[name] = value
        if vectors["timestamps"].ndim != 1 or vectors["timestamps"].size < 2:
            raise ValueError("timestamps must be a vector with at least two values.")
        shape = vectors["q_rad"].shape
        if len(shape) != 2 or shape[0] != vectors["timestamps"].size:
            raise ValueError("q_rad must be shaped sample_count x dimension.")
        for name in ("qd_rad_s", "qdd_rad_s2", "jerk_rad_s3"):
            if vectors[name].shape != shape:
                raise ValueError(f"{name} must match q_rad shape.")
        if vectors["source_waypoint_timestamps"].shape != vectors[
            "waypoint_timestamps"
        ].shape:
            raise ValueError(
                "source_waypoint_timestamps must match waypoint_timestamps."
            )
        waypoint_shape = vectors["waypoint_q_rad"].shape
        if (
            len(waypoint_shape) != 2
            or waypoint_shape[0] != vectors["waypoint_timestamps"].size
            or waypoint_shape[1] != shape[1]
        ):
            raise ValueError(
                "waypoint_q_rad must match waypoint timestamps and trajectory dimension."
            )
        for name in ("knot_qd_rad_s", "knot_qdd_rad_s2"):
            if vectors[name].shape != waypoint_shape:
                raise ValueError(f"{name} must match waypoint_q_rad shape.")
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def dimension(self) -> int:
        return int(self.q_rad.shape[1])


def _normalize_waypoints(
    timestamps: Sequence[float] | np.ndarray,
    q_waypoints_rad: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float).reshape(-1).copy()
    q = np.asarray(q_waypoints_rad, dtype=float).copy()
    if times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("timestamps must contain at least two finite values.")
    if q.ndim != 2 or q.shape[0] != times.size or q.shape[1] < 1:
        raise ValueError(
            "q_waypoints_rad must be shaped waypoint_count x positive_dimension."
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("q_waypoints_rad must contain only finite values.")
    if np.any(np.diff(times) <= _TIME_EPSILON_SEC):
        raise ValueError("waypoint timestamps must be strictly increasing.")
    return times, q


def normalize_offline_waypoint_timing_mode(value: str | None) -> str:
    mode = str(value or OFFLINE_WAYPOINT_TIMING_STRICT).strip().lower()
    aliases = {
        "strict": OFFLINE_WAYPOINT_TIMING_STRICT,
        "fixed": OFFLINE_WAYPOINT_TIMING_STRICT,
        "retime": OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
        "safe_retime": OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL,
    }
    mode = aliases.get(mode, mode)
    if mode not in OFFLINE_WAYPOINT_TIMING_MODES:
        raise ValueError(
            f"timing_mode must be one of {OFFLINE_WAYPOINT_TIMING_MODES}, "
            f"got {value!r}."
        )
    return mode


def _fixed_rate_grid_through(
    start_time: float,
    deadline: float,
    *,
    hz: float,
) -> np.ndarray:
    duration = max(0.0, float(deadline) - float(start_time))
    final_index = int(math.ceil(duration * float(hz) - _TIME_EPSILON_SEC))
    return float(start_time) + np.arange(final_index + 1, dtype=float) / float(hz)


def _normalize_endpoint_vector(
    value: Sequence[float] | np.ndarray | None,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros(dimension, dtype=float)
    vector = np.asarray(value, dtype=float).reshape(-1).copy()
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite {dimension}-vector.")
    return vector


def _estimate_knot_velocities(
    timestamps: np.ndarray,
    q_waypoints: np.ndarray,
    *,
    initial_velocity_rad_s: np.ndarray,
    final_velocity_rad_s: np.ndarray,
) -> np.ndarray:
    count, dimension = q_waypoints.shape
    velocities = np.zeros((count, dimension), dtype=float)
    for joint in range(dimension):
        # PCHIP supplies a shape-preserving derivative: monotone source motion
        # stays monotone and a genuine source reversal receives a zero tangent.
        interpolator = PchipInterpolator(
            timestamps,
            q_waypoints[:, joint],
            extrapolate=False,
        )
        velocities[:, joint] = np.asarray(
            interpolator.derivative(1)(timestamps),
            dtype=float,
        )
    velocities[0] = initial_velocity_rad_s
    velocities[-1] = final_velocity_rad_s
    return velocities


def _quintic_coefficients(
    q0: np.ndarray,
    q1: np.ndarray,
    qd0: np.ndarray,
    qd1: np.ndarray,
    qdd0: np.ndarray,
    qdd1: np.ndarray,
    duration_sec: float,
) -> np.ndarray:
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive.")
    c0 = np.asarray(q0, dtype=float)
    c1 = np.asarray(qd0, dtype=float)
    c2 = 0.5 * np.asarray(qdd0, dtype=float)
    position_residual = q1 - (c0 + c1 * duration + c2 * duration**2)
    velocity_residual = qd1 - (c1 + 2.0 * c2 * duration)
    acceleration_residual = qdd1 - 2.0 * c2
    system = np.asarray(
        (
            (duration**3, duration**4, duration**5),
            (3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4),
            (6.0 * duration, 12.0 * duration**2, 20.0 * duration**3),
        ),
        dtype=float,
    )
    tail = np.linalg.solve(
        system,
        np.stack(
            (position_residual, velocity_residual, acceleration_residual),
            axis=0,
        ),
    )
    return np.stack((c0, c1, c2, tail[0], tail[1], tail[2]), axis=0)


def _differentiate_coefficients(coefficients: np.ndarray, order: int) -> np.ndarray:
    result = np.asarray(coefficients, dtype=float)
    for _ in range(int(order)):
        if result.shape[0] <= 1:
            return np.zeros((1, result.shape[1]), dtype=float)
        powers = np.arange(1, result.shape[0], dtype=float)[:, None]
        result = result[1:] * powers
    return result


def _evaluate_coefficients(coefficients: np.ndarray, tau: float) -> np.ndarray:
    value = np.zeros(coefficients.shape[1], dtype=float)
    for coefficient in coefficients[::-1]:
        value = value * float(tau) + coefficient
    return value


def _real_roots_in_interval(coefficients: np.ndarray, duration: float) -> tuple[float, ...]:
    polynomial = np.asarray(coefficients, dtype=float).reshape(-1)
    while polynomial.size > 1 and abs(float(polynomial[-1])) <= 1e-14:
        polynomial = polynomial[:-1]
    if polynomial.size <= 1:
        return ()
    roots = np.polynomial.polynomial.polyroots(polynomial)
    values = []
    for root in roots:
        if abs(float(np.imag(root))) > _ROOT_IMAG_TOLERANCE:
            continue
        value = float(np.real(root))
        if -_TIME_EPSILON_SEC <= value <= duration + _TIME_EPSILON_SEC:
            values.append(float(np.clip(value, 0.0, duration)))
    return tuple(sorted(set(values)))


def _continuous_extrema(
    segment_coefficients: Sequence[np.ndarray],
    segment_durations: np.ndarray,
) -> dict[str, np.ndarray]:
    dimension = int(segment_coefficients[0].shape[1])
    minimum_position = np.full(dimension, np.inf, dtype=float)
    maximum_position = np.full(dimension, -np.inf, dtype=float)
    maximum_velocity = np.zeros(dimension, dtype=float)
    maximum_acceleration = np.zeros(dimension, dtype=float)
    maximum_jerk = np.zeros(dimension, dtype=float)
    for coefficients, duration in zip(segment_coefficients, segment_durations):
        derivative_sets = tuple(
            _differentiate_coefficients(coefficients, order)
            for order in range(5)
        )
        for joint in range(dimension):
            for order, target in (
                (0, None),
                (1, maximum_velocity),
                (2, maximum_acceleration),
                (3, maximum_jerk),
            ):
                candidates = [0.0, float(duration)]
                candidates.extend(
                    _real_roots_in_interval(
                        derivative_sets[order + 1][:, joint],
                        float(duration),
                    )
                )
                values = np.asarray(
                    [
                        _evaluate_coefficients(
                            derivative_sets[order][:, joint : joint + 1],
                            tau,
                        )[0]
                        for tau in candidates
                    ],
                    dtype=float,
                )
                if order == 0:
                    minimum_position[joint] = min(
                        minimum_position[joint],
                        float(np.min(values)),
                    )
                    maximum_position[joint] = max(
                        maximum_position[joint],
                        float(np.max(values)),
                    )
                else:
                    assert target is not None
                    target[joint] = max(target[joint], float(np.max(np.abs(values))))
    return {
        "min_position_rad": minimum_position,
        "max_position_rad": maximum_position,
        "max_abs_velocity_rad_s": maximum_velocity,
        "max_abs_acceleration_rad_s2": maximum_acceleration,
        "max_abs_jerk_rad_s3": maximum_jerk,
    }


def _certify_extrema(
    extrema: Mapping[str, np.ndarray],
    limits: JointHorizonLimits,
) -> dict[str, Any]:
    if limits.dimension != int(extrema["min_position_rad"].size):
        raise ValueError("trajectory limits dimension does not match waypoints.")
    violations: list[str] = []
    minimum_position = np.asarray(extrema["min_position_rad"], dtype=float)
    maximum_position = np.asarray(extrema["max_position_rad"], dtype=float)
    maximum_velocity = np.asarray(extrema["max_abs_velocity_rad_s"], dtype=float)
    maximum_acceleration = np.asarray(
        extrema["max_abs_acceleration_rad_s2"],
        dtype=float,
    )
    maximum_jerk = np.asarray(extrema["max_abs_jerk_rad_s3"], dtype=float)
    checks = (
        (
            "position_lower",
            minimum_position >= limits.min_position_rad - _CERTIFICATE_TOLERANCE,
        ),
        (
            "position_upper",
            maximum_position <= limits.max_position_rad + _CERTIFICATE_TOLERANCE,
        ),
        (
            "velocity",
            maximum_velocity <= limits.max_velocity_rad_s + _CERTIFICATE_TOLERANCE,
        ),
        (
            "acceleration",
            maximum_acceleration
            <= limits.max_acceleration_rad_s2 + _CERTIFICATE_TOLERANCE,
        ),
        (
            "jerk",
            maximum_jerk <= limits.max_jerk_rad_s3 + _CERTIFICATE_TOLERANCE,
        ),
    )
    for label, passed_by_joint in checks:
        for joint in np.flatnonzero(~passed_by_joint):
            violations.append(f"{label}:joint_{int(joint) + 1}")
    return {
        "passed": not violations,
        "violations": tuple(violations),
        **{name: np.asarray(value, dtype=float).tolist() for name, value in extrema.items()},
    }


def _dynamic_time_scale(
    extrema: Mapping[str, np.ndarray],
    limits: JointHorizonLimits,
) -> float:
    velocity_ratio = float(
        np.max(
            np.asarray(extrema["max_abs_velocity_rad_s"], dtype=float)
            / limits.max_velocity_rad_s
        )
    )
    acceleration_ratio = float(
        np.max(
            np.asarray(extrema["max_abs_acceleration_rad_s2"], dtype=float)
            / limits.max_acceleration_rad_s2
        )
    )
    jerk_ratio = float(
        np.max(
            np.asarray(extrema["max_abs_jerk_rad_s3"], dtype=float)
            / limits.max_jerk_rad_s3
        )
    )
    return max(
        1.0,
        velocity_ratio,
        math.sqrt(max(0.0, acceleration_ratio)),
        np.cbrt(max(0.0, jerk_ratio)),
    )


def _trajectory_segments(
    timestamps: np.ndarray,
    q_waypoints: np.ndarray,
    *,
    initial_velocity_rad_s: np.ndarray,
    final_velocity_rad_s: np.ndarray,
    initial_acceleration_rad_s2: np.ndarray,
    final_acceleration_rad_s2: np.ndarray,
    stop_at_waypoints: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], dict[str, np.ndarray]]:
    knot_velocity = _estimate_knot_velocities(
        timestamps,
        q_waypoints,
        initial_velocity_rad_s=initial_velocity_rad_s,
        final_velocity_rad_s=final_velocity_rad_s,
    )
    if stop_at_waypoints:
        knot_velocity[1:-1] = 0.0
    knot_acceleration = np.zeros_like(q_waypoints)
    knot_acceleration[0] = initial_acceleration_rad_s2
    knot_acceleration[-1] = final_acceleration_rad_s2
    durations = np.diff(timestamps)
    coefficients = tuple(
        _quintic_coefficients(
            q_waypoints[index],
            q_waypoints[index + 1],
            knot_velocity[index],
            knot_velocity[index + 1],
            knot_acceleration[index],
            knot_acceleration[index + 1],
            durations[index],
        )
        for index in range(timestamps.size - 1)
    )
    return (
        knot_velocity,
        knot_acceleration,
        coefficients,
        _continuous_extrema(coefficients, durations),
    )


def build_offline_quintic_waypoint_trajectory(
    timestamps: Sequence[float] | np.ndarray,
    q_waypoints_rad: Sequence[Sequence[float]] | np.ndarray,
    *,
    command_hz: float,
    limits: JointHorizonLimits,
    initial_velocity_rad_s: Sequence[float] | np.ndarray | None = None,
    initial_acceleration_rad_s2: Sequence[float] | np.ndarray | None = None,
    final_velocity_rad_s: Sequence[float] | np.ndarray | None = None,
    final_acceleration_rad_s2: Sequence[float] | np.ndarray | None = None,
    sample_timestamps: Sequence[float] | np.ndarray | None = None,
    timing_mode: str = OFFLINE_WAYPOINT_TIMING_STRICT,
    max_time_scale: float = 4.0,
    terminal_hold_sec: float = 0.0,
    stop_at_waypoints: bool = False,
) -> OfflineJointWaypointTrajectory:
    """Build an exact fixed-time C2 waypoint trajectory or fail closed.

    Internal knot velocities are shape-preserving and normally non-zero for
    sustained motion.  Endpoint derivatives default to zero because replay
    starts after explicit qpos initialization and ends in a hold.  All
    polynomial extrema are certified, not just the sampled command ticks.
    ``stop_at_waypoints`` explicitly preserves each straight joint-space leg
    with zero knot derivatives; the default retains continuous through-motion.
    """

    source_times, q_waypoints = _normalize_waypoints(timestamps, q_waypoints_rad)
    times = source_times.copy()
    if not isinstance(limits, JointHorizonLimits):
        raise TypeError("limits must be JointHorizonLimits.")
    if limits.dimension != q_waypoints.shape[1]:
        raise ValueError("limits dimension does not match q_waypoints_rad.")
    rate = validate_resample_rate(command_hz, name="command_hz")
    timing = normalize_offline_waypoint_timing_mode(timing_mode)
    maximum_time_scale = float(max_time_scale)
    if not math.isfinite(maximum_time_scale) or maximum_time_scale < 1.0:
        raise ValueError("max_time_scale must be finite and at least 1.0.")
    terminal_hold = float(terminal_hold_sec)
    if not math.isfinite(terminal_hold) or terminal_hold < 0.0:
        raise ValueError("terminal_hold_sec must be finite and non-negative.")
    if sample_timestamps is not None and timing != OFFLINE_WAYPOINT_TIMING_STRICT:
        raise ValueError(
            "custom sample_timestamps cannot be combined with safe retiming."
        )
    dimension = int(q_waypoints.shape[1])
    initial_velocity = _normalize_endpoint_vector(
        initial_velocity_rad_s,
        dimension=dimension,
        name="initial_velocity_rad_s",
    )
    final_velocity = _normalize_endpoint_vector(
        final_velocity_rad_s,
        dimension=dimension,
        name="final_velocity_rad_s",
    )
    initial_acceleration = _normalize_endpoint_vector(
        initial_acceleration_rad_s2,
        dimension=dimension,
        name="initial_acceleration_rad_s2",
    )
    final_acceleration = _normalize_endpoint_vector(
        final_acceleration_rad_s2,
        dimension=dimension,
        name="final_acceleration_rad_s2",
    )
    if terminal_hold > 0.0 and (
        np.any(np.abs(final_velocity) > _TIME_EPSILON_SEC)
        or np.any(np.abs(final_acceleration) > _TIME_EPSILON_SEC)
    ):
        raise ValueError(
            "terminal_hold_sec requires zero final velocity and acceleration."
        )
    if not isinstance(stop_at_waypoints, bool):
        raise TypeError("stop_at_waypoints must be boolean.")
    if stop_at_waypoints and any(
        np.any(np.abs(value) > _TIME_EPSILON_SEC)
        for value in (
            initial_velocity,
            final_velocity,
            initial_acceleration,
            final_acceleration,
        )
    ):
        raise ValueError("stop_at_waypoints requires zero endpoint derivatives.")
    (
        knot_velocity,
        knot_acceleration,
        coefficients,
        extrema,
    ) = _trajectory_segments(
        times,
        q_waypoints,
        initial_velocity_rad_s=initial_velocity,
        final_velocity_rad_s=final_velocity,
        initial_acceleration_rad_s2=initial_acceleration,
        final_acceleration_rad_s2=final_acceleration,
        stop_at_waypoints=stop_at_waypoints,
    )
    certificate = _certify_extrema(extrema, limits)
    time_scale = 1.0
    if not bool(certificate["passed"]) and timing == (
        OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL
    ):
        position_violations = tuple(
            violation
            for violation in certificate["violations"]
            if violation.startswith("position_")
        )
        if position_violations:
            raise OfflineWaypointTrajectoryInfeasibleError(
                "waypoint geometry violates hard position limits and cannot be "
                "fixed by retiming: " + ", ".join(position_violations)
            )
        time_scale = _dynamic_time_scale(extrema, limits) * (1.0 + 1e-6)
        for _ in range(8):
            if time_scale > maximum_time_scale + _TIME_EPSILON_SEC:
                break
            times = source_times[0] + (
                source_times - source_times[0]
            ) * time_scale
            (
                knot_velocity,
                knot_acceleration,
                coefficients,
                extrema,
            ) = _trajectory_segments(
                times,
                q_waypoints,
                initial_velocity_rad_s=initial_velocity,
                final_velocity_rad_s=final_velocity,
                initial_acceleration_rad_s2=initial_acceleration,
                final_acceleration_rad_s2=final_acceleration,
                stop_at_waypoints=stop_at_waypoints,
            )
            certificate = _certify_extrema(extrema, limits)
            if bool(certificate["passed"]):
                break
            time_scale *= 1.01
    if not bool(certificate["passed"]):
        raise OfflineWaypointTrajectoryInfeasibleError(
            f"continuous waypoint trajectory violates hard limits under "
            f"timing_mode={timing!r}, time_scale={time_scale:.6f}, "
            f"max_time_scale={maximum_time_scale:.6f}: "
            + ", ".join(certificate["violations"])
        )

    durations = np.diff(times)

    if sample_timestamps is None:
        sample_times = _fixed_rate_grid_through(
            float(times[0]),
            float(times[-1]) + terminal_hold,
            hz=rate,
        )
    else:
        sample_times = np.asarray(sample_timestamps, dtype=float).reshape(-1).copy()
        if (
            sample_times.size < 2
            or not np.all(np.isfinite(sample_times))
            or np.any(np.diff(sample_times) <= _TIME_EPSILON_SEC)
        ):
            raise ValueError(
                "sample_timestamps must contain at least two strictly increasing "
                "finite values."
            )
        if sample_times[0] < times[0] - _TIME_EPSILON_SEC:
            raise ValueError("sample_timestamps cannot precede the first waypoint.")
    sample_q = np.empty((sample_times.size, dimension), dtype=float)
    sample_qd = np.empty_like(sample_q)
    sample_qdd = np.empty_like(sample_q)
    sample_jerk = np.empty_like(sample_q)
    for sample_index, timestamp in enumerate(sample_times):
        if timestamp >= times[-1] - _TIME_EPSILON_SEC:
            sample_q[sample_index] = q_waypoints[-1]
            sample_qd[sample_index] = final_velocity
            sample_qdd[sample_index] = final_acceleration
            sample_jerk[sample_index] = 0.0
            continue
        segment_index = int(
            np.clip(
                np.searchsorted(times, timestamp, side="right") - 1,
                0,
                len(coefficients) - 1,
            )
        )
        tau = float(np.clip(timestamp - times[segment_index], 0.0, durations[segment_index]))
        segment = coefficients[segment_index]
        sample_q[sample_index] = _evaluate_coefficients(segment, tau)
        sample_qd[sample_index] = _evaluate_coefficients(
            _differentiate_coefficients(segment, 1),
            tau,
        )
        sample_qdd[sample_index] = _evaluate_coefficients(
            _differentiate_coefficients(segment, 2),
            tau,
        )
        sample_jerk[sample_index] = _evaluate_coefficients(
            _differentiate_coefficients(segment, 3),
            tau,
        )

    return OfflineJointWaypointTrajectory(
        timestamps=sample_times,
        q_rad=sample_q,
        qd_rad_s=sample_qd,
        qdd_rad_s2=sample_qdd,
        jerk_rad_s3=sample_jerk,
        source_waypoint_timestamps=source_times,
        waypoint_timestamps=times,
        waypoint_q_rad=q_waypoints,
        knot_qd_rad_s=knot_velocity,
        knot_qdd_rad_s2=knot_acceleration,
        metadata={
            "planner": "offline_c2_quintic_waypoint_v1",
            "timing_mode": timing,
            "time_scale": float(time_scale),
            "retimed": bool(time_scale > 1.0 + _TIME_EPSILON_SEC),
            "max_time_scale": maximum_time_scale,
            "source_duration_sec": float(source_times[-1] - source_times[0]),
            "execution_duration_sec": float(times[-1] - times[0]),
            "terminal_hold_sec": terminal_hold,
            "waypoint_position_exact": True,
            "internal_knot_zero_velocity_required": stop_at_waypoints,
            "endpoint_velocity_default": "zero",
            "position_interpolation": "piecewise_quintic_hermite",
            "knot_velocity_estimator": "zero_at_waypoints"
            if stop_at_waypoints
            else "pchip_shape_preserving",
            "knot_acceleration": "shared_zero_internal",
            "continuous_hard_limit_certificate": certificate,
            "command_hz": rate,
            "sample_count": int(sample_times.size),
            "waypoint_count": int(times.size),
            "custom_sample_timestamps": sample_timestamps is not None,
        },
    )


__all__ = [
    "OFFLINE_WAYPOINT_TIMING_MODES",
    "OFFLINE_WAYPOINT_TIMING_SAFE_RETIME_GLOBAL",
    "OFFLINE_WAYPOINT_TIMING_STRICT",
    "OfflineJointWaypointTrajectory",
    "OfflineWaypointTrajectoryInfeasibleError",
    "build_offline_quintic_waypoint_trajectory",
    "normalize_offline_waypoint_timing_mode",
]
