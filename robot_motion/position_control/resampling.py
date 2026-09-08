from __future__ import annotations

import copy
import math
import threading
from bisect import bisect_right
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.pose import interpolate_pose
from robot_motion.transforms import parse_pose_matrix, validate_pose_matrix

from .interpolation import DEFAULT_POSITION_REFERENCE_HZ


_TIME_EPSILON_SEC = 1e-12
POSE_ACTION_INTERVAL_VELOCITY_SCHEMA = (
    "robot_motion.pose_action_interval_velocity/v1"
)


def _finite_time(value: float, *, name: str) -> float:
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError(f"{name} must be finite")
    return timestamp


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        out = np.array(value, copy=True)
        out.setflags(write=False)
        return out
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _freeze_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _normalize_pose_mapping(
    values: Mapping[str, np.ndarray | Sequence[float]],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, raw_pose in values.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError(f"{name} contains an empty side")
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}")
        try:
            pose = parse_pose_matrix(raw_pose)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{side} is not a supported pose") from exc
        valid, reason = validate_pose_matrix(pose)
        if not valid:
            raise ValueError(f"{name}.{side} is invalid: {reason}")
        pose = np.array(pose, dtype=float, copy=True)
        pose.setflags(write=False)
        normalized[side] = pose
    return MappingProxyType(normalized)


def _normalize_qpos_mapping(
    values: Mapping[str, np.ndarray | Sequence[float]],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, raw_qpos in values.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError(f"{name} contains an empty side")
        if side in normalized:
            raise ValueError(f"{name} contains duplicate side {side!r}")
        qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
        if qpos.size == 0 or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}.{side} must be a non-empty finite vector")
        qpos = qpos.astype(float, copy=True)
        qpos.setflags(write=False)
        normalized[side] = qpos
    return MappingProxyType(normalized)


def _validate_modalities(
    pose_by_side: Mapping[str, np.ndarray],
    qpos_by_side: Mapping[str, np.ndarray],
    *,
    name: str,
) -> None:
    if not pose_by_side and not qpos_by_side:
        raise ValueError(f"{name} must contain pose_by_side and/or qpos_by_side")
    if pose_by_side and qpos_by_side and set(pose_by_side) != set(qpos_by_side):
        raise ValueError(
            f"{name} pose_by_side and qpos_by_side must contain the same sides when both are set"
        )


@dataclass(frozen=True)
class TimestampedMotionTarget:
    """A timestamped pose and/or qpos keyframe from an action/reference stream.

    ``timestamp`` is the target's logical motion time. ``available_timestamp``
    records when the producer made that target available to the consumer. They
    are normally equal, but predicted action chunks legitimately make a future
    logical target available now. Keeping both clocks prevents a causal
    resampler from confusing known lookahead with unavailable future input.
    """

    timestamp: float
    source_index: int
    pose_by_side: Mapping[str, np.ndarray | Sequence[float]] = field(default_factory=dict)
    qpos_by_side: Mapping[str, np.ndarray | Sequence[float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    available_timestamp: float | None = None

    def __post_init__(self) -> None:
        pose_by_side = _normalize_pose_mapping(self.pose_by_side, name="pose_by_side")
        qpos_by_side = _normalize_qpos_mapping(self.qpos_by_side, name="qpos_by_side")
        _validate_modalities(pose_by_side, qpos_by_side, name="motion target")
        timestamp = _finite_time(self.timestamp, name="timestamp")
        available_timestamp = (
            timestamp
            if self.available_timestamp is None
            else _finite_time(
                self.available_timestamp,
                name="available_timestamp",
            )
        )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "available_timestamp", available_timestamp)
        object.__setattr__(
            self,
            "source_index",
            _non_negative_int(self.source_index, name="source_index"),
        )
        object.__setattr__(self, "pose_by_side", pose_by_side)
        object.__setattr__(self, "qpos_by_side", qpos_by_side)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def sides(self) -> tuple[str, ...]:
        values = self.pose_by_side if self.pose_by_side else self.qpos_by_side
        return tuple(values)


@dataclass(frozen=True)
class ResampledMotionReference:
    """One immutable result of timestamp-driven motion resampling."""

    sequence: int
    timestamp: float
    left_source_index: int
    right_source_index: int
    left_source_timestamp: float
    right_source_timestamp: float
    interpolation_alpha: float
    pose_by_side: Mapping[str, np.ndarray | Sequence[float]] = field(default_factory=dict)
    qpos_by_side: Mapping[str, np.ndarray | Sequence[float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pose_by_side = _normalize_pose_mapping(self.pose_by_side, name="pose_by_side")
        qpos_by_side = _normalize_qpos_mapping(self.qpos_by_side, name="qpos_by_side")
        _validate_modalities(pose_by_side, qpos_by_side, name="resampled motion reference")
        alpha = float(self.interpolation_alpha)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("interpolation_alpha must be finite and in [0, 1]")
        left_timestamp = _finite_time(
            self.left_source_timestamp,
            name="left_source_timestamp",
        )
        right_timestamp = _finite_time(
            self.right_source_timestamp,
            name="right_source_timestamp",
        )
        if right_timestamp < left_timestamp:
            raise ValueError("right_source_timestamp must be >= left_source_timestamp")
        object.__setattr__(
            self,
            "sequence",
            _non_negative_int(self.sequence, name="sequence"),
        )
        object.__setattr__(self, "timestamp", _finite_time(self.timestamp, name="timestamp"))
        object.__setattr__(
            self,
            "left_source_index",
            _non_negative_int(self.left_source_index, name="left_source_index"),
        )
        object.__setattr__(
            self,
            "right_source_index",
            _non_negative_int(self.right_source_index, name="right_source_index"),
        )
        object.__setattr__(self, "left_source_timestamp", left_timestamp)
        object.__setattr__(self, "right_source_timestamp", right_timestamp)
        object.__setattr__(self, "interpolation_alpha", alpha)
        object.__setattr__(self, "pose_by_side", pose_by_side)
        object.__setattr__(self, "qpos_by_side", qpos_by_side)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def sides(self) -> tuple[str, ...]:
        values = self.pose_by_side if self.pose_by_side else self.qpos_by_side
        return tuple(values)


def validate_resample_rate(value: float, *, name: str = "reference_hz") -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return rate


def build_uniform_time_grid(
    start_timestamp: float,
    end_timestamp: float,
    *,
    hz: float,
    include_final: bool = True,
) -> tuple[float, ...]:
    """Build a drift-free rate grid and optionally append an unaligned endpoint."""

    start = _finite_time(start_timestamp, name="start_timestamp")
    end = _finite_time(end_timestamp, name="end_timestamp")
    if end < start:
        raise ValueError("end_timestamp must be >= start_timestamp")
    rate = validate_resample_rate(hz, name="hz")
    if math.isclose(start, end, rel_tol=0.0, abs_tol=_TIME_EPSILON_SEC):
        return (start,)

    span = end - start
    regular_steps = max(0, int(math.floor(span * rate + _TIME_EPSILON_SEC)))
    values = [start + float(index) / rate for index in range(regular_steps + 1)]
    while values and values[-1] > end + _TIME_EPSILON_SEC:
        values.pop()
    if not values:
        values = [start]
    if math.isclose(values[-1], end, rel_tol=0.0, abs_tol=_TIME_EPSILON_SEC):
        values[-1] = end
    elif include_final:
        values.append(end)
    return tuple(float(value) for value in values)


def interpolate_pose_shortest_path(
    start_pose: np.ndarray | Sequence[float],
    end_pose: np.ndarray | Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Linearly interpolate xyz and use shortest-path quaternion SLERP."""

    start = parse_pose_matrix(start_pose)
    end = parse_pose_matrix(end_pose)
    valid_start, start_reason = validate_pose_matrix(start)
    valid_end, end_reason = validate_pose_matrix(end)
    if not valid_start:
        raise ValueError(f"start_pose is invalid: {start_reason}")
    if not valid_end:
        raise ValueError(f"end_pose is invalid: {end_reason}")
    blend = float(alpha)
    if not math.isfinite(blend) or blend < 0.0 or blend > 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    # robot_motion.pose.interpolate_pose uses scipy Rotation/Slerp. Rotation's
    # double-cover handling selects the shortest quaternion arc.
    return interpolate_pose(start, end, blend)


def interpolate_qpos_by_side(
    start_qpos_by_side: Mapping[str, np.ndarray | Sequence[float]],
    end_qpos_by_side: Mapping[str, np.ndarray | Sequence[float]],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Linearly interpolate matching finite qpos vectors for every side."""

    start = _normalize_qpos_mapping(start_qpos_by_side, name="start_qpos_by_side")
    end = _normalize_qpos_mapping(end_qpos_by_side, name="end_qpos_by_side")
    if set(start) != set(end):
        raise ValueError("start_qpos_by_side and end_qpos_by_side must contain the same sides")
    blend = float(alpha)
    if not math.isfinite(blend) or blend < 0.0 or blend > 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    out: dict[str, np.ndarray] = {}
    for side in start:
        if start[side].shape != end[side].shape:
            raise ValueError(f"{side} start and end qpos shapes do not match")
        out[side] = np.asarray(
            start[side] + (end[side] - start[side]) * blend,
            dtype=float,
        )
    return out


def _target_signature(
    target: TimestampedMotionTarget,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    pose_sides = tuple(sorted(target.pose_by_side))
    qpos_sides = tuple(sorted(target.qpos_by_side))
    return (
        pose_sides,
        qpos_sides,
        tuple(target.qpos_by_side[side].size for side in qpos_sides),
    )


def _normalize_targets(
    samples: Sequence[TimestampedMotionTarget],
) -> tuple[TimestampedMotionTarget, ...]:
    normalized = tuple(samples)
    if not normalized:
        raise ValueError("samples must contain at least one motion target")
    if any(not isinstance(sample, TimestampedMotionTarget) for sample in normalized):
        raise TypeError("samples must contain only TimestampedMotionTarget values")
    signature = _target_signature(normalized[0])
    previous_timestamp = normalized[0].timestamp
    for index, sample in enumerate(normalized):
        if _target_signature(sample) != signature:
            raise ValueError("all motion targets must have matching modalities, sides, and qpos shapes")
        if index > 0 and sample.timestamp <= previous_timestamp:
            raise ValueError("motion target timestamps must be strictly increasing")
        previous_timestamp = sample.timestamp
    return normalized


def _interpolate_targets(
    left: TimestampedMotionTarget,
    right: TimestampedMotionTarget,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pose_by_side = {
        side: interpolate_pose_shortest_path(
            left.pose_by_side[side],
            right.pose_by_side[side],
            alpha,
        )
        for side in left.pose_by_side
    }
    qpos_by_side = (
        interpolate_qpos_by_side(left.qpos_by_side, right.qpos_by_side, alpha)
        if left.qpos_by_side
        else {}
    )
    return pose_by_side, qpos_by_side


def _pose_action_interval_velocity(
    samples: Sequence[TimestampedMotionTarget],
    *,
    query_timestamp: float,
    right_position: int,
    causal: bool,
) -> dict[str, Any] | None:
    """Resolve pose speed from adjacent source actions, never 60 Hz deltas.

    At an exact source-action timestamp the forward ``current -> next`` pair is
    preferred when that next action is already available.  A single-action
    chunk and the final action of a chunk therefore fall back to
    ``previous -> current``.  The causal resampler only passes samples that
    were available at emission time, so this rule cannot look across an
    unknown chunk boundary.
    """

    if not samples or not samples[0].pose_by_side:
        return None

    query = float(query_timestamp)
    start: TimestampedMotionTarget | None = None
    end: TimestampedMotionTarget | None = None
    selection: str

    if 0 < right_position < len(samples):
        current = samples[right_position - 1]
        successor = samples[right_position]
        # bisect_right means this is either the exact current action or a
        # query inside current -> successor.  Both use the same source-action
        # interval velocity.
        start, end = current, successor
        selection = (
            "current_to_next"
            if math.isclose(
                query,
                current.timestamp,
                rel_tol=0.0,
                abs_tol=_TIME_EPSILON_SEC,
            )
            else "bracketing_current_to_next"
        )
    elif len(samples) >= 2 and right_position >= len(samples):
        # No available successor: single-frame/final chunk step fallback.
        start, end = samples[-2], samples[-1]
        selection = "previous_to_current_fallback"
    elif len(samples) >= 2 and right_position == 0:
        # Offline queries before the first action are unusual, but a known
        # successor still provides the only non-fabricated source interval.
        start, end = samples[0], samples[1]
        selection = "first_to_next_fallback"
    else:
        return {
            "schema": POSE_ACTION_INTERVAL_VELOCITY_SCHEMA,
            "source": "adjacent_pose_actions",
            "quality_valid": False,
            "quality_reason": "no_adjacent_pose_action",
            "causal": bool(causal),
            "selection": "unavailable",
            "by_side": {},
        }

    duration = float(end.timestamp - start.timestamp)
    if not math.isfinite(duration) or duration <= _TIME_EPSILON_SEC:
        return {
            "schema": POSE_ACTION_INTERVAL_VELOCITY_SCHEMA,
            "source": "adjacent_pose_actions",
            "quality_valid": False,
            "quality_reason": "invalid_action_interval_duration",
            "causal": bool(causal),
            "selection": selection,
            "start_source_index": int(start.source_index),
            "end_source_index": int(end.source_index),
            "by_side": {},
        }

    by_side: dict[str, dict[str, Any]] = {}
    for side in start.pose_by_side:
        start_pose = np.asarray(start.pose_by_side[side], dtype=float)
        end_pose = np.asarray(end.pose_by_side[side], dtype=float)
        translation_delta = end_pose[:3, 3] - start_pose[:3, 3]
        # This is the shortest world-frame rotation vector, consistent with
        # the quaternion shortest-path SLERP used for the pose references.
        rotation_delta = Rotation.from_matrix(
            end_pose[:3, :3] @ start_pose[:3, :3].T
        ).as_rotvec()
        linear_velocity = translation_delta / duration
        angular_velocity = rotation_delta / duration
        linear_speed = float(np.linalg.norm(linear_velocity))
        angular_speed = float(np.linalg.norm(angular_velocity))
        by_side[side] = {
            "quality_valid": True,
            "linear_velocity_m_s": linear_velocity.astype(float).tolist(),
            "angular_velocity_rad_s": angular_velocity.astype(float).tolist(),
            # Keep the speed field names consumed by the shared adaptive
            # output stage.  These are one action-interval values, not a
            # percentile recomputed over 60 Hz references.
            "linear_speed_p90_m_s": linear_speed,
            "angular_speed_p90_rad_s": angular_speed,
            "translation_span_m": float(np.linalg.norm(translation_delta)),
            "rotation_span_rad": float(np.linalg.norm(rotation_delta)),
            "speed_source": "adjacent_pose_action_interval",
        }
    return {
        "schema": POSE_ACTION_INTERVAL_VELOCITY_SCHEMA,
        "source": "adjacent_pose_actions",
        "quality_valid": True,
        "quality_reason": None,
        "causal": bool(causal),
        "selection": selection,
        "start_source_index": int(start.source_index),
        "end_source_index": int(end.source_index),
        "start_source_timestamp": float(start.timestamp),
        "end_source_timestamp": float(end.timestamp),
        "duration_sec": duration,
        "source_rate_hz": 1.0 / duration,
        "velocity_frame": "source_pose_reference_frame",
        "rotation_velocity_convention": "shortest_world_rotvec_per_sec",
        "by_side": by_side,
    }


def _resample_at(
    samples: Sequence[TimestampedMotionTarget],
    *,
    query_timestamp: float,
    sequence: int,
    causal: bool,
    emitted_timestamp: float,
    source_timestamps: Sequence[float] | None = None,
) -> ResampledMotionReference:
    timestamps = (
        tuple(float(value) for value in source_timestamps)
        if source_timestamps is not None
        else tuple(sample.timestamp for sample in samples)
    )
    if len(timestamps) != len(samples):
        raise ValueError("source_timestamps must match the motion target count")
    query = _finite_time(query_timestamp, name="query_timestamp")
    emitted = _finite_time(emitted_timestamp, name="emitted_timestamp")

    right_position = bisect_right(timestamps, query)
    fallback_reason: str | None = None
    if right_position == 0:
        left = right = samples[0]
        alpha = 0.0
        fallback_reason = "query_before_first_available_sample"
    elif right_position >= len(samples):
        left = right = samples[-1]
        alpha = 0.0
        if query > samples[-1].timestamp + _TIME_EPSILON_SEC:
            fallback_reason = "right_bracket_unavailable"
    else:
        left = samples[right_position - 1]
        if math.isclose(query, left.timestamp, rel_tol=0.0, abs_tol=_TIME_EPSILON_SEC):
            right = left
            alpha = 0.0
        else:
            right = samples[right_position]
            duration = right.timestamp - left.timestamp
            alpha = float(np.clip((query - left.timestamp) / duration, 0.0, 1.0))

    pose_by_side, qpos_by_side = _interpolate_targets(left, right, alpha)
    interpolated = left.source_index != right.source_index
    if causal:
        if interpolated:
            mode = (
                "causal_available_lookahead_interpolation"
                if emitted - query <= _TIME_EPSILON_SEC
                else "causal_delayed_interpolation"
            )
        elif fallback_reason is not None:
            mode = "causal_hold"
        else:
            mode = "causal_exact"
    else:
        mode = "offline_interpolation" if interpolated else "offline_exact"
    metadata = {
        "resampling_mode": mode,
        "causal": bool(causal),
        "lookahead_used": bool(interpolated and right.timestamp > query),
        "lookahead_available_at_emit": bool(
            not causal
            or float(right.available_timestamp) <= emitted + _TIME_EPSILON_SEC
        ),
        "left_source_available_timestamp": float(left.available_timestamp),
        "right_source_available_timestamp": float(right.available_timestamp),
        "query_timestamp": query,
        "emitted_timestamp": emitted if causal else None,
        "output_lag_sec": emitted - query if causal else None,
        "source_time_lookahead_sec": max(0.0, right.timestamp - query),
        "fallback_reason": fallback_reason,
        "pose_interpolation": (
            "xyz_linear+quaternion_shortest_path_slerp" if pose_by_side else None
        ),
        "qpos_interpolation": "linear" if qpos_by_side else None,
    }
    pose_action_velocity = _pose_action_interval_velocity(
        samples,
        query_timestamp=query,
        right_position=right_position,
        causal=causal,
    )
    if pose_action_velocity is not None:
        metadata["pose_action_interval_velocity"] = pose_action_velocity
    return ResampledMotionReference(
        sequence=sequence,
        timestamp=query,
        left_source_index=left.source_index,
        right_source_index=right.source_index,
        left_source_timestamp=left.timestamp,
        right_source_timestamp=right.timestamp,
        interpolation_alpha=alpha,
        pose_by_side=pose_by_side,
        qpos_by_side=qpos_by_side,
        metadata=metadata,
    )


class MotionTargetResampler:
    """Validate and freeze a source timeline for lazy timestamp evaluation."""

    def __init__(self, samples: Sequence[TimestampedMotionTarget]) -> None:
        self._samples = _normalize_targets(samples)
        self._source_timestamps = tuple(
            float(sample.timestamp) for sample in self._samples
        )

    @property
    def samples(self) -> tuple[TimestampedMotionTarget, ...]:
        return self._samples

    @property
    def source_timestamps(self) -> tuple[float, ...]:
        return self._source_timestamps

    def resample_at(
        self,
        query_timestamp: float,
        *,
        sequence: int,
        causal: bool = False,
        emitted_timestamp: float | None = None,
    ) -> ResampledMotionReference:
        query = _finite_time(query_timestamp, name="query_timestamp")
        emitted = (
            query
            if emitted_timestamp is None
            else _finite_time(emitted_timestamp, name="emitted_timestamp")
        )
        return _resample_at(
            self._samples,
            query_timestamp=query,
            sequence=sequence,
            causal=bool(causal),
            emitted_timestamp=emitted,
            source_timestamps=self._source_timestamps,
        )


def resample_motion_targets(
    samples: Sequence[TimestampedMotionTarget],
    *,
    reference_hz: float = DEFAULT_POSITION_REFERENCE_HZ,
    include_final: bool = True,
) -> list[ResampledMotionReference]:
    """Offline, timestamp-driven resampling with source-time lookahead.

    The time grid is derived from timestamps, not from an assumed integer rate
    ratio. The final source timestamp is appended when it does not land on the
    reference grid so jittered/non-integer input periods are preserved.
    """

    resampler = MotionTargetResampler(samples)
    targets = resampler.samples
    rate = validate_resample_rate(reference_hz, name="reference_hz")
    timestamps = build_uniform_time_grid(
        targets[0].timestamp,
        targets[-1].timestamp,
        hz=rate,
        include_final=include_final,
    )
    return [
        resampler.resample_at(
            timestamp,
            sequence=sequence,
            causal=False,
            emitted_timestamp=timestamp,
        )
        for sequence, timestamp in enumerate(timestamps)
    ]


class CausalMotionResampler:
    """Online resampler which never reads a target unavailable at emission.

    Calling ``sample_at(now, emitted_timestamp=now)`` normally has no
    source-time lookahead and therefore holds the latest input. A producer may
    also push a future logical target whose ``available_timestamp`` is already
    no later than ``now`` (for example an action chunk). Such a target is valid
    causal lookahead and permits zero-lag linear/SLERP interpolation. Otherwise
    callers can still obtain interpolation by querying a delayed timestamp.
    """

    def __init__(self, *, max_samples: int = 256) -> None:
        if (
            isinstance(max_samples, bool)
            or int(max_samples) != max_samples
            or int(max_samples) < 2
        ):
            raise ValueError("max_samples must be an integer >= 2")
        self.max_samples = int(max_samples)
        self._samples: list[TimestampedMotionTarget] = []
        self._signature: (
            tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]] | None
        ) = None
        self._next_sequence = 0
        self._lock = threading.RLock()

    def push(self, sample: TimestampedMotionTarget) -> None:
        if not isinstance(sample, TimestampedMotionTarget):
            raise TypeError("sample must be a TimestampedMotionTarget")
        signature = _target_signature(sample)
        with self._lock:
            if self._signature is not None and signature != self._signature:
                raise ValueError(
                    "all online motion targets must have matching modalities, "
                    "sides, and qpos shapes"
                )
            if self._samples and sample.timestamp <= self._samples[-1].timestamp:
                raise ValueError("online motion target timestamps must be strictly increasing")
            self._signature = signature
            self._samples.append(sample)
            if len(self._samples) > self.max_samples:
                del self._samples[: len(self._samples) - self.max_samples]

    def sample_at(
        self,
        query_timestamp: float,
        *,
        emitted_timestamp: float | None = None,
    ) -> ResampledMotionReference:
        query = _finite_time(query_timestamp, name="query_timestamp")
        emitted = query if emitted_timestamp is None else _finite_time(
            emitted_timestamp,
            name="emitted_timestamp",
        )
        if query > emitted + _TIME_EPSILON_SEC:
            raise ValueError(
                "query_timestamp cannot be later than emitted_timestamp in causal mode"
            )
        with self._lock:
            eligible = tuple(
                sample
                for sample in self._samples
                if float(sample.available_timestamp)
                <= emitted + _TIME_EPSILON_SEC
            )
            if not eligible:
                raise RuntimeError("no causal motion target is available at emitted_timestamp")
            sequence = self._next_sequence
            self._next_sequence += 1
        return _resample_at(
            eligible,
            query_timestamp=query,
            sequence=sequence,
            causal=True,
            emitted_timestamp=emitted,
        )

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._signature = None
            self._next_sequence = 0

    @property
    def buffered_samples(self) -> tuple[TimestampedMotionTarget, ...]:
        with self._lock:
            return tuple(self._samples)


__all__ = [
    "CausalMotionResampler",
    "MotionTargetResampler",
    "POSE_ACTION_INTERVAL_VELOCITY_SCHEMA",
    "ResampledMotionReference",
    "TimestampedMotionTarget",
    "build_uniform_time_grid",
    "interpolate_pose_shortest_path",
    "interpolate_qpos_by_side",
    "resample_motion_targets",
    "validate_resample_rate",
]
