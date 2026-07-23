"""Transport-neutral audit contracts for the high-rate position pipeline.

The contracts in this module deliberately do not know about RM65, CANFD, or
the runtime's file format.  A simulator and a hardware backend can therefore
emit the same reference/command/timing events, while a runtime-specific sink
decides whether to aggregate or persist them.
"""

from __future__ import annotations

import copy
import math
import threading
from collections import Counter, deque
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import numpy as np

from .contracts import PositionReferenceFrame


POSITION_AUDIT_SCHEMA_VERSION = "position_audit.v1"
POSITION_TRANSPORT_TARGET_SEQUENCE_KEY = "position_transport_target_sequence"
POSITION_TRANSPORT_TARGET_SEQUENCES_KEY = "position_transport_target_sequences"


class PositionAuditMode(str, Enum):
    """Amount of position-control audit data retained by a sink."""

    OFF = "off"
    SUMMARY = "summary"
    FULL = "full"


def normalize_position_audit_mode(value: PositionAuditMode | str) -> PositionAuditMode:
    if isinstance(value, PositionAuditMode):
        return value
    normalized = str(value).strip().lower()
    try:
        return PositionAuditMode(normalized)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in PositionAuditMode)
        raise ValueError(f"Unsupported position audit mode {value!r}; expected {choices}.") from exc


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _optional_finite(value: float | None, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _optional_non_negative_int(value: int | None, *, name: str) -> int | None:
    return None if value is None else _non_negative_int(value, name=name)


def _vector(value: Sequence[float] | np.ndarray | None, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return tuple(float(item) for item in array)


def _matrix_rows(
    value: Sequence[float] | np.ndarray | None,
    *,
    name: str,
) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a vector or matrix.")
    return tuple(tuple(float(item) for item in row) for row in array)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(
        {} if value is None else {str(key): _freeze_value(item) for key, item in value.items()}
    )


@dataclass(frozen=True)
class PositionReferenceSideAudit:
    side: str
    q_ik_raw: Sequence[float] | np.ndarray | None
    q_ref: Sequence[float] | np.ndarray
    pose_ref: Sequence[float] | np.ndarray
    valid: bool
    hold: bool
    modification_reasons: Sequence[str] = ()
    frame_metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = str(self.side).strip()
        if not side:
            raise ValueError("side must not be empty.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "q_ik_raw", _vector(self.q_ik_raw, name=f"{side}.q_ik_raw"))
        object.__setattr__(self, "q_ref", _vector(self.q_ref, name=f"{side}.q_ref"))
        object.__setattr__(self, "pose_ref", _matrix_rows(self.pose_ref, name=f"{side}.pose_ref"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "hold", bool(self.hold))
        object.__setattr__(
            self,
            "modification_reasons",
            tuple(str(reason) for reason in self.modification_reasons),
        )
        object.__setattr__(self, "frame_metadata", _freeze_mapping(self.frame_metadata))
        object.__setattr__(self, "profile_metadata", _freeze_mapping(self.profile_metadata))


@dataclass(frozen=True)
class PositionReferenceAuditEvent:
    """One atomic high-rate reference publication.

    The event name is frequency-neutral; the configured and measured rates are
    carried in execution metadata rather than encoded in the schema name.
    """

    epoch: int
    reference_sequence: int
    source: str
    source_wall_time: float
    generated_wall_time: float
    published_wall_time: float
    generated_monotonic_time: float
    published_monotonic_time: float
    references: Mapping[str, PositionReferenceSideAudit]
    transport_target_sequences: Mapping[str, int] = field(default_factory=dict)
    frame_metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=POSITION_AUDIT_SCHEMA_VERSION, init=False)
    event_type: str = field(default="position_reference", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _non_negative_int(self.epoch, name="epoch"))
        object.__setattr__(
            self,
            "reference_sequence",
            _non_negative_int(self.reference_sequence, name="reference_sequence"),
        )
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must not be empty.")
        object.__setattr__(self, "source", source)
        for name in (
            "source_wall_time",
            "generated_wall_time",
            "published_wall_time",
            "generated_monotonic_time",
            "published_monotonic_time",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.published_monotonic_time < self.generated_monotonic_time:
            raise ValueError(
                "published_monotonic_time must be >= generated_monotonic_time."
            )
        if not self.references:
            raise ValueError("references must not be empty.")
        detached: dict[str, PositionReferenceSideAudit] = {}
        for raw_side, reference in self.references.items():
            side = str(raw_side).strip()
            if not isinstance(reference, PositionReferenceSideAudit):
                raise TypeError(f"references[{side!r}] must be PositionReferenceSideAudit.")
            if reference.side != side:
                raise ValueError(f"reference key {side!r} does not match side {reference.side!r}.")
            detached[side] = reference
        object.__setattr__(self, "references", MappingProxyType(detached))
        target_sequences = {
            str(side): _non_negative_int(sequence, name=f"transport_target_sequences[{side!r}]")
            for side, sequence in self.transport_target_sequences.items()
        }
        object.__setattr__(self, "transport_target_sequences", MappingProxyType(target_sequences))
        object.__setattr__(self, "frame_metadata", _freeze_mapping(self.frame_metadata))
        object.__setattr__(self, "profile_metadata", _freeze_mapping(self.profile_metadata))


@dataclass(frozen=True)
class PositionCommandAuditEvent:
    """One command-rate scheduler tick and its optional transport attempt/result.

    ``position_command_150`` remains the compatibility event name even though
    the production command clock is now 120 Hz; the actual rate is carried by
    profile metadata.
    """

    side: str
    tick_sequence: int
    timestamp_wall: float
    timestamp_monotonic: float
    scheduled_monotonic_time: float
    wakeup_monotonic_time: float
    deadline_monotonic_time: float
    wakeup_lag_sec: float
    raw_wakeup_lag_sec: float
    skipped_tick_count: int
    deadline_missed: bool
    planner_sample_monotonic_time: float | None = None
    planner_lock_delay_sec: float | None = None
    wakeup_gap_sec: float | None = None
    target_active: bool = False
    target_sequence: int | None = None
    target_source: str | None = None
    target_timestamp_wall: float | None = None
    target_timestamp_monotonic: float | None = None
    target_age_sec: float | None = None
    reference_epoch: int | None = None
    reference_sequence: int | None = None
    segment_start_qpos: Sequence[float] | np.ndarray | None = None
    segment_target_qpos: Sequence[float] | np.ndarray | None = None
    segment_start_monotonic_time: float | None = None
    segment_duration_sec: float | None = None
    interpolation_alpha: float | None = None
    q_cmd: Sequence[float] | np.ndarray | None = None
    sdk_attempted: bool = False
    sdk_attempt_sequence: int | None = None
    sdk_attempt_wall_time: float | None = None
    sdk_attempt_monotonic_time: float | None = None
    sdk_result_monotonic_time: float | None = None
    sdk_send_gap_sec: float | None = None
    sdk_call_ms: float | None = None
    sdk_result_code: int | None = None
    sdk_result_text: str | None = None
    sdk_accepted: bool | None = None
    sdk_accepted_sequence: int | None = None
    stale_target_expired: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=POSITION_AUDIT_SCHEMA_VERSION, init=False)
    event_type: str = field(default="position_command_150", init=False)

    def __post_init__(self) -> None:
        side = str(self.side).strip()
        if not side:
            raise ValueError("side must not be empty.")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "tick_sequence",
            _non_negative_int(self.tick_sequence, name="tick_sequence"),
        )
        for name in (
            "timestamp_wall",
            "timestamp_monotonic",
            "scheduled_monotonic_time",
            "wakeup_monotonic_time",
            "deadline_monotonic_time",
            "wakeup_lag_sec",
            "raw_wakeup_lag_sec",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.wakeup_lag_sec < 0.0 or self.raw_wakeup_lag_sec < 0.0:
            raise ValueError("wakeup lag values must be non-negative.")
        object.__setattr__(
            self,
            "skipped_tick_count",
            _non_negative_int(self.skipped_tick_count, name="skipped_tick_count"),
        )
        optional_finite_fields = (
            "planner_sample_monotonic_time",
            "planner_lock_delay_sec",
            "wakeup_gap_sec",
            "target_timestamp_wall",
            "target_timestamp_monotonic",
            "target_age_sec",
            "segment_start_monotonic_time",
            "segment_duration_sec",
            "interpolation_alpha",
            "sdk_attempt_wall_time",
            "sdk_attempt_monotonic_time",
            "sdk_result_monotonic_time",
            "sdk_send_gap_sec",
            "sdk_call_ms",
        )
        for name in optional_finite_fields:
            object.__setattr__(
                self,
                name,
                _optional_finite(getattr(self, name), name=name),
            )
        for name in (
            "planner_lock_delay_sec",
            "wakeup_gap_sec",
            "target_age_sec",
            "segment_duration_sec",
            "sdk_send_gap_sec",
            "sdk_call_ms",
        ):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if (
            self.planner_sample_monotonic_time is not None
            and self.planner_sample_monotonic_time < self.wakeup_monotonic_time
        ):
            raise ValueError(
                "planner_sample_monotonic_time must not precede wakeup_monotonic_time."
            )
        if self.deadline_monotonic_time <= self.scheduled_monotonic_time:
            raise ValueError(
                "deadline_monotonic_time must be greater than scheduled_monotonic_time."
            )
        if self.interpolation_alpha is not None and not 0.0 <= self.interpolation_alpha <= 1.0:
            raise ValueError("interpolation_alpha must be in [0, 1].")
        for name in (
            "target_sequence",
            "reference_epoch",
            "reference_sequence",
            "sdk_attempt_sequence",
            "sdk_accepted_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _optional_non_negative_int(getattr(self, name), name=name),
            )
        if self.sdk_result_code is not None:
            object.__setattr__(self, "sdk_result_code", int(self.sdk_result_code))
        object.__setattr__(
            self,
            "target_source",
            None if self.target_source is None else str(self.target_source),
        )
        object.__setattr__(
            self,
            "sdk_result_text",
            None if self.sdk_result_text is None else str(self.sdk_result_text),
        )
        for name in ("segment_start_qpos", "segment_target_qpos", "q_cmd"):
            object.__setattr__(self, name, _vector(getattr(self, name), name=name))
        object.__setattr__(self, "target_active", bool(self.target_active))
        object.__setattr__(self, "deadline_missed", bool(self.deadline_missed))
        object.__setattr__(self, "sdk_attempted", bool(self.sdk_attempted))
        object.__setattr__(self, "stale_target_expired", bool(self.stale_target_expired))
        if self.sdk_accepted is not None:
            object.__setattr__(self, "sdk_accepted", bool(self.sdk_accepted))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class PositionTimingAuditEvent:
    """Duration/deadline observation for any pipeline stage."""

    stage: str
    elapsed_ms: float
    timestamp_wall: float
    timestamp_monotonic: float
    side: str | None = None
    epoch: int | None = None
    reference_sequence: int | None = None
    target_sequence: int | None = None
    tick_sequence: int | None = None
    deadline_ms: float | None = None
    deadline_missed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=POSITION_AUDIT_SCHEMA_VERSION, init=False)
    event_type: str = field(default="position_timing", init=False)

    def __post_init__(self) -> None:
        stage = str(self.stage).strip()
        if not stage:
            raise ValueError("stage must not be empty.")
        object.__setattr__(self, "stage", stage)
        for name in ("elapsed_ms", "timestamp_wall", "timestamp_monotonic"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "deadline_ms",
            _optional_finite(self.deadline_ms, name="deadline_ms"),
        )
        for name in ("epoch", "reference_sequence", "target_sequence", "tick_sequence"):
            object.__setattr__(
                self,
                name,
                _optional_non_negative_int(getattr(self, name), name=name),
            )
        object.__setattr__(self, "side", None if self.side is None else str(self.side))
        object.__setattr__(self, "deadline_missed", bool(self.deadline_missed))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


PositionAuditEvent: TypeAlias = (
    PositionReferenceAuditEvent | PositionCommandAuditEvent | PositionTimingAuditEvent
)
PositionAuditSink: TypeAlias = Callable[[PositionAuditEvent], None]


def position_reference_audit_event(frame: PositionReferenceFrame) -> PositionReferenceAuditEvent:
    """Detach an immutable timeline frame into a JSON-friendly audit event."""

    raw_target_sequences = frame.frame_metadata.get(POSITION_TRANSPORT_TARGET_SEQUENCES_KEY, {})
    target_sequences = (
        raw_target_sequences if isinstance(raw_target_sequences, Mapping) else {}
    )
    references = {
        side: PositionReferenceSideAudit(
            side=side,
            q_ik_raw=reference.q_ik_raw,
            q_ref=reference.q_ref,
            pose_ref=reference.pose_ref,
            valid=reference.valid,
            hold=reference.hold,
            modification_reasons=reference.modification_reasons,
            frame_metadata=reference.frame_metadata,
            profile_metadata=reference.profile_metadata,
        )
        for side, reference in frame.references.items()
    }
    return PositionReferenceAuditEvent(
        epoch=frame.epoch,
        reference_sequence=frame.sequence,
        source=frame.source,
        source_wall_time=frame.source_wall_time,
        generated_wall_time=frame.generated_wall_time,
        published_wall_time=frame.published_wall_time,
        generated_monotonic_time=frame.generated_monotonic_time,
        published_monotonic_time=frame.published_monotonic_time,
        references=references,
        transport_target_sequences=target_sequences,
        frame_metadata=frame.frame_metadata,
        profile_metadata=frame.profile_metadata,
    )


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def position_audit_event_to_dict(event: PositionAuditEvent) -> dict[str, Any]:
    if not isinstance(
        event,
        (PositionReferenceAuditEvent, PositionCommandAuditEvent, PositionTimingAuditEvent),
    ):
        raise TypeError(f"Unsupported position audit event type: {type(event).__name__}")
    return _json_value(event)


@dataclass(frozen=True)
class PercentileStatsSnapshot:
    """Exact percentiles over the retained sample window."""

    count: int
    sample_count: int
    dropped_sample_count: int
    p50: float | None
    p95: float | None
    p99: float | None
    max: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "count": int(self.count),
            "sample_count": int(self.sample_count),
            "dropped_sample_count": int(self.dropped_sample_count),
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        }


class PercentileStats:
    """Thread-safe, bounded percentile accumulator.

    Percentiles are exact for the retained window.  ``count`` remains a
    lifetime count, and ``dropped_sample_count`` makes window truncation
    explicit instead of silently presenting an approximation as all-time data.
    """

    def __init__(self, *, max_samples: int = 100_000) -> None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive.")
        self.max_samples = int(max_samples)
        self._samples: deque[float] = deque(maxlen=self.max_samples)
        self._count = 0
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        sample = _finite(value, name="percentile sample")
        with self._lock:
            self._samples.append(sample)
            self._count += 1

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._count = 0

    def snapshot(self) -> PercentileStatsSnapshot:
        with self._lock:
            # Keep the producer-side critical section to a bounded copy.  The
            # O(n log n) sort happens after releasing the lock so the command-rate
            # recorder is never serialized behind percentile calculation.
            values = list(self._samples)
            count = int(self._count)
        values.sort()
        sample_count = len(values)
        return PercentileStatsSnapshot(
            count=count,
            sample_count=sample_count,
            dropped_sample_count=max(0, count - sample_count),
            p50=_linear_percentile(values, 0.50),
            p95=_linear_percentile(values, 0.95),
            p99=_linear_percentile(values, 0.99),
            max=None if not values else float(values[-1]),
        )


def _linear_percentile(sorted_values: Sequence[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * float(quantile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


class PositionAuditStatistics:
    """Named counters and percentile metrics shared by runtime audit sinks."""

    def __init__(self, *, max_samples_per_metric: int = 100_000) -> None:
        if int(max_samples_per_metric) <= 0:
            raise ValueError("max_samples_per_metric must be positive.")
        self._max_samples_per_metric = int(max_samples_per_metric)
        self._counters: Counter[str] = Counter()
        self._metrics: dict[str, PercentileStats] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, value: int = 1) -> None:
        normalized = int(value)
        with self._lock:
            self._counters[str(name)] += normalized

    def observe(self, name: str, value: float | None) -> None:
        if value is None:
            return
        key = str(name)
        with self._lock:
            metric = self._metrics.get(key)
            if metric is None:
                metric = PercentileStats(max_samples=self._max_samples_per_metric)
                self._metrics[key] = metric
        metric.record(value)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._metrics.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            metrics = dict(self._metrics)
        return {
            "counters": {key: int(value) for key, value in counters.items()},
            "percentiles": {
                key: metric.snapshot().as_dict() for key, metric in metrics.items()
            },
        }


class PositionAuditFanout:
    """Copy-on-write audit sink multiplexer with per-sink fault isolation.

    The reference/command-rate call path takes no fanout lock: add/remove replaces an
    immutable sink tuple, while emission snapshots that tuple and invokes each
    sink independently.  Sinks are expected to honor ``PositionAuditSink``'s
    non-blocking contract (for example, ``PositionDebugRecorder`` only performs
    ``put_nowait``); the fanout never serializes producers on its own lock.
    """

    def __init__(self, sinks: Sequence[PositionAuditSink] = ()) -> None:
        self._mutation_lock = threading.Lock()
        self._sinks: tuple[PositionAuditSink, ...] = ()
        self._error_lock = threading.Lock()
        self._error_count = 0
        self._last_error: str | None = None
        for sink in sinks:
            self.add_sink(sink)

    @property
    def sink_count(self) -> int:
        return len(self._sinks)

    @property
    def error_count(self) -> int:
        with self._error_lock:
            return int(self._error_count)

    @property
    def last_error(self) -> str | None:
        with self._error_lock:
            return self._last_error

    def sinks(self) -> tuple[PositionAuditSink, ...]:
        return self._sinks

    def add_sink(self, sink: PositionAuditSink) -> bool:
        if not callable(sink):
            raise TypeError("Position audit sink must be callable.")
        with self._mutation_lock:
            if any(existing is sink for existing in self._sinks):
                return False
            self._sinks = (*self._sinks, sink)
            return True

    add_recorder = add_sink

    def remove_sink(self, sink: PositionAuditSink) -> bool:
        with self._mutation_lock:
            remaining = tuple(existing for existing in self._sinks if existing is not sink)
            if len(remaining) == len(self._sinks):
                return False
            self._sinks = remaining
            return True

    remove_recorder = remove_sink

    def clear(self) -> None:
        with self._mutation_lock:
            self._sinks = ()

    def __call__(self, event: PositionAuditEvent) -> None:
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 - one recorder must not affect another.
                with self._error_lock:
                    self._error_count += 1
                    self._last_error = (
                        f"{type(sink).__name__}: {type(exc).__name__}: {exc}"
                    )

    def snapshot(self) -> dict[str, Any]:
        return {
            "sink_count": self.sink_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


PositionAuditMultiplexer = PositionAuditFanout


@dataclass(frozen=True)
class PeriodicTickDecision:
    """Normalized scheduler timing for one executed periodic tick."""

    original_scheduled_monotonic_time: float
    scheduled_monotonic_time: float
    wakeup_monotonic_time: float
    deadline_monotonic_time: float
    period_sec: float
    raw_wakeup_lag_sec: float
    wakeup_lag_sec: float
    skipped_tick_count: int

    def deadline_missed(self, finished_monotonic_time: float) -> bool:
        return _finite(finished_monotonic_time, name="finished_monotonic_time") > (
            self.deadline_monotonic_time
        )


def resolve_periodic_tick(
    *,
    scheduled_monotonic_time: float,
    wakeup_monotonic_time: float,
    period_sec: float,
) -> PeriodicTickDecision:
    """Account for every elapsed deadline without scheduling a catch-up burst."""

    scheduled = _finite(scheduled_monotonic_time, name="scheduled_monotonic_time")
    wakeup = _finite(wakeup_monotonic_time, name="wakeup_monotonic_time")
    period = _finite(period_sec, name="period_sec")
    if period <= 0.0:
        raise ValueError("period_sec must be positive.")
    raw_lag = max(0.0, wakeup - scheduled)
    skipped = int(math.floor(raw_lag / period))
    effective_scheduled = scheduled + skipped * period
    normalized_lag = max(0.0, wakeup - effective_scheduled)
    return PeriodicTickDecision(
        original_scheduled_monotonic_time=scheduled,
        scheduled_monotonic_time=effective_scheduled,
        wakeup_monotonic_time=wakeup,
        deadline_monotonic_time=effective_scheduled + period,
        period_sec=period,
        raw_wakeup_lag_sec=raw_lag,
        wakeup_lag_sec=normalized_lag,
        skipped_tick_count=skipped,
    )


__all__ = [
    "POSITION_AUDIT_SCHEMA_VERSION",
    "POSITION_TRANSPORT_TARGET_SEQUENCE_KEY",
    "POSITION_TRANSPORT_TARGET_SEQUENCES_KEY",
    "PercentileStats",
    "PercentileStatsSnapshot",
    "PeriodicTickDecision",
    "PositionAuditEvent",
    "PositionAuditFanout",
    "PositionAuditMode",
    "PositionAuditMultiplexer",
    "PositionAuditSink",
    "PositionAuditStatistics",
    "PositionCommandAuditEvent",
    "PositionReferenceAuditEvent",
    "PositionReferenceSideAudit",
    "PositionTimingAuditEvent",
    "normalize_position_audit_mode",
    "position_audit_event_to_dict",
    "position_reference_audit_event",
    "resolve_periodic_tick",
]
