"""Read-only RM65 free-drive observation source.

The concrete RM65 backend remains the owner of SDK access, hardware refresh,
FK, and observation-only normalization.  This adapter only validates one
fresh state cut and converts it into the backend-neutral
``RM65FreeDriveSample`` consumed by the B601 retargeter.  It never sends a
target, enables a motor, starts a stream, or falls back to an older sample.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from robot_motion.rebot_b601_rs.rm65_retarget import RM65FreeDriveSample
from robot_motion.transforms import parse_pose_matrix, validate_pose_matrix

RM65_FREE_DRIVE_SOURCE_CONTRACT_ID_V1 = (
    "control_station_rm65_observation_only_free_drive_v1"
)
RM65_FREE_DRIVE_SOURCE_CLOCK_DOMAIN_V1 = "local_wall_time"
RM65_FREE_DRIVE_SOURCE_REFERENCE_FRAME_V1 = "right_arm_base"
RM65_FREE_DRIVE_SOURCE_CONTROLLED_FRAME_V1 = "tool_tcp"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite, not bool.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _qpos(value: object) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("RM65 hardware_qpos must contain six finite values.") from exc
    if len(values) != 6 or any(not math.isfinite(item) for item in values):
        raise ValueError("RM65 hardware_qpos must contain six finite values.")
    return values


def _pose(value: object) -> tuple[tuple[float, ...], ...]:
    try:
        matrix = parse_pose_matrix(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("RM65 current_tcp_pose must be a supported pose.") from exc
    valid, reason = validate_pose_matrix(matrix, atol=1e-8)
    if not valid:
        raise ValueError(f"RM65 current_tcp_pose is invalid: {reason}")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class RM65FreeDriveSourceSpec:
    """Explicit state and freshness contract for one RM65 arm."""

    source_id: str = "rm65-right-observation-only-free-drive"
    component_id: str = "right_arm"
    reference_frame_id: str = RM65_FREE_DRIVE_SOURCE_REFERENCE_FRAME_V1
    controlled_frame_id: str = RM65_FREE_DRIVE_SOURCE_CONTROLLED_FRAME_V1
    clock_domain_id: str = RM65_FREE_DRIVE_SOURCE_CLOCK_DOMAIN_V1
    maximum_measurement_age_sec: float = 0.25
    maximum_receive_age_sec: float = 0.25
    maximum_future_skew_sec: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "component_id",
            "reference_frame_id",
            "controlled_frame_id",
            "clock_domain_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        ages = {
            "maximum_measurement_age_sec": self.maximum_measurement_age_sec,
            "maximum_receive_age_sec": self.maximum_receive_age_sec,
            "maximum_future_skew_sec": self.maximum_future_skew_sec,
        }
        for name, value in ages.items():
            normalized = _finite(value, name=name)
            if normalized < 0.0 or (
                name != "maximum_future_skew_sec" and normalized <= 0.0
            ):
                raise ValueError(f"{name} must be non-negative and finite.")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class RM65FreeDriveStep:
    """One admitted sample or an explicit current-tick rejection."""

    state: Mapping[str, object]
    sample: RM65FreeDriveSample | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping.")
        if (self.sample is None) == (self.rejection_reason is None):
            raise ValueError("A free-drive step requires exactly one sample or rejection.")
        if self.sample is not None and not isinstance(
            self.sample,
            RM65FreeDriveSample,
        ):
            raise TypeError("sample must be RM65FreeDriveSample.")
        if self.rejection_reason is not None and not str(self.rejection_reason).strip():
            raise ValueError("rejection_reason must be non-empty.")

    @property
    def admitted(self) -> bool:
        return self.sample is not None


@dataclass(frozen=True)
class RM65FreeDrivePollResult:
    """One source read produced by the optional observation worker."""

    step: RM65FreeDriveStep
    deadline_monotonic_sec: float
    started_monotonic_sec: float
    completed_monotonic_sec: float
    lateness_sec: float
    schedule_skipped: bool

    def __post_init__(self) -> None:
        if not isinstance(self.step, RM65FreeDriveStep):
            raise TypeError("step must be RM65FreeDriveStep.")
        for name in (
            "deadline_monotonic_sec",
            "started_monotonic_sec",
            "completed_monotonic_sec",
            "lateness_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.completed_monotonic_sec < self.started_monotonic_sec:
            raise ValueError("completed_monotonic_sec must not precede start.")
        if self.lateness_sec < 0.0:
            raise ValueError("lateness_sec must be non-negative.")


class RM65FreeDriveSourcePoller:
    """Poll one RM65 source without blocking the simulation/dispatch loop.

    The worker is deliberately a source producer, not a second control owner.
    It publishes every result in order; the bridge consumes those results from
    the single mailbox owner.  A bounded mailbox fails closed if a consumer
    stalls instead of silently dropping an invalid observation.
    """

    def __init__(
        self,
        source: RM65FreeDriveSource,
        *,
        poll_hz: float,
        clock: Callable[[], float] = time.monotonic,
        maximum_pending_results: int = 256,
    ) -> None:
        if not callable(getattr(source, "step", None)):
            raise TypeError("source must provide step().")
        normalized_hz = float(poll_hz)
        if not math.isfinite(normalized_hz) or normalized_hz <= 0.0:
            raise ValueError("poll_hz must be finite and positive.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if isinstance(maximum_pending_results, bool) or int(maximum_pending_results) <= 0:
            raise ValueError("maximum_pending_results must be positive.")
        self.source = source
        self.poll_hz = normalized_hz
        self._period_sec = 1.0 / normalized_hz
        self._clock = clock
        self._maximum_pending_results = int(maximum_pending_results)
        self._lock = threading.RLock()
        self._pending: deque[RM65FreeDrivePollResult] = deque()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._thread is not None and self._thread.is_alive())

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    raise RuntimeError("RM65 source poller is already running.")
                self._thread = None
            self._pending.clear()
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="rm65_free_drive_source",
            )
            self._thread = thread
            thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError(
                "RM65 source poller did not stop within 1.000s; "
                "backend close is intentionally deferred."
            )
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def take(self) -> RM65FreeDrivePollResult | None:
        with self._lock:
            if not self._pending:
                return None
            return self._pending.popleft()

    def _run(self) -> None:
        try:
            next_deadline = float(self._clock()) + self._period_sec
            while not self._stop_event.is_set():
                now = float(self._clock())
                if not math.isfinite(now):
                    raise ValueError("source poll clock returned a non-finite value")
                wait_sec = next_deadline - now
                if wait_sec > 0.0:
                    self._stop_event.wait(wait_sec)
                    continue
                started = float(self._clock())
                lateness = max(0.0, started - next_deadline)
                try:
                    step = self.source.step()
                except Exception as exc:  # noqa: BLE001 - publish failure to the owner.
                    step = RM65FreeDriveStep(
                        state={},
                        sample=None,
                        rejection_reason=(
                            f"source_worker_exception:{type(exc).__name__}:{exc}"
                        ),
                    )
                completed = float(self._clock())
                if not math.isfinite(completed):
                    raise ValueError("source poll clock returned a non-finite value")
                candidate = next_deadline + self._period_sec
                schedule_skipped = False
                if candidate <= completed + 1e-12:
                    candidate = completed + self._period_sec
                    schedule_skipped = True
                result = RM65FreeDrivePollResult(
                    step=step,
                    deadline_monotonic_sec=next_deadline,
                    started_monotonic_sec=started,
                    completed_monotonic_sec=completed,
                    lateness_sec=lateness,
                    schedule_skipped=schedule_skipped,
                )
                with self._lock:
                    if len(self._pending) >= self._maximum_pending_results:
                        self._pending.clear()
                        overflow_step = RM65FreeDriveStep(
                            state={},
                            sample=None,
                            rejection_reason="source_worker_mailbox_overflow",
                        )
                        result = RM65FreeDrivePollResult(
                            step=overflow_step,
                            deadline_monotonic_sec=next_deadline,
                            started_monotonic_sec=started,
                            completed_monotonic_sec=completed,
                            lateness_sec=lateness,
                            schedule_skipped=True,
                        )
                    self._pending.append(result)
                next_deadline = candidate
        except Exception as exc:  # noqa: BLE001 - fail closed at the bridge owner.
            now = float(self._clock())
            if not math.isfinite(now):
                now = time.monotonic()
            failure = RM65FreeDriveStep(
                state={},
                sample=None,
                rejection_reason=f"source_worker_terminal:{type(exc).__name__}:{exc}",
            )
            result = RM65FreeDrivePollResult(
                step=failure,
                deadline_monotonic_sec=now,
                started_monotonic_sec=now,
                completed_monotonic_sec=now,
                lateness_sec=0.0,
                schedule_skipped=True,
            )
            with self._lock:
                self._pending.clear()
                self._pending.append(result)


@runtime_checkable
class _RM65HardwareObservationBackend(Protocol):
    def refresh_state_from_hardware(
        self,
        *,
        include_derived_pose: bool = True,
        include_timing_percentiles: bool = True,
        include_pose_source_attestation: bool | None = None,
    ) -> Mapping[str, object]: ...


class RM65FreeDriveSource:
    """Thin observation-only adapter over the existing RM65 backend."""

    def __init__(
        self,
        backend: _RM65HardwareObservationBackend,
        spec: RM65FreeDriveSourceSpec | None = None,
        *,
        wall_clock: Any = time.time,
        monotonic_clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(backend, _RM65HardwareObservationBackend):
            raise TypeError("backend must provide refresh_state_from_hardware.")
        if not callable(wall_clock) or not callable(monotonic_clock):
            raise TypeError("wall_clock and monotonic_clock must be callable.")
        self.backend = backend
        self.spec = spec if spec is not None else RM65FreeDriveSourceSpec()
        if not isinstance(self.spec, RM65FreeDriveSourceSpec):
            raise TypeError("spec must be RM65FreeDriveSourceSpec.")
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self._last_measurement_sequence: int | None = None

    @property
    def last_measurement_sequence(self) -> int | None:
        return self._last_measurement_sequence

    def step(self) -> RM65FreeDriveStep:
        """Refresh once and admit only the current complete hardware cut."""

        config = getattr(self.backend, "config", None)
        if config is not None and getattr(config, "observation_only", None) is not True:
            return RM65FreeDriveStep(
                state={},
                sample=None,
                rejection_reason="backend_not_observation_only",
            )
        try:
            state = self.backend.refresh_state_from_hardware(
                include_derived_pose=True,
                include_timing_percentiles=False,
                include_pose_source_attestation=False,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return RM65FreeDriveStep(
                state={},
                sample=None,
                rejection_reason=(
                    f"backend_refresh_rejected:{type(exc).__name__}:{exc}"
                ),
            )
        if not isinstance(state, Mapping):
            return RM65FreeDriveStep(
                state={},
                sample=None,
                rejection_reason="backend_refresh_returned_non_mapping",
            )
        try:
            self._validate_state_flags(state)
            sequence = self._positive_sequence(
                state.get("hardware_measurement_sequence"),
                name="hardware_measurement_sequence",
            )
            if (
                self._last_measurement_sequence is not None
                and sequence <= self._last_measurement_sequence
            ):
                raise ValueError(
                    "hardware measurement sequence is not advancing: "
                    f"{sequence} <= {self._last_measurement_sequence}"
                )
            source_timestamp = _finite(
                state.get("hardware_measurement_timestamp"),
                name="hardware_measurement_timestamp",
            )
            receive_timestamp = _finite(
                state.get("hardware_read_time"),
                name="hardware_read_time",
            )
            receive_monotonic = _finite(
                state.get("hardware_read_timestamp_monotonic"),
                name="hardware_read_timestamp_monotonic",
            )
            self._validate_freshness(
                source_timestamp,
                receive_timestamp,
                receive_monotonic,
            )
            qpos = _qpos(state.get("hardware_qpos"))
            pose = _pose(state.get("current_tcp_pose"))
            sample = RM65FreeDriveSample(
                sequence=sequence,
                source_timestamp_sec=source_timestamp,
                receive_timestamp_sec=receive_timestamp,
                receive_timestamp_monotonic_sec=receive_monotonic,
                qpos_rad=qpos,
                pose_source_reference_source_controlled=pose,
                source_id=self.spec.source_id,
                source_clock_domain_id=self.spec.clock_domain_id,
                source_reference_frame_id=self.spec.reference_frame_id,
                source_controlled_frame_id=self.spec.controlled_frame_id,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return RM65FreeDriveStep(
                state=state,
                sample=None,
                rejection_reason=f"state_rejected:{type(exc).__name__}:{exc}",
            )
        self._last_measurement_sequence = sequence
        return RM65FreeDriveStep(state=state, sample=sample, rejection_reason=None)

    def _validate_state_flags(self, state: Mapping[str, object]) -> None:
        if state.get("observation_only") is not True:
            raise ValueError("RM65 backend state is not observation-only")
        if state.get("arm_state_source") != "hardware_observation_only":
            raise ValueError("RM65 state source is not hardware observation-only")
        if state.get("hardware_measurement_valid") is not True:
            raise ValueError("RM65 hardware measurement is not valid")
        if state.get("hardware_read_success") is not True:
            raise ValueError("RM65 hardware read did not succeed")
        if (
            state.get("hardware_measurement_clock_domain_id")
            != self.spec.clock_domain_id
        ):
            raise ValueError("RM65 hardware measurement clock domain is incompatible")
        background = state.get("background_streaming")
        if isinstance(background, Mapping) and background.get("enabled") is True:
            raise ValueError("RM65 observation-only backend has background streaming enabled")

    @staticmethod
    def _positive_sequence(value: object, *, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a positive integer.")
        sequence = int(value)
        if sequence != value or sequence <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return sequence

    def _validate_freshness(
        self,
        source_timestamp: float,
        receive_timestamp: float,
        receive_monotonic: float,
    ) -> None:
        now_wall = _finite(self.wall_clock(), name="wall_clock")
        now_monotonic = _finite(self.monotonic_clock(), name="monotonic_clock")
        source_age = now_wall - source_timestamp
        receive_age = now_monotonic - receive_monotonic
        if source_timestamp > receive_timestamp + self.spec.maximum_future_skew_sec:
            raise ValueError("hardware measurement timestamp is newer than its read")
        if source_age < -self.spec.maximum_future_skew_sec:
            raise ValueError(f"hardware measurement is from the future: age={source_age:.6f}s")
        if receive_age < -self.spec.maximum_future_skew_sec:
            raise ValueError(f"hardware read is from the future: age={receive_age:.6f}s")
        if source_age > self.spec.maximum_measurement_age_sec:
            raise ValueError(f"hardware measurement is stale: age={source_age:.6f}s")
        if receive_age > self.spec.maximum_receive_age_sec:
            raise ValueError(f"hardware read is stale: age={receive_age:.6f}s")


__all__ = [
    "RM65_FREE_DRIVE_SOURCE_CLOCK_DOMAIN_V1",
    "RM65_FREE_DRIVE_SOURCE_CONTRACT_ID_V1",
    "RM65_FREE_DRIVE_SOURCE_CONTROLLED_FRAME_V1",
    "RM65_FREE_DRIVE_SOURCE_REFERENCE_FRAME_V1",
    "RM65FreeDrivePollResult",
    "RM65FreeDriveSource",
    "RM65FreeDriveSourcePoller",
    "RM65FreeDriveSourceSpec",
    "RM65FreeDriveStep",
]
