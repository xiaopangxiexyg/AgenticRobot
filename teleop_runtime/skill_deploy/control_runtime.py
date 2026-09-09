"""Single-owner control cycle for an exclusive real Skill deployment.

The runtime is deliberately independent of RealTeleop. Concrete deployment
assembly injects already-started observation, safety/admission, dispatch, and
HOLD ports from the lower real-control stack. The Agent Gateway never receives
those ports and cannot bypass this cycle.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.command_override import TeleopCommandOverride

from .motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortSnapshot,
)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


class RealSkillControlLifecycle(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    HOLD = "hold"
    FAULT = "fault"
    CLOSED = "closed"


class RealSkillMotionObservationUnavailable(RuntimeError):
    """A valid lower measurement could not form one fresh coherent cut.

    This type is reserved for transient availability loss.  Invalid,
    non-authoritative, regressed, or permanently failed measurements remain
    ordinary runtime faults and must never be treated as skippable.
    """


class RealSkillMotionTargetSynchronizer(Protocol):
    """Lower numerical port that aligns the coupled target solver."""

    def __call__(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
        /,
    ) -> None: ...


class RealSkillMeasuredQposObserver(Protocol):
    """Passive projection invoked by the sole lower control-cycle owner."""

    def __call__(
        self,
        cut: RealSkillMotionObservationCut,
        *,
        derived_available_at: float,
    ) -> None: ...


@dataclass(frozen=True)
class RealSkillMotionObservationCut:
    """One already-read lower observation and its exact dual-arm qpos cut."""

    observation: TeleopObservation
    qpos_by_side: Mapping[str, np.ndarray]
    measured_at: float
    available_at: float
    evidence_refs: tuple[str, ...]
    source_sequence_by_side: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TeleopObservation):
            raise TypeError("observation must be TeleopObservation.")
        measured_at = _finite(self.measured_at, field_name="measured_at")
        available_at = _finite(self.available_at, field_name="available_at")
        if available_at < measured_at:
            raise ValueError("available_at cannot precede measured_at.")
        qpos = {
            str(side): np.asarray(value, dtype=float).reshape(-1).copy()
            for side, value in dict(self.qpos_by_side).items()
        }
        if set(qpos) != {"left", "right"}:
            raise ValueError("qpos_by_side must contain exactly left and right.")
        if any(
            value.shape != (6,) or not np.all(np.isfinite(value))
            for value in qpos.values()
        ):
            raise ValueError("each qpos_by_side value must contain six finite values.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("an observation cut requires lower evidence.")
        source_sequences = dict(self.source_sequence_by_side)
        if source_sequences and set(source_sequences) != {"left", "right"}:
            raise ValueError(
                "source_sequence_by_side must be empty or contain left and right."
            )
        normalized_sequences: dict[str, int] = {}
        for side, value in source_sequences.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"source_sequence_by_side[{side!r}] must be a positive integer."
                )
            normalized_sequences[str(side)] = value
        object.__setattr__(self, "qpos_by_side", MappingProxyType(qpos))
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "source_sequence_by_side",
            MappingProxyType(normalized_sequences),
        )


@dataclass(frozen=True)
class RealSkillMotionDispatchReceipt:
    """Lower dispatch result; only a successful receipt advances evidence."""

    succeeded: bool
    evidence_refs: tuple[str, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if self.succeeded and not refs:
            raise ValueError("a successful dispatch requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


@dataclass(frozen=True)
class RealSkillMotionHoldReceipt:
    """Lower-owned HOLD result retained by the exclusive deployment owner."""

    succeeded: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if self.succeeded and not refs:
            raise ValueError("a successful HOLD requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class RealSkillMotionEmergencyStopReceipt:
    """Lower-owned, per-side emergency-stop acknowledgement evidence."""

    succeeded: bool
    evidence_refs: tuple[str, ...]
    status_by_side: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        statuses = {
            _required(side, field_name="status side"): _required(
                status,
                field_name="emergency-stop status",
            )
            for side, status in dict(self.status_by_side).items()
        }
        if set(statuses) != {"left", "right"}:
            raise ValueError(
                "emergency-stop status_by_side must contain left and right."
            )
        if self.succeeded and (
            not refs or set(statuses.values()) != {"confirmed"}
        ):
            raise ValueError(
                "successful emergency stop requires two confirmed lower sides."
            )
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "status_by_side", MappingProxyType(statuses))


@dataclass(frozen=True)
class RealSkillControlStep:
    lifecycle: RealSkillControlLifecycle
    port_snapshot: RealAgentMotionPortSnapshot
    command_present: bool
    dispatch_attempted: bool
    dispatch_succeeded: bool
    dispatch_evidence_refs: tuple[str, ...]


@dataclass
class RealSkillControlTraceSample:
    """Ephemeral control-cut view copied by an optional bounded trace sink.

    This value never owns control state.  A sink must copy the fields it needs
    synchronously and must not retain or mutate the command/cut objects.
    """

    cycle_sequence: int
    cycle_started_monotonic: float
    control_time: float | None = None
    observation_completed_monotonic: float | None = None
    synchronization_completed_monotonic: float | None = None
    owner_requests_completed_monotonic: float | None = None
    ik_completed_monotonic: float | None = None
    admission_completed_monotonic: float | None = None
    dispatch_ready_at: float | None = None
    dispatch_completed_monotonic: float | None = None
    cycle_finished_monotonic: float | None = None
    cut: RealSkillMotionObservationCut | None = None
    measured_pose_cut: Any | None = None
    command: TeleopCommand | None = None
    dispatch_receipt: RealSkillMotionDispatchReceipt | None = None
    port_snapshot: RealAgentMotionPortSnapshot | None = None
    command_present: bool = False
    admitted: bool = False
    expired_reference: bool = False
    dispatch_attempted: bool = False
    dispatch_succeeded: bool = False
    outcome: str = "started"
    error: str | None = None


class RealSkillControlTraceSink(Protocol):
    """Non-authoritative best-effort consumer of one completed control cut."""

    def __call__(self, sample: RealSkillControlTraceSample, /) -> object: ...


class ExclusiveRealSkillControlRuntime:
    """One control owner for a Skill-only real deployment.

    The injected callbacks are lower application ports. They are not exposed
    through the Gateway, and this runtime never accepts a second command source.
    """

    def __init__(
        self,
        *,
        motion_port: RealAgentMotionPort,
        observation_provider: Callable[[], RealSkillMotionObservationCut],
        synchronize_target_builder: RealSkillMotionTargetSynchronizer,
        measured_qpos_observer: RealSkillMeasuredQposObserver | None = None,
        command_admission: Callable[[TeleopCommand, TeleopObservation, float], bool],
        command_dispatch: Callable[[TeleopCommand], RealSkillMotionDispatchReceipt],
        hold_controller: Callable[[str], RealSkillMotionHoldReceipt],
        max_measurement_age_sec: float = 0.2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be RealAgentMotionPort.")
        if not motion_port.config.exclusive_command_claim:
            raise ValueError("Skill deployment requires an exclusive motion port.")
        for name, callback in (
            ("observation_provider", observation_provider),
            ("synchronize_target_builder", synchronize_target_builder),
            ("command_admission", command_admission),
            ("command_dispatch", command_dispatch),
            ("hold_controller", hold_controller),
            ("clock", clock),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable.")
        self.motion_port = motion_port
        self._observation_provider = observation_provider
        self._synchronize_target_builder = synchronize_target_builder
        if measured_qpos_observer is not None and not callable(measured_qpos_observer):
            raise TypeError("measured_qpos_observer must be callable or None.")
        self._measured_qpos_observer = measured_qpos_observer
        self._command_admission = command_admission
        self._command_dispatch = command_dispatch
        self._hold_controller = hold_controller
        self._max_measurement_age_sec = _finite(
            max_measurement_age_sec,
            field_name="max_measurement_age_sec",
        )
        if self._max_measurement_age_sec <= 0.0:
            raise ValueError("max_measurement_age_sec must be positive.")
        self._clock = clock
        self._lock = threading.Lock()
        # Serialize an entire lower control cut against fail-closed/HOLD.  A
        # lifecycle flag alone is insufficient: a step that has already
        # passed admission could otherwise dispatch after a concurrent HOLD.
        self._cycle_lock = threading.Lock()
        self._lifecycle = RealSkillControlLifecycle.CREATED
        self._last_hold_receipt: RealSkillMotionHoldReceipt | None = None
        self._last_published_source_sequence_by_side: dict[str, int] | None = None
        self._control_cycle_count = 0
        self._latest_control_cycle_sec = 0.0
        self._max_control_cycle_sec = 0.0
        self._expired_reference_skip_count = 0
        self._consecutive_expired_reference_skips = 0
        self._latest_expired_measurement_age_sec: float | None = None
        self._transient_observation_skip_count = 0
        self._consecutive_transient_observation_skips = 0
        self._last_transient_observation_error: str | None = None
        self._motion_trace_sink: RealSkillControlTraceSink | None = None
        self._motion_trace_error_count = 0
        self._last_motion_trace_error: str | None = None

    @property
    def lifecycle(self) -> RealSkillControlLifecycle:
        with self._lock:
            return self._lifecycle

    @property
    def last_hold_receipt(self) -> RealSkillMotionHoldReceipt | None:
        with self._lock:
            return self._last_hold_receipt

    def control_cycle_snapshot(self) -> dict[str, object]:
        """Return bounded control timing evidence without lower or device I/O."""

        with self._lock:
            return {
                "cycle_count": self._control_cycle_count,
                "latest_cycle_sec": self._latest_control_cycle_sec,
                "max_cycle_sec": self._max_control_cycle_sec,
                "expired_reference_skip_count": (
                    self._expired_reference_skip_count
                ),
                "consecutive_expired_reference_skips": (
                    self._consecutive_expired_reference_skips
                ),
                "latest_expired_measurement_age_sec": (
                    self._latest_expired_measurement_age_sec
                ),
                "transient_observation_skip_count": (
                    self._transient_observation_skip_count
                ),
                "consecutive_transient_observation_skips": (
                    self._consecutive_transient_observation_skips
                ),
                "last_transient_observation_error": (
                    self._last_transient_observation_error
                ),
                "max_measurement_age_sec": self._max_measurement_age_sec,
                "last_published_source_sequence_by_side": dict(
                    self._last_published_source_sequence_by_side or {}
                ),
                "motion_trace_attached": self._motion_trace_sink is not None,
                "motion_trace_error_count": self._motion_trace_error_count,
                "last_motion_trace_error": self._last_motion_trace_error,
            }

    def set_motion_trace_sink(
        self,
        sink: RealSkillControlTraceSink | None,
    ) -> None:
        """Attach a best-effort trace sink without granting it control authority."""

        if sink is not None and not callable(sink):
            raise TypeError("motion trace sink must be callable or None.")
        with self._lock:
            self._motion_trace_sink = sink

    def start(self) -> str:
        with self._lock:
            if self._lifecycle != RealSkillControlLifecycle.CREATED:
                raise RuntimeError("exclusive Skill runtime can start only once.")
            evidence = self.motion_port.confirm_exclusive_attachment()
            self._lifecycle = RealSkillControlLifecycle.RUNNING
            return evidence

    def skill_motion_owner_state(self) -> tuple[str, str]:
        lifecycle = self.lifecycle
        if lifecycle == RealSkillControlLifecycle.RUNNING:
            return "running", "ready"
        if lifecycle == RealSkillControlLifecycle.FAULT:
            return "fault", "fault"
        return "hold", "hold"

    def hold_skill_motion(self, reason: str) -> bool:
        receipt = self._hold_controller(_required(reason, field_name="hold_reason"))
        if not isinstance(receipt, RealSkillMotionHoldReceipt):
            raise TypeError("hold_controller must return RealSkillMotionHoldReceipt.")
        with self._lock:
            self._last_hold_receipt = receipt
            self._lifecycle = (
                RealSkillControlLifecycle.HOLD
                if receipt.succeeded
                else RealSkillControlLifecycle.FAULT
            )
        return receipt.succeeded

    def _resume_confirmed_hold_for_pending_activation(self) -> bool:
        """Reopen only a successful HOLD that now has a new activation."""

        if self.motion_port.snapshot().phase != RealAgentMotionPhase.PENDING_ACTIVATION:
            return False
        with self._lock:
            if self._lifecycle != RealSkillControlLifecycle.HOLD:
                return False
            receipt = self._last_hold_receipt
            if receipt is None or not receipt.succeeded:
                self._lifecycle = RealSkillControlLifecycle.FAULT
                raise RuntimeError(
                    "exclusive Skill runtime cannot resume without a successful "
                    "lower HOLD receipt."
                )
            # The previous receipt proves only the previous terminal boundary.
            # Clear it before exposing RUNNING so shutdown or completion must
            # obtain new lower HOLD evidence for the new operation.
            self._last_hold_receipt = None
            self._lifecycle = RealSkillControlLifecycle.RUNNING
            return True

    def step(self, *, now: float | None = None) -> RealSkillControlStep:
        cycle_started = time.monotonic()
        with self._lock:
            sink = self._motion_trace_sink
            cycle_sequence = self._control_cycle_count + 1
        trace = (
            None
            if sink is None
            else RealSkillControlTraceSample(
                cycle_sequence=cycle_sequence,
                cycle_started_monotonic=cycle_started,
            )
        )
        try:
            with self._cycle_lock:
                result = self._step(
                    now=now,
                    cycle_started=cycle_started,
                    trace=trace,
                )
            return result
        except BaseException as exc:
            if trace is not None:
                trace.outcome = "exception"
                trace.error = f"{type(exc).__name__}: {exc}"[:512]
            raise
        finally:
            cycle_finished = time.monotonic()
            elapsed = max(0.0, cycle_finished - cycle_started)
            with self._lock:
                self._control_cycle_count += 1
                self._latest_control_cycle_sec = elapsed
                self._max_control_cycle_sec = max(
                    self._max_control_cycle_sec,
                    elapsed,
                )
            if trace is not None and sink is not None:
                trace.cycle_finished_monotonic = cycle_finished
                try:
                    sink(trace)
                except Exception as exc:  # noqa: BLE001 - diagnostics are isolated.
                    with self._lock:
                        self._motion_trace_error_count += 1
                        self._last_motion_trace_error = (
                            f"{type(exc).__name__}: {exc}"[:512]
                        )

    def _step(
        self,
        *,
        now: float | None = None,
        cycle_started: float | None = None,
        trace: RealSkillControlTraceSample | None = None,
    ) -> RealSkillControlStep:
        if cycle_started is None:
            cycle_started = time.monotonic()
        current_time = _finite(
            self._clock() if now is None else now,
            field_name="now",
        )
        if trace is not None:
            trace.control_time = current_time
        if self.lifecycle in {
            RealSkillControlLifecycle.CREATED,
            RealSkillControlLifecycle.CLOSED,
        }:
            raise RuntimeError("exclusive Skill runtime is not active.")
        try:
            cut = self._observation_provider()
        except RealSkillMotionObservationUnavailable as exc:
            snapshot = self.motion_port.snapshot()
            lifecycle = self.lifecycle
            if (
                lifecycle
                not in {
                    RealSkillControlLifecycle.RUNNING,
                    RealSkillControlLifecycle.HOLD,
                }
                or snapshot.phase
                not in {
                    RealAgentMotionPhase.INACTIVE,
                    RealAgentMotionPhase.HELD,
                }
            ):
                raise
            diagnostic = f"{type(exc).__name__}: {exc}"[:512]
            with self._lock:
                self._transient_observation_skip_count += 1
                self._consecutive_transient_observation_skips += 1
                self._last_transient_observation_error = diagnostic
            if trace is not None:
                trace.port_snapshot = snapshot
                trace.outcome = "transient_observation_unavailable"
                trace.error = diagnostic
            return RealSkillControlStep(
                lifecycle=lifecycle,
                port_snapshot=snapshot,
                command_present=False,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        with self._lock:
            self._consecutive_transient_observation_skips = 0
        if trace is not None:
            trace.observation_completed_monotonic = time.monotonic()
            trace.cut = cut
        if not isinstance(cut, RealSkillMotionObservationCut):
            raise TypeError(
                "observation_provider must return RealSkillMotionObservationCut."
            )
        control_cut_at = max(current_time, cut.available_at)
        source_sequences = dict(cut.source_sequence_by_side)
        publish_measurement = True
        previous_sequences = self._last_published_source_sequence_by_side
        if source_sequences and previous_sequences is not None:
            regressed = [
                side
                for side in ("left", "right")
                if source_sequences[side] < previous_sequences[side]
            ]
            if regressed:
                raise RuntimeError(
                    "lower measurement source sequence regressed for "
                    + ",".join(regressed)
                    + "."
                )
            publish_measurement = source_sequences != previous_sequences
        if publish_measurement:
            self._synchronize_target_builder(cut.qpos_by_side)
            measured_pose_cut = self.motion_port.publish_measured_qpos_pair(
                qpos_by_side=cut.qpos_by_side,
                measured_at=cut.measured_at,
                # The source cut may already have been available before this
                # cycle.  The derived TCP cut only becomes available to this
                # control owner when the cycle consumes it.
                available_at=control_cut_at,
                source_evidence_refs=cut.evidence_refs,
            )
            if trace is not None:
                trace.measured_pose_cut = measured_pose_cut
            if source_sequences:
                self._last_published_source_sequence_by_side = source_sequences
            if self._measured_qpos_observer is not None:
                self._measured_qpos_observer(
                    cut,
                    derived_available_at=control_cut_at,
                )
        elif trace is not None:
            trace.measured_pose_cut = self.motion_port.latest_measured_pose()
        if trace is not None:
            trace.synchronization_completed_monotonic = time.monotonic()
        self._resume_confirmed_hold_for_pending_activation()
        snapshot = self.motion_port.apply_control_owner_requests(
            owner=self,
            now=control_cut_at,
        )
        self.motion_port.publish_control_owner_state(
            owner=self,
            captured_at=control_cut_at,
        )
        if trace is not None:
            trace.owner_requests_completed_monotonic = time.monotonic()
            trace.port_snapshot = snapshot
        if self.lifecycle != RealSkillControlLifecycle.RUNNING:
            if trace is not None:
                trace.outcome = "owner_not_running"
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=snapshot,
                command_present=False,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        value = self.motion_port.command_for_observation(
            cut.observation,
            current_time,
        )
        command = value.command if isinstance(value, TeleopCommandOverride) else value
        if trace is not None:
            trace.ik_completed_monotonic = time.monotonic()
            trace.command = command
        if command is None:
            if trace is not None:
                trace.outcome = "no_command"
                trace.port_snapshot = self.motion_port.snapshot()
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=self.motion_port.snapshot(),
                command_present=False,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        if not isinstance(command, TeleopCommand):
            raise TypeError("motion port returned an unsupported command value.")
        if trace is not None:
            trace.command_present = True
        admitted = bool(
            self.motion_port.admit_command(command, cut.observation, current_time)
            and self._command_admission(command, cut.observation, current_time)
        )
        if trace is not None:
            trace.admission_completed_monotonic = time.monotonic()
            trace.admitted = admitted
        if not admitted:
            self.motion_port.observe_session_step(
                command=command,
                command_dispatched=False,
            )
            if trace is not None:
                trace.outcome = "admission_rejected"
                trace.port_snapshot = self.motion_port.snapshot()
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=self.motion_port.snapshot(),
                command_present=True,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        dispatch_ready_at = control_cut_at + max(
            0.0,
            time.monotonic() - cycle_started,
        )
        measurement_age_sec = max(0.0, dispatch_ready_at - cut.measured_at)
        if trace is not None:
            trace.dispatch_ready_at = dispatch_ready_at
        if measurement_age_sec > self._max_measurement_age_sec:
            # An expensive IK/collision cycle must never dispatch a reference
            # built from a cut that expired while it was being computed.  The
            # next cycle can consume a newer independently sampled cut.
            self.motion_port.observe_session_step(
                command=command,
                command_dispatched=False,
            )
            with self._lock:
                self._expired_reference_skip_count += 1
                self._consecutive_expired_reference_skips += 1
                self._latest_expired_measurement_age_sec = measurement_age_sec
            if trace is not None:
                trace.expired_reference = True
                trace.outcome = "reference_expired"
                trace.port_snapshot = self.motion_port.snapshot()
            return RealSkillControlStep(
                lifecycle=self.lifecycle,
                port_snapshot=self.motion_port.snapshot(),
                command_present=True,
                dispatch_attempted=False,
                dispatch_succeeded=False,
                dispatch_evidence_refs=(),
            )
        with self._lock:
            self._consecutive_expired_reference_skips = 0
        receipt = self._command_dispatch(command)
        if trace is not None:
            trace.dispatch_completed_monotonic = time.monotonic()
            trace.dispatch_receipt = receipt
            trace.dispatch_attempted = True
        if not isinstance(receipt, RealSkillMotionDispatchReceipt):
            raise TypeError(
                "command_dispatch must return RealSkillMotionDispatchReceipt."
            )
        self.motion_port.observe_session_step(
            command=command,
            command_dispatched=receipt.succeeded,
        )
        if trace is not None:
            trace.dispatch_succeeded = receipt.succeeded
            trace.outcome = (
                "dispatch_succeeded" if receipt.succeeded else "dispatch_failed"
            )
        if not receipt.succeeded:
            operation_id = self.motion_port.snapshot().operation_id
            if operation_id is not None:
                self.motion_port.request_hold(
                    operation_id=operation_id,
                    reason="lower_dispatch_failed",
                    requested_at=current_time,
                )
                self.motion_port.apply_control_owner_requests(
                    owner=self,
                    now=current_time,
                )
        final_snapshot = self.motion_port.snapshot()
        if trace is not None:
            trace.port_snapshot = final_snapshot
        return RealSkillControlStep(
            lifecycle=self.lifecycle,
            port_snapshot=final_snapshot,
            command_present=True,
            dispatch_attempted=True,
            dispatch_succeeded=receipt.succeeded,
            dispatch_evidence_refs=receipt.evidence_refs,
        )

    def fail_closed(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> RealSkillMotionHoldReceipt:
        """Reach lower-confirmed HOLD without requiring another observation."""

        with self._cycle_lock:
            return self._fail_closed(reason, now=now)

    def _fail_closed(
        self,
        reason: str,
        *,
        now: float | None = None,
    ) -> RealSkillMotionHoldReceipt:

        current_time = _finite(
            self._clock() if now is None else now,
            field_name="now",
        )
        if self.lifecycle == RealSkillControlLifecycle.CLOSED:
            raise RuntimeError("closed exclusive Skill runtime cannot request HOLD.")
        snapshot = self.motion_port.snapshot()
        receipt = self.last_hold_receipt
        if (
            snapshot.phase == RealAgentMotionPhase.HELD
            and snapshot.hold_succeeded is True
            and receipt is not None
            and receipt.succeeded
        ):
            return receipt
        normalized_reason = _required(reason, field_name="hold_reason")
        if snapshot.phase not in {
            RealAgentMotionPhase.HOLD_REQUESTED,
            RealAgentMotionPhase.HOLD_APPLYING,
        }:
            if snapshot.operation_id is None:
                self.motion_port.request_domain_hold(
                    reason=normalized_reason,
                    requested_at=current_time,
                )
            else:
                self.motion_port.request_hold(
                    operation_id=snapshot.operation_id,
                    reason=normalized_reason,
                    requested_at=current_time,
                )
        snapshot = self.motion_port.apply_control_owner_requests(
            owner=self,
            now=current_time,
        )
        self.motion_port.publish_control_owner_state(
            owner=self,
            captured_at=current_time,
        )
        receipt = self.last_hold_receipt
        if (
            snapshot.phase != RealAgentMotionPhase.HELD
            or snapshot.hold_succeeded is not True
            or receipt is None
            or not receipt.succeeded
        ):
            raise RuntimeError(
                "exclusive Skill runtime failed to reach lower-confirmed HOLD."
            )
        return receipt

    def close(self, *, reason: str = "exclusive_skill_runtime_shutdown") -> None:
        if self.lifecycle == RealSkillControlLifecycle.CLOSED:
            return
        self.fail_closed(reason)
        with self._lock:
            self._lifecycle = RealSkillControlLifecycle.CLOSED


__all__ = [
    "ExclusiveRealSkillControlRuntime",
    "RealSkillControlLifecycle",
    "RealSkillControlStep",
    "RealSkillControlTraceSample",
    "RealSkillControlTraceSink",
    "RealSkillMotionDispatchReceipt",
    "RealSkillMotionEmergencyStopReceipt",
    "RealSkillMotionHoldReceipt",
    "RealSkillMotionObservationCut",
    "RealSkillMotionObservationUnavailable",
    "RealSkillMotionTargetSynchronizer",
]
