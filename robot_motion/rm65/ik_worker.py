"""Bounded latest-only worker for coupled RM65 IK proposals.

The worker owns the mutable coupled solver and performs only proposal
calculation.  The caller remains the sole owner of result admission, canonical
safety, reference activation, dispatch, and hardware acknowledgement.

One request may be in flight and one newer request may be pending.  Replacing a
pending request is intentional: an old Cartesian target must never form a FIFO
backlog behind a difficult solve.  Every request is rebound to the exact
accepted dual-arm qref supplied by the caller before solving, and every
completion carries enough lineage for the caller to reject stale work.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from robot_motion.runtime.thread_affinity import bind_current_thread_to_cpu

from .dual_arm_target_solver import (
    BimanualIKResult,
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)


class BimanualIKProposalSolver(Protocol):
    """Mutable coupled solver surface owned exclusively by the worker."""

    def solve_target_pair(self, target: BimanualPoseTarget) -> BimanualIKResult: ...

    def set_current_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray | None = None,
        right_qpos: Sequence[float] | np.ndarray | None = None,
        reset_tracking: bool = True,
    ) -> None: ...

    def reset_target_pose_smoothing(self) -> None: ...


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _qpos_pair(value: Mapping[str, Any], *, name: str) -> dict[str, np.ndarray]:
    if set(value) != {"left", "right"}:
        raise ValueError(f"{name} must contain exactly left and right.")
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        qpos = np.asarray(value[side], dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}.{side} must contain 6 finite values.")
        result[side] = qpos.copy()
    return result


def _copy_target(target: BimanualPoseTarget) -> BimanualPoseTarget:
    if not isinstance(target, BimanualPoseTarget):
        raise TypeError("target must be a BimanualPoseTarget.")
    sources: dict[str, BimanualPoseTargetSource | None] = {}
    for side in ("left", "right"):
        source = target.source_by_side[side]
        sources[side] = (
            None
            if source is None
            else BimanualPoseTargetSource(
                command_type=source.command_type,
                value=source.value,
                frame_name=source.frame_name,
                metadata=source.metadata,
            )
        )
    return BimanualPoseTarget(
        timestamp=target.timestamp,
        target_pose_by_side=target.target_pose_by_side,
        active_sides=target.active_sides,
        hard_hold_inactive_sides=target.hard_hold_inactive_sides,
        world_to_base_by_side=target.world_to_base_by_side,
        source_by_side=sources,
        input_skipped_by_side=target.input_skipped_by_side,
        target_smoothing_metadata=target.target_smoothing_metadata,
    )


@dataclass(frozen=True)
class ThreadedBimanualIKWorkerConfig:
    """Lifecycle and scheduling configuration for the proposal worker."""

    worker_name: str = "rm65-bimanual-ik-proposal"
    worker_cpu: int | None = None
    startup_timeout_sec: float = 2.0
    join_timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        name = str(self.worker_name).strip()
        if not name:
            raise ValueError("worker_name must be non-empty.")
        object.__setattr__(self, "worker_name", name)
        cpu = self.worker_cpu
        if cpu is not None:
            cpu = _nonnegative_integer(cpu, name="worker_cpu")
            object.__setattr__(self, "worker_cpu", cpu)
        for field_name in ("startup_timeout_sec", "join_timeout_sec"):
            value = _finite_time(getattr(self, field_name), name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class BimanualIKProposalRequest:
    """Immutable input and lineage for one detached coupled solve."""

    worker_generation: int
    control_epoch: int
    anchor_generation: int
    request_sequence: int
    submitted_at: float
    valid_until: float
    input_state_timestamp: float
    reference_qpos_by_side: Mapping[str, np.ndarray]
    target: BimanualPoseTarget
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "worker_generation",
            "control_epoch",
            "anchor_generation",
            "request_sequence",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(getattr(self, field_name), name=field_name),
            )
        submitted_at = _finite_time(self.submitted_at, name="submitted_at")
        valid_until = _finite_time(self.valid_until, name="valid_until")
        input_timestamp = _finite_time(
            self.input_state_timestamp,
            name="input_state_timestamp",
        )
        if valid_until <= submitted_at:
            raise ValueError("valid_until must be later than submitted_at.")
        if input_timestamp > submitted_at + 1e-9:
            raise ValueError("input_state_timestamp cannot be later than submitted_at.")
        object.__setattr__(self, "submitted_at", submitted_at)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "input_state_timestamp", input_timestamp)
        object.__setattr__(
            self,
            "reference_qpos_by_side",
            _qpos_pair(self.reference_qpos_by_side, name="reference_qpos_by_side"),
        )
        object.__setattr__(self, "target", _copy_target(self.target))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))


@dataclass(frozen=True)
class BimanualIKProposalCompletion:
    """Detached proposal result; it is never an execution admission receipt."""

    request: BimanualIKProposalRequest
    started_at: float
    completed_at: float
    result: BimanualIKResult | None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, BimanualIKProposalRequest):
            raise TypeError("request must be a BimanualIKProposalRequest.")
        started_at = _finite_time(self.started_at, name="started_at")
        completed_at = _finite_time(self.completed_at, name="completed_at")
        if started_at + 1e-12 < self.request.submitted_at:
            raise ValueError("started_at cannot precede request submission.")
        if completed_at + 1e-12 < started_at:
            raise ValueError("completed_at cannot precede started_at.")
        if (self.result is None) == (self.error is None):
            raise ValueError("completion must contain exactly one of result or error.")
        if self.result is not None and not isinstance(self.result, BimanualIKResult):
            raise TypeError("result must be a BimanualIKResult or None.")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)

    @property
    def elapsed_ms(self) -> float:
        return max(0.0, self.completed_at - self.started_at) * 1000.0

    @property
    def deadline_missed(self) -> bool:
        return self.completed_at > self.request.valid_until


@dataclass(frozen=True)
class BimanualIKProposalAssessment:
    """Transport-freshness decision made by the authoritative caller.

    Eligibility only means that the proposal may enter the caller's existing
    canonical admission path.  It does not certify collision safety, rate
    limits, reference activation, or dispatch.
    """

    eligible_for_admission: bool
    reasons: tuple[str, ...]
    result: BimanualIKResult | None
    reference_qpos_max_abs_drift_rad_by_side: Mapping[str, float]


def assess_bimanual_ik_proposal(
    completion: BimanualIKProposalCompletion,
    *,
    now: float,
    required_worker_generation: int,
    required_control_epoch: int,
    required_anchor_generation: int,
    current_reference_qpos_by_side: Mapping[str, Any],
    max_reference_drift_rad: float,
    minimum_request_sequence: int | None = None,
) -> BimanualIKProposalAssessment:
    """Reject stale transport work before canonical result admission."""

    if not isinstance(completion, BimanualIKProposalCompletion):
        raise TypeError("completion must be a BimanualIKProposalCompletion.")
    timestamp = _finite_time(now, name="now")
    worker_generation = _nonnegative_integer(
        required_worker_generation,
        name="required_worker_generation",
    )
    control_epoch = _nonnegative_integer(
        required_control_epoch,
        name="required_control_epoch",
    )
    anchor_generation = _nonnegative_integer(
        required_anchor_generation,
        name="required_anchor_generation",
    )
    current_reference = _qpos_pair(
        current_reference_qpos_by_side,
        name="current_reference_qpos_by_side",
    )
    drift_limit = _finite_time(
        max_reference_drift_rad,
        name="max_reference_drift_rad",
    )
    if drift_limit < 0.0:
        raise ValueError("max_reference_drift_rad must be non-negative.")
    minimum_sequence = (
        None
        if minimum_request_sequence is None
        else _nonnegative_integer(
            minimum_request_sequence,
            name="minimum_request_sequence",
        )
    )
    request = completion.request
    reasons: list[str] = []
    drift_by_side = {
        side: float(
            np.max(
                np.abs(
                    current_reference[side]
                    - request.reference_qpos_by_side[side]
                )
            )
        )
        for side in ("left", "right")
    }
    if request.worker_generation != worker_generation:
        reasons.append("worker_generation_mismatch")
    if request.control_epoch != control_epoch:
        reasons.append("control_epoch_mismatch")
    if request.anchor_generation != anchor_generation:
        reasons.append("anchor_generation_mismatch")
    if minimum_sequence is not None and request.request_sequence < minimum_sequence:
        reasons.append("request_sequence_too_old")
    for side in ("left", "right"):
        if drift_by_side[side] > drift_limit + 1e-12:
            reasons.append(f"reference_qpos_drift_exceeded:{side}")
    if completion.error is not None:
        reasons.append("solver_error")
    if completion.deadline_missed:
        reasons.append("completed_after_valid_until")
    if timestamp > request.valid_until:
        reasons.append("result_expired")
    if completion.result is None:
        reasons.append("result_missing")
    return BimanualIKProposalAssessment(
        eligible_for_admission=not reasons,
        reasons=tuple(reasons),
        result=completion.result if not reasons else None,
        reference_qpos_max_abs_drift_rad_by_side=MappingProxyType(
            drift_by_side
        ),
    )


@dataclass(frozen=True)
class _ResetRequest:
    worker_generation: int
    control_epoch: int
    qpos_by_side: Mapping[str, np.ndarray]


class ThreadedBimanualIKProposalWorker:
    """Own one coupled solver behind a non-blocking latest-only mailbox."""

    def __init__(
        self,
        solver: BimanualIKProposalSolver,
        *,
        config: ThreadedBimanualIKWorkerConfig | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for method_name in (
            "solve_target_pair",
            "set_current_qpos_pair",
            "reset_target_pose_smoothing",
        ):
            if not callable(getattr(solver, method_name, None)):
                raise TypeError(f"solver must expose {method_name}().")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        self.solver = solver
        self.config = config or ThreadedBimanualIKWorkerConfig()
        self.clock = monotonic_clock

        self._condition = threading.Condition(threading.RLock())
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._startup_ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._affinity_receipt: Mapping[str, Any] | None = None
        self._started = False

        self._worker_generation = 0
        self._control_epoch = 0
        self._ready_generation: int | None = None
        self._next_sequence = 0
        self._pending_reset: _ResetRequest | None = None
        self._pending_request: BimanualIKProposalRequest | None = None
        self._completion: BimanualIKProposalCompletion | None = None
        self._in_flight = False
        self._last_error: str | None = None

        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._pending_superseded_count = 0
        self._completion_superseded_count = 0
        self._discarded_stale_completion_count = 0
        self._deadline_miss_count = 0
        self._reset_count = 0
        self._last_elapsed_ms: float | None = None

    @property
    def started(self) -> bool:
        with self._condition:
            return self._started

    @property
    def generation(self) -> int:
        with self._condition:
            return self._worker_generation

    @property
    def control_epoch(self) -> int:
        with self._condition:
            return self._control_epoch

    @property
    def ready(self) -> bool:
        with self._condition:
            return bool(
                self._started
                and self._ready_generation == self._worker_generation
                and self._startup_error is None
                and self._thread is not None
                and self._thread.is_alive()
            )

    def _publish_completion_locked(
        self,
        completion: BimanualIKProposalCompletion,
    ) -> None:
        if (
            completion.request.worker_generation != self._worker_generation
            or completion.request.control_epoch != self._control_epoch
            or self._pending_reset is not None
        ):
            self._discarded_stale_completion_count += 1
            return
        if self._completion is not None:
            self._completion_superseded_count += 1
        self._completion = completion
        self._last_elapsed_ms = completion.elapsed_ms
        if completion.deadline_missed:
            self._deadline_miss_count += 1
        if completion.error is None:
            self._completed_count += 1
            self._last_error = None
        else:
            self._failed_count += 1
            self._last_error = (
                f"{type(completion.error).__name__}:{completion.error}"
            )

    def _apply_reset(self, reset: _ResetRequest) -> None:
        qpos = reset.qpos_by_side
        self.solver.set_current_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
            reset_tracking=True,
        )
        self.solver.reset_target_pose_smoothing()

    def _worker_loop(self) -> None:
        try:
            receipt = bind_current_thread_to_cpu(
                self.config.worker_cpu,
                role=self.config.worker_name,
            )
            with self._condition:
                self._affinity_receipt = dict(receipt)
        except Exception as exc:  # noqa: BLE001 - startup must fail closed.
            with self._condition:
                self._startup_error = exc
                self._last_error = f"{type(exc).__name__}:{exc}"
            self._startup_ready.set()
            return
        self._startup_ready.set()

        while True:
            with self._condition:
                while (
                    not self._stop_requested
                    and self._pending_reset is None
                    and self._pending_request is None
                ):
                    self._condition.wait()
                if self._stop_requested:
                    return
                reset = self._pending_reset
                if reset is not None:
                    self._pending_reset = None
                    request = None
                else:
                    request = self._pending_request
                    self._pending_request = None
                    self._in_flight = True

            if reset is not None:
                try:
                    self._apply_reset(reset)
                except Exception as exc:  # noqa: BLE001 - report to owner.
                    with self._condition:
                        if reset.worker_generation == self._worker_generation:
                            self._last_error = f"{type(exc).__name__}:{exc}"
                            self._failed_count += 1
                            self._ready_generation = None
                            if self._pending_request is not None:
                                self._pending_superseded_count += 1
                            self._pending_request = None
                            self._condition.notify_all()
                    continue
                with self._condition:
                    if (
                        reset.worker_generation == self._worker_generation
                        and reset.control_epoch == self._control_epoch
                    ):
                        self._ready_generation = reset.worker_generation
                        self._reset_count += 1
                        self._last_error = None
                    self._condition.notify_all()
                continue

            assert request is not None
            started_at = _finite_time(self.clock(), name="clock")
            try:
                qpos = request.reference_qpos_by_side
                self.solver.set_current_qpos_pair(
                    left_qpos=qpos["left"],
                    right_qpos=qpos["right"],
                    reset_tracking=False,
                )
                result = self.solver.solve_target_pair(request.target)
                if not isinstance(result, BimanualIKResult):
                    raise TypeError("solver returned a non-BimanualIKResult value.")
                error = None
            except Exception as exc:  # noqa: BLE001 - report to owner.
                result = None
                error = exc
            completed_at = _finite_time(self.clock(), name="clock")
            completion = BimanualIKProposalCompletion(
                request=request,
                started_at=started_at,
                completed_at=completed_at,
                result=result,
                error=error,
            )
            with self._condition:
                self._in_flight = False
                self._publish_completion_locked(completion)
                self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            if self._thread is not None:
                raise RuntimeError("A closed bimanual IK worker cannot be restarted.")
            self._stop_requested = False
            self._startup_ready.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=self.config.worker_name,
                daemon=True,
            )
            self._thread.start()
        if not self._startup_ready.wait(self.config.startup_timeout_sec):
            raise TimeoutError("bimanual IK worker affinity startup timed out.")
        with self._condition:
            if self._startup_error is not None:
                error = self._startup_error
                self._stop_requested = True
                self._condition.notify_all()
            else:
                self._started = True
                return
        assert error is not None
        raise RuntimeError("bimanual IK worker affinity startup failed") from error

    def reset(
        self,
        *,
        control_epoch: int,
        current_qpos_by_side: Mapping[str, Any],
    ) -> int:
        """Queue a lifecycle reset ahead of every solve and return its generation."""

        epoch = _nonnegative_integer(control_epoch, name="control_epoch")
        qpos = _qpos_pair(current_qpos_by_side, name="current_qpos_by_side")
        with self._condition:
            if not self._started:
                raise RuntimeError("ThreadedBimanualIKProposalWorker.start() is required.")
            self._worker_generation += 1
            generation = self._worker_generation
            self._control_epoch = epoch
            self._ready_generation = None
            self._pending_reset = _ResetRequest(
                worker_generation=generation,
                control_epoch=epoch,
                qpos_by_side=qpos,
            )
            if self._pending_request is not None:
                self._pending_superseded_count += 1
            self._pending_request = None
            if self._completion is not None:
                self._discarded_stale_completion_count += 1
            self._completion = None
            self._last_error = None
            self._condition.notify_all()
            return generation

    def wait_until_ready(self, generation: int, *, timeout_sec: float) -> bool:
        """Bounded lifecycle wait for startup/rearm paths, never a control tick."""

        expected = _nonnegative_integer(generation, name="generation")
        timeout = _finite_time(timeout_sec, name="timeout_sec")
        if timeout < 0.0:
            raise ValueError("timeout_sec must be non-negative.")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._ready_generation == expected:
                    return True
                if expected != self._worker_generation or self._last_error is not None:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)

    def submit(
        self,
        target: BimanualPoseTarget,
        *,
        control_epoch: int,
        anchor_generation: int,
        reference_qpos_by_side: Mapping[str, Any],
        input_state_timestamp: float,
        valid_until: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> BimanualIKProposalRequest:
        """Publish a proposal request without waiting for the current solve."""

        epoch = _nonnegative_integer(control_epoch, name="control_epoch")
        anchor = _nonnegative_integer(anchor_generation, name="anchor_generation")
        submitted_at = _finite_time(self.clock(), name="clock")
        with self._condition:
            if not self._started:
                raise RuntimeError("ThreadedBimanualIKProposalWorker.start() is required.")
            if epoch != self._control_epoch:
                raise ValueError(
                    "control_epoch does not match the current worker activation."
                )
            if (
                self._ready_generation != self._worker_generation
                and self._pending_reset is None
            ):
                raise RuntimeError(
                    "bimanual IK worker requires a successful reset before submit."
                )
            self._next_sequence += 1
            request = BimanualIKProposalRequest(
                worker_generation=self._worker_generation,
                control_epoch=epoch,
                anchor_generation=anchor,
                request_sequence=self._next_sequence,
                submitted_at=submitted_at,
                valid_until=valid_until,
                input_state_timestamp=input_state_timestamp,
                reference_qpos_by_side=reference_qpos_by_side,
                target=target,
                metadata={} if metadata is None else metadata,
            )
            if self._pending_request is not None:
                self._pending_superseded_count += 1
            self._pending_request = request
            self._submitted_count += 1
            self._condition.notify_all()
            return request

    def poll_completion(self) -> BimanualIKProposalCompletion | None:
        """Take the latest completion without performing canonical admission."""

        with self._condition:
            if not self._started:
                raise RuntimeError("ThreadedBimanualIKProposalWorker.start() is required.")
            completion = self._completion
            self._completion = None
            return completion

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "worker_name": self.config.worker_name,
                "started": self._started,
                "ready": self.ready,
                "worker_alive": bool(self._thread and self._thread.is_alive()),
                "worker_generation": self._worker_generation,
                "control_epoch": self._control_epoch,
                "ready_generation": self._ready_generation,
                "in_flight": self._in_flight,
                "pending_request": self._pending_request is not None,
                "pending_reset": self._pending_reset is not None,
                "completion_available": self._completion is not None,
                "submitted_count": self._submitted_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "pending_superseded_count": self._pending_superseded_count,
                "completion_superseded_count": self._completion_superseded_count,
                "discarded_stale_completion_count": (
                    self._discarded_stale_completion_count
                ),
                "deadline_miss_count": self._deadline_miss_count,
                "reset_count": self._reset_count,
                "last_elapsed_ms": self._last_elapsed_ms,
                "last_error": self._last_error,
                "affinity_receipt": (
                    None
                    if self._affinity_receipt is None
                    else dict(self._affinity_receipt)
                ),
            }

    def close(self) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                self._started = False
                return
            self._worker_generation += 1
            self._ready_generation = None
            self._pending_reset = None
            self._pending_request = None
            self._completion = None
            self._stop_requested = True
            self._condition.notify_all()
        thread.join(timeout=self.config.join_timeout_sec)
        alive = thread.is_alive()
        with self._condition:
            self._started = False
        if alive:
            raise RuntimeError(
                "bimanual IK proposal worker did not drain before close timeout."
            )


__all__ = [
    "BimanualIKProposalAssessment",
    "BimanualIKProposalCompletion",
    "BimanualIKProposalRequest",
    "BimanualIKProposalSolver",
    "ThreadedBimanualIKProposalWorker",
    "ThreadedBimanualIKWorkerConfig",
    "assess_bimanual_ik_proposal",
]
