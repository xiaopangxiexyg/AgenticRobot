"""Single-flight blocking operations for the real-teleop collection boundary.

The real-time control loop owns admission and publishes completed state.  This
module only runs one explicitly submitted blocking operation on a daemon
thread and makes its immutable completion available for non-blocking polling.
It owns no robot command, dataset schema, episode policy, or GUI state.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any


class CollectionControlOperationBusy(RuntimeError):
    """Another collection control-plane operation has not been consumed yet."""


@dataclass(frozen=True)
class CollectionControlOperationResult:
    generation: int
    kind: str
    started_at: float
    completed_at: float
    value: Any | None = None
    error: BaseException | None = None
    cancelled: bool = False
    discard_error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return bool(
            not self.cancelled
            and self.error is None
            and self.discard_error is None
        )

    @property
    def elapsed_sec(self) -> float:
        return max(0.0, float(self.completed_at) - float(self.started_at))


@dataclass
class _PendingOperation:
    generation: int
    kind: str
    discard_value: Callable[[Any], None] | None
    cancelled: bool = False


class CollectionControlPlane:
    """Run one blocking lifecycle operation without waiting in the caller.

    The caller must poll and consume each completion before submitting the next
    operation.  Cancellation is cooperative at the publication boundary: the
    running callable is not interrupted, but a value produced after
    cancellation is discarded before the cancelled completion is observable.
    """

    def __init__(
        self,
        *,
        name: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("collection control-plane name must be non-empty")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.name = normalized_name
        self._clock = clock
        self._lock = threading.Lock()
        self._next_generation = 1
        self._pending: _PendingOperation | None = None
        self._completed: CollectionControlOperationResult | None = None
        self._completed_discard: Callable[[Any], None] | None = None
        self._worker: threading.Thread | None = None
        self._closed = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def active_kind(self) -> str | None:
        with self._lock:
            if self._pending is not None:
                return self._pending.kind
            if self._completed is not None:
                return self._completed.kind
            return None

    @property
    def active_operation(self) -> tuple[int, str] | None:
        """Return the one pending or unconsumed completion identity."""

        with self._lock:
            if self._pending is not None:
                return self._pending.generation, self._pending.kind
            if self._completed is not None:
                return self._completed.generation, self._completed.kind
            return None

    def submit(
        self,
        kind: str,
        operation: Callable[[], Any],
        *,
        discard_value: Callable[[Any], None] | None = None,
    ) -> int:
        normalized_kind = str(kind).strip()
        if not normalized_kind:
            raise ValueError("collection operation kind must be non-empty")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if discard_value is not None and not callable(discard_value):
            raise TypeError("discard_value must be callable or None")
        with self._lock:
            if self._closed:
                raise RuntimeError("collection control plane is closed")
            if self._pending is not None or self._completed is not None:
                active_kind = self.active_kind_unlocked()
                raise CollectionControlOperationBusy(
                    f"collection control-plane operation already pending: {active_kind}"
                )
            generation = self._next_generation
            self._next_generation += 1
            pending = _PendingOperation(
                generation=generation,
                kind=normalized_kind,
                discard_value=discard_value,
            )
            self._pending = pending
            try:
                worker = threading.Thread(
                    target=self._run,
                    args=(pending, operation),
                    name=f"{self.name}:{normalized_kind}:{generation}",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
            except BaseException:
                # Thread construction/start is part of submission, not the
                # operation.  No worker owns ``pending`` when it fails, so
                # retaining it would wedge this single-flight plane forever.
                self._pending = None
                self._worker = None
                raise
        return generation

    def active_kind_unlocked(self) -> str | None:
        if self._pending is not None:
            return self._pending.kind
        if self._completed is not None:
            return self._completed.kind
        return None

    def _run(
        self,
        pending: _PendingOperation,
        operation: Callable[[], Any],
    ) -> None:
        started_at = float(self._clock())
        value: Any | None = None
        error: BaseException | None = None
        try:
            value = operation()
        except BaseException as exc:  # completion, not the daemon, owns failure
            error = exc
        result = CollectionControlOperationResult(
            generation=pending.generation,
            kind=pending.kind,
            started_at=started_at,
            completed_at=float(self._clock()),
            value=value,
            error=error,
        )
        with self._lock:
            cancelled = bool(
                self._closed
                or self._pending is not pending
                or pending.cancelled
            )
            if not cancelled:
                # Publication and cancellation share this linearization point.
                # A later cancel owns a completed value, never an old snapshot
                # that can still be published as success after cancellation.
                self._publish_result_unlocked(pending, result, pending.discard_value)
                return
            if error is not None or pending.discard_value is None:
                self._publish_result_unlocked(
                    pending,
                    replace(result, value=None, cancelled=True),
                    None,
                )
                return

        # Keep the pending generation and its worker until retirement returns.
        # Blocking cleanup runs outside the lock and never on the control thread.
        self._discard_cancelled_completion(
            pending,
            result,
            pending.discard_value,
            completed_at=result.completed_at,
        )

    def _publish_result_unlocked(
        self,
        pending: _PendingOperation,
        result: CollectionControlOperationResult,
        discard_value: Callable[[Any], None] | None,
    ) -> None:
        """Transfer one terminal result while the caller holds the plane lock."""

        if self._pending is pending:
            self._pending = None
            if not self._closed:
                self._completed = result
                self._completed_discard = discard_value
        if self._worker is threading.current_thread():
            self._worker = None

    def _discard_cancelled_completion(
        self,
        pending: _PendingOperation,
        result: CollectionControlOperationResult,
        discard_value: Callable[[Any], None],
        *,
        completed_at: float | None = None,
    ) -> None:
        """Retire an unconsumed completed value without blocking its caller.

        Cancellation can race the control thread's next poll: the blocking
        operation may already have published a ready writer candidate while
        the operator is requesting Hold.  Closing that process-backed value
        in :meth:`cancel` would put Queue/process teardown back onto the
        realtime thread.  Keep the original generation single-flight until
        cleanup has returned, then publish the cancelled completion.
        """

        discard_error: BaseException | None = None
        try:
            discard_value(result.value)
        except BaseException as exc:  # cleanup evidence belongs to completion
            discard_error = exc
        cancelled_result = replace(
            result,
            value=None,
            # Running-operation timing excludes retirement; an already
            # published value retains its historical cleanup-completion timing.
            completed_at=(
                float(self._clock()) if completed_at is None else completed_at
            ),
            cancelled=True,
            discard_error=discard_error,
        )
        with self._lock:
            self._publish_result_unlocked(pending, cancelled_result, None)

    def _start_completed_discard_unlocked(
        self,
        result: CollectionControlOperationResult,
        discard_value: Callable[[Any], None],
    ) -> bool:
        """Transfer a ready value to retirement, or retain it for a retry."""

        pending = _PendingOperation(
            generation=result.generation,
            kind=result.kind,
            discard_value=None,
            cancelled=True,
        )
        self._completed = None
        self._completed_discard = None
        try:
            worker = threading.Thread(
                target=self._discard_cancelled_completion,
                args=(pending, result, discard_value),
                name=f"{self.name}:{result.kind}:cancel:{result.generation}",
                daemon=True,
            )
            self._pending = pending
            self._worker = worker
            worker.start()
        except BaseException:
            # Neither construction nor start transferred ownership.  Closed
            # planes retain the same value too, for a later close() retry.
            self._pending = None
            self._worker = None
            self._completed = result
            self._completed_discard = discard_value
            return False
        return True

    def poll(
        self,
        generation: int | None = None,
    ) -> CollectionControlOperationResult | None:
        with self._lock:
            if self._closed:
                # A failed retirement submission may retain a ready value for
                # retry, but shutdown must never publish it to a consumer.
                return None
            result = self._completed
            if result is None:
                return None
            if generation is not None and int(generation) != result.generation:
                return None
            self._completed = None
            self._completed_discard = None
            return result

    def wait(
        self,
        generation: int,
        *,
        timeout_sec: float,
    ) -> CollectionControlOperationResult | None:
        """Wait at most *timeout_sec* and consume one exact completion."""

        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError(
                "collection control-plane wait timeout must be finite "
                "and non-negative"
            )
        requested_generation = int(generation)
        with self._lock:
            completed = self._completed
            if (
                completed is not None
                and completed.generation == requested_generation
            ):
                worker = None
            elif (
                self._pending is not None
                and self._pending.generation == requested_generation
            ):
                worker = self._worker
            else:
                return None
        if worker is not None:
            if worker is threading.current_thread():
                return None
            worker.join(timeout=timeout)
        return self.poll(requested_generation)

    def cancel(self, generation: int) -> bool:
        with self._lock:
            if (
                self._pending is not None
                and self._pending.generation == int(generation)
            ):
                if self._pending.cancelled:
                    return False
                self._pending.cancelled = True
                return True
            result = self._completed
            if result is None or result.generation != int(generation):
                return False
            if result.cancelled:
                return False
            discard_value = self._completed_discard
            if result.error is not None or discard_value is None:
                self._completed = replace(result, value=None, cancelled=True)
                self._completed_discard = None
                return True
            return self._start_completed_discard_unlocked(result, discard_value)

    def close(self, *, timeout_sec: float = 5.0) -> bool:
        """Cancel publication, discard owned output, and wait for the worker.

        Running Python callables cannot be interrupted safely.  Closing is
        therefore cooperative, but unlike :meth:`cancel` it is also a bounded
        resource barrier: ``True`` means the submitted callable and its
        cancellation cleanup have both returned.  A ``False`` result means
        the caller must not treat dependencies used by that callable as fully
        quiesced yet; a later call may finish draining the same closed plane.
        Already completed values use the same retirement worker as cancel(),
        so their blocking cleanup is included in this wait budget.  Failure to
        start that worker returns False and retains the value for a later retry.
        """

        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError(
                "collection control-plane close timeout must be finite "
                "and non-negative"
            )
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
            if self._pending is not None:
                self._pending.cancelled = True
            if self._completed is not None:
                result = self._completed
                discard_value = self._completed_discard
                if result.error is None and discard_value is not None:
                    if not self._start_completed_discard_unlocked(result, discard_value):
                        return False
                else:
                    self._completed = None
                    self._completed_discard = None
            worker = self._worker
        if worker is None:
            return True
        if worker is threading.current_thread():
            return False
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        return not worker.is_alive()


__all__ = [
    "CollectionControlOperationBusy",
    "CollectionControlOperationResult",
    "CollectionControlPlane",
]
