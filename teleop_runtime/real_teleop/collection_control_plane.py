"""Single-flight blocking operations for the real-teleop collection boundary.

The real-time control loop owns admission and publishes completed state.  This
module only runs one explicitly submitted blocking operation on a daemon
thread and makes its immutable completion available for non-blocking polling.
It owns no robot command, dataset schema, episode policy, or GUI state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
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
        worker = threading.Thread(
            target=self._run,
            args=(pending, operation),
            name=f"{self.name}:{normalized_kind}:{generation}",
            daemon=True,
        )
        worker.start()
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
        completed_at = float(self._clock())

        with self._lock:
            current = self._pending
            cancelled = bool(
                self._closed
                or current is not pending
                or pending.cancelled
            )
        discard_error: BaseException | None = None
        if cancelled and error is None and pending.discard_value is not None:
            try:
                pending.discard_value(value)
            except BaseException as exc:  # cleanup evidence belongs to completion
                discard_error = exc

        result = CollectionControlOperationResult(
            generation=pending.generation,
            kind=pending.kind,
            started_at=started_at,
            completed_at=completed_at,
            value=None if cancelled else value,
            error=error,
            cancelled=cancelled,
            discard_error=discard_error,
        )
        with self._lock:
            if self._pending is not pending:
                return
            self._pending = None
            if not self._closed:
                self._completed = result
                self._completed_discard = pending.discard_value

    def poll(
        self,
        generation: int | None = None,
    ) -> CollectionControlOperationResult | None:
        with self._lock:
            result = self._completed
            if result is None:
                return None
            if generation is not None and int(generation) != result.generation:
                return None
            self._completed = None
            self._completed_discard = None
            return result

    def cancel(self, generation: int) -> bool:
        value_to_discard: Any | None = None
        discard_value: Callable[[Any], None] | None = None
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
            value_to_discard = result.value
            discard_value = self._completed_discard
            self._completed = CollectionControlOperationResult(
                generation=result.generation,
                kind=result.kind,
                started_at=result.started_at,
                completed_at=result.completed_at,
                error=result.error,
                cancelled=True,
            )
            self._completed_discard = None
        if result.error is None and discard_value is not None:
            try:
                discard_value(value_to_discard)
            except BaseException as exc:
                with self._lock:
                    current = self._completed
                    if current is not None and current.generation == int(generation):
                        self._completed = CollectionControlOperationResult(
                            generation=current.generation,
                            kind=current.kind,
                            started_at=current.started_at,
                            completed_at=current.completed_at,
                            error=current.error,
                            cancelled=True,
                            discard_error=exc,
                        )
        return True

    def close(self) -> None:
        value_to_discard: Any | None = None
        discard_value: Callable[[Any], None] | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pending is not None:
                self._pending.cancelled = True
            if self._completed is not None:
                if self._completed.error is None:
                    value_to_discard = self._completed.value
                    discard_value = self._completed_discard
                self._completed = None
                self._completed_discard = None
        if discard_value is not None:
            try:
                discard_value(value_to_discard)
            except BaseException:
                # Shutdown has no remaining publication channel.  Cleanup is
                # deliberately best-effort here; cancel()/poll() retain cleanup
                # failures when the runtime is still able to surface them.
                pass


__all__ = [
    "CollectionControlOperationBusy",
    "CollectionControlOperationResult",
    "CollectionControlPlane",
]
