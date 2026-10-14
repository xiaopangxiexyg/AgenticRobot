"""Native RealSense lifecycle serialization and retained stop ownership."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

# Native start/stop/reset edges share process state across camera consumers.
# Frame capture retains one owner per camera and does not take this lock.
SDK_LIFECYCLE_LOCK = threading.RLock()


def clear_native_tracebacks(error: BaseException) -> None:
    """Keep exception types/messages/causes without retaining SDK frame locals."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        for related in (current.__cause__, current.__context__):
            if related is not None:
                pending.append(related)


class NativePipelineStop:
    """Own one stop invocation and handle release until its worker exits.

    The caller must relinquish its pipeline reference before starting the
    worker, and retain this operation after a wait timeout. A timeout does not
    cancel a native call and must never authorize another start/reset.
    """

    def __init__(self, pipeline: Any, *, allow_not_started: bool = False,
                 before_stop: Callable[[], None] | None = None):
        self.started_at = time.monotonic()
        self.lock_acquired_at: float | None = None
        self.before_stop_started_at: float | None = None
        self.before_stop_finished_at: float | None = None
        self.sdk_started_at: float | None = None
        self.sdk_finished_at: float | None = None
        self.completed_at: float | None = None
        self.error: BaseException | None = None
        self.started = False
        self.allow_not_started = allow_not_started
        self.before_stop = before_stop
        self.worker = threading.Thread(
            target=self._stop_native, args=([pipeline],),
            name="realsense_pipeline_stop", daemon=True,
        )

    def start(self) -> None:
        try:
            self.worker.start()
            self.started = True
        except BaseException as exc:  # noqa: BLE001 - retain cleanup failure.
            clear_native_tracebacks(exc)
            self.error = exc
            self.completed_at = time.monotonic()

    def _stop_native(self, pipeline_holder: list[Any]) -> None:
        try:
            with SDK_LIFECYCLE_LOCK:
                self.lock_acquired_at = time.monotonic()
                pipeline = pipeline_holder.pop()
                try:
                    if self.before_stop is not None:
                        self.before_stop_started_at = time.monotonic()
                        self.before_stop()
                except BaseException as exc:  # noqa: BLE001 - still attempt native stop.
                    clear_native_tracebacks(exc)
                    self.error = exc
                finally:
                    if self.before_stop_started_at is not None:
                        self.before_stop_finished_at = time.monotonic()
                    self.before_stop = None
                self.sdk_started_at = time.monotonic()
                try:
                    pipeline.stop()
                except BaseException as exc:  # noqa: BLE001 - retain cleanup failure.
                    clear_native_tracebacks(exc)
                    # SDK reports this for a pipeline whose start failed before
                    # streaming. No other stop failure is safe to suppress.
                    if not (self.allow_not_started and isinstance(exc, RuntimeError)
                            and "stop() cannot be called before start()" in str(exc)):
                        self.error = (exc if self.error is None else
                                      RuntimeError(f"{self.error}; pipeline.stop: {exc}"))
                finally:
                    self.sdk_finished_at = time.monotonic()
                    pipeline = None
        except BaseException as exc:  # noqa: BLE001 - retain cleanup failure.
            clear_native_tracebacks(exc)
            self.error = exc
        finally:
            self.completed_at = time.monotonic()

    def wait(self, timeout_sec: float) -> BaseException | None:
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_sec must be finite and >= 0.")
        if self.started:
            self.worker.join(timeout=timeout)
        if self.worker.is_alive():
            return TimeoutError(f"native RealSense pipeline.stop did not finish within {timeout:.3f}s")
        return self.error

    def snapshot(self) -> dict[str, Any]:
        alive = self.worker.is_alive()
        phase = "finished"
        if alive:
            phase = ("waiting_for_lifecycle_lock" if self.lock_acquired_at is None else
                     "before_stop" if self.sdk_started_at is None else
                     "sdk_call" if self.sdk_finished_at is None else "releasing_native_handle")
        elif not self.started and self.error is not None:
            phase = "worker_start_failed"
        return {"worker_alive": alive, "phase": phase,
                "elapsed_sec": (self.completed_at or time.monotonic()) - self.started_at,
                "lock_wait_sec": None if self.lock_acquired_at is None else self.lock_acquired_at - self.started_at,
                "before_stop_sec": None if self.before_stop_started_at is None else
                    (self.before_stop_finished_at or time.monotonic()) - self.before_stop_started_at,
                "sdk_call_sec": None if self.sdk_started_at is None else
                    (self.sdk_finished_at or time.monotonic()) - self.sdk_started_at,
                "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}"}
