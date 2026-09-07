from __future__ import annotations

import queue
import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)


class AsyncWriterError(RuntimeError):
    """Raised when the async writer pipeline cannot accept or process jobs."""


@dataclass(frozen=True)
class WriteJob:
    kind: str
    payload: Any
    critical: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class AsyncWriterConfig:
    worker_count: int = 4
    max_queue_size: int = 512
    put_timeout_sec: float = 0.25
    backpressure_policy: str = "block"
    fail_fast: bool = True
    join_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if int(self.worker_count) <= 0:
            raise ValueError("worker_count must be positive.")
        if int(self.max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive.")
        if not math.isfinite(float(self.put_timeout_sec)) or float(
            self.put_timeout_sec
        ) < 0.0:
            raise ValueError("put_timeout_sec must be >= 0.")
        if not math.isfinite(float(self.join_timeout_sec)) or float(
            self.join_timeout_sec
        ) <= 0.0:
            raise ValueError("join_timeout_sec must be positive.")
        if self.backpressure_policy not in {"block", "raise", "drop_noncritical"}:
            raise ValueError(
                "backpressure_policy must be one of: block, raise, drop_noncritical."
            )


@dataclass(frozen=True)
class AsyncWriterStats:
    queued: int
    submitted: int
    completed: int
    dropped: int
    failures: int
    worker_count: int
    max_queue_size: int
    queue_high_water: int
    submit_wait_total_sec: float
    submit_wait_max_sec: float


_QueueItem = WriteJob | object
_SENTINEL = object()


class AsyncWriterPool:
    def __init__(
        self,
        handler: Callable[[WriteJob], None],
        *,
        config: AsyncWriterConfig | None = None,
        name: str = "async_writer",
    ):
        self._handler = handler
        self.config = config if config is not None else AsyncWriterConfig()
        self.name = str(name)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=self.config.max_queue_size)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        # The writer is shared by the normal row path and optional per-camera
        # storage routes.  Serialize lifecycle transitions and reserve active
        # submissions so close cannot publish stop sentinels while a producer
        # is still entering the queue.
        self._lifecycle_lock = threading.RLock()
        self._admission_condition = threading.Condition(self._lifecycle_lock)
        self._active_submissions = 0
        self._close_in_progress = False
        self._failure_lock = threading.Lock()
        # Queue operations are thread-safe, but the diagnostic counters below
        # are updated by submitters and worker threads concurrently.  Keep one
        # small lock for a coherent stats snapshot and to avoid lost updates
        # during fail-fast queue eviction.
        self._counter_lock = threading.Lock()
        self._failures: list[BaseException] = []
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._queue_high_water = 0
        self._submit_wait_total_sec = 0.0
        self._submit_wait_max_sec = 0.0

    def _record_failure(self, exc: BaseException) -> None:
        with self._lifecycle_lock:
            with self._failure_lock:
                self._failures.append(exc)
            if self.config.fail_fast:
                self._closed = True
                self._admission_condition.notify_all()
        if self.config.fail_fast:
            # Once one writer handler has failed, queued jobs cannot be
            # considered durable: continuing to submit them either repeats
            # the same failure for every job or keeps shutdown waiting behind
            # a permanently unhealthy sink.  Remove only jobs that have not
            # started yet; the worker that called us still completes its own
            # ``task_done`` in ``_worker_loop``.
            self._discard_pending_jobs()

    def _discard_pending_jobs(self) -> int:
        """Drop queued jobs while preserving ``Queue.unfinished_tasks``.

        This method is intentionally best-effort and idempotent.  It is used
        from both the failing worker and the bounded close path, so it must
        never raise merely because another worker drained the queue first.
        """

        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                # Keep a stop marker available for the close path.  Removing
                # and re-inserting it leaves the queue accounting unchanged.
                self._queue.task_done()
                try:
                    self._queue.put_nowait(_SENTINEL)
                except queue.Full:
                    # A concurrent producer cannot legally enqueue after the
                    # pool is closed; if it raced us, dropping the marker is
                    # safe because ``close`` will add another one.
                    pass
                break
            self._queue.task_done()
            dropped += 1
        if dropped:
            with self._counter_lock:
                self._dropped += dropped
        return dropped

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            if not self._failures:
                return
            first = self._failures[0]
        raise AsyncWriterError(f"{self.name} failed: {first}") from first

    def start(self) -> None:
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._started:
            if any(thread.is_alive() for thread in self._threads):
                return
            # The pool may have lost all workers without a normal close.  Do
            # not enqueue into a queue that no thread will drain.
            self._threads = []
            self._started = False
            if not self._closed:
                raise AsyncWriterError(f"{self.name} workers exited unexpectedly.")
        if self._closed:
            # Preserve the pool's historical clean close/start reuse while
            # refusing to resurrect a fail-fast pool whose handler already
            # failed.  A route queue remains one-shot; this compatibility is
            # only for direct AsyncWriterPool users.
            with self._failure_lock:
                failed = bool(self._failures)
            if failed:
                raise AsyncWriterError(f"{self.name} is closed after a handler failure.")
            self._closed = False
        self._threads = []
        try:
            for index in range(int(self.config.worker_count)):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.name}_{index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except BaseException as start_error:  # noqa: BLE001 - stop partial worker set.
            self._started = bool(self._threads)
            partial_threads = tuple(self._threads)
            # This helper does not re-enter close() while the lifecycle lock is
            # held.  It still sends bounded stop markers and joins any workers
            # that were successfully created before the failure.
            rollback_failures = collect_lifecycle_failures(
                (
                    (
                        "workers",
                        lambda: self._rollback_partial_start(partial_threads),
                    ),
                )
            )
            if rollback_failures:
                raise LifecycleStartupError(
                    f"{self.name}.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
        self._started = True

    def _rollback_partial_start(self, threads: tuple[threading.Thread, ...]) -> None:
        self._closed = True
        self._discard_pending_jobs()
        for thread in threads:
            if thread.is_alive():
                try:
                    # A queue smaller than the worker count may briefly be
                    # full with the previous worker's sentinel.  Wait a
                    # bounded interval for that worker to consume it so every
                    # partial worker receives its own stop marker.
                    self._queue.put(
                        _SENTINEL,
                        timeout=max(0.1, float(self.config.put_timeout_sec)),
                    )
                except queue.Full:
                    # Keep the lifecycle failure visible; close() can retry
                    # the stop signal once the startup lock is released.
                    raise RuntimeError(
                        f"{self.name} could not enqueue startup rollback sentinel"
                    )
        # ``_start_locked`` is invoked while the lifecycle lock is held.  Do
        # not join here: a worker that reports a startup/handler failure may
        # need that same lock in ``_record_failure``.  The sentinels make the
        # partial workers terminate promptly; a later close() call performs
        # the bounded join after the lock is released.
        self._threads = list(threads)
        self._started = bool(self._threads)

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    def submit(self, job: WriteJob) -> bool:
        self.raise_if_failed()
        with self._lifecycle_lock:
            if self._closed:
                raise AsyncWriterError(f"{self.name} is closed and cannot accept new jobs.")
            self._start_locked()
            self._active_submissions += 1
        policy = self.config.backpressure_policy
        submit_started_at = time.perf_counter()
        try:
            if policy == "raise":
                try:
                    self._queue.put_nowait(job)
                except queue.Full as exc:
                    raise AsyncWriterError(f"{self.name} queue is full.") from exc
            elif policy == "drop_noncritical" and not bool(job.critical):
                try:
                    self._queue.put_nowait(job)
                except queue.Full:
                    with self._counter_lock:
                        self._dropped += 1
                    return False
            else:
                timeout = float(self.config.put_timeout_sec)
                deadline = time.monotonic() + timeout
                while True:
                    self.raise_if_failed()
                    with self._lifecycle_lock:
                        if self._closed:
                            raise AsyncWriterError(
                                f"{self.name} is closed and cannot accept new jobs."
                            )
                    try:
                        if timeout <= 0.0:
                            self._queue.put_nowait(job)
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0.0:
                                raise queue.Full
                            self._queue.put(job, timeout=min(0.05, remaining))
                        break
                    except queue.Full as exc:
                        if timeout <= 0.0 or time.monotonic() >= deadline:
                            raise AsyncWriterError(
                                f"{self.name} queue is full after waiting {timeout:.3f}s."
                            ) from exc

            submit_elapsed = time.perf_counter() - submit_started_at
            with self._counter_lock:
                self._submit_wait_total_sec += float(submit_elapsed)
                self._submit_wait_max_sec = max(
                    self._submit_wait_max_sec, float(submit_elapsed)
                )
                self._submitted += 1
                self._queue_high_water = max(
                    self._queue_high_water, int(self._queue.qsize())
                )
            return True
        finally:
            with self._lifecycle_lock:
                self._active_submissions -= 1
                self._admission_condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                assert isinstance(item, WriteJob)
                try:
                    self._handler(item)
                except BaseException as exc:  # noqa: BLE001
                    self._record_failure(exc)
                else:
                    with self._counter_lock:
                        self._completed += 1
            finally:
                self._queue.task_done()

    def _wait_for_queue_drain(self, timeout_sec: float | None = None) -> None:
        timeout = (
            float(self.config.join_timeout_sec)
            if timeout_sec is None
            else float(timeout_sec)
        )
        if timeout <= 0.0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if self.config.fail_fast:
                with self._failure_lock:
                    failed = bool(self._failures)
                if failed:
                    self._discard_pending_jobs()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise AsyncWriterError(
                    f"{self.name} queue did not drain within {timeout:.3f}s "
                    f"(unfinished={self._queue.unfinished_tasks})."
                )
            # ``Queue.join`` has no timeout.  Poll the task counter in short
            # intervals so a stalled handler cannot wedge collection shutdown
            # or the GUI forever; worker threads remain intact for the normal
            # close path and are handled by the bounded join below.
            time.sleep(min(0.01, remaining))

    def join(self, *, timeout_sec: float | None = None) -> None:
        self._wait_for_queue_drain(timeout_sec)
        self.raise_if_failed()

    def close(self, *, wait: bool = True) -> None:
        with self._lifecycle_lock:
            while self._close_in_progress:
                self._admission_condition.wait(timeout=0.05)
            if self._closed and not self._started and not self._threads:
                return
            self._closed = True
            self._close_in_progress = True
            while self._active_submissions:
                self._admission_condition.wait(timeout=0.05)
            workers_snapshot = tuple(self._threads)
        failures = []
        try:
            if wait:
                # Drain first, but do not surface handler failures until every worker
                # has received a stop signal and a bounded join attempt.
                failures.extend(
                    collect_lifecycle_failures(
                        (
                            (
                                "queue_drain",
                                lambda: self._wait_for_queue_drain(
                                    self.config.join_timeout_sec
                                ),
                            ),
                        )
                    )
                )
            else:
                # ``wait=False`` is the rollback/abort path.  It must not spend
                # the caller's latency budget replaying jobs after a failed
                # handler; active jobs are still joined below, while queued jobs
                # are explicitly marked done and counted as dropped.
                self._discard_pending_jobs()

            with self._failure_lock:
                has_failures = bool(self._failures)
            if has_failures:
                self._discard_pending_jobs()

            stop_put_timeout = max(0.1, float(self.config.put_timeout_sec))
            workers_to_stop = [thread for thread in workers_snapshot if thread.is_alive()]
            failures.extend(
                collect_lifecycle_failures(
                    (
                        (
                            f"worker[{index}].stop_signal",
                            lambda timeout=stop_put_timeout: self._queue.put(
                                _SENTINEL,
                                timeout=timeout,
                            ),
                        )
                        for index, _thread in enumerate(workers_to_stop)
                    )
                )
            )

            def join_worker(thread: threading.Thread) -> None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    raise AsyncWriterError(
                        f"{self.name} worker {thread.name!r} did not stop within 1.000s."
                    )

            failures.extend(
                collect_lifecycle_failures(
                    (
                        (
                            f"worker[{thread.name}].join",
                            lambda thread=thread: join_worker(thread),
                        )
                        for thread in reversed(workers_snapshot)
                    )
                )
            )
            failures.extend(
                collect_lifecycle_failures((("handler_failures", self.raise_if_failed),))
            )
        finally:
            with self._lifecycle_lock:
                self._threads = [thread for thread in self._threads if thread.is_alive()]
                self._started = bool(self._threads)
                self._close_in_progress = False
                self._admission_condition.notify_all()
        raise_lifecycle_failures(f"{self.name}.close", failures)

    def stats(self) -> AsyncWriterStats:
        with self._counter_lock:
            submitted = int(self._submitted)
            completed = int(self._completed)
            dropped = int(self._dropped)
            queue_high_water = int(self._queue_high_water)
            submit_wait_total_sec = float(self._submit_wait_total_sec)
            submit_wait_max_sec = float(self._submit_wait_max_sec)
        with self._failure_lock:
            failures = len(self._failures)
        return AsyncWriterStats(
            queued=int(self._queue.qsize()),
            submitted=submitted,
            completed=completed,
            dropped=dropped,
            failures=failures,
            worker_count=int(self.config.worker_count),
            max_queue_size=int(self.config.max_queue_size),
            queue_high_water=queue_high_water,
            submit_wait_total_sec=submit_wait_total_sec,
            submit_wait_max_sec=submit_wait_max_sec,
        )

    def __enter__(self) -> "AsyncWriterPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(wait=True)
