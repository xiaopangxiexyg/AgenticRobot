from __future__ import annotations

import queue
import threading
import time
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

    def __post_init__(self) -> None:
        if int(self.worker_count) <= 0:
            raise ValueError("worker_count must be positive.")
        if int(self.max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive.")
        if float(self.put_timeout_sec) < 0.0:
            raise ValueError("put_timeout_sec must be >= 0.")
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
        self._failure_lock = threading.Lock()
        self._failures: list[BaseException] = []
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._queue_high_water = 0
        self._submit_wait_total_sec = 0.0
        self._submit_wait_max_sec = 0.0

    def _record_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            self._failures.append(exc)
        if self.config.fail_fast:
            self._closed = True

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            if not self._failures:
                return
            first = self._failures[0]
        raise AsyncWriterError(f"{self.name} failed: {first}") from first

    def start(self) -> None:
        if self._started:
            return
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
            rollback_failures = collect_lifecycle_failures(
                (("workers", lambda: self.close(wait=False)),)
            )
            if rollback_failures:
                raise LifecycleStartupError(
                    f"{self.name}.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
        self._started = True

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    def submit(self, job: WriteJob) -> bool:
        self.raise_if_failed()
        if self._closed:
            raise AsyncWriterError(f"{self.name} is closed and cannot accept new jobs.")
        self._ensure_started()
        policy = self.config.backpressure_policy
        submit_started_at = time.perf_counter()

        if policy == "raise":
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                raise AsyncWriterError(f"{self.name} queue is full.") from exc
        elif policy == "drop_noncritical" and not bool(job.critical):
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._dropped += 1
                return False
        else:
            try:
                self._queue.put(job, timeout=float(self.config.put_timeout_sec))
            except queue.Full as exc:
                raise AsyncWriterError(
                    f"{self.name} queue is full after waiting {self.config.put_timeout_sec:.3f}s."
                ) from exc

        submit_elapsed = time.perf_counter() - submit_started_at
        self._submit_wait_total_sec += float(submit_elapsed)
        self._submit_wait_max_sec = max(self._submit_wait_max_sec, float(submit_elapsed))
        self._submitted += 1
        self._queue_high_water = max(self._queue_high_water, int(self._queue.qsize()))
        return True

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
                    self._completed += 1
            finally:
                self._queue.task_done()

    def join(self) -> None:
        self._queue.join()
        self.raise_if_failed()

    def close(self, *, wait: bool = True) -> None:
        if self._closed and not self._started:
            return
        self._closed = True
        failures = []
        if wait:
            # Drain first, but do not surface handler failures until every worker
            # has received a stop signal and a bounded join attempt.
            failures.extend(
                collect_lifecycle_failures((("queue_drain", self._queue.join),))
            )

        stop_put_timeout = max(0.1, float(self.config.put_timeout_sec))
        workers_to_stop = [thread for thread in self._threads if thread.is_alive()]
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
                    for thread in reversed(self._threads)
                )
            )
        )
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        self._started = bool(self._threads)
        failures.extend(
            collect_lifecycle_failures((("handler_failures", self.raise_if_failed),))
        )
        raise_lifecycle_failures(f"{self.name}.close", failures)

    def stats(self) -> AsyncWriterStats:
        return AsyncWriterStats(
            queued=int(self._queue.qsize()),
            submitted=int(self._submitted),
            completed=int(self._completed),
            dropped=int(self._dropped),
            failures=len(self._failures),
            worker_count=int(self.config.worker_count),
            max_queue_size=int(self.config.max_queue_size),
            queue_high_water=int(self._queue_high_water),
            submit_wait_total_sec=float(self._submit_wait_total_sec),
            submit_wait_max_sec=float(self._submit_wait_max_sec),
        )

    def __enter__(self) -> "AsyncWriterPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(wait=True)
