from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class AsyncWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class WriteJob:
    kind: str
    payload: Any
    critical: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class AsyncWriterConfig:
    worker_count: int = 2
    max_queue_size: int = 256
    put_timeout_sec: float = 0.25
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if int(self.worker_count) <= 0:
            raise ValueError("worker_count must be positive.")
        if int(self.max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive.")
        if float(self.put_timeout_sec) < 0.0:
            raise ValueError("put_timeout_sec must be >= 0.")


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
        self._queue: queue.Queue[WriteJob | object] = queue.Queue(
            maxsize=int(self.config.max_queue_size)
        )
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._failure_lock = threading.Lock()
        self._failures: list[BaseException] = []

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
        for index in range(int(self.config.worker_count)):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.name}_{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self._started = True

    def submit(self, job: WriteJob) -> bool:
        self.raise_if_failed()
        if self._closed:
            raise AsyncWriterError(f"{self.name} is closed.")
        if not self._started:
            self.start()
        try:
            self._queue.put(job, timeout=float(self.config.put_timeout_sec))
        except queue.Full as exc:
            raise AsyncWriterError(f"{self.name} queue is full.") from exc
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
            finally:
                self._queue.task_done()

    def join(self) -> None:
        self._queue.join()
        self.raise_if_failed()

    def close(self, *, wait: bool = True) -> None:
        if self._closed and not self._started:
            return
        self._closed = True
        if wait:
            self.join()
        for _ in self._threads:
            self._queue.put(_SENTINEL)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        self._started = False
        self.raise_if_failed()
