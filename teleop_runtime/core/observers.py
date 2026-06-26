from __future__ import annotations

import queue
import threading
from typing import Any, Protocol


class RuntimeObserver(Protocol):
    def start(self) -> None:
        ...

    def on_step(self, result: Any) -> None:
        ...

    def close(self) -> None:
        ...


class NullRuntimeObserver:
    def start(self) -> None:
        pass

    def on_step(self, result: Any) -> None:
        pass

    def close(self) -> None:
        pass


class CompositeRuntimeObserver:
    def __init__(self, observers: list[RuntimeObserver] | tuple[RuntimeObserver, ...]):
        self.observers = list(observers)

    def start(self) -> None:
        for observer in self.observers:
            start = getattr(observer, "start", None)
            if callable(start):
                start()

    def on_step(self, result: Any) -> None:
        for observer in self.observers:
            observer.on_step(result)

    def close(self) -> None:
        for observer in reversed(self.observers):
            close = getattr(observer, "close", None)
            if callable(close):
                close()


_SENTINEL = object()


class DropOldestAsyncObserver:
    """Best-effort observer wrapper for preview, writing, and status sinks."""

    def __init__(
        self,
        observer: RuntimeObserver,
        *,
        max_queue_size: int = 1,
        name: str = "runtime_observer",
    ):
        if int(max_queue_size) <= 0:
            raise ValueError("max_queue_size must be positive.")
        self.observer = observer
        self.name = str(name)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(max_queue_size))
        self._thread: threading.Thread | None = None
        self._closed = False
        self._started = False
        self._dropped = 0
        self._failures: list[BaseException] = []
        self._failure_lock = threading.Lock()

    @property
    def dropped(self) -> int:
        return int(self._dropped)

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            if not self._failures:
                return
            first = self._failures[0]
        raise RuntimeError(f"{self.name} failed: {first}") from first

    def start(self) -> None:
        if self._started:
            return
        start = getattr(self.observer, "start", None)
        if callable(start):
            start()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=self.name,
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                try:
                    self.observer.on_step(item)
                except BaseException as exc:  # noqa: BLE001
                    with self._failure_lock:
                        self._failures.append(exc)
            finally:
                self._queue.task_done()

    def on_step(self, result: Any) -> None:
        if self._closed:
            return
        if not self._started:
            self.start()
        try:
            self._queue.put_nowait(result)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self._queue.task_done()
            self._dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        close = getattr(self.observer, "close", None)
        if callable(close):
            close()
