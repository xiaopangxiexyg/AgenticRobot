from __future__ import annotations

import queue
import threading
from typing import Any, Protocol

from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
    run_best_effort_cleanup,
)


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
        rollback_steps = []
        try:
            for index, observer in enumerate(self.observers):
                start = getattr(observer, "start", None)
                close = getattr(observer, "close", None)
                if callable(start):
                    if callable(close):
                        rollback_steps.append((f"observer[{index}]", close))
                    start()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    "composite_runtime_observer.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise

    def on_step(self, result: Any) -> None:
        for observer in self.observers:
            observer.on_step(result)

    def close(self) -> None:
        run_best_effort_cleanup(
            "composite_runtime_observer.close",
            (
                (f"observer[{index}]", close)
                for index, observer in reversed(tuple(enumerate(self.observers)))
                if callable(close := getattr(observer, "close", None))
            ),
        )


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
        close = getattr(self.observer, "close", None)
        observer_started = False
        try:
            if callable(start):
                start()
                observer_started = True
            self._closed = False
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=self.name,
                daemon=True,
            )
            self._thread.start()
        except BaseException as start_error:  # noqa: BLE001 - rollback observer startup.
            rollback_failures = collect_lifecycle_failures(
                (("observer", close),)
                if observer_started and callable(close)
                else ()
            )
            self._thread = None
            self._started = False
            if rollback_failures:
                raise LifecycleStartupError(
                    f"{self.name}.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
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
        failures = []
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
                worker = self._thread

                def join_worker() -> None:
                    worker.join(timeout=2.0)
                    if worker.is_alive():
                        raise RuntimeError(
                            f"{self.name} worker did not stop within 2.000s."
                        )

                failures.extend(
                    collect_lifecycle_failures((("worker.join", join_worker),))
                )
        close = getattr(self.observer, "close", None)
        if callable(close):
            failures.extend(
                collect_lifecycle_failures((("observer", close),))
            )
        failures.extend(
            collect_lifecycle_failures((("worker_failures", self.raise_if_failed),))
        )
        raise_lifecycle_failures(f"{self.name}.close", failures)
