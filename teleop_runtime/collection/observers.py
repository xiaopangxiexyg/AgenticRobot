from __future__ import annotations

import time
import queue
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from robot_motion.runtime.lifecycle import (
    collect_lifecycle_failures,
    raise_lifecycle_failures,
    run_best_effort_cleanup,
)

from ..session import SessionStepResult
from .action_record import build_collection_action_record
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents


SampleBuilder = Callable[[SessionStepResult, CollectorComponents], Any | None]
WritePlanner = Callable[[Any, CollectorComponents], Iterable[WriteJob]]


@dataclass(frozen=True)
class CollectionWriterObserverConfig:
    name: str = "collection_writer_observer"
    storage_queue_size: int = 0

    def __post_init__(self) -> None:
        if int(self.storage_queue_size) < 0:
            raise ValueError("storage_queue_size must be >= 0.")


@dataclass(frozen=True)
class StorageRouteStats:
    key: str
    queued: int
    enqueued: int
    submitted: int
    dropped: int
    high_water: int
    max_queue_size: int


_StorageItem = WriteJob | object
_STORAGE_SENTINEL = object()


class _StorageRouteQueue:
    def __init__(
        self,
        *,
        key: str,
        writer_pool: AsyncWriterPool,
        max_queue_size: int,
    ):
        self.key = str(key)
        self.writer_pool = writer_pool
        self.max_queue_size = int(max_queue_size)
        self._queue: queue.Queue[_StorageItem] = queue.Queue(maxsize=self.max_queue_size)
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._failure_lock = threading.Lock()
        self._failures: list[BaseException] = []
        self._enqueued = 0
        self._submitted = 0
        self._dropped = 0
        self._high_water = 0

    def start(self) -> None:
        if self._started:
            return
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"storage_queue_{self.key}",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _record_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            self._failures.append(exc)
        self._closed = True

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            if not self._failures:
                return
            first = self._failures[0]
        raise RuntimeError(f"storage queue '{self.key}' failed: {first}") from first

    def enqueue(self, job: WriteJob) -> bool:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError(f"storage queue '{self.key}' is closed.")
        self.start()
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._queue.task_done()
                self._dropped += 1
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._dropped += 1
                return False
        self._enqueued += 1
        self._high_water = max(self._high_water, int(self._queue.qsize()))
        return True

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STORAGE_SENTINEL:
                    return
                assert isinstance(item, WriteJob)
                try:
                    if self.writer_pool.submit(item):
                        self._submitted += 1
                except BaseException as exc:  # noqa: BLE001
                    self._record_failure(exc)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        self._queue.join()
        self.raise_if_failed()

    def close(self, *, wait: bool = True) -> None:
        if self._closed and not self._started:
            return
        self._closed = True
        failures = []
        if wait:
            failures.extend(
                collect_lifecycle_failures((("queue_drain", self._queue.join),))
            )
        if self._started and self._thread is not None and self._thread.is_alive():
            while True:
                try:
                    self._queue.put_nowait(_STORAGE_SENTINEL)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        continue
                    self._queue.task_done()
                    self._dropped += 1
        if self._thread is not None:
            worker = self._thread

            def join_worker() -> None:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    raise RuntimeError(
                        f"storage queue '{self.key}' worker did not stop within 1.000s."
                    )

            failures.extend(
                collect_lifecycle_failures(
                    ((f"worker[{worker.name}].join", join_worker),)
                )
            )
            if not worker.is_alive():
                self._thread = None
        self._started = self._thread is not None
        failures.extend(
            collect_lifecycle_failures((("submit_failures", self.raise_if_failed),))
        )
        raise_lifecycle_failures(f"storage_queue[{self.key}].close", failures)

    def stats(self) -> StorageRouteStats:
        return StorageRouteStats(
            key=self.key,
            queued=int(self._queue.qsize()),
            enqueued=int(self._enqueued),
            submitted=int(self._submitted),
            dropped=int(self._dropped),
            high_water=int(self._high_water),
            max_queue_size=int(self.max_queue_size),
        )


class CollectionWriterObserver:
    """Build and submit collection samples from a session step result."""

    def __init__(
        self,
        *,
        components: CollectorComponents,
        sample_builder: SampleBuilder,
        write_planner: WritePlanner,
        writer_pool: AsyncWriterPool,
        config: CollectionWriterObserverConfig | None = None,
    ):
        self.components = components
        self.sample_builder = sample_builder
        self.write_planner = write_planner
        self.writer_pool = writer_pool
        self.config = config if config is not None else CollectionWriterObserverConfig()
        self._submitted_samples = 0
        self._submitted_jobs = 0
        self._last_sample: Any | None = None
        self._storage_routes: dict[str, _StorageRouteQueue] = {}

    @property
    def submitted_samples(self) -> int:
        return int(self._submitted_samples)

    @property
    def submitted_jobs(self) -> int:
        return int(self._submitted_jobs)

    @property
    def last_sample(self) -> Any | None:
        return self._last_sample

    def start(self) -> None:
        self.writer_pool.start()

    @property
    def storage_queue_enabled(self) -> bool:
        return int(self.config.storage_queue_size) > 0

    def _storage_route_key(self, job: WriteJob) -> str | None:
        if not self.storage_queue_enabled:
            return None
        storage_queue = str(job.metadata.get("storage_queue", ""))
        if storage_queue not in {"video", "wrist_depth"}:
            return None
        camera_key = job.metadata.get("camera_key")
        if camera_key is None and isinstance(job.payload, dict):
            camera_key = job.payload.get("camera_key")
        if camera_key is None:
            return None
        if storage_queue == "video":
            return str(camera_key)
        return f"{storage_queue}:{camera_key}"

    def _storage_route(self, key: str) -> _StorageRouteQueue:
        route_key = str(key)
        route = self._storage_routes.get(route_key)
        if route is None:
            route = _StorageRouteQueue(
                key=route_key,
                writer_pool=self.writer_pool,
                max_queue_size=int(self.config.storage_queue_size),
            )
            self._storage_routes[route_key] = route
        return route

    def _submit_job(self, job: WriteJob) -> bool:
        route_key = self._storage_route_key(job)
        if route_key is not None:
            return self._storage_route(route_key).enqueue(job)
        return self.writer_pool.submit(job)

    def _ensure_action_record(self, result: SessionStepResult) -> None:
        if result.action_record is None:
            result.action_record = build_collection_action_record(result)

    def handle_step(
        self,
        result: SessionStepResult,
        *,
        phase_recorder: Callable[[str, float], None] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Any | None:
        phase_started = time.perf_counter()
        local_phases: list[tuple[str, float]] = []

        def mark_phase(name: str) -> None:
            nonlocal phase_started
            current = time.perf_counter()
            elapsed = current - phase_started
            local_phases.append((str(name), float(elapsed)))
            if phase_recorder is not None:
                phase_recorder(name, elapsed)
            phase_started = current

        self._ensure_action_record(result)
        mark_phase("runtime.action_record")

        self._last_sample = None
        sample = self.sample_builder(result, self.components)
        mark_phase("runtime.sample_builder")
        if sample is None:
            return None
        self._last_sample = sample
        phase_timings = {name: elapsed * 1000.0 for name, elapsed in local_phases}
        if hasattr(sample, "metadata"):
            sample.metadata.setdefault("runtime_phase_timings_ms", {}).update(phase_timings)
        elif isinstance(sample, dict):
            sample.setdefault("metadata", {}).setdefault("runtime_phase_timings_ms", {}).update(phase_timings)
        if metadata_extra:
            if hasattr(sample, "metadata"):
                sample.metadata.update(dict(metadata_extra))
            elif isinstance(sample, dict):
                sample.setdefault("metadata", {}).update(dict(metadata_extra))

        if phase_recorder is None:
            for job in self.write_planner(sample, self.components):
                if self._submit_job(job):
                    self._submitted_jobs += 1
        else:
            jobs = list(self.write_planner(sample, self.components))
            mark_phase("runtime.write_planner")
            for job in jobs:
                if self._submit_job(job):
                    self._submitted_jobs += 1
            mark_phase("runtime.writer_submit")
        self._submitted_samples += 1
        return sample

    def on_step(self, result: SessionStepResult) -> None:
        self.handle_step(result)

    def flush(self) -> None:
        steps = [
            (f"storage_route[{key}]", route.flush)
            for key, route in self._storage_routes.items()
        ]
        steps.append(("writer_pool", self.writer_pool.join))
        run_best_effort_cleanup(f"{self.config.name}.flush", steps)

    def close(self, *, flush: bool = True) -> None:
        steps = [
            (
                f"storage_route[{key}]",
                lambda route=route: route.close(wait=flush),
            )
            for key, route in reversed(tuple(self._storage_routes.items()))
        ]
        steps.append(("writer_pool", lambda: self.writer_pool.close(wait=flush)))
        run_best_effort_cleanup(f"{self.config.name}.close", steps)

    def storage_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.storage_queue_enabled,
            "storage_queue_size": int(self.config.storage_queue_size),
            "routes": {
                key: asdict(route.stats())
                for key, route in sorted(self._storage_routes.items())
            },
        }
