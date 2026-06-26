from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..session import SessionStepResult
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents
from .observers import CollectionWriterObserver

SampleBuilder = Callable[[SessionStepResult, CollectorComponents], Any | None]
WritePlanner = Callable[[Any, CollectorComponents], Iterable[WriteJob]]


@dataclass
class CollectionRuntimeConfig:
    name: str = "collection_runtime"
    auto_start_session: bool = True
    auto_start_writer_pool: bool = True
    flush_on_close: bool = True


class CollectionRuntime:
    def __init__(
        self,
        *,
        components: CollectorComponents,
        sample_builder: SampleBuilder,
        write_planner: WritePlanner,
        writer_pool: AsyncWriterPool,
        config: CollectionRuntimeConfig | None = None,
    ):
        self.components = components
        self.sample_builder = sample_builder
        self.write_planner = write_planner
        self.writer_pool = writer_pool
        self.config = config if config is not None else CollectionRuntimeConfig()
        self.writer_observer = CollectionWriterObserver(
            components=self.components,
            sample_builder=self.sample_builder,
            write_planner=self.write_planner,
            writer_pool=self.writer_pool,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if self.config.auto_start_writer_pool:
            self.writer_observer.start()
        if self.config.auto_start_session:
            self.components.session.start()
        self._started = True

    def step(
        self,
        now: float | None = None,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> SessionStepResult:
        phase_started = time.perf_counter()

        def mark_phase(name: str) -> None:
            nonlocal phase_started
            if phase_recorder is None:
                return
            current = time.perf_counter()
            phase_recorder(name, current - phase_started)
            phase_started = current

        if not self._started:
            self.start()
            mark_phase("runtime.start")
        result = self.components.session.run_step(
            now=now,
            phase_recorder=phase_recorder,
        )
        mark_phase("runtime.session.run_step")
        self.writer_observer.handle_step(result, phase_recorder=phase_recorder)
        return result

    def flush(self) -> None:
        self.writer_observer.flush()

    def close(self) -> None:
        try:
            if self.config.flush_on_close:
                self.flush()
        finally:
            self.writer_pool.close(wait=False if not self.config.flush_on_close else True)
            self.components.session.close()
            self._started = False

    @property
    def submitted_samples(self) -> int:
        return self.writer_observer.submitted_samples

    @property
    def last_sample(self) -> Any | None:
        return self.writer_observer.last_sample

    def __enter__(self) -> "CollectionRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
