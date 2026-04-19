from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..session import SessionStepResult
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents

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
        self._started = False
        self._submitted_samples = 0

    def start(self) -> None:
        if self._started:
            return
        if self.config.auto_start_writer_pool:
            self.writer_pool.start()
        if self.config.auto_start_session:
            self.components.session.start()
        self._started = True

    def step(self, now: float | None = None) -> SessionStepResult:
        if not self._started:
            self.start()
        result = self.components.session.run_step(now=now)
        sample = self.sample_builder(result, self.components)
        if sample is None:
            return result
        for job in self.write_planner(sample, self.components):
            self.writer_pool.submit(job)
        self._submitted_samples += 1
        return result

    def flush(self) -> None:
        self.writer_pool.join()

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
        return int(self._submitted_samples)

    def __enter__(self) -> "CollectionRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
