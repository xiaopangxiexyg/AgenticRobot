from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..session import SessionStepResult
from .action_record import build_collection_action_record
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents


SampleBuilder = Callable[[SessionStepResult, CollectorComponents], Any | None]
WritePlanner = Callable[[Any, CollectorComponents], Iterable[WriteJob]]


@dataclass(frozen=True)
class CollectionWriterObserverConfig:
    name: str = "collection_writer_observer"


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

    def _ensure_action_record(self, result: SessionStepResult) -> None:
        if result.action_record is None:
            result.action_record = build_collection_action_record(result)

    def handle_step(
        self,
        result: SessionStepResult,
        *,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> Any | None:
        phase_started = time.perf_counter()

        def mark_phase(name: str) -> None:
            nonlocal phase_started
            if phase_recorder is None:
                return
            current = time.perf_counter()
            phase_recorder(name, current - phase_started)
            phase_started = current

        self._ensure_action_record(result)
        mark_phase("runtime.action_record")

        self._last_sample = None
        sample = self.sample_builder(result, self.components)
        mark_phase("runtime.sample_builder")
        if sample is None:
            return None
        self._last_sample = sample

        if phase_recorder is None:
            for job in self.write_planner(sample, self.components):
                if self.writer_pool.submit(job):
                    self._submitted_jobs += 1
        else:
            jobs = list(self.write_planner(sample, self.components))
            mark_phase("runtime.write_planner")
            for job in jobs:
                if self.writer_pool.submit(job):
                    self._submitted_jobs += 1
            mark_phase("runtime.writer_submit")
        self._submitted_samples += 1
        return sample

    def on_step(self, result: SessionStepResult) -> None:
        self.handle_step(result)

    def flush(self) -> None:
        self.writer_pool.join()

    def close(self, *, flush: bool = True) -> None:
        if flush:
            self.flush()
        self.writer_pool.close(wait=flush)
