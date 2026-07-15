from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from robot_motion.runtime.lifecycle import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    run_best_effort_cleanup,
)

from ..session import SessionStepResult
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents
from .observers import CollectionWriterObserver, CollectionWriterObserverConfig

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
        writer_observer_config: CollectionWriterObserverConfig | None = None,
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
            config=writer_observer_config,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        rollback_steps: list[tuple[str, Callable[[], None]]] = []
        try:
            if self.config.auto_start_writer_pool:
                rollback_steps.append(
                    (
                        "writer_observer",
                        lambda: self.writer_observer.close(flush=False),
                    )
                )
                self.writer_observer.start()
            if self.config.auto_start_session:
                rollback_steps.append(("session", self.components.session.close))
                self.components.session.start()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            self._started = False
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    f"{self.config.name}.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise
        self._started = True

    def step(
        self,
        now: float | None = None,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> SessionStepResult:
        result = self.control_step(now=now, phase_recorder=phase_recorder)
        self.record_step(result, phase_recorder=phase_recorder)
        return result

    def control_step(
        self,
        now: float | None = None,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> SessionStepResult:
        """Run one control/reference update without constructing a dataset sample.

        Position control uses this at its reference rate (nominally 75 Hz).  Sampling
        is intentionally a separate operation so camera/state/action alignment can
        run on an independent 15 Hz clock instead of relying on a fixed 1:3 ratio.
        """
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
        return result

    def record_step(
        self,
        result: SessionStepResult,
        *,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> Any | None:
        """Build and enqueue one dataset sample from an already completed step."""
        return self.writer_observer.handle_step(result, phase_recorder=phase_recorder)

    def flush(self) -> None:
        self.writer_observer.flush()

    def close(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = [
            ("session", self.components.session.close),
            (
                "writer_observer",
                lambda: self.writer_observer.close(
                    flush=bool(self.config.flush_on_close)
                ),
            ),
        ]
        try:
            # Startup order is writer -> session, so teardown reverses it.  This
            # stops hardware first while still guaranteeing the writer receives
            # its drain/termination attempt if session cleanup fails.
            run_best_effort_cleanup(f"{self.config.name}.close", steps)
        finally:
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
