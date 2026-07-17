"""Shared lifecycle boundary for collection sample writing.

Both the formal GUI teleop frontend and the headless compatibility frontend
produce :class:`~teleop_runtime.session.SessionStepResult` values.  They may
keep different episode UX and finalize policies, but the sample -> jobs ->
bounded writer pipeline must have one implementation and one lifecycle owner.

This class deliberately does not own a hardware session or an episode.  Those
are frontend concerns.  It owns only the reusable writer observer boundary so
future policy/human/replay frontends can attach the same recorder core without
creating another dispatch or serialization path.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from ..session import SessionStepResult
from .async_writer import AsyncWriterPool, WriteJob
from .build_collect_components import CollectorComponents
from .observers import CollectionWriterObserver, CollectionWriterObserverConfig


SampleBuilder = Callable[[SessionStepResult, CollectorComponents], Any | None]
WritePlanner = Callable[[Any, CollectorComponents], Iterable[WriteJob]]


class CollectionWriterRuntime:
    """Own the shared sample-writing lifecycle for a collection frontend.

    ``CollectionRuntime`` uses this object for its normal headless path, while
    the formal GUI recorder uses it as the writer core.  The public aliases
    ``writer_pool`` and ``writer_observer`` are intentional: existing status
    and diagnostics consumers can migrate independently of this extraction.
    """

    def __init__(
        self,
        *,
        components: CollectorComponents,
        sample_builder: SampleBuilder,
        write_planner: WritePlanner,
        writer_pool: AsyncWriterPool,
        config: CollectionWriterObserverConfig | None = None,
    ) -> None:
        self.writer_pool = writer_pool
        self.writer_observer = CollectionWriterObserver(
            components=components,
            sample_builder=sample_builder,
            write_planner=write_planner,
            writer_pool=writer_pool,
            config=config,
        )

    def start(self) -> None:
        self.writer_observer.start()

    def handle_step(
        self,
        result: SessionStepResult,
        *,
        phase_recorder: Callable[[str, float], None] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Any | None:
        return self.writer_observer.handle_step(
            result,
            phase_recorder=phase_recorder,
            metadata_extra=metadata_extra,
        )

    def flush(self) -> None:
        self.writer_observer.flush()

    def close(self, *, flush: bool = True) -> None:
        self.writer_observer.close(flush=bool(flush))

    @property
    def submitted_samples(self) -> int:
        return self.writer_observer.submitted_samples

    @property
    def submitted_jobs(self) -> int:
        return self.writer_observer.submitted_jobs

    @property
    def last_sample(self) -> Any | None:
        return self.writer_observer.last_sample

    def storage_stats(self) -> dict[str, Any]:
        return self.writer_observer.storage_stats()


__all__ = ["CollectionWriterRuntime"]
