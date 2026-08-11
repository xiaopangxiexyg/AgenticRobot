"""Per-control-domain linearization for receiver-side fact admission."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, TypeVar

from .identity import ClockDomainDescriptor


T = TypeVar("T")


class ControlAdmissionStore(Protocol[T]):
    def _insert_admitted(self, record: T) -> None:
        """Synchronously insert a fully built record or raise without mutation."""
        ...


class ControlAdmissionBatchStore(Protocol[T]):
    def _insert_admitted_batch(self, records: tuple[T, ...]) -> None:
        """Atomically insert a prebuilt record batch or raise without mutation."""

        ...


@dataclass(frozen=True)
class ControlCut:
    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    clock_reset_epoch: int
    cutoff_at: float
    cutoff_admission_sequence: int

    def __post_init__(self) -> None:
        if not str(self.runtime_session_id).strip():
            raise ValueError("runtime_session_id must be non-empty.")
        if not str(self.control_domain_id).strip():
            raise ValueError("control_domain_id must be non-empty.")
        if not str(self.clock_domain_id).strip():
            raise ValueError("clock_domain_id must be non-empty.")
        if int(self.clock_reset_epoch) < 0:
            raise ValueError("clock_reset_epoch must be non-negative.")
        if not math.isfinite(float(self.cutoff_at)):
            raise ValueError("cutoff_at must be finite.")
        if int(self.cutoff_admission_sequence) < 0:
            raise ValueError("cutoff_admission_sequence must be non-negative.")


@dataclass(frozen=True)
class ControlCutCapture:
    cut: ControlCut
    projections: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "projections", MappingProxyType(dict(self.projections)))


class ControlAdmissionCoordinator:
    """Commit facts and capture cuts under one short domain-local lock.

    The supplied commit callback must synchronously insert the fact into its
    target in-memory store.  The high-water mark advances only after that
    callback returns successfully.
    """

    def __init__(
        self,
        control_domain_id: str,
        *,
        clock_domain: ClockDomainDescriptor,
        clock: Callable[[], float] = time.monotonic,
    ):
        domain = str(control_domain_id).strip()
        if not domain:
            raise ValueError("control_domain_id must be non-empty.")
        self.control_domain_id = domain
        if clock_domain.runtime_session_id.strip() == "":
            raise ValueError("clock_domain runtime_session_id must be non-empty.")
        if not clock_domain.monotonic:
            raise ValueError("Control admission requires a monotonic local clock domain.")
        self.clock_domain = clock_domain
        self._clock = clock
        self._lock = threading.RLock()
        self._committed_high_watermark = 0

    @property
    def committed_high_watermark(self) -> int:
        with self._lock:
            return self._committed_high_watermark

    def commit(
        self,
        *,
        store: ControlAdmissionStore[T],
        build: Callable[[float, int], T],
    ) -> T:
        """Stamp and synchronously insert one fact, then publish its sequence."""

        with self._lock:
            sequence = self._committed_high_watermark + 1
            available_at = float(self._clock())
            if not math.isfinite(available_at):
                raise ValueError("Receiver monotonic clock returned a non-finite value.")
            record = build(available_at, sequence)
            store._insert_admitted(record)
            self._committed_high_watermark = sequence
            return record

    def commit_batch(
        self,
        *,
        store: ControlAdmissionBatchStore[T],
        count: int,
        build: Callable[[float, tuple[int, ...]], tuple[T, ...]],
    ) -> tuple[T, ...]:
        """Stamp and atomically publish one non-empty fact batch.

        Every record shares the same receiver ``available_at``. Admission
        sequences remain globally unique and consecutive, while cut capture
        cannot observe a strict subset because the high-water mark advances
        only after the store accepts the complete batch under this lock.
        """

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Control admission batch count must be positive.")
        with self._lock:
            first = self._committed_high_watermark + 1
            sequences = tuple(range(first, first + count))
            available_at = float(self._clock())
            if not math.isfinite(available_at):
                raise ValueError("Receiver monotonic clock returned a non-finite value.")
            records = tuple(build(available_at, sequences))
            if len(records) != count:
                raise ValueError(
                    "Control admission batch builder returned another record count."
                )
            store._insert_admitted_batch(records)
            self._committed_high_watermark = sequences[-1]
            return records

    def capture_cut(
        self,
        *,
        cutoff_at: float | None = None,
        projectors: Mapping[str, Callable[[ControlCut], Any]] | None = None,
    ) -> ControlCutCapture:
        """Atomically freeze a time+sequence cut and optional store projections."""

        with self._lock:
            timestamp = float(self._clock() if cutoff_at is None else cutoff_at)
            cut = ControlCut(
                runtime_session_id=self.clock_domain.runtime_session_id,
                control_domain_id=self.control_domain_id,
                clock_domain_id=self.clock_domain.clock_domain_id,
                clock_reset_epoch=self.clock_domain.reset_epoch,
                cutoff_at=timestamp,
                cutoff_admission_sequence=self._committed_high_watermark,
            )
            projections = {
                str(name): projector(cut)
                for name, projector in dict(projectors or {}).items()
            }
            return ControlCutCapture(cut=cut, projections=projections)


def admitted_at_or_before(record: Any, cut: ControlCut) -> bool:
    return (
        str(getattr(record, "control_domain_id")) == cut.control_domain_id
        and float(getattr(record, "available_at")) <= cut.cutoff_at
        and int(getattr(record, "control_admission_sequence"))
        <= cut.cutoff_admission_sequence
    )


__all__ = [
    "ControlAdmissionCoordinator",
    "ControlAdmissionBatchStore",
    "ControlAdmissionStore",
    "ControlCut",
    "ControlCutCapture",
    "admitted_at_or_before",
]
