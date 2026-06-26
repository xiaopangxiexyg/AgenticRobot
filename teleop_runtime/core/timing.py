from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PhaseTiming:
    name: str
    elapsed_sec: float


class PhaseTimer:
    def __init__(self, *, clock: Callable[[], float] | None = None):
        self._clock = clock if clock is not None else time.perf_counter
        self._started_at = float(self._clock())
        self._last_at = self._started_at
        self._phases: list[PhaseTiming] = []

    def mark(self, name: str) -> float:
        now = float(self._clock())
        elapsed = now - self._last_at
        self._last_at = now
        self.record(name, elapsed)
        return elapsed

    def record(self, name: str, elapsed_sec: float) -> None:
        self._phases.append(PhaseTiming(str(name), float(elapsed_sec)))

    @property
    def phases(self) -> tuple[PhaseTiming, ...]:
        return tuple(self._phases)

    def as_tuples(self) -> tuple[tuple[str, float], ...]:
        return tuple((phase.name, phase.elapsed_sec) for phase in self._phases)

    def elapsed_sec(self) -> float:
        return float(self._clock()) - self._started_at
