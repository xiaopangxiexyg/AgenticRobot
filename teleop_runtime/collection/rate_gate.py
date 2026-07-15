from __future__ import annotations

import math


class PeriodicRateGate:
    """Deadline-based rate gate for independently clocked runtime consumers.

    The gate advances from its previous deadline instead of from the time at which
    work happened.  Scheduler jitter therefore does not turn a nominal 15 Hz
    sampler into a fixed "every N control frames" policy or accumulate phase drift.
    When work is late, missed deadlines are skipped rather than replayed in a burst.
    """

    def __init__(self, hz: float):
        self.hz = float(hz)
        if not math.isfinite(self.hz) or self.hz <= 0.0:
            raise ValueError("hz must be finite and positive.")
        self._next_deadline: float | None = None

    @property
    def period_sec(self) -> float:
        return 1.0 / self.hz

    @property
    def next_deadline(self) -> float | None:
        return self._next_deadline

    def is_due(self, monotonic_time: float) -> bool:
        now = float(monotonic_time)
        if not math.isfinite(now):
            raise ValueError("monotonic_time must be finite.")
        if self._next_deadline is None:
            self._next_deadline = now + self.period_sec
            return True
        if now + 1e-12 < self._next_deadline:
            return False
        missed_periods = max(
            1,
            int(math.floor((now - self._next_deadline) / self.period_sec)) + 1,
        )
        self._next_deadline += missed_periods * self.period_sec
        return True

    def reset(self) -> None:
        self._next_deadline = None
