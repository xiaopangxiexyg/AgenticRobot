from __future__ import annotations


class PreviewRateLimiter:
    def __init__(self, max_hz: float):
        self.max_hz = float(max_hz)
        if self.max_hz <= 0.0:
            raise ValueError("max_hz must be positive.")
        self._last_update_at: float | None = None

    @property
    def min_interval_sec(self) -> float:
        return 1.0 / self.max_hz

    def should_update(self, now: float) -> bool:
        current = float(now)
        if self._last_update_at is None:
            self._last_update_at = current
            return True
        if current - self._last_update_at + 1e-12 < self.min_interval_sec:
            return False
        self._last_update_at = current
        return True

    def reset(self) -> None:
        self._last_update_at = None
