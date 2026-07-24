"""Time-window retention contracts for receiver-admitted composition facts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


TIME_WINDOW_RETENTION_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.composition_time_window_retention_spec.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class TimeWindowRetentionSpec:
    """Bound receiver memory by declared time semantics, not record count.

    ``history_window_sec`` is the maximum past interval that a strategy may
    request after the closed target watermark. ``max_future_lead_sec`` and
    ``max_effective_duration_sec`` bound how far one admitted fact can extend
    into the future. ``max_late_arrival_sec`` is normally zero for real-time
    profiles and may be positive only for an explicitly logical/replay clock.
    """

    spec_id: str
    history_window_sec: float
    max_future_lead_sec: float
    max_effective_duration_sec: float
    max_late_arrival_sec: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _required(self.spec_id, name="spec_id"))
        object.__setattr__(
            self,
            "history_window_sec",
            _nonnegative(self.history_window_sec, name="history_window_sec"),
        )
        object.__setattr__(
            self,
            "max_future_lead_sec",
            _nonnegative(self.max_future_lead_sec, name="max_future_lead_sec"),
        )
        object.__setattr__(
            self,
            "max_effective_duration_sec",
            _positive(
                self.max_effective_duration_sec,
                name="max_effective_duration_sec",
            ),
        )
        object.__setattr__(
            self,
            "max_late_arrival_sec",
            _nonnegative(self.max_late_arrival_sec, name="max_late_arrival_sec"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "history_window_sec": self.history_window_sec,
                "max_future_lead_sec": self.max_future_lead_sec,
                "max_effective_duration_sec": self.max_effective_duration_sec,
                "max_late_arrival_sec": self.max_late_arrival_sec,
            },
            domain=TIME_WINDOW_RETENTION_SPEC_HASH_DOMAIN_V1,
        )

    def validate_admitted_interval(
        self,
        *,
        available_at: float,
        effective_from: float,
        effective_until: float,
    ) -> None:
        available = float(available_at)
        start = float(effective_from)
        end = float(effective_until)
        if not all(math.isfinite(value) for value in (available, start, end)):
            raise ValueError("Retention interval times must be finite.")
        if end <= start:
            raise ValueError("Effective interval must be non-empty.")
        if start > available + self.max_future_lead_sec:
            raise ValueError(
                "effective_from exceeds the retention profile future-lead bound."
            )
        if end - start > self.max_effective_duration_sec:
            raise ValueError(
                "Effective interval exceeds the retention profile duration bound."
            )
        if end <= available - self.max_late_arrival_sec:
            raise ValueError(
                "Effective interval expired before receiver admission."
            )

    def prune_before(self, *, closed_target_through: float) -> float:
        target = float(closed_target_through)
        if not math.isfinite(target):
            raise ValueError("closed_target_through must be finite.")
        return target - self.history_window_sec


@dataclass(frozen=True)
class TimeWindowStoreSnapshot:
    retention_spec_hash: str
    closed_target_through: float | None
    retained_record_count: int
    pruned_record_count: int
    producer_epoch_tombstone_count: int


__all__ = [
    "TIME_WINDOW_RETENTION_SPEC_HASH_DOMAIN_V1",
    "TimeWindowRetentionSpec",
    "TimeWindowStoreSnapshot",
]
