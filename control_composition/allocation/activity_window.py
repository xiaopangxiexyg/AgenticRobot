"""Time-window retention for source activity samples."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .metrics import SourceActionActivitySample


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class SourceActionActivityWindowSnapshot:
    window_duration_sec: float
    cutoff_at: float
    clock_domain_id: str | None
    action_schema_hash: str | None
    samples: tuple[SourceActionActivitySample, ...]
    dropped_sample_count: int

    @property
    def ready(self) -> bool:
        return len(self.samples) >= 2


class SourceActionActivityWindow:
    """Keep only samples that can affect a declared recent-time window."""

    def __init__(self, *, window_duration_sec: float) -> None:
        duration = _finite(
            window_duration_sec,
            name="window_duration_sec",
        )
        if duration <= 0.0:
            raise ValueError("window_duration_sec must be positive.")
        self.window_duration_sec = duration
        self._lock = threading.RLock()
        self._samples: list[SourceActionActivitySample] = []
        self._sample_hash_by_id: dict[str, str] = {}
        self._clock_domain_id: str | None = None
        self._action_schema_hash: str | None = None
        self._dropped_sample_count = 0

    def offer(self, sample: SourceActionActivitySample) -> bool:
        """Insert one strictly newer sample; exact duplicates are idempotent."""

        if not isinstance(sample, SourceActionActivitySample):
            raise TypeError("sample must be SourceActionActivitySample.")
        schema_hash = sample.action.action_schema.semantic_content_hash
        with self._lock:
            existing_hash = self._sample_hash_by_id.get(sample.sample_id)
            if existing_hash is not None:
                if existing_hash != sample.semantic_content_hash:
                    raise ValueError("Activity sample ID was reused with another hash.")
                return False
            if self._clock_domain_id is None:
                self._clock_domain_id = sample.clock_domain_id
                self._action_schema_hash = schema_hash
            elif sample.clock_domain_id != self._clock_domain_id:
                raise ValueError("Activity window clock domain changed.")
            elif schema_hash != self._action_schema_hash:
                raise ValueError("Activity window action schema changed.")
            if self._samples and sample.sampled_at <= self._samples[-1].sampled_at:
                raise ValueError("Activity samples must arrive in increasing time order.")
            self._samples.append(sample)
            self._sample_hash_by_id[sample.sample_id] = sample.semantic_content_hash
            self._prune_locked(reference_at=sample.sampled_at)
            return True

    def _prune_locked(self, *, reference_at: float) -> None:
        retain_from = reference_at - self.window_duration_sec
        first_retained = 0
        while (
            first_retained < len(self._samples)
            and self._samples[first_retained].sampled_at < retain_from
        ):
            first_retained += 1
        if first_retained == 0:
            return
        removed = self._samples[:first_retained]
        self._samples = self._samples[first_retained:]
        for sample in removed:
            self._sample_hash_by_id.pop(sample.sample_id, None)
        self._dropped_sample_count += len(removed)

    def snapshot(self, *, cutoff_at: float) -> SourceActionActivityWindowSnapshot:
        """Read the exact samples in ``[cutoff-window, cutoff]``."""

        cutoff = _finite(cutoff_at, name="cutoff_at")
        retain_from = cutoff - self.window_duration_sec
        with self._lock:
            selected = tuple(
                sample
                for sample in self._samples
                if retain_from <= sample.sampled_at <= cutoff
            )
            return SourceActionActivityWindowSnapshot(
                window_duration_sec=self.window_duration_sec,
                cutoff_at=cutoff,
                clock_domain_id=self._clock_domain_id,
                action_schema_hash=self._action_schema_hash,
                samples=selected,
                dropped_sample_count=self._dropped_sample_count,
            )

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._sample_hash_by_id.clear()
            self._clock_domain_id = None
            self._action_schema_hash = None


__all__ = [
    "SourceActionActivityWindow",
    "SourceActionActivityWindowSnapshot",
]
