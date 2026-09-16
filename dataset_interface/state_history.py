"""Causal, thread-safe buffering for policy-state history inputs."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from dataset_contracts.state_history import (
    PolicyStateHistorySnapshot,
    PolicyStateHistorySpec,
)


@dataclass(frozen=True)
class _HistorySample:
    timestamp: float
    step_index: int
    continuity_segment: int
    state: tuple[float, ...]


class PolicyStateHistoryBuffer:
    """Retain exact control-tick samples for asynchronous request preparation.

    A step or timestamp discontinuity starts a new internal continuity segment.
    Requests can only select exact frame offsets within one segment.  Existing
    samples are immutable so a worker preparing an older request cannot observe
    a newer version of that request's history.
    """

    def __init__(
        self,
        spec: PolicyStateHistorySpec | None = None,
        *,
        retention_steps: int = 128,
    ) -> None:
        if spec is None:
            spec = PolicyStateHistorySpec()
        if not isinstance(spec, PolicyStateHistorySpec):
            raise TypeError("spec must be PolicyStateHistorySpec.")
        if int(retention_steps) < spec.capacity:
            raise ValueError("retention_steps must cover the largest frame offset.")
        self.spec = spec
        self._samples: deque[_HistorySample] = deque(maxlen=int(retention_steps))
        self._continuity_segment = 0
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._continuity_segment += 1

    def append(
        self,
        *,
        timestamp: float,
        step_index: int,
        state: np.ndarray,
    ) -> None:
        timestamp = float(timestamp)
        step_index = int(step_index)
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if not math.isfinite(timestamp):
            raise ValueError("state-history timestamp must be finite.")
        if step_index < 0:
            raise ValueError("state-history step index must be non-negative.")
        if values.shape != (self.spec.state_dim,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"state-history state must be finite with shape ({self.spec.state_dim},)."
            )
        immutable = tuple(float(value) for value in values)
        with self._lock:
            if self._samples:
                previous = self._samples[-1]
                if step_index == previous.step_index:
                    if timestamp == previous.timestamp and immutable == previous.state:
                        return
                    raise ValueError("one control step cannot have two state-history samples.")
                if step_index != previous.step_index + 1 or timestamp <= previous.timestamp:
                    self._samples.clear()
                    self._continuity_segment += 1
            self._samples.append(
                _HistorySample(
                    timestamp=timestamp,
                    step_index=step_index,
                    continuity_segment=self._continuity_segment,
                    state=immutable,
                )
            )

    def snapshot(
        self,
        *,
        anchor_timestamp: float,
        anchor_step_index: int,
    ) -> PolicyStateHistorySnapshot | None:
        anchor_timestamp = float(anchor_timestamp)
        anchor_step_index = int(anchor_step_index)
        with self._lock:
            by_step = {sample.step_index: sample for sample in self._samples}
            anchor = by_step.get(anchor_step_index)
            if anchor is None or not math.isclose(
                anchor.timestamp,
                anchor_timestamp,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                return None
            selected: list[_HistorySample] = []
            for offset in self.spec.frame_offsets:
                sample = by_step.get(anchor_step_index - offset)
                if sample is None or sample.continuity_segment != anchor.continuity_segment:
                    return None
                selected.append(sample)
            ages = tuple(anchor.timestamp - sample.timestamp for sample in selected)
            if any(age <= 0.0 for age in ages):
                return None
            return PolicyStateHistorySnapshot(
                frame_offsets=self.spec.frame_offsets,
                states=tuple(sample.state for sample in selected),
                ages_sec=ages,
                anchor_timestamp=anchor.timestamp,
                anchor_step_index=anchor.step_index,
                continuity_segment=anchor.continuity_segment,
            )

    def is_ready(self, *, anchor_timestamp: float, anchor_step_index: int) -> bool:
        return self.snapshot(
            anchor_timestamp=anchor_timestamp,
            anchor_step_index=anchor_step_index,
        ) is not None


__all__ = ["PolicyStateHistoryBuffer"]
