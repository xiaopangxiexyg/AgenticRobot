"""Bounded non-blocking ingress used by composition shadow observers."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from .policy_human_shadow_contracts import (
    PassiveCompositionIngressStatus,
    normalize_required_text,
)


@dataclass(frozen=True)
class ObservedShadowEmission:
    ingress_sequence: int
    observed_at: float
    emission: object


class LatestPreservingShadowIngress:
    """Small queue whose pressure can never reach a source control loop."""

    def __init__(self, source_kind: str, *, capacity: int) -> None:
        self.source_kind = normalize_required_text(
            source_kind,
            name="source_kind",
        )
        size = int(capacity)
        if size <= 0:
            raise ValueError("Shadow ingress capacity must be positive.")
        self._queue: queue.Queue[ObservedShadowEmission] = queue.Queue(
            maxsize=size
        )
        self._lock = threading.RLock()
        self._offered = 0
        self._accepted = 0
        self._evicted = 0
        self._rejected = 0
        self._drained = 0
        self._ignored_disabled = 0

    def ignore_while_disabled(self) -> None:
        with self._lock:
            self._offered += 1
            self._ignored_disabled += 1

    def offer(self, item: ObservedShadowEmission) -> bool:
        with self._lock:
            self._offered += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._queue.task_done()
                with self._lock:
                    self._evicted += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self._rejected += 1
                return False
        with self._lock:
            self._accepted += 1
        return True

    def drain_nowait(self) -> tuple[ObservedShadowEmission, ...]:
        drained: list[ObservedShadowEmission] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            drained.append(item)
            self._queue.task_done()
        with self._lock:
            self._drained += len(drained)
        return tuple(drained)

    def status(self) -> PassiveCompositionIngressStatus:
        with self._lock:
            return PassiveCompositionIngressStatus(
                source_kind=self.source_kind,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                offered_count=self._offered,
                accepted_count=self._accepted,
                evicted_count=self._evicted,
                rejected_count=self._rejected,
                drained_count=self._drained,
                ignored_while_disabled_count=self._ignored_disabled,
            )


__all__ = ["LatestPreservingShadowIngress", "ObservedShadowEmission"]
