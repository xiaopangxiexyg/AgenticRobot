"""Passive observation utilities for already-produced Human emissions.

These utilities are backend and GUI neutral.  They own bounded passive
observation only: no device, retargeter, worker, readiness, authority, safety,
or dispatch lifecycle.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

from .human_proposal import HumanCommandEmission


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


@runtime_checkable
class HumanPoseSourceContract(Protocol):
    """Structural source contract shared by synthetic and live pose inputs."""

    source_id: str
    reference_frame_by_effector: tuple[tuple[str, str], ...]
    pose_frame_contract_hash: str
    action_spec_hash_by_effector: tuple[tuple[str, str], ...]
    units_convention_id: str

    @property
    def semantic_content_hash(self) -> str: ...


@dataclass(frozen=True)
class PassiveHumanCommandEmissionIngressStatus:
    observer_binding_id: str
    queue_size: int
    queue_capacity: int
    observed_count: int
    accepted_count: int
    evicted_count: int
    stale_count: int
    acquired_count: int


class PassiveHumanCommandEmissionIngress:
    """Bounded latest-preserving input for a lower-owned observer callback.

    ``observe_emission`` is nonblocking and preserves the exact immutable
    object. When full, the oldest passive observation is evicted so a future
    readiness wait can still see the newest command. Waiting is intended for
    an application side-effect worker, never the teleoperation control loop.

    Emission freshness alone does not prove deadman re-arm or retargeter
    rebase, so this class is deliberately not a takeover-readiness provider.
    """

    nonblocking_bounded = True

    def __init__(
        self,
        observer_binding_id: str,
        *,
        queue_capacity: int = 8,
    ) -> None:
        self.observer_binding_id = _required(
            observer_binding_id,
            field_name="observer_binding_id",
        )
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("Human emission ingress queue_capacity must be positive.")
        self._queue: queue.Queue[HumanCommandEmission] = queue.Queue(
            maxsize=capacity
        )
        self._lock = threading.RLock()
        self._observed_count = 0
        self._accepted_count = 0
        self._evicted_count = 0
        self._stale_count = 0
        self._acquired_count = 0

    def observe_emission(self, emission: HumanCommandEmission) -> bool:
        """Offer one exact emission without propagating queue pressure."""

        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Human emission ingress accepts HumanCommandEmission only.")
        with self._lock:
            self._observed_count += 1
        try:
            self._queue.put_nowait(emission)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                # A concurrent waiter freed the slot between Full and get.
                pass
            else:
                self._queue.task_done()
                with self._lock:
                    self._evicted_count += 1
            try:
                self._queue.put_nowait(emission)
            except queue.Full:
                # Another producer won the bounded slot. This observation is
                # best-effort and must never delay the lower control loop.
                return False
        with self._lock:
            self._accepted_count += 1
        return True

    def wait_for_post_boundary_emission(
        self,
        *,
        boundary_at: float,
        timeout_sec: float,
    ) -> HumanCommandEmission:
        """Return the first queued emission at/after an explicit time boundary."""

        boundary = _finite(boundary_at, field_name="boundary_at")
        timeout = _finite(timeout_sec, field_name="timeout_sec")
        if timeout <= 0.0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("Timed out waiting for a post-boundary human emission.")
            try:
                emission = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for a post-boundary human emission."
                ) from exc
            try:
                if emission.emitted_at < boundary:
                    with self._lock:
                        self._stale_count += 1
                    continue
                with self._lock:
                    self._acquired_count += 1
                return emission
            finally:
                self._queue.task_done()

    def status(self) -> PassiveHumanCommandEmissionIngressStatus:
        with self._lock:
            return PassiveHumanCommandEmissionIngressStatus(
                observer_binding_id=self.observer_binding_id,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                observed_count=self._observed_count,
                accepted_count=self._accepted_count,
                evicted_count=self._evicted_count,
                stale_count=self._stale_count,
                acquired_count=self._acquired_count,
            )


@dataclass
class _HumanObserverCounters:
    offer_count: int = 0
    rejection_count: int = 0
    error_count: int = 0
    last_error: str = ""


class PassiveHumanCommandObserverFanout:
    """Exception-isolated callback fanout over one canonical Human source.

    Subscribers must themselves be passive and nonblocking. The fanout does
    not create a second source or alter the canonical callback; an application
    root may use it to combine existing observer sinks.
    """

    def __init__(
        self,
        subscribers: Mapping[
            str,
            Callable[[HumanCommandEmission], object],
        ],
    ) -> None:
        normalized: dict[str, Callable[[HumanCommandEmission], object]] = {}
        for subscriber_id, callback in subscribers.items():
            name = _required(subscriber_id, field_name="subscriber_id")
            if name in normalized:
                raise ValueError(f"Duplicate human observer subscriber_id: {name!r}.")
            if not callable(callback):
                raise TypeError(f"Human observer subscriber {name!r} is not callable.")
            normalized[name] = callback
        if not normalized:
            raise ValueError("Human observer fanout requires at least one subscriber.")
        self._subscribers = normalized
        self._counters = {
            subscriber_id: _HumanObserverCounters()
            for subscriber_id in normalized
        }
        self._lock = threading.RLock()
        self._input_offer_count = 0

    def observe_emission(self, emission: HumanCommandEmission) -> bool:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Human observer fanout accepts HumanCommandEmission only.")
        with self._lock:
            self._input_offer_count += 1
            subscribers = tuple(self._subscribers.items())
        all_accepted = True
        for subscriber_id, callback in subscribers:
            try:
                accepted = callback(emission)
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.offer_count += 1
                    if accepted is False:
                        counters.rejection_count += 1
                        all_accepted = False
            except Exception as exc:  # passive subscriber failure is isolated
                all_accepted = False
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.error_count += 1
                    counters.last_error = f"{type(exc).__name__}: {exc}"
        return all_accepted

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "authoritative": False,
                "input_offer_count": self._input_offer_count,
                "subscriber_count": len(self._subscribers),
                "subscribers": {
                    subscriber_id: {
                        "offer_count": counters.offer_count,
                        "rejection_count": counters.rejection_count,
                        "error_count": counters.error_count,
                        "last_error": counters.last_error,
                    }
                    for subscriber_id, counters in self._counters.items()
                },
            }


__all__ = [
    "HumanPoseSourceContract",
    "PassiveHumanCommandEmissionIngress",
    "PassiveHumanCommandEmissionIngressStatus",
    "PassiveHumanCommandObserverFanout",
]
