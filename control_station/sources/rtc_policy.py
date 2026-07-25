"""Passive Control Station consumer for formal policy emissions.

The deploy frontend remains the sole owner of inference, timed-plan merge,
due-pop, emission commit, legacy execution, and producer lifecycle.  This
adapter observes only committed ``PolicyActionEmission`` values.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from robot_motion.deploy import (
    PolicyActionEmission,
    RtcPolicyEmissionSourceBinding,
)


RTC_POLICY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2d_rtc_policy_binding.v1"
)

# Existing imports keep the same object identity while the duplicate Station
# implementation is removed.
RtcPolicyEmissionBinding = RtcPolicyEmissionSourceBinding


@dataclass(frozen=True)
class RtcPolicyStationConsumerStatus:
    started: bool
    closed: bool
    shadow_enabled: bool
    queue_size: int
    queue_capacity: int
    received_emission_count: int
    observed_emission_count: int
    inactive_emission_count: int
    rejected_emission_count: int
    error_count: int
    fence_sequence: int
    last_error: str
    last_emission_id: str | None


@dataclass
class _RtcShadowFence:
    sequence: int
    enabled: bool
    applied: threading.Event = field(default_factory=threading.Event)


class RtcPolicyStationConsumer:
    """Consume formal emissions behind an ordered passive shadow fence.

    The start/stop fence travels through the same queue as emissions.  An
    emission offered before ``start_shadow`` therefore cannot be observed after
    activation merely because it was waiting in the subscriber queue.
    """

    authoritative = False
    producer_lifecycle_owned = False
    motion_capable = False

    def __init__(
        self,
        binding: RtcPolicyEmissionSourceBinding,
        observe_emission: Callable[[PolicyActionEmission], object],
        *,
        emission_queue: queue.Queue[object] | None = None,
        queue_capacity: int = 256,
        fence_timeout_sec: float = 2.0,
    ) -> None:
        if not isinstance(binding, RtcPolicyEmissionSourceBinding):
            raise TypeError("binding must be an RtcPolicyEmissionSourceBinding.")
        if not callable(observe_emission):
            raise TypeError("observe_emission must be callable.")
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("RTC Station queue_capacity must be positive.")
        timeout = float(fence_timeout_sec)
        if timeout <= 0.0:
            raise ValueError("fence_timeout_sec must be positive.")
        self.binding = binding
        if emission_queue is not None and not isinstance(emission_queue, queue.Queue):
            raise TypeError("emission_queue must be a queue.Queue or None.")
        if emission_queue is not None and int(emission_queue.maxsize) <= 0:
            raise ValueError("An injected Station emission_queue must be bounded.")
        self.emission_queue: queue.Queue[object] = (
            queue.Queue(maxsize=capacity)
            if emission_queue is None
            else emission_queue
        )
        self._observe_emission = observe_emission
        self._fence_timeout_sec = timeout
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._shadow_enabled = False
        self._fence_sequence = 0
        self._received_emission_count = 0
        self._observed_emission_count = 0
        self._inactive_emission_count = 0
        self._rejected_emission_count = 0
        self._error_count = 0
        self._last_error = ""
        self._last_emission_id: str | None = None

    def start(self) -> None:
        """Start only the passive observer worker; never the RTC producer."""

        with self._lock:
            if self._closed:
                raise RuntimeError("RTC Station consumer is closed.")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="control-station-rtc-policy-consumer",
                daemon=True,
            )
            self._thread.start()

    def set_shadow_enabled(self, enabled: bool) -> int:
        """Apply an ordered local observation fence without touching RTC."""

        self.start()
        with self._lock:
            if self._closed:
                raise RuntimeError("RTC Station consumer is closed.")
            sequence = self._fence_sequence + 1
            fence = _RtcShadowFence(sequence=sequence, enabled=bool(enabled))
            try:
                self.emission_queue.put_nowait(fence)
            except queue.Full as exc:
                raise RuntimeError("RTC Station shadow fence queue is full.") from exc
        if not fence.applied.wait(timeout=self._fence_timeout_sec):
            raise TimeoutError("RTC Station shadow fence was not applied in time.")
        return sequence

    def _run(self) -> None:
        while not self._stop.is_set() or not self.emission_queue.empty():
            try:
                item = self.emission_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if isinstance(item, _RtcShadowFence):
                    with self._lock:
                        self._shadow_enabled = item.enabled
                        self._fence_sequence = item.sequence
                    item.applied.set()
                    continue
                if not isinstance(item, PolicyActionEmission):
                    raise TypeError(
                        "RTC Station queue accepts PolicyActionEmission only."
                    )
                with self._lock:
                    self._received_emission_count += 1
                    enabled = self._shadow_enabled and not self._closed
                if not enabled:
                    with self._lock:
                        self._inactive_emission_count += 1
                    continue
                emission = self.binding.validate_emission(item)
                self._observe_emission(emission)
                with self._lock:
                    self._observed_emission_count += 1
                    self._last_emission_id = emission.emission_id
            except (TypeError, ValueError, RuntimeError) as exc:
                with self._lock:
                    self._rejected_emission_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # passive observation cannot fault deploy
                with self._lock:
                    self._error_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.emission_queue.task_done()

    def status(self) -> RtcPolicyStationConsumerStatus:
        with self._lock:
            return RtcPolicyStationConsumerStatus(
                started=self._thread is not None,
                closed=self._closed,
                shadow_enabled=self._shadow_enabled,
                queue_size=self.emission_queue.qsize(),
                queue_capacity=self.emission_queue.maxsize,
                received_emission_count=self._received_emission_count,
                observed_emission_count=self._observed_emission_count,
                inactive_emission_count=self._inactive_emission_count,
                rejected_emission_count=self._rejected_emission_count,
                error_count=self._error_count,
                fence_sequence=self._fence_sequence,
                last_error=self._last_error,
                last_emission_id=self._last_emission_id,
            )

    def close(self, *, timeout_sec: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            thread = self._thread
        if thread is not None:
            self.set_shadow_enabled(False)
        with self._lock:
            self._closed = True
            self._shadow_enabled = False
        self._stop.set()
        if thread is not None:
            thread.join(timeout=float(timeout_sec))
            if thread.is_alive():
                raise TimeoutError("RTC Station consumer did not stop before deadline.")


__all__ = [
    "RTC_POLICY_BINDING_HASH_DOMAIN_V1",
    "RtcPolicyEmissionBinding",
    "RtcPolicyStationConsumer",
    "RtcPolicyStationConsumerStatus",
]
