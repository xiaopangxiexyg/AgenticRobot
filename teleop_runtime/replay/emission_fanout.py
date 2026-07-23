"""Best-effort fanout for the canonical replay source-frame tap.

The replay executor still builds and offers exactly one immutable
``ReplaySourceFrameEmission`` per legacy source frame.  This adapter only
duplicates the object reference into explicitly bounded passive subscriber
queues.  It owns no cursor, schedule, resampler, consumer thread, or motion
authority.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Mapping

from robot_motion.replay.proposal_source import ReplaySourceFrameEmission


@dataclass(frozen=True)
class ReplayEmissionSubscriberStatus:
    subscriber_id: str
    queue_size: int
    queue_capacity: int
    offer_count: int
    drop_count: int
    error_count: int
    last_error: str


@dataclass
class _SubscriberCounters:
    offer_count: int = 0
    drop_count: int = 0
    error_count: int = 0
    last_error: str = ""


class ReplaySourceFrameEmissionFanout:
    """One nonblocking replay observer sink with bounded passive branches."""

    # ``ReplayExecutionController`` uses this explicit capability marker rather
    # than mistaking a fanout for an unbounded queue.  Every branch is checked
    # separately at construction time.
    nonblocking_bounded = True

    def __init__(
        self,
        subscribers: Mapping[
            str,
            queue.Queue[ReplaySourceFrameEmission],
        ],
    ) -> None:
        normalized: dict[str, queue.Queue[ReplaySourceFrameEmission]] = {}
        for subscriber_id, subscriber_queue in subscribers.items():
            name = str(subscriber_id).strip()
            if not name:
                raise ValueError("Replay emission subscriber_id must be non-empty.")
            if name in normalized:
                raise ValueError(f"Duplicate replay emission subscriber_id: {name!r}.")
            if int(getattr(subscriber_queue, "maxsize", 0)) <= 0:
                raise ValueError(
                    f"Replay emission subscriber {name!r} must use a bounded queue."
                )
            normalized[name] = subscriber_queue
        if not normalized:
            raise ValueError("Replay emission fanout requires at least one subscriber.")
        self._subscribers = normalized
        self._counters = {
            subscriber_id: _SubscriberCounters()
            for subscriber_id in normalized
        }
        self._lock = threading.RLock()
        self._input_offer_count = 0

    def put_nowait(self, emission: ReplaySourceFrameEmission) -> None:
        """Offer the exact same object to every branch without propagating errors."""

        if not isinstance(emission, ReplaySourceFrameEmission):
            raise TypeError("Replay emission fanout accepts ReplaySourceFrameEmission only.")
        with self._lock:
            self._input_offer_count += 1
            subscribers = tuple(self._subscribers.items())
        for subscriber_id, subscriber_queue in subscribers:
            try:
                subscriber_queue.put_nowait(emission)
                with self._lock:
                    self._counters[subscriber_id].offer_count += 1
            except queue.Full:
                with self._lock:
                    self._counters[subscriber_id].drop_count += 1
            except Exception as exc:  # passive branch failure cannot fault replay
                with self._lock:
                    counters = self._counters[subscriber_id]
                    counters.error_count += 1
                    counters.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, object]:
        with self._lock:
            subscribers = tuple(
                ReplayEmissionSubscriberStatus(
                    subscriber_id=subscriber_id,
                    queue_size=subscriber_queue.qsize(),
                    queue_capacity=int(subscriber_queue.maxsize),
                    offer_count=counters.offer_count,
                    drop_count=counters.drop_count,
                    error_count=counters.error_count,
                    last_error=counters.last_error,
                )
                for subscriber_id, subscriber_queue in self._subscribers.items()
                for counters in (self._counters[subscriber_id],)
            )
            return {
                "enabled": True,
                "authoritative": False,
                "input_offer_count": self._input_offer_count,
                "subscriber_count": len(subscribers),
                "subscribers": {
                    item.subscriber_id: {
                        "queue_size": item.queue_size,
                        "queue_capacity": item.queue_capacity,
                        "offer_count": item.offer_count,
                        "drop_count": item.drop_count,
                        "error_count": item.error_count,
                        "last_error": item.last_error,
                    }
                    for item in subscribers
                },
            }


__all__ = [
    "ReplayEmissionSubscriberStatus",
    "ReplaySourceFrameEmissionFanout",
]
