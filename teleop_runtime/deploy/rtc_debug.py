"""Non-authoritative, bounded RTC debug event observation.

This module is an application diagnostic helper.  It owns no RTC producer
state, source binding activation, trace/WAL lifecycle, target authority,
safety, reference, backend, teleoperation, replay, or HITL contract.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Mapping


RTC_DEBUG_LEVEL_OFF = "off"
RTC_DEBUG_LEVEL_SUMMARY = "summary"
RTC_DEBUG_LEVEL_TRACE = "trace"
RTC_DEBUG_LEVELS = (
    RTC_DEBUG_LEVEL_OFF,
    RTC_DEBUG_LEVEL_SUMMARY,
    RTC_DEBUG_LEVEL_TRACE,
)
_RTC_DEBUG_LEVEL_RANK = {
    RTC_DEBUG_LEVEL_OFF: 0,
    RTC_DEBUG_LEVEL_SUMMARY: 1,
    RTC_DEBUG_LEVEL_TRACE: 2,
}


def normalize_rtc_debug_level(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in RTC_DEBUG_LEVELS:
        raise ValueError(f"Unsupported RTC debug level: {normalized!r}.")
    return normalized


@dataclass(frozen=True)
class RtcDebugEventObserverStatus:
    level: str
    queue_size: int
    queue_capacity: int
    offered_event_count: int
    written_event_count: int
    dropped_event_count: int
    sink_error_count: int
    last_sink_error: str
    closed: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "level": self.level,
            "queue_size": self.queue_size,
            "queue_capacity": self.queue_capacity,
            "offered_event_count": self.offered_event_count,
            "written_event_count": self.written_event_count,
            "dropped_event_count": self.dropped_event_count,
            "sink_error_count": self.sink_error_count,
            "last_sink_error": self.last_sink_error,
            "closed": self.closed,
            "non_authoritative": True,
        }


class RtcDebugEventObserver:
    """Offer RTC debug events without blocking the control or worker thread."""

    nonblocking_bounded = True
    authoritative = False

    def __init__(
        self,
        *,
        level: str,
        sink: Callable[[str, Mapping[str, object]], None],
        queue_capacity: int = 2048,
    ) -> None:
        normalized_level = normalize_rtc_debug_level(level)
        if not callable(sink):
            raise TypeError("RTC debug sink must be callable.")
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("RTC debug queue_capacity must be positive.")
        self.level = normalized_level
        self._sink = sink
        self._queue: queue.Queue[tuple[str, dict[str, object]]] = queue.Queue(
            maxsize=capacity
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._offered_event_count = 0
        self._written_event_count = 0
        self._dropped_event_count = 0
        self._sink_error_count = 0
        self._last_sink_error = ""
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="rtc-debug-event-writer",
                daemon=True,
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.level != RTC_DEBUG_LEVEL_OFF

    def accepts(self, minimum_level: str = RTC_DEBUG_LEVEL_SUMMARY) -> bool:
        required = normalize_rtc_debug_level(minimum_level)
        return (
            not self._closed
            and _RTC_DEBUG_LEVEL_RANK[self.level]
            >= _RTC_DEBUG_LEVEL_RANK[required]
        )

    def offer(
        self,
        event: str,
        payload: Mapping[str, object],
        *,
        minimum_level: str = RTC_DEBUG_LEVEL_SUMMARY,
    ) -> bool:
        """Offer one detached event; full queues drop diagnostics only."""

        if not self.accepts(minimum_level):
            return False
        event_name = str(event).strip()
        if not event_name:
            raise ValueError("RTC debug event name must be non-empty.")
        detached = dict(payload)
        detached.setdefault("non_authoritative", True)
        with self._lock:
            if self._closed:
                return False
            self._offered_event_count += 1
        try:
            self._queue.put_nowait((event_name, detached))
            return True
        except queue.Full:
            with self._lock:
                self._dropped_event_count += 1
            return False

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event, payload = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._sink(event, payload)
                with self._lock:
                    self._written_event_count += 1
            except Exception as exc:  # debug observation cannot fault deploy
                with self._lock:
                    self._sink_error_count += 1
                    self._last_sink_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def status(self) -> RtcDebugEventObserverStatus:
        with self._lock:
            return RtcDebugEventObserverStatus(
                level=self.level,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                offered_event_count=self._offered_event_count,
                written_event_count=self._written_event_count,
                dropped_event_count=self._dropped_event_count,
                sink_error_count=self._sink_error_count,
                last_sink_error=self._last_sink_error,
                closed=self._closed,
            )

    def close(self, *, timeout_sec: float = 5.0) -> RtcDebugEventObserverStatus:
        with self._lock:
            if self._closed:
                return self.status()
            self._closed = True
            thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join(timeout=float(timeout_sec))
            if thread.is_alive():
                raise TimeoutError("RTC debug writer did not drain before deadline.")
        return self.status()


__all__ = [
    "RTC_DEBUG_LEVEL_OFF",
    "RTC_DEBUG_LEVEL_SUMMARY",
    "RTC_DEBUG_LEVEL_TRACE",
    "RTC_DEBUG_LEVELS",
    "RtcDebugEventObserver",
    "RtcDebugEventObserverStatus",
    "normalize_rtc_debug_level",
]
