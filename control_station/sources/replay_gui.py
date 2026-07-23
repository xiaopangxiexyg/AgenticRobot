"""Control Station consumer for the canonical GUI replay emission seam.

This application-leaf adapter receives the exact immutable emission produced
by the existing replay executor.  It never reads a dataset, advances a cursor,
resamples a schedule, or owns a dispatch token.  Publication remains guarded
by the injected Station composition callback.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

from robot_motion.replay.proposal_source import ReplaySourceFrameEmission


@dataclass(frozen=True)
class ReplayGuiStationConsumerStatus:
    started: bool
    closed: bool
    queue_size: int
    queue_capacity: int
    received_count: int
    submitted_count: int
    rejected_count: int
    error_count: int
    last_error: str
    last_emission_id: str | None


class ReplayGuiStationConsumer:
    """Bounded default-off bridge from GUI replay to a Station source slot."""

    def __init__(
        self,
        submit_source_frame_emission: Callable[[ReplaySourceFrameEmission], object],
        *,
        queue_capacity: int = 4096,
    ) -> None:
        if not callable(submit_source_frame_emission):
            raise TypeError("submit_source_frame_emission must be callable.")
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("Replay GUI Station queue_capacity must be positive.")
        self.emission_queue: queue.Queue[ReplaySourceFrameEmission] = queue.Queue(
            maxsize=capacity
        )
        self._submit = submit_source_frame_emission
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._received_count = 0
        self._submitted_count = 0
        self._rejected_count = 0
        self._error_count = 0
        self._last_error = ""
        self._last_emission_id: str | None = None

    def start(self) -> None:
        """Start consumption explicitly; constructing the bridge has no effect."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Replay GUI Station consumer is closed.")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="control-station-replay-gui-consumer",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() or not self.emission_queue.empty():
            try:
                emission = self.emission_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    self._received_count += 1
                    self._last_emission_id = emission.emission_id
                result = self._submit(emission)
                returned_emission = getattr(result, "source_emission", emission)
                if returned_emission is not emission:
                    raise RuntimeError(
                        "Station replay submission rebuilt the source emission."
                    )
                with self._lock:
                    self._submitted_count += 1
            except (RuntimeError, ValueError) as exc:
                # Publication barrier closure, inactive authority, or binding
                # rejection affects this Station branch only.
                with self._lock:
                    self._rejected_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # application observer must not fault replay
                with self._lock:
                    self._error_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.emission_queue.task_done()

    def status(self) -> ReplayGuiStationConsumerStatus:
        with self._lock:
            return ReplayGuiStationConsumerStatus(
                started=self._thread is not None,
                closed=self._closed,
                queue_size=self.emission_queue.qsize(),
                queue_capacity=self.emission_queue.maxsize,
                received_count=self._received_count,
                submitted_count=self._submitted_count,
                rejected_count=self._rejected_count,
                error_count=self._error_count,
                last_error=self._last_error,
                last_emission_id=self._last_emission_id,
            )

    def close(self, *, timeout_sec: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join(timeout=float(timeout_sec))
            if thread.is_alive():
                raise TimeoutError(
                    "Replay GUI Station consumer did not stop before deadline."
                )


__all__ = [
    "ReplayGuiStationConsumer",
    "ReplayGuiStationConsumerStatus",
]
