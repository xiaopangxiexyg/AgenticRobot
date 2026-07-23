"""Background lifecycle for the scoped M4B spool sink."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .sink import ControlTraceSinkFlushReport, ControlTraceSpoolSink


@dataclass(frozen=True)
class ControlTraceSpoolWorkerSnapshot:
    started: bool
    stopped: bool
    thread_alive: bool
    flush_count: int
    last_report: ControlTraceSinkFlushReport | None
    fatal_error_type: str | None


class ControlTraceSpoolWorker:
    """Poll-based storage worker; it never owns runtime or episode lifecycle."""

    def __init__(
        self,
        sink: ControlTraceSpoolSink,
        *,
        poll_interval_sec: float = 0.01,
        thread_name: str = "control-trace-spool",
    ) -> None:
        interval = float(poll_interval_sec)
        if interval <= 0 or interval > 1.0:
            raise ValueError("poll_interval_sec must be in (0, 1].")
        name = str(thread_name).strip()
        if not name:
            raise ValueError("thread_name must be non-empty.")
        self.sink = sink
        self.poll_interval_sec = interval
        self.thread_name = name
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._flush_count = 0
        self._last_report: ControlTraceSinkFlushReport | None = None
        self._fatal_error_type: str | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("ControlTrace spool worker was already started.")
            thread = threading.Thread(
                target=self._run,
                name=self.thread_name,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def notify(self) -> None:
        """Wake the worker; safe for a frontend after successful admission."""

        self._wake.set()

    def flush_once(self) -> ControlTraceSinkFlushReport:
        # Episode seal may request a bounded synchronous tail flush while the
        # background poller is active.  The sink/WAL has one logical writer;
        # serialize both callers so a sequence cannot be appended/acked twice.
        with self._flush_lock:
            report = self.sink.flush_pending()
        with self._state_lock:
            self._flush_count += 1
            self._last_report = report
        return report

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.wait(self.poll_interval_sec)
                self._wake.clear()
                report = self.flush_once()
                if report.failure_code is not None:
                    return
            self.flush_once()
        except Exception as exc:  # evidence is exposed; frontend decides policy
            with self._state_lock:
                self._fatal_error_type = type(exc).__name__

    def stop(self, *, timeout_sec: float = 5.0) -> None:
        with self._state_lock:
            thread = self._thread
        if thread is None:
            return
        self._stop.set()
        self._wake.set()
        thread.join(timeout=float(timeout_sec))
        if thread.is_alive():
            raise TimeoutError("ControlTrace spool worker did not stop before deadline.")

    def snapshot(self) -> ControlTraceSpoolWorkerSnapshot:
        with self._state_lock:
            thread = self._thread
            return ControlTraceSpoolWorkerSnapshot(
                started=thread is not None,
                stopped=self._stop.is_set() and (thread is None or not thread.is_alive()),
                thread_alive=False if thread is None else thread.is_alive(),
                flush_count=self._flush_count,
                last_report=self._last_report,
                fatal_error_type=self._fatal_error_type,
            )

    def __enter__(self) -> "ControlTraceSpoolWorker":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()


__all__ = ["ControlTraceSpoolWorker", "ControlTraceSpoolWorkerSnapshot"]
