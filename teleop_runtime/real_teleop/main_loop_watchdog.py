"""Observational main-loop stall watchdog for real teleoperation."""

from __future__ import annotations

import faulthandler
import sys
import threading
import time


class MainLoopStallWatchdog:
    def __init__(
        self,
        *,
        enabled: bool,
        timeout_sec: float = 5.0,
        report_interval_sec: float = 5.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.report_interval_sec = max(1.0, float(report_interval_sec))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat = time.monotonic()
        self._last_report = 0.0
        self._phase = "init"
        self._step_index = -1

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop_event.clear()
        self._last_beat = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="real_teleop_main_loop_watchdog",
        )
        self._thread.start()

    def beat(self, phase: str, step_index: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._last_beat = time.monotonic()
            self._phase = str(phase)
            self._step_index = int(step_index)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        worker = self._thread
        worker.join(timeout=0.5)
        if not worker.is_alive():
            self._thread = None
            return
        raise RuntimeError("real teleop main-loop watchdog did not stop within 0.500s")

    def _run(self) -> None:
        while not self._stop_event.wait(0.5):
            now = time.monotonic()
            with self._lock:
                elapsed = now - self._last_beat
                phase = self._phase
                step_index = self._step_index
                last_report = self._last_report
                should_report = (
                    elapsed >= self.timeout_sec
                    and (last_report <= 0.0 or now - last_report >= self.report_interval_sec)
                )
                if should_report:
                    self._last_report = now
            if not should_report:
                continue
            print(
                "WARNING: real teleop main loop appears stalled "
                f"for {elapsed:.1f}s at step={step_index} phase={phase}; "
                "dumping Python thread stacks.",
                file=sys.stderr,
                flush=True,
            )
            try:
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except Exception as exc:  # noqa: BLE001 - diagnostics must not break teleop.
                print(
                    f"WARNING: failed to dump thread stacks: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )


__all__ = ["MainLoopStallWatchdog"]
