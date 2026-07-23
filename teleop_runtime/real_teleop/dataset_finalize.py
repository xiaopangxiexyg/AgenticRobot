"""Thread-safe state for asynchronous dataset finalization.

The task object records lifecycle metadata only; the recorder still owns the
actual writer and filesystem work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import threading
import time
from typing import Any

@dataclass
class RealDatasetFinalizeTask:
    episode_index: int
    frame_count: int
    reason: str
    queued_at: float
    summary: dict[str, Any] = field(default_factory=dict, repr=False)
    before_finalize: Callable[[], Mapping[str, Any] | None] | None = field(
        default=None,
        repr=False,
    )
    thread: threading.Thread | None = None
    started_at: float | None = None
    done: bool = False
    success: bool = False
    error: str = ""
    completed_at: float | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    control_trace_release: Any | None = field(default=None, repr=False)
    control_trace_finalize_report: Any | None = field(default=None, repr=False)
    control_trace_error: str = ""
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def pending(self) -> bool:
        return bool(not self.done and self.thread is None)

    @property
    def running(self) -> bool:
        return bool(not self.done and self.thread is not None)

    def mark_started(self) -> None:
        with self._state_lock:
            if self.started_at is None:
                self.started_at = time.time()

    def record_stage_timing(self, name: str, elapsed_sec: float) -> None:
        with self._state_lock:
            self.stage_timings_ms[str(name)] = max(0.0, float(elapsed_sec) * 1000.0)

    def mark_done(self, *, success: bool, error: str = "") -> None:
        with self._state_lock:
            self.done = True
            self.success = bool(success)
            self.error = str(error or "")
            self.completed_at = time.time()

__all__ = ["RealDatasetFinalizeTask"]
