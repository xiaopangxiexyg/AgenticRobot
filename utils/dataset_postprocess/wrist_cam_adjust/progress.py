from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


@dataclass
class ProgressPrinter:
    label: str
    total_items: int = 1
    total_units: int = 0
    enabled: bool = True
    min_interval_sec: float = 0.25
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _started_at: float = field(default_factory=time.perf_counter, init=False)
    _last_render_at: float = field(default=0.0, init=False)
    _unit_by_item: dict[str, int] = field(default_factory=dict, init=False)
    _completed_items: int = field(default=0, init=False)
    _active_item: str = field(default="", init=False)
    _finished: bool = field(default=False, init=False)
    _dirty: bool = field(default=False, init=False)

    def start_item(self, item: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active_item = str(item)
            self._unit_by_item.setdefault(str(item), 0)
            self._dirty = True
            self._render(force=True)

    def callback(self, item: str) -> Callable[[int], None] | None:
        if not self.enabled:
            return None

        def update(current_units: int) -> None:
            self.update_item(item, current_units)

        return update

    def update_item(self, item: str, current_units: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            key = str(item)
            self._active_item = key
            self._unit_by_item[key] = max(int(current_units), self._unit_by_item.get(key, 0))
            self._dirty = True
            self._render(force=False)

    def complete_item(self, item: str, final_units: int | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            key = str(item)
            self._active_item = key
            if final_units is not None:
                self._unit_by_item[key] = max(int(final_units), self._unit_by_item.get(key, 0))
            self._completed_items += 1
            self._dirty = True
            self._render(force=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._finished:
                return
            self._finished = True
            if self._dirty:
                self._render(force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self._last_render_at < float(self.min_interval_sec):
            return
        self._last_render_at = now
        done_units = sum(self._unit_by_item.values())
        elapsed = now - self._started_at
        fps = 0.0 if elapsed <= 0.0 else float(done_units) / elapsed
        if self.total_units > 0:
            percent = min(100.0, 100.0 * float(done_units) / float(self.total_units))
            unit_text = f"{done_units}/{self.total_units} frames {percent:5.1f}%"
        else:
            unit_text = f"{done_units} frames"
        item_text = f"jobs {self._completed_items}/{max(1, int(self.total_items))}"
        active = f" active={self._active_item}" if self._active_item else ""
        message = (
            f"\r{self.label}: {unit_text} | {item_text} | "
            f"{fps:5.1f} fps | elapsed {_format_elapsed(elapsed)}{active}\x1b[K"
        )
        sys.stderr.write(message)
        sys.stderr.flush()
        self._dirty = False
