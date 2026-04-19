from __future__ import annotations

import os
import sys
import time
from typing import Any


_MISSING = object()
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_TAG_COLORS = {
    "STAGE": "\033[95m",
    "LINKCHECK": "\033[96m",
    "VIRTUALDESK": "\033[94m",
    "WRISTCOARSE": "\033[94m",
    "HOMECAPTURE": "\033[93m",
    "TRACKERFRAMEVIZ": "\033[92m",
    "VALIDATE": "\033[92m",
    "SAVE": "\033[92m",
    "CALIBRATION": "\033[96m",
}
DEFAULT_LOG_HEARTBEAT_SEC = 2


def _style(text: str, *codes: str) -> str:
    if not _USE_COLOR or not codes:
        return text
    return "".join(codes) + text + _RESET


def _message_color(message: str) -> str | None:
    upper = message.upper()
    if any(word in upper for word in ("REJECTED", "FAILED", "ERROR", "CANNOT", "MISSING", "TIMEOUT")):
        return "\033[91m"
    if any(word in upper for word in ("ACCEPTED", "READY", "UPDATED", "WRITTEN", "CONTINUING", "LOADED")):
        return "\033[92m"
    if any(word in upper for word in ("PRESS", "CAPTURE STARTED", "PROGRESS", "WAIT", "RESET", "STEP", "LIVE")):
        return "\033[93m"
    return None


def log_event(tag: str, message: str, *, now: float | None = None) -> None:
    current_time = time.time() if now is None else float(now)
    timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
    tag_upper = str(tag).upper()
    tag_color = _TAG_COLORS.get(tag_upper, "\033[96m")
    timestamp_text = _style(f"[{timestamp}]", _DIM)
    tag_text = _style(f"[{tag}]", _BOLD, tag_color)
    message_color = _message_color(message)
    message_text = _style(message, message_color) if message_color else message
    print(f"{timestamp_text} {tag_text} {message_text}", flush=True)


def log_stage(tag: str, title: str, *, now: float | None = None) -> None:
    log_event(tag, f"=== STEP: {title.upper()} ===", now=now)


class ChangeGate:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._last_emit_times: dict[str, float] = {}

    def should_emit(
        self,
        key: str,
        value: Any,
        *,
        now: float | None = None,
        heartbeat_sec: float | None = None,
    ) -> bool:
        current_time = time.time() if now is None else float(now)
        previous = self._values.get(key, _MISSING)
        if previous is _MISSING or previous != value:
            self._values[key] = value
            self._last_emit_times[key] = current_time
            return True
        if heartbeat_sec is not None:
            last_emit = self._last_emit_times.get(key, current_time)
            if current_time - last_emit >= heartbeat_sec:
                self._last_emit_times[key] = current_time
                return True
        return False
