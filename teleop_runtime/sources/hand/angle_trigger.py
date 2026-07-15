from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel


class AngleTriggerSource(ObservationSource):
    """Expose the calibrated left/right angle triggers as scalar observations."""

    def __init__(self, reader: Any, *, rate_hz: float = 30.0, config_root: Path | None = None) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive.")
        self._reader = reader
        self._rate_hz = float(rate_hz)
        self._config_root = None if config_root is None else str(config_root)
        self._lock = threading.Lock()
        self._latest: dict[str, ObservationChannel] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._errors = 0

    @property
    def name(self) -> str:
        return "angle_trigger"

    def start(self) -> None:
        if self._running:
            return
        self._reader.open()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="angle_trigger", daemon=True)
        self._thread.start()

    def _publish_invalid(self, error: str) -> None:
        now = time.time()
        with self._lock:
            for side in ("left", "right"):
                previous = self._latest.get(f"{side}_angle_trigger")
                value = 0.0 if previous is None else float(previous.value)
                self._latest[f"{side}_angle_trigger"] = ObservationChannel(
                    channel_name=f"{side}_angle_trigger", capability="scalar", side=side,
                    value=value, frame_name=None, timestamp=now, source_name=self.name,
                    valid=False, critical=False,
                    metadata={"read_error": error, "reader_error_count": self._errors,
                              "config_root": self._config_root},
                )

    def _read_loop(self) -> None:
        period = 1.0 / self._rate_hz
        while self._running:
            started = time.monotonic()
            try:
                values = self._reader.read()
                now = time.time()
                with self._lock:
                    for side in ("left", "right"):
                        self._latest[f"{side}_angle_trigger"] = ObservationChannel(
                            channel_name=f"{side}_angle_trigger", capability="scalar", side=side,
                            value=float(values[side]), frame_name=None, timestamp=now,
                            source_name=self.name, valid=True, critical=False,
                            metadata={"config_root": self._config_root, "rate_hz": self._rate_hz,
                                      "reader_error_count": self._errors},
                        )
            except Exception as exc:  # pragma: no cover - hardware recovery path
                self._errors += 1
                self._publish_invalid(f"{type(exc).__name__}: {exc}")
                try:
                    self._reader.close()
                    if self._running:
                        self._reader.open()
                except Exception:
                    pass
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        with self._lock:
            return {name: channel.copy() for name, channel in self._latest.items()}

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._reader.close()
