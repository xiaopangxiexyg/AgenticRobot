"""Latched force-limit guard for real RM65 position teleoperation.

The guard is intentionally independent of the position command clock.  Force
samples come from the per-arm observation connections, while a trip invokes a
single bimanual emergency-stop callback.  Once tripped, the monitor cannot be
re-armed; restarting the runtime is required.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


POSITION_FORCE_SOURCES = ("raw", "zero", "work", "tool")


@dataclass(frozen=True)
class PositionForceGuardConfig:
    source: str = "tool"
    max_force_n: float = 15.0
    max_torque_nm: float = 2.0
    poll_hz: float = 20.0
    consecutive_limit_samples: int = 1
    max_consecutive_read_errors: int = 3
    stale_timeout_sec: float = 0.25
    startup_timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        source = str(self.source or "tool").strip()
        if source not in POSITION_FORCE_SOURCES:
            raise ValueError(
                f"position force source must be one of {POSITION_FORCE_SOURCES}, "
                f"got {source!r}"
            )
        for name in (
            "max_force_n",
            "max_torque_nm",
            "poll_hz",
            "stale_timeout_sec",
            "startup_timeout_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
            object.__setattr__(self, name, value)
        # The RM65 SDK documents a minimum 50 ms period for repeated force
        # queries.  Reject unsafe configurations instead of silently clamping.
        if float(self.poll_hz) > 20.0:
            raise ValueError("poll_hz must be <= 20 Hz for RM65 Get_Force_Data")
        for name in ("consecutive_limit_samples", "max_consecutive_read_errors"):
            raw_value = getattr(self, name)
            value = int(raw_value)
            if (
                isinstance(raw_value, bool)
                or float(raw_value) != float(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "max_force_n": self.max_force_n,
            "max_torque_nm": self.max_torque_nm,
            "poll_hz": self.poll_hz,
            "consecutive_limit_samples": self.consecutive_limit_samples,
            "max_consecutive_read_errors": self.max_consecutive_read_errors,
            "stale_timeout_sec": self.stale_timeout_sec,
            "startup_timeout_sec": self.startup_timeout_sec,
        }


@dataclass(frozen=True)
class PositionForceReading:
    side: str
    source: str
    values: tuple[float, ...]
    force_norm_n: float
    torque_norm_nm: float
    monotonic_time_sec: float
    wall_time_sec: float

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "source": self.source,
            "values": list(self.values),
            "force_norm_n": self.force_norm_n,
            "torque_norm_nm": self.torque_norm_nm,
            "monotonic_time_sec": self.monotonic_time_sec,
            "wall_time_sec": self.wall_time_sec,
        }


@dataclass(frozen=True)
class PositionForceGuardFault:
    reason: str
    side: str | None
    message: str
    monotonic_time_sec: float
    wall_time_sec: float
    reading: PositionForceReading | None = None
    consecutive_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "side": self.side,
            "message": self.message,
            "monotonic_time_sec": self.monotonic_time_sec,
            "wall_time_sec": self.wall_time_sec,
            "reading": None if self.reading is None else self.reading.to_dict(),
            "consecutive_count": self.consecutive_count,
        }


class PositionForceGuardTripped(RuntimeError):
    def __init__(self, fault: PositionForceGuardFault):
        super().__init__(fault.message)
        self.fault = fault


class PositionForceGuardMonitor:
    """Poll both arms and invoke one latched bimanual stop on any fault."""

    def __init__(
        self,
        *,
        readers_by_side: Mapping[str, Callable[[], Mapping[str, Any]]],
        emergency_stop: Callable[[PositionForceGuardFault], None],
        config: PositionForceGuardConfig | None = None,
    ):
        self.config = config if config is not None else PositionForceGuardConfig()
        self._readers = {
            str(side): reader
            for side, reader in dict(readers_by_side).items()
            if callable(reader)
        }
        if not self._readers:
            raise ValueError("position force guard requires at least one force reader")
        if not callable(emergency_stop):
            raise TypeError("emergency_stop must be callable")
        self._emergency_stop = emergency_stop
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stale_watchdog_armed = threading.Event()
        self._ready_by_side = {
            side: threading.Event() for side in self._readers
        }
        self._threads: list[threading.Thread] = []
        self._watchdog_thread: threading.Thread | None = None
        self._last_reading_by_side: dict[str, PositionForceReading] = {}
        self._last_success_monotonic_by_side: dict[str, float] = {}
        self._consecutive_limit_by_side = {side: 0 for side in self._readers}
        self._consecutive_errors_by_side = {side: 0 for side in self._readers}
        self._last_error_by_side: dict[str, str] = {}
        self._fault: PositionForceGuardFault | None = None
        self._emergency_stop_error: str | None = None

    @property
    def fault(self) -> PositionForceGuardFault | None:
        with self._lock:
            return self._fault

    @property
    def ready(self) -> bool:
        return bool(
            self._ready_by_side
            and all(event.is_set() for event in self._ready_by_side.values())
            and self.fault is None
        )

    @property
    def stale_watchdog_armed(self) -> bool:
        return self._stale_watchdog_armed.is_set()

    @property
    def motion_ready(self) -> bool:
        return bool(self.ready and self.stale_watchdog_armed)

    def start(
        self,
        *,
        arm_stale_watchdog: bool = True,
    ) -> dict[str, object]:
        if self._threads:
            if self.fault is not None:
                raise PositionForceGuardTripped(self.fault)
            if arm_stale_watchdog and not self.stale_watchdog_armed:
                return self.arm_stale_watchdog()
            return self.snapshot()
        self._stop_event.clear()
        self._stale_watchdog_armed.clear()
        for side, reader in self._readers.items():
            thread = threading.Thread(
                target=self._poll_loop,
                args=(side, reader),
                daemon=True,
                name=f"rm65_position_force_guard_{side}",
            )
            thread.start()
            self._threads.append(thread)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="rm65_position_force_guard_watchdog",
        )
        self._watchdog_thread.start()

        deadline = time.monotonic() + float(self.config.startup_timeout_sec)
        while not self.ready and self.fault is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                missing = [
                    side
                    for side, event in self._ready_by_side.items()
                    if not event.is_set()
                ]
                self._trip(
                    PositionForceGuardFault(
                        reason="startup_timeout",
                        side=None,
                        message=(
                            "RM65 position force guard did not receive a valid startup "
                            f"sample for sides={missing} within "
                            f"{self.config.startup_timeout_sec:.3f}s"
                        ),
                        monotonic_time_sec=time.monotonic(),
                        wall_time_sec=time.time(),
                    )
                )
                break
            self._stop_event.wait(min(0.01, remaining))
        fault = self.fault
        if fault is not None:
            raise PositionForceGuardTripped(fault)
        if arm_stale_watchdog:
            # Startup readiness already guarantees one valid sample per side.
            # Immediate arming preserves the historical standalone behavior.
            self._stale_watchdog_armed.set()
        return self.snapshot()

    def arm_stale_watchdog(
        self,
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, object]:
        """Require a post-request sample per side, then enable stale trips.

        Real-robot startup may deliberately defer only the stale watchdog
        while blocking camera SDK initialization runs.  Force limits and
        consecutive read errors remain active throughout that interval.
        Before any motion is admitted, this method proves that every reader
        has produced a new valid sample and only then enables stale checking.
        """

        if not self._threads:
            raise RuntimeError(
                "position force guard must be started before arming stale watchdog"
            )
        fault = self.fault
        if fault is not None:
            raise PositionForceGuardTripped(fault)
        if self.stale_watchdog_armed:
            return self.snapshot()
        requested_at = time.monotonic()
        resolved_timeout = (
            float(self.config.startup_timeout_sec)
            if timeout_sec is None
            else float(timeout_sec)
        )
        if not math.isfinite(resolved_timeout) or resolved_timeout <= 0.0:
            raise ValueError("stale watchdog arm timeout must be positive and finite")
        deadline = requested_at + resolved_timeout
        while self.fault is None:
            with self._lock:
                missing = [
                    side
                    for side in self._readers
                    if float(
                        self._last_success_monotonic_by_side.get(side, 0.0)
                    )
                    <= requested_at
                ]
            if not missing:
                self._stale_watchdog_armed.set()
                return self.snapshot()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._trip(
                    PositionForceGuardFault(
                        reason="stale_watchdog_arm_timeout",
                        side=None,
                        message=(
                            "RM65 position force guard could not obtain fresh "
                            f"post-startup sample(s) for sides={missing} within "
                            f"{resolved_timeout:.3f}s; stale watchdog was not armed"
                        ),
                        monotonic_time_sec=time.monotonic(),
                        wall_time_sec=time.time(),
                    )
                )
                break
            if self._stop_event.wait(min(0.01, remaining)):
                raise RuntimeError(
                    "position force guard stopped while arming stale watchdog"
                )
        fault = self.fault
        if fault is not None:
            raise PositionForceGuardTripped(fault)
        raise RuntimeError("position force guard stale watchdog arming failed")

    def close(self) -> None:
        self._stop_event.set()
        for thread in tuple(self._threads):
            thread.join(timeout=1.0)
        watchdog = self._watchdog_thread
        if watchdog is not None:
            watchdog.join(timeout=1.0)
        alive = [
            thread.name
            for thread in (*self._threads, *((watchdog,) if watchdog is not None else ()))
            if thread.is_alive()
        ]
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        self._watchdog_thread = (
            watchdog if watchdog is not None and watchdog.is_alive() else None
        )
        if alive:
            raise RuntimeError(
                "position force guard worker(s) did not stop within 1.000s: "
                + ", ".join(alive)
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self.ready,
                "stale_watchdog_armed": self.stale_watchdog_armed,
                "motion_ready": self.motion_ready,
                "config": self.config.to_dict(),
                "last_reading_by_side": {
                    side: reading.to_dict()
                    for side, reading in self._last_reading_by_side.items()
                },
                "consecutive_limit_by_side": dict(self._consecutive_limit_by_side),
                "consecutive_errors_by_side": dict(self._consecutive_errors_by_side),
                "last_error_by_side": dict(self._last_error_by_side),
                "fault": None if self._fault is None else self._fault.to_dict(),
                "emergency_stop_error": self._emergency_stop_error,
            }

    def _poll_loop(
        self,
        side: str,
        reader: Callable[[], Mapping[str, Any]],
    ) -> None:
        period_sec = 1.0 / float(self.config.poll_hz)
        while not self._stop_event.is_set() and self.fault is None:
            iteration_start = time.monotonic()
            try:
                payload = reader()
                reading = self._parse_reading(side, payload)
            except Exception as exc:  # noqa: BLE001 - read failures are safety inputs.
                with self._lock:
                    count = int(self._consecutive_errors_by_side[side]) + 1
                    self._consecutive_errors_by_side[side] = count
                    self._last_error_by_side[side] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                if count >= int(self.config.max_consecutive_read_errors):
                    self._trip(
                        PositionForceGuardFault(
                            reason="force_read_failed",
                            side=side,
                            message=(
                                f"{side} RM65 force read failed for {count} consecutive "
                                f"sample(s): {type(exc).__name__}: {exc}"
                            ),
                            monotonic_time_sec=time.monotonic(),
                            wall_time_sec=time.time(),
                            consecutive_count=count,
                        )
                    )
            else:
                exceeded = bool(
                    reading.force_norm_n > float(self.config.max_force_n)
                    or reading.torque_norm_nm > float(self.config.max_torque_nm)
                )
                with self._lock:
                    self._last_reading_by_side[side] = reading
                    self._last_success_monotonic_by_side[side] = (
                        reading.monotonic_time_sec
                    )
                    self._consecutive_errors_by_side[side] = 0
                    self._last_error_by_side.pop(side, None)
                    count = (
                        int(self._consecutive_limit_by_side[side]) + 1
                        if exceeded
                        else 0
                    )
                    self._consecutive_limit_by_side[side] = count
                if (
                    exceeded
                    and count >= int(self.config.consecutive_limit_samples)
                ):
                    self._trip(
                        PositionForceGuardFault(
                            reason="force_limit_exceeded",
                            side=side,
                            message=(
                                f"{side} RM65 position force guard exceeded on "
                                f"{reading.source}: |F|={reading.force_norm_n:.3f}N "
                                f"limit={self.config.max_force_n:.3f}N, "
                                f"|T|={reading.torque_norm_nm:.3f}Nm "
                                f"limit={self.config.max_torque_nm:.3f}Nm "
                                f"for {count} consecutive sample(s)"
                            ),
                            monotonic_time_sec=reading.monotonic_time_sec,
                            wall_time_sec=reading.wall_time_sec,
                            reading=reading,
                            consecutive_count=count,
                        )
                    )
                else:
                    # Do not publish startup readiness for the sample that
                    # trips the guard.  Otherwise start() can race the polling
                    # thread and briefly admit an already-over-limit arm.
                    self._ready_by_side[side].set()
            remaining = period_sec - (time.monotonic() - iteration_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def _watchdog_loop(self) -> None:
        interval_sec = min(0.025, float(self.config.stale_timeout_sec) / 4.0)
        while not self._stop_event.wait(interval_sec):
            if self.fault is not None:
                return
            if not self.stale_watchdog_armed:
                continue
            now = time.monotonic()
            with self._lock:
                stale = [
                    (side, now - timestamp)
                    for side, timestamp in self._last_success_monotonic_by_side.items()
                    if self._ready_by_side[side].is_set()
                    and now - timestamp > float(self.config.stale_timeout_sec)
                ]
            if stale:
                side, age_sec = max(stale, key=lambda item: item[1])
                self._trip(
                    PositionForceGuardFault(
                        reason="force_read_stale",
                        side=side,
                        message=(
                            f"{side} RM65 force reading is stale: "
                            f"{age_sec:.3f}s > "
                            f"{self.config.stale_timeout_sec:.3f}s"
                        ),
                        monotonic_time_sec=now,
                        wall_time_sec=time.time(),
                    )
                )
                return

    def _trip(self, fault: PositionForceGuardFault) -> None:
        with self._lock:
            if self._fault is not None:
                return
            self._fault = fault
        try:
            self._emergency_stop(fault)
        except Exception as exc:  # noqa: BLE001 - the fault remains latched.
            with self._lock:
                self._emergency_stop_error = f"{type(exc).__name__}: {exc}"

    def _parse_reading(
        self,
        side: str,
        payload: Mapping[str, Any],
    ) -> PositionForceReading:
        if not isinstance(payload, Mapping):
            raise TypeError("force reader must return a mapping")
        sources = payload.get("sources", payload)
        if not isinstance(sources, Mapping):
            raise ValueError("force reader payload does not contain source mappings")
        values = np.asarray(sources.get(self.config.source), dtype=float).reshape(-1)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{side} {self.config.source} force must contain 6 finite values"
            )
        monotonic_time_sec = float(payload.get("monotonic_time_sec", time.monotonic()))
        wall_time_sec = float(payload.get("wall_time_sec", time.time()))
        if not math.isfinite(monotonic_time_sec) or not math.isfinite(wall_time_sec):
            raise ValueError("force reader timestamps must be finite")
        return PositionForceReading(
            side=side,
            source=self.config.source,
            values=tuple(float(value) for value in values),
            force_norm_n=float(np.linalg.norm(values[:3])),
            torque_norm_nm=float(np.linalg.norm(values[3:])),
            monotonic_time_sec=monotonic_time_sec,
            wall_time_sec=wall_time_sec,
        )


__all__ = [
    "POSITION_FORCE_SOURCES",
    "PositionForceGuardConfig",
    "PositionForceGuardFault",
    "PositionForceGuardMonitor",
    "PositionForceGuardTripped",
    "PositionForceReading",
]
