"""Latched force-limit guard shared by real RM65 control frontends.

The guard is intentionally independent of the position command clock.  Force
samples are polled at a bounded low rate and a trip invokes a single bimanual
emergency-stop callback.  RM65 deployments may share the vendor SDK handle
between force, state, and command calls; therefore the live 60 Hz reference /
120 Hz command profile leaves this monitor disabled by default and only uses
it for explicit diagnostics or non-following paths.  A trip remains latched
until an operator completes a fresh runtime-home calibration and explicitly
re-arms the monitor; no automatic recovery is permitted.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading

from utils.owned_thread import OwnedThread
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


POSITION_FORCE_SOURCES = ("raw", "zero", "work", "tool")
# trip: a reading older than stale_timeout_sec latches a fault and stops both
# arms (historical default). warn: staleness is logged and counted only; it never
# latches a fault, blocks motion admission or stops an executing motion. Force or
# torque over the limits and consecutive reader errors still trip in both modes.
POSITION_FORCE_STALE_POLICIES = ("trip", "warn")
_STALE_WARNING_LOG_INTERVAL_SEC = 5.0
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionForceGuardConfig:
    source: str = "tool"
    max_force_n: float = 30.0
    max_torque_nm: float = 5.0
    poll_hz: float = 20.0
    consecutive_limit_samples: int = 1
    max_consecutive_read_errors: int = 3
    stale_timeout_sec: float = 0.4
    startup_timeout_sec: float = 2.0
    stale_policy: str = "trip"

    def __post_init__(self) -> None:
        if self.stale_policy not in POSITION_FORCE_STALE_POLICIES:
            raise ValueError(
                f"position force stale_policy must be one of {POSITION_FORCE_STALE_POLICIES}, "
                f"got {self.stale_policy!r}"
            )
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
            "stale_policy": self.stale_policy,
        }


def position_force_guard_config(
    args: argparse.Namespace,
) -> PositionForceGuardConfig:
    return PositionForceGuardConfig(
        source=str(getattr(args, "position_force_guard_source", "tool")),
        max_force_n=float(
            getattr(args, "position_force_guard_max_force_n", 30.0)
        ),
        max_torque_nm=float(
            getattr(args, "position_force_guard_max_torque_nm", 5.0)
        ),
        poll_hz=float(getattr(args, "position_force_guard_hz", 20.0)),
        consecutive_limit_samples=int(
            getattr(args, "position_force_guard_consecutive_samples", 1)
        ),
        max_consecutive_read_errors=int(
            getattr(args, "position_force_guard_max_read_errors", 3)
        ),
        stale_timeout_sec=float(
            getattr(args, "position_force_guard_stale_sec", 0.4)
        ),
        startup_timeout_sec=float(
            getattr(args, "position_force_guard_startup_timeout_sec", 2.0)
        ),
        stale_policy=str(getattr(args, "position_force_guard_stale_policy", "trip")),
    )


@dataclass(frozen=True)
class PositionForceReading:
    side: str
    source: str
    values: tuple[float, ...]
    force_norm_n: float
    torque_norm_nm: float
    monotonic_time_sec: float
    wall_time_sec: float
    read_elapsed_sec: float | None = None
    read_phase_timing: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "source": self.source,
            "values": list(self.values),
            "force_norm_n": self.force_norm_n,
            "torque_norm_nm": self.torque_norm_nm,
            "monotonic_time_sec": self.monotonic_time_sec,
            "wall_time_sec": self.wall_time_sec,
            "read_elapsed_sec": self.read_elapsed_sec,
            "read_phase_timing": dict(self.read_phase_timing),
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
        diagnostic_enabled: Callable[[], bool] | None = None,
        diagnostic_sink: Callable[[str, Mapping[str, object]], None] | None = None,
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
        if (diagnostic_enabled is None) != (diagnostic_sink is None):
            raise ValueError("force diagnostics require both an enable check and a sink")
        self._emergency_stop = emergency_stop
        self._diagnostic_enabled = diagnostic_enabled
        self._diagnostic_sink = diagnostic_sink
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stale_watchdog_armed = threading.Event()
        self._stale_watchdog_generation = 0
        self._non_motion_read_window_active = False
        self._ready_by_side = {
            side: threading.Event() for side in self._readers
        }
        self._threads: list[OwnedThread] = []
        self._watchdog_thread: OwnedThread | None = None
        self._last_reading_by_side: dict[str, PositionForceReading] = {}
        self._last_success_monotonic_by_side: dict[str, float] = {}
        self._reader_inflight_since_by_side: dict[str, float] = {}
        self._consecutive_limit_by_side = {side: 0 for side in self._readers}
        self._consecutive_errors_by_side = {side: 0 for side in self._readers}
        self._last_error_by_side: dict[str, str] = {}
        self._fault: PositionForceGuardFault | None = None
        self._emergency_stop_error: str | None = None
        self._stale_warning_count_by_side = {side: 0 for side in self._readers}
        self._last_stale_warning_by_side: dict[str, str] = {}
        self._last_stale_sample_by_side: dict[str, float | None] = {}
        self._last_stale_log_by_side: dict[str, float] = {}

    def _diagnostics_active(self) -> bool:
        if self._diagnostic_enabled is None or self._diagnostic_sink is None:
            return False
        try:
            return bool(self._diagnostic_enabled())
        except Exception:
            return False  # Diagnostics cannot affect the safety monitor.

    def _emit_diagnostic(self, event: str, payload: Mapping[str, object]) -> None:
        if not self._diagnostics_active():
            return
        try:
            sink = self._diagnostic_sink
            if sink is not None:
                sink(event, payload)
        except Exception:
            pass  # A full or failed debug sink must not delay force protection.

    def _reader_stack(self, side: str) -> list[dict[str, object]]:
        thread = next(
            (
                item
                for item in self._threads
                if item.name == f"rm65_position_force_guard_{side}"
            ),
            None,
        )
        frame = (
            None
            if thread is None or thread.ident is None
            else sys._current_frames().get(thread.ident)
        )
        stack: list[dict[str, object]] = []
        while frame is not None and len(stack) < 12:
            stack.append(
                {
                    "file": frame.f_code.co_filename,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                }
            )
            frame = frame.f_back
        return stack

    def _stale_diagnostic(
        self,
        *,
        side: str,
        now: float,
        age_sec: float,
        inflight_since: float | None,
        last_reading: PositionForceReading | None,
    ) -> dict[str, object] | None:
        if not self._diagnostics_active():
            return None
        try:
            return {
                "side": side,
                "last_success_age_sec": age_sec,
                "inflight_age_sec": (
                    None if inflight_since is None else max(0.0, now - inflight_since)
                ),
                "gap_before_inflight_sec": (
                    None
                    if inflight_since is None or last_reading is None
                    else max(0.0, inflight_since - last_reading.monotonic_time_sec)
                ),
                "last_success_read_elapsed_sec": (
                    None if last_reading is None else last_reading.read_elapsed_sec
                ),
                "last_success_phase_timing": (
                    {} if last_reading is None else dict(last_reading.read_phase_timing)
                ),
                "reader_stack": self._reader_stack(side),
            }
        except Exception as exc:
            return {"side": side, "diagnostic_error": f"{type(exc).__name__}: {exc}"}

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

    def suspend_stale_watchdog_for_non_motion(self) -> int:
        """Pause only stale trips while a caller performs blocking reads.

        This window is valid only while no motion can be admitted. Force/torque
        limits and consecutive reader errors remain active in the pollers. The
        returned generation token must be passed to
        :meth:`resume_stale_watchdog_after_non_motion`, which requires a new
        valid force sample from every side before motion readiness is restored.
        """

        if not self._threads:
            raise RuntimeError(
                "position force guard must be started before a non-motion read window"
            )
        with self._lock:
            if self._fault is not None:
                raise PositionForceGuardTripped(self._fault)
            if self._non_motion_read_window_active:
                raise RuntimeError(
                    "position force guard non-motion read window is already active"
                )
            self._non_motion_read_window_active = True
            self._stale_watchdog_armed.clear()
            self._stale_watchdog_generation += 1
            return int(self._stale_watchdog_generation)

    def resume_stale_watchdog_after_non_motion(
        self,
        token: int,
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, object]:
        """Require post-window force samples before restoring motion readiness."""

        with self._lock:
            if self._fault is not None:
                raise PositionForceGuardTripped(self._fault)
            if not self._non_motion_read_window_active:
                raise RuntimeError(
                    "position force guard non-motion read window is not active"
                )
            if int(token) != int(self._stale_watchdog_generation):
                raise RuntimeError(
                    "position force guard non-motion read window token is stale"
                )
            self._non_motion_read_window_active = False
        return self.arm_stale_watchdog(timeout_sec=timeout_sec)

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
            thread = OwnedThread(
                target=self._poll_loop,
                args=(side, reader),
                daemon=True,
                name=f"rm65_position_force_guard_{side}",
            )
            self._threads.append(thread)
            thread.start()
        self._watchdog_thread = OwnedThread(
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
            with self._lock:
                self._stale_watchdog_generation += 1
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
        if self.config.stale_policy == "warn":
            # Staleness never gates motion in warn mode: no fresh-sample wait,
            # no arm timeout. Limits and reader errors stay active.
            with self._lock:
                self._stale_watchdog_generation += 1
                self._stale_watchdog_armed.set()
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
                with self._lock:
                    self._stale_watchdog_generation += 1
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

    def rearm_after_manual_home(
        self,
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, object]:
        """Restart sampling after an acknowledged, operator-led Home.

        This is deliberately a separate lifecycle transition from `start`.
        The force-trip callback has already latched the monitor and stopped the
        command backends; clearing that latch without a new Home would allow a
        stale target to be admitted.  The caller must therefore invoke this
        only after the real-arm qpos has been refreshed and every command
        backend has been manually re-armed.
        """

        with self._lock:
            fault = self._fault
            stop_error = self._emergency_stop_error
        if fault is None:
            return self.snapshot()
        if stop_error is not None:
            raise RuntimeError(
                "position force guard emergency stop was not acknowledged; "
                "restart is required before motion can resume: " + stop_error
            )

        # A fault causes all poll/watchdog loops to leave their normal loop
        # condition.  Join them before replacing the Events so no old worker
        # can publish a sample into the new generation.
        self._stop_event.set()
        old_threads = tuple(self._threads)
        old_watchdog = self._watchdog_thread
        alive = [
            thread.name
            for thread in (*old_threads, *((old_watchdog,) if old_watchdog is not None else ()))
            if not thread.join_and_confirm(timeout=1.0)
        ]
        if alive:
            raise RuntimeError(
                "position force guard worker(s) did not stop before manual re-arm: "
                + ", ".join(alive)
            )

        with self._lock:
            self._threads = []
            self._watchdog_thread = None
            self._stop_event.clear()
            self._stale_watchdog_armed.clear()
            self._stale_watchdog_generation += 1
            self._non_motion_read_window_active = False
            self._ready_by_side = {
                side: threading.Event() for side in self._readers
            }
            self._last_reading_by_side.clear()
            self._last_success_monotonic_by_side.clear()
            self._reader_inflight_since_by_side.clear()
            self._consecutive_limit_by_side = {side: 0 for side in self._readers}
            self._consecutive_errors_by_side = {side: 0 for side in self._readers}
            self._last_error_by_side.clear()
            self._fault = None
            self._emergency_stop_error = None

        # Keep force-limit and read-error protection active immediately, but
        # require fresh post-Home samples before enabling the stale watchdog.
        self.start(arm_stale_watchdog=False)
        return self.arm_stale_watchdog(timeout_sec=timeout_sec)

    def close(self) -> None:
        self._stop_event.set()
        remaining = [
            thread for thread in tuple(self._threads)
            if not thread.join_and_confirm(timeout=1.0)
        ]
        watchdog = self._watchdog_thread
        if watchdog is not None and watchdog.join_and_confirm(timeout=1.0):
            self._watchdog_thread = None
        self._threads = remaining
        alive = [thread.name for thread in (
            *remaining,
            *((self._watchdog_thread,) if self._watchdog_thread is not None else ()),
        )]
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
                "stale_watchdog_generation": int(
                    self._stale_watchdog_generation
                ),
                "non_motion_read_window_active": bool(
                    self._non_motion_read_window_active
                ),
                "motion_ready": self.motion_ready,
                "config": self.config.to_dict(),
                "last_reading_by_side": {
                    side: reading.to_dict()
                    for side, reading in self._last_reading_by_side.items()
                },
                "consecutive_limit_by_side": dict(self._consecutive_limit_by_side),
                "consecutive_errors_by_side": dict(self._consecutive_errors_by_side),
                "last_error_by_side": dict(self._last_error_by_side),
                "reader_inflight_age_by_side": {
                    side: max(0.0, time.monotonic() - started)
                    for side, started in self._reader_inflight_since_by_side.items()
                },
                "stale_warning_count_by_side": dict(self._stale_warning_count_by_side),
                "last_stale_warning_by_side": dict(self._last_stale_warning_by_side),
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
            with self._lock:
                self._reader_inflight_since_by_side[side] = iteration_start
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
                    previous_success = self._last_success_monotonic_by_side.get(side)
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
                    gap_sec = (
                        None
                        if previous_success is None
                        else max(0.0, iteration_start - previous_success)
                    )
                    if self.fault is None and (
                        (reading.read_elapsed_sec or 0.0) >= period_sec
                        or (gap_sec or 0.0) >= 2.0 * period_sec
                    ):
                        self._emit_diagnostic(
                            "force_read_slow",
                            {
                                "side": side,
                                "read_elapsed_sec": reading.read_elapsed_sec,
                                "gap_before_read_sec": gap_sec,
                                "read_phase_timing": dict(reading.read_phase_timing),
                            },
                        )
            finally:
                with self._lock:
                    self._reader_inflight_since_by_side.pop(side, None)
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
                watchdog_generation = int(self._stale_watchdog_generation)
                stale = [
                    (side, now - timestamp)
                    for side, timestamp in self._last_success_monotonic_by_side.items()
                    if self._ready_by_side[side].is_set()
                    and now - timestamp > float(self.config.stale_timeout_sec)
                ]
            if stale and self.config.stale_policy == "warn":
                self._warn_stale(stale, now)
                continue
            if stale:
                side, age_sec = max(stale, key=lambda item: item[1])
                with self._lock:
                    inflight_since = self._reader_inflight_since_by_side.get(side)
                    last_reading = self._last_reading_by_side.get(side)
                inflight_text = (
                    f"; force reader total time in flight for "
                    f"{max(0.0, now - inflight_since):.3f}s"
                    if inflight_since is not None
                    else ""
                )
                elapsed_text = (
                    f"; last successful read_elapsed="
                    f"{float(last_reading.read_elapsed_sec):.3f}s"
                    if last_reading is not None
                    and last_reading.read_elapsed_sec is not None
                    else ""
                )
                diagnostic = self._stale_diagnostic(
                    side=side,
                    now=now,
                    age_sec=age_sec,
                    inflight_since=inflight_since,
                    last_reading=last_reading,
                )
                tripped = self._trip(
                    PositionForceGuardFault(
                        reason="force_read_stale",
                        side=side,
                        message=(
                            f"{side} RM65 force reading is stale: "
                            f"{age_sec:.3f}s > "
                            f"{self.config.stale_timeout_sec:.3f}s"
                            f"{inflight_text}{elapsed_text}"
                        ),
                        monotonic_time_sec=now,
                        wall_time_sec=time.time(),
                    ),
                    stale_watchdog_generation=watchdog_generation,
                )
                if tripped:
                    if diagnostic is not None:
                        self._emit_diagnostic("force_read_stale", diagnostic)
                    return

    def _warn_stale(self, stale: list[tuple[str, float]], now: float) -> None:
        """Record stale force samples without any fault, stop or admission effect."""
        for side, age_sec in stale:
            with self._lock:
                inflight_since = self._reader_inflight_since_by_side.get(side)
                last_success = self._last_success_monotonic_by_side.get(side)
                # Count one event per stale sample, not per watchdog tick.
                if side in self._last_stale_sample_by_side and self._last_stale_sample_by_side[side] == last_success:
                    continue
                self._last_stale_sample_by_side[side] = last_success
                self._stale_warning_count_by_side[side] = self._stale_warning_count_by_side.get(side, 0) + 1
                count = self._stale_warning_count_by_side[side]
                message = (f"{side} RM65 force reading is stale: {age_sec:.3f}s > "
                           f"{self.config.stale_timeout_sec:.3f}s"
                           + ("" if inflight_since is None else
                              f"; reader in flight {max(0.0, now - inflight_since):.3f}s")
                           + "; stale_policy=warn (no fault, no stop, motion unaffected)")
                self._last_stale_warning_by_side[side] = message
                last_log = self._last_stale_log_by_side.get(side)
                log_now = last_log is None or now - last_log >= _STALE_WARNING_LOG_INTERVAL_SEC
                if log_now:
                    self._last_stale_log_by_side[side] = now
            if log_now:
                _LOGGER.warning("%s; stale events on %s so far: %d", message, side, count)
            self._emit_diagnostic("force_read_stale_warning",
                                  {"side": side, "age_sec": age_sec, "count": count})

    def _trip(
        self,
        fault: PositionForceGuardFault,
        *,
        stale_watchdog_generation: int | None = None,
    ) -> bool:
        with self._lock:
            if self._fault is not None:
                return False
            if stale_watchdog_generation is not None and (
                not self._stale_watchdog_armed.is_set()
                or int(stale_watchdog_generation)
                != int(self._stale_watchdog_generation)
            ):
                return False
            self._fault = fault
        try:
            self._emergency_stop(fault)
        except Exception as exc:  # noqa: BLE001 - the fault remains latched.
            with self._lock:
                self._emergency_stop_error = f"{type(exc).__name__}: {exc}"
        return True

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
        raw_read_elapsed = payload.get("read_elapsed_sec")
        read_elapsed_sec = None
        if raw_read_elapsed is not None:
            read_elapsed_sec = float(raw_read_elapsed)
            if not math.isfinite(read_elapsed_sec) or read_elapsed_sec < 0.0:
                raise ValueError("force reader read_elapsed_sec must be finite and non-negative")
        raw_phase_timing = payload.get("read_phase_timing", {})
        if not isinstance(raw_phase_timing, Mapping):
            raise ValueError("force reader read_phase_timing must be a mapping")
        read_phase_timing: list[tuple[str, float]] = []
        for raw_name, raw_value in raw_phase_timing.items():
            name = str(raw_name)
            value = float(raw_value)
            if not name or not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "force reader read_phase_timing values must be finite and non-negative"
                )
            read_phase_timing.append((name, value))
        return PositionForceReading(
            side=side,
            source=self.config.source,
            values=tuple(float(value) for value in values),
            force_norm_n=float(np.linalg.norm(values[:3])),
            torque_norm_nm=float(np.linalg.norm(values[3:])),
            monotonic_time_sec=monotonic_time_sec,
            wall_time_sec=wall_time_sec,
            read_elapsed_sec=read_elapsed_sec,
            read_phase_timing=tuple(read_phase_timing),
        )


__all__ = [
    "POSITION_FORCE_SOURCES",
    "POSITION_FORCE_STALE_POLICIES",
    "PositionForceGuardConfig",
    "PositionForceGuardFault",
    "PositionForceGuardMonitor",
    "PositionForceGuardTripped",
    "PositionForceReading",
    "position_force_guard_config",
]
