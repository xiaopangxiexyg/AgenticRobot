"""Latched bimanual force guard for real RM65 impedance deployment.

The impedance controller already checks its active hard-stop limits before
sending a command.  This deployment-owned monitor gives those samples one
authoritative bimanual emergency-stop path, independent of the policy loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from utils.owned_thread import OwnedThread
import time
from typing import Any, Callable, Mapping

from .arm_safety_runtime import bimanual_emergency_stop


@dataclass(frozen=True)
class ImpedanceForceGuardConfig:
    poll_hz: float = 80.0
    stale_timeout_sec: float = 0.20
    startup_timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        for name in ("poll_hz", "stale_timeout_sec", "startup_timeout_sec"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
            object.__setattr__(self, name, value)
        if self.poll_hz > 200.0:
            raise ValueError("poll_hz must be <= 200 Hz")

    def to_dict(self) -> dict[str, float]:
        return {
            "poll_hz": self.poll_hz,
            "stale_timeout_sec": self.stale_timeout_sec,
            "startup_timeout_sec": self.startup_timeout_sec,
        }


@dataclass(frozen=True)
class ImpedanceForceGuardReading:
    side: str
    force_source: str
    force_norm_n: float
    torque_norm_nm: float
    max_force_n: float
    max_torque_nm: float
    monotonic_time_sec: float
    wall_time_sec: float
    step: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "force_source": self.force_source,
            "force_norm_n": self.force_norm_n,
            "torque_norm_nm": self.torque_norm_nm,
            "max_force_n": self.max_force_n,
            "max_torque_nm": self.max_torque_nm,
            "monotonic_time_sec": self.monotonic_time_sec,
            "wall_time_sec": self.wall_time_sec,
            "step": self.step,
        }


@dataclass(frozen=True)
class ImpedanceForceGuardFault:
    reason: str
    side: str | None
    message: str
    monotonic_time_sec: float
    wall_time_sec: float
    reading: ImpedanceForceGuardReading | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "side": self.side,
            "message": self.message,
            "monotonic_time_sec": self.monotonic_time_sec,
            "wall_time_sec": self.wall_time_sec,
            "reading": None if self.reading is None else self.reading.to_dict(),
        }


class ImpedanceForceGuardTripped(RuntimeError):
    def __init__(self, fault: ImpedanceForceGuardFault):
        super().__init__(fault.message)
        self.fault = fault


class _NoForceSample(RuntimeError):
    pass


class _BackendForceFault(RuntimeError):
    pass


class ImpedanceForceGuardMonitor:
    """Monitor both impedance backends and latch one bimanual stop."""

    def __init__(
        self,
        *,
        readers_by_side: Mapping[str, Callable[[], Mapping[str, Any]]],
        emergency_stop: Callable[[ImpedanceForceGuardFault], None],
        config: ImpedanceForceGuardConfig | None = None,
    ):
        self.config = config if config is not None else ImpedanceForceGuardConfig()
        self._readers = {
            str(side): reader
            for side, reader in dict(readers_by_side).items()
            if callable(reader)
        }
        if not self._readers:
            raise ValueError("impedance force guard requires at least one reader")
        if not callable(emergency_stop):
            raise TypeError("emergency_stop must be callable")
        self._emergency_stop = emergency_stop
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._ready_by_side = {side: threading.Event() for side in self._readers}
        self._threads: list[OwnedThread] = []
        self._worker_lifecycle_lock = threading.RLock()
        self._worker_closed = False
        self._last_reading_by_side: dict[str, ImpedanceForceGuardReading] = {}
        self._last_success_monotonic_by_side: dict[str, float] = {}
        self._last_error_by_side: dict[str, str] = {}
        self._fault: ImpedanceForceGuardFault | None = None
        self._emergency_stop_error: str | None = None

    @property
    def fault(self) -> ImpedanceForceGuardFault | None:
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
    def motion_ready(self) -> bool:
        return self.ready

    def start(self) -> dict[str, object]:
        with self._worker_lifecycle_lock:
            if self._worker_closed:
                raise RuntimeError("worker owner is closed")
            existing_fault = self.fault
            if existing_fault is not None:
                raise ImpedanceForceGuardTripped(existing_fault)
            if self._threads:
                fault = self.fault
                if fault is not None:
                    raise ImpedanceForceGuardTripped(fault)
                return self.snapshot()
            self._stop_event.clear()
            for side, reader in self._readers.items():
                thread = OwnedThread(
                    target=self._poll_loop,
                    args=(side, reader),
                    daemon=True,
                    name=f"rm65_impedance_force_guard_{side}",
                )
                self._threads.append(thread)
                thread.start()

            deadline = time.monotonic() + self.config.startup_timeout_sec
            while not self.ready and self.fault is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = [
                        side
                        for side, event in self._ready_by_side.items()
                        if not event.is_set()
                    ]
                    self._trip(
                        ImpedanceForceGuardFault(
                            reason="startup_timeout",
                            side=None,
                            message=(
                                "RM65 impedance force guard did not receive a valid "
                                f"startup sample for sides={missing} within "
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
                raise ImpedanceForceGuardTripped(fault)
            return self.snapshot()

    def notify_backend_hard_stop(self, side: str, message: str) -> None:
        """Latch the guard from a controller-side hard-stop exception."""

        self._trip(
            ImpedanceForceGuardFault(
                reason="controller_hard_stop",
                side=str(side),
                message=str(message),
                monotonic_time_sec=time.monotonic(),
                wall_time_sec=time.time(),
            )
        )

    def rearm_after_manual_home(
        self,
        *,
        start: bool = True,
    ) -> dict[str, object]:
        with self._worker_lifecycle_lock:
            if self._worker_closed:
                raise RuntimeError("worker owner is closed")
            """Start a new guard generation after explicit backend re-arm."""

            with self._lock:
                fault = self._fault
                stop_error = self._emergency_stop_error
            if fault is None:
                return self.snapshot()
            if stop_error is not None:
                raise RuntimeError(
                    "impedance force guard emergency stop was not acknowledged; "
                    "restart is required: "
                    + stop_error
                )
            self._stop_event.set()
            old_threads = tuple(self._threads)
            self._drain_workers()
            with self._lock:
                self._threads = []
                self._stop_event.clear()
                self._ready_by_side = {side: threading.Event() for side in self._readers}
                self._last_reading_by_side.clear()
                self._last_success_monotonic_by_side.clear()
                self._last_error_by_side.clear()
                self._fault = None
                self._emergency_stop_error = None
            return self.start() if start else self.snapshot()

    def close(self) -> None:
        with self._worker_lifecycle_lock:
            self._worker_closed = True
            self._stop_event.set()
            self._drain_workers()

    def _drain_workers(self) -> None:
        errors: list[BaseException] = []
        for thread in tuple(self._threads):
            try:
                if not thread.join_and_confirm(timeout=1.0):
                    raise RuntimeError(f"impedance force guard worker {thread.name!r} did not stop within 1.000s")
                self._threads.remove(thread)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("impedance force guard worker shutdown unconfirmed: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)) from errors[0]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self.ready,
                "motion_ready": self.motion_ready,
                "config": self.config.to_dict(),
                "last_reading_by_side": {
                    side: reading.to_dict()
                    for side, reading in self._last_reading_by_side.items()
                },
                "last_success_age_by_side": {
                    side: max(0.0, time.monotonic() - timestamp)
                    for side, timestamp in self._last_success_monotonic_by_side.items()
                },
                "last_error_by_side": dict(self._last_error_by_side),
                "fault": None if self._fault is None else self._fault.to_dict(),
                "emergency_stop_error": self._emergency_stop_error,
            }

    def _poll_loop(
        self,
        side: str,
        reader: Callable[[], Mapping[str, Any]],
    ) -> None:
        period_sec = 1.0 / self.config.poll_hz
        while not self._stop_event.is_set() and self.fault is None:
            iteration_start = time.monotonic()
            try:
                reading = self._parse_reading(side, reader())
            except _NoForceSample as exc:
                with self._lock:
                    self._last_error_by_side[side] = str(exc)
                    already_ready = self._ready_by_side[side].is_set()
                if already_ready:
                    self._trip(
                        ImpedanceForceGuardFault(
                            reason="force_sample_unavailable",
                            side=side,
                            message=f"{side} impedance force sample unavailable: {exc}",
                            monotonic_time_sec=time.monotonic(),
                            wall_time_sec=time.time(),
                        )
                    )
            except _BackendForceFault as exc:
                with self._lock:
                    self._last_error_by_side[side] = str(exc)
                self._trip(
                    ImpedanceForceGuardFault(
                        reason="backend_fault",
                        side=side,
                        message=str(exc),
                        monotonic_time_sec=time.monotonic(),
                        wall_time_sec=time.time(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - safety input fails closed.
                with self._lock:
                    self._last_error_by_side[side] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                self._trip(
                    ImpedanceForceGuardFault(
                        reason="force_sample_read_failed",
                        side=side,
                        message=(
                            f"{side} impedance force guard reader failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        monotonic_time_sec=time.monotonic(),
                        wall_time_sec=time.time(),
                    )
                )
            else:
                sample_age = max(0.0, time.monotonic() - reading.monotonic_time_sec)
                exceeded = bool(
                    reading.force_norm_n > reading.max_force_n
                    or reading.torque_norm_nm > reading.max_torque_nm
                )
                with self._lock:
                    self._last_reading_by_side[side] = reading
                    self._last_success_monotonic_by_side[side] = (
                        reading.monotonic_time_sec
                    )
                    self._last_error_by_side.pop(side, None)
                if sample_age > self.config.stale_timeout_sec:
                    self._trip(
                        ImpedanceForceGuardFault(
                            reason="force_sample_stale",
                            side=side,
                            message=(
                                f"{side} impedance force sample is stale: "
                                f"{sample_age:.3f}s > "
                                f"{self.config.stale_timeout_sec:.3f}s"
                            ),
                            monotonic_time_sec=time.monotonic(),
                            wall_time_sec=time.time(),
                            reading=reading,
                        )
                    )
                elif exceeded:
                    self._trip(
                        ImpedanceForceGuardFault(
                            reason="force_limit_exceeded",
                            side=side,
                            message=(
                                f"{side} RM65 impedance force guard exceeded on "
                                f"{reading.force_source}: |F|={reading.force_norm_n:.3f}N "
                                f"limit={reading.max_force_n:.3f}N, "
                                f"|T|={reading.torque_norm_nm:.3f}Nm "
                                f"limit={reading.max_torque_nm:.3f}Nm"
                            ),
                            monotonic_time_sec=reading.monotonic_time_sec,
                            wall_time_sec=reading.wall_time_sec,
                            reading=reading,
                        )
                    )
                else:
                    self._ready_by_side[side].set()
            remaining = period_sec - (time.monotonic() - iteration_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    @staticmethod
    def _parse_reading(
        side: str,
        payload: Mapping[str, Any],
    ) -> ImpedanceForceGuardReading:
        if not isinstance(payload, Mapping):
            raise TypeError("impedance force guard reader must return a mapping")
        backend_fault = payload.get("backend_fault")
        if backend_fault:
            raise _BackendForceFault(
                f"{side} impedance backend fault: {backend_fault}"
            )
        sample = payload.get("sample")
        if sample is None:
            raise _NoForceSample(str(payload.get("sample_error") or "no sample yet"))
        if not isinstance(sample, Mapping):
            raise TypeError("impedance force guard sample must be a mapping")

        def finite(name: str) -> float:
            value = float(sample.get(name))
            if not math.isfinite(value):
                raise ValueError(f"{side} impedance force {name} is not finite")
            return value

        force_norm = finite("force_norm_n")
        torque_norm = finite("torque_norm_nm")
        max_force = finite("max_force_n")
        max_torque = finite("max_torque_nm")
        if max_force <= 0.0 or max_torque <= 0.0:
            raise ValueError(f"{side} impedance force guard limits must be positive")
        monotonic_time = finite("monotonic_time_sec")
        wall_time = finite("wall_time_sec")
        raw_step = sample.get("step")
        step = None if raw_step is None else int(raw_step)
        return ImpedanceForceGuardReading(
            side=str(sample.get("side") or side),
            force_source=str(sample.get("force_source") or "unknown"),
            force_norm_n=force_norm,
            torque_norm_nm=torque_norm,
            max_force_n=max_force,
            max_torque_nm=max_torque,
            monotonic_time_sec=monotonic_time,
            wall_time_sec=wall_time,
            step=step,
        )

    def _trip(self, fault: ImpedanceForceGuardFault) -> None:
        with self._lock:
            if self._fault is not None:
                return
            self._fault = fault
            self._stop_event.set()
        try:
            self._emergency_stop(fault)
        except Exception as exc:  # noqa: BLE001 - the fault remains latched.
            with self._lock:
                self._emergency_stop_error = f"{type(exc).__name__}: {exc}"


__all__ = [
    "ImpedanceForceGuardConfig",
    "ImpedanceForceGuardFault",
    "ImpedanceForceGuardMonitor",
    "ImpedanceForceGuardReading",
    "ImpedanceForceGuardTripped",
    "bimanual_emergency_stop",
]
