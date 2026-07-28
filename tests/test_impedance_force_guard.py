from __future__ import annotations

import threading
import time

import pytest

from teleop_runtime.deploy.impedance_force_guard import (
    ImpedanceForceGuardConfig,
    ImpedanceForceGuardMonitor,
    ImpedanceForceGuardTripped,
    bimanual_emergency_stop,
)


def _payload(
    *,
    force_norm_n: float = 1.0,
    torque_norm_nm: float = 0.1,
    max_force_n: float = 60.0,
    max_torque_nm: float = 10.0,
    side: str = "left",
) -> dict[str, object]:
    now_mono = time.monotonic()
    return {
        "sample": {
            "side": side,
            "force_source": "work",
            "force_norm_n": force_norm_n,
            "torque_norm_nm": torque_norm_nm,
            "max_force_n": max_force_n,
            "max_torque_nm": max_torque_nm,
            "monotonic_time_sec": now_mono,
            "wall_time_sec": time.time(),
            "step": 1,
        }
    }


def test_impedance_force_guard_arms_after_both_valid_samples() -> None:
    stops = []
    monitor = ImpedanceForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(side="left"),
            "right": lambda: _payload(side="right"),
        },
        emergency_stop=stops.append,
        config=ImpedanceForceGuardConfig(
            poll_hz=80.0,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        snapshot = monitor.start()
        assert snapshot["motion_ready"] is True
        assert set(snapshot["last_reading_by_side"]) == {"left", "right"}
        assert stops == []
    finally:
        monitor.close()


def test_impedance_force_guard_over_limit_latches_one_stop() -> None:
    stops = []
    monitor = ImpedanceForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(side="left", force_norm_n=60.1),
            "right": lambda: _payload(side="right"),
        },
        emergency_stop=stops.append,
        config=ImpedanceForceGuardConfig(startup_timeout_sec=0.5),
    )
    try:
        with pytest.raises(ImpedanceForceGuardTripped) as captured:
            monitor.start()
        assert captured.value.fault.reason == "force_limit_exceeded"
        assert captured.value.fault.side == "left"
        assert len(stops) == 1
        assert monitor.snapshot()["fault"]["reason"] == "force_limit_exceeded"
    finally:
        monitor.close()


def test_backend_hard_stop_notification_uses_guard_stop_path_once() -> None:
    stops = []
    monitor = ImpedanceForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(side="left"),
            "right": lambda: _payload(side="right"),
        },
        emergency_stop=stops.append,
        config=ImpedanceForceGuardConfig(startup_timeout_sec=0.5),
    )
    try:
        monitor.start()
        monitor.notify_backend_hard_stop(
            "right",
            "RuntimeError: right: force 60.1N > hard_stop_force_n",
        )
        assert monitor.fault is not None
        assert monitor.fault.reason == "controller_hard_stop"
        assert monitor.fault.side == "right"
        assert len(stops) == 1
    finally:
        monitor.close()


def test_bimanual_emergency_stop_requests_both_sides_concurrently() -> None:
    barrier = threading.Barrier(2)

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def emergency_stop(self) -> None:
            self.calls += 1
            barrier.wait(timeout=0.5)

    left = Backend()
    right = Backend()
    assert bimanual_emergency_stop(
        {"left": left, "right": right},
        reason="test_force_trip",
    ) == {}
    assert left.calls == 1
    assert right.calls == 1
