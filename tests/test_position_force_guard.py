from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.real_teleop.position_force_guard import (
    PositionForceGuardConfig,
    PositionForceGuardMonitor,
    PositionForceGuardTripped,
)


def _payload(*, force_x: float = 0.0, torque_x: float = 0.0):
    values = [force_x, 0.0, 0.0, torque_x, 0.0, 0.0]
    return {
        "sources": {
            "raw": values,
            "zero": values,
            "work": values,
            "tool": values,
        },
        "monotonic_time_sec": time.monotonic(),
        "wall_time_sec": time.time(),
    }


def test_position_force_guard_rejects_sdk_oversampling():
    with pytest.raises(ValueError, match="<= 20 Hz"):
        PositionForceGuardConfig(poll_hz=20.1)


def test_position_force_guard_rejects_fractional_sample_counts():
    with pytest.raises(ValueError, match="positive integer"):
        PositionForceGuardConfig(consecutive_limit_samples=1.5)


def test_position_force_guard_arms_after_both_valid_samples():
    stops = []
    monitor = PositionForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(force_x=1.0),
            "right": lambda: _payload(torque_x=0.1),
        },
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.2,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        snapshot = monitor.start()
        assert snapshot["ready"] is True
        assert snapshot["stale_watchdog_armed"] is True
        assert snapshot["motion_ready"] is True
        assert set(snapshot["last_reading_by_side"]) == {"left", "right"}
        assert stops == []
    finally:
        monitor.close()


def test_position_force_guard_over_limit_latches_one_bimanual_stop():
    stops = []
    monitor = PositionForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(force_x=16.0),
            "right": lambda: _payload(),
        },
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            max_force_n=15.0,
            poll_hz=20.0,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        with pytest.raises(PositionForceGuardTripped) as captured:
            monitor.start()
        assert captured.value.fault.reason == "force_limit_exceeded"
        assert captured.value.fault.side == "left"
        assert len(stops) == 1
        assert stops[0] == captured.value.fault
        summary = real_teleop._position_force_guard_collision_summary(
            captured.value.fault
        )
        assert summary["force_collision_detected"] is True
        assert summary["force_collision_label"] == "detected_collision"
        assert summary["force_collision_event"]["source"] == "position_force_guard"
    finally:
        monitor.close()


def test_force_guard_trip_context_labels_episode_before_quit_closes_it():
    fault = real_teleop.PositionForceGuardFault(
        reason="force_limit_exceeded",
        side="left",
        message="left force exceeded",
        monotonic_time_sec=10.0,
        wall_time_sec=20.0,
    )

    end_reason, summary = real_teleop._force_guard_recording_stop_context(
        fault,
        fallback_end_reason="quit_requested",
        fallback_summary={"last_step_index": 12},
    )

    assert end_reason == "position_force_guard"
    assert summary["reason"] == "position_force_guard"
    assert summary["last_step_index"] == 12
    assert summary["position_force_guard_fault"]["reason"] == "force_limit_exceeded"
    assert summary["force_collision_detected"] is True
    assert summary["force_collision_label"] == "detected_collision"


def test_force_guard_stop_context_preserves_normal_quit_without_fault():
    end_reason, summary = real_teleop._force_guard_recording_stop_context(
        None,
        fallback_end_reason="quit_requested",
        fallback_summary={"last_step_index": 3},
    )

    assert end_reason == "quit_requested"
    assert summary == {"last_step_index": 3}


def test_position_force_guard_read_errors_fail_closed():
    stops = []

    def fail():
        raise OSError("sensor unavailable")

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": fail},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            max_consecutive_read_errors=2,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        with pytest.raises(PositionForceGuardTripped) as captured:
            monitor.start()
        assert captured.value.fault.reason == "force_read_failed"
        assert captured.value.fault.consecutive_count == 2
        assert len(stops) == 1
        assert (
            real_teleop._position_force_guard_collision_summary(
                captured.value.fault
            )
            == {}
        )
    finally:
        monitor.close()


def test_impedance_terminal_force_limit_exception_becomes_collision_summary():
    summary = real_teleop._terminal_force_collision_summary(
        RuntimeError("left: force 55.0000N > hard_stop_force_n"),
        control_mode="impedance",
    )

    assert summary["force_collision_detected"] is True
    assert summary["force_collision_label"] == "detected_collision"
    assert summary["force_collision_event"]["source"] == "controller_hard_stop"
    assert summary["force_collision_event"]["control_mode"] == "impedance"


def test_non_collision_terminal_exception_does_not_mark_collision():
    assert (
        real_teleop._terminal_force_collision_summary(
            RuntimeError("left arm transport timeout"),
            control_mode="impedance",
        )
        == {}
    )


def test_position_force_guard_latches_fault_when_stop_callback_fails():
    def fail_stop(_fault):
        raise OSError("Arm_Stop timeout")

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": lambda: _payload(force_x=16.0)},
        emergency_stop=fail_stop,
        config=PositionForceGuardConfig(startup_timeout_sec=0.5),
    )
    try:
        with pytest.raises(PositionForceGuardTripped):
            monitor.start()
        snapshot = monitor.snapshot()
        assert snapshot["fault"]["reason"] == "force_limit_exceeded"
        assert snapshot["emergency_stop_error"] == "OSError: Arm_Stop timeout"
    finally:
        monitor.close()


def test_position_force_guard_watchdog_stops_on_blocked_reader():
    stops = []
    first = True
    block = threading.Event()

    def reader():
        nonlocal first
        if first:
            first = False
            return _payload()
        block.wait(1.0)
        return _payload()

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start()
        deadline = time.monotonic() + 0.5
        while monitor.fault is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_read_stale"
        assert len(stops) == 1
    finally:
        block.set()
        monitor.close()


def test_deferred_stale_watchdog_ignores_blocking_camera_startup_window():
    stops = []
    first = True
    block = threading.Event()

    def reader():
        nonlocal first
        if first:
            first = False
            return _payload()
        block.wait(1.0)
        return _payload()

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        snapshot = monitor.start(arm_stale_watchdog=False)
        assert snapshot["ready"] is True
        assert snapshot["stale_watchdog_armed"] is False
        assert snapshot["motion_ready"] is False
        time.sleep(0.12)
        assert monitor.fault is None
        assert stops == []

        block.set()
        armed = monitor.arm_stale_watchdog(timeout_sec=0.5)
        assert armed["stale_watchdog_armed"] is True
        assert armed["motion_ready"] is True
        assert monitor.fault is None
    finally:
        block.set()
        monitor.close()


def test_deferred_stale_watchdog_arm_fails_closed_without_fresh_sample():
    stops = []
    first = True
    block = threading.Event()

    def reader():
        nonlocal first
        if first:
            first = False
            return _payload()
        block.wait(1.0)
        return _payload()

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start(arm_stale_watchdog=False)
        with pytest.raises(PositionForceGuardTripped) as captured:
            monitor.arm_stale_watchdog(timeout_sec=0.1)
        assert captured.value.fault.reason == "stale_watchdog_arm_timeout"
        assert len(stops) == 1
    finally:
        block.set()
        monitor.close()


def test_force_limit_still_trips_while_stale_watchdog_is_deferred():
    stops = []
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        return _payload(force_x=0.0 if calls == 1 else 16.0)

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start(arm_stale_watchdog=False)
        deadline = time.monotonic() + 0.5
        while monitor.fault is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_limit_exceeded"
        assert len(stops) == 1
    finally:
        monitor.close()


def test_read_errors_still_trip_while_stale_watchdog_is_deferred():
    stops = []
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _payload()
        raise OSError("force SDK unavailable")

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            max_consecutive_read_errors=2,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start(arm_stale_watchdog=False)
        deadline = time.monotonic() + 0.5
        while monitor.fault is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_read_failed"
        assert monitor.fault.consecutive_count == 2
        assert len(stops) == 1
    finally:
        monitor.close()


def test_rm65_backend_force_read_validates_and_serializes_payload():
    class FakeArm:
        def Get_Force_Data(self):
            return (
                0,
                [1, 2, 3, 4, 5, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [7, 8, 9, 1, 2, 3],
            )

    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="127.0.0.1",
            side="left",
            connect_hardware=False,
        )
    )
    backend.arm = FakeArm()
    payload = backend.read_force_data()
    assert payload["result_code"] == 0
    assert payload["sources"]["tool"].tolist() == [7, 8, 9, 1, 2, 3]


def test_rm65_backend_serializes_concurrent_force_reads():
    class FakeArm:
        def __init__(self):
            self.active_calls = 0
            self.max_active_calls = 0
            self.call_times = []
            self.lock = threading.Lock()

        def Get_Force_Data(self):
            with self.lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
                self.call_times.append(time.monotonic())
            time.sleep(0.02)
            with self.lock:
                self.active_calls -= 1
            values = [0, 0, 0, 0, 0, 0]
            return 0, values, values, values, values

    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="127.0.0.1",
            side="left",
            connect_hardware=False,
        )
    )
    arm = FakeArm()
    backend.arm = arm
    threads = [
        threading.Thread(target=backend.read_force_data)
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=0.5)
        assert not thread.is_alive()
    assert arm.max_active_calls == 1
    assert arm.call_times[1] - arm.call_times[0] >= 0.045


def test_bimanual_emergency_stop_requests_both_arms_in_parallel():
    barrier = threading.Barrier(2)

    class Backend:
        def __init__(self):
            self.calls = 0

        def emergency_stop(self):
            self.calls += 1
            barrier.wait(timeout=0.5)

    left = Backend()
    right = Backend()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left,
        right_command_backend=right,
    )
    errors = real_teleop._emergency_stop_real_arms(
        components,
        reason="test_force_trip",
        now=1.0,
        raise_on_failure=True,
    )
    assert errors == {}
    assert left.calls == 1
    assert right.calls == 1


def test_bimanual_emergency_stop_reports_unacknowledged_side():
    class Backend:
        def __init__(self, *, fail=False):
            self.fail = fail

        def emergency_stop(self):
            if self.fail:
                raise OSError("controller timeout")

    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=Backend(fail=True),
        right_command_backend=Backend(),
    )
    with pytest.raises(RuntimeError, match="left=OSError: controller timeout"):
        real_teleop._emergency_stop_real_arms(
            components,
            reason="test_force_trip",
            now=1.0,
            raise_on_failure=True,
        )
