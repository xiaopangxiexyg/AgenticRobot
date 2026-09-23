from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime import position_force_guard as canonical_position_force_guard
from teleop_runtime.real_teleop import control_loop
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.position_force_guard import (
    PositionForceGuardConfig,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
    PositionForceGuardTripped,
    position_force_guard_config,
)


def test_realteleop_force_guard_path_is_object_identical_compatibility() -> None:
    assert (
        PositionForceGuardMonitor
        is canonical_position_force_guard.PositionForceGuardMonitor
    )
    assert (
        PositionForceGuardConfig
        is canonical_position_force_guard.PositionForceGuardConfig
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


def test_position_force_guard_defaults_are_30n_5nm_with_400ms_stale_timeout():
    config = PositionForceGuardConfig()
    namespace_config = position_force_guard_config(SimpleNamespace())
    args = build_real_teleop_argument_parser().parse_args([])

    assert config.max_force_n == pytest.approx(30.0)
    assert config.max_torque_nm == pytest.approx(5.0)
    assert config.stale_timeout_sec == pytest.approx(0.4)
    assert namespace_config.max_force_n == pytest.approx(30.0)
    assert namespace_config.max_torque_nm == pytest.approx(5.0)
    assert namespace_config.stale_timeout_sec == pytest.approx(0.4)
    assert args.position_force_guard_max_force_n == pytest.approx(30.0)
    assert args.position_force_guard_max_torque_nm == pytest.approx(5.0)
    assert args.position_force_guard_stale_sec == pytest.approx(0.4)


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


def test_position_force_guard_preserves_backend_read_phase_timing():
    phase_timing = {
        "total_wall_ms": 12.5,
        "total_thread_cpu_ms": 1.25,
        "sdk_call_wall_ms": 4.0,
    }

    def reader():
        payload = _payload(force_x=1.0)
        payload["read_elapsed_sec"] = 0.0125
        payload["read_phase_timing"] = phase_timing
        return payload

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader, "right": reader},
        emergency_stop=lambda _fault: None,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.2,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        snapshot = monitor.start()
        for reading in snapshot["last_reading_by_side"].values():
            assert reading["read_elapsed_sec"] == pytest.approx(0.0125)
            assert reading["read_phase_timing"] == phase_timing
    finally:
        monitor.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_force_read_slow_diagnostic_requires_debug_mode(enabled: bool):
    events: list[tuple[str, dict[str, object]]] = []

    def reader():
        payload = _payload()
        payload["read_elapsed_sec"] = 0.08
        payload["read_phase_timing"] = {"hardware_read_lock_wait_wall_ms": 70.0}
        return payload

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=lambda _fault: None,
        diagnostic_enabled=lambda: enabled,
        diagnostic_sink=lambda event, payload: events.append((event, dict(payload))),
        config=PositionForceGuardConfig(stale_timeout_sec=0.2),
    )
    try:
        monitor.start()
        deadline = time.monotonic() + 0.3
        while enabled and not events and time.monotonic() < deadline:
            time.sleep(0.005)
        assert bool(events) is enabled
        if enabled:
            event, payload = events[0]
            assert event == "force_read_slow"
            assert payload["read_phase_timing"] == {
                "hardware_read_lock_wait_wall_ms": 70.0
            }
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
        readers_by_side={"left": lambda: _payload(force_x=31.0)},
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
    events: list[tuple[str, dict[str, object]]] = []
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
        diagnostic_enabled=lambda: True,
        diagnostic_sink=lambda event, payload: events.append((event, dict(payload))),
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.08,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start()
        deadline = time.monotonic() + 0.5
        while (monitor.fault is None or not events) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_read_stale"
        assert len(stops) == 1
        assert events and events[-1][0] == "force_read_stale"
        diagnostic = events[-1][1]
        assert diagnostic["last_success_age_sec"] >= 0.08
        assert diagnostic["inflight_age_sec"] is not None
        assert any(
            frame["function"] == "reader" for frame in diagnostic["reader_stack"]
        )
    finally:
        block.set()
        monitor.close()


def test_force_debug_sink_failure_does_not_block_stale_emergency_stop():
    stops = []
    block = threading.Event()
    first = True

    def reader():
        nonlocal first
        if first:
            first = False
            return _payload()
        block.wait(1.0)
        return _payload()

    def broken_sink(_event, _payload):
        raise RuntimeError("diagnostic sink unavailable")

    monitor = PositionForceGuardMonitor(
        readers_by_side={"left": reader},
        emergency_stop=stops.append,
        diagnostic_enabled=lambda: True,
        diagnostic_sink=broken_sink,
        config=PositionForceGuardConfig(stale_timeout_sec=0.08),
    )
    try:
        monitor.start()
        deadline = time.monotonic() + 0.5
        while not stops and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(stops) == 1
        assert stops[0].reason == "force_read_stale"
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


def test_non_motion_read_window_suppresses_only_stale_until_fresh_rearm():
    stops = []
    calls = 0
    block = threading.Event()

    def reader():
        nonlocal calls
        calls += 1
        if calls == 1:
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
        token = monitor.suspend_stale_watchdog_for_non_motion()
        assert monitor.motion_ready is False
        time.sleep(0.12)
        assert monitor.fault is None
        assert stops == []

        block.set()
        snapshot = monitor.resume_stale_watchdog_after_non_motion(
            token,
            timeout_sec=0.5,
        )
        assert snapshot["motion_ready"] is True
        assert snapshot["non_motion_read_window_active"] is False
        assert monitor.fault is None
    finally:
        block.set()
        monitor.close()


def test_force_limit_still_trips_inside_non_motion_read_window():
    stops = []
    over_limit = threading.Event()

    def reader():
        return _payload(force_x=31.0 if over_limit.is_set() else 0.0)

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
        monitor.suspend_stale_watchdog_for_non_motion()
        over_limit.set()
        deadline = time.monotonic() + 0.5
        while monitor.fault is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_limit_exceeded"
        assert len(stops) == 1
    finally:
        monitor.close()


def test_force_limit_still_trips_while_stale_watchdog_is_deferred():
    stops = []
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        return _payload(force_x=0.0 if calls == 1 else 31.0)

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


def test_force_guard_rearms_only_after_explicit_manual_home():
    stops = []
    monitor = PositionForceGuardMonitor(
        readers_by_side={
            "left": lambda: _payload(force_x=0.0),
            "right": lambda: _payload(force_x=0.0),
        },
        emergency_stop=stops.append,
        config=PositionForceGuardConfig(
            poll_hz=20.0,
            stale_timeout_sec=0.2,
            startup_timeout_sec=0.5,
        ),
    )
    try:
        monitor.start()
        monitor._trip(
            PositionForceGuardFault(
                reason="force_limit_exceeded",
                side="left",
                message="synthetic collision",
                monotonic_time_sec=time.monotonic(),
                wall_time_sec=time.time(),
            )
        )
        assert monitor.fault is not None
        assert monitor.motion_ready is False

        snapshot = monitor.rearm_after_manual_home(timeout_sec=0.5)
        assert monitor.fault is None
        assert snapshot["motion_ready"] is True
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
    phase_timing = payload["read_phase_timing"]
    assert set(phase_timing) == {
        "total_wall_ms",
        "total_thread_cpu_ms",
        "pacing_wait_requested_ms",
        "pacing_wait_wall_ms",
        "pacing_wait_thread_cpu_ms",
        "hardware_read_lock_wait_wall_ms",
        "hardware_read_lock_wait_thread_cpu_ms",
        "sdk_call_wall_ms",
        "sdk_call_thread_cpu_ms",
        "validation_wall_ms",
        "validation_thread_cpu_ms",
    }
    assert all(value >= 0.0 for value in phase_timing.values())
    assert phase_timing["total_wall_ms"] >= phase_timing["sdk_call_wall_ms"]


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


def test_rm65_command_stream_yields_to_waiting_force_reader():
    command_started = threading.Event()
    command_dispatched = threading.Event()

    class FakeArm:
        def Movej_CANFD(self, *_args, **_kwargs):
            command_dispatched.set()
            return 0

    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="127.0.0.1",
            side="right",
            connect_hardware=False,
        )
    )
    backend.arm = FakeArm()
    backend._mark_force_read_waiting()

    def send_command() -> None:
        command_started.set()
        backend._send_joint_target(np.zeros(6))

    command_thread = threading.Thread(target=send_command)
    command_thread.start()
    try:
        assert command_started.wait(0.5)
        assert not command_dispatched.wait(0.05)
    finally:
        backend._mark_force_read_acquired()
        command_thread.join(timeout=0.5)

    assert not command_thread.is_alive()
    assert command_dispatched.is_set()


def test_rm65_background_state_read_yields_to_inflight_force_reader():
    force_started = threading.Event()
    release_force = threading.Event()

    class FakeArm:
        state_calls = 0

        def Get_Current_Arm_State(self):
            self.state_calls += 1
            values = [0.0] * 6
            return 0, values, values, 0, 0

        def Get_Force_Data(self):
            force_started.set()
            assert release_force.wait(1.0)
            values = [0.0] * 6
            return 0, values, values, values, values

    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="127.0.0.1",
            side="right",
            connect_hardware=False,
        )
    )
    arm = FakeArm()
    backend.arm = arm
    force_thread = threading.Thread(target=backend.read_force_data)
    force_thread.start()
    try:
        assert force_started.wait(0.5)
        started = time.perf_counter()
        state = backend.refresh_state_from_hardware(
            hardware_read_retries=0,
            hardware_read_priority="background",
            hardware_read_lock_timeout_sec=0.02,
            include_derived_pose=False,
            include_timing_percentiles=False,
        )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.1
        assert state["hardware_read_success"] is False
        assert "background hardware state read admission timed out" in str(
            state["hardware_result_text"]
        )
        assert arm.state_calls == 0
    finally:
        release_force.set()
        force_thread.join(timeout=0.5)
        assert not force_thread.is_alive()


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


def test_emergency_stop_lookup_failure_isolated_to_one_side():
    class Backend:
        def __init__(self):
            self.calls = 0

        def emergency_stop(self):
            self.calls += 1

    right = Backend()
    # The left command backend attribute is intentionally absent.  The
    # bimanual stop must still reach the healthy right side and return a
    # structured error instead of escaping from the worker future.
    components = SimpleNamespace(
        active_sides=("left", "right"),
        right_command_backend=right,
    )

    errors = real_teleop._emergency_stop_real_arms(
        components,
        reason="dispatch_safety_race",
        now=1.0,
    )

    assert right.calls == 1
    assert "left" in errors
    assert "AttributeError" in errors["left"]


def test_safety_recording_stop_failure_is_contained():
    logs = []
    messages = []

    class UI:
        def append_message(self, tag, message):
            messages.append((tag, message))

    failure = control_loop._stop_recording_best_effort(
        recording_active=lambda: True,
        stop_recording=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("writer flush failed")
        ),
        now=1.0,
        end_reason="force_collision",
        summary={"force_collision_detected": True},
        log=lambda tag, message, **kwargs: logs.append((tag, message)),
        control_ui=UI(),
        ui_tag="ForceControl",
        ui_failure_message="collision save failed",
    )

    assert isinstance(failure, RuntimeError)
    assert any("writer flush failed" in message for _tag, message in logs)
    assert messages == [("ForceControl", "collision save failed")]


def test_safety_hold_request_failure_keeps_a_local_stop_latch():
    clutch_input = SimpleNamespace(
        request_hold=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("UI publish failed")
        )
    )
    logs = []

    control_loop._request_safety_hold_best_effort(
        clutch_input,
        reason="impedance_force_collision",
        log=lambda tag, message, **kwargs: logs.append((tag, message)),
        now=1.0,
    )

    assert clutch_input._hold_requested is True
    assert clutch_input._hold_active is True
    assert clutch_input._hold_request_reason == "impedance_force_collision"
    assert clutch_input._gripper_control_enabled is False
    assert any("UI publish failed" in message for _tag, message in logs)


def test_safety_dispatch_classifies_only_stop_races():
    assert control_loop._is_emergency_stop_latch_error(
        RuntimeError("left_arm emergency_stop is latched; refusing command dispatch")
    )
    assert control_loop._is_emergency_stop_latch_error(
        RuntimeError("emergency stop was not acknowledged")
    )
    assert not control_loop._is_emergency_stop_latch_error(
        RuntimeError("left arm transport timeout")
    )


def _joint_hard_limit_dispatch_error() -> RuntimeError:
    cause = ValueError(
        "right_arm pre-singularity target joint 6=2.094395rad exceeds "
        "absolute limits [-2.094395, 2.094395]rad."
    )
    error = RuntimeError(
        "bimanual_rig dispatch failed; group fail-stop requested: "
        f"right_arm: ValueError: {cause}"
    )
    error.__cause__ = cause
    return error


def test_joint_hard_limit_dispatch_classifier_is_narrow_and_structured():
    context = control_loop._joint_hard_limit_dispatch_context(
        _joint_hard_limit_dispatch_error()
    )

    assert context is not None
    assert context["effector_name"] == "right_arm"
    assert context["target_label"] == "pre-singularity target"
    assert context["joint_index"] == 6
    assert context["target_rad"] == pytest.approx(2.094395)
    assert context["lower_rad"] == pytest.approx(-2.094395)
    assert context["upper_rad"] == pytest.approx(2.094395)
    assert context["limit_side"] == "upper"

    assert (
        control_loop._joint_hard_limit_dispatch_context(
            ValueError("right_arm transport timeout")
        )
        is None
    )
    assert (
        control_loop._joint_hard_limit_dispatch_context(
            ValueError("right_arm joint_vector must be a 6D joint vector")
        )
        is None
    )


def test_joint_hard_limit_dispatch_stops_episode_and_retains_hold():
    stops = []
    recording_stops = []
    countdown_cancels = []
    hold_reasons = []
    logs = []
    messages = []
    components = SimpleNamespace(name="components")
    clutch_input = SimpleNamespace(
        request_hold=lambda *, reason: hold_reasons.append(reason)
    )
    control_ui = SimpleNamespace(
        append_message=lambda tag, message: messages.append((tag, message))
    )

    consumed = control_loop._consume_joint_hard_limit_dispatch(
        _joint_hard_limit_dispatch_error(),
        components=components,
        clutch_input=clutch_input,
        recording_active=lambda: True,
        stop_recording=lambda now, **kwargs: recording_stops.append((now, kwargs)),
        cancel_recording_countdown=lambda now, **kwargs: countdown_cancels.append(
            (now, kwargs)
        ),
        emergency_stop_real_arms=lambda received, **kwargs: stops.append(
            (received, kwargs)
        )
        or {},
        now=12.5,
        step_index=77,
        last_state="RUNNING",
        log=lambda tag, message, **kwargs: logs.append((tag, message, kwargs)),
        control_ui=control_ui,
    )

    assert consumed is True
    assert stops == [
        (
            components,
            {
                "reason": "joint_hard_limit_dispatch:right_arm:joint_6",
                "now": 12.5,
                "raise_on_failure": False,
            },
        )
    ]
    assert countdown_cancels == [(12.5, {"reason": "joint_hard_limit"})]
    assert len(recording_stops) == 1
    stop_now, stop_kwargs = recording_stops[0]
    assert stop_now == 12.5
    assert stop_kwargs["end_reason"] == "joint_hard_limit"
    summary = stop_kwargs["summary"]
    assert summary["joint_hard_limit_detected"] is True
    assert summary["joint_hard_limit_event"]["joint_index"] == 6
    assert "force_collision_detected" not in summary
    assert hold_reasons == ["joint_hard_limit"]
    assert messages and messages[-1][0] == "JointHardLimit"
    assert any("GUI/devices retained" in message for _tag, message, _kwargs in logs)


def test_joint_hard_limit_dispatch_stop_failure_still_closes_episode_and_holds():
    recording_stops = []
    hold_reasons = []
    messages = []

    def fail_arm_stop(*_args, **_kwargs):
        raise RuntimeError("Arm_Stop transport failure")

    consumed = control_loop._consume_joint_hard_limit_dispatch(
        _joint_hard_limit_dispatch_error(),
        components=SimpleNamespace(),
        clutch_input=SimpleNamespace(
            request_hold=lambda *, reason: hold_reasons.append(reason)
        ),
        recording_active=lambda: True,
        stop_recording=lambda now, **kwargs: recording_stops.append((now, kwargs)),
        cancel_recording_countdown=lambda *_args, **_kwargs: None,
        emergency_stop_real_arms=fail_arm_stop,
        now=20.0,
        step_index=88,
        last_state="RUNNING",
        log=lambda *_args, **_kwargs: None,
        control_ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message))
        ),
    )

    assert consumed is True
    assert recording_stops[0][1]["summary"]["emergency_stop_errors"] == {
        "__all__": "RuntimeError: Arm_Stop transport failure"
    }
    assert hold_reasons == ["joint_hard_limit"]
    assert "Arm_Stop 未确认" in messages[-1][1]


def test_impedance_issue_provider_force_error_enters_collision_callback():
    class Session:
        def __init__(self):
            self.state_machine = SimpleNamespace(
                state=real_teleop.TeleopRunState.RUNNING
            )

    session = Session()
    components = SimpleNamespace(session=session)
    collisions = []

    with patch.object(
        real_teleop,
        "_impedance_force_control_issue",
        side_effect=RuntimeError("left: force 55.0000N > hard_stop_force_n"),
    ):
        ok = real_teleop._guard_impedance_force_control(
            SimpleNamespace(),
            components,
            context="main_loop",
            now=1.0,
            collision_handler=collisions.append,
        )

    assert ok is False
    assert len(collisions) == 1
    assert "hard_stop_force_n" in collisions[0]

    with patch.object(
        real_teleop,
        "_impedance_force_control_issue",
        side_effect=RuntimeError("left arm transport timeout"),
    ):
        with pytest.raises(RuntimeError, match="transport timeout"):
            real_teleop._guard_impedance_force_control(
                SimpleNamespace(),
                components,
                context="main_loop",
                now=1.0,
                collision_handler=collisions.append,
            )
