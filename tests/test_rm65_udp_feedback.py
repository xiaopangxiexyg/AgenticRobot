"""Offline wire replay and localhost UDP only; no SDK/controller connection."""
from __future__ import annotations

import copy
import json
import socket
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from hardware.rm65.realtime_feedback import (
    RM65UDPFeedbackReceiver,
    decode_rm65_feedback,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig


def packet(coordinate=2):
    # Representative fields from the retained 20261014 left-arm arrival probe.
    return {"state": "realtime_arm_joint_state", "err": [0],
        "joint_status": {"joint_position": [81760, 51403, 118009, -9398, -82533, 3485],
            "joint_err_code": [0] * 6, "joint_en_flag": [1] * 6},
        "waypoint": {"position": [-28475, -536935, 209432], "euler": [533, 1510, -1016]},
        "six_force_sensor": {"coordinate": coordinate, "force": [2381, -3267, -3374, -26, 494, 68],
            "zero_force": [-1483, -74, -1828, 13, -20, 8]}}


def decode(value=None, *, side="left", ip="127.0.0.1", sequence=1):
    return decode_rm65_feedback(json.dumps(packet() if value is None else value).encode(),
        side=side, controller_ip=ip, sequence=sequence,
        wall_time_sec=time.time(), monotonic_time_sec=time.monotonic())


def test_wire_units_are_explicit_not_guessed_by_magnitude():
    sample = decode()
    assert sample.joint_deg == pytest.approx([81.76, 51.403, 118.009, -9.398, -82.533, 3.485])
    assert sample.pose_6d == pytest.approx([-.028475, -.536935, .209432, .533, 1.51, -1.016])
    assert sample.force_sources()["tool"] == pytest.approx([-1.483, -.074, -1.828, .013, -.020, .008])
    value = packet()
    value["joint_status"]["joint_position"] = [1, 2, 3, 4, 5, 6]
    assert decode(value).joint_deg == pytest.approx([.001, .002, .003, .004, .005, .006])


@pytest.mark.parametrize("coordinate, source", [(0, "zero"), (1, "work"), (2, "tool")])
def test_force_coordinates_are_never_silently_relabelled(coordinate, source):
    sources = decode(packet(coordinate)).force_sources()
    assert set(sources) == {"raw", source}


@pytest.mark.parametrize("path, value", [
    (("state",), "another_message"), (("six_force_sensor", "coordinate"), 2.0),
    (("six_force_sensor", "coordinate"), True), (("six_force_sensor", "coordinate"), 3),
    (("joint_status", "joint_position"), [0] * 7),
    (("joint_status", "joint_position"), [float("nan")] * 6),
    (("joint_status", "joint_position"), [1.2] * 6),
    (("joint_status", "joint_en_flag"), [2] * 6),
    (("six_force_sensor", "zero_force"), None), (("err",), []),
])
def test_malformed_known_feedback_is_rejected(path, value):
    payload = copy.deepcopy(packet())
    owner = payload
    for name in path[:-1]:
        owner = owner[name]
    owner[path[-1]] = value
    with pytest.raises(ValueError):
        decode(payload)


@pytest.fixture
def receiver():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    server = RM65UDPFeedbackReceiver(arm_ip_by_side={"left": "127.0.0.1", "right": "127.0.0.2"},
        port=port, bind_ip="127.0.0.1")
    yield server
    server.close()


def send(server, *, side="left", value=None):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.bind(({"left": "127.0.0.1", "right": "127.0.0.2", "unknown": "127.0.0.3"}[side], 0))
        sender.sendto(json.dumps(packet() if value is None else value).encode(), ("127.0.0.1", server.port))


def await_sample(server, side="left"):
    deadline = time.monotonic() + 1.
    while True:
        try:
            return server.snapshot(side)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.001)


def test_one_receiver_demultiplexes_both_arms_and_keeps_packet_identity(receiver):
    receiver.start()
    send(receiver, side="right")
    send(receiver, side="left")
    send(receiver, side="unknown")
    receiver.wait_ready(timeout_sec=1., max_age_sec=.2)
    assert receiver.snapshot("left").controller_ip == "127.0.0.1"
    assert receiver.snapshot("right").controller_ip == "127.0.0.2"
    first = receiver.snapshot("left")
    assert receiver.snapshot("left") is first
    deadline = time.monotonic() + 1.
    while not receiver.diagnostics()["unknown_source_count"] and time.monotonic() < deadline:
        time.sleep(.001)
    receiver.close()
    assert not receiver.diagnostics()["thread_alive"]
    assert receiver.diagnostics()["unknown_source_count"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        receiver.snapshot("left")


def test_port_conflict_never_reuses_or_interrupts_the_existing_owner(receiver):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as existing:
        existing.bind(("127.0.0.1", receiver.port))
        with pytest.raises(OSError):
            receiver.start()
        assert existing.getsockname()[1] == receiver.port


def test_queued_packets_retain_kernel_age_instead_of_parse_time(receiver, monkeypatch):
    release = threading.Event()
    run = receiver._run
    monkeypatch.setattr(receiver, "_run", lambda: (release.wait(1.), run()))
    receiver.start()
    send(receiver)
    send(receiver, side="right")
    time.sleep(.08)
    release.set()
    sample = await_sample(receiver)
    assert time.monotonic() - sample.monotonic_time_sec >= .07
    with pytest.raises(RuntimeError, match="stale"):
        receiver.wait_ready(timeout_sec=.02, max_age_sec=.04)


def test_sensor_coordinate_stream_cannot_enable_tool_force_protection(receiver):
    receiver.start()
    send(receiver, value=packet(0))
    send(receiver, side="right", value=packet(0))
    await_sample(receiver)
    with pytest.raises(RuntimeError, match="force_coordinate=2"):
        receiver.wait_ready(timeout_sec=.02, max_age_sec=.2)


@pytest.mark.parametrize("field, values", [("joint_err_code", [1, 0, 0, 0, 0, 0]),
                                           ("joint_en_flag", [0, 1, 1, 1, 1, 1])])
def test_controller_faults_and_disabled_joints_are_not_healthy(field, values):
    payload = packet()
    payload["joint_status"][field] = values
    with pytest.raises(RuntimeError, match="error or disabled"):
        decode(payload).require_healthy()


def udp_backend(sample_holder):
    backend = RM65ArmBackend(RM65ArmBackendConfig(arm_ip="127.0.0.1", side="left",
        connect_hardware=True, force_position_control=False, background_streaming_enabled=False,
        configure_singularity_thresholds=False, udp_feedback_reader=lambda: sample_holder[0]))
    class NoSDKRead:
        def __getattr__(self, name):
            raise AssertionError(f"UDP must not query native SDK: {name}")
    backend.arm = NoSDKRead()
    return backend


def test_udp_backend_updates_only_on_a_new_packet_without_sdk_or_age_renewal():
    holder = [decode()]
    backend = udp_backend(holder)
    options = {"include_derived_pose": False, "include_timing_percentiles": False,
               "include_pose_source_attestation": False, "include_background_streaming": False}
    first = backend.refresh_state_from_hardware(**options)
    again = backend.refresh_state_from_hardware(**options)
    assert again["measured_sequence"] == first["measured_sequence"] == 1
    assert again["measured_timestamp"] == first["measured_timestamp"] == holder[0].wall_time_sec
    np.testing.assert_allclose(again["hardware_qpos"], np.deg2rad(holder[0].joint_deg))
    force = backend.read_force_data()
    assert force["monotonic_time_sec"] == holder[0].monotonic_time_sec
    assert force["sources"]["tool"] == holder[0].external_force
    holder[0] = replace(holder[0], sequence=2, wall_time_sec=time.time(),
                        monotonic_time_sec=time.monotonic())
    assert backend.refresh_state_from_hardware(**options)["measured_sequence"] == 2


def test_concurrent_force_snapshot_is_not_a_false_sequence_regression():
    older = decode()
    holder = [replace(older, sequence=2)]
    backend = udp_backend(holder)
    backend.refresh_state_from_hardware(include_derived_pose=False)
    # Simulate a force reader's cut taken before the state writer published 2.
    holder[0] = older
    assert backend.read_force_data()["feedback_sequence"] == 1
    state = backend.refresh_state_from_hardware(include_derived_pose=False)
    assert state["measured_sequence"] == 1
    assert backend._udp_feedback_sequence == 2


@pytest.mark.parametrize("fields", [{"side": "right"}, {"controller_ip": "127.0.0.2"}])
def test_backend_rejects_feedback_for_the_wrong_controller(fields):
    backend = udp_backend([replace(decode(), **fields)])
    with pytest.raises(RuntimeError, match="identity"):
        backend.refresh_state_from_hardware(include_derived_pose=False)
    with pytest.raises(RuntimeError, match="identity"):
        backend.read_force_data()


def test_udp_reads_do_not_wait_for_the_native_command_lock():
    backend = udp_backend([decode()])
    result = []
    def read():
        result.append(backend.refresh_state_from_hardware(include_derived_pose=False))
        result.append(backend.read_force_data())
    with backend._hardware_read_lock:
        worker = threading.Thread(target=read)
        worker.start()
        worker.join(timeout=.5)
        complete_without_sdk_lock = not worker.is_alive()
    worker.join(timeout=1.)
    assert complete_without_sdk_lock
    assert len(result) == 2


def test_udp_loss_still_trips_the_existing_force_stale_guard(receiver):
    from teleop_runtime.position_force_guard import (
        PositionForceGuardConfig,
        PositionForceGuardMonitor,
    )

    receiver.start()
    stop = threading.Event()
    def publish():
        while not stop.is_set():
            send(receiver)
            send(receiver, side="right")
            stop.wait(.005)
    producer = threading.Thread(target=publish)
    producer.start()
    faults = []
    def reading(side):
        sample = receiver.snapshot(side)
        return {"sources": sample.force_sources(), "monotonic_time_sec": sample.monotonic_time_sec,
                "wall_time_sec": sample.wall_time_sec}
    monitor = PositionForceGuardMonitor(readers_by_side={side: lambda side=side: reading(side)
        for side in ("left", "right")}, emergency_stop=faults.append,
        config=PositionForceGuardConfig(stale_timeout_sec=.12, startup_timeout_sec=1.))
    try:
        receiver.wait_ready(timeout_sec=1., max_age_sec=.2)
        monitor.start()
        assert monitor.motion_ready
        stop.set()
        producer.join(timeout=1.)
        deadline = time.monotonic() + 1.
        while monitor.fault is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert monitor.fault is not None
        assert monitor.fault.reason == "force_read_stale"
        assert len(faults) == 1
    finally:
        stop.set()
        producer.join(timeout=1.)
        monitor.close()


def test_external_receiver_uses_sdk_dual_mode_not_its_udp_callback(monkeypatch):
    from hardware.rm65 import rm_api2_adapter as adapter

    modes = []
    class Robot:
        arm_dof = 6
        def __init__(self, mode):
            modes.append(mode)
        def rm_create_robot_arm(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(id=1)
    monkeypatch.setattr(adapter, "RoboticArm", Robot)
    adapter.Arm(adapter.RM65, "127.0.0.1", external_udp_receiver=True)
    adapter.Arm(adapter.RM65, "127.0.0.1")
    assert modes == [adapter.rm_thread_mode_e.RM_DUAL_MODE_E, adapter.rm_thread_mode_e.RM_TRIPLE_MODE_E]


@pytest.mark.parametrize("pause_after", ["wall", "monotonic"])
def test_clock_pair_delay_never_advances_kernel_arrival(receiver, monkeypatch, pause_after):
    from types import SimpleNamespace
    from hardware.rm65 import realtime_feedback as feedback

    # The fixed wall/monotonic anchor includes ample room for real localhost
    # delivery. Advance only the controlled clock between the two reads.
    wall_anchor = time.time() + 60.0
    monotonic_anchor = time.monotonic() + 60.0
    elapsed = [0.0]

    def wall_clock():
        value = wall_anchor + elapsed[0]
        if pause_after == "wall":
            elapsed[0] += 0.5
        return value

    def monotonic_clock():
        value = monotonic_anchor + elapsed[0]
        if pause_after == "monotonic":
            elapsed[0] += 0.5
        return value

    monkeypatch.setattr(feedback, "time", SimpleNamespace(
        time=wall_clock, monotonic=monotonic_clock,
    ))
    receiver.start()
    send(receiver)
    sample = await_sample(receiver)
    latest_safe_arrival = monotonic_anchor - (wall_anchor - sample.wall_time_sec)
    assert sample.monotonic_time_sec <= latest_safe_arrival + 1e-8
