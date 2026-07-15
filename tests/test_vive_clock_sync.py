from __future__ import annotations

import socket
import time

import pytest

from hardware.VIVE.mocap.clock_sync import (
    ClockSyncMeasurement,
    ClockSyncSampleStore,
    ClockSyncResult,
    make_probe,
    make_result,
    measurement_from_result,
    parse_reply,
)
from teleop_runtime.sources.pose.vive_clock_sync import (
    ViveClockSyncConfig,
    ViveClockSyncServer,
)


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_four_timestamp_measurement_has_explicit_offset_direction() -> None:
    # Sender clock is 1,000 ns behind collector. The one-way paths differ by
    # 10 ns, so NTP's symmetric-delay estimate is 5 ns low by construction.
    measurement = measurement_from_result(
        ClockSyncResult(
            sender_instance_id="sender-a",
            probe_id=4,
            t1_sender_monotonic_ns=100,
            t2_collector_monotonic_ns=1_110,
            t3_collector_monotonic_ns=1_115,
            t4_sender_monotonic_ns=135,
        )
    )

    assert measurement is not None
    assert measurement.collector_minus_sender_offset_ns == 995
    assert measurement.round_trip_ns == 30
    assert measurement.uncertainty_ns == 15
    assert measurement.sender_to_collector_monotonic_ns(500) == 1_495


def test_sample_store_selects_low_rtt_and_invalidates_previous_sender_instance() -> None:
    store = ClockSyncSampleStore(max_round_trip_ns=100)
    noisy = ClockSyncMeasurement(
        sender_instance_id="sender-a",
        probe_id=1,
        t1_sender_monotonic_ns=0,
        t2_collector_monotonic_ns=900,
        t3_collector_monotonic_ns=900,
        t4_sender_monotonic_ns=100,
        collector_minus_sender_offset_ns=850,
        round_trip_ns=100,
        uncertainty_ns=50,
    )
    clean = ClockSyncMeasurement(
        sender_instance_id="sender-a",
        probe_id=2,
        t1_sender_monotonic_ns=200,
        t2_collector_monotonic_ns=1_205,
        t3_collector_monotonic_ns=1_205,
        t4_sender_monotonic_ns=210,
        collector_minus_sender_offset_ns=1_000,
        round_trip_ns=10,
        uncertainty_ns=5,
    )
    replacement = ClockSyncMeasurement(
        sender_instance_id="sender-b",
        probe_id=0,
        t1_sender_monotonic_ns=0,
        t2_collector_monotonic_ns=2_002,
        t3_collector_monotonic_ns=2_002,
        t4_sender_monotonic_ns=4,
        collector_minus_sender_offset_ns=2_000,
        round_trip_ns=4,
        uncertainty_ns=2,
    )

    store.add(noisy)
    estimate = store.add(clean)
    assert estimate is not None
    assert estimate.collector_minus_sender_offset_ns == 1_000
    assert estimate.sample_count == 2

    estimate = store.add(replacement)
    assert estimate is not None
    assert estimate.sender_instance_id == "sender-b"
    assert estimate.sample_count == 1
    assert estimate.collector_minus_sender_offset_ns == 2_000


@pytest.mark.integration
def test_clock_sync_server_round_trip_accepts_separate_control_datagrams(
    require_udp_loopback: None,
) -> None:
    port = _free_udp_port()
    server = ViveClockSyncServer(
        ViveClockSyncConfig(
            bind_host="127.0.0.1",
            port=port,
            min_samples=1,
            max_round_trip_sec=1.0,
            expected_sender_host="127.0.0.1",
        )
    )
    server.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.settimeout(1.0)
    try:
        t1 = time.perf_counter_ns()
        sender.sendto(
            make_probe(
                sender_instance_id="loopback-sender",
                probe_id=7,
                t1_sender_monotonic_ns=t1,
            ),
            ("127.0.0.1", port),
        )
        reply_packet, _ = sender.recvfrom(4096)
        reply = parse_reply(reply_packet)
        assert reply is not None
        assert reply.sender_instance_id == "loopback-sender"
        assert reply.probe_id == 7

        sender.sendto(
            make_result(
                sender_instance_id="loopback-sender",
                probe_id=7,
                t1_sender_monotonic_ns=t1,
                t2_collector_monotonic_ns=reply.t2_collector_monotonic_ns,
                t3_collector_monotonic_ns=reply.t3_collector_monotonic_ns,
                t4_sender_monotonic_ns=time.perf_counter_ns(),
            ),
            ("127.0.0.1", port),
        )
        assert server.wait_until_ready(1.0, min_samples=1)
        state = server.peer_state(
            "127.0.0.1",
            now_collector_monotonic_ns=time.perf_counter_ns(),
        )
        assert state.status == "synchronized"
        assert state.estimate is not None
        assert state.estimate.sample_count == 1
    finally:
        sender.close()
        server.close()
