from __future__ import annotations

import math
from types import SimpleNamespace

from hardware.VIVE.mocap import vive_tracker_benchmark_sender as sender
from teleop_runtime.debug import vive_tracker_benchmark_receiver as receiver


def test_sender_stage_stability_requires_rate_ack_and_valid_trackers() -> None:
    stage = sender.StageMetrics(
        stage_id=1,
        target_hz=100.0,
        started_at=0.0,
        sent_frames=96,
        valid_frames=96,
        acknowledged_sequences=set(range(96)),
        rtt_ms=[1.0, 2.0, 3.0],
    )
    args = SimpleNamespace(
        min_rate_ratio=0.95,
        min_ack_ratio=0.99,
        min_valid_ratio=1.0,
        max_rtt_p95_ms=0.0,
    )

    stable, reason, actual_hz, ack_ratio, valid_ratio = sender.stage_sender_stability(
        stage, 1.0, args
    )

    assert stable
    assert reason == "ok"
    assert actual_hz == 96.0
    assert ack_ratio == 1.0
    assert valid_ratio == 1.0


def test_receiver_stage_summary_reports_loss_as_unstable() -> None:
    stage = receiver.StageState(
        run_id="test-run",
        stage_id=1,
        target_hz=100.0,
        received_sequences=set(range(95)),
        received_at_monotonic=[index / 100.0 for index in range(95)],
        valid_frames=95,
        sent_frames=100,
        stage_duration_sec=1.0,
    )
    args = SimpleNamespace(
        min_rate_ratio=0.95,
        min_delivery_ratio=0.99,
        min_valid_ratio=1.0,
        assume_clock_synced=False,
    )

    summary = receiver.finalise_stage(stage, args)

    assert summary["receiver_hz"] == 95.0
    assert summary["loss"] == 5
    assert not summary["stable"]
    assert "delivery ratio" in summary["reason"]


def test_direct_link_clock_sync_recovers_clock_offset_and_rtt() -> None:
    offset_sec, rtt_sec = sender.calculate_clock_offset(
        t1_sender_sent=10.0,
        t2_receiver_received=13.001,
        t3_receiver_sent=13.002,
        t4_sender_received=10.003,
    )

    assert offset_sec == 3.0
    assert math.isclose(rtt_sec, 0.002, rel_tol=0.0, abs_tol=1e-12)


def test_receiver_uses_sender_direct_link_clock_sync_for_one_way_latency() -> None:
    stage = receiver.StageState(
        run_id="test-run",
        stage_id=1,
        target_hz=100.0,
        received_sequences=set(range(100)),
        received_at_monotonic=[index / 100.0 for index in range(100)],
        valid_frames=100,
        sent_frames=100,
        stage_duration_sec=1.0,
        clock_sync_packets=100,
        clock_sync_rtt_ms=0.4,
        send_to_receiver_ms=[0.2, 0.3],
        sample_to_receiver_ms=[1.0, 1.1],
    )
    args = SimpleNamespace(
        min_rate_ratio=0.95,
        min_delivery_ratio=0.99,
        min_valid_ratio=1.0,
        assume_clock_synced=False,
    )

    summary = receiver.finalise_stage(stage, args)

    assert summary["latency_method"] == "direct-link clock-sync estimate"
    assert summary["clock_sync_rtt_ms"] == 0.4
    assert summary["send_to_receiver_p95_ms"] == 0.3
