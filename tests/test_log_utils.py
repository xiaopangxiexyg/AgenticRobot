from __future__ import annotations

from teleop_runtime.log_utils import ChangeGate


def test_change_gate_coalesces_frequent_changes_until_minimum_interval() -> None:
    gate = ChangeGate()

    assert gate.should_emit("status", "a", now=10.0, min_interval_sec=1.0)
    assert not gate.should_emit("status", "b", now=10.1, min_interval_sec=1.0)
    assert not gate.should_emit("status", "c", now=10.9, min_interval_sec=1.0)
    assert gate.should_emit("status", "c", now=11.0, min_interval_sec=1.0)
    assert not gate.should_emit("status", "c", now=11.1, min_interval_sec=1.0)


def test_change_gate_zero_minimum_preserves_immediate_change_output() -> None:
    gate = ChangeGate()

    assert gate.should_emit("status", "a", now=10.0, min_interval_sec=1.0)
    assert gate.should_emit("status", "b", now=10.1, min_interval_sec=0.0)


def test_change_gate_pending_change_can_return_to_unthrottled_mode() -> None:
    gate = ChangeGate()

    assert gate.should_emit("status", "a", now=10.0, min_interval_sec=1.0)
    assert not gate.should_emit("status", "b", now=10.1, min_interval_sec=1.0)
    assert gate.should_emit("status", "b", now=10.2)


def test_change_gate_heartbeat_can_flush_a_pending_change() -> None:
    gate = ChangeGate()

    assert gate.should_emit("status", "a", now=10.0, min_interval_sec=5.0)
    assert not gate.should_emit(
        "status",
        "b",
        now=10.2,
        min_interval_sec=5.0,
        heartbeat_sec=1.0,
    )
    assert gate.should_emit(
        "status",
        "b",
        now=11.0,
        min_interval_sec=5.0,
        heartbeat_sec=1.0,
    )
