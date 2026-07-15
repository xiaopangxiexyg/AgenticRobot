from __future__ import annotations

import pytest

from teleop_runtime.collection.rate_gate import PeriodicRateGate


def test_rate_gate_uses_time_not_control_tick_ratio() -> None:
    gate = PeriodicRateGate(15.0)

    due = [
        gate.is_due(timestamp)
        for timestamp in (0.0, 0.020, 0.045, 0.067, 0.089, 0.1335)
    ]

    assert due == [True, False, False, True, False, True]


def test_rate_gate_skips_missed_deadlines_without_bursting() -> None:
    gate = PeriodicRateGate(10.0)

    assert gate.is_due(1.0)
    assert gate.is_due(1.35)
    assert not gate.is_due(1.35)
    assert gate.next_deadline == pytest.approx(1.4)


def test_rate_gate_rejects_invalid_clock_values() -> None:
    with pytest.raises(ValueError):
        PeriodicRateGate(0.0)
    gate = PeriodicRateGate(15.0)
    with pytest.raises(ValueError):
        gate.is_due(float("nan"))
