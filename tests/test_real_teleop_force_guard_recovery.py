from __future__ import annotations

from teleop_runtime.real_teleop import control_loop


class _CalibrationNoOpClutch:
    """Model the real input runtime before the first Home is admitted."""

    def __init__(self) -> None:
        self._teleop_started = False
        self._hold_requested = False
        self._hold_active = False
        self._gripper_control_enabled = True
        self._hold_request_reason = "manual"
        self.calls: list[str] = []

    def request_hold(self, *, reason: str) -> None:
        # OperatorControlRuntime deliberately ignores ordinary Hold requests
        # during Calibration.  A safety caller must detect this no-op.
        self.calls.append(reason)


class _LatchedClutch:
    def __init__(self) -> None:
        self._hold_requested = False
        self._hold_active = False
        self._gripper_control_enabled = True
        self._hold_request_reason = "manual"

    def request_hold(self, *, reason: str) -> None:
        self._hold_requested = True
        self._hold_active = True
        self._hold_request_reason = reason


def test_safety_hold_promotes_a_calibration_request_hold_noop() -> None:
    clutch = _CalibrationNoOpClutch()
    logs: list[str] = []

    control_loop._request_safety_hold_best_effort(
        clutch,
        reason="position_force_guard_collision",
        log=lambda _tag, message, **_kwargs: logs.append(message),
        now=1.0,
    )

    assert clutch.calls == ["position_force_guard_collision"]
    assert clutch._hold_requested is True
    assert clutch._hold_active is True
    assert clutch._gripper_control_enabled is False
    assert clutch._hold_request_reason == "position_force_guard_collision"
    assert any("returned without latching Hold" in message for message in logs)


def test_safety_hold_preserves_a_successful_runtime_latch() -> None:
    clutch = _LatchedClutch()
    logs: list[str] = []

    control_loop._request_safety_hold_best_effort(
        clutch,
        reason="position_force_guard_fault",
        log=lambda _tag, message, **_kwargs: logs.append(message),
        now=2.0,
    )

    assert clutch._hold_requested is True
    assert clutch._hold_active is True
    assert clutch._gripper_control_enabled is True
    assert clutch._hold_request_reason == "position_force_guard_fault"
    assert not any("forcing the local Hold latch" in message for message in logs)


def test_force_guard_fault_is_pending_during_calibration() -> None:
    """A latched fault is consumed even before teleop admission."""

    sentinel_fault = object()

    assert control_loop._position_force_guard_fault_pending(
        sentinel_fault,
        handled=False,
    ) is True
    assert control_loop._position_force_guard_fault_pending(
        sentinel_fault,
        handled=True,
    ) is False
    assert control_loop._position_force_guard_fault_pending(
        None,
        handled=False,
    ) is False
