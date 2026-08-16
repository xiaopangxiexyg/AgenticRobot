from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from teleop_runtime.deploy import arm_safety_runtime, run_policy


class _HoldBackend:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.reasons: list[str] = []

    def hold_motion(self, reason: str) -> None:
        self.reasons.append(reason)
        if self.error is not None:
            raise self.error


class _SoftHoldBackend:
    def __init__(self, receipt: object = None, *, error: Exception | None = None) -> None:
        self.receipt = receipt
        self.error = error
        self.reasons: list[str] = []

    def request_policy_inference_soft_hold(self, reason: str):
        self.reasons.append(reason)
        if self.error is not None:
            raise self.error
        return self.receipt


class _UnsupportedSoftHoldBackend:
    pass


def test_deploy_hold_attempts_every_side_before_aggregating_errors() -> None:
    left = _HoldBackend(error=OSError("left timeout"))
    right = _HoldBackend(error=RuntimeError("right rejected"))

    with pytest.raises(
        RuntimeError,
        match=(
            "deploy arm hold failed: left: OSError: left timeout; "
            "right: RuntimeError: right rejected"
        ),
    ):
        arm_safety_runtime.hold_deploy_arm_backends(
            {"left": left, "right": right},
            reason="safety fault",
        )

    assert left.reasons == ["safety fault"]
    assert right.reasons == ["safety fault"]


def test_deploy_hold_returns_only_after_all_sides_acknowledge() -> None:
    left = _HoldBackend()
    right = _HoldBackend()

    assert (
        arm_safety_runtime.hold_deploy_arm_backends(
            {"left": left, "right": right},
            reason="operator hold",
        )
        is None
    )
    assert left.reasons == ["operator hold"]
    assert right.reasons == ["operator hold"]


def test_soft_hold_preserves_supported_and_unsupported_receipts() -> None:
    left = _SoftHoldBackend({"active": True, "status": "activated"})
    right = _UnsupportedSoftHoldBackend()

    receipts = arm_safety_runtime.request_policy_inference_soft_hold(
        {"left": left, "right": right},
        reason="inference wait",
    )

    assert receipts == {
        "left": {"active": True, "status": "activated"},
        "right": {
            "active": False,
            "status": "unsupported",
            "backend_type": "_UnsupportedSoftHoldBackend",
        },
    }
    assert left.reasons == ["inference wait"]


def test_soft_hold_preserves_existing_fail_fast_side_order() -> None:
    left = _SoftHoldBackend(error=RuntimeError("no measured state"))
    right = _SoftHoldBackend({"active": True})

    with pytest.raises(RuntimeError, match="no measured state"):
        arm_safety_runtime.request_policy_inference_soft_hold(
            {"left": left, "right": right},
            reason="inference wait",
        )

    assert left.reasons == ["inference wait"]
    assert right.reasons == []


def test_deploy_emergency_stop_forwards_reason_and_sorts_failures() -> None:
    backends = {"right": object(), "left": object()}
    with patch.object(
        arm_safety_runtime,
        "bimanual_emergency_stop",
        return_value={"right": "timeout", "left": "rejected"},
    ) as stop:
        with pytest.raises(
            RuntimeError,
            match="left=rejected; right=timeout",
        ):
            arm_safety_runtime.emergency_stop_deploy_arm_backends(
                backends,
                reason="force guard",
            )

    stop.assert_called_once_with(backends, reason="force guard")


@pytest.mark.parametrize(
    ("send_actions", "arms_enabled", "mode", "backends", "expected"),
    (
        (False, True, "position", {"left": object()}, False),
        (True, False, "position", {"left": object()}, False),
        (True, True, "impedance", {"left": object()}, False),
        (True, True, "position", {}, False),
        (True, True, "position", {"left": object()}, True),
    ),
)
def test_keyboard_interrupt_stop_preserves_position_active_gate(
    send_actions: bool,
    arms_enabled: bool,
    mode: str,
    backends: dict[str, object],
    expected: bool,
) -> None:
    emergency_stop = Mock()

    result = arm_safety_runtime.emergency_stop_position_deploy_on_keyboard_interrupt(
        SimpleNamespace(arms_enabled=arms_enabled, arm_control_mode=mode),
        send_actions=send_actions,
        arm_backends=backends,
        emergency_stop=emergency_stop,
    )

    assert result is expected
    if expected:
        emergency_stop.assert_called_once_with(
            backends,
            reason="keyboard_interrupt",
        )
    else:
        emergency_stop.assert_not_called()


def test_run_policy_keeps_safety_exports_and_keyboard_monkeypatch_seam() -> None:
    assert (
        run_policy.hold_deploy_arm_backends
        is arm_safety_runtime.hold_deploy_arm_backends
    )
    assert (
        run_policy.request_policy_inference_soft_hold
        is arm_safety_runtime.request_policy_inference_soft_hold
    )
    assert (
        run_policy.emergency_stop_deploy_arm_backends
        is arm_safety_runtime.emergency_stop_deploy_arm_backends
    )
    args = SimpleNamespace(arms_enabled=True, arm_control_mode="position")
    backends = {"left": object()}

    with patch.object(run_policy, "emergency_stop_deploy_arm_backends") as stop:
        assert run_policy.emergency_stop_position_deploy_on_keyboard_interrupt(
            args,
            send_actions=True,
            arm_backends=backends,
        )

    stop.assert_called_once_with(backends, reason="keyboard_interrupt")
