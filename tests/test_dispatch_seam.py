from __future__ import annotations

import pytest

from robot_motion.runtime.dispatch_owner import (
    AdmittedSafetyControlEvidence,
    DispatchOwnerRegistry,
    LifecycleControlAuthorization,
)
from robot_motion.runtime.dispatch_seam import (
    DispatchPermitInvocation,
    DispatchSeamGate,
    DispatchSeamRejectedError,
    DispatchSeamSpec,
    PermittedDispatchAdapter,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def registry() -> DispatchOwnerRegistry:
    return DispatchOwnerRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        initial_authority_epoch=2,
        initial_execution_epoch=4,
        clock=ManualClock(),
    )


def motion_setup():
    value = registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    permit = value.issue_motion_permit(
        token=token,
        authority_epoch=2,
        execution_epoch=4,
        command_lineage_id="legacy-command-1",
        directive_lineage_id="legacy-directive-1",
    )
    gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-arm.pre-sdk",
            permit_kind="motion",
            call_stage="pre_sdk_call",
        ),
    )
    return value, permit, gate


def test_accepted_permit_calls_backend_once_and_preserves_result() -> None:
    _value, permit, gate = motion_setup()
    calls: list[tuple[float, ...]] = []

    outcome = gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: calls.append((1.0, 2.0)) or {"accepted": True},
    )

    assert calls == [(1.0, 2.0)]
    assert outcome.result == {"accepted": True}
    assert outcome.validation.accepted
    assert outcome.validation.status == "accepted"
    assert outcome.validation.owner_validation_receipt_id is not None


def test_epoch_rejection_is_typed_and_never_calls_backend() -> None:
    _value, permit, gate = motion_setup()
    calls: list[str] = []

    with pytest.raises(DispatchSeamRejectedError) as captured:
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=5,
            backend_call=lambda: calls.append("called"),
        )

    assert calls == []
    assert captured.value.receipt.status == "execution_epoch_mismatch"
    assert not captured.value.receipt.accepted


def test_permit_kind_mismatch_does_not_consume_permit_or_call_backend() -> None:
    value, permit, motion_gate = motion_setup()
    safety_gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-arm.stop",
            permit_kind="safety_control",
            operation="hold",
            call_stage="pre_sdk_call",
        ),
    )
    calls: list[str] = []

    with pytest.raises(DispatchSeamRejectedError) as captured:
        safety_gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda: calls.append("called"),
        )
    assert captured.value.receipt.status == "permit_kind_mismatch"
    assert calls == []

    # Local seam mismatch did not consume a valid motion permit.
    accepted = motion_gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: "motion-sent",
    )
    assert accepted.result == "motion-sent"


def test_safety_operation_mismatch_fails_before_registry_consumption() -> None:
    value = registry()
    evidence = AdmittedSafetyControlEvidence(
        event_id="safety-event-1",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        execution_epoch=4,
        evidence_kind="safety_event",
        safety_source="shared_safety_guard",
        control_admission_sequence=1,
    )
    permit = value.issue_safety_control_permit(
        operation="stop",
        evidence=evidence,
        safety_source="shared_safety_guard",
    )
    hold_gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-arm.hold",
            permit_kind="safety_control",
            operation="hold",
            call_stage="pre_sdk_call",
        ),
    )
    stop_gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-arm.stop",
            permit_kind="safety_control",
            operation="stop",
            call_stage="pre_sdk_call",
        ),
    )

    with pytest.raises(DispatchSeamRejectedError) as captured:
        hold_gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda: None,
        )
    assert captured.value.receipt.status == "operation_mismatch"

    assert stop_gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: "stopped",
    ).result == "stopped"


def test_one_shot_permit_blocks_a_second_backend_call() -> None:
    _value, permit, gate = motion_setup()
    calls: list[str] = []
    gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: calls.append("first"),
    )

    with pytest.raises(DispatchSeamRejectedError) as captured:
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda: calls.append("second"),
        )

    assert calls == ["first"]
    assert captured.value.receipt.status == "permit_already_consumed"


def test_backend_exception_is_not_rewritten() -> None:
    _value, permit, gate = motion_setup()
    error = LookupError("backend failed")

    def fail() -> None:
        raise error

    with pytest.raises(LookupError) as captured:
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=fail,
        )
    assert captured.value is error
    with pytest.raises(DispatchSeamRejectedError) as replayed:
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda: None,
        )
    assert replayed.value.receipt.status == "permit_already_consumed"


def test_each_async_worker_tick_requires_a_distinct_one_shot_permit() -> None:
    value = registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-rm65-background.pre-sdk",
            permit_kind="motion",
            call_stage="pre_sdk_call",
        ),
    )
    sent: list[tuple[str, float]] = []

    permits = tuple(
        value.issue_motion_permit(
            token=token,
            authority_epoch=2,
            execution_epoch=4,
            command_lineage_id=f"reference-frame-{index}",
            directive_lineage_id="selected-segment-1",
        )
        for index in range(3)
    )
    for index, permit in enumerate(permits):
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda index=index: sent.append(
                (f"reference-frame-{index}", float(index))
            ),
        )

    assert sent == [
        ("reference-frame-0", 0.0),
        ("reference-frame-1", 1.0),
        ("reference-frame-2", 2.0),
    ]


def test_legacy_callback_adapter_preserves_values_count_order_and_return_type() -> None:
    value = registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-direct-arm.pre-sdk",
            permit_kind="motion",
            call_stage="pre_sdk_call",
        ),
    )
    issued: list[tuple[float, ...]] = []
    backend_values: list[tuple[float, ...]] = []

    def invocation_provider(command: tuple[float, ...]) -> DispatchPermitInvocation:
        issued.append(command)
        snapshot = value.snapshot()
        return DispatchPermitInvocation(
            permit=value.issue_motion_permit(
                token=token,
                authority_epoch=snapshot.current_authority_epoch,
                execution_epoch=snapshot.current_execution_epoch,
                command_lineage_id=f"legacy-command-{len(issued)}",
                directive_lineage_id=f"legacy-directive-{len(issued)}",
            ),
            current_authority_epoch=snapshot.current_authority_epoch,
            current_execution_epoch=snapshot.current_execution_epoch,
        )

    def backend_call(command: tuple[float, ...]) -> dict[str, object]:
        backend_values.append(command)
        return {"accepted": True, "command": command}

    adapter = PermittedDispatchAdapter(
        gate=gate,
        invocation_provider=invocation_provider,
        backend_call=backend_call,
    )
    commands = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    results = tuple(adapter(command) for command in commands)

    assert issued == list(commands)
    assert backend_values == list(commands)
    assert results == tuple(
        {"accepted": True, "command": command}
        for command in commands
    )


def test_adapter_rechecks_live_epoch_after_permit_provider_snapshot() -> None:
    value = registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-worker.pre-sdk",
            permit_kind="motion",
            call_stage="pre_sdk_call",
        ),
    )
    permit = value.issue_motion_permit(
        token=token,
        authority_epoch=2,
        execution_epoch=4,
        command_lineage_id="old-reference-frame",
        directive_lineage_id="old-segment",
    )
    backend_calls: list[str] = []
    adapter = PermittedDispatchAdapter(
        gate=gate,
        invocation_provider=lambda _payload: DispatchPermitInvocation(
            permit=permit,
            current_authority_epoch=2,
            # Model a worker reading the now-live epoch after queued work was
            # created under epoch 4.
            current_execution_epoch=5,
        ),
        backend_call=lambda payload: backend_calls.append(payload),
    )

    with pytest.raises(DispatchSeamRejectedError) as captured:
        adapter("queued-old-command")
    assert captured.value.receipt.status == "execution_epoch_mismatch"
    assert backend_calls == []


def test_lifecycle_seam_requires_exact_authorized_operation() -> None:
    value = registry()
    permit = value.issue_lifecycle_control_permit(
        LifecycleControlAuthorization(
            authorization_event_id="lifecycle-auth-1",
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            lifecycle_state_machine_id="legacy-lifecycle",
            requester_kind="lifecycle_state_machine",
            operation="enable",
            target_execution_epoch=4,
            operator_or_safety_policy_id="operator-session-1",
        )
    )
    enable_gate = DispatchSeamGate(
        registry=value,
        spec=DispatchSeamSpec.create(
            seam_id="fake-rm65-force-enable",
            permit_kind="lifecycle_control",
            operation="enable",
            call_stage="pre_sdk_call",
        ),
    )

    calls: list[str] = []
    outcome = enable_gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: calls.append("enabled") or 0,
    )
    assert calls == ["enabled"]
    assert outcome.result == 0
    assert outcome.validation.actual_operation == "enable"


@pytest.mark.parametrize(
    ("permit_kind", "operation"),
    (("motion", "send"), ("safety_control", None), ("lifecycle_control", None)),
)
def test_seam_spec_rejects_ambiguous_operation_contracts(
    permit_kind: str,
    operation: str | None,
) -> None:
    with pytest.raises(ValueError):
        DispatchSeamSpec.create(
            seam_id="invalid",
            permit_kind=permit_kind,
            operation=operation,
        )
