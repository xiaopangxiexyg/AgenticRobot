from __future__ import annotations

import pytest

from robot_motion.runtime import (
    DispatchOwnerRegistry,
    DispatchSeamGate,
    DispatchSeamRejectedError,
    DispatchSeamSpec,
    RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1,
    SafetyControlOperation,
    SafetyControlOperationRegistry,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _operation(clock: ManualClock) -> SafetyControlOperation:
    return SafetyControlOperation.create_rm65_quick_stop(
        operation_id="rm65-quick-stop-0",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        safety_source="runtime_supervisor",
        source_event_id="collision-stop-0",
        target_execution_epoch=4,
        affected_selection_groups=("arms",),
        frozen_stop_target_hash="frozen-stop-trajectory-hash",
        maximum_duration_sec=0.1,
        maximum_tick_count=3,
        authorized_at=clock(),
    )


def _registries(clock: ManualClock):
    operation_registry = SafetyControlOperationRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        clock=clock,
    )
    dispatch_registry = DispatchOwnerRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        initial_authority_epoch=2,
        initial_execution_epoch=4,
        clock=clock,
    )
    gate = DispatchSeamGate(
        registry=dispatch_registry,
        spec=DispatchSeamSpec.create(
            seam_id="rm65.quick-stop.pre-sdk",
            permit_kind="safety_control",
            operation=RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1,
            call_stage="pre_sdk_call",
        ),
    )
    return operation_registry, dispatch_registry, gate


def test_each_quick_stop_tick_requires_new_operation_evidence_and_permit() -> None:
    clock = ManualClock()
    operations, dispatch, gate = _registries(clock)
    operation = _operation(clock)
    operations.admit(operation)
    sent: list[int] = []

    for tick_sequence in (1, 2, 3):
        tick = operations.admit_tick(
            operation_id=operation.operation_id,
            tick_sequence=tick_sequence,
            current_execution_epoch=4,
            command_lineage_id=f"quick-stop-command-{tick_sequence}",
        )
        permit = dispatch.issue_safety_control_permit(
            operation=operation.allowed_dispatch_operation,
            evidence=tick.as_dispatch_evidence(operation),
            safety_source=operation.safety_source,
        )
        result = gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda tick_sequence=tick_sequence: sent.append(
                tick_sequence
            ),
        )
        assert result.validation.accepted

    assert sent == [1, 2, 3]
    with pytest.raises(RuntimeError, match="budget"):
        operations.admit_tick(
            operation_id=operation.operation_id,
            tick_sequence=4,
            current_execution_epoch=4,
            command_lineage_id="quick-stop-command-4",
        )


def test_operation_timeout_epoch_change_and_terminal_block_future_ticks() -> None:
    clock = ManualClock()
    operations, _dispatch, _gate = _registries(clock)
    operation = _operation(clock)
    operations.admit(operation)

    with pytest.raises(RuntimeError, match="epoch"):
        operations.admit_tick(
            operation_id=operation.operation_id,
            tick_sequence=1,
            current_execution_epoch=5,
            command_lineage_id="wrong-epoch",
        )

    clock.value = 10.2
    with pytest.raises(RuntimeError, match="expired"):
        operations.admit_tick(
            operation_id=operation.operation_id,
            tick_sequence=1,
            current_execution_epoch=4,
            command_lineage_id="expired",
        )

    terminal_clock = ManualClock()
    terminal_operations, _dispatch, _gate = _registries(terminal_clock)
    terminal_operation = _operation(terminal_clock)
    terminal_operations.admit(terminal_operation)
    terminal = terminal_operations.terminate(
        operation_id=terminal_operation.operation_id,
        status="canceled",
        reason="hold_reasserted",
    )
    assert terminal.last_admitted_tick_sequence == 0
    with pytest.raises(RuntimeError, match="terminal"):
        terminal_operations.admit_tick(
            operation_id=terminal_operation.operation_id,
            tick_sequence=1,
            current_execution_epoch=4,
            command_lineage_id="after-terminal",
        )


def test_operation_authorization_is_not_a_reusable_dispatch_permit() -> None:
    clock = ManualClock()
    operations, dispatch, gate = _registries(clock)
    operation = _operation(clock)
    operations.admit(operation)
    tick = operations.admit_tick(
        operation_id=operation.operation_id,
        tick_sequence=1,
        current_execution_epoch=4,
        command_lineage_id="quick-stop-command-1",
    )
    permit = dispatch.issue_safety_control_permit(
        operation=operation.allowed_dispatch_operation,
        evidence=tick.as_dispatch_evidence(operation),
        safety_source=operation.safety_source,
    )
    gate.invoke(
        permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
        backend_call=lambda: None,
    )

    with pytest.raises(DispatchSeamRejectedError) as captured:
        gate.invoke(
            permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
            backend_call=lambda: None,
        )
    assert captured.value.receipt.status == "permit_already_consumed"


def test_tick_builds_new_epoch_safety_work_without_motion_lineage() -> None:
    clock = ManualClock()
    operations, _dispatch, _gate = _registries(clock)
    operation = _operation(clock)
    operations.admit(operation)
    tick = operations.admit_tick(
        operation_id=operation.operation_id,
        tick_sequence=1,
        current_execution_epoch=4,
        command_lineage_id="quick-stop-command-1",
    )

    work = tick.as_execution_work_item(
        operation,
        authority_epoch=2,
        segment_generation_by_group={"arms": 8},
    )

    assert work.work_kind == "safety_backend_command"
    assert work.execution_epoch == 4
    assert work.source_decision_id is None
    assert work.source_reference_segment_id is None
    assert work.source_safety_operation_id == operation.operation_id
    assert work.source_safety_tick_evidence_id == tick.tick_evidence_id
