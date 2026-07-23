"""Pure ControlTrace projections for cancellation and safety execution facts."""

from __future__ import annotations

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.runtime import (
    ExecutionCancellation,
    ExecutionFaultEscalationRequest,
    ExecutionWorkItem,
    SafetyControlOperation,
    SafetyControlOperationTickEvidence,
    WorkerEpochDrainAggregate,
    WorkerEpochDrainReceipt,
)


def _scope(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> dict[str, str]:
    values = {
        "runtime_session_id": str(runtime_session_id).strip(),
        "control_domain_id": str(control_domain_id).strip(),
        "embodiment_instance_id": str(embodiment_instance_id).strip(),
        "clock_domain_id": str(clock_domain_id).strip(),
    }
    if any(not value for value in values.values()):
        raise ValueError("ControlTrace execution projection scope must be complete.")
    return values


def project_safety_control_operation(
    operation: SafetyControlOperation,
    *,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(operation, SafetyControlOperation):
        raise TypeError("operation must be a SafetyControlOperation.")
    return ControlTraceFact.create(
        fact_id=operation.operation_id,
        event_type="safety_control_operation",
        **_scope(
            runtime_session_id=operation.runtime_session_id,
            control_domain_id=operation.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        occurred_at=operation.authorized_at,
        payload={
            name: getattr(operation, name)
            for name in operation.__dataclass_fields__
        },
    )


def project_safety_control_tick(
    tick: SafetyControlOperationTickEvidence,
    work_item: ExecutionWorkItem,
    *,
    operation: SafetyControlOperation,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(tick, SafetyControlOperationTickEvidence):
        raise TypeError("tick must be SafetyControlOperationTickEvidence.")
    if not isinstance(work_item, ExecutionWorkItem):
        raise TypeError("work_item must be an ExecutionWorkItem.")
    if tick.operation_id != operation.operation_id:
        raise ValueError("Safety tick belongs to another operation.")
    if work_item.work_kind != "safety_backend_command":
        raise ValueError("Safety tick projection requires safety backend work.")
    if (
        work_item.source_safety_operation_id != operation.operation_id
        or work_item.source_safety_tick_evidence_id != tick.tick_evidence_id
        or work_item.execution_epoch != tick.target_execution_epoch
    ):
        raise ValueError("Safety work lineage does not match tick evidence.")
    return ControlTraceFact.create(
        fact_id=tick.tick_evidence_id,
        event_type="safety_control_tick",
        **_scope(
            runtime_session_id=operation.runtime_session_id,
            control_domain_id=operation.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        occurred_at=tick.admitted_at,
        payload={
            **{
                name: getattr(tick, name)
                for name in tick.__dataclass_fields__
            },
            "work_item_id": work_item.work_item_id,
            "work_kind": work_item.work_kind,
            "authority_epoch": work_item.authority_epoch,
            "segment_generation_by_group": dict(
                work_item.segment_generation_by_group
            ),
            "source_safety_operation_id": work_item.source_safety_operation_id,
            "source_safety_tick_evidence_id": (
                work_item.source_safety_tick_evidence_id
            ),
        },
    )


def project_worker_epoch_drain(
    receipt: WorkerEpochDrainReceipt,
    *,
    cancellation: ExecutionCancellation,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    if not isinstance(receipt, WorkerEpochDrainReceipt):
        raise TypeError("receipt must be a WorkerEpochDrainReceipt.")
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation.")
    if receipt.cancellation_id != cancellation.cancellation_id:
        raise ValueError("Worker drain belongs to another cancellation.")
    return ControlTraceFact.create(
        fact_id=receipt.receipt_id,
        event_type="execution_worker_drain",
        **_scope(
            runtime_session_id=cancellation.runtime_session_id,
            control_domain_id=cancellation.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=cancellation.effective_clock_domain_id,
        ),
        occurred_at=receipt.acknowledged_at,
        payload={
            name: getattr(receipt, name)
            for name in receipt.__dataclass_fields__
        },
    )


def project_worker_epoch_drain_aggregate(
    aggregate: WorkerEpochDrainAggregate,
    *,
    cancellation: ExecutionCancellation,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    if not isinstance(aggregate, WorkerEpochDrainAggregate):
        raise TypeError("aggregate must be a WorkerEpochDrainAggregate.")
    if aggregate.cancellation_id != cancellation.cancellation_id:
        raise ValueError("Worker drain aggregate belongs to another cancellation.")
    return ControlTraceFact.create(
        fact_id=aggregate.aggregate_id,
        event_type="execution_worker_drain_aggregate",
        **_scope(
            runtime_session_id=cancellation.runtime_session_id,
            control_domain_id=cancellation.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=cancellation.effective_clock_domain_id,
        ),
        occurred_at=aggregate.aggregated_at,
        payload={
            name: getattr(aggregate, name)
            for name in aggregate.__dataclass_fields__
        },
    )


def project_execution_fault_escalation(
    escalation: ExecutionFaultEscalationRequest,
    *,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(escalation, ExecutionFaultEscalationRequest):
        raise TypeError("escalation must be an ExecutionFaultEscalationRequest.")
    return ControlTraceFact.create(
        fact_id=escalation.escalation_id,
        event_type="execution_fault_escalation",
        **_scope(
            runtime_session_id=escalation.runtime_session_id,
            control_domain_id=escalation.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        occurred_at=escalation.observed_at,
        payload={
            name: getattr(escalation, name)
            for name in escalation.__dataclass_fields__
        },
    )


__all__ = [
    "project_execution_fault_escalation",
    "project_safety_control_operation",
    "project_safety_control_tick",
    "project_worker_epoch_drain",
    "project_worker_epoch_drain_aggregate",
]
