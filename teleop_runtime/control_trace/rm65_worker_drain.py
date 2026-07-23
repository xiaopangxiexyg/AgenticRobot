"""Best-effort trace admission for already-completed RM65 worker drains.

Cancellation safety cannot depend on storage health.  This adapter therefore
projects immutable receipts only after workers have drained, reports every
trace-admission failure, and never upgrades a partial report to evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.runtime import ExecutionCancellation
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainResult,
)

from .execution_projection import (
    project_worker_epoch_drain,
    project_worker_epoch_drain_aggregate,
)


@dataclass(frozen=True)
class RM65WorkerDrainTraceAdmissionReport:
    cancellation_id: str
    attempted_fact_ids: tuple[str, ...]
    accepted_fact_ids: tuple[str, ...]
    failed_fact_ids: tuple[str, ...]
    failure_messages: tuple[str, ...]
    complete: bool


def admit_rm65_worker_drain_trace(
    result: RM65ArmsOnlyWorkerDrainResult,
    *,
    cancellation: ExecutionCancellation,
    embodiment_instance_id: str,
    fact_sink: Callable[[ControlTraceFact], object],
) -> RM65WorkerDrainTraceAdmissionReport:
    """Admit drain facts without changing the completed cancellation result."""

    if not isinstance(result, RM65ArmsOnlyWorkerDrainResult):
        raise TypeError("result must be RM65ArmsOnlyWorkerDrainResult.")
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation.")
    if result.cancellation_id != cancellation.cancellation_id:
        raise ValueError("Worker drain result belongs to another cancellation.")
    if not callable(fact_sink):
        raise TypeError("fact_sink must be callable.")

    facts = tuple(
        project_worker_epoch_drain(
            receipt,
            cancellation=cancellation,
            embodiment_instance_id=embodiment_instance_id,
        )
        for receipt in result.receipts
    ) + (
        project_worker_epoch_drain_aggregate(
            result.aggregate,
            cancellation=cancellation,
            embodiment_instance_id=embodiment_instance_id,
        ),
    )
    accepted: list[str] = []
    failed: list[str] = []
    messages: list[str] = []
    for fact in facts:
        try:
            fact_sink(fact)
        except Exception as exc:  # noqa: BLE001 - storage failure is evidence.
            failed.append(fact.fact_id)
            messages.append(f"{fact.fact_id}:{type(exc).__name__}:{exc}")
        else:
            accepted.append(fact.fact_id)
    return RM65WorkerDrainTraceAdmissionReport(
        cancellation_id=cancellation.cancellation_id,
        attempted_fact_ids=tuple(fact.fact_id for fact in facts),
        accepted_fact_ids=tuple(accepted),
        failed_fact_ids=tuple(failed),
        failure_messages=tuple(messages),
        complete=not failed,
    )


__all__ = [
    "RM65WorkerDrainTraceAdmissionReport",
    "admit_rm65_worker_drain_trace",
]
