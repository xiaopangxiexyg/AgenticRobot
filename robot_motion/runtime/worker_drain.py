"""Typed evidence that canceled worker epochs can no longer reach a backend.

Clearing a queue is not sufficient cancellation evidence when a worker may
already hold old work.  This module represents per-participant drain evidence,
aggregates every required worker for one cancellation, and only then projects
the result into the existing ``software_queue`` acknowledgement level.
It does not stop threads or call a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset_contracts.canonical_encoding import canonical_content_hash

from .cancellation import (
    CancellationAcknowledgement,
    CancellationAcknowledgementTracker,
    ExecutionCancellation,
)


EXECUTION_CANCELLATION_PARTICIPANT_HASH_DOMAIN_V1 = (
    "dexteleop.execution_cancellation_participant.v1"
)
WORKER_EPOCH_DRAIN_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.worker_epoch_drain_receipt.v1"
)
WORKER_EPOCH_DRAIN_AGGREGATE_HASH_DOMAIN_V1 = (
    "dexteleop.worker_epoch_drain_aggregate.v1"
)

WORKER_DRAIN_KINDS_V1 = frozenset(
    {
        "reference_worker",
        "position_command_worker",
        "backend_stream_worker",
        "controller_worker",
        "follow_worker",
    }
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique.")
    return normalized


def _finite_time(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class ExecutionCancellationParticipantSpec:
    participant_id: str
    worker_kind: str
    affected_selection_group_ids: tuple[str, ...]
    acknowledgement_level: str
    evidence_capabilities: tuple[str, ...]
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        participant_id: str,
        worker_kind: str,
        affected_selection_group_ids: tuple[str, ...],
        evidence_capabilities: tuple[str, ...] = (
            "discarded_work_ids",
            "in_flight_old_epoch_count",
            "no_future_old_epoch_calls",
        ),
    ) -> "ExecutionCancellationParticipantSpec":
        normalized_kind = _identifier(worker_kind, name="worker_kind")
        if normalized_kind not in WORKER_DRAIN_KINDS_V1:
            raise ValueError("Unsupported worker drain kind.")
        payload = {
            "participant_id": _identifier(participant_id, name="participant_id"),
            "worker_kind": normalized_kind,
            "affected_selection_group_ids": _identifiers(
                affected_selection_group_ids,
                name="affected_selection_group_ids",
            ),
            "acknowledgement_level": "software_queue",
            "evidence_capabilities": _identifiers(
                evidence_capabilities,
                name="evidence_capabilities",
            ),
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=EXECUTION_CANCELLATION_PARTICIPANT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class WorkerEpochDrainReceipt:
    receipt_id: str
    cancellation_id: str
    participant_id: str
    participant_spec_hash: str
    invalidated_execution_epoch: int
    observed_execution_epoch: int
    affected_selection_group_ids: tuple[str, ...]
    discarded_work_ids: tuple[str, ...]
    last_old_epoch_invocation_started_at: float | None
    last_old_epoch_invocation_completed_at: float | None
    in_flight_old_epoch_count: int
    no_future_old_epoch_calls: bool
    acknowledged_at: float
    accepted: bool
    status: str
    semantic_content_hash: str

    @classmethod
    def evaluate(
        cls,
        *,
        cancellation: ExecutionCancellation,
        participant: ExecutionCancellationParticipantSpec,
        observed_execution_epoch: int,
        discarded_work_ids: tuple[str, ...],
        last_old_epoch_invocation_started_at: float | None,
        last_old_epoch_invocation_completed_at: float | None,
        in_flight_old_epoch_count: int,
        no_future_old_epoch_calls: bool,
        acknowledged_at: float,
    ) -> "WorkerEpochDrainReceipt":
        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        if not isinstance(participant, ExecutionCancellationParticipantSpec):
            raise TypeError(
                "participant must be an ExecutionCancellationParticipantSpec."
            )
        observed_epoch = _non_negative_int(
            observed_execution_epoch,
            name="observed_execution_epoch",
        )
        in_flight = _non_negative_int(
            in_flight_old_epoch_count,
            name="in_flight_old_epoch_count",
        )
        started = (
            None
            if last_old_epoch_invocation_started_at is None
            else _finite_time(
                last_old_epoch_invocation_started_at,
                name="last_old_epoch_invocation_started_at",
            )
        )
        completed = (
            None
            if last_old_epoch_invocation_completed_at is None
            else _finite_time(
                last_old_epoch_invocation_completed_at,
                name="last_old_epoch_invocation_completed_at",
            )
        )
        if completed is not None and started is None:
            raise ValueError("A completed invocation requires its start timestamp.")
        if started is not None and completed is not None and completed < started:
            raise ValueError("Old-epoch invocation completion cannot predate its start.")
        timestamp = _finite_time(acknowledged_at, name="acknowledged_at")
        if timestamp < cancellation.effective_at:
            raise ValueError("Worker drain acknowledgement cannot predate cancellation.")

        if observed_epoch != cancellation.execution_epoch:
            status = "execution_epoch_not_observed"
        elif in_flight != 0:
            status = "old_epoch_invocations_in_flight"
        elif not bool(no_future_old_epoch_calls):
            status = "future_old_epoch_calls_not_excluded"
        else:
            status = "drained"
        discarded = _identifiers(
            discarded_work_ids,
            name="discarded_work_ids",
            allow_empty=True,
        )
        semantic_payload = {
            "cancellation_id": cancellation.cancellation_id,
            "participant_id": participant.participant_id,
            "participant_spec_hash": participant.semantic_content_hash,
            "invalidated_execution_epoch": cancellation.previous_execution_epoch,
            "observed_execution_epoch": observed_epoch,
            "affected_selection_group_ids": participant.affected_selection_group_ids,
            "discarded_work_ids": discarded,
            "last_old_epoch_invocation_started_at": started,
            "last_old_epoch_invocation_completed_at": completed,
            "in_flight_old_epoch_count": in_flight,
            "no_future_old_epoch_calls": bool(no_future_old_epoch_calls),
            "acknowledged_at": timestamp,
            "accepted": status == "drained",
            "status": status,
        }
        semantic_hash = canonical_content_hash(
            semantic_payload,
            domain=WORKER_EPOCH_DRAIN_RECEIPT_HASH_DOMAIN_V1,
        )
        return cls(
            receipt_id=f"worker-epoch-drain:{semantic_hash}",
            **semantic_payload,
            semantic_content_hash=semantic_hash,
        )


@dataclass(frozen=True)
class WorkerEpochDrainAggregate:
    aggregate_id: str
    cancellation_id: str
    required_participant_ids: tuple[str, ...]
    accepted_participant_ids: tuple[str, ...]
    missing_participant_ids: tuple[str, ...]
    rejected_participant_ids: tuple[str, ...]
    receipt_refs: tuple[tuple[str, str], ...]
    aggregated_at: float
    complete: bool
    status: str
    semantic_content_hash: str


def aggregate_worker_epoch_drains(
    *,
    cancellation: ExecutionCancellation,
    participants: tuple[ExecutionCancellationParticipantSpec, ...],
    receipts: tuple[WorkerEpochDrainReceipt, ...],
    aggregated_at: float,
) -> WorkerEpochDrainAggregate:
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation.")
    if not participants:
        raise ValueError("Worker drain aggregation requires participants.")
    if any(
        not isinstance(item, ExecutionCancellationParticipantSpec)
        for item in participants
    ):
        raise TypeError("participants contain an unsupported value.")
    if any(not isinstance(item, WorkerEpochDrainReceipt) for item in receipts):
        raise TypeError("receipts contain an unsupported value.")
    required = tuple(sorted(item.participant_id for item in participants))
    if len(set(required)) != len(required):
        raise ValueError("Worker drain participant IDs must be unique.")
    receipt_by_participant: dict[str, WorkerEpochDrainReceipt] = {}
    participant_by_id = {item.participant_id: item for item in participants}
    for receipt in receipts:
        if receipt.cancellation_id != cancellation.cancellation_id:
            raise ValueError("Worker drain receipt belongs to another cancellation.")
        if receipt.participant_id not in participant_by_id:
            raise ValueError("Worker drain receipt references an unexpected participant.")
        if receipt.participant_id in receipt_by_participant:
            raise ValueError("Worker drain participant has duplicate receipts.")
        participant = participant_by_id[receipt.participant_id]
        if receipt.participant_spec_hash != participant.semantic_content_hash:
            raise ValueError("Worker drain receipt uses another participant spec.")
        receipt_by_participant[receipt.participant_id] = receipt

    accepted = tuple(
        participant_id
        for participant_id in required
        if participant_id in receipt_by_participant
        and receipt_by_participant[participant_id].accepted
    )
    missing = tuple(
        participant_id
        for participant_id in required
        if participant_id not in receipt_by_participant
    )
    rejected = tuple(
        participant_id
        for participant_id in required
        if participant_id in receipt_by_participant
        and not receipt_by_participant[participant_id].accepted
    )
    if missing:
        status = "missing_participants"
    elif rejected:
        status = "participant_drain_rejected"
    else:
        status = "complete"
    timestamp = _finite_time(aggregated_at, name="aggregated_at")
    if timestamp < cancellation.effective_at:
        raise ValueError("Worker drain aggregation cannot predate cancellation.")
    receipt_refs = tuple(
        sorted(
            (
                receipt.participant_id,
                receipt.semantic_content_hash,
            )
            for receipt in receipts
        )
    )
    semantic_payload = {
        "cancellation_id": cancellation.cancellation_id,
        "required_participant_ids": required,
        "accepted_participant_ids": accepted,
        "missing_participant_ids": missing,
        "rejected_participant_ids": rejected,
        "receipt_refs": receipt_refs,
        "aggregated_at": timestamp,
        "complete": status == "complete",
        "status": status,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=WORKER_EPOCH_DRAIN_AGGREGATE_HASH_DOMAIN_V1,
    )
    return WorkerEpochDrainAggregate(
        aggregate_id=f"worker-epoch-drain-aggregate:{semantic_hash}",
        **semantic_payload,
        semantic_content_hash=semantic_hash,
    )


def acknowledge_worker_epoch_drain(
    aggregate: WorkerEpochDrainAggregate,
    *,
    tracker: CancellationAcknowledgementTracker,
    clock_domain_id: str,
) -> CancellationAcknowledgement:
    """Project a complete multi-worker drain into ``software_queue`` ack."""

    if not isinstance(aggregate, WorkerEpochDrainAggregate):
        raise TypeError("aggregate must be a WorkerEpochDrainAggregate.")
    if not isinstance(tracker, CancellationAcknowledgementTracker):
        raise TypeError("tracker must be a CancellationAcknowledgementTracker.")
    if not aggregate.complete:
        raise RuntimeError("Incomplete worker drain cannot acknowledge software_queue.")
    return tracker.acknowledge(
        aggregate.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id=_identifier(clock_domain_id, name="clock_domain_id"),
        source_id=aggregate.aggregate_id,
        acknowledged_at=aggregate.aggregated_at,
        status="all_required_workers_drained",
    )
