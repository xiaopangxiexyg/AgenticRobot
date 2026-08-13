"""Pure M5.8 legacy-owner worker-drain and backend-HOLD convergence.

This module performs no worker drain and no backend call.  It validates typed
evidence produced by those lower operations and only then advances the unique
``LegacyControlDomainExecutionContext`` lifecycle/execution gate.  The policy-
motion binding remains fake-only; the production-scoped path accepted here is
the separately authorized stop-only preflight and confers no motion authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import ExecutionCancellation, aggregate_worker_epoch_drains
from teleop_runtime.action import (
    LegacyControlDomainExecutionContextSnapshot,
    LegacyControlOwnerStateTransitionRequest,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainResult,
)


@runtime_checkable
class RM65LegacyControlStopBinding(Protocol):
    """Minimum exact owner scope accepted by the stop convergence core."""

    binding_id: str
    context: object
    worker_drain_plan: object

    @property
    def production_binding_enabled(self) -> bool: ...


@runtime_checkable
class RM65IdleWorkerStopEvidence(Protocol):
    """Structural lower stop receipt; avoids a deployment/backend import cycle."""

    side: str
    evidence_event_id: str
    validation_receipt_id: str
    dispatch_accepted: bool
    dispatch_status: str
    dispatch_validation_hash: str
    sdk_result_code: int | None
    sdk_succeeded: bool


@runtime_checkable
class RM65IdleWorkerStopEvidenceBinding(RM65LegacyControlStopBinding, Protocol):
    """Authorized stop-only binding that owns its lower SDK receipts."""

    @property
    def stop_receipts(self) -> tuple[RM65IdleWorkerStopEvidence, ...]: ...


RM65_LEGACY_BACKEND_HOLD_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_legacy_backend_hold_receipt.v1"
)
RM65_LEGACY_CONTROL_STOP_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_legacy_control_stop_receipt.v1"
)
RM65_LEGACY_REQUIRED_BACKEND_IDS_V1 = ("left", "right")


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite_time(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class RM65LegacyBackendHoldReceipt:
    receipt_id: str
    cancellation_id: str
    cancellation_hash: str
    execution_epoch: int
    required_backend_ids: tuple[str, ...]
    acknowledged_backend_ids: tuple[str, ...]
    rejected_backend_ids: tuple[str, ...]
    acknowledged_at: float
    clock_domain_id: str
    evidence_refs: tuple[str, ...]
    complete: bool
    status: str
    production_binding_enabled: bool
    semantic_content_hash: str

    @classmethod
    def evaluate(
        cls,
        *,
        cancellation: ExecutionCancellation,
        acknowledgement_by_backend_id: Mapping[str, bool],
        acknowledged_at: float,
        clock_domain_id: str,
        evidence_refs: tuple[str, ...],
        production_binding_enabled: bool = False,
    ) -> RM65LegacyBackendHoldReceipt:
        if bool(production_binding_enabled):
            raise ValueError(
                "Production backend HOLD evidence must be compiled from typed "
                "lower stop receipts."
            )
        return cls._build(
            cancellation=cancellation,
            acknowledgement_by_backend_id=acknowledgement_by_backend_id,
            acknowledged_at=acknowledged_at,
            clock_domain_id=clock_domain_id,
            evidence_refs=evidence_refs,
            production_binding_enabled=False,
        )

    @classmethod
    def _build(
        cls,
        *,
        cancellation: ExecutionCancellation,
        acknowledgement_by_backend_id: Mapping[str, bool],
        acknowledged_at: float,
        clock_domain_id: str,
        evidence_refs: tuple[str, ...],
        production_binding_enabled: bool,
    ) -> RM65LegacyBackendHoldReceipt:
        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        acknowledgements = {
            _identifier(key, name="backend_id"): bool(value)
            for key, value in dict(acknowledgement_by_backend_id).items()
        }
        if set(acknowledgements) != set(RM65_LEGACY_REQUIRED_BACKEND_IDS_V1):
            raise ValueError(
                "RM65 backend HOLD receipt must exactly cover left and right."
            )
        timestamp = _finite_time(acknowledged_at, name="acknowledged_at")
        if timestamp < cancellation.effective_at:
            raise ValueError("Backend HOLD acknowledgement predates cancellation.")
        refs = tuple(
            sorted({_identifier(value, name="evidence_ref") for value in evidence_refs})
        )
        if not refs:
            raise ValueError("Backend HOLD receipt requires lower evidence.")
        acknowledged = tuple(
            backend_id
            for backend_id in RM65_LEGACY_REQUIRED_BACKEND_IDS_V1
            if acknowledgements[backend_id]
        )
        rejected = tuple(
            backend_id
            for backend_id in RM65_LEGACY_REQUIRED_BACKEND_IDS_V1
            if not acknowledgements[backend_id]
        )
        payload = {
            "cancellation_id": cancellation.cancellation_id,
            "cancellation_hash": cancellation.semantic_content_hash,
            "execution_epoch": cancellation.execution_epoch,
            "required_backend_ids": RM65_LEGACY_REQUIRED_BACKEND_IDS_V1,
            "acknowledged_backend_ids": acknowledged,
            "rejected_backend_ids": rejected,
            "acknowledged_at": timestamp,
            "clock_domain_id": _identifier(
                clock_domain_id,
                name="clock_domain_id",
            ),
            "evidence_refs": refs,
            "complete": not rejected,
            "status": "backend_hold_acknowledged" if not rejected else "backend_hold_rejected",
            "production_binding_enabled": bool(production_binding_enabled),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RM65_LEGACY_BACKEND_HOLD_RECEIPT_HASH_DOMAIN_V1,
        )
        return cls(
            receipt_id="rm65-legacy-backend-hold:" + semantic_hash,
            **payload,
            semantic_content_hash=semantic_hash,
        )


def compile_rm65_idle_worker_backend_hold_receipt(
    *,
    binding: RM65IdleWorkerStopEvidenceBinding,
    cancellation: ExecutionCancellation,
    acknowledged_at: float,
    clock_domain_id: str,
) -> RM65LegacyBackendHoldReceipt:
    """Compile exact dual-side successful stop calls into production HOLD evidence."""

    if not isinstance(binding, RM65IdleWorkerStopEvidenceBinding):
        raise TypeError("binding does not own typed idle-worker stop receipts.")
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation.")
    context_snapshot = binding.context.snapshot()
    if (
        not binding.production_binding_enabled
        or context_snapshot.composition_mode
        != "hardware_idle_drain_stop_only_preflight"
        or not context_snapshot.hardware_authorization_id
        or context_snapshot.last_execution_cancellation_id
        != cancellation.cancellation_id
        or context_snapshot.last_execution_cancellation_hash
        != cancellation.semantic_content_hash
        or context_snapshot.execution_epoch != cancellation.execution_epoch
    ):
        raise ValueError(
            "Production HOLD compilation requires the exact authorized current "
            "idle-worker stop context."
        )
    if clock_domain_id != binding.context.clock_domain_id:
        raise ValueError("Backend HOLD clock does not match its owner context.")
    receipts = tuple(binding.stop_receipts)
    if any(not isinstance(item, RM65IdleWorkerStopEvidence) for item in receipts):
        raise TypeError("stop_receipts must contain typed idle-worker receipts.")
    by_side = {item.side: item for item in receipts}
    if len(by_side) != len(receipts) or set(by_side) != set(
        RM65_LEGACY_REQUIRED_BACKEND_IDS_V1
    ):
        raise ValueError("Production backend HOLD requires exact left/right receipts.")
    acknowledgement_by_backend_id: dict[str, bool] = {}
    evidence_refs: list[str] = []
    for side in RM65_LEGACY_REQUIRED_BACKEND_IDS_V1:
        receipt = by_side[side]
        if receipt.evidence_event_id != cancellation.cancellation_id:
            raise ValueError(
                f"{side} stop receipt is bound to another cancellation."
            )
        accepted = bool(
            receipt.dispatch_accepted
            and receipt.dispatch_status == "accepted"
            and receipt.sdk_succeeded
            and receipt.sdk_result_code == 0
        )
        acknowledgement_by_backend_id[side] = accepted
        evidence_refs.extend(
            (receipt.validation_receipt_id, receipt.dispatch_validation_hash)
        )
    return RM65LegacyBackendHoldReceipt._build(
        cancellation=cancellation,
        acknowledgement_by_backend_id=acknowledgement_by_backend_id,
        acknowledged_at=acknowledged_at,
        clock_domain_id=clock_domain_id,
        evidence_refs=tuple(evidence_refs),
        production_binding_enabled=True,
    )


@dataclass(frozen=True)
class RM65LegacyControlStopReceipt:
    receipt_id: str
    binding_id: str
    cancellation_id: str
    cancellation_hash: str
    worker_drain_aggregate_id: str
    worker_drain_aggregate_hash: str
    backend_hold_receipt_id: str
    backend_hold_receipt_hash: str
    lifecycle_state: str
    execution_gate: str
    owner_state_revision: int
    owner_state_hash: str
    completed_at: float
    complete: bool
    status: str
    production_binding_enabled: bool
    semantic_content_hash: str


def converge_rm65_legacy_control_stop(
    *,
    binding: RM65LegacyControlStopBinding,
    cancellation: ExecutionCancellation,
    worker_drain: RM65ArmsOnlyWorkerDrainResult,
    backend_hold: RM65LegacyBackendHoldReceipt,
) -> RM65LegacyControlStopReceipt:
    """Advance to HOLD/FAULT only after exact drain and backend evidence."""

    if not isinstance(binding, RM65LegacyControlStopBinding):
        raise TypeError("binding does not provide the exact legacy stop scope.")
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation.")
    if not isinstance(worker_drain, RM65ArmsOnlyWorkerDrainResult):
        raise TypeError("worker_drain must be an RM65ArmsOnlyWorkerDrainResult.")
    if not isinstance(backend_hold, RM65LegacyBackendHoldReceipt):
        raise TypeError("backend_hold must be a typed RM65 receipt.")
    context = binding.context
    before = context.snapshot()
    if (
        cancellation.runtime_session_id != context.runtime_session_id
        or cancellation.control_domain_id != context.control_domain_id
        or before.last_execution_cancellation_id != cancellation.cancellation_id
        or before.last_execution_cancellation_hash
        != cancellation.semantic_content_hash
        or before.execution_epoch != cancellation.execution_epoch
    ):
        raise ValueError("Control-stop cancellation is not current in this context.")
    if (
        worker_drain.plan_id != binding.worker_drain_plan.plan_id
        or worker_drain.cancellation_id != cancellation.cancellation_id
        or worker_drain.aggregate.cancellation_id != cancellation.cancellation_id
    ):
        raise ValueError("Worker-drain evidence is bound to another plan/cancellation.")
    recomputed_drain = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=binding.worker_drain_plan.participants,
        receipts=worker_drain.receipts,
        aggregated_at=worker_drain.aggregate.aggregated_at,
    )
    if recomputed_drain != worker_drain.aggregate:
        raise ValueError("Worker-drain aggregate content/hash is invalid.")
    if (
        backend_hold.cancellation_id != cancellation.cancellation_id
        or backend_hold.cancellation_hash != cancellation.semantic_content_hash
        or backend_hold.execution_epoch != cancellation.execution_epoch
        or backend_hold.clock_domain_id != context.clock_domain_id
    ):
        raise ValueError("Backend HOLD evidence is bound to another cancellation/clock.")
    if backend_hold.production_binding_enabled != binding.production_binding_enabled:
        raise ValueError("Backend HOLD evidence maturity disagrees with the binding.")
    expected_acknowledged = tuple(
        backend_id
        for backend_id in backend_hold.required_backend_ids
        if backend_id not in backend_hold.rejected_backend_ids
    )
    hold_payload = {
        "cancellation_id": backend_hold.cancellation_id,
        "cancellation_hash": backend_hold.cancellation_hash,
        "execution_epoch": backend_hold.execution_epoch,
        "required_backend_ids": backend_hold.required_backend_ids,
        "acknowledged_backend_ids": backend_hold.acknowledged_backend_ids,
        "rejected_backend_ids": backend_hold.rejected_backend_ids,
        "acknowledged_at": backend_hold.acknowledged_at,
        "clock_domain_id": backend_hold.clock_domain_id,
        "evidence_refs": backend_hold.evidence_refs,
        "complete": backend_hold.complete,
        "status": backend_hold.status,
        "production_binding_enabled": backend_hold.production_binding_enabled,
    }
    expected_hold_hash = canonical_content_hash(
        hold_payload,
        domain=RM65_LEGACY_BACKEND_HOLD_RECEIPT_HASH_DOMAIN_V1,
    )
    if (
        backend_hold.required_backend_ids != RM65_LEGACY_REQUIRED_BACKEND_IDS_V1
        or expected_acknowledged != backend_hold.acknowledged_backend_ids
        or backend_hold.complete != (not backend_hold.rejected_backend_ids)
        or backend_hold.status
        != (
            "backend_hold_acknowledged"
            if backend_hold.complete
            else "backend_hold_rejected"
        )
        or backend_hold.semantic_content_hash != expected_hold_hash
        or backend_hold.receipt_id != "rm65-legacy-backend-hold:" + expected_hold_hash
    ):
        raise ValueError("Backend HOLD receipt content/hash is invalid.")

    stop_complete = bool(worker_drain.aggregate.complete and backend_hold.complete)
    target_lifecycle = (
        "fault" if cancellation.reason == "fault" or not stop_complete else "hold"
    )
    target_gate = "fault" if target_lifecycle == "fault" else "hold"
    status = (
        "fault_stop_converged"
        if cancellation.reason == "fault" and stop_complete
        else "hold_converged"
        if stop_complete
        else "stop_evidence_incomplete_faulted"
    )
    transition = context.transition_owner_state(
        LegacyControlOwnerStateTransitionRequest(
            target_lifecycle_state=target_lifecycle,
            target_execution_gate=target_gate,
            reason=status,
            source_event_ids=(
                cancellation.cancellation_id,
                worker_drain.aggregate.aggregate_id,
                backend_hold.receipt_id,
            ),
        )
    )
    snapshot: LegacyControlDomainExecutionContextSnapshot = transition.snapshot
    completed_at = max(
        cancellation.effective_at,
        worker_drain.aggregate.aggregated_at,
        backend_hold.acknowledged_at,
        snapshot.captured_at,
    )
    payload = {
        "binding_id": binding.binding_id,
        "cancellation_id": cancellation.cancellation_id,
        "cancellation_hash": cancellation.semantic_content_hash,
        "worker_drain_aggregate_id": worker_drain.aggregate.aggregate_id,
        "worker_drain_aggregate_hash": worker_drain.aggregate.semantic_content_hash,
        "backend_hold_receipt_id": backend_hold.receipt_id,
        "backend_hold_receipt_hash": backend_hold.semantic_content_hash,
        "lifecycle_state": snapshot.lifecycle_state,
        "execution_gate": snapshot.execution_gate,
        "owner_state_revision": snapshot.owner_state_revision,
        "owner_state_hash": snapshot.semantic_content_hash,
        "completed_at": completed_at,
        "complete": stop_complete,
        "status": status,
        "production_binding_enabled": binding.production_binding_enabled,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=RM65_LEGACY_CONTROL_STOP_RECEIPT_HASH_DOMAIN_V1,
    )
    return RM65LegacyControlStopReceipt(
        receipt_id="rm65-legacy-control-stop:" + semantic_hash,
        **payload,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "RM65_LEGACY_REQUIRED_BACKEND_IDS_V1",
    "RM65LegacyBackendHoldReceipt",
    "RM65LegacyControlStopBinding",
    "RM65LegacyControlStopReceipt",
    "RM65IdleWorkerStopEvidence",
    "RM65IdleWorkerStopEvidenceBinding",
    "compile_rm65_idle_worker_backend_hold_receipt",
    "converge_rm65_legacy_control_stop",
]
