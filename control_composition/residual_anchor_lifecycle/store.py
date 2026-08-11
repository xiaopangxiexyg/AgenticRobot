"""Receiver-linearized store for residual-anchor pending/ready transitions."""

from __future__ import annotations

import threading

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.producer_epoch import ProducerEpochError, ProducerEpochRegistry

from ..activation import CompositionActivationScope
from .contracts import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    RESIDUAL_ANCHOR_LIFECYCLE_RECORD_HASH_DOMAIN_DEVELOPMENT,
    ResidualAnchorLifecycleAdmissionError,
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleRecord,
    ResidualAnchorLifecycleSpec,
    _required,
    prepare_residual_anchor_lifecycle,
    residual_anchor_lifecycle_record_semantic_payload,
)


class ResidualAnchorLifecycleStore:
    """Low-rate activation-local facts valid until explicitly superseded."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
    ) -> None:
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Lifecycle store and epochs must share one receiver.")
        session = _required(runtime_session_id, name="runtime_session_id")
        if session != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Lifecycle store session differs from receiver clock.")
        self.runtime_session_id = session
        self.control_domain_id = coordinator.control_domain_id
        self.coordinator = coordinator
        self.producer_epochs = producer_epochs
        self._records: list[ResidualAnchorLifecycleRecord] = []
        self._by_id: dict[str, ResidualAnchorLifecycleRecord] = {}
        self._last_by_scope: dict[tuple[str, str], ResidualAnchorLifecycleRecord] = {}
        self._lock = threading.RLock()

    def admit(
        self,
        draft: ResidualAnchorLifecycleDraft,
        *,
        spec: ResidualAnchorLifecycleSpec,
        activation_scope: CompositionActivationScope,
    ) -> ResidualAnchorLifecycleAdmissionReceipt:
        record_id: str | None = None
        try:
            prepared = prepare_residual_anchor_lifecycle(
                draft, spec=spec, activation_scope=activation_scope
            )
            record_id = prepared.record_id
            normalized = prepared.draft
            if normalized.runtime_session_id != self.runtime_session_id:
                raise ResidualAnchorLifecycleAdmissionError(
                    "session_mismatch", "Lifecycle belongs to another session."
                )
            if normalized.control_domain_id != self.control_domain_id:
                raise ResidualAnchorLifecycleAdmissionError(
                    "domain_mismatch", "Lifecycle belongs to another domain."
                )

            def build(available_at: float, admission_sequence: int):
                payload = {
                    "lifecycle_record_id": prepared.record_id,
                    "runtime_session_id": normalized.runtime_session_id,
                    "control_domain_id": normalized.control_domain_id,
                    "producer_id": normalized.producer_id,
                    "producer_epoch": normalized.producer_epoch,
                    "sequence": normalized.sequence,
                    "lifecycle_scope_id": normalized.lifecycle_scope_id,
                    "lifecycle_spec_hash": normalized.lifecycle_spec_hash,
                    "phase": normalized.phase,
                    "effective_for_target_at": normalized.effective_for_target_at,
                    "observed_at": normalized.observed_at,
                    "observed_clock_domain_id": normalized.observed_clock_domain_id,
                    "available_at": available_at,
                    "available_clock_domain_id": (
                        self.coordinator.clock_domain.clock_domain_id
                    ),
                    "control_admission_sequence": admission_sequence,
                    "composition_activation_id": activation_scope.activation_id,
                    "composition_activation_hash": (
                        activation_scope.semantic_content_hash
                    ),
                    "anchor_request": normalized.anchor_request,
                    "anchor_receipt": normalized.anchor_receipt,
                }
                provisional = ResidualAnchorLifecycleRecord(
                    **payload, semantic_content_hash="pending"
                )
                return ResidualAnchorLifecycleRecord(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        residual_anchor_lifecycle_record_semantic_payload(provisional),
                        domain=(
                            RESIDUAL_ANCHOR_LIFECYCLE_RECORD_HASH_DOMAIN_DEVELOPMENT
                        ),
                    ),
                )

            record = self.coordinator.commit(store=self, build=build)
            return ResidualAnchorLifecycleAdmissionReceipt(
                accepted=True,
                lifecycle_record_id=record.lifecycle_record_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                lifecycle_record=record,
            )
        except (
            ProducerEpochError,
            ResidualAnchorLifecycleAdmissionError,
            TypeError,
            ValueError,
        ) as exc:
            code = (
                exc.code
                if isinstance(exc, ResidualAnchorLifecycleAdmissionError)
                else "validation_failed"
            )
            if code == "duplicate" and record_id is not None:
                with self._lock:
                    existing = self._by_id.get(record_id)
                if existing is not None:
                    return ResidualAnchorLifecycleAdmissionReceipt(
                        accepted=True,
                        lifecycle_record_id=existing.lifecycle_record_id,
                        available_at=existing.available_at,
                        available_clock_domain_id=existing.available_clock_domain_id,
                        control_admission_sequence=existing.control_admission_sequence,
                        semantic_content_hash=existing.semantic_content_hash,
                        lifecycle_record=existing,
                        accepted_existing=True,
                    )
            return ResidualAnchorLifecycleAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                lifecycle_record_id=record_id,
            )

    def _insert_admitted(self, record: ResidualAnchorLifecycleRecord) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=record.producer_id,
                    producer_epoch=record.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise ResidualAnchorLifecycleAdmissionError(
                    "inactive_producer_epoch", str(exc)
                ) from exc
            if record.lifecycle_record_id in self._by_id:
                raise ResidualAnchorLifecycleAdmissionError(
                    "duplicate", "Lifecycle record is already admitted."
                )
            key = (record.composition_activation_id, record.lifecycle_scope_id)
            previous = self._last_by_scope.get(key)
            request = record.anchor_request
            if previous is None:
                if (
                    record.phase != RESIDUAL_ANCHOR_LIFECYCLE_PENDING
                    or request.request_generation != 0
                ):
                    raise ResidualAnchorLifecycleAdmissionError(
                        "initial_pending_required",
                        "Lifecycle must begin with generation-zero pending.",
                    )
            elif (
                previous.producer_id != record.producer_id
                or previous.producer_epoch != record.producer_epoch
                or previous.lifecycle_spec_hash != record.lifecycle_spec_hash
                or previous.composition_activation_hash
                != record.composition_activation_hash
            ):
                raise ResidualAnchorLifecycleAdmissionError(
                    "scope_owner_changed", "Lifecycle owner or protocol changed."
                )
            elif record.sequence <= previous.sequence:
                raise ResidualAnchorLifecycleAdmissionError(
                    "sequence_rollback", "Lifecycle sequence must increase."
                )
            elif previous.phase == RESIDUAL_ANCHOR_LIFECYCLE_PENDING:
                if (
                    record.phase != RESIDUAL_ANCHOR_LIFECYCLE_READY
                    or request.semantic_content_hash
                    != previous.anchor_request.semantic_content_hash
                    or record.effective_for_target_at
                    != previous.effective_for_target_at
                ):
                    raise ResidualAnchorLifecycleAdmissionError(
                        "pending_completion_mismatch",
                        "Pending may only complete its exact request and boundary.",
                    )
            else:
                previous_receipt = previous.anchor_receipt
                assert previous_receipt is not None
                if (
                    record.phase != RESIDUAL_ANCHOR_LIFECYCLE_PENDING
                    or request.request_generation
                    != previous_receipt.request_generation + 1
                    or request.previous_request_id != previous_receipt.request_id
                    or request.previous_request_hash != previous_receipt.request_hash
                    or record.effective_for_target_at
                    < previous.effective_for_target_at
                ):
                    raise ResidualAnchorLifecycleAdmissionError(
                        "successor_request_mismatch",
                        "Ready may only advance to its exact successor request.",
                    )
            self._records.append(record)
            self._by_id[record.lifecycle_record_id] = record
            self._last_by_scope[key] = record

    def records_at_or_before(
        self, cut: ControlCut
    ) -> tuple[ResidualAnchorLifecycleRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(
                record for record in self._records if admitted_at_or_before(record, cut)
            )

    def record_by_id(self, record_id: str) -> ResidualAnchorLifecycleRecord | None:
        with self._lock:
            return self._by_id.get(str(record_id))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


__all__ = ["ResidualAnchorLifecycleStore"]
