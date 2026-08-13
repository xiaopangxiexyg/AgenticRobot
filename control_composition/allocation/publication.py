"""Receipt-driven state commit for one allocation publication scope."""

from __future__ import annotations

import threading
import time
from typing import Callable

from control_composition.activation import CompositionActivationScope
from control_composition.composition.context import (
    CompositionContextAdmissionReceipt,
    CompositionContextDraft,
    CompositionContextRecord,
    composition_context_submission_semantic_hash,
)

from .context import build_composition_allocation_context_draft
from .contracts import (
    COMPOSITION_ALLOCATION_STATUS_NO_DECISION,
    COMPOSITION_ALLOCATION_STATUS_READY,
    CompositionAllocationSpec,
)
from .decision import (
    CompositionAllocationDecision,
    CompositionAllocationState,
)
from .publication_contracts import (
    ALLOCATION_PUBLICATION_ATTEMPT_HASH_DOMAIN_DEVELOPMENT,
    ALLOCATION_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT,
    ALLOCATION_PUBLICATION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
    ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED,
    ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED,
    ALLOCATION_PUBLICATION_OUTCOME_HASH_DOMAIN_DEVELOPMENT,
    ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
    ALLOCATION_PUBLICATION_OUTCOME_LATE,
    ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
    ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION,
    ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
    ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN,
    ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED,
    ALLOCATION_PUBLICATION_RUNTIME_PENDING,
    ALLOCATION_PUBLICATION_RUNTIME_READY,
    AllocationPublicationBasis,
    AllocationPublicationOutcome,
    AllocationPublicationRuntimeSnapshot,
    PendingAllocationPublication,
    allocation_publication_control_cut_semantic_hash,
    _finite,
    _nonnegative_int,
    _required,
)
from .reducer import evaluate_composition_allocation
from .reducer import empty_composition_allocation_state
from .override import (
    CompositionAllocationOverrideReceipt,
    CompositionAllocationOverrideRequest,
    CompositionAllocationOverrideState,
    apply_composition_allocation_override,
)


ContextAdmitter = Callable[
    [CompositionContextDraft], CompositionContextAdmissionReceipt
]
ContextResolver = Callable[[str], CompositionContextRecord | None]


class AllocationPublicationRuntime:
    """Serialize evaluation, context admission, and reducer state commit."""

    def __init__(
        self,
        *,
        allocation_scope_id: str,
        allocation_spec: CompositionAllocationSpec,
        activation_scope: CompositionActivationScope,
        runtime_session_id: str,
        control_domain_id: str,
        control_clock_reset_epoch: int,
        context_channel: str,
        producer_id: str,
        producer_epoch: int,
        initial_reducer_state: CompositionAllocationState,
        initial_override_state: CompositionAllocationOverrideState | None = None,
        persistent_override_selection_lease_sec: float | None = None,
        initial_publication_sequence: int = -1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(allocation_spec, CompositionAllocationSpec):
            raise TypeError("allocation_spec must be CompositionAllocationSpec.")
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        if not isinstance(initial_reducer_state, CompositionAllocationState):
            raise TypeError("initial_reducer_state must be allocation state.")
        if initial_reducer_state.allocation_spec_hash != (
            allocation_spec.semantic_content_hash
        ):
            raise ValueError("Initial reducer state belongs to another spec.")
        if initial_override_state is not None:
            if not isinstance(
                initial_override_state,
                CompositionAllocationOverrideState,
            ):
                raise TypeError("initial_override_state must be typed or None.")
            if (
                initial_override_state.runtime_session_id
                != runtime_session_id
                or initial_override_state.control_domain_id != control_domain_id
                or initial_override_state.composition_activation_id
                != activation_scope.activation_id
                or initial_override_state.composition_activation_hash
                != activation_scope.semantic_content_hash
                or initial_override_state.allocation_spec_hash
                != allocation_spec.semantic_content_hash
                or initial_override_state.component_ids
                != allocation_spec.component_ids
                or initial_override_state.clock_domain_id
                != activation_scope.clock_domain_id
            ):
                raise ValueError(
                    "Initial override belongs to another allocation activation."
                )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        override_lease = (
            None
            if persistent_override_selection_lease_sec is None
            else _finite(
                persistent_override_selection_lease_sec,
                name="persistent_override_selection_lease_sec",
            )
        )
        if override_lease is not None and override_lease <= 0.0:
            raise ValueError(
                "persistent_override_selection_lease_sec must be positive."
            )
        if isinstance(initial_publication_sequence, bool) or not isinstance(
            initial_publication_sequence, int
        ) or initial_publication_sequence < -1:
            raise ValueError("initial_publication_sequence must be >= -1.")
        self.allocation_scope_id = _required(
            allocation_scope_id,
            name="allocation_scope_id",
        )
        self.allocation_spec = allocation_spec
        self.activation_scope = activation_scope
        self.runtime_session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.control_domain_id = _required(
            control_domain_id,
            name="control_domain_id",
        )
        self.control_clock_reset_epoch = _nonnegative_int(
            control_clock_reset_epoch,
            name="control_clock_reset_epoch",
        )
        self.context_channel = _required(context_channel, name="context_channel")
        self.producer_id = _required(producer_id, name="producer_id")
        self.producer_epoch = _nonnegative_int(
            producer_epoch,
            name="producer_epoch",
        )
        self._clock = clock
        self.persistent_override_selection_lease_sec = override_lease
        self._committed_state = initial_reducer_state
        self._override_state = initial_override_state
        self._last_reserved_sequence = initial_publication_sequence
        self._last_terminal_sequence: int | None = None
        self._last_accepted_sequence: int | None = None
        self._pending: PendingAllocationPublication | None = None
        self._invalidated_pending: PendingAllocationPublication | None = None
        self._latest_coalesced: AllocationPublicationBasis | None = None
        self._last_evaluated_basis_key: tuple[int, int, float, str] | None = None
        self._pending_retry_count = 0
        self._last_attempt_at: float | None = None
        self._submission_in_progress = False
        self._invalidation_reason: str | None = None
        self._active_anchor_request_generation: int | None = None
        self._fenced_attempt_by_id: dict[str, PendingAllocationPublication] = {}
        self._override_fenced_attempt_by_id: dict[
            str, PendingAllocationPublication
        ] = {}
        self._lock = threading.RLock()

    def _now(self) -> float:
        return _finite(self._clock(), name="allocation_publication_clock")

    @property
    def runtime_status(self) -> str:
        with self._lock:
            if self._invalidation_reason is not None:
                return ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED
            if self._pending is not None:
                return ALLOCATION_PUBLICATION_RUNTIME_PENDING
            return ALLOCATION_PUBLICATION_RUNTIME_READY

    @property
    def committed_reducer_state(self) -> CompositionAllocationState:
        with self._lock:
            return self._committed_state

    def snapshot(self) -> AllocationPublicationRuntimeSnapshot:
        with self._lock:
            return AllocationPublicationRuntimeSnapshot(
                runtime_status=self.runtime_status,
                committed_reducer_state=self._committed_state,
                last_reserved_sequence=self._last_reserved_sequence,
                last_terminal_sequence=self._last_terminal_sequence,
                last_accepted_sequence=self._last_accepted_sequence,
                pending_attempt=self._pending,
                latest_coalesced_basis=self._latest_coalesced,
                pending_retry_count=self._pending_retry_count,
                last_attempt_at=self._last_attempt_at,
                submission_in_progress=self._submission_in_progress,
                invalidation_reason=self._invalidation_reason,
                active_anchor_request_generation=(
                    self._active_anchor_request_generation
                ),
                allocation_override_state=self._override_state,
            )

    @property
    def allocation_override_state(
        self,
    ) -> CompositionAllocationOverrideState | None:
        with self._lock:
            return self._override_state

    def apply_override(
        self,
        request: CompositionAllocationOverrideRequest,
    ) -> CompositionAllocationOverrideReceipt:
        """Commit one button/profile override and fence older pending context."""

        if not isinstance(request, CompositionAllocationOverrideRequest):
            raise TypeError("request must be a typed allocation override request.")
        now = self._now()
        with self._lock:
            if self._invalidation_reason is not None:
                raise RuntimeError("Allocation publication runtime is invalidated.")
            state = self._override_state
            if state is None:
                raise RuntimeError("Allocation overrides are not enabled.")
            receipt = apply_composition_allocation_override(
                state,
                request,
                admitted_at=now,
            )
            if receipt.status == "duplicate":
                return receipt
            fenced = self._pending
            if fenced is not None:
                if len(self._override_fenced_attempt_by_id) >= 32:
                    oldest_id = min(
                        self._override_fenced_attempt_by_id,
                        key=lambda attempt_id: (
                            self._override_fenced_attempt_by_id[attempt_id]
                            .publication_sequence
                        ),
                    )
                    self._override_fenced_attempt_by_id.pop(oldest_id, None)
                self._override_fenced_attempt_by_id[fenced.attempt_id] = fenced
            self._override_state = receipt.state_after
            self._pending = None
            self._latest_coalesced = None
            self._pending_retry_count = 0
            self._submission_in_progress = False
            self._last_evaluated_basis_key = None
            return receipt

    def advance_anchor_generation(
        self,
        request_generation: int,
    ) -> AllocationPublicationOutcome:
        """Fence old reducer work while keeping this activation usable."""

        generation = _nonnegative_int(
            request_generation,
            name="request_generation",
        )
        now = self._now()
        with self._lock:
            if self._invalidation_reason is not None:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
                    reason=self._invalidation_reason,
                    occurred_at=now,
                )
            active = self._active_anchor_request_generation
            if active is not None:
                if generation == active:
                    return self._outcome(
                        ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
                        reason="anchor_generation_already_active",
                        occurred_at=now,
                    )
                if generation != active + 1:
                    raise ValueError(
                        "Anchor request generation must advance by exactly one."
                    )
            fenced = self._pending
            if fenced is not None:
                if len(self._fenced_attempt_by_id) >= 32:
                    oldest_id = min(
                        self._fenced_attempt_by_id,
                        key=lambda attempt_id: (
                            self._fenced_attempt_by_id[attempt_id]
                            .publication_sequence
                        ),
                    )
                    self._fenced_attempt_by_id.pop(oldest_id, None)
                self._fenced_attempt_by_id[fenced.attempt_id] = fenced
            self._pending = None
            self._latest_coalesced = None
            self._pending_retry_count = 0
            self._submission_in_progress = False
            self._last_evaluated_basis_key = None
            self._committed_state = empty_composition_allocation_state(
                self._committed_state.reducer_id,
                spec=self.allocation_spec,
            )
            self._active_anchor_request_generation = generation
            return self._outcome(
                ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
                reason="anchor_generation_advanced",
                occurred_at=now,
                attempt=fenced,
                decision=(
                    None if fenced is None else fenced.allocation_decision
                ),
            )

    def _validate_basis(self, basis: AllocationPublicationBasis) -> None:
        if not isinstance(basis, AllocationPublicationBasis):
            raise TypeError("basis must be AllocationPublicationBasis.")
        cut = basis.control_cut
        if basis.allocation_scope_id != self.allocation_scope_id:
            raise ValueError("Allocation basis belongs to another scope.")
        if cut.runtime_session_id != self.runtime_session_id:
            raise ValueError("Allocation basis belongs to another runtime session.")
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Allocation basis belongs to another control domain.")
        if cut.clock_reset_epoch != self.control_clock_reset_epoch:
            raise ValueError("Allocation basis belongs to another clock reset epoch.")
        if (
            basis.activation_scope.activation_id
            != self.activation_scope.activation_id
            or basis.activation_scope.semantic_content_hash
            != self.activation_scope.semantic_content_hash
        ):
            raise ValueError("Allocation basis belongs to another activation.")

    def _outcome(
        self,
        kind: str,
        *,
        reason: str,
        occurred_at: float,
        attempt: PendingAllocationPublication | None = None,
        decision: CompositionAllocationDecision | None = None,
        receipt_context: CompositionContextRecord | None = None,
        rejection_code: str | None = None,
    ) -> AllocationPublicationOutcome:
        return AllocationPublicationOutcome(
            outcome_kind=kind,
            allocation_scope_id=self.allocation_scope_id,
            runtime_status=self.runtime_status,
            occurred_at=occurred_at,
            committed_reducer_state_hash=(
                self._committed_state.semantic_content_hash
            ),
            reason=reason,
            decision_id=None if decision is None else decision.decision_id,
            decision_hash=(
                None if decision is None else decision.semantic_content_hash
            ),
            attempt_id=None if attempt is None else attempt.attempt_id,
            attempt_hash=(
                None if attempt is None else attempt.semantic_content_hash
            ),
            publication_sequence=(
                None if attempt is None else attempt.publication_sequence
            ),
            receipt_context_id=(
                None if receipt_context is None else receipt_context.context_id
            ),
            receipt_context_hash=(
                None
                if receipt_context is None
                else receipt_context.semantic_content_hash
            ),
            rejection_code=rejection_code,
        )

    def offer_basis(
        self,
        basis: AllocationPublicationBasis,
    ) -> AllocationPublicationOutcome:
        """Evaluate now or coalesce the latest basis behind one pending attempt."""

        self._validate_basis(basis)
        now = self._now()
        with self._lock:
            if self._invalidation_reason is not None:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
                    reason=self._invalidation_reason,
                    occurred_at=now,
                )
            basis_generation = basis.anchor_request_generation
            active_generation = self._active_anchor_request_generation
            if active_generation is None:
                self._active_anchor_request_generation = basis_generation
            elif basis_generation != active_generation:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED,
                    reason=(
                        "basis_anchor_generation_is_fenced"
                        if basis_generation < active_generation
                        else "basis_anchor_generation_not_admitted"
                    ),
                    occurred_at=now,
                )
            if self._pending is not None:
                if (
                    basis.ordering_key > self._pending.activity_basis.ordering_key
                    and (
                        self._latest_coalesced is None
                        or basis.ordering_key
                        > self._latest_coalesced.ordering_key
                    )
                ):
                    self._latest_coalesced = basis
                    kind = ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED
                    reason = "latest_basis_retained_behind_pending_attempt"
                else:
                    kind = ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED
                    reason = "basis_not_newer_than_latest_coalesced"
                return self._outcome(
                    kind,
                    reason=reason,
                    occurred_at=now,
                    attempt=self._pending,
                    decision=self._pending.allocation_decision,
                )
            if (
                self._last_evaluated_basis_key is not None
                and basis.ordering_key <= self._last_evaluated_basis_key
            ):
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED,
                    reason="basis_not_newer_than_last_evaluation",
                    occurred_at=now,
                )
            if now < basis.control_cut.cutoff_at:
                raise ValueError("Publication clock precedes the allocation cut.")
            self._last_evaluated_basis_key = basis.ordering_key
            effective_from = max(now, basis.requested_effective_from)
            if effective_from >= basis.support_until:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION,
                    reason="allocation_basis_support_expired",
                    occurred_at=now,
                )
            override_state = self._override_state
            override_active = (
                override_state is not None
                and any(
                    mode is not None
                    for mode in override_state.override_mode_by_component.values()
                )
            )
            override_valid_until = (
                None
                if not override_active
                or self.persistent_override_selection_lease_sec is None
                else effective_from
                + self.persistent_override_selection_lease_sec
            )
            evaluation = evaluate_composition_allocation(
                self._committed_state,
                spec=self.allocation_spec,
                activity_by_component=basis.activity_by_component,
                residual_magnitude_by_component=(
                    basis.residual_magnitude_by_component
                ),
                evaluated_at=now,
                clock_domain_id=basis.control_cut.clock_domain_id,
                override_state=override_state,
                persistent_override_valid_until=override_valid_until,
            )
            decision = evaluation.decision
            if decision.output_status == COMPOSITION_ALLOCATION_STATUS_NO_DECISION:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION,
                    reason=decision.reason,
                    occurred_at=now,
                    decision=decision,
                )
            if decision.output_status != COMPOSITION_ALLOCATION_STATUS_READY:
                raise AssertionError("Unhandled allocation decision status.")
            sequence = self._last_reserved_sequence + 1
            draft = build_composition_allocation_context_draft(
                decision,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                producer_id=self.producer_id,
                producer_epoch=self.producer_epoch,
                sequence=sequence,
                context_channel=self.context_channel,
                produced_at=now,
                produced_clock_domain_id=basis.control_cut.clock_domain_id,
                effective_from=effective_from,
                effective_clock_domain_id=basis.control_cut.clock_domain_id,
                support_until=(
                    basis.support_until
                    if override_valid_until is None
                    else max(basis.support_until, override_valid_until)
                ),
                allocation_scope_id=self.allocation_scope_id,
                activity_basis_id=basis.basis_id,
                activity_basis_hash=basis.semantic_content_hash,
                control_cut_hash=basis.control_cut_hash,
                activation_scope=self.activation_scope,
            )
            attempt = PendingAllocationPublication(
                allocation_scope_id=self.allocation_scope_id,
                publication_sequence=sequence,
                activity_basis=basis,
                previous_reducer_state_hash=(
                    self._committed_state.semantic_content_hash
                ),
                candidate_next_state=evaluation.state,
                allocation_decision=decision,
                context_draft=draft,
                created_at=now,
            )
            self._last_reserved_sequence = sequence
            self._pending = attempt
            self._pending_retry_count = 0
            return self._outcome(
                ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
                reason="immutable_publication_attempt_prepared",
                occurred_at=now,
                attempt=attempt,
                decision=decision,
            )

    def prepare_latest_coalesced(self) -> AllocationPublicationOutcome | None:
        """Prepare at most the single latest basis after pending terminates."""

        with self._lock:
            if self._pending is not None or self._invalidation_reason is not None:
                return None
            basis = self._latest_coalesced
            self._latest_coalesced = None
        return None if basis is None else self.offer_basis(basis)

    @staticmethod
    def _exact_record(
        attempt: PendingAllocationPublication,
        record: CompositionContextRecord | None,
    ) -> bool:
        return bool(
            record is not None
            and record.context_id == attempt.context_draft_id
            and composition_context_submission_semantic_hash(record)
            == attempt.context_draft_hash
        )

    def _commit(
        self,
        attempt: PendingAllocationPublication,
        *,
        occurred_at: float,
        record: CompositionContextRecord,
        existing: bool,
    ) -> AllocationPublicationOutcome:
        self._committed_state = attempt.candidate_next_state
        self._last_terminal_sequence = attempt.publication_sequence
        self._last_accepted_sequence = attempt.publication_sequence
        self._pending = None
        self._pending_retry_count = 0
        return self._outcome(
            (
                ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING
                if existing
                else ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
            ),
            reason=(
                "exact_context_fact_already_admitted"
                if existing
                else "context_admission_accepted"
            ),
            occurred_at=occurred_at,
            attempt=attempt,
            decision=attempt.allocation_decision,
            receipt_context=record,
        )

    def _unknown(
        self,
        attempt: PendingAllocationPublication,
        *,
        occurred_at: float,
        reason: str,
    ) -> AllocationPublicationOutcome:
        self._pending_retry_count += 1
        return self._outcome(
            ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN,
            reason=reason,
            occurred_at=occurred_at,
            attempt=attempt,
            decision=attempt.allocation_decision,
        )

    def resolve_admission(
        self,
        attempt_id: str,
        receipt: CompositionContextAdmissionReceipt,
        *,
        existing_context: CompositionContextRecord | None = None,
        occurred_at: float | None = None,
    ) -> AllocationPublicationOutcome:
        """Apply one terminal receipt or retain pending when evidence is unknown."""

        if not isinstance(receipt, CompositionContextAdmissionReceipt):
            raise TypeError("receipt must be CompositionContextAdmissionReceipt.")
        normalized_attempt_id = _required(attempt_id, name="attempt_id")
        now = self._now() if occurred_at is None else _finite(
            occurred_at,
            name="occurred_at",
        )
        with self._lock:
            override_fenced = self._override_fenced_attempt_by_id.pop(
                normalized_attempt_id,
                None,
            )
            if override_fenced is not None:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_LATE,
                    reason="receipt_belongs_to_superseded_allocation_override",
                    occurred_at=now,
                    attempt=override_fenced,
                    decision=override_fenced.allocation_decision,
                )
            fenced = self._fenced_attempt_by_id.pop(
                normalized_attempt_id,
                None,
            )
            if fenced is not None:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_LATE,
                    reason="receipt_belongs_to_fenced_anchor_generation",
                    occurred_at=now,
                    attempt=fenced,
                    decision=fenced.allocation_decision,
                )
            if self._invalidation_reason is not None:
                orphan = self._invalidated_pending
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_LATE,
                    reason="receipt_belongs_to_invalidated_activation",
                    occurred_at=now,
                    attempt=(
                        orphan
                        if orphan is not None
                        and orphan.attempt_id == normalized_attempt_id
                        else None
                    ),
                    decision=(
                        orphan.allocation_decision
                        if orphan is not None
                        and orphan.attempt_id == normalized_attempt_id
                        else None
                    ),
                )
            attempt = self._pending
            if attempt is None or attempt.attempt_id != normalized_attempt_id:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_LATE,
                    reason="receipt_does_not_match_current_pending_attempt",
                    occurred_at=now,
                )
            if receipt.accepted:
                record = receipt.context
                if not self._exact_record(attempt, record):
                    return self._unknown(
                        attempt,
                        occurred_at=now,
                        reason="accepted_receipt_cannot_be_reconciled_exactly",
                    )
                assert record is not None
                if (
                    receipt.context_id != record.context_id
                    or receipt.semantic_content_hash
                    != record.semantic_content_hash
                ):
                    return self._unknown(
                        attempt,
                        occurred_at=now,
                        reason="accepted_receipt_identity_is_incomplete_or_conflicting",
                    )
                return self._commit(
                    attempt,
                    occurred_at=now,
                    record=record,
                    existing=False,
                )
            if receipt.rejection_code == "duplicate":
                if self._exact_record(attempt, existing_context):
                    assert existing_context is not None
                    return self._commit(
                        attempt,
                        occurred_at=now,
                        record=existing_context,
                        existing=True,
                    )
                if existing_context is None:
                    return self._unknown(
                        attempt,
                        occurred_at=now,
                        reason="duplicate_receipt_requires_exact_record_reconciliation",
                    )
                self._last_terminal_sequence = attempt.publication_sequence
                self._pending = None
                self._pending_retry_count = 0
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
                    reason="duplicate_context_identity_conflict_or_unreconciled",
                    occurred_at=now,
                    attempt=attempt,
                    decision=attempt.allocation_decision,
                    receipt_context=existing_context,
                    rejection_code=receipt.rejection_code,
                )
            self._last_terminal_sequence = attempt.publication_sequence
            self._pending = None
            self._pending_retry_count = 0
            return self._outcome(
                ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
                reason=receipt.message or "context_admission_terminally_rejected",
                occurred_at=now,
                attempt=attempt,
                decision=attempt.allocation_decision,
                rejection_code=receipt.rejection_code,
            )

    def attempt_admission(
        self,
        admit_context: ContextAdmitter,
        *,
        resolve_context: ContextResolver | None = None,
    ) -> AllocationPublicationOutcome | None:
        """Submit the exact pending draft without holding the runtime lock.

        The supplied admission callback may itself block.  Applications must
        call this convenience from a non-control-loop worker, or transport the
        snapshot's pending draft asynchronously and later call
        :meth:`resolve_admission` with its receipt.
        """

        if not callable(admit_context):
            raise TypeError("admit_context must be callable.")
        if resolve_context is not None and not callable(resolve_context):
            raise TypeError("resolve_context must be callable.")
        attempted_at = self._now()
        with self._lock:
            if self._invalidation_reason is not None or self._pending is None:
                return None
            if self._submission_in_progress:
                return self._outcome(
                    ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN,
                    reason="admission_submission_already_in_progress",
                    occurred_at=attempted_at,
                    attempt=self._pending,
                    decision=self._pending.allocation_decision,
                )
            attempt = self._pending
            self._last_attempt_at = attempted_at
            self._submission_in_progress = True
        try:
            receipt = admit_context(attempt.context_draft)
            existing = None
            if (
                isinstance(receipt, CompositionContextAdmissionReceipt)
                and not receipt.accepted
                and receipt.rejection_code == "duplicate"
                and resolve_context is not None
            ):
                existing = resolve_context(attempt.context_draft_id)
        except Exception as exc:  # outcome is intentionally treated as unknown.
            finished_at = self._now()
            with self._lock:
                self._submission_in_progress = False
                if self._invalidation_reason is not None:
                    return self._outcome(
                        ALLOCATION_PUBLICATION_OUTCOME_LATE,
                        reason="admission_exception_after_invalidation",
                        occurred_at=finished_at,
                        attempt=self._invalidated_pending,
                        decision=(
                            None
                            if self._invalidated_pending is None
                            else self._invalidated_pending.allocation_decision
                        ),
                    )
                if self._pending is not attempt:
                    return self._outcome(
                        ALLOCATION_PUBLICATION_OUTCOME_LATE,
                        reason="admission_exception_after_attempt_finalized",
                        occurred_at=finished_at,
                    )
                return self._unknown(
                    attempt,
                    occurred_at=finished_at,
                    reason=f"admission_outcome_unknown:{type(exc).__name__}",
                )
        finished_at = self._now()
        with self._lock:
            self._submission_in_progress = False
            if not isinstance(receipt, CompositionContextAdmissionReceipt):
                if self._invalidation_reason is not None:
                    return self._outcome(
                        ALLOCATION_PUBLICATION_OUTCOME_LATE,
                        reason="invalid_receipt_after_invalidation",
                        occurred_at=finished_at,
                        attempt=self._invalidated_pending,
                    )
                if self._pending is attempt:
                    return self._unknown(
                        attempt,
                        occurred_at=finished_at,
                        reason="admission_returned_invalid_receipt",
                    )
        return self.resolve_admission(
            attempt.attempt_id,
            receipt,
            existing_context=existing,
            occurred_at=finished_at,
        )

    def invalidate(
        self,
        *,
        reason: str,
    ) -> AllocationPublicationOutcome:
        """Fence this activation permanently; a new activation needs a new runtime."""

        normalized_reason = _required(reason, name="reason")
        now = self._now()
        with self._lock:
            if self._invalidation_reason is None:
                self._invalidation_reason = normalized_reason
                self._invalidated_pending = self._pending
                self._pending = None
                self._latest_coalesced = None
                self._submission_in_progress = False
            return self._outcome(
                ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
                reason=self._invalidation_reason,
                occurred_at=now,
                attempt=self._invalidated_pending,
                decision=(
                    None
                    if self._invalidated_pending is None
                    else self._invalidated_pending.allocation_decision
                ),
            )

__all__ = [
    "ALLOCATION_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_ATTEMPT_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_OUTCOME_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_RUNTIME_READY",
    "ALLOCATION_PUBLICATION_RUNTIME_PENDING",
    "ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED",
    "ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY",
    "ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION",
    "ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED",
    "ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED",
    "ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW",
    "ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING",
    "ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION",
    "ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN",
    "ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED",
    "ALLOCATION_PUBLICATION_OUTCOME_LATE",
    "ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED",
    "AllocationPublicationBasis",
    "AllocationPublicationOutcome",
    "AllocationPublicationRuntime",
    "AllocationPublicationRuntimeSnapshot",
    "ContextAdmitter",
    "ContextResolver",
    "PendingAllocationPublication",
    "allocation_publication_control_cut_semantic_hash",
]
