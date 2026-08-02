"""Durable request-to-fact controller for collection progress."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from dataset_contracts.progress_failure_triggers import (
    FailureTriggerCaptureRecordV1,
)
from dataset_contracts.progress_facts import (
    AttemptFailureEventV1,
    AttemptRetryEventV1,
    EpisodeProgressStartedV1,
    EpisodeProgressTerminationV1,
    HoldPauseEffectReceiptV1,
    ProgressAdvanceEventV1,
    ProgressFactEnvelopeV1,
    ProgressFactPayloadV1,
    ProgressUndoEventV1,
    ResumeAfterFailureReceiptV1,
)

from .reducer import (
    ProgressReductionError,
    ProgressState,
    reduce_progress_fact,
)
from .failure_trigger_capture_store import (
    FailureTriggerCaptureStoreError,
    record_failure_trigger_capture,
)
from .requests import (
    AbortEpisodeProgressRequest,
    AdvanceProgressStepRequest,
    CompleteEpisodeProgressRequest,
    ProgressRequestContext,
    ReportAttemptFailureRequest,
    ResumeFailedAttemptRequest,
    StartEpisodeProgressRequest,
    UndoLastProgressAdvanceRequest,
)
from .store import ProgressFactStore, ProgressSealReport


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEffectRequest:
    """Application-facing effect request; the durable Failure is its outbox."""

    idempotency_key: str
    source_failure_fact_id: str
    requested_operations: tuple[str, ...]


@dataclass(frozen=True)
class ProgressControllerResult:
    fact: ProgressFactEnvelopeV1
    state: ProgressState
    effect_requests: tuple[ProgressEffectRequest, ...] = ()
    idempotent: bool = False


class ProgressController:
    """Own one canonical reducer state over one durable progress store."""

    def __init__(
        self,
        *,
        store: ProgressFactStore,
        allow_rooted_tree: bool = False,
    ) -> None:
        if not isinstance(store, ProgressFactStore):
            raise TypeError("store must be ProgressFactStore.")
        if not store.fsync_on_append:
            raise ValueError(
                "ProgressController requires fsync_on_append so Failure facts "
                "are durable before effects are published."
            )
        if store.profile.topology_kind == "ROOTED_TREE" and not allow_rooted_tree:
            raise ValueError(
                "ROOTED_TREE runtime is gated; pass allow_rooted_tree=True only "
                "for an explicit offline/approved adapter."
            )
        self.store = store
        self.profile = store.profile
        self.binding = store.binding
        self.state = store.replay_state()

    def start(
        self,
        request: StartEpisodeProgressRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        return self._append(
            EpisodeProgressStartedV1(
                progress_step_id=self.binding.entry_step_id,
                attempt_id=request.attempt_id,
                attempt_ordinal_in_capture=1,
            ),
            context=context,
        )

    def advance(
        self,
        request: AdvanceProgressStepRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        existing = self._existing_for_context(context)
        if existing is not None:
            payload = existing.payload
            target = request.target_progress_step_id
            if isinstance(payload, ProgressAdvanceEventV1) and target is None:
                allowed = self.profile.allowed_next_step_ids(
                    payload.from_progress_step_id
                )
                target = allowed[0] if len(allowed) == 1 else None
            if (
                not isinstance(payload, ProgressAdvanceEventV1)
                or payload.to_attempt_id != request.to_attempt_id
                or target is None
                or payload.to_progress_step_id != target
            ):
                raise ProgressReductionError(
                    "Repeated Advance request content changed."
                )
            return self._existing_result(existing)
        if (
            self.state.phase != "ACTIVE_ATTEMPT"
            or self.state.current_progress_step_id is None
            or self.state.active_attempt_id is None
            or self.state.active_attempt_start_fact_id is None
        ):
            raise ProgressReductionError("Advance requires an active Attempt.")
        allowed = self.profile.allowed_next_step_ids(
            self.state.current_progress_step_id
        )
        if not allowed:
            raise ProgressReductionError("The current progress step is terminal.")
        target = request.target_progress_step_id
        if target is None:
            if len(allowed) != 1:
                raise ProgressReductionError(
                    "A branch point requires target_progress_step_id."
                )
            target = allowed[0]
        if target not in allowed:
            raise ProgressReductionError("Requested progress target is not legal.")
        return self._append(
            ProgressAdvanceEventV1(
                from_progress_step_id=self.state.current_progress_step_id,
                from_attempt_id=self.state.active_attempt_id,
                to_progress_step_id=target,
                to_attempt_id=request.to_attempt_id,
            ),
            context=context,
            source_fact_ids=(self.state.active_attempt_start_fact_id,),
        )

    def undo_last_advance(
        self,
        request: UndoLastProgressAdvanceRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        return self._append(
            ProgressUndoEventV1(advance_fact_id=request.advance_fact_id),
            context=context,
            source_fact_ids=(request.advance_fact_id,),
        )

    def report_failure(
        self,
        request: ReportAttemptFailureRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        existing = self._existing_for_context(context)
        if existing is not None:
            payload = existing.payload
            if not isinstance(payload, AttemptFailureEventV1):
                raise ProgressReductionError(
                    "Repeated Failure request content changed."
                )
            result = self._existing_result(existing)
            if request.capture_failure_trigger:
                self._capture_failure_trigger_receipt(result.fact)
            return ProgressControllerResult(
                fact=result.fact,
                state=result.state,
                effect_requests=self.pending_effect_requests(),
                idempotent=True,
            )
        if (
            self.state.phase != "ACTIVE_ATTEMPT"
            or self.state.current_progress_step_id is None
            or self.state.active_attempt_id is None
            or self.state.active_attempt_start_fact_id is None
        ):
            raise ProgressReductionError("Failure requires an active Attempt.")
        result = self._append(
            AttemptFailureEventV1(
                progress_step_id=self.state.current_progress_step_id,
                attempt_id=self.state.active_attempt_id,
                reported_at=context.effective_at,
                timing_semantics=request.timing_semantics,
            ),
            context=context,
            source_fact_ids=(self.state.active_attempt_start_fact_id,),
        )
        if request.capture_failure_trigger:
            self._capture_failure_trigger_receipt(result.fact)
        return ProgressControllerResult(
            fact=result.fact,
            state=result.state,
            effect_requests=self.pending_effect_requests(),
            idempotent=result.idempotent,
        )

    def record_hold_pause_receipt(
        self,
        receipt: HoldPauseEffectReceiptV1,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        result = self._append(
            receipt,
            context=context,
            source_fact_ids=(receipt.source_failure_fact_id,),
        )
        return ProgressControllerResult(
            fact=result.fact,
            state=result.state,
            effect_requests=self.pending_effect_requests(),
            idempotent=result.idempotent,
        )

    def record_resume_receipt(
        self,
        receipt: ResumeAfterFailureReceiptV1,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        return self._append(
            receipt,
            context=context,
            source_fact_ids=(
                receipt.source_failure_fact_id,
                receipt.hold_pause_receipt_fact_id,
            ),
        )

    def resume_failed_attempt(
        self,
        request: ResumeFailedAttemptRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        existing = self._existing_for_context(context)
        if existing is not None:
            payload = existing.payload
            if (
                not isinstance(payload, AttemptRetryEventV1)
                or payload.new_attempt_id != request.new_attempt_id
            ):
                raise ProgressReductionError(
                    "Repeated Retry request content changed."
                )
            return self._existing_result(existing)
        if (
            self.state.phase != "RESUME_RECEIVED"
            or self.state.pending_failure_fact_id is None
            or self.state.hold_pause_receipt_fact_id is None
            or self.state.resume_receipt_fact_id is None
            or self.state.current_progress_step_id is None
        ):
            raise ProgressReductionError(
                "Retry requires an applied Resume receipt."
            )
        resume_fact = self.store.fact_by_id(self.state.resume_receipt_fact_id)
        if (
            resume_fact is None
            or not isinstance(
                resume_fact.payload,
                ResumeAfterFailureReceiptV1,
            )
        ):
            raise ProgressReductionError("Applied Resume receipt is unavailable.")
        resumed_at = resume_fact.payload.formal_capture_resume.effective_at
        resumed_clock = resume_fact.payload.formal_capture_resume.clock_domain_id
        if resumed_at is None:
            raise ProgressReductionError(
                "Applied formal capture Resume has no effective time."
            )
        if context.effective_clock_domain_id != resumed_clock:
            raise ProgressReductionError(
                "Retry and formal capture Resume must use the bound Episode clock."
            )
        if context.effective_at < resumed_at:
            raise ProgressReductionError(
                "Retry cannot begin before formal capture Resume."
            )
        return self._append(
            AttemptRetryEventV1(
                source_failure_fact_id=self.state.pending_failure_fact_id,
                hold_pause_receipt_fact_id=(
                    self.state.hold_pause_receipt_fact_id
                ),
                resume_receipt_fact_id=self.state.resume_receipt_fact_id,
                progress_step_id=self.state.current_progress_step_id,
                new_attempt_id=request.new_attempt_id,
                attempt_ordinal_in_capture=(
                    self.state.latest_attempt_ordinal_in_capture + 1
                ),
            ),
            context=context,
            source_fact_ids=(
                self.state.pending_failure_fact_id,
                self.state.hold_pause_receipt_fact_id,
                self.state.resume_receipt_fact_id,
            ),
        )

    def complete(
        self,
        request: CompleteEpisodeProgressRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        existing = self._existing_for_context(context)
        if existing is not None:
            payload = existing.payload
            expected_disposition = (
                "SUCCESS"
                if self.binding.recording_scope == "FULL_TASK"
                or request.completed_progress_step
                else "TRUNCATED_AT_CAPTURE_END"
            )
            if (
                not isinstance(payload, EpisodeProgressTerminationV1)
                or payload.active_attempt_disposition
                != expected_disposition
            ):
                raise ProgressReductionError(
                    "Repeated completion request content changed."
                )
            return self._existing_result(existing)
        if self.state.phase != "ACTIVE_ATTEMPT":
            raise ProgressReductionError(
                "Normal progress completion requires an active Attempt."
            )
        if self.binding.recording_scope == "FULL_TASK":
            payload = EpisodeProgressTerminationV1(
                termination_cause="FULL_TASK_COMPLETED",
                active_attempt_disposition="SUCCESS",
                full_task_outcome="SUCCESS",
            )
        else:
            payload = EpisodeProgressTerminationV1(
                termination_cause="WINDOW_COMPLETED",
                active_attempt_disposition=(
                    "SUCCESS"
                    if request.completed_progress_step
                    else "TRUNCATED_AT_CAPTURE_END"
                ),
                full_task_outcome="NOT_APPLICABLE",
            )
        sources = (
            ()
            if self.state.active_attempt_start_fact_id is None
            else (self.state.active_attempt_start_fact_id,)
        )
        return self._append(
            payload,
            context=context,
            source_fact_ids=sources,
        )

    def abort(
        self,
        request: AbortEpisodeProgressRequest,
        *,
        context: ProgressRequestContext,
    ) -> ProgressControllerResult:
        existing = self._existing_for_context(context)
        if existing is not None:
            payload = existing.payload
            if (
                not isinstance(payload, EpisodeProgressTerminationV1)
                or payload.termination_cause != request.termination_cause
            ):
                raise ProgressReductionError(
                    "Repeated Abort request content changed."
                )
            return self._existing_result(existing)
        if request.termination_cause in {
            "FULL_TASK_COMPLETED",
            "WINDOW_COMPLETED",
        }:
            raise ProgressReductionError(
                "Abort cannot use a normal completion cause."
            )
        active = self.state.phase == "ACTIVE_ATTEMPT"
        unresolved_failures = (
            (self.state.pending_failure_fact_id,)
            if self.state.phase == "AWAITING_HOLD_RECEIPT"
            and self.state.pending_failure_fact_id is not None
            else ()
        )
        sources = tuple(
            value
            for value in (
                self.state.active_attempt_start_fact_id,
                self.state.pending_failure_fact_id,
                self.state.hold_pause_receipt_fact_id,
                self.state.resume_receipt_fact_id,
            )
            if value is not None
        )
        return self._append(
            EpisodeProgressTerminationV1(
                termination_cause=request.termination_cause,
                active_attempt_disposition=(
                    "ABORTED" if active else "NO_ACTIVE_ATTEMPT"
                ),
                full_task_outcome=(
                    "UNKNOWN"
                    if self.binding.recording_scope == "FULL_TASK"
                    else "NOT_APPLICABLE"
                ),
                unresolved_source_failure_fact_ids=unresolved_failures,
            ),
            context=context,
            source_fact_ids=sources,
        )

    def pending_effect_requests(self) -> tuple[ProgressEffectRequest, ...]:
        if (
            self.state.phase != "AWAITING_HOLD_RECEIPT"
            or self.state.pending_failure_fact_id is None
        ):
            return ()
        failure = self.store.fact_by_id(self.state.pending_failure_fact_id)
        if failure is None or not isinstance(
            failure.payload,
            AttemptFailureEventV1,
        ):
            raise ProgressReductionError("Pending Failure outbox is unavailable.")
        return (
            ProgressEffectRequest(
                idempotency_key=failure.fact_id,
                source_failure_fact_id=failure.fact_id,
                requested_operations=failure.payload.requested_operations,
            ),
        )

    def seal(
        self,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> ProgressSealReport:
        return self.store.seal(
            sealed_at=sealed_at,
            sealed_clock_domain_id=sealed_clock_domain_id,
        )

    def _append(
        self,
        payload: ProgressFactPayloadV1,
        *,
        context: ProgressRequestContext,
        source_fact_ids: tuple[str, ...] = (),
    ) -> ProgressControllerResult:
        existing = self.store.fact_by_id(context.request_id)
        if existing is not None:
            probe = ProgressFactEnvelopeV1(
                fact_id=context.request_id,
                episode_capture_identity_hash=(
                    self.binding.episode_capture_identity_hash
                ),
                progress_sequence_index=existing.progress_sequence_index,
                effective_at=context.effective_at,
                effective_clock_domain_id=context.effective_clock_domain_id,
                available_at=context.available_at,
                available_clock_domain_id=context.available_clock_domain_id,
                origin=context.origin,
                actor_id=context.actor_id,
                payload=payload,
                source_fact_ids=source_fact_ids,
            )
            if probe.semantic_content_hash != existing.semantic_content_hash:
                raise ProgressReductionError(
                    f"Request ID {context.request_id!r} was reused with new content."
                )
            return ProgressControllerResult(
                fact=existing,
                state=self.state,
                idempotent=True,
            )
        fact = ProgressFactEnvelopeV1(
            fact_id=context.request_id,
            episode_capture_identity_hash=(
                self.binding.episode_capture_identity_hash
            ),
            progress_sequence_index=self.store.next_sequence_index,
            effective_at=context.effective_at,
            effective_clock_domain_id=context.effective_clock_domain_id,
            available_at=context.available_at,
            available_clock_domain_id=context.available_clock_domain_id,
            origin=context.origin,
            actor_id=context.actor_id,
            payload=payload,
            source_fact_ids=source_fact_ids,
        )
        next_state = reduce_progress_fact(
            profile=self.profile,
            binding=self.binding,
            state=self.state,
            fact=fact,
        )
        append_result = self.store.append_fact(fact)
        if not append_result.durable:
            raise ProgressReductionError(
                "Progress fact was accepted without durability."
            )
        self.state = next_state
        return ProgressControllerResult(fact=fact, state=self.state)

    def _existing_for_context(
        self,
        context: ProgressRequestContext,
    ) -> ProgressFactEnvelopeV1 | None:
        existing = self.store.fact_by_id(context.request_id)
        if existing is None:
            return None
        if (
            existing.effective_at != context.effective_at
            or existing.effective_clock_domain_id
            != context.effective_clock_domain_id
            or existing.available_at != context.available_at
            or existing.available_clock_domain_id
            != context.available_clock_domain_id
            or existing.actor_id != context.actor_id
            or existing.origin != context.origin
        ):
            raise ProgressReductionError(
                f"Request ID {context.request_id!r} was reused with new context."
            )
        return existing

    def _existing_result(
        self,
        fact: ProgressFactEnvelopeV1,
    ) -> ProgressControllerResult:
        return ProgressControllerResult(
            fact=fact,
            state=self.state,
            idempotent=True,
        )

    def _capture_failure_trigger_receipt(
        self,
        fact: ProgressFactEnvelopeV1,
    ) -> None:
        """Best-effort additive capture provenance after durable Failure."""

        payload = fact.payload
        if not isinstance(payload, AttemptFailureEventV1):
            return
        record = FailureTriggerCaptureRecordV1(
            source_failure_fact_id=fact.fact_id,
            coarse_attempt_id=payload.attempt_id,
            progress_step_id=payload.progress_step_id,
            recorded_at=payload.reported_at,
            clock_domain_id=fact.effective_clock_domain_id,
        )
        try:
            record_failure_trigger_capture(
                episode_dir=self.store.episode_dir,
                record=record,
            )
        except (
            FailureTriggerCaptureStoreError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            # The existing Failure fact is already durable and authoritative.
            # A receipt write must never make the operator's failure path fail.
            _LOGGER.warning(
                "Could not persist optional failure-trigger capture receipt "
                "for %s: %s",
                fact.fact_id,
                exc,
            )


__all__ = [
    "ProgressController",
    "ProgressControllerResult",
    "ProgressEffectRequest",
]
