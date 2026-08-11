"""Read-only canonical timeline resolution for collection-progress facts.

The values in this module are derived views, not persisted dataset contracts.
They intentionally do not import ``collection_runtime``: read-time validation
must be independently rebuildable from pure contracts and sealed facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dataset_contracts.progress_facts import (
    AttemptFailureEventV1,
    AttemptRetryEventV1,
    EpisodeProgressStartedV1,
    EpisodeProgressTerminationV1,
    HoldPauseEffectReceiptV1,
    ProgressAdvanceEventV1,
    ProgressEpisodeManifestV1,
    ProgressFactEnvelopeV1,
    ProgressOperationReceiptV1,
    ProgressUndoEventV1,
    ResumeAfterFailureReceiptV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
    validate_progress_episode_binding,
)


class ProgressTimelineResolutionError(ValueError):
    """Raised when capture facts cannot produce one unambiguous timeline."""


@dataclass(frozen=True)
class ResolvedProgressAttempt:
    attempt_id: str
    progress_step_id: str
    attempt_ordinal_in_capture: int
    start_fact_id: str
    start_at: float
    clock_domain_id: str
    end_fact_id: str | None
    end_at: float | None
    outcome: str
    failure_fact_id: str | None = None
    failure_reported_at: float | None = None
    failure_timing_semantics: str | None = None


@dataclass(frozen=True)
class ResolvedProgressTransition:
    advance_fact_id: str
    from_progress_step_id: str
    from_attempt_id: str
    to_progress_step_id: str
    to_attempt_id: str
    effective_at: float
    clock_domain_id: str


@dataclass(frozen=True)
class ResolvedCapturePauseInterval:
    source_failure_fact_id: str
    pause_receipt_fact_id: str
    start_at: float
    clock_domain_id: str
    resume_receipt_fact_id: str | None = None
    end_at: float | None = None

    @property
    def closed(self) -> bool:
        return self.end_at is not None


@dataclass(frozen=True)
class ResolvedProgressTimeline:
    episode_capture_identity_hash: str
    task_progress_profile_hash: str
    progress_episode_binding_hash: str
    clock_domain_id: str
    fact_bundle_hash: str
    fact_count: int
    sealed: bool
    phase: str
    attempts: tuple[ResolvedProgressAttempt, ...]
    transitions: tuple[ResolvedProgressTransition, ...]
    capture_pause_intervals: tuple[ResolvedCapturePauseInterval, ...]
    undone_advance_fact_ids: tuple[str, ...]
    invalidated_attempt_ids: tuple[str, ...]
    current_progress_step_id: str | None
    active_attempt_id: str | None
    terminal_fact_id: str | None
    termination_cause: str | None
    full_task_outcome: str | None
    resolution_kind: str = "CAPTURE_CAUSAL"
    source_progress_manifest_hash: str | None = None


@dataclass
class _AttemptBuilder:
    attempt_id: str
    progress_step_id: str
    ordinal: int
    start_fact_id: str
    start_at: float
    clock_domain_id: str
    end_fact_id: str | None = None
    end_at: float | None = None
    outcome: str = "ACTIVE"
    failure_fact_id: str | None = None
    failure_reported_at: float | None = None
    failure_timing_semantics: str | None = None

    def materialize(self) -> ResolvedProgressAttempt:
        return ResolvedProgressAttempt(
            attempt_id=self.attempt_id,
            progress_step_id=self.progress_step_id,
            attempt_ordinal_in_capture=self.ordinal,
            start_fact_id=self.start_fact_id,
            start_at=self.start_at,
            clock_domain_id=self.clock_domain_id,
            end_fact_id=self.end_fact_id,
            end_at=self.end_at,
            outcome=self.outcome,
            failure_fact_id=self.failure_fact_id,
            failure_reported_at=self.failure_reported_at,
            failure_timing_semantics=self.failure_timing_semantics,
        )


@dataclass
class _PauseBuilder:
    source_failure_fact_id: str
    pause_receipt_fact_id: str
    start_at: float
    clock_domain_id: str
    resume_receipt_fact_id: str | None = None
    end_at: float | None = None

    def materialize(self) -> ResolvedCapturePauseInterval:
        return ResolvedCapturePauseInterval(
            source_failure_fact_id=self.source_failure_fact_id,
            pause_receipt_fact_id=self.pause_receipt_fact_id,
            start_at=self.start_at,
            clock_domain_id=self.clock_domain_id,
            resume_receipt_fact_id=self.resume_receipt_fact_id,
            end_at=self.end_at,
        )


@dataclass
class _UndoCandidate:
    advance_fact_id: str
    from_attempt_id: str
    to_attempt_id: str
    transition_index: int
    restored_ordinal: int


def _require_not_before(
    *,
    later_at: float,
    later_clock: str,
    earlier_at: float,
    earlier_clock: str,
    message: str,
) -> None:
    if later_clock == earlier_clock and later_at < earlier_at:
        raise ProgressTimelineResolutionError(message)


def _require_sources(
    fact: ProgressFactEnvelopeV1,
    required: tuple[str, ...],
) -> None:
    missing = set(required) - set(fact.source_fact_ids)
    if missing:
        raise ProgressTimelineResolutionError(
            f"Progress fact {fact.fact_id!r} omits required sources "
            f"{sorted(missing)}."
        )


def _validate_operation_availability(
    *,
    fact: ProgressFactEnvelopeV1,
    operation: ProgressOperationReceiptV1,
) -> None:
    for at, label in (
        (operation.requested_at, "requested"),
        (operation.effective_at, "effective"),
    ):
        if at is None:
            continue
        _require_not_before(
            later_at=fact.available_at,
            later_clock=fact.available_clock_domain_id,
            earlier_at=at,
            earlier_clock=operation.clock_domain_id,
            message=(
                f"Operation {operation.operation} {label} time is after its "
                f"receipt fact {fact.fact_id!r} became available."
            ),
        )


def _validate_termination(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    phase: str,
    current_progress_step_id: str | None,
    payload: EpisodeProgressTerminationV1,
    pending_failure_fact_id: str | None,
) -> None:
    active = phase == "ACTIVE_ATTEMPT"
    cause = payload.termination_cause
    disposition = payload.active_attempt_disposition
    outcome = payload.full_task_outcome
    if cause == "FULL_TASK_COMPLETED":
        if (
            binding.recording_scope != "FULL_TASK"
            or not active
            or current_progress_step_id not in profile.terminal_step_ids
            or disposition != "SUCCESS"
            or outcome != "SUCCESS"
        ):
            raise ProgressTimelineResolutionError(
                "FULL_TASK_COMPLETED requires an active terminal Attempt and "
                "full-task SUCCESS."
            )
    elif cause == "WINDOW_COMPLETED":
        if binding.recording_scope == "FULL_TASK":
            raise ProgressTimelineResolutionError(
                "FULL_TASK cannot terminate as WINDOW_COMPLETED."
            )
        if outcome != "NOT_APPLICABLE":
            raise ProgressTimelineResolutionError(
                "A recording window cannot claim a full-task outcome."
            )
        if active and disposition not in {
            "SUCCESS",
            "TRUNCATED_AT_CAPTURE_END",
        }:
            raise ProgressTimelineResolutionError(
                "An active window must end as SUCCESS or TRUNCATED."
            )
        if not active and disposition != "NO_ACTIVE_ATTEMPT":
            raise ProgressTimelineResolutionError(
                "A window without an active Attempt requires "
                "NO_ACTIVE_ATTEMPT."
            )
    else:
        if active and disposition not in {
            "ABORTED",
            "TRUNCATED_AT_CAPTURE_END",
            "UNKNOWN",
        }:
            raise ProgressTimelineResolutionError(
                "Abnormal termination cannot mark an active Attempt successful."
            )
        if not active and disposition != "NO_ACTIVE_ATTEMPT":
            raise ProgressTimelineResolutionError(
                "Termination without an active Attempt requires "
                "NO_ACTIVE_ATTEMPT."
            )
        if binding.recording_scope != "FULL_TASK" and outcome != "NOT_APPLICABLE":
            raise ProgressTimelineResolutionError(
                "A partial recording cannot claim a full-task outcome."
            )
        if (
            binding.recording_scope == "FULL_TASK"
            and outcome not in {"FAILURE", "UNKNOWN"}
        ):
            raise ProgressTimelineResolutionError(
                "An abnormal full-task termination must use FAILURE or UNKNOWN."
            )
    expected_unresolved = (
        (pending_failure_fact_id,)
        if phase == "AWAITING_HOLD_RECEIPT"
        and pending_failure_fact_id is not None
        else ()
    )
    if payload.unresolved_source_failure_fact_ids != expected_unresolved:
        raise ProgressTimelineResolutionError(
            "Termination does not exactly explain unresolved Failure effects."
        )
    if payload.unresolved_resume_operation_ids:
        raise ProgressTimelineResolutionError(
            "The current capture-fact model has no unresolved Resume operation."
        )


def _verify_manifest(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    facts: tuple[ProgressFactEnvelopeV1, ...],
    manifest: ProgressEpisodeManifestV1,
    terminal_fact_id: str | None,
) -> None:
    if (
        manifest.episode_capture_identity_hash
        != binding.episode_capture_identity_hash
        or manifest.task_progress_profile_hash != profile.semantic_content_hash
        or manifest.progress_episode_binding_hash
        != binding.semantic_content_hash
        or manifest.fact_count != len(facts)
        or manifest.fact_bundle_hash
        != ProgressEpisodeManifestV1.fact_bundle_hash_for(facts)
        or manifest.terminal_fact_id != terminal_fact_id
        or manifest.sealed_clock_domain_id != binding.bound_clock_domain_id
        or not facts
        or manifest.sealed_at < facts[-1].effective_at
    ):
        raise ProgressTimelineResolutionError(
            "Progress manifest does not bind the resolved fact timeline."
        )


def resolve_progress_timeline(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    facts: Iterable[ProgressFactEnvelopeV1],
    manifest: ProgressEpisodeManifestV1 | None = None,
    require_sealed: bool = False,
) -> ResolvedProgressTimeline:
    """Resolve one canonical capture-time timeline.

    Undo is applied retrospectively: the transient target Attempt and its
    implicit source success disappear from ``attempts`` while the immutable
    Undo and Advance facts remain auditable through their IDs.
    """

    validate_progress_episode_binding(profile=profile, binding=binding)
    ordered = tuple(facts)
    if require_sealed and manifest is None:
        raise ProgressTimelineResolutionError(
            "A sealed progress manifest is required."
        )

    phase = "NOT_STARTED"
    effective_clock_domain_id = binding.bound_clock_domain_id
    current_progress_step_id: str | None = None
    active_attempt_id: str | None = None
    latest_ordinal = 0
    pending_failure_fact_id: str | None = None
    applied_hold_pause_receipt_fact_id: str | None = None
    applied_resume_receipt_fact_id: str | None = None
    terminal_fact_id: str | None = None
    termination_cause: str | None = None
    full_task_outcome: str | None = None
    fact_ids: list[str] = []
    fact_by_id: dict[str, ProgressFactEnvelopeV1] = {}
    seen_attempt_ids: set[str] = set()
    attempt_order: list[str] = []
    attempt_by_id: dict[str, _AttemptBuilder] = {}
    transitions: list[ResolvedProgressTransition] = []
    undone_advance_fact_ids: list[str] = []
    invalidated_attempt_ids: list[str] = []
    pause_by_failure: dict[str, _PauseBuilder] = {}
    pause_order: list[str] = []
    undo_candidate: _UndoCandidate | None = None
    latest_effective_at: float | None = None

    for sequence, fact in enumerate(ordered, start=1):
        if not isinstance(fact, ProgressFactEnvelopeV1):
            raise TypeError("facts must contain ProgressFactEnvelopeV1 values.")
        if phase == "TERMINATED":
            raise ProgressTimelineResolutionError(
                "No progress fact may follow termination."
            )
        if (
            fact.episode_capture_identity_hash
            != binding.episode_capture_identity_hash
        ):
            raise ProgressTimelineResolutionError(
                "Progress fact Episode identity mismatch."
            )
        if fact.progress_sequence_index != sequence:
            raise ProgressTimelineResolutionError(
                "progress_sequence_index must be contiguous."
            )
        if fact.fact_id in fact_by_id:
            raise ProgressTimelineResolutionError(
                f"Duplicate progress fact ID {fact.fact_id!r}."
            )
        unknown_sources = set(fact.source_fact_ids) - set(fact_ids)
        if unknown_sources:
            raise ProgressTimelineResolutionError(
                f"Progress fact {fact.fact_id!r} references unknown or future "
                f"sources {sorted(unknown_sources)}."
            )
        for source_fact_id in fact.source_fact_ids:
            source = fact_by_id[source_fact_id]
            _require_not_before(
                later_at=fact.effective_at,
                later_clock=fact.effective_clock_domain_id,
                earlier_at=source.effective_at,
                earlier_clock=source.effective_clock_domain_id,
                message=(
                    f"Progress fact {fact.fact_id!r} is effective before source "
                    f"{source_fact_id!r}."
                ),
            )
        if fact.effective_clock_domain_id != effective_clock_domain_id:
            raise ProgressTimelineResolutionError(
                "Progress fact effective time must use the bound Episode clock."
            )
        if (
            latest_effective_at is not None
            and fact.effective_at < latest_effective_at
        ):
            raise ProgressTimelineResolutionError(
                "Progress fact effective time cannot move backwards."
            )
        latest_effective_at = fact.effective_at

        payload = fact.payload
        prior_undo_candidate = undo_candidate
        undo_candidate = None

        if isinstance(payload, EpisodeProgressStartedV1):
            if phase != "NOT_STARTED" or sequence != 1:
                raise ProgressTimelineResolutionError(
                    "Progress start must be the first fact."
                )
            if (
                payload.progress_step_id != binding.entry_step_id
                or payload.attempt_ordinal_in_capture != 1
            ):
                raise ProgressTimelineResolutionError(
                    "Progress start does not match the Episode binding."
                )
            builder = _AttemptBuilder(
                attempt_id=payload.attempt_id,
                progress_step_id=payload.progress_step_id,
                ordinal=1,
                start_fact_id=fact.fact_id,
                start_at=fact.effective_at,
                clock_domain_id=fact.effective_clock_domain_id,
            )
            attempt_by_id[payload.attempt_id] = builder
            attempt_order.append(payload.attempt_id)
            seen_attempt_ids.add(payload.attempt_id)
            active_attempt_id = payload.attempt_id
            current_progress_step_id = payload.progress_step_id
            latest_ordinal = 1
            phase = "ACTIVE_ATTEMPT"

        elif isinstance(payload, ProgressAdvanceEventV1):
            if phase != "ACTIVE_ATTEMPT" or active_attempt_id is None:
                raise ProgressTimelineResolutionError(
                    "Advance requires an active Attempt."
                )
            active = attempt_by_id[active_attempt_id]
            if (
                payload.from_attempt_id != active_attempt_id
                or payload.from_progress_step_id
                != current_progress_step_id
            ):
                raise ProgressTimelineResolutionError(
                    "Advance source does not match the active Attempt."
                )
            if payload.to_progress_step_id not in profile.allowed_next_step_ids(
                payload.from_progress_step_id
            ):
                raise ProgressTimelineResolutionError(
                    "Advance target is not a legal child."
                )
            if payload.to_attempt_id in seen_attempt_ids:
                raise ProgressTimelineResolutionError(
                    "Attempt identities cannot be reused."
                )
            _require_sources(fact, (active.start_fact_id,))
            _require_not_before(
                later_at=fact.effective_at,
                later_clock=fact.effective_clock_domain_id,
                earlier_at=active.start_at,
                earlier_clock=active.clock_domain_id,
                message="Advance cannot precede its active Attempt.",
            )
            active.end_fact_id = fact.fact_id
            active.end_at = fact.effective_at
            active.outcome = "SUCCESS"
            latest_ordinal += 1
            target = _AttemptBuilder(
                attempt_id=payload.to_attempt_id,
                progress_step_id=payload.to_progress_step_id,
                ordinal=latest_ordinal,
                start_fact_id=fact.fact_id,
                start_at=fact.effective_at,
                clock_domain_id=fact.effective_clock_domain_id,
            )
            attempt_by_id[target.attempt_id] = target
            attempt_order.append(target.attempt_id)
            seen_attempt_ids.add(target.attempt_id)
            transition_index = len(transitions)
            transitions.append(
                ResolvedProgressTransition(
                    advance_fact_id=fact.fact_id,
                    from_progress_step_id=payload.from_progress_step_id,
                    from_attempt_id=payload.from_attempt_id,
                    to_progress_step_id=payload.to_progress_step_id,
                    to_attempt_id=payload.to_attempt_id,
                    effective_at=fact.effective_at,
                    clock_domain_id=fact.effective_clock_domain_id,
                )
            )
            undo_candidate = _UndoCandidate(
                advance_fact_id=fact.fact_id,
                from_attempt_id=active.attempt_id,
                to_attempt_id=target.attempt_id,
                transition_index=transition_index,
                restored_ordinal=active.ordinal,
            )
            active_attempt_id = target.attempt_id
            current_progress_step_id = target.progress_step_id

        elif isinstance(payload, ProgressUndoEventV1):
            candidate = prior_undo_candidate
            if (
                phase != "ACTIVE_ATTEMPT"
                or candidate is None
                or payload.advance_fact_id != candidate.advance_fact_id
                or active_attempt_id != candidate.to_attempt_id
                or candidate.transition_index != len(transitions) - 1
            ):
                raise ProgressTimelineResolutionError(
                    "Undo must immediately reference the last effective Advance."
                )
            _require_sources(fact, (candidate.advance_fact_id,))
            transient = attempt_by_id.pop(candidate.to_attempt_id)
            if transient.end_fact_id is not None:
                raise ProgressTimelineResolutionError(
                    "An ended Attempt cannot be removed by Undo."
                )
            if attempt_order[-1] != transient.attempt_id:
                raise ProgressTimelineResolutionError(
                    "Undo target is not the latest Attempt."
                )
            attempt_order.pop()
            invalidated_attempt_ids.append(transient.attempt_id)
            transition = transitions.pop()
            if transition.advance_fact_id != candidate.advance_fact_id:
                raise ProgressTimelineResolutionError(
                    "Undo transition identity mismatch."
                )
            restored = attempt_by_id[candidate.from_attempt_id]
            restored.end_fact_id = None
            restored.end_at = None
            restored.outcome = "ACTIVE"
            active_attempt_id = restored.attempt_id
            current_progress_step_id = restored.progress_step_id
            latest_ordinal = candidate.restored_ordinal
            undone_advance_fact_ids.append(candidate.advance_fact_id)

        elif isinstance(payload, AttemptFailureEventV1):
            if phase != "ACTIVE_ATTEMPT" or active_attempt_id is None:
                raise ProgressTimelineResolutionError(
                    "Failure requires an active Attempt."
                )
            active = attempt_by_id[active_attempt_id]
            if (
                payload.attempt_id != active_attempt_id
                or payload.progress_step_id != current_progress_step_id
            ):
                raise ProgressTimelineResolutionError(
                    "Failure does not match the active Attempt."
                )
            _require_sources(fact, (active.start_fact_id,))
            _require_not_before(
                later_at=fact.effective_at,
                later_clock=fact.effective_clock_domain_id,
                earlier_at=active.start_at,
                earlier_clock=active.clock_domain_id,
                message="Failure cannot precede its active Attempt.",
            )
            if payload.reported_at != fact.effective_at:
                raise ProgressTimelineResolutionError(
                    "Operator Failure report time must equal effective_at."
                )
            active.end_fact_id = fact.fact_id
            active.end_at = fact.effective_at
            active.outcome = "FAILURE"
            active.failure_fact_id = fact.fact_id
            active.failure_reported_at = payload.reported_at
            active.failure_timing_semantics = payload.timing_semantics
            active_attempt_id = None
            pending_failure_fact_id = fact.fact_id
            applied_hold_pause_receipt_fact_id = None
            applied_resume_receipt_fact_id = None
            phase = "AWAITING_HOLD_RECEIPT"

        elif isinstance(payload, HoldPauseEffectReceiptV1):
            if (
                phase != "AWAITING_HOLD_RECEIPT"
                or payload.source_failure_fact_id
                != pending_failure_fact_id
            ):
                raise ProgressTimelineResolutionError(
                    "Hold/Pause receipt does not match the pending Failure."
                )
            assert pending_failure_fact_id is not None
            _require_sources(fact, (pending_failure_fact_id,))
            _validate_operation_availability(
                fact=fact,
                operation=payload.application_hold,
            )
            _validate_operation_availability(
                fact=fact,
                operation=payload.formal_capture_pause,
            )
            pause_operation = payload.formal_capture_pause
            if pause_operation.clock_domain_id != effective_clock_domain_id:
                raise ProgressTimelineResolutionError(
                    "Formal capture pause must use the Episode clock."
                )
            if pause_operation.status == "APPLIED":
                assert pause_operation.effective_at is not None
                existing_pause = pause_by_failure.get(pending_failure_fact_id)
                if existing_pause is None:
                    pause_by_failure[pending_failure_fact_id] = _PauseBuilder(
                        source_failure_fact_id=pending_failure_fact_id,
                        pause_receipt_fact_id=fact.fact_id,
                        start_at=pause_operation.effective_at,
                        clock_domain_id=pause_operation.clock_domain_id,
                    )
                    pause_order.append(pending_failure_fact_id)
                elif (
                    existing_pause.start_at != pause_operation.effective_at
                    or existing_pause.clock_domain_id
                    != pause_operation.clock_domain_id
                ):
                    raise ProgressTimelineResolutionError(
                        "Repeated applied capture-pause receipts disagree."
                    )
            if payload.applied:
                applied_hold_pause_receipt_fact_id = fact.fact_id
                phase = "PAUSED_NO_ACTIVE_ATTEMPT"

        elif isinstance(payload, ResumeAfterFailureReceiptV1):
            if (
                phase != "PAUSED_NO_ACTIVE_ATTEMPT"
                or payload.source_failure_fact_id
                != pending_failure_fact_id
                or payload.hold_pause_receipt_fact_id
                != applied_hold_pause_receipt_fact_id
            ):
                raise ProgressTimelineResolutionError(
                    "Resume receipt does not match the applied Hold/Pause."
                )
            assert pending_failure_fact_id is not None
            assert applied_hold_pause_receipt_fact_id is not None
            _require_sources(
                fact,
                (
                    pending_failure_fact_id,
                    applied_hold_pause_receipt_fact_id,
                ),
            )
            _validate_operation_availability(
                fact=fact,
                operation=payload.application_resume,
            )
            _validate_operation_availability(
                fact=fact,
                operation=payload.formal_capture_resume,
            )
            if (
                payload.formal_capture_resume.clock_domain_id
                != effective_clock_domain_id
            ):
                raise ProgressTimelineResolutionError(
                    "Formal capture resume must use the Episode clock."
                )
            if payload.applied:
                pause = pause_by_failure.get(pending_failure_fact_id)
                resume_at = payload.formal_capture_resume.effective_at
                if pause is None or resume_at is None:
                    raise ProgressTimelineResolutionError(
                        "Applied Resume requires a recorded capture pause."
                    )
                if (
                    payload.formal_capture_resume.clock_domain_id
                    != pause.clock_domain_id
                ):
                    raise ProgressTimelineResolutionError(
                        "Capture pause/resume clocks differ."
                    )
                _require_not_before(
                    later_at=resume_at,
                    later_clock=pause.clock_domain_id,
                    earlier_at=pause.start_at,
                    earlier_clock=pause.clock_domain_id,
                    message="Formal capture Resume precedes capture Pause.",
                )
                pause.resume_receipt_fact_id = fact.fact_id
                pause.end_at = resume_at
                applied_resume_receipt_fact_id = fact.fact_id
                phase = "RESUME_RECEIVED"

        elif isinstance(payload, AttemptRetryEventV1):
            if (
                phase != "RESUME_RECEIVED"
                or payload.source_failure_fact_id
                != pending_failure_fact_id
                or payload.hold_pause_receipt_fact_id
                != applied_hold_pause_receipt_fact_id
                or payload.resume_receipt_fact_id
                != applied_resume_receipt_fact_id
                or payload.progress_step_id != current_progress_step_id
            ):
                raise ProgressTimelineResolutionError(
                    "Retry references do not match the resumed Failure."
                )
            if not profile.allow_retry_after_failure:
                raise ProgressTimelineResolutionError(
                    "The progress profile forbids Retry."
                )
            if payload.new_attempt_id in seen_attempt_ids:
                raise ProgressTimelineResolutionError(
                    "Attempt identities cannot be reused."
                )
            if payload.attempt_ordinal_in_capture != latest_ordinal + 1:
                raise ProgressTimelineResolutionError(
                    "Retry Attempt ordinal is not contiguous."
                )
            assert pending_failure_fact_id is not None
            assert applied_hold_pause_receipt_fact_id is not None
            assert applied_resume_receipt_fact_id is not None
            _require_sources(
                fact,
                (
                    pending_failure_fact_id,
                    applied_hold_pause_receipt_fact_id,
                    applied_resume_receipt_fact_id,
                ),
            )
            resume_fact = fact_by_id[applied_resume_receipt_fact_id]
            assert isinstance(
                resume_fact.payload,
                ResumeAfterFailureReceiptV1,
            )
            resumed_at = (
                resume_fact.payload.formal_capture_resume.effective_at
            )
            assert resumed_at is not None
            _require_not_before(
                later_at=fact.effective_at,
                later_clock=fact.effective_clock_domain_id,
                earlier_at=resumed_at,
                earlier_clock=(
                    resume_fact.payload.formal_capture_resume.clock_domain_id
                ),
                message="Retry Attempt begins before formal capture Resume.",
            )
            latest_ordinal = payload.attempt_ordinal_in_capture
            retry = _AttemptBuilder(
                attempt_id=payload.new_attempt_id,
                progress_step_id=payload.progress_step_id,
                ordinal=latest_ordinal,
                start_fact_id=fact.fact_id,
                start_at=fact.effective_at,
                clock_domain_id=fact.effective_clock_domain_id,
            )
            attempt_by_id[retry.attempt_id] = retry
            attempt_order.append(retry.attempt_id)
            seen_attempt_ids.add(retry.attempt_id)
            active_attempt_id = retry.attempt_id
            pending_failure_fact_id = None
            applied_hold_pause_receipt_fact_id = None
            applied_resume_receipt_fact_id = None
            phase = "ACTIVE_ATTEMPT"

        elif isinstance(payload, EpisodeProgressTerminationV1):
            _validate_termination(
                profile=profile,
                binding=binding,
                phase=phase,
                current_progress_step_id=current_progress_step_id,
                payload=payload,
                pending_failure_fact_id=pending_failure_fact_id,
            )
            if active_attempt_id is not None:
                active = attempt_by_id[active_attempt_id]
                _require_sources(fact, (active.start_fact_id,))
                _require_not_before(
                    later_at=fact.effective_at,
                    later_clock=fact.effective_clock_domain_id,
                    earlier_at=active.start_at,
                    earlier_clock=active.clock_domain_id,
                    message="Termination cannot precede its active Attempt.",
                )
                active.end_fact_id = fact.fact_id
                active.end_at = fact.effective_at
                active.outcome = payload.active_attempt_disposition
                active_attempt_id = None
            terminal_fact_id = fact.fact_id
            termination_cause = payload.termination_cause
            full_task_outcome = payload.full_task_outcome
            phase = "TERMINATED"

        else:  # pragma: no cover - the envelope rejects unregistered payloads.
            raise ProgressTimelineResolutionError(
                f"Unsupported progress payload {type(payload).__name__}."
            )

        fact_ids.append(fact.fact_id)
        fact_by_id[fact.fact_id] = fact

    if manifest is not None:
        _verify_manifest(
            profile=profile,
            binding=binding,
            facts=ordered,
            manifest=manifest,
            terminal_fact_id=terminal_fact_id,
        )
        if phase != "TERMINATED":
            raise ProgressTimelineResolutionError(
                "A sealed manifest requires a terminated progress timeline."
            )

    attempts = tuple(
        attempt_by_id[attempt_id].materialize()
        for attempt_id in attempt_order
    )
    pauses = tuple(
        pause_by_failure[failure_id].materialize()
        for failure_id in pause_order
    )
    return ResolvedProgressTimeline(
        episode_capture_identity_hash=binding.episode_capture_identity_hash,
        task_progress_profile_hash=profile.semantic_content_hash,
        progress_episode_binding_hash=binding.semantic_content_hash,
        clock_domain_id=binding.bound_clock_domain_id,
        fact_bundle_hash=ProgressEpisodeManifestV1.fact_bundle_hash_for(ordered),
        fact_count=len(ordered),
        sealed=manifest is not None,
        phase=phase,
        attempts=attempts,
        transitions=tuple(transitions),
        capture_pause_intervals=pauses,
        undone_advance_fact_ids=tuple(undone_advance_fact_ids),
        invalidated_attempt_ids=tuple(invalidated_attempt_ids),
        current_progress_step_id=current_progress_step_id,
        active_attempt_id=active_attempt_id,
        terminal_fact_id=terminal_fact_id,
        termination_cause=termination_cause,
        full_task_outcome=full_task_outcome,
        resolution_kind="CAPTURE_CAUSAL",
        source_progress_manifest_hash=(
            manifest.semantic_content_hash if manifest is not None else None
        ),
    )


__all__ = [
    "ProgressTimelineResolutionError",
    "ResolvedCapturePauseInterval",
    "ResolvedProgressAttempt",
    "ResolvedProgressTimeline",
    "ResolvedProgressTransition",
    "resolve_progress_timeline",
]
