"""Pure state reduction for typed collection-progress facts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

from dataset_contracts.progress_facts import (
    AttemptFailureEventV1,
    AttemptRetryEventV1,
    EpisodeProgressStartedV1,
    EpisodeProgressTerminationV1,
    HoldPauseEffectReceiptV1,
    ProgressAdvanceEventV1,
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


PROGRESS_PHASES = frozenset(
    {
        "NOT_STARTED",
        "ACTIVE_ATTEMPT",
        "AWAITING_HOLD_RECEIPT",
        "PAUSED_NO_ACTIVE_ATTEMPT",
        "RESUME_RECEIVED",
        "TERMINATED",
    }
)


class ProgressReductionError(ValueError):
    """Raised when a fact cannot follow the current canonical state."""


@dataclass(frozen=True)
class ProgressState:
    phase: str = "NOT_STARTED"
    fact_count: int = 0
    fact_ids: tuple[str, ...] = ()
    latest_effective_at: float | None = None
    current_progress_step_id: str | None = None
    active_attempt_id: str | None = None
    active_attempt_start_fact_id: str | None = None
    latest_attempt_ordinal_in_capture: int = 0
    attempt_ids: tuple[str, ...] = ()
    pending_failure_fact_id: str | None = None
    hold_pause_receipt_fact_id: str | None = None
    resume_receipt_fact_id: str | None = None
    resume_operation_id: str | None = None
    last_effective_advance_fact_id: str | None = None
    undo_restore_progress_step_id: str | None = None
    undo_restore_attempt_id: str | None = None
    undo_restore_attempt_start_fact_id: str | None = None
    undo_restore_attempt_ordinal_in_capture: int | None = None
    terminal_fact_id: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PROGRESS_PHASES:
            raise ProgressReductionError(f"Unsupported progress phase {self.phase!r}.")
        if self.fact_count != len(self.fact_ids):
            raise ProgressReductionError("fact_count/fact_ids mismatch.")
        if self.fact_count == 0:
            if self.latest_effective_at is not None:
                raise ProgressReductionError(
                    "An empty progress state cannot have an effective time."
                )
        elif (
            not isinstance(self.latest_effective_at, (int, float))
            or isinstance(self.latest_effective_at, bool)
            or not math.isfinite(float(self.latest_effective_at))
        ):
            raise ProgressReductionError(
                "A non-empty progress state requires a finite effective time."
            )
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ProgressReductionError("Progress state contains duplicate fact IDs.")
        if len(self.attempt_ids) != len(set(self.attempt_ids)):
            raise ProgressReductionError("Progress state contains duplicate Attempt IDs.")
        if self.phase == "ACTIVE_ATTEMPT":
            if (
                self.current_progress_step_id is None
                or self.active_attempt_id is None
                or self.active_attempt_start_fact_id is None
            ):
                raise ProgressReductionError(
                    "ACTIVE_ATTEMPT requires a current step and active Attempt."
                )
        elif self.active_attempt_id is not None:
            raise ProgressReductionError(
                f"{self.phase} cannot retain an active Attempt."
            )
        if self.phase == "TERMINATED" and self.terminal_fact_id is None:
            raise ProgressReductionError("TERMINATED requires terminal_fact_id.")


def initial_progress_state() -> ProgressState:
    return ProgressState()


def _clear_undo(state: ProgressState) -> ProgressState:
    return replace(
        state,
        last_effective_advance_fact_id=None,
        undo_restore_progress_step_id=None,
        undo_restore_attempt_id=None,
        undo_restore_attempt_start_fact_id=None,
        undo_restore_attempt_ordinal_in_capture=None,
    )


def _with_fact(
    state: ProgressState,
    fact: ProgressFactEnvelopeV1,
) -> ProgressState:
    return replace(
        state,
        fact_count=state.fact_count + 1,
        fact_ids=state.fact_ids + (fact.fact_id,),
        latest_effective_at=fact.effective_at,
    )


def _validate_common(
    *,
    binding: ProgressEpisodeBindingV1,
    state: ProgressState,
    fact: ProgressFactEnvelopeV1,
) -> None:
    if state.phase == "TERMINATED":
        raise ProgressReductionError("No fact may follow Episode termination.")
    if (
        fact.episode_capture_identity_hash
        != binding.episode_capture_identity_hash
    ):
        raise ProgressReductionError("Progress fact Episode identity mismatch.")
    if fact.effective_clock_domain_id != binding.bound_clock_domain_id:
        raise ProgressReductionError(
            "Progress fact effective time must use the bound Episode clock."
        )
    if fact.progress_sequence_index != state.fact_count + 1:
        raise ProgressReductionError(
            "progress_sequence_index must be contiguous and append-owned."
        )
    if (
        state.latest_effective_at is not None
        and fact.effective_at < state.latest_effective_at
    ):
        raise ProgressReductionError(
            "Progress fact effective time cannot move backwards."
        )
    if fact.fact_id in state.fact_ids:
        raise ProgressReductionError(f"Duplicate progress fact ID {fact.fact_id!r}.")
    unknown_sources = set(fact.source_fact_ids) - set(state.fact_ids)
    if unknown_sources:
        raise ProgressReductionError(
            f"Progress fact references unknown sources: {sorted(unknown_sources)}."
        )


def _require_sources(
    fact: ProgressFactEnvelopeV1,
    required: tuple[str, ...],
) -> None:
    missing = set(required) - set(fact.source_fact_ids)
    if missing:
        raise ProgressReductionError(
            f"Progress fact omits required source facts: {sorted(missing)}."
        )


def _validate_operation_availability(
    *,
    fact: ProgressFactEnvelopeV1,
    operation: ProgressOperationReceiptV1,
) -> None:
    """Keep live acceptance aligned with the sealed read-time resolver."""

    for at, label in (
        (operation.requested_at, "requested"),
        (operation.effective_at, "effective"),
    ):
        if (
            at is not None
            and operation.clock_domain_id == fact.available_clock_domain_id
            and at > fact.available_at
        ):
            raise ProgressReductionError(
                f"Operation {operation.operation} {label} time is after its "
                f"receipt fact {fact.fact_id!r} became available."
            )


def _validate_termination(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    state: ProgressState,
    payload: EpisodeProgressTerminationV1,
) -> None:
    cause = payload.termination_cause
    disposition = payload.active_attempt_disposition
    outcome = payload.full_task_outcome
    active = state.phase == "ACTIVE_ATTEMPT"

    if cause == "FULL_TASK_COMPLETED":
        if (
            binding.recording_scope != "FULL_TASK"
            or not active
            or state.current_progress_step_id not in profile.terminal_step_ids
            or disposition != "SUCCESS"
            or outcome != "SUCCESS"
        ):
            raise ProgressReductionError(
                "FULL_TASK_COMPLETED requires an active terminal Attempt and "
                "full-task SUCCESS."
            )
    elif cause == "WINDOW_COMPLETED":
        if binding.recording_scope == "FULL_TASK":
            raise ProgressReductionError(
                "FULL_TASK cannot terminate as WINDOW_COMPLETED."
            )
        if outcome != "NOT_APPLICABLE":
            raise ProgressReductionError(
                "A window cannot claim a full-task outcome."
            )
        if active and disposition not in {
            "SUCCESS",
            "TRUNCATED_AT_CAPTURE_END",
        }:
            raise ProgressReductionError(
                "An active window must end as SUCCESS or TRUNCATED."
            )
        if not active and disposition != "NO_ACTIVE_ATTEMPT":
            raise ProgressReductionError(
                "A window without an active Attempt must use NO_ACTIVE_ATTEMPT."
            )
    else:
        if active and disposition not in {
            "ABORTED",
            "TRUNCATED_AT_CAPTURE_END",
            "UNKNOWN",
        }:
            raise ProgressReductionError(
                "Abnormal termination cannot mark an active Attempt successful."
            )
        if not active and disposition != "NO_ACTIVE_ATTEMPT":
            raise ProgressReductionError(
                "Termination without an active Attempt requires NO_ACTIVE_ATTEMPT."
            )
        if binding.recording_scope != "FULL_TASK" and outcome != "NOT_APPLICABLE":
            raise ProgressReductionError(
                "A partial recording cannot claim a full-task outcome."
            )
        if (
            binding.recording_scope == "FULL_TASK"
            and outcome not in {"FAILURE", "UNKNOWN"}
        ):
            raise ProgressReductionError(
                "An abnormal full-task termination must use FAILURE or UNKNOWN."
            )

    unresolved_failures = set(payload.unresolved_source_failure_fact_ids)
    expected_unresolved = (
        {state.pending_failure_fact_id}
        if state.phase == "AWAITING_HOLD_RECEIPT"
        and state.pending_failure_fact_id is not None
        else set()
    )
    if unresolved_failures != expected_unresolved:
        raise ProgressReductionError(
            "Termination must exactly explain unresolved Failure effects."
        )
    if payload.unresolved_resume_operation_ids:
        raise ProgressReductionError(
            "No unresolved Resume operation exists in the current p2 state."
        )


def reduce_progress_fact(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    state: ProgressState,
    fact: ProgressFactEnvelopeV1,
) -> ProgressState:
    """Apply one already-typed fact without IO or application side effects."""

    validate_progress_episode_binding(profile=profile, binding=binding)
    _validate_common(binding=binding, state=state, fact=fact)
    payload = fact.payload

    if isinstance(payload, EpisodeProgressStartedV1):
        if state.phase != "NOT_STARTED" or state.fact_count:
            raise ProgressReductionError("Progress start must be the first fact.")
        if (
            payload.progress_step_id != binding.entry_step_id
            or payload.attempt_ordinal_in_capture != 1
        ):
            raise ProgressReductionError(
                "Progress start must create Attempt 1 at the bound entry step."
            )
        result = replace(
            state,
            phase="ACTIVE_ATTEMPT",
            current_progress_step_id=payload.progress_step_id,
            active_attempt_id=payload.attempt_id,
            active_attempt_start_fact_id=fact.fact_id,
            latest_attempt_ordinal_in_capture=1,
            attempt_ids=(payload.attempt_id,),
        )
    elif isinstance(payload, ProgressAdvanceEventV1):
        if state.phase != "ACTIVE_ATTEMPT":
            raise ProgressReductionError("Advance requires an active Attempt.")
        if (
            payload.from_progress_step_id != state.current_progress_step_id
            or payload.from_attempt_id != state.active_attempt_id
        ):
            raise ProgressReductionError("Advance source does not match active state.")
        allowed = profile.allowed_next_step_ids(payload.from_progress_step_id)
        if payload.to_progress_step_id not in allowed:
            raise ProgressReductionError("Advance target is not a legal child.")
        if payload.to_attempt_id == payload.from_attempt_id:
            raise ProgressReductionError("Advance must create a new Attempt identity.")
        if payload.to_attempt_id in state.attempt_ids:
            raise ProgressReductionError("Attempt identities cannot be reused.")
        assert state.active_attempt_start_fact_id is not None
        _require_sources(fact, (state.active_attempt_start_fact_id,))
        result = replace(
            state,
            phase="ACTIVE_ATTEMPT",
            current_progress_step_id=payload.to_progress_step_id,
            active_attempt_id=payload.to_attempt_id,
            active_attempt_start_fact_id=fact.fact_id,
            latest_attempt_ordinal_in_capture=(
                state.latest_attempt_ordinal_in_capture + 1
            ),
            attempt_ids=state.attempt_ids + (payload.to_attempt_id,),
            last_effective_advance_fact_id=fact.fact_id,
            undo_restore_progress_step_id=state.current_progress_step_id,
            undo_restore_attempt_id=state.active_attempt_id,
            undo_restore_attempt_start_fact_id=state.active_attempt_start_fact_id,
            undo_restore_attempt_ordinal_in_capture=(
                state.latest_attempt_ordinal_in_capture
            ),
        )
    elif isinstance(payload, ProgressUndoEventV1):
        if (
            state.phase != "ACTIVE_ATTEMPT"
            or payload.advance_fact_id != state.last_effective_advance_fact_id
            or state.undo_restore_progress_step_id is None
            or state.undo_restore_attempt_id is None
            or state.undo_restore_attempt_start_fact_id is None
            or state.undo_restore_attempt_ordinal_in_capture is None
        ):
            raise ProgressReductionError(
                "Undo must immediately reference the last effective advance."
            )
        _require_sources(fact, (payload.advance_fact_id,))
        result = replace(
            state,
            current_progress_step_id=state.undo_restore_progress_step_id,
            active_attempt_id=state.undo_restore_attempt_id,
            active_attempt_start_fact_id=state.undo_restore_attempt_start_fact_id,
            latest_attempt_ordinal_in_capture=(
                state.undo_restore_attempt_ordinal_in_capture
            ),
        )
        result = _clear_undo(result)
    elif isinstance(payload, AttemptFailureEventV1):
        if state.phase != "ACTIVE_ATTEMPT":
            raise ProgressReductionError("Failure requires an active Attempt.")
        if (
            payload.progress_step_id != state.current_progress_step_id
            or payload.attempt_id != state.active_attempt_id
        ):
            raise ProgressReductionError("Failure does not match the active Attempt.")
        assert state.active_attempt_start_fact_id is not None
        _require_sources(fact, (state.active_attempt_start_fact_id,))
        if payload.reported_at != fact.effective_at:
            raise ProgressReductionError(
                "Online operator failure time must equal the fact effective time."
            )
        result = replace(
            state,
            phase="AWAITING_HOLD_RECEIPT",
            active_attempt_id=None,
            active_attempt_start_fact_id=None,
            pending_failure_fact_id=fact.fact_id,
            hold_pause_receipt_fact_id=None,
            resume_receipt_fact_id=None,
            resume_operation_id=None,
        )
        result = _clear_undo(result)
    elif isinstance(payload, HoldPauseEffectReceiptV1):
        if (
            state.phase != "AWAITING_HOLD_RECEIPT"
            or payload.source_failure_fact_id != state.pending_failure_fact_id
        ):
            raise ProgressReductionError(
                "Hold/Pause receipt does not match the pending Failure."
            )
        assert state.pending_failure_fact_id is not None
        _require_sources(fact, (state.pending_failure_fact_id,))
        _validate_operation_availability(
            fact=fact,
            operation=payload.application_hold,
        )
        _validate_operation_availability(
            fact=fact,
            operation=payload.formal_capture_pause,
        )
        if (
            payload.formal_capture_pause.clock_domain_id
            != binding.bound_clock_domain_id
        ):
            raise ProgressReductionError(
                "Formal capture Pause must use the bound Episode clock."
            )
        if payload.applied:
            result = replace(
                state,
                phase="PAUSED_NO_ACTIVE_ATTEMPT",
                hold_pause_receipt_fact_id=fact.fact_id,
            )
        else:
            result = state
        result = _clear_undo(result)
    elif isinstance(payload, ResumeAfterFailureReceiptV1):
        if (
            state.phase != "PAUSED_NO_ACTIVE_ATTEMPT"
            or payload.source_failure_fact_id != state.pending_failure_fact_id
            or payload.hold_pause_receipt_fact_id
            != state.hold_pause_receipt_fact_id
        ):
            raise ProgressReductionError(
                "Resume receipt does not match the applied Hold/Pause."
            )
        assert state.pending_failure_fact_id is not None
        assert state.hold_pause_receipt_fact_id is not None
        _require_sources(
            fact,
            (
                state.pending_failure_fact_id,
                state.hold_pause_receipt_fact_id,
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
            != binding.bound_clock_domain_id
        ):
            raise ProgressReductionError(
                "Formal capture Resume must use the bound Episode clock."
            )
        if payload.applied:
            result = replace(
                state,
                phase="RESUME_RECEIVED",
                resume_receipt_fact_id=fact.fact_id,
                resume_operation_id=payload.resume_operation_id,
            )
        else:
            result = state
        result = _clear_undo(result)
    elif isinstance(payload, AttemptRetryEventV1):
        if (
            state.phase != "RESUME_RECEIVED"
            or payload.source_failure_fact_id != state.pending_failure_fact_id
            or payload.hold_pause_receipt_fact_id
            != state.hold_pause_receipt_fact_id
            or payload.resume_receipt_fact_id != state.resume_receipt_fact_id
            or payload.progress_step_id != state.current_progress_step_id
        ):
            raise ProgressReductionError(
                "Retry references do not match the paused Attempt."
            )
        if not profile.allow_retry_after_failure:
            raise ProgressReductionError("The profile forbids retry after failure.")
        if (
            payload.attempt_ordinal_in_capture
            != state.latest_attempt_ordinal_in_capture + 1
        ):
            raise ProgressReductionError("Retry Attempt ordinal is not contiguous.")
        if payload.new_attempt_id in state.attempt_ids:
            raise ProgressReductionError("Attempt identities cannot be reused.")
        assert state.pending_failure_fact_id is not None
        assert state.hold_pause_receipt_fact_id is not None
        assert state.resume_receipt_fact_id is not None
        _require_sources(
            fact,
            (
                state.pending_failure_fact_id,
                state.hold_pause_receipt_fact_id,
                state.resume_receipt_fact_id,
            ),
        )
        result = replace(
            state,
            phase="ACTIVE_ATTEMPT",
            active_attempt_id=payload.new_attempt_id,
            active_attempt_start_fact_id=fact.fact_id,
            latest_attempt_ordinal_in_capture=(
                payload.attempt_ordinal_in_capture
            ),
            attempt_ids=state.attempt_ids + (payload.new_attempt_id,),
            pending_failure_fact_id=None,
            hold_pause_receipt_fact_id=None,
            resume_receipt_fact_id=None,
            resume_operation_id=None,
        )
        result = _clear_undo(result)
    elif isinstance(payload, EpisodeProgressTerminationV1):
        _validate_termination(
            profile=profile,
            binding=binding,
            state=state,
            payload=payload,
        )
        result = replace(
            state,
            phase="TERMINATED",
            active_attempt_id=None,
            active_attempt_start_fact_id=None,
            terminal_fact_id=fact.fact_id,
        )
        result = _clear_undo(result)
    else:  # pragma: no cover - envelope registration already rejects this.
        raise ProgressReductionError(
            f"Unsupported progress payload {type(payload).__name__}."
        )
    return _with_fact(result, fact)


def replay_progress_facts(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    facts: Iterable[ProgressFactEnvelopeV1],
) -> ProgressState:
    state = initial_progress_state()
    for fact in facts:
        state = reduce_progress_fact(
            profile=profile,
            binding=binding,
            state=state,
            fact=fact,
        )
    return state


__all__ = [
    "PROGRESS_PHASES",
    "ProgressReductionError",
    "ProgressState",
    "initial_progress_state",
    "reduce_progress_fact",
    "replay_progress_facts",
]
