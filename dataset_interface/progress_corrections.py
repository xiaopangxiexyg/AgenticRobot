"""Read-only retrospective corrections over sealed progress timelines.

This module never rewrites capture facts.  Its first deliberately narrow
operation proves that a sealed capture timeline can produce a different,
validated retrospective view.  It is not the final post-annotation correction
model; that model may replace a bounded timeline segment after profile
validation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from dataset_contracts.progress_facts import (
    ProgressEpisodeManifestV1,
    ProgressFactEnvelopeV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)

from .progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
    resolve_progress_timeline,
)


class ProgressCorrectionResolutionError(ValueError):
    """Raised when a correction cannot be applied without guessing."""


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ProgressCorrectionResolutionError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True)
class FailureAttemptReassignment:
    """Narrow prototype instruction bound to one sealed progress manifest."""

    correction_id: str
    source_progress_manifest_hash: str
    source_failure_fact_id: str
    target_attempt_id: str

    def __post_init__(self) -> None:
        for name in (
            "correction_id",
            "source_progress_manifest_hash",
            "source_failure_fact_id",
            "target_attempt_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        digest = self.source_progress_manifest_hash.lower()
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ProgressCorrectionResolutionError(
                "source_progress_manifest_hash must be a lowercase SHA-256 digest."
            )
        object.__setattr__(self, "source_progress_manifest_hash", digest)


@dataclass(frozen=True)
class ResolvedFailureAttemptReassignment:
    """Atomic retrospective result; source capture facts remain untouched."""

    correction_id: str
    source_progress_manifest_hash: str
    source_failure_fact_id: str
    target_attempt_id: str
    invalidated_advance_fact_ids: tuple[str, ...]
    invalidated_attempt_ids: tuple[str, ...]
    timeline: ResolvedProgressTimeline


def _failure_owner(
    timeline: ResolvedProgressTimeline,
    *,
    failure_fact_id: str,
) -> tuple[int, ResolvedProgressAttempt]:
    matches = tuple(
        (index, attempt)
        for index, attempt in enumerate(timeline.attempts)
        if attempt.failure_fact_id == failure_fact_id
    )
    if len(matches) != 1:
        raise ProgressCorrectionResolutionError(
            "source_failure_fact_id must identify exactly one failed Attempt."
        )
    return matches[0]


def reassign_failure_to_attempt(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
    facts: Iterable[ProgressFactEnvelopeV1],
    manifest: ProgressEpisodeManifestV1,
    correction: FailureAttemptReassignment,
) -> ResolvedFailureAttemptReassignment:
    """Reassign one terminal linear Failure to an earlier Attempt.

    The correction invalidates every Advance from the target Attempt through
    the original Failure owner and removes the Attempts those Advances created.
    A later Retry or later Attempt is rejected for now because preserving that
    suffix requires an explicit branch/step correction rather than inference.
    """

    if profile.topology_kind != "LINEAR":
        raise ProgressCorrectionResolutionError(
            "Failure reassignment currently supports LINEAR profiles only."
        )
    if (
        correction.source_progress_manifest_hash
        != manifest.semantic_content_hash
    ):
        raise ProgressCorrectionResolutionError(
            "Correction is not bound to the supplied sealed progress manifest."
        )
    ordered = tuple(facts)
    source = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=ordered,
        manifest=manifest,
        require_sealed=True,
    )
    source_index, source_attempt = _failure_owner(
        source,
        failure_fact_id=correction.source_failure_fact_id,
    )
    if source_index != len(source.attempts) - 1:
        raise ProgressCorrectionResolutionError(
            "Failure reassignment with a later Retry or Attempt requires an "
            "explicit suffix correction and is not inferred."
        )
    target_matches = tuple(
        (index, attempt)
        for index, attempt in enumerate(source.attempts)
        if attempt.attempt_id == correction.target_attempt_id
    )
    if len(target_matches) != 1:
        raise ProgressCorrectionResolutionError(
            "target_attempt_id must identify exactly one canonical Attempt."
        )
    target_index, target_attempt = target_matches[0]
    if target_index >= source_index:
        raise ProgressCorrectionResolutionError(
            "Failure reassignment target must precede the original Failure owner."
        )
    if target_attempt.outcome != "SUCCESS":
        raise ProgressCorrectionResolutionError(
            "Failure reassignment target must be an Advance-closed SUCCESS Attempt."
        )

    chain_attempts = source.attempts[target_index : source_index + 1]
    chain_transitions = source.transitions[target_index:source_index]
    if len(chain_transitions) != len(chain_attempts) - 1:
        raise ProgressCorrectionResolutionError(
            "Target-to-Failure transition chain is incomplete."
        )
    for prior, following, transition in zip(
        chain_attempts[:-1],
        chain_attempts[1:],
        chain_transitions,
        strict=True,
    ):
        if (
            transition.from_attempt_id != prior.attempt_id
            or transition.to_attempt_id != following.attempt_id
        ):
            raise ProgressCorrectionResolutionError(
                "Target-to-Failure transition chain is not contiguous."
            )

    corrected_target = replace(
        target_attempt,
        end_fact_id=source_attempt.end_fact_id,
        end_at=source_attempt.end_at,
        outcome="FAILURE",
        failure_fact_id=source_attempt.failure_fact_id,
        failure_reported_at=source_attempt.failure_reported_at,
        failure_timing_semantics=source_attempt.failure_timing_semantics,
    )
    removed_attempt_ids = tuple(
        attempt.attempt_id for attempt in chain_attempts[1:]
    )
    invalidated_attempt_ids = tuple(
        dict.fromkeys(source.invalidated_attempt_ids + removed_attempt_ids)
    )
    corrected_timeline = replace(
        source,
        attempts=source.attempts[:target_index] + (corrected_target,),
        transitions=source.transitions[:target_index],
        invalidated_attempt_ids=invalidated_attempt_ids,
        current_progress_step_id=corrected_target.progress_step_id,
        active_attempt_id=None,
        resolution_kind="RETROSPECTIVE_CORRECTION",
    )
    return ResolvedFailureAttemptReassignment(
        correction_id=correction.correction_id,
        source_progress_manifest_hash=manifest.semantic_content_hash,
        source_failure_fact_id=correction.source_failure_fact_id,
        target_attempt_id=correction.target_attempt_id,
        invalidated_advance_fact_ids=tuple(
            transition.advance_fact_id for transition in chain_transitions
        ),
        invalidated_attempt_ids=removed_attempt_ids,
        timeline=corrected_timeline,
    )


__all__ = [
    "FailureAttemptReassignment",
    "ProgressCorrectionResolutionError",
    "ResolvedFailureAttemptReassignment",
    "reassign_failure_to_attempt",
]
