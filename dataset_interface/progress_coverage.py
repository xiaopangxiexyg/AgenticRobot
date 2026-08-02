"""Rebuildable refinement-annotation coverage views.

Coverage is a structural read view over one sealed capture-time progress
timeline and one sealed annotation set.  It is deliberately not a persisted
dataset contract: deleting every coverage artifact must not lose capture or
annotation truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.progress_annotations import (
    RefinementBoundaryAnnotationV1,
)
from dataset_contracts.progress_profile import TaskProgressProfileV1
from dataset_contracts.progress_refinement import (
    TaskRefinementProfileV1,
    validate_task_refinement_profile,
)

from .progress_refinement import SelectedRefinementAnnotationSet
from .progress_timeline import ResolvedProgressAttempt, ResolvedProgressTimeline


REFINEMENT_COVERAGE_STATUSES = frozenset(
    {
        "NOT_APPLICABLE",
        "UNANNOTATED",
        "PARTIAL",
        "COMPLETE",
        "CONFLICTED",
        "INVALID",
    }
)


class ProgressCoverageResolutionError(ValueError):
    """Raised when a coverage view cannot be derived unambiguously."""


@dataclass(frozen=True)
class RefinementAttemptCoverage:
    """Structural refinement coverage for one resolved coarse Attempt."""

    attempt_id: str
    progress_step_id: str
    status: str
    required_boundary_ids: tuple[str, ...]
    observed_boundary_ids: tuple[str, ...]
    missing_boundary_ids: tuple[str, ...]
    conflicting_boundary_ids: tuple[str, ...]
    annotation_ids: tuple[str, ...]
    invalid_annotation_ids: tuple[str, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RefinementAnnotationCoverageView:
    """Episode-local coverage rebuilt from canonical sealed inputs."""

    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    source_annotation_manifest_hash: str
    task_refinement_profile_hash: str
    annotation_set_id: str
    clock_domain_id: str
    attempts: tuple[RefinementAttemptCoverage, ...]
    unassigned_annotation_ids: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(REFINEMENT_COVERAGE_STATUSES)}
        for attempt in self.attempts:
            counts[attempt.status] += 1
        return counts

    @property
    def has_invalid_annotations(self) -> bool:
        return bool(
            self.unassigned_annotation_ids
            or any(item.invalid_annotation_ids for item in self.attempts)
        )


def _validate_inputs(
    *,
    timeline: ResolvedProgressTimeline,
    progress_profile: TaskProgressProfileV1,
    refinement_profile: TaskRefinementProfileV1,
    annotation_set: SelectedRefinementAnnotationSet,
) -> None:
    validate_task_refinement_profile(
        progress_profile=progress_profile,
        refinement_profile=refinement_profile,
    )
    if not timeline.sealed or timeline.source_progress_manifest_hash is None:
        raise ProgressCoverageResolutionError(
            "Refinement coverage requires a sealed progress timeline."
        )
    if timeline.resolution_kind != "CAPTURE_CAUSAL":
        raise ProgressCoverageResolutionError(
            "Refinement coverage currently requires a capture-causal timeline; "
            "retrospective annotation-set selection is not frozen."
        )
    if (
        timeline.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressCoverageResolutionError(
            "Progress timeline/profile hash mismatch."
        )
    if not annotation_set.sealed:
        raise ProgressCoverageResolutionError(
            "Refinement coverage requires a sealed annotation set."
        )
    if (
        annotation_set.source_progress_manifest_hash
        != timeline.source_progress_manifest_hash
    ):
        raise ProgressCoverageResolutionError(
            "Annotation set is not bound to the resolved progress manifest."
        )
    if (
        annotation_set.task_refinement_profile_hash
        != refinement_profile.semantic_content_hash
    ):
        raise ProgressCoverageResolutionError(
            "Annotation/refinement profile hash mismatch."
        )


def _attempt_coverage(
    *,
    attempt: ResolvedProgressAttempt,
    timeline: ResolvedProgressTimeline,
    refinement_profile: TaskRefinementProfileV1,
    annotations: tuple[RefinementBoundaryAnnotationV1, ...],
) -> RefinementAttemptCoverage:
    definitions = refinement_profile.boundaries_for_step(
        attempt.progress_step_id
    )
    boundary_by_id = refinement_profile.boundary_by_id
    required_ids = tuple(
        item.refinement_boundary_id for item in definitions
    )
    required_id_set = set(required_ids)
    annotation_ids = tuple(item.annotation_id for item in annotations)
    invalid_ids: set[str] = set()
    issues: set[str] = set()
    valid_by_boundary: dict[str, list[RefinementBoundaryAnnotationV1]] = {}

    if attempt.end_at is None:
        issues.add("ATTEMPT_INTERVAL_IS_OPEN")

    for annotation in annotations:
        annotation_id = annotation.annotation_id
        boundary_id = annotation.refinement_boundary_id
        boundary = boundary_by_id.get(boundary_id)
        reason: str | None = None
        if boundary is None:
            reason = "UNKNOWN_BOUNDARY"
        elif boundary.progress_step_id != attempt.progress_step_id:
            reason = "BOUNDARY_STEP_MISMATCH"
        elif annotation.clock_domain_id != timeline.clock_domain_id:
            reason = "FOREIGN_CLOCK"
        elif (
            attempt.end_at is None
            or annotation.effective_at < attempt.start_at
            or annotation.effective_at > attempt.end_at
        ):
            reason = "OUTSIDE_ATTEMPT_INTERVAL"
        if reason is not None:
            invalid_ids.add(annotation_id)
            issues.add(f"ANNOTATION:{annotation_id}:{reason}")
            continue
        valid_by_boundary.setdefault(boundary_id, []).append(annotation)

    conflicting_ids = tuple(
        boundary_id
        for boundary_id in required_ids
        if len(valid_by_boundary.get(boundary_id, ())) > 1
    )
    for boundary_id in conflicting_ids:
        issues.add(f"BOUNDARY:{boundary_id}:DUPLICATE_ANNOTATIONS")

    unique_in_profile_order: list[RefinementBoundaryAnnotationV1] = []
    for boundary_id in required_ids:
        matches = valid_by_boundary.get(boundary_id, ())
        if len(matches) == 1:
            unique_in_profile_order.append(matches[0])
    for earlier, later in zip(
        unique_in_profile_order,
        unique_in_profile_order[1:],
    ):
        if later.effective_at < earlier.effective_at:
            earlier_id = earlier.annotation_id
            later_id = later.annotation_id
            invalid_ids.update((earlier_id, later_id))
            issues.add(
                f"ANNOTATION_ORDER:{earlier_id}:{later_id}:DECREASES"
            )

    observed_ids = tuple(
        boundary_id
        for boundary_id in required_ids
        if valid_by_boundary.get(boundary_id)
    )
    missing_ids = tuple(
        boundary_id
        for boundary_id in required_ids
        if boundary_id not in set(observed_ids)
    )

    if issues and (
        invalid_ids or "ATTEMPT_INTERVAL_IS_OPEN" in issues
    ):
        status = "INVALID"
    elif conflicting_ids:
        status = "CONFLICTED"
    elif not required_ids:
        status = "NOT_APPLICABLE"
    elif not observed_ids:
        status = "UNANNOTATED"
    elif missing_ids:
        status = "PARTIAL"
    elif set(observed_ids) == required_id_set:
        status = "COMPLETE"
    else:  # Defensive: every finite state above should be exhaustive.
        status = "INVALID"
        issues.add("UNCLASSIFIED_COVERAGE_STATE")

    return RefinementAttemptCoverage(
        attempt_id=attempt.attempt_id,
        progress_step_id=attempt.progress_step_id,
        status=status,
        required_boundary_ids=required_ids,
        observed_boundary_ids=observed_ids,
        missing_boundary_ids=missing_ids,
        conflicting_boundary_ids=conflicting_ids,
        annotation_ids=annotation_ids,
        invalid_annotation_ids=tuple(sorted(invalid_ids)),
        issues=tuple(sorted(issues)),
    )


def derive_refinement_annotation_coverage(
    *,
    timeline: ResolvedProgressTimeline,
    progress_profile: TaskProgressProfileV1,
    refinement_profile: TaskRefinementProfileV1,
    annotation_set: SelectedRefinementAnnotationSet,
) -> RefinementAnnotationCoverageView:
    """Derive per-Attempt structural coverage without training-policy guesses."""

    _validate_inputs(
        timeline=timeline,
        progress_profile=progress_profile,
        refinement_profile=refinement_profile,
        annotation_set=annotation_set,
    )
    attempt_by_id = {item.attempt_id: item for item in timeline.attempts}
    if len(attempt_by_id) != len(timeline.attempts):
        raise ProgressCoverageResolutionError(
            "Resolved timeline contains duplicate Attempt identities."
        )

    grouped: dict[str, list[RefinementBoundaryAnnotationV1]] = {}
    unassigned_ids: list[str] = []
    view_issues: list[str] = []
    for annotation in annotation_set.observations:
        attempt_id = annotation.coarse_attempt_id
        if attempt_id not in attempt_by_id:
            unassigned_ids.append(annotation.annotation_id)
            view_issues.append(
                f"ANNOTATION:{annotation.annotation_id}:UNKNOWN_ATTEMPT:"
                f"{attempt_id}"
            )
            continue
        grouped.setdefault(attempt_id, []).append(annotation)

    attempts = tuple(
        _attempt_coverage(
            attempt=attempt,
            timeline=timeline,
            refinement_profile=refinement_profile,
            annotations=tuple(grouped.get(attempt.attempt_id, ())),
        )
        for attempt in timeline.attempts
    )
    return RefinementAnnotationCoverageView(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        source_progress_manifest_hash=timeline.source_progress_manifest_hash,
        source_annotation_manifest_hash=(
            annotation_set.source_annotation_manifest_hash
        ),
        task_refinement_profile_hash=(
            refinement_profile.semantic_content_hash
        ),
        annotation_set_id=annotation_set.annotation_set_id,
        clock_domain_id=timeline.clock_domain_id,
        attempts=attempts,
        unassigned_annotation_ids=tuple(sorted(unassigned_ids)),
        issues=tuple(sorted(view_issues)),
    )


__all__ = [
    "REFINEMENT_COVERAGE_STATUSES",
    "ProgressCoverageResolutionError",
    "RefinementAnnotationCoverageView",
    "RefinementAttemptCoverage",
    "derive_refinement_annotation_coverage",
]
