"""Read-only coarse Failure exclusions over a sealed capture timeline.

This first derived view uses only capture-time coarse progress facts.  It does
not interpret refinement annotations or retrospective correction overlays and
does not define a persisted training-exclusion schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from .progress_samples import project_progress_timeline_to_samples
from .progress_timeline import ResolvedProgressAttempt, ResolvedProgressTimeline
from .sample_axis import EpisodeSampleAxis, SampleIndexInterval


class ProgressExclusionResolutionError(ValueError):
    """Raised when a coarse Failure exclusion cannot be derived safely."""


_CAPTURE_CAUSAL_FAILURE_CUT_SEMANTICS = frozenset(
    {"OPERATOR_REPORT_TIME", "PROMOTED_HOLD_TIME"}
)


@dataclass(frozen=True)
class CoarseFailureTrainingExclusion:
    source_failure_fact_id: str
    attempt_id: str
    progress_step_id: str
    attempt_start_at: float
    failure_cut_at: float
    sample_interval: SampleIndexInterval


@dataclass(frozen=True)
class CoarseFailureTrainingView:
    episode_capture_identity_hash: str
    progress_episode_binding_hash: str
    source_progress_manifest_hash: str
    clock_domain_id: str
    sample_count: int
    exclusions: tuple[CoarseFailureTrainingExclusion, ...]


def _failure_cut(
    attempt: ResolvedProgressAttempt,
) -> tuple[str, float] | None:
    failure_values = (
        attempt.failure_fact_id,
        attempt.failure_reported_at,
        attempt.failure_timing_semantics,
    )
    if attempt.outcome != "FAILURE":
        if any(value is not None for value in failure_values):
            raise ProgressExclusionResolutionError(
                f"Non-Failure Attempt {attempt.attempt_id!r} carries "
                "Failure metadata."
            )
        return None
    if (
        attempt.end_at is None
        or attempt.failure_fact_id is None
        or attempt.failure_reported_at is None
        or attempt.failure_timing_semantics
        not in _CAPTURE_CAUSAL_FAILURE_CUT_SEMANTICS
    ):
        raise ProgressExclusionResolutionError(
            f"Failure Attempt {attempt.attempt_id!r} is missing its online Failure cut."
        )
    if attempt.failure_reported_at != attempt.end_at:
        raise ProgressExclusionResolutionError(
            f"Failure Attempt {attempt.attempt_id!r} has divergent report and cut times."
        )
    return attempt.failure_fact_id, attempt.end_at


def derive_coarse_failure_training_view(
    *,
    timeline: ResolvedProgressTimeline,
    sample_axis: EpisodeSampleAxis,
) -> CoarseFailureTrainingView:
    """Exclude every capture-time failed Attempt from start through its cut.

    The input must be the sealed canonical capture-time timeline.  A later
    refinement/correction resolver will own alternative starts or corrected
    Failure cuts and must carry its own resolution identity.
    """

    if not timeline.sealed or timeline.phase != "TERMINATED":
        raise ProgressExclusionResolutionError(
            "Coarse training exclusions require a sealed terminated timeline."
        )
    if timeline.resolution_kind != "CAPTURE_CAUSAL":
        raise ProgressExclusionResolutionError(
            "Coarse training exclusions require the capture-time causal timeline."
        )
    if timeline.source_progress_manifest_hash is None:
        raise ProgressExclusionResolutionError(
            "Coarse training exclusions require sealed manifest provenance."
        )
    projected = project_progress_timeline_to_samples(
        timeline=timeline,
        sample_axis=sample_axis,
    )
    projected_by_attempt_id = {
        item.attempt_id: item for item in projected.attempts
    }
    if len(projected_by_attempt_id) != len(projected.attempts):
        raise ProgressExclusionResolutionError(
            "Resolved timeline contains duplicate Attempt identities."
        )

    seen_failure_fact_ids: set[str] = set()
    exclusions: list[CoarseFailureTrainingExclusion] = []
    for attempt in timeline.attempts:
        failure_cut = _failure_cut(attempt)
        if failure_cut is None:
            continue
        failure_fact_id, failure_cut_at = failure_cut
        if failure_fact_id in seen_failure_fact_ids:
            raise ProgressExclusionResolutionError(
                "One Failure fact cannot close multiple Attempts."
            )
        seen_failure_fact_ids.add(failure_fact_id)
        projected_attempt = projected_by_attempt_id.get(attempt.attempt_id)
        if projected_attempt is None:
            raise ProgressExclusionResolutionError(
                f"Failure Attempt {attempt.attempt_id!r} has no sample projection."
            )
        if not projected_attempt.end_boundary_inclusive:
            raise ProgressExclusionResolutionError(
                f"Failure Attempt {attempt.attempt_id!r} excludes its Failure cut."
            )
        exclusions.append(
            CoarseFailureTrainingExclusion(
                source_failure_fact_id=failure_fact_id,
                attempt_id=attempt.attempt_id,
                progress_step_id=attempt.progress_step_id,
                attempt_start_at=attempt.start_at,
                failure_cut_at=failure_cut_at,
                sample_interval=projected_attempt.interval,
            )
        )

    return CoarseFailureTrainingView(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        progress_episode_binding_hash=timeline.progress_episode_binding_hash,
        source_progress_manifest_hash=timeline.source_progress_manifest_hash,
        clock_domain_id=timeline.clock_domain_id,
        sample_count=projected.sample_count,
        exclusions=tuple(exclusions),
    )


__all__ = [
    "CoarseFailureTrainingExclusion",
    "CoarseFailureTrainingView",
    "ProgressExclusionResolutionError",
    "derive_coarse_failure_training_view",
]
