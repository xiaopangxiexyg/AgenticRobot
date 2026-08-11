"""Resolve ordered fine stages and rebuild efficient training intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stage_annotations import (
    StageAnnotationBundleV1,
    StageOccurrenceAnnotationV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
    validate_task_stage_profile,
)

from .progress_samples import project_progress_timeline_to_samples
from .progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from .retrospective_progress import (
    ResolvedRetrospectiveAttempt,
    ResolvedRetrospectiveProgressWindow,
)
from .sample_axis import (
    EpisodeSampleAxis,
    SampleIndexInterval,
)


STAGE_ANNOTATION_COVERAGE_STATUSES_V1 = frozenset(
    {"UNANNOTATED", "PARTIAL", "COMPLETE", "EXCLUDED_EPISODE"}
)


class ProgressStageResolutionError(ValueError):
    """Raised when stage annotations cannot form one ordered interpretation."""


@dataclass(frozen=True)
class ResolvedStageOccurrence:
    annotation_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    occurrence_index_in_attempt: int
    start_at: float
    end_at: float
    outcome: str
    source_failure_fact_id: str | None
    sample_interval: SampleIndexInterval


@dataclass(frozen=True)
class StageTrainingExclusion:
    exclusion_kind: str
    sample_interval: SampleIndexInterval
    source_annotation_id: str
    source_failure_fact_id: str | None = None
    progress_step_id: str | None = None
    stage_id: str | None = None
    coarse_attempt_id: str | None = None


@dataclass(frozen=True)
class StageTrainingIndex:
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    task_stage_profile_hash: str
    annotation_bundle_hash: str
    clock_domain_id: str
    sample_count: int
    coverage_status: str
    annotated_sample_count: int
    unannotated_sample_count: int
    occurrences: tuple[ResolvedStageOccurrence, ...]
    exclusions: tuple[StageTrainingExclusion, ...]


@dataclass(frozen=True)
class _EffectiveAttempt:
    source: ResolvedProgressAttempt | ResolvedRetrospectiveAttempt
    start_at: float
    end_at: float | None


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def _intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return left_start < right_end and right_start < left_end


def _timestamp_is_covered_by_abnormal(
    timestamp: float | None,
    abnormal_intervals: tuple[object, ...],
) -> bool:
    if timestamp is None:
        return False
    return any(
        (
            abnormal.start_at <= timestamp <= abnormal.end_at
            or _close(timestamp, abnormal.start_at)
            or _close(timestamp, abnormal.end_at)
        )
        for abnormal in abnormal_intervals
    )


def _effective_attempts(
    *,
    timeline: ResolvedProgressTimeline | ResolvedRetrospectiveProgressWindow,
    annotations: StageAnnotationBundleV1,
) -> tuple[_EffectiveAttempt, ...]:
    starts = {item.attempt_id: item.start_at for item in timeline.attempts}
    ends = {item.attempt_id: item.end_at for item in timeline.attempts}
    transition_by_fact = {
        item.advance_fact_id: item for item in timeline.transitions
    }
    seen: set[str] = set()
    for override in annotations.progress_boundary_overrides:
        transition = transition_by_fact.get(override.source_advance_fact_id)
        if transition is None:
            raise ProgressStageResolutionError(
                "Progress boundary override references an unknown or undone "
                "Advance fact."
            )
        if override.source_advance_fact_id in seen:
            raise ProgressStageResolutionError(
                "Only one boundary override may target an Advance fact."
            )
        if override.clock_domain_id != timeline.clock_domain_id:
            raise ProgressStageResolutionError(
                "Progress boundary override uses a foreign clock."
            )
        seen.add(override.source_advance_fact_id)
        corrected = override.corrected_effective_at
        starts[transition.to_attempt_id] = corrected
        ends[transition.from_attempt_id] = corrected
    effective = tuple(
        _EffectiveAttempt(
            source=attempt,
            start_at=starts[attempt.attempt_id],
            end_at=ends[attempt.attempt_id],
        )
        for attempt in timeline.attempts
    )
    for item in effective:
        if item.end_at is not None and item.end_at < item.start_at:
            raise ProgressStageResolutionError(
                "A corrected coarse boundary inverted an Attempt interval."
            )
    for earlier, later in zip(effective, effective[1:]):
        if later.start_at < earlier.start_at:
            raise ProgressStageResolutionError(
                "Corrected coarse boundaries move Attempt time backwards."
            )
    return effective


def _entry_stage_id(
    *,
    progress_profile: TaskProgressProfileV1,
    progress_binding: ProgressEpisodeBindingV1,
    stage_profile: TaskStageProfileV1,
    stage_entry_binding: ProgressStageEntryBindingV1 | None,
    timeline: ResolvedProgressTimeline | ResolvedRetrospectiveProgressWindow,
) -> str:
    stages = stage_profile.stages_for_step(progress_binding.entry_step_id)
    if not stages:
        raise ProgressStageResolutionError(
            "The coarse entry step has no ordered stages."
        )
    if stage_entry_binding is None:
        return stages[0].stage_id
    if (
        stage_entry_binding.episode_capture_identity_hash
        != timeline.episode_capture_identity_hash
    ):
        raise ProgressStageResolutionError(
            "Stage entry and resolved Episode identity differ."
        )
    validate_progress_stage_entry_binding(
        progress_profile=progress_profile,
        progress_entry_step_id=progress_binding.entry_step_id,
        stage_profile=stage_profile,
        stage_binding=stage_entry_binding,
    )
    return stage_entry_binding.entry_stage_id


def _validate_occurrence_basics(
    *,
    occurrence: StageOccurrenceAnnotationV1,
    attempt: _EffectiveAttempt,
    stage_profile: TaskStageProfileV1,
    timeline: ResolvedProgressTimeline | ResolvedRetrospectiveProgressWindow,
    abnormal_intervals: tuple[object, ...],
) -> None:
    if occurrence.clock_domain_id != timeline.clock_domain_id:
        raise ProgressStageResolutionError(
            "Stage occurrence uses a foreign Episode clock."
        )
    if occurrence.progress_step_id != attempt.source.progress_step_id:
        raise ProgressStageResolutionError(
            "Stage occurrence progress step differs from its coarse Attempt."
        )
    stage = stage_profile.stage_by_id.get(occurrence.stage_id)
    if stage is None or stage.progress_step_id != occurrence.progress_step_id:
        raise ProgressStageResolutionError(
            "Stage occurrence references an unknown or foreign stage."
        )
    if occurrence.start_at < attempt.start_at:
        raise ProgressStageResolutionError(
            "Stage occurrence begins before its coarse Attempt."
        )
    if (
        attempt.end_at is not None
        and occurrence.end_at > attempt.end_at
        and not _close(occurrence.end_at, attempt.end_at)
    ):
        raise ProgressStageResolutionError(
            "Stage occurrence ends after its coarse Attempt."
        )
    for abnormal in abnormal_intervals:
        if _intervals_overlap(
            occurrence.start_at,
            occurrence.end_at,
            abnormal.start_at,
            abnormal.end_at,
        ):
            raise ProgressStageResolutionError(
                "A stage occurrence cannot overlap an abnormal interval."
            )
    if occurrence.outcome == "TRUNCATED_BY_ABNORMAL" and not any(
        _close(occurrence.start_at, abnormal.end_at)
        or _close(occurrence.end_at, abnormal.start_at)
        for abnormal in abnormal_intervals
    ):
        raise ProgressStageResolutionError(
            "TRUNCATED_BY_ABNORMAL must touch an abnormal interval boundary."
        )


def _covered_count(
    intervals: tuple[SampleIndexInterval, ...],
) -> int:
    nonempty = sorted(
        (item for item in intervals if not item.empty),
        key=lambda item: (
            item.sample_index_begin,
            item.sample_index_end_exclusive,
        ),
    )
    if not nonempty:
        return 0
    total = 0
    begin = nonempty[0].sample_index_begin
    end = nonempty[0].sample_index_end_exclusive
    for item in nonempty[1:]:
        if item.sample_index_begin <= end:
            end = max(end, item.sample_index_end_exclusive)
        else:
            total += end - begin
            begin = item.sample_index_begin
            end = item.sample_index_end_exclusive
    return total + end - begin


def _intersection_count(
    required: tuple[SampleIndexInterval, ...],
    covered: tuple[SampleIndexInterval, ...],
) -> int:
    """Count covered sample indices only inside required Attempt intervals."""

    normalized_required = tuple(
        sorted(
            (item for item in required if not item.empty),
            key=lambda item: (
                item.sample_index_begin,
                item.sample_index_end_exclusive,
            ),
        )
    )
    normalized_covered = tuple(
        sorted(
            (item for item in covered if not item.empty),
            key=lambda item: (
                item.sample_index_begin,
                item.sample_index_end_exclusive,
            ),
        )
    )
    intersections: list[SampleIndexInterval] = []
    required_index = 0
    covered_index = 0
    while (
        required_index < len(normalized_required)
        and covered_index < len(normalized_covered)
    ):
        required_item = normalized_required[required_index]
        covered_item = normalized_covered[covered_index]
        begin = max(
            required_item.sample_index_begin,
            covered_item.sample_index_begin,
        )
        end = min(
            required_item.sample_index_end_exclusive,
            covered_item.sample_index_end_exclusive,
        )
        if begin < end:
            intersections.append(SampleIndexInterval(begin, end))
        if (
            required_item.sample_index_end_exclusive
            <= covered_item.sample_index_end_exclusive
        ):
            required_index += 1
        else:
            covered_index += 1
    return _covered_count(tuple(intersections))


def resolve_stage_training_index(
    *,
    progress_profile: TaskProgressProfileV1,
    progress_binding: ProgressEpisodeBindingV1,
    stage_profile: TaskStageProfileV1,
    stage_entry_binding: ProgressStageEntryBindingV1 | None,
    timeline: ResolvedProgressTimeline | ResolvedRetrospectiveProgressWindow,
    annotations: StageAnnotationBundleV1,
    sample_axis: EpisodeSampleAxis,
) -> StageTrainingIndex:
    """Validate strict stage order and derive loader-ready sample intervals."""

    validate_task_stage_profile(
        progress_profile=progress_profile,
        stage_profile=stage_profile,
    )
    if not timeline.sealed or timeline.source_progress_manifest_hash is None:
        raise ProgressStageResolutionError(
            "Stage annotation requires a sealed progress timeline."
        )
    if (
        timeline.episode_capture_identity_hash
        != progress_binding.episode_capture_identity_hash
        or timeline.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressStageResolutionError(
            "Stage resolver inputs do not bind the same Episode/profile."
        )
    if timeline.clock_domain_id != sample_axis.clock_domain_id:
        raise ProgressStageResolutionError(
            "Stage timeline and sample axis use different clocks."
        )
    entry_stage_id = _entry_stage_id(
        progress_profile=progress_profile,
        progress_binding=progress_binding,
        stage_profile=stage_profile,
        stage_entry_binding=stage_entry_binding,
        timeline=timeline,
    )
    attempts = _effective_attempts(
        timeline=timeline,
        annotations=annotations,
    )
    attempt_by_id = {item.source.attempt_id: item for item in attempts}
    if len(attempt_by_id) != len(attempts):
        raise ProgressStageResolutionError("Duplicate coarse Attempt identity.")
    occurrences_by_attempt: dict[str, list[StageOccurrenceAnnotationV1]] = {}
    for occurrence in annotations.occurrences:
        attempt = attempt_by_id.get(occurrence.coarse_attempt_id)
        if attempt is None:
            raise ProgressStageResolutionError(
                "Stage occurrence references an unknown coarse Attempt."
            )
        _validate_occurrence_basics(
            occurrence=occurrence,
            attempt=attempt,
            stage_profile=stage_profile,
            timeline=timeline,
            abnormal_intervals=annotations.abnormal_intervals,
        )
        occurrences_by_attempt.setdefault(
            occurrence.coarse_attempt_id,
            [],
        ).append(occurrence)

    resolved_occurrences: list[ResolvedStageOccurrence] = []
    exclusions: list[StageTrainingExclusion] = []
    stage_index_by_step = {
        step.progress_step_id: {
            stage.stage_id: index
            for index, stage in enumerate(
                stage_profile.stages_for_step(step.progress_step_id)
            )
        }
        for step in progress_profile.steps
    }
    expected_by_step = {
        step.progress_step_id: 0 for step in progress_profile.steps
    }
    expected_by_step[progress_binding.entry_step_id] = (
        stage_index_by_step[progress_binding.entry_step_id][entry_stage_id]
    )
    first_annotated = min(
        annotations.occurrences,
        key=lambda item: (
            item.start_at,
            item.end_at,
            item.annotation_id,
        ),
        default=None,
    )
    if (
        first_annotated is not None
        and first_annotated.progress_step_id
        == progress_binding.entry_step_id
    ):
        first_stage_index = stage_index_by_step[
            progress_binding.entry_step_id
        ][first_annotated.stage_id]
        declared_entry_index = expected_by_step[
            progress_binding.entry_step_id
        ]
        if first_stage_index >= declared_entry_index:
            # A post-capture annotator may deliberately omit a leading stage
            # prefix.  The first remaining stage identity is explicit evidence
            # of the truncated prefix; no later jump or backward move receives
            # this exception.
            expected_by_step[
                progress_binding.entry_step_id
            ] = first_stage_index
    previous_occurrence: StageOccurrenceAnnotationV1 | None = None
    step_has_abnormal: set[str] = set()
    whole_episode_excluded = bool(
        annotations.episode_abnormal is not None
        or annotations.episode_review_request is not None
    )
    transitioned_attempt_ids = frozenset(
        item.from_attempt_id for item in timeline.transitions
    )
    final_attempt_id = attempts[-1].source.attempt_id if attempts else None

    for attempt in attempts:
        values = sorted(
            occurrences_by_attempt.get(attempt.source.attempt_id, ()),
            key=lambda item: (
                item.start_at,
                item.end_at,
                item.annotation_id,
            ),
        )
        indices = tuple(item.occurrence_index_in_attempt for item in values)
        if indices and indices != tuple(range(1, len(indices) + 1)):
            raise ProgressStageResolutionError(
                "Stage occurrence indices must be contiguous within an Attempt."
            )
        for earlier, later in zip(values, values[1:]):
            if later.start_at < earlier.end_at:
                raise ProgressStageResolutionError(
                    "Stage occurrences cannot overlap within an Attempt."
                )
        for abnormal in annotations.abnormal_intervals:
            end = attempt.end_at
            if end is not None and _intervals_overlap(
                attempt.start_at,
                end,
                abnormal.start_at,
                abnormal.end_at,
            ):
                step_has_abnormal.add(attempt.source.progress_step_id)

        expected = expected_by_step[attempt.source.progress_step_id]
        stage_count = len(
            stage_profile.stages_for_step(
                attempt.source.progress_step_id
            )
        )
        failure_seen = False
        for occurrence in values:
            ordinal_index = stage_index_by_step[
                attempt.source.progress_step_id
            ][occurrence.stage_id]
            abnormal_between = bool(
                any(
                    interval.end_at <= occurrence.start_at
                    and (
                        previous_occurrence is None
                        or interval.start_at >= previous_occurrence.end_at
                    )
                    for interval in annotations.abnormal_intervals
                )
            )
            if not whole_episode_excluded:
                if abnormal_between and ordinal_index >= expected:
                    expected = ordinal_index
                if ordinal_index != expected:
                    raise ProgressStageResolutionError(
                        "Fine stages must follow the profile order; a failed "
                        "stage may repeat, but completed stages cannot repeat "
                        "or move backwards."
                    )
                if failure_seen:
                    raise ProgressStageResolutionError(
                        "No stage occurrence may follow Failure in one Attempt."
                    )
                if occurrence.outcome == "SUCCESS":
                    expected += 1
                elif occurrence.outcome == "FAILURE":
                    failure_seen = True
            if occurrence.outcome == "FAILURE":
                source_failure = attempt.source.failure_fact_id
                if (
                    occurrence.source_failure_fact_id is not None
                    and source_failure is not None
                ):
                    if (
                        occurrence.source_failure_fact_id
                        != source_failure
                        or attempt.end_at is None
                        or not _close(occurrence.end_at, attempt.end_at)
                    ):
                        raise ProgressStageResolutionError(
                            "A captured Failure occurrence must preserve the "
                            "original Failure fact and timestamp."
                        )
                elif (
                    occurrence.source_failure_fact_id is not None
                    and source_failure is None
                ):
                    raise ProgressStageResolutionError(
                        "Posthoc tail Failure references a foreign capture fact."
                    )
                elif (
                    occurrence.source_failure_fact_id is None
                    and source_failure is None
                    and (
                        attempt.source.attempt_id != final_attempt_id
                        or attempt.source.attempt_id
                        in transitioned_attempt_ids
                        or attempt.end_at is None
                        or not _close(occurrence.end_at, attempt.end_at)
                    )
                ):
                    raise ProgressStageResolutionError(
                        "A Failure without a capture fact is only permitted on "
                        "the final captured stage occurrence."
                    )
            end_is_inclusive = bool(
                occurrence.outcome == "FAILURE"
                or (
                    attempt.end_at is not None
                    and _close(occurrence.end_at, attempt.end_at)
                    and attempt.source.attempt_id
                    not in transitioned_attempt_ids
                )
            )
            interval = (
                sample_axis.closed_time_interval(
                    start_at=occurrence.start_at,
                    end_at=occurrence.end_at,
                )
                if end_is_inclusive
                else sample_axis.half_open_time_interval(
                    start_at=occurrence.start_at,
                    end_at=occurrence.end_at,
                )
            )
            resolved_occurrences.append(
                ResolvedStageOccurrence(
                    annotation_id=occurrence.annotation_id,
                    coarse_attempt_id=occurrence.coarse_attempt_id,
                    progress_step_id=occurrence.progress_step_id,
                    stage_id=occurrence.stage_id,
                    occurrence_index_in_attempt=(
                        occurrence.occurrence_index_in_attempt
                    ),
                    start_at=occurrence.start_at,
                    end_at=occurrence.end_at,
                    outcome=occurrence.outcome,
                    source_failure_fact_id=(
                        occurrence.source_failure_fact_id
                    ),
                    sample_interval=interval,
                )
            )
            if occurrence.outcome == "FAILURE":
                exclusions.append(
                    StageTrainingExclusion(
                        exclusion_kind="FAILED_STAGE_OCCURRENCE",
                        sample_interval=interval,
                        source_annotation_id=occurrence.annotation_id,
                        source_failure_fact_id=(
                            occurrence.source_failure_fact_id
                        ),
                        progress_step_id=occurrence.progress_step_id,
                        stage_id=occurrence.stage_id,
                        coarse_attempt_id=occurrence.coarse_attempt_id,
                    )
                )
            previous_occurrence = occurrence
        expected_by_step[attempt.source.progress_step_id] = expected

        if (
            not whole_episode_excluded
            and attempt.source.outcome == "FAILURE"
            and not _timestamp_is_covered_by_abnormal(
                attempt.source.failure_reported_at,
                annotations.abnormal_intervals,
            )
        ):
            captured_failures = tuple(
                item
                for item in values
                if item.outcome == "FAILURE"
                and item.source_failure_fact_id
                == attempt.source.failure_fact_id
            )
            if len(captured_failures) != 1:
                raise ProgressStageResolutionError(
                    "Every captured Failure Attempt requires exactly one "
                    "stage Failure occurrence."
                )
        if (
            not whole_episode_excluded
            and attempt.source.outcome == "SUCCESS"
            and attempt.source.progress_step_id not in step_has_abnormal
            and not (
                values
                and values[-1].outcome == "FAILURE"
                and values[-1].source_failure_fact_id is None
                and attempt.source.attempt_id == final_attempt_id
                and attempt.source.attempt_id not in transitioned_attempt_ids
            )
            and expected != stage_count
        ):
            raise ProgressStageResolutionError(
                "A successful coarse progress step requires every remaining "
                "fine stage to succeed exactly once."
            )

    for abnormal in annotations.abnormal_intervals:
        if abnormal.clock_domain_id != timeline.clock_domain_id:
            raise ProgressStageResolutionError(
                "Abnormal interval uses a foreign Episode clock."
            )
        interval = sample_axis.half_open_time_interval(
            start_at=abnormal.start_at,
            end_at=abnormal.end_at,
        )
        exclusions.append(
            StageTrainingExclusion(
                exclusion_kind="ABNORMAL_INTERVAL",
                sample_interval=interval,
                source_annotation_id=abnormal.annotation_id,
            )
        )

    projected = project_progress_timeline_to_samples(
        timeline=timeline,
        sample_axis=sample_axis,
    )
    attempt_intervals = tuple(item.interval for item in projected.attempts)
    annotated_intervals = tuple(
        item.sample_interval for item in resolved_occurrences
    ) + tuple(
        item.sample_interval
        for item in exclusions
        if item.exclusion_kind == "ABNORMAL_INTERVAL"
    )
    required_count = _covered_count(attempt_intervals)
    annotated_count = _intersection_count(
        attempt_intervals,
        annotated_intervals,
    )
    if annotations.episode_abnormal is not None:
        coverage_status = "EXCLUDED_EPISODE"
        exclusions = [
            StageTrainingExclusion(
                exclusion_kind="WHOLE_EPISODE_ABNORMAL",
                sample_interval=SampleIndexInterval(
                    0,
                    sample_axis.sample_count,
                ),
                source_annotation_id=(
                    annotations.episode_abnormal.annotation_id
                ),
            )
        ]
        unannotated_count = 0
    elif annotations.episode_review_request is not None:
        coverage_status = "EXCLUDED_EPISODE"
        exclusions = [
            StageTrainingExclusion(
                exclusion_kind="WHOLE_EPISODE_REVIEW_PENDING",
                sample_interval=SampleIndexInterval(
                    0,
                    sample_axis.sample_count,
                ),
                source_annotation_id=(
                    annotations.episode_review_request.annotation_id
                ),
            )
        ]
        unannotated_count = 0
    elif not resolved_occurrences and not annotations.abnormal_intervals:
        coverage_status = "UNANNOTATED"
        unannotated_count = required_count
    else:
        unannotated_count = max(0, required_count - annotated_count)
        coverage_status = (
            "COMPLETE" if unannotated_count == 0 else "PARTIAL"
        )
    return StageTrainingIndex(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        source_progress_manifest_hash=(
            timeline.source_progress_manifest_hash
        ),
        task_stage_profile_hash=stage_profile.semantic_content_hash,
        annotation_bundle_hash=annotations.semantic_content_hash,
        clock_domain_id=timeline.clock_domain_id,
        sample_count=sample_axis.sample_count,
        coverage_status=coverage_status,
        annotated_sample_count=annotated_count,
        unannotated_sample_count=unannotated_count,
        occurrences=tuple(resolved_occurrences),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item.sample_interval.sample_index_begin,
                    item.sample_interval.sample_index_end_exclusive,
                    item.exclusion_kind,
                ),
            )
        ),
    )


__all__ = [
    "STAGE_ANNOTATION_COVERAGE_STATUSES_V1",
    "ProgressStageResolutionError",
    "ResolvedStageOccurrence",
    "StageTrainingExclusion",
    "StageTrainingIndex",
    "resolve_stage_training_index",
]
