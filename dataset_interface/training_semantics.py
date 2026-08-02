"""Read-time semantic selection and action-window planning for training.

This module composes already-verified capture Progress, ordered-stage
annotations, Failure-trigger annotations, and the Episode sample axis.  It is
deliberately a derived dataset interface:

* it never changes capture facts, annotations, or frame payloads;
* it does not own annotation-set selection or dataset lifecycle;
* it keeps padding/drop behavior in :class:`PoseActionChunkContract`;
* it permits windows to cross normal stage and progress-step transitions, but
  never a Failure, abnormal, pause, or explicit continuity boundary.

The resulting objects are provisional Python read views, not persisted dataset
contracts.  A future typed annotation can participate by projecting its point
or interval into the same ``SemanticAnchor`` / ``SemanticSpan`` vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from dataset_contracts.pose_chunk import (
    POSE_CHUNK_TAIL_DROP,
    PoseActionChunkContract,
)
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationV1,
)

from .progress_samples import project_progress_timeline_to_samples
from .progress_stages import StageTrainingIndex
from .progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from .sample_axis import EpisodeSampleAxis, SampleIndexInterval


SEMANTIC_SPAN_ATTEMPT = "PROGRESS_ATTEMPT"
SEMANTIC_SPAN_STAGE = "STAGE_OCCURRENCE"
SEMANTIC_SPAN_FAILURE = "FAILURE_REGION"
SEMANTIC_SPAN_ABNORMAL = "ABNORMAL_REGION"
SEMANTIC_SPAN_RECOVERY = "RECOVERY_ATTEMPT"

SEMANTIC_ANCHOR_PROGRESS_STEP_BEGIN = "PROGRESS_STEP_BEGIN"
SEMANTIC_ANCHOR_STAGE_BEGIN = "STAGE_BEGIN"
SEMANTIC_ANCHOR_FAILURE_REGION_BEGIN = "FAILURE_REGION_BEGIN"
SEMANTIC_ANCHOR_FAILURE_TRIGGER = "FAILURE_TRIGGER"
SEMANTIC_ANCHOR_FAILURE_END = "FAILURE_END"
SEMANTIC_ANCHOR_RECOVERY_BEGIN = "RECOVERY_BEGIN"

TRAINING_DOMAIN_NOMINAL = "NOMINAL"
TRAINING_DOMAIN_FAILURE = "FAILURE"
TRAINING_DOMAINS = frozenset(
    {TRAINING_DOMAIN_NOMINAL, TRAINING_DOMAIN_FAILURE}
)

FAILURE_RESOLUTION_REFINED_TRIGGER = "REFINED_TRIGGER"
FAILURE_RESOLUTION_FAILED_STAGE = "FAILED_STAGE_START"
FAILURE_RESOLUTION_FAILED_ATTEMPT = "FAILED_ATTEMPT_START"


class TrainingSemanticResolutionError(ValueError):
    """Raised when semantic artifacts cannot form one training interpretation."""


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


@dataclass(frozen=True)
class SemanticSpan:
    """One typed half-open sample interval with exact source provenance."""

    semantic_kind: str
    sample_interval: SampleIndexInterval
    progress_step_id: str | None = None
    attempt_id: str | None = None
    stage_id: str | None = None
    outcome: str | None = None
    entry_condition: str | None = None
    source_id: str | None = None
    source_failure_fact_id: str | None = None
    resolution_kind: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.semantic_kind).strip().upper()
        if not kind:
            raise TrainingSemanticResolutionError(
                "semantic_kind must not be empty."
            )
        object.__setattr__(self, "semantic_kind", kind)
        for field_name in (
            "progress_step_id",
            "attempt_id",
            "stage_id",
            "outcome",
            "entry_condition",
            "source_id",
            "source_failure_fact_id",
            "resolution_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_optional(getattr(self, field_name)),
            )


@dataclass(frozen=True)
class SemanticAnchor:
    """One typed discrete training anchor on the Episode sample axis."""

    semantic_kind: str
    sample_index: int
    progress_step_id: str | None = None
    attempt_id: str | None = None
    stage_id: str | None = None
    source_id: str | None = None
    source_failure_fact_id: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.semantic_kind).strip().upper()
        if not kind:
            raise TrainingSemanticResolutionError(
                "semantic_kind must not be empty."
            )
        if (
            not isinstance(self.sample_index, int)
            or isinstance(self.sample_index, bool)
            or self.sample_index < 0
        ):
            raise TrainingSemanticResolutionError(
                "sample_index must be a non-negative integer."
            )
        object.__setattr__(self, "semantic_kind", kind)
        for field_name in (
            "progress_step_id",
            "attempt_id",
            "stage_id",
            "source_id",
            "source_failure_fact_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_optional(getattr(self, field_name)),
            )


@dataclass(frozen=True)
class EpisodeTrainingSemanticIndex:
    """One sealed Episode's rebuildable semantic training index."""

    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    clock_domain_id: str
    sample_count: int
    spans: tuple[SemanticSpan, ...]
    anchors: tuple[SemanticAnchor, ...]
    nominal_intervals: tuple[SampleIndexInterval, ...]
    failure_intervals: tuple[SampleIndexInterval, ...]

    def spans_of_kind(self, semantic_kind: str) -> tuple[SemanticSpan, ...]:
        kind = str(semantic_kind).strip().upper()
        return tuple(item for item in self.spans if item.semantic_kind == kind)

    def anchors_of_kind(
        self,
        semantic_kind: str,
    ) -> tuple[SemanticAnchor, ...]:
        kind = str(semantic_kind).strip().upper()
        return tuple(item for item in self.anchors if item.semantic_kind == kind)


@dataclass(frozen=True)
class SemanticWindowQuery:
    """Select window starts by owner subtask and/or semantic anchors.

    ``progress_step_ids`` filters the owner of the observation/start sample.  A
    window may subsequently cross normal stage or progress-step transitions.
    ``start_anchor_kinds`` switches from dense starts to exact typed anchors.
    """

    progress_step_ids: tuple[str, ...] = ()
    attempt_outcomes: tuple[str, ...] = ()
    start_anchor_kinds: tuple[str, ...] = ()
    training_domain: str = TRAINING_DOMAIN_NOMINAL

    def __post_init__(self) -> None:
        for field_name in (
            "progress_step_ids",
            "attempt_outcomes",
            "start_anchor_kinds",
        ):
            values = tuple(
                str(value).strip()
                if field_name == "progress_step_ids"
                else str(value).strip().upper()
                for value in getattr(self, field_name)
            )
            if any(not value for value in values) or len(values) != len(set(values)):
                raise TrainingSemanticResolutionError(
                    f"{field_name} must contain unique non-empty values."
                )
            object.__setattr__(self, field_name, values)
        domain = str(self.training_domain).strip().upper()
        if domain not in TRAINING_DOMAINS:
            raise TrainingSemanticResolutionError(
                f"training_domain must be one of {sorted(TRAINING_DOMAINS)}."
            )
        object.__setattr__(self, "training_domain", domain)


@dataclass(frozen=True)
class SemanticActionWindowLocation:
    """One legal model window start and its semantic-tail length."""

    episode_capture_identity_hash: str
    start_index: int
    valid_action_count: int
    owner_progress_step_id: str
    owner_attempt_id: str
    owner_attempt_outcome: str
    training_domain: str
    containing_interval: SampleIndexInterval
    source_anchor_kind: str | None = None
    source_anchor_id: str | None = None


def _merge_intervals(
    intervals: Iterable[SampleIndexInterval],
) -> tuple[SampleIndexInterval, ...]:
    ordered = sorted(
        (item for item in intervals if not item.empty),
        key=lambda item: (
            item.sample_index_begin,
            item.sample_index_end_exclusive,
        ),
    )
    if not ordered:
        return ()
    result: list[SampleIndexInterval] = []
    begin = ordered[0].sample_index_begin
    end = ordered[0].sample_index_end_exclusive
    for item in ordered[1:]:
        if item.sample_index_begin <= end:
            end = max(end, item.sample_index_end_exclusive)
            continue
        result.append(SampleIndexInterval(begin, end))
        begin = item.sample_index_begin
        end = item.sample_index_end_exclusive
    result.append(SampleIndexInterval(begin, end))
    return tuple(result)


def _subtract_intervals(
    bases: Iterable[SampleIndexInterval],
    exclusions: Iterable[SampleIndexInterval],
) -> tuple[SampleIndexInterval, ...]:
    remaining = list(_merge_intervals(bases))
    for exclusion in _merge_intervals(exclusions):
        updated: list[SampleIndexInterval] = []
        for current in remaining:
            if (
                exclusion.sample_index_end_exclusive
                <= current.sample_index_begin
                or exclusion.sample_index_begin
                >= current.sample_index_end_exclusive
            ):
                updated.append(current)
                continue
            if current.sample_index_begin < exclusion.sample_index_begin:
                updated.append(
                    SampleIndexInterval(
                        current.sample_index_begin,
                        min(
                            exclusion.sample_index_begin,
                            current.sample_index_end_exclusive,
                        ),
                    )
                )
            if exclusion.sample_index_end_exclusive < current.sample_index_end_exclusive:
                updated.append(
                    SampleIndexInterval(
                        max(
                            exclusion.sample_index_end_exclusive,
                            current.sample_index_begin,
                        ),
                        current.sample_index_end_exclusive,
                    )
                )
        remaining = updated
    return tuple(item for item in remaining if not item.empty)


def _split_intervals(
    intervals: Iterable[SampleIndexInterval],
    break_indices: Sequence[int],
    *,
    sample_count: int,
) -> tuple[SampleIndexInterval, ...]:
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in break_indices
    ):
        raise TrainingSemanticResolutionError(
            "continuity_break_indices must contain integers."
        )
    breaks = tuple(sorted(set(break_indices)))
    if any(value <= 0 or value >= sample_count for value in breaks):
        raise TrainingSemanticResolutionError(
            "continuity_break_indices must lie strictly inside the sample axis."
        )
    result: list[SampleIndexInterval] = []
    for interval in intervals:
        cuts = tuple(
            value
            for value in breaks
            if interval.sample_index_begin < value < interval.sample_index_end_exclusive
        )
        begin = interval.sample_index_begin
        for cut in cuts:
            result.append(SampleIndexInterval(begin, cut))
            begin = cut
        result.append(SampleIndexInterval(begin, interval.sample_index_end_exclusive))
    return tuple(item for item in result if not item.empty)


def _attempt_by_id(
    timeline: ResolvedProgressTimeline,
) -> dict[str, ResolvedProgressAttempt]:
    result = {item.attempt_id: item for item in timeline.attempts}
    if len(result) != len(timeline.attempts):
        raise TrainingSemanticResolutionError("Duplicate Attempt identity.")
    return result


def _validate_stage_index(
    *,
    timeline: ResolvedProgressTimeline,
    sample_axis: EpisodeSampleAxis,
    stage_index: StageTrainingIndex,
) -> None:
    if (
        stage_index.episode_capture_identity_hash
        != timeline.episode_capture_identity_hash
        or stage_index.source_progress_manifest_hash
        != timeline.source_progress_manifest_hash
        or stage_index.clock_domain_id != sample_axis.clock_domain_id
        or stage_index.sample_count != sample_axis.sample_count
    ):
        raise TrainingSemanticResolutionError(
            "Stage index does not bind the same sealed Episode/sample axis."
        )


def build_episode_training_semantic_index(
    *,
    timeline: ResolvedProgressTimeline,
    sample_axis: EpisodeSampleAxis,
    stage_index: StageTrainingIndex | None = None,
    failure_triggers: Sequence[FailureTriggerAnnotationV1] = (),
    episode_entry_condition: str = "NOMINAL",
    continuity_break_indices: Sequence[int] = (),
) -> EpisodeTrainingSemanticIndex:
    """Compose one sealed Episode into queryable semantic spans and anchors.

    A ``CAPTURE_DEFAULT`` Failure trigger is intentionally treated as missing:
    it records the Failure end, not a reviewed onset.  The fallback is the
    linked failed-stage start, then the failed-Attempt start.
    """

    if not timeline.sealed or timeline.source_progress_manifest_hash is None:
        raise TrainingSemanticResolutionError(
            "Training semantics require a sealed progress timeline."
        )
    if timeline.clock_domain_id != sample_axis.clock_domain_id:
        raise TrainingSemanticResolutionError(
            "Progress timeline and sample axis use different clocks."
        )
    entry_condition = str(episode_entry_condition).strip().upper()
    if entry_condition not in {"NOMINAL", "RECOVERY_FROM_FAILURE"}:
        raise TrainingSemanticResolutionError(
            "Unsupported episode_entry_condition."
        )
    if stage_index is not None:
        _validate_stage_index(
            timeline=timeline,
            sample_axis=sample_axis,
            stage_index=stage_index,
        )

    projected = project_progress_timeline_to_samples(
        timeline=timeline,
        sample_axis=sample_axis,
    )
    attempts_by_id = _attempt_by_id(timeline)
    attempt_samples_by_id = {item.attempt_id: item for item in projected.attempts}
    spans: list[SemanticSpan] = []
    anchors: list[SemanticAnchor] = []
    for item in projected.attempts:
        attempt = attempts_by_id[item.attempt_id]
        spans.append(
            SemanticSpan(
                semantic_kind=SEMANTIC_SPAN_ATTEMPT,
                sample_interval=item.interval,
                progress_step_id=item.progress_step_id,
                attempt_id=item.attempt_id,
                outcome=item.outcome,
                source_id=attempt.start_fact_id,
            )
        )
        if not item.interval.empty:
            anchors.append(
                SemanticAnchor(
                    semantic_kind=SEMANTIC_ANCHOR_PROGRESS_STEP_BEGIN,
                    sample_index=item.interval.sample_index_begin,
                    progress_step_id=item.progress_step_id,
                    attempt_id=item.attempt_id,
                    source_id=attempt.start_fact_id,
                )
            )

    abnormal_intervals: list[SampleIndexInterval] = []
    failed_stage_by_fact: dict[str, SemanticSpan] = {}
    if stage_index is not None:
        for occurrence in stage_index.occurrences:
            span = SemanticSpan(
                semantic_kind=SEMANTIC_SPAN_STAGE,
                sample_interval=occurrence.sample_interval,
                progress_step_id=occurrence.progress_step_id,
                attempt_id=occurrence.coarse_attempt_id,
                stage_id=occurrence.stage_id,
                outcome=occurrence.outcome,
                source_id=occurrence.annotation_id,
                source_failure_fact_id=occurrence.source_failure_fact_id,
            )
            spans.append(span)
            if not occurrence.sample_interval.empty:
                anchors.append(
                    SemanticAnchor(
                        semantic_kind=SEMANTIC_ANCHOR_STAGE_BEGIN,
                        sample_index=occurrence.sample_interval.sample_index_begin,
                        progress_step_id=occurrence.progress_step_id,
                        attempt_id=occurrence.coarse_attempt_id,
                        stage_id=occurrence.stage_id,
                        source_id=occurrence.annotation_id,
                        source_failure_fact_id=occurrence.source_failure_fact_id,
                    )
                )
            if (
                occurrence.outcome == "FAILURE"
                and occurrence.source_failure_fact_id is not None
            ):
                if occurrence.source_failure_fact_id in failed_stage_by_fact:
                    raise TrainingSemanticResolutionError(
                        "A Failure fact maps to more than one failed stage."
                    )
                failed_stage_by_fact[occurrence.source_failure_fact_id] = span
        for exclusion in stage_index.exclusions:
            if exclusion.exclusion_kind not in {
                "ABNORMAL_INTERVAL",
                "WHOLE_EPISODE_ABNORMAL",
            }:
                # The old stage view conservatively excludes a whole failed
                # occurrence.  This composed view replaces that policy with a
                # refined trigger/stage/Attempt fallback below.
                continue
            abnormal_intervals.append(exclusion.sample_interval)
            spans.append(
                SemanticSpan(
                    semantic_kind=SEMANTIC_SPAN_ABNORMAL,
                    sample_interval=exclusion.sample_interval,
                    source_id=exclusion.source_annotation_id,
                    resolution_kind=exclusion.exclusion_kind,
                )
            )

    trigger_by_failure: dict[str, FailureTriggerAnnotationV1] = {}
    for trigger in failure_triggers:
        failure_id = trigger.source_failure_fact_id
        if failure_id in trigger_by_failure:
            raise TrainingSemanticResolutionError(
                "A Failure fact has more than one selected trigger annotation."
            )
        trigger_by_failure[failure_id] = trigger

    failed_attempt_by_fact = {
        item.failure_fact_id: item
        for item in timeline.attempts
        if item.outcome == "FAILURE" and item.failure_fact_id is not None
    }
    for failure_id, trigger in trigger_by_failure.items():
        attempt = failed_attempt_by_fact.get(failure_id)
        if attempt is None:
            raise TrainingSemanticResolutionError(
                "Failure trigger references an unknown failed Attempt."
            )
        projected_attempt = attempt_samples_by_id[attempt.attempt_id]
        if (
            projected_attempt.interval.empty
            or trigger.coarse_attempt_id != attempt.attempt_id
            or trigger.progress_step_id != attempt.progress_step_id
            or trigger.lower_bound_sample_index
            < projected_attempt.interval.sample_index_begin
            or trigger.upper_bound_sample_index
            != projected_attempt.interval.sample_index_end_exclusive - 1
            or trigger.trigger_sample_index
            < trigger.lower_bound_sample_index
            or trigger.trigger_sample_index > trigger.upper_bound_sample_index
        ):
            raise TrainingSemanticResolutionError(
                "Failure trigger does not bind its failed Attempt interval."
            )
        failed_stage = failed_stage_by_fact.get(failure_id)
        if (
            failed_stage is not None
            and trigger.stage_id != failed_stage.stage_id
        ):
            raise TrainingSemanticResolutionError(
                "Failure trigger and failed-stage identity differ."
            )

    failure_intervals: list[SampleIndexInterval] = []
    for attempt in timeline.attempts:
        if attempt.outcome != "FAILURE" or attempt.failure_fact_id is None:
            continue
        projected_attempt = attempt_samples_by_id[attempt.attempt_id]
        if projected_attempt.interval.empty:
            continue
        end_exclusive = projected_attempt.interval.sample_index_end_exclusive
        trigger = trigger_by_failure.get(attempt.failure_fact_id)
        failed_stage = failed_stage_by_fact.get(attempt.failure_fact_id)
        precise_trigger = bool(
            trigger is not None and trigger.source != "CAPTURE_DEFAULT"
        )
        if precise_trigger:
            assert trigger is not None
            begin = trigger.trigger_sample_index
            stage_id = trigger.stage_id
            source_id = trigger.annotation_id
            resolution = FAILURE_RESOLUTION_REFINED_TRIGGER
        elif failed_stage is not None:
            begin = failed_stage.sample_interval.sample_index_begin
            stage_id = failed_stage.stage_id
            source_id = failed_stage.source_id
            resolution = FAILURE_RESOLUTION_FAILED_STAGE
        else:
            begin = projected_attempt.interval.sample_index_begin
            stage_id = None
            source_id = attempt.failure_fact_id
            resolution = FAILURE_RESOLUTION_FAILED_ATTEMPT
        interval = SampleIndexInterval(begin, end_exclusive)
        failure_intervals.append(interval)
        spans.append(
            SemanticSpan(
                semantic_kind=SEMANTIC_SPAN_FAILURE,
                sample_interval=interval,
                progress_step_id=attempt.progress_step_id,
                attempt_id=attempt.attempt_id,
                stage_id=stage_id,
                outcome="FAILURE",
                source_id=source_id,
                source_failure_fact_id=attempt.failure_fact_id,
                resolution_kind=resolution,
            )
        )
        anchors.append(
            SemanticAnchor(
                semantic_kind=SEMANTIC_ANCHOR_FAILURE_REGION_BEGIN,
                sample_index=begin,
                progress_step_id=attempt.progress_step_id,
                attempt_id=attempt.attempt_id,
                stage_id=stage_id,
                source_id=source_id,
                source_failure_fact_id=attempt.failure_fact_id,
            )
        )
        if precise_trigger:
            assert trigger is not None
            anchors.append(
                SemanticAnchor(
                    semantic_kind=SEMANTIC_ANCHOR_FAILURE_TRIGGER,
                    sample_index=trigger.trigger_sample_index,
                    progress_step_id=attempt.progress_step_id,
                    attempt_id=attempt.attempt_id,
                    stage_id=trigger.stage_id,
                    source_id=trigger.annotation_id,
                    source_failure_fact_id=attempt.failure_fact_id,
                )
            )
        anchors.append(
            SemanticAnchor(
                semantic_kind=SEMANTIC_ANCHOR_FAILURE_END,
                sample_index=end_exclusive - 1,
                progress_step_id=attempt.progress_step_id,
                attempt_id=attempt.attempt_id,
                stage_id=stage_id,
                source_id=attempt.failure_fact_id,
                source_failure_fact_id=attempt.failure_fact_id,
            )
        )

    pause_by_failure = {
        item.source_failure_fact_id: item for item in timeline.capture_pause_intervals
    }
    for index, attempt in enumerate(timeline.attempts):
        source_failure_id: str | None = None
        recovery_condition: str | None = None
        if index == 0 and entry_condition == "RECOVERY_FROM_FAILURE":
            recovery_condition = entry_condition
        elif index > 0:
            previous = timeline.attempts[index - 1]
            pause = (
                None
                if previous.failure_fact_id is None
                else pause_by_failure.get(previous.failure_fact_id)
            )
            if (
                previous.outcome == "FAILURE"
                and previous.failure_fact_id is not None
                and previous.progress_step_id == attempt.progress_step_id
                and pause is not None
                and pause.end_at is not None
                and attempt.start_at >= pause.end_at
            ):
                source_failure_id = previous.failure_fact_id
                recovery_condition = "RECOVERY_FROM_FAILURE"
        if recovery_condition is None:
            continue
        projected_attempt = attempt_samples_by_id[attempt.attempt_id]
        if projected_attempt.interval.empty:
            continue
        spans.append(
            SemanticSpan(
                semantic_kind=SEMANTIC_SPAN_RECOVERY,
                sample_interval=projected_attempt.interval,
                progress_step_id=attempt.progress_step_id,
                attempt_id=attempt.attempt_id,
                outcome=attempt.outcome,
                entry_condition=recovery_condition,
                source_id=attempt.start_fact_id,
                source_failure_fact_id=source_failure_id,
            )
        )
        anchors.append(
            SemanticAnchor(
                semantic_kind=SEMANTIC_ANCHOR_RECOVERY_BEGIN,
                sample_index=projected_attempt.interval.sample_index_begin,
                progress_step_id=attempt.progress_step_id,
                attempt_id=attempt.attempt_id,
                source_id=attempt.start_fact_id,
                source_failure_fact_id=source_failure_id,
            )
        )

    attempt_intervals = tuple(item.interval for item in projected.attempts)
    pause_intervals = tuple(item.interval for item in projected.capture_pauses)
    nominal = _subtract_intervals(
        attempt_intervals,
        tuple(failure_intervals) + tuple(abnormal_intervals) + pause_intervals,
    )
    nominal = _split_intervals(
        nominal,
        continuity_break_indices,
        sample_count=sample_axis.sample_count,
    )
    failures = _subtract_intervals(failure_intervals, abnormal_intervals)
    failures = _split_intervals(
        failures,
        continuity_break_indices,
        sample_count=sample_axis.sample_count,
    )
    return EpisodeTrainingSemanticIndex(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        source_progress_manifest_hash=timeline.source_progress_manifest_hash,
        clock_domain_id=sample_axis.clock_domain_id,
        sample_count=sample_axis.sample_count,
        spans=tuple(
            sorted(
                spans,
                key=lambda item: (
                    item.sample_interval.sample_index_begin,
                    item.sample_interval.sample_index_end_exclusive,
                    item.semantic_kind,
                    item.source_id or "",
                ),
            )
        ),
        anchors=tuple(
            sorted(
                anchors,
                key=lambda item: (
                    item.sample_index,
                    item.semantic_kind,
                    item.source_id or "",
                ),
            )
        ),
        nominal_intervals=nominal,
        failure_intervals=failures,
    )


def _owner_attempt(
    *,
    index: EpisodeTrainingSemanticIndex,
    sample_index: int,
) -> SemanticSpan | None:
    matches = tuple(
        item
        for item in index.spans
        if item.semantic_kind == SEMANTIC_SPAN_ATTEMPT
        and item.sample_interval.sample_index_begin <= sample_index
        < item.sample_interval.sample_index_end_exclusive
    )
    if len(matches) > 1:
        raise TrainingSemanticResolutionError(
            "One sample belongs to more than one progress Attempt."
        )
    return None if not matches else matches[0]


def _containing_interval(
    intervals: Sequence[SampleIndexInterval],
    sample_index: int,
) -> SampleIndexInterval | None:
    return next(
        (
            interval
            for interval in intervals
            if interval.sample_index_begin <= sample_index
            < interval.sample_index_end_exclusive
        ),
        None,
    )


def plan_semantic_action_windows(
    *,
    index: EpisodeTrainingSemanticIndex,
    chunk_contract: PoseActionChunkContract,
    query: SemanticWindowQuery = SemanticWindowQuery(),
) -> tuple[SemanticActionWindowLocation, ...]:
    """Enumerate legal starts; chunk tail behavior remains contract-owned."""

    domain_intervals = (
        index.nominal_intervals
        if query.training_domain == TRAINING_DOMAIN_NOMINAL
        else index.failure_intervals
    )
    anchor_by_start: dict[int, SemanticAnchor] = {}
    if query.start_anchor_kinds:
        selected_kinds = frozenset(query.start_anchor_kinds)
        for anchor in index.anchors:
            if anchor.semantic_kind not in selected_kinds:
                continue
            if anchor.sample_index in anchor_by_start:
                raise TrainingSemanticResolutionError(
                    "Selected semantic anchors resolve to the same sample; "
                    "narrow the query by anchor kind."
                )
            anchor_by_start[anchor.sample_index] = anchor
        candidate_starts = tuple(sorted(anchor_by_start))
    else:
        candidate_starts = tuple(
            sample_index
            for interval in domain_intervals
            for sample_index in range(
                interval.sample_index_begin,
                interval.sample_index_end_exclusive,
            )
        )

    selected_steps = frozenset(query.progress_step_ids)
    selected_outcomes = frozenset(query.attempt_outcomes)
    result: list[SemanticActionWindowLocation] = []
    for start_index in candidate_starts:
        interval = _containing_interval(domain_intervals, start_index)
        if interval is None:
            continue
        owner = _owner_attempt(index=index, sample_index=start_index)
        if (
            owner is None
            or owner.progress_step_id is None
            or owner.attempt_id is None
            or owner.outcome is None
        ):
            continue
        if selected_steps and owner.progress_step_id not in selected_steps:
            continue
        if selected_outcomes and owner.outcome.upper() not in selected_outcomes:
            continue
        valid_count = min(
            chunk_contract.action_horizon,
            interval.sample_index_end_exclusive - start_index,
        )
        if (
            chunk_contract.tail_policy == POSE_CHUNK_TAIL_DROP
            and valid_count < chunk_contract.action_horizon
        ):
            continue
        anchor = anchor_by_start.get(start_index)
        result.append(
            SemanticActionWindowLocation(
                episode_capture_identity_hash=(
                    index.episode_capture_identity_hash
                ),
                start_index=start_index,
                valid_action_count=valid_count,
                owner_progress_step_id=owner.progress_step_id,
                owner_attempt_id=owner.attempt_id,
                owner_attempt_outcome=owner.outcome,
                training_domain=query.training_domain,
                containing_interval=interval,
                source_anchor_kind=(
                    None if anchor is None else anchor.semantic_kind
                ),
                source_anchor_id=(
                    None if anchor is None else anchor.source_id
                ),
            )
        )
    return tuple(result)


__all__ = [
    "EpisodeTrainingSemanticIndex",
    "FAILURE_RESOLUTION_FAILED_ATTEMPT",
    "FAILURE_RESOLUTION_FAILED_STAGE",
    "FAILURE_RESOLUTION_REFINED_TRIGGER",
    "SEMANTIC_ANCHOR_FAILURE_END",
    "SEMANTIC_ANCHOR_FAILURE_REGION_BEGIN",
    "SEMANTIC_ANCHOR_FAILURE_TRIGGER",
    "SEMANTIC_ANCHOR_PROGRESS_STEP_BEGIN",
    "SEMANTIC_ANCHOR_RECOVERY_BEGIN",
    "SEMANTIC_ANCHOR_STAGE_BEGIN",
    "SEMANTIC_SPAN_ABNORMAL",
    "SEMANTIC_SPAN_ATTEMPT",
    "SEMANTIC_SPAN_FAILURE",
    "SEMANTIC_SPAN_RECOVERY",
    "SEMANTIC_SPAN_STAGE",
    "SemanticActionWindowLocation",
    "SemanticAnchor",
    "SemanticSpan",
    "SemanticWindowQuery",
    "TRAINING_DOMAIN_FAILURE",
    "TRAINING_DOMAIN_NOMINAL",
    "TrainingSemanticResolutionError",
    "build_episode_training_semantic_index",
    "plan_semantic_action_windows",
]
