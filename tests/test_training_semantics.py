from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationV1,
)
from dataset_interface.progress_stages import (
    ResolvedStageOccurrence,
    StageTrainingExclusion,
    StageTrainingIndex,
)
from dataset_interface.progress_timeline import (
    ResolvedCapturePauseInterval,
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
    ResolvedProgressTransition,
)
from dataset_interface.sample_axis import EpisodeSampleAxis, SampleIndexInterval
from dataset_interface.training_semantics import (
    FAILURE_RESOLUTION_FAILED_ATTEMPT,
    FAILURE_RESOLUTION_FAILED_STAGE,
    FAILURE_RESOLUTION_REFINED_TRIGGER,
    SEMANTIC_ANCHOR_FAILURE_TRIGGER,
    SEMANTIC_ANCHOR_RECOVERY_BEGIN,
    SEMANTIC_SPAN_FAILURE,
    SemanticWindowQuery,
    TRAINING_DOMAIN_FAILURE,
    TrainingSemanticResolutionError,
    build_episode_training_semantic_index,
    plan_semantic_action_windows,
)


def _timeline() -> ResolvedProgressTimeline:
    attempts = (
        ResolvedProgressAttempt(
            attempt_id="attempt-a",
            progress_step_id="step-a",
            attempt_ordinal_in_capture=1,
            start_fact_id="start-a",
            start_at=0.0,
            clock_domain_id="episode-clock",
            end_fact_id="advance-a-b",
            end_at=6.0,
            outcome="SUCCESS",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt-b-failed",
            progress_step_id="step-b",
            attempt_ordinal_in_capture=2,
            start_fact_id="advance-a-b",
            start_at=6.0,
            clock_domain_id="episode-clock",
            end_fact_id="failure-b",
            end_at=12.0,
            outcome="FAILURE",
            failure_fact_id="failure-b",
            failure_reported_at=12.0,
            failure_timing_semantics="OPERATOR_REPORT_TIME",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt-b-retry",
            progress_step_id="step-b",
            attempt_ordinal_in_capture=3,
            start_fact_id="retry-b",
            start_at=14.0,
            clock_domain_id="episode-clock",
            end_fact_id="advance-b-c",
            end_at=17.0,
            outcome="SUCCESS",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt-c",
            progress_step_id="step-c",
            attempt_ordinal_in_capture=4,
            start_fact_id="advance-b-c",
            start_at=17.0,
            clock_domain_id="episode-clock",
            end_fact_id="complete",
            end_at=19.0,
            outcome="SUCCESS",
        ),
    )
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="episode-identity",
        task_progress_profile_hash="progress-profile",
        progress_episode_binding_hash="progress-binding",
        clock_domain_id="episode-clock",
        fact_bundle_hash="fact-bundle",
        fact_count=12,
        sealed=True,
        phase="TERMINATED",
        attempts=attempts,
        transitions=(
            ResolvedProgressTransition(
                advance_fact_id="advance-a-b",
                from_progress_step_id="step-a",
                from_attempt_id="attempt-a",
                to_progress_step_id="step-b",
                to_attempt_id="attempt-b-failed",
                effective_at=6.0,
                clock_domain_id="episode-clock",
            ),
            ResolvedProgressTransition(
                advance_fact_id="advance-b-c",
                from_progress_step_id="step-b",
                from_attempt_id="attempt-b-retry",
                to_progress_step_id="step-c",
                to_attempt_id="attempt-c",
                effective_at=17.0,
                clock_domain_id="episode-clock",
            ),
        ),
        capture_pause_intervals=(
            ResolvedCapturePauseInterval(
                source_failure_fact_id="failure-b",
                pause_receipt_fact_id="pause-b",
                start_at=12.0,
                clock_domain_id="episode-clock",
                resume_receipt_fact_id="resume-b",
                end_at=14.0,
            ),
        ),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id=None,
        active_attempt_id=None,
        terminal_fact_id="complete",
        termination_cause="FULL_TASK_COMPLETED",
        full_task_outcome="SUCCESS",
        source_progress_manifest_hash="progress-manifest",
    )


def _axis() -> EpisodeSampleAxis:
    return EpisodeSampleAxis(
        clock_domain_id="episode-clock",
        cut_times=tuple(float(value) for value in range(20)),
    )


def _stage_index(*, abnormal: bool = False) -> StageTrainingIndex:
    occurrences = (
        ResolvedStageOccurrence(
            annotation_id="stage-a",
            coarse_attempt_id="attempt-a",
            progress_step_id="step-a",
            stage_id="a1",
            occurrence_index_in_attempt=1,
            start_at=0.0,
            end_at=6.0,
            outcome="SUCCESS",
            source_failure_fact_id=None,
            sample_interval=SampleIndexInterval(0, 6),
        ),
        ResolvedStageOccurrence(
            annotation_id="stage-b-failure",
            coarse_attempt_id="attempt-b-failed",
            progress_step_id="step-b",
            stage_id="b2",
            occurrence_index_in_attempt=1,
            start_at=8.0,
            end_at=12.0,
            outcome="FAILURE",
            source_failure_fact_id="failure-b",
            sample_interval=SampleIndexInterval(8, 13),
        ),
        ResolvedStageOccurrence(
            annotation_id="stage-b-retry",
            coarse_attempt_id="attempt-b-retry",
            progress_step_id="step-b",
            stage_id="b2",
            occurrence_index_in_attempt=1,
            start_at=14.0,
            end_at=17.0,
            outcome="SUCCESS",
            source_failure_fact_id=None,
            sample_interval=SampleIndexInterval(14, 17),
        ),
        ResolvedStageOccurrence(
            annotation_id="stage-c",
            coarse_attempt_id="attempt-c",
            progress_step_id="step-c",
            stage_id="c1",
            occurrence_index_in_attempt=1,
            start_at=17.0,
            end_at=19.0,
            outcome="SUCCESS",
            source_failure_fact_id=None,
            sample_interval=SampleIndexInterval(17, 20),
        ),
    )
    exclusions = [
        StageTrainingExclusion(
            exclusion_kind="FAILED_STAGE_OCCURRENCE",
            sample_interval=SampleIndexInterval(8, 13),
            source_annotation_id="stage-b-failure",
            source_failure_fact_id="failure-b",
            progress_step_id="step-b",
            stage_id="b2",
            coarse_attempt_id="attempt-b-failed",
        )
    ]
    if abnormal:
        exclusions.append(
            StageTrainingExclusion(
                exclusion_kind="ABNORMAL_INTERVAL",
                sample_interval=SampleIndexInterval(18, 20),
                source_annotation_id="abnormal-c-tail",
            )
        )
    return StageTrainingIndex(
        episode_capture_identity_hash="episode-identity",
        source_progress_manifest_hash="progress-manifest",
        task_stage_profile_hash="stage-profile",
        annotation_bundle_hash="annotation-bundle",
        clock_domain_id="episode-clock",
        sample_count=20,
        coverage_status="COMPLETE",
        annotated_sample_count=19,
        unannotated_sample_count=1,
        occurrences=occurrences,
        exclusions=tuple(exclusions),
    )


def _trigger(source: str = "POST_CAPTURE_REFINED") -> FailureTriggerAnnotationV1:
    trigger_index = 10 if source != "CAPTURE_DEFAULT" else 12
    return FailureTriggerAnnotationV1(
        annotation_id=f"trigger-{source.lower()}",
        source_failure_fact_id="failure-b",
        coarse_attempt_id="attempt-b-failed",
        progress_step_id="step-b",
        stage_id="b2",
        lower_bound_sample_index=8,
        upper_bound_sample_index=12,
        trigger_sample_index=trigger_index,
        lower_bound_at=8.0,
        upper_bound_at=12.0,
        trigger_at=float(trigger_index),
        clock_domain_id="episode-clock",
        confidence=1.0,
        source=source,
    )


def test_refined_failure_region_and_cross_progress_windows() -> None:
    index = build_episode_training_semantic_index(
        timeline=_timeline(),
        sample_axis=_axis(),
        stage_index=_stage_index(),
        failure_triggers=(_trigger(),),
    )
    failure = index.spans_of_kind(SEMANTIC_SPAN_FAILURE)
    assert len(failure) == 1
    assert failure[0].sample_interval == SampleIndexInterval(10, 13)
    assert failure[0].resolution_kind == FAILURE_RESOLUTION_REFINED_TRIGGER
    assert index.nominal_intervals == (
        SampleIndexInterval(0, 10),
        SampleIndexInterval(14, 20),
    )

    windows = plan_semantic_action_windows(
        index=index,
        chunk_contract=PoseActionChunkContract(
            action_horizon=4,
            tail_policy="pad_last",
        ),
        query=SemanticWindowQuery(progress_step_ids=("step-a",)),
    )
    crossing = next(item for item in windows if item.start_index == 5)
    assert crossing.owner_progress_step_id == "step-a"
    assert crossing.valid_action_count == 4
    assert crossing.containing_interval == SampleIndexInterval(0, 10)


def test_default_trigger_is_unannotated_and_falls_back_to_failed_stage() -> None:
    index = build_episode_training_semantic_index(
        timeline=_timeline(),
        sample_axis=_axis(),
        stage_index=_stage_index(),
        failure_triggers=(_trigger("CAPTURE_DEFAULT"),),
    )
    failure = index.spans_of_kind(SEMANTIC_SPAN_FAILURE)[0]
    assert failure.sample_interval == SampleIndexInterval(8, 13)
    assert failure.resolution_kind == FAILURE_RESOLUTION_FAILED_STAGE
    assert index.anchors_of_kind(SEMANTIC_ANCHOR_FAILURE_TRIGGER) == ()


def test_failure_without_stage_falls_back_to_failed_attempt() -> None:
    index = build_episode_training_semantic_index(
        timeline=_timeline(),
        sample_axis=_axis(),
    )
    failure = index.spans_of_kind(SEMANTIC_SPAN_FAILURE)[0]
    assert failure.sample_interval == SampleIndexInterval(6, 13)
    assert failure.resolution_kind == FAILURE_RESOLUTION_FAILED_ATTEMPT


def test_foreign_failure_trigger_binding_fails_closed() -> None:
    trigger = replace(_trigger(), coarse_attempt_id="attempt-a")
    with pytest.raises(
        TrainingSemanticResolutionError,
        match="does not bind its failed Attempt interval",
    ):
        build_episode_training_semantic_index(
            timeline=_timeline(),
            sample_axis=_axis(),
            stage_index=_stage_index(),
            failure_triggers=(trigger,),
        )


def test_failure_and_recovery_anchor_queries_use_their_own_domains() -> None:
    index = build_episode_training_semantic_index(
        timeline=_timeline(),
        sample_axis=_axis(),
        stage_index=_stage_index(),
        failure_triggers=(_trigger(),),
    )
    padded = PoseActionChunkContract(action_horizon=4, tail_policy="pad_last")
    failure_windows = plan_semantic_action_windows(
        index=index,
        chunk_contract=padded,
        query=SemanticWindowQuery(
            progress_step_ids=("step-b",),
            start_anchor_kinds=(SEMANTIC_ANCHOR_FAILURE_TRIGGER,),
            training_domain=TRAINING_DOMAIN_FAILURE,
        ),
    )
    assert [
        (item.start_index, item.valid_action_count) for item in failure_windows
    ] == [(10, 3)]

    recovery_windows = plan_semantic_action_windows(
        index=index,
        chunk_contract=padded,
        query=SemanticWindowQuery(
            progress_step_ids=("step-b",),
            start_anchor_kinds=(SEMANTIC_ANCHOR_RECOVERY_BEGIN,),
        ),
    )
    assert [
        (item.start_index, item.valid_action_count) for item in recovery_windows
    ] == [(14, 4)]


def test_abnormal_and_continuity_boundaries_truncate_for_padding_or_drop() -> None:
    index = build_episode_training_semantic_index(
        timeline=_timeline(),
        sample_axis=_axis(),
        stage_index=_stage_index(abnormal=True),
        failure_triggers=(_trigger(),),
        continuity_break_indices=(16,),
    )
    assert index.nominal_intervals == (
        SampleIndexInterval(0, 10),
        SampleIndexInterval(14, 16),
        SampleIndexInterval(16, 18),
    )
    assert index.continuity_break_indices == (16,)
    padded = plan_semantic_action_windows(
        index=index,
        chunk_contract=PoseActionChunkContract(
            action_horizon=4,
            tail_policy="pad_last",
        ),
        query=SemanticWindowQuery(
            progress_step_ids=("step-b",),
            start_anchor_kinds=(SEMANTIC_ANCHOR_RECOVERY_BEGIN,),
        ),
    )
    assert [(item.start_index, item.valid_action_count) for item in padded] == [(14, 2)]

    dropped = plan_semantic_action_windows(
        index=index,
        chunk_contract=PoseActionChunkContract(
            action_horizon=4,
            tail_policy="drop",
        ),
        query=SemanticWindowQuery(
            progress_step_ids=("step-b",),
            start_anchor_kinds=(SEMANTIC_ANCHOR_RECOVERY_BEGIN,),
        ),
    )
    assert dropped == ()
