from __future__ import annotations

from dataclasses import replace

import pytest

from collection_runtime.progress.failure_trigger_store import (
    commit_failure_trigger_annotation_set,
    load_failure_trigger_annotation_set,
)
from collection_runtime.progress.failure_trigger_capture_store import (
    load_failure_trigger_capture_records,
    record_failure_trigger_capture,
)
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationSetV1,
    FailureTriggerCaptureRecordV1,
)
from dataset_contracts.progress_stage_annotations import (
    StageOccurrenceAnnotationV1,
)
from dataset_interface.progress_failure_triggers import (
    FailureTriggerResolutionError,
    build_failure_trigger_annotations,
    derive_failure_trigger_analysis_view,
    derive_failure_trigger_windows,
    failure_trigger_sample_axis_hash,
    normalize_failure_trigger_drafts,
)
from dataset_interface.progress_failure_trigger_artifacts import (
    load_failure_trigger_annotation_set as load_readonly_failure_trigger_set,
    load_optional_failure_trigger_annotation_set,
)
from dataset_interface.progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from dataset_interface.sample_axis import EpisodeSampleAxis


pytestmark = [pytest.mark.contract]


def _timeline() -> ResolvedProgressTimeline:
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash="2" * 64,
        progress_episode_binding_hash="3" * 64,
        clock_domain_id="episode_clock",
        fact_bundle_hash="4" * 64,
        fact_count=4,
        sealed=True,
        phase="TERMINATED",
        attempts=(
            ResolvedProgressAttempt(
                attempt_id="attempt:1",
                progress_step_id="step:a",
                attempt_ordinal_in_capture=1,
                start_fact_id="fact:start",
                start_at=0.0,
                clock_domain_id="episode_clock",
                end_fact_id="fact:failure",
                end_at=5.0,
                outcome="FAILURE",
                failure_fact_id="fact:failure",
                failure_reported_at=5.0,
                failure_timing_semantics="OPERATOR_REPORT_TIME",
            ),
        ),
        transitions=(),
        capture_pause_intervals=(),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id=None,
        active_attempt_id=None,
        terminal_fact_id="fact:terminal",
        termination_cause="MANUAL_STOP",
        full_task_outcome="FAILURE",
        source_progress_manifest_hash="5" * 64,
    )


def _occurrence(start_at: float = 2.0) -> StageOccurrenceAnnotationV1:
    return StageOccurrenceAnnotationV1(
        annotation_id="occurrence:failure",
        coarse_attempt_id="attempt:1",
        progress_step_id="step:a",
        stage_id="stage:b2",
        occurrence_index_in_attempt=1,
        start_at=start_at,
        end_at=5.0,
        clock_domain_id="episode_clock",
        outcome="FAILURE",
        source_failure_fact_id="fact:failure",
    )


def _axis() -> EpisodeSampleAxis:
    return EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=tuple(float(value) for value in range(6)),
    )


def test_failure_trigger_window_is_current_stage_start_to_recorded_failure() -> None:
    window = derive_failure_trigger_windows(
        timeline=_timeline(),
        occurrences=(_occurrence(),),
        sample_axis=_axis(),
    )[0]

    assert window.lower_bound_sample_index == 2
    assert window.upper_bound_sample_index == 5
    assert window.lower_bound_at == 2.0
    assert window.upper_bound_at == 5.0


def test_non_sample_failure_time_uses_discrete_upper_projection() -> None:
    timeline = _timeline()
    attempt = replace(
        timeline.attempts[0],
        end_at=5.4,
        failure_reported_at=5.4,
    )
    occurrence = replace(_occurrence(), end_at=5.4)
    window = derive_failure_trigger_windows(
        timeline=replace(timeline, attempts=(attempt,)),
        occurrences=(occurrence,),
        sample_axis=_axis(),
    )[0]

    assert window.upper_bound_sample_index == 5
    assert window.upper_bound_at == 5.0


def test_default_and_refined_trigger_are_discrete_and_roundtrip() -> None:
    axis = _axis()
    timeline = _timeline()
    occurrence = _occurrence()
    default = normalize_failure_trigger_drafts(
        timeline=timeline,
        occurrences=(occurrence,),
        sample_axis=axis,
    )
    assert default[0]["trigger_sample_index"] == 5
    assert default[0]["trigger_at"] == 5.0
    assert default[0]["source"] == "CAPTURE_DEFAULT"

    refined = replace(
        occurrence,
        annotation_id="occurrence:failure-refined-fixture",
    )
    draft = [
        {
            **default[0],
            "trigger_sample_index": 3,
            "trigger_at": 3.0,
            "source": "POST_CAPTURE_REFINED",
        }
    ]
    annotations = build_failure_trigger_annotations(
        timeline=timeline,
        occurrences=(refined,),
        sample_axis=axis,
        drafts=draft,
    )
    assert annotations[0].trigger_sample_index == 3
    assert annotations[0].trigger_at == 3.0
    analysis = derive_failure_trigger_analysis_view(annotations)[0]
    assert analysis.normal_prefix_end_at == 3.0
    assert analysis.failure_suffix_start_at == 3.0


def test_trigger_outside_current_failed_stage_is_rejected() -> None:
    axis = _axis()
    default = normalize_failure_trigger_drafts(
        timeline=_timeline(),
        occurrences=(_occurrence(),),
        sample_axis=axis,
    )[0]
    with pytest.raises(
        FailureTriggerResolutionError,
        match="outside the failed-stage window",
    ):
        normalize_failure_trigger_drafts(
            timeline=_timeline(),
            occurrences=(_occurrence(),),
            sample_axis=axis,
            existing=[
                {
                    **default,
                    "trigger_sample_index": 1,
                    "trigger_at": 1.0,
                }
            ],
        )


@pytest.mark.integration
def test_trigger_sidecar_is_additive_and_immutable(tmp_path) -> None:
    axis = _axis()
    annotation = build_failure_trigger_annotations(
        timeline=_timeline(),
        occurrences=(_occurrence(),),
        sample_axis=axis,
        drafts=normalize_failure_trigger_drafts(
            timeline=_timeline(),
            occurrences=(_occurrence(),),
            sample_axis=axis,
        ),
    )[0]
    value = FailureTriggerAnnotationSetV1(
        annotation_set_id="stage-set:1",
        stage_annotation_set_id="stage-set:1",
        episode_capture_identity_hash="1" * 64,
        source_progress_manifest_hash="5" * 64,
        task_stage_profile_hash="6" * 64,
        stage_annotation_bundle_hash="7" * 64,
        sample_axis_hash=failure_trigger_sample_axis_hash(axis),
        bound_clock_domain_id="episode_clock",
        annotator_id="fixture",
        created_at=10.0,
        created_clock_domain_id="wall_clock",
        annotations=(annotation,),
    )
    report = commit_failure_trigger_annotation_set(
        annotation_dir=tmp_path / "stage-set:1",
        annotation_set=value,
    )
    assert not report.idempotent
    loaded = load_failure_trigger_annotation_set(tmp_path / "stage-set:1")
    assert loaded == value
    assert load_readonly_failure_trigger_set(
        tmp_path / "stage-set:1",
        expected_stage_annotation_set_id="stage-set:1",
        expected_clock_domain_id="episode_clock",
    ) == value
    assert commit_failure_trigger_annotation_set(
        annotation_dir=tmp_path / "stage-set:1",
        annotation_set=value,
    ).idempotent
    assert load_optional_failure_trigger_annotation_set(tmp_path / "legacy") is None


@pytest.mark.integration
def test_capture_receipt_is_optional_immutable_and_source_keyed(tmp_path) -> None:
    record = FailureTriggerCaptureRecordV1(
        source_failure_fact_id="fact:failure",
        coarse_attempt_id="attempt:1",
        progress_step_id="step:a",
        recorded_at=5.0,
        clock_domain_id="episode_clock",
    )
    report = record_failure_trigger_capture(
        episode_dir=tmp_path / "progress",
        record=record,
    )
    assert not report.idempotent
    assert report.path.is_file()
    assert load_failure_trigger_capture_records(tmp_path / "progress") == (
        record,
    )
    assert record_failure_trigger_capture(
        episode_dir=tmp_path / "progress",
        record=record,
    ).idempotent
    assert load_failure_trigger_capture_records(tmp_path / "empty") == ()
