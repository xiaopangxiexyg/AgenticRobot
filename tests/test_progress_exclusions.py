from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_interface.progress_exclusions import (
    ProgressExclusionResolutionError,
    derive_coarse_failure_training_view,
)
from dataset_interface.progress_timeline import (
    ResolvedCapturePauseInterval,
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
    ResolvedProgressTransition,
)
from dataset_interface.sample_axis import (
    EpisodeSampleAxis,
    SampleAxisError,
    SampleIndexInterval,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _timeline() -> ResolvedProgressTimeline:
    attempts = (
        ResolvedProgressAttempt(
            attempt_id="attempt:a:failed",
            progress_step_id="a",
            attempt_ordinal_in_capture=1,
            start_fact_id="start:a",
            start_at=0.0,
            clock_domain_id="episode_clock",
            end_fact_id="failure:a",
            end_at=1.0,
            outcome="FAILURE",
            failure_fact_id="failure:a",
            failure_reported_at=1.0,
            failure_timing_semantics="OPERATOR_REPORT_TIME",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt:a:retry",
            progress_step_id="a",
            attempt_ordinal_in_capture=2,
            start_fact_id="retry:a",
            start_at=2.0,
            clock_domain_id="episode_clock",
            end_fact_id="advance:a:b",
            end_at=3.0,
            outcome="SUCCESS",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt:b:failed",
            progress_step_id="b",
            attempt_ordinal_in_capture=3,
            start_fact_id="advance:a:b",
            start_at=3.0,
            clock_domain_id="episode_clock",
            end_fact_id="failure:b",
            end_at=4.5,
            outcome="FAILURE",
            failure_fact_id="failure:b",
            failure_reported_at=4.5,
            failure_timing_semantics="OPERATOR_REPORT_TIME",
        ),
    )
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash="2" * 64,
        progress_episode_binding_hash="3" * 64,
        clock_domain_id="episode_clock",
        fact_bundle_hash="4" * 64,
        fact_count=10,
        sealed=True,
        phase="TERMINATED",
        attempts=attempts,
        transitions=(
            ResolvedProgressTransition(
                advance_fact_id="advance:a:b",
                from_progress_step_id="a",
                from_attempt_id="attempt:a:retry",
                to_progress_step_id="b",
                to_attempt_id="attempt:b:failed",
                effective_at=3.0,
                clock_domain_id="episode_clock",
            ),
        ),
        capture_pause_intervals=(
            ResolvedCapturePauseInterval(
                source_failure_fact_id="failure:a",
                pause_receipt_fact_id="pause:a",
                start_at=1.1,
                clock_domain_id="episode_clock",
                resume_receipt_fact_id="resume:a",
                end_at=2.0,
            ),
        ),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id="b",
        active_attempt_id=None,
        terminal_fact_id="termination",
        termination_cause="SAFETY_STOP",
        full_task_outcome="FAILURE",
        source_progress_manifest_hash="5" * 64,
    )


def _axis() -> EpisodeSampleAxis:
    return EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(
            0.0,
            0.5,
            1.0,
            1.0,
            1.1,
            2.0,
            3.0,
            4.0,
            4.5,
            4.5,
            5.0,
        ),
    )


def test_coarse_failure_view_excludes_only_failed_attempts_through_cut() -> None:
    view = derive_coarse_failure_training_view(
        timeline=_timeline(),
        sample_axis=_axis(),
    )

    assert view.sample_count == 11
    assert [
        (
            item.source_failure_fact_id,
            item.attempt_id,
            item.sample_interval,
        )
        for item in view.exclusions
    ] == [
        ("failure:a", "attempt:a:failed", SampleIndexInterval(0, 4)),
        ("failure:b", "attempt:b:failed", SampleIndexInterval(6, 10)),
    ]
    assert all(
        item.attempt_id != "attempt:a:retry" for item in view.exclusions
    )
    assert all(
        not (
            item.sample_interval.sample_index_begin
            <= 4
            < item.sample_interval.sample_index_end_exclusive
        )
        for item in view.exclusions
    )


def test_coarse_failure_view_accepts_verified_promoted_hold_cut() -> None:
    promoted = replace(
        _timeline().attempts[0],
        failure_timing_semantics="PROMOTED_HOLD_TIME",
    )
    timeline = replace(
        _timeline(),
        attempts=(promoted,) + _timeline().attempts[1:],
    )

    view = derive_coarse_failure_training_view(
        timeline=timeline,
        sample_axis=_axis(),
    )

    assert view.exclusions[0].source_failure_fact_id == "failure:a"
    assert view.exclusions[0].failure_cut_at == 1.0
    assert view.exclusions[0].sample_interval == SampleIndexInterval(0, 4)


def test_coarse_failure_view_preserves_empty_exclusion_for_auditing() -> None:
    timeline = replace(
        _timeline(),
        attempts=(
            replace(
                _timeline().attempts[0],
                start_at=6.0,
                end_at=7.0,
                failure_reported_at=7.0,
            ),
        ),
        transitions=(),
        capture_pause_intervals=(),
    )
    view = derive_coarse_failure_training_view(
        timeline=timeline,
        sample_axis=_axis(),
    )

    assert len(view.exclusions) == 1
    assert view.exclusions[0].sample_interval == SampleIndexInterval(11, 11)


def test_coarse_failure_view_accepts_sealed_success_without_exclusions() -> None:
    successful = replace(
        _timeline().attempts[1],
        end_fact_id="termination",
        end_at=4.0,
        outcome="SUCCESS",
    )
    timeline = replace(
        _timeline(),
        attempts=(successful,),
        transitions=(),
        capture_pause_intervals=(),
    )
    view = derive_coarse_failure_training_view(
        timeline=timeline,
        sample_axis=_axis(),
    )

    assert view.exclusions == ()


def test_coarse_failure_view_rejects_open_or_malformed_timeline() -> None:
    with pytest.raises(
        ProgressExclusionResolutionError,
        match="sealed terminated",
    ):
        derive_coarse_failure_training_view(
            timeline=replace(
                _timeline(),
                sealed=False,
                phase="ACTIVE_ATTEMPT",
            ),
            sample_axis=_axis(),
        )

    with pytest.raises(
        ProgressExclusionResolutionError,
        match="capture-time causal",
    ):
        derive_coarse_failure_training_view(
            timeline=replace(
                _timeline(),
                resolution_kind="RETROSPECTIVE_CORRECTION",
            ),
            sample_axis=_axis(),
        )

    malformed = replace(
        _timeline().attempts[0],
        failure_reported_at=None,
    )
    with pytest.raises(
        ProgressExclusionResolutionError,
        match="missing its online Failure cut",
    ):
        derive_coarse_failure_training_view(
            timeline=replace(
                _timeline(),
                attempts=(malformed,) + _timeline().attempts[1:],
            ),
            sample_axis=_axis(),
        )

    non_failure_with_metadata = replace(
        _timeline().attempts[0],
        outcome="UNKNOWN",
    )
    with pytest.raises(
        ProgressExclusionResolutionError,
        match="carries Failure metadata",
    ):
        derive_coarse_failure_training_view(
            timeline=replace(
                _timeline(),
                attempts=(
                    non_failure_with_metadata,
                )
                + _timeline().attempts[1:],
            ),
            sample_axis=_axis(),
        )


def test_coarse_failure_view_rejects_foreign_sample_clock() -> None:
    with pytest.raises(SampleAxisError, match="different clocks"):
        derive_coarse_failure_training_view(
            timeline=_timeline(),
            sample_axis=EpisodeSampleAxis(
                clock_domain_id="foreign_clock",
                cut_times=(0.0,),
            ),
        )
