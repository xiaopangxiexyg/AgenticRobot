from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from dataset_contracts.v25p_native import V25pCausalSample
from dataset_interface.progress_samples import (
    project_progress_timeline_to_samples,
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


@dataclass(frozen=True)
class _Point:
    sample_index: int
    cut_at: float
    cut_clock_domain_id: str = "episode_clock"


def test_sample_axis_has_explicit_equal_timestamp_boundaries() -> None:
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(1.0, 2.0, 2.0, 3.0),
    )

    assert axis.first_at_or_after(2.0) == 1
    assert axis.first_strictly_after(2.0) == 3
    assert axis.half_open_time_interval(
        start_at=2.0,
        end_at=3.0,
    ) == SampleIndexInterval(1, 3)
    assert axis.closed_time_interval(
        start_at=2.0,
        end_at=3.0,
    ) == SampleIndexInterval(1, 4)


def test_sample_axis_accepts_empty_episode_and_rejects_ambiguous_axes() -> None:
    empty = EpisodeSampleAxis.from_causal_samples(
        (),
        clock_domain_id="episode_clock",
    )
    assert empty.sample_count == 0
    assert empty.from_time_to_end(start_at=1.0).empty

    with pytest.raises(SampleAxisError, match="contiguous"):
        EpisodeSampleAxis.from_causal_samples(
            (_Point(0, 1.0), _Point(2, 2.0)),
            clock_domain_id="episode_clock",
        )
    with pytest.raises(SampleAxisError, match="bound Episode clock"):
        EpisodeSampleAxis.from_causal_samples(
            (_Point(0, 1.0, "foreign_clock"),),
            clock_domain_id="episode_clock",
        )
    with pytest.raises(SampleAxisError, match="non-decreasing"):
        EpisodeSampleAxis(
            clock_domain_id="episode_clock",
            cut_times=(2.0, 1.0),
        )
    with pytest.raises(SampleAxisError, match="contiguous"):
        EpisodeSampleAxis.from_causal_samples(
            (_Point(True, 1.0),),
            clock_domain_id="episode_clock",
        )
    with pytest.raises(SampleAxisError, match="finite number"):
        EpisodeSampleAxis(
            clock_domain_id="episode_clock",
            cut_times=(object(),),
        )


def test_sample_axis_consumes_native_v25p_causal_samples_without_adapter() -> None:
    sample = V25pCausalSample(
        episode_uuid="episode",
        legacy_episode_index=0,
        component_stream_profile_hash="1" * 64,
        sample_index=0,
        cut_at=1.25,
        cut_clock_domain_id="episode_clock",
        record_hash_by_stream={"action.arm": "2" * 64},
    )
    axis = EpisodeSampleAxis.from_causal_samples(
        (sample,),
        clock_domain_id="episode_clock",
    )

    assert axis.cut_times == (1.25,)
    assert axis.closed_time_interval(
        start_at=1.25,
        end_at=1.25,
    ) == SampleIndexInterval(0, 1)


def _timeline() -> ResolvedProgressTimeline:
    attempts = (
        ResolvedProgressAttempt(
            attempt_id="attempt:a",
            progress_step_id="a",
            attempt_ordinal_in_capture=1,
            start_fact_id="start:a",
            start_at=0.0,
            clock_domain_id="episode_clock",
            end_fact_id="advance:a:b",
            end_at=1.0,
            outcome="SUCCESS",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt:b:1",
            progress_step_id="b",
            attempt_ordinal_in_capture=2,
            start_fact_id="advance:a:b",
            start_at=1.0,
            clock_domain_id="episode_clock",
            end_fact_id="failure:b",
            end_at=2.0,
            outcome="FAILURE",
            failure_fact_id="failure:b",
            failure_reported_at=2.0,
            failure_timing_semantics="OPERATOR_REPORT_TIME",
        ),
        ResolvedProgressAttempt(
            attempt_id="attempt:b:2",
            progress_step_id="b",
            attempt_ordinal_in_capture=3,
            start_fact_id="retry:b",
            start_at=3.0,
            clock_domain_id="episode_clock",
            end_fact_id="termination",
            end_at=4.0,
            outcome="SUCCESS",
        ),
    )
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash="2" * 64,
        progress_episode_binding_hash="3" * 64,
        clock_domain_id="episode_clock",
        fact_bundle_hash="4" * 64,
        fact_count=8,
        sealed=True,
        phase="TERMINATED",
        attempts=attempts,
        transitions=(
            ResolvedProgressTransition(
                advance_fact_id="advance:a:b",
                from_progress_step_id="a",
                from_attempt_id="attempt:a",
                to_progress_step_id="b",
                to_attempt_id="attempt:b:1",
                effective_at=1.0,
                clock_domain_id="episode_clock",
            ),
        ),
        capture_pause_intervals=(
            ResolvedCapturePauseInterval(
                source_failure_fact_id="failure:b",
                pause_receipt_fact_id="pause:b",
                start_at=2.1,
                clock_domain_id="episode_clock",
                resume_receipt_fact_id="resume:b",
                end_at=3.0,
            ),
        ),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id="b",
        active_attempt_id=None,
        terminal_fact_id="termination",
        termination_cause="FULL_TASK_COMPLETED",
        full_task_outcome="SUCCESS",
        source_progress_manifest_hash="5" * 64,
    )


def test_progress_projection_preserves_transition_and_failure_boundaries() -> None:
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(0.0, 0.5, 1.0, 1.5, 2.0, 2.1, 2.9, 3.0, 3.5, 4.0),
    )
    projected = project_progress_timeline_to_samples(
        timeline=_timeline(),
        sample_axis=axis,
    )

    assert projected.sample_count == 10
    assert projected.attempts[0].interval == SampleIndexInterval(0, 2)
    assert projected.attempts[0].end_boundary_inclusive is False
    assert projected.attempts[1].interval == SampleIndexInterval(2, 5)
    assert projected.attempts[1].end_boundary_inclusive is True
    assert projected.capture_pauses[0].interval == SampleIndexInterval(5, 7)
    assert projected.attempts[2].interval == SampleIndexInterval(7, 10)


def test_progress_projection_rejects_clock_mismatch() -> None:
    with pytest.raises(SampleAxisError, match="different clocks"):
        project_progress_timeline_to_samples(
            timeline=_timeline(),
            sample_axis=EpisodeSampleAxis(
                clock_domain_id="foreign_clock",
                cut_times=(0.0,),
            ),
        )

    mismatched_attempt = replace(
        _timeline().attempts[0],
        clock_domain_id="foreign_clock",
    )
    timeline = replace(
        _timeline(),
        attempts=(mismatched_attempt,) + _timeline().attempts[1:],
    )
    with pytest.raises(SampleAxisError, match="sample-axis clock"):
        project_progress_timeline_to_samples(
            timeline=timeline,
            sample_axis=EpisodeSampleAxis(
                clock_domain_id="episode_clock",
                cut_times=(0.0,),
            ),
        )
