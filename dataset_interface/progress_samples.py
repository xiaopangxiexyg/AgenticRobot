"""Project resolved progress time intervals onto one causal sample axis."""

from __future__ import annotations

from dataclasses import dataclass

from .progress_timeline import (
    ResolvedCapturePauseInterval,
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from .sample_axis import EpisodeSampleAxis, SampleAxisError, SampleIndexInterval


@dataclass(frozen=True)
class ResolvedProgressAttemptSamples:
    attempt_id: str
    progress_step_id: str
    outcome: str
    interval: SampleIndexInterval
    end_boundary_inclusive: bool


@dataclass(frozen=True)
class ResolvedCapturePauseSamples:
    source_failure_fact_id: str
    interval: SampleIndexInterval
    closed: bool


@dataclass(frozen=True)
class ResolvedProgressSampleTimeline:
    episode_capture_identity_hash: str
    progress_episode_binding_hash: str
    clock_domain_id: str
    sample_count: int
    attempts: tuple[ResolvedProgressAttemptSamples, ...]
    capture_pauses: tuple[ResolvedCapturePauseSamples, ...]


def _attempt_interval(
    *,
    attempt: ResolvedProgressAttempt,
    transitioned_attempt_ids: frozenset[str],
    axis: EpisodeSampleAxis,
) -> tuple[SampleIndexInterval, bool]:
    if attempt.clock_domain_id != axis.clock_domain_id:
        raise SampleAxisError(
            f"Attempt {attempt.attempt_id!r} does not use the sample-axis clock."
        )
    if attempt.end_at is None:
        return axis.from_time_to_end(start_at=attempt.start_at), False
    if attempt.attempt_id in transitioned_attempt_ids:
        return (
            axis.half_open_time_interval(
                start_at=attempt.start_at,
                end_at=attempt.end_at,
            ),
            False,
        )
    return (
        axis.closed_time_interval(
            start_at=attempt.start_at,
            end_at=attempt.end_at,
        ),
        True,
    )


def _pause_interval(
    *,
    pause: ResolvedCapturePauseInterval,
    axis: EpisodeSampleAxis,
) -> SampleIndexInterval:
    if pause.clock_domain_id != axis.clock_domain_id:
        raise SampleAxisError(
            "Capture Pause does not use the sample-axis clock."
        )
    if pause.end_at is None:
        return axis.from_time_to_end(start_at=pause.start_at)
    return axis.half_open_time_interval(
        start_at=pause.start_at,
        end_at=pause.end_at,
    )


def project_progress_timeline_to_samples(
    *,
    timeline: ResolvedProgressTimeline,
    sample_axis: EpisodeSampleAxis,
) -> ResolvedProgressSampleTimeline:
    """Map Attempt and intentional Pause times without reading stream payloads."""

    if timeline.clock_domain_id != sample_axis.clock_domain_id:
        raise SampleAxisError(
            "Progress timeline and causal sample axis use different clocks."
        )
    transitioned_attempt_ids = frozenset(
        transition.from_attempt_id for transition in timeline.transitions
    )
    attempt_results: list[ResolvedProgressAttemptSamples] = []
    for attempt in timeline.attempts:
        interval, inclusive = _attempt_interval(
            attempt=attempt,
            transitioned_attempt_ids=transitioned_attempt_ids,
            axis=sample_axis,
        )
        attempt_results.append(
            ResolvedProgressAttemptSamples(
                attempt_id=attempt.attempt_id,
                progress_step_id=attempt.progress_step_id,
                outcome=attempt.outcome,
                interval=interval,
                end_boundary_inclusive=inclusive,
            )
        )
    pause_results = tuple(
        ResolvedCapturePauseSamples(
            source_failure_fact_id=pause.source_failure_fact_id,
            interval=_pause_interval(pause=pause, axis=sample_axis),
            closed=pause.closed,
        )
        for pause in timeline.capture_pause_intervals
    )
    return ResolvedProgressSampleTimeline(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        progress_episode_binding_hash=timeline.progress_episode_binding_hash,
        clock_domain_id=sample_axis.clock_domain_id,
        sample_count=sample_axis.sample_count,
        attempts=tuple(attempt_results),
        capture_pauses=pause_results,
    )


__all__ = [
    "ResolvedCapturePauseSamples",
    "ResolvedProgressAttemptSamples",
    "ResolvedProgressSampleTimeline",
    "project_progress_timeline_to_samples",
]
