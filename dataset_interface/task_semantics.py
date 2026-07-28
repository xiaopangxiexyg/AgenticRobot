"""Read-only projections of authoritative subtask-attempt event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dataset_contracts.task_semantics import (
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    validate_subtask_attempt_events,
)


@dataclass(frozen=True)
class TaskSemanticsFrameSnapshot:
    frame_index: int
    frame_time: float
    track_id: str
    active_attempt_id: str | None
    active_subtask_id: str | None
    previous_attempt_id: str | None
    previous_subtask_id: str | None
    previous_outcome: str | None
    source_event_sequence: int | None


def project_task_semantics_to_frames(
    *,
    profile: LongHorizonTaskProfile,
    events: Sequence[SubtaskAttemptEvent],
    frame_times: Sequence[float],
    frame_clock_domain_id: str,
    track_id: str,
) -> tuple[TaskSemanticsFrameSnapshot, ...]:
    """Materialize a rebuildable stage snapshot for one semantic track.

    ``effective_at`` controls the historical label.  ``available_at`` remains
    in the source event for causal-training filters; this projection does not
    erase or rewrite that provenance.
    """

    if track_id not in profile.track_ids:
        raise ValueError(f"Unknown task semantics track {track_id!r}.")
    validate_subtask_attempt_events(profile, events, require_closed=False)
    frame_clock = str(frame_clock_domain_id).strip()
    if not frame_clock:
        raise ValueError("frame_clock_domain_id must be non-empty.")
    event_clocks = {event.effective_clock_domain_id for event in events}
    if event_clocks and event_clocks != {frame_clock}:
        raise ValueError(
            "frame_clock_domain_id must match the task event effective clock domain."
        )
    times = tuple(float(value) for value in frame_times)
    if any(later < earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("frame_times must be monotonic.")
    relevant = tuple(
        sorted(
            (event for event in events if event.track_id == track_id),
            key=lambda event: (event.effective_at, event.event_sequence),
        )
    )
    cursor = 0
    active: SubtaskAttemptEvent | None = None
    previous: SubtaskAttemptEvent | None = None
    last_sequence: int | None = None
    snapshots: list[TaskSemanticsFrameSnapshot] = []
    for frame_index, frame_time in enumerate(times):
        while cursor < len(relevant) and relevant[cursor].effective_at <= frame_time:
            event = relevant[cursor]
            last_sequence = event.event_sequence
            if event.event_kind == "begin":
                active = event
            else:
                previous = event
                active = None
            cursor += 1
        snapshots.append(
            TaskSemanticsFrameSnapshot(
                frame_index=frame_index,
                frame_time=frame_time,
                track_id=track_id,
                active_attempt_id=None if active is None else active.attempt_id,
                active_subtask_id=None if active is None else active.subtask_id,
                previous_attempt_id=None if previous is None else previous.attempt_id,
                previous_subtask_id=None if previous is None else previous.subtask_id,
                previous_outcome=None if previous is None else previous.outcome,
                source_event_sequence=last_sequence,
            )
        )
    return tuple(snapshots)


__all__ = ["TaskSemanticsFrameSnapshot", "project_task_semantics_to_frames"]
