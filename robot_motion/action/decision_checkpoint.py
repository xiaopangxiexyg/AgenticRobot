"""Exact scheduler/publication history at a completed decision boundary.

The caller excludes trigger, make_tick and publish for the entire cut. These
owners have no internal transaction lock. Restoring never makes a new tick or
publication, rebases the grid, or replays coalesced trigger evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .arbitration_runtime import (
    ArbitrationRuntime,
    DecisionScheduler,
    DecisionScheduleSpec,
)

_SCHEDULER_LAYOUT = {
    "spec",
    "_clock",
    "_next_due_at",
    "_tick_index",
    "_pending_triggers",
    "_pending_trigger_count",
}
_PUBLICATION_LAYOUT = {
    "spec",
    "_last_published_at",
    "_last_published_target_effective_at",
}


@dataclass(frozen=True)
class DecisionCheckpoint:
    spec_hash: str
    captured_at: float
    next_due_at: float | None
    tick_index: int
    pending_triggers: tuple[tuple[str, str], ...]
    pending_trigger_count: int
    last_published_at: float | None
    last_published_target_effective_at: float | None


@dataclass(frozen=True)
class RestoredDecisionOwners:
    scheduler: DecisionScheduler
    publication: ArbitrationRuntime


def _validated(state, spec):
    if (
        type(state) is not DecisionCheckpoint
        or type(spec) is not DecisionScheduleSpec
        or state.spec_hash != spec.semantic_content_hash
        or type(state.captured_at) is not float
        or not math.isfinite(state.captured_at)
        or type(state.tick_index) is not int
        or state.tick_index < 0
        or type(state.pending_trigger_count) is not int
        or state.pending_trigger_count < 0
        or type(state.pending_triggers) is not tuple
        or len(state.pending_triggers)
        != min(state.pending_trigger_count, int(spec.max_pending_trigger_evidence))
        or (state.next_due_at is None) != (state.tick_index == 0)
        or (state.last_published_at is None)
        != (state.last_published_target_effective_at is None)
    ):
        raise ValueError("Invalid decision checkpoint or schedule binding.")
    for value in (
        state.next_due_at,
        state.last_published_at,
        state.last_published_target_effective_at,
    ):
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("Invalid decision timeline.")
    for item in state.pending_triggers:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str for value in item)
        ):
            raise ValueError("Invalid pending trigger evidence.")
    return state


def capture_decision_owners(
    scheduler: DecisionScheduler, publication: ArbitrationRuntime
) -> DecisionCheckpoint:
    """Capture caller-fenced owners, preserving truncated evidence and counts."""
    if (
        type(scheduler) is not DecisionScheduler
        or set(vars(scheduler)) != _SCHEDULER_LAYOUT
        or type(publication) is not ArbitrationRuntime
        or set(vars(publication)) != _PUBLICATION_LAYOUT
        or type(scheduler.spec) is not DecisionScheduleSpec
        or type(publication.spec) is not DecisionScheduleSpec
        or scheduler.spec.semantic_content_hash
        != publication.spec.semantic_content_hash
    ):
        raise ValueError("Decision owner layout or shared schedule differs.")
    return _validated(
        DecisionCheckpoint(
            spec_hash=scheduler.spec.semantic_content_hash,
            captured_at=float(scheduler._clock()),
            next_due_at=scheduler._next_due_at,
            tick_index=scheduler._tick_index,
            pending_triggers=tuple(scheduler._pending_triggers),
            pending_trigger_count=scheduler._pending_trigger_count,
            last_published_at=publication._last_published_at,
            last_published_target_effective_at=publication._last_published_target_effective_at,
        ),
        scheduler.spec,
    )


def restore_decision_owners(
    state: DecisionCheckpoint, *, spec: DecisionScheduleSpec, clock: Callable[[], float]
) -> RestoredDecisionOwners:
    """Build fresh owners with the same grid, publication gate and caller clock."""
    _validated(state, spec)
    if not callable(clock) or clock() != state.captured_at:
        raise ValueError("Destination logical decision clock differs.")
    scheduler = DecisionScheduler(spec, clock=clock)
    publication = ArbitrationRuntime(spec)
    scheduler._next_due_at = state.next_due_at
    scheduler._tick_index = state.tick_index
    scheduler._pending_triggers = list(state.pending_triggers)
    scheduler._pending_trigger_count = state.pending_trigger_count
    publication._last_published_at = state.last_published_at
    publication._last_published_target_effective_at = (
        state.last_published_target_effective_at
    )
    return RestoredDecisionOwners(scheduler, publication)
