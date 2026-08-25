"""Declarative rule-based collection plans built on the existing replay API.

This module only expands a finite plan into typed action directives.  It does
not dispatch actions, own safety or IK, start devices, or write an episode.
Callers may pass each directive to their existing collection/runtime adapter;
the per-step capture selection tells that adapter which already-produced
streams belong in the dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.action.schema import normalize_side
from robot_motion.action.sources import RobotActionSource

from .action_frame import replay_robot_action_at_index
from .rule_based_relative import (
    RuleBasedBoundaryState,
    rebase_replay_action_at_boundary,
)
from .trajectory import (
    QPOS_INIT_AUTO,
    QPOS_INIT_REPLAY_QPOS,
    REPLAY_TRAJECTORY_ACTION_POSE,
    ReplayTrajectoryGroup,
    normalize_qpos_init_source,
    normalize_replay_trajectory_source,
    select_replay_qpos_init_targets,
)

RULE_BASED_COLLECTION_PLAN_SCHEMA_V1 = "dexteleop.rule_based_collection_plan.v1"


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    normalized = _non_negative_int(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _normalize_sides(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_side(str(value).strip().lower()) for value in values)
    if not normalized:
        raise ValueError("active_sides must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("active_sides must not contain duplicates.")
    return normalized


@dataclass(frozen=True)
class RuleBasedCaptureSelection:
    """Per-step capture intent; storage remains owned by the existing writer."""

    enabled: bool = True
    stream_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stream_ids = tuple(_required_text(value, name="stream_id") for value in self.stream_ids)
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream_ids must not contain duplicates.")
        object.__setattr__(self, "stream_ids", stream_ids)

    def allows(self, stream_id: str) -> bool:
        """Return whether an existing producer stream is selected for capture."""

        return bool(self.enabled) and (
            not self.stream_ids or str(stream_id) in set(self.stream_ids)
        )


@dataclass(frozen=True)
class RuleBasedInitialize:
    """Move through the existing initialization path to one replay qpos frame."""

    step_id: str
    group_key: str
    frame_index: int = 0
    active_sides: tuple[str, ...] = ("left", "right")
    qpos_source: str = QPOS_INIT_AUTO
    capture: RuleBasedCaptureSelection = field(
        default_factory=lambda: RuleBasedCaptureSelection(enabled=False)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        object.__setattr__(self, "group_key", _required_text(self.group_key, name="group_key"))
        object.__setattr__(self, "frame_index", _non_negative_int(self.frame_index, name="frame_index"))
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "qpos_source", normalize_qpos_init_source(self.qpos_source))
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")


@dataclass(frozen=True)
class RuleBasedReplay:
    """Replay an explicit range, optionally rebased at a runtime boundary."""

    step_id: str
    group_key: str
    start_index: int
    end_index: int
    active_sides: tuple[str, ...] = ("left", "right")
    source: str = REPLAY_TRAJECTORY_ACTION_POSE
    relative_to: str | None = None
    source_start_index: int | None = None
    capture: RuleBasedCaptureSelection = field(default_factory=RuleBasedCaptureSelection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        object.__setattr__(self, "group_key", _required_text(self.group_key, name="group_key"))
        object.__setattr__(self, "start_index", _non_negative_int(self.start_index, name="start_index"))
        object.__setattr__(self, "end_index", _positive_int(self.end_index, name="end_index"))
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index.")
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "source", normalize_replay_trajectory_source(self.source))
        relative_to = self.relative_to
        if relative_to is not None:
            relative_to = _required_text(relative_to, name="relative_to")
            if self.source != REPLAY_TRAJECTORY_ACTION_POSE:
                raise ValueError("Relative replay currently supports pose actions only.")
        object.__setattr__(self, "relative_to", relative_to)
        source_start_index = self.source_start_index
        if source_start_index is not None:
            source_start_index = _non_negative_int(
                source_start_index,
                name="source_start_index",
            )
        object.__setattr__(self, "source_start_index", source_start_index)
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")


@dataclass(frozen=True)
class RuleBasedCustomAction:
    """Repeat one existing canonical RobotAction for a bounded number of ticks."""

    step_id: str
    action: RobotAction
    repeats: int = 1
    capture: RuleBasedCaptureSelection = field(default_factory=RuleBasedCaptureSelection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        if not isinstance(self.action, RobotAction):
            raise TypeError("action must be RobotAction.")
        object.__setattr__(self, "repeats", _positive_int(self.repeats, name="repeats"))
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")


@dataclass(frozen=True)
class RuleBasedSegmentContext:
    """Inputs supplied to a runtime rule-based segment stop predicate."""

    observation: Any
    action: RobotAction
    emitted_steps: int
    plan_step_index: int
    context: Any = None


@dataclass(frozen=True)
class RuleBasedSegmentStopCondition:
    """A bounded, editable stop rule for a runtime action segment.

    ``max_steps`` is always a hard upper bound.  ``stop_when`` is evaluated
    after each produced action and may terminate the segment earlier, subject
    to ``min_steps``.  This keeps a user-provided predicate from accidentally
    creating an unbounded collection loop.
    """

    max_steps: int
    min_steps: int = 1
    stop_when: Callable[[RuleBasedSegmentContext], bool] | None = None
    name: str = "rule_based_segment"

    def __post_init__(self) -> None:
        max_steps = _positive_int(self.max_steps, name="max_steps")
        min_steps = _non_negative_int(self.min_steps, name="min_steps")
        if min_steps > max_steps:
            raise ValueError("min_steps must be less than or equal to max_steps.")
        if self.stop_when is not None and not callable(self.stop_when):
            raise TypeError("stop_when must be callable when provided.")
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "min_steps", min_steps)
        object.__setattr__(self, "name", _required_text(self.name, name="name"))

    def evaluate(self, context: RuleBasedSegmentContext) -> tuple[bool, str | None]:
        """Return ``(terminal, reason)`` after one action was produced."""

        if not isinstance(context, RuleBasedSegmentContext):
            raise TypeError("context must be RuleBasedSegmentContext.")
        if context.emitted_steps >= self.max_steps:
            return True, "max_steps"
        if context.emitted_steps < self.min_steps:
            return False, None
        if self.stop_when is not None and bool(self.stop_when(context)):
            return True, "predicate"
        return False, None


@dataclass(frozen=True)
class RuleBasedActionSegment:
    """Run an existing canonical action source for a bounded capture segment.

    The source is invoked by :meth:`RuleBasedCollectionCursor.next_runtime`;
    this object does not start or close it.  Source lifecycle, observation
    acquisition, dispatch, safety, and persistence remain owned by their
    existing runtime owners.
    """

    step_id: str
    action_source: RobotActionSource
    stop_condition: RuleBasedSegmentStopCondition
    capture: RuleBasedCaptureSelection = field(default_factory=RuleBasedCaptureSelection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        if not callable(getattr(self.action_source, "produce_action", None)):
            raise TypeError("action_source must implement produce_action().")
        if not isinstance(self.stop_condition, RuleBasedSegmentStopCondition):
            raise TypeError("stop_condition must be RuleBasedSegmentStopCondition.")
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")


RuleBasedStep: TypeAlias = (
    RuleBasedInitialize
    | RuleBasedReplay
    | RuleBasedCustomAction
    | RuleBasedActionSegment
)
RuleBasedBoundaryProvider: TypeAlias = (
    Mapping[str, RuleBasedBoundaryState]
    | Callable[[str], RuleBasedBoundaryState]
)


@dataclass(frozen=True)
class RuleBasedCollectionPlan:
    """Ordered rule sequence; it is declarative and has no execution authority."""

    steps: tuple[RuleBasedStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("A rule-based collection plan requires at least one step.")
        if any(
            not isinstance(
                step,
                (
                    RuleBasedInitialize,
                    RuleBasedReplay,
                    RuleBasedCustomAction,
                    RuleBasedActionSegment,
                ),
            )
            for step in steps
        ):
            raise TypeError("steps must contain supported rule-based step objects.")
        step_ids = tuple(step.step_id for step in steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step_id values must be unique within a plan.")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True)
class RuleBasedCollectionDirective:
    """One plan output for an existing collection/runtime adapter."""

    sequence_index: int
    step_id: str
    kind: str
    capture: RuleBasedCaptureSelection
    action: RobotAction | None = None
    qpos_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)
    group_key: str | None = None
    replay_frame_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    segment_terminal: bool = False
    segment_stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"initialize", "replay", "custom_action"}:
            raise ValueError(f"Unsupported directive kind {self.kind!r}.")
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")
        if self.kind == "initialize" and (
            self.action is not None or not self.qpos_by_side
        ):
            raise ValueError("initialize directives require qpos_by_side only.")
        if self.kind != "initialize" and self.action is None:
            raise ValueError(f"{self.kind} directives require action.")
        object.__setattr__(
            self,
            "qpos_by_side",
            {
                str(side): np.asarray(values, dtype=float).reshape(-1).copy()
                for side, values in self.qpos_by_side.items()
            },
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.segment_stop_reason is not None:
            object.__setattr__(
                self,
                "segment_stop_reason",
                _required_text(self.segment_stop_reason, name="segment_stop_reason"),
            )
        if self.segment_stop_reason is not None and not self.segment_terminal:
            raise ValueError("segment_stop_reason requires segment_terminal=True.")


class RuleBasedCollectionCursor(Iterator[RuleBasedCollectionDirective]):
    """Expand a plan once, without dispatching or persisting anything."""

    def __init__(
        self,
        plan: RuleBasedCollectionPlan,
        trajectory_groups: Mapping[str, ReplayTrajectoryGroup],
        *,
        boundary_provider: RuleBasedBoundaryProvider | None = None,
    ) -> None:
        if not isinstance(plan, RuleBasedCollectionPlan):
            raise TypeError("plan must be RuleBasedCollectionPlan.")
        self.plan = plan
        self.trajectory_groups = dict(trajectory_groups)
        self.boundary_provider = boundary_provider
        self._step_index = 0
        self._directive_index = 0
        self._pending: Iterator[RuleBasedCollectionDirective] = iter(())
        self._active_segment: RuleBasedActionSegment | None = None
        self._active_segment_plan_index: int | None = None
        self._active_segment_emitted = 0
        self._active_segment_terminal = False
        self._active_segment_stop_reason: str | None = None
        for step in plan.steps:
            if (
                isinstance(step, (RuleBasedInitialize, RuleBasedReplay))
                and step.group_key not in self.trajectory_groups
            ):
                raise KeyError(
                    f"Missing trajectory group {step.group_key!r} for step {step.step_id!r}."
                )

    @property
    def done(self) -> bool:
        return self._step_index >= len(self.plan.steps)

    def reset(self) -> None:
        self._step_index = 0
        self._directive_index = 0
        self._pending = iter(())
        self._active_segment = None
        self._active_segment_plan_index = None
        self._active_segment_emitted = 0
        self._active_segment_terminal = False
        self._active_segment_stop_reason = None

    def __iter__(self) -> RuleBasedCollectionCursor:
        return self

    def __next__(self) -> RuleBasedCollectionDirective:
        if self._active_segment is not None:
            if not self._active_segment_terminal:
                raise TypeError(
                    "A runtime action segment requires next_runtime(observation, context=...)."
                )
            self._finish_active_segment()
        while True:
            try:
                directive = next(self._pending)
                result = replace(directive, sequence_index=self._directive_index)
                self._directive_index += 1
                return result
            except StopIteration:
                if self.done:
                    raise
                step = self.plan.steps[self._step_index]
                step_index = self._step_index
                if isinstance(step, RuleBasedActionSegment):
                    raise TypeError(
                        "A runtime action segment requires next_runtime(observation, context=...)."
                    )
                self._step_index += 1
                self._pending = self._expand_step(step, step_index=step_index)

    def next_runtime(
        self,
        observation: Any,
        *,
        context: Any = None,
    ) -> RuleBasedCollectionDirective:
        """Produce one directive using the current runtime observation.

        Static plan steps are emitted exactly as with :func:`next`.  A
        :class:`RuleBasedActionSegment` is driven one action at a time through
        the existing ``RobotActionSource`` contract.  Its terminal directive
        is returned before the cursor advances, so the caller can capture and
        publish a measured boundary before requesting a following relative
        replay step.
        """

        while True:
            if self._pending is not None:
                try:
                    directive = next(self._pending)
                    result = replace(directive, sequence_index=self._directive_index)
                    self._directive_index += 1
                    return result
                except StopIteration:
                    pass

            if self._active_segment is not None:
                if self._active_segment_terminal:
                    self._finish_active_segment()
                    continue
                return self._next_runtime_segment_action(observation, context=context)

            if self.done:
                raise StopIteration
            step = self.plan.steps[self._step_index]
            step_index = self._step_index
            if isinstance(step, RuleBasedActionSegment):
                self._active_segment = step
                self._active_segment_plan_index = step_index
                self._active_segment_emitted = 0
                self._active_segment_terminal = False
                self._active_segment_stop_reason = None
                continue
            self._step_index += 1
            self._pending = self._expand_step(step, step_index=step_index)

    def _finish_active_segment(self) -> None:
        if self._active_segment is None or self._active_segment_plan_index is None:
            raise RuntimeError("No active runtime action segment is available to finish.")
        self._step_index = self._active_segment_plan_index + 1
        self._active_segment = None
        self._active_segment_plan_index = None
        self._active_segment_emitted = 0
        self._active_segment_terminal = False
        self._active_segment_stop_reason = None

    def _next_runtime_segment_action(
        self,
        observation: Any,
        *,
        context: Any,
    ) -> RuleBasedCollectionDirective:
        step = self._active_segment
        step_index = self._active_segment_plan_index
        if step is None or step_index is None:
            raise RuntimeError("No active runtime action segment is available.")
        action = step.action_source.produce_action(observation, context=context)
        if action is None:
            raise ValueError(
                f"Runtime action source for segment {step.step_id!r} returned None; "
                "use stop_when to terminate after an explicit action."
            )
        if not isinstance(action, RobotAction):
            raise TypeError(
                f"Runtime action source for segment {step.step_id!r} must return RobotAction."
            )
        self._active_segment_emitted += 1
        segment_context = RuleBasedSegmentContext(
            observation=observation,
            action=action,
            emitted_steps=self._active_segment_emitted,
            plan_step_index=step_index,
            context=context,
        )
        terminal, reason = step.stop_condition.evaluate(segment_context)
        self._active_segment_terminal = bool(terminal)
        self._active_segment_stop_reason = reason
        source_name = str(getattr(step.action_source, "name", "")).strip()
        if not source_name:
            source_name = step.action_source.__class__.__name__
        return RuleBasedCollectionDirective(
            sequence_index=step_index,
            step_id=step.step_id,
            kind="custom_action",
            capture=step.capture,
            action=action,
            metadata={
                "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                "rule_based_step_index": int(step_index),
                "custom_action_source": source_name,
                "custom_action_mode": "runtime_source",
                "segment_emitted_steps": int(self._active_segment_emitted),
                "segment_stop_condition": step.stop_condition.name,
            },
            segment_terminal=self._active_segment_terminal,
            segment_stop_reason=self._active_segment_stop_reason,
        )

    def _expand_step(
        self,
        step: RuleBasedStep,
        *,
        step_index: int,
    ) -> Iterator[RuleBasedCollectionDirective]:
        if isinstance(step, RuleBasedInitialize):
            group = self.trajectory_groups[step.group_key]
            if step.frame_index >= int(group.frame_count):
                raise IndexError(
                    f"Initialization frame {step.frame_index} is outside the trajectory."
                )
            targets, sources = select_replay_qpos_init_targets(
                group,
                step.active_sides,
                index=step.frame_index,
                source=step.qpos_source,
            )
            if set(targets) != set(step.active_sides):
                missing = sorted(set(step.active_sides) - set(targets))
                raise ValueError(f"Initialization has no finite qpos target for side(s): {missing}.")
            yield RuleBasedCollectionDirective(
                sequence_index=step_index,
                step_id=step.step_id,
                kind="initialize",
                capture=step.capture,
                qpos_by_side=targets,
                group_key=step.group_key,
                replay_frame_index=step.frame_index,
                metadata={
                    "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                    "rule_based_step_index": int(step_index),
                    "qpos_source_by_side": dict(sources),
                },
            )
            return

        if isinstance(step, RuleBasedReplay):
            group = self.trajectory_groups[step.group_key]
            if step.end_index > int(group.frame_count):
                raise IndexError(
                    f"Replay range [{step.start_index}, {step.end_index}) exceeds "
                    f"trajectory frame_count={group.frame_count}."
                )
            for frame_index in range(step.start_index, step.end_index):
                action = self._replay_action(group, step, frame_index=frame_index)
                yield RuleBasedCollectionDirective(
                    sequence_index=step_index,
                    step_id=step.step_id,
                    kind="replay",
                    capture=step.capture,
                    action=action,
                    group_key=step.group_key,
                    replay_frame_index=frame_index,
                    metadata={
                        "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                        "rule_based_step_index": int(step_index),
                        "replay_source": step.source,
                        **(
                            {
                                "relative_to": step.relative_to,
                                "source_start_index": int(
                                    step.source_start_index
                                    if step.source_start_index is not None
                                    else step.start_index
                                ),
                            }
                            if step.relative_to is not None
                            else {}
                        ),
                    },
                )
            return

        if isinstance(step, RuleBasedCustomAction):
            for _ in range(step.repeats):
                yield RuleBasedCollectionDirective(
                    sequence_index=step_index,
                    step_id=step.step_id,
                    kind="custom_action",
                    capture=step.capture,
                    action=step.action,
                    metadata={
                        "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                        "rule_based_step_index": int(step_index),
                        "custom_action_source": step.action.source,
                    },
                )
            return
        if isinstance(step, RuleBasedActionSegment):
            raise TypeError(
                "Runtime action segments must be expanded with next_runtime()."
            )
        raise TypeError(f"Unsupported rule-based step {type(step).__name__}.")

    def _boundary_for(self, boundary_ref: str) -> RuleBasedBoundaryState:
        provider = self.boundary_provider
        if provider is None:
            raise ValueError(
                f"Relative replay requires boundary_provider for {boundary_ref!r}."
            )
        if callable(provider):
            boundary = provider(str(boundary_ref))
        else:
            boundary = provider.get(str(boundary_ref))
        if not isinstance(boundary, RuleBasedBoundaryState):
            raise KeyError(
                f"Boundary provider has no RuleBasedBoundaryState for {boundary_ref!r}."
            )
        return boundary

    def _replay_action(
        self,
        group: ReplayTrajectoryGroup,
        step: RuleBasedReplay,
        *,
        frame_index: int,
    ) -> RobotAction:
        if step.relative_to is not None:
            boundary = self._boundary_for(step.relative_to)
            return rebase_replay_action_at_boundary(
                group,
                step.active_sides,
                index=frame_index,
                source_start_index=(
                    step.source_start_index
                    if step.source_start_index is not None
                    else step.start_index
                ),
                boundary=boundary,
                source="rule_based_relative_replay",
            )
        if step.source == REPLAY_TRAJECTORY_ACTION_POSE:
            return replay_robot_action_at_index(
                group,
                step.active_sides,
                index=frame_index,
                source="rule_based_replay",
            )
        targets, _ = select_replay_qpos_init_targets(
            group,
            step.active_sides,
            index=frame_index,
            source=QPOS_INIT_REPLAY_QPOS,
        )
        if set(targets) != set(step.active_sides):
            missing = sorted(set(step.active_sides) - set(targets))
            raise ValueError(f"Replay has no finite qpos target for side(s): {missing}.")
        return RobotAction.bimanual_qpos(
            left_arm=targets.get("left"),
            right_arm=targets.get("right"),
            source="rule_based_replay",
            metadata={"replay_frame_index": int(frame_index)},
        )


__all__ = [
    "RULE_BASED_COLLECTION_PLAN_SCHEMA_V1",
    "RuleBasedActionSegment",
    "RuleBasedBoundaryProvider",
    "RuleBasedCaptureSelection",
    "RuleBasedCollectionCursor",
    "RuleBasedCollectionDirective",
    "RuleBasedCollectionPlan",
    "RuleBasedCustomAction",
    "RuleBasedInitialize",
    "RuleBasedReplay",
    "RuleBasedSegmentContext",
    "RuleBasedSegmentStopCondition",
]
