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
from robot_motion.relative_action import pose_matrix_to_xyzquat

from .action_frame import (
    replay_action_metadata,
    replay_end_effector_targets_at_index,
    replay_robot_action_at_index,
)
from .fixed_trajectory import (
    build_fixed_trajectory_targets,
    normalize_fixed_trajectory_direction,
)
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
RULE_BASED_FIXED_TRAJECTORY_SCHEMA_V1 = "dexteleop.rule_based_fixed_trajectory.v1"


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


def _positive_float(value: object, *, name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
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
    # Appended after the historical fields to preserve positional construction
    # compatibility for callers that passed ``capture`` as the sixth argument.
    include_end_effectors: bool = False
    end_effector_sides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        object.__setattr__(self, "group_key", _required_text(self.group_key, name="group_key"))
        object.__setattr__(self, "frame_index", _non_negative_int(self.frame_index, name="frame_index"))
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "qpos_source", normalize_qpos_init_source(self.qpos_source))
        include_end_effectors = bool(self.include_end_effectors)
        end_effector_sides = (
            self.active_sides
            if include_end_effectors and not self.end_effector_sides
            else _normalize_sides(self.end_effector_sides)
            if self.end_effector_sides
            else ()
        )
        object.__setattr__(self, "include_end_effectors", include_end_effectors)
        object.__setattr__(self, "end_effector_sides", end_effector_sides)
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
    # Keep these additions after ``capture`` so historical positional callers
    # retain their original argument meaning.
    include_end_effectors: bool = False
    end_effector_sides: tuple[str, ...] = ()
    wait_for_terminal_arrival: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        object.__setattr__(self, "group_key", _required_text(self.group_key, name="group_key"))
        object.__setattr__(self, "start_index", _non_negative_int(self.start_index, name="start_index"))
        object.__setattr__(self, "end_index", _positive_int(self.end_index, name="end_index"))
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index.")
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "source", normalize_replay_trajectory_source(self.source))
        include_end_effectors = bool(self.include_end_effectors)
        end_effector_sides = (
            self.active_sides
            if include_end_effectors and not self.end_effector_sides
            else _normalize_sides(self.end_effector_sides)
            if self.end_effector_sides
            else ()
        )
        object.__setattr__(self, "include_end_effectors", include_end_effectors)
        object.__setattr__(self, "end_effector_sides", end_effector_sides)
        object.__setattr__(
            self,
            "wait_for_terminal_arrival",
            bool(self.wait_for_terminal_arrival),
        )
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
class RuleBasedFixedTrajectory:
    """Replay one catalogued canonical-desk path from a measured boundary.

    The trajectory provider is hardware-free. ``boundary_ref`` is resolved
    through the cursor's existing boundary provider at expansion time, so a
    caller can publish the actual end state of the preceding phase before the
    next directive is requested.
    """

    step_id: str
    trajectory_id: str
    boundary_ref: str
    scale_m: float
    linear_speed_m_s: float
    active_sides: tuple[str, ...] = ("left", "right")
    side: str = "right"
    direction: str = "up"
    yaw_rad: float | None = None
    catalog_path: str | None = None
    capture: RuleBasedCaptureSelection = field(default_factory=RuleBasedCaptureSelection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        object.__setattr__(
            self,
            "trajectory_id",
            _required_text(self.trajectory_id, name="trajectory_id"),
        )
        object.__setattr__(
            self,
            "boundary_ref",
            _required_text(self.boundary_ref, name="boundary_ref"),
        )
        object.__setattr__(self, "scale_m", _positive_float(self.scale_m, name="scale_m"))
        object.__setattr__(
            self,
            "linear_speed_m_s",
            _positive_float(self.linear_speed_m_s, name="linear_speed_m_s"),
        )
        active_sides = _normalize_sides(self.active_sides)
        object.__setattr__(self, "active_sides", active_sides)
        side = normalize_side(str(self.side).strip().lower())
        if side not in active_sides:
            raise ValueError("side must be included in active_sides.")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "direction",
            normalize_fixed_trajectory_direction(self.direction),
        )
        if self.yaw_rad is not None:
            yaw_rad = float(self.yaw_rad)
            if not np.isfinite(yaw_rad):
                raise ValueError("yaw_rad must be finite.")
            object.__setattr__(self, "yaw_rad", yaw_rad)
        if self.catalog_path is not None:
            object.__setattr__(self, "catalog_path", str(self.catalog_path))
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
    allow_empty: bool = False
    planned_steps_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, name="step_id"))
        if not callable(getattr(self.action_source, "produce_action", None)):
            raise TypeError("action_source must implement produce_action().")
        if not isinstance(self.stop_condition, RuleBasedSegmentStopCondition):
            raise TypeError("stop_condition must be RuleBasedSegmentStopCondition.")
        if not isinstance(self.capture, RuleBasedCaptureSelection):
            raise TypeError("capture must be RuleBasedCaptureSelection.")
        object.__setattr__(self, "allow_empty", bool(self.allow_empty))
        if self.planned_steps_hint is not None:
            planned_steps_hint = _non_negative_int(
                self.planned_steps_hint,
                name="planned_steps_hint",
            )
            if planned_steps_hint > self.stop_condition.max_steps:
                raise ValueError("planned_steps_hint must not exceed max_steps.")
            object.__setattr__(self, "planned_steps_hint", planned_steps_hint)


RuleBasedStep: TypeAlias = (
    RuleBasedInitialize
    | RuleBasedReplay
    | RuleBasedCustomAction
    | RuleBasedFixedTrajectory
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
                    RuleBasedFixedTrajectory,
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
    # Appended after all historical fields to avoid reinterpreting positional
    # construction of collection directives.
    hand_qpos_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"initialize", "replay", "custom_action", "fixed_trajectory"}:
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
        object.__setattr__(
            self,
            "hand_qpos_by_side",
            {
                str(side): np.asarray(values, dtype=float).reshape(-1).copy()
                for side, values in self.hand_qpos_by_side.items()
            },
        )
        if any(
            values.shape != (6,) or not np.all(np.isfinite(values))
            for values in self.hand_qpos_by_side.values()
        ):
            raise ValueError("hand_qpos_by_side values must contain six finite values.")
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
        for step in self.plan.steps:
            if not isinstance(step, RuleBasedActionSegment):
                continue
            reset_source = getattr(step.action_source, "reset", None)
            if callable(reset_source):
                reset_source()
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
            if step.allow_empty and self._active_segment_emitted == 0:
                self._active_segment_terminal = True
                self._active_segment_stop_reason = "empty"
                self._finish_active_segment()
                return self.next_runtime(observation, context=context)
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
                "segment_max_steps": int(step.stop_condition.max_steps),
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
            hand_targets: dict[str, np.ndarray] = {}
            hand_metadata: dict[str, Any] = {}
            if step.include_end_effectors:
                hand_sides = step.end_effector_sides or step.active_sides
                hand_targets = replay_end_effector_targets_at_index(
                    group,
                    hand_sides,
                    index=int(step.frame_index),
                )
                if set(hand_targets) != set(hand_sides):
                    missing = sorted(set(hand_sides) - set(hand_targets))
                    raise ValueError(
                        f"Initialization has no finite end-effector target for side(s): {missing}."
                    )
                hand_metadata = replay_action_metadata(
                    group,
                    active_sides=hand_sides,
                )
            yield RuleBasedCollectionDirective(
                sequence_index=step_index,
                step_id=step.step_id,
                kind="initialize",
                capture=step.capture,
                qpos_by_side=targets,
                hand_qpos_by_side=hand_targets,
                group_key=step.group_key,
                replay_frame_index=step.frame_index,
                metadata={
                    "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                    "rule_based_step_index": int(step_index),
                    "qpos_source_by_side": dict(sources),
                    **hand_metadata,
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
                terminal = bool(
                    step.wait_for_terminal_arrival
                    and frame_index == int(step.end_index) - 1
                )
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
                    segment_terminal=terminal,
                    segment_stop_reason=(
                        "replay_terminal_arrival" if terminal else None
                    ),
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
        if isinstance(step, RuleBasedFixedTrajectory):
            boundary = self._boundary_for(step.boundary_ref)
            missing_sides = sorted(set(step.active_sides) - set(boundary.tcp_pose_by_side))
            if missing_sides:
                raise ValueError(
                    f"Fixed trajectory {step.step_id!r} boundary is missing side(s): "
                    f"{missing_sides}"
                )
            noncanonical_frames = {
                side: boundary.pose_frame_by_side[side]
                for side in step.active_sides
                if boundary.pose_frame_by_side.get(side) != "canonical_desk"
            }
            if noncanonical_frames:
                raise ValueError(
                    "Fixed trajectory requires canonical_desk boundary poses; "
                    f"got {noncanonical_frames}"
                )
            targets = build_fixed_trajectory_targets(
                boundary.tcp_pose_by_side,
                step.trajectory_id,
                scale_m=step.scale_m,
                linear_speed_m_s=step.linear_speed_m_s,
                side=step.side,
                direction=step.direction,
                yaw_rad=step.yaw_rad,
                catalog_path=step.catalog_path,
            )
            for trajectory_index, target in enumerate(targets):
                poses = {
                    side: pose_matrix_to_xyzquat(target.pose_by_side[side])
                    for side in step.active_sides
                }
                action = RobotAction.bimanual_ee_pose(
                    left_arm=poses.get("left"),
                    right_arm=poses.get("right"),
                    timestamp=float(target.timestamp),
                    source="rule_based_fixed_trajectory",
                    metadata={
                        **dict(target.metadata),
                        "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                        "fixed_trajectory_schema": RULE_BASED_FIXED_TRAJECTORY_SCHEMA_V1,
                        "rule_based_step_index": int(step_index),
                        "fixed_trajectory_index": int(trajectory_index),
                        "fixed_trajectory_boundary_ref": step.boundary_ref,
                    },
                )
                yield RuleBasedCollectionDirective(
                    sequence_index=step_index,
                    step_id=step.step_id,
                    kind="fixed_trajectory",
                    capture=step.capture,
                    action=action,
                    metadata={
                        **dict(target.metadata),
                        "rule_based_schema": RULE_BASED_COLLECTION_PLAN_SCHEMA_V1,
                        "fixed_trajectory_schema": RULE_BASED_FIXED_TRAJECTORY_SCHEMA_V1,
                        "rule_based_step_index": int(step_index),
                        "fixed_trajectory_index": int(trajectory_index),
                        "fixed_trajectory_boundary_ref": step.boundary_ref,
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
                include_end_effectors=bool(step.include_end_effectors),
                end_effector_sides=step.end_effector_sides,
                source="rule_based_relative_replay",
            )
        if step.source == REPLAY_TRAJECTORY_ACTION_POSE:
            return replay_robot_action_at_index(
                group,
                step.active_sides,
                index=frame_index,
                include_end_effectors=bool(step.include_end_effectors),
                end_effector_sides=step.end_effector_sides,
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
        hand_targets = (
            replay_end_effector_targets_at_index(
                group,
                step.end_effector_sides or step.active_sides,
                index=frame_index,
            )
            if step.include_end_effectors
            else {}
        )
        return RobotAction.bimanual_qpos(
            left_arm=targets.get("left"),
            right_arm=targets.get("right"),
            left_hand=hand_targets.get("left"),
            right_hand=hand_targets.get("right"),
            source="rule_based_replay",
            metadata={"replay_frame_index": int(frame_index)},
        )


__all__ = [
    "RULE_BASED_COLLECTION_PLAN_SCHEMA_V1",
    "RULE_BASED_FIXED_TRAJECTORY_SCHEMA_V1",
    "RuleBasedActionSegment",
    "RuleBasedBoundaryProvider",
    "RuleBasedCaptureSelection",
    "RuleBasedCollectionCursor",
    "RuleBasedCollectionDirective",
    "RuleBasedCollectionPlan",
    "RuleBasedCustomAction",
    "RuleBasedFixedTrajectory",
    "RuleBasedInitialize",
    "RuleBasedReplay",
    "RuleBasedSegmentContext",
    "RuleBasedSegmentStopCondition",
]
