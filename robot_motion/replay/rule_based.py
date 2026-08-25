"""Declarative rule-based collection plans built on the existing replay API.

This module only expands a finite plan into typed action directives.  It does
not dispatch actions, own safety or IK, start devices, or write an episode.
Callers may pass each directive to their existing collection/runtime adapter;
the per-step capture selection tells that adapter which already-produced
streams belong in the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.action.schema import normalize_side

from .action_frame import replay_robot_action_at_index
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
    """Replay an explicit half-open frame range through existing replay helpers."""

    step_id: str
    group_key: str
    start_index: int
    end_index: int
    active_sides: tuple[str, ...] = ("left", "right")
    source: str = REPLAY_TRAJECTORY_ACTION_POSE
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


RuleBasedStep: TypeAlias = RuleBasedInitialize | RuleBasedReplay | RuleBasedCustomAction


@dataclass(frozen=True)
class RuleBasedCollectionPlan:
    """Ordered rule sequence; it is declarative and has no execution authority."""

    steps: tuple[RuleBasedStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("A rule-based collection plan requires at least one step.")
        if any(
            not isinstance(step, (RuleBasedInitialize, RuleBasedReplay, RuleBasedCustomAction))
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


class RuleBasedCollectionCursor(Iterator[RuleBasedCollectionDirective]):
    """Expand a plan once, without dispatching or persisting anything."""

    def __init__(
        self,
        plan: RuleBasedCollectionPlan,
        trajectory_groups: Mapping[str, ReplayTrajectoryGroup],
    ) -> None:
        if not isinstance(plan, RuleBasedCollectionPlan):
            raise TypeError("plan must be RuleBasedCollectionPlan.")
        self.plan = plan
        self.trajectory_groups = dict(trajectory_groups)
        self._step_index = 0
        self._directive_index = 0
        self._pending: Iterator[RuleBasedCollectionDirective] = iter(())
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

    def __iter__(self) -> RuleBasedCollectionCursor:
        return self

    def __next__(self) -> RuleBasedCollectionDirective:
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
                self._step_index += 1
                self._pending = self._expand_step(step, step_index=step_index)

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
                    },
                )
            return

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

    @staticmethod
    def _replay_action(
        group: ReplayTrajectoryGroup,
        step: RuleBasedReplay,
        *,
        frame_index: int,
    ) -> RobotAction:
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
    "RuleBasedCaptureSelection",
    "RuleBasedCollectionCursor",
    "RuleBasedCollectionDirective",
    "RuleBasedCollectionPlan",
    "RuleBasedCustomAction",
    "RuleBasedInitialize",
    "RuleBasedReplay",
]
