"""Collection-facing adapter for declarative rule-based plans.

This module is intentionally a boundary adapter.  It resolves the canonical
``RobotAction`` values produced by ``robot_motion.replay.rule_based`` into the
existing ``TeleopCommand`` contract, while leaving dispatch, safety, IK,
session lifecycle, and episode persistence to their existing owners.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from robot_motion.action import RobotAction
from robot_motion.action.executor import ActionResolver, RobotActionResolver
from robot_motion.replay import (
    RuleBasedCollectionCursor,
    RuleBasedCollectionDirective,
)
from robot_motion.runtime import TeleopCommand


@dataclass(frozen=True)
class RuleBasedCollectionEmission:
    """One action and its capture intent for an existing collection loop."""

    directive: RuleBasedCollectionDirective
    action: RobotAction
    command: TeleopCommand

    @property
    def capture(self):
        """Return the per-step capture selection declared by the plan."""

        return self.directive.capture

    @property
    def recordable(self) -> bool:
        """Whether the caller should submit this step to its existing writer."""

        return bool(self.capture.enabled)

    def allows_stream(self, stream_id: str) -> bool:
        """Check the plan's allowlist without interpreting dataset schema."""

        return self.capture.allows(stream_id)


class RuleBasedCollectionAdapter(Iterator[RuleBasedCollectionEmission]):
    """Adapt a rule-based cursor to the existing command boundary.

    The adapter performs no dispatch.  A caller may send ``emission.command``
    through its normal session/runtime path and call its existing recording
    path only when ``emission.recordable`` is true.  Stream selection remains
    an explicit intent for that caller; this class never rewrites samples or
    writer jobs.
    """

    def __init__(
        self,
        cursor: RuleBasedCollectionCursor,
        *,
        resolver: ActionResolver | None = None,
    ) -> None:
        if not isinstance(cursor, RuleBasedCollectionCursor):
            raise TypeError("cursor must be RuleBasedCollectionCursor.")
        self.cursor = cursor
        self.resolver = resolver if resolver is not None else RobotActionResolver()

    def __iter__(self) -> RuleBasedCollectionAdapter:
        return self

    def __next__(self) -> RuleBasedCollectionEmission:
        directive = next(self.cursor)
        return self._emission_for(directive)

    def next_runtime(
        self,
        observation: object,
        *,
        context: object = None,
    ) -> RuleBasedCollectionEmission:
        """Resolve one observation-driven directive without owning its source."""

        directive = self.cursor.next_runtime(observation, context=context)
        return self._emission_for(directive)

    def _emission_for(
        self,
        directive: RuleBasedCollectionDirective,
    ) -> RuleBasedCollectionEmission:
        action = self._action_for(directive)
        command = self.resolver.resolve(action)
        return RuleBasedCollectionEmission(
            directive=directive,
            action=action,
            command=command,
        )

    def reset(self) -> None:
        """Reset the underlying finite plan for a deliberate second pass."""

        self.cursor.reset()

    @staticmethod
    def _action_for(directive: RuleBasedCollectionDirective) -> RobotAction:
        if directive.kind == "initialize":
            if not directive.qpos_by_side:
                raise ValueError("Initialization directive has no qpos targets.")
            return RobotAction.bimanual_qpos(
                left_arm=directive.qpos_by_side.get("left"),
                right_arm=directive.qpos_by_side.get("right"),
                source="rule_based_initialize",
                metadata=dict(directive.metadata),
            )
        if directive.action is None:
            raise ValueError(f"Directive {directive.step_id!r} has no action.")
        return directive.action


__all__ = [
    "RuleBasedCollectionAdapter",
    "RuleBasedCollectionEmission",
]
