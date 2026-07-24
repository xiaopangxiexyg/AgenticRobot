"""Narrow dependency-inversion ports for control composition.

The protocols deliberately expose existing immutable lower-runtime objects.
They do not grant authority or motion permission: proposal admission remains a
RobotMotion responsibility and trace persistence remains an injected owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dataset_contracts.control_trace import ControlTraceFact
    from robot_motion.action.proposal import (
        ActionProposalDraft,
        ProposalAdmissionReceipt,
    )


@runtime_checkable
class CompositionClock(Protocol):
    """Return time in the runtime's explicitly bound monotonic clock domain."""

    def __call__(self) -> float:
        ...


@runtime_checkable
class CompositionTraceEmitter(Protocol):
    """Publish an atomic batch of non-owning ControlTrace facts."""

    def __call__(self, facts: tuple["ControlTraceFact", ...]) -> None:
        ...


@runtime_checkable
class CompositionProposalAdmitter(Protocol):
    """Submit one composed proposal draft to the canonical admission owner."""

    def __call__(
        self,
        draft: "ActionProposalDraft",
        *,
        source_event_id: str,
    ) -> "ProposalAdmissionReceipt":
        ...


__all__ = [
    "CompositionClock",
    "CompositionProposalAdmitter",
    "CompositionTraceEmitter",
]
