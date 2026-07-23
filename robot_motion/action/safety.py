"""Typed boundaries for source, shared canonical, and backend safety stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robot_motion.runtime.immutable import deep_freeze

from .proposal import freeze_robot_action
from .schema import RobotAction


@dataclass(frozen=True)
class SourceValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))


@dataclass(frozen=True)
class SharedCanonicalSafetyResult:
    input_action_id: str
    safe_action: RobotAction | None
    modified_components: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    status: str = "safe"
    safety_contract_id: str | None = None
    safety_contract_hash: str | None = None
    anchor_id: str | None = None
    anchor_kind: str | None = None
    anchor_hash: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"safe", "modified", "hold", "fault"}:
            raise ValueError("Unsupported shared safety status.")
        if self.safe_action is not None:
            object.__setattr__(self, "safe_action", freeze_robot_action(self.safe_action))
        object.__setattr__(self, "modified_components", tuple(self.modified_components))
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))
        if (self.safety_contract_id is None) != (self.safety_contract_hash is None):
            raise ValueError("Safety contract ID/hash must be provided together.")
        anchor_values = (self.anchor_id, self.anchor_kind, self.anchor_hash)
        if any(value is not None for value in anchor_values) and not all(
            value is not None for value in anchor_values
        ):
            raise ValueError("Safety anchor ID/kind/hash must be provided together.")
        object.__setattr__(self, "details", deep_freeze(dict(self.details)))


@dataclass(frozen=True)
class BackendGuardResult:
    accepted: bool
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "rejected", "hold", "fault"}:
            raise ValueError("Unsupported backend guard status.")
        if bool(self.accepted) != (self.status == "accepted"):
            raise ValueError("Backend guard accepted flag and status disagree.")
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))


__all__ = [
    "BackendGuardResult",
    "SharedCanonicalSafetyResult",
    "SourceValidationResult",
]
