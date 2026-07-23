"""Dataset-scoped DM-Tac gripper force profile contract.

The force selected from the GUI is a dataset property once it is explicitly
saved.  Keeping this contract independent from the task profile lets the
collector reject a force change before an episode can be appended.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


GRIPPER_FORCE_PROFILE_SCHEMA_V1 = "dexteleop.gripper_force_profile.v1"


def _normalized_force_by_side(values: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    for raw_side, raw_force in values.items():
        side = str(raw_side).strip()
        if side not in {"left", "right"}:
            raise ValueError(f"gripper force profile side must be left/right, got {side!r}")
        if isinstance(raw_force, bool):
            raise ValueError(f"gripper force for {side} must be a number in [0, 1]")
        force = float(raw_force)
        if not math.isfinite(force) or not 0.0 <= force <= 1.0:
            raise ValueError(f"gripper force for {side} must be in [0, 1]")
        normalized.append((side, force))
    if not normalized:
        raise ValueError("gripper force profile must contain at least one active side")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class GripperForceDatasetContract:
    """Immutable force profile shared by every episode in one dataset."""

    force_by_side: tuple[tuple[str, float], ...]
    profile_name: str = ""
    task: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "force_by_side",
            _normalized_force_by_side(dict(self.force_by_side)),
        )
        object.__setattr__(self, "profile_name", str(self.profile_name or ""))
        object.__setattr__(self, "task", str(self.task or ""))

    @property
    def force_by_side_dict(self) -> dict[str, float]:
        return {side: float(force) for side, force in self.force_by_side}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRIPPER_FORCE_PROFILE_SCHEMA_V1,
            "scope": "dataset",
            "profile_name": self.profile_name,
            "task": self.task,
            "force_by_side": self.force_by_side_dict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GripperForceDatasetContract":
        if payload.get("schema_version") != GRIPPER_FORCE_PROFILE_SCHEMA_V1:
            raise ValueError(
                "unsupported gripper force profile schema_version="
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("scope", "dataset") != "dataset":
            raise ValueError("gripper force profile scope must be 'dataset'")
        force_by_side = payload.get("force_by_side")
        if not isinstance(force_by_side, Mapping):
            raise ValueError("gripper force profile force_by_side must be an object")
        return cls(
            force_by_side=tuple(force_by_side.items()),
            profile_name=str(payload.get("profile_name") or ""),
            task=str(payload.get("task") or ""),
        )


__all__ = [
    "GRIPPER_FORCE_PROFILE_SCHEMA_V1",
    "GripperForceDatasetContract",
]
