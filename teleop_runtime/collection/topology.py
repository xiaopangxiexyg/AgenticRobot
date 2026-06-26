from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_SIDES: tuple[str, ...] = ("left", "right")


def parse_active_sides(value: object) -> tuple[str, ...]:
    text = str(value).strip().lower()
    if text in {"both", "dual", "bimanual"}:
        return DEFAULT_SIDES
    if text in DEFAULT_SIDES:
        return (text,)
    raise ValueError("active-sides must be one of: left, right, both")


def normalize_active_sides(sides: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw_side in sides:
        side = str(raw_side).strip().lower()
        if side not in DEFAULT_SIDES:
            raise ValueError(f"Unsupported side '{raw_side}'. Expected one of {DEFAULT_SIDES}.")
        if side not in seen:
            seen.append(side)
    return tuple(side for side in DEFAULT_SIDES if side in set(seen))


@dataclass(frozen=True)
class SideTopology:
    side: str
    arm_effector: str
    hand_effector: str
    wrist_pose_channel: str
    hand_q6_channel: str
    clutch_button_channel: str
    wrist_tracker_role: str


def default_side_topology(side: str) -> SideTopology:
    normalized = normalize_active_sides((side,))
    if not normalized:
        raise ValueError("side is required.")
    side_name = normalized[0]
    return SideTopology(
        side=side_name,
        arm_effector=f"{side_name}_arm",
        hand_effector=f"{side_name}_hand",
        wrist_pose_channel=f"{side_name}_wrist_pose",
        hand_q6_channel=f"{side_name}_hand_q6",
        clutch_button_channel=f"{side_name}_clutch_button",
        wrist_tracker_role=f"{side_name}_wrist",
    )


@dataclass(frozen=True)
class CollectionTopology:
    sides: tuple[SideTopology, ...]

    @classmethod
    def for_active_sides(cls, active_sides: Iterable[str]) -> "CollectionTopology":
        return cls(tuple(default_side_topology(side) for side in normalize_active_sides(active_sides)))

    @property
    def active_sides(self) -> tuple[str, ...]:
        return tuple(spec.side for spec in self.sides)

    def require_side(self, side: str) -> SideTopology:
        side_name = str(side).strip().lower()
        for spec in self.sides:
            if spec.side == side_name:
                return spec
        raise KeyError(f"Side '{side}' is not active in topology {self.active_sides}.")

    def arm_effector(self, side: str) -> str:
        return self.require_side(side).arm_effector

    def hand_effector(self, side: str) -> str:
        return self.require_side(side).hand_effector

    def wrist_pose_channel(self, side: str) -> str:
        return self.require_side(side).wrist_pose_channel

    def hand_q6_channel(self, side: str) -> str:
        return self.require_side(side).hand_q6_channel

    def clutch_button_channel(self, side: str) -> str:
        return self.require_side(side).clutch_button_channel

    def wrist_tracker_role(self, side: str) -> str:
        return self.require_side(side).wrist_tracker_role

    def arm_effectors(self) -> tuple[str, ...]:
        return tuple(spec.arm_effector for spec in self.sides)

    def hand_effectors(self) -> tuple[str, ...]:
        return tuple(spec.hand_effector for spec in self.sides)
