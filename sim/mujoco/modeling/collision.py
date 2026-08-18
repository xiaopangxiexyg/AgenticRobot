from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from sim.contracts.scene import CollisionSpec
from sim.contracts.versions import MUJOCO_COLLISION_PROFILE_VERSION

from ..urdf import URDFModel


ENVIRONMENT_GROUP = "environment"
ROBOT_GROUP = "robot"
GRIPPER_GROUP = "gripper"
OBJECT_GROUP = "object"


@dataclass(frozen=True)
class ContactGroup:
    name: str
    bit: int


class CollisionFilter:
    """Semantic contact groups compiled to MuJoCo contype/conaffinity masks."""

    profile_version = MUJOCO_COLLISION_PROFILE_VERSION
    groups = (
        ContactGroup(ENVIRONMENT_GROUP, 1 << 0),
        ContactGroup(ROBOT_GROUP, 1 << 1),
        ContactGroup(GRIPPER_GROUP, 1 << 2),
        ContactGroup(OBJECT_GROUP, 1 << 3),
    )
    # Environment-environment is intentionally absent. All manipulation-relevant
    # pairs are symmetric and enabled, including cross-arm/self collision.
    allowed_group_pairs = frozenset(
        {
            (ENVIRONMENT_GROUP, ROBOT_GROUP),
            (ENVIRONMENT_GROUP, GRIPPER_GROUP),
            (ENVIRONMENT_GROUP, OBJECT_GROUP),
            (ROBOT_GROUP, ROBOT_GROUP),
            (ROBOT_GROUP, GRIPPER_GROUP),
            (ROBOT_GROUP, OBJECT_GROUP),
            (GRIPPER_GROUP, GRIPPER_GROUP),
            (GRIPPER_GROUP, OBJECT_GROUP),
            (OBJECT_GROUP, OBJECT_GROUP),
        }
    )

    def __init__(self) -> None:
        self._bits = {group.name: int(group.bit) for group in self.groups}
        if len(set(self._bits.values())) != len(self._bits):
            raise ValueError("Collision contact group bits must be unique.")

    @staticmethod
    def _canonical_pair(first: str, second: str) -> tuple[str, str]:
        order = {
            ENVIRONMENT_GROUP: 0,
            ROBOT_GROUP: 1,
            GRIPPER_GROUP: 2,
            OBJECT_GROUP: 3,
        }
        return (first, second) if order[first] <= order[second] else (second, first)

    def allows(self, first: str, second: str) -> bool:
        return self._canonical_pair(first, second) in self.allowed_group_pairs

    def masks_for_group(self, group_name: str) -> tuple[int, int]:
        name = str(group_name)
        if name not in self._bits:
            raise KeyError(f"Unknown collision group {name!r}.")
        affinity = sum(
            bit for other, bit in self._bits.items() if self.allows(name, other)
        )
        return self._bits[name], int(affinity)

    def geom_attributes(self, group_name: str) -> dict[str, str]:
        contype, conaffinity = self.masks_for_group(group_name)
        return {
            "contype": str(contype),
            "conaffinity": str(conaffinity),
        }

    def group_for_urdf_link(self, link_name: str) -> str:
        name = str(link_name).lower()
        if name == "table" or "camera_pole" in name or "head_camera" in name:
            return ENVIRONMENT_GROUP
        # The official EG2-4B2 CAD export uses l_*/r_* link names rather than
        # the proxy's explicit "finger" names.  Keep those links in the
        # gripper contact group so the CAD model preserves the same collision
        # semantics as the older parallel-gripper scene.
        if (
            "gripper" in name
            or "finger" in name
            or "contact_frame" in name
            or "eg2_4b2_cad" in name
        ):
            return GRIPPER_GROUP
        return ROBOT_GROUP

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_version": self.profile_version,
            "groups": {name: bit for name, bit in self._bits.items()},
            "allowed_group_pairs": [list(pair) for pair in sorted(self.allowed_group_pairs)],
        }


def _normalized_body_pair(first: str, second: str) -> tuple[str, str] | None:
    raw_names = (str(first), str(second))
    if "world" in raw_names:
        return None
    names = sorted(raw_names)
    if not names[0] or names[0] == names[1]:
        return None
    return names[0], names[1]


def excluded_body_pairs(
    model: URDFModel,
    spec: CollisionSpec,
) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    if spec.exclude_parent_child:
        for joint in model.joints:
            pair = _normalized_body_pair(joint.parent, joint.child)
            if pair is not None:
                pairs.add(pair)
    for first, second in spec.excluded_body_pairs:
        pair = _normalized_body_pair(first, second)
        if pair is not None:
            pairs.add(pair)
    if not spec.self_collision_enabled:
        collision_filter = CollisionFilter()
        for side in ("left", "right"):
            side_bodies = [
                name
                for name in model.links
                if name.startswith(f"{side}_")
                and collision_filter.group_for_urdf_link(name) in {ROBOT_GROUP, GRIPPER_GROUP}
            ]
            for first, second in combinations(side_bodies, 2):
                pair = _normalized_body_pair(first, second)
                if pair is not None:
                    pairs.add(pair)
    return tuple(sorted(pairs))


def validate_excluded_bodies(
    model: URDFModel,
    pairs: Iterable[tuple[str, str]],
) -> None:
    known = set(model.links) - {"world"}
    missing = sorted({name for pair in pairs for name in pair if name not in known})
    if missing:
        raise ValueError(f"Collision exclusions reference missing URDF bodies: {missing}")


__all__ = [
    "CollisionFilter",
    "ENVIRONMENT_GROUP",
    "GRIPPER_GROUP",
    "OBJECT_GROUP",
    "ROBOT_GROUP",
    "excluded_body_pairs",
    "validate_excluded_bodies",
]
