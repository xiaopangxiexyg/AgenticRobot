from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionSafetyPolicy:
    """Hard safety policy for robot command targets and active contacts.

    Object/table and gripper/object contact remain physical task contacts.  A
    commanded robot is not allowed to enter fixed environment geometry, hit
    task objects with non-gripper links, or self-collide.
    """

    minimum_clearance_m: float = 0.002
    hard_penetration_m: float = 0.001
    sweep_max_joint_step_rad: float = 0.01
    projection_iterations: int = 8
    # Some gripper CAD meshes overlap at their normalized closed endpoint.
    # This opt-in exception is limited to finger-vs-finger self contact; it
    # never permits gripper/environment, robot/environment, or robot/gripper
    # self contact. The strict default preserves existing task safety.
    allow_gripper_self_contact: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.minimum_clearance_m)) or self.minimum_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be non-negative.")
        if not math.isfinite(float(self.hard_penetration_m)) or self.hard_penetration_m < 0.0:
            raise ValueError("hard_penetration_m must be non-negative.")
        if not math.isfinite(float(self.sweep_max_joint_step_rad)) or self.sweep_max_joint_step_rad <= 0.0:
            raise ValueError("sweep_max_joint_step_rad must be positive.")
        if self.projection_iterations <= 0:
            raise ValueError("projection_iterations must be positive.")
        if not isinstance(self.allow_gripper_self_contact, bool):
            raise TypeError("allow_gripper_self_contact must be a bool.")

    def pair_is_forbidden(self, first_group: int, second_group: int) -> bool:
        pair = frozenset((int(first_group), int(second_group)))
        # CollisionFilter group bits: environment=1, robot=2, gripper=4,
        # object=8. Equal-group pairs collapse to a one-element frozenset.
        if pair == frozenset((4,)) and self.allow_gripper_self_contact:
            return False
        return pair in {
            frozenset((1, 2)),
            frozenset((1, 4)),
            frozenset((2, 8)),
            frozenset((2,)),
            frozenset((2, 4)),
            frozenset((4,)),
        }

    @property
    def distance_query_limit_m(self) -> float:
        return max(0.05, self.minimum_clearance_m * 2.0)


@dataclass(frozen=True)
class NonContactObjectMovePolicy(CollisionSafetyPolicy):
    """Existing noncontact TCP query policy, shared with attached geometry."""

    @property
    def distance_query_limit_m(self) -> float:
        # Broad-radius GJK diagnostics may return zero for separated CAD
        # meshes. The clipped query retains the required prevention margin.
        return float(self.minimum_clearance_m)

    def pair_is_forbidden(self, first_group: int, second_group: int) -> bool:
        return super().pair_is_forbidden(first_group, second_group) or frozenset(
            (int(first_group), int(second_group))
        ) == frozenset((4, 8))
