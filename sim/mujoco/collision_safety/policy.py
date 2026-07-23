from __future__ import annotations

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

    def __post_init__(self) -> None:
        if self.minimum_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be non-negative.")
        if self.hard_penetration_m < 0.0:
            raise ValueError("hard_penetration_m must be non-negative.")
        if self.sweep_max_joint_step_rad <= 0.0:
            raise ValueError("sweep_max_joint_step_rad must be positive.")
        if self.projection_iterations <= 0:
            raise ValueError("projection_iterations must be positive.")

    def pair_is_forbidden(self, first_group: int, second_group: int) -> bool:
        pair = frozenset((int(first_group), int(second_group)))
        # CollisionFilter group bits: environment=1, robot=2, gripper=4,
        # object=8. Equal-group pairs collapse to a one-element frozenset.
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
