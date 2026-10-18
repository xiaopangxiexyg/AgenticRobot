from __future__ import annotations

import math
from dataclasses import dataclass

FORBIDDEN_COLLISION_CATEGORIES = (
    "robot_environment",
    "gripper_environment",
    "robot_object",
    "robot_self",
    "robot_gripper_self",
    "gripper_self",
)

_CATEGORY_BY_PAIR_CLASS = {
    "1:2": "robot_environment",
    "1:4": "gripper_environment",
    "2:8": "robot_object",
    "2:2": "robot_self",
    "2:4": "robot_gripper_self",
    "4:4": "gripper_self",
}


def forbidden_collision_category(pair_class: str | None) -> str | None:
    """Return a semantic category for one forbidden collision-group pair."""

    if pair_class is None:
        return None
    return _CATEGORY_BY_PAIR_CLASS.get(str(pair_class))


@dataclass(frozen=True)
class CollisionCheckResult:
    safe: bool
    minimum_distance_m: float
    geom1_id: int | None = None
    geom2_id: int | None = None
    geom1_name: str | None = None
    geom2_name: str | None = None
    body1_name: str | None = None
    body2_name: str | None = None
    pair_class: str | None = None
    minimum_distance_is_lower_bound: bool = False
    distance_query_disagreement_count: int = 0
    distance_query_raw_distance_m: float | None = None
    required_clearance_m: float | None = None

    @property
    def penetration_depth_m(self) -> float:
        return max(0.0, -float(self.minimum_distance_m))

    @property
    def collision_category(self) -> str | None:
        return forbidden_collision_category(self.pair_class)

    def to_dict(self) -> dict[str, object]:
        distance = float(self.minimum_distance_m)
        values = {
            "safe": bool(self.safe),
            "minimum_distance_m": distance if math.isfinite(distance) else None,
            "penetration_depth_m": self.penetration_depth_m,
            "geom1": self.geom1_name,
            "geom2": self.geom2_name,
            "body1": self.body1_name,
            "body2": self.body2_name,
            "pair_class": self.pair_class,
            "collision_category": self.collision_category,
        }
        if self.minimum_distance_is_lower_bound:
            values["minimum_distance_is_lower_bound"] = True
        if self.distance_query_disagreement_count:
            values["distance_query_disagreement_count"] = self.distance_query_disagreement_count
        if self.distance_query_raw_distance_m is not None:
            values["distance_query_raw_distance_m"] = self.distance_query_raw_distance_m
        if self.required_clearance_m is not None:
            values["required_clearance_m"] = self.required_clearance_m
        return values
