from __future__ import annotations

from dataclasses import dataclass
import math


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

    @property
    def penetration_depth_m(self) -> float:
        return max(0.0, -float(self.minimum_distance_m))

    def to_dict(self) -> dict[str, object]:
        distance = float(self.minimum_distance_m)
        return {
            "safe": bool(self.safe),
            "minimum_distance_m": distance if math.isfinite(distance) else None,
            "penetration_depth_m": self.penetration_depth_m,
            "geom1": self.geom1_name,
            "geom2": self.geom2_name,
            "body1": self.body1_name,
            "body2": self.body2_name,
            "pair_class": self.pair_class,
        }
