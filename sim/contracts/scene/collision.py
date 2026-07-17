from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..versions import MUJOCO_COLLISION_PROFILE_VERSION, require_version
from .contact import ContactMaterialSpec


@dataclass(frozen=True)
class CollisionSpec:
    enabled: bool = True
    profile_version: str = MUJOCO_COLLISION_PROFILE_VERSION
    self_collision_enabled: bool = True
    exclude_parent_child: bool = True
    excluded_body_pairs: tuple[tuple[str, str], ...] = ()
    safety_minimum_clearance_m: float = 0.002
    safety_hard_penetration_m: float = 0.001
    safety_sweep_max_joint_step_rad: float = 0.01
    safety_projection_iterations: int = 8
    contact: ContactMaterialSpec = field(default_factory=ContactMaterialSpec)

    def __post_init__(self) -> None:
        require_version(
            self.profile_version,
            MUJOCO_COLLISION_PROFILE_VERSION,
            field="MuJoCo collision profile",
        )
        pairs: list[tuple[str, str]] = []
        for pair in self.excluded_body_pairs:
            if len(pair) != 2:
                raise ValueError("Each excluded body pair must contain exactly two names.")
            first, second = sorted((str(pair[0]), str(pair[1])))
            if not first or first == second:
                raise ValueError("Excluded body pair names must be non-empty and distinct.")
            if (first, second) not in pairs:
                pairs.append((first, second))
        object.__setattr__(self, "excluded_body_pairs", tuple(pairs))
        if float(self.safety_minimum_clearance_m) < 0.0:
            raise ValueError("safety_minimum_clearance_m must be non-negative.")
        if float(self.safety_hard_penetration_m) < 0.0:
            raise ValueError("safety_hard_penetration_m must be non-negative.")
        if float(self.safety_sweep_max_joint_step_rad) <= 0.0:
            raise ValueError("safety_sweep_max_joint_step_rad must be positive.")
        if int(self.safety_projection_iterations) <= 0:
            raise ValueError("safety_projection_iterations must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile_version": self.profile_version,
            "self_collision_enabled": bool(self.self_collision_enabled),
            "exclude_parent_child": bool(self.exclude_parent_child),
            "excluded_body_pairs": [list(pair) for pair in self.excluded_body_pairs],
            "safety_minimum_clearance_m": float(self.safety_minimum_clearance_m),
            "safety_hard_penetration_m": float(self.safety_hard_penetration_m),
            "safety_sweep_max_joint_step_rad": float(
                self.safety_sweep_max_joint_step_rad
            ),
            "safety_projection_iterations": int(self.safety_projection_iterations),
            "contact": self.contact.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CollisionSpec":
        data = dict(payload or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            profile_version=str(
                data.get("profile_version", MUJOCO_COLLISION_PROFILE_VERSION)
            ),
            self_collision_enabled=bool(data.get("self_collision_enabled", True)),
            exclude_parent_child=bool(data.get("exclude_parent_child", True)),
            excluded_body_pairs=tuple(
                tuple(str(value) for value in pair)
                for pair in data.get("excluded_body_pairs", ())
            ),
            safety_minimum_clearance_m=float(
                data.get("safety_minimum_clearance_m", 0.002)
            ),
            safety_hard_penetration_m=float(
                data.get("safety_hard_penetration_m", 0.001)
            ),
            safety_sweep_max_joint_step_rad=float(
                data.get("safety_sweep_max_joint_step_rad", 0.01)
            ),
            safety_projection_iterations=int(
                data.get("safety_projection_iterations", 8)
            ),
            contact=ContactMaterialSpec.from_dict(data.get("contact")),
        )


__all__ = ["CollisionSpec"]
