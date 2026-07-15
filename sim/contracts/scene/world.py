from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contact import ContactMaterialSpec
from .validation import finite_tuple


@dataclass(frozen=True)
class GroundSpec:
    enabled: bool = True
    name: str = "ground"
    z_m: float = -0.05
    half_size_xy_m: tuple[float, float] = (2.0, 2.0)
    rgba: tuple[float, float, float, float] = (0.12, 0.12, 0.12, 1.0)
    contact: ContactMaterialSpec = field(default_factory=ContactMaterialSpec)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("GroundSpec.name must be non-empty.")
        size = finite_tuple(self.half_size_xy_m, 2, "half_size_xy_m")
        rgba = finite_tuple(self.rgba, 4, "ground rgba")
        if any(value <= 0.0 for value in size):
            raise ValueError("Ground half-size must be positive.")
        if not math.isfinite(float(self.z_m)):
            raise ValueError("Ground z_m must be finite.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "half_size_xy_m", size)
        object.__setattr__(self, "rgba", rgba)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "name": self.name,
            "z_m": float(self.z_m),
            "half_size_xy_m": list(self.half_size_xy_m),
            "rgba": list(self.rgba),
            "contact": self.contact.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "GroundSpec":
        data = dict(payload or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            name=str(data.get("name", "ground")),
            z_m=float(data.get("z_m", -0.05)),
            half_size_xy_m=tuple(data.get("half_size_xy_m", (2.0, 2.0))),
            rgba=tuple(data.get("rgba", (0.12, 0.12, 0.12, 1.0))),
            contact=ContactMaterialSpec.from_dict(data.get("contact")),
        )


__all__ = ["GroundSpec"]
