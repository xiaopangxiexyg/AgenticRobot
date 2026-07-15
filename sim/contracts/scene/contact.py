from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .validation import finite_tuple


@dataclass(frozen=True)
class ContactMaterialSpec:
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    condim: int = 4
    restitution_target: float = 0.0

    def __post_init__(self) -> None:
        friction = finite_tuple(self.friction, 3, "friction")
        solref = finite_tuple(self.solref, 2, "solref")
        solimp = finite_tuple(self.solimp, 3, "solimp")
        if any(value < 0.0 for value in friction):
            raise ValueError("Contact friction values must be non-negative.")
        if solref[0] <= 0.0 or solref[1] <= 0.0:
            raise ValueError("Contact solref values must be positive.")
        restitution = float(self.restitution_target)
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("restitution_target must be in [0, 1].")
        if int(self.condim) not in {1, 3, 4, 6}:
            raise ValueError("condim must be one of 1, 3, 4, or 6.")
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "solref", solref)
        object.__setattr__(self, "solimp", solimp)
        object.__setattr__(self, "condim", int(self.condim))
        object.__setattr__(self, "restitution_target", restitution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "friction": list(self.friction),
            "solref": list(self.solref),
            "solimp": list(self.solimp),
            "condim": self.condim,
            # Calibration target only; MuJoCo has no direct restitution geom field.
            "restitution_target": self.restitution_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ContactMaterialSpec":
        data = dict(payload or {})
        return cls(
            friction=tuple(data.get("friction", (1.0, 0.005, 0.0001))),
            solref=tuple(data.get("solref", (0.02, 1.0))),
            solimp=tuple(data.get("solimp", (0.9, 0.95, 0.001))),
            condim=int(data.get("condim", 4)),
            restitution_target=float(data.get("restitution_target", 0.0)),
        )


__all__ = ["ContactMaterialSpec"]
