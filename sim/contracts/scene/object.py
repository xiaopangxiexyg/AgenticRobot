from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contact import ContactMaterialSpec
from .validation import finite_tuple


def _box_inertia(
    mass: float,
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = size
    scale = mass / 12.0
    return (
        scale * (y * y + z * z),
        scale * (x * x + z * z),
        scale * (x * x + y * y),
    )


def _default_inertia(
    mass: float,
    shape: str,
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = size
    if shape == "box":
        return _box_inertia(mass, size)
    if shape == "sphere":
        radius = x / 2.0
        moment = 0.4 * mass * radius * radius
        return (moment, moment, moment)
    if shape == "cylinder":
        radius = x / 2.0
        axial = 0.5 * mass * radius * radius
        transverse = mass * (3.0 * radius * radius + z * z) / 12.0
        return (transverse, transverse, axial)
    if shape == "ellipsoid":
        a, b, c = x / 2.0, y / 2.0, z / 2.0
        return (
            mass * (b * b + c * c) / 5.0,
            mass * (a * a + c * c) / 5.0,
            mass * (a * a + b * b) / 5.0,
        )
    raise ValueError(f"Unsupported free-body shape {shape!r}.")


def _validate_shape_size(shape: str, size: tuple[float, float, float]) -> None:
    if shape not in {"box", "cylinder", "sphere", "ellipsoid"}:
        raise ValueError(
            "Free-body shape must be one of box, cylinder, sphere, or ellipsoid."
        )
    if any(value <= 0.0 for value in size):
        raise ValueError("Object dimensions must be finite and positive.")
    scale = max(size)
    tolerance = max(1e-9, scale * 1e-6)
    if shape == "sphere" and max(size) - min(size) > tolerance:
        raise ValueError("Sphere size_m must have equal x, y, and z extents.")
    if shape == "cylinder" and abs(size[0] - size[1]) > tolerance:
        raise ValueError("Cylinder size_m must have equal x and y diameters.")


@dataclass(frozen=True)
class FreeBodyObjectSpec:
    name: str
    shape: str = "box"
    size_m: tuple[float, float, float] = (0.03, 0.03, 0.03)
    initial_position_m: tuple[float, float, float] = (-0.472, 0.0, 0.026)
    initial_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass_kg: float = 0.05
    inertia_diagonal_kg_m2: tuple[float, float, float] | None = None
    rgba: tuple[float, float, float, float] = (0.85, 0.18, 0.12, 1.0)
    contact: ContactMaterialSpec = field(default_factory=ContactMaterialSpec)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("FreeBodyObjectSpec.name must be non-empty.")
        shape = str(self.shape)
        size = finite_tuple(self.size_m, 3, "object size_m")
        _validate_shape_size(shape, size)
        position = finite_tuple(self.initial_position_m, 3, "object initial_position_m")
        quat = finite_tuple(self.initial_quaternion_wxyz, 4, "object quaternion")
        mass = float(self.mass_kg)
        if mass <= 0.0 or not math.isfinite(mass):
            raise ValueError("Object mass and size must be finite and positive.")
        quat_norm = math.sqrt(sum(value * value for value in quat))
        if quat_norm <= 1e-12:
            raise ValueError("Object quaternion must have non-zero norm.")
        quat = tuple(value / quat_norm for value in quat)
        inertia = finite_tuple(
            _default_inertia(mass, shape, size)
            if self.inertia_diagonal_kg_m2 is None
            else self.inertia_diagonal_kg_m2,
            3,
            "object inertia",
        )
        if any(value <= 0.0 for value in inertia):
            raise ValueError("Object inertia values must be positive.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "initial_position_m", position)
        object.__setattr__(self, "initial_quaternion_wxyz", quat)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "inertia_diagonal_kg_m2", inertia)
        object.__setattr__(self, "rgba", finite_tuple(self.rgba, 4, "object rgba"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "size_m": list(self.size_m),
            "initial_position_m": list(self.initial_position_m),
            "initial_quaternion_wxyz": list(self.initial_quaternion_wxyz),
            "mass_kg": self.mass_kg,
            "inertia_diagonal_kg_m2": list(self.inertia_diagonal_kg_m2 or ()),
            "rgba": list(self.rgba),
            "contact": self.contact.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FreeBodyObjectSpec":
        inertia = payload.get("inertia_diagonal_kg_m2")
        return cls(
            name=str(payload["name"]),
            shape=str(payload.get("shape", "box")),
            size_m=tuple(payload.get("size_m", (0.03, 0.03, 0.03))),
            initial_position_m=tuple(
                payload.get("initial_position_m", (-0.472, 0.0, 0.026))
            ),
            initial_quaternion_wxyz=tuple(
                payload.get("initial_quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))
            ),
            mass_kg=float(payload.get("mass_kg", 0.05)),
            inertia_diagonal_kg_m2=(
                None if not inertia else tuple(float(value) for value in inertia)
            ),
            rgba=tuple(payload.get("rgba", (0.85, 0.18, 0.12, 1.0))),
            contact=ContactMaterialSpec.from_dict(payload.get("contact")),
        )


__all__ = ["FreeBodyObjectSpec"]
