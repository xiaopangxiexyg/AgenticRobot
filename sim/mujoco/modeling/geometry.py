from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..transforms import format_float_list
from ..urdf import URDFGeometry, URDFModel
from .obj_materials import ObjMaterialMeshPart, obj_material_mesh_parts


def safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text))
    if not name:
        return "asset"
    return f"asset_{name}" if name[0].isdigit() else name


@dataclass(frozen=True)
class VisualGeomPart:
    """One MJCF visual geom representation of a URDF visual geometry."""

    attributes: Mapping[str, str]
    rgba: tuple[float, float, float, float] | None = None
    material_name: str | None = None


class MeshAssetRegistry:
    """Deduplicate mesh assets while visual and collision geoms share conversion."""

    def __init__(self, asset: ET.Element, model: URDFModel):
        self.asset = asset
        self.model = model
        self._by_path: dict[tuple[Path, str | None], str] = {}
        self._used_names: set[str] = set()
        # A registry belongs to one MJCF conversion. Repeated left/right
        # visuals share resolution here; new conversions reread their inputs.
        self._visual_parts_by_path: dict[Path, tuple[ObjMaterialMeshPart, ...]] = {}

    def _mesh_name(self, path: Path, *, inertia: str | None = None) -> str:
        key = (path, inertia)
        existing = self._by_path.get(key)
        if existing is not None:
            return existing
        stem = safe_name(path.stem)
        name = stem
        suffix = 1
        while name in self._used_names:
            suffix += 1
            name = f"{stem}_{suffix}"
        self._used_names.add(name)
        self._by_path[key] = name
        attributes = {"name": name, "file": str(path)}
        if inertia is not None:
            attributes["inertia"] = inertia
        ET.SubElement(self.asset, "mesh", attributes)
        return name

    def geom_attributes(self, geometry: URDFGeometry) -> dict[str, str]:
        if geometry.kind == "mesh":
            if geometry.filename is None:
                raise ValueError("URDF mesh geometry is missing filename.")
            mesh_path = self.model.resolve_mesh_path(geometry.filename)
            if not mesh_path.exists():
                raise FileNotFoundError(f"URDF mesh not found: {mesh_path}")
            return {"type": "mesh", "mesh": self._mesh_name(mesh_path)}
        if geometry.kind == "box":
            half_size = np.asarray(geometry.values, dtype=float).reshape(3) / 2.0
            return {"type": "box", "size": format_float_list(half_size)}
        if geometry.kind == "cylinder":
            radius, length = geometry.values
            return {
                "type": "cylinder",
                "size": format_float_list((radius, length / 2.0)),
            }
        if geometry.kind == "sphere":
            (radius,) = geometry.values
            return {"type": "sphere", "size": f"{float(radius):.12g}"}
        raise ValueError(f"Unsupported URDF geometry kind: {geometry.kind}")

    def visual_geom_parts(self, geometry: URDFGeometry) -> tuple[VisualGeomPart, ...]:
        """Expand a visual OBJ into one part per MTL material when possible.

        Collision callers continue to use :meth:`geom_attributes`, so OBJ/MTL
        handling cannot change collision geometry or contact behavior.
        """

        if geometry.kind != "mesh" or geometry.filename is None:
            return (VisualGeomPart(attributes=self.geom_attributes(geometry)),)
        mesh_path = self.model.resolve_mesh_path(geometry.filename)
        if not mesh_path.exists():
            raise FileNotFoundError(f"URDF mesh not found: {mesh_path}")
        if mesh_path not in self._visual_parts_by_path:
            self._visual_parts_by_path[mesh_path] = obj_material_mesh_parts(mesh_path)
        parts = self._visual_parts_by_path[mesh_path]
        if not parts:
            return (VisualGeomPart(attributes=self.geom_attributes(geometry)),)
        return tuple(
            VisualGeomPart(
                attributes={
                    "type": "mesh",
                    # A material partition is generally an open surface even
                    # when the source CAD mesh is watertight.  It is visual
                    # only, so shell inertia is the valid compiler setting.
                    "mesh": self._mesh_name(part.mesh_path, inertia="shell"),
                },
                rgba=part.rgba,
                material_name=part.material_name,
            )
            for part in parts
        )


__all__ = ["MeshAssetRegistry", "VisualGeomPart", "safe_name"]
