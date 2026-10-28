from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from sim.contracts.scene import FreeBodyObjectSpec

from ..transforms import format_float_list
from .collision import CollisionFilter, OBJECT_GROUP
from .geometry import MeshAssetRegistry
from .world import contact_attributes


def _mujoco_geom_size(spec: FreeBodyObjectSpec) -> tuple[float, ...]:
    x, y, z = spec.size_m
    if spec.shape == "box":
        return (x / 2.0, y / 2.0, z / 2.0)
    if spec.shape == "cylinder":
        return (x / 2.0, z / 2.0)
    if spec.shape == "sphere":
        return (x / 2.0,)
    if spec.shape == "ellipsoid":
        return (x / 2.0, y / 2.0, z / 2.0)
    raise ValueError(f"Unsupported free-body shape {spec.shape!r}.")


def add_free_body_object(
    worldbody: ET.Element,
    spec: FreeBodyObjectSpec,
    collision_filter: CollisionFilter,
    *,
    visual_mesh_path: str | Path | None = None,
    geometry_registry: MeshAssetRegistry | None = None,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": spec.name,
            "pos": format_float_list(spec.initial_position_m),
            # MJCF quaternions use wxyz.
            "quat": format_float_list(spec.initial_quaternion_wxyz),
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{spec.name}_freejoint"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{spec.mass_kg:.12g}",
            "diaginertia": format_float_list(spec.inertia_diagonal_kg_m2 or ()),
        },
    )
    shape = {"type": spec.shape, "size": format_float_list(_mujoco_geom_size(spec))}
    if visual_mesh_path is None:
        visual_shape = shape
    else:
        if geometry_registry is None:
            raise ValueError("A geometry registry is required for object visual meshes.")
        visual_shape = {
            "type": "mesh",
            "mesh": geometry_registry.visual_mesh_name(visual_mesh_path),
        }
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{spec.name}_visual",
            **visual_shape,
            "rgba": format_float_list(spec.rgba),
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{spec.name}_collision",
            **shape,
            "rgba": "0 0 0 0",
            "density": "0",
            "group": "3",
            **collision_filter.geom_attributes(OBJECT_GROUP),
            **contact_attributes(spec.contact),
        },
    )


def add_free_body_objects(
    worldbody: ET.Element,
    specs: tuple[FreeBodyObjectSpec, ...],
    collision_filter: CollisionFilter,
    *,
    visual_mesh_path_by_name: Mapping[str, str | Path] | None = None,
    geometry_registry: MeshAssetRegistry | None = None,
) -> None:
    visual_paths = {} if visual_mesh_path_by_name is None else visual_mesh_path_by_name
    for spec in specs:
        add_free_body_object(
            worldbody, spec, collision_filter,
            visual_mesh_path=visual_paths.get(spec.name), geometry_registry=geometry_registry,
        )


__all__ = ["add_free_body_object", "add_free_body_objects"]
