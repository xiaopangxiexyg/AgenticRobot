from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from sim.contracts.scene import FreeBodyObjectSpec

from ..transforms import format_float_list
from .collision import CollisionFilter, OBJECT_GROUP
from .world import contact_attributes


def add_free_body_object(
    worldbody: ET.Element,
    spec: FreeBodyObjectSpec,
    collision_filter: CollisionFilter,
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
    half_size = np.asarray(spec.size_m, dtype=float) / 2.0
    shape = {"type": spec.shape, "size": format_float_list(half_size)}
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{spec.name}_visual",
            **shape,
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
) -> None:
    for spec in specs:
        add_free_body_object(worldbody, spec, collision_filter)


__all__ = ["add_free_body_object", "add_free_body_objects"]
