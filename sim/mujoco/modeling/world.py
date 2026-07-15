from __future__ import annotations

import xml.etree.ElementTree as ET

from sim.contracts.scene import ContactMaterialSpec, GroundSpec

from ..transforms import format_float_list
from .collision import CollisionFilter, ENVIRONMENT_GROUP


def contact_attributes(contact: ContactMaterialSpec) -> dict[str, str]:
    return {
        "friction": format_float_list(contact.friction),
        "solref": format_float_list(contact.solref),
        "solimp": format_float_list(contact.solimp),
        "condim": str(contact.condim),
    }


def add_ground(
    worldbody: ET.Element,
    spec: GroundSpec,
    collision_filter: CollisionFilter,
) -> None:
    if not spec.enabled:
        return
    common = {
        "type": "plane",
        "pos": format_float_list((0.0, 0.0, spec.z_m)),
        "size": format_float_list((*spec.half_size_xy_m, 0.1)),
    }
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": f"{spec.name}_visual",
            **common,
            "rgba": format_float_list(spec.rgba),
            "contype": "0",
            "conaffinity": "0",
            "group": "0",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": f"{spec.name}_collision",
            **common,
            "rgba": "0 0 0 0",
            "group": "3",
            **collision_filter.geom_attributes(ENVIRONMENT_GROUP),
            **contact_attributes(spec.contact),
        },
    )


__all__ = ["add_ground", "contact_attributes"]
