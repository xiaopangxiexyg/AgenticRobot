"""MJCF assembly helpers for physical RM65 position actuators."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from sim.rm65.actuation import PositionActuationProfile


def add_position_actuators(
    mujoco_root: ET.Element,
    profile: PositionActuationProfile,
) -> None:
    """Add joint damping and force-limited MuJoCo position servos.

    The actuator controls are qpos targets in radians.  This helper adds no
    task policy or safety logic and is intentionally independent from runtime
    execution.
    """

    joint_elements = {
        str(element.attrib.get("name")): element
        for element in mujoco_root.iter("joint")
        if element.attrib.get("name")
    }
    actuator = ET.SubElement(mujoco_root, "actuator")
    for side, joint_names in profile.joint_names_by_side.items():
        for index, joint_name in enumerate(joint_names):
            try:
                joint = joint_elements[joint_name]
            except KeyError as exc:
                raise KeyError(
                    f"Actuation profile references unknown MJCF joint {joint_name!r}."
                ) from exc
            joint.attrib.update(
                {
                    "armature": f"{profile.joint_armature_kg_m2[index]:.12g}",
                    "damping": f"{profile.joint_damping_nm_s_rad[index]:.12g}",
                    "frictionloss": f"{profile.joint_frictionloss_nm[index]:.12g}",
                }
            )
            limit = profile.force_limit_nm[index]
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": profile.actuator_name(side, index),
                    "joint": joint_name,
                    "kp": f"{profile.position_kp_nm_rad[index]:.12g}",
                    "kv": f"{profile.position_kv_nm_s_rad[index]:.12g}",
                    "forcelimited": "true",
                    "forcerange": f"{-limit:.12g} {limit:.12g}",
                },
            )


__all__ = ["add_position_actuators"]
