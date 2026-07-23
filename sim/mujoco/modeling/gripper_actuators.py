"""MJCF assembly for force-limited parallel-gripper position servos."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..actuation.gripper_profile import GripperActuationProfile


def _find_or_add(root: ET.Element, tag: str) -> ET.Element:
    element = root.find(tag)
    return element if element is not None else ET.SubElement(root, tag)


def add_gripper_position_actuators(
    mujoco_root: ET.Element,
    profile: GripperActuationProfile,
) -> None:
    """Add symmetric finger constraints and independent force-limited servos."""

    joints = {
        str(element.attrib["name"]): element
        for element in mujoco_root.iter("joint")
        if element.attrib.get("name")
    }
    geoms = {
        str(element.attrib["name"]): element
        for element in mujoco_root.iter("geom")
        if element.attrib.get("name")
    }
    equality = _find_or_add(mujoco_root, "equality")
    actuator = _find_or_add(mujoco_root, "actuator")
    for gripper_name, joint_names in profile.joint_names_by_gripper.items():
        missing = [name for name in joint_names if name not in joints]
        if missing:
            raise KeyError(
                f"Gripper profile {gripper_name!r} references unknown joints {missing}."
            )
        ET.SubElement(
            equality,
            "joint",
            {
                "name": profile.equality_name(gripper_name),
                "joint1": joint_names[0],
                "joint2": joint_names[1],
                "polycoef": "0 1 0 0 0",
                "solref": "0.005 1",
            },
        )
        for index, joint_name in enumerate(joint_names):
            joints[joint_name].attrib.update(
                {
                    "damping": f"{profile.joint_damping_n_s_m:.12g}",
                    "frictionloss": f"{profile.joint_frictionloss_n:.12g}",
                }
            )
            limit = profile.force_limit_n
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": profile.actuator_name(gripper_name, index),
                    "joint": joint_name,
                    "kp": f"{profile.position_kp_n_m:.12g}",
                    "kv": f"{profile.position_kv_n_s_m:.12g}",
                    "forcelimited": "true",
                    "forcerange": f"{-limit:.12g} {limit:.12g}",
                },
            )

        tuning = profile.contact_tuning
        for side_name in ("left", "right"):
            geom_name = f"{gripper_name}_{side_name}_finger_collision_0"
            geom = geoms.get(geom_name)
            if geom is None:
                continue
            if tuning.friction is not None:
                geom.attrib["friction"] = " ".join(str(value) for value in tuning.friction)
            if tuning.solref is not None:
                geom.attrib["solref"] = " ".join(str(value) for value in tuning.solref)
            if tuning.solimp is not None:
                geom.attrib["solimp"] = " ".join(str(value) for value in tuning.solimp)


__all__ = ["add_gripper_position_actuators"]
