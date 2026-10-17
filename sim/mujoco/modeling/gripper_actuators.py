"""MJCF assembly for force-limited parallel-gripper position servos."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from ..actuation.gripper_profile import GripperActuationProfile


def _find_or_add(root: ET.Element, tag: str) -> ET.Element:
    element = root.find(tag)
    return element if element is not None else ET.SubElement(root, tag)


def add_gripper_position_actuators(
    mujoco_root: ET.Element,
    profile: GripperActuationProfile,
) -> None:
    """Add mimic-linked finger constraints and force-limited servos.

    The profile selects the master joint. This makes CAD URDF mimic
    semantics explicit in generated MJCF because the local URDF parser keeps
    joint limits and axes but intentionally does not import ``<mimic>`` tags.
    """

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
        for relation_index, (master_joint, mimic_joint) in enumerate(
            profile.equality_joint_pairs(gripper_name)
        ):
            ET.SubElement(
                equality,
                "joint",
                {
                    "name": profile.equality_name(gripper_name, relation_index),
                    "joint1": master_joint,
                    "joint2": mimic_joint,
                    "polycoef": "0 1 0 0 0",
                    "solref": " ".join(map(str, profile.contact_tuning.mimic_solref or (.005, 1.))),
                    **({"solimp": " ".join(map(str, profile.contact_tuning.mimic_solimp))}
                        if profile.contact_tuning.mimic_solimp is not None else {}),
                },
            )
        for index, joint_name in enumerate(joint_names):
            opened = profile.open_qpos_by_gripper[gripper_name][index]
            closed = profile.closed_qpos_by_gripper[gripper_name][index]
            operating_range = (min(opened, closed), max(opened, closed))
            hard_range = joints[joint_name].get("range")
            if hard_range is not None:
                limits = tuple(float(v) for v in hard_range.split())
                if (
                    len(limits) != 2 or not all(math.isfinite(v) for v in limits)
                    or limits[0] >= limits[1]
                    or operating_range[0] < limits[0] - 1e-9
                    or operating_range[1] > limits[1] + 1e-9
                ):
                    raise ValueError(f"{joint_name} operating range exceeds model joint limits.")
            # The compiled range also supplies normalized scene feedback and
            # reset/preview mapping. Keep it identical to the servo profile;
            # never leave feedback normalized to the old bare-hand stroke.
            joints[joint_name].attrib.update(
                {
                    "range": " ".join(f"{v:.17g}" for v in operating_range),
                    "limited": "true",
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
        if tuning.geom_pairs:
            continue
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

    tuning = profile.contact_tuning
    if tuning.solver_cone is not None or tuning.solver_impratio is not None:
        option = _find_or_add(mujoco_root, "option")
        if tuning.solver_cone is not None:
            option.set("cone", tuning.solver_cone)
        if tuning.solver_impratio is not None:
            option.set("impratio", str(tuning.solver_impratio))
    for first, second in tuning.geom_pairs:
        if first not in geoms or second not in geoms:
            raise KeyError(f"Contact profile references unknown geom pair {(first, second)!r}.")
        if any(geoms[name].get("contype") == "0" and geoms[name].get("conaffinity") == "0"
               for name in (first, second)):
            raise ValueError("Explicit contact pairs cannot select visual-only geoms.")
        attributes = {"geom1": first, "geom2": second}
        for name in ("solref", "solimp"):
            if getattr(tuning, name) is not None:
                attributes[name] = " ".join(map(str, getattr(tuning, name)))
        if tuning.friction is not None:
            slide, spin, roll = tuning.friction
            attributes["friction"] = " ".join(map(str, (slide, slide, spin, roll, roll)))
        attributes["condim"] = str(max(int(geoms[name].get("condim", "3")) for name in (first, second)))
        ET.SubElement(_find_or_add(mujoco_root, "contact"), "pair", attributes)


__all__ = ["add_gripper_position_actuators"]
