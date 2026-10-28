"""MJCF assembly for force-limited parallel-gripper position servos."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from ..actuation.gripper_profile import GripperActuationProfile, _object_contact_pair_name
from .collision import CollisionFilter, GRIPPER_GROUP, OBJECT_GROUP


def _find_or_add(root: ET.Element, tag: str) -> ET.Element:
    element = root.find(tag)
    return element if element is not None else ET.SubElement(root, tag)


def _validate_object_contact_bindings(mujoco_root, profile, joints, geoms) -> None:
    """Bind adapter-generated pairs to native free objects and their own pads."""
    names = profile.contact_tuning.object_contact_geom_names
    if not names:
        return
    worldbody = mujoco_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Object contact binding requires a worldbody.")
    parents = {child: parent for parent in mujoco_root.iter() for child in parent}
    filters = CollisionFilter()
    object_bit = str(filters.masks_for_group(OBJECT_GROUP)[0])
    gripper_bit = str(filters.masks_for_group(GRIPPER_GROUP)[0])
    for name in names:
        geom = geoms.get(name)
        if geom is None:
            raise KeyError(f"Contact profile references unknown geom pair containing {name!r}.")
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            raise ValueError("Explicit contact pairs cannot select visual-only geoms.")
        body = parents.get(geom)
        if (geom.get("contype") != object_bit or body is None or body.tag != "body"
                or parents.get(body) is not worldbody
                or (body.find("freejoint") is None
                    and not any(joint.get("type") == "free" for joint in body.findall("joint")))):
            raise ValueError(f"Object contact target {name!r} must be a free-body object collision geom.")
    pairs = {tuple(sorted(pair)) for pair in profile.contact_tuning.geom_pairs}
    for gripper, side in profile.side_by_gripper.items():
        master_name = profile.master_joint_by_gripper[gripper]
        master = joints.get(master_name)
        if master is None:
            raise KeyError(f"Gripper profile references unknown master joint {master_name!r}.")
        master_body = parents.get(master)
        palm = parents.get(master_body) if master_body is not None else None
        if palm is None or palm.tag != "body":
            raise ValueError(f"Gripper {gripper!r} has no owning palm body.")
        for finger in ("l", "r"):
            sensor_name = f"{side}_gripper_{finger}_dm_tac_w2m_sensor_collision_0"
            sensor = geoms.get(sensor_name)
            if sensor is None:
                raise KeyError(f"Contact profile references unknown geom pair containing {sensor_name!r}.")
            sensor_body = parents.get(sensor)
            expected_body = f"{side}_gripper_{finger}_dm_tac_w2m_sensor"
            if sensor_body is None or sensor_body.tag != "body" or sensor_body.get("name") != expected_body:
                raise ValueError(f"Contact sensor {sensor_name!r} must belong to the configured gripper sensor body.")
            while sensor_body is not None and sensor_body is not palm:
                sensor_body = parents.get(sensor_body)
            if sensor.get("contype") != gripper_bit or sensor_body is not palm:
                raise ValueError(f"Contact sensor {sensor_name!r} must belong to the configured gripper.")
            if any(tuple(sorted((name, sensor_name))) not in pairs for name in names):
                raise ValueError("Object contact binding requires every configured pad pair.")


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
    _validate_object_contact_bindings(mujoco_root, profile, joints, geoms)
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
        sensors = {f"{side}_gripper_{finger}_dm_tac_w2m_sensor_collision_0"
                   for side in ("left", "right") for finger in ("l", "r")}
        if (first in tuning.object_contact_geom_names and second in sensors
                or second in tuning.object_contact_geom_names and first in sensors):
            attributes["name"] = _object_contact_pair_name(first, second, tuning)
        for name in ("solref", "solimp"):
            if getattr(tuning, name) is not None:
                attributes[name] = " ".join(map(str, getattr(tuning, name)))
        if tuning.friction is not None:
            slide, spin, roll = tuning.friction
            attributes["friction"] = " ".join(map(str, (slide, slide, spin, roll, roll)))
        attributes["condim"] = str(max(int(geoms[name].get("condim", "3")) for name in (first, second)))
        ET.SubElement(_find_or_add(mujoco_root, "contact"), "pair", attributes)


__all__ = ["add_gripper_position_actuators"]
