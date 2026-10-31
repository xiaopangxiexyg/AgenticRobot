"""Derive the RM65-6F motion and Station models from the reviewed standard RM65 assets.

The RealMan RM65-6F carries a six-axis force sensor between the standard
link6 and the mechanical flange: the controller's flange is
``RM65_6F_FLANGE_EXTENSION_M`` further along the joint-6 axis than the
standard model's link6 (constant over 152 calibration poses, 2026-10-31).

Both outputs lengthen each arm's joint-6 origin by that amount, so link6 is the
6F flange and every flange-relative geometry (hand proxies, gripper, wrist
cameras, the tool TCP of the tool profiles) keeps its reviewed offsets. The
standard link6 body is kept where it was (its mesh and inertial origin move
back by the extension) and the Station model adds the sensor body between it
and the flange, so the capture self filter still covers the whole wrist.

The Station model additionally binds the camera registry's calibration:
``head_camera_frame`` takes the registry's ``T_camera_to_parent`` and each
``<side>_camera_rgb`` takes its ``calibration.T_camera_to_arm_end`` relative to
the 6F flange. Every written transform is read back and compared with the
registry (1e-9).

The asset store is not versioned; this script is the reviewed derivation. It
writes only new files and refuses to replace a different existing file.

usage: make_rm65_6f_station_assets.py [--asset-dir DIR] [--registry PATH] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from robot_motion.rm65.scene_profiles import (
    RM65_6F_FLANGE_EXTENSION_M,
    RM65_STANDARD_JOINT6_OFFSET_M,
)

PLANNING_SOURCE = "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
PLANNING_OUTPUT = "rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict_rm65_6f.urdf"
STATION_SOURCE = "realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"
STATION_OUTPUT = "realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm_rm65_6f_20261031.urdf"
DEFAULT_REGISTRY = REPO / "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json"
# Sensor body between the standard flange face (r = 28.5 mm) and the 6F flange.
FORCE_SENSOR_RADIUS_M = 0.030


def _floats(text: str | None) -> np.ndarray:
    return np.asarray((text or "0 0 0").split(), dtype=float)


def _fmt(values) -> str:
    return " ".join(repr(float(value)) for value in values)


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """Same fixed-axis rpy formula as agent_gateway's camera geometry."""
    xyz = _floats(None if origin is None else origin.get("xyz"))
    roll, pitch, yaw = _floats(None if origin is None else origin.get("rpy"))
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    out = np.eye(4)
    out[:3, :3] = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                   (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                   (-sp, cp * sr, cp * cr))
    out[:3, 3] = xyz
    return out


def _set_origin(element: ET.Element, matrix: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    rpy = Rotation.from_matrix(u @ vt).as_euler("xyz")
    origin.set("xyz", _fmt(np.asarray(matrix, dtype=float)[:3, 3]))
    origin.set("rpy", _fmt(rpy))


def _parse(path: Path) -> ET.ElementTree:
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def _unique(root: ET.Element, tag: str, name: str) -> ET.Element:
    found = [element for element in root.findall(tag) if element.get("name") == name]
    if len(found) != 1:
        raise SystemExit(f"expected one <{tag} name={name!r}>, found {len(found)}")
    return found[0]


def _lengthen_joint6(root: ET.Element, joint_name: str, link_name: str, *, add_sensor: bool) -> None:
    joint = _unique(root, "joint", joint_name)
    if joint.find("child").get("link") != link_name:
        raise SystemExit(f"{joint_name} does not drive {link_name}")
    origin = _origin_matrix(joint.find("origin"))
    length = float(np.linalg.norm(origin[:3, 3]))
    if abs(length - RM65_STANDARD_JOINT6_OFFSET_M) > 1e-9:
        raise SystemExit(f"{joint_name} is not the standard RM65 joint 6 ({length} m)")
    # Translate along the child (joint-6) z axis: the sensor stacks on the flange face.
    origin[:3, 3] = origin[:3, 3] + origin[:3, :3] @ np.asarray((0.0, 0.0, RM65_6F_FLANGE_EXTENSION_M))
    joint.find("origin").set("xyz", _fmt(origin[:3, 3]))
    link = _unique(root, "link", link_name)
    # Keep the standard link6 body where it was relative to link5.
    for element in (*link.findall("visual"), *link.findall("collision"), *link.findall("inertial")):
        local = element.find("origin")
        if local is None:
            local = ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        xyz = _floats(local.get("xyz"))
        if np.abs(_floats(local.get("rpy"))).max() > 0.0:
            raise SystemExit(f"{link_name} body origins must be unrotated")
        xyz[2] -= RM65_6F_FLANGE_EXTENSION_M
        local.set("xyz", _fmt(xyz))
    if add_sensor:
        prefix = link_name[: -len("link6")]
        sensor = f"{prefix}force_sensor_6f"
        body = ET.Element("link", {"name": sensor})
        for kind in ("visual", "collision"):
            part = ET.SubElement(body, kind)
            ET.SubElement(part, "origin", {"xyz": _fmt((0.0, 0.0, -RM65_6F_FLANGE_EXTENSION_M / 2.0)), "rpy": "0 0 0"})
            geometry = ET.SubElement(part, "geometry")
            ET.SubElement(geometry, "cylinder", {"radius": repr(FORCE_SENSOR_RADIUS_M),
                                                 "length": repr(RM65_6F_FLANGE_EXTENSION_M)})
            if kind == "visual":
                ET.SubElement(part, "material", {"name": f"{sensor}_material"}).append(
                    ET.Element("color", {"rgba": "0.2 0.2 0.22 1"}))
        fixed = ET.Element("joint", {"name": f"{link_name}_to_{sensor}", "type": "fixed"})
        ET.SubElement(fixed, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(fixed, "parent", {"link": link_name})
        ET.SubElement(fixed, "child", {"link": sensor})
        children = list(root)
        index = children.index(joint) + 1
        root.insert(index, fixed)
        root.insert(index, body)


def _chain(root: ET.Element, ancestor: str, descendant: str) -> tuple[np.ndarray, list[ET.Element]]:
    """ancestor -> descendant through fixed joints, and those joints from the descendant up."""
    by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    transform, joints, current = np.eye(4), [], descendant
    while current != ancestor:
        joint = by_child[current]
        if joint.get("type") != "fixed":
            raise SystemExit(f"{ancestor}->{descendant} is not a fixed chain")
        transform = _origin_matrix(joint.find("origin")) @ transform
        joints.append(joint)
        current = joint.find("parent").get("link")
    return transform, joints


def _station(root: ET.Element, registry: dict) -> None:
    for side in ("left", "right"):
        _lengthen_joint6(root, f"{side}_arm_joint6", f"{side}_arm_link6", add_sensor=True)
    head = registry["cameras"]["front"]["extrinsics"]["frames"]["head_camera_frame"]
    if head["parent_frame"] != "left_arm_base_link":
        raise SystemExit("head camera registry parent must be left_arm_base_link")
    head_joint = _unique(root, "joint", "left_base_to_head_camera_frame")
    _set_origin(head_joint, np.asarray(head["T_camera_to_parent"], dtype=float))
    for side, key in (("left", "left_wrist"), ("right", "right_wrist")):
        frame = registry["cameras"][key]["extrinsics"]["frames"][f"{side}_camera_rgb"]
        flange_from_camera = np.asarray(frame["calibration"]["T_camera_to_arm_end"], dtype=float)
        _, joints = _chain(root, f"{side}_arm_link6", f"{side}_camera_rgb")
        last = joints[0]
        flange_from_parent, _ = _chain(root, f"{side}_arm_link6", last.find("parent").get("link"))
        _set_origin(last, np.linalg.inv(flange_from_parent) @ flange_from_camera)


def _verify_station(root: ET.Element, registry: dict) -> dict:
    report = {}
    head = np.asarray(registry["cameras"]["front"]["extrinsics"]["frames"]["head_camera_frame"]["T_camera_to_parent"])
    actual, _ = _chain(root, "left_arm_base_link", "head_camera_frame")
    report["head_camera_max_abs_error"] = float(np.abs(actual - head).max())
    for side, key in (("left", "left_wrist"), ("right", "right_wrist")):
        expected = np.asarray(registry["cameras"][key]["extrinsics"]["frames"][f"{side}_camera_rgb"]
                              ["calibration"]["T_camera_to_arm_end"])
        actual, _ = _chain(root, f"{side}_arm_link6", f"{side}_camera_rgb")
        report[f"{side}_wrist_max_abs_error"] = float(np.abs(actual - expected).max())
    if max(report.values()) > 1e-9:
        raise SystemExit(f"Station model does not reproduce the registry: {report}")
    return report


def _write(tree: ET.ElementTree, path: Path, header: str, *, check: bool) -> str:
    ET.indent(tree, space="  ")
    body = ET.tostring(tree.getroot(), encoding="unicode")
    text = f'<?xml version="1.0" ?>\n<!-- {header} -->\n{body}\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with other content; not replaced")
        return digest
    if check:
        raise SystemExit(f"{path} is missing")
    path.write_text(text)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset-dir", type=Path, default=REPO / "assets")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--check", action="store_true", help="verify existing outputs; write nothing")
    args = parser.parse_args()
    asset_dir = args.asset_dir.resolve(strict=True)
    registry = json.loads(args.registry.read_text())
    sources = {name: hashlib.sha256((asset_dir / name).read_bytes()).hexdigest()
               for name in (PLANNING_SOURCE, STATION_SOURCE)}
    registry_sha = hashlib.sha256(args.registry.read_bytes()).hexdigest()

    planning = _parse(asset_dir / PLANNING_SOURCE)
    for joint, link in (("arm_joint6", "arm_link6"), ("right_arm_joint6", "right_arm_link6")):
        _lengthen_joint6(planning.getroot(), joint, link, add_sensor=False)
    planning_sha = _write(planning, asset_dir / PLANNING_OUTPUT,
        f"RM65-6F motion model: {PLANNING_SOURCE} (sha256 {sources[PLANNING_SOURCE]}) with joint 6 "
        f"lengthened by {RM65_6F_FLANGE_EXTENSION_M} m; generated by scripts/assets/make_rm65_6f_station_assets.py",
        check=args.check)

    station = _parse(asset_dir / STATION_SOURCE)
    _station(station.getroot(), registry)
    report = _verify_station(station.getroot(), registry)
    station_sha = _write(station, asset_dir / STATION_OUTPUT,
        f"RM65-6F Station model: {STATION_SOURCE} (sha256 {sources[STATION_SOURCE]}) with joint 6 lengthened by "
        f"{RM65_6F_FLANGE_EXTENSION_M} m, a {FORCE_SENSOR_RADIUS_M} m force-sensor body, and the cameras of "
        f"{args.registry.name} (sha256 {registry_sha}); generated by scripts/assets/make_rm65_6f_station_assets.py",
        check=args.check)
    print(json.dumps({"sources": sources, "registry_sha256": registry_sha,
                      "outputs": {PLANNING_OUTPUT: planning_sha, STATION_OUTPUT: station_sha},
                      "registry_reproduction": report}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
