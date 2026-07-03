from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .config import CameraMount
from .transforms import format_float_list, pose_to_pos_quat_text
from .urdf import URDFGeometry, URDFInertial, URDFJoint, URDFModel, URDFVisual, parse_urdf


@dataclass(frozen=True)
class MJCFConversion:
    urdf_path: Path
    mjcf_path: Path
    camera_mounts: tuple[CameraMount, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]


def _safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    if not name:
        return "asset"
    if name[0].isdigit():
        return f"asset_{name}"
    return name


def _mesh_asset_name(path: Path, used: set[str]) -> str:
    stem = _safe_name(path.stem)
    name = stem
    index = 1
    while name in used:
        index += 1
        name = f"{stem}_{index}"
    used.add(name)
    return name


def _add_inertial(body: ET.Element, inertial: URDFInertial | None) -> None:
    if inertial is None or inertial.mass <= 0.0:
        return
    ixx, iyy, izz, ixy, ixz, iyz = inertial.inertia
    if max(abs(ixx), abs(iyy), abs(izz), abs(ixy), abs(ixz), abs(iyz)) <= 0.0:
        return
    pos_text, _quat_text = pose_to_pos_quat_text(inertial.origin)
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": pos_text,
            "mass": f"{float(inertial.mass):.12g}",
            "fullinertia": format_float_list((ixx, iyy, izz, ixy, ixz, iyz)),
        },
    )


def _visual_rgba(visual: URDFVisual) -> str:
    rgba = visual.rgba if visual.rgba is not None else (0.82, 0.82, 0.82, 1.0)
    return format_float_list(rgba)


def _add_geom(
    body: ET.Element,
    *,
    link_name: str,
    index: int,
    visual: URDFVisual,
    model: URDFModel,
    asset: ET.Element,
    mesh_assets: dict[Path, str],
    used_mesh_names: set[str],
) -> None:
    pos_text, quat_text = pose_to_pos_quat_text(visual.origin)
    geom_attrs = {
        "name": f"{link_name}_visual_{index}",
        "pos": pos_text,
        "quat": quat_text,
        "rgba": _visual_rgba(visual),
        "contype": "0",
        "conaffinity": "0",
    }
    geometry = visual.geometry
    if geometry.kind == "mesh":
        if geometry.filename is None:
            return
        mesh_path = model.resolve_mesh_path(geometry.filename)
        if not mesh_path.exists():
            raise FileNotFoundError(f"URDF mesh not found: {mesh_path}")
        mesh_name = mesh_assets.get(mesh_path)
        if mesh_name is None:
            mesh_name = _mesh_asset_name(mesh_path, used_mesh_names)
            mesh_assets[mesh_path] = mesh_name
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_path)})
        geom_attrs.update({"type": "mesh", "mesh": mesh_name})
    elif geometry.kind == "box":
        size = np.asarray(geometry.values, dtype=float).reshape(3) / 2.0
        geom_attrs.update({"type": "box", "size": format_float_list(size)})
    elif geometry.kind == "cylinder":
        radius, length = geometry.values
        geom_attrs.update({"type": "cylinder", "size": format_float_list((radius, length / 2.0))})
    elif geometry.kind == "sphere":
        (radius,) = geometry.values
        geom_attrs.update({"type": "sphere", "size": f"{float(radius):.12g}"})
    else:
        raise ValueError(f"Unsupported URDF geometry kind: {geometry.kind}")
    ET.SubElement(body, "geom", geom_attrs)


def _add_joint(body: ET.Element, joint: URDFJoint) -> None:
    if joint.joint_type in {"fixed", ""}:
        return
    if joint.joint_type not in {"revolute", "continuous", "prismatic"}:
        raise ValueError(f"Unsupported joint type for MuJoCo conversion: {joint.joint_type}")
    attrs = {
        "name": joint.name,
        "type": "slide" if joint.joint_type == "prismatic" else "hinge",
        "axis": format_float_list(joint.axis),
    }
    if joint.joint_type == "revolute" and joint.limit is not None:
        if joint.limit.lower is not None and joint.limit.upper is not None:
            attrs["range"] = format_float_list((joint.limit.lower, joint.limit.upper))
            attrs["limited"] = "true"
    ET.SubElement(body, "joint", attrs)


def _add_camera(body: ET.Element, mount: CameraMount) -> None:
    ET.SubElement(
        body,
        "camera",
        {
            "name": mount.name,
            "pos": "0 0 0",
            # URDF camera_rgb/head_camera_frame is treated as an optical frame:
            # x right, y down, z forward. MuJoCo/OpenGL cameras look along -z
            # with +y up, so the render camera is optical->OpenGL = Rx(pi).
            "quat": "0 1 0 0",
            "fovy": f"{float(mount.fovy_deg):.12g}",
        },
    )


def convert_urdf_to_mjcf_xml(
    model: URDFModel,
    *,
    camera_mounts: Iterable[CameraMount],
    timestep: float = 1.0 / 30.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    mounts_by_link: dict[str, list[CameraMount]] = {}
    for mount in camera_mounts:
        mounts_by_link.setdefault(mount.link_name, []).append(mount)

    mujoco = ET.Element("mujoco", {"model": model.name})
    ET.SubElement(
        mujoco,
        "compiler",
        {
            "angle": "radian",
            "autolimits": "true",
            "balanceinertia": "true",
            "boundmass": "1e-9",
            "boundinertia": "1e-9",
        },
    )
    ET.SubElement(
        mujoco,
        "option",
        {
            "timestep": f"{float(timestep):.12g}",
            "gravity": format_float_list(gravity),
        },
    )
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25"})
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "10"})
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")

    mesh_assets: dict[Path, str] = {}
    used_mesh_names: set[str] = set()

    def populate_body(body: ET.Element, link_name: str, parent_joint: URDFJoint | None) -> None:
        link = model.links[link_name]
        _add_inertial(body, link.inertial)
        if parent_joint is not None:
            _add_joint(body, parent_joint)
        for index, visual_item in enumerate(link.visuals):
            _add_geom(
                body,
                link_name=link_name,
                index=index,
                visual=visual_item,
                model=model,
                asset=asset,
                mesh_assets=mesh_assets,
                used_mesh_names=used_mesh_names,
            )
        for mount in mounts_by_link.get(link_name, []):
            _add_camera(body, mount)
        for child_joint in model.children_by_parent.get(link_name, []):
            pos_text, quat_text = pose_to_pos_quat_text(child_joint.origin)
            child_body = ET.SubElement(
                body,
                "body",
                {"name": child_joint.child, "pos": pos_text, "quat": quat_text},
            )
            populate_body(child_body, child_joint.child, child_joint)

    root = model.root_link
    if root == "world":
        for child in model.children_by_parent.get(root, []):
            pos_text, quat_text = pose_to_pos_quat_text(child.origin)
            body = ET.SubElement(
                worldbody,
                "body",
                {"name": child.child, "pos": pos_text, "quat": quat_text},
            )
            populate_body(body, child.child, child)
    else:
        body = ET.SubElement(worldbody, "body", {"name": root, "pos": "0 0 0", "quat": "1 0 0 0"})
        populate_body(body, root, None)

    ET.indent(mujoco, space="  ")
    return ET.tostring(mujoco, encoding="unicode")


def write_mjcf(
    urdf_path: str | Path,
    *,
    camera_mounts: Iterable[CameraMount],
    output_path: str | Path | None = None,
    timestep: float = 1.0 / 30.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MJCFConversion:
    model = parse_urdf(urdf_path)
    mounts = tuple(camera_mounts)
    xml_text = convert_urdf_to_mjcf_xml(
        model,
        camera_mounts=mounts,
        timestep=timestep,
        gravity=gravity,
    )
    if output_path is None:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=f"{model.path.stem}_",
            delete=False,
            encoding="utf-8",
        )
        mjcf_path = Path(handle.name)
        with handle:
            handle.write(xml_text)
    else:
        mjcf_path = Path(output_path).expanduser().resolve()
        mjcf_path.parent.mkdir(parents=True, exist_ok=True)
        mjcf_path.write_text(xml_text, encoding="utf-8")
    return MJCFConversion(
        urdf_path=model.path,
        mjcf_path=mjcf_path,
        camera_mounts=mounts,
        joint_names=tuple(joint.name for joint in model.joints if joint.joint_type != "fixed"),
        body_names=tuple(model.links),
    )
