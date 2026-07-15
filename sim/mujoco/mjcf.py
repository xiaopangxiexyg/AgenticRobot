from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sim.contracts.scene import SceneSpec

from .config import CameraMount
from .modeling.collision import (
    CollisionFilter,
    excluded_body_pairs,
    validate_excluded_bodies,
)
from .modeling.geometry import MeshAssetRegistry, safe_name
from .modeling.objects import add_free_body_objects
from .modeling.world import add_ground, contact_attributes
from .transforms import format_float_list, pose_to_pos_quat_text
from .urdf import (
    URDFCollision,
    URDFInertial,
    URDFJoint,
    URDFModel,
    URDFVisual,
    parse_urdf,
)


@dataclass(frozen=True)
class MJCFConversion:
    urdf_path: Path
    mjcf_path: Path
    camera_mounts: tuple[CameraMount, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    scene_profile_name: str | None = None


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


def _visual_rgba(
    visual: URDFVisual,
    *,
    mesh_rgba: tuple[float, float, float, float] | None = None,
) -> str:
    rgba = (
        visual.rgba
        if visual.rgba is not None
        else mesh_rgba
        if mesh_rgba is not None
        else (0.82, 0.82, 0.82, 1.0)
    )
    return format_float_list(rgba)


def _proxy_debug_rgba(link_name: str) -> tuple[float, float, float, float]:
    name = str(link_name).lower()
    if "left" in name and "hand" in name:
        return (1.0, 0.45, 0.05, 0.35)
    if "left" in name:
        return (1.0, 0.12, 0.12, 0.35)
    if "right" in name and "hand" in name:
        return (0.1, 1.0, 0.35, 0.35)
    if "right" in name:
        return (0.05, 0.45, 1.0, 0.35)
    return (1.0, 0.95, 0.05, 0.35)


def _is_proxy_link(link_name: str) -> bool:
    return "proxy" in str(link_name).lower()


def _is_translucent_box_visual(visual: URDFVisual) -> bool:
    rgba = visual.rgba
    if rgba is None or visual.geometry.kind != "box":
        return False
    return float(rgba[3]) < 1.0


def _add_visual_geom(
    body: ET.Element,
    *,
    link_name: str,
    index: int,
    visual: URDFVisual,
    geometry_registry: MeshAssetRegistry,
    debug_proxy_visuals: bool = False,
) -> None:
    is_proxy = _is_proxy_link(link_name)
    is_hidden_debug_visual = is_proxy or _is_translucent_box_visual(visual)
    if is_hidden_debug_visual and not debug_proxy_visuals:
        return
    pos_text, quat_text = pose_to_pos_quat_text(visual.origin)
    parts = geometry_registry.visual_geom_parts(visual.geometry)
    for part_index, part in enumerate(parts):
        suffix = ""
        if len(parts) > 1:
            suffix = f"__{safe_name(part.material_name or str(part_index))}"
        geom_attrs = {
            "name": f"{link_name}_visual_{index}{suffix}",
            "pos": pos_text,
            "quat": quat_text,
            "rgba": _visual_rgba(visual, mesh_rgba=part.rgba),
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        }
        if debug_proxy_visuals and is_hidden_debug_visual:
            if is_proxy:
                geom_attrs["rgba"] = format_float_list(_proxy_debug_rgba(link_name))
            geom_attrs["group"] = "4"
        geom_attrs.update(part.attributes)
        ET.SubElement(body, "geom", geom_attrs)


def _add_collision_geom(
    body: ET.Element,
    *,
    link_name: str,
    index: int,
    collision: URDFCollision,
    geometry_registry: MeshAssetRegistry,
    collision_filter: CollisionFilter,
    contact: ContactMaterialSpec,
) -> None:
    pos_text, quat_text = pose_to_pos_quat_text(collision.origin)
    group_name = collision_filter.group_for_urdf_link(link_name)
    attrs = {
        "name": f"{link_name}_collision_{index}",
        "pos": pos_text,
        "quat": quat_text,
        "rgba": "0 0 0 0",
        "group": "3",
        **collision_filter.geom_attributes(group_name),
        **contact_attributes(contact),
        **geometry_registry.geom_attributes(collision.geometry),
    }
    ET.SubElement(body, "geom", attrs)


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
    if joint.joint_type in {"revolute", "prismatic"} and joint.limit is not None:
        if joint.limit.lower is not None and joint.limit.upper is not None:
            attrs["range"] = format_float_list((joint.limit.lower, joint.limit.upper))
            attrs["limited"] = "true"
    ET.SubElement(body, "joint", attrs)


def _add_camera(body: ET.Element, mount: CameraMount) -> None:
    attrs = {
        "name": mount.name,
        "pos": "0 0 0",
        # URDF camera_rgb/head_camera_frame is treated as an optical frame:
        # x right, y down, z forward. MuJoCo/OpenGL cameras look along -z
        # with +y up, so the render camera is optical->OpenGL = Rx(pi).
        "quat": "0 1 0 0",
        # Keep the render projection on the stable fovy path. Full RealSense
        # intrinsics stay in metadata; MuJoCo pixel principal-point projection
        # shifted the apparent camera views relative to earlier checked output.
        "fovy": f"{float(mount.fovy_deg):.12g}",
    }
    ET.SubElement(body, "camera", attrs)


def convert_urdf_to_mjcf_xml(
    model: URDFModel,
    *,
    camera_mounts: Iterable[CameraMount],
    timestep: float = 1.0 / 30.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    debug_proxy_visuals: bool = False,
    scene_spec: SceneSpec | None = None,
) -> str:
    mounts_by_link: dict[str, list[CameraMount]] = {}
    for mount in camera_mounts:
        mounts_by_link.setdefault(mount.link_name, []).append(mount)

    resolved_gravity = gravity if scene_spec is None else scene_spec.gravity_mps2
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
            "gravity": format_float_list(resolved_gravity),
        },
    )
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25"})
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "10"})
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")
    geometry_registry = MeshAssetRegistry(asset, model)
    collision_filter = CollisionFilter()
    collision_enabled = bool(scene_spec is not None and scene_spec.collision.enabled)
    urdf_contact = None if scene_spec is None else scene_spec.collision.contact

    if collision_enabled:
        add_ground(worldbody, scene_spec.ground, collision_filter)
        add_free_body_objects(worldbody, scene_spec.objects, collision_filter)

    def populate_body(body: ET.Element, link_name: str, parent_joint: URDFJoint | None) -> None:
        link = model.links[link_name]
        _add_inertial(body, link.inertial)
        if parent_joint is not None:
            _add_joint(body, parent_joint)
        for index, visual_item in enumerate(link.visuals):
            _add_visual_geom(
                body,
                link_name=link_name,
                index=index,
                visual=visual_item,
                geometry_registry=geometry_registry,
                debug_proxy_visuals=debug_proxy_visuals,
            )
        if collision_enabled:
            for index, collision_item in enumerate(link.collisions):
                _add_collision_geom(
                    body,
                    link_name=link_name,
                    index=index,
                    collision=collision_item,
                    geometry_registry=geometry_registry,
                    collision_filter=collision_filter,
                    contact=urdf_contact,
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

    if collision_enabled:
        pairs = excluded_body_pairs(model, scene_spec.collision)
        validate_excluded_bodies(model, pairs)
        if pairs:
            contact = ET.SubElement(mujoco, "contact")
            for first, second in pairs:
                ET.SubElement(contact, "exclude", {"body1": first, "body2": second})

    ET.indent(mujoco, space="  ")
    return ET.tostring(mujoco, encoding="unicode")


def write_mjcf(
    urdf_path: str | Path,
    *,
    camera_mounts: Iterable[CameraMount],
    output_path: str | Path | None = None,
    timestep: float = 1.0 / 30.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    debug_proxy_visuals: bool = False,
    scene_spec: SceneSpec | None = None,
) -> MJCFConversion:
    model = parse_urdf(urdf_path)
    mounts = tuple(camera_mounts)
    xml_text = convert_urdf_to_mjcf_xml(
        model,
        camera_mounts=mounts,
        timestep=timestep,
        gravity=gravity,
        debug_proxy_visuals=debug_proxy_visuals,
        scene_spec=scene_spec,
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
        body_names=(
            tuple(model.links)
            + (() if scene_spec is None else tuple(obj.name for obj in scene_spec.objects))
        ),
        scene_profile_name=None if scene_spec is None else scene_spec.profile_name,
    )
