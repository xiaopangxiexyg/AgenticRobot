from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import CameraMount, default_camera_registry_path
from .transforms import pose_from_xyz_rpy


def _floats(text: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return tuple(float(value) for value in default)
    return tuple(float(part) for part in str(text).replace(",", " ").split())


def _origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=float)
    xyz = _floats(element.attrib.get("xyz"), default=(0.0, 0.0, 0.0))
    rpy = _floats(element.attrib.get("rpy"), default=(0.0, 0.0, 0.0))
    return pose_from_xyz_rpy(xyz, rpy)


@dataclass(frozen=True)
class URDFGeometry:
    kind: str
    values: tuple[float, ...] = ()
    filename: str | None = None


@dataclass(frozen=True)
class URDFVisual:
    origin: np.ndarray
    geometry: URDFGeometry
    rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class URDFInertial:
    origin: np.ndarray
    mass: float
    inertia: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class URDFLink:
    name: str
    visuals: tuple[URDFVisual, ...] = ()
    inertial: URDFInertial | None = None


@dataclass(frozen=True)
class URDFJointLimit:
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit: URDFJointLimit | None = None


@dataclass
class URDFModel:
    path: Path
    name: str
    links: dict[str, URDFLink]
    joints: list[URDFJoint]
    children_by_parent: dict[str, list[URDFJoint]] = field(default_factory=dict)
    parent_joint_by_child: dict[str, URDFJoint] = field(default_factory=dict)
    root_link: str = ""

    def link_names(self) -> tuple[str, ...]:
        return tuple(self.links)

    def resolve_mesh_path(self, filename: str) -> Path:
        mesh = Path(filename)
        if mesh.is_absolute():
            return mesh
        return (self.path.parent / mesh).resolve()


def _parse_geometry(geometry_element: ET.Element) -> URDFGeometry:
    mesh = geometry_element.find("mesh")
    if mesh is not None:
        return URDFGeometry(kind="mesh", filename=mesh.attrib["filename"])
    box = geometry_element.find("box")
    if box is not None:
        return URDFGeometry(kind="box", values=_floats(box.attrib.get("size"), default=(1, 1, 1)))
    cylinder = geometry_element.find("cylinder")
    if cylinder is not None:
        return URDFGeometry(
            kind="cylinder",
            values=(
                float(cylinder.attrib.get("radius", "0.01")),
                float(cylinder.attrib.get("length", "0.01")),
            ),
        )
    sphere = geometry_element.find("sphere")
    if sphere is not None:
        return URDFGeometry(kind="sphere", values=(float(sphere.attrib.get("radius", "0.01")),))
    raise ValueError("Unsupported or empty URDF geometry element.")


def _parse_rgba(visual_element: ET.Element) -> tuple[float, float, float, float] | None:
    material = visual_element.find("material")
    if material is None:
        return None
    color = material.find("color")
    if color is None:
        return None
    values = _floats(color.attrib.get("rgba"), default=(1.0, 1.0, 1.0, 1.0))
    if len(values) != 4:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _parse_visual(visual_element: ET.Element) -> URDFVisual | None:
    geometry_element = visual_element.find("geometry")
    if geometry_element is None:
        return None
    return URDFVisual(
        origin=_origin(visual_element.find("origin")),
        geometry=_parse_geometry(geometry_element),
        rgba=_parse_rgba(visual_element),
    )


def _parse_inertial(link_element: ET.Element) -> URDFInertial | None:
    inertial = link_element.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None:
        return None
    return URDFInertial(
        origin=_origin(inertial.find("origin")),
        mass=float(mass_element.attrib.get("value", "0")),
        inertia=(
            float(inertia_element.attrib.get("ixx", "0")),
            float(inertia_element.attrib.get("iyy", "0")),
            float(inertia_element.attrib.get("izz", "0")),
            float(inertia_element.attrib.get("ixy", "0")),
            float(inertia_element.attrib.get("ixz", "0")),
            float(inertia_element.attrib.get("iyz", "0")),
        ),
    )


def parse_urdf(path: str | Path) -> URDFModel:
    urdf_path = Path(path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected URDF root tag 'robot', got {root.tag!r}.")
    links: dict[str, URDFLink] = {}
    for link_element in root.findall("link"):
        name = link_element.attrib["name"]
        visuals = tuple(
            visual
            for visual in (_parse_visual(element) for element in link_element.findall("visual"))
            if visual is not None
        )
        links[name] = URDFLink(
            name=name,
            visuals=visuals,
            inertial=_parse_inertial(link_element),
        )

    joints: list[URDFJoint] = []
    for joint_element in root.findall("joint"):
        parent = joint_element.find("parent")
        child = joint_element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint_element.attrib.get('name')} is missing parent/child.")
        axis = joint_element.find("axis")
        limit = joint_element.find("limit")
        joint_limit = None
        if limit is not None:
            joint_limit = URDFJointLimit(
                lower=float(limit.attrib["lower"]) if "lower" in limit.attrib else None,
                upper=float(limit.attrib["upper"]) if "upper" in limit.attrib else None,
                effort=float(limit.attrib["effort"]) if "effort" in limit.attrib else None,
                velocity=float(limit.attrib["velocity"]) if "velocity" in limit.attrib else None,
            )
        joints.append(
            URDFJoint(
                name=joint_element.attrib["name"],
                joint_type=joint_element.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin=_origin(joint_element.find("origin")),
                axis=tuple(_floats(None if axis is None else axis.attrib.get("xyz"), default=(0, 0, 1))),  # type: ignore[arg-type]
                limit=joint_limit,
            )
        )

    children_by_parent: dict[str, list[URDFJoint]] = {name: [] for name in links}
    parent_joint_by_child: dict[str, URDFJoint] = {}
    for joint in joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)
        parent_joint_by_child[joint.child] = joint

    child_links = {joint.child for joint in joints}
    roots = [name for name in links if name not in child_links]
    root_link = roots[0] if roots else ""
    return URDFModel(
        path=urdf_path,
        name=root.attrib.get("name", urdf_path.stem),
        links=links,
        joints=joints,
        children_by_parent=children_by_parent,
        parent_joint_by_child=parent_joint_by_child,
        root_link=root_link,
    )


_CAMERA_LINK_RE = re.compile(r"(^camera_rgb$|_camera_rgb$|^head_camera_frame$)")


def _fovy_from_intrinsics(width: int, height: int, intrinsics: Mapping[str, object] | None) -> float | None:
    if not isinstance(intrinsics, Mapping):
        return None
    fy = intrinsics.get("fy")
    if fy is None:
        return None
    fy_value = float(fy)
    if fy_value <= 0.0:
        return None
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * fy_value)))


def _registry_camera_mounts(
    model: URDFModel,
    *,
    registry_path: Path,
    default_width: int,
    default_height: int,
    default_fovy_deg: float,
    camera_intrinsics_by_name: Mapping[str, Mapping[str, object]] | None = None,
) -> list[CameraMount]:
    if not registry_path.exists():
        return []
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cameras = registry.get("cameras", {})
    if not isinstance(cameras, Mapping):
        return []
    repo_root = Path(__file__).resolve().parents[2]
    mounts: list[CameraMount] = []
    for camera_key, camera_info in cameras.items():
        if not isinstance(camera_info, Mapping):
            continue
        profile = camera_info.get("profile") if isinstance(camera_info.get("profile"), Mapping) else {}
        width = int(profile.get("width", default_width))
        height = int(profile.get("height", default_height))
        intrinsics = camera_info.get("intrinsics")
        runtime_intrinsics = (
            camera_intrinsics_by_name.get(str(camera_key))
            if isinstance(camera_intrinsics_by_name, Mapping)
            else None
        )
        if isinstance(runtime_intrinsics, Mapping):
            merged_intrinsics = dict(intrinsics) if isinstance(intrinsics, Mapping) else {}
            merged_intrinsics.update(dict(runtime_intrinsics))
            intrinsics = merged_intrinsics
            width = int(merged_intrinsics.get("width", width))
            height = int(merged_intrinsics.get("height", height))
        fovy = _fovy_from_intrinsics(width, height, intrinsics) or float(default_fovy_deg)
        extrinsics = camera_info.get("extrinsics")
        frames = extrinsics.get("frames") if isinstance(extrinsics, Mapping) else None
        if not isinstance(frames, Mapping):
            continue
        for frame_name, frame_info in frames.items():
            if str(frame_name) not in model.links:
                continue
            placements = frame_info.get("placements") if isinstance(frame_info, Mapping) else None
            if isinstance(placements, list):
                matched = False
                for placement in placements:
                    if not isinstance(placement, Mapping):
                        continue
                    placement_urdf = placement.get("urdf")
                    placement_frame = placement.get("frame")
                    if placement_frame is not None and str(placement_frame) != str(frame_name):
                        continue
                    if placement_urdf is None:
                        matched = True
                        break
                    candidate = (repo_root / str(placement_urdf)).resolve()
                    if candidate == model.path:
                        matched = True
                        break
                if not matched:
                    continue
            mounts.append(
                CameraMount(
                    name=str(camera_key),
                    link_name=str(frame_name),
                    width=width,
                    height=height,
                    fovy_deg=fovy,
                    intrinsics=intrinsics if isinstance(intrinsics, Mapping) else None,
                    source=f"registry:{registry_path}",
                )
            )
    return mounts


def discover_camera_mounts(
    model: URDFModel,
    *,
    camera_names: Sequence[str] = ("all",),
    registry_path: str | Path | None = None,
    default_width: int = 640,
    default_height: int = 480,
    default_fovy_deg: float = 60.0,
    camera_intrinsics_by_name: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[CameraMount, ...]:
    registry = Path(registry_path).expanduser().resolve() if registry_path else default_camera_registry_path()
    mounts = _registry_camera_mounts(
        model,
        registry_path=registry,
        default_width=default_width,
        default_height=default_height,
        default_fovy_deg=default_fovy_deg,
        camera_intrinsics_by_name=camera_intrinsics_by_name,
    )
    if not mounts:
        mounts = [
            CameraMount(
                name=name,
                link_name=name,
                width=default_width,
                height=default_height,
                fovy_deg=default_fovy_deg,
                source="urdf_link_name",
            )
            for name in model.links
            if _CAMERA_LINK_RE.search(name) and not name.startswith("debug_")
        ]

    requested = tuple(str(name) for name in camera_names)
    if not requested or "all" in requested:
        return tuple(mounts)

    by_name = {mount.name: mount for mount in mounts}
    by_link = {mount.link_name: mount for mount in mounts}
    selected: list[CameraMount] = []
    missing: list[str] = []
    for name in requested:
        mount = by_name.get(name) or by_link.get(name)
        if mount is not None:
            selected.append(mount)
        elif name in model.links:
            selected.append(
                CameraMount(
                    name=name,
                    link_name=name,
                    width=default_width,
                    height=default_height,
                    fovy_deg=default_fovy_deg,
                    source="explicit_urdf_link",
                )
            )
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            "Requested camera(s) not found in registry or URDF links: " + ", ".join(missing)
        )
    return tuple(selected)
