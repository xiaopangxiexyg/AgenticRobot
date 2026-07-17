"""RM65 dual-arm collision scene profiles.

This module contains only model/profile data needed by shared motion planning.
Simulation viewers and the old SAPIEN simulator are intentionally not part of
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from .assets import require_asset_path
from .constants import DEFAULT_COLLISION_SCENE


DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (
    ("arm_link3_proxy_0", "right_arm_link3_proxy_0"),
    ("arm_link3_proxy_0", "right_hand_proxy_0"),
    ("left_hand_proxy_0", "right_arm_link3_proxy_0"),
    ("left_hand_proxy_0", "right_hand_proxy_0"),
    ("left_hand_proxy_0", "camera_pole_0"),
    ("right_hand_proxy_0", "camera_pole_0"),
    ("left_hand_proxy_0", "table_0"),
    ("right_hand_proxy_0", "table_0"),
    ("arm_link3_proxy_0", "table_0"),
    ("right_arm_link3_proxy_0", "table_0"),
    ("arm_link3_proxy_0", "left_camera_pole_0"),
    ("right_arm_link3_proxy_0", "right_camera_pole_0"),
)

DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES = (
    "camera_pole_0",
    "left_camera_pole_0",
    "right_camera_pole_0",
    "table_0",
)

_POSE_TRANSFORM_LEFT = np.array(
    [
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)
_POSE_TRANSFORM_RIGHT = np.array(
    [
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class HandProxyBoxSpec:
    name: str
    side: Literal["left", "right"]
    geometry_name: str
    parent_link_name: str
    box_size_xyz: tuple[float, float, float]
    joint_origin_xyz: tuple[float, float, float]
    joint_origin_rpy: tuple[float, float, float]
    box_origin_xyz: tuple[float, float, float]
    box_origin_rpy: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]

    def parent_to_box_pose_matrix(self) -> np.ndarray:
        return _pose_matrix_from_xyz_rpy(
            self.joint_origin_xyz,
            self.joint_origin_rpy,
        ) @ _pose_matrix_from_xyz_rpy(
            self.box_origin_xyz,
            self.box_origin_rpy,
        )


@dataclass(frozen=True)
class FixedSceneBoxSpec:
    name: str
    geometry_name: str
    parent_link_name: str
    box_size_xyz: tuple[float, float, float]
    joint_origin_xyz: tuple[float, float, float]
    joint_origin_rpy: tuple[float, float, float]
    box_origin_xyz: tuple[float, float, float]
    box_origin_rpy: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]

    def world_to_box_pose_matrix(self) -> np.ndarray:
        return _pose_matrix_from_xyz_rpy(
            self.joint_origin_xyz,
            self.joint_origin_rpy,
        ) @ _pose_matrix_from_xyz_rpy(
            self.box_origin_xyz,
            self.box_origin_rpy,
        )


@dataclass(frozen=True)
class DualArmSceneProfile:
    name: str
    urdf_path: Path
    package_dirs: tuple[Path, ...]
    left_frame_name: str
    right_frame_name: str
    collision_pairs: tuple[tuple[str, str], ...]
    hand_proxy_boxes: tuple[HandProxyBoxSpec, ...]
    fixed_scene_boxes: tuple[FixedSceneBoxSpec, ...]
    left_pose_transform: np.ndarray
    right_pose_transform: np.ndarray
    solution_layout: Literal["auto", "dual_arm_hand"]


def _parse_float_tuple(
    text: str | None,
    *,
    count: int,
    default: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    if text is None:
        if default is None:
            raise ValueError("Missing numeric tuple text.")
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {values}.")
    return values


def _pose_matrix_from_xyz_rpy(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


def _find_named_xml_element(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.findall(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"Could not find <{tag} name='{name}'> in scene URDF.")


def _parse_origin(
    element: ET.Element,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = element.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = _parse_float_tuple(origin.get("xyz"), count=3, default=(0.0, 0.0, 0.0))
    rpy = _parse_float_tuple(origin.get("rpy"), count=3, default=(0.0, 0.0, 0.0))
    return xyz, rpy


def _extract_link_box_geometry(
    link_element: ET.Element,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    for container_tag in ("collision", "visual"):
        for container in link_element.findall(container_tag):
            box = container.find("geometry/box")
            if box is None:
                continue
            size = _parse_float_tuple(box.get("size"), count=3)
            xyz, rpy = _parse_origin(container)
            return size, xyz, rpy
    raise ValueError(f"Link '{link_element.get('name')}' does not contain a box geometry.")


def _extract_link_color_rgba(link_element: ET.Element) -> tuple[float, float, float, float]:
    for visual in link_element.findall("visual"):
        color = visual.find("material/color")
        if color is None:
            continue
        return _parse_float_tuple(color.get("rgba"), count=4)
    return (0.8, 0.8, 0.8, 0.25)


def _find_fixed_joint_for_link(root: ET.Element, link_name: str) -> ET.Element:
    for candidate in root.findall("joint"):
        child = candidate.find("child")
        if child is not None and child.get("link") == link_name:
            return candidate
    raise ValueError(f"Could not find fixed joint for link '{link_name}'.")


def _build_hand_proxy_spec(
    root: ET.Element,
    *,
    link_name: str,
    side: Literal["left", "right"],
) -> HandProxyBoxSpec:
    link_element = _find_named_xml_element(root, "link", link_name)
    size, box_origin_xyz, box_origin_rpy = _extract_link_box_geometry(link_element)
    joint_element = _find_fixed_joint_for_link(root, link_name)
    parent = joint_element.find("parent")
    if parent is None or parent.get("link") is None:
        raise ValueError(f"Proxy link '{link_name}' is missing a parent link in its joint.")
    joint_origin_xyz, joint_origin_rpy = _parse_origin(joint_element)
    return HandProxyBoxSpec(
        name=link_name,
        side=side,
        geometry_name=f"{link_name}_0",
        parent_link_name=parent.get("link"),
        box_size_xyz=size,
        joint_origin_xyz=joint_origin_xyz,
        joint_origin_rpy=joint_origin_rpy,
        box_origin_xyz=box_origin_xyz,
        box_origin_rpy=box_origin_rpy,
        color_rgba=_extract_link_color_rgba(link_element),
    )


def _build_fixed_scene_box_spec(
    root: ET.Element,
    *,
    link_name: str,
) -> FixedSceneBoxSpec:
    link_element = _find_named_xml_element(root, "link", link_name)
    size, box_origin_xyz, box_origin_rpy = _extract_link_box_geometry(link_element)
    joint_element = _find_fixed_joint_for_link(root, link_name)
    parent = joint_element.find("parent")
    if parent is None or parent.get("link") is None:
        raise ValueError(f"Scene box '{link_name}' is missing a parent link in its joint.")
    joint_origin_xyz, joint_origin_rpy = _parse_origin(joint_element)
    return FixedSceneBoxSpec(
        name=link_name,
        geometry_name=f"{link_name}_0",
        parent_link_name=parent.get("link"),
        box_size_xyz=size,
        joint_origin_xyz=joint_origin_xyz,
        joint_origin_rpy=joint_origin_rpy,
        box_origin_xyz=box_origin_xyz,
        box_origin_rpy=box_origin_rpy,
        color_rgba=_extract_link_color_rgba(link_element),
    )


def _load_hand_proxy_boxes(urdf_path: Path) -> tuple[HandProxyBoxSpec, ...]:
    root = ET.parse(urdf_path).getroot()
    return (
        _build_hand_proxy_spec(root, link_name="left_hand_proxy", side="left"),
        _build_hand_proxy_spec(root, link_name="right_hand_proxy", side="right"),
    )


def _load_link3_proxy_boxes(urdf_path: Path) -> tuple[HandProxyBoxSpec, ...]:
    root = ET.parse(urdf_path).getroot()
    return (
        _build_hand_proxy_spec(root, link_name="arm_link3_proxy", side="left"),
        _build_hand_proxy_spec(root, link_name="right_arm_link3_proxy", side="right"),
    )


def _load_fixed_scene_box(urdf_path: Path, *, link_name: str) -> FixedSceneBoxSpec:
    root = ET.parse(urdf_path).getroot()
    return _build_fixed_scene_box_spec(root, link_name=link_name)


def build_dual_arm_scene_profile(name: str = DEFAULT_COLLISION_SCENE) -> DualArmSceneProfile:
    """Build a pure RM65 dual-arm collision/model profile."""

    name = str(name or DEFAULT_COLLISION_SCENE)
    asset_dir = require_asset_path()
    if name == "dual_arm":
        left_links = (
            "arm_link1_0",
            "arm_link2_0",
            "arm_link3_proxy_0",
            "arm_link4_0",
            "arm_link5_0",
        )
        right_links = (
            "right_arm_link1_0",
            "right_arm_link2_0",
            "right_arm_link3_proxy_0",
            "right_arm_link4_0",
            "right_arm_link5_0",
        )
        collision_pairs = tuple((left, right) for left in left_links for right in right_links)
        return DualArmSceneProfile(
            name=name,
            urdf_path=asset_dir / "rm65_dual_arm.urdf",
            package_dirs=(asset_dir,),
            left_frame_name="arm_link6",
            right_frame_name="right_arm_link6",
            collision_pairs=collision_pairs,
            hand_proxy_boxes=(),
            fixed_scene_boxes=(),
            left_pose_transform=_POSE_TRANSFORM_LEFT.copy(),
            right_pose_transform=_POSE_TRANSFORM_RIGHT.copy(),
            solution_layout="auto",
        )

    if name in {"dual_arm_hand_proxy_open", "dual_arm_hand_proxy_strict"}:
        urdf_name = (
            "rm65_dual_arm_hand_proxy_strict.urdf"
            if name == "dual_arm_hand_proxy_strict"
            else "rm65_dual_arm_hand_proxy_open.urdf"
        )
        urdf_path = asset_dir / urdf_name
        hand_proxy_boxes = _load_hand_proxy_boxes(urdf_path)
        link3_proxy_boxes = _load_link3_proxy_boxes(urdf_path)
        fixed_scene_boxes = (
            _load_fixed_scene_box(urdf_path, link_name="camera_pole"),
            _load_fixed_scene_box(urdf_path, link_name="left_camera_pole"),
            _load_fixed_scene_box(urdf_path, link_name="right_camera_pole"),
            _load_fixed_scene_box(urdf_path, link_name="table"),
        )
        if tuple(box.geometry_name for box in fixed_scene_boxes) != (
            DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES
        ):
            raise ValueError(
                "Hand-proxy collision scene URDF no longer exposes the required "
                f"fixed geometry set {DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES}."
            )
        return DualArmSceneProfile(
            name=name,
            urdf_path=urdf_path,
            package_dirs=(asset_dir,),
            left_frame_name="arm_link6",
            right_frame_name="right_arm_link6",
            collision_pairs=DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
            hand_proxy_boxes=(*hand_proxy_boxes, *link3_proxy_boxes),
            fixed_scene_boxes=fixed_scene_boxes,
            left_pose_transform=_POSE_TRANSFORM_LEFT.copy(),
            right_pose_transform=_POSE_TRANSFORM_RIGHT.copy(),
            solution_layout="auto",
        )

    if name == "dual_arm_hand":
        left_links = (
            "left_arm_link1_0",
            "left_arm_link2_0",
            "left_arm_link3_0",
            "left_arm_link4_0",
            "left_arm_link5_0",
            "left_thumb_1_0",
            "left_thumb_2_0",
            "left_thumb_3_0",
            "left_thumb_4_0",
            "left_index_1_0",
            "left_index_2_0",
            "left_middle_1_0",
            "left_middle_2_0",
            "left_ring_1_0",
            "left_ring_2_0",
            "left_little_1_0",
            "left_little_2_0",
        )
        right_links = (
            "right_arm_link1_0",
            "right_arm_link2_0",
            "right_arm_link3_0",
            "right_arm_link4_0",
            "right_arm_link5_0",
            "right_thumb_1_0",
            "right_thumb_2_0",
            "right_thumb_3_0",
            "right_thumb_4_0",
            "right_index_1_0",
            "right_index_2_0",
            "right_middle_1_0",
            "right_middle_2_0",
            "right_ring_1_0",
            "right_ring_2_0",
            "right_little_1_0",
            "right_little_2_0",
        )
        collision_pairs = tuple((left, right) for left in left_links for right in right_links)
        return DualArmSceneProfile(
            name=name,
            urdf_path=asset_dir / "rm65_dual_arm_hand.urdf",
            package_dirs=(asset_dir,),
            left_frame_name="left_arm_link6",
            right_frame_name="right_arm_link6",
            collision_pairs=collision_pairs,
            hand_proxy_boxes=(),
            fixed_scene_boxes=(),
            left_pose_transform=_POSE_TRANSFORM_LEFT.copy(),
            right_pose_transform=_POSE_TRANSFORM_RIGHT.copy(),
            solution_layout="dual_arm_hand",
        )

    if name == "dual_arm_cylinder_table":
        left_links = (
            "arm_link1_0",
            "arm_link2_0",
            "arm_link3_0",
            "arm_link4_0",
            "arm_link5_0",
            "left_hand_0",
        )
        right_links = (
            "xxx_arm_link1_0",
            "xxx_arm_link2_0",
            "xxx_arm_link3_0",
            "xxx_arm_link4_0",
            "xxx_arm_link5_0",
            "right_hand_0",
        )
        collision_pairs = [(left, right) for left in left_links for right in right_links]
        collision_pairs.extend(
            [
                ("left_hand_0", "table_0"),
                ("right_hand_0", "table_0"),
            ]
        )
        return DualArmSceneProfile(
            name=name,
            urdf_path=asset_dir / "rm65_dual_arm_cylinder_table.urdf",
            package_dirs=(asset_dir,),
            left_frame_name="arm_link6",
            right_frame_name="xxx_arm_link6",
            collision_pairs=tuple(collision_pairs),
            hand_proxy_boxes=(),
            fixed_scene_boxes=(),
            left_pose_transform=_POSE_TRANSFORM_LEFT.copy(),
            right_pose_transform=_POSE_TRANSFORM_RIGHT.copy(),
            solution_layout="auto",
        )

    raise ValueError(f"Unknown dual arm scene profile: {name}")


def build_rm65_dual_arm_scene_profile(
    name: str = DEFAULT_COLLISION_SCENE,
) -> DualArmSceneProfile:
    return build_dual_arm_scene_profile(name)


__all__ = [
    "DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS",
    "DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES",
    "DualArmSceneProfile",
    "FixedSceneBoxSpec",
    "HandProxyBoxSpec",
    "build_dual_arm_scene_profile",
    "build_rm65_dual_arm_scene_profile",
]
