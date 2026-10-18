"""RM65 dual-arm collision scene profiles.

This module contains only model/profile data needed by shared motion planning.
Simulation viewers and the old SAPIEN simulator are intentionally not part of
this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..scene_obstacles import AxisAlignedBoxObstacle
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


def _grasp_full_collision_pairs() -> tuple[tuple[str, str], ...]:
    """Cover the complete arms without changing the live teleop proxy contract.

    Adjacent arm meshes are structural contacts, not free-space self-collision
    pairs.  Coarse link3/hand proxy boxes may overlap their own arm by design,
    so self-collision uses the actual arm meshes, not those envelopes.  Proxy
    checks against the other arm and fixed environment remain active.
    """

    arms = {
        "left": tuple((f"arm_link{index}_0", index) for index in range(1, 7))
        + (("arm_link3_proxy_0", 3), ("left_hand_proxy_0", 6)),
        "right": tuple((f"right_arm_link{index}_0", index) for index in range(1, 7))
        + (("right_arm_link3_proxy_0", 3), ("right_hand_proxy_0", 6)),
    }
    pairs = list(DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS)
    for side, geometries in arms.items():
        pairs.extend(
            (first, second)
            for (first, first_joint), (second, second_joint) in combinations(
                geometries[:6], 2
            )
            if abs(first_joint - second_joint) > 1
        )
        pairs.extend(
            (moving, fixed)
            for moving, _joint in geometries
            for fixed in DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES
        )
        own_base, other_base = (
            ("base_link_0", "right_base_link_0")
            if side == "left"
            else ("right_base_link_0", "base_link_0")
        )
        pairs.extend((moving, own_base) for moving, joint in geometries if joint > 1)
        pairs.extend((moving, other_base) for moving, _joint in geometries)
    pairs.extend(
        (left, right)
        for left, _left_joint in arms["left"]
        for right, _right_joint in arms["right"]
    )
    unique: dict[frozenset[str], tuple[str, str]] = {}
    for pair in pairs:
        unique.setdefault(frozenset(pair), pair)
    return tuple(unique.values())


def _build_dmtac_hand_proxy_collision_pairs(
    left_hand_geometries: tuple[str, ...] = ("left_hand_proxy_0", "left_hand_proxy_1"),
    right_hand_geometries: tuple[str, ...] = ("right_hand_proxy_0", "right_hand_proxy_1"),
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    pairs.extend(("arm_link3_proxy_0", geometry) for geometry in right_hand_geometries)
    pairs.extend((geometry, "right_arm_link3_proxy_0") for geometry in left_hand_geometries)
    pairs.extend(
        (left_geometry, right_geometry)
        for left_geometry in left_hand_geometries
        for right_geometry in right_hand_geometries
    )
    pairs.extend((geometry, "camera_pole_0") for geometry in left_hand_geometries)
    pairs.extend((geometry, "camera_pole_0") for geometry in right_hand_geometries)
    pairs.extend((geometry, "table_0") for geometry in left_hand_geometries)
    pairs.extend((geometry, "table_0") for geometry in right_hand_geometries)
    pairs.extend(
        (
            ("arm_link3_proxy_0", "table_0"),
            ("right_arm_link3_proxy_0", "table_0"),
            ("arm_link3_proxy_0", "left_camera_pole_0"),
            ("right_arm_link3_proxy_0", "right_camera_pole_0"),
        )
    )
    return tuple(pairs)


DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS = (
    _build_dmtac_hand_proxy_collision_pairs()
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
    # Geometry-only overrides are applied before native collision data exists.
    # They never change the URDF frame graph or joint placements.
    fixed_scene_box_overrides: tuple[FixedSceneBoxSpec, ...] = ()


def fixed_scene_aabb_obstacles(
    profile: DualArmSceneProfile,
) -> tuple[AxisAlignedBoxObstacle, ...]:
    """Expose axis-aligned URDF table/poles to the sampled path planner."""

    if not isinstance(profile, DualArmSceneProfile):
        raise TypeError("profile must be a DualArmSceneProfile.")
    obstacles = []
    for box in profile.fixed_scene_boxes:
        if box.parent_link_name != "world":
            raise ValueError(
                f"fixed scene box {box.name!r} is not attached directly to world"
            )
        world_from_box = box.world_to_box_pose_matrix()
        if not np.allclose(world_from_box[:3, :3], np.eye(3), atol=1e-9):
            raise ValueError(
                f"fixed scene box {box.name!r} is rotated and cannot be "
                "represented by an axis-aligned obstacle"
            )
        obstacles.append(
            AxisAlignedBoxObstacle(
                obstacle_id=f"fixed-scene:{box.name}",
                center_xyz_m=tuple(float(value) for value in world_from_box[:3, 3]),
                size_xyz_m=box.box_size_xyz,
            )
        )
    return tuple(obstacles)


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


def _extract_link_visual_color_rgba(
    link_element: ET.Element,
    *,
    visual_index: int,
) -> tuple[float, float, float, float]:
    visuals = link_element.findall("visual")
    if 0 <= visual_index < len(visuals):
        color = visuals[visual_index].find("material/color")
        if color is not None:
            return _parse_float_tuple(color.get("rgba"), count=4)
    return _extract_link_color_rgba(link_element)


def _find_fixed_joint_for_link(root: ET.Element, link_name: str) -> ET.Element:
    for candidate in root.findall("joint"):
        child = candidate.find("child")
        if child is not None and child.get("link") == link_name:
            return candidate
    raise ValueError(f"Could not find fixed joint for link '{link_name}'.")


def _build_hand_proxy_specs(
    root: ET.Element,
    *,
    link_name: str,
    side: Literal["left", "right"],
    include_all_collision_boxes: bool = False,
) -> tuple[HandProxyBoxSpec, ...]:
    link_element = _find_named_xml_element(root, "link", link_name)
    if include_all_collision_boxes:
        containers = tuple(
            container
            for container in link_element.findall("collision")
            if container.find("geometry/box") is not None
        )
        if not containers:
            containers = tuple(
                container
                for container in link_element.findall("visual")
                if container.find("geometry/box") is not None
            )
    else:
        containers = ()
        for container_tag in ("collision", "visual"):
            containers = tuple(
                container
                for container in link_element.findall(container_tag)
                if container.find("geometry/box") is not None
            )
            if containers:
                containers = containers[:1]
                break
    if not containers:
        raise ValueError(f"Link '{link_name}' does not contain a box geometry.")
    joint_element = _find_fixed_joint_for_link(root, link_name)
    parent = joint_element.find("parent")
    if parent is None or parent.get("link") is None:
        raise ValueError(f"Proxy link '{link_name}' is missing a parent link in its joint.")
    joint_origin_xyz, joint_origin_rpy = _parse_origin(joint_element)
    specs = []
    for index, container in enumerate(containers):
        size = _parse_float_tuple(container.find("geometry/box").get("size"), count=3)
        box_origin_xyz, box_origin_rpy = _parse_origin(container)
        specs.append(
            HandProxyBoxSpec(
                name=(
                    link_name
                    if index == 0 and not include_all_collision_boxes
                    else f"{link_name}_{index}"
                ),
                side=side,
                geometry_name=f"{link_name}_{index}",
                parent_link_name=parent.get("link"),
                box_size_xyz=size,
                joint_origin_xyz=joint_origin_xyz,
                joint_origin_rpy=joint_origin_rpy,
                box_origin_xyz=box_origin_xyz,
                box_origin_rpy=box_origin_rpy,
                color_rgba=_extract_link_visual_color_rgba(
                    link_element,
                    visual_index=index,
                ),
            )
        )
    return tuple(specs)


def _build_hand_proxy_spec(
    root: ET.Element,
    *,
    link_name: str,
    side: Literal["left", "right"],
) -> HandProxyBoxSpec:
    return _build_hand_proxy_specs(
        root,
        link_name=link_name,
        side=side,
    )[0]


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


def _load_hand_proxy_boxes(
    urdf_path: Path,
    *,
    include_all_collision_boxes: bool = False,
) -> tuple[HandProxyBoxSpec, ...]:
    root = ET.parse(urdf_path).getroot()
    return (
        *_build_hand_proxy_specs(
            root,
            link_name="left_hand_proxy",
            side="left",
            include_all_collision_boxes=include_all_collision_boxes,
        ),
        *_build_hand_proxy_specs(
            root,
            link_name="right_hand_proxy",
            side="right",
            include_all_collision_boxes=include_all_collision_boxes,
        ),
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


def _hand_boxes_in_flange_frame(urdf_path: Path, boxes: tuple[HandProxyBoxSpec, ...]) -> tuple[HandProxyBoxSpec, ...]:
    """Preserve fixed hand-base offsets when binding observation to CAD FK."""
    root = ET.parse(urdf_path).getroot()
    result = []
    for box in boxes:
        flange = 'arm_link6' if box.side == 'left' else 'right_arm_link6'
        cursor, seen = box.parent_link_name, set()
        pose = _pose_matrix_from_xyz_rpy(box.joint_origin_xyz, box.joint_origin_rpy)
        while cursor != flange:
            if cursor in seen:
                raise ValueError('Self proxy fixed chain contains a cycle')
            seen.add(cursor)
            joint = _find_fixed_joint_for_link(root, cursor)
            xyz, rpy = _parse_origin(joint)
            pose = _pose_matrix_from_xyz_rpy(xyz, rpy) @ pose
            cursor = joint.find('parent').get('link')
        result.append(replace(box, parent_link_name=flange,
            joint_origin_xyz=tuple(pose[:3, 3]),
            joint_origin_rpy=tuple(R.from_matrix(pose[:3, :3]).as_euler('xyz'))))
    return tuple(result)


def load_dmtac_self_observation_proxy_boxes(urdf_path: str | Path) -> tuple[HandProxyBoxSpec, ...]:
    """Load explicit stepped observation geometry without selecting a motion model."""
    path = Path(urdf_path).resolve(strict=True)
    root = ET.parse(path).getroot()
    for side in ("left", "right"):
        cursor = side + "_hand_proxy"
        flange = "arm_link6" if side == "left" else "right_arm_link6"
        seen = set()
        while cursor != flange:
            if cursor in seen:
                raise ValueError("Self proxy fixed chain contains a cycle")
            seen.add(cursor)
            joints = [joint for joint in root.findall("joint")
                if joint.find("child") is not None
                and joint.find("child").get("link") == cursor]
            if len(joints) != 1 or joints[0].get("type") != "fixed":
                raise ValueError("Self proxy requires a unique fixed chain to its flange")
            parent = joints[0].find("parent")
            if parent is None or not parent.get("link"):
                raise ValueError("Self proxy fixed chain requires a parent link")
            cursor = parent.get("link")
    boxes = _load_hand_proxy_boxes(path, include_all_collision_boxes=True)
    expected = tuple(f"{side}_hand_proxy_{index}"
        for side in ("left", "right") for index in range(3))
    if tuple(box.geometry_name for box in boxes) != expected:
        raise ValueError("Self observation requires core and two stepped terminal boxes per hand")
    return _hand_boxes_in_flange_frame(path, tuple(
        box for box in boxes if not box.geometry_name.endswith("_0")
    ))


def build_dual_arm_scene_profile(
    name: str = DEFAULT_COLLISION_SCENE, *, dmtac_proxy_urdf_path: Path | None = None,
) -> DualArmSceneProfile:
    """Build a pure RM65 dual-arm collision/model profile."""

    name = str(name or DEFAULT_COLLISION_SCENE)
    if dmtac_proxy_urdf_path is not None and name not in {
        "dual_arm_hand_proxy_dmtac_strict", "dual_arm_hand_proxy_dmtac_grasp"
    }:
        raise ValueError("A DMTAC proxy candidate requires a DMTAC scene profile.")
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

    if name in {
        "dual_arm_hand_proxy_open",
        "dual_arm_hand_proxy_strict",
        "dual_arm_hand_proxy_grasp_full",
        "dual_arm_hand_proxy_dmtac_strict",
        "dual_arm_hand_proxy_dmtac_grasp",
    }:
        is_dmtac_profile = name in {
            "dual_arm_hand_proxy_dmtac_strict", "dual_arm_hand_proxy_dmtac_grasp"
        }
        if is_dmtac_profile:
            urdf_name = "rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
        elif name == "dual_arm_hand_proxy_open":
            urdf_name = "rm65_dual_arm_hand_proxy_open.urdf"
        else:
            urdf_name = "rm65_dual_arm_hand_proxy_strict.urdf"
        urdf_path = (Path(dmtac_proxy_urdf_path).resolve(strict=True)
            if dmtac_proxy_urdf_path is not None else asset_dir / urdf_name)
        hand_proxy_boxes = _load_hand_proxy_boxes(
            urdf_path,
            include_all_collision_boxes=is_dmtac_profile,
        )
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
        if is_dmtac_profile:
            count = len(hand_proxy_boxes) // 2
            allowed_counts = (2, 3) if dmtac_proxy_urdf_path is not None else (2,)
            expected = tuple(f"{side}_hand_proxy_{index}"
                for side in ("left", "right") for index in range(count))
            if count not in allowed_counts or tuple(
                box.geometry_name for box in hand_proxy_boxes
            ) != expected:
                raise ValueError("DMTAC proxy must expose matching sequential boxes per hand; "
                    "three boxes require an explicit candidate path.")
        fixed_scene_box_overrides = ()
        if name == "dual_arm_hand_proxy_dmtac_grasp":
            cad_table = _load_fixed_scene_box(
                asset_dir.parent / "realman_dual_arm_inspire_gripper_dmtacm"
                / "realman_dual_arm_inspire_gripper_dmtacm.urdf",
                link_name="table",
            )
            strict_table = fixed_scene_boxes[-1]
            if (
                cad_table.parent_link_name != "world"
                or cad_table.joint_origin_xyz != strict_table.joint_origin_xyz
                or cad_table.joint_origin_rpy != strict_table.joint_origin_rpy
                or cad_table.box_size_xyz != strict_table.box_size_xyz
            ):
                raise ValueError(
                    "DMTAC grasp table must retain the strict table frame and size; "
                    "only its reviewed CAD collision placement may differ."
                )
            fixed_scene_boxes = (*fixed_scene_boxes[:-1], cad_table)
            fixed_scene_box_overrides = (cad_table,)
        return DualArmSceneProfile(
            name=name,
            urdf_path=urdf_path,
            package_dirs=(asset_dir,),
            left_frame_name="arm_link6",
            right_frame_name="right_arm_link6",
            collision_pairs=(
                _build_dmtac_hand_proxy_collision_pairs(
                    tuple(box.geometry_name for box in hand_proxy_boxes if box.side == "left"),
                    tuple(box.geometry_name for box in hand_proxy_boxes if box.side == "right"),
                )
                if is_dmtac_profile
                else _grasp_full_collision_pairs()
                if name == "dual_arm_hand_proxy_grasp_full"
                else DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS
            ),
            hand_proxy_boxes=(*hand_proxy_boxes, *link3_proxy_boxes),
            fixed_scene_boxes=fixed_scene_boxes,
            left_pose_transform=_POSE_TRANSFORM_LEFT.copy(),
            right_pose_transform=_POSE_TRANSFORM_RIGHT.copy(),
            solution_layout="auto",
            fixed_scene_box_overrides=fixed_scene_box_overrides,
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
    *, dmtac_proxy_urdf_path: Path | None = None,
) -> DualArmSceneProfile:
    return build_dual_arm_scene_profile(name, dmtac_proxy_urdf_path=dmtac_proxy_urdf_path)


__all__ = [
    "load_dmtac_self_observation_proxy_boxes",
    "DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS",
    "DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS",
    "DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES",
    "DualArmSceneProfile",
    "FixedSceneBoxSpec",
    "HandProxyBoxSpec",
    "build_dual_arm_scene_profile",
    "build_rm65_dual_arm_scene_profile",
    "fixed_scene_aabb_obstacles",
]
