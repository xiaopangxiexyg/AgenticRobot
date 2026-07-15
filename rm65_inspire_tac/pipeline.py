from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from .ik import SingleArmIKSolver
from .multitask_ik import MultiTaskIKSolver
from .paths import require_asset_path
from .schemas import (
    ArmExecutionResult,
    BimanualExecutionResult,
    BimanualRetargetingResult,
)

DEFAULT_LEFT_ARM_QPOS = np.array([1.57, -0.47, 1.94, 0.0, 0.70, -1.57], dtype=float)
DEFAULT_RIGHT_ARM_QPOS = np.array([-1.57, -0.47, 1.94, 3.10, -0.70, -1.57], dtype=float)

# Explicit collision-pair set for both hand-proxy profiles. Keep this as the
# set of pairs to check rather than deriving it from broad link cross-products:
# adding or removing a checked pair is a safety-policy change that needs an
# accompanying scene review, documentation update, and golden-test update.
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
# Fixed geometries participating in the configured collision checks.
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


def _parse_origin(element: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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


def _build_hand_proxy_spec(
    root: ET.Element,
    *,
    link_name: str,
    side: Literal["left", "right"],
) -> HandProxyBoxSpec:
    link_element = _find_named_xml_element(root, "link", link_name)
    size, box_origin_xyz, box_origin_rpy = _extract_link_box_geometry(link_element)
    joint_element = None
    for candidate in root.findall("joint"):
        child = candidate.find("child")
        if child is not None and child.get("link") == link_name:
            joint_element = candidate
            break
    if joint_element is None:
        raise ValueError(f"Could not find fixed joint for proxy link '{link_name}'.")
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
    joint_element = None
    for candidate in root.findall("joint"):
        child = candidate.find("child")
        if child is not None and child.get("link") == link_name:
            joint_element = candidate
            break
    if joint_element is None:
        raise ValueError(f"Could not find fixed joint for scene box link '{link_name}'.")
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


def build_dual_arm_scene_profile(name: str) -> DualArmSceneProfile:
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
            left_pose_transform=_POSE_TRANSFORM_LEFT,
            right_pose_transform=_POSE_TRANSFORM_RIGHT,
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
        profile_proxy_boxes = (*hand_proxy_boxes, *link3_proxy_boxes)
        camera_pole_box = _load_fixed_scene_box(urdf_path, link_name="camera_pole")
        left_camera_pole_box = _load_fixed_scene_box(urdf_path, link_name="left_camera_pole")
        right_camera_pole_box = _load_fixed_scene_box(urdf_path, link_name="right_camera_pole")
        table_box = _load_fixed_scene_box(urdf_path, link_name="table")
        fixed_scene_boxes = (
            camera_pole_box,
            left_camera_pole_box,
            right_camera_pole_box,
            table_box,
        )
        if tuple(box.geometry_name for box in fixed_scene_boxes) != (
            DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES
        ):
            raise ValueError(
                "Hand-proxy collision scene URDF no longer exposes the required collision "
                "fixed geometry set "
                f"{DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES}."
            )
        return DualArmSceneProfile(
            name=name,
            urdf_path=urdf_path,
            package_dirs=(asset_dir,),
            left_frame_name="arm_link6",
            right_frame_name="right_arm_link6",
            collision_pairs=DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
            hand_proxy_boxes=profile_proxy_boxes,
            fixed_scene_boxes=fixed_scene_boxes,
            left_pose_transform=_POSE_TRANSFORM_LEFT,
            right_pose_transform=_POSE_TRANSFORM_RIGHT,
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
            left_pose_transform=_POSE_TRANSFORM_LEFT,
            right_pose_transform=_POSE_TRANSFORM_RIGHT,
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
            left_pose_transform=_POSE_TRANSFORM_LEFT,
            right_pose_transform=_POSE_TRANSFORM_RIGHT,
            solution_layout="auto",
        )
    raise ValueError(f"Unknown dual arm scene profile: {name}")


class IndependentArmSimulator:
    def __init__(
        self,
        arm_urdf_path: str | Path | None = None,
        eef_name: str = "hand_base",
        arm_max_ang_vel: float = 4.0,
        low_control_dt: float = 0.02,
        policy_control_dt: float = 0.1,
        left_initial_qpos: np.ndarray | None = None,
        right_initial_qpos: np.ndarray | None = None,
        renderer: Any | None = None,
        sleep: bool = False,
    ):
        self.arm_solver = SingleArmIKSolver(urdf_path=arm_urdf_path, eef_name=eef_name)
        self.left_arm_qpos = (
            DEFAULT_LEFT_ARM_QPOS.copy()
            if left_initial_qpos is None
            else np.asarray(left_initial_qpos, dtype=float)
        )
        self.right_arm_qpos = (
            DEFAULT_RIGHT_ARM_QPOS.copy()
            if right_initial_qpos is None
            else np.asarray(right_initial_qpos, dtype=float)
        )
        self.max_arm_step_size = low_control_dt * arm_max_ang_vel
        self.low_control_dt = low_control_dt
        self.policy_control_dt = policy_control_dt
        self.decimation = max(int(round(policy_control_dt / low_control_dt)), 1)
        self.renderer = renderer
        self.sleep = sleep

    def step(self, teleop: BimanualRetargetingResult) -> BimanualExecutionResult:
        left_result = teleop.left
        right_result = teleop.right

        for _ in range(self.decimation):
            start_time = time.time()
            if left_result is not None:
                self.left_arm_qpos = self.arm_solver.solve_step(
                    left_result.wrist_pose_robot,
                    self.left_arm_qpos,
                    max_joint_step=self.max_arm_step_size,
                )
            if right_result is not None:
                self.right_arm_qpos = self.arm_solver.solve_step(
                    right_result.wrist_pose_robot,
                    self.right_arm_qpos,
                    max_joint_step=self.max_arm_step_size,
                )
            if self.sleep:
                remaining = self.low_control_dt - (time.time() - start_time)
                if remaining > 0:
                    time.sleep(remaining)

        rendered_rgb = None
        if self.renderer is not None:
            rendered_rgb = self.renderer.render_data(
                self.left_arm_qpos if left_result is not None else None,
                left_result.retargeted_hand_qpos if left_result is not None else None,
                self.right_arm_qpos if right_result is not None else None,
                right_result.retargeted_hand_qpos if right_result is not None else None,
            )

        return BimanualExecutionResult(
            left=None
            if left_result is None
            else ArmExecutionResult(
                side="left",
                arm_qpos=self.left_arm_qpos.copy(),
                wrist_pose_robot=left_result.wrist_pose_robot.copy(),
                retargeted_hand_qpos=left_result.retargeted_hand_qpos.copy(),
                hardware_hand_qpos=left_result.hardware_hand_qpos.copy(),
            ),
            right=None
            if right_result is None
            else ArmExecutionResult(
                side="right",
                arm_qpos=self.right_arm_qpos.copy(),
                wrist_pose_robot=right_result.wrist_pose_robot.copy(),
                retargeted_hand_qpos=right_result.retargeted_hand_qpos.copy(),
                hardware_hand_qpos=right_result.hardware_hand_qpos.copy(),
            ),
            rendered_rgb=rendered_rgb,
            timestamp=teleop.timestamp,
            metadata={"controller": "independent_arm_simulator"},
        )


class DualArmMultiTaskSimulator:
    def __init__(
        self,
        scene_profile: str | DualArmSceneProfile = "dual_arm",
        solver: str | None = None,
        frequency: float = 7.0,
        low_control_dt: float = 0.02,
        policy_control_dt: float = 0.1,
        left_initial_qpos: np.ndarray | None = None,
        right_initial_qpos: np.ndarray | None = None,
        renderer: Any | None = None,
        debug_viz: bool = False,
        sleep: bool = False,
    ):
        self.profile = (
            build_dual_arm_scene_profile(scene_profile)
            if isinstance(scene_profile, str)
            else scene_profile
        )
        self.left_arm_qpos = (
            DEFAULT_LEFT_ARM_QPOS.copy()
            if left_initial_qpos is None
            else np.asarray(left_initial_qpos, dtype=float)
        )
        self.right_arm_qpos = (
            DEFAULT_RIGHT_ARM_QPOS.copy()
            if right_initial_qpos is None
            else np.asarray(right_initial_qpos, dtype=float)
        )
        self.low_control_dt = low_control_dt
        self.policy_control_dt = policy_control_dt
        self.decimation = max(int(round(policy_control_dt / low_control_dt)), 1)
        self.sleep = sleep
        self.renderer = renderer

        self.ik_solver = MultiTaskIKSolver(
            robot_urdf_path=self.profile.urdf_path,
            robot_package_dirs=list(self.profile.package_dirs),
            robot_root_joint=None,
            robot_list_of_joints_to_lock=None,
            robot_collision_pair_names=list(self.profile.collision_pairs),
            solver=solver,
            frequency=frequency,
            debug_viz=debug_viz,
        )
        self.ik_solver.add_frame_task(self.profile.left_frame_name)
        self.ik_solver.add_frame_task(self.profile.right_frame_name)
        self.ik_solver.add_posture_task(cost=1e-1)

    def step(self, teleop: BimanualRetargetingResult) -> BimanualExecutionResult:
        if teleop.left is None or teleop.right is None:
            raise ValueError("DualArmMultiTaskSimulator requires both left and right teleop inputs.")

        for _ in range(self.decimation):
            start_time = time.time()
            left_pose = self.profile.left_pose_transform @ teleop.left.wrist_pose_robot
            right_pose = self.profile.right_pose_transform @ teleop.right.wrist_pose_robot
            qpos_both = self.ik_solver.solve(
                targets=[self._pose_to_target(left_pose), self._pose_to_target(right_pose)]
            )
            self.left_arm_qpos, self.right_arm_qpos = self._split_solution(qpos_both)
            if self.sleep:
                remaining = self.low_control_dt - (time.time() - start_time)
                if remaining > 0:
                    time.sleep(remaining)

        rendered_rgb = None
        if self.renderer is not None:
            rendered_rgb = self.renderer.render_data(
                self.left_arm_qpos,
                teleop.left.retargeted_hand_qpos,
                self.right_arm_qpos,
                teleop.right.retargeted_hand_qpos,
            )

        return BimanualExecutionResult(
            left=ArmExecutionResult(
                side="left",
                arm_qpos=self.left_arm_qpos.copy(),
                wrist_pose_robot=teleop.left.wrist_pose_robot.copy(),
                retargeted_hand_qpos=teleop.left.retargeted_hand_qpos.copy(),
                hardware_hand_qpos=teleop.left.hardware_hand_qpos.copy(),
            ),
            right=ArmExecutionResult(
                side="right",
                arm_qpos=self.right_arm_qpos.copy(),
                wrist_pose_robot=teleop.right.wrist_pose_robot.copy(),
                retargeted_hand_qpos=teleop.right.retargeted_hand_qpos.copy(),
                hardware_hand_qpos=teleop.right.hardware_hand_qpos.copy(),
            ),
            rendered_rgb=rendered_rgb,
            timestamp=teleop.timestamp,
            metadata={
                "controller": "dual_arm_multitask_simulator",
                "scene_profile": self.profile.name,
            },
        )

    def _pose_to_target(self, pose: np.ndarray) -> np.ndarray:
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        return np.array(
            [pose[0, 3], pose[1, 3], pose[2, 3], quat[0], quat[1], quat[2], quat[3]],
            dtype=float,
        )

    def _split_solution(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float)
        if self.profile.solution_layout == "dual_arm_hand" and qpos.shape[0] >= 24:
            return qpos[0:6], qpos[18:24]
        if qpos.shape[0] == 14:
            return qpos[1:7], qpos[8:14]
        if qpos.shape[0] == 12:
            return qpos[0:6], qpos[6:12]
        raise ValueError(f"Unsupported dual-arm IK solution shape: {qpos.shape}")
