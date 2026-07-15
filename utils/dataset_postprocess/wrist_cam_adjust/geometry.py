from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SIDES = ("left", "right")


def _floats(text: str | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return tuple(float(value) for value in default)
    return tuple(float(part) for part in str(text).replace(",", " ").split())


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def pose_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rpy_to_matrix(rpy)
    pose[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return pose


def invert_pose(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("joint axis norm must be positive")
    x, y, z = vector / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    c1 = 1.0 - c
    return np.asarray(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def joint_motion(joint_type: str, axis: Sequence[float], value: float) -> np.ndarray:
    motion = np.eye(4, dtype=float)
    if joint_type in {"fixed", ""}:
        return motion
    if joint_type in {"revolute", "continuous"}:
        motion[:3, :3] = axis_angle_matrix(axis, value)
        return motion
    if joint_type == "prismatic":
        vector = np.asarray(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("joint axis norm must be positive")
        motion[:3, 3] = vector / norm * float(value)
        return motion
    raise ValueError(f"unsupported URDF joint type for FK: {joint_type!r}")


def pose_from_xyz_quat_xyzw(values: Sequence[float]) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = np.asarray(values, dtype=float).reshape(7)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    pose[:3, :3] = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    return pose


def pose6_xyz_rpy_to_matrix(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(6)
    return pose_from_xyz_rpy(array[:3], array[3:])


def pose7_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        return None
    return pose_from_xyz_quat_xyzw(array)


def vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.shape != (int(length),) or not np.all(np.isfinite(array)):
        return None
    return array.astype(float, copy=True)


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class BoxVisual:
    link_name: str
    origin: np.ndarray
    size: tuple[float, float, float]


@dataclass
class SimpleURDF:
    path: Path
    links: set[str]
    joints: list[Joint]
    boxes_by_link: dict[str, list[BoxVisual]]

    @property
    def joint_by_child(self) -> dict[str, Joint]:
        return {joint.child: joint for joint in self.joints}

    def transform_between(
        self,
        parent_link: str,
        child_link: str,
        *,
        joint_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        if parent_link == child_link:
            return np.eye(4, dtype=float)
        joint_by_child = self.joint_by_child
        chain: list[Joint] = []
        current = str(child_link)
        while current != parent_link:
            joint = joint_by_child.get(current)
            if joint is None:
                raise KeyError(f"no URDF chain from {parent_link!r} to {child_link!r}")
            chain.append(joint)
            current = joint.parent
        transform = np.eye(4, dtype=float)
        values = joint_values or {}
        for joint in reversed(chain):
            value = float(values.get(joint.name, 0.0))
            transform = transform @ joint.origin @ joint_motion(joint.joint_type, joint.axis, value)
        return transform

    def first_box_visual(self, link_name: str) -> BoxVisual:
        boxes = self.boxes_by_link.get(link_name, [])
        if not boxes:
            raise KeyError(f"link {link_name!r} has no box visual in {self.path}")
        return boxes[0]


def _parse_origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=float)
    xyz = _floats(element.attrib.get("xyz"), default=(0.0, 0.0, 0.0))
    rpy = _floats(element.attrib.get("rpy"), default=(0.0, 0.0, 0.0))
    return pose_from_xyz_rpy(xyz, rpy)


def parse_urdf(path: str | Path) -> SimpleURDF:
    urdf_path = Path(path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"expected URDF root tag 'robot', got {root.tag!r}")
    links = {element.attrib["name"] for element in root.findall("link")}
    boxes_by_link: dict[str, list[BoxVisual]] = {}
    for link in root.findall("link"):
        link_name = str(link.attrib["name"])
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            box = None if geometry is None else geometry.find("box")
            if box is None:
                continue
            size = _floats(box.attrib.get("size"), default=(1.0, 1.0, 1.0))
            if len(size) != 3:
                raise ValueError(f"box visual for {link_name!r} must have 3 size values")
            boxes_by_link.setdefault(link_name, []).append(
                BoxVisual(
                    link_name=link_name,
                    origin=_parse_origin(visual.find("origin")),
                    size=(float(size[0]), float(size[1]), float(size[2])),
                )
            )

    joints: list[Joint] = []
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {element.attrib.get('name')} is missing parent/child")
        axis = element.find("axis")
        joints.append(
            Joint(
                name=str(element.attrib["name"]),
                joint_type=str(element.attrib.get("type", "fixed")),
                parent=str(parent.attrib["link"]),
                child=str(child.attrib["link"]),
                origin=_parse_origin(element.find("origin")),
                axis=tuple(_floats(None if axis is None else axis.attrib.get("xyz"), default=(0.0, 0.0, 1.0))),  # type: ignore[arg-type]
            )
        )
    return SimpleURDF(path=urdf_path, links=links, joints=joints, boxes_by_link=boxes_by_link)


@dataclass(frozen=True)
class SideGeometry:
    side: str
    base_link: str
    end_link: str
    camera_link: str
    joint_names: tuple[str, ...]
    T_world_base: np.ndarray
    T_base_world: np.ndarray
    T_end_camera: np.ndarray
    T_camera_end: np.ndarray

    def camera_from_base_end_pose(self, T_base_end: np.ndarray) -> np.ndarray:
        return np.asarray(T_base_end, dtype=float).reshape(4, 4) @ self.T_end_camera

    def camera_from_base_tool_pose(self, T_base_tool: np.ndarray, T_end_tool: np.ndarray) -> np.ndarray:
        T_base_end = np.asarray(T_base_tool, dtype=float).reshape(4, 4) @ invert_pose(T_end_tool)
        return self.camera_from_base_end_pose(T_base_end)

    def world_from_base_pose(self, T_base_frame: np.ndarray) -> np.ndarray:
        return self.T_world_base @ np.asarray(T_base_frame, dtype=float).reshape(4, 4)


@dataclass(frozen=True)
class TableGeometry:
    link_name: str
    T_world_visual: np.ndarray
    T_visual_world: np.ndarray
    size: tuple[float, float, float]

    @property
    def top_z_visual(self) -> float:
        return float(self.size[2]) * 0.5


@dataclass(frozen=True)
class SceneGeometry:
    urdf_path: Path
    urdf: SimpleURDF
    table: TableGeometry
    sides: dict[str, SideGeometry]


def load_scene_geometry(
    urdf_path: str | Path,
    *,
    table_link: str = "table",
    world_link: str = "world",
    base_links: Mapping[str, str] | None = None,
    end_links: Mapping[str, str] | None = None,
    camera_links: Mapping[str, str] | None = None,
) -> SceneGeometry:
    urdf = parse_urdf(urdf_path)
    base_links = dict(base_links or {})
    end_links = dict(end_links or {})
    camera_links = dict(camera_links or {})

    table_visual = urdf.first_box_visual(table_link)
    T_world_table = urdf.transform_between(world_link, table_link)
    T_world_visual = T_world_table @ table_visual.origin
    table = TableGeometry(
        link_name=table_link,
        T_world_visual=T_world_visual,
        T_visual_world=invert_pose(T_world_visual),
        size=table_visual.size,
    )

    sides: dict[str, SideGeometry] = {}
    for side in SIDES:
        base_link = base_links.get(side, f"{side}_arm_base_link")
        end_link = end_links.get(side, f"{side}_arm_link6")
        camera_link = camera_links.get(side, f"{side}_camera_rgb")
        joint_names = tuple(f"{side}_arm_joint{index}" for index in range(1, 7))
        T_world_base = urdf.transform_between(world_link, base_link)
        T_end_camera = urdf.transform_between(end_link, camera_link)
        sides[side] = SideGeometry(
            side=side,
            base_link=base_link,
            end_link=end_link,
            camera_link=camera_link,
            joint_names=joint_names,
            T_world_base=T_world_base,
            T_base_world=invert_pose(T_world_base),
            T_end_camera=T_end_camera,
            T_camera_end=invert_pose(T_end_camera),
        )
    return SceneGeometry(
        urdf_path=urdf.path,
        urdf=urdf,
        table=table,
        sides=sides,
    )


def tool_transform_from_profile(profile_path: str | Path) -> np.ndarray:
    import json

    path = Path(profile_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"tool profile does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tcp = payload.get("tcp") if isinstance(payload, Mapping) else None
    pose = tcp.get("pose_flange_m_rad") if isinstance(tcp, Mapping) else None
    if pose is None:
        raise ValueError(f"tool profile is missing tcp.pose_flange_m_rad: {path}")
    return pose6_xyz_rpy_to_matrix(pose)


def resolve_repo_relative_path(value: str | Path | None, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = (repo_root / path).resolve()
        if candidate.exists():
            return candidate
    # Historical metadata can contain repository-relative paths captured before
    # package moves. These aliases are data migration support only; runtime code
    # must import the canonical ``utils.dataset_postprocess`` package.
    marker_roots = {
        "hardware/": "hardware/",
        "teleop_records/": "teleop_records/",
        "dataset_postprocess/": "utils/dataset_postprocess/",
        "utils/dataset_postprocess/": "utils/dataset_postprocess/",
        "cameras/": "cameras/",
        "camera_process/configs/": "cameras/configs/",
        "camera_process/real_undistort/": "utils/dataset_postprocess/real_undistort/",
        "camera_process/wrist_cam_adjust/": "utils/dataset_postprocess/wrist_cam_adjust/",
        "assets/": "assets/",
        "utils/": "utils/",
    }
    for marker, replacement in marker_roots.items():
        index = text.find(marker)
        if index >= 0:
            suffix = text[index + len(marker) :]
            candidate = (repo_root / replacement / suffix).resolve()
            if candidate.exists():
                return candidate
    return path
