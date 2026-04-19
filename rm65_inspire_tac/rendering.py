from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_runtime.scene_layout import (
    rm65_base_y_for_side,
    rm65_base_yaw_for_side,
    rm65_viewer_camera_pose,
)

from .paths import require_asset_path

DEFAULT_ARM_JOINT_NAMES = (
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
)


@dataclass(frozen=True)
class DebugBoxTemplate:
    size_xyz: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class DebugBoxOverlay:
    pose_matrix: np.ndarray
    size_xyz: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]

    def template(self) -> DebugBoxTemplate:
        return DebugBoxTemplate(
            size_xyz=self.size_xyz,
            color_rgba=self.color_rgba,
        )


class SapienArmHandRenderer:
    def __init__(
        self,
        left_urdf_path: str | Path | None,
        left_retargeting_joint_names: Sequence[str] | None,
        right_urdf_path: str | Path | None,
        right_retargeting_joint_names: Sequence[str] | None,
        arm_joint_names: Sequence[str] = DEFAULT_ARM_JOINT_NAMES,
        left_right_base_distance: float = 0.525,
        viewer_resolution: tuple[int, int] = (1600, 900),
        camera_resolution: tuple[int, int] = (1280, 720),
        show_viewer: bool = True,
        show_desk_plane: bool = True,
        debug_frame_axis_length: float = 0.08,
        debug_frame_axis_thickness: float = 0.005,
    ):
        try:
            import sapien
            from sapien.asset import create_dome_envmap
            from sapien.utils import Viewer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SapienArmHandRenderer requires sapien to be importable."
            ) from exc

        self.sapien = sapien
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)
        self.scene.set_environment_map(
            create_dome_envmap(
                sky_color=[0.2, 0.2, 0.2],
                ground_color=[0.2, 0.2, 0.2],
            )
        )
        self.scene.add_directional_light(-np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
        self.scene.add_directional_light(-np.array([0, 0, -1]), np.array([1.8, 1.6, 1.6]), shadow=False)
        self.scene.add_directional_light(np.array([0, 0, -2]), -np.array([1.8, 1.6, 1.6]), shadow=False)
        self.scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self._debug_frame_axis_length = float(debug_frame_axis_length)
        self._debug_frame_axis_thickness = float(debug_frame_axis_thickness)
        self._debug_frame_actors: dict[str, Any] = {}
        self._debug_box_actors: dict[str, Any] = {}
        self._debug_box_templates: dict[str, DebugBoxTemplate] = {}
        self._hidden_pose = None

        if show_desk_plane:
            desk_builder = self.scene.create_actor_builder()
            desk_builder.add_box_visual(
                pose=sapien.Pose(p=[0.0, 0.0, -0.005]),
                half_size=[0.45, 0.65, 0.005],
                material=sapien.render.RenderMaterial(base_color=[0.82, 0.82, 0.84, 1.0]),
                name="desk_plane",
            )
            self.desk_plane = desk_builder.build_kinematic(name="desk_plane")
        else:
            self.desk_plane = None

        self.left_robot = None
        self.right_robot = None
        if left_urdf_path is not None and left_retargeting_joint_names is not None:
            self.left_robot, self.left_retargeting_to_sapien = self._load_robot(
                loader=loader,
                urdf_path=str(left_urdf_path),
                base_position=[0.0, rm65_base_y_for_side("left", left_right_base_distance), 0.0],
                base_euler=[0.0, 0.0, rm65_base_yaw_for_side("left")],
                arm_joint_names=arm_joint_names,
                hand_joint_names=left_retargeting_joint_names,
            )
        if right_urdf_path is not None and right_retargeting_joint_names is not None:
            self.right_robot, self.right_retargeting_to_sapien = self._load_robot(
                loader=loader,
                urdf_path=str(right_urdf_path),
                base_position=[0.0, rm65_base_y_for_side("right", left_right_base_distance), 0.0],
                base_euler=[0.0, 0.0, rm65_base_yaw_for_side("right")],
                arm_joint_names=arm_joint_names,
                hand_joint_names=right_retargeting_joint_names,
            )

        self.viewer = None
        if show_viewer:
            self.viewer = Viewer(resolutions=viewer_resolution)
            self.viewer.set_scene(self.scene)
            # View from desk +X looking toward the origin so logical left_arm(-Y)
            # appears on the screen's left side in every teleop_runtime viewer.
            camera_pose = rm65_viewer_camera_pose()
            self.viewer.set_camera_xyz(*camera_pose.xyz)
            self.viewer.set_camera_rpy(*camera_pose.rpy)

        width, height = camera_resolution
        self.camera = self.scene.add_camera(
            name="portable_renderer_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(90.0),
            near=0.1,
            far=100.0,
        )
        camera_quat = R.from_euler("xyz", [3.14, 1.57, 3.14]).as_quat()
        self.camera.set_pose(
            sapien.Pose(
                p=[0.0, 0.0, 0.0],
                q=[camera_quat[3], camera_quat[0], camera_quat[1], camera_quat[2]],
            )
        )
        self._hidden_pose = self.sapien.Pose(p=[1000.0, 1000.0, 1000.0])

    def _matrix_to_pose(self, pose_matrix: np.ndarray):
        pose_matrix = np.asarray(pose_matrix, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(pose_matrix)):
            return self._hidden_pose
        rotation = pose_matrix[:3, :3]
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        quat_xyzw = R.from_matrix(rotation).as_quat()
        return self.sapien.Pose(
            p=pose_matrix[:3, 3].tolist(),
            q=[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        )

    def _build_debug_frame_actor(self, name: str):
        axis_length = self._debug_frame_axis_length
        axis_thickness = self._debug_frame_axis_thickness
        origin_radius = axis_thickness * 1.8
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(
            pose=self.sapien.Pose(),
            radius=origin_radius,
            material=self.sapien.render.RenderMaterial(base_color=[0.95, 0.95, 0.95, 1.0]),
            name=f"{name}_origin",
        )
        builder.add_box_visual(
            pose=self.sapien.Pose(p=[axis_length * 0.5, 0.0, 0.0]),
            half_size=[axis_length * 0.5, axis_thickness, axis_thickness],
            material=self.sapien.render.RenderMaterial(base_color=[0.95, 0.2, 0.2, 1.0]),
            name=f"{name}_x",
        )
        builder.add_box_visual(
            pose=self.sapien.Pose(p=[0.0, axis_length * 0.5, 0.0]),
            half_size=[axis_thickness, axis_length * 0.5, axis_thickness],
            material=self.sapien.render.RenderMaterial(base_color=[0.2, 0.95, 0.2, 1.0]),
            name=f"{name}_y",
        )
        builder.add_box_visual(
            pose=self.sapien.Pose(p=[0.0, 0.0, axis_length * 0.5]),
            half_size=[axis_thickness, axis_thickness, axis_length * 0.5],
            material=self.sapien.render.RenderMaterial(base_color=[0.2, 0.4, 0.95, 1.0]),
            name=f"{name}_z",
        )
        return builder.build_kinematic(name=name)

    def _build_debug_box_actor(self, name: str, template: DebugBoxTemplate):
        size = np.asarray(template.size_xyz, dtype=float).reshape(3)
        half_size = (size * 0.5).tolist()
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            pose=self.sapien.Pose(),
            half_size=half_size,
            material=self.sapien.render.RenderMaterial(base_color=list(template.color_rgba)),
            name=name,
        )
        return builder.build_kinematic(name=name)

    def _ensure_debug_box_actor(self, name: str, template: DebugBoxTemplate):
        existing = self._debug_box_templates.get(name)
        if existing is not None:
            if existing != template:
                raise ValueError(
                    f"Debug box '{name}' was already reserved with a different size/color."
                )
            return self._debug_box_actors[name]
        actor = self._build_debug_box_actor(name, template)
        actor.set_pose(self._hidden_pose)
        self._debug_box_actors[name] = actor
        self._debug_box_templates[name] = template
        return actor

    def reserve_debug_frames(self, names: Sequence[str]) -> None:
        for name in names:
            if name in self._debug_frame_actors:
                continue
            actor = self._build_debug_frame_actor(name)
            actor.set_pose(self._hidden_pose)
            self._debug_frame_actors[name] = actor

    def reserve_debug_boxes(self, boxes: Mapping[str, DebugBoxTemplate]) -> None:
        for name, template in boxes.items():
            self._ensure_debug_box_actor(name, template)

    def set_debug_frames(self, frames: Mapping[str, np.ndarray] | None) -> None:
        if frames is None:
            frames = {}
        for name, pose_matrix in frames.items():
            actor = self._debug_frame_actors.get(name)
            if actor is None:
                actor = self._build_debug_frame_actor(name)
                self._debug_frame_actors[name] = actor
            actor.set_pose(self._matrix_to_pose(np.asarray(pose_matrix, dtype=float)))

        for name, actor in self._debug_frame_actors.items():
            if name not in frames:
                actor.set_pose(self._hidden_pose)

    def set_debug_boxes(self, boxes: Mapping[str, DebugBoxOverlay] | None) -> None:
        if boxes is None:
            boxes = {}
        for name, overlay in boxes.items():
            actor = self._ensure_debug_box_actor(name, overlay.template())
            actor.set_pose(self._matrix_to_pose(np.asarray(overlay.pose_matrix, dtype=float)))

        for name, actor in self._debug_box_actors.items():
            if name not in boxes:
                actor.set_pose(self._hidden_pose)

    @classmethod
    def from_retargeter(
        cls,
        retargeter: Any,
        left_urdf_path: str | Path | None = None,
        right_urdf_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "SapienArmHandRenderer":
        if left_urdf_path is None and retargeter.use_left_hand:
            left_urdf_path = require_asset_path("rm65_inspire_tac_left.urdf")
        if right_urdf_path is None and retargeter.use_right_hand:
            right_urdf_path = require_asset_path("rm65_inspire_tac_right.urdf")
        return cls(
            left_urdf_path=left_urdf_path if retargeter.use_left_hand else None,
            left_retargeting_joint_names=retargeter.left_hand_joint_names,
            right_urdf_path=right_urdf_path if retargeter.use_right_hand else None,
            right_retargeting_joint_names=retargeter.right_hand_joint_names,
            **kwargs,
        )

    def _load_robot(
        self,
        loader: Any,
        urdf_path: str,
        base_position: list[float],
        base_euler: list[float],
        arm_joint_names: Sequence[str],
        hand_joint_names: Sequence[str],
    ) -> tuple[Any, np.ndarray]:
        robot = loader.load(urdf_path)
        quat_xyzw = R.from_euler("xyz", base_euler).as_quat()
        robot.set_pose(
            self.sapien.Pose(
                base_position,
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            )
        )
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        retargeting_joint_names = list(arm_joint_names) + list(hand_joint_names)
        index_map = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names], dtype=int
        )
        return robot, index_map

    def render_data(
        self,
        left_qpos_arm: np.ndarray | None,
        left_qpos_hand: np.ndarray | None,
        right_qpos_arm: np.ndarray | None,
        right_qpos_hand: np.ndarray | None,
        *,
        capture_image: bool = True,
    ) -> np.ndarray | None:
        if self.left_robot is not None and left_qpos_arm is not None and left_qpos_hand is not None:
            qpos = np.concatenate([left_qpos_arm, left_qpos_hand])
            self.left_robot.set_qpos(qpos[self.left_retargeting_to_sapien])
        if self.right_robot is not None and right_qpos_arm is not None and right_qpos_hand is not None:
            qpos = np.concatenate([right_qpos_arm, right_qpos_hand])
            self.right_robot.set_qpos(qpos[self.right_retargeting_to_sapien])

        self.scene.update_render()
        if self.viewer is not None:
            self.viewer.render()
        if not capture_image:
            return None
        self.camera.take_picture()
        image = self.camera.get_picture("Color")
        return (image * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
