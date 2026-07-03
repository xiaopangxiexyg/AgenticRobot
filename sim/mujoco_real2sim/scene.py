from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import ArmSpec, CameraMount
from .mjcf import MJCFConversion, write_mjcf
from .transforms import pose_to_xyz_quat_xyzw
from .urdf import discover_camera_mounts, parse_urdf


@dataclass(frozen=True)
class SceneLoadResult:
    conversion: MJCFConversion
    cameras: tuple[CameraMount, ...]
    arms: dict[str, ArmSpec]


def _load_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for sim.mujoco_real2sim. Install it in dexteleop310 with "
            "`python -m pip install mujoco`."
        ) from exc
    return mujoco


def _joint_exists(mujoco, model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0


def _body_exists(mujoco, model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def _infer_arm_spec(mujoco, model, side: str) -> ArmSpec | None:
    prefix_candidates = (f"{side}_arm_joint", "arm_joint", "joint")
    joint_names: tuple[str, ...] | None = None
    for prefix in prefix_candidates:
        candidate = tuple(f"{prefix}{index}" for index in range(1, 7))
        if all(_joint_exists(mujoco, model, name) for name in candidate):
            joint_names = candidate
            break
    if joint_names is None:
        return None
    body_candidates = (f"{side}_arm_link6", "arm_link6", f"{side}_link6")
    controlled_body = next(
        (name for name in body_candidates if _body_exists(mujoco, model, name)),
        "",
    )
    if not controlled_body:
        return None
    return ArmSpec(
        side=side,
        joint_names=joint_names,
        controlled_body=controlled_body,
        effector_name=f"{side}_arm",
    )


class MujocoScene:
    """Position-controlled MuJoCo scene loaded from an explicit URDF."""

    def __init__(
        self,
        *,
        mujoco,
        model,
        data,
        load_result: SceneLoadResult,
    ):
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.load_result = load_result
        self.conversion = load_result.conversion
        self.camera_mounts = load_result.cameras
        self.arm_specs = load_result.arms
        self._joint_qpos_addr = self._build_joint_qpos_addr()
        self._joint_dof_addr = self._build_joint_dof_addr()
        self._body_ids = self._build_body_ids()
        self._camera_ids = self._build_camera_ids()
        self.forward()

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        *,
        camera_names: Sequence[str] = ("all",),
        camera_registry_path: str | Path | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        default_camera_fovy_deg: float = 60.0,
        timestep: float = 1.0 / 30.0,
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mjcf_path: str | Path | None = None,
    ) -> "MujocoScene":
        mujoco = _load_mujoco()
        urdf_model = parse_urdf(urdf_path)
        cameras = discover_camera_mounts(
            urdf_model,
            camera_names=camera_names,
            registry_path=camera_registry_path,
            default_width=camera_width,
            default_height=camera_height,
            default_fovy_deg=default_camera_fovy_deg,
        )
        conversion = write_mjcf(
            urdf_model.path,
            camera_mounts=cameras,
            output_path=mjcf_path,
            timestep=timestep,
            gravity=gravity,
        )
        model = mujoco.MjModel.from_xml_path(str(conversion.mjcf_path))
        data = mujoco.MjData(model)
        arms = {
            side: spec
            for side in ("left", "right")
            if (spec := _infer_arm_spec(mujoco, model, side)) is not None
        }
        return cls(
            mujoco=mujoco,
            model=model,
            data=data,
            load_result=SceneLoadResult(conversion=conversion, cameras=cameras, arms=arms),
        )

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(mount.name for mount in self.camera_mounts if mount.name in self._camera_ids)

    def _build_joint_qpos_addr(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for joint_index in range(self.model.njnt):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if name:
                out[str(name)] = int(self.model.jnt_qposadr[joint_index])
        return out

    def _build_joint_dof_addr(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for joint_index in range(self.model.njnt):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if name:
                out[str(name)] = int(self.model.jnt_dofadr[joint_index])
        return out

    def _build_body_ids(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for body_index in range(self.model.nbody):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_index)
            if name:
                out[str(name)] = int(body_index)
        return out

    def _build_camera_ids(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for camera_index in range(self.model.ncam):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera_index)
            if name:
                out[str(name)] = int(camera_index)
        return out

    def forward(self) -> None:
        self.mujoco.mj_forward(self.model, self.data)

    def joint_qpos_addr(self, joint_name: str) -> int:
        return self._joint_qpos_addr[joint_name]

    def joint_dof_addr(self, joint_name: str) -> int:
        return self._joint_dof_addr[joint_name]

    def body_id(self, body_name: str) -> int:
        return self._body_ids[body_name]

    def camera_id(self, camera_name: str) -> int:
        return self._camera_ids[camera_name]

    def get_arm_qpos(self, side: str) -> np.ndarray:
        spec = self.arm_specs[side]
        return np.asarray(
            [self.data.qpos[self.joint_qpos_addr(joint)] for joint in spec.joint_names],
            dtype=float,
        )

    def set_arm_qpos(self, side: str, qpos: Sequence[float]) -> None:
        spec = self.arm_specs[side]
        values = np.asarray(qpos, dtype=float).reshape(6)
        for joint_name, value in zip(spec.joint_names, values):
            self.data.qpos[self.joint_qpos_addr(joint_name)] = float(value)
        self.forward()

    def set_qpos_by_side(self, qpos_by_side: Mapping[str, Sequence[float]]) -> None:
        for side, qpos in qpos_by_side.items():
            spec = self.arm_specs[side]
            values = np.asarray(qpos, dtype=float).reshape(6)
            for joint_name, value in zip(spec.joint_names, values):
                self.data.qpos[self.joint_qpos_addr(joint_name)] = float(value)
        self.forward()

    def arm_qpos_by_side(self) -> dict[str, list[float]]:
        return {
            side: [float(value) for value in self.get_arm_qpos(side)]
            for side in sorted(self.arm_specs)
        }

    def body_pose(self, body_name: str) -> np.ndarray:
        body_id = self.body_id(body_name)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(self.data.xpos[body_id], dtype=float)
        pose[:3, :3] = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
        return pose

    def camera_pose(self, camera_name: str) -> np.ndarray:
        camera_id = self.camera_id(camera_name)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(self.data.cam_xpos[camera_id], dtype=float)
        pose[:3, :3] = np.asarray(self.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        return pose

    def camera_link_pose(self, camera_name: str) -> np.ndarray:
        mount = next(mount for mount in self.camera_mounts if mount.name == camera_name)
        return self.body_pose(mount.link_name)

    def camera_metadata(self) -> dict[str, dict[str, object]]:
        metadata: dict[str, dict[str, object]] = {}
        for mount in self.camera_mounts:
            if mount.name not in self._camera_ids:
                continue
            metadata[mount.name] = {
                "link_name": mount.link_name,
                "width": int(mount.width),
                "height": int(mount.height),
                "fovy_deg": float(mount.fovy_deg),
                "intrinsics": None if mount.intrinsics is None else dict(mount.intrinsics),
                "source": mount.source,
                "camera_pose_world_xyz_quat_xyzw": pose_to_xyz_quat_xyzw(
                    self.camera_pose(mount.name)
                ),
                "link_pose_world_xyz_quat_xyzw": pose_to_xyz_quat_xyzw(
                    self.camera_link_pose(mount.name)
                ),
            }
        return metadata


class MujocoCameraRenderer:
    def __init__(self, scene: MujocoScene, *, width: int = 640, height: int = 480):
        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.renderer = scene.mujoco.Renderer(scene.model, height=self.height, width=self.width)

    def render(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.scene.data, camera=camera_name)
        return np.asarray(self.renderer.render()).copy()

    def render_all(self, camera_names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        names = tuple(camera_names) if camera_names is not None else self.scene.camera_names
        return {name: self.render(name) for name in names}

    def close(self) -> None:
        close = getattr(self.renderer, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "MujocoCameraRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

