from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from project_paths import resolve_path
from sim.contracts import SceneSpec, SimulationContract

from .config import ArmSpec, CameraMount, GripperSpec
from .mjcf import MJCFConversion, write_mjcf
from .transforms import pose_to_xyz_quat_xyzw
from .urdf import discover_camera_mounts, parse_urdf

if TYPE_CHECKING:
    from .actuation.gripper_profile import GripperActuationProfile
    from sim.rm65.actuation import PositionActuationProfile


@dataclass(frozen=True)
class SceneLoadResult:
    conversion: MJCFConversion
    cameras: tuple[CameraMount, ...]
    arms: dict[str, ArmSpec]
    grippers: dict[str, GripperSpec]
    scene_spec: SceneSpec | None = None


@dataclass(frozen=True)
class MujocoContact:
    """One named contact reported by the current MuJoCo data buffer."""

    geom1_name: str
    geom2_name: str
    body1_name: str
    body2_name: str
    distance_m: float


def _load_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for sim.mujoco. Install it in dexteleop310 with "
            "`python -m pip install mujoco`."
        ) from exc
    return mujoco


def _joint_exists(mujoco, model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0


def _joint_range(mujoco, model, name: str) -> tuple[float, float] | None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return None
    if not bool(model.jnt_limited[joint_id]):
        return None
    lower, upper = model.jnt_range[joint_id]
    return float(lower), float(upper)


def _body_exists(mujoco, model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def _infer_arm_spec(mujoco, model, side: str) -> ArmSpec | None:
    # The legacy single-arm RM65 URDF convention is unqualified
    # ``arm_joint1..6``/``arm_link6`` and is used by the right-side entrypoint.
    # Do not probe that same physical chain for both sides: doing so creates
    # duplicate qpos addresses before --mode can select the active side.
    prefix_candidates = [f"{side}_arm_joint"]
    body_candidates = [f"{side}_arm_link6", f"{side}_link6"]
    if side == "right":
        prefix_candidates.extend(("arm_joint", "joint"))
        body_candidates.append("arm_link6")
    joint_names: tuple[str, ...] | None = None
    for prefix in prefix_candidates:
        candidate = tuple(f"{prefix}{index}" for index in range(1, 7))
        if all(_joint_exists(mujoco, model, name) for name in candidate):
            joint_names = candidate
            break
    if joint_names is None:
        return None
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


def _parallel_gripper_from_joints(
    mujoco,
    model,
    *,
    name: str,
    joint_names: tuple[str, ...],
    effector_name: str = "gripper",
    side: str | None = None,
    open_at_upper: bool = True,
) -> GripperSpec | None:
    if not all(_joint_exists(mujoco, model, joint_name) for joint_name in joint_names):
        return None
    ranges = [_joint_range(mujoco, model, joint_name) for joint_name in joint_names]
    if any(item is None for item in ranges):
        return None
    lower = tuple(float(item[0]) for item in ranges if item is not None)
    upper = tuple(float(item[1]) for item in ranges if item is not None)
    if open_at_upper:
        open_qpos, closed_qpos = upper, lower
    else:
        open_qpos, closed_qpos = lower, upper
    return GripperSpec(
        name=name,
        joint_names=joint_names,
        open_qpos=open_qpos,
        closed_qpos=closed_qpos,
        effector_name=effector_name,
        side=side,
    )


def _infer_gripper_specs(mujoco, model) -> dict[str, GripperSpec]:
    specs: dict[str, GripperSpec] = {}
    cad_joint_suffixes = (
        "l_1_joint",
        "l_3_joint",
        "r_1_joint",
        "r_3_joint",
        "l_2_joint",
        "r_2_joint",
    )
    candidates = (
        (
            "eg2_4b2",
            ("eg2_4b2_left_finger_joint", "eg2_4b2_right_finger_joint"),
            "gripper",
            None,
            True,
        ),
        (
            "parallel_gripper",
            ("left_finger_joint", "right_finger_joint"),
            "gripper",
            None,
            True,
        ),
        (
            "left_gripper",
            ("left_gripper_left_finger_joint", "left_gripper_right_finger_joint"),
            "left_hand",
            "left",
            True,
        ),
        (
            "right_gripper",
            ("right_gripper_left_finger_joint", "right_gripper_right_finger_joint"),
            "right_hand",
            "right",
            True,
        ),
        (
            "eg2_4b2_cad",
            tuple(f"eg2_4b2_cad_{suffix}" for suffix in cad_joint_suffixes),
            "gripper",
            "right",
            False,
        ),
        *(
            (
                f"{side}_eg2_4b2_cad",
                tuple(
                    f"{side}_eg2_4b2_cad_{suffix}"
                    for suffix in cad_joint_suffixes
                ),
                f"{side}_gripper",
                side,
                False,
            )
            for side in ("left", "right")
        ),
    )
    for name, joint_names, effector_name, side, open_at_upper in candidates:
        spec = _parallel_gripper_from_joints(
            mujoco,
            model,
            name=name,
            joint_names=joint_names,
            effector_name=effector_name,
            side=side,
            open_at_upper=open_at_upper,
        )
        if spec is not None:
            specs[spec.name] = spec
    return specs


class MujocoScene:
    """Loaded MuJoCo scene facade.

    Arm commands are still applied kinematically in this stage. The physical
    contract adds contact/world/object modeling but intentionally does not
    expose full-scene stepping before an actuator model exists.
    """

    def __init__(
        self,
        *,
        mujoco,
        model,
        data,
        load_result: SceneLoadResult,
    ):
        self._bind_native_layout(
            mujoco=mujoco,
            model=model,
            data=data,
            load_result=load_result,
        )
        self.forward()

    def _bind_native_layout(self, *, mujoco, model, data, load_result: SceneLoadResult):
        """Bind this fresh scene's indices/caches without evaluating physics."""
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.load_result = load_result
        self.conversion = load_result.conversion
        self.camera_mounts = load_result.cameras
        self.arm_specs = load_result.arms
        self.gripper_specs = load_result.grippers
        self.scene_spec = load_result.scene_spec
        self._joint_ids = self._build_joint_ids()
        self._joint_qpos_addr = self._build_joint_qpos_addr()
        self._joint_dof_addr = self._build_joint_dof_addr()
        self._body_ids = self._build_body_ids()
        self._geom_ids = self._build_geom_ids()
        self._camera_ids = self._build_camera_ids()
        self._kinematic_qpos_indices = self._build_kinematic_qpos_indices()
        self._kinematic_qpos_offset = {
            int(address): offset
            for offset, address in enumerate(self._kinematic_qpos_indices)
        }
        self._collision_projector = None
        self._collision_projectors_by_gripper_self_contact: dict[bool, object] = {}

    @classmethod
    def from_native_checkpoint(
        cls,
        state,
        *,
        load_result: SceneLoadResult,
        max_native_bytes: int,
    ) -> MujocoScene:
        """Bind a fresh scene to an independently restored native candidate.

        This restores the physics facade only, not a task or execution rig.
        The concrete profile must bind trusted scene metadata and all owners.
        Renderers, collision projectors and their scratch data are never shared.
        """
        from .native_checkpoint import restore_native_state

        if cls is not MujocoScene or type(load_result) is not SceneLoadResult:
            raise ValueError(
                "Native scene restore requires the exact scene metadata type."
            )
        # SceneSpec owns an immutable mappingproxy, which is not pickleable.
        # Reuse its typed roundtrip instead of sharing it or adding a generic
        # deepcopy fallback for unknown metadata types.
        memo = {}
        if load_result.scene_spec is not None:
            memo[id(load_result.scene_spec)] = SceneSpec.from_dict(
                load_result.scene_spec.to_dict()
            )
        metadata = copy.deepcopy(load_result, memo)
        data = restore_native_state(state, max_native_bytes=max_native_bytes)
        scene = cls.__new__(cls)
        scene._bind_native_layout(
            mujoco=_load_mujoco(),
            model=data.model,
            data=data,
            load_result=metadata,
        )
        return scene

    @classmethod
    def from_contract(
        cls,
        contract: SimulationContract,
        *,
        mjcf_path: str | Path | None = None,
        debug_proxy_visuals: bool = False,
        gripper_specs: Mapping[str, GripperSpec] | Sequence[GripperSpec] | None = None,
        object_visual_mesh_paths: Mapping[str, str | Path] | None = None,
        position_actuation_profile: PositionActuationProfile | None = None,
        gripper_actuation_profile: GripperActuationProfile | None = None,
    ) -> "MujocoScene":
        """Build the canonical MuJoCo scene from one versioned contract."""

        return cls.from_urdf(
            resolve_path(contract.scene.urdf_path),
            camera_names=contract.observation.image_keys,
            camera_registry_path=resolve_path(contract.scene.camera_registry_path),
            timestep=contract.timing.physics_timestep_sec,
            gravity=contract.scene.gravity_mps2,
            mjcf_path=mjcf_path,
            debug_proxy_visuals=debug_proxy_visuals,
            gripper_specs=gripper_specs,
            scene_spec=contract.scene,
            object_visual_mesh_paths=object_visual_mesh_paths,
            position_actuation_profile=position_actuation_profile,
            gripper_actuation_profile=gripper_actuation_profile,
        )

    @classmethod
    def from_compiled_contract(
        cls,
        contract: SimulationContract,
        *,
        model,
        mjcf_path: str | Path,
    ) -> "MujocoScene":
        """Wrap a cached compiled model while rebuilding lightweight metadata."""

        mujoco = _load_mujoco()
        urdf_model = parse_urdf(resolve_path(contract.scene.urdf_path))
        cameras = discover_camera_mounts(
            urdf_model,
            camera_names=contract.observation.image_keys,
            registry_path=resolve_path(contract.scene.camera_registry_path),
        )
        conversion = MJCFConversion(
            urdf_path=urdf_model.path,
            mjcf_path=Path(mjcf_path).expanduser().resolve(),
            camera_mounts=cameras,
            joint_names=tuple(
                joint.name for joint in urdf_model.joints if joint.joint_type != "fixed"
            ),
            body_names=tuple(urdf_model.links)
            + tuple(obj.name for obj in contract.scene.objects),
            scene_profile_name=contract.scene.profile_name,
        )
        return cls._from_compiled_model(
            mujoco=mujoco,
            model=model,
            conversion=conversion,
            cameras=cameras,
            scene_spec=contract.scene,
            gripper_specs=None,
        )

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
        camera_intrinsics_by_name: Mapping[str, Mapping[str, object]] | None = None,
        timestep: float = 1.0 / 30.0,
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mjcf_path: str | Path | None = None,
        debug_proxy_visuals: bool = False,
        gripper_specs: Mapping[str, GripperSpec] | Sequence[GripperSpec] | None = None,
        scene_spec: SceneSpec | None = None,
        object_visual_mesh_paths: Mapping[str, str | Path] | None = None,
        position_actuation_profile: PositionActuationProfile | None = None,
        gripper_actuation_profile: GripperActuationProfile | None = None,
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
            camera_intrinsics_by_name=camera_intrinsics_by_name,
        )
        conversion = write_mjcf(
            urdf_model.path,
            camera_mounts=cameras,
            output_path=mjcf_path,
            timestep=timestep,
            gravity=gravity,
            debug_proxy_visuals=debug_proxy_visuals,
            scene_spec=scene_spec,
            object_visual_mesh_paths=object_visual_mesh_paths,
            position_actuation_profile=position_actuation_profile,
            gripper_actuation_profile=gripper_actuation_profile,
        )
        model = mujoco.MjModel.from_xml_path(str(conversion.mjcf_path))
        return cls._from_compiled_model(
            mujoco=mujoco,
            model=model,
            conversion=conversion,
            cameras=cameras,
            scene_spec=scene_spec,
            gripper_specs=gripper_specs,
        )

    @classmethod
    def _from_compiled_model(
        cls,
        *,
        mujoco,
        model,
        conversion: MJCFConversion,
        cameras: tuple[CameraMount, ...],
        scene_spec: SceneSpec | None,
        gripper_specs: Mapping[str, GripperSpec] | Sequence[GripperSpec] | None,
    ) -> "MujocoScene":
        data = mujoco.MjData(model)
        arms = {
            side: spec
            for side in ("left", "right")
            if (spec := _infer_arm_spec(mujoco, model, side)) is not None
        }
        inferred_grippers = _infer_gripper_specs(mujoco, model)
        if gripper_specs is None:
            grippers = inferred_grippers
        elif isinstance(gripper_specs, Mapping):
            grippers = {str(name): spec for name, spec in gripper_specs.items()}
        else:
            grippers = {spec.name: spec for spec in gripper_specs}
        missing_gripper_joints = sorted(
            {
                joint_name
                for spec in grippers.values()
                for joint_name in spec.joint_names
                if not _joint_exists(mujoco, model, joint_name)
            }
        )
        if missing_gripper_joints:
            raise KeyError(f"Gripper spec references missing MuJoCo joints: {missing_gripper_joints}")
        return cls(
            mujoco=mujoco,
            model=model,
            data=data,
            load_result=SceneLoadResult(
                conversion=conversion,
                cameras=cameras,
                arms=arms,
                grippers=grippers,
                scene_spec=scene_spec,
            ),
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

    def _build_kinematic_qpos_indices(self) -> np.ndarray:
        addresses: list[int] = []
        for side in sorted(self.arm_specs):
            addresses.extend(
                self.joint_qpos_addr(name) for name in self.arm_specs[side].joint_names
            )
        for name in sorted(self.gripper_specs):
            addresses.extend(
                self.joint_qpos_addr(joint) for joint in self.gripper_specs[name].joint_names
            )
        if len(addresses) != len(set(addresses)):
            raise RuntimeError("Kinematic control layout contains duplicate qpos addresses.")
        return np.asarray(addresses, dtype=np.int32)

    def _build_joint_ids(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for joint_index in range(self.model.njnt):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if name:
                out[str(name)] = int(joint_index)
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

    def _build_geom_ids(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for geom_index in range(self.model.ngeom):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_index)
            if name:
                out[str(name)] = int(geom_index)
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

    def reset_data(self) -> None:
        """Reset MuJoCo runtime state without rebuilding the compiled scene."""

        self.mujoco.mj_resetData(self.model, self.data)
        self.forward()

    def joint_qpos_addr(self, joint_name: str) -> int:
        return self._joint_qpos_addr[joint_name]

    def joint_id(self, joint_name: str) -> int:
        return self._joint_ids[joint_name]

    def joint_dof_addr(self, joint_name: str) -> int:
        return self._joint_dof_addr[joint_name]

    def freejoint_qpos(self, joint_name: str) -> np.ndarray:
        """Return one named freejoint pose as ``xyz + quaternion(wxyz)``."""

        qpos_addr, _dof_addr = self._freejoint_addrs(joint_name)
        return np.asarray(self.data.qpos[qpos_addr : qpos_addr + 7], dtype=float).copy()

    def freejoint_qvel(self, joint_name: str) -> np.ndarray:
        """Return one named freejoint velocity as linear + angular velocity."""

        _qpos_addr, dof_addr = self._freejoint_addrs(joint_name)
        return np.asarray(self.data.qvel[dof_addr : dof_addr + 6], dtype=float).copy()

    def set_freejoint_state(
        self,
        joint_name: str,
        *,
        qpos: Sequence[float],
        qvel: Sequence[float] | None = None,
        forward: bool = True,
    ) -> None:
        """Set a named freejoint state with finite, normalized quaternion input.

        This is intentionally a general scene operation: task code supplies the
        object/reset semantics and never accesses MuJoCo qpos addresses itself.
        """

        qpos_addr, dof_addr = self._freejoint_addrs(joint_name)
        pose = np.asarray(qpos, dtype=float).reshape(7).copy()
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"freejoint {joint_name!r} qpos must be finite.")
        quaternion_norm = float(np.linalg.norm(pose[3:]))
        if quaternion_norm <= 1e-12:
            raise ValueError(f"freejoint {joint_name!r} quaternion must be non-zero.")
        pose[3:] /= quaternion_norm
        self.data.qpos[qpos_addr : qpos_addr + 7] = pose
        if qvel is not None:
            velocity = np.asarray(qvel, dtype=float).reshape(6)
            if not np.all(np.isfinite(velocity)):
                raise ValueError(f"freejoint {joint_name!r} qvel must be finite.")
            self.data.qvel[dof_addr : dof_addr + 6] = velocity
        if forward:
            self.forward()

    def contacts(self) -> tuple[MujocoContact, ...]:
        """Expose current contacts with stable geom/body names and distance."""

        contacts: list[MujocoContact] = []
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            contacts.append(
                MujocoContact(
                    geom1_name=self._geom_name(geom1_id),
                    geom2_name=self._geom_name(geom2_id),
                    body1_name=self._body_name(body1_id),
                    body2_name=self._body_name(body2_id),
                    distance_m=float(contact.dist),
                )
            )
        return tuple(contacts)

    def state_is_finite(self) -> bool:
        """Whether the dynamic MuJoCo state contains no NaN/inf values."""

        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    def body_id(self, body_name: str) -> int:
        return self._body_ids[body_name]

    def has_body(self, body_name: str) -> bool:
        return str(body_name) in self._body_ids

    def _freejoint_addrs(self, joint_name: str) -> tuple[int, int]:
        name = str(joint_name)
        try:
            joint_id = self._joint_ids[name]
        except KeyError as exc:
            raise KeyError(f"Scene has no joint named {name!r}.") from exc
        if int(self.model.jnt_type[joint_id]) != int(self.mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Joint {name!r} is not a MuJoCo freejoint.")
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _geom_name(self, geom_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return str(name) if name else f"<unnamed_geom:{geom_id}>"

    def _body_name(self, body_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
        return str(name) if name else f"<unnamed_body:{body_id}>"

    def camera_id(self, camera_name: str) -> int:
        return self._camera_ids[camera_name]

    def get_arm_qpos(self, side: str) -> np.ndarray:
        spec = self.arm_specs[side]
        return np.asarray(
            [self.data.qpos[self.joint_qpos_addr(joint)] for joint in spec.joint_names],
            dtype=float,
        )

    def set_arm_qpos(self, side: str, qpos: Sequence[float]) -> None:
        """Teleport one arm for reset/FK/faithful replay; no safety projection."""
        spec = self.arm_specs[side]
        values = np.asarray(qpos, dtype=float).reshape(6)
        for joint_name, value in zip(spec.joint_names, values):
            self.data.qpos[self.joint_qpos_addr(joint_name)] = float(value)
        self.forward()

    def set_qpos_by_side(self, qpos_by_side: Mapping[str, Sequence[float]]) -> None:
        """Teleport arms for reset/FK/faithful replay; no safety projection."""
        for side, qpos in qpos_by_side.items():
            spec = self.arm_specs[side]
            values = np.asarray(qpos, dtype=float).reshape(6)
            for joint_name, value in zip(spec.joint_names, values):
                self.data.qpos[self.joint_qpos_addr(joint_name)] = float(value)
        self.forward()

    @property
    def collision_safety_available(self) -> bool:
        return bool(self.scene_spec is not None and self.scene_spec.collision.enabled)

    def kinematic_controlled_qpos(self) -> np.ndarray:
        return self.data.qpos[self._kinematic_qpos_indices].copy()

    @property
    def kinematic_qpos_indices(self) -> np.ndarray:
        view = self._kinematic_qpos_indices.view()
        view.setflags(write=False)
        return view

    def arm_qpos_from_kinematic_vector(
        self, controlled_qpos: Sequence[float], side: str
    ) -> np.ndarray:
        values = np.asarray(controlled_qpos, dtype=float).reshape(-1)
        if values.shape != (self._kinematic_qpos_indices.size,):
            raise ValueError("controlled_qpos does not match scene control layout.")
        return np.asarray(
            [
                values[self._kinematic_qpos_offset[self.joint_qpos_addr(joint)]]
                for joint in self.arm_specs[str(side)].joint_names
            ],
            dtype=float,
        )

    def gripper_close_from_kinematic_vector(
        self, controlled_qpos: Sequence[float], name: str
    ) -> float:
        values = np.asarray(controlled_qpos, dtype=float).reshape(-1)
        if values.shape != (self._kinematic_qpos_indices.size,):
            raise ValueError("controlled_qpos does not match scene control layout.")
        spec = self.gripper_specs[self._resolve_gripper_name(name)]
        qpos = np.asarray(
            [
                values[self._kinematic_qpos_offset[self.joint_qpos_addr(joint)]]
                for joint in spec.joint_names
            ],
            dtype=float,
        )
        return spec.close_from_qpos(qpos)

    def kinematic_target_vector(
        self,
        base_qpos: Sequence[float],
        qpos_by_side: Mapping[str, Sequence[float]],
        *,
        gripper_close_by_name: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        target = np.asarray(base_qpos, dtype=float).reshape(-1).copy()
        if target.shape != (self._kinematic_qpos_indices.size,):
            raise ValueError("base_qpos does not match the scene kinematic control layout.")
        for side, qpos in qpos_by_side.items():
            spec = self.arm_specs[str(side)]
            values = np.asarray(qpos, dtype=float).reshape(6)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{side} qpos must be finite.")
            for joint_name, value in zip(spec.joint_names, values):
                address = self.joint_qpos_addr(joint_name)
                target[self._kinematic_qpos_offset[address]] = value
        for name, close in (gripper_close_by_name or {}).items():
            spec = self.gripper_specs[self._resolve_gripper_name(name)]
            normalized_close = float(close)
            if not np.isfinite(normalized_close) or not 0.0 <= normalized_close <= 1.0:
                raise ValueError(f"gripper close for {spec.name!r} must be finite and in [0, 1].")
            for joint_name, value in zip(spec.joint_names, spec.qpos_from_close(normalized_close)):
                address = self.joint_qpos_addr(joint_name)
                target[self._kinematic_qpos_offset[address]] = value
        return target

    def collision_projector(self, *, allow_gripper_self_contact: bool = False):
        if not self.collision_safety_available:
            return None
        if not isinstance(allow_gripper_self_contact, bool):
            raise TypeError("allow_gripper_self_contact must be a bool.")
        if not allow_gripper_self_contact and self._collision_projector is not None:
            return self._collision_projector
        if (
            allow_gripper_self_contact
            in self._collision_projectors_by_gripper_self_contact
        ):
            return self._collision_projectors_by_gripper_self_contact[
                allow_gripper_self_contact
            ]
        if self._collision_projector is None or allow_gripper_self_contact:
            from .collision_safety import (
                CollisionSafetyPolicy,
                KinematicCollisionProjector,
            )

            collision = self.scene_spec.collision

            projector = KinematicCollisionProjector(
                self.mujoco,
                self.model,
                self._kinematic_qpos_indices,
                CollisionSafetyPolicy(
                    minimum_clearance_m=collision.safety_minimum_clearance_m,
                    hard_penetration_m=collision.safety_hard_penetration_m,
                    sweep_max_joint_step_rad=collision.safety_sweep_max_joint_step_rad,
                    projection_iterations=collision.safety_projection_iterations,
                    allow_gripper_self_contact=allow_gripper_self_contact,
                ),
            )
            self._collision_projectors_by_gripper_self_contact[
                allow_gripper_self_contact
            ] = projector
            if not allow_gripper_self_contact:
                self._collision_projector = projector
        return self._collision_projectors_by_gripper_self_contact.get(
            allow_gripper_self_contact,
            self._collision_projector,
        )

    def write_kinematic_controlled_qpos(
        self,
        controlled_qpos: Sequence[float],
        *,
        zero_controlled_qvel: bool = True,
    ) -> None:
        """Raw accepted-state write used only after collision validation."""

        values = np.asarray(controlled_qpos, dtype=float).reshape(-1)
        if values.shape != (self._kinematic_qpos_indices.size,):
            raise ValueError("controlled_qpos does not match scene control layout.")
        self.data.qpos[self._kinematic_qpos_indices] = values
        if zero_controlled_qvel:
            dof_addresses: list[int] = []
            for side in sorted(self.arm_specs):
                dof_addresses.extend(
                    self.joint_dof_addr(name) for name in self.arm_specs[side].joint_names
                )
            for name in sorted(self.gripper_specs):
                dof_addresses.extend(
                    self.joint_dof_addr(joint)
                    for joint in self.gripper_specs[name].joint_names
                )
            self.data.qvel[np.asarray(dof_addresses, dtype=np.int32)] = 0.0

    def apply_safe_kinematic_target(
        self,
        qpos_by_side: Mapping[str, Sequence[float]],
        *,
        gripper_close_by_name: Mapping[str, float] | None = None,
        project_invalid: bool = False,
    ):
        """Apply a collision-projected kinematic command without stepping time."""

        start = self.kinematic_controlled_qpos()
        requested = self.kinematic_target_vector(
            start,
            qpos_by_side,
            gripper_close_by_name=gripper_close_by_name,
        )
        projector = self.collision_projector()
        if projector is None:
            result = None
            accepted = requested
        else:
            result = projector.project(
                self.data,
                start_qpos=start,
                requested_qpos=requested,
            )
            if result.status == "invalid_safe_state":
                raise RuntimeError(
                    "Cannot apply command from a collision-unsafe state: "
                    f"{result.to_dict()}"
                )
            accepted = (
                result.accepted_qpos
                if project_invalid or not result.modified
                else start
            )
        self.write_kinematic_controlled_qpos(accepted)
        self.forward()
        return result

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side: Mapping[str, Sequence[float]],
        *,
        gripper_close_by_name: Mapping[str, float] | None = None,
        zero_controlled_qvel: bool = True,
    ) -> None:
        """Advance one physics tick with prescribed robot joint positions.

        This is intentionally a debug-only bridge for task-pipeline work before
        an actuator model has been identified.  It re-applies arm and gripper
        qpos before and after each ``mj_step`` so gravity/contact integration
        cannot move the prescribed robot away from its target. The free object
        still receives the contact dynamics computed during ``mj_step``. It
        does *not* model a real position servo or target tracking.
        """

        target = self.kinematic_target_vector(
            self.kinematic_controlled_qpos(),
            qpos_by_side,
            gripper_close_by_name=gripper_close_by_name,
        )
        self.write_kinematic_controlled_qpos(
            target,
            zero_controlled_qvel=zero_controlled_qvel,
        )
        self.forward()
        self.mujoco.mj_step(self.model, self.data)
        self.write_kinematic_controlled_qpos(
            target,
            zero_controlled_qvel=zero_controlled_qvel,
        )
        self.forward()

    def arm_qpos_by_side(self) -> dict[str, list[float]]:
        return {
            side: [float(value) for value in self.get_arm_qpos(side)]
            for side in sorted(self.arm_specs)
        }

    def _resolve_gripper_name(self, name: str | None = None) -> str:
        if name is not None:
            key = str(name)
            if key in self.gripper_specs:
                return key
            # Task and mapping contracts historically used the logical
            # ``left_gripper``/``right_gripper`` names.  Reviewed CAD scene
            # assets expose the physical module as
            # ``*_eg2_4b2_cad`` instead.  Resolve the legacy logical name
            # only when the scene has exactly one gripper on that side; an
            # ambiguous scene still fails closed instead of guessing.
            legacy_side = {
                "left_gripper": "left",
                "right_gripper": "right",
            }.get(key)
            if legacy_side is not None:
                side_matches = tuple(
                    candidate
                    for candidate, spec in self.gripper_specs.items()
                    if spec.side == legacy_side
                )
                if len(side_matches) == 1:
                    return side_matches[0]
            raise KeyError(f"Scene has no gripper spec named {key!r}.")
        if len(self.gripper_specs) == 1:
            return next(iter(self.gripper_specs))
        if not self.gripper_specs:
            raise KeyError("Scene has no gripper specs.")
        raise KeyError(
            "Multiple gripper specs are available; specify one of "
            f"{sorted(self.gripper_specs)}."
        )

    def get_gripper_qpos(self, name: str | None = None) -> np.ndarray:
        spec = self.gripper_specs[self._resolve_gripper_name(name)]
        return np.asarray(
            [self.data.qpos[self.joint_qpos_addr(joint)] for joint in spec.joint_names],
            dtype=float,
        )

    def set_gripper_qpos(self, name: str | None, qpos: Sequence[float]) -> None:
        """Teleport gripper joints for reset/replay; no safety projection."""
        spec = self.gripper_specs[self._resolve_gripper_name(name)]
        values = np.asarray(qpos, dtype=float).reshape(len(spec.joint_names))
        for joint_name, value in zip(spec.joint_names, values):
            self.data.qpos[self.joint_qpos_addr(joint_name)] = float(value)
        self.forward()

    def get_gripper_close(self, name: str | None = None) -> float:
        spec = self.gripper_specs[self._resolve_gripper_name(name)]
        return spec.close_from_qpos(self.get_gripper_qpos(spec.name))

    def set_gripper_close(self, name: str | None, close: float) -> None:
        """Teleport gripper close state for reset/replay; no safety projection."""
        spec = self.gripper_specs[self._resolve_gripper_name(name)]
        self.set_gripper_qpos(spec.name, spec.qpos_from_close(close))

    def gripper_close_by_name(self) -> dict[str, float]:
        return {
            name: float(self.get_gripper_close(name))
            for name in sorted(self.gripper_specs)
        }

    def gripper_qpos_by_name(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in self.get_gripper_qpos(name)]
            for name in sorted(self.gripper_specs)
        }

    def gripper_metadata(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                **spec.metadata(),
                "current_qpos": [float(value) for value in self.get_gripper_qpos(name)],
                "current_close": float(self.get_gripper_close(name)),
            }
            for name, spec in sorted(self.gripper_specs.items())
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
                # Pinhole depth uses +x right, +y down, +z forward; MuJoCo
                # renders along -z with +y up (the converter adds Rx(pi)).
                "optical_camera_pose_world_xyz_quat_xyzw": pose_to_xyz_quat_xyzw(
                    self.camera_pose(mount.name) @ np.diag((1., -1., -1., 1.))
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
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.scene.data, camera=camera_name)
        return np.asarray(self.renderer.render()).copy()

    def render_all(self, camera_names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        names = tuple(camera_names) if camera_names is not None else self.scene.camera_names
        return {name: self.render(name) for name in names}

    def render_depth(self, camera_name: str) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.scene.data, camera=camera_name)
            return np.asarray(self.renderer.render(), dtype=np.float32).copy()
        finally:
            self.renderer.disable_depth_rendering()

    def render_all_depth(self, camera_names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        names = tuple(camera_names) if camera_names is not None else self.scene.camera_names
        return {name: self.render_depth(name) for name in names}

    def close(self) -> None:
        close = getattr(self.renderer, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "MujocoCameraRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
