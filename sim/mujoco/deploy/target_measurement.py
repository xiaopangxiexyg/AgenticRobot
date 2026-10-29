"""Shared MuJoCo target identity and measurement adapters.

These helpers derive immutable frame/tool identities and read target state from
an already-owned MuJoCo scene.  They do not select an application profile,
construct authority, apply safety, resolve IK, or dispatch commands.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.pose_frames import load_pose_frame_contract
from project_paths import resolve_path
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    CanonicalActionSafetyAnchor,
    HandActionSpec,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import (
    STATION_RETARGET_FRAME_CONTRACT_V1,
    TargetFrameProfile,
    ToolFrameBinding,
    ToolFrameProfile,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile

if TYPE_CHECKING:
    from .target_preparation import MujocoSimHILPreparedTarget


MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1 = "mujoco_sim_hil/world"
MUJOCO_SIM_HIL_STATION_FRAME_ID_V1 = "mujoco_sim_hil/station"


def build_mujoco_sim_hil_pose_observation_projector(
    prepared_target: MujocoSimHILPreparedTarget,
    component_execution_binding: Any,
) -> Any:
    """Bind the active component profile to private-MjData FK observation."""

    from .target_preparation import MujocoSimHILPreparedTarget

    if not isinstance(prepared_target, MujocoSimHILPreparedTarget):
        raise TypeError(
            "prepared_target must be a MujocoSimHILPreparedTarget."
        )
    materializer = getattr(component_execution_binding, "materializer", None)
    profile = getattr(materializer, "profile", None)
    arm_side_by_component = getattr(profile, "arm_side_by_component", None)
    action_schema = getattr(component_execution_binding, "action_schema", None)
    if not isinstance(arm_side_by_component, Mapping) or action_schema is None:
        raise TypeError(
            "Component execution binding must expose its MuJoCo arm profile."
        )
    from .pose_kinematics import build_mujoco_pose_observation_projector

    scene = prepared_target.scene
    task_config = prepared_target.task_config
    tool_profile_path_by_source = {
        "left": task_config.left_tool_profile_path,
        "right": task_config.right_tool_profile_path,
    }
    endpoint_to_controlled_by_component = {}
    reference_body_by_component = {}
    endpoint_body_by_component = {}
    joint_names_by_source = {}
    for component_id, raw_source in arm_side_by_component.items():
        source = str(raw_source)
        payload_schema = action_schema.payload_schema_by_component[component_id]
        if payload_schema.controlled_frame != "tool_tcp":
            raise ValueError(
                "The current Sim-HIL FK profile requires tool_tcp arm outputs; "
                "another controlled frame must provide its own explicit offset."
            )
        tool_profile = load_rm65_tool_profile(
            tool_profile_path_by_source[source],
            side=source,
        )
        endpoint_to_controlled_by_component[component_id] = (
            endpoint_to_tool_transform(tool_profile)
        )
        reference_body_by_component[component_id] = _arm_base_body_name(
            scene,
            source,
        )
        endpoint_body_by_component[component_id] = (
            scene.arm_specs[source].controlled_body
        )
        joint_names_by_source[source] = tuple(
            scene.arm_specs[source].joint_names
        )

    model_contract_hash = canonical_content_hash(
        {
            "target_frame_profile_hash": (
                prepared_target.target_frame_profile.semantic_content_hash
            ),
            "tool_frame_profile_hash": (
                prepared_target.tool_frame_profile.semantic_content_hash
            ),
            "action_schema_hash": action_schema.semantic_content_hash,
            "scene_profile_name": scene.conversion.scene_profile_name,
            "model_nq": int(scene.model.nq),
            "model_nv": int(scene.model.nv),
        },
        domain="dexteleop.sim.mujoco.sim_hil_fk_model_contract.development",
    )
    return build_mujoco_pose_observation_projector(
        scene,
        action_schema=action_schema,
        qpos_source_by_component={
            str(component_id): str(source)
            for component_id, source in arm_side_by_component.items()
        },
        joint_names_by_source=joint_names_by_source,
        reference_body_by_component=reference_body_by_component,
        endpoint_body_by_component=endpoint_body_by_component,
        endpoint_to_controlled_by_component=(
            endpoint_to_controlled_by_component
        ),
        model_contract_id=(
            f"mujoco-sim-hil-fk:{scene.conversion.scene_profile_name}"
        ),
        model_contract_hash=model_contract_hash,
    )


def _file_sha256(path: str | Path) -> str:
    candidate = resolve_path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_base_body_name(scene: Any, side: str) -> str:
    candidates = (f"{side}_arm_base_link", "arm_base_link", "base_link")
    base_name = next((name for name in candidates if scene.has_body(name)), None)
    if base_name is None:
        raise KeyError(f"Could not resolve {side} arm base body.")
    return base_name


def build_mujoco_sim_hil_coordinate_profiles(
    scene: Any,
    task_config: Any,
    *,
    active_sides: tuple[str, ...] = ("left", "right"),
    station_pose_contract_path: str | Path | None = None,
) -> tuple[TargetFrameProfile, ToolFrameProfile]:
    """Derive immutable target/tool identities from the loaded MuJoCo scene."""

    world_frame_id = MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1
    canonical_frame_id = MUJOCO_SIM_HIL_STATION_FRAME_ID_V1
    station_contract = (
        None
        if station_pose_contract_path is None
        else load_pose_frame_contract(station_pose_contract_path)
    )
    if station_contract is not None:
        if station_contract.pose_storage_frame != "desk_absolute":
            raise ValueError(
                "MuJoCo station pose contract must store desk_absolute poses."
            )
        if station_contract.canonical_absolute_frame != "desk":
            raise ValueError(
                "MuJoCo station pose contract canonical frame must be desk."
            )
    sides = tuple(str(side).strip().lower() for side in active_sides)
    if (
        not sides
        or any(side not in {"left", "right"} for side in sides)
        or len(sides) != len(set(sides))
    ):
        raise ValueError("active_sides must contain unique known sides.")

    base_body_by_side = {
        side: _arm_base_body_name(scene, side) for side in sides
    }
    world_from_base_by_side = {
        side: scene.body_pose(base_body_by_side[side]) for side in sides
    }
    if station_contract is None:
        station_from_world = np.eye(4, dtype=np.float64)
        station_transform_basis = "historical_scene_world_identity"
    else:
        station_from_world_by_side = {
            side: (
                station_contract.source.effective_raw_to_canonical_desk_transform(
                    side
                )
                @ np.linalg.inv(world_from_base_by_side[side])
            )
            for side in sides
        }
        station_from_world = station_from_world_by_side[sides[0]]
        if any(
            not np.allclose(
                candidate,
                station_from_world,
                rtol=0.0,
                atol=1e-9,
            )
            for candidate in station_from_world_by_side.values()
        ):
            raise ValueError(
                "MuJoCo arm mounts and Station contract disagree on world-to-Station."
            )
        station_transform_basis = "reviewed_station_pose_contract_and_scene_mounts"

    frames = [
        CoordinateFrameSpec(
            world_frame_id,
            frame_version="1",
            semantic_parameters={
                "frame_family": "simulation_world",
                "backend": "mujoco",
                "axis_contract": "mujoco_world_xyz_right_handed_v1",
            },
        ),
        CoordinateFrameSpec(
            canonical_frame_id,
            frame_version="1",
            semantic_parameters={
                "frame_family": "target_station",
                "source": (
                    "historical_scene_world_identity"
                    if station_contract is None
                    else "reviewed_station_pose_contract"
                ),
                "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
                **(
                    {}
                    if station_contract is None
                    else {"pose_contract_hash": station_contract.contract_hash}
                ),
            },
        )
    ]
    transforms = [
        RigidFrameTransformSpec(
            transform_id="mujoco-scene:world-to-canonical-station-v1",
            transform_version="1",
            source_frame_id=world_frame_id,
            target_frame_id=canonical_frame_id,
            matrix_target_source=station_from_world,
            semantic_parameters={
                "source": station_transform_basis,
                **(
                    {}
                    if station_contract is None
                    else {"pose_contract_hash": station_contract.contract_hash}
                ),
            },
        )
    ]
    reference_by_component: dict[str, str] = {}
    for side in sides:
        component_id = f"{side}_arm"
        reference_frame_id = f"{side}_arm_base"
        body_name = base_body_by_side[side]
        frames.append(
            CoordinateFrameSpec(
                reference_frame_id,
                semantic_parameters={
                    "frame_family": "arm_base",
                    "side": side,
                    "scene_body_name": body_name,
                },
            )
        )
        transforms.append(
            RigidFrameTransformSpec(
                transform_id=f"mujoco-scene:{reference_frame_id}-to-world-v1",
                transform_version="1",
                source_frame_id=reference_frame_id,
                target_frame_id=world_frame_id,
                matrix_target_source=world_from_base_by_side[side],
                semantic_parameters={
                    "source": "loaded_scene_body_pose",
                    "scene_body_name": body_name,
                },
            )
        )
        reference_by_component[component_id] = reference_frame_id

    scene_profile_name = str(scene.conversion.scene_profile_name).strip()
    if not scene_profile_name:
        raise ValueError("scene conversion profile name must be non-empty.")
    urdf_path = Path(scene.conversion.urdf_path).expanduser().resolve()
    target_profile = TargetFrameProfile(
        profile_id=f"mujoco-target-frame:{scene_profile_name}",
        target_kind="simulation",
        # The resolved path is a deployment-local locator, not semantic
        # identity. Bind the logical scene profile here and retain the exact
        # file bytes separately through artifact_hash so equivalent Git
        # checkouts and worktrees publish identical contracts.
        artifact_id=f"dexteleop.mujoco.target_urdf.v1:{scene_profile_name}",
        artifact_hash=_file_sha256(urdf_path),
        frame_graph=CoordinateFrameGraph(
            graph_id=f"mujoco-target-frame:{scene_profile_name}",
            frames=tuple(frames),
            transforms=tuple(transforms),
        ),
        canonical_reference_frame_id=canonical_frame_id,
        reference_frame_by_component=reference_by_component,
        semantic_parameters={
            "backend": "mujoco",
            "derivation": (
                "loaded_scene_body_pose"
                if station_contract is None
                else "reviewed_station_pose_contract"
            ),
            **(
                {}
                if station_contract is None
                else {
                    "station_pose_contract_hash": station_contract.contract_hash
                }
            ),
            "canonical_target_frame_contract_id": (
                STATION_RETARGET_FRAME_CONTRACT_V1
            ),
        },
    )

    tool_bindings = []
    profile_path_by_side = {
        "left": task_config.left_tool_profile_path,
        "right": task_config.right_tool_profile_path,
    }
    task_profile_name = str(task_config.name).strip()
    if not task_profile_name:
        raise ValueError("task profile name must be non-empty.")
    for side in sides:
        profile_path = profile_path_by_side[side]
        resolved = resolve_path(profile_path)
        tool_bindings.append(
            ToolFrameBinding(
                component_id=f"{side}_arm",
                controlled_frame_id="tool_tcp",
                # Keep the local source path out of the content-addressed
                # motion contract. The task profile and component identify
                # the semantic artifact while artifact_hash proves the exact
                # calibration bytes used by this deployment.
                artifact_id=(
                    "dexteleop.mujoco.tool_profile.v1:"
                    f"{task_profile_name}:{side}"
                ),
                artifact_hash=_file_sha256(resolved),
            )
        )
    tool_profile = ToolFrameProfile(
        profile_id=f"mujoco-tool-frame:{task_profile_name}",
        target_frame_profile_hash=target_profile.semantic_content_hash,
        bindings=tuple(tool_bindings),
    )
    return target_profile, tool_profile


# (profile name, side) -> (resolved path, (mtime_ns, size), endpoint_T_tool).
# Held-object guards call the TCP capture on every 4 kHz physics step; the
# calibration file is re-read only when its identity on disk changes.
_ENDPOINT_TO_TOOL_CACHE: dict[tuple[str, str], tuple[str, tuple[int, int], np.ndarray]] = {}
_ENDPOINT_TO_TOOL_LOCK = threading.Lock()


def _file_identity(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _endpoint_to_tool_by_side(task_config: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        name = getattr(task_config, side + "_tool_profile_path")
        key = (str(name), side)
        cached = _ENDPOINT_TO_TOOL_CACHE.get(key)
        if cached is not None and _file_identity(cached[0]) == cached[1]:
            result[side] = cached[2]
            continue
        with _ENDPOINT_TO_TOOL_LOCK:
            profile = load_rm65_tool_profile(name, side=side)
            path = str(profile["_profile_path"])
            identity = _file_identity(path)
            transform = np.array(endpoint_to_tool_transform(profile), dtype=float)
            transform.setflags(write=False)
            if identity is not None:
                _ENDPOINT_TO_TOOL_CACHE[key] = (path, identity, transform)
        result[side] = transform
    return result


def capture_mujoco_sim_hil_tool_tcp_pose_by_side(
    scene: Any,
    task_config: Any,
) -> dict[str, np.ndarray]:
    """Capture current ``T_arm_base_tool_tcp`` for each active arm."""

    result: dict[str, np.ndarray] = {}
    endpoint_to_tool = _endpoint_to_tool_by_side(task_config)
    for side in ("left", "right"):
        base_name = _arm_base_body_name(scene, side)
        endpoint = scene.body_pose(scene.arm_specs[side].controlled_body)
        pose = np.linalg.inv(scene.body_pose(base_name)) @ endpoint
        result[side] = pose @ endpoint_to_tool[side]
    return result


def capture_mujoco_sim_hil_safety_anchor(
    scene: Any,
    task_config: Any,
    hand_specs: Mapping[str, HandActionSpec],
    *,
    now: float,
) -> CanonicalActionSafetyAnchor:
    """Capture one measured bimanual pose/hand anchor for shared safety."""

    arm_pose_by_side: dict[str, np.ndarray] = {}
    tool_tcp_pose_by_side = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
        scene,
        task_config,
    )
    for side, pose in tool_tcp_pose_by_side.items():
        vector = pose_matrix_to_xyzquat(pose)
        if vector is None:
            raise ValueError(
                f"Current {side} TCP pose cannot be represented as xyzquat."
            )
        arm_pose_by_side[side] = np.asarray(vector, dtype=np.float32)

    hand_qpos_by_side: dict[str, np.ndarray] = {}
    gripper_name_by_side = {
        "left": task_config.left_gripper_name,
        "right": task_config.right_gripper_name,
    }
    for side in hand_specs:
        name = gripper_name_by_side[side]
        values = np.zeros(6, dtype=np.float32)
        values[0] = float(scene.get_gripper_close(name))
        hand_qpos_by_side[side] = values
    return CanonicalActionSafetyAnchor(
        anchor_id=f"mujoco-sim-hil-measured-anchor:{float(now):.9f}",
        anchor_kind="measured_state",
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side=arm_pose_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        hand_action_specs_by_side=hand_specs,
        frame_by_side={side: f"{side}_arm_base" for side in ("left", "right")},
        observed_at=float(now),
    )


__all__ = [
    "MUJOCO_SIM_HIL_STATION_FRAME_ID_V1",
    "MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1",
    "build_mujoco_sim_hil_coordinate_profiles",
    "build_mujoco_sim_hil_pose_observation_projector",
    "capture_mujoco_sim_hil_safety_anchor",
    "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
]
