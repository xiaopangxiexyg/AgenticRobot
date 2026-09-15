"""MuJoCo-only B601 contact-scene and end-effector wrench observation.

The observer reads one native MuJoCo ``sensor/contact`` result.  It performs no
control, filtering, force reflection, or hardware I/O.  The reported wrench is
world/base-axis aligned and shifted to the ``gripper_end`` origin so it can be
compared with the existing B601 Jacobian-facing wrench contract.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    B601SpatialWrench,
)
from sim.contracts import (
    CollisionSpec,
    FreeBodyObjectSpec,
    GroundSpec,
    SceneSpec,
)
from sim.mujoco.scene import MujocoScene

B601_RS_CONTACT_SENSOR_NAME = "b601_rs_gripper_end_net_contact"
B601_RS_CONTACT_SENSOR_FIELDS = ("found", "force", "torque", "pos")
B601_RS_CONTACT_SENSOR_DIM = 10
B601_RS_CONTACT_SCENE_PROFILE_NAME = "rebot_b601_rs_contact_observation_v1"


@dataclass(frozen=True)
class B601MujocoContactObservation:
    """One net contact wrench acting on the B601 end-effector subtree."""

    wrench: B601SpatialWrench
    physics_time_sec: float
    matched_contact_count: int
    application_point_world_m: tuple[float, float, float] | None
    sensor_name: str = B601_RS_CONTACT_SENSOR_NAME
    source: str = "mujoco_contact_sensor_subtree2_netforce"
    force_calibrated: bool = False

    @property
    def in_contact(self) -> bool:
        return self.matched_contact_count > 0


def b601_contact_scene_spec(
    *,
    objects: Sequence[FreeBodyObjectSpec] = (),
    ground_z_m: float = -0.05,
    gravity_mps2: Sequence[float] = (0.0, 0.0, -9.81),
) -> SceneSpec:
    """Return the opt-in B601 physics/contact scene contract.

    The generic scene compiler owns ground, object, contact-material, and
    collision-filter semantics.  This helper supplies only B601 identities and
    conservative defaults.  It deliberately does not copy the upstream RS
    MJCF or its meshes.
    """

    return SceneSpec(
        profile_name=B601_RS_CONTACT_SCENE_PROFILE_NAME,
        backend_role="b601_contact_observation_diagnostic",
        urdf_path=B601_RS_URDF_RELATIVE_PATH.as_posix(),
        camera_registry_path="not_used_by_b601_contact_scene",
        gravity_mps2=tuple(float(value) for value in gravity_mps2),
        ground=GroundSpec(z_m=float(ground_z_m)),
        objects=tuple(objects),
        collision=CollisionSpec(
            enabled=True,
            self_collision_enabled=True,
            exclude_parent_child=True,
        ),
    )


def add_b601_contact_wrench_sensor(root: ET.Element) -> None:
    """Add one deterministic net-contact sensor to a B601 MJCF document."""

    if root.find(f".//sensor/contact[@name='{B601_RS_CONTACT_SENSOR_NAME}']") is not None:
        raise ValueError("B601 contact sensor is already present")
    sensor_root = root.find("./sensor")
    if sensor_root is None:
        sensor_root = ET.SubElement(root, "sensor")
    # ``subtree2`` fixes the contact direction on the end-effector side.
    # ``netforce`` returns one synthetic contact in world orientation whose
    # wrench is expressed about the returned force-weighted contact position.
    ET.SubElement(
        sensor_root,
        "contact",
        {
            "name": B601_RS_CONTACT_SENSOR_NAME,
            "subtree2": B601_RS_END_EFFECTOR_FRAME,
            "num": "1",
            "data": " ".join(B601_RS_CONTACT_SENSOR_FIELDS),
            "reduce": "netforce",
        },
    )


def build_b601_contact_box_fixture(
    source_root: str | Path,
    initial_arm_qpos: Sequence[float],
) -> tuple[MujocoScene, FreeBodyObjectSpec]:
    """Build the shared deterministic contact-test fixture, not a plant profile."""
    # Scene compilation imports sensor assembly from this module. Keep the
    # factory import local rather than introducing a second compiler.
    from .scene import build_b601_mujoco_scene

    def build(objects=()):
        return build_b601_mujoco_scene(
            source_root,
            initial_arm_qpos=initial_arm_qpos,
            gravity_mps2=(0.0, 0.0, -9.81),
            actuation_mode="effort",
            scene_spec=b601_contact_scene_spec(objects=objects),
            enable_contact_observation=True,
        )

    scene = build()
    geom_id = scene.mujoco.mj_name2id(
        scene.model,
        scene.mujoco.mjtObj.mjOBJ_GEOM,
        "gripper_end_collision_0",
    )
    if (
        geom_id < 0
        or scene.model.geom_type[geom_id] != scene.mujoco.mjtGeom.mjGEOM_MESH
    ):
        raise ValueError("contact-box requires the pinned gripper collision mesh")
    mesh_id = int(scene.model.geom_dataid[geom_id])
    address = int(scene.model.mesh_vertadr[mesh_id])
    count = int(scene.model.mesh_vertnum[mesh_id])
    vertices = (
        scene.model.mesh_vert[address : address + count]
        @ scene.data.geom_xmat[geom_id].reshape(3, 3).T
        + scene.data.geom_xpos[geom_id]
    )
    top = vertices[np.argmax(vertices[:, 2])]
    box = FreeBodyObjectSpec(
        name="b601_contact_box",
        size_m=(0.02,) * 3,
        mass_kg=0.05,
        initial_position_m=tuple(float(v) for v in top + (0.0, 0.0, 0.0095)),
    )
    return build(objects=(box,)), box


def read_b601_contact_observation(
    scene: MujocoScene,
) -> B601MujocoContactObservation:
    """Read and shift the native net contact wrench to ``gripper_end``.

    MuJoCo's ``netforce`` contact result is world-oriented and expressed about
    its reported application point.  The returned moment is therefore shifted
    with ``r x f`` to the current ``gripper_end`` origin.  No scaling or
    filtering is applied here.  ``mj_step`` leaves derived fields at the
    pre-integration state, so forward dynamics is evaluated on an independent
    data copy before reading the sensor and pose.  The live scene is unchanged.
    Call from the scene-owner thread, not concurrently with a physics step.
    """

    if not isinstance(scene, MujocoScene):
        raise TypeError("scene must be a MujocoScene")
    sensor_id = int(
        scene.mujoco.mj_name2id(
            scene.model,
            scene.mujoco.mjtObj.mjOBJ_SENSOR,
            B601_RS_CONTACT_SENSOR_NAME,
        )
    )
    if sensor_id < 0:
        raise ValueError("B601 MuJoCo scene has no contact wrench sensor")
    sensor_dim = int(scene.model.sensor_dim[sensor_id])
    if sensor_dim != B601_RS_CONTACT_SENSOR_DIM:
        raise ValueError(
            "B601 contact wrench sensor dimension mismatch: "
            f"expected {B601_RS_CONTACT_SENSOR_DIM}, got {sensor_dim}"
        )
    sensor_address = int(scene.model.sensor_adr[sensor_id])

    def sensor_values(sensordata: np.ndarray) -> np.ndarray:
        values = np.asarray(
            sensordata[
                sensor_address : sensor_address + B601_RS_CONTACT_SENSOR_DIM
            ],
            dtype=float,
        ).copy()
        if values.shape != (B601_RS_CONTACT_SENSOR_DIM,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("B601 contact wrench sensor returned non-finite data")
        found = float(values[0])
        if found < 0.0 or not math.isclose(found, round(found), abs_tol=1e-9):
            raise ValueError("B601 contact wrench sensor returned an invalid count")
        return values

    # Do not conceal a corrupt cached result by recomputing it.
    sensor_values(scene.data.sensordata)
    if scene.mujoco.get_mjcb_control() is not None:
        raise ValueError("B601 contact observation requires no control callback")
    data = scene.mujoco.MjData(scene.model)
    scene.mujoco.mj_copyData(data, scene.model, scene.data)
    physics_time_sec = float(data.time)
    if not math.isfinite(physics_time_sec) or physics_time_sec < 0.0:
        raise ValueError("B601 MuJoCo physics time must be finite and non-negative")
    state_spec = scene.mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(scene.mujoco.mj_stateSize(scene.model, state_spec))
    scene.mujoco.mj_getState(scene.model, data, state, state_spec)
    if not np.all(np.isfinite(state)):
        raise ValueError("B601 contact observation requires finite integration state")
    # Never run the control callback or advance the live solver/integrator.
    scene.mujoco.mj_forward(scene.model, data)
    values = sensor_values(data.sensordata)

    matched_contact_count = round(float(values[0]))
    if matched_contact_count == 0:
        return B601MujocoContactObservation(
            wrench=B601SpatialWrench(
                force_n=(0.0, 0.0, 0.0),
                torque_nm=(0.0, 0.0, 0.0),
                frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
            ),
            physics_time_sec=physics_time_sec,
            matched_contact_count=0,
            application_point_world_m=None,
        )

    force_world = values[1:4]
    torque_at_contact_world = values[4:7]
    application_point_world = values[7:10]
    end_effector_position_world = np.asarray(
        data.xpos[scene.body_id(B601_RS_END_EFFECTOR_FRAME)], dtype=float
    )
    torque_at_end_effector_world = torque_at_contact_world + np.cross(
        application_point_world - end_effector_position_world,
        force_world,
    )
    return B601MujocoContactObservation(
        wrench=B601SpatialWrench(
            force_n=tuple(float(value) for value in force_world),
            torque_nm=tuple(float(value) for value in torque_at_end_effector_world),
            frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
        ),
        physics_time_sec=physics_time_sec,
        matched_contact_count=matched_contact_count,
        application_point_world_m=tuple(
            float(value) for value in application_point_world
        ),
    )


__all__ = [
    "B601_RS_CONTACT_SCENE_PROFILE_NAME",
    "B601_RS_CONTACT_SENSOR_DIM",
    "B601_RS_CONTACT_SENSOR_FIELDS",
    "B601_RS_CONTACT_SENSOR_NAME",
    "B601MujocoContactObservation",
    "add_b601_contact_wrench_sensor",
    "b601_contact_scene_spec",
    "read_b601_contact_observation",
]
