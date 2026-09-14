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


def read_b601_contact_observation(
    scene: MujocoScene,
) -> B601MujocoContactObservation:
    """Read and shift the native net contact wrench to ``gripper_end``.

    MuJoCo's ``netforce`` contact result is world-oriented and expressed about
    its reported application point.  The returned moment is therefore shifted
    with ``r x f`` to the current ``gripper_end`` origin.  No scaling or
    filtering is applied here.
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
    values = np.asarray(
        scene.data.sensordata[
            sensor_address : sensor_address + B601_RS_CONTACT_SENSOR_DIM
        ],
        dtype=float,
    ).copy()
    if values.shape != (B601_RS_CONTACT_SENSOR_DIM,) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("B601 contact wrench sensor returned non-finite data")
    physics_time_sec = float(scene.data.time)
    if not math.isfinite(physics_time_sec) or physics_time_sec < 0.0:
        raise ValueError("B601 MuJoCo physics time must be finite and non-negative")

    found = float(values[0])
    matched_contact_count = round(found)
    if (
        found < 0.0
        or not math.isclose(found, matched_contact_count, abs_tol=1e-9)
    ):
        raise ValueError("B601 contact wrench sensor returned an invalid count")
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
    end_effector_position_world = scene.body_pose(B601_RS_END_EFFECTOR_FRAME)[:3, 3]
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
