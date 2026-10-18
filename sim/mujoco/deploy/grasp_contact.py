"""Target-specific terminal contact; no authority, controller or physics owner."""

import math
from dataclasses import dataclass

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


@dataclass(frozen=True)
class MujocoGraspContactBinding:
    target_body_name: str
    sensor_geom_names: tuple[str, ...]
    max_approach_m: float = 0.04
    lateral_tolerance_m: float = 0.003
    rotation_tolerance_rad: float = math.radians(2)
    max_penetration_m: float = 0.001

    def __post_init__(self):
        if (
            not isinstance(self.target_body_name, str)
            or not self.target_body_name.strip()
            or self.target_body_name != self.target_body_name.strip()
            or not self.sensor_geom_names
        ):
            raise ValueError(
                "Grasp contact requires a target and explicit sensor geoms."
            )
        object.__setattr__(self, "sensor_geom_names", tuple(self.sensor_geom_names))
        if any(
            not isinstance(n, str) or not n.strip() or n != n.strip()
            for n in self.sensor_geom_names
        ):
            raise ValueError("Grasp contact sensor geoms must be non-empty names.")
        if len(set(self.sensor_geom_names)) != len(self.sensor_geom_names):
            raise ValueError("Grasp contact sensor geoms must be unique.")
        for value in (
            self.max_approach_m,
            self.lateral_tolerance_m,
            self.rotation_tolerance_rad,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Grasp contact corridor bounds must be positive and finite."
                )
        if not math.isfinite(self.max_penetration_m) or self.max_penetration_m < 0:
            raise ValueError(
                "Grasp contact penetration bound must be finite and non-negative."
            )

    def to_payload(self):
        return {
            "target_body_name": self.target_body_name,
            "sensor_geom_names": self.sensor_geom_names,
            "max_approach_m": self.max_approach_m,
            "lateral_tolerance_m": self.lateral_tolerance_m,
            "rotation_tolerance_rad": self.rotation_tolerance_rad,
            "max_penetration_m": self.max_penetration_m,
        }

    @property
    def semantic_content_hash(self):
        return canonical_content_hash(
            self.to_payload(), domain="mujoco.grasp_contact.v1"
        )

    def pair_clearances(self, scene):
        model = scene.model
        projector = scene.collision_projector()
        if projector is None:
            raise ValueError(
                "Grasp contact requires the existing scene collision checker."
            )
        if self.max_penetration_m > projector.policy.hard_penetration_m:
            raise ValueError(
                "Target contact cannot exceed the scene hard penetration bound."
            )
        target = model.body(self.target_body_name).id
        objects = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == target and model.geom_contype[i] == 8
        ]
        if not objects:
            raise ValueError("Grasp contact target must own object collision geoms.")
        sensors = [model.geom(name).id for name in self.sensor_geom_names]
        if any(model.geom_contype[i] != 4 for i in sensors):
            raise ValueError("Grasp contact sensors must be hand collision geoms.")
        return {
            tuple(sorted((sensor, obj))): -self.max_penetration_m
            for sensor in sensors
            for obj in objects
        }

    def in_corridor(self, current, target):
        local = target[:3, :3].T @ (current[:3, 3] - target[:3, 3])
        angle = math.acos(
            float(
                np.clip((np.trace(current[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)
            )
        )
        return (
            np.linalg.norm(local[:2]) <= self.lateral_tolerance_m
            and -self.max_approach_m <= local[2] <= self.lateral_tolerance_m
            and angle <= self.rotation_tolerance_rad
        )


def bind_task_grasp_contact(scene, gripper_profile, target_body_name):
    """Resolve only the task profile's explicit right-hand sensor/object pairs."""
    model = scene.model
    target = model.body(target_body_name).id
    target_geoms = {
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == target and model.geom_contype[i] == 8
    }
    hands = [
        name
        for name, side in gripper_profile.side_by_gripper.items()
        if side == "right"
    ]
    if len(hands) != 1:
        raise ValueError(
            "Terminal contact currently requires one configured right hand."
        )
    master_body = model.jnt_bodyid[
        model.joint(gripper_profile.master_joint_by_gripper[hands[0]]).id
    ]
    # All fingers share the palm parent of the configured master knuckle.
    palm = model.body_parentid[master_body]
    names = []
    for pair in gripper_profile.contact_tuning.geom_pairs:
        if not target_geoms.intersection(pair):
            continue
        for name in pair:
            if name in target_geoms:
                continue
            geom = model.geom(name).id
            body = model.geom_bodyid[geom]
            while body and body != palm:
                body = model.body_parentid[body]
            if body == palm and model.geom_contype[geom] == 4:
                names.append(name)
    binding = MujocoGraspContactBinding(
        target_body_name,
        tuple(sorted(set(names))),
        max_penetration_m=scene.collision_projector().policy.hard_penetration_m,
    )
    binding.pair_clearances(scene)
    return binding
