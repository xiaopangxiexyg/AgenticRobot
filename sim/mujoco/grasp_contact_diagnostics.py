"""Read-only native contact diagnostics, never reward or motion admission.

A sensor body or an inward collision-box face is not automatically a validated
tactile pad. Keep that distinction explicit until a finite pad contract exists.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .grasp_contact import GripperContactSurfaceSpec, classify_gripper_contact_point


class MujocoGraspContactSampler:
    """Inspect measured contacts without stepping or modifying a native scene."""

    def __init__(
        self,
        scene: Any,
        *,
        object_body_name: str,
        surface_specs: tuple[GripperContactSurfaceSpec, ...],
        minimum_normal_force_n: float = 0.01,
        box_face_tolerance_m: float = 0.001,
    ) -> None:
        for name, value in (("minimum_normal_force_n", minimum_normal_force_n),
                            ("box_face_tolerance_m", box_face_tolerance_m)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not surface_specs or len({s.side for s in surface_specs}) != len(surface_specs):
            raise ValueError("Contact diagnostics require distinct finger sides.")
        if len({s.body_name for s in surface_specs}) != len(surface_specs):
            raise ValueError("Contact diagnostics require distinct finger bodies.")
        self.scene = scene
        self.object_body_id = scene.body_id(object_body_name)
        self.specs = {scene.body_id(s.body_name): s for s in surface_specs}
        if self.object_body_id in self.specs:
            raise ValueError("The observed object cannot be a finger body.")
        self.minimum_normal_force_n = float(minimum_normal_force_n)
        self.box_face_tolerance_m = float(box_face_tolerance_m)
        model, mj = scene.model, scene.mujoco
        self.body_names = tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                                for i in range(model.nbody))
        self.geom_names = tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
                                for i in range(model.ngeom))

    def sample(self) -> tuple[dict[str, Any], ...]:
        scene = self.scene
        model, data, mj = scene.model, scene.data, scene.mujoco
        records = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            if self.object_body_id not in (body1, body2):
                continue
            other_geom, other_body = ((geom2, body2) if body1 == self.object_body_id
                                      else (geom1, body1))
            force = np.zeros(6)
            active = int(contact.efc_address) >= 0
            if active:
                mj.mj_contactForce(model, data, index, force)
            position = np.asarray(contact.pos, dtype=float).reshape(3)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            normal = frame[0]
            if not (np.isfinite(force).all() and np.isfinite(position).all()
                    and np.isfinite(frame).all() and np.isfinite(contact.dist)):
                raise ValueError("Native contact diagnostic contains nonfinite data.")
            # mjContact.pos is the midpoint, not either surface. Signed distance
            # and the geom1->geom2 normal recover the other geom's surface point,
            # including penetration. Do not inflate the box tolerance to hide it.
            surface_sign = -1. if other_geom == geom1 else 1.
            surface_point = position + surface_sign * .5 * float(contact.dist) * normal
            body_rotation = np.asarray(data.xmat[other_body]).reshape(3, 3)
            local_point = body_rotation.T @ (surface_point - data.xpos[other_body])
            spec = self.specs.get(other_body)
            region = "not_configured_finger"
            box_evidence = None
            inner_score = outer_score = None
            if spec is not None:
                inner_score, outer_score = classify_gripper_contact_point(
                    surface_point, data.xpos[other_body], body_rotation, spec)
                region = "finger_body_without_finite_pad_contract"
                if int(model.geom_type[other_geom]) == int(mj.mjtGeom.mjGEOM_BOX):
                    geom_rotation = np.asarray(data.geom_xmat[other_geom]).reshape(3, 3)
                    geom_point = geom_rotation.T @ (surface_point - data.geom_xpos[other_geom])
                    half = np.asarray(model.geom_size[other_geom], dtype=float)
                    residuals = np.abs(np.abs(geom_point) - half)
                    axis = int(np.argmin(residuals))
                    sign = 1.0 if geom_point[axis] >= 0 else -1.0
                    normal_local = np.zeros(3)
                    normal_local[axis] = sign
                    expected = body_rotation @ np.asarray(spec.inner_normal_local)
                    alignment = float((geom_rotation @ normal_local) @ expected)
                    edge = int(np.count_nonzero(residuals <= self.box_face_tolerance_m)) > 1
                    on_box = bool(residuals[axis] <= self.box_face_tolerance_m
                                  and np.all(np.abs(geom_point) <= half + self.box_face_tolerance_m))
                    region = ("collision_box_surface_unresolved" if not on_box
                              else "collision_box_edge" if edge
                              else "inward_collision_box_face" if alignment >= 0.9
                              else "other_collision_box_face")
                    box_evidence = {
                        "point_geom_local_m": geom_point.tolist(),
                        "nearest_face_axis": axis,
                        "nearest_face_sign": int(sign),
                        "face_distance_m": float(residuals[axis]),
                        "inward_normal_alignment": alignment,
                    }
            records.append({
                "geom1": self.geom_names[geom1], "geom2": self.geom_names[geom2],
                "other_body": self.body_names[other_body],
                "other_geom": self.geom_names[other_geom],
                "finger_side": None if spec is None else spec.side,
                "position_world_m": position.tolist(),
                "surface_point_other_world_m": surface_point.tolist(),
                "point_other_body_local_m": local_point.tolist(),
                "normal_geom1_to_geom2_world": normal.tolist(),
                "contact_frame_rows_world": frame.tolist(),
                "wrench_convention": "force:torque on geom2 at contact midpoint; opposite on geom1",
                "force_on_object_world_n": ((1. if body2 == self.object_body_id else -1.)
                    * (frame.T @ force[:3])).tolist(),
                "torque_on_object_at_contact_world_nm": ((1. if body2 == self.object_body_id else -1.)
                    * (frame.T @ force[3:])).tolist(),
                "distance_m": float(contact.dist), "active_constraint": active,
                "force_contact_frame": force.tolist(), "normal_force_n": float(force[0]),
                "load_bearing": bool(active and force[0] > self.minimum_normal_force_n),
                "region": region, "collision_box_evidence": box_evidence,
                "inner_half_space_score": inner_score, "outer_half_space_score": outer_score,
                "physical_tactile_pad_membership": "unknown",
            })
        return tuple(records)


__all__ = ["MujocoGraspContactSampler"]
