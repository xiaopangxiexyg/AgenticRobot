"""Object-centric transport inputs. Observations do not grant motion authority.

An object frame is supplied by an observation owner (or simulator), not inferred
from a grasp score or an axis-aligned point-cloud bounding box. Partial geometry
cannot certify the swept volume of a held object.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from robot_motion.pose import pose_delta_metrics


def rigid_pose(value, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-8, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    result = pose.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GraspedObjectObservation:
    """One correlated post-grasp cut, supplied only by the bound observer.

    ``grasp_evidence_ref`` must identify physical grasp verification, not a
    successful close command. ``reference_from_object`` describes a stable
    material frame; unobservable/symmetric axes cannot be invented.
    """

    object_id: str
    side: str
    reference_frame_id: str
    scene_revision: int
    measured_at: float
    reference_from_object: np.ndarray
    reference_from_tool: np.ndarray
    qpos_by_side: Mapping[str, np.ndarray]
    box_size_m: tuple[float, float, float]
    object_from_box: np.ndarray
    geometry_complete: bool
    pose_evidence_ref: str
    grasp_evidence_ref: str

    def __post_init__(self):
        for name in ("object_id", "reference_frame_id", "pose_evidence_ref", "grasp_evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must identify observed evidence")
        if self.side not in {"left", "right"}:
            raise ValueError("Invalid grasping side")
        if (isinstance(self.scene_revision, bool) or not isinstance(self.scene_revision, int)
                or self.scene_revision < 0 or not np.isfinite(self.measured_at)):
            raise ValueError("Invalid observation revision or time")
        if self.geometry_complete is not True:
            raise ValueError("Transport requires a complete conservative object geometry")
        size = np.asarray(self.box_size_m, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Object box dimensions must be positive finite XYZ")
        object.__setattr__(self, "box_size_m", tuple(float(x) for x in size))
        for name in ("reference_from_object", "reference_from_tool", "object_from_box"):
            object.__setattr__(self, name, rigid_pose(getattr(self, name), name=name))
        if set(self.qpos_by_side) != {"left", "right"}:
            raise ValueError("A transport observation requires both arm states")
        frozen = {}
        for side, values in self.qpos_by_side.items():
            array = np.asarray(values, dtype=float).copy()
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError("Invalid measured qpos")
            array.setflags(write=False)
            frozen[side] = array
        object.__setattr__(self, "qpos_by_side", MappingProxyType(frozen))

    @property
    def tool_from_object(self) -> np.ndarray:
        return np.linalg.inv(self.reference_from_tool) @ self.reference_from_object

    def tool_goal(self, reference_from_object_goal: np.ndarray) -> np.ndarray:
        goal = rigid_pose(reference_from_object_goal, name="object goal")
        return goal @ np.linalg.inv(self.tool_from_object)

    def require_current(self, *, now: float, scene_revision: int, reference_frame_id: str,
                        object_id: str, side: str, max_age_sec: float = .2) -> None:
        if (not np.isfinite(now) or not np.isfinite(max_age_sec) or max_age_sec <= 0
                or not 0 <= now-self.measured_at <= max_age_sec):
            raise ValueError("Object/hand observation is stale or from the future")
        if (self.object_id, self.side, self.reference_frame_id, self.scene_revision) != (
                object_id, side, reference_frame_id, scene_revision):
            raise ValueError("Object/hand observation identity does not match the request")


def verify_rigid_follow(previous: GraspedObjectObservation, current: GraspedObjectObservation,
                        *, translation_tolerance_m: float = .003,
                        rotation_tolerance_deg: float = 2.) -> dict[str, float]:
    """Verify observed hand/object relation after a separately authorized motion.

    This function never creates the probing motion. Static images and a close
    receipt cannot demonstrate that an object follows the gripper.
    """
    if (previous.object_id, previous.side, previous.reference_frame_id) != (
            current.object_id, current.side, current.reference_frame_id):
        raise ValueError("Rigid-follow observations describe different objects/frames")
    if (current.measured_at <= previous.measured_at
            or previous.box_size_m != current.box_size_m
            or not np.allclose(previous.object_from_box, current.object_from_box, atol=1e-8, rtol=0)):
        raise ValueError("Rigid-follow observation time or geometry changed")
    for limit in (translation_tolerance_m, rotation_tolerance_deg):
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Rigid-follow tolerances must be positive")
    displacement, angle = pose_delta_metrics(previous.reference_from_tool, current.reference_from_tool)
    if displacement < .003 and angle < 1.:
        raise ValueError("Static observations do not prove rigid following")
    slip, slip_angle = pose_delta_metrics(previous.tool_from_object, current.tool_from_object)
    if slip > translation_tolerance_m or slip_angle > rotation_tolerance_deg:
        raise ValueError("Object slipped relative to the gripper; reobserve and replan")
    return {"tool_displacement_m": displacement, "tool_rotation_deg": angle,
            "relative_translation_change_m": slip, "relative_rotation_change_deg": slip_angle}
