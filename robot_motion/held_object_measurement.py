"""Held-object box and grip metrics from measured surfaces, in the holding tool frame.

Hosts supply each view's measured object surface (eroded mask, largest cluster,
see ``robot_motion.tool_entity_scene.object_surface_from_mask``) together with
the holding arm's TCP at that view's capture qpos, and the reviewed gripper
entity with its closing region. This module performs no acquisition,
segmentation, kinematics or motion decision.

The box is axis-aligned with the TCP axes and covers every measured surface
point plus a margin. Two conservative completions bound geometry that no view
measured:

* along the closing axis the box is made symmetric about the closing-region
  centre, since the object is held between both pads;
* when the object rests on a measured support, every measured point is also
  dropped onto that support plane, so the hidden part down to the support is
  inside the box.

It remains a measured-surface bound, not a certified full shape: surfaces that
are hidden from every view and extend past these completions are not covered.
Semantics: ``agenticrobot.held_object.measured_box.v1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_motion.tool_entity_goals import (
    ToolEntityBox,
    fuse_tool_object_relations,
    object_surface_cluster,
    tool_object_relation,
)

HELD_OBJECT_BOX_SEMANTICS_V1 = "agenticrobot.held_object.measured_box.v1"
HELD_OBJECT_BOX_MARGIN_M = 0.008
HELD_OBJECT_MIN_POINTS = 20
HELD_OBJECT_MAX_EXTENT_M = 0.40
# Points farther than this from the TCP cannot belong to the held object
# (larger than any graspable object here); they are mask bleed or background.
HELD_OBJECT_MAX_TCP_DISTANCE_M = 0.30


@dataclass(frozen=True)
class HeldObjectView:
    """One view's measured surface of the held object and that frame's TCP."""

    role: str
    points_world_m: np.ndarray
    world_from_tcp: np.ndarray

    def __post_init__(self) -> None:
        points = np.array(self.points_world_m, dtype=float).reshape(-1, 3)
        pose = np.array(self.world_from_tcp, dtype=float).reshape(4, 4)
        if not len(points) or not np.isfinite(points).all() or not np.isfinite(pose).all():
            raise ValueError("A held-object view needs finite points and a finite TCP pose.")
        if (not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)):
            raise ValueError("A held-object view TCP pose must be rigid.")
        points.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, "points_world_m", points)
        object.__setattr__(self, "world_from_tcp", pose)


@dataclass(frozen=True)
class SupportPlane:
    """A measured support plane the object rests on when the box is measured."""

    anchor_world_m: tuple[float, float, float]
    normal_world: tuple[float, float, float]

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_world_m, dtype=float).reshape(-1)
        normal = np.asarray(self.normal_world, dtype=float).reshape(-1)
        if anchor.shape != (3,) or normal.shape != (3,) or not np.isfinite([*anchor, *normal]).all():
            raise ValueError("Support plane needs a finite anchor and normal.")
        length = float(np.linalg.norm(normal))
        if length < 1e-9 or normal[2] / length < math.cos(math.radians(15.0)):
            raise ValueError("Support plane normal must point up within 15 degrees.")
        object.__setattr__(self, "anchor_world_m", tuple(float(v) for v in anchor))
        object.__setattr__(self, "normal_world", tuple(float(v) for v in normal / length))


@dataclass(frozen=True)
class HeldObjectBox:
    """Measured box fixed in the holding tool frame, with the fused relation and grip metrics."""

    status: str
    reason: str
    tool_from_box: np.ndarray | None
    size_xyz_m: tuple[float, float, float] | None
    relation: dict
    view_relations: tuple[dict, ...]
    metrics: dict

    @property
    def held(self) -> bool:
        return self.status == "held"

    def as_dict(self) -> dict:
        return {"semantics": HELD_OBJECT_BOX_SEMANTICS_V1, "status": self.status, "reason": self.reason,
                "tool_from_box": None if self.tool_from_box is None else self.tool_from_box.tolist(),
                "size_xyz_m": None if self.size_xyz_m is None else list(self.size_xyz_m),
                "relation": dict(self.relation), "view_relations": [dict(v) for v in self.view_relations],
                "metrics": dict(self.metrics)}


def _tcp_points(view: HeldObjectView) -> np.ndarray:
    pose = view.world_from_tcp
    return (view.points_world_m - pose[:3, 3]) @ pose[:3, :3]


def measure_held_object_box(*, views: Sequence[HeldObjectView], entity: ToolEntityBox,
                            support_plane: SupportPlane | None = None,
                            margin_m: float = HELD_OBJECT_BOX_MARGIN_M) -> HeldObjectBox:
    """Fuse the views into one tool-frame box and report whether the object is between the fingers.

    ``status`` is ``held`` (between the fingers in at least one usable view and
    a bounded box), ``not_between_fingers`` (measured, but not in the closing
    region: dropped, pushed away or never grasped) or ``insufficient_surface``.
    Grip metrics: offset of the measured centroid from the closing-region
    centre along the closing and pad axes, and the angle between the object's
    principal axis and the nearest TCP axis (crooked grasp evidence).
    """
    if entity.grasp_region is None or not entity.parts:
        raise ValueError("Held-object measurement needs the gripper closing region.")
    if not math.isfinite(margin_m) or not 0.0 <= margin_m <= 0.05:
        raise ValueError("Held-object box margin must be within [0, 50 mm].")
    rotation = np.asarray(entity.tcp_from_box_rotation, dtype=float).reshape(3, 3)
    region_centre = np.asarray(entity.grasp_region[0], dtype=float)
    closing_axis, pad_axis = rotation[:, 0], rotation[:, 1]
    relations, kept, view_rows = [], [], []
    for view in views:
        local = _tcp_points(view)
        near = local[np.linalg.norm(local, axis=1) <= HELD_OBJECT_MAX_TCP_DISTANCE_M]
        row = {"role": view.role, "input_point_count": len(local), "near_tcp_point_count": len(near)}
        if len(near) < 3:
            row["relation"] = "insufficient_surface"
            view_rows.append(row)
            continue
        surface, cluster = object_surface_cluster(near)
        world = surface @ view.world_from_tcp[:3, :3].T + view.world_from_tcp[:3, 3]
        relation = tool_object_relation(entity=entity, world_from_tcp=view.world_from_tcp, object_points_m=world)
        relations.append(relation)
        kept.append(surface)
        row.update(relation.as_dict(), kept_voxel_count=cluster["kept_voxel_count"],
                   component_count=cluster["component_count"])
        view_rows.append(row)
    fused = (fuse_tool_object_relations(relations) if relations else
             {"relation": "insufficient_surface", "view_count": len(views), "usable_view_count": 0})
    if not kept or sum(len(points) for points in kept) < HELD_OBJECT_MIN_POINTS:
        return HeldObjectBox(status="insufficient_surface", reason="held_object_surface_insufficient",
                             tool_from_box=None, size_xyz_m=None, relation=fused,
                             view_relations=tuple(view_rows), metrics={})
    points = np.concatenate(kept)
    metrics = {"measured_point_count": len(points),
               "views_between_fingers": [row["role"] for row in view_rows
                                         if str(row.get("relation", "")).startswith("between_fingers")]}
    centroid = points.mean(axis=0)
    metrics["centroid_offset_closing_axis_m"] = float((centroid - region_centre) @ closing_axis)
    metrics["centroid_offset_pad_axis_m"] = float((centroid - region_centre) @ pad_axis)
    if len(points) >= 10:
        _, values, vectors = np.linalg.svd(points - centroid, full_matrices=False)
        principal = vectors[0]
        cosines = np.abs(principal)
        nearest_axis = int(np.argmax(cosines))
        elongation = float(values[0] / max(values[1], 1e-9))
        metrics["principal_axis_elongation"] = elongation
        # The principal axis of a compact surface (cube, ball) is noise; report
        # its tilt only for elongated objects (bottles, bars).
        if elongation >= 1.5:
            metrics["principal_axis_tcp"] = [float(v) for v in principal]
            metrics["principal_axis_nearest_tcp_axis"] = "xyz"[nearest_axis]
            metrics["principal_axis_tilt_deg"] = float(math.degrees(math.acos(min(1.0, cosines[nearest_axis]))))
    if not str(fused["relation"]).startswith("between_fingers"):
        return HeldObjectBox(status="not_between_fingers", reason="held_object_not_between_fingers",
                             tool_from_box=None, size_xyz_m=None, relation=fused,
                             view_relations=tuple(view_rows), metrics=metrics)
    covered = [points]
    if support_plane is not None:
        # The object rests on the support: its hidden lower part reaches the plane.
        anchor = np.asarray(support_plane.anchor_world_m)
        normal = np.asarray(support_plane.normal_world)
        pose = views[0].world_from_tcp
        world = points @ pose[:3, :3].T + pose[:3, 3]
        dropped = world - np.outer((world - anchor) @ normal, normal)
        covered.append((dropped - pose[:3, 3]) @ pose[:3, :3])
        metrics["support_completion"] = True
    allpoints = np.concatenate(covered)
    low, high = allpoints.min(axis=0), allpoints.max(axis=0)
    # Symmetric about the closing-region centre along the closing axis.
    axis_index = int(np.argmax(np.abs(closing_axis)))
    centre = float(region_centre[axis_index])
    half = max(abs(low[axis_index] - centre), abs(high[axis_index] - centre))
    low[axis_index], high[axis_index] = centre - half, centre + half
    low, high = low - margin_m, high + margin_m
    size = high - low
    metrics.update(box_margin_m=float(margin_m), closing_axis_tcp_index=axis_index)
    if np.any(size > HELD_OBJECT_MAX_EXTENT_M):
        return HeldObjectBox(status="insufficient_surface", reason="held_object_box_too_large",
                             tool_from_box=None, size_xyz_m=None, relation=fused,
                             view_relations=tuple(view_rows), metrics={**metrics, "size_xyz_m": size.tolist()})
    tool_from_box = np.eye(4)
    tool_from_box[:3, 3] = (low + high) / 2.0
    tool_from_box.setflags(write=False)
    return HeldObjectBox(status="held", reason="held_object_between_fingers", tool_from_box=tool_from_box,
                         size_xyz_m=tuple(float(v) for v in size), relation=fused,
                         view_relations=tuple(view_rows), metrics=metrics)


__all__ = ["HELD_OBJECT_BOX_SEMANTICS_V1", "HeldObjectBox", "HeldObjectView", "SupportPlane",
           "measure_held_object_box"]
