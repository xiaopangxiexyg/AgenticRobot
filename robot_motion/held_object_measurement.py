"""Held-object box and grip metrics from measured surfaces, in the holding tool frame.

Hosts supply each view's measured object surface (eroded mask, largest cluster,
see ``robot_motion.tool_entity_scene.object_surface_from_mask``) together with
the holding arm's TCP at that view's capture qpos, and the reviewed gripper
entity with its closing region. This module performs no acquisition,
segmentation, kinematics or motion decision.

The primary view is the holding arm's own wrist camera. It is rigid with the
gripper, so only its hand-eye calibration places the closing region in its
image; every other camera adds its station calibration. On the saved real
samples those cameras disagreed on the table height by 1.5-3 cm, which moved
the gripper's own fingers into the closing region of the other views. Hence:

* whether something is between the fingers is decided by the primary view;
* another view only extends a primary measurement, and only within
  ``HELD_OBJECT_VIEW_EXTENSION_M`` of the primary surface's box on every TCP
  axis, so its calibration error, a finger in its mask or the support the
  object stands on cannot grow the box further;
* grip metrics (centroid offsets, principal axis) use the primary surface.

The box is axis-aligned with the TCP axes and covers the primary surface, the
accepted extension points and a margin. Two conservative completions bound
geometry that no view measured:

* along the closing axis the box is made symmetric about the closing-region
  centre, since the object is held between both pads;
* when the object rests on a measured support, every measured point is also
  dropped onto that support plane, so the hidden part down to the support is
  inside the box.

It remains a measured-surface bound, not a certified full shape: surfaces that
are hidden from every view and extend past these completions are not covered.
Semantics: ``agenticrobot.held_object.measured_box.v2``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robot_motion.grasp_visible_scene import (
    VisibleGraspDepthFilterConfig,
    filter_isolated_background_depth,
)
from robot_motion.tool_entity_goals import (
    ToolEntityBox,
    _signed_box_distance,
    fuse_tool_object_relations,
    object_surface_cluster,
    tool_object_relation,
)

HELD_OBJECT_BOX_SEMANTICS_V2 = "agenticrobot.held_object.measured_box.v2"
HELD_OBJECT_BOX_MARGIN_M = 0.008
HELD_OBJECT_MIN_POINTS = 20
HELD_OBJECT_MAX_EXTENT_M = 0.40
# Points farther than this from the TCP cannot belong to the held object
# (larger than any graspable object here); they are mask bleed or background.
HELD_OBJECT_MAX_TCP_DISTANCE_M = 0.30
# Another view may grow the primary surface's box by at most this much per side.
HELD_OBJECT_VIEW_EXTENSION_M = 0.03


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
        return {"semantics": HELD_OBJECT_BOX_SEMANTICS_V2, "status": self.status, "reason": self.reason,
                "tool_from_box": None if self.tool_from_box is None else self.tool_from_box.tolist(),
                "size_xyz_m": None if self.size_xyz_m is None else list(self.size_xyz_m),
                "relation": dict(self.relation), "view_relations": [dict(v) for v in self.view_relations],
                "metrics": dict(self.metrics)}


def _tcp_points(view: HeldObjectView) -> np.ndarray:
    pose = view.world_from_tcp
    return (view.points_world_m - pose[:3, 3]) @ pose[:3, :3]


HELD_OBJECT_SEED_REGION_SHRINK_M = 0.003
# Seed points also keep this clearance from every gripper solid (palm, finger
# screening solids, URDF sensor boxes at the open gripper): on the saved real
# samples the open fingers' inner edges reached up to 1 mm into the shrunk region.
HELD_OBJECT_SEED_SOLID_CLEARANCE_M = 0.005
HELD_OBJECT_SEED_MINIMUM_PIXELS = 15
# Seeds use only depth-edge-free pixels: all 8 neighbours valid and within
# 12 mm. Flying pixels between a finger edge and the background never qualify.
HELD_OBJECT_SEED_DEPTH_EDGE_FILTER = VisibleGraspDepthFilterConfig(
    neighborhood_radius_px=1, maximum_neighbor_depth_delta_m=0.012, minimum_supporting_neighbors=8)
# Another view's seed: a projected primary surface point whose measured depth
# in that view agrees within this (it sees that surface, not an occluder in
# front of it or the background behind it).
HELD_OBJECT_VIEW_DEPTH_AGREEMENT_M = 0.03


def _depth_edge_free(depth: np.ndarray, maximum_depth_m: float) -> np.ndarray:
    """Valid depth pixels away from depth edges (``HELD_OBJECT_SEED_DEPTH_EDGE_FILTER``)."""
    valid = np.isfinite(depth) & (depth > 0) & (depth <= maximum_depth_m)
    keep, _ = filter_isolated_background_depth(
        depth_m=np.where(valid, depth, 0.0), target_mask=np.zeros(depth.shape, dtype=bool),
        config=HELD_OBJECT_SEED_DEPTH_EDGE_FILTER)
    return keep


def gripper_solid_distance_m(entity: ToolEntityBox, points_tcp_m) -> np.ndarray:
    """Smallest signed distance (negative inside) of TCP-frame points to every gripper solid.

    The solids are the entity's palm and finger screening solids and its
    contact-only solids, all at the open gripper.
    """
    points = np.asarray(points_tcp_m, dtype=float).reshape(-1, 3)
    rotation = np.asarray(entity.tcp_from_box_rotation, dtype=float).reshape(3, 3)
    local = points @ rotation
    distances = [_signed_box_distance(local, np.asarray(centre) @ rotation, half) for centre, half in entity.parts]
    for pose, half in entity.contact_solids:
        transform = np.asarray(pose, dtype=float).reshape(4, 4)
        distances.append(_signed_box_distance((points - transform[:3, 3]) @ transform[:3, :3], np.zeros(3), half))
    return np.min(distances, axis=0)


def _nearest_centroid_pixel(cols: np.ndarray, rows: np.ndarray) -> tuple[int, int]:
    # The pixel nearest the image centroid of the candidates lies inside the
    # visible object part, away from its depth-bleeding boundary.
    pixels = np.column_stack((cols, rows)).astype(float)
    best = int(np.argmin(np.linalg.norm(pixels - pixels.mean(axis=0), axis=1)))
    return int(cols[best]), int(rows[best])


def grip_seed_pixel(*, depth_m, intrinsics, world_from_camera, world_from_tcp, entity: ToolEntityBox,
                    maximum_depth_m: float, stride_px: int = 2) -> tuple[tuple[int, int] | None, dict]:
    """Pixel showing the held object between the pads, for a point-prompt segmentation.

    Meant for the holding arm's own wrist camera. Back-projects the view's
    valid depth (every ``stride_px`` pixel) and keeps pixels whose 3D point
    lies inside the gripper's closing region shrunk by 3 mm, at least
    ``HELD_OBJECT_SEED_SOLID_CLEARANCE_M`` from every gripper solid and away
    from depth edges. Returns the one nearest their image centroid and the
    evidence counts (``closing_region_pixel_count`` kept, and the pixels of
    the shrunk region dropped as depth edges or as too close to a gripper
    solid). ``None`` when fewer than ``HELD_OBJECT_SEED_MINIMUM_PIXELS``
    pixels see anything between the pads (empty gripper, dropped object or
    occluded view).
    """
    if entity.grasp_region is None:
        raise ValueError("Seed selection needs the gripper closing region.")
    depth = np.asarray(depth_m, dtype=float)
    interior = _depth_edge_free(depth, maximum_depth_m)
    rows, cols = np.mgrid[0:depth.shape[0]:stride_px, 0:depth.shape[1]:stride_px]
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z > 0) & (z <= maximum_depth_m)
    evidence = {"closing_region_pixel_count": 0, "depth_edge_pixel_count": 0, "gripper_solid_pixel_count": 0}
    if not np.any(valid):
        return None, evidence
    rows, cols, z = rows[valid], cols[valid], z[valid]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    camera = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
    pose = np.asarray(world_from_camera, dtype=float).reshape(4, 4)
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    tcp = np.asarray(world_from_tcp, dtype=float).reshape(4, 4)
    local = (world - tcp[:3, 3]) @ tcp[:3, :3]
    rotation = np.asarray(entity.tcp_from_box_rotation, dtype=float).reshape(3, 3)
    centre, half = (np.asarray(value, dtype=float) for value in entity.grasp_region)
    shrunk = np.maximum(half - HELD_OBJECT_SEED_REGION_SHRINK_M, 1e-4)
    in_region = np.all(np.abs((local - centre) @ rotation) < shrunk, axis=1)
    on_edge = in_region & ~interior[rows, cols]
    near_solid = in_region & ~on_edge
    near_solid[near_solid] = gripper_solid_distance_m(entity, local[near_solid]) <= HELD_OBJECT_SEED_SOLID_CLEARANCE_M
    candidates = in_region & ~on_edge & ~near_solid
    count = int(np.count_nonzero(candidates))
    evidence = {"closing_region_pixel_count": count, "depth_edge_pixel_count": int(np.count_nonzero(on_edge)),
                "gripper_solid_pixel_count": int(np.count_nonzero(near_solid))}
    if count < HELD_OBJECT_SEED_MINIMUM_PIXELS:
        return None, evidence
    return _nearest_centroid_pixel(cols[candidates], rows[candidates]), evidence


def projected_seed_pixel(*, depth_m, intrinsics, world_from_camera, points_world_m,
                         maximum_depth_m: float) -> tuple[tuple[int, int] | None, dict]:
    """Pixel of another view showing the object the primary view measured.

    Projects the primary surface points into this view and keeps the pixels
    they land on that have depth-edge-free measured depth within
    ``HELD_OBJECT_VIEW_DEPTH_AGREEMENT_M`` of the projected depth. Returns
    the agreeing pixel nearest their image centroid and the number of
    agreeing pixels (``projected_object_pixel_count``); ``None`` when fewer
    than ``HELD_OBJECT_SEED_MINIMUM_PIXELS`` agree (this view does not see the
    object, or sees an occluder in front of it).
    """
    depth = np.asarray(depth_m, dtype=float)
    points = np.asarray(points_world_m, dtype=float).reshape(-1, 3)
    pose = np.asarray(world_from_camera, dtype=float).reshape(4, 4)
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    camera = camera[camera[:, 2] > 1e-3]
    evidence = {"projected_object_pixel_count": 0}
    if not len(camera):
        return None, evidence
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    cols = np.rint(camera[:, 0] / camera[:, 2] * fx + cx).astype(int)
    rows = np.rint(camera[:, 1] / camera[:, 2] * fy + cy).astype(int)
    inside = (cols >= 0) & (cols < depth.shape[1]) & (rows >= 0) & (rows < depth.shape[0])
    cols, rows, expected = cols[inside], rows[inside], camera[inside, 2]
    interior = _depth_edge_free(depth, maximum_depth_m)
    agree = interior[rows, cols] & (np.abs(depth[rows, cols] - expected) <= HELD_OBJECT_VIEW_DEPTH_AGREEMENT_M)
    pixels = np.unique(np.column_stack((cols[agree], rows[agree])), axis=0)
    evidence["projected_object_pixel_count"] = len(pixels)
    if len(pixels) < HELD_OBJECT_SEED_MINIMUM_PIXELS:
        return None, evidence
    return _nearest_centroid_pixel(pixels[:, 0], pixels[:, 1]), evidence


def _view_surface(view: HeldObjectView):
    """Near-TCP largest surface cluster of one view in TCP coordinates, with its evidence row."""
    local = _tcp_points(view)
    near = local[np.linalg.norm(local, axis=1) <= HELD_OBJECT_MAX_TCP_DISTANCE_M]
    row = {"role": view.role, "input_point_count": len(local), "near_tcp_point_count": len(near)}
    if len(near) < 3:
        return None, row
    surface, cluster = object_surface_cluster(near)
    row.update(kept_voxel_count=cluster["kept_voxel_count"], component_count=cluster["component_count"])
    return surface, row


def _closing_axis_box(points: np.ndarray, *, region_centre, axis_index: int, margin_m: float):
    """(low, high) of the points, symmetric about the closing-region centre along the closing axis, plus margin."""
    low, high = points.min(axis=0), points.max(axis=0)
    centre = float(region_centre[axis_index])
    half = max(abs(low[axis_index] - centre), abs(high[axis_index] - centre))
    low[axis_index], high[axis_index] = centre - half, centre + half
    return low - margin_m, high + margin_m


def measure_held_object_box(*, views: Sequence[HeldObjectView], entity: ToolEntityBox, primary_role: str,
                            support_plane: SupportPlane | None = None,
                            margin_m: float = HELD_OBJECT_BOX_MARGIN_M,
                            extension_m: float = HELD_OBJECT_VIEW_EXTENSION_M) -> HeldObjectBox:
    """Fuse the views into one tool-frame box and report whether the object is between the fingers.

    ``primary_role`` names the holding arm's own wrist view; it alone decides
    ``status``: ``held`` (its surface is between the fingers and the box is
    bounded), ``not_between_fingers`` (measured, but not in the closing
    region: dropped, pushed away or never grasped) or ``insufficient_surface``
    (no primary surface). Every other view contributes only its points within
    ``extension_m`` of the primary surface's box (evidence per view in
    ``view_relations``). Grip metrics from the primary surface: offset of its
    centroid from the closing-region centre along the closing and pad axes,
    and the angle between an elongated surface's principal axis and the
    nearest TCP axis (crooked grasp evidence).
    """
    if entity.grasp_region is None or not entity.parts:
        raise ValueError("Held-object measurement needs the gripper closing region.")
    if not math.isfinite(margin_m) or not 0.0 <= margin_m <= 0.05:
        raise ValueError("Held-object box margin must be within [0, 50 mm].")
    if not math.isfinite(extension_m) or not 0.0 <= extension_m <= 0.10:
        raise ValueError("Held-object view extension must be within [0, 100 mm].")
    if sum(view.role == primary_role for view in views) > 1:
        raise ValueError("Held-object measurement takes at most one primary view.")
    rotation = np.asarray(entity.tcp_from_box_rotation, dtype=float).reshape(3, 3)
    region_centre = np.asarray(entity.grasp_region[0], dtype=float)
    closing_axis, pad_axis = rotation[:, 0], rotation[:, 1]
    axis_index = int(np.argmax(np.abs(closing_axis)))
    view_rows = []
    primary_view = next((view for view in views if view.role == primary_role), None)
    primary, relation = None, None
    if primary_view is not None:
        primary, row = _view_surface(primary_view)
        row["use"] = "primary"
        if primary is None:
            row["relation"] = "insufficient_surface"
        else:
            world = primary @ primary_view.world_from_tcp[:3, :3].T + primary_view.world_from_tcp[:3, 3]
            relation = tool_object_relation(entity=entity, world_from_tcp=primary_view.world_from_tcp,
                                            object_points_m=world)
            row.update(relation.as_dict())
        view_rows.append(row)
    metrics: dict = {"primary_role": primary_role}
    if primary is None or len(primary) < HELD_OBJECT_MIN_POINTS:
        for view in views:
            if view.role != primary_role:
                view_rows.append({"role": view.role, "use": "not_used_without_primary_surface"})
        fused = ({"relation": "insufficient_surface", "view_count": len(views), "usable_view_count": 0}
                 if relation is None else fuse_tool_object_relations([relation]))
        return HeldObjectBox(status="insufficient_surface", reason=(
            "held_object_primary_view_unmeasured" if primary_view is None else "held_object_surface_insufficient"),
            tool_from_box=None, size_xyz_m=None, relation=fused, view_relations=tuple(view_rows), metrics=metrics)
    metrics["primary_point_count"] = len(primary)
    centroid = primary.mean(axis=0)
    metrics["centroid_offset_closing_axis_m"] = float((centroid - region_centre) @ closing_axis)
    metrics["centroid_offset_pad_axis_m"] = float((centroid - region_centre) @ pad_axis)
    if len(primary) >= 10:
        _, values, vectors = np.linalg.svd(primary - centroid, full_matrices=False)
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
    if not relation.between_fingers:
        for view in views:
            if view.role != primary_role:
                view_rows.append({"role": view.role, "use": "not_used_primary_not_between_fingers"})
        return HeldObjectBox(status="not_between_fingers", reason="held_object_not_between_fingers",
                             tool_from_box=None, size_xyz_m=None, relation=fuse_tool_object_relations([relation]),
                             view_relations=tuple(view_rows), metrics=metrics)
    low, high = primary.min(axis=0) - extension_m, primary.max(axis=0) + extension_m
    relations, kept = [relation], [primary]
    for view in views:
        if view.role == primary_role:
            continue
        surface, row = _view_surface(view)
        if surface is None:
            row.update(use="no_surface", extension_point_count=0)
            view_rows.append(row)
            continue
        inside = np.all((surface >= low) & (surface <= high), axis=1)
        row.update(extension_point_count=int(np.count_nonzero(inside)),
                   outside_extension_point_count=int(np.count_nonzero(~inside)))
        if np.count_nonzero(inside) < 3:
            row["use"] = "outside_primary_extension"
            view_rows.append(row)
            continue
        extension = surface[inside]
        world = extension @ view.world_from_tcp[:3, :3].T + view.world_from_tcp[:3, 3]
        view_relation = tool_object_relation(entity=entity, world_from_tcp=view.world_from_tcp,
                                             object_points_m=world)
        relations.append(view_relation)
        kept.append(extension)
        row.update(view_relation.as_dict(), use="extension")
        view_rows.append(row)
    fused = fuse_tool_object_relations(relations)
    points = np.concatenate(kept)
    primary_low, primary_high = _closing_axis_box(primary, region_centre=region_centre, axis_index=axis_index,
                                                  margin_m=margin_m)
    metrics.update(measured_point_count=len(points), extension_point_count=len(points) - len(primary),
                   extension_views=[row["role"] for row in view_rows if row.get("use") == "extension"],
                   view_extension_limit_m=float(extension_m),
                   primary_box_size_xyz_m=(primary_high - primary_low).tolist(),
                   views_between_fingers=[row["role"] for row in view_rows
                                          if str(row.get("relation", "")).startswith("between_fingers")])
    covered = [points]
    if support_plane is not None:
        # The object rests on the support: its hidden lower part reaches the plane.
        anchor = np.asarray(support_plane.anchor_world_m)
        normal = np.asarray(support_plane.normal_world)
        pose = primary_view.world_from_tcp
        world = points @ pose[:3, :3].T + pose[:3, 3]
        dropped = world - np.outer((world - anchor) @ normal, normal)
        covered.append((dropped - pose[:3, 3]) @ pose[:3, :3])
        metrics["support_completion"] = True
    low, high = _closing_axis_box(np.concatenate(covered), region_centre=region_centre, axis_index=axis_index,
                                  margin_m=margin_m)
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


__all__ = ["HELD_OBJECT_BOX_SEMANTICS_V2", "HeldObjectBox", "HeldObjectView", "SupportPlane",
           "grip_seed_pixel", "gripper_solid_distance_m", "measure_held_object_box", "projected_seed_pixel"]
