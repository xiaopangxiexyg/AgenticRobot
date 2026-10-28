"""Tool-entity hover proposals over measured surfaces.

A TCP point above a surface does not say where the attached tool is. This
module places a rigid, TCP-attached tool box instead: the centre of its
approach-side face (``tip``) is offset horizontally from a measured anchor, and
the box's lowest point keeps an explicit vertical clearance above the highest
observed point under the box's horizontal footprint.

It is pure geometry over explicit inputs. It performs no IK, collision, path or
execution decision, never turns a proposal into an observation, and does not
change the existing TCP relation semantics in ``robot_motion.spatial_goals``.
Unobserved space under the footprint is not known to be free; the caller's
planner still checks its own collision model against the observed scene.

The same gripper solids also classify measured gripper-object relations
(``agenticrobot.relation.tool_object.v1``): one object's largest measured
surface cluster against the palm/finger solids and the open closing region.
That is observation evidence for the upper layer, not a grasp or contact claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from robot_motion.spatial_goals import compute_spatial_goal, finite_vector

TOOL_ENTITY_HOVER_SEMANTICS_V1 = "agenticrobot.relation.tool_entity_hover.v1"
HOVER_ORIENTATION_POLICIES = ("preserve", "tool_axis_down")
MAX_HOVER_TIP_OFFSET_M = 0.25
MAX_HOVER_CLEARANCE_M = 0.30
# A tilt cone wider than this is no longer a "tool down" request.
MAX_TOOL_AXIS_TILT_RAD = math.radians(60.)
_ROUND = 12


def _vector3(value, *, name: str) -> np.ndarray:
    return finite_vector(list(np.asarray(value, dtype=float).reshape(-1)), 3, name=name)


@dataclass(frozen=True)
class ToolEntityBox:
    """Rigid oriented box fixed to the controller tool TCP.

    ``tcp_from_box_*`` locate the box centre and axes in the TCP frame.
    ``tip_center_tcp_m`` is the centre of the approach-side face. ``source``
    records where the numbers came from; it is identity, not authority.
    """

    tcp_from_box_xyz_m: tuple[float, float, float]
    tcp_from_box_rotation: tuple[float, ...]
    half_extents_m: tuple[float, float, float]
    tip_center_tcp_m: tuple[float, float, float]
    source: Mapping[str, object] = field(default_factory=dict)
    # Optional finer geometry sharing the box axes: (centre_tcp, half) solids
    # (palm and fingers) and the closing region between the fingers.
    parts: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    grasp_region: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        centre = _vector3(self.tcp_from_box_xyz_m, name="tcp_from_box_xyz_m")
        tip = _vector3(self.tip_center_tcp_m, name="tip_center_tcp_m")
        half = _vector3(self.half_extents_m, name="half_extents_m")
        if np.any(half <= 0):
            raise ValueError("Tool entity half extents must be positive.")
        rotation = np.asarray(self.tcp_from_box_rotation, dtype=float)
        if rotation.size != 9 or not np.isfinite(rotation).all():
            raise ValueError("Tool entity rotation must contain nine finite values.")
        rotation = rotation.reshape(3, 3)
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0.)):
            raise ValueError("Tool entity rotation must be a right-handed orthonormal matrix.")

        def box(value, name):
            if len(value) != 2:
                raise ValueError(f"{name} must be (centre, half extents).")
            c, h = _vector3(value[0], name=name + " centre"), _vector3(value[1], name=name + " half")
            if np.any(h <= 0):
                raise ValueError(f"{name} half extents must be positive.")
            return tuple(float(x) for x in c), tuple(float(x) for x in h)
        object.__setattr__(self, "parts", tuple(box(part, "tool part") for part in self.parts))
        if self.grasp_region is not None:
            object.__setattr__(self, "grasp_region", box(self.grasp_region, "grasp region"))
        object.__setattr__(self, "tcp_from_box_xyz_m", tuple(float(x) for x in centre))
        object.__setattr__(self, "tip_center_tcp_m", tuple(float(x) for x in tip))
        object.__setattr__(self, "half_extents_m", tuple(float(x) for x in half))
        object.__setattr__(self, "tcp_from_box_rotation", tuple(float(x) for x in rotation.reshape(-1)))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))

    @classmethod
    def from_parallel_jaw_geometry(cls, geometry, *, source: Mapping[str, object]) -> ToolEntityBox:
        """Envelope the palm and both finger solids at the geometry's maximum opening.

        Uses the same solid frame and extents as the grasp-screening masks:
        fingers span ``palm_z .. palm_z + finger_length`` along the approach
        axis, the palm ``palm_z - finger_thickness .. palm_z``; the width is
        ``opening / 2 + finger_thickness`` per side and the depth is the pad
        height. Opening-dependent finger positions are deliberately enveloped.
        The separate palm/finger solids and the open closing region are kept
        for measured tool-object relations.
        """
        opening, thickness = geometry.maximum_opening_m, geometry.finger_thickness_m
        half_width = opening / 2. + thickness
        lower = np.array((-half_width, -geometry.pad_height_m / 2., geometry.palm_z_m - thickness))
        upper = np.array((half_width, geometry.pad_height_m / 2., geometry.palm_z_m + geometry.finger_length_m))
        tcp_from_solid = np.asarray(geometry.solid_pose(np.eye(4)), dtype=float)
        rotation, origin = tcp_from_solid[:3, :3], tcp_from_solid[:3, 3]

        def solid(low, high):
            low, high = np.asarray(low, float), np.asarray(high, float)
            return tuple(rotation @ ((low + high) / 2.) + origin), tuple((high - low) / 2.)
        pad = geometry.pad_height_m / 2.
        finger_z = (geometry.palm_z_m, geometry.palm_z_m + geometry.finger_length_m)
        parts = (solid((-half_width, -pad, lower[2]), (half_width, pad, geometry.palm_z_m)),
                 solid((-half_width, -pad, finger_z[0]), (-opening / 2., pad, finger_z[1])),
                 solid((opening / 2., -pad, finger_z[0]), (half_width, pad, finger_z[1])))
        grasp = solid((-opening / 2., -pad, finger_z[0]), (opening / 2., pad, finger_z[1]))
        return cls(tcp_from_box_xyz_m=tuple(rotation @ ((lower + upper) / 2.) + origin),
                   tcp_from_box_rotation=tuple(rotation.reshape(-1)),
                   half_extents_m=tuple((upper - lower) / 2.),
                   tip_center_tcp_m=tuple(rotation @ np.array((0., 0., upper[2])) + origin),
                   source=source, parts=parts, grasp_region=grasp)

    def corners_tcp(self) -> np.ndarray:
        rotation = np.asarray(self.tcp_from_box_rotation).reshape(3, 3)
        signs = np.array([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        return np.asarray(self.tcp_from_box_xyz_m) + (signs * np.asarray(self.half_extents_m)) @ rotation.T

    def corners_world(self, world_from_tcp: np.ndarray) -> np.ndarray:
        transform = np.asarray(world_from_tcp, dtype=float)
        return self.corners_tcp() @ transform[:3, :3].T + transform[:3, 3]

    def tip_world(self, world_from_tcp: np.ndarray) -> np.ndarray:
        transform = np.asarray(world_from_tcp, dtype=float)
        return transform[:3, :3] @ np.asarray(self.tip_center_tcp_m) + transform[:3, 3]

    def identity(self) -> dict[str, object]:
        return {"tcp_from_box_xyz_m": list(self.tcp_from_box_xyz_m),
                "tcp_from_box_rotation": list(self.tcp_from_box_rotation),
                "half_extents_m": list(self.half_extents_m),
                "tip_center_tcp_m": list(self.tip_center_tcp_m), "source": dict(self.source),
                "parts": [[list(c), list(h)] for c, h in self.parts],
                "grasp_region": None if self.grasp_region is None else [list(v) for v in self.grasp_region]}


def _signed_box_distance(points_local: np.ndarray, centre, half) -> np.ndarray:
    """Signed distance to an axis-aligned box; negative inside."""
    delta = np.abs(points_local - np.asarray(centre)) - np.asarray(half)
    return np.linalg.norm(np.maximum(delta, 0.), axis=1) + np.minimum(np.max(delta, axis=1), 0.)


TOOL_OBJECT_RELATION_SEMANTICS_V1 = "agenticrobot.relation.tool_object.v1"
TOOL_OBJECT_RELATIONS = ("between_fingers_in_contact_band", "between_fingers", "in_contact_band", "near", "far")


def object_surface_cluster(points_m, *, voxel_size_m: float = 0.003,
                           link_distance_m: float = 0.010) -> tuple[np.ndarray, dict]:
    """Return the largest connected measured surface of one object's mask points.

    Points are first thinned to one point per ``voxel_size_m`` cell, so counts
    are metric and do not depend on camera range. Voxels closer than
    ``link_distance_m`` are connected; only the largest component is kept.
    A mask that also covers a nearer occluder (for example a finger in front
    of the object) yields a separate depth cluster and is not object evidence.
    Deterministic for identical input order.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Object points must be a nonempty finite N x 3 array.")
    if not (math.isfinite(voxel_size_m) and 0. < voxel_size_m <= 0.02
            and math.isfinite(link_distance_m) and voxel_size_m <= link_distance_m <= 0.05):
        raise ValueError("Voxel size must be in (0, 0.02] m and the link distance in [voxel, 0.05] m.")
    _, first = np.unique(np.floor(points / voxel_size_m).astype(np.int64), axis=0, return_index=True)
    voxels = points[np.sort(first)]
    pairs = cKDTree(voxels).query_pairs(r=link_distance_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                       (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))), shape=(len(voxels), len(voxels)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    kept = voxels[labels == int(np.argmax(sizes))]
    return kept, {"input_point_count": int(len(points)), "voxel_count": int(len(voxels)),
                  "component_count": int(count), "kept_voxel_count": int(len(kept)),
                  "voxel_size_m": float(voxel_size_m), "link_distance_m": float(link_distance_m)}


@dataclass(frozen=True)
class ToolObjectRelation:
    """Measured relation between one arm's gripper solids and one object surface.

    Inputs are the object's measured surface points (see
    ``object_surface_cluster``) and the gripper box parts at the measured TCP.
    The palm/finger solids are the maximum-opening screening envelope, so real
    finger surfaces lie a few millimetres inside them; ``in_contact_band``
    means "surface within the band of that envelope", not a contact, force,
    grasp or stability claim. ``surface_distance_m`` is the k-th smallest
    signed point distance (k = the contact count threshold), negative inside.
    """

    relation: str
    surface_distance_m: float
    contact_band_point_count: int
    grasp_region_point_count: int
    object_point_count: int

    @property
    def between_fingers(self) -> bool:
        return self.relation.startswith("between_fingers")

    @property
    def in_contact_band(self) -> bool:
        return self.relation in ("between_fingers_in_contact_band", "in_contact_band")

    def as_dict(self) -> dict:
        return {"relation": self.relation, "surface_distance_m": self.surface_distance_m,
                "contact_band_point_count": self.contact_band_point_count,
                "grasp_region_point_count": self.grasp_region_point_count,
                "object_point_count": self.object_point_count}


def tool_object_relation(*, entity: ToolEntityBox, world_from_tcp: np.ndarray, object_points_m,
                         contact_distance_m: float = 0.005, near_distance_m: float = 0.05,
                         contact_minimum_points: int = 3,
                         region_minimum_points: int = 10) -> ToolObjectRelation:
    """Classify one measured object surface against one arm's gripper.

    ``between_fingers`` needs ``region_minimum_points`` surface points inside
    the open closing region between the finger solids; the contact band needs
    ``contact_minimum_points`` points within ``contact_distance_m`` of (or
    inside) the palm/finger solids. Pass voxel-thinned points so the counts
    are metric. Thresholds were calibrated on the saved operator samples.
    """
    if not entity.parts or entity.grasp_region is None:
        raise ValueError("Tool-object relations need the gripper solids and closing region.")
    points = np.asarray(object_points_m, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Object points must be a nonempty finite N x 3 array.")
    if (not 0. < contact_distance_m < near_distance_m <= 0.5
            or type(contact_minimum_points) is not int or contact_minimum_points < 1
            or type(region_minimum_points) is not int or region_minimum_points < 1):
        raise ValueError("Need 0 < contact < near <= 0.5 m and positive integer point thresholds.")
    transform = np.asarray(world_from_tcp, dtype=float).reshape(4, 4)
    rotation = np.asarray(entity.tcp_from_box_rotation).reshape(3, 3)
    # World -> TCP coordinates, then into the shared solid axes (no translation).
    local = ((points - transform[:3, 3]) @ transform[:3, :3]) @ rotation
    distance = np.min([_signed_box_distance(local, np.asarray(c) @ rotation, h) for c, h in entity.parts], axis=0)
    centre, half = entity.grasp_region
    inside = np.all(np.abs(local - np.asarray(centre) @ rotation) < np.asarray(half), axis=1)
    region = int(np.count_nonzero(inside))
    band = int(np.count_nonzero(distance <= contact_distance_m))
    robust = float(np.partition(distance, min(contact_minimum_points, len(distance)) - 1)[
        min(contact_minimum_points, len(distance)) - 1])
    between, touching = region >= region_minimum_points, band >= contact_minimum_points
    if between:
        relation = "between_fingers_in_contact_band" if touching else "between_fingers"
    elif touching:
        relation = "in_contact_band"
    elif robust <= near_distance_m:
        relation = "near"
    else:
        relation = "far"
    return ToolObjectRelation(relation=relation, surface_distance_m=robust, contact_band_point_count=band,
                              grasp_region_point_count=region, object_point_count=int(len(points)))


def fuse_tool_object_relations(relations: Sequence[ToolObjectRelation]) -> dict:
    """Combine one object's per-view relations for one arm.

    Every view is partial surface evidence, so the union is used: the object is
    between the fingers or in the contact band if any view shows it, and the
    distance is the smallest robust distance. ``views_agree`` reports whether
    all views gave the same class, so disagreement stays visible.
    """
    if not relations:
        raise ValueError("At least one view relation is required.")
    between = any(item.between_fingers for item in relations)
    band = any(item.in_contact_band for item in relations)
    distance = min(item.surface_distance_m for item in relations)
    if between:
        relation = "between_fingers_in_contact_band" if band else "between_fingers"
    elif band:
        relation = "in_contact_band"
    else:
        # All views are near/far here; nearest view decides.
        relation = min(relations, key=lambda item: item.surface_distance_m).relation
    return {"relation": relation, "surface_distance_m": distance, "view_count": len(relations),
            "views_agree": len({item.relation for item in relations}) == 1}


@dataclass(frozen=True)
class EntityHoverMetrics:
    """Measured-scene geometry of one tool-box pose; not a collision certificate."""

    tip_offset_xy_m: tuple[float, float]
    clearance_m: float
    support_height_m: float
    lowest_point_z_m: float
    footprint_point_count: int
    overhead_point_count: int


@dataclass(frozen=True)
class EntityHoverCandidate:
    """One explicit hover sample; no IK, path or motion claim."""

    pose_xyz_quat_xyzw: tuple[float, ...]
    tip_offset_xy_m: tuple[float, float]
    clearance_m: float
    metrics: EntityHoverMetrics


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    unique = sorted({(round(float(x), _ROUND), round(float(y), _ROUND)) for x, y in points})
    if len(unique) < 3:
        raise ValueError("Tool footprint must have a positive area.")

    def half(sequence):
        hull = []
        for point in sequence:
            while len(hull) >= 2 and (
                    (hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1])
                    - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0])) <= 0:
                hull.pop()
            hull.append(point)
        return hull
    lower, upper = half(unique), half(reversed(unique))
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError("Tool footprint must have a positive area.")
    return hull


def footprint_support(points_m, corners_world: np.ndarray, *, floor_z_m: float,
                      ceiling_z_m: float) -> tuple[float, int, int]:
    """Highest observed point inside the box's horizontal footprint.

    Points above ``ceiling_z_m`` are counted as overhead and excluded from the
    support height. The result is never below ``floor_z_m``. Returns
    ``(support_height_m, footprint_point_count, overhead_point_count)``.
    """
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Observed points must be finite.")
    hull = _convex_hull_xy(np.asarray(corners_world, dtype=float)[:, :2])
    low, high = hull.min(axis=0), hull.max(axis=0)
    selected = points[(points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
                      & (points[:, 1] >= low[1]) & (points[:, 1] <= high[1])]
    if len(selected):
        inside = np.ones(len(selected), dtype=bool)
        for start, end in zip(hull, np.roll(hull, -1, axis=0)):
            edge, relative = end - start, selected[:, :2] - start
            inside &= edge[0] * relative[:, 1] - edge[1] * relative[:, 0] >= -1e-12
        selected = selected[inside]
    below = selected[selected[:, 2] <= ceiling_z_m]
    overhead = int(np.count_nonzero(selected[:, 2] > ceiling_z_m))
    height = max(float(floor_z_m), float(below[:, 2].max())) if len(below) else float(floor_z_m)
    return height, int(len(below)), overhead


def entity_hover_metrics(*, entity: ToolEntityBox, world_from_tcp: np.ndarray, anchor_xyz_m,
                         points_m, support_window_m: float, support_margin_m: float = 0.) -> EntityHoverMetrics:
    """Tip offset and vertical clearance of the tool box at one TCP pose.

    Support height is ``max(anchor z, observed z under the footprint up to
    anchor z + support_window_m) + support_margin_m``.
    """
    anchor = _vector3(anchor_xyz_m, name="anchor_xyz_m")
    corners = entity.corners_world(world_from_tcp)
    tip = entity.tip_world(world_from_tcp)
    support, count, overhead = footprint_support(points_m, corners, floor_z_m=anchor[2],
                                                 ceiling_z_m=anchor[2] + support_window_m)
    support += support_margin_m
    lowest = float(corners[:, 2].min())
    return EntityHoverMetrics(tip_offset_xy_m=(float(tip[0] - anchor[0]), float(tip[1] - anchor[1])),
                              clearance_m=lowest - support, support_height_m=support,
                              lowest_point_z_m=lowest, footprint_point_count=count,
                              overhead_point_count=overhead)


def tool_axis_tilt_rad(rotation) -> float:
    """Angle between the TCP approach axis (+Z) and public reference -Z."""
    axis = np.asarray(rotation, dtype=float).reshape(3, 3)[:, 2]
    return float(math.acos(max(-1., min(1., -axis[2] / np.linalg.norm(axis)))))


def _tilt(value) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0. <= value <= MAX_TOOL_AXIS_TILT_RAD):
        raise ValueError("tool_axis_max_tilt_rad must be finite within [0, 60 deg].")
    return float(value)


def hover_orientation(current_tcp_pose_xyz_quat_xyzw: Sequence[float], policy: str, *,
                      tool_axis_max_tilt_rad: float = 0.) -> np.ndarray:
    """Return the hover TCP rotation for an explicit policy.

    ``preserve`` keeps the measured TCP orientation. ``tool_axis_down`` turns
    the TCP approach axis (+Z) toward public reference -Z with the least
    rotation, using the shared ``compute_spatial_goal`` alignment convention,
    and stops at the cone ``tool_axis_max_tilt_rad`` around -Z (0 = exactly
    down). A tool already inside the cone keeps its measured orientation.
    """
    pose = finite_vector(list(current_tcp_pose_xyz_quat_xyzw), 7, name="current TCP pose")
    if policy not in HOVER_ORIENTATION_POLICIES:
        raise ValueError("Unknown hover orientation policy.")
    tilt = _tilt(tool_axis_max_tilt_rad)
    if policy == "preserve":
        if tilt:
            raise ValueError("A tool-axis tilt cone applies only to tool_axis_down.")
        goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(pose), anchor_xyz_m=list(pose[:3]),
            offset_xyz_m=[0., 0., 0.], offset_frame="reference", orientation_mode="preserve")
        return Rotation.from_quat(goal.pose_xyz_quat_xyzw[3:]).as_matrix()
    goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(pose), anchor_xyz_m=list(pose[:3]),
        offset_xyz_m=[0., 0., 0.], offset_frame="reference", orientation_mode="align_axis",
        tool_axis="z", axis_reference_xyz=[0., 0., -1.])
    exact = Rotation.from_quat(goal.pose_xyz_quat_xyzw[3:])
    if not tilt:
        return exact.as_matrix()
    current = Rotation.from_quat(pose[3:] / np.linalg.norm(pose[3:]))
    angle = tool_axis_tilt_rad(current.as_matrix())
    if angle <= tilt:
        return current.as_matrix()
    # Same least rotation as the exact alignment, stopped at the cone boundary.
    turn = (exact * current.inv()).as_rotvec()
    return (Rotation.from_rotvec(turn * ((angle - tilt) / np.linalg.norm(turn))) * current).as_matrix()


def _interval(value, *, name: str, lower: float, upper: float) -> tuple[float, float]:
    low, high = finite_vector(list(value), 2, name=name).tolist()
    if low > high or low < lower or high > upper:
        raise ValueError(f"{name} must be ordered within [{lower}, {upper}].")
    return low, high


def generate_entity_hover_goals(
    *, current_tcp_pose_xyz_quat_xyzw: Sequence[float], anchor_xyz_m: Sequence[float],
    tip_offset_bounds_xy_m: Sequence[Sequence[float]], clearance_bounds_m: Sequence[float],
    entity: ToolEntityBox, points_m, orientation_policy: str = "preserve", max_candidates: int = 8,
    boundary_inset_m: float = 0., support_window_m: float = 0.30, support_margin_m: float = 0.,
    tool_axis_max_tilt_rad: float = 0.,
) -> tuple[EntityHoverCandidate, ...]:
    """Propose tool-box hover poses inside explicit tip-offset and clearance bounds.

    Proposal coordinates are ``(tip dx, tip dy, clearance)``. Order: the point
    nearest the measured tool state (at the hover orientation), the centre,
    the six face centres by distance from that state, then unscrambled Halton
    fill; duplicates are dropped and at most ``max_candidates`` are returned.
    ``boundary_inset_m`` shrinks proposals only (room for arrival tolerance);
    callers must validate achieved poses against the ORIGINAL bounds.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be an integer from 1 to 8.")
    for value, name in ((boundary_inset_m, "boundary_inset_m"), (support_window_m, "support_window_m"),
                        (support_margin_m, "support_margin_m")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if len(tip_offset_bounds_xy_m) != 2:
        raise ValueError("tip_offset_bounds_xy_m must contain X and Y intervals.")
    bounds = np.array([
        _interval(tip_offset_bounds_xy_m[0], name="tip X offset", lower=-MAX_HOVER_TIP_OFFSET_M,
                  upper=MAX_HOVER_TIP_OFFSET_M),
        _interval(tip_offset_bounds_xy_m[1], name="tip Y offset", lower=-MAX_HOVER_TIP_OFFSET_M,
                  upper=MAX_HOVER_TIP_OFFSET_M),
        _interval(clearance_bounds_m, name="clearance", lower=0., upper=MAX_HOVER_CLEARANCE_M)])
    current = finite_vector(list(current_tcp_pose_xyz_quat_xyzw), 7, name="current TCP pose")
    anchor = _vector3(anchor_xyz_m, name="anchor_xyz_m")
    rotation = hover_orientation(current, orientation_policy, tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    inset = np.minimum(boundary_inset_m, (bounds[:, 1] - bounds[:, 0]) / 2.)
    low, high = bounds[:, 0] + inset, bounds[:, 1] - inset
    centre = low + (high - low) / 2.

    def metrics_at(transform):
        return entity_hover_metrics(entity=entity, world_from_tcp=transform, anchor_xyz_m=anchor,
            points_m=points, support_window_m=support_window_m, support_margin_m=support_margin_m)

    measured = np.eye(4)
    measured[:3, :3], measured[:3, 3] = rotation, current[:3]
    state = metrics_at(measured)
    here = np.array((*state.tip_offset_xy_m, state.clearance_m))

    def place(proposal):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        tip_from_tcp = rotation @ np.asarray(entity.tip_center_tcp_m)
        transform[:2, 3] = anchor[:2] + proposal[:2] - tip_from_tcp[:2]
        # The footprint depends only on horizontal placement; solve height last.
        at_zero = metrics_at(transform)
        transform[2, 3] = at_zero.support_height_m + proposal[2] - at_zero.lowest_point_z_m
        metrics = metrics_at(transform)
        if abs(metrics.clearance_m - proposal[2]) > 1e-9:
            raise RuntimeError("Hover clearance placement is inconsistent.")
        quaternion = Rotation.from_matrix(rotation).as_quat()
        return EntityHoverCandidate(pose_xyz_quat_xyzw=tuple(float(x) for x in (*transform[:3, 3], *quaternion)),
            tip_offset_xy_m=(float(proposal[0]), float(proposal[1])), clearance_m=float(proposal[2]),
            metrics=metrics)

    faces = []
    for axis in range(3):
        for edge in (low[axis], high[axis]):
            face = centre.copy()
            face[axis] = edge
            faces.append(face)
    faces.sort(key=lambda value: math.dist(here, value))
    fill = low + qmc.Halton(d=3, scramble=False).random(2 * max_candidates) * (high - low)
    candidates, seen = [], set()
    for proposal in (np.clip(here, low, high), centre, *faces, *fill):
        key = tuple(round(float(x), _ROUND) for x in proposal)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(place(np.asarray(proposal, dtype=float)))
        if len(candidates) == max_candidates:
            break
    return tuple(candidates)


__all__ = [
    "EntityHoverCandidate",
    "EntityHoverMetrics",
    "HOVER_ORIENTATION_POLICIES",
    "MAX_HOVER_CLEARANCE_M",
    "MAX_HOVER_TIP_OFFSET_M",
    "MAX_TOOL_AXIS_TILT_RAD",
    "TOOL_ENTITY_HOVER_SEMANTICS_V1",
    "TOOL_OBJECT_RELATIONS",
    "TOOL_OBJECT_RELATION_SEMANTICS_V1",
    "ToolEntityBox",
    "ToolObjectRelation",
    "entity_hover_metrics",
    "footprint_support",
    "fuse_tool_object_relations",
    "generate_entity_hover_goals",
    "hover_orientation",
    "object_surface_cluster",
    "tool_axis_tilt_rad",
    "tool_object_relation",
]
