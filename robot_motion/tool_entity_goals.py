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
        if not np.allclose(rotation.T @ (tip - centre), (0., 0., half[2]), atol=1e-9, rtol=0.):
            raise ValueError("Tool entity tip must be the approach face center.")
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
        """
        half_width = geometry.maximum_opening_m / 2. + geometry.finger_thickness_m
        lower = np.array((-half_width, -geometry.pad_height_m / 2., geometry.palm_z_m - geometry.finger_thickness_m))
        upper = np.array((half_width, geometry.pad_height_m / 2., geometry.palm_z_m + geometry.finger_length_m))
        tcp_from_solid = np.asarray(geometry.solid_pose(np.eye(4)), dtype=float)
        rotation, origin = tcp_from_solid[:3, :3], tcp_from_solid[:3, 3]
        return cls(tcp_from_box_xyz_m=tuple(rotation @ ((lower + upper) / 2.) + origin),
                   tcp_from_box_rotation=tuple(rotation.reshape(-1)),
                   half_extents_m=tuple((upper - lower) / 2.),
                   tip_center_tcp_m=tuple(rotation @ np.array((0., 0., upper[2])) + origin),
                   source=source)

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
                "tip_center_tcp_m": list(self.tip_center_tcp_m), "source": dict(self.source)}


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
                      ceiling_z_m: float, horizontal_margin_m: float = 0.) -> tuple[float, int, int]:
    """Highest observed point inside the box's horizontal footprint.

    ``horizontal_margin_m`` conservatively includes nearby voxel centers whose
    represented points can overlap the footprint. Points above ``ceiling_z_m`` are counted as overhead and excluded from the
    support height. The result is never below ``floor_z_m``. Returns
    ``(support_height_m, footprint_point_count, overhead_point_count)``.
    """
    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Observed points must be finite.")
    if (isinstance(horizontal_margin_m, bool) or not math.isfinite(horizontal_margin_m)
            or horizontal_margin_m < 0.):
        raise ValueError("Horizontal support margin must be finite and nonnegative.")
    hull = _convex_hull_xy(np.asarray(corners_world, dtype=float)[:, :2])
    low, high = hull.min(axis=0) - horizontal_margin_m, hull.max(axis=0) + horizontal_margin_m
    selected = points[(points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
                      & (points[:, 1] >= low[1]) & (points[:, 1] <= high[1])]
    if len(selected):
        inside = np.ones(len(selected), dtype=bool)
        for start, end in zip(hull, np.roll(hull, -1, axis=0)):
            edge, relative = end - start, selected[:, :2] - start
            inside &= (edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
                       >= -horizontal_margin_m * np.linalg.norm(edge) - 1e-12)
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
        ceiling_z_m=anchor[2] + support_window_m + support_margin_m,
        horizontal_margin_m=support_margin_m)
    support += support_margin_m
    lowest = float(corners[:, 2].min())
    return EntityHoverMetrics(tip_offset_xy_m=(float(tip[0] - anchor[0]), float(tip[1] - anchor[1])),
                              clearance_m=lowest - support, support_height_m=support,
                              lowest_point_z_m=lowest, footprint_point_count=count,
                              overhead_point_count=overhead)


def hover_orientation(current_tcp_pose_xyz_quat_xyzw: Sequence[float], policy: str) -> np.ndarray:
    """Return the hover TCP rotation for an explicit policy.

    ``preserve`` keeps the measured TCP orientation. ``tool_axis_down`` turns
    the TCP approach axis (+Z) onto public reference -Z with the least
    rotation, using the shared ``compute_spatial_goal`` alignment convention.
    """
    pose = finite_vector(list(current_tcp_pose_xyz_quat_xyzw), 7, name="current TCP pose")
    if policy not in HOVER_ORIENTATION_POLICIES:
        raise ValueError("Unknown hover orientation policy.")
    if policy == "preserve":
        goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(pose), anchor_xyz_m=list(pose[:3]),
            offset_xyz_m=[0., 0., 0.], offset_frame="reference", orientation_mode="preserve")
    else:
        goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(pose), anchor_xyz_m=list(pose[:3]),
            offset_xyz_m=[0., 0., 0.], offset_frame="reference", orientation_mode="align_axis",
            tool_axis="z", axis_reference_xyz=[0., 0., -1.])
    return Rotation.from_quat(goal.pose_xyz_quat_xyzw[3:]).as_matrix()


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
    rotation = hover_orientation(current, orientation_policy)
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
    "TOOL_ENTITY_HOVER_SEMANTICS_V1",
    "ToolEntityBox",
    "entity_hover_metrics",
    "footprint_support",
    "generate_entity_hover_goals",
    "hover_orientation",
]
