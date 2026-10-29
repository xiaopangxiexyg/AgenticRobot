"""Scene-level tool-entity relations and hover search shared by every host.

The offline saved-scene probe and the RealMan and MuJoCo Gateways all use
these functions, so one measured scene gives the same answer everywhere.
Hosts own acquisition, segmentation, kinematics and planning; this module
receives their explicit outputs (aligned depth views, object masks, per-view
capture TCP poses, a self-filtered observed cloud and a planner callable).
It performs no acquisition, IK, collision check or execution decision.

Relations follow ``agenticrobot.relation.tool_object.v1`` and hover search
follows ``agenticrobot.relation.tool_entity_hover.v1`` (see
``robot_motion.tool_entity_goals`` for the pure geometry).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.tool_entity_goals import (
    HOVER_ORIENTATION_POLICIES,
    MAX_HOVER_CLEARANCE_M,
    MAX_HOVER_TIP_OFFSET_M,
    TOOL_ENTITY_HOVER_SEMANTICS_V1,
    TOOL_OBJECT_RELATION_SEMANTICS_V1,
    EntityHoverCandidate,
    ToolEntityBox,
    entity_hover_metrics,
    fuse_tool_object_relations,
    generate_entity_hover_goals,
    hover_orientation,
    object_surface_cluster,
    tool_axis_tilt_rad,
    tool_object_relation,
)

# Saved or live object masks are eroded before back-projection so boundary
# pixels (mask bleed, mixed depth) do not become object surface evidence.
OBJECT_MASK_EROSION_PX = 2
# A view whose second-largest surface component reaches this share of the kept
# one (and at least 10 voxels) is reported as ambiguous; the class still uses
# the largest component. On the 21 saved samples every disagreeing secondary
# component was mask bleed onto another object or a gripper part.
SURFACE_AMBIGUITY_SHARE = 0.25
SURFACE_AMBIGUITY_MINIMUM_VOXELS = 10
# Object-mask surfaces use only depth up to the safe-motion scene's range
# (the calibration of the 21 saved samples); farther pixels are not evidence.
RELATION_MAXIMUM_DEPTH_M = 1.5
# Height above the anchor below which observed points count as support under a
# hovering tool footprint; higher points are reported as overhead obstacles.
ENTITY_HOVER_SUPPORT_WINDOW_M = 0.30
# One hover search: every candidate starts at the same measured state, at most
# this long per candidate and in total. Reoriented hovers need more than the
# 3 s TCP-region cap: on the saved samples a 45-degree cone found 3/12 paths at
# 3 s/20 s and 8/12 at 5 s/20 s (user decision 2026-10-29: 5 s per candidate).
ENTITY_HOVER_REGION_BUDGET_SEC = 20.
ENTITY_HOVER_CANDIDATE_BUDGET_SEC = 5.
SURFACE_CLUSTER_FIELDS = ("voxel_count", "component_count", "kept_voxel_count",
                          "second_component_voxel_count")
RELATION_EVIDENCE_SCOPE = "measured_visible_surfaces_vs_gripper_screening_solids_and_bound_contact_solids"


def pose_matrix(pose_xyz_quat_xyzw) -> np.ndarray:
    """Full-precision pose matrix (the shared public encoder rounds to float32)."""
    pose = np.asarray(pose_xyz_quat_xyzw, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Pose must contain seven finite values.")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def pose_xyz_quat_xyzw(transform) -> list[float]:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return [*transform[:3, 3].tolist(), *Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()]


def backproject_mask(depth_m, intrinsics, world_from_camera, mask):
    """Pinhole back-projection of the masked pixels with positive depth."""
    depth = np.asarray(depth_m)
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool) & (depth > 0))
    z = depth[rows, columns]
    points = np.column_stack(((columns - intrinsics["cx"]) * z / intrinsics["fx"],
                              (rows - intrinsics["cy"]) * z / intrinsics["fy"], z))
    pose = np.asarray(world_from_camera).reshape(4, 4)
    return points @ pose[:3, :3].T + pose[:3, 3], rows, columns


def object_surface_from_mask(mask, capture, *, maximum_depth_m, robot_mask=None):
    """Measured surface of one object mask: eroded, valid depth, largest cluster.

    ``capture`` provides ``depth_m``, ``intrinsics`` (fx, fy, cx, cy) and
    ``source_from_camera`` (world from camera). ``robot_mask`` optionally marks
    pixels known to show the robot; they are never object evidence. (The
    planning self-filter is deliberately generous and also erases object points
    next to the gripper, so it is not used for relations.) Returns
    ``(points, evidence)``; ``points`` is ``None`` when no valid measured pixel
    remains. Observation only: never a pose or full shape.
    """
    from scipy import ndimage

    eroded = ndimage.binary_erosion(np.asarray(mask, dtype=bool), iterations=OBJECT_MASK_EROSION_PX)
    depth = capture.depth_m
    if eroded.shape != np.asarray(depth).shape:
        raise ValueError("Object mask must match the aligned depth frame.")
    eroded &= np.isfinite(depth) & (depth > 0) & (depth <= maximum_depth_m)
    robot_removed = 0
    if robot_mask is not None:
        robot = np.asarray(robot_mask, dtype=bool)
        if robot.shape != eroded.shape:
            raise ValueError("Robot mask must match the object mask frame.")
        robot_removed = int(np.count_nonzero(eroded & robot))
        eroded &= ~robot
    evidence = {"mask_erosion_px": OBJECT_MASK_EROSION_PX, "robot_pixels_removed": robot_removed,
                "robot_mask_applied": robot_mask is not None,
                "valid_pixel_count": int(np.count_nonzero(eroded))}
    if not evidence["valid_pixel_count"]:
        return None, evidence
    points, _, _ = backproject_mask(depth, capture.intrinsics, capture.source_from_camera, eroded)
    surface, cluster = object_surface_cluster(points)
    return surface, {**evidence, **cluster}


def surface_ambiguous(surface_cluster: Mapping) -> bool | None:
    """Whether a second surface component is large enough to report."""
    if "second_component_voxel_count" not in surface_cluster:
        return None
    return bool(surface_cluster["second_component_voxel_count"] >= max(
        SURFACE_AMBIGUITY_MINIMUM_VOXELS, SURFACE_AMBIGUITY_SHARE * surface_cluster["kept_voxel_count"]))


@dataclass(frozen=True)
class MeasuredObjectSurface:
    """One labelled object's measured surface in one camera view.

    ``tcp_pose_by_side`` is each arm's TCP at that frame's capture qpos; a
    relation is a property of that frame, not of a later planning start.
    ``None`` means the host only has one shared measured state, passed
    separately as the fallback pose.
    """

    reference: str
    label: str
    role: str | None
    points_m: np.ndarray
    surface_cluster: Mapping[str, object] = field(default_factory=dict)
    tcp_pose_by_side: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        points = np.array(self.points_m, dtype=float).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError("Object surfaces must be nonempty finite N x 3 arrays.")
        points.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        if self.tcp_pose_by_side is not None:
            checked = {}
            for side, pose in self.tcp_pose_by_side.items():
                pose = tuple(float(x) for x in pose)
                if len(pose) != 7 or not all(math.isfinite(x) for x in pose) or abs(
                        math.hypot(*pose[3:]) - 1.) > 1e-5:
                    raise ValueError("Capture TCP poses must be finite with unit quaternions.")
                checked[side] = pose
            object.__setattr__(self, "tcp_pose_by_side", checked)


def describe_tool_object_relations(*, surfaces: Sequence[MeasuredObjectSurface],
                                   tool_entity_by_side: Mapping[str, ToolEntityBox],
                                   fallback_tcp_pose_by_side: Mapping[str, Sequence[float]] | None = None,
                                   all_sides: Sequence[str] = ("left", "right")) -> dict:
    """Measured gripper-object relations of every surface against every bound arm.

    Each surface (eroded mask, largest measured cluster) is compared with each
    bound arm's gripper at that view's capture TCP: the maximum-opening
    screening solids with their closing region plus any bound contact-only
    solids, reported as separate distances. ``objects`` fuses the views of one
    label (one physical object); ``views`` keeps each view with its cluster
    counts and ambiguity flag. Unbound arms are listed, never silently omitted.
    The contact band is proximity evidence, never a contact, grasp or force
    claim. Returns the semantics-level fields; hosts add scope and provenance.
    """
    if not set(tool_entity_by_side) <= set(all_sides) or not all(
            isinstance(entity, ToolEntityBox) for entity in tool_entity_by_side.values()):
        raise ValueError("Tool entities must be ToolEntityBox values keyed by arm.")
    pairs = [(item.label, item.role) for item in surfaces]
    if len(set(pairs)) != len(pairs):
        # Fusion treats one label as one physical object seen from several views.
        raise ValueError("Each measured object label may have at most one surface per camera role.")
    views, grouped = [], {}
    for item in surfaces:
        cluster = item.surface_cluster or {}
        surface = {key: cluster[key] for key in SURFACE_CLUSTER_FIELDS if key in cluster}
        ambiguous = surface_ambiguous(surface)
        for side in sorted(tool_entity_by_side):
            if item.tcp_pose_by_side is not None:
                pose, source = item.tcp_pose_by_side[side], "frame_capture_qpos"
            elif fallback_tcp_pose_by_side is not None:
                pose, source = fallback_tcp_pose_by_side[side], "planning_start_qpos"
            else:
                raise ValueError("A relation needs the gripper pose of its frame.")
            relation = tool_object_relation(entity=tool_entity_by_side[side],
                world_from_tcp=pose_matrix(pose), object_points_m=item.points_m)
            views.append({"side": side, "target_reference": item.reference, "label": item.label,
                          "role": item.role, **relation.as_dict(), "gripper_pose_source": source,
                          "surface_ambiguous": ambiguous, **({"surface_cluster": surface} if surface else {})})
            grouped.setdefault((side, item.label), []).append((relation, ambiguous))
    objects = [{"side": side, "label": label, **fuse_tool_object_relations([relation for relation, _ in items]),
                "ambiguous_view_count": sum(flag is True for _, flag in items)}
               for (side, label), items in sorted(grouped.items())]
    gripper = {side: {"screening_solids": "graspgenx_screening_solids_maximum_opening_envelope",
                      "contact_solids": entity.source.get("contact_solids"),
                      "contact_solid_count": len(entity.contact_solids)}
               for side, entity in sorted(tool_entity_by_side.items())}
    return {"semantics": TOOL_OBJECT_RELATION_SEMANTICS_V1, "objects": objects, "views": views,
            "evaluated_sides": sorted(tool_entity_by_side),
            "unbound_sides": sorted(set(all_sides) - set(tool_entity_by_side)),
            "gripper_geometry_by_side": gripper, "evidence_scope": RELATION_EVIDENCE_SCOPE}


def _bounded_interval(value, *, name: str, lower: float, upper: float) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in value)):
        raise ValueError(f"{name} must contain two finite numbers.")
    low, high = float(value[0]), float(value[1])
    if low > high or low < lower or high > upper:
        raise ValueError(f"{name} must be ordered within [{lower}, {upper}].")
    return low, high


def validate_hover_request(*, tip_offset_bounds_xy_m, clearance_bounds_m, orientation_policy,
                           tool_axis_max_tilt_rad, max_candidates) -> None:
    """Reject a malformed hover request before any capture or planning.

    The checks equal ``generate_entity_hover_goals`` and ``hover_orientation``
    (interval order and limits, policy, tilt cone only for ``tool_axis_down``).
    """
    if orientation_policy not in HOVER_ORIENTATION_POLICIES:
        raise ValueError("Unknown hover orientation policy.")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be an integer from 1 to 8.")
    if not isinstance(tip_offset_bounds_xy_m, (list, tuple)) or len(tip_offset_bounds_xy_m) != 2:
        raise ValueError("tip_offset_bounds_xy_m must contain X and Y intervals.")
    for interval, name in zip(tip_offset_bounds_xy_m, ("tip X offset", "tip Y offset")):
        _bounded_interval(interval, name=name, lower=-MAX_HOVER_TIP_OFFSET_M, upper=MAX_HOVER_TIP_OFFSET_M)
    _bounded_interval(clearance_bounds_m, name="clearance", lower=0., upper=MAX_HOVER_CLEARANCE_M)
    # The orientation owner validates the cone (finite, within its limit, and
    # only with tool_axis_down) without needing a measured scene.
    hover_orientation([0., 0., 0., 0., 0., 0., 1.], orientation_policy,
                      tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)


def entity_hover_proposals(*, entity: ToolEntityBox, anchor_xyz_m, current_tcp_pose_xyz_quat_xyzw,
                           tip_offset_bounds_xy_m, clearance_bounds_m, points_m, support_margin_m,
                           orientation_policy="preserve", max_candidates=8, boundary_inset_m=0.,
                           tool_axis_max_tilt_rad=0.) -> tuple[EntityHoverCandidate, ...]:
    """Ordered hover proposals for one measured anchor and observed cloud."""
    return generate_entity_hover_goals(
        current_tcp_pose_xyz_quat_xyzw=list(current_tcp_pose_xyz_quat_xyzw),
        anchor_xyz_m=list(np.asarray(anchor_xyz_m, dtype=float)),
        tip_offset_bounds_xy_m=tip_offset_bounds_xy_m, clearance_bounds_m=clearance_bounds_m,
        entity=entity, points_m=points_m, orientation_policy=orientation_policy,
        max_candidates=max_candidates, boundary_inset_m=boundary_inset_m,
        support_window_m=ENTITY_HOVER_SUPPORT_WINDOW_M, support_margin_m=support_margin_m,
        tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)


def hover_candidate_relation(candidate: EntityHoverCandidate, *, orientation_policy,
                             tool_axis_max_tilt_rad) -> dict:
    """Audit fields of one proposal (the saved-scene query records these)."""
    return {"tip_offset_xy_m": list(candidate.tip_offset_xy_m), "clearance_m": candidate.clearance_m,
            "support_height_m": candidate.metrics.support_height_m,
            "footprint_point_count": candidate.metrics.footprint_point_count,
            "overhead_point_count": candidate.metrics.overhead_point_count,
            "orientation_policy": orientation_policy, "tool_axis_max_tilt_rad": tool_axis_max_tilt_rad}


def validate_entity_hover_endpoint(*, entity: ToolEntityBox, endpoint_pose_xyz_quat_xyzw, anchor_xyz_m,
                                   points_m, support_margin_m, tip_offset_bounds_xy_m, clearance_bounds_m,
                                   orientation_policy, tool_axis_max_tilt_rad,
                                   rotation_tolerance_rad) -> tuple[dict, bool]:
    """Re-measure an achieved (planned FK) pose against the ORIGINAL request."""
    transform = pose_matrix(endpoint_pose_xyz_quat_xyzw)
    metrics = entity_hover_metrics(entity=entity, world_from_tcp=transform,
        anchor_xyz_m=np.asarray(anchor_xyz_m, dtype=float), points_m=points_m,
        support_window_m=ENTITY_HOVER_SUPPORT_WINDOW_M, support_margin_m=support_margin_m)
    (x_low, x_high), (y_low, y_high) = tip_offset_bounds_xy_m
    c_low, c_high = clearance_bounds_m
    dx, dy = metrics.tip_offset_xy_m
    within = (x_low - 1e-9 <= dx <= x_high + 1e-9 and y_low - 1e-9 <= dy <= y_high + 1e-9
              and c_low - 1e-9 <= metrics.clearance_m <= c_high + 1e-9)
    record = {"achieved_tip_offset_xy_m": [dx, dy], "achieved_clearance_m": metrics.clearance_m,
              "achieved_support_height_m": metrics.support_height_m,
              "achieved_footprint_point_count": metrics.footprint_point_count}
    if orientation_policy == "tool_axis_down":
        tilt = tool_axis_tilt_rad(transform[:3, :3])
        record["achieved_tool_axis_tilt_rad"] = tilt
        within = within and tilt <= tool_axis_max_tilt_rad + 1e-9 + rotation_tolerance_rad
    record["inside_requested_region"] = within
    return record, within


@dataclass(frozen=True)
class HoverPlanOutcome:
    """One host planner answer for one hover proposal; never an execution.

    ``status`` is ``path_found``, ``path_rejected`` (with ``retryable`` telling
    whether another proposal may still succeed) or ``query_incomplete`` (the
    per-candidate budget ended first). A found path carries its achieved
    endpoint pose in the public reference frame.
    """

    status: str
    reason: str
    retryable: bool = False
    endpoint_pose_xyz_quat_xyzw: tuple[float, ...] | None = None
    trajectory_duration_sec: float | None = None
    details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.status not in ("path_found", "path_rejected", "query_incomplete"):
            raise ValueError("Unknown hover planning status.")
        if (self.status == "path_found") != (self.endpoint_pose_xyz_quat_xyzw is not None):
            raise ValueError("Exactly a found path carries its planned endpoint.")
        if self.endpoint_pose_xyz_quat_xyzw is not None:
            pose = tuple(float(x) for x in self.endpoint_pose_xyz_quat_xyzw)
            if len(pose) != 7 or not all(math.isfinite(x) for x in pose):
                raise ValueError("Planned endpoint must contain seven finite values.")
            object.__setattr__(self, "endpoint_pose_xyz_quat_xyzw", pose)


def search_entity_hover_paths(*, entity: ToolEntityBox, anchor_xyz_m, current_tcp_pose_xyz_quat_xyzw,
                              points_m, support_margin_m, tip_offset_bounds_xy_m, clearance_bounds_m,
                              orientation_policy, tool_axis_max_tilt_rad, max_candidates,
                              translation_tolerance_m, rotation_tolerance_rad,
                              plan_candidate: Callable[..., HoverPlanOutcome],
                              cancel_requested: Callable[[], bool] = lambda: False,
                              clock: Callable[[], float] = time.monotonic,
                              deadline: float | None = None,
                              region_budget_sec: float = ENTITY_HOVER_REGION_BUDGET_SEC,
                              candidate_budget_sec: float = ENTITY_HOVER_CANDIDATE_BUDGET_SEC) -> dict:
    """First certified hover path inside explicit bounds, from one measured state.

    Proposals are inset by the arrival translation tolerance; every planned
    endpoint is re-measured against the ORIGINAL bounds. ``plan_candidate(pose,
    time_budget_sec=...)`` must plan only (no dispatch) from the same measured
    start for every proposal and stop within that budget in its own clock. The
    target a separate safe motion Skill may later be asked to reach; this
    search grants no execution authority.
    """
    validate_hover_request(tip_offset_bounds_xy_m=tip_offset_bounds_xy_m,
        clearance_bounds_m=clearance_bounds_m, orientation_policy=orientation_policy,
        tool_axis_max_tilt_rad=tool_axis_max_tilt_rad, max_candidates=max_candidates)
    for value, name in ((region_budget_sec, "region_budget_sec"), (candidate_budget_sec, "candidate_budget_sec")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    began = clock()
    stop = began + region_budget_sec if deadline is None else min(began + region_budget_sec, deadline)
    tip_bounds = [[float(x) for x in interval] for interval in tip_offset_bounds_xy_m]
    clearance_bounds = [float(x) for x in clearance_bounds_m]
    summary = {"semantics": TOOL_ENTITY_HOVER_SEMANTICS_V1, "status": "path_rejected",
               "reason": "region_no_candidate_path", "candidate_policy": "entity_proximity",
               "orientation_policy": orientation_policy, "tool_axis_max_tilt_rad": float(tool_axis_max_tilt_rad),
               "support_window_m": ENTITY_HOVER_SUPPORT_WINDOW_M, "support_margin_m": float(support_margin_m),
               "region_budget_sec": float(region_budget_sec), "candidate_budget_sec": float(candidate_budget_sec),
               "candidate_count": 0, "attempted_candidate_count": 0, "unresolved_candidate_count": 0,
               "candidate_attempts": [], "selected_candidate_index": None,
               "target_tcp_pose_xyz_quat_xyzw": None, "planned_endpoint_pose_xyz_quat_xyzw": None,
               "selected_tip_offset_xy_m": None, "selected_clearance_m": None,
               "achieved": None, "trajectory_duration_sec": None}
    if cancel_requested():
        raise InterruptedError("Hover query cancelled.")
    if clock() >= stop:
        candidates = ()
        summary.update(status="query_incomplete", reason="query_budget_exhausted")
    else:
        candidates = entity_hover_proposals(entity=entity, anchor_xyz_m=anchor_xyz_m,
            current_tcp_pose_xyz_quat_xyzw=current_tcp_pose_xyz_quat_xyzw,
            tip_offset_bounds_xy_m=tip_bounds, clearance_bounds_m=clearance_bounds, points_m=points_m,
            support_margin_m=support_margin_m, orientation_policy=orientation_policy,
            max_candidates=max_candidates, boundary_inset_m=translation_tolerance_m,
            tool_axis_max_tilt_rad=float(tool_axis_max_tilt_rad))
    summary["candidate_generation_wall_sec"] = clock() - began
    summary["candidate_count"] = len(candidates)
    for index, candidate in enumerate(candidates):
        if cancel_requested():
            raise InterruptedError("Hover query cancelled.")
        started = clock()
        remaining = min(float(candidate_budget_sec), stop - started)
        if remaining <= 0:
            summary.update(status="query_incomplete", reason="query_budget_exhausted")
            break
        summary["attempted_candidate_count"] += 1
        outcome = plan_candidate(tuple(float(x) for x in candidate.pose_xyz_quat_xyzw), time_budget_sec=remaining)
        if not isinstance(outcome, HoverPlanOutcome):
            raise TypeError("Hover planner must return a HoverPlanOutcome.")
        finished = clock()
        attempt = {"candidate_index": index, "status": outcome.status, "reason": outcome.reason,
                   "wall_sec": finished - started,
                   "proposal": hover_candidate_relation(candidate, orientation_policy=orientation_policy,
                       tool_axis_max_tilt_rad=float(tool_axis_max_tilt_rad))}
        summary["candidate_attempts"].append(attempt)
        overran = finished - started > remaining or finished > stop
        if outcome.status == "path_found":
            record, within = validate_entity_hover_endpoint(entity=entity,
                endpoint_pose_xyz_quat_xyzw=outcome.endpoint_pose_xyz_quat_xyzw, anchor_xyz_m=anchor_xyz_m,
                points_m=points_m, support_margin_m=support_margin_m, tip_offset_bounds_xy_m=tip_bounds,
                clearance_bounds_m=clearance_bounds, orientation_policy=orientation_policy,
                tool_axis_max_tilt_rad=float(tool_axis_max_tilt_rad), rotation_tolerance_rad=rotation_tolerance_rad)
            attempt.update(record)
            if overran:
                # A late answer is not a certified result of this budget.
                attempt["region_validation_budget_exceeded"] = True
                summary["unresolved_candidate_count"] += 1
                if finished < stop:
                    continue
                summary.update(status="query_incomplete", reason="query_budget_exhausted")
                break
            if not within:
                continue
            summary.update(status="path_found", reason="planned_not_executed", selected_candidate_index=index,
                target_tcp_pose_xyz_quat_xyzw=[float(x) for x in candidate.pose_xyz_quat_xyzw],
                planned_endpoint_pose_xyz_quat_xyzw=list(outcome.endpoint_pose_xyz_quat_xyzw),
                selected_tip_offset_xy_m=list(candidate.tip_offset_xy_m),
                selected_clearance_m=candidate.clearance_m, achieved=record,
                trajectory_duration_sec=outcome.trajectory_duration_sec)
            break
        if outcome.status == "query_incomplete":
            summary["unresolved_candidate_count"] += 1
            if clock() < stop:
                continue
            summary.update(status="query_incomplete", reason="query_budget_exhausted")
            break
        if not outcome.retryable:
            # A fatal start or scene fault applies to every proposal.
            summary.update(status="path_rejected", reason=outcome.reason)
            break
    finished = clock()
    summary["path_planning_wall_sec"] = finished - began
    if (summary["status"] == "path_rejected" and summary["reason"] == "region_no_candidate_path"
            and (summary["unresolved_candidate_count"] or finished > stop)):
        summary.update(status="query_incomplete", reason="region_candidate_search_incomplete")
    return summary


__all__ = [
    "ENTITY_HOVER_CANDIDATE_BUDGET_SEC",
    "ENTITY_HOVER_REGION_BUDGET_SEC",
    "ENTITY_HOVER_SUPPORT_WINDOW_M",
    "OBJECT_MASK_EROSION_PX",
    "RELATION_EVIDENCE_SCOPE",
    "RELATION_MAXIMUM_DEPTH_M",
    "SURFACE_AMBIGUITY_MINIMUM_VOXELS",
    "SURFACE_AMBIGUITY_SHARE",
    "SURFACE_CLUSTER_FIELDS",
    "HoverPlanOutcome",
    "MeasuredObjectSurface",
    "backproject_mask",
    "describe_tool_object_relations",
    "entity_hover_proposals",
    "hover_candidate_relation",
    "object_surface_from_mask",
    "pose_matrix",
    "pose_xyz_quat_xyzw",
    "search_entity_hover_paths",
    "surface_ambiguous",
    "validate_entity_hover_endpoint",
    "validate_hover_request",
]
