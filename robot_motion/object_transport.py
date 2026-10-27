"""Object-centric transport inputs. Observations do not grant motion authority.

An object frame is supplied by an observation owner (or simulator), not inferred
from a grasp score or an axis-aligned point-cloud bounding box. Partial geometry
cannot certify the swept volume of a held object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import hashlib
import math
from types import MappingProxyType

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


def tool_goal_from_object_goal(reference_from_object_goal, tool_from_object) -> np.ndarray:
    """Convert an observed object relation; no grasp or execution is inferred."""
    goal = rigid_pose(reference_from_object_goal, name="object goal")
    relation = rigid_pose(tool_from_object, name="observed tool_from_object")
    with np.errstate(over="ignore", invalid="ignore"):
        result = goal @ np.linalg.inv(relation)
    return rigid_pose(result, name="derived tool goal")


def object_translation_feedback_tool_goal(reference_from_tool, reference_from_object,
                                          reference_from_object_goal, nominal_tool_goal,
                                          *, maximum_correction_m: float = .003) -> np.ndarray:
    """Bound a translation-only correction; never replace a grasp relation.

    All four poses must share one reference frame and the measured pair must be
    correlated. Rotation remains the nominal tool rotation. Both the step and
    total offset from the original tool goal are bounded, not integrated.
    Collision checking and the original frozen-relation guard remain callers'
    responsibilities; this geometric proposal grants no execution authority.
    """
    if (isinstance(maximum_correction_m, (bool, np.bool_))
            or not np.isfinite(maximum_correction_m) or not 0 < maximum_correction_m <= .003):
        raise ValueError("Object feedback requires a positive correction bound at most 3mm")
    tool = rigid_pose(reference_from_tool, name="measured tool")
    obj = rigid_pose(reference_from_object, name="measured object")
    goal = rigid_pose(reference_from_object_goal, name="immutable object goal")
    nominal = rigid_pose(nominal_tool_goal, name="nominal tool goal")
    if math.hypot(*(tool[:3, 3]-nominal[:3, 3])) > maximum_correction_m:
        raise ValueError("Object feedback requires the tool inside the nominal correction region")
    desired = tool[:3, 3] + goal[:3, 3] - obj[:3, 3]
    offset = desired - nominal[:3, 3]
    length = math.hypot(*offset)
    if length > maximum_correction_m:
        desired = nominal[:3, 3] + offset * (maximum_correction_m/length)
    step = desired-tool[:3, 3]
    length = math.hypot(*step)
    if length > maximum_correction_m:
        desired = tool[:3, 3] + step * (maximum_correction_m/length)
    result = nominal.copy()
    result[:3, 3] = desired
    return rigid_pose(result, name="bounded object translation feedback tool goal")


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
        return tool_goal_from_object_goal(reference_from_object_goal, self.tool_from_object)

    def require_current(self, *, now: float, scene_revision: int, reference_frame_id: str,
                        object_id: str, side: str, max_age_sec: float = .2) -> None:
        if (not np.isfinite(now) or not np.isfinite(max_age_sec) or max_age_sec <= 0
                or not 0 <= now-self.measured_at <= max_age_sec):
            raise ValueError("Object/hand observation is stale or from the future")
        if (self.object_id, self.side, self.reference_frame_id, self.scene_revision) != (
                object_id, side, reference_frame_id, scene_revision):
            raise ValueError("Object/hand observation identity does not match the request")


@dataclass(frozen=True)
class SupportSurfaceObservation:
    """Measured support plane and a fully supported inner XY rectangle.

    top_z_m is the conservative plane height at the rectangle's XY center.
    The observation producer must prove that every point in that rectangle is
    supported; a rotated object's outer AABB or raw mask quantile bounds do
    not satisfy geometry_complete.
    """

    support_id: str
    reference_frame_id: str
    scene_revision: int
    measured_at: float
    top_z_m: float
    normal_reference_xyz: tuple[float, float, float]
    region_min_xy_m: tuple[float, float]
    region_max_xy_m: tuple[float, float]
    evidence_ref: str
    geometry_complete: bool
    source_measurement_times_sec: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("support_id", "reference_frame_id", "evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must identify measured support evidence")
        if (isinstance(self.scene_revision, bool) or not isinstance(self.scene_revision, int)
                or self.scene_revision < 0 or isinstance(self.measured_at, bool)
                or isinstance(self.top_z_m, bool) or not math.isfinite(self.measured_at)
                or self.measured_at < 0 or not math.isfinite(self.top_z_m)):
            raise ValueError("Support time, revision and height must be finite")
        if self.geometry_complete is not True:
            raise ValueError("Release approach requires complete conservative support geometry")
        normal = np.asarray(self.normal_reference_xyz, dtype=float)
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6, rtol=0)
                or normal[2] < math.cos(math.radians(1.))):
            raise ValueError("Release approach requires a measured near-horizontal upward support")
        object.__setattr__(self, "normal_reference_xyz", tuple(float(value) for value in normal))
        for name in ("region_min_xy_m", "region_max_xy_m"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite Station XY bounds")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if any(low >= high for low, high in zip(self.region_min_xy_m, self.region_max_xy_m)):
            raise ValueError("Support bounds must have positive area")
        times = tuple(self.source_measurement_times_sec)
        if times:
            if (len(times) != 2 or any(isinstance(value, (bool, np.bool_))
                    or not math.isfinite(value) or value < 0 for value in times)
                    or self.measured_at != max(times)):
                raise ValueError("Fused support must retain both source measurement times")
        object.__setattr__(self, "source_measurement_times_sec", times)


@dataclass(frozen=True)
class CalibratedBoxSupportObservation:
    """A complete known box and its measured pose; no native-body privilege is required.

    The producer must bind the box identity to its geometry catalog and the
    pose to an actual observation. A visible top face by itself is insufficient.
    This type does not grant motion or infer that a box is still stationary.
    """

    support_id: str
    reference_frame_id: str
    scene_revision: int
    measured_at: float
    reference_from_box: np.ndarray
    box_size_m: tuple[float, float, float]
    geometry_evidence_ref: str
    pose_evidence_ref: str
    geometry_complete: bool

    def __post_init__(self):
        for name in ("support_id", "reference_frame_id", "geometry_evidence_ref",
                     "pose_evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must identify an authenticated support source")
        if (isinstance(self.scene_revision, bool) or not isinstance(self.scene_revision, int)
                or self.scene_revision < 0 or isinstance(self.measured_at, (bool, np.bool_))
                or not math.isfinite(self.measured_at) or self.measured_at < 0):
            raise ValueError("Support pose revision and time are invalid")
        if self.geometry_complete is not True:
            raise ValueError("A visible or unknown box cannot certify complete support")
        if any(isinstance(x, (bool, np.bool_)) for x in self.box_size_m):
            raise ValueError("Known support box needs positive finite dimensions")
        size = np.asarray(self.box_size_m, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Known support box needs positive finite dimensions")
        object.__setattr__(self, "box_size_m", tuple(float(x) for x in size))
        object.__setattr__(self, "reference_from_box",
                           rigid_pose(self.reference_from_box, name="observed support box pose"))

    def bind_measured_top(
        self,
        *,
        plane_measured_at: float,
        measured_upper_top_z_m: float,
        measured_normal_reference_xyz: tuple[float, float, float],
        measured_region_center_xy_m: tuple[float, float],
        depth_evidence_ref: str,
        lateral_uncertainty_m: float,
        vertical_uncertainty_m: float,
        maximum_pair_skew_sec: float = .05,
        maximum_top_disagreement_m: float = .005,
    ) -> SupportSurfaceObservation:
        """Fuse a measured plane with a known complete box into an inner region.

        The square is inscribed in the top face after XY projection, including
        yaw and small tilt, then shrunk by the supplied lateral uncertainty.
        The top height is the larger of model and measured upper bounds plus
        vertical uncertainty. Consumer collision checking remains authoritative.
        """
        if not isinstance(depth_evidence_ref, str) or not depth_evidence_ref.strip():
            raise ValueError("Measured support plane requires depth evidence")
        for name, value in (("plane_measured_at", plane_measured_at),
                            ("measured_upper_top_z_m", measured_upper_top_z_m),
                            ("lateral_uncertainty_m", lateral_uncertainty_m),
                            ("vertical_uncertainty_m", vertical_uncertainty_m),
                            ("maximum_pair_skew_sec", maximum_pair_skew_sec),
                            ("maximum_top_disagreement_m", maximum_top_disagreement_m)):
            if (isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (maximum_pair_skew_sec <= 0 or maximum_pair_skew_sec > .05
                or maximum_top_disagreement_m <= 0 or maximum_top_disagreement_m > .005
                or abs(self.measured_at - plane_measured_at) > maximum_pair_skew_sec):
            raise ValueError("Support pose and plane are not a correlated cut")
        measured_normal = np.asarray(measured_normal_reference_xyz, dtype=float)
        if (measured_normal.shape != (3,) or not np.isfinite(measured_normal).all()
                or not math.isclose(float(np.linalg.norm(measured_normal)), 1., abs_tol=1e-6)):
            raise ValueError("Measured support normal is invalid")
        rotation = self.reference_from_box[:3, :3]
        model_normal = rotation[:, 2]
        if (model_normal[2] < math.cos(math.radians(1.))
                or measured_normal[2] < math.cos(math.radians(1.))
                or float(np.dot(model_normal, measured_normal)) < math.cos(math.radians(2.))):
            raise ValueError("Support model and measured top plane disagree in orientation")
        half = np.asarray(self.box_size_m) / 2.
        center = self.reference_from_box[:3, 3] + rotation @ np.array([0., 0., half[2]])
        projected_axes = rotation[:2, :2]
        inverse_axes = np.linalg.inv(projected_axes)
        radius = min(half[0] / np.sum(np.abs(inverse_axes[0])),
                     half[1] / np.sum(np.abs(inverse_axes[1]))) - lateral_uncertainty_m
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Known support top has no conservative inner region")
        measured_center_xy = np.asarray(measured_region_center_xy_m, dtype=float)
        if measured_center_xy.shape != (2,) or not np.isfinite(measured_center_xy).all():
            raise ValueError("Measured support patch center must be finite Station XY")
        local_patch_center = inverse_axes @ (measured_center_xy - center[:2])
        local_uncertainty = lateral_uncertainty_m * np.sum(np.abs(inverse_axes), axis=1)
        if np.any(np.abs(local_patch_center) + local_uncertainty >= half[:2]):
            raise ValueError("Measured top patch is outside the bound support box")
        model_z_at_patch = center[2] - (model_normal[:2]
            @ (measured_center_xy - center[:2])) / model_normal[2]
        if abs(model_z_at_patch - measured_upper_top_z_m) > maximum_top_disagreement_m:
            raise ValueError("Support model and measured top height disagree")
        corners = np.asarray(tuple(product((-1., 1.), repeat=2)), dtype=float)
        region_corners_xy = center[:2] + corners * radius
        measured_z_at_corners = measured_upper_top_z_m - (
            (region_corners_xy - measured_center_xy) @ measured_normal[:2]
        ) / measured_normal[2]
        model_slope_at_corners = -(
            (region_corners_xy - center[:2]) @ model_normal[:2]
        ) / model_normal[2]
        # A single output plane retains the known box normal, but its center
        # height must dominate the measured plane at *every* corner. The
        # maximum of two affine planes occurs at one of these corners.
        top_z = max(center[2], float(np.max(measured_z_at_corners - model_slope_at_corners)))
        top_z += vertical_uncertainty_m
        identity = (self.geometry_evidence_ref, self.pose_evidence_ref,
                    depth_evidence_ref, self.support_id, self.reference_frame_id,
                    str(self.scene_revision))
        evidence_ref = "support-box-fusion:" + hashlib.sha256(
            "\0".join(identity).encode("utf-8")
        ).hexdigest()
        return SupportSurfaceObservation(
            support_id=self.support_id,
            reference_frame_id=self.reference_frame_id,
            scene_revision=self.scene_revision,
            measured_at=max(self.measured_at, plane_measured_at),
            top_z_m=top_z,
            normal_reference_xyz=tuple(float(x) for x in model_normal),
            region_min_xy_m=tuple(float(x) for x in center[:2] - radius),
            region_max_xy_m=tuple(float(x) for x in center[:2] + radius),
            evidence_ref=evidence_ref,
            geometry_complete=True,
            source_measurement_times_sec=(self.measured_at, plane_measured_at),
        )


@dataclass(frozen=True)
class ReleaseApproachProposal:
    """A support-relative goal proposal; existing motion admission remains authoritative."""

    support_id: str
    reference_frame_id: str
    object_goal: np.ndarray
    tool_goal: np.ndarray
    release_clearance_m: float
    evidence_refs: tuple[str, str, str]

    def __post_init__(self):
        object.__setattr__(self, "object_goal", rigid_pose(self.object_goal, name="release object goal"))
        object.__setattr__(self, "tool_goal", rigid_pose(self.tool_goal, name="release tool goal"))


def _oriented_box_offsets(held: GraspedObjectObservation,
                          object_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    box = held.object_from_box
    half = np.asarray(held.box_size_m, dtype=float) / 2.
    corners_box = np.asarray(tuple(product((-1., 1.), repeat=3)), dtype=float) * half
    corners_object = corners_box @ box[:3, :3].T + box[:3, 3]
    offsets = corners_object @ object_rotation.T
    return offsets, offsets.min(axis=0), offsets.max(axis=0), object_rotation @ box[:3, 3]


def _release_approach_at_box_center(
    held: GraspedObjectObservation,
    support: SupportSurfaceObservation,
    *,
    object_rotation: np.ndarray,
    box_center_xy_m: np.ndarray,
    clearance_m: float,
    lateral_margin_m: float,
) -> ReleaseApproachProposal:
    offsets, low, high, object_center_offset = _oriented_box_offsets(held, object_rotation)
    object_goal = held.reference_from_object.copy()
    object_goal[:3, :3] = object_rotation
    object_goal[:2, 3] = box_center_xy_m - object_center_offset[:2]
    normal = np.asarray(support.normal_reference_xyz)
    support_center_xy = (np.asarray(support.region_min_xy_m)
                         + support.region_max_xy_m) / 2.
    corner_xy = object_goal[:2, 3] + offsets[:, :2]
    signed_without_object_z = ((corner_xy - support_center_xy) @ normal[:2]
                               + (offsets[:, 2] - support.top_z_m) * normal[2])
    object_goal[2, 3] = (clearance_m - float(np.min(signed_without_object_z))) / normal[2]
    footprint_min = object_goal[:2, 3] + low[:2]
    footprint_max = object_goal[:2, 3] + high[:2]
    if (np.any(footprint_min < np.asarray(support.region_min_xy_m) + lateral_margin_m)
            or np.any(footprint_max > np.asarray(support.region_max_xy_m) - lateral_margin_m)):
        raise ValueError("Complete held footprint does not fit on the selected support")
    object_goal = rigid_pose(object_goal, name="support-relative release object goal")
    return ReleaseApproachProposal(
        support_id=support.support_id, reference_frame_id=held.reference_frame_id,
        object_goal=object_goal, tool_goal=held.tool_goal(object_goal),
        release_clearance_m=float(clearance_m),
        evidence_refs=(held.pose_evidence_ref, held.grasp_evidence_ref, support.evidence_ref),
    )


def _require_release_observations(
    held: GraspedObjectObservation,
    support: SupportSurfaceObservation,
    *,
    now: float,
    clearance_m: float,
    minimum_planning_clearance_m: float,
    maximum_clearance_m: float,
    maximum_measurement_age_sec: float = .2,
    maximum_pair_skew_sec: float = .05,
    lateral_margin_m: float = 0.,
) -> None:
    if not isinstance(held, GraspedObjectObservation) or not isinstance(support, SupportSurfaceObservation):
        raise TypeError("Release planning needs typed held-object and support observations")
    if held.object_id == support.support_id:
        raise ValueError("A held object cannot support its own release")
    for name, value in (("now", now),
                        ("maximum_measurement_age_sec", maximum_measurement_age_sec),
                        ("clearance_m", clearance_m),
                        ("minimum_planning_clearance_m", minimum_planning_clearance_m),
                        ("maximum_clearance_m", maximum_clearance_m),
                        ("lateral_margin_m", lateral_margin_m),
                        ("maximum_pair_skew_sec", maximum_pair_skew_sec)):
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (maximum_measurement_age_sec <= 0 or maximum_pair_skew_sec <= 0
            or minimum_planning_clearance_m <= 0 or maximum_clearance_m <= 0
            or not minimum_planning_clearance_m <= clearance_m <= maximum_clearance_m):
        raise ValueError("Release clearance is outside the original planning profile")
    held.require_current(now=now, scene_revision=support.scene_revision,
        reference_frame_id=support.reference_frame_id, object_id=held.object_id,
        side=held.side, max_age_sec=maximum_measurement_age_sec)
    support_times = (support.measured_at, *support.source_measurement_times_sec)
    if any(not 0 <= now - stamp <= maximum_measurement_age_sec
           or abs(held.measured_at - stamp) > maximum_pair_skew_sec
           for stamp in support_times):
        raise ValueError("Held object and support require fresh correlated observations")


def plan_release_approach(
    held: GraspedObjectObservation,
    support: SupportSurfaceObservation,
    *,
    now: float,
    clearance_m: float,
    minimum_planning_clearance_m: float,
    maximum_clearance_m: float,
    maximum_measurement_age_sec: float = .2,
    maximum_pair_skew_sec: float = .05,
    lateral_margin_m: float = 0.,
    target_box_center_xy_m: tuple[float, float] | None = None,
) -> ReleaseApproachProposal:
    """Center a fully bounded held box above a measured support without contact.

    The release pose preserves the observed object orientation and hand-object
    relation. The support surface and object box must be complete observations;
    no segmentation-visible top face is promoted into an unseen object bottom.
    """
    _require_release_observations(
        held, support, now=now, clearance_m=clearance_m,
        minimum_planning_clearance_m=minimum_planning_clearance_m,
        maximum_clearance_m=maximum_clearance_m,
        maximum_measurement_age_sec=maximum_measurement_age_sec,
        maximum_pair_skew_sec=maximum_pair_skew_sec,
        lateral_margin_m=lateral_margin_m,
    )
    rotation = held.reference_from_object[:3, :3]
    support_center_xy = (np.asarray(support.region_min_xy_m)
                         + support.region_max_xy_m) / 2.
    if target_box_center_xy_m is None:
        center_xy = support_center_xy
    else:
        if (not isinstance(target_box_center_xy_m, (tuple, list, np.ndarray))
                or len(target_box_center_xy_m) != 2
                or any(isinstance(value, (bool, np.bool_)) for value in target_box_center_xy_m)):
            raise ValueError("Target box center must be finite Station XY")
        center_xy = np.asarray(target_box_center_xy_m, dtype=float)
        if center_xy.shape != (2,) or not np.isfinite(center_xy).all():
            raise ValueError("Target box center must be finite Station XY")
    return _release_approach_at_box_center(
        held, support, object_rotation=rotation,
        box_center_xy_m=center_xy, clearance_m=clearance_m,
        lateral_margin_m=lateral_margin_m,
    )


def plan_release_approach_for_relation(
    held: GraspedObjectObservation,
    support: SupportSurfaceObservation,
    *,
    relation: str,
    now: float,
    clearance_m: float,
    minimum_planning_clearance_m: float,
    maximum_clearance_m: float,
    lateral_margin_m: float = 0.,
    maximum_measurement_age_sec: float = .05,
    maximum_pair_skew_sec: float = .05,
) -> ReleaseApproachProposal:
    """Resolve an explicit support-frame relation without exposing pose floats.

    X-side names refer to the declared reference frame, never an image's
    visually ambiguous left/right. The original release planner still owns
    freshness, complete-footprint, clearance and frozen-relation validation.
    """
    if relation not in {"center", "reference_x_negative", "reference_x_positive"}:
        raise ValueError("Unsupported support relation")
    if not isinstance(held, GraspedObjectObservation) or not isinstance(support, SupportSurfaceObservation):
        raise TypeError("Relation placement needs typed held and support observations")
    if (isinstance(lateral_margin_m, (bool, np.bool_))
            or not math.isfinite(lateral_margin_m) or lateral_margin_m < 0):
        raise ValueError("Lateral margin must be finite and nonnegative")
    support_min = np.asarray(support.region_min_xy_m, dtype=float)
    support_max = np.asarray(support.region_max_xy_m, dtype=float)
    support_center = (support_min + support_max) / 2.
    half_box = np.asarray(held.box_size_m, dtype=float) / 2.
    corners = np.asarray(tuple(product((-1., 1.), repeat=3)), dtype=float) * half_box
    reference_box_rotation = (held.reference_from_object[:3, :3]
                              @ held.object_from_box[:3, :3])
    rotated = corners @ reference_box_rotation.T
    extent_xy = np.max(np.abs(rotated[:, :2]), axis=0)
    feasible_min = support_min + lateral_margin_m + extent_xy
    feasible_max = support_max - lateral_margin_m - extent_xy
    if not (np.isfinite(extent_xy).all() and np.isfinite(feasible_min).all()
            and np.isfinite(feasible_max).all()):
        raise ValueError("Support relation geometry must be finite")
    if np.any(feasible_min > feasible_max):
        raise ValueError("Complete held footprint has no feasible support center")
    target_center = support_center.copy()
    if relation == "reference_x_negative":
        if feasible_min[0] >= support_center[0] - 1e-9:
            raise ValueError("Selected support has no reference-negative-X placement region")
        target_center[0] = (feasible_min[0] + support_center[0]) / 2.
    elif relation == "reference_x_positive":
        if feasible_max[0] <= support_center[0] + 1e-9:
            raise ValueError("Selected support has no reference-positive-X placement region")
        target_center[0] = (feasible_max[0] + support_center[0]) / 2.
    return plan_release_approach(
        held, support, now=now, clearance_m=clearance_m,
        minimum_planning_clearance_m=minimum_planning_clearance_m,
        maximum_clearance_m=maximum_clearance_m,
        lateral_margin_m=lateral_margin_m,
        maximum_measurement_age_sec=maximum_measurement_age_sec,
        maximum_pair_skew_sec=maximum_pair_skew_sec,
        target_box_center_xy_m=tuple(float(value) for value in target_center),
    )


@dataclass(frozen=True)
class ReleaseApproachCandidate:
    """One measured-geometry proposal, not an IK, path or execution certificate."""

    box_center_xy_m: tuple[float, float]
    yaw_offset_rad: float
    proposal: ReleaseApproachProposal


def plan_release_approach_candidates(
    held: GraspedObjectObservation,
    support: SupportSurfaceObservation,
    *,
    now: float,
    clearance_m: float,
    minimum_planning_clearance_m: float,
    maximum_clearance_m: float,
    lateral_margin_m: float = 0.,
    maximum_measurement_age_sec: float = .05,
    maximum_pair_skew_sec: float = .05,
    allow_quarter_turns: bool = False,
) -> tuple[ReleaseApproachCandidate, ...]:
    """Bounded on-support proposals for the existing payload-safe planner.

    When it fits, the current-orientation center remains first. Four interior
    corners expose room for IK without asking an Agent to copy coordinates.
    Quarter turns are opt-in because a task may require the observed orientation. No
    candidate is an authority to move, and the selected support must be
    observed again when execution starts.
    """
    if type(allow_quarter_turns) is not bool:
        raise TypeError("Quarter-turn sampling must be explicitly boolean")
    _require_release_observations(
        held, support, now=now, clearance_m=clearance_m,
        minimum_planning_clearance_m=minimum_planning_clearance_m,
        maximum_clearance_m=maximum_clearance_m,
        maximum_measurement_age_sec=maximum_measurement_age_sec,
        maximum_pair_skew_sec=maximum_pair_skew_sec,
        lateral_margin_m=lateral_margin_m,
    )
    center = (np.asarray(support.region_min_xy_m)
              + support.region_max_xy_m) / 2.
    candidates = []
    angles = (0., -math.pi / 2., math.pi / 2., math.pi) if allow_quarter_turns else (0.,)
    fractions = ((0., 0.), (.85, .85), (.85, -.85), (-.85, .85), (-.85, -.85))
    for angle in angles:
        z_rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.],
            [math.sin(angle), math.cos(angle), 0.],
            [0., 0., 1.],
        ])
        rotation = z_rotation @ held.reference_from_object[:3, :3]
        _, low, high, center_offset = _oriented_box_offsets(held, rotation)
        feasible_min = np.asarray(support.region_min_xy_m) + lateral_margin_m - low[:2] + center_offset[:2]
        feasible_max = np.asarray(support.region_max_xy_m) - lateral_margin_m - high[:2] + center_offset[:2]
        if np.any(feasible_min > feasible_max):
            continue
        for x_fraction, y_fraction in fractions:
            fraction = np.array([x_fraction, y_fraction])
            span = np.where(fraction >= 0., feasible_max-center, center-feasible_min)
            box_center = center + fraction * span
            if np.any(box_center < feasible_min) or np.any(box_center > feasible_max):
                continue
            proposal = _release_approach_at_box_center(
                held, support, object_rotation=rotation,
                box_center_xy_m=box_center,
                clearance_m=clearance_m,
                lateral_margin_m=lateral_margin_m,
            )
            if any(item.yaw_offset_rad == angle and
                   np.allclose(item.box_center_xy_m, box_center, atol=1e-12, rtol=0.)
                   for item in candidates):
                continue
            candidates.append(ReleaseApproachCandidate(
                box_center_xy_m=tuple(float(value) for value in box_center),
                yaw_offset_rad=angle, proposal=proposal,
            ))
    if not candidates:
        raise ValueError("No complete held footprint fits the selected support")
    return tuple(candidates)


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
