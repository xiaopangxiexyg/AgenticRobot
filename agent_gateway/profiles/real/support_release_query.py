"""RealMan registration and port for ``perception.query_support_release_pose@2``.

One read-only request while the arm stands still:

1. capture one stationary aligned RGB-D cut from every bound camera
   (``RealToolScenePort``) and measure the held object in it with the
   deployment's held-object observer. The grip must be verified and must be the
   grip whose box an object motion retained since its lift
   (``RealHeldObjectPort.retained_grip_box``): only that box reaches down to
   where the object stood, a part the cameras stop seeing once it is lifted;
2. text-segment the named support in each view (the holding arm's wrist camera
   first) and certify up to three fully observed top rectangles from that
   view's depth with the real depth tolerance (``measure_real_support_tops``).
   The holding arm's wrist view is used when it certifies one (the place motion
   measures with that camera), otherwise the view with the largest rectangle;
3. propose release poses on those rectangles with the shared
   ``plan_release_approach_candidates`` (center, nearest to the held object and
   interior corners, a bounded yaw sample, then the lean pass) and, for each, a
   pre-place pose straight above it;
4. predict the place motion's own support check: where the holding arm's wrist
   camera already sees the release footprint, fit its plane as the place does
   (``measure_support_under``) and set the release height from it, otherwise
   from the certified plane (the place measures again from the pre-place pose);
5. check each candidate with a dedicated instance of the deployment's observed
   safe-move planner over the cloud of the same cut: the carry leg (current
   pose to pre-place, carried box) and the place leg (pre-place to release,
   carried box, support allowance, exact target). The first candidate with both
   legs planned is returned.

Nothing is dispatched, retained as a plan or authorized.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.back_projection import LiveAlignedDepthCapture
from agent_gateway.profiles.measured_support import (
    maximal_true_rectangles,
    plane_coverage_grid,
)
from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception import support_release_query_v2 as skill
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import (
    MAXIMUM_RELEASE_LEAN_RAD,
    GraspedObjectObservation,
    NoReleaseCandidates,
    SupportSurfaceObservation,
    plan_release_approach_candidates,
)
from robot_motion.relative_action import pose_matrix_to_xyzquat
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65CarriedObjectBox,
    RM65CarriedObjectSupport,
    RM65GraspTrajectoryPlanningError,
)

from .held_object import REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE
from .motion_observed_scene import obstacle_cloud_from_captures
from .object_motion import (
    REAL_OBJECT_MOTION_SUPPORT_HEIGHT_TOLERANCE_M,
    REAL_OBJECT_MOTION_SUPPORT_MARGIN_M,
    REAL_OBJECT_MOTION_SUPPORT_MIN_POINTS,
    RealHeldObjectPort,
    measure_support_under,
)

REAL_SUPPORT_RELEASE_MAX_OPERATION_MS = 120_000
# Pre-place pose: straight above the release pose, high enough that the carry
# keeps the observed clearance from the support top and the holding arm's
# wrist camera looks down on the release footprint before the place.
REAL_RELEASE_PREPLACE_HEIGHT_M = 0.05
# Yaw turns of the held object about the support normal; the held yaw first.
REAL_RELEASE_YAW_OFFSETS_RAD = (0., -math.pi / 2., math.pi / 2., math.pi)
# The lean pass of the MuJoCo query (calibrated there; the real reach differs).
REAL_RELEASE_LEAN_RAD = math.radians(6.)
REAL_RELEASE_LEAN_YAW_OFFSETS_RAD = (0.,)
REAL_RELEASE_LATERAL_MARGIN_M = 0.005
# Bound of the shared candidate generator; the Agent's clearance is the v2 band.
REAL_RELEASE_MAXIMUM_CLEARANCE_M = 0.05
REAL_SUPPORT_SEGMENTATION_MIN_SCORE = 0.3
# Real metric depth: a support top is a plane within this distance (p95), the
# same bound decides which grid samples are observed on it.
REAL_SUPPORT_PLANE_RESIDUAL_M = 0.006
# Camera calibration tilts a measured top by about a degree; a top tilted more
# than this is not a support. The release planner then treats it as horizontal
# at its highest point over the rectangle.
REAL_SUPPORT_MAX_TILT_DEG = 5.0
REAL_SUPPORT_GRID_M = 0.004
REAL_SUPPORT_MIN_REGION_SIDE_M = 0.03
REAL_SUPPORT_MIN_TOP_POINTS = 150
REAL_SUPPORT_MAX_GRID_CELLS = 100_000
REAL_SUPPORT_REGIONS = 3
# The holding arm's wrist plane and the certified top may disagree by camera
# calibration; more than this means the wrist sees another surface.
REAL_SUPPORT_HEIGHT_DISAGREEMENT_M = 0.02
REAL_RELEASE_PLANNING_BUDGET_SEC = 60.
REAL_RELEASE_LEG_BUDGET_SEC = 8.
REAL_RELEASE_MAX_CHECKED_CANDIDATES = 24
# One stationary cut: every view and the held object share it.
REAL_RELEASE_MAX_OBSERVATION_AGE_SEC = 1.0
_RESULT_RESERVE_SEC = 1.0


def _translation(xyz) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


def _box_corners(world_from_box, size) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    world_from_box = np.asarray(world_from_box, dtype=float)
    return (signs * np.asarray(size, dtype=float)) @ world_from_box[:3, :3].T + world_from_box[:3, 3]


def _pose_list(pose) -> list[float]:
    return [float(value) for value in pose_matrix_to_xyzquat(np.asarray(pose, dtype=float))]


# Support top measurement from real depth -------------------------------------


@dataclass(frozen=True)
class RealMeasuredSupportTop:
    """A certified, fully observed rectangle on one support top measured from real depth.

    Every 4 mm grid sample of the rectangle projects into the support mask with
    a measured depth point within ``plane_residual_m`` of the fitted plane;
    occluded or unobserved parts are excluded, never assumed.
    """

    plane_anchor_xyz_m: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    region_min_xy_m: tuple[float, float]
    region_max_xy_m: tuple[float, float]
    observed_min_xy_m: tuple[float, float]
    observed_max_xy_m: tuple[float, float]
    plane_residual_m: float
    tolerance_m: float
    tilt_deg: float
    top_point_count: int
    footprint_sample_count: int

    @property
    def area_m2(self) -> float:
        return float((self.region_max_xy_m[0] - self.region_min_xy_m[0])
                     * (self.region_max_xy_m[1] - self.region_min_xy_m[1]))

    def height_at(self, xy) -> float:
        anchor = np.asarray(self.plane_anchor_xyz_m)
        normal = np.asarray(self.normal_xyz)
        return float(anchor[2] - (normal[0] * (float(xy[0]) - anchor[0])
                                  + normal[1] * (float(xy[1]) - anchor[1])) / normal[2])

    @property
    def top_z_m(self) -> float:
        """Highest plane point over the rectangle plus the depth tolerance."""
        return max(self.height_at((x, y)) for x in (self.region_min_xy_m[0], self.region_max_xy_m[0])
                   for y in (self.region_min_xy_m[1], self.region_max_xy_m[1])) + self.tolerance_m

    def observation(self, *, support_id: str, reference_frame_id: str, scene_revision: int,
                    measured_at: float, evidence_ref: str) -> SupportSurfaceObservation:
        # The shared release generator takes a horizontal support: a measured
        # tilt (camera calibration, a few degrees at most) is covered by the
        # conservative top height. The release height is set from the plane later.
        return SupportSurfaceObservation(
            support_id=support_id, reference_frame_id=reference_frame_id, scene_revision=scene_revision,
            measured_at=measured_at, top_z_m=self.top_z_m, normal_reference_xyz=(0., 0., 1.),
            region_min_xy_m=self.region_min_xy_m, region_max_xy_m=self.region_max_xy_m,
            evidence_ref=evidence_ref, geometry_complete=True)

    def evidence(self) -> dict:
        return {"top_point_count": int(self.top_point_count), "plane_residual_m": float(self.plane_residual_m),
                "plane_tilt_deg": float(self.tilt_deg), "footprint_sample_count": int(self.footprint_sample_count),
                "observed_region_min_xy_m": [float(v) for v in self.observed_min_xy_m],
                "observed_region_max_xy_m": [float(v) for v in self.observed_max_xy_m]}


def _mask_points(capture: LiveAlignedDepthCapture, mask: np.ndarray, maximum_depth_m: float) -> np.ndarray:
    depth = capture.depth_m
    valid = mask & np.isfinite(depth) & (depth > 0.) & (depth <= maximum_depth_m)
    rows, cols = np.nonzero(valid)
    intrinsics = capture.intrinsics
    d = depth[rows, cols].astype(np.float64)
    camera = np.column_stack(((cols - float(intrinsics["cx"])) * d / float(intrinsics["fx"]),
                              (rows - float(intrinsics["cy"])) * d / float(intrinsics["fy"]), d))
    transform = capture.reference_from_camera
    return camera @ transform[:3, :3].T + transform[:3, 3]


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - centre, full_matrices=False)
    if singular[1] <= 1e-6:
        raise ValueError("Support top is geometrically degenerate")
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return centre, normal


def measure_real_support_tops(capture: LiveAlignedDepthCapture, mask: np.ndarray, *, maximum_depth_m: float,
                              plane_residual_m: float = REAL_SUPPORT_PLANE_RESIDUAL_M,
                              maximum_tilt_deg: float = REAL_SUPPORT_MAX_TILT_DEG,
                              grid_m: float = REAL_SUPPORT_GRID_M,
                              lateral_margin_m: float = REAL_RELEASE_LATERAL_MARGIN_M,
                              minimum_region_side_m: float = REAL_SUPPORT_MIN_REGION_SIDE_M,
                              minimum_top_points: int = REAL_SUPPORT_MIN_TOP_POINTS,
                              limit: int = REAL_SUPPORT_REGIONS) -> tuple[RealMeasuredSupportTop, ...]:
    """Certify up to ``limit`` distinct fully observed top rectangles, largest first.

    The top plane is fitted to the highest fifth of the masked points (sides
    seen at an angle lie lower), grown to every masked point within
    ``plane_residual_m`` of it and refitted. A mask clipped by the image border
    is allowed: only the observed part of the top is certified.
    """
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must be a paired full-resolution boolean grid")
    for name, value in (("plane_residual_m", plane_residual_m), ("maximum_tilt_deg", maximum_tilt_deg),
                        ("grid_m", grid_m), ("minimum_region_side_m", minimum_region_side_m)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(lateral_margin_m, bool) or not math.isfinite(lateral_margin_m) or lateral_margin_m < 0:
        raise ValueError("lateral_margin_m must be finite and non-negative")
    points = _mask_points(capture, selected, maximum_depth_m)
    if len(points) < minimum_top_points:
        raise ValueError("Support mask has insufficient valid metric depth")
    seed = points[points[:, 2] >= float(np.quantile(points[:, 2], .8)) - 1e-9]
    if len(seed) < 16:
        raise ValueError("Support top has insufficient measured coverage")
    anchor, normal = _fit_plane(seed)
    for _ in range(3):
        inliers = points[np.abs((points - anchor) @ normal) <= plane_residual_m]
        if len(inliers) < minimum_top_points:
            raise ValueError("Support top has insufficient measured coverage")
        anchor, normal = _fit_plane(inliers)
    inliers = points[np.abs((points - anchor) @ normal) <= plane_residual_m]
    if len(inliers) < minimum_top_points:
        raise ValueError("Support top has insufficient measured coverage")
    residual = float(np.quantile(np.abs((inliers - anchor) @ normal), .95))
    tilt = math.degrees(math.acos(max(-1., min(1., float(normal[2])))))
    if tilt > maximum_tilt_deg:
        raise ValueError("Observed support top is not near-horizontal")
    low = inliers[:, :2].min(axis=0)
    high = inliers[:, :2].max(axis=0)
    counts = np.floor((high - low) / grid_m).astype(int) + 1
    if int(counts[0]) * int(counts[1]) > REAL_SUPPORT_MAX_GRID_CELLS:
        raise ValueError("Support extent exceeds the bounded sample budget")
    xs = low[0] + grid_m * np.arange(int(counts[0]))
    ys = low[1] + grid_m * np.arange(int(counts[1]))
    plane = SimpleNamespace(plane_anchor_reference_xyz_m=tuple(float(v) for v in anchor),
                            normal_reference_xyz=tuple(float(v) for v in normal))
    grid = plane_coverage_grid(capture, selected, plane, xs=xs, ys=ys, maximum_plane_residual_m=plane_residual_m)
    tops = []
    for row0, col0, row1, col1 in maximal_true_rectangles(grid, limit=limit):
        region_min = np.array([xs[col0], ys[row0]]) + lateral_margin_m
        region_max = np.array([xs[col1], ys[row1]]) - lateral_margin_m
        if np.any(region_max - region_min < minimum_region_side_m):
            continue
        tops.append(RealMeasuredSupportTop(
            plane_anchor_xyz_m=plane.plane_anchor_reference_xyz_m, normal_xyz=plane.normal_reference_xyz,
            region_min_xy_m=(float(region_min[0]), float(region_min[1])),
            region_max_xy_m=(float(region_max[0]), float(region_max[1])),
            observed_min_xy_m=(float(low[0]), float(low[1])), observed_max_xy_m=(float(high[0]), float(high[1])),
            plane_residual_m=residual, tolerance_m=float(plane_residual_m), tilt_deg=tilt,
            top_point_count=len(inliers), footprint_sample_count=int((row1 - row0 + 1) * (col1 - col0 + 1))))
    if not tops:
        raise ValueError("Observed support top is smaller than the minimum release region")
    return tuple(tops)


# Registration ----------------------------------------------------------------


@dataclass(frozen=True)
class RealSupportReleaseRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet
    profile_hash: str
    roles: tuple[str, ...]
    prompt_by_support_id: Mapping[str, str]
    support_object_ids: tuple[str, ...]
    reference_frame_id: str
    side: str


def build_real_support_release_registration(*, segmentation, back_projection, prompt_by_support_id,
                                            object_motion_route, object_motion_profile,
                                            planner_profile_hash: str) -> RealSupportReleaseRegistration:
    """Bind the query to the segmenter, the aligned RGB-D roles, the object motion Route and its planner."""
    if not (0. < REAL_RELEASE_LEAN_RAD <= MAXIMUM_RELEASE_LEAN_RAD
            and MAXIMUM_RELEASE_LEAN_RAD == skill.MAX_RELEASE_LEAN_RAD):
        raise ValueError("Release lean limits of the planner and the result schema differ.")
    if not 0. < REAL_RELEASE_PREPLACE_HEIGHT_M <= skill.MAX_PREPLACE_HEIGHT_M:
        raise ValueError("The pre-place height exceeds the result schema's bound.")
    prompts = {str(key): str(value) for key, value in dict(prompt_by_support_id).items()}
    supports = tuple(sorted(prompts))
    if not supports or any(not key.strip() or key != key.strip() or not value.strip()
                           for key, value in prompts.items()):
        raise ValueError("Release queries need named supports with non-empty prompts.")
    if not isinstance(planner_profile_hash, str) or not planner_profile_hash:
        raise ValueError("Release queries need their payload planner identity.")
    roles = tuple(back_projection.profile.roles)
    side = str(object_motion_profile.side)
    if REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE.get(side) not in roles:
        raise ValueError("Release queries need the holding arm's wrist camera.")
    if any(segmentation.profile.source_id_by_role.get(role)
           != back_projection.profile.source_id_by_role.get(role) for role in roles):
        raise ValueError("Release query roles need matching segmentation and aligned RGB-D sources.")
    for key in ("runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
                "execution_manifest_hash", "target_kind"):
        if getattr(segmentation.binding, key) != getattr(back_projection.binding, key):
            raise ValueError("Release query sources belong to another deployment.")
    frame = back_projection.profile.reference_frame_id
    if frame != object_motion_profile.reference_frame_id:
        # Both returned poses are submitted unchanged to the object motion Route.
        raise ValueError("Release query must use the object motion Route's public frame.")
    for prompt in prompts.values():
        for role in roles:
            segmentation.payload_validator.validate_parameters({
                "role": role, "language_prompt": prompt, "target_pixel_uv": None,
                "min_score": REAL_SUPPORT_SEGMENTATION_MIN_SCORE})
    profile_hash = canonical_content_hash({
        "segmentation_route": segmentation.skill.routes[0].semantic_content_hash,
        "segmentation_profile": segmentation.profile.semantic_content_hash,
        "back_projection_profile": back_projection.profile.semantic_content_hash,
        "roles": list(roles), "view_selection": "holding_wrist_if_certified_else_largest_certified_rectangle",
        "prompt_by_support_id": dict(sorted(prompts.items())),
        "segmentation_min_score": REAL_SUPPORT_SEGMENTATION_MIN_SCORE,
        "support_measurement": {"plane": "upper_fifth_seed_grown_within_residual", "mask_border": "observed_part_only",
                                "plane_residual_m": REAL_SUPPORT_PLANE_RESIDUAL_M,
                                "maximum_tilt_deg": REAL_SUPPORT_MAX_TILT_DEG, "grid_m": REAL_SUPPORT_GRID_M,
                                "minimum_region_side_m": REAL_SUPPORT_MIN_REGION_SIDE_M,
                                "minimum_top_points": REAL_SUPPORT_MIN_TOP_POINTS,
                                "regions": {"maximum": REAL_SUPPORT_REGIONS, "order": "largest_first"}},
        "object_motion_route": object_motion_route.semantic_content_hash,
        "object_motion_profile": object_motion_profile.semantic_content_hash,
        "held_object": "verified_grip_retained_since_its_lift",
        "yaw_offsets_rad": list(REAL_RELEASE_YAW_OFFSETS_RAD),
        "positions_per_yaw": ["center", "nearest_to_held", "interior_corners_0.85"],
        "lean": {"rad": REAL_RELEASE_LEAN_RAD, "yaw_offsets_rad": list(REAL_RELEASE_LEAN_YAW_OFFSETS_RAD),
                 "toward": "holding_arm_base", "order": "after_the_held_orientation_candidates_of_each_region",
                 "maximum_tilt_from_support_normal_rad": MAXIMUM_RELEASE_LEAN_RAD},
        "lateral_margin_m": REAL_RELEASE_LATERAL_MARGIN_M,
        "preplace_height_m": REAL_RELEASE_PREPLACE_HEIGHT_M,
        "release_height": "place_check_plane_of_the_holding_wrist_if_seen_else_certified_plane",
        "support_height_disagreement_m": REAL_SUPPORT_HEIGHT_DISAGREEMENT_M,
        "legs": {"carry": "within_arrival_tolerance", "place": "preserve_target_with_support_allowance"},
        "planning_time_budget_sec": REAL_RELEASE_PLANNING_BUDGET_SEC,
        "leg_time_budget_sec": REAL_RELEASE_LEG_BUDGET_SEC,
        "max_checked_candidates": REAL_RELEASE_MAX_CHECKED_CANDIDATES,
        "planner": planner_profile_hash,
    }, domain="agenticrobot.real.support_release_query.profile.v1")
    definition = skill.definition()
    schema = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash=profile_hash,
                          support_object_ids=supports, reference_frame_id=frame)
    io_contract = inline_only_skill_io_contract(
        skill_id=skill.SKILL_ID, skill_version=skill.SKILL_VERSION,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    base_route = back_projection.skill.routes[0]
    if base_route.control_effect_class != "read_only" or base_route.may_move_robot:
        raise ValueError("Release queries require a read-only base Route.")
    route = replace(base_route, skill_id=skill.SKILL_ID, skill_version=skill.SKILL_VERSION,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_route.route_id + ".support_release_pose", route_version=skill.SKILL_VERSION,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash,
        max_operation_duration_ms=REAL_SUPPORT_RELEASE_MAX_OPERATION_MS)
    base_binding = back_projection.binding.to_ref()
    binding = replace(base_binding,
        deployment_binding_id="support-release-pose:" + base_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": base_binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.real.support_release_query.binding.v1"))

    def validate_parameters(value):
        skill.validate_parameters(value, support_object_ids=supports, reference_frame_id=frame)

    def validate_result(value):
        skill.validate_result(value, support_object_ids=supports, reference_frame_id=frame)

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        skill.validate_result_for_parameters(parameters, result)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="real-support-release-pose:" + profile_hash, skill_id=skill.SKILL_ID,
        skill_version=skill.SKILL_VERSION, route_id=route.route_id, route_version=skill.SKILL_VERSION,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    entry = SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
        availability_hash=canonical_content_hash({
            "route": route.semantic_content_hash, "profile": profile_hash,
            "status": "configured_requires_lifted_held_object_live_rgbd_text_segmentation_and_payload_planner"},
            domain="agenticrobot.real.support_release_query.availability.v1"))
    return RealSupportReleaseRegistration(
        SkillRegistrationTemplate(definition, schema, io_contract, (route,)), entry, validator,
        profile_hash, roles, prompts, supports, frame, side)


# Plan-only legs --------------------------------------------------------------


class RealSupportReleasePlanner:
    """Plan-only carry and place legs with the deployment's observed safe-move semantics.

    A dedicated planner instance (``planner_factory``, same reviewed
    configuration as the safe move): a query never holds the motion planner's
    lock, never dispatches and never retains a plan.
    """

    def __init__(self, *, planner_factory: Callable[[], object], self_filter, observation_config,
                 geometry_frame_id: str, start_clearance: str, speed_scale: float,
                 profile_identity: Mapping[str, object], max_inter_camera_skew_sec: float = 0.1):
        if not callable(planner_factory) or self_filter is None:
            raise TypeError("Release planning needs a planner factory and the captured-robot filter.")
        speed = float(speed_scale)
        skew = float(max_inter_camera_skew_sec)
        if not 0. < speed <= 1. or not 0. < skew < float("inf"):
            raise ValueError("Release planning needs a speed in (0, 1] and a finite camera skew bound.")
        self._planner_factory = planner_factory
        self._planner = None
        self._lock = threading.Lock()
        self.self_filter = self_filter
        self.observation_config = observation_config
        self.geometry_frame_id = geometry_frame_id
        self.start_clearance = start_clearance
        self.speed_scale = speed
        self.max_inter_camera_skew_sec = skew
        self.profile_hash = canonical_content_hash({
            **dict(profile_identity), "speed_scale": speed, "start_clearance": start_clearance,
            "max_inter_camera_skew_sec": skew,
            "legs": {"carry": "within_arrival_tolerance", "place": "preserve_target"}},
            domain="agenticrobot.real.support_release_planner.v1")

    def _instance(self):
        with self._lock:
            if self._planner is None:
                self._planner = self._planner_factory()
            return self._planner

    def prepare_cloud(self, *, captures, planning_qpos_by_side, cancel_requested):
        if cancel_requested():
            raise InterruptedError("Support release query cancelled.")
        # Build the planner before the search so its one-time preparation is
        # never charged to a single candidate's budget.
        self._instance()
        return obstacle_cloud_from_captures(captures=captures, config=self.observation_config,
            reference_frame_id=self.geometry_frame_id, self_filter=self.self_filter,
            planning_qpos_by_side=planning_qpos_by_side,
            max_inter_camera_skew_sec=self.max_inter_camera_skew_sec)

    def plan_leg(self, *, cloud, qpos_by_side, side, target_tool_tcp_world, carried_object,
                 carried_object_support=None, pose_goal_policy, time_budget_sec, cancel_requested):
        """End configuration of a planned leg and ``path_found``, or None and the planner's reason."""
        if cancel_requested():
            raise InterruptedError("Support release query cancelled.")
        planner = self._instance()
        try:
            trajectory = planner.plan_with_observed_cloud(cloud,
                qpos_by_side={s: np.asarray(q, dtype=float).copy() for s, q in qpos_by_side.items()},
                side=side, target_tool_tcp_world=np.asarray(target_tool_tcp_world, dtype=float),
                speed_scale=self.speed_scale, require_grasp_approach=False,
                time_budget_sec=float(time_budget_sec), pose_goal_policy=pose_goal_policy,
                carried_object=carried_object,
                **({} if carried_object_support is None else {"carried_object_support": carried_object_support}))
        except RM65GraspTrajectoryPlanningError as error:
            return None, str(error.reason_code)
        return ({s: np.asarray(trajectory.qpos_by_side[s][-1], dtype=float).copy() for s in ("left", "right")},
                "path_found")


# Query -------------------------------------------------------------------------


def _rejection_list(counts: Mapping[str, int]) -> list[dict]:
    return [{"reason_code": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def _rejected(*, parameters, revision, frame, reason, held_pose=None, segmentation=None, measured=None,
              candidate_count=0, attempted=0, planning_wall_sec=0., views=(), counts=None):
    return {
        "semantics": skill.SEMANTICS, "status": "no_release_pose", "reason": reason,
        "support_object_id": parameters["support_object_id"], "reference_frame_id": frame,
        "station_revision": revision, "release_object_pose_xyz_quat_xyzw": None,
        skill.PREPLACE_KEY: None, "release_clearance_m": skill.requested_clearance_m(parameters),
        "held_object_pose_xyz_quat_xyzw": held_pose, "yaw_change_rad": None, "release_lean_rad": None,
        "support_top_z_m": None if measured is None else measured["top_z_m"],
        "support_region_min_xy_m": None if measured is None else measured["region_min_xy_m"],
        "support_region_max_xy_m": None if measured is None else measured["region_max_xy_m"],
        "segmentation": segmentation, "measured_support": None if measured is None else measured["evidence"],
        "candidate_count": candidate_count, "attempted_candidate_count": attempted,
        "selected_candidate_index": None, "path_planning_wall_sec": float(planning_wall_sec),
        "candidate_rejections": _rejection_list(counts or {}), "views": list(views),
        "motion_executed": False, "execution_authorized": False}


@dataclass(frozen=True)
class _Checked:
    reason: str
    release: np.ndarray | None = None
    preplace: np.ndarray | None = None
    support_top_z_m: float | None = None
    plane_source: str | None = None


class RealSupportReleaseQuery:
    """Held box, measured support and plan-only legs from one stationary cut."""

    def __init__(self, *, registration: RealSupportReleaseRegistration, scene_port, segmentation_provider,
                 held_port: RealHeldObjectPort, planner: RealSupportReleasePlanner, object_id: str,
                 public_from_geometry, object_workspace_bounds, arm_base_xy_m, maximum_depth_m: float,
                 clock: Callable[[], float]):
        if not isinstance(registration, RealSupportReleaseRegistration):
            raise TypeError("Release query needs its typed registration.")
        if not isinstance(held_port, RealHeldObjectPort) or not isinstance(planner, RealSupportReleasePlanner):
            raise TypeError("Release query needs the held-object port and its plan-only planner.")
        if not callable(getattr(scene_port, "capture_views", None)) or not callable(
                getattr(segmentation_provider, "try_segment_capture", None)):
            raise TypeError("Release query needs stationary captures and segmentation.")
        if tuple(held_port.observer.roles) != registration.roles:
            raise ValueError("Release query and held-object observation must measure the same views.")
        bounds = tuple(float(value) for value in object_workspace_bounds)
        if len(bounds) != 6 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("Release query needs the object motion's workspace bounds.")
        if not callable(clock) or not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0:
            raise ValueError("Release query needs the Gateway clock and a positive depth bound.")
        self.registration = registration
        self.scene_port = scene_port
        self.segmentation_provider = segmentation_provider
        self.held_port = held_port
        self.planner = planner
        self.side = registration.side
        self.object_id = str(object_id)
        self.public_from_geometry = np.asarray(public_from_geometry, dtype=float).reshape(4, 4).copy()
        self.geometry_from_public = np.linalg.inv(self.public_from_geometry)
        self.object_workspace_bounds = bounds
        self.arm_base_xy_m = (float(arm_base_xy_m[0]), float(arm_base_xy_m[1]))
        self.maximum_depth_m = float(maximum_depth_m)
        self.clock = clock
        primary = REAL_HELD_OBJECT_PRIMARY_ROLE_BY_SIDE[self.side]
        self.primary_role = primary
        self.view_order = (primary, *(role for role in registration.roles if role != primary))
        self._lock = threading.Lock()

    def query(self, parameters: Mapping, *, expected_station_revision: int,
              cancel_requested: Callable[[], bool], deadline: float) -> tuple[dict, tuple[str, ...]]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Support release query is already running.")
        try:
            return self._query(parameters, revision=expected_station_revision,
                               cancel_requested=cancel_requested, deadline=deadline)
        finally:
            self._lock.release()

    def _depth_capture(self, capture) -> LiveAlignedDepthCapture:
        intrinsics = dict(capture.intrinsics)
        intrinsics.setdefault("width", int(capture.width))
        intrinsics.setdefault("height", int(capture.height))
        return LiveAlignedDepthCapture(
            source_id=capture.source_id, role=capture.role, captured_at=float(capture.acquired_at),
            station_revision=int(capture.station_revision), capture_group_id=capture.capture_group_id,
            depth_content_hash=capture.depth_content_hash, intrinsics=intrinsics,
            reference_from_camera=self.public_from_geometry @ np.asarray(
                capture.source_from_camera, dtype=float).reshape(4, 4),
            depth_m=np.asarray(capture.depth_m, dtype=np.float32))

    def _query(self, parameters, *, revision, cancel_requested, deadline):
        registration = self.registration
        frame = registration.reference_frame_id
        support_id = parameters["support_object_id"]
        clearance = skill.requested_clearance_m(parameters)
        common = {"parameters": parameters, "revision": revision, "frame": frame}
        side = self.side
        captures = tuple(self.scene_port.capture_views(
            roles=registration.roles, expected_station_revision=revision, cancel_requested=cancel_requested))
        evidence = ["real-rgbd-cut:" + capture.role + ":" + str(capture.capture_group_id) for capture in captures]
        observation = self.held_port.observer.observe(
            side=side, label="held object", language_prompt=None, min_score=self.held_port.min_score,
            expected_station_revision=revision, cancel_requested=cancel_requested, roles=registration.roles,
            captures=captures)
        evidence.append(observation.evidence_ref)
        box = self.held_port.retained_grip_box(side, observation) if observation.holding_verified else None
        if box is None:
            # Not held, or not the grip an object motion lifted: the box would
            # miss the object's lower part.
            return _rejected(reason="held_object_not_verified_lifted", **common), tuple(evidence)
        low, high = (np.asarray(value, dtype=float) for value in box)
        size = high - low
        tool_from_object = _translation((low + high) / 2.)
        world_from_tcp = np.asarray(observation.world_from_tcp, dtype=float)
        public_from_tool = self.public_from_geometry @ world_from_tcp
        current = public_from_tool @ tool_from_object
        held_pose = _pose_list(current)
        by_role = {capture.role: capture for capture in captures}
        now = max(float(capture.acquired_at) for capture in captures)
        held = GraspedObjectObservation(
            object_id=self.object_id, side=side, reference_frame_id=frame, scene_revision=revision,
            measured_at=float(by_role[self.primary_role].acquired_at), reference_from_object=current,
            reference_from_tool=public_from_tool,
            qpos_by_side={s: np.asarray(q, dtype=float) for s, q in observation.capture_qpos_by_side.items()},
            box_size_m=tuple(float(v) for v in size), object_from_box=np.eye(4), geometry_complete=True,
            pose_evidence_ref=observation.evidence_ref, grasp_evidence_ref=observation.evidence_ref)
        prompt = registration.prompt_by_support_id[support_id]
        views, measured_views = [], {}
        for role in self.view_order:
            if cancel_requested():
                raise InterruptedError("Support release query cancelled.")
            capture = by_role[role]
            prepared = self.segmentation_provider.try_segment_capture(
                capture=self.scene_port.segmentation_input(capture),
                parameters={"role": role, "language_prompt": prompt, "target_pixel_uv": None,
                            "min_score": REAL_SUPPORT_SEGMENTATION_MIN_SCORE})
            if prepared is None:
                views.append({"role": role, "outcome": "not_segmented", "score": None, "certified_area_m2": None})
                continue
            result = prepared.execution_result
            mask = decode_segmentation_mask_npy(
                prepared.mask_npy, width=int(result["image_width"]), height=int(result["image_height"]),
                expected_count=int(result["mask_pixel_count"]),
                expected_bounds=tuple(int(v) for v in result["mask_bounds_xyxy"]))
            segmentation = {"role": role, "language_prompt": prompt, "score": float(result["score"]),
                            "mask_pixel_count": int(result["mask_pixel_count"]),
                            "mask_bounds_xyxy": [int(v) for v in result["mask_bounds_xyxy"]]}
            evidence.append("support-mask:" + role + ":" + str(result["mask_media_content_hash"]))
            try:
                tops = measure_real_support_tops(self._depth_capture(capture), mask,
                                                 maximum_depth_m=self.maximum_depth_m)
            except ValueError:
                views.append({"role": role, "outcome": "top_not_measured", "score": segmentation["score"],
                              "certified_area_m2": None})
                continue
            views.append({"role": role, "outcome": "measured", "score": segmentation["score"],
                          "certified_area_m2": tops[0].area_m2})
            measured_views[role] = (tops, segmentation, str(result["mask_media_content_hash"]), capture)
        common["views"] = views
        if not measured_views:
            reason = ("support_not_segmented" if all(view["outcome"] == "not_segmented" for view in views)
                      else "support_top_not_measured")
            return _rejected(reason=reason, held_pose=held_pose, **common), tuple(evidence)
        # The place motion measures with the holding arm's wrist camera: prefer it.
        role = (self.primary_role if self.primary_role in measured_views
                else max(measured_views, key=lambda item: measured_views[item][0][0].area_m2))
        tops, segmentation, mask_hash, support_capture = measured_views[role]
        first = tops[0]
        common.update(held_pose=held_pose, segmentation=segmentation, measured={
            "top_z_m": first.top_z_m, "region_min_xy_m": list(first.region_min_xy_m),
            "region_max_xy_m": list(first.region_max_xy_m), "evidence": first.evidence()})
        began = time.monotonic()
        cloud = self.planner.prepare_cloud(captures=captures, planning_qpos_by_side=observation.capture_qpos_by_side,
                                           cancel_requested=cancel_requested)
        context = SimpleNamespace(
            captures=captures, cloud=cloud, size=size, tool_from_object=tool_from_object,
            current_box_world=world_from_tcp @ tool_from_object, qpos=observation.capture_qpos_by_side,
            clearance=clearance, cancel_requested=cancel_requested,
            deadline=min(float(deadline), float(self.clock()) + REAL_RELEASE_PLANNING_BUDGET_SEC),
            carried=RM65CarriedObjectBox(object_id=self.object_id, side=side, tool_from_box=tool_from_object,
                                         size_xyz_m=tuple(float(v) for v in size),
                                         evidence_ref=observation.evidence_ref))
        counts: dict[str, int] = {}
        offset = checked = 0
        fitted = out_of_budget = False
        passes = ({"yaw_offsets_rad": REAL_RELEASE_YAW_OFFSETS_RAD},
                  {"yaw_offsets_rad": REAL_RELEASE_LEAN_YAW_OFFSETS_RAD, "lean_rad": REAL_RELEASE_LEAN_RAD,
                   "lean_toward_xy_m": self.arm_base_xy_m})
        for region in tops:
            region_ref = "support-measured-top:" + canonical_content_hash({
                "support_id": support_id, "role": role, "depth": support_capture.depth_content_hash,
                "mask": mask_hash, "region": [list(region.region_min_xy_m), list(region.region_max_xy_m)],
                "plane": [list(region.plane_anchor_xyz_m), list(region.normal_xyz)]},
                domain="agenticrobot.real.support_release_query.support.v1")
            support = region.observation(support_id=support_id, reference_frame_id=frame, scene_revision=revision,
                                         measured_at=float(support_capture.acquired_at), evidence_ref=region_ref)
            recorded = False
            for orientation in passes:
                try:
                    candidates = plan_release_approach_candidates(
                        held, support, now=now, clearance_m=clearance,
                        minimum_planning_clearance_m=skill.MIN_CLEARANCE_M,
                        maximum_clearance_m=REAL_RELEASE_MAXIMUM_CLEARANCE_M,
                        lateral_margin_m=REAL_RELEASE_LATERAL_MARGIN_M,
                        maximum_measurement_age_sec=REAL_RELEASE_MAX_OBSERVATION_AGE_SEC,
                        maximum_pair_skew_sec=REAL_RELEASE_MAX_OBSERVATION_AGE_SEC,
                        include_nearest_to_held=True, **orientation)
                except NoReleaseCandidates:
                    # Nothing fits this rectangle in this pass (or the lean would tilt
                    # the object too far); stale or invalid observations still raise.
                    continue
                fitted = True
                if not recorded:
                    evidence.append(region_ref)
                    recorded = True
                for local, candidate in enumerate(candidates):
                    if checked >= REAL_RELEASE_MAX_CHECKED_CANDIDATES or float(self.clock()) >= context.deadline:
                        out_of_budget = True
                        break
                    if cancel_requested():
                        raise InterruptedError("Support release query cancelled.")
                    checked += 1
                    outcome = self._check(candidate, region=region, context=context)
                    if outcome.release is None:
                        counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
                        continue
                    index = offset + local
                    release, preplace = _pose_list(outcome.release), _pose_list(outcome.preplace)
                    evidence.append("support-release-plan:" + canonical_content_hash({
                        "candidate_index": index, "release_pose": release, "preplace_pose": preplace,
                        "plane_source": outcome.plane_source},
                        domain="agenticrobot.real.support_release_query.plan.v1"))
                    return {
                        "semantics": skill.SEMANTICS, "status": "release_pose_found",
                        "reason": "collision_free_release_path_found", "support_object_id": support_id,
                        "reference_frame_id": frame, "station_revision": revision,
                        "release_object_pose_xyz_quat_xyzw": release, skill.PREPLACE_KEY: preplace,
                        "release_clearance_m": clearance, "held_object_pose_xyz_quat_xyzw": held_pose,
                        "yaw_change_rad": float(candidate.yaw_offset_rad),
                        "release_lean_rad": float(candidate.lean_rad),
                        "support_top_z_m": float(outcome.support_top_z_m),
                        "support_region_min_xy_m": list(region.region_min_xy_m),
                        "support_region_max_xy_m": list(region.region_max_xy_m),
                        "segmentation": segmentation, "measured_support": region.evidence(),
                        "candidate_count": offset + len(candidates), "attempted_candidate_count": checked,
                        "selected_candidate_index": index,
                        "path_planning_wall_sec": float(time.monotonic() - began),
                        "candidate_rejections": _rejection_list(counts), "views": views,
                        "motion_executed": False, "execution_authorized": False}, tuple(evidence)
                offset += len(candidates)
                if out_of_budget:
                    break
            if out_of_budget:
                break
        if not fitted:
            return _rejected(reason="held_footprint_does_not_fit_support", **common), tuple(evidence)
        return _rejected(reason="query_budget_exhausted" if out_of_budget else "no_collision_free_release_path",
                         candidate_count=offset, attempted=checked, planning_wall_sec=time.monotonic() - began,
                         counts=counts, **common), tuple(evidence)

    @staticmethod
    def _plane_z(anchor, normal, xy) -> float:
        return float(anchor[2] - (normal[0] * (float(xy[0]) - anchor[0])
                                  + normal[1] * (float(xy[1]) - anchor[1])) / normal[2])

    def _check(self, candidate, *, region: RealMeasuredSupportTop, context) -> _Checked:
        """Predict the place motion's support check, set the release height, then plan both legs."""
        geometry_goal = self.geometry_from_public @ np.asarray(candidate.proposal.object_goal, dtype=float)
        corners = _box_corners(geometry_goal, context.size)
        footprint_low, footprint_high = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
        centre = (footprint_low + footprint_high) / 2.
        certified_anchor = (self.geometry_from_public @ np.r_[np.asarray(region.plane_anchor_xyz_m), 1.])[:3]
        certified_normal = self.geometry_from_public[:3, :3] @ np.asarray(region.normal_xyz)
        # The place motion fits the holding arm's wrist plane around the footprint,
        # below the box bottom of the pose it starts from (the pre-place pose).
        fitted, stats = measure_support_under(
            captures=context.captures, side=self.side, footprint_min_xy=footprint_low,
            footprint_max_xy=footprint_high,
            below_z_m=float(corners[:, 2].min()) + REAL_RELEASE_PREPLACE_HEIGHT_M + 0.01,
            exclude_box=(context.current_box_world, context.size))
        if fitted is not None:
            anchor, normal, window_low, window_high = (np.asarray(value, dtype=float) for value in fitted)
            if abs(self._plane_z(anchor, normal, centre)
                   - self._plane_z(certified_anchor, certified_normal, centre)) > REAL_SUPPORT_HEIGHT_DISAGREEMENT_M:
                return _Checked("place:support_height_disagrees")
            source = "holding_wrist_place_check"
        elif int(stats.get("support_point_count", 0)) >= REAL_OBJECT_MOTION_SUPPORT_MIN_POINTS:
            # The wrist sees the footprint but not one plane around it: the place would refuse.
            return _Checked("place:support_not_measured")
        else:
            # Not seen yet; the place measures again from the pre-place pose.
            anchor, normal = certified_anchor, certified_normal
            window_low = footprint_low - REAL_OBJECT_MOTION_SUPPORT_MARGIN_M
            window_high = footprint_high + REAL_OBJECT_MOTION_SUPPORT_MARGIN_M
            source = "certified_support_top"
        gap = float(np.min((corners - anchor) @ normal))
        geometry_goal = geometry_goal.copy()
        geometry_goal[2, 3] += (context.clearance - gap) / float(normal[2])
        goal = self.public_from_geometry @ geometry_goal
        preplace = goal.copy()
        preplace[2, 3] += REAL_RELEASE_PREPLACE_HEIGHT_M
        bounds = self.object_workspace_bounds
        if any(not bounds[2 * i] <= pose[i, 3] <= bounds[2 * i + 1] for pose in (goal, preplace) for i in range(3)):
            return _Checked("workspace:goal_outside_object_workspace")
        object_from_tool = np.linalg.inv(context.tool_from_object)
        preplace_tool = self.geometry_from_public @ preplace @ object_from_tool
        release_tool = self.geometry_from_public @ goal @ object_from_tool
        support = RM65CarriedObjectSupport(
            support_id="measured_support_under_target", plane_anchor_xyz_m=tuple(float(v) for v in anchor),
            normal_xyz=tuple(float(v) for v in normal), region_min_xy_m=tuple(float(v) for v in window_low),
            region_max_xy_m=tuple(float(v) for v in window_high), evidence_ref=context.carried.evidence_ref,
            height_tolerance_m=REAL_OBJECT_MOTION_SUPPORT_HEIGHT_TOLERANCE_M)
        remaining = context.deadline - float(self.clock())
        if remaining <= 0.:
            return _Checked("carry:planning_time_budget_exhausted")
        end, reason = self.planner.plan_leg(
            cloud=context.cloud, qpos_by_side=context.qpos, side=self.side, target_tool_tcp_world=preplace_tool,
            carried_object=context.carried, pose_goal_policy="within_arrival_tolerance",
            time_budget_sec=min(REAL_RELEASE_LEG_BUDGET_SEC, remaining), cancel_requested=context.cancel_requested)
        if end is None:
            return _Checked("carry:" + reason)
        remaining = context.deadline - float(self.clock())
        if remaining <= 0.:
            return _Checked("place:planning_time_budget_exhausted")
        end, reason = self.planner.plan_leg(
            cloud=context.cloud, qpos_by_side=end, side=self.side, target_tool_tcp_world=release_tool,
            carried_object=context.carried, carried_object_support=support, pose_goal_policy="preserve_target",
            time_budget_sec=min(REAL_RELEASE_LEG_BUDGET_SEC, remaining), cancel_requested=context.cancel_requested)
        if end is None:
            return _Checked("place:" + reason)
        top = self.public_from_geometry @ np.r_[centre, self._plane_z(anchor, normal, centre), 1.]
        return _Checked("path_found", release=goal, preplace=preplace, support_top_z_m=float(top[2]),
                        plane_source=source)


def build_real_support_release_executor(*, registration: RealSupportReleaseRegistration, query_port,
                                        current_station_revision, clock, max_retained_operations):
    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        revision = request.expected_station_revision
        result, evidence = query_port.query(
            request.parameters, expected_station_revision=revision,
            cancel_requested=lambda: executor.cancel_requested(operation.operation_id),
            deadline=operation.deadline_at - _RESULT_RESERVE_SEC)
        if current_station_revision() != revision:
            raise RuntimeError("Station revision changed during the release query; retry.")
        return PassiveReadResult(result, ("support-release-pose:" + operation.operation_id, *evidence),
                                 reason_code="support_release_pose_assessed")

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.real-support-release-pose.v2", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)
    return executor


__all__ = ["REAL_RELEASE_PREPLACE_HEIGHT_M", "REAL_SUPPORT_RELEASE_MAX_OPERATION_MS", "RealMeasuredSupportTop",
           "RealSupportReleasePlanner", "RealSupportReleaseQuery", "RealSupportReleaseRegistration",
           "build_real_support_release_executor", "build_real_support_release_registration",
           "measure_real_support_tops"]
