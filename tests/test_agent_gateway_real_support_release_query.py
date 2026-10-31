"""RealMan ``perception.query_support_release_pose@2``: contract, support measurement and query port.

No device, model or planner is started. Downward synthetic RGB-D views of a
tray on the table stand in for the stationary cut, a scripted segmenter for
SAM3 and a recording stub for the observed safe-move planner. The held-object
port and the place motion's support check are the production code.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.back_projection import LiveAlignedDepthCapture
from agent_gateway.profiles.real import support_release_query as module
from agent_gateway.profiles.real.held_object import RealHeldObjectObserver
from agent_gateway.profiles.real.object_motion import RealHeldObjectPort
from agent_gateway.profiles.segmentation import PreparedSegmentation
from agent_gateway.skills.perception import support_release_query as skill_v1
from agent_gateway.skills.perception import support_release_query_v2 as skill
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65CarriedObjectSupport,
    RM65GraspTrajectoryPlanningError,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

W, H, F = 160, 120, 96.0
ROLES = ("main", "wrist_left", "wrist_right")
FRAME = "canonical_station"
SUPPORT, PROMPT = "green_tray", "green tray"
REVISION = 3
# Geometry (planner) frame: table top z = 0, a 16 cm tray top at z = 0.05.
TRAY_MIN, TRAY_MAX, TRAY_TOP = (0.02, -0.28), (0.18, -0.12), 0.05
CUBE = 0.04


def _pose(rotation=None, translation=(0., 0., 0.)) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.eye(3) if rotation is None else rotation
    pose[:3, 3] = translation
    return pose


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


# Public (Agent) frame: a quarter turn of the geometry frame plus a shift.
PUBLIC_FROM_GEOMETRY = _pose(_rot_z(math.pi / 2.), (0.3, -0.1, -0.02))
GEOMETRY_FROM_PUBLIC = np.linalg.inv(PUBLIC_FROM_GEOMETRY)
DOWN = np.diag([1., -1., -1.])          # camera / tool z points down
CAMERA_POSES = {"main": _pose(DOWN, (0.0, -0.2, 0.95)), "wrist_left": _pose(DOWN, (-0.3, 0.3, 0.4)),
                "wrist_right": _pose(DOWN, (0.10, -0.20, 0.36))}
# The held cube's centre is 5 cm below the TCP along the tool axis.
TOOL_FROM_BOX = _pose(translation=(0., 0., 0.05))
WORLD_FROM_TCP = _pose(DOWN, (0.0, -0.45, 0.21))


def _render(camera_pose, *, tray=(TRAY_MIN, TRAY_MAX, TRAY_TOP), tilt_rad=0.):
    """Depth of a downward pinhole camera over the table and one tray top; and the tray-top pixels."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = (u - W / 2.) / F, (v - H / 2.) / F
    cz = camera_pose[2, 3]
    low, high, top = tray

    def hit(height):
        t = cz - height
        return t, camera_pose[0, 3] + t * dx, camera_pose[1, 3] - t * dy

    depth, _, _ = hit(0.)
    t, x, y = hit(top)
    # A tilted top rises along x by tan(tilt) (first order is enough here).
    t = t - np.tan(tilt_rad) * (x - low[0])
    on_tray = (x >= low[0]) & (x <= high[0]) & (y >= low[1]) & (y <= high[1])
    depth = np.where(on_tray, t, depth)
    return depth.astype(np.float32), on_tray


def _capture(role, index, depth, pose):
    digest = hashlib.sha256(role.encode() + depth.tobytes()).hexdigest()
    return SimpleNamespace(
        role=role, source_id="realsense:" + role, acquired_at=100. + 0.01 * index, station_revision=REVISION,
        capture_group_id="real-aligned-rgbd:" + digest[:16], depth_content_hash=digest,
        intrinsics={"fx": F, "fy": F, "cx": W / 2., "cy": H / 2., "width": W, "height": H}, width=W, height=H,
        source_from_camera=tuple(pose.reshape(-1)), depth_m=depth,
        rgb=np.zeros((H, W, 3), dtype=np.uint8), rgb_content_hash=digest,
        capture_qpos_by_side={"left": np.zeros(6), "right": np.full(6, 0.1)})


def _scene(poses=None, **render):
    captures, masks = {}, {}
    for index, role in enumerate(ROLES):
        pose = (poses or CAMERA_POSES)[role]
        depth, on_tray = _render(pose, **render)
        captures[role], masks[role] = _capture(role, index, depth, pose), on_tray
    return captures, masks


def _prepared(mask):
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(mask, dtype=bool), allow_pickle=False)
    rows, cols = np.nonzero(mask)
    return PreparedSegmentation(execution_result={
        "image_width": W, "image_height": H, "mask_pixel_count": int(mask.sum()),
        "mask_bounds_xyxy": [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1],
        "score": 0.9, "mask_media_content_hash": hashlib.sha256(buffer.getvalue()).hexdigest()},
        mask_npy=buffer.getvalue(), overlay_jpeg=b"")


class _ScenePort:
    def __init__(self, captures):
        self.captures = captures
        self.calls = 0

    def capture_views(self, *, roles, expected_station_revision, cancel_requested):
        assert expected_station_revision == REVISION
        self.calls += 1
        return tuple(self.captures[role] for role in roles)

    @staticmethod
    def segmentation_input(capture):
        return capture


class _Segmenter:
    def __init__(self, masks, roles):
        self.masks, self.roles, self.calls = masks, tuple(roles), []

    def try_segment_capture(self, *, capture, parameters):
        self.calls.append((capture.role, parameters["language_prompt"]))
        mask = self.masks.get(capture.role)
        if parameters["language_prompt"] != PROMPT or capture.role not in self.roles or mask is None or not mask.any():
            return None
        return _prepared(mask)


class _Planner:
    """Records every plan request; the scripted outcomes come first (None plans)."""

    def __init__(self, refusals=()):
        self.refusals, self.calls = list(refusals), []

    def plan_with_observed_cloud(self, cloud, **kwargs):
        assert cloud == "cloud"
        self.calls.append(kwargs)
        reason = self.refusals.pop(0) if self.refusals else None
        if reason is not None:
            raise RM65GraspTrajectoryPlanningError(reason)
        end = np.full(6, 0.01 * len(self.calls))
        return SimpleNamespace(qpos_by_side={"left": [np.zeros(6)], "right": [np.zeros(6), end]})


def _observation(*, verified=True, size=CUBE, attempt="attempt-1", world_from_tcp=WORLD_FROM_TCP):
    return SimpleNamespace(
        holding_verified=verified, evidence_ref="real-held-object:" + "c" * 64, world_from_tcp=world_from_tcp,
        capture_qpos_by_side={"left": (0.,) * 6, "right": (0.1,) * 6},
        measurement=SimpleNamespace(tool_from_box=TOOL_FROM_BOX, size_xyz_m=(size, size, size)),
        gripper={"measured_close": 0.6, "last_command_attempt_ref": attempt})


def _query(monkeypatch, *, planner=None, segmented=("main", "wrist_right"), retained=True, observation=None,
           size=CUBE, scene=None, world_from_tcp=WORLD_FROM_TCP):
    monkeypatch.setattr(module, "obstacle_cloud_from_captures", lambda **kwargs: "cloud")
    captures, masks = scene or _scene()
    port = _ScenePort(captures)
    segmenter = _Segmenter(masks, segmented)
    observer = RealHeldObjectObserver(scene_port=port, segmentation_provider=segmenter,
                                      tool_entity_by_side={"right": object()}, roles=ROLES)
    observed = observation or _observation(size=size, world_from_tcp=world_from_tcp)
    observer.observe = lambda **kwargs: observed
    held = RealHeldObjectPort(observer=observer, public_from_geometry=PUBLIC_FROM_GEOMETRY,
                              current_station_revision=lambda: REVISION, clock=lambda: 100.)
    if retained:
        half = np.full(3, size / 2.)
        held.remember("right", TOOL_FROM_BOX[:3, 3] - half, TOOL_FROM_BOX[:3, 3] + half,
                      _observation(size=size, world_from_tcp=world_from_tcp))
    stub = planner or _Planner()
    planner_port = module.RealSupportReleasePlanner(
        planner_factory=lambda: stub, self_filter=object(), observation_config=None, geometry_frame_id="world",
        start_clearance="ignore-start-contacts-v1", speed_scale=0.2, profile_identity={"test": True})
    registration = module.RealSupportReleaseRegistration(
        skill=None, entry=None, payload_validator=None, profile_hash="profile", roles=ROLES,
        prompt_by_support_id={SUPPORT: PROMPT}, support_object_ids=(SUPPORT,), reference_frame_id=FRAME,
        side="right")
    base = PUBLIC_FROM_GEOMETRY @ _pose(translation=(-0.05, 0.25, 0.))
    query = module.RealSupportReleaseQuery(
        registration=registration, scene_port=port, segmentation_provider=segmenter, held_port=held,
        planner=planner_port, object_id="right_held_object", public_from_geometry=PUBLIC_FROM_GEOMETRY,
        object_workspace_bounds=(-1., 1., -1., 1., -0.05, 1.), arm_base_xy_m=(base[0, 3], base[1, 3]),
        maximum_depth_m=1.5, clock=time.monotonic)
    return SimpleNamespace(query=query, planner=stub, segmenter=segmenter, held=held, port=port)


def _run(parts, **parameters):
    return parts.query.query({"support_object_id": SUPPORT, "reference_frame_id": FRAME, **parameters},
                             expected_station_revision=REVISION, cancel_requested=lambda: False,
                             deadline=time.monotonic() + 60.)


def _validate(result):
    skill.validate_result(result, support_object_ids=(SUPPORT,), reference_frame_id=FRAME)


def _geometry(pose_xyz_quat):
    from robot_motion.relative_action import xyzquat_to_pose_matrix

    return GEOMETRY_FROM_PUBLIC @ xyzquat_to_pose_matrix(pose_xyz_quat)


def _depth_capture(role, depth):
    return LiveAlignedDepthCapture(
        source_id="realsense:" + role, role=role, captured_at=1.0, station_revision=0, capture_group_id="group",
        depth_content_hash="d" * 64, intrinsics={"fx": F, "fy": F, "cx": W / 2., "cy": H / 2., "width": W, "height": H},
        reference_from_camera=CAMERA_POSES[role], depth_m=depth)


def _dilated(mask, radius):
    grown = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            grown |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return grown


# Contract ---------------------------------------------------------------------


def test_v2_contract_is_bounded_and_leaves_v1_unchanged():
    definition = skill.definition()
    assert (definition.skill_id, definition.skill_version) == (skill_v1.SKILL_ID, "2")
    assert len(definition.summary) <= 2000  # RPent's Skill summary bound
    assert "phase=carry" in definition.summary and "phase=place" in definition.summary
    entry = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash="a" * 64,
                         support_object_ids=(SUPPORT,), reference_frame_id=FRAME)
    assert set(entry.execution_result_schema["properties"]) == skill.RESULT_KEYS
    clearance = entry.parameter_schema["properties"]["clearance_m"]
    assert (clearance["minimum"], clearance["maximum"]) == (0.005, 0.015)
    v1 = skill_v1.schema(definition_hash=skill_v1.definition().semantic_content_hash, profile_hash="a" * 64,
                         support_object_ids=(SUPPORT,), reference_frame_id=FRAME)
    assert skill.PREPLACE_KEY not in v1.execution_result_schema["properties"]
    assert v1.parameter_schema["properties"]["clearance_m"]["maximum"] == skill_v1.MAX_CLEARANCE_M
    assert v1.parameter_schema_hash != entry.parameter_schema_hash
    for bad in (0.004, 0.02):
        with pytest.raises(ValueError):
            skill.validate_parameters({"support_object_id": SUPPORT, "reference_frame_id": FRAME, "clearance_m": bad},
                                      support_object_ids=(SUPPORT,), reference_frame_id=FRAME)
    assert skill.requested_clearance_m({}) == 0.01


# Support measurement ------------------------------------------------------------


def test_real_support_top_is_certified_inside_the_observed_top():
    depth, on_tray = _render(CAMERA_POSES["wrist_right"])
    noisy = (depth + np.random.default_rng(0).normal(0., 0.0015, depth.shape)).astype(np.float32)
    # The mask also bleeds two pixels onto the table around the tray.
    tops = module.measure_real_support_tops(_depth_capture("wrist_right", noisy), _dilated(on_tray, 2),
                                            maximum_depth_m=1.5)
    top = tops[0]
    for axis in (0, 1):
        assert TRAY_MIN[axis] <= top.region_min_xy_m[axis] < top.region_max_xy_m[axis] <= TRAY_MAX[axis]
    assert top.area_m2 > 0.6 * 0.16 ** 2
    centre = [(a + b) / 2. for a, b in zip(top.region_min_xy_m, top.region_max_xy_m)]
    assert top.height_at(centre) == pytest.approx(TRAY_TOP, abs=0.002)
    assert TRAY_TOP < top.top_z_m <= TRAY_TOP + module.REAL_SUPPORT_PLANE_RESIDUAL_M + 0.002
    observation = top.observation(support_id=SUPPORT, reference_frame_id="world", scene_revision=0,
                                  measured_at=1.0, evidence_ref="support")
    assert observation.normal_reference_xyz == (0., 0., 1.) and observation.top_z_m == top.top_z_m


def test_real_support_top_refuses_a_steep_top_and_a_tiny_mask():
    depth, on_tray = _render(CAMERA_POSES["wrist_right"], tilt_rad=math.radians(12.))
    with pytest.raises(ValueError, match="near-horizontal"):
        module.measure_real_support_tops(_depth_capture("wrist_right", depth), on_tray, maximum_depth_m=1.5)
    depth, on_tray = _render(CAMERA_POSES["wrist_right"])
    rows, cols = np.nonzero(on_tray)
    tiny = np.zeros_like(on_tray)
    tiny[rows[:40], cols[:40]] = True
    with pytest.raises(ValueError, match="insufficient"):
        module.measure_real_support_tops(_depth_capture("wrist_right", depth), tiny, maximum_depth_m=1.5)


def test_real_support_top_plane_is_the_support_not_an_object_standing_on_it():
    # A 6 cm block on a quarter of the tray and inside its mask (as a mask bleeding onto it):
    # the block top is more than the highest fifth of the masked points, but the most
    # populated height band is still the tray top.
    camera = CAMERA_POSES["wrist_right"]
    depth, on_tray = _render(camera)
    v, u = np.mgrid[0:H, 0:W].astype(float)
    distance = camera[2, 3] - (TRAY_TOP + 0.06)
    x = camera[0, 3] + distance * (u - W / 2.) / F
    y = camera[1, 3] - distance * (v - H / 2.) / F
    block_low, block_high = (0.10, -0.28), (0.18, -0.20)
    on_block = (x >= block_low[0]) & (x <= block_high[0]) & (y >= block_low[1]) & (y <= block_high[1])
    depth = np.where(on_block, distance, depth)
    depth = (depth + np.random.default_rng(1).normal(0., 0.0015, depth.shape)).astype(np.float32)
    mask = on_tray | on_block
    assert on_block.sum() > 0.2 * mask.sum()
    top = module.measure_real_support_tops(_depth_capture("wrist_right", depth), mask, maximum_depth_m=1.5)[0]
    centre = [(a + b) / 2. for a, b in zip(top.region_min_xy_m, top.region_max_xy_m)]
    assert top.height_at(centre) == pytest.approx(TRAY_TOP, abs=0.002)
    assert top.tilt_deg < 1.
    # The certified rectangle is on the tray and clear of the block.
    for axis in (0, 1):
        assert TRAY_MIN[axis] <= top.region_min_xy_m[axis] < top.region_max_xy_m[axis] <= TRAY_MAX[axis]
    assert (top.region_max_xy_m[0] <= block_low[0] or top.region_min_xy_m[0] >= block_high[0]
            or top.region_max_xy_m[1] <= block_low[1] or top.region_min_xy_m[1] >= block_high[1])


# Held object ---------------------------------------------------------------------


def test_retained_grip_box_reads_without_storing_or_discarding():
    observer = RealHeldObjectObserver(scene_port=_ScenePort({}), segmentation_provider=_Segmenter({}, ()),
                                      tool_entity_by_side={"right": object()}, roles=ROLES)
    port = RealHeldObjectPort(observer=observer, public_from_geometry=np.eye(4),
                              current_station_revision=lambda: 0, clock=lambda: 0.)
    fresh = _observation()
    assert port.retained_grip_box("right", fresh) is None  # no lift has retained a box yet
    # The first lift completed the box down to the support the cube stood on.
    low, high = np.array([-0.02, -0.02, 0.03]), np.array([0.02, 0.02, 0.09])
    port.remember("right", low, high, fresh)
    before = dict(port._retained)
    read_low, read_high = port.retained_grip_box("right", fresh)
    fused_low, fused_high, fused = port.fused_box("right", fresh)
    assert fused is True
    np.testing.assert_allclose(read_low, fused_low)
    np.testing.assert_allclose(read_high, fused_high)
    assert port.retained_grip_box("right", _observation(attempt="attempt-2")) is None  # a new gripper command
    assert port.retained_grip_box("right", _observation(verified=False)) is None
    assert port._retained == before  # reads neither store nor discard


# Query ------------------------------------------------------------------------------


def test_query_returns_checked_preplace_and_release_poses_from_the_holding_wrist(monkeypatch):
    parts = _query(monkeypatch)
    result, evidence = _run(parts)
    assert result["status"] == "release_pose_found", result
    _validate(result)
    assert result["segmentation"]["role"] == "wrist_right"  # the place motion measures with it
    assert [view["role"] for view in result["views"]] == ["wrist_right", "main", "wrist_left"]
    assert result["views"][0]["outcome"] == "measured" and result["views"][2]["outcome"] == "not_segmented"
    assert (result["yaw_change_rad"], result["release_lean_rad"]) == (0., 0.)
    release = _geometry(result["release_object_pose_xyz_quat_xyzw"])
    preplace = _geometry(result[skill.PREPLACE_KEY])
    # Box bottom 1 cm above the tray top the holding wrist measured, footprint on the tray.
    assert release[2, 3] - CUBE / 2. == pytest.approx(TRAY_TOP + 0.01, abs=1e-3)
    for axis in (0, 1):
        assert TRAY_MIN[axis] < release[axis, 3] - CUBE / 2. and release[axis, 3] + CUBE / 2. < TRAY_MAX[axis]
    public_release = np.asarray(result["release_object_pose_xyz_quat_xyzw"])
    public_preplace = np.asarray(result[skill.PREPLACE_KEY])
    np.testing.assert_allclose(public_preplace - public_release, [0., 0., 0.05, 0., 0., 0., 0.], atol=1e-9)
    top = PUBLIC_FROM_GEOMETRY @ np.array([release[0, 3], release[1, 3], TRAY_TOP, 1.])
    assert result["support_top_z_m"] == pytest.approx(top[2], abs=1e-3)
    assert result["held_object_pose_xyz_quat_xyzw"] is not None
    carry, place = parts.planner.calls
    assert carry["pose_goal_policy"] == "within_arrival_tolerance" and "carried_object_support" not in carry
    assert place["pose_goal_policy"] == "preserve_target"
    support = place["carried_object_support"]
    assert isinstance(support, RM65CarriedObjectSupport)
    assert support.plane_anchor_xyz_m[2] == pytest.approx(TRAY_TOP, abs=1e-3)
    # The place leg starts where the carry leg ended; both carry the retained grip box.
    np.testing.assert_allclose(place["qpos_by_side"]["right"], np.full(6, 0.01))
    for call in (carry, place):
        assert call["carried_object"].size_xyz_m == pytest.approx((CUBE, CUBE, CUBE))
        np.testing.assert_allclose(call["carried_object"].tool_from_box, TOOL_FROM_BOX)
        assert call["require_grasp_approach"] is False and call["side"] == "right"
    np.testing.assert_allclose(carry["target_tool_tcp_world"], preplace @ np.linalg.inv(TOOL_FROM_BOX), atol=1e-9)
    np.testing.assert_allclose(place["target_tool_tcp_world"], release @ np.linalg.inv(TOOL_FROM_BOX), atol=1e-9)
    assert parts.port.calls == 1  # one stationary cut for the held object, the support and the cloud
    assert any(ref.startswith("support-release-plan:") for ref in evidence)
    # The pre-place pose must stay straight above the release pose with its orientation.
    for broken in (public_preplace + [0.001, 0, 0, 0, 0, 0, 0], public_preplace + [0, 0, 0.2, 0, 0, 0, 0],
                   public_release, np.r_[public_preplace[:3], 0., 0., 0., 1.]):
        with pytest.raises(ValueError):
            _validate({**result, skill.PREPLACE_KEY: [float(v) for v in broken]})


def test_query_requires_the_grip_an_object_motion_lifted(monkeypatch):
    parts = _query(monkeypatch, retained=False)
    result, _ = _run(parts)
    assert (result["status"], result["reason"]) == ("no_release_pose", "held_object_not_verified_lifted")
    _validate(result)
    assert parts.segmenter.calls == [] and parts.planner.calls == []
    result, _ = _run(_query(monkeypatch, observation=_observation(verified=False)))
    assert result["reason"] == "held_object_not_verified_lifted"
    with pytest.raises(ValueError):
        _validate({**result, skill.PREPLACE_KEY: [0., 0., 0.1, 0., 0., 0., 1.]})


def test_query_reports_an_unseen_or_unmeasured_support(monkeypatch):
    result, _ = _run(_query(monkeypatch, segmented=()))
    assert result["reason"] == "support_not_segmented"
    assert {view["outcome"] for view in result["views"]} == {"not_segmented"}
    _validate(result)
    captures, masks = _scene()
    rows, cols = np.nonzero(masks["wrist_right"])
    tiny = np.zeros_like(masks["wrist_right"])
    tiny[rows[:30], cols[:30]] = True
    result, _ = _run(_query(monkeypatch, segmented=("wrist_right",), scene=(captures, {"wrist_right": tiny})))
    assert result["reason"] == "support_top_not_measured"
    _validate(result)


def test_query_refuses_a_held_footprint_larger_than_the_certified_top(monkeypatch):
    parts = _query(monkeypatch, size=0.2)
    result, _ = _run(parts)
    assert result["reason"] == "held_footprint_does_not_fit_support"
    _validate(result)
    assert parts.planner.calls == []


def test_query_counts_refusals_by_leg_and_returns_a_later_candidate(monkeypatch):
    planner = _Planner(["ik_search_exhausted", None, "endpoint_collision_blocked", "path_search_exhausted"])
    result, _ = _run(_query(monkeypatch, planner=planner))
    assert result["status"] == "release_pose_found"
    _validate(result)
    counts = {item["reason_code"]: item["count"] for item in result["candidate_rejections"]}
    # Centre: carry refused; nearest position: place refused; first corner: carry refused.
    assert counts == {"carry:ik_search_exhausted": 1, "place:endpoint_collision_blocked": 1,
                      "carry:path_search_exhausted": 1}
    assert result["selected_candidate_index"] == result["attempted_candidate_count"] - 1 == 3
    assert result["yaw_change_rad"] == 0.
    assert len(planner.calls) == 6


# Held right above the tray (the pre-place pose of its centre): the place check is predicted.
ABOVE_TRAY_TCP = _pose(DOWN, ((TRAY_MIN[0] + TRAY_MAX[0]) / 2., (TRAY_MIN[1] + TRAY_MAX[1]) / 2.,
                              TRAY_TOP + 0.01 + CUBE + module.REAL_RELEASE_PREPLACE_HEIGHT_M + 0.05 - CUBE / 2.))


def test_query_from_the_preplace_pose_takes_the_release_height_from_the_place_check(monkeypatch):
    parts = _query(monkeypatch, world_from_tcp=ABOVE_TRAY_TCP)
    result, _ = _run(parts)
    assert result["status"] == "release_pose_found", result
    _validate(result)
    release = _geometry(result["release_object_pose_xyz_quat_xyzw"])
    assert release[2, 3] - CUBE / 2. == pytest.approx(TRAY_TOP + 0.01, abs=1e-3)
    _carry, place = parts.planner.calls[-2:]
    support = place["carried_object_support"]
    # The place check's window: footprint +- 4 cm, the plane of the holding wrist.
    window = np.asarray(support.region_max_xy_m) - np.asarray(support.region_min_xy_m)
    np.testing.assert_allclose(window, CUBE + 2 * 0.04, atol=1e-6)
    assert support.plane_anchor_xyz_m[2] == pytest.approx(TRAY_TOP, abs=1e-3)


def test_query_from_above_predicts_the_place_refusal_on_a_narrow_support(monkeypatch):
    # A 9 cm tray: the place check's window (footprint + 4 cm each side) reaches the table.
    monkeypatch.setattr(module, "REAL_RELEASE_MAX_CHECKED_CANDIDATES", 1000)
    narrow = ((0.055, -0.245), (0.145, -0.155), TRAY_TOP)
    parts = _query(monkeypatch, world_from_tcp=ABOVE_TRAY_TCP, scene=_scene(tray=narrow))
    result, _ = _run(parts)
    assert result["reason"] == "no_collision_free_release_path"
    _validate(result)
    counts = {item["reason_code"]: item["count"] for item in result["candidate_rejections"]}
    assert counts == {"place:support_not_measured": result["attempted_candidate_count"]}
    assert parts.planner.calls == []


def test_query_without_a_planned_candidate_reports_the_budget_or_no_path(monkeypatch):
    result, _ = _run(_query(monkeypatch, planner=_Planner(["ik_search_exhausted"] * 500)))
    assert result["reason"] == "query_budget_exhausted"  # candidates left untried at the check cap
    assert result["attempted_candidate_count"] == module.REAL_RELEASE_MAX_CHECKED_CANDIDATES
    _validate(result)
    monkeypatch.setattr(module, "REAL_RELEASE_MAX_CHECKED_CANDIDATES", 1000)
    result, _ = _run(_query(monkeypatch, planner=_Planner(["ik_search_exhausted"] * 500)))
    assert result["reason"] == "no_collision_free_release_path"
    assert result["attempted_candidate_count"] == result["candidate_count"] == 30  # 4 yaws x 6, then 6 leaned
    _validate(result)


def test_query_checks_the_held_yaw_on_every_rectangle_before_turning(monkeypatch):
    # Two certified rectangles (the tray's two halves); every carry into the first half is out
    # of reach. The held yaw on the second half comes before the turned yaws of the first:
    # rectangle by rectangle the first half's 4 yaws alone would use the 24-check budget.
    original = module.measure_real_support_tops
    middle_by_role = {}

    def two_halves(capture, mask, **kwargs):
        top = original(capture, mask, **kwargs)[0]
        middle = (top.region_min_xy_m[0] + top.region_max_xy_m[0]) / 2.
        middle_by_role[capture.role] = middle
        return (replace(top, region_max_xy_m=(middle, top.region_max_xy_m[1])),
                replace(top, region_min_xy_m=(middle, top.region_min_xy_m[1])))

    monkeypatch.setattr(module, "measure_real_support_tops", two_halves)

    class FirstHalfOutOfReach(_Planner):
        def plan_with_observed_cloud(self, cloud, **kwargs):
            box = PUBLIC_FROM_GEOMETRY @ kwargs["target_tool_tcp_world"] @ TOOL_FROM_BOX
            if (kwargs["pose_goal_policy"] == "within_arrival_tolerance"
                    and box[0, 3] < middle_by_role["wrist_right"]):
                self.calls.append(kwargs)
                raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")
            return super().plan_with_observed_cloud(cloud, **kwargs)

    planner = FirstHalfOutOfReach()
    result, _ = _run(_query(monkeypatch, planner=planner))
    assert result["status"] == "release_pose_found", result
    _validate(result)
    assert result["segmentation"]["role"] == "wrist_right"
    assert result["yaw_change_rad"] == 0. and result["release_lean_rad"] == 0.
    assert result["support_region_min_xy_m"][0] == pytest.approx(middle_by_role["wrist_right"])
    refused = result["attempted_candidate_count"] - 1
    assert {item["reason_code"]: item["count"] for item in result["candidate_rejections"]} == {
        "carry:ik_search_exhausted": refused}
    assert 1 <= refused <= 6  # only the first half's held-yaw positions
    assert result["candidate_count"] > module.REAL_RELEASE_MAX_CHECKED_CANDIDATES


def test_query_falls_back_to_the_certified_plane_when_the_wrist_does_not_see_the_target(monkeypatch):
    poses = {**CAMERA_POSES, "wrist_right": _pose(DOWN, (-0.35, -0.65, 0.36))}  # looking at bare table
    parts = _query(monkeypatch, segmented=("main",), scene=_scene(poses))
    result, _ = _run(parts)
    assert result["status"] == "release_pose_found", result
    _validate(result)
    assert result["segmentation"]["role"] == "main"
    assert result["views"][0] == {"role": "wrist_right", "outcome": "not_segmented", "score": None,
                                  "certified_area_m2": None}
    release = _geometry(result["release_object_pose_xyz_quat_xyzw"])
    assert release[2, 3] - CUBE / 2. == pytest.approx(TRAY_TOP + 0.01, abs=2e-3)
    _carry, place = parts.planner.calls
    assert place["carried_object_support"].plane_anchor_xyz_m[2] == pytest.approx(TRAY_TOP, abs=2e-3)


# Composition and CLI ----------------------------------------------------------------

QUERY_REF = ("perception.query_support_release_pose", "2")
SIX_F = "dual_arm_hand_proxy_dmtac_grasp_rm65_6f"


@pytest.fixture
def planners(monkeypatch):
    from agent_gateway.profiles.real import rm65_composition as composition_module

    created = []

    class Planner:
        def __call__(self, **kwargs):
            return None

        def plan_with_observed_cloud(self, cloud, **kwargs):
            return cloud

    def planner(**kwargs):
        created.append(kwargs)
        return Planner()

    monkeypatch.setattr(composition_module, "RM65GraspJointTrajectoryPlanner", planner)
    return created


def _lower(monkeypatch, collision_scene_profile):
    from test_agent_gateway_real_rm65_composition import (
        REPO_ROOT,
        _Backend,
        _ForceGuard,
        _TargetBuilder,
    )

    from teleop_runtime.skill_deploy import rm65_profile as profile_module
    from teleop_runtime.skill_deploy.rm65_lower_assembly import (
        assemble_exclusive_rm65_skill_lower,
    )
    from teleop_runtime.skill_deploy.rm65_profile import (
        ExclusiveRM65SkillDeploymentProfile,
        resolve_exclusive_rm65_skill_deployment_profile,
    )

    monkeypatch.setattr(profile_module, "load_assignment", lambda path: {
        "assignments": {"left": {"arm_ip": "192.0.2.10"}, "right": {"arm_ip": "192.0.2.11"}}})
    calibrations = REPO_ROOT / "hardware/rm65/utils/configs/calibrations"
    profile = resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=REPO_ROOT / "README.md",
        tool_profile_path_by_side={side: calibrations / f"rm65_{side}_DMTAC_M.json" for side in ("left", "right")},
        envelope_profile_path=REPO_ROOT / "teleop_runtime/configs/skill_deploy/agent_primitive_right_relative_tight_v1.json",
        collision_scene_profile=collision_scene_profile))
    guard = _ForceGuard()
    lower = assemble_exclusive_rm65_skill_lower(profile=profile, backend_factory=_Backend,
        target_builder_factory=lambda config: _TargetBuilder(), force_guard_factory=lambda backends: guard)
    return lower, guard


class _Hand:
    config = SimpleNamespace(side="right", gripper_port="/dev/fake", gripper_id=1)

    def __getattr__(self, name):
        if name in {"start", "close", "send_fixed_joint_pos", "read_gripper_state_now", "get_state",
                    "hold_gripper_motion", "is_ready", "readiness_report", "force_reconnect_gripper"}:
            def forbidden(*args, **kwargs):
                raise AssertionError("composition must not access hand hardware: " + name)

            return forbidden
        raise AttributeError(name)


def _compose(lower, tmp_path, **kwargs):
    from test_agent_gateway_real_rm65_composition import REPO_ROOT, _Localizer

    from agent_gateway.profiles.real.gateway_cli import (
        DEFAULT_RM65_CAMERA_MOUNT_URDF,
        DEFAULT_RM65_CAMERA_REGISTRY_V4,
    )
    from agent_gateway.profiles.real.motion_observed_scene import (
        SafeMotionObservationConfig,
    )
    from agent_gateway.profiles.real.one_shot_authorization import (
        REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
    )
    from agent_gateway.profiles.real.rm65_composition import (
        assemble_exclusive_rm65_agent_primitive_composition,
    )

    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    base = {"lower": lower, "socket_path": tmp_path / "release.sock", "authorized_side": "right",
            "authorized_action": None, "right_gripper_backend": _Hand(),
            "authorized_actions": ("move_tcp_to_pose_safe", "move_grasped_object_to_pose", "set_gripper_state"),
            "confirmation_token": REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN, "client_id": "rpent:test", "client_uid": 1000,
            "runtime_session_id": "release:test", "control_domain_id": "release:domain",
            "embodiment_instance_id": "release:arms",
            "camera_config_path": REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json",
            "camera_registry_path": DEFAULT_RM65_CAMERA_REGISTRY_V4, "camera_mount_urdf_path": DEFAULT_RM65_CAMERA_MOUNT_URDF,
            "artifact_root": tmp_path / "artifacts", "enable_back_projection": True, "object_localizer": _Localizer(),
            "safe_motion_observation_config": SafeMotionObservationConfig(), "enable_tool_object_relations": True}
    return assemble_exclusive_rm65_agent_primitive_composition(**{**base, **kwargs})


def _require_six_f_assets():
    from test_agent_gateway_real_rm65_composition import REPO_ROOT

    model = REPO_ROOT / "assets/rm65_inspire_tac/rm65_dual_arm_hand_proxy_dmtac_strict_rm65_6f.urdf"
    if not model.is_file():
        pytest.skip("RM65-6F planning model not generated (scripts/assets/make_rm65_6f_station_assets.py)")


def test_real_composition_publishes_the_query_only_when_named_supports_are_bound(monkeypatch, tmp_path, planners):
    _require_six_f_assets()
    lower, guard = _lower(monkeypatch, SIX_F)
    composition = _compose(lower, tmp_path / "off")
    session = composition.deployment.gateway_session
    try:
        published = {(e.route.skill_id, e.route.skill_version) for e in session.contract_bundle.route_manifest.skill_routes}
        assert ("motion.move_grasped_object_to_pose", "1") in published and QUERY_REF not in published
        assert session.support_release_executor is None
    finally:
        session.close()
    lower, guard = _lower(monkeypatch, SIX_F)
    composition = _compose(lower, tmp_path / "on", support_release_prompt_by_id={SUPPORT: PROMPT})
    session = composition.deployment.gateway_session
    try:
        bundle = session.contract_bundle
        route = bundle.route_manifest.route(*QUERY_REF).route
        assert (route.control_effect_class, route.privilege_tier, route.cancellation_policy) == (
            "read_only", "t0_read_only", "cancel_only")
        assert route.may_move_robot is False
        assert route.max_operation_duration_ms == module.REAL_SUPPORT_RELEASE_MAX_OPERATION_MS
        assert QUERY_REF in session.passive_read_executors_by_skill_ref
        registration = bundle.support_release_registration
        assert registration.support_object_ids == (SUPPORT,) and registration.side == "right"
        # Both poses go unchanged to the held-object motion Route: same public frame.
        assert registration.reference_frame_id == bundle.object_motion_profile.reference_frame_id
        assert registration.roles == bundle.perception_capture.back_projection.profile.roles
        assert len(registration.skill.definition.summary) <= 2000
        assert all(backend.starts == 0 for backend in lower.arm_backends.values()) and guard.starts == 0
    finally:
        session.close()


@pytest.mark.parametrize(("profile", "kwargs", "message"), [
    (SIX_F, {"authorized_actions": ("move_tcp_to_pose_safe", "set_gripper_state")}, "held-object motion"),
    ("dual_arm_hand_proxy_dmtac_grasp", {}, "RM65-6F"),
    (SIX_F, {"support_release_prompt_by_id": {SUPPORT: " "}}, "non-empty prompts"),
])
def test_real_support_release_prerequisites_fail_before_resources(monkeypatch, tmp_path, planners, profile, kwargs,
                                                                  message):
    _require_six_f_assets()
    lower, guard = _lower(monkeypatch, profile)
    with pytest.raises(ValueError, match=message):
        _compose(lower, tmp_path, **{"support_release_prompt_by_id": {SUPPORT: PROMPT}, **kwargs})
    assert guard.starts == 0


def test_real_cli_parses_and_checks_support_release_prompts(monkeypatch, tmp_path, capsys):
    from test_agent_gateway_real_rm65_cli import _argv

    from agent_gateway.profiles.real import gateway_cli
    from agent_gateway.profiles.real.session_authorization import (
        REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    )

    base = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN, socket_path=tmp_path / "g.sock")
    parser = gateway_cli.build_parser()
    args = parser.parse_args([*base, "--support-release-prompt", "green_tray=green tray",
                              "--support-release-prompt", "blue_box=blue box"])
    assert args.support_release_prompt == ["green_tray=green tray", "blue_box=blue box"]
    assert parser.parse_args(base).support_release_prompt is None
    monkeypatch.setattr(gateway_cli, "resolve_exclusive_rm65_skill_deployment_profile",
                        lambda *a, **k: pytest.fail("prompt errors must stop before device resolution"))
    for extra, message in (
            (["--support-release-prompt", "green tray"], "SUPPORT_ID=PROMPT"),
            (["--support-release-prompt", "a=x", "--support-release-prompt", "a=y"], "SUPPORT_ID=PROMPT"),
            (["--support-release-prompt", "green_tray=green tray"], "requires the move_grasped_object_to_pose")):
        with pytest.raises(SystemExit) as error:
            gateway_cli.main([*base, *extra])
        assert error.value.code == 2
        assert message in capsys.readouterr().err
