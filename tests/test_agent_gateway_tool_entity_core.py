"""Gateway cores of the tool-entity Skills over fake ports (default test gate).

Covers the hover query frame round trip with the real Station yaw, both
targets' planner outcome mappings, the RealMan hover planner adapter, the
relation budget path, and negative cases of the result validators.
"""

from __future__ import annotations

import hashlib
import io
import math
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles import tool_entity as core
from agent_gateway.profiles.mujoco.tool_entity import mujoco_hover_rejection_outcome
from agent_gateway.profiles.real import tool_entity as real
from agent_gateway.skills.perception import tool_entity as skill
from robot_motion import tool_entity_goals as goals
from robot_motion.tool_entity_scene import (
    HoverPlanOutcome,
    pose_matrix,
    pose_xyz_quat_xyzw,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

pytestmark = [pytest.mark.unit]

# canonical_station_from_dual_arm_solver_world on the reviewed station (180 deg yaw).
PUBLIC_FROM_WORLD = np.array([[-1., 0., 0., -0.472], [0., -1., 0., 0.], [0., 0., 1., -0.01], [0., 0., 0., 1.]])
CUBE_WORLD = np.array([-0.27, 0., 0.08])  # top-face centre
DOWN = [1., 0., 0., 0.]  # tool +Z along world -Z


def _entity():
    return goals.ToolEntityBox(
        tcp_from_box_xyz_m=(0., 0., -0.05), tcp_from_box_rotation=tuple(np.eye(3).reshape(-1)),
        half_extents_m=(0.05, 0.02, 0.05), tip_center_tcp_m=(0., 0., 0.),
        parts=(((0., 0.04, -0.05), (0.01, 0.005, 0.05)), ((0., -0.04, -0.05), (0.01, 0.005, 0.05))),
        grasp_region=((0., 0., -0.03), (0.01, 0.035, 0.03)), source={"geometry": "test"})


def _world_points():
    xs = np.arange(-0.5, 0.0, 0.01)
    table = np.array([(x, y, 0.01) for x in xs for y in np.arange(-0.25, 0.25, 0.01)])
    top = np.array([(CUBE_WORLD[0] + dx, dy, CUBE_WORLD[2]) for dx in np.linspace(-0.03, 0.03, 7)
                    for dy in np.linspace(-0.03, 0.03, 7)])
    return np.vstack([table, top])


class ScenePort:
    def __init__(self, public_from_geometry=PUBLIC_FROM_WORLD):
        self.public_from_geometry = public_from_geometry
        self.captured = []

    def capture_views(self, *, roles, expected_station_revision, cancel_requested):
        self.captured.append(tuple(roles))
        return tuple(SimpleNamespace(role=role, station_revision=expected_station_revision,
                                     capture_qpos_by_side={"left": np.zeros(6), "right": np.full(6, .1)})
                     for role in roles)

    def tcp_pose_by_side(self, qpos_by_side):
        return {"left": [0., 0.4, 0.5, 0., 0., 0., 1.], "right": [-0.25, 0.02, 0.40, *DOWN]}


class PlannerPort:
    translation_tolerance_m = 0.003
    rotation_tolerance_rad = 0.03
    start_clearance = "inherited-positive-v1"

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.targets = []

    def prepare_scene(self, *, captures, planning_qpos_by_side, cancel_requested):
        return core.ToolHoverScene(frame_id="world", points_m=_world_points(), quantization_margin_m=0.,
                                   planning_input="scene")

    def plan(self, *, scene, qpos_by_side, side, target_pose_xyz_quat_xyzw, time_budget_sec, cancel_requested):
        assert scene.planning_input == "scene" and side == "right" and 0 < time_budget_sec <= 5.
        self.targets.append(np.asarray(target_pose_xyz_quat_xyzw, dtype=float))
        if self.outcomes:
            return self.outcomes.pop(0)
        return HoverPlanOutcome(status="path_found", reason="planned_not_executed",
                                endpoint_pose_xyz_quat_xyzw=tuple(target_pose_xyz_quat_xyzw), trajectory_duration_sec=2.)


HOVER = {"component_id": "right_arm", "reference_frame_id": "station",
         "tip_offset_bounds_xy_m": [[-0.02, 0.02], [-0.02, 0.02]], "clearance_bounds_m": [0.03, 0.08],
         "orientation_policy": "tool_axis_down", "tool_axis_max_tilt_rad": 0., "max_candidates": 4}


def _hover(planner=None, scene_port=None, parameters=HOVER):
    anchor_public = (PUBLIC_FROM_WORLD @ np.r_[CUBE_WORLD, 1.])[:3]
    planner = PlannerPort() if planner is None else planner
    query = core.run_tool_hover_query(parameters=parameters, anchor_xyz_m=anchor_public.tolist(), side="right",
        roles=("main",), reference_frame_id="station", scene_port=ScenePort() if scene_port is None else scene_port,
        planner_port=planner, entity=_entity(), expected_station_revision=3, cancel_requested=lambda: False,
        clock=lambda: 0., deadline=30.)
    return query, planner, anchor_public


def test_hover_round_trips_the_station_yaw_and_finds_the_box_above_the_cube():
    query, planner, anchor_public = _hover()
    assert query["status"] == "path_found" and query["achieved"]["inside_requested_region"] is True
    skill.validate_hover_query(query, component_ids=("right_arm",), reference_frame_id="station")
    # The planner plans in the geometry (Station world) frame ...
    target_public = pose_matrix(query["target_tcp_pose_xyz_quat_xyzw"])
    sent = pose_matrix(planner.targets[query["selected_candidate_index"]])
    np.testing.assert_allclose(PUBLIC_FROM_WORLD @ sent, target_public, atol=1e-9)
    # ... and in that frame the gripper tip is above the cube, inside the bounds.
    tip_world = _entity().tip_world(sent)
    assert np.linalg.norm(tip_world[:2] - CUBE_WORLD[:2]) <= 0.02 * math.sqrt(2) + 1e-9
    lowest = float(_entity().corners_world(sent)[:, 2].min())
    assert 0.03 - 1e-9 <= lowest - CUBE_WORLD[2] <= 0.08 + 1e-9
    np.testing.assert_allclose(query["anchor_xyz_m"], anchor_public, atol=1e-12)
    source_world = pose_matrix(ScenePort().tcp_pose_by_side(None)["right"])
    np.testing.assert_allclose(pose_matrix(query["source_tcp_pose_xyz_quat_xyzw"]), PUBLIC_FROM_WORLD @ source_world,
                               atol=1e-9)
    assert query["motion_executed"] is False and query["execution_authorized"] is False
    assert query["start_clearance"] == "inherited-positive-v1" and query["planning_frame_id"] == "world"


def test_hover_retries_per_target_rejections_and_stops_on_scene_faults():
    retry = real.real_hover_rejection_outcome("target_safety_envelope_blocked")
    query, planner, _ = _hover(PlannerPort([retry]))
    assert query["status"] == "path_found" and query["selected_candidate_index"] == 1 and len(planner.targets) == 2
    fatal = real.real_hover_rejection_outcome("path_start_collision_blocked")
    query, planner, _ = _hover(PlannerPort([fatal]))
    assert (query["status"], query["reason"], len(planner.targets)) == ("path_rejected", "path_start_collision_blocked", 1)


def test_hover_requires_a_vertical_public_frame_before_capture():
    tilted = PUBLIC_FROM_WORLD.copy()
    tilted[:3, :3] = tilted[:3, :3] @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    port = ScenePort(tilted)
    with pytest.raises(ValueError, match="vertical"):
        _hover(scene_port=port)
    assert port.captured == []


@pytest.mark.parametrize("reason", sorted(real.REAL_HOVER_RETRYABLE_REASONS))
def test_real_per_target_rejections_are_retryable(reason):
    outcome = real.real_hover_rejection_outcome(reason)
    assert (outcome.status, outcome.retryable) == ("path_rejected", True)


def test_real_rejection_mapping_includes_the_planner_fallback_reasons():
    assert {"target_safety_envelope_blocked", "endpoint_safety_envelope_blocked", "path_search_exhausted",
            "path_target_collision_blocked"} <= real.REAL_HOVER_RETRYABLE_REASONS
    assert real.real_hover_rejection_outcome("path_planning_time_budget_exhausted").status == "query_incomplete"
    for reason in ("path_start_collision_blocked", "path_start_collision_check_unavailable"):
        outcome = real.real_hover_rejection_outcome(reason)
        assert (outcome.status, outcome.retryable) == ("path_rejected", False)


def test_mujoco_rejection_mapping():
    assert mujoco_hover_rejection_outcome("safe_tcp_path_search_exhausted", deadline_passed=False).status == (
        "query_incomplete")
    assert mujoco_hover_rejection_outcome("safe_tcp_target_blocked", deadline_passed=True).status == "query_incomplete"
    for reason in ("safe_tcp_endpoint_numerical_failure", "safe_tcp_target_blocked",
                   "safe_tcp_inherited_start_clearance_not_exited"):
        outcome = mujoco_hover_rejection_outcome(reason, deadline_passed=False)
        assert (outcome.status, outcome.retryable) == ("path_rejected", True)
    outcome = mujoco_hover_rejection_outcome("safe_tcp_start_blocked", deadline_passed=False)
    assert (outcome.status, outcome.retryable) == ("path_rejected", False)


class RealPlanner:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def plan_with_observed_cloud(self, cloud, **kwargs):
        self.calls.append((cloud, kwargs))
        if self.error is not None:
            raise RM65GraspTrajectoryPlanningError(self.error)
        return SimpleNamespace(qpos_by_side={side: np.full((3, 6), .2) for side in ("left", "right")},
                               time_from_start_sec=np.array([0., .5, 1.5]))

    def tool_tcp_poses_for_qpos_pair(self, *, qpos_by_side):
        pose = np.eye(4)
        pose[:3, 3] = (-0.27, 0., 0.2)
        return {"left": np.eye(4), "right": pose}


def test_real_hover_planner_builds_once_before_the_search_and_uses_the_provider_skew(monkeypatch):
    built, clouds = [], []
    planner = RealPlanner()

    def factory():
        built.append(1)
        return planner

    def cloud_builder(**kwargs):
        clouds.append(kwargs)
        return SimpleNamespace(points_m=np.zeros((4, 3)), quantization_margin_m=0.002)

    monkeypatch.setattr(real, "obstacle_cloud_from_captures", cloud_builder)
    port = real.RealToolHoverPlanner(planner_factory=factory, self_filter=object(), observation_config="config",
        geometry_frame_id="world", translation_tolerance_m=.01, rotation_tolerance_rad=.08,
        start_clearance="strict", profile_identity={"test": 1}, max_inter_camera_skew_sec=0.05)
    other = real.RealToolHoverPlanner(planner_factory=factory, self_filter=object(), observation_config="config",
        geometry_frame_id="world", translation_tolerance_m=.01, rotation_tolerance_rad=.08,
        start_clearance="strict", profile_identity={"test": 1}, max_inter_camera_skew_sec=0.1)
    assert port.profile_hash != other.profile_hash and built == []
    scene = port.prepare_scene(captures=("c",), planning_qpos_by_side={}, cancel_requested=lambda: False)
    assert built == [1] and clouds[0]["max_inter_camera_skew_sec"] == 0.05
    outcome = port.plan(scene=scene, qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)}, side="right",
                        target_pose_xyz_quat_xyzw=[-0.27, 0., 0.2, 0., 0., 0., 1.], time_budget_sec=5.,
                        cancel_requested=lambda: False)
    assert outcome.status == "path_found" and built == [1]
    np.testing.assert_allclose(outcome.endpoint_pose_xyz_quat_xyzw[:3], (-0.27, 0., 0.2))
    assert planner.calls[0][1]["time_budget_sec"] == 5. and planner.calls[0][1]["pose_goal_policy"] == (
        "within_arrival_tolerance")
    planner.error = "path_target_collision_blocked"
    rejected = port.plan(scene=scene, qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)}, side="right",
                         target_pose_xyz_quat_xyzw=[-0.27, 0., 0.2, 0., 0., 0., 1.], time_budget_sec=5.,
                         cancel_requested=lambda: False)
    assert (rejected.status, rejected.retryable) == ("path_rejected", True)


class MaskSegmenter:
    def __init__(self, clock, step):
        self.clock, self.step = clock, step
        self.calls = 0

    def try_segment_capture(self, *, capture, parameters):
        self.calls += 1
        self.clock[0] += self.step
        mask = np.zeros(capture.depth_m.shape, dtype=bool)
        mask[8:32, 8:32] = True
        buffer = io.BytesIO()
        np.save(buffer, mask, allow_pickle=False)
        data = buffer.getvalue()
        return SimpleNamespace(mask_npy=data, execution_result={"image_width": 40, "image_height": 40,
            "mask_pixel_count": int(mask.sum()), "mask_bounds_xyxy": [8, 8, 32, 32], "score": 0.9,
            "mask_media_content_hash": hashlib.sha256(data).hexdigest()})


class RelationPort:
    def capture_views(self, *, roles, expected_station_revision, cancel_requested):
        camera = np.eye(4)
        camera[:3, :3], camera[:3, 3] = np.diag([1., -1., -1.]), (0., 0., 1.)
        return tuple(SimpleNamespace(role=role, source_id="cam-" + role, acquired_at=1.0,
            station_revision=expected_station_revision, depth_m=np.full((40, 40), 0.9, dtype=np.float32),
            intrinsics={"fx": 100., "fy": 100., "cx": 20., "cy": 20.}, source_from_camera=camera,
            depth_content_hash="d" * 64, rgb_content_hash="r" * 64,
            capture_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)}) for role in roles)

    @staticmethod
    def segmentation_input(capture):
        return capture

    @staticmethod
    def tcp_pose_by_side(qpos_by_side):
        return {"left": [5., 5., 5., 0., 0., 0., 1.], "right": [0., 0., 0.4, *DOWN]}


RELATION = {"objects": [{"label": "cube", "language_prompt": "blue cube"},
                        {"label": "cup", "language_prompt": "black cup"}], "roles": None, "min_score": 0.3}
ROLES = ("main", "wrist_right")


def _relations(step=0.0, deadline=None):
    clock = [0.]
    result, _ = core.run_tool_object_relations(parameters=RELATION, roles=ROLES, reference_frame_id="station",
        scene_port=RelationPort(), segmentation_provider=MaskSegmenter(clock, step),
        tool_entity_by_side={"right": _entity()}, maximum_depth_m=1.5, expected_station_revision=3,
        cancel_requested=lambda: False, clock=lambda: clock[0], deadline=deadline)
    return result


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _validate_relations(result, parameters=RELATION):
    skill.validate_relation_result(result, roles=ROLES, sides=("left", "right"), reference_frame_id="station")
    skill.validate_relation_result_for_parameters(parameters, result, roles=ROLES)


def test_relations_keep_finished_views_when_the_budget_ends():
    result = _relations(step=0.6, deadline=1.0)
    _validate_relations(result)
    _validate_relations(_frozen(result), _frozen(RELATION))
    assert {(item["label"], item["role"]) for item in result["views"]} == {("cube", "main"), ("cup", "main")}
    assert {(item["label"], item["role"], item["reason"]) for item in result["unmeasured"]} == {
        ("cube", "wrist_right", "relation_budget_exhausted"), ("cup", "wrist_right", "relation_budget_exhausted")}
    assert result["unbound_sides"] == ["left"] and result["motion_executed"] is False


def test_relation_validators_reject_changed_or_incomplete_results():
    result = _relations()
    _validate_relations(result)
    mutations = {
        "unrequested label": lambda r: r["views"][0].update(label="other"),
        "missing view": lambda r: r["views"].pop(),
        "missing unmeasured": lambda r: r["unmeasured"].append({"label": "cube", "role": "main",
                                                                  "reason": "no_valid_depth"}),
        "role subset for null roles": lambda r: r.update(captures=r["captures"][:1],
            views=[v for v in r["views"] if v["role"] == "main"]),
        "motion claim": lambda r: r.update(motion_executed=True),
        "unknown class": lambda r: r["objects"][0].update(relation="grasped"),
        "unlisted arm": lambda r: r.update(unbound_sides=[]),
    }
    import copy

    for name, mutate in mutations.items():
        broken = copy.deepcopy(result)
        mutate(broken)
        with pytest.raises(ValueError):
            _validate_relations(broken)
        assert name


def _anchored(query, revision=3):
    return {"surface_anchor": {"measurement": {"world_xyz_m": list(query["anchor_xyz_m"]),
                                               "station_revision": revision}},
            "anchor_artifact_id": "artifact", "hover_query": query}


def test_hover_validators_reject_changed_or_uncertified_results():
    query, _, _ = _hover()
    skill.validate_hover_query(query, component_ids=("right_arm",), reference_frame_id="station")
    skill.validate_hover_result_for_parameters(HOVER, _anchored(query))
    skill.validate_hover_result_for_parameters(_frozen(HOVER), _frozen(_anchored(query)))
    import copy

    query_mutations = {
        "motion claim": lambda q: q.update(motion_executed=True),
        "authority claim": lambda q: q.update(execution_authorized=True),
        "uncertified path": lambda q: q["achieved"].update(inside_requested_region=False),
        "found without target": lambda q: q.update(target_tcp_pose_xyz_quat_xyzw=None),
        "unfound with target": lambda q: q.update(status="path_rejected"),
        "bad revision": lambda q: q.update(station_revision=-1),
        "other frame": lambda q: q.update(reference_frame_id="world"),
    }
    for mutate in query_mutations.values():
        broken = copy.deepcopy(query)
        mutate(broken)
        with pytest.raises(ValueError):
            skill.validate_hover_query(broken, component_ids=("right_arm",), reference_frame_id="station")
    result_mutations = {
        "bounds": lambda r: r["hover_query"].update(clearance_bounds_m=[0.02, 0.08]),
        "anchor": lambda r: r["hover_query"].update(anchor_xyz_m=[0., 0., 0.]),
        "orientation": lambda r: r["hover_query"].update(orientation_policy="preserve"),
        "revision": lambda r: r["surface_anchor"]["measurement"].update(station_revision=4),
    }
    for mutate in result_mutations.values():
        broken = copy.deepcopy(_anchored(query))
        mutate(broken)
        with pytest.raises(ValueError):
            skill.validate_hover_result_for_parameters(HOVER, broken)


def test_pose_helpers_round_trip():
    pose = [0.1, -0.2, 0.3, *np.array([0.1, 0.2, 0.3, 0.9]) / np.linalg.norm([0.1, 0.2, 0.3, 0.9])]
    back = pose_xyz_quat_xyzw(pose_matrix(pose))
    np.testing.assert_allclose(pose_matrix(back), pose_matrix(pose), atol=1e-12)
