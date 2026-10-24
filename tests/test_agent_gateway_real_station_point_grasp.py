from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real import point_grasp_render_gateway, realsense_perception
from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.perception_deployment import (
    RealPerceptionCaptureProfile,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.point_grasp_gateway import (
    StationBoundPointGraspProvider,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from agent_gateway.profiles.real.rm65_grasp_reachability import (
    RM65PointGraspReachabilityAdapter,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    StationPointGraspArmReachabilityProfile,
    build_station_bound_point_grasp_registration,
)
from agent_gateway.profiles.real.station_point_grasp_render_deployment import (
    REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
    build_station_bound_point_grasp_render_registration,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)
from deployments.perception.sam2_graspnet_point_grasp import (
    _normalize_graspnet_rigid_matrix,
)
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectSegment,
    encode_grounded_object_segment_artifacts,
)
from deployments.perception.sam2_vgn_point_grasp import (
    PointGraspCandidate,
    PointGraspResult,
)
from teleop_runtime.video_frame import VideoFrame


@pytest.mark.parametrize("axis", ((0., 0., 0.), (float("nan"), 0., 0.), (0., 0., 1.), (1., 0.)))
def test_wrist_camera_orientation_binding_rejects_invalid_radial_axis(axis):
    profile = StationPointGraspArmReachabilityProfile(arm_side="right",
        evaluator_profile_id="evaluator", evaluator_profile_hash="evaluator-hash",
        solver_reference_frame_id="solver", solver_reference_frame_spec_hash="solver-hash",
        frame_graph_hash="frames")
    with pytest.raises(ValueError, match="radial unit vector"):
        replace(profile, wrist_camera_side_axis_tool=axis)
    assert replace(profile, wrist_camera_side_axis_tool=(-1., 0., 0.)).semantic_content_hash != profile.semantic_content_hash


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("upright_feasible", (True, False))
def test_horizontal_grasp_excludes_camera_below_even_if_shorter_and_higher_score(side, upright_feasible):
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side,
        wrist_camera_side_axis_tool=(-1., 0., 0.))
    adapter.arm_base_origin_candidate_m = (-1., 0., 0.)
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    below = np.eye(4)
    below[:3, :3] = [[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]
    above = below @ np.diag([-1., -1., 1., 1.])
    candidates = tuple(SimpleNamespace(base_from_tcp=pose.copy(), score=score)
        for pose, score in ((below, .9), (above, .3)))
    assessed = []

    def assess(**kwargs):
        pose = kwargs["target_tool_tcp_world"]
        assessed.append(pose.copy())
        return SimpleNamespace(feasible=upright_feasible, selected_pregrasp_distance_m=.06,
            elapsed_ms=1., reason_code=None if upright_feasible else "collision",
            attempts=(), joint_travel_rad=(3., 1., 2., 1., .1, .3))

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(candidates, (0., 0., 0.), 5,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert result.selected_indices == ((1,) if upright_feasible else ())
    assert assessed and all(np.array_equal(pose, above) for pose in assessed)
    assert result.diagnostics["arm_shortlist_wrist_orientation_rejected_count"] == 1
    assert "side_grasp_wrist_camera_not_above" in result.diagnostics["arm_shortlist_wrist_orientation_rejections_json"]
    assert np.array_equal(candidates[0].base_from_tcp, below)
    assert np.array_equal(candidates[1].base_from_tcp, above)


def test_camera_upright_rule_leaves_vertical_grasp_roll_unconstrained():
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side="right",
        wrist_camera_side_axis_tool=(-1., 0., 0.))
    adapter.arm_base_origin_candidate_m = (-1., 0., 0.)
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    adapter.planner = SimpleNamespace(assess_local_grasp_chain=lambda **kwargs: SimpleNamespace(
        feasible=True, selected_pregrasp_distance_m=.06, elapsed_ms=1., reason_code=None,
        attempts=(), joint_travel_rad=(.1,) * 6))
    pose = np.diag([-1., 1., -1., 1.])
    result = adapter.select_candidates((SimpleNamespace(base_from_tcp=pose, score=.9),), (0.,0.,0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert result.selected_indices == (0,)
    assert result.diagnostics["arm_shortlist_wrist_orientation_rejected_count"] == 0


@pytest.mark.parametrize("side", ("left", "right"))
def test_shortlist_compares_checked_finger_exchange_and_prefers_small_proximal_motion(side):
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (1., 0., 0.)
    adapter.clock = lambda: 10.
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    candidates = []
    for index in range(2):
        pose = np.diag([-1., 1., -1., 1.])
        pose[0, 3] = index * .01
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.9 if index==0 else .3,
            symmetry_parent_index=None if index==0 else 0,
            orientation_variant_axis=None if index==0 else "z",
            orientation_variant_angle_deg=0. if index==0 else 180.))
    assessed = []
    def assess(**kwargs):
        index = int(round(kwargs["target_tool_tcp_world"][0, 3] * 100))
        assessed.append(index)
        return SimpleNamespace(feasible=True, selected_pregrasp_distance_m=.06,
            elapsed_ms=1., reason_code=None, attempts=(),
            joint_travel_rad=(3.6,1.,2.,1.,.1,.3) if index==0 else (.5,.3,1.,1.,2.7,.3))
    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(tuple(candidates), (0.,0.,0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert set(assessed) == {0, 1}
    assert result.selected_indices == (1,)
    assert result.diagnostics["arm_shortlist_local_chain_feasible_count"] == 2
    candidates[1].symmetry_parent_index = 2
    with pytest.raises(ValueError, match="lineage"):
        adapter.select_candidates(tuple(candidates), (0.,0.,0.), 1,
            measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)


def test_offline_reachability_never_invents_a_live_qpos_cut():
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.motion_port = None
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200)
    with pytest.raises(RuntimeError, match="capture-matched"):
        adapter._resolve_measured_qpos(measured_qpos_cut=None, measurement_reference_time=10.)
    cut = SimpleNamespace(measured_at=9.95)
    resolved, age = adapter._resolve_measured_qpos(measured_qpos_cut=cut, measurement_reference_time=10.)
    assert resolved is cut
    assert age == pytest.approx(50.)
    with pytest.raises(RuntimeError, match="camera-matched"):
        adapter._resolve_measured_qpos(measured_qpos_cut=cut, measurement_reference_time=11.)


def test_embodied_shortlist_reports_bounded_rejection_reasons() -> None:
    pose = np.eye(4)
    pose[2, 2] = -1.0
    pose[0, 0] = -1.0
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9 - i * 0.1)
        for i in range(3)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    reasons = iter(("approach_endpoint_collision_blocked", None, "approach_ik_search_exhausted"))

    def assess(**_kwargs):
        reason = next(reasons)
        return SimpleNamespace(
            feasible=reason is None,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.08 if reason is None else None,
            elapsed_ms=25.0,
            reason_code=reason,
            attempts=({"reason_code": reason},) if reason else (),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        3,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == (1,)
    attempts = json.loads(result.diagnostics["arm_shortlist_attempts_json"])
    assert len(attempts) == 3
    assert attempts[0]["reason_code"] == "approach_endpoint_collision_blocked"
    assert attempts[0]["branch_attempts"] == [
        {"reason_code": "approach_endpoint_collision_blocked"}
    ]
    assert json.loads(result.diagnostics["arm_shortlist_rejections_json"]) == {
        "approach_endpoint_collision_blocked": 1,
        "approach_ik_search_exhausted": 1,
    }
    assert result.diagnostics["arm_shortlist_rescue_activated"] is False


@pytest.mark.parametrize("reason", (
    "static_scene_clearance_blocked", "static_scene_collision_blocked",
    "static_scene_collision_check_unavailable",
))
def test_embodied_shortlist_reports_fixed_scene_without_spending_candidate_budget(reason):
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side="right")
    adapter.clock = lambda: 10.
    pair = {side: np.zeros(6) for side in ("left", "right")}

    def static_check(*, qpos_by_side, side):
        assert qpos_by_side is pair and side == "right"
        raise RM65GraspTrajectoryPlanningError(reason, details={
            "planning_phase": "static_scene_admission",
            "closest_geometry_names": ("left_hand_proxy_1", "table_0"),
            "minimum_distance_m": .015769077420228917,
        })

    def must_not_search(**_kwargs):
        pytest.fail("A fixed scene violation must not start IK, probes or rescue")

    adapter.planner = SimpleNamespace(require_static_grasp_scene=static_check,
        assess_local_grasp_chain=must_not_search, probe_grasp_endpoint_clearance=must_not_search)
    candidate = SimpleNamespace(base_from_tcp=tuple(np.eye(4).reshape(-1)), score=1.)
    result = adapter.select_candidates((candidate,), (0., 0., 0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side=pair),
        measurement_reference_time=10.)
    assert result.selected_indices == ()
    assert result.diagnostics["arm_shortlist_status"] == "static_scene_blocked"
    assert result.diagnostics["arm_shortlist_local_chain_assessment_count"] == 0
    assert not result.diagnostics["arm_shortlist_rescue_activated"]
    assert json.loads(result.diagnostics["arm_shortlist_static_scene_check_json"])["reason_code"] == reason


def test_embodied_shortlist_preserves_tilted_pose_and_direction_diversity() -> None:
    candidates = []
    poses = []
    for index, tilt_deg in enumerate((-35.0, -33.0, 5.0)):
        tilt = np.deg2rad(tilt_deg)
        approach = np.array((-np.cos(tilt), 0.0, np.sin(tilt)))
        pose = np.eye(4)
        pose[:3, 0] = np.cross((0.0, 1.0, 0.0), approach)
        pose[:3, 2] = approach
        pose[0, 3] = float(index)
        poses.append(pose.copy())
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=0.9 - index * 0.1))

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value.copy()
    assessed = []

    def assess(**kwargs):
        pose = kwargs["target_tool_tcp_world"]
        index = int(pose[0, 3])
        np.testing.assert_allclose(pose, poses[index])
        assessed.append(index)
        return SimpleNamespace(
            feasible=True, selected_pregrasp_distance_m=0.06,
            joint_travel_rad=(0.,) * 6,
            elapsed_ms=1.0, reason_code=None, attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        tuple(candidates), (0.0, 0.0, 0.0), 2,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    # The 40-degree-separated approach must precede a near duplicate.
    assert assessed == [0, 2]
    assert result.selected_indices == (0, 2)
    for candidate, pose in zip(candidates, poses, strict=True):
        np.testing.assert_array_equal(candidate.base_from_tcp, pose)


@pytest.mark.parametrize("classification", ["unresolved", "not_probed"])
def test_embodied_shortlist_explores_unknown_before_known_collision(
    monkeypatch, classification
) -> None:
    from agent_gateway.profiles.real import rm65_grasp_reachability as reachability

    if classification == "not_probed":
        monkeypatch.setattr(reachability, "_ARM_SHORTLIST_PROBE_MAX_COUNT", 1)
    candidates = []
    for index in range(2):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(SimpleNamespace(base_from_tcp=tuple(pose.flat), score=0.9 - index * 0.1))
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value.copy()
    assessed = []

    def probe(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        return SimpleNamespace(
            classification="collision_blocked" if index == 0 else classification,
            elapsed_ms=1.0, reason_code=None,
            best_minimum_distance_m=None, attempted_seed_count=1,
        )

    def assess(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 1, selected_pregrasp_distance_m=0.06,
            joint_travel_rad=(0.,) * 6,
            elapsed_ms=1.0, reason_code=None if index == 1 else "endpoint_collision_blocked",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe, assess_local_grasp_chain=assess,
    )
    result = adapter.select_candidates(
        tuple(candidates), (0.0, 0.0, 0.0), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert assessed == [1]
    assert result.selected_indices == (1,)


def test_embodied_shortlist_uses_endpoint_probe_only_to_reorder_full_checks() -> None:
    candidates = []
    for index in range(3):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.9 - index * 0.1,
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    probed = []
    assessed = []
    classifications = {
        0: "collision_blocked",
        1: "clearance_blocked",
        2: "clearance_safe",
    }

    def probe(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        probed.append(index)
        return SimpleNamespace(
            classification=classifications[index],
            elapsed_ms=1.0,
            reason_code=None,
            best_minimum_distance_m=0.03 if index == 2 else 0.003,
            attempted_seed_count=2,
        )

    def assess(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 2,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.06 if index == 2 else None,
            elapsed_ms=2.0,
            reason_code=None if index == 2 else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe,
        assess_local_grasp_chain=assess,
    )
    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert probed == [0, 1, 2]
    assert assessed == [2]
    assert result.selected_indices == (2,)
    assert result.diagnostics["arm_shortlist_endpoint_probe_count"] == 3


def test_embodied_shortlist_probes_support_clearance_family_before_model_scores() -> None:
    candidates = []
    for index in range(6):
        pose = np.diag((-1.0, 1.0, -1.0, 1.0))
        pose[0, 3] = float(index)
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.9 - index * 0.1,
                pose_source=(
                    "dmtac_support_clearance_variant_v1"
                    if index == 5
                    else "model_scored"
                ),
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(
        max_measured_qpos_age_ms=200.0,
        arm_side="right",
    )
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    probed = []
    assessed = []

    def probe(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        probed.append(index)
        return SimpleNamespace(
            classification=("clearance_safe" if index == 5 else "collision_blocked"),
            elapsed_ms=1.0,
            reason_code=None,
            best_minimum_distance_m=0.03 if index == 5 else -0.01,
            attempted_seed_count=2,
        )

    def assess(**kwargs):
        index = int(round(float(kwargs["target_tool_tcp_world"][0, 3])))
        assessed.append(index)
        return SimpleNamespace(
            feasible=index == 5,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.06 if index == 5 else None,
            elapsed_ms=2.0,
            reason_code=None if index == 5 else "endpoint_collision_blocked",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=probe,
        assess_local_grasp_chain=assess,
    )

    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert probed[0] == 5
    assert assessed == [5]
    assert result.selected_indices == (5,)


def test_embodied_shortlist_does_not_reject_an_unresolved_endpoint_probe() -> None:
    pose = np.diag((-1.0, 1.0, -1.0, 1.0))
    candidate = SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9)
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0
    adapter._solver_pose = lambda value: value
    adapter.planner = SimpleNamespace(
        probe_grasp_endpoint_clearance=lambda **_kwargs: SimpleNamespace(
            classification="unresolved",
            elapsed_ms=1.0,
            reason_code="path_planning_time_budget_exhausted",
            best_minimum_distance_m=None,
            attempted_seed_count=1,
        ),
        assess_local_grasp_chain=lambda **_kwargs: SimpleNamespace(
            feasible=True,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.04,
            elapsed_ms=2.0,
            reason_code=None,
            attempts=(),
        ),
    )

    result = adapter.select_candidates(
        (candidate,),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )

    assert result.selected_indices == (0,)


def test_embodied_shortlist_rescues_late_feasible_pose_after_extreme_scene_budget() -> None:
    """The recorded 8-second shortlist inspected only six of 66-97 poses."""

    pose = np.eye(4)
    pose[0, 0] = -1.0
    pose[2, 2] = -1.0
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=0.9 - i * 0.01)
        for i in range(10)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    # First five costs mirror the failed field run; the sixth has only a small
    # remainder under the old total deadline. The seventh pose is a valid
    # local chain, analogous to the later successful grasp on the same qpos.
    durations = (1.009, 1.662, 2.016, 1.068, 2.009, 0.238, 0.174)
    budgets = []

    def assess(**kwargs):
        index = len(budgets)
        budgets.append(kwargs["time_budget_sec"])
        elapsed[0] += durations[index]
        feasible = index == 6
        return SimpleNamespace(
            feasible=feasible,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.06 if feasible else None,
            elapsed_ms=durations[index] * 1000.0,
            reason_code=None if feasible else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == (6,)
    assert len(budgets) == 7
    assert all(0 < value <= 2.0 for value in budgets)
    assert result.diagnostics["arm_shortlist_rescue_activated"] is True
    assert result.diagnostics["arm_shortlist_rescue_assessment_count"] == 2
    assert result.diagnostics["arm_shortlist_budget_sec"] == 16.0


def test_embodied_shortlist_rescue_stops_at_total_deadline() -> None:
    pose = np.diag((-1.0, 1.0, -1.0, 1.0))
    candidates = tuple(
        SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=1.0 - i * 0.01)
        for i in range(20)
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    budgets = []

    def assess(**kwargs):
        budgets.append(kwargs["time_budget_sec"])
        elapsed[0] += 2.0
        return SimpleNamespace(
            feasible=False,
            joint_travel_rad=None,
            selected_pregrasp_distance_m=None,
            elapsed_ms=2000.0,
            reason_code="ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        candidates,
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert result.selected_indices == ()
    assert len(budgets) == 8
    assert elapsed[0] == 16.0
    assert result.diagnostics["arm_shortlist_budget_exhausted"] is True
    assert result.diagnostics["arm_shortlist_rescue_assessment_count"] == 4


def test_embodied_shortlist_checks_nearby_pose_before_spending_first_window() -> None:
    approaches = (
        (-1.0, 0.0, 0.0),
        (-0.8660254, -0.5, 0.0),
        (-0.5, -0.8660254, 0.0),
        (-0.8660254, 0.5, 0.0),
        (-0.5, 0.8660254, 0.0),
        (-1.0, 0.0, 0.0),
    )
    candidates = []
    for index, approach in enumerate(approaches):
        pose = np.eye(4)
        pose[:3, 2] = approach
        pose[0, 3] = index
        candidates.append(
            SimpleNamespace(
                base_from_tcp=tuple(pose.reshape(-1)),
                score=0.2 if index == 5 else 0.9 - index * 0.01,
            )
        )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200.0, arm_side="right")
    adapter.arm_base_origin_candidate_m = (1.0, 0.0, 0.0)
    adapter.clock = lambda: 10.0
    elapsed = [0.0]
    adapter.monotonic = lambda: elapsed[0]
    adapter._solver_pose = lambda value: value
    assessed = []

    def assess(**kwargs):
        index = int(kwargs["target_tool_tcp_world"][0, 3])
        assessed.append(index)
        elapsed[0] += 1.0
        feasible = index == 5
        return SimpleNamespace(
            feasible=feasible,
            joint_travel_rad=(0.,) * 6,
            selected_pregrasp_distance_m=0.06 if feasible else None,
            elapsed_ms=1000.0,
            reason_code=None if feasible else "ik_search_exhausted",
            attempts=(),
        )

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(
        tuple(candidates),
        (0.0, 0.0, 0.0),
        1,
        measured_qpos_cut=SimpleNamespace(measured_at=10.0, qpos_by_side={}),
        measurement_reference_time=10.0,
    )
    assert assessed == [0, 1, 3, 2, 5]
    assert result.selected_indices == (5,)
    assert result.diagnostics["arm_shortlist_rescue_activated"] is False


class _Source:
    def __init__(self, *, width=4, height=3) -> None:
        self.config = SimpleNamespace(
            camera_key="front",
            width=width,
            height=height,
            depth_width=width,
            depth_height=height,
            align_depth_to_color=True,
            enable_depth=True,
            color_space="rgb",
            serial="front-serial",
        )
        self.read_count = 0
        self.frame = VideoFrame(
            image=np.full((height, width, 3), 32, dtype=np.uint8),
            source_timestamp=9.95,
            receive_timestamp=9.951,
            available_timestamp=9.952,
            frame_index=11,
            metadata={
                "camera_key": "front",
                "serial": "front-serial",
                "source_generation": 2,
            },
            depth_image=np.full((height, width), 500, dtype=np.uint16),
            depth_metadata={
                "aligned_to": "color",
                "depth_scale_m_per_unit": 0.001,
            },
        )

    def get_latest_frame(self):
        self.read_count += 1
        return self.frame

    def device_info(self):
        return {
            "running": True,
            "lifecycle_state": "running",
            "worker_alive": True,
            "serial_number": "front-serial",
            "generation": 2,
            "intrinsics": {
                "width": self.config.width,
                "height": self.config.height,
                "fx": 4.0,
                "fy": 3.0,
                "ppx": 2.0,
                "ppy": 1.5,
            },
        }

    def close(self) -> None:
        return None


class _Planner:
    backend_id = "vgn"

    def __init__(self, point_segmenter: object, revision=None, events=None) -> None:
        self.point_segmenter = point_segmenter
        self.model_spec = SimpleNamespace(
            model_profile_id="vgn:test",
            semantic_content_hash="vgn-model:test",
        )
        self.semantic_content_hash = "station-point-grasp-planner:test"
        self.revision = revision
        self.events = events
        self.starts = 0
        self.closes = 0
        self.base_from_camera = None
        self.visible_scene_factory = None

    def start(self) -> None:
        self.starts += 1
        if self.events is not None:
            self.events.append("planner.start")

    def close(self) -> None:
        self.closes += 1
        if self.events is not None:
            self.events.append("planner.close")

    def plan(
        self,
        image,
        depth_m,
        *,
        pixel_uv,
        intrinsics,
        base_from_camera,
        max_candidates,
        visible_scene_factory=None,
    ) -> PointGraspResult:
        assert image.size == (4, 3)
        assert depth_m.shape == (3, 4)
        assert pixel_uv == (1, 1)
        assert intrinsics["width"] == 4
        assert max_candidates == 2
        self.base_from_camera = np.asarray(base_from_camera).copy()
        self.visible_scene_factory = visible_scene_factory
        camera_from_tcp = np.eye(4)
        camera_from_tcp[:3, 3] = (0.0, 0.0, 0.5)
        reference_from_tcp = self.base_from_camera @ camera_from_tcp
        target_camera = np.asarray((0.0, 0.0, 0.5, 1.0))
        target_reference = self.base_from_camera @ target_camera
        if self.revision is not None:
            self.revision["value"] += 1
        return PointGraspResult(
            target_point_camera_m=tuple(target_camera[:3]),
            target_point_base_m=tuple(target_reference[:3]),
            target_mask_bbox_xyxy=(0, 0, 4, 3),
            candidates=(
                PointGraspCandidate(
                    score=0.9,
                    model_score=0.95,
                    click_distance_m=0.01,
                    gripper_width_m=0.04,
                    camera_from_tcp=tuple(camera_from_tcp.reshape(-1)),
                    base_from_tcp=tuple(reference_from_tcp.reshape(-1)),
                    visible_collision_free=True,
                ),
            ),
        )


class _Localizer:
    def __init__(self, events=None) -> None:
        self.model_spec = SimpleNamespace(
            model_profile_id="sam2:test",
            semantic_content_hash="sam2-model:test",
        )
        self.semantic_content_hash = "sam2-localizer:test"
        self.events = events

    def start(self) -> None:
        if self.events is not None:
            self.events.append("localizer.start")

    def close(self) -> None:
        if self.events is not None:
            self.events.append("localizer.close")

    def localize(self, *_args, **_kwargs):
        return ()

    def segment_point(self, image, *, pixel_uv):
        mask = np.ones((image.height, image.width), dtype=bool)
        assert mask[pixel_uv[1], pixel_uv[0]]
        return mask

    def segment(
        self, image, *, language_prompt, pixel_uv, min_score
    ) -> GroundedObjectSegment:
        mask = np.ones((image.height, image.width), dtype=bool)
        return GroundedObjectSegment(
            mask=mask,
            score=max(0.9, float(min_score)),
            label=language_prompt,
            box_xyxy=(0, 0, image.width, image.height),
        )

    encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)


def _frame_binding():
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return build_real_perception_station_frame_binding(
        collision_scene_profile="station-point-grasp-test",
        scene_mount_hash="station-point-grasp-mounts:test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )


def _provider_stack(
    *,
    revision=None,
    arm_reachability_evaluator=None,
    backend_id="vgn",
    captured_robot_observation_filter=None,
    image_shape=(3, 4),
):
    height, width = image_shape
    source = _Source(width=width, height=height)
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:station-point-grasp-test",
    )
    revision = {"value": 7} if revision is None else revision
    perception = RealSenseAgentPerceptionProvider(
        source_binding=binding,
        current_station_revision=lambda: revision["value"],
        clock=lambda: 10.0,
    )
    perception.bind_profile(
        RealPerceptionCaptureProfile(
            rgb_sources=(("front", "main", width, height),),
            depth_sources=(("front", "main", width, height),),
            source_binding_hash=binding.semantic_content_hash,
        )
    )
    frame_binding = _frame_binding()
    localizer = _Localizer()
    planner = _Planner(localizer, revision=revision if revision.get("mutate") else None)
    planner.backend_id = backend_id
    if backend_id == "graspgenx":
        from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry

        planner.last_target_mask = np.ones((3, 4), dtype=bool)
        planner.solid_geometry = ParallelJawSolidGeometry(
            -0.043, 0.05, 0.01, 0.0255, 0.07
        )
    reachability_profile = (
        None
        if arm_reachability_evaluator is None
        else StationPointGraspArmReachabilityProfile(
            arm_side="right",
            evaluator_profile_id="rm65.reachability:test",
            evaluator_profile_hash="rm65-reachability-profile:test",
            solver_reference_frame_id=frame_binding.source_frame_id,
            solver_reference_frame_spec_hash=frame_binding.source_frame_spec_hash,
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        )
    )
    registration = build_station_bound_point_grasp_registration(
        aligned_rgbd_source_catalog=tuple(
            dict(item) for item in perception.aligned_rgbd_source_catalog()
        ),
        source_binding_hash=binding.semantic_content_hash,
        frame_binding_hash=frame_binding.semantic_content_hash,
        frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        source_frame_id=frame_binding.source_frame_id,
        source_frame_spec_hash=frame_binding.source_frame_spec_hash,
        reference_frame_id=frame_binding.reference_frame_id,
        reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
        backend_id=planner.backend_id,
        model_profile_id=planner.model_spec.model_profile_id,
        model_profile_hash=planner.model_spec.semantic_content_hash,
        inference_implementation_hash=planner.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        control_domain_id="real-perception-only:test",
        target_binding_id="real-realsense-only:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        arm_reachability_profile=reachability_profile,
    )
    provider = StationBoundPointGraspProvider(
        planner=planner,
        registration=registration,
        perception_provider=perception,
        frame_binding=frame_binding,
        current_station_revision=lambda: revision["value"],
        arm_reachability_evaluator=arm_reachability_evaluator,
        captured_robot_observation_filter=captured_robot_observation_filter,
    )
    return source, revision, planner, registration, provider


def _request(*, include_arm_reachability=None):
    parameters = {
        "role": "main",
        # Wire decoding freezes JSON arrays before provider validation.
        "target_pixel_uv": (1, 1),
        "max_candidates": 2,
    }
    if include_arm_reachability is not None:
        parameters["include_arm_reachability"] = include_arm_reachability
    return SimpleNamespace(
        parameters=parameters,
        input_artifact_bindings=(),
        expected_station_revision=7,
    )


def test_station_point_grasp_plan_runs_inside_non_motion_read_window() -> None:
    _source, _revision, _planner, _registration, provider = _provider_stack()
    events: list[str] = []

    class _Window:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            events.append("exit")
            return False

    provider.non_motion_read_window = lambda: _Window()
    provider.plan_request(_request())

    assert events == ["enter", "exit"]


@pytest.mark.parametrize("stage", ("sam2", "graspgenx"))
@pytest.mark.parametrize("outcome", ("expired", "cancelled", "model_error"))
def test_preparation_guard_reaches_actual_planner_model_boundaries(stage, outcome):
    import threading
    from deployments.perception.dmtac_grasp_overlay import dmtac_graspgenx_alignment_profile
    from deployments.perception.sam2_graspgenx_point_grasp import Sam2GraspGenXPointGraspPlanner
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError

    _, _, original, _, provider = _provider_stack(backend_id="graspgenx", image_shape=(8, 8))
    now, cancelled, calls = [0.0], [False], []
    model_error = ValueError("original model failure")

    def guard():
        if cancelled[0]:
            raise RuntimeError("cancelled during preparation")
        if now[0] >= 1.0:
            raise RM65GraspTrajectoryPlanningError("grasp_fast_search_budget_exhausted")

    def finish(name):
        calls.append(name)
        if name == stage:
            now[0] = 1.0
            cancelled[0] = outcome in {"cancelled", "model_error"}
            if outcome == "model_error":
                raise model_error

    def segment(image, *, pixel_uv):
        finish("sam2")
        return np.ones((image.height, image.width), dtype=bool)

    def infer(*args, **kwargs):
        finish("graspgenx")
        return np.eye(4)[None], np.asarray([0.8])

    # Use actual provider/plan/_prepare_pool methods with already-running fake
    # model ports. No constructor, model start, device or output publication.
    planner = object.__new__(Sam2GraspGenXPointGraspPlanner)
    planner._alignment = dmtac_graspgenx_alignment_profile()
    planner._lock = threading.RLock()
    planner._started, planner._closed = True, False
    planner.point_segmenter = SimpleNamespace(segment_point=segment)
    planner.client = SimpleNamespace(infer_object=infer)
    planner._solid_geometry = original.solid_geometry
    planner.planner, planner.num_grasps = "fake", 1
    planner.score_threshold, planner.server_top_k = 0.0, 1
    provider.planner = planner

    expected = ValueError if outcome == "model_error" else RuntimeError
    with pytest.raises(expected) as caught:
        provider.plan_request(_request(), preparation_guard=guard)
    if outcome == "model_error":
        assert caught.value is model_error
    elif outcome == "expired":
        assert isinstance(caught.value, RM65GraspTrajectoryPlanningError)
        assert caught.value.reason_code == "grasp_fast_search_budget_exhausted"
    else:
        assert not isinstance(caught.value, RM65GraspTrajectoryPlanningError)
    assert calls == (["sam2"] if stage == "sam2" else ["sam2", "graspgenx"])
    assert planner.last_plan_timing_ms[stage + "_inference"] >= 0.0


@pytest.mark.parametrize("boundary", ("window", "lock"))
def test_preparation_guard_stops_after_actual_provider_wait_boundary(boundary):
    from contextlib import nullcontext
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspTrajectoryPlanningError

    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")
    now = [0.0]

    class Wait:
        def __enter__(self):
            now[0] = 1.0
            return self

        def __exit__(self, *args):
            return False

    if boundary == "window":
        provider.non_motion_read_window = Wait
    else:
        provider.non_motion_read_window = nullcontext
        provider._lock = Wait()

    def guard():
        if now[0] >= 1.0:
            raise RM65GraspTrajectoryPlanningError("grasp_fast_search_budget_exhausted")

    with pytest.raises(RM65GraspTrajectoryPlanningError, match="grasp_fast_search_budget_exhausted"):
        provider.plan_request(_request(), preparation_guard=guard)
    assert source.read_count == 0


def test_graspgenx_capture_filter_is_wired_to_planner() -> None:
    qpos = SimpleNamespace(
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        semantic_content_hash="measured-qpos:test",
        measured_at=9.95,
        available_at=9.952,
        revision=1,
        evidence_refs=("qpos:test",),
    )
    _, _, planner, _, provider = _provider_stack(
        backend_id="graspgenx",
        captured_robot_observation_filter=lambda **_kwargs: None,
    )
    provider.motion_port = SimpleNamespace(
        latest_measured_qpos=lambda: qpos,
    )

    provider.plan_request(_request())

    assert planner.visible_scene_factory is not None

def test_station_point_grasp_route_is_live_bound_read_only_and_proposal_only() -> None:
    source, _revision, planner, registration, provider = _provider_stack()
    route = registration.skill.routes[0]
    assert route.skill_version == "2"
    assert route.supported_modes == ("real_observe",)
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.may_move_robot is False
    assert route.requires_operator_confirmation is False
    assert registration.skill.io_contract.input_artifact_slots == ()
    assert registration.skill.io_contract.output_artifact_slots[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )

    prepared = provider.plan_request(_request())

    assert source.read_count == 1
    assert planner.base_from_camera is not None
    assert prepared.execution_result["reference_frame_id"] == "canonical_station"
    assert prepared.execution_result["station_revision"] == 7
    assert prepared.execution_result["source_generation"] == 2
    assert prepared.execution_result["frame_index"] == 11
    assert prepared.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert prepared.execution_result["downstream_control_effects"] == "forbidden"
    assert "selected_ordinal" not in prepared.execution_result
    registration.payload_validator.validate_parameters(_request().parameters)
    with pytest.raises(ValueError, match="parameter fields changed"):
        registration.payload_validator.validate_parameters(
            _request().parameters
            | {"T_reference_from_camera": list(np.eye(4).reshape(-1))}
        )
    registration.payload_validator.validate_execution_result(prepared.execution_result)
    registration.payload_validator.validate_execution_result_for_parameters(
        _request().parameters,
        prepared.execution_result,
    )


def test_station_point_grasp_profile_accepts_graspgenx_backend() -> None:
    _source, _revision, _planner, registration, _provider = _provider_stack()

    profile = replace(registration.profile, backend_id="graspgenx")

    assert profile.backend_id == "graspgenx"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "station_revision",
        "target_pixel_uv",
        "camera_intrinsics",
        "T_reference_from_camera",
        "depth_content_hash",
        "candidate_score",
        "snapshot_bytes",
        "snapshot_length",
        "missing_snapshot",
    ],
)
@pytest.mark.parametrize("cad_bound", [False, True])
def test_graspgenx_capture_snapshot_artifact_is_replayed_through_real_profile(
    tmp_path, monkeypatch, mutation, cad_bound
) -> None:
    from agent_gateway.profiles.real.grasp_approach_plan import (
        build_grasp_visible_scene_reader,
    )
    from agent_gateway.profiles.real.point_grasp_gateway import (
        build_station_bound_point_grasp_executor,
    )
    from agent_gateway.profiles.real.station_point_grasp_deployment import (
        REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
    )
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
    from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot

    source, _, planner, registration, provider = _provider_stack(backend_id="graspgenx")
    from deployments.perception.grasp_planning_priority import GRASP_PLANNING_ORDER_SCOPE
    planner.last_filter_summary = {"arm_shortlist_planning_order_json": "[0]",
        "arm_shortlist_planning_order_scope": GRASP_PLANNING_ORDER_SCOPE}
    if cad_bound:
        from deployments.perception.dmtac_grasp_overlay import (
            reviewed_4b2_cad_solid_geometry,
        )

        planner.solid_geometry = reviewed_4b2_cad_solid_geometry()
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    executor = build_station_bound_point_grasp_executor(
        registration=registration,
        provider=provider,
        artifact_store=store,
        runtime_session_id="station-point-grasp:test",
        current_station_revision=lambda: 7,
        max_retained_operations=4,
        clock=lambda: 10.0,
    )
    # Exercise the real adapter and Artifact store without opening any device.
    request = _request()
    request.semantic_content_hash = "a" * 64
    result = executor._read(request, SimpleNamespace(operation_id="snapshot:test"))
    bindings = result.output_artifact_bindings
    snapshot = next(
        x.artifact
        for x in bindings
        if x.slot_id == REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID
    )
    arrays, metadata = decode_grasp_rgbd_snapshot(
        store.managed_read_path(snapshot.artifact_id).read_bytes()
    )
    assert source.read_count == 1
    assert arrays["depth_m"].shape == (3, 4)
    assert metadata["capture"] == result.execution_result
    assert metadata["candidate_filter_summary"] == planner.last_filter_summary
    assert "planning_candidate_ordinals" not in result.execution_result
    assert metadata["measured_qpos"] is None
    assert arrays["left_qpos"].size == arrays["right_qpos"].size == 0
    reader = build_grasp_visible_scene_reader(
        artifact_store=store,
        frame_graph=provider.frame_binding.frame_graph,
        execution_frame_id=provider.frame_binding.source_frame_id,
    )
    # Retained operations freeze JSON arrays to tuples, unlike NPZ metadata.
    retained = AgentSkillOperation(
        operation_id="snapshot:test",
        request_hash=request.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        status="completed",
        reason_code="passive_read_completed",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=9.0,
        updated_at=10.0,
        deadline_at=20.0,
        execution_outcome="completed",
        execution_receipt_refs=("point-grasp-receipt:test",),
        execution_result=result.execution_result,
        output_artifact_bindings=bindings,
    )
    assert isinstance(metadata["capture"]["target_pixel_uv"], list)
    assert isinstance(retained.execution_result["target_pixel_uv"], tuple)
    scene = reader(retained, retained.execution_result)
    assert len(scene.points) == 12
    assert scene.geometry == planner.solid_geometry
    assert scene.capture_qpos_by_side is None
    with pytest.raises(ValueError, match="capture differ"):
        reader(
            SimpleNamespace(output_artifact_bindings=bindings),
            {**result.execution_result, "station_revision": 8},
        )
    if mutation is None:
        return
    changed = copy.deepcopy(result.execution_result)
    expected_error = "capture differ"
    expected_exception = ValueError
    if mutation == "station_revision":
        changed[mutation] += 1
    elif mutation == "target_pixel_uv":
        changed[mutation][0] += 1
    elif mutation == "camera_intrinsics":
        changed[mutation]["fx"] += 1
    elif mutation == "T_reference_from_camera":
        changed[mutation][3] += 0.001
    elif mutation == "depth_content_hash":
        changed[mutation] = "0" * 64
    elif mutation == "candidate_score":
        changed["candidates"][0]["score"] += 0.01
    elif mutation in {"snapshot_bytes", "snapshot_length"}:
        path = store.managed_read_path(snapshot.artifact_id)
        original = path.read_bytes()
        # Simulate corrupted read bytes without modifying the immutable CAS.
        corrupted = tmp_path / "corrupted_snapshot.npz"
        corrupted.write_bytes(
            original[:-1]
            if mutation == "snapshot_length"
            else original[:-1] + bytes([original[-1] ^ 1])
        )
        # Integrity belongs to the one-open store read, not a second read of
        # a previously verified path. Keep the immutable original CAS intact.
        monkeypatch.setattr(store, "_blob_path", lambda _sha256: corrupted)
        from agent_gateway.runtime.artifact_store import ArtifactStoreIntegrityError
        expected_exception = ArtifactStoreIntegrityError
        expected_error = "failed verification"
    else:
        assert mutation == "missing_snapshot"
        retained = replace(retained, output_artifact_bindings=())
        expected_error = "no unique RGB-D snapshot"
    retained = replace(retained, execution_result=changed)
    with pytest.raises(expected_exception, match=expected_error):
        reader(retained, retained.execution_result)


def test_grasp_scene_reader_filters_robot_observations_from_measured_qpos_snapshot(
    tmp_path,
) -> None:
    from agent_gateway.profiles.real.grasp_approach_plan import (
        build_grasp_visible_scene_reader,
    )
    from agent_gateway.profiles.real.point_grasp_gateway import (
        build_station_bound_point_grasp_executor,
    )
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
    from robot_motion.rm65.captured_robot_observations import (
        RM65CapturedRobotObservationFilterResult,
    )

    source, _, planner, registration, provider = _provider_stack(backend_id="graspgenx")
    target_mask = np.zeros((3, 4), dtype=bool)
    target_mask[1, 1] = True
    planner.last_target_mask = target_mask
    capture_qpos = {
        "left": np.linspace(0.0, 0.5, 6),
        "right": np.linspace(0.1, 0.6, 6),
    }
    provider.motion_port = SimpleNamespace(
        latest_measured_qpos=lambda: SimpleNamespace(
            qpos_by_side=capture_qpos,
            semantic_content_hash="measured-qpos:test",
            revision=4,
            measured_at=9.95,
            available_at=9.952,
            evidence_refs=("measured-qpos-receipt:test",),
        )
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    executor = build_station_bound_point_grasp_executor(
        registration=registration,
        provider=provider,
        artifact_store=store,
        runtime_session_id="station-point-grasp:test",
        current_station_revision=lambda: 7,
        max_retained_operations=4,
        clock=lambda: 10.0,
    )
    request = _request()
    request.semantic_content_hash = "b" * 64
    result = executor._read(
        request,
        SimpleNamespace(operation_id="snapshot:with-qpos"),
    )
    retained = AgentSkillOperation(
        operation_id="snapshot:with-qpos",
        request_hash=request.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        status="completed",
        reason_code="passive_read_completed",
        station_revision_at_admission=7,
        current_station_revision=7,
        started_at=9.0,
        updated_at=10.0,
        deadline_at=20.0,
        execution_outcome="completed",
        execution_receipt_refs=("point-grasp-receipt:test",),
        execution_result=result.execution_result,
        output_artifact_bindings=result.output_artifact_bindings,
    )
    calls = []

    def filter_capture(**kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(kwargs["target_mask"], target_mask)
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                kwargs["capture_qpos_by_side"][side], capture_qpos[side]
            )
        filtered_depth = np.asarray(kwargs["depth_m"], dtype=float).copy()
        filtered_depth[0, 0] = 0.0
        removed = np.zeros(filtered_depth.shape, dtype=bool)
        removed[0, 0] = True
        return RM65CapturedRobotObservationFilterResult(
            depth_m=filtered_depth,
            removed_mask=removed,
            evidence={
                "schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
                "removed_background_points": 1,
                "removed_target_points": 0,
            },
        )

    reader = build_grasp_visible_scene_reader(
        artifact_store=store,
        frame_graph=provider.frame_binding.frame_graph,
        execution_frame_id=provider.frame_binding.source_frame_id,
        captured_robot_observation_filter=filter_capture,
    )
    scene = reader(retained, retained.execution_result)

    assert source.read_count == 1
    assert len(calls) == 1
    assert len(scene.points) == 11
    assert scene.target_mask.sum() == 1
    assert scene.capture_self_observation_evidence["removed_background_points"] == 1
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            scene.capture_qpos_by_side[side], capture_qpos[side]
        )


def test_graspgenx_inference_failure_retains_original_input_without_faking_a_mask():
    from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot

    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")

    def fail(*_args, **_kwargs):
        raise RuntimeError("no candidate found")

    planner.plan = fail
    retained = []
    with pytest.raises(RuntimeError, match="no candidate found"):
        provider.plan_request(_request(), input_snapshot_sink=retained.append)
    assert source.read_count == 1 and len(retained) == 1
    arrays, metadata = decode_grasp_rgbd_snapshot(retained[0])
    assert arrays["depth_m"].shape == (3, 4)
    assert metadata["target_mask_available"] is False
    assert metadata["scope"] == "pre_inference_input_not_a_validated_grasp_plan"


def test_missing_measured_qpos_stops_inference_but_keeps_camera_input():
    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")
    provider.motion_port = SimpleNamespace(latest_measured_qpos=lambda: None)

    def must_not_infer(*_args, **_kwargs):
        pytest.fail("missing qpos must stop this path")

    planner.plan = must_not_infer
    retained = []
    with pytest.raises(RuntimeError, match="camera-matched measured qpos"):
        provider.plan_request(_request(), input_snapshot_sink=retained.append)
    assert source.read_count == 1 and len(retained) == 1


def test_station_point_grasp_v3_reachability_is_request_gated_and_annotation_only() -> (
    None
):
    calls = []

    def evaluate(candidates):
        calls.append(candidates)
        profile = registration.profile.arm_reachability_profile
        assert profile is not None
        context = {
            "arm_side": profile.arm_side,
            "evaluator_profile_id": profile.evaluator_profile_id,
            "evaluator_profile_hash": profile.evaluator_profile_hash,
            "candidate_reference_frame_id": registration.profile.reference_frame_id,
            "solver_reference_frame_id": profile.solver_reference_frame_id,
            "solver_reference_frame_spec_hash": (
                profile.solver_reference_frame_spec_hash
            ),
            "frame_graph_hash": profile.frame_graph_hash,
            "time_budget_ms": profile.time_budget_ms,
            "max_solve_cycles": profile.max_solve_cycles,
            "max_measured_qpos_age_ms": profile.max_measured_qpos_age_ms,
            "standard_position_tolerance_m": (profile.standard_position_tolerance_m),
            "standard_orientation_tolerance_deg": (
                profile.standard_orientation_tolerance_deg
            ),
            "basic_position_tolerance_m": profile.basic_position_tolerance_m,
            "basic_orientation_tolerance_deg": (
                profile.basic_orientation_tolerance_deg
            ),
            "measured_qpos_revision": 4,
            "measured_qpos_at": 9.95,
            "measured_qpos_age_ms": 50.0,
            "evaluation_scope": ("finite_time_endpoint_ik_not_path_or_grasp_success"),
            "candidate_order_effect": "annotation_only_order_unchanged",
        }
        annotation = {
            "classification": "standard_reachable",
            "position_residual_m": 0.001,
            "orientation_residual_deg": 0.5,
            "solve_cycles": 2,
            "elapsed_ms": 4.0,
            "termination_reason": "standard_tolerance_met",
        }
        return context, tuple(annotation for _candidate in candidates)

    _source, _revision, _planner, registration, provider = _provider_stack(
        arm_reachability_evaluator=evaluate
    )
    assert registration.skill.definition.skill_version == "3"

    disabled_request = _request(include_arm_reachability=False)
    disabled = provider.plan_request(disabled_request).execution_result
    assert calls == []
    assert "arm_reachability_context" not in disabled
    assert "arm_reachability" not in disabled["candidates"][0]
    registration.payload_validator.validate_execution_result(disabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        disabled_request.parameters, disabled
    )

    enabled_request = _request(include_arm_reachability=True)
    enabled = provider.plan_request(enabled_request).execution_result
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0][0],
        np.asarray(enabled["candidates"][0]["T_reference_from_tool_tcp"]).reshape(4, 4),
    )
    assert enabled["candidates"][0]["arm_reachability"]["classification"] == (
        "standard_reachable"
    )
    registration.payload_validator.validate_parameters(enabled_request.parameters)
    registration.payload_validator.validate_execution_result(enabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        enabled_request.parameters, enabled
    )

    seed_exhausted = copy.deepcopy(enabled)
    seed_exhausted_annotation = seed_exhausted["candidates"][0]["arm_reachability"]
    seed_exhausted_annotation.update(
        {
            "classification": "difficult_to_reach",
            "position_residual_m": 0.06,
            "orientation_residual_deg": 25.0,
            "solve_cycles": (
                registration.profile.arm_reachability_profile.max_solve_cycles
            ),
            "termination_reason": "seed_budget_exhausted",
        }
    )
    registration.payload_validator.validate_execution_result(seed_exhausted)

    with pytest.raises(ValueError, match="does not match the request"):
        registration.payload_validator.validate_execution_result_for_parameters(
            disabled_request.parameters, enabled
        )


def test_rm65_reachability_adapter_accepts_normalized_graspnet_candidate() -> None:
    binding = _frame_binding()
    raw_pose = np.asarray(
        [
            [
                -0.17259738000158176,
                0.18359564713816373,
                0.9677307159760798,
                -0.13079618433586204,
            ],
            [
                0.04199108059770058,
                -0.9802105728615037,
                0.19345253616022093,
                -0.07363994251000638,
            ],
            [
                0.9840969607643857,
                0.07402546072335396,
                0.16147240422405731,
                0.1374787059167376,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = _normalize_graspnet_rigid_matrix(
        raw_pose,
        field_name="GraspNet camera_from_grasp",
    )

    class _MotionPort:
        @staticmethod
        def latest_measured_qpos():
            return SimpleNamespace(
                qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
                revision=4,
                measured_at=9.95,
            )

    class _ReachabilityPlanner:
        @staticmethod
        def assess_endpoint_ik(**kwargs):
            np.testing.assert_allclose(
                kwargs["target_tool_tcp_world"][:3, :3].T
                @ kwargs["target_tool_tcp_world"][:3, :3],
                np.eye(3),
                atol=1e-12,
                rtol=0.0,
            )
            return SimpleNamespace(
                to_dict=lambda: {
                    "classification": "standard_reachable",
                    "position_residual_m": 0.001,
                    "orientation_residual_deg": 0.5,
                    "solve_cycles": 1,
                    "elapsed_ms": 1.0,
                    "termination_reason": "standard_tolerance_met",
                }
            )

    profile = SimpleNamespace(
        solver_reference_frame_id=binding.source_frame_id,
        solver_reference_frame_spec_hash=binding.source_frame_spec_hash,
        frame_graph_hash=binding.frame_graph.semantic_content_hash,
        arm_side="right",
        evaluator_profile_id="rm65.reachability:test",
        evaluator_profile_hash="rm65-reachability-profile:test",
        time_budget_ms=20.0,
        max_solve_cycles=2,
        max_measured_qpos_age_ms=200.0,
        standard_position_tolerance_m=0.005,
        standard_orientation_tolerance_deg=2.0,
        basic_position_tolerance_m=0.010,
        basic_orientation_tolerance_deg=5.0,
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = profile
    adapter.planner = _ReachabilityPlanner()
    adapter.motion_port = _MotionPort()
    adapter.frame_graph = binding.frame_graph
    adapter.candidate_reference_frame_id = binding.reference_frame_id
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0

    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        adapter((raw_pose,))
    context, annotations = adapter((normalized,))

    assert context["candidate_order_effect"] == "annotation_only_order_unchanged"
    assert annotations[0]["classification"] == "standard_reachable"
    adapter.clock = lambda: 10.3
    matched_cut = adapter.motion_port.latest_measured_qpos()
    capture_context, _ = adapter(
        (normalized,),
        measured_qpos_cut=matched_cut,
        measurement_reference_time=10.0,
    )
    assert capture_context["measured_qpos_age_ms"] == pytest.approx(50.0)


def test_station_point_grasp_fails_if_revision_changes_during_inference() -> None:
    revision = {"value": 7, "mutate": True}
    _source, _revision, _planner, _registration, provider = _provider_stack(
        revision=revision
    )

    with pytest.raises(RuntimeError, match="during point-grasp model inference"):
        provider.plan_request(_request())


def test_debug_target_coordinate_overlay_projects_depth_target_in_station_frame() -> (
    None
):
    image = Image.new("RGB", (200, 160), (10, 10, 10))
    rendered = point_grasp_render_gateway._render_target_coordinate_debug_overlay(
        image,
        target_point_reference_m=(0.0, 0.0, 1.0),
        reference_from_camera=np.eye(4),
        intrinsics={
            "width": 200,
            "height": 160,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 100.0,
            "cy": 80.0,
        },
    )

    assert rendered is not image
    assert image.getpixel((100, 74)) == (10, 10, 10)
    assert rendered.getpixel((100, 74)) == (0, 255, 255)
    assert rendered.getpixel((103, 80)) == (255, 64, 64)
    assert rendered.getpixel((100, 84)) == (64, 255, 64)


def test_debug_target_coordinate_profile_is_default_off_and_hash_bound() -> None:
    _source, _revision, _planner, point_grasp, _provider = _provider_stack()
    default = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
    )
    enabled = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
        debug_target_coordinate_overlay=True,
    )

    assert default.profile.debug_target_coordinate_overlay is False
    assert enabled.profile.debug_target_coordinate_overlay is True
    assert (
        default.profile.semantic_content_hash != enabled.profile.semantic_content_hash
    )


def test_point_grasp_is_default_off_and_lifecycle_order_is_reversible(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    source = _Source()
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:composition-test",
    )
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id="rpent:test",
        primary_client_uid=1000,
    )
    default = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "default.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:default-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        clock=lambda: 10.0,
    )
    try:
        assert default.station_point_grasp_registration is None
        assert default.station_point_grasp_render_registration is None
        assert [
            entry.route.skill_id
            for entry in default.contract_bundle.route_manifest.skill_routes
        ] == [
            "perception.back_project",
            "perception.capture_depth",
            "perception.capture_rgb",
        ]
    finally:
        default.application.close()

    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda _sources: events.append("cameras.start"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda _sources: events.append("cameras.admit"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda _sources: events.append("cameras.close"),
    )
    localizer = _Localizer(events)
    planner = _Planner(localizer, events=events)
    composition = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "enabled.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:enabled-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        object_localizer=localizer,
        point_grasp_planner=planner,
        clock=lambda: 10.0,
    )

    class _Server:
        def serve_forever(self, poll_interval=0.5):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            events.append("gateway.close")
            composition.session.close()

    composition.application.server_factory = lambda: (
        events.append("socket.create") or _Server()
    )
    composition.application.start()
    assert events[:5] == [
        "cameras.start",
        "cameras.admit",
        "localizer.start",
        "planner.start",
        "socket.create",
    ]
    adapter = composition.session.wire_adapter(authenticated_client_id="rpent:test")
    observation = adapter.get_observation_snapshot()
    manifest = composition.contract_bundle.route_manifest
    entry = manifest.route("perception.plan_point_grasp", "2")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    submitted = adapter.submit_read_only_skill_request(
        AgentSkillRequest(
            client_id="rpent:test",
            request_id="request:station-point-grasp:test",
            idempotency_key="idempotency:station-point-grasp:test",
            client_sequence=1,
            runtime_session_id=manifest.runtime_session_id,
            skill_route_manifest_hash=manifest.semantic_content_hash,
            expected_station_revision=int(observation["station_revision"]),
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=(entry.deployment_binding.deployment_binding_hash),
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.parameter_profile_hash,
            parameters={
                "role": "main",
                "target_pixel_uv": [1, 1],
                "max_candidates": 2,
            },
            observation_snapshot_id=str(observation["snapshot_id"]),
            requested_mode="real_observe",
            requested_timeout_ms=100_000,
        ).to_wire()
    )
    assert submitted["accepted"] is True
    operation_id = submitted["operation"]["operation_id"]
    assert composition.session.station_point_grasp_executor is not None
    assert composition.session.station_point_grasp_executor.wait(
        operation_id,
        timeout_sec=2.0,
    )
    operation = composition.session.service.operation_store.get(operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert operation.output_artifact_bindings[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )
    render_entry = manifest.route("perception.render_point_grasp_candidate", "1")
    render_route = render_entry.route
    assert render_route.control_effect_class == "read_only"
    assert render_route.privilege_tier == "t0_read_only"
    assert render_route.cancellation_policy == "cancel_only"
    assert render_route.supported_modes == ("real_observe",)
    assert render_route.may_move_robot is False
    assert render_route.requires_operator_confirmation is False
    assert render_route.requires_deadman is False
    render_slot = composition.station_point_grasp_render_registration
    assert render_slot is not None
    output_slot = render_slot.skill.io_contract.output_artifact_slots[0]
    assert output_slot.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert output_slot.allowed_schema_ids == (POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,)
    assert output_slot.allowed_media_types == (
        LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    )
    assert output_slot.allowed_encodings == (LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,)
    assert output_slot.allowed_compressions == (
        LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    )
    render_profile = render_route.allowed_parameter_profiles[0]

    def submit_render(parameters, *, sequence, suffix, station_observation=None):
        selected_observation = (
            observation if station_observation is None else station_observation
        )
        return adapter.submit_read_only_skill_request(
            AgentSkillRequest(
                client_id="rpent:test",
                request_id=f"request:station-point-grasp-render:{suffix}",
                idempotency_key=f"idempotency:station-point-grasp-render:{suffix}",
                client_sequence=sequence,
                runtime_session_id=manifest.runtime_session_id,
                skill_route_manifest_hash=manifest.semantic_content_hash,
                expected_station_revision=int(selected_observation["station_revision"]),
                skill_id=render_route.skill_id,
                skill_version=render_route.skill_version,
                skill_definition_hash=render_route.skill_definition_hash,
                route_id=render_route.route_id,
                route_version=render_route.route_version,
                route_hash=render_route.semantic_content_hash,
                deployment_binding_hash=(
                    render_entry.deployment_binding.deployment_binding_hash
                ),
                parameter_profile_id=render_profile.parameter_profile_id,
                parameter_profile_hash=render_profile.parameter_profile_hash,
                parameters=parameters,
                observation_snapshot_id=str(selected_observation["snapshot_id"]),
                requested_mode="real_observe",
                requested_timeout_ms=20_000,
            ).to_wire()
        )

    render_parameters = {
        "point_grasp_operation_id": operation_id,
        "candidate_ordinal": 0,
        "view_role": "main",
    }
    rendered = submit_render(render_parameters, sequence=2, suffix="success")
    assert rendered["accepted"] is True
    render_operation_id = rendered["operation"]["operation_id"]
    render_executor = composition.session.station_point_grasp_render_executor
    assert render_executor is not None
    assert render_executor.wait(render_operation_id, timeout_sec=2.0)
    render_operation = composition.session.service.operation_store.get(
        render_operation_id
    )
    assert render_operation is not None
    assert render_operation.status == "completed"
    result = render_operation.execution_result
    assert result is not None
    assert result["point_grasp_operation_id"] == operation_id
    assert result["point_grasp_request_hash"] == operation.request_hash
    assert (
        result["source_capture_group_id"]
        == operation.execution_result["capture_group_id"]
    )
    assert result["candidate_ordinal"] == 0
    assert result["view_role"] == "main"
    assert result["view_source_id"] == "front"
    assert result["view_station_revision"] == result["source_station_revision"]
    assert result["depth_occlusion_tested"] is False
    assert result["downstream_control_effects"] == "forbidden"
    reference_from_tcp = np.asarray(result["T_reference_from_tool_tcp"]).reshape(4, 4)
    reference_from_camera = np.asarray(result["T_reference_from_camera"]).reshape(4, 4)
    camera_from_tcp = np.asarray(result["T_camera_from_tool_tcp"]).reshape(4, 4)
    np.testing.assert_allclose(
        np.linalg.inv(reference_from_camera) @ reference_from_tcp,
        camera_from_tcp,
    )
    assert (
        result["frame_binding_hash"] == composition.frame_binding.semantic_content_hash
    )
    assert result["frame_graph_hash"] == (
        composition.frame_binding.frame_graph.semantic_content_hash
    )
    assert result["source_frame_id"] == composition.frame_binding.source_frame_id
    assert result["reference_frame_id"] == (
        composition.frame_binding.reference_frame_id
    )
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame == {
        "ordinal": 0,
        "role": "main",
        "width": 4,
        "height": 3,
        "media_type": "image/jpeg",
        "media_content_hash": render_operation.output_artifact_bindings[
            0
        ].artifact.sha256,
    }
    artifact_binding = render_operation.output_artifact_bindings[0]
    assert artifact_binding.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert artifact_binding.ordinal == 0
    assert artifact_binding.artifact.schema_id == POINT_GRASP_ANNOTATED_SCHEMA_ID_V1
    assert artifact_binding.artifact.media_type == "image/jpeg"
    assert artifact_binding.artifact.encoding == "jpeg.rgb8.v1"
    assert artifact_binding.artifact.compression == "none"
    assert source.read_count == 2  # proposal generation and rendered view
    inconsistent = dict(result)
    inconsistent_camera_from_tcp = list(result["T_camera_from_tool_tcp"])
    inconsistent_camera_from_tcp[3] += 0.01
    inconsistent["T_camera_from_tool_tcp"] = inconsistent_camera_from_tcp
    with pytest.raises(ValueError, match="transform chain is inconsistent"):
        render_slot.payload_validator.validate_execution_result(inconsistent)

    unknown = submit_render(
        render_parameters | {"point_grasp_operation_id": "operation:unknown"},
        sequence=3,
        suffix="unknown",
    )
    assert unknown["accepted"] is True
    unknown_id = unknown["operation"]["operation_id"]
    assert render_executor.wait(unknown_id, timeout_sec=2.0)
    unknown_operation = composition.session.service.operation_store.get(unknown_id)
    assert unknown_operation is not None
    assert unknown_operation.status == "failed"
    assert unknown_operation.reason_code == "passive_read_executor_exception:ValueError"

    missing_ordinal = submit_render(
        render_parameters | {"candidate_ordinal": 1},
        sequence=4,
        suffix="missing-ordinal",
    )
    assert missing_ordinal["accepted"] is True
    missing_ordinal_id = missing_ordinal["operation"]["operation_id"]
    assert render_executor.wait(missing_ordinal_id, timeout_sec=2.0)
    missing_ordinal_operation = composition.session.service.operation_store.get(
        missing_ordinal_id
    )
    assert missing_ordinal_operation is not None
    assert missing_ordinal_operation.status == "failed"

    injected_matrix = submit_render(
        render_parameters | {"T_reference_from_tool_tcp": list(np.eye(4).reshape(-1))},
        sequence=5,
        suffix="injected-matrix",
    )
    assert injected_matrix["accepted"] is False
    assert injected_matrix["reason_code"] == "parameter_payload_rejected"

    unavailable_role = submit_render(
        render_parameters | {"view_role": "wrist_left"},
        sequence=6,
        suffix="unavailable-role",
    )
    assert unavailable_role["accepted"] is False
    assert unavailable_role["reason_code"] == "parameter_payload_rejected"

    composition.session.observation_producer.revision_tracker.observe(
        {"forced_test_control_fingerprint": "changed"}
    )
    changed_observation = adapter.get_observation_snapshot()
    assert changed_observation["station_revision"] > observation["station_revision"]
    stale_source = submit_render(
        render_parameters,
        sequence=7,
        suffix="stale-source",
        station_observation=changed_observation,
    )
    assert stale_source["accepted"] is True
    stale_source_id = stale_source["operation"]["operation_id"]
    assert render_executor.wait(stale_source_id, timeout_sec=2.0)
    stale_source_operation = composition.session.service.operation_store.get(
        stale_source_id
    )
    assert stale_source_operation is not None
    assert stale_source_operation.status == "failed"
    assert stale_source_operation.output_artifact_bindings == ()
    # Planning and the repeated render requests above share the one application
    # camera resource transaction; Skill calls must never restart it.
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 0
    composition.application.close()
    assert planner.starts == 1
    assert planner.closes == 1
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 1
    assert events[-4:] == [
        "gateway.close",
        "planner.close",
        "localizer.close",
        "cameras.close",
    ]


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("rigid_safe_index", (None, 29))
def test_rigid_endpoint_priority_covers_low_scores_but_never_vetoes_ik(side, rigid_safe_index):
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (1., 0., 0.)
    adapter.clock = lambda: 10.
    adapter.monotonic = lambda: 1.
    adapter._solver_pose = lambda p: p
    candidates = []
    for i in range(30):
        pose = np.diag((-1., 1., -1., 1.))
        pose[0, 3] = i
        candidates.append(SimpleNamespace(base_from_tcp=tuple(pose.reshape(-1)), score=1.-i*.02))
    queried, probed, assessed = [], [], []
    def index(kw):
        assert kw["side"] == side
        return int(kw["target_tool_tcp_world"][0, 3])
    def rigid(**kw):
        i = index(kw)
        queried.append(i)
        return SimpleNamespace(collision_check_available=True, safe=i == rigid_safe_index,
            minimum_distance_m=.04 if i == rigid_safe_index else -.03, metadata={})
    def probe(**kw):
        i = index(kw)
        probed.append(i)
        return SimpleNamespace(classification="clearance_safe", elapsed_ms=1., reason_code=None,
            best_minimum_distance_m=.04, attempted_seed_count=1)
    def assess(**kw):
        i = index(kw)
        assessed.append(i)
        return SimpleNamespace(feasible=True, selected_pregrasp_distance_m=.06, elapsed_ms=1.,
            reason_code=None, joint_travel_rad=(.01,)*6, attempts=())
    adapter.planner = SimpleNamespace(check_grasp_rigid_endpoint_clearance=rigid,
        probe_grasp_endpoint_clearance=probe, assess_local_grasp_chain=assess)
    result = adapter.select_candidates(tuple(candidates), (0., 0., 0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert set(queried) == set(range(30))
    assert len(probed) == 24
    expected = 0 if rigid_safe_index is None else rigid_safe_index
    assert probed[0] == expected and assessed == [expected]
    assert result.selected_indices == (expected,)
    assert result.diagnostics["arm_shortlist_rigid_endpoint_scope"] == "advisory_only_not_ik_or_path_admission"


@pytest.mark.parametrize("side", ("left", "right"))
def test_shortlist_exposes_only_the_selected_endpoint_hint(side):
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (1., 0., 0.)
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    candidates = []
    for i in range(2):
        pose = np.diag([-1., 1., -1., 1.])
        pose[0, 3] = i * .01
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.9 - i * .1,
            symmetry_parent_index=None if i == 0 else 0,
            orientation_variant_axis=None if i == 0 else "z", orientation_variant_angle_deg=0. if i == 0 else 180.))

    def assess(**kwargs):
        index = round(kwargs["target_tool_tcp_world"][0, 3] * 100)
        return SimpleNamespace(feasible=True, selected_pregrasp_distance_m=.06,
            elapsed_ms=1., reason_code=None, attempts=(), joint_travel_rad=(1. - index * .5,) * 6,
            terminal_qpos=(float(index),) * 6)

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    baseline = adapter.select_candidates(tuple(candidates), (0., 0., 0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    hints = {}
    selection = adapter.select_candidates(tuple(candidates), (0., 0., 0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}),
        measurement_reference_time=10., endpoint_seeds=hints)
    assert selection.selected_indices == baseline.selected_indices
    assert hints == {tuple(candidates[i].base_from_tcp.reshape(-1)): (float(i),) * 6
        for i in baseline.selected_indices}


def test_provider_passes_hints_only_within_one_request_and_times_annotation(caplog):
    from deployments.perception.sam2_graspgenx_point_grasp import (
        GraspGenXCandidateSelection,
    )
    from teleop_runtime.skill_deploy.motion_port import RealAgentMotionMeasuredQposCut

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    _, _, planner, registration, provider = _provider_stack(
        backend_id="graspgenx", arm_reachability_evaluator=adapter)
    adapter.profile = registration.profile.arm_reachability_profile
    adapter.candidate_reference_frame_id = registration.profile.reference_frame_id
    adapter.clock = lambda: 10.
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    cut = RealAgentMotionMeasuredQposCut(revision=4, measured_at=9.95, available_at=9.96,
        qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, source_evidence_refs=("test-cut",))
    adapter.motion_port = SimpleNamespace(latest_measured_qpos=lambda: cut)
    mappings, received = [], []

    def select(candidates, point, maximum, *, endpoint_seeds, **_):
        mappings.append(endpoint_seeds)
        if len(mappings) == 1:
            for candidate in candidates:
                endpoint_seeds[tuple(np.asarray(candidate.base_from_tcp).reshape(-1))] = (.1,) * 6
        return GraspGenXCandidateSelection(tuple(range(len(candidates))), {})

    adapter.select_candidates = select

    def annotate(**kwargs):
        received.append(kwargs.get("initial_seed_qpos"))
        return SimpleNamespace(to_dict=lambda: {"classification": "standard_reachable",
            "position_residual_m": .001, "orientation_residual_deg": .5,
            "solve_cycles": 1, "elapsed_ms": 1., "termination_reason": "standard_tolerance_met"})

    adapter.planner = SimpleNamespace(assess_endpoint_ik=annotate)
    original_plan = planner.plan

    def plan(*args, candidate_selector, **kwargs):
        result = original_plan(*args, **kwargs)
        candidate_selector(result.candidates, result.target_point_base_m, kwargs["max_candidates"])
        return result

    planner.plan = plan
    with caplog.at_level(logging.INFO):
        first = provider.plan_request(_request(include_arm_reachability=True))
        first_count = len(received)
        second = provider.plan_request(_request(include_arm_reachability=True))
    assert mappings[0] is not mappings[1]
    assert mappings[0] and not mappings[1]
    assert all(hint is not None for hint in received[:first_count])
    assert all(hint is None for hint in received[first_count:])
    assert first.execution_result["candidates"] == second.execution_result["candidates"]
    registration.payload_validator.validate_execution_result(first.execution_result)
    events = [json.loads(record.message.split("Point-grasp provider timing: ")[1])
        for record in caplog.records if record.message.startswith("Point-grasp provider timing: ")]
    assert len(events) == 2
    assert all(event["status"] == "completed" for event in events)
    assert all("endpoint_annotation" in event["timing_ms"] for event in events)
    assert all(event["timing_ms"]["provider_total"] >= event["timing_ms"]["proposal_planning"] for event in events)


def test_failed_point_request_keeps_original_error_and_records_its_phase(caplog):
    _, _, planner, _, provider = _provider_stack(backend_id="graspgenx")

    def fail(*_, **__):
        raise RuntimeError("original_model_failure")

    planner.plan = fail
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="original_model_failure"):
        provider.plan_request(_request())
    event = next(json.loads(r.message.split("Point-grasp provider timing: ")[1])
        for r in caplog.records if r.message.startswith("Point-grasp provider timing: "))
    assert event["status"] == "failed"
    assert event["timing_ms"]["proposal_planning"] >= 0.
    assert "result_encoding" not in event["timing_ms"]


def test_provider_records_window_factory_failure_without_replacing_error(caplog):
    _, _, _, _, provider = _provider_stack(backend_id="graspgenx")

    def fail():
        raise RuntimeError("original_window_failure")

    provider.non_motion_read_window = fail
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="original_window_failure"):
        provider.plan_request(_request())
    event = next(json.loads(record.message.split("Point-grasp provider timing: ")[1])
        for record in caplog.records if record.message.startswith("Point-grasp provider timing: "))
    assert event["status"] == "failed"
    assert set(event["timing_ms"]) == {"provider_total"}


@pytest.mark.parametrize("phase", ("capture", "encoding"))
def test_provider_records_entered_failed_phase_without_replacing_error(monkeypatch, caplog, phase):
    from agent_gateway.profiles.real import point_grasp_gateway as gateway

    _, _, _, _, provider = _provider_stack(backend_id="graspgenx")

    def fail(*_args, **_kwargs):
        raise RuntimeError("original_phase_failure")

    if phase == "capture":
        monkeypatch.setattr(provider.perception_provider, "capture_aligned_rgbd", fail)
    else:
        monkeypatch.setattr(gateway, "encode_grasp_rgbd_snapshot", fail)
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="original_phase_failure"):
        provider.plan_request(_request())
    event = next(json.loads(record.message.split("Point-grasp provider timing: ")[1])
        for record in caplog.records if record.message.startswith("Point-grasp provider timing: "))
    assert event["status"] == "failed"
    key = "capture_and_preparation" if phase == "capture" else "result_encoding"
    assert event["timing_ms"][key] >= 0.
    assert ("proposal_planning" in event["timing_ms"]) == (phase == "encoding")


def test_producer_records_failed_publication_without_success_receipt(monkeypatch, caplog, tmp_path):
    from agent_gateway.profiles.real import point_grasp_gateway as gateway
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore

    _, _, _, registration, provider = _provider_stack(backend_id="graspgenx")
    store = RecoverableAgentArtifactStore(root=tmp_path/"artifacts",
        quota_bytes=32*1024*1024, max_artifact_bytes=16*1024*1024, max_chunk_bytes=1024*1024)
    original = gateway.commit_artifact_bytes

    def commit(**kwargs):
        if kwargs["producer_id"] == "agent_gateway.provider.station_bound_point_grasp":
            raise RuntimeError("original_publication_failure")
        return original(**kwargs)

    monkeypatch.setattr(gateway, "commit_artifact_bytes", commit)
    executor = gateway.build_station_bound_point_grasp_executor(
        registration=registration, provider=provider, artifact_store=store,
        runtime_session_id="station-point-grasp:test", current_station_revision=lambda: 7,
        max_retained_operations=4, clock=lambda: 10.)
    request = _request()
    request.semantic_content_hash = "a"*64
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="original_publication_failure"):
        executor._read(request, SimpleNamespace(operation_id="publication:failure"))
    event = next(json.loads(record.message.split("Point-grasp producer timing: ")[1])
        for record in caplog.records if record.message.startswith("Point-grasp producer timing: "))
    assert event["status"] == "failed"
    assert event["timing_ms"]["artifact_publication"] >= 0.


def test_uninspectable_selector_does_not_replace_original_model_failure(monkeypatch):
    from teleop_runtime.skill_deploy.motion_port import RealAgentMotionMeasuredQposCut

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    _, _, planner, registration, provider = _provider_stack(
        backend_id="graspgenx", arm_reachability_evaluator=adapter)
    cut = RealAgentMotionMeasuredQposCut(revision=4, measured_at=9.95, available_at=9.96,
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")}, source_evidence_refs=("test-cut",))
    adapter.motion_port = SimpleNamespace(latest_measured_qpos=lambda: cut)

    def select(*_args, **_kwargs):
        pytest.fail("The failing model must not enter candidate selection")

    select.__signature__ = "not-a-signature"
    adapter.select_candidates = select

    def fail(*_args, **_kwargs):
        raise RuntimeError("original_model_failure")

    monkeypatch.setattr(planner, "plan", fail)
    with pytest.raises(RuntimeError, match="original_model_failure"):
        provider.plan_request(_request(include_arm_reachability=True))


import io

@pytest.mark.parametrize("visible_feasible", [True, False])
def test_shortlist_uses_capture_bound_scene_as_advisory_without_vetoing_full_search(visible_feasible):
    from robot_motion.grasp_visible_scene import (
        ParallelJawSolidGeometry,
        VisibleGraspScene,
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side="right")
    adapter.arm_base_origin_candidate_m = (-1., 0., 0.)
    adapter.monotonic = lambda: 0.
    transform = np.eye(4)
    transform[:3, 3] = [1., 2., 3.]
    adapter._solver_pose = lambda value: transform @ value
    scene = VisibleGraspScene(points=np.array([[0., 0., .1], [.2, .1, .3]]),
        target_mask=np.array([True, False]), geometry=ParallelJawSolidGeometry(-.04, .05, .01, .02, .07))
    seen = []

    def assess(**kwargs):
        seen.append(kwargs)
        visible = kwargs.get("visible_scene")
        return SimpleNamespace(feasible=True if visible is None else visible_feasible,
            selected_pregrasp_distance_m=.06, elapsed_ms=1.,
            reason_code=None if visible is None or visible_feasible else "visible_gripper_path_collision",
            attempts=(), joint_travel_rad=(.1,) * 6, estimated_execution_time_sec=.4)

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    candidate = SimpleNamespace(base_from_tcp=np.diag([1., -1., -1., 1.]), score=.8)
    result = adapter.select_candidates((candidate,), (0.,0.,0.), 1, visible_scene=scene,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert result.selected_indices == (0,)  # A preliminary miss is not an execution permit or veto.
    assert len(seen) == 2 and "visible_scene" not in seen[0]
    np.testing.assert_array_equal(seen[1]["visible_scene"].points, scene.points+[1.,2.,3.])
    np.testing.assert_array_equal(seen[1]["visible_scene"].target_mask, scene.target_mask)
    np.testing.assert_array_equal(scene.points, [[0.,0.,.1],[.2,.1,.3]])
    assert result.diagnostics["arm_shortlist_visible_scene_check_count"] == 1
    assert result.diagnostics["arm_shortlist_visible_scene_feasible_count"] == int(visible_feasible)


@pytest.mark.parametrize("side", ("left", "right"))
def test_shortlist_searches_beyond_static_page_and_ranks_whole_arm_time(side):
    from robot_motion.grasp_visible_scene import (
        ParallelJawSolidGeometry,
        VisibleGraspScene,
    )

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (-1., 0., 0.)
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda value: value
    scene = VisibleGraspScene(points=np.array([[0., 0., .1]]),
        target_mask=np.array([False]), geometry=ParallelJawSolidGeometry(-.04, .05, .01, .02, .07))
    candidates = []
    for index in range(7):
        pose = np.diag([1., -1., -1., 1.])
        pose[0, 3] = index * .001
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.9-index*.01))
    checked = []

    def assess(**kwargs):
        index = round(kwargs["target_tool_tcp_world"][0, 3] * 1000)
        visible = kwargs.get("visible_scene") is not None
        checked.append((index, visible))
        # The first five are statically feasible, but observed local checking
        # fails. Keep them as reserve, and continue to the later compact wrist.
        feasible = not visible or index >= 5
        return SimpleNamespace(feasible=feasible, selected_pregrasp_distance_m=.06,
            elapsed_ms=1., reason_code=None if feasible else "visible_gripper_path_collision",
            attempts=(), joint_travel_rad=(.1,)*3 + ((3.,)*3 if index < 6 else (.2,)*3),
            estimated_execution_time_sec=3. if index < 6 else .3)

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(candidates, (0., 0., 0.), 5, visible_scene=scene,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert result.selected_indices[:2] == (6, 5)
    assert len(result.selected_indices) == 5  # Three static-feasible reserves remain.
    assert checked[:5] == [(index, False) for index in range(5)]
    assert sorted(checked) == [(index, visible) for index in range(7) for visible in (False, True)]
    assert result.diagnostics["arm_shortlist_visible_scene_feasible_count"] == 2


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("initial_visible_feasible", [False, True])
def test_sparse_static_pool_reaches_observed_rescue_without_unbounded_budget(side, initial_visible_feasible):
    from robot_motion.grasp_visible_scene import (
        ParallelJawSolidGeometry,
        VisibleGraspScene,
    )

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (-1., 0., 0.)
    now = [0.]
    adapter.monotonic = lambda: now[0]
    adapter._solver_pose = lambda value: value
    scene = VisibleGraspScene(points=np.array([[0., 0., .1]]),
        target_mask=np.array([False]), geometry=ParallelJawSolidGeometry(-.04, .05, .01, .02, .07))
    candidates = []
    for index in range(20):
        pose = np.diag([1., -1., -1., 1.])
        pose[0, 3] = index*.001
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.9-index*.01))
    checked = []

    def assess(**kwargs):
        index = round(kwargs["target_tool_tcp_world"][0, 3]*1000)
        visible = kwargs.get("visible_scene") is not None
        checked.append((index, visible))
        now[0] += min(.3 if visible else .7, kwargs["time_budget_sec"])
        feasible = index == 14 or (index == 0 and (not visible or initial_visible_feasible))
        return SimpleNamespace(feasible=feasible, selected_pregrasp_distance_m=.06,
            elapsed_ms=1., reason_code=None if feasible else "visible_gripper_path_collision",
            attempts=(), joint_travel_rad=(.1,)*6, estimated_execution_time_sec=.3)

    adapter.planner = SimpleNamespace(assess_local_grasp_chain=assess)
    result = adapter.select_candidates(candidates, (0., 0., 0.), 1, visible_scene=scene,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side={}), measurement_reference_time=10.)
    assert result.selected_indices == ((0,) if initial_visible_feasible else (14,))
    assert (0, True) in checked
    assert ((14, True) in checked) is not initial_visible_feasible
    assert result.diagnostics["arm_shortlist_rescue_activated"] is not initial_visible_feasible
    assert now[0] <= result.diagnostics["arm_shortlist_budget_sec"] == (
        8. if initial_visible_feasible else 16.)


@pytest.mark.parametrize("side", ("left", "right"))
def test_shortlist_native_reach_preserves_independent_pair_and_unknown_geometry(side):
    from robot_motion.rm65.dual_arm_target_solver import DualArmToolTCPReachCheck
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = SimpleNamespace(max_measured_qpos_age_ms=200., arm_side=side)
    adapter.arm_base_origin_candidate_m = (1., 0., 0.)
    adapter.clock = lambda: 10.
    adapter.monotonic = lambda: 0.
    adapter._solver_pose = lambda pose: pose.copy()
    pair = {name: np.zeros(6) for name in ("left", "right")}
    candidates = []
    for index in range(3):
        pose = np.diag([-1., 1., -1., 1.])
        pose[0, 3] = index * .01
        candidates.append(SimpleNamespace(base_from_tcp=pose, score=.9-index*.1,
            symmetry_parent_index=0 if index == 1 else None,
            orientation_variant_axis="z" if index == 1 else None,
            orientation_variant_angle_deg=180. if index == 1 else 0.))
    before = [c.base_from_tcp.copy() for c in candidates]
    checked, probed, assessed = [], [], []
    def ordinal(kwargs):
        assert kwargs["side"] == side and kwargs["qpos_by_side"] is pair
        return round(kwargs["target_tool_tcp_world"][0, 3] * 100)
    def geometric(**kwargs):
        index = ordinal(kwargs)
        checked.append(index)
        return (DualArmToolTCPReachCheck(.6 if index == 0 else .4, .466, .008)
                if index < 2 else None)
    def probe(**kwargs):
        probed.append(ordinal(kwargs))
        return SimpleNamespace(classification="clearance_safe", elapsed_ms=1., reason_code=None,
            best_minimum_distance_m=.04, attempted_seed_count=1)
    def assess(**kwargs):
        index = ordinal(kwargs)
        assessed.append(index)
        return SimpleNamespace(feasible=index == 1, selected_pregrasp_distance_m=.06,
            joint_travel_rad=(.1,) * 6, elapsed_ms=1.,
            reason_code=None if index == 1 else "endpoint_collision_blocked", attempts=())
    adapter.planner = SimpleNamespace(check_grasp_geometric_reach=geometric,
        probe_grasp_endpoint_clearance=probe, assess_local_grasp_chain=assess)
    result = adapter.select_candidates(tuple(candidates), (0., 0., 0.), 1,
        measured_qpos_cut=SimpleNamespace(measured_at=10., qpos_by_side=pair), measurement_reference_time=10.)
    assert checked == [0, 1, 2] and set(probed) == set(assessed) == {1, 2}
    assert result.selected_indices == (1,)
    assert result.diagnostics["arm_shortlist_geometric_checked_count"] == 2
    assert result.diagnostics["arm_shortlist_geometric_rejected_count"] == 1
    assert result.diagnostics["arm_shortlist_geometric_retained_count"] == 2
    assert result.diagnostics["arm_shortlist_side_admissible_count"] == 3
    examples = json.loads(result.diagnostics["arm_shortlist_geometric_rejection_examples_json"])
    assert examples[0]["index"] == 0 and examples[0]["outside"]
    assert json.loads(result.diagnostics["arm_shortlist_rejections_json"])["endpoint_geometric_reach_exceeded"] == 1
    for actual, expected in zip(candidates, before):
        np.testing.assert_array_equal(actual.base_from_tcp, expected)


@pytest.mark.parametrize("prepare_fails", [False, True])
def test_provider_prepares_static_filter_before_model_and_propagates_start_failure(prepare_fails):
    events = []

    def prepare():
        events.append("static.prepare")
        if prepare_fails:
            raise RuntimeError("static CAD unavailable")

    def filter_capture(**kwargs):
        pytest.fail("startup must not read or filter a captured frame")

    filter_capture.prepare = prepare
    _, _, planner, _, provider = _provider_stack(
        captured_robot_observation_filter=filter_capture)
    planner.start = lambda: events.append("model.start")
    if prepare_fails:
        with pytest.raises(RuntimeError, match="static CAD unavailable"):
            provider.start()
        assert events == ["static.prepare"]
        assert not provider.started
    else:
        provider.start()
        assert events == ["static.prepare", "model.start"]
        assert provider.started


def test_provider_passes_same_capture_scene_in_candidate_frame_with_controlled_tcp_geometry():
    from deployments.perception.sam2_graspgenx_point_grasp import (
        GraspGenXCandidateSelection,
    )
    from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry
    from teleop_runtime.skill_deploy.motion_port import RealAgentMotionMeasuredQposCut

    received = []

    def select(candidates, point, maximum, *, visible_scene, measured_qpos_cut, **_):
        received.append((visible_scene, measured_qpos_cut))
        return GraspGenXCandidateSelection(tuple(range(len(candidates))), {})

    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    _, _, planner, registration, provider = _provider_stack(
        backend_id="graspgenx", arm_reachability_evaluator=adapter)
    adapter.select_candidates = select
    adapter.profile = registration.profile.arm_reachability_profile
    adapter.candidate_reference_frame_id = registration.profile.reference_frame_id
    adapter._solver_pose = lambda value: value
    adapter.monotonic = lambda: 0.
    adapter.planner = SimpleNamespace(assess_endpoint_ik=lambda **_: SimpleNamespace(
        to_dict=lambda: {"classification": "standard_reachable", "position_residual_m": .001,
            "orientation_residual_deg": .5, "solve_cycles": 1, "elapsed_ms": 1.,
            "termination_reason": "standard_tolerance_met"}))
    cut = RealAgentMotionMeasuredQposCut(revision=4, measured_at=9.95, available_at=9.96,
        qpos_by_side={s: np.zeros(6) for s in ("left", "right")}, source_evidence_refs=("test-cut",))
    provider.motion_port = SimpleNamespace(latest_measured_qpos=lambda: cut)
    original_plan = planner.plan
    model_geometry = ParallelJawSolidGeometry(-.01, .02, .01, .02, .07)
    from robot_motion.grasp_visible_scene import VisibleGraspScene

    original_scene = VisibleGraspScene(points=np.array([[.1, .2, .3], [.2, .3, .4]]),
        target_mask=np.array([True, False]), geometry=model_geometry)

    # Exercise the actual provider's wrapper without depending on depth denoising.
    # Its inner factory is called through the single-view capture filter path.
    from unittest.mock import patch

    def plan(*args, visible_scene_factory, candidate_selector, **kwargs):
        with patch.object(VisibleGraspScene, "from_depth", return_value=original_scene):
            returned = visible_scene_factory(depth_m=args[1], target_mask=np.ones((3, 4), bool),
                intrinsics=kwargs["intrinsics"], geometry=model_geometry)
        result = original_plan(*args, **kwargs)
        candidate_selector(result.candidates, result.target_point_base_m, kwargs["max_candidates"])
        assert returned is original_scene
        return result

    from robot_motion.rm65.captured_robot_observations import (
        RM65CapturedRobotObservationFilterResult,
    )
    provider.captured_robot_observation_filter = lambda **kw: RM65CapturedRobotObservationFilterResult(
        depth_m=kw["depth_m"], removed_mask=np.zeros_like(kw["depth_m"], dtype=bool), evidence={})
    planner.plan = plan
    provider.plan_request(_request(include_arm_reachability=True))
    scene, seen_cut = received[0]
    assert seen_cut is cut
    assert scene.geometry is planner.solid_geometry and scene.geometry is not model_geometry
    np.testing.assert_allclose(scene.points,
        original_scene.points @ planner.base_from_camera[:3, :3].T + planner.base_from_camera[:3, 3])
    np.testing.assert_array_equal(scene.target_mask, original_scene.target_mask)
    for side in ("left", "right"):
        np.testing.assert_array_equal(scene.capture_qpos_by_side[side], cut.qpos_by_side[side])


@pytest.mark.parametrize("retention_mode", ("ok", "store_rejected", "oversized", "object_array"))
def test_known_grasp_failure_projects_stage_and_keeps_mask_pool_and_scene_for_debug(retention_mode):
    from agent_gateway.runtime.passive_read_executor import PassiveReadError
    from deployments.perception.point_grasp_failure import PointGraspPlanningError
    source, _, planner, _, provider = _provider_stack(backend_id="graspgenx")
    target_mask = np.zeros((3, 4), dtype=bool)
    target_mask[1, 1] = True
    scene_points = np.array([[1., 2., 3.], [2., 3., 4.]])
    summary = {"arm_shortlist_local_chain_assessment_count": 31,
               "arm_shortlist_local_chain_feasible_count": 0,
               "arm_shortlist_rejections_json": '{"approach_ik_search_exhausted":31}',
               "arm_shortlist_attempts_json": "x" * 10000}
    def fail(*_args, **_kwargs):
        arrays = {
            "target_mask": target_mask, "raw_grasps_reference": np.eye(4)[None],
            "raw_scores": np.ones(1), "scene_points_camera": scene_points,
            "scene_target_mask": np.array([True, False])}
        if retention_mode == "oversized":
            arrays["oversized"] = np.zeros(16 * 1024 * 1024 + 1, dtype=np.uint8)
        if retention_mode == "object_array":
            arrays["invalid"] = np.array([object()])
        raise PointGraspPlanningError(summary, replay_arrays=arrays)
    planner.plan = fail
    retained = []
    def save(data):
        if retention_mode == "store_rejected":
            raise OSError("fixture store is full")
        retained.append(data)
        return {"sha256": "fixture-content-hash"}
    request = _request()
    request.request_id = "fixture-request"
    with pytest.raises(PassiveReadError) as raised:
        provider.plan_request(request, failure_snapshot_sink=save)
    assert source.read_count == 1
    assert raised.value.reason_code == "point_grasp_no_local_grasp_chain_candidate"
    assert raised.value.diagnostic["counts"]["local_chain_assessed"] == 31
    if retention_mode != "ok":
        assert retained == []
        assert "debug_evidence" not in raised.value.diagnostic
        failure = raised.value.diagnostic["debug_evidence_failure"]
        assert failure["exception_type"] == ("OSError" if retention_mode == "store_rejected" else "ValueError")
        return
    assert len(retained) == 1
    assert raised.value.diagnostic["debug_evidence"]["sha256"] == "fixture-content-hash"
    with np.load(io.BytesIO(retained[0]), allow_pickle=False) as data:
        np.testing.assert_array_equal(data["target_mask"], target_mask)
        np.testing.assert_array_equal(data["scene_points_camera"], scene_points)
        metadata = json.loads(data["metadata"].tobytes())
    assert metadata["filter_summary"] == summary
    assert metadata["scope"] == "rejected_proposals_not_an_executable_plan"
    assert metadata["request_id"] == request.request_id


def test_failed_evidence_retention_is_timed_separately_from_model_planning(monkeypatch, caplog):
    import json
    import logging
    from agent_gateway.runtime.passive_read_executor import PassiveReadError
    from deployments.perception.point_grasp_failure import PointGraspPlanningError

    now = [0.]
    monkeypatch.setattr("agent_gateway.profiles.real.point_grasp_gateway.time.perf_counter", lambda: now[0])
    _, _, planner, _, provider = _provider_stack(backend_id="graspgenx")

    def fail(*args, **kwargs):
        now[0] += 2.
        raise PointGraspPlanningError({"detected": 1})

    def retain(data):
        now[0] += 3.
        return {"sha256": "retained:test"}

    planner.plan = fail
    request = _request()
    request.request_id = "timing:test"
    caplog.set_level(logging.INFO, logger="agent_gateway.profiles.real.point_grasp_gateway")
    with pytest.raises(PassiveReadError):
        provider.plan_request(request, failure_snapshot_sink=retain)
    record = next(json.loads(row.message.split("Point-grasp provider timing: ")[1])
        for row in caplog.records if row.message.startswith("Point-grasp provider timing: "))
    assert record["timing_ms"]["proposal_planning"] == 2000.
    assert record["timing_ms"]["failure_evidence_retention"] == 3000.
    assert record["timing_ms"]["provider_total"] == 5000.


@pytest.mark.parametrize("options", [
    {"refine_grasp_centers": "true"},
    {"refine_grasp_centers": True},
    {"refine_grasp_centers": True, "annotate_result": False},
    {"refine_grasp_centers": True, "selection_factory": lambda *_: None},
    {"refine_grasp_axial_depth": "true"},
    {"refine_grasp_axial_depth": True},
    {"refine_grasp_axial_depth": True, "annotate_result": False},
    {"refine_grasp_axial_depth": True, "selection_factory": lambda *_: None},
])
def test_center_refinement_cannot_leak_into_public_candidate_review(options):
    _, _, _, _, provider = _provider_stack(backend_id="graspgenx")
    with pytest.raises((TypeError, ValueError), match="refinement|refine_grasp"):
        provider.plan_request(_request(), **options)


def test_public_candidate_review_preserves_default_model_call():
    _, _, planner, registration, provider = _provider_stack(backend_id="graspgenx")
    original = planner.plan
    received = []

    def spy(*args, **kwargs):
        received.append(kwargs.copy())
        return original(*args, **kwargs)

    planner.plan = spy
    prepared = provider.plan_request(_request())
    assert len(received) == 1 and "refine_grasp_centers" not in received[0]
    assert "refine_grasp_axial_depth" not in received[0]
    registration.payload_validator.validate_execution_result(prepared.execution_result)
