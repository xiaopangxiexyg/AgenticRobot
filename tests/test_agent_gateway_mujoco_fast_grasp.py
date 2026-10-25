"""Public native chain tests; fixed proposals are not fresh neural inference."""

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_gateway_graspgenx_point_grasp import _model_spec, _PlannerClient
from test_agent_gateway_mujoco_grasp_approach import _retained_world_grasp

from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentArtifactStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from dataset_contracts.grasp_multiview_snapshot import decode_grasp_multiview_snapshot
from dataset_contracts.grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot
from dataset_contracts.perception_artifacts import (
    PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1,
)
from deployments.perception.point_grasp_failure import PointGraspPlanningError
from deployments.dmtac_graspgenx_alignment_ids import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    Sam2GraspGenXPointGraspPlanner,
)
from deployments.perception.sam2_vgn_point_grasp import PointGraspCandidate
from project_paths import resolve_path
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

CLIENT = "test:native-fast-grasp"
TASK = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")


@pytest.mark.integration
@pytest.mark.parametrize("profile_id", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS)
def test_native_factory_preserves_the_explicit_alignment_choice(tmp_path, profile_id):
    from deployments.perception.dmtac_grasp_overlay import reviewed_graspgenx_dmtac_solid_geometry

    session = _session(tmp_path, alignment_profile_id=profile_id)
    try:
        session.application.start_observation_ready()
        model = session.fast_grasp_planner
        assert model.alignment_profile_id == profile_id
        assert model.solid_geometry == reviewed_graspgenx_dmtac_solid_geometry(
            maximum_opening_m=model.max_gripper_width_m, alignment_profile_id=profile_id)
        assert not model.client.calls and not session.application.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("scores",[(.1,1.),(1.,1.)])
def test_native_fast_selector_prefers_deeper_support_at_equal_motion_cost(tmp_path, monkeypatch,scores):
    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        provider = session.fast_grasp_provider
        native = app.target.planned_tcp_motion
        # A test-only path result isolates preference ordering, not feasibility.
        plan = SimpleNamespace(reference_poses=((0., 0., 0., 0., 0., 0., 1.),),
            trajectory=SimpleNamespace(timestamps=np.array([0., 1.])))
        queried = []

        def query(goals, **kwargs):
            queried.append(goals)
            return plan

        monkeypatch.setattr(native, "plan_chain", query)
        select, _, box = provider._selector(parameters={"speed_scale": 1.},
            deadline=time.perf_counter() + 20., cancelled=lambda: False)
        pose = _retained_world_grasp()
        candidates = tuple(SimpleNamespace(base_from_tcp=pose, score=.8,
            visible_collision_free=True, capture_support_score=score,
            capture_height_support_score=(.1 if i == 0 and scores == (1.,1.) else 1.))
            for i,score in enumerate(scores))
        result = select(candidates, tuple(pose[:3, 3]), 1)
        assert result.selected_indices == (1,)
        assert [row["index"] for row in box["result"].attempts] == [1]
        assert len(queried) == 1 and app.held and not native.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("reject_child", [False, True])
def test_native_axial_child_and_parent_use_exact_pose_and_single_terminal_offset(tmp_path, monkeypatch, reject_child):
    from deployments.perception.grasp_first_path import GraspPathQueryRejected
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.transforms import pose_to_xyz_quat_xyzw
    session = _session(tmp_path, terminal_offset_m=.01)
    try:
        app = session.application
        app.start_observation_ready()
        provider = session.fast_grasp_provider
        provider.monotonic = lambda: 0.
        profile = provider.contract.execution_contract.motion_profile
        # The historical literal fixture is rounded; use a proper rotation
        # before comparing an exact quaternion pose round trip.
        parent_pose = xyzquat_to_pose_matrix(pose_to_xyz_quat_xyzw(_retained_world_grasp()))
        child_pose = parent_pose.copy()
        child_pose[:3, 3] += parent_pose[:3, 2] * .01
        candidates = (SimpleNamespace(base_from_tcp=parent_pose, score=.8,
            visible_collision_free=True), SimpleNamespace(base_from_tcp=child_pose,
            score=.8, visible_collision_free=True, axial_parent_index=0))
        queries = []
        plan = SimpleNamespace(reference_poses=((0., 0., 0., 0., 0., 0., 1.),),
            trajectory=SimpleNamespace(timestamps=np.array([0., 1.])))
        def query(goals, **kwargs):
            queries.append((goals, kwargs["time_budget_sec"]))
            if reject_child and len(queries) == 2:
                raise GraspPathQueryRejected("safe_tcp_target_blocked")
            return plan
        monkeypatch.setattr(app.target.planned_tcp_motion, "plan_chain", query)
        def refine(original, goals, *, validate_plan, **kwargs):
            assert original is plan
            result = query(goals, **kwargs)
            validate_plan(result)
            return result
        monkeypatch.setattr(app.target.planned_tcp_motion, "refine_pending_chain", refine)
        select, _, box = provider._selector(parameters={"speed_scale": 1.},
            deadline=20., cancelled=lambda: False)
        selected = select(candidates, parent_pose[:3, 3], 1)
        assert selected.selected_indices == ((0,) if reject_child else (1,))
        assert box["candidate"] is candidates[selected.selected_indices[0]]
        assert len(queries) == 2 and queries[0][1] == 8. and queries[1][1] == 3.
        for index, (goals, _) in enumerate(queries):
            expected = (parent_pose if index == 0 else child_pose).copy()
            expected[:3, 3] += expected[:3, 2] * .01
            # The shared public encoder intentionally uses float32. Compare
            # the exact encoded goal, not a double-precision inverse roundtrip.
            encoded = app.transform_tcp_pose_xyzquat(tuple(pose_to_xyz_quat_xyzw(expected)),
                source_frame_id=profile.reference_frame_id,
                target_frame_id=profile.execution_reference_frame_id)
            assert goals[-1].target_pose == encoded
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_proposal_preparation_stops_at_the_shared_deadline(tmp_path):
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
    session = _session(tmp_path)
    try:
        session.application.start_observation_ready()
        provider = session.fast_grasp_provider
        clock = [0.]
        provider.monotonic = lambda: clock[0]
        _, priority, _ = provider._selector(parameters={"speed_scale": 1.},
            deadline=1., cancelled=lambda: False)
        clock[0] = 1.
        with pytest.raises(MujocoTcpPlanRejected, match="safe_tcp_planning_budget_exhausted"):
            priority(_retained_world_grasp(), .8, (0., 0., 0.))
        assert not session.fast_grasp_planner.client.calls
        assert not session.application.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("failure", [None, "query", "validation", "cancel", "scene", "deadline", "foreign"])
def test_native_refinement_replaces_only_owned_unchanged_fully_validated_plan(monkeypatch, failure):
    from mujoco_plan_ownership_fixture import pending_chain_planner
    from deployments.perception.grasp_first_path import GraspPathQueryRejected
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

    # The transaction never queries geometry: keep the real planner and locks,
    # with a small scene cut instead of rebuilding a complete Gateway per failure case.
    native, original = pending_chain_planner()
    refined = replace(original, trajectory_hash="b" * 64)
    native._pending_plan = original if failure != "foreign" else refined
    clock, cancelled = [0.], [False]
    def preview(_goals, **kwargs):
        assert native.rig.scene_lock._is_owned()
        assert native.owns_pending_plan(original)
        np.testing.assert_array_equal(kwargs["preview_model_qpos"], original.source_model_qpos)
        if failure == "query":
            raise MujocoTcpPlanRejected("safe_tcp_target_blocked")
        if failure == "cancel":
            cancelled[0] = True
        if failure == "scene":
            native.scene.data.qpos[0] += .001
        if failure == "deadline":
            clock[0] = 4.
        return refined
    monkeypatch.setattr(native, "plan_chain", preview)
    def validate(plan):
        assert native.rig.scene_lock._is_owned()
        assert plan is refined and native.owns_pending_plan(original)
        if failure == "validation":
            raise GraspPathQueryRejected("grasp_fast_path_exceeds_execution_budget")
    def query():
        return native.refine_pending_chain(original, (), cancel_requested=lambda: cancelled[0],
            time_budget_sec=3., deadline_at=20., monotonic=lambda: clock[0], validate_plan=validate)
    if failure is None:
        assert query() is refined and native.owns_pending_plan(refined)
    else:
        reasons = {
            "query": "safe_tcp_target_blocked",
            "validation": "grasp_fast_path_exceeds_execution_budget",
            "cancel": "safe_tcp_planning_cancelled",
            "scene": "safe_tcp_scene_changed_during_planning",
            "deadline": "safe_tcp_planning_budget_exhausted",
            "foreign": "safe_tcp_plan_not_owned_by_runtime",
        }
        with pytest.raises((MujocoTcpPlanRejected, GraspPathQueryRejected)) as error:
            query()
        assert error.value.reason_code == reasons[failure]
        assert native.owns_pending_plan(refined if failure == "foreign" else original)
    assert not native.active


@pytest.mark.integration
@pytest.mark.parametrize("boundary", ["owner_reject", "cross_subdeadline", "cross_global_deadline"])
def test_native_refinement_shared_deadlines_preserve_pending_owner(tmp_path, monkeypatch, boundary):
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpMotionPlan
    from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        provider = session.fast_grasp_provider
        native = app.target.planned_tcp_motion
        clock = [0.]
        provider.monotonic = lambda: clock[0]
        original = MujocoTcpMotionPlan(
            path=None,
            trajectory=SimpleNamespace(timestamps=np.array([0., 1.])),
            source_model_qpos=native._model_qpos().copy(),
            reference_poses=((0., 0., 0., 0., 0., 0., 1.),),
            trajectory_hash="a" * 64,
        )
        refined = replace(original, trajectory_hash="b" * 64)
        pose = xyzquat_to_pose_matrix(pose_to_xyz_quat_xyzw(_retained_world_grasp()))
        child_pose = pose.copy()
        child_pose[:3, 3] += pose[:3, 2] * .005
        candidates = (
            SimpleNamespace(base_from_tcp=pose, score=.8, visible_collision_free=True,
                capture_support_score=.1, opposed_capture_support_score=.1,
                opposed_capture_support_observed_sides=2),
            SimpleNamespace(base_from_tcp=child_pose, score=.8, visible_collision_free=True,
                capture_support_score=.8, opposed_capture_support_score=.8,
                opposed_capture_support_observed_sides=2, axial_parent_index=0),
        )
        # Use the actual provider/shared refinement and native transaction.
        # Only the expensive path result and controlled clock are substituted.
        preview_finished = [False]
        queries, discarded = [], []

        def plan_chain(goals, **kwargs):
            queries.append(kwargs["time_budget_sec"])
            assert len(goals) == 3
            if "preview_model_qpos" not in kwargs:
                native._pending_plan = original
                return original
            assert native.owns_pending_plan(original)
            np.testing.assert_array_equal(kwargs["preview_model_qpos"], original.source_model_qpos)
            clock[0] = 3.0000001 if boundary == "owner_reject" else 2.9999999
            preview_finished[0] = True
            return refined

        model_qpos = native._model_qpos

        def captured_model_qpos():
            state = model_qpos()
            if preview_finished[0] and boundary != "owner_reject":
                # Cross during the final scene read. The owner must check
                # again before replacing the original pending handle.
                clock[0] = 20.0000001 if boundary == "cross_global_deadline" else 3.0000001
            return state

        discard = native.discard_pending_plan

        def discard_pending(plan):
            discarded.append(plan)
            discard(plan)

        monkeypatch.setattr(native, "plan_chain", plan_chain)
        monkeypatch.setattr(native, "_model_qpos", captured_model_qpos)
        monkeypatch.setattr(native, "discard_pending_plan", discard_pending)
        select, _, box = provider._selector(parameters={"speed_scale": 1.},
            deadline=20., cancelled=lambda: False)
        selected = select(candidates, pose[:3, 3], 1)
        result = box["result"]
        assert queries == [8., 3.]
        assert len(result.attempts) == 2
        if boundary in {"owner_reject", "cross_subdeadline"}:
            assert selected.selected_indices == (0,)
            assert result.trajectory is original and result.reason_code == "path_found"
            assert result.attempts[-1]["reason_code"] == "safe_tcp_planning_budget_exhausted"
            assert native.owns_pending_plan(original) and not native.owns_pending_plan(refined)
            assert not discarded
        else:
            assert selected.selected_indices == ()
            assert result.trajectory is None and box["candidate"] is None
            assert result.reason_code == "grasp_fast_search_budget_exhausted"
            assert result.attempts[-1]["reason_code"] == result.reason_code
            # This must come from the existing provider cleanup branch, not
            # from test cleanup or a simulated shared-layer rollback.
            assert len(discarded) == 1 and discarded[0] is original
            assert not native.owns_pending_plan(original) and not native.owns_pending_plan(refined)
        assert app.held and not native.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("candidate_order,distance_m,farther_support,expected", [
    # Native binding samples each decision; cheap selector tests retain order matrices.
    ((1, 0), .01, 1., 1), ((0, 1), .20, 1., 0), ((1, 0), .01, .20, 0),
])
def test_native_fast_selector_balances_motion_and_capture_support(
    tmp_path, monkeypatch, candidate_order, distance_m, farther_support, expected
):
    from robot_motion.relative_action import xyzquat_to_pose_matrix

    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        provider = session.fast_grasp_provider
        profile = provider.contract.execution_contract.motion_profile
        current = xyzquat_to_pose_matrix(app.transform_tcp_pose_xyzquat(
            app.capture_tcp_pose_by_component()["right_arm"],
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id))
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = [.25, -.1, .3]
        farther = pose.copy()
        direction = pose[:3, 3] - current[:3, 3]
        farther[:3, 3] += distance_m * direction / np.linalg.norm(direction)
        candidates = (
            SimpleNamespace(base_from_tcp=pose, score=.8,
                visible_collision_free=True, capture_support_score=.20),
            SimpleNamespace(base_from_tcp=farther, score=.8,
                visible_collision_free=True, capture_support_score=farther_support),
        )
        candidates = tuple(candidates[i] for i in candidate_order)
        plan = SimpleNamespace(reference_poses=((0., 0., 0., 0., 0., 0., 1.),),
            trajectory=SimpleNamespace(timestamps=np.array([0., 1.])))
        monkeypatch.setattr(app.target.planned_tcp_motion, "plan_chain", lambda *args, **kwargs: plan)
        select, priority, box = provider._selector(parameters={"speed_scale": 1.},
            deadline=time.perf_counter() + 20., cancelled=lambda: False)
        assert priority(farther, .8, pose[:3, 3]) - priority(pose, .8, pose[:3, 3]) == pytest.approx(distance_m / .5)
        result = select(candidates, tuple(pose[:3, 3]), 1)
        assert result.selected_indices == (candidate_order.index(expected),)
        assert [r["index"] for r in box["result"].attempts] == [candidate_order.index(expected)]
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize(
    "probe_timeout", [False, True],
    ids=["probe-success", "probe-timeout-normal-success"],
)
def test_native_selector_recovers_raw_pose_after_cheap_derived_endpoint_rejection(
    tmp_path, monkeypatch, probe_timeout
):
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        provider = session.fast_grasp_provider
        profile = provider.contract.execution_contract.motion_profile
        current = xyzquat_to_pose_matrix(app.transform_tcp_pose_xyzquat(
            app.capture_tcp_pose_by_component()['right_arm'],
            source_frame_id=profile.execution_reference_frame_id, target_frame_id=profile.reference_frame_id))
        pose = np.diag([1., -1., -1., 1.])
        pose[:3,3] = [.25, -.1, .3]
        direction = pose[:3,3] - current[:3,3]
        direction /= np.linalg.norm(direction)
        candidates = []
        for i,distance in enumerate((0., .2, .01)):
            target = pose.copy()
            target[:3,3] += distance * direction
            candidates.append(SimpleNamespace(base_from_tcp=tuple(target.ravel()), score=.8,
                capture_support_score=.7, visible_collision_free=True,
                pose_source='dmtac_local_orientation_variant_v1' if i == 0 else 'model_scored'))
        # Only the native return port is fake: this proves selector wiring and
        # shared budgets, not geometry or a successful physical grasp.
        budgets = []
        plan = SimpleNamespace(reference_poses=((0.,0.,0.,0.,0.,0.,1.),),
            trajectory=SimpleNamespace(timestamps=np.array([0.,1.])))
        def query(*args, **kwargs):
            budgets.append(kwargs['time_budget_sec'])
            if len(budgets) == 1:
                raise MujocoTcpPlanRejected(
                    'safe_tcp_endpoint_search_exhausted',
                    {'chain_stage_index': 0, 'reason': 'test:derived_endpoint',
                     'endpoint_attempts': ('seed-0',)},
                )
            if probe_timeout and len(budgets) == 2:
                raise MujocoTcpPlanRejected(
                    'safe_tcp_planning_budget_exhausted',
                    {'chain_stage_index': 1, 'reason': 'test:probe_timeout',
                     'endpoint_attempts': ('seed-0', 'seed-1')},
                )
            if probe_timeout and len(budgets) == 3:
                raise MujocoTcpPlanRejected(
                    'safe_tcp_target_blocked',
                    {'chain_stage_index': 2, 'reason': 'test:normal_target_blocked',
                     'endpoint_attempts': ('seed-0', 'seed-1', 'seed-2')},
                )
            return plan
        monkeypatch.setattr(app.target.planned_tcp_motion,'plan_chain',query)
        select,_,box = provider._selector(parameters={'speed_scale':1.},
            deadline=time.perf_counter()+20., cancelled=lambda:False)
        assert select(tuple(candidates),tuple(pose[:3,3]),1).selected_indices == (2,)
        attempts = box['result'].attempts
        details = box['query_details']
        assert budgets == ([8., 3., 8., 8.] if probe_timeout else [8., 3.])
        assert [row['index'] for row in attempts] == (
            [0, 2, 1, 2] if probe_timeout else [0, 2]
        )
        assert len(details) == len(attempts) == len(budgets)
        assert len({id(detail) for detail in details}) == len(details)
        assert details[0] == {
            'failed_stage': 0, 'lower_reason': 'test:derived_endpoint',
            'endpoint_attempt_count': 1,
        }
        assert attempts[1]['query_mode'] == 'original_model_neighbor'
        if probe_timeout:
            assert details[1] == {
                'failed_stage': 1, 'lower_reason': 'test:probe_timeout',
                'endpoint_attempt_count': 2,
            }
            assert details[2] == {
                'failed_stage': 2, 'lower_reason': 'test:normal_target_blocked',
                'endpoint_attempt_count': 3,
            }
            assert 'query_mode' not in attempts[-1]
        assert details[-1] == {}
        merged = [dict(row, **detail)
            for row, detail in zip(attempts, details, strict=True)]
        assert [row.get('lower_reason') for row in merged] == (
            ['test:derived_endpoint', 'test:probe_timeout',
             'test:normal_target_blocked', None]
            if probe_timeout else ['test:derived_endpoint', None]
        )
        assert merged[-1]['reason_code'] == 'path_found'
        assert all(key not in merged[-1] for key in (
            'failed_stage', 'lower_reason', 'endpoint_attempt_count'))
        assert app.held and not app.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


class _BlueMask:
    """Deterministic test segmentation; not SAM2 or localization evidence."""

    semantic_content_hash = "test:rgb-blue-mask"

    def __init__(self):
        self.starts = self.closes = 0

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1

    def segment_point(self, image, *, pixel_uv):
        rgb = np.asarray(image)
        mask = (rgb[:, :, 2] > 100) & (rgb[:, :, 2].astype(float) > rgb[:, :, 0] * 1.5)
        assert mask[
            pixel_uv[1], pixel_uv[0]
        ], "Test target must be visible in its retained image."
        return mask


def _session(tmp_path, *, trace=None, retained_selector=False, terminal_offset_m=0.01,
             pregrasp_distance_m=0.07, alignment_profile_id=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID):
    from mujoco_gateway_fixture import build_gateway_session
    (tmp_path / "artifacts").mkdir()
    client = _PlannerClient(np.empty((0, 4, 4)), np.empty(0))
    model = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_BlueMask(),
        client=client,
        alignment_profile_id=alignment_profile_id,
        enable_finger_exchange=True,
    )
    session = build_gateway_session(
        build_mujoco_agent_tcp_pose_gateway_session,
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path(TASK),
            realtime_worker=False,
            runtime_session_id=CLIENT,
            control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=(CLIENT,),
        trace_sink=None if trace is None else trace.append,
        artifact_storage=MujocoAgentArtifactStorageConfig(root=tmp_path / "artifacts"),
        enable_safe_tcp_pose=True,
        enable_grippers=True,
        enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube",
        fast_grasp_planner=model,
        fast_grasp_approach_config={
            "pregrasp_distance_m": pregrasp_distance_m,
            "near_distance_m": 0.025,
            "terminal_offset_m": terminal_offset_m,
            "terminal_speed_scale": 0.25,
        },
    )
    # Explicit axis-aligned top-grasp test proposals in the model frame.
    # This fixture exercises filtering/search/physics, not neural model quality.
    profile = (
        session.contract_bundle.fast_grasp_contract.execution_contract.motion_profile
    )
    transform = session.application.target_frame_profile.frame_graph.resolve(
        MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, profile.reference_frame_id
    ).matrix_target_source
    poses = []
    for height in (0.12, 0.13, 0.14):
        for yaw in (0.0, np.pi / 2):
            pose = np.diag([-1.0, 1.0, -1.0, 1.0])
            rotation = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )
            pose[:3, :3] = pose[:3, :3] @ rotation
            pose[:3, 3] = [-0.272, 0.0, height]
            poses.append(transform @ pose)
    client.grasps = np.asarray(poses)
    client.scores = np.linspace(0.8, 0.7, len(poses))
    if retained_selector:
        # Isolate public pairing and native execution with the same exact
        # retained TCP fixture as the chain-owner regression. This adapter is
        # NOT model inference or evidence of target capture/physical grasp.
        calls = []

        def plan(image, depth, **kwargs):
            calls.append("retained_proposal_test_port")
            candidates = []
            for exchange in (False, True):
                tcp = _retained_world_grasp()
                if exchange:
                    tcp[:3, :3] = tcp[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
                tcp = transform @ tcp
                camera = np.linalg.inv(kwargs["base_from_camera"]) @ tcp
                candidates.append(
                    PointGraspCandidate(
                        score=0.5,
                        model_score=0.7,
                        click_distance_m=0.01,
                        gripper_width_m=0.07,
                        base_from_tcp=tuple(tcp.ravel()),
                        camera_from_tcp=tuple(camera.ravel()),
                        visible_collision_free=True,
                        pose_source=(
                            PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1
                            if exchange
                            else "model_scored"
                        ),
                        orientation_variant_axis="z" if exchange else None,
                        orientation_variant_angle_deg=180.0 if exchange else 0.0,
                    )
                )
            result = kwargs["candidate_selector"](
                tuple(candidates),
                tuple(candidates[0].base_from_tcp[i] for i in (3, 7, 11)),
                1,
            )
            if not result.selected_indices:
                raise PointGraspPlanningError({"selected": 0})
            return result

        model.plan = plan
        model.test_proposal_calls = calls
    return session


def _request(session, adapter, skill_id, parameters, sequence):
    observation = adapter.get_observation_snapshot()
    manifest = session.contract_bundle.route_manifest
    entry = manifest.route(skill_id, "2" if skill_id == "manipulation.set_gripper_state" else "1")
    route, parameter = entry.route, entry.route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=CLIENT,
        request_id=f"fast:{sequence}",
        idempotency_key=f"fast:{sequence}",
        client_sequence=sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=observation["station_revision"],
        skill_id=skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=parameter.parameter_profile_id,
        parameter_profile_hash=parameter.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id=observation["snapshot_id"],
        requested_mode="sim_active",
        requested_timeout_ms=route.max_operation_duration_ms,
    )


def _run(session, adapter, request):
    answer = adapter.submit_skill_request(request.to_wire())
    assert answer["accepted"], answer
    identifier = answer["operation"]["operation_id"]
    executor = {
        "perception.capture_grasp_observation": session.grasp_observation_executor,
        "perception.plan_grasp_fast": session.fast_grasp_executor,
        "motion.execute_grasp_approach_plan": session.execute_grasp_plan_executor,
        "manipulation.set_gripper_state": session.gripper_executor,
    }[request.skill_id]
    if request.skill_id.startswith("perception."):
        assert executor.wait_all(timeout_sec=90)
    else:
        assert executor.wait(timeout_sec=90)
    operation = session.service.get_operation(identifier)
    assert operation.terminal, operation
    return operation


@pytest.mark.integration
@pytest.mark.parametrize("initial_open", (False, True))
def test_grazing_retained_target_can_close_then_reopen_without_linkage_failure(
    tmp_path, monkeypatch, record_property, initial_open
):
    # Exact TCP selected by the prior native Agent run. This is a physical
    # recovery regression, not neural inference or successful acquisition.
    target = np.array([
        [-.9875469374978568, -.1401068322851155, .07156201356610258, -.17731913924217224],
        [-.09900551373467692, .906959776013465, .4094165030183057, -.0418371818959713],
        [-.12226591711812916, .39723297979980987, -.9095366981439653, .17180603742599487],
        [0., 0., 0., 1.],
    ])
    session = _session(tmp_path, terminal_offset_m=0.)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        assert session.contract_bundle.fast_grasp_contract.execution_contract.motion_profile.reference_frame_id == "mujoco_sim_hil/station"

        def retained_target(image, depth, **kwargs):
            camera = np.linalg.inv(kwargs["base_from_camera"]) @ target
            candidate = PointGraspCandidate(score=.5, model_score=.7,
                click_distance_m=.01, gripper_width_m=.06891660971730919,
                base_from_tcp=tuple(target.ravel()), camera_from_tcp=tuple(camera.ravel()),
                visible_collision_free=True)
            return kwargs["candidate_selector"]((candidate,), tuple(target[:3, 3]), 1)

        monkeypatch.setattr(session.fast_grasp_planner, "plan", retained_target)
        rig = app.runtime.rig
        step = rig.step_physics
        joints = app.scene.gripper_specs["right_eg2_4b2_cad"].joint_names
        addresses = [app.scene.joint_qpos_addr(name) for name in joints]
        maximum_linkage_error = 0.
        minimum_sensor_contact = 0.

        def observed_step():
            nonlocal maximum_linkage_error, minimum_sensor_contact
            try:
                step()
            except Exception:
                record_property("grazing_failed_physics", json.dumps({
                    "snapshot": rig.snapshot(),
                    "qpos": app.scene.data.qpos.tolist(),
                    "qvel": app.scene.data.qvel.tolist(),
                    "contacts": [str(c) for c in app.scene.contacts()],
                }, default=str))
                raise
            def measure(scene):
                return float(np.ptp(scene.data.qpos[addresses])), min((
                    c.distance_m for c in scene.contacts()
                    if "debug_cube" in (c.body1_name, c.body2_name)
                    and any("dm_tac" in name for name in (c.body1_name, c.body2_name))
                ), default=0.)
            error, contact = rig.capture_scene(measure)
            maximum_linkage_error = max(maximum_linkage_error, error)
            minimum_sensor_contact = min(minimum_sensor_contact, contact)

        monkeypatch.setattr(rig, "step_physics", observed_step)
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        sequence = 0
        if initial_open:
            opened = _run(session, adapter, _request(session, adapter,
                "manipulation.set_gripper_state",
                {"component_id": "right_gripper", "target_state": "open"}, sequence))
            assert opened.status == "completed", opened
            sequence += 1
        source, pixel, _ = _capture(session, adapter, sequence=sequence)
        planned = _run(session, adapter, _plan_request(session, adapter, source, pixel,
            sequence=sequence + 1))
        result = planned.execution_result
        assert result["eligible_for_execution"], result
        retained_native = session.fast_grasp_provider._retained.native_plan
        record_property("grazing_native_plan", json.dumps({
            "trajectory_hash": retained_native.trajectory_hash,
            "q_rad": retained_native.trajectory.q_rad.tolist(),
            "source_model_qpos": retained_native.source_model_qpos.tolist(),
        }))
        executed = _run(session, adapter, _request(session, adapter,
            "motion.execute_grasp_approach_plan", {key: result[key] for key in
                ("component_id", "grasp_approach_plan_id", "grasp_approach_plan_hash")}, sequence + 2))
        assert executed.status == "completed", executed
        for gripper_sequence, state in enumerate(("closed", "open"), sequence + 3):
            operation = _run(session, adapter, _request(session, adapter,
                "manipulation.set_gripper_state", {"component_id": "right_gripper", "target_state": state}, gripper_sequence))
            assert operation.status == "completed", (state, operation.reason_code, operation.progress_summary)
            assert app.held
        assert session.gripper_ports["right_gripper"].observe_state().measured_close < .01
        assert maximum_linkage_error < .02
        assert minimum_sensor_contact >= -.001
        record_property("retained_grazing_recovery", json.dumps({
            "maximum_linkage_error_rad": maximum_linkage_error,
            "minimum_sensor_contact_m": minimum_sensor_contact,
            "physics_timestep_sec": float(app.scene.model.opt.timestep),
            "physical_grasp_success_asserted": False,
        }))
    finally:
        assert session.close().accepted


def _capture(session, adapter, *, sequence=0):
    operation = _run(
        session,
        adapter,
        _request(
            session,
            adapter,
            "perception.capture_grasp_observation",
            {"selected_roles": ["main", "wrist_left", "wrist_right"]},
            sequence,
        ),
    )
    assert operation.status == "completed", operation
    artifact = next(
        b.artifact
        for b in operation.output_artifact_bindings
        if b.slot_id == "grasp_observation"
    )
    views, metadata, points, _ = decode_grasp_multiview_snapshot(
        session.artifact_store.managed_read_path(artifact.artifact_id).read_bytes()
    )
    assert points is None and len(views) == 3
    arrays, view = next(
        decode_grasp_rgbd_snapshot(data)
        for data in views
        if decode_grasp_rgbd_snapshot(data)[1]["role"] == "main"
    )
    rgb = arrays["rgb"]
    mask = (rgb[:, :, 2] > 100) & (rgb[:, :, 2].astype(float) > rgb[:, :, 0] * 1.5)
    y, x = np.nonzero(mask)
    assert len(x) > 32
    return operation, (int(np.median(x)), int(np.median(y))), (arrays, view, metadata)


def _plan_request(session, adapter, source, pixel, sequence=1):
    return _request(
        session,
        adapter,
        "perception.plan_grasp_fast",
        {
            "component_id": "right_arm",
            "observation_operation_id": source.operation_id,
            "role": "main",
            "target_pixel_uv": pixel,
            "view_policy": "single",
            "speed_scale": 1.0,
            "planning_budget_sec": 20.0,
        },
        sequence,
    )


@pytest.mark.integration
def test_native_public_frozen_observation_is_atomic_and_model_lifecycle_is_session_owned(
    tmp_path, monkeypatch
):
    session = _session(tmp_path)
    app = session.application
    model = session.fast_grasp_planner
    try:
        app.start_observation_ready()
        session.start_perception_resources()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        state = app.capture_grasp_scene_state()
        for name in (
            "request_hold",
            "activate_operation",
            "submit_action",
            "close_after_hold",
        ):
            if hasattr(app, name):
                monkeypatch.setattr(
                    app,
                    name,
                    lambda *a, _name=name, **kw: pytest.fail(
                        f"Read-only called {_name}"
                    ),
                )
        operation, _, (_, view, metadata) = _capture(session, adapter)
        assert (
            app.capture_grasp_scene_state() == state == metadata["native_scene_state"]
        )
        assert (
            view["reference_frame_id"]
            == session.contract_bundle.motion_profile.reference_frame_id
        )
        assert view["depth_valid_range_m"] == [0, 10]
        assert len(operation.output_artifact_bindings) == 4
        assert model.point_segmenter.starts == 1 and not model.client.calls
        assert not app.target.planned_tcp_motion.active
    finally:
        monkeypatch.undo()
        assert session.close().accepted
        assert session.close().accepted
        assert model.point_segmenter.closes == 1 and model.client.closed_for_test
        with pytest.raises(RuntimeError, match="Closed Gateway"):
            session.start_perception_resources()


@pytest.mark.integration
def test_published_fast_grasp_pairs_retained_proposal_native_chain_without_replanning(
    tmp_path, monkeypatch, record_property
):
    facts = []
    session = _session(tmp_path, trace=facts, retained_selector=True)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        before = app.capture_grasp_scene_state()
        result = _run(session, adapter, _plan_request(session, adapter, source, pixel))
        record_property(
            "fast_operation_json", json.dumps(result.to_wire(), default=str)
        )
        assert result.status == "completed", result
        assert app.capture_grasp_scene_state() == before
        output = result.execution_result
        assert output["planning_status"] == "path_found", (
            output,
            session.fast_grasp_planner.last_filter_summary,
        )
        assert (
            not output["motion_or_hold_performed"] and output["eligible_for_execution"]
        )
        assert session.fast_grasp_planner.test_proposal_calls == [
            "retained_proposal_test_port"
        ]
        assert not session.fast_grasp_planner.client.calls
        assert output["diagnostics"]["path_query_count"] <= session.contract_bundle.fast_grasp_contract.policy.maximum_path_queries
        retained = session.fast_grasp_provider._retained
        assert len(retained.native_plan.stages) == 3
        assert not any(f.event_type == "command" for f in facts)
        planner = app.target.planned_tcp_motion
        monkeypatch.setattr(
            planner, "plan", lambda *a, **kw: pytest.fail("Execution must not replan.")
        )
        monkeypatch.setattr(
            planner,
            "plan_chain",
            lambda *a, **kw: pytest.fail("Execution must not replan a chain."),
        )
        monkeypatch.setattr(
            app.runtime.pose_resolver.ik,
            "solve",
            lambda *a, **kw: pytest.fail("Execution must not rerun IK."),
        )
        parameters = {
            "component_id": "right_arm",
            "grasp_approach_plan_id": output["grasp_approach_plan_id"],
            "grasp_approach_plan_hash": output["grasp_approach_plan_hash"],
        }
        completed = _run(
            session,
            adapter,
            _request(
                session, adapter, "motion.execute_grasp_approach_plan", parameters, 2
            ),
        )
        record_property(
            "native_execution_json", json.dumps(completed.to_wire(), default=str)
        )
        if completed.status != "completed":
            record_property("native_execution_failed_physics", json.dumps({
                "snapshot": app.runtime.rig.snapshot(),
                "qpos": app.scene.data.qpos.tolist(),
                "qvel": app.scene.data.qvel.tolist(),
                "facts": [{"event_type": f.event_type, "payload": dict(f.payload)}
                          for f in facts[-48:]],
            }, default=str))
        assert completed.status == "completed", completed
        assert completed.task_outcome == "succeeded", completed
        assert not planner.owns_pending_plan(retained.native_plan) and app.held
        assert any(f.event_type == "command" for f in facts)
        replay = _run(
            session,
            adapter,
            _request(
                session, adapter, "motion.execute_grasp_approach_plan", parameters, 3
            ),
        )
        assert replay.status == "failed"
        record_property(
            "evidence_scope",
            "actual Gateway/native physics; fixed NN proposal and color-mask test ports, not fresh model inference or grasp/lift success",
        )
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_fast_planning_rejects_changed_frozen_scene_before_model_inference(tmp_path):
    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        app.scene.data.qpos[-1] += 0.001
        failure = _run(session, adapter, _plan_request(session, adapter, source, pixel))
        assert failure.status == "failed"
        assert failure.reason_code == "grasp_fast_capture_scene_changed", failure
        assert not session.fast_grasp_planner.client.calls
        assert not app.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_actual_proposal_filter_and_search_report_rejection_without_motion(
    tmp_path, record_property
):
    """Use the real proposal pipeline, but deterministic mask/NN-output ports."""
    facts = []
    # Keep inference/filtering real, but require an explicitly unreachable
    # pregrasp. Ordinary proposals can now find a valid chain on the mainline.
    session = _session(tmp_path, trace=facts, pregrasp_distance_m=10.0)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        state = app.capture_grasp_scene_state()
        operation = _run(
            session, adapter, _plan_request(session, adapter, source, pixel)
        )
        assert operation.status == "completed", operation
        result = operation.execution_result
        assert result["planning_status"] == "rejected", result
        assert (
            not result["eligible_for_execution"]
            and result["grasp_approach_plan_id"] is None
        )
        summary = session.fast_grasp_planner.last_filter_summary
        assert summary["detected"] == 6 and summary["visible_collision_valid"] > 0
        assert result["diagnostics"]["path_query_count"] > 0
        assert len(session.fast_grasp_planner.client.calls) == 1
        assert not app.target.planned_tcp_motion.active
        assert session.fast_grasp_provider._retained is None
        assert app.capture_grasp_scene_state() == state
        assert not any(f.event_type == "command" for f in facts)
        record_property(
            "fixed_proposal_rejection_json", json.dumps(result, default=dict)
        )
        record_property("fixed_proposal_filter_summary_json", json.dumps(summary))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_fast_publishes_all_sixty_four_rejections_with_bounded_summary(tmp_path, monkeypatch, caplog):
    """Mock goal failures to exercise the public result/log bounds, not physics."""
    from types import SimpleNamespace

    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

    session = _session(tmp_path)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        state = app.capture_grasp_scene_state()
        candidates = tuple(SimpleNamespace(base_from_tcp=np.eye(4), score=.8, visible_collision_free=True) for _ in range(65))
        queries = []

        def rejected_chain(*args, **kwargs):
            queries.append(kwargs["time_budget_sec"])
            raise MujocoTcpPlanRejected("safe_tcp_endpoint_search_exhausted", details={
                "chain_stage_index": 0, "reason": "no_valid_endpoint",
                "endpoint_attempts": [{}] * 8,
            })

        def proposals(*args, **kwargs):
            assert kwargs["refine_grasp_centers"] is True
            kwargs["candidate_selector"](candidates, (0., 0., .1), 1)
            raise PointGraspPlanningError({"selected": 0, "arm_shortlist_full_path_query_count": len(queries)})

        monkeypatch.setattr(app.target.planned_tcp_motion, "plan_chain", rejected_chain)
        monkeypatch.setattr(session.fast_grasp_planner, "plan", proposals)
        with caplog.at_level("INFO", logger="agent_gateway.profiles.mujoco.grasp_fast_plan"):
            operation = _run(session, adapter, _plan_request(session, adapter, source, pixel))
        assert operation.status == "completed", operation
        result = operation.execution_result
        assert result["planning_status"] == "rejected" and not result["eligible_for_execution"]
        assert result["diagnostics"]["path_query_count"] == len(queries) == 64
        encoded = result["diagnostics"]["attempts_json"]
        assert len(encoded) <= 8192 and len(json.loads(encoded)) == 64
        assert {r["index"] for r in json.loads(encoded)} == set(range(64))
        logs = [r.message.split("Native fast grasp path attempts: ")[1]
                for r in caplog.records if r.message.startswith("Native fast grasp path attempts: ")]
        detailed = json.loads(logs[0])["attempts"]
        assert len(logs) == 1 and len(detailed) == 64
        assert all(r["endpoint_attempt_count"] == 8 and r["failed_stage"] == 0 for r in detailed)
        assert app.capture_grasp_scene_state() == state
        assert session.fast_grasp_provider._retained is None
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize(
    "failure,reason",
    (
        ("hash", "grasp_path_plan_binding_mismatch"),
        ("expired", "grasp_path_plan_expired"),
        ("pending_producer", "grasp_path_plan_not_committed"),
        ("failed_producer", "grasp_path_plan_not_committed"),
        ("source_evicted", "grasp_path_plan_source_unavailable"),
        ("source_changed", "grasp_path_plan_source_unavailable"),
        ("superseded", "grasp_path_plan_no_longer_pending"),
    ),
)
def test_invalid_paired_reference_finishes_with_receipt_before_dispatch(
    tmp_path, failure, reason
):
    facts = []
    session = _session(tmp_path, trace=facts, retained_selector=True)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        planned = _run(session, adapter, _plan_request(session, adapter, source, pixel))
        result = planned.execution_result
        assert result["eligible_for_execution"], result
        provider = session.fast_grasp_provider
        retained = provider._retained
        parameters = {
            "component_id": "right_arm",
            "grasp_approach_plan_id": result["grasp_approach_plan_id"],
            "grasp_approach_plan_hash": result["grasp_approach_plan_hash"],
        }
        lookup = provider.operation_lookup
        if failure == "hash":
            parameters["grasp_approach_plan_hash"] = "f" * 64
        elif failure == "expired":
            provider.clock = lambda: retained.expires_at
        elif failure in {"pending_producer", "failed_producer"}:
            modified = replace(
                planned,
                status="running" if failure == "pending_producer" else "failed",
                execution_outcome=(
                    "pending" if failure == "pending_producer" else "failed"
                ),
                execution_result=None,
            )
            provider.operation_lookup = lambda identifier: (
                modified if identifier == planned.operation_id else lookup(identifier)
            )
        elif failure == "source_evicted":
            provider.operation_lookup = lambda identifier: (
                None if identifier == source.operation_id else lookup(identifier)
            )
        elif failure == "source_changed":
            app.scene.data.qpos[-1] += 0.001
        elif failure == "superseded":
            app.target.planned_tcp_motion.discard_pending_plan(retained.native_plan)
        operation = _run(
            session,
            adapter,
            _request(
                session, adapter, "motion.execute_grasp_approach_plan", parameters, 2
            ),
        )
        assert operation.status == "failed", operation
        assert operation.reason_code == reason, operation
        assert operation.progress_summary["lower_hold_confirmed"] is True
        assert not any(f.event_type == "command" for f in facts)
        assert not app.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_cancel_after_query_does_not_publish_or_hold_from_readonly_route(
    tmp_path, monkeypatch
):
    facts = []
    session = _session(tmp_path, trace=facts, retained_selector=True)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        original = session.fast_grasp_planner.plan

        def cancel_after_native_query(*args, **kwargs):
            value = original(*args, **kwargs)
            assert session.service.cancel_operation(
                operation_id=session.service.operation_store.active_operation_id,
                reason="test:cancel-after-native-query",
                current_station_revision=session.observation_producer.current_station_revision,
            ).accepted
            return value

        monkeypatch.setattr(
            session.fast_grasp_planner, "plan", cancel_after_native_query
        )
        monkeypatch.setattr(
            app,
            "request_hold",
            lambda *a, **kw: pytest.fail("Read-only cancellation cannot HOLD."),
        )
        operation = _run(
            session, adapter, _plan_request(session, adapter, source, pixel)
        )
        assert operation.status == "cancelled", operation
        assert session.fast_grasp_provider._retained is None
        assert app.target.planned_tcp_motion._pending_plan is None
        assert not any(f.event_type == "command" for f in facts)
    finally:
        monkeypatch.undo()
        assert session.close().accepted


@pytest.mark.integration
def test_paired_ttl_expiring_after_source_validation_rejects_before_consumption(
    tmp_path, monkeypatch
):
    facts = []
    session = _session(tmp_path, trace=facts, retained_selector=True)
    try:
        app = session.application
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        planned = _run(session, adapter, _plan_request(session, adapter, source, pixel))
        assert planned.execution_result["eligible_for_execution"]
        provider = session.fast_grasp_provider
        retained = provider._retained
        clock = [retained.created_at]
        provider.clock = lambda: clock[0]
        native = app.target.planned_tcp_motion
        owned = native.owns_pending_plan
        count = 0

        def cross_deadline(plan):
            nonlocal count
            answer = owned(plan)
            count += 1
            if count == 2:
                # Both provider TTL reads passed; source work / projection /
                # lock wait can still cross expiry before native activation.
                clock[0] = retained.expires_at
            return answer

        monkeypatch.setattr(native, "owns_pending_plan", cross_deadline)
        parameters = {
            "component_id": "right_arm",
            "grasp_approach_plan_id": retained.plan_id,
            "grasp_approach_plan_hash": retained.content_hash,
        }
        operation = _run(
            session,
            adapter,
            _request(
                session, adapter, "motion.execute_grasp_approach_plan", parameters, 2
            ),
        )
        assert count == 2
        assert (
            operation.status == "failed"
            and operation.reason_code == "grasp_path_plan_expired"
        ), operation
        assert operation.progress_summary["lower_hold_confirmed"]
        assert not any(f.event_type == "command" for f in facts)
        assert not native.active
    finally:
        monkeypatch.undo()
        assert session.close().accepted
