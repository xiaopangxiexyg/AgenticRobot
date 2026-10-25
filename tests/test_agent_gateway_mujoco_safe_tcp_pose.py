from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import (
    AgentOwnOperationCancelRequest,
    AgentSkillRequest,
)
from agent_gateway.skills.motion.definition import ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1
from project_paths import resolve_path
from sim.mujoco.collision_safety import CollisionSafetyOracle
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoJointSceneOracle,
    MujocoTcpPlanRejected,
    _NonContactMovePolicy,
)

TASK = Path("sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json")
CLIENT = "test:mujoco-safe-tcp"


def _session(trace, *, after_fact=None, task=TASK, spatial=False, grasp=False,
             contact_target=None, motion_side="right", safe_profile="local_step", queries=False, path_constraints=False, continuations=False):
    def consume(fact):
        trace.append(fact)
        if after_fact is not None:
            after_fact(fact)

    return build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path(task),
            realtime_worker=False,
            runtime_session_id="test:mujoco-safe-tcp",
            control_domain_id="test:mujoco-safe-tcp",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(
                motion_side=motion_side,
            ),
            exposed_pose_sides=(motion_side,),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=(CLIENT,),
        motion_side=motion_side,
        trace_sink=consume,
        enable_safe_tcp_pose=True,
        enable_safe_tcp_path_constraints=path_constraints,
        enable_tcp_continuation_queries=continuations,
        enable_spatial_goals=spatial,
        enable_object_goal_queries=queries,
        safe_tcp_motion_profile=safe_profile,
        enable_grippers=grasp,
        enable_grasp_approach=grasp,
        grasp_contact_target_body=contact_target,
    )


def _request(session, adapter, target, *, sequence=0, skill_id="motion.move_tcp_to_pose_safe", policy=None, timeout_ms=120_000, version="1", path_constraint=None):
    observation = adapter.get_observation_snapshot()
    entry = session.contract_bundle.route_manifest.route(
        skill_id, version
    )
    route = entry.route
    profile = session.contract_bundle.motion_profile
    parameter = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=CLIENT,
        request_id=f"request:safe:{sequence + 1}",
        idempotency_key=f"safe:{sequence + 1}",
        client_sequence=sequence,
        runtime_session_id=route_manifest(session).runtime_session_id,
        skill_route_manifest_hash=route_manifest(session).semantic_content_hash,
        expected_station_revision=int(observation["station_revision"]),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=parameter.parameter_profile_id,
        parameter_profile_hash=parameter.parameter_profile_hash,
        parameters={
            "pose_targets_by_component": {profile.component_id: target},
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            **({"pose_goal_policy": policy} if policy is not None else {}),
            **({"path_constraint": path_constraint} if version == "2" else {}),
        },
        observation_snapshot_id=observation["snapshot_id"],
        requested_mode="sim_active",
        requested_timeout_ms=min(timeout_ms, route.max_operation_duration_ms),
    )


def route_manifest(session):
    return session.contract_bundle.route_manifest


def _place_cube_at_collision(application, plan, *, terminal):
    """Construct a native collision fixture with a verified free start."""
    scene = application.scene
    scratch = scene.mujoco.MjData(scene.model)
    scratch.qpos[:] = scene.data.qpos
    index = -1 if terminal else len(plan.trajectory.q_rad) // 2
    sample = plan.trajectory.q_rad[index]
    for name, value in zip(scene.arm_specs["right"].joint_names, sample):
        scratch.qpos[scene.joint_qpos_addr(name)] = value
    scene.mujoco.mj_kinematics(scene.model, scratch)
    obj = next(i for i in range(scene.model.ngeom) if scene.model.geom_contype[i] == 8)
    body = int(scene.model.geom_bodyid[obj])
    joint = int(scene.model.body_jntadr[body])
    address = int(scene.model.jnt_qposadr[joint])
    oracle = MujocoJointSceneOracle(scene, scene.data.qpos.copy())
    offsets = [
        np.zeros(3),
        *(sign * axis * 0.8 for axis in np.eye(3) for sign in (-1, 1)),
    ]
    for geom in range(scene.model.ngeom):
        name = (
            scene.mujoco.mj_id2name(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_BODY,
                int(scene.model.geom_bodyid[geom]),
            )
            or ""
        )
        if scene.model.geom_contype[geom] != 4 or not name.startswith("right_"):
            continue
        center, half = scene.model.geom_aabb[geom, :3], scene.model.geom_aabb[geom, 3:]
        for offset in offsets:
            point = scratch.geom_xpos[geom] + scratch.geom_xmat[geom].reshape(3, 3) @ (
                center + half * offset
            )
            oracle.source.qpos[address : address + 3] = point
            if not oracle.check(plan.path.start_qpos_by_component).safe:
                continue
            hit = oracle.check(
                {"left": plan.path.start_qpos_by_component["left"], "right": sample}
            )
            if hit.safe or (terminal and hit.metadata["pair_class"] != "4:8"):
                continue
            if (
                not terminal
                and not oracle.check(plan.path.target_qpos_by_component).safe
            ):
                continue
            scene.data.qpos[address : address + 3] = point
            # This synthetic obstacle is stationary in mid-air for the test;
            # no gravity change is made to either arm or to production scenes.
            scene.model.body_gravcomp[body] = 1.0
            scene.data.qvel[
                int(scene.model.jnt_dofadr[joint]) : int(scene.model.jnt_dofadr[joint])
                + 6
            ] = 0
            # MuJoCo caches the nonzero-gravcomp body count in mjModel.
            # Recompute constants for this fixture, retaining its state cut.
            qpos, qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
            scene.mujoco.mj_setConst(scene.model, scene.data)
            scene.data.qpos[:] = qpos
            scene.data.qvel[:] = qvel
            assert scene.model.ngravcomp > 0
            scene.forward()
            return point.tolist()
    raise AssertionError(
        "Could not construct the declared start-free collision fixture"
    )


@pytest.mark.parametrize("safe_profile", ["local_step", "object_relation_precise"])
def test_native_safe_session_does_not_republish_archived_debug_skills(safe_profile):
    session = _session([], safe_profile=safe_profile, queries=True)
    try:
        bundle = session.contract_bundle
        published = {entry.route.skill_id for entry in bundle.route_manifest.skill_routes}
        assert {"motion.move_tcp_to_pose", "motion.move_tcp_to_pose_safe"} <= published
        assert ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1.isdisjoint(published)
        assert len(session._motion_executors()) == 2
        for skill_id in sorted(ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1):
            with pytest.raises(KeyError):
                bundle.route_manifest.route(skill_id, "1")
            with pytest.raises(KeyError):
                bundle.definition_catalog.definition(skill_id, "1")
            with pytest.raises(KeyError):
                bundle.schema_catalog.entry(skill_id, "1")
            with pytest.raises(KeyError):
                bundle.io_catalog.contract(skill_id, "1")
        assert not session.application.started
    finally:
        outcome = session.close()
        assert not outcome.accepted
        assert outcome.reason_code == "station_hold_not_confirmed_no_active_operation"
        assert session.application.closed


def test_native_gateway_rejects_archived_skill_requests_without_dispatch():
    facts = []
    session = _session(facts)
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        data = session.application.scene.data
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time), len(facts))
        for sequence, skill_id in enumerate(sorted(ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1)):
            request = replace(
                _request(session, adapter, [.4, -.1, .5, 0., 0., 0., 1.], sequence=sequence),
                skill_id=skill_id,
            )
            outcome = adapter.submit_skill_request(request.to_wire())
            assert not outcome["accepted"], outcome
            assert outcome["reason_code"] == "skill_route_not_advertised"
        assert session.motion_executor.workers == ()
        assert session.safe_tcp_pose_executor.workers == ()
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        assert float(data.time) == before[2]
        assert len(facts) == before[3]
    finally:
        assert session.close().accepted




@pytest.mark.integration
@pytest.mark.parametrize("task", [TASK, Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")])
def test_spatial_goal_calculation_then_existing_safe_move(task, record_property):
    session = _session([], task=task, spatial=True)
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        snapshot = adapter.get_observation_snapshot()
        initial = snapshot["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        profile = session.contract_bundle.motion_profile
        entry = session.contract_bundle.spatial_goal_registration.entry
        route = entry.route
        parameter_profile = route.allowed_parameter_profiles[0]
        request = replace(_request(session, adapter, initial),
            skill_id=route.skill_id, skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash, route_id=route.route_id,
            route_version=route.route_version, route_hash=route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter_profile.parameter_profile_id,
            parameter_profile_hash=parameter_profile.parameter_profile_hash,
            requested_timeout_ms=5_000,
            parameters={"component_id": "right_arm", "reference_frame_id": profile.reference_frame_id,
                "current_tcp_pose_xyz_quat_xyzw": initial, "anchor_xyz_m": initial[:3],
                "offset_xyz_m": [0., 0., .03], "offset_frame": "reference",
                "orientation_mode": "preserve", "tool_axis": None, "axis_reference_xyz": None})
        before = session.application.scene.data.qpos.copy()
        before_velocity = session.application.scene.data.qvel.copy()
        before_time = float(session.application.scene.data.time)
        readonly = session.wire_adapter(authenticated_client_id=CLIENT, read_only_access=True)
        admitted = readonly.submit_read_only_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.spatial_goal_executor.wait_all(timeout_sec=5)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "completed", operation
        goal = operation.execution_result
        np.testing.assert_array_equal(session.application.scene.data.qpos, before)
        np.testing.assert_array_equal(session.application.scene.data.qvel, before_velocity)
        assert float(session.application.scene.data.time) == before_time
        assert goal["path_checked"] is False and goal["motion_executed"] is False
        assert goal["constraint_scope"] == "endpoint_only"
        target = goal["pose_targets_by_component"]["right_arm"]
        admitted = adapter.submit_skill_request(_request(session, adapter, target, sequence=1).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "completed", operation
        final = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        error = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
        assert error <= profile.translation_tolerance_m
        record_property("relation_goal_evidence_json", json.dumps({
            "task": str(task), "initial": initial, "goal": target, "final": final,
            "position_error_m": error, "spatial_calculation_changed_state": False,
            "motion_outcome": operation.execution_outcome, "hardware_used": False,
        }))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_safe_tcp_consumes_planned_joints_without_re_ik(monkeypatch, record_property):
    facts = []
    session = _session(facts)
    plans = []
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        initial = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"
        ]["right_arm"]
        target = list(initial)
        target[2] += 0.03
        planner = session.application.target.planned_tcp_motion
        original = planner.plan

        def capture_plan(*args, **kwargs):
            try:
                plan = original(*args, **kwargs)
            except MujocoTcpPlanRejected as error:
                record_property(
                    "planning_rejection_json", json.dumps(error.details, default=str)
                )
                raise
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture_plan)
        monkeypatch.setattr(
            session.application.runtime.pose_resolver.ik,
            "solve",
            lambda *a, **kw: pytest.fail("Safe path must not be replaced by new IK."),
        )
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target).to_wire()
        )
        assert admitted["accepted"]
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        if operation.status != "completed":
            record_property(
                "failure_trace_json",
                json.dumps([fact.to_payload() for fact in facts[-16:]], default=str),
            )
        assert operation.status == "completed", (
            f"{operation.reason_code}: {operation.progress_summary}"
        )
        assert operation.task_outcome == "succeeded", operation
        assert len(plans) == 1
        plan = plans[0]
        rig_config = session.application.runtime.rig.config
        effective_velocity = np.minimum(
            rig_config.max_velocity_rad_s,
            np.deg2rad(rig_config.max_command_step_deg) * rig_config.timing.command_hz,
        )
        certificate = plan.trajectory.metadata["continuous_hard_limit_certificate"]
        assert np.all(
            np.asarray(certificate["max_abs_velocity_rad_s"])
            <= effective_velocity + 1e-7
        )
        references = [
            fact.payload["payload"]["output_values_by_effector"]["right_arm"]
            for fact in facts
            if fact.event_type == "component_execution_target"
        ]
        assert len(references) >= len(plan.trajectory.q_rad)
        np.testing.assert_allclose(
            references[: len(plan.trajectory.q_rad)], plan.trajectory.q_rad, atol=1e-7
        )
        assert any(
            "mujoco-safe-joint-trajectory:" in ref
            for ref in operation.execution_receipt_refs
        )
        final = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"
        ]["right_arm"]
        error = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
        assert error <= session.contract_bundle.motion_profile.translation_tolerance_m
        assert not planner.active
        record_property(
            "evidence_json",
            json.dumps(
                {
                    "plan_hash": plan.trajectory_hash,
                    "path_metadata": dict(plan.path.metadata),
                    "reference_count": len(references),
                    "planned_reference_count": len(plan.trajectory.q_rad),
                    "target_error_m": error,
                    "hardware_used": False,
                },
                default=str,
            ),
        )
    finally:
        outcome = session.close()
        assert outcome.accepted


@pytest.mark.integration
@pytest.mark.parametrize("grasp", (False, True))
def test_safe_tcp_rejects_gripper_object_collision_without_dispatch(record_property, grasp):
    facts = []
    session = _session(facts, grasp=grasp)
    try:
        session.application.start_observation_ready()
        application = session.application
        planner = application.target.planned_tcp_motion
        start_base = list(application.capture_tcp_pose_by_component()["right_arm"])
        target_base = list(start_base)
        target_base[2] += 0.08
        profile = session.contract_bundle.motion_profile
        free_plan = planner.plan(
            tuple(target_base),
            translation_tolerance_m=profile.translation_tolerance_m,
            rotation_tolerance_rad=profile.rotation_tolerance_rad,
            cancel_requested=lambda: False,
        )
        scene = application.scene

        application.runtime.rig.capture_scene(
            lambda _scene: _place_cube_at_collision(
                application, free_plan, terminal=True
            )
        )
        oracle = MujocoJointSceneOracle(scene, planner._model_qpos())
        assert oracle.check(free_plan.path.start_qpos_by_component).safe
        assert not oracle.check(free_plan.path.target_qpos_by_component).safe
        target = application.transform_tcp_pose_xyzquat(
            target_base,
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id,
        )
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        baseline = sum(fact.event_type == "command" for fact in facts)
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target, skill_id=(
                "motion.approach_grasp_pose" if grasp else "motion.move_tcp_to_pose_safe"
            )).to_wire()
        )
        assert admitted["accepted"]
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation.status == "failed", operation
        assert "safe_tcp_target_blocked" in operation.reason_code, operation
        assert sum(fact.event_type == "command" for fact in facts) == baseline
        assert application.held
        assert not planner.active
        record_property(
            "evidence_json",
            json.dumps(
                {
                    "reason": operation.reason_code,
                    "dispatch_count": 0,
                    "hardware_used": False,
                }
            ),
        )
    finally:
        outcome = session.close()
        assert outcome.accepted


@pytest.mark.integration
def test_safe_tcp_searches_around_blocked_direct_path(monkeypatch, record_property):
    facts = []
    session = _session(facts)
    plans = []
    try:
        session.application.start_observation_ready()
        application = session.application
        planner = application.target.planned_tcp_motion
        profile = session.contract_bundle.motion_profile
        target_base = list(application.capture_tcp_pose_by_component()["right_arm"])
        # Move across the hand's thin dimension so a 3 cm obstacle can block
        # the middle without also occupying either endpoint.
        target_base[0] += 0.08
        original = planner.plan
        free = original(
            tuple(target_base),
            translation_tolerance_m=profile.translation_tolerance_m,
            rotation_tolerance_rad=profile.rotation_tolerance_rad,
            cancel_requested=lambda: False,
        )
        assert free.path.metadata["direct_path"]
        point = application.runtime.rig.capture_scene(
            lambda _scene: _place_cube_at_collision(application, free, terminal=False)
        )

        def capture_plan(*args, **kwargs):
            try:
                plan = original(*args, **kwargs)
            except MujocoTcpPlanRejected as error:
                record_property(
                    "planning_rejection_json", json.dumps(error.details, default=str)
                )
                raise
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture_plan)
        target = application.transform_tcp_pose_xyzquat(
            target_base,
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id,
        )
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target).to_wire()
        )
        assert admitted["accepted"]
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation.status == "completed", (
            f"{operation.reason_code}: {operation.progress_summary}"
        )
        assert operation.task_outcome == "succeeded"
        assert len(plans) == 1
        plan = plans[0]
        assert not plan.path.metadata["direct_path"]
        assert len(plan.path.waypoints) > 2
        assert application.held
        np.testing.assert_allclose(
            application.scene.body_pose(application.task_config.object_body_name)[
                :3, 3
            ],
            point,
            atol=1e-5,
            rtol=0,
        )
        record_property(
            "evidence_json",
            json.dumps(
                {
                    "obstacle_xyz_m": point,
                    "path_metadata": dict(plan.path.metadata),
                    "waypoint_count": len(plan.path.waypoints),
                    "trajectory_hash": plan.trajectory_hash,
                    "task_outcome": operation.task_outcome,
                    "hardware_used": False,
                },
                default=str,
            ),
        )
    finally:
        assert session.close().accepted


@pytest.mark.parametrize(
    "separation,expected", [(0.01, True), (0.001, False), (-0.002, False)]
)
def test_safe_move_threshold_query_preserves_real_collision_rejection(
    monkeypatch, separation, expected
):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom type="box" size=".02 .02 .02" contype="4"/></body>'
        f'<body pos="{0.04 + separation} 0 0"><joint type="slide"/>'
        '<geom type="box" size=".02 .02 .02" contype="8"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original = mujoco.mj_geomDistance
    radii = []

    def inconsistent_broad_query(model, data, first, second, radius, witness):
        radii.append(radius)
        return (
            0.0
            if radius > 0.002
            else original(model, data, first, second, radius, witness)
        )

    monkeypatch.setattr(mujoco, "mj_geomDistance", inconsistent_broad_query)
    oracle = CollisionSafetyOracle(
        mujoco, model, _NonContactMovePolicy(minimum_clearance_m=0.002)
    )
    check = oracle.check(data)
    assert check.safe is expected
    assert check.minimum_distance_m == pytest.approx(min(0.002, separation))
    assert all(radius == 0.002 for radius in radii)


@pytest.mark.integration
def test_safe_plan_ownership_reference_order_and_cancellation():
    session = _session([])
    try:
        session.application.start_observation_ready()
        application = session.application
        planner = application.target.planned_tcp_motion
        before = planner._model_qpos()
        target = list(application.capture_tcp_pose_by_component()["right_arm"])
        target[2] += 0.03
        profile = session.contract_bundle.motion_profile
        kwargs = {
            "translation_tolerance_m": profile.translation_tolerance_m,
            "rotation_tolerance_rad": profile.rotation_tolerance_rad,
        }
        with pytest.raises(MujocoTcpPlanRejected, match="planning_cancelled"):
            planner.plan(tuple(target), **kwargs, cancel_requested=lambda: True)
        assert not planner.active
        plan = planner.plan(tuple(target), **kwargs, cancel_requested=lambda: False)
        with pytest.raises(MujocoTcpPlanRejected, match="not_owned"):
            planner.activate(replace(plan, trajectory_hash="0" * 64))
        planner.activate(plan)
        with pytest.raises(ValueError, match="non-negative"):
            planner.reference(-1)
        with pytest.raises(MujocoTcpPlanRejected, match="sequence_changed"):
            planner.reference(2)
        assert planner.reference(0) == plan.reference_poses[0]
        assert (
            application.request_hold(reason="test:cancel-registered-plan") is not None
        )
        assert not planner.active
        with pytest.raises(MujocoTcpPlanRejected, match="no_active_plan"):
            planner.reference(0)
        # Planning, rejected substitutions and cancellation emit no motion.
        np.testing.assert_allclose(planner._model_qpos(), before, atol=1e-12)
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize(
    "offset_m", [0.0, 0.005], ids=["exact_pose", "within_tolerance"]
)
def test_safe_move_already_satisfied_confirms_hold_without_commands(
    offset_m, record_property
):
    facts = []
    session = _session(facts)
    try:
        application = session.application
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        target = list(
            adapter.get_observation_snapshot()["measurement_summary"][
                "pose_xyz_quat_xyzw_by_component"
            ]["right_arm"]
        )
        target[2] += offset_m
        before = sum(fact.event_type == "command" for fact in facts)
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target).to_wire()
        )
        assert admitted["accepted"]
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation.status == "completed", operation
        assert operation.task_outcome == "succeeded", operation
        commands = sum(fact.event_type == "command" for fact in facts) - before
        record_property(
            "already_satisfied_json",
            json.dumps(
                {
                    "offset_m": offset_m,
                    "command_count": commands,
                    "execution_receipt_refs": operation.execution_receipt_refs,
                    "hardware_used": False,
                }
            ),
        )
        assert commands == 0
        assert application.held and not application.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


def _exercise_cancelled_safe_move_replans_from_measured_hold_without_old_references(
    monkeypatch, record_property, motion_side
):
    facts, plans = [], []
    executing = threading.Event()
    reference_count = 0

    def observe_progress(fact):
        nonlocal reference_count
        if fact.event_type == "component_execution_target":
            reference_count += 1
            if reference_count == 3:
                executing.set()

    session = _session(facts, after_fact=observe_progress, motion_side=motion_side)
    try:
        application = session.application
        component_id = application.component_id
        assert application.motion_side == motion_side
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        planner = application.target.planned_tcp_motion
        original = planner.plan

        def capture_plan(*args, **kwargs):
            plan = original(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture_plan)
        observation = adapter.get_observation_snapshot()
        target = list(
            observation["measurement_summary"]["pose_xyz_quat_xyzw_by_component"][
                component_id
            ]
        )
        target[2] += 0.08
        first = adapter.submit_skill_request(
            _request(session, adapter, target).to_wire()
        )
        assert first["accepted"]
        assert executing.wait(timeout=10), first
        observation = adapter.get_observation_snapshot()
        cancelled = adapter.cancel_own_operation(
            AgentOwnOperationCancelRequest(
                client_id=CLIENT,
                control_request_id="control:safe:cancel",
                idempotency_key="control:safe:cancel",
                runtime_session_id=route_manifest(session).runtime_session_id,
                skill_route_manifest_hash=route_manifest(session).semantic_content_hash,
                operation_id=first["operation"]["operation_id"],
                observation_snapshot_id=observation["snapshot_id"],
                expected_station_revision=observation["station_revision"],
                reason_code="agent_task_aborted",
            ).to_wire()
        )
        assert cancelled["accepted"], cancelled
        assert session.wait(timeout_sec=10)
        first_operation = session.service.operation_store.get(
            first["operation"]["operation_id"]
        )
        assert first_operation.status == "cancelled", first_operation
        assert cancelled["port_receipt"]["succeeded"], cancelled
        assert cancelled["port_receipt"]["proves_motion_or_hold"], cancelled
        assert application.held and not planner.active
        with pytest.raises(MujocoTcpPlanRejected, match="no_active_plan"):
            planner.reference(0)
        held_q = application.runtime.rig.measured_qpos_by_side()[motion_side].copy()
        observation = adapter.get_observation_snapshot()
        target = list(
            observation["measurement_summary"]["pose_xyz_quat_xyzw_by_component"][
                component_id
            ]
        )
        target[2] += 0.03
        restart_trace_index = len(facts)
        second = adapter.submit_skill_request(
            _request(session, adapter, target, sequence=1).to_wire()
        )
        assert second["accepted"], second
        assert session.wait(timeout_sec=35)
        second_operation = session.service.operation_store.get(
            second["operation"]["operation_id"]
        )
        assert second_operation.status == "completed", second_operation
        assert second_operation.task_outcome == "succeeded", second_operation
        assert len(plans) == 2
        assert plans[0].trajectory_hash != plans[1].trajectory_hash
        np.testing.assert_allclose(
            plans[1].path.start_qpos_by_component[motion_side], held_q, atol=1e-6
        )
        restarted_references = [
            fact.payload["payload"]["output_values_by_effector"][component_id]
            for fact in facts[restart_trace_index:]
            if fact.event_type == "component_execution_target"
        ]
        np.testing.assert_allclose(restarted_references[0], held_q, atol=1e-6)
        assert application.held and not planner.active
        record_property(
            "cancel_and_replan_json",
            json.dumps(
                {
                    "motion_side": motion_side,
                    "component_id": component_id,
                    "cancel_receipt": cancelled["port_receipt"],
                    "cancelled_operation_id": first_operation.operation_id,
                    "new_operation_id": second_operation.operation_id,
                    "trajectory_hashes": [plan.trajectory_hash for plan in plans],
                    "held_qpos": held_q.tolist(),
                    "new_plan_initial_qpos": plans[1].trajectory.q_rad[0].tolist(),
                    "hardware_used": False,
                }
            ),
        )
    finally:
        assert session.close().accepted


def _exercise_safe_gateway_close_fences_an_inflight_request(record_property, motion_side):
    session = _session([], motion_side=motion_side)
    try:
        component_id = session.application.component_id
        assert session.application.motion_side == motion_side
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        target = list(
            adapter.get_observation_snapshot()["measurement_summary"][
                "pose_xyz_quat_xyzw_by_component"
            ][component_id]
        )
        target[2] += 0.08
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target).to_wire()
        )
        assert admitted["accepted"]
        outcome = session.close()
        assert outcome.accepted
        assert outcome.port_receipt.proves_motion_or_hold
        assert session.application.closed
        assert not session.application.target.planned_tcp_motion.active
        assert session.safe_tcp_pose_executor.wait(timeout_sec=0)
        assert all(
            worker.state in {"finished", "closed"}
            for worker in session.safe_tcp_pose_executor.workers
        )
        record_property(
            "close_evidence",
            json.dumps(
                {
                    "motion_side": motion_side,
                    "component_id": component_id,
                    "accepted": outcome.accepted,
                    "lower_hold_proved": outcome.port_receipt.proves_motion_or_hold,
                }
            ),
        )
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_cancelled_safe_move_replans_from_measured_hold_without_old_references(monkeypatch, record_property):
    _exercise_cancelled_safe_move_replans_from_measured_hold_without_old_references(monkeypatch, record_property, "right")


@pytest.mark.integration
def test_left_cancelled_safe_move_replans_from_measured_hold_without_old_references(monkeypatch, record_property):
    _exercise_cancelled_safe_move_replans_from_measured_hold_without_old_references(monkeypatch, record_property, "left")


@pytest.mark.integration
def test_safe_gateway_close_fences_an_inflight_request(record_property):
    _exercise_safe_gateway_close_fences_an_inflight_request(record_property, "right")


@pytest.mark.integration
def test_left_safe_gateway_close_fences_an_inflight_request(record_property):
    _exercise_safe_gateway_close_fences_an_inflight_request(record_property, "left")


@pytest.mark.integration
@pytest.mark.parametrize("policy", [None, "within_arrival_tolerance", "preserve_target"])
@pytest.mark.parametrize("motion_side", ["left", "right"])
@pytest.mark.parametrize("task", [TASK, Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")])
def test_safe_tcp_consumes_planned_joints_without_re_ik_pose_goal_policy(monkeypatch, record_property, policy, motion_side, task):
    import agent_gateway.profiles.mujoco.motion_skill as lifecycle_module
    terminal_hash_inputs = []
    original_hash = lifecycle_module.canonical_content_hash
    def capture_terminal_hash(value, *, domain):
        digest = original_hash(value, domain=domain)
        if domain == "dexteleop.agent_skill.mujoco_tcp_terminal_state.v1":
            terminal_hash_inputs.append((dict(value), digest))
        return digest
    monkeypatch.setattr(lifecycle_module, "canonical_content_hash", capture_terminal_hash)
    facts = []
    session = _session(facts, motion_side=motion_side, task=task)
    plans = []
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        initial = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"
        ][motion_side + "_arm"]
        target = list(initial)
        target[2] += 0.03
        planner = session.application.target.planned_tcp_motion
        original = planner.plan

        def capture_plan(*args, **kwargs):
            profile = session.contract_bundle.motion_profile
            assert kwargs["translation_tolerance_m"] == (
                min(profile.translation_tolerance_m, .003) if policy == "preserve_target"
                else profile.translation_tolerance_m)
            assert kwargs["rotation_tolerance_rad"] == pytest.approx(
                min(profile.rotation_tolerance_rad, np.deg2rad(1)) if policy == "preserve_target"
                else profile.rotation_tolerance_rad)
            try:
                plan = original(*args, **kwargs)
            except MujocoTcpPlanRejected as error:
                record_property(
                    "planning_rejection_json", json.dumps(error.details, default=str)
                )
                raise
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture_plan)
        monkeypatch.setattr(
            session.application.runtime.pose_resolver.ik,
            "solve",
            lambda *a, **kw: pytest.fail("Safe path must not be replaced by new IK."),
        )
        admitted = adapter.submit_skill_request(
            _request(session, adapter, target, policy=policy).to_wire()
        )
        assert admitted["accepted"]
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        if operation.status != "completed":
            record_property(
                "failure_trace_json",
                json.dumps([fact.to_payload() for fact in facts[-16:]], default=str),
            )
        assert operation.status == "completed", (
            f"{operation.reason_code}: {operation.progress_summary}"
        )
        assert operation.task_outcome == "succeeded", operation
        report = operation.execution_result["pose_goal_report"]
        assert report["pose_goal_policy"] == (policy or "within_arrival_tolerance")
        assert report["reference_frame_id"] == session.contract_bundle.motion_profile.reference_frame_id
        np.testing.assert_allclose(report["requested_pose_xyz_quat_xyzw"], target)
        assert report["translation_error_m"] <= report["arrival_translation_tolerance_m"]
        assert report["rotation_error_rad"] <= report["arrival_rotation_tolerance_rad"]
        assert len(terminal_hash_inputs) == 1
        preimage, terminal_hash = terminal_hash_inputs[0]
        assert operation.execution_result["final_measured_state_hash"] == terminal_hash
        assert preimage["component_id"] == motion_side + "_arm"
        assert preimage["reference_frame_id"] == report["reference_frame_id"]
        np.testing.assert_allclose(preimage["pose_xyz_quat_xyzw"], report["achieved_pose_xyz_quat_xyzw"])
        measured = np.asarray(report["achieved_pose_xyz_quat_xyzw"], dtype=float)
        requested = np.asarray(report["requested_pose_xyz_quat_xyzw"], dtype=float)
        distance = float(np.linalg.norm(measured[:3] - requested[:3]))
        qa, qb = measured[3:] / np.linalg.norm(measured[3:]), requested[3:] / np.linalg.norm(requested[3:])
        angle = 2. * float(np.arccos(np.clip(abs(np.dot(qa, qb)), 0., 1.)))
        assert report["translation_error_m"] == pytest.approx(distance, abs=1e-7)
        assert report["rotation_error_rad"] == pytest.approx(angle, abs=1e-7)
        assert len(plans) == 1
        plan = plans[0]
        rig_config = session.application.runtime.rig.config
        effective_velocity = np.minimum(
            rig_config.max_velocity_rad_s,
            np.deg2rad(rig_config.max_command_step_deg) * rig_config.timing.command_hz,
        )
        certificate = plan.trajectory.metadata["continuous_hard_limit_certificate"]
        assert np.all(
            np.asarray(certificate["max_abs_velocity_rad_s"])
            <= effective_velocity + 1e-7
        )
        references = [
            fact.payload["payload"]["output_values_by_effector"][motion_side + "_arm"]
            for fact in facts
            if fact.event_type == "component_execution_target"
        ]
        assert len(references) >= len(plan.trajectory.q_rad)
        np.testing.assert_allclose(
            references[: len(plan.trajectory.q_rad)], plan.trajectory.q_rad, atol=1e-7
        )
        assert any(
            "mujoco-safe-joint-trajectory:" in ref
            for ref in operation.execution_receipt_refs
        )
        final = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"
        ][motion_side + "_arm"]
        error = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
        assert error <= session.contract_bundle.motion_profile.translation_tolerance_m
        assert not planner.active
        record_property(
            "evidence_json",
            json.dumps(
                {
                    "plan_hash": plan.trajectory_hash,
                    "path_metadata": dict(plan.path.metadata),
                    "reference_count": len(references),
                    "planned_reference_count": len(plan.trajectory.q_rad),
                    "target_error_m": error,
                    "hardware_used": False,
                },
                default=str,
            ),
        )
    finally:
        outcome = session.close()
        assert outcome.accepted


@pytest.mark.integration
def test_safe_tcp_policy_rejects_invalid_values_before_planning_pose_goal_policy(monkeypatch):
    session = _session([])
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        pose = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"]["right_arm"]
        planner = session.application.target.planned_tcp_motion
        monkeypatch.setattr(planner, "plan", lambda *_, **__: pytest.fail("invalid policy reached planner"))
        request = _request(session, adapter, pose)
        route = session.contract_bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1").route
        schema = session.contract_bundle.schema_catalog.entry("motion.move_tcp_to_pose_safe", "1")
        assert schema.parameter_schema["properties"]["pose_goal_policy"]["default"] == "within_arrival_tolerance"
        assert route.parameter_schema_hash == schema.parameter_schema_hash
        for index, policy in enumerate(("invalid", "", None, True, [], {})):
            bad = replace(request, request_id=f"invalid-policy:{index}", idempotency_key=f"invalid-policy:{index}",
                parameters={**request.parameters, "pose_goal_policy": policy})
            result = adapter.submit_skill_request(bad.to_wire())
            assert not result["accepted"], result
        forged = replace(request, request_id="forged-policy-profile", idempotency_key="forged-policy-profile",
            parameter_profile_hash="0" * 64,
            parameters={**request.parameters, "pose_goal_policy": "preserve_target"})
        result = adapter.submit_skill_request(forged.to_wire())
        assert not result["accepted"], result
        assert not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("motion_side", ["left", "right"])
def test_preserve_target_reports_existing_arrival_region_without_claiming_exact_convergence(
    monkeypatch, motion_side, record_property,
):
    facts = []
    session = _session(facts, motion_side=motion_side)
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        component = motion_side + "_arm"
        target = list(adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"][component])
        target[2] += .006
        request = _request(session, adapter, target, policy="preserve_target")
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.status == "completed" and operation.task_outcome == "succeeded", operation
        report = operation.execution_result["pose_goal_report"]
        assert .003 < report["translation_error_m"] <= report["arrival_translation_tolerance_m"]
        assert not any(f.event_type == "component_execution_target" for f in facts)
        session.contract_bundle.payload_validators.validate_execution_result(
            route=session.contract_bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1").route,
            payload=operation.execution_result)
        record_property("pose_goal_report_json", json.dumps(dict(report)))
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("anchor, expected", [([-.209302, .001432, .071703], "path_found"),
                                             ([10., 10., 10.], "no_path_found")])
def test_object_path_query_is_passive_and_does_not_own_execution(anchor, expected, record_property):
    from agent_gateway.skills.perception.object_goal_query import SKILL_ID

    facts = []
    session = _session(facts, task=Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
                       safe_profile="object_relation", queries=True)
    try:
        application = session.application
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        entry = session.contract_bundle.route_manifest.route(SKILL_ID, "1")
        profile = entry.route.allowed_parameter_profiles[0]
        assert entry.route.control_effect_class == "read_only"
        assert entry.route.cancellation_policy == "cancel_only"
        live_planner = application.target.planned_tcp_motion
        pending = live_planner._pending_plan
        safety = application.tcp_pose_envelope_runtime.snapshot()
        before = (application.scene.data.qpos.copy(), application.scene.data.qvel.copy(), float(application.scene.data.time))
        facts_before = len(facts)
        request = replace(_request(session, adapter, before[0][:7].tolist(), timeout_ms=entry.route.max_operation_duration_ms),
            skill_id=SKILL_ID, skill_definition_hash=entry.route.skill_definition_hash,
            route_id=entry.route.route_id, route_hash=entry.route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
            parameters={"component_id": "right_arm", "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id,
                "anchor_xyz_m": anchor, "heights_m": [.125], "tool_yaw_offsets_rad": [-np.pi/2], "time_budget_sec": 8.})
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.object_goal_query_executor.wait_all(timeout_sec=15)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "completed", (operation, session.object_goal_query_executor.failure_diagnostics)
        result = operation.to_wire()["execution_result"]
        record_property("query_evidence_json", json.dumps(result))
        assert result["query_status"] == expected
        assert result["motion_executed"] is False and result["execution_authorized"] is False
        assert result["all_candidates_checked"] and result["checked_count"] == 1
        np.testing.assert_array_equal(application.scene.data.qpos, before[0])
        np.testing.assert_array_equal(application.scene.data.qvel, before[1])
        assert float(application.scene.data.time) == before[2]
        assert application.tcp_pose_envelope_runtime.snapshot() == safety
        assert live_planner._pending_plan is pending and not live_planner.active
        assert len(facts) == facts_before
        validator = session.contract_bundle.object_goal_query_registration.payload_validator
        validator.validate_execution_result_for_parameters(request.parameters, result)
        with pytest.raises(ValueError):
            validator.validate_execution_result_for_parameters(request.parameters, {**result, "execution_authorized": True})
        with pytest.raises(ValueError):
            validator.validate_execution_result_for_parameters(request.parameters, {**result, "checked_count": 2})
        with pytest.raises(ValueError):
            validator.validate_execution_result_for_parameters(request.parameters, {**result, "anchor_xyz_m": [1, 2, 3]})
    finally:
        assert session.close().accepted


def test_tcp_objective_revision_changes_only_precise_profile_hash(monkeypatch):
    from agent_gateway.profiles.mujoco import motion_deployment as deployment

    profile = deployment.MujocoAgentTcpPoseMotionProfile("ref", "ref-spec", "graph", "tool", "calibration")
    original_hash = deployment.canonical_content_hash
    payloads = []

    def capture(payload, **kwargs):
        payloads.append(dict(payload))
        return original_hash(payload, **kwargs)

    monkeypatch.setattr(deployment, "canonical_content_hash", capture)
    default_hash = profile.semantic_content_hash
    assert "ik_objective_mode" not in payloads[-1]
    assert default_hash == original_hash(payloads[-1], domain="dexteleop.agent_skill.mujoco_tcp_pose_profile.v1")
    precise_hash = replace(profile, normalize_ik_objective=True).semantic_content_hash
    assert payloads[-1]["ik_objective_mode"] == "calibrated_tcp_tolerance_normalized"
    old = {**payloads[-1], "ik_objective_mode": "tolerance_normalized"}
    assert precise_hash != original_hash(old, domain="dexteleop.agent_skill.mujoco_tcp_pose_profile.v1")


@pytest.mark.parametrize("name,enabled", [("unknown", True), ("object_relation", False),
                                          ("object_relation_precise", False)])
def test_invalid_safe_motion_profile_fails_before_scene_preparation(monkeypatch, name, enabled):
    monkeypatch.setattr("agent_gateway.profiles.mujoco.motion_factory.prepare_mujoco_sim_hil_target",
                        lambda *a, **kw: pytest.fail("Invalid profile must not prepare resources."))
    with pytest.raises(ValueError):
        build_mujoco_agent_tcp_pose_gateway_session(
            target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK)),
            allowed_client_ids=(CLIENT,), enable_safe_tcp_pose=enabled, safe_tcp_motion_profile=name)


@pytest.mark.parametrize("safe_profile,tolerance_m,angle_deg", [
    ("object_relation", .01, 5.), ("object_relation_precise", .003, 1.),
])
def test_object_relation_profile_changes_only_explicit_safe_route(safe_profile, tolerance_m, angle_deg):
    local = _session([])
    relation = _session([], safe_profile=safe_profile)
    try:
        local.application.start_observation_ready()
        relation.application.start_observation_ready()
        original = local.contract_bundle.safe_tcp_pose_contract
        expanded = relation.contract_bundle.safe_tcp_pose_contract
        assert original.motion_profile.max_translation_delta_m == .10
        assert original.motion_profile.max_rotation_delta_rad == pytest.approx(np.deg2rad(15))
        assert expanded.motion_profile.max_translation_delta_m == .50
        assert expanded.motion_profile.max_rotation_delta_rad == pytest.approx(np.pi)
        assert expanded.motion_profile.translation_tolerance_m == tolerance_m
        assert expanded.motion_profile.rotation_tolerance_rad == pytest.approx(np.deg2rad(angle_deg))
        assert expanded.motion_profile.normalize_ik_objective == (safe_profile == "object_relation_precise")
        assert expanded.motion_profile.profile_id == f"mujoco.agent_safe_tcp_pose.{safe_profile}.v1"
        assert original.motion_profile.translation_tolerance_m == .01
        assert original.motion_profile.rotation_tolerance_rad == pytest.approx(np.deg2rad(5))
        assert not original.motion_profile.normalize_ik_objective
        assert (replace(expanded.motion_profile, normalize_ik_objective=not expanded.motion_profile.normalize_ik_objective).semantic_content_hash
                != expanded.motion_profile.semantic_content_hash)
        assert original.motion_profile.semantic_content_hash != expanded.motion_profile.semantic_content_hash
        assert relation.contract_bundle.motion_profile.max_translation_delta_m == .10
        assert relation.contract_bundle.motion_profile.translation_tolerance_m == .01
        ordinary = "motion.move_tcp_to_pose"
        assert (local.contract_bundle.route_manifest.route(ordinary, "1").route.semantic_content_hash
                == relation.contract_bundle.route_manifest.route(ordinary, "1").route.semantic_content_hash)
        expanded_refs = expanded.deployment_binding.safety_envelope_refs
        assert expanded_refs and expanded_refs != original.deployment_binding.safety_envelope_refs
    finally:
        outcomes = (local.close(), relation.close())
        assert all(outcome.accepted for outcome in outcomes)


@pytest.mark.integration
@pytest.mark.parametrize("safe_profile,target", [
    ("object_relation", (-.173863, .001569, .196716, -.7260704636573792, .6876203417778015,
     3.828142905803149e-17, -3.6394808751810345e-17)),
    ("object_relation_precise", (-.173863, .001569, .196716, -.7260704636573792, .6876203417778015,
     3.828142905803149e-17, -3.6394808751810345e-17)),
    ("object_relation_precise", (-.161097, -.026571, .211707, .3574257791042328, .9339415431022644,
     -1.8974884668533564e-17, -4.9295067598934135e-17)),
], ids=["hover_default", "hover_precise", "retained_relation_offset_precise"])
def test_retained_agent_hover_precision_uses_deployed_contract(safe_profile, target, monkeypatch, record_property):
    """Replay retained Agent targets through native physics and measured readback."""
    from robot_motion.relative_action import xyzquat_to_pose_matrix

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    facts, plans = [], []
    session = _session(facts, task=task, safe_profile=safe_profile)
    try:
        application = session.application
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        profile = session.contract_bundle.safe_tcp_pose_contract.motion_profile
        planner = application.target.planned_tcp_motion
        original_plan = planner.plan

        def capture(*args, **kwargs):
            assert kwargs["translation_tolerance_m"] == profile.translation_tolerance_m
            assert kwargs["rotation_tolerance_rad"] == profile.rotation_tolerance_rad
            plan = original_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture)
        admitted = adapter.submit_skill_request(_request(session, adapter, target, timeout_ms=60_000).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=65)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        final = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        assert len(plans) == 1
        endpoint = application.transform_tcp_pose_xyzquat(plans[0].reference_poses[-1],
            source_frame_id=profile.execution_reference_frame_id, target_frame_id=profile.reference_frame_id)
        endpoint_error = float(np.linalg.norm(np.asarray(endpoint[:3]) - target[:3]))
        measured_error = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
        rotation = xyzquat_to_pose_matrix(final)[:3, :3].T @ xyzquat_to_pose_matrix(target)[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        record_property("hover_precision_evidence_json", json.dumps({
            "safe_profile": safe_profile, "profile_hash": profile.semantic_content_hash,
            "target": target, "planned_endpoint": endpoint, "measured_readback": final,
            "endpoint_error_m": endpoint_error, "readback_error_m": measured_error,
            "rotation_error_rad": angle, "translation_tolerance_m": profile.translation_tolerance_m,
            "rotation_tolerance_rad": profile.rotation_tolerance_rad,
            "trajectory_hash": plans[0].trajectory_hash,
            "duration_sec": float(plans[0].trajectory.timestamps[-1]),
            "max_joint_excursion_rad": float(np.max(np.abs(plans[0].trajectory.q_rad - plans[0].trajectory.q_rad[0]))),
            "operation": operation.to_wire(), "hardware_connected": False,
        }))
        assert operation.execution_outcome == "completed" and operation.task_outcome == "succeeded", operation
        assert endpoint_error <= profile.translation_tolerance_m
        assert measured_error <= profile.translation_tolerance_m
        assert angle <= profile.rotation_tolerance_rad
        assert application.held and any(fact.event_type == "dispatch_receipt" for fact in facts)
        if safe_profile == "object_relation_precise":
            assert measured_error <= .005 and angle <= np.deg2rad(2)
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("maximum_failure_case", [False, True], ids=["hover_18", "failure_64"])
def test_query_and_execution_share_precise_hover_contract(maximum_failure_case, record_property):
    from agent_gateway.skills.perception.object_goal_query import SKILL_ID
    from robot_motion.relative_action import xyzquat_to_pose_matrix

    session = _session([], task=Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"),
                       safe_profile="object_relation_precise", queries=True)
    try:
        application = session.application
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        entry = session.contract_bundle.route_manifest.route(SKILL_ID, "1")
        profile = session.contract_bundle.safe_tcp_pose_contract.motion_profile
        parameter = entry.route.allowed_parameter_profiles[0]
        request = replace(_request(session, adapter, [0.] * 7, timeout_ms=entry.route.max_operation_duration_ms),
            skill_id=SKILL_ID, skill_definition_hash=entry.route.skill_definition_hash,
            route_id=entry.route.route_id, route_hash=entry.route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter.parameter_profile_id, parameter_profile_hash=parameter.parameter_profile_hash,
            parameters={"component_id": "right_arm", "reference_frame_id": profile.reference_frame_id,
                "anchor_xyz_m": [-.173863, .001569, .071716],
                "heights_m": [round(.25 + i*.01, 2) for i in range(8)] if maximum_failure_case else [.125, .1375, .15],
                "tool_yaw_offsets_rad": np.linspace(-np.pi, np.pi, 8, endpoint=False).tolist() if maximum_failure_case
                    else [0., .7854, -.7854, 1.5708, -1.5708, 3.1416], "time_budget_sec": 12.})
        before = (application.scene.data.qpos.copy(), application.scene.data.qvel.copy(), float(application.scene.data.time))
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.object_goal_query_executor.wait_all(timeout_sec=20)
        query = session.service.operation_store.get(admitted["operation"]["operation_id"])
        result = query.to_wire()["execution_result"]
        record_property("precise_query_json", json.dumps(result))
        expected_count = 64 if maximum_failure_case else 18
        assert query.execution_outcome == "completed", query
        assert result["candidate_count"] == result["checked_count"] == expected_count and result["all_candidates_checked"]
        assert len(json.dumps(result, separators=(",", ":")).encode()) <= 65_536
        np.testing.assert_array_equal(application.scene.data.qpos, before[0])
        np.testing.assert_array_equal(application.scene.data.qvel, before[1])
        assert float(application.scene.data.time) == before[2] and not application.target.planned_tcp_motion.active
        if maximum_failure_case:
            assert result["query_status"] == "no_path_found"
            assert any("endpoint_error_pairs_m_rad" in item.get("details", {}) for item in result["candidate_diagnostics"])
            return
        assert result["query_status"] == "path_found"
        target = result["pose_targets_by_component"]["right_arm"]
        admitted = adapter.submit_skill_request(_request(session, adapter, target, sequence=1, timeout_ms=60_000).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=65)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        final = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        error = float(np.linalg.norm(np.asarray(final[:3]) - target[:3]))
        rotation = xyzquat_to_pose_matrix(final)[:3, :3].T @ xyzquat_to_pose_matrix(target)[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        record_property("precise_execution_json", json.dumps({"target": target, "readback": final,
            "readback_error_m": error, "rotation_error_rad": angle, "operation": operation.to_wire()}))
        assert operation.execution_outcome == "completed" and operation.task_outcome == "succeeded", operation
        assert error <= profile.translation_tolerance_m < .005
        assert angle <= profile.rotation_tolerance_rad < np.deg2rad(2.)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_observed_object_goal_region_selects_shorter_path_then_executes_existing_skill(record_property, monkeypatch):
    """Replay a retained model-observed surface point; this is not new perception."""
    import time

    from dataset_contracts.canonical_encoding import canonical_content_hash
    from robot_motion.path_planning import JointConfigurationCheck
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from robot_motion.spatial_goals import generate_object_above_goals

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    facts = []
    session = _session(facts, task=task, safe_profile="object_relation")
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        initial = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        candidates = generate_object_above_goals(current_tcp_pose_xyz_quat_xyzw=initial,
            anchor_xyz_m=[-.209302, .001432, .071703], heights_m=[.10, .125, .15],
            tool_yaw_offsets_rad=np.deg2rad([0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]).tolist())
        profile = session.contract_bundle.safe_tcp_pose_contract.motion_profile
        application = session.application
        planner = application.target.planned_tcp_motion
        goals = [application.transform_tcp_pose_xyzquat(candidate.goal.pose_xyz_quat_xyzw,
            source_frame_id=profile.reference_frame_id, target_frame_id=profile.execution_reference_frame_id)
            for candidate in candidates]
        before = (application.scene.data.qpos.copy(), application.scene.data.qvel.copy(), float(application.scene.data.time))
        envelope = application.tcp_pose_envelope_runtime
        envelope_before = envelope.snapshot()
        route_id = session.contract_bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1").route.route_id
        project_envelope = envelope.planning_projector_for_route(route_id=route_id)

        def constraint(model_qpos, pair):
            pose = planner.pose_for_configuration(model_qpos, pair)
            projected = project_envelope({"right": xyzquat_to_pose_matrix(pose)})
            return JointConfigurationCheck(safe=not projected.modified_sides,
                reason="tcp_envelope_blocked" if projected.modified_sides else None,
                metadata=projected.to_payload())

        started = time.perf_counter()
        selection = planner.plan_candidates(goals, translation_tolerance_m=profile.translation_tolerance_m,
            rotation_tolerance_rad=profile.rotation_tolerance_rad, cancel_requested=lambda: False,
            time_budget_sec=30., configuration_constraint=constraint,
            constraint_identity=canonical_content_hash({"profile": profile.semantic_content_hash,
                "reference": envelope_before.safety_reference_hash}, domain="test.object_relation.preview.v1"))
        planning_wall_sec = time.perf_counter()-started
        np.testing.assert_array_equal(application.scene.data.qpos, before[0])
        np.testing.assert_array_equal(application.scene.data.qvel, before[1])
        assert float(application.scene.data.time) == before[2] and not planner.active
        assert envelope.snapshot() == envelope_before
        costs = [item["ranking_cost"] for item in selection.diagnostics if item["reason_code"] == "path_found"]
        assert costs
        selected_cost = min(costs)
        q = selection.plan.trajectory.q_rad
        assert float(np.max(np.abs(q-q[0]))) == pytest.approx(selected_cost[0])
        # The original zero-yaw, 15 cm proposal needed 154.6 deg on one joint.
        assert np.rad2deg(selected_cost[0]) < 154.0
        target = candidates[selection.candidate_index].goal.pose_xyz_quat_xyzw
        plans = []
        original_plan = planner.plan

        def capture(*args, **kwargs):
            plan = original_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture)
        execution_started = time.perf_counter()
        sim_started = float(application.scene.data.time)
        admitted = adapter.submit_skill_request(_request(session, adapter, target, timeout_ms=60_000).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=65)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        execution_wall_sec = time.perf_counter()-execution_started
        sim_elapsed_sec = float(application.scene.data.time)-sim_started
        final = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"]
        position_error_m = float(np.linalg.norm(np.asarray(final[:3])-target[:3]))
        rotation_dot = abs(np.dot(final[3:], target[3:])) / (np.linalg.norm(final[3:])*np.linalg.norm(target[3:]))
        rotation_error_rad = float(2*np.arccos(np.clip(rotation_dot, 0, 1)))
        dispatches = [fact for fact in facts if fact.event_type == "dispatch_receipt"]
        measured_q = [fact.payload["measurement"]["measured_qpos_by_side"]["right"]
                      for fact in facts if fact.event_type == "measurement_alignment"]
        evidence = {"hardware_connected": False, "retained_anchor_xyz_m": [-.209302, .001432, .071703],
            "initial_tcp": initial, "selected_target": target, "final_tcp": final,
            "height_m": candidates[selection.candidate_index].height_m,
            "tool_yaw_rad": candidates[selection.candidate_index].tool_yaw_rad,
            "candidate_count": len(candidates), "all_candidates_checked": selection.all_candidates_checked,
            "candidate_diagnostics": selection.diagnostics, "planning_wall_sec": planning_wall_sec,
            "max_joint_excursion_deg": float(np.rad2deg(selected_cost[0])),
            "planned_duration_sec": selected_cost[1], "operation": operation.to_wire(),
            "execution_wall_sec": execution_wall_sec, "execution_sim_elapsed_sec": sim_elapsed_sec,
            "position_error_m": position_error_m, "rotation_error_rad": rotation_error_rad,
            "dispatch_count": len(dispatches), "native_grasp_success_asserted": False,
            "measured_joint_path_rad": measured_q,
            "execution_plan_hashes": [plan.trajectory_hash for plan in plans]}
        record_property("object_goal_evidence_json", json.dumps(evidence, default=str))
        assert operation.execution_outcome == "completed", operation
        assert operation.task_outcome == "succeeded", operation
        assert len(plans) == 1 and not planner.active
        assert position_error_m <= profile.translation_tolerance_m
        assert rotation_error_rad <= profile.rotation_tolerance_rad
        assert dispatches and all(fact.payload["all_accepted"] for fact in dispatches)
        assert measured_q and application.held
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("constraint", [None, {
    "kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": .17453292519943295,
}])
def test_safe_tcp_v2_preserves_sampled_axis_and_retains_null_behavior(constraint, record_property):
    facts = []
    session = _session(facts, path_constraints=True)
    try:
        application = session.application
        application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        target = list(adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"])
        target[2] += .03
        request = _request(session, adapter, target, version="2", path_constraint=constraint)
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "completed", (operation.reason_code, operation.progress_summary)
        evidence = operation.execution_result["path_constraint_evidence"]
        if constraint is None:
            assert evidence is None
        else:
            assert evidence["planned_max_deviation_rad"] <= constraint["max_deviation_rad"] + 1e-9
            assert evidence["scope"] == "sampled_planned_and_live_references"
            assert evidence["validated_reference_count"] > 1
            assert np.linalg.norm(evidence["activation_axis_in_reference_frame"]) == pytest.approx(1.)
        entry = session.contract_bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "2")
        validator = session.contract_bundle.payload_validators.validator_for(entry.route)
        validator.validate_execution_result_for_parameters(request.parameters, operation.execution_result)
        record_property("path_constraint_evidence_json", json.dumps(
            operation.to_wire()["execution_result"]["path_constraint_evidence"]))
        assert any("mujoco-safe-joint-trajectory:" in ref for ref in operation.execution_receipt_refs)
        assert not application.target.planned_tcp_motion.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_safe_tcp_v2_endpoint_outside_axis_cone_is_rejected_without_dispatch(record_property):
    from scipy.spatial.transform import Rotation

    facts = []
    session = _session(facts, path_constraints=True)
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        target = list(adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"])
        initial = Rotation.from_quat(target[3:])
        target[3:] = (initial * Rotation.from_euler("x", 10., degrees=True)).as_quat().tolist()
        constraint = {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": .08726646259971647}
        admitted = adapter.submit_skill_request(_request(
            session, adapter, target, version="2", path_constraint=constraint).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "failed"
        assert operation.progress_summary["lower_hold_confirmed"]
        assert not any(fact.event_type == "component_execution_target" for fact in facts)
        assert not session.application.target.planned_tcp_motion.active
        record_property("rejection_json", json.dumps(operation.to_wire(), default=str))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_safe_tcp_v2_rechecks_actual_to_reference_axis_before_dispatch(monkeypatch, record_property):
    from agent_gateway.profiles.mujoco.motion_skill import (
        mujoco_agent_skill_hold_evidence_from_transition,
    )
    from robot_motion.path_constraints import ToolAxisCone

    facts, injected, holds = [], {}, []
    session = _session(facts, path_constraints=True)
    try:
        app = session.application
        app.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        planner = app.target.planned_tcp_motion
        original_hold = app.request_hold

        def remember_hold(**kwargs):
            transition = original_hold(**kwargs)
            holds.append(mujoco_agent_skill_hold_evidence_from_transition(
                transition, lower_operation_id=kwargs.get("operation_id")))
            return transition

        monkeypatch.setattr(app, "request_hold", remember_hold)
        target = list(adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"])
        target[2] += .03
        original_activate = planner.activate

        def displace_simulated_plant_after_activation(plan):
            original_activate(plan)
            source = planner._model_qpos()
            oracle = MujocoJointSceneOracle(app.scene, source)
            pair = {side: source[address].copy() for side, address in oracle.addresses.items()}
            cone = ToolAxisCone.from_pose(planner.pose_for_configuration(source, pair),
                tool_axis="z", max_deviation_rad=np.deg2rad(5.))
            # Model an unexpected plant displacement, not a command. Pick a
            # collision-free joint perturbation so the axis check is decisive.
            for index in reversed(range(len(pair["right"]))):
                candidate = {side: q.copy() for side, q in pair.items()}
                candidate["right"][index] += .25
                deviation = cone.deviation_rad(planner.pose_for_configuration(source, candidate))
                if deviation > cone.max_deviation_rad and oracle.check(candidate).safe:
                    app.scene.data.qpos[oracle.addresses["right"]] = candidate["right"]
                    app.scene.mujoco.mj_forward(app.scene.model, app.scene.data)
                    injected.update(deviation_rad=deviation, joint_index=index,
                        constraint_identity=plan.constraint_identity)
                    return
            raise AssertionError("No collision-free axis-violating test perturbation")

        monkeypatch.setattr(planner, "activate", displace_simulated_plant_after_activation)
        policy = {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": np.deg2rad(5.)}
        admitted = adapter.submit_skill_request(_request(
            session, adapter, target, version="2", path_constraint=policy).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert injected
        assert operation.execution_outcome == "failed", operation
        assert operation.reason_code == "safe_tcp_live_reference_constraint_blocked", operation
        assert app.held and any(hold.lower_hold_confirmed for hold in holds)
        assert not any(fact.event_type == "component_execution_target" for fact in facts)
        assert not planner.active
        record_property("live_axis_rejection_json", json.dumps({
            "injected_simulator_state_only": injected, "operation": operation.to_wire()}))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_safe_tcp_v2_narrow_cone_accounts_for_published_float32_qref(record_property):
    from robot_motion.path_constraints import ToolAxisCone

    facts = []
    session = _session(facts, path_constraints=True)
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        oracle = MujocoJointSceneOracle(app.scene, source)
        pair = {side: source[address].copy() for side, address in oracle.addresses.items()}
        rounding_deviation = None
        for index in reversed(range(len(pair["right"]))):
            candidate = {side: q.copy() for side, q in pair.items()}
            candidate["right"][index] += .123456789
            cone = ToolAxisCone.from_pose(planner.pose_for_configuration(source, candidate),
                tool_axis="z", max_deviation_rad=1e-12)
            rounded = {side: q.astype(np.float32) for side, q in candidate.items()}
            deviation = cone.deviation_rad(planner.pose_for_configuration(source, rounded))
            if deviation > 2e-9 and oracle.check(candidate).safe:
                app.scene.data.qpos[oracle.addresses["right"]] = candidate["right"]
                app.scene.mujoco.mj_forward(app.scene.model, app.scene.data)
                rounding_deviation = deviation
                break
        assert rounding_deviation is not None
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        target = list(adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"]["right_arm"])
        target[2] += .03
        policy = {"kind": "preserve_start_tool_axis", "tool_axis": "z", "max_deviation_rad": 1e-12}
        admitted = adapter.submit_skill_request(_request(
            session, adapter, target, version="2", path_constraint=policy).to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "failed", operation
        assert operation.reason_code == "safe_tcp_start_blocked", operation
        assert operation.progress_summary["lower_hold_confirmed"]
        assert not any(fact.event_type == "component_execution_target" for fact in facts)
        record_property("float32_constraint_json", json.dumps({
            "rounding_deviation_rad": rounding_deviation, "policy": policy,
            "operation": operation.to_wire(), "hardware_used": False}))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_continuation_query_rejects_retained_dead_end_before_motion(record_property):
    from agent_gateway.skills.perception.tcp_continuation_query import SKILL_ID

    facts = []
    session = _session(facts, safe_profile="object_relation_precise", path_constraints=True,
        continuations=True, task=Path("sim/configs/tasks/rm65_right_dmtacm_clutter_boxes_agent_v1.json"))
    try:
        app = session.application
        app.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        entry = session.contract_bundle.route_manifest.route(SKILL_ID, "1")
        parameter_profile = entry.route.allowed_parameter_profiles[0]
        live = app.target.planned_tcp_motion
        before = (app.scene.data.qpos.copy(), app.scene.data.qvel.copy(), float(app.scene.data.time),
            app.tcp_pose_envelope_runtime.snapshot(), len(facts), live._pending_plan)
        # First candidate replays the retained first-stage endpoint, not its
        # future measurement. The second leaves room for the required +3 cm.
        pose = [-.17339, -.028584, .211701, -.7260692119598389,
                .6876216530799866, 3.8281362883582485e-17, -3.63948782349818e-17]
        parameters = {"component_id": "right_arm",
            "reference_frame_id": session.contract_bundle.motion_profile.reference_frame_id,
            "first_pose_candidates_xyz_quat_xyzw": [pose, [pose[0], pose[1]-.04, pose[2], *pose[3:]]],
            "continuation_delta_xyz_m": [0, 0, .03],
            "path_constraint": {"kind": "preserve_start_tool_axis", "tool_axis": "z",
                                "max_deviation_rad": np.deg2rad(10.)}, "time_budget_sec": 30.}
        request = replace(_request(session, adapter, pose, timeout_ms=40_000),
            skill_id=SKILL_ID, skill_definition_hash=entry.route.skill_definition_hash,
            route_id=entry.route.route_id, route_hash=entry.route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter_profile.parameter_profile_id,
            parameter_profile_hash=parameter_profile.parameter_profile_hash, parameters=parameters)
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.tcp_continuation_query_executor.wait_all(timeout_sec=40)
        operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "completed", (operation,
            session.tcp_continuation_query_executor.failure_diagnostics)
        result = operation.to_wire()["execution_result"]
        record_property("continuation_evidence_json", json.dumps(result))
        assert result["query_status"] == "path_found" and result["selected_index"] == 1, result
        diagnostics = {row["candidate_index"]: row for row in result["candidate_diagnostics"]}
        assert diagnostics[0]["failed_stage"] == "continuation", result
        assert diagnostics[0]["stage_diagnostics"][0]["reason_code"] == "path_found"
        assert diagnostics[1]["reason_code"] == "path_found" and len(diagnostics[1]["stage_diagnostics"]) == 2
        np.testing.assert_allclose(np.asarray(result["continuation_pose_target_xyz_quat_xyzw"][:3])
            - result["predicted_first_tcp_pose_xyz_quat_xyzw"][:3], [0, 0, .03], atol=1e-12)
        assert result["continuation_source"] == "planned_first_endpoint"
        assert not result["motion_executed"] and not result["execution_authorized"]
        np.testing.assert_array_equal(app.scene.data.qpos, before[0])
        np.testing.assert_array_equal(app.scene.data.qvel, before[1])
        assert float(app.scene.data.time) == before[2]
        assert app.tcp_pose_envelope_runtime.snapshot() == before[3]
        assert len(facts) == before[4] and live._pending_plan is before[5] and not live.active
    finally:
        assert session.close().accepted


def test_continuation_query_requires_matching_safe_v2_before_resources():
    with pytest.raises(ValueError, match="Continuation queries require"):
        _session([], continuations=True)


@pytest.mark.parametrize("outcome", ["success", "cancel", "budget", "fault", "rank_pruned", "prefix_failed",
    "source_start_fault", "binding_fault", "continuation_start_blocked", "target_envelope_blocked"])
@pytest.mark.parametrize("relation_input", [False, True, "region"])
def test_continuation_query_propagates_joint_state_and_clears_only_private_plans(monkeypatch, outcome, relation_input):
    from agent_gateway.profiles.mujoco import object_goal_query as module

    now, cancelled, contexts = [0.], [False], []
    source = np.array([9., 1., 42.])
    captured, invocations = [], []
    # Region sampling checks the centre before the lower endpoint, unlike the
    # historical explicit list. Both still exercise the identical planning loop.
    alternative = -4. if relation_input == "region" else 8.

    class Planner:
        def __init__(self, **kwargs):
            self.source = kwargs["source_model_qpos"].copy()
            self.cleared = False
            contexts.append(self)

        def pose_for_configuration(self, qpos, pair):
            return (float(pair["right"][0]), 0., 0., 0., 0., 0., 1.)

        def plan(self, target, **kwargs):
            invocations.append((self.source[1], target[0], kwargs.get("incumbent_max_joint_excursion_rad")))
            np.testing.assert_array_equal(self.source[[0, 2]], source[[0, 2]])
            assert kwargs["deadline_monotonic"] == 1.
            if self.source[1] != 1.:
                assert "incumbent_max_joint_excursion_rad" not in kwargs
            if self.source[1] == 1. and target[0] == alternative:
                if outcome == "rank_pruned":
                    assert kwargs["incumbent_max_joint_excursion_rad"] == 2.
                    raise MujocoTcpPlanRejected("safe_tcp_candidate_rank_pruned", {
                        "lower_bound_max_joint_excursion_rad": abs(alternative - 1.),
                        "incumbent_max_joint_excursion_rad": 2., "path_search_skipped": True})
                assert "incumbent_max_joint_excursion_rad" not in kwargs
            if self.source[1] == 1. and target[0] == 2.:
                if outcome == "source_start_fault":
                    raise MujocoTcpPlanRejected("safe_tcp_start_blocked")
                if outcome == "binding_fault":
                    raise MujocoTcpPlanRejected("safe_tcp_envelope_binding_changed")
                if outcome == "target_envelope_blocked":
                    raise MujocoTcpPlanRejected("safe_tcp_target_envelope_blocked")
            if self.source[1] == 2:
                if outcome == "cancel":
                    cancelled[0] = True
                    raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
                if outcome == "fault":
                    raise RuntimeError("unexpected solver failure")
                if outcome == "prefix_failed":
                    raise MujocoTcpPlanRejected("safe_tcp_target_blocked")
                if outcome == "continuation_start_blocked":
                    raise MujocoTcpPlanRejected("safe_tcp_start_blocked")
            if outcome == "budget":
                now[0] = 2.
            return SimpleNamespace(reference_poses=[target, target], trajectory_hash="a"*64,
                phase_timing_sec={"joint_path_search": .01},
                path=SimpleNamespace(metadata={"direct_path": True, "search_iterations": 0,
                    "search_node_count": 2}),
                trajectory=SimpleNamespace(q_rad=np.array([[self.source[1]], [target[0]]]),
                                           timestamps=np.array([0., 1.])))

        def clear(self):
            self.cleared = True

    monkeypatch.setattr(module, "MujocoPlannedTcpMotion", Planner)
    monkeypatch.setattr(module, "MujocoJointSceneOracle", lambda *a, **kw: SimpleNamespace(scene_hash="b"*64))
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    scene = SimpleNamespace(arm_specs={side: SimpleNamespace(joint_names=[name])
                                      for side, name in (("left", "0"), ("right", "1"))}, joint_qpos_addr=int)

    def capture(callback):
        captured.append(True)
        return callback(SimpleNamespace(data=SimpleNamespace(qpos=source, time=0.)))

    query = module.MujocoObjectGoalQuery(scene=scene, rig_config=None, task_config=None,
        pose_projector=None, reference_hz=60., capture_scene=capture,
        transform_pose=lambda pose, **kw: pose,
        prepare_projection=lambda: (lambda poses: SimpleNamespace(modified_sides=()), "envelope:test"),
        profile=SimpleNamespace(reference_frame_id="station", execution_reference_frame_id="execution",
            max_translation_delta_m=10., max_rotation_delta_rad=3.14, stable_measurement_count=1,
            max_control_cycles=10, translation_tolerance_m=.003, rotation_tolerance_rad=.02,
            normalize_ik_objective=True), clearance_padding_m=.002)
    parameters = {"component_id": "right_arm", "reference_frame_id": "station",
        "first_pose_candidates_xyz_quat_xyzw": [[2., 0, 0, 0, 0, 0, 1]],
        "continuation_delta_xyz_m": [1., 0, 0], "path_constraint": None, "time_budget_sec": 1.}
    if outcome in {"rank_pruned", "prefix_failed", "continuation_start_blocked", "target_envelope_blocked"}:
        parameters["first_pose_candidates_xyz_quat_xyzw"].append([alternative, 0, 0, 0, 0, 0, 1])
    expected_poses = parameters["first_pose_candidates_xyz_quat_xyzw"]
    invoke = query.query_continuations
    if relation_input == "region":
        parameters = {key: value for key, value in parameters.items() if key != "first_pose_candidates_xyz_quat_xyzw"}
        parameters.update(anchor_xyz_m=[0., 0., 0.], offset_frame="tool", orientation_mode="preserve",
            offset_bounds_xyz_m=[[-4., 8.] if len(expected_poses) == 2 else [2., 2.], [0., 0.], [0., 0.]],
            tool_axis=None, axis_reference_xyz=None, tool_axis_rotation_bounds_rad=[0., 0.],
            max_candidates=len(expected_poses))
        invoke = query.query_region_continuations
    elif relation_input:
        parameters = {key: value for key, value in parameters.items() if key != "first_pose_candidates_xyz_quat_xyzw"}
        parameters.update(anchor_xyz_m=[0., 0., 0.], offset_frame="tool", orientation_mode="preserve",
            offsets_xyz_m=[pose[:3] for pose in expected_poses], tool_axis=None, axis_reference_xyz=None,
            tool_axis_rotation_offsets_rad=[0.])
        invoke = query.query_relation_continuations
    fatal_reasons = {"fault": "unexpected solver failure", "cancel": "safe_tcp_planning_cancelled",
        "source_start_fault": "safe_tcp_start_blocked", "binding_fault": "safe_tcp_envelope_binding_changed"}
    if outcome in fatal_reasons:
        with pytest.raises(RuntimeError, match=fatal_reasons[outcome]):
            invoke(parameters, cancel_requested=lambda: cancelled[0])
    else:
        result = invoke(parameters, cancel_requested=lambda: cancelled[0])
        if relation_input:
            assert result["first_pose_candidates_xyz_quat_xyzw"] == expected_poses
        assert result["query_status"] == ("no_path_found" if outcome == "budget" else "path_found")
        if outcome == "success":
            assert result["predicted_first_tcp_pose_xyz_quat_xyzw"][0] == 2.
            assert result["continuation_pose_target_xyz_quat_xyzw"][0] == 3.
            assert [item.source[1] for item in contexts] == [1., 2.]
        elif outcome == "target_envelope_blocked":
            assert result["attempted_count"] == 2 and result["all_candidates_attempted"]
            assert result["selected_index"] == 1
            assert invocations == [(1., 2., None), (1., alternative, None), (alternative, alternative + 1., None)]
            rejected = result["candidate_diagnostics"][0]
            assert rejected["reason_code"] == "safe_tcp_target_envelope_blocked"
            assert rejected["failed_stage"] == "first" and len(rejected["stage_diagnostics"]) == 1
        elif outcome in {"rank_pruned", "prefix_failed", "continuation_start_blocked"}:
            assert result["attempted_count"] == 2 and result["all_candidates_attempted"]
            assert invocations[:2] == [(1., 2., None), (2., 3., None)]
            first = result["candidate_diagnostics"][0]["stage_diagnostics"][0]
            assert first["details"]["phase_timing_sec"] == {"joint_path_search": .01}
            assert first["details"]["path_search"]["direct_path"] is True
            if outcome == "rank_pruned":
                assert result["selected_index"] == 0 and invocations[-1] == (1., alternative, 2.)
                pruned = result["candidate_diagnostics"][1]
                assert pruned["reason_code"] == "safe_tcp_candidate_rank_pruned"
                assert pruned["failed_stage"] == "first" and len(pruned["stage_diagnostics"]) == 1
                assert "trajectory_hash" not in pruned["stage_diagnostics"][0]
                assert len(contexts) == 2
            else:
                assert result["selected_index"] == 1
                assert invocations[-2:] == [(1., alternative, None), (alternative, alternative + 1., None)]
                rejected = result["candidate_diagnostics"][0]
                assert rejected["failed_stage"] == "continuation"
                assert rejected["reason_code"] == ("safe_tcp_start_blocked"
                    if outcome == "continuation_start_blocked" else "safe_tcp_target_blocked")
        else:
            assert result["candidate_diagnostics"][0]["reason_code"] == "query_budget_exhausted"
    assert captured == [True] and all(item.cleared for item in contexts)
    np.testing.assert_array_equal(source, [9., 1., 42.])
    assert not query._lock.locked()
