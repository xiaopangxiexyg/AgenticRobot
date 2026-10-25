from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

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
             contact_target=None, motion_side="right", use_model_cache=True):
    from mujoco_gateway_fixture import build_gateway_session
    def consume(fact):
        trace.append(fact)
        if after_fact is not None:
            after_fact(fact)

    return build_gateway_session(
        build_mujoco_agent_tcp_pose_gateway_session, use_model_cache=use_model_cache,
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
        enable_spatial_goals=spatial,
        enable_grippers=grasp,
        enable_grasp_approach=grasp,
        grasp_contact_target_body=contact_target,
    )


def _request(session, adapter, target, *, sequence=0, skill_id="motion.move_tcp_to_pose_safe", policy=None):
    observation = adapter.get_observation_snapshot()
    entry = session.contract_bundle.route_manifest.route(
        skill_id, "1"
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
        },
        observation_snapshot_id=observation["snapshot_id"],
        requested_mode="sim_active",
        requested_timeout_ms=min(120_000, route.max_operation_duration_ms),
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
@pytest.mark.parametrize("policy,motion_side,task", [
    # All policy/side, policy/task and side/task pairs; omit triple duplicates.
    pytest.param(None, "left", TASK, id="reach-left-default"),
    pytest.param(None, "right", Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"), id="dmtac-right-default"),
    pytest.param("within_arrival_tolerance", "right", TASK, id="reach-right-arrival"),
    pytest.param("within_arrival_tolerance", "left", Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"), id="dmtac-left-arrival"),
    pytest.param("preserve_target", "left", TASK, id="reach-left-preserve"),
    pytest.param("preserve_target", "right", Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"), id="dmtac-right-preserve"),
])
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
