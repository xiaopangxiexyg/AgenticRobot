"""The opt-in approach Route shares the existing runtime and safety owner."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT, _request, _session

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
)
from agent_gateway.profiles.mujoco.motion_deployment import (
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from agent_gateway.profiles.mujoco.motion_gateway_session import (
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw


def _retained_world_grasp():
    return np.array([
        [-0.9992830955, 0.0167132372, -0.0339700262, -0.2313394386],
        [0.0208265641, 0.9919899135, -0.1245883856, 0.0113815844],
        [0.0316156481, -0.1252065465, -0.9916268308, 0.1746701841],
        [0, 0, 0, 1],
    ])


def _compact_world_grasp():
    """Explicit selected exchange from the same retained RGB-D/model pool."""
    return np.array([
        [.9985387640, -.0167128886, -.0513908171, -.3013165546],
        [-.0229960802, -.9919888972, -.1242143634, .0128636749],
        [-.0489031391, .1252146442, -.9909236983, .1774971271],
        [0, 0, 0, 1],
    ])


@pytest.mark.integration
def test_original_far_branch_is_rejected_before_dispatch_without_changing_target():
    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    facts = []
    session = _session(facts, grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        goal = _retained_world_grasp()
        goal[:3, 3] -= goal[:3, 2] * .07
        base = np.linalg.inv(app.scene.body_pose("right_arm_base_link")) @ goal
        from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
        with pytest.raises(MujocoTcpPlanRejected, match="motion_quality_exceeded") as failure:
            planner.plan(tuple(pose_to_xyz_quat_xyzw(base)), translation_tolerance_m=.003,
                rotation_tolerance_rad=np.deg2rad(2), cancel_requested=lambda: False)
        assert any(abs(row["joint_net_delta_deg"][0]) > 219 for row in failure.value.details["endpoint_attempts"])
        np.testing.assert_array_equal(planner._model_qpos(), source)
        assert not any(f.event_type == "command" for f in facts)
        assert not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_selector_previews_full_finger_exchange_chain_without_motion(record_property):
    from agent_gateway.profiles.mujoco.grasp_candidate_selection import (
        MujocoGraspSelectionConfig,
    )
    from dataset_contracts.perception_artifacts import (
        PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1,
    )
    from deployments.perception.sam2_vgn_point_grasp import PointGraspCandidate

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        before_time = float(app.scene.data.time)
        envelope_before = app.tcp_pose_envelope_runtime.snapshot()
        candidates = []
        for exchange in (False, True):
            goal = _retained_world_grasp()
            if exchange:
                goal[:3, :3] = goal[:3, :3] @ np.diag([-1., -1., 1.])
            candidates.append(PointGraspCandidate(score=.5, model_score=.7,
                click_distance_m=.01, gripper_width_m=.07,
                camera_from_tcp=tuple(goal.ravel()), base_from_tcp=tuple(goal.ravel()),
                visible_collision_free=True,
                pose_source=PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1 if exchange else "model_scored",
                orientation_variant_axis="z" if exchange else None,
                orientation_variant_angle_deg=180. if exchange else 0.))
        selector = session.build_grasp_candidate_selector(world_from_candidate_reference=np.eye(4),
            config=MujocoGraspSelectionConfig(terminal_offset_m=.01))
        result = selector(tuple(candidates), (0., 0., 0.), 2)
        record_property("selector_attempts", json.dumps(selector.last_attempts, default=str))
        assert result.selected_indices == (1,), selector.last_attempts
        assert result.diagnostics["arm_shortlist_pool_size"] == 2
        assert result.diagnostics["arm_shortlist_not_checked"] == 0
        assert selector.last_attempts[0]["reason"] == "safe_tcp_motion_quality_exceeded"
        accepted = selector.last_attempts[1]
        assert [s["phase"] for s in accepted["stages"]] == ["pregrasp", "near", "terminal"]
        assert accepted["stages"][0]["motion_quality"]["tcp_detour_ratio"] < 1.5
        assert accepted["motion_quality"]["joint_total_travel_deg"][0] < 45
        assert float(app.scene.data.time) == before_time
        np.testing.assert_array_equal(planner._model_qpos(), source)
        assert app.tcp_pose_envelope_runtime.snapshot() == envelope_before
        assert planner._pending_plan is None and not planner.active
        from dataclasses import replace
        corrupt = replace(candidates[1], base_from_tcp=tuple(np.full(16, np.nan)))
        with pytest.raises(ValueError, match="finite rigid"):
            selector((corrupt,), (0., 0., 0.), 1)
        assert planner._pending_plan is None and not planner.active
        with pytest.raises(ValueError, match="rigid"):
            session.build_grasp_candidate_selector(world_from_candidate_reference=np.full((4, 4), np.nan))
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_far_first_ik_result_does_not_hide_later_compact_branch(monkeypatch):
    from robot_motion.rm65 import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        goal = _retained_world_grasp()
        a = np.deg2rad(15)
        goal[:3, :3] = goal[:3, :3] @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        goal[:3, 3] -= goal[:3, 2] * .07
        goal = np.linalg.inv(app.scene.body_pose("right_arm_base_link")) @ goal
        far = np.deg2rad([133.347305, -22.323732, -54.353287, 3.886475, -96.964890, 60.113350])
        near = np.deg2rad([-46.652883, 22.323002, 54.353769, 3.886921, 96.964320, -119.886807])
        solver = build_rm65_single_arm_ik_solver(RM65SingleArmIKConfig(controlled_frame="arm_endpoint"))
        planner._solver = solver
        results = [far, near]

        def solve(**kwargs):
            del kwargs
            return results.pop(0).copy() if results else near.copy()

        monkeypatch.setattr(solver, "solve_iterative", solve)
        plan = planner.plan(tuple(pose_to_xyz_quat_xyzw(goal)), translation_tolerance_m=.003,
            rotation_tolerance_rad=np.deg2rad(2), cancel_requested=lambda: False)
        np.testing.assert_array_equal(plan.path.target_qpos_by_component["right"], near)
        assert not plan.path.metadata["endpoint_attempts"][0]["proximal_travel_within_limit"]
        assert plan.path.metadata["motion_quality"]["joint_total_travel_deg"][0] < 45
        assert plan.path.metadata["direct_path"]
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("local,expected", (
    ((0, 0, -.039), True), ((0, 0, -.041), False),
    ((.004, 0, -.02), False), ((0, .004, -.02), False),
    ((0, 0, .004), False), ((0, 0, 0), True),
))
def test_contact_corridor_uses_tool_axes(local, expected):
    from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding

    binding = MujocoGraspContactBinding("target", ("left_pad", "right_pad"))
    target = np.eye(4)
    target[:3, :3] = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))
    current = target.copy()
    current[:3, 3] = target[:3, :3] @ local
    assert bool(binding.in_corridor(current, target)) is expected


@pytest.mark.integration
def test_contact_scope_is_validated_and_measured_collision_is_checked_after_plan_clear():
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.deploy.grasp_contact import bind_task_grasp_contact
    from sim.mujoco.tasks.cube_lift.config import (
        resolve_dmtac_gripper_actuation_profile,
    )

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        binding = session.contract_bundle.grasp_contact_contract.contact_binding
        from dataclasses import replace
        with pytest.raises(ValueError, match="hard penetration"):
            replace(binding, max_penetration_m=.002).pair_clearances(app.scene)
        task_profile = resolve_dmtac_gripper_actuation_profile(app.task_config.gripper_actuation_profile_name)
        with pytest.raises(ValueError):
            bind_task_grasp_contact(app.scene, task_profile, "right_arm_link6")
        pose = app.capture_tcp_pose_by_component()["right_arm"]
        initial = planner.plan(pose, translation_tolerance_m=.001, rotation_tolerance_rad=.02,
            cancel_requested=lambda: False, contact_binding=binding)
        assert initial.contact_binding is binding
        planner.activate(initial)
        assert app.request_hold(reason="test:contact-cancel").snapshot.status == "hold"
        assert not planner.active
        assert planner.check_measured_contact(binding, pose).safe
        ordinary = planner.plan(pose, translation_tolerance_m=.001, rotation_tolerance_rad=.02,
            cancel_requested=lambda: False)
        assert ordinary.contact_binding is None
        model_qpos = planner._model_qpos()
        try:
            cube = app.scene.model.body("debug_cube").id
            address = app.scene.model.jnt_qposadr[app.scene.model.body_jntadr[cube]]
            pad = app.scene.model.geom(binding.sensor_geom_names[0]).id
            app.scene.data.qpos[address:address+3] = app.scene.data.geom_xpos[pad]
            app.scene.mujoco.mj_forward(app.scene.model, app.scene.data)
            assert not planner.check_measured_contact(binding, pose).safe
        finally:
            app.scene.data.qpos[:] = model_qpos
            app.scene.mujoco.mj_forward(app.scene.model, app.scene.data)
        goal = xyzquat_to_pose_matrix(pose)
        goal[:3, 3] += goal[:3, 0] * .02
        from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
        with pytest.raises(MujocoTcpPlanRejected, match="start_blocked"):
            planner.plan(tuple(pose_to_xyz_quat_xyzw(goal)), translation_tolerance_m=.001,
                rotation_tolerance_rad=.02, cancel_requested=lambda: False, contact_binding=binding)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_target_contact_route_finishes_retained_approach_without_relaxing_safe_move(record_property, monkeypatch):
    from dataclasses import replace

    from sim.mujoco.collision_safety import CollisionSafetyOracle

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    facts = []
    session = _session(facts, grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        bundle = session.contract_bundle
        contact = bundle.grasp_contact_contract
        assert contact.contact_binding.sensor_geom_names == (
            "right_gripper_l_dm_tac_w2m_sensor_collision_0",
            "right_gripper_r_dm_tac_w2m_sensor_collision_0",
        )
        assert session.grasp_contact_executor in session._motion_executors()
        with pytest.raises(ValueError, match="differs"):
            replace(contact, motion_profile=replace(contact.motion_profile, contact_policy_hash="wrong"))
        profile = contact.motion_profile
        contact_ticks = []
        check_contacts = CollisionSafetyOracle.check_contacts

        def capture_contact_tick(oracle, data):
            if oracle._pair_clearance_m:
                contact_ticks.append(float(data.time))
            return check_contacts(oracle, data)

        monkeypatch.setattr(CollisionSafetyOracle, "check_contacts", capture_contact_tick)
        measured_checks = []
        planner = app.target.planned_tcp_motion
        original_check = planner.check_measured_contact

        def capture_check(*args):
            measured_checks.append(planner.active)
            return original_check(*args)

        monkeypatch.setattr(planner, "check_measured_contact", capture_check)
        for seq, (offset, skill, success) in enumerate((
            (-.07, "motion.approach_grasp_pose", True),
            (-.025, "motion.approach_grasp_pose", True),
            # The exchanged CAD pose has different near-target clearance.
            # This negative explicitly targets below the table, rather than
            # assuming that an old orientation's contact depth is unchanged.
            (.12, "motion.approach_grasp_pose", False),
            (.01, "motion.approach_grasp_contact", True),
        )):
            goal = _compact_world_grasp()
            goal[:3, 3] += goal[:3, 2] * offset
            target = app.transform_tcp_pose_xyzquat(
                tuple(pose_to_xyz_quat_xyzw(np.linalg.inv(app.scene.body_pose("right_arm_base_link")) @ goal)),
                source_frame_id=profile.execution_reference_frame_id,
                target_frame_id=profile.reference_frame_id)
            request = _request(session, adapter, target, sequence=seq, skill_id=skill)
            admission = adapter.submit_skill_request(request.to_wire())
            assert admission["accepted"], admission
            assert session.wait(timeout_sec=60)
            op = session.service.get_operation(admission["operation"]["operation_id"])
            record_property(f"phase_{seq}", json.dumps({"skill":skill, "outcome":op.execution_outcome,
                "reason":op.reason_code, "progress":dict(op.progress_summary)}, default=str))
            assert (op.execution_outcome == "completed") is success, (skill, op.reason_code, op.progress_summary)
            assert app.held
        assert all(p.observe_state().measured_close < .01 for p in session.gripper_ports.values())
        assert not app.target.planned_tcp_motion.active
        assert True in measured_checks and False in measured_checks
        assert len(contact_ticks) > 100
        assert not planner._contact_guard_bound
        # Close remains an independent Skill; no object weld or lift is used.
        entry = bundle.route_manifest.route("manipulation.set_gripper_state", "2")
        route, param = entry.route, entry.route.allowed_parameter_profiles[0]
        base_request = _request(session, adapter, target, sequence=4, skill_id="motion.approach_grasp_pose")
        close_request = replace(base_request, skill_id=route.skill_id, skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash, route_id=route.route_id,
            route_version=route.route_version, route_hash=route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=param.parameter_profile_id, parameter_profile_hash=param.parameter_profile_hash,
            requested_timeout_ms=route.max_operation_duration_ms,
            parameters={"component_id": "right_gripper", "target_state": "closed"})
        admitted = adapter.submit_skill_request(close_request.to_wire())
        assert admitted["accepted"] and session.wait(timeout_sec=20)
        assert session.service.get_operation(admitted["operation"]["operation_id"]).execution_outcome == "completed"
        sensors = set()
        for index in range(app.scene.data.ncon):
            c = app.scene.data.contact[index]
            names = {app.scene.model.geom(int(g)).name for g in (c.geom1, c.geom2)}
            if "debug_cube_collision" in names:
                sensors.update(names.intersection(contact.contact_binding.sensor_geom_names))
        assert sensors == set(contact.contact_binding.sensor_geom_names)
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("grippers,realtime", ((False, False), (True, True)))
def test_grasp_approach_rejects_incomplete_ownership_before_resources(
    grippers, realtime
):
    with pytest.raises(ValueError, match="gripper|physics ownership"):
        build_mujoco_agent_tcp_pose_gateway_session(
            target_config=MujocoSimHILTargetConfig(
                task_config_path=Path("missing.json"), realtime_worker=realtime
            ),
            allowed_client_ids=(CLIENT,),
            enable_grippers=grippers,
            enable_grasp_approach=True,
        )


def test_contact_route_requires_approach_before_preparing_resources():
    with pytest.raises(ValueError, match="explicit grasp"):
        build_mujoco_agent_tcp_pose_gateway_session(
            target_config=MujocoSimHILTargetConfig(task_config_path=Path("missing.json"), realtime_worker=False),
            allowed_client_ids=(CLIENT,), grasp_contact_target_body="debug_cube")


@pytest.mark.integration
def test_public_contact_assembler_requires_approach_composition(monkeypatch):
    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        bundle = replace(session.contract_bundle, grasp_approach_contract=None)
        rig = app.runtime.rig
        time_before = float(rig.scene.data.time)
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                pytest.fail("Incomplete contact composition must fail before physics")

            patch.setattr(rig, "queue_gripper_close_targets", forbidden)
            patch.setattr(rig, "step_physics", forbidden)
            with pytest.raises(ValueError, match="separate grasp approach Route"):
                assemble_mujoco_agent_tcp_pose_gateway_session(
                    application=app, contract_bundle=bundle,
                    gripper_ports=session.gripper_ports, allowed_client_ids=(CLIENT,),
                )
        assert float(rig.scene.data.time) == time_before
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("invalid_owner", ("missing_grippers", "realtime_physics"))
def test_public_approach_assembler_rejects_incomplete_owner_before_resources(
    monkeypatch, invalid_owner
):
    session = _session([], grasp=True)
    try:
        app = session.application
        app.start_observation_ready()
        rig = app.runtime.rig
        bundle = session.contract_bundle
        ports = session.gripper_ports
        if invalid_owner == "missing_grippers":
            bundle = build_mujoco_agent_tcp_pose_contract_bundle(
                application=app, enable_grasp_approach=True
            )
            assert bundle.grasp_approach_contract is not None
            assert bundle.gripper_contract is None
            ports = None
        time_before = float(rig.scene.data.time)
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                pytest.fail("Invalid approach owner must fail before execution or physics")

            patch.setattr(rig, "queue_gripper_close_targets", forbidden)
            patch.setattr(rig, "step_physics", forbidden)
            if invalid_owner == "realtime_physics":
                # Exercise actual rig-config admission without starting a worker.
                patch.setattr(rig, "config", replace(rig.config, realtime_worker=True))
            with pytest.raises(ValueError, match="composed grippers|physics ownership"):
                assemble_mujoco_agent_tcp_pose_gateway_session(
                    application=app, contract_bundle=bundle,
                    gripper_ports=ports, allowed_client_ids=(CLIENT,),
                )
        assert float(rig.scene.data.time) == time_before
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_approach_has_own_absolute_envelope_without_relaxing_safe_move():
    session = _session([], grasp=True)
    try:
        session.application.start_observation_ready()
        bundle = session.contract_bundle
        ordinary = bundle.safe_tcp_pose_contract.motion_profile
        approach = bundle.grasp_approach_contract.motion_profile
        assert ordinary.max_translation_delta_m == 0.1
        assert ordinary.max_rotation_delta_rad == pytest.approx(np.deg2rad(15))
        assert approach.max_rotation_delta_rad == pytest.approx(np.pi)
        assert approach.workspace_bounds == ordinary.workspace_bounds
        assert approach.translation_tolerance_m < ordinary.translation_tolerance_m
        route = bundle.route_manifest.route("motion.approach_grasp_pose", "1")
        (envelope,) = route.deployment_binding.safety_envelope_refs
        assert envelope.allowed_route_ids == (route.route.route_id,)
        assert session.grasp_approach_executor is not session.safe_tcp_pose_executor
        assert session.grasp_approach_executor in session._motion_executors()
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("large_pregrasp", (False, True))
def test_native_cad_approach_reuses_exact_joint_execution_and_confirmed_hold(
    large_pregrasp, record_property, monkeypatch
):
    facts = []
    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session(facts, grasp=True, task=task)
    try:
        app = session.application
        app.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        profile = session.contract_bundle.grasp_approach_contract.motion_profile
        target = list(
            adapter.get_observation_snapshot()["measurement_summary"][
                "pose_xyz_quat_xyzw_by_component"
            ]["right_arm"]
        )
        if large_pregrasp:
            # Fixed target from the retained RGB-D candidate endpoint diagnostic.
            # This is a motion fixture, not proof of online model inference.
            world_from_tcp = np.array(
                [
                    [-0.9992830955, 0.0167132372, -0.0339700262, -0.2313394386],
                    [0.0208265641, 0.9919899135, -0.1245883856, 0.0113815844],
                    [0.0316156481, -0.1252065465, -0.9916268308, 0.1746701841],
                    [0, 0, 0, 1],
                ]
            )
            # Explicit finger-exchange candidate, not an execution-time flip.
            world_from_tcp[:3, :3] = world_from_tcp[:3, :3] @ np.diag([-1., -1., 1.])
            world_from_tcp[:3, 3] -= world_from_tcp[:3, 2] * 0.07
            base_from_tcp = (
                np.linalg.inv(app.scene.body_pose("right_arm_base_link"))
                @ world_from_tcp
            )
            target = app.transform_tcp_pose_xyzquat(
                tuple(pose_to_xyz_quat_xyzw(base_from_tcp)),
                source_frame_id=profile.execution_reference_frame_id,
                target_frame_id=profile.reference_frame_id,
            )
        else:
            target[2] += 0.03
        request = _request(
            session, adapter, target, skill_id="motion.approach_grasp_pose"
        )
        plans = []
        planner = app.target.planned_tcp_motion
        plan_original = planner.plan

        def capture_plan(*args, **kwargs):
            started = time.perf_counter()
            plan = plan_original(*args, **kwargs)
            record_property("planning_wall_sec", time.perf_counter() - started)
            record_property(
                "path_search_budget_sec", kwargs["path_search_time_budget_sec"]
            )
            record_property(
                "trajectory_duration_sec", float(plan.trajectory.timestamps[-1])
            )
            plans.append(plan)
            return plan

        monkeypatch.setattr(planner, "plan", capture_plan)
        monkeypatch.setattr(
            app.runtime.pose_resolver.ik,
            "solve",
            lambda *_a, **_k: pytest.fail(
                "Approach cannot replace admitted joint references with new IK"
            ),
        )
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=60)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        record_property(
            "operation_json",
            json.dumps(
                {
                    "status": operation.status,
                    "reason": operation.reason_code,
                    "execution_outcome": operation.execution_outcome,
                    "task_outcome": operation.task_outcome,
                    "lower_hold_confirmed": operation.progress_summary.get(
                        "lower_hold_confirmed"
                    ),
                }
            ),
        )
        assert operation.execution_outcome == "completed", (
            operation.reason_code,
            operation.progress_summary,
        )
        assert app.held and not planner.active
        final = adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"
        ]["right_arm"]
        assert (
            np.linalg.norm(np.array(final[:3]) - target[:3])
            <= profile.translation_tolerance_m
        )
        assert len(plans) == 1
        references = [
            f.payload["payload"]["output_values_by_effector"]["right_arm"]
            for f in facts
            if f.event_type == "component_execution_target"
        ]
        np.testing.assert_allclose(
            references[: len(plans[0].trajectory.q_rad)],
            plans[0].trajectory.q_rad,
            atol=1e-7,
        )
        assert all(
            p.observe_state().measured_close < 0.01
            for p in session.gripper_ports.values()
        )
    finally:
        assert session.close().accepted
