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
