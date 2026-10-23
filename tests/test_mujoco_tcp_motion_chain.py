"""A complete approach retains one exact native trajectory, never a prefix."""

from dataclasses import replace
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_agent_gateway_mujoco_grasp_approach import _retained_world_grasp
from test_agent_gateway_mujoco_safe_tcp_pose import (
    CLIENT,
    _place_cube_at_collision,
    _request,
    _session,
)

from agent_gateway.profiles.mujoco import motion_gateway_session
from agent_gateway.profiles.mujoco.motion_skill import (
    MujocoAgentPreparedTcpPoseSkillLifecycle,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.path_planning import JointConfigurationCheck
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoTcpPlanGoal,
    MujocoTcpPlanRejected,
)
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw


def _vertical_goals(app):
    start = app.capture_tcp_pose_by_component()[app.component_id]
    return tuple(
        MujocoTcpPlanGoal(
            tuple(v + delta if i == 2 else v for i, v in enumerate(start)),
            0.003,
            0.03,
        )
        for delta in (0.01, 0.02, 0.03)
    )


@pytest.mark.integration
@pytest.mark.parametrize("motion_side", ("right", "left"))
def test_native_chain_preview_ownership_continuity_and_one_use(motion_side):
    facts = []
    session = _session(facts, motion_side=motion_side)
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        stamp = float(app.scene.data.time)
        goals = _vertical_goals(app)
        plan = planner.plan_chain(
            goals, cancel_requested=lambda: False, time_budget_sec=15
        )
        assert len(plan.stages) == 3 and not planner.active
        assert planner._pending_plan is plan
        np.testing.assert_array_equal(planner._model_qpos(), source)
        assert float(app.scene.data.time) == stamp
        assert not any(f.event_type == "command" for f in facts)
        assert np.all(np.diff(plan.trajectory.timestamps) > 0)
        assert len(plan.reference_poses) == len(plan.trajectory.timestamps)
        assert not plan.trajectory.q_rad.flags.writeable
        for index, stage in plan.stages:
            count = len(stage.reference_poses)
            np.testing.assert_array_equal(
                plan.trajectory.q_rad[index : index + count], stage.trajectory.q_rad
            )
            assert plan.reference_poses[index : index + count] == stage.reference_poses
            assert np.max(np.abs(plan.trajectory.qd_rad_s[index])) < 1e-10
            assert np.max(np.abs(plan.trajectory.qdd_rad_s2[index])) < 1e-10
        preview = planner.plan_chain(
            goals,
            cancel_requested=lambda: False,
            time_budget_sec=15,
            preview_model_qpos=source,
        )
        assert planner._pending_plan is plan
        for foreign in (preview, replace(plan, trajectory_hash="modified")):
            with pytest.raises(MujocoTcpPlanRejected, match="not_owned"):
                planner.activate(foreign)
        planner.activate(plan)
        pose = planner.reference(0)
        answer = planner.resolve(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1, arm_pose_by_side={motion_side: pose}
            )
        )
        assert answer.metadata["trajectory_hash"] == plan.trajectory_hash
        inactive_side = "left" if motion_side == "right" else "right"
        np.testing.assert_array_equal(
            answer.qpos_by_side[inactive_side], plan.path.start_qpos_by_component[inactive_side]
        )
        assert app.request_hold(reason="chain-test").snapshot.status == "hold"
        with pytest.raises(MujocoTcpPlanRejected, match="not_owned"):
            planner.activate(plan)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_failed_stage_does_not_retain_an_executable_prefix():
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        goals = _vertical_goals(app)
        reject = lambda *_: JointConfigurationCheck(
            safe=False, minimum_distance_m=None, reason="test-obstacle"
        )
        bad = replace(
            goals[1],
            configuration_constraint=reject,
            constraint_identity="test:obstacle",
        )
        with pytest.raises(MujocoTcpPlanRejected, match="start_blocked") as captured:
            planner.plan_chain(
                (goals[0], bad, goals[2]),
                cancel_requested=lambda: False,
                time_budget_sec=15,
            )
        assert captured.value.details["chain_stage_index"] == 1
        assert captured.value.details["completed_stage_count"] == 1
        assert planner._pending_plan is None and not planner.active
        np.testing.assert_array_equal(planner._model_qpos(), source)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_noncontact_stage_installs_physics_tick_guard_for_realized_hand_object_contact():
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        plan = planner.plan_chain(
            _vertical_goals(app), cancel_requested=lambda: False, time_budget_sec=15
        )
        planner.activate(plan)
        assert planner._contact_guard_bound
        assert planner._live_oracle.projector.policy.pair_is_forbidden(4, 8)
        _place_cube_at_collision(app, plan, terminal=True)
        for name, value in zip(
            app.scene.arm_specs["right"].joint_names, plan.trajectory.q_rad[-1]
        ):
            app.scene.data.qpos[app.scene.joint_qpos_addr(name)] = value
        app.scene.mujoco.mj_forward(app.scene.model, app.scene.data)
        canonical = app.scene.collision_projector()
        assert canonical.oracle.check_contacts(app.scene.data).safe
        assert not planner._live_oracle.projector.oracle.check_contacts(
            app.scene.data
        ).safe
        with pytest.raises(RuntimeError, match="forbidden hard contact"):
            app.runtime.rig._execution._check_active_contact_or_latch(
                canonical, phase="test:between-references"
            )
        assert app.request_hold(reason="native-contact-test").snapshot.status == "hold"
        assert not planner._contact_guard_bound
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("scale", (True, 0, 1.01, float("nan")))
def test_goal_rejects_invalid_retiming_scale(scale):
    with pytest.raises(ValueError, match="speed_scale"):
        MujocoTcpPlanGoal(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.003, 0.03, speed_scale=scale
        )


@pytest.mark.integration
def test_tiny_positive_speed_is_rejected_before_trajectory_grid_allocation(monkeypatch):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        monkeypatch.setattr(
            "sim.mujoco.deploy.planned_tcp_motion.build_offline_quintic_waypoint_trajectory",
            lambda *a, **kw: pytest.fail(
                "Oversized grid must be rejected before builder allocation."
            ),
        )
        with pytest.raises(MujocoTcpPlanRejected, match="trajectory_budget_exhausted"):
            planner.plan(
                _vertical_goals(app)[0].target_pose,
                translation_tolerance_m=0.003,
                rotation_tolerance_rad=0.03,
                cancel_requested=lambda: False,
                speed_scale=1e-12,
            )
        assert not planner.active and planner._pending_plan is None
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_contact_retiming_is_certified_before_publication_without_midcurve_waits():
    from robot_motion.relative_action import xyzquat_to_pose_matrix

    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        initial = xyzquat_to_pose_matrix(
            app.capture_tcp_pose_by_component()["right_arm"]
        )
        target = initial.copy()
        target[:3, 3] += target[:3, 2] * 0.01
        binding = session.contract_bundle.grasp_contact_contract.contact_binding
        plan = planner.plan(
            tuple(pose_to_xyz_quat_xyzw(target)),
            translation_tolerance_m=0.001,
            rotation_tolerance_rad=np.deg2rad(1),
            contact_binding=binding,
            cancel_requested=lambda: False,
            speed_scale=0.25,
        )
        certificate = plan.trajectory.metadata["continuous_hard_limit_certificate"]
        config = app.runtime.rig.config
        velocity = np.minimum(
            config.max_velocity_rad_s,
            np.deg2rad(config.max_command_step_deg) * config.timing.command_hz,
        )
        assert np.all(
            np.asarray(certificate["max_abs_velocity_rad_s"]) <= velocity * 0.25 + 1e-7
        )
        assert not plan.trajectory.waypoint_timestamps[1:-1].size
        planner.activate(plan)
        midpoint = len(plan.reference_poses) // 2
        assert np.max(np.abs(plan.trajectory.qd_rad_s[midpoint])) > 1e-6
        for cycle in range(midpoint + 1):
            planner.reference(cycle)
        # Simulate large following lag without dispatching anything. A timing
        # certificate cannot be preserved by pausing this nonzero-velocity knot.
        assert planner._reference_index == midpoint
        assert (
            np.linalg.norm(
                xyzquat_to_pose_matrix(plan.reference_poses[midpoint])[:3, 3]
                - initial[:3, 3]
            )
            > binding.lateral_tolerance_m / 4
        )
        planner.reference(midpoint + 1)
        assert planner._reference_index == midpoint + 1
        np.testing.assert_array_equal(planner._model_qpos(), source)
        assert app.request_hold(reason="retime-test").snapshot.status == "hold"
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_total_budget_includes_planner_lock_wait(monkeypatch):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        goals = _vertical_goals(app)
        began = threading.Event()
        clock = [0.0]

        def now():
            began.set()
            return clock[0]

        monkeypatch.setattr(
            planner,
            "plan",
            lambda *a, **kw: pytest.fail("Expired query must not start native search."),
        )
        with ThreadPoolExecutor(max_workers=1) as workers:
            with planner._lock:
                pending = workers.submit(
                    planner.plan_chain,
                    goals,
                    cancel_requested=lambda: False,
                    time_budget_sec=5,
                    monotonic=now,
                )
                assert began.wait(timeout=2)
                clock[0] = 6.0
            with pytest.raises(MujocoTcpPlanRejected, match="budget_exhausted"):
                pending.result(timeout=2)
        assert not planner.active and planner._pending_plan is None
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("elapsed,cancel,reason", (
    (0.25, False, None),
    (1.0, False, "safe_tcp_planning_budget_exhausted"),
    (1.01, False, "safe_tcp_planning_budget_exhausted"),
    (1.0, True, "safe_tcp_planning_cancelled"),
))
def test_refinement_slice_includes_planner_lock_wait(monkeypatch, elapsed, cancel, reason):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        original = planner.plan_chain(_vertical_goals(app),
            cancel_requested=lambda: False, time_budget_sec=15)
        refined = replace(original, trajectory_hash="test:refined-after-lock-wait")
        clock, cancelled = [0.0], [False]
        waiting = threading.Event()
        owner = threading.get_ident()
        lock = planner._lock

        class ObservedLock:
            def __enter__(self):
                if threading.get_ident() != owner:
                    waiting.set()
                lock.acquire()
                return self

            def __exit__(self, *args):
                lock.release()

        monkeypatch.setattr(planner, "_lock", ObservedLock())
        calls, validated = [], []

        def preview(*args, **kwargs):
            calls.append(kwargs["time_budget_sec"])
            return refined

        monkeypatch.setattr(planner, "plan_chain", preview)
        with ThreadPoolExecutor(max_workers=1) as workers:
            with lock:
                pending = workers.submit(planner.refine_pending_chain,
                    original, (), cancel_requested=lambda: cancelled[0],
                    time_budget_sec=1.0, deadline_at=10.0,
                    monotonic=lambda: clock[0], validate_plan=validated.append)
                assert waiting.wait(timeout=2), "Worker must reach the planner lock after taking the rig."
                clock[0], cancelled[0] = elapsed, cancel
            if reason is not None:
                with pytest.raises(MujocoTcpPlanRejected) as error:
                    pending.result(timeout=3)
                assert error.value.reason_code == reason
                assert calls == validated == []
                assert planner._pending_plan is original
            else:
                assert pending.result(timeout=3) is refined
                assert calls == [pytest.approx(1.0 - elapsed)]
                assert validated == [refined] and planner._pending_plan is refined
        assert not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("cancel", (False, True))
def test_refinement_final_scene_read_cannot_replace_after_cancel_or_expiry(monkeypatch, cancel):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        original = planner.plan_chain(_vertical_goals(app),
            cancel_requested=lambda: False, time_budget_sec=15)
        refined = replace(original, trajectory_hash="test:refined-before-final-cut")
        clock, cancelled, reads = [0.0], [False], []
        model_qpos = planner._model_qpos

        def read():
            result = model_qpos()
            reads.append(True)
            if len(reads) == 2:
                if cancel:
                    cancelled[0] = True
                else:
                    clock[0] = 1.0
            return result

        monkeypatch.setattr(planner, "_model_qpos", read)
        monkeypatch.setattr(planner, "plan_chain", lambda *args, **kwargs: refined)
        with pytest.raises(MujocoTcpPlanRejected) as error:
            planner.refine_pending_chain(original, (),
                cancel_requested=lambda: cancelled[0], time_budget_sec=1.0,
                deadline_at=10.0, monotonic=lambda: clock[0], validate_plan=lambda plan: None)
        assert error.value.reason_code == (
            "safe_tcp_planning_cancelled" if cancel else "safe_tcp_planning_budget_exhausted"
        )
        assert len(reads) == 2 and planner._pending_plan is original and not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_query_and_execution_use_the_same_rig_then_planner_lock_order(
    monkeypatch,
):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        rig = app.runtime.rig
        original = rig.capture_scene
        local = threading.local()
        captures = []

        def capture(project):
            depth = getattr(local, "depth", 0)
            if depth == 0:
                assert (
                    not planner._lock._is_owned()
                ), "planner -> rig is an inverted lock order"
            local.depth = depth + 1
            try:
                captures.append(depth)
                return original(project)
            finally:
                local.depth = depth

        monkeypatch.setattr(rig, "capture_scene", capture)
        plan = planner.plan_chain(
            _vertical_goals(app), cancel_requested=lambda: False, time_budget_sec=15
        )
        planner.activate(plan)
        pose = planner.reference(0)
        planner.resolve(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1, arm_pose_by_side={"right": pose}
            )
        )
        assert app.request_hold(reason="lock-order-test").snapshot.status == "hold"
        assert any(depth > 0 for depth in captures)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_native_pending_consumption_checks_deadline_after_lock_wait(monkeypatch):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        plan = planner.plan_chain(_vertical_goals(app), cancel_requested=lambda: False,
                                  time_budget_sec=15)
        began = threading.Event()
        capture = app.runtime.rig.capture_scene

        def waiting(project):
            began.set()
            return capture(project)

        monkeypatch.setattr(app.runtime.rig, "capture_scene", waiting)
        clock = [0.]
        with ThreadPoolExecutor(max_workers=1) as workers:
            with planner._lock:
                pending = workers.submit(planner.activate, plan, deadline_at=1., clock=lambda: clock[0])
                assert began.wait(timeout=2)
                clock[0] = 1.
            with pytest.raises(MujocoTcpPlanRejected, match="grasp_path_plan_expired"):
                pending.result(timeout=3)
        assert planner.owns_pending_plan(plan)
        assert not planner.active and not planner._contact_guard_bound
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("cancel", (False, True))
def test_final_scene_read_cannot_publish_after_cancel_or_expiry(monkeypatch, cancel):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        goal = _vertical_goals(app)[0]
        model_qpos = planner._model_qpos
        count = 0
        clock = [0.0]
        cancelled = [False]

        def read():
            nonlocal count
            count += 1
            cut = model_qpos()
            if count == 2:
                if cancel:
                    cancelled[0] = True
                else:
                    clock[0] = 6.0
            return cut

        monkeypatch.setattr(planner, "_model_qpos", read)
        with pytest.raises(
            MujocoTcpPlanRejected,
            match="planning_cancelled" if cancel else "budget_exhausted",
        ):
            planner.plan_chain(
                (goal,),
                cancel_requested=lambda: cancelled[0],
                time_budget_sec=5,
                monotonic=lambda: clock[0],
            )
        assert count == 2 and planner._pending_plan is None
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("failure", ("cancel", "budget", "scene"))
def test_chain_cancellation_deadline_and_scene_drift_never_publish(
    monkeypatch, failure
):
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        goals = _vertical_goals(app)
        original = planner.plan
        cancelled = False
        clock = [0.0]

        def query(*a, **kw):
            nonlocal cancelled
            plan = original(*a, **kw)
            if failure == "cancel":
                cancelled = True
            elif failure == "budget":
                clock[0] = 16.0
            elif failure == "scene":
                app.scene.data.qpos[0] += 0.01
            return plan

        monkeypatch.setattr(planner, "plan", query)
        reason = {
            "cancel": "planning_cancelled",
            "budget": "budget_exhausted",
            "scene": "scene_changed",
        }[failure]
        with pytest.raises(MujocoTcpPlanRejected, match=reason):
            planner.plan_chain(
                goals,
                cancel_requested=lambda: cancelled,
                time_budget_sec=15,
                **({"monotonic": lambda: clock[0]} if failure == "budget" else {}),
            )
        assert planner._pending_plan is None and not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
@pytest.mark.parametrize("return_to_source", (False, True))
def test_gateway_lifecycle_consumes_complete_retained_chain_without_replanning(
    monkeypatch, record_property, return_to_source
):
    facts = []
    session = _session(facts)
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        goals = _vertical_goals(app)
        if return_to_source:
            initial = tuple(app.capture_tcp_pose_by_component()["right_arm"])
            goals = (*goals, MujocoTcpPlanGoal(initial, 0.003, 0.03))
        plan = planner.plan_chain(
            goals, cancel_requested=lambda: False, time_budget_sec=15
        )
        profile = session.contract_bundle.motion_profile
        public_target = app.transform_tcp_pose_xyzquat(
            plan.reference_poses[-1],
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id,
        )
        monkeypatch.setattr(
            motion_gateway_session,
            "MujocoAgentSafeTcpPoseSkillLifecycle",
            lambda **kw: MujocoAgentPreparedTcpPoseSkillLifecycle(
                prepared_plan=plan,
                **{k: v for k, v in kw.items() if k != "contact_binding"},
            ),
        )
        monkeypatch.setattr(
            planner,
            "plan",
            lambda *a, **kw: pytest.fail("A retained chain must never replan."),
        )
        monkeypatch.setattr(
            app.runtime.pose_resolver.ik,
            "solve",
            lambda *a, **kw: pytest.fail("No execution-time IK."),
        )
        indices = []
        original_reference = planner.reference

        def select(cycle):
            pose = original_reference(cycle)
            indices.append(planner._reference_index)
            return pose

        monkeypatch.setattr(planner, "reference", select)
        admitted = adapter.submit_skill_request(
            _request(session, adapter, public_target).to_wire()
        )
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation.execution_outcome == "completed", operation
        assert operation.task_outcome == "succeeded", operation
        measured = app.capture_tcp_pose_by_component()["right_arm"]
        assert (
            np.linalg.norm(np.asarray(measured[:3]) - plan.reference_poses[-1][:3])
            < 0.003
        )
        assert app.held and not planner.active and planner._pending_plan is None
        # The lower runtime may repeat held knots; it may not skip or reorder a
        # retained trajectory sample. The cursor is still the native owner.
        expected = np.asarray(plan.reference_poses, dtype=np.float32)
        assert indices == sorted(indices)
        assert set(indices) == set(range(len(expected)))
        record_property("native_chain_hash", plan.trajectory_hash)
        record_property("native_chain_samples", len(expected))
        record_property(
            "native_chain_duration_sec", float(plan.trajectory.timestamps[-1])
        )
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_full_grasp_chain_keeps_contact_rights_only_in_terminal_stage(
    monkeypatch, record_property
):
    task = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
    session = _session([], grasp=True, task=task, contact_target="debug_cube")
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        goal = _retained_world_grasp()
        goal[:3, :3] = goal[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        base = np.linalg.inv(app.scene.body_pose("right_arm_base_link"))
        selector = session.build_grasp_candidate_selector(
            world_from_candidate_reference=np.eye(4)
        )
        goals = []
        for offset, contract in (
            (-0.07, session.contract_bundle.grasp_approach_contract),
            (-0.025, session.contract_bundle.grasp_approach_contract),
            (0.01, session.contract_bundle.grasp_contact_contract),
        ):
            target = goal.copy()
            target[:3, 3] += offset * goal[:3, 2]
            constraint, identity = selector._constraint(contract)
            goals.append(
                MujocoTcpPlanGoal(
                    tuple(pose_to_xyz_quat_xyzw(base @ target)),
                    contract.motion_profile.translation_tolerance_m,
                    contract.motion_profile.rotation_tolerance_rad,
                    constraint,
                    identity,
                    contract.contact_binding,
                    speed_scale=0.25 if contract.contact_binding is not None else 1.0,
                )
            )
        chain = planner.plan_chain(
            tuple(goals), cancel_requested=lambda: False, time_budget_sec=20
        )
        assert [s.contact_binding is not None for _, s in chain.stages] == [
            False,
            False,
            True,
        ]
        np.testing.assert_array_equal(planner._model_qpos(), source)
        planner.activate(chain)
        assert (
            planner._contact_guard_bound and planner._live_stage.contact_binding is None
        )
        # Exercise reverse guard binding without advancing a reference/physics.
        # Noncontact must restore its strict oracle, not merely remove contact
        # permissions and fall back to the more permissive canonical task guard.
        contact_stage = chain.stages[-1][1]
        planner._bind_live_stage(
            contact_stage, planner._oracle_for_stage(contact_stage)
        )
        first_stage = chain.stages[0][1]
        strict = planner._oracle_for_stage(first_stage)
        planner._bind_live_stage(first_stage, strict)
        assert (
            app.runtime.rig._execution._additional_contact_oracle
            is strict.projector.oracle
        )
        assert strict.projector.policy.pair_is_forbidden(4, 8)
        assert (
            app.request_hold(reason="grasp-chain-ownership-test").snapshot.status
            == "hold"
        )
        # HOLD consumes/invalidates that native handle, so obtain a fresh exact
        # chain. The public paired Route is intentionally not claimed here.
        chain = planner.plan_chain(
            tuple(goals), cancel_requested=lambda: False, time_budget_sec=20
        )
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        profile = session.contract_bundle.motion_profile
        public_target = app.transform_tcp_pose_xyzquat(
            chain.reference_poses[-1],
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id,
        )
        monkeypatch.setattr(
            motion_gateway_session,
            "MujocoAgentSafeTcpPoseSkillLifecycle",
            lambda **kw: MujocoAgentPreparedTcpPoseSkillLifecycle(
                prepared_plan=chain,
                **{k: v for k, v in kw.items() if k != "contact_binding"},
            ),
        )
        monkeypatch.setattr(
            planner,
            "plan",
            lambda *a, **kw: pytest.fail("Retained grasp must not replan."),
        )
        policies = []
        # The native resolver callable was bound by the target at construction;
        # observe the same owner's stage publication instead of replacing it.
        original_bind = planner._bind_live_stage

        def bind(stage, oracle):
            policies.append(stage.contact_binding is not None)
            return original_bind(stage, oracle)

        monkeypatch.setattr(planner, "_bind_live_stage", bind)
        admitted = adapter.submit_skill_request(
            _request(
                session, adapter, public_target, skill_id="motion.approach_grasp_pose"
            ).to_wire()
        )
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=35)
        operation = session.service.operation_store.get(
            admitted["operation"]["operation_id"]
        )
        assert operation.execution_outcome == "completed", operation
        assert policies == [False, False, True]
        assert app.held and not planner._contact_guard_bound
        record_property("grasp_chain_hash", chain.trajectory_hash)
        record_property(
            "grasp_chain_duration_sec", float(chain.trajectory.timestamps[-1])
        )
    finally:
        assert session.close().accepted
