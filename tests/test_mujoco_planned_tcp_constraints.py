"""Planning constraints remain bound through native exact-reference admission."""

import numpy as np
import pytest
from test_agent_gateway_mujoco_safe_tcp_pose import _session

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.path_planning import JointConfigurationCheck
from sim.mujoco.deploy import planned_tcp_motion as planning_module
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoJointSceneOracle,
    MujocoTcpPlanRejected,
)


@pytest.mark.integration
def test_configuration_constraint_rejects_search_without_mutating_plant():
    facts = []
    session = _session(facts)
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        before = planner._model_qpos()
        oracle = MujocoJointSceneOracle(app.scene, before)
        pair = {side: before[address] for side, address in oracle.addresses.items()}
        assert oracle.check(pair).safe

        def reject(_model_qpos, _pair):
            return JointConfigurationCheck(
                safe=False,
                minimum_distance_m=None,
                reason="test_tcp_envelope_blocked",
                metadata={"observed_by": "bound_configuration_constraint"},
            )

        constrained = MujocoJointSceneOracle(
            app.scene,
            before,
            configuration_constraint=reject,
            constraint_identity="test:fixed-envelope",
        )
        assert constrained.scene_hash != oracle.scene_hash
        assert constrained.check(pair).reason == "test_tcp_envelope_blocked"
        with pytest.raises(ValueError, match="bound together"):
            MujocoJointSceneOracle(app.scene, before, configuration_constraint=reject)
        kwargs = {
            "translation_tolerance_m": 0.003,
            "rotation_tolerance_rad": 0.03,
            "cancel_requested": lambda: False,
        }
        target = app.capture_tcp_pose_by_component()["right_arm"]
        with pytest.raises(
            MujocoTcpPlanRejected, match="safe_tcp_start_blocked"
        ) as failure:
            planner.plan(
                target,
                **kwargs,
                configuration_constraint=reject,
                constraint_identity="test:fixed-envelope",
            )
        assert failure.value.details["reason"] == "test_tcp_envelope_blocked"
        for budget in (0, float("nan"), True):
            with pytest.raises(ValueError, match="time_budget"):
                planner.plan(target, **kwargs, path_search_time_budget_sec=budget)
        assert not any(f.event_type == "command" for f in facts)
        np.testing.assert_array_equal(planner._model_qpos(), before)
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_registered_plan_keeps_configuration_constraint_for_live_admission():
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        allowed = True
        checks = []

        def constraint(model_qpos, pair):
            checks.append(planner.pose_for_configuration(model_qpos, pair))
            return JointConfigurationCheck(
                safe=allowed,
                minimum_distance_m=None,
                reason=None if allowed else "test_constraint_binding_changed",
            )

        target = app.capture_tcp_pose_by_component()["right_arm"]
        plan = planner.plan(
            target,
            translation_tolerance_m=0.003,
            rotation_tolerance_rad=0.03,
            cancel_requested=lambda: False,
            configuration_constraint=constraint,
            constraint_identity="test:original-binding",
        )
        assert checks
        assert plan.configuration_constraint is constraint
        assert plan.constraint_identity == "test:original-binding"
        planner.activate(plan)
        pose = planner.reference(0)
        allowed = False
        with pytest.raises(
            MujocoTcpPlanRejected, match="live_scene_blocked"
        ) as failure:
            planner.resolve(
                RobotAction(
                    action_space=BIMANUAL_EE_POSE_V1, arm_pose_by_side={"right": pose}
                )
            )
        assert (
            failure.value.details["check"]["reason"]
            == "test_constraint_binding_changed"
        )
        app.request_hold(reason="test:invalidate-constraint-plan")
        assert not planner.active
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_search_cancellation_does_not_wait_out_long_budget(monkeypatch):
    facts = []
    session = _session(facts)
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        cancelled = False
        original_search = planning_module.plan_collision_aware_joint_path

        def cancel_at_search(**kwargs):
            nonlocal cancelled
            cancelled = True
            return original_search(**kwargs)

        monkeypatch.setattr(
            planning_module, "plan_collision_aware_joint_path", cancel_at_search
        )
        target = list(app.capture_tcp_pose_by_component()["right_arm"])
        target[2] += 0.03
        with pytest.raises(MujocoTcpPlanRejected, match="planning_cancelled"):
            planner.plan(
                tuple(target),
                translation_tolerance_m=0.003,
                rotation_tolerance_rad=0.03,
                cancel_requested=lambda: cancelled,
                path_search_time_budget_sec=15.0,
            )
        assert cancelled and not planner.active
        assert not any(f.event_type == "command" for f in facts)
    finally:
        assert session.close().accepted
