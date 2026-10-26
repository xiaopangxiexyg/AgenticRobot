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


@pytest.mark.integration
def test_preview_does_not_replace_pending_plan_or_allow_activation():
    session = _session([])
    try:
        app = session.application
        app.start_observation_ready()
        planner = app.target.planned_tcp_motion
        source = planner._model_qpos()
        target = app.capture_tcp_pose_by_component()["right_arm"]
        kwargs = {"translation_tolerance_m": .003, "rotation_tolerance_rad": .03,
            "cancel_requested": lambda: False}
        owned = planner.plan(target, **kwargs)
        before_time = float(app.scene.data.time)
        preview = planner.plan(target, preview_model_qpos=source, **kwargs)
        with pytest.raises(MujocoTcpPlanRejected, match="not_owned"):
            planner.activate(preview)
        assert planner._pending_plan is owned
        assert not planner.active
        assert float(app.scene.data.time) == before_time
        np.testing.assert_array_equal(planner._model_qpos(), source)
        planner.activate(owned)
        assert app.request_hold(reason="preview-test").snapshot.status == "hold"
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("scenario,reason,expected_queries", (
    ("all_miss", "safe_tcp_endpoint_search_exhausted", 1),
    ("rescued_collision", "safe_tcp_target_blocked", 1),
    ("rescued_envelope", "safe_tcp_target_envelope_blocked", 1),
    ("original_collision", "safe_tcp_target_blocked", 0),
    ("cancel", "safe_tcp_planning_cancelled", 1),
    ("nonfinite", "safe_tcp_endpoint_numerical_failure", 1),
    ("expired", "safe_tcp_endpoint_search_exhausted", 0),
))
def test_endpoint_rescue_preserves_admission_deadline_and_diagnostics(monkeypatch, scenario, reason, expected_queries):
    from types import SimpleNamespace
    from threading import RLock

    planner = object.__new__(planning_module.MujocoPlannedTcpMotion)
    planner._source_model_qpos = None
    planner._solver_orientation_weight = None
    planner._motion_side = "right"
    planner._lock = RLock()
    planner._active = planner._pending_plan = None
    planner.scene = SimpleNamespace(model=SimpleNamespace(nq=12))
    scene_cuts = []
    def capture_scene(operation):
        scene_cuts.append("capture")
        return operation(planner.scene)
    planner.rig = SimpleNamespace(capture_scene=capture_scene)
    planner._model_qpos = lambda: np.zeros(12)
    planner.task_config = SimpleNamespace(right_tool_profile_path="unused")
    planner.planning_clearance_padding_m = .002
    planner.pose_for_configuration = lambda model,pair:(float(pair["right"][0]),0.,0.,0.,0.,0.,1.)
    query_calls = []
    def solve_iterative(**kwargs):
        assert kwargs["max_iterations"]==100 and kwargs["deadline_monotonic"]==2.
        return np.array([1. if scenario=="original_collision" else .2,0.,0.,0.,0.,0.])
    def solve_bounded(goal, seed, **kwargs):
        query_calls.append(kwargs)
        assert kwargs["deadline_monotonic"]==2. and kwargs["maximum_evaluations"]==100
        assert kwargs["translation_tolerance_m"]==.003 and kwargs["rotation_tolerance_rad"]==.03
        if scenario=="cancel":
            raise InterruptedError("fixture cancel")
        if scenario=="nonfinite":
            from robot_motion.rm65.single_arm_ik import BoundedIKNumericalError
            raise BoundedIKNumericalError("fixture non-finite FK")
        return np.array([1. if scenario in ("rescued_collision", "rescued_envelope") else .3,0.,0.,0.,0.,0.])
    planner._solver = SimpleNamespace(q_min=np.full(6,-np.pi),q_max=np.full(6,np.pi),
        compute_pose=lambda q:np.eye(4),solve_iterative=solve_iterative,
        solve_bounded_pose=solve_bounded,last_solve_iterations=100,
        last_solve_converged=False,last_solve_deadline_exceeded=False)
    def check(pair):
        return JointConfigurationCheck(safe=not np.any(pair["right"]),
            reason="tcp_envelope_blocked" if scenario=="rescued_envelope" else "fixture_collision",
            minimum_distance_m=.0017,
            metadata={"geom1": "table", "geom2": "sensor", "collision_category": "gripper_environment"})
    monkeypatch.setattr(planning_module,"MujocoJointSceneOracle",lambda *a,**kw:SimpleNamespace(
        addresses={"left":np.arange(6),"right":np.arange(6,12)},check=check))
    monkeypatch.setattr(planning_module,"load_rm65_tool_profile",lambda *a,**kw:None)
    monkeypatch.setattr(planning_module,"endpoint_to_tool_transform",lambda _:np.eye(4))
    times=iter([0.]+[3.]*40 if scenario=="expired" else [0.]+[.1]*40)
    monkeypatch.setattr(planning_module,"time",SimpleNamespace(perf_counter=lambda:next(times)))
    with pytest.raises(MujocoTcpPlanRejected) as failure:
        planner.plan((1.,0.,0.,0.,0.,0.,1.),translation_tolerance_m=.003,
            rotation_tolerance_rad=.03,cancel_requested=lambda:False)
    assert failure.value.reason_code==reason
    assert len(query_calls)==expected_queries
    assert scene_cuts == ["capture"]
    assert planner._pending_plan is None and planner._active is None
    if scenario not in ("cancel", "nonfinite"):
        diagnostic=failure.value.details["endpoint_search_diagnostics"]
        assert diagnostic["budget_sec"]==2. and diagnostic["seed_count"]==8
        assert diagnostic["attempted_seed_count"]==(0 if scenario=="expired" else 8+expected_queries)
        if scenario=="all_miss":
            assert failure.value.details["endpoint_attempts"]==[]
            assert len(diagnostic["attempts"])==9
        checks = failure.value.details["endpoint_checks"]
        assert len(checks) == len(failure.value.details["endpoint_attempts"])
        if scenario in ("rescued_collision", "rescued_envelope", "original_collision"):
            assert checks and all(not check["safe"] for check in checks)
            assert all(check["metadata"]["geom2"] == "sensor" for check in checks)
            assert all(check["minimum_distance_m"] == .0017 for check in checks)
            # Keep successful path metadata/hash inputs unchanged. Native
            # collision details are a separate rejection-only field.
            assert all("collision_check" not in row for row in failure.value.details["endpoint_attempts"])
