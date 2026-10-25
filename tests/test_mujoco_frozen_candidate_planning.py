"""Frozen queries reuse the native planner without gaining execution authority."""
from __future__ import annotations
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from robot_motion.path_planning import JointConfigurationCheck
from sim.mujoco.deploy import planned_tcp_motion
from sim.mujoco.deploy.planned_tcp_motion import MujocoPlannedTcpMotion, MujocoTcpPlanRejected


@pytest.mark.integration
@pytest.mark.parametrize("side", ["left", "right"])
def test_frozen_native_query_retains_private_state_without_motion_authority(side):
    from test_agent_gateway_mujoco_safe_tcp_pose import _session

    facts = []
    session = _session(facts, motion_side=side)
    try:
        app = session.application
        app.start_observation_ready()
        live = app.target.planned_tcp_motion
        source = live._model_qpos()
        addresses = {arm: [app.scene.joint_qpos_addr(name)
            for name in app.scene.arm_specs[arm].joint_names] for arm in ("left", "right")}
        pair = {arm: source[index].copy() for arm, index in addresses.items()}
        pose = live.pose_for_configuration(source, pair)
        arguments = dict(translation_tolerance_m=.003, rotation_tolerance_rad=.03,
                         cancel_requested=lambda: False)
        pending = live.plan(pose, **arguments)
        before = (source.copy(), app.scene.data.qvel.copy(), float(app.scene.data.time),
                  app.tcp_pose_envelope_runtime.snapshot(), len(facts))
        rig = app.runtime.rig
        query = MujocoPlannedTcpMotion(
            rig=SimpleNamespace(scene=app.scene, config=rig.config, capture_scene=rig.capture_scene),
            task_config=live.task_config, pose_projector=live.pose_projector,
            reference_hz=live.reference_hz, motion_side=side, source_model_qpos=source)
        source[:] = 100.
        np.testing.assert_array_equal(query._model_qpos(), before[0])
        copied = query._model_qpos()
        copied[:] = -100.
        np.testing.assert_array_equal(query._model_qpos(), before[0])
        selection = query.plan_candidates([pose, pose], **arguments,
            normalize_ik_objective=True, prioritize_current_seed=True, time_budget_sec=5.)
        assert selection.candidate_index == 0 and selection.all_candidates_checked
        assert query.owns_pending_plan(selection.plan)
        with pytest.raises(MujocoTcpPlanRejected, match="query_only_plan"):
            query.activate(selection.plan)
        query.clear()
        np.testing.assert_array_equal(app.scene.data.qpos, before[0])
        np.testing.assert_array_equal(app.scene.data.qvel, before[1])
        assert float(app.scene.data.time) == before[2]
        assert app.tcp_pose_envelope_runtime.snapshot() == before[3]
        assert len(facts) == before[4]
        assert live.owns_pending_plan(pending) and not live.active and not query.active
    finally:
        assert session.close().accepted


@pytest.mark.parametrize("source", [np.zeros(11), np.full(12, np.nan), np.zeros((12, 1))])
def test_frozen_query_requires_a_finite_complete_model_cut(source):
    rig = SimpleNamespace(scene=SimpleNamespace(model=SimpleNamespace(nq=12)))
    with pytest.raises(ValueError, match="finite complete"):
        MujocoPlannedTcpMotion(rig=rig, task_config=None, pose_projector=None,
            reference_hz=60., source_model_qpos=source)


@pytest.mark.parametrize("external_cancel", [False, True])
def test_candidate_planner_lock_wait_cannot_restart_its_total_budget(monkeypatch, external_cancel):
    now, cancelled, outcomes = [0.], [False], []
    monkeypatch.setattr(planned_tcp_motion.time, "perf_counter", lambda: now[0])
    planner = _query_planner(lambda *a, **kw: pytest.fail("Expired query reached planning."))
    guard, waiting = threading.RLock(), threading.Event()

    class ContendedLock:
        def __enter__(self):
            waiting.set()
            guard.acquire()

        def __exit__(self, *args):
            guard.release()

    planner._lock = ContendedLock()

    def query():
        try:
            planner.plan_candidates([(0,)], translation_tolerance_m=.01, rotation_tolerance_rad=.1,
                cancel_requested=lambda: cancelled[0], time_budget_sec=1.)
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=query)
    try:
        with guard:
            worker.start()
            try:
                assert waiting.wait(timeout=2.)
                now[0] = 2.
                cancelled[0] = external_cancel
            finally:
                # Release even when the synchronization assertion fails.
                now[0] = 2.
    finally:
        worker.join(timeout=2.)
    assert not worker.is_alive() and len(outcomes) == 1
    assert isinstance(outcomes[0], MujocoTcpPlanRejected)
    assert outcomes[0].reason_code == (
        "safe_tcp_planning_cancelled" if external_cancel else "safe_tcp_no_candidate_path")
    assert planner._pending_plan is None


@pytest.mark.parametrize("incumbent,pruned", [(.01, True), (.1, False)])
def test_candidate_endpoint_bound_uses_every_admitted_ik_branch(monkeypatch, incumbent, pruned):
    planner, pose, calls = _endpoint_planner(monkeypatch, [.2, .05, .1, .1, .1, .1, .1, .1])
    oracle = planned_tcp_motion.MujocoJointSceneOracle(None, None)
    oracle.check = lambda pair: JointConfigurationCheck(safe=True)
    with pytest.raises(RuntimeError) as caught:
        planner.plan(pose, translation_tolerance_m=.01, rotation_tolerance_rad=.1,
            cancel_requested=lambda: False, incumbent_max_joint_excursion_rad=incumbent)
    assert len(calls["ik"]) == 8
    if pruned:
        assert caught.value.reason_code == "safe_tcp_candidate_rank_pruned"
        assert caught.value.details["lower_bound_max_joint_excursion_rad"] == pytest.approx(.05)
        assert not calls["path"]
    else:
        assert str(caught.value) == "path search reached"
        assert len(calls["path"]) == 1

def _query_planner(plan):
    planner = object.__new__(MujocoPlannedTcpMotion)
    planner._lock = threading.RLock()
    planner._active = planner._pending_plan = None
    planner.rig = SimpleNamespace(capture_scene=lambda fn: fn(None))
    planner.plan = plan
    return planner


def _endpoint_planner(monkeypatch, offsets):
    """Exercise the real IK-to-path boundary with controlled solver endpoints."""
    pose = (0., 0., 0., 0., 0., 0., 1.)
    start = np.array([.5, -.25, .1, 0., .2, -.3])
    model_qpos = np.concatenate((np.zeros(6), start))
    calls = {"ik": [], "checks": [], "path": []}

    def solve(**kwargs):
        calls["ik"].append(kwargs)
        return start + np.array([offsets[len(calls["ik"]) - 1], 0., 0., 0., 0., 0.])

    def check(pair):
        calls["checks"].append(pair)
        return JointConfigurationCheck(safe=not calls["ik"] or len(calls["ik"]) == len(offsets))

    oracle = SimpleNamespace(addresses={"left": np.arange(6), "right": np.arange(6, 12)},
        scene_hash="test:endpoints", check=check)
    monkeypatch.setattr(planned_tcp_motion, "MujocoJointSceneOracle", lambda *a, **kw: oracle)
    monkeypatch.setattr(planned_tcp_motion, "load_rm65_tool_profile", lambda *a, **kw: None)
    monkeypatch.setattr(planned_tcp_motion, "endpoint_to_tool_transform", lambda _: np.eye(4))

    def search(**kwargs):
        calls["path"].append(kwargs)
        raise RuntimeError("path search reached")

    monkeypatch.setattr(planned_tcp_motion, "plan_collision_aware_joint_path", search)
    scene = SimpleNamespace(model=SimpleNamespace(nq=12),
        data=SimpleNamespace(qpos=model_qpos), body_pose=lambda _: np.eye(4))
    rig = SimpleNamespace(scene=scene, capture_scene=lambda fn: fn(scene),
        config=SimpleNamespace(max_velocity_rad_s=np.ones(6), max_command_step_deg=1.,
            timing=SimpleNamespace(command_hz=60.)))
    projector = SimpleNamespace(project=lambda **kw: {"right_arm": pose})
    planner = MujocoPlannedTcpMotion(rig=rig,
        task_config=SimpleNamespace(right_tool_profile_path="unused"),
        pose_projector=projector, reference_hz=60.)
    planner._solver = SimpleNamespace(q_min=np.full(6, -4.), q_max=np.full(6, 4.),
        compute_pose=lambda _: np.eye(4), solve_iterative=solve)
    planner._pending_plan = object()
    return planner, pose, calls


def _preview_planner(monkeypatch, offsets):
    planner, pose, calls = _endpoint_planner(monkeypatch, offsets)
    planner._source_model_qpos = planner._model_qpos()
    planner.scene.arm_specs = {
        side: SimpleNamespace(joint_names=[str(i) for i in indexes])
        for side, indexes in (("left", range(6)), ("right", range(6, 12)))}
    planner.scene.joint_qpos_addr = int
    return planner, pose, calls


def test_current_seed_preview_orders_without_collision_or_candidate_removal(monkeypatch):
    planner, pose, calls = _preview_planner(monkeypatch, [2., np.nan, .2, .2])
    before = planner._model_qpos()
    order, estimates = planner._preview_candidate_order(
        [pose]*4, translation_tolerance_m=.01, rotation_tolerance_rad=.1,
        normalize_ik_objective=False, deadline=time.perf_counter()+10,
        cancel_requested=lambda: False)
    assert order == (2, 3, 0, 1)
    assert set(order) == set(range(4)) and len(estimates) == 4
    assert estimates[1]["estimated_max_joint_excursion_rad"] is None
    assert not estimates[1]["current_seed_ik_qualified"]
    assert all(not hint["collision_checked"] for hint in estimates.values())
    assert calls["checks"] == calls["path"] == []
    np.testing.assert_array_equal(planner._model_qpos(), before)
    for call in calls["ik"]:
        np.testing.assert_array_equal(call["q_init"], before[6:])


def test_preview_total_travel_does_not_depend_on_nearly_equal_max_joint_scores(monkeypatch):
    planner, pose, _ = _preview_planner(monkeypatch, [])
    start = planner._model_qpos()[6:]
    endpoints = iter([start+[.6, .2, 0, 0, 0, 0], start+[.6, 0, 0, 0, 0, 0],
        start+[.6, 0, 0, 0, 0, 0]])
    planner._solver.solve_iterative = lambda **kwargs: next(endpoints)
    order, estimates = planner._preview_candidate_order(
        [pose]*3, translation_tolerance_m=.01, rotation_tolerance_rad=.1,
        normalize_ik_objective=False, deadline=time.perf_counter()+10,
        cancel_requested=lambda: False)
    assert order == (1, 2, 0)
    assert estimates[0]["estimated_total_joint_excursion_rad"] == pytest.approx(.8)


def test_preview_time_is_charged_to_the_existing_whole_query_deadline(monkeypatch):
    now = [1.]
    monkeypatch.setattr(planned_tcp_motion.time, "perf_counter", lambda: now[0])
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="winner")
    path_budgets = []
    def plan(target, **kwargs):
        assert kwargs["deadline_monotonic"] == 2.
        path_budgets.append(kwargs["path_search_time_budget_sec"])
        now[0] += .2
        return candidate
    planner = _query_planner(plan)
    def preview(*args, **kwargs):
        assert kwargs["deadline"] == pytest.approx(1.1)
        # One non-preemptible preparation may overrun the cooperative preview
        # slice, but it must not receive a fresh full query budget afterwards.
        now[0] = 1.4
        return (1, 0), {}
    planner._preview_candidate_order = preview
    planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False,
        prioritize_current_seed=True, time_budget_sec=1.)
    assert path_budgets == pytest.approx([.6, .4])


def test_preview_total_cost_and_formal_max_joint_ranking_are_separate(monkeypatch):
    planner, pose, _ = _preview_planner(monkeypatch, [])
    start = planner._model_qpos()[6:]
    deltas = [np.array([.6, .6, 0, 0, 0, 0]), np.array([1., 0, 0, 0, 0, 0])]
    remaining = iter(deltas)
    planner._solver.solve_iterative = lambda **kwargs: start+next(remaining)
    visited = []
    plans = [SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([start, start+delta]),
        timestamps=np.array([0., 1.])), trajectory_hash=str(index)) for index, delta in enumerate(deltas)]
    def plan(target, **kwargs):
        index = 1 if not visited else 0
        visited.append(index)
        return plans[index]
    planner.plan = plan
    result = planner.plan_candidates([pose, pose], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False, prioritize_current_seed=True)
    assert [row["candidate_index"] for row in result.diagnostics] == [1, 0]
    assert result.candidate_index == 0 and result.plan is plans[0]


@pytest.mark.parametrize("cancel", [False, True])
def test_current_seed_preview_distinguishes_budget_exhaustion_from_cancellation(monkeypatch, cancel):
    planner, pose, calls = _preview_planner(monkeypatch, [1.])
    kwargs = {"translation_tolerance_m": .01, "rotation_tolerance_rad": .1,
        "normalize_ik_objective": False, "deadline": time.perf_counter()-1,
        "cancel_requested": lambda: cancel}
    if cancel:
        with pytest.raises(MujocoTcpPlanRejected, match="safe_tcp_planning_cancelled"):
            planner._preview_candidate_order([pose]*2, **kwargs)
    else:
        assert planner._preview_candidate_order([pose]*2, **kwargs) == ((0, 1), {})
    assert not calls["ik"]


@pytest.mark.parametrize("problem", ["mutable", "model_mismatch", "solver_error"])
def test_current_seed_preview_does_not_hide_contract_errors(monkeypatch, problem):
    planner, pose, calls = _preview_planner(monkeypatch, [1.])
    if problem == "mutable":
        planner._source_model_qpos = None
    elif problem == "model_mismatch":
        def other_pose(q):
            value = np.eye(4)
            value[0, 3] = .1
            return value
        planner._solver.compute_pose = other_pose
    else:
        def fail(**kwargs):
            raise RuntimeError("native solver failure")
        planner._solver.solve_iterative = fail
    with pytest.raises((ValueError, RuntimeError)):
        planner._preview_candidate_order([pose]*2, translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, normalize_ik_objective=False,
            deadline=time.perf_counter()+10, cancel_requested=lambda: False)
    assert not calls["path"]


@pytest.mark.parametrize("normalized", [False, True])
def test_preview_uses_shared_calibrated_ik_and_independent_pose_tolerances(monkeypatch, normalized):
    planner, pose, calls = _preview_planner(monkeypatch, [0., 0.])
    weights = []
    solver = planner._solver
    def get_solver(weight):
        weights.append(weight)
        return solver
    planner._solver_for_ik = get_solver
    far = (.1, *pose[1:])
    order, estimates = planner._preview_candidate_order(
        [far, pose], translation_tolerance_m=.003, rotation_tolerance_rad=.02,
        normalize_ik_objective=normalized, deadline=time.perf_counter()+10,
        cancel_requested=lambda: False)
    assert order == (1, 0)
    assert estimates[0]["current_seed_ik_qualified"] is False
    assert weights == ([.15] if normalized else [None])
    assert all(("eef_from_controlled_frame" in call) == normalized for call in calls["ik"])
    assert all(call["max_iterations"] == 100 for call in calls["ik"])


@pytest.mark.parametrize("external_cancel", [False, True])
def test_preview_deadline_inside_solver_preserves_unvisited_candidates(monkeypatch, external_cancel):
    planner, pose, calls = _preview_planner(monkeypatch, [0.])
    now = [1.]
    cancelled = [False]
    monkeypatch.setattr(planned_tcp_motion.time, "perf_counter", lambda: now[0])
    def solve(**kwargs):
        assert kwargs["deadline_monotonic"] == 2.
        now[0] = 3.
        cancelled[0] = external_cancel
        kwargs["check_cancelled"]()
    planner._solver.solve_iterative = solve
    kwargs = {"translation_tolerance_m": .01, "rotation_tolerance_rad": .1,
        "normalize_ik_objective": False, "deadline": 2.,
        "cancel_requested": lambda: cancelled[0]}
    if external_cancel:
        with pytest.raises(MujocoTcpPlanRejected, match="safe_tcp_planning_cancelled"):
            planner._preview_candidate_order([pose]*3, **kwargs)
    else:
        assert planner._preview_candidate_order([pose]*3, **kwargs) == ((0, 1, 2), {})
    assert not calls["path"]


def test_preview_priority_still_runs_full_checks_and_keeps_original_indices():
    calls = []
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="winner")
    def plan(target, **kwargs):
        calls.append(target[0])
        if target[0] == 2:
            raise MujocoTcpPlanRejected("safe_tcp_target_blocked")
        return candidate
    planner = _query_planner(plan)
    def preview(targets, **kwargs):
        assert 0 < kwargs["deadline"]-time.perf_counter() <= .5
        return (2, 1, 0), {i: {"collision_checked": False} for i in range(3)}
    planner._preview_candidate_order = preview
    result = planner.plan_candidates([(0,), (1,), (2,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False, prioritize_current_seed=True)
    assert calls == [2, 1, 0] and result.candidate_index == 1
    assert [row["candidate_index"] for row in result.diagnostics] == calls
    assert result.diagnostics[0]["reason_code"] == "safe_tcp_target_blocked"
    assert all(row["scheduling_preview"] == {"collision_checked": False} for row in result.diagnostics)
    assert planner._pending_plan is candidate and result.all_candidates_checked


def test_preview_exception_clears_existing_pending_plan():
    planner = _query_planner(lambda *args, **kwargs: pytest.fail("Must not plan"))
    planner._pending_plan = object()
    def fail(*args, **kwargs):
        raise RuntimeError("bad preview binding")
    planner._preview_candidate_order = fail
    with pytest.raises(RuntimeError, match="bad preview binding"):
        planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, cancel_requested=lambda: False, prioritize_current_seed=True)
    assert planner._pending_plan is None


def test_normalized_ik_cache_rebuilds_on_mode_or_tolerance_ratio_change(monkeypatch):
    planner, pose, calls = _endpoint_planner(monkeypatch, [0.])
    solver, configurations = planner._solver, []

    def build(config):
        configurations.append(config)
        return solver

    monkeypatch.setattr(planned_tcp_motion, "build_rm65_single_arm_ik_solver", build)
    for normalized, tolerance_m in [(False, .01), (True, .003), (True, .003), (True, .002), (False, .01)]:
        calls["ik"].clear()
        with pytest.raises(RuntimeError, match="path search reached"):
            planner.plan(pose, translation_tolerance_m=tolerance_m, rotation_tolerance_rad=np.deg2rad(1.),
                         normalize_ik_objective=normalized, cancel_requested=lambda: False)
        if normalized:
            np.testing.assert_array_equal(calls["ik"][0]["eef_from_controlled_frame"], np.eye(4))
        else:
            assert "eef_from_controlled_frame" not in calls["ik"][0]
    assert len(configurations) == 3
    for config, tolerance_m in zip(configurations[:2], (.003, .002)):
        objective = config.ik_tuning.objective_weights()
        assert objective.single_arm_position_weight == 1.
        assert objective.single_arm_orientation_weight == pytest.approx(tolerance_m / np.deg2rad(1.))
    assert configurations[-1].ik_tuning is None


@pytest.mark.parametrize("normalized", [False, True])
def test_precise_planner_binds_nonidentity_tcp_offset_once(monkeypatch, normalized):
    from scipy.spatial.transform import Rotation

    planner, pose, calls = _endpoint_planner(monkeypatch, [0.])
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .3]).as_matrix()
    offset[:3, 3] = [.02, -.01, .14]
    monkeypatch.setattr(planned_tcp_motion, "endpoint_to_tool_transform", lambda _: offset)
    planner._solver.compute_pose = lambda _: np.linalg.inv(offset)
    planner._solver_orientation_weight = .003 / np.deg2rad(1.) if normalized else None
    with pytest.raises(RuntimeError, match="path search reached"):
        planner.plan(pose, translation_tolerance_m=.003, rotation_tolerance_rad=np.deg2rad(1.),
                     normalize_ik_objective=normalized, cancel_requested=lambda: False)
    arguments = calls["ik"][0]
    if normalized:
        np.testing.assert_allclose(arguments["target_pose"], np.eye(4))
        np.testing.assert_array_equal(arguments["eef_from_controlled_frame"], offset)
    else:
        np.testing.assert_allclose(arguments["target_pose"], np.linalg.inv(offset))
        assert "eef_from_controlled_frame" not in arguments


def test_candidate_selection_retains_owned_best_not_last_success():
    plans = [SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [delta]]),
             timestamps=np.array([0., duration])), trajectory_hash=str(index))
             for index, (delta, duration) in enumerate(((2., 1.), (.5, 3.), (1., 2.)))]

    def plan(target, **kwargs):
        del kwargs
        selected = plans[target[0]]
        planner._pending_plan = selected
        return selected

    planner = _query_planner(plan)
    selection = planner.plan_candidates([(0,), (1,), (2,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False)
    assert selection.candidate_index == 1 and selection.plan is plans[1]
    assert planner._pending_plan is plans[1] and selection.all_candidates_checked
    assert not planner.active


@pytest.mark.parametrize("has_best", [False, True])
@pytest.mark.parametrize("failure", ["safe_tcp_collision_check_unavailable", "safe_tcp_ik_model_tool_mismatch",
    "safe_tcp_collision_geometry_unavailable", "safe_tcp_start_blocked", "safe_tcp_invalid_request"])
def test_candidate_selection_does_not_fallback_on_scene_or_binding_fault(failure, has_best):
    calls = []
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="candidate")

    def plan(target, **kwargs):
        del kwargs
        calls.append(1)
        if has_best and target[0] == 0:
            planner._pending_plan = candidate
            return candidate
        raise MujocoTcpPlanRejected(failure)

    planner = _query_planner(plan)
    with pytest.raises(MujocoTcpPlanRejected, match=failure):
        planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, cancel_requested=lambda: False)
    assert len(calls) == 1 + has_best and planner._pending_plan is None


@pytest.mark.parametrize("duration", [1., 2.], ids=["shorter_duration", "less_travel"])
def test_candidate_rank_pruning_retains_owned_winner_and_secondary_ranking(duration):
    first = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.5], [-.5], [.5]]),
        timestamps=np.array([0., 2.])), trajectory_hash="first")
    winner = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.5]]),
        timestamps=np.array([0., duration])), trajectory_hash="winner")
    bounds = []

    def plan(target, **kwargs):
        bounds.append(kwargs.get("incumbent_max_joint_excursion_rad"))
        planner._pending_plan = None
        if target[0] % 2:
            raise MujocoTcpPlanRejected("safe_tcp_candidate_rank_pruned", {"path_search_skipped": True})
        planner._pending_plan = first if target[0] == 0 else winner
        return planner._pending_plan

    planner = _query_planner(plan)
    result = planner.plan_candidates([(0,), (1,), (2,), (3,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False)
    assert bounds == [None, .5, .5, .5]
    assert result.candidate_index == 2 and result.plan is winner
    assert planner._pending_plan is winner and not planner.active
    assert result.all_candidates_checked and len(result.diagnostics) == 4
    assert [item["reason_code"] for item in result.diagnostics] == [
        "path_found", "safe_tcp_candidate_rank_pruned", "path_found", "safe_tcp_candidate_rank_pruned"]


@pytest.mark.parametrize("reason", ["safe_tcp_endpoint_search_exhausted", "safe_tcp_target_blocked",
    "safe_tcp_path_search_exhausted", "safe_tcp_retimed_path_blocked"])
def test_failed_first_candidate_does_not_establish_rank_incumbent(reason):
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="candidate")

    def plan(target, **kwargs):
        assert "incumbent_max_joint_excursion_rad" not in kwargs
        if target[0] == 0:
            raise MujocoTcpPlanRejected(reason)
        return candidate

    planner = _query_planner(plan)
    result = planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False)
    assert result.candidate_index == 1 and planner._pending_plan is candidate
    assert result.all_candidates_checked and result.diagnostics[0]["reason_code"] == reason


def test_candidate_selection_budget_and_cancel_leave_no_dispatchable_plan():
    planner = _query_planner(lambda *a, **kw: pytest.fail("No candidate should be planned."))
    with pytest.raises(MujocoTcpPlanRejected, match="no_candidate_path"):
        planner.plan_candidates([(0,)], translation_tolerance_m=.01, rotation_tolerance_rad=.1,
            cancel_requested=lambda: False, time_budget_sec=1e-12)
    with pytest.raises(MujocoTcpPlanRejected, match="planning_cancelled"):
        planner.plan_candidates([(0,)], translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, cancel_requested=lambda: True)
    assert planner._pending_plan is None


def test_candidate_query_reference_budget_rejects_long_path_before_ranking():
    long = SimpleNamespace(reference_poses=(1, 2, 3))
    short = SimpleNamespace(reference_poses=(1,), trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="test")

    def plan(target, **kwargs):
        assert kwargs["deadline_monotonic"] > 0
        assert "incumbent_max_joint_excursion_rad" not in kwargs
        return (long, short)[target[0]]

    planner = _query_planner(plan)
    result = planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01, rotation_tolerance_rad=.1,
        cancel_requested=lambda: False, max_reference_count=3)
    assert result.candidate_index == 1 and result.plan is short
    assert result.diagnostics[0]["reason_code"] == "safe_tcp_path_exceeds_control_budget"


def test_candidate_deadline_keeps_previous_owned_winner(monkeypatch):
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="candidate")
    now = [0.]
    monkeypatch.setattr(planned_tcp_motion.time, "perf_counter", lambda: now[0])

    def plan(target, **kwargs):
        if target[0] == 1:
            now[0] = 20.
            assert kwargs["cancel_requested"]()
            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
        planner._pending_plan = candidate
        return candidate

    planner = _query_planner(plan)
    result = planner.plan_candidates([(0,), (1,), (2,)], translation_tolerance_m=.01,
        rotation_tolerance_rad=.1, cancel_requested=lambda: False, time_budget_sec=10.)
    assert result.candidate_index == 0 and result.plan is planner._pending_plan is candidate
    assert not result.all_candidates_checked and len(result.diagnostics) == 1


def test_cancel_after_one_candidate_does_not_restore_the_earlier_winner():
    calls = []
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="candidate")

    def plan(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        planner._pending_plan = candidate
        return candidate

    planner = _query_planner(plan)
    with pytest.raises(MujocoTcpPlanRejected, match="planning_cancelled"):
        planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, cancel_requested=lambda: bool(calls))
    assert len(calls) == 1 and planner._pending_plan is None


def test_candidate_query_unexpected_error_clears_a_previous_pending_winner():
    calls = []
    candidate = SimpleNamespace(trajectory=SimpleNamespace(q_rad=np.array([[0.], [.1]]),
        timestamps=np.array([0., 1.])), trajectory_hash="candidate")

    def plan(*args, **kwargs):
        del args, kwargs
        if calls:
            raise RuntimeError("query failed")
        calls.append(1)
        planner._pending_plan = candidate
        return candidate

    planner = _query_planner(plan)
    with pytest.raises(RuntimeError, match="query failed"):
        planner.plan_candidates([(0,), (1,)], translation_tolerance_m=.01,
            rotation_tolerance_rad=.1, cancel_requested=lambda: False)
    assert planner._pending_plan is None


