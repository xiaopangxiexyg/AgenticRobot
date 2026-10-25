"""Offline side binding over fake scene/IK boundaries; never build MuJoCo."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import ExecutionObservationPlan
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.observed_planning import ObservedJointPlanningInput
from robot_motion.position_control import POSITION_15_60_120_PROFILE
from sim.mujoco.deploy import agent_skill_target as targets
from sim.mujoco.deploy import planned_tcp_motion as planning

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_ADDRESSES = {"left": np.arange(6), "right": np.arange(6, 12)}
_OFFSETS = {"left": np.array([0.01, 0.02, 0.03]), "right": np.array([-0.04, 0.05, 0.06])}


def _matrix(xyz):
    value = np.eye(4)
    value[:3, 3] = xyz
    return value


def _fixture(monkeypatch, *, motion_side=None):
    """Keep the real search, ranking and retiming; fake only model boundaries."""
    scene = SimpleNamespace(
        data=SimpleNamespace(qpos=np.array([
            0.1, 0.2, 0.3, 0., 0., 0., -0.2, -0.1, 0.4, 0., 0., 0., 1., 2., 3., 4.,
        ])),
        model=SimpleNamespace(nq=16),
    )
    calls = SimpleNamespace(oracles=[], checks=[], tools=[], bases=[], projections=[], seeds=[], searches=[], projection_inputs=[],
                            sweep_modified=False)

    def body_pose(name):
        calls.bases.append(name)
        result = _matrix([0.7, 0.8, 0.9])
        if name == "left_arm_base_link":
            result[:2, :2] = [[0., -1.], [1., 0.]]
        else:
            assert name == "right_arm_base_link"
            result[:2, :2] = [[0., 1.], [-1., 0.]]
        return result

    scene.body_pose = body_pose
    scene.arm_qpos_by_side = lambda: {
        side: scene.data.qpos[address].copy() for side, address in _ADDRESSES.items()
    }
    rig = SimpleNamespace(
        scene=scene,
        capture_scene=lambda capture: capture(scene),
        measured_qpos_by_side=scene.arm_qpos_by_side,
        config=SimpleNamespace(
            max_velocity_rad_s=(2.,) * 6,
            max_acceleration_rad_s2=(4.,) * 6,
            max_command_step_deg=10.,
            timing=SimpleNamespace(command_hz=30.),
            velocity_ramp_sec=0.2,
        ),
    )

    class Projector:
        def project(self, *, model_qpos, qpos_by_source, component_ids):
            assert np.asarray(model_qpos).shape == (16,)
            calls.projections.append(component_ids)
            calls.projection_inputs.append(np.asarray(model_qpos).copy())
            return {
                component: (
                    *(np.asarray(qpos_by_source[component.removesuffix("_arm")])[:3]
                      + _OFFSETS[component.removesuffix("_arm")]), 0., 0., 0., 1.,
                )
                for component in component_ids
            }

    class Solver:
        q_min = np.full(6, -2.)
        q_max = np.full(6, 2.)

        def compute_pose(self, qpos):
            return _matrix(np.asarray(qpos)[:3])

        def solve_iterative(self, *, target_pose, q_init, **kwargs):
            calls.seeds.append(np.asarray(q_init).copy())
            return np.r_[target_pose[:3, 3], np.zeros(3)]

    class Oracle:
        scene_hash = "fixture:complete-scene"
        addresses = _ADDRESSES

        def __init__(self, scene, model_qpos, *, configuration_constraint=None,
                     constraint_identity=None, cancel_requested=None, **kwargs):
            self.source = SimpleNamespace(qpos=np.asarray(model_qpos).copy())
            self.configuration_constraint = configuration_constraint
            self.constraint_identity = constraint_identity
            self.cancel_requested = cancel_requested
            self.projector = SimpleNamespace(project=self.sweep)
            calls.oracles.append(self)

        def check(self, pair):
            assert set(pair) == {"left", "right"}
            calls.checks.append({side: np.asarray(q).copy() for side, q in pair.items()})
            if self.cancel_requested is not None and self.cancel_requested():
                raise planning.MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if self.configuration_constraint is not None:
                return self.configuration_constraint(self.source.qpos, pair)
            return JointConfigurationCheck(safe=True)

        def sweep(self, source, *, start_qpos, requested_qpos):
            assert source.qpos.shape == (16,)
            assert start_qpos.shape == requested_qpos.shape == (12,)
            return SimpleNamespace(
                modified=calls.sweep_modified,
                check=JointConfigurationCheck(safe=True),
                to_dict=lambda: {"fixture_sweep_modified": calls.sweep_modified},
            )

    def load_tool(path, *, side):
        calls.tools.append((path, side))
        return {"arm": side, "tcp": {"pose_flange_m_rad": [*_OFFSETS[side], 0., 0., 0.]}}

    solver = Solver()
    monkeypatch.setattr(planning, "MujocoJointSceneOracle", Oracle)
    monkeypatch.setattr(planning, "build_rm65_single_arm_ik_solver", lambda _config: solver)
    monkeypatch.setattr(planning, "load_rm65_tool_profile", load_tool)
    # The existing path hash includes the remaining planner wall-time budget.
    monkeypatch.setattr(planning, "time", SimpleNamespace(perf_counter=lambda: 100.))
    search = planning.plan_collision_aware_joint_path

    def record_search(**kwargs):
        calls.searches.append(kwargs)
        return search(**kwargs)

    monkeypatch.setattr(planning, "plan_collision_aware_joint_path", record_search)
    projector = Projector()
    kwargs = {} if motion_side is None else {"motion_side": motion_side}
    planner = planning.MujocoPlannedTcpMotion(
        rig=rig,
        task_config=SimpleNamespace(left_tool_profile_path="fixture:left-tool.json",
                                    right_tool_profile_path="fixture:right-tool.json"),
        pose_projector=projector,
        reference_hz=15.,
        **kwargs,
    )
    return SimpleNamespace(planner=planner, rig=rig, projector=projector, solver=solver,
                           calls=calls, side=motion_side or "right")


def _plan(fixture, **kwargs):
    pair = fixture.rig.measured_qpos_by_side()
    pose = fixture.projector.project(
        model_qpos=fixture.rig.scene.data.qpos,
        qpos_by_source=pair,
        component_ids=(fixture.side + "_arm",),
    )[fixture.side + "_arm"]
    target = tuple(np.r_[np.asarray(pose[:3]) + [0.02, 0., 0.], pose[3:]])
    return fixture.planner.plan(
        target,
        translation_tolerance_m=0.003,
        rotation_tolerance_rad=0.03,
        cancel_requested=kwargs.pop("cancel_requested", lambda: False),
        **kwargs,
    )


def _observed_input(fixture, *, check=None, clock=lambda: 10.):
    scene = fixture.rig.scene
    scene.model.qpos0 = np.zeros(16)
    scene.arm_specs = {s: SimpleNamespace(joint_names=tuple(s + str(i) for i in range(6)))
                       for s in ("left", "right")}
    scene.joint_qpos_addr = lambda name: next(
        _ADDRESSES[s][i] for s in ("left", "right") for i in range(6) if name == s + str(i))
    oracle = SimpleNamespace(scene_hash="sensor:rgbd-fixed",
        check=check or (lambda pair: JointConfigurationCheck(safe=True)))
    return ObservedJointPlanningInput(oracle, fixture.rig.measured_qpos_by_side(),
        acquired_at=10., maximum_age_sec=30., maximum_start_drift_rad=.01, clock=clock)


@pytest.mark.parametrize("side", ("left", "right"))
def test_observed_search_does_not_consult_native_objects_but_execution_keeps_native_guard(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    source = _observed_input(fixture)
    native_plan = fixture.planner.plan

    def observe_planner(*args, **kwargs):
        # Exclude the helper's measured-TCP target construction; observe the
        # actual planner FK inputs without replacing its implementation.
        fixture.calls.projection_inputs.clear()
        return native_plan(*args, **kwargs)

    monkeypatch.setattr(fixture.planner, "plan", observe_planner)
    first = _plan(fixture, observed_planning_input=source)
    assert not fixture.calls.oracles
    fixture.rig.scene.data.qpos[12:] = [100., 200., 300., 400.]
    second = _plan(fixture, observed_planning_input=source)
    assert first.trajectory_hash == second.trajectory_hash
    assert first.path.semantic_content_hash == second.path.semantic_content_hash
    assert second.path.metadata["planning_scene_source"] == "calibrated_rgbd_and_measured_joints"
    # Native lifecycle diagnostics retain the real cut; identical observed
    # paths above prove these object coordinates were not used for selection.
    np.testing.assert_array_equal(second.source_model_qpos[12:], [100., 200., 300., 400.])
    assert fixture.calls.projection_inputs
    assert all(np.array_equal(q[12:], np.zeros(4)) for q in fixture.calls.projection_inputs)
    assert not fixture.calls.oracles
    fixture.planner.activate(second)
    assert len(fixture.calls.oracles) == 1
    np.testing.assert_array_equal(fixture.calls.oracles[0].source.qpos[12:], [100., 200., 300., 400.])
    fixture.calls.sweep_modified = True
    action = SimpleNamespace(arm_pose_by_side={side: fixture.planner.reference(0)})
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_live_reference_sweep_blocked"):
        fixture.planner.resolve(action)


@pytest.mark.parametrize("stage", ("before", "after", "activation"))
def test_observed_scene_age_checked_at_original_plan_and_activation_boundaries(monkeypatch, stage):
    fixture = _fixture(monkeypatch)
    now = [10.]
    def check(pair):
        if stage == "after":
            now[0] = 41.
        return JointConfigurationCheck(safe=True)
    source = _observed_input(fixture, check=check, clock=lambda: now[0])
    if stage == "before":
        now[0] = 41.
    if stage == "activation":
        plan = _plan(fixture, observed_planning_input=source)
        now[0] = 41.
        call = lambda: fixture.planner.activate(plan)
    else:
        call = lambda: _plan(fixture, observed_planning_input=source)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="observed_planning_scene_stale"):
        call()
    assert not fixture.planner.active
    assert not fixture.calls.oracles


def test_observed_activation_rechecks_age_after_native_oracle_construction(monkeypatch):
    fixture = _fixture(monkeypatch)
    now = [10.]
    plan = _plan(fixture, observed_planning_input=_observed_input(fixture, clock=lambda: now[0]))
    original = fixture.planner._oracle_for_stage

    def cross_age(stage):
        oracle = original(stage)
        now[0] = 41.
        return oracle

    monkeypatch.setattr(fixture.planner, "_oracle_for_stage", cross_age)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="observed_planning_scene_stale"):
        fixture.planner.activate(plan)
    assert not fixture.planner.active and fixture.planner._pending_plan is plan
    assert len(fixture.calls.oracles) == 1


def test_observed_freshness_work_cannot_cross_the_final_activation_deadline(monkeypatch):
    fixture = _fixture(monkeypatch)
    deadline_clock, in_consumption = [10.], [False]

    def observation_clock():
        if in_consumption[0]: deadline_clock[0] = 21.
        return 10.

    plan = _plan(fixture, observed_planning_input=_observed_input(fixture, clock=observation_clock))
    original = fixture.planner._oracle_for_stage

    def enter_consumption(stage):
        oracle = original(stage)
        in_consumption[0] = True
        return oracle

    monkeypatch.setattr(fixture.planner, "_oracle_for_stage", enter_consumption)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="grasp_path_plan_expired"):
        fixture.planner.activate(plan, deadline_at=20., clock=lambda: deadline_clock[0])
    assert not fixture.planner.active and fixture.planner._pending_plan is plan


def test_observed_plan_cannot_refine_through_a_native_truth_query(monkeypatch):
    fixture = _fixture(monkeypatch)
    plan = _plan(fixture, observed_planning_input=_observed_input(fixture))
    goal = planning.MujocoTcpPlanGoal(plan.reference_poses[-1], .003, .03)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="observed_chain_refinement_unsupported"):
        fixture.planner.refine_pending_chain(plan, (goal,), cancel_requested=lambda: False,
            time_budget_sec=1., deadline_at=20., monotonic=lambda: 10., validate_plan=lambda _: None)
    assert fixture.planner._pending_plan is plan and not fixture.calls.oracles


def test_observed_collision_and_envelope_reject_before_native_query(monkeypatch):
    fixture = _fixture(monkeypatch)
    source = _observed_input(fixture, check=lambda pair: JointConfigurationCheck(safe=False, reason="sensor_obstacle"))
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_start_blocked"):
        _plan(fixture, observed_planning_input=source)
    source = _observed_input(fixture)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_start_blocked"):
        _plan(fixture, observed_planning_input=source,
              configuration_constraint=lambda _model, _pair: JointConfigurationCheck(safe=False),
              constraint_identity="fixture:envelope")
    assert not fixture.calls.oracles


def _right_contact_binding():
    return planning.MujocoGraspContactBinding(
        target_body_name="fixture:target",
        sensor_geom_names=("fixture:right-fingertip",),
        lateral_tolerance_m=0.03,
    )


@pytest.mark.parametrize("preview", (False, True))
@pytest.mark.parametrize("valid_binding", (False, True))
def test_left_contact_plan_is_rejected_before_capture_or_native_work(
    monkeypatch, preview, valid_binding,
):
    fixture = _fixture(monkeypatch, motion_side="left")
    planner = fixture.planner
    pending = object()
    planner._pending_plan = pending

    def unexpected(*_args, **_kwargs):
        pytest.fail("Unsupported left contact must reject before callbacks or model work")

    monkeypatch.setattr(fixture.rig, "capture_scene", unexpected)
    monkeypatch.setattr(fixture.projector, "project", unexpected)
    monkeypatch.setattr(planning, "MujocoJointSceneOracle", unexpected)
    monkeypatch.setattr(planning, "build_rm65_single_arm_ik_solver", unexpected)
    binding = _right_contact_binding() if valid_binding else False
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        planner.plan(
            (0., 0., 0., 0., 0., 0., 1.),
            translation_tolerance_m=0.003,
            rotation_tolerance_rad=0.03,
            cancel_requested=unexpected,
            contact_binding=binding,
            preview_model_qpos=np.zeros(16) if preview else None,
        )
    assert error.value.reason_code == "safe_tcp_left_grasp_contact_unsupported"
    assert planner._pending_plan is pending
    assert not planner.active
    assert not any((
        fixture.calls.oracles, fixture.calls.checks, fixture.calls.tools,
        fixture.calls.bases, fixture.calls.projections, fixture.calls.seeds,
        fixture.calls.searches,
    ))


@pytest.mark.parametrize("valid_binding", (False, True))
def test_left_measured_contact_is_rejected_before_capture_or_native_work(
    monkeypatch, valid_binding,
):
    fixture = _fixture(monkeypatch, motion_side="left")

    def unexpected(*_args, **_kwargs):
        pytest.fail("Unsupported left contact must reject before model work")

    monkeypatch.setattr(fixture.rig, "capture_scene", unexpected)
    monkeypatch.setattr(fixture.projector, "project", unexpected)
    monkeypatch.setattr(planning, "MujocoJointSceneOracle", unexpected)
    binding = _right_contact_binding() if valid_binding else False
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        fixture.planner.check_measured_contact(binding, (0., 0., 0., 0., 0., 0., 1.))
    assert error.value.reason_code == "safe_tcp_left_grasp_contact_unsupported"
    assert fixture.planner._pending_plan is None
    assert not fixture.planner.active
    assert not fixture.calls.oracles


def test_right_contact_plan_and_measured_check_remain_available(monkeypatch):
    fixture = _fixture(monkeypatch)
    binding = _right_contact_binding()
    plan = _plan(fixture, contact_binding=binding)
    assert plan.contact_binding is binding
    assert fixture.planner._pending_plan is plan
    assert fixture.planner.check_measured_contact(binding, plan.reference_poses[-1]).safe


def test_default_right_baseline_fingerprint(monkeypatch):
    fixture = _fixture(monkeypatch)
    before = fixture.rig.scene.data.qpos.copy()
    plan = _plan(fixture)
    assert plan.path.movable_components == ("right",)
    np.testing.assert_array_equal(fixture.rig.scene.data.qpos, before)
    # Captured from 79b6 with this fake scene/IK and fixed planner clock.
    assert plan.path.semantic_content_hash == (
        "32853efee2b9967e8d2870f34f5ec02a09d7be346dbc3851ba9b60b971277c28"
    )
    assert plan.trajectory_hash == (
        "fd91525610e12ac942da4f3692bcff30b9dc9f8cf3dfbd6c6b1814f9894640fe"
    )
    assert fixture.planner.motion_side == "right"
    assert fixture.planner.component_id == "right_arm"


@pytest.mark.parametrize("side", ("left", "right"))
def test_plan_uses_selected_tool_seed_base_limits_and_complete_scene(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    before = fixture.rig.scene.data.qpos.copy()
    inactive = "right" if side == "left" else "left"
    plan = _plan(fixture)

    assert fixture.planner.motion_side == side
    assert fixture.planner.component_id == side + "_arm"
    assert fixture.calls.tools == [(f"fixture:{side}-tool.json", side)]
    assert set(fixture.calls.bases) == {side + "_arm_base_link"}
    assert set(fixture.calls.projections) == {(side + "_arm",)}
    np.testing.assert_array_equal(fixture.calls.seeds[0], before[_ADDRESSES[side]])
    assert plan.path.movable_components == (side,)
    search = fixture.calls.searches[0]
    assert set(search["joint_lower_by_component"]) == {side}
    assert set(search["joint_upper_by_component"]) == {side}
    np.testing.assert_array_equal(search["joint_lower_by_component"][side], fixture.solver.q_min)
    np.testing.assert_array_equal(search["joint_upper_by_component"][side], fixture.solver.q_max)
    np.testing.assert_array_equal(plan.source_model_qpos, before)
    assert not plan.source_model_qpos.flags.writeable
    np.testing.assert_array_equal(fixture.rig.scene.data.qpos, before)
    np.testing.assert_allclose(plan.trajectory.q_rad[-1] - before[_ADDRESSES[side]],
                               [0.02, 0., 0., 0., 0., 0.], atol=1e-12)
    for pair in [*fixture.calls.checks, *plan.path.waypoints]:
        assert set(pair) == {"left", "right"}
        np.testing.assert_array_equal(pair[inactive], before[_ADDRESSES[inactive]])
    assert not plan.path.execution_certified
    assert not plan.path.real_motion_authorized


def test_explicit_right_preserves_the_default_path_and_hash(monkeypatch):
    default = _plan(_fixture(monkeypatch))
    explicit = _plan(_fixture(monkeypatch, motion_side="right"))
    assert default.path.to_dict() == explicit.path.to_dict()
    assert default.reference_poses == explicit.reference_poses
    assert default.trajectory_hash == explicit.trajectory_hash
    np.testing.assert_array_equal(default.trajectory.q_rad, explicit.trajectory.q_rad)
    np.testing.assert_array_equal(default.trajectory.timestamps, explicit.trajectory.timestamps)


@pytest.mark.parametrize("bad", (None, True, False, "", "LEFT", " left", "right ",
                                  "both", ("left", "right"), ["left"], {"left"}))
def test_invalid_motion_side_is_rejected_before_resource_preparation(monkeypatch, bad):
    def unexpected_prepare(_config):
        pytest.fail("Invalid side must not prepare resources")

    monkeypatch.setattr(targets, "prepare_mujoco_sim_hil_target", unexpected_prepare)
    with pytest.raises(ValueError, match="motion_side"):
        planning.MujocoPlannedTcpMotion(
            rig=None, task_config=None, pose_projector=None, reference_hz=15., motion_side=bad,
        )
    with pytest.raises(ValueError, match="motion_side"):
        targets.build_mujoco_agent_skill_motion_target(
            config=targets.MujocoSimHILTargetConfig(task_config_path=Path("not-opened.json"),
                                                   allow_scripted_task_reset=False),
            component_execution_binding=None,
            motion_side=bad,
        )
    with pytest.raises(ValueError, match="motion_side"):
        targets.MujocoAgentSkillMotionTarget(
            task_config=None, scene=None, runtime=None, target_frame_profile=None,
            tool_frame_profile=None, task_initialization_info={}, motion_side=bad,
        )


@pytest.mark.parametrize("side", ("left", "right"))
def test_side_and_component_properties_are_read_only(monkeypatch, side):
    planner = _fixture(monkeypatch, motion_side=side).planner
    target = targets.MujocoAgentSkillMotionTarget(
        task_config=None, scene=None, runtime=None, target_frame_profile=None,
        tool_frame_profile=None, task_initialization_info={}, motion_side=side,
    )
    assert target.motion_side == side
    assert target.component_id == side + "_arm"
    for name, value in (("motion_side", "right" if side == "left" else "left"),
                        ("component_id", "other_arm")):
        with pytest.raises(AttributeError):
            setattr(planner, name, value)
        with pytest.raises(FrozenInstanceError):
            setattr(target, name, value)


@pytest.mark.parametrize("side", ("left", "right"))
def test_selected_tool_fk_mismatch_is_still_rejected(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    other = "right" if side == "left" else "left"
    monkeypatch.setattr(planning, "load_rm65_tool_profile", lambda *_args, **_kwargs: {
        "arm": other, "tcp": {"pose_flange_m_rad": [*_OFFSETS[other], 0., 0., 0.]},
    })
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_ik_model_tool_mismatch"):
        _plan(fixture)
    assert fixture.planner._pending_plan is None


@pytest.mark.parametrize("side", ("left", "right"))
def test_resolve_keeps_idle_qpos_and_rejects_drift_at_the_original_limit(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    plan = _plan(fixture)
    planner = fixture.planner
    inactive = "right" if side == "left" else "left"
    planner.activate(plan)
    pose = planner.reference(0)
    action = SimpleNamespace(arm_pose_by_side={side: pose})
    result = planner.resolve(action)
    assert set(result.qpos_by_side) == {"left", "right"}
    assert result.position_error_m == result.orientation_error_rad == {side: 0.}
    np.testing.assert_array_equal(result.qpos_by_side[inactive],
                                  plan.path.start_qpos_by_component[inactive])
    address = _ADDRESSES[inactive][0]
    fixture.rig.scene.data.qpos[address] += 0.0009
    planner.resolve(action)
    fixture.rig.scene.data.qpos[address] += 0.0002
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_passive_arm_changed"):
        planner.resolve(action)


@pytest.mark.parametrize("side", ("left", "right"))
def test_live_background_constraint_and_sweep_remain_fail_closed(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)

    def constraint(model_qpos, _pair):
        return JointConfigurationCheck(safe=model_qpos[12] < 1.5, reason="fixture:background")

    plan = _plan(fixture, configuration_constraint=constraint, constraint_identity="fixture:constraint")
    planner = fixture.planner
    planner.activate(plan)
    action = SimpleNamespace(arm_pose_by_side={side: planner.reference(0)})
    fixture.calls.sweep_modified = True
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_live_reference_sweep_blocked"):
        planner.resolve(action)
    fixture.calls.sweep_modified = False
    fixture.rig.scene.data.qpos[12] = 2.
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_live_scene_blocked") as error:
        planner.resolve(action)
    assert error.value.details["changed_background_qpos"]["12"] == [1., 2.]


@pytest.mark.parametrize("side", ("left", "right"))
def test_plan_ownership_preview_and_reference_identity_are_retained(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    planner = fixture.planner
    owned = _plan(fixture)
    preview = _plan(fixture, preview_model_qpos=fixture.rig.scene.data.qpos.copy())
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_plan_not_owned_by_runtime"):
        planner.activate(preview)
    assert planner._pending_plan is owned
    planner.activate(owned)
    pose = planner.reference(0)
    changed = list(pose)
    changed[0] += 0.001
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_reference_changed_after_safety"):
        planner.resolve(SimpleNamespace(arm_pose_by_side={side: changed}))
    assert planner.reference(0) == pose
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_reference_sequence_changed"):
        planner.reference(2)
    planner.clear()
    assert not planner.active
    assert planner._pending_plan is None


@pytest.mark.parametrize("side", ("left", "right"))
def test_corner_progress_uses_the_selected_arm_measurement(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    planner = fixture.planner
    inactive = "right" if side == "left" else "left"
    q = fixture.rig.measured_qpos_by_side()[side]
    samples = np.array([q + i * 0.2 for i in range(4)])
    poses = tuple((float(i), 0., 0., 0., 0., 0., 1.) for i in range(4))
    planner._active = SimpleNamespace(
        stages=(),
        trajectory=SimpleNamespace(q_rad=samples, timestamps=np.array([0., 1., 2., 3.]),
                                   waypoint_timestamps=np.array([0., 1., 3.])),
        reference_poses=poses,
    )
    planner._reference_index, planner._last_cycle = 1, 0
    fixture.rig.scene.data.qpos[_ADDRESSES[inactive]] = samples[1]
    assert planner.reference(1) == poses[1]
    fixture.rig.scene.data.qpos[_ADDRESSES[side]] = samples[1]
    fixture.rig.scene.data.qpos[_ADDRESSES[inactive]] = -99.
    assert planner.reference(2) == poses[2]


@pytest.mark.parametrize("side", ("left", "right"))
def test_both_start_states_are_checked_before_activation(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    plan = _plan(fixture)
    inactive = "right" if side == "left" else "left"
    fixture.rig.scene.data.qpos[_ADDRESSES[inactive][0]] += 0.0011
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_start_state_changed"):
        fixture.planner.activate(plan)
    assert not fixture.planner.active


@pytest.mark.parametrize("side", ("left", "right"))
def test_cancellation_does_not_promote_a_pending_plan(monkeypatch, side):
    fixture = _fixture(monkeypatch, motion_side=side)
    before = fixture.rig.scene.data.qpos.copy()
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_planning_cancelled"):
        _plan(fixture, cancel_requested=lambda: True)
    cancelled = False
    search = planning.plan_collision_aware_joint_path

    def cancel_after_search(**kwargs):
        nonlocal cancelled
        result = search(**kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(planning, "plan_collision_aware_joint_path", cancel_after_search)
    with pytest.raises(planning.MujocoTcpPlanRejected, match="safe_tcp_planning_cancelled"):
        _plan(fixture, cancel_requested=lambda: cancelled)
    assert fixture.planner._pending_plan is None
    assert not fixture.planner.active
    np.testing.assert_array_equal(fixture.rig.scene.data.qpos, before)


def _target_inputs(side, scene):
    component = side + "_arm"
    config = targets.MujocoSimHILTargetConfig(
        task_config_path=Path("fixture:never-opened.json"),
        realtime_worker=False,
        allow_scripted_task_reset=False,
        exposed_pose_sides=(side,),
        execution_observation_plan=ExecutionObservationPlan(
            plan_id="fixture:single-pose", reference_mode="full_frames", measurement_mode="full_frames",
            component_ids=(component,), required_representation_by_component={component: "cartesian_pose"},
        ),
    )
    prepared = targets.MujocoSimHILPreparedTarget(
        config=config, scene=scene,
        task_config=SimpleNamespace(left_tool_profile_path="fixture:left-tool.json",
                                    right_tool_profile_path="fixture:right-tool.json"),
        position_profile=POSITION_15_60_120_PROFILE, compose_target_hz=15., timing=object(),
        arm_profile=object(), gripper_profile=object(), target_frame_profile=object(),
        tool_frame_profile=object(), task_initialization_info={},
    )
    binding = SimpleNamespace(
        action_schema=SimpleNamespace(component_ids=(component,)),
        materializer=SimpleNamespace(profile=SimpleNamespace(arm_side_by_component={component: side},
                                                            gripper_name_by_component={})),
        support_only_output_effectors=(("right" if side == "left" else "left") + "_arm",),
    )
    return config, prepared, binding


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("invalid", ("both", "opposite", "wrong_map", "hand"))
def test_target_builder_retains_exact_singleton_component_admission(monkeypatch, side, invalid):
    config, prepared, binding = _target_inputs(side, SimpleNamespace())
    if invalid == "both":
        binding.action_schema.component_ids = ("left_arm", "right_arm")
    elif invalid == "opposite":
        binding.action_schema.component_ids = (("right" if side == "left" else "left") + "_arm",)
    elif invalid == "wrong_map":
        binding.materializer.profile.arm_side_by_component[side + "_arm"] = (
            "right" if side == "left" else "left")
    else:
        binding.materializer.profile.gripper_name_by_component = {"hand": "fixture:gripper"}
    with pytest.raises(ValueError, match="requires one " + side + "_arm pose"):
        targets.build_mujoco_agent_skill_motion_target(
            config=config, prepared_target=prepared, component_execution_binding=binding, motion_side=side,
        )


@pytest.mark.parametrize("side_arg", (None, "left", "right"))
def test_target_builder_forwards_side_without_replacing_the_dual_rig(monkeypatch, side_arg):
    from robot_motion import rm65
    from sim.mujoco import adapters, ik
    from sim.mujoco.deploy import pose_resolution, runtime

    side = side_arg or "right"
    fixture = _fixture(monkeypatch, motion_side=side)
    scene = fixture.rig.scene
    scene.scene_spec = SimpleNamespace(collision=SimpleNamespace(safety_minimum_clearance_m=0.002))
    scene.get_arm_qpos = lambda arm: scene.data.qpos[_ADDRESSES[arm]].copy()
    fixture.rig.accepted_qcmd_by_side = fixture.rig.measured_qpos_by_side
    config, prepared, binding = _target_inputs(side, scene)
    rig_configs = []

    def rig_for(captured_scene, rig_config):
        assert captured_scene is scene
        rig_configs.append(rig_config)
        return fixture.rig

    monkeypatch.setattr(adapters, "MujocoPositionExecutionConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(adapters, "MujocoPositionExecutionRig", rig_for)
    monkeypatch.setattr(rm65, "load_ik_tuning_config", lambda: None)
    monkeypatch.setattr(rm65, "with_rm65_position_ik_objective", lambda *_a, **_kw: None)
    monkeypatch.setattr(rm65, "DualArmCollisionAwareRetargeterConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(rm65, "DualArmCollisionAwareRetargeter", lambda _config: SimpleNamespace(profile=None))
    monkeypatch.setattr(rm65, "load_world_from_arm_base_transforms", lambda _profile: {})
    monkeypatch.setattr(rm65, "load_rm65_tool_profile", lambda _path, *, side: {
        "tcp": {"pose_flange_m_rad": [*_OFFSETS[side], 0., 0., 0.]},
    })
    monkeypatch.setattr(ik, "RM65RetargeterIK", lambda *_a, **_kw: object())
    monkeypatch.setattr(pose_resolution, "MujocoCoupledPoseToQposResolver",
                        lambda *_a, **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(runtime, "MujocoDeployConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(runtime, "MujocoDeployRuntime",
                        lambda rig, config, **kw: SimpleNamespace(rig=rig, config=config, **kw))
    monkeypatch.setattr(targets, "build_mujoco_sim_hil_pose_observation_projector",
                        lambda *_args: fixture.projector)

    target = targets.build_mujoco_agent_skill_motion_target(
        config=config, prepared_target=prepared, component_execution_binding=binding,
        enable_safe_tcp_pose=True, **({} if side_arg is None else {"motion_side": side_arg}),
    )
    assert target.motion_side == target.planned_tcp_motion.motion_side == side
    assert target.component_id == target.planned_tcp_motion.component_id == side + "_arm"
    assert target.planned_tcp_motion.rig is target.runtime.rig is fixture.rig
    assert rig_configs[0].active_sides == ("left", "right")
    assert target.runtime.pose_resolver.support_only_arm_sides == (
        "right" if side == "left" else "left",
    )
    assert target.runtime.config.enable_component_execution_handoff is True
    assert target.runtime.config.enable_primary_policy_authority is False
    assert target.runtime.config.enable_composer_authority is False
    assert target.runtime.config.enable_replay_authority is False
