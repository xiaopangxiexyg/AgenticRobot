"""Observed-scene start clearance over the offline planner fixture; never builds MuJoCo.

The fake scene/IK boundaries come from the motion-side planner tests. The
collision answer is a real, small RM65ObstacleCollisionOracle, so admission,
inherited thresholds and dense egress distances use the reviewed RealMan code.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.observed_obstacles import ObservedObstacleCloud
from robot_motion.observed_planning import ObservedJointPlanningInput
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.rm65.obstacle_collision import (
    RM65InheritedStartClearanceOracle,
    RM65ObstacleCollisionOracle,
)
from robot_motion.scene_obstacles import AxisAlignedBoxObstacle, ObstacleSceneSnapshot
from sim.mujoco.deploy import planned_tcp_motion as planning
from test_mujoco_agent_motion_side_planner import _fixture, _observed_input

pin = pytest.importorskip("pinocchio")
hppfcl = pytest.importorskip("hppfcl")

pytestmark = [pytest.mark.unit, pytest.mark.contract]

STRICT = planning.OBSERVED_START_CLEARANCE_STRICT
INHERITED = planning.OBSERVED_START_CLEARANCE_INHERITED
_CLEARANCE_M = .02
_HAND_HALF_M = .05
# The fixture's right-arm joints 0..2 place this oracle's hand box.
_RIGHT_HAND = np.array([-.2, -.1, .4])


def _oracle(*, fixed_gap_m=None, observed_gap_m=None, observed_inside=False):
    """Two six-joint arms; only the right hand box approaches near obstacles."""
    model, collision = pin.Model(), pin.GeometryModel()
    indices = {}
    for side, base_y in (("left", 5.), ("right", 0.)):
        parent = 0
        for axis, joint in enumerate((pin.JointModelPX, pin.JointModelPY, pin.JointModelPZ,
                                      pin.JointModelRX, pin.JointModelRY, pin.JointModelRZ)):
            offset = np.array((0., base_y, 0.)) if axis == 0 else np.zeros(3)
            parent = model.addJoint(parent, joint(), pin.SE3(np.eye(3), offset), f"{side}_joint{axis}")
            model.appendBodyToJoint(parent, pin.Inertia.Random(), pin.SE3.Identity())
        collision.addGeometryObject(pin.GeometryObject(side + "_hand", parent, 0,
            pin.SE3.Identity(), hppfcl.Box(*(2 * _HAND_HALF_M,) * 3)))
        indices[side] = tuple(range(model.nq - 6, model.nq))
    obstacles = [AxisAlignedBoxObstacle(obstacle_id="far_floor", center_xyz_m=(0., 0., -5.),
                                        size_xyz_m=(1., 1., 1.))]
    if fixed_gap_m is not None:
        obstacles.append(AxisAlignedBoxObstacle(obstacle_id="near_box",
            center_xyz_m=tuple(_RIGHT_HAND - (_HAND_HALF_M + fixed_gap_m + .1, 0., 0.)),
            size_xyz_m=(.2, .2, .2)))
    template = RM65ObstacleCollisionOracle(model=model, base_collision_model=collision,
        reference_full_q=pin.neutral(model), qpos_indices_by_component=indices,
        scene=ObstacleSceneSnapshot(reference_frame_id="world", reference_frame_spec_hash="world-v1",
            revision=1, minimum_clearance_m=_CLEARANCE_M, obstacles=tuple(obstacles)))
    points = [(10., 10., 10.)]
    if observed_gap_m is not None:
        points.append(tuple(_RIGHT_HAND - (_HAND_HALF_M + observed_gap_m, 0., 0.)))
    if observed_inside:
        points.append(tuple(_RIGHT_HAND))
    # Same request-owned entry as MujocoObservedSafeMotionProvider.capture.
    return template.for_observed_cloud(ObservedObstacleCloud(reference_frame_id="world",
        points_m=np.asarray(points), evidence={"source": "fixture_rgbd"}))


def _values(fixture, oracle):
    base = _observed_input(fixture)  # Binds the fake scene's arm joint layout.
    return {"oracle": oracle, "qpos_by_component": base.qpos_by_component, "acquired_at": 10.,
            "maximum_age_sec": 30., "maximum_start_drift_rad": .01, "clock": base.clock}


def _source(fixture, oracle, start_clearance=None):
    if start_clearance is None:
        return ObservedJointPlanningInput(**_values(fixture, oracle))
    return planning.MujocoObservedJointPlanningInput(**_values(fixture, oracle),
                                                     start_clearance=start_clearance)


def _plan_x(fixture, dx, **kwargs):
    pair = fixture.rig.measured_qpos_by_side()
    pose = fixture.projector.project(model_qpos=fixture.rig.scene.data.qpos,
        qpos_by_source=pair, component_ids=("right_arm",))["right_arm"]
    target = tuple(np.r_[np.asarray(pose[:3]) + [dx, 0., 0.], pose[3:]])
    return fixture.planner.plan(target, translation_tolerance_m=.003, rotation_tolerance_rad=.03,
        cancel_requested=kwargs.pop("cancel_requested", lambda: False), **kwargs)


def _start(fixture):
    return {side: np.asarray(q, dtype=float) for side, q in fixture.rig.measured_qpos_by_side().items()}


_NEAR = {"fixed": {"fixed_gap_m": .01}, "observed": {"observed_gap_m": .012}}


@pytest.mark.parametrize("near", sorted(_NEAR))
@pytest.mark.parametrize("start_clearance", (None, STRICT))
def test_strict_observed_start_inside_margin_keeps_the_original_rejection(monkeypatch, near, start_clearance):
    fixture = _fixture(monkeypatch)
    oracle = _oracle(**_NEAR[near])
    baseline = oracle.check(_start(fixture))
    assert baseline.collision_check_available and not baseline.safe
    assert 0. < baseline.minimum_distance_m < _CLEARANCE_M
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, oracle, start_clearance))
    assert error.value.reason_code == "safe_tcp_start_blocked"
    assert error.value.details == baseline.to_dict()
    assert fixture.planner._pending_plan is None and not fixture.calls.oracles


@pytest.mark.parametrize("near", sorted(_NEAR))
def test_inherited_positive_start_plans_dense_proven_egress(monkeypatch, near):
    fixture = _fixture(monkeypatch)
    oracle = _oracle(**_NEAR[near])
    start = _start(fixture)
    samples = []
    name = "inherited_active_distances" if near == "fixed" else "observed_active_distances"
    original = getattr(RM65InheritedStartClearanceOracle, name)

    def record(self, qpos):
        samples.append(float(qpos["right"][0] - start["right"][0]))
        return original(self, qpos)

    monkeypatch.setattr(RM65InheritedStartClearanceOracle, name, record)
    plan = _plan_x(fixture, .02, observed_planning_input=_source(fixture, oracle, INHERITED))
    fixed = near == "fixed"
    assert dict(plan.path.metadata["inherited_start_clearance"]) == {
        "static_pair_count": 0, "active_pair_count": int(fixed), "active_pairs_exited": True,
        "observed_static_box_count": 0, "observed_active_box_count": int(not fixed),
        "observed_active_boxes_exited": True}
    final = {**start, "right": plan.trajectory.q_rad[-1]}
    np.testing.assert_allclose(final["right"] - start["right"], [.02, 0., 0., 0., 0., 0.], atol=1e-12)
    np.testing.assert_array_equal(plan.path.target_qpos_by_component["left"], start["left"])
    assert oracle.check(final).safe and oracle.check(final).minimum_distance_m > _CLEARANCE_M
    # Every ordered dense sample was proven; the reviewed template stays strict.
    assert len(samples) > len(plan.trajectory.q_rad)
    assert samples == sorted(samples) and samples[0] == 0. and samples[-1] == pytest.approx(.02)
    assert not oracle.check(start).safe
    strict_scene = canonical_content_hash({"observed_scene": oracle.scene_hash, "constraint": None},
        domain="dexteleop.mujoco.observed_planning_query.v1")
    assert plan.path.scene_hash != strict_scene
    assert plan.path.metadata["observed_scene_hash"] == oracle.scene_hash
    assert fixture.planner._pending_plan is plan and not fixture.calls.oracles


@pytest.mark.parametrize("near", sorted(_NEAR))
def test_inherited_start_rejects_a_path_that_never_leaves_the_margin(monkeypatch, near):
    fixture = _fixture(monkeypatch)
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .005, observed_planning_input=_source(fixture, _oracle(**_NEAR[near]), INHERITED))
    assert error.value.reason_code == "safe_tcp_inherited_start_clearance_not_exited"
    assert error.value.details["reason"] == ("inherited_start_clearance_not_exited" if near == "fixed"
                                             else "inherited_observed_clearance_not_exited")
    assert fixture.planner._pending_plan is None


@pytest.mark.parametrize("near", sorted(_NEAR))
def test_inherited_start_rejects_a_regressing_dense_path(monkeypatch, near):
    fixture = _fixture(monkeypatch)
    start = _start(fixture)
    name = "inherited_active_distances" if near == "fixed" else "observed_active_distances"
    original = getattr(RM65InheritedStartClearanceOracle, name)

    def dip(self, qpos):
        # Inject a retreat only into the egress proof. The ordinary inherited
        # check still accepts the path, so this proves the dense proof rejects.
        values = original(self, qpos)
        moved = float(qpos["right"][0] - start["right"][0])
        return tuple(value - .012 for value in values) if moved > .01 else values

    monkeypatch.setattr(RM65InheritedStartClearanceOracle, name, dip)
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, _oracle(**_NEAR[near]), INHERITED))
    assert error.value.reason_code == "safe_tcp_inherited_start_clearance_regressed"
    details = error.value.details
    assert details["reason"] == "inherited_start_clearance_regressed"
    assert details["distance_m"] < details["previous_distance_m"]
    assert {"reference_index", "edge_alpha", "inherited_pair_index"} <= set(details)
    assert details.get("source") == (None if near == "fixed" else "observed_rgbd")
    assert fixture.planner._pending_plan is None


@pytest.mark.parametrize("contact", ("fixed", "observed"))
@pytest.mark.parametrize("start_clearance", (STRICT, INHERITED))
def test_contact_or_penetration_start_is_rejected_in_both_modes(monkeypatch, contact, start_clearance):
    fixture = _fixture(monkeypatch)
    oracle = _oracle(**({"fixed_gap_m": -.005} if contact == "fixed" else {"observed_inside": True}))
    baseline = oracle.check(_start(fixture))
    assert not baseline.safe and baseline.minimum_distance_m <= 0.
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, oracle, start_clearance))
    assert error.value.reason_code == "safe_tcp_start_blocked"
    expected = baseline.to_dict()
    if start_clearance == INHERITED:
        expected.update(planning_phase="inherited_start_clearance_admission",
                        start_clearance=INHERITED, inherited_start_clearance_admitted=False)
    assert error.value.details == expected


def test_inherited_start_reports_masked_observed_overlap_as_blocked(monkeypatch):
    fixture = _fixture(monkeypatch)
    # The fixed near pair fails first and hides the observed overlap from
    # the ordinary check; admission must still find and reject it.
    oracle = _oracle(fixed_gap_m=.01, observed_inside=True)
    baseline = oracle.check(_start(fixture))
    assert baseline.reason == "collision_clearance_violated" and baseline.minimum_distance_m > 0.
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, oracle, INHERITED))
    assert error.value.reason_code == "safe_tcp_start_blocked"
    details = error.value.details
    assert details["planning_phase"] == "inherited_start_clearance_admission"
    assert details["inherited_start_clearance_admitted"] is False
    assert details["source"] == "observed_rgbd" and details["distance_m"] <= 0.
    assert details["geometry_name"] == "right_hand"


@pytest.mark.parametrize("blocked", ("start", "target"))
def test_envelope_constraint_still_applies_on_the_inherited_oracle(monkeypatch, blocked):
    fixture = _fixture(monkeypatch)
    start = _start(fixture)
    limit = start["right"][0] + (-.001 if blocked == "start" else .01)

    def envelope(_model_qpos, pair):
        inside = pair["right"][0] <= limit
        return JointConfigurationCheck(safe=inside, reason=None if inside else "tcp_envelope_blocked")

    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, _oracle(fixed_gap_m=.01), INHERITED),
                configuration_constraint=envelope, constraint_identity="fixture:envelope")
    if blocked == "start":
        assert error.value.reason_code == "safe_tcp_start_blocked"
        assert error.value.details["reason"] == "tcp_envelope_blocked"
        assert error.value.details["inherited_start_clearance_admitted"] is True
        assert error.value.details["baseline_check"]["reason"] == "collision_clearance_violated"
    else:
        assert error.value.reason_code == "safe_tcp_target_envelope_blocked"


def test_unreviewed_observed_oracle_type_is_not_eligible(monkeypatch):
    fixture = _fixture(monkeypatch)
    near = JointConfigurationCheck(safe=False, reason="collision_clearance_violated", minimum_distance_m=.01)
    fake = _observed_input(fixture, check=lambda _pair: near).oracle
    with pytest.raises(planning.MujocoTcpPlanRejected) as error:
        _plan_x(fixture, .02, observed_planning_input=_source(fixture, fake, INHERITED))
    assert error.value.reason_code == "safe_tcp_start_blocked"
    assert error.value.details["inherited_start_clearance_admitted"] is False
    assert error.value.details["reason"] == "collision_clearance_violated"


def test_clear_start_plan_is_identical_in_every_mode(monkeypatch):
    plans = []
    for start_clearance in (None, STRICT, INHERITED):
        fixture = _fixture(monkeypatch)
        oracle = _oracle()
        assert oracle.check(_start(fixture)).safe
        plans.append((_plan_x(fixture, .02, observed_planning_input=_source(fixture, oracle, start_clearance)),
                      oracle))
    for plan, oracle in plans:
        assert "inherited_start_clearance" not in plan.path.metadata
        assert plan.path.scene_hash == canonical_content_hash(
            {"observed_scene": oracle.scene_hash, "constraint": None},
            domain="dexteleop.mujoco.observed_planning_query.v1")
    assert len({plan.path.semantic_content_hash for plan, _ in plans}) == 1
    assert len({plan.trajectory_hash for plan, _ in plans}) == 1


@pytest.mark.parametrize("value", ("STRICT", "", "inherited-positive-v2", None, True))
def test_observed_planning_input_start_clearance_is_an_exact_mode(monkeypatch, value):
    fixture = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="start_clearance"):
        planning.MujocoObservedJointPlanningInput(**_values(fixture, _oracle()), start_clearance=value)
    default = planning.MujocoObservedJointPlanningInput(**_values(fixture, _oracle()))
    assert default.start_clearance == STRICT and isinstance(default, ObservedJointPlanningInput)
