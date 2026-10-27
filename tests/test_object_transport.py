from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.object_transport import (
    CalibratedBoxSupportObservation,
    GraspedObjectObservation,
    SupportSurfaceObservation,
    object_translation_feedback_tool_goal,
    plan_release_approach,
    plan_release_approach_candidates,
    plan_release_approach_for_relation,
    tool_goal_from_object_goal,
    verify_rigid_follow,
)


def test_translation_feedback_corrects_object_error_without_changing_nominal_rotation():
    tool = pose((0, 0, .1), (170, 0, 90))
    obj, goal = pose((0, 0, .02)), pose((.002, -.001, .02))
    before = [matrix.copy() for matrix in (tool, obj, goal)]
    corrected = object_translation_feedback_tool_goal(tool, obj, goal, tool)
    np.testing.assert_allclose(corrected[:3, 3], [.002, -.001, .1])
    np.testing.assert_array_equal(corrected[:3, :3], tool[:3, :3])
    for actual, original in zip((tool, obj, goal), before, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert not corrected.flags.writeable


def test_translation_feedback_bounds_each_step_and_total_nominal_offset():
    nominal = pose((0, 0, .1))
    measured = pose((-.002, 0, .1))
    for _ in range(12):
        goal = object_translation_feedback_tool_goal(measured, pose(), pose((1, 0, 0)), nominal)
        assert np.linalg.norm(goal[:3, 3]-nominal[:3, 3]) <= .003+1e-12
        assert np.linalg.norm(goal[:3, 3]-measured[:3, 3]) <= .003+1e-12
        measured = goal
    assert measured[0, 3] == pytest.approx(.003)
    with pytest.raises(ValueError, match="nominal correction region"):
        object_translation_feedback_tool_goal(pose((.004, 0, .1)), pose(), pose(), nominal)


@pytest.mark.parametrize("bound", [True, 0, -.001, .004, float('nan'), float('inf')])
def test_translation_feedback_rejects_invalid_bounds(bound):
    with pytest.raises(ValueError):
        object_translation_feedback_tool_goal(pose(), pose(), pose(), pose(), maximum_correction_m=bound)


def pose(xyz=(0, 0, 0), angles=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


def observation(**overrides):
    fields = dict(object_id="cube", side="right", reference_frame_id="world", scene_revision=1,
        measured_at=10., reference_from_object=pose((.3, -.2, .4), (10, 20, 30)),
        reference_from_tool=pose((.27, -.25, .48), (170, 0, 90)),
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")}, box_size_m=(.04, .05, .06),
        object_from_box=pose(), geometry_complete=True, pose_evidence_ref="pose:post-grasp",
        grasp_evidence_ref="contact-and-follow:trial-lift")
    fields.update(overrides)
    return GraspedObjectObservation(**fields)


def test_tool_goal_preserves_rotated_off_center_object_relation():
    held = observation()
    goal = pose((.4, .1, .6), (30, -20, 80))
    np.testing.assert_allclose(held.tool_goal(goal) @ held.tool_from_object, goal, atol=1e-12)
    assert np.linalg.norm(held.tool_goal(goal)[:3, 3]-goal[:3, 3]) > .05
    assert not held.reference_from_object.flags.writeable
    assert not held.qpos_by_side["right"].flags.writeable


@pytest.mark.parametrize("change", [dict(geometry_complete=False), dict(grasp_evidence_ref=""),
    dict(reference_from_object=np.zeros((4, 4))), dict(object_from_box=np.diag([1, 1, -1, 1])),
    dict(box_size_m=(0, .1, .1)), dict(measured_at=float("nan"))])
def test_rejects_unknown_geometry_unverified_grasp_or_invalid_frames(change):
    with pytest.raises(ValueError):
        observation(**change)


@pytest.mark.parametrize("change", [dict(now=10.3), dict(now=9.99), dict(scene_revision=2),
    dict(side="left"), dict(object_id="other"), dict(reference_frame_id="camera")])
def test_freshness_identity_and_frame_must_match(change):
    fields = dict(now=10.1, scene_revision=1, reference_frame_id="world", object_id="cube", side="right")
    fields.update(change)
    with pytest.raises(ValueError):
        observation().require_current(**fields)


def test_rigid_follow_requires_actual_motion_and_detects_slip():
    before = observation()
    delta = pose((.01, 0, .04), (0, 3, 0))
    after = replace(before, measured_at=10.5, reference_from_tool=delta @ before.reference_from_tool,
                    reference_from_object=delta @ before.reference_from_object)
    assert verify_rigid_follow(before, after)["relative_translation_change_m"] < 1e-10
    with pytest.raises(ValueError, match="Static"):
        verify_rigid_follow(before, replace(before, measured_at=10.5))
    with pytest.raises(ValueError, match="slipped"):
        verify_rigid_follow(before, replace(after, reference_from_object=pose((.02, 0, 0)) @ after.reference_from_object))


def test_fcl_carried_box_orientation_and_explicit_other_arm_pairs():
    pin = pytest.importorskip("pinocchio")
    fcl = pytest.importorskip("hppfcl")
    from robot_motion.rm65.obstacle_collision import AttachedBoxCollisionProxy, RM65ObstacleCollisionOracle
    from robot_motion.scene_obstacles import ObstacleSceneSnapshot
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelPX(), pin.SE3.Identity(), "x")
    frame = model.addFrame(pin.Frame("tool", joint, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    geometries = pin.GeometryModel()
    geometries.addGeometryObject(pin.GeometryObject("other_arm_proxy", 0, 0,
        pin.SE3(np.eye(3), np.array([0., .2, 0.])), fcl.Sphere(.025)))
    def build(rotation, targets):
        return RM65ObstacleCollisionOracle(model=model, base_collision_model=geometries,
            reference_full_q=np.zeros(1), qpos_indices_by_component={"right": (0,)},
            scene=ObstacleSceneSnapshot("world", "frame", 1),
            attached_box_proxies=(AttachedBoxCollisionProxy("held", "tool", (0, 0, 0), (.4, .04, .04),
                tuple(rotation.reshape(-1)), targets),))
    unrotated = build(np.eye(3), ("other_arm_proxy",))
    assert unrotated.check({"right": np.zeros(1)}).safe
    rotated = build(Rotation.from_euler("z", 90, degrees=True).as_matrix(), ("other_arm_proxy",))
    assert not rotated.check({"right": np.zeros(1)}).safe
    assert rotated.check({"right": np.array([.2])}).safe
    assert len(geometries.geometryObjects) == 1 and len(geometries.collisionPairs) == 0
    with pytest.raises(ValueError, match="Invalid attached"):
        build(np.eye(3), ("missing",))


@pytest.mark.parametrize("bound", [1e-200, 1e-300])
def test_translation_feedback_preserves_extremely_small_positive_bounds(bound):
    result = object_translation_feedback_tool_goal(
        pose(), pose(), pose((1e-170, 0, 0)), pose(), maximum_correction_m=bound,
    )
    assert 0 < math.hypot(*result[:3, 3]) <= bound


@pytest.mark.parametrize("bound", [1e-200, 1e-300])
def test_translation_feedback_rejects_measured_tool_outside_a_tiny_bound(bound):
    with pytest.raises(ValueError, match="nominal correction region"):
        object_translation_feedback_tool_goal(
            pose((1e-170, 0, 0)), pose(), pose(), pose(), maximum_correction_m=bound,
        )


@pytest.mark.parametrize("use_observation", [False, True])
def test_tool_goal_rejects_nonfinite_composition_of_finite_input_poses(use_observation):
    goal, relation = pose((1e308, 0, 0)), pose((-1e308, 0, 0))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite rigid"):
            if use_observation:
                observation(reference_from_tool=pose(), reference_from_object=relation).tool_goal(goal)
            else:
                tool_goal_from_object_goal(goal, relation)


def test_tool_goal_helper_preserves_relation_and_returns_an_immutable_pose():
    goal, relation = pose((.4, -.2, .1), (15, -20, 45)), pose((.01, .02, .03), (0, 15, 0))
    before = [p.copy() for p in (goal, relation)]
    result = tool_goal_from_object_goal(goal, relation)
    np.testing.assert_allclose(result @ relation, goal, atol=1e-12)
    assert not result.flags.writeable
    for actual, original in zip((goal, relation), before, strict=True):
        np.testing.assert_array_equal(actual, original)


def support(**overrides):
    values = dict(support_id="table-region", reference_frame_id="world", scene_revision=1,
        measured_at=10.01, top_z_m=0., normal_reference_xyz=(0., 0., 1.),
        region_min_xy_m=(-.4, -.1),
        region_max_xy_m=(-.2, .1), evidence_ref="rgbd:measured-support-region",
        geometry_complete=True)
    values.update(overrides)
    return SupportSurfaceObservation(**values)


def known_box_support(**overrides):
    values = dict(
        support_id="table", reference_frame_id="world", scene_revision=1,
        measured_at=10., reference_from_box=pose((-.3, 0., -.02)),
        box_size_m=(.07, .07, .04),
        geometry_evidence_ref="catalog:known-table-box",
        pose_evidence_ref="registered-rgbd:table-pose",
        geometry_complete=True,
    )
    values.update(overrides)
    return CalibratedBoxSupportObservation(**values)


def bind_known_box(box, **overrides):
    values = dict(
        plane_measured_at=10.01, measured_upper_top_z_m=.002,
        measured_normal_reference_xyz=(0., 0., 1.),
        measured_region_center_xy_m=(-.3, 0.),
        depth_evidence_ref="rgbd:top-plane", lateral_uncertainty_m=.005,
        vertical_uncertainty_m=.002,
    )
    values.update(overrides)
    return box.bind_measured_top(**values)


def test_known_box_and_measured_plane_produce_inscribed_support_not_outer_aabb():
    box = known_box_support()
    observed = bind_known_box(box)
    assert observed.geometry_complete is True
    assert observed.support_id == "table"
    assert observed.top_z_m == pytest.approx(.004)
    np.testing.assert_allclose(observed.region_min_xy_m, (-.33, -.03))
    np.testing.assert_allclose(observed.region_max_xy_m, (-.27, .03))
    held = observation(object_id="cube", box_size_m=(.035, .035, .035),
                       reference_from_object=pose((-.3, 0., .0566)),
                       reference_from_tool=pose((-.31, .01, .15)))
    proposed = plan_release_approach(
        held, observed, now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    assert proposed.object_goal[2, 3] == pytest.approx(.0275)
    assert proposed.support_id == observed.support_id
    assert observed.evidence_ref.startswith("support-box-fusion:")
    assert not box.reference_from_box.flags.writeable


def test_rotated_known_box_uses_inner_square_not_unsafe_outer_bounds():
    box = known_box_support(reference_from_box=pose((-.3, 0., -.02), (0., 0., 45.)))
    observed = bind_known_box(box)
    radius = .035 / math.sqrt(2.) - .005
    assert observed.region_max_xy_m[0] == pytest.approx(-.3 + radius)
    assert observed.region_max_xy_m[1] == pytest.approx(radius)
    assert observed.region_max_xy_m[0] < -.3 + .035 * math.sqrt(2.)
    with pytest.raises(ValueError, match="footprint"):
        plan_release_approach(
            observation(object_id="cube", box_size_m=(.035, .035, .035)),
            observed, now=10.02, clearance_m=.006,
            minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
            lateral_margin_m=.005)


def test_fused_tilted_plane_dominates_measured_upper_plane_at_all_region_corners():
    box = known_box_support(
        reference_from_box=pose((0., 0., -.49), (.9, 0., 0.)),
        box_size_m=(1., 1., 1.),
    )
    surface = bind_known_box(
        box, measured_upper_top_z_m=.0149,
        measured_region_center_xy_m=(0., 0.),
        lateral_uncertainty_m=0., vertical_uncertainty_m=0.,
    )
    for x in surface.region_min_xy_m[0], surface.region_max_xy_m[0]:
        for y in surface.region_min_xy_m[1], surface.region_max_xy_m[1]:
            center = (np.asarray(surface.region_min_xy_m)
                      + surface.region_max_xy_m) / 2.
            normal = np.asarray(surface.normal_reference_xyz)
            fused_z = surface.top_z_m - (
                normal[:2] @ (np.array([x, y]) - center)
            ) / normal[2]
            assert fused_z >= .0149 - 1e-12


def test_fused_support_checks_each_source_time_against_held_and_now():
    box = known_box_support(measured_at=10.)
    fused = bind_known_box(box, plane_measured_at=10.04)
    assert fused.measured_at == 10.04
    assert fused.source_measurement_times_sec == (10., 10.04)
    held = observation(measured_at=10.04, box_size_m=(.035, .035, .035),
                       reference_from_object=pose((-.3, 0., .08)))
    with pytest.raises(ValueError, match="fresh correlated"):
        plan_release_approach(
            held, fused, now=10.23, clearance_m=.006,
            minimum_planning_clearance_m=.004, maximum_clearance_m=.05)
    with pytest.raises(ValueError, match="fresh correlated"):
        plan_release_approach(
            replace(held, measured_at=10.085), fused, now=10.10,
            clearance_m=.006, minimum_planning_clearance_m=.004,
            maximum_clearance_m=.05)


def test_fused_support_rejects_dropped_constituent_timestamp():
    with pytest.raises(ValueError, match="retain both"):
        support(measured_at=10.04, source_measurement_times_sec=(10.,))


@pytest.mark.parametrize("box_change, plane_change, reason", [
    (dict(geometry_complete=False), {}, "visible or unknown"),
    (dict(box_size_m=(.07, 0., .04)), {}, "positive finite"),
    (dict(box_size_m=(.07, True, .04)), {}, "positive finite"),
    ({}, dict(plane_measured_at=10.2), "correlated cut"),
    ({}, dict(measured_upper_top_z_m=.02), "top height disagree"),
    ({}, dict(measured_normal_reference_xyz=(0., 1., 0.)), "orientation"),
    ({}, dict(measured_region_center_xy_m=(0., 0.)), "outside the bound"),
    ({}, dict(lateral_uncertainty_m=.04), "no conservative inner region"),
])
def test_known_box_fusion_rejects_unknown_geometry_or_bad_observation(
    box_change, plane_change, reason,
):
    with pytest.raises(ValueError, match=reason):
        bind_known_box(known_box_support(**box_change), **plane_change)


def test_release_approach_uses_full_held_box_and_measured_surface_not_visible_center():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((-.3, 0., .0566)),
        reference_from_tool=pose((-.31, .01, .15)))
    result = plan_release_approach(held, support(), now=10.02,
        clearance_m=.006, minimum_planning_clearance_m=.004,
        maximum_clearance_m=.05)
    np.testing.assert_allclose(result.object_goal[:3, 3], [-.3, 0., .0235])
    np.testing.assert_allclose(result.tool_goal @ held.tool_from_object, result.object_goal)
    assert result.evidence_refs == (
        held.pose_evidence_ref, held.grasp_evidence_ref, support().evidence_ref)
    assert not result.object_goal.flags.writeable and not result.tool_goal.flags.writeable
    assert held.reference_from_object[2, 3] == pytest.approx(.0566)


def test_release_approach_respects_rotated_offset_box_and_support_footprint():
    held = observation(box_size_m=(.04, .02, .06),
        reference_from_object=pose((0., 0., .2), (0., 30., 25.)),
        object_from_box=pose((.01, 0., .005), (0., 0., 10.)))
    result = plan_release_approach(held, support(region_min_xy_m=(-.1, -.1),
        region_max_xy_m=(.1, .1)), now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.002)
    half = np.asarray(held.box_size_m) / 2.
    corners = np.asarray([(x, y, z) for x in (-half[0], half[0])
                          for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    points = (result.object_goal @ held.object_from_box)[:3, :3] @ corners.T
    points += (result.object_goal @ held.object_from_box)[:3, 3:4]
    assert points[2].min() == pytest.approx(.006)
    assert np.all(points[:2].min(axis=1) >= -.1 + .002)
    assert np.all(points[:2].max(axis=1) <= .1 - .002)


def test_release_approach_preserves_full_box_clearance_above_a_tilted_plane():
    held = observation(reference_from_object=pose((0., 0., .2)),
        reference_from_tool=pose((0., 0., .3)), box_size_m=(.2, .05, .04))
    normal = (-.015707317311820675, 0., .9998766324816606)
    surface = support(normal_reference_xyz=normal,
        region_min_xy_m=(-.4, -.1), region_max_xy_m=(.4, .1))
    result = plan_release_approach(held, surface, now=10.02,
        clearance_m=.004, minimum_planning_clearance_m=.004,
        maximum_clearance_m=.05)
    half = np.asarray(held.box_size_m) / 2.
    corners = np.asarray([(x, y, z) for x in (-half[0], half[0])
                          for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    world = (result.object_goal[:3, :3] @ corners.T).T + result.object_goal[:3, 3]
    anchor = np.array([0., 0., surface.top_z_m])
    signed = (world - anchor) @ np.asarray(normal)
    assert signed.min() == pytest.approx(.004, abs=1e-12)


def test_release_target_can_use_an_observed_offcenter_relation_goal():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((-.3, 0., .08)),
        reference_from_tool=pose((-.31, .01, .15)))
    surface = support(region_min_xy_m=(-.4, -.1),
                      region_max_xy_m=(-.2, .1))
    result = plan_release_approach(
        held, surface, now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005,
        target_box_center_xy_m=(-.35, .04))
    np.testing.assert_allclose(result.object_goal[:3, 3], (-.35, .04, .0235))
    np.testing.assert_allclose(result.tool_goal @ held.tool_from_object,
                               result.object_goal)
    with pytest.raises(ValueError, match="footprint"):
        plan_release_approach(
            held, surface, now=10.02, clearance_m=.006,
            minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
            lateral_margin_m=.005,
            target_box_center_xy_m=(-.39, .04))


def test_named_support_relation_uses_complete_rotated_footprint():
    held = observation(box_size_m=(.04, .02, .06),
        reference_from_object=pose((-.3, 0., .2), (0., 30., 25.)),
        reference_from_tool=pose((-.31, .01, .3)),
        object_from_box=pose((.01, 0., .005), (0., 0., 10.)))
    surface = support(region_min_xy_m=(-.45, -.12),
        region_max_xy_m=(-.15, .12))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    middle = plan_release_approach_for_relation(
        held, surface, relation="center", **options)
    direct = plan_release_approach(held, surface, **options)
    np.testing.assert_allclose(middle.object_goal, direct.object_goal, atol=1e-12)
    left = plan_release_approach_for_relation(
        held, surface, relation="reference_x_negative", **options)
    right = plan_release_approach_for_relation(
        held, surface, relation="reference_x_positive", **options)
    # The box center, rather than object-frame origin, follows the named
    # support relation. The unchanged lower function checks all eight corners.
    center_offset = (held.reference_from_object[:3, :3]
                     @ held.object_from_box[:3, 3])[0]
    assert left.object_goal[0, 3] + center_offset < middle.object_goal[0, 3] + center_offset
    assert middle.object_goal[0, 3] + center_offset < right.object_goal[0, 3] + center_offset
    for result in (left, right):
        assert result.release_clearance_m == .006
        assert result.evidence_refs == middle.evidence_refs
        np.testing.assert_allclose(result.tool_goal @ held.tool_from_object,
                                   result.object_goal, atol=1e-12)


def test_named_support_relation_rejects_ambiguous_or_unavailable_region():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((0., 0., .1)),
        reference_from_tool=pose((0., 0., .2)))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    with pytest.raises(ValueError, match="Unsupported support relation"):
        plan_release_approach_for_relation(
            held, support(), relation="right", **options)
    with pytest.raises(ValueError, match="no feasible support center"):
        plan_release_approach_for_relation(
            held, support(region_min_xy_m=(-.01, -.01),
                          region_max_xy_m=(.01, .01)), relation="center", **options)
    just_fits_x = support(region_min_xy_m=(-.0225 - 5e-11, -.1),
                          region_max_xy_m=(.0225 + 5e-11, .1))
    plan_release_approach_for_relation(
        held, just_fits_x, relation="center", **options)
    for relation in ("reference_x_negative", "reference_x_positive"):
        with pytest.raises(ValueError, match="no reference-.*-X placement region"):
            plan_release_approach_for_relation(
                held, just_fits_x, relation=relation, **options)
    with pytest.raises(ValueError, match="fresh correlated observations"):
        plan_release_approach_for_relation(
            held, support(measured_at=9.0), relation="center", **options)


def test_release_candidates_keep_one_skill_goal_and_complete_measured_footprints():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((-.285, -.02, .0566)),
        reference_from_tool=pose((-.27, .03, .15)))
    surface = support(support_id="large_blue_cube", top_z_m=.051,
        region_min_xy_m=(-.41, -.02), region_max_xy_m=(-.33, .06))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    original = plan_release_approach(held, surface, **options)
    simple = plan_release_approach_candidates(held, surface, **options)
    assert len(simple) == 5 and all(item.yaw_offset_rad == 0 for item in simple)
    np.testing.assert_array_equal(simple[0].proposal.object_goal, original.object_goal)
    expanded = plan_release_approach_candidates(
        held, surface, allow_quarter_turns=True, **options)
    assert len(expanded) == 20
    assert {item.yaw_offset_rad for item in expanded} == {0., -math.pi / 2., math.pi / 2., math.pi}
    assert any(item.yaw_offset_rad == -math.pi / 2. and
               np.linalg.norm(np.asarray(item.box_center_xy_m) - [-.355, .035]) < .002
               for item in expanded)
    corners = np.asarray([(x, y, z) for x in (-.0175, .0175)
                          for y in (-.0175, .0175) for z in (-.0175, .0175)])
    for item in expanded:
        proposal = item.proposal
        world = corners @ proposal.object_goal[:3, :3].T + proposal.object_goal[:3, 3]
        assert np.all(world[:, :2].min(axis=0) >= np.asarray(surface.region_min_xy_m) + .005 - 1e-12)
        assert np.all(world[:, :2].max(axis=0) <= np.asarray(surface.region_max_xy_m) - .005 + 1e-12)
        assert world[:, 2].min() - surface.top_z_m == pytest.approx(.006, abs=1e-12)
        np.testing.assert_allclose(proposal.tool_goal @ held.tool_from_object,
                                   proposal.object_goal, atol=1e-12)
        assert proposal.evidence_refs == original.evidence_refs


def test_release_candidates_reject_stale_support_and_nonboolean_orientation_policy():
    held = observation(box_size_m=(.035, .035, .035))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05)
    with pytest.raises(ValueError, match="fresh correlated observations"):
        plan_release_approach_candidates(
            held, support(measured_at=9.), **options)
    with pytest.raises(TypeError, match="explicitly boolean"):
        plan_release_approach_candidates(
            held, support(), allow_quarter_turns=1, **options)
    with pytest.raises(TypeError, match="typed held-object and support"):
        plan_release_approach_candidates(
            held, object(), **options)


def test_release_candidates_can_rotate_a_rectangular_object_onto_narrow_support():
    held = observation(box_size_m=(.06, .02, .02),
        reference_from_object=pose((0., 0., .1)),
        reference_from_tool=pose((0., 0., .2)))
    surface = support(region_min_xy_m=(-.0175, -.045),
                      region_max_xy_m=(.0175, .045))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    with pytest.raises(ValueError, match="footprint"):
        plan_release_approach(held, surface, **options)
    with pytest.raises(ValueError, match="No complete held footprint"):
        plan_release_approach_candidates(held, surface, **options)
    candidates = plan_release_approach_candidates(
        held, surface, allow_quarter_turns=True, **options)
    assert candidates and {item.yaw_offset_rad for item in candidates} == {
        -math.pi / 2., math.pi / 2.}
    assert all(item.proposal.release_clearance_m == .006 for item in candidates)


def test_release_candidates_accept_explicit_bounded_yaw_samples_without_changing_default():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((-.285, -.02, .0566)))
    surface = support(region_min_xy_m=(-.41, -.02),
                      region_max_xy_m=(-.33, .06))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005)
    original = plan_release_approach_candidates(held, surface, **options)
    angles = (0., -math.radians(80.), -math.pi / 2.)
    candidates = plan_release_approach_candidates(
        held, surface, yaw_offsets_rad=angles, **options)
    assert len(original) == 5 and len(candidates) == 15
    assert tuple(dict.fromkeys(item.yaw_offset_rad for item in candidates)) == angles
    np.testing.assert_array_equal(candidates[0].proposal.object_goal,
                                  original[0].proposal.object_goal)
    assert all(candidate.proposal.evidence_refs == original[0].proposal.evidence_refs
               for candidate in candidates)
    for invalid in ((0., 0.), (True,), (float("nan"),), (math.pi + .01,), ()):
        with pytest.raises(ValueError, match="Yaw samples|1..8 angles"):
            plan_release_approach_candidates(
                held, surface, yaw_offsets_rad=invalid, **options)
    with pytest.raises(ValueError, match="replace quarter-turn"):
        plan_release_approach_candidates(
            held, surface, yaw_offsets_rad=angles,
            allow_quarter_turns=True, **options)


def test_release_candidates_recompute_clearance_after_yaw_for_offset_box():
    held = observation(box_size_m=(.04, .02, .03),
        reference_from_object=pose((-.3, 0., .16), (0., 15., 10.)),
        object_from_box=pose((.008, -.004, .003), (0., 0., 8.)))
    normal = np.asarray((-.008, .004, math.sqrt(1. - .008**2 - .004**2)))
    surface = support(top_z_m=.08,
        normal_reference_xyz=tuple(normal),
        region_min_xy_m=(-.4, -.1), region_max_xy_m=(-.2, .1))
    candidates = plan_release_approach_candidates(
        held, surface, now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        lateral_margin_m=.005, allow_quarter_turns=True)
    assert len(candidates) == 20
    half = np.asarray(held.box_size_m) / 2.
    corners = np.asarray([(x, y, z) for x in (-half[0], half[0])
                          for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    support_center = (np.asarray(surface.region_min_xy_m)
                      + surface.region_max_xy_m) / 2.
    plane_anchor = np.array([*support_center, surface.top_z_m])
    for candidate in candidates:
        world_box = candidate.proposal.object_goal @ held.object_from_box
        np.testing.assert_allclose(world_box[:2, 3], candidate.box_center_xy_m, atol=1e-12)
        world_corners = corners @ world_box[:3, :3].T + world_box[:3, 3]
        assert np.min((world_corners - plane_anchor) @ normal) == pytest.approx(.006, abs=1e-12)
        assert np.all(world_corners[:, :2].min(axis=0) >= np.asarray(surface.region_min_xy_m) + .005 - 1e-12)
        assert np.all(world_corners[:, :2].max(axis=0) <= np.asarray(surface.region_max_xy_m) - .005 + 1e-12)


def test_offcenter_release_uses_support_plane_anchor_not_target_center():
    held = observation(box_size_m=(.035, .035, .035),
        reference_from_object=pose((-.3, 0., .08)),
        reference_from_tool=pose((-.31, .01, .15)))
    normal = (-.015707317311820675, 0., .9998766324816606)
    surface = support(normal_reference_xyz=normal)
    result = plan_release_approach(
        held, surface, now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05,
        target_box_center_xy_m=(-.35, .04))
    half = np.asarray(held.box_size_m) / 2.
    corners = np.asarray([(x, y, z) for x in (-half[0], half[0])
                          for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    world = (result.object_goal[:3, :3] @ corners.T).T + result.object_goal[:3, 3]
    anchor = np.array([-.3, 0., surface.top_z_m])
    assert np.min((world - anchor) @ np.asarray(normal)) == pytest.approx(.006)


@pytest.mark.parametrize("change", [
    dict(clearance_m=.0015), dict(clearance_m=.051),
    dict(clearance_m=float("nan")), dict(clearance_m=True),
    dict(lateral_margin_m=.09), dict(lateral_margin_m=-.001),
    dict(maximum_pair_skew_sec=-.01), dict(now=10.3),
    dict(maximum_measurement_age_sec=True),
    dict(maximum_measurement_age_sec=np.bool_(True)),
    dict(target_box_center_xy_m=(True, 0.)),
    dict(target_box_center_xy_m=(float("nan"), 0.)),
    dict(target_box_center_xy_m=0.),
    dict(support_override=dict(measured_at=9.9)),
    dict(support_override=dict(scene_revision=2)),
    dict(support_override=dict(reference_frame_id="camera")),
    dict(support_override=dict(support_id="cube")),
    dict(support_override=dict(region_min_xy_m=(-.01, -.01), region_max_xy_m=(.01, .01))),
])
def test_release_approach_rejects_missing_clearance_stale_or_unsupported_geometry(change):
    change = dict(change)
    surface = support(**change.pop("support_override", {}))
    options = dict(now=10.02, clearance_m=.006,
        minimum_planning_clearance_m=.004, maximum_clearance_m=.05)
    options.update(change)
    with pytest.raises(ValueError):
        plan_release_approach(observation(), surface, **options)


@pytest.mark.parametrize("change", [
    dict(geometry_complete=False), dict(top_z_m=float("nan")),
    dict(top_z_m=True), dict(scene_revision=True),
    dict(normal_reference_xyz=(0., 1., 0.)),
    dict(region_min_xy_m=(-.1, float("inf"))),
    dict(region_min_xy_m=(.2, 0.), region_max_xy_m=(.1, .1)),
])
def test_release_support_requires_complete_finite_measured_geometry(change):
    with pytest.raises(ValueError):
        support(**change)
