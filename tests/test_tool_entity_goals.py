"""Pure geometry contracts for tool-entity hover proposals."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry
from robot_motion.tool_entity_goals import (
    ToolEntityBox,
    entity_hover_metrics,
    footprint_support,
    generate_entity_hover_goals,
    hover_orientation,
)

pytestmark = pytest.mark.unit

DOWN = [0., 0., 0., 1., 0., 0., 0.]  # TCP +Z points to reference -Z.


def simple_entity():
    geometry = ParallelJawSolidGeometry(palm_z_m=-.02, finger_length_m=.08, finger_thickness_m=.01,
                                        pad_height_m=.02, maximum_opening_m=.06)
    return ToolEntityBox.from_parallel_jaw_geometry(geometry, source={"test": True})


def scene_points():
    """Table at z=0 and a 4 cm object top at z=0.05 around (0.1, 0.2)."""
    grid = np.stack(np.meshgrid(np.arange(-.2, .4, .01), np.arange(-.1, .5, .01)), -1).reshape(-1, 2)
    table = np.column_stack([grid, np.zeros(len(grid))])
    top = np.stack(np.meshgrid(np.arange(.08, .121, .005), np.arange(.18, .221, .005)), -1).reshape(-1, 2)
    return np.vstack([table, np.column_stack([top, np.full(len(top), .05)])])


def pose_matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def test_envelope_matches_grasp_solids_at_maximum_opening():
    entity = simple_entity()
    corners = entity.corners_tcp()
    np.testing.assert_allclose(corners.min(axis=0), [-.04, -.01, -.03])
    np.testing.assert_allclose(corners.max(axis=0), [.04, .01, .06])
    np.testing.assert_allclose(entity.tip_center_tcp_m, [0., 0., .06])
    assert entity.identity()["source"] == {"test": True}


def test_reviewed_dmtac_alignment_box_extends_beyond_tcp():
    from deployments.perception.dmtac_grasp_overlay import dmtac_graspgenx_alignment_profile
    alignment = dmtac_graspgenx_alignment_profile()
    geometry = alignment.collision_geometry(maximum_opening_m=alignment.nominal_geometry.maximum_opening_m)
    entity = ToolEntityBox.from_parallel_jaw_geometry(geometry, source={})
    corners = entity.corners_tcp()
    # Fingertips lie about 98.7 mm beyond the 140 mm tool TCP along its approach axis.
    assert corners[:, 2].max() == pytest.approx(.0987, abs=5e-4)
    np.testing.assert_allclose(entity.tip_center_tcp_m, [.0235, 0., .0985], atol=5e-4)


def test_footprint_support_uses_footprint_window_and_floor():
    entity = simple_entity()
    corners = entity.corners_world(pose_matrix([.1, .2, .3, *DOWN[3:]]))
    points = np.array([[.1, .2, .05], [.3, .2, .2], [.1, .2, .5], [.11, .2, .04]])
    assert footprint_support(points, corners, floor_z_m=0., ceiling_z_m=.3) == (.05, 2, 1)
    assert footprint_support(points[1:2], corners, floor_z_m=.01, ceiling_z_m=.3) == (.01, 0, 0)
    with pytest.raises(ValueError):
        footprint_support([[0., 0., float("nan")]], corners, floor_z_m=0., ceiling_z_m=.3)


def test_hover_places_the_box_not_the_tcp_inside_original_bounds():
    entity, points = simple_entity(), scene_points()
    candidates = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[-.02, .02], [-.02, .02]],
        clearance_bounds_m=[.03, .06], entity=entity, points_m=points, boundary_inset_m=.01)
    assert len(candidates) == 8
    first = candidates[0]
    # Measured tip is above the anchor and too high: nearest keeps XY, clamps clearance.
    np.testing.assert_allclose([*first.tip_offset_xy_m, first.clearance_m], [0., 0., .05], atol=1e-12)
    # Down-pointing box: lowest point is the fingertip face 6 cm beyond TCP.
    np.testing.assert_allclose(first.pose_xyz_quat_xyzw[:3], [.1, .2, .05 + .05 + .06], atol=1e-12)
    seen = set()
    for candidate in candidates:
        key = (*candidate.tip_offset_xy_m, candidate.clearance_m)
        assert key not in seen
        seen.add(key)
        assert all(-.01 - 1e-12 <= v <= .01 + 1e-12 for v in candidate.tip_offset_xy_m)
        assert .04 - 1e-12 <= candidate.clearance_m <= .05 + 1e-12
        metrics = entity_hover_metrics(entity=entity, world_from_tcp=pose_matrix(candidate.pose_xyz_quat_xyzw),
            anchor_xyz_m=[.1, .2, .05], points_m=points, support_window_m=.3)
        assert metrics.clearance_m == pytest.approx(candidate.clearance_m, abs=1e-9)
        np.testing.assert_allclose(metrics.tip_offset_xy_m, candidate.tip_offset_xy_m, atol=1e-9)
        assert metrics.support_height_m == pytest.approx(.05)
        rotation = Rotation.from_quat(candidate.pose_xyz_quat_xyzw[3:])
        assert (rotation.inv() * Rotation.from_quat(DOWN[3:])).magnitude() < 1e-9


def test_taller_neighbor_under_footprint_raises_the_hover():
    entity = simple_entity()
    points = np.vstack([scene_points(), [[.13, .2, .12]]])
    candidate, = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]],
        clearance_bounds_m=[.04, .04], entity=entity, points_m=points, max_candidates=1)
    assert candidate.metrics.support_height_m == pytest.approx(.12)
    assert candidate.pose_xyz_quat_xyzw[2] == pytest.approx(.12 + .04 + .06)
    # Overhead points above the window are reported, not used as support.
    high = np.vstack([scene_points(), [[.1, .2, .5]]])
    candidate, = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]],
        clearance_bounds_m=[.04, .04], entity=entity, points_m=high, max_candidates=1)
    assert candidate.metrics.overhead_point_count == 1
    assert candidate.metrics.support_height_m == pytest.approx(.05)


def test_support_margin_is_added_once():
    entity = simple_entity()
    candidate, = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]],
        clearance_bounds_m=[.04, .04], entity=entity, points_m=scene_points(), max_candidates=1,
        support_margin_m=.0035)
    assert candidate.metrics.support_height_m == pytest.approx(.0535)
    assert candidate.pose_xyz_quat_xyzw[2] == pytest.approx(.0535 + .04 + .06)


def test_tool_axis_down_policy_turns_the_approach_axis_with_least_rotation():
    horizontal = Rotation.from_euler("y", 90, degrees=True).as_quat()  # TCP +Z along reference +X
    rotation = hover_orientation([0., 0., .5, *horizontal], "tool_axis_down")
    np.testing.assert_allclose(rotation[:, 2], [0., 0., -1.], atol=1e-9)
    candidate = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *horizontal],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]],
        clearance_bounds_m=[.04, .04], entity=simple_entity(), points_m=scene_points(),
        orientation_policy="tool_axis_down", max_candidates=1)[0]
    np.testing.assert_allclose(pose_matrix(candidate.pose_xyz_quat_xyzw)[:3, 2], [0., 0., -1.], atol=1e-9)
    preserved = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *horizontal],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]],
        clearance_bounds_m=[.04, .04], entity=simple_entity(), points_m=scene_points(), max_candidates=1)[0]
    # A horizontal box hangs only half its width (4 cm) below the TCP.
    assert preserved.metrics.lowest_point_z_m == pytest.approx(preserved.pose_xyz_quat_xyzw[2] - .04)


def test_fixed_bounds_yield_one_candidate():
    candidates = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]],
        anchor_xyz_m=[.1, .2, .05], tip_offset_bounds_xy_m=[[.01, .01], [-.01, -.01]],
        clearance_bounds_m=[.05, .05], entity=simple_entity(), points_m=scene_points())
    assert len(candidates) == 1
    assert candidates[0].tip_offset_xy_m == pytest.approx((.01, -.01))


@pytest.mark.parametrize("change", [
    {"tip_offset_bounds_xy_m": [[.02, -.02], [0., 0.]]},
    {"tip_offset_bounds_xy_m": [[0., 0.]]},
    {"tip_offset_bounds_xy_m": [[-.3, 0.], [0., 0.]]},
    {"clearance_bounds_m": [-.01, .02]},
    {"clearance_bounds_m": [.02, float("nan")]},
    {"clearance_bounds_m": [.02, .5]},
    {"max_candidates": 0}, {"max_candidates": 9}, {"max_candidates": True},
    {"boundary_inset_m": -1.}, {"support_window_m": float("inf")},
    {"orientation_policy": "free"},
    {"anchor_xyz_m": [0., 0.]},
])
def test_invalid_hover_requests_are_rejected(change):
    request = dict(current_tcp_pose_xyz_quat_xyzw=[.1, .2, .5, *DOWN[3:]], anchor_xyz_m=[.1, .2, .05],
                   tip_offset_bounds_xy_m=[[-.02, .02], [-.02, .02]], clearance_bounds_m=[.03, .06],
                   entity=simple_entity(), points_m=scene_points())
    request.update(change)
    with pytest.raises(ValueError):
        generate_entity_hover_goals(**request)


def test_entity_rejects_invalid_box():
    with pytest.raises(ValueError):
        ToolEntityBox(tcp_from_box_xyz_m=(0., 0., 0.), tcp_from_box_rotation=tuple(np.eye(3).ravel()),
                      half_extents_m=(0., .01, .01), tip_center_tcp_m=(0., 0., .01))
    with pytest.raises(ValueError):
        ToolEntityBox(tcp_from_box_xyz_m=(0., 0., 0.), tcp_from_box_rotation=tuple((2 * np.eye(3)).ravel()),
                      half_extents_m=(.01, .01, .01), tip_center_tcp_m=(0., 0., .01))
    assert math.isclose(simple_entity().half_extents_m[2], .045)


# --- measured tool-object relations -------------------------------------------------

from robot_motion.tool_entity_goals import (  # noqa: E402
    TOOL_OBJECT_RELATIONS,
    ToolObjectRelation,
    fuse_tool_object_relations,
    object_surface_cluster,
    tool_object_relation,
)


def box_surface(centre, half, spacing=.003):
    """Points on the six faces of an axis-aligned box (TCP frame of simple_entity)."""
    centre, half = np.asarray(centre, float), np.asarray(half, float)
    faces = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        grids = [np.arange(-half[a], half[a] + 1e-9, spacing) for a in others]
        uu, vv = np.meshgrid(*grids)
        for sign in (-1., 1.):
            face = np.zeros((uu.size, 3))
            face[:, others[0]], face[:, others[1]] = uu.ravel(), vv.ravel()
            face[:, axis] = sign * half[axis]
            faces.append(face + centre)
    return np.unique(np.round(np.vstack(faces), 9), axis=0)


WORLD_FROM_TCP = pose_matrix([.4, -.1, .3, *Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_quat()])


def to_world(points_tcp):
    return np.asarray(points_tcp) @ WORLD_FROM_TCP[:3, :3].T + WORLD_FROM_TCP[:3, 3]


def relation_of(points_tcp, **kwargs):
    return tool_object_relation(entity=simple_entity(), world_from_tcp=WORLD_FROM_TCP,
                                object_points_m=to_world(points_tcp), **kwargs)


def test_parts_and_closing_region_partition_the_envelope():
    entity = simple_entity()
    palm, left, right = entity.parts
    np.testing.assert_allclose(palm[0], [0., 0., -.025], atol=1e-12)
    np.testing.assert_allclose(palm[1], [.04, .01, .005], atol=1e-12)
    np.testing.assert_allclose(left[0], [-.035, 0., .02], atol=1e-12)
    np.testing.assert_allclose(right[0], [.035, 0., .02], atol=1e-12)
    np.testing.assert_allclose(left[1], [.005, .01, .04], atol=1e-12)
    centre, half = entity.grasp_region
    np.testing.assert_allclose(centre, [0., 0., .02], atol=1e-12)
    np.testing.assert_allclose(half, [.03, .01, .04], atol=1e-12)
    identity = entity.identity()
    assert len(identity["parts"]) == 3
    np.testing.assert_allclose(identity["grasp_region"], [centre, half])


@pytest.mark.parametrize("centre,half,expected,distance", [
    # 2 cm cube centred between the fingers: 2 cm from either finger.
    ((0., 0., .03), (.01, .008, .01), "between_fingers", .02),
    # Nearly finger-wide object between the fingers: inside the 5 mm band.
    ((0., 0., .03), (.028, .008, .01), "between_fingers_in_contact_band", .002),
    # Flat object 1 mm outside the right finger's outer face.
    ((.051, 0., .02), (.01, .01, .02), "in_contact_band", .001),
    # Same object 3 cm away: near, not in the band.
    ((.08, 0., .02), (.01, .01, .02), "near", .03),
    ((.4, 0., .02), (.01, .01, .02), "far", .35),
])
def test_relation_classes_follow_measured_surface_geometry(centre, half, expected, distance):
    relation = relation_of(box_surface(centre, half))
    assert relation.relation == expected
    assert relation.surface_distance_m == pytest.approx(distance, abs=1e-6)
    assert relation.between_fingers is expected.startswith("between")
    assert relation.in_contact_band is ("contact_band" in expected)
    assert expected in TOOL_OBJECT_RELATIONS


def test_relation_counts_are_robust_to_a_few_stray_points():
    far = box_surface((.4, 0., .02), (.01, .01, .02))
    in_finger = np.array([[.035, 0., .02], [.035, 0., .03]])
    relation = relation_of(np.vstack([far, in_finger]))
    assert relation.contact_band_point_count == 2 and relation.relation == "far"
    assert relation.surface_distance_m == pytest.approx(.35)  # third-smallest, not the stray minimum
    in_region = np.column_stack([np.zeros(9), np.zeros(9), np.linspace(0., .04, 9)])
    sparse = relation_of(np.vstack([far, in_region]))
    assert sparse.grasp_region_point_count == 9 and sparse.relation == "near"
    strict = relation_of(np.vstack([far, in_finger, in_region]), contact_minimum_points=2, region_minimum_points=9)
    assert strict.relation == "between_fingers_in_contact_band"
    assert strict.surface_distance_m == pytest.approx(-.005)


def test_relation_requires_parts_points_and_valid_thresholds():
    entity = simple_entity()
    bare = ToolEntityBox(tcp_from_box_xyz_m=entity.tcp_from_box_xyz_m,
                         tcp_from_box_rotation=entity.tcp_from_box_rotation,
                         half_extents_m=entity.half_extents_m, tip_center_tcp_m=entity.tip_center_tcp_m)
    with pytest.raises(ValueError, match="solids"):
        tool_object_relation(entity=bare, world_from_tcp=np.eye(4), object_points_m=[[0., 0., 0.]])
    for points in ([], [[0., 0., float("nan")]]):
        with pytest.raises(ValueError):
            tool_object_relation(entity=entity, world_from_tcp=np.eye(4), object_points_m=points)
    for change in ({"contact_distance_m": .06}, {"contact_minimum_points": 0}, {"region_minimum_points": 1.5}):
        with pytest.raises(ValueError):
            tool_object_relation(entity=entity, world_from_tcp=np.eye(4), object_points_m=[[1., 0., 0.]], **change)


def test_surface_cluster_thins_to_voxels_and_drops_a_nearer_occluder():
    plane = np.stack(np.meshgrid(np.arange(0., .06, .001), np.arange(0., .06, .001)), -1).reshape(-1, 2)
    obj = np.column_stack([plane, np.full(len(plane), .30)])
    occluder = np.column_stack([plane[:200] * .2, np.full(200, .20)])  # 10 cm in front
    kept, evidence = object_surface_cluster(np.vstack([obj, obj, occluder]))
    assert evidence["input_point_count"] == 2 * len(obj) + 200
    assert evidence["component_count"] == 2
    assert evidence["kept_voxel_count"] == len(kept) == 400  # 20 x 20 voxels of 3 mm
    np.testing.assert_allclose(kept[:, 2], .30)
    again, _ = object_surface_cluster(np.vstack([obj, obj, occluder]))
    np.testing.assert_array_equal(kept, again)
    single, info = object_surface_cluster([[0., 0., 0.]])
    assert len(single) == 1 and info["component_count"] == 1
    for bad in ({"voxel_size_m": 0.}, {"link_distance_m": .001}, {"voxel_size_m": .03}):
        with pytest.raises(ValueError):
            object_surface_cluster(obj, **bad)


def test_fused_relation_is_the_union_of_partial_views():
    def make(relation, distance):
        return ToolObjectRelation(relation=relation, surface_distance_m=distance, contact_band_point_count=0,
                                  grasp_region_point_count=0, object_point_count=10)
    fused = fuse_tool_object_relations([make("near", .02), make("between_fingers", .008), make("in_contact_band", .001)])
    assert fused == {"relation": "between_fingers_in_contact_band", "surface_distance_m": .001,
                     "view_count": 3, "views_agree": False}
    assert fuse_tool_object_relations([make("far", .2), make("near", .03)])["relation"] == "near"
    assert fuse_tool_object_relations([make("far", .2)])["views_agree"] is True
    with pytest.raises(ValueError):
        fuse_tool_object_relations([])


# --- bounded tool-axis-down cone ----------------------------------------------------

from robot_motion.tool_entity_goals import MAX_TOOL_AXIS_TILT_RAD, tool_axis_tilt_rad  # noqa: E402

# Measured RealMan-like diagonal tool: approach axis about 61.5 deg from down.
DIAGONAL = Rotation.align_vectors([[-.527, -.704, -.477]], [[0., 0., 1.]])[0]


def test_tool_axis_cone_uses_the_least_rotation_and_stops_at_its_boundary():
    pose = [.1, .2, .3, *DIAGONAL.as_quat()]
    start = tool_axis_tilt_rad(DIAGONAL.as_matrix())
    assert start == pytest.approx(math.acos(.477 / np.linalg.norm([.527, .704, .477])))
    exact = hover_orientation(pose, "tool_axis_down")
    assert tool_axis_tilt_rad(exact) == pytest.approx(0., abs=1e-9)
    for tilt in (math.radians(30.), math.radians(45.)):
        cone = hover_orientation(pose, "tool_axis_down", tool_axis_max_tilt_rad=tilt)
        assert tool_axis_tilt_rad(cone) == pytest.approx(tilt, abs=1e-9)
        # Least rotation: turning by exactly (start - tilt), along the exact alignment's path.
        turn = Rotation.from_matrix(cone) * DIAGONAL.inv()
        assert turn.magnitude() == pytest.approx(start - tilt, abs=1e-9)
        rest = Rotation.from_matrix(exact) * Rotation.from_matrix(cone).inv()
        assert rest.magnitude() == pytest.approx(tilt, abs=1e-9)
    inside = Rotation.from_euler("x", 170, degrees=True)  # axis 10 deg from down
    kept = hover_orientation([0., 0., 0., *inside.as_quat()], "tool_axis_down",
                             tool_axis_max_tilt_rad=math.radians(30.))
    np.testing.assert_allclose(kept, inside.as_matrix(), atol=1e-12)


@pytest.mark.parametrize("policy,tilt", [
    ("preserve", .1), ("tool_axis_down", -.1), ("tool_axis_down", MAX_TOOL_AXIS_TILT_RAD + .01),
    ("tool_axis_down", float("nan")), ("tool_axis_down", True),
])
def test_tool_axis_cone_rejects_invalid_requests(policy, tilt):
    with pytest.raises(ValueError):
        hover_orientation([0., 0., 0., *DIAGONAL.as_quat()], policy, tool_axis_max_tilt_rad=tilt)


def test_hover_goals_carry_the_cone_orientation():
    pose = [.1, .2, .5, *DIAGONAL.as_quat()]
    goals = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=pose, anchor_xyz_m=[.1, .2, .05],
        tip_offset_bounds_xy_m=[[-.02, .02], [-.02, .02]], clearance_bounds_m=[.03, .06],
        entity=simple_entity(), points_m=scene_points(), orientation_policy="tool_axis_down",
        tool_axis_max_tilt_rad=math.radians(30.), max_candidates=3)
    assert len(goals) == 3
    for goal in goals:
        tilt = tool_axis_tilt_rad(Rotation.from_quat(goal.pose_xyz_quat_xyzw[3:]).as_matrix())
        assert tilt == pytest.approx(math.radians(30.), abs=1e-9)
        assert .03 - 1e-9 <= goal.clearance_m <= .06 + 1e-9
