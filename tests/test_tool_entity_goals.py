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


@pytest.mark.parametrize("local_tip", [(.01, 0., .06), (0., 0., .05), (0., 0., -.06)])
def test_entity_tip_must_be_center_of_the_rotated_approach_face(local_tip):
    rotation = Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix()
    center = np.array([.03, -.02, .01])
    def box(tip):
        return ToolEntityBox(tcp_from_box_xyz_m=tuple(center),
            tcp_from_box_rotation=tuple(rotation.ravel()), half_extents_m=(.04, .01, .06),
            tip_center_tcp_m=tuple(center + rotation @ np.asarray(tip)))
    good = box((0., 0., .06))
    np.testing.assert_allclose(good.tip_center_tcp_m, center + rotation[:, 2] * .06)
    with pytest.raises(ValueError, match="face center"):
        box(local_tip)


@pytest.mark.parametrize("anchor,raw", [
    ([.0015, 0., .05], [.0405, .001, .091]),
    ([.0015, 0., .0485], [.0015, .001, .3484]),
])
def test_hover_counts_voxels_overlapping_footprint_or_support_window(anchor, raw):
    from robot_motion.observed_obstacles import ObservedObstacleCloud

    cloud = ObservedObstacleCloud(reference_frame_id="station", points_m=np.asarray([anchor, raw]), evidence={})
    goal, = generate_entity_hover_goals(current_tcp_pose_xyz_quat_xyzw=[anchor[0], anchor[1], .5, *DOWN[3:]],
        anchor_xyz_m=anchor, tip_offset_bounds_xy_m=[[0., 0.], [0., 0.]], clearance_bounds_m=[.04, .04],
        entity=simple_entity(), points_m=cloud.points_m, support_margin_m=cloud.quantization_margin_m,
        max_candidates=1)
    assert goal.metrics.support_height_m >= raw[2]
    assert goal.metrics.lowest_point_z_m >= raw[2] + .04


def test_rotated_footprint_expansion_includes_only_nearby_voxels():
    angle, margin = math.pi / 4., math.sqrt(3.) * .002
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    corners = simple_entity().corners_world(transform)
    direction = np.array([math.cos(angle), math.sin(angle)])
    near = direction * (.04 + .002)
    far = direction * (.04 + margin + .01)
    points = np.array([[*near, .09], [*far, .2]])
    height, count, overhead = footprint_support(points, corners, floor_z_m=.05, ceiling_z_m=.35,
                                                 horizontal_margin_m=margin)
    assert (height, count, overhead) == pytest.approx((.09, 1, 0))
