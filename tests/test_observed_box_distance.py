"""Exactness of memoized observed-voxel box distances."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.observed_obstacles import ObservedBoxDistance, ObservedObstacleCloud

pytestmark = pytest.mark.unit


def reference_check_boxes(distance, boxes, *, clearance_m):
    """The pre-memo algorithm, kept verbatim as the oracle for exactness."""
    cloud = distance.cloud
    required = float(clearance_m) + cloud.quantization_margin_m
    minimum, nearest = float(np.nextafter(clearance_m, np.inf)), None
    for name, world_from_box, half_size in boxes:
        transform = np.asarray(world_from_box, dtype=float)
        half = np.asarray(half_size, dtype=float)
        candidates = distance.tree.query_ball_point(transform[:3, 3], float(np.linalg.norm(half)) + required)
        if not candidates:
            continue
        local = (cloud.points_m[candidates] - transform[:3, 3]) @ transform[:3, :3]
        delta = np.abs(local) - half
        signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
        index = int(np.argmin(signed))
        value = float(signed[index]) - cloud.quantization_margin_m
        if value < minimum:
            minimum = value
            nearest = {"geometry_name": name, "occupied_center_world_m": cloud.points_m[candidates[index]].tolist()}
    return minimum, nearest


@pytest.fixture
def distance():
    rng = np.random.default_rng(7)
    grid = np.stack(np.meshgrid(np.arange(-.3, .3, .004), np.arange(-.3, .3, .004)), -1).reshape(-1, 2)
    table = np.column_stack([grid, np.zeros(len(grid))])
    clutter = rng.uniform([-.2, -.2, 0.], [.2, .2, .25], (3000, 3))
    return ObservedBoxDistance(ObservedObstacleCloud("station", np.vstack([table, clutter]), {"test": True}))


def random_box(rng, name):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    transform[:3, 3] = rng.uniform([-.25, -.25, -.02], [.25, .25, .35])
    return name, transform, rng.uniform([.01, .01, .02], [.09, .05, .19])


def test_memoized_results_equal_the_reference_bit_for_bit(distance):
    rng = np.random.default_rng(11)
    fixed = [random_box(rng, f"fixed-{i}") for i in range(4)]
    for clearance in (.02, .005):
        for step in range(300):
            moving = [random_box(rng, f"moving-{i}") for i in range(4)]
            boxes = tuple(fixed + moving)
            assert distance.check_boxes(boxes, clearance_m=clearance) == reference_check_boxes(
                distance, boxes, clearance_m=clearance)
    # The fixed boxes were measured once per clearance and then reused.
    assert distance.box_cache_hits >= 2 * 299 * 4
    assert len(distance._box_cache) <= ObservedBoxDistance._CACHE_LIMIT


def test_any_pose_or_radius_change_is_a_new_measurement(distance):
    name, transform, half = random_box(np.random.default_rng(3), "box")
    first = distance.check_boxes(((name, transform, half),), clearance_m=.02)
    misses = distance.box_cache_misses
    moved = transform.copy()
    moved[0, 3] = np.nextafter(moved[0, 3], np.inf)
    distance.check_boxes(((name, moved, half),), clearance_m=.02)
    distance.check_boxes(((name, transform, half),), clearance_m=.03)
    assert distance.box_cache_misses == misses + 2
    assert distance.check_boxes(((name, transform, half),), clearance_m=.02) == first
    assert distance.box_cache_misses == misses + 2


def test_box_touching_the_table_reports_the_reference_nearest_point(distance):
    transform = np.eye(4)
    transform[:3, 3] = [.01, .02, .015]
    boxes = (("hand", transform, np.array([.035, .035, .02])),)
    result = distance.check_boxes(boxes, clearance_m=.02)
    assert result == reference_check_boxes(distance, boxes, clearance_m=.02)
    assert result[0] < 0 and result[1]["geometry_name"] == "hand"
    assert distance.check_boxes(boxes, clearance_m=.02) == result
