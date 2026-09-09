from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.rm65 import (
    DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
    DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES,
    build_dual_arm_scene_profile,
)
from robot_motion.rm65.scene_profiles import fixed_scene_aabb_obstacles


_EXPECTED_COLLISION_PAIRS = (
    ("arm_link3_proxy_0", "right_arm_link3_proxy_0"),
    ("arm_link3_proxy_0", "right_hand_proxy_0"),
    ("left_hand_proxy_0", "right_arm_link3_proxy_0"),
    ("left_hand_proxy_0", "right_hand_proxy_0"),
    ("left_hand_proxy_0", "camera_pole_0"),
    ("right_hand_proxy_0", "camera_pole_0"),
    ("left_hand_proxy_0", "table_0"),
    ("right_hand_proxy_0", "table_0"),
    ("arm_link3_proxy_0", "table_0"),
    ("right_arm_link3_proxy_0", "table_0"),
    ("arm_link3_proxy_0", "left_camera_pole_0"),
    ("right_arm_link3_proxy_0", "right_camera_pole_0"),
)


def test_dual_arm_hand_proxy_open_excludes_link1_link2_and_link5_camera_pairs():
    profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_open")

    forbidden_geometries = {
        "arm_link1_0",
        "right_arm_link1_0",
        "arm_link2_0",
        "right_arm_link2_0",
        "arm_link3_0",
        "right_arm_link3_0",
        "arm_link4_0",
        "right_arm_link4_0",
        "arm_link5_0",
        "right_arm_link5_0",
    }
    forbidden_pairs = {
        frozenset(("arm_link5_0", "camera_pole_0")),
        frozenset(("right_arm_link5_0", "camera_pole_0")),
    }

    assert DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS == _EXPECTED_COLLISION_PAIRS
    assert profile.collision_pairs == _EXPECTED_COLLISION_PAIRS
    assert tuple(box.geometry_name for box in profile.fixed_scene_boxes) == (
        DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES
    )
    assert {box.geometry_name for box in profile.hand_proxy_boxes} == {
        "arm_link3_proxy_0",
        "left_hand_proxy_0",
        "right_arm_link3_proxy_0",
        "right_hand_proxy_0",
    }
    assert frozenset(("left_hand_proxy_0", "table_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("right_hand_proxy_0", "table_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("arm_link3_proxy_0", "right_arm_link3_proxy_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("arm_link3_proxy_0", "table_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("arm_link3_proxy_0", "left_camera_pole_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("right_arm_link3_proxy_0", "right_camera_pole_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    assert frozenset(("right_arm_link3_proxy_0", "table_0")) in {
        frozenset(pair) for pair in profile.collision_pairs
    }
    active_geometries = {geometry for pair in profile.collision_pairs for geometry in pair}
    assert "arm_link3_proxy_0" in active_geometries
    assert "right_arm_link3_proxy_0" in active_geometries
    for pair in profile.collision_pairs:
        assert pair[0] not in forbidden_geometries
        assert pair[1] not in forbidden_geometries
        assert frozenset(pair) not in forbidden_pairs


def test_dual_arm_hand_proxy_strict_uses_strict_urdf_with_same_profile_rules():
    open_profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_open")
    strict_profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_strict")

    assert strict_profile.urdf_path.name == "rm65_dual_arm_hand_proxy_strict.urdf"
    assert strict_profile.collision_pairs == open_profile.collision_pairs
    assert strict_profile.solution_layout == open_profile.solution_layout

    assert {box.name: box.box_size_xyz for box in open_profile.hand_proxy_boxes} == {
        "left_hand_proxy": (0.06, 0.06, 0.34),
        "right_hand_proxy": (0.06, 0.06, 0.34),
        "arm_link3_proxy": (0.07, 0.165, 0.10),
        "right_arm_link3_proxy": (0.07, 0.165, 0.10),
    }
    assert {box.name: box.box_size_xyz for box in strict_profile.hand_proxy_boxes} == {
        "left_hand_proxy": (0.07, 0.07, 0.36),
        "right_hand_proxy": (0.07, 0.07, 0.36),
        "arm_link3_proxy": (0.10, 0.165, 0.12),
        "right_arm_link3_proxy": (0.10, 0.165, 0.12),
    }

    open_table = next(box for box in open_profile.fixed_scene_boxes if box.name == "table")
    strict_table = next(box for box in strict_profile.fixed_scene_boxes if box.name == "table")
    assert open_table.world_to_box_pose_matrix()[2, 3] == pytest.approx(-0.015)
    assert strict_table.world_to_box_pose_matrix()[2, 3] == pytest.approx(0.005)


def test_fixed_scene_boxes_are_exposed_as_planner_obstacles():
    obstacles = fixed_scene_aabb_obstacles(
        build_dual_arm_scene_profile("dual_arm_hand_proxy_open")
    )

    assert {obstacle.obstacle_id for obstacle in obstacles} == {
        "fixed-scene:camera_pole",
        "fixed-scene:left_camera_pole",
        "fixed-scene:right_camera_pole",
        "fixed-scene:table",
    }
    table = next(
        obstacle
        for obstacle in obstacles
        if obstacle.obstacle_id == "fixed-scene:table"
    )
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            table.center_xyz_m,
            (-0.472, 0.0, -0.015),
        )
    ) < 1e-12
    assert abs(table.maximum_xyz_m[2] - 0.01) < 1e-12


def test_rotated_fixed_scene_box_is_not_silently_approximated_as_an_aabb():
    profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_open")
    rotated = replace(
        profile.fixed_scene_boxes[0],
        joint_origin_rpy=(0.0, 0.0, 0.1),
    )

    with pytest.raises(ValueError, match="rotated"):
        fixed_scene_aabb_obstacles(
            replace(profile, fixed_scene_boxes=(rotated,))
        )
    nested = replace(profile.fixed_scene_boxes[0], parent_link_name="fixture")
    with pytest.raises(ValueError, match="directly to world"):
        fixed_scene_aabb_obstacles(
            replace(profile, fixed_scene_boxes=(nested,))
        )
