from __future__ import annotations

from rm65_inspire_tac.pipeline import (
    DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
    DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES,
    build_dual_arm_scene_profile,
)


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

    open_table = next(box for box in open_profile.fixed_scene_boxes if box.name == "table")
    strict_table = next(box for box in strict_profile.fixed_scene_boxes if box.name == "table")
    assert strict_table.world_to_box_pose_matrix()[2, 3] > open_table.world_to_box_pose_matrix()[2, 3]
