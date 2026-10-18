from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.rm65 import (
    DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS,
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


def test_grasp_full_profile_adds_arm_self_cross_and_fixed_pairs_without_teleop_drift():
    strict = build_dual_arm_scene_profile("dual_arm_hand_proxy_strict")
    full = build_dual_arm_scene_profile("dual_arm_hand_proxy_grasp_full")
    pairs = {frozenset(pair) for pair in full.collision_pairs}

    assert strict.collision_pairs == _EXPECTED_COLLISION_PAIRS
    assert full.urdf_path == strict.urdf_path
    assert {frozenset(pair) for pair in strict.collision_pairs} <= pairs
    assert len(pairs) == len(full.collision_pairs)
    for prefix in ("", "right_"):
        assert frozenset((f"{prefix}arm_link3_0", f"{prefix}arm_link5_0")) in pairs
        assert frozenset((f"{prefix}arm_link3_0", f"{prefix}arm_link4_0")) not in pairs
        assert frozenset((f"{prefix}arm_link3_0", f"{prefix}arm_link3_proxy_0")) not in pairs
        for index in range(1, 7):
            for fixed in DUAL_ARM_HAND_PROXY_COLLISION_FIXED_GEOMETRIES:
                assert frozenset((f"{prefix}arm_link{index}_0", fixed)) in pairs
    for left_index in range(1, 7):
        for right_index in range(1, 7):
            assert frozenset(
                (f"arm_link{left_index}_0", f"right_arm_link{right_index}_0")
            ) in pairs


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


def test_dual_arm_hand_proxy_dmtac_is_opt_in_and_loads_both_hand_boxes():
    dmtac_profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_strict")
    open_profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_open")

    assert dmtac_profile.urdf_path.name == "rm65_dual_arm_hand_proxy_dmtac_strict.urdf"
    assert open_profile.urdf_path.name == "rm65_dual_arm_hand_proxy_open.urdf"
    assert dmtac_profile.name != open_profile.name
    assert dmtac_profile.collision_pairs == DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS

    hand_boxes = {
        box.geometry_name: box
        for box in dmtac_profile.hand_proxy_boxes
        if "hand_proxy" in box.geometry_name
    }
    assert set(hand_boxes) == {
        "left_hand_proxy_0",
        "left_hand_proxy_1",
        "right_hand_proxy_0",
        "right_hand_proxy_1",
    }
    assert hand_boxes["left_hand_proxy_0"].box_size_xyz == (0.07, 0.07, 0.36)
    assert hand_boxes["right_hand_proxy_0"].box_size_xyz == (0.07, 0.07, 0.36)
    assert hand_boxes["left_hand_proxy_1"].box_size_xyz == (
        0.169923167782,
        0.039281625101,
        0.082386181773,
    )
    assert hand_boxes["right_hand_proxy_1"].box_size_xyz == (
        0.169865422888,
        0.039404159584,
        0.082502845746,
    )

    known_geometries = {
        box.geometry_name
        for box in (*dmtac_profile.hand_proxy_boxes, *dmtac_profile.fixed_scene_boxes)
    }
    for pair in dmtac_profile.collision_pairs:
        assert pair[0] in known_geometries
        assert pair[1] in known_geometries


def test_dmtac_grasp_profile_changes_only_table_geometry_to_reviewed_cad():
    strict = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_strict")
    grasp = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_grasp")
    assert grasp.urdf_path == strict.urdf_path
    assert grasp.collision_pairs == strict.collision_pairs
    assert len(grasp.collision_pairs) == 20
    assert grasp.hand_proxy_boxes == strict.hand_proxy_boxes
    assert grasp.fixed_scene_boxes[:-1] == strict.fixed_scene_boxes[:-1]
    assert strict.fixed_scene_box_overrides == ()
    assert grasp.fixed_scene_box_overrides == (grasp.fixed_scene_boxes[-1],)
    old_table, new_table = (p.fixed_scene_boxes[-1] for p in (strict, grasp))
    assert old_table.joint_origin_xyz == new_table.joint_origin_xyz
    assert old_table.joint_origin_rpy == new_table.joint_origin_rpy
    assert old_table.box_size_xyz == new_table.box_size_xyz
    assert old_table.world_to_box_pose_matrix()[2, 3] == pytest.approx(.005)
    assert new_table.world_to_box_pose_matrix()[2, 3] == pytest.approx(-.015)
    table = next(b for b in fixed_scene_aabb_obstacles(grasp)
        if b.obstacle_id == "fixed-scene:table")
    assert table.maximum_xyz_m[2] == pytest.approx(.01)
    np.testing.assert_array_equal(strict.left_pose_transform, grasp.left_pose_transform)
    np.testing.assert_array_equal(strict.right_pose_transform, grasp.right_pose_transform)


def test_dmtac_grasp_table_override_reaches_native_collision_and_visual_models():
    pin = pytest.importorskip("pinocchio")
    from robot_motion.rm65.ik import (
        RM65DualArmCollisionIKConfig, build_rm65_dual_arm_collision_ik_solver,
    )
    from robot_motion.rm65.position_collision import load_world_from_fixed_link_transform

    bundles = [build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(scene_profile=name)) for name in (
            "dual_arm_hand_proxy_dmtac_strict", "dual_arm_hand_proxy_dmtac_grasp")]
    strict, grasp = bundles
    # Direct URDF loading is an independent reference, not the override helper.
    raw = pin.RobotWrapper.BuildFromURDF(str(strict.scene_profile.urdf_path),
        package_dirs=[str(p) for p in strict.scene_profile.package_dirs])
    cad_path = strict.scene_profile.urdf_path.parent.parent / (
        "realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf")
    cad = pin.RobotWrapper.BuildFromURDF(str(cad_path), package_dirs=[str(cad_path.parent)])
    for kind in ("collision_model", "visual_model"):
        models = [getattr(b.solver.robot, kind) for b in bundles]
        old = models[0].geometryObjects[models[0].getGeometryId("table_0")]
        new = models[1].geometryObjects[models[1].getGeometryId("table_0")]
        raw_model, cad_model = getattr(raw, kind), getattr(cad, kind)
        raw_table = raw_model.geometryObjects[raw_model.getGeometryId("table_0")]
        cad_table = cad_model.geometryObjects[cad_model.getGeometryId("table_0")]
        np.testing.assert_allclose(old.placement.homogeneous, raw_table.placement.homogeneous)
        np.testing.assert_allclose(new.placement.homogeneous, cad_table.placement.homogeneous)
        np.testing.assert_array_equal(old.geometry.halfSide, new.geometry.halfSide)
        assert len(models[0].geometryObjects) == len(models[1].geometryObjects)
        for geometry in models[0].geometryObjects:
            if geometry.name != "table_0":
                other = models[1].geometryObjects[models[1].getGeometryId(geometry.name)]
                np.testing.assert_array_equal(geometry.placement.homogeneous, other.placement.homogeneous)
    for frame in ("workspace_near_boundary_frame", "base_link", "right_base_link"):
        np.testing.assert_array_equal(
            load_world_from_fixed_link_transform(strict.scene_profile, link_name=frame),
            load_world_from_fixed_link_transform(grasp.scene_profile, link_name=frame))
    for q in (pin.neutral(raw.model), np.linspace(-.3, .3, raw.model.nq)):
        for robot in (strict.solver.robot, grasp.solver.robot):
            pin.framesForwardKinematics(robot.model, robot.data, q)
        for frame in ("arm_link6", "right_arm_link6", "workspace_near_boundary_frame"):
            np.testing.assert_array_equal(
                strict.solver.robot.data.oMf[strict.solver.robot.model.getFrameId(frame)].homogeneous,
                grasp.solver.robot.data.oMf[grasp.solver.robot.model.getFrameId(frame)].homogeneous)


@pytest.mark.parametrize("change", ("missing", "articulated", "size", "parent"))
def test_native_fixed_scene_override_rejects_non_geometry_only_changes(change):
    pin = pytest.importorskip("pinocchio")
    from robot_motion.rm65.dual_arm_ik import MultiTaskIKSolver

    profile = build_dual_arm_scene_profile("dual_arm_hand_proxy_dmtac_grasp")
    box = profile.fixed_scene_box_overrides[0]
    if change == "missing":
        box = replace(box, geometry_name="missing_table")
    elif change == "articulated":
        box = replace(box, geometry_name="left_hand_proxy_0", box_size_xyz=(.07, .07, .36))
    elif change == "size":
        box = replace(box, box_size_xyz=(1., 1., 1.))
    else:
        box = replace(box, parent_link_name="table")
    solver = object.__new__(MultiTaskIKSolver)
    solver.pin = pin
    with pytest.raises(ValueError):
        solver._initialize_robot(urdf_path=str(profile.urdf_path),
            package_dirs=[str(p) for p in profile.package_dirs], root_joint=None,
            list_of_joints_to_lock=None, collision_pair_names=[],
            fixed_scene_box_overrides=(box,))
