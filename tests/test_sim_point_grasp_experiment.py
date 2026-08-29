from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.sim_point_grasp.evaluation import (
    SimPointGraspTrialResult,
    apply_parallel_jaw_wrist_symmetry,
    bounded_pose_action,
    build_pregrasp_pose,
    build_sim_point_grasp_task_config,
    draw_grasp_overlay,
    optical_camera_pose_world,
    project_world_point,
    save_summary_sheet,
)
from sim.mujoco.pointcloud import render_intrinsics_from_fovy
from sim.mujoco.tasks.cube_lift.pose_action import inset_joint_position_target


def test_optical_camera_pose_does_not_use_mujoco_render_frame() -> None:
    optical = np.eye(4)
    optical[:3, 3] = [0.1, 0.2, 0.3]
    render = optical.copy()
    render[:3, :3] = np.diag([1.0, -1.0, -1.0])

    class _Scene:
        def camera_link_pose(self, name):
            assert name == "front"
            return optical

        def camera_pose(self, name):
            assert name == "front"
            return render

    assert np.array_equal(optical_camera_pose_world(_Scene()), optical)


def test_project_world_point_uses_optical_xyz_and_image_bounds() -> None:
    intrinsics = render_intrinsics_from_fovy(width=640, height=480, fovy_deg=60.0)
    u, v, depth = project_world_point(
        np.asarray((0.0, 0.0, 1.0)),
        world_from_camera=np.eye(4),
        intrinsics=intrinsics,
    )
    assert (u, v) == (320, 240)
    assert depth == 1.0


def test_bounded_pose_action_reaches_limits_without_changing_gripper() -> None:
    current = np.eye(4)
    desired = np.eye(4)
    desired[:3, 3] = [0.04, -0.01, 0.005]
    desired[:3, :3] = Rotation.from_rotvec([0.30, -0.05, 0.0]).as_matrix()
    action = bounded_pose_action(
        current,
        desired,
        translation_step_m=(0.02, 0.02, 0.02),
        rotation_step_rad=(0.15, 0.15, 0.15),
    )
    np.testing.assert_allclose(action, [1.0, -0.5, 0.25, 1.0, -1.0 / 3.0, 0.0, 0.0], atol=1e-6)


def test_pregrasp_backs_away_on_candidate_approach_axis() -> None:
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
    grasp[:3, 3] = [0.4, -0.2, 0.3]
    pregrasp = build_pregrasp_pose(grasp, 0.07)
    np.testing.assert_allclose(pregrasp[:3, :3], grasp[:3, :3])
    np.testing.assert_allclose(pregrasp[:3, 3], grasp[:3, 3] - grasp[:3, 2] * 0.07)


def test_parallel_jaw_wrist_symmetry_swaps_fingers_but_preserves_approach() -> None:
    grasp = np.eye(4)
    grasp[:3, 3] = [0.1, 0.2, 0.3]
    flipped = apply_parallel_jaw_wrist_symmetry(grasp, 180)
    np.testing.assert_allclose(flipped[:3, 0], -grasp[:3, 0])
    np.testing.assert_allclose(flipped[:3, 1], -grasp[:3, 1])
    np.testing.assert_allclose(flipped[:3, 2], grasp[:3, 2])
    np.testing.assert_allclose(flipped[:3, 3], grasp[:3, 3])


def test_pose_qref_is_inset_without_widening_hard_limits() -> None:
    lower = np.asarray([-1.0, -2.0])
    upper = np.asarray([1.0, 2.0])
    clipped = inset_joint_position_target(
        np.asarray([-1.5, 2.5]),
        lower,
        upper,
        margin_rad=1e-3,
    )
    np.testing.assert_allclose(clipped, [-0.999, 1.999])
    assert np.all(clipped > lower)
    assert np.all(clipped < upper)


def test_grasp_overlay_changes_pixels_and_keeps_shape() -> None:
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    intrinsics = {"width": 120, "height": 100, "fx": 100.0, "fy": 100.0, "cx": 60.0, "cy": 50.0}
    camera_from_tcp = np.eye(4)
    camera_from_tcp[:3, 3] = [0.0, 0.0, 0.5]
    output = draw_grasp_overlay(
        rgb,
        camera_from_tcp=camera_from_tcp,
        gripper_width_m=0.04,
        intrinsics=intrinsics,
        click_uv=(60, 50),
        bbox_xyxy=(45, 35, 75, 65),
    )
    assert output.shape == rgb.shape
    assert int(output.sum()) > 0


def test_evaluation_task_randomizes_observable_scene_and_requires_contact() -> None:
    config = build_sim_point_grasp_task_config(
        cube_spawn_xy_jitter_m=(0.01, 0.02),
        cube_spawn_yaw_jitter_rad=math.pi / 6.0,
    )
    assert config.wrist_rgb_enabled
    assert config.wrist_rgb_camera_name == "front"
    assert config.randomizes_cube_spawn
    assert config.success_requires_bilateral_contact
    assert config.object_size_m == (0.04, 0.04, 0.10)
    assert config.gripper_actuation_profile_name == (
        "dmtac_bimanual_position_eg2_4b2_cad_20260818.v1"
    )
    assert config.left_gripper_name == "left_eg2_4b2_cad"
    assert config.right_gripper_name == "right_eg2_4b2_cad"
    assert config.right_grasp_body_name == "right_eg2_4b2_cad_ee_link_4"
    assert config.right_tool_profile_path.endswith(
        "rm65_right_DMTAC_M_4b2_cad_tool_tcp.json"
    )
    assert config.grasp_latch_left_finger_body_name == (
        "right_eg2_4b2_cad_ee_link_2"
    )
    assert config.grasp_latch_right_finger_body_name == (
        "right_eg2_4b2_cad_ee_link_3"
    )
    assert "sim_point_grasp/configs" in config.base_contract_path
    contract = config.base_contract()
    assert contract.scene.urdf_path.endswith(
        "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
    )
    assert contract.scene.objects[0].size_m == config.object_size_m
    first = config.sample_cube_spawn_qpos(np.random.default_rng(7))
    second = config.sample_cube_spawn_qpos(np.random.default_rng(7))
    assert first == second


def test_summary_sheet_labels_trial_outcomes(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    from PIL import Image

    Image.new("RGB", (120, 60), color=(20, 30, 40)).save(
        trial / "execution_montage.png"
    )
    result = SimPointGraspTrialResult(
        seed=3,
        success=False,
        failure_stage="perception",
        reason="no candidate",
        termination_reason=None,
        selected_score=None,
        selected_model_score=None,
        selected_width_m=None,
        candidate_count=0,
        click_uv=(10, 10),
        mask_bbox_xyxy=None,
        max_lift_m=0.0,
        bilateral_contact_observed=False,
        motion_steps=0,
        safety_intervention_steps=0,
        ik_rejection_steps=0,
        trial_dir=str(trial),
    )
    output = tmp_path / "summary.jpg"
    save_summary_sheet(output, [result])
    assert output.is_file()
    assert Image.open(output).size == (960, 274)
