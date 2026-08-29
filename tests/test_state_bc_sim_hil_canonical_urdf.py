from __future__ import annotations

from pathlib import Path

import pytest

from control_station.sim_hil_profile.config import (
    LOCAL_DEFAULT_TASK_CONFIG,
    resolve_default_task_config_path,
)
from sim.mujoco.scene import MujocoScene
from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TASK_CONFIG = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose_canonical.json"
)
CANONICAL_URDF = (
    "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
 )
CANONICAL_CONTRACT_SHA256 = (
    "dbf0cd77b5e8ffbd0f7951f9a4ae10c6c25757e6f32657e16f1fe14cf100938d"
)
CANONICAL_GRIPPER_PROFILE = "dmtac_bimanual_position_eg2_4b2_cad_20260818.v1"


def test_sim_hil_defaults_to_canonical_state_bc_task() -> None:
    assert LOCAL_DEFAULT_TASK_CONFIG == CANONICAL_TASK_CONFIG
    assert resolve_default_task_config_path({}) == CANONICAL_TASK_CONFIG


def test_canonical_task_resolves_cad_gripper_and_physics_contract() -> None:
    task = CubeLiftTaskConfig.load(CANONICAL_TASK_CONFIG)
    contract = task.base_contract()

    assert task.base_contract_path == "sim/configs/rm65_bimanual_physics_v1.json"
    assert task.base_contract_sha256 == CANONICAL_CONTRACT_SHA256
    assert contract.scene.urdf_path == CANONICAL_URDF
    assert task.gripper_actuation_profile().name == CANONICAL_GRIPPER_PROFILE
    assert task.left_gripper_name == "left_eg2_4b2_cad"
    assert task.right_gripper_name == "right_eg2_4b2_cad"
    assert task.right_grasp_body_name == "right_eg2_4b2_cad_ee_link_4"
    assert task.grasp_latch_left_finger_body_name == "right_eg2_4b2_cad_l_3"
    assert task.grasp_latch_right_finger_body_name == "right_eg2_4b2_cad_r_3"
    assert {spec.body_name for spec in task.gripper_contact_surface_specs} == {
        "right_eg2_4b2_cad_l_3",
        "right_eg2_4b2_cad_r_3",
    }


@pytest.mark.optional
@pytest.mark.slow_sim
def test_canonical_task_builds_headless_mujoco_scene() -> None:
    pytest.importorskip("mujoco")
    task = CubeLiftTaskConfig.load(CANONICAL_TASK_CONFIG)
    scene = MujocoScene.from_contract(
        task.base_contract(),
        position_actuation_profile=task.position_actuation_profile(),
        gripper_actuation_profile=task.gripper_actuation_profile(),
    )

    assert set(scene.gripper_specs) == {
        "left_eg2_4b2_cad",
        "right_eg2_4b2_cad",
    }
    for body_name in (
        "right_eg2_4b2_cad_ee_link_4",
        "right_eg2_4b2_cad_l_3",
        "right_eg2_4b2_cad_r_3",
        "debug_cube",
        "table",
    ):
        assert scene.has_body(body_name), body_name
    assert scene.state_is_finite()
