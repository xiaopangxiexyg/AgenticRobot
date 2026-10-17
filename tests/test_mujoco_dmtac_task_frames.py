"""Current CAD tasks must use the calibrated TCP, not a missing legacy link."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco.tool_frames import (
    load_endpoint_to_tool_transform,
    resolve_task_grasp_frame,
)

TASK = "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"


def test_legacy_body_frame_keeps_metadata_and_body_identity():
    config = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
    )
    assert "grasp_frame_source" not in config.to_metadata()
    names = []
    scene = SimpleNamespace(body_id=lambda name: names.append(name) or 42)
    body_id, offset = resolve_task_grasp_frame(scene, config)
    assert body_id == 42 and names == [config.right_grasp_body_name]
    np.testing.assert_array_equal(offset, np.eye(4))


@pytest.mark.parametrize("side", ("left", "right"))
def test_calibrated_grasp_frame_uses_physical_side_endpoint_and_profile(side):
    config = SimpleNamespace(grasp_frame_source="tool_tcp", controlled_side=side,
        **{f"{side}_tool_profile_path":
           f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"})
    names = []
    scene = SimpleNamespace(arm_specs={side: SimpleNamespace(controlled_body=f"{side}_arm_link6")},
                            body_id=lambda name: names.append(name) or 17)
    body_id, offset = resolve_task_grasp_frame(scene, config)
    assert body_id == 17 and names == [f"{side}_arm_link6"]
    np.testing.assert_allclose(offset[:3, 3], [0., 0., .14])
    with pytest.raises(ValueError, match="expected"):
        load_endpoint_to_tool_transform(getattr(config, f"{side}_tool_profile_path"),
                                        side="left" if side == "right" else "right")


def test_current_task_is_explicit_physical_contact_not_scripted_or_latched():
    config = CubeLiftTaskConfig.load(TASK)
    assert config.grasp_frame_source == "tool_tcp"
    assert config.to_metadata()["grasp_frame_source"] == "tool_tcp"
    assert config.uses_actuated_gripper and config.uses_pose_action
    assert config.object_size_m == (.07, .07, .07)
    assert not config.scripted_pregrasp_reset and not config.scripted_contact_reset
    assert config.success_requires_bilateral_contact
    with pytest.raises(ValueError, match="grasp_frame_source"):
        replace(config, grasp_frame_source="invented")
    with pytest.raises(ValueError, match="physical actuated"):
        replace(config, grasp_mode="contact_latch_debug")


@pytest.mark.integration
def test_current_cad_kernel_reward_offset_matches_endpoint_tcp_after_rotation():
    pytest.importorskip("mujoco")
    from sim.mujoco.tasks.cube_lift.kernel import CubeLiftKernel

    config = CubeLiftTaskConfig.load(TASK)
    kernel = CubeLiftKernel(config, use_model_cache=False, use_settled_snapshot=False)
    try:
        observation, _ = kernel.reset(seed=0)
        assert np.all(np.isfinite(observation))
        scene = kernel.scene
        assert scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_BODY,
                                      config.right_grasp_body_name) == -1
        idx = scene.joint_qpos_addr(scene.arm_specs["right"].joint_names[-1])
        scene.data.qpos[idx] += .1
        scene.mujoco.mj_forward(scene.model, scene.data)
        endpoint = scene.body_pose(scene.arm_specs["right"].controlled_body)
        tcp = endpoint @ load_endpoint_to_tool_transform(config.right_tool_profile_path, side="right")
        features = kernel._training_reward_state()
        np.testing.assert_allclose(features[20:23],
            scene.body_pose(config.object_body_name)[:3, 3] - tcp[:3, 3], atol=1e-7)
        assert len(scene.camera_names) == 3
        assert scene.model.npair == 4
        np.testing.assert_allclose(scene.model.pair_friction[:, :2], 1.2)
    finally:
        kernel.close()
