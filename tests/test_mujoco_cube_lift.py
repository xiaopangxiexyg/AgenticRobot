from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.mujoco.artifacts import ModelArtifactCache
from sim.mujoco.diagnostics.cube_lift import build_arg_parser
from sim.mujoco.scene import MujocoScene
from sim.mujoco.tasks.cube_lift import (
    CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
    CubeLiftActionSpec,
    CubeLiftEnv,
    CubeLiftObservationSpec,
    CubeLiftTaskConfig,
)


TASK_CONFIG_PATH = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")


def _cached_cube_lift_env() -> CubeLiftEnv:
    config = CubeLiftTaskConfig.load(TASK_CONFIG_PATH)
    scene, _model_artifact = ModelArtifactCache().load_scene(config.base_contract())
    return CubeLiftEnv(config, scene=scene)


def test_cube_lift_cli_parses_random_viewer_options_without_launching_gui() -> None:
    args = build_arg_parser().parse_args(
        [
            "--viewer",
            "--random-actions",
            "--random-action-scale",
            "0.25",
            "--steps",
            "300",
        ]
    )

    assert args.viewer is True
    assert args.random_actions is True
    assert args.random_action_scale == pytest.approx(0.25)
    assert args.steps == 300


@pytest.mark.integration
def test_cube_lift_config_freezes_center_cube_and_task_local_layouts() -> None:
    config = CubeLiftTaskConfig.load(TASK_CONFIG_PATH)
    contract = config.base_contract()
    cube = next(item for item in contract.scene.objects if item.name == config.object_body_name)

    assert config.action_dim == 7
    assert config.observation_dim == 31
    assert config.cube_center_xy_m == pytest.approx((-0.472, 0.0))
    assert config.cube_spawn_qpos[:3] == pytest.approx((-0.472, 0.0, 0.026))
    assert cube.size_m == pytest.approx((0.03, 0.03, 0.03))
    assert cube.rgba == pytest.approx((0.85, 0.18, 0.12, 1.0))
    assert config.to_metadata()["observation_mode"] == "state_privileged"


def test_cube_lift_specs_reject_invalid_action_and_observation() -> None:
    action_spec = CubeLiftActionSpec()
    observation_spec = CubeLiftObservationSpec()

    np.testing.assert_array_equal(action_spec.coerce(np.zeros(7, dtype=np.float32)), np.zeros(7))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        action_spec.coerce(np.full(7, 1.01, dtype=np.float32))
    with pytest.raises(RuntimeError, match="observation"):
        observation_spec.coerce(np.zeros(30, dtype=np.float32))


@pytest.mark.integration
def test_physical_cube_lift_uses_tcp_pose_action_contract() -> None:
    config = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v1.json"
    )
    assert config.uses_pose_action is True
    assert config.action_layout == CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION
    assert CubeLiftActionSpec(layout_version=config.action_layout).dim == 7


@pytest.mark.optional
@pytest.mark.slow_sim
def test_scene_freejoint_helper_resets_center_cube_and_reports_table_contact() -> None:
    pytest.importorskip("mujoco")
    config = CubeLiftTaskConfig.load(TASK_CONFIG_PATH)
    scene = MujocoScene.from_contract(config.base_contract())
    scene.reset_data()
    scene.set_freejoint_state(
        config.object_freejoint_name,
        qpos=config.cube_spawn_qpos,
        qvel=np.zeros(6),
    )

    np.testing.assert_allclose(scene.freejoint_qpos(config.object_freejoint_name), config.cube_spawn_qpos)
    np.testing.assert_allclose(scene.freejoint_qvel(config.object_freejoint_name), np.zeros(6))
    for _ in range(1000):
        scene.mujoco.mj_step(scene.model, scene.data)
    scene.forward()

    cube_qpos = scene.freejoint_qpos(config.object_freejoint_name)
    assert cube_qpos[0:2] == pytest.approx((-0.472, 0.0), abs=1e-6)
    assert cube_qpos[2] == pytest.approx(0.025, abs=0.004)
    assert any(
        {contact.body1_name, contact.body2_name}
        == {config.object_body_name, config.table_body_name}
        for contact in scene.contacts()
    )
    assert scene.state_is_finite()


@pytest.mark.optional
def test_cube_lift_env_has_gymnasium_reset_step_contract() -> None:
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    env = CubeLiftEnv(TASK_CONFIG_PATH)
    observation, info = env.reset(seed=7)

    assert observation.shape == (31,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["execution_mode"] == "kinematic_direct_qpos_debug"
    assert info["grasp_mode"] == "contact_latch_debug"
    assert info["contacts"]["table_contact"]

    next_observation, reward, terminated, truncated, step_info = env.step(
        np.zeros(7, dtype=np.float32)
    )
    assert next_observation.shape == (31,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["scheduler"]["physics_ticks"] in {66, 67}
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.optional
def test_cube_lift_reset_and_action_trace_are_deterministic() -> None:
    pytest.importorskip("gymnasium")
    env = _cached_cube_lift_env()
    actions = (
        np.asarray((0.2, -0.1, 0.0, 0.1, 0.0, -0.1, 0.25), dtype=np.float32),
        np.asarray((-0.1, 0.2, 0.1, 0.0, -0.2, 0.0, -0.25), dtype=np.float32),
    )

    def rollout() -> tuple[np.ndarray, list[tuple[np.ndarray, float, bool, bool]]]:
        initial, _info = env.reset(seed=19)
        result: list[tuple[np.ndarray, float, bool, bool]] = []
        for action in actions:
            observation, reward, terminated, truncated, _info = env.step(action)
            result.append((observation, reward, terminated, truncated))
        return initial, result

    initial_a, trace_a = rollout()
    initial_b, trace_b = rollout()
    np.testing.assert_allclose(initial_a, initial_b, rtol=0.0, atol=1e-12)
    for left, right in zip(trace_a, trace_b):
        np.testing.assert_allclose(left[0], right[0], rtol=0.0, atol=1e-12)
        assert left[1:] == right[1:]
    env.close()


@pytest.mark.optional
def test_cube_lift_holds_left_home_and_maps_right_delta_action() -> None:
    pytest.importorskip("gymnasium")
    env = _cached_cube_lift_env()
    try:
        env.reset(seed=3)
        left_before = env.scene.get_arm_qpos("left")
        action = np.asarray((1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.5), dtype=np.float32)
        env.step(action)

        np.testing.assert_allclose(env.scene.get_arm_qpos("left"), left_before, atol=1e-12)
        expected_right_target = np.clip(
            np.asarray(env.config.right_home_qpos)
            + action[:6] * np.asarray(env.config.right_joint_delta_rad),
            np.asarray(env.config.right_qpos_lower),
            np.asarray(env.config.right_qpos_upper),
        )
        np.testing.assert_allclose(env._right_target_qpos, expected_right_target, atol=1e-12)
        # ``mj_step`` may add sub-0.1-milliradian constraint residual after the
        # direct bridge writes its held qpos. The target above—not this measured
        # post-step state—is the command contract in debug mode.
        np.testing.assert_allclose(
            env.scene.get_arm_qpos("right"),
            expected_right_target,
            rtol=0.0,
            atol=1e-4,
        )
        assert env.scene.get_gripper_close("right_gripper") == pytest.approx(0.1)
    finally:
        env.close()


@pytest.mark.optional
def test_cube_lift_success_and_out_of_bounds_termination_predicates() -> None:
    pytest.importorskip("gymnasium")
    env = _cached_cube_lift_env()
    try:
        env.reset(seed=5)
        lifted_qpos = env.scene.freejoint_qpos(env.config.object_freejoint_name)
        lifted_qpos[2] = env._resting_center_z_m + env.config.lift_delta_m + 0.005
        env.scene.set_freejoint_state(
            env.config.object_freejoint_name,
            qpos=lifted_qpos,
            qvel=np.zeros(6),
        )
        for _ in range(env.config.success_hold_physics_ticks - 1):
            assert not env._check_physics_termination(None)
        assert env._check_physics_termination(None)
        assert env._termination_reason == "height_only_lift_success"

        env.reset(seed=5)
        outside_qpos = env.scene.freejoint_qpos(env.config.object_freejoint_name)
        outside_qpos[0] = env.config.out_of_bounds_xyz_upper_m[0] + 0.01
        env.scene.set_freejoint_state(
            env.config.object_freejoint_name,
            qpos=outside_qpos,
            qvel=np.zeros(6),
        )
        assert env._check_physics_termination(None)
        assert env._termination_reason == "cube_out_of_bounds"
    finally:
        env.close()


@pytest.mark.optional
def test_normal_v2_lift_requires_bilateral_finger_contact() -> None:
    pytest.importorskip("mujoco")
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json",
        include_training_reward_state=True,
    )
    try:
        _observation, info = env.reset(seed=5)
        assert info["left_finger_cube_contact"] is False
        assert info["right_finger_cube_contact"] is False
        assert info["bilateral_finger_cube_contact"] is False

        lifted_qpos = env.kernel.scene.freejoint_qpos(
            env.config.object_freejoint_name
        )
        lifted_qpos[2] = (
            env.kernel._resting_center_z + env.config.lift_delta_m + 0.005
        )
        env.kernel.scene.set_freejoint_state(
            env.config.object_freejoint_name,
            qpos=lifted_qpos,
            qvel=np.zeros(6),
            forward=True,
        )
        for _ in range(env.config.success_hold_physics_ticks + 1):
            assert env.kernel._check_termination(check_success=True) is False
        assert env.kernel._termination_reason is None
    finally:
        env.close()


@pytest.mark.optional
@pytest.mark.slow_sim
def test_cube_lift_hundred_random_headless_resets_remain_finite() -> None:
    pytest.importorskip("gymnasium")
    rng = np.random.default_rng(20260715)
    env = _cached_cube_lift_env()
    try:
        for episode_index in range(100):
            _observation, _info = env.reset(seed=episode_index)
            observation, _reward, _terminated, _truncated, _info = env.step(
                rng.uniform(-1.0, 1.0, 7).astype(np.float32)
            )
            assert env.observation_space.contains(observation)
            assert np.all(np.isfinite(observation))
    finally:
        env.close()
