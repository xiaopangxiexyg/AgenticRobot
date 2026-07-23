from pathlib import Path

import numpy as np
import pytest


TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")
DEFAULT_PHYSICAL_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json"
)
PRIVILEGED_EXPERT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_v1.json"
)
NORMAL_RGB_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v1.json"
)
PRIVILEGED_EXPERT_RGB_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v1.json"
)
NORMAL_RGB_48_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json"
)
NORMAL_RGB_MOTION_TRACKING_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json"
)
NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json"
)
PRIVILEGED_EXPERT_RGB_48_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v2.json"
)


def test_temp_training_defaults_to_fully_actuated_task() -> None:
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.task_config == DEFAULT_PHYSICAL_TASK_CONFIG
    assert args.log_interval_updates == 1
    assert args.print_full_metrics is False


def test_normal_rgb_v4_is_the_no_episode_clock_real_observation_profile() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_observation_contract

    config = CubeLiftTaskConfig.load(NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_TASK_CONFIG)
    contract = build_observation_contract(config)
    assert config.policy_class == "normal"
    assert config.proprio_observation_dim == 28
    assert config.observation_dim == 28 + 3 * 48 * 48
    assert config.wrist_rgb_shape == (3, 48, 48)
    assert "episode_progress" not in contract.field_names
    assert config.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v4")


def test_temp_training_progress_line_shows_progress_and_key_metrics() -> None:
    from sim.mujoco_temp_pipeline.train import format_training_progress

    line = format_training_progress(
        {
            "update": 10,
            "global_step": 10240,
            "wall_time_seconds": 50.0,
            "transitions_per_second": 204.8,
            "completed_episodes": 71,
            "successful_episodes": 2,
            "success_rate": 2.0 / 71.0,
            "raw_reward_mean": -0.0022,
            "shaped_reward_mean": 0.0365,
            "recent_episode_raw_return_mean": -0.304,
            "reward_distance_m_mean": 0.334,
            "reward_gripper_close_mean": 0.39,
            "reward_lift_fraction_mean": 0.0,
            "safety_projection_rate": 0.13,
            "safety_rejection_rate": 0.02,
            "clearance_violation_count": 0,
            "hard_penetration_count": 0,
            "arm_tracking_error_rad_mean": 0.0179,
            "gripper_tracking_error_mean": 0.09,
            "policy_loss": -0.004,
            "value_loss": 0.33,
            "approx_kl": 0.0006,
        },
        total_timesteps=2_000_000,
        next_checkpoint=50_000,
    )

    assert "steps=10,240/2,000,000 (  0.5%)" in line
    assert "eta=" in line
    assert "succ=2( 2.8%)" in line
    assert "reach=0.334m" in line
    assert "proj=13.0%" in line
    assert "clr=0 pen=0" in line
    assert "armerr=0.0179rad" in line
    assert "griperr=0.090" in line
    assert "kl=6.00e-04" in line
    assert "ckpt=next=50,000" in line
    assert len(line) < 700


def test_temp_training_can_restore_full_json_terminal_output() -> None:
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--log-interval-updates", "10", "--print-full-metrics"]
    )
    assert args.log_interval_updates == 10
    assert args.print_full_metrics is True


def test_temp_training_rgb_health_guard_is_explicit() -> None:
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args(["--require-rgb-health"])
    assert args.require_rgb_health is True


def test_normal_and_expert_configs_share_identified_actuators() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(DEFAULT_PHYSICAL_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_TASK_CONFIG)
    assert normal.policy_class == "normal"
    assert normal.observation_mode == "proprioceptive_real_motion_tracking_no_episode_clock_wrist_rgb"
    assert normal.observation_dim == 28 + 3 * 48 * 48
    assert normal.grasp_mode == "physical_contact"
    assert expert.policy_class == "privileged_expert"
    assert expert.observation_mode == "state_privileged"
    assert expert.observation_dim == 31
    assert expert.grasp_mode == "contact_latch_debug"
    assert normal.position_actuation_profile_name == expert.position_actuation_profile_name
    assert normal.position_actuation_profile_sha256 == expert.position_actuation_profile_sha256
    assert normal.gripper_actuation_profile_name == expert.gripper_actuation_profile_name
    assert normal.gripper_actuation_profile_sha256 == expert.gripper_actuation_profile_sha256


def test_rgb_policy_configs_append_only_active_wrist_camera() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(NORMAL_RGB_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_RGB_TASK_CONFIG)
    assert normal.wrist_rgb_enabled is True
    assert normal.wrist_rgb_camera_name == "right_wrist"
    assert normal.wrist_rgb_shape == (3, 32, 32)
    assert normal.observation_dim == 15 + 3 * 32 * 32
    assert expert.observation_dim == 31 + 3 * 32 * 32
    assert normal.action_layout == expert.action_layout
    assert normal.position_actuation_profile_sha256 == expert.position_actuation_profile_sha256
    assert normal.gripper_actuation_profile_sha256 == expert.gripper_actuation_profile_sha256


def test_current_rgb_policy_configs_use_versioned_48_by_48_input() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(NORMAL_RGB_48_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_RGB_48_TASK_CONFIG)
    assert normal.wrist_rgb_shape == (3, 48, 48)
    assert expert.wrist_rgb_shape == (3, 48, 48)
    assert normal.observation_dim == 15 + 3 * 48 * 48
    assert expert.observation_dim == 31 + 3 * 48 * 48
    assert normal.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v2")
    assert expert.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v2")


def test_normal_rgb_v3_uses_real_observable_motion_tracking_contract() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(NORMAL_RGB_MOTION_TRACKING_TASK_CONFIG)
    assert config.policy_class == "normal"
    assert config.normal_proprio_profile == (
        "qpos_close_motion_tracking_action_progress.v1"
    )
    assert config.proprio_observation_dim == 29
    assert config.observation_dim == 29 + 3 * 48 * 48
    assert config.wrist_rgb_spatial_pool_size == 3
    assert config.observation_layout.endswith(
        "right_wrist_rgb_chw_3x48x48.v3"
    )
    assert config.reward_shaping["staged_grasp"] is True


def test_motion_tracking_observation_spec_has_no_privileged_cube_fields() -> None:
    from sim.mujoco.tasks.cube_lift.spec import CubeLiftObservationSpec

    spec = CubeLiftObservationSpec(
        layout_version=(
            "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
            "right_target_error6_right_gripper_target_error1_last_action7_progress1_"
            "right_wrist_rgb_chw_3x48x48.v3"
        )
    )
    assert spec.proprio_dim == 29
    assert "cube_to_right_grasp_x_m" not in spec.field_names
    assert spec.wrist_rgb_shape == (3, 48, 48)
    assert spec.dim == 29 + 3 * 48 * 48


def test_normal_config_rejects_debug_grasp_latch() -> None:
    import json

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    payload = json.loads(DEFAULT_PHYSICAL_TASK_CONFIG.read_text())
    payload["grasp_mode"] = "contact_latch_debug"
    with pytest.raises(ValueError, match="cannot use the debug grasp latch"):
        CubeLiftTaskConfig.from_dict(payload)


def test_temp_reward_shaping_preserves_raw_reward_when_disabled() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    next_observations = observations.copy()
    raw_rewards = np.asarray((1.0, -0.5), dtype=np.float32)
    result, terms = shape_rewards(
        observations,
        next_observations,
        raw_rewards,
        RewardShapingConfig(enabled=False),
    )
    np.testing.assert_array_equal(result, raw_rewards)
    assert terms["distance_m"].shape == (2,)


def test_temp_reward_shaping_rewards_reaching_and_near_closure() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = np.zeros((1, 31), dtype=np.float32)
    previous[0, 20] = 0.10
    current[0, 20] = 0.02
    current[0, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert shaped[0] > 0.0
    assert terms["reach_progress_term"][0] > 0.0
    assert terms["gripper_term"][0] > 0.0
    assert terms["grasp_region_term"][0] > 0.0
    assert terms["grasp_gate"][0] > 0.5


def test_temp_dense_reward_penalizes_closing_far_from_cube() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = 0.30
    current[:, 20] = 0.30
    current[1, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["grasp_gate"][1] < 1e-4
    assert terms["premature_close_term"][1] < 0.0
    assert shaped[1] < shaped[0]


def test_temp_dense_reach_reward_has_long_range_ordering() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((3, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = (0.40, 0.20, 0.05)
    current[:, 20] = (0.40, 0.20, 0.05)
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(3, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["reach_term"][0] > 0.05
    assert shaped[0] < shaped[1] < shaped[2]


def test_temp_dense_reach_targets_pregrasp_height_above_cube() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    # cube - grasp z: -0.03 means the grasp frame is 30 mm above the cube.
    previous[:, 22] = (-0.03, 0.0)
    current[:, 22] = (-0.03, 0.0)
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(grasp_height_offset_m=0.03),
    )
    assert terms["vertical_error_m"].tolist() == pytest.approx((0.0, 0.03))
    assert shaped[0] > shaped[1]


def test_temp_dense_reward_prioritizes_grasped_lift() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 9] = 0.025
    current[:, 9] = 0.065
    previous[:, 20] = 0.02
    current[:, 20] = 0.02
    current[0, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.asarray((100.0, 100.0), dtype=np.float32),
        RewardShapingConfig(),
        resting_cube_z_m=np.asarray((0.025, 0.025), dtype=np.float32),
        lift_delta_m=0.08,
    )
    assert terms["lift_fraction"].tolist() == pytest.approx((0.5, 0.5))
    assert terms["lift_term"][0] > 0.0
    assert terms["lift_progress_term"][0] > 0.0
    assert terms["lift_term"][1] == pytest.approx(0.0)
    assert shaped[0] > shaped[1]
    # Enabled dense shaping does not inherit the task's repeatedly accumulated
    # absolute-height raw reward.
    zero_raw, _ = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
        resting_cube_z_m=np.asarray((0.025, 0.025), dtype=np.float32),
        lift_delta_m=0.08,
    )
    np.testing.assert_allclose(shaped, zero_raw)


def test_temp_dense_reward_credits_closing_progress_near_grasp() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 22] = -0.03
    current[0, 22] = -0.027
    current[0, 6] = 0.2
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["gripper_progress_term"][0] > 0.0
    assert shaped[0] > 0.0


def test_temp_dense_reward_does_not_move_reach_target_when_gripper_closes() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    # Both samples are at the same 30 mm pre-grasp pose; only the second closes.
    previous[:, 22] = -0.03
    current[:, 22] = -0.03
    current[1, 6] = 0.8
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["distance_m"][0] == pytest.approx(terms["distance_m"][1])
    assert terms["reach_term"][0] == pytest.approx(terms["reach_term"][1])
    assert terms["gripper_term"][1] > terms["gripper_term"][0]
    assert shaped[1] > shaped[0]


def test_temp_dense_reward_has_explicit_success_bonus() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observation = np.zeros((2, 31), dtype=np.float32)
    shaped, terms = shape_rewards(
        observation,
        observation,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(success_bonus=10.0),
        success=np.asarray((False, True)),
    )
    assert terms["success_term"].tolist() == pytest.approx((0.0, 10.0))
    assert shaped[1] - shaped[0] == pytest.approx(10.0)


def test_staged_reward_opens_close_gate_before_strict_lift_gate() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 20] = 0.12
    current[0, 20] = 0.08
    current[0, 6] = 0.5
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(
            staged_grasp=True,
            grasp_acquire_gate=0.0,
            grasp_acquire_bonus=1.0,
        ),
    )
    assert terms["close_gate"][0] > terms["grasp_gate"][0]
    assert terms["reach_phase_weight"][0] < 1.0
    assert terms["grasp_acquire_term"][0] == pytest.approx(1.0)
    assert terms["gripper_progress_term"][0] > 0.0
    assert shaped[0] > terms["gripper_term"][0]


def test_temp_reward_shaping_penalizes_collision_projection() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(enabled=False, safety_projection_penalty=0.25),
        safety_projection_alpha=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    assert shaped[0] == pytest.approx(0.0)
    assert shaped[1] == pytest.approx(-0.25)
    assert terms["safety_projection_term"].tolist() == pytest.approx((0.0, -0.25))


def test_temp_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        save_checkpoint,
    )
    from sim.mujoco_temp_pipeline.config import PPOConfig, RewardShapingConfig
    from sim.mujoco_temp_pipeline.normalization import RunningMeanStd
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    torch.manual_seed(4)
    policy = ActorCritic(hidden_dim=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    normalizer = RunningMeanStd.create((31,))
    normalizer.update(np.ones((3, 31), dtype=np.float32))
    path = save_checkpoint(
        tmp_path / "policy.pt",
        policy=policy,
        optimizer=optimizer,
        observation_normalizer=normalizer,
        ppo_config=PPOConfig(
            total_timesteps=16,
            num_envs=2,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=16,
            hidden_dim=16,
        ),
        reward_config=RewardShapingConfig(),
        global_step=16,
        update=1,
        task_config_path=TASK_CONFIG,
    )
    restored, restored_normalizer, payload = build_policy_from_checkpoint(path)
    observation = torch.zeros((2, 31))
    with torch.no_grad():
        expected = policy.act(observation, deterministic=True)[0]
        actual = restored.act(observation, deterministic=True)[0]
    torch.testing.assert_close(actual, expected)
    np.testing.assert_allclose(restored_normalizer.mean, normalizer.mean)
    assert payload["global_step"] == 16


def test_task_config_mismatch_requires_explicit_diagnostic_override() -> None:
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
    )

    payload = {"task_config_sha256": "0" * 64}
    with pytest.raises(ValueError, match="legacy collision diagnostics"):
        check_task_config_compatibility(payload, TASK_CONFIG)

    result = check_task_config_compatibility(
        payload,
        TASK_CONFIG,
        allow_mismatch=True,
    )
    assert result["task_config_mismatch"] is True
    assert result["checkpoint_task_config_sha256"] == "0" * 64


def test_legacy_collision_diagnostic_cli_flags_are_explicit() -> None:
    from sim.mujoco_temp_pipeline.evaluate import build_arg_parser as eval_parser
    from sim.mujoco_temp_pipeline.play import build_arg_parser as play_parser

    assert eval_parser().parse_args([]).allow_task_config_mismatch is False
    play_args = play_parser().parse_args(
        ["--allow-task-config-mismatch", "--print-safety-events"]
    )
    assert play_args.allow_task_config_mismatch is True
    assert play_args.print_safety_events is True


def test_temp_policy_actions_and_log_probs_are_finite() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(hidden_dim=16)
    observation = torch.zeros((5, 31))
    action, log_prob, value = policy.act(observation)
    evaluated_log_prob, entropy, evaluated_value = policy.evaluate_actions(
        observation, action
    )
    assert action.shape == (5, 7)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    torch.testing.assert_close(log_prob, evaluated_log_prob)
    torch.testing.assert_close(value, evaluated_value)
    assert torch.isfinite(entropy).all()


def test_temp_rgb_policy_uses_cnn_and_keeps_action_contract() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(
        observation_dim=15 + 3 * 32 * 32,
        action_dim=7,
        hidden_dim=32,
        proprio_dim=15,
        image_shape=(3, 32, 32),
    )
    observation = torch.zeros((4, 15 + 3 * 32 * 32))
    action, log_prob, value = policy.act(observation, deterministic=True)
    assert action.shape == (4, 7)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert policy.image_shape == (3, 32, 32)


def test_temp_rgb_v3_policy_preserves_spatial_features() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    observation_dim = 29 + 3 * 48 * 48
    policy = ActorCritic(
        observation_dim=observation_dim,
        action_dim=7,
        hidden_dim=32,
        proprio_dim=29,
        image_shape=(3, 48, 48),
        image_pool_size=3,
    )
    observation = torch.zeros((2, observation_dim))
    action, log_prob, value = policy.act(observation, deterministic=True)
    assert action.shape == (2, 7)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert policy.image_pool_size == 3
    assert policy.image_encoder[-2].in_features == 64 * 3 * 3


def test_temp_run_recorder_writes_machine_and_human_readable_outputs(
    tmp_path: Path,
) -> None:
    import csv
    import json

    from sim.mujoco_temp_pipeline.records import RunRecorder

    recorder = RunRecorder(
        tmp_path,
        run_name="smoke run",
        config={"ppo": {"learning_rate": 3e-4}, "device": "cpu"},
    )
    metrics = {
        "update": 1,
        "global_step": 16,
        "wall_time_seconds": 0.5,
        "transitions_per_second": 32.0,
        "raw_reward_mean": 0.1,
        "shaped_reward_mean": 0.2,
        "completed_episodes": 1,
        "successful_episodes": 1,
        "success_rate": 1.0,
        "policy_loss": -0.01,
        "value_loss": 0.5,
        "entropy": 2.0,
        "approx_kl": 0.001,
        "clip_fraction": 0.0,
        "policy_std_mean": 0.6,
        "learning_rate": 3e-4,
    }
    recorder.record_update(
        metrics,
        (
            {
                "env_id": 0,
                "length": 10,
                "raw_return": 1.0,
                "shaped_return": 2.0,
                "success": True,
                "termination_reason": "height_only_lift_success",
            },
        ),
    )
    recorder.finish({"global_step": 16, "success_rate": 1.0})

    assert json.loads((recorder.directory / "run_config.json").read_text())["device"] == "cpu"
    assert json.loads(recorder.metrics_jsonl.read_text().splitlines()[0])["global_step"] == 16
    assert json.loads(recorder.episodes_jsonl.read_text().splitlines()[0])["success"] is True
    with recorder.metrics_csv.open(newline="") as handle:
        assert next(csv.DictReader(handle))["policy_loss"] == "-0.01"
    assert json.loads(recorder.summary_json.read_text())["success_rate"] == 1.0
    report = recorder.report_html.read_text(encoding="utf-8")
    assert "Cube-Lift temporary PPO" in report
    assert '"global_step": 16' in report
