from pathlib import Path

import numpy as np
import pytest


TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")


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
