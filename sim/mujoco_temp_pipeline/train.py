"""CLI for the temporary batched Cube-Lift PPO smoke training."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .checkpoint import file_sha256, save_checkpoint
from .config import PPOConfig, RewardShapingConfig
from .ppo import PPOSmokeTrainer
from .records import RunRecorder


DEFAULT_TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "latest.pt"
DEFAULT_RECORD_ROOT = Path(__file__).resolve().parent / "records"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a minimal PPO directly on CubeLiftVectorExecutor."
    )
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--start-method", default="spawn", choices=("spawn", "forkserver"))
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--no-reward-shaping",
        action="store_true",
        help="Train on the original sparse Cube-Lift reward only.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        start_method=args.start_method,
        device=args.device,
    )
    reward_config = RewardShapingConfig(enabled=not args.no_reward_shaping)
    trainer = PPOSmokeTrainer(
        ppo_config,
        reward_config,
        task_config_path=str(args.task_config),
        model_cache_dir=(
            None if args.model_cache_dir is None else str(args.model_cache_dir)
        ),
    )
    recorder = RunRecorder(
        args.record_root,
        run_name=args.run_name,
        config={
            "schema_version": "dexteleop.sim.mujoco_temp_ppo_run.v1",
            "ppo": ppo_config.to_dict(),
            "reward_shaping": reward_config.to_dict(),
            "task_config": str(Path(args.task_config).resolve()),
            "task_config_sha256": file_sha256(args.task_config),
            "checkpoint_target": str(Path(args.checkpoint).resolve()),
            "model_cache_dir": (
                None if args.model_cache_dir is None else str(Path(args.model_cache_dir).resolve())
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "resolved_device": str(trainer.device),
        },
    )
    started = time.monotonic()
    next_checkpoint = ppo_config.checkpoint_interval
    completed_episodes = 0
    successful_episodes = 0
    recent_returns: list[float] = []
    try:
        while trainer.global_step < ppo_config.total_timesteps:
            rollout = trainer.collect_rollout()
            losses = trainer.update(rollout)
            completed_episodes += rollout.completed_episodes
            successful_episodes += rollout.successful_episodes
            recent_returns.extend(rollout.episode_raw_returns)
            recent_returns = recent_returns[-100:]
            elapsed = max(time.monotonic() - started, 1e-9)
            record = {
                "update": trainer.update_index,
                "global_step": trainer.global_step,
                "wall_time_seconds": elapsed,
                "transitions_per_second": trainer.global_step / elapsed,
                "raw_reward_mean": rollout.raw_reward_mean,
                "shaped_reward_mean": rollout.shaped_reward_mean,
                "completed_episodes": completed_episodes,
                "successful_episodes": successful_episodes,
                "success_rate": (
                    successful_episodes / completed_episodes if completed_episodes else 0.0
                ),
                "rollout_completed_episodes": rollout.completed_episodes,
                "rollout_successful_episodes": rollout.successful_episodes,
                "recent_episode_raw_return_mean": (
                    float(np.mean(recent_returns)) if recent_returns else None
                ),
                "policy_std_mean": float(
                    torch.exp(torch.clamp(trainer.policy.log_std.detach(), -5.0, 1.0))
                    .mean()
                    .cpu()
                ),
                "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                "safety_projection_rate": rollout.safety_projection_rate,
                "safety_rejection_rate": rollout.safety_rejection_rate,
                "safety_projection_alpha_mean": rollout.safety_projection_alpha_mean,
                "safety_minimum_distance_m": rollout.safety_minimum_distance_m,
                "hard_penetration_count": rollout.hard_penetration_count,
                "clearance_violation_count": rollout.clearance_violation_count,
                **rollout.reward_term_means,
                **losses,
            }
            recorder.record_update(record, rollout.episodes)
            print(json.dumps(record, sort_keys=True), flush=True)
            if trainer.global_step >= next_checkpoint:
                save_checkpoint(
                    args.checkpoint,
                    policy=trainer.policy,
                    optimizer=trainer.optimizer,
                    observation_normalizer=trainer.observation_normalizer,
                    ppo_config=ppo_config,
                    reward_config=reward_config,
                    global_step=trainer.global_step,
                    update=trainer.update_index,
                    task_config_path=args.task_config,
                )
                save_checkpoint(
                    recorder.checkpoint_directory / f"step_{trainer.global_step:09d}.pt",
                    policy=trainer.policy,
                    optimizer=trainer.optimizer,
                    observation_normalizer=trainer.observation_normalizer,
                    ppo_config=ppo_config,
                    reward_config=reward_config,
                    global_step=trainer.global_step,
                    update=trainer.update_index,
                    task_config_path=args.task_config,
                )
                while next_checkpoint <= trainer.global_step:
                    next_checkpoint += ppo_config.checkpoint_interval

        destination = save_checkpoint(
            args.checkpoint,
            policy=trainer.policy,
            optimizer=trainer.optimizer,
            observation_normalizer=trainer.observation_normalizer,
            ppo_config=ppo_config,
            reward_config=reward_config,
            global_step=trainer.global_step,
            update=trainer.update_index,
            task_config_path=args.task_config,
        )
        run_checkpoint = save_checkpoint(
            recorder.checkpoint_directory / f"step_{trainer.global_step:09d}_final.pt",
            policy=trainer.policy,
            optimizer=trainer.optimizer,
            observation_normalizer=trainer.observation_normalizer,
            ppo_config=ppo_config,
            reward_config=reward_config,
            global_step=trainer.global_step,
            update=trainer.update_index,
            task_config_path=args.task_config,
        )
        elapsed = time.monotonic() - started
        summary = {
            "checkpoint": str(destination),
            "run_checkpoint": str(run_checkpoint),
            "record_directory": str(recorder.directory),
            "report_html": str(recorder.report_html),
            "global_step": trainer.global_step,
            "updates": trainer.update_index,
            "elapsed_seconds": elapsed,
            "transitions_per_second": trainer.global_step / max(elapsed, 1e-9),
            "completed_episodes": completed_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / completed_episodes if completed_episodes else 0.0,
            "device": str(trainer.device),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "reward_shaping": reward_config.enabled,
        }
        recorder.finish(summary)
        return summary
    finally:
        trainer.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
