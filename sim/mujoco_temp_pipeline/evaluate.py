"""Headless deterministic evaluation for a temporary PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import is_cube_lift_success_reason

from .checkpoint import build_policy_from_checkpoint, check_task_config_compatibility
from .policy import resolve_device
from .train import DEFAULT_CHECKPOINT, DEFAULT_TASK_CONFIG


def _parse_action_offsets(value: str) -> tuple[float, ...]:
    try:
        offsets = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "action offsets must be comma-separated finite floats"
        ) from exc
    if not offsets or not all(np.isfinite(offset) for offset in offsets):
        raise argparse.ArgumentTypeError(
            "action offsets must be comma-separated finite floats"
        )
    return offsets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a temporary Cube-Lift PPO checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--start-method", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the policy distribution instead of using its mean.",
    )
    parser.add_argument(
        "--action-offsets",
        type=_parse_action_offsets,
        default=(0.0,),
        help=(
            "Comma-separated compact-action offsets. Non-zero robustness "
            "offsets require a one-dimensional policy and balanced env/episode counts."
        ),
    )
    parser.add_argument(
        "--allow-task-config-mismatch",
        action="store_true",
        help="Diagnostics only: run a legacy checkpoint in the current environment.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive.")
    action_offsets = tuple(float(value) for value in args.action_offsets)
    if args.num_envs % len(action_offsets) != 0:
        raise ValueError("num_envs must be divisible by the number of action offsets.")
    if args.episodes % args.num_envs != 0:
        raise ValueError("episodes must be divisible by num_envs for balanced evaluation.")
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, normalizer, payload = build_policy_from_checkpoint(
        args.checkpoint, device=device
    )
    if policy.action_dim != 1 and any(
        abs(offset) > 0.0 for offset in action_offsets
    ):
        raise ValueError(
            "Non-zero action offsets require a one-dimensional compact policy."
        )
    compatibility = check_task_config_compatibility(
        payload,
        args.task_config,
        allow_mismatch=getattr(args, "allow_task_config_mismatch", False),
    )
    env = CubeLiftVectorExecutor(
        args.num_envs,
        config_path=args.task_config,
        model_cache_dir=args.model_cache_dir,
        auto_reset=True,
        master_seed=args.seed,
        start_method=args.start_method,
    )
    returns = np.zeros(args.num_envs, dtype=np.float64)
    lengths = np.zeros(args.num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    target_episodes_per_env = args.episodes // args.num_envs
    completed_by_env = np.zeros(args.num_envs, dtype=np.int32)
    offset_by_env = np.asarray(
        [
            action_offsets[env_id % len(action_offsets)]
            for env_id in range(args.num_envs)
        ],
        dtype=np.float32,
    )
    offset_metrics = {
        offset: {
            "episodes": 0,
            "successes": 0,
            "returns": [],
            "lengths": [],
        }
        for offset in action_offsets
    }
    successes = 0
    projected_actions = 0
    rejected_actions = 0
    evaluated_actions = 0
    minimum_safety_distance = float("inf")
    hard_penetrations = 0
    clearance_violations = 0
    tracking_error_sum = 0.0
    tracking_error_samples = 0
    tracking_error_max = 0.0
    gripper_tracking_error_sum = 0.0
    gripper_tracking_error_max = 0.0
    try:
        observation, _infos = env.reset()
        max_steps = args.episodes * 600
        for _ in range(max_steps):
            with torch.no_grad():
                action, _log_prob, _value = policy.act(
                    torch.as_tensor(normalizer.normalize(observation), device=device),
                    deterministic=not args.stochastic,
                )
                if policy.action_dim == 1:
                    action[:, 0] = torch.clamp(
                        action[:, 0]
                        + torch.as_tensor(offset_by_env, device=device),
                        -1.0,
                        1.0,
                    )
                environment_action = policy.expand_action(action)
            observation, reward, terminated, truncated, infos = env.step(
                environment_action.cpu().numpy().astype(np.float32, copy=False)
            )
            evaluated_actions += args.num_envs
            for info in infos:
                tracking_error_sum += float(info.get("arm_tracking_error_rad_mean", 0.0))
                tracking_error_max = max(
                    tracking_error_max,
                    float(info.get("arm_tracking_error_rad_max", 0.0)),
                )
                tracking_error_samples += 1
                gripper_error = abs(
                    float(info.get("right_gripper_tracking_error", 0.0))
                )
                gripper_tracking_error_sum += gripper_error
                gripper_tracking_error_max = max(
                    gripper_tracking_error_max, gripper_error
                )
                projected_actions += int(info.get("safety_modified", False))
                rejected_actions += int(
                    float(info.get("safety_projection_alpha", 1.0)) <= 1e-8
                )
                distance = float(info.get("safety_minimum_distance_m", float("inf")))
                if np.isfinite(distance):
                    minimum_safety_distance = min(minimum_safety_distance, distance)
                hard_penetrations += int(
                    info.get("termination_reason") == "robot_environment_penetration"
                )
                clearance_violations += int(
                    info.get("termination_reason")
                    in {
                        "robot_environment_clearance_violation",
                        "sim_collision_margin_violation",
                    }
                )
            returns += reward
            lengths += 1
            done = np.logical_or(terminated, truncated)
            for env_id in np.flatnonzero(done):
                if completed_by_env[env_id] < target_episodes_per_env:
                    episode_return = float(returns[env_id])
                    episode_length = int(lengths[env_id])
                    success = int(
                        is_cube_lift_success_reason(
                            infos[env_id].get("termination_reason")
                        )
                    )
                    completed_returns.append(episode_return)
                    completed_lengths.append(episode_length)
                    successes += success
                    completed_by_env[env_id] += 1
                    offset = action_offsets[env_id % len(action_offsets)]
                    metrics = offset_metrics[offset]
                    metrics["episodes"] += 1
                    metrics["successes"] += success
                    metrics["returns"].append(episode_return)
                    metrics["lengths"].append(episode_length)
                returns[env_id] = 0.0
                lengths[env_id] = 0
            if np.all(completed_by_env >= target_episodes_per_env):
                break
        if len(completed_returns) < args.episodes:
            raise RuntimeError(
                f"Evaluation completed only {len(completed_returns)}/{args.episodes} episodes."
            )
    finally:
        env.close()
    return {
        **compatibility,
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "deterministic": not args.stochastic,
        "policy_action_dim": policy.action_dim,
        "environment_action_dim": policy.environment_action_dim,
        "active_action_indices": list(policy.active_action_indices),
        "action_offset_results": [
            {
                "action_offset": offset,
                "episodes": int(offset_metrics[offset]["episodes"]),
                "successes": int(offset_metrics[offset]["successes"]),
                "success_rate": (
                    float(offset_metrics[offset]["successes"])
                    / max(int(offset_metrics[offset]["episodes"]), 1)
                ),
                "raw_return_mean": float(
                    np.mean(offset_metrics[offset]["returns"])
                ),
                "episode_length_mean": float(
                    np.mean(offset_metrics[offset]["lengths"])
                ),
            }
            for offset in action_offsets
        ],
        "raw_return_mean": float(np.mean(completed_returns)),
        "raw_return_std": float(np.std(completed_returns)),
        "episode_length_mean": float(np.mean(completed_lengths)),
        "projected_action_rate": projected_actions / max(evaluated_actions, 1),
        "fully_rejected_action_rate": rejected_actions / max(evaluated_actions, 1),
        "minimum_robot_environment_distance_m": (
            None if not np.isfinite(minimum_safety_distance) else minimum_safety_distance
        ),
        "hard_penetration_count": hard_penetrations,
        "clearance_violation_count": clearance_violations,
        "arm_tracking_error_rad_mean": (
            tracking_error_sum / max(tracking_error_samples, 1)
        ),
        "arm_tracking_error_rad_max": tracking_error_max,
        "gripper_tracking_error_mean": (
            gripper_tracking_error_sum / max(tracking_error_samples, 1)
        ),
        "gripper_tracking_error_max": gripper_tracking_error_max,
        "device": str(device),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
